import colorsys
import enum
import glob
import logging
import math
import multiprocessing
import os
import pickle
import queue
import random
import sys
from multiprocessing import Process, Queue, Array, Lock
from operator import itemgetter
from time import time

import PIL.Image
import cv2
import numpy as np
import pygame
import tensorflow as tf
import tensorflow_hub as hub
from PIL import ImageStat, ImageColor
from pygame.locals import *

import label_map_util

logging.basicConfig(level=logging.DEBUG, format='%(process)d :: %(asctime)s :: %(levelname)s :: %(message)s')
a_logger = logging.getLogger()
output_file_handler = logging.FileHandler("output.log")
stdout_handler = logging.StreamHandler(sys.stdout)
a_logger.addHandler(output_file_handler)
a_logger.addHandler(stdout_handler)

# define some parameter depending on whether we want to display it on 4k or 1800 display
FOUR_K_MODE = False

if FOUR_K_MODE:
    TILE_SIZE = 24  # or 30
    SCREEN_SIZE = (3840, 2160)
    LINE_WIDTH = 12
    FONT_SIZE = 40
    NUM_SAMPLES = 14400  # 32036  # total number of "palette" images from which we will choose from
    INPUT_TILES_DIR = "./tiles/4k/"
else:
    TILE_SIZE = 15
    SCREEN_SIZE = (1920, 1080)
    LINE_WIDTH = 6
    FONT_SIZE = 20
    NUM_SAMPLES = 9216  # total number of "palette" images from which we will choose from
    INPUT_TILES_DIR = "./tiles/1800/"

# ENLARGEMENT = 8  # the mosaic image will be this many times wider and taller than the original TODO del
TILE_MATCH_RES = 5  # tile matching resolution (higher values give better fit but require more processing)
IMAGE_SIZE = (800, 450)
TILE_BLOCK_SIZE = TILE_SIZE / max(min(TILE_MATCH_RES, TILE_SIZE), 1)
WORKER_COUNT = max(multiprocessing.cpu_count() - 4, 1)
EOQ_VALUE = None
MAX_REUSE = 1  # max number of times a single image should be used in the mosaic

# used for threshold condition in the swapping algorithm
UNIT_TIME_SWAP_THRESHOLD = 10.0

INPUT_IMAGES_DIR = "./images/"
INPUT_INIT_IMAGES_DIR = "./init_images/"
OUTPUT_DIR = "./output/"

OUTPUT_FRAME_IMAGE_FILE_NAME = OUTPUT_DIR + "mosaic_output" + str(int(time() * 1000)) + ".jpg"
OUTPUT_SCREEN_IMAGE_FILE_NAME = OUTPUT_DIR + "mosaic_screen_output" + str(int(time() * 1000)) + ".jpg"
OUTPUT_VIDEO_FILE_NAME = OUTPUT_DIR + "mosaic_output" + str(int(time() * 1000)) + ".avi"

# frames per second rate for video
FPS = 15
# Save / record every nth frame (1 meaning every frame)
FTS = 1

PIL.Image.MAX_IMAGE_PIXELS = 933120000

# support either downloading the models from the web, or from a local folder "tfhub"
#MODEL_DIR_OPENIMAGES = "tfhub/faster_rcnn_openimages_v4_inception_resnet_v2_1/"
MODEL_DIR_OPENIMAGES = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
#MODEL_DIR_COCO2017 = "tfhub/efficientdet_d7_1/"
MODEL_DIR_COCO2017 = "https://tfhub.dev/tensorflow/efficientdet/d7/1"
# required for mapping coco label ids to class names
COCO2017_LABELS = "tfhub/mscoco_label_map.pbtxt"

# max detection boxes to display (model returns 100)
MAX_BOXES = 25

# permutation encoding
ACB = 1
BAC = 2
BCA = 3
CAB = 4
CBA = 5


# calculate the perceived brightness of a given image
def get_brightness(im):
    stat = ImageStat.Stat(im)
    r, g, b = stat.mean
    return math.sqrt(0.241 * (r ** 2) + 0.691 * (g ** 2) + 0.068 * (b ** 2))


# calculate the hue value of an image
def get_hue(im):
    img2 = im.resize((1, 1), PIL.Image.ANTIALIAS)
    color = img2.getpixel((0, 0))

    try:
        hls = colorsys.rgb_to_hls(color[0], color[1], color[2])
    except ZeroDivisionError as e:
        a_logger.warning(e)
        pass
        hls = colorsys.rgb_to_hls(color[0] + 1, color[1] + 2, color[2] + 3)
    return hls[0]


# 3 ways to init the mosaic:
# Random - random spread of all the tile images
# Hue - organize the tiles by hue
# Image - organize the tiles according to some image (has to be prepared beforehand -
# export the image as a mosaic, to figure out tiles placement
class InitMode(enum.Enum):
    Random = 1
    Image = 2
    Hue = 3


# 4 algorithms for switching tiles in runtime:
# Random - iterate over all tiles at random order, replace tile with best fitting tile for that spot
# Brightness - iterate over all tiles according to brightness (starting with least brightest),
#               replace tile with best fitting tile for that spot.
# BrightnessReverse - iterate over all tiles according to brightness (starting with  brightest),
#               replace tile with best fitting tile for that spot
# Switch - choose two tiles in random, switch between them if improves (using dynamic improvement function)
class BuildMode(enum.Enum):
    Random = 1
    Brightness = 2
    BrightnessReverse = 3
    Switch = 4
    Ordered = 5


class Config:
    def __init__(self, create_video, save_images, init_order, building_order, image_file, init_image, tiles_file,
                 wait_for_keypress, use_coco):
        self.create_video = create_video
        self.save_images = save_images
        self.init_order = init_order
        self.building_order = building_order
        self.init_image = init_image
        self.wait_for_keypress = wait_for_keypress
        self.image_file = image_file
        self.tiles_file = tiles_file
        self.use_coco = use_coco


class ProgressCounter:
    def __init__(self, total, text):
        self.total = total
        self.counter = 0
        self.text = text

    def update(self):
        self.counter += 1
        print("\r{} Progress for {}: {:04.1f}%".format(os.getpid(), self.text,
                                                       100 * self.counter / self.total), end="")


# input: x, y coordinates
# output: returns a large bounding box (for actual tile placement) and a small bounding box (used for diff comparison)
def get_boxes(x, y):
    large_box = (x * TILE_SIZE, y * TILE_SIZE, (x + 1) * TILE_SIZE, (y + 1) * TILE_SIZE)
    small_box = (
        x * TILE_SIZE / TILE_BLOCK_SIZE, y * TILE_SIZE / TILE_BLOCK_SIZE,
        (x + 1) * TILE_SIZE / TILE_BLOCK_SIZE,
        (y + 1) * TILE_SIZE / TILE_BLOCK_SIZE)
    return large_box, small_box


# input: a list of tile images output: a sorted list of these image tiles, sorted bu brightness values (with
# parameter to define ascending / descending) brightness is defined as perceived brightness distance from grey
def get_brightness_data(original_img_small, x_tile_count, y_tile_count):
    by_brightness = []
    size = int(TILE_SIZE / TILE_BLOCK_SIZE)

    progress = ProgressCounter(x_tile_count * y_tile_count, "Sorting image tiles by brightness")
    for x in range(x_tile_count):
        for y in range(y_tile_count):
            large_box, small_box = get_boxes(x, y)

            img = PIL.Image.new('RGB', (size, size))
            img.putdata(original_img_small.crop(small_box).getdata())
            b = get_brightness(img)
            by_brightness.append((b, x * y_tile_count + y))
            progress.update()

    return by_brightness


class TileProcessor:
    def __init__(self, tiles_path):
        self.tiles_path = tiles_path

    @staticmethod
    def __process_tile(tile_path):
        try:
            img = PIL.Image.open(tile_path)
            # tiles must be square, so get the largest square that fits inside the image
            w = img.size[0]
            h = img.size[1]
            min_dimension = min(w, h)
            w_crop = (w - min_dimension) / 2
            h_crop = (h - min_dimension) / 2
            img = img.crop((w_crop, h_crop, w - w_crop, h - h_crop))

            large_tile_img = img.resize((TILE_SIZE, TILE_SIZE), PIL.Image.ANTIALIAS)
            small_tile_img = img.resize((int(TILE_SIZE / TILE_BLOCK_SIZE), int(TILE_SIZE / TILE_BLOCK_SIZE)),
                                        PIL.Image.ANTIALIAS)

            return large_tile_img.convert('RGB'), small_tile_img.convert('RGB')
        except Exception as e:
            a_logger.exception(e)
            return None, None

    def get_tiles(self):
        if os.path.isfile(self.tiles_path):
            tiles_file = open(self.tiles_path, 'rb')
            tiles_data = pickle.load(tiles_file)
            tiles_file.close()
            return tiles_data

        large_tiles = []
        small_tiles = []

        files = []
        ext = ('*.jpg', '*.jpeg')
        for e in ext:
            files.extend(glob.glob(os.path.join(self.tiles_path, e)))

        # Choose a random subset of the overall images
        # this helps to ensure we have a representative sample also
        # when we init by hue or brightness
        if len(files) > NUM_SAMPLES:
            files = random.sample(files, NUM_SAMPLES)

        progress = ProgressCounter(len(files), "Reading tiles")

        for tile_path in files:
            large_tile, small_tile = self.__process_tile(tile_path)

            if large_tile:
                large_tiles.append(large_tile)
                small_tiles.append(small_tile)
            progress.update()

        a_logger.debug('Processed {} tiles.'.format(len(large_tiles)))

        return large_tiles, small_tiles


class TargetImage:
    def __init__(self, image_path):
        self.image_path = image_path

    def get_data(self):
        a_logger.debug('Processing main image...')
        img = PIL.Image.open(self.image_path)
        img = img.resize(SCREEN_SIZE, PIL.Image.ANTIALIAS)

        # use SCREEN_SIZE as the target size for the mosaic
        w = SCREEN_SIZE[0]  # img.size[0] * ENLARGEMENT
        h = SCREEN_SIZE[1]  # img.size[1] * ENLARGEMENT
        large_img = img.resize((w, h), PIL.Image.ANTIALIAS)
        w_diff = (w % TILE_SIZE) / 2
        h_diff = (h % TILE_SIZE) / 2

        # if necessary, crop the image slightly so we use a whole number of tiles horizontally and vertically
        if w_diff or h_diff:
            large_img = large_img.crop((w_diff, h_diff, w - w_diff, h - h_diff))

        small_img = large_img.resize((int(w / TILE_BLOCK_SIZE), int(h / TILE_BLOCK_SIZE)), PIL.Image.ANTIALIAS)

        image_data = (large_img.convert('RGB'), small_img.convert('RGB'))

        a_logger.debug('Main image processed.')

        return image_data


class TileFitter:
    def __init__(self, tiles_data, diff_cache):
        self.tiles_data = tiles_data
        self.last_change = -1
        self.alpha = 0.3
        self.total_positive = 0
        self.total_negatives = 0
        self.start_time = time()
        self.local_diff_cache = diff_cache

    @staticmethod
    def __get_tile_diff(t1, t2, bail_out_value):

        diff = 0
        for i in range(len(t1)):
            diff += ((t1[i][0] - t2[i][0]) ** 2 + (t1[i][1] - t2[i][1]) ** 2 + (t1[i][2] - t2[i][2]) ** 2)
            if diff > bail_out_value:
                # we know already that this isn't going to be the best fit, so no point continuing with this tile
                return diff
        return diff

    def is_switch_good(self, img_data, coords, coords2id, lock):

        img1 = img_data[0]
        img2 = img_data[1]
        img3 = img_data[2]

        id1 = coords2id[coords[0]]
        id2 = coords2id[coords[1]]
        id3 = coords2id[coords[2]]

        tile_data1 = self.tiles_data[id1]
        tile_data2 = self.tiles_data[id2]
        tile_data3 = self.tiles_data[id3]

        # get the diff between the images and the tiles in the current state
        # use the cached value if possible

        # will be used to define the switch permutation between 3 tiles
        abc = -1

        # Switch condition: if at least 1 side improves by more than 1-alpha %, and the other one does
        # not get worse
        # we also keep track of how many successful swaps we did since last time reset
        # we use that as a proxy to headroom (we stop the program, when we go below a minimal threshold

        # BAC
        if img1 is None:
            new_diffB2A = sys.maxsize - 1
            new_diffC2A = sys.maxsize - 1
            curr_diff1 = sys.maxsize
        else:
            curr_diff1 = self.__get_tile_diff(img1, tile_data1,
                                              sys.maxsize)  # self.get_current_diff( id1, img1, lock, tile_data1)
            new_diffB2A = self.__get_tile_diff(img1, tile_data2, curr_diff1)

        curr_diff2 = self.__get_tile_diff(img2, tile_data2, sys.maxsize)
        curr_diff3 = self.__get_tile_diff(img3, tile_data3, sys.maxsize)

        new_diffA2B = self.__get_tile_diff(img2, tile_data1, curr_diff2)

        if (new_diffB2A < self.alpha * curr_diff1 and new_diffA2B < curr_diff2) or (
                new_diffB2A < curr_diff1 and new_diffA2B < self.alpha * curr_diff2):
            self.last_change = time()
            self.total_positive += 1
            return True, new_diffB2A, new_diffA2B, curr_diff3, BAC

        # CBA
        if img1 is not None:
            new_diffC2A = self.__get_tile_diff(img1, tile_data3, curr_diff1)

        new_diffA2C = self.__get_tile_diff(img3, tile_data1, curr_diff3)

        if (new_diffC2A < self.alpha * curr_diff1 and new_diffA2C < curr_diff3) or (
                new_diffC2A < curr_diff1 and new_diffA2C < self.alpha * curr_diff3):
            self.last_change = time()
            self.total_positive += 1
            return True, new_diffB2A, new_diffA2B, curr_diff3, CBA

        # ACB
        new_diffC2B = self.__get_tile_diff(img2, tile_data3, curr_diff2)
        new_diffB2C = self.__get_tile_diff(img3, tile_data2, curr_diff3)

        if (new_diffB2C < self.alpha * curr_diff3 and new_diffC2B < curr_diff2) or (
                new_diffB2C < curr_diff3 and new_diffC2B < self.alpha * curr_diff2):
            self.last_change = time()
            self.total_positive += 1
            return True, new_diffB2A, new_diffA2B, curr_diff3, ACB

        # CAB

        if new_diffC2A < self.alpha * curr_diff1 and new_diffA2B < curr_diff2 and new_diffB2C < curr_diff3 or \
                new_diffC2A < curr_diff1 and new_diffA2B < self.alpha * curr_diff2 and new_diffB2C < curr_diff3 or \
                new_diffC2A < curr_diff1 and new_diffA2B < curr_diff2 and new_diffB2C < self.alpha * curr_diff3:
            self.last_change = time()
            self.total_positive += 1
            abc = CAB

            return True, new_diffC2A, new_diffA2B, new_diffB2C, abc

        # BCA
        if new_diffB2A < self.alpha * curr_diff1 and new_diffC2B < curr_diff2 and new_diffA2C < curr_diff3 or \
                new_diffB2A < curr_diff1 and new_diffC2B < self.alpha * curr_diff2 and new_diffA2C < curr_diff3 or \
                new_diffB2A < curr_diff1 and new_diffC2B < curr_diff2 and new_diffA2C < self.alpha * curr_diff3:
            self.last_change = time()
            self.total_positive += 1
            abc = BCA

            return True, new_diffB2A, new_diffC2B, new_diffA2C, abc

        if time() - self.start_time >= UNIT_TIME_SWAP_THRESHOLD:
            a_logger.debug(
                "Positives: {} Negatives: {} Time lapse: {}".format(self.total_positive, self.total_negatives,
                                                                    time() - self.start_time))

            # heuristic for swapping threshold:
            if self.total_positive == 0 or (
                    self.total_positive > 0 and self.total_negatives / self.total_positive > 1000):  # todo
                self.alpha += 0.1
                a_logger.debug("updating alpha to {}".format(self.alpha))
                if self.alpha >= 1.0:
                    a_logger.debug("No more improvements!")
                    return EOQ_VALUE, EOQ_VALUE, EOQ_VALUE, EOQ_VALUE, EOQ_VALUE

            self.total_positive = 0
            self.total_negatives = 0
            self.start_time = time()

        self.total_negatives += 1
        return False, curr_diff1, curr_diff2, curr_diff3, abc

    def should_switch(self, id1, id2, lock, new_diff1, new_diff2, curr_diff1, curr_diff2):  # diff_cache

        if (new_diff1 < self.alpha * curr_diff1 and new_diff2 < curr_diff2) or (
                new_diff1 < curr_diff1 and new_diff2 < self.alpha * curr_diff2):
            self.last_change = time()
            self.total_positive += 1
            return True
        return False

    def get_best_fit_tile(self, img_data, remaining_tiles, lock):
        best_fit_tile_index = -1
        min_diff = sys.maxsize
        candidates = []

        # go through each tile in turn looking for the best match for the part of the image represented by 'img_data'
        for key in remaining_tiles.keys():
            tile_data = self.tiles_data[key]
            diff = self.__get_tile_diff(img_data, tile_data, min_diff)
            if diff < min_diff:
                min_diff = diff
                candidates.append(key)

        lock.acquire()
        for best_fit_tile_index in reversed(candidates):
            times_used = remaining_tiles.get(best_fit_tile_index)
            if times_used is not None:
                remaining_tiles.pop(best_fit_tile_index)
                break
        lock.release()

        return best_fit_tile_index

# switch between 3 tiles to get a better arrangement, switching 2 tiles can lead to local maxima
def switch_tiles(x_tile_count, y_tile_count, result_queue, coords2id, terminating_event,
                 diff_cache, lock, cfg, mosaic_ready_event):
    _, original_img_small = TargetImage(cfg.image_file).get_data()
    _, tiles = TileProcessor(cfg.tiles_file).get_tiles()

    tiles_data = [list(tile.getdata()) for tile in tiles]

    # this function gets run by the worker processes, one on each CPU core
    image_cache = {}

    for x in range(x_tile_count):
        for y in range(y_tile_count):
            large_box, small_box = get_boxes(x, y)
            image_cache[x + y * x_tile_count] = list(original_img_small.crop(small_box).getdata())

    if not mosaic_ready_event.is_set():
        a_logger.debug("Waiting for Mosaic process")
        mosaic_ready_event.wait()

    tile_fitter = TileFitter(tiles_data, diff_cache)

    num_tiles = len(tiles)
    num_tiles_in_image = x_tile_count * y_tile_count

    while not terminating_event.is_set():
        try:

            xy1 = int(num_tiles * random.random())
            x1 = int(xy1 / y_tile_count)
            y1 = xy1 % y_tile_count

            xy2 = int(num_tiles_in_image * random.random())
            x2 = int(xy2 / y_tile_count)
            y2 = xy2 % y_tile_count

            xy3 = int(num_tiles_in_image * random.random())
            x3 = int(xy3 / y_tile_count)
            y3 = xy3 % y_tile_count

            img_coords = (xy1, x2 + y2 * x_tile_count, x3 + y3 * x_tile_count)

            if xy1 > x_tile_count * y_tile_count - 1:
                imgdata1 = None
            else:
                imgdata1 = image_cache.get(img_coords[0])

            img_data = (imgdata1, image_cache.get(img_coords[1]), image_cache.get(img_coords[2]))

            is_switch_good, diff1, diff2, diff3, abc = tile_fitter.is_switch_good(img_data, img_coords, coords2id, lock)

            if is_switch_good == EOQ_VALUE:
                a_logger.debug("Ending switch_tiles")
                break

            if is_switch_good:

                lock.acquire()
                id1 = coords2id[img_coords[0]]
                id2 = coords2id[img_coords[1]]
                id3 = coords2id[img_coords[2]]

                #abc defines the switch permutation
                if abc == BAC:
                    coords2id[img_coords[0]] = id2
                    coords2id[img_coords[1]] = id1

                elif abc == BCA:
                    coords2id[img_coords[0]] = id2
                    coords2id[img_coords[1]] = id3
                    coords2id[img_coords[2]] = id1

                elif abc == CBA:
                    coords2id[img_coords[0]] = id3
                    coords2id[img_coords[2]] = id1

                elif abc == CAB:
                    coords2id[img_coords[0]] = id3
                    coords2id[img_coords[1]] = id1
                    coords2id[img_coords[2]] = id2

                elif abc == ACB:
                    coords2id[img_coords[1]] = id3
                    coords2id[img_coords[2]] = id2

                lock.release()

                result_queue.put((img_coords, (id1, id2, id3), abc), block=False, timeout=0.1)

        except queue.Full:
            a_logger.warning("Switch tiles: Queue is full !")
            continue
        except KeyboardInterrupt:
            pass
            break

    # let the result handler know that this worker has finished everything
    result_queue.put((EOQ_VALUE, EOQ_VALUE, EOQ_VALUE))


def fit_tiles(sorted_coords, y_tile_count, result_queue, remaining_tiles, diff_cache, lock, cfg, mosaic_ready_event):
    _, original_img_small = TargetImage(cfg.image_file).get_data()
    _, tiles = TileProcessor(cfg.tiles_file).get_tiles()

    tiles_data = [list(tile.getdata()) for tile in tiles]

    # this function gets run by the worker processes, one on each CPU core
    tile_fitter = TileFitter(tiles_data, diff_cache)

    if not mosaic_ready_event.is_set():
        a_logger.debug("Waiting for Mosaic process")
        mosaic_ready_event.wait()

    progress = ProgressCounter(len(sorted_coords), "Fitting tiles to mosaic")

    for coord in sorted_coords:
        try:
            y = coord % y_tile_count
            x = int(coord / y_tile_count)

            large_box, small_box = get_boxes(x, y)
            img_data = list(original_img_small.crop(small_box).getdata())
            img_coords = large_box

            tile_index = tile_fitter.get_best_fit_tile(img_data, remaining_tiles, lock)
            result_queue.put((img_coords, tile_index))
            progress.update()
        except queue.Full:
            a_logger.warning("Queue full while putting - shouldn't happen!")
            continue
        except KeyboardInterrupt:
            pass
            break
    # let the result handler know that this worker has finished everything
    result_queue.put((EOQ_VALUE, EOQ_VALUE))


class MosaicImage:
    def __init__(self, size, tiles_file, init_mode, init_image, use_coco, coords2id, request_q):

        tiles, _ = TileProcessor(tiles_file).get_tiles()
        pygame.init()
        self.coords2id = coords2id
        self.window_name = "Rosy AI"
        self.frame_counter = 0
        self.pseudo_frame_counter = 0
        self.last_frame_time = 0
        self.requestQ = request_q
        self.fpsClock = pygame.time.Clock()
        self.font = pygame.font.SysFont('robotoregular', FONT_SIZE)
        self.colors = list(ImageColor.colormap.values())
        self.colors.remove('#000000')
        self.screen = pygame.display.set_mode(size, pygame.FULLSCREEN)
        pygame.display.set_caption(self.window_name)
        self.blits = []

        self.x_tile_count = int(size[0] / TILE_SIZE)
        self.y_tile_count = int(size[1] / TILE_SIZE)
        self.total_tiles_in_image = self.x_tile_count * self.y_tile_count
        surface_tiles = [pygame.surfarray.make_surface(np.array(tile).reshape((TILE_SIZE, TILE_SIZE, 3))).convert() for
                         tile in
                         tiles]
        self.category_index = label_map_util.create_category_index_from_labelmap(COCO2017_LABELS,
                                                                                 use_display_name=True)  # TODO COCO
        self.coco_model = use_coco  # TODO COCO
        a_logger.debug("Screen size: {}, {}".format(pygame.display.Info().current_w, pygame.display.Info().current_h))

        self.ratio = 1  # self.screen.get_width() / size[0] #TODO

        self.new_ts = min(int(self.ratio * TILE_SIZE), TILE_SIZE)
        if self.ratio != 1:
            self.tiles = [pygame.transform.smoothscale(t, (self.new_ts, self.new_ts)) for t in surface_tiles]
        else:
            self.tiles = surface_tiles
        self.img = pygame.Surface((self.new_ts * self.x_tile_count, self.new_ts * self.y_tile_count))
        self.rects = pygame.Surface((self.new_ts * self.x_tile_count, self.new_ts * self.y_tile_count))
        self.rects.set_colorkey((0, 0, 0))
        self.output = pygame.Surface((self.new_ts * self.x_tile_count, self.new_ts * self.y_tile_count))

        self.ratio = self.new_ts / TILE_SIZE

        if init_mode == InitMode.Random:
            self.init_randomly(self.tiles)
        elif init_mode == InitMode.Hue:
            self.init_by_hue(tiles)
        elif init_mode == InitMode.Image and init_image is not None:
            init_img = pygame.image.load(init_image).convert()
            if int(self.img.get_width() / self.img.get_height()) != int(init_img.get_width() / init_img.get_height()):
                raise AttributeError("Init image size doesn't match that of mosaic image")
            try:
                x = pickle.load(open(init_image + ".coords2id.pkl", "rb"))
                for i in range(len(x)):
                    self.coords2id[i] = x[i]
                self.coords2id = pickle.load(open(init_image + ".coords2id.pkl", "rb"))
            except Exception as e:
                a_logger.exception("Error reading coords2id pkl file: {}".format(e))
            init_img = pygame.transform.scale(init_img, (self.img.get_width(), self.img.get_height()))
            self.img.blit(init_img, (0, 0))
        self.screen.blit(self.img, (0, 0))

        pygame.display.update()

    def init_randomly(self, tiles):

        progress = ProgressCounter(self.total_tiles_in_image, "initializing mosaic image randomly")
        # will be used to get random tiles without replacement, unless tiles in image >> number of tile file

        rand_ids = list(range(len(tiles)))
        random.shuffle(rand_ids)

        blit_list = []

        for i in range(len(tiles)):
            if i < self.total_tiles_in_image:
                y = i % self.y_tile_count
                x = int(i / self.y_tile_count)

                coords = (x * self.new_ts, y * self.new_ts, (x + 1) * self.new_ts, (y + 1) * self.new_ts)

                random_index = rand_ids[i]
                tile = tiles[random_index]
                blit_list.append((tile, coords))
                self.coords2id[y * self.x_tile_count + x] = random_index
            else:
                random_index = rand_ids[i]
                self.coords2id[i] = random_index

            progress.update()

        self.img.blits(blit_list)

    def init_by_hue(self, tiles):

        progress = ProgressCounter(self.total_tiles_in_image, "initializing mosaic image by hue")

        # get a representative sample of dataset, so we don't take the first n elements from a sorted list
        sample = tiles
        if len(tiles) > self.x_tile_count * self.y_tile_count:
            sample = random.sample(tiles, self.x_tile_count * self.y_tile_count)

        hue = self.sort_by_hue(sample)

        blit_list = []
        for x in range(self.x_tile_count):
            for y in range(self.y_tile_count):
                coords = (x * self.new_ts, y * self.new_ts, (x + 1) * self.new_ts, (y + 1) * self.new_ts)

                tile_idx = hue[(x * self.y_tile_count + y) % len(hue)][1]

                tile = pygame.surfarray.make_surface(np.array(sample[tile_idx]).reshape((TILE_SIZE, TILE_SIZE, 3)))
                tile = pygame.transform.scale(tile, (self.new_ts, self.new_ts))

                blit_list.append((tile, coords))
                self.coords2id[y * self.x_tile_count + x] = tile_idx
                progress.update()

        self.img.blits(blit_list)

    # input: a list of tile images
    # output: a sorted list of these image tiles, sorted bu hue values
    @staticmethod
    def sort_by_hue(tiles):
        hue = []

        for i, im in enumerate(tiles):
            h = get_hue(im)
            hue.append((h, i))

        hue.sort(key=itemgetter(0))
        _, coords = zip(*hue)
        return hue

    def add_tile(self, coords, index, video_camera, switch, results, abc):
        self.pseudo_frame_counter += 1
        if switch:
            c1 = coords[0]
            id1 = index[0]
            c2 = coords[1]
            id2 = index[1]
            c3 = coords[2]
            id3 = index[2]

            if abc == BAC:
                self.switch_and_add(c2, id1)
                self.switch_and_add(c1, id2)
            elif abc == BCA:
                self.switch_and_add(c1, id2)
                self.switch_and_add(c2, id3)
                self.switch_and_add(c3, id1)

            elif abc == CBA:
                self.switch_and_add(c3, id1)
                self.switch_and_add(c1, id3)

            elif abc == CAB:
                self.switch_and_add(c1, id3)
                self.switch_and_add(c2, id1)
                self.switch_and_add(c3, id2)

            elif abc == ACB:
                self.switch_and_add(c3, id2)
                self.switch_and_add(c2, id3)
        else:
            tile_data = self.tiles[index]
            x_c = int(coords[0] * self.ratio)
            y_c = int(coords[1] * self.ratio)
            self.blits.append((tile_data, (x_c, y_c)))
            #self.img.blit(tile_data, (x_c, y_c))

        if self.pseudo_frame_counter % 8 == 0 or results is not None or (
                int(time() * 1000) - self.last_frame_time >= 100):
            self.frame_counter += 1
            self.last_frame_time = int(time() * 1000)
            self.img.blits(self.blits)
            self.blits.clear()
            self.show_frame(results, video_camera)

    def switch_and_add(self, c_i, id_j):
        if c_i < self.total_tiles_in_image:
            tile_data1 = self.tiles[id_j]

            x = c_i % self.x_tile_count
            y = int(c_i / self.x_tile_count)
            coords = (x * self.new_ts, y * self.new_ts, (x + 1) * self.new_ts, (y + 1) * self.new_ts)

            self.blits.append((tile_data1, coords))
            # self.img.blit(tile_data1, coords)

    def show_frame(self, result, vidcam=None):

        frame_annotate_rate = 400

        if self.frame_counter % frame_annotate_rate == 1:
            # convert to array for tf model. array3d swaps axes, so need to swap back
            resized = pygame.surfarray.array3d(self.img)
            resized = resized.swapaxes(0, 1)

            try:
                self.requestQ.put((resized, self.coco_model), block=False)
            except queue.Full:
                a_logger.warning("Ml queue full")
                # self.frame_counter-=40
                pass
        if result is not None:
            if self.coco_model:
                self.draw_boxes(
                    result["detection_boxes"][0],
                    result["detection_classes"][0], result["detection_scores"][0])
            else:  # TODO coco
                self.draw_boxes(
                    result["detection_boxes"],
                    result["detection_class_entities"], result["detection_scores"])

        self.output.blit(self.img, (0, 0))
        self.output.blit(self.rects, (0, 0))
        self.screen.blit(self.output, (0, 0))
        pygame.display.flip()
        self.fpsClock.tick(15)

        if vidcam is not None:
            frame = pygame.surfarray.array3d(self.output)
            frame = frame.swapaxes(0, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            vidcam.write(frame)

    @staticmethod
    def save(surface, path):
        pygame.image.save(surface, path)

    def draw_boxes(self, boxes, class_names, scores, max_boxes=MAX_BOXES, min_score=0.01):
        # Overlay labeled boxes on an image with formatted scores and label names.
        self.rects.fill((0, 0, 0))

        im_width, im_height = self.rects.get_size()

        max_results = min(boxes.shape[0], max_boxes)
        for i in range(max_results - 1, -1, -1):
            if scores[i] >= min_score:
                ymin, xmin, ymax, xmax = tuple(boxes[i])
                if self.coco_model:
                    class_name = self.category_index[class_names[i]]['name']
                    display_str = "{} ({}%)".format(class_name, int(100 * scores[i]))
                else:  # TODO coco
                    display_str = "{} ({}%)".format(class_names[i].decode("ascii"), int(100 * scores[i]))

                color = self.colors[hash(class_names[i]) % len(self.colors)]
                font_color = self.get_contrast_color(color)

                (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                              ymin * im_height, ymax * im_height)

                text = self.font.render(display_str, True, font_color)
                text_rect = text.get_rect()
                text_width, text_height = text_rect.width, text_rect.height

                if top - text_height > 0:
                    text_bottom = top
                else:
                    text_bottom = top + text_height

                if left + text_width > im_width:
                    text_left = right - text_width
                else:
                    text_left = left

                text_rect.bottomleft = (text_left, text_bottom)

                pygame.draw.lines(self.rects, color, True, [(left, top), (left, bottom), (right, bottom), (right, top)],
                                  width=LINE_WIDTH)
                pygame.draw.rect(self.rects, color, text_rect)
                self.rects.blit(text, text_rect)

    @staticmethod
    def get_contrast_color(color):
        try:
            i_color = ImageColor.getrgb(color)
            luma = ((0.2126 * i_color[0]) + (0.7152 * i_color[1]) + (0.0722 * i_color[2])) / 255
            wl = 0.05 / (luma + 0.05)
            bl = (luma + 0.05) / 1.05
            if bl > wl:
                font_color = (16, 16, 16)  # black
            else:
                font_color = (240, 240, 240)
        except AttributeError:
            a_logger.error("Problem with color: {}".format(color))
            font_color = (16, 16, 16)  # black
            pass
        return font_color


def model_runner(request_q, answers_q, terminating_event, ready_event, path):
    a_logger.debug("Loading model")

    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)

    module_handle = MODEL_DIR_COCO2017
    coco_detector = hub.load(module_handle)
    converted_img = tf.expand_dims(tf.image.convert_image_dtype(img, tf.uint8), axis=0)
    # to "warm up" the model, so consecutive calls are fast
    coco_detector(converted_img)

    module_handle = MODEL_DIR_OPENIMAGES
    openimage_detector = hub.load(module_handle).signatures['default']
    converted_img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    openimage_detector(converted_img)

    a_logger.debug("Model loaded")
    ready_event.set()

    while not terminating_event.is_set():
        try:
            img, use_coco = request_q.get(block=True, timeout=0.05)

            if use_coco:
                converted_img = tf.expand_dims(tf.image.convert_image_dtype(img, tf.uint8), axis=0)  # TODO COCO
                detector = coco_detector
            else:
                converted_img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
                detector = openimage_detector

            start_time = time()
            result = detector(converted_img)
            end_time = time()

            result = {key: value.numpy() for key, value in result.items()}

            a_logger.debug("Found %d objects." % len(result["detection_scores"]))
            a_logger.debug("Inference time: {} ".format(end_time - start_time))

            answers_q.put(result)

        except queue.Empty:
            continue
        except KeyboardInterrupt:
            pass
            break


def build_mosaic(result_queue, size, coords2id, cfg, terminating_event, request_q, answer_q,
                 mosaic_ready_event, ml_ready_event):
    create_video = cfg.create_video
    switch = cfg.building_order == BuildMode.Switch

    active_workers = WORKER_COUNT
    video_camera = None

    a_logger.debug("Creating mosaic image")
    mosaic = MosaicImage(size, cfg.tiles_file, cfg.init_order, cfg.init_image, cfg.use_coco, coords2id, request_q)

    if not ml_ready_event.is_set():
        a_logger.debug("Waiting for ML model")
        show_text(mosaic.screen, "Loading Machine Learning model...")
        for _ in pygame.event.get():
            pass
        ml_ready_event.wait()

    # signal that working processes can start
    a_logger.debug("Mosaic is ready!")
    mosaic_ready_event.set()

    if create_video:
        size = (mosaic.output.get_size())  # TODO
        video_camera = cv2.VideoWriter(OUTPUT_VIDEO_FILE_NAME, cv2.VideoWriter_fourcc(*'DIVX'), FPS, size)
    counter = 0

    # Sanity check to empty queue from last run
    try:
        answer_q.get(block=False)
    except queue.Empty:
        pass

    abc = -1
    while True:
        try:
            for e in pygame.event.get():
                if e.type == pygame.QUIT or e.type == KEYDOWN and e.key == K_ESCAPE:
                    raise KeyboardInterrupt

            if switch:
                img_coords, index, abc = result_queue.get(block=True, timeout=0.01)
            else:
                img_coords, index = result_queue.get(block=True, timeout=0.01)
            if counter % 75 == 0:  # TODO
                try:
                    results = answer_q.get(block=False)
                except queue.Empty:
                    results = None
                    pass
            else:
                results = None

            if img_coords == EOQ_VALUE:
                active_workers -= 1
                a_logger.debug("{} remaining working process out of {}".format(active_workers, WORKER_COUNT))
                if not active_workers:
                    # analyze the last frame
                    mosaic.show_frame(results, video_camera)
                    break
            else:
                mosaic.add_tile(img_coords, index, video_camera, switch, results, abc)
            counter += 1
        except queue.Empty:
            continue
        except KeyboardInterrupt:
            pass
            break

    terminating_event.set()

    if video_camera is not None:
        video_camera.release()
    mosaic.save(mosaic.img, OUTPUT_FRAME_IMAGE_FILE_NAME)
    mosaic.save(mosaic.output, OUTPUT_SCREEN_IMAGE_FILE_NAME)
    a_logger.debug('Finished, output is in {}, {}'.format(OUTPUT_FRAME_IMAGE_FILE_NAME, OUTPUT_SCREEN_IMAGE_FILE_NAME))

    to_save = [coords2id[i] for i in range(len(coords2id))]
    pickle.dump(to_save, open(OUTPUT_FRAME_IMAGE_FILE_NAME + ".coords2id.pkl", "wb"))

    if cfg.wait_for_keypress:
        while True:
            event = pygame.event.wait()
            if event.type == QUIT or event.type == KEYDOWN:
                break

    pygame.quit()


# Display message center screen
def show_text(screen, text):
    font = pygame.font.SysFont('robotoregular', 20)
    text = font.render(text, True, (16, 16, 16))
    text_rect = text.get_rect(topleft=(25, 25))
    pygame.draw.rect(screen, (50, 255, 50), text_rect)
    screen.blit(text, text_rect)
    pygame.display.update()


def compose(conf, request_q, answer_q, ml_ready_event):
    a_logger.debug('Building mosaic, press Ctrl-C to abort...')

    original_img_large, original_img_small = TargetImage(conf.image_file).get_data()
    all_tile_data_large, all_tile_data_small = TileProcessor(conf.tiles_file).get_tiles()

    x_tile_count = int(original_img_large.size[0] / TILE_SIZE)
    y_tile_count = int(original_img_large.size[1] / TILE_SIZE)
    total_tiles = x_tile_count * y_tile_count

    global MAX_REUSE
    MAX_REUSE = max(1, int(total_tiles / len(all_tile_data_large)))
    a_logger.debug("MAX_REUSE: {}, mosaic.total_tiles: {},  all_tile_data_large: {}".format(MAX_REUSE, total_tiles,
                                                                                            len(all_tile_data_large)))

    if MAX_REUSE > 1:
        logging.warning(
            "mosaic total tiles ({}) is bigger than total number of tiles ({}). Some tiles will be reused more than "
            "once".format(
                MAX_REUSE, total_tiles,
                len(all_tile_data_large)))

    result_queue = Queue()
    coords2id = Array('i', len(all_tile_data_large))  # todo was total_tiles
    diff_cache = Array('i', len(all_tile_data_large))
    terminating_event = multiprocessing.Event()
    mosaic_ready_event = multiprocessing.Event()
    lock = Lock()
    manager = multiprocessing.Manager()
    remaining_tiles = manager.dict({idx: 0 for idx in range(len(all_tile_data_large))})

    try:

        a_logger.debug("Starting build mosaic process")

        # start the worker processes that will build the mosaic image
        p = Process(target=build_mosaic,
                    args=(
                        result_queue, original_img_large.size, coords2id, conf, terminating_event,
                        request_q, answer_q, mosaic_ready_event, ml_ready_event))
        p.start()
        processes = []

        a_logger.debug("Adding tiles to image")
        if conf.building_order == BuildMode.Switch:
            for n in range(WORKER_COUNT):
                a_logger.debug("Starting worker queue {}".format(n))
                w = Process(target=switch_tiles,
                            args=(x_tile_count, y_tile_count, result_queue,
                                  coords2id, terminating_event, diff_cache, lock, conf, mosaic_ready_event))
                w.start()
                processes.append(w)
        else:
            if conf.building_order == BuildMode.Random:
                sorted_list = list(range(x_tile_count * y_tile_count))
                random.shuffle(sorted_list)
            elif conf.building_order == BuildMode.Ordered:
                flavor = random.randint(1, 4)
                if flavor == 1:
                    sorted_list = [y + x * y_tile_count for y in range(y_tile_count) for x in range(x_tile_count)]
                elif flavor == 2:
                    sorted_list = [y + x * y_tile_count for y in range(y_tile_count) for x in range(x_tile_count)]
                    sorted_list.reverse()
                elif flavor == 3:
                    sorted_list = list(range(x_tile_count * y_tile_count))
                elif flavor == 4:
                    sorted_list = list(range(x_tile_count * y_tile_count - 1, -1, -1))
            else:
                sorted_list = get_brightness_data(original_img_small, x_tile_count, y_tile_count)
                grey = math.sqrt(0.241 * (128 ** 2) + 0.691 * (128 ** 2) + 0.068 * (128 ** 2))
                random.shuffle(sorted_list)

            chunk_size = int(len(sorted_list) / (WORKER_COUNT)) + 1

            # create n working processes, each responsible for a chunk of the tiles
            for n in range(0, len(sorted_list), chunk_size):
                a_logger.debug(
                    "Starting worker queue for range {} - {}".format(n, min(n + chunk_size, len(sorted_list))))
                sub_list = sorted_list[n: min(n + chunk_size, len(sorted_list))]

                if conf.building_order == BuildMode.BrightnessReverse or conf.building_order == BuildMode.Brightness:
                    sub_list.sort(key=lambda l: (l[0] - grey) ** 2,
                                  reverse=(conf.building_order == BuildMode.BrightnessReverse))
                    _, sub_list = zip(*sub_list)

                w = Process(target=fit_tiles,
                            args=(
                                sub_list, y_tile_count, result_queue, remaining_tiles, diff_cache, lock, conf,
                                mosaic_ready_event))
                w.start()
                processes.append(w)
    except Exception as e:
        pass
        a_logger.exception(e)
        terminating_event.set()

    a_logger.debug("Done setting up working processes, waiting for them to complete")
    for w in processes:
        w.join()
    p.join()
    a_logger.debug("All processes done!")


def random_run(request_q, answer_q, ready_event):
    try:
        build_modes = len(BuildMode)
        bm = random.randint(1, build_modes)

        init_mode = len(InitMode) - 1  # minus 1 since we ended up disliking the Hue sorting TODO
        im = random.randint(1, init_mode)

        src_images = glob.glob(INPUT_IMAGES_DIR + "*.jpg")
        img = random.choice(src_images)

        tiles_files = glob.glob(INPUT_TILES_DIR + "*.pkl")
        tiles_file = random.choice(tiles_files)

        init_images = glob.glob(INPUT_INIT_IMAGES_DIR + "*.jpg")
        init_img = random.choice(init_images)

        #use_coco = bool(random.getrandbits(1))
        use_coco = False #todo - I am not very fund of COCO results

        a_logger.debug("BuildMode: {}, InitMode: {}, src image: {}".format(BuildMode(bm).name, InitMode(im).name, img))

        cfg = Config(create_video=True, save_images=False, init_order=InitMode(im), building_order=BuildMode(bm),
                     image_file=img, init_image=init_img, tiles_file=tiles_file, wait_for_keypress=False,
                     use_coco=use_coco)

        compose(cfg, request_q, answer_q, ready_event)
    except Exception as msg:
        a_logger.exception(msg)
        pass


def main():
    request_q = Queue(5)
    answer_q = Queue()
    terminating_event = multiprocessing.Event()
    ready_event = multiprocessing.Event()

    src_images = glob.glob(INPUT_IMAGES_DIR + "*.jpg")
    path = random.choice(src_images)

    p = Process(target=model_runner, args=(request_q, answer_q, terminating_event, ready_event, path))
    p.start()

    while True:
        a_logger.debug("Rolling the dice!")
        random_run(request_q, answer_q, ready_event)

    terminating_event.set()
    p.join()


if __name__ == '__main__':
    main()
