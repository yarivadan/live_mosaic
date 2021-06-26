# img_viewer.py
import pickle

import PySimpleGUI as sg
import os.path
from mosaic_art import TileProcessor

# First the window layout in 2 columns

def create_perfect_pallete():
    input=open("tiles/mosaic_output1606317843514.jpg.coords2id.pkl", 'rb') #large
    k25_tiles = pickle.load(input)
    input.close()
    input = open("tiles/mosaic_output1606320182077.jpg.coords2id.pkl", 'rb') #small
    k11_tiles = pickle.load(input)
    input.close()

    input=open("tiles/30ktiles.pkl", 'rb')
    tiles = pickle.load(input)
    input.close()
    large_tiles=tiles[0]
    small_tiles =tiles[1]

    filtered_25_large=[large_tiles[i] for i in k25_tiles]
    filtered_25_small = [small_tiles[i] for i in k25_tiles]
    filtered25 = (filtered_25_large,filtered_25_small)
    output=open("tiles/25ktiles.pkl", 'wb')
    pickle.dump(filtered25, output)
    output.close()

    filtered_11_large=[large_tiles[i] for i in k11_tiles]
    filtered_11_small = [small_tiles[i] for i in k11_tiles]
    filtered11 = (filtered_11_large,filtered_11_small)
    output=open("tiles/11ktiles.pkl", 'wb')
    pickle.dump(filtered11, output)
    output.close()

    a,b, =TileProcessor("tiles/25ktiles.pkl").get_tiles()
    a,b, =TileProcessor("tiles/11ktiles.pkl").get_tiles()

file_list_column = [
    [
        sg.Text("Pleae select the folder with the desired images:")
    ],
    [
        #sg.Text("Pleae select the folder with the desired images:"),
        sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse(),
    ],
    [
        sg.Listbox(
            values=[], enable_events=True, size=(40, 20), key="-FILE LIST-"
        )
    ],
    [
        sg.Button("Create Tiles", enable_events=True, key="-IMPORT-BUTTON-")
    ]
]

# ----- Full layout -----
layout = [
    [
        sg.Column(file_list_column),
        sg.VSeperator(),
    ]
]

window = sg.Window("Tiles Creator", layout)

# Run the Event Loop
while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    # Folder name was filled in, make a list of files in the folder
    if event == "-FOLDER-":
        folder = values["-FOLDER-"]
        try:
            # Get list of files in folder
            file_list = os.listdir(folder)
        except:
            file_list = []

        fnames = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder, f))
            and f.lower().endswith((".png", ".gif", ".jpg", ".jpeg"))
        ]
        window["-FILE LIST-"].update(fnames)
    elif event == "-FILE LIST-":  # A file was chosen from the listbox
        try:
            filename = os.path.join(
                values["-FOLDER-"], values["-FILE LIST-"][0]
            )
            window["-TOUT-"].update(filename)
            window["-IMAGE-"].update(filename=filename)
        except:
            pass
    elif event == "-IMPORT-BUTTON-":  # A file was chosen from the listbox
        try:
            sg.popup_animated("busy.gif")
            f = values["-FOLDER-"]
            t1, t2 = TileProcessor(values["-FOLDER-"]).get_tiles()
            tiles=(t1,t2)
            output=open("tiles.pkl",'wb')
            pickle.dump(tiles,output)
            output.close()
            sg.popup_animated(image_source=None)
            sg.popup_ok("tiles.pkl" + " created")
        except Exception as e:
            sg.popup_error("An error occured. Can't create tiles.")


window.close()