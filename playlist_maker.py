import tkinter as tk
import pandas as pd
import numpy as np
from scipy.spatial import distance
from tabulate import tabulate


def Recommendation(h, e, l ):

    coords = np.array([[h, e]])
    distances = distance.cdist(latent_space_quantiled, coords, 'euclidean')
    arg_distances = np.argsort(distances, axis=0)
    top20 = np.random.choice(arg_distances[:l*10].reshape(-1), l)

    return top20


def openNewWindow(h, e, l):
    global top20

    top20 = Recommendation(h,e,l)

    # Toplevel object which will
    # be treated as a new window
    newWindow = tk.Toplevel(root)

    # sets the title of the
    # Toplevel widget
    newWindow.title("New Playlist")

    # sets the geometry of toplevel
    newWindow.geometry("900x{}".format(75+l*12))

    # A Label widget to show in toplevel
    tk.Label(newWindow,
          text="Happiness: {}%, Energy: {}%\n".format(int(h*100), int(e*100)), font = "fixedsys 12 bold").pack()

    tk.Label(newWindow, text=tabulate(data.iloc[top20,:][["name", "artists"]], headers=["name", "artists"], tablefmt='psql', showindex = False)).pack()
          # text=data.iloc[top20,:][["name", "artists"]].to_string(index = False)).pack()

# READ LATENT SPACE FILE
latent_space_quantiled = np.load("./data/latent_space_quantiled.npy")

# IMPORT DATA

data = pd.read_csv("./data/spotify_data.csv")
data = data.sort_values("name")

# GUI

happiness = 0
energy = 0

root = tk.Tk()
root.title("Sergio's playlist generator")

root.geometry("700x450")
title = tk.Label(root, text = "WELCOME TO SERGIO'S PLAYLIST GENERATOR", font = "fixedsys 20 bold")
title.pack()

instructions = tk.Label(root, text = "\nAdjust the values and press the button below to generate a new playlist.\n", font = "fixedsys 10")
instructions.pack()

happy = tk.Label(root, text = "\nHAPPINESS", font = "fixedsys 10 bold")
happy.pack()
happiness_scale = tk.Scale(root, from_=0, to = 100, orient = "horizontal")
happiness_scale.pack()

energia = tk.Label(root, text = "\nENERGY", font = "fixedsys 10 bold")
energia.pack()
energy_scale = tk.Scale(root, from_=0, to = 100, orient = "horizontal")
energy_scale.pack()

length = tk.Label(root, text = "\nNumber of songs", font = "fixedsys 10 bold")
length.pack()
length_scale = tk.Scale(root, from_=0, to = 50, orient = "horizontal")
length_scale.pack()

empty = tk.Label(root, text = "\n", font = "fixedsys 10 bold")
empty.pack()

btn2 = tk.Button(root,text="GENERATE PLAYLIST",command=lambda:openNewWindow(happiness_scale.get()/100,energy_scale.get()/100, length_scale.get()))
btn2.pack()

root.mainloop()