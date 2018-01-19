# TODO: 1. Clicking on Run gives the scrolling command -- Done
# 2. Show results very easy to understand: -- 
# 2.1 Eating minutes
# 2.2 Number FG
# 2.3 KCal Est.



try: # python 3
    import tkinter
    from queue import Queue, Empty
except ImportError: # python 2
    import Tkinter as tkinter
    from Queue import Queue, Empty

from PIL import ImageTk, Image
from sandals import *
import sandals
import time
import logging
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'MOTIF_module','IS_code'))
from run_motif import *
from textwrap import dedent
import subprocess
from threading import Thread



logger = logging.getLogger(__name__)


file_sel_flag = 0
input_file = ""


def run_ml(input_file):
    output_file = run_motif('text.txt', input_file)

    # Real time display test
    # for i in range(20):
    #     with open("text.txt", "a") as myfile:
    #         myfile.write('Hello World'+str(i)+'\n')
    #     time.sleep(0.1)


# This function will be run every N milliseconds
def get_text(root,val,name):
    # try to open the file and set the value of val to its contents 
    try:
        with open(name,"r") as f:
            val.set(f.read())
    except IOError as e:
        print e
    else:
        # schedule the function to be run again after 1000 milliseconds  
        root.after(100,lambda:get_text(root,val,name))



def image(filename):
    im = Image.open(filename)
    photo = ImageTk.PhotoImage(im)
    label = tkinter.Label(sandals._root, image=photo)
    label.image = photo
    label.pack(side=sandals._pack_side)

    return label
    


open('text.txt', 'w').close()

with window_sized("Feeding Gesture Prediction", 1200, 800):

    logger.info("Deleting folder")
    label("MOTIF method", font = "Verdana 12 bold")
    image("MOTIF_resize.png")

    with stack(padx=2, pady=2, borderwidth=1, relief=tkinter.SUNKEN):

        with stack(borderwidth=1, relief=tkinter.SUNKEN):
            file_label = label("Select data file", font = "Verdana 12")

        with flow():
            @button("Open file")
            def pick_file():
                with askOpenFile() as file:
                    global file_sel_flag
                    global input_file

                    file_sel_flag = 1
                    file_label.text = file.name
                    input_file = file_label.text

        runStatusLabel = label("")

        @button("Predict eating episode", font = "Veranda 12")
        def change_that_text():
            if file_sel_flag:
                global input_file
                # runStatusLabel.text = ""
                # runStatusLabel.text = "Running..."
                thread = Thread(target = run_ml, args=(input_file,))
                thread.start()
                # runStatusLabel.text = "Done"
            else:
                runStatusLabel.text = "Please select input file."

        with flow():
            @button("Save result")
            def pick_file():
                with askSaveAsFile() as file1:
                    time.sleep(1)
                    # while(1)
                    # exit()

    eins = tkinter.StringVar()
    data1 = textWithVar(sandals._root, textvariable=eins)
    data1.config(font=('times', 24))
    data1.pack()
    get_text(sandals._root,eins, "text.txt")


