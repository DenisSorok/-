import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import pandas as pd
def lo(track, answer):
  audio_data = track
  x , sr = librosa.load(audio_data)
  S = np.abs(librosa.stft(x, n_fft=4096))**2
  chroma = librosa.feature.chroma_stft(S=S, sr=sr)
  su = []
  for i in range(len(chroma)-1,-1,-1):
    for j in range(len(chroma[i])):
        if chroma[i][j]<0.89:
          chroma[i][j]=0
        else:
          chroma[i][j] = 1
    su.append(str(sum(chroma[i])))
  slovar =   {"1":su[0], "2":su[1], "3":su[2], "4":su[3], "5":su[4], "6":su[5], "7":su[6], "8":su[7], "9":su[8], "10":su[9], "11":su[10], "12":su[11], "answer":answer}
  return slovar

def tlo(track):
  audio_data = track
  x, sr = librosa.load(audio_data)
  S = np.abs(librosa.stft(x, n_fft=4096)) ** 2
  chroma = librosa.feature.chroma_stft(S=S, sr=sr)
  su = []
  for i in range(len(chroma) - 1, -1, -1):
    for j in range(len(chroma[i])):
      if chroma[i][j] < 0.89:
        chroma[i][j] = 0
      else:
        chroma[i][j] = 1
    su.append(str(sum(chroma[i])))
  slovar = {"1": su[0], "2": su[1], "3": su[2], "4": su[3], "5": su[4], "6": su[5], "7": su[6], "8": su[7],
            "9": su[8], "10": su[9], "11": su[10], "12": su[11]}
  return slovar
import pandas
trans_df = pandas.read_csv("chroma.csv", header =0)


x = trans_df.drop('answer',axis=1)
y = trans_df['answer']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.01, random_state = 42)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=1000)
#rfc.fit(x_train,y_train)
#prediction=rfc.predict(x_test)
from sklearn.metrics import accuracy_score
#accuracy_score(y_test, prediction)
#my_test = x.loc[150:150]
#my_test.head()
#prediction = rfc.predict(my_test)#(x_test) # делаем предсказания
#print(prediction)
rfc.fit(x_train,y_train)
print(1)


src = r"ffff.mp3"
dst = "test.wav"
import csv

import csv
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
root = Tk()
def browse_file():
     global chord
     chord = filedialog.askopenfilename()
def sok():
    z = tlo(chord)
    FILENAME = "test.csv"
    with open(FILENAME, "w", newline="") as file:
          columns = ["1","2","3","4","5","6","7","8","9","10","11","12"]
          writer = csv.DictWriter(file, fieldnames=columns)
          writer.writeheader()
    with open(FILENAME, "a") as file:
          columns = ["1","2","3","4","5","6","7","8","9","10","11","12"]
          writer = csv.DictWriter(file, fieldnames=columns)
          writer.writerow(z)

    tdf = pandas.read_csv("test.csv", header =0)
    #print(tdf.head())
    my_test = tdf.loc[0:0]
    #print(my_test)
    #my_test.head()
    prediction = rfc.predict(my_test)#(x_test) # делаем предсказания
    #messagebox.showinfo(prediction)
    prediction = str(prediction)
    prediction =  prediction.replace("'", '')
    prediction = prediction.replace("[", '')
    prediction = prediction.replace(']', '')
    lbl.configure(text = "Аккорд: " + prediction)
    lbl.configure(bg = '#1EDB1E')
    lbl.configure(fg='#1C251C')
C = Canvas(root, bg="blue", height=250, width=300)
from random import random, randint
i = randint(0,2)
print(i)
pngs = ['egr.png', 'guit.png', 'hend.png']
image = PhotoImage(file = pngs[i])
background_label = Label(root, image=image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)
C.pack()
lbl1 = Label(root, text = 'Загрузите аудиофайл с аккордом и нажмите кнопку "Распознать"', font = ("Calibri", 30), bg='black', fg = 'white')
lbl1.pack()
btn = Button(text="Загрузить файл", height = 2, width = 10, background="#1EDB1E", foreground="#1C251C",
              padx="20", pady="8", font="16", command=browse_file)
btn.pack()
btn1 = Button(text="Распознать", height = 2, width = 7, background="#1EDB1E", foreground="#1C251C",
             padx="20", pady="8", font="16", command=sok)
btn1.pack()

lbl = Label(root, text="",font=("Calibri", 50),bg="black",fg="white")
lbl.pack()
root.geometry("1280x720")
root.mainloop()