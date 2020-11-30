import librosa
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import librosa.display


df = pd.read_csv('data/train.csv')

def create_spec(filename, name):
    clip, sr = librosa.load('data/filename', sr = None)
    fig, ax = plt.subplots()
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sr)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))

y = df['ebird_code']
