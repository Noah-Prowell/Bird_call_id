import librosa
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import librosa.display

# Read in dataset
df = pd.read_csv('data/train.csv')


def create_spec(filename, folder, name):
    """Crate spectrograms for all audio files

    Args:
        filename (str): filename to lopad into librosa
        folder (str): foler to save to
        name (str): name to save file to
    """
    clip, sr = librosa.load(filename, sr = None)
    fig, ax = plt.subplots()
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sr)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    plt.savefig(f'train_imgs/{folder}/{name}.png', format = 'png')
    plt.close()    
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename,name,clip,sr,fig,ax,S
y = df['species']


# Use dataset names to read in all audio files and save the subsequent spectrogram
for folder, fil in zip(df['ebird_code'][12329:], df['filename'][12329:]):
    if os.path.isdir(f'/home/noahprowell/galvanize/capstones/Capstone_2/train_imgs/{folder}') == False:
        os.mkdir(f'train_imgs/{folder}')
    else:
        pass
    create_spec(f'data/train_audio/{folder}/{fil}',folder, f'{folder}_{fil}')