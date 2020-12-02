import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

df = pd.read_csv('data/train.csv')

# columns to use: pitch, speed, latitude, longitude, elevation, volume

X = df[['pitch', 'speed', 'latitude', 'longitude', 'elevation', 'volume']]
y = df.pop('species')

conv_dict = {
    'latitude': float,
    'longitude': float, 
    'elevation': int
}

X['pitch'] = X['pitch'].astype('category')
X['speed'] = X['speed'].astype('category')

