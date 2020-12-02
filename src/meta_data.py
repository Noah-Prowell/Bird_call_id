import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

df = pd.read_csv('data/train.csv')

# columns to use: pitch, speed, latitude, longitude, elevation, volume

X = df[['pitch', 'latitude', 'longitude', 'elevation', 'volume']]
y = df.pop('species')

X['elevation'] = [num.replace(' m', '') for num in X['elevation']]
X['elevation'] = [num.replace('?', '') for num in X['elevation']]
X['elevation'] = [num.replace('Unknown', '') for num in X['elevation']]
X['elevation'] = [num.replace(',', '') for num in X['elevation']]
X['elevation'] = [num.replace('~', '') for num in X['elevation']]
conv = lambda i : i or None
X['elevation'] = [conv(i) for i in X['elevation']] 
X['pitch'] = X['pitch'].astype('category')
X['volume'] = X['volume'].astype('category')

X.dropna(inplace = True)
# X['latitude'] = ast.literal_eval(X['latitude'])
# X['longitude'] = X['longitude'].astype(float)
X['elevation'] = X['elevation'].astype(float)



pd.get_dummies(X, columns = ['pitch', 'volume'], drop_first= True)


layers.experimental.preprocessing.Rescaling(1./256., input_shape=(256, 256, 3))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(264, activation='softmax'))