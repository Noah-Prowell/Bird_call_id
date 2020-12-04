import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from PIL import Image 
from PIL import ImageFilter 
import os, fileinput, sys
from skimage.filters import gaussian
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix
# df = pd.read_csv('data/train.csv')

# # columns to use: pitch, speed, latitude, longitude, elevation, volume

# X = df[['pitch', 'latitude', 'longitude', 'elevation', 'volume']]
# y = df.pop('species')

# X['elevation'] = [num.replace(' m', '') for num in X['elevation']]
# X['elevation'] = [num.replace('?', '') for num in X['elevation']]
# X['elevation'] = [num.replace('Unknown', '') for num in X['elevation']]
# X['elevation'] = [num.replace(',', '') for num in X['elevation']]
# X['elevation'] = [num.replace('~', '') for num in X['elevation']]
# conv = lambda i : i or None
# X['elevation'] = [conv(i) for i in X['elevation']] 
# X['pitch'] = X['pitch'].astype('category')
# X['volume'] = X['volume'].astype('category')

# X.dropna(inplace = True)
# # X['latitude'] = ast.literal_eval(X['latitude'])
# # X['longitude'] = X['longitude'].astype(float)
# X['elevation'] = X['elevation'].astype(float)



# pd.get_dummies(X, columns = ['pitch', 'volume'], drop_first= True)
test = tf.keras.preprocessing.image_dataset_from_directory('train_imgs/five_test', labels = 'inferred',color_mode= 'grayscale', validation_split = .2, subset = 'validation',
                                                            image_size=(128,128), batch_size=6000, seed = 42)

model = keras.models.load_model('twentythree_model.h5')
prediction = model.predict(test)
predicted_index = np.argmax(prediction, axis=1)