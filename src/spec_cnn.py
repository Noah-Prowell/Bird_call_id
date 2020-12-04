import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import regularizers, optimizers
import pandas as pd
import numpy as np
import PIL
import matplotlib.pyplot as plt 
from tensorflow.keras.metrics import CategoricalAccuracy
import os
from PIL import Image 
from PIL import ImageFilter 
import os, fileinput, sys

def make_more_imgs():
    for entry in os.scandir('/home/ubuntu/Capstone_2/train_imgs/five_test/aldfly_t'): 
        # if entry.path.endswith('.png'):
        img = Image.open(entry.path)
        fil_img = img.filter(ImageFilter.CONTOUR)
        (name, extension) = os.path.splitext(entry.path)
        # Save with "_blur" added to the filename
        fil_img.save(name + '_filter_min' + extension)
    for entry in os.scandir('/home/ubuntu/Capstone_2/train_imgs/five_test/amegfi'): 
        # if entry.path.endswith('.png'):
        img = Image.open(entry.path)
        fil_img = img.filter(ImageFilter.CONTOUR)
        (name, extension) = os.path.splitext(entry.path)

        # Save with "_blur" added to the filename
        fil_img.save(name + '_filter_min' + extension)
    for entry in os.scandir('/home/ubuntu/Capstone_2/train_imgs/five_test/amepip'): 
        # if entry.path.endswith('.png'):
        img = Image.open(entry.path)
        fil_img = img.filter(ImageFilter.CONTOUR)
        (name, extension) = os.path.splitext(entry.path)
        # Save with "_blur" added to the filename
        fil_img.save(name + '_filter_min' + extension)
    for entry in os.scandir('/home/ubuntu/Capstone_2/train_imgs/five_test/astfly'): 
        # if entry.path.endswith('.png'):
        img = Image.open(entry.path)
        fil_img = img.filter(ImageFilter.CONTOUR)
        (name, extension) = os.path.splitext(entry.path)
        # Save with "_blur" added to the filename
        fil_img.save(name + '_filter_min' + extension)
    for entry in os.scandir('/home/ubuntu/Capstone_2/train_imgs/five_test/balori'): 
        # if entry.path.endswith('.png'):
        img = Image.open(entry.path)
        fil_img = img.filter(ImageFilter.CONTOUR)
        (name, extension) = os.path.splitext(entry.path)
        # Save with "_blur" added to the filename
        fil_img.save(name + '_filter_min' + extension)
###
# make_more_imgs()
###
train = tf.keras.preprocessing.image_dataset_from_directory('train_imgs/five_test', labels = 'inferred',color_mode= 'grayscale', validation_split = .2, subset = 'training',
                                                            image_size=(128,128), batch_size=32, seed = 42)
test = tf.keras.preprocessing.image_dataset_from_directory('train_imgs/five_test', labels = 'inferred',color_mode= 'grayscale', validation_split = .2, subset = 'validation',
                                                            image_size=(128,128), batch_size=6000, seed = 42)

# test = tf.keras.preprocessing.image_dataset_from_directory('train_imgs/five_test/amegfi', labels = 'inferred',color_mode= 'grayscale',image_size=(128,128), batch_size=32, seed = 42)
# AUTOTUNE = tf.data.experimental.AUTOTUNE

# train = train.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
# test = test.cache().prefetch(buffer_size=AUTOTUNE)



# model = Sequential()
# layers.experimental.preprocessing.Rescaling(1./255., input_shape=(128, 128, 1))
# model.add(Conv2D(32, kernel_size=(3, 3),padding = 'same', activation='relu', input_shape=(128, 128, 1)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))
# model.add(Conv2D(64, kernel_size=(3, 3),padding = 'same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))
# model.add(Conv2D(64, kernel_size=(3, 3),padding = 'same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))
# model.add(Conv2D(128, kernel_size=(3, 3),padding = 'same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))
# model.add(Flatten())
# model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(5, activation='softmax'))
# model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=CategoricalAccuracy())
# print(model.summary())
# epochs=10000
checkpoint_cb = keras.callbacks.ModelCheckpoint('twentyfour_model.h5', save_best_only= True)
# early_stopping_cb = keras.callbacks.EarlyStopping(patience=10)
# # tensorboard_cb = keras.callbacks.TensorBoard()

# history1 = model.fit(
#   train,
#   validation_data=test,
#   epochs=1200, 
#   batch_size= 10,
#   callbacks=[checkpoint_cb]
# )
# history2 = model.fit(
#   train,
#   validation_data=test,
#   epochs=epochs, 
#   batch_size= 10,
#   callbacks=[checkpoint_cb, early_stopping_cb]
# )

# acc = history1.history['categorical_accuracy']
# val_acc = history1.history['val_categorical_accuracy']

# loss = history1.history['loss']
# val_loss = history1.history['val_loss']

# epochs_range = range(1200)

# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.savefig('twentyfour.png')

model = keras.models.load_model('twentythree_model.h5')
history3 = model.fit(
  train,
  validation_data=test,
  epochs=200, 
  batch_size= 10,
  callbacks=[checkpoint_cb]
)
# prediction = model.predict(test)
# predicted_index = np.argmax(prediction, axis=1)