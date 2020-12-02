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


train = tf.keras.preprocessing.image_dataset_from_directory('train_imgs', labels = 'inferred', validation_split = .2, subset = 'training',
                                                            image_size=(128,128), batch_size=32, seed = 42)
test = tf.keras.preprocessing.image_dataset_from_directory('train_imgs', labels = 'inferred', validation_split = .2, subset = 'validation',
                                                            image_size=(128,128), batch_size=32, seed = 42,shuffle=False)


AUTOTUNE = tf.data.experimental.AUTOTUNE

train = train.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
test = test.cache().prefetch(buffer_size=AUTOTUNE)

# class_names = train.class_names
# normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)
# normalized_ds = train.map(lambda x, y: (normalization_layer(x), y))


# model = Sequential()
# layers.experimental.preprocessing.Rescaling(1./256., input_shape=(256, 256, 3))
# model.add(Conv2D(32, (2, 2), padding='same', input_shape=(256,256,3)))
# model.add(Activation('relu'))
# model.add(Conv2D(64, (2, 2)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Conv2D(64, (2, 2), padding='same'))
# model.add(Activation('relu'))
# model.add(Conv2D(64, (2, 2)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# # model.add(Dropout(0.5))
# model.add(Conv2D(128, (2, 2), padding='same'))
# model.add(Activation('relu'))
# model.add(Conv2D(128, (2, 2)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# # model.add(Dropout(0.5))
# model.add(Flatten())
# model.add(Dense(512))
# model.add(Activation('relu'))
# # model.add(Dropout(0.5))
# model.add(Dense(264, activation='softmax'))
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
model = Sequential()
layers.experimental.preprocessing.Rescaling(1./255., input_shape=(128, 128, 3))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(4, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(264, activation='softmax'))
#Compile
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# model = Sequential()
# layers.experimental.preprocessing.Rescaling(1./256., input_shape=(256, 256, 3))
# model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(256, 256, 3)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(256, 256, 3)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(264, activation='softmax'))
# model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
print(model.summary())
# epochs=10000
# checkpoint_cb = keras.callbacks.ModelCheckpoint('fifth_model.h5', save_best_only= True)
# early_stopping_cb = keras.callbacks.EarlyStopping(patience=10)
# tensorboard_cb = keras.callbacks.TensorBoard()

# history1 = model.fit(
#   train,
#   validation_data=test,
#   epochs=150, 
#   batch_size= 32,
#   callbacks=[checkpoint_cb, tensorboard_cb]
# )
# history2 = model.fit(
#   train,
#   validation_data=test,
#   epochs=epochs, 
#   batch_size= 32,
#   callbacks=[checkpoint_cb, early_stopping_cb, tensorboard_cb]
#   validation_steps=4274//32,
#   steps_per_epoch=17099//32
)
