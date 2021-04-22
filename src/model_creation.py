import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, RMSprop

def print_model_properties(model, indices = 0):
     for i, layer in enumerate(model.layers[indices:]):
        print(f"Layer {i+indices} | Name: {layer.name} | Trainable: {layer.trainable}")


def change_trainable_layers(model, trainable_index):
    for layer in model.layers[:trainable_index]:
        layer.trainable = False
    for layer in model.layers[trainable_index:]:
        layer.trainable = True

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255., width_shift_range=0.2)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'five_cat_train_img/train',
        color_mode='rgb',
        batch_size=32,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        'five_cat_train_img/validation',
        color_mode='rgb',
        batch_size=32,
        class_mode='categorical')


def create_transfer_model(input_size, n_categories, weights = 'imagenet'):
        # note that the "top" is not included in the weights below
        base_model = InceptionResNetV2(weights=weights,
                          include_top=False,
                          input_shape=input_size)
        model = base_model.output
        model = GlobalAveragePooling2D()(model)
        predictions = Dense(n_categories, activation='softmax')(model)
        model = Model(inputs=base_model.input, outputs=predictions)
        
        return model

model = create_transfer_model((255,255,3),5)
_ = change_trainable_layers(model, 780)
print_model_properties(model, 778)
model.compile(optimizer=RMSprop(lr=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint_cb = keras.callbacks.ModelCheckpoint('transfer_learn.h5', save_best_only= True)

history = model.fit(
  train_generator,
  validation_data=validation_generator,
  epochs=200, 
  batch_size= 10,
  callbacks=[checkpoint_cb]
)

acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']