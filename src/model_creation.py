import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, RMSprop

# Load and Preproccess data
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255., width_shift_range=0.3)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'img_dataset/train',
        color_mode='rgb',
        batch_size=32,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        'img_dataset/test',
        color_mode='rgb',
        batch_size=32,
        class_mode='categorical')

""" Creating the Model """

def print_model_properties(model, indices = 0):
        """Prints the model

        Args:
            model (tensorflow model): The model to print
            indices (int, optional): How many layers deep to print. Defaults to 0.
        """
        for i, layer in enumerate(model.layers[indices:]):
                print(f"Layer {i+indices} | Name: {layer.name} | Trainable: {layer.trainable}")

def change_trainable_layers(model, trainable_index):
        """Change the trainable layers in the transfer learning model

        Args:
            model (TF model): Transfer learning model
            trainable_index (int): Which layer to unfreeze
        """
        for layer in model.layers[:trainable_index]:
                layer.trainable = False
        for layer in model.layers[trainable_index:]:
                layer.trainable = True


def create_transfer_model(input_size, n_categories, weights = 'imagenet'):
        """Uploads the transfer model and adds layers to the bottom of the model

        Args:
            input_size (Tuple): Size of the image
            n_categories (int): Number of categories to let the NN predict on
            weights (str, optional): Pre trained weights to use. Defaults to 'imagenet'.

        Returns:
            Tensorflow model: returns the tensorflow model(uncompiled)
        """
        # note that the "top" is not included in the weights below
        base_model = InceptionResNetV2(weights=weights,
                          include_top=False,
                          input_shape=input_size)
        model = base_model.output
        model = Dropout(.4)(model)
        model = GlobalAveragePooling2D()(model)
        predictions = Dense(n_categories, activation='softmax')(model)
        model = Model(inputs=base_model.input, outputs=predictions)
        
        return model

model = create_transfer_model((3,255,255),264)
_ = change_trainable_layers(model, 774)
print_model_properties(model, 770)

learning_rate = 0.00005
momentum = .9

model.compile(optimizer=RMSprop(lr=learning_rate, momentum=momentum), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
checkpoint_cb = keras.callbacks.ModelCheckpoint('transfer_learn_all_classes.h5', save_best_only= True)
epochs = 200
history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs, 
        batch_size= 10,
        callbacks=[checkpoint_cb]
        )

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Plot accuracies and losses
epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('full_data_200.png')