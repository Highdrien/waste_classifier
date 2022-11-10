import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

from parameters import * 

def create_model():
    model = keras.Sequential([
        layers.Conv2D(32, KERNEL_SIZE, activation='gelu', padding='same', strides=1), 
        layers.Conv2D(32, KERNEL_SIZE, activation='gelu', padding='same', strides=1),
        layers.MaxPool2D(pool_size=2),
        layers.Conv2D(64, KERNEL_SIZE, activation='gelu', padding='same', strides=1),
        layers.Conv2D(64, KERNEL_SIZE, activation='gelu', padding='same', strides=1),
        layers.MaxPool2D(pool_size=2),
        layers.Conv2D(128, KERNEL_SIZE, activation='gelu', padding='same', strides=1), 
        layers.Conv2D(128, KERNEL_SIZE, activation='gelu', padding='same', strides=1),
        layers.MaxPool2D(pool_size=2),
        layers.Conv2D(256, KERNEL_SIZE, activation='gelu', padding='same', strides=1), 
        layers.Conv2D(256, KERNEL_SIZE, activation='gelu', padding='same', strides=1),
        layers.MaxPool2D(pool_size=2),
        # layers.Conv2D(512, KERNEL_SIZE, activation='relu', padding='same', strides=1), 
        # layers.Conv2D(512, KERNEL_SIZE, activation='relu', padding='same', strides=1),
        # layers.MaxPool2D(pool_size=2),
        layers.Flatten(),
        layers.Dense(100, activation='gelu'),
        layers.Dropout(rate=DROPOUT_RATE),
        layers.Dense(6, activation='softmax')
    ])
    model.build(input_shape=(TRAINING_BATCH_SIZE, *TRAINING_IMAGE_SIZE, NUMBER_OF_CHANNELS))
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    model.summary()
    return model