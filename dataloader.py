import os 
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
# import keras
from tensorflow.keras.utils import load_img
from tensorflow.keras.preprocessing.image import smart_resize
from tensorflow.keras.layers import *
from tensorflow.keras import layers

from parameters import * 


def create_generators(data_path=DATASET_PATH):
    'Returns three generators'
    image_paths = []
    for folder in os.listdir(data_path):
        paths_to_add = [os.path.join(folder, path) for path in os.listdir(os.path.join(data_path, folder)) if path.endswith('jpg')]
        image_paths = image_paths + paths_to_add

    train_list, val_list, test_list = data_split(np.asarray(image_paths))

    train_data_generator = DataGeneratorClassifier(train_list, TRAINING_BATCH_SIZE, TRAINING_IMAGE_SIZE)
    validation_data_generator = DataGeneratorClassifier(val_list, VALIDATION_BATCH_SIZE, VALIDATION_IMAGE_SIZE)
    test_data_generator = DataGeneratorClassifier(test_list, TESTING_BATCH_SIZE, TESTING_IMAGE_SIZE)
    return train_data_generator, validation_data_generator, test_data_generator


def data_split(paths_list):
    'Splits the paths list into three splits'
    np.random.shuffle(paths_list)
    return paths_list[:len(paths_list)-100], paths_list[len(paths_list)-100:len(paths_list)-50], paths_list[len(paths_list)-50:]


class DataGeneratorClassifier(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size, image_size, data_path=DATASET_PATH, n_channels=NUMBER_OF_CHANNELS, shuffle=SHUFFLE_DATA):
        'Initialisation'
        self.classes = os.listdir(data_path)
        self.image_size = image_size
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
        self.data_path=data_path

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = self.list_IDs[indexes]
        X, y = self.__data_generation(list_IDs_temp)
        
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *image_size, n_channels)
        X = np.empty((self.batch_size, *self.image_size, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        for i, ID in enumerate(list_IDs_temp):
            Xi = load_img(os.path.join(self.data_path, ID))
            Xi = smart_resize(np.asarray(Xi), self.image_size)
      
            # because as we alreay load image into the memory, so we are using flow() function, to apply transformation
            X[i,:] = Xi
            y[i] = self.classes.index(ID.split('\\')[0])
        tf.get_logger().setLevel('ERROR')
        data_augm=tf.keras.Sequential([layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),layers.experimental.preprocessing.RandomRotation(0.0),layers.experimental.preprocessing.RandomContrast(0.01)])
        X=data_augm(X)
        return X,tf.keras.utils.to_categorical(y,num_classes=6)


def show_batch(generator, batch_number=0):
    images, labels = generator.__getitem__(batch_number)
    width = int(np.floor(np.sqrt(labels.shape[0])))
    height = int(np.ceil(labels.shape[0]/float(width)))
    total_height = int(0.09*height*images.shape[1])
    total_width = int(0.09*width*images.shape[2])
    f, axarr = plt.subplots(height,width, figsize=(total_height,total_width))
    for image in range(images.shape[0]):
        image_to_show = (images[image])/np.max(images[image])
        axarr[image//width,image%width].imshow(image_to_show)
        axarr[image//width,image%width].set_title(generator.classes[np.argmax(labels[image])])
    f.tight_layout()
    plt.show()


if __name__=="__main__":
    train_generator = create_generators(DATASET_PATH)[0]
    show_batch(train_generator, batch_number=0)