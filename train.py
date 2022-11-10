import matplotlib.pyplot as plt
import tensorflow as tf

from model import create_model
from dataloader import create_generators
from parameters import * 

def train():
  training_generator, validation_generator, _ = create_generators(data_path=DATASET_PATH)

  model = create_model()

  # Create a callback that saves the model's weights
  cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH, save_weights_only=True, verbose=1)

  history = model.fit(training_generator, validation_data=validation_generator, epochs=NOMBRE_EPOCHS, callbacks=[cp_callback])

  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()
  # summarize history for loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc = 'upper left')
  plt.show()

if __name__ == "__main__":
  train()