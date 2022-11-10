# General data parameters
DATASET_PATH = '../dataset'
SHUFFLE_DATA = True

# Data generator parameters
TRAINING_BATCH_SIZE = 84 
TRAINING_IMAGE_SIZE = (128, 128)  
VALIDATION_BATCH_SIZE = 50
VALIDATION_IMAGE_SIZE = (128, 128)
TESTING_BATCH_SIZE = 16
TESTING_IMAGE_SIZE = (128, 128)
NUMBER_OF_CHANNELS = 3
TRAIN_SIZE = 0.6
VALIDATION_SIZE = 0.2

# Model parameters
KERNEL_SIZE = (2, 2)
DROPOUT_RATE = 0.2

# Training parameters
NOMBRE_EPOCHS = 65

# Save model
CHECKPOINT_PATH = '../checkpoints/cp.ckpt'