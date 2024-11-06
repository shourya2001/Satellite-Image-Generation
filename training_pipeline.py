# importing necessary libraries

import tensorflow as tf
from keras.layers import Lambda, Reshape, Permute, Input, GaussianNoise, concatenate, Conv2D
from keras.layers import ConvLSTM2D, BatchNormalization, TimeDistributed, Add, Dropout, MaxPooling2D, UpSampling2D
from keras.models import Model
import wandb
from keras.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt

import random
import subprocess
import os
from PIL import Image
from matplotlib.pyplot import imshow, figure
from keras.models import Sequential
from keras.callbacks import Callback
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, UpSampling2D
from keras import backend as K
from keras.preprocessing import image
import glob

from sklearn.model_selection import train_test_split


# Defining the hyperparameters
HEG = 256 # Height of the image
WID = 444 # Width of the image
FRAME = 5 # Number of frames

# initializing the wandb
wandb.login(key="gogetyourgoddammkey-mai-nahi-bataunga")
hyperparams = {"num_epochs": 25,
          "batch_size": 4,
          "height": HEG,
          "width": WID}

wandb.init(config=hyperparams)
config = wandb.config

# Defining the function to load the data
def load_data_sequence(img_folder, img_width, img_height, sequence_length=5):
    '''
    This function will load the data from the folder and will return the images and the next frame

    img_folder: str: Path to the folder containing the images
    img_width: int: Width of the images
    img_height: int: Height of the images
    sequence_length: int: Number of frames per sequence

    '''
    # This will hold all sequences, each with sequence_length - 1 input images,
    # and 1 target image, all stacked along the last dimension
    video_data_list = []
    next_frame_data_list = []

    img_paths = sorted(glob.glob(img_folder + '/*.png'))
    num_sequences = len(img_paths) - sequence_length + 1

    # Loop over the dataset to create sequences
    for seq_start in range(num_sequences):
        seq_end = seq_start + sequence_length
        sequence_imgs = [
            image.img_to_array(
                image.load_img(img_paths[i], target_size=(img_width, img_height), color_mode='rgb')
            ) for i in range(seq_start, seq_end)
        ]

        # Normalize images to [-1, 1]
        sequence_imgs = [(img - 127.5) / 127.5 for img in sequence_imgs]
        # sequence_imgs = [img/255 for img in sequence_imgs]

        # Stack the sequence along the channel axis to form a single array
        video_data_list.append(np.concatenate(sequence_imgs[:-1], axis=-1))
        next_frame_data_list.append(sequence_imgs[-1])

    return np.array(video_data_list), np.array(next_frame_data_list)

# Usage
dataset_path = '/content/drive/MyDrive/Deep-Learning/Geo/Data/USC'
img_width = HEG
img_height = WID
sequence_length = FRAME  # Adjust based on how many frames you want per sequence
videos, next_frames = load_data_sequence(dataset_path, img_width, img_height, sequence_length)

# Split data into training and validation sets
train_X, validation_X, train_y, validation_y = train_test_split(videos, next_frames, test_size=0.2, random_state=42, shuffle=False)

class ImageCallback(Callback):
    def __init__(self, validation_data, num_samples=15):
        """
        Initializes the callback with validation data.

        Args:
        - validation_data: A tuple (validation_X, validation_y) of numpy arrays.
        - num_samples: Number of samples to log images for. Defaults to 15.
        """
        super().__init__()
        self.validation_data = validation_data
        self.num_samples = num_samples

    def on_epoch_end(self, epoch, logs=None):
        # Sample a batch of data
        validation_X, validation_y = self.validation_data
        idxs = np.random.choice(range(len(validation_X)), size=self.num_samples, replace=False)
        validation_X_sample = validation_X[idxs]
        validation_y_sample = validation_y[idxs]

        # Predict
        output = self.model.predict(validation_X_sample)

        # Log images to wandb
        wandb.log({
            "input": [wandb.Image(np.concatenate(np.split(c, 4, axis=2), axis=1)) for c in validation_X_sample],
            "output": [wandb.Image(np.concatenate([validation_y_sample[i], o], axis=1)) for i, o in enumerate(output)]
        }, commit=False)

# Function for measuring how similar two images are
def perceptual_distance(y_true, y_pred):
    '''
    This function will calculate the perceptual distance between the true and the predicted images
    
    y_true: np.array: True image
    y_pred: np.array: Predicted image
    '''
    y_true *= 255.
    y_pred *= 255.
    rmean = (y_true[:, :, :, 0] + y_pred[:, :, :, 0]) / 2
    r = y_true[:, :, :, 0] - y_pred[:, :, :, 0]
    g = y_true[:, :, :, 1] - y_pred[:, :, :, 1]
    b = y_true[:, :, :, 2] - y_pred[:, :, :, 2]

    return K.mean(K.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256)))

# Defining the model

def slice(x):
    return x[:,:,:,:, -1]

wandb.init(config=hyperparams)
config = wandb.config

c=4

inp = Input((config.height, config.width, 4 * 3))
reshaped = Reshape((config.height,config.width,4,3))(inp)
permuted = Permute((1,2,4,3))(reshaped)
noise = GaussianNoise(0.1)(permuted)
last_layer = Lambda(slice, input_shape=(config.width,config.height,3,4), output_shape=(config.height,config.width,3))(noise)
x = Permute((4,1,2,3))(noise)
x =(ConvLSTM2D(filters=c, kernel_size=(3,3),padding='same',name='conv_lstm1', return_sequences=True))(x)

c1=(BatchNormalization())(x)
x = Dropout(0.2)(x)
x =(TimeDistributed(MaxPooling2D(pool_size=(2,2))))(c1)

x =(ConvLSTM2D(filters=2*c,kernel_size=(3,3),padding='same',name='conv_lstm3',return_sequences=True))(x)
c2=(BatchNormalization())(x)
x = Dropout(0.2)(x)

x =(TimeDistributed(MaxPooling2D(pool_size=(2,2))))(c2)
x =(ConvLSTM2D(filters=4*c,kernel_size=(3,3),padding='same',name='conv_lstm4',return_sequences=True))(x)

x =(TimeDistributed(UpSampling2D(size=(2, 2))))(x)
x =(ConvLSTM2D(filters=4*c,kernel_size=(3,3),padding='same',name='conv_lstm5',return_sequences=True))(x)
x =(BatchNormalization())(x)

x =(ConvLSTM2D(filters=2*c,kernel_size=(3,3),padding='same',name='conv_lstm6',return_sequences=True))(x)
x =(BatchNormalization())(x)
x = Add()([c2, x])
x = Dropout(0.2)(x)

x =(TimeDistributed(UpSampling2D(size=(2, 2))))(x)
x =(ConvLSTM2D(filters=c,kernel_size=(3,3),padding='same',name='conv_lstm7',return_sequences=False))(x)
x =(BatchNormalization())(x)
combined = concatenate([last_layer, x])
combined = Conv2D(3, (1,1))(combined)
model=Model(inputs=[inp], outputs=[combined])

model.compile(optimizer='adam', loss='mse', metrics=[perceptual_distance])

# Assuming train_X, train_y, validation_X, validation_y are pre-loaded numpy arrays
model.fit(train_X, train_y,
          batch_size=config.batch_size,
          epochs=config.num_epochs,
          callbacks=[ImageCallback(validation_data=(validation_X, validation_y)), WandbCallback()],
          validation_data=(validation_X, validation_y),
          steps_per_epoch=max(1, len(train_X) // config.batch_size),
          validation_steps=max(1, len(validation_X) // config.batch_size))

model = tf.keras.models.load_model('/content/drive/MyDrive/USC_Model.h5', custom_objects={'perceptual_distance': perceptual_distance})