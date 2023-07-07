import tqdm
import random
import pathlib
import itertools
import collections

import cv2
import einops
import numpy as np
import remotezip as rz
import seaborn as sns
import matplotlib.pyplot as plt
from tcn import TCN

import tensorflow as tf
import keras
from keras import layers
from keras.regularizers import l2
from utils import *

URL = 'https://storage.googleapis.com/thumos14_files/UCF101_videos.zip'
download_dir = pathlib.Path('./UCF101_subset/')
subset_paths = download_ufc_101_subset(URL, 
                        num_classes = 10, 
                        splits = {"train": 30, "val": 10, "test": 10},
                        download_dir = download_dir)

n_frames = 10
batch_size = 8

output_signature = (tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32),
                    tf.TensorSpec(shape = (), dtype = tf.int16))

train_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['train'], n_frames, training=True),
                                          output_signature = output_signature)


# Batch the data
train_ds = train_ds.batch(batch_size)

val_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['val'], n_frames),
                                        output_signature = output_signature)
val_ds = val_ds.batch(batch_size)

test_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['test'], n_frames),
                                         output_signature = output_signature)

test_ds = test_ds.batch(batch_size)

# Define the dimensions of one frame in the set of frames created
HEIGHT = 224
WIDTH = 224

input_shape = (None, n_frames, HEIGHT, WIDTH, 3)
input = layers.Input(shape=(input_shape[1:]))
x = input

# Reshape the input to fit the TCN architecture
x = layers.Reshape((n_frames, -1))(x)

# Replace the Conv2Plus1D layers with TCN layers
x = TCN(nb_filters=32, kernel_size=5, dilations=[1, 2, 4], return_sequences=True)(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Dropout(0.3)(x)

# Replace the Conv2Plus1D layers with TCN layers
x = TCN(nb_filters=48, kernel_size=5, dilations=[1, 2, 4], return_sequences=True)(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Dropout(0.3)(x)

# # Block 3
# x = add_residual_block(x, 64, (3, 3, 3))
# x = ResizeVideo(HEIGHT // 16, WIDTH // 16)(x)

# # Block 4
# x = add_residual_block(x, 128, (3, 3, 3))

# x = layers.GlobalAveragePooling1D()(x)
x = layers.Flatten()(x)
x = layers.Dense(10, kernel_regularizer=l2(0.001))(x)

model = keras.Model(input, x)

frames, label = next(iter(train_ds))
model.build(frames)

class EarlyStopping(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    '''
    Stops training when 95% accuracy is reached
    '''
    # Get the current accuracy and check if it is above 95%
    if(logs.get('val_accuracy') > 0.7):

      # Stop training if condition is met
      print("\nThreshold reached. Stopping training...")
      self.model.stop_training = True

# instantiate ES class
early_stopping = EarlyStopping()

model.compile(loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              optimizer = keras.optimizers.Adam(learning_rate = 0.001), 
              metrics = ['accuracy'])

history = model.fit(x = train_ds,
                    epochs = 100, 
                    validation_data = val_ds,
                    callbacks=[early_stopping])

plot_history(history)

model.evaluate(test_ds, return_dict=True)