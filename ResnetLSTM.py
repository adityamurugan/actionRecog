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

import tensorflow as tf
import keras
from keras import layers
from utils import *
from keras.applications import ResNet152V2

URL = 'https://storage.googleapis.com/thumos14_files/UCF101_videos.zip'
download_dir = pathlib.Path('./UCF101_subset/')
subset_paths = download_ufc_101_subset(URL, 
                        num_classes=10, 
                        splits={"train": 30, "val": 10, "test": 10},
                        download_dir=download_dir)

n_frames = 2
fc_hidden1 = 512
fc_hidden2 = 256
batch_size = 8

output_signature = (tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32),
                    tf.TensorSpec(shape=(), dtype=tf.int16))

train_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['train'], n_frames, training=True),
                                          output_signature=output_signature)

# Batch the data
train_ds = train_ds.batch(batch_size)

val_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['val'], n_frames),
                                        output_signature=output_signature)
val_ds = val_ds.batch(batch_size)

test_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['test'], n_frames),
                                         output_signature=output_signature)

test_ds = test_ds.batch(batch_size)

# Define the dimensions of one frame in the set of frames created
HEIGHT = 224
WIDTH = 224

input_shape = (None, n_frames, HEIGHT, WIDTH, 3)
input = layers.Input(shape=(input_shape[1:]))
x = input

# Reshape the input to a 4D tensor
x = layers.Reshape((n_frames, HEIGHT, WIDTH, 3))(x)

# ResNet-50 backbone
base_model = ResNet152V2(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
base_model.trainable = True

# Apply the ResNet-50 model to each frame
frame_features = []
for i in range(n_frames):
    frame = x[:, i, :, :, :]
    frame = tf.keras.applications.resnet.preprocess_input(frame)
    frame_feature = base_model(frame)
    frame_features.append(frame_feature)

# Concatenate frame features
x = tf.stack(frame_features, axis=1)
x = layers.Reshape((-1, 2048))(x)

# Fully connected layers
x = layers.Dense(fc_hidden1)(x)
x = layers.BatchNormalization(momentum=0.01)(x)
x = layers.ReLU()(x)

x = layers.Dense(fc_hidden2)(x)
x = layers.BatchNormalization(momentum=0.01)(x)
x = layers.ReLU()(x)

x = layers.Dense(128)(x)
x = layers.LSTM(256)(x)
x = layers.Dense(10)(x)

model = keras.Model(input, x)

frames, label = next(iter(train_ds))
model.build(frames.shape)

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              optimizer=keras.optimizers.Adam(learning_rate=0.0001), 
              metrics=['accuracy'])

history = model.fit(x=train_ds,
                    epochs=50, 
                    validation_data=val_ds)

plot_history(history)

model.evaluate(test_ds, return_dict=True)
