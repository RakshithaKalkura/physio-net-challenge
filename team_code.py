#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import joblib
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sys

from helper_code import *

import tensorflow as tf
from tensorflow.keras import layers, models
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

def preprocess_signal(signal, target_length=5000):
    """Pad or truncate the signal to a fixed length."""
    num_samples, num_channels = signal.shape
    if num_samples > target_length:
        preprocess_signal.truncated += 1
        return signal[:target_length]
    elif num_samples < target_length:
        preprocess_signal.padded += 1
        pad_width = target_length - num_samples
        return np.pad(signal, ((0, pad_width), (0, 0)), 'constant')
    else:
        return signal

# Track counts
preprocess_signal.truncated = 0
preprocess_signal.padded = 0
# Train your model.
def train_model(data_folder, model_folder, verbose):
    # Find the data files.
    if verbose:
        print('Finding the Challenge data...')
    records = find_records(data_folder)
    num_records = len(records)
    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    if verbose:
        print('Loading, normalizing, and processing the data...')

    signals_list = []
    for recording in recordings:
        signal, header = load_challenge_data(recording)
        signal = preprocess_signal(signal)
        signals_list.append(signal)

# Convert to array after uniform shape ensured
    signals = np.array(signals_list, dtype=np.float32)

# Print stats
    print(f"[INFO] Padded signals: {preprocess_signal.padded}")
    print(f"[INFO] Truncated signals: {preprocess_signal.truncated}")

    
    labels_list = []
    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')
        record_path = os.path.join(data_folder, records[i])
        # load_signals returns (num_leads, num_samples)
        signal, fields = load_signals(record_path)
        # Transpose to (num_samples, num_leads)
        signal = signal.T

        # Z normalization per channel: subtract mean and divide by std (avoid division by zero)
        channel_mean = np.mean(signal, axis=0)
        channel_std = np.std(signal, axis=0) + 1e-8
        signal_norm = (signal - channel_mean) / channel_std

        signals_list.append(signal_norm)
        labels_list.append(load_label(record_path))

    # Convert lists to arrays.
    # Assuming all signals have the same shape (num_samples, num_leads)
    signals = np.array(signals_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.float32)

    if verbose:
        print(f'Loaded {signals.shape[0]} records with shape {signals.shape[1:]} each.')

    # Build a Conv1D-based classifier that includes an "encoder" (i.e. autoencoder-like feature extractor)
    input_shape = signals.shape[1:]  # (num_samples, num_leads)
    latent_dim = 128  # Size of the encoded representation

    inputs = layers.Input(shape=input_shape)

    x = layers.Conv1D(32, kernel_size=5, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(64, kernel_size=5, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.GlobalAveragePooling1D()(x)
    latent = layers.Dense(latent_dim, activation='relu', name='encoder')(x)
    outputs = layers.Dense(1, activation='sigmoid')(latent)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    if verbose:
        model.summary()

    if verbose:
        print('Training the Conv1D model...')
    model.fit(signals, labels, epochs=10, batch_size=16, verbose=1)

    # Save the model using the Keras save format.
    os.makedirs(model_folder, exist_ok=True)
    save_path = os.path.join(model_folder, 'model.h5')
    model.save(save_path)
    if verbose:
        print(f'Model saved to {save_path}')
        print('Done.')
        print()

def load_model(model_folder, verbose):

    model_path = os.path.join(model_folder, 'model.h5')
    model = tf.keras.models.load_model(model_path)
    if verbose:
        print(f'Loaded model from {model_path}')
    return model
# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):
    model_filename = os.path.join(model_folder, 'model.sav')
    model = joblib.load(model_filename)
    return model

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(record, model, verbose):
    # Given a record path, load the signal, preprocess it, and run the model.
    # load_signals returns (num_leads, num_samples)
    signal, fields = load_signals(record)
    # Transpose to (num_samples, num_leads)
    signal = signal.T
    # Z normalization per channel.
    channel_mean = np.mean(signal, axis=0)
    channel_std = np.std(signal, axis=0) + 1e-8
    signal_norm = (signal - channel_mean) / channel_std

    # Expand dimensions to form a batch of 1.
    input_signal = np.expand_dims(signal_norm, axis=0).astype(np.float32)
    probability_output = model.predict(input_signal)[0][0]
    binary_output = 1 if probability_output >= 0.5 else 0
    return binary_output, probability_output

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Extract your features.
def extract_features(record):
    header = load_header(record)
    age = get_age(header)
    sex = get_sex(header)

    one_hot_encoding_sex = np.zeros(3, dtype=bool)
    if sex == 'Female':
        one_hot_encoding_sex[0] = 1
    elif sex == 'Male':
        one_hot_encoding_sex[1] = 1
    else:
        one_hot_encoding_sex[2] = 1

    signal, fields = load_signals(record)

    # TO-DO: Update to compute per-lead features. Check lead order and update and use functions for reordering leads as needed.

    num_finite_samples = np.size(np.isfinite(signal))
    if num_finite_samples > 0:
        signal_mean = np.nanmean(signal)
    else:
        signal_mean = 0.0
    if num_finite_samples > 1:
        signal_std = np.nanstd(signal)
    else:
        signal_std = 0.0

    features = np.concatenate(([age], one_hot_encoding_sex, [signal_mean, signal_std]))

    return np.asarray(features, dtype=np.float32)

# Save your trained model.
def save_model(model_folder, model):
    d = {'model': model}
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)