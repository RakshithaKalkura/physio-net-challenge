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

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler


################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

def preprocess(record_path):
    signal, _ = load_signals(record_path)
    signal = np.nan_to_num(signal)

    if signal.ndim > 1:
        signal = np.mean(signal, axis=0)

    signal = signal[:1000] if len(signal) > 1000 else np.pad(signal, (0, 1000 - len(signal)))
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

    return signal.astype(np.float32)
################################################################################
# Required functions
################################################################################
# Encoder (from autoencoder)
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

    def forward(self, x):
        return self.encoder(x)

# Classifier on top of encoder
class ECGClassifier(nn.Module):
    def __init__(self):
        super(ECGClassifier, self).__init__()
        self.encoder = Encoder()
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(32 * 250, 128),  # depends on input size 
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        return self.classifier(x)
def train_model(data_folder, model_folder, verbose):
    if verbose:
        print("Loading and preprocessing data...")

    records = find_records(data_folder)
    X_raw, y = [], []

    for rec in records:
        record_path = os.path.join(data_folder, rec)
        x = preprocess(record_path)
        X_raw.append(x)
        y.append(load_label(record_path))

    X_raw = np.array(X_raw)
    y = np.array(y).astype(np.float32)

    X_tensor = torch.tensor(X_raw).unsqueeze(1).float()
    y_tensor = torch.tensor(y).unsqueeze(1).float()

    model = ECGClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    if verbose:
        print("Training full model (encoder + classifier)...")

    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()

        if verbose:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    os.makedirs(model_folder, exist_ok=True)
    save_model(model_folder, model)

    if verbose:
        print("Model saved.\n")

def load_model(model_folder, verbose):
    model_path = os.path.join(model_folder, 'model.pth')
    model = ECGClassifier()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=False))
    model.eval()
    return {'model': model}

def run_model(record, model, verbose):
    model = model['model']
    x = preprocess(record)
    x_tensor = torch.tensor(x).unsqueeze(0).unsqueeze(0).float()

    with torch.no_grad():
        prob = model(x_tensor).item()

    binary_output = prob > 0.5
    return binary_output, prob

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
    filename = os.path.join(model_folder, 'model.pt')
    torch.save(model.state_dict(), filename)