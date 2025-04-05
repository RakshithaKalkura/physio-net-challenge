#!/usr/bin/env python

# This is your main PhysioNet team code script using a Residual 1D CNN for binary classification.

import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from helper_code import *

################################################################################
# Preprocessing
################################################################################

def preprocess(record_path):
    signal, _ = load_signals(record_path)
    signal = np.nan_to_num(signal)

    if signal.ndim > 1:
        signal = np.mean(signal, axis=0)

    signal = signal[:1000] if len(signal) > 1000 else np.pad(signal, (0, 1000 - len(signal)))
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

    return signal.astype(np.float32)

################################################################################
# Residual 1D CNN
################################################################################

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=False):
        super(ResidualBlock, self).__init__()
        padding = kernel_size // 2
        self.downsample = downsample

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        if self.downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class Residual1DCNN(nn.Module):
    def __init__(self, input_channels=1, input_length=1000, num_classes=1):
        super(Residual1DCNN, self).__init__()
        self.layer1 = ResidualBlock(input_channels, 16, kernel_size=7)
        self.pool1 = nn.MaxPool1d(2)

        self.layer2 = ResidualBlock(16, 32)
        self.pool2 = nn.MaxPool1d(2)

        self.layer3 = ResidualBlock(32, 64)
        self.pool3 = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(64, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool1(self.layer1(x))
        x = self.pool2(self.layer2(x))
        x = self.pool3(self.layer3(x))
        x = x.view(x.size(0), -1)
        return self.sigmoid(self.fc(x))

################################################################################
# Required Functions
################################################################################
from torch.utils.data import Dataset, DataLoader

# Custom Dataset
class ECGDataset(Dataset):
    def __init__(self, data_folder, records):
        self.data_folder = data_folder
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record_path = os.path.join(self.data_folder, self.records[idx])
        x = preprocess(record_path)
        y = load_label(record_path)
        return torch.tensor(x).unsqueeze(0), torch.tensor(y, dtype=torch.float32)

def train_model(data_folder, model_folder, verbose):
    if verbose:
        print("Loading record names...")

    records = find_records(data_folder)
    dataset = ECGDataset(data_folder, records)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

    model = Residual1DCNN(input_channels=1, input_length=1000, num_classes=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    model.train()
    if verbose:
        print("Training model using DataLoader...")

    for epoch in range(10):
        epoch_loss = 0.0
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if verbose:
            print(f"Epoch {epoch+1}, Avg Loss: {epoch_loss / len(dataloader):.4f}")

    os.makedirs(model_folder, exist_ok=True)
    save_model(model_folder, model)

    if verbose:
        print("Model training completed and saved.")


def load_model(model_folder, verbose):
    model = Residual1DCNN(input_channels=1, input_length=1000, num_classes=1)
    model.load_state_dict(torch.load(os.path.join(model_folder, 'model.pth'), map_location=torch.device('cpu'), weights_only=False))
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
# Optional Features (if needed later)
################################################################################

def save_model(model_folder, model):
    d = {'model': model}
    filename = os.path.join(model_folder, 'model.pth')
    torch.save(model.state_dict(), filename)

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
    num_finite_samples = np.size(np.isfinite(signal))
    signal_mean = np.nanmean(signal) if num_finite_samples > 0 else 0.0
    signal_std = np.nanstd(signal) if num_finite_samples > 1 else 0.0

    features = np.concatenate(([age], one_hot_encoding_sex, [signal_mean, signal_std]))
    return np.asarray(features, dtype=np.float32)
