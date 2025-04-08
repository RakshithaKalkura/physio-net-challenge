import os
import numpy as np
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from model import get_model  
import wfdb
from keras.layers import Dense
from keras.activations import sigmoid



SIGNAL_LENGTH = 4096
NUM_LEADS = 12
NUM_CLASSES = 1 #chagas yes or no
TARGET_FS = 500   # Set based on your task

from scipy.signal import resample

def z_normalize(signal):
    mean = np.mean(signal, axis=0)
    std = np.std(signal, axis=0)
    std[std == 0] = 1e-6  # prevent division by zero
    return (signal - mean) / std

def load_challenge_data(data_folder, max_samples=None):
    signals = []
    labels = []

    for i, file in enumerate(os.listdir(data_folder)):
        if file.endswith('.hea'):
            if max_samples and len(signals) >= max_samples:
                break

            record_name = file[:-4]
            record_path = os.path.join(data_folder, record_name)

            try:
                record = wfdb.rdrecord(record_path)
                signal = record.p_signal
                fs = record.fs

                # Resample
                if fs != TARGET_FS:
                    num_samples = int(signal.shape[0] * TARGET_FS / fs)
                    signal = resample(signal, num_samples, axis=0)

                # Pad/truncate
                if signal.shape[0] > SIGNAL_LENGTH:
                    signal = signal[:SIGNAL_LENGTH]
                elif signal.shape[0] < SIGNAL_LENGTH:
                    pad_width = SIGNAL_LENGTH - signal.shape[0]
                    signal = np.pad(signal, ((0, pad_width), (0, 0)), 'constant')

                signal = z_normalize(signal).astype(np.float32)

                signals.append(signal)
                labels.append(0)  # Replace with actual label

            except Exception as e:
                print(f"Error loading {record_name}: {e}")
                continue

    return np.array(signals, dtype=np.float32), np.array(labels)



def train_model(data_folder, model_folder, verbose=False):
    os.makedirs(model_folder, exist_ok=True)

    if verbose: print("Loading data...")
    X, y = load_challenge_data(data_folder, max_samples=20000) # add max_samples if needed
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    y = y.reshape(-1, 1)

    print("Data shape:", X.shape, y.shape)

    if verbose: print("Building model...")
    model = get_model(NUM_CLASSES, last_layer='sigmoid')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



    checkpoint_path = os.path.join(model_folder, 'best_model.keras')
    checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', mode='min')

    if verbose: print("Training...")
    model.fit(X, y, epochs=10, batch_size=16, validation_split=0.1, callbacks=[checkpoint], verbose=1 if verbose else 0)

    model.save(os.path.join(model_folder, 'model.keras'))
    if verbose: print("Model saved to", model_folder)


def load_model(model_folder, verbose=False):
    model_path = os.path.join(model_folder, 'model.keras')
    if os.path.exists(model_path):
        model = keras_load_model(model_path)
        if verbose:
            print(f"Loaded model from {model_path}")
        return {'model': model}
    else:
        raise FileNotFoundError(f"Model not found at {model_path}")


def run_model(record_path, model, verbose=False):
    try:
        if model is None or 'model' not in model:
            raise ValueError("Model not properly loaded")

        signal, _ = wfdb.rdsamp(record_path)
        signal = signal[:SIGNAL_LENGTH, :NUM_LEADS]

        if signal.shape[0] < SIGNAL_LENGTH:
            pad_width = SIGNAL_LENGTH - signal.shape[0]
            signal = np.pad(signal, ((0, pad_width), (0, 0)), mode='constant')
        elif signal.shape[0] > SIGNAL_LENGTH:
            signal = signal[:SIGNAL_LENGTH, :]

        if signal.shape[1] < NUM_LEADS:
            lead_pad = NUM_LEADS - signal.shape[1]
            signal = np.pad(signal, ((0, 0), (0, lead_pad)), mode='constant')

        signal = np.expand_dims(signal.astype(np.float32), axis=0)
        model_instance = model['model']
        prediction = model_instance.predict(signal)[0]
        binary_prediction = (prediction >= 0.5).astype(int)

        return binary_prediction, prediction

    except Exception as e:
        if verbose:
            print(f"Error in run_model: {e}")
        return np.zeros(NUM_CLASSES), np.zeros(NUM_CLASSES)
