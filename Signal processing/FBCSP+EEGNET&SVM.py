# ==========================================================================
# HYBRID EEG MOTOR IMAGERY CLASSIFIER
# FBCSP + EEGNet (TensorFlow) + SVM
# Dataset: BCI Competition IV – Dataset 2b
# Task: LEFT vs RIGHT Motor Imagery
# ==========================================================================

import os
from xml.parsers.expat import model
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import mne
from mne.decoding import CSP
from scipy.signal import butter, filtfilt, iirnotch

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- TENSORFLOW FOR EEGNET ----------------
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, DepthwiseConv2D, SeparableConv2D,
    BatchNormalization, Activation, AveragePooling2D,
    Dropout, Flatten, Dense
)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    DATASET_PATH = r"C:\Users\91638\Desktop\BCI\DATA"

    SUBJECT_ID = 1
    SESSION_ID = 1

    MOTOR_CORTEX_CHANNELS = ['C3', 'Cz', 'C4']

    # Epoching
    TMIN = 2.0
    TMAX = 4.0

    # Preprocessing
    BANDPASS_LOW = 8.0
    BANDPASS_HIGH = 30.0
    NOTCH_FREQ = 50.0

    # Filter banks
    FILTER_BANKS = [(8,12),(12,16),(16,20),(20,30)]

    # CSP
    N_CSP_COMPONENTS = 4

    # Feature selection
    N_SELECTED_FEATURES = 24

    RANDOM_STATE = 42


# ============================================================================
# STAGE 1: LOAD DATA (LEFT vs RIGHT ONLY)
# ============================================================================

def load_bci_competition_data(config):

    session_type = 'T' if config.SESSION_ID <= 3 else 'E'
    filename = f"B{config.SUBJECT_ID:02d}{config.SESSION_ID:02d}{session_type}.gdf"
    filepath = os.path.join(config.DATASET_PATH, filename)

    raw = mne.io.read_raw_gdf(filepath, preload=True, verbose=False)
    events, event_dict = mne.events_from_annotations(raw, verbose=False)

    # Dataset 2b labels: 769 = Left, 770 = Right
    selected_events = {
        'left_hand':  event_dict['769'],
        'right_hand': event_dict['770']
    }

    eeg_channels = [ch for ch in raw.ch_names
                    if 'EEG' in ch and ch.replace('EEG:', '') in config.MOTOR_CORTEX_CHANNELS]
    raw.pick_channels(eeg_channels)

    epochs = mne.Epochs(raw, events, event_id=selected_events,
                        tmin=config.TMIN, tmax=config.TMAX,
                        baseline=None, preload=True, verbose=False)

    labels = epochs.events[:, -1]
    labels = np.array([0 if l == selected_events['left_hand'] else 1 for l in labels])

    print("\nClass distribution:", np.bincount(labels))

    return epochs, labels


# ============================================================================
# STAGE 2: PREPROCESSING
# ============================================================================

def apply_bandpass_filter(data, low_freq, high_freq, sfreq, order=5):
    nyquist = sfreq / 2.0
    b, a = butter(order, [low_freq/nyquist, high_freq/nyquist], btype='band')
    return filtfilt(b, a, data, axis=-1)


def apply_notch_filter(data, notch_freq, sfreq, Q=30):
    b, a = iirnotch(notch_freq, Q, sfreq)
    return filtfilt(b, a, data, axis=-1)


def preprocess_data(epochs, config):

    data = epochs.get_data()
    sfreq = epochs.info['sfreq']

    data = apply_notch_filter(data, config.NOTCH_FREQ, sfreq)
    data = apply_bandpass_filter(data, config.BANDPASS_LOW, config.BANDPASS_HIGH, sfreq)

    # Normalize
    data = (data - np.mean(data, axis=2, keepdims=True)) / \
        (np.std(data, axis=2, keepdims=True) + 1e-8)

    return data


# ============================================================================
# STAGE 3: FBCSP
# ============================================================================

def extract_fbcsp_features(data, labels, config, sfreq, is_training=True, csp_filters=None):

    all_features = []
    if is_training:
        csp_filters = []

    for low, high in config.FILTER_BANKS:

        band_data = apply_bandpass_filter(data, low, high, sfreq)

        if is_training:
            csp = CSP(n_components=config.N_CSP_COMPONENTS, log=True)
            csp.fit(band_data, labels)
            csp_filters.append(csp)
        else:
            csp = csp_filters.pop(0)

        all_features.append(csp.transform(band_data))

    features = np.concatenate(all_features, axis=1)

    if is_training:
        return features, csp_filters
    else:
        return features


# ============================================================================
# EEGNET FEATURE EXTRACTOR (BINARY)
# ============================================================================

def train_eegnet_and_extract_features(X_train, y_train, X_val, y_val, X_test):

    n_channels = X_train.shape[1]
    n_samples  = X_train.shape[2]

    X_train = X_train[..., np.newaxis]
    X_val   = X_val[..., np.newaxis]
    X_test  = X_test[..., np.newaxis]

    inputs = Input(shape=(n_channels, n_samples, 1))

    x = Conv2D(16, (1, 64), padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)

    x = DepthwiseConv2D((n_channels, 1), use_bias=False, depth_multiplier=2)(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((1, 4))(x)
    x = Dropout(0.5)(x)

    x = SeparableConv2D(32, (1, 16), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((1, 8))(x)
    x = Dropout(0.5)(x)

    x = Flatten()(x)

    features = x
    outputs = Dense(2, activation='softmax')(features)

    model = Model(inputs, outputs)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\nTraining EEGNet Feature Extractor...")
    
    history = model.fit(X_train, y_train,
                    epochs=40, batch_size=16,
                    validation_data=(X_val, y_val),
                    verbose=1)


    feature_model = Model(inputs, features)

    return (feature_model.predict(X_train),
        feature_model.predict(X_val),
        feature_model.predict(X_test),
        history)



# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():

    cfg = Config()

    print("\nLoading data...")
    epochs, labels = load_bci_competition_data(cfg)
    data = preprocess_data(epochs, cfg)
    sfreq = epochs.info['sfreq']

    # Train / Val / Test split
    X_temp, X_test, y_temp, y_test = train_test_split(
        data, labels, test_size=0.15,
        random_state=cfg.RANDOM_STATE, stratify=labels
    )

    val_ratio = 0.15 / 0.85

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio,
        random_state=cfg.RANDOM_STATE, stratify=y_temp
    )

    # ===================== FBCSP =====================
    X_train_fbcsp, csp_filters = extract_fbcsp_features(X_train, y_train, cfg, sfreq, True)
    X_val_fbcsp  = extract_fbcsp_features(X_val, y_val, cfg, sfreq, False, csp_filters.copy())
    X_test_fbcsp = extract_fbcsp_features(X_test, y_test, cfg, sfreq, False, csp_filters.copy())

    # ===================== EEGNET =====================
    eeg_train, eeg_val, eeg_test, history = train_eegnet_and_extract_features(
    X_train, y_train, X_val, y_val, X_test
)

    # ===================== HYBRID FUSION =====================
    X_train_h = np.concatenate([X_train_fbcsp, eeg_train], axis=1)
    X_val_h   = np.concatenate([X_val_fbcsp,   eeg_val],   axis=1)
    X_test_h  = np.concatenate([X_test_fbcsp,  eeg_test],  axis=1)

    # ===================== FEATURE SELECTION =====================
    selector = SelectKBest(mutual_info_classif, k=cfg.N_SELECTED_FEATURES)
    X_train_sel = selector.fit_transform(X_train_h, y_train)
    X_test_sel  = selector.transform(X_test_h)

    # ===================== SVM CLASSIFIER =====================
    print("\nTraining SVM on Hybrid Features...")

    svm = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=10, gamma='scale', probability=True))
    ])

    svm.fit(X_train_sel, y_train)
    y_pred = svm.predict(X_test_sel)

    # ===================== EVALUATION =====================
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    cm   = confusion_matrix(y_test, y_pred)

    print("\nFINAL TEST PERFORMANCE (LEFT vs RIGHT)")
    print(f"Accuracy  : {acc*100:.2f} %")
    print(f"Precision : {prec*100:.2f} %")
    print(f"Recall    : {rec*100:.2f} %")
    print(f"F1-Score  : {f1*100:.2f} %")
    
    # ===================== FINAL VISUALIZATION (ONE FIGURE) =====================

    plt.figure(figsize=(12,5))

    # ---- Subplot 1: EEGNet Accuracy Curve ----
    plt.subplot(1, 2, 1)

    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('EEGNet Training vs Validation Accuracy')
    plt.legend()
    plt.grid(True)

    # ---- Subplot 2: Confusion Matrix ----
    plt.subplot(1, 2, 2)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Left","Right"],
                yticklabels=["Left","Right"])

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    main()
