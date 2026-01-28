# ==========================================================================
# HYBRID EEG MOTOR IMAGERY CLASSIFIER (FBCSP + EEGNet-TensorFlow + MLP + CV)
# Dataset: BCI Competition IV – Dataset 2b
# Includes: Bandpass + Notch Filter, FBCSP, EEGNet (TensorFlow), Hybrid Fusion
# ===========================================================================

import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import mne
from mne.decoding import CSP
from scipy.signal import butter, filtfilt, iirnotch

from xgboost import XGBClassifier


from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_curve, auc
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

from tensorflow.keras.optimizers import Adam

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    DATASET_PATH = r"C:\Users\91638\Desktop\BCI\DATA"

    SUBJECT_ID = 5
    SESSION_ID = 1

    MOTOR_CORTEX_CHANNELS = ['C3', 'Cz', 'C4']

    # Epoching
    TMIN = 2.0
    TMAX = 4.0

    # Preprocessing
    BANDPASS_LOW = 8.0
    BANDPASS_HIGH = 30.0
    NOTCH_FREQ = 50.0          # Power line frequency (India = 50 Hz)
    FILTER_ORDER = 5

    # Filter banks
    FILTER_BANKS = [(8,12),(12,16),(16,20),(20,30)]

    # CSP
    N_CSP_COMPONENTS = 4

    # Feature selection
    N_SELECTED_FEATURES = 24

    # Split
    TEST_SIZE = 0.15
    VAL_SIZE = 0.15
    RANDOM_STATE = 42

# ============================================================================
# STAGE 1: LOAD DATA
# ============================================================================

def load_bci_competition_data(config):

    session_type = 'T' if config.SESSION_ID <= 3 else 'E'
    filename = f"B{config.SUBJECT_ID:02d}{config.SESSION_ID:02d}{session_type}.gdf"
    filepath = os.path.join(config.DATASET_PATH, filename)

    raw = mne.io.read_raw_gdf(filepath, preload=True, verbose=False)
    events, event_dict = mne.events_from_annotations(raw, verbose=False)

    selected_events = {
        'left_hand': event_dict['769'],
        'right_hand': event_dict['770']
    }

    eeg_channels = [ch for ch in raw.ch_names if 'EEG' in ch and ch.replace('EEG:', '') in config.MOTOR_CORTEX_CHANNELS]
    raw.pick_channels(eeg_channels)

    epochs = mne.Epochs(raw, events, event_id=selected_events,
                        tmin=config.TMIN, tmax=config.TMAX,
                        baseline=None, preload=True, verbose=False)

    labels = epochs.events[:, -1]
    labels = np.array([0 if l == selected_events['left_hand'] else 1 for l in labels])

    return epochs, labels

# ============================================================================
# STAGE 2: PREPROCESSING (BANDPASS + NOTCH + NORMALIZATION)
# ============================================================================

def apply_bandpass_filter(data, low_freq, high_freq, sfreq, order=5):
    nyquist = sfreq / 2.0
    b, a = butter(order, [low_freq/nyquist, high_freq/nyquist], btype='band')
    filtered = filtfilt(b, a, data, axis=-1)
    return filtered


def apply_notch_filter(data, notch_freq, sfreq, Q=30):
    b, a = iirnotch(notch_freq, Q, sfreq)
    filtered = filtfilt(b, a, data, axis=-1)
    return filtered


def preprocess_data(epochs, config):

    data = epochs.get_data()
    sfreq = epochs.info['sfreq']

    # Notch filter (remove power line noise)
    data = apply_notch_filter(data, config.NOTCH_FREQ, sfreq)

    # Bandpass filter
    data = apply_bandpass_filter(data, config.BANDPASS_LOW, config.BANDPASS_HIGH, sfreq)

    # Normalize
    norm = (data - np.mean(data, axis=2, keepdims=True)) / (np.std(data, axis=2, keepdims=True) + 1e-8)

    return norm

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

        csp_feat = csp.transform(band_data)
        all_features.append(csp_feat)

    features = np.concatenate(all_features, axis=1)

    if is_training:
        return features, csp_filters
    else:
        return features

# ============================================================================
# EEGNET MODEL (TENSORFLOW)
# ============================================================================

def build_eegnet(n_channels, n_samples):

    input_layer = Input(shape=(n_channels, n_samples, 1))

    x = Conv2D(16, (1, 64), padding='same', use_bias=False)(input_layer)
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

    model = Model(inputs=input_layer, outputs=x)

    return model

# ============================================================================
# TRAIN EEGNET & EXTRACT FEATURES (TENSORFLOW)
# ============================================================================

def train_eegnet_and_extract_features(X_train, y_train, X_val, y_val, X_test):
    
    n_channels = X_train.shape[1]
    n_samples  = X_train.shape[2]

    # Reshape for CNN: (trials, channels, samples, 1)
    X_train = X_train[..., np.newaxis]
    X_val   = X_val[..., np.newaxis]
    X_test  = X_test[..., np.newaxis]

    # ---------------- EEGNet Architecture ----------------
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

    model_full = Model(inputs, outputs)

    model_full.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\nTraining EEGNet Feature Extractor (TensorFlow)...")

    model_full.fit(
        X_train, y_train,
        epochs=40,
        batch_size=16,
        verbose=1,
        validation_data=(X_val, y_val)
    )

    # -------- Feature extractor model --------
    feature_model = Model(inputs, features)

    train_feat = feature_model.predict(X_train)
    val_feat   = feature_model.predict(X_val)
    test_feat  = feature_model.predict(X_test)

    return train_feat, val_feat, test_feat

# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main():
    
    cfg = Config()

    # ===================== LOAD & PREPROCESS =====================
    print("\nLoading and preprocessing data...")
    epochs, labels = load_bci_competition_data(cfg)
    sfreq = epochs.info['sfreq']
    data = preprocess_data(epochs, cfg)

    # ===================== DATA SPLIT (70% TRAIN / 15% VAL / 15% TEST) =====================
    X_temp, X_test, y_temp, y_test = train_test_split(
        data, labels,
        test_size=0.15,                  # 15% final test
        random_state=cfg.RANDOM_STATE,
        stratify=labels
    )

    val_ratio = 0.15 / (1 - 0.15)        # validation from remaining 85%

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,             # 15% validation
        random_state=cfg.RANDOM_STATE,
        stratify=y_temp
    )

    print("\nData Shapes:")
    print("Train:", X_train.shape)
    print("Val  :", X_val.shape)
    print("Test :", X_test.shape)

    # ===================== FBCSP =====================
    print("\nExtracting FBCSP features...")

    X_train_fbcsp, csp_filters = extract_fbcsp_features(
        X_train, y_train, cfg, sfreq, True
    )

    X_val_fbcsp = extract_fbcsp_features(
        X_val, y_val, cfg, sfreq, False, csp_filters.copy()
    )

    X_test_fbcsp = extract_fbcsp_features(
        X_test, y_test, cfg, sfreq, False, csp_filters.copy()
    )

    # ===================== EEGNET =====================
    print("\nTraining EEGNet and extracting deep features...")

    eegnet_train_feat, eegnet_val_feat, eegnet_test_feat = train_eegnet_and_extract_features(
        X_train, y_train,
        X_val, y_val,
        X_test
    )

    # ===================== HYBRID FUSION =====================
    print("\nBuilding Hybrid Features...")

    X_train_hybrid = np.concatenate([X_train_fbcsp, eegnet_train_feat], axis=1)
    X_val_hybrid   = np.concatenate([X_val_fbcsp,   eegnet_val_feat],  axis=1)
    X_test_hybrid  = np.concatenate([X_test_fbcsp,  eegnet_test_feat], axis=1)

    print("Hybrid Train:", X_train_hybrid.shape)
    print("Hybrid Val  :", X_val_hybrid.shape)
    print("Hybrid Test :", X_test_hybrid.shape)

    # ===================== FEATURE SELECTION =====================
    print("\nSelecting best features...")

    selector = SelectKBest(mutual_info_classif, k=24)

    X_train_sel = selector.fit_transform(X_train_hybrid, y_train)
    X_val_sel   = selector.transform(X_val_hybrid)
    X_test_sel  = selector.transform(X_test_hybrid)

    print("Selected Feature Shape:", X_train_sel.shape)

    # ===================== MLP + XGBOOST ENSEMBLE =====================
    print("\nTraining MLP + XGBoost Hybrid Ensemble...")

    # ----------- MLP MODEL -----------
    mlp = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            max_iter=800,
            early_stopping=True,
            random_state=42
        ))
    ])

    mlp.fit(X_train_sel, y_train)
    P_mlp = mlp.predict_proba(X_test_sel)

    # ----------- XGBOOST MODEL -----------
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42
    )

    xgb.fit(X_train_sel, y_train)
    P_xgb = xgb.predict_proba(X_test_sel)

    # ----------- SOFT VOTING ENSEMBLE -----------
    P_final = (P_mlp + P_xgb) / 2
    y_pred = np.argmax(P_final, axis=1)

    # ===================== EVALUATION (FINAL TEST SET ONLY) =====================
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    cm   = confusion_matrix(y_test, y_pred)

    print("\nFINAL TEST PERFORMANCE (UNSEEN DATA)")
    print(f"Accuracy  : {acc*100:.2f} %")
    print(f"Precision : {prec*100:.2f} %")
    print(f"Recall    : {rec*100:.2f} %")
    print(f"F1-Score  : {f1*100:.2f} %")

    # ===================== VISUALIZATION =====================
    plt.figure(figsize=(12,4))

    # Confusion Matrix
    plt.subplot(1,2,1)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Left","Right"],
                yticklabels=["Left","Right"])
    plt.title("Confusion Matrix")

    # Accuracy Bar
    plt.subplot(1,2,2)
    plt.bar(["Test Accuracy"], [acc*100])
    plt.ylim(0,100)
    plt.ylabel("Accuracy (%)")
    plt.title("Final Test Accuracy")
    plt.text(0, acc*100+1, f"{acc*100:.2f}%", ha='center')

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    main()
