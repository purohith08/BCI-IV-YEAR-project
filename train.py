import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import mne
from mne.decoding import CSP
from scipy.signal import butter, filtfilt

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline



import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- PYTORCH FOR EEGNET ----------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    DATASET_PATH = r"C:\Users\91638\Desktop\BCI\DATA"

    SUBJECT_ID = 7
    SESSION_ID = 3
    DATASET_TYPE = '2b'

    MOTOR_CORTEX_CHANNELS = ['C3', 'Cz', 'C4']

    # Epoching
    TMIN = 2.0
    TMAX = 4.0

    # Preprocessing
    BANDPASS_LOW = 8.0
    BANDPASS_HIGH = 30.0
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

    # MLP
    MLP_HIDDEN_LAYERS = (128, 64)
    MLP_MAX_ITER = 600
    VERBOSE = True


# ============================================================================
# STAGE 1: LOAD DATA + AUTO EVENT DETECTION
# ============================================================================

def load_bci_competition_data(config):

    session_type = 'T' if config.SESSION_ID <= 3 else 'E'
    filename = f"B{config.SUBJECT_ID:02d}{config.SESSION_ID:02d}{session_type}.gdf"
    filepath = os.path.join(config.DATASET_PATH, filename)

    raw = mne.io.read_raw_gdf(filepath, preload=True, verbose=False)
    events, event_dict = mne.events_from_annotations(raw, verbose=False)

    # Auto-detect MI events (769 = Left, 770 = Right)
    selected_events = {
        'left_hand': event_dict['769'],
        'right_hand': event_dict['770']
    }

    # Pick motor cortex EEG channels
    eeg_channels = [ch for ch in raw.ch_names if 'EEG' in ch and ch.replace('EEG:', '') in config.MOTOR_CORTEX_CHANNELS]
    raw.pick_channels(eeg_channels)

    epochs = mne.Epochs(
    raw, events, event_id=selected_events,
    tmin=config.TMIN, tmax=config.TMAX,
    baseline=None,          # <<< VERY IMPORTANT FIX
    preload=True, verbose=False
)

    labels = epochs.events[:, -1]
    labels = np.array([0 if l == selected_events['left_hand'] else 1 for l in labels])

    return epochs, labels


# ============================================================================
# STAGE 2: PREPROCESSING
# ============================================================================

def apply_bandpass_filter(data, low_freq, high_freq, sfreq, order=5):
    nyquist = sfreq / 2.0
    b, a = butter(order, [low_freq/nyquist, high_freq/nyquist], btype='band')

    filtered = np.zeros_like(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            filtered[i, j] = filtfilt(b, a, data[i, j])

    return filtered


def preprocess_data(epochs, config):
    data = epochs.get_data()
    sfreq = epochs.info['sfreq']

    filtered = apply_bandpass_filter(data, config.BANDPASS_LOW, config.BANDPASS_HIGH, sfreq)

    norm = (filtered - np.mean(filtered, axis=2, keepdims=True)) / (np.std(filtered, axis=2, keepdims=True) + 1e-8)

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
# EEGNET MODEL (FEATURE EXTRACTOR)
# ============================================================================

class EEGNet(nn.Module):
    def __init__(self, n_channels, n_samples):
        super().__init__()

        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(16)
        )

        self.depthwise = nn.Sequential(
            nn.Conv2d(16, 32, (n_channels, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.5)
        )

        self.separable = nn.Sequential(
            nn.Conv2d(32, 32, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.5)
        )

        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwise(x)
        x = self.separable(x)
        x = self.flatten(x)
        return x


# ============================================================================
# TRAIN EEGNET & EXTRACT FEATURES
# ============================================================================

def train_eegnet_and_extract_features(X_train, y_train, X_val, X_test):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)

    X_val_t   = torch.tensor(X_val,  dtype=torch.float32).unsqueeze(1).to(device)
    X_test_t  = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)

    model = EEGNet(X_train.shape[1], X_train.shape[2]).to(device)

    clf_head = nn.Linear(model(X_train_t[:1]).shape[1], 2).to(device)

    optimizer = optim.Adam(list(model.parameters()) + list(clf_head.parameters()), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(X_train_t, y_train_t)
    loader  = DataLoader(dataset, batch_size=16, shuffle=True)

    print("\nTraining EEGNet Feature Extractor...")

    for epoch in range(40):   # more epochs = better features
        for xb, yb in loader:
            optimizer.zero_grad()
            feats = model(xb)
            out = clf_head(feats)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        train_feat = model(X_train_t).cpu().numpy()
        val_feat   = model(X_val_t).cpu().numpy()
        test_feat  = model(X_test_t).cpu().numpy()

    return train_feat, val_feat, test_feat


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    
    cfg = Config()

    # ===================== STAGE 1: LOAD & PREPROCESS =====================
    print("\nLoading and preprocessing data...")
    epochs, labels = load_bci_competition_data(cfg)
    sfreq = epochs.info['sfreq']
    data = preprocess_data(epochs, cfg)

    # ===================== DATA SPLITTING =====================
    X_temp, X_test, y_temp, y_test = train_test_split(
        data, labels,
        test_size=cfg.TEST_SIZE,
        random_state=cfg.RANDOM_STATE,
        stratify=labels
    )

    val_ratio = cfg.VAL_SIZE / (1 - cfg.TEST_SIZE)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        random_state=cfg.RANDOM_STATE,
        stratify=y_temp
    )

    print("\nData Split Shapes:")
    print(" Train :", X_train.shape)
    print(" Val   :", X_val.shape)
    print(" Test  :", X_test.shape)

    # ===================== STAGE 3: FBCSP =====================
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

    # ===================== STAGE 4: EEGNET =====================
    print("\nTraining EEGNet and extracting deep features...")

    eegnet_train_feat, eegnet_val_feat, eegnet_test_feat = train_eegnet_and_extract_features(
        X_train, y_train, X_val, X_test
    )

    # ===================== HYBRID FUSION =====================
    print("\nBuilding Hybrid Features (FBCSP + EEGNet)...")

    X_train_hybrid = np.concatenate([X_train_fbcsp, eegnet_train_feat], axis=1)
    X_val_hybrid   = np.concatenate([X_val_fbcsp,   eegnet_val_feat],  axis=1)
    X_test_hybrid  = np.concatenate([X_test_fbcsp,  eegnet_test_feat], axis=1)

    print(" Hybrid Train :", X_train_hybrid.shape)
    print(" Hybrid Val   :", X_val_hybrid.shape)
    print(" Hybrid Test  :", X_test_hybrid.shape)

    # ===================== FEATURE SELECTION =====================
    selector = SelectKBest(mutual_info_classif, k=cfg.N_SELECTED_FEATURES)

    X_train_sel = selector.fit_transform(X_train_hybrid, y_train)
    X_val_sel   = selector.transform(X_val_hybrid)
    X_test_sel  = selector.transform(X_test_hybrid)

    print("\nSelected Feature Shape:", X_train_sel.shape)

    # ===================== CLASSIFIER TRAINING (IMPROVED MLP) =====================
    print("\nTraining Improved MLP Classifier (Regularized + Scaled)...")

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,                 # L2 regularization
            batch_size=16,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=800,
            shuffle=True,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=cfg.RANDOM_STATE,
            verbose=True
        ))
    ])

    clf.fit(X_train_sel, y_train)

    # ===================== VALIDATION PERFORMANCE =====================
    print("\n" + "="*70)
    print("VALIDATION PERFORMANCE")
    print("="*70)

    y_val_pred = clf.predict(X_val_sel)

    val_acc  = accuracy_score(y_val, y_val_pred)
    val_prec = precision_score(y_val, y_val_pred)
    val_rec  = recall_score(y_val, y_val_pred)
    val_f1   = f1_score(y_val, y_val_pred)

    print(f"Validation Accuracy  : {val_acc*100:.2f} %")
    print(f"Validation Precision : {val_prec*100:.2f} %")
    print(f"Validation Recall    : {val_rec*100:.2f} %")
    print(f"Validation F1-Score  : {val_f1*100:.2f} %")

    # ===================== CROSS-VALIDATION =====================
    print("\n" + "="*70)
    print("CROSS-VALIDATION PERFORMANCE")
    print("="*70)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.RANDOM_STATE)

    cv_scores = cross_val_score(
        clf,
        X_train_sel,
        y_train,
        cv=cv,
        scoring='accuracy'
    )

    for i, score in enumerate(cv_scores):
        print(f" Fold {i+1}: {score*100:.2f} %")

    cv_mean = np.mean(cv_scores)
    cv_std  = np.std(cv_scores)

    print("\nMean CV Accuracy :", cv_mean*100, "%")
    print("Std  CV Accuracy :", cv_std*100, "%")

    # ===================== FINAL TEST =====================
    print("\n" + "="*70)
    print("FINAL TEST PERFORMANCE")
    print("="*70)

    y_pred = clf.predict(X_test_sel)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    cm   = confusion_matrix(y_test, y_pred)

    print(f"Test Accuracy  : {acc*100:.2f} %")
    print(f"Test Precision: {prec*100:.2f} %")
    print(f"Test Recall   : {rec*100:.2f} %")
    print(f"Test F1-Score : {f1*100:.2f} %")

    # ===================== ROC CURVE =====================
    y_prob = clf.predict_proba(X_test_sel)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # ===================== FINAL SCIENTIFIC REPORT =====================
    print("\n" + "="*80)
    print("FINAL MODEL VALIDATION REPORT")
    print("="*80)

    print(f"Validation Accuracy        : {val_acc*100:.2f} %")
    print(f"Mean CV Accuracy           : {cv_mean*100:.2f} ± {cv_std*100:.2f} %")
    print(f"Test Accuracy              : {acc*100:.2f} %")

    print("\nClassification Quality (Test Set):")
    print(f" Precision                 : {prec*100:.2f} %")
    print(f" Recall                    : {rec*100:.2f} %")
    print(f" F1-Score                  : {f1*100:.2f} %")

    print("="*80)

    # ===================== VISUALIZATION =====================

    plt.figure(figsize=(18,4))

    # ---- Confusion Matrix ----
    plt.subplot(1,4,1)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Left","Right"],
                yticklabels=["Left","Right"])
    plt.title("Confusion Matrix")

    # ---- Validation vs Test Accuracy ----
    plt.subplot(1,4,2)
    scores = [val_acc*100, acc*100]
    labels_plot = ["Validation", "Test"]
    plt.bar(labels_plot, scores)
    plt.ylim(0,100)
    plt.ylabel("Accuracy (%)")
    plt.title("Validation vs Test")

    for i, v in enumerate(scores):
        plt.text(i, v+1, f"{v:.2f}%", ha='center', fontsize=11)

    # ---- Cross-Validation Stability ----
    plt.subplot(1,4,3)
    plt.boxplot(cv_scores*100)
    plt.ylabel("Accuracy (%)")
    plt.title("Cross-Validation Stability")

    # ---- ROC Curve ----
    plt.subplot(1,4,4)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    plt.suptitle("Hybrid FBCSP + EEGNet + Improved MLP Validation", fontsize=14)
    plt.tight_layout()
    plt.show()

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    main()
