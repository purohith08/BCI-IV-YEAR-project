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
import matplotlib.pyplot as plt
import seaborn as sns



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
    BASELINE = None

    # Preprocessing
    BANDPASS_LOW = 8.0
    BANDPASS_HIGH = 30.0
    FILTER_ORDER = 5

    # Filter banks
    FILTER_BANKS = [(8,12),(12,16),(16,20),(20,30)]

    # CSP
    N_CSP_COMPONENTS = 4

    # Feature selection
    USE_FEATURE_SELECTION = True
    N_SELECTED_FEATURES = 16

    # Split
    TEST_SIZE = 0.15
    VAL_SIZE = 0.15
    RANDOM_STATE = 42

    # MLP
    MLP_HIDDEN_LAYERS = (64, 32)
    MLP_MAX_ITER = 400
    VERBOSE = True


# ============================================================================
# STAGE 1: LOAD DATA + AUTO-DETECT EVENTS
# ============================================================================

def load_bci_competition_data(config):

    print("\n" + "="*70)
    print("STAGE 1: LOADING BCI COMPETITION IV DATASET")
    print("="*70)

    session_type = 'T' if config.SESSION_ID <= 3 else 'E'
    filename = f"B{config.SUBJECT_ID:02d}{config.SESSION_ID:02d}{session_type}.gdf"
    filepath = os.path.join(config.DATASET_PATH, filename)

    raw = mne.io.read_raw_gdf(filepath, preload=True, verbose=False)

    events, event_dict = mne.events_from_annotations(raw, verbose=False)

    print("\nEVENT DICTIONARY:")
    for k, v in event_dict.items():
        print(f"  {k} -> {v}")

    # Auto detect left & right
    left_key, right_key = None, None

    for key in event_dict.keys():
        if '769' in key:
            left_key = key
        if '770' in key:
            right_key = key

    if left_key is None or right_key is None:
        raise RuntimeError("ERROR: Could not detect Left/Right events")

    print(f"\nDetected Left  = {left_key}")
    print(f"Detected Right = {right_key}")

    selected_events = {
        'left_hand': event_dict[left_key],
        'right_hand': event_dict[right_key]
    }

    # Pick only EEG channels
    eeg_channels = []
    for ch in raw.ch_names:
        if 'EEG' in ch:
            name = ch.replace('EEG:', '')
            if name in config.MOTOR_CORTEX_CHANNELS:
                eeg_channels.append(ch)

    raw.pick_channels(eeg_channels)
    print("Using channels:", raw.ch_names)

    # Epoching
    epochs = mne.Epochs(
        raw, events, event_id=selected_events,
        tmin=config.TMIN, tmax=config.TMAX,
        baseline=config.BASELINE, preload=True, verbose=False
    )

    labels = epochs.events[:, -1]
    label_mapping = {
        selected_events['left_hand']: 0,
        selected_events['right_hand']: 1
    }
    labels = np.array([label_mapping[l] for l in labels])

    print(f"✓ Epochs created: {len(epochs)}")
    print("Label distribution:", np.unique(labels, return_counts=True))

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

    norm = np.zeros_like(filtered)
    for i in range(filtered.shape[0]):
        norm[i] = (filtered[i] - np.mean(filtered[i])) / (np.std(filtered[i]) + 1e-8)

    print("✓ Preprocessed shape:", norm.shape)
    return norm


# ============================================================================
# STAGE 3: FBCSP (FIXED VERSION)
# ============================================================================

def extract_fbcsp_features(data, labels, config, sfreq, is_training=True, csp_filters=None):
    
    print("\n" + "="*70)
    print("STAGE 3: FEATURE EXTRACTION (FBCSP)")
    print("="*70)

    all_features = []
    if is_training:
        csp_filters = []

    for low, high in config.FILTER_BANKS:

        # Bandpass for this sub-band
        band_data = apply_bandpass_filter(data, low, high, sfreq)

        # Train or load CSP
        if is_training:
            csp = CSP(n_components=config.N_CSP_COMPONENTS, log=False)
            csp.fit(band_data, labels)
            csp_filters.append(csp)
        else:
            csp = csp_filters.pop(0)

        # CSP transform
        csp_feat = csp.transform(band_data)   # shape = (epochs, components)

        # IMPORTANT: CSP already gives variance features -> just take log
        log_var = np.log(np.abs(csp_feat))

        # Ensure correct 2D shape
        if log_var.ndim == 1:
            log_var = log_var.reshape(-1, 1)

        all_features.append(log_var)

    # Concatenate all filter-bank features
    features = np.concatenate(all_features, axis=1)
    print("✓ FBCSP feature shape:", features.shape)

    if is_training:
        return features, csp_filters
    else:
        return features


# ============================================================================
# STAGE 4: FEATURE SELECTION
# ============================================================================

def select_features(X_train, y_train, X_val, X_test, config):

    if not config.USE_FEATURE_SELECTION:
        return X_train, X_val, X_test, None

    selector = SelectKBest(mutual_info_classif, k=config.N_SELECTED_FEATURES)

    X_train_sel = selector.fit_transform(X_train, y_train)
    X_val_sel = selector.transform(X_val)
    X_test_sel = selector.transform(X_test)

    print("✓ Selected features:", X_train_sel.shape[1])
    return X_train_sel, X_val_sel, X_test_sel, selector


# ============================================================================
# STAGE 5: CLASSIFIER
# ============================================================================

def train_mlp_classifier(X_train, y_train, X_val, y_val, config):

    clf = MLPClassifier(
        hidden_layer_sizes=config.MLP_HIDDEN_LAYERS,
        max_iter=config.MLP_MAX_ITER,
        random_state=config.RANDOM_STATE,
        verbose=config.VERBOSE
    )

    clf.fit(X_train, y_train)

    val_pred = clf.predict(X_val)
    print("✓ Validation accuracy:", accuracy_score(y_val, val_pred))

    return clf


# ============================================================================
# STAGE 6: EVALUATION
# ============================================================================

def evaluate_model(clf, X_test, y_test):
    
    print("\n" + "="*70)
    print("STAGE 6: FINAL EVALUATION METRICS")
    print("="*70)

    # Predictions
    y_pred = clf.predict(X_test)

    # Core metrics
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Print metrics in percentage
    print("\nFINAL PERFORMANCE METRICS:")
    print(f"Accuracy  : {accuracy*100:.2f} %")
    print(f"Precision : {precision*100:.2f} %")
    print(f"Recall    : {recall*100:.2f} %")
    print(f"F1-Score  : {f1*100:.2f} %")

    print("\nCONFUSION MATRIX:")
    print(cm)

    # Detailed classification report
    print("\nCLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred, target_names=["Left Hand", "Right Hand"]))

    return accuracy, precision, recall, f1, cm

# ============================================================================
# COMBINED VISUALIZATION (CONFUSION MATRIX + ACCURACY IN ONE FIGURE)
# ============================================================================

def plot_results(cm, accuracy):

    plt.figure(figsize=(10,4))

    # ---------------- Confusion Matrix ----------------
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Left Hand", "Right Hand"],
                yticklabels=["Left Hand", "Right Hand"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    # ---------------- Accuracy Bar ----------------
    plt.subplot(1, 2, 2)
    plt.bar(["Accuracy"], [accuracy*100])
    plt.ylim(0, 100)
    plt.ylabel("Accuracy (%)")
    plt.title("Final Accuracy")

    # Value on top of bar
    plt.text(0, accuracy*100 + 1, f"{accuracy*100:.2f}%", ha='center', fontsize=12)

    plt.suptitle("Motor Imagery Classification Performance", fontsize=14)
    plt.tight_layout()
    plt.show()


def main():
    
    print("\n" + "="*70)
    print("EEG MOTOR IMAGERY CLASSIFIER - OFFLINE TRAINING PIPELINE")
    print("="*70)

    config = Config()

    # ---------------- STAGE 1 ----------------
    epochs, labels = load_bci_competition_data(config)
    sfreq = epochs.info['sfreq']

    # ---------------- STAGE 2 ----------------
    data = preprocess_data(epochs, config)

    # ---------------- SPLITTING ----------------
    X_temp, X_test, y_temp, y_test = train_test_split(
        data, labels,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=labels
    )

    val_ratio = config.VAL_SIZE / (1 - config.TEST_SIZE)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        random_state=config.RANDOM_STATE,
        stratify=y_temp
    )

    # ---------------- STAGE 3 ----------------
    X_train_fbcsp, csp_filters = extract_fbcsp_features(
        X_train, y_train, config, sfreq, True
    )

    X_val_fbcsp = extract_fbcsp_features(
        X_val, y_val, config, sfreq, False, csp_filters.copy()
    )

    X_test_fbcsp = extract_fbcsp_features(
        X_test, y_test, config, sfreq, False, csp_filters.copy()
    )

    # ---------------- STAGE 4 ----------------
    X_train_sel, X_val_sel, X_test_sel, selector = select_features(
        X_train_fbcsp, y_train,
        X_val_fbcsp, X_test_fbcsp,
        config
    )

    # ---------------- STAGE 5 ----------------
    clf = train_mlp_classifier(
        X_train_sel, y_train,
        X_val_sel, y_val,
        config
    )

    # ====================================================================
    # STAGE 5B: CROSS-VALIDATION (RESEARCH-GRADE)
    # ====================================================================

    print("\n" + "="*70)
    print("STAGE 5B: CROSS-VALIDATION PERFORMANCE")
    print("="*70)

    from sklearn.model_selection import StratifiedKFold, cross_val_score

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores = cross_val_score(
        clf,
        X_train_sel,
        y_train,
        cv=cv,
        scoring='accuracy'
    )

    print("\nCross-Validation Accuracies (5-fold):")
    for i, score in enumerate(cv_scores):
        print(f" Fold {i+1}: {score*100:.2f} %")

    print("\nMean CV Accuracy :", np.mean(cv_scores)*100, "%")
    print("Std  CV Accuracy :", np.std(cv_scores)*100, "%")

    # ---------------- STAGE 6 ----------------
    acc, precision, recall, f1, cm = evaluate_model(clf, X_test_sel, y_test)

    # ---------------- FINAL SUMMARY ----------------
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Accuracy  : {acc*100:.2f} %")
    print(f"Precision : {precision*100:.2f} %")
    print(f"Recall    : {recall*100:.2f} %")
    print(f"F1-Score  : {f1*100:.2f} %")
    print("="*70)

    # ---------------- PLOT RESULTS ----------------
    plot_results(cm, acc)

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    main()
