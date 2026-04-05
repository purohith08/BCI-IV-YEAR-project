# 🧠 BCI Maze Navigation Game

A real-time, closed-loop **Brain–Computer Interface (BCI)** maze game using
simulated (or live) EEG motor-imagery signals to navigate a 2D maze.

---

## 📦 Requirements

```bash
pip install pygame numpy scipy scikit-learn
```

| Package | Version tested |
|---------|---------------|
| Python  | 3.9 +         |
| pygame  | 2.x           |
| numpy   | 1.24 +        |
| scipy   | 1.10 +        |
| scikit-learn | 1.2 +  |

---

## ▶️ Running

```bash
python bci_maze_game.py
```

You will be prompted for:
- **Subject ID** (e.g. `S01`)
- **Mode**: `FREE` (no ground-truth) or `EVAL` (cued trials + accuracy)

---

## 🕹️ Keyboard Controls

| Key | Action |
|-----|--------|
| `←` / `→` arrows | Manual direction override |
| `R` | Restart maze |
| `Q` or `ESC` | Quit and save CSV results |

---

## 🗂️ File Structure

```
bci_maze_game.py       ← Main game script (all-in-one)
model.pkl              ← (optional) your pre-trained scikit-learn model
bci_results/           ← CSV output directory (auto-created)
```

---

## 🔌 Plug In Your Real EEG Hardware

Find the `get_live_eeg()` function and replace the body with your SDK call:

```python
def get_live_eeg(n_channels, n_samples):
    # Example: BrainFlow
    # board.get_board_data(n_samples)
    # Example: pylsl
    # samples, _ = inlet.pull_chunk(max_samples=n_samples)
    ...
```

Then in `BCIMazeGame._eeg_step()` swap the `simulate_eeg(...)` call for
`get_live_eeg(...)`.

---

## 🤖 Plug In Your Own Model

Drop a scikit-learn `.pkl` file (fitted with `predict` / `predict_proba`)
and set the path in `CFG`:

```python
CFG = {
    "model_path": "model.pkl",   # ← your model file here
    ...
}
```

The model receives a **1-D feature vector** from `extract_features()` and
must return `0` (LEFT) or `1` (RIGHT).

If the file is not found the game uses the built-in heuristic demo model
(hemispheric mu-band asymmetry).

---

## ⚙️ Configuration (`CFG` dict)

| Key | Default | Description |
|-----|---------|-------------|
| `sample_rate` | 256 | EEG sampling rate (Hz) |
| `n_channels` | 8 | Number of EEG channels |
| `window_sec` | 2.0 | Analysis window length |
| `window_step_sec` | 0.5 | Prediction step size |
| `bandpass` | (8, 30) | Mu + beta bandpass |
| `smooth_n` | 5 | Majority-vote window |
| `forward_interval` | 1.5 s | Auto-step interval |
| `cell_size` | 48 px | Maze cell pixel size |
| `maze_rows/cols` | 13 × 19 | Maze dimensions |
| `cue_duration` | 3.0 s | Evaluation cue length |

---

## 📊 Output CSV

After every session a CSV is saved to `bci_results/`:

```
timestamp, true, predicted, correct, confidence
0.502, LEFT, LEFT, 1, 0.7812
1.004, RIGHT, LEFT, 0, 0.5431
...
```

---

## 🧩 Module Map

| Function / Class | Purpose |
|-----------------|---------|
| `EEGBuffer` | Circular ring buffer for EEG stream |
| `simulate_eeg()` | Realistic mu/beta EEG simulation |
| `get_live_eeg()` | ← **replace with hardware SDK** |
| `bandpass_filter()` | Butterworth bandpass |
| `preprocess()` | Filter + CAR |
| `extract_features()` | Stat + band-power features |
| `load_model()` | Load `.pkl` or fallback model |
| `predict_direction()` | Single-window inference |
| `smooth_prediction()` | Majority-vote smoothing |
| `Player` | Position + pixel interpolation |
| `AccuracyTracker` | Confusion matrix + trial log |
| `save_results()` | CSV export |
| `BCIMazeGame` | Full game loop |

---

## 🔬 Extending Feature Extraction

`extract_features(buffer, sr)` is modular — add CSP, PLV, Hjorth, or
Riemannian features here without touching anything else.

---

## 📝 License
MIT — free to use, modify, and share.
