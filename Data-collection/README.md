# Motor Imagery EEG Data Collection System
## AO + MI Protocol — Research Grade BCI Experiment

---

## Overview

A full Python-based EEG data collection system for **Action Observation (AO)** and
**Motor Imagery (MI)** experiments. Features real-time EEG collection (or simulation),
animated visual cues for left/right hand movements, precise timing synchronization,
and automatic structured dataset saving.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Generate image-based hand animation frames
python generate_assets.py

# 3. Run the experiment
python mi_eeg_system.py
```

---

## Protocol Design

Each trial follows this sequence:

| Phase            | Duration | Label      | Description                        |
|------------------|----------|------------|------------------------------------|
| REST             | 7 sec    | REST       | Breathing animation, relax         |
| PREPARE LEFT     | 3 sec    | PREPARE    | Countdown 3->2->1, get ready         |
| LEFT MI          | 15 sec   | LEFT_MI    | Animated left hand, imagine motion |
| REST             | 7 sec    | REST       | Recovery rest                      |
| PREPARE RIGHT    | 3 sec    | PREPARE    | Countdown 3->2->1                    |
| RIGHT MI         | 15 sec   | RIGHT_MI   | Animated right hand, imagine motion|
| FINAL REST       | 7 sec    | REST       | Post-trial rest                    |
| INTER-TRIAL      | 10 sec   | INTER      | Break between trials               |

**Total: 20 trials** — approximately 74 seconds per trial × 20 = ~24.6 minutes.

---

## Directory Structure

```
MI_EEG_System/
├── mi_eeg_system.py        # Main experiment script
├── generate_assets.py      # Generates hand animation frames
├── requirements.txt
├── README.md
├── mi_eeg_experiment.log   # Auto-generated runtime log
│
├── assets/                 # (Optional) Hand animation assets
│   ├── left_hand.gif       # OR
│   ├── right_hand.gif
│   ├── left_frames/        # PNG sequence (frame001.png ...)
│   └── right_frames/
│
└── EEG_Dataset/
    └── Subject_S01/
        └── 2025-01-15/
            ├── metadata.json
            ├── SS01_T01.csv
            ├── SS01_T02.csv
            └── ...
```

---

## CSV Data Format

Each trial CSV contains:

```
sample_idx, timestamp, C3, C4, Cz, Fz, label_int, label_str
0, 0.0039, 7.2341, -2.1204, 5.8821, 1.2341, 0, REST
1, 0.0078, 7.1205, -2.3401, 5.9012, 1.1820, 0, REST
...
```

| Column      | Description                                          |
|-------------|------------------------------------------------------|
| sample_idx  | Sequential sample index within trial                 |
| timestamp   | Time in seconds from EEG thread start                |
| C3, C4, Cz, Fz | EEG channels in µV                              |
| label_int   | 0=REST, 1=PREPARE, 2=LEFT_MI, 3=RIGHT_MI, 4=INTER   |
| label_str   | Human-readable label                                 |

---

## metadata.json Example

```json
{
  "subject_id": "S01",
  "session_date": "2025-01-15",
  "protocol": "AO + MI",
  "n_trials_completed": 20,
  "sampling_rate_hz": 256,
  "n_channels": 4,
  "channel_names": ["C3", "C4", "Cz", "Fz"],
  "timing_seconds": {
    "rest_duration": 7,
    "preparation_duration": 3,
    "imagery_duration": 15,
    "inter_trial_duration": 10,
    "total_per_trial": 49
  },
  "labels": {
    "0": "REST", "1": "PREPARE",
    "2": "LEFT_MI", "3": "RIGHT_MI", "4": "INTER"
  }
}
```

---

## Animation System

### Procedural (Default — no files needed)
The system draws animated hands using Pygame's drawing API:
- **Palm** — rotated polygon following a sine-wave motion
- **4 fingers + thumb** — spread/close with movement
- **Directional arrows** — fade-in arrows showing movement direction
- **Glow effect** — radial soft glow pulsing with the animation
- **Background particles** — flowing energy particles in phase colors

### Image-Based (Optional)
Place your own assets in `assets/`:
```
assets/left_hand.gif          # Animated GIF (requires Pillow)
assets/right_hand.gif
assets/left_frames/frame*.png # PNG sequence
assets/right_frames/frame*.png
```

---

## EEG Simulation Details

The simulated EEG generates physiologically plausible signals:

| Phase    | Signal Characteristics                                  |
|----------|---------------------------------------------------------|
| REST     | Strong mu (8–12 Hz) + alpha, low noise                 |
| LEFT_MI  | Mu suppression at C4 (contralateral), beta boost       |
| RIGHT_MI | Mu suppression at C3 (contralateral), beta boost       |
| PREPARE  | Moderate mu, transitional activity                     |

---

## Real EEG Hardware Integration

To use real EEG hardware, replace the `eeg_simulation_thread()` function with:

```python
# Example: BrainFlow universal EEG driver
from brainflow.board_shim import BoardShim, BrainFlowInputParams

def eeg_hardware_thread(board_id, serial_port, ...):
    params = BrainFlowInputParams()
    params.serial_port = serial_port
    board = BoardShim(board_id, params)
    board.prepare_session()
    board.start_stream()
    # ... collect samples and push to eeg_buffer
```

Supported hardware via BrainFlow:
- OpenBCI (Cyton, Ganglion, Daisy)
- Neurosity Crown
- Muse (1, 2, S)
- g.tec g.USB Amp
- Emotiv EPOC / Insight

---

## Controls

| Key | Action                    |
|-----|---------------------------|
| Q   | Quit experiment safely    |
| ESC | Exit (on input screen)    |

---

## Configuration

Edit `CONFIG` at the top of `mi_eeg_system.py`:

```python
CONFIG = {
    "sampling_rate": 256,      # Hz
    "n_channels":    4,
    "n_trials":      20,
    "timing": {
        "rest":        7,      # seconds
        "preparation": 3,
        "imagery":     15,
        "inter_trial": 10,
    },
    "fps":            60,      # display FPS
    "animation_fps":  18,      # hand animation FPS
    "assets_dir":    "assets",
}
```
