"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          BCI MAZE NAVIGATION GAME — Real-Time EEG-Controlled                ║
║          Python + Pygame + Scikit-learn / NumPy / SciPy                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

OVERVIEW:
  A closed-loop Brain–Computer Interface (BCI) maze game.
  EEG motor imagery (LEFT vs RIGHT) drives player movement through a 2D maze.
  Supports:
    • Simulated EEG (sine + noise, for demo/testing)
    • Placeholder hook for real EEG hardware
    • Scikit-learn .pkl model  OR  a lightweight built-in fallback model
    • Free Mode (no ground truth) and Evaluation Mode (cued trials)
    • Prediction smoothing via majority vote
    • Full accuracy / confusion-matrix tracking
    • CSV export of every trial

USAGE:
  pip install pygame numpy scipy scikit-learn
  python bci_maze_game.py

KEYBOARD SHORTCUTS (in-game):
  R   – Restart maze
  Q   – Quit and save CSV results
  ESC – Same as Q
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import csv
import time
import random
import pickle
import datetime
import collections
import math

import numpy as np
import pygame
from scipy.signal import butter, sosfilt

# Optional: scikit-learn (used for the built-in fallback model)
try:
    from sklearn.dummy import DummyClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  (edit freely)
# ─────────────────────────────────────────────────────────────────────────────
CFG = {
    # EEG
    "sample_rate":       256,          # Hz
    "n_channels":        8,            # EEG channels
    "window_sec":        2.0,          # analysis window length (seconds)
    "window_step_sec":   0.5,          # step between windows (seconds)
    "bandpass":          (8, 30),      # mu + beta band (Hz)

    # Model
    "model_path":        "model.pkl",  # set to None to use built-in demo model

    # Prediction smoothing
    "smooth_n":          5,            # majority-vote window size

    # Movement
    "auto_forward":      True,         # player advances forward automatically
    "forward_interval":  1.5,          # seconds between automatic steps
    "move_on_predict":   True,         # move every prediction cycle

    # Game / display
    "cell_size":         48,           # pixels per maze cell
    "fps":               30,
    "sidebar_w":         280,          # info panel width

    # Maze
    "maze_rows":         13,
    "maze_cols":         19,

    # Evaluation mode
    "cue_duration":      3.0,          # seconds the directional cue is shown

    # Output
    "output_dir":        "bci_results",
}

# ─────────────────────────────────────────────────────────────────────────────
# COLOURS  (neon-on-dark palette for maximum sci-fi flair)
# ─────────────────────────────────────────────────────────────────────────────
C = {
    "bg":         (10,  12,  20),
    "wall":       (20,  25,  50),
    "wall_edge":  (40,  55, 110),
    "path":       (18,  22,  38),
    "player":     (0,  230, 200),
    "player_glow":(0,  180, 160),
    "goal":       (255, 200,   0),
    "goal_glow":  (180, 140,   0),
    "sidebar":    (14,  16,  28),
    "text":       (200, 210, 240),
    "text_dim":   (90, 100, 130),
    "accent":     (0,  200, 255),
    "left_col":   (80, 160, 255),
    "right_col":  (255, 100, 120),
    "correct":    (80, 230, 120),
    "wrong":      (255,  80,  80),
    "cue_left":   (60, 140, 255),
    "cue_right":  (255,  80, 100),
    "hud_bg":     (20,  24,  44),
    "bar_bg":     (30,  34,  58),
    "bar_fill":   (0,  200, 140),
    "grid_line":  (25,  30,  55),
}

FONT_MONO = None   # loaded in main()

# ─────────────────────────────────────────────────────────────────────────────
# ── 1. EEG SIMULATION & ACQUISITION ──────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class EEGBuffer:
    """Circular ring buffer storing multichannel EEG samples."""

    def __init__(self, n_channels: int, capacity: int):
        self.n_channels = n_channels
        self.capacity   = capacity
        self.data       = np.zeros((n_channels, capacity), dtype=np.float32)
        self._ptr       = 0
        self._filled    = 0

    def push(self, samples: np.ndarray):
        """samples: (n_channels, n_samples)"""
        n = samples.shape[1]
        for i in range(n):
            self.data[:, self._ptr] = samples[:, i]
            self._ptr = (self._ptr + 1) % self.capacity
        self._filled = min(self._filled + n, self.capacity)

    def get_window(self) -> np.ndarray:
        """Returns (n_channels, capacity) with oldest sample first."""
        idx = [(self._ptr + i) % self.capacity for i in range(self.capacity)]
        return self.data[:, idx]

    @property
    def ready(self) -> bool:
        return self._filled >= self.capacity


def simulate_eeg(n_channels: int, n_samples: int, sr: int,
                 label_hint: int = 0) -> np.ndarray:
    """
    Generate plausible mu-rhythm EEG for demo purposes.
    label_hint 0 = LEFT  (mu suppression left hemisphere)
               1 = RIGHT (mu suppression right hemisphere)
    Returns (n_channels, n_samples) float32.
    """
    t = np.linspace(0, n_samples / sr, n_samples)
    data = np.zeros((n_channels, n_samples), dtype=np.float32)

    for ch in range(n_channels):
        # Base broadband noise
        sig = np.random.randn(n_samples).astype(np.float32) * 5.0
        # Mu oscillation (10 Hz) with hemisphere-dependent suppression
        mu_amp = 3.0
        if label_hint == 0 and ch < n_channels // 2:   # left-hand → right hemi mu down
            mu_amp *= 0.4
        elif label_hint == 1 and ch >= n_channels // 2: # right-hand → left hemi mu down
            mu_amp *= 0.4
        sig += mu_amp * np.sin(2 * np.pi * 10 * t + random.uniform(0, 2*math.pi))
        # Beta (20 Hz)
        sig += 1.5 * np.sin(2 * np.pi * 20 * t + random.uniform(0, 2*math.pi))
        data[ch] = sig.astype(np.float32)

    return data


def get_live_eeg(n_channels: int, n_samples: int) -> np.ndarray:
    """
    ── PLACEHOLDER FOR REAL EEG HARDWARE ──
    Replace this function body with your acquisition SDK call.
    E.g. brainflow, pylsl, mne-realtime, etc.

    Should return: np.ndarray of shape (n_channels, n_samples)
    """
    # Example stub — replace with: board.get_board_data(n_samples)
    raise NotImplementedError(
        "Real EEG not connected. Using simulation mode instead."
    )

# ─────────────────────────────────────────────────────────────────────────────
# ── 2. SIGNAL PREPROCESSING ───────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def bandpass_filter(data: np.ndarray, lo: float, hi: float,
                    sr: int, order: int = 4) -> np.ndarray:
    """Butterworth bandpass over axis=1 (samples). Returns same shape."""
    nyq = sr / 2.0
    sos = butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")
    return sosfilt(sos, data, axis=1).astype(np.float32)


def preprocess(raw: np.ndarray, sr: int, band=(8, 30)) -> np.ndarray:
    """Bandpass → common-average-reference."""
    filtered = bandpass_filter(raw, band[0], band[1], sr)
    car = filtered - filtered.mean(axis=0, keepdims=True)
    return car

# ─────────────────────────────────────────────────────────────────────────────
# ── 3. FEATURE EXTRACTION ─────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def band_power(data: np.ndarray, sr: int, lo: float, hi: float) -> np.ndarray:
    """Log band power per channel via Welch-style DFT. Returns (n_ch,)."""
    freqs  = np.fft.rfftfreq(data.shape[1], 1.0 / sr)
    fft_sq = np.abs(np.fft.rfft(data, axis=1)) ** 2
    mask   = (freqs >= lo) & (freqs <= hi)
    power  = fft_sq[:, mask].mean(axis=1)
    return np.log1p(power).astype(np.float32)


def extract_features(buffer: np.ndarray, sr: int) -> np.ndarray:
    """
    Modular feature extraction.
    Extend this function with more sophisticated features (CSP, PLV, etc.)

    Input : (n_channels, n_samples)
    Output: 1-D feature vector (float32)
    """
    ch, _ = buffer.shape

    # Statistical features
    mean_feat = buffer.mean(axis=1)
    var_feat  = buffer.var(axis=1)
    kurt_feat = (((buffer - mean_feat[:, None]) ** 4).mean(axis=1)
                 / (var_feat ** 2 + 1e-6))

    # Band power features
    mu_power   = band_power(buffer, sr, 8,  13)
    beta_power = band_power(buffer, sr, 13, 30)

    # Hemisphere asymmetry (right - left)
    half = ch // 2
    mu_asym = mu_power[half:].mean() - mu_power[:half].mean()

    feat = np.concatenate([
        mean_feat, var_feat, kurt_feat,
        mu_power, beta_power,
        [mu_asym]
    ]).astype(np.float32)

    return feat

# ─────────────────────────────────────────────────────────────────────────────
# ── 4. MODEL LOADING & PREDICTION ─────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class BuiltinDemoModel:
    """
    Demo model: uses mu-band hemispheric asymmetry heuristic.
    Replace or wrap with your own trained model.
    LEFT  (0) : right hemisphere mu suppression  → asymmetry < 0
    RIGHT (1) : left  hemisphere mu suppression  → asymmetry > 0
    """
    classes_ = [0, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        # last feature is mu_asym
        preds = (X[:, -1] > 0).astype(int)
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        asym   = X[:, -1]
        prob_r = 1 / (1 + np.exp(-asym))          # sigmoid
        return np.column_stack([1 - prob_r, prob_r])


def load_model(path: str):
    """
    Load a scikit-learn .pkl model.
    Falls back to BuiltinDemoModel if file not found.
    """
    if path and os.path.isfile(path):
        with open(path, "rb") as f:
            model = pickle.load(f)
        print(f"[MODEL] Loaded from {path}")
        return model
    print(f"[MODEL] '{path}' not found → using built-in demo heuristic model.")
    return BuiltinDemoModel()


def predict_direction(model, features: np.ndarray):
    """
    Returns (label: int, confidence: float).
    label: 0 = LEFT, 1 = RIGHT
    """
    X = features.reshape(1, -1)
    label = int(model.predict(X)[0])
    if hasattr(model, "predict_proba"):
        proba      = model.predict_proba(X)[0]
        confidence = float(proba[label])
    else:
        confidence = 1.0
    return label, confidence


def smooth_prediction(history: collections.deque, new_label: int,
                      new_conf: float, n: int = 5):
    """
    Majority vote over last n predictions.
    Returns final smoothed label.
    """
    history.append((new_label, new_conf))
    while len(history) > n:
        history.popleft()
    counts = collections.Counter(lbl for lbl, _ in history)
    return counts.most_common(1)[0][0]

# ─────────────────────────────────────────────────────────────────────────────
# ── 5. MAZE GENERATION ────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def generate_maze(rows: int, cols: int) -> np.ndarray:
    """
    Recursive backtracker DFS maze.
    Returns 2D array: 1 = wall, 0 = path.
    Rows and cols are forced to be odd for proper cell/wall spacing.
    """
    rows = rows if rows % 2 == 1 else rows + 1
    cols = cols if cols % 2 == 1 else cols + 1
    grid = np.ones((rows, cols), dtype=np.int8)

    def carve(r, c):
        grid[r][c] = 0
        directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
        random.shuffle(directions)
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                grid[r + dr//2][c + dc//2] = 0
                carve(nr, nc)

    carve(1, 1)
    # Ensure border is wall
    grid[0, :] = 1;  grid[-1, :] = 1
    grid[:, 0] = 1;  grid[:, -1] = 1
    return grid


def find_path_cells(maze: np.ndarray):
    """Return list of (row,col) path cells."""
    return [(r, c)
            for r in range(maze.shape[0])
            for c in range(maze.shape[1])
            if maze[r, c] == 0]

# ─────────────────────────────────────────────────────────────────────────────
# ── 6. PLAYER & MOVEMENT ──────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

DIR_VECTORS = {
    "N": (-1,  0),
    "S": ( 1,  0),
    "E": ( 0,  1),
    "W": ( 0, -1),
}
TURN_LEFT  = {"N": "W", "W": "S", "S": "E", "E": "N"}
TURN_RIGHT = {"N": "E", "E": "S", "S": "W", "W": "N"}


class Player:
    def __init__(self, row: int, col: int, facing: str = "E"):
        self.row    = row
        self.col    = col
        self.facing = facing

        # Smooth pixel interpolation
        self.px     = float(col)  # pixel position in cell units
        self.py     = float(row)
        self.speed  = 8.0         # cells per second

    def turn_left(self):
        self.facing = TURN_LEFT[self.facing]

    def turn_right(self):
        self.facing = TURN_RIGHT[self.facing]

    def try_move_forward(self, maze: np.ndarray) -> bool:
        dr, dc = DIR_VECTORS[self.facing]
        nr, nc = self.row + dr, self.col + dc
        if (0 <= nr < maze.shape[0] and
                0 <= nc < maze.shape[1] and
                maze[nr, nc] == 0):
            self.row, self.col = nr, nc
            return True
        return False

    def update_pixel_pos(self, dt: float):
        """Interpolate pixel pos toward logical pos for smooth rendering."""
        tx, ty = float(self.col), float(self.row)
        self.px += (tx - self.px) * min(1.0, self.speed * dt)
        self.py += (ty - self.py) * min(1.0, self.speed * dt)

# ─────────────────────────────────────────────────────────────────────────────
# ── 7. ACCURACY TRACKING ──────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class AccuracyTracker:
    """Confusion matrix + per-step trial log."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.total    = 0
        self.correct  = 0
        # confusion[true][pred]
        self.conf_mat = np.zeros((2, 2), dtype=int)
        self.log: list[dict] = []   # trial records

    def record(self, true_label: int, pred_label: int,
               confidence: float, timestamp: float):
        is_correct = int(true_label == pred_label)
        self.total    += 1
        self.correct  += is_correct
        self.conf_mat[true_label][pred_label] += 1
        self.log.append({
            "timestamp":  round(timestamp, 3),
            "true":       "LEFT" if true_label == 0 else "RIGHT",
            "predicted":  "LEFT" if pred_label == 0 else "RIGHT",
            "correct":    is_correct,
            "confidence": round(confidence, 4),
        })

    @property
    def accuracy(self) -> float:
        return self.correct / max(1, self.total)

    def record_free(self, pred_label: int, confidence: float, timestamp: float):
        """Free mode: no ground truth, just log the prediction."""
        self.log.append({
            "timestamp":  round(timestamp, 3),
            "true":       "N/A",
            "predicted":  "LEFT" if pred_label == 0 else "RIGHT",
            "correct":    "N/A",
            "confidence": round(confidence, 4),
        })


def save_results(tracker: AccuracyTracker, subject_id: str,
                 mode: str, output_dir: str):
    """Save trial log to timestamped CSV."""
    os.makedirs(output_dir, exist_ok=True)
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"{subject_id}_{mode}_{ts}.csv")
    if not tracker.log:
        print("[SAVE] No data to save.")
        return
    keys = tracker.log[0].keys()
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(tracker.log)
    print(f"[SAVE] Results saved → {path}")
    return path

# ─────────────────────────────────────────────────────────────────────────────
# ── 8. RENDERING HELPERS ──────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def draw_text(surf, text, x, y, font, color=None, anchor="topleft", shadow=False):
    color = color or C["text"]
    img   = font.render(text, True, color)
    rect  = img.get_rect(**{anchor: (x, y)})
    if shadow:
        sh = font.render(text, True, (0, 0, 0))
        surf.blit(sh, rect.move(1, 1))
    surf.blit(img, rect)
    return rect


def draw_maze(surf, maze: np.ndarray, cell: int, offset_x: int, offset_y: int,
              player: Player, goal: tuple, pulse: float):
    rows, cols = maze.shape

    # Draw each cell
    for r in range(rows):
        for c in range(cols):
            x = offset_x + c * cell
            y = offset_y + r * cell
            if maze[r, c] == 1:
                pygame.draw.rect(surf, C["wall"], (x, y, cell, cell))
                # subtle inner edge highlight
                pygame.draw.rect(surf, C["wall_edge"],
                                 (x+1, y+1, cell-2, cell-2), 1)
            else:
                pygame.draw.rect(surf, C["path"], (x, y, cell, cell))

    # Grid lines (subtle)
    for r in range(rows + 1):
        pygame.draw.line(surf, C["grid_line"],
                         (offset_x, offset_y + r * cell),
                         (offset_x + cols * cell, offset_y + r * cell))
    for c in range(cols + 1):
        pygame.draw.line(surf, C["grid_line"],
                         (offset_x + c * cell, offset_y),
                         (offset_x + c * cell, offset_y + rows * cell))

    # Goal glow + icon
    gr, gc = goal
    gx = offset_x + gc * cell + cell // 2
    gy = offset_y + gr * cell + cell // 2
    glow_r = int(cell * 0.55 + 5 * math.sin(pulse * 2))
    for alpha, rad in [(40, glow_r + 10), (80, glow_r), (160, glow_r - 6)]:
        glow_surf = pygame.Surface((rad*2, rad*2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*C["goal_glow"], alpha), (rad, rad), rad)
        surf.blit(glow_surf, (gx - rad, gy - rad))
    pygame.draw.circle(surf, C["goal"], (gx, gy), cell // 3)
    # star symbol
    _draw_star(surf, gx, gy, cell // 4, cell // 7, 5, C["bg"])

    # Player
    px_screen = offset_x + int(player.px * cell + cell / 2)
    py_screen = offset_y + int(player.py * cell + cell / 2)
    pr        = int(cell * 0.38)

    # Player glow
    for alpha, rad in [(30, pr + 10), (70, pr + 5)]:
        g2 = pygame.Surface((rad*2, rad*2), pygame.SRCALPHA)
        pygame.draw.circle(g2, (*C["player_glow"], alpha), (rad, rad), rad)
        surf.blit(g2, (px_screen - rad, py_screen - rad))

    pygame.draw.circle(surf, C["player"], (px_screen, py_screen), pr)

    # Direction arrow
    dr, dc = DIR_VECTORS[player.facing]
    ax = px_screen + int(dc * pr * 0.6)
    ay = py_screen + int(dr * pr * 0.6)
    arrow_surf = pygame.Surface((pr*2, pr*2), pygame.SRCALPHA)
    tip_x = pr + int(dc * pr * 0.7)
    tip_y = pr + int(dr * pr * 0.7)
    pygame.draw.circle(arrow_surf, C["bg"], (tip_x, tip_y), pr // 3)
    surf.blit(arrow_surf, (px_screen - pr, py_screen - pr))


def _draw_star(surf, cx, cy, r_outer, r_inner, n_points, color):
    """Draw an n-point star."""
    pts = []
    for i in range(2 * n_points):
        angle = math.pi / n_points * i - math.pi / 2
        r     = r_outer if i % 2 == 0 else r_inner
        pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    if len(pts) >= 3:
        pygame.draw.polygon(surf, color, pts)


def draw_sidebar(surf, sidebar_x: int, w: int, h: int,
                 subject_id: str, mode: str,
                 last_pred: int, confidence: float,
                 smoothed: int,
                 tracker: AccuracyTracker,
                 step_count: int,
                 cue_label: int,   # -1 = none
                 cue_remaining: float,
                 font_lg, font_md, font_sm, font_xs,
                 pulse: float):

    # Background panel
    pygame.draw.rect(surf, C["sidebar"], (sidebar_x, 0, w, h))
    pygame.draw.line(surf, C["accent"], (sidebar_x, 0), (sidebar_x, h), 2)

    x0 = sidebar_x + 16
    y  = 18

    # ── Title ──
    draw_text(surf, "BCI MAZE", x0, y, font_lg, C["accent"], shadow=True)
    y += 38
    draw_text(surf, "NAVIGATOR", x0 + 4, y, font_md, C["text_dim"])
    y += 30

    pygame.draw.line(surf, C["wall_edge"], (sidebar_x+8, y), (sidebar_x+w-8, y))
    y += 12

    # ── Subject / Mode ──
    draw_text(surf, f"Subject : {subject_id}", x0, y, font_sm)
    y += 22
    mode_color = C["correct"] if mode == "FREE" else C["goal"]
    draw_text(surf, f"Mode    : {mode}", x0, y, font_sm, mode_color)
    y += 22
    draw_text(surf, f"Steps   : {step_count}", x0, y, font_sm)
    y += 30

    pygame.draw.line(surf, C["wall_edge"], (sidebar_x+8, y), (sidebar_x+w-8, y))
    y += 12

    # ── Prediction display ──
    draw_text(surf, "RAW PREDICTION", x0, y, font_xs, C["text_dim"])
    y += 18
    lbl_text  = "◀  LEFT" if last_pred == 0 else "RIGHT  ▶"
    lbl_color = C["left_col"] if last_pred == 0 else C["right_col"]
    draw_text(surf, lbl_text, x0, y, font_md, lbl_color, shadow=True)
    y += 28

    # Confidence bar
    draw_text(surf, f"Confidence: {confidence*100:.1f}%", x0, y, font_xs, C["text_dim"])
    y += 16
    bar_w  = w - 32
    pygame.draw.rect(surf, C["bar_bg"], (x0, y, bar_w, 10), border_radius=5)
    fill_w = int(bar_w * confidence)
    fill_c = C["correct"] if confidence > 0.7 else (C["goal"] if confidence > 0.5 else C["wrong"])
    pygame.draw.rect(surf, fill_c, (x0, y, fill_w, 10), border_radius=5)
    y += 22

    # Smoothed prediction
    draw_text(surf, "SMOOTHED (VOTED)", x0, y, font_xs, C["text_dim"])
    y += 18
    slbl  = "◀  LEFT" if smoothed == 0 else "RIGHT  ▶"
    scol  = C["left_col"] if smoothed == 0 else C["right_col"]
    # pulsing border box
    bw = w - 32
    bh = 36
    bx = x0
    by = y
    # glow border
    border_col = (*scol, int(80 + 60 * abs(math.sin(pulse * 3))))
    border_s   = pygame.Surface((bw, bh), pygame.SRCALPHA)
    pygame.draw.rect(border_s, (*scol, 30), (0, 0, bw, bh), border_radius=6)
    pygame.draw.rect(border_s, (*scol, 120), (0, 0, bw, bh), 2, border_radius=6)
    surf.blit(border_s, (bx, by))
    draw_text(surf, slbl, bx + bw//2, by + bh//2, font_md, scol, anchor="center")
    y += bh + 14

    pygame.draw.line(surf, C["wall_edge"], (sidebar_x+8, y), (sidebar_x+w-8, y))
    y += 12

    # ── Accuracy ──
    if mode == "EVAL":
        acc_pct = tracker.accuracy * 100
        draw_text(surf, "ACCURACY", x0, y, font_xs, C["text_dim"])
        y += 18
        acc_col  = C["correct"] if acc_pct >= 70 else (C["goal"] if acc_pct >= 50 else C["wrong"])
        draw_text(surf, f"{acc_pct:.1f}%", x0, y, font_lg, acc_col, shadow=True)
        y += 40

        draw_text(surf, f"Trials : {tracker.total}", x0, y, font_sm)
        y += 20
        draw_text(surf, f"Correct: {tracker.correct}", x0, y, font_sm, C["correct"])
        y += 20
        draw_text(surf, f"Wrong  : {tracker.total - tracker.correct}", x0, y, font_sm, C["wrong"])
        y += 28

        # Mini confusion matrix
        cm = tracker.conf_mat
        draw_text(surf, "CONFUSION MATRIX", x0, y, font_xs, C["text_dim"])
        y += 18
        labels = ["L", "R"]
        cell_s = 32
        for ti, tl in enumerate(labels):
            for pi, pl in enumerate(labels):
                bx_ = x0 + pi * (cell_s + 4)
                by_ = y  + ti * (cell_s + 4)
                val = cm[ti][pi]
                bg  = C["correct"] if ti == pi else C["wrong"]
                alpha_v = min(255, 40 + val * 30)
                cs  = pygame.Surface((cell_s, cell_s), pygame.SRCALPHA)
                pygame.draw.rect(cs, (*bg, alpha_v), (0, 0, cell_s, cell_s), border_radius=4)
                pygame.draw.rect(cs, (*bg, 180), (0, 0, cell_s, cell_s), 1, border_radius=4)
                surf.blit(cs, (bx_, by_))
                draw_text(surf, str(val), bx_ + cell_s//2, by_ + cell_s//2,
                          font_sm, C["text"], anchor="center")
        draw_text(surf, "T\\P", x0 + 2*(cell_s+4)+6, y, font_xs, C["text_dim"])
        for i, l in enumerate(labels):
            draw_text(surf, l, x0 + i*(cell_s+4)+cell_s//2,
                      y - 14, font_xs, C["text_dim"], anchor="center")
            draw_text(surf, l, x0 - 14, y + i*(cell_s+4)+cell_s//2,
                      font_xs, C["text_dim"], anchor="center")
        y += 2 * (cell_s + 4) + 16
    else:
        draw_text(surf, "FREE MODE", x0, y, font_xs, C["text_dim"])
        y += 18
        draw_text(surf, f"Moves : {tracker.total}", x0, y, font_sm)
        y += 30

    pygame.draw.line(surf, C["wall_edge"], (sidebar_x+8, y), (sidebar_x+w-8, y))
    y += 12

    # ── Evaluation cue ──
    if mode == "EVAL" and cue_label >= 0:
        cue_text  = "◀  THINK LEFT" if cue_label == 0 else "THINK RIGHT  ▶"
        cue_color = C["cue_left"] if cue_label == 0 else C["cue_right"]
        draw_text(surf, "CUE", x0, y, font_xs, C["text_dim"])
        y += 18
        # Animated cue box
        bw = w - 32
        bh = 44
        pulse_alpha = int(100 + 80 * abs(math.sin(pulse * 4)))
        cs = pygame.Surface((bw, bh), pygame.SRCALPHA)
        pygame.draw.rect(cs, (*cue_color, 40), (0, 0, bw, bh), border_radius=8)
        pygame.draw.rect(cs, (*cue_color, pulse_alpha), (0, 0, bw, bh), 3, border_radius=8)
        surf.blit(cs, (x0, y))
        draw_text(surf, cue_text, x0 + bw//2, y + bh//2,
                  font_md, cue_color, anchor="center", shadow=True)
        y += bh + 8
        draw_text(surf, f"  {cue_remaining:.1f}s remaining", x0, y, font_xs, C["text_dim"])
        y += 22

    pygame.draw.line(surf, C["wall_edge"], (sidebar_x+8, y), (sidebar_x+w-8, y))
    y += 12

    # ── Controls ──
    draw_text(surf, "CONTROLS", x0, y, font_xs, C["text_dim"])
    y += 18
    for key, action in [("R", "Restart"), ("Q/ESC", "Quit & Save"),
                         ("←/→ arrows", "Manual override")]:
        draw_text(surf, f"[{key}]", x0, y, font_xs, C["accent"])
        draw_text(surf, action, x0 + 70, y, font_xs, C["text_dim"])
        y += 16


def draw_win_screen(surf, rect, font_lg, font_md, subject_id, steps, accuracy_pct, mode):
    ow, oh = rect.width, rect.height
    overlay = pygame.Surface((ow, oh), pygame.SRCALPHA)
    overlay.fill((5, 8, 20, 200))
    surf.blit(overlay, rect.topleft)
    cx, cy = rect.centerx, rect.centery
    draw_text(surf, "MAZE SOLVED!", cx, cy - 60, font_lg, C["goal"], "center", shadow=True)
    draw_text(surf, f"Subject: {subject_id}", cx, cy - 10, font_md, C["text"], "center")
    draw_text(surf, f"Steps : {steps}", cx, cy + 25, font_md, C["text"], "center")
    if mode == "EVAL":
        draw_text(surf, f"Accuracy: {accuracy_pct:.1f}%", cx, cy + 60, font_md,
                  C["correct"], "center")
    draw_text(surf, "Press  R  to restart", cx, cy + 100, font_md, C["accent"], "center")


def draw_eeg_waveform(surf, data: np.ndarray, rect: pygame.Rect,
                      font_xs, sr: int, pulse: float):
    """Draw a mini EEG waveform strip for one channel."""
    x, y, w, h = rect.x, rect.y, rect.width, rect.height
    pygame.draw.rect(surf, C["hud_bg"], rect, border_radius=4)
    pygame.draw.rect(surf, C["wall_edge"], rect, 1, border_radius=4)

    if data is None or data.shape[1] == 0:
        return

    sig    = data[0, :]                    # channel 0
    sig_n  = sig[-w:] if len(sig) > w else sig
    n      = len(sig_n)
    mi, ma = sig_n.min(), sig_n.max()
    rng    = max(ma - mi, 1e-4)

    pts = []
    for i in range(n):
        px_ = x + int(i / n * w)
        py_ = y + h - int((sig_n[i] - mi) / rng * (h - 4)) - 2
        pts.append((px_, py_))

    if len(pts) >= 2:
        pygame.draw.lines(surf, C["accent"], False, pts, 1)

    draw_text(surf, "EEG ch0", x + 4, y + 2, font_xs, C["text_dim"])

# ─────────────────────────────────────────────────────────────────────────────
# ── 9. MAIN GAME CLASS ────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class BCIMazeGame:
    def __init__(self, subject_id: str, mode: str):
        self.subject_id = subject_id
        self.mode       = mode.upper()   # "FREE" or "EVAL"
        self.cfg        = CFG

        # ── Pygame setup ──
        pygame.init()
        pygame.display.set_caption("BCI Maze Navigation System")

        sr   = CFG["sample_rate"]
        rows = CFG["maze_rows"]
        cols = CFG["maze_cols"]
        cell = CFG["cell_size"]
        sw_  = CFG["sidebar_w"]

        self.maze_px_w = cols * cell
        self.maze_px_h = rows * cell
        self.win_w     = self.maze_px_w + sw_
        self.win_h     = max(self.maze_px_h, 700)
        self.maze_off_x = 0
        self.maze_off_y = (self.win_h - self.maze_px_h) // 2

        self.screen = pygame.display.set_mode((self.win_w, self.win_h))
        self.clock  = pygame.time.Clock()

        # Fonts (monospace feel)
        self._init_fonts()

        # ── EEG / model ──
        self.sr     = sr
        win_samples = int(CFG["window_sec"] * sr)
        self.buffer = EEGBuffer(CFG["n_channels"], win_samples)
        self.model  = load_model(CFG["model_path"])
        self.step_samples = int(CFG["window_step_sec"] * sr)
        self._samples_since_predict = 0

        # ── Prediction state ──
        self.pred_history = collections.deque()
        self.last_pred    = 0
        self.smoothed     = 0
        self.confidence   = 0.5

        # ── Game state ──
        self.tracker   = AccuracyTracker()
        self.step_count = 0
        self.won        = False
        self.start_time = time.time()
        self.pulse      = 0.0            # animation timer

        # Forward auto-movement
        self._fwd_timer = 0.0

        # Evaluation cue
        self._cue_label     = -1
        self._cue_timer     = 0.0
        self._cue_interval  = 4.0       # seconds between cues
        self._cue_until_cue = self._cue_interval

        # Last raw EEG window for display
        self._last_eeg_window = None

        # Simulation label hint (toggles to add variety)
        self._sim_label = 0

        self._init_game()

    def _init_fonts(self):
        global FONT_MONO
        # Try system monospace fonts
        candidates = ["Courier New", "Courier", "monospace", "DejaVu Sans Mono"]
        found = None
        for name in candidates:
            try:
                pygame.font.SysFont(name, 12)
                found = name
                break
            except Exception:
                pass
        FONT_MONO = found

        self.font_lg = pygame.font.SysFont(FONT_MONO, 28, bold=True)
        self.font_md = pygame.font.SysFont(FONT_MONO, 20, bold=True)
        self.font_sm = pygame.font.SysFont(FONT_MONO, 16)
        self.font_xs = pygame.font.SysFont(FONT_MONO, 13)

    def _init_game(self):
        """(Re)initialise maze, player, and game state."""
        rows = CFG["maze_rows"]
        cols = CFG["maze_cols"]
        self.maze   = generate_maze(rows, cols)
        paths       = find_path_cells(self.maze)
        start, *rest = paths
        # Goal: try to pick far corner
        goal_candidates = [(r, c) for r, c in paths
                           if abs(r - start[0]) + abs(c - start[1]) > (rows + cols) // 2]
        goal = random.choice(goal_candidates) if goal_candidates else rest[-1]
        self.start  = start
        self.goal   = goal
        self.player = Player(start[0], start[1])
        self.won    = False
        self.step_count = 0
        self.tracker.reset()
        self._fwd_timer = 0.0
        self._cue_label = -1
        self._cue_timer = 0.0
        self._cue_until_cue = self._cue_interval

    # ─────────────────────────────────────────────────────────────────────────
    # EEG pipeline step
    # ─────────────────────────────────────────────────────────────────────────

    def _eeg_step(self, dt: float) -> bool:
        """
        Fetch EEG chunk, push to buffer, run predict pipeline
        when enough samples have accumulated.
        Returns True if a new prediction was made.
        """
        n_new = max(1, int(dt * self.sr))
        # Simulation: use sim_label hint for realism
        raw = simulate_eeg(CFG["n_channels"], n_new, self.sr,
                           label_hint=self._sim_label)
        self.buffer.push(raw)
        self._samples_since_predict += n_new

        if self._samples_since_predict < self.step_samples:
            return False
        self._samples_since_predict = 0

        if not self.buffer.ready:
            return False

        window  = self.buffer.get_window()
        cleaned = preprocess(window, self.sr, CFG["bandpass"])
        feats   = extract_features(cleaned, self.sr)
        label, conf = predict_direction(self.model, feats)

        self._last_eeg_window = cleaned
        self.last_pred  = label
        self.confidence = conf
        self.smoothed   = smooth_prediction(self.pred_history, label, conf,
                                             CFG["smooth_n"])
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Movement
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_prediction(self, label: int, manual: bool = False):
        """Turn player then step forward."""
        if self.won:
            return

        old_facing = self.player.facing
        if label == 0:
            self.player.turn_left()
        else:
            self.player.turn_right()

        # Try to move forward; if blocked, undo turn
        moved = self.player.try_move_forward(self.maze)
        if not moved:
            # Try without turning (go straight)
            self.player.facing = old_facing
            moved = self.player.try_move_forward(self.maze)

        if moved:
            self.step_count += 1

        # Evaluation mode accuracy tracking
        if self.mode == "EVAL" and not manual and self._cue_label >= 0:
            self.tracker.record(
                true_label  = self._cue_label,
                pred_label  = label,
                confidence  = self.confidence,
                timestamp   = time.time() - self.start_time,
            )
        elif not manual:
            self.tracker.record_free(label, self.confidence,
                                      time.time() - self.start_time)

        # Check win
        if (self.player.row, self.player.col) == self.goal:
            self.won = True

    # ─────────────────────────────────────────────────────────────────────────
    # Main loop
    # ─────────────────────────────────────────────────────────────────────────

    def run(self):
        running = True
        while running:
            dt     = self.clock.tick(CFG["fps"]) / 1000.0
            dt     = min(dt, 0.1)   # clamp large frames
            self.pulse += dt

            # ── Events ──
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_r:
                        self._init_game()
                    elif not self.won:
                        # Manual override keys
                        if event.key == pygame.K_LEFT:
                            self._apply_prediction(0, manual=True)
                        elif event.key == pygame.K_RIGHT:
                            self._apply_prediction(1, manual=True)

            if not self.won:
                # ── EEG pipeline ──
                new_pred = self._eeg_step(dt)

                # ── Auto-forward with BCI prediction ──
                self._fwd_timer += dt
                if CFG["move_on_predict"] and new_pred:
                    self._apply_prediction(self.smoothed)
                    self._fwd_timer = 0.0
                elif CFG["auto_forward"] and self._fwd_timer >= CFG["forward_interval"]:
                    self._apply_prediction(self.smoothed)
                    self._fwd_timer = 0.0

                # ── Evaluation cue timer ──
                if self.mode == "EVAL":
                    if self._cue_label >= 0:
                        self._cue_timer += dt
                        if self._cue_timer >= CFG["cue_duration"]:
                            self._cue_label     = -1
                            self._cue_timer     = 0.0
                            self._cue_until_cue = self._cue_interval
                            # Alternate sim label to cue
                            self._sim_label = 1 - self._sim_label
                    else:
                        self._cue_until_cue -= dt
                        if self._cue_until_cue <= 0:
                            self._cue_label     = random.randint(0, 1)
                            self._cue_timer     = 0.0
                            self._sim_label     = self._cue_label  # guide sim

            # ── Smooth pixel movement ──
            self.player.update_pixel_pos(dt)

            # ── Render ──
            self._render()

        # ── Teardown ──
        save_results(self.tracker, self.subject_id,
                     self.mode, CFG["output_dir"])
        pygame.quit()

    # ─────────────────────────────────────────────────────────────────────────
    # Rendering
    # ─────────────────────────────────────────────────────────────────────────

    def _render(self):
        self.screen.fill(C["bg"])

        cell  = CFG["cell_size"]
        sw_   = CFG["sidebar_w"]
        ox    = self.maze_off_x
        oy    = self.maze_off_y
        sbar_x = self.win_w - sw_

        # ── Draw maze ──
        maze_rect = pygame.Rect(ox, oy, self.maze_px_w, self.maze_px_h)
        draw_maze(self.screen, self.maze, cell, ox, oy,
                  self.player, self.goal, self.pulse)

        # ── EEG waveform strip ──
        eeg_rect = pygame.Rect(ox + 4, 4, self.maze_px_w - 8, 40)
        draw_eeg_waveform(self.screen, self._last_eeg_window,
                          eeg_rect, self.font_xs, self.sr, self.pulse)

        # ── Sidebar ──
        cue_remaining = max(0.0, CFG["cue_duration"] - self._cue_timer)
        draw_sidebar(
            self.screen, sbar_x, sw_, self.win_h,
            self.subject_id, self.mode,
            self.last_pred, self.confidence, self.smoothed,
            self.tracker, self.step_count,
            self._cue_label, cue_remaining,
            self.font_lg, self.font_md, self.font_sm, self.font_xs,
            self.pulse,
        )

        # ── Win overlay ──
        if self.won:
            draw_win_screen(
                self.screen, maze_rect,
                self.font_lg, self.font_md,
                self.subject_id, self.step_count,
                self.tracker.accuracy * 100, self.mode,
            )

        # ── FPS ──
        fps_text = f"FPS {self.clock.get_fps():.0f}"
        draw_text(self.screen, fps_text,
                  self.win_w - sw_ - 6, self.win_h - 16,
                  self.font_xs, C["text_dim"], anchor="bottomright")

        pygame.display.flip()

# ─────────────────────────────────────────────────────────────────────────────
# ── 10. ENTRY POINT ───────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def get_user_input():
    """Simple terminal prompt for subject ID and game mode."""
    print("=" * 60)
    print("   BCI MAZE NAVIGATION SYSTEM")
    print("=" * 60)
    sid = input("  Enter Subject ID  [default: S01] : ").strip() or "S01"
    print()
    print("  Select Mode:")
    print("    1 → FREE  (no ground-truth cues)")
    print("    2 → EVAL  (directional cues + accuracy)")
    choice = input("  Enter 1 or 2      [default: 1]  : ").strip()
    mode   = "EVAL" if choice == "2" else "FREE"
    print()
    print(f"  Starting: subject={sid!r}  mode={mode}")
    print("=" * 60)
    return sid, mode


def main():
    subject_id, mode = get_user_input()
    game = BCIMazeGame(subject_id=subject_id, mode=mode)
    game.run()
    print("Session ended. Thank you!")


if __name__ == "__main__":
    main()
