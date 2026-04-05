import pygame
import numpy as np
import threading
import csv
import json
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import time
import logging
import math
from datetime import datetime
from collections import deque
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ══════════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("mi_eeg_experiment.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("MI_EEG")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — Edit these to customize the experiment
# ══════════════════════════════════════════════════════════════════════════════
CONFIG = {
    # EEG Hardware
    "sampling_rate": 256,          # Hz
    "n_channels": 4,
    "channel_names": ["C3", "C4", "Cz", "Fz"],

    # Trial Parameters
    "n_trials": 20,

    # Phase Durations (seconds)
    "timing": {
        "rest":        10,
        "preparation": 5,
        "imagery":     15,
        "inter_trial": 10,
    },

    # Display
    "fps":            60,
    "animation_fps":  18,          # Hand animation update rate

    # Assets folder (optional — procedural animation used if not found)
    "assets_dir": "assets",
}

# ── Labels ────────────────────────────────────────────────────────────────────
LABEL = {
    "REST":       0,
    "PREPARE":    1,
    "LEFT_MI":    2,
    "RIGHT_MI":   3,
    "INTER":      4,
}

# ── Color Palette (Dark BCI-Lab Aesthetic) ────────────────────────────────────
C = {
    "bg":           (8,   10,  22),
    "bg_rest":      (12,  38,  28),
    "bg_left":      (10,  25,  70),
    "bg_right":     (70,  22,  10),
    "bg_prep_l":    (15,  40,  90),
    "bg_prep_r":    (90,  32,  15),
    "bg_inter":     (20,  20,  40),

    "text_hi":      (225, 235, 255),
    "text_mid":     (160, 170, 200),
    "text_dim":     (90,  95,  120),

    "left_hand":    (60,  160, 255),
    "left_glow":    (30,  80,  180),
    "right_hand":   (255, 140,  50),
    "right_glow":   (180,  70,  20),

    "rest_circle":  (50,  180, 110),
    "rest_glow":    (20,   80,  50),

    "timer_bg":     (30,  30,  55),
    "timer_left":   (40,  140, 240),
    "timer_right":  (240, 120,  40),
    "timer_rest":   (50,  180, 100),
    "timer_prep":   (200, 200,  60),

    "white":        (255, 255, 255),
    "black":        (0,   0,   0),
    "overlay":      (0,   0,   0,  160),
}


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL STATE
# ══════════════════════════════════════════════════════════════════════════════
eeg_buffer    = deque()          # Thread-safe sample queue
buffer_lock   = threading.Lock()
eeg_running   = threading.Event()
quit_event    = threading.Event()

current_label = LABEL["REST"]    # Modified by main thread, read by EEG thread
current_ts    = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 1. FOLDER / FILE MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════
def create_folders(subject_id: str, session_date: str) -> Path:
    """Create EEG_Dataset/Subject_XX/YYYY-MM-DD/ directory tree."""
    base = Path("EEG_Dataset") / f"Subject_{subject_id}" / session_date
    base.mkdir(parents=True, exist_ok=True)
    logger.info(f"Dataset folder: {base}")
    return base


def save_trial_data(data: list, filepath: Path) -> None:
    """Write trial EEG samples to CSV."""
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "C3", "C4", "Cz", "Fz", "label", "label_name"])
        writer.writerows(data)
    logger.debug(f"Saved {len(data)} samples -> {filepath}")


def save_metadata(meta: dict, folder: Path) -> None:
    """Serialize experiment metadata to JSON."""
    path = folder / "metadata.json"
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Metadata saved -> {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. EEG SIMULATION (runs in background thread)
# ══════════════════════════════════════════════════════════════════════════════
def eeg_simulation_thread(sr: int, n_ch: int) -> None:
    """
    Simulates multi-channel EEG at `sr` Hz with physiologically plausible signals.
    Adds mu-rhythm suppression (8–12 Hz) during MI phases and alpha during rest.
    Writes (timestamp, ch0..chN, label) tuples into eeg_buffer.
    """
    interval = 1.0 / sr
    t        = 0.0
    freqs    = np.array([10.0, 10.2, 9.8, 10.5])   # Mu-band per channel
    alpha    = np.array([12.0, 11.5, 11.8, 12.2])   # Alpha per channel

    logger.info("EEG simulation thread started")
    eeg_running.set()

    while not quit_event.is_set():
        tick_start = time.perf_counter()

        lbl = current_label

        # ── Signal generation by phase ────────────────────────────────────────
        signals = np.zeros(n_ch)
        for ch in range(n_ch):
            noise = np.random.randn() * 2.0                        # Base noise (µV)

            if lbl == LABEL["REST"]:
                # Strong alpha + mu during rest
                signals[ch] = (
                    8.0 * math.sin(2 * math.pi * freqs[ch] * t)   # mu
                  + 5.0 * math.sin(2 * math.pi * alpha[ch] * t)   # alpha
                  + noise
                )
            elif lbl == LABEL["LEFT_MI"]:
                # Mu suppression contralateral (C4) during left MI
                mu_amp = 2.0 if ch == 1 else 6.0                   # C4 suppressed
                signals[ch] = (
                    mu_amp * math.sin(2 * math.pi * freqs[ch] * t)
                  + 3.0   * math.sin(2 * math.pi * 20.0 * t)      # beta
                  + noise * 1.5
                )
            elif lbl == LABEL["RIGHT_MI"]:
                # Mu suppression contralateral (C3) during right MI
                mu_amp = 2.0 if ch == 0 else 6.0                   # C3 suppressed
                signals[ch] = (
                    mu_amp * math.sin(2 * math.pi * freqs[ch] * t)
                  + 3.0   * math.sin(2 * math.pi * 20.0 * t)
                  + noise * 1.5
                )
            elif lbl == LABEL["PREPARE"]:
                signals[ch] = (
                    5.0 * math.sin(2 * math.pi * freqs[ch] * t)
                  + noise
                )
            else:
                signals[ch] = noise * 1.2

        sample = (t,) + tuple(round(s, 4) for s in signals) + (lbl,)
        with buffer_lock:
            eeg_buffer.append(sample)

        t += interval

        # Precise sleep to maintain sample rate
        elapsed = time.perf_counter() - tick_start
        sleep_t = interval - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)

    logger.info("EEG simulation thread stopped")


def collect_eeg_data(label_name: str) -> list:
    """Drain the EEG buffer and annotate with label name string."""
    with buffer_lock:
        samples = list(eeg_buffer)
        eeg_buffer.clear()
    annotated = [s + (label_name,) for s in samples]
    return annotated


# ══════════════════════════════════════════════════════════════════════════════
# 3. PYGAME DRAWING UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
def draw_glow_circle(surf: pygame.Surface, color: tuple, pos: tuple,
                     radius: int, alpha: int = 80) -> None:
    """Draw a soft radial glow around a point."""
    glow = pygame.Surface((radius * 4, radius * 4), pygame.SRCALPHA)
    for r in range(radius * 2, 0, -4):
        a = int(alpha * (r / (radius * 2)) ** 0.5)
        pygame.draw.circle(glow, (*color, a),
                           (radius * 2, radius * 2), r)
    surf.blit(glow, (pos[0] - radius * 2, pos[1] - radius * 2))


def draw_text_centered(surf: pygame.Surface, text: str, font: pygame.font.Font,
                        color: tuple, y: int, shadow: bool = True) -> None:
    """Render centered text with optional drop shadow."""
    if shadow:
        sh = font.render(text, True, (0, 0, 0))
        sr = sh.get_rect(centerx=surf.get_width() // 2, y=y + 3)
        surf.blit(sh, sr)
    tx = font.render(text, True, color)
    r  = tx.get_rect(centerx=surf.get_width() // 2, y=y)
    surf.blit(tx, r)


def draw_progress_bar(surf: pygame.Surface, x: int, y: int, w: int, h: int,
                       progress: float, color: tuple, bg: tuple = C["timer_bg"],
                       radius: int = 8) -> None:
    """Draw a rounded progress/timer bar."""
    pygame.draw.rect(surf, bg,    (x, y, w, h),          border_radius=radius)
    filled = int(w * max(0, min(1, progress)))
    if filled > 0:
        pygame.draw.rect(surf, color, (x, y, filled, h), border_radius=radius)
    pygame.draw.rect(surf, (*color, 100), (x, y, w, h),
                     width=2, border_radius=radius)


def fill_bg_gradient(surf: pygame.Surface, color_top: tuple,
                      color_bot: tuple) -> None:
    """Vertical gradient background fill."""
    h = surf.get_height()
    w = surf.get_width()
    for y in range(h):
        t   = y / h
        col = tuple(int(color_top[i] + (color_bot[i] - color_top[i]) * t)
                    for i in range(3))
        pygame.draw.line(surf, col, (0, y), (w, y))


# ══════════════════════════════════════════════════════════════════════════════
# 4. HAND ANIMATION ENGINE (Procedural — no external files required)
# ══════════════════════════════════════════════════════════════════════════════
def draw_hand(surf: pygame.Surface, cx: int, cy: int,
              t: float, side: str) -> None:
    """
    Draw a fully procedural animated hand silhouette.

    Args:
        surf:  Target surface
        cx/cy: Center position
        t:     Elapsed time (seconds) for animation phase
        side:  'left' or 'right'
    """
    direction = -1 if side == "left" else 1
    hand_col  = C["left_hand"]  if side == "left" else C["right_hand"]
    glow_col  = C["left_glow"]  if side == "left" else C["right_glow"]

    # ── Animation parameters ──────────────────────────────────────────────────
    # Wave cycle: 2 seconds per cycle
    cycle   = 2.0
    phase   = (t % cycle) / cycle                     # 0 -> 1 within each cycle
    wave    = math.sin(2 * math.pi * phase)           # -1 -> 1
    wave_sq = math.sin(2 * math.pi * phase) ** 2      # 0 -> 1, always positive

    # Lateral drift: hand sweeps in the direction of the side
    drift_x = direction * wave * 55
    # Vertical bob
    drift_y = -abs(wave) * 18

    # Rotation: slight tilt following the sweep
    rotation = math.radians(wave * 12 * direction)

    # Finger spread (opens during outward, closes during return)
    spread  = 0.5 + 0.5 * (wave * direction)          # 0–1

    # ── Draw glow / aura ──────────────────────────────────────────────────────
    glow_r = int(110 + wave_sq * 30)
    draw_glow_circle(surf, glow_col,
                     (cx + int(drift_x), cy + int(drift_y)),
                     glow_r, alpha=60)

    # ── Compute rotated points helper ─────────────────────────────────────────
    def rot(px, py, angle):
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        rx = px * cos_a - py * sin_a
        ry = px * sin_a + py * cos_a
        return (cx + int(drift_x + rx), cy + int(drift_y + ry))

    # ── Palm (rounded rectangle via polygon) ─────────────────────────────────
    pw, ph = 68, 72                  # palm width, height
    palm_pts = [
        rot(-pw//2 + 10, -ph//2 + 5,  rotation),
        rot( pw//2 - 10, -ph//2 + 5,  rotation),
        rot( pw//2,       ph//2 - 10, rotation),
        rot(-pw//2,       ph//2 - 10, rotation),
    ]
    pygame.draw.polygon(surf, hand_col, palm_pts)
    pygame.draw.polygon(surf, (*hand_col, 200), palm_pts, 2)

    # ── Fingers ───────────────────────────────────────────────────────────────
    # finger_defs: (base_x_offset, base_y, finger_angle_deg, length)
    finger_defs = [
        (-26, -ph//2 + 5,  -15,  52),  # index
        ( -8, -ph//2,        -5,  60),  # middle
        (  8, -ph//2,         5,  58),  # ring
        ( 24, -ph//2 + 5,   15,  50),  # pinky
    ]

    for i, (bx, by, fang, flen) in enumerate(finger_defs):
        # Spread fingers apart with animation
        spread_offset = direction * (i - 1.5) * spread * 8
        tip_ang = math.radians(fang) + rotation + math.radians(spread_offset * 0.3)
        tip_len = flen + int(spread * 8)

        base_pt = rot(bx, by, rotation)
        tip_pt  = (
            base_pt[0] + int(math.sin(tip_ang) * tip_len),
            base_pt[1] - int(math.cos(tip_ang) * tip_len),
        )

        fw = max(8, 14 - i * 1)       # finger width (tapers from index to pinky)
        # Draw finger as a thick anti-aliased line with rounded cap
        pygame.draw.line(surf, hand_col, base_pt, tip_pt, fw)
        pygame.draw.circle(surf, hand_col, tip_pt, fw // 2)
        # Knuckle highlight
        mid_pt = (
            (base_pt[0] + tip_pt[0]) // 2,
            (base_pt[1] + tip_pt[1]) // 2,
        )
        pygame.draw.circle(surf, (*[min(255, c + 40) for c in hand_col], 160),
                           mid_pt, fw // 2 - 1)

    # ── Thumb ─────────────────────────────────────────────────────────────────
    thumb_bx =  direction * (pw // 2 - 5)
    thumb_by  = -8
    thumb_ang = rotation + math.radians(direction * (55 + spread * 15))
    thumb_len = 40
    tb_base = rot(thumb_bx, thumb_by, rotation)
    tb_tip  = (
        tb_base[0] + int(math.cos(thumb_ang) * thumb_len * direction),
        tb_base[1] - int(math.sin(thumb_ang) * thumb_len * 0.6),
    )
    pygame.draw.line(surf, hand_col, tb_base, tb_tip, 14)
    pygame.draw.circle(surf, hand_col, tb_tip, 6)

    # ── Wrist label arrow ─────────────────────────────────────────────────────
    # Small directional arrow below palm indicating movement direction
    arrow_y  = cy + drift_y + ph // 2 + 20
    arrow_cx = cx + int(drift_x)
    arr_col  = (*hand_col, 200)
    for i in range(3):
        offset = direction * i * 16
        ax     = arrow_cx + offset
        pts    = [
            (ax,                  arrow_y + 10),
            (ax + direction * 14, arrow_y),
            (ax,                  arrow_y - 10),
        ]
        # Alpha-based fading arrowheads
        alpha_val = 255 - i * 70
        ar_surf = pygame.Surface((40, 30), pygame.SRCALPHA)
        local_pts = [(p[0] - ax + 20, p[1] - arrow_y + 15) for p in pts]
        pygame.draw.polygon(ar_surf, (*hand_col, alpha_val), local_pts)
        surf.blit(ar_surf, (ax - 20, arrow_y - 15))


# ══════════════════════════════════════════════════════════════════════════════
# 5. BREATHING CIRCLE (REST animation)
# ══════════════════════════════════════════════════════════════════════════════
def draw_breathing_circle(surf: pygame.Surface, cx: int, cy: int,
                           t: float) -> None:
    """Expanding/contracting circle with glow for REST phase."""
    # 4s inhale, 4s exhale cycle
    cycle   = 8.0
    phase   = (t % cycle) / cycle
    # Smooth ease: grows 0->1 in first half, shrinks 1->0 in second half
    if phase < 0.5:
        scale = math.sin(math.pi * phase)
    else:
        scale = math.sin(math.pi * phase)

    r_min, r_max = 55, 110
    radius = int(r_min + (r_max - r_min) * scale)

    # Outer glow rings (fade with radius)
    for gr in range(radius + 40, radius - 5, -6):
        a = max(0, int(50 * (1 - abs(gr - radius) / 45)))
        glow_s = pygame.Surface((gr * 2 + 4, gr * 2 + 4), pygame.SRCALPHA)
        pygame.draw.circle(glow_s, (*C["rest_circle"], a),
                           (gr + 2, gr + 2), gr)
        surf.blit(glow_s, (cx - gr - 2, cy - gr - 2))

    # Main circle
    pygame.draw.circle(surf, C["rest_circle"], (cx, cy), radius)

    # Inner highlight
    hi_r = max(10, int(radius * 0.45))
    hi_s = pygame.Surface((hi_r * 2, hi_r * 2), pygame.SRCALPHA)
    pygame.draw.circle(hi_s, (255, 255, 255, 40), (hi_r, hi_r), hi_r)
    surf.blit(hi_s, (cx - hi_r, cy - hi_r - hi_r // 3))


# ══════════════════════════════════════════════════════════════════════════════
# 6. HUD ELEMENTS — trial counter, phase label, timer bar
# ══════════════════════════════════════════════════════════════════════════════
def draw_hud(surf: pygame.Surface, fonts: dict,
             trial_num: int, total_trials: int,
             phase_name: str, elapsed: float, duration: float,
             timer_color: tuple) -> None:
    """Render persistent HUD overlay: trial counter + phase bar."""
    W = surf.get_width()
    H = surf.get_height()

    # ── Trial counter (top-left) ──────────────────────────────────────────────
    trial_text = f"Trial  {trial_num} / {total_trials}"
    t_surf = fonts["small"].render(trial_text, True, C["text_mid"])
    surf.blit(t_surf, (30, 20))

    # ── Phase label (top-right) ───────────────────────────────────────────────
    p_surf = fonts["small"].render(phase_name, True, timer_color)
    surf.blit(p_surf, (W - p_surf.get_width() - 30, 20))

    # ── Timer bar (bottom) ────────────────────────────────────────────────────
    bar_w   = int(W * 0.6)
    bar_h   = 16
    bar_x   = (W - bar_w) // 2
    bar_y   = H - 55
    progress = 1.0 - (elapsed / duration) if duration > 0 else 0
    draw_progress_bar(surf, bar_x, bar_y, bar_w, bar_h,
                      progress, timer_color)

    # Remaining seconds
    remain = max(0, duration - elapsed)
    rem_s  = fonts["small"].render(f"{remain:.1f}s", True, timer_color)
    surf.blit(rem_s, (bar_x + bar_w + 12, bar_y - 2))

    # ── EEG indicator (bottom-left) ───────────────────────────────────────────
    blink = int(time.time() * 2) % 2
    eeg_col = (80, 220, 80) if blink else (40, 140, 40)
    pygame.draw.circle(surf, eeg_col, (22, H - 28), 7)
    eeg_s = fonts["tiny"].render("EEG ●", True, eeg_col)
    surf.blit(eeg_s, (35, H - 37))


# ══════════════════════════════════════════════════════════════════════════════
# 7. PHASE RENDERING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def show_rest(screen: pygame.Surface, fonts: dict,
              duration: float, trial_num: int, total_trials: int,
              clock: pygame.time.Clock, phase_label: str = "REST") -> list:
    """
    Render the REST phase: breathing circle + 'REST' text.
    Returns EEG data collected during this phase.
    """
    global current_label
    current_label = LABEL["REST"]

    W, H = screen.get_size()
    cx, cy = W // 2, H // 2
    trial_data = []
    start = time.perf_counter()

    while True:
        elapsed = time.perf_counter() - start
        if elapsed >= duration:
            break

        # ── Event handling ────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_event.set()
                return trial_data
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                quit_event.set()
                return trial_data

        if quit_event.is_set():
            break

        # ── Background gradient ───────────────────────────────────────────────
        fill_bg_gradient(screen, C["bg_rest"], C["bg"])

        # ── Breathing circle ──────────────────────────────────────────────────
        draw_breathing_circle(screen, cx, cy - 40, elapsed)

        # ── Text ──────────────────────────────────────────────────────────────
        draw_text_centered(screen, "REST", fonts["large"], C["text_hi"], cy + 90)
        draw_text_centered(screen, "Breathe and relax",
                           fonts["medium"], C["text_mid"], cy + 155)

        # ── HUD ───────────────────────────────────────────────────────────────
        draw_hud(screen, fonts, trial_num, total_trials,
                 phase_label, elapsed, duration, C["timer_rest"])

        pygame.display.flip()
        clock.tick(CONFIG["fps"])

        # Collect EEG
        trial_data.extend(collect_eeg_data("REST"))

    return trial_data


def show_preparation(screen: pygame.Surface, fonts: dict,
                     duration: float, side: str,
                     trial_num: int, total_trials: int,
                     clock: pygame.time.Clock) -> list:
    """
    Render PREPARATION phase: countdown + 'GET READY' message.
    """
    global current_label
    current_label = LABEL["PREPARE"]

    W, H = screen.get_size()
    cx, cy = W // 2, H // 2

    bg_col = C["bg_prep_l"] if side == "left" else C["bg_prep_r"]
    bar_col = C["timer_left"] if side == "left" else C["timer_right"]
    side_str = "LEFT" if side == "left" else "RIGHT"
    hand_col = C["left_hand"] if side == "left" else C["right_hand"]

    trial_data = []
    start = time.perf_counter()

    while True:
        elapsed = time.perf_counter() - start
        if elapsed >= duration:
            break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_event.set()
                return trial_data
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                quit_event.set()
                return trial_data

        if quit_event.is_set():
            break

        fill_bg_gradient(screen, bg_col, C["bg"])

        # Countdown integer (3 -> 2 -> 1)
        countdown = max(1, int(math.ceil(duration - elapsed)))
        # Pulse scale on the countdown number
        pulse = 1.0 + 0.12 * math.sin(elapsed * math.pi * 2)

        # Large countdown with pulsing scale
        cd_font = pygame.font.SysFont("Times New Roman", int(120 * pulse), bold=True)
        cd_surf = cd_font.render(str(countdown), True, hand_col)
        cd_rect = cd_surf.get_rect(center=(cx, cy - 20))
        # Glow behind countdown
        draw_glow_circle(screen, hand_col, (cx, cy - 20), 90, alpha=50)
        screen.blit(cd_surf, cd_rect)

        # "GET READY" label
        draw_text_centered(screen,
                           f"GET READY  —  {side_str}",
                           fonts["large"], C["text_hi"], cy - 140)
        draw_text_centered(screen,
                           f"Prepare to imagine your {side_str.lower()} hand movement",
                           fonts["small"], C["text_mid"], cy + 80)

        draw_hud(screen, fonts, trial_num, total_trials,
                 f"PREPARE {side_str}", elapsed, duration, bar_col)

        pygame.display.flip()
        clock.tick(CONFIG["fps"])
        trial_data.extend(collect_eeg_data("PREPARE"))

    return trial_data


def show_animation_phase(screen: pygame.Surface, fonts: dict,
                         duration: float, side: str,
                         trial_num: int, total_trials: int,
                         clock: pygame.time.Clock) -> list:
    """
    Render the MI phase with looping hand animation.
    This is the core scientific phase — animation loops for full duration.
    """
    global current_label
    current_label = LABEL["LEFT_MI"] if side == "left" else LABEL["RIGHT_MI"]

    W, H = screen.get_size()
    cx, cy = W // 2, H // 2

    bg_col   = C["bg_left"]  if side == "left"  else C["bg_right"]
    bar_col  = C["timer_left"] if side == "left" else C["timer_right"]
    hand_col = C["left_hand"] if side == "left"  else C["right_hand"]
    side_str = "LEFT" if side == "left" else "RIGHT"
    lbl_str  = "LEFT_MI" if side == "left" else "RIGHT_MI"

    trial_data = []
    start = time.perf_counter()

    # Try to load GIF / frame images from assets (optional enhancement)
    frames = load_hand_frames(side)

    frame_idx     = 0
    frame_timer   = 0.0
    frame_interval = 1.0 / CONFIG["animation_fps"]

    # Particle system for background energy
    particles = [
        {
            "x": np.random.uniform(0, W),
            "y": np.random.uniform(0, H),
            "vx": np.random.uniform(-0.5, 0.5) * (1 if side == "right" else -1),
            "vy": np.random.uniform(-0.3, 0.3),
            "r":  np.random.uniform(2, 6),
            "alpha": np.random.uniform(30, 120),
        }
        for _ in range(35)
    ]

    prev_time = time.perf_counter()

    while True:
        now     = time.perf_counter()
        dt      = now - prev_time
        prev_time = now
        elapsed = now - start

        if elapsed >= duration:
            break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_event.set()
                return trial_data
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                quit_event.set()
                return trial_data

        if quit_event.is_set():
            break

        # ── Background ────────────────────────────────────────────────────────
        fill_bg_gradient(screen, bg_col, C["bg"])

        # ── Particle field (flowing energy) ───────────────────────────────────
        for p in particles:
            p["x"] += p["vx"] * 1.5
            p["y"] += p["vy"] * 1.2
            # Wrap
            if p["x"] < -10:  p["x"] = W + 10
            if p["x"] > W+10: p["x"] = -10
            if p["y"] < -10:  p["y"] = H + 10
            if p["y"] > H+10: p["y"] = -10

            ps = pygame.Surface((int(p["r"]) * 2 + 2,
                                  int(p["r"]) * 2 + 2), pygame.SRCALPHA)
            pygame.draw.circle(ps, (*hand_col, int(p["alpha"])),
                                (int(p["r"]) + 1, int(p["r"]) + 1), int(p["r"]))
            screen.blit(ps, (int(p["x"] - p["r"]), int(p["y"] - p["r"])))

        # ── Hand animation ────────────────────────────────────────────────────
        if frames:
            # Use loaded image frames (GIF / PNG sequence)
            frame_timer += dt
            if frame_timer >= frame_interval:
                frame_timer = 0.0
                frame_idx   = (frame_idx + 1) % len(frames)
            img  = frames[frame_idx]
            rect = img.get_rect(center=(cx, cy - 20))
            screen.blit(img, rect)
        else:
            # Procedural animated hand
            draw_hand(screen, cx, cy - 20, elapsed, side)

        # ── MI instruction text ───────────────────────────────────────────────
        draw_text_centered(screen,
                           f"Imagine moving your {side_str} hand",
                           fonts["large"], C["text_hi"], cy + 115)


        # Phase badge (top center)
        badge_col = hand_col
        draw_text_centered(screen, f"◀  {lbl_str}  ▶" if side == "left"
                           else f"▶  {lbl_str}  ◀",
                           fonts["medium"], badge_col, 60)

        draw_hud(screen, fonts, trial_num, total_trials,
                 lbl_str, elapsed, duration, bar_col)

        pygame.display.flip()
        clock.tick(CONFIG["fps"])
        trial_data.extend(collect_eeg_data(lbl_str))

    return trial_data


def show_inter_trial(screen: pygame.Surface, fonts: dict,
                     duration: float, trial_num: int, total_trials: int,
                     clock: pygame.time.Clock) -> list:
    """Inter-trial break screen."""
    global current_label
    current_label = LABEL["INTER"]

    W, H = screen.get_size()
    cx, cy = W // 2, H // 2
    trial_data = []
    start = time.perf_counter()

    while True:
        elapsed = time.perf_counter() - start
        if elapsed >= duration:
            break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_event.set()
                return trial_data
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                quit_event.set()
                return trial_data

        if quit_event.is_set():
            break

        fill_bg_gradient(screen, C["bg_inter"], C["bg"])

        remain = max(0, duration - elapsed)
        draw_text_centered(screen, "INTER-TRIAL BREAK", fonts["large"],
                           C["text_mid"], cy - 60)
        draw_text_centered(screen, f"Next trial begins in  {remain:.0f}s",
                           fonts["medium"], C["text_dim"], cy + 10)

        if trial_num < total_trials:
            draw_text_centered(screen, f"Trial {trial_num + 1} of {total_trials} coming up",
                               fonts["small"], C["text_dim"], cy + 60)

        draw_progress_bar(screen,
                          W // 2 - 200, cy + 110, 400, 12,
                          elapsed / duration, C["timer_bg"],
                          bg=(20, 20, 40))

        draw_hud(screen, fonts, trial_num, total_trials,
                 "BREAK", elapsed, duration, C["timer_bg"])

        pygame.display.flip()
        clock.tick(CONFIG["fps"])
        trial_data.extend(collect_eeg_data("INTER"))

    return trial_data


# ══════════════════════════════════════════════════════════════════════════════
# 8. ASSET LOADING (Optional — fallback to procedural)
# ══════════════════════════════════════════════════════════════════════════════
def load_hand_frames(side: str) -> list:
    """
    Attempt to load hand animation frames from assets directory.
    Returns list of pygame.Surface or empty list (uses procedural fallback).

    Supported:
      assets/left_hand.gif   / assets/right_hand.gif
      assets/left_frames/frame*.png
      assets/right_frames/frame*.png
    """
    assets = Path(CONFIG["assets_dir"])
    frames = []

    # Try GIF
    gif_path = assets / f"{side}_hand.gif"
    if gif_path.exists():
        try:
            # Pygame doesn't natively animate GIFs — load via PIL if available
            from PIL import Image
            gif = Image.open(gif_path)
            for frame_num in range(gif.n_frames):
                gif.seek(frame_num)
                frame = gif.convert("RGBA")
                py_surf = pygame.image.fromstring(
                    frame.tobytes(), frame.size, "RGBA")
                py_surf = pygame.transform.scale(py_surf, (320, 280))
                frames.append(py_surf)
            logger.info(f"Loaded {len(frames)} frames from {gif_path}")
            return frames
        except ImportError:
            logger.warning("Pillow not found — GIF loading unavailable. Using procedural animation.")
        except Exception as e:
            logger.warning(f"GIF load failed: {e}")

    # Try PNG frame sequence
    frame_dir = assets / f"{side}_frames"
    if frame_dir.exists():
        png_files = sorted(frame_dir.glob("frame*.png"))
        for p in png_files:
            try:
                img = pygame.image.load(str(p)).convert_alpha()
                img = pygame.transform.scale(img, (320, 280))
                frames.append(img)
            except Exception as e:
                logger.warning(f"Frame load failed {p}: {e}")
        if frames:
            logger.info(f"Loaded {len(frames)} PNG frames for {side} hand")
            return frames

    logger.info(f"No asset files found for {side} hand -> using procedural animation")
    return []


# ══════════════════════════════════════════════════════════════════════════════
# 9. SUBJECT INPUT SCREEN
# ══════════════════════════════════════════════════════════════════════════════
def get_subject_id(screen: pygame.Surface, fonts: dict,
                   clock: pygame.time.Clock) -> str:
    """
    Render an input screen for the Subject ID before the experiment.
    Returns the entered subject ID string.
    """
    W, H = screen.get_size()
    cx, cy = W // 2, H // 2
    subject_id = ""
    cursor_blink = True
    last_blink   = time.time()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN and subject_id.strip():
                    return subject_id.strip()
                elif event.key == pygame.K_BACKSPACE:
                    subject_id = subject_id[:-1]
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                elif len(subject_id) < 12 and event.unicode.isprintable():
                    subject_id += event.unicode

        # Blink cursor
        if time.time() - last_blink > 0.5:
            cursor_blink = not cursor_blink
            last_blink   = time.time()

        fill_bg_gradient(screen, (10, 12, 35), C["bg"])

        # ── Title ─────────────────────────────────────────────────────────────
        draw_text_centered(screen,
                           "MOTOR IMAGERY EEG SYSTEM",
                           fonts["title"], C["left_hand"], cy - 160)
        draw_text_centered(screen,
                           "DATA COLLECTION",
                           fonts["small"], C["text_dim"], cy - 105)

        # ── Divider ───────────────────────────────────────────────────────────
        pygame.draw.line(screen, C["text_dim"],
                         (cx - 200, cy - 80), (cx + 200, cy - 80), 1)

        # ── Prompt ────────────────────────────────────────────────────────────
        draw_text_centered(screen, "Enter Subject ID",
                           fonts["medium"], C["text_mid"], cy - 55)

        # Input box
        box_w, box_h = 360, 58
        box_rect = pygame.Rect(cx - box_w // 2, cy - 10, box_w, box_h)
        pygame.draw.rect(screen, (20, 25, 55), box_rect, border_radius=10)
        pygame.draw.rect(screen, C["left_hand"], box_rect,
                         width=2, border_radius=10)

        display_text = subject_id + ("|" if cursor_blink else " ")
        id_surf = fonts["large"].render(display_text, True, C["text_hi"])
        id_rect = id_surf.get_rect(center=box_rect.center)
        screen.blit(id_surf, id_rect)

        draw_text_centered(screen, "Press ENTER to start  ·  ESC to quit",
                           fonts["tiny"], C["text_dim"], cy + 75)

        # Protocol summary
        draw_text_centered(screen,
                           f"{CONFIG['n_trials']} trials  ·  "
                           f"{CONFIG['sampling_rate']} Hz  ·  "
                           f"{CONFIG['n_channels']} channels  · ",
                           fonts["tiny"], C["text_dim"], cy + 115)

        pygame.display.flip()
        clock.tick(30)


# ══════════════════════════════════════════════════════════════════════════════
# 10. COUNTDOWN INTRO
# ══════════════════════════════════════════════════════════════════════════════
def show_experiment_intro(screen: pygame.Surface, fonts: dict,
                           clock: pygame.time.Clock,
                           subject_id: str) -> None:
    """3-2-1 countdown before experiment begins."""
    W, H = screen.get_size()
    cx, cy = W // 2, H // 2

    for count in [3, 2, 1]:
        start = time.perf_counter()
        while time.perf_counter() - start < 1.0:
            fill_bg_gradient(screen, C["bg"], (5, 5, 15))
            t = time.perf_counter() - start
            pulse = 1.0 + 0.2 * (1 - t)
            draw_text_centered(screen, "Experiment starting in",
                               fonts["medium"], C["text_mid"], cy - 80)
            cd_font = pygame.font.SysFont("Times New Roman", int(160 * pulse), bold=True)
            cd_s = cd_font.render(str(count), True, C["rest_circle"])
            cd_r = cd_s.get_rect(center=(cx, cy + 10))
            draw_glow_circle(screen, C["rest_circle"], (cx, cy + 10), 100, alpha=60)
            screen.blit(cd_s, cd_r)
            draw_text_centered(screen, f"Subject: {subject_id}",
                               fonts["small"], C["text_dim"], cy + 110)
            pygame.display.flip()
            clock.tick(60)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()


def show_end_screen(screen: pygame.Surface, fonts: dict,
                     clock: pygame.time.Clock,
                     subject_id: str, n_trials: int,
                     save_path: str) -> None:
    """Display experiment completion screen."""
    W, H = screen.get_size()
    cx, cy = W // 2, H // 2
    start  = time.time()

    while time.time() - start < 8.0:
        for event in pygame.event.get():
            if event.type in (pygame.QUIT, pygame.KEYDOWN):
                return

        fill_bg_gradient(screen, C["bg_rest"], C["bg"])
        draw_text_centered(screen, "✓  EXPERIMENT COMPLETE",
                           fonts["large"], C["rest_circle"], cy - 100)
        draw_text_centered(screen, f"Subject: {subject_id}  ·  {n_trials} trials recorded",
                           fonts["medium"], C["text_mid"], cy - 40)
        draw_text_centered(screen, f"Data saved to:", fonts["small"],
                           C["text_dim"], cy + 20)
        draw_text_centered(screen, save_path, fonts["small"],
                           C["text_hi"], cy + 55)
        draw_text_centered(screen, "Press any key to exit",
                           fonts["tiny"], C["text_dim"], cy + 120)

        pygame.display.flip()
        clock.tick(30)


# ══════════════════════════════════════════════════════════════════════════════
# 11. MAIN TRIAL RUNNER
# ══════════════════════════════════════════════════════════════════════════════
def run_trial(screen: pygame.Surface, fonts: dict, clock: pygame.time.Clock,
              trial_num: int, n_trials: int,
              subject_id: str, session_folder: Path) -> bool:
    """
    Execute one full trial of the AO+MI protocol.

    Protocol:
        REST(10) -> PREP_LEFT(5) -> LEFT_MI(15) ->
        REST(10) -> PREP_RIGHT(5) -> RIGHT_MI(15) ->
        FINAL_REST(10) -> INTER_TRIAL(10)

    Returns False if quit was requested.
    """
    T    = CONFIG["timing"]
    trial_data: list = []

    logger.info(f"══ Trial {trial_num}/{n_trials} START ══")

    # ── Phase 1: REST ─────────────────────────────────────────────────────────
    trial_data.extend(
        show_rest(screen, fonts, T["rest"], trial_num, n_trials, clock,
                  "REST (pre-left)"))
    if quit_event.is_set():
        return False

    # ── Phase 2: PREPARE LEFT ─────────────────────────────────────────────────
    trial_data.extend(
        show_preparation(screen, fonts, T["preparation"], "left",
                         trial_num, n_trials, clock))
    if quit_event.is_set():
        return False

    # ── Phase 3: LEFT MI ──────────────────────────────────────────────────────
    trial_data.extend(
        show_animation_phase(screen, fonts, T["imagery"], "left",
                             trial_num, n_trials, clock))
    if quit_event.is_set():
        return False

    # ── Phase 4: REST ─────────────────────────────────────────────────────────
    trial_data.extend(
        show_rest(screen, fonts, T["rest"], trial_num, n_trials, clock,
                  "REST (mid)"))
    if quit_event.is_set():
        return False

    # ── Phase 5: PREPARE RIGHT ────────────────────────────────────────────────
    trial_data.extend(
        show_preparation(screen, fonts, T["preparation"], "right",
                         trial_num, n_trials, clock))
    if quit_event.is_set():
        return False

    # ── Phase 6: RIGHT MI ─────────────────────────────────────────────────────
    trial_data.extend(
        show_animation_phase(screen, fonts, T["imagery"], "right",
                             trial_num, n_trials, clock))
    if quit_event.is_set():
        return False

    # ── Phase 7: FINAL REST ───────────────────────────────────────────────────
    trial_data.extend(
        show_rest(screen, fonts, T["rest"], trial_num, n_trials, clock,
                  "REST (post)"))
    if quit_event.is_set():
        return False

    # ── Phase 8: INTER-TRIAL BREAK (skip on last trial) ──────────────────────
    if trial_num < n_trials:
        trial_data.extend(
            show_inter_trial(screen, fonts, T["inter_trial"],
                             trial_num, n_trials, clock))
        if quit_event.is_set():
            return False

    # ── Save trial data ────────────────────────────────────────────────────────
    trial_file = session_folder / f"S{subject_id}_T{trial_num:02d}.csv"
    # Reformat: prepend readable label names, add index column
    formatted  = []
    for i, row in enumerate(trial_data):
        # row = (timestamp, ch0, ch1, ch2, ch3, label_int, label_str)
        formatted.append([i] + list(row))

    # Rewrite header to include index
    with open(trial_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_idx", "timestamp",
                          "C3", "C4", "Cz", "Fz",
                          "label_int", "label_str"])
        writer.writerows(formatted)

    logger.info(f"Trial {trial_num} saved: {len(formatted)} samples -> {trial_file}")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# 12. FONT LOADER
# ══════════════════════════════════════════════════════════════════════════════
def load_fonts() -> dict:
    """Load system fonts in various sizes used throughout the UI."""
    def sf(size, bold=False):
        return pygame.font.SysFont("Times New Roman", size, bold=bold)

    return {
        "title":  sf(56, bold=True),
        "large":  sf(44, bold=True),
        "medium": sf(32),
        "small":  sf(24),
        "tiny":   sf(18),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 13. MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def main():
    logger.info("=== MI EEG System initializing ===")

    # ── Pygame init ───────────────────────────────────────────────────────────
    pygame.init()
    pygame.display.set_caption("Motor Imagery EEG — AO+MI Protocol")

    # Fullscreen (use SCALED for best cross-platform behaviour)
    info   = pygame.display.Info()
    screen = pygame.display.set_mode(
        (info.current_w, info.current_h),
        pygame.FULLSCREEN | pygame.SCALED,
    )
    pygame.mouse.set_visible(False)
    clock  = pygame.time.Clock()
    fonts  = load_fonts()

    logger.info(f"Display: {screen.get_width()}×{screen.get_height()}")

    # ── EEG thread ────────────────────────────────────────────────────────────
    eeg_thread = threading.Thread(
        target=eeg_simulation_thread,
        args=(CONFIG["sampling_rate"], CONFIG["n_channels"]),
        daemon=True,
    )
    eeg_thread.start()
    eeg_running.wait(timeout=2.0)
    logger.info("EEG thread ready")

    # ── Subject ID input ──────────────────────────────────────────────────────
    subject_id   = get_subject_id(screen, fonts, clock)
    session_date = datetime.now().strftime("%Y-%m-%d")
    logger.info(f"Subject: {subject_id}  Session: {session_date}")

    # ── Create folders ────────────────────────────────────────────────────────
    session_folder = create_folders(subject_id, session_date)

    # ── Intro countdown ───────────────────────────────────────────────────────
    show_experiment_intro(screen, fonts, clock, subject_id)

    # ── Run trials ────────────────────────────────────────────────────────────
    n_trials    = CONFIG["n_trials"]
    completed   = 0
    experiment_start = datetime.now().isoformat()

    for trial_num in range(1, n_trials + 1):
        if quit_event.is_set():
            logger.warning("Quit event detected — stopping experiment")
            break

        success = run_trial(screen, fonts, clock, trial_num, n_trials,
                             subject_id, session_folder)
        if success:
            completed += 1

        if quit_event.is_set():
            break

    # ── Save metadata ─────────────────────────────────────────────────────────
    T = CONFIG["timing"]
    total_per_trial = (T["rest"] * 3 + T["preparation"] * 2
                       + T["imagery"] * 2 + T["inter_trial"])

    metadata = {
        "subject_id":     subject_id,
        "session_date":   session_date,
        "experiment_start": experiment_start,
        "experiment_end": datetime.now().isoformat(),
        "protocol":       "AO + MI",
        "n_trials_planned":   n_trials,
        "n_trials_completed": completed,
        "sampling_rate_hz":   CONFIG["sampling_rate"],
        "n_channels":         CONFIG["n_channels"],
        "channel_names":      CONFIG["channel_names"],
        "timing_seconds": {
            "rest_duration":        T["rest"],
            "preparation_duration": T["preparation"],
            "imagery_duration":     T["imagery"],
            "inter_trial_duration": T["inter_trial"],
            "total_per_trial":      total_per_trial,
        },
        "labels": {
            "0": "REST",
            "1": "PREPARE",
            "2": "LEFT_MI",
            "3": "RIGHT_MI",
            "4": "INTER",
        },
        "eeg_mode":    "simulated" if True else "hardware",
        "animation":   "procedural_pygame",
        "notes":       "Generated by MI EEG System v1.0",
    }
    save_metadata(metadata, session_folder)

    # ── End screen ────────────────────────────────────────────────────────────
    quit_event.clear()
    pygame.mouse.set_visible(True)
    show_end_screen(screen, fonts, clock, subject_id, completed,
                     str(session_folder))

    # ── Cleanup ───────────────────────────────────────────────────────────────
    quit_event.set()
    logger.info(f"Experiment complete. {completed}/{n_trials} trials. "
                f"Data: {session_folder}")
    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
