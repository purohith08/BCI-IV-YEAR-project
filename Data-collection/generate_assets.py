import pygame
import math
from pathlib import Path
from PIL import Image

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
N_FRAMES = 36
FRAME_W, FRAME_H = 400, 340
FPS = 12   # GIF speed

HAND_L = (60, 160, 255)
HAND_R = (255, 140, 50)
GLOW_L = (20, 60, 150, 60)
GLOW_R = (150, 60, 20, 60)
BG_L   = (10, 25, 70)
BG_R   = (70, 22, 10)


# ─────────────────────────────────────────────────────────────
# DRAW FUNCTION
# ─────────────────────────────────────────────────────────────
def draw_frame_hand(surf, frame, n_frames, side):
    cx, cy = FRAME_W // 2, FRAME_H // 2 - 20
    phase  = frame / n_frames
    wave   = math.sin(2 * math.pi * phase)
    direction = -1 if side == "left" else 1

    hand_col = HAND_L if side == "left" else HAND_R
    bg_col   = BG_L if side == "left" else BG_R

    surf.fill(bg_col)

    # Glow
    glow_r = int(90 + abs(wave) * 25)
    glow_s = pygame.Surface((glow_r * 4, glow_r * 4), pygame.SRCALPHA)
    for gr in range(glow_r * 2, 0, -6):
        a = int(50 * (gr / (glow_r * 2)))
        pygame.draw.circle(glow_s, (*hand_col, a),
                           (glow_r * 2, glow_r * 2), gr)

    drift_x = int(direction * wave * 45)
    surf.blit(glow_s, (cx + drift_x - glow_r * 2, cy - glow_r * 2))

    # Palm
    rotation = math.radians(wave * 10 * direction)

    def rot(px, py):
        cos_a, sin_a = math.cos(rotation), math.sin(rotation)
        return (cx + drift_x + int(px * cos_a - py * sin_a),
                cy + int(px * sin_a + py * cos_a))

    pw, ph = 70, 75
    palm = [rot(-pw//2+10, -ph//2+5),
            rot(pw//2-10, -ph//2+5),
            rot(pw//2, ph//2-8),
            rot(-pw//2, ph//2-8)]
    pygame.draw.polygon(surf, hand_col, palm)

    # Simple finger lines
    for i in range(4):
        start = rot(-20 + i*15, -30)
        end   = rot(-20 + i*15, -80)
        pygame.draw.line(surf, hand_col, start, end, 10)

    # Thumb
    tb = rot(direction * 30, 0)
    pygame.draw.line(surf, hand_col, tb, (tb[0]+direction*40, tb[1]-20), 12)


# ─────────────────────────────────────────────────────────────
# GENERATE FRAMES + GIF
# ─────────────────────────────────────────────────────────────
def generate_frames_and_gif(side, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    surf = pygame.Surface((FRAME_W, FRAME_H))
    images = []

    print(f"\nGenerating {side.upper()} hand...")

    for f in range(N_FRAMES):
        draw_frame_hand(surf, f, N_FRAMES, side)

        # Save PNG
        png_path = out_dir / f"frame{f+1:03d}.png"
        pygame.image.save(surf, str(png_path))

        # Convert to PIL image for GIF
        data = pygame.image.tostring(surf, "RGB")
        img = Image.frombytes("RGB", (FRAME_W, FRAME_H), data)
        images.append(img)

    # Save GIF
    gif_path = out_dir.parent / f"{side}_hand.gif"
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / FPS),
        loop=0
    )

    print(f"✔ PNG frames saved in: {out_dir}")
    print(f"✔ GIF saved at: {gif_path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    pygame.init()
    pygame.display.set_mode((1, 1))  # headless

    assets = Path("assets")

    generate_frames_and_gif("left", assets / "left_frames")
    generate_frames_and_gif("right", assets / "right_frames")

    print("\n✅ All assets generated successfully!")
    pygame.quit()


if __name__ == "__main__":
    main()