"""Build the Seed app icon resource pack from the final rounded-shell master.

Input : E:/Seed/design/variants/seed-shell-final-mark.png   (1024x1024 RGBA, transparent bg)
Output: E:/Seed/design/icons/
        - seed-icon.ico                (multi-size: 16/24/32/48/64/128/256)
        - seed-icon-{N}.png            (rounded-square RGBA tiles, N in 16..256)
        - seed-icon-tile-1024.png      (master tile for future derivation)

Rules: tile background = ivory sampled from the brand anchor; corners transparent;
rounded-square corner radius 22%.

Owner-tuned decisions (2026-09-28):
  - shell plumped to w/h 0.94 (roundness pick #3)
  - mark fills the tile at 84% (was 72% -> "logo too small in the icon")
  - tree grown to fill the rounder shell, trunk rooted into the shell stroke
  - palette refreshed to a young spring green
"""

import os

from PIL import Image, ImageDraw

MARK = r"E:/Seed/design/variants/seed-shell-final-mark.png"
ANCHOR = r"E:/Seed/seed-logo_assets/1044f28f-miora_edit_image-1790567714143-0-bb84fe254933.png"
OUT = r"E:/Seed/design/icons"
SIZES = [16, 24, 32, 48, 64, 128, 256]
RADIUS_RATIO = 0.22  # rounded-square corner radius
MARK_RATIO = 0.84  # mark size relative to tile

bg = Image.open(ANCHOR).convert("RGB").getpixel((5, 5))
mark = Image.open(MARK).convert("RGBA")
print("master size:", mark.size, "tile bg:", bg)

# --- square crop around the mark (alpha bbox, small padding) ---
bbox = mark.getbbox()
cx, cy = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
half = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) // 2 + 12
square = mark.crop((cx - half, cy - half, cx + half, cy + half))
print("mark bbox:", bbox, "-> square crop:", square.size)

os.makedirs(OUT, exist_ok=True)


def make_tile(size):
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [0, 0, size - 1, size - 1], radius=max(2, round(size * RADIUS_RATIO)), fill=255
    )
    inner = Image.new("RGBA", (size, size), bg + (255,))
    side = int(round(size * MARK_RATIO))
    m = square.resize((side, side), Image.LANCZOS)
    inner.alpha_composite(m, ((size - side) // 2, (size - side) // 2))
    return Image.composite(inner, Image.new("RGBA", (size, size), (0, 0, 0, 0)), mask)


make_tile(1024).save(os.path.join(OUT, "seed-icon-tile-1024.png"))
for s in SIZES:
    make_tile(s).save(os.path.join(OUT, f"seed-icon-{s}.png"))

make_tile(256).save(os.path.join(OUT, "seed-icon.ico"), sizes=[(s, s) for s in SIZES])
print("done ->", OUT)
