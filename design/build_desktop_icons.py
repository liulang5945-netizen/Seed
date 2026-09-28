"""Re-cut the desktop icon rasters from the approved Seed icon master.

Same composition rules as design/icons/build_icons.py (ivory board sampled from the
brand anchor, rounded-square radius 22%, mark 84%), written at the exact sizes the
desktop packaging consumes: resources/icon.png (1104, electron-builder base),
icon-windows.png (1024, copied to the app icon) and icon-macos.png (1024, inset
rounded board for legacy ICNS).

Emitted PNGs are only written when their size matches the file they replace, and a
side-by-side contact sheet is produced for visual sign-off.
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw

MARK = r"E:/Seed/design/variants/seed-shell-final-mark.png"
ANCHOR = r"E:/Seed/seed-logo_assets/1044f28f-miora_edit_image-1790567714143-0-bb84fe254933.png"
RESOURCES = r"E:/Seed/taiji-harness/apps/desktop/resources"
SHEET = r"E:/Seed/design/logo/desktop-icons-contact-sheet.png"

RADIUS_RATIO = 0.22
MARK_RATIO = 0.84
TARGETS = {"icon.png": 1104, "icon-windows.png": 1024, "icon-macos.png": 1024}

bg = Image.open(ANCHOR).convert("RGB").getpixel((5, 5))
mark = Image.open(MARK).convert("RGBA")
bbox = mark.getbbox()
center_x, center_y = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
half = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) // 2 + 12
square = mark.crop((center_x - half, center_y - half, center_x + half, center_y + half))
print(f"mark={mark.size} bbox={bbox} board={bg}")


def make_tile(size: int) -> Image.Image:
    tile = Image.new("RGBA", (size, size), bg + (255,))
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [0, 0, size - 1, size - 1], radius=max(2, round(size * RADIUS_RATIO)), fill=255)
    inner = Image.new("RGBA", (size, size), bg + (255,))
    side = int(round(size * MARK_RATIO))
    inner.alpha_composite(square.resize((side, side), Image.LANCZOS), ((size - side) // 2, (size - side) // 2))
    return Image.composite(inner, Image.new("RGBA", (size, size), (0, 0, 0, 0)), mask)


for name, size in TARGETS.items():
    path = os.path.join(RESOURCES, name)
    before = Image.open(path)
    if before.size != (size, size):
        raise SystemExit(f"REFUSE: {name} is {before.size}, expected {(size, size)}")
    tile = make_tile(size)
    assert tile.size == (size, size)
    tiles = np.asarray(tile)[..., 3]
    coverage = float((tiles > 128).mean())
    before_rgba = np.asarray(before.convert("RGBA"))[..., 3]
    print(f"{name}: size={tile.size} coverage={coverage:.3f} previous_coverage={(before_rgba > 128).mean():.3f}")
    if not 0.9 < coverage < 1.0:
        raise SystemExit(f"REFUSE: {name} tile coverage {coverage:.3f} is not a full rounded square")
    tile.save(path)

cells = []
for name, size in TARGETS.items():
    cell = Image.open(os.path.join(RESOURCES, name)).convert("RGBA")
    canvas = Image.new("RGBA", (size + 24, size + 24), (255, 255, 255, 255))
    canvas.alpha_composite(cell, (12, 12))
    cells.append(canvas.resize((256, 256), Image.LANCZOS))
sheet = Image.new("RGB", (256 * len(cells) + 16 * (len(cells) + 1), 256 + 32), (240, 240, 240))
for index, cell in enumerate(cells):
    sheet.paste(cell, (16 + index * (256 + 16), 32))
sheet.save(SHEET)
print("contact sheet ->", SHEET)
