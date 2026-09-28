"""Compose installer/banner artwork from the approved anchor's own pixels.

Drafts only: nothing under taiji-harness is overwritten here. The anchor is
measured rather than re-typeset (mark rows 158..727 / cols 333..692, wordmark
rows 798..906 / cols 375..666 as four letter groups), so the lettering stays the
owner's drawing instead of a substitute font.

Outputs design/brand-candidates/*.png plus one contact sheet.
"""
from __future__ import annotations

import glob
import os

import numpy as np
from PIL import Image

ANCHOR = glob.glob(r"E:/Seed/seed-logo_assets/*bb84fe254933.png")[0]
OUT = r"E:/Seed/design/brand-candidates"
SHEET = os.path.join(OUT, "contact-sheet.png")
MARK_ROWS = (158, 727)
MARK_COLS = (333, 692)
WORD_ROWS = (798, 906)
WORD_COLS = (375, 666)
GAP_RATIO = 0.22          # gap between mark and wordmark, as a share of canvas height
MARK_HEIGHT_RATIO = 0.82  # mark height as a share of canvas height

os.makedirs(OUT, exist_ok=True)
anchor = np.asarray(Image.open(ANCHOR).convert("RGB")).astype(np.int16)
background = anchor[5, 5]


def cut(rows: tuple[int, int], cols: tuple[int, int]) -> Image.Image:
    """Crop and knock out the ivory board so the artwork sits on transparency."""
    box = anchor[rows[0]:rows[1], cols[0]:cols[1]]
    alpha = (np.abs(box - background).max(axis=2) > 24).astype(np.uint8) * 255
    rgba = np.dstack([box.astype(np.uint8), alpha])
    return Image.fromarray(rgba, "RGBA")


mark = cut(MARK_ROWS, MARK_COLS)
wordmark = cut(WORD_ROWS, WORD_COLS)
white_wordmark = Image.new("RGBA", wordmark.size)
white_wordmark.paste((255, 255, 255, 255), (0, 0), wordmark.split()[3])
dark_mark = Image.new("RGBA", mark.size)
dark_mark.paste((255, 255, 255, 255), (0, 0), mark.split()[3])
print(f"mark={mark.size} wordmark={wordmark.size}")


def banner(width: int, height: int, light: bool) -> Image.Image:
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    symbol, word = (mark, wordmark) if light else (dark_mark, white_wordmark)
    mark_h = max(8, round(height * MARK_HEIGHT_RATIO))
    gap = round(height * GAP_RATIO)
    scaled_mark = symbol.resize((round(mark_h * symbol.width / symbol.height), mark_h), Image.LANCZOS)
    word_h = round(height * 0.30)
    scaled_word = word.resize((round(word_h * word.width / word.height), word_h), Image.LANCZOS)
    total = scaled_mark.width + gap + scaled_word.width
    left = max(0, round((width - total) / 2))
    canvas.paste(scaled_mark, (left, round((height - scaled_mark.height) / 2)), scaled_mark)
    canvas.paste(scaled_word, (left + scaled_mark.width + gap, round((height - scaled_word.height) / 2)), scaled_word)
    return canvas


for (name, size, light) in (
    ("brand", (600, 196), True),
    ("brand-2x", (1200, 392), True),
    ("brand-dark", (600, 196), False),
    ("brand-dark-2x", (1200, 392), False),
    ("badge", (726, 120), True),
):
    image = banner(*size, light=light)
    if name == "badge":
        board = Image.new("RGBA", image.size, (255, 255, 255, 255))
        board.alpha_composite(image)
        image = board
    image.save(os.path.join(OUT, f"{name}.png"))
    print(f"{name}: {image.size} light={light}")

side = Image.new("RGBA", (164, 314), (0, 0, 0, 0))
# Fit the stack: the mark keeps the width margin, the wordmark takes what is left
# of the height, and the pair is centred vertically.
side_word_h = 34
side_gap = 16
side_mark_h = 314 - 2 * 24 - side_word_h - side_gap
side_mark = mark.resize((round(side_mark_h * mark.width / mark.height), side_mark_h), Image.LANCZOS)
word_side = wordmark.resize((round(side_word_h * wordmark.width / wordmark.height), side_word_h), Image.LANCZOS)
if side_mark.width > 164 - 2 * 8:
    side_mark = mark.resize((164 - 2 * 8, round((164 - 2 * 8) * mark.height / mark.width)), Image.LANCZOS)
stack_h = side_mark.height + side_gap + side_word_h
top = round((314 - stack_h) / 2)
side.paste(side_mark, (round((164 - side_mark.width) / 2), top), side_mark)
side.paste(word_side, (round((164 - word_side.width) / 2), top + side_mark.height + side_gap), word_side)
side = Image.new("RGBA", side.size, (255, 255, 255, 255)).convert("RGBA") if False else side
# the shipped sidebar and badge sit on an opaque white board (NSIS converts to BMP)
side_board = Image.new("RGBA", side.size, (255, 255, 255, 255))
side_board.alpha_composite(side)
side_board.save(os.path.join(OUT, "uninstaller-sidebar.png"))
print(f"uninstaller-sidebar: {side.size} mark={side_mark.size} word={word_side.size} stack={stack_h} top={top}")

tiles = [("light 600x196", os.path.join(OUT, "brand.png")),
         ("dark 600x196", os.path.join(OUT, "brand-dark.png")),
         ("badge 726x120", os.path.join(OUT, "badge.png")),
         ("sidebar 164x314", os.path.join(OUT, "uninstaller-sidebar.png")),
         ("anchor", ANCHOR)]
cells = []
for label, path in tiles:
    art = Image.open(path).convert("RGBA")
    art.thumbnail((300, 300), Image.LANCZOS)
    cell = Image.new("RGBA", (320, 340), (250, 248, 244, 255))
    checker = Image.new("RGBA", (320, 340), (255, 255, 255, 255))
    d = np.asarray(checker).copy()
    mask = (np.indices(d.shape[:2]).sum(0) // 20) % 2 == 0
    d[mask] = (236, 232, 226, 255)
    checker = Image.fromarray(d, "RGBA")
    checker.alpha_composite(art, (max(0, (320 - art.width) // 2), max(0, (340 - art.height) // 2)))
    from PIL import ImageDraw
    ImageDraw.Draw(checker).text((8, 6), label, fill=(30, 30, 30))
    cells.append(checker)
sheet = Image.new("RGBA", (320 * len(cells), 340), (255, 255, 255, 255))
for index, cell in enumerate(cells):
    sheet.paste(cell, (index * 320, 0))
sheet.save(SHEET)
print("contact sheet ->", SHEET)
