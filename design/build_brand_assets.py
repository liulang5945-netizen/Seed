"""Rebuild all brand assets from the final Seed mark (vector + raster).

Vector files keep their existing structure; only the traced path pair
(struct #124A38 / foliage #AAD66A) or the mono path is swapped in, so
viewBoxes, cards and gradients are untouched. Raster files are composed
from the master mark PNG (design/variants/seed-shell-final-mark.png).
"""

import os
import re

from PIL import Image, ImageDraw, ImageFont

ROOT = r"E:/Seed/taiji-harness"
LOGO = r"E:/Seed/design/logo"
MARK = r"E:/Seed/design/variants/seed-shell-final-mark.png"
STRUCT = (18, 74, 56)
LEAF = (170, 214, 106)

mono_d = open(os.path.join(LOGO, "seed-mark-mono.path.txt"), encoding="utf-8").read().strip()
color = open(os.path.join(LOGO, "seed-mark-color.svg"), encoding="utf-8").read()
paths = re.findall(r'<path d="([^"]+)"', color)
struct_d, foliage_d = paths[0], paths[1]
print("path lens:", len(mono_d), len(struct_d), len(foliage_d))


def patch(path, subs):
    """Apply regex substitutions to a text file in place."""
    t = open(path, encoding="utf-8").read()
    for pat, rep in subs:
        t, n = re.subn(pat, rep, t)
        if n == 0:
            print(f"  !! no match in {os.path.basename(path)} for {pat[:40]}")
    open(path, "w", encoding="utf-8", newline="\n").write(t)


# ---------- vector: swap the traced paths into the existing structures ----------
OLD_STRUCT = r'M12\.42 17\.11[^"]+'
OLD_MONO = r'M12\.42 17\.11[^"]+'  # same family; distinguish by fill attribute

patch(
    rf"{ROOT}/apps/web/public/favicon.svg",
    [
        (r'(<path d=")' + OLD_STRUCT + r'(" fill="#124A38")', r"\g<1>" + struct_d + r"\g<2>"),
        (r'(<path d=")M14\.44 12\.72[^"]+(" fill="#AAD66A")', r"\g<1>" + foliage_d + r"\g<2>"),
    ],
)
patch(
    rf"{ROOT}/apps/web/public/favicon-dark.svg",
    [
        (r'(<path d=")' + OLD_MONO + r'(" fill="#fff")', r"\g<1>" + mono_d + r"\g<2>"),
    ],
)
patch(
    rf"{ROOT}/website/public/favicon.svg",
    [
        (r'(<path d=")' + OLD_STRUCT + r'(" fill="#124A38")', r"\g<1>" + struct_d + r"\g<2>"),
        (r'(<path d=")M14\.44 12\.72[^"]+(" fill="#AAD66A")', r"\g<1>" + foliage_d + r"\g<2>"),
    ],
)
patch(
    rf"{ROOT}/website/public/wordmark.svg",
    [
        (r'(<path d=")' + OLD_MONO + r'(" fill="currentColor")', r"\g<1>" + mono_d + r"\g<2>"),
    ],
)
# the new mark is much wider in the 24 grid than the old one — nudge the
# wordmark text right so the mark (scaled 0.875 ≈ 18 units wide) never overlaps
patch(
    rf"{ROOT}/website/public/wordmark.svg",
    [
        (r'<text x="30"', '<text x="34"'),
    ],
)

for icon in ("icon.svg", "icon-windows.svg", "icon-macos.svg"):
    patch(
        rf"{ROOT}/apps/desktop/resources/{icon}",
        [
            (r'(<path d=")' + OLD_STRUCT + r'(" fill="#124A38")', r"\g<1>" + struct_d + r"\g<2>"),
            (r'(<path d=")M14\.44 12\.72[^"]+(" fill="#AAD66A")', r"\g<1>" + foliage_d + r"\g<2>"),
            (r"scale\(22\.5\)", "scale(39)"),  # mark fills the card like the 0.84 tile
        ],
    )
print("vector assets patched")

# ---------- raster: compose from the master mark ----------
master = Image.open(MARK).convert("RGBA")
bbox = master.getbbox()
mark = master.crop(bbox)


def mono_tinted(img, rgb):
    """White/dark silhouette of the mark."""
    a = img.getchannel("A")
    out = Image.new("RGBA", img.size, rgb + (0,))
    out.putalpha(a)
    return out


mark_white = mono_tinted(mark, (255, 255, 255))
mark_black = mono_tinted(mark, (17, 17, 17))


def load_font(name, size):
    for cand in name:
        if os.path.exists(cand):
            return ImageFont.truetype(cand, size)
    return ImageFont.load_default()


SERIF = [r"C:\Windows\Fonts\georgia.ttf", r"C:\Windows\Fonts\times.ttf"]
SANS_B = [r"C:\Windows\Fonts\segoeuib.ttf", r"C:\Windows\Fonts\arialbd.ttf"]


def text_w(font, s):
    b = font.getbbox(s)
    return b[2] - b[0]


def brand_strip(w, h, mark_img, text_color, font, scale=0.86):
    """Transparent strip: mark left, wordmark text right, vertically centred."""
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    mh = int(h * scale)
    mw = int(mark_img.size[0] * mh / mark_img.size[1])
    m = mark_img.resize((mw, mh), Image.LANCZOS)
    img.alpha_composite(m, (int(w * 0.03), (h - mh) // 2))
    fs = int(h * 0.42)
    f = load_font(font, fs)
    tx = int(w * 0.03) + mw + int(w * 0.07)
    ty = (h - fs) // 2 - int(fs * 0.12)
    d = ImageDraw.Draw(img)
    d.text((tx, ty), "Seed", font=f, fill=text_color)
    return img


DEST_I = rf"{ROOT}/apps/desktop/installer/assets"
DEST_B = rf"{ROOT}/packages/skill/skill-badge/assets"
os.makedirs(DEST_I, exist_ok=True)

brand_strip(600, 196, mark, STRUCT, SERIF).save(os.path.join(DEST_I, "brand.png"))
brand_strip(1200, 392, mark, STRUCT, SERIF).save(os.path.join(DEST_I, "brand-2x.png"))
brand_strip(600, 196, mark_white, (255, 255, 255), SERIF).save(
    os.path.join(DEST_I, "brand-dark.png")
)
brand_strip(1200, 392, mark_white, (255, 255, 255), SERIF).save(
    os.path.join(DEST_I, "brand-dark-2x.png")
)

# uninstaller sidebar 164x314: colour mark on top, wordmark below
side = Image.new("RGBA", (164, 314), (0, 0, 0, 0))
mw = 122
mh = int(mark.size[1] * mw / mark.size[0])
m = mark.resize((mw, mh), Image.LANCZOS)
side.alpha_composite(m, ((164 - mw) // 2, 18))
f = load_font(SERIF, 42)
tw = text_w(f, "Seed")
d = ImageDraw.Draw(side)
d.text(((164 - tw) // 2, 250), "Seed", font=f, fill=STRUCT)
side.save(os.path.join(DEST_I, "uninstaller-sidebar.png"))

# skill badge 726x120: black mono mark + "Powered by Seed"
badge = Image.new("RGBA", (726, 120), (0, 0, 0, 0))
bh = 78
bw = int(mark_black.size[0] * bh / mark_black.size[1])
bm = mark_black.resize((bw, bh), Image.LANCZOS)
badge.alpha_composite(bm, (28, (120 - bh) // 2))
f = load_font(SANS_B, 58)
badge_t = ImageDraw.Draw(badge)
badge_t.text((28 + bw + 36, 22), "Powered by Seed", font=f, fill=(17, 17, 17))
badge.save(os.path.join(DEST_B, "dsh-badge.png"))

# desktop icon PNGs from the confirmed tile
tile = Image.open(r"E:/Seed/design/icons/seed-icon-tile-1024.png").convert("RGBA")
tile.resize((1104, 1104), Image.LANCZOS).save(rf"{ROOT}/apps/desktop/resources/icon.png")
tile.resize((1024, 1024), Image.LANCZOS).save(rf"{ROOT}/apps/desktop/resources/icon-windows.png")
tile.resize((1024, 1024), Image.LANCZOS).save(rf"{ROOT}/apps/desktop/resources/icon-macos.png")

print("raster assets done")
