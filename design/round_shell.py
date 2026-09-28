"""Round the Seed shell: keep the tree, redraw the shell from a smooth fitted profile.

Profile fitted to the original silhouette:
  top half  (t < t_m): pointed tip   -> hw = M * (1 - u^2)^p_top
  bottom half        : round bottom  -> hw = M * (1 - v^2)^0.5
t_m = 0.54 (widest point of the original), M = shell half-width.
"""
import os
import numpy as np
from PIL import Image, ImageDraw

SRC = r"E:/Seed/seed-logo_assets/1044f28f-miora_edit_image-1790567714143-0-bb84fe254933.png"
OUT = r"E:/Seed/design/variants"
os.makedirs(OUT, exist_ok=True)

CX = 512
STROKE = 13
SS = 4
T_M = 0.54
SHELL_COLOR = (19, 60, 50)

# ---------- load & mask ----------
im = Image.open(SRC).convert("RGB")
a = np.asarray(im).astype(np.int16)
bg = tuple(int(v) for v in a[5, 5])
mask = np.abs(a - np.array(bg)).max(axis=2) > 18
band = np.zeros_like(mask); band[150:740, :] = True
mask &= band

def _shift(m, dy, dx):
    out = np.zeros_like(m)
    ys = slice(max(0, dy), m.shape[0] + min(0, dy))
    yd = slice(max(0, -dy), m.shape[0] + min(0, -dy))
    xs = slice(max(0, dx), m.shape[1] + min(0, dx))
    xd = slice(max(0, -dx), m.shape[1] + min(0, -dx))
    out[yd, xd] = m[ys, xs]
    return out

def dilate(m):
    return m | _shift(m, 1, 0) | _shift(m, -1, 0) | _shift(m, 0, 1) | _shift(m, 0, -1)

def erode(m, it):
    for _ in range(it):
        m = m & _shift(m, 1, 0) & _shift(m, -1, 0) & _shift(m, 0, 1) & _shift(m, 0, -1)
    return m

def fill_holes(m):
    outside = ~m
    outside[1:-1, 1:-1] = False
    outside[0, :] = ~m[0, :]; outside[-1, :] = ~m[-1, :]
    outside[:, 0] = ~m[:, 0]; outside[:, -1] = ~m[:, -1]
    while True:
        nxt = dilate(outside) & ~m
        if nxt.sum() == outside.sum():
            return ~outside
        outside = nxt

filled = fill_holes(mask)
inner9 = erode(filled, 9)
tree_mask = mask & inner9

ys, xs = np.where(filled)
Y_TOP, Y_BOT = int(ys.min()), int(ys.max())
TIP = Y_TOP + 18                      # skip the little stem hook rows
H = Y_BOT - TIP

# ---------- tree layer: crop to bbox (flat cut at the bottom is hidden by the shell stroke) ----------
tys, txs = np.where(tree_mask)
TX0, TX1, TY0 = int(txs.min()), int(txs.max()), int(tys.min())
tree_rgba = np.zeros((a.shape[0], a.shape[1], 4), dtype=np.uint8)
tree_rgba[..., :3] = a
tree_rgba[..., 3] = np.where(tree_mask, 255, 0).astype(np.uint8)
tree_crop = Image.fromarray(tree_rgba, "RGBA").crop((TX0, TY0, TX1 + 1, int(tys.max()) + 1))
print(f"tip={TIP} bottom={Y_BOT} H={H}  tree crop={tree_crop.size}")

def place_tree(canvas, scale):
    w = int(tree_crop.size[0] * scale)
    h = int(tree_crop.size[1] * scale)
    t = tree_crop.resize((w, h), Image.LANCZOS)
    y = (Y_BOT - STROKE + 5) - h          # bottom sits just inside the shell stroke
    canvas.alpha_composite(t, (int(CX - w / 2), y))
    return canvas

def shell_points(M, p_top):
    pts = []
    for i in range(H + 1):
        t = i / H
        if t < T_M:
            u = (T_M - t) / T_M
            hw = M * (1 - u * u) ** p_top
        else:
            v = (t - T_M) / (1 - T_M)
            hw = M * (1 - v * v) ** 0.5
        pts.append((t, hw))
    return pts

def build(ratio, p_top, tree_scale, tag):
    M = ratio * H / 2.0
    S = 1024 * SS
    prof = shell_points(M, p_top)

    canvas = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(canvas)
    # centerline = outer profile inset by half the stroke; stroked with constant width
    cl_r, cl_l = [], []
    for t, hw in prof:
        y = (TIP + t * H) * SS
        hc = max(0.0, hw - STROKE / 2.0)
        cl_r.append((CX * SS + hc * SS, y))
        cl_l.append((CX * SS - hc * SS, y))
    poly = cl_r + cl_l[::-1]
    d.line(poly + [poly[0]], fill=SHELL_COLOR + (255,),
           width=max(1, int(STROKE * SS)), joint="curve")
    shell = canvas.resize((1024, 1024), Image.LANCZOS)
    out = place_tree(shell, tree_scale)

    # small stem hook at the tip
    hook = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    hd = ImageDraw.Draw(hook)
    lw = int(3.4 * SS)
    hd.line([(CX * SS, (TIP + 8) * SS), (CX * SS, (TIP - 16) * SS)],
            fill=SHELL_COLOR + (255,), width=lw)
    hd.arc([CX * SS - 3 * SS, (TIP - 30) * SS, CX * SS + 15 * SS, (TIP - 12) * SS],
           195, 350, fill=SHELL_COLOR + (255,), width=lw)
    out = Image.alpha_composite(out, hook.resize((1024, 1024), Image.LANCZOS))

    out.save(os.path.join(OUT, f"seed-shell-{tag}-mark.png"))
    prev = Image.new("RGB", (1024, 1024), bg)
    prev.paste(out, (0, 0), out)
    prev.crop((196, 76, 828, 820)).resize((440, 519), Image.LANCZOS) \
        .save(os.path.join(OUT, f"seed-shell-{tag}-preview.png"))
    return prev, out

VARIANTS = [
    ("1-subtle", 0.78, 0.80, 1.10),
    ("2-round", 0.86, 0.70, 1.12),
    ("3-egg", 0.94, 0.62, 1.14),
]

def make_icon(mark, tag):
    """Same rule as design/icons/build_icons.py: square crop around the mark, 72% of the tile."""
    bbox = mark.getbbox()
    cx, cy = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
    half = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) // 2 + 14
    crop = mark.crop((cx - half, cy - half, cx + half, cy + half))
    TILE, R = 1024, 0.72
    side = int(TILE * R)
    mm = crop.resize((side, side), Image.LANCZOS)
    layer = Image.new("RGBA", (TILE, TILE), (0, 0, 0, 0))
    layer.alpha_composite(mm, ((TILE - side) // 2, (TILE - side) // 2))
    tile = Image.alpha_composite(Image.new("RGBA", (TILE, TILE), bg + (255,)), layer)
    maskr = Image.new("L", (TILE, TILE), 0)
    ImageDraw.Draw(maskr).rounded_rectangle([0, 0, TILE - 1, TILE - 1], radius=int(TILE * 0.22), fill=255)
    tile = Image.composite(tile, Image.new("RGBA", (TILE, TILE), (0, 0, 0, 0)), maskr)
    out = tile.resize((512, 512), Image.LANCZOS)
    out.save(os.path.join(OUT, f"icon-{tag}.png"))
    return out

orig_prev = Image.open(SRC).convert("RGB").crop((196, 76, 828, 820)).resize((440, 519), Image.LANCZOS)
tiles = []
for tag, ratio, p_top, scale in VARIANTS:
    prev, mark = build(ratio, p_top, scale, tag)
    strip = Image.new("RGB", (1120, 620), bg)
    strip.paste(orig_prev, (80, 50))
    strip.paste(prev.crop((196, 76, 828, 820)).resize((440, 519), Image.LANCZOS), (600, 50))
    d = ImageDraw.Draw(strip)
    d.line([(560, 40), (560, 580)], fill=(165, 160, 145), width=2)
    strip.save(os.path.join(OUT, f"compare-{tag}.png"))
    tiles.append((tag, make_icon(mark, tag)))

# overview sheet: original icon | 3 variants
sheet = Image.new("RGBA", (4 * 520 + 40, 620), bg + (255,))
m_orig = np.abs(a - np.array(bg)).max(axis=2) > 18
m_orig[:150] = False; m_orig[740:] = False
orig_rgba = np.dstack([a.astype(np.uint8), np.where(m_orig, 255, 0).astype(np.uint8)])
orig_mark = Image.fromarray(orig_rgba, "RGBA")
ob = orig_mark.getbbox()
oh = max(ob[2] - ob[0], ob[3] - ob[1]) // 2 + 14
ocx, ocy = (ob[0] + ob[2]) // 2, (ob[1] + ob[3]) // 2
ocrop = orig_mark.crop((ocx - oh, ocy - oh, ocx + oh, ocy + oh))
side = int(1024 * 0.72)
omm = ocrop.resize((side, side), Image.LANCZOS)
olay = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
olay.alpha_composite(omm, ((1024 - side) // 2, (1024 - side) // 2))
otile = Image.alpha_composite(Image.new("RGBA", (1024, 1024), bg + (255,)), olay)
omk = Image.new("L", (1024, 1024), 0)
ImageDraw.Draw(omk).rounded_rectangle([0, 0, 1023, 1023], radius=225, fill=255)
otile = Image.composite(otile, Image.new("RGBA", (1024, 1024), (0, 0, 0, 0)), omk).resize((512, 512))
sheet.paste(otile, (20, 54), otile)
for i, (tag, t) in enumerate(tiles):
    sheet.paste(t, (20 + (i + 1) * 520, 54), t)
sheet.convert("RGB").save(os.path.join(OUT, "sheet-icons.png"))

print("done ->", OUT)