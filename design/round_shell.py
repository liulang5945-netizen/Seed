"""Seed mark: rounded shell + grown tree (auto-fitted) + fresh palette, trunk rooted into the shell.

Owner feedback (2026-09-28, round 2):
  1. mark too small inside the icon tile        -> tile MARK_RATIO 0.72 -> 0.84
  2. trunk not connected to the shell           -> flat trunk cut bridged into the shell stroke
  3. shell got rounder, so the tree should grow -> tree auto-fitted to the largest size that
                                                   still clears the shell stroke
  4. palette reads "old"                        -> recoloured to a fresh spring green

The tree is still the anchor's own artwork: only scaled and recoloured (luminance remap).
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
STRUCT = (18, 74, 56)  # #124A38  shell stroke + trunk + branches
LEAF_HI = (170, 214, 106)  # #AAD66A  young-leaf highlight

im = Image.open(SRC).convert("RGB")
a = np.asarray(im).astype(np.int16)
bg = tuple(int(v) for v in a[5, 5])
mask = np.abs(a - np.array(bg)).max(axis=2) > 18
band = np.zeros_like(mask)
band[150:740, :] = True
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
    outside[0, :] = ~m[0, :]
    outside[-1, :] = ~m[-1, :]
    outside[:, 0] = ~m[:, 0]
    outside[:, -1] = ~m[:, -1]
    while True:
        nxt = dilate(outside) & ~m
        if nxt.sum() == outside.sum():
            return ~outside
        outside = nxt


filled = fill_holes(mask)
tree_mask = mask & erode(filled, 9)
ys, xs = np.where(filled)
Y_TOP, Y_BOT = int(ys.min()), int(ys.max())
TIP = Y_TOP + 18
H = Y_BOT - TIP

# ---------- tree layer: crop + recolour ----------
tys, txs = np.where(tree_mask)
TX0, TX1, TY0 = int(txs.min()), int(txs.max()), int(tys.min())
lum = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
tl = lum[tree_mask]
Ld, Ll = float(np.percentile(tl, 3)), float(np.percentile(tl, 97))
t = np.clip((lum - Ld) / max(1.0, Ll - Ld), 0, 1) ** 0.85
rgb = np.zeros((a.shape[0], a.shape[1], 4), dtype=np.uint8)
for c in range(3):
    rgb[..., c] = (STRUCT[c] + t * (LEAF_HI[c] - STRUCT[c])).astype(np.uint8)
rgb[..., 3] = np.where(tree_mask, 255, 0).astype(np.uint8)
tree_crop = Image.fromarray(rgb, "RGBA").crop((TX0, TY0, TX1 + 1, int(tys.max()) + 1))
cw, ch = tree_crop.size
print(f"shell {TIP}..{Y_BOT} H={H} | tree crop {tree_crop.size} | lum {Ld:.0f}..{Ll:.0f}")


def shell_hw(tt, ratio, p_top):
    M = ratio * H / 2.0
    if tt < 0 or tt > 1:
        return -1.0
    if tt < T_M:
        u = (T_M - tt) / T_M
        return M * (1 - u * u) ** p_top
    v = (tt - T_M) / (1 - T_M)
    return M * (1 - v * v) ** 0.5


def fits(scale, stretch_x, ratio, p_top, margin):
    """True if the scaled tree stays inside the shell's inner edge (margin = clearance)."""
    w, h = int(round(cw * scale * stretch_x)), int(round(ch * scale))
    al = np.asarray(tree_crop.resize((w, h), Image.LANCZOS))[..., 3] > 40
    x0 = CX - w / 2.0
    y0 = (Y_BOT - STROKE + 2) - h
    for j in range(h):
        idx = np.where(al[j])[0]
        if len(idx) == 0:
            continue
        hw = shell_hw((y0 + j - TIP) / H, ratio, p_top) - STROKE / 2.0 - margin
        if hw <= 0:
            return False
        if (x0 + idx.min()) < CX - hw or (x0 + idx.max()) > CX + hw:
            return False
    return True


def best_fit(ratio, p_top, margin):
    best = None
    for sx in (1.18, 1.12, 1.06, 1.00):
        for sc in [1.30 - 0.02 * i for i in range(16)]:
            if fits(sc, sx, ratio, p_top, margin):
                if best is None or sc * sx > best[0] * best[1]:
                    best = (sc, sx)
                break
    return best


def build(ratio, p_top, scale, stretch_x, tag, marker_ratio=0.84):
    S = 1024 * SS
    canvas = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(canvas)
    rr, ll = [], []
    for i in range(H + 1):
        tt = i / H
        y = (TIP + tt * H) * SS
        hc = max(0.0, shell_hw(tt, ratio, p_top) - STROKE / 2.0)
        rr.append((CX * SS + hc * SS, y))
        ll.append((CX * SS - hc * SS, y))
    poly = rr + ll[::-1]
    d.line(poly + [poly[0]], fill=STRUCT + (255,), width=max(1, int(STROKE * SS)), joint="curve")
    out = canvas.resize((1024, 1024), Image.LANCZOS)

    # tree: grown + rooted into the shell stroke
    w, h = int(round(cw * scale * stretch_x)), int(round(ch * scale))
    tr = tree_crop.resize((w, h), Image.LANCZOS)
    x0, y0 = int(round(CX - w / 2.0)), int((Y_BOT - STROKE + 2) - h)
    out.alpha_composite(tr, (x0, y0))

    # bridge the flat trunk cut into the shell stroke (same colour => seamless)
    last = np.asarray(tr)[-1, :, 3]
    nz = np.where(last > 60)[0]
    if len(nz):
        conn = Image.new("RGBA", (S, S), (0, 0, 0, 0))
        A, B = x0 + int(nz.min()), x0 + int(nz.max())
        ImageDraw.Draw(conn).polygon(
            [
                (A * SS, (y0 + h - 3) * SS),
                (B * SS, (y0 + h - 3) * SS),
                ((B + 4) * SS, (Y_BOT - 4) * SS),
                ((A - 4) * SS, (Y_BOT - 4) * SS),
            ],
            fill=STRUCT + (255,),
        )
        out = Image.alpha_composite(out, conn.resize((1024, 1024), Image.LANCZOS))

    # stem hook at the tip
    hook = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    hd = ImageDraw.Draw(hook)
    lw = int(3.4 * SS)
    hd.line([(CX * SS, (TIP + 8) * SS), (CX * SS, (TIP - 16) * SS)], fill=STRUCT + (255,), width=lw)
    hd.arc(
        [CX * SS - 3 * SS, (TIP - 30) * SS, CX * SS + 15 * SS, (TIP - 12) * SS],
        195,
        350,
        fill=STRUCT + (255,),
        width=lw,
    )
    out = Image.alpha_composite(out, hook.resize((1024, 1024), Image.LANCZOS))
    out.save(os.path.join(OUT, f"seed-shell-{tag}-mark.png"))

    bbox = out.getbbox()
    cx, cy = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
    half = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) // 2 + 12
    sq = out.crop((cx - half, cy - half, cx + half, cy + half))
    TILE = 1024
    side = int(TILE * marker_ratio)
    tile = Image.new("RGBA", (TILE, TILE), bg + (255,))
    lay = Image.new("RGBA", (TILE, TILE), (0, 0, 0, 0))
    mm = sq.resize((side, side), Image.LANCZOS)
    lay.alpha_composite(mm, ((TILE - side) // 2, (TILE - side) // 2))
    tile = Image.alpha_composite(tile, lay)
    mk = Image.new("L", (TILE, TILE), 0)
    ImageDraw.Draw(mk).rounded_rectangle(
        [0, 0, TILE - 1, TILE - 1], radius=int(TILE * 0.22), fill=255
    )
    tile = Image.composite(tile, Image.new("RGBA", (TILE, TILE), (0, 0, 0, 0)), mk)
    tile.resize((512, 512), Image.LANCZOS).save(os.path.join(OUT, f"icon-{tag}.png"))
    prev = Image.new("RGB", (1024, 1024), bg)
    prev.paste(out, (0, 0), out)
    print(f"  {tag}: scale={scale:.2f} stretchX={stretch_x:.2f} treebox={w}x{h}")
    return prev, tile


o = Image.open(SRC).convert("RGB").crop((196, 76, 828, 820)).resize((440, 519), Image.LANCZOS)
tiles = []
for tag, margin in (("final", 16),):
    sc, sx = best_fit(0.94, 0.62, margin)
    prev, tile = build(0.94, 0.62, sc, sx, tag)
    tiles.append((tag, tile))
    strip = Image.new("RGB", (1120, 620), bg)
    strip.paste(o, (80, 50))
    strip.paste(prev.crop((196, 76, 828, 820)).resize((440, 519), Image.LANCZOS), (600, 50))
    ImageDraw.Draw(strip).line([(560, 40), (560, 580)], fill=(165, 160, 145), width=2)
    strip.save(os.path.join(OUT, f"compare-{tag}.png"))
    zz = Image.new("RGBA", (3 * 260 + 40, 300), bg + (255,))
    for i, s in enumerate((16, 24, 32)):
        sm = tile.resize((s, s), Image.LANCZOS).resize((s * 8, s * 8), Image.NEAREST)
        zz.paste(sm, (20 + i * 260, 20), sm)
    zz.convert("RGB").save(os.path.join(OUT, f"small-{tag}.png"))

sheet = Image.new("RGBA", (len(tiles) * 520 + 40, 620), bg + (255,))
for i, (_, t) in enumerate(tiles):
    tt = t.resize((512, 512), Image.LANCZOS)
    sheet.paste(tt, (20 + i * 520, 54), tt)
sheet.convert("RGB").save(os.path.join(OUT, "sheet-final.png"))
print("done ->", OUT)
