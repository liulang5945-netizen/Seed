"""Vectorise the Seed master mark into SVG (mono + colour) for in-app use.

Input : E:/Seed/design/variants/seed-shell-final-mark.png  (1024 RGBA, transparent bg)
Output: E:/Seed/design/logo/
          - seed-mark-mono.svg    24x24, currentColor, evenodd (in-app single colour)
          - seed-mark-color.svg   coloured (structural deep green + young leaf green)
        plus 16/24/48/128 PNG previews for QA.

Method: pixel-edge contour tracing (inside on the left) -> collinear merge -> RDP simplify.
No hand drawing: the shapes are the master artwork itself.
"""

import os

import numpy as np
from PIL import Image

SRC = r"E:/Seed/design/variants/seed-shell-final-mark.png"
OUT = r"E:/Seed/design/logo"
os.makedirs(OUT, exist_ok=True)
WORK = 320  # working resolution (mark ~160 px)
RDP_TOL = 0.9  # simplification tolerance in working px
STRUCT = (18, 74, 56)

im = Image.open(SRC).convert("RGBA")
a = np.asarray(im).astype(float)
al = a[..., 3]
dist = np.abs(a[..., :3] - np.array(STRUCT)).max(axis=2)
struct = (al > 128) & (dist <= 30)
foliage = (al > 128) & (dist > 30)
solid = struct | foliage
print("px solid/struct/foliage:", int(solid.sum()), int(struct.sum()), int(foliage.sum()))


def resize_mask(m, size):
    img = Image.fromarray((m * 255).astype(np.uint8), "L").resize((size, size), Image.LANCZOS)
    return np.asarray(img) >= 128


def trace(m):
    """Pixel-edge contour tracing. Returns list of loops as lists of (x, y) floats."""
    H, W = m.shape
    pad = np.zeros((H + 2, W + 2), bool)
    pad[1:-1, 1:-1] = m
    inside = pad
    segs = {}  # start point -> end point
    ys, xs = np.where(inside)
    for y, x in zip(ys, xs):
        if not inside[y - 1, x]:
            segs[(x, y)] = (x + 1, y)
        if not inside[y, x + 1]:
            segs[(x + 1, y)] = (x + 1, y + 1)
        if not inside[y + 1, x]:
            segs[(x + 1, y + 1)] = (x, y + 1)
        if not inside[y, x - 1]:
            segs[(x, y + 1)] = (x, y)
    loops, visited = [], set()
    for start in list(segs):
        if start in visited:
            continue
        loop, p = [], start
        while p not in visited:
            loop.append(p)
            visited.add(p)
            nxt = segs.get(p)
            if nxt is None or nxt == start:
                break
            p = nxt
        if len(loop) >= 4:
            loops.append(loop)
    return loops


def rdp(pts, eps):
    if len(pts) < 3:
        return pts
    (x0, y0), (x1, y1) = pts[0], pts[-1]
    dx, dy = x1 - x0, y1 - y0
    n = np.hypot(dx, dy) or 1.0
    dmax, idx = 0.0, 0
    for i in range(1, len(pts) - 1):
        x, y = pts[i]
        d = abs(dy * (x - x0) - dx * (y - y0)) / n
        if d > dmax:
            dmax, idx = d, i
    if dmax > eps:
        return rdp(pts[: idx + 1], eps)[:-1] + rdp(pts[idx:], eps)
    return [pts[0], pts[-1]]


def to_path(m, size, scale, eps=RDP_TOL):
    """Trace a boolean mask and return (path_d, area) in the 24x24 user space."""
    mm = resize_mask(m, size)
    out, area = [], 0.0
    for loop in trace(mm):
        pts = np.array(loop, dtype=float)
        # collinear merge
        keep = [0]
        for i in range(1, len(pts) - 1):
            a1 = pts[i] - pts[keep[-1]]
            a2 = pts[i + 1] - pts[i]
            if abs(a1[0] * a2[1] - a1[1] * a2[0]) > 1e-9:
                keep.append(i)
        pts = pts[keep]
        closed = np.vstack([pts, pts[0]])
        simp = np.array(rdp([tuple(p) for p in closed], eps))
        if len(simp) < 4:
            continue
        simp = simp[:-1]
        x = simp[:, 0] / size * 24.0
        y = simp[:, 1] / size * 24.0
        area += 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
        d = "M" + "L".join(f"{v:.2f}" for v in np.round(np.stack([x, y], axis=1).ravel(), 2))
        out.append(d + "Z")
    return "".join(out), area


struct_s = resize_mask(struct, WORK)
foliage_s = resize_mask(foliage, WORK)
solid_s = resize_mask(solid, WORK)

mono_d, mono_area = to_path(solid_s, WORK, 1.0)
struct_d, struct_area = to_path(struct_s, WORK, 1.0)
foliage_d, foliage_area = to_path(foliage_s, WORK, 1.0)
print(f"loops: mono area={mono_area:.1f} struct={struct_area:.1f} foliage={foliage_area:.1f}")
print(f"path chars: mono={len(mono_d)} struct={len(struct_d)} foliage={len(foliage_d)}")

MONO = f"""<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none">
  <path d="{mono_d}" fill="currentColor" fill-rule="evenodd"/>
</svg>
"""
COLOR = f"""<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none">
  <path d="{struct_d}" fill="#124A38" fill-rule="nonzero"/>
  <path d="{foliage_d}" fill="#AAD66A" fill-rule="nonzero"/>
</svg>
"""
for name, body in (
    ("seed-mark-mono.svg", MONO),
    ("seed-mark-color.svg", COLOR),
    ("seed-mark-mono.path.txt", mono_d),
):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as handle:
        handle.write(body)

# QA renders happen in node/sharp (see icons/rasterize step); here we only emit SVG.
print("svg written ->", OUT)
print("MONO_PATH_START")
print(mono_d)
print("MONO_PATH_END")
