"""Vectorise the Seed master mark into SVG (mono + colour) for in-app use.

Input : E:/Seed/design/variants/seed-shell-final-mark.png  (1024 RGBA, transparent bg)
Output: E:/Seed/design/logo/
          - seed-mark-mono.svg    24x24, currentColor, evenodd (in-app single colour)
          - seed-mark-color.svg   coloured (structural deep green + young leaf green)
          - seed-mark-mono.path.txt  the mono path data (pasted into FishLogo.tsx)
          - preview-*.png         QA renders

Method: pixel-edge contour tracing -> collinear merge -> closed-loop RDP
(farthest-point split, so the simplification baseline is never degenerate) ->
bbox normalisation onto the 24x24 grid. No hand drawing: the shapes are the
master artwork itself.
"""
import os
import numpy as np
from PIL import Image

SRC = r"E:/Seed/design/variants/seed-shell-final-mark.png"
OUT = r"E:/Seed/design/logo"
os.makedirs(OUT, exist_ok=True)
WORK = 480            # working resolution for tracing
RDP_TOL = 1.1         # simplification tolerance in working px
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
    """Pixel-edge contour tracing with Eulerian edge consumption: every loop is
    guaranteed CLOSED (the walk only ends back at its start), so no chord artefacts.
    Returns a list of closed loops of integer corner points."""
    pad = np.zeros((m.shape[0] + 2, m.shape[1] + 2), bool)
    pad[1:-1, 1:-1] = m
    inside = pad
    segs = {}                     # corner -> list of unused outgoing edges
    ys, xs = np.where(inside)
    for y, x in zip(ys, xs):
        if not inside[y - 1, x]:   segs.setdefault((x, y), []).append((x + 1, y))
        if not inside[y, x + 1]:   segs.setdefault((x + 1, y), []).append((x + 1, y + 1))
        if not inside[y + 1, x]:   segs.setdefault((x + 1, y + 1), []).append((x, y + 1))
        if not inside[y, x - 1]:   segs.setdefault((x, y + 1), []).append((x, y))
    loops = []
    for start in list(segs):
        while segs.get(start):
            loop, p, prev_dir = [start], start, (1, 0)
            while True:
                cand = segs.get(p, [])
                if not cand:
                    break                      # cannot happen in a balanced graph
                if len(cand) == 1:
                    nxt = cand[0]
                else:
                    def turn(v):
                        cross = prev_dir[0] * v[1] - prev_dir[1] * v[0]
                        return np.arctan2(cross, prev_dir[0] * v[0] + prev_dir[1] * v[1])
                    nxt = min(cand, key=lambda e: turn((e[0] - p[0], e[1] - p[1])))
                cand.remove(nxt)               # consume the edge
                if nxt == start:
                    break                      # loop closed
                loop.append(nxt)
                prev_dir = (nxt[0] - p[0], nxt[1] - p[1])
                p = nxt
            if len(loop) >= 4:
                loops.append(loop)
    return loops

def rdp_open(pts, eps):
    """RDP on an OPEN polyline (first != last)."""
    if len(pts) < 3:
        return pts[:]
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
        left = rdp_open(pts[:idx + 1], eps)
        right = rdp_open(pts[idx:], eps)
        return left[:-1] + right
    return [pts[0], pts[-1]]

def rdp_closed(pts, eps):
    """RDP on a closed loop [p0, ..., pn, p0]: split at the point farthest from p0
    so both RDP baselines are non-degenerate."""
    arr = np.asarray(pts, dtype=float)
    d = np.hypot(arr[:, 0] - arr[0, 0], arr[:, 1] - arr[0, 1])
    far = int(d.argmax())
    if far == 0 or far == len(pts) - 1:
        far = len(pts) // 2
    half1 = rdp_open(pts[:far + 1], eps)
    half2 = rdp_open(pts[far:] + [pts[0]], eps)
    return half1[:-1] + half2[:-1]

def collinear_merge(loop):
    pts = np.array(loop, dtype=float)
    if len(pts) < 3:
        return pts
    keep = [0]
    for i in range(1, len(pts) - 1):
        a1 = pts[i] - pts[keep[-1]]
        a2 = pts[i + 1] - pts[i]
        if abs(a1[0] * a2[1] - a1[1] * a2[0]) > 1e-9:
            keep.append(i)
    keep.append(len(pts) - 1)
    return pts[keep]

def trace_paths(m):
    """Trace mask -> simplified loops in working coords."""
    mm = resize_mask(m, WORK)
    res = []
    for loop in trace(mm):
        pts = collinear_merge(loop)
        if len(pts) < 3:
            continue
        simp = np.array(rdp_closed([tuple(p) for p in pts], RDP_TOL))
        if len(simp) < 3:
            continue
        area = 0.5 * float(np.sum(simp[:, 0] * np.roll(simp[:, 1], -1)
                                  - np.roll(simp[:, 0], -1) * simp[:, 1]))
        if abs(area) < 2.0:      # working px^2; specks only
            continue
        res.append((simp, area))
    return res

def normalise_transform(loops):
    """Shared transform: map the union bbox onto the 24x24 grid, 1-unit padding, centred."""
    allpts = np.vstack([s for s, _ in loops])
    x0, y0 = allpts.min(axis=0); x1, y1 = allpts.max(axis=0)
    scale = 22.0 / max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    return scale, cx, cy

def apply_transform(loops, scale, cx, cy):
    return [(simp - [cx, cy]) * scale + 12.0 for simp, _ in loops]

def loops_to_d(loops):
    ds = []
    for p in loops:
        pts = np.round(p, 2)
        ds.append("M" + "L".join(f"{a:g} {b:g}" for a, b in pts) + "Z")
    return "".join(ds)

solid_loops = trace_paths(solid)
scale, cx, cy = normalise_transform(solid_loops)
mono_d = loops_to_d(apply_transform(solid_loops, scale, cx, cy))
struct_d = loops_to_d(apply_transform(trace_paths(struct), scale, cx, cy))
foliage_d = loops_to_d(apply_transform(trace_paths(foliage), scale, cx, cy))
print(f"path chars: mono={len(mono_d)} struct={len(struct_d)} foliage={len(foliage_d)}")

MONO = ('<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none">\n'
        f'  <path d="{mono_d}" fill="currentColor" fill-rule="evenodd"/>\n</svg>\n')
COLOR = ('<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none">\n'
         f'  <path d="{struct_d}" fill="#124A38" fill-rule="evenodd"/>\n'
         f'  <path d="{foliage_d}" fill="#AAD66A" fill-rule="evenodd"/>\n</svg>\n')
open(os.path.join(OUT, "seed-mark-mono.svg"), "w", encoding="utf-8").write(MONO)
open(os.path.join(OUT, "seed-mark-color.svg"), "w", encoding="utf-8").write(COLOR)
open(os.path.join(OUT, "seed-mark-mono.path.txt"), "w", encoding="utf-8").write(mono_d)

# QA raster via sharp (installed in .dsh-sbx2/icon-tools)
qa = ("const sharp = require('sharp');\n"
      "const fs = require('fs');\n"
      "const dir = 'E:/Seed/design/logo';\n"
      "(async () => {\n"
      "  for (const name of ['seed-mark-mono.svg', 'seed-mark-color.svg']) {\n"
      "    const svg = fs.readFileSync(dir + '/' + name);\n"
      "    for (const s of [24, 128]) {\n"
      "      const bg = name.includes('mono') ? { r: 240, g: 235, b: 223, alpha: 1 } : { r: 255, g: 255, b: 255, alpha: 0 };\n"
      "      await sharp(svg, { density: 300 }).resize(s, s).flatten({ background: bg })\n"
      "        .png().toFile(dir + '/preview-' + name.replace('.svg', '') + '-' + s + '.png');\n"
      "    }\n"
      "  }\n"
      "  console.log('qa done');\n"
      "})();\n")
open(os.path.join(OUT, "_qa.cjs"), "w").write(qa)
os.system('node "' + os.path.join(OUT, "_qa.cjs") + '"')
print("done ->", OUT)
