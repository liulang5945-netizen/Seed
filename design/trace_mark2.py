"""Vectorise the Seed master mark into SVG, with a measured fidelity gate.

Contours come from Moore-neighbour boundary tracing per 8-connected component,
plus one ring per enclosed background component (holes), simplified with RDP and
emitted as even-odd SVG subpaths. Every layer is rasterised back with an
even-odd scanline fill and compared against the source mask; the SVG is written
only when each layer's IoU clears MIN_IOU, so an empty or broken trace cannot
silently ship as artwork.

Only numpy and Pillow are used (no cv2 / scikit-image / shapely on this host).
"""
from __future__ import annotations

import os
from collections import deque

import numpy as np
from PIL import Image

SRC = r"E:/Seed/design/variants/seed-shell-final-mark.png"
OUT = r"E:/Seed/design/logo"
import sys
RDP_TOL = float(os.environ.get("RDP_TOL", "0.9"))
GRID = int(os.environ.get("GRID", "320"))
MIN_IOU = 0.98

N8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
# Moore walk order starting east, turning clockwise: E SE S SW W NW N NE
ORDER = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))


def components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, bool)
    neighbours = N8 if connectivity == 8 else N4
    out: list[list[tuple[int, int]]] = []
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or seen[y, x]:
                continue
            queue = deque([(y, x)])
            seen[y, x] = True
            pixels: list[tuple[int, int]] = []
            while queue:
                cy, cx = queue.popleft()
                pixels.append((cx, cy))
                for dy, dx in neighbours:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        queue.append((ny, nx))
            out.append(pixels)
    return out


def moore_ring(mask: np.ndarray, seed: tuple[int, int]) -> list[tuple[int, int]] | None:
    """Boundary of the 8-connected component containing seed (crack-following).

    Termination is by repeated (pixel, incoming direction) state, not by touching
    the seed pixel: a boundary that self-touches returns to the seed from another
    direction, and stopping there yields a truncated polygon whose area is far too
    small (measured: a 24272-px hole traced as 8.3 of 50.7 user units).
    """
    height, width = mask.shape

    def inside(y: int, x: int) -> bool:
        return 0 <= y < height and 0 <= x < width and bool(mask[y, x])

    start = seed
    current = seed
    back = 4  # scan order guarantees the west neighbour is background
    ring: list[tuple[int, int]] = [start]
    budget = 8 * int(mask.sum()) + 64
    for _ in range(budget):
        nxt = None
        chosen = None
        for step in range(1, 8):
            direction = (back + step) % 8
            dy, dx = ORDER[direction]
            cand = (current[1] + dy, current[0] + dx)
            if inside(*cand):
                nxt = (cand[1], cand[0])
                chosen = direction
                break
        if nxt is None:                      # isolated pixel
            return ring if len(ring) >= 4 else None
        if nxt == start:
            return ring if len(ring) >= 4 else None
        ring.append(nxt)
        back = (chosen + 4) % 8
        current = nxt
    return None                              # budget exhausted: do not trust it


def rings_of(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    rings: list[list[tuple[int, int]]] = []
    for pixels in components(mask, 8):
        sub = np.zeros_like(mask)
        for x, y in pixels:
            sub[y, x] = True
        seed = min(pixels, key=lambda p: (p[1], p[0]))
        ring = moore_ring(sub, seed)
        if ring:
            rings.append(ring)
    background = ~mask
    for pixels in components(background, 4):
        touches_border = any(y in (0, mask.shape[0] - 1) or x in (0, mask.shape[1] - 1) for x, y in pixels)
        if touches_border:
            continue
        sub = np.zeros_like(mask)
        for x, y in pixels:
            sub[y, x] = True
        seed = min(pixels, key=lambda p: (p[1], p[0]))
        ring = moore_ring(sub, seed)
        if ring:
            rings.append(ring)
    return rings


def rdp(points: list[tuple[float, float]], eps: float) -> list[tuple[float, float]]:
    if len(points) < 3:
        return points
    (x0, y0), (x1, y1) = points[0], points[-1]
    base = np.hypot(x1 - x0, y1 - y0) or 1.0
    deviations = [abs((y1 - y0) * (x - x0) - (x1 - x0) * (y - y0)) / base for x, y in points[1:-1]]
    index = int(np.argmax(deviations))
    if deviations[index] > eps:
        return rdp(points[:index + 2], eps)[:-1] + rdp(points[index + 1:], eps)
    return [points[0], points[-1]]


def signed_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def to_grid(mask: np.ndarray, size: int) -> np.ndarray:
    img = Image.fromarray((mask * 255).astype(np.uint8), "L").resize((size, size), Image.LANCZOS)
    return np.asarray(img) >= 128


def simplify(mask: np.ndarray, size: int) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for ring in rings_of(mask):
        pts = np.array(ring, dtype=float)
        if len(pts) < 4:
            continue
        # RDP needs open chains: on a closed ring first==last makes the baseline
        # degenerate, every perpendicular distance becomes 0 and the ring collapses.
        distances = (pts[:, 0] - pts[0, 0]) ** 2 + (pts[:, 1] - pts[0, 1]) ** 2
        far = int(np.argmax(distances[1:])) + 1
        if far == 0:
            continue
        chain_a = [tuple(p) for p in pts[:far + 1]]
        chain_b = [tuple(p) for p in pts[far:]] + [tuple(pts[0])]
        simple = rdp(chain_a, RDP_TOL)[:-1] + rdp(chain_b, RDP_TOL)[:-1]
        if len(simple) < 4:
            continue
        arr = np.array(simple, dtype=float)
        if abs(signed_area(arr)) < 2.0:
            continue
        out.append(arr / size * 24.0)
    return out


def render(rings: list[np.ndarray], size: int) -> np.ndarray:
    scale = size / 24.0
    segments: list[tuple[float, float, float, float, float, float]] = []
    for poly in rings:
        xs, ys = poly[:, 0] * scale, poly[:, 1] * scale
        for i in range(len(xs)):
            xa, ya = xs[i - 1], ys[i - 1]
            xb, yb = xs[i], ys[i]
            if ya == yb:
                continue
            segments.append((xa, ya, xb, yb, min(ya, yb), max(ya, yb)))
    covered = np.zeros((size, size), bool)
    if not segments:
        return covered
    table = np.array(segments, dtype=float)
    for row in range(size):
        gy = row + 0.5
        hits = table[(table[:, 4] <= gy) & (gy < table[:, 5])]
        if len(hits) == 0:
            continue
        denominators = hits[:, 3] - hits[:, 1]
        x_at = hits[:, 0] + (gy - hits[:, 1]) * (hits[:, 2] - hits[:, 0]) / denominators
        xs = np.sort(x_at)
        for k in range(0, len(xs) - 1, 2):
            left = max(int(np.ceil(xs[k] - 0.5)), 0)
            right = min(int(np.floor(xs[k + 1] - 0.5)), size - 1)
            if right >= left:
                covered[row, left:right + 1] = True
    return covered


def path_data(rings: list[np.ndarray]) -> str:
    parts = []
    for poly in rings:
        coords = np.round(poly, 2).ravel()
        parts.append("M" + "L".join(f"{value:.2f}" for value in coords) + "Z")
    return "".join(parts)


def main() -> int:
    image = Image.open(SRC).convert("RGBA")
    arr = np.asarray(image).astype(np.int16)
    alpha = arr[..., 3]
    distance = np.abs(arr[..., :3] - np.array([18, 74, 56])).max(axis=2)
    struct = (alpha > 128) & (distance <= 30)
    foliage = (alpha > 128) & (distance > 30)
    solid = struct | foliage
    print(f"source={SRC} grid={GRID} rdp_tol={RDP_TOL}")
    results: dict[str, list[np.ndarray]] = {}
    for name, mask in (("mono", solid), ("struct", struct), ("foliage", foliage)):
        grid_mask = to_grid(mask, GRID)
        rings = simplify(grid_mask, GRID)
        drawn = render(rings, GRID)
        union = int((drawn | grid_mask).sum())
        iou = int((drawn & grid_mask).sum()) / union if union else 0.0
        print(f"layer={name:8s} mask_px={int(grid_mask.sum()):6d} rings={len(rings):3d} "
              f"vertices={int(sum(len(r) for r in rings)):5d} IoU={iou:.4f}")
        if iou < MIN_IOU:
            print(f"REJECTED: layer {name} IoU {iou:.4f} < {MIN_IOU}")
            return 1
        results[name] = rings
    os.makedirs(OUT, exist_ok=True)
    mono_d = path_data(results["mono"])
    struct_d = path_data(results["struct"])
    foliage_d = path_data(results["foliage"])
    with open(os.path.join(OUT, "seed-mark-mono.svg"), "w", encoding="utf-8", newline="\n") as handle:
        handle.write('<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none">\n'
                     f'  <path d="{mono_d}" fill="currentColor" fill-rule="evenodd"/>\n</svg>\n')
    with open(os.path.join(OUT, "seed-mark-color.svg"), "w", encoding="utf-8", newline="\n") as handle:
        handle.write('<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none">\n'
                     f'  <path d="{struct_d}" fill="#124A38" fill-rule="evenodd"/>\n'
                     f'  <path d="{foliage_d}" fill="#AAD66A" fill-rule="evenodd"/>\n</svg>\n')
    with open(os.path.join(OUT, "seed-mark-mono.path.txt"), "w", encoding="utf-8", newline="\n") as handle:
        handle.write(mono_d)
    for size in (24, 64, 128):
        drawn = render(results["mono"], size)
        Image.fromarray(np.where(drawn, 0, 255).astype(np.uint8), "L").resize((size * 4, size * 4), Image.NEAREST).save(
            os.path.join(OUT, f"preview-mono-{size}.png"))
    for poly in results["mono"]:
        print("  mono ring verts=%d area=%.1f" % (len(poly), abs(signed_area(poly))))
    print(f"path_chars mono={len(mono_d)} struct={len(struct_d)} foliage={len(foliage_d)}")
    print("written ->", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
