"""Render the GitHub social preview card (1280x640) for the Seed repository.

2026-09 redesign: the card now leads with the architecture identity -- the
Taiji wordmark and an ink emblem -- instead of a single kernel benchmark
number, matching the current README narrative (architecture -> capabilities
-> status).  The kernel facts are demoted to a footnote line and the
lesion-control culture sentence stays as the footer signature.

The emblem is a classic ink taiji whose halves carry a sparse network
(paper-colored nodes on the ink half, ink nodes on the paper half, one
seal-red apical node) -- the architecture in one mark.  Node placement is
seeded with seed 7, the committed benchmark seed, so the render is
deterministic.

Palette is sampled from the existing brand assets in frontend/public
(rice-paper white #FAFBF6, ink black #060604, seal red #B02A1E).

Usage:
    python scripts/make_social_preview.py
Output:
    frontend/public/social-preview.png

GitHub shows the social preview from the repository settings upload, so
after regenerating this asset the image must be re-uploaded once via
Settings -> General -> Social preview.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 640
SS = 2  # supersampling factor for crisp curves

PAPER = (250, 251, 246)
INK = (6, 6, 4)
INK_SOFT = (72, 72, 66)
INK_FAINT = (168, 168, 158)
INK_RULE = (206, 206, 196)
ACCENT = (176, 42, 30)

MARGIN = 96

FONT_DIR = Path("C:/Windows/Fonts")
BOLD = FONT_DIR / "segoeuib.ttf"
SEMI = FONT_DIR / "seguisb.ttf"
REG = FONT_DIR / "segoeui.ttf"
MONO = FONT_DIR / "consolab.ttf"

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "frontend" / "public" / "social-preview.png"

EMBLEM_CX, EMBLEM_CY, EMBLEM_R = 1024, 318, 160


def font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size * SS)


def s(v: float) -> int:
    return int(round(v * SS))


def in_ink(x: float, y: float) -> bool:
    """Classic taiji region test: which half of the emblem covers (x, y)."""

    half = EMBLEM_R / 2
    in_top = (x - EMBLEM_CX) ** 2 + (y - (EMBLEM_CY - half)) ** 2 <= half * half
    in_bot = (x - EMBLEM_CX) ** 2 + (y - (EMBLEM_CY + half)) ** 2 <= half * half
    return (x < EMBLEM_CX and not in_top) or in_bot


def draw_emblem(d: ImageDraw.ImageDraw) -> None:
    cx, cy, r = EMBLEM_CX, EMBLEM_CY, EMBLEM_R

    # outer ink ring with a hairline inner ring for craft
    d.ellipse([s(cx - r), s(cy - r), s(cx + r), s(cy + r)], outline=INK, width=3 * SS)
    inner = r - 9
    d.ellipse(
        [s(cx - inner), s(cy - inner), s(cx + inner), s(cy + inner)],
        outline=INK_RULE,
        width=max(1, SS),
    )

    # classic halves: left ink, top bump paper, bottom bump ink, opposing eyes
    d.pieslice([s(cx - r), s(cy - r), s(cx + r), s(cy + r)], 90, 270, fill=INK)
    half = r / 2
    d.ellipse([s(cx - half), s(cy - r), s(cx + half), s(cy)], fill=PAPER)
    d.ellipse([s(cx - half), s(cy), s(cx + half), s(cy + r)], fill=INK)
    eye = r / 7
    d.ellipse([s(cx - eye), s(cy - half - eye), s(cx + eye), s(cy - half + eye)], fill=INK)
    d.ellipse([s(cx - eye), s(cy + half - eye), s(cx + eye), s(cy + half + eye)], fill=PAPER)

    # sparse network: seeded with the committed benchmark seed (7)
    rng = random.Random(7)
    nodes: list[tuple[float, float]] = []
    guard = 0
    while len(nodes) < 26 and guard < 6000:
        guard += 1
        ang = rng.uniform(0.0, 2.0 * math.pi)
        rad = math.sqrt(rng.uniform(0.0, 1.0)) * (r - 30)
        x, y = cx + rad * math.cos(ang), cy + rad * math.sin(ang)
        if (x - cx) ** 2 + (y - cy) ** 2 > (r - 26) ** 2:
            continue
        if any((x - nx) ** 2 + (y - ny) ** 2 < 30**2 for nx, ny in nodes):
            continue
        nodes.append((x, y))

    pairs: set[tuple[int, int]] = set()
    for i, (x, y) in enumerate(nodes):
        order = sorted(
            (j for j in range(len(nodes)) if j != i),
            key=lambda j: (nodes[j][0] - x) ** 2 + (nodes[j][1] - y) ** 2,
        )
        for j in order[:2]:
            pairs.add((min(i, j), max(i, j)))

    apex = min(range(len(nodes)), key=lambda j: nodes[j][1])
    apex_pairs = {pair for pair in pairs if apex in pair}

    for i, j in sorted(pairs):
        (x1, y1), (x2, y2) = nodes[i], nodes[j]
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        color = ACCENT if (i, j) in apex_pairs else (PAPER if in_ink(mx, my) else INK)
        d.line([s(x1), s(y1), s(x2), s(y2)], fill=color, width=max(1, SS))

    for j, (x, y) in enumerate(nodes):
        if j == apex:
            d.ellipse([s(x - 7), s(y - 7), s(x + 7), s(y + 7)], fill=ACCENT)
        else:
            d.ellipse(
                [s(x - 4), s(y - 4), s(x + 4), s(y + 4)],
                fill=PAPER if in_ink(x, y) else INK,
            )


def main() -> None:
    img = Image.new("RGB", (W * SS, H * SS), PAPER)
    d = ImageDraw.Draw(img)

    # top rule + eyebrow
    d.line([s(MARGIN), s(88), s(W - MARGIN), s(88)], fill=INK_RULE, width=max(1, SS))
    d.text(
        (s(MARGIN), s(50)),
        "SEED  \u00b7  TAIJI NATIVE COGNITIVE ARCHITECTURE",
        font=font(SEMI, 21),
        fill=INK_SOFT,
    )

    # wordmark + statement
    d.text((s(MARGIN), s(122)), "Taiji", font=font(BOLD, 140), fill=INK)
    d.text((s(MARGIN), s(300)), "a native cognitive architecture", font=font(SEMI, 40), fill=INK)
    d.line([s(MARGIN), s(364), s(MARGIN + 108), s(364)], fill=ACCENT, width=max(3, 3 * SS))

    # the three pillars -- self-evolution stays one pillar, not the headline
    f_pillar = font(REG, 24)
    pillar = "persistent state  \u00b7  causal body  \u00b7  self-evolving structure"
    d.text((s(MARGIN), s(386)), pillar, font=f_pillar, fill=INK_SOFT)

    # kernel facts demoted to a footnote line
    f_kernel = font(REG, 23)
    kernel = "no backprop  \u00b7  online 0% \u2192 94.12% byte-cycle accuracy"
    d.text((s(MARGIN), s(424)), kernel, font=f_kernel, fill=INK_FAINT)

    # text must not run under the emblem
    limit = EMBLEM_CX - EMBLEM_R - 40
    for text, f in ((pillar, f_pillar), (kernel, f_kernel)):
        if s(MARGIN) + d.textlength(text, font=f) > s(limit):
            raise SystemExit(f"support line collides with the emblem: {text!r}")

    draw_emblem(d)

    # footer
    d.line([s(MARGIN), s(548), s(W - MARGIN), s(548)], fill=INK_RULE, width=max(1, SS))
    f_url = font(MONO, 22)
    d.text((s(MARGIN), s(566)), "github.com/liulang5945-netizen/Seed", font=f_url, fill=INK)
    f_tail = font(REG, 20)
    tail = "every claim lesion-controlled  \u00b7  failures reported as failures"
    d.text(
        (s(W - MARGIN) - d.textlength(tail, font=f_tail), s(566)),
        tail,
        font=f_tail,
        fill=INK_SOFT,
    )
    if d.textlength("github.com/liulang5945-netizen/Seed", font=f_url) + d.textlength(
        tail, font=f_tail
    ) > s(W - 2 * MARGIN - 24):
        raise SystemExit("footer texts collide")

    # Assert every drawn glyph stays inside the safe margins, because Twitter,
    # Slack and WeChat each crop the card at a different aspect ratio.
    bbox = img.convert("L").point(lambda p: 255 if p < 230 else 0).getbbox()
    if bbox is None:
        raise SystemExit("nothing was drawn")
    left, top, right, bottom = (v / SS for v in bbox)
    if left < MARGIN - 2 or top < 40 or right > W - MARGIN + 2 or bottom > H - 30:
        raise SystemExit(
            f"content escapes the safe area: bbox=({left:.0f},{top:.0f},"
            f"{right:.0f},{bottom:.0f}) for {W}x{H} margin={MARGIN}"
        )

    img = img.resize((W, H), Image.LANCZOS)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT, "PNG", optimize=True)

    kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT.relative_to(ROOT)}  {img.size[0]}x{img.size[1]}  {kb:.1f} KB")
    print(f"ink bbox: ({left:.0f}, {top:.0f}) -> ({right:.0f}, {bottom:.0f})")
    if kb >= 1024:
        raise SystemExit("image exceeds GitHub's 1 MB social preview limit")


if __name__ == "__main__":
    main()
