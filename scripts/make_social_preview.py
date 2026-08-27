"""Render the GitHub social preview card (1280x640) for the Seed repository.

Numbers are pinned to the committed benchmark reported in README.md:
byte-cycle accuracy 0% -> 94.12% on the two-region [64, 48] benchmark, seed 7.

Palette is sampled from the existing brand assets in frontend/public
(rice-paper white #FAFBF6, ink black #060604) so the card matches the
taiji ink identity instead of introducing a new look.

Usage:
    python scripts/make_social_preview.py
Output:
    frontend/public/social-preview.png
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 640
SS = 2  # supersampling factor for crisp curves

PAPER = (250, 251, 246)
INK = (6, 6, 4)
INK_SOFT = (72, 72, 66)
INK_FAINT = (168, 168, 158)
INK_RULE = (206, 206, 196)
WATERMARK = (237, 238, 231)
WATERMARK_EDGE = (228, 229, 221)
ACCENT = (176, 42, 30)

MARGIN = 100

FONT_DIR = Path("C:/Windows/Fonts")
BOLD = FONT_DIR / "segoeuib.ttf"
SEMI = FONT_DIR / "seguisb.ttf"
REG = FONT_DIR / "segoeui.ttf"
MONO = FONT_DIR / "consolab.ttf"

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "frontend" / "public" / "social-preview.png"


def font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size * SS)


def s(v: int) -> int:
    return v * SS


def draw_taiji(d: ImageDraw.ImageDraw, cx: int, cy: int, r: int) -> None:
    """Ink taiji mark, kept as a pale background watermark behind the text."""
    cx, cy, r = s(cx), s(cy), s(r)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=WATERMARK_EDGE, width=max(1, SS))
    d.pieslice([cx - r, cy - r, cx + r, cy + r], 90, 270, fill=WATERMARK)
    half = r // 2
    d.ellipse([cx - half, cy - r, cx + half, cy], fill=WATERMARK)
    d.ellipse([cx - half, cy, cx + half, cy + r], fill=PAPER)
    eye = r // 6
    d.ellipse([cx - eye, cy - half - eye, cx + eye, cy - half + eye], fill=PAPER)
    d.ellipse([cx - eye, cy + half - eye, cx + eye, cy + half + eye], fill=WATERMARK)


def main() -> None:
    img = Image.new("RGB", (W * SS, H * SS), PAPER)
    d = ImageDraw.Draw(img)

    # Headline metrics are measured first so the watermark can be placed in the
    # leftover space on the right instead of colliding with the numbers.
    f_hero = font(BOLD, 132)
    hero_parts = (("0%", INK_SOFT), ("  \u2192  ", INK_FAINT), ("94.12%", INK))
    hero_w = sum(d.textlength(t, font=f_hero) for t, _ in hero_parts)
    hero_right = s(MARGIN) + hero_w

    f_label = font(REG, 27)
    label = "byte-cycle accuracy  \u00b7  two-region [64, 48] benchmark  \u00b7  seed 7"
    text_right = max(hero_right, s(MARGIN) + d.textlength(label, font=f_label))

    gap = s(46)
    wm_r = min(s(150), max(s(70), (s(W) - gap - text_right) // 2))
    wm_cx = s(W) - s(MARGIN) - wm_r
    if wm_cx - wm_r > text_right + gap:
        draw_taiji(d, cx=wm_cx // SS, cy=300, r=wm_r // SS)

    # top rule + eyebrow
    d.line([s(MARGIN), s(96), s(W - MARGIN), s(96)], fill=INK_RULE, width=max(1, SS))
    f_brow = font(SEMI, 21)
    d.text((s(MARGIN), s(58)), "SEED  \u00b7  TAIJI NATIVE COGNITIVE ARCHITECTURE",
           font=f_brow, fill=INK_SOFT)

    # headline: the measured jump
    y_hero = s(150)
    x = s(MARGIN)
    for text, fill in hero_parts:
        d.text((x, y_hero), text, font=f_hero, fill=fill)
        x += d.textlength(text, font=f_hero)

    d.text((s(MARGIN), s(310)), label, font=f_label, fill=INK_SOFT)

    # the claim
    f_claim = font(BOLD, 54)
    d.text((s(MARGIN), s(360)), "no backprop  /  no attention", font=f_claim, fill=INK)

    d.line([s(MARGIN), s(440), s(MARGIN + 108), s(440)], fill=ACCENT, width=max(3, 3 * SS))

    # supporting facts
    f_fact = font(REG, 25)
    d.text((s(MARGIN), s(464)),
           "learns online from local prediction errors  \u00b7  sparse fixed-fan-in synapses",
           font=f_fact, fill=INK_SOFT)
    d.text((s(MARGIN), s(500)),
           "slot-free episodic memory  \u00b7  surprise 5.4041 \u2192 0.1069  (98.02% reduction)",
           font=f_fact, fill=INK_SOFT)

    # footer
    d.line([s(MARGIN), s(548), s(W - MARGIN), s(548)], fill=INK_RULE, width=max(1, SS))
    f_foot = font(MONO, 23)
    d.text((s(MARGIN), s(566)), "github.com/liulang5945-netizen/Seed", font=f_foot, fill=INK)
    f_foot_r = font(REG, 23)
    tail = "every claim backed by a lesion-controlled script"
    d.text((s(W - MARGIN) - d.textlength(tail, font=f_foot_r), s(566)),
           tail, font=f_foot_r, fill=INK_SOFT)

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
