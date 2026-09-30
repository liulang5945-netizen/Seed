"""Wire the measured iso-contour Seed mark into the app surfaces.

Replaces the v5 hand-written geometry (amber disc + three-circle tree) with the
traced paths emitted by design/trace_mark_skimage.py, which were accepted only
after every disagreeing cell fell inside the 2-pixel edge band.

Run from the repository root:  python design/wire_mark.py
Targets are resolved against ``taiji-harness/`` (the fork), assets against
``design/logo/`` — both derived from this file's location rather than a hardcoded
drive letter, so the script is portable across machines and checkouts.
"""

from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
FORK = REPO / "taiji-harness"
LOGO = HERE / "logo"


def _read(path: Path) -> str:
    with path.open(encoding="utf-8", newline="") as handle:
        return handle.read()


def _write(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)


MONO = _read(LOGO / "seed-mark-mono.path.txt").strip()
COLOR_SVG = _read(LOGO / "seed-mark-color.svg")
STRUCT, FOLIAGE = re.findall(r'd="([^"]+)"', COLOR_SVG)
assert MONO and STRUCT and FOLIAGE, "traced paths missing"

OLD_DISC = "M23 12A11 11 0 1 1 1 12A11 11 0 1 1 23 12Z"
OLD_TREE = (
    "M17.2 8.4A5.2 5.2 0 1 1 6.8 8.4A5.2 5.2 0 1 1 17.2 8.4Z"
    "M11.4 11A3.7 3.7 0 1 1 4 11A3.7 3.7 0 1 1 11.4 11Z"
    "M20 11A3.7 3.7 0 1 1 12.6 11A3.7 3.7 0 1 1 20 11Z"
    "M13.2 13.6C13.3 16 13.4 18.2 13.5 20.4L10.5 20.4C10.6 18.2 10.7 16 10.8 13.6Z"
)
OLD_MONO = OLD_DISC + OLD_TREE

# ---- two-colour assets: disc -> structural green, tree -> young-leaf green
for rel in (
    "apps/web/public/favicon.svg",
    "website/public/favicon.svg",
    "apps/desktop/resources/icon.svg",
    "apps/desktop/resources/icon-windows.svg",
    "apps/desktop/resources/icon-macos.svg",
):
    path = FORK / rel
    text = _read(path)
    assert f'd="{OLD_DISC}"' in text and f'd="{OLD_TREE}"' in text, path
    text = text.replace(
        f'd="{OLD_DISC}" fill="#E8A33C"', f'd="{STRUCT}" fill="#124A38" fill-rule="evenodd"'
    )
    text = text.replace(
        f'd="{OLD_TREE}" fill="#3F8F45"', f'd="{FOLIAGE}" fill="#AAD66A" fill-rule="evenodd"'
    )
    _write(path, text)
    print(f"{rel}: colour layers traced")

# ---- single-colour assets: combined silhouette, even-odd knockout
for rel, colour in (
    ("apps/web/public/favicon-dark.svg", "#fff"),
    ("website/public/wordmark.svg", "currentColor"),
):
    path = FORK / rel
    text = _read(path)
    assert f'd="{OLD_MONO}"' in text, path
    text = text.replace(f'd="{OLD_MONO}"', f'd="{MONO}"')
    _write(path, text)
    print(f"{rel}: mono traced ({colour})")

# ---- the React constant
logo = FORK / "packages/client/ui-primitives/src/FishLogo.tsx"
text = _read(logo)
start = text.index("/** The round seed disc")
end = text.index("/**\n * Render the brand mark.")
new_block = """/**
 * The Seed mark: an egg-shaped seed shell drawn around a full-canopy tree.
 * Traced from the approved master (design/trace_mark_skimage.py) as even-odd
 * iso-contours, so `currentColor` paints the shell and the tree while the gap
 * between them stays transparent.
 */
export const FISH_LOGO_PATH = '{MONO}'

""".replace("{MONO}", MONO)
text = text[:start] + new_block + text[end:]
text = text.replace(
    "// Historical `Fish*` names: this module is the shared brand mark, now drawn as\n// the Seed mark — a tree inside a round seed. The old name is kept so the three\n// consumers (sidebar rail, hero, wordmark) and their imports stay stable.",
    "// Historical `Fish*` names: this module is the shared brand mark, drawn as the Seed\n// mark — a tree inside a seed shell. The old name is kept so the three consumers\n// (sidebar rail, hero, wordmark) and their imports stay stable.",
)
_write(logo, text)
print(f"{logo.name}: FISH_LOGO_PATH replaced, DISC/TREE constants retired")

wordmark = FORK / "packages/client/ui-primitives/src/BrandWordmark.tsx"
text = _read(wordmark)
old = '<path d={FISH_LOGO_PATH} fill="currentColor" />'
assert text.count(old) == 1
text = text.replace(old, '<path d={FISH_LOGO_PATH} fill="currentColor" fillRule="evenodd" />')
_write(wordmark, text)
print(f"{wordmark.name}: even-odd fill rule added")
