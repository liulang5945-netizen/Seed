"""Wire the measured iso-contour Seed mark into the app surfaces.

Replaces the v5 hand-written geometry (amber disc + three-circle tree) with the
traced paths emitted by design/trace_mark_skimage.py, which were accepted only
after every disagreeing cell fell inside the 2-pixel edge band.
"""
from __future__ import annotations

import re

MONO = open(r"E:/Seed/design/logo/seed-mark-mono.path.txt", encoding="utf-8").read().strip()
COLOR_SVG = open(r"E:/Seed/design/logo/seed-mark-color.svg", encoding="utf-8").read()
STRUCT, FOLIAGE = re.findall(r'd="([^"]+)"', COLOR_SVG)
assert MONO and STRUCT and FOLIAGE, "traced paths missing"

OLD_DISC = "M23 12A11 11 0 1 1 1 12A11 11 0 1 1 23 12Z"
OLD_TREE = ("M17.2 8.4A5.2 5.2 0 1 1 6.8 8.4A5.2 5.2 0 1 1 17.2 8.4Z"
            "M11.4 11A3.7 3.7 0 1 1 4 11A3.7 3.7 0 1 1 11.4 11Z"
            "M20 11A3.7 3.7 0 1 1 12.6 11A3.7 3.7 0 1 1 20 11Z"
            "M13.2 13.6C13.3 16 13.4 18.2 13.5 20.4L10.5 20.4C10.6 18.2 10.7 16 10.8 13.6Z")
OLD_MONO = OLD_DISC + OLD_TREE

# ---- two-colour assets: disc -> structural green, tree -> young-leaf green
for path in (
    "apps/web/public/favicon.svg",
    "website/public/favicon.svg",
    "apps/desktop/resources/icon.svg",
    "apps/desktop/resources/icon-windows.svg",
    "apps/desktop/resources/icon-macos.svg",
):
    text = open(path, encoding="utf-8", newline="").read()
    assert f'd="{OLD_DISC}"' in text and f'd="{OLD_TREE}"' in text, path
    text = text.replace(f'd="{OLD_DISC}" fill="#E8A33C"', f'd="{STRUCT}" fill="#124A38" fill-rule="evenodd"')
    text = text.replace(f'd="{OLD_TREE}" fill="#3F8F45"', f'd="{FOLIAGE}" fill="#AAD66A" fill-rule="evenodd"')
    open(path, "w", encoding="utf-8", newline="").write(text)
    print(f"{path}: colour layers traced")

# ---- single-colour assets: combined silhouette, even-odd knockout
for path, colour in (("apps/web/public/favicon-dark.svg", "#fff"), ("website/public/wordmark.svg", "currentColor")):
    text = open(path, encoding="utf-8", newline="").read()
    assert f'd="{OLD_MONO}"' in text, path
    text = text.replace(f'd="{OLD_MONO}"', f'd="{MONO}"')
    open(path, "w", encoding="utf-8", newline="").write(text)
    print(f"{path}: mono traced ({colour})")

# ---- the React constant
logo = "packages/client/ui-primitives/src/FishLogo.tsx"
text = open(logo, encoding="utf-8", newline="").read()
start = text.index("/** The round seed disc")
end = text.index("/**\n * Render the brand mark.")
new_block = '''/**
 * The Seed mark: an egg-shaped seed shell drawn around a full-canopy tree.
 * Traced from the approved master (design/trace_mark_skimage.py) as even-odd
 * iso-contours, so `currentColor` paints the shell and the tree while the gap
 * between them stays transparent.
 */
export const FISH_LOGO_PATH = '{MONO}'

'''.replace("{MONO}", MONO)
text = text[:start] + new_block + text[end:]
text = text.replace("// Historical `Fish*` names: this module is the shared brand mark, now drawn as\n// the Seed mark — a tree inside a round seed. The old name is kept so the three\n// consumers (sidebar rail, hero, wordmark) and their imports stay stable.",
                    "// Historical `Fish*` names: this module is the shared brand mark, drawn as the Seed\n// mark — a tree inside a seed shell. The old name is kept so the three consumers\n// (sidebar rail, hero, wordmark) and their imports stay stable.")
open(logo, "w", encoding="utf-8", newline="").write(text)
print(f"{logo}: FISH_LOGO_PATH replaced, DISC/TREE constants retired")

wordmark = "packages/client/ui-primitives/src/BrandWordmark.tsx"
text = open(wordmark, encoding="utf-8", newline="").read()
old = '<path d={FISH_LOGO_PATH} fill="currentColor" />'
assert text.count(old) == 1
text = text.replace(old, '<path d={FISH_LOGO_PATH} fill="currentColor" fillRule="evenodd" />')
open(wordmark, "w", encoding="utf-8", newline="").write(text)
print(f"{wordmark}: even-odd fill rule added")
