"""Audit the source_hashes bindings of every taiji core card in reports/.

Every card claims the code bytes that produced its numbers. Shared modules keep
changing (10.1b extended the world, 09.6.1 added lesion_trained, 09.6.2 added the
arrivals hook, ...), so a card written earlier can bind bytes that no longer exist
on disk. This audit does not judge the readings: it separates

* ``fresh``       -- every bound path still hashes to the stored value;
* ``superseded``  -- stale, but a later card refutes it: either the declared ``supersedes``
                     chain names a fresh successor, or a fresh ``*_v2_*`` sibling exists;
* ``unresolved``  -- stale with no re-issued card, so reproduction from today's
                     tree is impossible and the ledger has to say so.

v2 (2026-11-02): bindings are read from both places cards actually write them --
top-level ``source_hashes`` and ``identity.source_hashes``. The first version read only
the former, so every 10.21-line card reported ``no_hashes`` while it in fact binds code
bytes (10.21.1 binds eleven, one of which is stale). Supersession now also follows the
``supersedes`` field instead of relying on one filename shape.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

BUDGET_CPU_SECONDS = 120
BUDGET_WALL_SECONDS = 240


def hashes_of(card: dict) -> dict:
    """Cards bind code bytes in two places: a top-level ``source_hashes`` block, and an
    ``identity.source_hashes`` block (the 10.21 line writes the latter). Reading only the
    first one silently reports those cards as ``no_hashes`` -- which is how a stale binding
    on 10.21.1 went unseen."""
    found: dict[str, str] = {}
    for block in (card.get("source_hashes"), (card.get("identity") or {}).get("source_hashes")):
        if isinstance(block, dict):
            found.update(block)
    return found


def bound_stale(card: dict) -> list[str]:
    stale = []
    for relative, expected in hashes_of(card).items():
        path = ROOT / relative
        if not path.exists():
            stale.append(f"{relative} [missing]")
            continue
        current = hashlib.sha256(path.read_bytes()).hexdigest()
        if current != expected:
            stale.append(relative)
    return stale


def v2_sibling(path: Path) -> Path | None:
    stem = path.stem
    if "_v2_" in stem:
        return None
    parts = stem.split("_2026")
    if len(parts) != 2:
        return None
    candidate = path.with_name(f"{parts[0]}_v2_2026{parts[1]}{path.suffix}")
    return candidate if candidate.exists() else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("hash audit card is write-once")

    total_cpu, total_wall = time.process_time(), time.perf_counter()
    paths = [Path(item) for item in sorted(glob.glob(str(ROOT / "reports" / "taiji_core_*.json")))]
    loaded: dict[str, dict] = {}
    for path in paths:
        try:
            loaded[path.name] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
    # A card naming another card in ``supersedes`` is that card's declared successor,
    # whatever filename shape each author used. The `_v2_` shape match alone cannot see a
    # three-generation chain, which is how four already-refuted cards kept reading unresolved.
    declared: dict[str, str] = {}
    for name, card in loaded.items():
        target = card.get("supersedes")
        if isinstance(target, str) and target:
            declared[Path(target).name] = name

    rows = []
    for path in paths:
        name = path.name
        card = loaded.get(name)
        if card is None:
            rows.append({"card": name, "status": "unreadable", "error": "could not parse"})
            continue
        hashes = hashes_of(card)
        if not hashes:
            rows.append({"card": name, "status": "no_hashes", "bound": 0})
            continue
        stale = bound_stale(card)
        if not stale:
            rows.append({"card": name, "status": "fresh", "bound": len(hashes)})
            continue
        sibling = v2_sibling(path)
        successor = declared.get(name)
        successor_fresh = False
        if successor is not None:
            successor_fresh = not bound_stale(loaded[successor])
        elif sibling is not None:
            try:
                other = json.loads(sibling.read_text(encoding="utf-8"))
                successor_fresh = not bound_stale(other)
            except (OSError, json.JSONDecodeError):
                successor_fresh = False
        rows.append(
            {
                "card": name,
                "status": "superseded" if successor_fresh else "unresolved",
                "bound": len(hashes),
                "stale": stale,
                "v2_sibling": sibling.name if sibling is not None else None,
                "declared_successor": successor,
                "successor_fresh": successor_fresh,
            }
        )

    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    unresolved = [row["card"] for row in rows if row["status"] == "unresolved"]
    stale_paths = sorted(
        {path for row in rows for path in row.get("stale", []) if "[missing]" not in path}
    )

    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    if cpu_seconds > BUDGET_CPU_SECONDS or wall_seconds > BUDGET_WALL_SECONDS:
        status = "budget_stop"
    elif unresolved:
        status = "cards_stale_unresolved"
    elif any(row["status"] in ("superseded",) for row in rows):
        status = "cards_stale_superseded_only"
    else:
        status = "cards_fresh"

    report = {
        "schema": "taiji-card-hash-audit-v1",
        "supersedes": "reports/taiji_core_card_hash_audit_10_12_v2_20261101.json",
        "step": "10.12",
        "date": "2026-11-01",
        "status": status,
        "counts": counts,
        "unresolved_cards": unresolved,
        "stale_paths": stale_paths,
        "cards": rows,
        "limits": (
            "Bookkeeping only. A stale binding does not invalidate the reading: the card "
            "bound the bytes that existed when it ran. It does mean the number cannot be "
            "reproduced from today's tree, which is what 'unresolved' names. Nothing is "
            "re-run, re-hashed in place or deleted."
        ),
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "source_hashes": {
            "scripts/training/audit_taiji_card_hashes.py": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest()
        },
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    print(
        json.dumps(
            {
                "status": status,
                "counts": counts,
                "unresolved": len(unresolved),
                "stale_paths": len(stale_paths),
                "cpu_seconds": round(cpu_seconds, 3),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
