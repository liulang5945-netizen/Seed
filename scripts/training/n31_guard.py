"""N31 pre-flight guard: enforce pre-registration and confirmation-seed discipline.

Rules enforced (STAGE_GATES_AND_STOPPING §2.2-§2.3, §3.1; D-20261124-485):
  1. Every card N<stage>.<line>.<card> must have a pre-registration file under
     plans/reference/ whose name starts with DESIGN-N<stage>-<line>- and whose text
     contains the exact card id, before any training run.
  2. Exploration cards (card number 01-89) must not use confirmation seeds.
  3. A confirmation seed may be read once, on one line; afterwards it is void everywhere
     (D-20261125-503).  Reads are recorded in plans/manifests/n31_confirmation_seed_reads.json.

Usage from a training script (call before building any model):
    from scripts.training.n31_guard import preflight
    preflight("N31.1.02", seeds=[20261120, 20261121, 20261122])

CLI (exit code 0 = allowed):
    python scripts/training/n31_guard.py N31.1.02 --seeds 20261120,20261121,20261122
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "plans" / "reference"
READS = ROOT / "plans" / "manifests" / "n31_confirmation_seed_reads.json"
CONFIRMATION_SEEDS = frozenset(
    {20261201, 20261202, 20261203, 20261211, 20261212, 20261213}  # N31 (all void)
    | {20261301, 20261302, 20261303, 20261311, 20261312, 20261313}  # N32 (D-20261125-503)
    | {20261401, 20261402, 20261403, 20261411, 20261412, 20261413}  # N32.2 (D-20261125-506)
    | {20261501, 20261502, 20261503, 20261511, 20261512, 20261513}  # N33 (D-20261125-555)
    | {20261601, 20261602, 20261603, 20261611, 20261612, 20261613}  # N33.2 (D-20261125-579)
)
CARD_RE = re.compile(r"^N(\d+)\.(\d+)\.(\d{2})$")


class PreflightError(RuntimeError):
    pass


def parse_card(card: str) -> tuple[int, int, int]:
    m = CARD_RE.match(card)
    if not m:
        raise PreflightError(f"bad card id {card!r}; expected N<stage>.<line>.<two digits>")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def find_design(card: str) -> Path:
    stage, line, number = parse_card(card)
    prefix = f"DESIGN-N{stage}-{line}-"
    # strict (D-20261125-506): only the card's own design file counts, never the line card
    for path in sorted(REFERENCE.glob(f"{prefix}{number:02d}_*.md")):
        if card in path.read_text(encoding="utf-8", errors="ignore"):
            return path
    raise PreflightError(f"{card}: no pre-registration {prefix}{number:02d}_*.md mentioning {card}")


def _load_reads() -> list[dict]:
    if not READS.exists():
        return []
    data = json.loads(READS.read_text(encoding="utf-8"))
    return list(data.get("reads", []))


def preflight(card: str, seeds: list[int], *, record: bool = True) -> Path:
    stage, line, number = parse_card(card)
    design = find_design(card)
    confirmation = number >= 90
    used = sorted(set(seeds) & CONFIRMATION_SEEDS)
    if not confirmation and used:
        raise PreflightError(f"{card}: exploration card uses confirmation seeds {used}")
    if confirmation:
        if not used or len(used) != len(set(seeds)):
            raise PreflightError(f"{card}: confirmation run must use only confirmation seeds")
        line_key = f"N{stage}.{line}"
        reads = _load_reads()
        # core requirements §2.1 v4, rule 3: a seed read as confirmation on ANY line is void
        seen = {s for r in reads for s in r.get("seeds", [])}
        already = sorted(seen & set(seeds))
        if already:
            raise PreflightError(f"{card}: seeds {already} already read (void on every line)")
        if record:
            reads.append(
                {
                    "card": card,
                    "line": line_key,
                    "seeds": sorted(set(seeds)),
                    "design": design.name,
                    "utc": datetime.now(UTC).isoformat(),
                }
            )
            READS.parent.mkdir(parents=True, exist_ok=True)
            READS.write_text(
                json.dumps({"reads": reads}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )
    return design


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("card")
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--no-record", action="store_true")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    try:
        design = preflight(args.card, seeds, record=not args.no_record)
    except PreflightError as exc:
        print(f"PREFLIGHT FAIL: {exc}")
        return 2
    print(f"PREFLIGHT OK: {args.card} -> {design.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
