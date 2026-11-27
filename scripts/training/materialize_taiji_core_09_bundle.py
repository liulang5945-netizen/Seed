"""Rebuild the 09.8 subject bundle on demand (clean checkouts and CI).

``output/taiji_core_09/subject_bundle.pt`` is ignored by ``*.pt``, so no clean
checkout has it.  The 2026-11-10 ruling (ledger D-20261110-08) found it
``regenerable_exact``: the declared construction path reproduces the sealed
bundle byte for byte.  This script is the CI-callable entry point for that
ruling; it is a script rather than a hand-typed command so the CI step and the
ledger can point at the same thing.

It replays only the deterministic prefix of ``verify_taiji_closeout_09.py`` up
to the ``save_subject`` call -- the declared eight updates and one career step --
and stops there.  The activity training that follows in the closeout runs after
the bundle is already on disk and does not feed it.

The produced file must be a loadable subject bundle; that is the only condition
this step enforces.  The sealed sha256 and byte count are printed as diagnostics,
because ledger D-20261111-03 moved equivalence to the readout determinism gate:
the same job rebuilds twice and compares the readouts, so a host that serialises
a different (but equivalent) file is not a failure here.  An existing file that
already matches the sealed identity is left untouched.
"""

from __future__ import annotations

import argparse
import sys
import time
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import scripts.training.verify_taiji_closeout_09 as closeout  # noqa: E402
from taiji.core_09_bundle import (  # noqa: E402
    BUNDLE_RELATIVE_PATH,
    EXPECTED_SHA256,
    mismatch,
)


def build(bundle_path: Path) -> str:
    """Write the bundle at ``bundle_path`` and return the file sha256.

    Every constant and helper is taken from the closeout producer itself, so the
    construction path cannot drift away from the sealed reading while a second
    copy of the seed, the step count or the learning rate is maintained here.
    """
    torch.set_num_threads(1)
    torch.manual_seed(closeout.SEED)
    parent = closeout.LivingCoreReference(closeout.pricing.PRINCIPLE).double()
    directions = closeout.bd.consensus_directions(
        deepcopy(parent),
        closeout.pricing.LONG,
        parts=closeout.PARTS,
        count=closeout.UNITS_PER_GROUP,
    )
    child, _plan = closeout.gc.birth(deepcopy(parent), directions, closeout.rule._credits(parent))
    provider = closeout.rule.open_gates(closeout.bd.RULE_NAME)
    optimizer = torch.optim.Adam(child.parameters(), lr=closeout.LR)
    closeout.development.declared_feedback_updates(
        child, closeout.pricing.LONG, optimizer, provider, steps=closeout.STEPS
    )
    with torch.no_grad():
        state, _ = closeout.resume_career(
            child,
            child.initial_state("taiji-core-09"),
            closeout.pricing.SHORT,
            0,
            provider,
            limit=1,
        )
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    return closeout.save_subject(
        bundle_path,
        child,
        state,
        career=closeout.pricing.SHORT,
        journal_cursor=1,
        optimizer=optimizer,
        eligibility=torch.zeros(closeout.pricing.PRINCIPLE.groups, dtype=torch.float64),
        journal_entries=[{"seq": 0, "kind": "closeout_seed"}],
        pending=None,
        growth_gates=[dict(provider(0))],
    )


def display(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=ROOT / "output" / "taiji_core_09",
        help="directory that receives subject_bundle.pt",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild even when the file on disk already matches the sealed identity",
    )
    args = parser.parse_args()
    bundle_path = args.bundle_dir / Path(BUNDLE_RELATIVE_PATH).name

    if not args.force:
        reason = mismatch(bundle_path)
        if reason is None:
            print(f"bundle already matches the sealed identity: {display(bundle_path)}")
            return 0

    started = time.perf_counter()
    written = build(bundle_path)
    wall_seconds = time.perf_counter() - started

    try:
        payload = torch.load(bundle_path, map_location="cpu", weights_only=True)
    except Exception as error:  # noqa: BLE001 - a corrupt rebuild must be reported as such
        print(f"rebuilt file is not loadable: {error}", file=sys.stderr)
        return 1
    if not isinstance(payload, dict) or not {"parameters", "optimizer_state", "rng_state"} <= set(
        payload
    ):
        print("rebuilt file is loadable but is not a subject bundle", file=sys.stderr)
        return 1

    reason = mismatch(bundle_path)
    seal = "sealed identity" if reason is None else "not the sealed identity"
    print(
        f"rebuilt {display(bundle_path)} in {wall_seconds:.1f}s "
        f"sha256={written} bytes={bundle_path.stat().st_size} ({seal})"
    )
    if reason is not None:
        print(
            f"diagnostic: {reason or f'sha256 {written} != {EXPECTED_SHA256}'}; "
            "equivalence is decided by the readout determinism gate",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
