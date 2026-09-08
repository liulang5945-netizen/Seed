"""Audit the M2.R1 record-disjoint continuation data contract.

This audit builds the exact phase-A/phase-B source lineages for the evaluated
cohort, then verifies that the shared phase-C and phase-C' courses contain no
record selected by any source lineage.  It does not load or modify a model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m2r1_phase_c_canary import (  # noqa: E402
    COHORT_SEEDS,
    _phase_chain_metadata,
    build_disjoint_phase_chain,
)

DEFAULT_CORPUS = (
    Path("data") / "simple_zh" / "dialogue_extended_clean.jsonl",
)
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m2r1_data_contract_20260906.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", nargs="+", type=Path, default=list(DEFAULT_CORPUS))
    parser.add_argument("--cohort-seeds", nargs="+", type=int, default=list(COHORT_SEEDS))
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)

    chain = build_disjoint_phase_chain(
        tuple(args.corpus),
        cohort_seeds=tuple(args.cohort_seeds),
    )
    phase_chain = _phase_chain_metadata(chain)
    payload = {
        "format": "taiji-m2r1-data-contract-audit-v1",
        "version": 1,
        "status": "passed" if phase_chain["record_disjoint"] else "failed",
        "can_promote": False,
        "phase_chain": phase_chain,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "report": str(args.report),
                "status": payload["status"],
                "cohort_seeds": phase_chain["cohort_seeds"],
                "phase_c_digest": phase_chain["phase_c"]["digest"],
                "phase_c2_digest": phase_chain["phase_c2"]["digest"],
                "phase_c3_digest": phase_chain["phase_c3"]["digest"],
                "nonzero_overlaps": {
                    key: value
                    for key, value in phase_chain["overlap_counts"].items()
                    if value
                },
            },
            ensure_ascii=False,
        )
    )
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
