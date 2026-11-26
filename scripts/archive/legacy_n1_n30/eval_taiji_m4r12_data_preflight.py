"""M4.R12 data preflight gate (read-only).

Verifies that the converted UltraData no_think corpus satisfies the
foundation data contract before any pre-registered data-source comparison:
manifest digest stability, record well-formedness, byte-budget fill under
the foundation profile, and full record-disjoint phase-chain construction
(A/B lineage + C/C2/C3) with the real cohort seeds.  No training happens
here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m2r1_phase_c_canary import (  # noqa: E402
    COHORT_SEEDS,
    build_disjoint_phase_chain,
)
from taiji.foundation_training import FoundationTrainingDataset  # noqa: E402

FORMAT = "taiji-m4r12-data-preflight-v1"
MIN_RECORDS = 1000


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    corpus_bytes = args.corpus.read_bytes()
    corpus_lines = corpus_bytes.decode("utf-8").splitlines()

    well_formed = 0
    malformed = 0
    prefix_ok = 0
    for line in corpus_lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        text = record.get("text") if isinstance(record, dict) else None
        if not isinstance(text, str) or not text.strip():
            malformed += 1
            continue
        well_formed += 1
        if text.startswith("问：") and "\n答：" in text:
            prefix_ok += 1

    computed_digest = hashlib.sha256("\n".join(corpus_lines).encode("utf-8")).hexdigest()
    single_dataset = FoundationTrainingDataset.from_jsonl(
        [args.corpus],
        profile="foundation",
        partition_seed=20260909,
        track_record_digests=True,
    )
    chain = build_disjoint_phase_chain([args.corpus], cohort_seeds=COHORT_SEEDS)

    checks = {
        "manifest_digest_matches_output": computed_digest == str(manifest["output_sha256"]),
        "record_count_at_least_1000": len(corpus_lines) >= MIN_RECORDS,
        "all_records_well_formed": malformed == 0 and well_formed == len(corpus_lines),
        "qa_prefix_style_present": prefix_ok == well_formed and prefix_ok > 0,
        "foundation_budget_filled": (
            len(single_dataset.train) == 1_048_576
            and len(single_dataset.holdout) == 131_072
            and len(single_dataset.retention) == 131_072
        ),
        "phase_chain_record_disjoint": all(
            len(chain.phase_a_by_seed[seed].selected_record_digests) > 0
            and len(chain.phase_b_by_seed[seed].selected_record_digests) > 0
            for seed in COHORT_SEEDS
        )
        and len(chain.phase_c.selected_record_digests) > 0
        and len(chain.phase_c2.selected_record_digests) > 0
        and len(chain.phase_c3.selected_record_digests) > 0,
        "phase_chain_budgets_filled": all(
            len(partition.train) == 1_048_576 and len(partition.holdout) == 131_072
            for partition in (
                chain.phase_c,
                chain.phase_c2,
                chain.phase_c3,
            )
        ),
    }
    technical_gate_all_passed = all(bool(v) for v in checks.values())

    report = {
        "format": FORMAT,
        "version": 1,
        "generated_at_epoch": int(time.time()),
        "status": "passed" if technical_gate_all_passed else "failed",
        "can_promote": False,
        "corpus": str(args.corpus.as_posix()),
        "manifest": str(args.manifest.as_posix()),
        "manifest_output_sha256": str(manifest["output_sha256"]),
        "computed_output_sha256": computed_digest,
        "records": {
            "total": len(corpus_lines),
            "well_formed": well_formed,
            "malformed": malformed,
            "qa_prefix": prefix_ok,
        },
        "single_dataset_probe": {
            "train_bytes": len(single_dataset.train),
            "holdout_bytes": len(single_dataset.holdout),
            "retention_bytes": len(single_dataset.retention),
            "selected_records": len(single_dataset.selected_record_digests),
        },
        "phase_chain_probe": {
            "phase_c_selected_records": len(chain.phase_c.selected_record_digests),
            "phase_c2_selected_records": len(chain.phase_c2.selected_record_digests),
            "phase_c3_selected_records": len(chain.phase_c3.selected_record_digests),
            "phase_c_train_bytes": len(chain.phase_c.train),
            "phase_c_holdout_bytes": len(chain.phase_c.holdout),
        },
        "checks": checks,
        "technical_gate_all_passed": technical_gate_all_passed,
        "resources": {"total_elapsed_seconds": time.perf_counter() - started},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.report),
                "technical_gate_all_passed": technical_gate_all_passed,
                "checks": checks,
                "records": report["records"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if technical_gate_all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
