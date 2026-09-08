"""M4.R12 data-source canary: UltraData no_think vs simple_zh on the frozen
cascade harness.

Single-variable design: the same identity-generation child, the same A/B
protected lineage (simple_zh, the child's real origin), the same
predictive_update_scale=0.5 with consolidation disabled (that direction was
frozen in M4.R11), the same partition seeds and the same R7 budget — only
the C/C2/C3 continuation corpus changes to the converted UltraData no_think
records.  The simple_zh reference arm is not re-run: the R10 formal baseline
arm used the identical configuration and its metrics are joined by
reference.

Record disjointness holds automatically across corpora (disjoint source
texts produce disjoint sha256 record digests) and is asserted explicitly.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m2r1_phase_c_canary import (  # noqa: E402
    C2_PARTITION_SEED,
    C3_PARTITION_SEED,
    C_PARTITION_SEED,
    CHECKPOINT_OUTPUT_DIR,
    COHORT_SEEDS,
    _run_cascade_arm,
    build_disjoint_phase_chain,
    load_joint_child,
)
from taiji.foundation_training import FoundationTrainingDataset  # noqa: E402

FORMAT = "taiji-m4r12-data-source-canary-v1"
VERSION = 1
R10_BASELINE_REFERENCE = "reports/taiji_m4r10_rule_audit_formal_aggregate_20260908.json"


def _disjoint_checks(
    chain: Any,
    ultra_parts: dict[str, FoundationTrainingDataset],
) -> dict[str, bool]:
    lineage_digests: set[str] = set()
    for seed in COHORT_SEEDS:
        lineage_digests.update(chain.phase_a_by_seed[seed].selected_record_digests)
        lineage_digests.update(chain.phase_b_by_seed[seed].selected_record_digests)
    part_digests = {
        name: set(dataset.selected_record_digests) for name, dataset in ultra_parts.items()
    }
    names = sorted(part_digests)
    pairwise = {
        f"{left}_vs_{right}_disjoint": part_digests[left].isdisjoint(part_digests[right])
        for index, left in enumerate(names)
        for right in names[index + 1 :]
    }
    return {
        "ultra_parts_disjoint_from_lineage": all(
            part_digests[name].isdisjoint(lineage_digests) for name in names
        ),
        "ultra_parts_fully_selected": all(part_digests[name] for name in names),
        **pairwise,
    }


def _relative_to_project(path: Path) -> Path:
    """Normalize a corpus path to a project-relative form.

    ``FoundationTrainingDataset`` records ``str(path)`` inside its digest, so
    the same file yields a different dataset digest when referenced through
    an absolute path.  The child lineage was built with project-relative
    forward-slash paths; enforce the same convention here.
    """
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return path
    return relative


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--simple-zh-corpus",
        type=Path,
        default=Path("data") / "simple_zh" / "dialogue_extended_clean.jsonl",
    )
    parser.add_argument(
        "--ultradata-corpus",
        type=Path,
        required=True,
        help="Converted UltraData no_think corpus (foundation contract).",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train-bytes", type=int, default=65536)
    parser.add_argument("--eval-bytes", type=int, default=16384)
    parser.add_argument("--chunk-bytes", type=int, default=65536)
    parser.add_argument("--progress-root", type=Path, default=None)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    simple_zh_corpus = _relative_to_project(args.simple_zh_corpus)
    ultradata_corpus = _relative_to_project(args.ultradata_corpus)
    progress_root = args.progress_root or (
        PROJECT_ROOT / "output" / "taiji_m4r12_data_source_canary" / f"seed{args.seed}"
    )
    progress_root.mkdir(parents=True, exist_ok=True)

    joint_payload, source_model = load_joint_child(args.checkpoint, expected_seed=args.seed)
    lineage_seeds = tuple(dict.fromkeys((*COHORT_SEEDS, int(args.seed))))
    # A/B lineage is rebuilt from simple_zh — the child's real origin — and
    # must match the checkpoint lineage digests exactly.
    ab_chain = build_disjoint_phase_chain([simple_zh_corpus], cohort_seeds=lineage_seeds)
    phase_a = ab_chain.phase_a_by_seed[int(args.seed)]
    phase_b = ab_chain.phase_b_by_seed[int(args.seed)]
    lineage_checks = {
        "ab_protected_digest_matches_child": str(joint_payload.get("protected_dataset_digest"))
        == phase_a.digest,
        "ab_active_digest_matches_child": str(joint_payload.get("dataset_digest"))
        == phase_b.digest,
    }
    if not all(lineage_checks.values()):
        raise RuntimeError(
            "A/B lineage rebuilt from simple_zh does not match the child "
            f"checkpoint: {lineage_checks}"
        )

    ultra_parts: dict[str, FoundationTrainingDataset] = {
        "phase_c": FoundationTrainingDataset.from_jsonl(
            [ultradata_corpus],
            profile="foundation",
            partition_seed=C_PARTITION_SEED,
            track_record_digests=True,
        )
    }

    # The exclude_datasets contract requires the same ordered sources, so
    # C2/C3 only exclude the previous UltraData parts.  Disjointness against
    # the simple_zh A/B lineage holds structurally (disjoint source texts
    # produce disjoint record digests) and is asserted in _disjoint_checks.

    # Build C2/C3 sequentially (they exclude the previous parts).
    ultra_parts["phase_c2"] = FoundationTrainingDataset.from_jsonl(
        [ultradata_corpus],
        profile="foundation",
        partition_seed=C2_PARTITION_SEED,
        exclude_datasets=(ultra_parts["phase_c"],),
        track_record_digests=True,
    )
    ultra_parts["phase_c3"] = FoundationTrainingDataset.from_jsonl(
        [ultradata_corpus],
        profile="foundation",
        partition_seed=C3_PARTITION_SEED,
        exclude_datasets=(ultra_parts["phase_c"], ultra_parts["phase_c2"]),
        track_record_digests=True,
    )

    data_checks = _disjoint_checks(ab_chain, ultra_parts)
    data_checks.update(lineage_checks)
    data_checks["ultra_budgets_filled"] = all(
        len(part.train) == 1_048_576 and len(part.holdout) == 131_072
        for part in ultra_parts.values()
    )
    if not all(data_checks.values()):
        raise RuntimeError(f"M4.R12 data checks failed: {data_checks}")

    result = _run_cascade_arm(
        source_model=source_model,
        c_train=ultra_parts["phase_c"].train[: args.train_bytes],
        c2_train=ultra_parts["phase_c2"].train[: args.train_bytes],
        c3_train=ultra_parts["phase_c3"].train[: args.train_bytes],
        c_holdout=ultra_parts["phase_c"].holdout[: args.eval_bytes],
        c2_holdout=ultra_parts["phase_c2"].holdout[: args.eval_bytes],
        c3_holdout=ultra_parts["phase_c3"].holdout[: args.eval_bytes],
        a_retention=phase_a.retention[: args.eval_bytes],
        epochs=1,
        seed=int(args.seed),
        chunk_bytes=args.chunk_bytes,
        checkpoint_interval=1,
        progress_dir=progress_root,
        resume=False,
        consolidation_strength=0.0,
        predictive_update_scale=0.5,
    )
    # Archive the fixed-name final checkpoint per seed to avoid overwrite.
    final_name = f"seed{args.seed}_cascade_c{args.train_bytes}.pt"
    final_path = CHECKPOINT_OUTPUT_DIR / final_name
    candidate = progress_root / final_name
    if final_path.is_file():
        final_path.replace(candidate)
        result["round_trip"]["final_checkpoint_path"] = str(candidate)

    checks = dict(result["checks"])
    checks["ab_lineage_matches_child"] = all(lineage_checks.values())
    checks["ultra_data_chain_disjoint"] = all(
        v
        for k, v in data_checks.items()
        if k != "ab_protected_digest_matches_child" and k != "ab_active_digest_matches_child"
    )
    technical_gate_all_passed = all(bool(v) for v in checks.values())

    report = {
        "format": FORMAT,
        "version": VERSION,
        "generated_at_epoch": int(time.time()),
        "status": "passed" if technical_gate_all_passed else "failed",
        "can_promote": False,
        "seed": int(args.seed),
        "source_checkpoint": str(args.checkpoint),
        "source_checkpoint_digest": str(joint_payload.get("checkpoint_digest", "")),
        "simple_zh_corpus": str(simple_zh_corpus.as_posix()),
        "ultradata_corpus": str(ultradata_corpus.as_posix()),
        "configuration": {
            "profile": "foundation",
            "epochs": 1,
            "train_bytes": int(args.train_bytes),
            "eval_bytes": int(args.eval_bytes),
            "chunk_bytes": int(args.chunk_bytes),
            "predictive_update_scale": 0.5,
            "consolidation_strength": 0.0,
            "fixed_capacity": True,
            "partition_seeds": [C_PARTITION_SEED, C2_PARTITION_SEED, C3_PARTITION_SEED],
        },
        "data_checks": data_checks,
        "checks": checks,
        "technical_gate_all_passed": technical_gate_all_passed,
        "capability": result["capability"],
        "round_trip": result["round_trip"],
        "training": result["training"],
        "resources": {
            **result["resources"],
            "total_elapsed_seconds": time.perf_counter() - started,
        },
        "reference_arm": {
            "description": "M4.R10 formal baseline arm (identical config, simple_zh C/C2/C3)",
            "aggregate_report": R10_BASELINE_REFERENCE,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.report),
                "seed": int(args.seed),
                "technical_gate_all_passed": technical_gate_all_passed,
                "capability": result["capability"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if technical_gate_all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
