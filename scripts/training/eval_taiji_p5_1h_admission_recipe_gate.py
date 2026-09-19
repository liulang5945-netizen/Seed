"""P5.1h admission recipe gate: replay-weight sweep on the frozen P5.1g corpus.

Contract: plans/reference/M5_P5_1H_ADMISSION_PACKAGE_DRAFT_20260919.md
(section 6 -- pre-registered admission lines, frozen 2026-09-19).

Single recipe knob: ``procedural_replay_weight`` (experience replay of the
retention partition during consolidation).  P5.1g left replay unused
(``replay_digest`` empty, sourced ranking pairs 0) and measured procedural
retention 0.4189 against the 0.5 admission line; the sweep tests whether the
replay recipe closes that gap WITHOUT moving any line.

Pre-registered lines (per recipe; never adjusted after runs):
  L1  procedural retention accuracy >= 0.5            (p51g line, unchanged)
  L2  independent-slice accuracy - 0.367 >= 0.15      (new disjoint a-gate
       slice, chained after the P5.1g a-gate; frozen per-tick majority margin)
  L3  semantic retention non-degradation              (after <= before)
  L4  lesion discipline (lesion holdout accuracy <= 0.05) + checkpoint
       roundtrip preserved

TRAINING BUDGET GATE: every recipe runs a real consolidation pass, so the
sweep is refused unless ``--budget-approved`` is passed; the approval string
is recorded verbatim in the report.  Without it the tool is a dry shell for
tests and static review only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))

from eval_taiji_p5_1f_real_corpus_same_budget_gate import (  # noqa: E402
    DocumentEmbedder,
    _MemoizedEmbedder,
    _sample_agate,
)
from eval_taiji_p5_1g_real_corpus_quota_budget_gate import (  # noqa: E402
    ADMISSION_REVISION as P51G_ADMISSION_REVISION,
)
from eval_taiji_p5_1g_real_corpus_quota_budget_gate import (  # noqa: E402
    ARM_COUNT,
    FROZEN_CALL_QUOTAS,
    FROZEN_TICK_MAJORITY_BASELINE,
    HOLDOUT_COUNT,
    SOURCED_PATH,
    TRAIN_COUNT,
    _accuracy,
    _build_arm_from_partitions,
    _checkpoint_roundtrip,
    _gate_records_g,
    _ranking_pairs,
    _sample_arm,
    _trainer_kwargs,
)

from taiji import ArtifactInternalizationTrainer  # noqa: E402
from taiji.artifact_internalization import SemanticArtifactKnowledgeEncoder  # noqa: E402

REPORT_FORMAT = "taiji-p5-1h-admission-recipe-report-v1"
VERSION = 1
CONTRACT = "plans/reference/M5_P5_1H_ADMISSION_PACKAGE_DRAFT_20260919.md"
P51G_PREREGISTRATION = "plans/reference/M5_P5_1G_REAL_CORPUS_QUOTA_BUDGET_PREREGISTRATION_20260912.md"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_p5_1h_admission_recipe_20260919.json"

#: Pre-registered lines (contract section 6 -- frozen 2026-09-19).
RETENTION_LINE = 0.5
INDEPENDENT_MARGIN = 0.15
FROZEN_TICK_MAJORITY = FROZEN_TICK_MAJORITY_BASELINE  # 0.367
LESION_CEILING = 0.05

DEFAULT_REPLAY_WEIGHTS = (0.0, 0.25, 0.5, 1.0)


def evaluate_lines(
    *,
    retention_accuracy: float,
    independent_accuracy: float,
    semantic_retention_before: float,
    semantic_retention_after: float,
    lesion_accuracy: float,
    roundtrip_preserved: bool,
) -> dict[str, Any]:
    """The four pre-registered lines (pure function; contract section 6)."""

    l1 = retention_accuracy >= RETENTION_LINE
    l2 = (independent_accuracy - FROZEN_TICK_MAJORITY) >= INDEPENDENT_MARGIN
    l3 = semantic_retention_after <= semantic_retention_before
    l4 = lesion_accuracy <= LESION_CEILING and bool(roundtrip_preserved)
    lines = {"L1_retention": l1, "L2_independent_transfer": l2, "L3_semantic_retention": l3, "L4_lesion_roundtrip": l4}
    return {
        "lines": lines,
        "all_pass": all(lines.values()),
        "thresholds": {
            "retention_line": RETENTION_LINE,
            "independent_accuracy_line": FROZEN_TICK_MAJORITY + INDEPENDENT_MARGIN,
            "lesion_ceiling": LESION_CEILING,
        },
    }


def _independent_slice(sourced_path: Path, train_vocabulary: frozenset[str], *, after_line: int):
    """New a-gate slice chained after the P5.1g a-gate (disjoint by line order,
    same AGATE_COUNT, same in-train-vocabulary pruning rule)."""

    return _sample_agate(sourced_path, train_vocabulary, after_line=after_line)


def run_sweep(
    *,
    replay_weights: tuple[float, ...] = DEFAULT_REPLAY_WEIGHTS,
    budget_approved: bool = False,
    budget_approval_note: str = "",
    use_memoization: bool = True,
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Sweep the replay weight over the frozen P5.1g corpus partitions.

    Refuses to touch any data unless ``budget_approved`` is True: every recipe
    runs a real consolidation pass, which is exactly the training budget the
    user approves separately (contract section 4).
    """

    if not budget_approved:
        raise SystemExit(
            "P5.1h sweep refused: the replay sweep runs real consolidation "
            "passes and requires the separately-approved training budget.  "
            "Pass --budget-approved only after the user grants it."
        )

    started = time.perf_counter()
    sourced_path = PROJECT_ROOT / SOURCED_PATH
    sourced_sample = _sample_arm(sourced_path)
    sourced_partitions = {
        "train": sourced_sample.trajectories[:TRAIN_COUNT],
        "holdout": sourced_sample.trajectories[TRAIN_COUNT : TRAIN_COUNT + HOLDOUT_COUNT],
        "retention": sourced_sample.trajectories[TRAIN_COUNT + HOLDOUT_COUNT : ARM_COUNT],
    }
    measured_calls = {
        name: sum(len(item.tool_calls) for item in sourced_partitions[name])
        for name in ("train", "holdout", "retention")
    }
    if measured_calls != FROZEN_CALL_QUOTAS:
        raise ValueError(
            f"sourced partition calls {measured_calls} != frozen quotas {FROZEN_CALL_QUOTAS}"
        )

    train_sourced = sourced_partitions["train"]
    train_vocabulary = frozenset(
        f"tool.{name}" for trajectory in train_sourced for name in trajectory.tool_calls
    )
    p51g_agate = _sample_agate(sourced_path, train_vocabulary, after_line=sourced_sample.last_line)
    independent = _independent_slice(sourced_path, train_vocabulary, after_line=p51g_agate.last_line)
    if set(t.line_index for t in p51g_agate.trajectories) & set(
        t.line_index for t in independent.trajectories
    ):
        raise ValueError("independent slice overlaps the P5.1g a-gate slice")

    raw_embedder = DocumentEmbedder()
    shared_embedder = _MemoizedEmbedder(raw_embedder) if use_memoization else raw_embedder

    per_recipe: list[dict[str, Any]] = []
    for weight in replay_weights:
        recipe_started = time.perf_counter()
        encoder = SemanticArtifactKnowledgeEncoder(embedder=shared_embedder)
        trainer = ArtifactInternalizationTrainer(**_trainer_kwargs(encoder))
        arm = _build_arm_from_partitions(sourced_partitions)
        examples = trainer._examples(*arm["train"])
        pairs = _ranking_pairs(examples)
        report = trainer.consolidate(
            arm["train"][0],
            holdout_artifacts=arm["holdout"][0],
            retention_artifacts=arm["retention"][0],
            train_experiences=arm["train"][1],
            holdout_experiences=arm["holdout"][1],
            retention_experiences=arm["retention"][1],
            ranking_pairs=pairs,
            procedural_replay_weight=float(weight),
        )
        # Measure on the adopted child readout itself (trainer.procedural is
        # the once-consolidated trial after _adopt); a separate _trial_learner
        # here would double-train and misreport the child.
        independent_records = _gate_records_g(independent, encoder=encoder)
        independent_accuracy = _accuracy(trainer.procedural, independent_records)
        roundtrip = _checkpoint_roundtrip(trainer, arm["train"][0][0])
        lines = evaluate_lines(
            retention_accuracy=float(report.procedural_retention_accuracy),
            independent_accuracy=float(independent_accuracy),
            semantic_retention_before=float(report.semantic.retention_loss_before),
            semantic_retention_after=float(report.semantic.retention_loss_after),
            lesion_accuracy=float(report.procedural_lesion_holdout_accuracy),
            roundtrip_preserved=bool(roundtrip["checkpoint_digest_preserved"]),
        )
        per_recipe.append(
            {
                "replay_weight": float(weight),
                "procedural_retention_accuracy": float(report.procedural_retention_accuracy),
                "procedural_holdout_accuracy": float(report.procedural_holdout_accuracy),
                "independent_accuracy": float(independent_accuracy),
                "semantic_retention_before": float(report.semantic.retention_loss_before),
                "semantic_retention_after": float(report.semantic.retention_loss_after),
                "lesion_accuracy": float(report.procedural_lesion_holdout_accuracy),
                "admitted_in_report": bool(report.admitted),
                "lines": lines,
                "elapsed_seconds": time.perf_counter() - recipe_started,
            }
        )

    qualified = [entry for entry in per_recipe if entry["lines"]["all_pass"]]
    qualified.sort(
        key=lambda entry: (-entry["procedural_retention_accuracy"], -entry["independent_accuracy"])
    )
    elapsed = time.perf_counter() - started
    report_payload: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "version": VERSION,
        "contract": CONTRACT,
        "preregistration": P51G_PREREGISTRATION,
        "budget_approval_note": budget_approval_note,
        "dataset": {
            "name": "openbmb/UltraData-SFT-Agent-2609",
            "license": "Apache-2.0",
        },
        "corpus_identity": {
            "sourced_path": SOURCED_PATH,
            "calls_quotas": FROZEN_CALL_QUOTAS,
            "admission_revision_inherited": P51G_ADMISSION_REVISION,
        },
        "recipe_grid": list(replay_weights),
        "independent_slice": {
            "records": len(independent.trajectories),
            "first_line": independent.first_line,
            "last_line": independent.last_line,
            "pruned_out_of_vocabulary": independent.pruned_out_of_vocabulary,
        },
        "per_recipe": per_recipe,
        "qualified_recipes": [entry["replay_weight"] for entry in qualified],
        "selected": qualified[0] if qualified else None,
        "elapsed_seconds": elapsed,
        "outcome": "passed" if qualified else "no_qualified_recipe",
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return report_payload


def run_adoption(
    *,
    replay_weight: float,
    budget_approved: bool = False,
    budget_approval_note: str = "",
    use_memoization: bool = True,
    checkpoint_path: Path | None = None,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Adoption execution: run the selected recipe once, persist the child
    checkpoint atomically, and write the p51h:admission manifest (or the
    rejected/rolled-back state when any pre-registered line fails)."""

    if not budget_approved:
        raise SystemExit(
            "P5.1h adoption refused: adoption runs a real consolidation pass "
            "and requires the separately-approved training budget."
        )
    import hashlib

    from taiji.persistence import atomic_save

    sourced_path = PROJECT_ROOT / SOURCED_PATH
    sourced_sample = _sample_arm(sourced_path)
    sourced_partitions = {
        "train": sourced_sample.trajectories[:TRAIN_COUNT],
        "holdout": sourced_sample.trajectories[TRAIN_COUNT : TRAIN_COUNT + HOLDOUT_COUNT],
        "retention": sourced_sample.trajectories[TRAIN_COUNT + HOLDOUT_COUNT : ARM_COUNT],
    }
    train_vocabulary = frozenset(
        f"tool.{name}" for trajectory in sourced_partitions["train"] for name in trajectory.tool_calls
    )
    p51g_agate = _sample_agate(sourced_path, train_vocabulary, after_line=sourced_sample.last_line)
    independent = _independent_slice(sourced_path, train_vocabulary, after_line=p51g_agate.last_line)

    encoder = SemanticArtifactKnowledgeEncoder(embedder=shared_embedder_for_adoption(use_memoization))
    trainer = ArtifactInternalizationTrainer(**_trainer_kwargs(encoder))
    arm = _build_arm_from_partitions(sourced_partitions)
    pairs = _ranking_pairs(trainer._examples(*arm["train"]))
    report = trainer.consolidate(
        arm["train"][0],
        holdout_artifacts=arm["holdout"][0],
        retention_artifacts=arm["retention"][0],
        train_experiences=arm["train"][1],
        holdout_experiences=arm["holdout"][1],
        retention_experiences=arm["retention"][1],
        ranking_pairs=pairs,
        procedural_replay_weight=float(replay_weight),
    )
    independent_records = _gate_records_g(independent, encoder=encoder)
    independent_accuracy = _accuracy(trainer.procedural, independent_records)
    roundtrip = _checkpoint_roundtrip(trainer, arm["train"][0][0])
    lines = evaluate_lines(
        retention_accuracy=float(report.procedural_retention_accuracy),
        independent_accuracy=float(independent_accuracy),
        semantic_retention_before=float(report.semantic.retention_loss_before),
        semantic_retention_after=float(report.semantic.retention_loss_after),
        lesion_accuracy=float(report.procedural_lesion_holdout_accuracy),
        roundtrip_preserved=bool(roundtrip["checkpoint_digest_preserved"]),
    )

    checkpoint_sha = None
    if lines["all_pass"] and checkpoint_path is not None:
        saved = atomic_save(trainer.checkpoint(), checkpoint_path)
        checkpoint_sha = hashlib.sha256(Path(saved).read_bytes()).hexdigest()
    manifest = {
        "format": "taiji-p5-1h-adoption-manifest-v1",
        "admission_revision": "p51h:admission",
        "contract": CONTRACT,
        "replay_weight": float(replay_weight),
        "budget_approval_note": budget_approval_note,
        "child_checkpoint_digest": str(report.child_checkpoint_digest),
        "parent_checkpoint_digest": str(report.parent_checkpoint_digest),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_sha256": checkpoint_sha,
        "metrics": {
            "procedural_retention_accuracy": float(report.procedural_retention_accuracy),
            "procedural_holdout_accuracy": float(report.procedural_holdout_accuracy),
            "independent_accuracy": float(independent_accuracy),
            "semantic_retention_before": float(report.semantic.retention_loss_before),
            "semantic_retention_after": float(report.semantic.retention_loss_after),
            "lesion_accuracy": float(report.procedural_lesion_holdout_accuracy),
        },
        "lines": lines,
        "status": "admitted" if lines["all_pass"] else "rolled_back",
        "growth_admitted": False,
        "can_promote": False,
    }
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return manifest


def shared_embedder_for_adoption(use_memoization: bool) -> Any:
    raw = DocumentEmbedder()
    return _MemoizedEmbedder(raw) if use_memoization else raw


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replay-weights",
        default="0,0.25,0.5,1.0",
        help="comma-separated procedural replay weights (0.0 = P5.1g control)",
    )
    parser.add_argument(
        "--budget-approved",
        action="store_true",
        help="record the user's training-budget approval; the sweep is refused without it",
    )
    parser.add_argument(
        "--budget-approval-note",
        default="user approved training budget for the P5.1h calibration sweep (2026-09-19)",
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--no-memoization", action="store_true")
    parser.add_argument(
        "--adopt-with",
        type=float,
        default=None,
        help="run the adoption execution for this replay weight instead of the sweep",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p5_1h_child_20260919.pt",
    )
    parser.add_argument(
        "--adoption-manifest",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p5_1h_adoption_manifest_20260919.json",
    )
    args = parser.parse_args()

    if args.adopt_with is not None:
        manifest = run_adoption(
            replay_weight=float(args.adopt_with),
            budget_approved=args.budget_approved,
            budget_approval_note=args.budget_approval_note,
            use_memoization=not args.no_memoization,
            checkpoint_path=args.checkpoint,
            manifest_path=args.adoption_manifest,
        )
        print(
            json.dumps(
                {
                    "status": manifest["status"],
                    "replay_weight": manifest["replay_weight"],
                    "retention": manifest["metrics"]["procedural_retention_accuracy"],
                    "independent": manifest["metrics"]["independent_accuracy"],
                    "checkpoint_sha256": manifest["checkpoint_sha256"],
                },
                ensure_ascii=False,
            )
        )
        return 0 if manifest["status"] == "admitted" else 1

    weights = tuple(float(item) for item in args.replay_weights.split(","))
    report = run_sweep(
        replay_weights=weights,
        budget_approved=args.budget_approved,
        budget_approval_note=args.budget_approval_note,
        use_memoization=not args.no_memoization,
        report_path=args.report,
    )
    print(
        json.dumps(
            {
                "outcome": report["outcome"],
                "qualified": report["qualified_recipes"],
                "retention_per_recipe": {
                    entry["replay_weight"]: round(entry["procedural_retention_accuracy"], 4)
                    for entry in report["per_recipe"]
                },
                "elapsed_seconds": round(report["elapsed_seconds"], 1),
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["outcome"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
