"""P5.1 sourced-knowledge content-transfer Gate runner.

Two same-budget corpora are internalized through the E4 governance boundary
(``SkillArtifactAdapter`` -> admitted -> ``ArtifactInternalizationTrainer``):
the ``sourced`` arm shares the holdout tasks' workflow family, the
``placebo`` arm uses a capability family with zero vocabulary overlap with
the holdout tasks.  The frozen gate is the between-arm content-transfer
margin on unseen-task procedural accuracy (next-action prediction), plus
the E4 mechanical gates (checkpoint roundtrip, quarantine rejection,
semantic internalization) and the budget/vocabulary assertions.

Preregistration: ``plans/reference
/M5_P5_1_SOURCED_KNOWLEDGE_TRANSFER_PREREGISTRATION_20260912.md``.
``growth_admitted=false`` and ``can_promote=false`` throughout.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.probe_taiji_p5_1_sourced_content_transfer import (  # noqa: E402
    FAMILY_A,
    FAMILY_B,
    _family_arm,
)
from taiji import ArtifactInternalizationTrainer  # noqa: E402
from taiji.internalization import content_digest  # noqa: E402

REPORT_FORMAT = "taiji-p5-1-sourced-knowledge-transfer-v1"
VERSION = 1
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_p5_1_sourced_knowledge_transfer_20260912.json"
FROZEN_MARGIN = 0.15
TOTAL_SECONDS_CAP = 300.0
SOURCED_SOURCES = (
    "skill.p51.sourced.a",
    "skill.p51.sourced.b",
    "skill.p51.sourced.c",
)
PLACEBO_SOURCES = (
    "skill.p51.placebo.a",
    "skill.p51.placebo.b",
    "skill.p51.placebo.c",
)


def _capability_vocabulary(artifacts) -> set[str]:
    """The workflow vocabulary that drives family transfer (procedure steps)."""

    vocabulary: set[str] = set()
    for artifact in artifacts:
        if artifact.unit_kind != "procedure":
            continue
        steps = artifact.content.get("steps")
        if isinstance(steps, list):
            for step in steps:
                if isinstance(step, dict) and step.get("action_kind"):
                    vocabulary.add(str(step["action_kind"]))
    return vocabulary


def _quarantine_rejected(train_artifacts, train_experiences, holdout, retention) -> bool:
    quarantined = train_artifacts[0].with_status("quarantined")
    try:
        ArtifactInternalizationTrainer(
            feature_dim=64,
            procedural_hidden_dim=16,
            affordance_feature_dim=12,
            seed=17,
        ).consolidate(
            (quarantined,),
            holdout_artifacts=holdout[0],
            retention_artifacts=retention[0],
            train_experiences=train_experiences,
            holdout_experiences=holdout[1],
            retention_experiences=retention[1],
        )
    except ValueError as exc:
        return "admitted" in str(exc)
    return False


def _arm_report(train_artifacts, train_experiences, holdout, retention) -> dict[str, Any]:
    trainer = ArtifactInternalizationTrainer(
        feature_dim=64,
        procedural_hidden_dim=16,
        affordance_feature_dim=12,
        seed=17,
        semantic_passes=12,
        procedural_epochs=250,
        affordance_epochs=200,
    )
    report = trainer.consolidate(
        train_artifacts,
        holdout_artifacts=holdout[0],
        retention_artifacts=retention[0],
        train_experiences=train_experiences,
        holdout_experiences=holdout[1],
        retention_experiences=retention[1],
    )
    restored = ArtifactInternalizationTrainer.from_checkpoint(trainer.checkpoint())
    return {
        "holdout_accuracy": report.procedural_holdout_accuracy,
        "lesion_holdout_accuracy": report.procedural_lesion_holdout_accuracy,
        "retention_accuracy": report.procedural_retention_accuracy,
        "train_accuracy": report.procedural_train_accuracy,
        "semantic_internalized": bool(report.semantic.passed),
        "checkpoint_roundtrip": content_digest(restored.checkpoint())
        == content_digest(trainer.checkpoint()),
    }


def run_gate() -> dict[str, Any]:
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "version": VERSION,
        "status": "failed",
        "sealed_payload_read": False,
        "growth_admitted": False,
        "can_promote": False,
        "frozen_margin": FROZEN_MARGIN,
    }
    try:
        holdout = _family_arm(
            FAMILY_A,
            sources=("skill.p51.holdout.a", "skill.p51.holdout.b"),
            partition="holdout",
            target="workspace-holdout",
        )
        retention = _family_arm(
            FAMILY_A,
            sources=("skill.p51.retention.a", "skill.p51.retention.b"),
            partition="retention",
            target="workspace-retention",
        )
        sourced = _family_arm(
            FAMILY_A,
            sources=SOURCED_SOURCES,
            partition="train",
            target="workspace-train",
        )
        placebo = _family_arm(
            FAMILY_B,
            sources=PLACEBO_SOURCES,
            partition="train",
            target="network-train",
        )

        # ---- Frozen assertion: capability vocabulary disjointness.
        holdout_vocab = _capability_vocabulary(holdout[0])
        sourced_vocab = _capability_vocabulary(sourced[0])
        placebo_vocab = _capability_vocabulary(placebo[0])
        vocabulary_gate = {
            "holdout_vocabulary": sorted(holdout_vocab),
            "sourced_superset_of_holdout": holdout_vocab.issubset(sourced_vocab),
            "placebo_disjoint_from_holdout": not (placebo_vocab & holdout_vocab),
        }

        # ---- Frozen assertion: same budget, item by item.
        budget_gate = {
            "sourced_artifacts": len(sourced[0]),
            "placebo_artifacts": len(placebo[0]),
            "sourced_events": len(sourced[1]),
            "placebo_events": len(placebo[1]),
            "same_budget": len(sourced[0]) == len(placebo[0])
            and len(sourced[1]) == len(placebo[1]),
            "sourced_unit_kinds": sorted({item.unit_kind for item in sourced[0]}),
            "placebo_unit_kinds": sorted({item.unit_kind for item in placebo[0]}),
            "unit_kinds_match": sorted({item.unit_kind for item in sourced[0]})
            == sorted({item.unit_kind for item in placebo[0]}),
        }

        # ---- Arms (identical trainer configuration).
        sourced_report = _arm_report(sourced[0], sourced[1], holdout, retention)
        placebo_report = _arm_report(placebo[0], placebo[1], holdout, retention)
        content_delta = sourced_report["holdout_accuracy"] - placebo_report["holdout_accuracy"]

        # ---- E4 mechanical gate: quarantine rejection.
        quarantine_rejected = _quarantine_rejected(sourced[0], sourced[1], holdout, retention)

        gates = {
            "same_budget_enforced": bool(budget_gate["same_budget"])
            and bool(budget_gate["unit_kinds_match"]),
            "capability_vocabulary_disjoint": bool(
                vocabulary_gate["sourced_superset_of_holdout"]
                and vocabulary_gate["placebo_disjoint_from_holdout"]
            ),
            "sourced_transfer_beats_lesion": sourced_report["holdout_accuracy"]
            > sourced_report["lesion_holdout_accuracy"],
            "content_transfer_margin": content_delta >= FROZEN_MARGIN,
            "checkpoint_roundtrip_both_arms": sourced_report["checkpoint_roundtrip"]
            and placebo_report["checkpoint_roundtrip"],
            "quarantined_artifact_rejected": quarantine_rejected,
            "semantic_internalized_both_arms": sourced_report["semantic_internalized"]
            and placebo_report["semantic_internalized"],
        }
        total_wall = time.perf_counter() - started
        budget_gate["total_wall_seconds"] = round(total_wall, 3)
        budget_gate["wall_within_cap"] = total_wall <= TOTAL_SECONDS_CAP
        gates["resource_within_cap"] = budget_gate["wall_within_cap"]

        all_passed = all(gates.values())
        if all_passed:
            outcome = "sourced_knowledge_transfer_supported"
            status = "completed"
        elif all(
            gates[key]
            for key in (
                "same_budget_enforced",
                "capability_vocabulary_disjoint",
                "checkpoint_roundtrip_both_arms",
                "quarantined_artifact_rejected",
                "semantic_internalized_both_arms",
                "resource_within_cap",
            )
        ):
            outcome = "sourced_content_insufficient"
            status = "completed"
        else:
            outcome = "mechanical_failure"
            status = "failed"
        payload.update(
            {
                "status": status,
                "preregistration": (
                    "plans/reference/"
                    "M5_P5_1_SOURCED_KNOWLEDGE_TRANSFER_PREREGISTRATION_20260912.md"
                ),
                "vocabulary_gate": vocabulary_gate,
                "budget_gate": budget_gate,
                "sourced_arm": sourced_report,
                "placebo_arm": placebo_report,
                "content_delta": round(content_delta, 6),
                "gates": gates,
                "outcome": outcome,
                "experiment_passed": bool(outcome == "sourced_knowledge_transfer_supported"),
                "growth_admitted": False,
                "can_promote": False,
                "interpretation": (
                    (
                        f"completed: outcome={outcome}; same-budget content "
                        "transfer through the E4 governance boundary; the "
                        "mechanism carrier stays the five-class synthetic scope "
                        "and no structural growth is admitted"
                    )
                    if status == "completed"
                    else "failed: mechanical gates did not pass; nothing is admitted"
                ),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        )
    except Exception as exc:  # noqa: BLE001
        payload.update(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        )
    _write(payload, DEFAULT_REPORT)
    return payload


def _write(payload: dict[str, Any], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(report_path)


def main() -> int:
    result = run_gate()
    print(
        json.dumps(
            {
                "report": str(DEFAULT_REPORT),
                "status": result.get("status"),
                "outcome": result.get("outcome"),
                "experiment_passed": result.get("experiment_passed"),
                "content_delta": result.get("content_delta"),
                "error": result.get("error"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
