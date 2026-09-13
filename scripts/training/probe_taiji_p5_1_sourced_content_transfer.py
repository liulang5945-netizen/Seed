"""P5.1 validation-only probe: does sourced-corpus content transfer?

Read-only numeric probe ahead of the P5.1 preregistration (P4.9 precedent:
the probe informs the frozen margin and arm design; it sets no gate).  Two
same-budget corpora are internalized through the E4 artifact boundary:

- ``sourced``: train-partition Skill artifacts whose workflow family matches
  the unseen holdout tasks (shared capability vocabulary);
- ``placebo``: identical count/structure/admission, but content from a
  capability family with zero vocabulary overlap with the holdout tasks.

The probe reports procedural holdout / lesion / retention accuracies for
both arms plus vocabulary-overlap assertions.  Nothing here is a gate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.verify_taiji_e4_artifact_internalization import (  # noqa: E402
    _admitted,
)
from seed_platform.evolution_adapters import SkillArtifactAdapter  # noqa: E402
from taiji import ArtifactInternalizationTrainer  # noqa: E402
from taiji.artifact_internalization import ArtifactKnowledgeEncoder  # noqa: E402
from taiji.internalization import content_digest  # noqa: E402

FAMILY_A = ("editor.open", "editor.read", "editor.inspect")
FAMILY_B = ("network.search", "index.scan", "cache.fetch")


def _skill_fixture(
    source_id: str,
    scope_id: str,
    *,
    partition: str,
    capabilities: tuple[str, ...],
    target: str,
) -> tuple[tuple, tuple]:
    projection = SkillArtifactAdapter().project(
        {
            "skill_id": source_id,
            "version": "1",
            "publisher": "seed",
            "scope_id": scope_id,
            "name": f"Workflow {source_id}",
            "description": f"Bounded workflow over {target} using " + ", ".join(capabilities) + ".",
            "instructions": [
                {"action_kind": capability, "target": target} for capability in capabilities
            ],
            "capabilities": list(capabilities),
            "constraints": ["read_only"],
        },
        partition=partition,
    )
    events = tuple(
        projection.project_event(
            {
                "event_id": f"{source_id}:step:{index}",
                "event_kind": "invoke",
                "status": "success",
                "success": True,
                "capability_id": capability,
                "episode_id": f"{source_id}:episode",
                "tick": index,
                "result": {"ok": True},
                "reward_components": {"quality": 1.0},
            },
            parent_checkpoint_digest="a" * 64 if partition == "train" else "b" * 64,
            partition=partition,
        )
        for index, capability in enumerate(capabilities, start=1)
    )
    return _admitted(projection, source_id), events


def _family_arm(
    family: tuple[str, ...],
    *,
    sources: tuple[str, ...],
    partition: str,
    target: str,
) -> tuple[tuple, tuple]:
    artifacts: list = []
    events: list = []
    for index, source_id in enumerate(sources):
        # Rotate the chain so corpora are not degenerate copies while the
        # capability vocabulary stays inside the family.
        capabilities = family[index % len(family) :] + family[: index % len(family)]
        artifact_batch, event_batch = _skill_fixture(
            source_id,
            f"{source_id}.scope",
            partition=partition,
            capabilities=capabilities,
            target=target,
        )
        artifacts.extend(artifact_batch)
        events.extend(event_batch)
    return tuple(artifacts), tuple(events)


def _vocabulary(artifacts) -> set[str]:
    encoder = ArtifactKnowledgeEncoder(feature_dim=64)
    tokens: set[str] = set()
    for artifact in artifacts:
        tokens.update(encoder._tokens(artifact))
    return tokens


def _arm_report(label: str, train_artifacts, train_experiences, holdout, retention):
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
        "label": label,
        "train_artifacts": len(train_artifacts),
        "holdout_accuracy": report.procedural_holdout_accuracy,
        "lesion_holdout_accuracy": report.procedural_lesion_holdout_accuracy,
        "retention_accuracy": report.procedural_retention_accuracy,
        "train_accuracy": report.procedural_train_accuracy,
        "semantic_internalized": bool(report.semantic.passed),
        "checkpoint_roundtrip": content_digest(restored.checkpoint())
        == content_digest(trainer.checkpoint()),
    }


def run_probe() -> dict:
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
        sources=("skill.p51.sourced.a", "skill.p51.sourced.b", "skill.p51.sourced.c"),
        partition="train",
        target="workspace-train",
    )
    placebo = _family_arm(
        FAMILY_B,
        sources=("skill.p51.placebo.a", "skill.p51.placebo.b", "skill.p51.placebo.c"),
        partition="train",
        target="network-train",
    )

    holdout_tokens = _vocabulary(holdout[0])
    sourced_tokens = _vocabulary(sourced[0])
    placebo_tokens = _vocabulary(placebo[0])
    overlap = {
        "holdout_tokens": len(holdout_tokens),
        "sourced_overlap_with_holdout": len(sourced_tokens & holdout_tokens),
        "placebo_overlap_with_holdout": len(placebo_tokens & holdout_tokens),
        "sourced_vs_placebo_shared": len(sourced_tokens & placebo_tokens),
    }

    budget = {
        "sourced_artifacts": len(sourced[0]),
        "placebo_artifacts": len(placebo[0]),
        "sourced_events": len(sourced[1]),
        "placebo_events": len(placebo[1]),
        "same_budget": len(sourced[0]) == len(placebo[0]) and len(sourced[1]) == len(placebo[1]),
    }

    sourced_report = _arm_report("sourced", sourced[0], sourced[1], holdout, retention)
    placebo_report = _arm_report("placebo", placebo[0], placebo[1], holdout, retention)
    delta = sourced_report["holdout_accuracy"] - placebo_report["holdout_accuracy"]
    return {
        "probe": "p5.1-sourced-content-transfer",
        "validation_only": True,
        "vocabulary_overlap": overlap,
        "budget": budget,
        "sourced": sourced_report,
        "placebo": placebo_report,
        "holdout_accuracy_delta": round(delta, 6),
        "notes": "probe only; informs the frozen margin, sets no gate",
    }


def main() -> int:
    result = run_probe()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
