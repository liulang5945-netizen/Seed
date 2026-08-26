"""Evaluate Taiji concept capacity, interference and registry lesions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji import ConceptFormationOrgan, EpisodicMemoryRecord, Outcome  # noqa: E402

MANIFEST_FORMAT = "taiji-concept-capacity-manifest-v1"
REPORT_FORMAT = "taiji-concept-capacity-v1"


def _record(
    index: int,
    concept_index: int,
    cue: torch.Tensor,
) -> EpisodicMemoryRecord:
    object_id = f"object-{concept_index}-{index}"
    return EpisodicMemoryRecord(
        memory_id=f"concept-memory-{index}",
        episode_id=f"concept-episode-{index}",
        tick=1,
        cue=cue,
        outcome=Outcome(
            intent_id=f"concept-intent-{index}",
            reward=1.0,
            success=True,
            tick=1,
        ),
        event_ids=(f"concept-event-{index}",),
        assembly_ids=(f"concept-assembly-{index}",),
        object_ids=(object_id,),
        relation_ids=(f"agent:relation-{concept_index}:{object_id}",),
    )


def build_corpus() -> tuple[EpisodicMemoryRecord, ...]:
    patterns = (
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, 1.0]),
        torch.tensor([-1.0, 0.0]),
        torch.tensor([0.0, -1.0]),
    )
    records: list[EpisodicMemoryRecord] = []
    for concept_index, pattern in enumerate(patterns):
        records.extend(
            (
                _record(2 * concept_index, concept_index, pattern),
                _record(
                    2 * concept_index + 1,
                    concept_index,
                    pattern + torch.tensor([0.01, -0.01]),
                ),
            )
        )
    return tuple(records)


def evaluate_capacity(capacities: tuple[int, ...] = (1, 2, 4)) -> list[dict[str, object]]:
    records = build_corpus()
    curve: list[dict[str, object]] = []
    for capacity in capacities:
        if capacity <= 0:
            raise ValueError("concept capacity curve values must be positive")
        organ = ConceptFormationOrgan(capacity=capacity, prune_threshold=0.0)
        concepts = organ.consolidate(records, tick=1)
        curve.append(
            {
                "capacity": capacity,
                "candidate_concepts": 4,
                "retained_concepts": len(concepts),
                "capacity_respected": len(concepts) <= capacity,
            }
        )
    return curve


def evaluate() -> dict[str, object]:
    records = build_corpus()
    organ = ConceptFormationOrgan(capacity=4, prune_threshold=0.0)
    concepts = organ.consolidate(records, tick=1)
    checkpoint = ConceptFormationOrgan.from_checkpoint(organ.checkpoint())
    lesion_target = concepts[0].concept_id
    removed = organ.lesion((lesion_target,))
    capacity_curve = evaluate_capacity()
    capacity_passed = all(
        item["capacity_respected"] and item["retained_concepts"] == item["capacity"]
        for item in capacity_curve
    )
    checkpoint_passed = tuple(item.concept_id for item in checkpoint.concepts) == tuple(
        item.concept_id for item in concepts
    )
    lesion_passed = removed == (lesion_target,) and len(organ.concepts) == len(concepts) - 1
    return {
        "format": REPORT_FORMAT,
        "train_records": len(records),
        "candidate_concepts": len(concepts),
        "capacity_curve": capacity_curve,
        "metrics": {
            "capacity_curve_passed": capacity_passed,
            "checkpoint_continuation": checkpoint_passed,
            "lesion_removed_concept": lesion_passed,
            "post_lesion_concepts": len(organ.concepts),
        },
        "gate": {
            "passed": bool(capacity_passed and checkpoint_passed and lesion_passed),
            "criterion": "concept registry obeys capacity, preserves state through checkpoint, and responds to an explicit lesion",
        },
    }


def build_manifest() -> dict[str, object]:
    return {
        "format": MANIFEST_FORMAT,
        "task": "measure concept registry capacity and causal lesion behavior",
        "capacities": [1, 2, 4],
        "concept_patterns": 4,
        "repeats_per_pattern": 2,
        "controls": ["capacity_curve", "checkpoint_continuation", "concept_lesion"],
        "boundary": "registry capacity and concept-state control only; not open-domain semantic competence",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_concept_capacity_manifest_20260826.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_concept_capacity_20260826.json",
    )
    args = parser.parse_args()
    report = evaluate()
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(build_manifest(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
