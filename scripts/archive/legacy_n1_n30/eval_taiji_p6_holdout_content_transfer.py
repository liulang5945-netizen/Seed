"""Evaluate holdout content transfer across unseen intent and slot structures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji import (  # noqa: E402
    ContentCandidate,
    ContentSelectionContext,
    ContentSelector,
    ContentTrainingExample,
)

MANIFEST_FORMAT = "taiji-p6-holdout-content-transfer-manifest-v1"
REPORT_FORMAT = "taiji-p6-holdout-content-transfer-v1"


def _train_selector() -> ContentSelector:
    answer = ContentCandidate(
        candidate_id="train-answer",
        intent_id="train:intent",
        intent_kind="report_status",
        semantic_slots={"mode": "answer"},
        goal_alignment=0.9,
        world_relevance=0.3,
        information_gain=0.2,
        confidence=0.9,
        uncertainty=0.1,
        resource_cost=0.1,
    )
    ask = ContentCandidate(
        candidate_id="train-ask",
        intent_id="train:intent",
        intent_kind="request_information",
        semantic_slots={"mode": "clarify"},
        goal_alignment=0.6,
        world_relevance=0.9,
        information_gain=0.9,
        confidence=0.7,
        uncertainty=0.2,
        resource_cost=0.2,
    )
    certain = ContentSelectionContext(0.9, 0.1, novelty=0.1, resource_budget=0.8)
    uncertain = ContentSelectionContext(0.9, 0.9, novelty=0.8, resource_budget=0.8)
    selector = ContentSelector()
    selector.fit(
        (
            ContentTrainingExample(answer, certain, 1.0),
            ContentTrainingExample(ask, certain, 0.0),
            ContentTrainingExample(answer, uncertain, 0.0),
            ContentTrainingExample(ask, uncertain, 1.0),
        ),
        epochs=400,
    )
    return selector


def evaluate() -> dict[str, object]:
    holdout = ContentCandidate(
        candidate_id="holdout-forecast-42",
        intent_id="holdout:intent:forecast",
        intent_kind="forecast_digest",
        semantic_slots={"format": "digest", "regions": ["east", "south"]},
        goal_id="stay-informed",
        goal_alignment=0.55,
        world_relevance=0.88,
        information_gain=0.92,
        confidence=0.68,
        uncertainty=0.2,
        resource_cost=0.2,
    )
    answer = ContentCandidate(
        candidate_id="holdout-answer-7",
        intent_id="holdout:intent:forecast",
        intent_kind="static_summary",
        semantic_slots={"format": "summary"},
        goal_id="stay-informed",
        goal_alignment=0.9,
        world_relevance=0.3,
        information_gain=0.2,
        confidence=0.9,
        uncertainty=0.1,
        resource_cost=0.1,
    )
    context = ContentSelectionContext(0.9, 0.9, novelty=0.8, resource_budget=0.8)
    selector = _train_selector()
    decision = selector.select((holdout, answer), context)
    restored = ContentSelector.from_checkpoint(selector.checkpoint())
    restored_decision = restored.select((holdout, answer), context)
    gate_passed = bool(
        decision.selected.candidate_id == "holdout-forecast-42"
        and decision.selected.intent_kind == "forecast_digest"
        and decision.selected.semantic_slots == holdout.semantic_slots
        and restored_decision.selected == decision.selected
    )
    return {
        "format": REPORT_FORMAT,
        "metrics": {
            "training_intent_kinds": ["report_status", "request_information"],
            "holdout_intent_kind": holdout.intent_kind,
            "holdout_candidate_id": holdout.candidate_id,
            "selected_candidate": decision.selected.candidate_id,
            "selected_slots": dict(decision.selected.semantic_slots),
            "checkpoint_selection": restored_decision.selected.candidate_id,
        },
        "gate": {
            "passed": gate_passed,
            "criterion": "a new intent kind, candidate ID, and nested slot structure use learned context utility without an answer table",
        },
    }


def build_manifest() -> dict[str, object]:
    return {
        "format": MANIFEST_FORMAT,
        "task": "select a holdout content candidate with unseen intent kind, ID, and semantic slot structure",
        "lesions": [
            "candidate_id_lookup",
            "intent_kind_table",
            "slot_shape_table",
            "checkpoint_transfer",
        ],
        "signals": [
            "goal_residual",
            "world_uncertainty",
            "novelty",
            "candidate_features",
            "holdout_selection",
        ],
        "boundary": "holdout content utility transfer Gate only; no open-ended semantic invention claim",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "taiji_p6_holdout_content_transfer_manifest_20260825.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "taiji_p6_holdout_content_transfer_baseline_20260825.json",
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
