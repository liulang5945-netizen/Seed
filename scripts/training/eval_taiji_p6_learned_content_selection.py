"""Evaluate learned content selection from goal/world/context signals."""

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

MANIFEST_FORMAT = "taiji-p6-learned-content-selection-manifest-v1"
REPORT_FORMAT = "taiji-p6-learned-content-selection-v1"


def _candidates() -> tuple[ContentCandidate, ContentCandidate]:
    return (
        ContentCandidate(
            candidate_id="answer",
            intent_id="intent:content",
            intent_kind="report_status",
            semantic_slots={"mode": "answer"},
            goal_id="stay-informed",
            goal_alignment=0.9,
            world_relevance=0.3,
            information_gain=0.2,
            confidence=0.9,
            uncertainty=0.1,
            resource_cost=0.1,
        ),
        ContentCandidate(
            candidate_id="ask",
            intent_id="intent:content",
            intent_kind="request_information",
            semantic_slots={"mode": "clarify"},
            goal_id="stay-informed",
            goal_alignment=0.6,
            world_relevance=0.9,
            information_gain=0.9,
            confidence=0.7,
            uncertainty=0.2,
            resource_cost=0.2,
        ),
    )


def evaluate() -> dict[str, object]:
    answer, ask = _candidates()
    certain = ContentSelectionContext(0.9, 0.1, novelty=0.1, resource_budget=0.8)
    uncertain = ContentSelectionContext(0.9, 0.9, novelty=0.8, resource_budget=0.8)
    selector = ContentSelector()
    loss = selector.fit(
        (
            ContentTrainingExample(answer, certain, 1.0),
            ContentTrainingExample(ask, certain, 0.0),
            ContentTrainingExample(answer, uncertain, 0.0),
            ContentTrainingExample(ask, uncertain, 1.0),
        ),
        epochs=400,
    )
    certain_decision = selector.select((answer, ask), certain)
    uncertain_decision = selector.select((answer, ask), uncertain)
    selected_content = uncertain_decision.selected.to_content_plan()
    restored = ContentSelector.from_checkpoint(selector.checkpoint())
    gate_passed = bool(
        certain_decision.selected.candidate_id == "answer"
        and uncertain_decision.selected.candidate_id == "ask"
        and selected_content.semantic_slots == {"mode": "clarify"}
        and selected_content.source_goal_id == "stay-informed"
        and restored.select((answer, ask), uncertain).selected == uncertain_decision.selected
    )
    return {
        "format": REPORT_FORMAT,
        "metrics": {
            "fit_loss": loss,
            "certain_context_selection": certain_decision.selected.candidate_id,
            "uncertain_context_selection": uncertain_decision.selected.candidate_id,
            "selected_content_kind": selected_content.intent_kind,
            "selected_semantic_slots": dict(selected_content.semantic_slots),
            "checkpoint_selection": restored.select((answer, ask), uncertain).selected.candidate_id,
            "training_steps": selector.training_steps,
        },
        "gate": {
            "passed": gate_passed,
            "criterion": "learned utility selects different semantic content under different world uncertainty and restores the same selection from checkpoint",
        },
    }


def build_manifest() -> dict[str, object]:
    return {
        "format": MANIFEST_FORMAT,
        "task": "learn context-conditioned content selection from goal/world/context signals",
        "lesions": [
            "copy_action_intent",
            "fixed_content_table",
            "world_context",
            "content_selector_checkpoint",
        ],
        "signals": [
            "goal_residual",
            "world_uncertainty",
            "novelty",
            "resource_budget",
            "semantic_slots",
        ],
        "boundary": "learned content selection Gate only; no natural-language fluency or open-ended semantic invention claim",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "taiji_p6_learned_content_selection_manifest_20260825.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "taiji_p6_learned_content_selection_baseline_20260825.json",
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
