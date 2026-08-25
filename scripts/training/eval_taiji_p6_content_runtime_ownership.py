"""Evaluate runtime ownership and checkpointing of learned content selection."""

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
    GenerationController,
    Goal,
    TSKV8Adapter,
)

MANIFEST_FORMAT = "taiji-p6-content-runtime-ownership-manifest-v1"
REPORT_FORMAT = "taiji-p6-content-runtime-ownership-v1"


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


def _selector() -> ContentSelector:
    answer, ask = _candidates()
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
    candidates = _candidates()
    adapter = TSKV8Adapter()
    adapter.attach_content_selector(_selector())
    adapter.attach_generation_controller(GenerationController())
    adapter.set_goals((Goal("stay-informed", "get current information", priority=1.0),))
    adapter.observe(97, learn=False)
    decision = adapter.select_content(candidates, novelty=0.8, resource_budget=0.8)
    expression = adapter.express_selected_content(modality="text", channel="message")
    restored = TSKV8Adapter.from_native_checkpoint(adapter.native_checkpoint())
    restored_expression = restored.express_selected_content(modality="text", channel="message")
    gate_passed = bool(
        decision.selected.candidate_id == "ask"
        and expression.fields["semantic_slots"] == {"mode": "clarify"}
        and restored.last_content_selection == decision
        and restored_expression == expression
    )
    return {
        "format": REPORT_FORMAT,
        "metrics": {
            "selected_candidate": decision.selected.candidate_id,
            "goal_residual": decision.context.goal_residual,
            "world_uncertainty": decision.context.world_uncertainty,
            "expression_modality": expression.modality,
            "expression_channel": expression.channel,
            "native_checkpoint_selection": restored.last_content_selection.selected.candidate_id,
            "native_checkpoint_expression": restored_expression.to_payload(),
        },
        "gate": {
            "passed": gate_passed,
            "criterion": "adapter selects content from current goal/world state, expresses it, and restores selector decision from native checkpoint",
        },
    }


def build_manifest() -> dict[str, object]:
    return {
        "format": MANIFEST_FORMAT,
        "task": "own learned content selection and ContentPlan-to-ExpressionPlan runtime state in the adapter",
        "lesions": ["selector_runtime", "content_expression_bridge", "native_checkpoint_content"],
        "signals": ["goal_residual", "world_uncertainty", "selected_candidate", "expression", "checkpoint"],
        "boundary": "runtime ownership Gate only; no open-ended language generation claim",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p6_content_runtime_ownership_manifest_20260825.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p6_content_runtime_ownership_baseline_20260825.json",
    )
    args = parser.parse_args()
    report = evaluate()
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(build_manifest(), ensure_ascii=False, indent=2), encoding="utf-8")
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

