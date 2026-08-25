"""Evaluate online content utility updates from real adapter outcomes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji import ContentCandidate, ContentSelector, TSKV8Adapter  # noqa: E402

MANIFEST_FORMAT = "taiji-p6-online-content-credit-manifest-v1"
REPORT_FORMAT = "taiji-p6-online-content-credit-v1"


def _candidates() -> tuple[ContentCandidate, ContentCandidate]:
    return (
        ContentCandidate(
            candidate_id="answer",
            intent_id="intent:content",
            intent_kind="report_status",
            semantic_slots={"mode": "answer"},
            goal_id="stay-informed",
            goal_alignment=0.8,
            world_relevance=0.2,
            information_gain=0.1,
            confidence=0.8,
            uncertainty=0.1,
            resource_cost=0.1,
        ),
        ContentCandidate(
            candidate_id="ask",
            intent_id="intent:content",
            intent_kind="request_information",
            semantic_slots={"mode": "clarify"},
            goal_id="stay-informed",
            goal_alignment=0.5,
            world_relevance=0.8,
            information_gain=0.9,
            confidence=0.7,
            uncertainty=0.2,
            resource_cost=0.2,
        ),
    )


def evaluate() -> dict[str, object]:
    candidates = _candidates()
    selector = ContentSelector()
    adapter = TSKV8Adapter()
    adapter.attach_content_selector(selector)
    adapter.observe(97, learn=False)

    first = adapter.select_content(candidates, novelty=0.8, resource_budget=0.8)
    adapter.act((10,), sample=False)
    adapter.settle_action(-1.0, learn=False, success=False)
    first_error = adapter.last_content_prediction_error
    after_failure = selector.scores(candidates, first.context)
    adapter.observe(98, learn=False)

    second = adapter.select_content(candidates, novelty=0.8, resource_budget=0.8)
    adapter.act((10,), sample=False)
    adapter.settle_action(1.0, learn=False, success=True)
    second_error = adapter.last_content_prediction_error
    after_success = selector.scores(candidates, second.context)
    restored = TSKV8Adapter.from_native_checkpoint(adapter.native_checkpoint())
    gate_passed = bool(
        first.selected.candidate_id == "answer"
        and second.selected.candidate_id == "ask"
        and first_error is not None
        and first_error > 0.0
        and second_error is not None
        and second_error < 0.0
        and after_failure[0] < after_failure[1]
        and after_success[1] > after_success[0]
        and restored.last_content_selection == second
        and restored._content_feedback_applied
    )
    return {
        "format": REPORT_FORMAT,
        "metrics": {
            "first_content": first.selected.candidate_id,
            "first_outcome_reward": -1.0,
            "first_prediction_error": first_error,
            "second_content": second.selected.candidate_id,
            "second_outcome_reward": 1.0,
            "second_prediction_error": second_error,
            "scores_after_failure": after_failure,
            "scores_after_success": after_success,
            "training_steps": selector.training_steps,
            "checkpoint_feedback_applied": restored._content_feedback_applied,
        },
        "gate": {
            "passed": gate_passed,
            "criterion": "real adapter outcomes demote failed content, promote successful content, and restore one-time credit state from checkpoint",
        },
    }


def build_manifest() -> dict[str, object]:
    return {
        "format": MANIFEST_FORMAT,
        "task": "update selected semantic content utility from successive failed and successful adapter outcomes",
        "lesions": ["no_online_credit", "duplicate_feedback", "content_feedback_checkpoint"],
        "signals": ["reward", "prediction_error", "candidate_migration", "training_steps", "checkpoint"],
        "boundary": "online content credit Gate only; no open-ended semantic learning claim",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p6_online_content_credit_manifest_20260825.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p6_online_content_credit_baseline_20260825.json",
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
