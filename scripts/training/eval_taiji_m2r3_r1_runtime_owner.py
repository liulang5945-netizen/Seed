"""Evaluate the optional structured semantic owner in the Taiji runtime."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m2r3_r0_structured_semantics import build_corpus  # noqa: E402
from taiji import (  # noqa: E402
    StructuredSemanticLearner,
    TSKV8Adapter,
    content_digest,
)

REPORT_FORMAT = "taiji-m2r3-r1-runtime-semantic-owner-v1"
REPORT_VERSION = 1


def evaluate() -> dict[str, Any]:
    corpus = build_corpus()
    learner = StructuredSemanticLearner(corpus)
    training_losses = learner.fit(corpus.train, epochs=160, learning_rate=2.0)
    adapter = TSKV8Adapter()
    before_state_digest = content_digest(adapter.cognitive_snapshot().to_payload())
    adapter.attach_structured_semantic_learner(learner)
    result = adapter.infer_structured_semantics(corpus.test[0].percept)
    after_state_digest = content_digest(adapter.cognitive_snapshot().to_payload())
    checkpoint = adapter.native_checkpoint()
    restored = TSKV8Adapter.from_native_checkpoint(checkpoint)
    restored_result = restored.last_structured_semantic_result
    plain_checkpoint = TSKV8Adapter().native_checkpoint()
    adapter.attach_structured_semantic_learner(None)
    detached_checkpoint = adapter.native_checkpoint()
    checks = {
        "explicit_attach_required": "structured_semantic" not in plain_checkpoint["components"],
        "inference_is_state_read_only": before_state_digest == after_state_digest,
        "result_is_structured": (
            result.status == "resolved" and result.goal is not None and result.content_plan is not None
        ),
        "native_checkpoint_contains_owner": "structured_semantic" in checkpoint["components"],
        "checkpoint_round_trip": (
            restored.structured_semantic_learner is not None
            and restored_result is not None
            and content_digest(restored_result.to_payload()) == content_digest(result.to_payload())
        ),
        "restored_inference_matches": (
            restored_result is not None
            and content_digest(
                restored.infer_structured_semantics(corpus.test[0].percept).to_payload()
            )
            == content_digest(result.to_payload())
        ),
        "detach_clears_snapshot": (
            adapter.structured_semantic_learner is None
            and adapter.last_structured_semantic_result is None
            and "structured_semantic" not in detached_checkpoint["components"]
        ),
        "no_provider_or_execution": (
            result.content_plan is not None
            and adapter.last_task_interpretation is None
            and adapter.cognitive_snapshot().action_intent is None
        ),
    }
    passed = all(bool(value) for value in checks.values())
    return {
        "format": REPORT_FORMAT,
        "version": REPORT_VERSION,
        "status": "passed" if passed else "failed",
        "can_promote": False,
        "source_corpus_digest": corpus.source_digest,
        "preflight": {
            "checkpoint_digest": content_digest(checkpoint),
            "component_keys": sorted(checkpoint["components"]),
            "learner_parameter_count": learner.parameter_count,
            "training_losses": training_losses,
        },
        "result": {
            "status": result.status,
            "goal_id": None if result.goal is None else result.goal.goal_id,
            "content_id": None
            if result.content_plan is None
            else result.content_plan.content_id,
            "digest": content_digest(result.to_payload()),
        },
        "gate": {"passed": passed, "checks": checks},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_m2r3_r1_runtime_owner_20260907.json",
    )
    args = parser.parse_args()
    report = evaluate()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
