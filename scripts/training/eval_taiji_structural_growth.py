"""Evaluate budgeted, validated and rollbackable Taiji structural growth."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_concept_branch import build_branch_records  # noqa: E402
from scripts.training.eval_taiji_concept_branch_attribution import _runtime_config  # noqa: E402
from scripts.training.eval_taiji_concept_online_birth import _novel_transitions  # noqa: E402
from taiji import TSKV8Adapter  # noqa: E402

MANIFEST_FORMAT = "taiji-structural-growth-manifest-v1"
REPORT_FORMAT = "taiji-structural-growth-v1"


def _runtime(*, budget: int) -> TSKV8Adapter:
    runtime = TSKV8Adapter(
        replace(_runtime_config(), development_structural_budget=budget),
        episode_id=f"structural-growth-{budget}",
    )
    records = build_branch_records(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    good_records = tuple(
        record for record in records if record.episode_id.startswith("branch-episode-0-")
    )
    runtime.concept_formation.consolidate(good_records, tick=3)
    runtime._cognitive_state = replace(
        runtime._cognitive_state,
        concepts=runtime.concept_formation.concepts,
    )
    return runtime


def evaluate() -> dict[str, object]:
    transitions = _novel_transitions()
    runtime = _runtime(budget=1)
    concept = runtime.concept_formation.concepts[0]
    before_trace_count = len(concept.sequence_traces)
    trace_id = runtime.grow_online_concept_branch(
        concept.concept_id,
        tuple(
            (transition, error) for transition, error in zip(transitions, (0.05, 0.10), strict=True)
        ),
    )
    request = runtime.growth_requests[-1]
    accepted_checkpoint = runtime.native_checkpoint()
    recovered = TSKV8Adapter.from_native_checkpoint(accepted_checkpoint)
    recovered_request = next(
        item for item in recovered.growth_requests if item.request_id == request.request_id
    )
    rollback = recovered.rollback_growth_request(request.request_id)
    rolled_back_snapshot = recovered.cognitive_snapshot()
    rolled_back_trace_count = len(recovered.concept_formation.concepts[0].sequence_traces)
    exhausted = _runtime(budget=0)
    exhausted_concept = exhausted.concept_formation.concepts[0]
    rejected_trace = exhausted.grow_online_concept_branch(
        exhausted_concept.concept_id,
        tuple(
            (transition, error) for transition, error in zip(transitions, (0.05, 0.10), strict=True)
        ),
    )
    rejected_request = exhausted.growth_requests[-1]
    gate_passed = bool(
        trace_id is not None
        and request.status == "accepted"
        and request.validation_score == 1.0
        and request.parent_checkpoint_id is not None
        and runtime.cognitive_snapshot().development.structural_budget == 0
        and recovered_request.status == "accepted"
        and rollback
        and rolled_back_snapshot.development.structural_budget == 1
        and rolled_back_snapshot.development.growth_count == 0
        and rolled_back_trace_count == before_trace_count
        and rejected_trace is None
        and rejected_request.status == "rejected"
        and exhausted.cognitive_snapshot().development.structural_budget == 0
        and len(exhausted.concept_formation.concepts[0].sequence_traces) == before_trace_count
    )
    return {
        "format": REPORT_FORMAT,
        "metrics": {
            "before_trace_count": before_trace_count,
            "accepted_trace_id": trace_id,
            "accepted_request_id": request.request_id,
            "accepted_validation_score": request.validation_score,
            "budget_after_accept": runtime.cognitive_snapshot().development.structural_budget,
            "checkpoint_request_status": recovered_request.status,
            "rollback": rollback,
            "budget_after_rollback": rolled_back_snapshot.development.structural_budget,
            "growth_count_after_rollback": rolled_back_snapshot.development.growth_count,
            "trace_count_after_rollback": rolled_back_trace_count,
            "rejected_request_status": rejected_request.status,
            "budget_after_rejection": exhausted.cognitive_snapshot().development.structural_budget,
        },
        "gate": {
            "passed": gate_passed,
            "criterion": "online branch birth must consume a versioned structural budget only after checkpoint and lesion validation, preserve a parent snapshot for native-checkpoint rollback, and reject growth without budget",
        },
        "boundary": "This is a closed-world structural-growth governance gate; it does not claim unrestricted self-evolution or general intelligence.",
    }


def build_manifest() -> dict[str, object]:
    return {
        "format": MANIFEST_FORMAT,
        "task": "budgeted and rollbackable online Taiji structural growth",
        "approval": [
            "resource budget",
            "checkpoint roundtrip",
            "selective lesion",
            "trace replayability",
        ],
        "rollback": "restore the parent concept checkpoint, return the budget, and mark the request rolled_back",
        "failure": "zero structural budget rejects growth without mutating the concept",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report", type=Path, default=Path("reports/taiji_structural_growth_20260826.json")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("plans/manifests/taiji_structural_growth_v1.json")
    )
    args = parser.parse_args()
    report = evaluate()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(build_manifest(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
