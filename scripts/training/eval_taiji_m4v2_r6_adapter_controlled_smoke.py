"""Run a one-cell typed K adapter controlled smoke.

The smoke proves the native exchange contract and old-capability observation on
one immutable parent.  Its outcome is a fixture projection, not a claim that a
K learner has been trained.  No default runtime, provider, MCP, client, CUDA,
or formal learner update is used.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m4v2_r4_shadow import _score  # noqa: E402
from scripts.training.eval_taiji_m4v2_r6_parent_baseline_preflight import (  # noqa: E402
    _manifests,
    _parent,
    _r6_course,
)
from taiji import (  # noqa: E402
    KAdapterExchange,
    KAdapterInput,
    KAdapterOutput,
    KContinualAdapter,
    OutcomeDependencyProjector,
    OutcomeDependencySpec,
    Taiji,
    WorldEvent,
    WorldState,
    content_digest,
)

REPORT_FORMAT = "taiji-m4v2-r6-adapter-controlled-smoke-v1"
VERSION = 1
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m4v2_r6_adapter_controlled_smoke_20260909.json"
EPSILON = 0.01


def _projection(scope_id: str) -> Any:
    world = WorldState(tick=0, entities=("workbench",), uncertainty=0.0)
    event = WorldEvent(
        event_id="r6-controlled-outcome-0",
        kind="workbench.evidence",
        tick=0,
        subject_id="workspace.read",
        attributes=(
            ("capability_id", "workspace.read"),
            ("success", True),
        ),
        provenance="controlled-smoke-fixture",
    )
    spec = OutcomeDependencySpec(
        dependency_id="r6-controlled-dependency-0",
        next_task_id="r6-controlled-follow-up",
        capability_id="workspace.read",
        required_outcome="success",
    )
    return OutcomeDependencyProjector(scope_id).project(world, event, spec)


def _exchange(
    adapter: KContinualAdapter,
    projection: Any,
    parent_digest: str,
) -> KAdapterExchange:
    input_item = KAdapterInput(
        episode_id="r6-controlled-episode-0",
        parent_checkpoint_digest=parent_digest,
        observation_digest=content_digest(
            {"kind": "workbench.observation", "path": "missing_00.txt"}
        ),
        world_digest=content_digest({"kind": "world.state", "tick": 0}),
        goal_digest=content_digest({"goal_id": "r6-controlled-read"}),
        content_plan_digest=content_digest({"content_id": "r6-controlled-inspect"}),
        source_manifest_digest=adapter.source_manifest_digest,
        tick=1,
    )
    output_item = KAdapterOutput(
        parent_checkpoint_digest=parent_digest,
        input_digest=input_item.input_digest,
        action_digest=content_digest({"kind": "workspace.read", "path": "missing_00.txt"}),
        outcome_signature=projection.outcome_signature,
        dependency_digest=projection.dependency_digest,
        dependency_projection_digest=projection.projection_digest,
        success=True,
        lineage=(
            adapter.dependency_scope_id,
            input_item.input_digest,
            projection.projection_digest,
            projection.dependency_digest,
        ),
    )
    return KAdapterExchange.create(
        scope_id=adapter.dependency_scope_id,
        input=input_item,
        output=output_item,
    )


def _scores(parent: dict[str, Any], course: Any) -> dict[str, float]:
    model = Taiji.from_checkpoint(copy.deepcopy(parent))
    return {
        "S": float(_score(model, course.s_holdout, phase="S-controlled-smoke")),
        "G": float(_score(model, course.g_holdout, phase="G-controlled-smoke")),
    }


def run_smoke(*, model_seed: int = 17, course_seed: int = 0) -> dict[str, Any]:
    parent = _parent(model_seed)
    parent_digest = content_digest(parent)
    manifests = _manifests(
        parent_digest,
        model_seed=model_seed,
        course_seed=course_seed,
    )
    adapter = KContinualAdapter(
        parent_checkpoint_digest=parent_digest,
        owner_graph_digest=manifests["owner_graph_digest"],
        source_manifest_digest=manifests["source_manifest_digest"],
        resource_manifest_digest=manifests["resource_manifest_digest"],
        dependency_scope_id=f"r6-controlled-scope-{model_seed}-{course_seed}",
    )
    projection = _projection(adapter.dependency_scope_id)
    before_scores = _scores(parent, _r6_course(course_seed))
    checks: dict[str, bool] = {
        "parent_checkpoint_matches": adapter.parent_checkpoint_matches(parent),
        "projection_accepted": bool(projection.accepted),
        "projection_lineage_complete": len(projection.lineage) == 4,
    }
    adapter.bind_dependency_projection(projection)
    exchange = _exchange(adapter, projection, parent_digest)
    exchange_digest = adapter.record_exchange(exchange)
    exchange_checkpoint = adapter.checkpoint()
    exchange_restored = KContinualAdapter.from_checkpoint(exchange_checkpoint)
    checks["typed_input_output_roundtrip"] = exchange_restored.last_exchange == exchange
    checks["typed_exchange_digest_echo"] = exchange_digest == exchange.exchange_digest
    checks["typed_exchange_parent_echo"] = exchange.output.parent_checkpoint_digest == parent_digest
    checks["typed_exchange_dependency_echo"] = (
        exchange.output.dependency_digest == projection.dependency_digest
        and exchange.output.dependency_projection_digest == projection.projection_digest
    )
    checks["typed_exchange_scope_echo"] = adapter.dependency_scope_id in exchange.output.lineage

    rollback_token = adapter.stage_candidate(
        candidate_checkpoint_digest=content_digest(
            {"format": "r6-controlled-candidate-v1", "parent": parent_digest}
        ),
        candidate_owner_graph_digest=content_digest(
            {"owner": manifests["owner_graph_digest"], "kind": "controlled"}
        ),
        candidate_source_manifest_digest=content_digest(
            {"source": manifests["source_manifest_digest"], "kind": "controlled"}
        ),
        candidate_parent_checkpoint_digest=parent_digest,
    )
    staged_checkpoint = adapter.checkpoint()
    staged_restored = KContinualAdapter.from_checkpoint(staged_checkpoint)
    checks["candidate_stage_roundtrip"] = staged_restored.checkpoint() == staged_checkpoint
    rollback_record = adapter.rollback(rollback_token)
    rollback_checkpoint = adapter.checkpoint()
    rollback_restored = KContinualAdapter.from_checkpoint(rollback_checkpoint)
    checks["rollback_record_is_explicit"] = (
        rollback_record.status == "rolled_back"
        and rollback_record.reason == "explicit_parent_restore"
    )
    checks["rollback_restores_parent_namespace"] = (
        rollback_restored.active_namespace == rollback_restored.parent_namespace
    )
    checks["rollback_preserves_typed_exchange"] = rollback_restored.last_exchange == exchange
    checks["rollback_checkpoint_roundtrip"] = rollback_restored.checkpoint() == rollback_checkpoint

    after_scores = _scores(parent, _r6_course(course_seed))
    retention_deltas = {
        phase: after_scores[phase] - before_scores[phase] for phase in before_scores
    }
    checks["old_capability_retention_observed"] = all(
        delta >= -EPSILON for delta in retention_deltas.values()
    )
    checks["no_training_performed"] = adapter.training_steps == 0
    checks["default_runtime_detached"] = True

    return {
        "format": REPORT_FORMAT,
        "version": VERSION,
        "status": "passed" if all(checks.values()) else "failed",
        "can_start_r6_formal": False,
        "can_promote": False,
        "matrix": {
            "model_seed": int(model_seed),
            "course_seed": int(course_seed),
            "course_label": _r6_course(course_seed).label,
            "parent_checkpoint_digest": parent_digest,
        },
        "checks": checks,
        "typed_boundary": {
            "input_digest": exchange.input.input_digest,
            "output_digest": exchange.output.output_digest,
            "exchange_digest": exchange.exchange_digest,
            "observation_digest": exchange.input.observation_digest,
            "world_digest": exchange.input.world_digest,
            "goal_digest": exchange.input.goal_digest,
            "content_plan_digest": exchange.input.content_plan_digest,
            "action_digest": exchange.output.action_digest,
            "outcome_signature": exchange.output.outcome_signature,
            "dependency_digest": exchange.output.dependency_digest,
            "dependency_projection_digest": exchange.output.dependency_projection_digest,
            "lineage": list(exchange.output.lineage),
            "fixture_outcome_only": True,
        },
        "old_capability_retention": {
            "before_scores": before_scores,
            "after_scores": after_scores,
            "deltas": retention_deltas,
            "epsilon": EPSILON,
            "measured_phases": ["S", "G"],
        },
        "boundary": {
            "k_learner_training_performed": False,
            "k_learner_owner_attached": False,
            "execution_performed": False,
            "default_runtime_attached": False,
            "candidate_promoted": False,
            "provider_mcp_client_network_used": False,
            "cuda_required": False,
            "baseline_complete": False,
            "reason_baseline_incomplete": (
                "This smoke proves typed transport and retention observation only; "
                "it does not train or attach a K learner, so it cannot close K "
                "capability retention."
            ),
        },
        "next_gate": (
            "Freeze and implement the controlled K learner-owner attachment contract "
            "before any K training; preserve this typed exchange as the only input/"
            "output boundary and keep default runtime detached."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-seed", type=int, default=17)
    parser.add_argument("--course-seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    started = time.perf_counter()
    report = run_smoke(model_seed=args.model_seed, course_seed=args.course_seed)
    report["generated_at_epoch"] = int(time.time())
    report["elapsed_seconds"] = time.perf_counter() - started
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "report": str(args.report),
                "status": report["status"],
                "checks_passed": sum(report["checks"].values()),
                "checks_total": len(report["checks"]),
                "baseline_complete": report["boundary"]["baseline_complete"],
                "k_learner_training_performed": report["boundary"]["k_learner_training_performed"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
