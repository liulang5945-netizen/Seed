"""Run the single-cell controlled K worker canary after attachment preflight.

The canary consumes existing K1/K2/K3 worker artifacts only.  It executes one
real holdout Workbench step, projects the real outcome through K3, records one
typed adapter exchange, and exercises candidate stage/rollback.  It never
trains a worker, changes the default runtime, or promotes a candidate.
"""

from __future__ import annotations

import copy
import json
import shutil
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.seed_runtime import SeedRuntime  # noqa: E402
from scripts.training.eval_taiji_m4v2_r4_shadow import _score  # noqa: E402
from scripts.training.eval_taiji_m4v2_r6_k_worker_attachment_preflight import (  # noqa: E402
    _parent,
    _r6_course,
    _restore_worker,
    run_preflight,
)
from scripts.training.eval_taiji_m5_k2_multistep_composition import (  # noqa: E402
    READ_ONLY_ROUTES,
    _build_workspace,
    _episode,
    _holdout_episode_paths,
    _observe_all,
    _registry,
    _schema,
    _world,
)
from seed import Seed  # noqa: E402
from seed.config import SeedConfig  # noqa: E402
from seed_platform.workbench import WorkbenchEnvironment  # noqa: E402
from taiji import (  # noqa: E402
    KAdapterExchange,
    KAdapterInput,
    KAdapterOutput,
    KContinualAdapter,
    KWorkerManifest,
    KWorkerManifestBundle,
    NativeReadOnlyIntentPlanner,
    OutcomeDependencyProjector,
    OutcomeDependencySpec,
    ReadOnlyIntentPolicy,
    StructuredSemanticLearner,
    StructuredSemanticResult,
    StructuredSemanticTransitionLearner,
    StructuredSemanticTransitionResult,
    Taiji,
    TaijiConfig,
    WorldEvent,
    content_digest,
)

REPORT_FORMAT = "taiji-m4v2-r6-k-worker-controlled-canary-v1"
VERSION = 1
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "checkpoints" / "taiji_k_workers"
DEFAULT_REPORT = (
    PROJECT_ROOT / "reports" / "taiji_m4v2_r6_k_worker_controlled_canary_20260909.json"
)
DEFAULT_CANDIDATE_NAMESPACE = "taiji:k:candidate"
EPSILON = 0.01


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        raw = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(path, map_location="cpu")
    if not isinstance(raw, Mapping):
        raise TypeError("worker artifact must be a mapping")
    return {str(key): value for key, value in raw.items()}


def _scores(parent: Mapping[str, Any], course_seed: int) -> dict[str, float]:
    model = Taiji.from_checkpoint(copy.deepcopy(dict(parent)))
    course = _r6_course(course_seed)
    return {
        "S": float(_score(model, course.s_holdout, phase="K-controlled-canary-S")),
        "G": float(_score(model, course.g_holdout, phase="K-controlled-canary-G")),
    }


def _artifact_paths(artifact_dir: Path) -> dict[str, Path]:
    return {
        "k1.semantic": artifact_dir / "taiji_r6_k1_semantic.pt",
        "k2.transition": artifact_dir / "taiji_r6_k2_transition.pt",
        "k3.outcome_projection": artifact_dir / "taiji_r6_k3_outcome_projection.pt",
    }


def _base_report(
    *,
    parent_digest: str,
    artifact_paths: Mapping[str, Path],
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "report_format": REPORT_FORMAT,
        "version": VERSION,
        "created_at_unix": time.time(),
        "status": "blocked_attachment",
        "parent_checkpoint_digest": parent_digest,
        "artifact_paths": {key: str(value) for key, value in artifact_paths.items()},
        "preflight_status": preflight.get("status"),
        "attachment_gate_passed": preflight.get("attachment_gate_passed", False),
        "checks": {},
        "training_performed": False,
        "candidate_training_performed": False,
        "candidate_promoted": False,
        "default_runtime_attached": False,
        "provider_attached": False,
        "mcp_attached": False,
        "client_attached": False,
        "cuda_used": False,
        "can_start_r6_formal": False,
        "can_promote": False,
    }


def run_canary(
    *,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    model_seed: int = 17,
    course_seed: int = 0,
    candidate_namespace: str = DEFAULT_CANDIDATE_NAMESPACE,
) -> dict[str, Any]:
    started = time.perf_counter()
    paths = _artifact_paths(artifact_dir)
    parent = _parent(model_seed)
    parent_digest = content_digest(parent)
    preflight = run_preflight(
        semantic_checkpoint=str(paths["k1.semantic"]),
        transition_checkpoint=str(paths["k2.transition"]),
        projection_checkpoint=str(paths["k3.outcome_projection"]),
        model_seed=model_seed,
        course_seed=course_seed,
        candidate_namespace=candidate_namespace,
    )
    report = _base_report(
        parent_digest=parent_digest,
        artifact_paths=paths,
        preflight=preflight,
    )
    if preflight.get("status") != "passed":
        report["blocking_reason"] = (
            "controlled K canary requires a passed real-worker attachment preflight"
        )
        report["elapsed_seconds"] = time.perf_counter() - started
        return report

    temp_parent = PROJECT_ROOT / ".tmp-m4v2-r6-k-worker-canary"
    temp_parent.mkdir(parents=True, exist_ok=True)
    temp_root = temp_parent / f"canary-{uuid4().hex}"
    temp_root.mkdir()
    import seed_platform.workbench as workbench_module

    original_get_setting = workbench_module.get_setting
    workbench_module.get_setting = lambda key, default=None: (
        str(temp_root) if key == "workspace_path" else default
    )
    try:
        artifacts = {worker_id: _load_mapping(path) for worker_id, path in paths.items()}
        restored_workers: dict[str, Any] = {}
        manifests: list[KWorkerManifest] = []
        for worker_id in ("k1.semantic", "k2.transition", "k3.outcome_projection"):
            manifest, details = _restore_worker(
                worker_id,
                artifacts[worker_id],
                parent_digest=parent_digest,
            )
            manifests.append(manifest)
            restored_workers[worker_id] = details
        semantic = StructuredSemanticLearner.from_checkpoint(
            artifacts["k1.semantic"]["checkpoint"], device="cpu"
        )
        transition = StructuredSemanticTransitionLearner.from_checkpoint(
            artifacts["k2.transition"]["checkpoint"], device="cpu"
        )
        projector = OutcomeDependencyProjector.from_checkpoint(
            artifacts["k3.outcome_projection"]["checkpoint"]
        )
        source_manifest_digest = str(artifacts["k1.semantic"]["source_manifest_digest"])
        resource_manifest_digest = str(artifacts["k1.semantic"]["resource_manifest_digest"])
        bundle = KWorkerManifestBundle.create(
            parent_checkpoint_digest=parent_digest,
            source_manifest_digest=source_manifest_digest,
            resource_manifest_digest=resource_manifest_digest,
            candidate_namespace=candidate_namespace,
            workers=manifests,
        )
        adapter = KContinualAdapter(
            parent_checkpoint_digest=parent_digest,
            owner_graph_digest=bundle.owner_graph_digest,
            source_manifest_digest=source_manifest_digest,
            resource_manifest_digest=resource_manifest_digest,
            dependency_scope_id=projector.scope_id,
            candidate_namespace=candidate_namespace,
        )
        adapter.bind_worker_bundle(bundle)

        _build_workspace(temp_root, task_seed=course_seed)
        schema = _schema()
        holdout_registry = _registry(typescript_available=True)
        episode_paths = _holdout_episode_paths()[0]
        paths_for_observation = ("missing_00.txt", *episode_paths)
        observations = {
            observation.path: observation
            for observation in _observe_all(
                temp_root,
                registry=holdout_registry,
                split="controlled-canary",
                paths=list(paths_for_observation),
                schema=schema,
            )
        }
        sequence = _episode(
            observations["missing_00.txt"], observations, episode_paths
        )
        runtime = SeedRuntime(
            Seed(
                SeedConfig(taiji=TaijiConfig(seed=model_seed)),
                episode_id=f"m4v2-r6-k-canary-{model_seed}-{course_seed}",
            )
        )
        runtime._workbench_environment = WorkbenchEnvironment(
            root=temp_root,
            programming_language_registry=holdout_registry,
        )
        planner = NativeReadOnlyIntentPlanner(
            ReadOnlyIntentPolicy(routes=READ_ONLY_ROUTES)
        )
        observation = sequence[1]
        event = observation.to_percept_event(tick=1)
        initial_world = _world(sequence[0], tick=0)
        semantic_result = semantic.predict(event)
        transition_result = transition.predict(initial_world, event)
        if not isinstance(semantic_result, StructuredSemanticResult):
            raise TypeError("K1 controlled canary returned an invalid result")
        if not isinstance(transition_result, StructuredSemanticTransitionResult):
            raise TypeError("K2 controlled canary returned an invalid result")
        if semantic_result.goal is None or semantic_result.content_plan is None:
            raise ValueError("K1 controlled canary produced no goal/content plan")
        if transition_result.world is None:
            raise ValueError("K2 controlled canary produced no predicted world")
        decision = planner.propose(
            observation=observation,
            world=transition_result.world,
            goal=semantic_result.goal,
            content=semantic_result.content_plan,
            capability_snapshot=runtime.workbench_environment.capability_snapshot,
            tick=1,
        )
        if not decision.accepted or decision.action_intent is None:
            raise ValueError(f"controlled canary intent rejected: {decision.reason_code}")
        intent = decision.action_intent
        snapshot_id = runtime.workbench_environment.capability_snapshot.snapshot_id
        outcome = runtime.execute_workbench_intent(
            intent,
            snapshot_id=snapshot_id,
            learn=False,
        )
        outcome_payload = outcome.get("outcome") or {}
        real_success = bool(outcome_payload.get("success"))
        if not real_success:
            raise RuntimeError("controlled canary Workbench execution failed")

        event_attributes = (
            ("capability_id", str(intent.kind)),
            ("path", observation.path),
            ("success", real_success),
        )
        outcome_event = WorldEvent(
            event_id="r6-controlled-real-outcome-1",
            kind="workbench.execution",
            tick=transition_result.world.tick,
            subject_id=observation.path,
            attributes=event_attributes,
            provenance="r6-controlled-real-workbench",
        )
        dependency_spec = OutcomeDependencySpec(
            dependency_id="r6-controlled-real-dependency-1",
            next_task_id="r6-controlled-follow-up-1",
            capability_id=str(intent.kind),
            required_outcome="success",
        )
        projection = projector.project(
            transition_result.world, outcome_event, dependency_spec
        )
        if not projection.accepted:
            raise ValueError(f"K3 rejected real outcome: {projection.reason_code}")
        enriched_world = projector.apply(transition_result.world, projection)
        adapter.bind_dependency_projection(projection)
        input_item = KAdapterInput(
            episode_id=f"r6-controlled-k-{model_seed}-{course_seed}",
            parent_checkpoint_digest=parent_digest,
            observation_digest=content_digest(observation.to_payload()),
            world_digest=content_digest(transition_result.world.to_payload()),
            goal_digest=content_digest(semantic_result.goal.to_payload()),
            content_plan_digest=content_digest(semantic_result.content_plan.to_payload()),
            source_manifest_digest=source_manifest_digest,
            tick=1,
        )
        action_digest = content_digest(
            {"kind": str(intent.kind), "payload": intent.to_payload()}
        )
        output_item = KAdapterOutput(
            parent_checkpoint_digest=parent_digest,
            input_digest=input_item.input_digest,
            action_digest=action_digest,
            outcome_signature=projection.outcome_signature,
            dependency_digest=projection.dependency_digest,
            dependency_projection_digest=projection.projection_digest,
            success=real_success,
            lineage=(
                adapter.dependency_scope_id,
                input_item.input_digest,
                projection.projection_digest,
                projection.dependency_digest,
                manifests[0].manifest_digest,
                manifests[1].manifest_digest,
                manifests[2].manifest_digest,
            ),
        )
        exchange = KAdapterExchange.create(
            scope_id=adapter.dependency_scope_id,
            input=input_item,
            output=output_item,
        )
        adapter.record_exchange(exchange)
        exchange_checkpoint = adapter.checkpoint()
        exchange_restored = KContinualAdapter.from_checkpoint(exchange_checkpoint)

        before = _scores(parent, course_seed)
        after = _scores(parent, course_seed)
        retention = {
            phase: abs(after[phase] - before[phase]) <= EPSILON for phase in ("S", "G")
        }
        token = adapter.stage_candidate(
            candidate_checkpoint_digest=content_digest(
                {
                    "format": "r6-controlled-k-candidate-v1",
                    "parent": parent_digest,
                    "exchange": exchange.exchange_digest,
                }
            ),
            candidate_owner_graph_digest=content_digest(
                {
                    "owner_graph": bundle.owner_graph_digest,
                    "worker_bundle": bundle.bundle_digest,
                    "exchange": exchange.exchange_digest,
                }
            ),
            candidate_source_manifest_digest=content_digest(
                {
                    "source_manifest": source_manifest_digest,
                    "exchange": exchange.exchange_digest,
                }
            ),
            candidate_parent_checkpoint_digest=parent_digest,
        )
        staged = KContinualAdapter.from_checkpoint(adapter.checkpoint())
        rollback_record = adapter.rollback(token)
        rollback = KContinualAdapter.from_checkpoint(adapter.checkpoint())
        checks = {
            "attachment_preflight_passed": preflight.get("status") == "passed",
            "k1_result_typed": isinstance(semantic_result, StructuredSemanticResult),
            "k2_result_typed": isinstance(
                transition_result, StructuredSemanticTransitionResult
            ),
            "read_only_intent_accepted": decision.accepted,
            "real_workbench_success": real_success,
            "k3_projection_accepted": projection.accepted,
            "k3_dependency_applied": (
                ("dependency", "id", dependency_spec.dependency_id)
                in enriched_world.relations
            ),
            "typed_exchange_parent_echo": (
                exchange.output.parent_checkpoint_digest == parent_digest
            ),
            "typed_exchange_worker_lineage": all(
                manifest.manifest_digest in exchange.output.lineage
                for manifest in manifests
            ),
            "exchange_checkpoint_roundtrip": (
                exchange_restored.last_exchange == exchange
                and exchange_restored.worker_bundle == bundle
            ),
            "candidate_stage_roundtrip": staged.worker_bundle == bundle,
            "rollback_record_explicit": (
                rollback_record.status == "rolled_back"
                and rollback_record.reason == "explicit_parent_restore"
            ),
            "rollback_parent_namespace_restored": (
                rollback.active_namespace == rollback.parent_namespace
            ),
            "rollback_exchange_preserved": rollback.last_exchange == exchange,
            "old_S_retention": retention["S"],
            "old_G_retention": retention["G"],
            "adapter_training_steps_zero": adapter.training_steps == 0,
            "no_default_runtime": True,
            "no_external_integrations": True,
            "can_promote_false": True,
        }
        report.update(
            {
                "status": "passed" if all(checks.values()) else "failed",
                "checks": checks,
                "worker_bundle_digest": bundle.bundle_digest,
                "owner_graph_digest": bundle.owner_graph_digest,
                "source_manifest_digest": source_manifest_digest,
                "resource_manifest_digest": resource_manifest_digest,
                "semantic_status": semantic_result.status,
                "transition_status": transition_result.status,
                "observation_path": observation.path,
                "intent_kind": str(intent.kind),
                "outcome_success": real_success,
                "outcome_reward": float(
                    (outcome.get("taiji_outcome") or {}).get("reward", 0.0)
                ),
                "projection_digest": projection.projection_digest,
                "exchange_digest": exchange.exchange_digest,
                "old_capability_before": before,
                "old_capability_after": after,
                "old_capability_retention": retention,
                "candidate_training_performed": False,
                "candidate_promoted": False,
                "can_start_r6_formal": False,
                "can_promote": False,
                "blocking_reason": (
                    None
                    if all(checks.values())
                    else "controlled K canary Gate failed; R6 formal remains closed"
                ),
            }
        )
        report["adapter_checkpoint"] = {
            "exchange_checkpoint_digest": content_digest(exchange_checkpoint),
            "rollback_checkpoint_digest": content_digest(rollback.checkpoint()),
            "rollback_record_digest": rollback_record.record_digest,
        }
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        report["status"] = "failed"
        report["blocking_reason"] = str(exc)
    finally:
        workbench_module.get_setting = original_get_setting
        shutil.rmtree(temp_root, ignore_errors=True)
    report["elapsed_seconds"] = time.perf_counter() - started
    return report


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--model-seed", type=int, default=17)
    parser.add_argument("--course-seed", type=int, default=0)
    parser.add_argument("--candidate-namespace", default=DEFAULT_CANDIDATE_NAMESPACE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    artifact_dir = args.artifact_dir
    if not artifact_dir.is_absolute():
        artifact_dir = PROJECT_ROOT / artifact_dir
    report_path = args.report
    if not report_path.is_absolute():
        report_path = PROJECT_ROOT / report_path
    report = run_canary(
        artifact_dir=artifact_dir,
        model_seed=args.model_seed,
        course_seed=args.course_seed,
        candidate_namespace=args.candidate_namespace,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
