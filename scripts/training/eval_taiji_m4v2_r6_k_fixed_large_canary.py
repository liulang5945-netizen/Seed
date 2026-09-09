"""Run the R6 native fixed-large K control on one real holdout read."""

from __future__ import annotations

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
from scripts.training.eval_taiji_m4v2_r6_k_worker_attachment_preflight import (  # noqa: E402
    _parent,
)
from scripts.training.eval_taiji_m4v2_r6_k_worker_controlled_canary import (  # noqa: E402
    _scores,
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
    K_WORKER_INPUT_CONTRACT_DIGESTS,
    K_WORKER_OUTPUT_CONTRACT_DIGESTS,
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
    StructuredSemanticTransitionResult,
    TaijiConfig,
    WorldEvent,
    content_digest,
)
from taiji.k_fixed_large import (  # noqa: E402
    FIXED_LARGE_CHECKPOINT_FORMAT,
    FIXED_LARGE_CHECKPOINT_VERSION,
    NativeKFixedLargeEnsemble,
)

REPORT_FORMAT = "taiji-m4v2-r6-k-fixed-large-controlled-canary-v1"
VERSION = 1
EPSILON = 0.01
DEFAULT_ARTIFACT = (
    PROJECT_ROOT
    / "checkpoints"
    / "taiji_k_fixed_large"
    / "model_17"
    / "taiji_r6_k_fixed_large_ensemble.pt"
)
DEFAULT_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "taiji_m4v2_r6_k_fixed_large_controlled_canary_model_17_20260909.json"
)


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        raw = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(path, map_location="cpu")
    if not isinstance(raw, Mapping):
        raise TypeError("fixed-large artifact must be a mapping")
    return {str(key): value for key, value in raw.items()}


def _manifest(
    *,
    worker_id: str,
    checkpoint_format: str,
    checkpoint_version: int,
    checkpoint_digest: str,
    owner_digests: Mapping[str, str],
    parent_digest: str,
    namespace: str,
    source_digest: str,
    source_manifest_digest: str,
    training_steps: int,
) -> KWorkerManifest:
    return KWorkerManifest.create(
        worker_id=worker_id,
        checkpoint_format=checkpoint_format,
        checkpoint_version=checkpoint_version,
        worker_checkpoint_digest=checkpoint_digest,
        owner_digests=tuple(sorted(owner_digests.items())),
        source_digest=source_digest,
        input_contract_digest=K_WORKER_INPUT_CONTRACT_DIGESTS[worker_id],
        output_contract_digest=K_WORKER_OUTPUT_CONTRACT_DIGESTS[worker_id],
        parent_checkpoint_digest=parent_digest,
        candidate_namespace=namespace,
        training_steps=training_steps,
        optimizer_state_present=False,
    )


def _fixed_large_preflight(
    payload: Mapping[str, Any], *, parent_digest: str
) -> tuple[NativeKFixedLargeEnsemble, OutcomeDependencyProjector, KWorkerManifestBundle, dict[str, Any]]:
    if payload.get("format") != FIXED_LARGE_CHECKPOINT_FORMAT:
        raise ValueError("unsupported fixed-large ensemble artifact format")
    if int(payload.get("version", -1)) != FIXED_LARGE_CHECKPOINT_VERSION:
        raise ValueError("unsupported fixed-large ensemble artifact version")
    unsigned = {key: value for key, value in payload.items() if key != "ensemble_digest"}
    if str(payload.get("ensemble_digest", "")) != content_digest(unsigned):
        raise ValueError("fixed-large ensemble artifact digest mismatch")
    if str(payload.get("parent_checkpoint_digest", "")) != parent_digest:
        raise ValueError("fixed-large ensemble crosses the selected parent")
    if int(payload.get("ensemble_width", -1)) != 2:
        raise ValueError("fixed-large ensemble width is not the preregistered width")
    ensemble_payload = payload.get("ensemble_checkpoint")
    k3_payload = payload.get("k3_projection", {})
    if not isinstance(ensemble_payload, Mapping) or not isinstance(k3_payload, Mapping):
        raise ValueError("fixed-large ensemble or K3 checkpoint is missing")
    ensemble = NativeKFixedLargeEnsemble.from_checkpoint(ensemble_payload)
    projector_checkpoint = k3_payload.get("checkpoint")
    if not isinstance(projector_checkpoint, Mapping):
        raise ValueError("fixed-large K3 checkpoint is missing")
    projector = OutcomeDependencyProjector.from_checkpoint(projector_checkpoint)
    source_manifest_digest = str(payload["source_manifest_digest"])
    resource_manifest_digest = str(payload["resource_manifest_digest"])
    namespace = str(payload["candidate_namespace"])
    ensemble_digest = str(payload["ensemble_checkpoint_digest"])
    k1_checkpoint_digest = content_digest(
        {
            "format": FIXED_LARGE_CHECKPOINT_FORMAT,
            "worker_id": "k1.semantic",
            "ensemble_checkpoint": ensemble_payload,
        }
    )
    k2_checkpoint_digest = content_digest(
        {
            "format": FIXED_LARGE_CHECKPOINT_FORMAT,
            "worker_id": "k2.transition",
            "ensemble_checkpoint": ensemble_payload,
        }
    )
    k1_owners = {
        key: value
        for key, value in ensemble.owner_digests.items()
        if key.startswith("k1.replica.")
    }
    k2_owners = {
        key: value
        for key, value in ensemble.owner_digests.items()
        if key.startswith("k2.replica.")
    }
    manifests = (
        _manifest(
            worker_id="k1.semantic",
            checkpoint_format=FIXED_LARGE_CHECKPOINT_FORMAT,
            checkpoint_version=FIXED_LARGE_CHECKPOINT_VERSION,
            checkpoint_digest=k1_checkpoint_digest,
            owner_digests=k1_owners,
            parent_digest=parent_digest,
            namespace=namespace,
            source_digest=content_digest(
                {"source_manifest": source_manifest_digest, "worker": "k1.semantic"}
            ),
            source_manifest_digest=source_manifest_digest,
            training_steps=sum(int(value) for value in payload["training_steps"]["k1"]),
        ),
        _manifest(
            worker_id="k2.transition",
            checkpoint_format=FIXED_LARGE_CHECKPOINT_FORMAT,
            checkpoint_version=FIXED_LARGE_CHECKPOINT_VERSION,
            checkpoint_digest=k2_checkpoint_digest,
            owner_digests=k2_owners,
            parent_digest=parent_digest,
            namespace=namespace,
            source_digest=content_digest(
                {"source_manifest": source_manifest_digest, "worker": "k2.transition"}
            ),
            source_manifest_digest=source_manifest_digest,
            training_steps=sum(int(value) for value in payload["training_steps"]["k2"]),
        ),
        _manifest(
            worker_id="k3.outcome_projection",
            checkpoint_format=str(projector_checkpoint["format"]),
            checkpoint_version=int(projector_checkpoint["version"]),
            checkpoint_digest=str(k3_payload["worker_checkpoint_digest"]),
            owner_digests={"outcome_dependency_projector": str(k3_payload["worker_checkpoint_digest"])},
            parent_digest=parent_digest,
            namespace=namespace,
            source_digest=str(k3_payload["source_digest"]),
            source_manifest_digest=source_manifest_digest,
            training_steps=0,
        ),
    )
    bundle = KWorkerManifestBundle.create(
        parent_checkpoint_digest=parent_digest,
        source_manifest_digest=source_manifest_digest,
        resource_manifest_digest=resource_manifest_digest,
        candidate_namespace=namespace,
        workers=manifests,
    )
    checks = {
        "artifact_digest": True,
        "parent_digest": True,
        "ensemble_width_two": ensemble.ensemble_width == 2,
        "ensemble_checkpoint_digest": content_digest(ensemble.checkpoint()) == ensemble_digest,
        "k3_checkpoint_digest": content_digest(projector_checkpoint)
        == str(k3_payload["worker_checkpoint_digest"]),
        "bundle_digest": bool(bundle.bundle_digest),
        "optimizer_state_absent": not bool(payload.get("optimizer_state_present", True)),
        "cpu_only": str(payload["resource_manifest"]["device"]) == "cpu",
    }
    if not all(checks.values()):
        raise ValueError(f"fixed-large preflight failed: {checks}")
    return ensemble, projector, bundle, {
        "status": "passed",
        "checks": checks,
        "bundle_digest": bundle.bundle_digest,
        "owner_graph_digest": bundle.owner_graph_digest,
        "source_manifest_digest": source_manifest_digest,
        "resource_manifest_digest": resource_manifest_digest,
        "candidate_namespace": namespace,
        "ensemble_parameter_count": ensemble.parameter_count,
    }


def run_canary(
    *,
    artifact_path: Path = DEFAULT_ARTIFACT,
    model_seed: int = 17,
    course_seed: int = 0,
) -> dict[str, Any]:
    started = time.perf_counter()
    parent = _parent(model_seed)
    parent_digest = content_digest(parent)
    report: dict[str, Any] = {
        "report_format": REPORT_FORMAT,
        "version": VERSION,
        "created_at_unix": time.time(),
        "status": "blocked_input",
        "artifact_path": str(artifact_path),
        "parent_checkpoint_digest": parent_digest,
        "training_performed": False,
        "default_runtime_attached": False,
        "provider_attached": False,
        "mcp_attached": False,
        "client_attached": False,
        "cuda_used": False,
        "can_start_r6_formal": False,
        "can_promote": False,
    }
    temp_parent = PROJECT_ROOT / ".tmp-m4v2-r6-k-fixed-large-canary"
    temp_parent.mkdir(parents=True, exist_ok=True)
    temp_root = temp_parent / f"canary-{uuid4().hex}"
    temp_root.mkdir()
    import seed_platform.workbench as workbench_module

    original_get_setting = workbench_module.get_setting
    workbench_module.get_setting = lambda key, default=None: (
        str(temp_root) if key == "workspace_path" else default
    )
    try:
        payload = _load_mapping(artifact_path)
        ensemble, projector, bundle, preflight = _fixed_large_preflight(
            payload, parent_digest=parent_digest
        )
        namespace = str(payload["candidate_namespace"])
        _build_workspace(temp_root, task_seed=course_seed)
        schema = _schema()
        registry = _registry(typescript_available=True)
        episode_paths = _holdout_episode_paths()[0]
        observations = {
            observation.path: observation
            for observation in _observe_all(
                temp_root,
                registry=registry,
                split="fixed-large-controlled-canary",
                paths=["missing_00.txt", *episode_paths],
                schema=schema,
            )
        }
        sequence = _episode(observations["missing_00.txt"], observations, episode_paths)
        runtime = SeedRuntime(
            Seed(
                SeedConfig(taiji=TaijiConfig(seed=model_seed)),
                episode_id=f"m4v2-r6-k-fixed-large-{model_seed}-{course_seed}",
            )
        )
        runtime._workbench_environment = WorkbenchEnvironment(
            root=temp_root,
            programming_language_registry=registry,
        )
        planner = NativeReadOnlyIntentPlanner(ReadOnlyIntentPolicy(routes=READ_ONLY_ROUTES))
        observation = sequence[1]
        event = observation.to_percept_event(tick=1)
        initial_world = _world(sequence[0], tick=0)
        semantic_result = ensemble.predict_semantic(event)
        transition_result = ensemble.predict_transition(initial_world, event)
        if not isinstance(semantic_result, StructuredSemanticResult):
            raise TypeError("fixed-large K1 returned an invalid result")
        if not isinstance(transition_result, StructuredSemanticTransitionResult):
            raise TypeError("fixed-large K2 returned an invalid result")
        if semantic_result.goal is None or semantic_result.content_plan is None:
            raise ValueError("fixed-large K1 produced no goal/content plan")
        if transition_result.world is None:
            raise ValueError("fixed-large K2 produced no predicted world")
        lesioned_semantic = StructuredSemanticLearner.from_checkpoint(
            ensemble.semantic_replicas[0].checkpoint()
        )
        lesioned_semantic.zero_fact_head()
        lesioned_result = lesioned_semantic.predict(event)
        branch_lesion_observable = (
            lesioned_result.fact_scores != semantic_result.fact_scores
        )
        decision = planner.propose(
            observation=observation,
            world=transition_result.world,
            goal=semantic_result.goal,
            content=semantic_result.content_plan,
            capability_snapshot=runtime.workbench_environment.capability_snapshot,
            tick=1,
        )
        if not decision.accepted or decision.action_intent is None:
            raise ValueError(f"fixed-large intent rejected: {decision.reason_code}")
        intent = decision.action_intent
        snapshot_id = runtime.workbench_environment.capability_snapshot.snapshot_id
        outcome = runtime.execute_workbench_intent(intent, snapshot_id=snapshot_id, learn=False)
        outcome_payload = outcome.get("outcome") or {}
        real_success = bool(outcome_payload.get("success"))
        if not real_success:
            raise RuntimeError("fixed-large Workbench execution failed")
        outcome_event = WorldEvent(
            event_id="r6-fixed-large-real-outcome-1",
            kind="workbench.execution",
            tick=transition_result.world.tick,
            subject_id=observation.path,
            attributes=(
                ("capability_id", str(intent.kind)),
                ("path", observation.path),
                ("success", real_success),
            ),
            provenance="r6-fixed-large-real-workbench",
        )
        dependency_spec = OutcomeDependencySpec(
            dependency_id="r6-fixed-large-dependency-1",
            next_task_id="r6-fixed-large-follow-up-1",
            capability_id=str(intent.kind),
            required_outcome="success",
        )
        projection = projector.project(transition_result.world, outcome_event, dependency_spec)
        if not projection.accepted:
            raise ValueError(f"fixed-large K3 rejected real outcome: {projection.reason_code}")
        enriched_world = projector.apply(transition_result.world, projection)
        adapter = KContinualAdapter(
            parent_checkpoint_digest=parent_digest,
            owner_graph_digest=bundle.owner_graph_digest,
            source_manifest_digest=bundle.source_manifest_digest,
            resource_manifest_digest=bundle.resource_manifest_digest,
            dependency_scope_id=projector.scope_id,
            candidate_namespace=namespace,
        )
        adapter.bind_worker_bundle(bundle)
        adapter.bind_dependency_projection(projection)
        input_item = KAdapterInput(
            episode_id=f"r6-fixed-large-k-{model_seed}-{course_seed}",
            parent_checkpoint_digest=parent_digest,
            observation_digest=content_digest(observation.to_payload()),
            world_digest=content_digest(transition_result.world.to_payload()),
            goal_digest=content_digest(semantic_result.goal.to_payload()),
            content_plan_digest=content_digest(semantic_result.content_plan.to_payload()),
            source_manifest_digest=bundle.source_manifest_digest,
            tick=1,
        )
        exchange = KAdapterExchange.create(
            scope_id=adapter.dependency_scope_id,
            input=input_item,
            output=KAdapterOutput(
                parent_checkpoint_digest=parent_digest,
                input_digest=input_item.input_digest,
                action_digest=content_digest({"kind": str(intent.kind), "payload": intent.to_payload()}),
                outcome_signature=projection.outcome_signature,
                dependency_digest=projection.dependency_digest,
                dependency_projection_digest=projection.projection_digest,
                success=real_success,
                lineage=(
                    adapter.dependency_scope_id,
                    input_item.input_digest,
                    projection.projection_digest,
                    projection.dependency_digest,
                    *(manifest.manifest_digest for manifest in bundle.workers),
                ),
            ),
        )
        adapter.record_exchange(exchange)
        exchange_restored = KContinualAdapter.from_checkpoint(adapter.checkpoint())
        token = adapter.stage_candidate(
            candidate_checkpoint_digest=content_digest(
                {"format": REPORT_FORMAT, "parent": parent_digest, "exchange": exchange.exchange_digest}
            ),
            candidate_owner_graph_digest=content_digest(
                {"owner_graph": bundle.owner_graph_digest, "exchange": exchange.exchange_digest}
            ),
            candidate_source_manifest_digest=content_digest(
                {"source_manifest": bundle.source_manifest_digest, "exchange": exchange.exchange_digest}
            ),
            candidate_parent_checkpoint_digest=parent_digest,
        )
        staged = KContinualAdapter.from_checkpoint(adapter.checkpoint())
        rollback_record = adapter.rollback(token)
        rollback = KContinualAdapter.from_checkpoint(adapter.checkpoint())
        before = _scores(parent, course_seed)
        after = _scores(parent, course_seed)
        retention = {
            phase: abs(after[phase] - before[phase]) <= EPSILON for phase in ("S", "G")
        }
        checks = {
            "fixed_large_preflight_passed": preflight["status"] == "passed",
            "k1_result_typed": True,
            "k2_result_typed": True,
            "read_only_intent_accepted": decision.accepted,
            "real_workbench_success": real_success,
            "k3_projection_accepted": projection.accepted,
            "k3_dependency_applied": (
                ("dependency", "id", dependency_spec.dependency_id) in enriched_world.relations
            ),
            "fixed_large_branch_lesion_observable": branch_lesion_observable,
            "typed_exchange_worker_lineage": all(
                manifest.manifest_digest in exchange.output.lineage for manifest in bundle.workers
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
            "rollback_parent_namespace_restored": rollback.active_namespace == rollback.parent_namespace,
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
                "preflight": preflight,
                "checks": checks,
                "worker_bundle_digest": bundle.bundle_digest,
                "owner_graph_digest": bundle.owner_graph_digest,
                "source_manifest_digest": bundle.source_manifest_digest,
                "resource_manifest_digest": bundle.resource_manifest_digest,
                "candidate_namespace": namespace,
                "semantic_status": semantic_result.status,
                "transition_status": transition_result.status,
                "observation_path": observation.path,
                "intent_kind": str(intent.kind),
                "outcome_success": real_success,
                "outcome_reward": float((outcome.get("taiji_outcome") or {}).get("reward", 0.0)),
                "projection_accepted": projection.accepted,
                "projection_reason": projection.reason_code,
                "projection_digest": projection.projection_digest,
                "exchange_digest": exchange.exchange_digest,
                "old_capability_before": before,
                "old_capability_after": after,
                "old_capability_retention": retention,
                "resource": {
                    "parameter_count": ensemble.parameter_count,
                    "worker_parameter_count": ensemble.parameter_count,
                    "worker_parameter_bytes": ensemble.parameter_count * 4,
                    "candidate_parameter_bytes": ensemble.parameter_count * 4,
                    "checkpoint_write_bytes": artifact_path.stat().st_size,
                    "inference_trace_count": 1,
                    "training_update_steps": 0,
                    "wall_clock_seconds": time.perf_counter() - started,
                    "peak_working_set_method": "process_rss_before_after_lower_bound",
                },
                "candidate_promoted": False,
                "can_start_r6_formal": False,
                "can_promote": False,
                "blocking_reason": None
                if all(checks.values())
                else "fixed-large controlled canary Gate failed",
            }
        )
        report["adapter_checkpoint"] = {
            "exchange_checkpoint_digest": content_digest(adapter.checkpoint()),
            "rollback_checkpoint_digest": content_digest(rollback.checkpoint()),
            "rollback_record_digest": rollback_record.record_digest,
        }
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
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
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--model-seed", type=int, default=17)
    parser.add_argument("--course-seed", type=int, default=0)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    artifact = args.artifact if args.artifact.is_absolute() else PROJECT_ROOT / args.artifact
    report_path = args.report if args.report.is_absolute() else PROJECT_ROOT / args.report
    report = run_canary(
        artifact_path=artifact,
        model_seed=args.model_seed,
        course_seed=args.course_seed,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "report": report_path.relative_to(PROJECT_ROOT).as_posix(),
                "status": report["status"],
                "observation_path": report.get("observation_path"),
                "can_promote": report["can_promote"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
