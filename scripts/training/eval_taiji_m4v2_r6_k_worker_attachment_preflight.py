"""Preflight attachment of real K1/K2/K3 worker checkpoints to one parent.

This gate is deliberately narrower than a learner canary.  It restores worker
checkpoints that already exist, verifies their owner/source/contract lineage,
attaches the complete worker graph to one fixed-capacity parent, exercises the
typed K exchange, and round-trips the joint adapter checkpoint.  It never
trains, initializes a replacement worker, promotes a candidate, or attaches a
default runtime.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m4v2_r4_shadow import _score  # noqa: E402
from scripts.training.eval_taiji_m4v2_r6_parent_baseline_preflight import (  # noqa: E402
    _parent,
    _r6_course,
)
from taiji import (  # noqa: E402
    K_WORKER_IDS,
    K_WORKER_INPUT_CONTRACT_DIGESTS,
    K_WORKER_OUTPUT_CONTRACT_DIGESTS,
    KAdapterExchange,
    KAdapterInput,
    KAdapterOutput,
    KContinualAdapter,
    KWorkerManifest,
    KWorkerManifestBundle,
    OutcomeDependencyProjector,
    OutcomeDependencySpec,
    PerceptEvent,
    StructuredSemanticLearner,
    StructuredSemanticResult,
    StructuredSemanticTransitionLearner,
    StructuredSemanticTransitionResult,
    Taiji,
    WorldEvent,
    WorldState,
    content_digest,
)

REPORT_FORMAT = "taiji-m4v2-r6-k-worker-attachment-preflight-v1"
VERSION = 1
ARTIFACT_FORMAT = "taiji-k-worker-artifact-v1"
ARTIFACT_VERSION = 1
K3_SOURCE_FORMAT = "taiji-k3-projection-source-v1"
EPSILON = 0.01
DEFAULT_REPORT = (
    PROJECT_ROOT / "reports" / "taiji_m4v2_r6_k_worker_attachment_preflight_20260909.json"
)
DEFAULT_CANDIDATE_NAMESPACE = "taiji:k:candidate"
WORKER_KEYWORDS = {
    "k1.semantic": ("k1", "semantic"),
    "k2.transition": ("k2", "transition"),
    "k3.outcome_projection": ("k3", "outcome", "projection"),
}
CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt", ".bin", ".safetensors"}


def _load_torch_mapping(path: Path) -> dict[str, Any]:
    try:
        raw = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(path, map_location="cpu")
    if not isinstance(raw, Mapping):
        raise ValueError("checkpoint artifact must contain a mapping")
    return {str(key): value for key, value in raw.items()}


def _owner_pairs(value: Any) -> tuple[tuple[str, str], ...]:
    if isinstance(value, Mapping):
        pairs = tuple((str(key), str(item)) for key, item in value.items())
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        pairs = tuple((str(item[0]), str(item[1])) for item in value)
    else:
        raise ValueError("artifact owner_digests must be a mapping or pair sequence")
    if not pairs or len({name for name, _ in pairs}) != len(pairs):
        raise ValueError("artifact owner_digests must contain unique names")
    return tuple(sorted(pairs))


def _require_digest(value: Any, name: str) -> str:
    normalized = str(value).strip()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return normalized


def _artifact_unsigned(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "artifact_digest"}


def _discover(worker_id: str) -> list[Path]:
    roots = (
        PROJECT_ROOT / "checkpoints",
        PROJECT_ROOT / "artifacts",
        PROJECT_ROOT / "output",
    )
    keywords = WORKER_KEYWORDS[worker_id]
    found: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        try:
            paths = root.rglob("*")
            for path in paths:
                if not path.is_file() or path.suffix.lower() not in CHECKPOINT_SUFFIXES:
                    continue
                name = path.name.lower()
                if any(keyword in name for keyword in keywords):
                    found.add(path)
        except OSError:
            continue
    return sorted(found)


def _resolve_path(worker_id: str, explicit: str | None) -> tuple[Path | None, list[str]]:
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return (path if path.is_file() else None, [str(path)])
    candidates = _discover(worker_id)
    if len(candidates) == 1:
        return candidates[0], [str(candidates[0])]
    return None, [str(path) for path in candidates]


def _restore_worker(
    worker_id: str,
    artifact: Mapping[str, Any],
    *,
    parent_digest: str,
) -> tuple[KWorkerManifest, dict[str, Any]]:
    if artifact.get("format") != ARTIFACT_FORMAT:
        raise ValueError("unsupported K worker artifact format")
    if int(artifact.get("version", -1)) != ARTIFACT_VERSION:
        raise ValueError("unsupported K worker artifact version")
    if str(artifact.get("worker_id", "")) != worker_id:
        raise ValueError("worker artifact id does not match requested worker")
    source_manifest_digest = _require_digest(
        artifact.get("source_manifest_digest"), "source_manifest_digest"
    )
    resource_manifest_digest = _require_digest(
        artifact.get("resource_manifest_digest"), "resource_manifest_digest"
    )
    candidate_namespace = str(artifact.get("candidate_namespace", "")).strip()
    if not candidate_namespace:
        raise ValueError("candidate_namespace cannot be empty")
    if str(artifact.get("artifact_digest", "")) != content_digest(_artifact_unsigned(artifact)):
        raise ValueError("K worker artifact digest mismatch")
    if str(artifact.get("parent_checkpoint_digest", "")) != parent_digest:
        raise ValueError("K worker artifact crosses the selected parent")
    checkpoint = artifact.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("K worker artifact checkpoint is missing")
    checkpoint = dict(checkpoint)
    checkpoint_digest = content_digest(checkpoint)
    if str(artifact.get("worker_checkpoint_digest", "")) != checkpoint_digest:
        raise ValueError("K worker checkpoint digest mismatch")

    worker: Any
    if worker_id == "k1.semantic":
        worker = StructuredSemanticLearner.from_checkpoint(checkpoint, device="cpu")
        derived_owner = worker.owner_digests()
        derived_source = worker.source_digest
        derived_steps = int(worker.training_steps)
        probe = PerceptEvent(
            event_id="r6-k1-attachment-probe",
            observation_tick=0,
            modality="attachment-probe",
            features=torch.zeros(worker.feature_dim),
        )
        probe_result = worker.predict(probe)
        probe_type = type(probe_result).__name__
        if not isinstance(probe_result, StructuredSemanticResult):
            raise ValueError("K1 probe did not return StructuredSemanticResult")
        extra = {"probe_type": probe_type, "feature_dim": worker.feature_dim}
    elif worker_id == "k2.transition":
        worker = StructuredSemanticTransitionLearner.from_checkpoint(checkpoint, device="cpu")
        derived_owner = worker.owner_digests()
        derived_source = worker.source_digest
        derived_steps = int(worker.training_steps)
        probe_event = PerceptEvent(
            event_id="r6-k2-attachment-probe",
            observation_tick=0,
            modality="attachment-probe",
            features=torch.zeros(worker.event_dim),
        )
        probe_result = worker.predict(
            WorldState(tick=0, entities=(), uncertainty=0.0),
            probe_event,
        )
        probe_type = type(probe_result).__name__
        if not isinstance(probe_result, StructuredSemanticTransitionResult):
            raise ValueError("K2 probe did not return StructuredSemanticTransitionResult")
        extra = {"probe_type": probe_type, "event_dim": worker.event_dim}
    elif worker_id == "k3.outcome_projection":
        worker = OutcomeDependencyProjector.from_checkpoint(checkpoint)
        derived_owner = {
            "outcome_dependency_projector": checkpoint_digest,
        }
        derived_source = content_digest(
            {
                "format": K3_SOURCE_FORMAT,
                "scope_id": worker.scope_id,
                "lesioned": bool(worker.lesioned),
                "checkpoint_digest": checkpoint_digest,
            }
        )
        derived_steps = 0
        extra = {
            "scope_id": worker.scope_id,
            "lesioned": bool(worker.lesioned),
        }
    else:
        raise ValueError("unsupported K worker id")

    owner_pairs = _owner_pairs(artifact.get("owner_digests"))
    if dict(owner_pairs) != {str(key): str(value) for key, value in derived_owner.items()}:
        raise ValueError("K worker owner digest does not match restored state")
    if str(artifact.get("source_digest", "")) != derived_source:
        raise ValueError("K worker source digest does not match restored state")
    if int(artifact.get("training_steps", -1)) != derived_steps:
        raise ValueError("K worker training_steps does not match restored state")
    if bool(artifact.get("optimizer_state_present", True)):
        raise ValueError("worker artifact carries optimizer state that cannot be restored")
    if str(artifact.get("input_contract_digest", "")) != K_WORKER_INPUT_CONTRACT_DIGESTS[worker_id]:
        raise ValueError("K worker input contract digest mismatch")
    if (
        str(artifact.get("output_contract_digest", ""))
        != K_WORKER_OUTPUT_CONTRACT_DIGESTS[worker_id]
    ):
        raise ValueError("K worker output contract digest mismatch")

    manifest = KWorkerManifest.create(
        worker_id=worker_id,
        checkpoint_format=str(checkpoint.get("format", "")),
        checkpoint_version=int(checkpoint.get("version", -1)),
        worker_checkpoint_digest=checkpoint_digest,
        owner_digests=owner_pairs,
        source_digest=derived_source,
        input_contract_digest=str(artifact["input_contract_digest"]),
        output_contract_digest=str(artifact["output_contract_digest"]),
        parent_checkpoint_digest=parent_digest,
        candidate_namespace=candidate_namespace,
        training_steps=derived_steps,
        optimizer_state_present=False,
    )
    return manifest, {
        "checkpoint_digest": checkpoint_digest,
        "checkpoint_format": manifest.checkpoint_format,
        "checkpoint_version": manifest.checkpoint_version,
        "owner_digests": dict(owner_pairs),
        "source_digest": derived_source,
        "source_manifest_digest": source_manifest_digest,
        "resource_manifest_digest": resource_manifest_digest,
        "candidate_namespace": candidate_namespace,
        "training_steps": derived_steps,
        **extra,
    }


def _projection(scope_id: str) -> Any:
    world = WorldState(tick=0, entities=("workbench",), uncertainty=0.0)
    event = WorldEvent(
        event_id="r6-worker-attachment-outcome",
        kind="workbench.evidence",
        tick=0,
        subject_id="workspace.read",
        attributes=(("capability_id", "workspace.read"), ("success", True)),
        provenance="r6-worker-attachment-preflight",
    )
    spec = OutcomeDependencySpec(
        dependency_id="r6-worker-attachment-dependency",
        next_task_id="r6-worker-attachment-follow-up",
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
        episode_id="r6-worker-attachment-episode",
        parent_checkpoint_digest=parent_digest,
        observation_digest=content_digest({"kind": "attachment.probe"}),
        world_digest=content_digest({"kind": "world.state", "tick": 0}),
        goal_digest=content_digest({"goal_id": "r6-worker-attachment"}),
        content_plan_digest=content_digest({"content_id": "r6-worker-attachment"}),
        source_manifest_digest=adapter.source_manifest_digest,
        tick=1,
    )
    output_item = KAdapterOutput(
        parent_checkpoint_digest=parent_digest,
        input_digest=input_item.input_digest,
        action_digest=content_digest({"kind": "workspace.read", "path": "probe"}),
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


def _scores(parent: Mapping[str, Any], course_seed: int) -> dict[str, float]:
    model = Taiji.from_checkpoint(copy.deepcopy(dict(parent)))
    course = _r6_course(course_seed)
    return {
        "S": float(_score(model, course.s_holdout, phase="K-attachment-S")),
        "G": float(_score(model, course.g_holdout, phase="K-attachment-G")),
    }


def _base_report(
    *,
    parent_digest: str,
    artifact_paths: Mapping[str, Sequence[str]],
    missing: Sequence[str],
    errors: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "report_format": REPORT_FORMAT,
        "version": VERSION,
        "created_at_unix": time.time(),
        "artifact_format": ARTIFACT_FORMAT,
        "artifact_paths": {str(key): list(value) for key, value in artifact_paths.items()},
        "artifact_missing": list(missing),
        "artifact_errors": dict(errors),
        "parent_checkpoint_digest": parent_digest,
        "workers": {},
        "checks": {},
        "adapter_checkpoint": None,
        "training_performed": False,
        "candidate_training_performed": False,
        "adapter_training_steps": 0,
        "default_runtime_attached": False,
        "cuda_used": False,
        "provider_attached": False,
        "mcp_attached": False,
        "client_attached": False,
        "attachment_gate_passed": False,
        "can_start_k_controlled_canary": False,
        "can_start_r6_formal": False,
        "can_promote": False,
    }


def run_preflight(
    *,
    semantic_checkpoint: str | None = None,
    transition_checkpoint: str | None = None,
    projection_checkpoint: str | None = None,
    model_seed: int = 17,
    course_seed: int = 0,
    candidate_namespace: str = DEFAULT_CANDIDATE_NAMESPACE,
) -> dict[str, Any]:
    parent = _parent(model_seed)
    parent_digest = content_digest(parent)
    arguments = {
        "k1.semantic": semantic_checkpoint,
        "k2.transition": transition_checkpoint,
        "k3.outcome_projection": projection_checkpoint,
    }
    resolved: dict[str, Path | None] = {}
    artifact_paths: dict[str, list[str]] = {}
    missing: list[str] = []
    for worker_id in K_WORKER_IDS:
        path, candidates = _resolve_path(worker_id, arguments[worker_id])
        resolved[worker_id] = path
        artifact_paths[worker_id] = candidates
        if path is None:
            missing.append(worker_id)
    report = _base_report(
        parent_digest=parent_digest,
        artifact_paths=artifact_paths,
        missing=missing,
        errors={},
    )
    report["parent_restore"] = {
        "fresh_restore": content_digest(Taiji.from_checkpoint(copy.deepcopy(parent)).checkpoint())
        == parent_digest,
        "rollback_restore": content_digest(
            Taiji.from_checkpoint(copy.deepcopy(parent)).checkpoint()
        )
        == parent_digest,
    }
    if missing:
        report["status"] = "artifact_missing"
        report["blocking_reason"] = (
            "real K1/K2/K3 worker artifacts are required; no random initialization "
            "or standalone report can satisfy attachment"
        )
        return report

    manifests: list[KWorkerManifest] = []
    restored_details: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for worker_id in K_WORKER_IDS:
        path = resolved[worker_id]
        if path is None:
            errors[worker_id] = "artifact path resolution failed"
            continue
        try:
            artifact = _load_torch_mapping(path)
            manifest, details = _restore_worker(
                worker_id,
                artifact,
                parent_digest=parent_digest,
            )
            if manifest.candidate_namespace != candidate_namespace:
                raise ValueError("worker candidate namespace differs from preflight namespace")
            manifests.append(manifest)
            restored_details[worker_id] = details
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            errors[worker_id] = str(exc)
    report["artifact_errors"] = errors
    report["workers"] = restored_details
    if errors or len(manifests) != len(K_WORKER_IDS):
        report["status"] = "failed"
        report["blocking_reason"] = (
            "one or more real worker artifacts failed restore/lineage validation"
        )
        return report

    artifact_payloads: dict[str, dict[str, Any]] = {}
    for worker_id in K_WORKER_IDS:
        path = resolved[worker_id]
        if path is None:
            raise RuntimeError("worker artifact path disappeared during preflight")
        artifact_payloads[worker_id] = _load_torch_mapping(path)
    source_manifest_values = {
        str(payload["source_manifest_digest"]) for payload in artifact_payloads.values()
    }
    resource_manifest_values = {
        str(payload["resource_manifest_digest"]) for payload in artifact_payloads.values()
    }
    if len(source_manifest_values) != 1 or len(resource_manifest_values) != 1:
        report["status"] = "failed"
        report["blocking_reason"] = "worker artifacts do not share one source/resource manifest"
        report["checks"] = {
            "source_manifest_shared": len(source_manifest_values) == 1,
            "resource_manifest_shared": len(resource_manifest_values) == 1,
        }
        return report

    source_manifest_digest = next(iter(source_manifest_values))
    resource_manifest_digest = next(iter(resource_manifest_values))
    try:
        bundle = KWorkerManifestBundle.create(
            parent_checkpoint_digest=parent_digest,
            source_manifest_digest=source_manifest_digest,
            resource_manifest_digest=resource_manifest_digest,
            candidate_namespace=candidate_namespace,
            workers=manifests,
        )
        scope_id = str(restored_details["k3.outcome_projection"]["scope_id"])
        adapter = KContinualAdapter(
            parent_checkpoint_digest=parent_digest,
            owner_graph_digest=bundle.owner_graph_digest,
            source_manifest_digest=source_manifest_digest,
            resource_manifest_digest=resource_manifest_digest,
            dependency_scope_id=scope_id,
            candidate_namespace=candidate_namespace,
        )
        adapter.bind_worker_bundle(bundle)
        projection = _projection(scope_id)
        adapter.bind_dependency_projection(projection)
        exchange = _exchange(adapter, projection, parent_digest)
        adapter.record_exchange(exchange)
        exchange_checkpoint = adapter.checkpoint()
        exchange_restored = KContinualAdapter.from_checkpoint(exchange_checkpoint)

        before = _scores(parent, course_seed)
        after = _scores(parent, course_seed)
        retention = {phase: abs(after[phase] - before[phase]) <= EPSILON for phase in ("S", "G")}
        token = adapter.stage_candidate(
            candidate_checkpoint_digest=content_digest(
                {"format": "r6-k-worker-candidate-v1", "parent": parent_digest}
            ),
            candidate_owner_graph_digest=content_digest(
                {"owner_graph_digest": bundle.owner_graph_digest, "kind": "candidate"}
            ),
            candidate_source_manifest_digest=content_digest(
                {"source_manifest_digest": source_manifest_digest, "kind": "candidate"}
            ),
            candidate_parent_checkpoint_digest=parent_digest,
        )
        staged = KContinualAdapter.from_checkpoint(adapter.checkpoint())
        rollback_record = adapter.rollback(token)
        rollback = KContinualAdapter.from_checkpoint(adapter.checkpoint())
        checks = {
            "parent_fresh_restore": bool(report["parent_restore"]["fresh_restore"]),
            "parent_rollback_restore": bool(report["parent_restore"]["rollback_restore"]),
            "worker_bundle_exactly_k1_k2_k3": tuple(item.worker_id for item in bundle.workers)
            == K_WORKER_IDS,
            "worker_bundle_same_parent": all(
                item.parent_checkpoint_digest == parent_digest for item in bundle.workers
            ),
            "worker_owner_graph_bound": adapter.worker_bundle == bundle,
            "typed_exchange_parent_echo": (
                exchange.output.parent_checkpoint_digest == parent_digest
            ),
            "typed_exchange_worker_bundle_roundtrip": (exchange_restored.worker_bundle == bundle),
            "typed_exchange_ledger_roundtrip": (exchange_restored.last_exchange == exchange),
            "candidate_stage_worker_bundle_roundtrip": staged.worker_bundle == bundle,
            "rollback_worker_bundle_roundtrip": rollback.worker_bundle == bundle,
            "rollback_record_explicit": (
                rollback_record.status == "rolled_back"
                and rollback_record.reason == "explicit_parent_restore"
            ),
            "rollback_parent_namespace_restored": (
                rollback.active_namespace == rollback.parent_namespace
            ),
            "old_S_retention": retention["S"],
            "old_G_retention": retention["G"],
            "adapter_training_steps_zero": adapter.training_steps == 0,
            "no_default_runtime": report["default_runtime_attached"] is False,
            "no_cuda": report["cuda_used"] is False,
            "no_provider_mcp_client": not any(
                (
                    report["provider_attached"],
                    report["mcp_attached"],
                    report["client_attached"],
                )
            ),
        }
        report["status"] = "passed" if all(checks.values()) else "failed"
        report["checks"] = checks
        report["adapter_checkpoint"] = {
            "checkpoint_digest": content_digest(exchange_checkpoint),
            "worker_bundle_digest": bundle.bundle_digest,
            "owner_graph_digest": bundle.owner_graph_digest,
            "exchange_digest": exchange.exchange_digest,
            "rollback_record_digest": rollback_record.record_digest,
        }
        report["source_manifest_digest"] = source_manifest_digest
        report["resource_manifest_digest"] = resource_manifest_digest
        report["worker_bundle_digest"] = bundle.bundle_digest
        report["owner_graph_digest"] = bundle.owner_graph_digest
        report["old_capability_before"] = before
        report["old_capability_after"] = after
        report["old_capability_retention"] = retention
        report["candidate_training_performed"] = False
        report["attachment_gate_passed"] = report["status"] == "passed"
        report["can_start_k_controlled_canary"] = report["attachment_gate_passed"]
        report["can_start_r6_formal"] = False
        report["blocking_reason"] = (
            (
                "attachment passed; R6 formal remains closed until the controlled K "
                "canary and full S/G/K promotion Gate run"
            )
            if report["status"] == "passed"
            else "attachment checks failed; R6 formal remains closed"
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        report["status"] = "failed"
        report["blocking_reason"] = str(exc)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-checkpoint")
    parser.add_argument("--transition-checkpoint")
    parser.add_argument("--projection-checkpoint")
    parser.add_argument("--model-seed", type=int, default=17)
    parser.add_argument("--course-seed", type=int, default=0)
    parser.add_argument("--candidate-namespace", default=DEFAULT_CANDIDATE_NAMESPACE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = run_preflight(
        semantic_checkpoint=args.semantic_checkpoint,
        transition_checkpoint=args.transition_checkpoint,
        projection_checkpoint=args.projection_checkpoint,
        model_seed=args.model_seed,
        course_seed=args.course_seed,
        candidate_namespace=args.candidate_namespace,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] in {"passed", "artifact_missing"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
