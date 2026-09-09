"""Build and validate the explicit M4.V2.R6 formal input manifest.

The preflight materializes immutable parent checkpoints for model seeds 17, 23,
and 31, records content-addressed parent/course registries, and validates the
manifest without running a formal course.  Worker entries are deliberately not
inferred from a directory: a missing or incomplete worker registry blocks the
formal input gate.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
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

from scripts.training.eval_taiji_m4v2_r6_k_worker_attachment_preflight import (  # noqa: E402
    _restore_worker,
)
from scripts.training.eval_taiji_m4v2_r6_parent_baseline_preflight import (  # noqa: E402
    COURSE_SEEDS,
    COURSE_VARIANT_SEEDS,
    MODEL_SEEDS,
    REPEAT_SEEDS,
    _manifests,
    _parent,
    _r6_course,
)
from taiji import KWorkerManifest, KWorkerManifestBundle, Taiji, content_digest  # noqa: E402

REPORT_FORMAT = "taiji-m4v2-r6-formal-input-manifest-preflight-v1"
MANIFEST_FORMAT = "taiji-m4v2-r6-formal-runner-input-v1"
VERSION = 1
PHASE_ORDER = ("S", "G", "K")
ARM_IDS = (
    "frozen-parent",
    "matched-fixed-capacity",
    "candidate-continuation",
    "fixed-large",
    "lesion",
)
WORKER_IDS = ("k1.semantic", "k2.transition", "k3.outcome_projection")
DEFAULT_PARENT_DIR = PROJECT_ROOT / "checkpoints" / "taiji_r6_parents"
DEFAULT_WORKER_DIR = PROJECT_ROOT / "checkpoints" / "taiji_k_workers"
DEFAULT_MANIFEST = PROJECT_ROOT / "plans" / "manifests" / "taiji_m4v2_r6_formal_input_v1.json"
DEFAULT_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "taiji_m4v2_r6_formal_input_manifest_preflight_20260909.json"
)


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        raw = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(path, map_location="cpu")
    if not isinstance(raw, Mapping):
        raise TypeError(f"checkpoint at {path} must be a mapping")
    return {str(key): value for key, value in raw.items()}


def _relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"path must stay below project root: {path}") from exc
    return relative.as_posix()


def _resolve_repo_path(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("manifest path must be a non-empty string")
    candidate = (PROJECT_ROOT / Path(value.replace("/", "\\"))).resolve()
    try:
        candidate.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"manifest path escapes project root: {value}") from exc
    return candidate


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _course_entry(course_seed: int) -> dict[str, Any]:
    course = _r6_course(course_seed)
    holdout_payload = {
        "s_holdout": _sha256_bytes(course.s_holdout),
        "g_holdout": _sha256_bytes(course.g_holdout),
    }
    holdout_digest = content_digest(holdout_payload)
    source_manifest = {
        "format": "taiji-r6-course-source-manifest-v1",
        "course_seed": int(course_seed),
        "course_label": course.label,
        "course_variant_seed": int(COURSE_VARIANT_SEEDS[int(course_seed)]),
        "phase_order": list(PHASE_ORDER),
        "holdout_digest": holdout_digest,
        "root_isolation": True,
        "split_isolation": True,
        "read_only_routes_only": True,
    }
    return {
        "course_seed": int(course_seed),
        "course_label": course.label,
        "course_variant_seed": int(COURSE_VARIANT_SEEDS[int(course_seed)]),
        "source_manifest_digest": content_digest(source_manifest),
        "holdout_digest": holdout_digest,
        "phase_order": list(PHASE_ORDER),
        "workspace_contract": {
            "root_isolation": True,
            "split_isolation": True,
            "read_only_routes_only": True,
        },
    }


def _parent_entry(model_seed: int, parent_dir: Path) -> dict[str, Any]:
    parent = _parent(model_seed)
    parent_digest = content_digest(parent)
    target = parent_dir / f"model_{int(model_seed)}.pt"
    temporary = target.with_suffix(target.suffix + ".tmp")
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(copy.deepcopy(parent), temporary)
    loaded = _load_mapping(temporary)
    temporary.replace(target)
    if content_digest(loaded) != parent_digest:
        raise RuntimeError(f"parent checkpoint changed during save: model {model_seed}")
    restored = Taiji.from_checkpoint(copy.deepcopy(loaded))
    fresh_digest = content_digest(restored.checkpoint())
    if fresh_digest != parent_digest:
        raise RuntimeError(f"parent fresh restore mismatch: model {model_seed}")
    manifests = _manifests(parent_digest, model_seed=model_seed, course_seed=0)
    return {
        "model_seed": int(model_seed),
        "checkpoint_path": _relative_path(target),
        "checkpoint_digest": parent_digest,
        "owner_graph_digest": manifests["owner_graph_digest"],
        "source_manifest_digest": manifests["source_manifest_digest"],
        "resource_manifest_digest": manifests["resource_manifest_digest"],
        "namespace": f"taiji:parent:model-{int(model_seed)}",
        "fresh_restore_digest": fresh_digest,
    }


def _unsigned_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = copy.deepcopy(dict(payload))
    unsigned.pop("manifest_digest", None)
    return unsigned


def _manifest_digest(payload: Mapping[str, Any]) -> str:
    return content_digest(_unsigned_manifest(payload))


def _worker_paths(worker_dir: Path, model_seed: int) -> dict[str, Path]:
    root = worker_dir / f"model_{int(model_seed)}"
    return {
        worker_id: root / filename
        for worker_id, filename in {
            "k1.semantic": "taiji_r6_k1_semantic.pt",
            "k2.transition": "taiji_r6_k2_transition.pt",
            "k3.outcome_projection": "taiji_r6_k3_outcome_projection.pt",
        }.items()
    }


def _worker_entry(model_seed: int, worker_dir: Path, parent_digest: str) -> dict[str, Any]:
    paths = _worker_paths(worker_dir, model_seed)
    artifacts = {worker_id: _load_mapping(path) for worker_id, path in paths.items()}
    manifests: list[KWorkerManifest] = []
    artifact_digests: dict[str, str] = {}
    worker_checkpoint_digests: dict[str, str] = {}
    source_manifest_digest: str | None = None
    resource_manifest_digest: str | None = None
    candidate_namespace: str | None = None
    for worker_id in WORKER_IDS:
        artifact = artifacts[worker_id]
        manifest, _details = _restore_worker(
            worker_id,
            artifact,
            parent_digest=parent_digest,
        )
        manifests.append(manifest)
        artifact_digests[worker_id] = str(artifact["artifact_digest"])
        worker_checkpoint_digests[worker_id] = str(artifact["worker_checkpoint_digest"])
        current_source = str(artifact["source_manifest_digest"])
        current_resource = str(artifact["resource_manifest_digest"])
        current_namespace = str(artifact["candidate_namespace"])
        if source_manifest_digest is None:
            source_manifest_digest = current_source
            resource_manifest_digest = current_resource
            candidate_namespace = current_namespace
        elif (
            source_manifest_digest != current_source
            or resource_manifest_digest != current_resource
            or candidate_namespace != current_namespace
        ):
            raise ValueError(f"worker bundle metadata is not shared for model {model_seed}")
    assert source_manifest_digest is not None
    assert resource_manifest_digest is not None
    assert candidate_namespace is not None
    bundle = KWorkerManifestBundle.create(
        parent_checkpoint_digest=parent_digest,
        source_manifest_digest=source_manifest_digest,
        resource_manifest_digest=resource_manifest_digest,
        candidate_namespace=candidate_namespace,
        workers=manifests,
    )
    return {
        "model_seed": int(model_seed),
        "parent_checkpoint_digest": parent_digest,
        "candidate_namespace": candidate_namespace,
        "bundle_digest": bundle.bundle_digest,
        "owner_graph_digest": bundle.owner_graph_digest,
        "source_manifest_digest": source_manifest_digest,
        "resource_manifest_digest": resource_manifest_digest,
        "workers": {
            worker_id: {
                "path": _relative_path(paths[worker_id]),
                "artifact_digest": artifact_digests[worker_id],
                "worker_checkpoint_digest": worker_checkpoint_digests[worker_id],
            }
            for worker_id in WORKER_IDS
        },
    }


def _try_worker_registry(
    *,
    worker_dir: Path,
    parent_registry: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    parents = {int(entry["model_seed"]): str(entry["checkpoint_digest"]) for entry in parent_registry}
    try:
        return [
            _worker_entry(seed, worker_dir, parents[seed])
            for seed in MODEL_SEEDS
        ]
    except (FileNotFoundError, KeyError, TypeError, ValueError, RuntimeError):
        return []


def build_manifest(
    *,
    parent_dir: Path = DEFAULT_PARENT_DIR,
    worker_dir: Path = DEFAULT_WORKER_DIR,
) -> dict[str, Any]:
    parent_registry = [_parent_entry(seed, parent_dir) for seed in MODEL_SEEDS]
    course_registry = [_course_entry(seed) for seed in COURSE_SEEDS]
    worker_registry = _try_worker_registry(
        worker_dir=worker_dir,
        parent_registry=parent_registry,
    )
    payload: dict[str, Any] = {
        "format": MANIFEST_FORMAT,
        "version": VERSION,
        "experiment_id": "m4v2-r6-a8-promotion-formal-v1",
        "device": "cpu",
        "model_seeds": list(MODEL_SEEDS),
        "course_seeds": list(COURSE_SEEDS),
        "baseline_repeat_seeds": list(REPEAT_SEEDS),
        "phase_order": list(PHASE_ORDER),
        "arms": list(ARM_IDS),
        "parent_registry": parent_registry,
        "worker_registry": worker_registry,
        "course_registry": course_registry,
        "resource_contract": {
            "device": "cpu",
            "cuda_required": False,
            "peak_working_set_multiplier_cap": 1.25,
            "wall_clock_multiplier_cap": 1.5,
            "training_update_steps_frozen_before_run": True,
            "checkpoint_write_bytes_frozen_before_run": True,
            "candidate_parameter_bytes_frozen_before_run": True,
            "inference_trace_count_frozen_before_run": True,
        },
        "side_effect_contract": {
            "parent_namespace_read_only": True,
            "candidate_namespace_isolated": True,
            "default_runtime_attached": False,
            "provider_attached": False,
            "mcp_attached": False,
            "client_attached": False,
            "network_used": False,
            "cuda_used": False,
            "read_only_workbench_routes": True,
            "rollback_required": True,
        },
        "failure_attribution_contract": {
            "format": "taiji-r6-failure-record-v1",
            "priority": [
                "input_contract",
                "lineage",
                "checkpoint_restore",
                "course_harness",
                "environment_blocker",
                "worker_resolution",
                "workbench_outcome",
                "projection_contract",
                "causal_gate",
                "retention_gate",
                "resource_gate",
                "side_effect_gate",
                "aggregate_gate",
                "unexpected",
            ],
            "stop_on_first_cell_failure": True,
            "require_cell_arm_step_digests": True,
        },
    }
    payload["manifest_digest"] = _manifest_digest(payload)
    return payload


def _failure(
    *,
    failure_class: str,
    message: str,
    phase: str = "input",
    cell: Mapping[str, int] | None = None,
    arm: str | None = None,
    step: str | None = None,
    is_model_evidence: bool = False,
    is_environment_blocker: bool = False,
    recoverability: str = "input_fix_required",
    evidence_digests: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "class": failure_class,
        "phase": phase,
        "cell": None if cell is None else dict(cell),
        "arm": arm,
        "step": step,
        "is_model_evidence": bool(is_model_evidence),
        "is_environment_blocker": bool(is_environment_blocker),
        "stop_line": True,
        "recoverability": recoverability,
        "exception_type": None,
        "message": message,
        "evidence_digests": list(evidence_digests),
    }


def _validate_parent_entry(entry: Mapping[str, Any]) -> tuple[bool, str | None]:
    try:
        model_seed = int(entry["model_seed"])
        if model_seed not in MODEL_SEEDS:
            return False, f"unexpected parent model_seed={model_seed}"
        path = _resolve_repo_path(entry["checkpoint_path"])
        if not path.is_file():
            return False, f"parent checkpoint is missing: {entry['checkpoint_path']}"
        checkpoint = _load_mapping(path)
        checkpoint_digest = content_digest(checkpoint)
        if checkpoint_digest != str(entry["checkpoint_digest"]):
            return False, f"parent checkpoint digest mismatch for model {model_seed}"
        restored = Taiji.from_checkpoint(copy.deepcopy(checkpoint))
        fresh_digest = content_digest(restored.checkpoint())
        if fresh_digest != checkpoint_digest:
            return False, f"parent fresh restore mismatch for model {model_seed}"
        if fresh_digest != str(entry["fresh_restore_digest"]):
            return False, f"parent fresh restore digest entry mismatch for model {model_seed}"
        expected = _manifests(checkpoint_digest, model_seed=model_seed, course_seed=0)
        for key in ("owner_graph_digest", "source_manifest_digest", "resource_manifest_digest"):
            if str(entry[key]) != expected[key]:
                return False, f"parent {key} mismatch for model {model_seed}"
        expected_namespace = f"taiji:parent:model-{model_seed}"
        if str(entry["namespace"]) != expected_namespace:
            return False, f"parent namespace mismatch for model {model_seed}"
        return True, None
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        return False, f"parent registry entry invalid: {exc}"


def _validate_course_entry(entry: Mapping[str, Any]) -> tuple[bool, str | None]:
    try:
        course_seed = int(entry["course_seed"])
        if course_seed not in COURSE_SEEDS:
            return False, f"unexpected course_seed={course_seed}"
        expected = _course_entry(course_seed)
        for key in (
            "course_label",
            "course_variant_seed",
            "source_manifest_digest",
            "holdout_digest",
            "phase_order",
            "workspace_contract",
        ):
            if entry[key] != expected[key]:
                return False, f"course {key} mismatch for seed {course_seed}"
        return True, None
    except (KeyError, TypeError, ValueError) as exc:
        return False, f"course registry entry invalid: {exc}"


def _validate_worker_entry(
    entry: Mapping[str, Any],
    parent_registry: Mapping[int, Mapping[str, Any]],
) -> tuple[bool, str | None]:
    try:
        model_seed = int(entry["model_seed"])
        if model_seed not in MODEL_SEEDS:
            return False, f"unexpected worker model_seed={model_seed}"
        parent_digest = str(parent_registry[model_seed]["checkpoint_digest"])
        if str(entry["parent_checkpoint_digest"]) != parent_digest:
            return False, f"worker bundle crosses parent for model {model_seed}"
        expected_namespace = f"taiji:k:candidate:model-{model_seed}"
        if str(entry["candidate_namespace"]) != expected_namespace:
            return False, f"worker candidate namespace mismatch for model {model_seed}"
        worker_payload = entry["workers"]
        if not isinstance(worker_payload, Mapping) or tuple(sorted(worker_payload)) != tuple(sorted(WORKER_IDS)):
            return False, f"worker registry must contain K1/K2/K3 for model {model_seed}"
        paths: dict[str, Path] = {}
        for worker_id in WORKER_IDS:
            item = worker_payload[worker_id]
            if not isinstance(item, Mapping):
                return False, f"worker entry {worker_id} is not an object"
            path = _resolve_repo_path(item["path"])
            if not path.is_file():
                return False, f"worker artifact is missing: {item['path']}"
            artifact = _load_mapping(path)
            if str(artifact.get("artifact_digest")) != str(item["artifact_digest"]):
                return False, f"worker artifact digest mismatch for {worker_id}, model {model_seed}"
            if str(artifact.get("worker_checkpoint_digest")) != str(item["worker_checkpoint_digest"]):
                return False, f"worker checkpoint digest mismatch for {worker_id}, model {model_seed}"
            paths[worker_id] = path
        rebuilt = _worker_entry(model_seed, paths["k1.semantic"].parents[1], parent_digest)
        for key in (
            "candidate_namespace",
            "bundle_digest",
            "owner_graph_digest",
            "source_manifest_digest",
            "resource_manifest_digest",
        ):
            if str(entry[key]) != str(rebuilt[key]):
                return False, f"worker bundle {key} mismatch for model {model_seed}"
        for worker_id in WORKER_IDS:
            if entry["workers"][worker_id]["path"] != rebuilt["workers"][worker_id]["path"]:
                return False, f"worker path mismatch for {worker_id}, model {model_seed}"
        return True, None
    except (FileNotFoundError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        return False, f"worker registry entry invalid: {exc}"


def _validate_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    checks: dict[str, bool] = {}
    try:
        checks["format"] = payload.get("format") == MANIFEST_FORMAT
        checks["version"] = int(payload.get("version", -1)) == VERSION
        checks["device_cpu"] = payload.get("device") == "cpu"
        checks["model_seed_matrix"] = payload.get("model_seeds") == list(MODEL_SEEDS)
        checks["course_seed_matrix"] = payload.get("course_seeds") == list(COURSE_SEEDS)
        checks["baseline_repeat_matrix"] = payload.get("baseline_repeat_seeds") == list(REPEAT_SEEDS)
        checks["phase_order"] = payload.get("phase_order") == list(PHASE_ORDER)
        checks["arm_set"] = payload.get("arms") == list(ARM_IDS)
        checks["manifest_digest"] = payload.get("manifest_digest") == _manifest_digest(payload)
    except (TypeError, ValueError):
        checks.update(
            {
                "format": False,
                "version": False,
                "device_cpu": False,
                "model_seed_matrix": False,
                "course_seed_matrix": False,
                "baseline_repeat_matrix": False,
                "phase_order": False,
                "arm_set": False,
                "manifest_digest": False,
            }
        )
        failures.append(
            _failure(
                failure_class="input_contract",
                message="top-level manifest fields could not be validated",
            )
        )

    parent_registry = payload.get("parent_registry")
    parent_results: list[bool] = []
    if not isinstance(parent_registry, list) or len(parent_registry) != len(MODEL_SEEDS):
        checks["parent_registry_complete"] = False
        failures.append(
            _failure(
                failure_class="input_contract",
                message="parent_registry must contain exactly model seeds 17, 23, and 31",
            )
        )
    else:
        seen: set[int] = set()
        for entry in parent_registry:
            if not isinstance(entry, Mapping):
                parent_results.append(False)
                failures.append(
                    _failure(
                        failure_class="input_contract",
                        message="parent registry entry is not an object",
                    )
                )
                continue
            seed = int(entry.get("model_seed", -1))
            if seed in seen:
                parent_results.append(False)
                failures.append(
                    _failure(
                        failure_class="input_contract",
                        message=f"duplicate parent model seed: {seed}",
                    )
                )
                continue
            seen.add(seed)
            valid, reason = _validate_parent_entry(entry)
            parent_results.append(valid)
            if not valid:
                failures.append(
                    _failure(
                        failure_class="lineage" if "digest" in str(reason) else "checkpoint_restore",
                        message=str(reason),
                        evidence_digests=(str(entry.get("checkpoint_digest", "")),),
                    )
                )
        checks["parent_registry_complete"] = seen == set(MODEL_SEEDS)
    checks["parent_registry_fresh_restore"] = bool(parent_results) and all(parent_results)

    course_registry = payload.get("course_registry")
    course_results: list[bool] = []
    if not isinstance(course_registry, list) or len(course_registry) != len(COURSE_SEEDS):
        checks["course_registry_complete"] = False
        failures.append(
            _failure(
                failure_class="input_contract",
                message="course_registry must contain exactly course seeds 0, 1, and 2",
            )
        )
    else:
        seen_courses: set[int] = set()
        for entry in course_registry:
            if not isinstance(entry, Mapping):
                course_results.append(False)
                failures.append(
                    _failure(
                        failure_class="input_contract",
                        message="course registry entry is not an object",
                    )
                )
                continue
            seed = int(entry.get("course_seed", -1))
            if seed in seen_courses:
                course_results.append(False)
                failures.append(
                    _failure(
                        failure_class="input_contract",
                        message=f"duplicate course seed: {seed}",
                    )
                )
                continue
            seen_courses.add(seed)
            valid, reason = _validate_course_entry(entry)
            course_results.append(valid)
            if not valid:
                failures.append(
                    _failure(
                        failure_class="course_harness",
                        message=str(reason),
                    )
                )
        checks["course_registry_complete"] = seen_courses == set(COURSE_SEEDS)
    checks["course_registry_valid"] = bool(course_results) and all(course_results)

    workers = payload.get("worker_registry")
    parent_by_seed = {
        int(entry["model_seed"]): entry
        for entry in parent_registry
        if isinstance(entry, Mapping) and str(entry.get("model_seed", "")).isdigit()
    } if isinstance(parent_registry, list) else {}
    checks["worker_registry_complete"] = isinstance(workers, list) and len(workers) == len(MODEL_SEEDS)
    checks["worker_registry_valid"] = False
    if not checks["worker_registry_complete"]:
        failures.append(
            _failure(
                failure_class="input_contract",
                message="worker_registry is incomplete; formal runner may not infer workers from a directory",
                recoverability="worker_registry_required",
            )
        )
    else:
        worker_results: list[bool] = []
        worker_seeds: list[int] = []
        for entry in workers:
            if not isinstance(entry, Mapping):
                worker_results.append(False)
                failures.append(
                    _failure(
                        failure_class="input_contract",
                        message="worker registry entry is not an object",
                    )
                )
                continue
            seed = int(entry.get("model_seed", -1))
            worker_seeds.append(seed)
            valid, reason = _validate_worker_entry(entry, parent_by_seed)
            worker_results.append(valid)
            if not valid:
                failures.append(
                    _failure(
                        failure_class="lineage" if "digest" in str(reason) or "parent" in str(reason) else "checkpoint_restore",
                        message=str(reason),
                        recoverability="worker_registry_required",
                    )
                )
        checks["worker_registry_valid"] = (
            worker_seeds == list(MODEL_SEEDS)
            and bool(worker_results)
            and all(worker_results)
        )
        if not checks["worker_registry_valid"]:
            failures.append(
                _failure(
                    failure_class="lineage",
                    message="worker_registry model seeds are not exactly 17, 23, and 31",
                    recoverability="worker_registry_required",
                )
            )

    checks["resource_contract_cpu_only"] = (
        payload.get("resource_contract", {}).get("device") == "cpu"
        and payload.get("resource_contract", {}).get("cuda_required") is False
    )
    checks["side_effect_contract_closed"] = (
        payload.get("side_effect_contract", {}).get("parent_namespace_read_only") is True
        and payload.get("side_effect_contract", {}).get("candidate_namespace_isolated") is True
        and payload.get("side_effect_contract", {}).get("default_runtime_attached") is False
        and payload.get("side_effect_contract", {}).get("provider_attached") is False
        and payload.get("side_effect_contract", {}).get("mcp_attached") is False
        and payload.get("side_effect_contract", {}).get("client_attached") is False
        and payload.get("side_effect_contract", {}).get("network_used") is False
        and payload.get("side_effect_contract", {}).get("cuda_used") is False
    )
    checks["failure_contract_present"] = (
        payload.get("failure_attribution_contract", {}).get("format")
        == "taiji-r6-failure-record-v1"
    )
    if not checks["resource_contract_cpu_only"]:
        failures.append(
            _failure(
                failure_class="input_contract",
                message="resource contract is not CPU-only or requires CUDA",
            )
        )
    if not checks["side_effect_contract_closed"]:
        failures.append(
            _failure(
                failure_class="side_effect_gate",
                message="manifest side-effect contract is not fail-closed",
            )
        )
    if not checks["failure_contract_present"]:
        failures.append(
            _failure(
                failure_class="input_contract",
                message="failure attribution contract is missing or has the wrong format",
            )
        )

    formal_input_ready = all(checks.values()) and not failures
    return {
        "status": "passed" if formal_input_ready else "blocked_input",
        "checks": checks,
        "failures": failures,
        "formal_input_ready": formal_input_ready,
        "can_start_r6_formal": False,
        "can_promote": False,
    }


def run_preflight(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    report_path: Path = DEFAULT_REPORT,
    materialize_parents: bool = False,
    parent_dir: Path = DEFAULT_PARENT_DIR,
    worker_dir: Path = DEFAULT_WORKER_DIR,
) -> dict[str, Any]:
    started = time.perf_counter()
    manifest_path = manifest_path.resolve()
    report_path = report_path.resolve()
    parent_dir = parent_dir.resolve()
    worker_dir = worker_dir.resolve()
    if materialize_parents:
        payload = build_manifest(parent_dir=parent_dir, worker_dir=worker_dir)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError("formal input manifest must be a JSON object")
        validation = _validate_manifest(raw)
        manifest_digest = str(raw.get("manifest_digest", ""))
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        validation = {
            "status": "blocked_input",
            "checks": {"manifest_readable": False},
            "failures": [
                _failure(
                    failure_class="input_contract",
                    message=f"cannot load formal input manifest: {exc}",
                )
            ],
            "formal_input_ready": False,
            "can_start_r6_formal": False,
            "can_promote": False,
        }
        manifest_digest = ""
    report = {
        "report_format": REPORT_FORMAT,
        "version": VERSION,
        "created_at_unix": time.time(),
        "status": validation["status"],
        "manifest_path": _relative_path(manifest_path),
        "manifest_digest": manifest_digest,
        "parent_dir": _relative_path(parent_dir),
        "worker_dir": _relative_path(worker_dir),
        "checks": validation["checks"],
        "failures": validation["failures"],
        "formal_input_ready": validation["formal_input_ready"],
        "can_start_r6_formal": False,
        "can_promote": False,
        "training_performed": False,
        "default_runtime_attached": False,
        "provider_attached": False,
        "mcp_attached": False,
        "client_attached": False,
        "cuda_used": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--parent-dir", type=Path, default=DEFAULT_PARENT_DIR)
    parser.add_argument("--worker-dir", type=Path, default=DEFAULT_WORKER_DIR)
    parser.add_argument(
        "--materialize-parents",
        action="store_true",
        help="write and fresh-restore the three CPU parent checkpoints before validation",
    )
    args = parser.parse_args(argv)
    manifest_path = args.manifest if args.manifest.is_absolute() else PROJECT_ROOT / args.manifest
    report_path = args.report if args.report.is_absolute() else PROJECT_ROOT / args.report
    parent_dir = args.parent_dir if args.parent_dir.is_absolute() else PROJECT_ROOT / args.parent_dir
    worker_dir = args.worker_dir if args.worker_dir.is_absolute() else PROJECT_ROOT / args.worker_dir
    report = run_preflight(
        manifest_path=manifest_path,
        report_path=report_path,
        materialize_parents=args.materialize_parents,
        parent_dir=parent_dir,
        worker_dir=worker_dir,
    )
    print(
        json.dumps(
            {
                "report": _relative_path(report_path),
                "status": report["status"],
                "formal_input_ready": report["formal_input_ready"],
                "can_start_r6_formal": report["can_start_r6_formal"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
