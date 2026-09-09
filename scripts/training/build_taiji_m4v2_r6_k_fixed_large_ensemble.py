"""Build the native K fixed-large ensemble control for one R6 parent.

Two independent K1/K2 native worker replicas are trained on task slices 3 and
4.  The resulting envelope is a content-addressed arithmetic ensemble; it is
not attached to the default runtime and is only consumed by the R6 single-cell
comparator.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.build_taiji_m4v2_r6_k_worker_artifacts import (  # noqa: E402
    _course_inputs,
    _load_mapping,
)
from scripts.training.build_taiji_m4v2_r6_k_worker_artifacts import (  # noqa: E402
    run_build as build_worker_artifacts,
)
from scripts.training.eval_taiji_m4v2_r6_k_worker_attachment_preflight import (  # noqa: E402
    K3_SOURCE_FORMAT,
)
from scripts.training.eval_taiji_m4v2_r6_parent_baseline_preflight import (  # noqa: E402
    _parent,
)
from scripts.training.eval_taiji_m5_k2_multistep_composition import (  # noqa: E402
    _build_workspace,
    _episode,
    _holdout_episode_paths,
    _observe_all,
    _registry,
    _schema,
    _transition_examples,
)
from taiji import (  # noqa: E402
    K_WORKER_INPUT_CONTRACT_DIGESTS,
    K_WORKER_OUTPUT_CONTRACT_DIGESTS,
    OutcomeDependencyProjector,
    StructuredSemanticLearner,
    StructuredSemanticTransitionLearner,
    content_digest,
    semantic_input_digest,
)
from taiji.k_fixed_large import (  # noqa: E402
    FIXED_LARGE_CHECKPOINT_FORMAT,
    FIXED_LARGE_CHECKPOINT_VERSION,
    NativeKFixedLargeEnsemble,
)

REPORT_FORMAT = "taiji-m4v2-r6-k-fixed-large-ensemble-build-v1"
VERSION = 1
ENSEMBLE_WIDTH = 2
WORKER_TASK_SEEDS = (3, 4)
FORMAL_HOLDOUT_SEEDS = (0, 1, 2)
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "checkpoints" / "taiji_k_fixed_large" / "model_17"
DEFAULT_REPORT = (
    PROJECT_ROOT / "reports" / "taiji_m4v2_r6_k_fixed_large_ensemble_build_model_17_20260909.json"
)
DEFAULT_NAMESPACE = "taiji:k:fixed-large:model-17"


def _atomic_save(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    try:
        raw = torch.load(temporary, map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(temporary, map_location="cpu")
    if not isinstance(raw, Mapping):
        raise TypeError("fixed-large artifact roundtrip did not return a mapping")
    loaded = {str(key): value for key, value in raw.items()}
    if content_digest(loaded) != content_digest(dict(payload)):
        raise ValueError("fixed-large artifact changed during disk roundtrip")
    temporary.replace(path)
    return loaded


def _artifact_ref(
    artifact: Mapping[str, Any], *, task_seed: int, worker_id: str
) -> dict[str, Any]:
    required = (
        "artifact_digest",
        "worker_checkpoint_digest",
        "source_digest",
        "source_manifest_digest",
        "resource_manifest_digest",
        "parent_checkpoint_digest",
        "candidate_namespace",
        "checkpoint",
    )
    missing = [key for key in required if key not in artifact]
    if missing:
        raise ValueError(f"fixed-large {worker_id} artifact missing {missing}")
    return {
        "worker_id": worker_id,
        "worker_training_task_seed": int(task_seed),
        "artifact_digest": str(artifact["artifact_digest"]),
        "worker_checkpoint_digest": str(artifact["worker_checkpoint_digest"]),
        "source_digest": str(artifact["source_digest"]),
        "source_manifest_digest": str(artifact["source_manifest_digest"]),
        "resource_manifest_digest": str(artifact["resource_manifest_digest"]),
        "parent_checkpoint_digest": str(artifact["parent_checkpoint_digest"]),
        "candidate_namespace": str(artifact["candidate_namespace"]),
        "training_steps": int(artifact.get("training_steps", 0)),
        "optimizer_state_present": bool(artifact.get("optimizer_state_present", True)),
        "checkpoint": dict(artifact["checkpoint"]),
    }


def _public_artifact_ref(ref: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in ref.items() if key != "checkpoint"}


def _holdout_input_digests(*, task_seed: int) -> dict[str, object]:
    root_parent = PROJECT_ROOT / ".tmp-m4v2-r6-k-fixed-large-holdout"
    root_parent.mkdir(parents=True, exist_ok=True)
    root = root_parent / f"holdout-{uuid4().hex}"
    root.mkdir()
    try:
        _build_workspace(root, task_seed=task_seed)
        schema = _schema()
        registry = _registry(typescript_available=True)
        holdout_paths = sorted({path for episode in _holdout_episode_paths() for path in episode})
        observations = {
            observation.path: observation
            for observation in _observe_all(
                root,
                registry=registry,
                split="fixed-large-holdout",
                paths=holdout_paths,
                schema=schema,
            )
        }
        anchor = _observe_all(
            root,
            registry=registry,
            split="fixed-large-holdout-anchor",
            paths=["missing_00.txt"],
            schema=schema,
        )[0]
        episodes = tuple(
            _episode(anchor, observations, paths) for paths in _holdout_episode_paths()
        )
        semantic_digests = sorted(
            semantic_input_digest(observation.to_percept_event(tick=index + 1))
            for index, observation in enumerate(observations.values())
        )
        transition_digests = sorted(
            example.input_digest
            for index, sequence in enumerate(episodes)
            for example in _transition_examples(sequence, split=f"fixed-large-{index}")
        )
        return {
            "task_seed": int(task_seed),
            "paths": holdout_paths,
            "semantic_input_digests": semantic_digests,
            "transition_input_digests": transition_digests,
            "observation_digest": content_digest(
                {
                    "paths": holdout_paths,
                    "semantic_input_digests": semantic_digests,
                    "transition_input_digests": transition_digests,
                }
            ),
        }
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _worker_fit_inputs(*, task_seed: int) -> dict[str, object]:
    root_parent = PROJECT_ROOT / ".tmp-m4v2-r6-k-fixed-large-source"
    root_parent.mkdir(parents=True, exist_ok=True)
    root = root_parent / f"source-{uuid4().hex}"
    root.mkdir()
    try:
        semantic_corpus, transition_corpus, _ = _course_inputs(root, task_seed=task_seed)
        semantic_digests = sorted(item.input_digest for item in semantic_corpus.train)
        transition_digests = sorted(item.input_digest for item in transition_corpus.train)
        train_paths = ["missing_00.txt"]
        for path_group in (
            ("python_00.py", "rust_00.rs", "python_01.py"),
            ("rust_01.rs", "python_02.py", "typescript_00.ts"),
            ("typescript_01.ts", "rust_02.rs", "python_03.py"),
            ("python_02.py", "typescript_02.ts", "rust_03.rs"),
            ("rust_00.rs", "typescript_03.ts", "python_00.py"),
            ("typescript_02.ts", "python_01.py", "rust_01.rs"),
        ):
            train_paths.extend(path_group)
        return {
            "task_seed": int(task_seed),
            "paths": sorted(set(train_paths)),
            "semantic_input_digests": semantic_digests,
            "transition_input_digests": transition_digests,
            "fit_digest": content_digest(
                {
                    "semantic_input_digests": semantic_digests,
                    "transition_input_digests": transition_digests,
                }
            ),
        }
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _build_source_manifest(
    *,
    parent_digest: str,
    replicas: Sequence[Mapping[str, Any]],
    source_inputs: Sequence[Mapping[str, Any]],
    holdout_inputs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    semantic_fit = sorted(
        {
            str(digest)
            for item in source_inputs
            for digest in item["semantic_input_digests"]  # type: ignore[index]
        }
    )
    transition_fit = sorted(
        {
            str(digest)
            for item in source_inputs
            for digest in item["transition_input_digests"]  # type: ignore[index]
        }
    )
    semantic_holdout = sorted(
        {
            str(digest)
            for item in holdout_inputs
            for digest in item["semantic_input_digests"]  # type: ignore[index]
        }
    )
    transition_holdout = sorted(
        {
            str(digest)
            for item in holdout_inputs
            for digest in item["transition_input_digests"]  # type: ignore[index]
        }
    )
    semantic_overlap = sorted(set(semantic_fit) & set(semantic_holdout))
    transition_overlap = sorted(set(transition_fit) & set(transition_holdout))
    path_sets = {
        "training": sorted(
            {
                str(path)
                for item in source_inputs
                for path in item["paths"]  # type: ignore[index]
                if str(path) != "missing_00.txt"
            }
        ),
        "formal_holdout": sorted(
            {
                str(path)
                for item in holdout_inputs
                for path in item["paths"]  # type: ignore[index]
            }
        ),
    }
    path_overlap = sorted(set(path_sets["training"]) & set(path_sets["formal_holdout"]))
    if semantic_overlap or transition_overlap or path_overlap:
        raise ValueError(
            "fixed-large worker training overlaps formal holdout: "
            f"semantic={semantic_overlap}, transition={transition_overlap}, paths={path_overlap}"
        )
    return {
        "format": "taiji-k-fixed-large-source-manifest-v1",
        "version": 1,
        "parent_checkpoint_digest": parent_digest,
        "worker_training_task_seeds": list(WORKER_TASK_SEEDS),
        "formal_holdout_task_seeds": list(FORMAL_HOLDOUT_SEEDS),
        "worker_training_inputs": list(source_inputs),
        "formal_holdout_inputs": list(holdout_inputs),
        "path_sets": path_sets,
        "semantic_fit_digest": content_digest(semantic_fit),
        "transition_fit_digest": content_digest(transition_fit),
        "semantic_holdout_digest": content_digest(semantic_holdout),
        "transition_holdout_digest": content_digest(transition_holdout),
        "source_observation_overlap": {
            "semantic_input_digests": semantic_overlap,
            "transition_input_digests": transition_overlap,
            "paths": path_overlap,
        },
        "replica_artifact_digests": [
            {
                "task_seed": int(item["task_seed"]),
                "k1": str(item["k1"]["artifact_digest"]),  # type: ignore[index]
                "k2": str(item["k2"]["artifact_digest"]),  # type: ignore[index]
            }
            for item in replicas
        ],
    }


def run_build(
    *,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    model_seed: int = 17,
    candidate_namespace: str = DEFAULT_NAMESPACE,
) -> dict[str, Any]:
    started = time.perf_counter()
    parent = _parent(model_seed)
    parent_digest = content_digest(parent)
    replica_reports: list[dict[str, Any]] = []
    replica_payloads: list[dict[str, Any]] = []
    source_inputs = [_worker_fit_inputs(task_seed=seed) for seed in WORKER_TASK_SEEDS]
    holdout_inputs = [
        _holdout_input_digests(task_seed=seed) for seed in FORMAL_HOLDOUT_SEEDS
    ]
    replica_root = artifact_dir / "replicas"
    for task_seed in WORKER_TASK_SEEDS:
        replica_dir = replica_root / f"task_{task_seed}"
        report = build_worker_artifacts(
            artifact_dir=replica_dir,
            task_seed=task_seed,
            learner_seed=model_seed,
            candidate_namespace=(
                f"{candidate_namespace}:replica-{task_seed}"
            ),
        )
        if report.get("status") != "passed":
            raise RuntimeError(f"fixed-large worker build failed for task seed {task_seed}")
        paths = {
            "k1": replica_dir / "taiji_r6_k1_semantic.pt",
            "k2": replica_dir / "taiji_r6_k2_transition.pt",
            "k3": replica_dir / "taiji_r6_k3_outcome_projection.pt",
        }
        artifacts = {name: _load_mapping(path) for name, path in paths.items()}
        for name, artifact in artifacts.items():
            if str(artifact["parent_checkpoint_digest"]) != parent_digest:
                raise ValueError(f"fixed-large {name} replica crosses selected parent")
            if bool(artifact.get("optimizer_state_present", True)):
                raise ValueError(f"fixed-large {name} replica carries optimizer state")
        replica_payloads.append(
            {
                "task_seed": int(task_seed),
                "k1": _artifact_ref(artifacts["k1"], task_seed=task_seed, worker_id="k1.semantic"),
                "k2": _artifact_ref(artifacts["k2"], task_seed=task_seed, worker_id="k2.transition"),
                "k3": _artifact_ref(
                    artifacts["k3"], task_seed=task_seed, worker_id="k3.outcome_projection"
                ),
                "report": report,
            }
        )
        replica_reports.append(report)

    semantic_replicas = tuple(
        StructuredSemanticLearner.from_checkpoint(
            item["k1"]["checkpoint"], device="cpu"  # type: ignore[index]
        )
        for item in replica_payloads
    )
    transition_replicas = tuple(
        StructuredSemanticTransitionLearner.from_checkpoint(
            item["k2"]["checkpoint"], device="cpu"  # type: ignore[index]
        )
        for item in replica_payloads
    )
    ensemble = NativeKFixedLargeEnsemble(semantic_replicas, transition_replicas)
    ensemble_checkpoint = ensemble.checkpoint()
    restored_ensemble = NativeKFixedLargeEnsemble.from_checkpoint(ensemble_checkpoint)
    if content_digest(restored_ensemble.checkpoint()) != content_digest(ensemble_checkpoint):
        raise ValueError("fixed-large ensemble fresh restore digest mismatch")

    projector = OutcomeDependencyProjector(f"r6-k3-fixed-large-{model_seed}")
    projector_checkpoint = projector.checkpoint()
    restored_projector = OutcomeDependencyProjector.from_checkpoint(projector_checkpoint)
    projector_digest = content_digest(projector_checkpoint)
    k3_source_digest = content_digest(
        {
            "format": K3_SOURCE_FORMAT,
            "scope_id": projector.scope_id,
            "lesioned": bool(projector.lesioned),
            "checkpoint_digest": projector_digest,
        }
    )
    source_manifest = _build_source_manifest(
        parent_digest=parent_digest,
        replicas=replica_payloads,
        source_inputs=source_inputs,
        holdout_inputs=holdout_inputs,
    )
    source_manifest_digest = content_digest(source_manifest)
    resource_manifest = {
        "format": "taiji-r6-cpu-resource-manifest-v1",
        "version": 1,
        "device": "cpu",
        "cuda_required": False,
        "optimizer_state_present": False,
        "training_backend": "native-delta",
        "ensemble_width": ENSEMBLE_WIDTH,
    }
    resource_manifest_digest = content_digest(resource_manifest)
    k3_ref = {
        "worker_id": "k3.outcome_projection",
        "artifact_digest": content_digest(
            {
                "format": "taiji-k-fixed-large-k3-artifact-v1",
                "parent_checkpoint_digest": parent_digest,
                "checkpoint": projector_checkpoint,
                "source_digest": k3_source_digest,
                "resource_manifest_digest": resource_manifest_digest,
            }
        ),
        "worker_checkpoint_digest": projector_digest,
        "source_digest": k3_source_digest,
        "checkpoint": projector_checkpoint,
    }
    unsigned = {
        "format": FIXED_LARGE_CHECKPOINT_FORMAT,
        "version": FIXED_LARGE_CHECKPOINT_VERSION,
        "parent_checkpoint_digest": parent_digest,
        "candidate_namespace": candidate_namespace,
        "ensemble_width": ENSEMBLE_WIDTH,
        "worker_training_task_seeds": list(WORKER_TASK_SEEDS),
        "k1_replicas": [item["k1"] for item in replica_payloads],
        "k2_replicas": [item["k2"] for item in replica_payloads],
        "k3_projection": k3_ref,
        "input_contract_digests": dict(K_WORKER_INPUT_CONTRACT_DIGESTS),
        "output_contract_digests": dict(K_WORKER_OUTPUT_CONTRACT_DIGESTS),
        "source_manifest": source_manifest,
        "source_manifest_digest": source_manifest_digest,
        "resource_manifest": resource_manifest,
        "resource_manifest_digest": resource_manifest_digest,
        "ensemble_checkpoint": ensemble_checkpoint,
        "ensemble_checkpoint_digest": content_digest(ensemble_checkpoint),
        "owner_graph_digest": content_digest(
            {"ensemble": ensemble.owner_digests, "k3": projector_digest}
        ),
        "optimizer_state_present": False,
        "training_steps": {
            "k1": [int(item["k1"]["training_steps"]) for item in replica_payloads],
            "k2": [int(item["k2"]["training_steps"]) for item in replica_payloads],
            "k3": 0,
        },
    }
    payload = {**unsigned, "ensemble_digest": content_digest(unsigned)}
    artifact_path = artifact_dir / "taiji_r6_k_fixed_large_ensemble.pt"
    saved = _atomic_save(artifact_path, payload)
    restored_payload = _load_mapping(artifact_path)
    restored_from_artifact = NativeKFixedLargeEnsemble.from_checkpoint(
        restored_payload["ensemble_checkpoint"]
    )
    report = {
        "report_format": REPORT_FORMAT,
        "version": VERSION,
        "created_at_unix": time.time(),
        "status": "passed",
        "model_seed": int(model_seed),
        "parent_checkpoint_digest": parent_digest,
        "candidate_namespace": candidate_namespace,
        "artifact_path": str(artifact_path),
        "artifact_digest": str(saved["ensemble_digest"]),
        "ensemble_checkpoint_digest": str(saved["ensemble_checkpoint_digest"]),
        "source_manifest_digest": source_manifest_digest,
        "resource_manifest_digest": resource_manifest_digest,
        "ensemble_width": ENSEMBLE_WIDTH,
        "worker_training_task_seeds": list(WORKER_TASK_SEEDS),
        "formal_holdout_task_seeds": list(FORMAL_HOLDOUT_SEEDS),
        "replicas": [
            {
                "task_seed": int(item["task_seed"]),
                "k1": _public_artifact_ref(item["k1"]),
                "k2": _public_artifact_ref(item["k2"]),
                "k3": _public_artifact_ref(item["k3"]),
            }
            for item in replica_payloads
        ],
        "replica_reports": replica_reports,
        "prefit_checkpoint_gate": {
            str(seed): report["prefit_checkpoint_gate"] for seed, report in zip(WORKER_TASK_SEEDS, replica_reports, strict=True)
        },
        "postfit_restore_gate": {
            str(seed): report["postfit_restore_gate"] for seed, report in zip(WORKER_TASK_SEEDS, replica_reports, strict=True)
        },
        "ensemble_checkpoint_gate": {
            "fresh_restore": content_digest(restored_ensemble.checkpoint())
            == content_digest(ensemble_checkpoint),
            "artifact_fresh_restore": content_digest(restored_from_artifact.checkpoint())
            == content_digest(ensemble_checkpoint),
            "owner_graph_digest": payload["owner_graph_digest"],
            "parameter_count": ensemble.parameter_count,
        },
        "k3_checkpoint_gate": {
            "fresh_restore": restored_projector.checkpoint() == projector_checkpoint,
            "source_digest": k3_source_digest,
        },
        "source_observation_overlap": source_manifest["source_observation_overlap"],
        "training_performed": True,
        "default_runtime_attached": False,
        "cuda_used": False,
        "provider_attached": False,
        "mcp_attached": False,
        "client_attached": False,
        "can_start_r6_formal": False,
        "can_promote": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--model-seed", type=int, default=17)
    parser.add_argument("--candidate-namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    artifact_dir = args.artifact_dir if args.artifact_dir.is_absolute() else PROJECT_ROOT / args.artifact_dir
    report_path = args.report if args.report.is_absolute() else PROJECT_ROOT / args.report
    report = run_build(
        artifact_dir=artifact_dir,
        model_seed=args.model_seed,
        candidate_namespace=args.candidate_namespace,
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
                "artifact": str(artifact_dir / "taiji_r6_k_fixed_large_ensemble.pt"),
                "can_start_r6_formal": report["can_start_r6_formal"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
