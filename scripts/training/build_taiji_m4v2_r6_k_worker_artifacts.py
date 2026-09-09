"""Build checkpointable K1/K2 workers and a deterministic K3 artifact.

This is the first artifact-producing step after the R6 attachment contract.
It reuses the frozen M5.K2 course construction, performs a real CPU fit for
K1/K2, performs a save/fresh-restore check *before* fitting, then writes three
content-addressed worker envelopes tied to one R6 parent/source/resource
manifest.  It does not attach a default runtime or promote any worker.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m4v2_r6_k_worker_attachment_preflight import (  # noqa: E402
    ARTIFACT_FORMAT,
    ARTIFACT_VERSION,
    K3_SOURCE_FORMAT,
)
from scripts.training.eval_taiji_m4v2_r6_parent_baseline_preflight import (  # noqa: E402
    _parent,
)
from scripts.training.eval_taiji_m5_k1_skill_composition import (  # noqa: E402
    _typed_fact_feature_masks,
)
from scripts.training.eval_taiji_m5_k2_multistep_composition import (  # noqa: E402
    EXTENSIONS,
    FILES_PER_LANG,
    LANGS,
    SEMANTIC_EPOCHS,
    SEMANTIC_LR,
    TRAIN_EPOCHS,
    TRAIN_LR,
    _build_workspace,
    _observe_all,
    _registry,
    _schema,
    _semantic_and_transition_corpora,
    _typed_transition_input_masks,
)
from taiji import (  # noqa: E402
    K_WORKER_INPUT_CONTRACT_DIGESTS,
    K_WORKER_OUTPUT_CONTRACT_DIGESTS,
    OutcomeDependencyProjector,
    StructuredSemanticLearner,
    StructuredSemanticTransitionLearner,
    content_digest,
)

REPORT_FORMAT = "taiji-m4v2-r6-k-worker-artifact-build-v1"
VERSION = 1
SOURCE_MANIFEST_FORMAT = "taiji-k-worker-source-manifest-v1"
RESOURCE_MANIFEST_FORMAT = "taiji-r6-cpu-resource-manifest-v1"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "checkpoints" / "taiji_k_workers"
DEFAULT_REPORT = (
    PROJECT_ROOT / "reports" / "taiji_m4v2_r6_k_worker_artifact_build_20260909.json"
)
DEFAULT_CANDIDATE_NAMESPACE = "taiji:k:candidate"


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        raw = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(path, map_location="cpu")
    if not isinstance(raw, Mapping):
        raise TypeError("saved checkpoint is not a mapping")
    return {str(key): value for key, value in raw.items()}


def _save_roundtrip(
    *,
    path: Path,
    checkpoint: Mapping[str, Any],
    restore: Callable[[Mapping[str, Any]], Any],
    digest: Callable[[Any], str],
) -> dict[str, Any]:
    torch.save(dict(checkpoint), path)
    loaded = _load_mapping(path)
    restored = restore(loaded)
    return {
        "path": str(path),
        "saved_digest": content_digest(dict(checkpoint)),
        "loaded_digest": content_digest(loaded),
        "restored_digest": digest(restored),
        "roundtrip": digest(restored) == content_digest(dict(checkpoint)),
    }


def _atomic_save(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    loaded = _load_mapping(temporary)
    if content_digest(loaded) != content_digest(dict(payload)):
        raise ValueError("saved worker artifact changed during disk roundtrip")
    temporary.replace(path)
    return loaded


def _artifact(
    *,
    worker_id: str,
    checkpoint: Mapping[str, Any],
    owner_digests: Mapping[str, str],
    source_digest: str,
    parent_digest: str,
    source_manifest_digest: str,
    resource_manifest_digest: str,
    candidate_namespace: str,
) -> dict[str, Any]:
    checkpoint_payload = dict(checkpoint)
    unsigned: dict[str, Any] = {
        "format": ARTIFACT_FORMAT,
        "version": ARTIFACT_VERSION,
        "worker_id": worker_id,
        "parent_checkpoint_digest": parent_digest,
        "candidate_namespace": candidate_namespace,
        "source_manifest_digest": source_manifest_digest,
        "resource_manifest_digest": resource_manifest_digest,
        "source_digest": source_digest,
        "input_contract_digest": K_WORKER_INPUT_CONTRACT_DIGESTS[worker_id],
        "output_contract_digest": K_WORKER_OUTPUT_CONTRACT_DIGESTS[worker_id],
        "checkpoint": checkpoint_payload,
        "worker_checkpoint_digest": content_digest(checkpoint_payload),
        "owner_digests": [
            [str(key), str(value)] for key, value in sorted(owner_digests.items())
        ],
        "training_steps": int(checkpoint_payload.get("training_steps", 0)),
        "optimizer_state_present": False,
    }
    return {**unsigned, "artifact_digest": content_digest(unsigned)}


def _course_inputs(root: Path, *, task_seed: int) -> tuple[Any, Any, Any]:
    _build_workspace(root, task_seed=task_seed)
    schema = _schema()
    registry = _registry(typescript_available=False)
    all_paths = ["missing_00.txt"]
    for index in range(FILES_PER_LANG):
        all_paths.extend(
            f"{language}_{index:02d}{EXTENSIONS[language]}"
            for language in LANGS
        )
    observations = {
        observation.path: observation
        for observation in _observe_all(
            root,
            registry=registry,
            split="artifact-build",
            paths=all_paths,
            schema=schema,
        )
    }
    anchor = observations["missing_00.txt"]
    semantic_corpus, transition_corpus, _, _, _ = _semantic_and_transition_corpora(
        observations, anchor, schema
    )
    return semantic_corpus, transition_corpus, schema


def run_build(
    *,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    task_seed: int = 0,
    learner_seed: int = 17,
    candidate_namespace: str = DEFAULT_CANDIDATE_NAMESPACE,
) -> dict[str, Any]:
    started = time.perf_counter()
    parent = _parent(learner_seed)
    parent_digest = content_digest(parent)
    temp_parent = PROJECT_ROOT / ".tmp-m4v2-r6-k-worker-build"
    temp_parent.mkdir(parents=True, exist_ok=True)
    temp_root = temp_parent / f"course-{uuid4().hex}"
    temp_root.mkdir()
    try:
        torch.manual_seed(int(learner_seed))
        semantic_corpus, transition_corpus, schema = _course_inputs(
            temp_root, task_seed=task_seed
        )
        fact_masks = _typed_fact_feature_masks(semantic_corpus.fact_keys, schema)
        readout_excluded = tuple(
            key for key in semantic_corpus.fact_keys if key.split("::")[1] == "language"
        )
        transition_masks = _typed_transition_input_masks(
            transition_corpus.fact_keys,
            schema,
            event_dim=transition_corpus.event_dim,
        )
        semantic = StructuredSemanticLearner(
            semantic_corpus,
            fact_feature_masks=fact_masks,
            readout_excluded_facts=readout_excluded,
        )
        transition = StructuredSemanticTransitionLearner(
            transition_corpus,
            transition_input_masks=transition_masks,
        )

        prefit_dir = temp_root / "prefit-checkpoints"
        prefit_dir.mkdir()
        prefit_semantic = _save_roundtrip(
            path=prefit_dir / "k1-semantic.pt",
            checkpoint=semantic.checkpoint(),
            restore=lambda payload: StructuredSemanticLearner.from_checkpoint(
                payload, semantic_corpus
            ),
            digest=lambda worker: content_digest(worker.checkpoint()),
        )
        prefit_transition = _save_roundtrip(
            path=prefit_dir / "k2-transition.pt",
            checkpoint=transition.checkpoint(),
            restore=lambda payload: StructuredSemanticTransitionLearner.from_checkpoint(
                payload, transition_corpus
            ),
            digest=lambda worker: content_digest(worker.checkpoint()),
        )
        if not prefit_semantic["roundtrip"] or not prefit_transition["roundtrip"]:
            raise RuntimeError("prefit worker checkpoint gate failed; training is blocked")

        semantic_losses = semantic.fit(
            semantic_corpus.train,
            epochs=SEMANTIC_EPOCHS,
            learning_rate=SEMANTIC_LR,
        )
        transition_losses = transition.fit(
            transition_corpus.train,
            epochs=TRAIN_EPOCHS,
            learning_rate=TRAIN_LR,
        )
        semantic_checkpoint = semantic.checkpoint()
        transition_checkpoint = transition.checkpoint()
        semantic_restored = StructuredSemanticLearner.from_checkpoint(
            semantic_checkpoint, semantic_corpus
        )
        transition_restored = StructuredSemanticTransitionLearner.from_checkpoint(
            transition_checkpoint, transition_corpus
        )

        projector = OutcomeDependencyProjector(
            f"r6-k3-worker-{int(task_seed)}-{int(learner_seed)}"
        )
        projection_checkpoint = projector.checkpoint()
        projection_restored = OutcomeDependencyProjector.from_checkpoint(
            projection_checkpoint
        )
        projection_checkpoint_digest = content_digest(projection_checkpoint)
        k3_source_digest = content_digest(
            {
                "format": K3_SOURCE_FORMAT,
                "scope_id": projector.scope_id,
                "lesioned": bool(projector.lesioned),
                "checkpoint_digest": projection_checkpoint_digest,
            }
        )
        source_manifest = {
            "format": SOURCE_MANIFEST_FORMAT,
            "version": 1,
            "parent_checkpoint_digest": parent_digest,
            "course": "m5-k2-multistep-composition",
            "task_seed": int(task_seed),
            "learner_seed": int(learner_seed),
            "k1_source_digest": semantic.source_digest,
            "k2_source_digest": transition.source_digest,
            "k3_source_digest": k3_source_digest,
            "semantic_training": {
                "epochs": SEMANTIC_EPOCHS,
                "learning_rate": SEMANTIC_LR,
                "fact_masks": fact_masks,
                "readout_excluded_facts": list(readout_excluded),
            },
            "transition_training": {
                "epochs": TRAIN_EPOCHS,
                "learning_rate": TRAIN_LR,
                "transition_masks": transition_masks,
            },
        }
        source_manifest_digest = content_digest(source_manifest)
        resource_manifest = {
            "format": RESOURCE_MANIFEST_FORMAT,
            "version": 1,
            "device": "cpu",
            "cuda_required": False,
            "optimizer_state_present": False,
            "training_backend": "native-delta",
        }
        resource_manifest_digest = content_digest(resource_manifest)
        artifacts = {
            "k1.semantic": _artifact(
                worker_id="k1.semantic",
                checkpoint=semantic_checkpoint,
                owner_digests=semantic.owner_digests(),
                source_digest=semantic.source_digest,
                parent_digest=parent_digest,
                source_manifest_digest=source_manifest_digest,
                resource_manifest_digest=resource_manifest_digest,
                candidate_namespace=candidate_namespace,
            ),
            "k2.transition": _artifact(
                worker_id="k2.transition",
                checkpoint=transition_checkpoint,
                owner_digests=transition.owner_digests(),
                source_digest=transition.source_digest,
                parent_digest=parent_digest,
                source_manifest_digest=source_manifest_digest,
                resource_manifest_digest=resource_manifest_digest,
                candidate_namespace=candidate_namespace,
            ),
            "k3.outcome_projection": _artifact(
                worker_id="k3.outcome_projection",
                checkpoint=projection_checkpoint,
                owner_digests={
                    "outcome_dependency_projector": projection_checkpoint_digest
                },
                source_digest=k3_source_digest,
                parent_digest=parent_digest,
                source_manifest_digest=source_manifest_digest,
                resource_manifest_digest=resource_manifest_digest,
                candidate_namespace=candidate_namespace,
            ),
        }
        paths = {
            "k1.semantic": artifact_dir / "taiji_r6_k1_semantic.pt",
            "k2.transition": artifact_dir / "taiji_r6_k2_transition.pt",
            "k3.outcome_projection": artifact_dir / "taiji_r6_k3_outcome_projection.pt",
        }
        saved = {
            worker_id: _atomic_save(paths[worker_id], artifact)
            for worker_id, artifact in artifacts.items()
        }
        report = {
            "report_format": REPORT_FORMAT,
            "version": VERSION,
            "created_at_unix": time.time(),
            "status": "passed",
            "parent_checkpoint_digest": parent_digest,
            "source_manifest_digest": source_manifest_digest,
            "resource_manifest_digest": resource_manifest_digest,
            "candidate_namespace": candidate_namespace,
            "artifact_paths": {worker_id: str(path) for worker_id, path in paths.items()},
            "artifact_digests": {
                worker_id: str(payload["artifact_digest"])
                for worker_id, payload in saved.items()
            },
            "worker_checkpoint_digests": {
                worker_id: str(payload["worker_checkpoint_digest"])
                for worker_id, payload in saved.items()
            },
            "worker_training_steps": {
                "k1.semantic": int(semantic.training_steps),
                "k2.transition": int(transition.training_steps),
                "k3.outcome_projection": 0,
            },
            "losses": {
                "k1.semantic": semantic_losses,
                "k2.transition": transition_losses,
            },
            "prefit_checkpoint_gate": {
                "k1.semantic": {
                    **prefit_semantic,
                    "temporary_path_removed_after_run": True,
                },
                "k2.transition": {
                    **prefit_transition,
                    "temporary_path_removed_after_run": True,
                },
                "k3.outcome_projection": {
                    "checkpoint_digest": projection_checkpoint_digest,
                    "fresh_restore": projection_restored.checkpoint()
                    == projection_checkpoint,
                },
            },
            "postfit_restore_gate": {
                "k1.semantic": semantic.owner_digests()
                == semantic_restored.owner_digests(),
                "k2.transition": transition.owner_digests()
                == transition_restored.owner_digests(),
                "k3.outcome_projection": projection_restored.checkpoint()
                == projection_checkpoint,
            },
            "training_performed": True,
            "default_runtime_attached": False,
            "cuda_used": False,
            "provider_attached": False,
            "mcp_attached": False,
            "client_attached": False,
            "elapsed_seconds": time.perf_counter() - started,
        }
        return report
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--task-seed", type=int, default=0)
    parser.add_argument("--learner-seed", type=int, default=17)
    parser.add_argument("--candidate-namespace", default=DEFAULT_CANDIDATE_NAMESPACE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    artifact_dir = args.artifact_dir
    if not artifact_dir.is_absolute():
        artifact_dir = PROJECT_ROOT / artifact_dir
    report_path = args.report
    if not report_path.is_absolute():
        report_path = PROJECT_ROOT / report_path
    report = run_build(
        artifact_dir=artifact_dir,
        task_seed=args.task_seed,
        learner_seed=args.learner_seed,
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
