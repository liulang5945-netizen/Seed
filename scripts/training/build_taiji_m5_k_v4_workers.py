"""Rebuild the parent K1/K2 workers with the expanded 5-class vocabulary.

Preregistration: ``plans/reference/M5_K_SIGNAL_SPACE_EXPANSION
_PREREGISTRATION_20260910.md``.  The vocabulary grows naturally from the
training experiences: the frozen M5.K2 course episodes plus two new
episode families -- D (a ``.h`` header whose c/cpp extension tie resolves
ambiguous) and R (a missing first file, recover-target).  This grows the
fact vocabulary 13 -> 14 (``language_state::ambiguous``), the K1 goal
vocabulary 3 -> 4 (``clarify-language``), and the K2 goal/content
vocabulary 2 -> 4.  K3 is rebuilt as a fresh deterministic projector
(no learned state; behaviorally identical, new scope namespace -- the
prereg's "K3 unchanged" refers to the mechanism).  Artifacts land in
``checkpoints/taiji_k_workers_v4/model_{seed}/`` without touching the
v1 workers.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.build_taiji_m4v2_r6_k_worker_artifacts import (  # noqa: E402
    _artifact,
    _atomic_save,
)
from scripts.training.eval_taiji_m4v2_r6_k_worker_attachment_preflight import (  # noqa: E402
    K3_SOURCE_FORMAT,
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
    _dev_episode_paths,
    _episode,
    _observe_all,
    _registry,
    _schema,
    _semantic_example,
    _test_episode_paths,
    _train_episode_paths,
    _transition_examples,
    _typed_transition_input_masks,
)
from taiji import (  # noqa: E402
    OutcomeDependencyProjector,
    StructuredSemanticCorpus,
    StructuredSemanticLearner,
    StructuredSemanticTransitionCorpus,
    StructuredSemanticTransitionLearner,
    content_digest,
)

REPORT_FORMAT = "taiji-m5-k-v4-worker-build-v1"
VERSION = 1
SOURCE_MANIFEST_FORMAT = "taiji-k-worker-source-manifest-v4"
RESOURCE_MANIFEST_FORMAT = "taiji-r6-cpu-resource-manifest-v1"
MODEL_SEEDS = (17, 23, 31)
DEFAULT_ROOT = PROJECT_ROOT / "checkpoints" / "taiji_k_workers_v4"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_v4_worker_build_20260910.json"
HEADER_PATH = "shared_header.h"
HEADER_BODY = "#pragma once\nint shared_value;\n"
SECOND_MISSING = "missing_01.txt"
D_EPISODE = (HEADER_PATH, "python_00.py", "rust_00.rs")
R_EPISODE = (SECOND_MISSING, "python_00.py", "rust_00.rs")


def _course_inputs_v4(root: Path, *, task_seed: int) -> tuple[Any, Any, Any, dict[str, Any]]:
    _build_workspace(root, task_seed=task_seed)
    (root / HEADER_PATH).write_text(HEADER_BODY, encoding="utf-8")
    schema = _schema()
    registry = _registry(typescript_available=False)
    all_paths = [anchor := "missing_00.txt", SECOND_MISSING, HEADER_PATH]
    for index in range(FILES_PER_LANG):
        all_paths.extend(f"{language}_{index:02d}{EXTENSIONS[language]}" for language in LANGS)
    observations = {
        observation.path: observation
        for observation in _observe_all(
            root,
            registry=registry,
            split="artifact-build-v4",
            paths=all_paths,
            schema=schema,
        )
    }
    anchor = observations[anchor]

    train_paths = [anchor.path, SECOND_MISSING, HEADER_PATH]
    for episode_paths in _train_episode_paths():
        train_paths.extend(episode_paths)
    train_paths = list(dict.fromkeys(train_paths))
    dev_paths = list(_dev_episode_paths())
    test_paths = list(_test_episode_paths())

    semantic_corpus = StructuredSemanticCorpus.from_splits(
        train=tuple(
            _semantic_example(observations[path], split="train", tick=index + 1)
            for index, path in enumerate(train_paths)
        ),
        dev=tuple(
            _semantic_example(observations[path], split="dev", tick=index + 1)
            for index, path in enumerate(dev_paths)
        ),
        test=tuple(
            _semantic_example(observations[path], split="test", tick=index + 1)
            for index, path in enumerate(test_paths)
        ),
    )

    train_episodes = tuple(
        _episode(anchor, observations, paths) for paths in _train_episode_paths()
    ) + (
        _episode(anchor, observations, D_EPISODE),
        _episode(anchor, observations, R_EPISODE),
    )
    dev_episode = _episode(anchor, observations, _dev_episode_paths())
    test_episode = _episode(anchor, observations, _test_episode_paths())
    transition_corpus = StructuredSemanticTransitionCorpus.from_splits(
        train=tuple(
            item
            for index, sequence in enumerate(train_episodes)
            for item in _transition_examples(sequence, split=f"train-{index}")
        ),
        dev=_transition_examples(dev_episode, split="dev-0"),
        test=_transition_examples(test_episode, split="test-0"),
    )
    return semantic_corpus, transition_corpus, schema, observations


def run_build(
    *,
    artifact_dir: Path,
    task_seed: int = 0,
    learner_seed: int = 17,
) -> dict[str, Any]:
    started = time.perf_counter()
    parent = _parent(learner_seed)
    parent_digest = content_digest(parent)
    temp_parent = PROJECT_ROOT / ".tmp-m5-k-v4-worker-build"
    temp_parent.mkdir(parents=True, exist_ok=True)
    temp_root = temp_parent / f"course-{uuid4().hex}"
    temp_root.mkdir()
    try:
        torch.manual_seed(int(learner_seed))
        semantic_corpus, transition_corpus, schema, observations = _course_inputs_v4(
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
        parameter_ledger = {
            "k1.semantic": int(semantic.parameter_count),
            "k2.transition": int(transition.parameter_count),
            "k1_fact_keys": len(semantic_corpus.fact_keys),
            "k1_goal_ids": list(semantic_corpus.goal_ids),
            "k2_goal_ids": list(transition_corpus.goal_ids),
        }
        if "workbench::language_state::ambiguous" not in semantic_corpus.fact_keys:
            raise ValueError("v4 rebuild failed to grow the ambiguous fact into the vocabulary")
        if "goal:clarify-language" not in semantic_corpus.goal_ids:
            raise ValueError("v4 rebuild failed to grow clarify-language into the K1 vocabulary")
        if "goal:recover-target" not in transition_corpus.goal_ids:
            raise ValueError("v4 rebuild failed to grow recover-target into the K2 vocabulary")

        semantic_losses = semantic.fit(
            semantic_corpus.train, epochs=SEMANTIC_EPOCHS, learning_rate=SEMANTIC_LR
        )
        transition_losses = transition.fit(
            transition_corpus.train, epochs=TRAIN_EPOCHS, learning_rate=TRAIN_LR
        )
        semantic_checkpoint = semantic.checkpoint()
        transition_checkpoint = transition.checkpoint()
        semantic_restored = StructuredSemanticLearner.from_checkpoint(
            semantic_checkpoint, semantic_corpus
        )
        transition_restored = StructuredSemanticTransitionLearner.from_checkpoint(
            transition_checkpoint, transition_corpus
        )
        fresh_restore_gate = {
            "k1.semantic": semantic.owner_digests() == semantic_restored.owner_digests(),
            "k2.transition": transition.owner_digests()
            == transition_restored.owner_digests(),
        }
        if not all(fresh_restore_gate.values()):
            raise RuntimeError("v4 worker fresh restore gate failed")

        # K3 is a deterministic projector with no learned state; a fresh
        # instance is behaviorally identical to the v1 artifact up to the
        # scope namespace (recorded as the prereg's "K3 unchanged").
        projector = OutcomeDependencyProjector(
            f"r6-k3-worker-v4-{int(task_seed)}-{int(learner_seed)}"
        )
        projection_checkpoint = projector.checkpoint()
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
            "version": 4,
            "parent_checkpoint_digest": parent_digest,
            "course": "m5-k2-multistep-composition + D/R signal expansion",
            "task_seed": int(task_seed),
            "learner_seed": int(learner_seed),
            "k1_source_digest": semantic.source_digest,
            "k2_source_digest": transition.source_digest,
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
        resource_manifest_digest = content_digest(
            {
                "format": RESOURCE_MANIFEST_FORMAT,
                "version": 1,
                "device": "cpu",
                "cuda_required": False,
                "optimizer_state_present": False,
                "training_backend": "native-delta",
            }
        )
        artifacts = {
            "k1.semantic": _artifact(
                worker_id="k1.semantic",
                checkpoint=semantic_checkpoint,
                owner_digests=semantic.owner_digests(),
                source_digest=semantic.source_digest,
                parent_digest=parent_digest,
                source_manifest_digest=source_manifest_digest,
                resource_manifest_digest=resource_manifest_digest,
                candidate_namespace="taiji:k:candidate",
            ),
            "k2.transition": _artifact(
                worker_id="k2.transition",
                checkpoint=transition_checkpoint,
                owner_digests=transition.owner_digests(),
                source_digest=transition.source_digest,
                parent_digest=parent_digest,
                source_manifest_digest=source_manifest_digest,
                resource_manifest_digest=resource_manifest_digest,
                candidate_namespace="taiji:k:candidate",
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
                candidate_namespace="taiji:k:candidate",
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
        return {
            "learner_seed": learner_seed,
            "status": "passed",
            "parameter_ledger": parameter_ledger,
            "worker_training_steps": {
                "k1.semantic": int(semantic.training_steps),
                "k2.transition": int(transition.training_steps),
            },
            "losses": {
                "k1.semantic": semantic_losses,
                "k2.transition": transition_losses,
            },
            "fresh_restore_gate": fresh_restore_gate,
            "artifact_digests": {
                worker_id: str(payload["artifact_digest"])
                for worker_id, payload in saved.items()
            },
            "artifact_paths": {worker_id: str(path) for worker_id, path in paths.items()},
            "training_performed": True,
            "default_runtime_attached": False,
            "elapsed_seconds": time.perf_counter() - started,
        }
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    reports = []
    for learner_seed in MODEL_SEEDS:
        report = run_build(artifact_dir=args.root / f"model_{learner_seed}", learner_seed=learner_seed)
        reports.append(report)
        print(
            json.dumps(
                {
                    "learner_seed": learner_seed,
                    "parameters": report["parameter_ledger"],
                    "steps": report["worker_training_steps"],
                    "status": report["status"],
                }
            ),
            flush=True,
        )
    all_passed = all(report["status"] == "passed" for report in reports)
    payload = {
        "report_format": REPORT_FORMAT,
        "version": VERSION,
        "status": "passed" if all_passed else "failed",
        "can_promote": False,
        "preregistration": (
            "plans/reference/M5_K_SIGNAL_SPACE_EXPANSION_PREREGISTRATION_20260910.md"
        ),
        "worker_root": str(args.root),
        "reports": reports,
        "boundary": (
            "v4 parent worker rebuild only; no sealed read, no parity build, "
            "no default runtime/provider/MCP/client/CUDA."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"report": str(args.report), "status": payload["status"]}, indent=2))
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
