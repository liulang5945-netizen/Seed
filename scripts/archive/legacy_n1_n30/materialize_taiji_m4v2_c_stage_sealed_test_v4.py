"""Materialize the C-stage sealed-test split (fresh, never-read).

C-stage preregistration: ``plans/reference/M4V2_C_STAGE_FORMAL
_PREREGISTRATION_20260910.md``.  The weak-class primary criterion needs
D and R representation in the sealed split, so the three episodes' first
files cover D (``.h`` header), R (missing file), and A (python).  Task
seed 101; record-disjointness enforced against the train/validation
fixture and sealed v1/v2/v3 (empty digests from missing-file
observations excluded from the overlap check).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m4v2_b3_k_single_step import (  # noqa: E402
    _all_course_paths,
)
from scripts.training.eval_taiji_m5_k2_multistep_composition import (  # noqa: E402
    _build_workspace,
    _episode,
    _observe_all,
    _registry,
    _schema,
    _transition_examples,
)
from taiji import content_digest, semantic_input_digest  # noqa: E402

REPORT_FORMAT = "taiji-m4v2-c-stage-sealed-test-v4"
VERSION = 1
SEALED_TASK_SEED = 101
PRIOR_SEALED_ARTIFACTS = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m4v2_b3_k_c_sealed_test_v1.json",
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m4v2_b3_k_c_sealed_test_v2.json",
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_v4_sealed_test_v3.json",
)
DEFAULT_OUTPUT = PROJECT_ROOT / "plans" / "manifests" / "taiji_m4v2_c_stage_sealed_test_v4.json"

SEALED_EPISODE_PATHS = (
    ("sealed4_header_00.h", "sealed4_python_00.py", "sealed4_rust_00.rs"),
    ("sealed4_missing_00.txt", "sealed4_typescript_00.ts", "sealed4_python_01.py"),
    ("sealed4_python_02.py", "sealed4_rust_01.rs", "sealed4_typescript_01.ts"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_sealed_files(root: Path) -> None:
    for index, paths in enumerate(SEALED_EPISODE_PATHS):
        for path in paths:
            if path.startswith("sealed4_missing"):
                continue
            stem, extension = path.rsplit(".", 1)
            language = stem.removeprefix("sealed4_").rsplit("_", 1)[0]
            value = SEALED_TASK_SEED + index * 19 + len(stem)
            pad = "s" * (2 * index + len(stem))
            if language == "python":
                body = (
                    f"def sealed4_answer_{index}(value: int) -> int:\n"
                    f"    return value + {value}\n"
                    f"# taiji c-stage sealed {pad}\n"
                )
            elif language == "typescript":
                body = (
                    f"interface Sealed4Answer{index} {{ value: number; }}\n"
                    f"export const sealed4Answer{index} = "
                    f"(input: Sealed4Answer{index}): number => input.value + {value};\n"
                    f"// taiji c-stage sealed {pad}\n"
                )
            elif language == "rust":
                body = (
                    "fn main() {\n"
                    f'    println!("taiji-c-stage-sealed4-{index}-{value}");\n'
                    "}\n"
                    f"// taiji c-stage sealed {pad}\n"
                )
            elif language == "header":
                body = (
                    f"#pragma once\nint sealed4_shared_{index};\n"
                    f"// taiji c-stage sealed {pad}\n"
                )
            else:
                raise ValueError(f"unsupported sealed language in {path}")
            (root / path).write_text(body, encoding="utf-8")


def _materialize(output: Path) -> dict[str, object]:
    prior_digests: set[str] = set()
    for prior in PRIOR_SEALED_ARTIFACTS:
        if prior.exists():
            artifact = json.loads(prior.read_text(encoding="utf-8"))
            for payload in artifact.get("observation_payloads", []):
                digest = str(payload.get("file_digest", ""))
                if digest:
                    prior_digests.add(digest)
    temp_parent = PROJECT_ROOT / ".tmp-m4v2-c-stage-sealed-test"
    temp_parent.mkdir(parents=True, exist_ok=True)
    temp_root = temp_parent / uuid4().hex
    temp_root.mkdir()
    baseline_root = temp_parent / f"baseline-{uuid4().hex}"
    baseline_root.mkdir()
    try:
        _build_workspace(temp_root, task_seed=SEALED_TASK_SEED)
        _write_sealed_files(temp_root)
        schema = _schema()
        registry = _registry(typescript_available=True)
        all_sealed_paths = [path for episode in SEALED_EPISODE_PATHS for path in episode]
        sealed_observations = {
            observation.path: observation
            for observation in _observe_all(
                temp_root,
                registry=registry,
                split="c-stage-sealed",
                paths=all_sealed_paths,
                schema=schema,
            )
        }
        _build_workspace(baseline_root, task_seed=0)
        baseline_observations = _observe_all(
            baseline_root,
            registry=registry,
            split="c-stage-baseline",
            paths=_all_course_paths(),
            schema=schema,
        )
        baseline_file_digests = {
            observation.file_digest
            for observation in baseline_observations
            if observation.file_digest
        }
        sealed_file_digests = {
            observation.file_digest
            for observation in sealed_observations.values()
            if observation.file_digest
        }
        if sealed_file_digests & baseline_file_digests:
            raise ValueError("sealed v4 file digest overlaps train/validation fixture")
        if sealed_file_digests & prior_digests:
            raise ValueError("sealed v4 file digest overlaps a prior sealed artifact")
        if len(sealed_observations) != 9:
            raise ValueError("sealed v4 must contain nine unique step observations")

        anchor = _observe_all(
            temp_root,
            registry=registry,
            split="c-stage-sealed-anchor",
            paths=["missing_00.txt"],
            schema=schema,
        )[0]
        episodes = []
        for episode_index, paths in enumerate(SEALED_EPISODE_PATHS):
            sequence = _episode(anchor, sealed_observations, paths)
            transition = _transition_examples(
                sequence,
                split=f"c-stage-sealed-{episode_index}",
            )
            episodes.append(
                {
                    "episode_id": f"sealed-v4-{episode_index}",
                    "episode_class": {0: "D", 1: "R", 2: "A"}[episode_index],
                    "paths": list(paths),
                    "observation_digests": [
                        sealed_observations[path].observation_digest for path in paths
                    ],
                    "file_digests": [sealed_observations[path].file_digest for path in paths],
                    "semantic_input_digests": [
                        semantic_input_digest(observation.to_percept_event(tick=tick))
                        for tick, observation in enumerate(sequence[1:], start=1)
                    ],
                    "transition_input_digests": [example.input_digest for example in transition],
                }
            )

        unsigned = {
            "format": REPORT_FORMAT,
            "version": VERSION,
            "status": "materialized-unread",
            "task_seed": SEALED_TASK_SEED,
            "schema": schema.to_payload(),
            "episode_count": len(episodes),
            "step_count": len(sealed_observations),
            "episode_classes": ["D", "R", "A"],
            "record_disjoint_from_existing_fixture": True,
            "record_disjoint_from_prior_sealed": True,
            "raw_source_content_embedded": False,
            "scores_or_targets_embedded": False,
            "anchor_payload": anchor.to_payload(),
            "episodes": episodes,
            "observation_payloads": [
                sealed_observations[path].to_payload() for path in all_sealed_paths
            ],
        }
        artifact = {
            **unsigned,
            "artifact_digest": content_digest(unsigned),
            "materializer_sha256": _sha256(Path(__file__)),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return artifact
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
        shutil.rmtree(baseline_root, ignore_errors=True)
        if temp_parent.exists() and not any(temp_parent.iterdir()):
            temp_parent.rmdir()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    artifact = _materialize(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "artifact_digest": artifact["artifact_digest"],
                "task_seed": artifact["task_seed"],
                "episode_classes": artifact["episode_classes"],
                "status": artifact["status"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
