"""Materialize an unread, content-addressed C-entry sealed-test split.

The artifact records observations and input digests, not model scores or target
labels.  Its independent path namespace prevents the validation fixture from
being silently reused as the sealed test.
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

REPORT_FORMAT = "taiji-m4v2-b3-k-c-sealed-test-v1"
VERSION = 1
SEALED_TASK_SEED = 19
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m4v2_b3_k_c_sealed_test_v1.json"
)

SEALED_EPISODE_PATHS = (
    ("sealed_typescript_00.ts", "sealed_python_00.py", "sealed_rust_00.rs"),
    ("sealed_rust_01.rs", "sealed_typescript_01.ts", "sealed_python_01.py"),
    ("sealed_python_02.py", "sealed_rust_02.rs", "sealed_typescript_02.ts"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_sealed_files(root: Path) -> None:
    for index, paths in enumerate(SEALED_EPISODE_PATHS):
        for path in paths:
            stem, extension = path.rsplit(".", 1)
            language = stem.removeprefix("sealed_").rsplit("_", 1)[0]
            value = SEALED_TASK_SEED + index * 11 + len(stem)
            pad = "s" * (2 * index + len(stem))
            if language == "python":
                body = (
                    f"def sealed_answer_{index}(value: int) -> int:\n"
                    f"    return value + {value}\n"
                    f"# taiji c-entry sealed {pad}\n"
                )
            elif language == "typescript":
                body = (
                    f"interface SealedAnswer{index} {{ value: number; }}\n"
                    f"export const sealedAnswer{index} = "
                    f"(input: SealedAnswer{index}): number => input.value + {value};\n"
                    f"// taiji c-entry sealed {pad}\n"
                )
            elif language == "rust":
                body = (
                    "fn main() {\n"
                    f'    println!("taiji-c-entry-sealed-{index}-{value}");\n'
                    "}\n"
                    f"// taiji c-entry sealed {pad}\n"
                )
            else:
                raise ValueError(f"unsupported sealed language in {path}")
            (root / path).write_text(body, encoding="utf-8")


def _materialize(output: Path) -> dict[str, object]:
    temp_parent = PROJECT_ROOT / ".tmp-m4v2-b3-k-c-sealed-test"
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
                split="c-entry-sealed",
                paths=all_sealed_paths,
                schema=schema,
            )
        }
        _build_workspace(baseline_root, task_seed=0)
        baseline_observations = _observe_all(
            baseline_root,
            registry=registry,
            split="c-entry-baseline",
            paths=_all_course_paths(),
            schema=schema,
        )
        baseline_file_digests = {
            observation.file_digest for observation in baseline_observations
        }
        sealed_file_digests = {
            observation.file_digest for observation in sealed_observations.values()
        }
        if sealed_file_digests & baseline_file_digests:
            raise ValueError("sealed-test file digest overlaps train/validation fixture")
        if len(sealed_observations) != 9:
            raise ValueError("sealed-test must contain nine unique step observations")

        anchor = _observe_all(
            temp_root,
            registry=registry,
            split="c-entry-sealed-anchor",
            paths=["missing_00.txt"],
            schema=schema,
        )[0]
        episodes = []
        for episode_index, paths in enumerate(SEALED_EPISODE_PATHS):
            sequence = _episode(
                anchor,
                sealed_observations,
                paths,
            )
            transition = _transition_examples(
                sequence,
                split=f"c-entry-sealed-{episode_index}",
            )
            episodes.append(
                {
                    "episode_id": f"sealed-{episode_index}",
                    "paths": list(paths),
                    "observation_digests": [
                        sealed_observations[path].observation_digest for path in paths
                    ],
                    "file_digests": [
                        sealed_observations[path].file_digest for path in paths
                    ],
                    "semantic_input_digests": [
                        semantic_input_digest(
                            observation.to_percept_event(tick=tick)
                        )
                        for tick, observation in enumerate(sequence[1:], start=1)
                    ],
                    "transition_input_digests": [
                        example.input_digest for example in transition
                    ],
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
            "record_disjoint_from_existing_fixture": True,
            "raw_source_content_embedded": False,
            "scores_or_targets_embedded": False,
            "anchor_payload": anchor.to_payload(),
            "episodes": episodes,
            "observation_payloads": [
                sealed_observations[path].to_payload()
                for path in all_sealed_paths
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
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    artifact = _materialize(output)
    print(json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
