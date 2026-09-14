"""Materialize the v3 C-entry sealed-test split (fresh, never-read).

v3 preregistration: ``plans/reference/M4V2_B3_K_C_PARITY_V3_PREREGISTRATION
_20260910.md``.  The v1 sealed artifact was read by the v1 formal, so the
readout-revised v3 formal must score a brand-new sealed split.  Same
contract as v1 (observation/episode payloads embedded, no scores or
targets), fresh task seed 47, and record-disjointness enforced against
the train/validation fixture AND the v1 sealed digests.
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

REPORT_FORMAT = "taiji-m4v2-b3-k-c-sealed-test-v2"
VERSION = 1
SEALED_TASK_SEED = 47
V1_SEALED_ARTIFACT = PROJECT_ROOT / "plans" / "manifests" / "taiji_m4v2_b3_k_c_sealed_test_v1.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "plans" / "manifests" / "taiji_m4v2_b3_k_c_sealed_test_v2.json"

SEALED_EPISODE_PATHS = (
    ("sealed2_rust_00.rs", "sealed2_typescript_00.ts", "sealed2_python_00.py"),
    ("sealed2_python_01.py", "sealed2_rust_01.rs", "sealed2_typescript_01.ts"),
    ("sealed2_typescript_02.ts", "sealed2_python_02.py", "sealed2_rust_02.rs"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_sealed_files(root: Path) -> None:
    for index, paths in enumerate(SEALED_EPISODE_PATHS):
        for path in paths:
            stem, extension = path.rsplit(".", 1)
            language = stem.removeprefix("sealed2_").rsplit("_", 1)[0]
            value = SEALED_TASK_SEED + index * 13 + len(stem)
            pad = "s" * (2 * index + len(stem))
            if language == "python":
                body = (
                    f"def sealed2_answer_{index}(value: int) -> int:\n"
                    f"    return value + {value}\n"
                    f"# taiji c-entry sealed v2 {pad}\n"
                )
            elif language == "typescript":
                body = (
                    f"interface Sealed2Answer{index} {{ value: number; }}\n"
                    f"export const sealed2Answer{index} = "
                    f"(input: Sealed2Answer{index}): number => input.value + {value};\n"
                    f"// taiji c-entry sealed v2 {pad}\n"
                )
            elif language == "rust":
                body = (
                    "fn main() {\n"
                    f'    println!("taiji-c-entry-sealed2-{index}-{value}");\n'
                    "}\n"
                    f"// taiji c-entry sealed v2 {pad}\n"
                )
            else:
                raise ValueError(f"unsupported sealed language in {path}")
            (root / path).write_text(body, encoding="utf-8")


def _materialize(output: Path) -> dict[str, object]:
    v1_digests: set[str] = set()
    if V1_SEALED_ARTIFACT.exists():
        v1 = json.loads(V1_SEALED_ARTIFACT.read_text(encoding="utf-8"))
        for payload in v1.get("observation_payloads", []):
            v1_digests.add(str(payload.get("file_digest")))
    temp_parent = PROJECT_ROOT / ".tmp-m4v2-b3-k-c-sealed-test-v2"
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
                split="c-entry-sealed-v2",
                paths=all_sealed_paths,
                schema=schema,
            )
        }
        _build_workspace(baseline_root, task_seed=0)
        baseline_observations = _observe_all(
            baseline_root,
            registry=registry,
            split="c-entry-baseline-v2",
            paths=_all_course_paths(),
            schema=schema,
        )
        baseline_file_digests = {observation.file_digest for observation in baseline_observations}
        sealed_file_digests = {
            observation.file_digest for observation in sealed_observations.values()
        }
        if sealed_file_digests & baseline_file_digests:
            raise ValueError("sealed-test v2 file digest overlaps train/validation fixture")
        if sealed_file_digests & v1_digests:
            raise ValueError("sealed-test v2 file digest overlaps the v1 sealed artifact")
        if len(sealed_observations) != 9:
            raise ValueError("sealed-test v2 must contain nine unique step observations")

        anchor = _observe_all(
            temp_root,
            registry=registry,
            split="c-entry-sealed-v2-anchor",
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
                split=f"c-entry-sealed-v2-{episode_index}",
            )
            episodes.append(
                {
                    "episode_id": f"sealed-v2-{episode_index}",
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
            "record_disjoint_from_existing_fixture": True,
            "record_disjoint_from_v1_sealed": True,
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
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    artifact = _materialize(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "artifact_digest": artifact["artifact_digest"],
                "task_seed": artifact["task_seed"],
                "episode_count": artifact["episode_count"],
                "status": artifact["status"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
