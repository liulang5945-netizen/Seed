"""Build the repaired M5.K1 data contract for the P1 gate.

The v4 parity course deliberately varied file bytes while keeping the typed
K1/K2-visible state unchanged.  This builder keeps the inherited observation
schema and checkpoint vocabulary, but creates real state variation in fields
that the typed masks expose:

* A/B/C alternate between resolved and ambiguous language evidence;
* D alternates ambiguous and resolved header evidence, while R alternates an
  explicit recovery-language hint and no hint;
* validation uses a separate project and unseen file/template paths.

It is a data builder only.  It does not fit a learner or read sealed payloads.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from scripts.training.eval_taiji_m4v2_b3_k_single_step import _build_experience
from scripts.training.eval_taiji_m5_k1_skill_composition import (
    _build_workspace,
    _observe_all,
    _registry,
    _schema,
)
from seed_platform.programming_languages import ProgrammingLanguageRegistry
from taiji import content_digest

ANCHOR_PATH = "missing_00.txt"
HEADER_PATHS = ("shared_header.h", "shared_header_alt.h", "validation_header.h")
CLASS_ORDER = ("A", "B", "C", "D", "R")
TRAIN_VARIANTS: dict[str, tuple[tuple[str, ...], ...]] = {
    "A": (
        ("python_00.py", "rust_00.rs", "python_01.py"),
        ("python_01.py", "rust_01.rs", "python_02.py"),
    ),
    "B": (
        ("rust_00.rs", "python_00.py", "typescript_00.ts"),
        ("rust_01.rs", "python_01.py", "typescript_01.ts"),
    ),
    "C": (
        ("typescript_00.ts", "python_00.py", "rust_00.rs"),
        ("typescript_01.ts", "python_01.py", "rust_01.rs"),
    ),
    "D": (
        ("shared_header.h", "python_00.py", "rust_00.rs"),
        ("shared_header_alt.h", "python_01.py", "rust_01.rs"),
    ),
    "R": (
        ("missing_01.txt", "python_00.py", "rust_00.rs"),
        ("missing_02.txt", "python_01.py", "rust_01.rs"),
    ),
}
VALIDATION_VARIANTS: dict[str, tuple[str, ...]] = {
    "A": ("python_04.py", "rust_04.rs", "python_05.py"),
    "B": ("rust_04.rs", "python_04.py", "typescript_04.ts"),
    "C": ("typescript_04.ts", "python_04.py", "rust_04.rs"),
    "D": ("validation_header.h", "python_04.py", "rust_04.rs"),
    "R": ("missing_validation.txt", "python_04.py", "rust_04.rs"),
}


def _class_block(course_seed: int) -> tuple[str, ...]:
    blocks = (
        ("A", "B", "C", "D", "R"),
        ("A", "C", "B", "R", "D"),
        ("B", "A", "D", "C", "R"),
    )
    return blocks[int(course_seed) % len(blocks)]


def _state_profile(class_key: str, state_index: int) -> str:
    if class_key in {"A", "B", "C"}:
        return "ambiguous-language" if int(state_index) else "resolved-language"
    if class_key == "D":
        return "resolved-header" if int(state_index) else "ambiguous-header"
    return "recovery-selection-hint" if int(state_index) else "recovery-no-selection"


def _registry_for_state(first_path: str, state_profile: str) -> ProgrammingLanguageRegistry:
    registry = _registry(typescript_available=False)
    if not first_path.endswith(".h") or state_profile != "resolved-header":
        return registry
    definitions = []
    for definition in registry.definitions:
        if definition.language_id == "cpp":
            definition = replace(
                definition,
                extensions=tuple(
                    extension for extension in definition.extensions if extension != ".h"
                ),
            )
        definitions.append(definition)
    return ProgrammingLanguageRegistry(definitions)


def _prepare_workspace(
    root: Path,
    *,
    task_seed: int,
    first_path: str,
    state_profile: str,
) -> None:
    _build_workspace(root, task_seed=task_seed)
    (root / "shared_header.h").write_text("#pragma once\nint shared_value;\n", encoding="utf-8")
    (root / "shared_header_alt.h").write_text(
        "#pragma once\nlong shared_value_alt;\n", encoding="utf-8"
    )
    (root / "validation_header.h").write_text(
        "#pragma once\nint validation_value;\n", encoding="utf-8"
    )
    if state_profile == "resolved-header" and first_path.endswith(".h"):
        (root / first_path).write_text(
            "#include <stdio.h>\nint resolved_header_value;\n", encoding="utf-8"
        )
    if state_profile == "ambiguous-language":
        for filename in ("pyproject.toml", "Cargo.toml", "tsconfig.json", "package.json"):
            (root / filename).unlink(missing_ok=True)
        if not first_path.startswith("missing") and first_path.endswith((".py", ".rs", ".ts")):
            (root / first_path).write_text("\n", encoding="utf-8")


def _observations(
    root: Path,
    *,
    paths: tuple[str, ...],
    split: str,
    project_id: str,
    state_profile: str,
    schema: Any,
) -> tuple[Any, ...]:
    observations = _observe_all(
        root,
        registry=_registry_for_state(paths[0], state_profile),
        split=split,
        paths=[ANCHOR_PATH, *paths],
        schema=schema,
    )
    observations = tuple(
        replace(
            observation,
            project_id=project_id,
            task_id=f"{project_id}:{split}:{observation.path}",
        )
        for observation in observations
    )
    if state_profile == "recovery-selection-hint":
        observations = tuple(
            replace(
                observation,
                selection_state=(
                    "ambiguous" if observation.path == paths[0] else observation.selection_state
                ),
            )
            for observation in observations
        )
    return observations


def _template_id(
    *,
    split: str,
    class_key: str,
    variant_paths: tuple[str, ...],
    state_profile: str,
) -> str:
    return content_digest(
        {
            "format": "taiji-m5-k-p1-template-v1",
            "split": split,
            "class": class_key,
            "variant_paths": list(variant_paths),
            "state_profile": state_profile,
        }
    )


def _build_one(
    *,
    scratch: Path,
    index: int,
    class_key: str,
    variant_paths: tuple[str, ...],
    state_index: int,
    split: str,
    project_id: str,
    task_seed: int,
    parent_digest: str,
    bundle: Any,
    projector: Any,
    source_manifest_digest: str,
    experience_split: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    state_profile = _state_profile(class_key, state_index)
    root = scratch / f"{split}-{index:04d}"
    root.mkdir(parents=True, exist_ok=False)
    _prepare_workspace(
        root,
        task_seed=task_seed,
        first_path=variant_paths[0],
        state_profile=state_profile,
    )
    observations = _observations(
        root,
        paths=variant_paths,
        split=f"p1-{split}-{index:04d}",
        project_id=project_id,
        state_profile=state_profile,
        schema=_schema(),
    )
    experience = _build_experience(
        sequence=observations,
        split=experience_split or split,
        name=f"p1-{split}-{index:04d}",
        parent_digest=parent_digest,
        worker_bundle_digest=bundle.bundle_digest,
        source_manifest_digest=source_manifest_digest,
        projector=projector,
    )
    metadata = {
        "index": int(index),
        "class_key": class_key,
        "split": split,
        "project_id": project_id,
        "task_kind": "inspect-language",
        "anchor_path": ANCHOR_PATH,
        "variant_paths": list(variant_paths),
        "first_variant_path": variant_paths[0],
        "state_profile": state_profile,
        "state_index": int(state_index),
        "template_family_id": _template_id(
            split=split,
            class_key=class_key,
            variant_paths=variant_paths,
            state_profile=state_profile,
        ),
        "source_manifest_digest": source_manifest_digest,
    }
    return experience, metadata


def build_train_course(
    *,
    scratch: Path,
    course_seed: int,
    parent_digest: str,
    bundle: Any,
    projector: Any,
    source_manifest_digest: str,
    count: int = 150,
) -> tuple[list[Any], list[dict[str, Any]], dict[str, int], tuple[str, ...]]:
    scratch.mkdir(parents=True, exist_ok=True)
    class_block = _class_block(course_seed)
    experiences: list[Any] = []
    metadata: list[dict[str, Any]] = []
    for index in range(int(count)):
        class_key = class_block[index % len(class_block)]
        variants = TRAIN_VARIANTS[class_key]
        variant_paths = variants[(index // len(class_block)) % len(variants)]
        state_index = (index + int(course_seed)) % 2
        experience, item = _build_one(
            scratch=scratch,
            index=index,
            class_key=class_key,
            variant_paths=variant_paths,
            state_index=state_index,
            split="train",
            project_id=f"p1-train-project-{state_index}",
            task_seed=1000 + int(course_seed) * 100 + index,
            parent_digest=parent_digest,
            bundle=bundle,
            projector=projector,
            source_manifest_digest=source_manifest_digest,
        )
        item["course_seed"] = int(course_seed)
        experiences.append(experience)
        metadata.append(item)
    class_counts = {key: sum(item["class_key"] == key for item in metadata) for key in CLASS_ORDER}
    return experiences, metadata, class_counts, class_block


def build_validation_course(
    *,
    scratch: Path,
    parent_digest: str,
    bundle: Any,
    projector: Any,
    source_manifest_digest: str,
) -> tuple[list[Any], list[dict[str, Any]]]:
    scratch.mkdir(parents=True, exist_ok=True)
    experiences: list[Any] = []
    metadata: list[dict[str, Any]] = []
    for state_index in (0, 1):
        for offset, class_key in enumerate(CLASS_ORDER):
            index = state_index * len(CLASS_ORDER) + offset
            experience, item = _build_one(
                scratch=scratch,
                index=index,
                class_key=class_key,
                variant_paths=VALIDATION_VARIANTS[class_key],
                state_index=state_index,
                split="validation",
                experience_split="holdout",
                project_id="p1-validation-project",
                task_seed=2000 + index,
                parent_digest=parent_digest,
                bundle=bundle,
                projector=projector,
                source_manifest_digest=source_manifest_digest,
            )
            experiences.append(experience)
            metadata.append(item)
    return experiences, metadata


__all__ = [
    "ANCHOR_PATH",
    "CLASS_ORDER",
    "TRAIN_VARIANTS",
    "VALIDATION_VARIANTS",
    "build_train_course",
    "build_validation_course",
]
