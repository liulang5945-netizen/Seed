"""Extension-only validation for frozen R2 dialogue training sources.

The base runner intentionally requires its original builder bytes. An extension
may accept only the reviewed builder delta, backed by the caller's validated
per-arm compatibility report; the other eleven base inputs remain byte-exact.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.training import run_taiji_r2_daily_dialogue_full_turn_pretraining as base

BUILDER_RELATIVE_PATH = base.BUILDER.relative_to(base.ROOT).as_posix()
REVIEWED_BUILDER_CURRENT_SHA256 = "0173946126a4fdd304b5c55f6aa73342f7455205236cdc2bd08a9544cfc3116b"
REVIEWED_BUILDER_GIT_REVISION = "02e82d921d14782fca511276020957220910a5c6"
REVIEWED_BUILDER_COMPATIBILITY = "reviewed_parent_blob_compatibility"
EXACT_COMPATIBILITY = "exact_parent_bytes"
PARENT_BOUND_COMPATIBILITY = "parent_bound_external_artifact"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_ARMS = frozenset({"full_turn_1m", "warm_start", "fresh_same_capacity"})


@dataclass(frozen=True)
class ExpectedSource:
    """A source path and immutable parent digest in the base twelve-item map."""

    relative_path: str
    path: Path
    parent_sha256: str
    is_builder: bool = False


@dataclass(frozen=True)
class ExtensionSourceValidation:
    """Verified source hashes and the base runner's train-only corpus objects."""

    arm: str
    source_hashes: dict[str, str]
    source_compatibility_report_sha256: str
    pretrain_corpus: Any
    assistant_corpus: Any
    manifest: dict[str, Any]
    readiness: dict[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _base_expected_sources() -> tuple[ExpectedSource, ...]:
    rows = (
        (base.PLAN, base.PLAN_SHA256),
        (base.BUILDER, base.BUILDER_SHA256),
        (base.PRETRAIN_EPISODES, base.PRETRAIN_EPISODES_SHA256),
        (base.PRETRAIN_MANIFEST, base.PRETRAIN_MANIFEST_SHA256),
        (base.PRETRAIN_READINESS, base.PRETRAIN_READINESS_SHA256),
        (base.RAW_SOURCE, base.RAW_SOURCE_SHA256),
        (base.ASSISTANT_EPISODES, base.ASSISTANT_EPISODES_SHA256),
        (base.ASSISTANT_MANIFEST, base.ASSISTANT_MANIFEST_SHA256),
        (base.MATERIAL_INVENTORY, base.MATERIAL_INVENTORY_SHA256),
        (base.OLD_GENERATIONS, base.OLD_GENERATIONS_SHA256),
        (base.OLD_RESULT, base.OLD_RESULT_SHA256),
        (base.BASELINE_CHECKPOINT, base.BASELINE_CHECKPOINT_SHA256),
    )
    return tuple(
        ExpectedSource(
            relative_path=path.resolve().relative_to(base.ROOT.resolve()).as_posix(),
            path=path,
            parent_sha256=expected_sha256,
            is_builder=path == base.BUILDER,
        )
        for path, expected_sha256 in rows
    )


def _validated_report_rows(
    arm: str, report: Mapping[str, Any]
) -> tuple[dict[str, Mapping[str, Any]], Mapping[str, str]]:
    if not isinstance(report, Mapping) or report.get("arm") != arm:
        raise base.Refuse("per-arm source compatibility evidence is missing or mismatched")
    if not isinstance(report.get("report_path"), str) or not report["report_path"]:
        raise base.Refuse("source compatibility report path is missing")
    for key in ("report_sha256", "runner_sha256"):
        if not isinstance(report.get(key), str) or not _SHA256_RE.fullmatch(report[key]):
            raise base.Refuse(f"source compatibility {key} is malformed")

    parent_identity = report.get("parent_identity")
    identity_keys = (
        "parent_attempt_sha256",
        "parent_receipt_sha256",
        "parent_bundle_sha256",
    )
    if not isinstance(parent_identity, Mapping) or any(
        not isinstance(parent_identity.get(key), str)
        or not _SHA256_RE.fullmatch(parent_identity[key])
        for key in identity_keys
    ):
        raise base.Refuse("source compatibility parent identity is incomplete")

    files = report.get("files")
    if not isinstance(files, list) or not files:
        raise base.Refuse("source compatibility file evidence is missing")
    rows: dict[str, Mapping[str, Any]] = {}
    for row in files:
        if not isinstance(row, Mapping):
            raise base.Refuse("source compatibility file row is malformed")
        relative = row.get("path")
        if not isinstance(relative, str) or not relative:
            raise base.Refuse("source compatibility file path is malformed")
        if relative in rows:
            raise base.Refuse(f"duplicate source compatibility row: {relative}")
        for key in ("parent_sha256", "current_sha256"):
            if not isinstance(row.get(key), str) or not _SHA256_RE.fullmatch(row[key]):
                raise base.Refuse(f"source compatibility {key} is malformed: {relative}")
        rows[relative] = row
    return rows, parent_identity


def _validate_source_hashes(
    arm: str,
    validated_source_compatibility: Mapping[str, Any],
    source_specs: Sequence[ExpectedSource],
) -> dict[str, str]:
    """Check the base twelve-item set, allowing only the pinned builder proof.

    ``source_specs`` is an explicit input for synthetic contract tests. The
    public validator below always supplies the immutable base source map.
    """
    if arm not in _ARMS:
        raise base.Refuse(f"unknown extension arm: {arm}")
    if len(source_specs) != 12 or sum(spec.is_builder for spec in source_specs) != 1:
        raise base.Refuse("extension source map must contain twelve entries and one builder")
    relative_paths = [spec.relative_path for spec in source_specs]
    if len(set(relative_paths)) != 12:
        raise base.Refuse("extension source map contains duplicate paths")
    builder_specs = [spec for spec in source_specs if spec.is_builder]
    if (
        builder_specs[0].relative_path != BUILDER_RELATIVE_PATH
        or builder_specs[0].parent_sha256 != base.BUILDER_SHA256
    ):
        raise base.Refuse("extension source map identifies a different builder parent")

    rows, report_parent_identity = _validated_report_rows(arm, validated_source_compatibility)
    result: dict[str, str] = {}
    for spec in source_specs:
        row = rows.get(spec.relative_path)
        if row is None:
            raise base.Refuse(f"source compatibility row is missing: {spec.relative_path}")
        if row.get("parent_sha256") != spec.parent_sha256:
            raise base.Refuse(f"source parent SHA differs: {spec.relative_path}")
        if not spec.path.is_file():
            raise base.Refuse(f"expected source is missing: {spec.relative_path}")
        current_sha256 = _sha256_file(spec.path)
        if row.get("current_sha256") != current_sha256:
            raise base.Refuse(f"source compatibility current SHA differs: {spec.relative_path}")

        if spec.is_builder:
            if current_sha256 != REVIEWED_BUILDER_CURRENT_SHA256:
                raise base.Refuse("current builder SHA is outside the reviewed compatibility")
            if row.get("compatibility") != REVIEWED_BUILDER_COMPATIBILITY:
                raise base.Refuse("builder lacks reviewed parent-blob compatibility evidence")
            if row.get("git_blob_sha256") != base.BUILDER_SHA256:
                raise base.Refuse("builder Git blob SHA does not prove the original parent source")
            git_revision = row.get("git_revision")
            git_commit = row.get("git_commit_sha")
            git_blob_oid = row.get("git_blob_oid")
            if (
                git_revision != REVIEWED_BUILDER_GIT_REVISION
                or git_commit != REVIEWED_BUILDER_GIT_REVISION
                or not isinstance(git_blob_oid, str)
                or not _GIT_SHA1_RE.fullmatch(git_blob_oid)
            ):
                raise base.Refuse("builder reviewed Git provenance is missing or unexpected")
        else:
            if current_sha256 != spec.parent_sha256:
                raise base.Refuse(
                    f"non-builder source differs from its original SHA: {spec.relative_path}"
                )
            compatibility = row.get("compatibility")
            if compatibility == EXACT_COMPATIBILITY:
                pass
            elif compatibility == PARENT_BOUND_COMPATIBILITY:
                if not spec.relative_path.startswith("data/"):
                    raise base.Refuse(
                        "parent-bound external compatibility is limited to data/: "
                        f"{spec.relative_path}"
                    )
                if row.get("git_provenance_applicable") is not False or row.get(
                    "parent_identity"
                ) != dict(report_parent_identity):
                    raise base.Refuse(
                        f"parent-bound external source lineage differs: {spec.relative_path}"
                    )
            else:
                raise base.Refuse(
                    f"non-builder source lacks exact-parent proof: {spec.relative_path}"
                )
        result[str(spec.path.resolve())] = current_sha256
    return result


def _validate_manifest_readiness(manifest: Mapping[str, Any], readiness: Mapping[str, Any]) -> None:
    if not isinstance(manifest, Mapping) or not isinstance(readiness, Mapping):
        raise base.Refuse("pretraining manifest/readiness must be objects")
    target_bytes = {"assistant": 1_643_502, "user": 171_237}
    if (
        readiness.get("status") != "ready"
        or readiness.get("episode_count") != base.PRETRAIN_EPISODE_COUNT
        or readiness.get("family_count") != base.PRETRAIN_FAMILY_COUNT
        or readiness.get("target_bytes_with_end_markers_by_role") != target_bytes
        or manifest.get("episode_jsonl_sha256") != base.PRETRAIN_EPISODES_SHA256
        or manifest.get("train_family_count") != base.PRETRAIN_FAMILY_COUNT
        or manifest.get("episode_count") != base.PRETRAIN_EPISODE_COUNT
        or manifest.get("target_bytes_with_end_markers_by_role") != target_bytes
    ):
        raise base.Refuse("full-turn pretraining readiness or manifest differs")


def validate_training_sources(
    validated_source_compatibility: Mapping[str, Any],
) -> ExtensionSourceValidation:
    """Validate extension lineage and load the base train-only corpora.

    The caller must pass the already-validated per-arm compatibility report;
    there is no no-evidence skip mode. Source validation precedes all corpus
    loads. No model/checkpoint is loaded and no evaluation split is requested.
    """
    arm = validated_source_compatibility.get("arm")
    if not isinstance(arm, str) or arm not in _ARMS:
        raise base.Refuse("validated source compatibility has an invalid arm")
    source_hashes = _validate_source_hashes(
        arm, validated_source_compatibility, _base_expected_sources()
    )
    manifest = base._read_json(base.PRETRAIN_MANIFEST)
    readiness = base._read_json(base.PRETRAIN_READINESS)
    _validate_manifest_readiness(manifest, readiness)
    pretrain_corpus = base._load_pretrain_corpus()
    assistant_corpus = base._load_assistant_train_corpus()
    pretrain_target_bytes = sum(len(episode.target_bytes) for episode in pretrain_corpus.episodes)
    if pretrain_target_bytes != base.PRETRAIN_TARGET_BYTES_PER_EPOCH:
        raise base.Refuse("loaded full-turn training digest has a different target-byte total")
    return ExtensionSourceValidation(
        arm=arm,
        source_hashes=source_hashes,
        source_compatibility_report_sha256=str(validated_source_compatibility["report_sha256"]),
        pretrain_corpus=pretrain_corpus,
        assistant_corpus=assistant_corpus,
        manifest=dict(manifest),
        readiness=dict(readiness),
    )
