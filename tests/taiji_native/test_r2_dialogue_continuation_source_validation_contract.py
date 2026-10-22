from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.training import run_taiji_r2_daily_dialogue_full_turn_pretraining as base
from scripts.training.taiji_r2_dialogue_continuation_source_validation import (
    BUILDER_RELATIVE_PATH,
    EXACT_COMPATIBILITY,
    PARENT_BOUND_COMPATIBILITY,
    REVIEWED_BUILDER_COMPATIBILITY,
    REVIEWED_BUILDER_CURRENT_SHA256,
    REVIEWED_BUILDER_GIT_REVISION,
    ExpectedSource,
    _base_expected_sources,
    _validate_manifest_readiness,
    _validate_source_hashes,
    validate_training_sources,
)

ARMS = ("full_turn_1m", "warm_start", "fresh_same_capacity")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_source_contract(tmp_path: Path, arm: str):
    if _sha256(base.BUILDER) != REVIEWED_BUILDER_CURRENT_SHA256:
        pytest.skip("checkout no longer has the source version reviewed by this extension")

    parent_identity = {
        "parent_attempt_sha256": "1" * 64,
        "parent_receipt_sha256": "2" * 64,
        "parent_bundle_sha256": "3" * 64,
    }
    specs: list[ExpectedSource] = []
    rows: list[dict[str, object]] = []

    builder_copy = tmp_path / "reviewed-builder.py"
    builder_copy.write_bytes(base.BUILDER.read_bytes())
    specs.append(
        ExpectedSource(
            relative_path=BUILDER_RELATIVE_PATH,
            path=builder_copy,
            parent_sha256=base.BUILDER_SHA256,
            is_builder=True,
        )
    )
    rows.append(
        {
            "path": BUILDER_RELATIVE_PATH,
            "parent_sha256": base.BUILDER_SHA256,
            "current_sha256": _sha256(builder_copy),
            "compatibility": REVIEWED_BUILDER_COMPATIBILITY,
            "git_blob_sha256": base.BUILDER_SHA256,
            "git_revision": REVIEWED_BUILDER_GIT_REVISION,
            "git_commit_sha": REVIEWED_BUILDER_GIT_REVISION,
            "git_blob_oid": "4" * 40,
        }
    )

    for index in range(11):
        relative_path = (
            f"data/source-{index}.json" if index % 2 == 0 else f"reports/source-{index}.json"
        )
        path = tmp_path / f"source-{index}.json"
        path.write_bytes(f"frozen-source-{index}".encode())
        digest = _sha256(path)
        specs.append(
            ExpectedSource(
                relative_path=relative_path,
                path=path,
                parent_sha256=digest,
            )
        )
        if relative_path.startswith("data/"):
            rows.append(
                {
                    "path": relative_path,
                    "parent_sha256": digest,
                    "current_sha256": digest,
                    "compatibility": PARENT_BOUND_COMPATIBILITY,
                    "git_provenance_applicable": False,
                    "parent_identity": parent_identity,
                }
            )
        else:
            rows.append(
                {
                    "path": relative_path,
                    "parent_sha256": digest,
                    "current_sha256": digest,
                    "compatibility": EXACT_COMPATIBILITY,
                }
            )

    report = {
        "arm": arm,
        "report_path": str(tmp_path / "source-compatibility.json"),
        "report_sha256": "a" * 64,
        "runner_sha256": "b" * 64,
        "parent_identity": parent_identity,
        "files": rows,
    }
    return specs, report


def test_frozen_base_validator_reproduces_reviewed_builder_sha_rejection() -> None:
    specs = _base_expected_sources()
    assert len(specs) == 12
    assert _sha256(base.BUILDER) == REVIEWED_BUILDER_CURRENT_SHA256
    for spec in specs:
        if not spec.is_builder:
            assert _sha256(spec.path) == spec.parent_sha256

    with pytest.raises(base.Refuse, match="full-turn dialogue source changed") as error:
        base._validate_sources()

    assert str(base.BUILDER) in str(error.value)


@pytest.mark.parametrize("arm", ARMS)
def test_extension_hash_gate_accepts_only_per_arm_reviewed_builder_proof(
    tmp_path: Path, arm: str
) -> None:
    specs, report = _synthetic_source_contract(tmp_path, arm)

    hashes = _validate_source_hashes(arm, report, specs)

    assert len(hashes) == 12
    assert hashes[str(specs[0].path.resolve())] == REVIEWED_BUILDER_CURRENT_SHA256
    wrong_arm = "warm_start" if arm != "warm_start" else "full_turn_1m"
    with pytest.raises(base.Refuse, match="per-arm source compatibility"):
        _validate_source_hashes(wrong_arm, report, specs)


def test_builder_drift_after_compatibility_review_is_rejected(tmp_path: Path) -> None:
    specs, report = _synthetic_source_contract(tmp_path, "full_turn_1m")
    specs[0].path.write_bytes(b"different builder bytes")

    with pytest.raises(base.Refuse, match="source compatibility current SHA differs"):
        _validate_source_hashes("full_turn_1m", report, specs)


def test_non_builder_data_drift_is_rejected_even_with_builder_proof(tmp_path: Path) -> None:
    specs, report = _synthetic_source_contract(tmp_path, "warm_start")
    data_spec = next(spec for spec in specs if spec.relative_path.startswith("data/"))
    data_spec.path.write_bytes(b"changed parent-bound data")

    with pytest.raises(base.Refuse, match="source compatibility current SHA differs"):
        _validate_source_hashes("warm_start", report, specs)


def test_missing_compatibility_evidence_is_not_a_skip_path(tmp_path: Path) -> None:
    specs, _ = _synthetic_source_contract(tmp_path, "fresh_same_capacity")

    with pytest.raises(base.Refuse, match="source compatibility evidence is missing"):
        _validate_source_hashes("fresh_same_capacity", {}, specs)


def test_base_manifest_readiness_target_byte_gate_is_preserved() -> None:
    target_bytes = {"assistant": 1_643_502, "user": 171_237}
    manifest = {
        "episode_jsonl_sha256": base.PRETRAIN_EPISODES_SHA256,
        "train_family_count": base.PRETRAIN_FAMILY_COUNT,
        "episode_count": base.PRETRAIN_EPISODE_COUNT,
        "target_bytes_with_end_markers_by_role": target_bytes,
    }
    readiness = {
        "status": "ready",
        "episode_count": base.PRETRAIN_EPISODE_COUNT,
        "family_count": base.PRETRAIN_FAMILY_COUNT,
        "target_bytes_with_end_markers_by_role": target_bytes,
    }

    _validate_manifest_readiness(manifest, readiness)

    readiness["episode_count"] = base.PRETRAIN_EPISODE_COUNT - 1
    with pytest.raises(base.Refuse, match="readiness or manifest differs"):
        _validate_manifest_readiness(manifest, readiness)


def test_public_validator_reuses_base_train_only_loaders_when_inputs_exist() -> None:
    if any(not spec.path.is_file() for spec in _base_expected_sources()):
        pytest.skip("base R2 train-only source fixture is unavailable in this checkout")

    from scripts.training import run_taiji_r2_daily_dialogue_continuation_extension as extension

    arm = "full_turn_1m"
    parent_paths = extension._expected_parent_paths(arm)
    parent_attempt = extension.read_json(parent_paths["parent_attempt"])
    compatibility = extension.validate_source_compatibility(
        arm,
        parent_attempt["bindings"]["source_hashes"],
        {
            "parent_attempt_sha256": extension.PARENT_ATTEMPT_SHA256[arm],
            "parent_receipt_sha256": extension.PARENT_RECEIPT_SHA256[arm],
            "parent_bundle_sha256": extension.PARENT_BUNDLE_SHA256[arm],
        },
    )

    validation = validate_training_sources(compatibility)

    assert len(validation.pretrain_corpus.episodes) == base.PRETRAIN_EPISODE_COUNT
    assert len(validation.assistant_corpus.episodes) == base.ASSISTANT_TRAIN_EPISODE_COUNT
    assert validation.manifest["episode_jsonl_sha256"] == base.PRETRAIN_EPISODES_SHA256
    assert validation.readiness["status"] == "ready"
    assert validation.source_hashes[str(base.BUILDER.resolve())] == REVIEWED_BUILDER_CURRENT_SHA256
