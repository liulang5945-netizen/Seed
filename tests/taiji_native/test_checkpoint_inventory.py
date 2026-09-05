"""R0.6: checkpoint/owner inventory must admit sequence-derived memory children.

The legacy m2ad loader hard-coded ``training_phases == ["sequence"]`` and
silently excluded memory/identity-phase children that continue a sequence
child with the shared fabric frozen.  These tests pin the corrected
sequence-derived admission rule and the digest-integrity check that the
inventory tool relies on.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from taiji.internalization import content_digest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "training" / "inventory_taiji_checkpoints.py"
_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def inventory() -> object:
    spec = importlib.util.spec_from_file_location("inventory_taiji_checkpoints", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _write_joint_child(
    path: Path,
    *,
    training_phases: list[str],
    sequence_fabric_learning: bool,
    sequence_predictive_context_mode: str = "private-plastic-temporal-v1",
) -> Path:
    payload = {
        "format": "taiji-native-joint-training-v1",
        "version": 4,
        "model_tier": "joint",
        "training_phases": training_phases,
        "phase": training_phases[-1],
        "parent_checkpoint_digest": "parent" * 10,
        "continuation_source_checkpoint_digest": "ancestor" * 9,
        "dataset_digest": "data" * 8,
        "sequence_fabric_learning": sequence_fabric_learning,
        "sequence_predictive_context_mode": sequence_predictive_context_mode,
        "sequence_readout_mode": "dedicated-predictive-v1",
        "sequence_fabric_phase_checks": [],
        "sequence_predictive_context_phase_checks": [],
        "sequence_readout_phase_checks": [],
        "identity_growth": [],
        "model": {"fabric": {}, "predictive_context": {}, "predictive_readout": {}},
    }
    payload["checkpoint_digest"] = content_digest(
        {key: value for key, value in payload.items() if key != "checkpoint_digest"}
    )
    torch.save(payload, path)
    return path


def test_inventory_admits_sequence_derived_memory_child(inventory: object, tmp_path: Path) -> None:
    path = _write_joint_child(
        tmp_path / "memory-child.pt",
        training_phases=["memory"],
        sequence_fabric_learning=False,
    )

    entry = inventory.inspect_checkpoint(path)  # type: ignore[attr-defined]

    assert entry["digest_verified"] is True
    assert entry["format"] == "taiji-native-joint-training-v1"
    assert entry["training_phases"] == ["memory"]
    assert entry["sequence_derived_admission"] is True
    assert entry["organ_coverage"]["fabric"]["sequence_fabric_learning"] is False


def test_inventory_rejects_non_sequence_child_with_mutable_fabric(
    inventory: object, tmp_path: Path
) -> None:
    path = _write_joint_child(
        tmp_path / "fresh-only.pt",
        training_phases=["world"],
        sequence_fabric_learning=True,
    )

    entry = inventory.inspect_checkpoint(path)  # type: ignore[attr-defined]

    assert entry["digest_verified"] is True
    assert entry["sequence_derived_admission"] is False


def test_inventory_reports_missing_file(inventory: object, tmp_path: Path) -> None:
    entry = inventory.inspect_checkpoint(tmp_path / "absent.pt")  # type: ignore[attr-defined]

    assert entry["exists"] is False
    assert "digest_verified" not in entry
