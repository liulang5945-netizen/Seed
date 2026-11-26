"""Explicit parameter/moment-preserving rebaseline across an audited operator change."""

from __future__ import annotations

import hashlib
from pathlib import Path

from scripts.training.evidence_a_learner_22_5 import load_learner, source_pins
from taiji.u2_history import assert_values_equal, atomic_load, atomic_save

ROOT = Path(__file__).resolve().parents[2]


def rebase_parent(source, destination, reference):
    payload = atomic_load(source)
    current = source_pins()
    assert set(payload["source_pins"]) == set(current)
    archive = ROOT / reference["archive"]
    assert (
        hashlib.sha256(archive.read_text(encoding="utf-8-sig").encode()).hexdigest()
        == reference["sha256"]
    )
    differences = {name for name in current if current[name] != payload["source_pins"][name]}
    assert differences == {reference["source"]}
    assert payload["source_pins"][reference["source"]] == reference["sha256"]
    old_optimizer = payload["optimizer"]
    old_parameters = payload["model"]["parameters"]
    payload["source_pins"] = current
    payload["lineage"] = {
        **payload["lineage"],
        "operator_rebaseline": {
            "source_checkpoint_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
            "old_operator_sha256": reference["sha256"],
            "new_operator_sha256": current[reference["source"]],
            "scope": "same parameter and optimizer values; implementation revision plus one",
        },
    }
    if payload["architecture"].startswith("native-"):
        payload["model"]["parameter_revision"] += 1
        payload["model"]["state"]["revision"] += 1
    else:
        payload["model"]["revision"] += 1
    if Path(destination).exists():
        raise FileExistsError(destination)
    atomic_save(payload, destination)
    model, optimizer, loaded = load_learner(destination)
    assert_values_equal(old_parameters, model.state_dict(), "operator_migration.parameters")
    assert_values_equal(old_optimizer, optimizer.state_dict(), "operator_migration.optimizer")
    for name in ("torch_rng", "update", "cursor", "order", "data_digest"):
        assert_values_equal(payload[name], loaded[name], f"operator_migration.{name}")
    return model, optimizer, loaded
