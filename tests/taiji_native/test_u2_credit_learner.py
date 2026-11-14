"""P3.8.2b credit-update operator and full learner capsule contracts.

These are structural/recovery gates, not capability claims.  Each guard has a
red direction so a broken recovery path cannot pass silently.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from taiji import u2_credit_learner as capsule_mod
from taiji.u2_credit_learner import (
    BODY_TENSORS,
    HEAD_TENSORS,
    REGISTRY_TENSORS,
    build_capsule,
    capsule_summary,
    headless_runtime,
    load_capsule,
    save_capsule,
    validate_capsule,
)
from taiji.u2_history import value_digest

ROOT = Path(__file__).resolve().parents[2]
RESEARCH_OPERATOR = ROOT / "scripts/training/credit_update_22_5.py"
FORMAL_OPERATOR = ROOT / "taiji/u2_credit_update.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _trial(step):
    torch.manual_seed(20261114)
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    features = torch.tensor([[1.0, 2.0, 3.0]])
    target = torch.tensor([[1.0]])
    counters = {"optimizer_steps": 0}

    def objective():
        return (model(features) - target).square().mean()

    outcome = step(list(model.parameters()), optimizer, objective, counters)
    flat = torch.cat([item.detach().reshape(-1) for item in model.parameters()])
    return outcome, flat.numpy().tobytes(), value_digest(optimizer.state_dict()), counters


def make_capsule(*, u=0, v=0, proposal=None, runtime=None, moment=None):
    ids = [f"body.{index}" for index in range(BODY_TENSORS)] + [
        "content_head.weight",
        "content_head.bias",
    ]
    return build_capsule(
        native_runtime=runtime
        or {
            "model_parameters": {"body.0": torch.zeros(2, 2)},
            "field_count": 44,
            "fact_history": [],
            "pending": [],
            "rng": torch.get_rng_state(),
            "resources": {"cpu_seconds": 0.0},
        },
        learning_state={
            "body_moment": moment if moment is not None else {},
            "content_head": {
                "weight": torch.zeros(4, 8),
                "bias": torch.zeros(4),
            },
            "content_head_moment": {},
            "label_source": "train_answer_hex_v1",
            "class_weights": [1.0, 1.0, 1.0, 1.0],
            "parameter_ancestor": "a" * 64,
            "proposal_clock": u,
            "parameter_version": v,
        },
        sampling={
            "episode_cursor": 0,
            "first_answer_prefix": "",
            "prefix_source": "actual_first_answer",
            "data_rights": {"train": True, "dev": False, "check": False, "final": False},
            "sampling_rng": torch.get_rng_state(),
        },
        proposal=proposal,
        identity={
            "factory_schema": "taiji-u2-joint-token-query-v1",
            "source_pins": {"taiji/u2_credit_update.py": "0" * 64},
            "parameter_registry": {
                "parameter_ids": ids,
                "owners": {"body": BODY_TENSORS, "content_head": HEAD_TENSORS},
            },
            "body_digest": "b" * 64,
        },
    )


def test_formal_operator_is_byte_and_behaviour_identical_to_research():
    assert RESEARCH_OPERATOR.read_bytes() == FORMAL_OPERATOR.read_bytes()
    research = _load(RESEARCH_OPERATOR, "research_credit_update")
    formal = _load(FORMAL_OPERATOR, "formal_credit_update")
    left = _trial(research.accepted_adam_step)
    right = _trial(formal.accepted_adam_step)
    assert left == right
    assert left[3]["optimizer_steps"] == 1


def test_initial_capsule_roundtrip(tmp_path):
    capsule = make_capsule()
    path = tmp_path / "initial.bin"
    save_capsule(path, capsule)
    restored = load_capsule(path)
    assert value_digest(restored) == value_digest(capsule)
    assert restored["capsule_digest"] == capsule["capsule_digest"]
    assert capsule_summary(restored)["registry_tensors"] == REGISTRY_TENSORS


def test_tampered_capsule_is_rejected(tmp_path):
    benign = make_capsule()
    benign["sampling"]["episode_cursor"] = 5
    with pytest.raises(ValueError, match="capsule_digest_mismatch"):
        validate_capsule(benign)
    clock = make_capsule()
    clock["learning_state"]["parameter_version"] = 7
    with pytest.raises(ValueError, match="capsule_clock_order"):
        validate_capsule(clock)
    wrong_schema = make_capsule()
    wrong_schema["schema"] = "taiji-u2-native-learner-capsule-v0"
    with pytest.raises(ValueError, match="capsule_schema_mismatch"):
        validate_capsule(wrong_schema)
    assert not (tmp_path / "never_written.bin").exists()


def test_first_nonempty_moment_roundtrip(tmp_path):
    moment = {"body.0": {"exp_avg": torch.full((2, 2), 0.25), "step": torch.tensor(1)}}
    proposal = {
        "proposal_id": "p-000001",
        "alpha": 0.5,
        "ancestor_digest": "c" * 64,
        "objective_value": 0.125,
        "search_state": "computed",
    }
    capsule = make_capsule(u=1, v=0, proposal=proposal, moment=moment)
    path = tmp_path / "moment.bin"
    save_capsule(path, capsule)
    restored = load_capsule(path)
    assert restored["learning_state"]["body_moment"]["body.0"]["exp_avg"].tolist() == [
        [0.25, 0.25],
        [0.25, 0.25],
    ]
    assert restored["learning_state"]["proposal_clock"] == 1
    assert restored["learning_state"]["parameter_version"] == 0
    assert restored["proposal"] == proposal
    assert validate_capsule(restored)["sampling"]["episode_cursor"] == 0


def test_headless_runtime_excludes_training_only_state():
    capsule = make_capsule()
    runtime = headless_runtime(capsule)
    assert "content_head" not in runtime
    assert "field_count" in runtime
    polluted = make_capsule(runtime={"model_parameters": {}, "class_weights": [1.0]})
    with pytest.raises(ValueError, match="capsule_headless_leak"):
        headless_runtime(polluted)


def test_registry_and_clock_guards_fire():
    capsule = make_capsule()
    capsule["identity"]["parameter_registry"]["parameter_ids"] = capsule["identity"][
        "parameter_registry"
    ]["parameter_ids"][:-1]
    with pytest.raises(ValueError, match="capsule_registry_count|capsule_digest_mismatch"):
        validate_capsule(capsule)
    bad_clock = make_capsule(u=1, v=0)
    bad_clock["learning_state"]["parameter_version"] = 3
    bad_clock["capsule_digest"] = value_digest({key: bad_clock[key] for key in capsule_mod.PARTS})
    with pytest.raises(ValueError, match="capsule_clock_order"):
        validate_capsule(bad_clock)
