"""Contract tests for 22.5.V.1 hard-selection readout (zero training)."""

from __future__ import annotations

import json

import pytest
import torch

from scripts.training import run_v1_hard_selection_20261117 as v1
from taiji.u2_neural_units import UnitGraphSpec, apply_capsule, load_capsule


def contract() -> dict:
    return json.loads(v1.CONTRACT.read_text(encoding="utf-8"))


def episodes() -> dict:
    loaded = v1.u1.load_data(contract())
    return v1.u1.prepare_episodes(loaded, UnitGraphSpec().n_units)


def test_contract_frozen_with_measured_cap() -> None:
    data = contract()
    assert data["status"] == "frozen"
    assert data["training_authority"] is None
    assert data["criteria_sha256"] == v1.value_digest(data["criteria"])
    assert data["criteria"]["route"] == "B_hard_select"
    assert data["criteria"]["parent_route"] == "B_position_temp"
    assert data["plan"]["cap_cpu"] > 0
    assert data["plan"]["cap_wall"] > 0
    assert data["plan"]["design_cap_cpu"] >= data["plan"]["cap_cpu"]


def test_source_pins_and_parent_capsules_match() -> None:
    data = contract()
    for name, path in v1.pinned_files():
        assert data["source_pins"][name] == v1.sha_of(path), name
    for key, path in data["start_parents"].items():
        assert data["start_parent_sha256"][key] == v1.binary_sha(v1.ROOT / path), key


def test_parent_capsule_is_loudly_rejected() -> None:
    seed = 2026111601
    parent = load_capsule(v1.parent_capsule_path(seed))
    graph = v1.build_graph(seed)
    with pytest.raises(ValueError):
        apply_capsule(graph, parent)


def test_only_two_tensors_are_trainable() -> None:
    graph = v1.build_graph(2026111601)
    trainable = {name for name, param in graph.named_parameters() if param.requires_grad}
    assert trainable == set(v1.TRAINABLE)
    assert len(trainable) == 2


def test_hard_forward_matches_parent_chain() -> None:
    seed = 2026111601
    parent_capsule = load_capsule(v1.parent_capsule_path(seed))
    spec = UnitGraphSpec(**parent_capsule["runtime"]["spec"])
    parent = v1.r2.TemperaturePositionGraph(spec)
    parent.load_state_dict(parent_capsule["runtime"]["state_dict"])
    child = v1.build_graph(seed)
    item = episodes()["fit"][0]
    data = v1.u1.cast_item(item, spec.torch_dtype)
    with torch.no_grad():
        parent_out = parent(data["channels"], data["gates"], parent.initial_state())
        child_out = child(data["channels"], data["gates"], child.initial_state())
    assert torch.equal(parent_out["attention"], child_out["attention"])
    assert torch.equal(parent_out["attended"], child_out["attended"])
    assert child_out["selected"] == int(torch.argmax(child_out["attention"]).item())
    manual = child.position_head(child_out["projected"][child_out["selected"]])
    assert torch.equal(manual, child_out["readout"])
    assert set(parent.state_dict()) == set(child.state_dict())


def test_run_guards_refuse_without_authority() -> None:
    with pytest.raises(ValueError, match="missing_training_authority"):
        v1.guards(contract(), "run")


def test_price_report_matches_contract() -> None:
    data = contract()
    price = json.loads(v1.PRICE_REPORT.read_text(encoding="utf-8"))
    assert price["criteria_sha256"] == data["criteria_sha256"]
    assert price["cap_cpu"] == data["plan"]["cap_cpu"]
    assert price["cap_wall"] == data["plan"]["cap_wall"]
    assert price["cap_cpu"] == round(price["measured_cpu"] * v1.CAP_MARGIN, 4)
    assert price["models_discarded"] is True


def test_smoke_report_passed() -> None:
    smoke = json.loads(v1.SMOKE_REPORT.read_text(encoding="utf-8"))
    assert smoke["verdict"] == "smoke_passed"
    assert smoke["uncovered"] == []
    assert all(smoke["checks"].values())
    assert smoke["discipline"]["optimizer_steps"] == 0
