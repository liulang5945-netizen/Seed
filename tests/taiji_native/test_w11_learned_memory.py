"""Contract tests for 22.5.W.1.1 learned-memory DMS parent acquisition."""

from __future__ import annotations

import json

import pytest

from scripts.training import probe_u2_w_dms_20261117 as wprobe
from scripts.training import run_w11_learned_memory_20261117 as w11
from taiji import u2_unit_input_dms as dms
from taiji.u2_history import value_digest
from taiji.u2_neural_units import apply_capsule, build_capsule, load_capsule, save_capsule
from taiji.u2_unit_memory_readout import LearnedMemoryGraph


def contract() -> dict:
    return json.loads(w11.CONTRACT.read_text(encoding="utf-8"))


def test_contract_frozen_with_measured_cap() -> None:
    data = contract()
    assert data["status"] == "frozen"
    assert data["training_authority"] is None
    assert data["criteria_sha256"] == value_digest(data["criteria"])
    assert data["criteria"]["route"] == "B_learned_memory"
    assert data["plan"]["cap_cpu"] > 0
    assert data["plan"]["cap_wall"] > 0
    assert data["plan"]["design_cap_cpu"] >= data["plan"]["cap_cpu"]


def test_source_pins_match() -> None:
    data = contract()
    for name, path in w11.pinned_files():
        assert data["source_pins"][name] == w11.sha_of(path), name


def test_trainable_is_exactly_gate_and_readout() -> None:
    graph = w11.build_graph(2026111701)
    trainable = {name for name, param in graph.named_parameters() if param.requires_grad}
    assert trainable == set(w11.TRAINABLE)
    assert len(trainable) == 5


def test_neutral_birth_readout_and_gate() -> None:
    graph = w11.build_graph(2026111701)
    dataset = wprobe.make_dataset()
    logits = w11.run_episode(graph, dataset["train"][0])
    assert float(logits.abs().max().detach()) == 0.0
    assert float(graph.write_gate.weight.detach().abs().max()) == 0.0
    assert float(graph.memory_readout.weight.detach().abs().max()) == 0.0


def test_gradient_coverage() -> None:
    graph = w11.build_graph(2026111701)
    dataset = wprobe.make_dataset()
    graph.zero_grad(set_to_none=True)
    logits = w11.run_episode(graph, dataset["train"][0])
    logits.square().mean().backward()
    graded = {
        name
        for name, param in graph.named_parameters()
        if param.requires_grad and param.grad is not None
    }
    frozen = [
        name
        for name, param in graph.named_parameters()
        if not param.requires_grad and param.grad is not None
    ]
    assert graded == set(w11.TRAINABLE)
    assert frozen == []


def test_checkpoint_roundtrip() -> None:
    graph = LearnedMemoryGraph(w11.UnitGraphSpec(dtype="float64", seed=2026111701))
    capsule = build_capsule(graph)
    path = w11.ARTIFACTS / "test_roundtrip.bin"
    w11.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    save_capsule(path, capsule)
    reloaded = load_capsule(path)
    assert value_digest(reloaded) == value_digest(load_capsule(path))
    path.unlink(missing_ok=True)


def test_run_guards_refuse_without_authority() -> None:
    with pytest.raises(ValueError, match="missing_training_authority"):
        w11.guards(contract(), "run")


def test_price_matches_cap() -> None:
    data = contract()
    price = json.loads(w11.PRICE_REPORT.read_text(encoding="utf-8"))
    assert price["criteria_sha256"] == data["criteria_sha256"]
    assert price["cap_cpu"] == data["plan"]["cap_cpu"]
    assert abs(price["cap_cpu"] - price["measured_cpu"] * w11.CAP_MARGIN) <= 1e-3
    assert price["models_discarded"] is True


def test_smoke_report_passed() -> None:
    smoke = json.loads(w11.SMOKE_REPORT.read_text(encoding="utf-8"))
    assert smoke["verdict"] == "smoke_passed"
    assert smoke["uncovered"] == []
    assert all(smoke["checks"].values())
    assert smoke["discipline"]["optimizer_steps"] == 0


def test_encoding_and_partition() -> None:
    data = contract()
    assert data["encoding_contract_digest"] == value_digest(dms.encoding_contract())
    dataset = wprobe.make_dataset()
    assert len(dataset["train"]) == 64
    assert len(dataset["fit"]) == 32
    train = {(item["sample"], item["position"]) for item in dataset["train"]}
    fit = {(item["sample"], item["position"]) for item in dataset["fit"]}
    assert not (train & fit)


def test_capsule_route_mismatch_is_loud() -> None:
    graph = LearnedMemoryGraph(w11.UnitGraphSpec(dtype="float64", seed=2026111701))
    dataset = wprobe.make_dataset()
    w11.run_episode(graph, dataset["train"][0])
    capsule = build_capsule(graph)
    capsule["runtime"]["route"] = "B_other"
    with pytest.raises(ValueError):
        apply_capsule(graph, capsule)
