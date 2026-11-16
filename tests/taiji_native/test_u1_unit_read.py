"""Zero-training contract tests for 22.5.U.1 (unit-native read foundation)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts.training import run_u1_unit_read_20261116 as u1
from taiji.u2_history import atomic_load, value_digest
from taiji.u2_neural_units import UnitGraphSpec, build_route
from taiji.u2_unit_input import (
    CHANNELS,
    color_mass,
    derive_label,
    encode_scene,
    encoding_contract,
    scene_entity,
)

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "plans/manifests/taiji_u2_u1_unit_read_20261116.json"


@pytest.fixture(scope="module")
def contract() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def data(contract: dict) -> dict:
    payload = atomic_load(ROOT / contract["data_root"])
    assert value_digest(payload) == contract["data_digest"]
    return payload


def test_encoding_contract_digest_frozen(contract: dict) -> None:
    assert value_digest(encoding_contract()) == contract["encoding_contract_digest"]
    assert encoding_contract()["identity"] == contract["encoding_contract_identity"]


def test_encoding_shapes_and_channels(data: dict) -> None:
    example = data["fit"][0]
    channels = encode_scene(example["public"], torch.float64)
    assert set(channels) == set(CHANNELS)
    for name, value in channels.items():
        assert value.shape == (16, 8), name
    for name in ("audio", "proprio", "feedback"):
        assert float(channels[name].abs().sum()) == 0.0


def test_encoding_oracle_consistent_on_fit(data: dict) -> None:
    ok = 0
    for example in data["fit"]:
        mass = color_mass(example["public"]["observations"]["rgb"])
        entity = scene_entity(example["public"]["request"])
        ok += int(derive_label(mass, entity) == example["target"].decode("utf-8"))
    assert ok == len(data["fit"])


def test_source_pins_match(contract: dict) -> None:
    for name, pin in contract["source_pins"].items():
        path = {
            "runner": ROOT / "scripts/training/run_u1_unit_read_20261116.py",
            "unit_input": ROOT / "taiji/u2_unit_input.py",
            "neural_units": ROOT / "taiji/u2_neural_units.py",
            "support": ROOT / "scripts/training/first_a_support_22_5.py",
            "columns": ROOT / "taiji/u2_observation_columns.py",
            "episode": ROOT / "taiji/u2_task_episode.py",
        }[name]
        assert u1.sha_of(path) == pin, name


def test_b_identity_and_cut_h() -> None:
    graph = build_route("B", UnitGraphSpec(dtype="float64", seed=2026111601))
    assert all(u1.b_identity_checks(graph).values())
    item = {
        "channels": encode_scene(
            atomic_load(ROOT / "data/experiments/taiji_u2/first_a/20261112T074836Z/data.bin")[
                "fit"
            ][0]["public"],
            torch.float64,
        ),
        "gates": u1.make_gates(graph.units.n_units, torch.float64),
        "label": 0,
    }
    data = u1.cast_item(item, graph.spec.torch_dtype)
    base = graph(data["channels"], data["gates"], graph.initial_state())["readout"]
    h_state = {**graph.initial_state(), "H": graph.initial_state()["H"] + 0.5}
    cut = graph(data["channels"], data["gates"], graph.cut_h(h_state))["readout"]
    assert torch.allclose(base, cut, rtol=0.0, atol=0.0)


def test_neutral_birth_exact() -> None:
    graph = build_route("B", UnitGraphSpec(dtype="float64", seed=2026111601))
    payload = atomic_load(ROOT / "data/experiments/taiji_u2/first_a/20261112T074836Z/data.bin")
    item = {
        "channels": encode_scene(payload["fit"][0]["public"], torch.float64),
        "gates": u1.make_gates(graph.units.n_units, torch.float64),
        "label": 0,
    }
    result = u1.neutral_birth_check(graph, item)
    assert result["neutral_birth_exact"]
    assert result["neutral_birth_max_abs"] == 0.0
    assert result["new_unit_r_credit_path_wired"]


def test_intervention_mechanisms_wired() -> None:
    graph = build_route("B", UnitGraphSpec(dtype="float64", seed=2026111601))
    payload = atomic_load(ROOT / "data/experiments/taiji_u2/first_a/20261112T074836Z/data.bin")
    episodes = u1.prepare_episodes(payload, graph.units.n_units)
    item = episodes["fit"][0]
    history_item = episodes["fit"][1]
    hd = u1.cast_item(history_item, graph.spec.torch_dtype)
    hist_state = graph(hd["channels"], hd["gates"], graph.initial_state())["next_state"]
    probe = u1.mechanism_probe(graph)
    summary = u1.intervention_summary(u1.intervention_readouts(probe, item, hist_state))
    assert summary["without_edges_changes_readout"]
    assert summary["permute_source_changes_readout"]
    assert summary["history_changes_readout"]
    assert summary["cut_h_bit_identical"]
    assert summary["h_perturb_bit_identical"]


def test_gradient_coverage() -> None:
    graph = build_route("B", UnitGraphSpec(dtype="float64", seed=2026111601))
    payload = atomic_load(ROOT / "data/experiments/taiji_u2/first_a/20261112T074836Z/data.bin")
    episodes = u1.prepare_episodes(payload, graph.units.n_units)
    coverage = u1.gradient_coverage(graph, episodes["fit"][0])
    assert coverage["all_parameters_have_gradient"]
    assert coverage["named_nonzero"]


def test_criteria_gates_frozen(contract: dict) -> None:
    criteria = contract["criteria"]
    assert criteria["fit_gate"] == 0.80
    assert criteria["train_gate"] == 0.90
    assert criteria["distinct_min"] == 3
    assert criteria["control_agreement_max"] == 0.5
    assert len(criteria["seeds"]) == 3
    assert value_digest(criteria) == contract["criteria_sha256"]
    assert contract["status"] == "frozen"
    assert contract["plan"]["cap_cpu"] <= contract["plan"]["design_cap_cpu"]
