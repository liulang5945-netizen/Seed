"""Zero-training contract tests for 22.5.U.1 revision 1 (position-aware readout)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts.training import run_u1_unit_read_20261116 as u1
from scripts.training import run_u1r1_position_20261116 as u1r1
from taiji.u2_history import atomic_load, value_digest
from taiji.u2_unit_input import encode_scene, encoding_contract

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "plans/manifests/taiji_u2_u1r1_position_20261116.json"
DATA = ROOT / "data/experiments/taiji_u2/first_a/20261112T074836Z/data.bin"


@pytest.fixture(scope="module")
def contract() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def data() -> dict:
    return atomic_load(DATA)


def test_encoding_contract_frozen_unchanged(contract: dict) -> None:
    assert value_digest(encoding_contract()) == contract["encoding_contract_digest"]
    assert contract["encoding_contract_identity"] == "unit-input-column-color-v1"


def test_source_pins_match(contract: dict) -> None:
    paths = {
        "runner": ROOT / "scripts/training/run_u1r1_position_20261116.py",
        "position_readout": ROOT / "taiji/u2_unit_position_readout.py",
        "u1_runner": ROOT / "scripts/training/run_u1_unit_read_20261116.py",
        "unit_input": ROOT / "taiji/u2_unit_input.py",
        "neural_units": ROOT / "taiji/u2_neural_units.py",
        "support": ROOT / "scripts/training/first_a_support_22_5.py",
        "columns": ROOT / "taiji/u2_observation_columns.py",
        "episode": ROOT / "taiji/u2_task_episode.py",
    }
    for name, pin in contract["source_pins"].items():
        assert u1r1.sha_of(paths[name]) == pin, name


def test_position_identity_and_cut_h(data: dict) -> None:
    graph = u1r1.build_graph(2026111601)
    assert all(u1r1.b_identity_checks(graph).values())
    assert not any(
        name.startswith("h_readout") or name in ("ell", "w_g", "readout.weight", "query")
        for name, _ in graph.named_parameters()
    )
    item = {
        "channels": encode_scene(data["fit"][0]["public"], torch.float64),
        "gates": u1.make_gates(graph.units.n_units, torch.float64),
        "label": 0,
    }
    d = u1.cast_item(item, graph.spec.torch_dtype)
    base = graph(d["channels"], d["gates"], graph.initial_state())["readout"]
    assert base.shape == (4,)
    h_state = {**graph.initial_state(), "H": graph.initial_state()["H"] + 0.5}
    cut = graph(d["channels"], d["gates"], graph.cut_h(h_state))["readout"]
    assert torch.allclose(base, cut, rtol=0.0, atol=0.0)


def test_attention_is_question_conditioned(data: dict) -> None:
    graph = u1r1.build_graph(2026111601)
    item = {
        "channels": encode_scene(data["fit"][0]["public"], torch.float64),
        "gates": u1.make_gates(graph.units.n_units, torch.float64),
        "label": 0,
    }
    d = u1.cast_item(item, graph.spec.torch_dtype)
    a1 = graph(d["channels"], d["gates"], graph.initial_state())["attention"]
    fixed_q = torch.zeros_like(d["channels"]["question"])
    fixed_q[:, 1] = 1.0
    fixed_q[:, 6] = 1.0
    fixed_q[:, 7] = 1.0
    a2 = graph({**d["channels"], "question": fixed_q}, d["gates"], graph.initial_state())[
        "attention"
    ]
    assert a1.shape == (16,)
    assert torch.allclose(a1.sum(), torch.tensor(1.0, dtype=a1.dtype), atol=1e-6)
    assert not torch.allclose(a1, a2, rtol=0.0, atol=1e-8)


def test_neutral_birth_exact(data: dict) -> None:
    graph = u1r1.build_graph(2026111601)
    item = {
        "channels": encode_scene(data["fit"][0]["public"], torch.float64),
        "gates": u1.make_gates(graph.units.n_units, torch.float64),
        "label": 0,
    }
    result = u1r1.neutral_birth_check(graph, item)
    assert result["neutral_birth_exact"]
    assert result["neutral_birth_max_abs"] == 0.0
    assert result["new_unit_r_credit_path_wired"]


def test_intervention_mechanisms_wired(data: dict) -> None:
    graph = u1r1.build_graph(2026111601)
    episodes = u1.prepare_episodes(data, graph.units.n_units)
    item = episodes["fit"][0]
    hd = u1.cast_item(episodes["fit"][1], graph.spec.torch_dtype)
    hist_state = graph(hd["channels"], hd["gates"], graph.initial_state())["next_state"]
    probe = u1.mechanism_probe(graph)
    summary = u1.intervention_summary(u1r1.intervention_readouts(probe, item, hist_state))
    assert summary["without_edges_changes_readout"]
    assert summary["permute_source_changes_readout"]
    assert summary["history_changes_readout"]
    assert summary["cut_h_bit_identical"]
    assert summary["h_perturb_bit_identical"]


def test_gradient_coverage(data: dict) -> None:
    graph = u1r1.build_graph(2026111601)
    episodes = u1.prepare_episodes(data, graph.units.n_units)
    coverage = u1r1.gradient_coverage(graph, episodes["fit"][0])
    assert coverage["all_parameters_have_gradient"]
    assert coverage["named_nonzero"]


def test_criteria_frozen(contract: dict) -> None:
    criteria = contract["criteria"]
    assert criteria["fit_gate"] == 0.80
    assert criteria["train_gate"] == 0.90
    assert criteria["distinct_min"] == 3
    assert criteria["control_agreement_max"] == 0.5
    assert len(criteria["seeds"]) == 3
    assert value_digest(criteria) == contract["criteria_sha256"]
    assert contract["status"] == "frozen"
    assert contract["revision"] == 1
    assert contract["plan"]["cap_cpu"] <= contract["plan"]["design_cap_cpu"]
