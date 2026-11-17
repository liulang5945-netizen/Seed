"""Zero-training contract tests for 22.5.U.1 revision 2 (scale + attention supervision)."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest
import torch

from scripts.training import probe_u1r2_scale_focus_20261116 as sf
from scripts.training import run_u1_unit_read_20261116 as u1
from scripts.training import run_u1r1_position_20261116 as u1r1
from scripts.training import run_u1r2_scale_supervision_20261116 as u1r2
from taiji.u2_history import atomic_load, value_digest
from taiji.u2_neural_units import apply_capsule, load_capsule
from taiji.u2_unit_input import encoding_contract
from taiji.u2_unit_temperature_readout import TemperaturePositionGraph

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "plans/manifests/taiji_u2_u1r2_scale_supervision_20261116.json"
DATA = ROOT / "data/experiments/taiji_u2/first_a/20261112T074836Z/data.bin"
SEED = 2026111601
PIN_NAMES = {
    "runner",
    "temperature_readout",
    "position_readout",
    "u1_runner",
    "unit_input",
    "neural_units",
    "support",
    "columns",
    "episode",
    "scale_focus_probe",
    "focus_feasibility_probe",
    "value_side_probe",
}


@pytest.fixture(scope="module")
def contract() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def data() -> dict:
    return atomic_load(DATA)


@pytest.fixture(scope="module")
def episodes(data: dict) -> dict:
    return u1.prepare_episodes(data, 12)


def test_encoding_contract_frozen_unchanged(contract: dict) -> None:
    assert value_digest(encoding_contract()) == contract["encoding_contract_digest"]
    assert contract["encoding_contract_identity"] == "unit-input-column-color-v1"


def test_source_pins_match(contract: dict) -> None:
    paths = dict(u1r2.pinned_files())
    assert set(paths) == PIN_NAMES
    assert set(contract["source_pins"]) == PIN_NAMES
    for name, pin in contract["source_pins"].items():
        assert u1r2.sha_of(paths[name]) == pin, name


def test_chain_bit_identical_to_parent_at_zero(episodes: dict) -> None:
    anchor = u1r2.chain_anchor(SEED, episodes["fit"][0])
    assert anchor["bit_identical"]
    assert anchor["max_abs"] == 0.0
    assert anchor["parent_route"] == "B_position"
    assert anchor["child_route"] == "B_position_temp"
    assert all(anchor["per_tensor"].values())


def test_adds_exactly_one_scalar_parameter() -> None:
    torch.manual_seed(SEED)
    parent = u1r1.build_graph(SEED)
    child = u1r2.build_graph(SEED)
    added = set(child.state_dict()) - set(parent.state_dict())
    assert added == {"log_tau"}
    assert child.log_tau.numel() == 1
    assert float(child.log_tau.detach()) == 0.0
    assert len(list(child.named_parameters())) == len(list(parent.named_parameters())) + 1
    assert all(u1r2.b_identity_checks(child).values())


def test_r1_capsule_is_rejected_by_route() -> None:
    capsule = load_capsule(u1r1.initial_capsule_path(SEED))
    graph = u1r2.build_graph(SEED)
    with pytest.raises(ValueError, match="capsule_route_mismatch"):
        apply_capsule(graph, capsule)


def test_module_survives_deepcopy_after_forward(episodes: dict) -> None:
    graph = u1r2.build_graph(SEED)
    item = u1.cast_item(episodes["fit"][0], graph.spec.torch_dtype)
    graph(item["channels"], item["gates"], graph.initial_state())
    grown = graph.with_neutral_birth(0, ["u0000"])
    assert isinstance(grown, TemperaturePositionGraph)
    assert grown.units.n_units == graph.units.n_units + 1


def test_temperature_scalar_is_causally_wired(episodes: dict) -> None:
    graph = u1r2.build_graph(SEED)
    item = episodes["fit"][0]
    cold_attention, cold_logits = sf.capture_attention(graph, item)
    with torch.no_grad():
        graph.log_tau.fill_(math.log(100.0))
    hot_attention, hot_logits = sf.capture_attention(graph, item)
    assert torch.allclose(hot_logits, 100.0 * cold_logits, rtol=1e-9, atol=1e-12)
    assert float(hot_attention.max()) > float(cold_attention.max())


def test_supervision_credit_restricted_to_query(episodes: dict) -> None:
    graph = u1r2.build_graph(SEED)
    item = next(i for i in episodes["train"] if sf.labelled(i))
    rows = u1r2.supervision_gradient_rows(graph, item)
    assert rows["combined_matches_task_plus_restricted_focus"]
    assert rows["max_abs"] <= 1e-12
    assert rows["focus_parameters_not_in_task"] == []
    assert rows["focus_credit_off_query_total"] > 0.0
    assert set(rows["focus_query_credit"]) == set(u1r2.QUERY_PARAMS)
    assert all(value > 0.0 for value in rows["focus_query_credit"].values())


def test_supervision_face_is_train_labelled_only(episodes: dict) -> None:
    train_ids = {item["id"] for item in episodes["train"]}
    fit_ids = {item["id"] for item in episodes["fit"]}
    supervised = {item["id"] for item in episodes["train"] if sf.labelled(item)}
    assert supervised <= train_ids
    assert not (supervised & fit_ids)
    assert len(supervised) == 48
    assert len(train_ids) - len(supervised) == 16
    assert sum(1 for i in episodes["fit"] if sf.labelled(i)) == 24


def test_focus_ruler_has_dynamic_range(episodes: dict) -> None:
    graph = u1r2.build_graph(SEED)
    birth = sf.focus_stats(graph, episodes["fit"])
    oracle = sf.focus_stats(graph, episodes["fit"], "oracle_soft")
    assert not birth["focus_met"]
    assert oracle["focus_met"]
    assert birth["labelled_denominator"] == 24
    assert birth["denominator"] == 32


def test_parameter_groups_partition_every_parameter(contract: dict) -> None:
    graph = u1r2.build_graph(SEED)
    optimizer = u1r2.make_optimizer(graph, contract, SEED)
    seen = [id(p) for group in optimizer.param_groups for p in group["params"]]
    expected = [id(p) for _, p in graph.named_parameters()]
    assert len(seen) == len(set(seen))
    assert sorted(seen) == sorted(expected)
    assert len(optimizer.param_groups) == 3
    names = {id(p): name for name, p in graph.named_parameters()}
    rate_of = {
        names[id(p)]: group["lr"] for group in optimizer.param_groups for p in group["params"]
    }
    criteria = contract["criteria"]
    assert rate_of["W_q"] == criteria["query_lr"]["effective_lr"]
    assert rate_of["log_tau"] == criteria["logit_temperature"]["learning_rate_by_seed"][str(SEED)]
    assert rate_of["position_head.weight"] == criteria["learning_rate"]


def test_criteria_frozen(contract: dict) -> None:
    criteria = contract["criteria"]
    assert criteria["fit_gate"] == 0.80
    assert criteria["train_gate"] == 0.90
    assert criteria["distinct_min"] == 3
    assert criteria["control_agreement_max"] == 0.5
    assert criteria["intervention_change_min_fraction"] == 0.5
    assert len(criteria["seeds"]) == 3
    assert criteria["updates_per_seed"] == 384
    assert value_digest(criteria) == contract["criteria_sha256"]
    assert contract["revision"] == 2
    assert contract["revision_limit"] == 2
    assert contract["identity"] == "taiji-u2-unit-core-b-v1"
    assert criteria["focus_gate"]["answer_mass_min"] == sf.FOCUS_ANSWER_MASS_MIN
    assert criteria["focus_gate"]["effective_columns_max"] == sf.FOCUS_EFFECTIVE_COLUMNS_MAX
    assert contract["plan"]["cap_cpu"] <= contract["plan"]["design_cap_cpu"]


def test_exits_are_ordered_and_mutually_exclusive(contract: dict) -> None:
    exits = contract["criteria"]["exits"]
    assert [row["order"] for row in exits] == [1, 2, 3, 4, 5]
    verdicts = [row["verdict"] for row in exits]
    assert len(verdicts) == len(set(verdicts))
    assert verdicts == [
        "budget_stop",
        "engineering_failure",
        "focus_not_met",
        "focus_met_value_failed",
        "read_prerequisite_passed",
    ]


def test_run_mode_refuses_without_authority(contract: dict) -> None:
    draft = copy.deepcopy(contract)
    draft["status"] = "draft_pending_owner_ruling"
    draft["training_authority"] = None
    with pytest.raises(ValueError, match="contract_not_frozen"):
        u1r2.guards(draft, "run")
    draft["status"] = "frozen"
    with pytest.raises(ValueError, match="missing_training_authority"):
        u1r2.guards(draft, "run")


def test_frozen_contract_has_measured_caps(contract: dict) -> None:
    assert contract["status"] == "frozen"
    assert contract["plan"]["cap_cpu"] > 0
    assert contract["plan"]["cap_wall"] > 0
    assert contract["plan"]["price_report"]
    authority = contract["training_authority"]
    assert authority is None or (isinstance(authority, str) and authority.strip())


def test_actual_contract_run_gate(contract: dict) -> None:
    if contract["training_authority"]:
        u1r2.guards(contract, "run")
    else:
        with pytest.raises(ValueError, match="missing_training_authority"):
            u1r2.guards(contract, "run")


def test_guards_reject_tampered_pin(contract: dict) -> None:
    tampered = copy.deepcopy(contract)
    tampered["source_pins"]["temperature_readout"] = "0" * 64
    with pytest.raises(ValueError, match="source_pin_mismatch:temperature_readout"):
        u1r2.guards(tampered, "smoke")


def test_guards_reject_focus_gate_drift(contract: dict) -> None:
    drifted = copy.deepcopy(contract)
    drifted["criteria"]["focus_gate"]["answer_mass_min"] = 0.1
    with pytest.raises(ValueError, match="focus_answer_mass_gate_mismatch"):
        u1r2.guards(drifted, "smoke")


def test_guards_reject_credit_restriction_drift(contract: dict) -> None:
    drifted = copy.deepcopy(contract)
    drifted["criteria"]["supervision"]["credit_restriction"] = ["W_q"]
    with pytest.raises(ValueError, match="credit_restriction_mismatch"):
        u1r2.guards(drifted, "smoke")


def test_neutral_birth_and_interventions(episodes: dict) -> None:
    graph = u1r2.build_graph(SEED)
    item = episodes["fit"][0]
    birth = u1r1.neutral_birth_check(graph, item)
    assert birth["neutral_birth_exact"]
    assert birth["neutral_birth_max_abs"] == 0.0
    assert birth["new_unit_r_credit_path_wired"]
    hd = u1.cast_item(episodes["fit"][1], graph.spec.torch_dtype)
    with torch.no_grad():
        hist_state = graph(hd["channels"], hd["gates"], graph.initial_state())["next_state"]
    probe = u1.mechanism_probe(graph)
    summary = u1.intervention_summary(u1r1.intervention_readouts(probe, item, hist_state))
    assert summary["without_edges_changes_readout"]
    assert summary["permute_source_changes_readout"]
    assert summary["history_changes_readout"]
    assert summary["cut_h_bit_identical"]
    assert summary["h_perturb_bit_identical"]


def test_gradient_coverage_includes_temperature(episodes: dict) -> None:
    graph = u1r2.build_graph(SEED)
    coverage = u1r2.gradient_coverage(graph, episodes["fit"][0])
    assert coverage["all_parameters_have_gradient"]
    assert coverage["named_nonzero"]
    assert coverage["log_tau_nonzero"]
    assert coverage["log_tau_credit"] > 0.0


def test_encoding_oracle_still_faithful(data: dict) -> None:
    from taiji.u2_unit_input import color_mass, derive_label, scene_entity

    ok = 0
    total = 0
    for split in ("train", "fit"):
        for example in data[split]:
            mass = color_mass(example["public"]["observations"]["rgb"])
            got = derive_label(mass, scene_entity(example["public"]["request"]))
            total += 1
            ok += int(got == example["target"].decode("utf-8"))
    assert total == 96
    assert ok == total
