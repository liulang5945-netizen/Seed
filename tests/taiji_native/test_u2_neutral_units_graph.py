"""Contracts for the P3.9 A/B reference graphs and their full capsule."""

from __future__ import annotations

import copy

import pytest
import torch

from taiji.u2_neural_units import (
    C_CANDIDATES,
    COMPAT_FP32_ATOL,
    COMPAT_FP64_ATOL,
    LEGAL_INPUT_CHANNELS,
    NeuralUnits,
    RouteAGraph,
    RouteBGraph,
    UnitGraphSpec,
    apply_capsule,
    build_capsule,
    build_route,
    capsule_summary,
    legacy_rp_reference,
    load_capsule,
    save_capsule,
    validate_capsule,
)


def channels(
    dtype: torch.dtype, steps: int = 4, dim: int = 8, seed: int = 7
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        name: ((torch.rand(steps, dim, generator=generator, dtype=torch.float64) * 2 - 1) * 0.5).to(
            dtype
        )
        for name in LEGAL_INPUT_CHANNELS
    }


def gates(dtype: torch.dtype, steps: int, units: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(31)
    return torch.rand(steps, units, generator=generator, dtype=torch.float64).to(dtype)


@pytest.mark.parametrize(
    "dtype_name,atol", [("float64", COMPAT_FP64_ATOL), ("float32", COMPAT_FP32_ATOL)]
)
def test_compatible_endpoint_matches_legacy_rp(dtype_name, atol):
    spec = UnitGraphSpec(dtype=dtype_name)
    graph = RouteAGraph(spec)
    data = channels(spec.torch_dtype)
    gate = gates(spec.torch_dtype, 4, spec.n_units)
    with torch.no_grad():
        graph.units.c.zero_()
        graph.units.b.zero_()
        graph.units.w.zero_()
    state = graph.initial_state()
    got = graph.units(graph.encode_legal_inputs(data), graph.units.active_gate(gate), state)
    expected = legacy_rp_reference(graph, data, gate)
    assert torch.allclose(got["contribution"], expected, rtol=0.0, atol=atol)
    assert not torch.allclose(got["contribution"], torch.zeros_like(expected), rtol=0.0, atol=atol)
    # r/p gradients of the compatible endpoint must also align.
    graph.zero_grad(set_to_none=True)
    data_a = {key: value.clone().requires_grad_(True) for key, value in data.items()}
    with torch.no_grad():
        graph.units.c.zero_()
        graph.units.w.zero_()
    out = graph.units(graph.encode_legal_inputs(data_a), graph.units.active_gate(gate), state)
    (out["contribution"] * expected.detach()).sum().backward()
    assert graph.units.r.grad is not None and torch.isfinite(graph.units.r.grad).all()
    assert graph.units.p.grad is not None and torch.isfinite(graph.units.p.grad).all()


@pytest.mark.parametrize("route", ["A", "B"])
@pytest.mark.parametrize(
    "dtype_name,atol", [("float64", COMPAT_FP64_ATOL), ("float32", COMPAT_FP32_ATOL)]
)
def test_neutral_birth_is_exact_for_both_routes(route, dtype_name, atol):
    spec = UnitGraphSpec(dtype=dtype_name)
    graph = build_route(route, spec)
    data = channels(spec.torch_dtype)
    gate = gates(spec.torch_dtype, 4, spec.n_units)
    state = graph.initial_state()
    before = graph(data, gate, state)["readout"]
    grown = graph.with_neutral_birth(0, ["u0000"])
    # Same gates for the old units, exact zero gate for the newborn unit.
    big_gate = torch.cat([gate, torch.zeros(4, 1, dtype=spec.torch_dtype)], dim=1)
    big_state = graph.extend_state(state)
    after = grown(data, big_gate, big_state)["readout"]
    assert grown.units.n_units == spec.n_units + 1
    assert torch.allclose(before, after, rtol=0.0, atol=atol)
    new_unit = grown.units.identities["units"][-1]
    assert new_unit["lifecycle"] == "immature" and new_unit["parent_ids"] == ["u0000"]
    assert float(grown.units.c[-1].detach()) == 0.0 and float(grown.units.b[-1].detach()) == 0.0
    assert float(grown.units.eta[-1].detach()) == pytest.approx(0.1)
    assert torch.count_nonzero(grown.units.r[-1]) == 0
    assert torch.count_nonzero(grown.units.p[:, -1]) == 0


def test_route_b_readout_bypasses_h_while_route_a_does_not():
    spec = UnitGraphSpec(dtype="float64")
    data = channels(spec.torch_dtype)
    gate = gates(spec.torch_dtype, 4, spec.n_units)
    a, b = RouteAGraph(spec), RouteBGraph(spec)
    state_a, state_b = a.initial_state(), b.initial_state()
    out_a = a(data, gate, state_a)["readout"]
    # Non-trivial H: zeroing it must not move B's readout at all.
    h_state = {**state_b, "H": state_b["H"] + 0.3}
    out_b = b(data, gate, h_state)["readout"]
    assert not any(
        name.startswith("h_readout") or name == "ell" for name, _ in b.named_parameters()
    )
    assert torch.allclose(out_b, b(data, gate, b.cut_h(h_state))["readout"], rtol=0.0, atol=0.0)
    assert torch.count_nonzero(h_state["H"]) > 0
    perturbed = {**state_a, "H": state_a["H"] + 0.3}
    out_a_perturbed = a(data, gate, perturbed)["readout"]
    assert not torch.allclose(out_a, out_a_perturbed, rtol=0.0, atol=1e-8)
    # Cutting unit activity changes B's direct readout (green/red opposition).
    out_b_cut = b(data, torch.zeros_like(gate), state_b)["readout"]
    assert not torch.allclose(out_b, out_b_cut, rtol=0.0, atol=1e-8)


def test_missing_legal_input_fails_closed_and_candidates_are_frozen():
    graph = build_route("B", UnitGraphSpec(dtype="float64"))
    data = channels(graph.spec.torch_dtype)
    with pytest.raises(ValueError, match="missing_legal_input"):
        graph.encode_legal_inputs({key: value for key, value in data.items() if key != "question"})
    assert C_CANDIDATES == (0.0, 0.5, 0.9)
    assert len(LEGAL_INPUT_CHANNELS) == graph.spec.groups


def test_projection_keeps_exact_endpoints_and_edge_mask():
    spec = UnitGraphSpec(dtype="float64")
    units = NeuralUnits(spec)
    with torch.no_grad():
        units.c.fill_(5.0)
        units.eta.fill_(-3.0)
        units.b.fill_(9.0)
        units.w.fill_(7.0)
    units.project_parameters_()
    assert float(units.c.detach().max()) <= 1.0 - 1e-4
    assert float(units.eta.detach().min()) >= 1e-4
    assert float(units.b.detach().max()) <= 2.0
    assert torch.count_nonzero(units.w * (1 - units.edge_mask)) == 0
    with torch.no_grad():
        units.c.zero_()
    units.project_parameters_()
    assert torch.count_nonzero(units.c) == 0


@pytest.mark.parametrize("route", ["A", "B"])
def test_per_parameter_credit_and_structural_zero_gradients_are_reported(route):
    spec = UnitGraphSpec(dtype="float64")
    graph = build_route(route, spec)
    data = channels(spec.torch_dtype)
    gate = gates(spec.torch_dtype, 4, spec.n_units)
    graph.zero_grad(set_to_none=True)
    loss = graph(data, gate, graph.initial_state())["readout"].square().mean()
    loss.backward()
    credit = {
        name: float(param.grad.abs().sum())
        for name, param in graph.named_parameters()
        if param.grad is not None
    }
    assert len(credit) == len(list(graph.named_parameters()))
    for required in ("units.r", "units.p", "units.c", "input_projection"):
        assert credit[required] > 0.0, (route, required, credit[required])
    structural_zero = sorted(name for name, value in credit.items() if value == 0.0)
    assert len(structural_zero) < len(credit) - 3
    # Cutting the unit path must remove unit-parameter credit (red direction).
    graph.zero_grad(set_to_none=True)
    if route == "B":
        with torch.no_grad():
            graph.state_mix.zero_()
        cut_gate = gate
    else:
        cut_gate = torch.zeros_like(gate)
    graph(data, cut_gate, graph.initial_state())["readout"].square().mean().backward()
    assert graph.units.r.grad is not None and float(graph.units.r.grad.abs().sum()) == 0.0


def test_state_isolation_between_routes():
    spec = UnitGraphSpec(dtype="float64")
    a, b = RouteAGraph(spec), RouteBGraph(spec)
    data = channels(spec.torch_dtype)
    gate = gates(spec.torch_dtype, 4, spec.n_units)
    state_a, state_b = a.initial_state(), b.initial_state()
    before = b(data, gate, state_b)["readout"]
    with torch.no_grad():
        state_a["z"].add_(0.7)
    a_next = a(data, gate, state_a)["next_state"]
    after = b(data, gate, state_b)["readout"]
    assert torch.allclose(before, after, rtol=0.0, atol=0.0)
    assert float(state_b["z"].abs().sum()) == 0.0
    assert float(a_next["z"].abs().sum()) > 0.0
    assert float(b(data, gate, state_b)["next_state"]["z"].abs().sum()) > 0.0


def test_capsule_roundtrip_lifecycle_and_tamper_rejection(tmp_path):
    spec = UnitGraphSpec(dtype="float64")
    graph = build_route("B", spec)
    grown = graph.with_neutral_birth(0, ["u0000"])
    grown.units.set_lifecycle(grown.units.identities["units"][-1]["unit_id"], "mature")
    grown.units.set_lifecycle("u0001", "dormant")
    grown.units.set_lifecycle("u0001", "mature")
    capsule = build_capsule(grown, clocks=(3, 2))
    summary = capsule_summary(capsule)
    assert summary["units"] == spec.n_units + 1
    path = tmp_path / "capsule.bin"
    save_capsule(path, capsule)
    restored = load_capsule(path)
    assert capsule_summary(restored) == summary
    rebuilt = build_route("B", spec)
    apply_capsule(rebuilt, restored)
    assert rebuilt.units.n_units == spec.n_units + 1
    data = channels(spec.torch_dtype)
    gate = gates(spec.torch_dtype, 4, rebuilt.units.n_units)
    assert torch.isfinite(rebuilt(data, gate, rebuilt.initial_state())["readout"]).all()
    tampered = copy.deepcopy(capsule)
    tampered["runtime"]["unit_registry"][0]["lifecycle"] = "retired"
    with pytest.raises(ValueError, match="capsule_digest_mismatch"):
        validate_capsule(tampered)
    broken_clock = copy.deepcopy(capsule)
    broken_clock["learning_state"]["parameter_version"] = 9
    broken_clock["capsule_digest"] = "0" * 64
    with pytest.raises(ValueError, match="capsule_digest_mismatch"):
        validate_capsule(broken_clock)
