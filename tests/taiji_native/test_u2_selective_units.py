"""Contract tests for ``taiji/u2_selective_units.py``."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import torch
from torch import nn

from taiji.u2_neural_units import (
    LEGAL_INPUT_CHANNELS,
    RouteBGraph,
    UnitGraphSpec,
    build_capsule,
)
from taiji.u2_selective_units import (
    CAPSULE_V2_SCHEMA,
    SelectiveNeuralUnits,
    SelectivePersistentGraph,
    SelectiveUnitSpec,
    _AntitheticSelectiveCohort,
    apply_capsule_v2,
    build_capsule_v2,
    load_capsule_v2,
    save_capsule_v2,
)


def _sample_channels(
    batch_size: int, steps: int | None = None, seed: int = 11
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    shape = (batch_size, 8) if steps is None else (batch_size, steps, 8)
    return {
        name: torch.randn(*shape, generator=generator, dtype=torch.float32)
        for name in LEGAL_INPUT_CHANNELS
    }


def test_default_mode_is_bit_identical_to_base_graph() -> None:
    spec = UnitGraphSpec(
        groups=6, units_per_group=5, input_dim=8, hidden_dim=8, output_dim=4, seed=77
    )
    base_graph = RouteBGraph(spec)
    selective_graph = SelectivePersistentGraph(base_graph)
    assert not selective_graph.per_sample_state

    channels = _sample_channels(12, seed=91)
    gates = torch.ones(12, base_graph.units.n_units, dtype=torch.float32)
    base_out = base_graph(channels, gates, base_graph.initial_state())
    sel_out = selective_graph(channels, gates, selective_graph.initial_state())

    assert sel_out["path"] == base_out["path"]
    assert torch.equal(sel_out["readout"], base_out["readout"])
    assert torch.equal(sel_out["next_state"]["z"], base_out["next_state"]["z"])
    assert torch.equal(sel_out["next_state"]["a"], base_out["next_state"]["a"])


def test_per_sample_batch_isolation_eliminates_batch_as_time_bleed() -> None:
    base_graph = RouteBGraph(UnitGraphSpec(groups=6, units_per_group=5, seed=101))
    sel_spec = SelectiveUnitSpec(
        groups=6,
        units_per_group=5,
        per_sample_state=True,
        selective_gate=True,
        seed=101,
    )
    graph = SelectivePersistentGraph(base_graph, sel_spec)

    channels_a = _sample_channels(4, steps=6, seed=202)
    channels_b = {k: v.clone() for k, v in channels_a.items()}
    # Perturb only samples 1, 2, 3 strongly; sample 0 must remain bit-identical.
    for key in LEGAL_INPUT_CHANNELS:
        channels_b[key][1:] += 5.0

    gates = torch.ones(4, 6, sel_spec.n_units, dtype=torch.float32)
    out_a = graph(channels_a, gates, graph.initial_state(batch_size=4))
    out_b = graph(channels_b, gates, graph.initial_state(batch_size=4))

    assert out_a["state_final"].shape == (4, 2 * sel_spec.n_units)
    assert torch.equal(out_a["state_final"][0], out_b["state_final"][0])
    assert torch.equal(out_a["readout"][0], out_b["readout"][0])
    assert float((out_a["state_final"][1:] - out_b["state_final"][1:]).detach().abs().max()) > 1e-2


def test_negative_write_bias_preserves_gradient_across_20_ticks() -> None:
    def grad_norm_at_t0(init_bias: float) -> float:
        spec = SelectiveUnitSpec(
            groups=6,
            units_per_group=5,
            per_sample_state=True,
            selective_gate=True,
            init_write_bias=init_bias,
            seed=303,
        )
        units = SelectiveNeuralUnits(spec)
        generator = torch.Generator().manual_seed(777)
        inputs = torch.randn(
            8, 20, spec.groups, spec.hidden_dim, generator=generator, requires_grad=True
        )
        gates = torch.ones(8, 20, spec.n_units)
        result = units(inputs, gates, units.initial_state(batch_size=8))
        loss = result["z"].pow(2).sum()
        loss.backward()
        assert inputs.grad is not None
        return float(inputs.grad[:, 0].abs().mean())

    neg_bias_grad = grad_norm_at_t0(-2.0)
    zero_bias_grad = grad_norm_at_t0(0.0)
    assert neg_bias_grad > zero_bias_grad * 20.0


def test_capsule_v2_roundtrip_and_v1_compat(tmp_path: Path) -> None:
    base_spec = UnitGraphSpec(groups=6, units_per_group=4, output_dim=6, seed=404, dtype="float64")
    base_graph = RouteBGraph(base_spec)
    sel_spec = SelectiveUnitSpec(
        groups=6,
        units_per_group=4,
        output_dim=6,
        per_sample_state=True,
        selective_gate=True,
        init_write_bias=-2.5,
        seed=404,
        dtype="float64",
    )
    graph = SelectivePersistentGraph(base_graph, sel_spec)
    with torch.no_grad():
        graph.selective_units.write_bias.add_(0.25)

    capsule_v2 = build_capsule_v2(graph, clocks=(5, 2))
    assert capsule_v2["schema"] == CAPSULE_V2_SCHEMA
    saved_path = save_capsule_v2(capsule_v2, tmp_path / "selective_v2.pt")
    loaded = load_capsule_v2(saved_path)

    fresh = SelectivePersistentGraph(
        RouteBGraph(
            UnitGraphSpec(groups=6, units_per_group=4, output_dim=6, seed=999, dtype="float64")
        )
    )
    apply_capsule_v2(fresh, loaded)
    assert fresh.per_sample_state

    seq = {k: v.to(torch.float64) for k, v in _sample_channels(3, steps=5, seed=505).items()}
    gates = torch.ones(3, 5, sel_spec.n_units, dtype=torch.float64)
    out_orig = graph(seq, gates, graph.initial_state(batch_size=3))
    out_fresh = fresh(seq, gates, fresh.initial_state(batch_size=3))
    diff = float((out_orig["readout"] - out_fresh["readout"]).detach().abs().max())
    assert diff <= 1e-10

    # Also verify legacy v1 capsule restores cleanly with per_sample_state disabled.
    v1_capsule = build_capsule(base_graph, clocks=(3, 1))
    apply_capsule_v2(fresh, v1_capsule)
    assert not fresh.per_sample_state
    flat = {k: v.to(torch.float64) for k, v in _sample_channels(3, seed=606).items()}
    gates_v1 = torch.ones(3, base_graph.units.n_units, dtype=torch.float64)
    out_base = base_graph(flat, gates_v1, base_graph.initial_state())
    out_compat = fresh(flat, gates_v1, fresh.initial_state())
    assert float((out_base["readout"] - out_compat["readout"]).detach().abs().max()) <= 1e-10


def test_selective_read_gate_uses_only_final_tick_and_survives_capsule_v2(
    tmp_path: Path,
) -> None:
    base_spec = UnitGraphSpec(groups=6, units_per_group=5, output_dim=4, seed=707, dtype="float64")
    base_graph = RouteBGraph(base_spec)
    sel_spec = SelectiveUnitSpec(
        groups=6,
        units_per_group=5,
        output_dim=4,
        per_sample_state=True,
        selective_gate=True,
        seed=707,
        dtype="float64",
    )
    graph = SelectivePersistentGraph(base_graph, sel_spec)
    seq_base = {k: v.to(torch.float64) for k, v in _sample_channels(4, steps=6, seed=808).items()}
    gates = torch.ones(4, 6, sel_spec.n_units, dtype=torch.float64)
    out_base = graph(seq_base, gates, graph.initial_state(batch_size=4))

    # (a) Perturbing past ticks t=0..T-2 must leave read_gate bit-identical.
    seq_past = {k: v.clone() for k, v in seq_base.items()}
    for key in LEGAL_INPUT_CHANNELS:
        seq_past[key][:, :-1, :] += 3.5
    out_past = graph(seq_past, gates, graph.initial_state(batch_size=4))
    assert torch.equal(out_base["read_gate"], out_past["read_gate"])

    # (b) Perturbing final tick t=T-1 must change read_gate.
    seq_last = {k: v.clone() for k, v in seq_base.items()}
    for key in LEGAL_INPUT_CHANNELS:
        seq_last[key][:, -1, :] += 2.0
    out_last = graph(seq_last, gates, graph.initial_state(batch_size=4))
    assert float((out_base["read_gate"] - out_last["read_gate"]).detach().abs().max()) > 1e-2

    # (c) capsule_v2 roundtrip reproduces both read_gate and readout <= 1e-10.
    capsule = build_capsule_v2(graph, clocks=(2, 1))
    loaded = load_capsule_v2(save_capsule_v2(capsule, tmp_path / "read_gate_v2.pt"))
    rebuilt = SelectivePersistentGraph(
        RouteBGraph(
            UnitGraphSpec(groups=6, units_per_group=5, output_dim=4, seed=999, dtype="float64")
        )
    )
    apply_capsule_v2(rebuilt, loaded)
    out_rebuilt = rebuilt(seq_base, gates, rebuilt.initial_state(batch_size=4))
    assert float((out_base["read_gate"] - out_rebuilt["read_gate"]).detach().abs().max()) <= 1e-10
    assert float((out_base["readout"] - out_rebuilt["readout"]).detach().abs().max()) <= 1e-10


def test_grow_units_neutral_is_output_identical_and_receives_step1_gradient() -> None:
    base_graph = RouteBGraph(
        UnitGraphSpec(groups=4, units_per_group=1, output_dim=4, seed=901, dtype="float64")
    )
    sel_spec = SelectiveUnitSpec(
        groups=4,
        units_per_group=1,
        output_dim=4,
        readout_hidden=48,
        per_sample_state=True,
        selective_gate=True,
        seed=901,
        dtype="float64",
    )
    graph = SelectivePersistentGraph(base_graph, sel_spec)
    seq = {k: v.to(torch.float64) for k, v in _sample_channels(5, steps=6, seed=902).items()}
    gates_before = torch.ones(5, 6, graph.selective_units.n_units, dtype=torch.float64)
    out_before = graph(seq, gates_before, graph.initial_state(batch_size=5))

    new_ids = graph.grow_units_neutral_(2, seed=1001)
    assert new_ids == [4, 5, 6, 7, 8, 9, 10, 11]
    assert graph.growth_generation == 1
    assert graph.selective_units.n_units == 12
    assert torch.equal(
        graph.selective_units.unit_lifecycle[4:],
        torch.ones(8, dtype=torch.int64),
    )

    gates_after = torch.ones(5, 6, graph.selective_units.n_units, dtype=torch.float64)
    out_after = graph(seq, gates_after, graph.initial_state(batch_size=5))
    assert float((out_before["readout"] - out_after["readout"]).detach().abs().max()) <= 1e-10
    assert (
        float(
            (out_before["unit_result"]["z"] - out_after["unit_result"]["z"][:, :4])
            .detach()
            .abs()
            .max()
        )
        <= 1e-10
    )
    assert (
        float(
            (out_before["unit_result"]["a"] - out_after["unit_result"]["a"][:, :4])
            .detach()
            .abs()
            .max()
        )
        <= 1e-10
    )

    # Newborn weights must receive non-zero gradient on the very first backward pass.
    loss = out_after["readout"].pow(2).sum()
    loss.backward()
    ro_first = cast(nn.Linear, graph.selective_readout[0])
    ro_w_grad = ro_first.weight.grad
    assert ro_w_grad is not None
    assert float(ro_w_grad[:, 4:12].abs().max()) > 1e-8
    assert float(ro_w_grad[:, 16:24].abs().max()) > 1e-8


def test_lifecycle_dormant_blocks_grown_units_and_is_reversible() -> None:
    base_graph = RouteBGraph(
        UnitGraphSpec(groups=4, units_per_group=1, output_dim=4, seed=911, dtype="float64")
    )
    sel_spec = SelectiveUnitSpec(
        groups=4,
        units_per_group=1,
        output_dim=4,
        readout_hidden=32,
        per_sample_state=True,
        selective_gate=True,
        seed=911,
        dtype="float64",
    )
    graph = SelectivePersistentGraph(base_graph, sel_spec)
    seq = {k: v.to(torch.float64) for k, v in _sample_channels(4, steps=5, seed=912).items()}
    with torch.no_grad():
        out_pre = graph(
            seq, torch.ones(4, 5, 4, dtype=torch.float64), graph.initial_state(batch_size=4)
        )["readout"]

    new_ids = graph.grow_units_neutral_(3, seed=913)
    with torch.no_grad():
        cast(nn.Linear, graph.selective_readout[0]).weight[:, 4:16].fill_(0.5)
        out_active = graph(
            seq, torch.ones(4, 5, 16, dtype=torch.float64), graph.initial_state(batch_size=4)
        )["readout"]
    assert float((out_active - out_pre).abs().max()) > 1e-2

    graph.selective_units.set_unit_lifecycle_(new_ids, "dormant")
    with torch.no_grad():
        out_dormant = graph(
            seq, torch.ones(4, 5, 16, dtype=torch.float64), graph.initial_state(batch_size=4)
        )["readout"]
    assert float((out_dormant - out_pre).abs().max()) <= 1e-10

    graph.selective_units.set_unit_lifecycle_(new_ids, "mature")
    with torch.no_grad():
        out_restored = graph(
            seq, torch.ones(4, 5, 16, dtype=torch.float64), graph.initial_state(batch_size=4)
        )["readout"]
    assert float((out_restored - out_active).abs().max()) <= 1e-10


def test_capsule_v2_roundtrip_preserves_multi_gen_grown_graph(tmp_path: Path) -> None:
    base_graph = RouteBGraph(
        UnitGraphSpec(groups=4, units_per_group=1, output_dim=4, seed=921, dtype="float64")
    )
    sel_spec = SelectiveUnitSpec(
        groups=4,
        units_per_group=1,
        output_dim=4,
        readout_hidden=64,
        per_sample_state=True,
        selective_gate=True,
        seed=921,
        dtype="float64",
    )
    graph = SelectivePersistentGraph(base_graph, sel_spec)
    gen1_ids = graph.grow_units_neutral_(2, seed=922)
    with torch.no_grad():
        cast(nn.Linear, graph.selective_readout[0]).weight[:, 4:12].add_(0.15)
    graph.selective_units.promote_immature_to_mature_()
    gen2_ids = graph.grow_units_neutral_(1, seed=923)
    with torch.no_grad():
        cast(nn.Linear, graph.selective_readout[0]).weight[:, 12:16].sub_(0.10)
    assert len(gen1_ids) == 8 and len(gen2_ids) == 4
    assert graph.growth_generation == 2

    cap = build_capsule_v2(graph, clocks=(4, 2))
    loaded = load_capsule_v2(save_capsule_v2(cap, tmp_path / "grown_gen2.pt"))
    fresh = SelectivePersistentGraph(
        RouteBGraph(
            UnitGraphSpec(groups=4, units_per_group=1, output_dim=4, seed=999, dtype="float64")
        )
    )
    apply_capsule_v2(fresh, loaded)
    assert fresh.growth_generation == 2
    assert fresh.selective_units.n_units == 16
    assert torch.equal(fresh.selective_units.unit_generation, graph.selective_units.unit_generation)
    assert torch.equal(fresh.selective_units.unit_lifecycle, graph.selective_units.unit_lifecycle)

    seq = {k: v.to(torch.float64) for k, v in _sample_channels(4, steps=6, seed=924).items()}
    gates = torch.ones(4, 6, 16, dtype=torch.float64)
    with torch.no_grad():
        out_a = graph(seq, gates, graph.initial_state(batch_size=4))["readout"]
        out_b = fresh(seq, gates, fresh.initial_state(batch_size=4))["readout"]
    assert float((out_a - out_b).abs().max()) <= 1e-10


def test_antithetic_cohort_birth_neutral_step1_grad_and_dormant_lesion() -> None:
    base_graph = RouteBGraph(
        UnitGraphSpec(groups=1, units_per_group=4, output_dim=4, seed=931, dtype="float64")
    )
    sel_spec = SelectiveUnitSpec(
        groups=1,
        units_per_group=4,
        output_dim=4,
        readout_hidden=16,
        per_sample_state=True,
        selective_gate=True,
        seed=931,
        dtype="float64",
    )
    graph = SelectivePersistentGraph(base_graph, sel_spec)
    seq = {k: v.to(torch.float64) for k, v in _sample_channels(4, steps=6, seed=932).items()}
    gates = torch.ones(4, 6, 4, dtype=torch.float64)
    out_pre = graph(seq, gates, graph.initial_state(batch_size=4))["readout"].detach()

    c_idx = graph.grow_antithetic_cohort_(
        groups=2, units_per_group=3, readout_hidden=32, freeze_mature=True, seed=933
    )
    assert c_idx == 0
    assert graph.growth_generation == 1
    assert graph.total_selective_units == 16

    cohort = cast(_AntitheticSelectiveCohort, graph.growth_cohorts[0])
    assert cohort.freeze_anchor is True
    for nm, param in cohort.named_parameters():
        if nm.startswith("neg_"):
            assert param.requires_grad is False

    out_dict = graph(seq, gates, graph.initial_state(batch_size=4))
    out_birth = out_dict["readout"]
    assert len(out_dict["immature_cohort_logits"]) == 1
    assert float((out_birth - out_pre).detach().abs().max()) <= 1e-10

    # Step-1 gradient flows directly into newborn unit parameters (r and write_in.weight).
    loss = out_birth.pow(2).sum()
    loss.backward()
    assert cohort.pos_units.write_in.weight.grad is not None
    assert cohort.pos_units.r.grad is not None
    assert float(cohort.pos_units.write_in.weight.grad.abs().max()) > 1e-8
    assert float(cohort.pos_units.r.grad.abs().max()) > 1e-8

    # Mutate cohort readout -> promote immature -> mature -> dormant restores -> mature restores.
    with torch.no_grad():
        cast(nn.Linear, cohort.pos_readout[2]).weight.add_(0.25)
        out_active = graph(seq, gates, graph.initial_state(batch_size=4))["readout"]
    assert float((out_active - out_pre).abs().max()) > 1e-2

    graph.promote_cohorts_to_mature_()
    assert cohort.lifecycle == "mature"
    with torch.no_grad():
        promoted_dict = graph(seq, gates, graph.initial_state(batch_size=4))
    assert len(promoted_dict["immature_cohort_logits"]) == 0
    assert float((promoted_dict["readout"] - out_active).abs().max()) <= 1e-10

    graph.set_cohort_lifecycle_(0, "dormant")
    with torch.no_grad():
        out_dormant = graph(seq, gates, graph.initial_state(batch_size=4))["readout"]
    assert float((out_dormant - out_pre).abs().max()) <= 1e-10

    graph.set_cohort_lifecycle_(0, "mature")
    with torch.no_grad():
        out_restored = graph(seq, gates, graph.initial_state(batch_size=4))["readout"]
    assert float((out_restored - out_active).abs().max()) <= 1e-10


def test_capsule_v2_roundtrip_preserves_multi_gen_antithetic_cohorts(tmp_path: Path) -> None:
    base_graph = RouteBGraph(
        UnitGraphSpec(groups=1, units_per_group=4, output_dim=4, seed=941, dtype="float64")
    )
    sel_spec = SelectiveUnitSpec(
        groups=1,
        units_per_group=4,
        output_dim=4,
        readout_hidden=16,
        per_sample_state=True,
        selective_gate=True,
        seed=941,
        dtype="float64",
    )
    graph = SelectivePersistentGraph(base_graph, sel_spec)
    graph.grow_antithetic_cohort_(groups=2, units_per_group=3, readout_hidden=32, seed=942)
    with torch.no_grad():
        c0 = cast(_AntitheticSelectiveCohort, graph.growth_cohorts[0])
        cast(nn.Linear, c0.pos_readout[2]).weight.add_(0.20)
    graph.promote_cohorts_to_mature_()
    graph.grow_antithetic_cohort_(groups=2, units_per_group=4, readout_hidden=48, seed=943)
    with torch.no_grad():
        c1 = cast(_AntitheticSelectiveCohort, graph.growth_cohorts[1])
        cast(nn.Linear, c1.neg_readout[2]).weight.sub_(0.15)
    assert graph.growth_generation == 2
    assert graph.total_selective_units == 32

    cap = build_capsule_v2(graph, clocks=(6, 3))
    loaded = load_capsule_v2(save_capsule_v2(cap, tmp_path / "antithetic_gen2.pt"))
    fresh = SelectivePersistentGraph(
        RouteBGraph(
            UnitGraphSpec(groups=1, units_per_group=4, output_dim=4, seed=999, dtype="float64")
        )
    )
    apply_capsule_v2(fresh, loaded)
    assert fresh.growth_generation == 2
    assert fresh.total_selective_units == 32
    assert len(fresh.growth_cohorts) == 2
    assert fresh.growth_cohorts[0].lifecycle == "mature"
    assert fresh.growth_cohorts[1].lifecycle == "immature"

    seq = {k: v.to(torch.float64) for k, v in _sample_channels(4, steps=6, seed=944).items()}
    gates = torch.ones(4, 6, 4, dtype=torch.float64)
    with torch.no_grad():
        out_a = graph(seq, gates, graph.initial_state(batch_size=4))["readout"]
        out_b = fresh(seq, gates, fresh.initial_state(batch_size=4))["readout"]
    assert float((out_a - out_b).abs().max()) <= 1e-10


def test_step_mask_freezes_state_on_padding_ticks() -> None:
    base_graph = RouteBGraph(
        UnitGraphSpec(groups=1, units_per_group=4, output_dim=4, seed=951, dtype="float64")
    )
    sel_spec = SelectiveUnitSpec(
        groups=1,
        units_per_group=4,
        output_dim=4,
        readout_hidden=16,
        per_sample_state=True,
        selective_gate=True,
        seed=951,
        dtype="float64",
    )
    graph = SelectivePersistentGraph(base_graph, sel_spec)
    graph.grow_antithetic_cohort_(groups=2, units_per_group=2, readout_hidden=16, seed=952)
    with torch.no_grad():
        c0 = cast(_AntitheticSelectiveCohort, graph.growth_cohorts[0])
        cast(nn.Linear, c0.pos_readout[2]).weight.add_(0.25)

    seq5 = {k: v.to(torch.float64) for k, v in _sample_channels(4, steps=5, seed=953).items()}
    gates5 = torch.ones(4, 5, 4, dtype=torch.float64)
    ones5 = torch.ones(4, 5, dtype=torch.float64)

    noise3 = {k: v.to(torch.float64) for k, v in _sample_channels(4, steps=3, seed=954).items()}
    seq8 = {k: torch.cat([seq5[k], noise3[k]], dim=1) for k in LEGAL_INPUT_CHANNELS}
    for k in LEGAL_INPUT_CHANNELS:
        seq8[k][:, -1, :] = seq5[k][:, -1, :]
    gates8 = torch.ones(4, 8, 4, dtype=torch.float64)
    mask8 = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float64).expand(
        4, 8
    )

    with torch.no_grad():
        out_none = graph(seq5, gates5, graph.initial_state(batch_size=4))
        out_ones = graph(seq5, gates5, graph.initial_state(batch_size=4), step_mask=ones5)
        out_pad = graph(seq8, gates8, graph.initial_state(batch_size=4), step_mask=mask8)

    assert float((out_none["readout"] - out_ones["readout"]).abs().max()) <= 1e-12
    assert float((out_none["unit_result"]["z"] - out_ones["unit_result"]["z"]).abs().max()) <= 1e-12
    assert float((out_none["unit_result"]["z"] - out_pad["unit_result"]["z"]).abs().max()) <= 1e-12
    assert float((out_none["unit_result"]["a"] - out_pad["unit_result"]["a"]).abs().max()) <= 1e-12
    assert float((out_pad["unit_result"]["write_trace"][:, 5:, :]).abs().max()) <= 1e-12

    bad_mask = torch.ones(4, 3, dtype=torch.float64)
    try:
        graph(seq5, gates5, graph.initial_state(batch_size=4), step_mask=bad_mask)
    except ValueError as exc:
        assert "selective_step_mask_shape_mismatch" in str(exc)
    else:
        raise AssertionError("expected ValueError for invalid step_mask shape")
