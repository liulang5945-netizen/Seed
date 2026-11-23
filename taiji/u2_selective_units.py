"""Additive per-sample selective persistent state and cross-group unit routing.

Keeps ``taiji/u2_neural_units.py`` completely untouched so all pinned contracts
(22.7, v1, w11, b_parent_reference, 22.5.1 check2) stay byte-for-byte invariant.

Default behavior (``per_sample_state=False``) delegates directly to the wrapped
``RouteBGraph`` / ``MultiHeadGraph`` with zero arithmetic divergence.  When
``per_sample_state=True`` is enabled, batch elements keep isolated ``[B, N]``
state across explicit time axes ``[B, T, ...]``, units gain cross-channel input
mixing, cross-group recurrent edges, and keep-first input-dependent selective
write gates (negative write-bias init) so gradients survive long horizons.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from taiji.u2_history import atomic_load, atomic_save, value_digest
from taiji.u2_multihead import MultiHeadGraph
from taiji.u2_neural_units import (
    B_MAX,
    C_MAX,
    CAPSULE_SCHEMA,
    ETA_MAX,
    ETA_MIN,
    LEGAL_INPUT_CHANNELS,
    W_MAX,
    RouteBGraph,
    apply_capsule,
    build_capsule,
    validate_capsule,
)

CAPSULE_V2_SCHEMA = "taiji-u2-selective-capsule-v2"
SELECTIVE_LIFECYCLE = ("proposed", "immature", "mature", "dormant", "retired")
_LIFECYCLE_TO_ID = {name: idx for idx, name in enumerate(SELECTIVE_LIFECYCLE)}


@dataclass(frozen=True)
class SelectiveUnitSpec:
    """Configuration for additive selective persistent state modules."""

    groups: int = 6
    units_per_group: int = 5
    input_dim: int = 8
    hidden_dim: int = 8
    output_dim: int = 4
    query_slots: int = 4
    per_sample_state: bool = False
    selective_gate: bool = True
    init_write_bias: float = -2.0
    cross_channel_mixing: bool = True
    cross_group_edges: bool = True
    readout_hidden: int = 0
    seed: int = 20261121
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.groups < 1 or self.units_per_group < 1:
            raise ValueError("SelectiveUnitSpec groups and units_per_group must be positive")
        if self.input_dim < 1 or self.hidden_dim < 1 or self.output_dim < 1:
            raise ValueError("SelectiveUnitSpec dimensions must be positive")
        if self.readout_hidden < 0:
            raise ValueError("SelectiveUnitSpec readout_hidden must be non-negative")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError(f"Unsupported dtype: {self.dtype!r}")

    @property
    def n_units(self) -> int:
        return self.groups * self.units_per_group

    @property
    def torch_dtype(self) -> torch.dtype:
        return torch.float64 if self.dtype == "float64" else torch.float32


class SelectiveNeuralUnits(nn.Module):
    """Per-sample `[B, T, ...]` unit dynamics with optional keep-first selective gating."""

    group_index: torch.Tensor
    group_mask: torch.Tensor
    edge_mask: torch.Tensor
    unit_generation: torch.Tensor
    unit_lifecycle: torch.Tensor

    def __init__(self, spec: SelectiveUnitSpec) -> None:
        super().__init__()
        self.spec = spec
        n_units = spec.n_units
        dtype = spec.torch_dtype
        generator = torch.Generator().manual_seed(spec.seed)

        group_idx = torch.arange(n_units, dtype=torch.int64) // spec.units_per_group
        group_mask = (
            group_idx.unsqueeze(0) == torch.arange(spec.groups, dtype=torch.int64).unsqueeze(1)
        ).to(dtype)
        if spec.cross_group_edges:
            if n_units == 1:
                edge_mask = torch.zeros(1, 1, dtype=dtype)
            else:
                edge_mask = (1.0 - torch.eye(n_units, dtype=dtype)).to(dtype)
        else:
            same_group = (group_idx.unsqueeze(1) == group_idx.unsqueeze(0)).to(dtype)
            edge_mask = same_group * (1.0 - torch.eye(n_units, dtype=dtype))

        self.register_buffer("group_index", group_idx)
        self.register_buffer("group_mask", group_mask)
        self.register_buffer("edge_mask", edge_mask)
        self.register_buffer("unit_generation", torch.zeros(n_units, dtype=torch.int64))
        self.register_buffer(
            "unit_lifecycle",
            torch.full((n_units,), _LIFECYCLE_TO_ID["mature"], dtype=torch.int64),
        )

        r_raw = torch.randn(n_units, spec.hidden_dim, generator=generator, dtype=dtype)
        r_norm = r_raw / r_raw.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        p_raw = torch.randn(spec.hidden_dim, n_units, generator=generator, dtype=dtype)
        p_norm = p_raw / p_raw.norm(dim=0, keepdim=True).clamp(min=1e-6)
        w_scale = 0.15 / max(1.0, n_units**0.5)
        w_raw = (
            torch.randn(n_units, n_units, generator=generator, dtype=dtype) * w_scale * edge_mask
        )

        self.r = nn.Parameter(r_norm)
        self.p = nn.Parameter(p_norm)
        self.c = nn.Parameter(torch.full((n_units,), 0.85, dtype=dtype))
        self.eta = nn.Parameter(torch.full((n_units,), 0.25, dtype=dtype))
        self.b = nn.Parameter(torch.full((n_units,), 0.1, dtype=dtype))
        self.w = nn.Parameter(w_raw)

        ctx_dim = spec.groups * spec.hidden_dim
        self.write_in = nn.Linear(ctx_dim, n_units, bias=False, dtype=dtype)
        self.extra_proj: nn.Linear | None = None
        self.write_state = nn.Linear(n_units, n_units, bias=False, dtype=dtype)
        self.write_bias = nn.Parameter(torch.full((n_units,), spec.init_write_bias, dtype=dtype))
        with torch.no_grad():
            self.write_in.weight.copy_(
                torch.randn(n_units, ctx_dim, generator=generator, dtype=dtype) * 0.05
            )
            self.write_state.weight.copy_(
                torch.randn(n_units, n_units, generator=generator, dtype=dtype) * 0.02
            )

    @property
    def n_units(self) -> int:
        return self.spec.n_units

    def initial_state(
        self, batch_size: int, device: torch.device | None = None
    ) -> dict[str, torch.Tensor]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        dtype = self.spec.torch_dtype
        target_device = self.r.device if device is None else device
        return {
            "z": torch.zeros(batch_size, self.n_units, dtype=dtype, device=target_device),
            "a": torch.zeros(batch_size, self.n_units, dtype=dtype, device=target_device),
        }

    def project_parameters_(self) -> None:
        with torch.no_grad():
            self.c.clamp_(0.0, C_MAX)
            self.eta.clamp_(ETA_MIN, ETA_MAX)
            self.b.clamp_(0.0, B_MAX)
            self.w.mul_(self.edge_mask).clamp_(-W_MAX, W_MAX)

    def attach_extra_(self, extra_dim: int, *, seed: int | None = None) -> None:
        """Attach a learnable raw-input adapter feeding the write gate (zero-init no-op)."""
        if extra_dim <= 0:
            return
        proj = nn.Linear(extra_dim, self.n_units, bias=False, dtype=self.write_in.weight.dtype)
        with torch.no_grad():
            proj.weight.zero_()
        self.extra_proj = proj

    def set_unit_lifecycle_(self, indices: list[int] | tuple[int, ...] | range, state: str) -> None:
        """Assign named lifecycle phase to selected units (masks output & recurrence when inactive)."""
        if state not in _LIFECYCLE_TO_ID:
            raise ValueError(f"Unknown lifecycle phase: {state!r}")
        phase_id = _LIFECYCLE_TO_ID[state]
        idx = torch.tensor(list(indices), dtype=torch.int64, device=self.unit_lifecycle.device)
        if idx.numel() == 0:
            return
        if int(idx.min()) < 0 or int(idx.max()) >= self.n_units:
            raise ValueError("unit_lifecycle_index_out_of_range")
        self.unit_lifecycle[idx] = phase_id

    def promote_immature_to_mature_(self) -> None:
        """Promote all currently immature units to mature."""
        immature_id = _LIFECYCLE_TO_ID["immature"]
        mature_id = _LIFECYCLE_TO_ID["mature"]
        self.unit_lifecycle[self.unit_lifecycle == immature_id] = mature_id

    def _normalize_gates(self, gates: torch.Tensor, batch_size: int, steps: int) -> torch.Tensor:
        if gates.ndim == 2 and gates.shape == (batch_size, self.n_units):
            return gates.unsqueeze(1).expand(batch_size, steps, self.n_units)
        if gates.ndim == 3 and gates.shape == (batch_size, steps, self.n_units):
            return gates
        raise ValueError("selective_gate_shape_mismatch")

    def forward(
        self,
        inputs: torch.Tensor,
        gates: torch.Tensor,
        state: dict[str, torch.Tensor],
        *,
        step_mask: torch.Tensor | None = None,
        extra: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Step selective units over explicit ``inputs [B, T, G, D]`` and ``state [B, N]``."""
        if inputs.ndim != 4:
            raise ValueError("selective_inputs_must_be_4d_BTGD")
        batch_size, steps, groups, hidden_dim = inputs.shape
        if groups != self.spec.groups or hidden_dim != self.spec.hidden_dim:
            raise ValueError("selective_inputs_group_or_dim_mismatch")
        z_prev = state["z"]
        a_prev = state["a"]
        if (
            z_prev.ndim != 2
            or a_prev.ndim != 2
            or z_prev.shape != (batch_size, self.n_units)
            or a_prev.shape != (batch_size, self.n_units)
        ):
            raise ValueError("selective_state_must_be_BN_per_sample")

        mask_bt: torch.Tensor | None = None
        if step_mask is not None:
            if step_mask.ndim != 2 or step_mask.shape != (batch_size, steps):
                raise ValueError("selective_step_mask_shape_mismatch")
            mask_bt = step_mask.to(dtype=inputs.dtype, device=inputs.device)

        gates_bt = self._normalize_gates(gates, batch_size, steps)
        active_mask = (
            (
                (self.unit_lifecycle == _LIFECYCLE_TO_ID["immature"])
                | (self.unit_lifecycle == _LIFECYCLE_TO_ID["mature"])
            )
            .to(gates_bt.dtype)
            .view(1, -1)
        )
        # x: [B, T, N, D]
        x = inputs[:, :, self.group_index, :]
        drive = (x * self.r.view(1, 1, self.n_units, self.spec.hidden_dim)).sum(dim=-1)
        weights = self.w * self.edge_mask
        inputs_flat = inputs.reshape(batch_size, steps, groups * hidden_dim)

        z_trace, a_trace, activity, write_trace = [], [], [], []
        z = z_prev * active_mask
        a = a_prev * active_mask
        for tick in range(steps):
            gate = gates_bt[:, tick, :] * active_mask
            incoming = (gate * z) @ weights.T
            v = drive[:, tick, :] + incoming - self.b.unsqueeze(0) * a
            candidate = torch.tanh(v)
            if self.spec.selective_gate:
                write_logits = (
                    self.write_in(inputs_flat[:, tick, :])
                    + self.write_state(z)
                    + self.write_bias.unsqueeze(0)
                )
                if self.extra_proj is not None and extra is not None:
                    write_logits = write_logits + self.extra_proj(extra[:, tick, :])
                write_g = torch.sigmoid(write_logits)
                z_cand = ((1.0 - write_g) * z + write_g * candidate) * active_mask
            else:
                c = self.c.unsqueeze(0)
                write_g = 1.0 - c.expand(batch_size, self.n_units)
                z_cand = (c * z + write_g * candidate) * active_mask
            eta = self.eta.unsqueeze(0)
            a_cand = ((1.0 - eta) * a + eta * z_cand) * active_mask
            if mask_bt is not None:
                m = mask_bt[:, tick : tick + 1]
                z = (m * z_cand + (1.0 - m) * z) * active_mask
                a = (m * a_cand + (1.0 - m) * a) * active_mask
                write_recorded = write_g * active_mask * m
            else:
                z = z_cand
                a = a_cand
                write_recorded = write_g * active_mask
            unit_activity = gate * z
            activity.append(unit_activity)
            z_trace.append(z)
            a_trace.append(a)
            write_trace.append(write_recorded)

        stacked_activity = torch.stack(activity, dim=1)
        return {
            "activity": stacked_activity,
            "z": z,
            "a": a,
            "z_trace": torch.stack(z_trace, dim=1),
            "a_trace": torch.stack(a_trace, dim=1),
            "write_trace": torch.stack(write_trace, dim=1),
        }


class _AntitheticSelectiveCohort(nn.Module):
    """Twin newborn unit cohort emitting ``pos - neg`` (exact zero birth delta, full gradient)."""

    def __init__(
        self,
        cohort_spec: SelectiveUnitSpec,
        *,
        generation: int,
        freeze_anchor: bool = True,
    ) -> None:
        super().__init__()
        self.cohort_spec = cohort_spec
        self.generation = generation
        self.freeze_anchor = freeze_anchor
        self.lifecycle = "immature"

        dtype = cohort_spec.torch_dtype
        raw_dim = len(LEGAL_INPUT_CHANNELS) * cohort_spec.input_dim
        ctx_dim = cohort_spec.groups * cohort_spec.hidden_dim
        state_dim = 2 * cohort_spec.n_units
        ro_hidden = (
            cohort_spec.readout_hidden if cohort_spec.readout_hidden > 0 else max(32, state_dim)
        )

        self.pos_units = SelectiveNeuralUnits(cohort_spec)
        self.neg_units = SelectiveNeuralUnits(cohort_spec)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cohort_spec.seed)
            self.pos_mix = nn.Linear(raw_dim, ctx_dim, bias=True, dtype=dtype)
            self.neg_mix = nn.Linear(raw_dim, ctx_dim, bias=True, dtype=dtype)
            self.pos_read_gate = nn.Linear(raw_dim, state_dim, bias=True, dtype=dtype)
            self.neg_read_gate = nn.Linear(raw_dim, state_dim, bias=True, dtype=dtype)
            nn.init.constant_(self.pos_read_gate.bias, 1.0)
            nn.init.constant_(self.neg_read_gate.bias, 1.0)
            self.pos_readout = nn.Sequential(
                nn.Linear(state_dim, ro_hidden, dtype=dtype),
                nn.Tanh(),
                nn.Linear(ro_hidden, cohort_spec.output_dim, dtype=dtype),
            )
            self.neg_readout = nn.Sequential(
                nn.Linear(state_dim, ro_hidden, dtype=dtype),
                nn.Tanh(),
                nn.Linear(ro_hidden, cohort_spec.output_dim, dtype=dtype),
            )

        # Exact twin identity at birth => (pos - neg) == 0.0, d/dpos != 0.
        self.neg_units.load_state_dict(self.pos_units.state_dict())
        self.neg_mix.load_state_dict(self.pos_mix.state_dict())
        self.neg_read_gate.load_state_dict(self.pos_read_gate.state_dict())
        self.neg_readout.load_state_dict(self.pos_readout.state_dict())
        if self.freeze_anchor:
            for mod in (self.neg_units, self.neg_mix, self.neg_read_gate, self.neg_readout):
                for param in mod.parameters():
                    param.requires_grad_(False)
                    param.grad = None
        self.set_lifecycle_("immature")

    def set_lifecycle_(self, state: str) -> None:
        if state not in _LIFECYCLE_TO_ID:
            raise ValueError(f"Unknown lifecycle phase: {state!r}")
        self.lifecycle = state
        indices = range(self.cohort_spec.n_units)
        self.pos_units.set_unit_lifecycle_(indices, state)
        self.neg_units.set_unit_lifecycle_(indices, state)

    def project_parameters_(self) -> None:
        self.pos_units.project_parameters_()
        self.neg_units.project_parameters_()

    def _run_twin(
        self,
        raw_seq: torch.Tensor,
        raw_final: torch.Tensor,
        units: SelectiveNeuralUnits,
        mix: nn.Linear,
        read_gate: nn.Linear,
        readout: nn.Sequential,
        *,
        step_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, steps = raw_seq.shape[0], raw_seq.shape[1]
        enc = mix(raw_seq).reshape(
            batch_size, steps, self.cohort_spec.groups, self.cohort_spec.hidden_dim
        )
        gates = torch.ones(
            batch_size, steps, self.cohort_spec.n_units, dtype=raw_seq.dtype, device=raw_seq.device
        )
        res = units(
            enc,
            gates,
            units.initial_state(batch_size, device=raw_seq.device),
            step_mask=step_mask,
        )
        state_final = torch.cat([res["z"], res["a"]], dim=-1)
        read_g = torch.sigmoid(read_gate(raw_final))
        return cast(torch.Tensor, readout(state_final * read_g))

    def forward_parts(
        self,
        raw_seq: torch.Tensor,
        raw_final: torch.Tensor,
        *,
        step_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.lifecycle not in {"immature", "mature"}:
            zeros = torch.zeros(
                raw_seq.shape[0],
                self.cohort_spec.output_dim,
                dtype=raw_seq.dtype,
                device=raw_seq.device,
            )
            return zeros, zeros
        pos_out = self._run_twin(
            raw_seq,
            raw_final,
            self.pos_units,
            self.pos_mix,
            self.pos_read_gate,
            self.pos_readout,
            step_mask=step_mask,
        )
        if self.freeze_anchor:
            with torch.no_grad():
                neg_out = self._run_twin(
                    raw_seq,
                    raw_final,
                    self.neg_units,
                    self.neg_mix,
                    self.neg_read_gate,
                    self.neg_readout,
                    step_mask=step_mask,
                )
        else:
            neg_out = self._run_twin(
                raw_seq,
                raw_final,
                self.neg_units,
                self.neg_mix,
                self.neg_read_gate,
                self.neg_readout,
                step_mask=step_mask,
            )
        return pos_out - neg_out, pos_out

    def forward(
        self,
        raw_seq: torch.Tensor,
        raw_final: torch.Tensor,
        *,
        step_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_parts(raw_seq, raw_final, step_mask=step_mask)[0]

    def export_state(self) -> dict[str, Any]:
        return {
            "spec": asdict(self.cohort_spec),
            "generation": self.generation,
            "freeze_anchor": self.freeze_anchor,
            "lifecycle": self.lifecycle,
            "state_dict": {k: v.detach().clone() for k, v in self.state_dict().items()},
        }


class SelectivePersistentGraph(nn.Module):
    """Additive wrapper preserving ``RouteBGraph``/``MultiHeadGraph`` bit-for-bit by default."""

    def __init__(
        self,
        base_graph: RouteBGraph,
        selective_spec: SelectiveUnitSpec | None = None,
    ) -> None:
        super().__init__()
        self.base_graph = base_graph
        if selective_spec is None:
            selective_spec = SelectiveUnitSpec(
                groups=base_graph.spec.groups,
                units_per_group=base_graph.spec.units_per_group,
                input_dim=base_graph.spec.input_dim,
                hidden_dim=base_graph.spec.hidden_dim,
                output_dim=base_graph.spec.output_dim,
                query_slots=base_graph.spec.query_slots,
                per_sample_state=False,
                seed=base_graph.spec.seed,
                dtype=base_graph.spec.dtype,
            )
        self.selective_spec = selective_spec
        self.growth_generation = 0
        self.growth_cohorts: nn.ModuleList = nn.ModuleList()
        self.task_routes: dict[str, list[int]] = {}
        self._build_selective_branch(selective_spec)

    def _build_selective_branch(self, spec: SelectiveUnitSpec) -> None:
        dtype = spec.torch_dtype
        self.selective_units = SelectiveNeuralUnits(spec)
        raw_dim = len(LEGAL_INPUT_CHANNELS) * spec.input_dim
        ctx_dim = spec.groups * spec.hidden_dim
        self.cross_channel_mix = nn.Linear(raw_dim, ctx_dim, bias=True, dtype=dtype)
        state_dim = 2 * spec.n_units
        readout_hidden = spec.readout_hidden if spec.readout_hidden > 0 else max(32, state_dim)
        self.read_gate = nn.Linear(raw_dim, state_dim, bias=True, dtype=dtype)
        nn.init.constant_(self.read_gate.bias, 1.0)
        self.selective_readout = nn.Sequential(
            nn.Linear(state_dim, readout_hidden, dtype=dtype),
            nn.Tanh(),
            nn.Linear(readout_hidden, spec.output_dim, dtype=dtype),
        )

    def grow_units_neutral_(
        self, additional_per_group: int, *, seed: int | None = None
    ) -> list[int]:
        """Append ``groups * additional_per_group`` immature units with zero birth delta."""
        if additional_per_group < 1:
            raise ValueError("additional_per_group must be >= 1")
        old_units = self.selective_units
        old_n = old_units.n_units
        groups = self.selective_spec.groups
        add_n = groups * additional_per_group
        new_n = old_n + add_n
        dtype = self.selective_spec.torch_dtype
        device = old_units.r.device
        readout_in_layer = cast(nn.Linear, self.selective_readout[0])
        hidden_width = int(readout_in_layer.out_features)
        gen_seed = (
            self.selective_spec.seed + (self.growth_generation + 1) * 1009 if seed is None else seed
        )
        new_spec = replace(
            self.selective_spec,
            units_per_group=self.selective_spec.units_per_group + additional_per_group,
            readout_hidden=hidden_width,
        )
        new_units = SelectiveNeuralUnits(new_spec).to(device=device)

        old_group_idx = old_units.group_index.detach()
        added_group_idx = torch.arange(groups, dtype=torch.int64, device=device).repeat_interleave(
            additional_per_group
        )
        new_group_idx = torch.cat([old_group_idx, added_group_idx], dim=0)
        new_group_mask = (
            new_group_idx.unsqueeze(0)
            == torch.arange(groups, dtype=torch.int64, device=device).unsqueeze(1)
        ).to(dtype)
        if new_spec.cross_group_edges:
            new_edge_mask = (
                torch.zeros(1, 1, dtype=dtype, device=device)
                if new_n == 1
                else (1.0 - torch.eye(new_n, dtype=dtype, device=device)).to(dtype)
            )
        else:
            same_group = (new_group_idx.unsqueeze(1) == new_group_idx.unsqueeze(0)).to(dtype)
            new_edge_mask = same_group * (1.0 - torch.eye(new_n, dtype=dtype, device=device))

        new_gen_buf = torch.cat(
            [
                old_units.unit_generation.detach(),
                torch.full((add_n,), self.growth_generation + 1, dtype=torch.int64, device=device),
            ]
        )
        new_life_buf = torch.cat(
            [
                old_units.unit_lifecycle.detach(),
                torch.full(
                    (add_n,), _LIFECYCLE_TO_ID["immature"], dtype=torch.int64, device=device
                ),
            ]
        )
        new_units.group_index.copy_(new_group_idx)
        new_units.group_mask.copy_(new_group_mask)
        new_units.edge_mask.copy_(new_edge_mask)
        new_units.unit_generation.copy_(new_gen_buf)
        new_units.unit_lifecycle.copy_(new_life_buf)

        rng = torch.Generator().manual_seed(gen_seed)
        with torch.no_grad():
            r_extra = torch.randn(add_n, new_spec.hidden_dim, generator=rng, dtype=dtype)
            r_extra = r_extra / r_extra.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            new_units.r[:old_n].copy_(old_units.r.detach())
            new_units.r[old_n:].copy_(r_extra.to(device=device))

            p_extra = torch.randn(new_spec.hidden_dim, add_n, generator=rng, dtype=dtype)
            p_extra = p_extra / p_extra.norm(dim=0, keepdim=True).clamp(min=1e-6)
            new_units.p[:, :old_n].copy_(old_units.p.detach())
            new_units.p[:, old_n:].copy_(p_extra.to(device=device))

            new_units.c[:old_n].copy_(old_units.c.detach())
            new_units.c[old_n:].fill_(0.85)
            new_units.eta[:old_n].copy_(old_units.eta.detach())
            new_units.eta[old_n:].fill_(0.25)
            new_units.b[:old_n].copy_(old_units.b.detach())
            new_units.b[old_n:].fill_(0.1)
            new_units.write_bias[:old_n].copy_(old_units.write_bias.detach())
            new_units.write_bias[old_n:].fill_(new_spec.init_write_bias)

            w_scale = 0.15 / max(1.0, new_n**0.5)
            w_extra = (
                torch.randn(add_n, new_n, generator=rng, dtype=dtype).to(device=device) * w_scale
            )
            new_units.w[:old_n, :old_n].copy_(old_units.w.detach())
            new_units.w[:old_n, old_n:].zero_()
            new_units.w[old_n:, :].copy_(w_extra * new_edge_mask[old_n:, :])

            wi_extra = (
                torch.randn(
                    add_n, new_spec.groups * new_spec.hidden_dim, generator=rng, dtype=dtype
                ).to(device=device)
                * 0.05
            )
            new_units.write_in.weight[:old_n].copy_(old_units.write_in.weight.detach())
            new_units.write_in.weight[old_n:].copy_(wi_extra)

            ws_extra = (
                torch.randn(add_n, new_n, generator=rng, dtype=dtype).to(device=device) * 0.02
            )
            new_units.write_state.weight[:old_n, :old_n].copy_(
                old_units.write_state.weight.detach()
            )
            new_units.write_state.weight[:old_n, old_n:].zero_()
            new_units.write_state.weight[old_n:, :].copy_(ws_extra)

            # Expand read_gate [2*new_n, raw_dim] keeping z[:old_n] and a[old_n:2*old_n] exact.
            old_rg = self.read_gate
            new_rg = nn.Linear(old_rg.in_features, 2 * new_n, bias=True, dtype=dtype).to(
                device=device
            )
            rg_extra = (
                torch.randn(2 * new_n, old_rg.in_features, generator=rng, dtype=dtype).to(
                    device=device
                )
                * 0.02
            )
            new_rg.weight.copy_(rg_extra)
            new_rg.bias.fill_(1.0)
            new_rg.weight[:old_n].copy_(old_rg.weight[:old_n].detach())
            new_rg.bias[:old_n].copy_(old_rg.bias[:old_n].detach())
            new_rg.weight[new_n : new_n + old_n].copy_(old_rg.weight[old_n : 2 * old_n].detach())
            new_rg.bias[new_n : new_n + old_n].copy_(old_rg.bias[old_n : 2 * old_n].detach())

            # Expand selective_readout[0] [hidden, 2*new_n] with exact 0.0 on newborn cols.
            new_ro_in = nn.Linear(2 * new_n, hidden_width, bias=True, dtype=dtype).to(device=device)
            new_ro_in.weight.zero_()
            new_ro_in.bias.copy_(readout_in_layer.bias.detach())
            new_ro_in.weight[:, :old_n].copy_(readout_in_layer.weight[:, :old_n].detach())
            new_ro_in.weight[:, new_n : new_n + old_n].copy_(
                readout_in_layer.weight[:, old_n : 2 * old_n].detach()
            )

        self.selective_units = new_units
        self.read_gate = new_rg
        self.selective_readout[0] = new_ro_in
        self.selective_spec = new_spec
        self.growth_generation += 1
        return list(range(old_n, new_n))

    def set_task_route(self, name: str, unit_indices: list[int] | tuple[int, ...]) -> list[int]:
        n_units = self.selective_units.n_units
        indices = [int(i) for i in unit_indices]
        if not indices:
            raise ValueError("task_route_empty")
        if any(i < 0 or i >= n_units for i in indices):
            raise ValueError("task_route_index_out_of_range")
        if len(set(indices)) != len(indices):
            raise ValueError("task_route_duplicate_index")
        self.task_routes[name] = indices
        return list(indices)

    def task_route_mask(self, name: str) -> torch.Tensor:
        if name not in self.task_routes:
            raise KeyError(name)
        mask = torch.zeros(self.selective_units.n_units)
        mask[self.task_routes[name]] = 1.0
        return mask

    def export_task_routes(self) -> dict[str, list[int]]:
        return {name: list(indices) for name, indices in self.task_routes.items()}

    @property
    def total_selective_units(self) -> int:
        cohort_units = sum(
            2 * cast(_AntitheticSelectiveCohort, c).cohort_spec.n_units for c in self.growth_cohorts
        )
        return self.selective_units.n_units + cohort_units

    def grow_antithetic_cohort_(
        self,
        groups: int,
        units_per_group: int,
        *,
        readout_hidden: int = 32,
        freeze_mature: bool = True,
        freeze_anchor: bool = True,
        seed: int | None = None,
    ) -> int:
        """Spawn an antithetic twin cohort (exact zero birth diff, full gradient)."""
        if freeze_mature:
            for mod in (
                self.selective_units,
                self.cross_channel_mix,
                self.read_gate,
                self.selective_readout,
                self.growth_cohorts,
            ):
                for param in mod.parameters():
                    param.requires_grad_(False)
                    param.grad = None
        gen_seed = (
            self.selective_spec.seed + (self.growth_generation + 1) * 1009 if seed is None else seed
        )
        cohort_spec = SelectiveUnitSpec(
            groups=groups,
            units_per_group=units_per_group,
            input_dim=self.selective_spec.input_dim,
            hidden_dim=self.selective_spec.hidden_dim,
            output_dim=self.selective_spec.output_dim,
            query_slots=self.selective_spec.query_slots,
            per_sample_state=True,
            selective_gate=self.selective_spec.selective_gate,
            init_write_bias=self.selective_spec.init_write_bias,
            cross_channel_mixing=self.selective_spec.cross_channel_mixing,
            cross_group_edges=self.selective_spec.cross_group_edges,
            readout_hidden=readout_hidden,
            seed=gen_seed,
            dtype=self.selective_spec.dtype,
        )
        cohort = _AntitheticSelectiveCohort(
            cohort_spec,
            generation=self.growth_generation + 1,
            freeze_anchor=freeze_anchor,
        ).to(device=self.selective_units.r.device)
        self.growth_cohorts.append(cohort)
        self.growth_generation += 1
        return len(self.growth_cohorts) - 1

    def set_cohort_lifecycle_(self, cohort_index: int, state: str) -> None:
        cast(_AntitheticSelectiveCohort, self.growth_cohorts[cohort_index]).set_lifecycle_(state)

    def promote_cohorts_to_mature_(self) -> None:
        for cohort in self.growth_cohorts:
            typed_cohort = cast(_AntitheticSelectiveCohort, cohort)
            if typed_cohort.lifecycle == "immature":
                typed_cohort.set_lifecycle_("mature")

    def project_all_parameters_(self) -> None:
        self.selective_units.project_parameters_()
        for cohort in self.growth_cohorts:
            cast(_AntitheticSelectiveCohort, cohort).project_parameters_()

    @property
    def spec(self) -> Any:
        return self.base_graph.spec

    @property
    def per_sample_state(self) -> bool:
        return self.selective_spec.per_sample_state

    def set_per_sample_state(self, enabled: bool) -> None:
        self.selective_spec = replace(self.selective_spec, per_sample_state=enabled)
        self.selective_units.spec = self.selective_spec

    def freeze_base_(self) -> None:
        for parameter in self.base_graph.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None

    def selective_state_dict(self) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for prefix, module in (
            ("selective_units", self.selective_units),
            ("cross_channel_mix", self.cross_channel_mix),
            ("read_gate", self.read_gate),
            ("selective_readout", self.selective_readout),
        ):
            for key, tensor in module.state_dict().items():
                out[f"{prefix}.{key}"] = tensor.detach().clone()
        return out

    def load_selective_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        buckets: dict[str, dict[str, torch.Tensor]] = {
            "selective_units": {},
            "cross_channel_mix": {},
            "read_gate": {},
            "selective_readout": {},
        }
        for full_key, tensor in state_dict.items():
            prefix, _, subkey = full_key.partition(".")
            if prefix not in buckets:
                raise ValueError(f"Unknown selective capsule parameter prefix: {prefix!r}")
            buckets[prefix][subkey] = tensor
        self.selective_units.load_state_dict(buckets["selective_units"], strict=True)
        self.cross_channel_mix.load_state_dict(buckets["cross_channel_mix"], strict=True)
        self.read_gate.load_state_dict(buckets["read_gate"], strict=True)
        self.selective_readout.load_state_dict(buckets["selective_readout"], strict=True)

    def initial_state(self, batch_size: int | None = None) -> dict[str, torch.Tensor]:
        if not self.per_sample_state:
            return self.base_graph.initial_state()
        if batch_size is None:
            raise ValueError("batch_size is required when per_sample_state=True")
        unit_state = self.selective_units.initial_state(batch_size)
        return {
            **unit_state,
            "H": torch.zeros(
                batch_size,
                self.selective_spec.groups,
                self.selective_spec.hidden_dim,
                dtype=self.selective_spec.torch_dtype,
                device=unit_state["z"].device,
            ),
            "internal_tick": torch.zeros((), dtype=torch.int64, device=unit_state["z"].device),
        }

    def encode_sequence_inputs(self, channels: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode legal named channels of shape ``[B, T, D_in]`` (or ``[B, D_in]``) -> ``[B, T, G, D]``."""
        missing = [name for name in LEGAL_INPUT_CHANNELS if name not in channels]
        if missing:
            raise ValueError(("missing_legal_input", tuple(missing)))

        first = channels[LEGAL_INPUT_CHANNELS[0]]
        if first.ndim == 2:
            batch_size, steps = first.shape[0], 1
            normed = {k: v.unsqueeze(1) for k, v in channels.items() if k in LEGAL_INPUT_CHANNELS}
        elif first.ndim == 3:
            batch_size, steps = first.shape[0], first.shape[1]
            normed = {k: v for k, v in channels.items() if k in LEGAL_INPUT_CHANNELS}
        else:
            raise ValueError("selective_channels_must_be_2d_or_3d")

        raw_parts = []
        flat_channels: dict[str, torch.Tensor] = {}
        for channel in LEGAL_INPUT_CHANNELS:
            val = normed[channel]
            if val.ndim != 3 or val.shape[:2] != (batch_size, steps):
                raise ValueError(("selective_channel_shape", channel))
            if val.shape[2] != self.selective_spec.input_dim:
                raise ValueError(("selective_channel_dim", channel))
            raw_parts.append(val)
            flat_channels[channel] = val.reshape(batch_size * steps, self.selective_spec.input_dim)

        base_flat = self.base_graph.encode_legal_inputs(flat_channels)
        base_btgd = base_flat.reshape(
            batch_size, steps, self.base_graph.spec.groups, self.base_graph.spec.hidden_dim
        )
        if not self.selective_spec.cross_channel_mixing:
            if self.base_graph.spec.groups != self.selective_spec.groups:
                raise ValueError("group_count_mismatch_without_cross_channel_mix")
            return base_btgd

        raw_cat = torch.cat(raw_parts, dim=-1)
        mixed = cast(
            torch.Tensor,
            self.cross_channel_mix(raw_cat).reshape(
                batch_size, steps, self.selective_spec.groups, self.selective_spec.hidden_dim
            ),
        )
        if self.base_graph.spec.groups == self.selective_spec.groups:
            return base_btgd + mixed
        return mixed

    def _final_raw_channels(self, channels: dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract ``[B, 6 * input_dim]`` strictly from final tick ``t = T - 1``."""
        missing = [name for name in LEGAL_INPUT_CHANNELS if name not in channels]
        if missing:
            raise ValueError(("missing_legal_input", tuple(missing)))
        parts: list[torch.Tensor] = []
        for channel in LEGAL_INPUT_CHANNELS:
            val = channels[channel]
            if val.ndim == 2:
                parts.append(val)
            elif val.ndim == 3:
                parts.append(val[:, -1, :])
            else:
                raise ValueError("selective_channels_must_be_2d_or_3d")
        return torch.cat(parts, dim=-1)

    def _raw_sequence_and_final(
        self, channels: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract raw `[B, T, 6*input_dim]` and final `[B, 6*input_dim]`."""
        missing = [name for name in LEGAL_INPUT_CHANNELS if name not in channels]
        if missing:
            raise ValueError(("missing_legal_input", tuple(missing)))
        parts_seq: list[torch.Tensor] = []
        parts_final: list[torch.Tensor] = []
        for channel in LEGAL_INPUT_CHANNELS:
            val = channels[channel]
            if val.ndim == 2:
                parts_seq.append(val.unsqueeze(1))
                parts_final.append(val)
            elif val.ndim == 3:
                parts_seq.append(val)
                parts_final.append(val[:, -1, :])
            else:
                raise ValueError("selective_channels_must_be_2d_or_3d")
        return torch.cat(parts_seq, dim=-1), torch.cat(parts_final, dim=-1)

    def forward(
        self,
        channels: dict[str, torch.Tensor],
        gates: torch.Tensor,
        state: dict[str, torch.Tensor] | None = None,
        *,
        head: int = 0,
        step_mask: torch.Tensor | None = None,
        extra: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if not self.per_sample_state:
            if state is None:
                state = self.base_graph.initial_state()
            if isinstance(self.base_graph, MultiHeadGraph):
                return self.base_graph.forward(channels, gates, state, head=head)
            return self.base_graph.forward(channels, gates, state)

        encoded = self.encode_sequence_inputs(channels)
        batch_size, steps = encoded.shape[0], encoded.shape[1]
        if state is None:
            state = self.initial_state(batch_size=batch_size)
        unit_result = self.selective_units(encoded, gates, state, step_mask=step_mask, extra=extra)
        # Strictly final tick state + final tick raw channels only: zero KV history.
        state_final = torch.cat([unit_result["z"], unit_result["a"]], dim=-1)
        read_g = torch.sigmoid(self.read_gate(self._final_raw_channels(channels)))
        gated_state = state_final * read_g
        readout = self.selective_readout(gated_state)
        immature_cohort_logits: list[torch.Tensor] = []
        if len(self.growth_cohorts) > 0:
            raw_seq, raw_final = self._raw_sequence_and_final(channels)
            for cohort in self.growth_cohorts:
                typed_cohort = cast(_AntitheticSelectiveCohort, cohort)
                delta, pos_out = typed_cohort.forward_parts(raw_seq, raw_final, step_mask=step_mask)
                readout = readout + delta
                if typed_cohort.lifecycle == "immature":
                    immature_cohort_logits.append(pos_out)
        return {
            "readout": readout,
            "immature_cohort_logits": immature_cohort_logits,
            "state_final": state_final,
            "read_gate": read_g,
            "unit_result": unit_result,
            "next_state": {
                **state,
                "z": unit_result["z"],
                "a": unit_result["a"],
                "H": state["H"],
                "internal_tick": state["internal_tick"] + steps,
            },
            "path": "selective_persistent_state",
        }


def build_capsule_v2(
    graph: SelectivePersistentGraph,
    *,
    optimizer_state: dict[str, Any] | None = None,
    clocks: tuple[int, int] = (0, 0),
    identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize both wrapped v1 capsule and additive selective-state parameters."""
    base_capsule = build_capsule(
        graph.base_graph,
        optimizer_state=optimizer_state,
        clocks=clocks,
        identity=identity,
    )
    payload: dict[str, Any] = {
        "schema": CAPSULE_V2_SCHEMA,
        "base_capsule": base_capsule,
        "selective_spec": asdict(graph.selective_spec),
        "growth_generation": graph.growth_generation,
        "selective_state_dict": graph.selective_state_dict(),
        "growth_cohorts": [
            cast(_AntitheticSelectiveCohort, cohort).export_state()
            for cohort in graph.growth_cohorts
        ],
        "runtime_clocks": {
            "proposal_clock": clocks[0],
            "parameter_version": clocks[1],
        },
        "task_routes": graph.export_task_routes(),
    }
    payload["capsule_digest"] = value_digest(payload)
    return payload


def validate_capsule_v2(capsule: dict[str, Any]) -> dict[str, Any]:
    if capsule.get("schema") != CAPSULE_V2_SCHEMA:
        raise ValueError("capsule_v2_schema_mismatch")
    body = {k: v for k, v in capsule.items() if k != "capsule_digest"}
    if value_digest(body) != capsule.get("capsule_digest"):
        raise ValueError("capsule_v2_digest_mismatch")
    base_summary = validate_capsule(capsule["base_capsule"])
    clocks = capsule["runtime_clocks"]
    if clocks["proposal_clock"] < clocks["parameter_version"]:
        raise ValueError("capsule_v2_clock_inversion")
    SelectiveUnitSpec(**capsule["selective_spec"])
    routes = capsule.get("task_routes", {})
    if not isinstance(routes, dict):
        raise ValueError("capsule_v2_task_routes_type")
    n_units = SelectiveUnitSpec(**capsule["selective_spec"]).n_units
    for route_name, indices in routes.items():
        if not isinstance(route_name, str) or not isinstance(indices, (list, tuple)):
            raise ValueError("capsule_v2_task_route_shape")
        if len(set(int(i) for i in indices)) != len(indices):
            raise ValueError("capsule_v2_task_route_duplicate")
        if any(int(i) < 0 or int(i) >= n_units for i in indices):
            raise ValueError("capsule_v2_task_route_range")
    for item in capsule.get("growth_cohorts", []):
        SelectiveUnitSpec(**item["spec"])
        if item["lifecycle"] not in _LIFECYCLE_TO_ID:
            raise ValueError(f"Unknown cohort lifecycle: {item['lifecycle']!r}")
    return {
        "schema": CAPSULE_V2_SCHEMA,
        "base": base_summary,
        "selective_tensors": len(capsule["selective_state_dict"]),
        "cohorts": len(capsule.get("growth_cohorts", [])),
        "digest": capsule["capsule_digest"],
    }


def save_capsule_v2(capsule: dict[str, Any], destination: str | Path) -> Path:
    validate_capsule_v2(capsule)
    return atomic_save(capsule, destination)


def load_capsule_v2(source: str | Path) -> dict[str, Any]:
    capsule: dict[str, Any] = atomic_load(source)
    schema = capsule.get("schema")
    if schema == CAPSULE_SCHEMA:
        validate_capsule(capsule)
    elif schema == CAPSULE_V2_SCHEMA:
        validate_capsule_v2(capsule)
    else:
        raise ValueError(f"Unsupported capsule schema: {schema!r}")
    return capsule


def apply_capsule_v2(graph: SelectivePersistentGraph, capsule: dict[str, Any]) -> None:
    """Restore a v2 selective capsule or a legacy v1 base capsule cleanly."""
    schema = capsule.get("schema")
    if schema == CAPSULE_SCHEMA:
        apply_capsule(graph.base_graph, copy.deepcopy(capsule))
        graph.set_per_sample_state(False)
        return
    validate_capsule_v2(capsule)
    apply_capsule(graph.base_graph, copy.deepcopy(capsule["base_capsule"]))
    spec = SelectiveUnitSpec(**capsule["selective_spec"])
    if (
        asdict(spec) != asdict(graph.selective_spec)
        or graph.selective_units.n_units != spec.n_units
    ):
        graph.selective_spec = spec
        graph._build_selective_branch(spec)
    else:
        graph.selective_spec = spec
    graph.load_selective_state_dict(capsule["selective_state_dict"])
    device = graph.selective_units.r.device
    graph.task_routes = {
        str(name): [int(i) for i in indices]
        for name, indices in capsule.get("task_routes", {}).items()
    }
    graph.growth_cohorts = nn.ModuleList()
    for item in capsule.get("growth_cohorts", []):
        c_spec = SelectiveUnitSpec(**item["spec"])
        cohort = _AntitheticSelectiveCohort(
            c_spec,
            generation=int(item["generation"]),
            freeze_anchor=bool(item.get("freeze_anchor", True)),
        ).to(device=device)
        cohort.load_state_dict(item["state_dict"], strict=True)
        cohort.set_lifecycle_(item["lifecycle"])
        graph.growth_cohorts.append(cohort)
    graph.growth_generation = int(capsule.get("growth_generation", 0))
