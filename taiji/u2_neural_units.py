"""Stateful neural-unit operators and the two P3.9 reference graph factories.

Implements the synchronous recurrence of
SPEC-TAIJI_U2_NEURAL_UNIT_LIFECYCLE_20261113 section 3 and the A/B split of
DECISION-TAIJI_NEURAL_CORE_PATH_20261114 section 2. Route A keeps the fixed H
mediator; route B reads unit state directly and never routes its readout through
H. Nothing here loads a task dataset, runs a training step, or opens a sealed face.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from taiji.u2_history import atomic_load, atomic_save, value_digest

C_MAX = 1.0 - 1e-4
ETA_MIN = 1e-4
ETA_MAX = 1.0
B_MAX = 2.0
W_MAX = 1.0
C_CANDIDATES = (0.0, 0.5, 0.9)
MAX_SHADOW_EDGES_PER_UNIT = 2
MAX_NEW_UNITS_PER_GENERATION = 3
MAX_NEW_UNITS_FRACTION = 0.25
LEGAL_INPUT_CHANNELS = ("rgb", "audio", "proprio", "question", "goal", "feedback")
LIFECYCLE = ("proposed", "immature", "mature", "dormant", "retired")
CAPSULE_SCHEMA = "taiji-u2-neural-unit-capsule-v1"

COMPAT_FP64_ATOL = 1e-10
COMPAT_FP32_ATOL = 1e-5


@dataclass(frozen=True)
class UnitGraphSpec:
    groups: int = 6
    units_per_group: int = 2
    input_dim: int = 8
    hidden_dim: int = 8
    output_dim: int = 4
    seed: int = 2026111401
    query_slots: int = 4
    dtype: str = "float32"

    @property
    def n_units(self) -> int:
        return self.groups * self.units_per_group

    @property
    def torch_dtype(self) -> torch.dtype:
        return getattr(torch, self.dtype)


def make_identities(spec: UnitGraphSpec) -> dict[str, Any]:
    """Fresh unit/edge registries; IDs are never reused."""

    units = []
    for index in range(spec.n_units):
        units.append(
            {
                "unit_id": f"u{index:04d}",
                "group_id": index // spec.units_per_group,
                "parent_ids": [],
                "birth_version": 0,
                "lifecycle": "mature",
                "last_internal_tick": 0,
                "state_source_version": 0,
            }
        )
    edges = []
    for group in range(spec.groups):
        first = group * spec.units_per_group
        for offset in range(spec.units_per_group):
            source = first + ((offset + 1) % spec.units_per_group)
            target = first + offset
            edges.append(
                {
                    "edge_id": f"e{len(edges):04d}",
                    "source": units[source]["unit_id"],
                    "target": units[target]["unit_id"],
                    "create_version": 0,
                    "retire_version": None,
                    "optimizer_owner": "units.w",
                }
            )
    return {
        "units": units,
        "edges": edges,
        "generation": 0,
        "sparse_index_order": [edge["edge_id"] for edge in edges],
    }


def _identity_index(identities: dict[str, Any]) -> dict[str, int]:
    return {unit["unit_id"]: index for index, unit in enumerate(identities["units"])}


class NeuralUnits(nn.Module):
    """Synchronous z/a recurrence with explicit unit and edge registries."""

    def __init__(self, spec: UnitGraphSpec, identities: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.spec = spec
        self.identities = (
            copy.deepcopy(identities) if identities is not None else make_identities(spec)
        )
        dtype = spec.torch_dtype
        n = len(self.identities["units"])
        self.n_units = n
        d, h = spec.input_dim, spec.hidden_dim
        generator = torch.Generator().manual_seed(spec.seed)

        def rand(shape: tuple[int, ...], scale: float) -> torch.Tensor:
            values = torch.rand(shape, generator=generator, dtype=torch.float64) * 2.0 - 1.0
            return (values * scale).to(dtype)

        self.r = nn.Parameter(rand((n, d), 0.3))
        self.p = nn.Parameter(rand((h, n), 0.3))
        self.c = nn.Parameter(torch.zeros(n, dtype=dtype))
        self.eta = nn.Parameter(torch.full((n,), 0.1, dtype=dtype))
        self.b = nn.Parameter(torch.zeros(n, dtype=dtype))
        self.w = nn.Parameter(torch.zeros(n, n, dtype=dtype))

        index = _identity_index(self.identities)
        mask = torch.zeros(n, n, dtype=dtype)
        for edge in self.identities["edges"]:
            if edge["retire_version"] is None:
                mask[index[edge["target"]], index[edge["source"]]] = 1.0
        self.register_buffer("edge_mask", mask)
        self.register_buffer(
            "group_index", torch.tensor([u["group_id"] for u in self.identities["units"]])
        )
        group_mask = torch.zeros(spec.groups, n, dtype=dtype)
        for position, unit in enumerate(self.identities["units"]):
            group_mask[unit["group_id"], position] = 1.0
        self.register_buffer("group_mask", group_mask)
        self.project_parameters_()

    def project_parameters_(self) -> None:
        """Exact endpoints are representable: c=0 is a real zero, not a sigmoid."""

        with torch.no_grad():
            self.c.clamp_(0.0, C_MAX)
            self.eta.clamp_(ETA_MIN, ETA_MAX)
            self.b.clamp_(0.0, B_MAX)
            self.w.mul_(self.edge_mask).clamp_(-W_MAX, W_MAX)

    def initial_state(self) -> dict[str, torch.Tensor]:
        dtype = self.spec.torch_dtype
        return {
            "z": torch.zeros(self.n_units, dtype=dtype),
            "a": torch.zeros(self.n_units, dtype=dtype),
            "H": torch.zeros(self.spec.groups, self.spec.hidden_dim, dtype=dtype),
            "internal_tick": torch.zeros((), dtype=torch.int64),
        }

    def forward(
        self, inputs: torch.Tensor, gates: torch.Tensor, state: dict[str, torch.Tensor]
    ) -> dict[str, Any]:
        """inputs [T,G,D]; gates [T,N]. All old z/a are read before any write."""

        if inputs.ndim != 3 or gates.ndim != 2:
            raise ValueError("unit_input_shape")
        steps = inputs.shape[0]
        x = inputs[:, self.group_index, :]
        drive = (x * self.r.unsqueeze(0)).sum(-1)
        weights = self.w * self.edge_mask
        z, a = state["z"], state["a"]
        contributions, z_trace, a_trace, activity = [], [], [], []
        for tick in range(steps):
            gate = gates[tick]
            incoming = (gate * z) @ weights.T
            v = drive[tick] + incoming - self.b * a
            z = self.c * z + (1.0 - self.c) * torch.tanh(v)
            a = (1.0 - self.eta) * a + self.eta * z
            unit_activity = gate * z
            contributions.append(unit_activity)
            z_trace.append(z)
            a_trace.append(a)
            activity.append(unit_activity)
        stacked = torch.stack(activity)
        return {
            "activity": stacked,
            "contribution": torch.einsum("hn,gn,tn->tgh", self.p, self.group_mask, stacked),
            "z": z,
            "a": a,
            "z_trace": torch.stack(z_trace),
            "a_trace": torch.stack(a_trace),
        }

    def with_neutral_birth(self, group_id: int, parent_ids: list[str]) -> NeuralUnits:
        """Add one exactly neutral immature unit; output and states stay identical."""

        active = sum(1 for unit in self.identities["units"] if unit["lifecycle"] != "retired")
        limit = min(MAX_NEW_UNITS_PER_GENERATION, max(1, int(active * MAX_NEW_UNITS_FRACTION)))
        if limit < 1:
            raise ValueError("birth_limit_exhausted")
        old_n = self.n_units
        identities = copy.deepcopy(self.identities)
        new_id = f"u{len(identities['units']):04d}"
        identities["units"].append(
            {
                "unit_id": new_id,
                "group_id": group_id,
                "parent_ids": list(parent_ids),
                "birth_version": identities["generation"] + 1,
                "lifecycle": "immature",
                "last_internal_tick": 0,
                "state_source_version": 0,
            }
        )
        identities["generation"] += 1
        child = NeuralUnits(self.spec, identities)
        with torch.no_grad():
            child.r[:old_n].copy_(self.r)
            child.r[old_n:].zero_()
            child.p[:, :old_n].copy_(self.p)
            child.p[:, old_n:].zero_()
            child.c[:old_n].copy_(self.c)
            child.c[old_n:].zero_()
            child.eta[:old_n].copy_(self.eta)
            child.eta[old_n:].fill_(0.1)
            child.b[:old_n].copy_(self.b)
            child.b[old_n:].zero_()
            child.w[:old_n, :old_n].copy_(self.w)
            child.edge_mask[:old_n, :old_n].copy_(self.edge_mask)
            child.edge_mask[old_n:, :].zero_()
            child.edge_mask[:, old_n:].zero_()
            child.project_parameters_()
        return child

    def set_lifecycle(self, unit_id: str, phase: str) -> dict[str, Any]:
        if phase not in LIFECYCLE:
            raise ValueError("unknown_lifecycle_phase")
        for unit in self.identities["units"]:
            if unit["unit_id"] == unit_id:
                unit["lifecycle"] = phase
                return dict(unit)
        raise KeyError(unit_id)

    def active_gate(self, gates: torch.Tensor) -> torch.Tensor:
        """Dormant/retired units are cut from legal input gating."""

        active = torch.tensor(
            [1.0 if unit["lifecycle"] == "mature" else 0.0 for unit in self.identities["units"]],
            dtype=self.spec.torch_dtype,
        )
        return gates * active.unsqueeze(0)


class _BaseGraph(nn.Module):
    def __init__(self, spec: UnitGraphSpec, route: str) -> None:
        super().__init__()
        self.spec = spec
        self.route = route
        self.units = NeuralUnits(spec)
        dtype = spec.torch_dtype
        generator = torch.Generator().manual_seed(spec.seed + 17)
        self.input_projection = nn.Parameter(
            (
                (
                    torch.rand(
                        spec.input_dim, spec.input_dim, generator=generator, dtype=torch.float64
                    )
                    * 2
                    - 1
                )
                * 0.2
            ).to(dtype)
        )
        self.input_encoders = nn.ModuleDict(
            {
                channel: nn.Linear(spec.input_dim, spec.input_dim, bias=False, dtype=dtype)
                for channel in LEGAL_INPUT_CHANNELS
            }
        )
        with torch.no_grad():
            for layer in self.input_encoders.values():
                layer.weight.copy_(torch.eye(spec.input_dim, dtype=dtype))

    def encode_legal_inputs(self, channels: dict[str, torch.Tensor]) -> torch.Tensor:
        """Named legal inputs -> [T,G,D]; missing names fail closed."""

        missing = [name for name in LEGAL_INPUT_CHANNELS if name not in channels]
        if missing:
            raise ValueError(("missing_legal_input", tuple(missing)))
        columns = []
        for channel in LEGAL_INPUT_CHANNELS:
            value = channels[channel]
            if value.ndim != 2 or value.shape[1] != self.spec.input_dim:
                raise ValueError(("legal_input_shape", channel))
            columns.append(self.input_encoders[channel](value) @ self.input_projection)
        stacked = torch.stack(columns, dim=1)
        repeats = []
        for group in range(self.spec.groups):
            repeats.append(stacked[:, group % len(LEGAL_INPUT_CHANNELS), :])
        return torch.stack(repeats, dim=1)

    def initial_state(self) -> dict[str, torch.Tensor]:
        return self.units.initial_state()

    def with_neutral_birth(self, group_id: int, parent_ids: list[str]) -> _BaseGraph:
        """Clone the whole graph with one neutral immature unit and an extended state."""

        child_units = self.units.with_neutral_birth(group_id, parent_ids)
        clone = copy.deepcopy(self)
        clone.units = child_units
        return clone

    def extend_state(
        self, state: dict[str, torch.Tensor], extra: int = 1
    ) -> dict[str, torch.Tensor]:
        grown = {}
        for key, value in state.items():
            if key in ("z", "a"):
                grown[key] = torch.nn.functional.pad(value, (0, extra))
            else:
                grown[key] = value.clone() if isinstance(value, torch.Tensor) else value
        return grown


class RouteAGraph(_BaseGraph):
    """Compatible embedding: unit contributions are projected through fixed H."""

    def __init__(self, spec: UnitGraphSpec) -> None:
        super().__init__(spec, "A")
        dtype = spec.torch_dtype
        generator = torch.Generator().manual_seed(spec.seed + 31)
        self.w_g = nn.Parameter(
            (
                (
                    torch.rand(
                        spec.groups,
                        spec.hidden_dim,
                        spec.input_dim,
                        generator=generator,
                        dtype=torch.float64,
                    )
                    * 2
                    - 1
                )
                * 0.2
            ).to(dtype)
        )
        self.ell = nn.Parameter(torch.full((spec.groups,), 0.5, dtype=dtype))
        self.h_readout = nn.Linear(spec.hidden_dim, spec.output_dim, bias=False, dtype=dtype)

    def forward(
        self, channels: dict[str, torch.Tensor], gates: torch.Tensor, state: dict[str, torch.Tensor]
    ) -> dict[str, Any]:
        inputs = self.encode_legal_inputs(channels)
        result = self.units(inputs, self.units.active_gate(gates), state)
        H = state["H"]
        h_trace = []
        for tick in range(inputs.shape[0]):
            drive = (
                torch.einsum("ghd,tgd->tgh", self.w_g, inputs)[tick] + result["contribution"][tick]
            )
            H = (1.0 - self.ell.unsqueeze(-1)) * H + self.ell.unsqueeze(-1) * torch.tanh(drive)
            h_trace.append(H)
        return {
            "readout": self.h_readout(torch.stack(h_trace).mean(dim=1)),
            "h_trace": torch.stack(h_trace),
            "unit_result": result,
            "next_state": {
                **state,
                "z": result["z"],
                "a": result["a"],
                "H": H,
                "internal_tick": state["internal_tick"] + inputs.shape[0],
            },
            "path": "through_H",
        }


class RouteBGraph(_BaseGraph):
    """Unit-native core: direct query over z/a; H is scratch, never on the readout path."""

    def __init__(self, spec: UnitGraphSpec) -> None:
        super().__init__(spec, "B")
        dtype = spec.torch_dtype
        generator = torch.Generator().manual_seed(spec.seed + 47)
        self.query = nn.Parameter(
            (
                (
                    torch.rand(
                        spec.query_slots, spec.hidden_dim, generator=generator, dtype=torch.float64
                    )
                    * 2
                    - 1
                )
                * 0.3
            ).to(dtype)
        )
        self.state_mix = nn.Parameter(torch.tensor([0.7, 0.3], dtype=dtype))
        self.readout = nn.Linear(spec.query_slots, spec.output_dim, bias=False, dtype=dtype)

    def forward(
        self, channels: dict[str, torch.Tensor], gates: torch.Tensor, state: dict[str, torch.Tensor]
    ) -> dict[str, Any]:
        inputs = self.encode_legal_inputs(channels)
        active = self.units.active_gate(gates)
        result = self.units(inputs, active, state)
        # Direct unit query uses the *gated* unit activity, so legal content gates
        # and lifecycle cuts genuinely change the readout (no H intermediary).
        gated_a = result["a_trace"] * active
        state_view = self.state_mix[0] * result["activity"] + self.state_mix[1] * gated_a
        projected = torch.einsum("hn,tn->th", self.units.p, state_view)
        direct = projected @ self.query.T
        return {
            "readout": self.readout(direct),
            "direct_query": direct,
            "unit_result": result,
            "next_state": {
                **state,
                "z": result["z"],
                "a": result["a"],
                "H": state["H"],
                "internal_tick": state["internal_tick"] + inputs.shape[0],
            },
            "path": "direct_units",
        }

    def cut_h(self, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Zero/perturb H scratch; B readout must be bit-identical."""

        return {**state, "H": torch.zeros_like(state["H"])}


def build_route(route: str, spec: UnitGraphSpec | None = None) -> _BaseGraph:
    graph_spec = spec or UnitGraphSpec()
    if route == "A":
        return RouteAGraph(graph_spec)
    if route == "B":
        return RouteBGraph(graph_spec)
    raise ValueError("unknown_route")


def legacy_rp_reference(
    graph: _BaseGraph, channels: dict[str, torch.Tensor], gates: torch.Tensor
) -> torch.Tensor:
    """Old R/P form at the c=0,b=0,E=empty endpoint: p @ (gate * tanh(r^T x))."""

    inputs = graph.encode_legal_inputs(channels)
    x = inputs[:, graph.units.group_index, :]
    z = torch.tanh((x * graph.units.r.unsqueeze(0)).sum(-1))
    activity = gates * z
    return torch.einsum("hn,gn,tn->tgh", graph.units.p, graph.units.group_mask, activity)


# ---------------------------------------------------------------------------
# Generic full learning capsule for either reference graph
# ---------------------------------------------------------------------------


def _capsule_digest(payload: dict[str, Any]) -> str:
    """Hash the whole typed payload, tensors included, via the history digest."""

    return value_digest(payload)


def build_capsule(
    graph: _BaseGraph,
    *,
    optimizer_state: dict[str, Any] | None = None,
    clocks: tuple[int, int] = (0, 0),
    identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state_dict = {key: value.detach().clone() for key, value in graph.state_dict().items()}
    payload = {
        "runtime": {
            "route": graph.route,
            "spec": {
                "groups": graph.spec.groups,
                "units_per_group": graph.spec.units_per_group,
                "input_dim": graph.spec.input_dim,
                "hidden_dim": graph.spec.hidden_dim,
                "output_dim": graph.spec.output_dim,
                "seed": graph.spec.seed,
                "query_slots": graph.spec.query_slots,
                "dtype": graph.spec.dtype,
            },
            "state_dict": state_dict,
            "unit_registry": [dict(unit) for unit in graph.units.identities["units"]],
            "edge_registry": [dict(edge) for edge in graph.units.identities["edges"]],
            "sparse_index_order": list(graph.units.identities["sparse_index_order"]),
            "generation": graph.units.identities["generation"],
        },
        "learning_state": {
            "moments": {} if optimizer_state is None else copy.deepcopy(optimizer_state),
            "parameter_version": clocks[1],
            "proposal_clock": clocks[0],
        },
        "sampling": {"rng_state": torch.get_rng_state().tolist(), "episode_cursor": 0},
        "proposal": {"last_receipt": None},
        "identity": identity
        or {"parent": "frozen-research-start", "native_parent_qualified": False},
    }
    payload["schema"] = CAPSULE_SCHEMA
    payload["capsule_digest"] = _capsule_digest(payload)
    return payload


def validate_capsule(capsule: dict[str, Any]) -> dict[str, Any]:
    if capsule.get("schema") != CAPSULE_SCHEMA:
        raise ValueError("capsule_schema_mismatch")
    recomputed = _capsule_digest(
        {key: value for key, value in capsule.items() if key != "capsule_digest"}
    )
    if recomputed != capsule["capsule_digest"]:
        raise ValueError("capsule_digest_mismatch")
    registry = capsule["runtime"]["unit_registry"]
    ids = [unit["unit_id"] for unit in registry]
    if len(set(ids)) != len(ids):
        raise ValueError("capsule_duplicate_unit_id")
    if capsule["learning_state"]["proposal_clock"] < capsule["learning_state"]["parameter_version"]:
        raise ValueError("capsule_clock_inversion")
    return {
        "units": len(registry),
        "edges": len(capsule["runtime"]["edge_registry"]),
        "tensors": len(capsule["runtime"]["state_dict"]),
        "digest": capsule["capsule_digest"],
    }


def save_capsule(destination: str | Path, capsule: dict[str, Any]) -> Path:
    validate_capsule(capsule)
    return atomic_save(capsule, destination)


def load_capsule(source: str | Path) -> dict[str, Any]:
    capsule = atomic_load(source)
    validate_capsule(capsule)
    return capsule


def apply_capsule(graph: _BaseGraph, capsule: dict[str, Any]) -> None:
    validate_capsule(capsule)
    if capsule["runtime"]["route"] != graph.route:
        raise ValueError("capsule_route_mismatch")
    registry = capsule["runtime"]["unit_registry"]
    if len(registry) != graph.units.n_units:
        identities = {
            "units": copy.deepcopy(registry),
            "edges": copy.deepcopy(capsule["runtime"]["edge_registry"]),
            "sparse_index_order": list(capsule["runtime"]["sparse_index_order"]),
            "generation": capsule["runtime"]["generation"],
        }
        graph.units = NeuralUnits(graph.spec, identities)
    graph.load_state_dict(capsule["runtime"]["state_dict"])
    graph.units.identities = {
        "units": copy.deepcopy(registry),
        "edges": copy.deepcopy(capsule["runtime"]["edge_registry"]),
        "sparse_index_order": list(capsule["runtime"]["sparse_index_order"]),
        "generation": capsule["runtime"]["generation"],
    }


def capsule_summary(capsule: dict[str, Any]) -> dict[str, Any]:
    validate_capsule(capsule)
    return {
        "route": capsule["runtime"]["route"],
        "units": len(capsule["runtime"]["unit_registry"]),
        "edges": len(capsule["runtime"]["edge_registry"]),
        "tensors": len(capsule["runtime"]["state_dict"]),
        "proposal_clock": capsule["learning_state"]["proposal_clock"],
        "parameter_version": capsule["learning_state"]["parameter_version"],
        "digest": capsule["capsule_digest"],
    }
