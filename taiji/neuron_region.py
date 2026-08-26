"""Identity-preserving adaptive neuron regions for native Taiji growth.

This module is deliberately smaller than :class:`TaijiFabric`.  A fabric
region has a fixed state shape because its predictive equations are already
part of the v1 runtime; an adaptive region is the structural-growth organ
where a new unit can be born without redrawing or renumbering existing
units.  It uses the same physical sparse synapse representation as the
fabric, so growth changes stored topology and state dimensions rather than
introducing a dense layer or a Transformer block.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch

from .contracts import StructuralTopologyProposal
from .sparse import SparseSynapses, bound_norm

ADAPTIVE_NEURON_REGION_CHECKPOINT_FORMAT = "adaptive-neuron-region-v1"


@dataclass(frozen=True)
class NeuronRegionDynamics:
    """Explicit dynamics and plasticity controls for one adaptive region."""

    membrane_decay: float = 0.65
    trace_decay: float = 0.82
    recurrent_gain: float = 0.55
    learning_rate: float = 0.025
    threshold_learning_rate: float = 0.015
    target_activity: float = 0.12
    threshold_base: float = 0.02
    threshold_min: float = -0.20
    threshold_max: float = 1.50
    weight_decay: float = 1e-5
    weight_init_scale: float = 0.45
    max_weight_norm: float = 2.5
    max_state_norm: float = 8.0

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.membrane_decay) < 1.0:
            raise ValueError("membrane_decay must be in [0, 1)")
        if not 0.0 <= float(self.trace_decay) < 1.0:
            raise ValueError("trace_decay must be in [0, 1)")
        if float(self.recurrent_gain) < 0.0:
            raise ValueError("recurrent_gain must be non-negative")
        if float(self.learning_rate) <= 0.0:
            raise ValueError("learning_rate must be positive")
        if float(self.threshold_learning_rate) < 0.0:
            raise ValueError("threshold_learning_rate must be non-negative")
        if not 0.0 <= float(self.target_activity) <= 1.0:
            raise ValueError("target_activity must be in [0, 1]")
        if float(self.threshold_min) > float(self.threshold_max):
            raise ValueError("threshold_min cannot exceed threshold_max")
        if float(self.weight_decay) < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if float(self.weight_init_scale) <= 0.0:
            raise ValueError("weight_init_scale must be positive")
        if float(self.max_weight_norm) <= 0.0 or float(self.max_state_norm) <= 0.0:
            raise ValueError("norm limits must be positive")

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> NeuronRegionDynamics:
        return cls(**dict(payload))


class AdaptiveNeuronRegion:
    """A sparse, stateful neuron population that can grow by one stable unit.

    ``unit_ids`` are semantic identities, not array positions.  Existing
    rows, columns, and state entries retain their positions when a unit is
    appended.  The new unit receives fresh sparse contacts, while existing
    contacts and learned weights are copied byte-for-byte.  The input bank
    represents an explicitly named upstream stream; this keeps a cross-region
    connection a contract field rather than an implicit global router.
    """

    def __init__(
        self,
        *,
        region_id: str,
        input_dim: int,
        unit_ids: Sequence[str],
        fan_in: int,
        generator: torch.Generator,
        input_source_id: str | None = None,
        dynamics: NeuronRegionDynamics | None = None,
        recurrent_fan_in: int | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.region_id = str(region_id)
        self.input_dim = int(input_dim)
        self.fan_in = int(fan_in)
        self.input_source_id = None if input_source_id is None else str(input_source_id)
        self.device = torch.device(device)
        self.dynamics = dynamics or NeuronRegionDynamics()
        self._recurrent_fan_in = None if recurrent_fan_in is None else int(recurrent_fan_in)
        if self._recurrent_fan_in is not None and self._recurrent_fan_in <= 0:
            raise ValueError("recurrent_fan_in must be positive when provided")
        self._validate_identity_inputs(self.region_id, self.input_dim, self.fan_in, unit_ids)
        self._unit_ids = tuple(str(item) for item in unit_ids)
        self._lesioned_units: set[str] = set()

        self.incoming = self._new_incoming(len(self._unit_ids), generator)
        self.recurrent = self._new_recurrent(len(self._unit_ids), generator)
        self.membrane = torch.zeros(len(self._unit_ids), device=self.device)
        self.activity = torch.zeros(len(self._unit_ids), device=self.device)
        self.trace = torch.zeros(len(self._unit_ids), device=self.device)
        self.threshold = torch.full(
            (len(self._unit_ids),),
            float(self.dynamics.threshold_base),
            device=self.device,
        )

    @staticmethod
    def _validate_identity_inputs(
        region_id: str,
        input_dim: int,
        fan_in: int,
        unit_ids: Sequence[str],
    ) -> None:
        if not region_id:
            raise ValueError("region_id must not be empty")
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if fan_in <= 0:
            raise ValueError("fan_in must be positive")
        if not unit_ids:
            raise ValueError("unit_ids must not be empty")
        normalized = tuple(str(item) for item in unit_ids)
        if any(not item for item in normalized):
            raise ValueError("unit_ids must not contain empty identities")
        if len(set(normalized)) != len(normalized):
            raise ValueError("unit_ids must be unique")

    @property
    def unit_ids(self) -> tuple[str, ...]:
        return self._unit_ids

    @property
    def unit_count(self) -> int:
        return len(self._unit_ids)

    @property
    def lesioned_unit_ids(self) -> tuple[str, ...]:
        return tuple(item for item in self._unit_ids if item in self._lesioned_units)

    @property
    def edge_count(self) -> int:
        return self.incoming.edge_count + (
            0 if self.recurrent is None else self.recurrent.edge_count
        )

    def unit_index(self, unit_id: str) -> int:
        try:
            return self._unit_ids.index(str(unit_id))
        except ValueError as exc:
            raise KeyError(f"unknown neuron unit: {unit_id}") from exc

    def _new_incoming(self, unit_count: int, generator: torch.Generator) -> SparseSynapses:
        return SparseSynapses(
            unit_count,
            self.input_dim,
            self.fan_in,
            generator=generator,
            init_scale=self.dynamics.weight_init_scale,
            max_weight_norm=self.dynamics.max_weight_norm,
            device=self.device,
        )

    def _new_recurrent(
        self,
        unit_count: int,
        generator: torch.Generator,
    ) -> SparseSynapses | None:
        if unit_count <= 1:
            return None
        existing_recurrent = getattr(self, "recurrent", None)
        requested_fan_in = (
            self._recurrent_fan_in
            if existing_recurrent is None and self._recurrent_fan_in is not None
            else self.fan_in if existing_recurrent is None else existing_recurrent.row_fan_in
        )
        return SparseSynapses(
            unit_count,
            unit_count,
            requested_fan_in,
            generator=generator,
            init_scale=self.dynamics.weight_init_scale,
            max_weight_norm=self.dynamics.max_weight_norm,
            device=self.device,
            allow_self=False,
        )

    @staticmethod
    @torch.no_grad()
    def _copy_prefix(source: SparseSynapses, target: SparseSynapses) -> None:
        if source.row_fan_in != target.row_fan_in:
            raise ValueError("growth changed the existing synapse row width")
        target.pre_index[: source.out_features].copy_(source.pre_index)
        target.edge_weight[: source.out_features].copy_(source.edge_weight)

    def step(self, input_activity: torch.Tensor) -> torch.Tensor:
        """Advance the region one tick and return its non-negative activity."""

        if input_activity.shape != (self.input_dim,):
            raise ValueError(
                f"input activity shape must be ({self.input_dim},), "
                f"got {tuple(input_activity.shape)}"
            )
        drive = self.incoming.forward(input_activity)
        if self.recurrent is not None:
            drive = drive + float(self.dynamics.recurrent_gain) * self.recurrent.forward(
                self.activity
            )
        self.membrane.mul_(float(self.dynamics.membrane_decay)).add_(drive)
        self.membrane.copy_(bound_norm(self.membrane, self.dynamics.max_state_norm))
        activity = torch.tanh(torch.relu(self.membrane - self.threshold))
        if self._lesioned_units:
            activity[list(self._lesioned_indices())] = 0.0
        self.activity.copy_(activity)
        self.trace.mul_(float(self.dynamics.trace_decay)).add_(self.activity)
        self.trace.copy_(bound_norm(self.trace, self.dynamics.max_state_norm))
        return self.activity.clone()

    def learn(self, input_activity: torch.Tensor, postsynaptic_error: torch.Tensor) -> None:
        """Write a local error × eligibility update onto existing contacts."""

        if postsynaptic_error.shape != (self.unit_count,):
            raise ValueError(
                f"postsynaptic error shape must be ({self.unit_count},), "
                f"got {tuple(postsynaptic_error.shape)}"
            )
        self.incoming.local_update(
            postsynaptic_error,
            input_activity,
            learning_rate=self.dynamics.learning_rate,
            weight_decay=self.dynamics.weight_decay,
        )
        if self.recurrent is not None:
            self.recurrent.local_update(
                postsynaptic_error,
                self.activity,
                learning_rate=self.dynamics.learning_rate,
                weight_decay=self.dynamics.weight_decay,
            )
        self.threshold.add_(
            float(self.dynamics.threshold_learning_rate)
            * (self.activity - float(self.dynamics.target_activity))
        ).clamp_(float(self.dynamics.threshold_min), float(self.dynamics.threshold_max))

    def _lesioned_indices(self) -> tuple[int, ...]:
        return tuple(self.unit_index(unit_id) for unit_id in self._lesioned_units)

    def propose_unit_add(
        self,
        *,
        unit_id: str,
        evidence_ids: Sequence[str],
        source_region_id: str | None = None,
        parent_checkpoint_id: str | None = None,
        resource_cost: int = 1,
    ) -> StructuralTopologyProposal:
        """Describe one unit birth without changing the live region."""

        candidate = str(unit_id)
        if not candidate:
            raise ValueError("unit_id must not be empty")
        if candidate in self._unit_ids:
            raise ValueError("unit_id already exists")
        source_id = self.input_source_id if source_region_id is None else str(source_region_id)
        specification: list[tuple[str, Any]] = [
            ("region_id", self.region_id),
            ("unit_id", candidate),
            ("input_dim", self.input_dim),
            ("existing_unit_count", self.unit_count),
        ]
        if source_id is not None:
            specification.append(("source_region_id", source_id))
        proposal_id = f"topology:{self.region_id}:neuron:add:{candidate}"
        return StructuralTopologyProposal(
            proposal_id=proposal_id,
            substrate_id=self.region_id,
            target_kind="neuron",
            operation="add",
            specification=tuple(specification),
            evidence_ids=tuple(str(item) for item in evidence_ids),
            parent_checkpoint_id=parent_checkpoint_id,
            resource_cost=int(resource_cost),
        )

    def _unit_from_proposal(self, proposal: StructuralTopologyProposal) -> str:
        if proposal.status != "pending":
            raise ValueError("only pending neuron proposals can be applied")
        if proposal.target_kind != "neuron" or proposal.operation != "add":
            raise ValueError("proposal is not a neuron add")
        if proposal.substrate_id != self.region_id:
            raise ValueError("neuron proposal targets another region")
        if not proposal.evidence_ids:
            raise ValueError("neuron proposal requires evidence_ids")
        specification = dict(proposal.specification)
        if specification.get("region_id") != self.region_id:
            raise ValueError("neuron proposal region identity does not match")
        if int(specification.get("input_dim", -1)) != self.input_dim:
            raise ValueError("neuron proposal input dimension does not match")
        unit_id = str(specification.get("unit_id", ""))
        if not unit_id:
            raise ValueError("neuron proposal is missing unit identity")
        if unit_id in self._unit_ids:
            raise ValueError("neuron proposal unit identity already exists")
        if int(specification.get("existing_unit_count", -1)) != self.unit_count:
            raise ValueError("neuron proposal parent unit count has drifted")
        return unit_id

    @torch.no_grad()
    def apply_topology_proposal(
        self,
        proposal: StructuralTopologyProposal,
        *,
        generator: torch.Generator,
    ) -> bool:
        """Grow one unit while preserving every pre-existing coordinate."""

        unit_id = self._unit_from_proposal(proposal)
        old_count = self.unit_count
        incoming = self._new_incoming(old_count + 1, generator)
        self._copy_prefix(self.incoming, incoming)
        recurrent = self._new_recurrent(old_count + 1, generator)
        if self.recurrent is not None and recurrent is not None:
            self._copy_prefix(self.recurrent, recurrent)

        self.incoming = incoming
        self.recurrent = recurrent
        self._unit_ids = (*self._unit_ids, unit_id)
        self.membrane = torch.cat((self.membrane, torch.zeros(1, device=self.device)))
        self.activity = torch.cat((self.activity, torch.zeros(1, device=self.device)))
        self.trace = torch.cat((self.trace, torch.zeros(1, device=self.device)))
        self.threshold = torch.cat(
            (
                self.threshold,
                torch.full(
                    (1,),
                    float(self.dynamics.threshold_base),
                    device=self.device,
                ),
            )
        )
        return True

    @torch.no_grad()
    def lesion_topology_proposal(self, proposal: StructuralTopologyProposal) -> bool:
        """Silence one unit while preserving its identity and topology."""

        specification = dict(proposal.specification)
        if proposal.target_kind != "neuron" or proposal.operation != "add":
            raise ValueError("proposal is not a neuron add")
        if proposal.substrate_id != self.region_id:
            raise ValueError("neuron proposal targets another region")
        unit_id = str(specification.get("unit_id", ""))
        index = self.unit_index(unit_id)
        self.incoming.edge_weight[index].zero_()
        if self.recurrent is not None:
            self.recurrent.edge_weight[index].zero_()
            self.recurrent.edge_weight[self.recurrent.pre_index == index] = 0.0
        self.membrane[index] = 0.0
        self.activity[index] = 0.0
        self.trace[index] = 0.0
        self._lesioned_units.add(unit_id)
        return True

    def to_payload(self) -> dict[str, Any]:
        return {
            "storage": ADAPTIVE_NEURON_REGION_CHECKPOINT_FORMAT,
            "region_id": self.region_id,
            "input_dim": self.input_dim,
            "fan_in": self.fan_in,
            "input_source_id": self.input_source_id,
            "unit_ids": list(self._unit_ids),
            "lesioned_unit_ids": list(self._lesioned_units),
            "dynamics": self.dynamics.to_payload(),
            "incoming": self.incoming.to_payload(),
            "recurrent": None if self.recurrent is None else self.recurrent.to_payload(),
            "membrane": self.membrane.detach().cpu().clone(),
            "activity": self.activity.detach().cpu().clone(),
            "trace": self.trace.detach().cpu().clone(),
            "threshold": self.threshold.detach().cpu().clone(),
        }

    def load_payload(self, payload: Mapping[str, Any]) -> None:
        if payload.get("storage") != ADAPTIVE_NEURON_REGION_CHECKPOINT_FORMAT:
            raise ValueError("unsupported adaptive neuron region format")
        if str(payload["region_id"]) != self.region_id:
            raise ValueError("neuron region identity does not match")
        if int(payload["input_dim"]) != self.input_dim or int(payload["fan_in"]) != self.fan_in:
            raise ValueError("neuron region dimensions do not match")
        if tuple(str(item) for item in payload["unit_ids"]) != self._unit_ids:
            raise ValueError("neuron unit identities do not match")
        self.incoming.load_payload(payload["incoming"])
        recurrent_payload = payload.get("recurrent")
        if (recurrent_payload is None) != (self.recurrent is None):
            raise ValueError("recurrent neuron topology does not match")
        if recurrent_payload is not None:
            assert self.recurrent is not None
            self.recurrent.load_payload(recurrent_payload)
        for name in ("membrane", "activity", "trace", "threshold"):
            value = payload[name].detach().to(self.device, dtype=torch.float32).clone()
            if value.shape != (self.unit_count,):
                raise ValueError(f"neuron {name} state shape does not match")
            setattr(self, name, value)
        lesioned = {str(item) for item in payload.get("lesioned_unit_ids", ())}
        if not lesioned.issubset(set(self._unit_ids)):
            raise ValueError("lesioned neuron identity is not present")
        self._lesioned_units = lesioned

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        generator: torch.Generator,
        device: torch.device | str = "cpu",
    ) -> AdaptiveNeuronRegion:
        dynamics = NeuronRegionDynamics.from_payload(payload["dynamics"])
        region = cls(
            region_id=str(payload["region_id"]),
            input_dim=int(payload["input_dim"]),
            unit_ids=tuple(str(item) for item in payload["unit_ids"]),
            fan_in=int(payload["fan_in"]),
            input_source_id=(
                None if payload.get("input_source_id") is None else str(payload["input_source_id"])
            ),
            dynamics=dynamics,
            recurrent_fan_in=(
                None
                if payload.get("recurrent") is None
                else int(payload["recurrent"]["fan_in"])
            ),
            generator=generator,
            device=device,
        )
        region.load_payload(payload)
        return region

    def parameter_tensors(self) -> tuple[torch.Tensor, ...]:
        tensors = [self.incoming.edge_weight]
        if self.recurrent is not None:
            tensors.append(self.recurrent.edge_weight)
        return tuple(tensors)
