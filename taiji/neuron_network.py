"""Explicit cross-region wiring for Taiji adaptive neuron organs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from .contracts import StructuralTopologyProposal
from .neuron_region import AdaptiveNeuronRegion
from .sparse import SparseSynapses

ADAPTIVE_NEURON_NETWORK_CHECKPOINT_FORMAT = "adaptive-neuron-network-v1"


class AdaptiveNeuronNetwork:
    """A small directed network of identity-preserving adaptive regions.

    Cross-region wiring is an owned sparse projection, not a global attention
    router.  Its proposal records source/target identities and both endpoint
    dimensions.  When a region grows, only affected connection rows/columns
    are migrated; existing supports and weights are copied exactly.
    """

    def __init__(
        self,
        regions: Sequence[AdaptiveNeuronRegion],
        *,
        execution_order: Sequence[str] | None = None,
    ) -> None:
        if not regions:
            raise ValueError("adaptive neuron network requires at least one region")
        self._regions: dict[str, AdaptiveNeuronRegion] = {}
        for region in regions:
            if not isinstance(region, AdaptiveNeuronRegion):
                raise TypeError("regions must contain AdaptiveNeuronRegion instances")
            if region.region_id in self._regions:
                raise ValueError(f"duplicate adaptive region: {region.region_id}")
            self._regions[region.region_id] = region
        selected_order = tuple(execution_order or self._regions)
        if set(selected_order) != set(self._regions) or len(selected_order) != len(self._regions):
            raise ValueError("execution_order must contain each region exactly once")
        self.execution_order = selected_order
        self._connections: dict[str, tuple[str, str, SparseSynapses]] = {}
        self._lesioned_connections: set[str] = set()

    @property
    def regions(self) -> tuple[AdaptiveNeuronRegion, ...]:
        return tuple(self._regions.values())

    @property
    def region_ids(self) -> tuple[str, ...]:
        return tuple(self._regions)

    @property
    def connection_ids(self) -> tuple[str, ...]:
        return tuple(self._connections)

    @property
    def connections(self) -> tuple[tuple[str, str, str, SparseSynapses], ...]:
        return tuple(
            (connection_id, source_id, target_id, synapses)
            for connection_id, (source_id, target_id, synapses) in self._connections.items()
        )

    @property
    def edge_count(self) -> int:
        return sum(synapses.edge_count for _, _, synapses in self._connections.values())

    def _region(self, region_id: str) -> AdaptiveNeuronRegion:
        try:
            return self._regions[str(region_id)]
        except KeyError as exc:
            raise ValueError(f"unknown adaptive region: {region_id}") from exc

    def propose_connection_add(
        self,
        *,
        source_region_id: str,
        target_region_id: str,
        evidence_ids: Sequence[str],
        fan_in: int,
        parent_checkpoint_id: str | None = None,
        resource_cost: int = 1,
    ) -> StructuralTopologyProposal:
        """Describe one directed cross-region connection without mutating state."""

        source = self._region(source_region_id)
        target = self._region(target_region_id)
        if source.region_id == target.region_id:
            raise ValueError("cross-region connection requires distinct regions")
        if int(fan_in) <= 0:
            raise ValueError("cross-region fan_in must be positive")
        connection_id = f"connection:{source.region_id}->{target.region_id}"
        if connection_id in self._connections:
            raise ValueError("cross-region connection already exists")
        return StructuralTopologyProposal(
            proposal_id=f"topology:{connection_id}:add",
            substrate_id=connection_id,
            target_kind="region",
            operation="add",
            specification=(
                ("connection_id", connection_id),
                ("source_region_id", source.region_id),
                ("target_region_id", target.region_id),
                ("source_unit_count", source.unit_count),
                ("target_unit_count", target.unit_count),
                ("fan_in", int(fan_in)),
            ),
            evidence_ids=tuple(str(item) for item in evidence_ids),
            parent_checkpoint_id=parent_checkpoint_id,
            resource_cost=int(resource_cost),
        )

    @staticmethod
    def _new_connection(
        source: AdaptiveNeuronRegion,
        target: AdaptiveNeuronRegion,
        fan_in: int,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> SparseSynapses:
        return SparseSynapses(
            target.unit_count,
            source.unit_count,
            fan_in,
            generator=generator,
            init_scale=target.dynamics.weight_init_scale,
            max_weight_norm=target.dynamics.max_weight_norm,
            device=device,
        )

    @staticmethod
    @torch.no_grad()
    def _resize_connection(
        connection: SparseSynapses,
        *,
        out_features: int,
        in_features: int,
        generator: torch.Generator,
        target: AdaptiveNeuronRegion,
    ) -> SparseSynapses:
        if out_features < connection.out_features or in_features < connection.in_features:
            raise ValueError("adaptive region growth cannot shrink a connection")
        if out_features == connection.out_features and in_features == connection.in_features:
            return connection
        resized = SparseSynapses(
            out_features,
            in_features,
            connection.row_fan_in,
            generator=generator,
            init_scale=target.dynamics.weight_init_scale,
            max_weight_norm=connection.max_weight_norm,
            device=connection.device,
        )
        if resized.row_fan_in != connection.row_fan_in:
            raise ValueError("cross-region growth changed the existing connection width")
        resized.pre_index[: connection.out_features].copy_(connection.pre_index)
        resized.edge_weight[: connection.out_features].copy_(connection.edge_weight)
        return resized

    @torch.no_grad()
    def apply_topology_proposal(
        self,
        proposal: StructuralTopologyProposal,
        *,
        generator: torch.Generator,
    ) -> bool:
        """Create one directed sparse connection after validating endpoint shape."""

        if proposal.status != "pending":
            raise ValueError("only pending region proposals can be applied")
        if proposal.target_kind != "region" or proposal.operation != "add":
            raise ValueError("proposal is not a cross-region connection add")
        if not proposal.evidence_ids:
            raise ValueError("cross-region proposal requires evidence_ids")
        specification = dict(proposal.specification)
        connection_id = str(specification.get("connection_id", ""))
        if connection_id != proposal.substrate_id:
            raise ValueError("cross-region connection identity does not match")
        if connection_id in self._connections:
            raise ValueError("cross-region connection already exists")
        source = self._region(str(specification.get("source_region_id", "")))
        target = self._region(str(specification.get("target_region_id", "")))
        if int(specification.get("source_unit_count", -1)) != source.unit_count:
            raise ValueError("cross-region source dimension has drifted")
        if int(specification.get("target_unit_count", -1)) != target.unit_count:
            raise ValueError("cross-region target dimension has drifted")
        fan_in = int(specification.get("fan_in", 0))
        connection = self._new_connection(
            source,
            target,
            fan_in,
            generator=generator,
            device=target.device,
        )
        self._connections[connection_id] = (source.region_id, target.region_id, connection)
        return True

    @torch.no_grad()
    def migrate_region_growth(
        self,
        region_id: str,
        *,
        generator: torch.Generator,
    ) -> None:
        """Resize only connections touching a region after its unit birth."""

        changed = self._region(region_id)
        for connection_id, (source_id, target_id, connection) in tuple(self._connections.items()):
            if changed.region_id not in {source_id, target_id}:
                continue
            source = self._region(source_id)
            target = self._region(target_id)
            self._connections[connection_id] = (
                source_id,
                target_id,
                self._resize_connection(
                    connection,
                    out_features=target.unit_count,
                    in_features=source.unit_count,
                    generator=generator,
                    target=target,
                ),
            )

    @torch.no_grad()
    def apply_neuron_proposal(
        self,
        proposal: StructuralTopologyProposal,
        *,
        generator: torch.Generator,
    ) -> bool:
        """Grow a region and migrate every adjacent cross-region connection."""

        region = self._region(proposal.substrate_id)
        applied = region.apply_topology_proposal(proposal, generator=generator)
        self.migrate_region_growth(region.region_id, generator=generator)
        return applied

    @torch.no_grad()
    def lesion_topology_proposal(self, proposal: StructuralTopologyProposal) -> bool:
        """Silence a cross-region projection without deleting its topology."""

        connection_id = proposal.substrate_id
        try:
            connection = self._connections[connection_id][2]
        except KeyError as exc:
            raise ValueError("unknown cross-region connection") from exc
        connection.edge_weight.zero_()
        self._lesioned_connections.add(connection_id)
        return True

    def step(self, external_inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run one feed-forward tick in explicit region order."""

        activities: dict[str, torch.Tensor] = {}
        for region_id in self.execution_order:
            region = self._regions[region_id]
            external = external_inputs.get(region_id)
            if external is None:
                external = torch.zeros(region.input_dim, device=region.device)
            cross_drive = torch.zeros(region.unit_count, device=region.device)
            for source_id, target_id, connection in self._connections.values():
                if target_id == region_id and source_id in activities:
                    cross_drive.add_(connection.forward(activities[source_id]))
            activities[region_id] = region.step(external, additional_drive=cross_drive)
        return {region_id: value.clone() for region_id, value in activities.items()}

    def to_payload(self) -> dict[str, Any]:
        return {
            "storage": ADAPTIVE_NEURON_NETWORK_CHECKPOINT_FORMAT,
            "execution_order": list(self.execution_order),
            "regions": {
                region_id: region.to_payload()
                for region_id, region in self._regions.items()
            },
            "connections": {
                connection_id: {
                    "source_region_id": source_id,
                    "target_region_id": target_id,
                    "synapses": synapses.to_payload(),
                }
                for connection_id, (source_id, target_id, synapses) in self._connections.items()
            },
            "lesioned_connections": list(self._lesioned_connections),
        }

    def load_payload(self, payload: Mapping[str, Any]) -> None:
        if payload.get("storage") != ADAPTIVE_NEURON_NETWORK_CHECKPOINT_FORMAT:
            raise ValueError("unsupported adaptive neuron network format")
        if tuple(str(item) for item in payload["execution_order"]) != self.execution_order:
            raise ValueError("adaptive network execution order does not match")
        region_payloads = payload.get("regions", {})
        if not isinstance(region_payloads, dict) or set(region_payloads) != set(self._regions):
            raise ValueError("adaptive network region identities do not match")
        for region_id, region_payload in region_payloads.items():
            self._regions[str(region_id)].load_payload(region_payload)
        connections = payload.get("connections", {})
        if not isinstance(connections, dict):
            raise ValueError("adaptive network connections must be a mapping")
        self._connections = {}
        for connection_id, connection_payload in connections.items():
            if not isinstance(connection_payload, dict):
                raise ValueError("adaptive network connection must be a mapping")
            source_id = str(connection_payload["source_region_id"])
            target_id = str(connection_payload["target_region_id"])
            source = self._region(source_id)
            target = self._region(target_id)
            synapse_payload = connection_payload["synapses"]
            synapses = SparseSynapses(
                target.unit_count,
                source.unit_count,
                int(synapse_payload["fan_in"]),
                generator=torch.Generator(device="cpu").manual_seed(0),
                init_scale=target.dynamics.weight_init_scale,
                max_weight_norm=float(synapse_payload["max_weight_norm"]),
                device=target.device,
            )
            synapses.load_payload(synapse_payload)
            self._connections[str(connection_id)] = (source_id, target_id, synapses)
        lesioned = {str(item) for item in payload.get("lesioned_connections", ())}
        if not lesioned.issubset(set(self._connections)):
            raise ValueError("lesioned connection identity is not present")
        self._lesioned_connections = lesioned

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        generator: torch.Generator,
        device: torch.device | str = "cpu",
    ) -> AdaptiveNeuronNetwork:
        region_payloads = payload.get("regions", {})
        if not isinstance(region_payloads, dict):
            raise ValueError("adaptive network regions must be a mapping")
        regions = tuple(
            AdaptiveNeuronRegion.from_payload(
                region_payload,
                generator=generator,
                device=device,
            )
            for region_payload in region_payloads.values()
        )
        network = cls(
            regions,
            execution_order=tuple(str(item) for item in payload["execution_order"]),
        )
        network.load_payload(payload)
        return network

    def parameter_tensors(self) -> tuple[torch.Tensor, ...]:
        tensors: list[torch.Tensor] = []
        for region in self._regions.values():
            tensors.extend(region.parameter_tensors())
        for _, _, synapses in self._connections.values():
            tensors.append(synapses.edge_weight)
        return tuple(tensors)
