"""Explicit cross-region wiring for Taiji adaptive neuron organs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from .contracts import StructuralTopologyProposal
from .cross_region_learning import CrossRegionCooperationLearner
from .neuron_region import AdaptiveNeuronRegion, NeuronRegionDynamics
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
        self._connection_resource_costs: dict[str, float] = {}
        self._lesioned_connections: set[str] = set()
        self._lesioned_regions: set[str] = set()
        self._cooperation_learner: CrossRegionCooperationLearner | None = None

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

    @property
    def cooperation_learner(self) -> CrossRegionCooperationLearner | None:
        return self._cooperation_learner

    def attach_cooperation_learner(
        self,
        learner: CrossRegionCooperationLearner | None,
    ) -> None:
        """Attach the learner that selects among this network's explicit routes."""

        if learner is not None and not isinstance(learner, CrossRegionCooperationLearner):
            raise TypeError("learner must be a CrossRegionCooperationLearner or None")
        self._cooperation_learner = learner
        if learner is not None:
            for connection_id, _, _, connection in self.connections:
                learner.register_connection(
                    connection_id,
                    resource_cost=self._connection_resource_costs.get(
                        connection_id,
                        1.0,
                    ),
                )

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
                ("topology_role", "cross_region_connection"),
            ),
            evidence_ids=tuple(str(item) for item in evidence_ids),
            parent_checkpoint_id=parent_checkpoint_id,
            resource_cost=int(resource_cost),
        )

    def propose_region_add(
        self,
        *,
        region_id: str,
        input_dim: int,
        unit_count: int,
        fan_in: int,
        evidence_ids: Sequence[str],
        parent_checkpoint_id: str | None = None,
        resource_cost: int = 1,
        dynamics: NeuronRegionDynamics | None = None,
    ) -> StructuralTopologyProposal:
        """Describe a new adaptive region without mutating the network."""

        key = str(region_id)
        if not key:
            raise ValueError("adaptive region_id must not be empty")
        if key in self._regions:
            raise ValueError("adaptive region already exists")
        if int(input_dim) <= 0 or int(unit_count) <= 0 or int(fan_in) <= 0:
            raise ValueError("adaptive region dimensions must be positive")
        selected_dynamics = dynamics or NeuronRegionDynamics()
        unit_ids = tuple(f"{key}.u{index}" for index in range(int(unit_count)))
        return StructuralTopologyProposal(
            proposal_id=f"topology:{key}:region:add",
            substrate_id=key,
            target_kind="region",
            operation="add",
            specification=(
                ("region_id", key),
                ("input_dim", int(input_dim)),
                ("unit_count", int(unit_count)),
                ("unit_ids", unit_ids),
                ("fan_in", int(fan_in)),
                ("dynamics", selected_dynamics.to_payload()),
                ("topology_role", "region"),
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
        if specification.get("topology_role") != "cross_region_connection":
            raise ValueError("proposal is not a cross-region connection add")
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
        self._connection_resource_costs[connection_id] = float(proposal.resource_cost)
        if self._cooperation_learner is not None:
            self._cooperation_learner.register_connection(
                connection_id,
                resource_cost=float(proposal.resource_cost),
            )
        return True

    @torch.no_grad()
    def apply_region_proposal(
        self,
        proposal: StructuralTopologyProposal,
        *,
        generator: torch.Generator,
    ) -> bool:
        """Append one substrate region while preserving existing region order."""

        if proposal.status != "pending":
            raise ValueError("only pending region proposals can be applied")
        if proposal.target_kind != "region" or proposal.operation != "add":
            raise ValueError("proposal is not a region add")
        if not proposal.evidence_ids:
            raise ValueError("region proposal requires evidence_ids")
        specification = dict(proposal.specification)
        if specification.get("topology_role") != "region":
            raise ValueError("proposal is not a region add")
        region_id = str(specification.get("region_id", ""))
        if region_id != proposal.substrate_id:
            raise ValueError("region identity does not match proposal")
        if region_id in self._regions:
            raise ValueError("adaptive region already exists")
        unit_ids = tuple(str(item) for item in specification.get("unit_ids", ()))
        input_dim = int(specification.get("input_dim", 0))
        fan_in = int(specification.get("fan_in", 0))
        if int(specification.get("unit_count", -1)) != len(unit_ids):
            raise ValueError("region unit count does not match unit identities")
        if not unit_ids or input_dim <= 0 or fan_in <= 0:
            raise ValueError("region specification dimensions must be positive")
        dynamics_payload = specification.get("dynamics", {})
        if not isinstance(dynamics_payload, Mapping):
            raise ValueError("region dynamics must be a mapping")
        region = AdaptiveNeuronRegion(
            region_id=region_id,
            input_dim=input_dim,
            unit_ids=unit_ids,
            fan_in=fan_in,
            dynamics=NeuronRegionDynamics.from_payload(dynamics_payload),
            generator=generator,
            device=self._region_device(),
        )
        self._regions[region_id] = region
        self.execution_order = (*self.execution_order, region_id)
        return True

    def _region_device(self) -> torch.device:
        return self.regions[0].device

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

    def lesion_region(self, region_id: str) -> bool:
        """Functionally silence one region while retaining its topology."""

        region = self._region(region_id)
        self._lesioned_regions.add(region.region_id)
        region.activity.zero_()
        return True

    def observe_connection(
        self,
        connection_id: str,
        *,
        prediction_error: float,
        holdout_transfer: float,
        resource_state: float,
        selected: bool = True,
    ) -> float:
        """Credit one route with outcome evidence through the attached learner."""

        if self._cooperation_learner is None:
            raise RuntimeError("cross-region cooperation learner is not attached")
        if str(connection_id) not in self._connections:
            raise ValueError(f"unknown cross-region connection: {connection_id}")
        return self._cooperation_learner.observe(
            connection_id,
            prediction_error=prediction_error,
            holdout_transfer=holdout_transfer,
            resource_state=resource_state,
            selected=selected,
        )

    def credit_routes_from_outcome(
        self,
        actual_activities: Mapping[str, torch.Tensor],
        expected_activities: Mapping[str, torch.Tensor],
        *,
        active_connection_ids: Sequence[str],
        resource_budget: float,
        holdout: bool = False,
    ) -> dict[str, float]:
        """Convert target activity error into route-owned online credit.

        The caller supplies the expected target activity from the current
        experience, not a precomputed route score.  The network derives the
        prediction error and available-resource fraction, then optionally
        derives holdout transfer from the same unseen experience.  This keeps
        the route learner connected to a real runtime outcome instead of an
        evaluator-only manual feedback path.
        """

        if self._cooperation_learner is None:
            raise RuntimeError("cross-region cooperation learner is not attached")
        budget = float(resource_budget)
        if budget <= 0.0:
            raise ValueError("cross-region resource_budget must be positive")
        errors: dict[str, float] = {}
        for connection_id in tuple(str(item) for item in active_connection_ids):
            if connection_id not in self._connections:
                raise ValueError(f"unknown cross-region connection: {connection_id}")
            _, target_id, _ = self._connections[connection_id]
            actual = actual_activities.get(target_id)
            expected = expected_activities.get(target_id)
            if actual is None or expected is None:
                raise ValueError(
                    f"route credit requires actual and expected activity for target {target_id}"
                )
            if actual.shape != expected.shape:
                raise ValueError(f"route credit activity shape mismatch for target {target_id}")
            prediction_error = float(torch.mean(torch.abs(actual - expected)).clamp(0.0, 1.0).item())
            resource_state = budget / (budget + self._cooperation_learner.resource_cost(connection_id))
            route = self._cooperation_learner.route_state(connection_id)
            self.observe_connection(
                connection_id,
                prediction_error=prediction_error,
                holdout_transfer=(1.0 - prediction_error if holdout else route.holdout_transfer),
                resource_state=resource_state,
                selected=True,
            )
            errors[connection_id] = prediction_error
        return errors

    def selected_connection_ids(
        self,
        *,
        resource_budget: float = 1.0,
        max_connections: int = 1,
    ) -> tuple[str, ...]:
        """Return the learned feasible routes, excluding functional lesions."""

        candidates = tuple(
            connection_id
            for connection_id in self.connection_ids
            if connection_id not in self._lesioned_connections
            and self._connections[connection_id][0] not in self._lesioned_regions
            and self._connections[connection_id][1] not in self._lesioned_regions
        )
        if self._cooperation_learner is None:
            return candidates[: int(max_connections)]
        return self._cooperation_learner.select(
            candidates,
            resource_budget=resource_budget,
            max_connections=max_connections,
        )

    def step(
        self,
        external_inputs: Mapping[str, torch.Tensor],
        *,
        connection_ids: Sequence[str] | None = None,
        resource_budget: float = 1.0,
        max_connections: int = 1,
        expected_activities: Mapping[str, torch.Tensor] | None = None,
        holdout: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Run one feed-forward tick with optional learned route selection."""

        active_connection_ids = (
            self.selected_connection_ids(
                resource_budget=resource_budget,
                max_connections=max_connections,
            )
            if connection_ids is None
            else tuple(str(item) for item in connection_ids)
        )
        if len(set(active_connection_ids)) != len(active_connection_ids):
            raise ValueError("cross-region step cannot contain duplicate connections")
        for connection_id in active_connection_ids:
            if connection_id not in self._connections:
                raise ValueError(f"unknown cross-region connection: {connection_id}")

        activities: dict[str, torch.Tensor] = {}
        for region_id in self.execution_order:
            region = self._regions[region_id]
            external = external_inputs.get(region_id)
            if external is None:
                external = torch.zeros(region.input_dim, device=region.device)
            cross_drive = torch.zeros(region.unit_count, device=region.device)
            for connection_id in active_connection_ids:
                if connection_id in self._lesioned_connections:
                    continue
                source_id, target_id, connection = self._connections[connection_id]
                if target_id == region_id and source_id in activities:
                    cross_drive.add_(connection.forward(activities[source_id]))
            activities[region_id] = region.step(external, additional_drive=cross_drive)
            if region_id in self._lesioned_regions:
                activities[region_id].zero_()
                region.activity.zero_()
        if expected_activities is not None:
            self.credit_routes_from_outcome(
                activities,
                expected_activities,
                active_connection_ids=active_connection_ids,
                resource_budget=resource_budget,
                holdout=holdout,
            )
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
                    "resource_cost": self._connection_resource_costs[connection_id],
                    "synapses": synapses.to_payload(),
                }
                for connection_id, (source_id, target_id, synapses) in self._connections.items()
            },
            "lesioned_connections": list(self._lesioned_connections),
            "lesioned_regions": list(self._lesioned_regions),
            "cooperation_learner": (
                None
                if self._cooperation_learner is None
                else self._cooperation_learner.to_payload()
            ),
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
        self._connection_resource_costs = {}
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
            self._connection_resource_costs[str(connection_id)] = float(
                connection_payload.get("resource_cost", 1.0)
            )
        lesioned = {str(item) for item in payload.get("lesioned_connections", ())}
        if not lesioned.issubset(set(self._connections)):
            raise ValueError("lesioned connection identity is not present")
        self._lesioned_connections = lesioned
        lesioned_regions = {str(item) for item in payload.get("lesioned_regions", ())}
        if not lesioned_regions.issubset(set(self._regions)):
            raise ValueError("lesioned region identity is not present")
        self._lesioned_regions = lesioned_regions
        learner_payload = payload.get("cooperation_learner")
        self._cooperation_learner = (
            None
            if learner_payload is None
            else CrossRegionCooperationLearner.from_payload(learner_payload)
        )
        if self._cooperation_learner is not None:
            if set(self._cooperation_learner.route_ids) != set(self._connections):
                raise ValueError("cross-region learning route identities do not match network")

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
