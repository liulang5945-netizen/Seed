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

    def propose_region_prune(
        self,
        *,
        region_id: str,
        evidence_ids: Sequence[str],
        parent_checkpoint_id: str | None = None,
        resource_cost: int = 1,
    ) -> StructuralTopologyProposal:
        """Describe removal of one region without mutating the live network."""

        region = self._region(region_id)
        if len(self._regions) <= 1:
            raise ValueError("adaptive neuron network cannot prune its only region")
        return StructuralTopologyProposal(
            proposal_id=f"topology:{region.region_id}:region:prune",
            substrate_id=region.region_id,
            target_kind="region",
            operation="prune",
            specification=(
                ("region_id", region.region_id),
                ("topology_role", "region_prune"),
            ),
            evidence_ids=tuple(str(item) for item in evidence_ids),
            parent_checkpoint_id=parent_checkpoint_id,
            resource_cost=int(resource_cost),
        )

    def propose_region_split(
        self,
        *,
        region_id: str,
        first_unit_count: int,
        new_region_id: str | None = None,
        evidence_ids: Sequence[str],
        parent_checkpoint_id: str | None = None,
        resource_cost: int = 1,
    ) -> StructuralTopologyProposal:
        """Describe a stable partition of one region with explicit route migration."""

        region = self._region(region_id)
        if region.unit_count < 2:
            raise ValueError("region split requires at least two units")
        split_count = int(first_unit_count)
        if not 0 < split_count < region.unit_count:
            raise ValueError("region split first_unit_count must leave both partitions non-empty")
        retained_id = region.region_id
        candidate = (
            f"{retained_id}.split.1"
            if new_region_id is None
            else str(new_region_id)
        )
        if not candidate or candidate == retained_id or candidate in self._regions:
            raise ValueError("region split new_region_id must be a fresh identity")
        connection_migrations: list[tuple[str, tuple[str, ...]]] = []
        for connection_id, (source_id, target_id, _) in self._connections.items():
            if retained_id not in {source_id, target_id}:
                continue
            source_ids = (
                (retained_id, candidate) if source_id == retained_id else (source_id,)
            )
            target_ids = (
                (retained_id, candidate) if target_id == retained_id else (target_id,)
            )
            child_ids = tuple(
                f"connection:{child_source}->{child_target}"
                for child_source in source_ids
                for child_target in target_ids
            )
            connection_migrations.append((connection_id, child_ids))
        return StructuralTopologyProposal(
            proposal_id=f"topology:{retained_id}:split:{candidate}",
            substrate_id=retained_id,
            target_kind="region",
            operation="split",
            specification=(
                ("parent_region_id", retained_id),
                ("retained_region_id", retained_id),
                ("new_region_id", candidate),
                ("retained_unit_ids", region.unit_ids[:split_count]),
                ("new_unit_ids", region.unit_ids[split_count:]),
                ("parent_unit_count", region.unit_count),
                ("connection_migrations", tuple(connection_migrations)),
                ("topology_role", "region_split"),
            ),
            evidence_ids=tuple(str(item) for item in evidence_ids),
            parent_checkpoint_id=parent_checkpoint_id,
            resource_cost=int(resource_cost),
        )

    def propose_region_merge(
        self,
        *,
        region_ids: Sequence[str],
        evidence_ids: Sequence[str],
        parent_checkpoint_id: str | None = None,
        resource_cost: int = 1,
    ) -> StructuralTopologyProposal:
        """Describe a two-region merge with explicit external route aggregation."""

        selected = tuple(str(item) for item in region_ids)
        if len(selected) != 2 or len(set(selected)) != 2:
            raise ValueError("region merge requires two distinct regions")
        first = self._region(selected[0])
        second = self._region(selected[1])
        if first.input_dim != second.input_dim or first.fan_in != second.fan_in:
            raise ValueError("region merge requires matching input dimensions and fan_in")
        if first.input_source_id != second.input_source_id:
            raise ValueError("region merge requires matching input source identities")
        if first.dynamics.to_payload() != second.dynamics.to_payload():
            raise ValueError("region merge requires matching region dynamics")
        if selected[0] in self._lesioned_regions or selected[1] in self._lesioned_regions:
            raise ValueError("region merge cannot absorb a lesioned region")
        merge_set = set(selected)
        for source_id, target_id, _ in self._connections.values():
            if source_id in merge_set and target_id in merge_set:
                raise ValueError("region merge cannot absorb an internal cross-region connection")
        merged_unit_ids = first.unit_ids + second.unit_ids
        if len(set(merged_unit_ids)) != len(merged_unit_ids):
            raise ValueError("region merge requires globally unique unit identities")
        connection_merges: dict[str, list[str]] = {}
        for connection_id, (source_id, target_id, _) in self._connections.items():
            if source_id not in merge_set and target_id not in merge_set:
                continue
            merged_source = first.region_id if source_id in merge_set else source_id
            merged_target = first.region_id if target_id in merge_set else target_id
            new_connection_id = f"connection:{merged_source}->{merged_target}"
            connection_merges.setdefault(new_connection_id, []).append(connection_id)
        return StructuralTopologyProposal(
            proposal_id=f"topology:{first.region_id}+{second.region_id}:merge",
            substrate_id=first.region_id,
            target_kind="region",
            operation="merge",
            specification=(
                ("region_ids", selected),
                ("retained_region_id", first.region_id),
                ("merged_unit_ids", merged_unit_ids),
                ("connection_merges", tuple(
                    (new_id, tuple(old_ids))
                    for new_id, old_ids in connection_merges.items()
                )),
                ("topology_role", "region_merge"),
            ),
            evidence_ids=tuple(str(item) for item in evidence_ids),
            parent_checkpoint_id=parent_checkpoint_id,
            resource_cost=int(resource_cost),
        )

    def propose_connection_prune(
        self,
        *,
        connection_id: str,
        evidence_ids: Sequence[str],
        parent_checkpoint_id: str | None = None,
        resource_cost: int = 1,
    ) -> StructuralTopologyProposal:
        """Describe removal of one explicit cross-region connection."""

        key = str(connection_id)
        try:
            source_id, target_id, connection = self._connections[key]
        except KeyError as exc:
            raise ValueError(f"unknown cross-region connection: {connection_id}") from exc
        return StructuralTopologyProposal(
            proposal_id=f"topology:{key}:prune",
            substrate_id=key,
            target_kind="region",
            operation="prune",
            specification=(
                ("connection_id", key),
                ("source_region_id", source_id),
                ("target_region_id", target_id),
                ("source_unit_count", self._region(source_id).unit_count),
                ("target_unit_count", self._region(target_id).unit_count),
                ("edge_count", connection.edge_count),
                ("topology_role", "cross_region_connection_prune"),
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

    @torch.no_grad()
    def apply_region_prune(
        self,
        proposal: StructuralTopologyProposal,
    ) -> bool:
        """Remove one region and every owned cross-region projection touching it."""

        if proposal.status != "pending":
            raise ValueError("only pending region proposals can be applied")
        if proposal.target_kind != "region" or proposal.operation != "prune":
            raise ValueError("proposal is not a region prune")
        if dict(proposal.specification).get("topology_role") != "region_prune":
            raise ValueError("proposal is not a region prune")
        specification = dict(proposal.specification)
        region_id = str(specification.get("region_id", ""))
        if region_id != proposal.substrate_id:
            raise ValueError("region identity does not match proposal")
        if len(self._regions) <= 1:
            raise ValueError("adaptive neuron network cannot prune its only region")
        self._region(region_id)
        removed_connections = tuple(
            connection_id
            for connection_id, (source_id, target_id, _) in self._connections.items()
            if region_id in {source_id, target_id}
        )
        for connection_id in removed_connections:
            self._connections.pop(connection_id)
            self._connection_resource_costs.pop(connection_id, None)
            if self._cooperation_learner is not None:
                self._cooperation_learner.unregister_connection(connection_id)
        self._regions.pop(region_id)
        self.execution_order = tuple(
            item for item in self.execution_order if item != region_id
        )
        self._lesioned_regions.discard(region_id)
        return True

    @staticmethod
    @torch.no_grad()
    def _partition_region(
        region: AdaptiveNeuronRegion,
        *,
        region_id: str,
        unit_ids: Sequence[str],
        generator: torch.Generator,
    ) -> AdaptiveNeuronRegion:
        """Copy one identity-preserving unit partition into a new region object."""

        selected = tuple(str(item) for item in unit_ids)
        old_indices = tuple(region.unit_index(item) for item in selected)
        recurrent_fan_in = (
            None if region.recurrent is None else region.recurrent.row_fan_in
        )
        partition = AdaptiveNeuronRegion(
            region_id=region_id,
            input_dim=region.input_dim,
            unit_ids=selected,
            fan_in=region.fan_in,
            generator=generator,
            input_source_id=region.input_source_id,
            dynamics=region.dynamics,
            recurrent_fan_in=recurrent_fan_in,
            device=region.device,
        )
        row_index = torch.tensor(old_indices, dtype=torch.long, device=region.device)
        partition.incoming.pre_index.copy_(region.incoming.pre_index.index_select(0, row_index))
        partition.incoming.edge_weight.copy_(region.incoming.edge_weight.index_select(0, row_index))
        for target_index, source_index in enumerate(old_indices):
            partition.membrane[target_index] = region.membrane[source_index]
            partition.activity[target_index] = region.activity[source_index]
            partition.trace[target_index] = region.trace[source_index]
            partition.threshold[target_index] = region.threshold[source_index]
        if partition.recurrent is not None and region.recurrent is not None:
            old_to_new = {old: new for new, old in enumerate(old_indices)}
            partition.recurrent.edge_weight.zero_()
            for new_row, old_row in enumerate(old_indices):
                new_slot = 0
                for slot in range(region.recurrent.row_fan_in):
                    old_pre = int(region.recurrent.pre_index[old_row, slot].item())
                    if old_pre not in old_to_new or new_slot >= partition.recurrent.row_fan_in:
                        continue
                    partition.recurrent.pre_index[new_row, new_slot] = old_to_new[old_pre]
                    partition.recurrent.edge_weight[new_row, new_slot] = (
                        region.recurrent.edge_weight[old_row, slot]
                    )
                    new_slot += 1
        partition._lesioned_units = {
            unit_id for unit_id in selected if unit_id in region._lesioned_units
        }
        return partition

    @torch.no_grad()
    def apply_region_split(
        self,
        proposal: StructuralTopologyProposal,
        *,
        generator: torch.Generator,
    ) -> bool:
        """Split a region while preserving units, local state and affected routes."""

        if proposal.status != "pending":
            raise ValueError("only pending topology proposals can be applied")
        if proposal.target_kind != "region" or proposal.operation != "split":
            raise ValueError("proposal is not a region split")
        if dict(proposal.specification).get("topology_role") != "region_split":
            raise ValueError("proposal is not a region split")
        specification = dict(proposal.specification)
        parent_id = str(specification.get("parent_region_id", ""))
        retained_id = str(specification.get("retained_region_id", ""))
        new_region_id = str(specification.get("new_region_id", ""))
        if parent_id != proposal.substrate_id or retained_id != parent_id:
            raise ValueError("region split parent identity does not match proposal")
        if not new_region_id or new_region_id in self._regions:
            raise ValueError("region split child identity is not fresh")
        parent = self._region(parent_id)
        retained_units = tuple(str(item) for item in specification.get("retained_unit_ids", ()))
        new_units = tuple(str(item) for item in specification.get("new_unit_ids", ()))
        if (
            not retained_units
            or not new_units
            or retained_units + new_units != parent.unit_ids
            or int(specification.get("parent_unit_count", -1)) != parent.unit_count
        ):
            raise ValueError("region split unit partition has drifted")
        migration_payload = specification.get("connection_migrations", ())
        if not isinstance(migration_payload, (tuple, list)):
            raise ValueError("region split connection migrations must be a sequence")
        expected_migrations: dict[str, tuple[str, ...]] = {}
        for item in migration_payload:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError("region split connection migration must be a pair")
            old_id = str(item[0])
            child_ids = tuple(str(child_id) for child_id in item[1])
            if not old_id or not child_ids or old_id in expected_migrations:
                raise ValueError("region split connection migration identities are invalid")
            expected_migrations[old_id] = child_ids
        actual_touched = {
            connection_id: (source_id, target_id, connection)
            for connection_id, (source_id, target_id, connection) in self._connections.items()
            if parent_id in {source_id, target_id}
        }
        if set(expected_migrations) != set(actual_touched):
            raise ValueError("region split connection migrations do not match current topology")
        retained = self._partition_region(
            parent,
            region_id=retained_id,
            unit_ids=retained_units,
            generator=generator,
        )
        child = self._partition_region(
            parent,
            region_id=new_region_id,
            unit_ids=new_units,
            generator=generator,
        )
        migrated_connections: dict[str, tuple[str, str, SparseSynapses]] = {}
        migrated_costs: dict[str, float] = {}
        migrated_lesions: set[str] = set()
        for old_id, child_ids in expected_migrations.items():
            source_id, target_id, old_connection = actual_touched[old_id]
            source_ids = (
                (retained_id, new_region_id) if source_id == parent_id else (source_id,)
            )
            target_ids = (
                (retained_id, new_region_id) if target_id == parent_id else (target_id,)
            )
            expected_child_ids = tuple(
                f"connection:{child_source}->{child_target}"
                for child_source in source_ids
                for child_target in target_ids
            )
            if child_ids != expected_child_ids:
                raise ValueError("region split connection child identities have drifted")
            source_partitions = {
                retained_id: retained,
                new_region_id: child,
            }
            old_source = parent if source_id == parent_id else self._region(source_id)
            old_target = parent if target_id == parent_id else self._region(target_id)
            child_pairs = tuple(
                (child_source_id, child_target_id)
                for child_source_id in source_ids
                for child_target_id in target_ids
            )
            for (child_source_id, child_target_id), child_connection_id in zip(
                child_pairs,
                child_ids,
                strict=True,
            ):
                child_source = (
                    source_partitions[child_source_id]
                    if child_source_id in source_partitions
                    else self._region(child_source_id)
                )
                child_target = (
                    source_partitions[child_target_id]
                    if child_target_id in source_partitions
                    else self._region(child_target_id)
                )
                migrated = SparseSynapses(
                    child_target.unit_count,
                    child_source.unit_count,
                    old_connection.row_fan_in,
                    generator=generator,
                    init_scale=child_target.dynamics.weight_init_scale,
                    max_weight_norm=old_connection.max_weight_norm,
                    device=child_target.device,
                )
                migrated.edge_weight.zero_()
                old_source_units = old_source.unit_ids
                old_target_units = old_target.unit_ids
                child_source_unit_index = {
                    unit_id: index
                    for index, unit_id in enumerate(child_source.unit_ids)
                }
                child_target_unit_index = {
                    unit_id: index
                    for index, unit_id in enumerate(child_target.unit_ids)
                }
                target_unit_ids = child_target.unit_ids
                old_target_index = {unit_id: index for index, unit_id in enumerate(old_target_units)}
                for child_target_unit in target_unit_ids:
                    new_target_index = child_target_unit_index[child_target_unit]
                    old_target_index_value = old_target_index[child_target_unit]
                    new_slot = 0
                    for old_slot in range(old_connection.row_fan_in):
                        old_source_index_value = int(
                            old_connection.pre_index[old_target_index_value, old_slot].item()
                        )
                        old_source_unit = old_source_units[old_source_index_value]
                        if old_source_unit not in child_source_unit_index:
                            continue
                        if new_slot >= migrated.row_fan_in:
                            break
                        migrated.pre_index[new_target_index, new_slot] = (
                            child_source_unit_index[old_source_unit]
                        )
                        migrated.edge_weight[new_target_index, new_slot] = (
                            old_connection.edge_weight[old_target_index_value, old_slot]
                        )
                        new_slot += 1
                migrated_connections[child_connection_id] = (
                    child_source_id,
                    child_target_id,
                    migrated,
                )
                migrated_costs[child_connection_id] = self._connection_resource_costs.get(
                    old_id,
                    1.0,
                )
                if old_id in self._lesioned_connections:
                    migrated_lesions.add(child_connection_id)
                if (
                    self._cooperation_learner is not None
                    and child_connection_id != old_id
                ):
                    self._cooperation_learner.fork_connection(old_id, child_connection_id)
        self._regions[retained_id] = retained
        self._regions[new_region_id] = child
        split_order: list[str] = []
        for item in self.execution_order:
            if item == parent_id:
                split_order.extend((retained_id, new_region_id))
            else:
                split_order.append(item)
        self.execution_order = tuple(split_order)
        for old_id in actual_touched:
            self._connections.pop(old_id)
            self._connection_resource_costs.pop(old_id, None)
            self._lesioned_connections.discard(old_id)
            if (
                self._cooperation_learner is not None
                and old_id not in migrated_connections
            ):
                self._cooperation_learner.unregister_connection(old_id)
        self._connections.update(migrated_connections)
        self._connection_resource_costs.update(migrated_costs)
        self._lesioned_connections.update(migrated_lesions)
        return True

    @staticmethod
    @torch.no_grad()
    def _merge_regions(
        first: AdaptiveNeuronRegion,
        second: AdaptiveNeuronRegion,
        *,
        generator: torch.Generator,
    ) -> AdaptiveNeuronRegion:
        """Combine two compatible regions while retaining both unit coordinate sets."""

        merged_unit_ids = first.unit_ids + second.unit_ids
        recurrent_fan_in = (
            None
            if first.recurrent is None
            else max(
                first.recurrent.row_fan_in,
                1 if second.recurrent is None else second.recurrent.row_fan_in,
            )
        )
        merged = AdaptiveNeuronRegion(
            region_id=first.region_id,
            input_dim=first.input_dim,
            unit_ids=merged_unit_ids,
            fan_in=first.fan_in,
            generator=generator,
            input_source_id=first.input_source_id,
            dynamics=first.dynamics,
            recurrent_fan_in=recurrent_fan_in,
            device=first.device,
        )
        incoming_rows = first.incoming.out_features + second.incoming.out_features
        if incoming_rows != merged.unit_count:
            raise ValueError("region merge incoming dimensions are inconsistent")
        merged.incoming.pre_index[: first.unit_count].copy_(first.incoming.pre_index)
        merged.incoming.edge_weight[: first.unit_count].copy_(first.incoming.edge_weight)
        merged.incoming.pre_index[first.unit_count :].copy_(second.incoming.pre_index)
        merged.incoming.edge_weight[first.unit_count :].copy_(second.incoming.edge_weight)
        merged.membrane = torch.cat((first.membrane, second.membrane)).to(merged.device)
        merged.activity = torch.cat((first.activity, second.activity)).to(merged.device)
        merged.trace = torch.cat((first.trace, second.trace)).to(merged.device)
        merged.threshold = torch.cat((first.threshold, second.threshold)).to(merged.device)
        if merged.recurrent is not None:
            merged.recurrent.edge_weight.zero_()
            for source_region, offset in ((first, 0), (second, first.unit_count)):
                if source_region.recurrent is None:
                    continue
                for row in range(source_region.unit_count):
                    new_row = offset + row
                    new_slot = 0
                    for slot in range(source_region.recurrent.row_fan_in):
                        if new_slot >= merged.recurrent.row_fan_in:
                            break
                        old_pre = int(source_region.recurrent.pre_index[row, slot].item())
                        merged.recurrent.pre_index[new_row, new_slot] = offset + old_pre
                        merged.recurrent.edge_weight[new_row, new_slot] = (
                            source_region.recurrent.edge_weight[row, slot]
                        )
                        new_slot += 1
        merged._lesioned_units = first._lesioned_units | second._lesioned_units
        return merged

    @staticmethod
    @torch.no_grad()
    def _merge_connection_projection(
        old_connections: Sequence[tuple[str, str, SparseSynapses]],
        *,
        source_region: AdaptiveNeuronRegion,
        target_region: AdaptiveNeuronRegion,
        source_regions: Mapping[str, AdaptiveNeuronRegion],
        target_regions: Mapping[str, AdaptiveNeuronRegion],
        generator: torch.Generator,
    ) -> SparseSynapses:
        """Aggregate old sparse supports into one explicit merged projection."""

        rows: dict[int, dict[int, float]] = {}
        for source_id, target_id, connection in old_connections:
            old_source = source_regions[source_id]
            old_target = target_regions[target_id]
            source_index = {
                unit_id: index for index, unit_id in enumerate(source_region.unit_ids)
            }
            target_index = {
                unit_id: index for index, unit_id in enumerate(target_region.unit_ids)
            }
            for old_target_index, target_unit_id in enumerate(old_target.unit_ids):
                new_target_index = target_index[target_unit_id]
                row = rows.setdefault(new_target_index, {})
                for slot in range(connection.row_fan_in):
                    old_source_index = int(connection.pre_index[old_target_index, slot].item())
                    source_unit_id = old_source.unit_ids[old_source_index]
                    new_source_index = source_index[source_unit_id]
                    row[new_source_index] = row.get(new_source_index, 0.0) + float(
                        connection.edge_weight[old_target_index, slot].item()
                    )
        fan_in = max((len(row) for row in rows.values()), default=1)
        merged = SparseSynapses(
            target_region.unit_count,
            source_region.unit_count,
            fan_in,
            generator=generator,
            init_scale=target_region.dynamics.weight_init_scale,
            max_weight_norm=max(
                connection.max_weight_norm for _, _, connection in old_connections
            ),
            device=target_region.device,
        )
        merged.edge_weight.zero_()
        for row_index, supports in rows.items():
            for slot, (source_index, weight) in enumerate(sorted(supports.items())):
                if slot >= merged.row_fan_in:
                    break
                merged.pre_index[row_index, slot] = source_index
                merged.edge_weight[row_index, slot] = weight
        return merged

    @torch.no_grad()
    def apply_region_merge(
        self,
        proposal: StructuralTopologyProposal,
        *,
        generator: torch.Generator,
    ) -> bool:
        """Merge compatible regions and aggregate every affected external route."""

        if proposal.status != "pending":
            raise ValueError("only pending topology proposals can be applied")
        if proposal.target_kind != "region" or proposal.operation != "merge":
            raise ValueError("proposal is not a region merge")
        specification = dict(proposal.specification)
        if specification.get("topology_role") != "region_merge":
            raise ValueError("proposal is not a region merge")
        selected = tuple(str(item) for item in specification.get("region_ids", ()))
        if len(selected) != 2 or proposal.substrate_id != selected[0]:
            raise ValueError("region merge identities do not match proposal")
        first = self._region(selected[0])
        second = self._region(selected[1])
        expected_units = tuple(str(item) for item in specification.get("merged_unit_ids", ()))
        if expected_units != first.unit_ids + second.unit_ids:
            raise ValueError("region merge unit identities have drifted")
        if first.input_dim != second.input_dim or first.fan_in != second.fan_in:
            raise ValueError("region merge dimensions have drifted")
        if first.input_source_id != second.input_source_id:
            raise ValueError("region merge input sources have drifted")
        if first.dynamics.to_payload() != second.dynamics.to_payload():
            raise ValueError("region merge dynamics have drifted")
        merge_set = set(selected)
        if first.region_id in self._lesioned_regions or second.region_id in self._lesioned_regions:
            raise ValueError("region merge cannot absorb a lesioned region")
        for source_id, target_id, _ in self._connections.values():
            if source_id in merge_set and target_id in merge_set:
                raise ValueError("region merge cannot absorb an internal cross-region connection")
        payload = specification.get("connection_merges", ())
        if not isinstance(payload, (tuple, list)):
            raise ValueError("region merge connection migrations must be a sequence")
        expected_groups: dict[str, tuple[str, ...]] = {}
        for item in payload:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError("region merge connection migration must be a pair")
            new_id = str(item[0])
            old_ids = tuple(str(old_id) for old_id in item[1])
            if not new_id or not old_ids or len(set(old_ids)) != len(old_ids):
                raise ValueError("region merge connection migration identities are invalid")
            expected_groups[new_id] = old_ids
        actual_groups: dict[str, list[str]] = {}
        actual_connections = dict(self._connections)
        for connection_id, (source_id, target_id, _) in actual_connections.items():
            if source_id not in merge_set and target_id not in merge_set:
                continue
            merged_source = first.region_id if source_id in merge_set else source_id
            merged_target = first.region_id if target_id in merge_set else target_id
            actual_groups.setdefault(
                f"connection:{merged_source}->{merged_target}",
                [],
            ).append(connection_id)
        if {key: tuple(value) for key, value in actual_groups.items()} != expected_groups:
            raise ValueError("region merge connection migrations do not match topology")
        merged_region = self._merge_regions(first, second, generator=generator)
        migrated_connections: dict[str, tuple[str, str, SparseSynapses]] = {}
        migrated_costs: dict[str, float] = {}
        migrated_lesions: set[str] = set()
        for new_id, old_ids in expected_groups.items():
            old_entries = tuple(actual_connections[old_id] for old_id in old_ids)
            source_id = first.region_id if old_entries[0][0] in merge_set else old_entries[0][0]
            target_id = first.region_id if old_entries[0][1] in merge_set else old_entries[0][1]
            source_region = (
                merged_region
                if source_id == first.region_id
                else self._region(source_id)
            )
            target_region = (
                merged_region
                if target_id == first.region_id
                else self._region(target_id)
            )
            source_regions = {
                old_source_id: self._region(old_source_id)
                for old_source_id, _, _ in old_entries
            }
            target_regions = {
                old_target_id: self._region(old_target_id)
                for _, old_target_id, _ in old_entries
            }
            projection = self._merge_connection_projection(
                tuple((old_source_id, old_target_id, connection) for old_source_id, old_target_id, connection in old_entries),
                source_region=source_region,
                target_region=target_region,
                source_regions=source_regions,
                target_regions=target_regions,
                generator=generator,
            )
            migrated_connections[new_id] = (source_id, target_id, projection)
            migrated_costs[new_id] = max(
                self._connection_resource_costs.get(old_id, 1.0)
                for old_id in old_ids
            )
            if all(old_id in self._lesioned_connections for old_id in old_ids):
                migrated_lesions.add(new_id)
            if self._cooperation_learner is not None:
                self._cooperation_learner.merge_connections(
                    old_ids,
                    new_id,
                    resource_cost=migrated_costs[new_id],
                )
        for connection_id in actual_groups.values():
            for old_id in connection_id:
                self._connections.pop(old_id)
                self._connection_resource_costs.pop(old_id, None)
                self._lesioned_connections.discard(old_id)
        self._connections.update(migrated_connections)
        self._connection_resource_costs.update(migrated_costs)
        self._lesioned_connections.update(migrated_lesions)
        merged_regions: dict[str, AdaptiveNeuronRegion] = {}
        for region_id, region in self._regions.items():
            if region_id == first.region_id:
                merged_regions[region_id] = merged_region
            elif region_id == second.region_id:
                continue
            else:
                merged_regions[region_id] = region
        self._regions = merged_regions
        self.execution_order = tuple(
            region_id
            for region_id in self.execution_order
            if region_id != second.region_id
        )
        return True

    @torch.no_grad()
    def apply_connection_prune(
        self,
        proposal: StructuralTopologyProposal,
    ) -> bool:
        """Remove one explicit cross-region projection while retaining regions."""

        if proposal.status != "pending":
            raise ValueError("only pending topology proposals can be applied")
        if proposal.target_kind != "region" or proposal.operation != "prune":
            raise ValueError("proposal is not a cross-region connection prune")
        if (
            dict(proposal.specification).get("topology_role")
            != "cross_region_connection_prune"
        ):
            raise ValueError("proposal is not a cross-region connection prune")
        specification = dict(proposal.specification)
        connection_id = str(specification.get("connection_id", ""))
        if connection_id != proposal.substrate_id:
            raise ValueError("connection identity does not match proposal")
        try:
            self._connections[connection_id]
        except KeyError as exc:
            raise ValueError(f"unknown cross-region connection: {connection_id}") from exc
        self._connections.pop(connection_id)
        self._connection_resource_costs.pop(connection_id, None)
        self._lesioned_connections.discard(connection_id)
        if self._cooperation_learner is not None:
            self._cooperation_learner.unregister_connection(connection_id)
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
