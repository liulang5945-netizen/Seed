"""Compatibility adapter exposing TSK-v8 through Taiji v1 contracts."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch

from .affordance import LearnedAffordanceFeatures, WorldAffordanceGroundingProducer
from .concept_formation import ConceptFormationOrgan, ConceptMatch
from .content_selection import (
    ContentCandidate,
    ContentSelectionContext,
    ContentSelectionDecision,
    ContentSelector,
)
from .contracts import (
    CONTRACT_FORMAT,
    ActionIntent,
    Assembly,
    CognitiveState,
    Concept,
    DevelopmentState,
    EnvironmentCapability,
    EpisodicMemoryRecord,
    Event,
    Goal,
    GoalState,
    HomeostaticState,
    LearningState,
    NativeCheckpoint,
    Observation,
    Outcome,
    PerceptEvent,
    PlanCandidate,
    PlanningRecoveryState,
    PlanState,
    RecoveryBranchState,
    RecoveryBudgetState,
    SelfState,
    StructuralGrowthRequest,
    StructuralTopologyProposal,
    WorkingMemoryItem,
    WorkspaceCandidate,
    WorkspaceSelection,
    WorkspaceState,
    WorldAction,
    WorldAffordance,
    WorldCalibrationTrace,
    WorldPredictionRecord,
    WorldState,
    WorldTransition,
)
from .contracts import MemoryState as NativeMemoryState
from .cross_region_learning import CrossRegionCooperationLearner
from .environment import EnvironmentOutcome, TaijiEnvironment, TaijiToolEnvironment
from .episodic_memory import EpisodicMemoryStore
from .executive import (
    ExecutiveCandidate,
    ExecutiveContext,
    ExecutiveController,
    ExecutiveDecision,
)
from .fabric import TaijiFabric
from .generation import (
    ContentPlan,
    ExpressionPlan,
    GenerationController,
    GenerationTrace,
    ToolCall,
)
from .homeostasis import HomeostaticController, HomeostaticDrive
from .input_boundary import InputFrame, InputTrace
from .language_organ import (
    LanguageBackendRegistry,
    LanguageEmission,
    LanguageOrgan,
    LanguageProviderArtifact,
    LanguageValidation,
    StructuredTextLanguageOrgan,
)
from .model import Taiji
from .neuron_network import AdaptiveNeuronNetwork
from .neuron_region import AdaptiveNeuronRegion
from .perception import LearnedPerception
from .planning import (
    GoalPlanner,
    ImaginedRollout,
    PlanningCandidate,
    PlanningDecision,
    RecoveryPortfolio,
    RecoveryPortfolioArchive,
    RecoveryReaderContribution,
    RecoveryReaderDependency,
    RecoveryReaderDependencyGraph,
    RecoveryRolloutLineage,
    RecoveryStrategyApproval,
    RecoveryStrategyLedger,
    RolloutDecision,
)
from .procedural_memory import ProceduralMemoryLearner, ProceduralSequenceLearner
from .semantic_memory import SemanticMemoryLearner
from .state import TaijiDecision, TaijiOutcome, TaijiStep
from .structural_growth import (
    AdaptiveStructuralGrowthController,
    AdaptiveStructuralPruningController,
    StructuralGrowthDecision,
    StructuralMaintenanceResult,
    StructuralProposalCandidate,
    StructuralPruningDecision,
    StructuralRuntimeObservation,
)
from .workspace import WorkspaceRouter
from .world_learning import WorldDynamicsLearner, WorldSchema, WorldSchemaRegistry


def _checkpoint_digest(payload: Mapping[str, Any]) -> str:
    """Hash a reader checkpoint without depending on pickle ordering."""

    digest = hashlib.sha256()

    def update(value: Any) -> None:
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(b"tensor:")
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
            digest.update(tensor.numpy().tobytes())
            return
        if isinstance(value, Mapping):
            digest.update(b"mapping:")
            for key in sorted(value, key=str):
                update(str(key))
                update(value[key])
            return
        if isinstance(value, (tuple, list)):
            digest.update(b"sequence:")
            for item in value:
                update(item)
            return
        digest.update(repr(value).encode("utf-8"))

    update(payload)
    return digest.hexdigest()


def _payload_distance(left: Any, right: Any) -> float:
    """Return a deterministic L2-like distance for reader state payloads."""

    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if left.shape != right.shape:
            return 1.0 + float(abs(left.numel() - right.numel()))
        left_tensor = left.detach().cpu().to(dtype=torch.float32)
        right_tensor = right.detach().cpu().to(dtype=torch.float32)
        return float(torch.linalg.vector_norm(left_tensor - right_tensor))
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        keys = set(left) | set(right)
        return float(
            sum(_payload_distance(left.get(key), right.get(key)) ** 2 for key in keys) ** 0.5
        )
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        distance = sum(
            _payload_distance(left_item, right_item) ** 2
            for left_item, right_item in zip(left, right, strict=False)
        )
        distance += float(abs(len(left) - len(right)))
        return float(distance**0.5)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right))
    return 0.0 if left == right else 1.0


@dataclass(frozen=True)
class _PendingExecutiveCredit:
    """Keep an executive decision's causal context across a replan."""

    decision: ExecutiveDecision
    affordance: WorldAffordance | None
    percept_features: torch.Tensor | None
    world_latent: torch.Tensor | None
    world_uncertainty: float
    learn: bool


class TSKV8Adapter(Taiji):
    """Keep the TSK-v8 API while making v1 ownership explicit.

    This subclass is intentional: old callers still see a ``Taiji`` and old
    ``taiji-native-v8`` checkpoints remain readable.  New callers can use
    ``native_checkpoint`` and ``cognitive_snapshot`` without treating the
    kernel's byte prediction state as the complete v1 cognitive state.
    """

    ADAPTER_NAME = "tsk-v8"
    NATIVE_CHECKPOINT_FORMAT = CONTRACT_FORMAT
    SUPPORTED_INPUT_MODALITIES = frozenset({"text", "text-utf8", "text-byte"})

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.perception = LearnedPerception(self.config, device=self.device)
        self._cognitive_state = self._empty_cognitive_state(self._state.episode_id)
        self._world_dynamics: WorldDynamicsLearner | None = None
        self._workspace_router: WorkspaceRouter | None = None
        self._episodic_memory: EpisodicMemoryStore | None = None
        self._semantic_memory: SemanticMemoryLearner | None = None
        self._procedural_sequence_memory: ProceduralSequenceLearner | None = None
        self._concept_formation = ConceptFormationOrgan(
            similarity_threshold=self.config.concept_similarity_threshold,
            signal_weights=self.config.concept_signal_weights,
            capacity=self.config.concept_capacity,
            plasticity_rate=self.config.concept_plasticity_rate,
            prune_threshold=self.config.concept_prune_threshold,
        )
        self._online_concept_branches: dict[str, tuple[tuple[WorldTransition, float], ...]] = {}
        self._growth_requests: dict[str, StructuralGrowthRequest] = {}
        self._growth_request_snapshots: dict[str, dict[str, Any]] = {}
        self._topology_proposals: dict[str, StructuralTopologyProposal] = {}
        self._topology_parent_snapshots: dict[str, dict[str, Any]] = {}
        self._topology_network_ids: dict[str, str] = {}
        self._neuron_regions: dict[str, AdaptiveNeuronRegion] = {}
        self._neuron_networks: dict[str, AdaptiveNeuronNetwork] = {}
        self._structural_growth_controller: AdaptiveStructuralGrowthController | None = None
        self._structural_pruning_controller: AdaptiveStructuralPruningController | None = None
        self._structural_runtime_tick = 0
        self._structural_runtime_observations: list[StructuralRuntimeObservation] = []
        self._structural_runtime_previous_errors: dict[str, float] = {}
        self._structural_proposal_candidates: dict[str, StructuralProposalCandidate] = {}
        self._structural_candidate_proposals: dict[str, str] = {}
        self._structural_maintenance_results: list[StructuralMaintenanceResult] = []
        self._procedural_memory: ProceduralMemoryLearner | None = None
        self._homeostatic_controller: HomeostaticController | None = None
        self._goal_planner: GoalPlanner | None = None
        self._affordance_features: LearnedAffordanceFeatures | None = None
        self._affordance_grounding: WorldAffordanceGroundingProducer | None = None
        self._executive: ExecutiveController | None = None
        self._last_executive_decision: ExecutiveDecision | None = None
        self._last_executive_prediction_error: float | None = None
        self._last_delayed_executive_prediction_error: float | None = None
        self._last_affordance_prediction_error: float | None = None
        self._last_executive_world_action: WorldAction | None = None
        self._pending_executive_credit: _PendingExecutiveCredit | None = None
        self._planned_rollout: ImaginedRollout | None = None
        self._recovery_portfolio: RecoveryPortfolio | None = None
        self._recovery_archive = RecoveryPortfolioArchive(
            capacity=self.config.recovery_archive_capacity
        )
        self._recovery_strategy_ledger = RecoveryStrategyLedger(
            evidence_threshold=self.config.recovery_strategy_evidence_threshold,
            memory_budget=self.config.recovery_strategy_memory_budget,
            evidence_weight=self.config.recovery_strategy_evidence_weight,
            consistency_weight=self.config.recovery_strategy_consistency_weight,
            resource_weight=self.config.recovery_strategy_resource_weight,
        )
        self._recovery_reader_dependencies = RecoveryReaderDependencyGraph()
        self._recovery_generation = 0
        self._recovery_memory_epochs = 300
        self._recovery_semantic_learning_rate = 0.1
        self._recovery_procedural_learning_rate = 0.1
        self._recovery_memory_rebuild_count = 0
        self._replan_required = False
        self._last_rollout_prediction_error: float | None = None
        self._last_rollout_calibrated_confidence: float | None = None
        self._generation_controller: GenerationController | None = None
        self._last_generation_trace: GenerationTrace | None = None
        self._content_selector: ContentSelector | None = None
        self._last_content_selection: ContentSelectionDecision | None = None
        self._last_content_prediction_error: float | None = None
        self._content_feedback_applied = False
        self._language_backend_registry = LanguageBackendRegistry.default()
        self._language_provider_artifact: LanguageProviderArtifact | None = None
        self._language_organ: LanguageOrgan | None = None
        self._last_language_emission: LanguageEmission | None = None
        self._language_fallback_count = 0
        self._language_fallback_requires_replan = False

    def _empty_cognitive_state(self, episode_id: str) -> CognitiveState:
        empty = torch.empty(0, device=self.device)
        return CognitiveState(
            episode_id=episode_id,
            tick=0,
            observation=None,
            percept=None,
            workspace=WorkspaceState(tick=0, broadcast=empty),
            world=WorldState(tick=0, latent=empty, uncertainty=1.0),
            memory=NativeMemoryState(
                tick=0,
                semantic_context=empty,
                procedural_context=empty,
            ),
            goals=GoalState(tick=0),
            plan=PlanState(tick=0),
            self_state=SelfState(tick=0),
            homeostasis=HomeostaticState(tick=0),
            development=DevelopmentState(
                tick=0,
                structural_budget=self.config.development_structural_budget,
            ),
            learning=LearningState(tick=0),
        )

    @property
    def architecture_name(self) -> str:
        return "Taiji Native Architecture v1 via TSK-v8"

    def cognitive_snapshot(self) -> CognitiveState:
        """Return a detached contract snapshot owned by Taiji."""

        return CognitiveState.from_payload(self._cognitive_state.to_payload(), device=self.device)

    @property
    def concept_formation(self) -> ConceptFormationOrgan:
        """Expose the Taiji-owned concept registry for inspection and lesion tests."""

        return self._concept_formation

    @property
    def growth_requests(self) -> tuple[StructuralGrowthRequest, ...]:
        """Return structural growth decisions owned by the Taiji state."""

        return tuple(self._growth_requests.values())

    @property
    def topology_proposals(self) -> tuple[StructuralTopologyProposal, ...]:
        """Return substrate topology decisions owned by the Taiji state."""

        return tuple(self._topology_proposals.values())

    @property
    def neuron_regions(self) -> tuple[AdaptiveNeuronRegion, ...]:
        """Return explicitly attached adaptive regions owned by Taiji."""

        return tuple(self._neuron_regions.values())

    @property
    def neuron_networks(self) -> tuple[AdaptiveNeuronNetwork, ...]:
        """Return explicitly attached cross-region networks owned by Taiji."""

        return tuple(self._neuron_networks.values())

    @property
    def structural_growth_controller(self) -> AdaptiveStructuralGrowthController | None:
        """Return the optional substrate-driven structural development organ."""

        return self._structural_growth_controller

    @property
    def structural_pruning_controller(self) -> AdaptiveStructuralPruningController | None:
        """Return the optional substrate-driven structural pruning organ."""

        return self._structural_pruning_controller

    @property
    def structural_runtime_observations(self) -> tuple[StructuralRuntimeObservation, ...]:
        """Return recent structural evidence emitted by native network ticks."""

        return tuple(self._structural_runtime_observations)

    @property
    def structural_proposal_candidates(self) -> tuple[StructuralProposalCandidate, ...]:
        """Return pending structural candidates awaiting explicit ledger validation."""

        return tuple(self._structural_proposal_candidates.values())

    @property
    def structural_maintenance_results(self) -> tuple[StructuralMaintenanceResult, ...]:
        """Return recent per-candidate results from maintenance cycles."""

        return tuple(self._structural_maintenance_results)

    def materialize_structural_candidate(
        self,
        candidate_id: str,
    ) -> StructuralTopologyProposal | None:
        """Turn one candidate into a pending topology proposal without applying it."""

        key = str(candidate_id)
        existing_proposal_id = self._structural_candidate_proposals.get(key)
        if existing_proposal_id is not None:
            return self._topology_proposals.get(existing_proposal_id)
        candidate = self._structural_proposal_candidates.get(key)
        if candidate is None:
            return None
        if candidate.target_kind == "neuron" and candidate.operation == "add":
            specification = dict(candidate.specification)
            proposal = self.propose_neuron_add(
                region_id=str(specification.get("region_id", candidate.substrate_ids[0])),
                unit_id=str(specification["unit_id"]),
                evidence_ids=candidate.evidence_ids,
                parent_checkpoint_id=f"candidate-parent:{candidate.candidate_id}",
                resource_cost=candidate.resource_cost,
            )
            self._topology_proposals[proposal.proposal_id] = proposal
            self._topology_network_ids[proposal.proposal_id] = candidate.network_id
            self._structural_candidate_proposals[key] = proposal.proposal_id
            self._structural_proposal_candidates.pop(key)
            return proposal
        try:
            network = self._neuron_networks[candidate.network_id]
        except KeyError as exc:
            raise ValueError(f"unknown adaptive neuron network: {candidate.network_id}") from exc
        specification = dict(candidate.specification)
        parent_checkpoint_id = f"candidate-parent:{candidate.candidate_id}"
        if candidate.operation == "split":
            proposal = network.propose_region_split(
                region_id=str(specification.get("region_id", candidate.substrate_ids[0])),
                first_unit_count=int(specification["first_unit_count"]),
                evidence_ids=candidate.evidence_ids,
                parent_checkpoint_id=parent_checkpoint_id,
                resource_cost=candidate.resource_cost,
            )
        elif candidate.operation == "merge":
            region_ids = tuple(
                str(item) for item in specification.get("region_ids", candidate.substrate_ids)
            )
            proposal = network.propose_region_merge(
                region_ids=region_ids,
                evidence_ids=candidate.evidence_ids,
                parent_checkpoint_id=parent_checkpoint_id,
                resource_cost=candidate.resource_cost,
            )
        elif candidate.operation == "prune" and candidate.target_kind == "region":
            proposal = network.propose_region_prune(
                region_id=str(specification.get("region_id", candidate.substrate_ids[0])),
                evidence_ids=candidate.evidence_ids,
                parent_checkpoint_id=parent_checkpoint_id,
                resource_cost=candidate.resource_cost,
            )
        elif candidate.operation == "prune" and candidate.target_kind == "connection":
            proposal = network.propose_connection_prune(
                connection_id=str(specification.get("connection_id", candidate.substrate_ids[0])),
                evidence_ids=candidate.evidence_ids,
                parent_checkpoint_id=parent_checkpoint_id,
                resource_cost=candidate.resource_cost,
            )
        else:
            raise ValueError(
                f"candidate operation cannot be materialized: {candidate.operation}/"
                f"{candidate.target_kind}"
            )
        self._topology_proposals[proposal.proposal_id] = proposal
        self._topology_network_ids[proposal.proposal_id] = candidate.network_id
        self._structural_candidate_proposals[key] = proposal.proposal_id
        self._structural_proposal_candidates.pop(key)
        return proposal

    def validate_neuron_add_holdout(
        self,
        *,
        proposal_id: str,
        holdout_inputs: Sequence[torch.Tensor],
        expected_activities: Sequence[torch.Tensor],
    ) -> bool:
        """Validate direct neuron birth against a zero-padded parent baseline."""

        if self._structural_growth_controller is None:
            raise RuntimeError("structural growth controller is not attached")
        proposal = self._topology_proposals.get(str(proposal_id))
        if (
            proposal is None
            or proposal.status != "pending"
            or proposal.target_kind != "neuron"
            or proposal.operation != "add"
        ):
            raise ValueError("proposal is not a pending neuron add")
        if len(holdout_inputs) == 0 or len(holdout_inputs) != len(expected_activities):
            raise ValueError(
                "neuron add holdout inputs and expected activities must have equal size"
            )
        try:
            parent = self._neuron_regions[proposal.substrate_id]
        except KeyError as exc:
            raise ValueError(f"unknown adaptive neuron region: {proposal.substrate_id}") from exc
        parent_payload = parent.to_payload()
        baseline = AdaptiveNeuronRegion.from_payload(
            parent_payload,
            generator=self._checkpoint_region_generator(),
            device=self.device,
        )
        candidate = AdaptiveNeuronRegion.from_payload(
            parent_payload,
            generator=self._checkpoint_region_generator(),
            device=self.device,
        )
        candidate.apply_topology_proposal(proposal, generator=self._checkpoint_region_generator())
        baseline_errors: list[float] = []
        candidate_errors: list[float] = []
        for external_input, expected_activity in zip(
            holdout_inputs,
            expected_activities,
            strict=True,
        ):
            expected_value = expected_activity.to(self.device)
            if expected_value.shape != (candidate.unit_count,):
                raise ValueError("neuron add expected activity shape does not match grown region")
            candidate_activity = candidate.step(external_input)
            baseline_activity = baseline.step(external_input)
            padded_baseline = torch.cat(
                (
                    baseline_activity,
                    torch.zeros(candidate.unit_count - baseline.unit_count, device=self.device),
                )
            )
            baseline_errors.append(
                float(
                    torch.mean(torch.abs(padded_baseline - expected_value)).clamp(0.0, 1.0).item()
                )
            )
            candidate_errors.append(
                float(
                    torch.mean(torch.abs(candidate_activity - expected_value))
                    .clamp(0.0, 1.0)
                    .item()
                )
            )
        baseline_error = sum(baseline_errors) / len(baseline_errors)
        candidate_error = sum(candidate_errors) / len(candidate_errors)
        gain = max(0.0, baseline_error - candidate_error)
        score = 0.0 if baseline_error <= 1e-8 else min(1.0, gain / baseline_error)
        validated = score >= float(self._structural_growth_controller.dynamics.minimum_holdout_gain)
        self._topology_proposals[str(proposal_id)] = replace(
            proposal,
            validation_score=score,
        )
        previous = self._cognitive_state.development
        self._cognitive_state = replace(
            self._cognitive_state,
            development=replace(
                previous,
                tick=self.tick,
                last_update_source="neuron-add-holdout-validation",
                last_validation_status="validated" if validated else "rejected",
                validation_evidence_ids=self._bounded_ids(
                    previous.validation_evidence_ids,
                    f"holdout:{proposal_id}",
                    limit=self._lineage_limit(),
                ),
                lineage=self._bounded_ids(
                    previous.lineage,
                    f"holdout:{proposal_id}",
                    limit=self._lineage_limit(),
                ),
            ),
        )
        return validated

    def validate_structural_candidate_holdout(
        self,
        candidate_id: str,
        *,
        holdout_inputs: Sequence[Any],
        expected_activities: Sequence[Any],
    ) -> bool:
        """Dispatch one materialized candidate to its operation-specific holdout gate."""

        proposal = self.materialize_structural_candidate(candidate_id)
        if proposal is None:
            raise ValueError(f"unknown structural proposal candidate: {candidate_id}")
        if proposal.target_kind == "neuron" and proposal.operation == "add":
            return self.validate_neuron_add_holdout(
                proposal_id=proposal.proposal_id,
                holdout_inputs=holdout_inputs,
                expected_activities=expected_activities,
            )
        network_id = self._topology_network_ids.get(proposal.proposal_id)
        if network_id is None:
            if proposal.target_kind == "neuron" and proposal.operation == "add":
                return self.commit_neuron_add(proposal)
            raise ValueError("structural candidate proposal is not attached to a network")
        role = dict(proposal.specification).get("topology_role")
        if proposal.operation == "split" and role == "region_split":
            return self.validate_region_split_holdout(
                network_id=network_id,
                proposal_id=proposal.proposal_id,
                holdout_inputs=holdout_inputs,
                expected_activities=expected_activities,
            )
        if proposal.operation == "merge" and role == "region_merge":
            return self.validate_region_merge_holdout(
                network_id=network_id,
                proposal_id=proposal.proposal_id,
                holdout_inputs=holdout_inputs,
                expected_activities=expected_activities,
            )
        if proposal.operation == "prune" and role == "region_prune":
            return self.validate_region_prune_holdout(
                network_id=network_id,
                proposal_id=proposal.proposal_id,
                holdout_inputs=holdout_inputs,
                expected_activities=expected_activities,
            )
        if proposal.operation == "prune" and role == "cross_region_connection_prune":
            return self.validate_cross_region_connection_prune_holdout(
                network_id=network_id,
                proposal_id=proposal.proposal_id,
                holdout_inputs=holdout_inputs,
                expected_activities=expected_activities,
            )
        raise ValueError(
            f"candidate proposal has no holdout validator: {proposal.operation}/{role}"
        )

    def commit_structural_candidate(self, candidate_id: str) -> bool:
        """Dispatch one candidate through the operation-specific topology ledger."""

        proposal = self.materialize_structural_candidate(candidate_id)
        if proposal is None:
            raise ValueError(f"unknown structural proposal candidate: {candidate_id}")
        if proposal.target_kind == "neuron" and proposal.operation == "add":
            return self.commit_neuron_add(proposal)
        network_id = self._topology_network_ids.get(proposal.proposal_id)
        if network_id is None:
            raise ValueError("structural candidate proposal is not attached to a network")
        role = dict(proposal.specification).get("topology_role")
        if proposal.operation == "split" and role == "region_split":
            return self.commit_region_split(network_id, proposal)
        if proposal.operation == "merge" and role == "region_merge":
            return self.commit_region_merge(network_id, proposal)
        if proposal.operation == "prune" and role == "region_prune":
            return self.commit_region_prune(network_id, proposal)
        if proposal.operation == "prune" and role == "cross_region_connection_prune":
            return self.commit_cross_region_connection_prune(network_id, proposal)
        raise ValueError(
            f"candidate proposal has no commit dispatcher: {proposal.operation}/{role}"
        )

    def rollback_structural_candidate(self, candidate_id: str) -> bool:
        """Dispatch reverse rollback for the latest accepted candidate change."""

        key = str(candidate_id)
        proposal_id = self._structural_candidate_proposals.get(key)
        if proposal_id is None:
            raise ValueError(f"unknown materialized structural candidate: {candidate_id}")
        proposal = self._topology_proposals.get(proposal_id)
        if proposal is None:
            raise ValueError("structural candidate proposal is missing")
        if proposal.target_kind == "neuron" and proposal.operation == "add":
            return self.rollback_neuron_add(proposal_id)
        role = dict(proposal.specification).get("topology_role")
        if proposal.operation == "split" and role == "region_split":
            return self.rollback_region_split(proposal_id)
        if proposal.operation == "merge" and role == "region_merge":
            return self.rollback_region_merge(proposal_id)
        if proposal.operation == "prune" and role == "region_prune":
            return self.rollback_region_prune(proposal_id)
        if proposal.operation == "prune" and role == "cross_region_connection_prune":
            return self.rollback_cross_region_connection_prune(proposal_id)
        raise ValueError(
            f"candidate proposal has no rollback dispatcher: {proposal.operation}/{role}"
        )

    def _pending_structural_candidate_ids(self) -> tuple[str, ...]:
        ids = list(self._structural_proposal_candidates)
        for candidate_id, proposal_id in self._structural_candidate_proposals.items():
            proposal = self._topology_proposals.get(proposal_id)
            if proposal is not None and proposal.status == "pending" and candidate_id not in ids:
                ids.append(candidate_id)
        return tuple(ids)

    def _record_structural_maintenance_result(
        self,
        result: StructuralMaintenanceResult,
    ) -> None:
        self._structural_maintenance_results.append(result)
        self._structural_maintenance_results = self._structural_maintenance_results[
            -self._lineage_limit() :
        ]

    def _accepted_structural_candidate(self, candidate_id: str) -> bool:
        proposal_id = self._structural_candidate_proposals.get(str(candidate_id))
        if proposal_id is None:
            return False
        proposal = self._topology_proposals.get(proposal_id)
        return proposal is not None and proposal.status == "accepted"

    @staticmethod
    def _candidate_conflict_keys(
        candidate: StructuralProposalCandidate,
    ) -> tuple[str, ...]:
        if candidate.conflict_keys:
            return candidate.conflict_keys
        specification = dict(candidate.specification)
        if candidate.target_kind == "neuron" and candidate.operation == "add":
            unit_id = str(specification.get("unit_id", ""))
            if unit_id:
                return (f"{candidate.network_id}:neuron:{candidate.substrate_ids[0]}:{unit_id}",)
        substrate = "|".join(sorted(candidate.substrate_ids))
        return (f"{candidate.network_id}:substrate:{substrate}",)

    @staticmethod
    def _candidates_conflict(
        first: StructuralProposalCandidate,
        second: StructuralProposalCandidate,
    ) -> bool:
        if first.network_id != second.network_id:
            return False
        if not set(first.substrate_ids).intersection(second.substrate_ids):
            return False
        if (
            first.target_kind == "neuron"
            and first.operation == "add"
            and second.target_kind == "neuron"
            and second.operation == "add"
        ):
            first_unit = dict(first.specification).get("unit_id")
            second_unit = dict(second.specification).get("unit_id")
            return first_unit == second_unit
        return True

    def _prepare_structural_maintenance_cycle(
        self,
        candidate_ids: Sequence[str],
    ) -> tuple[tuple[str, ...], dict[str, str]]:
        """Resolve dependencies and reject conflicting queued candidates first."""

        ids = tuple(dict.fromkeys(str(item) for item in candidate_ids))
        queued = {
            candidate_id: self._structural_proposal_candidates[candidate_id]
            for candidate_id in ids
            if candidate_id in self._structural_proposal_candidates
        }
        errors: dict[str, str] = {}
        conflict_groups: dict[str, list[str]] = {}
        for candidate_id, candidate in queued.items():
            for conflict_key in self._candidate_conflict_keys(candidate):
                conflict_groups.setdefault(conflict_key, []).append(candidate_id)
        for conflict_key, group in conflict_groups.items():
            if len(group) <= 1:
                continue
            message = f"candidate conflict on {conflict_key}: " + ", ".join(group)
            for candidate_id in group:
                errors[candidate_id] = message
        queued_items = tuple(queued.items())
        for first_index, (first_id, first) in enumerate(queued_items):
            for second_id, second in queued_items[first_index + 1 :]:
                if not self._candidates_conflict(first, second):
                    continue
                message = f"candidate conflict between {first_id} and {second_id}"
                errors[first_id] = message
                errors[second_id] = message

        visiting: list[str] = []
        visited: set[str] = set()
        ordered: list[str] = []

        def visit(candidate_id: str) -> None:
            if candidate_id in visited:
                return
            if candidate_id in visiting:
                cycle_start = visiting.index(candidate_id)
                cycle = visiting[cycle_start:]
                message = "candidate dependency cycle: " + " -> ".join(cycle)
                for item in cycle:
                    errors[item] = message
                return
            visiting.append(candidate_id)
            candidate = queued[candidate_id]
            for dependency_id in candidate.depends_on_candidate_ids:
                if dependency_id in queued:
                    visit(dependency_id)
                    if dependency_id in errors:
                        errors[candidate_id] = (
                            f"candidate dependency is not admissible: {dependency_id}"
                        )
                elif not self._accepted_structural_candidate(dependency_id):
                    errors[candidate_id] = (
                        f"candidate dependency is missing or not accepted: {dependency_id}"
                    )
            visiting.pop()
            visited.add(candidate_id)
            if candidate_id not in errors:
                ordered.append(candidate_id)

        for candidate_id in queued:
            visit(candidate_id)
        return tuple(ordered), errors

    def run_structural_maintenance_cycle(
        self,
        *,
        holdout_inputs_by_candidate: Mapping[str, Sequence[Any]],
        expected_activities_by_candidate: Mapping[str, Sequence[Any]],
        candidate_ids: Sequence[str] | None = None,
    ) -> tuple[StructuralMaintenanceResult, ...]:
        """Process candidates independently through materialize/validate/commit gates."""

        ids = (
            self._pending_structural_candidate_ids()
            if candidate_ids is None
            else tuple(str(item) for item in candidate_ids)
        )
        ordered_ids, preflight_errors = self._prepare_structural_maintenance_cycle(ids)
        execution_ids = ordered_ids + tuple(
            candidate_id for candidate_id in ids if candidate_id not in ordered_ids
        )
        results: list[StructuralMaintenanceResult] = []
        results_by_candidate: dict[str, StructuralMaintenanceResult] = {}
        for candidate_id in execution_ids:
            if candidate_id in preflight_errors:
                result = StructuralMaintenanceResult(
                    candidate_id=candidate_id,
                    proposal_id=self._structural_candidate_proposals.get(candidate_id),
                    status="failed_closed",
                    error=preflight_errors[candidate_id],
                )
                self._record_structural_maintenance_result(result)
                results.append(result)
                results_by_candidate[candidate_id] = result
                continue
            candidate = self._structural_proposal_candidates.get(candidate_id)
            blocked_dependency = None
            if candidate is not None:
                for dependency_id in candidate.depends_on_candidate_ids:
                    dependency_result = results_by_candidate.get(dependency_id)
                    if dependency_result is not None and dependency_result.status != "committed":
                        blocked_dependency = dependency_id
                        break
            if blocked_dependency is not None:
                result = StructuralMaintenanceResult(
                    candidate_id=candidate_id,
                    proposal_id=self._structural_candidate_proposals.get(candidate_id),
                    status="failed_closed",
                    error=f"candidate dependency did not commit: {blocked_dependency}",
                )
                self._record_structural_maintenance_result(result)
                results.append(result)
                results_by_candidate[candidate_id] = result
                continue
            holdout_inputs = holdout_inputs_by_candidate.get(candidate_id)
            expected_activities = expected_activities_by_candidate.get(candidate_id)
            if not holdout_inputs or not expected_activities:
                result = StructuralMaintenanceResult(
                    candidate_id=candidate_id,
                    proposal_id=self._structural_candidate_proposals.get(candidate_id),
                    status="missing_holdout",
                    error="candidate requires non-empty holdout inputs and expected activities",
                )
                self._record_structural_maintenance_result(result)
                results.append(result)
                results_by_candidate[candidate_id] = result
                continue
            try:
                proposal = self.materialize_structural_candidate(candidate_id)
                if proposal is None:
                    raise ValueError("candidate is not available for materialization")
                if proposal.status != "pending":
                    result = StructuralMaintenanceResult(
                        candidate_id=candidate_id,
                        proposal_id=proposal.proposal_id,
                        status="already_applied",
                        validation_score=proposal.validation_score,
                    )
                    self._record_structural_maintenance_result(result)
                    results.append(result)
                    results_by_candidate[candidate_id] = result
                    continue
                validated = self.validate_structural_candidate_holdout(
                    candidate_id,
                    holdout_inputs=holdout_inputs,
                    expected_activities=expected_activities,
                )
                current = self._topology_proposals[proposal.proposal_id]
                if not validated:
                    self.commit_structural_candidate(candidate_id)
                    status = "rejected"
                else:
                    committed = self.commit_structural_candidate(candidate_id)
                    status = "committed" if committed else "rejected"
                result = StructuralMaintenanceResult(
                    candidate_id=candidate_id,
                    proposal_id=proposal.proposal_id,
                    status=status,
                    validation_score=current.validation_score,
                )
            except (IndexError, KeyError, RuntimeError, ValueError) as exc:
                proposal_id = self._structural_candidate_proposals.get(candidate_id)
                proposal = (
                    None if proposal_id is None else self._topology_proposals.get(proposal_id)
                )
                result = StructuralMaintenanceResult(
                    candidate_id=candidate_id,
                    proposal_id=proposal_id,
                    status="failed_closed",
                    validation_score=0.0 if proposal is None else proposal.validation_score,
                    error=str(exc),
                )
            self._record_structural_maintenance_result(result)
            results.append(result)
            results_by_candidate[candidate_id] = result
        return tuple(results)

    @staticmethod
    def _growth_request_identity(
        concept_id: str,
        transitions: Sequence[tuple[WorldTransition, float]],
    ) -> tuple[str, str]:
        items = tuple(transitions)
        actions = ">".join(transition.action.kind for transition, _ in items)
        first = items[0][0]
        last = items[-1][0]
        candidate_id = f"candidate:{concept_id}:{first.before.tick}:{last.after.tick}:{actions}"
        return f"growth:{candidate_id}", candidate_id

    def _record_growth_development(
        self,
        request: StructuralGrowthRequest,
        *,
        consume_budget: bool,
        evidence_id: str,
    ) -> None:
        previous = self._cognitive_state.development
        budget = int(previous.structural_budget)
        if consume_budget:
            budget -= int(request.requested_units)
        self._cognitive_state = replace(
            self._cognitive_state,
            development=replace(
                previous,
                tick=self.tick,
                stage="growth",
                structural_budget=budget,
                proposal_ids=self._bounded_ids(
                    previous.proposal_ids,
                    request.request_id,
                    limit=self._lineage_limit(),
                ),
                parent_checkpoint_id=request.parent_checkpoint_id,
                last_update_source="online-branch-growth",
                last_validation_status=request.status,
                validation_evidence_ids=self._bounded_ids(
                    previous.validation_evidence_ids,
                    evidence_id,
                    limit=self._lineage_limit(),
                ),
                growth_count=(
                    previous.growth_count + int(request.requested_units)
                    if consume_budget
                    else previous.growth_count
                ),
                lineage=self._bounded_ids(
                    previous.lineage,
                    request.request_id,
                    limit=self._lineage_limit(),
                ),
            ),
        )

    def _commit_online_concept_branch(
        self,
        concept_id: str,
        transitions: Sequence[tuple[WorldTransition, float]],
    ) -> str | None:
        items = tuple(transitions)
        if not items:
            return None
        request_id, candidate_id = self._growth_request_identity(concept_id, items)
        existing = self._growth_requests.get(request_id)
        if existing is not None and existing.status == "accepted":
            return existing.candidate_trace_id
        parent_checkpoint_id = f"growth-parent:{request_id}"
        pending = StructuralGrowthRequest(
            request_id=request_id,
            concept_id=str(concept_id),
            candidate_trace_id=candidate_id,
            evidence_ids=(),
            parent_checkpoint_id=parent_checkpoint_id,
            status="pending",
        )
        if self._cognitive_state.development.structural_budget < pending.requested_units:
            rejected = replace(pending, status="rejected")
            self._growth_requests[request_id] = rejected
            self._record_growth_development(
                rejected,
                consume_budget=False,
                evidence_id=candidate_id,
            )
            return None
        parent_snapshot = self._concept_formation.checkpoint()
        trial = ConceptFormationOrgan.from_checkpoint(parent_snapshot, device=self.device)
        trace_id = trial.grow_sequence_trace(concept_id, items)
        if trace_id is None:
            rejected = replace(pending, status="rejected")
            self._growth_requests[request_id] = rejected
            self._record_growth_development(
                rejected,
                consume_budget=False,
                evidence_id=candidate_id,
            )
            return None
        trial_checkpoint = trial.checkpoint()
        checkpoint_trial = ConceptFormationOrgan.from_checkpoint(
            trial_checkpoint,
            device=self.device,
        )
        checkpoint_concept = next(
            (concept for concept in checkpoint_trial.concepts if concept.concept_id == concept_id),
            None,
        )
        checkpoint_trace = (
            None
            if checkpoint_concept is None
            else next(
                (
                    trace
                    for trace in checkpoint_concept.sequence_traces
                    if trace.trace_id == trace_id
                ),
                None,
            )
        )
        lesion_trial = ConceptFormationOrgan.from_checkpoint(
            trial_checkpoint,
            device=self.device,
        )
        lesion_removed = lesion_trial.lesion_sequence_trace(concept_id, (trace_id,))
        replay_score = (
            checkpoint_trial.suffix_sequence_affinity(
                checkpoint_concept,
                tuple(transition.action.kind for transition, _ in items),
                current_state=items[0][0].before,
            )
            if checkpoint_concept is not None
            else 0.0
        )
        validated = bool(
            checkpoint_trace is not None and lesion_removed == (trace_id,) and replay_score > 0.0
        )
        if not validated:
            rejected = replace(
                pending,
                status="rejected",
                validation_score=max(0.0, min(1.0, float(replay_score))),
            )
            self._growth_requests[request_id] = rejected
            self._record_growth_development(
                rejected,
                consume_budget=False,
                evidence_id=candidate_id,
            )
            return None
        accepted = replace(
            pending,
            candidate_trace_id=trace_id,
            evidence_ids=(trace_id,),
            status="accepted",
            validation_score=1.0,
        )
        self._concept_formation = trial
        self._growth_requests[request_id] = accepted
        self._growth_request_snapshots[request_id] = parent_snapshot
        self._record_growth_development(
            accepted,
            consume_budget=True,
            evidence_id=trace_id,
        )
        self._cognitive_state = replace(
            self._cognitive_state,
            concepts=self._concept_formation.concepts,
        )
        return trace_id

    def grow_online_concept_branch(
        self,
        concept_id: str,
        transitions: Sequence[tuple[WorldTransition, float]],
    ) -> str | None:
        """Request, validate and commit a novel real-transition branch."""

        return self._commit_online_concept_branch(concept_id, transitions)

    def rollback_growth_request(self, request_id: str) -> bool:
        """Restore the parent concept checkpoint for the latest accepted growth."""

        request = self._growth_requests.get(str(request_id))
        snapshot = self._growth_request_snapshots.get(str(request_id))
        if request is None or request.status != "accepted" or snapshot is None:
            return False
        self._concept_formation = ConceptFormationOrgan.from_checkpoint(
            snapshot,
            device=self.device,
        )
        rolled_back = replace(request, status="rolled_back", validation_score=0.0)
        self._growth_requests[str(request_id)] = rolled_back
        self._growth_request_snapshots.pop(str(request_id), None)
        previous = self._cognitive_state.development
        self._cognitive_state = replace(
            self._cognitive_state,
            concepts=self._concept_formation.concepts,
            development=replace(
                previous,
                tick=self.tick,
                structural_budget=previous.structural_budget + request.requested_units,
                last_update_source="growth-rollback",
                last_validation_status="rolled_back",
                validation_evidence_ids=self._bounded_ids(
                    previous.validation_evidence_ids,
                    request.request_id,
                    limit=self._lineage_limit(),
                ),
                growth_count=max(0, previous.growth_count - request.requested_units),
                lineage=self._bounded_ids(
                    previous.lineage,
                    request.request_id,
                    limit=self._lineage_limit(),
                ),
            ),
        )
        return True

    def _record_topology_development(
        self,
        proposal: StructuralTopologyProposal,
        *,
        consume_budget: bool,
        evidence_id: str,
        source: str = "synapse-topology-rewire",
        counter: str = "growth_count",
    ) -> None:
        if counter not in {"growth_count", "prune_count", "split_merge_count"}:
            raise ValueError(f"unsupported topology development counter: {counter}")
        previous = self._cognitive_state.development
        budget = int(previous.structural_budget)
        if consume_budget:
            budget -= int(proposal.resource_cost)
        counter_updates = {
            "growth_count": previous.growth_count,
            "prune_count": previous.prune_count,
            "split_merge_count": previous.split_merge_count,
        }
        if consume_budget:
            counter_updates[counter] += int(proposal.requested_units)
        self._cognitive_state = replace(
            self._cognitive_state,
            development=replace(
                previous,
                tick=self.tick,
                stage="growth",
                structural_budget=budget,
                proposal_ids=self._bounded_ids(
                    previous.proposal_ids,
                    proposal.proposal_id,
                    limit=self._lineage_limit(),
                ),
                parent_checkpoint_id=proposal.parent_checkpoint_id,
                last_update_source=source,
                last_validation_status=proposal.status,
                validation_evidence_ids=self._bounded_ids(
                    previous.validation_evidence_ids,
                    evidence_id,
                    limit=self._lineage_limit(),
                ),
                growth_count=counter_updates["growth_count"],
                prune_count=counter_updates["prune_count"],
                split_merge_count=counter_updates["split_merge_count"],
                lineage=self._bounded_ids(
                    previous.lineage,
                    proposal.proposal_id,
                    limit=self._lineage_limit(),
                ),
            ),
        )

    def propose_synapse_rewire(
        self,
        *,
        substrate_id: str,
        post_index: int,
        slot_index: int,
        replacement_pre_index: int,
        evidence_ids: Sequence[str],
        parent_checkpoint_id: str | None = None,
        resource_cost: int = 1,
    ) -> StructuralTopologyProposal:
        """Create a topology proposal without bypassing the Taiji ledger."""

        proposal = self.fabric.propose_synapse_rewire(
            substrate_id=substrate_id,
            post_index=post_index,
            slot_index=slot_index,
            replacement_pre_index=replacement_pre_index,
            evidence_ids=evidence_ids,
            parent_checkpoint_id=parent_checkpoint_id,
            resource_cost=resource_cost,
        )
        if proposal.parent_checkpoint_id is None:
            proposal = replace(
                proposal,
                parent_checkpoint_id=f"topology-parent:{proposal.proposal_id}",
            )
        return proposal

    def commit_synapse_rewire(self, proposal: StructuralTopologyProposal) -> bool:
        """Validate, budget and commit a topology proposal transactionally."""

        existing = self._topology_proposals.get(proposal.proposal_id)
        if existing is not None:
            if existing.status == "accepted":
                return True
            if existing.status == "rolled_back":
                return False
        if proposal.status != "pending":
            raise ValueError("only pending topology proposals can be committed")
        if self._cognitive_state.development.structural_budget < proposal.resource_cost:
            rejected = replace(proposal, status="rejected")
            self._topology_proposals[proposal.proposal_id] = rejected
            self._record_topology_development(
                rejected,
                consume_budget=False,
                evidence_id=(
                    proposal.evidence_ids[0] if proposal.evidence_ids else proposal.proposal_id
                ),
            )
            return False

        parent_snapshot = self.fabric.to_payload()
        try:
            self.fabric.apply_synapse_rewire(proposal)
            accepted_snapshot = self.fabric.to_payload()
            trial = TaijiFabric(
                self.config,
                generator=torch.Generator(device="cpu").manual_seed(self.config.seed),
                device=self.device,
            )
            trial.load_payload(accepted_snapshot)
            source = self.fabric._topology_bank(proposal.substrate_id)
            restored = trial._topology_bank(proposal.substrate_id)
            if not torch.equal(source.pre_index, restored.pre_index):
                raise ValueError("topology checkpoint roundtrip changed the support")
        except (IndexError, KeyError, RuntimeError, ValueError):
            self.fabric.load_payload(parent_snapshot)
            rejected = replace(proposal, status="rejected")
            self._topology_proposals[proposal.proposal_id] = rejected
            self._record_topology_development(
                rejected,
                consume_budget=False,
                evidence_id=(
                    proposal.evidence_ids[0] if proposal.evidence_ids else proposal.proposal_id
                ),
            )
            return False

        accepted = replace(proposal, status="accepted", validation_score=1.0)
        self._topology_proposals[proposal.proposal_id] = accepted
        self._topology_parent_snapshots[proposal.proposal_id] = parent_snapshot
        self._record_topology_development(
            accepted,
            consume_budget=True,
            evidence_id=(
                proposal.evidence_ids[0] if proposal.evidence_ids else proposal.proposal_id
            ),
        )
        return True

    def validate_region_growth_holdout(
        self,
        *,
        network_id: str,
        proposal_id: str,
        holdout_inputs: Sequence[Mapping[str, torch.Tensor]],
        expected_activities: Sequence[Mapping[str, torch.Tensor]],
    ) -> bool:
        """Validate a born region against a silent baseline on unseen inputs."""

        if self._structural_growth_controller is None:
            raise RuntimeError("structural growth controller is not attached")
        proposal = self._topology_proposals.get(str(proposal_id))
        if (
            proposal is None
            or proposal.status != "accepted"
            or proposal.target_kind != "region"
            or dict(proposal.specification).get("topology_role") != "region"
        ):
            raise ValueError("proposal is not an accepted region add")
        if len(holdout_inputs) == 0 or len(holdout_inputs) != len(expected_activities):
            raise ValueError("region holdout inputs and expected activities must have equal size")
        try:
            network = self._neuron_networks[str(network_id)]
        except KeyError as exc:
            raise ValueError(f"unknown adaptive neuron network: {network_id}") from exc
        child_id = proposal.substrate_id
        try:
            child = next(region for region in network.regions if region.region_id == child_id)
        except StopIteration as exc:
            raise ValueError(f"unknown grown region: {child_id}") from exc
        payload = network.to_payload()
        candidate = AdaptiveNeuronNetwork.from_payload(
            payload,
            generator=self._checkpoint_region_generator(),
            device=self.device,
        )
        baseline = AdaptiveNeuronNetwork.from_payload(
            payload,
            generator=self._checkpoint_region_generator(),
            device=self.device,
        )
        baseline.lesion_region(child_id)
        candidate_errors: list[float] = []
        baseline_errors: list[float] = []
        for inputs, expected in zip(holdout_inputs, expected_activities, strict=True):
            expected_activity = expected.get(child_id)
            if expected_activity is None:
                raise ValueError(f"region holdout expected activity is missing {child_id}")
            if expected_activity.shape != (child.unit_count,):
                raise ValueError(f"region holdout expected activity shape mismatch for {child_id}")
            candidate_activity = candidate.step(inputs, connection_ids=())[child_id]
            baseline_activity = baseline.step(inputs, connection_ids=())[child_id]
            expected_value = expected_activity.to(self.device)
            candidate_errors.append(
                float(
                    torch.mean(torch.abs(candidate_activity - expected_value))
                    .clamp(0.0, 1.0)
                    .item()
                )
            )
            baseline_errors.append(
                float(
                    torch.mean(torch.abs(baseline_activity - expected_value)).clamp(0.0, 1.0).item()
                )
            )
        baseline_error = sum(baseline_errors) / len(baseline_errors)
        candidate_error = sum(candidate_errors) / len(candidate_errors)
        gain = max(0.0, baseline_error - candidate_error)
        score = 0.0 if baseline_error <= 1e-8 else min(1.0, gain / baseline_error)
        validated = score >= float(self._structural_growth_controller.dynamics.minimum_holdout_gain)
        self._topology_proposals[str(proposal_id)] = replace(
            proposal,
            validation_score=score,
        )
        previous = self._cognitive_state.development
        self._cognitive_state = replace(
            self._cognitive_state,
            development=replace(
                previous,
                tick=self.tick,
                last_update_source="region-holdout-validation",
                last_validation_status="validated" if validated else "rejected",
                validation_evidence_ids=self._bounded_ids(
                    previous.validation_evidence_ids,
                    f"holdout:{proposal_id}",
                    limit=self._lineage_limit(),
                ),
                lineage=self._bounded_ids(
                    previous.lineage,
                    f"holdout:{proposal_id}",
                    limit=self._lineage_limit(),
                ),
            ),
        )
        return validated

    def rollback_synapse_rewire(self, proposal_id: str) -> bool:
        """Rollback only the latest accepted topology proposal."""

        key = str(proposal_id)
        proposal = self._topology_proposals.get(key)
        snapshot = self._topology_parent_snapshots.get(key)
        if proposal is None or proposal.status != "accepted" or snapshot is None:
            return False
        active = [
            item.proposal_id
            for item in self._topology_proposals.values()
            if item.status == "accepted" and item.proposal_id in self._topology_parent_snapshots
        ]
        if not active or active[-1] != key:
            return False
        self.fabric.load_payload(snapshot)
        rolled_back = replace(proposal, status="rolled_back", validation_score=0.0)
        self._topology_proposals[key] = rolled_back
        self._topology_parent_snapshots.pop(key, None)
        previous = self._cognitive_state.development
        self._cognitive_state = replace(
            self._cognitive_state,
            development=replace(
                previous,
                tick=self.tick,
                structural_budget=previous.structural_budget + proposal.resource_cost,
                last_update_source="topology-rollback",
                last_validation_status="rolled_back",
                validation_evidence_ids=self._bounded_ids(
                    previous.validation_evidence_ids,
                    key,
                    limit=self._lineage_limit(),
                ),
                growth_count=max(0, previous.growth_count - proposal.requested_units),
                lineage=self._bounded_ids(
                    previous.lineage,
                    key,
                    limit=self._lineage_limit(),
                ),
            ),
        )
        return True

    def attach_adaptive_neuron_region(self, region: AdaptiveNeuronRegion | None) -> None:
        """Attach a dynamic neuron organ without changing the fixed fabric."""

        if region is not None and not isinstance(region, AdaptiveNeuronRegion):
            raise TypeError("region must be an AdaptiveNeuronRegion or None")
        if region is None:
            return
        if region.region_id in self._neuron_regions:
            raise ValueError(f"adaptive neuron region already attached: {region.region_id}")
        self._neuron_regions[region.region_id] = region

    def step_adaptive_neuron_region(
        self,
        region_id: str,
        external_input: torch.Tensor,
        *,
        expected_activity: torch.Tensor | None = None,
        holdout: bool = False,
    ) -> torch.Tensor:
        """Run one standalone native neuron tick and emit structural evidence.

        A standalone region is a valid native substrate, not a test-only
        topology object.  Runtime evidence can therefore request one new
        neuron through the same candidate/holdout/ledger path used by
        cross-region networks, while the live region remains unchanged until
        a separate commit.
        """

        try:
            region = self._neuron_regions[str(region_id)]
        except KeyError as exc:
            raise ValueError(f"unknown adaptive neuron region: {region_id}") from exc
        activity = region.step(external_input)
        self._structural_runtime_tick += 1
        runtime_tick = self._structural_runtime_tick
        resource_state = self._structural_runtime_resource_state()
        resource_pressure = 1.0 - resource_state
        usage = float(torch.mean(torch.abs(activity)).clamp(0.0, 1.0).item())
        prediction_error: float | None = None
        learning_gain = 0.0
        holdout_transfer = 0.0
        error_key = f"standalone:{region.region_id}"
        if expected_activity is not None:
            expected_value = expected_activity.to(activity.device)
            if expected_value.shape != activity.shape:
                raise ValueError(
                    f"standalone activity shape mismatch for region {region.region_id}"
                )
            prediction_error = float(
                torch.mean(torch.abs(activity - expected_value)).clamp(0.0, 1.0).item()
            )
            previous_error = self._structural_runtime_previous_errors.get(error_key)
            if previous_error is not None:
                learning_gain = max(0.0, min(1.0, previous_error - prediction_error))
            self._structural_runtime_previous_errors[error_key] = prediction_error
            if holdout:
                holdout_transfer = 1.0 - prediction_error

            if self._structural_growth_controller is not None:
                growth_decision = self._structural_growth_controller.observe(
                    region.region_id,
                    prediction_error=prediction_error,
                    resource_state=resource_state,
                    holdout_transfer=holdout_transfer,
                    evidence_ids=(f"runtime-structure:{region.region_id}:{runtime_tick}",),
                )
                if growth_decision.should_grow:
                    unit_id = self._structural_growth_controller.next_unit_id(
                        region.region_id,
                        region.unit_ids,
                    )
                    self._queue_structural_proposal_candidate(
                        StructuralProposalCandidate(
                            candidate_id=(
                                f"candidate:standalone:{region.region_id}:add:"
                                f"{growth_decision.proposal_ordinal}"
                            ),
                            network_id=f"standalone:{region.region_id}",
                            target_kind="neuron",
                            operation="add",
                            substrate_ids=(region.region_id,),
                            evidence_ids=growth_decision.evidence_ids,
                            source_tick=runtime_tick,
                            priority=min(
                                1.0,
                                growth_decision.error_ema
                                * max(growth_decision.holdout_transfer_ema, 0.0),
                            ),
                            specification=self._candidate_specification(
                                region_id=region.region_id,
                                unit_id=unit_id,
                            ),
                            resource_cost=(
                                self._structural_growth_controller.dynamics.growth_resource_cost
                            ),
                        )
                    )

        evidence_id = f"runtime-structure:{region.region_id}:{runtime_tick}"
        if self._structural_pruning_controller is not None:
            self._structural_pruning_controller.observe_substrate(
                region.region_id,
                usage=usage,
                resource_pressure=resource_pressure,
                learning_gain=learning_gain,
                evidence_ids=(evidence_id,),
            )
        observation = StructuralRuntimeObservation(
            network_id=f"standalone:{region.region_id}",
            region_id=region.region_id,
            tick=runtime_tick,
            usage=usage,
            resource_pressure=resource_pressure,
            prediction_error=prediction_error,
            learning_gain=learning_gain,
            holdout_transfer=holdout_transfer,
            evidence_id=evidence_id,
        )
        self._structural_runtime_observations.append(observation)
        self._structural_runtime_observations = self._structural_runtime_observations[
            -self._lineage_limit() :
        ]
        previous = self._cognitive_state.development
        self._cognitive_state = replace(
            self._cognitive_state,
            development=replace(
                previous,
                stage="structural-observation",
                resource_utilization=resource_pressure,
                last_update_source="runtime-structural-observation",
                last_validation_status="pending",
                validation_evidence_ids=self._bounded_ids(
                    previous.validation_evidence_ids,
                    evidence_id,
                    limit=self._lineage_limit(),
                ),
            ),
        )
        return activity

    def attach_structural_growth_controller(
        self,
        controller: AdaptiveStructuralGrowthController | None,
    ) -> None:
        """Attach or remove the substrate-only automatic growth signal organ."""

        if controller is not None and not isinstance(
            controller,
            AdaptiveStructuralGrowthController,
        ):
            raise TypeError("controller must be an AdaptiveStructuralGrowthController or None")
        self._structural_growth_controller = controller

    def attach_structural_pruning_controller(
        self,
        controller: AdaptiveStructuralPruningController | None,
    ) -> None:
        """Attach or remove the substrate-only structural pruning organ."""

        if controller is not None and not isinstance(
            controller,
            AdaptiveStructuralPruningController,
        ):
            raise TypeError("controller must be an AdaptiveStructuralPruningController or None")
        self._structural_pruning_controller = controller

    def propose_neuron_growth_from_error(
        self,
        *,
        region_id: str,
        prediction_error: float,
        resource_state: float,
        holdout_transfer: float,
        evidence_ids: Sequence[str],
    ) -> StructuralTopologyProposal | None:
        """Turn persistent substrate evidence into a ledger-ready neuron proposal."""

        if self._structural_growth_controller is None:
            raise RuntimeError("structural growth controller is not attached")
        try:
            region = self._neuron_regions[str(region_id)]
        except KeyError as exc:
            raise ValueError(f"unknown adaptive neuron region: {region_id}") from exc
        decision: StructuralGrowthDecision = self._structural_growth_controller.observe(
            region.region_id,
            prediction_error=prediction_error,
            resource_state=resource_state,
            holdout_transfer=holdout_transfer,
            evidence_ids=evidence_ids,
        )
        if not decision.should_grow:
            return None
        unit_id = self._structural_growth_controller.next_unit_id(
            region.region_id,
            region.unit_ids,
        )
        return self.propose_neuron_add(
            region_id=region.region_id,
            unit_id=unit_id,
            evidence_ids=decision.evidence_ids,
            parent_checkpoint_id=(f"growth-signal:{region.region_id}:{decision.proposal_ordinal}"),
            resource_cost=self._structural_growth_controller.dynamics.growth_resource_cost,
        )

    @staticmethod
    def _checkpoint_region_generator() -> torch.Generator:
        """Build only a disposable constructor RNG before loading exact tensors."""

        return torch.Generator(device="cpu").manual_seed(0)

    def propose_neuron_add(
        self,
        *,
        region_id: str,
        unit_id: str,
        evidence_ids: Sequence[str],
        source_region_id: str | None = None,
        parent_checkpoint_id: str | None = None,
        resource_cost: int = 1,
    ) -> StructuralTopologyProposal:
        """Create a neuron birth proposal owned by the runtime ledger."""

        try:
            region = self._neuron_regions[str(region_id)]
        except KeyError as exc:
            raise ValueError(f"unknown adaptive neuron region: {region_id}") from exc
        proposal = region.propose_unit_add(
            unit_id=unit_id,
            evidence_ids=evidence_ids,
            source_region_id=source_region_id,
            parent_checkpoint_id=parent_checkpoint_id,
            resource_cost=resource_cost,
        )
        if proposal.parent_checkpoint_id is None:
            proposal = replace(
                proposal,
                parent_checkpoint_id=f"neuron-parent:{proposal.proposal_id}",
            )
        return proposal

    def propose_region_growth_from_error(
        self,
        *,
        network_id: str,
        bottleneck_region_id: str,
        input_dim: int,
        unit_count: int,
        fan_in: int,
        prediction_error: float,
        resource_state: float,
        holdout_transfer: float,
        evidence_ids: Sequence[str],
    ) -> StructuralTopologyProposal | None:
        """Turn persistent regional pressure into a ledger-ready region birth."""

        if self._structural_growth_controller is None:
            raise RuntimeError("structural growth controller is not attached")
        try:
            network = self._neuron_networks[str(network_id)]
        except KeyError as exc:
            raise ValueError(f"unknown adaptive neuron network: {network_id}") from exc
        region_id = str(bottleneck_region_id)
        if region_id not in network.region_ids:
            raise ValueError(f"unknown adaptive network region: {bottleneck_region_id}")
        decision = self._structural_growth_controller.observe(
            region_id,
            prediction_error=prediction_error,
            resource_state=resource_state,
            holdout_transfer=holdout_transfer,
            evidence_ids=evidence_ids,
        )
        if not decision.should_grow:
            return None
        child_region_id = self._structural_growth_controller.next_region_id(
            region_id,
            network.region_ids,
        )
        return network.propose_region_add(
            region_id=child_region_id,
            input_dim=input_dim,
            unit_count=unit_count,
            fan_in=fan_in,
            evidence_ids=decision.evidence_ids,
            parent_checkpoint_id=(f"region-growth-signal:{region_id}:{decision.proposal_ordinal}"),
            resource_cost=self._structural_growth_controller.dynamics.growth_resource_cost,
        )

    def commit_region_add(
        self,
        network_id: str,
        proposal: StructuralTopologyProposal,
    ) -> bool:
        """Commit one region birth transactionally through the runtime ledger."""

        existing = self._topology_proposals.get(proposal.proposal_id)
        if existing is not None:
            if existing.status == "accepted":
                return True
            if existing.status in {"rejected", "rolled_back"}:
                return False
        if proposal.status != "pending":
            raise ValueError("only pending region proposals can be committed")
        specification = dict(proposal.specification)
        if (
            proposal.target_kind != "region"
            or proposal.operation != "add"
            or specification.get("topology_role") != "region"
        ):
            raise ValueError("proposal is not a region add")
        try:
            network = self._neuron_networks[str(network_id)]
        except KeyError as exc:
            raise ValueError(f"unknown adaptive neuron network: {network_id}") from exc
        if self._cognitive_state.development.structural_budget < proposal.resource_cost:
            rejected = replace(proposal, status="rejected")
            self._topology_proposals[proposal.proposal_id] = rejected
            self._record_topology_development(
                rejected,
                consume_budget=False,
                evidence_id=(
                    proposal.evidence_ids[0] if proposal.evidence_ids else proposal.proposal_id
                ),
                source="region-topology-growth",
            )
            return False

        parent_snapshot = network.to_payload()
        try:
            network.apply_region_proposal(proposal, generator=self._rng)
            accepted_snapshot = network.to_payload()
            trial = AdaptiveNeuronNetwork.from_payload(
                accepted_snapshot,
                generator=self._checkpoint_region_generator(),
                device=self.device,
            )
            if (
                trial.region_ids != network.region_ids
                or trial.execution_order != network.execution_order
                or trial.connection_ids != network.connection_ids
            ):
                raise ValueError("region checkpoint roundtrip changed topology identities")
            current = network.regions[-1]
            restored = trial.regions[-1]
            if current.unit_ids != restored.unit_ids:
                raise ValueError("region checkpoint roundtrip changed unit identities")
        except (IndexError, KeyError, RuntimeError, ValueError):
            self._neuron_networks[str(network_id)] = AdaptiveNeuronNetwork.from_payload(
                parent_snapshot,
                generator=self._checkpoint_region_generator(),
                device=self.device,
            )
            rejected = replace(proposal, status="rejected")
            self._topology_proposals[proposal.proposal_id] = rejected
            self._record_topology_development(
                rejected,
                consume_budget=False,
                evidence_id=(
                    proposal.evidence_ids[0] if proposal.evidence_ids else proposal.proposal_id
                ),
                source="region-topology-growth",
            )
            return False

        accepted = replace(proposal, status="accepted", validation_score=0.0)
        self._topology_proposals[proposal.proposal_id] = accepted
        self._topology_parent_snapshots[proposal.proposal_id] = parent_snapshot
        self._record_topology_development(
            accepted,
            consume_budget=True,
            evidence_id=(
                proposal.evidence_ids[0] if proposal.evidence_ids else proposal.proposal_id
            ),
            source="region-topology-growth",
        )
        return True

    def rollback_region_add(self, proposal_id: str) -> bool:
        """Rollback only the latest accepted region birth."""

        key = str(proposal_id)
        proposal = self._topology_proposals.get(key)
        snapshot = self._topology_parent_snapshots.get(key)
        if (
            proposal is None
            or proposal.status != "accepted"
            or proposal.target_kind != "region"
            or dict(proposal.specification).get("topology_role") != "region"
            or snapshot is None
        ):
            return False
        active = [
            item.proposal_id
            for item in self._topology_proposals.values()
            if item.status == "accepted" and item.proposal_id in self._topology_parent_snapshots
        ]
        if not active or active[-1] != key:
            return False
        network_id = next(
            (
                network_key
                for network_key, network in self._neuron_networks.items()
                if proposal.substrate_id in network.region_ids
            ),
            None,
        )
        if network_id is None:
            return False
        self._neuron_networks[network_id] = AdaptiveNeuronNetwork.from_payload(
            snapshot,
            generator=self._checkpoint_region_generator(),
            device=self.device,
        )
        rolled_back = replace(proposal, status="rolled_back", validation_score=0.0)
        self._topology_proposals[key] = rolled_back
        self._topology_parent_snapshots.pop(key, None)
        previous = self._cognitive_state.development
        self._cognitive_state = replace(
            self._cognitive_state,
            development=replace(
                previous,
                tick=self.tick,
                structural_budget=previous.structural_budget + proposal.resource_cost,
                last_update_source="region-topology-rollback",
                last_validation_status="rolled_back",
                validation_evidence_ids=self._bounded_ids(
                    previous.validation_evidence_ids,
                    key,
                    limit=self._lineage_limit(),
                ),
                growth_count=max(0, previous.growth_count - proposal.requested_units),
                lineage=self._bounded_ids(
                    previous.lineage,
                    key,
                    limit=self._lineage_limit(),
                ),
            ),
        )
        return True

    def propose_region_prune_from_underuse(
        self,
        *,
        network_id: str,
        region_id: str,
        usage: float,
        resource_pressure: float,
        learning_gain: float,
        evidence_ids: Sequence[str],
    ) -> StructuralTopologyProposal | None:
        """Turn persistent underuse and resource pressure into a prune proposal."""

        if self._structural_pruning_controller is None:
            raise RuntimeError("structural pruning controller is not attached")
        try:
            network = self._neuron_networks[str(network_id)]
        except KeyError as exc:
            raise ValueError(f"unknown adaptive neuron network: {network_id}") from exc
        region_key = str(region_id)
        if region_key not in network.region_ids:
            raise ValueError(f"unknown adaptive network region: {region_id}")
        decision: StructuralPruningDecision = self._structural_pruning_controller.observe(
            region_key,
            usage=usage,
            resource_pressure=resource_pressure,
            learning_gain=learning_gain,
            evidence_ids=evidence_ids,
        )
        if not decision.should_prune:
            return None
        proposal = network.propose_region_prune(
            region_id=region_key,
            evidence_ids=decision.evidence_ids,
            parent_checkpoint_id=(f"region-prune-signal:{region_key}:{decision.proposal_ordinal}"),
            resource_cost=self._structural_pruning_controller.dynamics.pruning_resource_cost,
        )
        self._topology_proposals[proposal.proposal_id] = proposal
        self._topology_network_ids[proposal.proposal_id] = str(network_id)
        return proposal

    def propose_cross_region_connection_prune_from_underuse(
        self,
        *,
        network_id: str,
        connection_id: str,
        usage: float,
        resource_pressure: float,
        learning_gain: float,
        evidence_ids: Sequence[str],
    ) -> StructuralTopologyProposal | None:
        """Turn persistent route underuse and stagnation into a prune proposal."""

        if self._structural_pruning_controller is None:
            raise RuntimeError("structural pruning controller is not attached")
        try:
            network = self._neuron_networks[str(network_id)]
        except KeyError as exc:
            raise ValueError(f"unknown adaptive neuron network: {network_id}") from exc
        connection_key = str(connection_id)
        if connection_key not in network.connection_ids:
            raise ValueError(f"unknown adaptive network connection: {connection_id}")
        decision: StructuralPruningDecision = self._structural_pruning_controller.observe_substrate(
            connection_key,
            usage=usage,
            resource_pressure=resource_pressure,
            learning_gain=learning_gain,
            evidence_ids=evidence_ids,
        )
        if not decision.should_prune:
            return None
        proposal = network.propose_connection_prune(
            connection_id=connection_key,
            evidence_ids=decision.evidence_ids,
            parent_checkpoint_id=(
                f"connection-prune-signal:{connection_key}:{decision.proposal_ordinal}"
            ),
            resource_cost=self._structural_pruning_controller.dynamics.pruning_resource_cost,
        )
        self._topology_proposals[proposal.proposal_id] = proposal
        self._topology_network_ids[proposal.proposal_id] = str(network_id)
        return proposal

    def propose_cross_region_connection_prune_from_route(
        self,
        *,
        network_id: str,
        connection_id: str,
        evidence_ids: Sequence[str],
    ) -> StructuralTopologyProposal | None:
        """Derive route maintenance evidence from the existing cooperation learner."""

        try:
            network = self._neuron_networks[str(network_id)]
        except KeyError as exc:
            raise ValueError(f"unknown adaptive neuron network: {network_id}") from exc
        learner = network.cooperation_learner
        if learner is None:
            raise RuntimeError("cross-region cooperation learner is not attached")
        route = learner.route_state(connection_id)
        evidence_count = max(1, int(route.evidence_count))
        usage = min(1.0, float(route.selection_count) / float(evidence_count))
        resource_pressure = 1.0 - float(route.resource_state)
        learning_gain = float(route.holdout_transfer) * (1.0 - float(route.prediction_error))
        return self.propose_cross_region_connection_prune_from_underuse(
            network_id=network_id,
            connection_id=connection_id,
            usage=usage,
            resource_pressure=resource_pressure,
            learning_gain=learning_gain,
            evidence_ids=evidence_ids,
        )

    def validate_cross_region_connection_prune_holdout(
        self,
        *,
        network_id: str,
        proposal_id: str,
        holdout_inputs: Sequence[Mapping[str, torch.Tensor]],
        expected_activities: Sequence[Mapping[str, torch.Tensor]],
    ) -> bool:
        """Validate that removing a route does not regress unseen network behavior."""

        if self._structural_pruning_controller is None:
            raise RuntimeError("structural pruning controller is not attached")
        proposal = self._topology_proposals.get(str(proposal_id))
        if (
            proposal is None
            or proposal.status != "pending"
            or proposal.target_kind != "region"
            or proposal.operation != "prune"
            or dict(proposal.specification).get("topology_role") != "cross_region_connection_prune"
        ):
            raise ValueError("proposal is not a pending cross-region connection prune")
        if len(holdout_inputs) == 0 or len(holdout_inputs) != len(expected_activities):
            raise ValueError(
                "connection prune holdout inputs and expected activities must have equal size"
            )
        try:
            network = self._neuron_networks[str(network_id)]
        except KeyError as exc:
            raise ValueError(f"unknown adaptive neuron network: {network_id}") from exc
        if self._topology_network_ids.get(str(proposal_id)) != str(network_id):
            raise ValueError("connection prune proposal belongs to another network")
        payload = network.to_payload()
        intact = AdaptiveNeuronNetwork.from_payload(
            payload,
            generator=self._checkpoint_region_generator(),
            device=self.device,
        )
        pruned = AdaptiveNeuronNetwork.from_payload(
            payload,
            generator=self._checkpoint_region_generator(),
            device=self.device,
        )
        pruned.apply_connection_prune(proposal)
        expected_region_ids = set(pruned.region_ids)
        baseline_errors: list[float] = []
        pruned_errors: list[float] = []
        for inputs, expected in zip(holdout_inputs, expected_activities, strict=True):
            if not expected or not set(expected).issubset(expected_region_ids):
                raise ValueError(
                    "connection prune expected activities must target surviving regions"
                )
            intact_activities = intact.step(
                inputs,
                connection_ids=intact.connection_ids,
            )
            pruned_activities = pruned.step(
                inputs,
                connection_ids=pruned.connection_ids,
            )
            for expected_region_id, expected_activity in expected.items():
                expected_value = expected_activity.to(self.device)
                intact_activity = intact_activities[expected_region_id]
                pruned_activity = pruned_activities[expected_region_id]
                if expected_value.shape != intact_activity.shape:
                    raise ValueError(
                        f"connection prune expected activity shape mismatch for {expected_region_id}"
                    )
                baseline_errors.append(
                    float(
                        torch.mean(torch.abs(intact_activity - expected_value))
                        .clamp(0.0, 1.0)
                        .item()
                    )
                )
                pruned_errors.append(
                    float(
                        torch.mean(torch.abs(pruned_activity - expected_value))
                        .clamp(0.0, 1.0)
                        .item()
                    )
                )
        if not baseline_errors:
            raise ValueError("connection prune holdout must contain expected activities")
        baseline_error = sum(baseline_errors) / len(baseline_errors)
        pruned_error = sum(pruned_errors) / len(pruned_errors)
        regression = max(0.0, pruned_error - baseline_error)
        score = max(0.0, min(1.0, 1.0 - regression))
        maximum_regression = self._structural_pruning_controller.dynamics.maximum_holdout_regression
        validated = regression <= float(maximum_regression)
        self._topology_proposals[str(proposal_id)] = replace(
            proposal,
            validation_score=score,
        )
        previous = self._cognitive_state.development
        self._cognitive_state = replace(
            self._cognitive_state,
            development=replace(
                previous,
                tick=self.tick,
                last_update_source="connection-prune-holdout-validation",
                last_validation_status="validated" if validated else "rejected",
                validation_evidence_ids=self._bounded_ids(
                    previous.validation_evidence_ids,
                    f"holdout:{proposal_id}",
                    limit=self._lineage_limit(),
                ),
                lineage=self._bounded_ids(
                    previous.lineage,
                    f"holdout:{proposal_id}",
                    limit=self._lineage_limit(),
                ),
            ),
        )
        return validated

    def commit_cross_region_connection_prune(
        self,
        network_id: str,
        proposal: StructuralTopologyProposal,
    ) -> bool:
        """Commit a validated route removal through the runtime ledger."""

        existing = self._topology_proposals.get(proposal.proposal_id)
        if existing is not None:
            if existing.status == "accepted":
                return True
            if existing.status in {"rejected", "rolled_back"}:
                return False
            proposal = existing
        if proposal.status != "pending":
            raise ValueError("only pending connection prune proposals can be committed")
        if (
            proposal.target_kind != "region"
            or proposal.operation != "prune"
            or dict(proposal.specification).get("topology_role") != "cross_region_connection_prune"
        ):
            raise ValueError("proposal is not a cross-region connection prune")
        if self._structural_pruning_controller is None:
            raise RuntimeError("structural pruning controller is not attached")
        if self._topology_network_ids.get(proposal.proposal_id) not in {
            None,
            str(network_id),
        }:
            raise ValueError("connection prune proposal belongs to another network")
        minimum_score = 1.0 - float(
            self._structural_pruning_controller.dynamics.maximum_holdout_regression
        )
        if proposal.validation_score < minimum_score:
            rejected = replace(proposal, status="rejected")
            self._topology_proposals[proposal.proposal_id] = rejected
            self._record_topology_development(
                rejected,
                consume_budget=False,
                evidence_id=(
                    proposal.evidence_ids[0] if proposal.evidence_ids else proposal.proposal_id
                ),
                source="cross-region-topology-pruning",
            )
            return False
        try:
            network = self._neuron_networks[str(network_id)]
        except KeyError as exc:
            raise ValueError(f"unknown adaptive neuron network: {network_id}") from exc
        if self._cognitive_state.development.structural_budget < proposal.resource_cost:
            rejected = replace(proposal, status="rejected")
            self._topology_proposals[proposal.proposal_id] = rejected
            self._record_topology_development(
                rejected,
                consume_budget=False,
                evidence_id=(
                    proposal.evidence_ids[0] if proposal.evidence_ids else proposal.proposal_id
                ),
                source="cross-region-topology-pruning",
            )
            return False
        parent_snapshot = network.to_payload()
        try:
            network.apply_connection_prune(proposal)
            accepted_snapshot = network.to_payload()
            trial = AdaptiveNeuronNetwork.from_payload(
                accepted_snapshot,
                generator=self._checkpoint_region_generator(),
                device=self.device,
            )
            if (
                trial.region_ids != network.region_ids
                or trial.execution_order != network.execution_order
                or trial.connection_ids != network.connection_ids
            ):
                raise ValueError(
                    "connection prune checkpoint roundtrip changed topology identities"
                )
        except (IndexError, KeyError, RuntimeError, ValueError):
            self._neuron_networks[str(network_id)] = AdaptiveNeuronNetwork.from_payload(
                parent_snapshot,
                generator=self._checkpoint_region_generator(),
                device=self.device,
            )
            rejected = replace(proposal, status="rejected")
            self._topology_proposals[proposal.proposal_id] = rejected
            self._record_topology_development(
                rejected,
                consume_budget=False,
                evidence_id=(
                    proposal.evidence_ids[0] if proposal.evidence_ids else proposal.proposal_id
                ),
                source="cross-region-topology-pruning",
            )
            return False
        accepted = replace(proposal, status="accepted")
        self._topology_proposals[proposal.proposal_id] = accepted
        self._topology_parent_snapshots[proposal.proposal_id] = parent_snapshot
        self._topology_network_ids[proposal.proposal_id] = str(network_id)
        self._record_topology_development(
            accepted,
            consume_budget=True,
            evidence_id=(
                proposal.evidence_ids[0] if proposal.evidence_ids else proposal.proposal_id
            ),
            source="cross-region-topology-pruning",
            counter="prune_count",
        )
        return True

    def rollback_cross_region_connection_prune(self, proposal_id: str) -> bool:
        """Rollback only the latest accepted cross-region connection removal."""

        key = str(proposal_id)
        proposal = self._topology_proposals.get(key)
        snapshot = self._topology_parent_snapshots.get(key)
        network_id = self._topology_network_ids.get(key)
        if (
            proposal is None
            or proposal.status != "accepted"
            or proposal.target_kind != "region"
            or proposal.operation != "prune"
            or dict(proposal.specification).get("topology_role") != "cross_region_connection_prune"
            or snapshot is None
            or network_id is None
        ):
            return False
        active = [
            item.proposal_id
            for item in self._topology_proposals.values()
            if item.status == "accepted" and item.proposal_id in self._topology_parent_snapshots
        ]
        if not active or active[-1] != key:
            return False
        self._neuron_networks[network_id] = AdaptiveNeuronNetwork.from_payload(
            snapshot,
            generator=self._checkpoint_region_generator(),
            device=self.device,
        )
        rolled_back = replace(proposal, status="rolled_back", validation_score=0.0)
        self._topology_proposals[key] = rolled_back
        self._topology_parent_snapshots.pop(key, None)
        self._topology_network_ids.pop(key, None)
        previous = self._cognitive_state.development
        self._cognitive_state = replace(
            self._cognitive_state,
            development=replace(
                previous,
                tick=self.tick,
                structural_budget=previous.structural_budget + proposal.resource_cost,
                last_update_source="cross-region-topology-prune-rollback",
                last_validation_status="rolled_back",
                validation_evidence_ids=self._bounded_ids(
                    previous.validation_evidence_ids,
                    key,
                    limit=self._lineage_limit(),
                ),
                prune_count=max(0, previous.prune_count - proposal.requested_units),
                lineage=self._bounded_ids(
                    previous.lineage,
                    key,
                    limit=self._lineage_limit(),
                ),
            ),
        )
        return True

    def validate_region_prune_holdout(
        self,
        *,
        network_id: str,
        proposal_id: str,
        holdout_inputs: Sequence[Mapping[str, torch.Tensor]],
        expected_activities: Sequence[Mapping[str, torch.Tensor]],
    ) -> bool:
        """Validate that removing a region does not regress unseen network behavior."""

        if self._structural_pruning_controller is None:
            raise RuntimeError("structural pruning controller is not attached")
        proposal = self._topology_proposals.get(str(proposal_id))
        if (
            proposal is None
            or proposal.status != "pending"
            or proposal.target_kind != "region"
            or proposal.operation != "prune"
            or dict(proposal.specification).get("topology_role") != "region_prune"
        ):
            raise ValueError("proposal is not a pending region prune")
        if len(holdout_inputs) == 0 or len(holdout_inputs) != len(expected_activities):
            raise ValueError(
                "region prune holdout inputs and expected activities must have equal size"
            )
        try:
            network = self._neuron_networks[str(network_id)]
        except KeyError as exc:
            raise ValueError(f"unknown adaptive neuron network: {network_id}") from exc
        if self._topology_network_ids.get(str(proposal_id)) != str(network_id):
            raise ValueError("region prune proposal belongs to another network")
        payload = network.to_payload()
        intact = AdaptiveNeuronNetwork.from_payload(
            payload,
            generator=self._checkpoint_region_generator(),
            device=self.device,
        )
        pruned = AdaptiveNeuronNetwork.from_payload(
            payload,
            generator=self._checkpoint_region_generator(),
            device=self.device,
        )
        pruned.apply_region_prune(proposal)
        expected_region_ids = set(pruned.region_ids)
        baseline_errors: list[float] = []
        pruned_errors: list[float] = []
        for inputs, expected in zip(holdout_inputs, expected_activities, strict=True):
            if not expected or not set(expected).issubset(expected_region_ids):
                raise ValueError("region prune expected activities must target surviving regions")
            intact_activities = intact.step(
                inputs,
                connection_ids=intact.connection_ids,
            )
            pruned_activities = pruned.step(
                inputs,
                connection_ids=pruned.connection_ids,
            )
            for expected_region_id, expected_activity in expected.items():
                expected_value = expected_activity.to(self.device)
                intact_activity = intact_activities[expected_region_id]
                pruned_activity = pruned_activities[expected_region_id]
                if expected_value.shape != intact_activity.shape:
                    raise ValueError(
                        f"region prune expected activity shape mismatch for {expected_region_id}"
                    )
                baseline_errors.append(
                    float(
                        torch.mean(torch.abs(intact_activity - expected_value))
                        .clamp(0.0, 1.0)
                        .item()
                    )
                )
                pruned_errors.append(
                    float(
                        torch.mean(torch.abs(pruned_activity - expected_value))
                        .clamp(0.0, 1.0)
                        .item()
                    )
                )
        if not baseline_errors:
            raise ValueError("region prune holdout must contain at least one expected activity")
        baseline_error = sum(baseline_errors) / len(baseline_errors)
        pruned_error = sum(pruned_errors) / len(pruned_errors)
        regression = max(0.0, pruned_error - baseline_error)
        score = max(0.0, min(1.0, 1.0 - regression))
        maximum_regression = self._structural_pruning_controller.dynamics.maximum_holdout_regression
        validated = regression <= float(maximum_regression)
        self._topology_proposals[str(proposal_id)] = replace(
            proposal,
            validation_score=score,
        )
        previous = self._cognitive_state.development
        self._cognitive_state = replace(
            self._cognitive_state,
            development=replace(
                previous,
                tick=self.tick,
                last_update_source="region-prune-holdout-validation",
                last_validation_status="validated" if validated else "rejected",
                validation_evidence_ids=self._bounded_ids(
                    previous.validation_evidence_ids,
                    f"holdout:{proposal_id}",
                    limit=self._lineage_limit(),
                ),
                lineage=self._bounded_ids(
                    previous.lineage,
                    f"holdout:{proposal_id}",
                    limit=self._lineage_limit(),
                ),
            ),
        )
        return validated

    def commit_region_prune(
        self,
        network_id: str,
        proposal: StructuralTopologyProposal,
    ) -> bool:
        """Commit a validated region removal through the runtime ledger."""

        existing = self._topology_proposals.get(proposal.proposal_id)
        if existing is not None:
            if existing.status == "accepted":
                return True
            if existing.status == "rejected" or existing.status == "rolled_back":
                return False
            proposal = existing
        if proposal.status != "pending":
            raise ValueError("only pending region prune proposals can be committed")
        if (
            proposal.target_kind != "region"
            or proposal.operation != "prune"
            or dict(proposal.specification).get("topology_role") != "region_prune"
        ):
            raise ValueError("proposal is not a region prune")
        if self._structural_pruning_controller is None:
            raise RuntimeError("structural pruning controller is not attached")
        if self._topology_network_ids.get(proposal.proposal_id) not in {None, str(network_id)}:
            raise ValueError("region prune proposal belongs to another network")
        minimum_score = 1.0 - float(
            self._structural_pruning_controller.dynamics.maximum_holdout_regression
        )
        if proposal.validation_score < minimum_score:
            rejected = replace(proposal, status="rejected")
            self._topology_proposals[proposal.proposal_id] = rejected
            self._record_topology_development(
                rejected,
                consume_budget=False,
                evidence_id=(
                    proposal.evidence_ids[0] if proposal.evidence_ids else proposal.proposal_id
                ),
                source="region-topology-pruning",
            )
            return False
        try:
            network = self._neuron_networks[str(network_id)]
        except KeyError as exc:
            raise ValueError(f"unknown adaptive neuron network: {network_id}") from exc
        if self._cognitive_state.development.structural_budget < proposal.resource_cost:
            rejected = replace(proposal, status="rejected")
            self._topology_proposals[proposal.proposal_id] = rejected
            self._record_topology_development(
                rejected,
                consume_budget=False,
                evidence_id=(
                    proposal.evidence_ids[0] if proposal.evidence_ids else proposal.proposal_id
                ),
                source="region-topology-pruning",
            )
            return False
        parent_snapshot = network.to_payload()
        try:
            network.apply_region_prune(proposal)
            accepted_snapshot = network.to_payload()
            trial = AdaptiveNeuronNetwork.from_payload(
                accepted_snapshot,
                generator=self._checkpoint_region_generator(),
                device=self.device,
            )
            if (
                trial.region_ids != network.region_ids
                or trial.execution_order != network.execution_order
                or trial.connection_ids != network.connection_ids
            ):
                raise ValueError("region prune checkpoint roundtrip changed topology identities")
        except (IndexError, KeyError, RuntimeError, ValueError):
            self._neuron_networks[str(network_id)] = AdaptiveNeuronNetwork.from_payload(
                parent_snapshot,
                generator=self._checkpoint_region_generator(),
                device=self.device,
            )
            rejected = replace(proposal, status="rejected")
            self._topology_proposals[proposal.proposal_id] = rejected
            self._record_topology_development(
                rejected,
                consume_budget=False,
                evidence_id=(
                    proposal.evidence_ids[0] if proposal.evidence_ids else proposal.proposal_id
                ),
                source="region-topology-pruning",
            )
            return False
        accepted = replace(proposal, status="accepted")
        self._topology_proposals[proposal.proposal_id] = accepted
        self._topology_parent_snapshots[proposal.proposal_id] = parent_snapshot
        self._topology_network_ids[proposal.proposal_id] = str(network_id)
        self._record_topology_development(
            accepted,
            consume_budget=True,
            evidence_id=(
                proposal.evidence_ids[0] if proposal.evidence_ids else proposal.proposal_id
            ),
            source="region-topology-pruning",
            counter="prune_count",
        )
        return True

    def rollback_region_prune(self, proposal_id: str) -> bool:
        """Rollback only the latest accepted region removal."""

        key = str(proposal_id)
        proposal = self._topology_proposals.get(key)
        snapshot = self._topology_parent_snapshots.get(key)
        network_id = self._topology_network_ids.get(key)
        if (
            proposal is None
            or proposal.status != "accepted"
            or proposal.target_kind != "region"
            or proposal.operation != "prune"
            or dict(proposal.specification).get("topology_role") != "region_prune"
            or snapshot is None
            or network_id is None
        ):
            return False
        active = [
            item.proposal_id
            for item in self._topology_proposals.values()
            if item.status == "accepted" and item.proposal_id in self._topology_parent_snapshots
        ]
        if not active or active[-1] != key:
            return False
        self._neuron_networks[network_id] = AdaptiveNeuronNetwork.from_payload(
            snapshot,
            generator=self._checkpoint_region_generator(),
            device=self.device,
        )
        rolled_back = replace(proposal, status="rolled_back", validation_score=0.0)
        self._topology_proposals[key] = rolled_back
        self._topology_parent_snapshots.pop(key, None)
        self._topology_network_ids.pop(key, None)
        previous = self._cognitive_state.development
        self._cognitive_state = replace(
            self._cognitive_state,
            development=replace(
                previous,
                tick=self.tick,
                structural_budget=previous.structural_budget + proposal.resource_cost,
                last_update_source="region-topology-prune-rollback",
                last_validation_status="rolled_back",
                validation_evidence_ids=self._bounded_ids(
                    previous.validation_evidence_ids,
                    key,
                    limit=self._lineage_limit(),
                ),
                prune_count=max(0, previous.prune_count - proposal.requested_units),
                lineage=self._bounded_ids(
                    previous.lineage,
                    key,
                    limit=self._lineage_limit(),
                ),
            ),
        )
        return True

    def propose_region_merge_from_redundancy(
        self,
        *,
        network_id: str,
        region_ids: Sequence[str],
        usage: float,
        resource_pressure: float,
        learning_gain: float,
        evidence_ids: Sequence[str],
    ) -> StructuralTopologyProposal | None:
        """Turn persistent redundant substrate pressure into a merge proposal."""

        if self._structural_pruning_controller is None:
            raise RuntimeError("structural pruning controller is not attached")
        try:
            network = self._neuron_networks[str(network_id)]
        except KeyError as exc:
            raise ValueError(f"unknown adaptive neuron network: {network_id}") from exc
        selected = tuple(str(item) for item in region_ids)
        for region_id in selected:
            if region_id not in network.region_ids:
                raise ValueError(f"unknown adaptive network region: {region_id}")
        substrate_id = "merge:" + "+".join(selected)
        decision = self._structural_pruning_controller.observe_substrate(
            substrate_id,
            usage=usage,
            resource_pressure=resource_pressure,
            learning_gain=learning_gain,
            evidence_ids=evidence_ids,
        )
        if not decision.should_prune:
            return None
        proposal = network.propose_region_merge(
            region_ids=selected,
            evidence_ids=decision.evidence_ids,
            parent_checkpoint_id=(
                f"region-merge-signal:{substrate_id}:{decision.proposal_ordinal}"
            ),
            resource_cost=self._structural_pruning_controller.dynamics.pruning_resource_cost,
        )
        self._topology_proposals[proposal.proposal_id] = proposal
        self._topology_network_ids[proposal.proposal_id] = str(network_id)
        return proposal

    def validate_region_merge_holdout(
        self,
        *,
        network_id: str,
        proposal_id: str,
        holdout_inputs: Sequence[Mapping[str, torch.Tensor]],
        expected_activities: Sequence[Mapping[str, torch.Tensor]],
    ) -> bool:
        """Validate a merge against unseen inputs in the combined unit space."""

        if self._structural_pruning_controller is None:
            raise RuntimeError("structural pruning controller is not attached")
        proposal = self._topology_proposals.get(str(proposal_id))
        if (
            proposal is None
            or proposal.status != "pending"
            or proposal.target_kind != "region"
            or proposal.operation != "merge"
            or dict(proposal.specification).get("topology_role") != "region_merge"
        ):
            raise ValueError("proposal is not a pending region merge")
        if len(holdout_inputs) == 0 or len(holdout_inputs) != len(expected_activities):
            raise ValueError(
                "region merge holdout inputs and expected activities must have equal size"
            )
        try:
            network = self._neuron_networks[str(network_id)]
        except KeyError as exc:
            raise ValueError(f"unknown adaptive neuron network: {network_id}") from exc
        if self._topology_network_ids.get(str(proposal_id)) != str(network_id):
            raise ValueError("region merge proposal belongs to another network")
        specification = dict(proposal.specification)
        selected = tuple(str(item) for item in specification.get("region_ids", ()))
        if len(selected) != 2:
            raise ValueError("region merge proposal must name two regions")
        retained_id, absorbed_id = selected
        payload = network.to_payload()
        intact = AdaptiveNeuronNetwork.from_payload(
            payload,
            generator=self._checkpoint_region_generator(),
            device=self.device,
        )
        merged = AdaptiveNeuronNetwork.from_payload(
            payload,
            generator=self._checkpoint_region_generator(),
            device=self.device,
        )
        merged.apply_region_merge(proposal, generator=self._checkpoint_region_generator())
        baseline_errors: list[float] = []
        merged_errors: list[float] = []
        for inputs, expected in zip(holdout_inputs, expected_activities, strict=True):
            if not expected:
                raise ValueError("region merge holdout must contain expected activities")
            if retained_id not in inputs or absorbed_id not in inputs:
                raise ValueError("region merge holdout inputs must contain both source regions")
            merged_inputs = dict(inputs)
            merged_inputs.pop(absorbed_id)
            intact_activities = intact.step(inputs)
            merged_activities = merged.step(merged_inputs)
            for expected_region_id, expected_activity in expected.items():
                expected_value = expected_activity.to(self.device)
                if expected_region_id == retained_id:
                    intact_activity = torch.cat(
                        (intact_activities[retained_id], intact_activities[absorbed_id])
                    )
                    merged_activity = merged_activities[retained_id]
                else:
                    if expected_region_id not in merged_activities:
                        raise ValueError("region merge expected activity targets a missing region")
                    intact_activity = intact_activities[expected_region_id]
                    merged_activity = merged_activities[expected_region_id]
                if expected_value.shape != intact_activity.shape:
                    raise ValueError(
                        f"region merge expected activity shape mismatch for {expected_region_id}"
                    )
                baseline_errors.append(
                    float(
                        torch.mean(torch.abs(intact_activity - expected_value))
                        .clamp(0.0, 1.0)
                        .item()
                    )
                )
                merged_errors.append(
                    float(
                        torch.mean(torch.abs(merged_activity - expected_value))
                        .clamp(0.0, 1.0)
                        .item()
                    )
                )
        baseline_error = sum(baseline_errors) / len(baseline_errors)
        merged_error = sum(merged_errors) / len(merged_errors)
        regression = max(0.0, merged_error - baseline_error)
        score = max(0.0, min(1.0, 1.0 - regression))
        maximum_regression = self._structural_pruning_controller.dynamics.maximum_holdout_regression
        validated = regression <= float(maximum_regression)
        self._topology_proposals[str(proposal_id)] = replace(
            proposal,
            validation_score=score,
        )
        previous = self._cognitive_state.development
        self._cognitive_state = replace(
            self._cognitive_state,
            development=replace(
                previous,
                tick=self.tick,
                last_update_source="region-merge-holdout-validation",
                last_validation_status="validated" if validated else "rejected",
                validation_evidence_ids=self._bounded_ids(
                    previous.validation_evidence_ids,
                    f"holdout:{proposal_id}",
                    limit=self._lineage_limit(),
                ),
                lineage=self._bounded_ids(
                    previous.lineage,
                    f"holdout:{proposal_id}",
                    limit=self._lineage_limit(),
                ),
            ),
        )
        return validated

    def commit_region_merge(
        self,
        network_id: str,
        proposal: StructuralTopologyProposal,
    ) -> bool:
        """Commit a validated region merge through the runtime ledger."""

        existing = self._topology_proposals.get(proposal.proposal_id)
        if existing is not None:
            if existing.status == "accepted":
                return True
            if existing.status in {"rejected", "rolled_back"}:
                return False
            proposal = existing
        if proposal.status != "pending":
            raise ValueError("only pending region merge proposals can be committed")
        if (
            proposal.target_kind != "region"
            or proposal.operation != "merge"
            or dict(proposal.specification).get("topology_role") != "region_merge"
        ):
            raise ValueError("proposal is not a region merge")
        if self._structural_pruning_controller is None:
            raise RuntimeError("structural pruning controller is not attached")
        if self._topology_network_ids.get(proposal.proposal_id) not in {
            None,
            str(network_id),
        }:
            raise ValueError("region merge proposal belongs to another network")
        minimum_score = 1.0 - float(
            self._structural_pruning_controller.dynamics.maximum_holdout_regression
        )
        if proposal.validation_score < minimum_score:
            rejected = replace(proposal, status="rejected")
            self._topology_proposals[proposal.proposal_id] = rejected
            self._record_topology_development(
                rejected,
                consume_budget=False,
                evidence_id=(
                    proposal.evidence_ids[0] if proposal.evidence_ids else proposal.proposal_id
                ),
                source="region-topology-merge",
            )
            return False
        try:
            network = self._neuron_networks[str(network_id)]
        except KeyError as exc:
            raise ValueError(f"unknown adaptive neuron network: {network_id}") from exc
        if self._cognitive_state.development.structural_budget < proposal.resource_cost:
            rejected = replace(proposal, status="rejected")
            self._topology_proposals[proposal.proposal_id] = rejected
            self._record_topology_development(
                rejected,
                consume_budget=False,
                evidence_id=(
                    proposal.evidence_ids[0] if proposal.evidence_ids else proposal.proposal_id
                ),
                source="region-topology-merge",
            )
            return False
        parent_snapshot = network.to_payload()
        try:
            network.apply_region_merge(
                proposal,
                generator=self._checkpoint_region_generator(),
            )
            accepted_snapshot = network.to_payload()
            trial = AdaptiveNeuronNetwork.from_payload(
                accepted_snapshot,
                generator=self._checkpoint_region_generator(),
                device=self.device,
            )
            if (
                trial.region_ids != network.region_ids
                or trial.execution_order != network.execution_order
                or trial.connection_ids != network.connection_ids
            ):
                raise ValueError("region merge checkpoint roundtrip changed topology identities")
        except (IndexError, KeyError, RuntimeError, ValueError):
            self._neuron_networks[str(network_id)] = AdaptiveNeuronNetwork.from_payload(
                parent_snapshot,
                generator=self._checkpoint_region_generator(),
                device=self.device,
            )
            rejected = replace(proposal, status="rejected")
            self._topology_proposals[proposal.proposal_id] = rejected
            self._record_topology_development(
                rejected,
                consume_budget=False,
                evidence_id=(
                    proposal.evidence_ids[0] if proposal.evidence_ids else proposal.proposal_id
                ),
                source="region-topology-merge",
            )
            return False
        accepted = replace(proposal, status="accepted")
        self._topology_proposals[proposal.proposal_id] = accepted
        self._topology_parent_snapshots[proposal.proposal_id] = parent_snapshot
        self._topology_network_ids[proposal.proposal_id] = str(network_id)
        self._record_topology_development(
            accepted,
            consume_budget=True,
            evidence_id=(
                proposal.evidence_ids[0] if proposal.evidence_ids else proposal.proposal_id
            ),
            source="region-topology-merge",
            counter="split_merge_count",
        )
        return True

    def rollback_region_merge(self, proposal_id: str) -> bool:
        """Rollback only the latest accepted region merge."""

        key = str(proposal_id)
        proposal = self._topology_proposals.get(key)
        snapshot = self._topology_parent_snapshots.get(key)
        network_id = self._topology_network_ids.get(key)
        if (
            proposal is None
            or proposal.status != "accepted"
            or proposal.target_kind != "region"
            or proposal.operation != "merge"
            or dict(proposal.specification).get("topology_role") != "region_merge"
            or snapshot is None
            or network_id is None
        ):
            return False
        active = [
            item.proposal_id
            for item in self._topology_proposals.values()
            if item.status == "accepted" and item.proposal_id in self._topology_parent_snapshots
        ]
        if not active or active[-1] != key:
            return False
        self._neuron_networks[network_id] = AdaptiveNeuronNetwork.from_payload(
            snapshot,
            generator=self._checkpoint_region_generator(),
            device=self.device,
        )
        rolled_back = replace(proposal, status="rolled_back", validation_score=0.0)
        self._topology_proposals[key] = rolled_back
        self._topology_parent_snapshots.pop(key, None)
        self._topology_network_ids.pop(key, None)
        previous = self._cognitive_state.development
        self._cognitive_state = replace(
            self._cognitive_state,
            development=replace(
                previous,
                tick=self.tick,
                structural_budget=previous.structural_budget + proposal.resource_cost,
                last_update_source="region-topology-merge-rollback",
                last_validation_status="rolled_back",
                validation_evidence_ids=self._bounded_ids(
                    previous.validation_evidence_ids,
                    key,
                    limit=self._lineage_limit(),
                ),
                split_merge_count=max(0, previous.split_merge_count - proposal.requested_units),
                lineage=self._bounded_ids(
                    previous.lineage,
                    key,
                    limit=self._lineage_limit(),
                ),
            ),
        )
        return True

    def propose_region_split_from_error(
        self,
        *,
        network_id: str,
        region_id: str,
        first_unit_count: int,
        prediction_error: float,
        resource_state: float,
        holdout_transfer: float,
        evidence_ids: Sequence[str],
    ) -> StructuralTopologyProposal | None:
        """Turn persistent substrate pressure into a region split proposal."""

        if self._structural_growth_controller is None:
            raise RuntimeError("structural growth controller is not attached")
        try:
            network = self._neuron_networks[str(network_id)]
        except KeyError as exc:
            raise ValueError(f"unknown adaptive neuron network: {network_id}") from exc
        region_key = str(region_id)
        if region_key not in network.region_ids:
            raise ValueError(f"unknown adaptive network region: {region_id}")
        decision = self._structural_growth_controller.observe(
            region_key,
            prediction_error=prediction_error,
            resource_state=resource_state,
            holdout_transfer=holdout_transfer,
            evidence_ids=evidence_ids,
        )
        if not decision.should_grow:
            return None
        proposal = network.propose_region_split(
            region_id=region_key,
            first_unit_count=first_unit_count,
            evidence_ids=decision.evidence_ids,
            parent_checkpoint_id=(f"region-split-signal:{region_key}:{decision.proposal_ordinal}"),
            resource_cost=self._structural_growth_controller.dynamics.growth_resource_cost,
        )
        self._topology_proposals[proposal.proposal_id] = proposal
        self._topology_network_ids[proposal.proposal_id] = str(network_id)
        return proposal

    def validate_region_split_holdout(
        self,
        *,
        network_id: str,
        proposal_id: str,
        holdout_inputs: Sequence[Mapping[str, torch.Tensor]],
        expected_activities: Sequence[Mapping[str, torch.Tensor]],
    ) -> bool:
        """Validate a split against unseen inputs in the parent coordinate space."""

        if self._structural_growth_controller is None:
            raise RuntimeError("structural growth controller is not attached")
        proposal = self._topology_proposals.get(str(proposal_id))
        if (
            proposal is None
            or proposal.status != "pending"
            or proposal.target_kind != "region"
            or proposal.operation != "split"
            or dict(proposal.specification).get("topology_role") != "region_split"
        ):
            raise ValueError("proposal is not a pending region split")
        if len(holdout_inputs) == 0 or len(holdout_inputs) != len(expected_activities):
            raise ValueError(
                "region split holdout inputs and expected activities must have equal size"
            )
        try:
            network = self._neuron_networks[str(network_id)]
        except KeyError as exc:
            raise ValueError(f"unknown adaptive neuron network: {network_id}") from exc
        if self._topology_network_ids.get(str(proposal_id)) != str(network_id):
            raise ValueError("region split proposal belongs to another network")
        specification = dict(proposal.specification)
        parent_id = str(specification.get("parent_region_id", ""))
        retained_id = str(specification.get("retained_region_id", ""))
        new_region_id = str(specification.get("new_region_id", ""))
        retained_units = tuple(str(item) for item in specification.get("retained_unit_ids", ()))
        new_units = tuple(str(item) for item in specification.get("new_unit_ids", ()))
        payload = network.to_payload()
        intact = AdaptiveNeuronNetwork.from_payload(
            payload,
            generator=self._checkpoint_region_generator(),
            device=self.device,
        )
        split = AdaptiveNeuronNetwork.from_payload(
            payload,
            generator=self._checkpoint_region_generator(),
            device=self.device,
        )
        split.apply_region_split(proposal, generator=self._checkpoint_region_generator())
        baseline_errors: list[float] = []
        split_errors: list[float] = []
        for inputs, expected in zip(holdout_inputs, expected_activities, strict=True):
            if not expected:
                raise ValueError("region split holdout must contain expected activities")
            if parent_id not in inputs:
                raise ValueError("region split holdout inputs must contain the parent region")
            split_inputs = dict(inputs)
            parent_input = split_inputs.pop(parent_id)
            split_inputs[retained_id] = parent_input
            split_inputs[new_region_id] = parent_input
            intact_activities = intact.step(inputs)
            split_activities = split.step(split_inputs)
            for expected_region_id, expected_activity in expected.items():
                expected_value = expected_activity.to(self.device)
                if expected_region_id == parent_id:
                    intact_activity = intact_activities[parent_id]
                    retained_activity = split_activities[retained_id]
                    new_activity = split_activities[new_region_id]
                    split_activity = torch.cat(
                        (
                            retained_activity[: len(retained_units)],
                            new_activity[: len(new_units)],
                        )
                    )
                else:
                    if expected_region_id not in split_activities:
                        raise ValueError("region split expected activity targets a missing region")
                    intact_activity = intact_activities[expected_region_id]
                    split_activity = split_activities[expected_region_id]
                if expected_value.shape != intact_activity.shape:
                    raise ValueError(
                        f"region split expected activity shape mismatch for {expected_region_id}"
                    )
                baseline_errors.append(
                    float(
                        torch.mean(torch.abs(intact_activity - expected_value))
                        .clamp(0.0, 1.0)
                        .item()
                    )
                )
                split_errors.append(
                    float(
                        torch.mean(torch.abs(split_activity - expected_value))
                        .clamp(0.0, 1.0)
                        .item()
                    )
                )
        baseline_error = sum(baseline_errors) / len(baseline_errors)
        split_error = sum(split_errors) / len(split_errors)
        regression = max(0.0, split_error - baseline_error)
        score = max(0.0, min(1.0, 1.0 - regression))
        maximum_regression = (
            self._structural_growth_controller.dynamics.maximum_restructure_holdout_regression
        )
        validated = regression <= float(maximum_regression)
        self._topology_proposals[str(proposal_id)] = replace(
            proposal,
            validation_score=score,
        )
        previous = self._cognitive_state.development
        self._cognitive_state = replace(
            self._cognitive_state,
            development=replace(
                previous,
                tick=self.tick,
                last_update_source="region-split-holdout-validation",
                last_validation_status="validated" if validated else "rejected",
                validation_evidence_ids=self._bounded_ids(
                    previous.validation_evidence_ids,
                    f"holdout:{proposal_id}",
                    limit=self._lineage_limit(),
                ),
                lineage=self._bounded_ids(
                    previous.lineage,
                    f"holdout:{proposal_id}",
                    limit=self._lineage_limit(),
                ),
            ),
        )
        return validated

    def commit_region_split(
        self,
        network_id: str,
        proposal: StructuralTopologyProposal,
    ) -> bool:
        """Commit a validated region split through the runtime ledger."""

        existing = self._topology_proposals.get(proposal.proposal_id)
        if existing is not None:
            if existing.status == "accepted":
                return True
            if existing.status in {"rejected", "rolled_back"}:
                return False
            proposal = existing
        if proposal.status != "pending":
            raise ValueError("only pending region split proposals can be committed")
        if (
            proposal.target_kind != "region"
            or proposal.operation != "split"
            or dict(proposal.specification).get("topology_role") != "region_split"
        ):
            raise ValueError("proposal is not a region split")
        if self._structural_growth_controller is None:
            raise RuntimeError("structural growth controller is not attached")
        if self._topology_network_ids.get(proposal.proposal_id) not in {
            None,
            str(network_id),
        }:
            raise ValueError("region split proposal belongs to another network")
        minimum_score = 1.0 - float(
            self._structural_growth_controller.dynamics.maximum_restructure_holdout_regression
        )
        if proposal.validation_score < minimum_score:
            rejected = replace(proposal, status="rejected")
            self._topology_proposals[proposal.proposal_id] = rejected
            self._record_topology_development(
                rejected,
                consume_budget=False,
                evidence_id=(
                    proposal.evidence_ids[0] if proposal.evidence_ids else proposal.proposal_id
                ),
                source="region-topology-split",
            )
            return False
        try:
            network = self._neuron_networks[str(network_id)]
        except KeyError as exc:
            raise ValueError(f"unknown adaptive neuron network: {network_id}") from exc
        if self._cognitive_state.development.structural_budget < proposal.resource_cost:
            rejected = replace(proposal, status="rejected")
            self._topology_proposals[proposal.proposal_id] = rejected
            self._record_topology_development(
                rejected,
                consume_budget=False,
                evidence_id=(
                    proposal.evidence_ids[0] if proposal.evidence_ids else proposal.proposal_id
                ),
                source="region-topology-split",
            )
            return False
        parent_snapshot = network.to_payload()
        try:
            network.apply_region_split(
                proposal,
                generator=self._checkpoint_region_generator(),
            )
            accepted_snapshot = network.to_payload()
            trial = AdaptiveNeuronNetwork.from_payload(
                accepted_snapshot,
                generator=self._checkpoint_region_generator(),
                device=self.device,
            )
            if (
                trial.region_ids != network.region_ids
                or trial.execution_order != network.execution_order
            ):
                raise ValueError("region split checkpoint roundtrip changed topology identities")
        except (IndexError, KeyError, RuntimeError, ValueError):
            self._neuron_networks[str(network_id)] = AdaptiveNeuronNetwork.from_payload(
                parent_snapshot,
                generator=self._checkpoint_region_generator(),
                device=self.device,
            )
            rejected = replace(proposal, status="rejected")
            self._topology_proposals[proposal.proposal_id] = rejected
            self._record_topology_development(
                rejected,
                consume_budget=False,
                evidence_id=(
                    proposal.evidence_ids[0] if proposal.evidence_ids else proposal.proposal_id
                ),
                source="region-topology-split",
            )
            return False
        accepted = replace(proposal, status="accepted")
        self._topology_proposals[proposal.proposal_id] = accepted
        self._topology_parent_snapshots[proposal.proposal_id] = parent_snapshot
        self._topology_network_ids[proposal.proposal_id] = str(network_id)
        self._record_topology_development(
            accepted,
            consume_budget=True,
            evidence_id=(
                proposal.evidence_ids[0] if proposal.evidence_ids else proposal.proposal_id
            ),
            source="region-topology-split",
            counter="split_merge_count",
        )
        return True

    def rollback_region_split(self, proposal_id: str) -> bool:
        """Rollback only the latest accepted region split."""

        key = str(proposal_id)
        proposal = self._topology_proposals.get(key)
        snapshot = self._topology_parent_snapshots.get(key)
        network_id = self._topology_network_ids.get(key)
        if (
            proposal is None
            or proposal.status != "accepted"
            or proposal.target_kind != "region"
            or proposal.operation != "split"
            or dict(proposal.specification).get("topology_role") != "region_split"
            or snapshot is None
            or network_id is None
        ):
            return False
        active = [
            item.proposal_id
            for item in self._topology_proposals.values()
            if item.status == "accepted" and item.proposal_id in self._topology_parent_snapshots
        ]
        if not active or active[-1] != key:
            return False
        self._neuron_networks[network_id] = AdaptiveNeuronNetwork.from_payload(
            snapshot,
            generator=self._checkpoint_region_generator(),
            device=self.device,
        )
        rolled_back = replace(proposal, status="rolled_back", validation_score=0.0)
        self._topology_proposals[key] = rolled_back
        self._topology_parent_snapshots.pop(key, None)
        self._topology_network_ids.pop(key, None)
        previous = self._cognitive_state.development
        self._cognitive_state = replace(
            self._cognitive_state,
            development=replace(
                previous,
                tick=self.tick,
                structural_budget=previous.structural_budget + proposal.resource_cost,
                last_update_source="region-topology-split-rollback",
                last_validation_status="rolled_back",
                validation_evidence_ids=self._bounded_ids(
                    previous.validation_evidence_ids,
                    key,
                    limit=self._lineage_limit(),
                ),
                split_merge_count=max(0, previous.split_merge_count - proposal.requested_units),
                lineage=self._bounded_ids(
                    previous.lineage,
                    key,
                    limit=self._lineage_limit(),
                ),
            ),
        )
        return True

    def commit_neuron_add(self, proposal: StructuralTopologyProposal) -> bool:
        """Commit a neuron birth transactionally through budget and checkpoint gates."""

        existing = self._topology_proposals.get(proposal.proposal_id)
        if existing is not None:
            if existing.status == "accepted":
                return True
            if existing.status in {"rejected", "rolled_back"}:
                return False
        if proposal.status != "pending":
            raise ValueError("only pending neuron proposals can be committed")
        if proposal.target_kind != "neuron" or proposal.operation != "add":
            raise ValueError("proposal is not a neuron add")
        if proposal.requested_units != 1:
            raise ValueError("neuron add currently supports exactly one unit")
        try:
            region = self._neuron_regions[proposal.substrate_id]
        except KeyError as exc:
            raise ValueError(f"unknown adaptive neuron region: {proposal.substrate_id}") from exc
        if self._cognitive_state.development.structural_budget < proposal.resource_cost:
            rejected = replace(proposal, status="rejected")
            self._topology_proposals[proposal.proposal_id] = rejected
            self._record_topology_development(
                rejected,
                consume_budget=False,
                evidence_id=(
                    proposal.evidence_ids[0] if proposal.evidence_ids else proposal.proposal_id
                ),
                source="neuron-topology-growth",
            )
            return False

        parent_snapshot = region.to_payload()
        try:
            region.apply_topology_proposal(proposal, generator=self._rng)
            accepted_snapshot = region.to_payload()
            trial = AdaptiveNeuronRegion.from_payload(
                accepted_snapshot,
                generator=self._checkpoint_region_generator(),
                device=self.device,
            )
            if trial.unit_ids != region.unit_ids:
                raise ValueError("neuron checkpoint roundtrip changed unit identities")
            if not torch.equal(trial.incoming.pre_index, region.incoming.pre_index):
                raise ValueError("neuron checkpoint roundtrip changed input support")
            if (
                trial.recurrent is not None
                and region.recurrent is not None
                and not torch.equal(trial.recurrent.pre_index, region.recurrent.pre_index)
            ):
                raise ValueError("neuron checkpoint roundtrip changed recurrent support")
        except (IndexError, KeyError, RuntimeError, ValueError):
            self._neuron_regions[proposal.substrate_id] = AdaptiveNeuronRegion.from_payload(
                parent_snapshot,
                generator=self._checkpoint_region_generator(),
                device=self.device,
            )
            rejected = replace(proposal, status="rejected")
            self._topology_proposals[proposal.proposal_id] = rejected
            self._record_topology_development(
                rejected,
                consume_budget=False,
                evidence_id=(
                    proposal.evidence_ids[0] if proposal.evidence_ids else proposal.proposal_id
                ),
                source="neuron-topology-growth",
            )
            return False

        accepted = replace(proposal, status="accepted", validation_score=1.0)
        self._topology_proposals[proposal.proposal_id] = accepted
        self._topology_parent_snapshots[proposal.proposal_id] = parent_snapshot
        self._record_topology_development(
            accepted,
            consume_budget=True,
            evidence_id=(
                proposal.evidence_ids[0] if proposal.evidence_ids else proposal.proposal_id
            ),
            source="neuron-topology-growth",
        )
        return True

    def rollback_neuron_add(self, proposal_id: str) -> bool:
        """Rollback only the latest accepted neuron birth."""

        key = str(proposal_id)
        proposal = self._topology_proposals.get(key)
        snapshot = self._topology_parent_snapshots.get(key)
        if (
            proposal is None
            or proposal.status != "accepted"
            or proposal.target_kind != "neuron"
            or snapshot is None
        ):
            return False
        active = [
            item.proposal_id
            for item in self._topology_proposals.values()
            if item.status == "accepted" and item.proposal_id in self._topology_parent_snapshots
        ]
        if not active or active[-1] != key:
            return False
        self._neuron_regions[proposal.substrate_id] = AdaptiveNeuronRegion.from_payload(
            snapshot,
            generator=self._checkpoint_region_generator(),
            device=self.device,
        )
        rolled_back = replace(proposal, status="rolled_back", validation_score=0.0)
        self._topology_proposals[key] = rolled_back
        self._topology_parent_snapshots.pop(key, None)
        previous = self._cognitive_state.development
        self._cognitive_state = replace(
            self._cognitive_state,
            development=replace(
                previous,
                tick=self.tick,
                structural_budget=previous.structural_budget + proposal.resource_cost,
                last_update_source="neuron-topology-rollback",
                last_validation_status="rolled_back",
                validation_evidence_ids=self._bounded_ids(
                    previous.validation_evidence_ids,
                    key,
                    limit=self._lineage_limit(),
                ),
                growth_count=max(0, previous.growth_count - proposal.requested_units),
                lineage=self._bounded_ids(
                    previous.lineage,
                    key,
                    limit=self._lineage_limit(),
                ),
            ),
        )
        return True

    def attach_adaptive_neuron_network(
        self,
        network_id: str,
        network: AdaptiveNeuronNetwork | None,
    ) -> None:
        """Attach an explicit cross-region network to the Taiji runtime."""

        key = str(network_id)
        if not key:
            raise ValueError("network_id must not be empty")
        if network is not None and not isinstance(network, AdaptiveNeuronNetwork):
            raise TypeError("network must be an AdaptiveNeuronNetwork or None")
        if network is None:
            self._neuron_networks.pop(key, None)
            return
        if key in self._neuron_networks:
            raise ValueError(f"adaptive neuron network already attached: {key}")
        if any(region.region_id in self._neuron_regions for region in network.regions):
            raise ValueError("network regions cannot also be attached as standalone regions")
        self._neuron_networks[key] = network

    def attach_cross_region_cooperation(
        self,
        network_id: str,
        learner: CrossRegionCooperationLearner | None,
    ) -> None:
        """Attach or remove the learner that selects this network's routes."""

        try:
            network = self._neuron_networks[str(network_id)]
        except KeyError as exc:
            raise ValueError(f"unknown adaptive neuron network: {network_id}") from exc
        network.attach_cooperation_learner(learner)

    def select_cross_region_connections(
        self,
        network_id: str,
        *,
        resource_budget: float = 1.0,
        max_connections: int = 1,
    ) -> tuple[str, ...]:
        """Expose learned cross-region competition through the runtime owner."""

        try:
            network = self._neuron_networks[str(network_id)]
        except KeyError as exc:
            raise ValueError(f"unknown adaptive neuron network: {network_id}") from exc
        return network.selected_connection_ids(
            resource_budget=resource_budget,
            max_connections=max_connections,
        )

    def observe_cross_region_connection(
        self,
        network_id: str,
        connection_id: str,
        *,
        prediction_error: float,
        holdout_transfer: float,
        resource_state: float,
        selected: bool = True,
    ) -> float:
        """Route outcome evidence to the attached cross-region learner."""

        try:
            network = self._neuron_networks[str(network_id)]
        except KeyError as exc:
            raise ValueError(f"unknown adaptive neuron network: {network_id}") from exc
        return network.observe_connection(
            connection_id,
            prediction_error=prediction_error,
            holdout_transfer=holdout_transfer,
            resource_state=resource_state,
            selected=selected,
        )

    def step_cross_region_network(
        self,
        network_id: str,
        external_inputs: Mapping[str, torch.Tensor],
        *,
        expected_activities: Mapping[str, torch.Tensor] | None = None,
        resource_budget: float = 1.0,
        max_connections: int = 1,
        holdout: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Run a native network tick and optionally credit its real outcome."""

        try:
            network = self._neuron_networks[str(network_id)]
        except KeyError as exc:
            raise ValueError(f"unknown adaptive neuron network: {network_id}") from exc
        activities = network.step(
            external_inputs,
            expected_activities=expected_activities,
            resource_budget=resource_budget,
            max_connections=max_connections,
            holdout=holdout,
        )
        self._observe_cross_region_structure(
            str(network_id),
            activities,
            expected_activities,
            holdout=holdout,
        )
        return activities

    def propose_cross_region_connection(
        self,
        *,
        network_id: str,
        source_region_id: str,
        target_region_id: str,
        evidence_ids: Sequence[str],
        fan_in: int,
        parent_checkpoint_id: str | None = None,
        resource_cost: int = 1,
    ) -> StructuralTopologyProposal:
        """Create a cross-region connection proposal for the runtime ledger."""

        try:
            network = self._neuron_networks[str(network_id)]
        except KeyError as exc:
            raise ValueError(f"unknown adaptive neuron network: {network_id}") from exc
        proposal = network.propose_connection_add(
            source_region_id=source_region_id,
            target_region_id=target_region_id,
            evidence_ids=evidence_ids,
            fan_in=fan_in,
            parent_checkpoint_id=parent_checkpoint_id,
            resource_cost=resource_cost,
        )
        if proposal.parent_checkpoint_id is None:
            proposal = replace(
                proposal,
                parent_checkpoint_id=f"cross-region-parent:{proposal.proposal_id}",
            )
        return proposal

    def commit_cross_region_connection(
        self,
        network_id: str,
        proposal: StructuralTopologyProposal,
    ) -> bool:
        """Commit a cross-region connection only after ledger validation."""

        existing = self._topology_proposals.get(proposal.proposal_id)
        if existing is not None:
            if existing.status == "accepted":
                return True
            if existing.status in {"rejected", "rolled_back"}:
                return False
        if proposal.status != "pending":
            raise ValueError("only pending cross-region proposals can be committed")
        if (
            proposal.target_kind != "region"
            or proposal.operation != "add"
            or dict(proposal.specification).get("topology_role") != "cross_region_connection"
        ):
            raise ValueError("proposal is not a cross-region connection add")
        specification = dict(proposal.specification)
        for endpoint_key in ("source_region_id", "target_region_id"):
            endpoint_id = str(specification.get(endpoint_key, ""))
            pending_region = next(
                (
                    item
                    for item in self._topology_proposals.values()
                    if item.status == "accepted"
                    and item.substrate_id == endpoint_id
                    and dict(item.specification).get("topology_role") == "region"
                ),
                None,
            )
            if (
                pending_region is not None
                and self._structural_growth_controller is not None
                and pending_region.validation_score
                < self._structural_growth_controller.dynamics.minimum_holdout_gain
            ):
                raise ValueError(f"region {endpoint_id} has not passed holdout validation")
        try:
            network = self._neuron_networks[str(network_id)]
        except KeyError as exc:
            raise ValueError(f"unknown adaptive neuron network: {network_id}") from exc
        if self._cognitive_state.development.structural_budget < proposal.resource_cost:
            rejected = replace(proposal, status="rejected")
            self._topology_proposals[proposal.proposal_id] = rejected
            self._record_topology_development(
                rejected,
                consume_budget=False,
                evidence_id=(
                    proposal.evidence_ids[0] if proposal.evidence_ids else proposal.proposal_id
                ),
                source="cross-region-topology-growth",
            )
            return False

        parent_snapshot = network.to_payload()
        try:
            network.apply_topology_proposal(proposal, generator=self._rng)
            accepted_snapshot = network.to_payload()
            trial = AdaptiveNeuronNetwork.from_payload(
                accepted_snapshot,
                generator=self._checkpoint_region_generator(),
                device=self.device,
            )
            if (
                trial.region_ids != network.region_ids
                or trial.connection_ids != network.connection_ids
            ):
                raise ValueError("cross-region checkpoint roundtrip changed identities")
            for current, restored in zip(network.connections, trial.connections, strict=True):
                if not torch.equal(current[3].pre_index, restored[3].pre_index):
                    raise ValueError("cross-region checkpoint roundtrip changed support")
        except (IndexError, KeyError, RuntimeError, ValueError):
            self._neuron_networks[str(network_id)] = AdaptiveNeuronNetwork.from_payload(
                parent_snapshot,
                generator=self._checkpoint_region_generator(),
                device=self.device,
            )
            rejected = replace(proposal, status="rejected")
            self._topology_proposals[proposal.proposal_id] = rejected
            self._record_topology_development(
                rejected,
                consume_budget=False,
                evidence_id=(
                    proposal.evidence_ids[0] if proposal.evidence_ids else proposal.proposal_id
                ),
                source="cross-region-topology-growth",
            )
            return False

        accepted = replace(proposal, status="accepted", validation_score=1.0)
        self._topology_proposals[proposal.proposal_id] = accepted
        self._topology_parent_snapshots[proposal.proposal_id] = parent_snapshot
        self._record_topology_development(
            accepted,
            consume_budget=True,
            evidence_id=(
                proposal.evidence_ids[0] if proposal.evidence_ids else proposal.proposal_id
            ),
            source="cross-region-topology-growth",
        )
        return True

    def rollback_cross_region_connection(self, proposal_id: str) -> bool:
        """Rollback only the latest accepted cross-region connection."""

        key = str(proposal_id)
        proposal = self._topology_proposals.get(key)
        snapshot = self._topology_parent_snapshots.get(key)
        if (
            proposal is None
            or proposal.status != "accepted"
            or proposal.target_kind != "region"
            or snapshot is None
        ):
            return False
        active = [
            item.proposal_id
            for item in self._topology_proposals.values()
            if item.status == "accepted" and item.proposal_id in self._topology_parent_snapshots
        ]
        if not active or active[-1] != key:
            return False
        network_id = next(
            (
                network_key
                for network_key, network in self._neuron_networks.items()
                if proposal.substrate_id in network.connection_ids
            ),
            None,
        )
        if network_id is None:
            return False
        self._neuron_networks[network_id] = AdaptiveNeuronNetwork.from_payload(
            snapshot,
            generator=self._checkpoint_region_generator(),
            device=self.device,
        )
        rolled_back = replace(proposal, status="rolled_back", validation_score=0.0)
        self._topology_proposals[key] = rolled_back
        self._topology_parent_snapshots.pop(key, None)
        previous = self._cognitive_state.development
        self._cognitive_state = replace(
            self._cognitive_state,
            development=replace(
                previous,
                tick=self.tick,
                structural_budget=previous.structural_budget + proposal.resource_cost,
                last_update_source="cross-region-topology-rollback",
                last_validation_status="rolled_back",
                validation_evidence_ids=self._bounded_ids(
                    previous.validation_evidence_ids,
                    key,
                    limit=self._lineage_limit(),
                ),
                growth_count=max(0, previous.growth_count - proposal.requested_units),
                lineage=self._bounded_ids(
                    previous.lineage,
                    key,
                    limit=self._lineage_limit(),
                ),
            ),
        )
        return True

    def _record_online_concept_transition(
        self,
        transition: WorldTransition,
        prediction_error: float,
        *,
        boundary: bool,
    ) -> tuple[str, ...]:
        """Buffer real transitions and birth novel branches at episode boundaries."""

        active_ids = set(self._cognitive_state.memory.concept_ids)
        if not active_ids:
            if boundary:
                self._online_concept_branches.clear()
            return ()
        matches = tuple(
            match
            for match in self._concept_matches_for_world(transition.before)
            if match.concept.concept_id in active_ids
        )
        owner_id = self._concept_formation.select_sequence_owner(
            matches,
            transition,
            prediction_error,
            weights=self.config.concept_branch_owner_weights,
            min_score=self.config.concept_branch_owner_min_score,
            min_margin=self.config.concept_branch_owner_min_margin,
        )
        if owner_id is None:
            self._online_concept_branches.clear()
            return ()
        if len(self._online_concept_branches) > 1:
            self._online_concept_branches.clear()
            return ()
        if self._online_concept_branches and owner_id not in self._online_concept_branches:
            self._online_concept_branches.clear()
            return ()
        born: list[str] = []
        history = self._online_concept_branches.get(owner_id, ())
        if history and history[-1][0].after.tick != transition.before.tick:
            history = ()
        history = (*history, (transition, float(prediction_error)))
        if boundary:
            trace_id = self.grow_online_concept_branch(owner_id, history)
            if trace_id is not None:
                born.append(trace_id)
            self._online_concept_branches.pop(owner_id, None)
        else:
            self._online_concept_branches[owner_id] = history
        return tuple(born)

    def _online_concept_branches_checkpoint(self) -> dict[str, list[dict[str, Any]]]:
        return {
            concept_id: [
                {
                    "transition": transition.to_payload(),
                    "prediction_error": prediction_error,
                }
                for transition, prediction_error in history
            ]
            for concept_id, history in self._online_concept_branches.items()
        }

    def _restore_online_concept_branches(self, payload: Any) -> None:
        self._online_concept_branches = {}
        if not isinstance(payload, dict):
            return
        for concept_id, entries in payload.items():
            if not isinstance(entries, (list, tuple)):
                raise ValueError("online concept branch checkpoint entries must be a sequence")
            history: list[tuple[WorldTransition, float]] = []
            for entry in entries:
                if not isinstance(entry, dict):
                    raise ValueError("online concept branch checkpoint entry must be a mapping")
                history.append(
                    (
                        WorldTransition.from_payload(entry["transition"], device=self.device),
                        float(entry.get("prediction_error", 0.0)),
                    )
                )
            if history:
                self._online_concept_branches[str(concept_id)] = tuple(history)

    def _growth_requests_checkpoint(self) -> dict[str, Any]:
        return {
            "requests": [request.to_payload() for request in self._growth_requests.values()],
            "snapshots": dict(self._growth_request_snapshots),
        }

    def _restore_growth_requests(self, payload: Any) -> None:
        self._growth_requests = {}
        self._growth_request_snapshots = {}
        if not isinstance(payload, dict):
            return
        requests = payload.get("requests", ())
        if not isinstance(requests, (list, tuple)):
            raise ValueError("growth request checkpoint requests must be a sequence")
        for item in requests:
            if not isinstance(item, dict):
                raise ValueError("growth request checkpoint entry must be a mapping")
            request = StructuralGrowthRequest.from_payload(item)
            self._growth_requests[request.request_id] = request
        snapshots = payload.get("snapshots", {})
        if not isinstance(snapshots, dict):
            raise ValueError("growth request checkpoint snapshots must be a mapping")
        self._growth_request_snapshots = {
            str(request_id): dict(snapshot)
            for request_id, snapshot in snapshots.items()
            if isinstance(snapshot, dict)
        }

    def _topology_proposals_checkpoint(self) -> dict[str, Any]:
        return {
            "proposals": [proposal.to_payload() for proposal in self._topology_proposals.values()],
            "snapshots": dict(self._topology_parent_snapshots),
            "network_ids": dict(self._topology_network_ids),
        }

    def _restore_topology_proposals(self, payload: Any) -> None:
        self._topology_proposals = {}
        self._topology_parent_snapshots = {}
        self._topology_network_ids = {}
        if not isinstance(payload, dict):
            return
        proposals = payload.get("proposals", ())
        if not isinstance(proposals, (list, tuple)):
            raise ValueError("topology proposal checkpoint proposals must be a sequence")
        for item in proposals:
            if not isinstance(item, dict):
                raise ValueError("topology proposal checkpoint entry must be a mapping")
            proposal = StructuralTopologyProposal.from_payload(item)
            self._topology_proposals[proposal.proposal_id] = proposal
        snapshots = payload.get("snapshots", {})
        if not isinstance(snapshots, dict):
            raise ValueError("topology proposal checkpoint snapshots must be a mapping")
        self._topology_parent_snapshots = {
            str(proposal_id): dict(snapshot)
            for proposal_id, snapshot in snapshots.items()
            if isinstance(snapshot, dict)
        }
        network_ids = payload.get("network_ids", {})
        if not isinstance(network_ids, dict):
            raise ValueError("topology proposal checkpoint network_ids must be a mapping")
        self._topology_network_ids = {
            str(proposal_id): str(network_id) for proposal_id, network_id in network_ids.items()
        }

    def _neuron_regions_checkpoint(self) -> dict[str, Any]:
        return {
            "regions": {
                region_id: region.to_payload() for region_id, region in self._neuron_regions.items()
            }
        }

    def _restore_neuron_regions(self, payload: Any) -> None:
        self._neuron_regions = {}
        if not isinstance(payload, dict):
            return
        regions = payload.get("regions", {})
        if not isinstance(regions, dict):
            raise ValueError("adaptive neuron checkpoint regions must be a mapping")
        for region_id, region_payload in regions.items():
            if not isinstance(region_payload, dict):
                raise ValueError("adaptive neuron checkpoint entry must be a mapping")
            region = AdaptiveNeuronRegion.from_payload(
                region_payload,
                generator=self._checkpoint_region_generator(),
                device=self.device,
            )
            if region.region_id != str(region_id):
                raise ValueError("adaptive neuron checkpoint region identity does not match")
            self._neuron_regions[region.region_id] = region

    def _neuron_networks_checkpoint(self) -> dict[str, Any]:
        return {
            "networks": {
                network_id: network.to_payload()
                for network_id, network in self._neuron_networks.items()
            }
        }

    def _restore_neuron_networks(self, payload: Any) -> None:
        self._neuron_networks = {}
        if not isinstance(payload, dict):
            return
        networks = payload.get("networks", {})
        if not isinstance(networks, dict):
            raise ValueError("adaptive neuron checkpoint networks must be a mapping")
        for network_id, network_payload in networks.items():
            if not isinstance(network_payload, dict):
                raise ValueError("adaptive neuron network checkpoint entry must be a mapping")
            self._neuron_networks[str(network_id)] = AdaptiveNeuronNetwork.from_payload(
                network_payload,
                generator=self._checkpoint_region_generator(),
                device=self.device,
            )

    def _structural_growth_checkpoint(self) -> dict[str, Any] | None:
        return (
            None
            if self._structural_growth_controller is None
            else self._structural_growth_controller.to_payload()
        )

    def _restore_structural_growth(self, payload: Any) -> None:
        self._structural_growth_controller = (
            None if payload is None else AdaptiveStructuralGrowthController.from_payload(payload)
        )

    def _structural_pruning_checkpoint(self) -> dict[str, Any] | None:
        return (
            None
            if self._structural_pruning_controller is None
            else self._structural_pruning_controller.to_payload()
        )

    def _restore_structural_pruning(self, payload: Any) -> None:
        self._structural_pruning_controller = (
            None if payload is None else AdaptiveStructuralPruningController.from_payload(payload)
        )

    def _structural_runtime_resource_state(self) -> float:
        """Map the live structural budget to a bounded resource state."""

        budget = max(0, int(self._cognitive_state.development.structural_budget))
        configured = max(1, int(self.config.development_structural_budget))
        return min(1.0, float(budget) / float(configured))

    def _queue_structural_proposal_candidate(
        self,
        candidate: StructuralProposalCandidate,
    ) -> None:
        """Keep one pending candidate per transformation identity.

        Region-level rewrites remain substrate-deduplicated.  Neuron births
        are narrower: two different unit identities may be queued so a
        dependency chain can grow a population over multiple ledger commits.
        """

        for existing in self._structural_proposal_candidates.values():
            if (
                existing.network_id == candidate.network_id
                and existing.operation == candidate.operation
                and existing.substrate_ids == candidate.substrate_ids
            ):
                if (
                    candidate.target_kind == "neuron"
                    and candidate.operation == "add"
                    and existing.target_kind == "neuron"
                ):
                    existing_unit_id = dict(existing.specification).get("unit_id")
                    candidate_unit_id = dict(candidate.specification).get("unit_id")
                    if existing_unit_id != candidate_unit_id:
                        continue
                return
        self._structural_proposal_candidates[candidate.candidate_id] = candidate
        limit = self._lineage_limit()
        while len(self._structural_proposal_candidates) > limit:
            self._structural_proposal_candidates.pop(
                next(iter(self._structural_proposal_candidates))
            )

    def _candidate_specification(
        self,
        **values: Any,
    ) -> tuple[tuple[str, Any], ...]:
        return tuple((str(key), value) for key, value in values.items())

    def _observe_cross_region_structure(
        self,
        network_id: str,
        activities: Mapping[str, torch.Tensor],
        expected_activities: Mapping[str, torch.Tensor] | None,
        *,
        holdout: bool,
    ) -> tuple[StructuralRuntimeObservation, ...]:
        """Feed real network activity into the structural maintenance organs."""

        network = self._neuron_networks[str(network_id)]
        self._structural_runtime_tick += 1
        runtime_tick = self._structural_runtime_tick
        resource_state = self._structural_runtime_resource_state()
        resource_pressure = 1.0 - resource_state
        expected = {} if expected_activities is None else dict(expected_activities)
        observations: list[StructuralRuntimeObservation] = []
        region_by_id = {region.region_id: region for region in network.regions}
        region_observation_by_id: dict[str, StructuralRuntimeObservation] = {}
        for region_id in network.execution_order:
            activity = activities[region_id]
            usage = float(torch.mean(torch.abs(activity)).clamp(0.0, 1.0).item())
            expected_activity = expected.get(region_id)
            prediction_error: float | None = None
            learning_gain = 0.0
            holdout_transfer = 0.0
            error_key = f"network:{network_id}:region:{region_id}"
            if expected_activity is not None:
                expected_value = expected_activity.to(activity.device)
                if expected_value.shape != activity.shape:
                    raise ValueError(
                        f"structural runtime activity shape mismatch for region {region_id}"
                    )
                prediction_error = float(
                    torch.mean(torch.abs(activity - expected_value)).clamp(0.0, 1.0).item()
                )
                previous_error = self._structural_runtime_previous_errors.get(error_key)
                if previous_error is not None:
                    learning_gain = max(0.0, min(1.0, previous_error - prediction_error))
                self._structural_runtime_previous_errors[error_key] = prediction_error
                if holdout:
                    holdout_transfer = 1.0 - prediction_error
            evidence_id = f"runtime-structure:{network_id}:{region_id}:{runtime_tick}"
            substrate_id = f"network:{network_id}:region:{region_id}"
            growth_decision: StructuralGrowthDecision | None = None
            if self._structural_growth_controller is not None and prediction_error is not None:
                growth_decision = self._structural_growth_controller.observe(
                    substrate_id,
                    prediction_error=prediction_error,
                    resource_state=resource_state,
                    holdout_transfer=holdout_transfer,
                    evidence_ids=(evidence_id,),
                )
                region = region_by_id[region_id]
                if growth_decision.should_grow and region.unit_count >= 2:
                    self._queue_structural_proposal_candidate(
                        StructuralProposalCandidate(
                            candidate_id=(
                                f"candidate:{network_id}:split:{region_id}:"
                                f"{growth_decision.proposal_ordinal}"
                            ),
                            network_id=str(network_id),
                            target_kind="region",
                            operation="split",
                            substrate_ids=(region_id,),
                            evidence_ids=growth_decision.evidence_ids,
                            source_tick=runtime_tick,
                            priority=min(
                                1.0,
                                growth_decision.error_ema
                                * max(growth_decision.holdout_transfer_ema, 0.0),
                            ),
                            specification=self._candidate_specification(
                                region_id=region_id,
                                first_unit_count=max(1, region.unit_count // 2),
                            ),
                            resource_cost=(
                                self._structural_growth_controller.dynamics.growth_resource_cost
                            ),
                        )
                    )
            pruning_decision: StructuralPruningDecision | None = None
            pruning_controller = self._structural_pruning_controller
            if pruning_controller is not None:
                pruning_decision = pruning_controller.observe_substrate(
                    substrate_id,
                    usage=usage,
                    resource_pressure=resource_pressure,
                    learning_gain=learning_gain,
                    evidence_ids=(evidence_id,),
                )
            observations.append(
                StructuralRuntimeObservation(
                    network_id=str(network_id),
                    region_id=region_id,
                    tick=runtime_tick,
                    usage=usage,
                    resource_pressure=resource_pressure,
                    prediction_error=prediction_error,
                    learning_gain=learning_gain,
                    holdout_transfer=holdout_transfer,
                    evidence_id=evidence_id,
                )
            )
            region_observation_by_id[region_id] = observations[-1]
            if (
                pruning_decision is not None
                and pruning_decision.should_prune
                and pruning_controller is not None
            ):
                self._queue_structural_proposal_candidate(
                    StructuralProposalCandidate(
                        candidate_id=(
                            f"candidate:{network_id}:prune:region:{region_id}:"
                            f"{pruning_decision.proposal_ordinal}"
                        ),
                        network_id=str(network_id),
                        target_kind="region",
                        operation="prune",
                        substrate_ids=(region_id,),
                        evidence_ids=pruning_decision.evidence_ids,
                        source_tick=runtime_tick,
                        priority=min(
                            1.0,
                            pruning_decision.resource_pressure_ema
                            * (1.0 - pruning_decision.learning_gain_ema),
                        ),
                        specification=self._candidate_specification(region_id=region_id),
                        resource_cost=(pruning_controller.dynamics.pruning_resource_cost),
                    )
                )
        learner = network.cooperation_learner
        if learner is not None and self._structural_pruning_controller is not None:
            for connection_id in network.connection_ids:
                route = learner.route_state(connection_id)
                route_key = f"network:{network_id}:route:{connection_id}"
                previous_error = self._structural_runtime_previous_errors.get(route_key)
                route_gain = 0.0
                if previous_error is not None:
                    route_gain = max(0.0, min(1.0, previous_error - route.prediction_error))
                self._structural_runtime_previous_errors[route_key] = route.prediction_error
                route_usage = min(
                    1.0,
                    float(route.selection_count) / float(max(1, runtime_tick)),
                )
                route_evidence_id = (
                    f"runtime-structure:{network_id}:route:{connection_id}:{runtime_tick}"
                )
                route_decision = self._structural_pruning_controller.observe_substrate(
                    connection_id,
                    usage=route_usage,
                    resource_pressure=1.0 - route.resource_state,
                    learning_gain=route_gain,
                    evidence_ids=(route_evidence_id,),
                )
                if route_decision.should_prune:
                    self._queue_structural_proposal_candidate(
                        StructuralProposalCandidate(
                            candidate_id=(
                                f"candidate:{network_id}:prune:connection:{connection_id}:"
                                f"{route_decision.proposal_ordinal}"
                            ),
                            network_id=str(network_id),
                            target_kind="connection",
                            operation="prune",
                            substrate_ids=(connection_id,),
                            evidence_ids=route_decision.evidence_ids,
                            source_tick=runtime_tick,
                            priority=min(
                                1.0,
                                route_decision.resource_pressure_ema
                                * (1.0 - route_decision.learning_gain_ema),
                            ),
                            specification=self._candidate_specification(
                                connection_id=connection_id,
                            ),
                            resource_cost=(
                                self._structural_pruning_controller.dynamics.pruning_resource_cost
                            ),
                        )
                    )
        if self._structural_pruning_controller is not None:
            regions = network.regions
            for first_index, first in enumerate(regions):
                for second in regions[first_index + 1 :]:
                    if (
                        first.input_source_id != second.input_source_id
                        or first.input_dim != second.input_dim
                        or first.fan_in != second.fan_in
                        or first.dynamics.to_payload() != second.dynamics.to_payload()
                    ):
                        continue
                    first_observation = region_observation_by_id[first.region_id]
                    second_observation = region_observation_by_id[second.region_id]
                    if (
                        first_observation.prediction_error is None
                        or second_observation.prediction_error is None
                    ):
                        continue
                    first_activity = activities[first.region_id]
                    second_activity = activities[second.region_id]
                    if first_activity.shape == second_activity.shape:
                        redundancy_usage = float(
                            torch.mean(torch.abs(first_activity - second_activity))
                            .clamp(0.0, 1.0)
                            .item()
                        )
                    else:
                        redundancy_usage = min(
                            1.0,
                            abs(first_observation.usage - second_observation.usage),
                        )
                    merge_evidence_id = (
                        f"runtime-structure:{network_id}:merge:{first.region_id}+"
                        f"{second.region_id}:{runtime_tick}"
                    )
                    merge_decision = self._structural_pruning_controller.observe_substrate(
                        f"network:{network_id}:merge:{first.region_id}+{second.region_id}",
                        usage=redundancy_usage,
                        resource_pressure=resource_pressure,
                        learning_gain=min(
                            1.0,
                            (first_observation.learning_gain + second_observation.learning_gain)
                            / 2.0,
                        ),
                        evidence_ids=(merge_evidence_id,),
                    )
                    if not merge_decision.should_prune:
                        continue
                    try:
                        network.propose_region_merge(
                            region_ids=(first.region_id, second.region_id),
                            evidence_ids=merge_decision.evidence_ids,
                            resource_cost=(
                                self._structural_pruning_controller.dynamics.pruning_resource_cost
                            ),
                        )
                    except ValueError:
                        continue
                    self._queue_structural_proposal_candidate(
                        StructuralProposalCandidate(
                            candidate_id=(
                                f"candidate:{network_id}:merge:{first.region_id}+"
                                f"{second.region_id}:{merge_decision.proposal_ordinal}"
                            ),
                            network_id=str(network_id),
                            target_kind="region",
                            operation="merge",
                            substrate_ids=(first.region_id, second.region_id),
                            evidence_ids=merge_decision.evidence_ids,
                            source_tick=runtime_tick,
                            priority=min(1.0, 1.0 - redundancy_usage),
                            specification=self._candidate_specification(
                                region_ids=(first.region_id, second.region_id),
                            ),
                            resource_cost=(
                                self._structural_pruning_controller.dynamics.pruning_resource_cost
                            ),
                        )
                    )
        self._structural_runtime_observations.extend(observations)
        self._structural_runtime_observations = self._structural_runtime_observations[
            -self._lineage_limit() :
        ]
        if observations:
            previous = self._cognitive_state.development
            self._cognitive_state = replace(
                self._cognitive_state,
                development=replace(
                    previous,
                    stage="structural-observation",
                    resource_utilization=resource_pressure,
                    last_update_source="runtime-structural-observation",
                    last_validation_status="pending",
                    validation_evidence_ids=self._bounded_ids(
                        previous.validation_evidence_ids,
                        observations[-1].evidence_id,
                        limit=self._lineage_limit(),
                    ),
                ),
            )
        return tuple(observations)

    def _structural_runtime_checkpoint(self) -> dict[str, Any]:
        return {
            "runtime_tick": self._structural_runtime_tick,
            "observations": [
                observation.to_payload() for observation in self._structural_runtime_observations
            ],
            "previous_errors": dict(self._structural_runtime_previous_errors),
            "proposal_candidates": [
                candidate.to_payload()
                for candidate in self._structural_proposal_candidates.values()
            ],
            "candidate_proposals": dict(self._structural_candidate_proposals),
            "maintenance_results": [
                result.to_payload() for result in self._structural_maintenance_results
            ],
        }

    def _restore_structural_runtime(self, payload: Any) -> None:
        self._structural_runtime_tick = 0
        self._structural_runtime_observations = []
        self._structural_runtime_previous_errors = {}
        self._structural_proposal_candidates = {}
        self._structural_candidate_proposals = {}
        self._structural_maintenance_results = []
        if payload is None:
            return
        if not isinstance(payload, Mapping):
            raise ValueError("structural runtime checkpoint must be a mapping")
        runtime_tick = int(payload.get("runtime_tick", 0))
        if runtime_tick < 0:
            raise ValueError("structural runtime tick cannot be negative")
        observations_payload = payload.get("observations", ())
        previous_errors = payload.get("previous_errors", {})
        if not isinstance(observations_payload, (tuple, list)):
            raise ValueError("structural runtime observations must be a sequence")
        if not isinstance(previous_errors, Mapping):
            raise ValueError("structural runtime previous_errors must be a mapping")
        candidates_payload = payload.get("proposal_candidates", ())
        candidate_proposals = payload.get("candidate_proposals", {})
        maintenance_results = payload.get("maintenance_results", ())
        if not isinstance(candidates_payload, (tuple, list)):
            raise ValueError("structural proposal candidates must be a sequence")
        if not isinstance(candidate_proposals, Mapping):
            raise ValueError("structural candidate_proposals must be a mapping")
        if not isinstance(maintenance_results, (tuple, list)):
            raise ValueError("structural maintenance results must be a sequence")
        if any(not isinstance(item, Mapping) for item in observations_payload):
            raise ValueError("structural runtime observation entry must be a mapping")
        observations = tuple(
            StructuralRuntimeObservation.from_payload(item) for item in observations_payload
        )
        if any(item.tick > runtime_tick for item in observations):
            raise ValueError("structural runtime observation is ahead of runtime tick")
        self._structural_runtime_tick = runtime_tick
        self._structural_runtime_observations = list(observations[-self._lineage_limit() :])
        self._structural_runtime_previous_errors = {
            str(key): float(value) for key, value in previous_errors.items()
        }
        for item in candidates_payload:
            if not isinstance(item, Mapping):
                raise ValueError("structural proposal candidate entry must be a mapping")
            candidate = StructuralProposalCandidate.from_payload(item)
            if candidate.source_tick > runtime_tick:
                raise ValueError("structural proposal candidate is ahead of runtime tick")
            self._queue_structural_proposal_candidate(candidate)
        self._structural_candidate_proposals = {
            str(candidate_id): str(proposal_id)
            for candidate_id, proposal_id in candidate_proposals.items()
        }
        if any(
            proposal_id not in self._topology_proposals
            for proposal_id in self._structural_candidate_proposals.values()
        ):
            raise ValueError("structural candidate proposal mapping references an unknown proposal")
        for item in maintenance_results:
            if not isinstance(item, Mapping):
                raise ValueError("structural maintenance result entry must be a mapping")
            self._record_structural_maintenance_result(
                StructuralMaintenanceResult.from_payload(item)
            )

    def attach_world_dynamics(self, learner: WorldDynamicsLearner | None) -> None:
        """Attach a Taiji-owned predictor used for runtime intervention scoring."""

        if learner is not None and not isinstance(learner, WorldDynamicsLearner):
            raise TypeError("learner must be a WorldDynamicsLearner or None")
        self._world_dynamics = learner

    def begin_episode(self, episode_id: str) -> None:
        """Start a new episode while carrying persistent world cognition forward.

        Learned organs, world state, event lineage, concepts, and world
        calibration remain available.  Action and perceptual transient state
        is cleared, while the kernel tick stays monotonic so a carried world
        snapshot remains causally valid.
        """

        if not episode_id:
            raise ValueError("episode_id cannot be empty")
        if self._state.pending_action is not None:
            raise RuntimeError("pending action must be settled before beginning an episode")
        if self._state.pending_experience is not None:
            raise RuntimeError(
                "pending experience must observe its outcome before beginning an episode"
            )
        previous = self._cognitive_state
        if previous.world.tick != self.tick:
            raise RuntimeError("world state must be observed at the current kernel tick")
        self._archive_current_recovery_portfolio()
        self.perception.reset_dynamics()
        self._state = replace(self._state, episode_id=str(episode_id))
        empty = self._empty_cognitive_state(str(episode_id))
        self._cognitive_state = replace(
            previous,
            episode_id=str(episode_id),
            tick=self.tick,
            observation=None,
            percept=None,
            workspace=replace(empty.workspace, tick=self.tick),
            plan=replace(empty.plan, tick=self.tick),
            action_intent=None,
            outcome=None,
            world_transition=None,
            world_prediction=None,
            planning_recovery=None,
            recovery_branch=None,
            recovery_budget=None,
            environment_capability=None,
            learning=replace(previous.learning, tick=self.tick),
        )
        self._last_executive_decision = None
        self._last_executive_prediction_error = None
        self._last_affordance_prediction_error = None
        self._last_executive_world_action = None
        self._last_generation_trace = None
        self._last_language_emission = None
        self._language_fallback_requires_replan = False
        self._last_content_selection = None
        self._last_content_prediction_error = None
        self._content_feedback_applied = False
        self._recovery_portfolio = None

    def attach_workspace_router(self, router: WorkspaceRouter | None) -> None:
        """Attach the capacity-limited candidate router used by runtime cognition."""

        if router is not None and not isinstance(router, WorkspaceRouter):
            raise TypeError("router must be a WorkspaceRouter or None")
        self._workspace_router = router

    def attach_episodic_memory(self, store: EpisodicMemoryStore | None) -> None:
        """Attach Taiji-owned working/episodic memory for the v1 runtime."""

        if store is not None and not isinstance(store, EpisodicMemoryStore):
            raise TypeError("store must be an EpisodicMemoryStore or None")
        self._episodic_memory = store

    def attach_semantic_memory(self, learner: SemanticMemoryLearner | None) -> None:
        """Attach the slow semantic learner fed by Taiji episodic outcomes."""

        if learner is not None and not isinstance(learner, SemanticMemoryLearner):
            raise TypeError("learner must be a SemanticMemoryLearner or None")
        if learner is not None and learner.cue_dim != self.perception.feature_dim:
            raise ValueError("semantic learner cue_dim must match the perception feature dimension")
        self._semantic_memory = learner

    def consolidate_semantic_memory(
        self, *, epochs: int = 300, learning_rate: float = 0.1
    ) -> float:
        """Replay Taiji-owned episodic outcomes into the attached semantic learner."""

        if self._semantic_memory is None:
            raise RuntimeError("semantic memory learner is not attached")
        if self._episodic_memory is None or self._episodic_memory.count == 0:
            raise RuntimeError("semantic consolidation requires episodic records")
        loss = self._semantic_memory.consolidate(
            self._episodic_memory,
            epochs=epochs,
            learning_rate=learning_rate,
        )
        concepts = self._concept_formation.consolidate(
            self._episodic_memory.records,
            tick=self.tick,
        )
        development = replace(
            self._cognitive_state.development,
            tick=self.tick,
            last_update_source="semantic-consolidation",
            lineage=self._bounded_ids(
                self._cognitive_state.development.lineage,
                f"semantic-consolidation:{self.tick}",
                limit=self._lineage_limit(),
            ),
        )
        self._cognitive_state = replace(
            self._cognitive_state,
            concepts=concepts,
            development=development,
        )
        return loss

    def attach_procedural_memory(self, learner: ProceduralMemoryLearner | None) -> None:
        """Attach the slow procedural learner used by explicit action routing."""

        if learner is not None and not isinstance(learner, ProceduralMemoryLearner):
            raise TypeError("learner must be a ProceduralMemoryLearner or None")
        if learner is not None and learner.cue_dim != self.perception.feature_dim:
            raise ValueError(
                "procedural learner cue_dim must match the perception feature dimension"
            )
        self._procedural_memory = learner

    def attach_procedural_sequence_memory(self, learner: ProceduralSequenceLearner | None) -> None:
        """Attach the recurrent procedural reader for ordered action traces."""

        if learner is not None and not isinstance(learner, ProceduralSequenceLearner):
            raise TypeError("learner must be a ProceduralSequenceLearner or None")
        if learner is not None and learner.cue_dim != self.perception.feature_dim:
            raise ValueError(
                "procedural sequence learner cue_dim must match the perception feature dimension"
            )
        self._procedural_sequence_memory = learner

    @property
    def procedural_sequence_memory(self) -> ProceduralSequenceLearner | None:
        """Return the optional recurrent procedural reader."""

        return self._procedural_sequence_memory

    def consolidate_procedural_memory(
        self, *, epochs: int = 300, learning_rate: float = 0.1
    ) -> float:
        """Replay Taiji-owned action experiences into the procedural learner."""

        if self._procedural_memory is None:
            raise RuntimeError("procedural memory learner is not attached")
        if self._episodic_memory is None or self._episodic_memory.count == 0:
            raise RuntimeError("procedural consolidation requires episodic records")
        return self._procedural_memory.consolidate(
            self._episodic_memory,
            epochs=epochs,
            learning_rate=learning_rate,
        )

    def attach_homeostatic_controller(self, controller: HomeostaticController | None) -> None:
        """Attach the event-driven controller for Taiji internal drives."""

        if controller is not None and not isinstance(controller, HomeostaticController):
            raise TypeError("controller must be a HomeostaticController or None")
        self._homeostatic_controller = controller

    def homeostatic_drive(self) -> HomeostaticDrive:
        if self._homeostatic_controller is None:
            raise RuntimeError("homeostatic controller is not attached")
        return self._homeostatic_controller.drive(self._cognitive_state.homeostasis)

    def homeostatic_transition(
        self,
        *,
        mode: str = "auto",
        prediction_error: float = 0.0,
        novelty: float = 0.0,
        reward: float = 0.0,
        resource_cost: float = 0.0,
    ) -> HomeostaticState:
        if self._homeostatic_controller is None:
            raise RuntimeError("homeostatic controller is not attached")
        if mode == "auto":
            mode = self._homeostatic_controller.select_mode(self._cognitive_state.homeostasis)
        state = self._homeostatic_controller.update(
            self._cognitive_state.homeostasis,
            prediction_error=prediction_error,
            novelty=novelty,
            reward=reward,
            resource_cost=resource_cost,
            mode=mode,
        )
        state = replace(state, tick=self.tick)
        self._cognitive_state = replace(self._cognitive_state, homeostasis=state)
        return state

    def attach_executive(self, controller: ExecutiveController | None) -> None:
        """Attach the learned Taiji executive over structured candidates."""

        if controller is not None and not isinstance(controller, ExecutiveController):
            raise TypeError("controller must be an ExecutiveController or None")
        self._executive = None if controller is None else controller.to(self.device)

    def attach_affordance_features(self, source: LearnedAffordanceFeatures | None) -> None:
        """Attach the learned numeric feature source for world affordances."""

        if source is not None and not isinstance(source, LearnedAffordanceFeatures):
            raise TypeError("source must be a LearnedAffordanceFeatures or None")
        if source is not None and source.context_dim != self.perception.feature_dim:
            raise ValueError(
                "affordance feature source context_dim must match Taiji perception feature_dim"
            )
        self._affordance_features = None if source is None else source.to(self.device)
        self._affordance_grounding = (
            None if source is None else WorldAffordanceGroundingProducer(source.input_dim)
        )
        self._cognitive_state = replace(
            self._cognitive_state,
            world=self._ground_world_state(self._cognitive_state.world),
        )

    def _ground_world_state(self, world: WorldState) -> WorldState:
        if self._affordance_grounding is None or not world.affordances:
            return world
        return replace(
            world,
            affordances=tuple(
                self._affordance_grounding.ground(world, affordance)
                for affordance in world.affordances
            ),
        )

    @property
    def last_executive_decision(self) -> ExecutiveDecision | None:
        return self._last_executive_decision

    @property
    def last_executive_prediction_error(self) -> float | None:
        return self._last_executive_prediction_error

    @property
    def last_delayed_executive_prediction_error(self) -> float | None:
        return self._last_delayed_executive_prediction_error

    @property
    def last_affordance_prediction_error(self) -> float | None:
        return self._last_affordance_prediction_error

    @property
    def last_executive_world_action(self) -> WorldAction | None:
        return self._last_executive_world_action

    def select_executive(
        self,
        candidates: Sequence[ExecutiveCandidate] | None = None,
        *,
        novelty: float = 0.0,
        resource_budget: float = 1.0,
    ) -> ExecutiveDecision:
        """Select an intent/content pair from current Taiji cognitive state."""

        if self._executive is None:
            raise RuntimeError("executive controller is not attached")
        candidates = (
            self.synthesize_executive_candidates() if candidates is None else tuple(candidates)
        )
        context = ExecutiveContext.from_state(
            self._cognitive_state,
            novelty=novelty,
            resource_budget=resource_budget,
        )
        decision = self._executive.select(candidates, context)
        plan_candidates = tuple(
            PlanCandidate(
                plan_id=candidate.candidate_id,
                action_kind=candidate.action_intent.kind,
                expected_value=float(decision.scores[candidate.candidate_id]),
                risk=1.0 - candidate.action_intent.confidence,
            )
            for candidate in candidates
        )
        self._last_executive_decision = decision
        self._last_executive_prediction_error = None
        self._last_delayed_executive_prediction_error = None
        self._last_affordance_prediction_error = None
        self._last_executive_world_action = None
        self._pending_executive_credit = None
        self._cognitive_state = replace(
            self._cognitive_state,
            plan=PlanState(
                tick=self.tick,
                candidates=plan_candidates,
                selected_plan_id=decision.selected.candidate_id,
            ),
            action_intent=decision.action_intent,
        )
        return decision

    def synthesize_executive_candidates(self) -> tuple[ExecutiveCandidate, ...]:
        """Derive structured candidates from Taiji-owned current affordances."""

        if self._cognitive_state.percept is None:
            raise RuntimeError("executive candidate synthesis requires a current perception")
        if self._affordance_features is None:
            raise RuntimeError(
                "executive candidate synthesis requires an attached learned affordance feature source"
            )
        percept_features, world_latent, world_uncertainty = self._affordance_context()
        feature_map = {
            affordance.affordance_id: tuple(
                float(value)
                for value in self._affordance_features.features_for(
                    affordance,
                    percept_features=percept_features,
                    world_latent=world_latent,
                    world_uncertainty=world_uncertainty,
                )
                .detach()
                .flatten()
            )
            for affordance in self._cognitive_state.world.affordances
        }
        return ExecutiveCandidate.synthesize_from_state(
            self._cognitive_state,
            features_by_affordance=feature_map,
        )

    def _affordance_context(self) -> tuple[torch.Tensor, torch.Tensor, float]:
        if self._cognitive_state.percept is None:
            raise RuntimeError("affordance context requires a current perception")
        percept_features = self._cognitive_state.percept.features
        world_latent = self._cognitive_state.world.latent
        if world_latent.numel() == 0:
            world_latent = percept_features
        return percept_features, world_latent, self._cognitive_state.world.uncertainty

    def record_executive_outcome(
        self,
        outcome: Outcome,
        *,
        learn: bool = True,
        source_affordance: WorldAffordance | None = None,
        affordance_context: tuple[torch.Tensor, torch.Tensor, float] | None = None,
    ) -> float:
        """Train executive selection from an outcome produced by an environment."""

        if self._executive is None:
            raise RuntimeError("executive controller is not attached")
        if self._last_executive_decision is None:
            raise RuntimeError("executive outcome requires a prior selection")
        if not isinstance(outcome, Outcome):
            raise TypeError("outcome must be a Taiji Outcome")
        if outcome.intent_id != self._last_executive_decision.action_intent.intent_id:
            raise ValueError("executive outcome must reference the selected ActionIntent")
        error = self._executive.update(self._last_executive_decision, outcome.reward)
        self._last_executive_prediction_error = error
        self._last_affordance_prediction_error = None
        if learn and self._affordance_features is not None:
            affordance_id = self._last_executive_decision.selected.source_affordance_id
            affordance = source_affordance
            if affordance is not None and affordance.affordance_id != affordance_id:
                raise ValueError("source_affordance must match the selected executive candidate")
            if affordance_id is not None and affordance is None:
                affordance = next(
                    (
                        item
                        for item in self._cognitive_state.world.affordances
                        if item.affordance_id == affordance_id
                    ),
                    None,
                )
            if affordance is not None:
                if affordance_context is None:
                    affordance_context = self._affordance_context()
                percept_features, world_latent, world_uncertainty = affordance_context
                self._last_affordance_prediction_error = self._affordance_features.online_update(
                    affordance,
                    outcome.reward,
                    percept_features=percept_features,
                    world_latent=world_latent,
                    world_uncertainty=world_uncertainty,
                )
        self._cognitive_state = replace(self._cognitive_state, outcome=outcome)
        return error

    def record_delayed_executive_credit(
        self,
        reward: float,
        *,
        learn: bool | None = None,
    ) -> float:
        """Credit the action that led to a later reward, even after replanning."""

        if self._executive is None:
            raise RuntimeError("executive controller is not attached")
        pending = self._pending_executive_credit
        if pending is None:
            raise RuntimeError("delayed executive credit requires an executed action")
        apply_learning = pending.learn if learn is None else bool(learn)
        error = self._executive.update(pending.decision, float(reward))
        self._last_delayed_executive_prediction_error = error
        self._last_executive_prediction_error = error
        self._last_affordance_prediction_error = None
        if (
            apply_learning
            and self._affordance_features is not None
            and pending.affordance is not None
        ):
            if pending.percept_features is None or pending.world_latent is None:
                raise RuntimeError("delayed affordance credit is missing its causal context")
            self._last_affordance_prediction_error = self._affordance_features.online_update(
                pending.affordance,
                float(reward),
                percept_features=pending.percept_features,
                world_latent=pending.world_latent,
                world_uncertainty=pending.world_uncertainty,
            )
        self._pending_executive_credit = None
        return error

    def execute_executive_action(
        self,
        environment: TaijiEnvironment,
        *,
        decision: ExecutiveDecision | None = None,
        action_symbol: int | None = None,
        learn: bool = True,
        learn_world: bool | None = None,
    ) -> Outcome:
        """Execute a selected executive intent through a motor environment.

        ``TaijiEnvironment`` currently exposes an integer motor channel.  The
        selected structured intent remains the owner of the action metadata;
        only its explicit ``action_symbol`` parameter crosses this terminal
        organ boundary.  The environment's returned sensation is fed back as
        the next Taiji observation.
        """

        if self._executive is None:
            raise RuntimeError("executive controller is not attached")
        if not isinstance(environment, TaijiEnvironment):
            raise TypeError("environment must implement TaijiEnvironment")
        selected = decision or self._last_executive_decision
        if selected is None:
            raise RuntimeError("executive action requires a prior selection")
        if decision is not None:
            self._last_executive_decision = decision
            self._cognitive_state = replace(
                self._cognitive_state,
                action_intent=decision.action_intent,
            )
        parameters = selected.action_intent.parameters
        selected_symbol = (
            parameters.get("action_symbol") if action_symbol is None else action_symbol
        )
        if isinstance(selected_symbol, bool) or not isinstance(selected_symbol, int):
            raise ValueError("executive ActionIntent requires an integer action_symbol")
        available = parameters.get("available_actions")
        if available is not None and int(selected_symbol) not in tuple(
            int(item) for item in available
        ):
            raise ValueError("executive action_symbol is not in the ActionIntent available_actions")
        world_action = selected.to_world_action(
            tick=self._cognitive_state.world.tick,
            provenance="planned",
        )
        kernel_decision = super().act((int(selected_symbol),), sample=False)
        if kernel_decision.action_symbol != int(selected_symbol):
            raise RuntimeError("Taiji motor bridge did not preserve executive action_symbol")
        self._last_executive_world_action = world_action
        if self._world_dynamics is not None:
            prediction = self._world_dynamics.predict(
                self._cognitive_state.world,
                world_action,
                register_parameters=False,
            )
            self._cognitive_state = replace(
                self._cognitive_state,
                world_prediction=WorldPredictionRecord(
                    action=world_action,
                    predicted_state=prediction.state,
                    predicted_reward=prediction.reward,
                    predicted_success_probability=prediction.success_probability,
                    uncertainty=prediction.uncertainty,
                    uncertainty_mode=prediction.uncertainty_mode,
                    online_update_count=self._world_dynamics.online_updates,
                ),
            )
        selected_affordance = next(
            (
                item
                for item in self._cognitive_state.world.affordances
                if item.affordance_id == selected.selected.source_affordance_id
            ),
            None,
        )
        affordance_context = (
            self._affordance_context()
            if selected_affordance is not None and self._affordance_features is not None
            else None
        )
        result = environment.step(int(selected_symbol))
        if not isinstance(result, EnvironmentOutcome):
            raise TypeError("environment must return an EnvironmentOutcome")
        self.settle_action(
            result.reward,
            learn=learn,
            success=result.success,
            terminal=result.terminal,
            world_state=result.world_state,
            world_action=world_action if result.world_state is not None else None,
            learn_world=learn_world,
            provenance="experienced",
        )
        experienced = self._cognitive_state.outcome
        if experienced is None:
            raise RuntimeError("executive environment outcome was not recorded")
        transition = self._cognitive_state.world_transition
        prediction_record = self._cognitive_state.world_prediction
        self.record_executive_outcome(
            experienced,
            learn=learn,
            source_affordance=selected_affordance,
            affordance_context=affordance_context,
        )
        if not result.terminal:
            self._pending_executive_credit = _PendingExecutiveCredit(
                decision=selected,
                affordance=selected_affordance,
                percept_features=(
                    None if affordance_context is None else affordance_context[0].detach().clone()
                ),
                world_latent=(
                    None if affordance_context is None else affordance_context[1].detach().clone()
                ),
                world_uncertainty=(0.0 if affordance_context is None else affordance_context[2]),
                learn=learn,
            )
        replan_requested = self._replan_required
        self._replan_required = bool(
            not result.terminal
            and (replan_requested or result.success is False or result.reward < 0.0)
        )
        self.observe(result.sensation, learn=learn)
        if transition is not None:
            self._cognitive_state = replace(
                self._cognitive_state,
                action_intent=selected.action_intent,
                outcome=experienced,
                world_transition=transition,
                world_prediction=prediction_record,
            )
        self._record_environment_capability(result)
        return experienced

    def replan_executive_after_failure(
        self,
        candidates: Sequence[ExecutiveCandidate],
        *,
        novelty: float = 0.0,
        resource_budget: float = 1.0,
    ) -> ExecutiveDecision:
        """Select an alternative executive candidate after a failed action."""

        if not self._replan_required:
            raise RuntimeError("executive replanning has not been requested")
        if self._last_executive_decision is None:
            raise RuntimeError("executive replanning requires a prior selection")
        alternatives = tuple(
            candidate
            for candidate in candidates
            if candidate.candidate_id != self._last_executive_decision.selected.candidate_id
        )
        if not alternatives:
            raise RuntimeError("executive replanning requires an alternative candidate")
        pending_credit = self._pending_executive_credit
        decision = self.select_executive(
            alternatives,
            novelty=novelty,
            resource_budget=resource_budget,
        )
        if pending_credit is not None:
            self._pending_executive_credit = pending_credit
        self._replan_required = False
        return decision

    def attach_generation_controller(self, controller: GenerationController | None) -> None:
        """Attach the organ bridge for content and structured tool generation."""

        if controller is not None and not isinstance(controller, GenerationController):
            raise TypeError("controller must be a GenerationController or None")
        self._generation_controller = controller

    @property
    def generation_trace(self) -> GenerationTrace | None:
        return self._last_generation_trace

    def generate_tool_call(
        self,
        *,
        tool_name: str | None = None,
        channel: str | None = None,
        provenance: str = "planned",
    ) -> ToolCall:
        """Render the current Taiji action intent through the tool organ."""

        if self._generation_controller is None:
            raise RuntimeError("generation controller is not attached")
        intent = self._cognitive_state.action_intent
        if intent is None:
            raise RuntimeError("tool generation requires a pending ActionIntent")
        source_goal_id = intent.source_goal_id
        if source_goal_id is None and self._planned_rollout is not None:
            source_goal_id = self._planned_rollout.goal_id
        trace = self._generation_controller.generate_tool_call(
            intent,
            tool_name=tool_name,
            source_goal_id=source_goal_id,
            channel=channel,
            provenance=provenance,
        )
        self._last_generation_trace = trace
        return trace.tool_call

    def attach_content_selector(self, selector: ContentSelector | None) -> None:
        """Attach Taiji-owned learned selection of semantic content candidates."""

        if selector is not None and not isinstance(selector, ContentSelector):
            raise TypeError("selector must be a ContentSelector or None")
        self._content_selector = selector

    @property
    def last_content_selection(self) -> ContentSelectionDecision | None:
        return self._last_content_selection

    @property
    def last_content_prediction_error(self) -> float | None:
        return self._last_content_prediction_error

    def select_content(
        self,
        candidates: Sequence[ContentCandidate],
        *,
        novelty: float = 0.0,
        resource_budget: float = 1.0,
    ) -> ContentSelectionDecision:
        """Select semantic content from the current Taiji goal/world state."""

        if self._content_selector is None:
            raise RuntimeError("content selector is not attached")
        context = ContentSelectionContext.from_state(
            self._cognitive_state.goals,
            self._cognitive_state.world,
            novelty=novelty,
            resource_budget=resource_budget,
        )
        decision = self._content_selector.select(tuple(candidates), context)
        self._last_content_selection = decision
        self._last_content_prediction_error = None
        self._content_feedback_applied = False
        return decision

    def replan_content_after_language_fallback(
        self,
        candidates: Sequence[ContentCandidate],
        *,
        novelty: float = 0.0,
        resource_budget: float = 1.0,
    ) -> ContentSelectionDecision:
        """Select an alternative semantic plan after unsafe text realization.

        The failed candidate is excluded for this replan attempt.  The
        language organ remains an effector: this method only chooses a new
        Taiji-owned content plan; ``express_selected_content`` performs the
        subsequent organ-specific realization.
        """

        if not self._language_fallback_requires_replan:
            raise RuntimeError("language fallback has not requested content replanning")
        if self._last_content_selection is None:
            raise RuntimeError("language fallback replanning requires a prior content selection")
        previous_id = self._last_content_selection.selected.candidate_id
        alternatives = tuple(
            candidate for candidate in candidates if candidate.candidate_id != previous_id
        )
        if not alternatives:
            raise RuntimeError(
                "language fallback replanning requires an alternative content candidate"
            )
        decision = self.select_content(
            alternatives,
            novelty=novelty,
            resource_budget=resource_budget,
        )
        self._replan_required = False
        self._language_fallback_requires_replan = False
        return decision

    def selected_content_plan(self) -> ContentPlan:
        if self._last_content_selection is None:
            raise RuntimeError("content selection has not been performed")
        return self._last_content_selection.selected.to_content_plan()

    def express_selected_content(
        self,
        *,
        modality: str = "tool",
        channel: str | None = None,
    ) -> ExpressionPlan:
        if self._generation_controller is None:
            raise RuntimeError("generation controller is not attached")
        return self._generation_controller.plan_expression(
            self.selected_content_plan(),
            modality=modality,
            channel=channel,
        )

    def attach_language_organ(self, organ: LanguageOrgan | None) -> None:
        """Attach a replaceable terminal text organ owned by Taiji's boundary."""

        if organ is not None:
            if (
                self._language_provider_artifact is not None
                and organ.backend_id != self._language_provider_artifact.backend_id
            ):
                raise ValueError("language organ backend does not match provider artifact")
            self._language_backend_registry.validate(organ)
        self._language_organ = organ

    def attach_language_provider_artifact(self, artifact: LanguageProviderArtifact | None) -> None:
        """Record an externally loaded provider without importing its runtime."""

        if artifact is not None:
            if not isinstance(artifact, LanguageProviderArtifact):
                raise TypeError("artifact must be a LanguageProviderArtifact or None")
            self._language_backend_registry.get(artifact.backend_id)
            if (
                self._language_organ is not None
                and self._language_organ.backend_id != artifact.backend_id
            ):
                raise ValueError("provider artifact backend does not match language organ")
        self._language_provider_artifact = artifact

    @property
    def language_provider_artifact(self) -> LanguageProviderArtifact | None:
        return self._language_provider_artifact

    def attach_language_backend_registry(self, registry: LanguageBackendRegistry | None) -> None:
        """Attach descriptors for allowed terminal language-organ backends."""

        selected_registry = registry or LanguageBackendRegistry.default()
        if not isinstance(selected_registry, LanguageBackendRegistry):
            raise TypeError("registry must be a LanguageBackendRegistry or None")
        if self._language_organ is not None:
            selected_registry.validate(self._language_organ)
        self._language_backend_registry = selected_registry

    @property
    def last_language_emission(self) -> LanguageEmission | None:
        return self._last_language_emission

    @property
    def last_language_validation(self) -> LanguageValidation | None:
        return (
            None
            if self._last_language_emission is None
            else self._last_language_emission.validation
        )

    @property
    def language_fallback_count(self) -> int:
        return self._language_fallback_count

    def _apply_content_feedback(self, reward: float) -> None:
        if (
            self._content_selector is not None
            and self._last_content_selection is not None
            and not self._content_feedback_applied
        ):
            self._last_content_prediction_error = self._content_selector.update(
                self._last_content_selection.selected,
                self._last_content_selection.context,
                reward,
            )
            self._content_feedback_applied = True

    def emit_language(
        self,
        expression: ExpressionPlan | None = None,
        *,
        channel: str = "message",
    ) -> LanguageEmission:
        """Emit text through the terminal organ without creating cognition.

        When no expression is supplied, the expression is derived from the
        already selected semantic content.  The organ cannot create content,
        goals, plans, or actions on its own.
        """

        if self._language_organ is None:
            raise RuntimeError("language organ is not attached")
        selected_expression = expression
        if selected_expression is None:
            selected_expression = self.express_selected_content(
                modality="text",
                channel=channel,
            )
        if not isinstance(selected_expression, ExpressionPlan):
            raise TypeError("language emission requires an ExpressionPlan")
        if selected_expression.modality != "text":
            raise ValueError("language emission requires a text ExpressionPlan")
        emission = self._language_organ.emit(selected_expression)
        if not isinstance(emission, LanguageEmission):
            raise TypeError("language organ must return a LanguageEmission")
        self._last_language_emission = emission
        if emission.fallback_used:
            self._language_fallback_count += 1
            self._language_fallback_requires_replan = True
            self._replan_required = True
            self._apply_content_feedback(-1.0)
        return emission

    def execute_tool_call(
        self,
        environment: TaijiToolEnvironment,
        *,
        call: ToolCall | None = None,
        learn: bool = True,
    ) -> Outcome:
        """Execute a generated call and feed its outcome back into Taiji."""

        if not isinstance(environment, TaijiToolEnvironment):
            raise TypeError("environment must implement TaijiToolEnvironment")
        selected_call = call
        if selected_call is None:
            if self._last_generation_trace is None:
                raise RuntimeError("tool execution requires a generated ToolCall")
            selected_call = self._last_generation_trace.tool_call
        intent = self._cognitive_state.action_intent
        if intent is None or intent.intent_id != selected_call.intent_id:
            raise ValueError("tool call must reference the pending ActionIntent")
        result = environment.execute_tool(
            selected_call.tool_name,
            dict(selected_call.parameters),
        )
        if not isinstance(result, EnvironmentOutcome):
            raise TypeError("tool environment must return an EnvironmentOutcome")
        taiji_outcome = self.settle_action(
            result.reward,
            learn=learn,
            success=result.success,
            terminal=result.terminal,
            provenance="experienced",
        )
        experienced = self._cognitive_state.outcome
        self.observe(result.sensation, learn=learn)
        return (
            experienced
            if experienced is not None
            else Outcome(
                intent_id=intent.intent_id,
                reward=taiji_outcome.reward,
                success=result.success,
                terminal=result.terminal,
                tick=taiji_outcome.tick,
            )
        )

    def attach_goal_planner(self, planner: GoalPlanner | None) -> None:
        """Attach the Taiji-owned planner for executable goal candidates."""

        if planner is not None and not isinstance(planner, GoalPlanner):
            raise TypeError("planner must be a GoalPlanner or None")
        self._goal_planner = planner

    def set_goals(self, goals: Sequence[Goal]) -> None:
        """Register the current goal hierarchy in the Taiji cognitive state."""

        goals = tuple(goals)
        if any(not isinstance(goal, Goal) for goal in goals):
            raise TypeError("goals must contain Goal contracts")
        self._cognitive_state = replace(
            self._cognitive_state,
            goals=GoalState(tick=self.tick, goals=goals),
        )

    def _apply_concept_affinity(
        self, candidates: Sequence[PlanningCandidate]
    ) -> tuple[PlanningCandidate, ...]:
        matches = self._concept_matches_for_world(self._cognitive_state.world)
        return tuple(
            replace(
                candidate,
                concept_affinity=max(
                    (
                        match.score * match.concept.confidence * match.concept.outcome_mean
                        for match in matches
                        if candidate.action.kind in match.concept.action_kinds
                    ),
                    default=candidate.concept_affinity,
                ),
            )
            for candidate in candidates
        )

    def _apply_concept_sequence_affinity(self, rollout: ImaginedRollout) -> ImaginedRollout:
        action_kinds = tuple(step.action.kind for step in rollout.steps)
        matches = self._concept_matches_for_world(self._cognitive_state.world)
        sequence_affinity = max(
            (
                match.score
                * match.concept.confidence
                * match.concept.outcome_mean
                * self._concept_formation.suffix_sequence_affinity(
                    match.concept,
                    action_kinds,
                    current_state=self._cognitive_state.world,
                )
                for match in matches
            ),
            default=rollout.concept_sequence_affinity,
        )
        return replace(rollout, concept_sequence_affinity=sequence_affinity)

    def plan_actions(
        self,
        candidates: Sequence[PlanningCandidate],
        *,
        goal_id: str | None = None,
    ) -> PlanningDecision:
        """Compare executable world candidates and persist the selected plan."""

        if self._goal_planner is None:
            raise RuntimeError("goal planner is not attached")
        enriched_candidates = self._apply_concept_affinity(tuple(candidates))
        decision = self._goal_planner.plan(
            self._cognitive_state.goals,
            enriched_candidates,
            tick=self.tick,
            goal_id=goal_id,
        )
        self._cognitive_state = replace(
            self._cognitive_state,
            plan=decision.plan,
            goals=replace(self._cognitive_state.goals, tick=self.tick),
        )
        return decision

    def predict_world_candidates(
        self,
        candidates: Sequence[PlanningCandidate],
    ) -> tuple[PlanningCandidate, ...]:
        """Project structured candidates through the attached world learner.

        Goal semantics such as expected progress remain planner-owned.  The
        world learner supplies only the numeric consequence estimates and the
        latest observed model error becomes candidate uncertainty.
        """

        if self._world_dynamics is None:
            raise RuntimeError("world dynamics is not attached")
        projected = []
        model_error = self._cognitive_state.world_prediction
        model_uncertainty = (
            0.0
            if model_error is None or model_error.state_error is None
            else min(1.0, max(0.0, float(model_error.state_error)))
        )
        for candidate in candidates:
            if not isinstance(candidate, PlanningCandidate):
                raise TypeError("candidates must contain PlanningCandidate values")
            if candidate.action.tick != self._cognitive_state.world.tick:
                raise ValueError("world planning candidates must act at the current world tick")
            prediction = self._world_dynamics.predict(
                self._cognitive_state.world,
                candidate.action,
                register_parameters=False,
            )
            projected.append(
                replace(
                    candidate,
                    predicted_reward=prediction.reward,
                    success_probability=prediction.success_probability,
                    uncertainty=max(
                        candidate.uncertainty,
                        model_uncertainty,
                        prediction.uncertainty,
                    ),
                    uncertainty_mode=prediction.uncertainty_mode,
                )
            )
        if not projected:
            raise ValueError("world planning requires executable candidates")
        return tuple(projected)

    def plan_world_actions(
        self,
        candidates: Sequence[PlanningCandidate],
        *,
        goal_id: str | None = None,
    ) -> PlanningDecision:
        """Plan executable candidates after world-dynamics projection."""

        return self.plan_actions(self.predict_world_candidates(candidates), goal_id=goal_id)

    def imagine_world_rollout(
        self,
        rollout_id: str,
        goal_id: str,
        steps: Sequence[PlanningCandidate],
        *,
        confidence: float = 1.0,
        recovery_lineage: RecoveryRolloutLineage | None = None,
    ) -> ImaginedRollout:
        """Roll the attached world learner forward over structured actions."""

        if self._world_dynamics is None:
            raise RuntimeError("world dynamics is not attached")
        if not steps:
            raise ValueError("world rollout requires at least one step")
        imagined_state = self._cognitive_state.world
        latest_prediction = self._cognitive_state.world_prediction
        model_uncertainty = (
            0.0
            if latest_prediction is None or latest_prediction.state_error is None
            else min(1.0, max(0.0, float(latest_prediction.state_error)))
        )
        imagined_steps = []
        for template in steps:
            if not isinstance(template, PlanningCandidate):
                raise TypeError("world rollout steps must contain PlanningCandidate values")
            if template.action.tick != imagined_state.tick:
                raise ValueError("world rollout actions must follow predicted world ticks")
            prediction = self._world_dynamics.predict(
                imagined_state,
                template.action,
                register_parameters=False,
            )
            imagined_steps.append(
                replace(
                    template,
                    action=replace(template.action, provenance="world-dynamics"),
                    predicted_reward=prediction.reward,
                    success_probability=prediction.success_probability,
                    uncertainty=max(
                        template.uncertainty,
                        model_uncertainty,
                        prediction.uncertainty,
                    ),
                    uncertainty_mode=prediction.uncertainty_mode,
                    prediction_provenance="world-dynamics",
                )
            )
            imagined_state = prediction.state
        return ImaginedRollout(
            rollout_id=rollout_id,
            goal_id=goal_id,
            steps=tuple(imagined_steps),
            confidence=confidence,
            recovery_lineage=recovery_lineage,
        )

    def synthesize_recovery_rollouts(
        self,
        *,
        available_actions: Sequence[int] | None = None,
        action_kinds: Sequence[str] | None = None,
        horizon: int = 1,
        goal_id: str | None = None,
        resource_budget: float = 1.0,
    ) -> tuple[ImaginedRollout, ...]:
        """Generate executable recovery branches from current world affordances."""

        if self._world_dynamics is None:
            raise RuntimeError("recovery rollout synthesis requires world dynamics")
        if not self._cognitive_state.world.affordances:
            raise RuntimeError("recovery rollout synthesis requires current world affordances")
        capability = self._cognitive_state.environment_capability
        if (available_actions is None) != (action_kinds is None):
            raise ValueError("recovery actions and action_kinds must be provided together")
        if available_actions is None or action_kinds is None:
            if capability is None:
                raise RuntimeError("recovery synthesis requires a current environment capability")
            actions = capability.actions
            kinds = capability.action_kinds
        else:
            actions = tuple(int(action) for action in available_actions)
            kinds = tuple(str(kind) for kind in action_kinds)
            if capability is not None and (actions, kinds) != (
                capability.actions,
                capability.action_kinds,
            ):
                raise ValueError("recovery synthesis received a stale environment capability")
        if capability is not None and capability.tick != self._cognitive_state.world.tick:
            raise RuntimeError(
                "recovery synthesis requires a capability from the current world tick"
            )
        if not actions or len(actions) != len(kinds) or len(set(actions)) != len(actions):
            raise ValueError("recovery actions and action_kinds must be aligned and unique")
        if len(set(kinds)) != len(kinds):
            raise ValueError("recovery action_kinds must be unique")
        if any(action < 0 or action >= self.config.alphabet_size for action in actions):
            raise ValueError("recovery action is outside the motor alphabet")
        if int(horizon) <= 0:
            raise ValueError("recovery rollout horizon must be positive")
        if not 0.0 <= float(resource_budget) <= 1.0:
            raise ValueError("recovery resource_budget must be in [0, 1]")
        active_goal = max(
            self._cognitive_state.goals.goals,
            key=lambda item: (item.priority, -item.progress, item.goal_id),
            default=None,
        )
        selected_goal_id = goal_id
        if selected_goal_id is None and self._cognitive_state.recovery_branch is not None:
            selected_goal_id = self._cognitive_state.recovery_branch.goal_id
        if selected_goal_id is None and active_goal is not None:
            selected_goal_id = active_goal.goal_id
        if selected_goal_id is None:
            raise RuntimeError("recovery rollout synthesis requires a goal")
        recovery_branch = self._cognitive_state.recovery_branch
        if recovery_branch is None:
            available_resource_budget = float(resource_budget)
        else:
            if recovery_branch.resource_budget is None:
                recovery_branch = replace(
                    recovery_branch,
                    resource_budget=float(resource_budget),
                )
                self._cognitive_state = replace(
                    self._cognitive_state,
                    recovery_branch=recovery_branch,
                )
            available_resource_budget = min(
                float(resource_budget),
                recovery_branch.remaining_resource,
            )
        recovery_budget = self._cognitive_state.recovery_budget
        if recovery_budget is None:
            recovery_budget = RecoveryBudgetState(total_budget=float(resource_budget))
        elif recovery_budget.total_budget is None:
            recovery_budget = replace(
                recovery_budget,
                total_budget=float(resource_budget),
            )
        self._cognitive_state = replace(
            self._cognitive_state,
            recovery_budget=recovery_budget,
        )
        available_resource_budget = min(
            available_resource_budget,
            recovery_budget.remaining_resource,
        )
        start_tick = self._cognitive_state.world.tick
        self._recovery_generation += 1
        portfolio_generation = self._recovery_generation
        recovery_prefix = (
            f"recovery:{self._state.episode_id}:tick-{start_tick}:generation-{portfolio_generation}"
        )
        rejected_key = None
        if recovery_branch is not None:
            rejected_key = self._world_dynamics.schema_registry.action_semantic_key(
                recovery_branch.rejected_action
            )
        prepared: list[tuple[WorldAffordance, WorldAction, float]] = []
        for affordance in self._cognitive_state.world.affordances:
            try:
                action_index = kinds.index(affordance.action_kind)
            except ValueError:
                continue
            parameters = dict(affordance.parameters)
            parameters["action_symbol"] = actions[action_index]
            resource_cost = float(parameters.get("resource_cost", 0.0))
            if not 0.0 <= resource_cost <= available_resource_budget:
                continue
            first_action = WorldAction(
                action_id=f"{recovery_prefix}:{affordance.affordance_id}",
                kind=affordance.action_kind,
                tick=start_tick,
                actor_id=affordance.actor_id,
                target_id=affordance.target_id,
                parameters=tuple(sorted(parameters.items())),
                provenance="recovery-synthesis",
            )
            if rejected_key is not None and (
                self._world_dynamics.schema_registry.action_semantic_key(first_action)
                == rejected_key
            ):
                continue
            prepared.append((affordance, first_action, resource_cost))
        if not prepared:
            raise RuntimeError("recovery rollout synthesis found no executable alternative")
        for _, first_action, _ in prepared:
            self._world_dynamics.register_open_set(
                self._cognitive_state.world,
                action=first_action,
                register_parameters=False,
            )
        schema_revision = self._world_dynamics.schema_registry.active_version
        rollouts = []
        for affordance, first_action, resource_cost in prepared:
            templates = tuple(
                PlanningCandidate(
                    candidate_id=(f"{recovery_prefix}:{affordance.affordance_id}:step:{index}"),
                    action=replace(
                        first_action,
                        action_id=f"{first_action.action_id}:step:{index}",
                        tick=start_tick + index,
                    ),
                    predicted_reward=0.0,
                    success_probability=0.0,
                    expected_progress=affordance.confidence,
                    resource_cost=resource_cost,
                    prediction_provenance="recovery-synthesis",
                )
                for index in range(int(horizon))
            )
            rollouts.append(
                self.imagine_world_rollout(
                    f"{recovery_prefix}:{affordance.affordance_id}",
                    selected_goal_id,
                    templates,
                    confidence=affordance.confidence,
                    recovery_lineage=(
                        None
                        if capability is None
                        else RecoveryRolloutLineage(
                            capability_tick=capability.tick,
                            capability_actions=capability.actions,
                            capability_action_kinds=capability.action_kinds,
                            affordance_id=affordance.affordance_id,
                            affordance_content_identity=affordance.content_identity,
                            action_semantic_key=self._world_dynamics.schema_registry.action_semantic_key(
                                first_action
                            ),
                            schema_revision=schema_revision,
                        )
                    ),
                )
            )
        generated = tuple(rollouts)
        portfolio_candidates = generated
        previous_portfolio = self._recovery_portfolio
        preserved_rollout: ImaginedRollout | None = None
        if (
            self._planned_rollout is not None
            and self._planned_rollout.recovery_lineage is not None
            and previous_portfolio is not None
            and previous_portfolio.status_for(self._planned_rollout.rollout_id)
            in {"active", "selected"}
            and self._recovery_rollout_is_fresh(self._planned_rollout)
        ):
            preserved_rollout = self._planned_rollout
        if preserved_rollout is not None:
            portfolio_candidates = (*generated, preserved_rollout)
        preserved_status = "active"
        if preserved_rollout is not None and previous_portfolio is not None:
            previous_status = previous_portfolio.status_for(preserved_rollout.rollout_id)
            if previous_status is not None:
                preserved_status = previous_status
        retired_rollout_ids: tuple[str, ...] = ()
        if previous_portfolio is not None:
            retired_rollout_ids = tuple(
                dict.fromkeys(
                    (
                        *previous_portfolio.retired_rollout_ids,
                        *(
                            candidate.rollout_id
                            for candidate in previous_portfolio.candidates
                            if preserved_rollout is None
                            or candidate.rollout_id != preserved_rollout.rollout_id
                        ),
                    )
                )
            )
        statuses = tuple(
            (
                candidate.rollout_id,
                (
                    preserved_status
                    if preserved_rollout is not None
                    and candidate.rollout_id == preserved_rollout.rollout_id
                    else "active"
                ),
            )
            for candidate in portfolio_candidates
        )
        self._recovery_portfolio = RecoveryPortfolio(
            portfolio_id=f"{recovery_prefix}:portfolio",
            goal_id=selected_goal_id,
            candidates=portfolio_candidates,
            statuses=statuses,
            retired_rollout_ids=retired_rollout_ids,
            selected_rollout_id=(
                None
                if preserved_rollout is None
                or previous_portfolio is None
                or previous_portfolio.status_for(preserved_rollout.rollout_id) != "selected"
                else preserved_rollout.rollout_id
            ),
        )
        return generated

    def plan_rollouts(
        self,
        rollouts: Sequence[ImaginedRollout],
        *,
        goal_id: str | None = None,
    ) -> RolloutDecision:
        """Persist the selected multi-step imagined rollout for execution."""

        if self._goal_planner is None:
            raise RuntimeError("goal planner is not attached")
        candidate_rollouts = tuple(rollouts)
        portfolio = self._recovery_portfolio
        archived_rollout_ids = set(self._recovery_archive.archived_rollout_ids)
        if portfolio is not None:
            candidate_rollouts = tuple(
                rollout
                for rollout in candidate_rollouts
                if rollout.rollout_id not in archived_rollout_ids
                if (
                    portfolio.status_for(rollout.rollout_id) in {"active", "selected"}
                    or (
                        portfolio.status_for(rollout.rollout_id) is None
                        and rollout.rollout_id not in portfolio.retired_rollout_ids
                    )
                )
            )
        elif archived_rollout_ids:
            candidate_rollouts = tuple(
                rollout
                for rollout in candidate_rollouts
                if rollout.rollout_id not in archived_rollout_ids
            )
        if not candidate_rollouts:
            if candidate_rollouts != tuple(rollouts) and all(
                rollout.rollout_id in archived_rollout_ids for rollout in rollouts
            ):
                raise RuntimeError("archived recovery rollout cannot be reintroduced")
            raise RuntimeError("recovery portfolio has no active candidates")
        fresh_rollouts = tuple(
            rollout for rollout in candidate_rollouts if self._recovery_rollout_is_fresh(rollout)
        )
        stale_portfolio_ids = tuple(
            rollout.rollout_id
            for rollout in candidate_rollouts
            if not self._recovery_rollout_is_fresh(rollout)
            and portfolio is not None
            and portfolio.status_for(rollout.rollout_id) in {"active", "selected"}
            and any(
                candidate.rollout_id == rollout.rollout_id and candidate == rollout
                for candidate in portfolio.candidates
            )
        )
        if stale_portfolio_ids and portfolio is not None:
            self._recovery_portfolio = portfolio.mark_expired(*stale_portfolio_ids)
        if candidate_rollouts and not fresh_rollouts:
            raise RuntimeError("recovery rollouts are stale; synthesize new candidates")
        enriched_rollouts = tuple(
            self._apply_concept_sequence_affinity(
                replace(rollout, steps=self._apply_concept_affinity(rollout.steps))
            )
            for rollout in fresh_rollouts
        )
        recovery_branch = self._cognitive_state.recovery_branch
        if recovery_branch is not None and self._world_dynamics is not None:
            rejected_key = self._world_dynamics.schema_registry.action_semantic_key(
                recovery_branch.rejected_action
            )
            enriched_rollouts = tuple(
                rollout
                for rollout in enriched_rollouts
                if not rollout.steps
                or self._world_dynamics.schema_registry.action_semantic_key(rollout.steps[0].action)
                != rejected_key
            )
            if not enriched_rollouts:
                raise RuntimeError(
                    "outcome recovery requires a rollout with a different first action"
                )
        decision = self._goal_planner.plan_rollouts(
            self._cognitive_state.goals,
            enriched_rollouts,
            tick=self.tick,
            goal_id=goal_id,
        )
        self._planned_rollout = decision.selected
        if (
            self._recovery_portfolio is not None
            and self._recovery_portfolio.status_for(decision.selected.rollout_id) is not None
        ):
            self._recovery_portfolio = self._recovery_portfolio.mark_selected(
                decision.selected.rollout_id
            )
        self._replan_required = False
        self._language_fallback_requires_replan = False
        self._last_rollout_prediction_error = None
        self._last_rollout_calibrated_confidence = None
        self._cognitive_state = replace(
            self._cognitive_state,
            plan=decision.plan,
            goals=replace(self._cognitive_state.goals, tick=self.tick),
            recovery_branch=(
                None
                if recovery_branch is None
                else replace(
                    recovery_branch,
                    replacement_rollout_id=decision.selected.rollout_id,
                )
            ),
        )
        return decision

    def execute_imagined_rollout_step(
        self,
        environment: TaijiEnvironment,
        *,
        available_actions: Sequence[int],
        action_kinds: Sequence[str],
        learn: bool = True,
        learn_world: bool | None = None,
    ) -> Outcome:
        """Execute one planned rollout step through the real environment."""

        if not isinstance(environment, TaijiEnvironment):
            raise TypeError("environment must implement TaijiEnvironment")
        rollout = self._planned_rollout
        if rollout is None:
            raise RuntimeError("imagined rollout execution requires a planned rollout")
        if not self._recovery_rollout_is_fresh(rollout):
            if self._recovery_portfolio is not None:
                self._recovery_portfolio = self._recovery_portfolio.mark_expired(rollout.rollout_id)
            self._planned_rollout = None
            self._replan_required = True
            raise RuntimeError("planned recovery rollout is stale; synthesize new candidates")
        step = rollout.steps[0]
        parameters = dict(step.action.parameters)
        action_symbol = parameters.get("action_symbol")
        if isinstance(action_symbol, bool) or not isinstance(action_symbol, int):
            raise ValueError("imagined rollout action requires an integer action_symbol")
        decision = self.act(
            available_actions,
            sample=False,
            procedural_action_kinds=action_kinds,
            use_plan=True,
            world_action=step.action,
        )
        if decision.action_symbol != int(action_symbol):
            raise ValueError("imagined rollout action_symbol does not match motor routing")
        result = environment.step(decision.action_symbol)
        if not isinstance(result, EnvironmentOutcome):
            raise TypeError("environment must return an EnvironmentOutcome")
        if result.world_state is None:
            raise ValueError(
                "imagined rollout execution requires an EnvironmentOutcome.world_state"
            )
        self.settle_action(
            result.reward,
            learn=learn,
            learn_world=learn_world,
            success=result.success,
            terminal=result.terminal,
            world_state=result.world_state,
            provenance="experienced",
        )
        experienced = self._cognitive_state.outcome
        if experienced is None:
            raise RuntimeError("imagined rollout environment outcome was not recorded")
        experienced_world = result.world_state
        self.observe(result.sensation, learn=learn)
        self._cognitive_state = replace(
            self._cognitive_state,
            world=replace(experienced_world, tick=self.tick),
        )
        self._record_environment_capability(result)
        self._refresh_concept_memory()
        if not result.terminal and (result.success is False or result.reward < 0.0):
            self._replan_required = True
        if self.replan_required or result.terminal or len(rollout.steps) == 1:
            self._planned_rollout = None
        else:
            suffix = ImaginedRollout(
                rollout_id=rollout.rollout_id,
                goal_id=rollout.goal_id,
                steps=rollout.steps[1:],
                confidence=rollout.confidence,
                recovery_lineage=rollout.recovery_lineage,
            )
            refreshed_suffix = self._refresh_recovery_rollout_suffix(suffix)
            if refreshed_suffix is None:
                self._planned_rollout = None
                self._replan_required = True
            else:
                self._planned_rollout = self._apply_concept_sequence_affinity(refreshed_suffix)
        if result.terminal and self._cognitive_state.planning_recovery is not None:
            self._cognitive_state = replace(self._cognitive_state, planning_recovery=None)
        if result.terminal and self._recovery_portfolio is not None:
            self._archive_current_recovery_portfolio(
                completed_rollout_id=rollout.rollout_id,
                outcome_reward=result.reward,
                outcome_success=result.success,
                terminal=True,
                evidence_count=(
                    0
                    if not self._cognitive_state.world_calibration_trace
                    else self._cognitive_state.world_calibration_trace[-1].ledger_evidence_count
                ),
                outcome_consistency=(
                    0.0
                    if not self._cognitive_state.world_calibration_trace
                    or self._cognitive_state.world_calibration_trace[-1].adjudication != "accepted"
                    else 1.0 - self._cognitive_state.world_calibration_trace[-1].ledger_uncertainty
                ),
            )
            self._recovery_portfolio = None
        if result.terminal and self._cognitive_state.recovery_branch is not None:
            self._cognitive_state = replace(self._cognitive_state, recovery_branch=None)
        return experienced

    def _refresh_recovery_rollout_suffix(self, rollout: ImaginedRollout) -> ImaginedRollout | None:
        """Rebind the next recovery step to the post-action runtime boundary."""

        if rollout.recovery_lineage is None:
            return rollout
        if not rollout.steps or self._world_dynamics is None:
            return None
        capability = self._cognitive_state.environment_capability
        if capability is None or capability.tick != self._cognitive_state.world.tick:
            return None
        lineage = rollout.recovery_lineage
        affordance = next(
            (
                item
                for item in self._cognitive_state.world.affordances
                if item.affordance_id == lineage.affordance_id
            ),
            None,
        )
        if affordance is None or affordance.content_identity != lineage.affordance_content_identity:
            return None
        next_action = rollout.steps[0].action
        try:
            capability_index = capability.action_kinds.index(next_action.kind)
        except ValueError:
            return None
        if (
            dict(next_action.parameters).get("action_symbol")
            != capability.actions[capability_index]
        ):
            return None
        rebased_steps = tuple(
            replace(
                step,
                action=replace(step.action, tick=self._cognitive_state.world.tick + index),
            )
            for index, step in enumerate(rollout.steps)
        )
        for step in rebased_steps:
            self._world_dynamics.register_open_set(
                self._cognitive_state.world,
                action=step.action,
                register_parameters=False,
            )
        refreshed_lineage = RecoveryRolloutLineage(
            capability_tick=capability.tick,
            capability_actions=capability.actions,
            capability_action_kinds=capability.action_kinds,
            affordance_id=affordance.affordance_id,
            affordance_content_identity=affordance.content_identity,
            action_semantic_key=self._world_dynamics.schema_registry.action_semantic_key(
                rebased_steps[0].action
            ),
            schema_revision=self._world_dynamics.schema_registry.active_version,
        )
        refreshed = self.imagine_world_rollout(
            rollout.rollout_id,
            rollout.goal_id,
            rebased_steps,
            confidence=rollout.confidence,
            recovery_lineage=refreshed_lineage,
        )
        if self._recovery_portfolio is not None:
            self._recovery_portfolio = self._recovery_portfolio.replace_candidate(refreshed)
        return refreshed

    def _recovery_rollout_is_fresh(self, rollout: ImaginedRollout) -> bool:
        lineage = rollout.recovery_lineage
        if lineage is None:
            return True
        capability = self._cognitive_state.environment_capability
        if capability is None or capability.tick != self._cognitive_state.world.tick:
            return False
        if lineage.capability_tick != capability.tick:
            return False
        if lineage.capability_actions != capability.actions:
            return False
        if lineage.capability_action_kinds != capability.action_kinds:
            return False
        if self._world_dynamics is None:
            return False
        if lineage.schema_revision != self._world_dynamics.schema_registry.active_version:
            return False
        current_affordance = next(
            (
                affordance
                for affordance in self._cognitive_state.world.affordances
                if affordance.affordance_id == lineage.affordance_id
            ),
            None,
        )
        if current_affordance is None:
            return False
        if current_affordance.content_identity != lineage.affordance_content_identity:
            return False
        if not rollout.steps:
            return False
        if (
            self._world_dynamics.schema_registry.action_semantic_key(rollout.steps[0].action)
            != lineage.action_semantic_key
        ):
            return False
        try:
            capability_index = capability.action_kinds.index(rollout.steps[0].action.kind)
        except ValueError:
            return False
        return (
            dict(rollout.steps[0].action.parameters).get("action_symbol")
            == capability.actions[capability_index]
        )

    @property
    def replan_required(self) -> bool:
        return self._replan_required

    @property
    def last_rollout_prediction_error(self) -> float | None:
        return self._last_rollout_prediction_error

    @property
    def last_rollout_calibrated_confidence(self) -> float | None:
        return self._last_rollout_calibrated_confidence

    @property
    def planning_recovery(self) -> PlanningRecoveryState | None:
        """Return the explicit runtime recovery mode, if one is active."""

        return self._cognitive_state.planning_recovery

    @property
    def recovery_branch(self) -> RecoveryBranchState | None:
        """Return the auditable outcome-driven recovery branch, if active."""

        return self._cognitive_state.recovery_branch

    @property
    def environment_capability(self) -> EnvironmentCapability | None:
        """Return the latest capability boundary reported by the environment."""

        return self._cognitive_state.environment_capability

    @property
    def recovery_budget(self) -> RecoveryBudgetState | None:
        """Return the episode-global recovery resource ledger."""

        return self._cognitive_state.recovery_budget

    @property
    def recovery_portfolio(self) -> RecoveryPortfolio | None:
        """Return the checkpointable recovery branch portfolio."""

        return self._recovery_portfolio

    @property
    def recovery_archive(self) -> RecoveryPortfolioArchive:
        """Return non-executable recovery summaries retained across episodes."""

        return self._recovery_archive

    def _archive_current_recovery_portfolio(
        self,
        *,
        completed_rollout_id: str | None = None,
        outcome_reward: float | None = None,
        outcome_success: bool | None = None,
        terminal: bool = False,
        evidence_count: int = 0,
        outcome_consistency: float = 1.0,
    ) -> None:
        portfolio = self._recovery_portfolio
        if portfolio is None:
            return
        entries = portfolio.archive_entries(
            self._state.episode_id,
            completed_rollout_id=completed_rollout_id,
            outcome_reward=outcome_reward,
            outcome_success=outcome_success,
            terminal=terminal,
            evidence_count=evidence_count,
            outcome_consistency=outcome_consistency,
        )
        self._recovery_archive = self._recovery_archive.append(entries)
        memory_ids = self._cognitive_state.memory.episodic_ids
        if memory_ids:
            memory_id = memory_ids[-1]
            for entry in entries:
                if entry.lifecycle != "completed":
                    continue
                self._recovery_strategy_ledger = self._recovery_strategy_ledger.admit(
                    entry,
                    memory_id=memory_id,
                )

    @property
    def recovery_strategy_ledger(self) -> RecoveryStrategyLedger:
        """Return the evidence gate for recovery-derived long-term memory."""

        return self._recovery_strategy_ledger

    @property
    def recovery_reader_dependencies(self) -> RecoveryReaderDependencyGraph:
        """Return the reader-level provenance graph for recovery memory."""

        return self._recovery_reader_dependencies

    def revoke_recovery_strategy(self, rollout_id: str) -> None:
        """Revoke a recovery strategy from future replay and consolidation."""

        previous = self._recovery_strategy_ledger
        self._recovery_strategy_ledger = previous.revoke(rollout_id)
        if self._recovery_strategy_ledger != previous:
            self._rebuild_recovery_memory(revoked_rollout_id=rollout_id)

    def _recovery_reader_contributions(
        self,
        reader_kind: str,
        baseline_payload: dict[str, Any],
        final_payload: dict[str, Any],
        records: tuple[EpisodicMemoryRecord, ...],
        approvals: Sequence[RecoveryStrategyApproval],
        *,
        epochs: int,
        learning_rate: float,
    ) -> tuple[RecoveryReaderContribution, ...]:
        """Attribute final reader state with deterministic leave-one-out replay."""

        if not approvals:
            return ()
        effects: list[float] = []
        for approval in approvals:
            ablated_records = tuple(
                record for record in records if record.memory_id != approval.memory_id
            )
            if reader_kind == "semantic":
                ablated_semantic = SemanticMemoryLearner.from_checkpoint(
                    dict(baseline_payload), device=self.device
                )
                if ablated_records:
                    ablated_semantic.consolidate(
                        ablated_records,
                        epochs=epochs,
                        learning_rate=learning_rate,
                    )
                ablated_payload = ablated_semantic.checkpoint()
                effect = _payload_distance(
                    final_payload.get("state_dict"), ablated_payload.get("state_dict")
                )
            elif reader_kind == "procedural":
                ablated_procedural = ProceduralMemoryLearner.from_checkpoint(
                    dict(baseline_payload), device=self.device
                )
                if ablated_records:
                    action_kinds = tuple(
                        str(kind) for kind in final_payload.get("action_kinds", ())
                    )
                    ablated_procedural.consolidate(
                        ablated_records,
                        epochs=epochs,
                        learning_rate=learning_rate,
                        action_kinds=action_kinds or None,
                    )
                ablated_payload = ablated_procedural.checkpoint()
                effect = _payload_distance(
                    final_payload.get("state_dict"), ablated_payload.get("state_dict")
                )
            elif reader_kind == "sequence":
                ablated_sequence = ProceduralSequenceLearner.from_checkpoint(
                    dict(baseline_payload), device=self.device
                )
                if ablated_records:
                    action_kinds = tuple(
                        str(kind) for kind in final_payload.get("action_kinds", ())
                    )
                    ablated_sequence.consolidate(
                        ablated_records,
                        epochs=epochs,
                        learning_rate=learning_rate,
                        action_kinds=action_kinds or None,
                    )
                ablated_payload = ablated_sequence.checkpoint()
                effect = _payload_distance(
                    final_payload.get("state_dict"), ablated_payload.get("state_dict")
                )
            elif reader_kind == "concept":
                ablated_concept = ConceptFormationOrgan.from_checkpoint(
                    dict(baseline_payload), device=self.device
                )
                if ablated_records:
                    ablated_concept.consolidate(ablated_records, tick=self.tick)
                ablated_payload = ablated_concept.checkpoint()
                effect = _payload_distance(
                    final_payload.get("concepts", ()), ablated_payload.get("concepts", ())
                )
            else:
                raise ValueError(f"unsupported recovery reader kind: {reader_kind}")
            effects.append(max(0.0, float(effect)))
        total_effect = sum(effects)
        if total_effect > 1e-12:
            credits = tuple(effect / total_effect for effect in effects)
        else:
            equal_credit = 1.0 / len(effects)
            credits = (equal_credit,) * len(effects)
        return tuple(
            RecoveryReaderContribution(
                reader_kind=reader_kind,
                strategy_rollout_id=approval.rollout_id,
                memory_id=approval.memory_id,
                effect_delta_l2=effect,
                credit=credit,
                replay_epochs=int(epochs),
                replay_learning_rate=float(learning_rate),
            )
            for approval, effect, credit in zip(approvals, effects, credits, strict=True)
        )

    def consolidate_recovery_memory(
        self,
        *,
        epochs: int = 300,
        semantic_learning_rate: float = 0.1,
        procedural_learning_rate: float = 0.1,
    ) -> dict[str, float]:
        """Consolidate only evidence-approved recovery records into long-term memory."""

        if self._episodic_memory is None:
            raise RuntimeError("recovery consolidation requires episodic records")
        approved_ids = set(self._recovery_strategy_ledger.selected_memory_ids)
        records = tuple(
            record for record in self._episodic_memory.records if record.memory_id in approved_ids
        )
        if not records:
            raise RuntimeError("recovery consolidation has no evidence-approved records")
        self._recovery_memory_epochs = int(epochs)
        self._recovery_semantic_learning_rate = float(semantic_learning_rate)
        self._recovery_procedural_learning_rate = float(procedural_learning_rate)
        losses: dict[str, float] = {}
        selected_approvals = self._recovery_strategy_ledger.selected_approvals
        semantic_baseline = (
            None if self._semantic_memory is None else self._semantic_memory.checkpoint()
        )
        procedural_baseline = (
            None if self._procedural_memory is None else self._procedural_memory.checkpoint()
        )
        sequence_baseline = (
            None
            if self._procedural_sequence_memory is None
            else self._procedural_sequence_memory.checkpoint()
        )
        concept_baseline = self._concept_formation.checkpoint()
        if self._semantic_memory is not None:
            losses["semantic"] = self._semantic_memory.consolidate(
                records,
                epochs=epochs,
                learning_rate=semantic_learning_rate,
            )
            semantic_final = self._semantic_memory.checkpoint()
            semantic_contributions = self._recovery_reader_contributions(
                "semantic",
                semantic_baseline or {},
                semantic_final,
                records,
                selected_approvals,
                epochs=epochs,
                learning_rate=semantic_learning_rate,
            )
            self._recovery_reader_dependencies = self._recovery_reader_dependencies.bind(
                "semantic",
                selected_approvals,
                contributions=semantic_contributions,
                base_checkpoint=semantic_baseline,
                base_checkpoint_digest=_checkpoint_digest(semantic_baseline or {}),
            )
        if self._procedural_memory is not None:
            losses["procedural"] = self._procedural_memory.consolidate(
                records,
                epochs=epochs,
                learning_rate=procedural_learning_rate,
            )
            procedural_final = self._procedural_memory.checkpoint()
            procedural_contributions = self._recovery_reader_contributions(
                "procedural",
                procedural_baseline or {},
                procedural_final,
                records,
                selected_approvals,
                epochs=epochs,
                learning_rate=procedural_learning_rate,
            )
            self._recovery_reader_dependencies = self._recovery_reader_dependencies.bind(
                "procedural",
                selected_approvals,
                contributions=procedural_contributions,
                base_checkpoint=procedural_baseline,
                base_checkpoint_digest=_checkpoint_digest(procedural_baseline or {}),
            )
        if self._procedural_sequence_memory is not None:
            losses["sequence"] = self._procedural_sequence_memory.consolidate(
                records,
                epochs=epochs,
                learning_rate=procedural_learning_rate,
            )
            sequence_final = self._procedural_sequence_memory.checkpoint()
            sequence_contributions = self._recovery_reader_contributions(
                "sequence",
                sequence_baseline or {},
                sequence_final,
                records,
                selected_approvals,
                epochs=epochs,
                learning_rate=procedural_learning_rate,
            )
            self._recovery_reader_dependencies = self._recovery_reader_dependencies.bind(
                "sequence",
                selected_approvals,
                contributions=sequence_contributions,
                base_checkpoint=sequence_baseline,
                base_checkpoint_digest=_checkpoint_digest(sequence_baseline or {}),
            )
        concepts = self._concept_formation.consolidate(records, tick=self.tick)
        self._cognitive_state = replace(self._cognitive_state, concepts=concepts)
        concept_final = self._concept_formation.checkpoint()
        concept_contributions = self._recovery_reader_contributions(
            "concept",
            concept_baseline,
            concept_final,
            records,
            selected_approvals,
            epochs=epochs,
            learning_rate=procedural_learning_rate,
        )
        self._recovery_reader_dependencies = self._recovery_reader_dependencies.bind(
            "concept",
            selected_approvals,
            contributions=concept_contributions,
            base_checkpoint=concept_baseline,
            base_checkpoint_digest=_checkpoint_digest(concept_baseline),
        )
        if not losses:
            raise RuntimeError(
                "recovery consolidation requires semantic, procedural, sequence, or concept memory"
            )
        return losses

    def _rebuild_recovery_memory(self, *, revoked_rollout_id: str | None = None) -> None:
        """Rebuild only readers that depended on the revoked recovery strategy."""

        if self._episodic_memory is None:
            return
        if self._recovery_reader_dependencies.dependencies:
            affected_readers = set(
                self._recovery_reader_dependencies.reader_kinds_for_rollout(
                    "" if revoked_rollout_id is None else revoked_rollout_id
                )
            )
        else:
            affected_readers = {"semantic", "procedural", "sequence", "concept"}
        self._recovery_reader_dependencies = self._recovery_reader_dependencies.retain_selected(
            self._recovery_strategy_ledger.selected_approvals
        )
        approved_memory_ids = set(self._recovery_strategy_ledger.approved_memory_ids)
        selected_memory_ids = set(self._recovery_strategy_ledger.selected_memory_ids)
        records = tuple(
            record
            for record in self._episodic_memory.records
            if record.memory_id not in approved_memory_ids
            or record.memory_id in selected_memory_ids
        )
        recovery_records = tuple(
            record
            for record in self._episodic_memory.records
            if record.memory_id in selected_memory_ids
        )
        selected_approvals = self._recovery_strategy_ledger.selected_approvals

        def dependency_for(reader_kind: str) -> RecoveryReaderDependency | None:
            return self._recovery_reader_dependencies.dependency_for(reader_kind)

        if self._semantic_memory is not None and "semantic" in affected_readers:
            dependency = dependency_for("semantic")
            if dependency is not None and dependency.base_checkpoint is not None:
                semantic = SemanticMemoryLearner.from_checkpoint(
                    dict(dependency.base_checkpoint), device=self.device
                )
                semantic_records = tuple(
                    record for record in recovery_records if record.outcome is not None
                )
            else:
                semantic = SemanticMemoryLearner(self._semantic_memory.cue_dim).to(self.device)
                semantic_records = tuple(record for record in records if record.outcome is not None)
            if semantic_records:
                semantic.consolidate(
                    semantic_records,
                    epochs=self._recovery_memory_epochs,
                    learning_rate=self._recovery_semantic_learning_rate,
                )
            self._semantic_memory = semantic
        if self._procedural_memory is not None and "procedural" in affected_readers:
            dependency = dependency_for("procedural")
            if dependency is not None and dependency.base_checkpoint is not None:
                procedural = ProceduralMemoryLearner.from_checkpoint(
                    dict(dependency.base_checkpoint), device=self.device
                )
                procedural_records = tuple(
                    record for record in recovery_records if record.action_intent is not None
                )
            else:
                procedural = ProceduralMemoryLearner(self._procedural_memory.cue_dim).to(
                    self.device
                )
                procedural_records = tuple(
                    record for record in records if record.action_intent is not None
                )
            if procedural_records:
                procedural.consolidate(
                    procedural_records,
                    epochs=self._recovery_memory_epochs,
                    learning_rate=self._recovery_procedural_learning_rate,
                    action_kinds=procedural.action_kinds or None,
                )
            self._procedural_memory = procedural
        if self._procedural_sequence_memory is not None and "sequence" in affected_readers:
            dependency = dependency_for("sequence")
            if dependency is not None and dependency.base_checkpoint is not None:
                sequence = ProceduralSequenceLearner.from_checkpoint(
                    dict(dependency.base_checkpoint), device=self.device
                )
                sequence_records = tuple(
                    record for record in recovery_records if record.action_intent is not None
                )
            else:
                previous_sequence = self._procedural_sequence_memory
                sequence = ProceduralSequenceLearner(
                    previous_sequence.cue_dim,
                    hidden_dim=previous_sequence.hidden_dim,
                    seed=previous_sequence.seed,
                ).to(self.device)
                sequence_records = tuple(
                    record for record in records if record.action_intent is not None
                )
            if sequence_records:
                sequence.consolidate(
                    sequence_records,
                    epochs=self._recovery_memory_epochs,
                    learning_rate=self._recovery_procedural_learning_rate,
                    action_kinds=sequence.action_kinds or None,
                )
            self._procedural_sequence_memory = sequence
        if "concept" in affected_readers:
            dependency = dependency_for("concept")
            if dependency is not None and dependency.base_checkpoint is not None:
                concepts = ConceptFormationOrgan.from_checkpoint(
                    dict(dependency.base_checkpoint), device=self.device
                )
                concept_records = tuple(
                    record
                    for record in recovery_records
                    if record.outcome is not None and record.event_ids
                )
            else:
                previous_concepts = self._concept_formation
                concepts = ConceptFormationOrgan(
                    similarity_threshold=previous_concepts.similarity_threshold,
                    signal_weights=(
                        float(previous_concepts.signal_weights[0]),
                        float(previous_concepts.signal_weights[1]),
                        float(previous_concepts.signal_weights[2]),
                    ),
                    capacity=previous_concepts.capacity,
                    plasticity_rate=previous_concepts.plasticity_rate,
                    prune_threshold=previous_concepts.prune_threshold,
                    credit_discount=previous_concepts.credit_discount,
                    trace_capacity=previous_concepts.trace_capacity,
                )
                concept_records = tuple(
                    record for record in records if record.outcome is not None and record.event_ids
                )
            if concept_records:
                concepts.consolidate(concept_records, tick=self.tick)
            self._concept_formation = concepts
            self._cognitive_state = replace(
                self._cognitive_state,
                concepts=concepts.concepts,
            )
        for reader_kind in affected_readers:
            dependency = dependency_for(reader_kind)
            if dependency is None or dependency.base_checkpoint is None:
                continue
            if reader_kind == "semantic" and self._semantic_memory is not None:
                final_payload = self._semantic_memory.checkpoint()
                learning_rate = self._recovery_semantic_learning_rate
            elif reader_kind == "procedural" and self._procedural_memory is not None:
                final_payload = self._procedural_memory.checkpoint()
                learning_rate = self._recovery_procedural_learning_rate
            elif reader_kind == "sequence" and self._procedural_sequence_memory is not None:
                final_payload = self._procedural_sequence_memory.checkpoint()
                learning_rate = self._recovery_procedural_learning_rate
            elif reader_kind == "concept":
                final_payload = self._concept_formation.checkpoint()
                learning_rate = self._recovery_procedural_learning_rate
            else:
                continue
            contributions = self._recovery_reader_contributions(
                reader_kind,
                dict(dependency.base_checkpoint),
                final_payload,
                recovery_records,
                selected_approvals,
                epochs=self._recovery_memory_epochs,
                learning_rate=learning_rate,
            )
            self._recovery_reader_dependencies = self._recovery_reader_dependencies.bind(
                reader_kind,
                selected_approvals,
                contributions=contributions,
                base_checkpoint=dependency.base_checkpoint,
                base_checkpoint_digest=dependency.base_checkpoint_digest,
            )
        self._recovery_memory_rebuild_count += 1

    @property
    def recovery_memory_rebuild_count(self) -> int:
        """Return the number of provenance-based memory rebuilds after revocation."""

        return self._recovery_memory_rebuild_count

    def _record_environment_capability(self, result: EnvironmentOutcome) -> None:
        actions = tuple(int(action) for action in result.available_actions)
        kinds = tuple(str(kind) for kind in result.action_kinds)
        if bool(actions) != bool(kinds):
            raise ValueError("environment capabilities must provide actions and kinds together")
        if not actions:
            return
        if any(action >= self.config.alphabet_size for action in actions):
            raise ValueError("environment capability action is outside the motor alphabet")
        self._cognitive_state = replace(
            self._cognitive_state,
            environment_capability=EnvironmentCapability(
                actions=actions,
                action_kinds=kinds,
                tick=self._cognitive_state.world.tick,
            ),
        )

    def reset_dynamics(self, *, episode_id: str | None = None) -> None:
        super().reset_dynamics(episode_id=episode_id)
        self.perception.reset_dynamics()
        self._cognitive_state = self._empty_cognitive_state(self._state.episode_id)
        self._planned_rollout = None
        self._recovery_archive = RecoveryPortfolioArchive(
            capacity=self.config.recovery_archive_capacity
        )
        self._recovery_strategy_ledger = RecoveryStrategyLedger(
            evidence_threshold=self.config.recovery_strategy_evidence_threshold,
            memory_budget=self.config.recovery_strategy_memory_budget,
            evidence_weight=self.config.recovery_strategy_evidence_weight,
            consistency_weight=self.config.recovery_strategy_consistency_weight,
            resource_weight=self.config.recovery_strategy_resource_weight,
        )
        self._recovery_reader_dependencies = RecoveryReaderDependencyGraph()
        self._recovery_generation = 0
        self._replan_required = False
        self._last_rollout_prediction_error = None
        self._last_rollout_calibrated_confidence = None
        self._last_executive_decision = None
        self._last_executive_prediction_error = None
        self._last_affordance_prediction_error = None
        self._last_executive_world_action = None
        self._last_generation_trace = None
        self._last_language_emission = None
        self._language_fallback_count = 0
        self._language_fallback_requires_replan = False
        self._last_content_selection = None
        self._last_content_prediction_error = None
        self._content_feedback_applied = False
        self._recovery_portfolio = None

    def _lineage_limit(self) -> int:
        return int(self.config.cognitive_lineage_history_limit)

    @staticmethod
    def _bounded_ids(existing: Sequence[str], *new_ids: str, limit: int) -> tuple[str, ...]:
        values = [str(item) for item in existing]
        for item in new_ids:
            if item and item not in values:
                values.append(str(item))
        return tuple(values[-int(limit) :])

    @staticmethod
    def _world_object_ids(world: WorldState) -> tuple[str, ...]:
        values = [str(item) for item in world.entities]
        values.extend(item.object_id for item in world.objects)
        return tuple(dict.fromkeys(values))

    @staticmethod
    def _world_relation_ids(world: WorldState) -> tuple[str, ...]:
        return tuple(
            f"{subject}:{predicate}:{object_id}"
            for subject, predicate, object_id in world.relations
        )

    def _record_percept_lineage(
        self, percept: PerceptEvent, world: WorldState
    ) -> tuple[tuple[Assembly, ...], tuple[Event, ...]]:
        """Materialize percept output as traceable assembly/event state."""

        previous = self._cognitive_state
        event_id = f"{self._state.episode_id}:event:{self.tick}"
        parent_event_ids = () if not previous.events else (previous.events[-1].event_id,)
        event = Event(
            event_id=event_id,
            start_tick=max(0, int(self.tick) - int(percept.duration) + 1),
            end_tick=int(self.tick),
            latent=percept.features.detach().clone(),
            assembly_ids=(percept.assembly_id,),
            parent_event_ids=parent_event_ids,
            object_ids=self._world_object_ids(world),
            relation_ids=self._world_relation_ids(world),
            prediction_error=percept.prediction_error,
            confidence=percept.confidence,
            provenance="perceptual",
        )
        existing = next(
            (item for item in previous.assemblies if item.assembly_id == percept.assembly_id),
            None,
        )
        coherence = 1.0
        if existing is not None and existing.activity.numel() and percept.features.numel():
            coherence = float(
                0.5
                * (
                    1.0
                    + torch.nn.functional.cosine_similarity(
                        existing.activity.unsqueeze(0), percept.features.unsqueeze(0)
                    ).item()
                )
            )
        assembly = Assembly(
            assembly_id=percept.assembly_id,
            start_tick=(
                int(percept.observation_tick) - int(percept.duration) + 1
                if existing is None
                else min(
                    existing.start_tick, int(percept.observation_tick) - int(percept.duration) + 1
                )
            ),
            end_tick=int(percept.observation_tick),
            activity=percept.features.detach().clone(),
            source_event_ids=(
                (event_id,)
                if existing is None
                else self._bounded_ids(
                    existing.source_event_ids,
                    event_id,
                    limit=self._lineage_limit(),
                )
            ),
            coherence=max(0.0, min(1.0, coherence)),
            prediction_error=percept.prediction_error,
            route_score=percept.confidence,
            provenance="perception",
            confidence=percept.confidence,
        )
        assemblies = [
            item for item in previous.assemblies if item.assembly_id != assembly.assembly_id
        ]
        assemblies.append(assembly)
        events = (*previous.events, event)
        limit = self._lineage_limit()
        return tuple(assemblies[-limit:]), tuple(events[-limit:])

    def _record_outcome_self_and_development(
        self,
        outcome: Outcome,
        *,
        prediction_error: float = 0.0,
    ) -> tuple[SelfState, DevelopmentState]:
        """Update self/development evidence only from a real outcome."""

        previous_self = self._cognitive_state.self_state
        previous_development = self._cognitive_state.development
        capability = dict(previous_self.capability_confidence)
        if self._cognitive_state.action_intent is not None:
            capability_key = self._cognitive_state.action_intent.kind
            old_confidence = float(capability.get(capability_key, 0.0))
            target = 1.0 if bool(outcome.success) else 0.0
            update_rate = float(self.config.self_capability_learning_rate)
            capability[capability_key] = old_confidence + update_rate * (target - old_confidence)
        confidence = (
            max(0.0, min(1.0, sum(capability.values()) / len(capability)))
            if capability
            else previous_self.confidence
        )
        self_state = replace(
            previous_self,
            tick=self.tick,
            confidence=confidence,
            capability_confidence=tuple(capability.items()),
            last_outcome_id=outcome.intent_id,
            last_update_source=outcome.provenance,
            last_prediction_error=max(0.0, float(prediction_error)),
            update_count=previous_self.update_count + 1,
            lineage=self._bounded_ids(
                previous_self.lineage,
                outcome.intent_id,
                limit=self._lineage_limit(),
            ),
        )
        validation_status = previous_development.last_validation_status
        if previous_development.proposal_ids:
            validation_status = "accepted" if bool(outcome.success) else "rejected"
        development = replace(
            previous_development,
            tick=self.tick,
            last_update_source=outcome.provenance,
            last_validation_status=validation_status,
            validation_evidence_ids=self._bounded_ids(
                previous_development.validation_evidence_ids,
                outcome.intent_id,
                limit=self._lineage_limit(),
            ),
            lineage=self._bounded_ids(
                previous_development.lineage,
                outcome.intent_id,
                limit=self._lineage_limit(),
            ),
        )
        return self_state, development

    def _consolidate_concepts(self) -> tuple[Concept, ...]:
        """Keep legacy callers pointed at the Taiji-owned concept organ."""

        if self._episodic_memory is None:
            return self._concept_formation.concepts
        return self._concept_formation.consolidate(
            self._episodic_memory.records,
            tick=self.tick,
        )

    def _refresh_last_event_world_lineage(self, world: WorldState) -> None:
        """Attach an externally supplied world observation to the current event."""

        if not self._cognitive_state.events:
            return
        events = self._cognitive_state.events
        event = replace(
            events[-1],
            object_ids=self._world_object_ids(world),
            relation_ids=self._world_relation_ids(world),
        )
        self._cognitive_state = replace(self._cognitive_state, events=(*events[:-1], event))

    def _concept_matches_for_world(self, world: WorldState) -> tuple[Any, ...]:
        cues = []
        if world.latent.numel():
            cues.append(world.latent)
        if self._cognitive_state.percept is not None:
            cues.append(self._cognitive_state.percept.features)
        matches: tuple[ConceptMatch, ...] = ()
        for cue in cues:
            matches = self._concept_formation.retrieve(
                cue,
                object_ids=self._world_object_ids(world),
                relation_ids=self._world_relation_ids(world),
                limit=self.config.concept_capacity,
            )
            if matches:
                break
        if matches or not self._cognitive_state.memory.concept_ids:
            return matches
        active_ids = set(self._cognitive_state.memory.concept_ids)
        active_score = self._cognitive_state.memory.concept_confidence
        return tuple(
            ConceptMatch(concept=concept, score=active_score)
            for concept in self._concept_formation.concepts
            if concept.concept_id in active_ids
        )

    def _refresh_concept_memory(self) -> None:
        matches = self._concept_matches_for_world(self._cognitive_state.world)
        self._cognitive_state = replace(
            self._cognitive_state,
            memory=replace(
                self._cognitive_state.memory,
                concept_ids=tuple(match.concept.concept_id for match in matches),
                concept_confidence=(matches[0].score if matches else 0.0),
            ),
        )

    def _recovery_memory_is_readable(self, memory_id: str) -> bool:
        """Hide revoked or budget-unselected recovery records from recall."""

        approved = set(self._recovery_strategy_ledger.approved_memory_ids)
        if (
            memory_id not in approved
            and memory_id not in self._recovery_strategy_ledger.revoked_memory_ids
        ):
            return True
        return memory_id in self._recovery_strategy_ledger.selected_memory_ids

    def observe(self, symbol: int, *args: Any, **kwargs: Any) -> TaijiStep:
        workspace_candidates: Sequence[WorkspaceCandidate] | None = kwargs.pop(
            "workspace_candidates", None
        )
        workspace_mode = str(kwargs.pop("workspace_mode", "learned"))
        step = super().observe(symbol, *args, **kwargs)
        observation = Observation(
            modality="text-byte",
            value=int(symbol),
            timestamp=self.tick,
            source="byte-sensor",
            provenance="external",
        )
        percept = self.perception.observe(
            int(symbol),
            tick=self.tick,
            stream_id=self._state.episode_id,
            learn=bool(kwargs.get("learn", True)),
        )
        features = percept.features.detach().clone()
        recall = step.memory_recall
        previous = self._cognitive_state
        concept_matches = self._concept_formation.retrieve(
            features,
            limit=self.config.concept_capacity,
        )
        if not concept_matches and previous.memory.concept_ids:
            concept_matches = self._concept_matches_for_world(previous.world)
        homeostasis = previous.homeostasis
        if self._homeostatic_controller is not None:
            homeostasis = self._homeostatic_controller.update(
                previous.homeostasis,
                prediction_error=percept.prediction_error,
                novelty=max(percept.prediction_error, 1.0 - recall.confidence),
                resource_cost=0.05,
                mode="wake",
            )
        selection: WorkspaceSelection | None = None
        candidates: tuple[WorkspaceCandidate, ...] = ()
        if self._workspace_router is not None and (
            workspace_candidates is not None or workspace_mode != "learned"
        ):
            candidates = tuple(workspace_candidates or ())
            selection = self._workspace_router.route(
                candidates,
                tick=self.tick,
                mode=workspace_mode,
            )
            workspace = WorkspaceState(
                tick=self.tick,
                focus=selection.selected_ids,
                broadcast=selection.broadcast,
                capacity=selection.capacity,
                candidates=candidates,
                selection=selection,
            )
        else:
            workspace = WorkspaceState(
                tick=self.tick,
                focus=("predictive-context",),
                broadcast=features,
                capacity=1,
            )
        world = WorldState(
            tick=self.tick,
            latent=features,
            entities=previous.world.entities,
            relations=previous.world.relations,
            objects=previous.world.objects,
            events=previous.world.events,
            affordances=previous.world.affordances,
            uncertainty=max(0.0, min(1.0, 1.0 - recall.confidence)),
        )
        assemblies, events = self._record_percept_lineage(percept, world)
        lineage_event = events[-1]
        workspace = replace(
            workspace,
            percept_event_id=lineage_event.event_id,
            percept_assembly_id=percept.assembly_id,
            percept_boundary_closed=percept.boundary,
        )
        world = replace(
            world,
            percept_event_id=lineage_event.event_id,
            percept_assembly_id=percept.assembly_id,
            percept_boundary_closed=percept.boundary,
        )
        memory = NativeMemoryState(
            tick=self.tick,
            episodic_confidence=max(0.0, min(1.0, recall.confidence)),
            semantic_context=recall.cortical_feedback.detach().clone(),
            procedural_context=recall.action_evidence.detach().clone(),
            working_ids=(f"{self._state.episode_id}:working:{self.tick}",),
            working_items=(
                WorkingMemoryItem(
                    item_id=f"{self._state.episode_id}:working:{self.tick}",
                    value=features,
                    salience=max(0.0, min(1.0, recall.confidence)),
                ),
            ),
            episodic_ids=(
                ()
                if self._episodic_memory is None
                else tuple(
                    hit.record.memory_id
                    for hit in self._episodic_memory.retrieve(features, limit=3)
                    if self._recovery_memory_is_readable(hit.record.memory_id)
                )
            ),
            concept_ids=tuple(match.concept.concept_id for match in concept_matches),
            concept_confidence=(concept_matches[0].score if concept_matches else 0.0),
        )
        self._cognitive_state = replace(
            previous,
            tick=self.tick,
            episode_id=self._state.episode_id,
            observation=observation,
            percept=percept,
            workspace=workspace,
            world=world,
            memory=memory,
            assemblies=assemblies,
            events=events,
            goals=replace(previous.goals, tick=self.tick),
            plan=replace(previous.plan, tick=self.tick),
            self_state=replace(
                previous.self_state,
                tick=self.tick,
                confidence=max(0.0, min(1.0, recall.confidence)),
            ),
            homeostasis=replace(homeostasis, tick=self.tick),
            development=replace(previous.development, tick=self.tick),
            learning=replace(
                previous.learning,
                tick=self.tick,
                local_updates=previous.learning.local_updates + int(kwargs.get("learn", True)),
            ),
            action_intent=None,
            outcome=None,
            world_transition=None,
            world_prediction=None,
        )
        return step

    def observe_event(
        self,
        observation: Observation,
        *,
        learn: bool = True,
        learn_motor: bool | None = None,
        use_memory: bool = True,
        world_state: WorldState | None = None,
        workspace_candidates: Sequence[WorkspaceCandidate] | None = None,
        workspace_mode: str = "learned",
    ) -> TaijiStep:
        """Ingest a v1 ``Observation`` while retaining the kernel API."""

        if observation.modality != "text-byte" or not isinstance(observation.value, int):
            raise ValueError("TSK-v8 adapter accepts only integer text-byte observations")
        step = self.observe(
            observation.value,
            learn=learn,
            learn_motor=learn_motor,
            use_memory=use_memory,
            workspace_candidates=workspace_candidates,
            workspace_mode=workspace_mode,
        )
        self._cognitive_state = replace(self._cognitive_state, observation=observation)
        if world_state is not None:
            if not isinstance(world_state, WorldState):
                raise TypeError("world_state must be a Taiji WorldState")
            if world_state.tick != self.tick:
                raise ValueError("observed world_state must match the adapter tick")
            current_world = self._cognitive_state.world
            self._cognitive_state = replace(
                self._cognitive_state,
                world=self._ground_world_state(
                    replace(
                        world_state,
                        percept_event_id=current_world.percept_event_id,
                        percept_assembly_id=current_world.percept_assembly_id,
                        percept_boundary_closed=current_world.percept_boundary_closed,
                    )
                ),
            )
            self._refresh_last_event_world_lineage(self._cognitive_state.world)
            self._refresh_concept_memory()
        return step

    def ingest_input(
        self,
        frame: InputFrame,
        *,
        learn: bool = True,
        learn_motor: bool | None = None,
        use_memory: bool = True,
        workspace_candidates: Sequence[WorkspaceCandidate] | None = None,
        workspace_mode: str = "learned",
    ) -> InputTrace:
        """Route one client frame through Taiji-owned perception.

        The adapter currently exposes byte-level text perception.  The frame
        keeps the product transport metadata intact, while each byte becomes
        a versioned ``Observation`` and learned ``PerceptEvent``.  No action
        intent is inferred at this boundary; an executive must earn that
        decision from the resulting cognitive state.
        """

        if not isinstance(frame, InputFrame):
            raise TypeError("frame must be a Taiji InputFrame")
        if frame.modality not in self.SUPPORTED_INPUT_MODALITIES:
            supported = ", ".join(sorted(self.SUPPORTED_INPUT_MODALITIES))
            raise ValueError(
                f"unsupported input modality {frame.modality!r}; supported: {supported}"
            )

        observations: list[Observation] = []
        percepts: list[Any] = []
        for symbol in frame.payload:
            observation = Observation(
                modality="text-byte",
                value=int(symbol),
                timestamp=frame.timestamp,
                source=frame.source,
                provenance=frame.provenance,
                confidence=frame.confidence,
            )
            self.observe_event(
                observation,
                learn=learn,
                learn_motor=learn_motor,
                use_memory=use_memory,
                workspace_candidates=workspace_candidates,
                workspace_mode=workspace_mode,
            )
            current = self.cognitive_snapshot()
            if current.observation is None or current.percept is None:
                raise RuntimeError("Taiji perception did not emit a complete input trace")
            observations.append(current.observation)
            percepts.append(current.percept)

        current = self.cognitive_snapshot()
        return InputTrace(
            input_id=frame.input_id,
            modality=frame.modality,
            observations=tuple(observations),
            percepts=tuple(percepts),
            action_intent=current.action_intent,
        )

    @torch.no_grad()
    def generate_input(
        self,
        frame: InputFrame,
        length: int,
        *,
        stop_at_boundary: bool = False,
        sample: bool = False,
        reset: bool = True,
    ) -> bytes:
        """Generate from a validated Taiji input frame.

        Generation remains a byte-level effector path for compatibility.  It
        does not manufacture an ``ActionIntent`` or semantic ``ContentPlan``
        from the client text.
        """

        if not isinstance(frame, InputFrame):
            raise TypeError("frame must be a Taiji InputFrame")
        if frame.modality not in self.SUPPORTED_INPUT_MODALITIES:
            supported = ", ".join(sorted(self.SUPPORTED_INPUT_MODALITIES))
            raise ValueError(
                f"unsupported input modality {frame.modality!r}; supported: {supported}"
            )
        return self.generate(
            frame.payload,
            length,
            stop_at_boundary=stop_at_boundary,
            sample=sample,
            reset=reset,
        )

    def act(self, available_actions: Any, *args: Any, **kwargs: Any) -> TaijiDecision:
        supplied_world_action = kwargs.pop("world_action", None)
        procedural_action_kinds = kwargs.pop("procedural_action_kinds", None)
        use_procedural = bool(kwargs.pop("use_procedural", False))
        use_plan = bool(kwargs.pop("use_plan", False))
        planned_kind = None
        if use_plan:
            if use_procedural:
                raise ValueError("use_plan and use_procedural cannot be enabled together")
            selected_plan_id = self._cognitive_state.plan.selected_plan_id
            if selected_plan_id is None:
                raise RuntimeError("planned action routing requires a selected plan")
            selected_plan = next(
                (
                    candidate
                    for candidate in self._cognitive_state.plan.candidates
                    if candidate.plan_id == selected_plan_id
                ),
                None,
            )
            if selected_plan is None:
                raise RuntimeError("selected plan is missing from the current plan state")
            planned_kind = selected_plan.action_kind
        action_kinds = (
            None
            if procedural_action_kinds is None
            else tuple(str(kind) for kind in procedural_action_kinds)
        )
        if action_kinds is not None:
            action_symbols = tuple(int(action) for action in available_actions)
            if len(action_kinds) != len(action_symbols):
                raise ValueError("procedural_action_kinds must align with available_actions")
            if len(set(action_kinds)) != len(action_kinds):
                raise ValueError("procedural_action_kinds must be unique")
        if use_procedural:
            if self._procedural_memory is None:
                raise RuntimeError("procedural action routing requires an attached learner")
            if action_kinds is None:
                raise ValueError("procedural action routing requires procedural_action_kinds")
            if self._cognitive_state.percept is None:
                raise RuntimeError("procedural action routing requires a current perception")
            predicted_kind = self._procedural_memory.predict(self._cognitive_state.percept.features)
            if predicted_kind not in action_kinds:
                raise ValueError("procedural learner predicted an unavailable action kind")
        else:
            predicted_kind = None
        if planned_kind is not None:
            if action_kinds is None:
                raise ValueError("planned action routing requires procedural_action_kinds")
            if planned_kind not in action_kinds:
                raise ValueError("selected plan is not available in the current affordances")
        decision = super().act(available_actions, *args, **kwargs)
        route_kind = predicted_kind if use_procedural else planned_kind
        if route_kind is not None:
            assert action_kinds is not None
            selected_action = tuple(action_kinds).index(route_kind)
            selected_symbol = tuple(int(action) for action in available_actions)[selected_action]
            pending = self._state.pending_action
            if pending is None:
                raise RuntimeError("kernel did not preserve a pending action")
            self._state.pending_action = replace(pending, action_symbol=selected_symbol)
            decision = replace(decision, action_symbol=selected_symbol)
        intent_parameters = {
            **({} if supplied_world_action is None else dict(supplied_world_action.parameters)),
            "action_symbol": decision.action_symbol,
            "available_actions": decision.available_actions,
        }
        intent = ActionIntent(
            intent_id=f"{self._state.episode_id}:intent:{decision.tick}",
            kind=(
                "byte-motor"
                if action_kinds is None
                else action_kinds[decision.available_actions.index(decision.action_symbol)]
            ),
            parameters=intent_parameters,
            expected_outcome="environment-feedback",
            confidence=float(decision.policy_probabilities[decision.action_symbol]),
            tick=decision.tick,
        )
        candidates = tuple(
            PlanCandidate(
                plan_id=f"{intent.intent_id}:candidate:{index}",
                action_kind=("byte-motor" if action_kinds is None else action_kinds[index]),
                expected_value=float(decision.policy_probabilities[action]),
            )
            for index, action in enumerate(decision.available_actions)
        )
        selected_index = decision.available_actions.index(decision.action_symbol)
        plan_state = (
            replace(self._cognitive_state.plan, tick=self.tick)
            if use_plan
            else PlanState(
                tick=self.tick,
                candidates=candidates,
                selected_plan_id=(candidates[selected_index].plan_id if candidates else None),
            )
        )
        world_action = None
        if supplied_world_action is not None:
            if not isinstance(supplied_world_action, WorldAction):
                raise TypeError("world_action must be a Taiji WorldAction")
            if supplied_world_action.tick != self._cognitive_state.world.tick:
                raise ValueError("world_action must act at the current world tick")
            world_action = WorldAction(
                action_id=intent.intent_id,
                kind=supplied_world_action.kind,
                tick=supplied_world_action.tick,
                actor_id=supplied_world_action.actor_id,
                target_id=supplied_world_action.target_id,
                parameters=supplied_world_action.parameters,
                provenance=supplied_world_action.provenance,
            )
        elif self._world_dynamics is not None:
            world_action = WorldAction(
                action_id=intent.intent_id,
                kind=intent.kind,
                tick=self._cognitive_state.world.tick,
                parameters=tuple(sorted(intent.parameters.items())),
            )
        world_prediction = None
        if self._world_dynamics is not None and world_action is not None:
            prediction = self._world_dynamics.predict(
                self._cognitive_state.world,
                world_action,
                register_parameters=False,
            )
            world_prediction = WorldPredictionRecord(
                action=world_action,
                predicted_state=prediction.state,
                predicted_reward=prediction.reward,
                predicted_success_probability=prediction.success_probability,
                uncertainty=prediction.uncertainty,
                uncertainty_mode=prediction.uncertainty_mode,
                online_update_count=self._world_dynamics.online_updates,
            )
        self._cognitive_state = replace(
            self._cognitive_state,
            plan=plan_state,
            action_intent=intent,
            world_prediction=world_prediction,
        )
        return decision

    def settle_action(self, reward: float, *args: Any, **kwargs: Any) -> TaijiOutcome:
        intent = self._cognitive_state.action_intent
        world_state = kwargs.pop("world_state", None)
        world_action = kwargs.pop("world_action", None)
        success = kwargs.pop("success", None)
        terminal = bool(kwargs.pop("terminal", False))
        sequence_boundary = bool(kwargs.pop("sequence_boundary", terminal))
        learn_world = kwargs.pop("learn_world", None)
        world_learning_rate = float(kwargs.pop("world_learning_rate", 0.005))
        world_learning_repeats = int(kwargs.pop("world_learning_repeats", 1))
        intent_id = intent.intent_id if intent is not None else f"kernel-action:{self.tick}"
        before = self._cognitive_state.world
        planned_rollout = self._planned_rollout
        planned_recovery_resource = (
            0.0
            if self._cognitive_state.recovery_branch is None or planned_rollout is None
            else float(planned_rollout.steps[0].resource_cost)
        )
        planned_recovery_action_id = (
            None
            if self._cognitive_state.recovery_branch is None or planned_rollout is None
            else planned_rollout.steps[0].action.action_id
        )
        recovery_budget = self._cognitive_state.recovery_budget
        planned_recovery_resource_delta = planned_recovery_resource
        if (
            planned_recovery_action_id is not None
            and recovery_budget is not None
            and planned_recovery_action_id in recovery_budget.consumed_action_ids
        ):
            planned_recovery_resource_delta = 0.0
        if world_state is not None:
            if not isinstance(world_state, WorldState):
                raise TypeError("world_state must be a Taiji WorldState")
            if world_state.tick != before.tick + 1:
                raise ValueError("world_state must advance the cognitive world tick by one")
            world_state = self._ground_world_state(world_state)
            world_state = replace(
                world_state,
                percept_event_id=before.percept_event_id,
                percept_assembly_id=before.percept_assembly_id,
                percept_boundary_closed=before.percept_boundary_closed,
            )
            if world_action is not None:
                if not isinstance(world_action, WorldAction):
                    raise TypeError("world_action must be a Taiji WorldAction")
                if world_action.action_id != intent_id:
                    raise ValueError("world_action must reference the pending ActionIntent")
        result = super().settle_action(reward, *args, **kwargs)
        outcome = Outcome(
            intent_id=intent_id,
            reward=float(result.reward),
            success=(float(result.reward) > 0.0 if success is None else bool(success)),
            terminal=terminal,
            provenance=str(kwargs.get("provenance", "experienced")),
            tick=(self.tick if world_state is None else int(world_state.tick)),
        )
        self._apply_content_feedback(outcome.reward)
        transition = None
        prediction_record = self._cognitive_state.world_prediction
        calibration_trace = self._cognitive_state.world_calibration_trace
        recovery_state = self._cognitive_state.planning_recovery
        recovery_branch = self._cognitive_state.recovery_branch
        world_model_replan = False
        world_outcome_replan = False
        if world_state is not None:
            if world_action is None:
                if intent is None:
                    raise RuntimeError("world_state requires a pending ActionIntent")
                world_action = (
                    prediction_record.action
                    if prediction_record is not None
                    else WorldAction(
                        action_id=intent_id,
                        kind=intent.kind,
                        tick=before.tick,
                        parameters=tuple(sorted(intent.parameters.items())),
                        provenance=str(kwargs.get("provenance", "experienced")),
                    )
                )
            if self._world_dynamics is not None:
                self._world_dynamics.register_open_set(
                    before,
                    world_state,
                    action=world_action,
                    register_parameters=False,
                )
            if prediction_record is None and self._world_dynamics is not None:
                prediction = self._world_dynamics.predict(
                    before,
                    world_action,
                    register_parameters=False,
                )
                prediction_record = WorldPredictionRecord(
                    action=world_action,
                    predicted_state=prediction.state,
                    predicted_reward=prediction.reward,
                    predicted_success_probability=prediction.success_probability,
                    uncertainty=prediction.uncertainty,
                    uncertainty_mode=prediction.uncertainty_mode,
                    online_update_count=self._world_dynamics.online_updates,
                )
            transition = WorldTransition(
                before=before,
                action=world_action,
                after=world_state,
                outcome=outcome,
            )
            if prediction_record is not None and self._world_dynamics is not None:
                online_update_count_before = self._world_dynamics.online_updates
                predicted = self._world_dynamics.schema.state_values(
                    prediction_record.predicted_state
                )
                actual = self._world_dynamics.schema.state_values(world_state)
                raw_state_error = float(torch.mean((predicted - actual) ** 2))
                prediction_record = replace(
                    prediction_record,
                    state_error=self._world_dynamics.schema.normalized_state_error(
                        prediction_record.predicted_state,
                        world_state,
                    ),
                    raw_state_error=raw_state_error,
                    reward_error=(prediction_record.predicted_reward - outcome.reward) ** 2,
                )
                world_error_threshold = (
                    None
                    if self._goal_planner is None
                    else self._goal_planner.world_prediction_error_threshold(
                        recovery=recovery_state is not None,
                        trigger_error=(
                            None if recovery_state is None else recovery_state.prediction_error
                        ),
                    )
                )
                world_model_replan = bool(
                    not terminal
                    and self._goal_planner is not None
                    and prediction_record.state_error is not None
                    and world_error_threshold is not None
                    and prediction_record.state_error > world_error_threshold
                )
                if world_model_replan:
                    assert world_error_threshold is not None
                    assert prediction_record.state_error is not None
                    threshold = float(world_error_threshold)
                    recovery_state = PlanningRecoveryState(
                        mode="world-error-recovery",
                        trigger="world-prediction-error",
                        prediction_error=float(prediction_record.state_error),
                        threshold=float(threshold),
                        source_rollout_id=(
                            None
                            if self._planned_rollout is None
                            else self._planned_rollout.rollout_id
                        ),
                        remaining_rollout_steps=(
                            0
                            if self._planned_rollout is None
                            else max(0, len(self._planned_rollout.steps) - 1)
                        ),
                    )
                if learn_world is None:
                    learn_world = bool(kwargs.get("learn", True))
                update_losses: list[float] = []
                adjudication = "not-applied"
                ledger_uncertainty = 1.0
                ledger_uncertainty_mode = "unseen"
                ledger_evidence_count = 0
                if learn_world:
                    transition_rejections_before = self._world_dynamics.transition_rejections
                    update_losses = self._world_dynamics.online_update(
                        transition,
                        learning_rate=world_learning_rate,
                        repeats=world_learning_repeats,
                        register_parameters=False,
                    )
                    adjudication = (
                        "rejected"
                        if self._world_dynamics.transition_rejections > transition_rejections_before
                        else "accepted"
                    )
                    ledger_key = self._world_dynamics.schema_registry.transition_evidence_key(
                        transition
                    )
                    ledger_uncertainty, ledger_uncertainty_mode = (
                        self._world_dynamics.schema_registry.transition_uncertainty(ledger_key)
                    )
                    ledger_evidence_count = sum(
                        int(item["evidence_count"])
                        for item in self._world_dynamics.schema_registry.transition_hypotheses.get(
                            ledger_key, ()
                        )
                    )
                    world_outcome_replan = bool(
                        not terminal
                        and (
                            adjudication == "rejected"
                            or outcome.success is False
                            or outcome.reward < 0.0
                        )
                    )
                    if world_outcome_replan:
                        recovery_branch = RecoveryBranchState(
                            source_rollout_id=(
                                None
                                if self._planned_rollout is None
                                else self._planned_rollout.rollout_id
                            ),
                            goal_id=(
                                None
                                if self._planned_rollout is None
                                else self._planned_rollout.goal_id
                            ),
                            rejected_action=transition.action,
                            evidence_key=ledger_key,
                            uncertainty_mode=ledger_uncertainty_mode,
                            remaining_rollout_steps=(
                                0
                                if self._planned_rollout is None
                                else max(0, len(self._planned_rollout.steps) - 1)
                            ),
                            reason=(
                                "outcome-adjudication"
                                if adjudication == "rejected"
                                else "environment-failure"
                            ),
                            resource_budget=(
                                None if recovery_branch is None else recovery_branch.resource_budget
                            ),
                            consumed_resource=(
                                0.0
                                if recovery_branch is None
                                else recovery_branch.consumed_resource
                            )
                            + planned_recovery_resource_delta,
                            failure_count=(
                                0 if recovery_branch is None else recovery_branch.failure_count
                            )
                            + int(
                                not terminal and (outcome.success is False or outcome.reward < 0.0)
                            ),
                            rejection_count=(
                                0 if recovery_branch is None else recovery_branch.rejection_count
                            )
                            + int(adjudication == "rejected"),
                        )
                    prediction_record = replace(
                        prediction_record,
                        online_update_count=self._world_dynamics.online_updates,
                    )
                calibration_trace = (
                    *calibration_trace,
                    WorldCalibrationTrace(
                        transition=transition,
                        prediction=prediction_record,
                        calibration_applied=bool(learn_world and update_losses),
                        online_update_count_before=online_update_count_before,
                        online_update_count_after=self._world_dynamics.online_updates,
                        adjudication=adjudication,
                        ledger_uncertainty=ledger_uncertainty,
                        ledger_uncertainty_mode=ledger_uncertainty_mode,
                        ledger_evidence_count=ledger_evidence_count,
                    ),
                )[-self.config.world_calibration_history_limit :]
        if recovery_branch is not None and planned_recovery_action_id is not None:
            if recovery_budget is None:
                recovery_budget = RecoveryBudgetState(
                    total_budget=(
                        1.0
                        if recovery_branch.resource_budget is None
                        else recovery_branch.resource_budget
                    )
                )
            recovery_budget = recovery_budget.consume(
                planned_recovery_action_id,
                planned_recovery_resource,
            )
        if (
            recovery_branch is not None
            and planned_recovery_resource_delta > 0.0
            and not world_outcome_replan
        ):
            recovery_branch = replace(
                recovery_branch,
                consumed_resource=recovery_branch.consumed_resource
                + planned_recovery_resource_delta,
                failure_count=recovery_branch.failure_count
                + int(not terminal and (outcome.success is False or outcome.reward < 0.0)),
            )
        prediction_error = (
            0.0
            if prediction_record is None or prediction_record.state_error is None
            else prediction_record.state_error
        )
        if transition is not None and intent is not None:
            self._concept_formation.update_sequence_trace(
                intent.kind,
                before_state=transition.before,
                after_state=transition.after,
                outcome=outcome,
                prediction_error=prediction_error,
            )
            self._record_online_concept_transition(
                transition,
                prediction_error,
                boundary=sequence_boundary,
            )
        elif sequence_boundary:
            self._online_concept_branches.clear()
        memory = self._cognitive_state.memory
        if self._planned_rollout is not None and self._goal_planner is not None:
            rollout = self._planned_rollout
            planning_error_threshold = self._goal_planner.world_prediction_error_threshold(
                recovery=recovery_state is not None,
                trigger_error=(None if recovery_state is None else recovery_state.prediction_error),
            )
            self._last_rollout_prediction_error = self._goal_planner.rollout_prediction_error(
                rollout, outcome
            )
            self._replan_required = self._language_fallback_requires_replan or (
                self._last_rollout_prediction_error > planning_error_threshold
            )
            self._last_rollout_calibrated_confidence = self._goal_planner.record_rollout_outcome(
                rollout, outcome
            )
            self._planned_rollout = None
        self._replan_required = bool(
            self._replan_required or world_model_replan or world_outcome_replan
        )
        goals = self._cognitive_state.goals
        if self._goal_planner is not None:
            goals = self._goal_planner.apply_outcome(goals, outcome)
        homeostasis = self._cognitive_state.homeostasis
        if self._homeostatic_controller is not None:
            homeostasis = self._homeostatic_controller.update(
                homeostasis,
                prediction_error=(
                    0.0
                    if prediction_record is None or prediction_record.state_error is None
                    else prediction_record.state_error
                ),
                reward=outcome.reward,
                resource_cost=0.10,
                mode="wake",
            )
        if self._episodic_memory is not None:
            cue = (
                self._cognitive_state.percept.features
                if self._cognitive_state.percept is not None
                else self._cognitive_state.world.latent
            )
            record = EpisodicMemoryRecord(
                memory_id=f"{self._state.episode_id}:memory:{self.tick}:{intent_id}",
                episode_id=self._state.episode_id,
                tick=outcome.tick,
                cue=cue.detach().clone(),
                action_intent=intent,
                outcome=outcome,
                world_transition=transition,
                prediction_error=prediction_error,
                provenance=outcome.provenance,
                event_ids=(
                    ()
                    if not self._cognitive_state.events
                    else (self._cognitive_state.events[-1].event_id,)
                ),
                assembly_ids=(
                    ()
                    if not self._cognitive_state.assemblies
                    else (self._cognitive_state.assemblies[-1].assembly_id,)
                ),
                object_ids=(
                    ()
                    if not self._cognitive_state.events
                    else self._cognitive_state.events[-1].object_ids
                ),
                relation_ids=(
                    ()
                    if not self._cognitive_state.events
                    else self._cognitive_state.events[-1].relation_ids
                ),
            )
            self._episodic_memory.write(record)
            memory = replace(
                memory,
                tick=self.tick,
                episodic_confidence=1.0,
                episodic_ids=(record.memory_id,),
            )
        self_state, development = self._record_outcome_self_and_development(
            outcome,
            prediction_error=prediction_error,
        )
        self._cognitive_state = replace(
            self._cognitive_state,
            tick=self.tick,
            world=(world_state if world_state is not None else self._cognitive_state.world),
            concepts=self._concept_formation.concepts,
            memory=memory,
            self_state=self_state,
            development=development,
            goals=goals,
            homeostasis=homeostasis,
            outcome=outcome,
            world_transition=transition,
            world_prediction=prediction_record,
            world_calibration_trace=calibration_trace,
            planning_recovery=recovery_state,
            recovery_branch=recovery_branch,
            recovery_budget=recovery_budget,
            learning=replace(
                self._cognitive_state.learning,
                tick=self.tick,
                lifetime_updates=self._cognitive_state.learning.lifetime_updates
                + int(kwargs.get("learn", True)),
            ),
        )
        if terminal and self._cognitive_state.recovery_branch is not None:
            self._cognitive_state = replace(self._cognitive_state, recovery_branch=None)
        return result

    def parameter_tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            *super().parameter_tensors(),
            *self.perception.parameter_tensors(),
            *(
                tensor
                for region in self._neuron_regions.values()
                for tensor in region.parameter_tensors()
            ),
            *(
                tensor
                for network in self._neuron_networks.values()
                for tensor in network.parameter_tensors()
            ),
            *(() if self._executive is None else self._executive.parameter_tensors()),
        )

    def parameter_count(self, *, active_only: bool = True) -> int:
        del active_only
        return int(
            super().parameter_count()
            + sum(parameter.numel() for parameter in self.perception.parameters())
            + sum(region.edge_count for region in self._neuron_regions.values())
            + sum(network.edge_count for network in self._neuron_networks.values())
            + (
                0
                if self._executive is None
                else sum(parameter.numel() for parameter in self._executive.parameter_tensors())
            )
        )

    def dense_equivalent_parameter_count(self) -> int:
        return int(
            super().dense_equivalent_parameter_count()
            + sum(parameter.numel() for parameter in self.perception.parameters())
            + sum(
                region.incoming.out_features * region.incoming.in_features
                + (
                    0
                    if region.recurrent is None
                    else region.recurrent.out_features * region.recurrent.in_features
                )
                for region in self._neuron_regions.values()
            )
            + sum(
                source.unit_count * target.unit_count
                for network in self._neuron_networks.values()
                for _, source_id, target_id, _ in network.connections
                for source in network.regions
                if source.region_id == source_id
                for target in network.regions
                if target.region_id == target_id
            )
            + (
                0
                if self._workspace_router is None
                else sum(parameter.numel() for parameter in self._workspace_router.parameters())
            )
        )

    def checkpoint(self) -> dict[str, Any]:
        payload = super().checkpoint()
        payload["adapter"] = self.ADAPTER_NAME
        payload["perception"] = self.perception.checkpoint()
        if self._world_dynamics is not None:
            payload["world_dynamics"] = self._world_dynamics_checkpoint()
        if self._workspace_router is not None:
            payload["workspace_router"] = self._workspace_router.checkpoint()
        if self._episodic_memory is not None:
            payload["episodic_memory"] = self._episodic_memory.checkpoint()
        if self._semantic_memory is not None:
            payload["semantic_memory"] = self._semantic_memory.checkpoint()
        payload["concept_formation"] = self._concept_formation.checkpoint()
        payload["growth_requests"] = self._growth_requests_checkpoint()
        payload["topology_proposals"] = self._topology_proposals_checkpoint()
        payload["neuron_regions"] = self._neuron_regions_checkpoint()
        payload["neuron_networks"] = self._neuron_networks_checkpoint()
        payload["structural_growth"] = self._structural_growth_checkpoint()
        payload["structural_pruning"] = self._structural_pruning_checkpoint()
        payload["online_concept_branches"] = self._online_concept_branches_checkpoint()
        if self._procedural_memory is not None:
            payload["procedural_memory"] = self._procedural_memory.checkpoint()
        if self._procedural_sequence_memory is not None:
            payload["procedural_sequence_memory"] = self._procedural_sequence_memory.checkpoint()
        if self._homeostatic_controller is not None:
            payload["homeostasis"] = self._homeostatic_controller.checkpoint()
        if self._goal_planner is not None:
            payload["planning"] = self._goal_planner.checkpoint()
        if self._affordance_features is not None:
            payload["affordance_features"] = self._affordance_features.checkpoint()
        if self._executive is not None:
            payload["executive"] = self._executive.checkpoint()
        if self._generation_controller is not None:
            payload["generation"] = self._generation_controller.checkpoint()
        if self._content_selector is not None:
            payload["content_selection"] = self._content_selector.checkpoint()
        if self._language_organ is not None:
            payload["language_organ"] = self._language_organ.checkpoint()
        payload["language_backend_registry"] = self._language_backend_registry.checkpoint()
        payload["language_provider_artifact"] = (
            None
            if self._language_provider_artifact is None
            else self._language_provider_artifact.to_payload()
        )
        payload["planned_rollout"] = (
            None if self._planned_rollout is None else self._planned_rollout.to_payload()
        )
        payload["recovery_portfolio"] = (
            None if self._recovery_portfolio is None else self._recovery_portfolio.to_payload()
        )
        payload["recovery_archive"] = self._recovery_archive.to_payload()
        payload["recovery_strategy_ledger"] = self._recovery_strategy_ledger.to_payload()
        payload["recovery_reader_dependencies"] = self._recovery_reader_dependencies.to_payload()
        payload["recovery_generation"] = self._recovery_generation
        payload["recovery_memory_epochs"] = self._recovery_memory_epochs
        payload["recovery_semantic_learning_rate"] = self._recovery_semantic_learning_rate
        payload["recovery_procedural_learning_rate"] = self._recovery_procedural_learning_rate
        payload["recovery_memory_rebuild_count"] = self._recovery_memory_rebuild_count
        payload["replan_required"] = self._replan_required
        payload["planning_recovery"] = (
            None
            if self._cognitive_state.planning_recovery is None
            else self._cognitive_state.planning_recovery.to_payload()
        )
        payload["last_rollout_prediction_error"] = self._last_rollout_prediction_error
        payload["last_rollout_calibrated_confidence"] = self._last_rollout_calibrated_confidence
        payload["last_content_selection"] = (
            None
            if self._last_content_selection is None
            else self._last_content_selection.to_payload()
        )
        payload["last_content_prediction_error"] = self._last_content_prediction_error
        payload["content_feedback_applied"] = self._content_feedback_applied
        payload["last_executive_decision"] = (
            None
            if self._last_executive_decision is None
            else self._last_executive_decision.to_payload()
        )
        payload["last_executive_prediction_error"] = self._last_executive_prediction_error
        payload["last_delayed_executive_prediction_error"] = (
            self._last_delayed_executive_prediction_error
        )
        payload["last_affordance_prediction_error"] = self._last_affordance_prediction_error
        payload["last_executive_world_action"] = (
            None
            if self._last_executive_world_action is None
            else self._last_executive_world_action.to_payload()
        )
        payload["pending_executive_credit"] = self._pending_executive_credit_checkpoint()
        payload["language_fallback_count"] = self._language_fallback_count
        payload["language_fallback_requires_replan"] = self._language_fallback_requires_replan
        payload["last_language_emission"] = (
            None
            if self._last_language_emission is None
            else self._last_language_emission.to_payload()
        )
        payload["cognitive_state"] = self._cognitive_state.to_payload()
        return payload

    def restore(self, checkpoint: Mapping[str, Any]) -> None:
        super().restore(checkpoint)
        self._restore_cognitive_state(checkpoint)
        if "perception" in checkpoint:
            self.perception.restore(checkpoint["perception"])
        self._restore_world_dynamics(checkpoint.get("world_dynamics"))
        self._restore_workspace_router(checkpoint.get("workspace_router"))
        self._restore_episodic_memory(checkpoint.get("episodic_memory"))
        self._restore_semantic_memory(checkpoint.get("semantic_memory"))
        self._restore_concept_formation(checkpoint.get("concept_formation"))
        self._restore_growth_requests(checkpoint.get("growth_requests"))
        self._restore_topology_proposals(checkpoint.get("topology_proposals"))
        self._restore_neuron_regions(checkpoint.get("neuron_regions"))
        self._restore_neuron_networks(checkpoint.get("neuron_networks"))
        self._restore_structural_growth(checkpoint.get("structural_growth"))
        self._restore_structural_pruning(checkpoint.get("structural_pruning"))
        self._restore_online_concept_branches(checkpoint.get("online_concept_branches"))
        self._restore_procedural_memory(checkpoint.get("procedural_memory"))
        self._restore_procedural_sequence_memory(checkpoint.get("procedural_sequence_memory"))
        self._restore_homeostatic_controller(checkpoint.get("homeostasis"))
        self._restore_goal_planner(checkpoint.get("planning"))
        self._restore_affordance_features(checkpoint.get("affordance_features"))
        self._restore_executive(checkpoint.get("executive"))
        self._restore_generation_controller(checkpoint.get("generation"))
        self._restore_content_selector(checkpoint.get("content_selection"))
        self._restore_language_backend_registry(checkpoint.get("language_backend_registry"))
        self._restore_language_provider_artifact(checkpoint)
        self._restore_language_organ(checkpoint.get("language_organ"))
        self._restore_rollout_state(checkpoint)
        self._restore_recovery_portfolio(checkpoint)
        self._restore_recovery_archive(checkpoint)
        self._restore_recovery_strategy_ledger(checkpoint)
        self._restore_recovery_reader_dependencies(checkpoint)
        self._restore_recovery_generation(checkpoint)
        self._restore_recovery_memory_state(checkpoint)
        self._restore_generation_trace(checkpoint)
        self._restore_content_selection(checkpoint)
        self._restore_executive_state(checkpoint)
        self._restore_language_emission(checkpoint)
        self._restore_language_fallback_state(checkpoint)

    def _world_dynamics_checkpoint(self) -> dict[str, Any]:
        if self._world_dynamics is None:
            raise RuntimeError("world dynamics is not attached")
        return {
            "schema": self._world_dynamics.schema.payload(),
            "schema_registry": self._world_dynamics.schema_registry.checkpoint(),
            "hidden_dim": self._world_dynamics.hidden_dim,
            "online_updates": self._world_dynamics.online_updates,
            "transition_acceptances": self._world_dynamics.transition_acceptances,
            "transition_rejections": self._world_dynamics.transition_rejections,
            "schema_evolution_count": self._world_dynamics.schema_evolution_count,
            "state_dict": {
                name: tensor.detach().cpu().clone()
                for name, tensor in self._world_dynamics.state_dict().items()
            },
            "schema_snapshots": {
                str(version): {
                    name: tensor.detach().cpu().clone() for name, tensor in snapshot.items()
                }
                for version, snapshot in self._world_dynamics._schema_snapshots.items()
            },
        }

    def _restore_world_dynamics(self, payload: Any) -> None:
        if payload is None:
            self._world_dynamics = None
            return
        schema = WorldSchema.from_payload(dict(payload["schema"]))
        registry_payload = payload.get("schema_registry")
        registry = (
            None
            if registry_payload is None
            else WorldSchemaRegistry.from_checkpoint(dict(registry_payload))
        )
        if registry is not None and registry.schema != schema:
            raise ValueError("world dynamics schema registry does not match learner schema")
        learner = WorldDynamicsLearner(
            schema,
            hidden_dim=int(payload["hidden_dim"]),
            seed=0,
            schema_registry=registry,
        )
        learner.load_state_dict(payload["state_dict"])
        learner.online_updates = int(payload.get("online_updates", 0))
        learner.transition_acceptances = int(payload.get("transition_acceptances", 0))
        learner.transition_rejections = int(payload.get("transition_rejections", 0))
        learner.schema_evolution_count = int(payload.get("schema_evolution_count", 0))
        snapshots = payload.get("schema_snapshots")
        if isinstance(snapshots, dict):
            learner._schema_snapshots = {
                int(version): {
                    str(name): tensor.detach().cpu().clone() for name, tensor in snapshot.items()
                }
                for version, snapshot in snapshots.items()
                if isinstance(snapshot, dict)
            }
            learner._schema_snapshots.setdefault(
                learner.schema_registry.active_version,
                learner._snapshot_state_dict(),
            )
        self._world_dynamics = learner

    def _restore_workspace_router(self, payload: Any) -> None:
        self._workspace_router = (
            None
            if payload is None
            else WorkspaceRouter.from_checkpoint(dict(payload), device=self.device)
        )

    def _restore_episodic_memory(self, payload: Any) -> None:
        self._episodic_memory = (
            None
            if payload is None
            else EpisodicMemoryStore.from_checkpoint(dict(payload), device=self.device)
        )

    def _restore_semantic_memory(self, payload: Any) -> None:
        self._semantic_memory = (
            None
            if payload is None
            else SemanticMemoryLearner.from_checkpoint(dict(payload), device=self.device)
        )

    def _restore_concept_formation(self, payload: Any) -> None:
        self._concept_formation = (
            ConceptFormationOrgan(
                similarity_threshold=self.config.concept_similarity_threshold,
                signal_weights=self.config.concept_signal_weights,
                capacity=self.config.concept_capacity,
                plasticity_rate=self.config.concept_plasticity_rate,
                prune_threshold=self.config.concept_prune_threshold,
            )
            if payload is None
            else ConceptFormationOrgan.from_checkpoint(dict(payload), device=self.device)
        )

    def _restore_procedural_memory(self, payload: Any) -> None:
        self._procedural_memory = (
            None
            if payload is None
            else ProceduralMemoryLearner.from_checkpoint(dict(payload), device=self.device)
        )

    def _restore_procedural_sequence_memory(self, payload: Any) -> None:
        self._procedural_sequence_memory = (
            None
            if payload is None
            else ProceduralSequenceLearner.from_checkpoint(dict(payload), device=self.device)
        )

    def _restore_homeostatic_controller(self, payload: Any) -> None:
        self._homeostatic_controller = (
            None if payload is None else HomeostaticController.from_checkpoint(dict(payload))
        )

    def _restore_goal_planner(self, payload: Any) -> None:
        self._goal_planner = None if payload is None else GoalPlanner.from_checkpoint(dict(payload))

    def _restore_affordance_features(self, payload: Any) -> None:
        self._affordance_features = (
            None
            if payload is None
            else LearnedAffordanceFeatures.from_checkpoint(dict(payload)).to(self.device)
        )
        self._affordance_grounding = (
            None
            if self._affordance_features is None
            else WorldAffordanceGroundingProducer(self._affordance_features.input_dim)
        )

    def _restore_executive(self, payload: Any) -> None:
        self._executive = (
            None
            if payload is None
            else ExecutiveController.from_checkpoint(dict(payload)).to(self.device)
        )

    def _restore_executive_state(self, payload: Any) -> None:
        decision = payload.get("last_executive_decision") if isinstance(payload, dict) else None
        self._last_executive_decision = (
            None
            if decision is None
            else ExecutiveDecision.from_payload(dict(decision), device=self.device)
        )
        error = (
            payload.get("last_executive_prediction_error") if isinstance(payload, dict) else None
        )
        self._last_executive_prediction_error = None if error is None else float(error)
        delayed_error = (
            payload.get("last_delayed_executive_prediction_error")
            if isinstance(payload, dict)
            else None
        )
        self._last_delayed_executive_prediction_error = (
            None if delayed_error is None else float(delayed_error)
        )
        affordance_error = (
            payload.get("last_affordance_prediction_error") if isinstance(payload, dict) else None
        )
        self._last_affordance_prediction_error = (
            None if affordance_error is None else float(affordance_error)
        )
        action = payload.get("last_executive_world_action") if isinstance(payload, dict) else None
        self._last_executive_world_action = (
            None if action is None else WorldAction.from_payload(dict(action), device=self.device)
        )
        self._restore_pending_executive_credit(
            payload.get("pending_executive_credit") if isinstance(payload, dict) else None
        )

    def _pending_executive_credit_checkpoint(self) -> dict[str, Any] | None:
        pending = self._pending_executive_credit
        if pending is None:
            return None
        return {
            "decision": pending.decision.to_payload(),
            "affordance": (None if pending.affordance is None else pending.affordance.to_payload()),
            "percept_features": (
                None
                if pending.percept_features is None
                else pending.percept_features.detach().cpu().clone()
            ),
            "world_latent": (
                None
                if pending.world_latent is None
                else pending.world_latent.detach().cpu().clone()
            ),
            "world_uncertainty": pending.world_uncertainty,
            "learn": pending.learn,
        }

    def _restore_pending_executive_credit(self, payload: Any) -> None:
        if payload is None:
            self._pending_executive_credit = None
            return
        if not isinstance(payload, dict):
            raise ValueError("pending executive credit checkpoint must be a mapping")
        affordance_payload = payload.get("affordance")
        self._pending_executive_credit = _PendingExecutiveCredit(
            decision=ExecutiveDecision.from_payload(dict(payload["decision"]), device=self.device),
            affordance=(
                None
                if affordance_payload is None
                else WorldAffordance.from_payload(dict(affordance_payload), device=self.device)
            ),
            percept_features=(
                None
                if payload.get("percept_features") is None
                else payload["percept_features"].detach().to(self.device).clone()
            ),
            world_latent=(
                None
                if payload.get("world_latent") is None
                else payload["world_latent"].detach().to(self.device).clone()
            ),
            world_uncertainty=float(payload.get("world_uncertainty", 1.0)),
            learn=bool(payload.get("learn", True)),
        )

    def _restore_generation_controller(self, payload: Any) -> None:
        self._generation_controller = (
            None if payload is None else GenerationController.from_checkpoint(dict(payload))
        )

    def _restore_generation_trace(self, payload: Any) -> None:
        trace = payload.get("last_generation_trace") if isinstance(payload, dict) else None
        self._last_generation_trace = (
            None if trace is None else GenerationTrace.from_payload(dict(trace))
        )

    def _restore_content_selector(self, payload: Any) -> None:
        self._content_selector = (
            None if payload is None else ContentSelector.from_checkpoint(dict(payload))
        )

    def _restore_content_selection(self, payload: Any) -> None:
        selection = payload.get("last_content_selection") if isinstance(payload, dict) else None
        self._last_content_selection = (
            None if selection is None else ContentSelectionDecision.from_payload(dict(selection))
        )
        error = payload.get("last_content_prediction_error") if isinstance(payload, dict) else None
        self._last_content_prediction_error = None if error is None else float(error)
        self._content_feedback_applied = bool(
            payload.get("content_feedback_applied", False) if isinstance(payload, dict) else False
        )

    def _restore_language_fallback_state(self, payload: Any) -> None:
        self._language_fallback_count = int(
            payload.get("language_fallback_count", 0) if isinstance(payload, dict) else 0
        )
        self._language_fallback_requires_replan = bool(
            payload.get("language_fallback_requires_replan", False)
            if isinstance(payload, dict)
            else False
        )

    def _restore_language_organ(self, payload: Any) -> None:
        if payload is None:
            self._language_organ = None
            return
        if not isinstance(payload, dict):
            raise ValueError("language organ checkpoint must be a mapping")
        if payload.get("backend") != StructuredTextLanguageOrgan.BACKEND_ID:
            raise ValueError(
                "only the structured language-organ stub can be restored without a backend registry"
            )
        restored = StructuredTextLanguageOrgan.from_checkpoint(payload)
        self._language_backend_registry.validate(restored)
        if (
            self._language_provider_artifact is not None
            and restored.backend_id != self._language_provider_artifact.backend_id
        ):
            raise ValueError("restored language organ backend does not match provider artifact")
        self._language_organ = restored

    def _restore_language_backend_registry(self, payload: Any) -> None:
        self._language_backend_registry = (
            LanguageBackendRegistry.default()
            if payload is None
            else LanguageBackendRegistry.from_checkpoint(dict(payload))
        )

    def _restore_language_provider_artifact(self, payload: Any) -> None:
        artifact = payload.get("language_provider_artifact") if isinstance(payload, dict) else None
        self._language_provider_artifact = (
            None if artifact is None else LanguageProviderArtifact.from_payload(dict(artifact))
        )

    def _restore_language_emission(self, payload: Any) -> None:
        emission = payload.get("last_language_emission") if isinstance(payload, dict) else None
        self._last_language_emission = (
            None if emission is None else LanguageEmission.from_payload(dict(emission))
        )

    def _restore_cognitive_state(self, payload: Any) -> None:
        state = payload.get("cognitive_state") if isinstance(payload, dict) else None
        if state is None:
            # 2026-08-26：旧信封没有该键，按内核状态重建，保证 tick/episode_id 与内核同步。
            self._cognitive_state = replace(
                self._empty_cognitive_state(self._state.episode_id), tick=self.tick
            )
            return
        self._cognitive_state = CognitiveState.from_payload(state, device=self.device)

    def _restore_rollout_state(self, payload: Any) -> None:
        rollout = payload.get("planned_rollout") if isinstance(payload, dict) else None
        self._planned_rollout = (
            None if rollout is None else ImaginedRollout.from_payload(dict(rollout))
        )
        self._replan_required = bool(
            payload.get("replan_required", False) if isinstance(payload, dict) else False
        )
        recovery = payload.get("planning_recovery") if isinstance(payload, dict) else None
        self._cognitive_state = replace(
            self._cognitive_state,
            planning_recovery=(
                None if recovery is None else PlanningRecoveryState.from_payload(recovery)
            ),
        )
        error = payload.get("last_rollout_prediction_error") if isinstance(payload, dict) else None
        self._last_rollout_prediction_error = None if error is None else float(error)
        confidence = (
            payload.get("last_rollout_calibrated_confidence") if isinstance(payload, dict) else None
        )
        self._last_rollout_calibrated_confidence = None if confidence is None else float(confidence)

    def _restore_recovery_portfolio(self, payload: Any) -> None:
        portfolio = payload.get("recovery_portfolio") if isinstance(payload, dict) else None
        self._recovery_portfolio = (
            None if portfolio is None else RecoveryPortfolio.from_payload(dict(portfolio))
        )

    def _restore_recovery_archive(self, payload: Any) -> None:
        archive = payload.get("recovery_archive") if isinstance(payload, dict) else None
        self._recovery_archive = (
            RecoveryPortfolioArchive(capacity=self.config.recovery_archive_capacity)
            if archive is None
            else RecoveryPortfolioArchive.from_payload(dict(archive))
        )

    def _restore_recovery_generation(self, payload: Any) -> None:
        self._recovery_generation = int(
            payload.get("recovery_generation", 0) if isinstance(payload, dict) else 0
        )

    def _restore_recovery_strategy_ledger(self, payload: Any) -> None:
        ledger = payload.get("recovery_strategy_ledger") if isinstance(payload, dict) else None
        self._recovery_strategy_ledger = (
            RecoveryStrategyLedger(
                evidence_threshold=self.config.recovery_strategy_evidence_threshold,
                memory_budget=self.config.recovery_strategy_memory_budget,
                evidence_weight=self.config.recovery_strategy_evidence_weight,
                consistency_weight=self.config.recovery_strategy_consistency_weight,
                resource_weight=self.config.recovery_strategy_resource_weight,
            )
            if ledger is None
            else RecoveryStrategyLedger.from_payload(dict(ledger))
        )

    def _restore_recovery_reader_dependencies(self, payload: Any) -> None:
        dependencies = (
            payload.get("recovery_reader_dependencies") if isinstance(payload, dict) else None
        )
        self._recovery_reader_dependencies = (
            RecoveryReaderDependencyGraph()
            if dependencies is None
            else RecoveryReaderDependencyGraph.from_payload(dict(dependencies))
        )

    def _restore_recovery_memory_state(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        self._recovery_memory_epochs = int(payload.get("recovery_memory_epochs", 300))
        self._recovery_semantic_learning_rate = float(
            payload.get("recovery_semantic_learning_rate", 0.1)
        )
        self._recovery_procedural_learning_rate = float(
            payload.get("recovery_procedural_learning_rate", 0.1)
        )
        self._recovery_memory_rebuild_count = int(payload.get("recovery_memory_rebuild_count", 0))

    def native_checkpoint(self) -> dict[str, Any]:
        """Serialize the v1 cognitive state and its TSK-v8 compatibility kernel."""

        components: dict[str, Any] = {"perception": self.perception.checkpoint()}
        if self._world_dynamics is not None:
            components["world_dynamics"] = self._world_dynamics_checkpoint()
        if self._workspace_router is not None:
            components["workspace_router"] = self._workspace_router.checkpoint()
        if self._episodic_memory is not None:
            components["episodic_memory"] = self._episodic_memory.checkpoint()
        if self._semantic_memory is not None:
            components["semantic_memory"] = self._semantic_memory.checkpoint()
        components["concept_formation"] = self._concept_formation.checkpoint()
        components["growth_requests"] = self._growth_requests_checkpoint()
        components["topology_proposals"] = self._topology_proposals_checkpoint()
        components["neuron_regions"] = self._neuron_regions_checkpoint()
        components["neuron_networks"] = self._neuron_networks_checkpoint()
        components["structural_growth"] = self._structural_growth_checkpoint()
        components["structural_pruning"] = self._structural_pruning_checkpoint()
        components["structural_runtime"] = self._structural_runtime_checkpoint()
        components["online_concept_branches"] = self._online_concept_branches_checkpoint()
        if self._procedural_memory is not None:
            components["procedural_memory"] = self._procedural_memory.checkpoint()
        if self._procedural_sequence_memory is not None:
            components["procedural_sequence_memory"] = self._procedural_sequence_memory.checkpoint()
        if self._homeostatic_controller is not None:
            components["homeostasis"] = self._homeostatic_controller.checkpoint()
        if self._goal_planner is not None:
            components["planning"] = self._goal_planner.checkpoint()
        if self._affordance_features is not None:
            components["affordance_features"] = self._affordance_features.checkpoint()
        if self._executive is not None:
            components["executive"] = self._executive.checkpoint()
        if self._generation_controller is not None:
            components["generation"] = self._generation_controller.checkpoint()
        if self._content_selector is not None:
            components["content_selection"] = self._content_selector.checkpoint()
        if self._language_organ is not None:
            components["language_organ"] = self._language_organ.checkpoint()
        components["language_backend_registry"] = self._language_backend_registry.checkpoint()
        components["language_provider_artifact"] = (
            None
            if self._language_provider_artifact is None
            else self._language_provider_artifact.to_payload()
        )
        components["planned_rollout"] = (
            None if self._planned_rollout is None else self._planned_rollout.to_payload()
        )
        components["recovery_portfolio"] = (
            None if self._recovery_portfolio is None else self._recovery_portfolio.to_payload()
        )
        components["recovery_archive"] = self._recovery_archive.to_payload()
        components["recovery_strategy_ledger"] = self._recovery_strategy_ledger.to_payload()
        components["recovery_reader_dependencies"] = self._recovery_reader_dependencies.to_payload()
        components["recovery_generation"] = self._recovery_generation
        components["recovery_memory_epochs"] = self._recovery_memory_epochs
        components["recovery_semantic_learning_rate"] = self._recovery_semantic_learning_rate
        components["recovery_procedural_learning_rate"] = self._recovery_procedural_learning_rate
        components["recovery_memory_rebuild_count"] = self._recovery_memory_rebuild_count
        components["replan_required"] = self._replan_required
        components["planning_recovery"] = (
            None
            if self._cognitive_state.planning_recovery is None
            else self._cognitive_state.planning_recovery.to_payload()
        )
        components["last_rollout_prediction_error"] = self._last_rollout_prediction_error
        components["last_rollout_calibrated_confidence"] = self._last_rollout_calibrated_confidence
        components["last_generation_trace"] = (
            None
            if self._last_generation_trace is None
            else self._last_generation_trace.to_payload()
        )
        components["last_content_selection"] = (
            None
            if self._last_content_selection is None
            else self._last_content_selection.to_payload()
        )
        components["last_content_prediction_error"] = self._last_content_prediction_error
        components["content_feedback_applied"] = self._content_feedback_applied
        components["last_executive_decision"] = (
            None
            if self._last_executive_decision is None
            else self._last_executive_decision.to_payload()
        )
        components["last_executive_prediction_error"] = self._last_executive_prediction_error
        components["last_delayed_executive_prediction_error"] = (
            self._last_delayed_executive_prediction_error
        )
        components["last_affordance_prediction_error"] = self._last_affordance_prediction_error
        components["last_executive_world_action"] = (
            None
            if self._last_executive_world_action is None
            else self._last_executive_world_action.to_payload()
        )
        components["pending_executive_credit"] = self._pending_executive_credit_checkpoint()
        components["language_fallback_count"] = self._language_fallback_count
        components["language_fallback_requires_replan"] = self._language_fallback_requires_replan
        components["last_language_emission"] = (
            None
            if self._last_language_emission is None
            else self._last_language_emission.to_payload()
        )
        return NativeCheckpoint(
            kernel=super().checkpoint(),
            cognitive_state=self.cognitive_snapshot(),
            adapter=self.ADAPTER_NAME,
            components=components,
        ).to_payload()

    def restore_native(self, checkpoint: dict[str, Any]) -> None:
        envelope = NativeCheckpoint.from_payload(checkpoint, device=self.device)
        if envelope.adapter != self.ADAPTER_NAME:
            raise ValueError(f"unsupported Taiji adapter: {envelope.adapter}")
        super().restore(envelope.kernel)
        if "perception" in envelope.components:
            self.perception.restore(envelope.components["perception"])
        self._restore_world_dynamics(envelope.components.get("world_dynamics"))
        self._restore_workspace_router(envelope.components.get("workspace_router"))
        self._restore_episodic_memory(envelope.components.get("episodic_memory"))
        self._restore_semantic_memory(envelope.components.get("semantic_memory"))
        self._restore_concept_formation(envelope.components.get("concept_formation"))
        self._restore_growth_requests(envelope.components.get("growth_requests"))
        self._restore_topology_proposals(envelope.components.get("topology_proposals"))
        self._restore_neuron_regions(envelope.components.get("neuron_regions"))
        self._restore_neuron_networks(envelope.components.get("neuron_networks"))
        self._restore_structural_growth(envelope.components.get("structural_growth"))
        self._restore_structural_pruning(envelope.components.get("structural_pruning"))
        self._restore_structural_runtime(envelope.components.get("structural_runtime"))
        self._restore_online_concept_branches(envelope.components.get("online_concept_branches"))
        self._restore_procedural_memory(envelope.components.get("procedural_memory"))
        self._restore_procedural_sequence_memory(
            envelope.components.get("procedural_sequence_memory")
        )
        self._restore_homeostatic_controller(envelope.components.get("homeostasis"))
        self._restore_goal_planner(envelope.components.get("planning"))
        self._restore_affordance_features(envelope.components.get("affordance_features"))
        self._restore_executive(envelope.components.get("executive"))
        self._restore_generation_controller(envelope.components.get("generation"))
        self._restore_content_selector(envelope.components.get("content_selection"))
        self._restore_language_backend_registry(
            envelope.components.get("language_backend_registry")
        )
        self._restore_language_provider_artifact(envelope.components)
        self._restore_language_organ(envelope.components.get("language_organ"))
        self._restore_rollout_state(envelope.components)
        self._restore_recovery_portfolio(envelope.components)
        self._restore_recovery_archive(envelope.components)
        self._restore_recovery_strategy_ledger(envelope.components)
        self._restore_recovery_reader_dependencies(envelope.components)
        self._restore_recovery_generation(envelope.components)
        self._restore_recovery_memory_state(envelope.components)
        self._restore_generation_trace(envelope.components)
        self._restore_content_selection(envelope.components)
        self._restore_executive_state(envelope.components)
        self._restore_language_emission(envelope.components)
        self._restore_language_fallback_state(envelope.components)
        state = envelope.cognitive_state
        if state.tick != self.tick or state.episode_id != self._state.episode_id:
            raise ValueError("native cognitive state is out of sync with kernel state")
        self._cognitive_state = state

    @classmethod
    def from_native_checkpoint(
        cls,
        checkpoint: dict[str, Any],
        *,
        device: torch.device | str = "cpu",
    ) -> TSKV8Adapter:
        """Construct an adapter from the atomic v1 envelope."""

        envelope = NativeCheckpoint.from_payload(checkpoint, device=device)
        config = cls._config_from_kernel_checkpoint(envelope.kernel)
        model = cls(config, device=device, episode_id=envelope.cognitive_state.episode_id)
        model.restore_native(checkpoint)
        return model

    @staticmethod
    def _config_from_kernel_checkpoint(checkpoint: Mapping[str, Any]) -> Any:
        # Import locally so the adapter's public import surface stays small.
        from .config import TaijiConfig

        return TaijiConfig.from_dict(dict(checkpoint["config"]))
