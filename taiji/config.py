"""Configuration contract for the native Taiji architecture."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

DEFAULT_RECOVERY_INTERACTION_RESIDUAL_TOLERANCE = 1e-7
DEFAULT_RECOVERY_INTERACTION_ORDER_TOLERANCE = 1e-7


@dataclass(frozen=True)
class CapacityPolicy:
    """Structural proportions used by the parameter-budget planner.

    Dynamics stay in :class:`TaijiConfig`; this policy owns only dimensions
    that may be searched or adapted between training runs.  Ratios are relative
    to the first cortical region, so changing depth no longer requires copying
    a complete configuration full of unrelated learning constants.
    """

    region_ratios: tuple[float, ...] = (1.0, 0.75, 0.50)
    synapse_fan_in_ratio: float = 0.1875
    motor_fan_in_ratio: float = 0.375
    memory_units_ratio: float = 1.50
    memory_fan_in_ratio: float = 0.25
    memory_meta_ratio: float = 0.375
    memory_readout_fan_in_ratio: float = 0.375
    memory_time_ratio: float = 0.0625
    memory_episode_ratio: float = 0.125
    lateral_fan_in_ratio: float = 0.125
    alignment: int = 8

    def __post_init__(self) -> None:
        if not self.region_ratios or any(float(value) <= 0.0 for value in self.region_ratios):
            raise ValueError("region_ratios must contain positive values")
        for name in (
            "synapse_fan_in_ratio",
            "motor_fan_in_ratio",
            "memory_units_ratio",
            "memory_fan_in_ratio",
            "memory_meta_ratio",
            "memory_readout_fan_in_ratio",
            "memory_time_ratio",
            "memory_episode_ratio",
            "lateral_fan_in_ratio",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if int(self.alignment) <= 0:
            raise ValueError("alignment must be positive")

    @classmethod
    def from_config(
        cls,
        config: TaijiConfig,
        *,
        alignment: int = 8,
    ) -> CapacityPolicy:
        """Recover structural proportions from an existing configuration."""

        width = float(config.region_sizes[0])
        return cls(
            region_ratios=tuple(float(size) / width for size in config.region_sizes),
            synapse_fan_in_ratio=float(config.synapse_fan_in) / width,
            motor_fan_in_ratio=float(config.motor_fan_in) / width,
            memory_units_ratio=float(config.memory_units) / width,
            memory_fan_in_ratio=float(config.memory_fan_in) / width,
            memory_meta_ratio=float(config.memory_meta_dim) / width,
            memory_readout_fan_in_ratio=float(config.memory_readout_fan_in) / width,
            memory_time_ratio=float(config.memory_time_dim) / width,
            memory_episode_ratio=float(config.memory_episode_dim) / width,
            lateral_fan_in_ratio=float(config.lateral_fan_in) / width,
            alignment=int(alignment),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["region_ratios"] = list(self.region_ratios)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CapacityPolicy:
        values = dict(payload)
        values["region_ratios"] = tuple(float(value) for value in values["region_ratios"])
        # Keep policy JSON written before the memory-dimension ratios existed
        # loadable while allowing new searches to control those dimensions.
        values.setdefault("memory_time_ratio", cls.memory_time_ratio)
        values.setdefault("memory_episode_ratio", cls.memory_episode_ratio)
        return cls(**values)


@dataclass(frozen=True)
class PerceptionConfig:
    """Learned local-feature and variable-duration assembly controls."""

    feature_dim: int = 32
    local_window: int = 4
    minimum_assembly_duration: int = 1
    maximum_assembly_duration: int = 16
    boundary_threshold: float = 0.55
    change_gain: float = 0.55
    surprise_gain: float = 0.45
    novelty_gain: float = 0.15
    learning_rate: float = 0.02
    surprise_baseline_rate: float = 0.1
    boundary_hysteresis: float = 0.05
    seed_offset: int = 3251

    def __post_init__(self) -> None:
        if self.feature_dim <= 0:
            raise ValueError("perception feature_dim must be positive")
        if self.local_window <= 0:
            raise ValueError("perception local_window must be positive")
        if self.minimum_assembly_duration <= 0:
            raise ValueError("minimum assembly duration must be positive")
        if self.maximum_assembly_duration < self.minimum_assembly_duration:
            raise ValueError("maximum assembly duration cannot be below minimum")
        if not 0.0 <= self.boundary_threshold <= 1.0:
            raise ValueError("perception boundary_threshold must be in [0, 1]")
        if self.change_gain < 0.0 or self.surprise_gain < 0.0 or self.novelty_gain < 0.0:
            raise ValueError("perception boundary gains cannot be negative")
        if self.change_gain + self.surprise_gain + self.novelty_gain <= 0.0:
            raise ValueError("at least one perception boundary gain must be active")
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError("perception learning_rate must be in (0, 1]")
        if not 0.0 < self.surprise_baseline_rate <= 1.0:
            raise ValueError("perception surprise_baseline_rate must be in (0, 1]")
        if not 0.0 <= self.boundary_hysteresis <= 1.0:
            raise ValueError("perception boundary_hysteresis must be in [0, 1]")
        if self.seed_offset <= 0:
            raise ValueError("perception seed_offset must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PerceptionConfig:
        return cls(**dict(payload))


@dataclass(frozen=True)
class TaijiConfig:
    """Shape and dynamics of one Taiji predictive fabric.

    Sizes are implementation parameters, not model identities.  The default
    sensor and motor alphabet is raw bytes plus one boundary action.
    """

    alphabet_size: int = 257
    boundary_symbol: int = 256
    region_sizes: tuple[int, ...] = (128, 96, 64)
    synapse_fan_in: int = 24
    motor_fan_in: int = 48

    memory_units: int = 192
    memory_fan_in: int = 32
    memory_readout_fan_in: int = 48
    memory_meta_dim: int = 48
    memory_iterations: int = 3
    memory_time_dim: int = 8
    memory_episode_dim: int = 16
    memory_action_decoder: str = "shared"

    membrane_decay: float = 0.65
    trace_decay: float = 0.82
    inhibition_decay: float = 0.80
    inhibition_gain: float = 0.55
    lateral_fan_in: int = 16
    lateral_learning_rate: float = 0.02
    lateral_seed_offset: int = 977
    consolidation_seed_offset: int = 1951
    bottom_up_gain: float = 1.00
    recurrent_gain: float = 0.55
    top_down_gain: float = 0.30

    threshold_base: float = 0.02
    threshold_min: float = -0.20
    threshold_max: float = 1.50
    homeostasis_rate: float = 0.015
    target_activity: float = 0.12
    cortical_baseline_rate: float = 0.00390625
    consolidation_read_gain: float = 1.00

    predictive_learning_rate: float = 0.025
    transition_learning_rate: float = 0.012
    motor_learning_rate: float = 0.10
    bias_learning_rate: float = 0.025
    reward_baseline_rate: float = 0.05
    episodic_learning_rate: float = 0.60
    episodic_readout_learning_rate: float = 0.85
    episodic_write_repeats: int = 2
    cortical_readout_learning_rate: float = 0.30
    cortical_readout_repeats: int = 8
    readout_episode_saturation: float = 8.0
    readout_value_saturation: float = 8.0
    # Lifetime write count is provenance, not evidence quality.  A non-zero
    # value is retained only as a legacy compatibility knob; the native
    # default does not make recall weaker merely because the field grew.
    memory_confidence_decay: float = 0.0
    synapse_decay: float = 1e-5

    weight_init_scale: float = 0.45
    max_weight_norm: float = 2.5
    max_membrane_norm: float = 8.0
    max_trace_norm: float = 5.0
    motor_context_norm: float = 4.0
    motor_temperature: float = 0.75
    memory_trace_decay: float = 0.72
    memory_inhibition_gain: float = 0.75
    memory_recurrent_gain: float = 1.35
    memory_event_gain: float = 0.80
    memory_action_binding_gain: float = 2.00
    memory_read_gain: float = 3.00
    memory_replay_read_gain: float = 1.00
    memory_feedback_gain: float = 0.25
    memory_novelty_gain: float = 0.70
    memory_reward_gain: float = 0.30
    replay_memory_learning_scale: float = 0.25

    replay_seed_gain: float = 0.65
    replay_noise_scale: float = 0.75
    replay_value_weight: float = 0.60
    replay_priority_threshold: float = 0.05
    replay_fatigue_gain: float = 1.20
    replay_learning_scale: float = 0.45
    replay_maturity_ticks: int = 50000
    replay_outcome_fast_scale: float = 0.0
    replay_outcome_slow_scale: float = 1.0
    replay_burst_repeats: int = 8
    replay_write_repeats: int = 8
    replay_winner_resource_retention: float = 0.90
    structural_turnover_ratio: float = 0.25
    structural_capture_target: float = 0.90
    structural_error_threshold: float = 0.35
    world_calibration_history_limit: int = 128
    cognitive_lineage_history_limit: int = 256
    recovery_archive_capacity: int = 256
    recovery_strategy_evidence_threshold: int = 2
    recovery_strategy_memory_budget: float = 1.0
    recovery_strategy_evidence_weight: float = 0.50
    recovery_strategy_consistency_weight: float = 0.30
    recovery_strategy_resource_weight: float = 0.20
    recovery_strategy_interaction_residual_tolerance: float = (
        DEFAULT_RECOVERY_INTERACTION_RESIDUAL_TOLERANCE
    )
    recovery_strategy_interaction_order_tolerance: float = (
        DEFAULT_RECOVERY_INTERACTION_ORDER_TOLERANCE
    )
    recovery_strategy_cross_reader_credit_drift_tolerance: float = 1.0
    recovery_strategy_cross_reader_credit_revision_history_limit: int = 4
    concept_similarity_threshold: float = 0.85
    concept_signal_weights: tuple[float, float, float] = (0.45, 0.35, 0.20)
    concept_capacity: int = 256
    concept_plasticity_rate: float = 0.25
    concept_prune_threshold: float = 0.15
    concept_branch_owner_weights: tuple[float, float, float] = (0.45, 0.40, 0.15)
    concept_branch_owner_min_score: float = 0.65
    concept_branch_owner_min_margin: float = 0.05
    development_structural_budget: int = 32
    self_capability_learning_rate: float = 0.20
    seed: int = 20260821
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)

    def __post_init__(self) -> None:
        if self.alphabet_size < 2:
            raise ValueError("alphabet_size must be at least 2")
        if not 0 <= self.boundary_symbol < self.alphabet_size:
            raise ValueError("boundary_symbol must be inside the alphabet")
        if not self.region_sizes or any(size <= 1 for size in self.region_sizes):
            raise ValueError("region_sizes must contain dimensions greater than 1")
        if (
            self.synapse_fan_in <= 0
            or self.motor_fan_in <= 0
            or self.memory_fan_in <= 0
            or self.lateral_fan_in <= 0
        ):
            raise ValueError("fan-in values must be positive")
        if self.lateral_seed_offset <= 0:
            raise ValueError(
                "lateral_seed_offset must be positive so the lateral bank draws "
                "from a stream distinct from the feedforward projections"
            )
        if self.consolidation_seed_offset <= 0:
            raise ValueError(
                "consolidation_seed_offset must select a positive independent random stream"
            )
        if self.consolidation_seed_offset == self.lateral_seed_offset:
            raise ValueError("consolidation and lateral banks require distinct random streams")
        if self.motor_fan_in > 2 * sum(self.region_sizes):
            raise ValueError("motor_fan_in cannot exceed the available cortical state")
        if self.memory_units <= 1:
            raise ValueError("memory_units must be greater than 1")
        if not 0 < self.memory_readout_fan_in <= self.memory_meta_dim:
            raise ValueError("memory_readout_fan_in must be in [1, memory_meta_dim]")
        if not 0 < self.memory_meta_dim <= self.memory_units:
            raise ValueError("memory_meta_dim must be in [1, memory_units]")
        if self.memory_iterations <= 0:
            raise ValueError("memory_iterations must be positive")
        if self.episodic_write_repeats <= 0:
            raise ValueError("episodic_write_repeats must be positive")
        if self.cortical_readout_repeats <= 0:
            raise ValueError("cortical_readout_repeats must be positive")
        if self.memory_time_dim < 2 or self.memory_time_dim % 2:
            raise ValueError("memory_time_dim must be a positive even dimension")
        if self.memory_episode_dim <= 0:
            raise ValueError("memory_episode_dim must be positive")
        if self.memory_action_decoder not in {"shared", "local", "cue_selective", "dual"}:
            raise ValueError(
                "memory_action_decoder must be 'shared', 'local', 'cue_selective', or 'dual'"
            )
        for name in (
            "membrane_decay",
            "trace_decay",
            "inhibition_decay",
            "target_activity",
            "memory_trace_decay",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.threshold_min >= self.threshold_max:
            raise ValueError("threshold_min must be smaller than threshold_max")
        for name in (
            "homeostasis_rate",
            "inhibition_gain",
            "predictive_learning_rate",
            "transition_learning_rate",
            "lateral_learning_rate",
            "motor_learning_rate",
            "bias_learning_rate",
            "episodic_learning_rate",
            "episodic_readout_learning_rate",
            "cortical_readout_learning_rate",
            "readout_episode_saturation",
            "readout_value_saturation",
            "weight_init_scale",
            "max_weight_norm",
            "max_membrane_norm",
            "max_trace_norm",
            "motor_context_norm",
            "motor_temperature",
            "memory_inhibition_gain",
            "memory_recurrent_gain",
            "memory_event_gain",
            "memory_action_binding_gain",
            "memory_read_gain",
            "memory_replay_read_gain",
            "replay_seed_gain",
            "replay_learning_scale",
            "replay_memory_learning_scale",
            "consolidation_read_gain",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.synapse_decay < 0.0:
            raise ValueError("synapse_decay cannot be negative")
        if self.memory_feedback_gain < 0.0:
            raise ValueError("memory_feedback_gain cannot be negative")
        if self.memory_confidence_decay < 0.0:
            raise ValueError("memory_confidence_decay cannot be negative")
        if not 0.0 < self.reward_baseline_rate <= 1.0:
            raise ValueError("reward_baseline_rate must be in (0, 1]")
        if not 0.0 < self.cortical_baseline_rate <= 1.0:
            raise ValueError("cortical_baseline_rate must be in (0, 1]")
        if not 0.0 <= self.memory_novelty_gain <= 1.0:
            raise ValueError("memory_novelty_gain must be in [0, 1]")
        if not 0.0 <= self.memory_reward_gain <= 1.0:
            raise ValueError("memory_reward_gain must be in [0, 1]")
        if self.memory_novelty_gain + self.memory_reward_gain <= 0.0:
            raise ValueError("at least one episodic write gate must be active")
        if self.replay_noise_scale < 0.0:
            raise ValueError("replay_noise_scale cannot be negative")
        if self.replay_fatigue_gain < 0.0:
            raise ValueError("replay_fatigue_gain cannot be negative")
        if not 0.0 <= self.replay_value_weight <= 1.0:
            raise ValueError("replay_value_weight must be in [0, 1]")
        if not 0.0 <= self.replay_priority_threshold < 1.0:
            raise ValueError("replay_priority_threshold must be in [0, 1)")
        if self.replay_maturity_ticks < 0:
            raise ValueError("replay_maturity_ticks cannot be negative")
        if not 0.0 <= self.replay_outcome_fast_scale <= 1.0:
            raise ValueError("replay_outcome_fast_scale must be in [0, 1]")
        if not 0.0 <= self.replay_outcome_slow_scale <= 1.0:
            raise ValueError("replay_outcome_slow_scale must be in [0, 1]")
        if self.replay_burst_repeats <= 0:
            raise ValueError("replay_burst_repeats must be positive")
        if self.replay_write_repeats <= 0:
            raise ValueError("replay_write_repeats must be positive")
        if not 0.0 < self.replay_winner_resource_retention <= 1.0:
            raise ValueError("replay_winner_resource_retention must be in (0, 1]")
        if not 0.0 <= self.structural_turnover_ratio <= 1.0:
            raise ValueError("structural_turnover_ratio must be in [0, 1]")
        if not 0.0 < self.structural_capture_target <= 1.0:
            raise ValueError("structural_capture_target must be in (0, 1]")
        if self.structural_error_threshold < 0.0:
            raise ValueError("structural_error_threshold must be non-negative")
        if self.world_calibration_history_limit <= 0:
            raise ValueError("world_calibration_history_limit must be positive")
        if self.cognitive_lineage_history_limit <= 0:
            raise ValueError("cognitive_lineage_history_limit must be positive")
        if self.recovery_archive_capacity <= 0:
            raise ValueError("recovery_archive_capacity must be positive")
        if self.recovery_strategy_evidence_threshold <= 0:
            raise ValueError("recovery_strategy_evidence_threshold must be positive")
        if self.recovery_strategy_memory_budget <= 0.0:
            raise ValueError("recovery_strategy_memory_budget must be positive")
        if self.recovery_strategy_cross_reader_credit_revision_history_limit <= 0:
            raise ValueError(
                "recovery_strategy_cross_reader_credit_revision_history_limit must be positive"
            )
        for name in (
            "recovery_strategy_interaction_residual_tolerance",
            "recovery_strategy_interaction_order_tolerance",
            "recovery_strategy_cross_reader_credit_drift_tolerance",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        strategy_weights = (
            self.recovery_strategy_evidence_weight,
            self.recovery_strategy_consistency_weight,
            self.recovery_strategy_resource_weight,
        )
        if (
            any(float(weight) <= 0.0 for weight in strategy_weights)
            or abs(sum(float(weight) for weight in strategy_weights) - 1.0) > 1e-6
        ):
            raise ValueError("recovery strategy weights must be positive and sum to 1")
        if not 0.0 < self.concept_similarity_threshold <= 1.0:
            raise ValueError("concept_similarity_threshold must be in (0, 1]")
        if (
            len(self.concept_signal_weights) != 3
            or any(float(weight) <= 0.0 for weight in self.concept_signal_weights)
            or abs(sum(float(weight) for weight in self.concept_signal_weights) - 1.0) > 1e-6
        ):
            raise ValueError("concept_signal_weights must be three positive weights summing to 1")
        if self.concept_capacity <= 0:
            raise ValueError("concept_capacity must be positive")
        if not 0.0 < self.concept_plasticity_rate <= 1.0:
            raise ValueError("concept_plasticity_rate must be in (0, 1]")
        if not 0.0 <= self.concept_prune_threshold <= 1.0:
            raise ValueError("concept_prune_threshold must be in [0, 1]")
        if (
            len(self.concept_branch_owner_weights) != 3
            or any(float(weight) <= 0.0 for weight in self.concept_branch_owner_weights)
            or abs(sum(float(weight) for weight in self.concept_branch_owner_weights) - 1.0) > 1e-6
        ):
            raise ValueError(
                "concept_branch_owner_weights must be three positive weights summing to 1"
            )
        if not 0.0 <= self.concept_branch_owner_min_score <= 1.0:
            raise ValueError("concept_branch_owner_min_score must be in [0, 1]")
        if not 0.0 <= self.concept_branch_owner_min_margin <= 1.0:
            raise ValueError("concept_branch_owner_min_margin must be in [0, 1]")
        if self.development_structural_budget < 0:
            raise ValueError("development_structural_budget must be non-negative")
        if not 0.0 < self.self_capability_learning_rate <= 1.0:
            raise ValueError("self_capability_learning_rate must be in (0, 1]")

    @classmethod
    def training_profile(cls, *, scale: int = 2, seed: int = 20260821) -> TaijiConfig:
        """Enlarge regions, dimensions and edge density for corpus training.

        The profile changes capacity only: every dynamics constant and every
        ``__post_init__`` constraint is inherited from the default fabric, so a
        scaled model remains the same architecture with more substrate.
        """

        if scale <= 0:
            raise ValueError("training profile scale must be positive")
        base = cls(seed=seed)
        return cls(
            region_sizes=tuple(size * scale for size in base.region_sizes),
            synapse_fan_in=base.synapse_fan_in * scale,
            motor_fan_in=base.motor_fan_in * scale,
            memory_units=base.memory_units * scale,
            memory_fan_in=base.memory_fan_in * scale,
            memory_readout_fan_in=base.memory_readout_fan_in * scale,
            memory_meta_dim=base.memory_meta_dim * scale,
            memory_time_dim=base.memory_time_dim * scale,
            memory_episode_dim=base.memory_episode_dim * scale,
            perception=base.perception,
            seed=seed,
        )

    @classmethod
    def capacity_profile(
        cls,
        target_active_parameters: int,
        *,
        template: TaijiConfig | None = None,
        policy: CapacityPolicy | None = None,
        seed: int | None = None,
        alignment: int | None = None,
    ) -> TaijiConfig:
        """Build the largest substrate that fits an active-parameter budget.

        The template supplies dynamics.  ``policy`` independently supplies
        structural depth and proportions; when omitted, proportions are
        recovered from the template for backward compatibility.  The returned
        configuration is deterministic and its exact learned-scalar count is
        available before any tensors are allocated.
        """

        target = int(target_active_parameters)
        if target <= 0:
            raise ValueError("target_active_parameters must be positive")
        base = cls(seed=cls.seed if seed is None else int(seed)) if template is None else template
        if policy is None:
            capacity = CapacityPolicy.from_config(
                base,
                alignment=8 if alignment is None else int(alignment),
            )
        else:
            if alignment is not None:
                raise ValueError("alignment belongs to policy when an explicit policy is supplied")
            capacity = policy
        chosen_seed = base.seed if seed is None else int(seed)
        dimension_alignment = int(capacity.alignment)
        first_region_ratio = float(capacity.region_ratios[0])

        def aligned_dimension(value: float, *, minimum: int) -> int:
            units = max(1, int(round(float(value) / dimension_alignment)))
            return max(int(minimum), units * dimension_alignment)

        def candidate(width_units: int) -> TaijiConfig:
            primary_width = int(width_units) * dimension_alignment
            values = base.to_dict()
            values.update(
                {
                    "region_sizes": [
                        aligned_dimension(
                            primary_width * float(ratio) / first_region_ratio,
                            minimum=max(2, dimension_alignment),
                        )
                        for ratio in capacity.region_ratios
                    ],
                    "synapse_fan_in": max(
                        1,
                        int(round(primary_width * capacity.synapse_fan_in_ratio)),
                    ),
                    "motor_fan_in": max(
                        1,
                        int(round(primary_width * capacity.motor_fan_in_ratio)),
                    ),
                    "memory_units": aligned_dimension(
                        primary_width * capacity.memory_units_ratio,
                        minimum=max(2, dimension_alignment),
                    ),
                    "memory_fan_in": max(
                        1,
                        int(round(primary_width * capacity.memory_fan_in_ratio)),
                    ),
                    "memory_meta_dim": aligned_dimension(
                        primary_width * capacity.memory_meta_ratio,
                        minimum=max(2, dimension_alignment),
                    ),
                    "memory_readout_fan_in": max(
                        1,
                        int(round(primary_width * capacity.memory_readout_fan_in_ratio)),
                    ),
                    "memory_time_dim": aligned_dimension(
                        primary_width * capacity.memory_time_ratio,
                        minimum=2,
                    ),
                    "memory_episode_dim": aligned_dimension(
                        primary_width * capacity.memory_episode_ratio,
                        minimum=1,
                    ),
                    "lateral_fan_in": max(
                        1,
                        int(round(primary_width * capacity.lateral_fan_in_ratio)),
                    ),
                    "seed": chosen_seed,
                }
            )
            values["memory_meta_dim"] = min(
                int(values["memory_meta_dim"]),
                int(values["memory_units"]),
            )
            values["memory_readout_fan_in"] = min(
                int(values["memory_readout_fan_in"]),
                int(values["memory_meta_dim"]),
            )
            return cls.from_dict(values)

        smallest = candidate(1)
        if smallest.planned_active_parameter_count > target:
            raise ValueError(
                "target_active_parameters is below the smallest valid aligned fabric "
                f"({smallest.planned_active_parameter_count})"
            )

        lower = 1
        upper = 2
        while candidate(upper).planned_active_parameter_count <= target:
            lower = upper
            upper *= 2

        while upper - lower > 1:
            middle = (lower + upper) // 2
            if candidate(middle).planned_active_parameter_count <= target:
                lower = middle
            else:
                upper = middle
        return candidate(lower)

    @property
    def planned_active_parameter_count(self) -> int:
        """Exact number of learned scalars allocated by this configuration."""

        lower_sizes = (self.alphabet_size, *self.region_sizes[:-1])
        fabric = 0
        for lower_size, region_size in zip(lower_sizes, self.region_sizes, strict=False):
            fabric += lower_size * min(self.synapse_fan_in, region_size)
            fabric += lower_size * region_size
            fabric += region_size * min(self.synapse_fan_in, region_size - 1)
            fabric += region_size * min(self.lateral_fan_in, region_size - 1)

        motor = self.alphabet_size * self.motor_context_dim + self.alphabet_size
        readout_width = min(self.memory_readout_fan_in, self.memory_meta_dim)
        readout_outputs = (
            2 * self.alphabet_size
            + 2
            + self.cortical_context_dim
            + self.memory_time_dim
            + self.memory_episode_dim
            + 4
        )
        memory = self.memory_units * min(self.memory_fan_in, self.memory_units - 1)
        memory += readout_outputs * readout_width
        # The local action decoder is allocated even when the shared decoder
        # is selected, so both modes can be compared without changing the
        # rest of the topology or checkpoint shape.
        memory += self.alphabet_size * min(self.memory_readout_fan_in, self.memory_units)
        memory += self.alphabet_size * min(self.memory_readout_fan_in, self.memory_units)
        return int(fabric + motor + memory)

    @property
    def cortical_context_dim(self) -> int:
        return 2 * sum(self.region_sizes)

    @property
    def motor_context_dim(self) -> int:
        return self.motor_fan_in

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["region_sizes"] = list(self.region_sizes)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TaijiConfig:
        values = dict(payload)
        values["region_sizes"] = tuple(values["region_sizes"])
        if "concept_signal_weights" in values:
            values["concept_signal_weights"] = tuple(values["concept_signal_weights"])
        if "concept_branch_owner_weights" in values:
            values["concept_branch_owner_weights"] = tuple(values["concept_branch_owner_weights"])
        perception = values.get("perception")
        if perception is not None and not isinstance(perception, PerceptionConfig):
            values["perception"] = PerceptionConfig.from_dict(dict(perception))
        return cls(**values)
