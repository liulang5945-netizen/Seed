"""End-to-end native Taiji byte-stream learner and generator."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from .config import TaijiConfig, validate_episodic_learning_target
from .fabric import TaijiFabric
from .identity_organ import (
    IDENTITY_ORGAN_UNBOUND_PROVENANCE,
    CueIdentityOrgan,
    IdentityRecall,
)
from .internalization import content_digest
from .memory import EpisodicField
from .organs import (
    ByteMotor,
    BytePredictiveContext,
    BytePredictiveReadout,
    ByteSensor,
    GatedMultiTimescaleTemporalResidual,
)
from .state import (
    PendingAction,
    PendingExperience,
    RegionState,
    TaijiConsolidation,
    TaijiDecision,
    TaijiOutcome,
    TaijiState,
    TaijiStep,
)
from .workbench_boundary import (
    WorkbenchBoundaryAuthorization,
    WorkbenchTaskBoundary,
    select_readout_generation,
)


class Taiji:
    """Complete sensor → predictive fabric ↔ episodic field → motor path.

    Learning occurs online at local predictive, memory and motor synapses.
    The class intentionally exposes no loss.backward() or optimizer contract.
    """

    # v10 gives F1 a private plastic temporal context.  v8/v9 stay readable
    # only through the one-way migration below; writing either old format
    # again would conceal which owner received sequence-learning updates.
    CHECKPOINT_FORMAT = "taiji-native-v10"
    LEGACY_CHECKPOINT_FORMATS = frozenset({"taiji-native-v8", "taiji-native-v9"})
    STATE_VERSION = 7
    LEGACY_STATE_VERSIONS = frozenset({5, 6})
    IDENTITY_GROWTH_FORMAT = "taiji-native-identity-growth-v1"
    GATED_TEMPORAL_CANDIDATE_KEY = "gated_temporal_candidate"
    GATED_TEMPORAL_CANDIDATE_SEED_OFFSET = 5927
    READOUT_REGISTRY_FORMAT = "taiji-predictive-readout-registry-v1"
    READOUT_REGISTRY_VERSION = 1

    def __init__(
        self,
        config: TaijiConfig | None = None,
        *,
        device: torch.device | str = "cpu",
        episode_id: str = "episode-0",
    ) -> None:
        self.config = config or TaijiConfig()
        self.device = torch.device(device)
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(self.config.seed)
        self.sensor = ByteSensor(self.config, device=self.device)
        self.fabric = TaijiFabric(self.config, generator=self._rng, device=self.device)
        self.motor = ByteMotor(self.config, generator=self._rng, device=self.device)
        # This generator is deliberately independent of ``self._rng``.  The
        # M2-2f organ must not re-seed existing fabric/memory/identity topology
        # merely by being introduced into an old checkpoint lineage.
        predictive_rng = torch.Generator(device="cpu")
        predictive_rng.manual_seed(
            int(self.config.seed) + int(self.config.predictive_readout_seed_offset)
        )
        self.predictive_readout = BytePredictiveReadout(
            self.config,
            generator=predictive_rng,
            device=self.device,
        )
        predictive_context_rng = torch.Generator(device="cpu")
        predictive_context_rng.manual_seed(
            int(self.config.seed) + int(self.config.predictive_context_seed_offset)
        )
        self.predictive_context = BytePredictiveContext(
            self.config,
            generator=predictive_context_rng,
            device=self.device,
        )
        # Optional R2 candidate.  It is lazy and absent from legacy
        # checkpoints, so the default v10 path retains its original payload
        # and prediction behavior exactly.
        self._gated_temporal_candidate: GatedMultiTimescaleTemporalResidual | None = None
        self._memory_rng = torch.Generator(device="cpu")
        self._memory_rng.set_state(self._rng.get_state().clone())
        self.memory = EpisodicField(self.config, generator=self._memory_rng, device=self.device)
        self.identity_organ = (
            CueIdentityOrgan(self.config, generator=self._rng, device=self.device)
            if self.config.identity_organ_enabled
            else None
        )
        # Lifetime development ticks survive episode boundaries: state.tick
        # restarts at every reset_dynamics, so it cannot carry the replay
        # maturity gate -- an experienced text resets it to tens of ticks and
        # a lived checkpoint would read as a fresh field (A2, diagnosis 24).
        self._development_ticks = 0
        self._identity_growth_history: list[dict[str, Any]] = []
        self._last_generation_route: dict[str, Any] | None = None
        # The protected decoder is the stable parent.  An active decoder is a
        # separately owned, explicitly registered branch; keeping the object
        # out of the default path makes accidental training of the active
        # branch impossible without a Workbench boundary.
        self._active_predictive_readout: BytePredictiveReadout | None = None
        self._active_predictive_readout_metadata: dict[str, Any] | None = None
        self._state = self._initial_state(episode_id)

    def _initial_state(self, episode_id: str) -> TaijiState:
        if not episode_id:
            raise ValueError("episode_id cannot be empty")
        uniform = torch.full(
            (self.config.alphabet_size,),
            1.0 / self.config.alphabet_size,
            device=self.device,
        )
        return TaijiState(
            version=self.STATE_VERSION,
            tick=0,
            episode_id=episode_id,
            regions=self.fabric.initial_state(),
            memory=self.memory.initial_state(),
            motor_context=torch.zeros(self.config.motor_context_dim, device=self.device),
            predictive_context_trace=torch.zeros(self.config.motor_context_dim, device=self.device),
            motor_probabilities=uniform,
            readout_kind="action",
            last_symbol=None,
            pending_action=None,
            pending_experience=None,
        )

    @property
    def tick(self) -> int:
        return self._state.tick

    @property
    def last_generation_route(self) -> dict[str, Any] | None:
        """Return the latest boundary-to-readout audit without exposing tensors."""

        return None if self._last_generation_route is None else dict(self._last_generation_route)

    @property
    def active_predictive_readout_metadata(self) -> dict[str, Any] | None:
        """Return active-branch ownership metadata without exposing tensors."""

        if self._active_predictive_readout_metadata is None:
            return None
        metadata = dict(self._active_predictive_readout_metadata)
        if self._active_predictive_readout is not None:
            metadata["readout_digest"] = content_digest(
                self._active_predictive_readout.to_payload()
            )
        return metadata

    @property
    def gated_temporal_candidate_enabled(self) -> bool:
        """Whether the opt-in R2 multi-timescale temporal candidate is attached."""

        return self._gated_temporal_candidate is not None

    def _new_gated_temporal_candidate(
        self,
        *,
        fast_decay: float,
        slow_decay: float,
        gate_temperature: float,
        residual_gain: float,
    ) -> GatedMultiTimescaleTemporalResidual:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            int(self.config.seed) + int(self.GATED_TEMPORAL_CANDIDATE_SEED_OFFSET)
        )
        return GatedMultiTimescaleTemporalResidual(
            self.config,
            generator=generator,
            fast_decay=fast_decay,
            slow_decay=slow_decay,
            gate_temperature=gate_temperature,
            residual_gain=residual_gain,
            device=self.device,
        )

    @torch.no_grad()
    def enable_gated_temporal_candidate(
        self,
        *,
        fast_decay: float = 0.50,
        slow_decay: float = 0.98,
        gate_temperature: float = 1.0,
        residual_gain: float = 0.20,
    ) -> dict[str, Any]:
        """Attach the zero-initialized R2 temporal candidate explicitly.

        Attaching it is an experimental state transition, not an implicit
        default.  The candidate starts with zero residual weights and an
        isolated RNG stream, so a generation canary is unchanged until the
        candidate is trained.
        """

        if self._gated_temporal_candidate is not None:
            raise RuntimeError("gated temporal candidate is already enabled")
        if self._state.pending_action is not None or self._state.pending_experience is not None:
            raise RuntimeError("temporal candidate attachment requires a settled state")
        candidate = self._new_gated_temporal_candidate(
            fast_decay=fast_decay,
            slow_decay=slow_decay,
            gate_temperature=gate_temperature,
            residual_gain=residual_gain,
        )
        self._gated_temporal_candidate = candidate
        self._state.predictive_context_slow_trace = torch.zeros(
            self.config.motor_context_dim,
            device=self.device,
        )
        return {
            "format": GatedMultiTimescaleTemporalResidual.PAYLOAD_FORMAT,
            "version": GatedMultiTimescaleTemporalResidual.PAYLOAD_VERSION,
            "enabled": True,
            "fast_decay": candidate.fast_decay,
            "slow_decay": candidate.slow_decay,
            "gate_temperature": candidate.gate_temperature,
            "residual_gain": candidate.residual_gain,
            "candidate_digest": content_digest(candidate.to_payload()),
        }

    @torch.no_grad()
    def disable_gated_temporal_candidate(self) -> None:
        """Remove the opt-in candidate and its optional slow trace."""

        self._gated_temporal_candidate = None
        self._state.predictive_context_slow_trace = None

    @torch.no_grad()
    def zero_gated_temporal_candidate(self) -> tuple[str, str]:
        """Lesion candidate residual weights and return before/after digests."""

        if self._gated_temporal_candidate is None:
            raise RuntimeError("gated temporal candidate is not enabled")
        before = content_digest(self._gated_temporal_candidate.to_payload())
        self._gated_temporal_candidate.fast.edge_weight.zero_()
        self._gated_temporal_candidate.slow.edge_weight.zero_()
        after = content_digest(self._gated_temporal_candidate.to_payload())
        return before, after

    def _protected_readout_parent_digest(self) -> str:
        """Digest the stable owners an active readout is allowed to branch from.

        Dynamics state and the shared sampling RNG are intentionally omitted:
        they change during an episode and are not the semantic parent of a
        learned readout branch.  Every persistent protected owner is included,
        so a branch cannot be restored onto a foreign substrate silently.
        """

        payload: dict[str, Any] = {
            "format": self.CHECKPOINT_FORMAT,
            "config": self.config.to_dict(),
            "fabric": self.fabric.to_payload(),
            "motor": self.motor.to_payload(),
            "predictive_context": self.predictive_context.to_payload(),
            "predictive_readout": self.predictive_readout.to_payload(),
            "memory": self.memory.to_payload(),
        }
        if self._gated_temporal_candidate is not None:
            payload[self.GATED_TEMPORAL_CANDIDATE_KEY] = (
                self._gated_temporal_candidate.to_payload()
            )
        if self.identity_organ is not None:
            payload["identity_organ"] = self.identity_organ.to_payload(
                parent_checkpoint_digest=content_digest(payload),
            )
        return content_digest(payload)

    def readout_registry_status(self) -> dict[str, Any]:
        """Return content-addressed protected/active readout ownership state."""

        active = self.active_predictive_readout_metadata
        return {
            "format": self.READOUT_REGISTRY_FORMAT,
            "version": self.READOUT_REGISTRY_VERSION,
            "protected": {
                "scope": "protected",
                "owner": "predictive_readout",
                "mutable": True,
                "readout_digest": content_digest(self.predictive_readout.to_payload()),
            },
            "active": active,
        }

    @torch.no_grad()
    def register_active_predictive_readout(
        self,
        readout: BytePredictiveReadout,
        *,
        boundary_digest: str,
        parent_checkpoint_digest: str | None = None,
    ) -> dict[str, Any]:
        """Attach one isolated active decoder to this Taiji instance.

        The caller must provide a Workbench boundary digest.  The parent
        digest defaults to the current protected substrate and is checked on
        both registration and restore.  Replacing a branch requires an
        explicit ``clear_active_predictive_readout`` first.
        """

        if not isinstance(readout, BytePredictiveReadout):
            raise TypeError("active readout must be a BytePredictiveReadout")
        if readout is self.predictive_readout:
            raise ValueError("active readout must not alias the protected readout")
        if readout.config != self.config:
            raise ValueError("active readout configuration does not match architecture")
        boundary = str(boundary_digest).strip()
        if not boundary:
            raise ValueError("boundary_digest cannot be empty")
        if self._active_predictive_readout is not None:
            raise RuntimeError("an active predictive readout is already registered")
        expected_parent = self._protected_readout_parent_digest()
        parent = expected_parent if parent_checkpoint_digest is None else str(
            parent_checkpoint_digest
        ).strip()
        if parent != expected_parent:
            raise ValueError("active readout parent does not match protected substrate")
        metadata = {
            "scope": "active",
            "owner": "predictive_readout.active",
            "mutable": True,
            "parent_checkpoint_digest": parent,
            "boundary_digest": boundary,
            "readout_digest": content_digest(readout.to_payload()),
        }
        self._active_predictive_readout = readout
        self._active_predictive_readout_metadata = metadata
        return dict(metadata)

    @torch.no_grad()
    def clone_protected_predictive_readout_as_active(
        self,
        *,
        boundary_digest: str,
    ) -> dict[str, Any]:
        """Create an active branch seeded from the protected decoder."""

        active_rng = torch.Generator(device="cpu")
        active_rng.manual_seed(
            int(self.config.seed)
            + int(self.config.predictive_readout_seed_offset)
            + 1
        )
        active = BytePredictiveReadout(
            self.config,
            generator=active_rng,
            device=self.device,
        )
        active.load_payload(self.predictive_readout.to_payload())
        return self.register_active_predictive_readout(
            active,
            boundary_digest=boundary_digest,
        )

    @torch.no_grad()
    def clear_active_predictive_readout(self) -> None:
        """Drop the active branch explicitly, returning to protected-only state."""

        self._active_predictive_readout = None
        self._active_predictive_readout_metadata = None

    def _predictive_readout_for_scope(
        self,
        scope: str,
        *,
        boundary_digest: str | None = None,
    ) -> BytePredictiveReadout:
        if scope == "protected":
            return self.predictive_readout
        if scope != "active":
            raise ValueError("unsupported predictive readout scope")
        if self._active_predictive_readout is None:
            raise RuntimeError(
                "requested readout generation is not attached to this Taiji instance"
            )
        mounted = str(
            (self._active_predictive_readout_metadata or {}).get("boundary_digest", "")
        ).strip()
        if boundary_digest is not None and mounted and boundary_digest != mounted:
            # A content-valid boundary token is not enough: the caller must
            # consume the exact generation mounted on this instance.  Otherwise
            # a fresh task boundary could silently read a branch that belongs
            # to a different (foreign) registry boundary.
            raise PermissionError(
                "active predictive readout boundary digest does not match mounted branch"
            )
        return self._active_predictive_readout

    def _readout_registry_payload(self) -> dict[str, Any] | None:
        """Serialize the active branch only when it is still parent-compatible."""

        metadata = self.active_predictive_readout_metadata
        if metadata is None or self._active_predictive_readout is None:
            return None
        parent = str(metadata["parent_checkpoint_digest"])
        if parent != self._protected_readout_parent_digest():
            raise RuntimeError(
                "active readout registry parent drifted; close or rebase the active branch"
            )
        readout_payload = self._active_predictive_readout.to_payload()
        return {
            "format": self.READOUT_REGISTRY_FORMAT,
            "version": self.READOUT_REGISTRY_VERSION,
            "parent_checkpoint_digest": parent,
            "generations": [
                {
                    **metadata,
                    "readout_digest": content_digest(readout_payload),
                    "readout": readout_payload,
                }
            ],
        }

    def _restore_readout_registry(self, payload: Any) -> None:
        """Restore an active branch after protected owners have been loaded."""

        self.clear_active_predictive_readout()
        if payload is None:
            return
        if not isinstance(payload, Mapping):
            raise ValueError("predictive readout registry checkpoint payload is invalid")
        allowed = {
            "format",
            "version",
            "parent_checkpoint_digest",
            "generations",
        }
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"predictive readout registry contains unknown fields: {unknown}")
        if payload.get("format") != self.READOUT_REGISTRY_FORMAT:
            raise ValueError("unsupported predictive readout registry format")
        if int(payload.get("version", -1)) != self.READOUT_REGISTRY_VERSION:
            raise ValueError("unsupported predictive readout registry version")
        parent = str(payload.get("parent_checkpoint_digest", "")).strip()
        if parent != self._protected_readout_parent_digest():
            raise ValueError("predictive readout registry parent does not match protected substrate")
        generations = payload.get("generations")
        if not isinstance(generations, list) or len(generations) != 1:
            raise ValueError("predictive readout registry must contain one active generation")
        record = generations[0]
        if not isinstance(record, Mapping):
            raise ValueError("predictive readout registry generation must be a mapping")
        record_allowed = {
            "scope",
            "owner",
            "mutable",
            "parent_checkpoint_digest",
            "boundary_digest",
            "readout_digest",
            "readout",
        }
        unknown = sorted(set(record) - record_allowed)
        if unknown:
            raise ValueError(
                f"predictive readout registry generation contains unknown fields: {unknown}"
            )
        if record.get("scope") != "active" or record.get("owner") != "predictive_readout.active":
            raise ValueError("predictive readout registry generation owner is invalid")
        if record.get("mutable") is not True:
            raise ValueError("active predictive readout must remain mutable")
        if str(record.get("parent_checkpoint_digest", "")).strip() != parent:
            raise ValueError("predictive readout generation parent is inconsistent")
        readout_payload = record.get("readout")
        if not isinstance(readout_payload, Mapping):
            raise ValueError("predictive readout registry readout is invalid")
        expected_readout_digest = content_digest(readout_payload)
        if str(record.get("readout_digest", "")).strip() != expected_readout_digest:
            raise ValueError("predictive readout registry digest does not match readout")
        active_rng = torch.Generator(device="cpu")
        active_rng.manual_seed(
            int(self.config.seed)
            + int(self.config.predictive_readout_seed_offset)
            + 1
        )
        active = BytePredictiveReadout(
            self.config,
            generator=active_rng,
            device=self.device,
        )
        active.load_payload(readout_payload)
        self.register_active_predictive_readout(
            active,
            boundary_digest=str(record.get("boundary_digest", "")),
            parent_checkpoint_digest=parent,
        )

    def snapshot(self) -> TaijiState:
        return self._state.clone()

    def reset_dynamics(self, *, episode_id: str | None = None) -> None:
        """Clear activity while preserving all learned synapses."""

        if self._state.pending_action is not None:
            raise RuntimeError("pending action must be settled before reset")
        if self._state.pending_experience is not None:
            raise RuntimeError("pending experience must observe its outcome before reset")
        self._development_ticks = max(self._development_ticks, int(self._state.tick))
        self.fabric.clear_cue_snapshot()
        self._state = self._initial_state(episode_id or self._state.episode_id)

    @property
    def identity_growth_history(self) -> tuple[dict[str, Any], ...]:
        """Return the append-only identity-capacity growth ledger."""

        return tuple(dict(item) for item in self._identity_growth_history)

    @torch.no_grad()
    def grow_identity_organ(
        self, new_capacity: int, *, reason: str = "capacity-pressure"
    ) -> dict[str, Any]:
        """Append identity slots without resetting any learned substrate."""

        if self.identity_organ is None:
            raise RuntimeError("cannot grow an identity organ that is disabled")
        if self._state.pending_action is not None or self._state.pending_experience is not None:
            raise RuntimeError("identity growth requires a settled dynamics state")
        target = int(new_capacity)
        old_capacity = int(self.identity_organ.capacity)
        if target <= old_capacity:
            raise ValueError("identity organ growth must increase capacity")
        if not str(reason).strip():
            raise ValueError("identity growth reason cannot be empty")
        parent_checkpoint_digest = content_digest(self.checkpoint())
        values = self.config.to_dict()
        values["identity_organ_capacity"] = target
        next_config = TaijiConfig.from_dict(values)
        growth = self.identity_organ.expand_capacity(next_config)
        self.config = next_config
        event = {
            "format": self.IDENTITY_GROWTH_FORMAT,
            "from_capacity": int(growth["from_capacity"]),
            "to_capacity": int(growth["to_capacity"]),
            "preserved_slots": int(growth["preserved_slots"]),
            "appended_slots": int(growth["appended_slots"]),
            "new_generation_start": int(growth["new_generation_start"]),
            "parent_checkpoint_digest": parent_checkpoint_digest,
            "reason": str(reason),
        }
        self._identity_growth_history.append(event)
        return dict(event)

    @torch.no_grad()
    def observe(
        self,
        symbol: int,
        *,
        learn: bool = True,
        learn_motor: bool | None = None,
        learn_fabric: bool | None = None,
        learn_predictive_context: bool | None = None,
        learn_predictive_readout: bool | None = None,
        readout: str = "action",
        use_memory: bool = True,
        use_identity: bool | None = None,
        use_delayed_memory_verdict: bool = False,
        identity_generation_scope: str = "all",
        _predictive_readout: BytePredictiveReadout | None = None,
        _preservation_readout: BytePredictiveReadout | None = None,
        _preservation_strength: float = 0.0,
    ) -> TaijiStep:
        """Advance one sensation tick.

        ``learn_motor=False`` keeps local fabric learning active without
        treating an externally caused sensation as the correct motor action.
        ``learn_fabric=False`` keeps the same forward cortical dynamics while
        withholding only persistent fabric plasticity.  It exists for causal
        sequence-learning experiments: F1 may still train its dedicated
        predictive readout and private temporal context without silently
        changing the shared F2/F4 context.
        ``learn_predictive_context`` and ``learn_predictive_readout`` expose
        the two F1 owners independently for causal continual-learning
        diagnostics.  ``None`` follows ``learn``; an explicit ``False``
        freezes only that owner while preserving the same forward path.
        ``readout="predictive"`` sends next-byte error to the dedicated F1
        decoder; it never writes the F4 action policy.  A single dynamics
        episode cannot silently switch readout ownership, because the prior
        probability would otherwise come from a different organ.
        ``use_identity`` scopes the optional episodic identity evidence to
        action/memory queries; byte prediction and language generation disable
        it explicitly so a cue route cannot contaminate their decoder.
        ``use_delayed_memory_verdict`` is the M1-66 organ-first gate: when
        True, a successfully routed cue owns the decision instead of being one
        input among equals.  Only delayed-memory evaluation paths (B2)
        enable it; continuous/instantaneous kernels stay on the original motor
        synthesis so the organ's default presence cannot hijack their intent.
        """

        if self._state.pending_action is not None:
            raise RuntimeError("pending action must be settled before observation")
        if readout not in {"action", "predictive"}:
            raise ValueError("readout must be 'action' or 'predictive'")
        if _predictive_readout is not None and not isinstance(
            _predictive_readout, BytePredictiveReadout
        ):
            raise TypeError("_predictive_readout must be a BytePredictiveReadout or None")
        if _predictive_readout is not None and readout != "predictive":
            raise ValueError("a predictive readout owner requires readout='predictive'")
        if _preservation_readout is not None and readout != "predictive":
            raise ValueError("a preservation readout requires readout='predictive'")
        if _preservation_readout is not None and not isinstance(
            _preservation_readout, BytePredictiveReadout
        ):
            raise TypeError("_preservation_readout must be a BytePredictiveReadout or None")
        preservation_strength = float(_preservation_strength)
        if not math.isfinite(preservation_strength) or preservation_strength < 0.0:
            raise ValueError("preservation strength must be finite and non-negative")
        if preservation_strength > 0.0 and _preservation_readout is None:
            raise ValueError("positive preservation strength requires a preservation readout")
        if identity_generation_scope not in {"all", "active"}:
            raise ValueError("identity generation scope must be 'all' or 'active'")
        if readout == "predictive" and learn_motor is True:
            raise ValueError("predictive readout cannot train the action motor")
        if readout == "predictive" and use_delayed_memory_verdict:
            raise ValueError("predictive readout cannot request a delayed-memory verdict")
        if learn_fabric is not None and not isinstance(learn_fabric, bool):
            raise TypeError("learn_fabric must be a bool or None")
        if not learn and learn_fabric is True:
            raise ValueError("fabric learning requires learn=True")
        for name, value in (
            ("learn_predictive_context", learn_predictive_context),
            ("learn_predictive_readout", learn_predictive_readout),
        ):
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"{name} must be a bool or None")
            if not learn and value is True:
                raise ValueError(f"{name} requires learn=True")
        if readout != "predictive" and (
            learn_predictive_context is True or learn_predictive_readout is True
        ):
            raise ValueError("predictive owner learning requires readout='predictive'")
        predictive_readout = (
            self.predictive_readout if _predictive_readout is None else _predictive_readout
        )
        if _preservation_readout is predictive_readout:
            raise RuntimeError("preservation readout must be distinct from the trained readout")
        symbol = int(symbol)
        sensory = self.sensor.encode(symbol)
        previous = self._state
        if previous.readout_kind != readout and (
            previous.last_symbol is not None or previous.pending_experience is not None
        ):
            raise RuntimeError(
                "readout changed inside an active dynamics episode; reset before switching"
            )
        motor_learning = learn if learn_motor is None else bool(learn_motor)
        fabric_learning = learn if learn_fabric is None else bool(learn_fabric)
        predictive_context_learning = (
            learn if learn_predictive_context is None else bool(learn_predictive_context)
        )
        predictive_readout_learning = (
            learn if learn_predictive_readout is None else bool(learn_predictive_readout)
        )
        memory_write_strength = 0.0
        if previous.pending_experience is not None:
            pending_experience = previous.pending_experience
            if pending_experience.learn_memory:
                memory_write = self.memory.write(
                    pending_experience.cortical_context,
                    action_symbol=pending_experience.action_symbol,
                    reward=pending_experience.reward,
                    outcome_symbol=symbol,
                    tick=pending_experience.tick,
                    episode_id=pending_experience.episode_id,
                    provenance=pending_experience.provenance,
                    learning_scale=pending_experience.memory_learning_scale,
                    learning_targets=pending_experience.memory_learning_targets,
                    threshold=previous.memory.threshold,
                )
                memory_write_strength = memory_write.strength
                if self.identity_organ is not None:
                    self.identity_organ.learn(
                        pending_experience.cortical_context,
                        pending_experience.action_symbol,
                        outcome_symbol=symbol,
                        reward=pending_experience.reward,
                    )

        prior_prediction: int | None = None
        prior_probability: float | None = None
        surprise: float | None = None
        if previous.last_symbol is not None:
            prior_prediction = int(previous.motor_probabilities.argmax().item())
            prior_probability = float(previous.motor_probabilities[symbol].item())
            surprise = -math.log(max(prior_probability, 1e-12))
            if readout == "predictive" and learn and (
                predictive_readout_learning or predictive_context_learning
            ):
                # Take the F1 feedback before changing decoder contacts: the
                # private residual must learn from the causal surface that
                # made this prior prediction, never from the post-update one.
                predictive_error = predictive_readout.prediction_error(
                    previous.motor_probabilities,
                    symbol,
                )
                if predictive_readout_learning:
                    preservation_probabilities = None
                    if _preservation_readout is not None and preservation_strength > 0.0:
                        preservation_probabilities = _preservation_readout.probabilities(
                            previous.motor_context
                        )
                    predictive_readout.learn(
                        previous.motor_context,
                        previous.motor_probabilities,
                        symbol,
                        preservation_probabilities=preservation_probabilities,
                        preservation_strength=preservation_strength,
                    )
                if predictive_context_learning:
                    predictive_feedback = predictive_readout.context_feedback(
                        predictive_error
                    )
                    if self._gated_temporal_candidate is None:
                        self.predictive_context.learn(
                            previous.predictive_context_trace,
                            predictive_feedback,
                        )
                    else:
                        self._gated_temporal_candidate.learn(
                            previous.motor_context,
                            previous.predictive_context_slow_trace,
                            predictive_feedback,
                            learning_rate=self.config.predictive_context_learning_rate,
                            weight_decay=self.config.synapse_decay,
                        )
            elif readout == "action" and motor_learning:
                self.motor.learn(
                    previous.motor_context,
                    previous.motor_probabilities,
                    symbol,
                )

        regions, activity_rates, error_norms = self.fabric.step(
            sensory,
            previous.regions,
            learn=fabric_learning,
            episodic_feedback=previous.memory.cortical_feedback,
        )
        cortical_state = self.fabric.cortical_context(regions)
        identity_recall: IdentityRecall | None = None
        identity_evidence: torch.Tensor | None = None
        identity_addressing_used = False
        if self.identity_organ is not None:
            identity_enabled = use_memory and (use_identity is None or bool(use_identity))
            identity_recall = self.identity_organ.recall(
                cortical_state,
                enabled=identity_enabled,
                generation_scope=identity_generation_scope,
            )
            if identity_recall.used:
                # M1-65: a successfully routed cue freezes the addressing key.
                # The identity organ and the episodic field route on the live
                # cortical context; freezing it here means interference symbols
                # observed later cannot wash the cue away, so a delayed read
                # addresses the same key it wrote under.
                self.fabric.stamp_cue_snapshot(cortical_state)
                identity_addressing_used = True
                identity_evidence = (
                    float(self.config.identity_organ_evidence_gain)
                    * identity_recall.action_evidence
                )
        memory_state, memory_recall = self.memory.recall(
            cortical_state,
            previous.memory,
            use_long_term=use_memory,
        )
        if readout == "predictive":
            prior_predictive_context = (
                previous.motor_context if previous.readout_kind == "predictive" else None
            )
            base_context, predictive_context_trace = self.predictive_context.encode(
                self.fabric.predictive_context(regions),
                prior_context=prior_predictive_context,
            )
            if self._gated_temporal_candidate is None:
                context = base_context
                predictive_context_slow_trace = None
            else:
                context, predictive_context_slow_trace = (
                    self._gated_temporal_candidate.encode(
                        base_context,
                        prior_context=prior_predictive_context,
                        prior_slow_context=(
                            previous.predictive_context_slow_trace
                            if previous.readout_kind == "predictive"
                            else None
                        ),
                    )
                )
        else:
            context = self.motor.encode_context(self.fabric.predictive_context(regions))
            predictive_context_trace = torch.zeros(
                self.config.motor_context_dim,
                device=self.device,
            )
            predictive_context_slow_trace = None
        cortical_prediction_evidence = float(
            self.config.consolidation_read_gain
        ) * self.fabric.consolidated_decode(0, regions[0].trace)
        episodic_evidence = cortical_prediction_evidence + (
            self.config.memory_read_gain * memory_recall.confidence * memory_recall.action_evidence
        )
        if identity_evidence is not None:
            episodic_evidence = episodic_evidence + identity_evidence
        if readout == "predictive":
            probabilities = predictive_readout.probabilities(
                context,
                episodic_evidence=episodic_evidence,
            )
        elif identity_addressing_used and use_delayed_memory_verdict:
            # M1-66: once the identity organ routes a cue, it owns the value
            # readout.  The motor's shared readout and the episodic field's
            # value head both stay chance-level on foundation-scale courses
            # (EpisodicField 0.462/0.769 measured), so an equal-weight sum would
            # bury the organ's correct value under high-confidence wrong
            # evidence.  Routing success therefore makes the organ's own
            # action distribution the decision, not one input among equals;
            # the motor and memory evidence are still computed (they feed
            # prediction/surprise/consolidation) but do not dilute the verdict.
            probabilities = identity_recall.action_probabilities
        elif (
            use_delayed_memory_verdict
            and identity_recall is not None
            and identity_recall.provenance == IDENTITY_ORGAN_UNBOUND_PROVENANCE
        ):
            # The organ is on the default path and was asked, but no prototype
            # matched.  A routed read that cannot bind its cue must not fabricate
            # a confident verdict from the shared readout, so the decision is a
            # near-uniform distribution: "no recall" is a real answer, and the
            # motor/memory synthesis is not allowed to leak a high-confidence
            # guess into the margin audit.  Disabled reads (B1's
            # use_identity=False, and the lesion arms) emit the DISABLED
            # provenance and stay on the original motor synthesis untouched.
            probabilities = torch.full(
                (self.config.alphabet_size,),
                1.0 / self.config.alphabet_size,
                device=self.device,
            )
        else:
            probabilities = self.motor.probabilities(
                context,
                episodic_evidence=episodic_evidence,
            )
        predicted_symbol = int(probabilities.argmax().item())
        self._state = TaijiState(
            version=self.STATE_VERSION,
            tick=previous.tick + 1,
            episode_id=previous.episode_id,
            regions=regions,
            memory=memory_state,
            motor_context=context,
            predictive_context_trace=predictive_context_trace,
            motor_probabilities=probabilities,
            readout_kind=readout,
            last_symbol=symbol,
            pending_action=None,
            pending_experience=None,
            predictive_context_slow_trace=predictive_context_slow_trace,
        )
        return TaijiStep(
            tick=previous.tick,
            observed_symbol=symbol,
            predicted_symbol=predicted_symbol,
            probabilities=probabilities.detach().clone(),
            prior_prediction=prior_prediction,
            prior_probability=prior_probability,
            surprise=surprise,
            activity_rates=activity_rates,
            local_error_norms=error_norms,
            memory_recall=memory_recall,
            memory_write_strength=float(memory_write_strength),
            identity_recall=identity_recall,
        )

    @torch.no_grad()
    def act(
        self,
        available_actions: Sequence[int],
        *,
        sample: bool = True,
    ) -> TaijiDecision:
        """Select one afforded action and preserve its local eligibility."""

        if self._state.pending_action is not None:
            raise RuntimeError("pending action must be settled before acting again")
        if self._state.pending_experience is not None:
            raise RuntimeError("pending experience must observe its outcome before acting")
        if self._state.readout_kind != "action":
            raise RuntimeError("action requires an action-readout observation after reset")
        actions = tuple(int(value) for value in available_actions)
        if not actions:
            raise ValueError("available_actions cannot be empty")
        if len(set(actions)) != len(actions):
            raise ValueError("available_actions cannot contain duplicates")
        if any(not 0 <= value < self.config.alphabet_size for value in actions):
            raise ValueError("available action is outside the motor alphabet")

        indices = torch.tensor(actions, device=self.device, dtype=torch.long)
        restricted = self._state.motor_probabilities[indices]
        restricted = restricted / restricted.sum().clamp_min(1e-12)
        policy = torch.zeros(self.config.alphabet_size, device=self.device)
        policy[indices] = restricted
        if sample:
            local_index = int(
                torch.multinomial(restricted.detach().cpu(), 1, generator=self._rng).item()
            )
        else:
            local_index = int(restricted.argmax().item())
        action_symbol = actions[local_index]
        pending = PendingAction(
            tick=self._state.tick,
            action_symbol=action_symbol,
            available_actions=actions,
            context=self._state.motor_context.detach().clone(),
            policy_probabilities=policy.detach().clone(),
        )
        self._state.pending_action = pending
        return TaijiDecision(
            tick=pending.tick,
            action_symbol=pending.action_symbol,
            available_actions=pending.available_actions,
            policy_probabilities=pending.policy_probabilities.detach().clone(),
        )

    @torch.no_grad()
    def settle_action(
        self,
        reward: float,
        *,
        learn: bool = True,
        learn_memory: bool | None = None,
        provenance: str = "experienced",
        memory_learning_scale: float = 1.0,
        memory_learning_targets: str = "all",
    ) -> TaijiOutcome:
        """Consume the pending action with a scalar environment outcome."""

        pending = self._state.pending_action
        if pending is None:
            raise RuntimeError("no pending action to settle")
        reward = float(reward)
        if not math.isfinite(reward):
            raise ValueError("reward must be finite")
        if provenance not in self.memory.PROVENANCE_KINDS:
            raise ValueError(f"unsupported episodic provenance: {provenance}")
        if not math.isfinite(float(memory_learning_scale)) or float(memory_learning_scale) <= 0.0:
            raise ValueError("memory_learning_scale must be finite and positive")
        validate_episodic_learning_target("memory_learning_targets", memory_learning_targets)
        modulation = reward - self.motor.reward_baseline
        error_norm = 0.0
        if learn:
            error, modulation = self.motor.learn_reward(
                pending.context,
                pending.policy_probabilities,
                pending.action_symbol,
                reward,
            )
            error_norm = float(error.norm().item())
        self._state.pending_action = None
        self._state.pending_experience = PendingExperience(
            tick=pending.tick,
            action_symbol=pending.action_symbol,
            reward=reward,
            cortical_context=self.fabric.cortical_context(self._state.regions).detach().clone(),
            episode_id=self._state.episode_id,
            provenance=provenance,
            learn_memory=(bool(learn) if learn_memory is None else bool(learn_memory)),
            memory_learning_scale=float(memory_learning_scale),
            memory_learning_targets=memory_learning_targets,
        )
        return TaijiOutcome(
            tick=pending.tick,
            action_symbol=pending.action_symbol,
            reward=reward,
            reward_prediction_error=float(modulation),
            learning_error_norm=error_norm,
        )

    @torch.no_grad()
    def consolidate(
        self,
        *,
        cycles: int = 1,
        learn: bool = True,
        replay_cue_chain: bool = True,
    ) -> TaijiConsolidation:
        """Sleep on what the field already holds, with no external input at all.

        Each cycle asks the episodic field to spontaneously regenerate one engram
        from its own value axis, clock, residual trace and noise.  The field's own
        priority gate decides whether that reactivation is worth anything; only
        accepted ones are replayed through the very same predictive fabric.  The
        field first reinstates its cortical cue with no external sensation and
        writes the recalled action from that settled basis; the action is then
        driven until the fabric settles and the recalled outcome is written from
        that basis.  The fabric learns from ordinary local prediction errors,
        scaled by how strongly the field vouched for the engram.

        No external replay list, no teacher target, and no weight is ever copied
        from the field into the fabric: the only channel is the episodic feedback
        gain that waking observation already uses.  The field itself is left
        unchanged, so a self generated pattern cannot reinforce itself.
        """

        if cycles <= 0:
            raise ValueError("consolidation cycles must be positive")
        if self._state.pending_action is not None:
            raise RuntimeError("pending action must be settled before consolidation")
        if self._state.pending_experience is not None:
            raise RuntimeError("pending experience must be observed before consolidation")
        if self.memory.write_count <= 0:
            raise RuntimeError("consolidation requires at least one episodic write")

        replayed_index = self.memory.PROVENANCE_KINDS.index("replayed")
        state = self._state
        regions = state.regions
        memory_state = state.memory
        tick = int(state.tick)
        # The maturity gate reads the lifetime counter, refreshed here so a
        # checkpoint loaded straight into sleep still counts as lived.
        self._development_ticks = max(self._development_ticks, tick)
        structural_before = int(self.fabric.structural_events)
        winner_resource = torch.ones(self.config.alphabet_size, device=self.device)

        accepted = 0
        priority_sum = 0.0
        novelty_sum = 0.0
        value_sum = 0.0
        confidence_sum = 0.0
        replayed_sum = 0.0
        error_sum = 0.0
        error_count = 0

        for _ in range(int(cycles)):
            memory_state, replay = self.memory.replay(
                memory_state,
                tick=tick,
                generator=self._memory_rng,
            )
            tick += 1
            priority_sum += replay.priority
            novelty_sum += replay.novelty
            value_sum += replay.value
            confidence_sum += memory_state.last_confidence
            replayed_sum += float(replay.provenance_probabilities[replayed_index].item())
            if not replay.accepted:
                continue
            accepted += 1
            threshold = float(self.config.replay_priority_threshold)
            endorsement = min(1.0, replay.priority / threshold) if threshold > 0.0 else 1.0
            learn_scale = float(self.config.replay_learning_scale) * endorsement if learn else 0.0
            # Topology and slow-store writes both move against the dream-basis
            # error, which reads the waking decoder off its distribution.  In a
            # crowded field the shared readout rows no longer locate any one
            # engram, so a marginally accepted reactivation rewires rows and
            # fits decoders against garbage error and the whole waking panel
            # pays for it (A2, phase 3).  Field maturity is the separator: a
            # fresh one-shot field keeps the lottery repair and the outcome
            # leg of the contingency store, while a field that has lived
            # through a corpus freezes both.  Neither the episodic write
            # count nor state.tick can carry this gate -- both restart at
            # every session/episode boundary, so a lived field's first
            # consolidation would read as fully trusted and rewire against
            # garbage error anyway (A2, diagnosis 21/24); only the lifetime
            # development counter survives the reset.
            field_trusted = self._development_ticks < int(self.config.replay_maturity_ticks)
            # The engram is read at its mode, not sampled.  The readouts are
            # softmaxes over the whole 257 byte alphabet, so a correct but
            # low-margin reactivation still looks nearly uniform: measured peak
            # mass is ~0.03 and only ~9% of the mass sits on the task vocabulary.
            # Sampling that distribution discards the ordering that carries all
            # of the information -- the mode recovers the true action/outcome
            # pair on every accepted tick, while multinomial draws recovered it
            # on none.  Stochastic exploration belongs in the seed, which is
            # already noise driven; the read-out stage must stay faithful.
            burst = tuple(
                int(probabilities.argmax().item())
                for probabilities in (
                    replay.action_probabilities,
                    replay.outcome_probabilities,
                )
            )
            # A replayed contingency is only worth anything if waking can read it
            # back, and waking arrives with the action alone.  That fixes the
            # shape of the burst completely.
            #
            # The eligibility carrier is the previous tick's slow trace, so the
            # basis a write lands on is whatever the fabric was holding one tick
            # earlier.  If the burst alternated action and outcome, that basis
            # would carry the preceding outcome's residue -- the trace decays at
            # 0.82, so the carryover dominates -- and no probe could reproduce it
            # without being shown the outcome, which would be leakage.  The
            # action is therefore driven on its own from a cleared state, and the
            # outcome is presented exactly once, at the end, as the only writing
            # tick.  The basis is then a pure function of the action.
            #
            # It also has to be the *settled* basis rather than the first
            # transient.  Region 0 is sparse: one tick after a clear leaves 4-9
            # of its 64 units active, while each decoder row draws a fixed fan-in
            # of 16 of those 64.  A 4-unit trace can miss a row's support
            # outright -- measured fan-in energy overlap ran from 50% down to a
            # hard 0%, which makes the corresponding pair unlearnable as
            # arithmetic, not as a matter of dose.  Driving the action until the
            # trace settles brings it to 21-34 active units, which every row's
            # support intersects.
            action_symbol, outcome_symbol = burst
            winner_gain = float(winner_resource[action_symbol].item())
            winner_resource[action_symbol].mul_(float(self.config.replay_winner_resource_retention))

            if replay_cue_chain:
                # Hippocampal-style cortical reinstatement is an activity path,
                # not a memory-to-weight copy.  With no external sensation, the
                # recalled cortical projection settles a cue basis through the
                # same fabric dynamics.  Pinning that basis while presenting the
                # recalled action applies the same local next-sensation rule as
                # the action->outcome phase below.
                cleared = self.fabric.clear_dynamics(regions)
                confidence = max(1e-8, float(replay.familiarity * replay.resonance))
                reinstated = replay.cortical_projection / confidence
                fast_offset = 0
                trace_offset = sum(self.config.region_sizes)
                cue_states = []
                for region_size, previous_region in zip(
                    self.config.region_sizes, cleared, strict=False
                ):
                    activity = torch.relu(reinstated[fast_offset : fast_offset + region_size])
                    trace = reinstated[
                        trace_offset + fast_offset : trace_offset + fast_offset + region_size
                    ]
                    # The cortical readout regresses a unit-normalised context,
                    # so the reinstated slices carry identity but not
                    # magnitude: measured reinstated traces sit near norm 0.05
                    # while consolidation decoders trained on waking bases grow
                    # row weights two orders of magnitude larger.  Direction is
                    # the memory; rescaling each slice to its native bound
                    # reinstates the basis at waking scale so the decoder fit
                    # lands where evaluation will read it.
                    activity_norm = float(activity.norm().item())
                    if activity_norm > 1e-8:
                        activity = activity * (float(self.config.max_membrane_norm) / activity_norm)
                    trace_norm = float(trace.norm().item())
                    if trace_norm > 1e-8:
                        trace = trace * (float(self.config.max_trace_norm) / trace_norm)
                    cue_states.append(
                        RegionState(
                            membrane=activity.detach().clone(),
                            activity=activity.detach().clone(),
                            trace=trace.detach().clone(),
                            prediction=torch.zeros_like(previous_region.prediction),
                            error=torch.zeros_like(previous_region.error),
                            threshold=previous_region.threshold.detach().clone(),
                            inhibition=torch.zeros_like(previous_region.inhibition),
                        )
                    )
                    fast_offset += region_size
                cue_settled = tuple(cue_states)
                tick += 1
                action_activity = self.sensor.encode(action_symbol)
                # The cue-chain write is the last dream-basis update left on a
                # lived field, and diagnosis 23 measured it alone dragging the
                # whole panel down (-0.23) while the very same night without it
                # improved every group (+0.13): the reinstated basis carries
                # the engram's identity but not the corpus statistics the slow
                # decoder already fitted, so writes from it fit the decoder to
                # dream garbage.  The maturity gate therefore covers this leg
                # too -- a fresh toy field keeps the mechanism M7 probes for,
                # a lived field rehearses without rewriting.
                cue_learn_scale = learn_scale * winner_gain if field_trusted else 0.0
                for _ in range(int(self.config.replay_write_repeats)):
                    regions, _rates, error_norms = self.fabric.step(
                        action_activity,
                        cue_settled,
                        learn=learn,
                        episodic_feedback=replay.cortical_projection,
                        learn_scale=0.0,
                        consolidation_learn_scale=cue_learn_scale,
                        use_consolidated=False,
                        adapt_homeostasis=False,
                    )
                    tick += 1
                    error_sum += sum(error_norms) / len(error_norms)
                    error_count += 1

            regions = self.fabric.clear_dynamics(regions)
            for _ in range(int(self.config.replay_burst_repeats)):
                regions, _rates, error_norms = self.fabric.step(
                    self.sensor.encode(action_symbol),
                    regions,
                    learn=False,
                    episodic_feedback=replay.cortical_projection,
                    learn_scale=learn_scale,
                    use_consolidated=False,
                    # ``clear_dynamics`` hands the burst the set point waking
                    # left, which is correct -- sleep must not discard what
                    # waking learned.  But a sixteen-tick burst of one symbol is
                    # not the varied traffic homeostasis integrates over, and
                    # letting it adapt inflated the set point on exactly the
                    # engram's units until the write basis collapsed.  The
                    # threshold is therefore read, not written, for the whole
                    # replay.  See the note in ``fabric.step``.
                    adapt_homeostasis=False,
                )
                tick += 1
                error_sum += sum(error_norms) / len(error_norms)
                error_count += 1
            # The settled state is now held fixed while the outcome is presented
            # repeatedly.  Advancing the fabric between writes would move the
            # eligibility trace off the one basis the probe can reproduce, so
            # every write is driven from the *same* captured state instead.
            #
            # This is not a disguised learning-rate multiplier.  Each step
            # recomputes ``lower_error`` from the current weights, so with the
            # basis pinned the row update becomes an error driven fixed point
            # iteration -- w <- w + lr * (target - w.t) * t / scale -- which
            # converges geometrically onto the contingency instead of drifting
            # linearly.  Raising the rate would overshoot the row-norm bound and
            # amplify whatever noise the first error happened to carry; iterating
            # a contracting map cannot.
            #
            # Recovering these writes matters because sleep is on a far smaller
            # plasticity budget than waking: pretraining lands ~3000 updates at
            # full rate, while a replay pass accepts a few hundred at the gated
            # rate.  Discarding all but one tick of each burst threw away the
            # only lever that does not distort the dynamics.
            settled = regions
            outcome_activity = self.sensor.encode(outcome_symbol)
            # Sleep writes to the slow consolidation pathway, not the fast
            # waking predictor.  The cue-chain phase already holds learn_scale
            # at zero for exactly this reason; the outcome write does the same,
            # scaled by ``replay_outcome_fast_scale``.  Churning the fast
            # decoder/transition/lateral weights on self generated engrams is
            # what dragged the whole frozen panel down after a night (A2), while
            # the slow pathway is the store evaluation reads through.
            fast_learn_scale = learn_scale * float(self.config.replay_outcome_fast_scale)
            # The slow store is read through a basis rescaled to the trace
            # bound, and only the cue-chain phase writes on such a basis (the
            # reinstated projection is rescaled before it enters the fabric).
            # The outcome burst settles on a raw waking-scale trace instead, so
            # writes from it land at a scale the read side does not reproduce:
            # a night of them grows the evidence channel on garbage scale and
            # the whole panel pays for it (A2, phase 3).  The outcome phase
            # therefore trains the slow store only while the field still trusts
            # its own readouts -- toy fields that way keep the action->outcome
            # leg M7 probes for, lived fields stop writing to it.
            slow_learn_scale = (
                learn_scale
                * float(self.config.replay_outcome_slow_scale)
                * winner_gain
                * (1.0 if field_trusted else 0.0)
            )
            for repeat in range(int(self.config.replay_write_repeats)):
                regions, _rates, error_norms = self.fabric.step(
                    outcome_activity,
                    settled,
                    learn=learn,
                    episodic_feedback=replay.cortical_projection,
                    learn_scale=fast_learn_scale,
                    consolidation_learn_scale=slow_learn_scale,
                    use_consolidated=False,
                    # Rewire once, on the opening write, then spend the rest of
                    # the burst growing what was just recruited.  Restructuring
                    # on every repeat would leave the final contact stranded at
                    # the zero weight it opens with, and each swap would discard
                    # the partner the previous one had only begun to train.
                    # Whether topology may move at all is the shared field
                    # trust gate computed above.
                    restructure=(learn and repeat == 0 and field_trusted),
                    adapt_homeostasis=False,
                )
                tick += 1
                error_sum += sum(error_norms) / len(error_norms)
                error_count += 1

        context = self.motor.encode_context(self.fabric.predictive_context(regions))
        self._state = TaijiState(
            version=self.STATE_VERSION,
            tick=tick,
            episode_id=state.episode_id,
            regions=regions,
            memory=memory_state,
            motor_context=context,
            predictive_context_trace=torch.zeros(
                self.config.motor_context_dim,
                device=self.device,
            ),
            motor_probabilities=self.motor.probabilities(context),
            readout_kind="action",
            last_symbol=None,
            pending_action=None,
            pending_experience=None,
        )
        attempts = float(int(cycles))
        return TaijiConsolidation(
            cycles=int(cycles),
            accepted=accepted,
            mean_priority=priority_sum / attempts,
            mean_novelty=novelty_sum / attempts,
            mean_value=value_sum / attempts,
            mean_confidence=confidence_sum / attempts,
            mean_error_norm=(error_sum / error_count) if error_count else 0.0,
            replayed_probability=replayed_sum / attempts,
            structural_events=int(self.fabric.structural_events) - structural_before,
        )

    def learn_bytes(
        self,
        data: bytes,
        *,
        epochs: int = 1,
        include_boundary: bool = True,
        include_start_boundary: bool | None = None,
        include_end_boundary: bool | None = None,
        reset: bool = True,
        use_memory: bool = False,
        learn_fabric: bool = True,
        learn_predictive_context: bool = True,
        learn_predictive_readout: bool = True,
        consolidation_strength: float = 0.0,
        boundary: WorkbenchTaskBoundary | Mapping[str, Any] | None = None,
        authorization: WorkbenchBoundaryAuthorization | None = None,
    ) -> dict[str, float]:
        """Develop on a byte stream using only online local updates.

        Raw-byte prediction is the F1 predictive pathway, not an episodic
        recall task. Keep the long-term episodic field lesioned by default so
        an unrelated F2 memory course cannot silently alter the language
        learning signal. Experiments studying memory-augmented prediction can
        explicitly opt in with ``use_memory=True``.

        ``learn_fabric=False`` is a causal isolation mode: it keeps the live
        forward dynamics and trains the predictive decoder, while leaving the
        shared fabric's persistent synapses and homeostatic statistics intact.

        ``include_start_boundary`` and ``include_end_boundary`` split the
        compatibility ``include_boundary`` switch into independent stream-edge
        markers.  A resumable caller can therefore continue adjacent chunks
        without teaching an artificial boundary between them.  ``reset=False``
        continues the current dynamics episode; it is intended for such a
        chunk continuation and leaves the default whole-stream behavior
        unchanged.

        When a Workbench boundary is supplied, only an ``active`` generation
        may be trained.  The protected generation remains the stable parent;
        active training must therefore explicitly freeze the shared fabric and
        predictive context owners.

        ``consolidation_strength`` is an opt-in M4.R1 candidate.  It keeps the
        protected predictive readout read-only and adds its probability
        distribution as a local preservation signal to active readout updates.
        It is rejected without an explicit active Workbench generation, so the
        default training path and protected owner remain unchanged.
        """

        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if not isinstance(include_boundary, bool):
            raise TypeError("include_boundary must be a bool")
        if include_start_boundary is not None and not isinstance(include_start_boundary, bool):
            raise TypeError("include_start_boundary must be a bool or None")
        if include_end_boundary is not None and not isinstance(include_end_boundary, bool):
            raise TypeError("include_end_boundary must be a bool or None")
        if not isinstance(reset, bool):
            raise TypeError("reset must be a bool")
        if not isinstance(learn_fabric, bool):
            raise TypeError("learn_fabric must be a bool")
        if not isinstance(learn_predictive_context, bool):
            raise TypeError("learn_predictive_context must be a bool")
        if not isinstance(learn_predictive_readout, bool):
            raise TypeError("learn_predictive_readout must be a bool")
        consolidation_strength = float(consolidation_strength)
        if not math.isfinite(consolidation_strength) or consolidation_strength < 0.0:
            raise ValueError("consolidation_strength must be finite and non-negative")
        if consolidation_strength > 0.0 and not learn_predictive_readout:
            raise ValueError("consolidation requires learn_predictive_readout=True")
        if (boundary is None) != (authorization is None):
            raise ValueError("boundary and authorization must be supplied together")
        active_readout: BytePredictiveReadout | None = None
        if boundary is not None and authorization is not None:
            resolved_boundary = (
                boundary
                if isinstance(boundary, WorkbenchTaskBoundary)
                else WorkbenchTaskBoundary.from_payload(boundary)
            )
            generation_scope = select_readout_generation(resolved_boundary, authorization)
            if generation_scope != "active":
                raise RuntimeError("protected predictive readout is read-only under an explicit boundary")
            if authorization.usage != "execute":
                raise PermissionError("active predictive readout training requires execute authorization")
            active_readout = self._predictive_readout_for_scope(
                generation_scope,
                boundary_digest=resolved_boundary.token_digest,
            )
            if learn_fabric or learn_predictive_context:
                raise ValueError(
                    "active readout training requires learn_fabric=False and "
                    "learn_predictive_context=False"
                )
        if consolidation_strength > 0.0 and active_readout is None:
            raise RuntimeError("consolidation requires an explicit active readout boundary")
        observations = 0
        correct = 0
        surprise_sum = 0.0
        for epoch in range(epochs):
            if reset or epoch > 0:
                self.reset_dynamics(episode_id=f"learn-{epoch}")
            for symbol in self.sensor.symbols(
                data,
                include_boundary=include_boundary,
                include_start_boundary=include_start_boundary,
                include_end_boundary=include_end_boundary,
            ):
                step = self.observe(
                    symbol,
                    learn=True,
                    learn_fabric=learn_fabric,
                    learn_predictive_context=learn_predictive_context,
                    learn_predictive_readout=learn_predictive_readout,
                    readout="predictive",
                    use_memory=use_memory,
                    use_identity=False,
                    _predictive_readout=active_readout,
                    _preservation_readout=(
                        self.predictive_readout if consolidation_strength > 0.0 else None
                    ),
                    _preservation_strength=consolidation_strength,
                )
                if step.prior_prediction is not None:
                    observations += 1
                    correct += int(step.prior_prediction == symbol)
                    surprise_sum += float(step.surprise or 0.0)
        return {
            "observations": float(observations),
            "online_accuracy": correct / max(1, observations),
            "mean_surprise": surprise_sum / max(1, observations),
        }

    def score_bytes(
        self,
        data: bytes,
        *,
        include_boundary: bool = True,
        use_memory: bool = False,
        boundary: WorkbenchTaskBoundary | Mapping[str, Any] | None = None,
        authorization: WorkbenchBoundaryAuthorization | None = None,
    ) -> dict[str, Any]:
        """Evaluate raw-byte prediction without mutating persistent state.

        The default deliberately matches :meth:`learn_bytes`: F1 scores the
        predictive path without long-term episodic feedback. This makes a
        score comparable before and after a separate F2 memory course.

        When ``boundary`` and ``authorization`` are supplied together, the
        effective readout owner is resolved with the same content-addressed
        rules as :meth:`generate`.  A protected scope reads the stable parent
        readout; an active scope reads the isolated task branch and raises if
        it is not attached.  The returned score and the auditable
        :attr:`last_generation_route` both record the effective owner, readout
        digest and boundary digest, so an active/protected comparison cannot
        silently compare the wrong organ.
        """

        if (boundary is None) != (authorization is None):
            raise ValueError("boundary and authorization must be supplied together")
        predictive_readout: BytePredictiveReadout | None = None
        scope = "protected"
        boundary_digest: str | None = None
        read_only_replay = False
        if boundary is not None and authorization is not None:
            resolved_boundary = (
                boundary
                if isinstance(boundary, WorkbenchTaskBoundary)
                else WorkbenchTaskBoundary.from_payload(boundary)
            )
            scope = select_readout_generation(resolved_boundary, authorization)
            predictive_readout = self._predictive_readout_for_scope(
                scope,
                boundary_digest=resolved_boundary.token_digest,
            )
            boundary_digest = resolved_boundary.token_digest
            read_only_replay = authorization.usage == "read_only_replay"
        effective_readout = (
            self.predictive_readout if predictive_readout is None else predictive_readout
        )
        readout_digest = content_digest(effective_readout.to_payload())
        owner = "predictive_readout" if scope == "protected" else "predictive_readout.active"
        self._last_generation_route = {
            "operation": "score",
            "boundary_digest": boundary_digest,
            "generation_scope": scope,
            "readout_owner": owner,
            "readout_digest": readout_digest,
            "read_only_replay": read_only_replay,
        }
        checkpoint = self.checkpoint()
        self.reset_dynamics(episode_id="evaluation")
        observations = 0
        correct = 0
        surprise_sum = 0.0
        try:
            for symbol in self.sensor.symbols(data, include_boundary=include_boundary):
                step = self.observe(
                    symbol,
                    learn=False,
                    readout="predictive",
                    use_memory=use_memory,
                    use_identity=False,
                    _predictive_readout=predictive_readout,
                )
                if step.prior_prediction is not None:
                    observations += 1
                    correct += int(step.prior_prediction == symbol)
                    surprise_sum += float(step.surprise or 0.0)
            return {
                "observations": float(observations),
                "accuracy": correct / max(1, observations),
                "mean_surprise": surprise_sum / max(1, observations),
                "scope": scope,
                "owner": owner,
                "readout_digest": readout_digest,
                "boundary_digest": boundary_digest,
            }
        finally:
            self.restore(checkpoint)

    @torch.no_grad()
    def generate(
        self,
        prompt: bytes,
        length: int,
        *,
        stop_at_boundary: bool = False,
        sample: bool = False,
        reset: bool = True,
        use_memory: bool = False,
        boundary: WorkbenchTaskBoundary | Mapping[str, Any] | None = None,
        authorization: WorkbenchBoundaryAuthorization | None = None,
    ) -> bytes:
        """Generate from the raw-byte predictive path.

        As with training and scoring, episodic augmentation is opt-in. The
        default prevents a populated delayed-memory field from hijacking
        native language generation outside an explicit memory-query task.
        """

        if length < 0:
            raise ValueError("length cannot be negative")
        if (boundary is None) != (authorization is None):
            raise ValueError("boundary and authorization must be supplied together")
        predictive_readout: BytePredictiveReadout | None = None
        if boundary is not None and authorization is not None:
            resolved_boundary = (
                boundary
                if isinstance(boundary, WorkbenchTaskBoundary)
                else WorkbenchTaskBoundary.from_payload(boundary)
            )
            generation_scope = select_readout_generation(resolved_boundary, authorization)
            predictive_readout = self._predictive_readout_for_scope(
                generation_scope,
                boundary_digest=resolved_boundary.token_digest,
            )
            self._last_generation_route = {
                "boundary_digest": resolved_boundary.token_digest,
                "generation_scope": generation_scope,
                "readout_owner": (
                    "predictive_readout"
                    if generation_scope == "protected"
                    else "predictive_readout.active"
                ),
                "read_only_replay": authorization.usage == "read_only_replay",
            }
        if reset:
            self.reset_dynamics(episode_id="generation")
        step = self.observe(
            self.config.boundary_symbol,
            learn=False,
            readout="predictive",
            use_memory=use_memory,
            use_identity=False,
            _predictive_readout=predictive_readout,
        )
        for symbol in prompt:
            step = self.observe(
                int(symbol),
                learn=False,
                readout="predictive",
                use_memory=use_memory,
                use_identity=False,
                _predictive_readout=predictive_readout,
            )

        generated = bytearray()
        for _ in range(length):
            if sample:
                next_symbol = int(
                    torch.multinomial(
                        step.probabilities.detach().cpu(), 1, generator=self._rng
                    ).item()
                )
            else:
                next_symbol = step.predicted_symbol
            if next_symbol == self.config.boundary_symbol and stop_at_boundary:
                break
            if not 0 <= next_symbol <= 255:
                next_symbol = 0
            generated.append(next_symbol)
            step = self.observe(
                next_symbol,
                learn=False,
                readout="predictive",
                use_memory=use_memory,
                use_identity=False,
                _predictive_readout=predictive_readout,
            )
        return bytes(generated)

    def parameter_tensors(self) -> tuple[torch.Tensor, ...]:
        tensors = (
            *self.fabric.parameter_tensors(),
            self.motor.synapses.edge_weight,
            self.motor.bias,
            self.predictive_context.recurrent.edge_weight,
            self.predictive_readout.synapses.edge_weight,
            self.predictive_readout.bias,
            *self.memory.parameter_tensors(),
        )
        if self._gated_temporal_candidate is not None:
            tensors += self._gated_temporal_candidate.parameter_tensors()
        if self._active_predictive_readout is not None:
            tensors += (
                self._active_predictive_readout.synapses.edge_weight,
                self._active_predictive_readout.bias,
            )
        if self.identity_organ is not None:
            tensors += self.identity_organ.parameter_tensors()
        return tensors

    def parameter_count(self, *, active_only: bool = True) -> int:
        active = (
            self.fabric.active_edge_count()
            + self.motor.synapses.edge_count
            + self.motor.bias.numel()
            + self.predictive_context.recurrent.edge_count
            + self.predictive_readout.synapses.edge_count
            + self.predictive_readout.bias.numel()
            + self.memory.active_edge_count()
        )
        if self._gated_temporal_candidate is not None:
            active += sum(tensor.numel() for tensor in self._gated_temporal_candidate.parameter_tensors())
        if self._active_predictive_readout is not None:
            active += (
                self._active_predictive_readout.synapses.edge_count
                + self._active_predictive_readout.bias.numel()
            )
        if self.identity_organ is not None:
            active += self.identity_organ.parameter_count
        if active_only:
            return active
        return active

    def dense_equivalent_parameter_count(self) -> int:
        """Return the learned scalar count a dense implementation would store."""

        count = (
            self.fabric.dense_equivalent_edge_count()
            + self.motor.synapses.dense_equivalent_count
            + self.motor.bias.numel()
            + self.predictive_context.recurrent.dense_equivalent_count
            + self.predictive_readout.synapses.dense_equivalent_count
            + self.predictive_readout.bias.numel()
            + self.memory.dense_equivalent_edge_count()
        )
        if self._gated_temporal_candidate is not None:
            count += (
                self._gated_temporal_candidate.fast.dense_equivalent_count
                + self._gated_temporal_candidate.slow.dense_equivalent_count
            )
        if self._active_predictive_readout is not None:
            count += (
                self._active_predictive_readout.synapses.dense_equivalent_count
                + self._active_predictive_readout.bias.numel()
            )
        if self.identity_organ is not None:
            count += self.identity_organ.capacity * self.identity_organ.pattern_dim
            count += self.identity_organ.action_synapses.dense_equivalent_count
            count += self.identity_organ.outcome_synapses.dense_equivalent_count
        return count

    def _checkpoint_core(self) -> dict[str, Any]:
        core = {
            "format": self.CHECKPOINT_FORMAT,
            "config": self.config.to_dict(),
            "fabric": self.fabric.to_payload(),
            "motor": self.motor.to_payload(),
            "predictive_context": self.predictive_context.to_payload(),
            "predictive_readout": self.predictive_readout.to_payload(),
            "memory": self.memory.to_payload(),
            "state": self._state.to_payload(),
            "rng_state": self._rng.get_state().clone(),
        }
        if self._gated_temporal_candidate is not None:
            core[self.GATED_TEMPORAL_CANDIDATE_KEY] = (
                self._gated_temporal_candidate.to_payload()
            )
        return core

    def checkpoint(self) -> dict[str, Any]:
        core = self._checkpoint_core()
        if self.identity_organ is None:
            # The optional identity organ remains absent until enabled; the
            # F1 predictive context/readout are part of every v10 core.
            payload = core
        else:
            payload = {
                **core,
                "identity_organ": self.identity_organ.to_payload(
                    parent_checkpoint_digest=content_digest(core),
                ),
            }
        if self._identity_growth_history:
            payload = {
                **payload,
                "identity_growth": {
                    "format": self.IDENTITY_GROWTH_FORMAT,
                    "version": 1,
                    "events": [dict(item) for item in self._identity_growth_history],
                },
            }
        readout_registry = self._readout_registry_payload()
        if readout_registry is not None:
            payload = {
                **payload,
                "predictive_readout_registry": readout_registry,
            }
        return payload

    def restore(self, checkpoint: Mapping[str, Any]) -> None:
        checkpoint_format = str(checkpoint.get("format", ""))
        if checkpoint_format not in {
            self.CHECKPOINT_FORMAT,
            *self.LEGACY_CHECKPOINT_FORMATS,
        }:
            raise ValueError("unsupported Taiji checkpoint format")
        is_legacy_checkpoint = checkpoint_format in self.LEGACY_CHECKPOINT_FORMATS
        actual = TaijiConfig.from_dict(dict(checkpoint["config"]))
        if actual != self.config:
            raise ValueError("checkpoint configuration does not match architecture")
        self.clear_active_predictive_readout()
        self.fabric.load_payload(checkpoint["fabric"])
        self.motor.load_payload(checkpoint["motor"])
        predictive_context_payload = checkpoint.get("predictive_context")
        if predictive_context_payload is None and not is_legacy_checkpoint:
            raise ValueError("v10 checkpoint is missing its predictive context")
        if predictive_context_payload is None:
            # M2-2h migration: v8/v9's F1 basis was the motor receptor map.
            # Copy it once into the new owner with a neutral temporal residual,
            # then let future F1 updates grow only the private substrate.
            self.predictive_context.load_legacy_motor_payload(checkpoint["motor"])
        elif not isinstance(predictive_context_payload, Mapping):
            raise ValueError("predictive context checkpoint payload is invalid")
        else:
            self.predictive_context.load_payload(predictive_context_payload)
        predictive_payload = checkpoint.get("predictive_readout")
        if predictive_payload is None and not is_legacy_checkpoint:
            raise ValueError("v10 checkpoint is missing its predictive readout")
        if predictive_payload is None:
            # M2-2f migration: legacy v8 stored F1 and F4 in the same motor
            # decoder.  Copy that learned F1 surface once, then make all future
            # byte updates land on the dedicated predictive organ.
            self.predictive_readout.load_legacy_motor_payload(checkpoint["motor"])
        elif not isinstance(predictive_payload, Mapping):
            raise ValueError("predictive readout checkpoint payload is invalid")
        else:
            self.predictive_readout.load_payload(predictive_payload)
        candidate_payload = checkpoint.get(self.GATED_TEMPORAL_CANDIDATE_KEY)
        self._gated_temporal_candidate = None
        if candidate_payload is not None:
            if is_legacy_checkpoint:
                raise ValueError("legacy checkpoint cannot contain a gated temporal candidate")
            if not isinstance(candidate_payload, Mapping):
                raise ValueError("gated temporal candidate checkpoint payload is invalid")
            candidate = self._new_gated_temporal_candidate(
                fast_decay=float(candidate_payload["fast_decay"]),
                slow_decay=float(candidate_payload["slow_decay"]),
                gate_temperature=float(candidate_payload["gate_temperature"]),
                residual_gain=float(candidate_payload["residual_gain"]),
            )
            candidate.load_payload(candidate_payload)
            self._gated_temporal_candidate = candidate
        self.memory.load_payload(checkpoint["memory"])
        identity_payload = checkpoint.get("identity_organ")
        if self.identity_organ is None:
            if identity_payload is not None:
                raise ValueError("checkpoint contains an enabled identity organ")
        else:
            if not isinstance(identity_payload, Mapping):
                raise ValueError("enabled identity organ checkpoint payload is missing")
            lineage = identity_payload.get("lineage")
            expected_parent_digest = content_digest(
                {
                    key: checkpoint[key]
                    for key in self._checkpoint_core_keys(
                        include_predictive="predictive_readout" in checkpoint,
                        include_predictive_context="predictive_context" in checkpoint,
                        include_temporal_candidate=(
                            self.GATED_TEMPORAL_CANDIDATE_KEY in checkpoint
                        ),
                    )
                }
            )
            if (
                not isinstance(lineage, Mapping)
                or str(lineage.get("parent_checkpoint_digest", "")) != expected_parent_digest
            ):
                raise ValueError("identity organ checkpoint lineage does not match Taiji core")
            self.identity_organ.load_payload(identity_payload)
        self._restore_readout_registry(checkpoint.get("predictive_readout_registry"))
        growth_payload = checkpoint.get("identity_growth")
        if growth_payload is None:
            self._identity_growth_history = []
        else:
            if not isinstance(growth_payload, Mapping):
                raise ValueError("identity growth checkpoint payload is invalid")
            if growth_payload.get("format") != self.IDENTITY_GROWTH_FORMAT:
                raise ValueError("unsupported identity growth checkpoint format")
            if int(growth_payload.get("version", -1)) != 1:
                raise ValueError("unsupported identity growth checkpoint version")
            events = growth_payload.get("events")
            if not isinstance(events, list):
                raise ValueError("identity growth checkpoint events must be a list")
            restored_events: list[dict[str, Any]] = []
            for event in events:
                if not isinstance(event, Mapping):
                    raise ValueError("identity growth checkpoint event must be a mapping")
                if event.get("format") != self.IDENTITY_GROWTH_FORMAT:
                    raise ValueError("identity growth checkpoint event format is invalid")
                restored_events.append(dict(event))
            self._identity_growth_history = restored_events
        state = TaijiState.from_payload(checkpoint["state"], device=self.device)
        if state.version not in {self.STATE_VERSION, *self.LEGACY_STATE_VERSIONS}:
            raise ValueError("unsupported Taiji state version")
        if not is_legacy_checkpoint and state.version != self.STATE_VERSION:
            raise ValueError("v10 checkpoint must carry the v7 Taiji state")
        if len(state.regions) != len(self.config.region_sizes):
            raise ValueError("checkpoint region state does not match architecture")
        memory_shape = (self.config.memory_units,)
        if (
            state.memory.activity.shape != memory_shape
            or state.memory.trace.shape != memory_shape
            or state.memory.threshold.shape != memory_shape
        ):
            raise ValueError("checkpoint memory state does not match architecture")
        if state.memory.cortical_feedback.shape != (self.config.cortical_context_dim,):
            raise ValueError("checkpoint memory feedback does not match architecture")
        if state.motor_context.shape != (self.config.motor_context_dim,):
            raise ValueError("checkpoint motor context does not match architecture")
        if state.predictive_context_trace.shape != (self.config.motor_context_dim,):
            raise ValueError("checkpoint predictive context trace does not match architecture")
        if self._gated_temporal_candidate is None:
            if state.predictive_context_slow_trace is not None:
                raise ValueError("checkpoint carries a slow temporal trace without its candidate")
        else:
            if state.predictive_context_slow_trace is None:
                state.predictive_context_slow_trace = torch.zeros(
                    self.config.motor_context_dim,
                    device=self.device,
                )
            elif state.predictive_context_slow_trace.shape != (
                self.config.motor_context_dim,
            ):
                raise ValueError("checkpoint slow temporal trace does not match architecture")
        if state.motor_probabilities.shape != (self.config.alphabet_size,):
            raise ValueError("checkpoint motor probabilities do not match architecture")
        if state.readout_kind not in {"action", "predictive"}:
            raise ValueError("checkpoint readout kind is invalid")
        if state.pending_action is not None:
            if state.readout_kind != "action":
                raise ValueError("predictive checkpoint state cannot hold an action")
            pending = state.pending_action
            if pending.context.shape != (self.config.motor_context_dim,):
                raise ValueError("checkpoint pending context does not match architecture")
            if pending.policy_probabilities.shape != (self.config.alphabet_size,):
                raise ValueError("checkpoint pending policy does not match architecture")
            if pending.action_symbol not in pending.available_actions:
                raise ValueError("checkpoint pending action is not afforded")
        if state.pending_experience is not None:
            experience = state.pending_experience
            if experience.cortical_context.shape != (self.config.cortical_context_dim,):
                raise ValueError("checkpoint pending experience does not match architecture")
            if not 0 <= experience.action_symbol < self.config.alphabet_size:
                raise ValueError("checkpoint pending experience action is invalid")
            if not math.isfinite(experience.reward):
                raise ValueError("checkpoint pending experience reward is invalid")
            if not experience.episode_id:
                raise ValueError("checkpoint pending experience episode is invalid")
            if experience.provenance not in self.memory.PROVENANCE_KINDS:
                raise ValueError("checkpoint pending experience provenance is invalid")
            validate_episodic_learning_target(
                "checkpoint pending experience learning targets",
                experience.memory_learning_targets,
            )
            if (
                not math.isfinite(experience.memory_learning_scale)
                or experience.memory_learning_scale <= 0.0
            ):
                raise ValueError("checkpoint pending experience learning scale is invalid")
        # v8/v9 state lacks the private residual eligibility trace.  Its
        # migration starts that trace at zero; v8 additionally had no explicit
        # predictive readout owner, so its state stays action-owned.
        if is_legacy_checkpoint:
            state.version = self.STATE_VERSION
        self._state = state
        self._rng.set_state(checkpoint["rng_state"].detach().cpu())

    @staticmethod
    def _checkpoint_core_keys(
        *,
        include_predictive: bool = True,
        include_predictive_context: bool | None = None,
        include_temporal_candidate: bool = False,
    ) -> tuple[str, ...]:
        if include_predictive_context is None:
            # Existing callers reconstructing a v8 lineage pass only
            # ``include_predictive=False``.  Preserve that intended core shape
            # while v10 callers can specify the two F1 organs independently.
            include_predictive_context = include_predictive
        # The optional F1 owners make this an extensible key set.  Without the
        # variadic annotation mypy infers the four literal entries as a fixed
        # length tuple and rejects each conditional extension below.
        keys: tuple[str, ...] = (
            "format",
            "config",
            "fabric",
            "motor",
        )
        if include_predictive:
            keys += ("predictive_readout",)
        if include_predictive_context:
            keys += ("predictive_context",)
        if include_temporal_candidate:
            keys += (Taiji.GATED_TEMPORAL_CANDIDATE_KEY,)
        return (*keys, "memory", "state", "rng_state")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Mapping[str, Any],
        *,
        device: torch.device | str = "cpu",
    ) -> Taiji:
        config = TaijiConfig.from_dict(dict(checkpoint["config"]))
        model = cls(config, device=device)
        model.restore(checkpoint)
        return model
