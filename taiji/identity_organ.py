"""Optional cue-identity organ for the native Taiji memory runtime.

The organ is deliberately narrower than a language model.  It learns a
bounded cue identity and emits motor evidence for the owning slot.  The
native motor remains the only component that turns evidence into a final
action distribution, and an unbound cue emits no identity evidence.

A bound slot carries two heads, so the organ is a first-class key/value
memory rather than an action-only reference: the action head feeds motor
evidence, while the outcome head is a read-only prediction channel that
mirrors ``MemoryRecall.outcome_probabilities`` and is never added to the
motor's action distribution.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from .config import TaijiConfig
from .cue_binding import CueBindingBank, CueBindingResult
from .sparse import SparseSynapses

IDENTITY_ORGAN_CHECKPOINT_FORMAT = "taiji-native-identity-organ-v2"
IDENTITY_ORGAN_VERSION = 2
IDENTITY_ORGAN_BOUND_PROVENANCE = "identity-organ:bound→motor-evidence"
IDENTITY_ORGAN_UNBOUND_PROVENANCE = "identity-organ:unbound→shared-fallback"
IDENTITY_ORGAN_DISABLED_PROVENANCE = "identity-organ:disabled→shared-fallback"


@dataclass(frozen=True, eq=False)
class IdentityRecall:
    """Read-only identity evidence produced for one cortical cue.

    ``action_evidence`` is the only field the motor is allowed to consume.
    The outcome fields are a prediction channel: they report what the bound
    slot expects to observe next and stay out of the action distribution.
    """

    action_evidence: torch.Tensor
    action_probabilities: torch.Tensor
    outcome_evidence: torch.Tensor
    outcome_probabilities: torch.Tensor
    slot_index: int | None
    similarity: float
    source: str
    provenance: str
    used: bool


class CueIdentityOrgan:
    """Bounded cue identity population with physical slot-to-value edges.

    Cue prototypes are owned by :class:`CueBindingBank`; the action and
    outcome associations are stored as fixed-fan-in synapses with one
    physical edge per slot/symbol pair.  A replacement clears the old slot's
    edges in both heads before the new association is learned, so capacity
    pressure cannot leak a stale action or outcome into an unrelated cue.
    """

    CHECKPOINT_FORMAT = IDENTITY_ORGAN_CHECKPOINT_FORMAT
    VERSION = IDENTITY_ORGAN_VERSION

    def __init__(
        self,
        config: TaijiConfig,
        *,
        generator: torch.Generator,
        device: torch.device | str = "cpu",
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.capacity = int(config.identity_organ_capacity)
        self.pattern_dim = int(config.cortical_context_dim)
        self.action_count = int(config.alphabet_size)
        self.outcome_count = int(config.alphabet_size)
        self.bank = CueBindingBank(
            self.capacity,
            self.pattern_dim,
            match_threshold=config.identity_organ_match_threshold,
            update_rate=config.identity_organ_update_rate,
            device=self.device,
        )
        # Full slot fan-in is intentional for the first native organ: every
        # action row can hear every bounded identity slot.  The storage is
        # still compressed fixed-fan-in synapses, not a dense matrix, and the
        # budget is explicit in TaijiConfig.planned_active_parameter_count.
        self.action_synapses = SparseSynapses(
            self.action_count,
            self.capacity,
            self.capacity,
            generator=generator,
            init_scale=config.weight_init_scale,
            max_weight_norm=config.max_weight_norm,
            device=self.device,
        )
        self.action_synapses.edge_weight.zero_()
        # The outcome head is constructed after the action head on purpose:
        # SparseSynapses draws one randperm per post row, so appending a new
        # population here leaves every pre-existing edge topology for a given
        # seed byte-for-byte unchanged.
        self.outcome_synapses = SparseSynapses(
            self.outcome_count,
            self.capacity,
            self.capacity,
            generator=generator,
            init_scale=config.weight_init_scale,
            max_weight_norm=config.max_weight_norm,
            device=self.device,
        )
        self.outcome_synapses.edge_weight.zero_()
        self.write_count = 0
        self.replacement_count = 0
        self.skipped_write_count = 0
        self.punished_write_count = 0

    @property
    def edge_count(self) -> int:
        return int(self.action_synapses.edge_count + self.outcome_synapses.edge_count)

    @property
    def parameter_count(self) -> int:
        return int(
            self.bank.prototypes.numel()
            + self.action_synapses.edge_count
            + self.outcome_synapses.edge_count
        )

    def parameter_tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.bank.prototypes,
            self.action_synapses.edge_weight,
            self.outcome_synapses.edge_weight,
        )

    def _slot_trace(self, slot_index: int) -> torch.Tensor:
        return self.bank.slot_code(int(slot_index)).to(self.device)

    def _clear_slot(self, slot_index: int) -> None:
        slot = int(slot_index)
        for synapses in (self.action_synapses, self.outcome_synapses):
            mask = synapses.pre_index == slot
            synapses.edge_weight.masked_fill_(mask, 0.0)

    def _validate_symbol(self, symbol: int, count: int, field: str) -> int:
        value = int(symbol)
        if not 0 <= value < count:
            raise ValueError(f"identity organ {field} is outside the motor alphabet")
        return value

    def _carries_cue(self, cortical_context: torch.Tensor) -> bool:
        """Report whether a cortical context carries any identity at all.

        A zero-norm context is not a malformed cue, it is the absence of one:
        the very first observation of a fresh model settles before any region
        has fired.  The organ is on the default path now, so it must degrade
        to ``unbound`` there instead of propagating the bank's write-path
        invariant that a prototype can never be allocated from an empty
        pattern.
        """

        if cortical_context.shape != (self.pattern_dim,):
            return False
        return float(cortical_context.norm().item()) > 1e-8

    @torch.no_grad()
    def _train_head(
        self,
        synapses: SparseSynapses,
        trace: torch.Tensor,
        target_index: int,
        count: int,
        modulation: float,
    ) -> None:
        if modulation == 0.0:
            return
        target = torch.zeros(count, device=self.device)
        target[target_index] = 1.0
        repeats = int(self.config.identity_organ_learning_repeats)
        for _ in range(repeats):
            logits = synapses.forward(trace)
            probabilities = torch.softmax(logits, dim=0)
            synapses.local_update(
                modulation * (target - probabilities),
                trace,
                learning_rate=self.config.identity_organ_learning_rate,
                weight_decay=0.0,
            )

    @torch.no_grad()
    def learn(
        self,
        cortical_context: torch.Tensor,
        action_symbol: int,
        *,
        outcome_symbol: int,
        reward: float = 1.0,
    ) -> CueBindingResult:
        """Bind one settled cue/action/outcome triple with local updates.

        ``outcome_symbol`` is keyword-only and required so no write path can
        train the action head while leaving the outcome head empty, which
        would silently degrade the organ back to an action-only reference.

        ``reward`` is the signed environment reward of the settled action and
        is what makes the organ safe on the default path.  Write eligibility
        is split along the key/value boundary:

        * The cue prototype (the key) is learned regardless of reward.  A cue
          observed during a failure is still that cue, so refusing to route it
          would make identity itself reward-dependent and lose the very
          discrimination the organ exists to provide.
        * The two heads (the values) are updated through the same three-factor
          modulation the motor uses, ``reward - identity_organ_write_baseline``.
          A punished action is therefore pushed *away* from its cue instead of
          being bound as strongly as a rewarded one, which is what kept a
          binary-cue task pinned at chance while the organ was unmodulated.

        The default ``1.0`` reproduces the unmodulated v2 update exactly, so
        existing checkpoints and evaluator evidence stay bit-comparable.
        """

        action = self._validate_symbol(action_symbol, self.action_count, "action")
        outcome = self._validate_symbol(outcome_symbol, self.outcome_count, "outcome")
        reward = float(reward)
        if not math.isfinite(reward):
            raise ValueError("identity organ write reward must be finite")
        if not self._carries_cue(cortical_context):
            # No cue means nothing to bind.  Refusing here keeps the bank's
            # prototypes free of meaningless assemblies while still letting
            # the default write path run without an exception.
            self.skipped_write_count += 1
            return CueBindingResult(
                slot_index=None,
                similarity=0.0,
                allocated=False,
                replaced=False,
            )
        modulation = reward - float(self.config.identity_organ_write_baseline)
        binding = self.bank.route(cortical_context, learn=True)
        if binding.slot_index is None:
            raise RuntimeError("identity organ binding did not return a slot")
        slot = int(binding.slot_index)
        if binding.replaced:
            self._clear_slot(slot)
            self.replacement_count += 1
        trace = self._slot_trace(slot)
        self._train_head(
            self.action_synapses, trace, action, self.action_count, modulation
        )
        self._train_head(
            self.outcome_synapses, trace, outcome, self.outcome_count, abs(modulation)
        )
        self.write_count += 1
        if modulation < 0.0:
            self.punished_write_count += 1
        return binding

    @torch.no_grad()
    def recall(
        self,
        cortical_context: torch.Tensor,
        *,
        enabled: bool = True,
    ) -> IdentityRecall:
        """Read identity evidence without changing prototypes or synapses."""

        zero = torch.zeros(self.action_count, device=self.device)
        uniform = torch.full(
            (self.action_count,),
            1.0 / self.action_count,
            device=self.device,
        )
        outcome_zero = torch.zeros(self.outcome_count, device=self.device)
        outcome_uniform = torch.full(
            (self.outcome_count,),
            1.0 / self.outcome_count,
            device=self.device,
        )
        if not enabled:
            return IdentityRecall(
                action_evidence=zero,
                action_probabilities=uniform,
                outcome_evidence=outcome_zero,
                outcome_probabilities=outcome_uniform,
                slot_index=None,
                similarity=0.0,
                source="shared-fallback",
                provenance=IDENTITY_ORGAN_DISABLED_PROVENANCE,
                used=False,
            )
        if not self._carries_cue(cortical_context):
            return IdentityRecall(
                action_evidence=zero,
                action_probabilities=uniform,
                outcome_evidence=outcome_zero,
                outcome_probabilities=outcome_uniform,
                slot_index=None,
                similarity=0.0,
                source="shared-fallback",
                provenance=IDENTITY_ORGAN_UNBOUND_PROVENANCE,
                used=False,
            )
        binding = self.bank.route(cortical_context, learn=False)
        if binding.slot_index is None:
            return IdentityRecall(
                action_evidence=zero,
                action_probabilities=uniform,
                outcome_evidence=outcome_zero,
                outcome_probabilities=outcome_uniform,
                slot_index=None,
                similarity=float(binding.similarity),
                source="shared-fallback",
                provenance=IDENTITY_ORGAN_UNBOUND_PROVENANCE,
                used=False,
            )
        trace = self._slot_trace(binding.slot_index)
        logits = self.action_synapses.forward(trace)
        probabilities = torch.softmax(logits, dim=0)
        outcome_logits = self.outcome_synapses.forward(trace)
        outcome_probabilities = torch.softmax(outcome_logits, dim=0)
        return IdentityRecall(
            action_evidence=logits.detach().clone(),
            action_probabilities=probabilities.detach().clone(),
            outcome_evidence=outcome_logits.detach().clone(),
            outcome_probabilities=outcome_probabilities.detach().clone(),
            slot_index=int(binding.slot_index),
            similarity=float(binding.similarity),
            source="identity-route",
            provenance=IDENTITY_ORGAN_BOUND_PROVENANCE,
            used=True,
        )

    @torch.no_grad()
    def lesion(self) -> None:
        """Remove all identity bindings and value evidence in-place."""

        self.bank.occupied.zero_()
        self.bank.prototypes.zero_()
        self.bank.visits.zero_()
        self.action_synapses.edge_weight.zero_()
        self.outcome_synapses.edge_weight.zero_()

    def to_payload(self, *, parent_checkpoint_digest: str) -> dict[str, Any]:
        if not parent_checkpoint_digest:
            raise ValueError("identity organ checkpoint needs a parent digest")
        return {
            "format": self.CHECKPOINT_FORMAT,
            "version": self.VERSION,
            "lineage": {
                "organ_id": "cue-identity-route",
                "organ_version": self.VERSION,
                "parent_checkpoint_digest": str(parent_checkpoint_digest),
            },
            "capacity": self.capacity,
            "pattern_dim": self.pattern_dim,
            "action_count": self.action_count,
            "outcome_count": self.outcome_count,
            "match_threshold": self.config.identity_organ_match_threshold,
            "update_rate": self.config.identity_organ_update_rate,
            "learning_rate": self.config.identity_organ_learning_rate,
            "learning_repeats": self.config.identity_organ_learning_repeats,
            "evidence_gain": self.config.identity_organ_evidence_gain,
            "write_baseline": self.config.identity_organ_write_baseline,
            "bank": self.bank.to_payload(),
            "action_synapses": self.action_synapses.to_payload(),
            "outcome_synapses": self.outcome_synapses.to_payload(),
            "write_count": self.write_count,
            "replacement_count": self.replacement_count,
            "skipped_write_count": self.skipped_write_count,
            "punished_write_count": self.punished_write_count,
        }

    def load_payload(self, payload: Mapping[str, Any]) -> None:
        if payload.get("format") != self.CHECKPOINT_FORMAT:
            raise ValueError("unsupported identity organ checkpoint format")
        if int(payload.get("version", -1)) != self.VERSION:
            raise ValueError("unsupported identity organ checkpoint version")
        expected = (
            self.capacity,
            self.pattern_dim,
            self.action_count,
            self.outcome_count,
            self.config.identity_organ_match_threshold,
            self.config.identity_organ_update_rate,
            self.config.identity_organ_learning_rate,
            self.config.identity_organ_learning_repeats,
            self.config.identity_organ_evidence_gain,
            float(self.config.identity_organ_write_baseline),
        )
        actual = (
            int(payload["capacity"]),
            int(payload["pattern_dim"]),
            int(payload["action_count"]),
            int(payload["outcome_count"]),
            float(payload["match_threshold"]),
            float(payload["update_rate"]),
            float(payload["learning_rate"]),
            int(payload["learning_repeats"]),
            float(payload["evidence_gain"]),
            # A payload written before the write baseline existed was trained
            # with an unmodulated update, which is exactly a ``0.0`` baseline
            # at the default reward of ``1.0``.  Defaulting here keeps those
            # checkpoints loadable without pretending the field was stored.
            float(payload.get("write_baseline", 0.0)),
        )
        if actual != expected:
            raise ValueError("identity organ checkpoint architecture does not match")
        self.bank.load_payload(dict(payload["bank"]))
        self.action_synapses.load_payload(dict(payload["action_synapses"]))
        self.outcome_synapses.load_payload(dict(payload["outcome_synapses"]))
        self.write_count = int(payload.get("write_count", 0))
        self.replacement_count = int(payload.get("replacement_count", 0))
        self.skipped_write_count = int(payload.get("skipped_write_count", 0))
        self.punished_write_count = int(payload.get("punished_write_count", 0))
        if (
            min(
                self.write_count,
                self.replacement_count,
                self.skipped_write_count,
                self.punished_write_count,
            )
            < 0
        ):
            raise ValueError("identity organ counters cannot be negative")
        if self.punished_write_count > self.write_count:
            raise ValueError("identity organ punished writes cannot exceed writes")


__all__ = [
    "IDENTITY_ORGAN_BOUND_PROVENANCE",
    "IDENTITY_ORGAN_CHECKPOINT_FORMAT",
    "IDENTITY_ORGAN_DISABLED_PROVENANCE",
    "IDENTITY_ORGAN_UNBOUND_PROVENANCE",
    "IDENTITY_ORGAN_VERSION",
    "CueIdentityOrgan",
    "IdentityRecall",
]
