"""Optional cue-identity organ for the native Taiji memory runtime.

The organ is deliberately narrower than a language model.  It learns a
bounded cue identity and emits motor evidence for the owning slot.  The
native motor remains the only component that turns evidence into a final
action distribution, and an unbound cue emits no identity evidence.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from .config import TaijiConfig
from .cue_binding import CueBindingBank, CueBindingResult
from .sparse import SparseSynapses

IDENTITY_ORGAN_CHECKPOINT_FORMAT = "taiji-native-identity-organ-v1"
IDENTITY_ORGAN_VERSION = 1
IDENTITY_ORGAN_BOUND_PROVENANCE = "identity-organ:bound→motor-evidence"
IDENTITY_ORGAN_UNBOUND_PROVENANCE = "identity-organ:unbound→shared-fallback"
IDENTITY_ORGAN_DISABLED_PROVENANCE = "identity-organ:disabled→shared-fallback"


@dataclass(frozen=True, eq=False)
class IdentityRecall:
    """Read-only identity evidence produced for one cortical cue."""

    action_evidence: torch.Tensor
    action_probabilities: torch.Tensor
    slot_index: int | None
    similarity: float
    source: str
    provenance: str
    used: bool


class CueIdentityOrgan:
    """Bounded cue identity population with physical slot-to-action edges.

    Cue prototypes are owned by :class:`CueBindingBank`; action associations
    are stored as fixed-fan-in synapses with one physical edge per
    slot/action pair.  A replacement clears the old slot's action edges before
    the new association is learned, so capacity pressure cannot leak a stale
    action into an unrelated cue.
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
        self.write_count = 0
        self.replacement_count = 0

    @property
    def edge_count(self) -> int:
        return int(self.action_synapses.edge_count)

    @property
    def parameter_count(self) -> int:
        return int(self.bank.prototypes.numel() + self.action_synapses.edge_count)

    def parameter_tensors(self) -> tuple[torch.Tensor, ...]:
        return (self.bank.prototypes, self.action_synapses.edge_weight)

    def _slot_trace(self, slot_index: int) -> torch.Tensor:
        return self.bank.slot_code(int(slot_index)).to(self.device)

    def _clear_slot(self, slot_index: int) -> None:
        slot = int(slot_index)
        mask = self.action_synapses.pre_index == slot
        self.action_synapses.edge_weight.masked_fill_(mask, 0.0)

    def _validate_action(self, action_symbol: int) -> int:
        action = int(action_symbol)
        if not 0 <= action < self.action_count:
            raise ValueError("identity organ action is outside the motor alphabet")
        return action

    @torch.no_grad()
    def learn(self, cortical_context: torch.Tensor, action_symbol: int) -> CueBindingResult:
        """Bind one settled cue/action pair using local synapse updates."""

        action = self._validate_action(action_symbol)
        binding = self.bank.route(cortical_context, learn=True)
        if binding.slot_index is None:
            raise RuntimeError("identity organ binding did not return a slot")
        slot = int(binding.slot_index)
        if binding.replaced:
            self._clear_slot(slot)
            self.replacement_count += 1
        trace = self._slot_trace(slot)
        target = torch.zeros(self.action_count, device=self.device)
        target[action] = 1.0
        repeats = int(self.config.identity_organ_learning_repeats)
        for _ in range(repeats):
            logits = self.action_synapses.forward(trace)
            probabilities = torch.softmax(logits, dim=0)
            self.action_synapses.local_update(
                target - probabilities,
                trace,
                learning_rate=self.config.identity_organ_learning_rate,
                weight_decay=0.0,
            )
        self.write_count += 1
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
        if not enabled:
            return IdentityRecall(
                action_evidence=zero,
                action_probabilities=uniform,
                slot_index=None,
                similarity=0.0,
                source="shared-fallback",
                provenance=IDENTITY_ORGAN_DISABLED_PROVENANCE,
                used=False,
            )
        binding = self.bank.route(cortical_context, learn=False)
        if binding.slot_index is None:
            return IdentityRecall(
                action_evidence=zero,
                action_probabilities=uniform,
                slot_index=None,
                similarity=float(binding.similarity),
                source="shared-fallback",
                provenance=IDENTITY_ORGAN_UNBOUND_PROVENANCE,
                used=False,
            )
        logits = self.action_synapses.forward(self._slot_trace(binding.slot_index))
        probabilities = torch.softmax(logits, dim=0)
        return IdentityRecall(
            action_evidence=logits.detach().clone(),
            action_probabilities=probabilities.detach().clone(),
            slot_index=int(binding.slot_index),
            similarity=float(binding.similarity),
            source="identity-route",
            provenance=IDENTITY_ORGAN_BOUND_PROVENANCE,
            used=True,
        )

    @torch.no_grad()
    def lesion(self) -> None:
        """Remove all identity bindings and action evidence in-place."""

        self.bank.occupied.zero_()
        self.bank.prototypes.zero_()
        self.bank.visits.zero_()
        self.action_synapses.edge_weight.zero_()

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
            "match_threshold": self.config.identity_organ_match_threshold,
            "update_rate": self.config.identity_organ_update_rate,
            "learning_rate": self.config.identity_organ_learning_rate,
            "learning_repeats": self.config.identity_organ_learning_repeats,
            "evidence_gain": self.config.identity_organ_evidence_gain,
            "bank": self.bank.to_payload(),
            "action_synapses": self.action_synapses.to_payload(),
            "write_count": self.write_count,
            "replacement_count": self.replacement_count,
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
            self.config.identity_organ_match_threshold,
            self.config.identity_organ_update_rate,
            self.config.identity_organ_learning_rate,
            self.config.identity_organ_learning_repeats,
            self.config.identity_organ_evidence_gain,
        )
        actual = (
            int(payload["capacity"]),
            int(payload["pattern_dim"]),
            int(payload["action_count"]),
            float(payload["match_threshold"]),
            float(payload["update_rate"]),
            float(payload["learning_rate"]),
            int(payload["learning_repeats"]),
            float(payload["evidence_gain"]),
        )
        if actual != expected:
            raise ValueError("identity organ checkpoint architecture does not match")
        self.bank.load_payload(dict(payload["bank"]))
        self.action_synapses.load_payload(dict(payload["action_synapses"]))
        self.write_count = int(payload.get("write_count", 0))
        self.replacement_count = int(payload.get("replacement_count", 0))
        if min(self.write_count, self.replacement_count) < 0:
            raise ValueError("identity organ counters cannot be negative")


__all__ = [
    "IDENTITY_ORGAN_BOUND_PROVENANCE",
    "IDENTITY_ORGAN_CHECKPOINT_FORMAT",
    "IDENTITY_ORGAN_DISABLED_PROVENANCE",
    "IDENTITY_ORGAN_UNBOUND_PROVENANCE",
    "IDENTITY_ORGAN_VERSION",
    "CueIdentityOrgan",
    "IdentityRecall",
]
