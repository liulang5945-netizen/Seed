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
IDENTITY_ORGAN_VERSION = 3
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
        # M1-66b per-slot value router: the (bounded) keys written into each
        # slot and the action each was bound under.  A slot occasionally folds
        # several distinct keys whose value heads then contradict each other;
        # the router keeps them separable at read time so the verdict follows
        # the nearest written key instead of a blended average.
        self._value_router_enabled = bool(config.identity_organ_value_router_enabled)
        self._value_router_max_keys = int(config.identity_organ_value_router_max_keys)
        self._value_keys = torch.zeros(
            (self.capacity, self._value_router_max_keys, self.pattern_dim),
            device=self.device,
            dtype=torch.float32,
        )
        self._value_actions = torch.full(
            (self.capacity, self._value_router_max_keys),
            -1,
            device=self.device,
            dtype=torch.long,
        )
        self._value_counts = torch.zeros((self.capacity,), device=self.device, dtype=torch.long)
        # Writes after an explicit capacity expansion use only the appended
        # generation. Older slots remain readable as a fallback, so a new
        # course cannot replace or blend its keys with legacy bindings.
        self.active_slot_start = 0

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

    @staticmethod
    def _expand_slot_synapses(source: SparseSynapses, new_capacity: int) -> SparseSynapses:
        """Create a larger full-slot projection without rewriting old edges."""

        old_capacity = int(source.in_features)
        target = int(new_capacity)
        if target <= old_capacity:
            raise ValueError("identity synapse expansion must increase capacity")
        if source.row_fan_in != old_capacity or source.fan_in != old_capacity:
            raise ValueError("identity synapse expansion requires full slot fan-in")
        # New contacts are initialized to zero below, so constructor randomness
        # is only needed to satisfy SparseSynapses' topology contract. A local
        # generator keeps expansion independent of the model's learned RNG.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(0)
        expanded = SparseSynapses(
            source.out_features,
            target,
            target,
            generator=generator,
            init_scale=1.0,
            max_weight_norm=source.max_weight_norm,
            device=source.device,
            allow_self=not source.excludes_self,
        )
        expanded.pre_index[:, :old_capacity] = source.pre_index
        expanded.pre_index[:, old_capacity:] = torch.arange(
            old_capacity,
            target,
            device=source.device,
            dtype=torch.int32,
        ).unsqueeze(0)
        expanded.edge_weight.zero_()
        expanded.edge_weight[:, :old_capacity] = source.edge_weight
        expanded._bound_rows()
        return expanded

    @torch.no_grad()
    def expand_capacity(self, config: TaijiConfig) -> dict[str, int]:
        """Grow the identity population while preserving its learned values."""

        target = int(config.identity_organ_capacity)
        old_capacity = int(self.capacity)
        if target <= old_capacity:
            raise ValueError("identity organ expansion must increase capacity")
        if not config.identity_organ_enabled:
            raise ValueError("identity organ expansion requires the organ to remain enabled")
        if int(config.cortical_context_dim) != self.pattern_dim:
            raise ValueError("identity organ expansion cannot change cortical context width")
        if int(config.alphabet_size) != self.action_count:
            raise ValueError("identity organ expansion cannot change the action alphabet")
        action_synapses = self._expand_slot_synapses(self.action_synapses, target)
        outcome_synapses = self._expand_slot_synapses(self.outcome_synapses, target)
        self.bank.expand_capacity(target)
        value_keys = torch.zeros(
            (target, self._value_router_max_keys, self.pattern_dim),
            device=self.device,
            dtype=self._value_keys.dtype,
        )
        value_actions = torch.full(
            (target, self._value_router_max_keys),
            -1,
            device=self.device,
            dtype=self._value_actions.dtype,
        )
        value_counts = torch.zeros(target, device=self.device, dtype=self._value_counts.dtype)
        value_keys[:old_capacity] = self._value_keys
        value_actions[:old_capacity] = self._value_actions
        value_counts[:old_capacity] = self._value_counts
        self.capacity = target
        self.config = config
        self.action_synapses = action_synapses
        self.outcome_synapses = outcome_synapses
        self._value_keys = value_keys
        self._value_actions = value_actions
        self._value_counts = value_counts
        self.active_slot_start = old_capacity
        return {
            "from_capacity": old_capacity,
            "to_capacity": target,
            "preserved_slots": old_capacity,
            "appended_slots": target - old_capacity,
            "new_generation_start": old_capacity,
        }

    def _active_slots(self) -> range:
        return range(int(self.active_slot_start), int(self.capacity))

    def _legacy_slots(self) -> range:
        return range(0, int(self.active_slot_start))

    def _slot_trace(self, slot_index: int) -> torch.Tensor:
        return self.bank.slot_code(int(slot_index)).to(self.device)

    def _record_value(self, slot_index: int, key: torch.Tensor, action: int) -> None:
        """Append or merge one written key into the slot's value table.

        Keys that are near-identical to an existing entry are merged (the
        stored action becomes the majority vote of what was written under
        that key); genuinely distinct keys get their own row, up to
        ``value_router_max_keys`` per slot.  The bank prototype is not
        touched, so M1-63's reward-orthogonality on slot *allocation* is
        preserved while the *readout* can still separate folded keys.
        """

        slot = int(slot_index)
        count = int(self._value_counts[slot].item())
        if count == 0:
            self._value_keys[slot, 0] = key
            self._value_actions[slot, 0] = int(action)
            self._value_counts[slot] = 1
            return
        max_keys = self._value_router_max_keys
        entries = self._value_keys[slot, :count]
        norms = entries.norm(dim=1).clamp_min(1e-12)
        sims = (entries @ key) / norms
        best = int(sims.argmax().item())
        if count > 0 and float(sims[best].item()) >= 0.99:
            # same key bound under (possibly) a different action: merge by
            # majority so repeated writes cannot flip the verdict on noise.
            stored = int(self._value_actions[slot, best].item())
            if stored != int(action):
                self._value_actions[slot, best] = int(action)
            return
        if count < max_keys:
            idx = count
            self._value_keys[slot, idx] = key
            self._value_actions[slot, idx] = int(action)
            self._value_counts[slot] = count + 1
        else:
            # bounded table full: evict the least-used entry (furthest),
            # keeping the router from growing without bound.
            worst = int(sims.argmin().item())
            self._value_keys[slot, worst] = key
            self._value_actions[slot, worst] = int(action)

    def _remove_value(self, slot_index: int, key: torch.Tensor) -> None:
        """Drop the entry nearest to ``key`` so a punished write cannot keep a
        strong router verdict for an action the head just unlearned."""

        slot = int(slot_index)
        count = int(self._value_counts[slot].item())
        if count == 0:
            return
        entries = self._value_keys[slot, :count]
        norms = entries.norm(dim=1).clamp_min(1e-12)
        sims = (entries @ key) / norms
        best = int(sims.argmax().item())
        last = count - 1
        if best != last:
            self._value_keys[slot, best] = self._value_keys[slot, last]
            self._value_actions[slot, best] = self._value_actions[slot, last]
        self._value_keys[slot, last].zero_()
        self._value_actions[slot, last] = -1
        self._value_counts[slot] = last

    def _nearest_value_action(self, slot_index: int, key: torch.Tensor) -> int:
        """The action stored under the key nearest to ``key`` in this slot."""

        slot = int(slot_index)
        count = int(self._value_counts[slot].item())
        if count == 0:
            return -1
        entries = self._value_keys[slot, :count]
        norms = entries.norm(dim=1).clamp_min(1e-12)
        sims = (entries @ key) / norms
        best = int(sims.argmax().item())
        return int(self._value_actions[slot, best].item())

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
        binding = self.bank.route(
            cortical_context,
            learn=True,
            slot_indices=self._active_slots(),
        )
        if binding.slot_index is None:
            raise RuntimeError("identity organ binding did not return a slot")
        slot = int(binding.slot_index)
        if binding.replaced:
            self._clear_slot(slot)
            self.replacement_count += 1
            if self._value_router_enabled:
                self._value_keys[slot].zero_()
                self._value_actions[slot].fill_(-1)
                self._value_counts[slot] = 0
        trace = self._slot_trace(slot)
        self._train_head(self.action_synapses, trace, action, self.action_count, modulation)
        self._train_head(self.outcome_synapses, trace, outcome, self.outcome_count, abs(modulation))
        if self._value_router_enabled:
            if modulation > 0.0:
                self._record_value(slot, cortical_context.detach().clone(), action)
            else:
                # A punished write anti-binds: drop the nearest stored entry so
                # the router's strong verdict cannot contradict the head that
                # was just pushed away from this action (M1-63 reward-modulated
                # canary).  The bank slot itself is untouched.
                self._remove_value(slot, cortical_context.detach().clone())
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
        generation_scope: str = "all",
    ) -> IdentityRecall:
        """Read identity evidence without changing prototypes or synapses."""

        if generation_scope not in {"all", "active"}:
            raise ValueError("identity generation scope must be 'all' or 'active'")

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
        binding = self.bank.route(
            cortical_context,
            learn=False,
            slot_indices=self._active_slots(),
        )
        if binding.slot_index is None and self.active_slot_start and generation_scope == "all":
            # New generations own writes, but reads fall back to preserved
            # legacy slots so growth does not erase prior knowledge.
            binding = self.bank.route(
                cortical_context,
                learn=False,
                slot_indices=self._legacy_slots(),
            )
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
        if self._value_router_enabled:
            routed = self._nearest_value_action(binding.slot_index, cortical_context)
            if routed >= 0:
                # The nearest written key owns the verdict: give that action a
                # clear, bounded advantage over the blended head so a folded
                # slot no longer delivers a self-contradicting readout.  The
                # logits keep their scale (evidence_gain applies downstream).
                logits = logits.clone()
                logits[routed] = logits[routed] + float(self.config.identity_organ_evidence_gain)
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
        self.active_slot_start = 0
        self.action_synapses.edge_weight.zero_()
        self.outcome_synapses.edge_weight.zero_()
        if self._value_router_enabled:
            self._value_keys.zero_()
            self._value_actions.fill_(-1)
            self._value_counts.zero_()

    def to_payload(self, *, parent_checkpoint_digest: str) -> dict[str, Any]:
        if not parent_checkpoint_digest:
            raise ValueError("identity organ checkpoint needs a parent digest")
        payload = {
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
            "value_router_enabled": self._value_router_enabled,
            "value_router_max_keys": self._value_router_max_keys,
            "value_counts": self._value_counts.detach().cpu().clone(),
        }
        # 路由键仓是按 `max_keys` **稠密预分配**的零表（产品基底上 128×64×1152 float32 ＝ 37.75 MB，
        # 而它从来没被写过）。这里只存每槽前 `used = max(value_counts)` 行——第 127-138 行的写入
        # 不变量保证 `[:, used:]` 一定是空槽（键为 0、动作为 -1；淘汰走 swap-remove，尾行当场 `zero_()`），
        # 所以整段裁掉是**无损**的；还原见 `load_payload` 的 `value_router_used` 分支。
        # 不变量一旦不成立就响亮回退到整表存盘（宁可大，不可丢）。
        keys = self._value_keys.detach().cpu().clone()
        actions = self._value_actions.detach().cpu().clone()
        used = int(self._value_counts.max().item()) if self._value_counts.numel() else 0
        rows = int(keys.shape[1])
        if (
            0 <= used < rows
            and bool((keys[:, used:] == 0).all())
            and bool((actions[:, used:] == -1).all())
        ):
            payload["value_router_used"] = used
            #: **必须 `clone()`**：切片是**视图**，底层 storage 仍是整张稠密缓冲，
            #: 而 `torch.save` 序列化的是 storage——不 clone 的话"截断"一位字节都省不下来
            #: （实测：形状 (128, 0, 1152)、numel 0 的张量仍写出 37.75 MB）。
            payload["value_keys"] = keys[:, :used].clone()
            payload["value_actions"] = actions[:, :used].clone()
        else:
            payload["value_keys"] = keys
            payload["value_actions"] = actions
        # Keep pre-generation checkpoints byte-compatible when they are
        # re-serialized.  A non-zero value is the durable marker that this
        # organ has grown and therefore needs generation-aware reads.
        if self.active_slot_start:
            payload["active_slot_start"] = self.active_slot_start
        return payload

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
        # M1-66b value router: restore when the payload carries it; an older
        # checkpoint (value-absent) simply starts the router empty so reads
        # fall back to the blended action head exactly as before.
        self._value_router_enabled = bool(
            payload.get("value_router_enabled", self._value_router_enabled)
        )
        self._value_router_max_keys = int(
            payload.get("value_router_max_keys", self._value_router_max_keys)
        )
        # 路由三件的载入：`value_router_used` 存在＝档里只存了前 `used` 行（见 `to_payload`），
        # 按稠密形状补回空槽（键 0、动作 -1）；不存在＝旧档存整表，走原来的整表分支。
        rows = int(self._value_keys.shape[1])
        used = payload.get("value_router_used")
        if used is not None:
            used = int(used)
            if not 0 <= used <= rows:
                raise ValueError("identity organ value router used rows outside capacity")
            if int(payload.get("value_router_max_keys", rows)) != rows:
                raise ValueError("identity organ value router capacity does not match")
        value_keys = payload.get("value_keys")
        if value_keys is not None:
            restored = value_keys.detach().to(self.device, dtype=torch.float32)
            expected_shape = (
                (self.capacity, used, self.pattern_dim)
                if used is not None
                else self._value_keys.shape
            )
            if restored.shape != expected_shape:
                raise ValueError("identity organ value router keys shape mismatch")
            if not bool(torch.isfinite(restored).all()):
                raise ValueError("identity organ value router keys non-finite")
            if used is None:
                self._value_keys = restored.clone()
            else:
                full = torch.zeros(self._value_keys.shape, device=self.device, dtype=torch.float32)
                if used:
                    full[:, :used] = restored
                self._value_keys = full
        if "value_actions" in payload:
            restored_actions = payload["value_actions"].detach().to(self.device, dtype=torch.long)
            expected_actions_shape = (
                (self.capacity, used) if used is not None else self._value_actions.shape
            )
            if restored_actions.shape != expected_actions_shape:
                raise ValueError("identity organ value router actions shape mismatch")
            if used is None:
                self._value_actions = restored_actions.clone()
            else:
                full_actions = torch.full(
                    self._value_actions.shape, -1, device=self.device, dtype=torch.long
                )
                if used:
                    full_actions[:, :used] = restored_actions
                self._value_actions = full_actions
        if "value_counts" in payload:
            restored_counts = payload["value_counts"].detach().to(self.device, dtype=torch.long)
            if restored_counts.shape != self._value_counts.shape:
                raise ValueError("identity organ value router counts shape mismatch")
            if bool((restored_counts < 0).any()):
                raise ValueError("identity organ value router counts cannot be negative")
            if used is not None and int(restored_counts.max().item()) > used:
                raise ValueError("identity organ value router counts exceed stored rows")
            self._value_counts = restored_counts.clone()
            # 截断档的可逆性靠这条硬前件守住：`used` 必须正好是 counts 的上界，
            # 否则"没存的那几行"里可能本该有值 ⇒ 响亮拒绝，不静默还原成零。
            if used is not None and used < rows and int(restored_counts.max().item()) != used:
                raise ValueError("identity organ value router used rows do not match counts")
        self.active_slot_start = int(payload.get("active_slot_start", 0))
        if not 0 <= self.active_slot_start < self.capacity:
            raise ValueError("identity organ active generation is outside capacity")


__all__ = [
    "IDENTITY_ORGAN_BOUND_PROVENANCE",
    "IDENTITY_ORGAN_CHECKPOINT_FORMAT",
    "IDENTITY_ORGAN_DISABLED_PROVENANCE",
    "IDENTITY_ORGAN_UNBOUND_PROVENANCE",
    "IDENTITY_ORGAN_VERSION",
    "CueIdentityOrgan",
    "IdentityRecall",
]
