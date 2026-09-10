"""Fast/slow continuation instance for the K workers (B3 pilot arm FS).

Preregistration: ``plans/reference/M4V2_B3_K_PILOT_PREREGISTRATION
_20260910.md``.  Frozen semantics:

- every learnable head tensor ``T`` is split into ``slow + fast`` with the
  effective weights ``= slow + fast``;
- **wake**: the same local delta the continuation arm would apply is
  written into ``fast`` (``slow`` stays bit-identical); because the delta
  is a function of the current effective state, the wake trajectory is
  bit-identical to direct continuation;
- **sleep**: a deterministic sample of already-consumed real experiences
  is replayed -- the local delta (computed against the current effective
  state) is applied to ``slow`` -- then ``slow += fast`` and ``fast = 0``;
  consolidation never changes the effective weights;
- the checkpoint stores slow and fast separately and restores the
  effective state bit-for-bit.

This is a variant module (like ``k_widened``): the frozen
``k_continuation`` v1 and ``k_widened`` v2 contracts are untouched.  No
optimizer state; learning stays detached local-delta.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import torch

from .internalization import content_digest
from .semantic_training import StructuredSemanticLearner
from .semantic_transition import StructuredSemanticTransitionLearner

TAIJI_K_FAST_SLOW_FORMAT = "taiji-k-fast-slow-v1"
TAIJI_K_FAST_SLOW_VERSION = 1
FAST_SLOW_WORKERS = ("k1.semantic", "k2.transition")


def _text(value: Any, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} cannot be empty")
    return normalized


def _digest(value: Any, name: str) -> str:
    normalized = _text(value, name)
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return normalized


def replay_sample_indices(
    *, buffer_size: int, sample_count: int, digest: str
) -> tuple[int, ...]:
    """Deterministic without-replacement sample of replay positions."""

    if buffer_size <= 0:
        raise ValueError("replay buffer is empty")
    if not 0 < sample_count <= buffer_size:
        raise ValueError("replay sample count must be within the buffer")
    seed_material = content_digest({"buffer": buffer_size, "digest": digest})

    def draw(index: int) -> int:
        return int(content_digest({"seed": seed_material, "counter": index})[:16], 16)

    order = list(range(buffer_size))
    for position in range(buffer_size - 1, 0, -1):
        order[position], order[draw(position) % (position + 1)] = (
            order[draw(position) % (position + 1)],
            order[position],
        )
    return tuple(order[:sample_count])


class FastSlowKInstance:
    """One fast/slow continuation instance over the K1/K2 workers."""

    def __init__(
        self,
        *,
        semantic_parent_checkpoint: Mapping[str, Any],
        transition_parent_checkpoint: Mapping[str, Any],
        parent_worker_bundle_digest: str,
        source_manifest_digest: str,
        parent_checkpoint_digest: str,
    ) -> None:
        self.format = TAIJI_K_FAST_SLOW_FORMAT
        self.version = TAIJI_K_FAST_SLOW_VERSION
        self.parent_checkpoint_digest = _digest(
            parent_checkpoint_digest, "parent_checkpoint_digest"
        )
        self.parent_worker_bundle_digest = _digest(
            parent_worker_bundle_digest, "parent_worker_bundle_digest"
        )
        self.source_manifest_digest = _digest(
            source_manifest_digest, "source_manifest_digest"
        )
        self._parent_checkpoints = {
            "k1.semantic": dict(semantic_parent_checkpoint),
            "k2.transition": dict(transition_parent_checkpoint),
        }
        self.slow: dict[str, dict[str, torch.Tensor]] = {}
        self.fast: dict[str, dict[str, torch.Tensor]] = {}
        for worker_id, checkpoint in self._parent_checkpoints.items():
            state = dict(checkpoint["state_dict"])
            self.slow[worker_id] = {
                key: value.detach().cpu().clone() for key, value in state.items()
            }
            self.fast[worker_id] = {
                key: torch.zeros_like(value) for key, value in state.items()
            }
        self.replay_digests: list[str] = []
        self.wake_steps = 0
        self.replay_steps = 0
        self.consolidations = 0
        self._scratch = self._build_scratch()

    def _build_scratch(self) -> dict[str, Any]:
        return {
            "k1.semantic": StructuredSemanticLearner.from_checkpoint(
                copy.deepcopy(self._parent_checkpoints["k1.semantic"]), device="cpu"
            ),
            "k2.transition": StructuredSemanticTransitionLearner.from_checkpoint(
                copy.deepcopy(self._parent_checkpoints["k2.transition"]), device="cpu"
            ),
        }

    # -- state helpers ----------------------------------------------------

    def _effective_state(self, worker_id: str) -> dict[str, torch.Tensor]:
        return {
            key: self.slow[worker_id][key] + self.fast[worker_id][key]
            for key in self.slow[worker_id]
        }

    def _sync_scratch(self, worker_id: str) -> None:
        self._scratch[worker_id].load_state_dict(self._effective_state(worker_id))

    def effective_state_digest(self, worker_id: str) -> str:
        return content_digest(self._effective_state(worker_id))

    def slow_state_digest(self, worker_id: str) -> str:
        return content_digest(self.slow[worker_id])

    def fast_norm(self, worker_id: str) -> float:
        total = 0.0
        for value in self.fast[worker_id].values():
            total += float(torch.sum(value * value))
        return total**0.5

    def is_fast_zero(self, worker_id: str) -> bool:
        return all(
            not torch.any(value != 0) for value in self.fast[worker_id].values()
        )

    def parameter_count(self) -> int:
        return (
            sum(int(value.numel()) for value in self.slow["k1.semantic"].values())
            + sum(int(value.numel()) for value in self.slow["k2.transition"].values())
        )

    def parameter_bytes(self) -> int:
        total = 0
        for worker_id in FAST_SLOW_WORKERS:
            for value in self.slow[worker_id].values():
                total += value.numel() * value.element_size()
            for value in self.fast[worker_id].values():
                total += value.numel() * value.element_size()
        return total

    # -- wake / sleep -----------------------------------------------------

    def wake_experience(
        self,
        experience: Any,
        *,
        semantic_epochs: int,
        semantic_lr: float,
        transition_epochs: int,
        transition_lr: float,
    ) -> None:
        """Apply the continuation local delta to ``fast`` (slow untouched)."""

        for worker_id, example, epochs, lr in (
            (
                "k1.semantic",
                experience.semantic_example,
                semantic_epochs,
                semantic_lr,
            ),
            (
                "k2.transition",
                experience.transition_example,
                transition_epochs,
                transition_lr,
            ),
        ):
            scratch = self._scratch[worker_id]
            self._sync_scratch(worker_id)
            before = {
                key: value.detach().cpu().clone()
                for key, value in scratch.state_dict().items()
            }
            scratch.fit((example,), epochs=epochs, learning_rate=lr)
            after = scratch.state_dict()
            for key in before:
                self.fast[worker_id][key] += after[key].detach().cpu() - before[key]
            # scratch post-fit already equals the new effective state.
        self.wake_steps += 1
        self.replay_digests.append(
            _digest(experience.experience_digest, "experience_digest")
        )

    def replay_experience(
        self,
        experience: Any,
        *,
        semantic_epochs: int,
        semantic_lr: float,
        transition_epochs: int,
        transition_lr: float,
    ) -> None:
        """Apply the local delta of one replayed experience to ``slow``."""

        for worker_id, example, epochs, lr in (
            (
                "k1.semantic",
                experience.semantic_example,
                semantic_epochs,
                semantic_lr,
            ),
            (
                "k2.transition",
                experience.transition_example,
                transition_epochs,
                transition_lr,
            ),
        ):
            scratch = self._scratch[worker_id]
            self._sync_scratch(worker_id)
            before = {
                key: value.detach().cpu().clone()
                for key, value in scratch.state_dict().items()
            }
            scratch.fit((example,), epochs=epochs, learning_rate=lr)
            after = scratch.state_dict()
            for key in before:
                self.slow[worker_id][key] += after[key].detach().cpu() - before[key]
        self.replay_steps += 1

    def consolidate(self) -> None:
        """``slow += fast``; ``fast = 0``; effective weights unchanged."""

        for worker_id in FAST_SLOW_WORKERS:
            for key in self.slow[worker_id]:
                self.slow[worker_id][key] += self.fast[worker_id][key]
                self.fast[worker_id][key] = torch.zeros_like(self.fast[worker_id][key])
        self.consolidations += 1

    def effective_learners(
        self,
    ) -> tuple[StructuredSemanticLearner, StructuredSemanticTransitionLearner]:
        k1 = StructuredSemanticLearner.from_checkpoint(
            copy.deepcopy(self._parent_checkpoints["k1.semantic"]), device="cpu"
        )
        k1.load_state_dict(self._effective_state("k1.semantic"))
        k1._apply_fact_feature_masks()
        k2 = StructuredSemanticTransitionLearner.from_checkpoint(
            copy.deepcopy(self._parent_checkpoints["k2.transition"]), device="cpu"
        )
        k2.load_state_dict(self._effective_state("k2.transition"))
        k2._apply_transition_input_masks()
        return k1, k2

    # -- checkpoint -------------------------------------------------------

    def checkpoint(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": self.version,
            "kind": "fast-slow-instance",
            "parent_checkpoint_digest": self.parent_checkpoint_digest,
            "parent_worker_bundle_digest": self.parent_worker_bundle_digest,
            "source_manifest_digest": self.source_manifest_digest,
            "parent_checkpoints": dict(self._parent_checkpoints),
            "slow": {worker: self.slow[worker] for worker in FAST_SLOW_WORKERS},
            "fast": {worker: self.fast[worker] for worker in FAST_SLOW_WORKERS},
            "replay_digests": list(self.replay_digests),
            "wake_steps": self.wake_steps,
            "replay_steps": self.replay_steps,
            "consolidations": self.consolidations,
        }

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> FastSlowKInstance:
        if payload.get("format") != TAIJI_K_FAST_SLOW_FORMAT:
            raise ValueError("unsupported fast/slow instance format")
        if int(payload.get("version", -1)) != TAIJI_K_FAST_SLOW_VERSION:
            raise ValueError("unsupported fast/slow instance version")
        item = cls(
            semantic_parent_checkpoint=dict(payload["parent_checkpoints"]["k1.semantic"]),
            transition_parent_checkpoint=dict(
                payload["parent_checkpoints"]["k2.transition"]
            ),
            parent_worker_bundle_digest=str(payload["parent_worker_bundle_digest"]),
            source_manifest_digest=str(payload["source_manifest_digest"]),
            parent_checkpoint_digest=str(payload["parent_checkpoint_digest"]),
        )
        for worker_id in FAST_SLOW_WORKERS:
            item.slow[worker_id] = {
                key: value.detach().cpu().clone()
                for key, value in payload["slow"][worker_id].items()
            }
            item.fast[worker_id] = {
                key: value.detach().cpu().clone()
                for key, value in payload["fast"][worker_id].items()
            }
            item._sync_scratch(worker_id)
        item.replay_digests = [str(value) for value in payload["replay_digests"]]
        item.wake_steps = int(payload["wake_steps"])
        item.replay_steps = int(payload["replay_steps"])
        item.consolidations = int(payload["consolidations"])
        return item
