"""Widened single-instance K continuation bundle (capacity-parity candidate).

The C-entry capacity-parity contract requires the candidate to reach the
fixed-large parameter budget (38,664 bytes = 9,666 parameters) without
becoming a replica ensemble.  This module freezes the widened route:

- **Topology**: one candidate bundle containing four native learners —
  two :class:`StructuredSemanticLearner` (K1) and two
  :class:`StructuredSemanticTransitionLearner` (K2).  Channel 1
  (``forward``) consumes the course in preregistered episode order;
  channel 2 (``anchored``) consumes the same experiences in a
  deterministic permutation anchored by the K3 projection digests of the
  course (every tick's update receipt carries the ordering witness).
- **Readout**: logit-domain channel sum — the two channel learners'
  head logits are added before the usual sigmoid/softmax reductions,
  which is mathematically distinct from the fixed-large probability
  average over same-order replicas.
- **Budget**: each channel consumes the identical per-head fit stream as
  one fixed-large replica (K1 2,083 + K2 5,043 = 7,126 local-delta
  updates), so the bundle performs exactly 14,252 real parameter updates
  per cell.

The K continuation v1 contract stays untouched; the widened variant is a
versioned extension (``taiji-k-continuation-widened-v2``).  Learning is
still detached local-delta with no optimizer state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .internalization import content_digest
from .k_worker_manifest import K_WORKER_IDS
from .semantic_training import StructuredSemanticLearner
from .semantic_transition import StructuredSemanticTransitionLearner

TAIJI_K_WIDENED_FORMAT = "taiji-k-continuation-widened-v2"
TAIJI_K_WIDENED_VERSION = 2
WIDENED_CHANNELS = ("forward", "anchored")
WIDENED_LEARNABLE_WORKERS = ("k1.semantic", "k2.transition")


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


def anchored_permutation(
    *, count: int, anchor_digests: Sequence[str], course_digest: str
) -> tuple[int, ...]:
    """Deterministic non-identity permutation anchored by K3 digests.

    The anchor is the ordered tuple of K3 projection digests of the course
    plus the course digest; the permutation is derived with SHA-256 counter
    expansion (Fisher-Yates over 64-bit draws).  It must never be the
    identity, so an anchored channel always consumes a genuinely different
    experience order than the forward channel.
    """

    if count <= 0:
        raise ValueError("anchored permutation requires a positive count")
    seed_material = content_digest(
        {"anchor": [str(item) for item in anchor_digests], "course": course_digest}
    )

    def draw(index: int) -> int:
        block = content_digest({"seed": seed_material, "counter": index})
        return int(block[:16], 16)

    order = list(range(count))
    for position in range(count - 1, 0, -1):
        order[position], order[draw(position) % (position + 1)] = (
            order[draw(position) % (position + 1)],
            order[position],
        )
    if tuple(order) == tuple(range(count)):
        order[0], order[-1] = order[-1], order[0]
    return tuple(order)


@dataclass(frozen=True)
class WidenedChannelReceipt:
    """Per-channel update audit for one widened continuation run."""

    channel: str
    order_digest: str
    k1_training_steps: int
    k2_training_steps: int
    k1_parameter_delta_digest: str
    k2_parameter_delta_digest: str
    k1_parameter_delta_norm: float
    k2_parameter_delta_norm: float
    projection_anchor_digests: tuple[str, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "order_digest": self.order_digest,
            "k1_training_steps": int(self.k1_training_steps),
            "k2_training_steps": int(self.k2_training_steps),
            "k1_parameter_delta_digest": self.k1_parameter_delta_digest,
            "k2_parameter_delta_digest": self.k2_parameter_delta_digest,
            "k1_parameter_delta_norm": float(self.k1_parameter_delta_norm),
            "k2_parameter_delta_norm": float(self.k2_parameter_delta_norm),
            "projection_anchor_digests": list(self.projection_anchor_digests),
        }


def _delta_digest(before: Any, after: Any) -> str:
    return content_digest(
        {
            "before": before.state_dict(),
            "after": after.state_dict(),
        }
    )


def _delta_norm(before: Any, after: Any) -> float:
    total = 0.0
    for name, value in after.state_dict().items():
        delta = value.detach().cpu() - before.state_dict()[name].detach().cpu()
        total += float(torch.sum(delta * delta))
    return total**0.5


def _copy_state(learner: Any) -> Any:
    return type(learner).from_checkpoint(learner.checkpoint(), device="cpu")


class WidenedKBundle:
    """One widened candidate bundle: two channels over K1/K2 learners."""

    def __init__(
        self,
        *,
        semantic_parent_checkpoint: Mapping[str, Any],
        transition_parent_checkpoint: Mapping[str, Any],
        parent_worker_bundle_digest: str,
        source_manifest_digest: str,
        parent_checkpoint_digest: str,
    ) -> None:
        self.format = TAIJI_K_WIDENED_FORMAT
        self.version = TAIJI_K_WIDENED_VERSION
        self.parent_checkpoint_digest = _digest(
            parent_checkpoint_digest, "parent_checkpoint_digest"
        )
        self.parent_worker_bundle_digest = _digest(
            parent_worker_bundle_digest, "parent_worker_bundle_digest"
        )
        self.source_manifest_digest = _digest(
            source_manifest_digest, "source_manifest_digest"
        )
        self.channels: dict[str, dict[str, Any]] = {}
        for channel in WIDENED_CHANNELS:
            self.channels[channel] = {
                "k1.semantic": StructuredSemanticLearner.from_checkpoint(
                    semantic_parent_checkpoint, device="cpu"
                ),
                "k2.transition": StructuredSemanticTransitionLearner.from_checkpoint(
                    transition_parent_checkpoint, device="cpu"
                ),
            }

    @property
    def parameter_count(self) -> int:
        return sum(int(parameter.numel()) for learner in self._learners() for parameter in learner.parameters())

    @property
    def parameter_bytes(self) -> int:
        return sum(
            int(parameter.numel() * parameter.element_size())
            for learner in self._learners()
            for parameter in learner.parameters()
        )

    def _learners(self) -> tuple[Any, ...]:
        return tuple(
            self.channels[channel][worker_id]
            for channel in WIDENED_CHANNELS
            for worker_id in WIDENED_LEARNABLE_WORKERS
        )

    def worker_checkpoint_digests(self) -> dict[str, str]:
        return {
            worker_id: content_digest(
                {
                    channel: self.channels[channel][worker_id].checkpoint()
                    for channel in WIDENED_CHANNELS
                }
            )
            for worker_id in WIDENED_LEARNABLE_WORKERS
        }

    def checkpoint(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": self.version,
            "kind": "widened-bundle",
            "parent_checkpoint_digest": self.parent_checkpoint_digest,
            "parent_worker_bundle_digest": self.parent_worker_bundle_digest,
            "source_manifest_digest": self.source_manifest_digest,
            "channels": {
                channel: {
                    worker_id: self.channels[channel][worker_id].checkpoint()
                    for worker_id in WIDENED_LEARNABLE_WORKERS
                }
                for channel in WIDENED_CHANNELS
            },
            "worker_checkpoint_digests": self.worker_checkpoint_digests(),
        }

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> WidenedKBundle:
        if payload.get("format") != TAIJI_K_WIDENED_FORMAT:
            raise ValueError("unsupported widened bundle format")
        if int(payload.get("version", -1)) != TAIJI_K_WIDENED_VERSION:
            raise ValueError("unsupported widened bundle version")
        channels = payload["channels"]
        item = cls(
            semantic_parent_checkpoint=channels["forward"]["k1.semantic"],
            transition_parent_checkpoint=channels["forward"]["k2.transition"],
            parent_worker_bundle_digest=payload["parent_worker_bundle_digest"],
            source_manifest_digest=payload["source_manifest_digest"],
            parent_checkpoint_digest=payload["parent_checkpoint_digest"],
        )
        for channel in WIDENED_CHANNELS:
            item.channels[channel]["k1.semantic"] = StructuredSemanticLearner.from_checkpoint(
                channels[channel]["k1.semantic"], device="cpu"
            )
            item.channels[channel][
                "k2.transition"
            ] = StructuredSemanticTransitionLearner.from_checkpoint(
                channels[channel]["k2.transition"], device="cpu"
            )
        if item.worker_checkpoint_digests() != dict(payload["worker_checkpoint_digests"]):
            raise ValueError("widened bundle worker checkpoint digest mismatch")
        return item

    def fit_channels(
        self,
        *,
        experiences: Sequence[Any],
        semantic_epochs: int,
        semantic_lr: float,
        transition_epochs: int,
        transition_lr: float,
        projection_anchor_digests: Sequence[str],
        course_digest: str,
    ) -> tuple[WidenedChannelReceipt, WidenedChannelReceipt]:
        """Run both channels over the sealed course; return per-channel receipts.

        Channel 1 (``forward``) consumes ``experiences`` in the given order;
        channel 2 (``anchored``) consumes the K3-anchored permutation of the
        same stream.  Both channels run the identical per-head fit stream as
        one fixed-large replica, so the bundle performs exactly twice the
        fixed-large per-replica update count.
        """

        experiences = tuple(experiences)
        if not experiences:
            raise ValueError("widened fit requires experiences")
        permutation = anchored_permutation(
            count=len(experiences),
            anchor_digests=projection_anchor_digests,
            course_digest=course_digest,
        )
        receipts: list[WidenedChannelReceipt] = []
        for channel, order in (
            ("forward", tuple(range(len(experiences)))),
            ("anchored", permutation),
        ):
            k1 = self.channels[channel]["k1.semantic"]
            k2 = self.channels[channel]["k2.transition"]
            k1_before = _copy_state(k1)
            k2_before = _copy_state(k2)
            for index in order:
                experience = experiences[index]
                k1.fit(
                    (experience.semantic_example,),
                    epochs=semantic_epochs,
                    learning_rate=semantic_lr,
                )
                k2.fit(
                    (experience.transition_example,),
                    epochs=transition_epochs,
                    learning_rate=transition_lr,
                )
            receipts.append(
                WidenedChannelReceipt(
                    channel=channel,
                    order_digest=content_digest({"order": list(order)}),
                    k1_training_steps=int(k1.training_steps),
                    k2_training_steps=int(k2.training_steps),
                    k1_parameter_delta_digest=_delta_digest(k1_before, k1),
                    k2_parameter_delta_digest=_delta_digest(k2_before, k2),
                    k1_parameter_delta_norm=_delta_norm(k1_before, k1),
                    k2_parameter_delta_norm=_delta_norm(k2_before, k2),
                    projection_anchor_digests=tuple(projection_anchor_digests),
                )
            )
        return receipts[0], receipts[1]

    def channel_divergence(self) -> dict[str, float]:
        """L2 distance between the two channel learners, per worker.

        A zero divergence means the anchored channel degenerated into the
        forward channel; the parity distinct-evidence gate fails on that.
        """

        result: dict[str, float] = {}
        for worker_id in WIDENED_LEARNABLE_WORKERS:
            forward = self.channels["forward"][worker_id]
            anchored = self.channels["anchored"][worker_id]
            total = 0.0
            for name, value in forward.state_dict().items():
                delta = value.detach().cpu() - anchored.state_dict()[name].detach().cpu()
                total += float(torch.sum(delta * delta))
            result[worker_id] = total**0.5
        return result

    def combined_fact_logits(self, *, worker_id: str, percept_input: torch.Tensor) -> torch.Tensor:
        """Logit-domain channel sum for one head (parity readout semantic)."""

        if worker_id not in WIDENED_LEARNABLE_WORKERS:
            raise ValueError(f"unknown widened worker: {worker_id}")
        head = "fact_head" if worker_id == "k1.semantic" else "transition_head"
        forward = self.channels["forward"][worker_id]
        anchored = self.channels["anchored"][worker_id]
        return getattr(forward, head)(percept_input) + getattr(anchored, head)(percept_input)


def widened_divergence_gate(
    divergence: Mapping[str, float], *, floor: float = 1e-08
) -> bool:
    """Distinct-evidence gate: both channels must diverge non-trivially."""

    return all(float(value) > floor for value in divergence.values())


def widened_worker_ids() -> tuple[str, ...]:
    return K_WORKER_IDS
