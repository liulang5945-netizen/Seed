"""Bounded train-only runner shared by R4 pricing and R5 training.

The runner works on the train split only (``plans/manifests/native_core_r1_train_v1.json``).
It declares a deterministic exposure schedule, replays each episode's 8-turn
timeline with a fresh store/history, builds teacher-forced losses, and saves a
complete restore point (parameters, AdamW moments, counters, frozen credit
groups, cursor, generator state).

Design boundaries:

* the schedule is a pure function of (seed, batch index), so all three objects
  and both Q2 arms see the same episodes in the same order;
* the native core keeps its persisted content latents, so the content encoder
  receives no task gradient and ``parameter_revision`` stays at the encoder
  revision that produced the latents;
* ``credit_stop`` freezes the declared reader/decoder recipients before AdamW,
  so the same structure and forward path run with a cut task-credit path.

No query/holdout split and no sealed data are read here.
"""

from __future__ import annotations

import copy
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .conditional_language_v3 import TaskCoreConfigV3
from .corrected_old_baseline import (
    CorrectedOldBaselineV1,
    CorrectedOldBundleV1,
    DiscreteFactStoreV1,
)
from .internalization import content_digest
from .native_content_core import (
    NativeContentBundleV1,
    NativeContentCoreConfigV1,
    NativeContentCoreV1,
    NativeContentStoreV1,
    span_sha256,
)
from .ordinary_baseline import OrdinaryBundleV1, OrdinaryEndToEndV1
from .persistence import atomic_save

PRICING_TRAINER_FORMAT = "taiji-native-content-pricing-trainer-v1"  # value kept for R4 checkpoints
TIME_FOR_CATEGORY = {
    "current_query": "current",
    "previous_query": "previous",
    "unknown_query": "unknown",
}
FACT_ROLES = frozenset({"fact_v0", "correction", "distractor"})
PROBE_TEXT = "雨伞的颜色是红色。"
PROBE_QUERY = "雨伞是什么颜色？".encode()


def native_core_config() -> NativeContentCoreConfigV1:
    """The declared R3 candidate shape (144,355 parameters)."""

    return NativeContentCoreConfigV1(
        width=64,
        embedding_width=32,
        key_width=32,
        slots=8,
        chunk_bytes=8,
        max_observation_bytes=512,
        max_input_evidence_bytes=512,
        max_output_position=128,
        max_generation_bytes=64,
        content_width=32,
        seed=401,
        position_extension_seed=402,
        read_seed=403,
    )


def baseline_config() -> TaskCoreConfigV3:
    """The declared V3 backbone shape shared by the two baselines."""

    return TaskCoreConfigV3(
        width=64,
        embedding_width=32,
        key_width=32,
        slots=8,
        chunk_bytes=8,
        max_observation_bytes=512,
        max_input_evidence_bytes=512,
        max_output_position=128,
        max_generation_bytes=64,
        seed=411,
        position_extension_seed=412,
    )


def episode_session(episode: Mapping[str, Any]) -> str:
    return str(episode["episode_id"]).replace(":", "_")


def episode_revision(turn: Mapping[str, Any]) -> int:
    return 0 if turn["role"] in {"fact_v0", "distractor"} else 1


def scored_turns(episode: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    turns = tuple(turn for turn in episode["turns"] if turn["scored"])
    if not turns:
        raise ValueError("pricing episode has no scored turn")
    for turn in turns:
        if turn.get("category") not in TIME_FOR_CATEGORY:
            raise ValueError("scored turn category is outside the declared set")
        target = turn.get("target", {}).get("target_bytes_utf8")
        if not target:
            raise ValueError("scored turn has no declared target bytes")
    return turns


def target_bytes(turn: Mapping[str, Any]) -> torch.Tensor:
    return torch.tensor(list(turn["target"]["target_bytes_utf8"]), dtype=torch.long)


@dataclass(frozen=True)
class PricingBatchV1:
    """One optimizer update: a fixed number of episodes in a declared order."""

    index: int
    episodes: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("pricing batch index cannot be negative")
        if not self.episodes:
            raise ValueError("pricing batch requires at least one episode")


def batch_schedule(
    episodes: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    updates: int,
    batch_size: int = 4,
) -> tuple[PricingBatchV1, ...]:
    """Deterministic schedule: shuffle per epoch, consume, reshuffle when empty."""

    if updates <= 0 or batch_size <= 0:
        raise ValueError("pricing schedule bounds must be positive")
    if not episodes:
        raise ValueError("pricing schedule requires episodes")
    rng = random.Random(seed)
    pending: list[Mapping[str, Any]] = []
    batches: list[PricingBatchV1] = []
    for index in range(updates):
        while len(pending) < batch_size:
            epoch = list(episodes)
            rng.shuffle(epoch)
            pending.extend(epoch)
        chunk = tuple(pending[:batch_size])
        del pending[:batch_size]
        batches.append(PricingBatchV1(index=index, episodes=chunk))
    return tuple(batches)


class EpisodeAdapterV1:
    """Per-object replay: fresh bundle, legal timeline, one loss term per turn."""

    name: str = ""

    def create_model(self) -> torch.nn.Module:
        raise NotImplementedError

    def fresh_bundle(self, model: Any) -> Any:
        raise NotImplementedError

    def from_bundle_payload(self, payload: Mapping[str, Any]) -> Any:
        raise NotImplementedError

    def loss_terms(
        self,
        model: Any,
        episode: Mapping[str, Any],
        *,
        owner: str,
        scope: str,
    ) -> tuple[list[torch.Tensor], int]:
        raise NotImplementedError

    def probe_digest(self, model: Any) -> str:
        raise NotImplementedError

    def gradient_group_of(self, model: Any, name: str) -> str:
        del model
        if name.startswith("core."):
            return "core"
        return name.split(".")[0]

    def parameter_counts(self, model: torch.nn.Module) -> dict[str, int]:
        parameters = list(model.parameters())
        return {
            "total": sum(item.numel() for item in parameters),
            "trainable": sum(item.numel() for item in parameters if item.requires_grad),
        }


class NativeContentAdapterV1(EpisodeAdapterV1):
    name = "new_core_v1"

    def create_model(self) -> NativeContentCoreV1:
        return NativeContentCoreV1(native_core_config())

    def fresh_bundle(self, model: NativeContentCoreV1) -> NativeContentBundleV1:
        return NativeContentBundleV1(model=model, store=NativeContentStoreV1(()), cursor={})

    def from_bundle_payload(self, payload: Mapping[str, Any]) -> NativeContentBundleV1:
        return NativeContentBundleV1.from_checkpoint(payload)

    def _ingest(
        self,
        bundle: NativeContentBundleV1,
        episode: Mapping[str, Any],
        turn: Mapping[str, Any],
        *,
        owner: str,
        scope: str,
        session: str,
    ) -> None:
        raw = str(turn["text"]).encode("utf-8")
        bundle.ingest(
            raw_span=raw,
            source_input_id=f"{episode['episode_id']}:{turn['index']}",
            source_sha256=span_sha256(raw),
            owner=owner,
            scope=scope,
            session=session,
            turn_index=int(turn["index"]),
            revision=episode_revision(turn),
            entity=turn.get("entity"),
            attribute=turn.get("attribute"),
            value=turn.get("value"),
            status="resolved",
        )

    def _turn_logits(
        self,
        bundle: NativeContentBundleV1,
        episode: Mapping[str, Any],
        turn: Mapping[str, Any],
        *,
        owner: str,
        scope: str,
        session: str,
    ) -> torch.Tensor:
        condition = bundle.condition(
            query=str(turn["text"]).encode("utf-8"),
            time_intent=TIME_FOR_CATEGORY[turn["category"]],
            owner=owner,
            scope=scope,
            session=session,
        )
        return bundle.model.teacher_forced_conditioned(
            condition,
            bytes(turn["target"]["target_bytes_utf8"]),
            scope_id=scope,
            session_id=session,
        )

    def loss_terms(
        self,
        model: NativeContentCoreV1,
        episode: Mapping[str, Any],
        *,
        owner: str,
        scope: str,
    ) -> tuple[list[torch.Tensor], int]:
        session = episode_session(episode)
        bundle = self.fresh_bundle(model)
        terms: list[torch.Tensor] = []
        total = 0
        for turn in episode["turns"]:
            if turn["role"] in FACT_ROLES:
                self._ingest(bundle, episode, turn, owner=owner, scope=scope, session=session)
            if not turn["scored"]:
                continue
            target = target_bytes(turn)
            logits = self._turn_logits(
                bundle, episode, turn, owner=owner, scope=scope, session=session
            )
            terms.append(F.cross_entropy(logits, target, reduction="sum"))
            total += int(target.numel())
        if total <= 0:
            raise ValueError("native episode produced no supervised bytes")
        return terms, total

    def probe_digest(self, model: NativeContentCoreV1) -> str:
        bundle = self.fresh_bundle(model)
        raw = PROBE_TEXT.encode("utf-8")
        bundle.ingest(
            raw_span=raw,
            source_input_id="probe:0",
            source_sha256=span_sha256(raw),
            owner="probe",
            scope="probe",
            session="probe",
            turn_index=0,
            revision=0,
            entity="雨伞",
            attribute="颜色",
            value="红色",
            status="resolved",
        )
        return bundle.probe_digest(query=PROBE_QUERY)

    def gradient_group_of(self, model: NativeContentCoreV1, name: str) -> str:
        return model.credit_group_of(name)


class CorrectedOldAdapterV1(EpisodeAdapterV1):
    name = "corrected_old_v1"

    def create_model(self) -> CorrectedOldBaselineV1:
        return CorrectedOldBaselineV1(baseline_config())

    def fresh_bundle(self, model: CorrectedOldBaselineV1) -> CorrectedOldBundleV1:
        return CorrectedOldBundleV1(model=model, store=DiscreteFactStoreV1(()), cursor={})

    def from_bundle_payload(self, payload: Mapping[str, Any]) -> CorrectedOldBundleV1:
        return CorrectedOldBundleV1.from_checkpoint(payload)

    def _ingest(
        self,
        bundle: CorrectedOldBundleV1,
        episode: Mapping[str, Any],
        turn: Mapping[str, Any],
        *,
        owner: str,
        scope: str,
        session: str,
    ) -> None:
        bundle.ingest(
            source_text=str(turn["text"]),
            entity=turn.get("entity"),
            attribute=turn.get("attribute"),
            value=turn.get("value"),
            source_input_id=f"{episode['episode_id']}:{turn['index']}",
            owner=owner,
            scope=scope,
            session=session,
            turn_index=int(turn["index"]),
            revision=episode_revision(turn),
        )

    def loss_terms(
        self,
        model: CorrectedOldBaselineV1,
        episode: Mapping[str, Any],
        *,
        owner: str,
        scope: str,
    ) -> tuple[list[torch.Tensor], int]:
        session = episode_session(episode)
        bundle = self.fresh_bundle(model)
        terms: list[torch.Tensor] = []
        total = 0
        for turn in episode["turns"]:
            if turn["role"] in FACT_ROLES:
                self._ingest(bundle, episode, turn, owner=owner, scope=scope, session=session)
            if not turn["scored"]:
                continue
            target = target_bytes(turn)
            logits, _ = bundle.score_target(
                query=str(turn["text"]).encode("utf-8"),
                target=bytes(turn["target"]["target_bytes_utf8"]),
                time_intent=TIME_FOR_CATEGORY[turn["category"]],
                owner=owner,
                scope=scope,
                session=session,
            )
            terms.append(F.cross_entropy(logits, target, reduction="sum"))
            total += int(target.numel())
        if total <= 0:
            raise ValueError("corrected-old episode produced no supervised bytes")
        return terms, total

    def probe_digest(self, model: CorrectedOldBaselineV1) -> str:
        bundle = self.fresh_bundle(model)
        bundle.ingest(
            source_text=PROBE_TEXT,
            entity="雨伞",
            attribute="颜色",
            value="红色",
            source_input_id="probe:0",
            owner="probe",
            scope="probe",
            session="probe",
            turn_index=0,
            revision=0,
        )
        return bundle.probe_digest(query=PROBE_QUERY)


class OrdinaryAdapterV1(EpisodeAdapterV1):
    name = "ordinary_v1"

    def create_model(self) -> OrdinaryEndToEndV1:
        return OrdinaryEndToEndV1(baseline_config())

    def fresh_bundle(self, model: OrdinaryEndToEndV1) -> OrdinaryBundleV1:
        return OrdinaryBundleV1(model=model, history=(), cursor={})

    def from_bundle_payload(self, payload: Mapping[str, Any]) -> OrdinaryBundleV1:
        return OrdinaryBundleV1.from_checkpoint(payload)

    def loss_terms(
        self,
        model: OrdinaryEndToEndV1,
        episode: Mapping[str, Any],
        *,
        owner: str,
        scope: str,
    ) -> tuple[list[torch.Tensor], int]:
        del owner, scope
        bundle = self.fresh_bundle(model)
        terms: list[torch.Tensor] = []
        total = 0
        for turn in episode["turns"]:
            if not turn["scored"]:
                bundle.observe(str(turn["text"]))
                continue
            target = target_bytes(turn)
            logits = bundle.score_target(
                query=str(turn["text"]).encode("utf-8"),
                target=bytes(turn["target"]["target_bytes_utf8"]),
                time_intent=TIME_FOR_CATEGORY[turn["category"]],
            )
            terms.append(F.cross_entropy(logits, target, reduction="sum"))
            total += int(target.numel())
            bundle.observe(str(turn["text"]))
        if total <= 0:
            raise ValueError("ordinary episode produced no supervised bytes")
        return terms, total

    def probe_digest(self, model: OrdinaryEndToEndV1) -> str:
        bundle = self.fresh_bundle(model)
        bundle.observe(PROBE_TEXT)
        return bundle.probe_digest(query=PROBE_QUERY)


ADAPTERS: dict[str, type[EpisodeAdapterV1]] = {
    NativeContentAdapterV1.name: NativeContentAdapterV1,
    CorrectedOldAdapterV1.name: CorrectedOldAdapterV1,
    OrdinaryAdapterV1.name: OrdinaryAdapterV1,
}


def adapter_for(name: str) -> EpisodeAdapterV1:
    try:
        factory = ADAPTERS[name]
    except KeyError as exc:
        raise ValueError(f"unknown pricing object: {name}") from exc
    return factory()


CREDIT_STOP_GROUPS = ("reader", "decoder")


def credit_on_configure(model: Any) -> tuple[str, ...]:
    """The declared recipient path stays trainable."""

    del model
    return ()


def credit_stop_configure(model: Any) -> tuple[str, ...]:
    """Freeze the declared reader/decoder task-credit recipients before AdamW."""

    for group in CREDIT_STOP_GROUPS:
        model.set_credit_group(group, False)
    return CREDIT_STOP_GROUPS


class NativeRunTrainerV1:
    """AdamW with declared clipping plus a complete, digest-checked checkpoint.

    The same trainer serves R4 pricing (50 updates) and R5 training runs (512
    updates); callers declare the schedule and the credit mode.
    """

    def __init__(
        self,
        adapter: EpisodeAdapterV1,
        model: Any,
        *,
        learning_rate: float = 0.001,
    ):
        self.adapter = adapter
        self.model = model
        self.learning_rate = learning_rate
        self.trainable = [item for item in model.parameters() if item.requires_grad]
        self.optimizer = torch.optim.AdamW(self.trainable, lr=learning_rate, weight_decay=0)
        self.steps = 0
        self.target_byte_exposures = 0
        self.cursor: dict[str, Any] = {"batch_index": 0, "seed": 0}
        self.frozen_groups: tuple[str, ...] = ()
        self.owner = "train"
        self.scope = "train"

    @classmethod
    def create(
        cls,
        adapter: EpisodeAdapterV1,
        *,
        learning_rate: float = 0.001,
        configure: Callable[[Any], Sequence[str]] | None = None,
    ) -> NativeRunTrainerV1:
        model = adapter.create_model()
        frozen_groups = tuple(configure(model)) if configure is not None else ()
        trainer = cls(adapter, model, learning_rate=learning_rate)
        trainer.frozen_groups = frozen_groups
        trainer.cursor["frozen_groups"] = list(frozen_groups)
        return trainer

    # --- training ---------------------------------------------------------

    def update(self, episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not episodes:
            raise ValueError("bounded update requires at least one episode")
        self.optimizer.zero_grad(set_to_none=True)
        terms: list[torch.Tensor] = []
        total = 0
        for episode in episodes:
            episode_terms, episode_bytes = self.adapter.loss_terms(
                self.model, episode, owner=self.owner, scope=self.scope
            )
            terms.extend(episode_terms)
            total += episode_bytes
        if total <= 0:
            raise ValueError("bounded update has no supervised bytes")
        loss_sum = 0.0
        for term in terms:
            value = float(term.detach())
            if not torch.isfinite(term.detach()).all():
                raise ValueError("bounded loss is not finite")
            loss_sum += value
            (term / total).backward()
            del term
        groups: dict[str, float] = {}
        for name, parameter in self.model.named_parameters():
            if parameter.grad is None:
                continue
            group = self.adapter.gradient_group_of(self.model, name)
            groups[group] = groups.get(group, 0.0) + float(parameter.grad.detach().square().sum())
        norm = float(torch.nn.utils.clip_grad_norm_(self.trainable, 1.0, error_if_nonfinite=True))
        self.optimizer.step()
        self.steps += 1
        self.target_byte_exposures += total
        missing = sorted(
            name
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        )
        return {
            "step": self.steps,
            "loss_nats": loss_sum / total,
            "loss_bits": loss_sum / total / math.log(2),
            "target_bytes": total,
            "gradient_norm_before_clip": norm,
            "gradient_group_norms": {name: value**0.5 for name, value in sorted(groups.items())},
            "trainable_without_gradient": missing,
        }

    # --- checkpoint -------------------------------------------------------

    def checkpoint(self) -> dict[str, Any]:
        bundle = self.adapter.fresh_bundle(self.model)
        payload: dict[str, Any] = {
            "format": PRICING_TRAINER_FORMAT,
            "version": 2,
            "object": self.adapter.name,
            "bundle": bundle.checkpoint(),
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "learning_rate": self.learning_rate,
            "steps": self.steps,
            "target_byte_exposures": self.target_byte_exposures,
            "frozen_groups": list(self.frozen_groups),
            "cursor": copy.deepcopy(self.cursor),
            "rng_state": self.model.generator.get_state().clone(),
        }
        payload["checkpoint_digest"] = content_digest(payload)
        return payload

    def save(self, path: Path) -> Path:
        target = Path(path)
        if target.exists():
            raise ValueError("run checkpoint is write-once")
        atomic_save(self.checkpoint(), target)
        return target

    @classmethod
    def from_checkpoint(
        cls,
        payload: Mapping[str, Any],
        *,
        adapter: EpisodeAdapterV1 | None = None,
    ) -> NativeRunTrainerV1:
        expected_v1 = {
            "format",
            "version",
            "object",
            "bundle",
            "optimizer",
            "learning_rate",
            "steps",
            "target_byte_exposures",
            "cursor",
            "rng_state",
            "checkpoint_digest",
        }
        if not isinstance(payload, Mapping):
            raise ValueError("run checkpoint must be a mapping")
        version = payload.get("version")
        if version == 1:
            expected = expected_v1
            frozen: tuple[str, ...] = ()
        elif version == 2:
            expected = expected_v1 | {"frozen_groups"}
            frozen = tuple(str(item) for item in payload["frozen_groups"])
        else:
            raise ValueError("unsupported run checkpoint version")
        if set(payload) != expected:
            raise ValueError("run checkpoint schema differs")
        if payload["format"] != PRICING_TRAINER_FORMAT:
            raise ValueError("unsupported run checkpoint format")
        if (
            content_digest(
                {name: value for name, value in payload.items() if name != "checkpoint_digest"}
            )
            != payload["checkpoint_digest"]
        ):
            raise ValueError("run checkpoint digest differs")
        resolved = adapter_for(str(payload["object"])) if adapter is None else adapter
        if resolved.name != payload["object"]:
            raise ValueError("run checkpoint object differs from the adapter")
        bundle = resolved.from_bundle_payload(payload["bundle"])
        model = bundle.model
        for group in frozen:
            if not hasattr(model, "set_credit_group"):
                raise ValueError("frozen credit groups require the native core")
            model.set_credit_group(group, False)
        trainer = cls(
            resolved,
            model,
            learning_rate=float(payload["learning_rate"]),
        )
        trainer.frozen_groups = frozen
        trainer.optimizer.load_state_dict(payload["optimizer"])
        trainer.steps = int(payload["steps"])
        trainer.target_byte_exposures = int(payload["target_byte_exposures"])
        trainer.cursor = copy.deepcopy(dict(payload["cursor"]))
        trainer.model.generator.set_state(payload["rng_state"])
        return trainer

    @classmethod
    def load(cls, path: Path) -> NativeRunTrainerV1:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        return cls.from_checkpoint(payload)

    # --- probes -----------------------------------------------------------

    def probe_digest(self) -> str:
        return self.adapter.probe_digest(self.model)

    def parameter_counts(self) -> dict[str, int]:
        return self.adapter.parameter_counts(self.model)

    def parameters_finite(self) -> bool:
        return all(bool(torch.isfinite(parameter).all()) for parameter in self.model.parameters())
