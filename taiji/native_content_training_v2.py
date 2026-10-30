"""R5b training pieces: the ``new_core_v2`` adapter and the self-prefix schedule.

The R5b contract (SPEC-NATIVE-CONTENT-CORE_R5B_COPYTRACK_CANDIDATE_20261030)
declares exactly two candidate changes.  The copy track lives in
:mod:`taiji.native_content_core_v2`; this module carries the second change:

* updates 1-256 are pure teacher forcing;
* from update 257 on, each episode independently replaces the source of the
  conditioning prefix for its scored turns with the model's own greedy prefix
  with probability 0.5.  The target bytes stay supervised unchanged (full
  exposure); only the input source of the first half changes.  The per-episode
  coin is a pure function of (schedule seed, update index, episode id), so the
  schedule is deterministic and replays identically after a checkpoint restore.

The adapter reuses the v1 native adapter's ingestion, exposure schedule and
gradient grouping; byte accounting is identical in both branches.  The trainer
owns the schedule position: it sets it once per update and supports one
explicitly declared override (the pricing branch probe), recording per-update
branch counters so a probe cannot silently measure the wrong branch.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch.nn import functional as F

from .native_content_core import NativeContentStoreV1
from .native_content_core_v2 import NativeContentBundleV2, NativeContentCoreV2
from .native_content_training import (
    FACT_ROLES,
    TIME_FOR_CATEGORY,
    NativeContentAdapterV1,
    NativeRunTrainerV1,
    episode_session,
    target_bytes,
)

SCHEDULE_OWN_PREFIX_START = 257
SCHEDULE_OWN_PREFIX_PROB = 0.5


def own_prefix_coin(*, seed: int, update: int, episode_id: str) -> bool:
    """The declared per-episode coin: pure function, replayable, seed-bound.

    Deliberately deterministic (not ``secrets``): the R5b contract requires the
    schedule to be a pure function of declared inputs so it replays identically
    after a checkpoint restore.  No security or confidentiality decision
    depends on this value.
    """

    token = f"r5b-self-prefix:{seed}:{update}:{episode_id}"
    return random.Random(token).random() < SCHEDULE_OWN_PREFIX_PROB


class NativeContentV2Adapter(NativeContentAdapterV1):
    """The v1 native adapter with the store-bound v2 bundle and schedule B."""

    name = "new_core_v2"
    schedule_update: int = 0
    schedule_seed: int = 0

    def __init__(self) -> None:
        self.branch_counters: dict[str, int] = {
            "episodes_seen": 0,
            "episodes_scheduled": 0,
            "supervised_bytes": 0,
        }

    def create_model(self) -> NativeContentCoreV2:
        return NativeContentCoreV2(native_core_config_v2())

    def fresh_bundle(self, model: NativeContentCoreV2) -> NativeContentBundleV2:
        return NativeContentBundleV2(model=model, store=NativeContentStoreV1(()), cursor={})

    def from_bundle_payload(self, payload: Mapping[str, Any]) -> NativeContentBundleV2:
        return NativeContentBundleV2.from_checkpoint(payload)

    def reset_branch_counters(self) -> None:
        self.branch_counters = {
            "episodes_seen": 0,
            "episodes_scheduled": 0,
            "supervised_bytes": 0,
        }

    def _supervised_turn(
        self,
        bundle: NativeContentBundleV2,
        episode: Mapping[str, Any],
        turn: Mapping[str, Any],
        *,
        owner: str,
        scope: str,
        session: str,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Logit rows over the full target, the target tensor, and the branch flag."""

        target_tensor = target_bytes(turn)
        target = bytes(turn["target"]["target_bytes_utf8"])
        query = str(turn["text"]).encode("utf-8")
        time_intent = TIME_FOR_CATEGORY[turn["category"]]
        scheduled = (
            self.schedule_update >= SCHEDULE_OWN_PREFIX_START
            and len(target) >= 2
            and own_prefix_coin(
                seed=self.schedule_seed,
                update=self.schedule_update,
                episode_id=str(episode["episode_id"]),
            )
        )
        if scheduled:
            rows, _prefix = bundle.scheduled_target(
                query=query,
                target=target,
                own_prefix_len=len(target) // 2,
                time_intent=time_intent,
                owner=owner,
                scope=scope,
                session=session,
            )
        else:
            rows = bundle.score_target(
                query=query,
                target=target,
                time_intent=time_intent,
                owner=owner,
                scope=scope,
                session=session,
            )
        return rows, target_tensor, scheduled

    def loss_terms(
        self,
        model: NativeContentCoreV2,
        episode: Mapping[str, Any],
        *,
        owner: str,
        scope: str,
    ) -> tuple[list[torch.Tensor], int]:
        session = episode_session(episode)
        bundle = self.fresh_bundle(model)
        terms: list[torch.Tensor] = []
        total = 0
        episode_scheduled = False
        for turn in episode["turns"]:
            if turn["role"] in FACT_ROLES:
                self._ingest(bundle, episode, turn, owner=owner, scope=scope, session=session)
            if not turn["scored"]:
                continue
            rows, slice_target, scheduled = self._supervised_turn(
                bundle, episode, turn, owner=owner, scope=scope, session=session
            )
            terms.append(F.cross_entropy(rows, slice_target, reduction="sum"))
            total += int(slice_target.numel())
            if rows.shape[0] != int(slice_target.numel()):
                raise ValueError("scheduled branch must supervise the unchanged full target")
            episode_scheduled = episode_scheduled or scheduled
        if total <= 0:
            raise ValueError("native v2 episode produced no supervised bytes")
        self.branch_counters["episodes_seen"] += 1
        self.branch_counters["episodes_scheduled"] += int(episode_scheduled)
        self.branch_counters["supervised_bytes"] += total
        return terms, total

    def schedule_is_active(self) -> bool:
        return self.schedule_update >= SCHEDULE_OWN_PREFIX_START


def native_core_config_v2():
    """The declared R3 candidate shape; the copy track adds no config fields."""

    from .native_content_training import native_core_config

    return native_core_config()


class R5BRunTrainer(NativeRunTrainerV1):
    """The v1 run trainer plus the schedule ownership and v2 checkpoint loading.

    ``schedule_position_override`` is the declared probe mechanism (R5b 08.3):
    when set, every update uses exactly that schedule position and records it,
    so a branch probe cannot be silently reset to the training position.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.schedule_position_override: int | None = None

    def update(self, episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        adapter = self.adapter
        if isinstance(adapter, NativeContentV2Adapter):
            if self.schedule_position_override is not None:
                adapter.schedule_update = int(self.schedule_position_override)
            else:
                adapter.schedule_update = self.steps + 1
            adapter.schedule_seed = int(self.cursor.get("seed", 0))
            adapter.reset_branch_counters()
        metrics = super().update(episodes)
        if isinstance(adapter, NativeContentV2Adapter):
            metrics["schedule_position"] = adapter.schedule_update
            metrics["schedule_override_active"] = self.schedule_position_override is not None
            metrics["episodes_seen"] = adapter.branch_counters["episodes_seen"]
            metrics["episodes_scheduled"] = adapter.branch_counters["episodes_scheduled"]
            metrics["scheduled_supervised_bytes"] = metrics["target_bytes"]
        return metrics

    @classmethod
    def from_checkpoint(
        cls,
        payload: Mapping[str, Any],
        *,
        adapter: Any = None,
    ) -> NativeRunTrainerV1:
        if adapter is None and payload.get("object") == NativeContentV2Adapter.name:
            adapter = NativeContentV2Adapter()
        return super().from_checkpoint(payload, adapter=adapter)
