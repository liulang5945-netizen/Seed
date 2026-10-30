"""R5b training pieces: the ``new_core_v2`` adapter and the self-prefix schedule.

The R5b contract (SPEC-NATIVE-CONTENT-CORE_R5B_COPYTRACK_CANDIDATE_20261030)
declares exactly two candidate changes.  The copy track lives in
:mod:`taiji.native_content_core_v2`; this module carries the second change:

* updates 1-256 are pure teacher forcing;
* from update 257 on, each scored turn independently replaces the first half
  of its target with the model's own greedy prefix with probability 0.5 (the
  target bytes are unchanged; only the prefix source changes).  The per-turn
  coin is a pure function of (schedule seed, update index, episode id, turn
  index), so the schedule is deterministic and replays identically after a
  checkpoint restore.

The adapter reuses the v1 native adapter's ingestion, exposure schedule and
gradient grouping; only the supervised-turn path and the byte accounting
differ (a scheduled turn supervises ``len(target) - own_prefix_len`` bytes).
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


def own_prefix_coin(*, seed: int, update: int, episode_id: str, turn_index: int) -> bool:
    """The declared per-turn coin: pure function, replayable, seed-bound.

    Deliberately deterministic (not ``secrets``): the R5b contract requires the
    schedule to be a pure function of declared inputs so it replays identically
    after a checkpoint restore.  No security or confidentiality decision depends
    on this value.
    """

    token = f"r5b-self-prefix:{seed}:{update}:{episode_id}:{turn_index}"
    return random.Random(token).random() < SCHEDULE_OWN_PREFIX_PROB


class NativeContentV2Adapter(NativeContentAdapterV1):
    """The v1 native adapter with the store-bound v2 bundle and schedule B."""

    name = "new_core_v2"
    schedule_update: int = 0
    schedule_seed: int = 0

    def create_model(self) -> NativeContentCoreV2:
        return NativeContentCoreV2(native_core_config_v2())

    def fresh_bundle(self, model: NativeContentCoreV2) -> NativeContentBundleV2:
        return NativeContentBundleV2(model=model, store=NativeContentStoreV1(()), cursor={})

    def from_bundle_payload(self, payload: Mapping[str, Any]) -> NativeContentBundleV2:
        return NativeContentBundleV2.from_checkpoint(payload)

    def _supervised_turn(
        self,
        bundle: NativeContentBundleV2,
        episode: Mapping[str, Any],
        turn: Mapping[str, Any],
        *,
        owner: str,
        scope: str,
        session: str,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Logit rows, the matching target slice, and the supervised byte count."""

        target_tensor = target_bytes(turn)
        target = bytes(turn["target"]["target_bytes_utf8"])
        query = str(turn["text"]).encode("utf-8")
        time_intent = TIME_FOR_CATEGORY[turn["category"]]
        scheduled = (
            self.schedule_update >= SCHEDULE_OWN_PREFIX_START
            and own_prefix_coin(
                seed=self.schedule_seed,
                update=self.schedule_update,
                episode_id=str(episode["episode_id"]),
                turn_index=int(turn["index"]),
            )
            and len(target) >= 2
        )
        if scheduled:
            rows, prefix = bundle.scheduled_target(
                query=query,
                target=target,
                own_prefix_len=len(target) // 2,
                time_intent=time_intent,
                owner=owner,
                scope=scope,
                session=session,
            )
            supervised = len(target) - prefix
        else:
            rows = bundle.score_target(
                query=query,
                target=target,
                time_intent=time_intent,
                owner=owner,
                scope=scope,
                session=session,
            )
            supervised = len(target)
        return rows, target_tensor[-rows.shape[0] :], supervised

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
        for turn in episode["turns"]:
            if turn["role"] in FACT_ROLES:
                self._ingest(bundle, episode, turn, owner=owner, scope=scope, session=session)
            if not turn["scored"]:
                continue
            rows, slice_target, supervised = self._supervised_turn(
                bundle, episode, turn, owner=owner, scope=scope, session=session
            )
            terms.append(F.cross_entropy(rows, slice_target, reduction="sum"))
            total += supervised
        if total <= 0:
            raise ValueError("native v2 episode produced no supervised bytes")
        return terms, total

    def schedule_is_active(self) -> bool:
        return self.schedule_update >= SCHEDULE_OWN_PREFIX_START


def native_core_config_v2():
    """The declared R3 candidate shape; the copy track adds no config fields."""

    from .native_content_training import native_core_config

    return native_core_config()


class R5BRunTrainer(NativeRunTrainerV1):
    """The v1 run trainer plus the schedule handoff and v2 checkpoint loading."""

    def update(self, episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        adapter = self.adapter
        if isinstance(adapter, NativeContentV2Adapter):
            adapter.schedule_update = self.steps + 1
            adapter.schedule_seed = int(self.cursor.get("seed", 0))
        return super().update(episodes)

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
