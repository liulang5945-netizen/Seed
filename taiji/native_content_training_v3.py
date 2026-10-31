"""R5c training pieces: the ``new_core_v3`` adapter with the pointer auxiliary CE.

The approved R5c contract freezes mechanism C as a training-side pointer
auxiliary CE (weight 0.3, declared target rule: the value-substring positions
of the selected record's span when the value occurs exactly once; turns
without an occurrence are excluded).  The schedule B regime (per-episode coin
from update 257, full-target supervision with the prefix source swapped) is
inherited unchanged from the v2 adapter; only the loss terms gain the
declared auxiliary component, accounted separately.

Forward paths, parameters and checkpoint tensors are identical to
``new_core_v2``; the candidate identity lives in the bundle format v3.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch.nn import functional as F

from .native_content_core import NativeContentStoreV1
from .native_content_core_v2 import NativeContentCoreV2
from .native_content_core_v3 import (
    NativeContentBundleV3,
    NativeContentCoreV3,
)
from .native_content_training import FACT_ROLES, native_core_config
from .native_content_training_v2 import (
    NativeContentV2Adapter,
    R5BRunTrainer,
    episode_session,
)

POINTER_CE_WEIGHT = 0.3


class NativeContentV3Adapter(NativeContentV2Adapter):
    """The v2 adapter plus the declared pointer auxiliary CE (mechanism C)."""

    name = "new_core_v3"

    def __init__(self) -> None:
        super().__init__()
        self.branch_counters["aux_turns"] = 0
        self.branch_counters["aux_loss_sum"] = 0.0

    def reset_branch_counters(self) -> None:
        super().reset_branch_counters()
        self.branch_counters["aux_turns"] = 0
        self.branch_counters["aux_loss_sum"] = 0.0

    def create_model(self) -> NativeContentCoreV3:
        return NativeContentCoreV3(native_core_config())

    def fresh_bundle(self, model: NativeContentCoreV3) -> NativeContentBundleV3:
        return NativeContentBundleV3(model=model, store=NativeContentStoreV1(()), cursor={})

    def from_bundle_payload(self, payload: Mapping[str, Any]) -> NativeContentBundleV3:
        return NativeContentBundleV3.from_checkpoint(payload)

    def _expected_value(self, episode: Mapping[str, Any], turn: Mapping[str, Any]) -> bytes | None:
        """Course-defined ground truth: latest (current) / previous value."""

        category = str(turn["category"])
        if category == "unknown_query":
            return None
        history = [
            candidate
            for candidate in episode["turns"]
            if int(candidate["index"]) < int(turn["index"])
            and candidate["role"] in FACT_ROLES
            and candidate.get("entity") == turn.get("entity")
            and candidate.get("attribute") == turn.get("attribute")
        ]
        history.sort(key=lambda item: int(item["index"]))
        if not history:
            return None
        if category == "current_query":
            return str(history[-1].get("value")).encode("utf-8")
        if category == "previous_query" and len(history) >= 2:
            return str(history[-2].get("value")).encode("utf-8")
        return None

    def _auxiliary_term(
        self,
        bundle: Any,
        episode: Mapping[str, Any],
        turn: Mapping[str, Any],
        *,
        owner: str,
        scope: str,
        session: str,
    ) -> tuple[torch.Tensor | None, float]:
        """The declared auxiliary pointer CE at the conditioned state."""

        value = self._expected_value(episode, turn)
        if value is None:
            return None, 0.0
        condition = bundle.condition(
            query=str(turn["text"]).encode("utf-8"),
            time_intent={
                "current_query": "current",
                "previous_query": "previous",
                "unknown_query": "unknown",
            }[str(turn["category"])],
            owner=owner,
            scope=scope,
            session=session,
        )
        state = bundle.model.conditioned_state(
            condition, native_state_ref="train:aux", scope_id=scope, session_id=session
        )
        cache = bundle.model.content_read_cache(state, condition, store=bundle.store)
        auxiliary = bundle.model.pointer_auxiliary_ce(state, cache, value)
        if auxiliary is None:
            return None, 0.0
        return POINTER_CE_WEIGHT * auxiliary, float(auxiliary.detach())

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
            if rows.shape[0] != int(slice_target.numel()):
                raise ValueError("scheduled branch must supervise the unchanged full target")
            terms.append(F.cross_entropy(rows, slice_target, reduction="sum"))
            total += int(slice_target.numel())
            episode_scheduled = episode_scheduled or scheduled
            auxiliary, aux_value = self._auxiliary_term(
                bundle, episode, turn, owner=owner, scope=scope, session=session
            )
            if auxiliary is not None:
                terms.append(auxiliary)
                self.branch_counters["aux_turns"] += 1
                self.branch_counters["aux_loss_sum"] += aux_value
        if total <= 0:
            raise ValueError("native v3 episode produced no supervised bytes")
        self.branch_counters["episodes_seen"] += 1
        self.branch_counters["episodes_scheduled"] += int(episode_scheduled)
        self.branch_counters["supervised_bytes"] += total
        return terms, total


def native_core_config_v3():
    """Identical shape to the v2 candidate; no new config fields."""

    return native_core_config()


class R5CRunTrainer(R5BRunTrainer):
    """The v2 run trainer plus v3 checkpoint loading and auxiliary accounting."""

    def update(self, episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        metrics = super().update(episodes)
        if isinstance(self.adapter, NativeContentV3Adapter):
            metrics["aux_pointer_loss"] = float(self.adapter.branch_counters["aux_loss_sum"])
            metrics["aux_turns"] = int(self.adapter.branch_counters["aux_turns"])
        return metrics

    @classmethod
    def from_checkpoint(
        cls,
        payload: Mapping[str, Any],
        *,
        adapter: Any = None,
    ) -> Any:
        if adapter is None and payload.get("object") == NativeContentV3Adapter.name:
            adapter = NativeContentV3Adapter()
        return super().from_checkpoint(payload, adapter=adapter)
