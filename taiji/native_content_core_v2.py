"""Native content core v2: the R5b copy-track candidate (``new_core_v2``).

R5 closed as ``FAIL_BEHAVIOR`` with a precise attribution: the answer path did
not reliably consume the selected content (receipt 0.429, cut_drop 0.166) and
free generation drifted away from teacher forcing (train loss 0.08 vs
interpolated exact 0.329).  This module implements the declared R5b candidate
changes and nothing else:

* a copy track: at every decoding step the model's own reader scores the store
  records, the selected record's value span bytes become an atomic copy
  fragment, and the final next-byte distribution mixes the vocabulary softmax
  with a pointer distribution over those span bytes.  The channel sees store
  content and model state only - the target bytes are structurally absent from
  every input of the copy path, and the cache must be bound to the declared
  store, so a host-forged condition is rejected;
* a self-prefix schedule lives on the training side (see
  :mod:`taiji.native_content_training_v2`), not here.

Everything else is byte-identical to ``new_core_v1``: store shape, read
scoring, conditions, credit-group semantics (copy parameters are assigned to
the ``decoder`` group), bundle checkpointing, and the fixed-denominator scoring
protocol.  With an empty store the module must reduce exactly to v1 logits.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .internalization import content_digest
from .native_content_core import (
    END_MARKER,
    NativeContentBundleV1,
    NativeContentConditionV1,
    NativeContentCoreConfigV1,
    NativeContentCoreV1,
    NativeContentStoreV1,
    _NativeContentReadCache,
)
from .utf8_state import utf8_allowed

NATIVE_CORE_BUNDLE_FORMAT_V2 = "taiji-native-content-bundle-v2"


@dataclass(frozen=True)
class NativeContentReadCacheV2(_NativeContentReadCache):
    """The v1 read cache plus the raw span bytes of every candidate record."""

    content_spans: tuple[bytes, ...]


class NativeContentCoreV2(NativeContentCoreV1):
    """The v1 core plus a store-bound copy track over the selected value span."""

    def __init__(
        self,
        config: NativeContentCoreConfigV1 | None = None,
        *,
        bundle_id: str = "native-content-core-v2",
    ):
        config = NativeContentCoreConfigV1() if config is None else config
        super().__init__(config, bundle_id=bundle_id)
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(config.read_seed + 1)
            self.copy_query = nn.Linear(
                config.width, config.embedding_width, bias=False, dtype=torch.float32
            )
            self.copy_key = nn.Linear(
                config.embedding_width, config.embedding_width, bias=False, dtype=torch.float32
            )
            self.copy_gate = nn.Linear(config.width + config.content_width, 1, dtype=torch.float32)

    # --- store-bound cache -------------------------------------------------

    def content_read_cache(
        self,
        state: Any,
        condition: NativeContentConditionV1,
        *,
        store: NativeContentStoreV1,
    ) -> NativeContentReadCacheV2:
        """Build the v1 read cache and bind it to the declared store.

        The binding is the gold-rejection boundary: every condition record must
        be present in the declared store with an identical hash and span, so a
        host cannot smuggle a fabricated record (for example one containing the
        expected answer) into the copy channel.
        """

        self.validate(state)
        known = {item.content_id: item for item in store.records}
        for record in condition.records:
            local = known.get(record.content_id)
            if (
                local is None
                or local.record_sha256 != record.record_sha256
                or local.raw_span != record.raw_span
            ):
                raise ValueError("native content v2 condition record is not in the declared store")
        base = super().content_read_cache(state, condition)
        return NativeContentReadCacheV2(
            base.bundle_id,
            base.scope_id,
            base.session_id,
            base.parameter_revision,
            base.byte_memory,
            base.hidden,
            base.keys,
            base.values,
            base.query_vector,
            base.time_vector,
            base.content_latents,
            base.content_status,
            base.content_mask,
            base.content_ids,
            tuple(record.raw_span for record in condition.records),
        )

    # --- per-step decoding with the copy track -----------------------------

    def next_logits(  # type: ignore[override]
        self, state: Any, *, cache: Any = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(cache, NativeContentReadCacheV2):
            raise ValueError("native content v2 decoding requires a store-bound v2 read cache")
        raw_logits, updated = super().next_logits(state, cache=cache)
        if len(cache.content_ids) == 0:
            return raw_logits, updated
        hidden = state.hidden if state.decoder_hidden is None else state.decoder_hidden
        read, scores = self._read(cache, hidden)
        selected = cache.content_spans[int(torch.argmax(scores))]
        if not selected:
            return raw_logits, updated
        tokens = torch.tensor(list(selected), dtype=torch.long, device=raw_logits.device)
        keys = self.copy_key(self.core.embedding(tokens))
        query = self.copy_query(hidden)
        positions = keys @ query / (self.config.embedding_width**0.5)
        weights = torch.softmax(positions, dim=0).to(raw_logits.dtype)
        copy_distribution = torch.zeros_like(raw_logits)
        copy_distribution.index_add_(0, tokens, weights)
        gate = torch.sigmoid(self.copy_gate(torch.cat((updated, read))))
        vocabulary = torch.softmax(raw_logits, dim=-1)
        mixture = (1.0 - gate) * vocabulary + gate * copy_distribution
        return torch.log(mixture.clamp_min(1e-12)), updated

    # --- supervised paths (store-bound) ------------------------------------

    def teacher_forced_conditioned(
        self,
        condition: NativeContentConditionV1,
        target: bytes,
        *,
        scope_id: str,
        session_id: str,
        store: NativeContentStoreV1,
        native_state_ref: str = "native-content:probe",
    ) -> torch.Tensor:
        """One logit row per target byte; the first target byte is supervised."""

        if not target or len(target) > self.config.max_generation_bytes:
            raise ValueError("native content target is empty or exceeds the generation bound")
        state = self.conditioned_state(
            condition, scope_id=scope_id, session_id=session_id, native_state_ref=native_state_ref
        )
        cache = self.content_read_cache(state, condition, store=store)
        logits: list[torch.Tensor] = []
        for symbol in target:
            current, hidden = self.next_logits(state, cache=cache)
            logits.append(current)
            state = self.observe_response(state, symbol, hidden)
        return torch.stack(logits)

    def scheduled_forced_conditioned(
        self,
        condition: NativeContentConditionV1,
        target: bytes,
        *,
        own_prefix_len: int,
        scope_id: str,
        session_id: str,
        store: NativeContentStoreV1,
        native_state_ref: str = "native-content:train",
    ) -> tuple[torch.Tensor, int]:
        """Own greedy prefix, then teacher forcing on the declared suffix.

        The prefix is generated greedily from the model itself (no gradient,
        no target information); the remaining target bytes are supervised on
        top of that prefix.  Returns the logit rows for the supervised suffix
        and the prefix length actually consumed.
        """

        if not target or len(target) > self.config.max_generation_bytes:
            raise ValueError("native content target is empty or exceeds the generation bound")
        if not 0 <= own_prefix_len < len(target):
            raise ValueError("self-prefix length must stay inside the declared target")
        state = self.conditioned_state(
            condition, scope_id=scope_id, session_id=session_id, native_state_ref=native_state_ref
        )
        cache = self.content_read_cache(state, condition, store=store)
        with torch.no_grad():
            for _ in range(own_prefix_len):
                logits, hidden = self.next_logits(state, cache=cache)
                allowed = utf8_allowed(state.utf8_remaining, state.utf8_lead)
                choices = torch.tensor(allowed, dtype=torch.long, device=logits.device)
                symbol = int(choices[logits[choices].argmax()])
                state = self.observe_response(state, symbol, hidden)
        rows: list[torch.Tensor] = []
        for symbol in target[own_prefix_len:]:
            current, hidden = self.next_logits(state, cache=cache)
            rows.append(current)
            state = self.observe_response(state, symbol, hidden)
        return torch.stack(rows), own_prefix_len

    @torch.no_grad()
    def answer_conditioned(
        self,
        condition: NativeContentConditionV1,
        *,
        scope_id: str,
        session_id: str,
        store: NativeContentStoreV1,
        native_state_ref: str = "native-content:probe",
        max_new_bytes: int | None = None,
        stop_on_marker: bool = True,
    ) -> Any:
        allowance = self.config.max_generation_bytes if max_new_bytes is None else max_new_bytes
        if allowance <= 0 or allowance > self.config.max_generation_bytes:
            raise ValueError("native content generation allowance is outside the declared bound")
        state = self.conditioned_state(
            condition, scope_id=scope_id, session_id=session_id, native_state_ref=native_state_ref
        )
        cache = self.content_read_cache(state, condition, store=store)
        generated: list[int] = []
        selected: list[str] = []
        rows: list[tuple[float, ...]] = []
        for _ in range(allowance):
            logits, hidden = self.next_logits(state, cache=cache)
            read, scores = self._read(cache, hidden)
            del read
            rows.append(tuple(float(item) for item in scores.tolist()))
            if cache.content_ids:
                selected.append(cache.content_ids[int(torch.argmax(scores))])
            else:
                selected.append("")
            allowed = utf8_allowed(state.utf8_remaining, state.utf8_lead)
            choices = torch.tensor(allowed, dtype=torch.long, device=logits.device)
            symbol = int(choices[logits[choices].argmax()])
            state = self.observe_response(state, symbol, hidden)
            generated.append(symbol)
            if stop_on_marker and bytes(generated).endswith(END_MARKER):
                break
        from .native_content_core import NativeContentAnswerV1

        return NativeContentAnswerV1(
            generated=bytes(generated),
            selected_ids=tuple(selected),
            step_scores=tuple(rows),
        )

    # --- credit groups ------------------------------------------------------

    def credit_group_of(self, name: str) -> str:  # type: ignore[override]
        if name.startswith("copy_"):
            return "decoder"
        return super().credit_group_of(name)


class NativeContentBundleV2(NativeContentBundleV1):
    """A complete recoverable v2 entry: copy-track core plus versioned store."""

    FORMAT = NATIVE_CORE_BUNDLE_FORMAT_V2

    @classmethod
    def create(
        cls,
        config: NativeContentCoreConfigV1 | None = None,
        *,
        bundle_id: str = "native-content-core-v2",
        cursor: Mapping[str, Any] | None = None,
    ) -> NativeContentBundleV2:
        model = NativeContentCoreV2(config, bundle_id=bundle_id)
        return cls(
            model=model,
            store=NativeContentStoreV1(()),
            cursor=copy.deepcopy(dict(cursor or {})),
        )

    # --- answering (all paths bind the declared store) ----------------------

    def answer(
        self,
        *,
        query: bytes,
        time_intent: str = "current",
        owner: str,
        scope: str,
        session: str,
        max_new_bytes: int | None = None,
    ) -> Any:
        condition = self.condition(
            query=query, time_intent=time_intent, owner=owner, scope=scope, session=session
        )
        return self.model.answer_conditioned(
            condition,
            scope_id=scope,
            session_id=session,
            store=self.store,
            max_new_bytes=max_new_bytes,
        )

    def score_target(
        self,
        *,
        query: bytes,
        target: bytes,
        time_intent: str = "current",
        owner: str,
        scope: str,
        session: str,
        reencode: bool = False,
    ) -> torch.Tensor:
        condition = self.condition(
            query=query,
            time_intent=time_intent,
            owner=owner,
            scope=scope,
            session=session,
            reencode=reencode,
        )
        return self.model.teacher_forced_conditioned(
            condition, target, scope_id=scope, session_id=session, store=self.store
        )

    def scheduled_target(
        self,
        *,
        query: bytes,
        target: bytes,
        own_prefix_len: int,
        time_intent: str = "current",
        owner: str,
        scope: str,
        session: str,
    ) -> tuple[torch.Tensor, int]:
        condition = self.condition(
            query=query, time_intent=time_intent, owner=owner, scope=scope, session=session
        )
        return self.model.scheduled_forced_conditioned(
            condition,
            target,
            own_prefix_len=own_prefix_len,
            scope_id=scope,
            session_id=session,
            store=self.store,
        )

    # --- re-encoding and checkpoint ----------------------------------------

    def probe_digest(self, *, query: bytes, time_intent: str = "current") -> str:
        condition = self.condition(
            query=query,
            time_intent=time_intent,
            owner="probe",
            scope="probe",
            session="probe",
        )
        with torch.no_grad():
            logits = self.model.teacher_forced_conditioned(
                condition, END_MARKER, scope_id="probe", session_id="probe", store=self.store
            )
        return content_digest(
            {
                "selected": [item.content_id for item in condition.records],
                "logits": [[round(float(value), 7) for value in row] for row in logits.tolist()],
                "answer": self.answer(
                    query=query,
                    time_intent=time_intent,
                    owner="probe",
                    scope="probe",
                    session="probe",
                ).answer_digest(),
            }
        )

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> NativeContentBundleV2:
        expected = {
            "format",
            "version",
            "bundle_id",
            "config",
            "parameter_revision",
            "parameters",
            "store",
            "cursor",
            "task_state",
            "checkpoint_digest",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("native content bundle payload schema differs")
        if (payload["format"], payload["version"]) != (cls.FORMAT, 1):
            raise ValueError("unsupported native content bundle version")
        if (
            content_digest(
                {name: value for name, value in payload.items() if name != "checkpoint_digest"}
            )
            != payload["checkpoint_digest"]
        ):
            raise ValueError("native content bundle checkpoint digest differs")
        model = NativeContentCoreV2(
            NativeContentCoreConfigV1(**payload["config"]), bundle_id=payload["bundle_id"]
        )
        model.load_state_dict(payload["parameters"], strict=True)
        model.parameter_revision = int(payload["parameter_revision"])
        store = NativeContentStoreV1.from_payload(payload["store"])
        stamp = content_digest(
            {
                "bundle_id": model.bundle_id,
                "parameter_revision": model.parameter_revision,
                "records": [item.record_sha256 for item in store.records],
            }
        )
        cursor = copy.deepcopy(dict(payload["cursor"]))
        cursor.setdefault("bundle_stamp", stamp)
        return cls(model=model, store=store, cursor=cursor)
