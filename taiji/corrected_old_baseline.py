"""Corrected old baseline v1: hard parse, discrete store, time choice, fixed value bias.

This is the R3b baseline required by the R2 candidate contract.  It keeps the
old graph's defining limitations -- a hard label parser gates content, the
store is discrete, the language head is frozen and one fixed value bias is
added at every generation step -- while fixing the semantic errors that made
the old F results uninterpretable:

* ``logits_t`` predicts ``target[t]`` and the first target byte is supervised;
* every scored turn stays in the denominator; parse failures count as wrong;
* ``current`` and ``previous`` are selected from the version chain instead of
  always returning ``store.current``;
* the cut/rescue probe restores the exact baseline output, and correctness is
  scored separately from output movement.

The language backbone is the same V3 byte decoder used by the R3 core, so the
comparison changes content handling rather than the observation encoder.
"""

from __future__ import annotations

import copy
import hashlib
import string
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .conditional_language_v3 import ConditionalLanguageCandidateV3, TaskCoreConfigV3
from .internalization import content_digest
from .persistence import atomic_save
from .utf8_state import utf8_allowed

DISCRETE_FACT_FORMAT = "taiji-corrected-old-fact-v1"
DISCRETE_STORE_FORMAT = "taiji-corrected-old-store-v1"
BASELINE_BUNDLE_FORMAT = "taiji-corrected-old-bundle-v1"

TIME_INTENTS = ("current", "previous", "unknown")
TIME_INTENT_INDEX = {name: index for index, name in enumerate(TIME_INTENTS)}
FACT_STATUSES = ("resolved", "ambiguous", "unresolved")
END_MARKER = b"<|end|>"

# The public R1 world vocabulary.  These are task-domain labels used by the
# hard parser and scoring; the model itself never receives them.
ENTITIES = ("雨伞", "背包", "水杯", "耳机", "台灯")
ATTRIBUTE_TERMS: dict[str, tuple[str, ...]] = {
    "颜色": ("颜色",),
    "位置": ("位置", "在哪"),
    "归属": ("归属", "归谁", "是谁的"),
}
VALUES = (
    "红色",
    "蓝色",
    "绿色",
    "黄色",
    "桌上",
    "抽屉",
    "柜子",
    "包里",
    "我的",
    "你的",
    "他的",
    "她的",
)
VALUE_INDEX = {value: index for index, value in enumerate(VALUES)}
NULL_VALUE_INDEX = len(VALUES)

_HEX = frozenset(string.hexdigits.lower())


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in _HEX for character in value.lower())


def _fact_payload(
    *,
    fact_id: str,
    entity: str,
    attribute: str,
    value: str,
    source_input_id: str,
    source_sha256: str,
    source_text: str,
    owner: str,
    scope: str,
    session: str,
    turn_index: int,
    revision: int,
    valid_from: int,
    valid_to: int | None,
    superseded_by: str | None,
    status: str,
) -> dict[str, Any]:
    return {
        "format": DISCRETE_FACT_FORMAT,
        "version": 1,
        "fact_id": fact_id,
        "entity": entity,
        "attribute": attribute,
        "value": value,
        "source_input_id": source_input_id,
        "source_sha256": source_sha256,
        "source_text": source_text,
        "owner": owner,
        "scope": scope,
        "session": session,
        "turn_index": turn_index,
        "revision": revision,
        "valid_from": valid_from,
        "valid_to": valid_to,
        "superseded_by": superseded_by,
        "status": status,
    }


@dataclass(frozen=True)
class DiscreteFactV1:
    """One hard-parsed fact; no latent, no unresolved content in the store."""

    fact_id: str
    entity: str
    attribute: str
    value: str
    source_input_id: str
    source_sha256: str
    source_text: str
    owner: str
    scope: str
    session: str
    turn_index: int
    revision: int
    valid_from: int
    valid_to: int | None
    superseded_by: str | None
    status: str
    fact_sha256: str

    FORMAT = DISCRETE_FACT_FORMAT

    def __post_init__(self) -> None:
        if not self.fact_id.startswith("old1:"):
            raise ValueError("corrected-old fact id is malformed")
        if self.status not in FACT_STATUSES:
            raise ValueError("corrected-old fact status is outside the declared set")
        if self.status != "resolved":
            raise ValueError("the hard-gated store only admits resolved facts")
        if not all(
            isinstance(item, str) and item for item in (self.entity, self.attribute, self.value)
        ):
            raise ValueError("a resolved fact needs entity/attribute/value")
        if not self.source_input_id or not _is_sha256(self.source_sha256):
            raise ValueError("corrected-old fact needs a source identity")
        if not all(
            isinstance(item, str) and item for item in (self.owner, self.scope, self.session)
        ):
            raise ValueError("corrected-old fact needs owner/scope/session")
        if any(int(item) < 0 for item in (self.turn_index, self.revision, self.valid_from)):
            raise ValueError("corrected-old fact counters cannot be negative")
        if self.valid_to is not None and self.valid_to < self.valid_from:
            raise ValueError("corrected-old valid_to precedes valid_from")
        if self.superseded_by is not None and not self.superseded_by.startswith("old1:"):
            raise ValueError("corrected-old supersession id is malformed")
        if self.fact_sha256 != content_digest(self._digest_payload()):
            raise ValueError("corrected-old fact digest differs")

    def _digest_payload(self) -> dict[str, Any]:
        return _fact_payload(
            fact_id=self.fact_id,
            entity=self.entity,
            attribute=self.attribute,
            value=self.value,
            source_input_id=self.source_input_id,
            source_sha256=self.source_sha256,
            source_text=self.source_text,
            owner=self.owner,
            scope=self.scope,
            session=self.session,
            turn_index=self.turn_index,
            revision=self.revision,
            valid_from=self.valid_from,
            valid_to=self.valid_to,
            superseded_by=self.superseded_by,
            status=self.status,
        )

    @classmethod
    def create(
        cls,
        *,
        entity: str,
        attribute: str,
        value: str,
        source_input_id: str,
        source_sha256: str,
        source_text: str,
        owner: str,
        scope: str,
        session: str,
        turn_index: int,
        revision: int,
        valid_from: int,
        status: str = "resolved",
    ) -> DiscreteFactV1:
        fact_id = (
            "old1:"
            + content_digest(
                {
                    "source_input_id": source_input_id,
                    "source_sha256": source_sha256,
                    "owner": owner,
                    "scope": scope,
                    "session": session,
                    "turn_index": turn_index,
                    "revision": revision,
                }
            )[:24]
        )
        digest = content_digest(
            _fact_payload(
                fact_id=fact_id,
                entity=entity,
                attribute=attribute,
                value=value,
                source_input_id=source_input_id,
                source_sha256=source_sha256,
                source_text=source_text,
                owner=owner,
                scope=scope,
                session=session,
                turn_index=turn_index,
                revision=revision,
                valid_from=valid_from,
                valid_to=None,
                superseded_by=None,
                status=status,
            )
        )
        return cls(
            fact_id=fact_id,
            entity=entity,
            attribute=attribute,
            value=value,
            source_input_id=source_input_id,
            source_sha256=source_sha256,
            source_text=source_text,
            owner=owner,
            scope=scope,
            session=session,
            turn_index=turn_index,
            revision=revision,
            valid_from=valid_from,
            valid_to=None,
            superseded_by=None,
            status=status,
            fact_sha256=digest,
        )

    def supersede(self, *, valid_to: int, superseded_by: str) -> DiscreteFactV1:
        if self.superseded_by is not None:
            raise ValueError("corrected-old fact is already superseded")
        digest = content_digest(
            _fact_payload(
                fact_id=self.fact_id,
                entity=self.entity,
                attribute=self.attribute,
                value=self.value,
                source_input_id=self.source_input_id,
                source_sha256=self.source_sha256,
                source_text=self.source_text,
                owner=self.owner,
                scope=self.scope,
                session=self.session,
                turn_index=self.turn_index,
                revision=self.revision,
                valid_from=self.valid_from,
                valid_to=int(valid_to),
                superseded_by=superseded_by,
                status=self.status,
            )
        )
        return replace(
            self, valid_to=int(valid_to), superseded_by=superseded_by, fact_sha256=digest
        )

    def reopen(self) -> DiscreteFactV1:
        if self.superseded_by is None and self.valid_to is None:
            return self
        digest = content_digest(
            _fact_payload(
                fact_id=self.fact_id,
                entity=self.entity,
                attribute=self.attribute,
                value=self.value,
                source_input_id=self.source_input_id,
                source_sha256=self.source_sha256,
                source_text=self.source_text,
                owner=self.owner,
                scope=self.scope,
                session=self.session,
                turn_index=self.turn_index,
                revision=self.revision,
                valid_from=self.valid_from,
                valid_to=None,
                superseded_by=None,
                status=self.status,
            )
        )
        return replace(self, valid_to=None, superseded_by=None, fact_sha256=digest)

    def to_payload(self) -> dict[str, Any]:
        return {**self._digest_payload(), "fact_sha256": self.fact_sha256}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> DiscreteFactV1:
        expected = {
            "format",
            "version",
            "fact_id",
            "entity",
            "attribute",
            "value",
            "source_input_id",
            "source_sha256",
            "source_text",
            "owner",
            "scope",
            "session",
            "turn_index",
            "revision",
            "valid_from",
            "valid_to",
            "superseded_by",
            "status",
            "fact_sha256",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("corrected-old fact payload schema differs")
        if (payload["format"], payload["version"]) != (DISCRETE_FACT_FORMAT, 1):
            raise ValueError("unsupported corrected-old fact version")
        return cls(
            fact_id=str(payload["fact_id"]),
            entity=str(payload["entity"]),
            attribute=str(payload["attribute"]),
            value=str(payload["value"]),
            source_input_id=str(payload["source_input_id"]),
            source_sha256=str(payload["source_sha256"]),
            source_text=str(payload["source_text"]),
            owner=str(payload["owner"]),
            scope=str(payload["scope"]),
            session=str(payload["session"]),
            turn_index=int(payload["turn_index"]),
            revision=int(payload["revision"]),
            valid_from=int(payload["valid_from"]),
            valid_to=None if payload["valid_to"] is None else int(payload["valid_to"]),
            superseded_by=(
                None if payload["superseded_by"] is None else str(payload["superseded_by"])
            ),
            status=str(payload["status"]),
            fact_sha256=str(payload["fact_sha256"]),
        )


@dataclass(frozen=True)
class DiscreteFactStoreV1:
    """Hard-gated discrete store; unresolved content cannot enter."""

    facts: tuple[DiscreteFactV1, ...]

    FORMAT = DISCRETE_STORE_FORMAT

    def __post_init__(self) -> None:
        if len({item.fact_id for item in self.facts}) != len(self.facts):
            raise ValueError("corrected-old store repeats a fact id")
        active: dict[tuple[str, str], str] = {}
        for item in self.facts:
            if item.valid_to is not None:
                continue
            key = (item.entity, item.attribute)
            if key in active:
                raise ValueError("corrected-old store has two active values for one key")
            active[key] = item.fact_id
        ids = {item.fact_id for item in self.facts}
        for item in self.facts:
            if item.superseded_by is not None and item.superseded_by not in ids:
                raise ValueError("corrected-old store supersession target is absent")

    def add(self, fact: DiscreteFactV1) -> DiscreteFactStoreV1:
        if any(item.fact_id == fact.fact_id for item in self.facts):
            raise ValueError("corrected-old store already contains this fact")
        if fact.superseded_by is not None or fact.valid_to is not None:
            raise ValueError("only a current fact can be added")
        previous = self.current(fact.entity, fact.attribute)
        rows = list(self.facts)
        if previous is not None:
            if fact.revision <= previous.revision:
                raise ValueError("corrected-old correction revision must advance")
            rows = [
                (
                    item.supersede(valid_to=fact.valid_from, superseded_by=fact.fact_id)
                    if item.fact_id == previous.fact_id
                    else item
                )
                for item in rows
            ]
        rows.append(fact)
        return DiscreteFactStoreV1(tuple(rows))

    def current(self, entity: str, attribute: str) -> DiscreteFactV1 | None:
        for item in self.facts:
            if item.entity == entity and item.attribute == attribute and item.valid_to is None:
                return item
        return None

    def history(self, entity: str, attribute: str) -> tuple[DiscreteFactV1, ...]:
        return tuple(
            sorted(
                (
                    item
                    for item in self.facts
                    if item.entity == entity and item.attribute == attribute
                ),
                key=lambda item: (item.revision, item.turn_index),
            )
        )

    def previous(self, entity: str, attribute: str) -> DiscreteFactV1 | None:
        chain = self.history(entity, attribute)
        current = self.current(entity, attribute)
        if current is None or len(chain) < 2:
            return None
        for position, item in enumerate(chain):
            if item.fact_id == current.fact_id:
                return chain[position - 1] if position > 0 else None
        return None

    def readable(self, *, owner: str, scope: str, session: str) -> tuple[DiscreteFactV1, ...]:
        return tuple(
            sorted(
                (
                    item
                    for item in self.facts
                    if item.owner == owner and item.scope == scope and item.session == session
                ),
                key=lambda item: item.fact_id,
            )
        )

    def cut(self, fact_id: str) -> DiscreteFactStoreV1:
        if not any(item.fact_id == fact_id for item in self.facts):
            raise ValueError("corrected-old cut target is absent")
        survivors = tuple(item for item in self.facts if item.fact_id != fact_id)
        return DiscreteFactStoreV1(
            tuple(item.reopen() if item.superseded_by == fact_id else item for item in survivors)
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": self.FORMAT,
            "version": 1,
            "facts": [item.to_payload() for item in self.facts],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> DiscreteFactStoreV1:
        required = {"format", "version", "facts"}
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError("corrected-old store payload schema differs")
        if (payload["format"], payload["version"]) != (cls.FORMAT, 1):
            raise ValueError("unsupported corrected-old store version")
        rows = payload["facts"]
        if not isinstance(rows, list):
            raise ValueError("corrected-old store facts must be a list")
        return cls(tuple(DiscreteFactV1.from_payload(item) for item in rows))


@dataclass(frozen=True)
class BaselineQueryParseV1:
    query: bytes
    entity: str | None
    attribute: str | None
    time_intent: str

    def __post_init__(self) -> None:
        if not isinstance(self.query, bytes) or not self.query:
            raise ValueError("baseline parse needs nonempty query bytes")
        if self.time_intent not in TIME_INTENT_INDEX:
            raise ValueError("baseline parse time intent is outside the declared set")

    @property
    def parsed(self) -> bool:
        return self.entity is not None and self.attribute is not None

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": "taiji-corrected-old-query-parse-v1",
            "version": 1,
            "query_hex": self.query.hex(),
            "entity": self.entity,
            "attribute": self.attribute,
            "time_intent": self.time_intent,
        }


def parse_query(query: bytes, *, time_intent: str) -> BaselineQueryParseV1:
    """Hard keyword parse; a missing entity or attribute is a parse failure."""

    if not isinstance(query, bytes) or not query:
        raise ValueError("baseline query must be nonempty bytes")
    text = query.decode("utf-8")
    entity = next((name for name in ENTITIES if name in text), None)
    attribute = next(
        (name for name, terms in ATTRIBUTE_TERMS.items() if any(term in text for term in terms)),
        None,
    )
    return BaselineQueryParseV1(
        query=query, entity=entity, attribute=attribute, time_intent=time_intent
    )


@dataclass(frozen=True)
class BaselineSelectionV1:
    parse: BaselineQueryParseV1
    fact: DiscreteFactV1 | None
    value: str | None
    reason: str

    @property
    def answerable(self) -> bool:
        return self.parse.parsed and self.fact is not None

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": "taiji-corrected-old-selection-v1",
            "version": 1,
            "parse": self.parse.to_payload(),
            "fact": None if self.fact is None else self.fact.to_payload(),
            "value": self.value,
            "reason": self.reason,
            "answerable": self.answerable,
        }


@dataclass(frozen=True)
class BaselineAnswerV1:
    generated: bytes
    selection: BaselineSelectionV1

    FORMAT = "taiji-corrected-old-answer-v1"

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": self.FORMAT,
            "version": 1,
            "generated_hex": self.generated.hex(),
            "selection": self.selection.to_payload(),
        }

    def answer_digest(self) -> str:
        return content_digest(self.to_payload())


class CorrectedOldBaselineV1(ConditionalLanguageCandidateV3):
    """Frozen V3 language head plus one fixed value bias at every step."""

    def __init__(
        self,
        config: TaskCoreConfigV3 | None = None,
        *,
        bundle_id: str = "corrected-old-v1",
        bias_seed: int = 1770311,
    ):
        super().__init__(config, bundle_id=bundle_id)
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(bias_seed)
            self.value_bias = nn.Embedding(NULL_VALUE_INDEX + 1, 257, dtype=torch.float32)
            nn.init.normal_(self.value_bias.weight, mean=0.0, std=0.02)
        self.bias_seed = bias_seed
        for name, parameter in self.named_parameters():
            if name != "value_bias.weight":
                parameter.requires_grad_(False)

    def trainable_parameters(self) -> tuple[str, ...]:
        return tuple(name for name, parameter in self.named_parameters() if parameter.requires_grad)

    def value_bias_for(self, value: str | None) -> torch.Tensor:
        index = NULL_VALUE_INDEX if value is None else VALUE_INDEX.get(value, NULL_VALUE_INDEX)
        bias: torch.Tensor = self.value_bias(torch.tensor(index, dtype=torch.long))
        return bias

    def _conditioned_state(self, query: bytes, *, owner: str, scope: str, session: str):
        state = self.new_state(scope_id=scope, session_id=session, native_state_ref="corrected-old")
        return self.step_task(
            query,
            state,
            scope_id=scope,
            session_id=session,
            source_id="corrected-old:query",
            resource_snapshot={
                "remaining_steps": 1,
                "max_observation_bytes": self.config.max_observation_bytes,
            },
        )

    def teacher_forced(
        self,
        *,
        query: bytes,
        target: bytes,
        value: str | None,
        owner: str = "u",
        scope: str = "w",
        session: str = "s",
    ) -> torch.Tensor:
        """One logit row per target byte; row 0 predicts target[0]."""

        if not target or len(target) > self.config.max_generation_bytes:
            raise ValueError("corrected-old target is empty or exceeds the generation bound")
        state = self._conditioned_state(query, owner=owner, scope=scope, session=session)
        cache = self.evidence_cache(state)
        bias = self.value_bias_for(value)
        rows: list[torch.Tensor] = []
        for symbol in target:
            logits, hidden = super().next_logits(state, cache=cache)
            rows.append(logits + bias)
            state = self.observe_response(state, symbol, hidden)
        return torch.stack(rows)

    @torch.no_grad()
    def answer(
        self,
        *,
        query: bytes,
        value: str | None,
        max_new_bytes: int = 64,
        owner: str = "u",
        scope: str = "w",
        session: str = "s",
    ) -> bytes:
        if max_new_bytes <= 0 or max_new_bytes > self.config.max_generation_bytes:
            raise ValueError("corrected-old generation allowance is outside the bound")
        state = self._conditioned_state(query, owner=owner, scope=scope, session=session)
        cache = self.evidence_cache(state)
        bias = self.value_bias_for(value)
        produced: list[int] = []
        for _ in range(max_new_bytes):
            logits, hidden = super().next_logits(state, cache=cache)
            logits = logits + bias
            allowed = utf8_allowed(state.utf8_remaining, state.utf8_lead)
            choices = torch.tensor(allowed, dtype=torch.long, device=logits.device)
            symbol = int(choices[logits[choices].argmax()])
            state = self.observe_response(state, symbol, hidden)
            produced.append(symbol)
            if bytes(produced).endswith(END_MARKER):
                break
        return bytes(produced)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


@dataclass
class CorrectedOldBundleV1:
    """Model plus discrete store; the whole baseline restores at zero steps."""

    model: CorrectedOldBaselineV1
    store: DiscreteFactStoreV1
    cursor: dict[str, Any]

    FORMAT = BASELINE_BUNDLE_FORMAT

    @classmethod
    def create(
        cls,
        config: TaskCoreConfigV3 | None = None,
        *,
        bundle_id: str = "corrected-old-v1",
        bias_seed: int = 1770311,
    ) -> CorrectedOldBundleV1:
        return cls(
            model=CorrectedOldBaselineV1(config, bundle_id=bundle_id, bias_seed=bias_seed),
            store=DiscreteFactStoreV1(()),
            cursor={},
        )

    def ingest(
        self,
        *,
        source_text: str,
        entity: str | None,
        attribute: str | None,
        value: str | None,
        source_input_id: str,
        owner: str,
        scope: str,
        session: str,
        turn_index: int,
        revision: int,
        valid_from: int | None = None,
        status: str = "resolved",
    ) -> DiscreteFactV1 | None:
        """The hard gate: unlabeled or unresolved content is dropped."""

        if status != "resolved" or not all((entity, attribute, value)):
            return None
        raw = source_text.encode("utf-8")
        fact = DiscreteFactV1.create(
            entity=str(entity),
            attribute=str(attribute),
            value=str(value),
            source_input_id=source_input_id,
            source_sha256=hashlib.sha256(raw).hexdigest(),
            source_text=source_text,
            owner=owner,
            scope=scope,
            session=session,
            turn_index=turn_index,
            revision=revision,
            valid_from=turn_index if valid_from is None else valid_from,
            status=status,
        )
        self.store = self.store.add(fact)
        return fact

    def condition(
        self,
        *,
        query: bytes,
        time_intent: str,
        owner: str,
        scope: str,
        session: str,
    ) -> BaselineSelectionV1:
        parse = parse_query(query, time_intent=time_intent)
        if not parse.parsed:
            return BaselineSelectionV1(parse=parse, fact=None, value=None, reason="parse_failure")
        entity, attribute = str(parse.entity), str(parse.attribute)
        visible = {
            item.fact_id for item in self.store.readable(owner=owner, scope=scope, session=session)
        }
        if time_intent == "current":
            fact = self.store.current(entity, attribute)
            reason = "selected"
        elif time_intent == "previous":
            fact = self.store.previous(entity, attribute)
            reason = "selected"
        else:
            # The old graph has no unknown path; this fallback is a declared
            # baseline limitation, reported by the probe instead of hidden.
            fact = self.store.current(entity, attribute)
            reason = "unknown_fallback_current"
        if fact is None:
            return BaselineSelectionV1(parse=parse, fact=None, value=None, reason="no_fact")
        if fact.fact_id not in visible:
            return BaselineSelectionV1(parse=parse, fact=None, value=None, reason="not_visible")
        return BaselineSelectionV1(parse=parse, fact=fact, value=fact.value, reason=reason)

    def answer(
        self,
        *,
        query: bytes,
        time_intent: str = "current",
        owner: str,
        scope: str,
        session: str,
        max_new_bytes: int = 64,
    ) -> BaselineAnswerV1:
        selection = self.condition(
            query=query, time_intent=time_intent, owner=owner, scope=scope, session=session
        )
        value = selection.value if selection.answerable else None
        return BaselineAnswerV1(
            generated=self.model.answer(
                query=query,
                value=value,
                max_new_bytes=max_new_bytes,
                owner=owner,
                scope=scope,
                session=session,
            ),
            selection=selection,
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
    ) -> tuple[torch.Tensor, BaselineSelectionV1]:
        selection = self.condition(
            query=query, time_intent=time_intent, owner=owner, scope=scope, session=session
        )
        value = selection.value if selection.answerable else None
        logits = self.model.teacher_forced(
            query=query,
            target=target,
            value=value,
            owner=owner,
            scope=scope,
            session=session,
        )
        return logits, selection

    def checkpoint(self, *, state: Any | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": self.FORMAT,
            "version": 1,
            "bundle_id": self.model.bundle_id,
            "config": asdict(self.model.config),
            "bias_seed": self.model.bias_seed,
            "parameter_revision": self.model.parameter_revision,
            "parameters": {
                name: value.detach().cpu().clone()
                for name, value in self.model.state_dict().items()
            },
            "store": self.store.to_payload(),
            "cursor": copy.deepcopy(self.cursor),
            "task_state": None if state is None else state.to_payload(),
        }
        payload["checkpoint_digest"] = content_digest(payload)
        return payload

    def save(self, path: Path, *, state: Any | None = None) -> Path:
        target = Path(path)
        if target.exists():
            raise ValueError("corrected-old bundle checkpoint is write-once")
        atomic_save(self.checkpoint(state=state), target)
        return target

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> CorrectedOldBundleV1:
        expected = {
            "format",
            "version",
            "bundle_id",
            "config",
            "bias_seed",
            "parameter_revision",
            "parameters",
            "store",
            "cursor",
            "task_state",
            "checkpoint_digest",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("corrected-old bundle payload schema differs")
        if (payload["format"], payload["version"]) != (cls.FORMAT, 1):
            raise ValueError("unsupported corrected-old bundle version")
        if (
            content_digest(
                {name: value for name, value in payload.items() if name != "checkpoint_digest"}
            )
            != payload["checkpoint_digest"]
        ):
            raise ValueError("corrected-old bundle checkpoint digest differs")
        model = CorrectedOldBaselineV1(
            TaskCoreConfigV3(**payload["config"]),
            bundle_id=payload["bundle_id"],
            bias_seed=int(payload["bias_seed"]),
        )
        model.load_state_dict(payload["parameters"], strict=True)
        model.parameter_revision = int(payload["parameter_revision"])
        return cls(
            model=model,
            store=DiscreteFactStoreV1.from_payload(payload["store"]),
            cursor=copy.deepcopy(dict(payload["cursor"])),
        )

    @classmethod
    def load(cls, path: Path) -> CorrectedOldBundleV1:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        return cls.from_checkpoint(payload)

    def probe_digest(
        self,
        *,
        query: bytes,
        time_intent: str = "current",
        owner="probe",
        scope="probe",
        session="probe",
    ) -> str:
        logits, selection = self.score_target(
            query=query,
            target=END_MARKER,
            time_intent=time_intent,
            owner=owner,
            scope=scope,
            session=session,
        )
        answer = self.answer(
            query=query,
            time_intent=time_intent,
            owner=owner,
            scope=scope,
            session=session,
        )
        return content_digest(
            {
                "selection": selection.to_payload(),
                "logits": [[round(float(value), 7) for value in row] for row in logits.tolist()],
                "answer": answer.answer_digest(),
            }
        )


@dataclass(frozen=True)
class BaselinePredictionV1:
    case_id: str
    generated: bytes
    parse_ok: bool

    def to_payload(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "generated_hex": self.generated.hex(),
            "parse_ok": self.parse_ok,
        }


def score_fixed_denominator(
    cases: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, BaselinePredictionV1],
) -> dict[str, Any]:
    """Every declared turn is scored; parse failures and abstentions count as wrong."""

    scored = len(cases)
    exact = 0
    parse_failures = 0
    abstentions = 0
    by_category: dict[str, dict[str, int]] = {}
    missing: list[str] = []
    for case in cases:
        case_id = str(case["fixture_id"])
        expected = bytes(case["target"]["target_bytes_utf8"])
        category = str(case["category"])
        bucket = by_category.setdefault(category, {"turns": 0, "exact": 0, "parse_failures": 0})
        bucket["turns"] += 1
        prediction = predictions.get(case_id)
        if prediction is None:
            missing.append(case_id)
            continue
        if not prediction.parse_ok:
            parse_failures += 1
            bucket["parse_failures"] += 1
            continue
        if not prediction.generated:
            abstentions += 1
            continue
        if prediction.generated == expected:
            exact += 1
            bucket["exact"] += 1
    return {
        "format": "taiji-corrected-old-fixed-denominator-score-v1",
        "version": 1,
        "scored_turns": scored,
        "exact": exact,
        "parse_failures": parse_failures,
        "abstentions": abstentions,
        "missing_predictions": missing,
        "answer_exact": exact / scored if scored else None,
        "by_category": by_category,
    }
