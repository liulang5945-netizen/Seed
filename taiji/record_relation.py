"""14.4 record-relation content subgraph: record encoding, record addressing
and byte-level reading inside one subject.

This module owns only the new content interface declared in
``SPEC-TAIJI-CORE_RECORD_RELATION_14_4``.  It does not replace the R021 state
graph; the same subject keeps H/F, the inherited byte encoder, the inherited
query read and the inherited generate/mix heads.  The new path encodes every
fully observed legal record with a shared bidirectional GRU, addresses records
before addressing bytes inside a record, and returns one content distribution
that both the recursive state update and the raw byte output reuse.

The host never splits a record at ``是``, never supplies entity spans, never
renders an action id and never fills the answer.  Source identities are used
for coverage alignment only; they are not semantic features.

Engineering only: no task training, no pricing, no adoption claim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from taiji.living_core import RawSource, position_features

RECORD_RELATION_SCHEMA = "taiji-record-relation-v1"
DEFAULT_SEED = 27031
DEFAULT_CHUNK = 8


def _matrix(
    rows: int, columns: int, generator: torch.Generator, dtype: torch.dtype
) -> nn.Parameter:
    value = torch.empty(rows, columns, dtype=dtype)
    std = 1.0 / math.sqrt(columns)
    value.normal_(mean=0.0, std=std, generator=generator)
    return nn.Parameter(value)


def _vector(size: int, std: float, generator: torch.Generator, dtype: torch.dtype) -> nn.Parameter:
    value = torch.empty(size, dtype=dtype)
    value.normal_(mean=0.0, std=std, generator=generator)
    return nn.Parameter(value)


def expected_parameter_count(width: int, key_width: int, hidden: int) -> int:
    """The exact count in SPEC §4.4, generalised only in d, k and h."""
    gru = 2 * (3 * hidden * width + 3 * hidden * hidden + 6 * hidden)
    return (
        gru
        + (width * (2 * hidden) + width)
        + (width * width + width + width)
        + (key_width * width + key_width)
        + key_width * width
        + (width * width + width)
        + (key_width * (2 * width + 64) + key_width)
        + key_width * (width + 4)
        + (key_width + width)
        + (key_width * (key_width + width) + key_width)
        + key_width * width
        + (width * width + width)
        + 1
    )


@dataclass(frozen=True)
class RecordRelationConfig:
    width: int = 128
    key_width: int = 32
    hidden: int = 64
    chunk: int = DEFAULT_CHUNK
    seed: int = DEFAULT_SEED

    def parameter_count(self) -> int:
        return expected_parameter_count(self.width, self.key_width, self.hidden)


@dataclass(frozen=True)
class RecordSourceView:
    """The formal source view handed to the reader.

    ``coverage_keys`` are the (source_id, version, byte_offset) identities the
    coverage tensor is aligned to.  Record order is not encoded as a position
    id; byte positions exist only inside a record.
    """

    keys: tuple[tuple[str, int], ...]
    coverage_keys: tuple[tuple[str, int, int], ...]
    source_ids: tuple[str, ...]
    record_offsets: tuple[tuple[int, int], ...]
    raw: tuple[bytes, ...]
    metadata: torch.Tensor
    units: tuple[torch.Tensor, ...]
    byte_ids: torch.Tensor
    lengths: tuple[int, ...]

    @property
    def record_count(self) -> int:
        return len(self.units)

    @property
    def total_bytes(self) -> int:
        return sum(self.lengths)

    def slice(self, index: int) -> slice:
        start, end = self.record_offsets[index]
        return slice(start, end)


def empty_view(reference: torch.Tensor, width: int) -> RecordSourceView:
    return RecordSourceView(
        keys=(),
        coverage_keys=(),
        source_ids=(),
        record_offsets=(),
        raw=(),
        metadata=reference.new_empty((0, 4)),
        units=(),
        byte_ids=reference.new_empty((0,), dtype=torch.long),
        lengths=(),
    )


def guard_records(records: tuple[RawSource, ...]) -> tuple[RawSource, ...]:
    """Refuse duplicate or malformed identities before any encoding starts."""
    if len({record.source_id for record in records}) != len(records):
        raise ValueError("Source identities must be unique")
    for record in records:
        if not record.source_id or record.observed_at < 0 or record.version < 0:
            raise ValueError("Malformed source metadata")
    return records


def align_coverage(
    coverage: torch.Tensor, keys: tuple[tuple[str, int, int], ...], view: RecordSourceView
) -> torch.Tensor:
    """Restore coverage by (source_id, version, byte_offset); new positions are 0."""
    old = {key: index for index, key in enumerate(keys)}
    zero = coverage.new_zeros(())
    if not view.coverage_keys:
        return coverage.new_empty(0)
    return torch.stack([coverage[old[key]] if key in old else zero for key in view.coverage_keys])


def advance_coverage(coverage: torch.Tensor, copy_weights: torch.Tensor) -> torch.Tensor:
    if coverage.numel() != copy_weights.numel():
        raise ValueError("Coverage and copy weights must share one byte grid")
    return coverage + copy_weights


@dataclass(frozen=True)
class RelationRead:
    """One read step.  ``alpha_null`` is the learned null-record mass."""

    content: torch.Tensor
    copy_weights: torch.Tensor
    copy_byte_mass: torch.Tensor
    alpha: torch.Tensor | None
    betas: tuple[torch.Tensor, ...]
    alpha_null: torch.Tensor
    record_mass: torch.Tensor
    forced: bool


class RecordRelationEncoder(nn.Module):
    """Shared bidirectional record/goal encoder and summary pool."""

    def __init__(self, config: RecordRelationConfig) -> None:
        super().__init__()
        self.config = config
        generator = torch.Generator().manual_seed(config.seed)
        dtype = torch.float64
        width, key_width, hidden = config.width, config.key_width, config.hidden

        self.gru = nn.GRU(width, hidden, bidirectional=True, dtype=dtype)
        with torch.no_grad():
            bound = 1.0 / math.sqrt(hidden)
            for parameter in self.gru.parameters():
                parameter.uniform_(-bound, bound, generator=generator)

        self.W_c = nn.Linear(2 * hidden, width, dtype=dtype)
        self.A_pool = nn.Linear(width, width, dtype=dtype)
        self.v_pool = nn.Linear(width, 1, bias=False, dtype=dtype)
        self.Q_goal = nn.Linear(width, key_width, dtype=dtype)
        self.K_goal = nn.Linear(width, key_width, bias=False, dtype=dtype)
        self.V_goal = nn.Linear(width, width, dtype=dtype)

        self.W_c.weight = _matrix(width, 2 * hidden, generator, dtype)
        self.A_pool.weight = _matrix(width, width, generator, dtype)
        self.v_pool.weight = _matrix(1, width, generator, dtype)
        self.Q_goal.weight = _matrix(key_width, width, generator, dtype)
        self.K_goal.weight = _matrix(key_width, width, generator, dtype)
        self.V_goal.weight = _matrix(width, width, generator, dtype)
        for layer in (self.W_c, self.A_pool, self.Q_goal, self.V_goal):
            nn.init.zeros_(layer.bias)

    def encode_units(self, subject, raw: bytes) -> torch.Tensor:
        """Encode one already-complete record (or the goal) into width units.

        ``a_ij`` is the inherited causal byte context plus a record-local
        position; the two GRU directions then see the whole record.  The
        inherited encoder parameters stay plastic in this candidate.
        """
        reference = self.W_c.weight
        if not raw:
            return reference.new_empty((0, self.config.width))
        context: torch.Tensor = subject.encoder.context(raw)
        local = position_features(len(raw), self.config.width, reference)
        activated = context + local
        if getattr(self, "local", False):
            return activated
        forward, _ = self.gru(activated.unsqueeze(1))
        return cast(torch.Tensor, activated + self.W_c(forward[:, 0, :]))

    def pool(self, units: torch.Tensor) -> torch.Tensor:
        if units.shape[0] == 0:
            return self.A_pool.weight.new_zeros(self.config.width)
        scores = self.v_pool(torch.tanh(self.A_pool(units))).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        return weights @ units

    def build_view(
        self,
        subject,
        records: tuple[RawSource, ...],
        *,
        owner: str = "owner",
        scope: str = "scope",
        now: int = 0,
    ) -> RecordSourceView:
        guard_records(records)
        legal = [
            record
            for record in records
            if record.owner == owner
            and record.scope == scope
            and record.observed_at <= now
            and record.real
            and record.raw
        ]
        if sum(len(record.raw) for record in legal) > subject.config.max_source_bytes:
            raise ValueError("Record source-byte budget exceeded")
        if not legal:
            return empty_view(self.W_c.weight, self.config.width)
        keys: list[tuple[str, int]] = []
        coverage_keys: list[tuple[str, int, int]] = []
        source_ids: list[str] = []
        offsets: list[tuple[int, int]] = []
        raw_rows: list[bytes] = []
        metadata: list[torch.Tensor] = []
        units: list[torch.Tensor] = []
        lengths: list[int] = []
        byte_rows: list[int] = []
        cursor = 0
        for record in legal:
            encoded = self.encode_units(subject, record.raw)
            length = len(record.raw)
            keys.extend((record.source_id, offset) for offset in range(length))
            coverage_keys.extend(
                (record.source_id, record.version, offset) for offset in range(length)
            )
            source_ids.append(record.source_id)
            offsets.append((cursor, cursor + length))
            raw_rows.append(record.raw)
            lengths.append(length)
            units.append(encoded)
            byte_rows.extend(record.raw)
            time = self.W_c.weight.new_tensor(
                [(now - record.observed_at) / 64, record.observed_at / 64, record.version / 16]
            ).tanh()
            metadata.append(torch.cat([time, self.W_c.weight.new_ones(1)]))
            cursor += length
        return RecordSourceView(
            keys=tuple(keys),
            coverage_keys=tuple(coverage_keys),
            source_ids=tuple(source_ids),
            record_offsets=tuple(offsets),
            raw=tuple(raw_rows),
            metadata=torch.stack(metadata),
            units=tuple(units),
            byte_ids=torch.tensor(byte_rows, dtype=torch.long, device=self.W_c.weight.device),
            lengths=tuple(lengths),
        )


def goal_attention(
    encoder: RecordRelationEncoder, s: torch.Tensor, goal_units: torch.Tensor
) -> torch.Tensor:
    q_goal: torch.Tensor = encoder.Q_goal(s)
    if goal_units.shape[0] == 0:
        return cast(torch.Tensor, encoder.V_goal(s.new_zeros(encoder.config.width)))
    keys: torch.Tensor = encoder.K_goal(goal_units)
    values: torch.Tensor = encoder.V_goal(goal_units)
    attention: torch.Tensor = torch.softmax(
        q_goal @ keys.T / math.sqrt(encoder.config.key_width), dim=-1
    )
    return attention @ values


class RecordRelationReader(nn.Module):
    """Record-level alpha, in-record beta and the shared content distribution."""

    def __init__(self, config: RecordRelationConfig) -> None:
        super().__init__()
        self.config = config
        generator = torch.Generator().manual_seed(config.seed + 1)
        dtype = torch.float64
        width, key_width = config.width, config.key_width

        self.Q_read = nn.Linear(2 * width + 64, key_width, dtype=dtype)
        self.K_record = nn.Linear(width + 4, key_width, bias=False, dtype=dtype)
        self.null_key = _vector(key_width, 1.0 / math.sqrt(key_width), generator, dtype)
        self.null_value = _vector(width, 1.0 / math.sqrt(width), generator, dtype)
        self.Q_value = nn.Linear(key_width + width, key_width, dtype=dtype)
        self.K_position = nn.Linear(width, key_width, bias=False, dtype=dtype)
        self.V_position = nn.Linear(width, width, dtype=dtype)
        self.eta = nn.Parameter(torch.zeros((), dtype=dtype))

        self.Q_read.weight = _matrix(key_width, 2 * width + 64, generator, dtype)
        self.K_record.weight = _matrix(key_width, width + 4, generator, dtype)
        self.Q_value.weight = _matrix(key_width, key_width + width, generator, dtype)
        self.K_position.weight = _matrix(key_width, width, generator, dtype)
        self.V_position.weight = _matrix(width, width, generator, dtype)
        for layer in (self.Q_read, self.Q_value, self.V_position):
            nn.init.zeros_(layer.bias)

    def read(
        self,
        encoder: RecordRelationEncoder,
        s: torch.Tensor,
        previous_embedding: torch.Tensor,
        goal_units: torch.Tensor,
        view: RecordSourceView,
        coverage: torch.Tensor,
        *,
        flat: bool,
        force: dict | None = None,
    ) -> RelationRead:
        query = self.Q_read(
            torch.cat([s, goal_attention(encoder, s, goal_units), previous_embedding])
        )
        softplus_eta = F.softplus(self.eta)
        if force is not None:
            return self._forced(encoder, view, coverage, force, flat)
        if flat:
            return self._flat(encoder, view, coverage, query, softplus_eta)
        return self._records(encoder, view, coverage, query, softplus_eta)

    def _byte_mass(
        self, view: RecordSourceView, coverage: torch.Tensor, copy_weights
    ) -> torch.Tensor:
        return coverage.new_zeros(257).index_add_(0, view.byte_ids, copy_weights)

    def _record_mass(
        self, view: RecordSourceView, coverage: torch.Tensor, copy_weights
    ) -> torch.Tensor:
        if not view.record_count:
            return coverage.new_empty(0)
        return torch.stack([copy_weights[view.slice(i)].sum() for i in range(view.record_count)])

    def _null_only(
        self, view: RecordSourceView, coverage: torch.Tensor, alpha_null
    ) -> RelationRead:
        return RelationRead(
            content=alpha_null * self.null_value,
            copy_weights=coverage.new_empty(0),
            copy_byte_mass=coverage.new_zeros(257),
            alpha=torch.stack([alpha_null]) if view.record_count == 0 else None,
            betas=(),
            alpha_null=alpha_null,
            record_mass=coverage.new_empty(0),
            forced=False,
        )

    def _flat(
        self,
        encoder: RecordRelationEncoder,
        view: RecordSourceView,
        coverage: torch.Tensor,
        query: torch.Tensor,
        softplus_eta: torch.Tensor,
    ) -> RelationRead:
        key_width = self.config.key_width
        null_logit = query @ self.null_key / math.sqrt(key_width)
        if view.record_count == 0:
            probability = torch.softmax(torch.stack([null_logit]), dim=-1)[0]
            return self._null_only(view, coverage, probability)
        units = torch.cat(view.units)
        position_logits = (
            self.K_position(units) @ query / math.sqrt(key_width) - softplus_eta * coverage
        )
        probabilities = torch.softmax(torch.cat([position_logits, null_logit.reshape(1)]), dim=-1)
        copy_weights = probabilities[:-1]
        alpha_null = probabilities[-1]
        content = copy_weights @ self.V_position(units) + alpha_null * self.null_value
        return RelationRead(
            content=content,
            copy_weights=copy_weights,
            copy_byte_mass=self._byte_mass(view, coverage, copy_weights),
            alpha=None,
            betas=(),
            alpha_null=alpha_null,
            record_mass=self._record_mass(view, coverage, copy_weights),
            forced=False,
        )

    def _records(
        self,
        encoder: RecordRelationEncoder,
        view: RecordSourceView,
        coverage: torch.Tensor,
        query: torch.Tensor,
        softplus_eta: torch.Tensor,
    ) -> RelationRead:
        key_width = self.config.key_width
        null_logit = query @ self.null_key / math.sqrt(key_width)
        if view.record_count == 0:
            probability = torch.softmax(torch.stack([null_logit]), dim=-1)[0]
            return self._null_only(view, coverage, probability)
        summaries = torch.stack([encoder.pool(view.units[i]) for i in range(view.record_count)])
        record_keys = self.K_record(torch.cat([summaries, view.metadata], dim=-1))
        alpha = torch.softmax(
            torch.cat([record_keys @ query / math.sqrt(key_width), null_logit.reshape(1)]), dim=-1
        )
        betas = []
        for index in range(view.record_count):
            q_value = self.Q_value(torch.cat([query, summaries[index]]))
            logits = (
                self.K_position(view.units[index]) @ q_value / math.sqrt(key_width)
                - softplus_eta * coverage[view.slice(index)]
            )
            betas.append(torch.softmax(logits, dim=-1))
        pieces = [
            (alpha[index] * betas[index]) @ self.V_position(view.units[index])
            for index in range(view.record_count)
        ]
        content = torch.stack(pieces).sum(0) + alpha[-1] * self.null_value
        copy_weights = torch.cat(
            [alpha[index] * betas[index] for index in range(view.record_count)]
        )
        return RelationRead(
            content=content,
            copy_weights=copy_weights,
            copy_byte_mass=self._byte_mass(view, coverage, copy_weights),
            alpha=alpha,
            betas=tuple(betas),
            alpha_null=alpha[-1],
            record_mass=self._record_mass(view, coverage, copy_weights),
            forced=False,
        )

    def _forced(
        self,
        encoder: RecordRelationEncoder,
        view: RecordSourceView,
        coverage: torch.Tensor,
        force: dict,
        flat: bool,
    ) -> RelationRead:
        if flat:
            probabilities = force["flat"]
            copy_weights = probabilities[:-1]
            alpha_null = probabilities[-1]
            units = (
                torch.cat(view.units)
                if view.record_count
                else coverage.new_empty((0, self.config.width))
            )
            content = copy_weights @ self.V_position(units) + alpha_null * self.null_value
            return RelationRead(
                content=content,
                copy_weights=copy_weights,
                copy_byte_mass=self._byte_mass(view, coverage, copy_weights),
                alpha=None,
                betas=(),
                alpha_null=alpha_null,
                record_mass=self._record_mass(view, coverage, copy_weights),
                forced=True,
            )
        alpha = force["alpha"]
        betas = force["betas"]
        pieces = [
            (alpha[index] * betas[index]) @ self.V_position(view.units[index])
            for index in range(view.record_count)
        ]
        content = (
            torch.stack(pieces).sum(0) if pieces else coverage.new_zeros(self.config.width)
        ) + alpha[-1] * self.null_value
        copy_weights = (
            torch.cat([alpha[index] * betas[index] for index in range(view.record_count)])
            if view.record_count
            else coverage.new_empty(0)
        )
        return RelationRead(
            content=content,
            copy_weights=copy_weights,
            copy_byte_mass=self._byte_mass(view, coverage, copy_weights),
            alpha=alpha,
            betas=tuple(betas),
            alpha_null=alpha[-1],
            record_mass=self._record_mass(view, coverage, copy_weights),
            forced=True,
        )


class RecordRelationModule(nn.Module):
    """The registered content subgraph ``record_relation_v1``."""

    def __init__(self, config: RecordRelationConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = RecordRelationEncoder(config)
        self.reader = RecordRelationReader(config)

    def encode_units(self, subject, raw: bytes) -> torch.Tensor:
        return self.encoder.encode_units(subject, raw)

    def build_view(self, subject, records: tuple[RawSource, ...], **kwargs) -> RecordSourceView:
        return self.encoder.build_view(subject, records, **kwargs)

    def pool(self, units: torch.Tensor) -> torch.Tensor:
        return self.encoder.pool(units)

    def read(
        self,
        s: torch.Tensor,
        previous_embedding: torch.Tensor,
        goal_units: torch.Tensor,
        view: RecordSourceView,
        coverage: torch.Tensor,
        *,
        flat: bool,
        force: dict | None = None,
    ) -> RelationRead:
        return self.reader.read(
            self.encoder, s, previous_embedding, goal_units, view, coverage, flat=flat, force=force
        )
