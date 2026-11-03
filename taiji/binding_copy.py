"""15.1 rev2 R1: 原文对齐、内容转移与后验连续复制.

The R0 profile is the inherited 14.6 point read.  R1 keeps the same subject,
record encoder, H/F update, generate/mix head and recovery protocol, and
replaces the record-level pooled selection with:

1. goal/source span alignment (E_goal, E_source, Q_w, Q_s, v_meta);
2. within-record content transport (Q_transport, K_transport, v_relative);
3. a carried copy posterior m with continuation gate lambda (continuation).

The 30,133 new parameters and the inherited/disabled groups are declared here.
No task training, no v2 data exposure, no confirm read.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch import nn

from taiji.living_core import EOS, LivingState
from taiji.record_relation import (
    RecordRelationConfig,
    RecordRelationReader,
    RecordSourceView,
)
from taiji.record_relation_core import (
    CONTENT_REVISION,
    MAX_OUTPUT_BYTES,
    MAX_PREDICTIONS,
    RecordRelationCandidate,
    _dtype_name,
    _gate,
    _labels_nll,
    _mode_vector,
    advance_h,
    build_subject_from_candidate_bundle,
    candidate_parameter_map,
    decode_endpoint,
    state_from_payload,
)
from taiji.record_relation_core import (
    PARAMETER_GROUPS as BASE_GROUPS,
)
from taiji.record_relation_core import (
    state_payload as living_payload,
)

BINDING_COPY_SCHEMA = "taiji-binding-copy-v1"
DEFAULT_WIDTH = 128
DEFAULT_KEY = 32
DEFAULT_SEED = 28031
SOURCE_WIDTH = 128
PREVIOUS_WIDTH = 64
META_WIDTH = 4
PE_WIDTH = 16
MAX_COVERAGE = 257

# Declared R1 parameter groups.  The reused V_position/null_key/null_value live
# in the inherited reader and are not repeated in the 30,133 new count.
R1_PARAMETER_GROUPS: dict[str, tuple[str, ...]] = {
    "E_goal": ("reader.E_goal.weight",),
    "E_source": ("reader.E_source.weight",),
    "Q_w": ("reader.Q_w.weight", "reader.Q_w.bias"),
    "Q_s": ("reader.Q_s.weight", "reader.Q_s.bias"),
    "Q_transport": ("reader.Q_transport.weight", "reader.Q_transport.bias"),
    "K_transport": ("reader.K_transport.weight",),
    "v_relative": ("reader.v_relative",),
    "continuation": ("reader.continuation.weight", "reader.continuation.bias"),
    "v_meta": ("reader.v_meta",),
}

R0_DISABLED: tuple[str, ...] = (
    "relation.encoder.Q_goal.weight",
    "relation.encoder.Q_goal.bias",
    "relation.encoder.K_goal.weight",
    "relation.encoder.V_goal.weight",
    "relation.encoder.V_goal.bias",
    "relation.reader.Q_read.weight",
    "relation.reader.Q_read.bias",
    "relation.reader.K_record.weight",
    "relation.reader.Q_value.weight",
    "relation.reader.Q_value.bias",
    "relation.reader.K_position.weight",
    "relation.reader.eta",
)

R1_TRAINABLE_BASE: tuple[str, ...] = (
    "relation.encoder.gru.weight_ih_l0",
    "relation.encoder.gru.weight_hh_l0",
    "relation.encoder.gru.bias_ih_l0",
    "relation.encoder.gru.bias_hh_l0",
    "relation.encoder.gru.weight_ih_l0_reverse",
    "relation.encoder.gru.weight_hh_l0_reverse",
    "relation.encoder.gru.bias_ih_l0_reverse",
    "relation.encoder.gru.bias_hh_l0_reverse",
    "relation.encoder.W_c.weight",
    "relation.encoder.W_c.bias",
    "relation.encoder.A_pool.weight",
    "relation.encoder.A_pool.bias",
    "relation.encoder.v_pool.weight",
    "relation.reader.null_key",
    "relation.reader.null_value",
    "relation.reader.V_position.weight",
    "relation.reader.V_position.bias",
)


def expected_new_parameter_count(
    key_width: int = DEFAULT_KEY, width: int = DEFAULT_WIDTH, *, reuse_null_key: bool = True
) -> int:
    total = (
        key_width * width
        + key_width * width
        + (key_width * (width + PREVIOUS_WIDTH) + key_width)
        + (key_width * (width + PREVIOUS_WIDTH) + key_width)
        + (key_width * (key_width + width) + key_width)
        + key_width * width
        + PE_WIDTH
        + (2 * width + PREVIOUS_WIDTH + 1)
        + META_WIDTH
    )
    if not reuse_null_key:
        total += key_width
    return total


def pe16(delta: torch.Tensor, width: int = PE_WIDTH) -> torch.Tensor:
    """Signed 16-dim sinusoidal relative position; no entity length encoded."""
    half = width // 2
    frequencies = torch.arange(half, dtype=delta.dtype, device=delta.device)
    frequencies = torch.exp(-math.log(10000.0) * frequencies / half)
    angles = delta.unsqueeze(-1) * frequencies
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


@dataclass(frozen=True)
class BindingCopyConfig:
    key_width: int = DEFAULT_KEY
    width: int = DEFAULT_WIDTH
    seed: int = DEFAULT_SEED

    def new_parameter_count(self) -> int:
        return expected_new_parameter_count(self.key_width, self.width)


@dataclass(frozen=True)
class BindingState:
    living: LivingState
    m: torch.Tensor
    coverage: torch.Tensor
    keys: tuple[tuple[str, int, int], ...]
    record_offsets: tuple[tuple[int, int], ...]
    prefix_cursor: int

    def payload(self) -> dict[str, Any]:
        return {
            "living": living_payload(self.living),
            "m": self.m.detach().cpu().clone(),
            "coverage": self.coverage.detach().cpu().clone(),
            "keys": [[source, version, offset] for source, version, offset in self.keys],
            "record_offsets": [list(pair) for pair in self.record_offsets],
            "prefix_cursor": self.prefix_cursor,
        }


@dataclass(frozen=True)
class BindingPrepared:
    base: Any
    state: BindingState


@dataclass(frozen=True)
class BindingRead:
    content: torch.Tensor
    copy_weights: torch.Tensor
    copy_byte_mass: torch.Tensor
    alpha: torch.Tensor
    betas: tuple[torch.Tensor, ...]
    alpha_null: torch.Tensor
    record_mass: torch.Tensor
    s_bytes: torch.Tensor
    q_bytes: torch.Tensor
    q_null: torch.Tensor
    m_mass: torch.Tensor
    continuation: torch.Tensor
    forced: bool
    pointer: tuple[torch.Tensor, ...] = ()
    pointer_logits: tuple[torch.Tensor, ...] = ()


@dataclass(frozen=True)
class BindingAnswerStep:
    probabilities: torch.Tensor
    generated: torch.Tensor
    read: BindingRead
    z: torch.Tensor
    gamma: torch.Tensor
    state: BindingState
    coverage: torch.Tensor
    gate_index: int
    accepted: int | None


class BindingCopyReader(nn.Module):
    """R1 span alignment, transport, and carried copy posterior."""

    def __init__(
        self,
        config: BindingCopyConfig,
        inherited: RecordRelationReader,
        *,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.config = config
        key_width, width = config.key_width, config.width
        generator = torch.Generator().manual_seed(config.seed)
        self.E_goal = nn.Linear(width, key_width, bias=False, dtype=dtype)
        self.E_source = nn.Linear(width, key_width, bias=False, dtype=dtype)
        self.Q_w = nn.Linear(width + PREVIOUS_WIDTH, key_width, dtype=dtype)
        self.Q_s = nn.Linear(width + PREVIOUS_WIDTH, key_width, dtype=dtype)
        self.Q_transport = nn.Linear(key_width + width, key_width, dtype=dtype)
        self.K_transport = nn.Linear(width, key_width, bias=False, dtype=dtype)
        self.v_relative = nn.Parameter(torch.zeros(PE_WIDTH, dtype=dtype))
        self.continuation = nn.Linear(2 * width + PREVIOUS_WIDTH, 1, dtype=dtype)
        self.v_meta = nn.Parameter(torch.zeros(META_WIDTH, dtype=dtype))

        for layer, cols in (
            (self.E_goal, width),
            (self.E_source, width),
            (self.Q_w, width + PREVIOUS_WIDTH),
            (self.Q_s, width + PREVIOUS_WIDTH),
            (self.Q_transport, key_width + width),
            (self.K_transport, width),
        ):
            std = 1.0 / math.sqrt(cols)
            with torch.no_grad():
                layer.weight.normal_(mean=0.0, std=std, generator=generator)
        for layer in (self.Q_w, self.Q_s, self.Q_transport):
            nn.init.zeros_(layer.bias)
        with torch.no_grad():
            self.continuation.weight.zero_()
            self.continuation.bias.fill_(-2.0)
        # The inherited reader supplies V_position/null_value; null_key is reused
        # only when its size matches the R1 key_width, otherwise a new null_key
        # parameter is created so the small config stays self-consistent.
        object.__setattr__(self, "inherited", inherited)
        if inherited.null_key.shape[0] == key_width:
            object.__setattr__(self, "reuse_null_key", True)
        else:
            self.null_key = nn.Parameter(torch.zeros(key_width, dtype=dtype))
            object.__setattr__(self, "reuse_null_key", False)

    def read(
        self,
        s: torch.Tensor,
        previous_embedding: torch.Tensor,
        goal_units: torch.Tensor,
        view: RecordSourceView,
        m: torch.Tensor,
        *,
        force: dict | None = None,
        lambda_override: torch.Tensor | float | None = None,
        pointer_bias: tuple[torch.Tensor, ...] | None = None,
        pointer_temperature: float = 1.0,
        pointer_hard: bool = False,
        m_override: torch.Tensor | None = None,
    ) -> BindingRead:
        key_width = self.config.key_width
        width = self.config.width
        query = torch.cat([s, previous_embedding])
        q_w = self.Q_w(query)
        q_s = self.Q_s(query)
        goal = goal_units if goal_units.shape[0] else self.E_goal.weight.new_zeros((1, width))
        g = self.E_goal(goal)
        logits_goal = q_w @ g.T / math.sqrt(key_width)
        w_p = torch.softmax(logits_goal, dim=-1)
        weighted_goal = w_p @ goal
        continuation = self.continuation(torch.cat([s, weighted_goal, previous_embedding])).squeeze(
            -1
        )
        if lambda_override is not None:
            continuation = torch.as_tensor(lambda_override, dtype=s.dtype, device=s.device)
        lam = torch.sigmoid(continuation)
        null_key = self.null_key if not self.reuse_null_key else self.inherited.null_key
        null_logit = q_s @ null_key / math.sqrt(key_width)

        if view.record_count == 0:
            alpha = torch.softmax(torch.stack([null_logit]), dim=-1)
            q_null = alpha[-1]
            empty = m.new_zeros(0)
            return BindingRead(
                content=q_null * self.inherited.null_value,
                copy_weights=empty,
                copy_byte_mass=m.new_zeros(MAX_COVERAGE),
                alpha=alpha,
                betas=(),
                alpha_null=q_null,
                record_mass=empty,
                s_bytes=empty,
                q_bytes=empty,
                q_null=q_null,
                m_mass=m.new_zeros(()),
                continuation=lam,
                forced=False,
            )

        source_keys = [self.E_source(view.units[i]) for i in range(view.record_count)]
        log_row = []
        for index in range(view.record_count):
            source = source_keys[index]
            match = g @ source.T / math.sqrt(key_width)
            l_ij = torch.logsumexp(torch.log(w_p).unsqueeze(-1) + match, dim=0)
            l_ij = l_ij + q_s @ source.T / math.sqrt(key_width) + self.v_meta @ view.metadata[index]
            log_row.append(l_ij)
        log_sums = torch.stack([torch.logsumexp(row, -1) - math.log(len(row)) for row in log_row])
        alpha = torch.softmax(torch.cat([log_sums, null_logit.reshape(1)]), dim=-1)
        betas = tuple(torch.softmax(row, -1) for row in log_row)

        if force is not None:
            alpha = force["alpha"].to(s.dtype)
            betas = tuple(row.to(s.dtype) for row in force["betas"])
        s_pieces = []
        pointer_rows: list[torch.Tensor] = []
        pointer_logit_rows: list[torch.Tensor] = []
        forced_pointer = force.get("pointer") if force is not None else None
        for index in range(view.record_count):
            anchor = alpha[index] * betas[index]
            bias = pointer_bias[index] if pointer_bias is not None else None
            if forced_pointer is not None:
                pointer = forced_pointer[index].to(s.dtype)
                row_logits = pointer
                s_piece = anchor.sum() * pointer
            else:
                s_piece, pointer, row_logits = self._point_transport(
                    anchor,
                    view.units[index],
                    q_w,
                    bias,
                    pointer_temperature,
                    pointer_hard,
                )
            s_pieces.append(s_piece)
            pointer_rows.append(pointer)
            pointer_logit_rows.append(row_logits)
        s_bytes = torch.cat(s_pieces)
        alpha_null = alpha[-1]
        if m_override is not None:
            m_bytes = m_override.to(s.dtype)
        else:
            m_bytes = m if m.numel() == s_bytes.numel() else s_bytes.new_zeros(s_bytes.shape)
        m_mass = m_bytes.sum()
        q_bytes = lam * m_bytes + (1 - lam * m_mass) * s_bytes
        q_null = (1 - lam * m_mass) * alpha_null
        q_total = torch.cat([q_bytes, q_null.reshape(1)])
        normalizer = q_total.sum()
        if float(normalizer.detach()) <= 0:
            raise RuntimeError("R1 Q has zero total mass")
        q_total = q_total / normalizer
        q_bytes = q_total[:-1]
        q_null = q_total[-1]
        units = torch.cat(view.units)
        content = q_bytes @ self.inherited.V_position(units) + q_null * self.inherited.null_value
        copy_byte_mass = q_bytes.new_zeros(MAX_COVERAGE).index_add_(0, view.byte_ids, q_bytes)
        record_mass = torch.stack(
            [q_bytes[view.slice(index)].sum() for index in range(view.record_count)]
        )
        return BindingRead(
            content=content,
            copy_weights=q_bytes,
            copy_byte_mass=copy_byte_mass,
            alpha=alpha,
            betas=betas,
            alpha_null=q_null,
            record_mass=record_mass,
            s_bytes=s_bytes,
            q_bytes=q_bytes,
            q_null=q_null,
            m_mass=m_mass,
            continuation=lam,
            forced=force is not None,
            pointer=tuple(pointer_rows),
            pointer_logits=tuple(pointer_logit_rows),
        )

    def _point_transport(
        self,
        anchor: torch.Tensor,
        units: torch.Tensor,
        q_w: torch.Tensor,
        pointer_bias: torch.Tensor | None,
        temperature: float,
        hard: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unit/hard point transport: one target position, not a soft row.

        The source position is the anchor's mode (betas), the target score is the
        learned transport row for that source, optionally biased by the span
        start head.  Training uses a temperature-annealed softmax; inference
        takes the argmax, so the copied position is a single point.
        """
        key_width = self.config.key_width
        length = units.shape[0]
        keys = self.K_transport(units)
        queries = self.Q_transport(torch.cat([q_w.expand(length, -1), units], dim=-1))
        relative = pe16(
            torch.arange(length, dtype=units.dtype, device=units.device).unsqueeze(0)
            - torch.arange(length, dtype=units.dtype, device=units.device).unsqueeze(1)
        )
        relative_score = relative @ self.v_relative
        logits = queries @ keys.T / math.sqrt(key_width) + relative_score
        if hard:
            source = int(torch.argmax(anchor))
            row_logits = logits[source]
        else:
            source_weights = torch.softmax(anchor / temperature, dim=-1)
            row_logits = (source_weights.unsqueeze(0) @ logits).squeeze(0)
        if pointer_bias is not None:
            row_logits = row_logits + pointer_bias
        mass = anchor.sum()
        if hard:
            index = int(torch.argmax(row_logits))
            pointer = torch.zeros(length, dtype=row_logits.dtype, device=row_logits.device)
            pointer[index] = 1.0
        else:
            pointer = torch.softmax(row_logits / temperature, dim=-1)
        return mass * pointer, pointer, row_logits


class BindingCopyCandidate(nn.Module):
    """Same subject plus the R1 reader; the base keeps the R0 path and V/null."""

    def __init__(
        self,
        base: RecordRelationCandidate,
        config: BindingCopyConfig,
        profile: str,
    ) -> None:
        super().__init__()
        # Keep the base object as a plain reference, but register its subject and
        # relation as this candidate's own submodules.  The parameter names stay
        # subject.* / relation.* for the inherited groups and reader.* for R1.
        object.__setattr__(self, "base", base)
        self.subject = base.subject
        self.relation = base.relation
        self.reader = BindingCopyReader(config, base.relation.reader)
        self.config = config
        self.profile = profile
        self.content_schema = BINDING_COPY_SCHEMA
        self.growth_gates = base.growth_gates
        self.arm = base.arm
        self.flat = base.flat
        self.configure_trainable()

    def configure_trainable(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        named = dict(self.named_parameters())
        for group in BASE_GROUPS.values():
            for name in group:
                if name in named:
                    named[name].requires_grad_(True)
        for name in R1_TRAINABLE_BASE:
            named[name].requires_grad_(True)
        for name in R0_DISABLED:
            named[name].requires_grad_(False)
        for group in R1_PARAMETER_GROUPS.values():
            for name in group:
                named[name].requires_grad_(True)
        if not self.reader.reuse_null_key and "reader.null_key" in named:
            named["reader.null_key"].requires_grad_(True)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def new_parameter_count(self) -> int:
        count = sum(parameter.numel() for parameter in self.reader.parameters())
        return count

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def group_parameters(self, group: str) -> list[nn.Parameter]:
        named = dict(self.named_parameters())
        if group == "null_key":
            if not self.reader.reuse_null_key and "reader.null_key" in named:
                return [named["reader.null_key"]]
            return []
        if group in R1_PARAMETER_GROUPS:
            names = R1_PARAMETER_GROUPS[group]
        elif group in BASE_GROUPS:
            names = BASE_GROUPS[group]
        else:
            raise KeyError(f"unknown parameter group {group!r}")
        return [named[name] for name in names if name in named]

    def trainable_group_parameters(self, group: str) -> list[nn.Parameter]:
        return [parameter for parameter in self.group_parameters(group) if parameter.requires_grad]

    def disabled_rows(self) -> dict[str, Any]:
        named = dict(self.named_parameters())
        names = [name for name in R0_DISABLED if name in named]
        return {
            "names": names,
            "elements": int(sum(named[name].numel() for name in names)),
            "groups": ["goal_attention", "old_record_reader"],
        }

    def reused_rows(self) -> dict[str, Any]:
        if self.reader.reuse_null_key:
            names = (
                "relation.reader.null_key",
                "relation.reader.null_value",
                "relation.reader.V_position.weight",
                "relation.reader.V_position.bias",
            )
        else:
            names = (
                "relation.reader.null_value",
                "relation.reader.V_position.weight",
                "relation.reader.V_position.bias",
            )
        named = dict(self.named_parameters())
        return {
            "names": list(names),
            "elements": int(sum(named[name].numel() for name in names)),
            "trainable": all(named[name].requires_grad for name in names),
        }


def build_binding_candidate(
    base: RecordRelationCandidate,
    *,
    config: BindingCopyConfig | None = None,
    profile: str = "R1-f64",
    dtype: torch.dtype = torch.float64,
) -> BindingCopyCandidate:
    candidate = BindingCopyCandidate(base, config or BindingCopyConfig(), profile)
    candidate.to(dtype)
    return candidate


def align_copy(
    value: torch.Tensor,
    old_keys: tuple[tuple[str, int, int], ...],
    new_keys: tuple[tuple[str, int, int], ...],
    reference: torch.Tensor,
) -> torch.Tensor:
    if not new_keys:
        return reference.new_zeros(0)
    old_index = {key: index for index, key in enumerate(old_keys)}
    zero = reference.new_zeros(())
    return torch.stack([value[old_index[key]] if key in old_index else zero for key in new_keys])


def initial_binding_state(
    candidate: BindingCopyCandidate, subject_id: str = "binding-copy-fixture"
) -> BindingState:
    living = candidate.subject.initial_state(subject_id, revision=CONTENT_REVISION)
    reference = candidate.subject.W
    return BindingState(
        living=living,
        m=reference.new_zeros(0),
        coverage=reference.new_zeros(0),
        keys=(),
        record_offsets=(),
        prefix_cursor=0,
    )


def prepare(
    candidate: BindingCopyCandidate,
    state: BindingState,
    goal: bytes,
    records: tuple,
    *,
    now: int = 0,
) -> BindingPrepared:
    from taiji.record_relation_core import prepare as base_prepare

    base_prepared = base_prepare(candidate.base, state.living, goal, records, now=now)
    keys = base_prepared.view.coverage_keys
    reference = candidate.subject.W
    m = align_copy(state.m, state.keys, keys, reference)
    coverage = align_copy(state.coverage, state.keys, keys, reference)
    new_state = BindingState(
        living=base_prepared.state,
        m=m,
        coverage=coverage,
        keys=keys,
        record_offsets=base_prepared.view.record_offsets,
        prefix_cursor=state.prefix_cursor,
    )
    return BindingPrepared(base=base_prepared, state=new_state)


def observe_goal(
    candidate: BindingCopyCandidate,
    prepared: BindingPrepared,
    *,
    gate_index: int = 0,
) -> tuple[BindingState, int]:
    subject = candidate.subject
    base = prepared.base
    state = prepared.state
    units = base.goal_units
    if units.shape[0] == 0:
        return state, gate_index
    mode_vector = _mode_vector(subject.W, 0)
    previous = subject.encoder.embedding(torch.tensor(EOS, device=subject.W.device))
    chunk = candidate.base.config.chunk
    for start in range(0, units.shape[0], chunk):
        end = min(start + chunk, units.shape[0])
        block = units[start:end]
        prefix = units[:end]
        r_prefix = candidate.base.relation.pool(prefix)
        event = subject.encoder.event(block.mean(0))
        xi = subject.xi(torch.cat([r_prefix, previous, mode_vector, base.nu]))
        s = subject.readout(state.living.h, xi)[0]
        read = candidate.reader.read(
            s,
            previous,
            prefix,
            base.view,
            state.m,
            lambda_override=0.0,
        )
        state = replace(
            state,
            living=replace(
                state.living,
                step=state.living.step + 1,
                h=advance_h(
                    subject,
                    state.living.h,
                    state.living.fast,
                    event,
                    base.goal_vector,
                    base.nu,
                    read.content,
                    _gate(candidate, gate_index),
                ),
                prior=subject.W.new_empty(0),
            ),
        )
        gate_index += 1
    return state, gate_index


def advance_copy(
    reader: BindingCopyReader,
    state: BindingState,
    read: BindingRead,
    probabilities: torch.Tensor,
    gamma: torch.Tensor,
    accepted: int,
    view: RecordSourceView,
) -> tuple[torch.Tensor, torch.Tensor]:
    if accepted < 0 or accepted >= MAX_COVERAGE:
        return state.m.new_zeros(state.m.shape), state.coverage
    p = probabilities[accepted]
    if accepted == EOS or float(p.detach()) <= 0:
        return state.m.new_zeros(state.m.shape), state.coverage
    raw_byte = reader.inherited.V_position.weight.new_tensor(accepted)
    mask = view.byte_ids == raw_byte
    r = (1 - gamma) * torch.where(mask, read.copy_weights / p, read.copy_weights.new_zeros(()))
    m_next = state.m.new_zeros(state.m.shape)
    for start, end in view.record_offsets:
        if end - start > 1:
            m_next[start + 1 : end] = r[start : end - 1]
    coverage = state.coverage + r
    return m_next, coverage


def answer_step(
    candidate: BindingCopyCandidate,
    prepared: BindingPrepared,
    state: BindingState,
    previous_byte: int,
    *,
    gate_index: int,
    accepted_byte: int | None = None,
    force: dict | None = None,
    gamma_override: torch.Tensor | float | None = None,
    lambda_override: torch.Tensor | float | None = None,
    m_override: torch.Tensor | None = None,
) -> BindingAnswerStep:
    subject = candidate.subject
    base = prepared.base
    previous = subject.encoder.embedding(torch.tensor(previous_byte, device=subject.W.device))
    mode_vector = _mode_vector(subject.W, 0)
    xi = subject.xi(torch.cat([base.r_goal, previous, mode_vector, base.nu]))
    s = subject.readout(state.living.h, xi)[0]
    read = candidate.reader.read(
        s,
        previous,
        base.goal_units,
        base.view,
        state.m,
        force=force,
        lambda_override=lambda_override,
        m_override=m_override,
    )
    if previous_byte == EOS:
        event = subject.W.new_zeros(subject.config.width)
    else:
        units = candidate.base.relation.encode_units(subject, bytes([previous_byte]))
        event = subject.encoder.event(units.mean(0))
    h = advance_h(
        subject,
        state.living.h,
        state.living.fast,
        event,
        base.goal_vector,
        base.nu,
        read.content,
        _gate(candidate, gate_index),
    )
    z = subject.readout(h, xi)[0]
    generated = torch.softmax(subject.expression.generate(torch.cat([z, previous])), dim=-1)
    gamma = torch.sigmoid(subject.expression.mix(torch.cat([z, read.content, previous])).squeeze(0))
    if gamma_override is not None:
        gamma = torch.as_tensor(gamma_override, dtype=gamma.dtype, device=gamma.device)
    probabilities = (gamma + (1 - gamma) * read.alpha_null) * generated + (
        1 - gamma
    ) * read.copy_byte_mass
    if accepted_byte is None:
        m_next = state.m
        coverage_next = state.coverage
    else:
        m_next, coverage_next = advance_copy(
            candidate.reader, state, read, probabilities, gamma, accepted_byte, base.view
        )
    living = replace(
        state.living,
        step=state.living.step + 1,
        h=h,
        prior=subject.W.new_empty(0),
    )
    child = BindingState(
        living=living,
        m=m_next,
        coverage=coverage_next,
        keys=state.keys,
        record_offsets=state.record_offsets,
        prefix_cursor=state.prefix_cursor + (0 if accepted_byte is None else 1),
    )
    return BindingAnswerStep(
        probabilities=probabilities,
        generated=generated,
        read=read,
        z=z,
        gamma=gamma,
        state=child,
        coverage=coverage_next,
        gate_index=gate_index + 1,
        accepted=accepted_byte,
    )


@dataclass
class BindingScoreResult:
    distributions: torch.Tensor
    nll: torch.Tensor
    observation: BindingState
    prefix: BindingState
    final: BindingState
    prepared: BindingPrepared
    gate_index: int


def score_example(
    candidate: BindingCopyCandidate,
    goal: bytes,
    records: tuple,
    prefix: tuple[int, ...],
    *,
    labels: tuple[int, ...] | None = None,
    subject_id: str = "binding-copy-fixture",
    now: int = 0,
) -> BindingScoreResult:
    labels = tuple(prefix) if labels is None else tuple(labels)
    if len(labels) != len(prefix):
        raise ValueError("The scoring labels must match the prefix length")
    state = initial_binding_state(candidate, subject_id)
    prepared = prepare(candidate, state, goal, records, now=now)
    state, gate_index = observe_goal(candidate, prepared)
    observation_state = state
    prefix_state = state
    distributions = []
    previous = EOS
    for position in range(len(prefix) + 1):
        if position == len(prefix):
            prefix_state = state
        accepted = prefix[position] if position < len(prefix) else None
        step = answer_step(
            candidate,
            prepared,
            state,
            previous,
            gate_index=gate_index,
            accepted_byte=accepted,
        )
        distributions.append(step.probabilities)
        state = step.state
        gate_index = step.gate_index
        if position < len(prefix):
            previous = prefix[position]
    stack = torch.stack(distributions)
    nll = _labels_nll(stack, labels + (EOS,))
    return BindingScoreResult(
        distributions=stack,
        nll=nll,
        observation=observation_state,
        prefix=prefix_state,
        final=state,
        prepared=prepared,
        gate_index=gate_index,
    )


def score_from_state(
    candidate: BindingCopyCandidate,
    prepared: BindingPrepared,
    state: BindingState,
    labels: tuple[int, ...],
    *,
    gate_index: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    distributions = []
    previous = EOS
    for position in range(len(labels) + 1):
        accepted = labels[position] if position < len(labels) else None
        step = answer_step(
            candidate,
            prepared,
            state,
            previous,
            gate_index=gate_index,
            accepted_byte=accepted,
        )
        distributions.append(step.probabilities)
        state = step.state
        gate_index = step.gate_index
        if position < len(labels):
            previous = labels[position]
    stack = torch.stack(distributions)
    return stack, _labels_nll(stack, tuple(labels) + (EOS,))


def generate_bytes(
    candidate: BindingCopyCandidate,
    goal: bytes,
    records: tuple,
    *,
    subject_id: str = "binding-copy-fixture",
    now: int = 0,
) -> dict:
    state = initial_binding_state(candidate, subject_id)
    prepared = prepare(candidate, state, goal, records, now=now)
    state, gate_index = observe_goal(candidate, prepared)
    previous = EOS
    raw: list[int] = []
    probabilities = []
    terminated = False
    for prediction in range(MAX_PREDICTIONS):
        step = answer_step(
            candidate,
            prepared,
            state,
            previous,
            gate_index=gate_index,
        )
        probabilities.append(step.probabilities)
        endpoint = decode_endpoint(step.probabilities)
        if endpoint["eos"]:
            terminated = True
            break
        raw.append(endpoint["byte"])
        if len(raw) > MAX_OUTPUT_BYTES:
            break
        # Update the copy posterior with the actually accepted generated byte.
        m_next, coverage_next = advance_copy(
            candidate.reader,
            state,
            step.read,
            step.probabilities,
            step.gamma,
            endpoint["byte"],
            prepared.base.view,
        )
        state = replace(
            step.state,
            m=m_next,
            coverage=coverage_next,
            prefix_cursor=step.state.prefix_cursor + 1,
        )
        gate_index = step.gate_index
        previous = endpoint["byte"]
        if prediction == MAX_PREDICTIONS - 1:
            break
    return {
        "raw_bytes": bytes(raw),
        "raw": raw,
        "terminated": terminated,
        "probabilities": torch.stack(probabilities),
        "state": state,
        "m": state.m,
        "coverage": state.coverage,
    }


def combined_loss(
    candidate: BindingCopyCandidate,
    fixtures: tuple[dict, ...],
    *,
    subject_id: str = "binding-copy-fixture",
) -> torch.Tensor:
    losses = []
    for fixture in fixtures:
        target = tuple(fixture["target_bytes"][:-1])
        result = score_example(
            candidate,
            fixture["goal"].encode(),
            fixture["records"],
            target,
            subject_id=subject_id,
        )
        losses.append(result.nll)
    return sum(losses) / len(losses)


def make_optimizer(candidate: BindingCopyCandidate) -> torch.optim.Adam:
    return torch.optim.Adam(
        candidate.trainable_parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )


def candidate_digest(candidate: BindingCopyCandidate) -> str:
    from taiji.candidate_manifest import parameter_digest

    return str(parameter_digest(candidate_parameter_map(candidate)))


def _all_groups() -> dict[str, tuple[str, ...]]:
    groups: dict[str, tuple[str, ...]] = {}
    for name, entries in BASE_GROUPS.items():
        groups[name] = entries
    for name, entries in R1_PARAMETER_GROUPS.items():
        groups[name] = entries
    groups["null_key"] = ("reader.null_key",)
    return groups


def optimizer_moment_rows(candidate: BindingCopyCandidate, optimizer) -> dict:
    rows = {}
    named = dict(candidate.named_parameters())
    for group, names in _all_groups().items():
        moments = []
        for name in names:
            parameter = named.get(name)
            if parameter is None:
                continue
            state = optimizer.state.get(parameter)
            if state is None:
                continue
            moments.append(
                {
                    "name": name,
                    "step": int(state["step"]),
                    "exp_avg_norm": float(state["exp_avg"].norm()),
                    "exp_avg_sq_norm": float(state["exp_avg_sq"].norm()),
                }
            )
        rows[group] = moments
    return rows


def group_delta(
    before: dict[str, torch.Tensor], candidate: BindingCopyCandidate, group: str
) -> dict:
    named = dict(candidate.named_parameters())
    dtype = next(candidate.parameters()).dtype
    finite = torch.finfo(dtype)
    groups = _all_groups()
    if group not in groups:
        raise KeyError(f"unknown parameter group {group!r}")
    rows = []
    for name in groups[group]:
        parameter = named.get(name)
        if parameter is None or name not in before:
            continue
        delta = (parameter.detach() - before[name]).abs()
        spacing = torch.where(
            parameter.detach().abs() > 0,
            parameter.detach().abs() * finite.eps * 2.0,
            torch.full_like(parameter.detach(), finite.tiny),
        )
        rows.append(
            {
                "name": name,
                "delta_max": float(delta.max()),
                "ulp_ratio_max": float((delta / spacing).max()),
                "dtype": _dtype_name(dtype),
            }
        )
    return {"group": group, "rows": rows}


def save_bundle(
    path: Path,
    *,
    profile: str,
    candidate: BindingCopyCandidate,
    state: BindingState,
    optimizer: torch.optim.Adam,
    update_index: int,
    cursor: dict,
    parent_identity: dict,
) -> dict:
    payload = {
        "schema": "taiji-binding-copy-bundle-v1",
        "step": "15.1.0b",
        "content_schema": BINDING_COPY_SCHEMA,
        "profile": profile,
        "dtype": str(next(candidate.parameters()).dtype).replace("torch.", ""),
        "relation_config": asdict(candidate.base.config),
        "binding_config": asdict(candidate.config),
        "base_arm": candidate.base.arm,
        "parent": parent_identity,
        "subject_config": dict(vars(candidate.subject.config)),
        "growth_parent_units": getattr(candidate.subject, "growth_parent_units", None),
        "growth_unit_branches": getattr(candidate.subject, "growth_unit_branches", []),
        "growth_gates": candidate.growth_gates,
        "parameters": candidate_parameter_map(candidate),
        "state": state.payload(),
        "optimizer": optimizer.state_dict(),
        "update_index": update_index,
        "cursor": dict(cursor),
        "rng_state": torch.get_rng_state().clone(),
    }
    torch.save(payload, path)
    return {
        "path": str(path),
        "bytes": Path(path).stat().st_size,
        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    }


def load_bundle(path: Path, dtype: torch.dtype) -> tuple:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    relation_config = RecordRelationConfig(**payload["relation_config"])
    binding_config = BindingCopyConfig(**payload["binding_config"])
    subject = build_subject_from_candidate_bundle(payload)
    base = RecordRelationCandidate(
        subject,
        relation_config,
        payload["base_arm"],
        list(payload["growth_gates"]),
    )
    base_named = dict(base.named_parameters())
    for name, value in payload["parameters"].items():
        if name.startswith("relation."):
            base_named[name].data.copy_(value)
    base.to(dtype)
    base.configure_trainable()
    candidate = BindingCopyCandidate(base, binding_config, payload["profile"])
    named = dict(candidate.named_parameters())
    parameters = payload["parameters"]
    if set(named) != set(parameters):
        raise ValueError("Bundle parameter names drifted from the R1 schema")
    for name, value in parameters.items():
        named[name].data.copy_(value)
    candidate.to(dtype)
    candidate.configure_trainable()
    optimizer = make_optimizer(candidate)
    optimizer.load_state_dict(payload["optimizer"])
    living = state_from_payload(payload["state"]["living"], candidate.subject.W)
    reference = candidate.subject.W
    state = BindingState(
        living=living,
        m=payload["state"]["m"].to(reference.dtype),
        coverage=payload["state"]["coverage"].to(reference.dtype),
        keys=tuple(
            (source, version, offset) for source, version, offset in payload["state"]["keys"]
        ),
        record_offsets=tuple(
            (int(start), int(end)) for start, end in payload["state"]["record_offsets"]
        ),
        prefix_cursor=int(payload["state"]["prefix_cursor"]),
    )
    torch.set_rng_state(payload["rng_state"])
    return (
        candidate,
        state,
        optimizer,
        payload["profile"],
        payload["update_index"],
        payload["cursor"],
        payload,
    )


def run_worker(spec: dict) -> dict:
    import time
    import traceback

    start_cpu, start_wall = time.process_time(), time.perf_counter()
    torch.set_num_threads(1)
    result_path = Path(spec["result"])
    try:
        dtype = {"float32": torch.float32, "float64": torch.float64}[spec["dtype"]]
        candidate, state, optimizer, profile, update_index, cursor, payload = load_bundle(
            Path(spec["bundle"]), dtype
        )
        del cursor, payload
        fixture = spec["fixtures"][spec["fixture_index"]]
        prepared = prepare(
            candidate,
            state,
            fixture["goal"].encode(),
            fixture["records"],
            now=0,
        )
        probe = answer_step(
            candidate,
            prepared,
            prepared.state,
            int(spec.get("previous_byte", EOS)),
            gate_index=int(spec.get("gate_index", 0)),
            accepted_byte=spec.get("accepted_byte"),
            gamma_override=spec.get("gamma_override"),
            lambda_override=spec.get("lambda_override"),
        )
        result = {
            "ok": True,
            "probabilities": probe.probabilities.detach(),
            "h": probe.state.living.h.detach(),
            "m": probe.state.m.detach(),
            "coverage": probe.state.coverage.detach(),
            "step": probe.state.living.step,
            "gamma": probe.gamma.detach(),
            "profile": profile,
            "update_index": update_index,
            "adam_calls": 0,
            "cpu_seconds": 0.0,
        }
        if spec["mode"] == "update":
            optimizer.zero_grad()
            fixtures = tuple(
                {
                    "goal": item["goal"],
                    "records": item["records"],
                    "target_bytes": tuple(item["target_bytes"]),
                }
                for item in spec["fixtures"]
            )
            loss = combined_loss(candidate, fixtures)
            loss.backward()
            clip = float(torch.nn.utils.clip_grad_norm_(candidate.trainable_parameters(), 1.0))
            optimizer.step()
            post = score_example(
                candidate,
                fixture["goal"].encode(),
                fixture["records"],
                tuple(fixture["target_bytes"][:-1]),
            )
            result.update(
                {
                    "parameters": candidate_parameter_map(candidate),
                    "optimizer": optimizer.state_dict(),
                    "loss": float(loss.detach()),
                    "clip_scale": clip,
                    "adam_calls": 1,
                }
            )
            del post
        result["cpu_seconds"] = time.process_time() - start_cpu
        result["wall_seconds"] = time.perf_counter() - start_wall
        torch.save(result, result_path)
        return {"ok": True, "result": str(result_path)}
    except Exception as exc:  # noqa: BLE001
        failure = {
            "ok": False,
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "cpu_seconds": time.process_time() - start_cpu,
            "wall_seconds": time.perf_counter() - start_wall,
        }
        torch.save(failure, result_path)
        return {"ok": False, "result": str(result_path)}


def forced_read(
    candidate: BindingCopyCandidate,
    prepared: BindingPrepared,
    state: BindingState,
    previous_byte: int,
    *,
    record_index: int | None,
    offset: int | None,
    gate_index: int = 0,
    gamma_override: torch.Tensor | float | None = None,
    lambda_override: torch.Tensor | float | None = None,
    force_position: bool = True,
) -> BindingAnswerStep:
    view = prepared.base.view
    dtype = next(candidate.parameters()).dtype
    if record_index is None:
        alpha = torch.zeros(view.record_count + 1, dtype=dtype)
        alpha[-1] = 1.0
        betas = tuple(torch.ones(view.lengths[i], dtype=dtype) for i in range(view.record_count))
        pointer = tuple(torch.zeros(view.lengths[i], dtype=dtype) for i in range(view.record_count))
        m_override = torch.zeros_like(state.m)
    else:
        alpha = torch.zeros(view.record_count + 1, dtype=dtype)
        alpha[record_index] = 1.0
        betas = tuple(torch.zeros(view.lengths[i], dtype=dtype) for i in range(view.record_count))
        betas[record_index][offset] = 1.0
        pointer = tuple(torch.zeros(view.lengths[i], dtype=dtype) for i in range(view.record_count))
        pointer[record_index][offset] = 1.0
        m_override = torch.zeros_like(state.m)
        m_override[view.slice(record_index).start + offset] = 1.0
    force = {"alpha": alpha, "betas": betas}
    if force_position:
        force["pointer"] = pointer
    return answer_step(
        candidate,
        prepared,
        state,
        previous_byte,
        gate_index=gate_index,
        force=force,
        gamma_override=gamma_override,
        lambda_override=lambda_override,
        m_override=m_override if force_position else None,
    )


def state_payload(state: BindingState) -> dict:
    return state.payload()
