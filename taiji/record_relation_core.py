"""14.4 content-interface replacement on the same recursive subject.

The subject keeps its R021 H/F graph, inherited byte encoder, inherited query
read, generate/mix heads and growth gate table.  This module routes the new
record-relation read into the ``recalled`` input slot and reuses the same
content distribution for the raw byte output.  The old episodic read, the old
copy-expression base attention and the continuation/prior mixing stay in the
bundle as inherited parameters but are not on this candidate path.

Everything here is engineering-only: no task training, no pricing, no claim
that the interface learns or that the subject can live across sessions.
"""

from __future__ import annotations

import hashlib
import time
import traceback
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn
from torch.nn import functional as F

from taiji.candidate_manifest import parameter_digest
from taiji.growth_credit import GatedGrowthCore
from taiji.living_core import (
    EOS,
    LivingCoreConfig,
    LivingCoreReference,
    LivingState,
    RawSource,
    state_normalize,
)
from taiji.record_relation import (
    RECORD_RELATION_SCHEMA,
    RecordRelationConfig,
    RecordRelationModule,
    RecordSourceView,
    advance_coverage,
    align_coverage,
)

ARM_FULL = "A_full"
ARM_LOCAL = "B_local"
ARM_FLAT = "C_flat"
ARMS = (ARM_FULL, ARM_LOCAL, ARM_FLAT)

LR = 1e-3
BETAS = (0.9, 0.999)
EPS = 1e-8
CLIP_NORM = 1.0
CONTENT_REVISION = 1
MAX_OUTPUT_BYTES = 64
MAX_PREDICTIONS = 65

SUBJECT_TRAINABLE_GROUPS: dict[str, tuple[str, ...]] = {
    "embedding": ("subject.encoder.embedding.weight",),
    "input": ("subject.encoder.input.weight", "subject.encoder.input.bias"),
    "convolutions": (
        "subject.encoder.convolutions.0.weight",
        "subject.encoder.convolutions.0.bias",
        "subject.encoder.convolutions.1.weight",
        "subject.encoder.convolutions.1.bias",
        "subject.encoder.convolutions.2.weight",
        "subject.encoder.convolutions.2.bias",
    ),
    "event": ("subject.encoder.event.weight", "subject.encoder.event.bias"),
    "xi": ("subject.xi.weight", "subject.xi.bias"),
    "query_read": (
        "subject.readout.query.0.weight",
        "subject.readout.query.1.weight",
        "subject.readout.keys.0.weight",
        "subject.readout.keys.1.weight",
        "subject.readout.keys.2.weight",
        "subject.readout.keys.3.weight",
        "subject.readout.keys.4.weight",
        "subject.readout.keys.5.weight",
        "subject.readout.keys.6.weight",
        "subject.readout.keys.7.weight",
        "subject.readout.values.0.weight",
        "subject.readout.values.1.weight",
        "subject.readout.values.2.weight",
        "subject.readout.values.3.weight",
        "subject.readout.values.4.weight",
        "subject.readout.values.5.weight",
        "subject.readout.values.6.weight",
        "subject.readout.values.7.weight",
        "subject.readout.output.weight",
    ),
    "W": ("subject.W",),
    "P": ("subject.P",),
    "R": ("subject.R",),
    "generate": (
        "subject.expression.generate.weight",
        "subject.expression.generate.bias",
    ),
    "mix": ("subject.expression.mix.weight", "subject.expression.mix.bias"),
}

NEW_PARAMETER_GROUPS: dict[str, tuple[str, ...]] = {
    "gru_forward": (
        "relation.encoder.gru.weight_ih_l0",
        "relation.encoder.gru.weight_hh_l0",
        "relation.encoder.gru.bias_ih_l0",
        "relation.encoder.gru.bias_hh_l0",
    ),
    "gru_backward": (
        "relation.encoder.gru.weight_ih_l0_reverse",
        "relation.encoder.gru.weight_hh_l0_reverse",
        "relation.encoder.gru.bias_ih_l0_reverse",
        "relation.encoder.gru.bias_hh_l0_reverse",
    ),
    "W_c": ("relation.encoder.W_c.weight", "relation.encoder.W_c.bias"),
    "pool": (
        "relation.encoder.A_pool.weight",
        "relation.encoder.A_pool.bias",
        "relation.encoder.v_pool.weight",
    ),
    "Q_goal": ("relation.encoder.Q_goal.weight", "relation.encoder.Q_goal.bias"),
    "K_goal": ("relation.encoder.K_goal.weight",),
    "V_goal": ("relation.encoder.V_goal.weight", "relation.encoder.V_goal.bias"),
    "Q_read": ("relation.reader.Q_read.weight", "relation.reader.Q_read.bias"),
    "K_record": ("relation.reader.K_record.weight",),
    "null_record": ("relation.reader.null_key", "relation.reader.null_value"),
    "Q_value": ("relation.reader.Q_value.weight", "relation.reader.Q_value.bias"),
    "K_position": ("relation.reader.K_position.weight",),
    "V_position": ("relation.reader.V_position.weight", "relation.reader.V_position.bias"),
    "eta": ("relation.reader.eta",),
}

PARAMETER_GROUPS: dict[str, tuple[str, ...]] = {
    **SUBJECT_TRAINABLE_GROUPS,
    **NEW_PARAMETER_GROUPS,
}

DISABLED_BY_ARM: dict[str, tuple[str, ...]] = {
    ARM_FULL: (),
    ARM_LOCAL: ("gru_forward", "gru_backward", "W_c"),
    ARM_FLAT: ("K_record", "Q_value"),
}

DISABLED_NAMES_BY_ARM: dict[str, tuple[str, ...]] = {
    ARM_FULL: (),
    ARM_LOCAL: (
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
    ),
    ARM_FLAT: (
        "relation.reader.K_record.weight",
        "relation.reader.Q_value.weight",
        "relation.reader.Q_value.bias",
        "relation.reader.null_value",
    ),
}


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _dtype_from_name(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def build_subject(parent: dict) -> nn.Module:
    """Rebuild the inherited subject from the parent payload, growth rows included."""
    config = LivingCoreConfig(
        **{
            key: value
            for key, value in parent["config"].items()
            if key in LivingCoreConfig.__dataclass_fields__
        }
    )
    grown = parent["growth_parent_units"] is not None
    subject: nn.Module = GatedGrowthCore(config) if grown else LivingCoreReference(config)
    subject.to(torch.float64)
    if grown:
        subject.growth_parent_units = parent["growth_parent_units"]
        subject.growth_unit_branches = parent["growth_unit_branches"]
        subject.R = nn.Parameter(parent["parameters"]["R"].clone())
        subject.P = nn.Parameter(parent["parameters"]["P"].clone())
    rest = {
        name: value
        for name, value in parent["parameters"].items()
        if name not in ("R", "P") or not grown
    }
    named = dict(subject.named_parameters())
    for name, value in rest.items():
        named[name].data.copy_(value)
    return subject


class RecordRelationCandidate(nn.Module):
    """The same subject with the registered ``record_relation_v1`` subgraph."""

    # The inherited subject is a plain ``nn.Module`` whose tensor attributes
    # (``W``, ``encoder``, ``readout``, ...) are resolved dynamically, so every
    # read through it is an unverifiable ``Tensor | Module`` union.
    subject: Any

    def __init__(
        self,
        subject: nn.Module,
        config: RecordRelationConfig,
        arm: str,
        growth_gates: list,
    ) -> None:
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}")
        self.subject = subject
        self.relation = RecordRelationModule(config)
        self.arm = arm
        self.config = config
        self.growth_gates = growth_gates
        cast(Any, self.relation.encoder).local = arm == ARM_LOCAL
        self.flat = arm == ARM_FLAT
        self.content_schema = RECORD_RELATION_SCHEMA
        self.configure_trainable()

    def configure_trainable(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        named = dict(self.named_parameters())
        for names in SUBJECT_TRAINABLE_GROUPS.values():
            for name in names:
                named[name].requires_grad_(True)
        for names in NEW_PARAMETER_GROUPS.values():
            for name in names:
                named[name].requires_grad_(True)
        for name in DISABLED_NAMES_BY_ARM[self.arm]:
            named[name].requires_grad_(False)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def new_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.relation.parameters())

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def group_parameters(self, group: str) -> list[nn.Parameter]:
        if group not in PARAMETER_GROUPS:
            raise KeyError(f"unknown parameter group {group!r}")
        named = dict(self.named_parameters())
        return [named[name] for name in PARAMETER_GROUPS[group]]

    def trainable_group_parameters(self, group: str) -> list[nn.Parameter]:
        return [parameter for parameter in self.group_parameters(group) if parameter.requires_grad]

    def disabled_rows(self) -> dict[str, Any]:
        named = dict(self.named_parameters())
        names = DISABLED_NAMES_BY_ARM[self.arm]
        return {
            "names": list(names),
            "elements": int(sum(named[name].numel() for name in names)),
            "groups": list(DISABLED_BY_ARM[self.arm]),
        }


def build_candidate(
    parent: dict,
    *,
    arm: str = ARM_FULL,
    config: RecordRelationConfig | None = None,
    dtype: torch.dtype = torch.float64,
) -> RecordRelationCandidate:
    subject = build_subject(parent)
    candidate = RecordRelationCandidate(
        subject,
        config or RecordRelationConfig(),
        arm,
        list(parent["growth_gates"]),
    )
    candidate.to(dtype)
    return candidate


@dataclass(frozen=True)
class PreparedSample:
    state: LivingState
    view: RecordSourceView
    goal_units: torch.Tensor
    r_goal: torch.Tensor
    goal_vector: torch.Tensor
    nu: torch.Tensor
    coverage: torch.Tensor
    goal_bytes: bytes


@dataclass(frozen=True)
class AnswerStep:
    probabilities: torch.Tensor
    generated: torch.Tensor
    read: Any
    z: torch.Tensor
    gamma: torch.Tensor
    state: LivingState
    coverage: torch.Tensor
    gate_index: int


@dataclass
class ScoreResult:
    distributions: torch.Tensor
    nll: torch.Tensor
    observation_state: LivingState
    observation_coverage: torch.Tensor
    prefix_state: LivingState
    prefix_coverage: torch.Tensor
    final_state: LivingState
    prepared: PreparedSample
    gate_index: int


def _mode_vector(reference: torch.Tensor, mode: int) -> torch.Tensor:
    return F.one_hot(torch.tensor(mode, device=reference.device), 7).to(reference.dtype)


def _gate(candidate: RecordRelationCandidate, index: int) -> dict:
    table = candidate.growth_gates
    if not table:
        return {}
    item = table[index % len(table)]
    if isinstance(item, dict):
        return dict(item)
    return {"copy": item[0], "generate": item[1]}


def prepare(
    candidate: RecordRelationCandidate,
    state: LivingState,
    goal: bytes,
    records: tuple[RawSource, ...],
    *,
    now: int = 0,
) -> PreparedSample:
    subject = candidate.subject
    view = candidate.relation.build_view(subject, records, now=now)
    goal_units = candidate.relation.encode_units(subject, goal)
    r_goal = candidate.relation.pool(goal_units)
    goal_vector = subject.encoder.goal_vector(goal)
    zero = subject.W.new_zeros(8)
    unknown = torch.zeros(8, dtype=torch.bool, device=subject.W.device)
    ema, nu, _ = subject.update_nu(state, zero, unknown)
    coverage = align_coverage(state.coverage, cast(Any, state.keys), view)
    prepared_state = replace(
        state,
        keys=cast(Any, view.coverage_keys),
        coverage=coverage,
        nu_ema=ema,
        prior=subject.W.new_empty(0),
    )
    return PreparedSample(
        state=prepared_state,
        view=view,
        goal_units=goal_units,
        r_goal=r_goal,
        goal_vector=goal_vector,
        nu=nu,
        coverage=coverage,
        goal_bytes=goal,
    )


def advance_h(
    subject,
    h: torch.Tensor,
    fast: torch.Tensor,
    event: torch.Tensor,
    goal_vector: torch.Tensor,
    nu: torch.Tensor,
    content: torch.Tensor,
    growth_gates: dict,
) -> torch.Tensor:
    """R021 H update with the new content in the recalled slot."""
    message = subject.messages(h, goal_vector)[0]
    fast_read = subject.read_fast(h, fast, event, goal_vector, nu)[0]
    x = torch.cat(
        [
            state_normalize(h),
            event.expand_as(h),
            message,
            fast_read,
            content.expand_as(h),
            goal_vector.expand_as(h),
            nu.expand_as(h),
        ],
        -1,
    )
    effective = subject.W + subject.U @ subject.V.transpose(1, 2)
    pre = torch.einsum("ndp,np->nd", effective, x)
    parent_units = getattr(subject, "growth_parent_units", None)
    if parent_units is None:
        parent_units = subject.R.shape[1]
    pre = pre + torch.einsum(
        "ndm,nm->nd",
        subject.P[:, :, :parent_units],
        torch.tanh(torch.einsum("nmp,np->nm", subject.R[:, :parent_units, :], x)),
    )
    if parent_units < subject.R.shape[1]:
        if growth_gates is None:
            raise ValueError("Gated newborn units require per-step branch gates")
        activated = torch.tanh(torch.einsum("nmp,np->nm", subject.R[:, parent_units:, :], x))
        new_units = subject.R.shape[1] - parent_units
        rows = []
        for group in range(subject.config.groups):
            branches = subject.growth_unit_branches[group]
            values = [growth_gates[branch] for branch in branches]
            values.extend(0.0 for _ in range(new_units - len(branches)))
            rows.append(torch.tensor(values, dtype=x.dtype, device=x.device))
        activated = activated * torch.stack(rows, dim=0)
        pre = pre + torch.einsum("ndm,nm->nd", subject.P[:, :, parent_units:], activated)
    tau = 0.5 + 127.5 * torch.sigmoid(subject.tau_logit)
    leak = (1 - torch.exp(-1 / tau))[:, None]
    return (1 - leak) * h + leak * torch.tanh(pre)


def observe_goal(
    candidate: RecordRelationCandidate,
    prepared: PreparedSample,
    *,
    gate_index: int = 0,
) -> tuple[LivingState, int]:
    """Consume the fully observed goal as 8-byte events through the same read."""
    subject = candidate.subject
    state = prepared.state
    units = prepared.goal_units
    coverage = prepared.coverage
    if units.shape[0] == 0:
        return state, gate_index
    mode_vector = _mode_vector(subject.W, 0)
    previous = subject.encoder.embedding(torch.tensor(EOS, device=subject.W.device))
    chunk = candidate.config.chunk
    for start in range(0, units.shape[0], chunk):
        end = min(start + chunk, units.shape[0])
        block = units[start:end]
        prefix = units[:end]
        r_prefix = candidate.relation.pool(prefix)
        event = subject.encoder.event(block.mean(0))
        xi = subject.xi(torch.cat([r_prefix, previous, mode_vector, prepared.nu]))
        s = subject.readout(state.h, xi)[0]
        read = candidate.relation.read(
            s, previous, prefix, prepared.view, coverage, flat=candidate.flat
        )
        state = replace(
            state,
            step=state.step + 1,
            h=advance_h(
                subject,
                state.h,
                state.fast,
                event,
                prepared.goal_vector,
                prepared.nu,
                read.content,
                _gate(candidate, gate_index),
            ),
        )
        gate_index += 1
    return state, gate_index


def answer_step(
    candidate: RecordRelationCandidate,
    prepared: PreparedSample,
    state: LivingState,
    coverage: torch.Tensor,
    previous_byte: int,
    *,
    gate_index: int,
    force: dict | None = None,
    gamma_override: torch.Tensor | float | None = None,
) -> AnswerStep:
    subject = candidate.subject
    previous = subject.encoder.embedding(torch.tensor(previous_byte, device=subject.W.device))
    mode_vector = _mode_vector(subject.W, 0)
    xi = subject.xi(torch.cat([prepared.r_goal, previous, mode_vector, prepared.nu]))
    s = subject.readout(state.h, xi)[0]
    read = candidate.relation.read(
        s,
        previous,
        prepared.goal_units,
        prepared.view,
        coverage,
        flat=candidate.flat,
        force=force,
    )
    if previous_byte == EOS:
        event = subject.W.new_zeros(subject.config.width)
    else:
        units = candidate.relation.encode_units(subject, bytes([previous_byte]))
        event = subject.encoder.event(units.mean(0))
    h = advance_h(
        subject,
        state.h,
        state.fast,
        event,
        prepared.goal_vector,
        prepared.nu,
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
    coverage_new = advance_coverage(coverage, read.copy_weights)
    child = replace(
        state,
        step=state.step + 1,
        h=h,
        coverage=coverage_new,
        prior=subject.W.new_empty(0),
    )
    return AnswerStep(
        probabilities=probabilities,
        generated=generated,
        read=read,
        z=z,
        gamma=gamma,
        state=child,
        coverage=coverage_new,
        gate_index=gate_index + 1,
    )


def _labels_nll(distributions: torch.Tensor, labels: tuple[int, ...]) -> torch.Tensor:
    if distributions.shape[0] != len(labels):
        raise ValueError("One distribution is required for every scoring label")
    targets = torch.tensor(labels, dtype=torch.long, device=distributions.device)
    rows = torch.arange(len(labels), device=distributions.device)
    picked = distributions[rows, targets]
    return -torch.mean(torch.log(picked + 1e-30))


def score_example(
    candidate: RecordRelationCandidate,
    goal: bytes,
    records: tuple[RawSource, ...],
    prefix: tuple[int, ...],
    *,
    labels: tuple[int, ...] | None = None,
    subject_id: str = "record-relation-fixture",
    now: int = 0,
) -> ScoreResult:
    """Teacher-forced scoring: the prefix drives the query, labels only score."""
    labels = tuple(prefix) if labels is None else tuple(labels)
    if len(labels) != len(prefix):
        raise ValueError("The scoring labels must match the prefix length")
    subject = candidate.subject
    state = subject.initial_state(subject_id, revision=CONTENT_REVISION)
    prepared = prepare(candidate, state, goal, records, now=now)
    observation_state, gate_index = observe_goal(candidate, prepared, gate_index=0)
    observation_coverage = prepared.coverage
    state = observation_state
    coverage = observation_coverage
    distributions = []
    prefix_state = state
    prefix_coverage = coverage
    previous = EOS
    for position in range(len(prefix) + 1):
        if position == len(prefix):
            prefix_state = state
            prefix_coverage = coverage
        step = answer_step(
            candidate,
            prepared,
            state,
            coverage,
            previous,
            gate_index=gate_index,
        )
        distributions.append(step.probabilities)
        state = step.state
        coverage = step.coverage
        gate_index = step.gate_index
        if position < len(prefix):
            previous = prefix[position]
    stack = torch.stack(distributions)
    nll = _labels_nll(stack, labels + (EOS,))
    return ScoreResult(
        distributions=stack,
        nll=nll,
        observation_state=observation_state,
        observation_coverage=observation_coverage,
        prefix_state=prefix_state,
        prefix_coverage=prefix_coverage,
        final_state=state,
        prepared=prepared,
        gate_index=gate_index,
    )


def score_from_state(
    candidate: RecordRelationCandidate,
    prepared: PreparedSample,
    state: LivingState,
    coverage: torch.Tensor,
    labels: tuple[int, ...],
    *,
    gate_index: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    distributions = []
    previous = EOS
    for position in range(len(labels) + 1):
        step = answer_step(candidate, prepared, state, coverage, previous, gate_index=gate_index)
        distributions.append(step.probabilities)
        state = step.state
        coverage = step.coverage
        gate_index = step.gate_index
        if position < len(labels):
            previous = labels[position]
    stack = torch.stack(distributions)
    return stack, _labels_nll(stack, tuple(labels) + (EOS,))


def _utf8_state(prefix: bytes) -> tuple[bool, int, int, int]:
    """Return ``(valid, need, lo, hi)`` for the next continuation byte."""
    need, lo, hi, valid = 0, 0x80, 0xBF, True
    for value in prefix:
        if need:
            if not lo <= value <= hi:
                valid = False
                break
            need -= 1
            lo, hi = 0x80, 0xBF
        elif value < 0x80:
            continue
        elif 0xC2 <= value <= 0xDF:
            need = 1
        elif value == 0xE0:
            need, lo = 2, 0xA0
        elif 0xE1 <= value <= 0xEC:
            need = 2
        elif value == 0xED:
            need, hi = 2, 0x9F
        elif 0xEE <= value <= 0xEF:
            need = 2
        elif value == 0xF0:
            need, lo = 3, 0x90
        elif 0xF1 <= value <= 0xF3:
            need = 3
        elif value == 0xF4:
            need, hi = 3, 0x8F
        else:
            valid = False
            break
    return valid, need, lo, hi


def utf8_allowed_mask(prefix: bytes, total: int = 257) -> tuple[torch.Tensor, bool]:
    """Legal next-byte mask over the 257-symbol endpoint vocabulary.

    An invalid prefix cannot be repaired by one byte, so the mask opens up (all
    bytes legal) while reporting ``prefix_valid=False``; a legal prefix mid
    multi-byte character admits only the continuation range and excludes EOS.
    """
    mask = torch.zeros(total, dtype=torch.bool)
    valid, need, lo, hi = _utf8_state(prefix)
    if not valid:
        mask[:] = True
        return mask, False
    if need:
        mask[lo : hi + 1] = True
        return mask, True
    mask[0:0x80] = True
    mask[0xC2:0xF5] = True
    mask[EOS] = True
    return mask, True


def decode_endpoint(probabilities: torch.Tensor, *, prefix: bytes | None = None) -> dict:
    """Output endpoint.

    ``prefix=None`` keeps the frozen pure-argmax reading (no repair, no mask).
    With ``prefix`` the decoder applies the UTF-8 legality mask before argmax, so
    an illegal byte is never emitted; ``pre_mask_byte`` and ``masked`` report the
    unmasked choice so a probe can tell generation-path illegality from
    copy-path illegality.
    """
    pre_byte = int(torch.argmax(probabilities))
    byte = pre_byte
    masked = False
    prefix_valid = True
    if prefix is not None:
        allowed, prefix_valid = utf8_allowed_mask(prefix, int(probabilities.shape[0]))
        masked = not bool(allowed[pre_byte])
        if masked:
            legal = probabilities.masked_fill(~allowed, float("-inf"))
            byte = int(torch.argmax(legal))
    return {
        "byte": byte,
        "eos": byte == EOS,
        "probabilities": probabilities.detach().clone(),
        "pre_mask_byte": pre_byte,
        "masked": masked,
        "prefix_valid": prefix_valid,
    }


def generate_bytes(
    candidate: RecordRelationCandidate,
    goal: bytes,
    records: tuple[RawSource, ...],
    *,
    subject_id: str = "record-relation-fixture",
    now: int = 0,
) -> dict:
    subject = candidate.subject
    state = subject.initial_state(subject_id, revision=CONTENT_REVISION)
    prepared = prepare(candidate, state, goal, records, now=now)
    state, gate_index = observe_goal(candidate, prepared, gate_index=0)
    coverage = prepared.coverage
    previous = EOS
    raw: list[int] = []
    probabilities = []
    terminated = False
    for prediction in range(MAX_PREDICTIONS):
        step = answer_step(candidate, prepared, state, coverage, previous, gate_index=gate_index)
        probabilities.append(step.probabilities)
        endpoint = decode_endpoint(step.probabilities, prefix=bytes(raw))
        if endpoint["eos"]:
            terminated = True
            break
        raw.append(endpoint["byte"])
        if len(raw) > MAX_OUTPUT_BYTES:
            break
        state = step.state
        coverage = step.coverage
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
        "coverage": coverage,
    }


def combined_loss(
    candidate: RecordRelationCandidate,
    fixtures: tuple[dict, ...],
    *,
    subject_id: str = "record-relation-fixture",
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
    return cast(torch.Tensor, sum(losses) / len(losses))


def make_optimizer(candidate: RecordRelationCandidate) -> torch.optim.Adam:
    return torch.optim.Adam(
        candidate.trainable_parameters(),
        lr=LR,
        betas=BETAS,
        eps=EPS,
        weight_decay=0.0,
    )


def optimizer_moment_rows(candidate: RecordRelationCandidate, optimizer) -> dict:
    rows = {}
    named = dict(candidate.named_parameters())
    for group, names in PARAMETER_GROUPS.items():
        moments = []
        for name in names:
            parameter = named[name]
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
    before: dict[str, torch.Tensor], candidate: RecordRelationCandidate, group: str
) -> dict:
    named = dict(candidate.named_parameters())
    dtype = next(candidate.parameters()).dtype
    finite = torch.finfo(dtype)
    rows = []
    for name in PARAMETER_GROUPS[group]:
        delta = (named[name].detach() - before[name]).abs()
        spacing = torch.where(
            named[name].detach().abs() > 0,
            named[name].detach().abs() * finite.eps * 2.0,
            torch.full_like(named[name].detach(), finite.tiny),
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


def candidate_parameter_map(candidate: RecordRelationCandidate) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone() for name, parameter in candidate.named_parameters()
    }


def candidate_digest(candidate: RecordRelationCandidate) -> str:
    return str(parameter_digest(candidate_parameter_map(candidate)))


def state_payload(state: LivingState) -> dict:
    return {
        "subject_id": state.subject_id,
        "revision": state.revision,
        "step": state.step,
        "h": state.h.detach().cpu().clone(),
        "fast": state.fast.detach().cpu().clone(),
        "nu_ema": state.nu_ema.detach().cpu().clone(),
        "keys": [[source, version, offset] for source, version, offset in cast(Any, state.keys)],
        "coverage": state.coverage.detach().cpu().clone(),
        "prior": state.prior.detach().cpu().clone(),
    }


def state_from_payload(payload: dict, reference: torch.Tensor) -> LivingState:
    return LivingState(
        subject_id=payload["subject_id"],
        revision=payload["revision"],
        step=payload["step"],
        h=payload["h"].to(reference.dtype),
        fast=payload["fast"].to(reference.dtype),
        nu_ema=payload["nu_ema"].to(reference.dtype),
        keys=tuple(
            cast(Any, (source, version, offset)) for source, version, offset in payload["keys"]
        ),
        coverage=payload["coverage"].to(reference.dtype),
        prior=payload["prior"].to(reference.dtype),
    )


def save_bundle(
    path: Path,
    *,
    profile: str,
    candidate: RecordRelationCandidate,
    state: LivingState,
    optimizer: torch.optim.Adam,
    update_index: int,
    cursor: dict,
    parent_identity: dict,
) -> dict:
    payload = {
        "schema": "taiji-record-relation-bundle-v1",
        "step": "14.4",
        "content_schema": RECORD_RELATION_SCHEMA,
        "profile": profile,
        "arm": candidate.arm,
        "dtype": _dtype_name(next(candidate.parameters()).dtype),
        "relation_config": asdict(candidate.config),
        "parent": parent_identity,
        "subject_config": dict(vars(candidate.subject.config)),
        "growth_parent_units": getattr(candidate.subject, "growth_parent_units", None),
        "growth_unit_branches": getattr(candidate.subject, "growth_unit_branches", []),
        "growth_gates": candidate.growth_gates,
        "parameters": candidate_parameter_map(candidate),
        "state": state_payload(state),
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


def build_subject_from_candidate_bundle(payload: dict) -> nn.Module:
    """Rebuild the inherited subject directly from the candidate parameter map."""
    parameters = payload["parameters"]
    config = LivingCoreConfig(**payload["subject_config"])
    grown = payload["growth_parent_units"] is not None
    subject: nn.Module = GatedGrowthCore(config) if grown else LivingCoreReference(config)
    subject.to(torch.float64)
    if grown:
        subject.growth_parent_units = payload["growth_parent_units"]
        subject.growth_unit_branches = payload["growth_unit_branches"]
        subject.R = nn.Parameter(parameters["subject.R"].clone())
        subject.P = nn.Parameter(parameters["subject.P"].clone())
    for name, parameter in subject.named_parameters():
        key = f"subject.{name}"
        if grown and name in ("R", "P"):
            continue
        parameter.data.copy_(parameters[key])
    return subject


def load_bundle(path: Path, dtype: torch.dtype) -> tuple:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = RecordRelationConfig(**payload["relation_config"])
    subject = build_subject_from_candidate_bundle(payload)
    candidate = RecordRelationCandidate(
        subject, config, payload["arm"], list(payload["growth_gates"])
    )
    named = dict(candidate.named_parameters())
    parameters = payload["parameters"]
    if set(named) != set(parameters):
        raise ValueError("Bundle parameter names drifted from the candidate schema")
    for name, value in parameters.items():
        if name.startswith("relation."):
            named[name].data.copy_(value)
    candidate.to(dtype)
    candidate.configure_trainable()
    optimizer = make_optimizer(candidate)
    optimizer.load_state_dict(payload["optimizer"])
    reference = candidate.subject.W
    state = state_from_payload(payload["state"], reference)
    torch.set_rng_state(payload["rng_state"])
    return (
        candidate,
        state,
        optimizer,
        payload["arm"],
        payload["profile"],
        payload["update_index"],
        payload["cursor"],
        payload,
    )


def _forces_from_spec(spec: dict | None) -> dict | None:
    if not spec:
        return None
    if "flat" in spec:
        return {"flat": torch.tensor(spec["flat"], dtype=torch.float64)}
    return {
        "alpha": torch.tensor(spec["alpha"], dtype=torch.float64),
        "betas": tuple(torch.tensor(row, dtype=torch.float64) for row in spec["betas"]),
    }


def run_worker(spec: dict) -> dict:
    """Independent-process restore: one probe, or one declared Adam update."""
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    torch.set_num_threads(1)
    result_path = Path(spec["result"])
    try:
        dtype = _dtype_from_name(spec["dtype"])
        candidate, state, optimizer, arm, profile, update_index, cursor, payload = load_bundle(
            Path(spec["bundle"]), dtype
        )
        _ = (arm, profile, update_index, cursor, payload)
        fixture = spec["fixtures"][spec["fixture_index"]]
        prepared = prepare(
            candidate,
            state,
            fixture["goal"].encode(),
            fixture["records"],
            now=0,
        )
        force = _forces_from_spec(spec.get("force"))
        if force is not None:
            target_dtype = next(candidate.parameters()).dtype
            force = (
                {"flat": force["flat"].to(target_dtype)}
                if "flat" in force
                else {
                    "alpha": force["alpha"].to(target_dtype),
                    "betas": tuple(row.to(target_dtype) for row in force["betas"]),
                }
            )
        probe = answer_step(
            candidate,
            prepared,
            state,
            state.coverage,
            int(spec["previous_byte"]),
            gate_index=int(spec.get("gate_index", 0)),
            force=force,
            gamma_override=spec.get("gamma_override"),
        )
        result = {
            "ok": True,
            "probabilities": probe.probabilities.detach(),
            "h": probe.state.h.detach(),
            "coverage": probe.coverage.detach(),
            "step": probe.state.step,
            "gamma": probe.gamma.detach(),
            "arm": candidate.arm,
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
            clip = float(
                torch.nn.utils.clip_grad_norm_(candidate.trainable_parameters(), CLIP_NORM)
            )
            optimizer.step()
            post = score_example(
                candidate,
                fixture["goal"].encode(),
                fixture["records"],
                tuple(fixture["target_bytes"][:-1]),
            )
            post_probe = answer_step(
                candidate,
                post.prepared,
                post.prefix_state,
                post.prefix_coverage,
                int(fixture["target_bytes"][-2]),
                gate_index=0,
            )
            result.update(
                {
                    "parameters": candidate_parameter_map(candidate),
                    "optimizer": optimizer.state_dict(),
                    "loss": float(loss.detach()),
                    "clip_scale": clip,
                    "post_probabilities": post_probe.probabilities.detach(),
                    "post_h": post_probe.state.h.detach(),
                    "adam_calls": 1,
                }
            )
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
    candidate: RecordRelationCandidate,
    prepared: PreparedSample,
    state: LivingState,
    coverage: torch.Tensor,
    previous_byte: int,
    *,
    record_index: int | None,
    offset: int | None,
    gate_index: int = 0,
    gamma_override: torch.Tensor | float | None = None,
) -> AnswerStep:
    """Formal intervention entry: one-hot alpha/beta, same read and output formula."""
    view = prepared.view
    dtype = next(candidate.parameters()).dtype
    if record_index is None:
        alpha = torch.zeros(view.record_count + 1, dtype=dtype)
        alpha[-1] = 1.0
        betas = tuple(torch.ones(view.lengths[i], dtype=dtype) for i in range(view.record_count))
    else:
        alpha = torch.zeros(view.record_count + 1, dtype=dtype)
        alpha[record_index] = 1.0
        betas = tuple(torch.zeros(view.lengths[i], dtype=dtype) for i in range(view.record_count))
        betas[record_index][offset] = 1.0
    return answer_step(
        candidate,
        prepared,
        state,
        coverage,
        previous_byte,
        gate_index=gate_index,
        force={"alpha": alpha, "betas": betas},
        gamma_override=gamma_override,
    )
