"""17.0 selection-routing step: ordered shared encoding, locked copy, gamma mix.

The 17.0 graph keeps the inherited subject, record encoder, within-record
transport, H/F update and generate/mix head.  It replaces the flat mean-pooled
selection input with an *ordered* pooled encoding, locks the copy posterior to
the route chosen at this step, and applies the gamma floor before the mix.

Only the independent selection head (37,120 parameters) and the optional
teacher-span pointer (258 or 4,194 parameters) are new.  No experience surface,
no external retrieval, no tokenizer, no self-evolution API.

Arm A keeps a straight-through path from the hard route into the soft route
(route gradient).  Arm B detaches the hard route from the task loss.  Arm C
uses the inherited R1 unlocked read (no route lock).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from taiji.binding_copy import (
    MAX_COVERAGE,
    BindingRead,
    advance_copy,
    observe_goal,
)
from taiji.living_core import EOS
from taiji.record_relation_core import (
    MAX_PREDICTIONS,
    _gate,
    _mode_vector,
    advance_h,
    decode_endpoint,
)

ROUTING_SCHEMA = "taiji-selection-routing-17-v1"
DEFAULT_WIDTH = 128
DEFAULT_PREV_WIDTH = 64
DEFAULT_HIDDEN = 64
GAMMA_FLOOR = 0.2
ROUTE_NULL = "null"
ARMS = ("A_route_lock", "B_stop_lock", "C_unlocked")


@dataclass(frozen=True)
class RoutingConfig:
    width: int = DEFAULT_WIDTH
    prev_width: int = DEFAULT_PREV_WIDTH
    hidden: int = DEFAULT_HIDDEN
    gamma_floor: float = GAMMA_FLOOR
    span_bottleneck: int = 0


DEFAULT_ROUTING_CONFIG = RoutingConfig()


def expected_head_parameter_count(config: RoutingConfig = DEFAULT_ROUTING_CONFIG) -> int:
    """37,120 for width 128 / prev 64 / hidden 64."""
    return (
        config.hidden * (4 * config.width + config.prev_width)
        + config.hidden
        + config.hidden
        + config.width
    )


def expected_span_parameter_count(config: RoutingConfig = DEFAULT_ROUTING_CONFIG) -> int:
    """258 without bottleneck; 4,194 with a 32-d projected bottleneck."""
    if config.span_bottleneck:
        return (
            config.width * config.span_bottleneck
            + config.span_bottleneck
            + 2 * (config.span_bottleneck + 1)
        )
    return 2 * (config.width + 1)


class OrderedSelectionHead(nn.Module):
    """Independent per-step route head over ordered pooled encodings."""

    def __init__(self, config: RoutingConfig = DEFAULT_ROUTING_CONFIG, *, seed: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.config = config
        self.W_hidden = nn.Linear(
            4 * config.width + config.prev_width, config.hidden, dtype=torch.float64
        )
        self.w_out = nn.Linear(config.hidden, 1, bias=False, dtype=torch.float64)
        self.k_null = nn.Parameter(torch.zeros(config.width, dtype=torch.float64))
        bound = 1.0 / float(config.hidden) ** 0.5
        with torch.no_grad():
            self.W_hidden.weight.uniform_(-bound, bound, generator=generator)
            self.W_hidden.bias.zero_()
            self.w_out.weight.uniform_(-bound, bound, generator=generator)
            self.k_null.uniform_(-bound, bound, generator=generator)

    def pooled(self, relation: Any, subject: Any, raw: bytes) -> torch.Tensor:
        return relation.pool(relation.encode_units(subject, raw))

    def score_rows(
        self,
        goal_vec: torch.Tensor,
        key_vecs: list[torch.Tensor],
        s_t: torch.Tensor,
        previous_embedding: torch.Tensor,
    ) -> torch.Tensor:
        rows = []
        for key in key_vecs:
            rows.append(torch.cat([goal_vec, key, s_t, previous_embedding, goal_vec * key]))
        null_key = self.k_null
        rows.append(torch.cat([goal_vec, null_key, s_t, previous_embedding, goal_vec * null_key]))
        stacked = torch.stack(rows)
        hidden = torch.tanh(self.W_hidden(stacked))
        return self.w_out(hidden).squeeze(-1)


class SpanPointerHead(nn.Module):
    """Teacher-span start/end pointer over the selected record's ordered units."""

    def __init__(self, config: RoutingConfig = DEFAULT_ROUTING_CONFIG, *, seed: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.config = config
        if config.span_bottleneck:
            self.proj = nn.Linear(config.width, config.span_bottleneck, dtype=torch.float64)
            self.start = nn.Linear(config.span_bottleneck, 1, dtype=torch.float64)
            self.end = nn.Linear(config.span_bottleneck, 1, dtype=torch.float64)
        else:
            self.proj = None
            self.start = nn.Linear(config.width, 1, dtype=torch.float64)
            self.end = nn.Linear(config.width, 1, dtype=torch.float64)
        bound = 1.0 / float(config.width) ** 0.5
        with torch.no_grad():
            if self.proj is not None:
                self.proj.weight.uniform_(-bound, bound, generator=generator)
                self.proj.bias.zero_()
            for layer in (self.start, self.end):
                layer.weight.uniform_(-bound, bound, generator=generator)
                layer.bias.zero_()

    def logits(self, units: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = units if self.proj is None else torch.tanh(self.proj(units))
        return self.start(hidden).squeeze(-1), self.end(hidden).squeeze(-1)


def span_loss(
    head: SpanPointerHead,
    units: torch.Tensor,
    start_index: int,
    end_index: int,
) -> dict[str, torch.Tensor]:
    start_logits, end_logits = head.logits(units)
    loss = F.cross_entropy(
        start_logits.unsqueeze(0), torch.tensor([start_index], dtype=torch.long)
    ) + F.cross_entropy(end_logits.unsqueeze(0), torch.tensor([end_index], dtype=torch.long))
    return {
        "loss": loss,
        "start_logits": start_logits,
        "end_logits": end_logits,
    }


def teacher_span(raw: bytes, entity: bytes) -> tuple[int, int]:
    """Byte span of an entity inside a frozen record; never a model output."""
    start = raw.find(entity)
    if start < 0:
        raise ValueError("entity not found in frozen record")
    return start, start + len(entity) - 1


@dataclass
class RoutingContext:
    goal_vec: torch.Tensor
    key_vecs: list[torch.Tensor]
    record_count: int


def build_context(candidate: Any, goal: bytes, keys: tuple[str, ...]) -> RoutingContext:
    relation = candidate.base.relation
    subject = candidate.subject
    goal_vec = relation.pool(relation.encode_units(subject, goal))
    key_vecs = [relation.pool(relation.encode_units(subject, key.encode("utf-8"))) for key in keys]
    return RoutingContext(goal_vec=goal_vec, key_vecs=key_vecs, record_count=len(keys))


@dataclass
class RoutingStep:
    probabilities: torch.Tensor
    generated: torch.Tensor
    gamma_raw: torch.Tensor
    gamma_eff: torch.Tensor
    read: BindingRead
    q_bytes: torch.Tensor
    q_null: torch.Tensor
    a_soft: torch.Tensor
    a_hard: torch.Tensor
    a_train: torch.Tensor
    scores: torch.Tensor
    content: torch.Tensor
    state: Any
    next_h: torch.Tensor
    gate_index: int
    fallback_records: tuple[int, ...] = ()
    pointer: tuple[torch.Tensor, ...] = ()
    pointer_logits: tuple[torch.Tensor, ...] = ()


def _s_t(candidate: Any, prepared: Any, state: Any, previous: int) -> torch.Tensor:
    subject = candidate.subject
    base = prepared.base
    previous_embedding = subject.encoder.embedding(torch.tensor(previous, device=subject.W.device))
    xi = subject.xi(
        torch.cat([base.r_goal, previous_embedding, _mode_vector(subject.W, 0), base.nu])
    )
    return subject.readout(state.living.h, xi)[0]


def _hard_route(scores: torch.Tensor) -> torch.Tensor:
    index = int(torch.argmax(scores))
    null_index = len(scores) - 1
    if scores[null_index] - scores[index] >= -1e-12:
        index = null_index
    return F.one_hot(torch.tensor([index], dtype=torch.long), num_classes=len(scores)).to(
        scores.dtype
    )[0]


def _locked_quantities(
    read: BindingRead,
    view: Any,
    a_soft: torch.Tensor,
    a_hard: torch.Tensor,
    arm: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...]]:
    if arm == "C_unlocked":
        return read.q_bytes, read.q_null, read.alpha, ()
    if arm == "A_route_lock":
        a_train = a_hard + (a_soft - a_soft.detach())
    elif arm == "B_stop_lock":
        a_train = a_hard.detach()
    else:
        raise ValueError(f"unknown arm {arm}")
    pieces = []
    fallback: list[int] = []
    for index in range(view.record_count):
        sl = view.slice(index)
        mass = read.q_bytes[sl].sum()
        if float(mass.detach()) > 1e-12:
            eta = read.q_bytes[sl] / mass
        else:
            eta = read.betas[index]
            fallback.append(index)
        pieces.append(a_train[index] * eta)
    q_bytes = torch.cat(pieces) if pieces else read.q_bytes[:0]
    q_null = a_train[-1]
    return q_bytes, q_null, a_train, tuple(fallback)


def routing_step(
    candidate: Any,
    prepared: Any,
    state: Any,
    previous: int,
    context: RoutingContext,
    head: OrderedSelectionHead | None,
    *,
    arm: str = "A_route_lock",
    gate_index: int = 0,
    span_head: SpanPointerHead | None = None,
    pointer_temperature: float = 1.0,
    pointer_hard: bool = False,
    forced_pointer_bias: tuple[torch.Tensor, ...] | None = None,
) -> RoutingStep:
    subject = candidate.subject
    base = prepared.base
    view = base.view
    previous_embedding = subject.encoder.embedding(torch.tensor(previous, device=subject.W.device))
    s_t = _s_t(candidate, prepared, state, previous)
    if head is None:
        # Head-ablation probe: no route choice, a uniform prior over the records
        # and null.  The position pointer stays active so the step still copies.
        records = view.record_count
        scores = torch.zeros(
            records + 1, dtype=context.goal_vec.dtype, device=context.goal_vec.device
        )
        a_soft = torch.softmax(scores, dim=0)
        a_hard = a_soft
    else:
        scores = head.score_rows(context.goal_vec, context.key_vecs, s_t, previous_embedding)
        a_soft = torch.softmax(scores, dim=0)
        a_hard = _hard_route(scores)
    pointer_bias = None
    if span_head is not None and view.record_count:
        pointer_bias = tuple(
            span_head.start(view.units[index]).squeeze(-1) for index in range(view.record_count)
        )
    if forced_pointer_bias is not None:
        pointer_bias = forced_pointer_bias
    read = candidate.reader.read(
        s_t,
        previous_embedding,
        base.goal_units,
        view,
        state.m,
        pointer_bias=pointer_bias,
        pointer_temperature=pointer_temperature,
        pointer_hard=pointer_hard,
    )
    q_bytes, q_null, a_train, fallback = _locked_quantities(read, view, a_soft, a_hard, arm)
    inherited = candidate.reader.inherited
    units = torch.cat(view.units) if view.record_count else torch.zeros(0, dtype=torch.float64)
    content = (
        q_bytes @ inherited.V_position(units) + q_null * inherited.null_value
        if len(q_bytes)
        else q_null * inherited.null_value
    )
    event = (
        torch.zeros_like(content)
        if previous == EOS
        else subject.encoder.event(
            candidate.base.relation.encode_units(subject, bytes([previous])).mean(0)
        )
    )
    next_h = advance_h(
        subject,
        state.living.h,
        state.living.fast,
        event,
        base.r_goal,
        base.nu,
        content,
        _gate(candidate, gate_index),
    )
    xi = subject.xi(
        torch.cat([base.r_goal, previous_embedding, _mode_vector(subject.W, 0), base.nu])
    )
    z = subject.readout(next_h, xi)[0]
    generated = torch.softmax(
        subject.expression.generate(torch.cat([z, previous_embedding])), dim=0
    )
    gamma_raw = torch.sigmoid(subject.expression.mix(torch.cat([z, content, previous_embedding])))
    floor = head.config.gamma_floor if head is not None else DEFAULT_ROUTING_CONFIG.gamma_floor
    gamma_eff = floor + (1.0 - floor) * gamma_raw
    copy_byte_mass = torch.zeros(MAX_COVERAGE, dtype=torch.float64)
    if len(q_bytes):
        copy_byte_mass = copy_byte_mass.index_add(0, view.byte_ids, q_bytes)
    probabilities = (gamma_eff + (1.0 - gamma_eff) * q_null) * generated + (
        1.0 - gamma_eff
    ) * copy_byte_mass
    return RoutingStep(
        probabilities=probabilities,
        generated=generated,
        gamma_raw=gamma_raw,
        gamma_eff=gamma_eff,
        read=read,
        q_bytes=q_bytes,
        q_null=q_null,
        a_soft=a_soft,
        a_hard=a_hard,
        a_train=a_train,
        scores=scores,
        content=content,
        state=state,
        next_h=next_h,
        gate_index=gate_index,
        fallback_records=fallback,
        pointer=read.pointer,
        pointer_logits=read.pointer_logits,
    )


def _advanced_state(
    candidate: Any, prepared: Any, step: RoutingStep, accepted: int
) -> tuple[Any, Any, int]:
    from dataclasses import replace

    copy_byte_mass = torch.zeros(MAX_COVERAGE, dtype=torch.float64)
    if len(step.q_bytes):
        copy_byte_mass = copy_byte_mass.index_add(0, prepared.base.view.byte_ids, step.q_bytes)
    locked = BindingRead(
        content=step.content,
        copy_weights=step.q_bytes,
        copy_byte_mass=copy_byte_mass,
        alpha=step.a_train,
        betas=step.read.betas,
        alpha_null=step.q_null,
        record_mass=step.read.record_mass,
        s_bytes=step.read.s_bytes,
        q_bytes=step.q_bytes,
        q_null=step.q_null,
        m_mass=step.read.m_mass,
        continuation=step.read.continuation,
        forced=False,
    )
    next_m, next_coverage = advance_copy(
        candidate.reader,
        step.state,
        locked,
        step.probabilities,
        step.gamma_eff,
        accepted,
        prepared.base.view,
    )
    living = replace(step.state.living, h=step.next_h)
    next_state = replace(step.state, living=living, m=next_m, coverage=next_coverage)
    next_gate = (step.gate_index + 1) % 4
    return next_state, next_coverage, next_gate


def resolve_prefix(prefix: tuple[int, ...], length: int) -> tuple[int, ...]:
    if len(prefix) >= length:
        return tuple(prefix[:length])
    return tuple(prefix) + (EOS,) * (length - len(prefix))


def routing_loss(
    candidate: Any,
    prepared: Any,
    prefix: tuple[int, ...],
    target_bytes: tuple[int, ...],
    route_index: tuple[int, ...],
    route_stage: tuple[int, ...],
    context: RoutingContext,
    head: OrderedSelectionHead,
    *,
    arm: str = "A_route_lock",
    collect: bool = False,
    span_head: SpanPointerHead | None = None,
    pointer_temperature: float = 1.0,
    pointer_hard: bool = False,
    pointer_entropy_coef: float = 0.0,
) -> dict[str, Any]:
    """Teacher-forced loss.  Only ``prefix[t]`` enters the step at time t."""
    state = prepared.state
    state, gate_index = observe_goal(candidate, prepared, gate_index=0)
    input_prefix = resolve_prefix(prefix, len(target_bytes) - 1)
    nll_terms: list[torch.Tensor] = []
    ce_terms: dict[int, list[torch.Tensor]] = {0: [], 1: [], 2: []}
    entropy_terms: list[torch.Tensor] = []
    distributions: list[torch.Tensor] = []
    routes: list[torch.Tensor] = []
    previous = EOS
    for t, label in enumerate(target_bytes):
        step = routing_step(
            candidate,
            prepared,
            state,
            previous,
            context,
            head,
            arm=arm,
            gate_index=gate_index,
            span_head=span_head,
            pointer_temperature=pointer_temperature,
            pointer_hard=pointer_hard,
        )
        nll_terms.append(-torch.log(step.probabilities[label] + 1e-30))
        route_label = route_index[t]
        if route_label >= 0:
            stage = route_stage[t]
            ce_terms[stage].append(
                F.cross_entropy(step.scores.unsqueeze(0), torch.tensor([route_label]))
            )
            if pointer_entropy_coef and stage in (0, 1):
                chosen = int(torch.argmax(step.scores))
                if chosen < len(step.pointer_logits):
                    probs = torch.softmax(step.pointer_logits[chosen] / pointer_temperature, dim=-1)
                    entropy_terms.append(-(probs * torch.log(probs + 1e-30)).sum())
        if collect:
            distributions.append(step.probabilities.detach().clone())
            routes.append(step.scores.detach().clone())
        if t == len(target_bytes) - 1:
            break
        accepted = input_prefix[t]
        state, _, gate_index = _advanced_state(candidate, prepared, step, accepted)
        previous = accepted
    nll = torch.stack(nll_terms).mean()
    stage_means = {
        stage: (torch.stack(values).mean() if values else None)
        for stage, values in ce_terms.items()
    }
    present = [value for value in stage_means.values() if value is not None]
    ce = torch.stack(present).mean() if present else torch.zeros((), dtype=torch.float64)
    entropy = (
        torch.stack(entropy_terms).mean() if entropy_terms else torch.zeros((), dtype=torch.float64)
    )
    result = {"nll": nll, "ce": ce, "stage_ce": stage_means, "entropy": entropy}
    if collect:
        result["distributions"] = distributions
        result["routes"] = routes
    return result


def selection_ce(
    candidate: Any,
    prepared: Any,
    prefix: tuple[int, ...],
    target_bytes: tuple[int, ...],
    route_index: tuple[int, ...],
    route_stage: tuple[int, ...],
    context: RoutingContext,
    head: OrderedSelectionHead,
) -> dict[str, Any]:
    """Selection-only loss with the shared graph treated as frozen forward."""
    state = prepared.state
    state, gate_index = observe_goal(candidate, prepared, gate_index=0)
    input_prefix = resolve_prefix(prefix, len(target_bytes) - 1)
    terms: dict[int, list[torch.Tensor]] = {0: [], 1: [], 2: []}
    goal_vec = context.goal_vec.detach()
    key_vecs = [key.detach() for key in context.key_vecs]
    previous = EOS
    for t, _label in enumerate(target_bytes):
        with torch.no_grad():
            step = routing_step(
                candidate,
                prepared,
                state,
                previous,
                context,
                head,
                arm="B_stop_lock",
                gate_index=gate_index,
            )
            s_t = _s_t(candidate, prepared, state, previous)
            previous_embedding = candidate.subject.encoder.embedding(
                torch.tensor(previous, device=candidate.subject.W.device)
            )
        scores = head.score_rows(goal_vec, key_vecs, s_t, previous_embedding)
        route_label = route_index[t]
        if route_label >= 0:
            terms[route_stage[t]].append(
                F.cross_entropy(scores.unsqueeze(0), torch.tensor([route_label]))
            )
        if t == len(target_bytes) - 1:
            break
        accepted = input_prefix[t]
        with torch.no_grad():
            state, _, gate_index = _advanced_state(candidate, prepared, step, accepted)
            previous = accepted
    present = [torch.stack(values).mean() for values in terms.values() if values]
    loss = torch.stack(present).mean() if present else torch.zeros((), dtype=torch.float64)
    return {"loss": loss, "stage_ce": terms}


def routing_generate(
    candidate: Any,
    prepared: Any,
    context: RoutingContext,
    head: OrderedSelectionHead | None,
    *,
    arm: str = "A_route_lock",
    span_head: SpanPointerHead | None = None,
    pointer_temperature: float = 1.0,
    pointer_hard: bool = False,
    forced_pointer_bias: tuple[torch.Tensor, ...] | None = None,
) -> dict[str, Any]:
    state = prepared.state
    state, gate_index = observe_goal(candidate, prepared, gate_index=0)
    previous = EOS
    raw: list[int] = []
    probabilities: list[torch.Tensor] = []
    gammas: list[float] = []
    terminated = False
    with torch.no_grad():
        for _prediction in range(MAX_PREDICTIONS):
            step = routing_step(
                candidate,
                prepared,
                state,
                previous,
                context,
                head,
                arm=arm,
                gate_index=gate_index,
                span_head=span_head,
                pointer_temperature=pointer_temperature,
                pointer_hard=pointer_hard,
                forced_pointer_bias=forced_pointer_bias,
            )
            probabilities.append(step.probabilities.detach().clone())
            gammas.append(float(step.gamma_eff.detach()))
            endpoint = decode_endpoint(step.probabilities)
            if endpoint["eos"]:
                terminated = True
                break
            raw.append(endpoint["byte"])
            if len(raw) > 64:
                break
            state, _, gate_index = _advanced_state(candidate, prepared, step, endpoint["byte"])
            previous = endpoint["byte"]
    return {
        "raw_bytes": bytes(raw),
        "raw": raw,
        "terminated": terminated,
        "probabilities": probabilities,
        "gamma": gammas,
    }


def route_labels(fixture: dict[str, Any]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Route index and stage per target position, derived from frozen data only.

    Stage 0 = object bytes [3,15), stage 1 = place bytes [21,27), stage 2 =
    template/null.  Route index is the record index, or ``record_count`` for
    null.  These labels never enter the forward pass.
    """
    record_count = len(fixture["record_keys"])
    object_index = int(fixture["selection_label"])
    place_index = fixture["record_keys"].index(fixture["place_description"])
    route_index = []
    route_stage = []
    for position in range(len(fixture["target_bytes"])):
        if 3 <= position < 15:
            route_index.append(object_index)
            route_stage.append(0)
        elif 21 <= position < 27:
            route_index.append(place_index)
            route_stage.append(1)
        else:
            route_index.append(record_count)
            route_stage.append(2)
    return tuple(route_index), tuple(route_stage)


def load_thresholds(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    for key in ("theta_repr", "theta_score", "noise_floor_repr", "noise_floor_score"):
        if key not in data:
            raise ValueError(f"threshold file missing {key}")
    return data
