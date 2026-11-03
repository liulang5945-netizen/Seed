#!/usr/bin/env python
"""N16.1 selection head and selection-locked decoding.

The head sees only the goal text and each record's description fragment. The
decoder replaces the record-level alpha with the head's one-hot choice while
keeping the model's own byte-level betas, and floors the copy gate gamma.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch import nn

from taiji.binding_copy import (
    BindingCopyCandidate,
    BindingCopyConfig,
    BindingState,
    advance_copy,
    answer_step,
    build_binding_candidate,
    initial_binding_state,
    observe_goal,
    prepare,
)
from taiji.candidate_manifest import parameter_digest
from taiji.living_core import EOS
from taiji.record_relation_core import (
    RecordRelationCandidate,
    build_subject_from_candidate_bundle,
    candidate_parameter_map,
    decode_endpoint,
    state_from_payload,
)

GAMMA_FLOOR = 0.2
SELECTION_SCHEMA = "taiji-binding-select-bundle-v1"
MAX_RECORDS = 12
BYTE_VOCAB = 256
HIDDEN = 64
EMBED = 32


@dataclass(frozen=True)
class SelectionConfig:
    byte_vocab: int = BYTE_VOCAB
    embed: int = EMBED
    hidden: int = HIDDEN
    max_records: int = MAX_RECORDS


def expected_selection_parameter_count(config: SelectionConfig | None = None) -> int:
    c = config or SelectionConfig()
    return (
        c.byte_vocab * c.embed
        + (c.embed * c.hidden + c.hidden)
        + (c.embed * c.hidden + c.hidden)
        + c.max_records * c.hidden
        + (3 * c.hidden * c.hidden + c.hidden)
        + (c.hidden + 1)
        + 1
    )


class SelectionHead(nn.Module):
    """Score each record description fragment against the goal."""

    def __init__(self, config: SelectionConfig | None = None) -> None:
        super().__init__()
        self.config = config or SelectionConfig()
        c = self.config
        self.byte_embed = nn.Embedding(c.byte_vocab, c.embed)
        self.goal_proj = nn.Linear(c.embed, c.hidden)
        self.key_proj = nn.Linear(c.embed, c.hidden)
        self.pos_embed = nn.Embedding(c.max_records, c.hidden)
        self.scorer1 = nn.Linear(3 * c.hidden, c.hidden)
        self.scorer2 = nn.Linear(c.hidden, 1)
        self.null_bias = nn.Parameter(torch.zeros(1))

    def encode(self, text: bytes) -> torch.Tensor:
        ids = torch.tensor(list(text), dtype=torch.long)
        return self.byte_embed(ids).mean(0)

    def scores(self, goal: bytes, keys: list[str]) -> torch.Tensor:
        goal_vec = self.goal_proj(self.encode(goal))
        rows = []
        for index, key in enumerate(keys[: self.config.max_records]):
            key_vec = self.key_proj(self.encode(key.encode("utf-8")))
            pos = self.pos_embed(torch.tensor(index, dtype=torch.long))
            rows.append(torch.cat([goal_vec, key_vec, pos]))
        stacked = torch.stack(rows)
        return self.scorer2(torch.tanh(self.scorer1(stacked))).squeeze(-1)

    def forward(self, goal: bytes, keys: list[str]) -> torch.Tensor:
        return self.scores(goal, keys)


class SelectionCandidate:
    """Binding-copy candidate plus a selection head."""

    def __init__(self, binding: BindingCopyCandidate, head: SelectionHead, profile: str) -> None:
        self.binding = binding
        self.head = head
        self.profile = profile

    def parameters(self):
        yield from self.binding.parameters()
        yield from self.head.parameters()

    def trainable_parameters(self):
        yield from self.binding.trainable_parameters()
        yield from self.head.parameters()

    def to(self, dtype):
        self.binding.to(dtype)
        self.head.to(dtype)
        return self

    def configure_trainable(self) -> None:
        self.binding.configure_trainable()
        for parameter in self.head.parameters():
            parameter.requires_grad_(True)


def build_selection_candidate(
    base: RecordRelationCandidate,
    *,
    profile: str,
    dtype: torch.dtype,
    binding_config: BindingCopyConfig | None = None,
    selection_config: SelectionConfig | None = None,
) -> SelectionCandidate:
    binding = build_binding_candidate(
        base, config=binding_config or BindingCopyConfig(), profile=profile, dtype=dtype
    )
    head = SelectionHead(selection_config).to(dtype)
    return SelectionCandidate(binding, head, profile)


def make_selection_optimizer(candidate: SelectionCandidate) -> torch.optim.Adam:
    return torch.optim.Adam(
        list(candidate.trainable_parameters()),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )


def select(
    candidate: SelectionCandidate,
    goal: bytes,
    keys: list[str],
    *,
    label: int | None = None,
) -> dict:
    scores = candidate.head(goal, keys)
    probabilities = torch.softmax(torch.cat([scores, candidate.head.null_bias]), dim=-1)
    chosen = int(torch.argmax(scores).item())
    out: dict[str, Any] = {
        "chosen": chosen,
        "confidence": float(probabilities[chosen].detach()),
        "null_mass": float(probabilities[-1].detach()),
        "scores": [float(value.detach()) for value in scores],
    }
    if label is not None:
        out["correct"] = chosen == label
    return out


def locked_step(
    candidate: SelectionCandidate,
    prepared,
    state: BindingState,
    previous: int,
    *,
    gate_index: int,
    chosen_record: int,
    gamma_floor: float = GAMMA_FLOOR,
) -> Any:
    """One decoding step with alpha locked to the selected record."""
    base = candidate.binding
    probe = answer_step(base, prepared, state, previous, gate_index=gate_index)
    view = prepared.base.view
    alpha = torch.zeros(view.record_count + 1, dtype=probe.read.alpha.dtype)
    alpha[max(0, min(chosen_record, view.record_count - 1))] = 1.0
    force = {"alpha": alpha, "betas": probe.read.betas}
    gamma = torch.clamp(probe.gamma, min=gamma_floor)
    return answer_step(
        base,
        prepared,
        state,
        previous,
        gate_index=gate_index,
        force=force,
        gamma_override=gamma,
    )


def locked_generate(
    candidate: SelectionCandidate,
    goal: bytes,
    records: tuple,
    keys: list[str],
    *,
    subject_id: str,
    gamma_floor: float = GAMMA_FLOOR,
    max_predictions: int = 65,
) -> dict:
    base = candidate.binding
    choice = select(candidate, goal, keys)
    state = initial_binding_state(base, subject_id)
    prepared = prepare(base, state, goal, records)
    state, gate_index = observe_goal(base, prepared)
    previous = EOS
    raw: list[int] = []
    terminated = False
    gamma_values: list[float] = []
    for _prediction in range(max_predictions):
        step = locked_step(
            candidate,
            prepared,
            state,
            previous,
            gate_index=gate_index,
            chosen_record=int(choice["chosen"]),
            gamma_floor=gamma_floor,
        )
        endpoint = decode_endpoint(step.probabilities)
        if endpoint["eos"]:
            terminated = True
            break
        raw.append(int(endpoint["byte"]))
        m_next, coverage_next = advance_copy(
            base.reader,
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
        gamma_values.append(float(step.gamma))
        gate_index = step.gate_index
        previous = int(endpoint["byte"])
    return {
        "raw_bytes": bytes(raw),
        "terminated": terminated,
        "selection": choice,
        "gamma_mean": (sum(gamma_values) / len(gamma_values)) if gamma_values else 0.0,
        "gamma_floor": gamma_floor,
    }


def locked_loss(
    candidate: SelectionCandidate,
    goal: bytes,
    records: tuple,
    keys: list[str],
    target_prefix: tuple[int, ...],
    *,
    subject_id: str,
    gamma_floor: float = GAMMA_FLOOR,
) -> torch.Tensor:
    """Teacher-forced NLL with alpha locked to the head's choice."""
    base = candidate.binding
    choice = select(candidate, goal, keys)
    state = initial_binding_state(base, subject_id)
    prepared = prepare(base, state, goal, records)
    state, gate_index = observe_goal(base, prepared)
    previous = EOS
    losses = []
    labels = tuple(target_prefix) + (EOS,)
    for position, label in enumerate(labels):
        step = locked_step(
            candidate,
            prepared,
            state,
            previous,
            gate_index=gate_index,
            chosen_record=int(choice["chosen"]),
            gamma_floor=gamma_floor,
        )
        losses.append(-torch.log(step.probabilities[label].clamp_min(1e-12)))
        if position == len(labels) - 1:
            break
        m_next, coverage_next = advance_copy(
            base.reader,
            state,
            step.read,
            step.probabilities,
            step.gamma,
            label,
            prepared.base.view,
        )
        state = replace(
            step.state,
            m=m_next,
            coverage=coverage_next,
            prefix_cursor=step.state.prefix_cursor + 1,
        )
        gate_index = step.gate_index
        previous = int(label)
    return sum(losses) / len(losses)


def selection_ce(
    candidate: SelectionCandidate, goal: bytes, keys: list[str], label: int
) -> torch.Tensor:
    logits = torch.cat([candidate.head(goal, keys), candidate.head.null_bias]).unsqueeze(0)
    return torch.nn.functional.cross_entropy(logits, torch.tensor([label]))


def candidate_digest(candidate: SelectionCandidate) -> str:
    params = dict(candidate_parameter_map(candidate.binding))
    params.update({f"head.{name}": value for name, value in candidate.head.state_dict().items()})
    return str(parameter_digest(params))


def state_payload(state: BindingState) -> dict:
    return state.payload()


def _state_from_payload(payload: dict, binding: BindingCopyCandidate) -> BindingState:
    living = state_from_payload(payload["living"], binding.subject.W)
    reference = binding.subject.W
    return BindingState(
        living=living,
        m=payload["m"].to(reference.dtype),
        coverage=payload["coverage"].to(reference.dtype),
        keys=tuple((source, version, offset) for source, version, offset in payload["keys"]),
        record_offsets=tuple((int(a), int(b)) for a, b in payload["record_offsets"]),
        prefix_cursor=int(payload["prefix_cursor"]),
    )


def save_bundle(
    path: Path,
    *,
    profile: str,
    candidate: SelectionCandidate,
    state: BindingState,
    optimizer,
    update_index: int,
    cursor: dict,
    parent_identity: dict,
) -> dict:
    payload = {
        "schema": SELECTION_SCHEMA,
        "step": "N16.1.0b",
        "profile": profile,
        "dtype": str(next(candidate.binding.parameters()).dtype).replace("torch.", ""),
        "relation_config": asdict(candidate.binding.base.config),
        "binding_config": asdict(candidate.binding.config),
        "selection_config": asdict(candidate.head.config),
        "subject_config": dict(vars(candidate.binding.subject.config)),
        "growth_parent_units": getattr(candidate.binding.subject, "growth_parent_units", None),
        "growth_unit_branches": getattr(candidate.binding.subject, "growth_unit_branches", []),
        "growth_gates": list(candidate.binding.growth_gates),
        "base_arm": candidate.binding.base.arm,
        "parent": parent_identity,
        "parameters": candidate_parameter_map(candidate.binding),
        "head_parameters": dict(candidate.head.state_dict()),
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


def render_view_bundle(payload: dict) -> BindingCopyCandidate:
    """Rebuild the binding candidate from a saved payload."""
    from taiji.record_relation import RecordRelationConfig

    config = RecordRelationConfig(**payload["relation_config"])
    subject = build_subject_from_candidate_bundle(payload)
    base = RecordRelationCandidate(
        subject, config, payload["base_arm"], list(payload["growth_gates"])
    )
    named = dict(base.named_parameters())
    for name, value in payload["parameters"].items():
        if name.startswith("relation."):
            named[name].data.copy_(value)
    base.to(torch.float64)
    base.configure_trainable()
    binding = build_binding_candidate(
        base,
        config=BindingCopyConfig(**payload["binding_config"]),
        profile=payload["profile"],
        dtype=torch.float64,
    )
    binding_named = dict(binding.named_parameters())
    if set(binding_named) != set(payload["parameters"]):
        raise ValueError("selection bundle parameter names drifted")
    for name, value in payload["parameters"].items():
        binding_named[name].data.copy_(value)
    binding.to(torch.float64)
    binding.configure_trainable()
    return binding


def load_bundle(path: Path, dtype: torch.dtype = torch.float64) -> tuple:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    binding = render_view_bundle(payload)
    head = SelectionHead(SelectionConfig(**payload["selection_config"])).to(dtype)
    head.load_state_dict(payload["head_parameters"])
    candidate = SelectionCandidate(binding, head, payload["profile"])
    optimizer = make_selection_optimizer(candidate)
    optimizer.load_state_dict(payload["optimizer"])
    state = _state_from_payload(payload["state"], binding)
    torch.set_rng_state(payload["rng_state"])
    return (
        candidate,
        state,
        optimizer,
        payload["profile"],
        int(payload["update_index"]),
        dict(payload["cursor"]),
        payload,
    )
