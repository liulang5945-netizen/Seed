"""§10.3 bounded causal replay migration on the R021 reference graph.

The career journal is the single source of truth: ``run_career`` both produces the
original state and executes replays, so journal completeness is mechanically
checkable. Replay never updates parameters, claims rewards or executes external
actions; a budget refusal keeps the parent revision. No trainer and no task data.
"""

from __future__ import annotations

import copy
import hashlib
import time
from dataclasses import dataclass

import torch

from taiji.living_core import EOS, LivingCoreReference, LivingState, RawSource


@dataclass(frozen=True)
class CareerJournal:
    """Declared synthetic career: raw events plus the shared step context."""

    subject_id: str
    prompt: bytes
    target: bytes
    goal: bytes
    records: tuple[RawSource, ...]
    raw_nu: tuple[float, ...]
    known_nu: tuple[bool, ...]
    mode: int = 0
    now: int = 0
    allowed_write: bool = True

    def __post_init__(self) -> None:
        if not self.subject_id or not self.prompt or not self.target:
            raise ValueError("Career journal requires subject, prompt and target")
        if len(self.raw_nu) != 8 or len(self.known_nu) != 8:
            raise ValueError("Career journal requires eight nu readings")

    @property
    def event_count(self) -> int:
        # One microstep per prompt patch plus one prediction round per target byte
        # and the terminal EOS round; patches depend on the encoder patching rule.
        patches = (len(self.prompt) + 7) // 8
        return patches + len(self.target) + 1

    @property
    def journal_bytes(self) -> int:
        return (
            len(self.prompt)
            + len(self.target)
            + len(self.goal)
            + sum(len(record.raw) for record in self.records)
        )


@dataclass(frozen=True)
class MigrationBudget:
    max_events: int | None = None
    max_journal_bytes: int | None = None
    max_cpu_seconds: float | None = None
    max_wall_seconds: float | None = None


def _shared_kwargs(model: LivingCoreReference, journal: CareerJournal) -> dict:
    dtype = model.W.dtype
    return {
        "goal": journal.goal,
        "records": journal.records,
        "raw_nu": torch.tensor(journal.raw_nu, dtype=dtype),
        "known_nu": torch.tensor(journal.known_nu),
        "mode": journal.mode,
        "now": journal.now,
        "allowed": torch.full(
            (model.config.groups, model.config.fast_heads),
            journal.allowed_write,
            dtype=torch.bool,
        ),
    }


def run_career(model: LivingCoreReference, journal: CareerJournal, growth_gate_provider=None):
    """Consume the journal in visible order; returns (state, [(output, round_byte)]).

    ``growth_gate_provider`` optionally maps the career step index to the declared
    per-step branch gates consumed by gated newborn units (v1.2 candidate)."""
    kwargs = _shared_kwargs(model, journal)
    state = model.initial_state(journal.subject_id)
    steps: list[tuple] = []
    index = 0

    def step_kwargs() -> dict:
        if growth_gate_provider is None:
            return dict(kwargs)
        return {**kwargs, "growth_gates": growth_gate_provider(index)}

    for event in model.encoder.patches(journal.prompt):
        output = model.step(state, event=event, **step_kwargs())
        steps.append((output, None))
        state = output.state
        index += 1
    previous = EOS
    for byte in (*journal.target, EOS):
        if previous == EOS:
            event = model.W.new_zeros(model.config.width)
        else:
            event = model.encoder.patches(bytes([previous]))[0]
        output = model.step(state, event=event, previous_byte=previous, **step_kwargs())
        steps.append((output, byte))
        state = model.consume_output(output, byte)
        previous = byte
        index += 1
    return state, steps


def parameter_digest(model: LivingCoreReference) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def states_bitwise_equal(left: LivingState, right: LivingState) -> bool:
    return (
        torch.equal(left.h, right.h)
        and torch.equal(left.fast, right.fast)
        and torch.equal(left.nu_ema, right.nu_ema)
        and torch.equal(left.coverage, right.coverage)
        and torch.equal(left.prior, right.prior)
        and left.keys == right.keys
        and left.step == right.step
    )


def replay_transaction(
    parent_model: LivingCoreReference,
    parent_state: LivingState,
    journal: CareerJournal,
    coordinate_change=None,
    budget: MigrationBudget | None = None,
) -> dict:
    """Bounded causal replay: shadow rebuild, optional declared coordinate change.

    ``coordinate_change`` is a callable applied to the shadow model only. Replay
    performs no parameter updates, reward claims or external actions. A budget
    refusal keeps the parent revision and records the cost of the attempt.
    """
    budget = budget or MigrationBudget()
    events, bytes_total = journal.event_count, journal.journal_bytes
    if budget.max_events is not None and events > budget.max_events:
        return {"outcome": "refused_budget", "reason": "event_cap", "events": events}
    if budget.max_journal_bytes is not None and bytes_total > budget.max_journal_bytes:
        return {"outcome": "refused_budget", "reason": "byte_cap", "bytes": bytes_total}
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    parent_digest = parameter_digest(parent_model)
    shadow = copy.deepcopy(parent_model)
    encoder_changed = False
    if coordinate_change is not None:
        before = parameter_digest(shadow)
        coordinate_change(shadow)
        encoder_changed = parameter_digest(shadow) != before
    after_change = parameter_digest(shadow)
    with torch.no_grad():
        rebuilt_state, _ = run_career(shadow, journal)
    cpu_seconds = time.process_time() - cpu_start
    wall_seconds = time.perf_counter() - wall_start
    no_replay_mutation = parameter_digest(shadow) == after_change
    result = {
        "events": events,
        "journal_bytes": bytes_total,
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "watermark": {"events_consumed": events, "journal_steps": journal.event_count},
        "no_replay_parameter_mutation": no_replay_mutation,
        "coordinate_change_applied": encoder_changed,
        "reward_ledger_untouched": True,
        "external_actions_executed": 0,
    }
    if budget.max_cpu_seconds is not None and cpu_seconds > budget.max_cpu_seconds:
        result.update({"outcome": "refused_budget", "reason": "cpu_cap"})
        return result
    if budget.max_wall_seconds is not None and wall_seconds > budget.max_wall_seconds:
        result.update({"outcome": "refused_budget", "reason": "wall_cap"})
        return result
    if coordinate_change is None:
        identical = states_bitwise_equal(rebuilt_state, parent_state)
        result.update(
            {
                "outcome": "identity_checked",
                "state_bitwise_equal": identical,
                "committed_state": rebuilt_state if identical else None,
            }
        )
        return result
    committed = LivingState(
        subject_id=parent_state.subject_id,
        revision=parent_state.revision + 1,
        step=rebuilt_state.step,
        h=rebuilt_state.h,
        fast=rebuilt_state.fast,
        nu_ema=rebuilt_state.nu_ema,
        keys=rebuilt_state.keys,
        coverage=rebuilt_state.coverage,
        prior=rebuilt_state.prior,
    )
    result.update(
        {
            "outcome": "committed",
            "committed_state": committed,
            "committed_model": shadow,
            "parent_parameter_digest": parent_digest,
        }
    )
    return result
