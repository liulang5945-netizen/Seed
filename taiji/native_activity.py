"""09.3 activity loop on the R021 reference graph (R025 contract).

Six declared activities (interact / recall / imagine / explore / maintain / wait)
driven by a model-owned policy head over the subject's own state quantities; the
host only provides the clock, budget and legal mask. Imagined events never write
the store. Every round appends a traceable decision entry to the monotonic
journal. A fixed-threshold scheduler ships as the declared control arm. No
trainer: the policy parameters stay at their seeded initial values. The carried
nu features are rebuilt from the subject EMA with unknown flags (declared).
"""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn

from taiji.living_core import LivingCoreReference, LivingState, RawSource, state_normalize

ACTIVITIES = ("interact", "recall", "imagine", "explore", "maintain", "wait")
HIDDEN = 16
POLICY_SEED = 20261031
IMAGINE_NORM = 0.1
ERROR_THRESHOLD = 0.10
WAIT_TICKS = 1
NOVEL_PROBE = b"novel-probe-sequence"


class ActivityPolicy(nn.Module):
    """Declared selection head; seeded init, never trained in this stage."""

    def __init__(self, width: int, seed: int = POLICY_SEED) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.input = nn.Linear(2 * width + 19, HIDDEN)
            self.output = nn.Linear(HIDDEN, len(ACTIVITIES))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.output(torch.tanh(self.input(features)))


class ActivityEnvironment:
    """Host-side environment: clock, message queue, store, pending outcomes."""

    def __init__(self, store=(), goal: bytes = b"", now: int = 0) -> None:
        self.message_queue: list[bytes] = []
        self.store = tuple(store)
        self.goal = goal
        self.pending_outcomes: list[str] = []
        self.now = now
        self.owner = "owner"
        self.scope = "scope"


def legal_store(store, owner: str, scope: str, now: int):
    """Filter illegal records (wrong owner/scope, future, unreal) before use."""
    legal, rejected = [], []
    for record in store:
        if (
            record.owner == owner
            and record.scope == scope
            and record.observed_at <= now
            and record.real
            and record.source_id
        ):
            legal.append(record)
        else:
            rejected.append(record.source_id)
    return tuple(legal), rejected


def env_to_extra(env: ActivityEnvironment) -> dict:
    """Serialize the activity environment into bundle-extra-safe plain data."""
    return {
        "message_queue": [list(message) for message in env.message_queue],
        "store": [
            {
                "source_id": record.source_id,
                "raw": list(record.raw),
                "version": record.version,
                "observed_at": record.observed_at,
                "owner": record.owner,
                "scope": record.scope,
                "real": record.real,
            }
            for record in env.store
        ],
        "pending_outcomes": list(env.pending_outcomes),
        "goal": list(env.goal),
        "now": env.now,
        "owner": env.owner,
        "scope": env.scope,
    }


def env_from_extra(data: dict) -> ActivityEnvironment:
    env = ActivityEnvironment(
        store=tuple(
            RawSource(
                source_id=record["source_id"],
                raw=bytes(record["raw"]),
                version=record["version"],
                observed_at=record["observed_at"],
                owner=record.get("owner", "owner"),
                scope=record.get("scope", "scope"),
                real=record.get("real", True),
            )
            for record in data["store"]
        ),
        goal=bytes(data["goal"]),
        now=data["now"],
    )
    env.message_queue = [bytes(message) for message in data["message_queue"]]
    env.pending_outcomes = list(data["pending_outcomes"])
    env.owner = data.get("owner", "owner")
    env.scope = data.get("scope", "scope")
    return env


def prediction_error(output, next_state: LivingState) -> float:
    """mean_i ||world_pred_i - pooled N(h_next)||_2; the world heads predict a
    subject-level d-dim successor representation from z, so the actual is the
    pooled normalized group state (declared reading)."""
    successor = state_normalize(next_state.h).mean(0)
    per_head = (output.prediction - successor.unsqueeze(0)).square().sum(-1).sqrt()
    return float(per_head.mean())


def _carried_nu_features(state: LivingState) -> torch.Tensor:
    scale = state.h.new_tensor([1, 1, 1, 1, 16, 1, 1, 1])
    return torch.cat([torch.tanh(state.nu_ema / scale), torch.zeros(8, dtype=state.h.dtype)])


def build_features(
    state: LivingState,
    goal_vector: torch.Tensor,
    prediction_error_value: float,
    env: ActivityEnvironment,
) -> torch.Tensor:
    pooled = state_normalize(state.h).mean(0)
    scalars = torch.tensor(
        [prediction_error_value, float(len(env.pending_outcomes)), float(len(env.message_queue))],
        dtype=pooled.dtype,
    )
    return torch.cat([pooled, goal_vector, _carried_nu_features(state), scalars])


def select_activity(policy: ActivityPolicy, features: torch.Tensor, legal: dict[str, bool]):
    logits = policy(features)
    mask = torch.tensor([bool(legal[name]) for name in ACTIVITIES], dtype=logits.dtype)
    masked = logits.masked_fill(mask == 0, float("-inf"))
    probabilities = torch.softmax(masked, dim=-1)
    index = int(torch.argmax(probabilities))
    return ACTIVITIES[index], probabilities.detach(), logits.detach()


def fixed_threshold_choice(env: ActivityEnvironment, prediction_error_value: float) -> str:
    """Declared control arm: a fixed ladder over the same trigger quantities."""
    if env.message_queue:
        return "interact"
    if prediction_error_value > ERROR_THRESHOLD and env.store:
        return "recall"
    if env.pending_outcomes:
        return "maintain"
    return "wait"


def legal_mask(env: ActivityEnvironment) -> dict[str, bool]:
    return {
        "interact": bool(env.message_queue),
        "recall": bool(env.store),
        "imagine": True,
        "explore": True,
        "maintain": True,
        "wait": True,
    }


def imagined_event(last_event: torch.Tensor, seq: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(POLICY_SEED + seq)
    noise = torch.randn(last_event.shape[0], generator=generator, dtype=last_event.dtype)
    return last_event + IMAGINE_NORM * noise


def _consume_raw(model, state, raw, env, mode, growth_gates=None):
    """Observe raw bytes step by step, keeping the last output and event."""
    kwargs = {"goal": env.goal, "records": env.store, "mode": mode, "now": env.now}
    if growth_gates is not None:
        kwargs["growth_gates"] = growth_gates
    last_output, last_event = None, None
    for event in model.encoder.patches(raw):
        output = model.step(state, event=event, **kwargs)
        last_output, last_event = output, event.detach().clone()
        state = output.state
    return state, last_output, last_event


def _maintain(model, state: LivingState):
    limit = model.config.fast_limit
    fast = state.fast
    clamped = []
    for group in range(fast.shape[0]):
        norm = float(fast[group].norm())
        if norm > limit:
            fast = fast.clone()
            fast[group] = fast[group] * (limit / norm)
            clamped.append(group)
    return replace(state, fast=fast), {"clamped_groups": clamped, "limit": limit}


def run_round(
    model: LivingCoreReference,
    state: LivingState,
    env: ActivityEnvironment,
    policy: ActivityPolicy,
    journal: list,
    *,
    last_event: torch.Tensor,
    prediction_error_value: float,
    chooser=None,
    gate_provider=None,
    gate_index: int = 0,
):
    """One activity round: select, execute, journal. Returns the updated five-tuple."""
    legal_records, rejected = legal_store(env.store, env.owner, env.scope, env.now)
    if rejected:
        env.store = legal_records
    goal_vector = model.encoder.goal_vector(env.goal)
    features = build_features(state, goal_vector, prediction_error_value, env)
    legal = legal_mask(env)
    if chooser is None:
        activity, probabilities, logits = select_activity(policy, features, legal)
        chosen_by = "model"
    else:
        activity = chooser(env, prediction_error_value)
        logits = policy(features).detach()
        full = torch.softmax(logits, dim=-1)
        probabilities = torch.tensor([float(full[ACTIVITIES.index(activity)])])
        chosen_by = "control"
    h_before = state.h.detach().clone()
    step_before = state.step
    imagined = False
    last_output = None
    note: dict = {"rejected_sources": rejected}
    gates = None if gate_provider is None else gate_provider(gate_index)
    triggers = {
        "queue_count": len(env.message_queue),
        "pending_count": len(env.pending_outcomes),
        "store_count": len(env.store),
        "prediction_error": prediction_error_value,
    }
    if activity == "interact":
        raw = env.message_queue.pop(0)
        state, last_output, last_event = _consume_raw(model, state, raw, env, 0, gates)
    elif activity == "recall":
        record = max(env.store, key=lambda item: item.observed_at)
        state, last_output, last_event = _consume_raw(model, state, record.raw, env, 1, gates)
    elif activity == "imagine":
        imagined = True
        event = imagined_event(last_event, len(journal))
        step_kwargs = {"goal": env.goal, "records": env.store, "mode": 2, "now": env.now}
        if gates is not None:
            step_kwargs["growth_gates"] = gates
        output = model.step(state, event=event, **step_kwargs)
        state = output.state
        last_output, last_event = output, event.detach().clone()
    elif activity == "explore":
        state, last_output, last_event = _consume_raw(model, state, NOVEL_PROBE, env, 3, gates)
    elif activity == "maintain":
        state, note = _maintain(model, state)
    else:
        env.now += WAIT_TICKS
    seq = max((entry["seq"] for entry in journal), default=-1) + 1
    entry = {
        "seq": seq,
        "kind": "activity_decision",
        "activity": activity,
        "logits": [float(value) for value in logits],
        "probs": [float(value) for value in probabilities],
        "chosen_by": chosen_by,
        "triggers": triggers,
        "h_norm_after": float(state.h.norm()),
        "h_changed": not torch.equal(h_before, state.h),
        "step_delta": state.step - step_before,
        "imagined": imagined,
        "store_size_after": len(env.store),
        "external_trigger": activity == "interact" and triggers["queue_count"] > 0,
        "note": note,
        "clock": env.now,
    }
    journal.append(entry)
    if last_output is not None:
        prediction_error_value = prediction_error(last_output, state)
    return state, env, entry, last_event, prediction_error_value
