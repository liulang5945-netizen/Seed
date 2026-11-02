"""09.6.2 fixture oracles: recurring arrivals, a reachable bound, and the guards.

The point of these tests is discriminating power, not coverage: the analytic
oracle must equal what a scripted optimal policy actually earns, and a degenerate
schedule must make the reachability gate fail.
"""

from __future__ import annotations

import torch

from taiji.activity_learning import INTERACT_REWARD, activity_reward
from taiji.living_core import LivingCoreConfig, LivingCoreReference, RawSource
from taiji.native_activity import (
    ActivityEnvironment,
    ActivityPolicy,
    fixed_threshold_choice,
    run_round,
)
from taiji.native_activity_fixture import (
    MESSAGES,
    ArrivalSource,
    legal_interactions,
    oracle_cumulative_reward,
    patches,
)
from taiji.state_migration import CareerJournal

CONFIG = LivingCoreConfig(groups=2, width=8, units=12, adapter_rank=2, key_width=4, value_width=4)
RECORDS = (RawSource("s1", "蓝色杯子在木桌上。".encode(), version=1, observed_at=1),)
JOURNAL = CareerJournal(
    subject_id="r036-fixture",
    prompt="蓝色杯子现在在哪里？".encode(),
    target="在窗台。".encode(),
    goal="根据已见材料回答。".encode(),
    records=RECORDS,
    raw_nu=(0.5, 2.0, 0.0, 0.8, 8.0, -0.4, 0.0, 0.3),
    known_nu=(True, True, False, True, True, True, False, True),
    now=10,
)
ROUNDS = 32


def test_arrivals_deliver_one_message_per_tick():
    env = ActivityEnvironment(store=RECORDS, goal=JOURNAL.goal, now=0)
    source = ArrivalSource(messages=MESSAGES[:3], interval=1)
    assert source(env) == 1
    assert env.message_queue == [MESSAGES[0]]
    assert source(env) == 0
    env.now = 1
    assert source(env) == 1
    assert env.message_queue[-1] == MESSAGES[1]
    env.now = 5
    assert source(env) == 1
    assert len(env.message_queue) == 3
    assert source(env) == 0


def test_a_fresh_source_restarts_the_schedule():
    first = ArrivalSource(messages=MESSAGES[:2], interval=1)
    env = ActivityEnvironment(store=RECORDS, goal=JOURNAL.goal, now=0)
    first(env)
    env.now = 1
    first(env)
    assert first.delivered == 2
    second = ArrivalSource(messages=MESSAGES[:2], interval=1)
    fresh = ActivityEnvironment(store=RECORDS, goal=JOURNAL.goal, now=0)
    assert second(fresh) == 1
    assert fresh.message_queue == [MESSAGES[0]]


def test_oracle_guard_fails_on_a_degenerate_schedule():
    reachable = oracle_cumulative_reward(ROUNDS)
    assert reachable["oracle_reward"] >= 0.5
    degenerate = oracle_cumulative_reward(ROUNDS, interval=10**6)
    assert degenerate["legal_interactions"] == 1
    assert degenerate["oracle_reward"] < 0.5
    assert legal_interactions(4, interval=1) == 2


def scripted_optimum(rounds: int) -> float:
    torch.manual_seed(3)
    model = LivingCoreReference(CONFIG).double()
    state = model.initial_state("r036-scripted")
    env = ActivityEnvironment(store=RECORDS, goal=JOURNAL.goal, now=0)
    policy = ActivityPolicy(CONFIG.width).double()
    last_event = torch.zeros(CONFIG.width, dtype=torch.float64)
    cumulative = 0.0
    for _ in range(rounds):
        before = state
        with torch.no_grad():
            state, env, entry, last_event, _ = run_round(
                model,
                state,
                env,
                policy,
                [],
                last_event=last_event,
                prediction_error_value=0.0,
                chooser=lambda current, _error: "interact" if current.message_queue else "wait",
                arrivals=ArrivalSource(),
            )
        cumulative += activity_reward(
            model,
            before,
            state,
            JOURNAL,
            JOURNAL,
            entry["step_delta"],
            actual_reward=INTERACT_REWARD if entry.get("external_trigger") else 0.0,
        )["reward"]
    return cumulative


def test_oracle_floor_is_a_lower_bound_reached_by_a_scripted_policy():
    """The analytic bound is a floor: the probe's NLL term can only add reward."""
    oracle = oracle_cumulative_reward(ROUNDS)
    realized = scripted_optimum(ROUNDS)
    assert realized >= oracle["oracle_reward"]
    assert realized >= 0.5


def test_declared_schedule_keeps_interact_legal_in_half_the_rounds():
    env = ActivityEnvironment(store=RECORDS, goal=JOURNAL.goal, now=0)
    source = ArrivalSource()
    legal_rounds = 0
    for _ in range(ROUNDS):
        source(env)
        legal_rounds += int(bool(env.message_queue))
        if env.message_queue:
            env.message_queue.pop(0)
        else:
            env.now += 1
    assert legal_rounds == legal_interactions(ROUNDS)
    assert legal_rounds >= ROUNDS // 2


def test_default_arrivals_path_is_unchanged():
    torch.manual_seed(4)
    model = LivingCoreReference(CONFIG).double()
    state = model.initial_state("r036-default")
    policy = ActivityPolicy(CONFIG.width).double()
    env = ActivityEnvironment(store=RECORDS, goal=JOURNAL.goal, now=0)
    env.message_queue.append(MESSAGES[0])
    with torch.no_grad():
        _, _, entry, _, _ = run_round(
            model,
            state,
            env,
            policy,
            [],
            last_event=torch.zeros(CONFIG.width, dtype=torch.float64),
            prediction_error_value=0.0,
            chooser=fixed_threshold_choice,
        )
    assert entry["activity"] == "interact"
    assert entry["external_trigger"] is True
    assert entry["step_delta"] == patches(MESSAGES[0])
    assert env.message_queue == []
