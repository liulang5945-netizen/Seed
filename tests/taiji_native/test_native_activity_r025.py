"""R025 activity loop oracles: traceable selection, imagined isolation, control arm."""

from __future__ import annotations

import copy
import hashlib
import tempfile
from dataclasses import replace
from pathlib import Path

import torch

from taiji.living_core import LivingCoreConfig, LivingCoreReference, RawSource
from taiji.native_activity import (
    ACTIVITIES,
    ActivityEnvironment,
    ActivityPolicy,
    build_features,
    fixed_threshold_choice,
    legal_mask,
    run_round,
    select_activity,
)
from taiji.state_migration import CareerJournal, load_subject, save_subject

CONFIG = LivingCoreConfig(groups=2, width=8, units=12, adapter_rank=2, key_width=4, value_width=4)
RECORDS = (
    RawSource("s1", b"abc target ab", version=1, observed_at=0),
    RawSource("s2", b"secondary ab", version=2, observed_at=4),
)
GOAL = b"activity-goal"
CAREER = CareerJournal(
    subject_id="r025-subject",
    prompt=b"observe-context-",
    target=b"ab",
    goal=GOAL,
    records=RECORDS,
    raw_nu=(0.5, 2.0, 0.0, 0.8, 8.0, -0.4, 0.0, 0.3),
    known_nu=(True, True, False, True, True, True, False, True),
    now=10,
)


def fresh_loop():
    torch.set_num_threads(1)
    model = LivingCoreReference(CONFIG).double()
    state = model.initial_state("r025-subject")
    env = ActivityEnvironment(store=RECORDS, goal=GOAL, now=10)
    policy = ActivityPolicy(CONFIG.width).double()
    return model, state, env, policy


def detach_state(state):
    return replace(
        state,
        h=state.h.detach(),
        fast=state.fast.detach(),
        nu_ema=state.nu_ema.detach(),
        coverage=state.coverage.detach(),
        prior=state.prior.detach(),
    )


def policy_digest(policy) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(policy.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def test_r025_loop_runs_without_user_messages():
    model, state, env, policy = fresh_loop()
    journal: list = []
    last_event = torch.zeros(CONFIG.width, dtype=torch.float64)
    error_value = 0.0
    store_before = len(env.store)
    for _ in range(6):
        state, env, entry, last_event, error_value = run_round(
            model,
            state,
            env,
            policy,
            journal,
            last_event=last_event,
            prediction_error_value=error_value,
        )
    assert len(journal) == 6
    assert [entry["seq"] for entry in journal] == list(range(6))
    for entry in journal:
        assert entry["chosen_by"] == "model"
        assert len(entry["logits"]) == len(ACTIVITIES)
        assert abs(sum(entry["probs"]) - 1.0) < 1e-9
        assert entry["triggers"]["queue_count"] == 0
    assert any(entry["activity"] != "wait" for entry in journal)
    assert any(entry["h_changed"] for entry in journal)
    assert all(entry["store_size_after"] == store_before for entry in journal)
    imagined = [entry for entry in journal if entry["imagined"]]
    assert imagined and all(entry["store_size_after"] == store_before for entry in imagined)


def test_r025_model_policy_differs_from_fixed_threshold_control():
    model_a, state_a, env_a, policy = fresh_loop()
    model_b, state_b, env_b, _ = fresh_loop()
    journal_model: list = []
    journal_control: list = []
    last_a = torch.zeros(CONFIG.width, dtype=torch.float64)
    last_b = torch.zeros(CONFIG.width, dtype=torch.float64)
    error_a = error_b = 0.0
    for _ in range(6):
        state_a, env_a, _, last_a, error_a = run_round(
            model_a,
            state_a,
            env_a,
            policy,
            journal_model,
            last_event=last_a,
            prediction_error_value=error_a,
        )
        state_b, env_b, _, last_b, error_b = run_round(
            model_b,
            state_b,
            env_b,
            policy,
            journal_control,
            last_event=last_b,
            prediction_error_value=error_b,
            chooser=fixed_threshold_choice,
        )
    model_sequence = [entry["activity"] for entry in journal_model]
    control_sequence = [entry["activity"] for entry in journal_control]
    assert all(entry["chosen_by"] == "model" for entry in journal_model)
    assert all(entry["chosen_by"] == "control" for entry in journal_control)
    assert model_sequence != control_sequence


def test_r025_triggers_reachable_and_recorded():
    model, state, env, policy = fresh_loop()
    env.message_queue.append(b"external message one")
    env.pending_outcomes.append("o-1")
    journal: list = []
    state, env, entry, _, error_value = run_round(
        model,
        state,
        env,
        policy,
        journal,
        last_event=torch.zeros(CONFIG.width, dtype=torch.float64),
        prediction_error_value=0.0,
    )
    assert entry["triggers"]["queue_count"] == 1
    assert entry["triggers"]["pending_count"] == 1
    assert entry["chosen_by"] == "model"
    state, env, interact_entry, _, interact_error = run_round(
        model,
        state,
        env,
        policy,
        journal,
        last_event=torch.zeros(CONFIG.width, dtype=torch.float64),
        prediction_error_value=0.0,
        chooser=lambda env_, error: "interact",
    )
    assert interact_entry["activity"] == "interact"
    assert interact_entry["triggers"]["queue_count"] == 1
    assert interact_error > 0.0
    state, env, wait_entry, _, carried = run_round(
        model,
        state,
        env,
        policy,
        journal,
        last_event=torch.zeros(CONFIG.width, dtype=torch.float64),
        prediction_error_value=interact_error,
        chooser=lambda env_, error: "wait",
    )
    assert wait_entry["activity"] == "wait"
    assert carried == interact_error
    features = build_features(state, model.encoder.goal_vector(env.goal), carried, env)
    assert features.shape[0] == 2 * CONFIG.width + 19
    assert select_activity(policy, features, legal_mask(env))[0] in ACTIVITIES


def test_r025_wait_and_maintain_semantics():
    model, state, env, policy = fresh_loop()
    journal: list = []
    clock_before = env.now
    state, env, wait_entry, _, _ = run_round(
        model,
        state,
        env,
        policy,
        journal,
        last_event=torch.zeros(CONFIG.width, dtype=torch.float64),
        prediction_error_value=0.0,
        chooser=lambda env_, error: "wait",
    )
    assert wait_entry["activity"] == "wait"
    assert wait_entry["h_changed"] is False and wait_entry["step_delta"] == 0
    assert env.now == clock_before + 1
    fast_before = state.fast.detach().clone()
    state, env, maintain_entry, _, _ = run_round(
        model,
        state,
        env,
        policy,
        journal,
        last_event=torch.zeros(CONFIG.width, dtype=torch.float64),
        prediction_error_value=0.0,
        chooser=lambda env_, error: "maintain",
    )
    assert maintain_entry["activity"] == "maintain"
    assert maintain_entry["note"]["clamped_groups"] == []
    assert torch.equal(state.fast, fast_before)


def test_r025_policy_unchanged_and_loop_restore_bitwise():
    model, state, env, policy = fresh_loop()
    digest_before = policy_digest(policy)
    journal: list = []
    last_event = torch.zeros(CONFIG.width, dtype=torch.float64)
    error_value = 0.0
    for _ in range(3):
        state, env, _, last_event, error_value = run_round(
            model,
            state,
            env,
            policy,
            journal,
            last_event=last_event,
            prediction_error_value=error_value,
        )
    assert policy_digest(policy) == digest_before
    state = detach_state(state)
    with tempfile.TemporaryDirectory() as directory:
        bundle = Path(directory) / "activity_bundle.pt"
        save_subject(
            bundle,
            model,
            state,
            career=CAREER,
            journal_cursor=0,
            eligibility=torch.zeros(CONFIG.groups, dtype=torch.float64),
            extra={"policy_state": policy.state_dict()},
        )
        restored = load_subject(bundle)
    policy_restored = ActivityPolicy(CONFIG.width).double()
    policy_restored.load_state_dict(restored["extra"]["policy_state"])
    clone_env = copy.deepcopy(env)
    entry_original = run_round(
        model,
        state,
        env,
        policy,
        list(journal),
        last_event=last_event,
        prediction_error_value=error_value,
    )[2]
    entry_restored = run_round(
        model,
        state,
        clone_env,
        policy_restored,
        list(journal),
        last_event=last_event.clone(),
        prediction_error_value=error_value,
    )[2]
    assert entry_original["activity"] == entry_restored["activity"]
    assert entry_original["logits"] == entry_restored["logits"]
    assert entry_original["h_norm_after"] == entry_restored["h_norm_after"]
