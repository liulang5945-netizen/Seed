"""R029 engineering-gap oracles: constrained updates, activity learning, lesion,
natural-Chinese fixtures and subprocess CPU accounting.

Every test is written so it can fail: solvers are checked in three distinct
regimes, rollback is proved against a permissive run of the same configuration,
shortcut rulers must have non-zero and non-full scores, and the worker CPU field
is compared against the parent's own measured CPU during the join.
"""

from __future__ import annotations

import multiprocessing
import tempfile
import time
from pathlib import Path

import torch

from taiji import constrained_learning as cl
from taiji import growth_credit as gc
from taiji.activity_learning import (
    ActivityLearner,
    evaluate_activity_policy,
    train_activity_episode,
)
from taiji.living_core import LivingCoreConfig, LivingCoreReference, RawSource
from taiji.native_activity import ActivityEnvironment, ActivityPolicy, run_round
from taiji.native_task_fixtures import build_fixtures, shortcut_probe
from taiji.state_migration import (
    CareerJournal,
    load_subject,
    resume_career,
    resume_subject,
    run_career,
    run_worker_job,
    save_subject,
)

CONFIG = LivingCoreConfig(groups=2, width=8, units=12, adapter_rank=2, key_width=4, value_width=4)
RECORDS = (
    RawSource("s1", "蓝色杯子在木桌上。".encode(), version=1, observed_at=1),
    RawSource("s2", "更正：蓝色杯子在窗台。".encode(), version=2, observed_at=5),
)
NU = (0.5, 2.0, 0.0, 0.8, 8.0, -0.4, 0.0, 0.3)
KNOWN = (True, True, False, True, True, True, False, True)
FIT = CareerJournal(
    subject_id="r029-fit",
    prompt="蓝色杯子现在在哪里？".encode(),
    target="在窗台。".encode(),
    goal="根据已见材料回答。".encode(),
    records=RECORDS,
    raw_nu=NU,
    known_nu=KNOWN,
    now=10,
)
OLD = CareerJournal(
    subject_id="r029-old",
    prompt="更正前蓝色杯子在哪里？".encode(),
    target="在木桌上。".encode(),
    goal="根据已见材料回答。".encode(),
    records=RECORDS,
    raw_nu=NU,
    known_nu=KNOWN,
    now=10,
)
_CHILD_CACHE = None


def new_model(subject: str):
    torch.set_num_threads(1)
    model = LivingCoreReference(CONFIG).double()
    return model, model.initial_state(subject)


def child_pipeline():
    global _CHILD_CACHE
    if _CHILD_CACHE is None:
        model = LivingCoreReference(CONFIG).double()
        book, _ = gc.collect_credit(model, FIT)
        directions, credits = {}, {}
        for branch, name in enumerate(book.branch_ids):
            credit = gc.branch_credit(book, branch)
            credits[name] = credit
            for group in range(CONFIG.groups):
                _, _, vectors = torch.linalg.svd(credit[group], full_matrices=False)
                directions[(name, group)] = [vectors[:, 0].detach().clone()]
        child, plan = gc.birth(model, directions, credits)
        _CHILD_CACHE = {"model": model, "book": book, "child": child, "plan": plan}
    return _CHILD_CACHE


def spawn_worker(spec: dict) -> dict:
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=run_worker_job, args=(spec,))
    process.start()
    process.join(timeout=600)
    if process.exitcode != 0:
        raise RuntimeError(f"worker failed with exitcode {process.exitcode}")
    return torch.load(spec["out"], weights_only=True)


def test_r029_solver_covers_three_regimes():
    dtype = torch.float64
    generator = torch.Generator().manual_seed(20261101)
    c_small = torch.rand(6, dtype=dtype, generator=generator) - 0.5
    delta, diagnostics = cl.solve_constrained_step(
        c_small, torch.zeros((0, 6), dtype=dtype), torch.zeros(0, dtype=dtype), 1.0
    )
    assert torch.equal(delta, -c_small)
    assert diagnostics.ball_active is False
    assert diagnostics.active_constraints == ()
    assert diagnostics.kkt_residual <= cl.KKT_TOLERANCE

    c_large = 50.0 * c_small
    delta, diagnostics = cl.solve_constrained_step(
        c_large, torch.zeros((0, 6), dtype=dtype), torch.zeros(0, dtype=dtype), 1.0
    )
    assert diagnostics.ball_active is True
    assert abs(diagnostics.delta_norm - 1.0) <= 1e-9
    assert diagnostics.kkt_residual <= cl.KKT_TOLERANCE

    row = c_small / c_small.norm()
    orthogonal = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=dtype)
    orthogonal = orthogonal - orthogonal.dot(row) * row
    orthogonal = orthogonal / orthogonal.norm()
    constraints = row.unsqueeze(0)
    epsilons = torch.zeros(1, dtype=dtype)
    # The steepest-descent step -c = 3*row + 2*orthogonal violates row.delta <= 0,
    # so the constraint must bind while the orthogonal component survives.
    delta, diagnostics = cl.solve_constrained_step(
        -3.0 * row - 2.0 * orthogonal, constraints, epsilons, 10.0
    )
    assert diagnostics.feasible
    assert diagnostics.active_constraints == (0,)
    assert diagnostics.constraint_values[0] <= 1e-10
    assert torch.allclose(delta, 2.0 * orthogonal, atol=1e-9)
    assert diagnostics.kkt_residual <= cl.KKT_TOLERANCE
    assert diagnostics.complementarity_residual <= 1e-8


def test_r029_solver_rejects_a_violating_step():
    dtype = torch.float64
    generator = torch.Generator().manual_seed(7)
    c = torch.rand(5, dtype=dtype, generator=generator) + 0.5
    constraints = torch.eye(5, dtype=dtype)[:2]
    epsilons = torch.zeros(2, dtype=dtype)
    delta, diagnostics = cl.solve_constrained_step(c, constraints, epsilons, 100.0)
    assert diagnostics.feasible
    assert all(value <= 1e-10 for value in diagnostics.constraint_values)
    assert cl._feasible(c, constraints, epsilons, 100.0) is False
    assert cl._feasible(delta, constraints, epsilons, 100.0) is True
    assert diagnostics.objective <= cl._objective(c, c)


def test_r029_gradient_vector_zero_fills_dead_parameters():
    model, _ = new_model("r029-dead")
    parameters = list(model.parameters())
    loss = parameters[0].square().sum()
    vector = cl.gradient_vector(loss, parameters)
    assert vector.numel() == sum(parameter.numel() for parameter in parameters)
    head = parameters[0].numel()
    assert float(vector[:head].abs().sum()) > 0.0
    assert float(vector[head:].abs().sum()) == 0.0


def test_r029_constrained_step_applies_a_feasible_delta():
    model, _ = new_model("r029-step")
    before = [parameter.detach().clone() for parameter in model.parameters()]
    row = cl.constrained_feedback_step(model, FIT, (OLD,), eta=1.0, epsilon=2e-3, rho=2e-2)
    diagnostics = row["diagnostics"]
    assert diagnostics.feasible
    assert diagnostics.kkt_residual <= cl.KKT_TOLERANCE
    assert diagnostics.delta_norm <= 2e-2 + 1e-10
    assert all(value <= 2e-3 + 1e-10 for value in diagnostics.constraint_values)
    assert row["gradient_norm"] > 0.0
    changed = [
        not torch.equal(parameter, snapshot)
        for parameter, snapshot in zip(model.parameters(), before, strict=True)
    ]
    assert any(changed)
    assert set(row["timings"]) == {"fit_gradient", "old_gradients", "solve", "apply"}


def test_r029_actual_probe_rolls_back_and_halves_rho():
    permissive_model, _ = new_model("r029-probe-a")
    permissive = cl.constrained_feedback_updates(
        permissive_model,
        FIT,
        (OLD,),
        steps=1,
        actual_epsilon=1.0,
        max_retries=2,
    )
    assert permissive[0]["accepted"] is True
    assert permissive[0]["actual_old_max_abs_change"] > 0.0

    model, _ = new_model("r029-probe-b")
    before = [parameter.detach().clone() for parameter in model.parameters()]
    strict = cl.constrained_feedback_updates(
        model, FIT, (OLD,), steps=2, actual_epsilon=0.0, max_retries=2
    )
    assert len(strict) == 1
    assert strict[0]["accepted"] is False
    assert strict[0]["attempts"] == 3
    assert strict[0]["rho_used"] == 2e-2 * 0.5**2
    assert all(
        torch.equal(parameter, snapshot)
        for parameter, snapshot in zip(model.parameters(), before, strict=True)
    )


def activity_setup(rounds: int):
    model, state = new_model(f"r029-activity-{rounds}")
    env = ActivityEnvironment(store=RECORDS, goal=FIT.goal, now=10)
    env.message_queue.append("蓝色杯子在窗台。".encode())
    learner = ActivityLearner.build(CONFIG.width)
    return model, state, env, learner


def test_r029_training_moves_policy_and_critic_and_roundtrips():
    model, state, env, learner = activity_setup(3)
    policy_before = {
        name: parameter.detach().clone() for name, parameter in learner.policy.named_parameters()
    }
    critic_before = {
        name: parameter.detach().clone() for name, parameter in learner.critic.named_parameters()
    }
    journal: list = []
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "learner.pt"
        _, _, _, _, rows = train_activity_episode(
            model,
            state,
            env,
            learner,
            FIT,
            OLD,
            rounds=3,
            journal=journal,
            last_event=torch.zeros(CONFIG.width, dtype=torch.float64),
            max_exposures=8,
        )
        assert [row["status"] for row in rows] == ["updated"] * 3
        assert all(row["chosen_by"] == "model_sampled" for row in rows)
        assert all(row["round_cpu_seconds"] >= 0.0 for row in rows)
        assert all("triggers" in row and "eligibility_norm" in row for row in rows)
        assert learner.exposures == 3
        assert any(
            not torch.equal(parameter, policy_before[name])
            for name, parameter in learner.policy.named_parameters()
        )
        assert any(
            not torch.equal(parameter, critic_before[name])
            for name, parameter in learner.critic.named_parameters()
        )
        torch.save(learner.state_dict(), path)
        restored = ActivityLearner.build(CONFIG.width)
        restored.load_state_dict(torch.load(path, weights_only=True))
        assert restored.exposures == learner.exposures
        for name, parameter in learner.policy.named_parameters():
            assert torch.equal(parameter, dict(restored.policy.named_parameters())[name])
        for name, parameter in learner.critic.named_parameters():
            assert torch.equal(parameter, dict(restored.critic.named_parameters())[name])
        for name, trace in learner.eligibility.items():
            assert torch.equal(trace, restored.eligibility[name])
        original_state = learner.critic_optimizer.state_dict()["state"]
        restored_state = restored.critic_optimizer.state_dict()["state"]
        assert original_state, "critic optimizer state must be populated before comparing"
        assert set(original_state) == set(restored_state)
        for key, moments in original_state.items():
            assert set(moments) == set(restored_state[key])
            for field, value in moments.items():
                other = restored_state[key][field]
                if torch.is_tensor(value):
                    assert torch.equal(value, other)
                else:
                    assert value == other
        assert len(journal) == 3


def test_r029_exposure_budget_stops_training():
    model, state, env, learner = activity_setup(1)
    rows = train_activity_episode(
        model,
        state,
        env,
        learner,
        FIT,
        OLD,
        rounds=4,
        journal=[],
        last_event=torch.zeros(CONFIG.width, dtype=torch.float64),
        max_exposures=1,
    )[-1]
    assert learner.exposures == 1
    assert [row["status"] for row in rows] == ["updated", "exposure_budget_stop"]


def test_r029_evaluation_arms_share_the_budget_and_ledger():
    model, state, env, learner = activity_setup(2)
    last_event = torch.zeros(CONFIG.width, dtype=torch.float64)
    _, _, _, _, model_rows, model_cumulative = evaluate_activity_policy(
        model,
        state,
        ActivityEnvironment(store=RECORDS, goal=FIT.goal, now=10),
        learner.policy,
        FIT,
        OLD,
        rounds=2,
        journal=[],
        last_event=last_event,
    )
    _, _, _, _, control_rows, control_cumulative = evaluate_activity_policy(
        model,
        state,
        ActivityEnvironment(store=RECORDS, goal=FIT.goal, now=10),
        ActivityPolicy(CONFIG.width).double(),
        FIT,
        OLD,
        rounds=2,
        journal=[],
        last_event=last_event,
        control=True,
    )
    assert all(row["chosen_by"] == "model" for row in model_rows)
    assert all(row["chosen_by"] == "control" for row in control_rows)
    assert len(model_rows) == len(control_rows) == 2
    assert model_cumulative == sum(row["reward"] for row in model_rows)
    assert control_cumulative == sum(row["reward"] for row in control_rows)


def test_r029_idle_activities_do_not_fabricate_microsteps():
    model, state, env, learner = activity_setup(0)
    last_event = torch.zeros(CONFIG.width, dtype=torch.float64)
    for activity in ("wait", "maintain"):
        journal: list = []
        round_env = ActivityEnvironment(store=RECORDS, goal=FIT.goal, now=10)
        _, _, entry, _, _ = run_round(
            model,
            state,
            round_env,
            learner.policy,
            journal,
            last_event=last_event,
            prediction_error_value=0.0,
            chooser=lambda _env, _error, activity=activity: activity,
        )
        assert entry["activity"] == activity
        assert entry["step_delta"] == 0
        assert entry["chosen_by"] == "control"
    wait_env = ActivityEnvironment(store=RECORDS, goal=FIT.goal, now=10)
    clock_before = wait_env.now
    _, _, entry, _, _ = run_round(
        model,
        state,
        wait_env,
        learner.policy,
        [],
        last_event=last_event,
        prediction_error_value=0.0,
        chooser=lambda _env, _error: "wait",
    )
    assert entry["h_changed"] is False
    assert wait_env.now == clock_before + 1


def test_r029_imagination_never_writes_the_store():
    model, state, env, learner = activity_setup(0)
    store_before = env.store
    _, _, entry, _, _ = run_round(
        model,
        state,
        env,
        learner.policy,
        [],
        last_event=torch.zeros(CONFIG.width, dtype=torch.float64),
        prediction_error_value=0.0,
        chooser=lambda _env, _error: "imagine",
    )
    assert entry["imagined"] is True
    assert entry["store_size_after"] == len(store_before)
    assert env.store == store_before
    assert entry["step_delta"] == 1


def test_r029_lesion_restores_the_parent_function_bitwise():
    bundle = child_pipeline()
    model, child, plan, book = (
        bundle["model"],
        bundle["child"],
        bundle["plan"],
        bundle["book"],
    )

    def provider(index: int) -> dict:
        values = book.gates[index % len(book.gates)]
        return {name: float(value) for name, value in zip(book.branch_ids, values, strict=True)}

    with torch.no_grad():
        parent_state, parent_steps = run_career(model, FIT)
        child_state, child_steps = run_career(child, FIT, growth_gate_provider=provider)
    assert child.growth_parent_units is not None
    gc.lesion(child, plan)
    assert child.growth_parent_units is None
    with torch.no_grad():
        lesion_state, lesion_steps = run_career(child, FIT)
    assert torch.equal(lesion_state.h, parent_state.h)
    assert all(
        torch.equal(left.probabilities, right.probabilities)
        for (left, _), (right, _) in zip(lesion_steps, parent_steps, strict=True)
    )
    assert len(child_steps) == len(parent_steps)


def test_r029_lesion_trained_keeps_learned_parent_rows():
    model = LivingCoreReference(CONFIG).double()
    book, _ = gc.collect_credit(model, OLD)
    directions, credits = {}, {}
    for branch, name in enumerate(book.branch_ids):
        credit = gc.branch_credit(book, branch)
        credits[name] = credit
        for group in range(CONFIG.groups):
            _, _, vectors = torch.linalg.svd(credit[group], full_matrices=False)
            directions[(name, group)] = [vectors[:, 0].detach().clone()]
    fresh_child, _ = gc.birth(model, directions, credits)
    parent_units = fresh_child.growth_parent_units
    assert parent_units is not None
    with torch.no_grad():
        fresh_child.R[:, :parent_units, :].mul_(1.5)
        fresh_child.P[:, :, :parent_units].mul_(1.5)
    trained_rows = fresh_child.R[:, :parent_units, :].detach().clone()
    report = gc.lesion_trained(fresh_child)
    assert report["removed_units_per_group"] > 0
    assert fresh_child.R.shape[1] == parent_units
    assert fresh_child.P.shape[2] == parent_units
    assert torch.equal(fresh_child.R, trained_rows)
    assert fresh_child.growth_parent_units is None


def test_r029_fixture_shortcut_ruler_has_range_and_order_sensitivity():
    fixtures = build_fixtures()
    assert {item.category for item in fixtures} == {
        "current",
        "previous",
        "unknown",
        "alias",
        "conflict",
    }
    probe = shortcut_probe(fixtures)
    for strategy, row in probe["as_given"].items():
        assert 0 < row["correct"] < row["total"], strategy
    assert probe["record_order_reversed"] != {
        strategy: row["correct"] for strategy, row in probe["as_given"].items()
    }
    for fixture in fixtures:
        assert fixture.answer_denominator == len(fixture.journal.target) + 1
        assert fixture.answer_denominator > 1


def test_r029_fixture_runtime_payload_excludes_scoring_metadata():
    fixtures = build_fixtures()
    forbidden = {"category", "expected_target", "shortcut_label", "answer_denominator", "gold"}
    for fixture in fixtures:
        assert not forbidden & set(fixture.journal.__dataclass_fields__)
        assert fixture.journal.target == fixture.expected_target
        for record in fixture.journal.records:
            assert record.source_id and record.raw and record.version >= 1
            assert record.owner and record.scope and record.real


def test_r029_worker_accounts_its_own_cpu():
    model, state = new_model("r029-worker")
    with tempfile.TemporaryDirectory() as directory:
        bundle = Path(directory) / "bundle.pt"
        out = Path(directory) / "out.pt"
        save_subject(
            bundle,
            model,
            state,
            career=FIT,
            journal_cursor=0,
            eligibility=torch.zeros(CONFIG.groups, dtype=torch.float64),
        )
        restored_model, restored_state, _ = resume_subject(load_subject(bundle))
        with torch.no_grad():
            local_state, local_steps = resume_career(
                restored_model, restored_state, FIT, 0, limit=1
            )
        parent_cpu = time.process_time()
        parent_wall = time.perf_counter()
        worker = spawn_worker(
            {"bundle": str(bundle), "mode": "microstep", "out": str(out), "out_bundle": None}
        )
        parent_cpu = time.process_time() - parent_cpu
        parent_wall = time.perf_counter() - parent_wall
    assert worker["worker_cpu_seconds"] > 0.0
    assert worker["worker_wall_seconds"] > 0.0
    assert worker["worker_cpu_seconds"] <= worker["worker_wall_seconds"]
    assert parent_cpu < worker["worker_wall_seconds"]
    assert parent_wall >= worker["worker_wall_seconds"]
    assert torch.equal(worker["h"], local_state.h)
    assert torch.equal(worker["probabilities"], local_steps[0][0].probabilities)
