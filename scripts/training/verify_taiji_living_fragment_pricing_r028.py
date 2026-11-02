"""R028 life-fragment pricing on the R021/R027 subject engineering path.

Measures real CPU/wall costs for forward, declared updates, activity rounds,
credit calibration, gated birth, migration, save/restore and the six-arm
development schedule. Synthetic fixtures only: no task split, no persistent
service and no product-default change. This report prices a candidate but does
not authorize training.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from taiji import growth_credit as gc  # noqa: E402
from taiji import native_development as development  # noqa: E402
from taiji.living_core import LivingCoreConfig, LivingCoreReference, RawSource  # noqa: E402
from taiji.native_activity import ActivityEnvironment, ActivityPolicy, run_round  # noqa: E402
from taiji.state_migration import (  # noqa: E402
    CareerJournal,
    MigrationBudget,
    load_subject,
    replay_transaction,
    resume_career,
    resume_subject,
    run_career,
    run_worker_job,
    save_subject,
)

SMALL = LivingCoreConfig(groups=2, width=8, units=12, adapter_rank=2, key_width=4, value_width=4)
PRINCIPLE = LivingCoreConfig()
RECORDS = (
    RawSource("s1", "甲在桌上。".encode(), version=1, observed_at=0),
    RawSource("s2", "乙在抽屉。".encode(), version=2, observed_at=4),
)
NU = (0.5, 2.0, 0.0, 0.8, 8.0, -0.4, 0.0, 0.3)
KNOWN = (True, True, False, True, True, True, False, True)
CHINESE_GOAL = "现在在哪里？".encode()
CHINESE_LONG_PROMPT = ("当前对象：蓝色杯子在木桌上。后来更正：蓝色杯子被移到窗台。" * 6).encode()
CHINESE_LONG_TARGET = "蓝色杯子现在在窗台。".encode() * 2
SHORT = CareerJournal(
    subject_id="r028-short",
    prompt=CHINESE_LONG_PROMPT[:32],
    target=CHINESE_LONG_TARGET[:12],
    goal=CHINESE_GOAL,
    records=RECORDS,
    raw_nu=NU,
    known_nu=KNOWN,
    now=10,
)
LONG = CareerJournal(
    subject_id="r028-long",
    prompt=CHINESE_LONG_PROMPT,
    target=CHINESE_LONG_TARGET,
    goal=CHINESE_GOAL,
    records=RECORDS,
    raw_nu=NU,
    known_nu=KNOWN,
    now=10,
)
HOLDOUT = CareerJournal(
    subject_id="r028-holdout",
    prompt=("未见材料：绿色钥匙挂在门后。随后更正：绿色钥匙在口袋。" * 6).encode(),
    target="绿色钥匙现在在口袋。".encode() * 2,
    goal=CHINESE_GOAL,
    records=RECORDS,
    raw_nu=NU,
    known_nu=KNOWN,
    now=10,
)
RETENTION = CareerJournal(
    subject_id="r028-retention",
    prompt=("旧约定：红色印章放在左盒。补充说明：红色印章位置未变。" * 6).encode(),
    target="红色印章在左盒。".encode() * 2,
    goal=CHINESE_GOAL,
    records=RECORDS,
    raw_nu=NU,
    known_nu=KNOWN,
    now=10,
)
BUDGET_CPU_SECONDS = 1800
BUDGET_WALL_SECONDS = 1800


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timed(function):
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    value = function()
    return value, time.process_time() - cpu_start, time.perf_counter() - wall_start


def measurement(function, repetitions=1):
    samples = []
    values = []
    for _ in range(repetitions):
        value, cpu, wall = timed(function)
        values.append(value)
        samples.append({"cpu_seconds": cpu, "wall_seconds": wall})
    return {
        "repetitions": repetitions,
        "samples": samples,
        "total_cpu_seconds": sum(row["cpu_seconds"] for row in samples),
        "total_wall_seconds": sum(row["wall_seconds"] for row in samples),
        "mean_cpu_seconds": sum(row["cpu_seconds"] for row in samples) / len(samples),
        "mean_wall_seconds": sum(row["wall_seconds"] for row in samples) / len(samples),
        "values": values,
    }


def fresh(config: LivingCoreConfig, subject: str):
    model = LivingCoreReference(config).double()
    return model, model.initial_state(subject)


def career_forward(config, subject, journal, repetitions):
    model, _ = fresh(config, subject)

    def once():
        with torch.no_grad():
            state, steps = run_career(model, journal)
        return {"step": state.step, "events": len(steps)}

    return measurement(once, repetitions)


def career_update(config, subject, journal, repetitions):
    model, _ = fresh(config, subject)
    optimizer = torch.optim.Adam(model.parameters(), lr=development.UPDATE_LR)

    def once():
        _, steps = run_career(model, journal)
        terms = [output.probabilities[byte] for output, byte in steps if byte is not None]
        loss = -torch.stack(terms).log().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    return measurement(once, repetitions)


def activity_measurement(config, activity: str):
    model, state = fresh(config, f"activity-{activity}")
    env = ActivityEnvironment(store=RECORDS, goal=CHINESE_GOAL, now=10)
    policy = ActivityPolicy(config.width).double()
    entries: list = []
    message = None
    if activity == "interact":
        message = CHINESE_LONG_TARGET
        env.message_queue.append(message)

    def once():
        with torch.no_grad():
            result = run_round(
                model,
                state,
                env,
                policy,
                entries,
                last_event=torch.zeros(config.width, dtype=torch.float64),
                prediction_error_value=0.0,
                chooser=lambda _env, _error: activity,
            )
        entry = result[2]
        return {"activity": entry["activity"], "step_delta": entry["step_delta"]}

    return measurement(once, 1)


def birth_measurement(config):
    model, _ = fresh(config, "birth-pricing")
    (book, state), collect_cpu, collect_wall = timed(lambda: gc.collect_credit(model, LONG))
    steps = len(book.deltas)
    fit_window = range(steps // 2)
    check_window = range(steps // 2, steps)

    def calibration():
        values = {}
        samples = []
        for branch, name in enumerate(book.branch_ids):
            tau, cpu, wall = timed(
                lambda branch=branch: gc.calibrate_tau_noise(book, branch, fit_window)
            )
            values[name] = tau
            samples.append({"branch": name, "cpu_seconds": cpu, "wall_seconds": wall})
        return values, samples

    tau_map, calibration_samples = calibration()

    def credits():
        full, fit = {}, {}
        for branch, name in enumerate(book.branch_ids):
            full[name] = gc.branch_credit(book, branch)
            fit[name] = gc.branch_credit(book, branch, fit_window)
        return full, fit

    (full_credits, fit_credits), credits_cpu, credits_wall = timed(credits)

    def verdicts():
        return {
            name: gc.branch_verdict(book, branch, tau_map[name], fit_window, check_window)
            for branch, name in enumerate(book.branch_ids)
        }

    verdict_table, verdict_cpu, verdict_wall = timed(verdicts)

    def allocate():
        proposals, allocation = {}, {}
        for group in range(config.groups):
            alive, caps = {}, {}
            for name, verdict in verdict_table.items():
                entry = verdict["groups"][group]
                if entry["enter_V"]:
                    alive[name] = gc.rho_curve(torch.tensor(entry["sigma"]), entry["frobenius"])
                    caps[name] = entry["rank_num"]
            proposals[group] = gc.allocate(alive, caps)
            for name, units in proposals[group]["allocation"].items():
                allocation[name] = units
        return proposals, allocation

    (proposals, allocation), allocation_cpu, allocation_wall = timed(allocate)

    def directions():
        values = {}
        for group in range(config.groups):
            for name, units in sorted(allocation.items()):
                _, _, vectors = torch.linalg.svd(fit_credits[name][group], full_matrices=False)
                values[(name, group)] = [vectors[:, index].detach() for index in range(units)]
        return values

    direction_table, direction_cpu, direction_wall = timed(directions)

    def build_child():
        return gc.birth(model, direction_table, full_credits)

    (child, plan), birth_cpu, birth_wall = timed(build_child)
    startup, startup_cpu, startup_wall = timed(lambda: gc.startup_report(plan))

    def gate_provider(index):
        values = book.gates[index % len(book.gates)]
        return {name: float(values[i]) for i, name in enumerate(book.branch_ids)}

    def neutral_compare():
        with torch.no_grad():
            parent_state, parent_steps = run_career(model, LONG)
            child_state, child_steps = run_career(child, LONG, growth_gate_provider=gate_provider)
        return bool(
            all(
                torch.equal(left.probabilities, right.probabilities)
                for (left, _), (right, _) in zip(parent_steps, child_steps, strict=True)
            )
            and torch.equal(parent_state.h, child_state.h)
            and torch.equal(parent_state.fast, child_state.fast)
        )

    neutral_equal, neutral_cpu, neutral_wall = timed(neutral_compare)
    return {
        "events": LONG.event_count,
        "journal_bytes": LONG.journal_bytes,
        "collect_credit": {"cpu_seconds": collect_cpu, "wall_seconds": collect_wall},
        "noise_calibration": {
            "permutations": gc.PERMUTATIONS,
            "quantile": gc.NOISE_QUANTILE,
            "samples": calibration_samples,
            "total_cpu_seconds": sum(row["cpu_seconds"] for row in calibration_samples),
            "total_wall_seconds": sum(row["wall_seconds"] for row in calibration_samples),
        },
        "branch_credits": {"cpu_seconds": credits_cpu, "wall_seconds": credits_wall},
        "signal_verdicts": {"cpu_seconds": verdict_cpu, "wall_seconds": verdict_wall},
        "allocation": {
            "cpu_seconds": allocation_cpu,
            "wall_seconds": allocation_wall,
            "states_checked": {str(k): v["states_checked"] for k, v in proposals.items()},
            "units_per_group": plan["units_per_group"],
            "padded_units_per_group": plan["padded_units_per_group"],
        },
        "directions": {"cpu_seconds": direction_cpu, "wall_seconds": direction_wall},
        "birth": {"cpu_seconds": birth_cpu, "wall_seconds": birth_wall},
        "startup": {
            "cpu_seconds": startup_cpu,
            "wall_seconds": startup_wall,
            "l_start": startup["l_start"],
            "units": [row["unit"] for row in startup["units"]],
        },
        "neutral_forward": {
            "cpu_seconds": neutral_cpu,
            "wall_seconds": neutral_wall,
            "bitwise_equal": neutral_equal,
        },
        "book_state": {"state_step": state.step, "round_count": book.round_count},
    }


def arm_measurement(config, book, fit_credits, full_credits, verdict_table):
    reference, _ = fresh(config, "arms-reference")

    def coverage_once():
        return development.coverage_birth_arm(reference, book, fit_credits, verdict_table)

    (_, _, coverage_plan), unit_cpu, unit_wall = timed(coverage_once)
    unit_total = coverage_plan["units_total"]

    def optimizer_factory(model):
        return torch.optim.Adam(
            model.parameters(), lr=development.UPDATE_LR, betas=(0.9, 0.999), eps=1e-8
        )

    builders = {
        "replay_only": None,
        "coverage_birth": lambda model: development.coverage_birth_arm(
            model, book, fit_credits, verdict_table
        ),
        "gradmax_schedule": lambda model: development.gradmax_arm(model, full_credits),
        "conditional_svd": lambda model: development.conditional_svd_arm(
            model, book, fit_credits, verdict_table
        ),
        "random_gate": lambda model: development.random_gate_arm(model, book, unit_total),
        "capacity_random": lambda model: development.capacity_random_arm(model, unit_total),
    }
    results = []
    for name, builder in builders.items():
        result, cpu, wall = timed(
            lambda name=name, builder=builder: development.run_arm(
                name,
                copy.deepcopy(reference),
                LONG,
                HOLDOUT,
                RETENTION,
                builder,
                optimizer_factory,
            )
        )
        losses = result["update_losses"]
        results.append(
            {
                "arm": name,
                "units_total": result["units_total"],
                "allocation": result.get("allocation"),
                "nll_fit_pre": result["nll_fit_pre"],
                "nll_fit_post": result["nll_fit_post"],
                "delta_nll_holdout": result["delta_nll_holdout"],
                "drift_retention": result["drift_retention"],
                "first_update_loss": losses[0],
                "last_update_loss": losses[-1],
                "internal_cpu_seconds": result["cpu_seconds"],
                "internal_wall_seconds": result["wall_seconds"],
                "external_cpu_seconds": cpu,
                "external_wall_seconds": wall,
            }
        )
    return {
        "unit_probe": {
            "units_total": unit_total,
            "cpu_seconds": unit_cpu,
            "wall_seconds": unit_wall,
        },
        "updates_per_arm": development.UPDATE_STEPS,
        "arms": results,
        "total_cpu_seconds": sum(row["external_cpu_seconds"] for row in results),
        "total_wall_seconds": sum(row["external_wall_seconds"] for row in results),
    }


def migration_measurement(config):
    model, _ = fresh(config, "migration-pricing")
    with torch.no_grad():
        state, _ = run_career(model, SHORT)

    identity, identity_cpu, identity_wall = timed(lambda: replay_transaction(model, state, SHORT))

    def change_encoder(candidate):
        candidate.encoder.input.weight.detach().mul_(1.001)

    coordinate, coordinate_cpu, coordinate_wall = timed(
        lambda: replay_transaction(model, state, SHORT, coordinate_change=change_encoder)
    )
    refused, refused_cpu, refused_wall = timed(
        lambda: replay_transaction(model, state, SHORT, budget=MigrationBudget(max_events=0))
    )
    return {
        "identity_replay": {
            "outcome": identity["outcome"],
            "bitwise_equal": identity["state_bitwise_equal"],
            "cpu_seconds": identity_cpu,
            "wall_seconds": identity_wall,
            "events": identity["events"],
        },
        "coordinate_commit": {
            "outcome": coordinate["outcome"],
            "coordinate_changed": coordinate["coordinate_change_applied"],
            "no_parameter_mutation_during_replay": coordinate["no_replay_parameter_mutation"],
            "cpu_seconds": coordinate_cpu,
            "wall_seconds": coordinate_wall,
        },
        "budget_refusal": {
            "outcome": refused["outcome"],
            "reason": refused["reason"],
            "cpu_seconds": refused_cpu,
            "wall_seconds": refused_wall,
        },
    }


def spawn_worker(spec: dict) -> dict:
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=run_worker_job, args=(spec,))
    process.start()
    process.join(timeout=600)
    if process.exitcode != 0:
        raise RuntimeError(f"restore worker exitcode={process.exitcode}")
    return torch.load(spec["out"], weights_only=True)


def save_restore_measurement(config):
    model, initial_state = fresh(config, "restore-pricing")
    optimizer = torch.optim.Adam(model.parameters(), lr=development.UPDATE_LR)

    def perform_update():
        _, steps = resume_career(model, initial_state, SHORT, 0)
        terms = [output.probabilities[byte] for output, byte in steps if byte is not None]
        loss = -torch.stack(terms).log().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return steps[-1][0].state

    state, update_cpu, update_wall = timed(perform_update)
    entries = [{"seq": 0, "kind": "pricing_seed"}]

    def save_once():
        return save_subject(
            path,
            model,
            state,
            career=SHORT,
            journal_cursor=1,
            optimizer=optimizer,
            eligibility=torch.zeros(config.groups, dtype=torch.float64),
            journal_entries=entries,
            pending={"outcome_id": "pricing", "reward": 0.1, "resolved": False},
        )

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "bundle.pt"
        bundle_sha, bundle_cpu, bundle_wall = timed(save_once)
        file_bytes = path.stat().st_size
        bundle, load_cpu, load_wall = timed(lambda: load_subject(path))
        restored, resume_cpu, resume_wall = timed(lambda: resume_subject(bundle))
        restore_model, restore_state, _ = restored
        with torch.no_grad():
            local_state, local_steps = resume_career(
                restore_model, restore_state, SHORT, 1, limit=1
            )
        worker_path = Path(directory) / "worker.pt"

        def worker_once():
            return spawn_worker(
                {
                    "bundle": str(path),
                    "mode": "microstep",
                    "out": str(worker_path),
                    "out_bundle": None,
                }
            )

        worker, worker_cpu, worker_wall = timed(worker_once)
        local_output = local_steps[0][0]
        return {
            "career_events": SHORT.event_count,
            "declared_update_for_bundle": {
                "cpu_seconds": update_cpu,
                "wall_seconds": update_wall,
            },
            "save_subject": {
                "cpu_seconds": bundle_cpu,
                "wall_seconds": bundle_wall,
                "bundle_sha256": bundle_sha,
                "bundle_bytes": file_bytes,
            },
            "in_process_load": {
                "cpu_seconds": load_cpu,
                "wall_seconds": load_wall,
            },
            "in_process_resume": {
                "cpu_seconds": resume_cpu,
                "wall_seconds": resume_wall,
            },
            "fresh_spawn_worker": {
                "cpu_seconds": worker_cpu,
                "wall_seconds": worker_wall,
                "h_bitwise_equal": bool(torch.equal(worker["h"], local_state.h)),
                "probabilities_bitwise_equal": bool(
                    torch.equal(worker["probabilities"], local_output.probabilities)
                ),
            },
        }


def safe_model_count(config):
    model, _ = fresh(config, "count")
    return sum(parameter.numel() for parameter in model.parameters())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("R028 pricing report is write-once")

    torch.set_num_threads(1)
    total_cpu, total_wall = time.process_time(), time.perf_counter()
    baseline = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    birth = birth_measurement(PRINCIPLE)
    arms_model, _ = fresh(PRINCIPLE, "arms-reference")
    book, _ = gc.collect_credit(arms_model, LONG)
    steps = len(book.deltas)
    fit_window = range(steps // 2)
    check_window = range(steps // 2, steps)
    tau_map = {
        name: gc.calibrate_tau_noise(book, branch, fit_window)
        for branch, name in enumerate(book.branch_ids)
    }
    full_credits, fit_credits = {}, {}
    for branch, name in enumerate(book.branch_ids):
        full_credits[name] = gc.branch_credit(book, branch)
        fit_credits[name] = gc.branch_credit(book, branch, fit_window)
    verdict_table = {
        name: gc.branch_verdict(book, branch, tau_map[name], fit_window, check_window)
        for branch, name in enumerate(book.branch_ids)
    }
    arms = arm_measurement(PRINCIPLE, book, fit_credits, full_credits, verdict_table)

    workloads = {
        "small_short_forward": career_forward(SMALL, "small-short", SHORT, 3),
        "principle_short_forward": career_forward(PRINCIPLE, "principle-short", SHORT, 3),
        "small_short_update": career_update(SMALL, "small-update", SHORT, 3),
        "principle_short_update": career_update(PRINCIPLE, "principle-update", SHORT, 3),
        "principle_long_forward": career_forward(PRINCIPLE, "principle-long", LONG, 3),
        "principle_long_update": career_update(PRINCIPLE, "principle-update-long", LONG, 1),
        "activity_rounds": {
            name: activity_measurement(PRINCIPLE, name)
            for name in ("interact", "recall", "imagine", "explore", "maintain", "wait")
        },
    }
    save_restore = save_restore_measurement(PRINCIPLE)
    migration = migration_measurement(PRINCIPLE)
    for section in (workloads, birth, arms, save_restore, migration):
        for value in section.values():
            if isinstance(value, dict) and "values" in value:
                value.pop("values")

    observed_updates = (
        3
        + 3
        + 1
        + 6 * development.UPDATE_STEPS
        + 1
        # arms unit probe builds no optimizer; the long update adds one.
    )
    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    checks = [
        {
            "id": "birth_neutral_bitwise",
            "passed": birth["neutral_forward"]["bitwise_equal"],
        },
        {
            "id": "arm_results_finite",
            "passed": all(
                value == value
                for row in arms["arms"]
                for value in (row["delta_nll_holdout"], row["drift_retention"])
            ),
        },
        {
            "id": "identity_migration_bitwise",
            "passed": migration["identity_replay"]["bitwise_equal"],
        },
        {
            "id": "fresh_restore_bitwise",
            "passed": save_restore["fresh_spawn_worker"]["h_bitwise_equal"]
            and save_restore["fresh_spawn_worker"]["probabilities_bitwise_equal"],
        },
        {
            "id": "budget_not_exceeded",
            "passed": cpu_seconds <= BUDGET_CPU_SECONDS and wall_seconds <= BUDGET_WALL_SECONDS,
        },
    ]
    status = (
        "passed_r028_pricing_engineering_only"
        if all(c["passed"] for c in checks)
        else "engineering_failure"
    )
    report = {
        "schema": "taiji-r028-life-fragment-pricing-v1",
        "date": "2026-11-01",
        "baseline_commit": baseline,
        "status": status,
        "configs": {
            "small_reference": asdict(SMALL),
            "principle_reference": asdict(PRINCIPLE),
            "principle_parameters": safe_model_count(PRINCIPLE),
        },
        "natural_fixture": {
            "prompt_bytes": len(CHINESE_LONG_PROMPT),
            "target_bytes": len(CHINESE_LONG_TARGET),
            "events": LONG.event_count,
            "note": "UTF-8 byte workload prices Chinese text; it does not claim semantic correctness.",
        },
        "workloads": workloads,
        "birth_pipeline": birth,
        "six_arm_pilot": arms,
        "save_restore": save_restore,
        "migration": migration,
        "optimizer_updates_observed": observed_updates,
        "data_splits_read": [],
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "checks": checks,
        "training_admitted": False,
        "task_capability_claimed": False,
        "default_model_replaced": False,
        "source_hashes": {
            relative: digest(ROOT / relative)
            for relative in (
                "taiji/living_core.py",
                "taiji/growth_credit.py",
                "taiji/native_development.py",
                "taiji/native_activity.py",
                "taiji/state_migration.py",
                "scripts/training/verify_taiji_living_fragment_pricing_r028.py",
            )
        },
        "limits": "Engineering pricing only. Costs depend on the fixed CPU and thread setting; no task performance, retention, growth benefit, novelty or training admission is established.",
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    print(
        json.dumps(
            {
                "status": status,
                "cpu_seconds": round(cpu_seconds, 3),
                "wall_seconds": round(wall_seconds, 3),
                "failed_checks": [c["id"] for c in checks if not c["passed"]],
            },
            ensure_ascii=True,
        )
    )
    return int(status != "passed_r028_pricing_engineering_only")


if __name__ == "__main__":
    raise SystemExit(main())
