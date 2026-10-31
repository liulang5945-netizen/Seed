"""R027 subject-gate oracles: one life sequence, eight bundle boundaries."""

from __future__ import annotations

import copy
import multiprocessing
import tempfile
from dataclasses import replace
from pathlib import Path

import torch

from taiji import growth_credit as gc
from taiji.living_core import LivingCoreConfig, LivingCoreReference, RawSource
from taiji.native_activity import (
    ActivityEnvironment,
    ActivityPolicy,
    env_to_extra,
    run_round,
)
from taiji.state_migration import (
    CareerJournal,
    career_step_specs,
    load_subject,
    replay_transaction,
    resume_career,
    run_worker_job,
    save_subject,
)


def declared_update(model, state, optimizer):
    """One declared Adam step on the teacher-forced replay branch over FIT."""
    _, steps = resume_career(model, state, FIT, 0)
    terms = [output.probabilities[byte] for output, byte in steps if byte is not None]
    loss = -torch.stack(terms).log().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


CONFIG = LivingCoreConfig(groups=2, width=8, units=12, adapter_rank=2, key_width=4, value_width=4)
RECORDS = (
    RawSource("s1", b"abc target ab", version=1, observed_at=0),
    RawSource("s2", b"secondary ab", version=2, observed_at=4),
)
FIT = CareerJournal(
    subject_id="r027-subject",
    prompt=b"observe-context-" * 4,
    target=(b"target-bytes-ab" * 7)[:100],
    goal=b"growth-goal",
    records=RECORDS,
    raw_nu=(0.5, 2.0, 0.0, 0.8, 8.0, -0.4, 0.0, 0.3),
    known_nu=(True, True, False, True, True, True, False, True),
    now=10,
)
SPECS = career_step_specs(FIT)
GATE_TABLE = None


def spawn_worker(spec: dict) -> dict:
    out = spec["out"]
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=run_worker_job, args=(spec,))
    process.start()
    process.join(timeout=600)
    if process.exitcode != 0:
        raise RuntimeError(f"worker failed with exitcode {process.exitcode}")
    return torch.load(out, weights_only=True)


def save_life_bundle(
    path,
    model,
    state,
    policy,
    env,
    entries,
    *,
    cursor=0,
    optimizer=None,
    pending=None,
    growth_gates=None,
    fee_total=0.1,
    last_event=None,
    prediction_error=0.0,
):
    extra = {
        "activity_env": {
            **env_to_extra(env),
            "last_event": (
                [0.0] * CONFIG.width
                if last_event is None
                else [float(value) for value in last_event.detach()]
            ),
            "prediction_error": prediction_error,
        },
        "policy_state": policy.state_dict(),
        "fee_total": fee_total,
        "probe_exposures": 3,
    }
    return save_subject(
        path,
        model,
        state,
        career=FIT,
        journal_cursor=cursor,
        optimizer=optimizer,
        eligibility=torch.zeros(CONFIG.groups, dtype=torch.float64),
        journal_entries=entries,
        pending=pending,
        growth_gates=growth_gates,
        extra=extra,
    )


def detach_state(state):
    return replace(
        state,
        h=state.h.detach(),
        fast=state.fast.detach(),
        nu_ema=state.nu_ema.detach(),
        coverage=state.coverage.detach(),
        prior=state.prior.detach(),
    )


def test_r027_unified_life_sequence_gates():
    torch.set_num_threads(1)
    model = LivingCoreReference(CONFIG).double()
    state = model.initial_state("r027-subject")
    env = ActivityEnvironment(store=RECORDS, goal=b"growth-goal", now=10)
    policy = ActivityPolicy(CONFIG.width).double()
    entries: list = []
    last_event = torch.zeros(CONFIG.width, dtype=torch.float64)
    carried_error = 0.0
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)

        def gate(name, model_now, state_now, cursor=0, optimizer=None, gates=None, pending=None):
            path = tmp / f"bundle_{name}.pt"
            save_life_bundle(
                path,
                model_now,
                state_now,
                policy,
                env,
                entries,
                cursor=cursor,
                optimizer=optimizer,
                growth_gates=gates,
                last_event=last_event,
                prediction_error=carried_error,
                pending=pending,
            )
            return path

        # 1. after_interaction: host injects a message; the host-driven round consumes it.
        env.message_queue.append(b"external message one")
        state, env, entry, last_event, carried_error = run_round(
            model,
            state,
            env,
            policy,
            entries,
            last_event=last_event,
            prediction_error_value=carried_error,
            chooser=lambda env_, error: "interact",
        )
        bundle = gate("after_interaction", model, state)
        pre_env = copy.deepcopy(env)
        worker = spawn_worker(
            {
                "bundle": str(bundle),
                "mode": "activity_round",
                "out": str(tmp / "r1.pt"),
                "out_bundle": None,
            }
        )
        replica_state, replica_env, replica_entry, replica_event, replica_error = run_round(
            model,
            state,
            pre_env,
            policy,
            list(entries),
            last_event=last_event,
            prediction_error_value=carried_error,
        )
        assert worker["decision_entry"]["activity"] == replica_entry["activity"]
        assert worker["decision_entry"]["logits"] == replica_entry["logits"]
        assert torch.equal(worker["h"], replica_state.h)
        state, env = replica_state, replica_env
        last_event, carried_error = replica_event, replica_error
        entries.append(dict(replica_entry))

        # 2. no_prompt_activity: two endogenous model rounds with an empty queue.
        for _ in range(2):
            state, env, entry, last_event, carried_error = run_round(
                model,
                state,
                env,
                policy,
                entries,
                last_event=last_event,
                prediction_error_value=carried_error,
            )
            entries.append(dict(entry))
        assert all(not item.get("external_trigger") for item in entries[-2:])
        bundle = gate("no_prompt", model, state)
        worker = spawn_worker(
            {
                "bundle": str(bundle),
                "mode": "activity_round",
                "out": str(tmp / "r2.pt"),
                "out_bundle": None,
            }
        )
        pre_env = copy.deepcopy(env)
        replica_state, replica_env, replica_entry, replica_event, replica_error = run_round(
            model,
            state,
            pre_env,
            policy,
            list(entries),
            last_event=last_event,
            prediction_error_value=carried_error,
        )
        assert worker["decision_entry"]["logits"] == replica_entry["logits"]
        assert torch.equal(worker["h"], replica_state.h)
        state, env = replica_state, replica_env
        last_event, carried_error = replica_event, replica_error
        entries.append(dict(replica_entry))

        # 3. sleep_wait: forced wait round; H bitwise unchanged, clock advances.
        h_before = state.h.detach().clone()
        clock_before = env.now
        state, env, wait_entry, last_event, _ = run_round(
            model,
            state,
            env,
            policy,
            entries,
            last_event=last_event,
            prediction_error_value=0.0,
            chooser=lambda env_, error: "wait",
        )
        entries.append(dict(wait_entry))
        assert torch.equal(state.h, h_before) and env.now == clock_before + 1
        bundle = gate("sleep_wait", model, state)
        worker = spawn_worker(
            {
                "bundle": str(bundle),
                "mode": "activity_round",
                "out": str(tmp / "r3.pt"),
                "out_bundle": None,
            }
        )
        pre_env = copy.deepcopy(env)
        replica_state, replica_env, replica_entry, replica_event, replica_error = run_round(
            model,
            state,
            pre_env,
            policy,
            list(entries),
            last_event=last_event,
            prediction_error_value=carried_error,
        )
        assert worker["decision_entry"]["logits"] == replica_entry["logits"]
        assert torch.equal(worker["h"], replica_state.h)
        state, env = replica_state, replica_env
        last_event, carried_error = replica_event, replica_error
        entries.append(dict(replica_entry))

        # 4. first_learning: declared update, then worker repeats it bitwise.
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, betas=(0.9, 0.999), eps=1e-8)
        saved_state = detach_state(state)
        declared_update(model, saved_state, optimizer)
        bundle = gate("first_learning", model, state, optimizer=optimizer)
        worker = spawn_worker(
            {
                "bundle": str(bundle),
                "mode": "update",
                "out": str(tmp / "r4.pt"),
                "out_bundle": None,
            }
        )
        next_loss = declared_update(model, saved_state, optimizer)
        assert next_loss > 0.0
        for name, parameter in model.named_parameters():
            assert torch.equal(parameter.detach().cpu(), worker["parameters_after"][name]), name

        # 5/6. before_birth and after_birth around the R023 gated birth.
        bundle = gate("before_birth", model, state)
        book, _ = gc.collect_credit(model, FIT)
        steps = len(book.deltas)
        fit = range(0, steps // 2)
        {n: gc.branch_credit(book, b) for b, n in enumerate(book.branch_ids)}
        fit_credits = {n: gc.branch_credit(book, b, fit) for b, n in enumerate(book.branch_ids)}
        check = range(steps // 2, steps)
        verdicts = {}
        for b, n in enumerate(book.branch_ids):
            tau_noise = gc.calibrate_tau_noise(book, b, fit)
            verdicts[n] = gc.branch_verdict(book, b, tau_noise, fit, check)
        child, provider, plan = build_birth(model, book, fit_credits, verdicts)
        assert plan["units_total"] > 0
        gate_table = [[m, 1.0 - m] for m, _ in book.gates]
        bundle = gate("after_birth", child, state, gates=gate_table)
        worker = spawn_worker(
            {
                "bundle": str(bundle),
                "mode": "activity_round",
                "out": str(tmp / "r6.pt"),
                "out_bundle": None,
            }
        )
        pre_env = copy.deepcopy(env)
        replica_state, replica_env, replica_entry, replica_event, replica_error = run_round(
            child,
            state,
            pre_env,
            policy,
            list(entries),
            last_event=last_event,
            prediction_error_value=carried_error,
            gate_provider=provider,
            gate_index=0,
        )
        assert worker["decision_entry"]["activity"] == replica_entry["activity"]
        assert torch.equal(worker["h"], replica_state.h)
        model, state, env = child, replica_state, replica_env
        last_event, carried_error = replica_event, replica_error
        entries.append(dict(replica_entry))

        # 7. after_migration: encoder revision bump committed, then restore.
        def coordinate_change(shadow):
            generator = torch.Generator().manual_seed(20261031)
            noise = torch.randn(shadow.encoder.input.weight.shape, generator=generator) * 1e-3
            shadow.encoder.input.weight = torch.nn.Parameter(
                shadow.encoder.input.weight.detach() + noise
            )

        committed = replay_transaction(
            model, state, FIT, coordinate_change, growth_gate_provider=provider
        )
        assert committed["outcome"] == "committed"
        model = committed["committed_model"]
        state = committed["committed_state"]
        bundle = gate("after_migration", model, state, gates=gate_table)
        worker = spawn_worker(
            {
                "bundle": str(bundle),
                "mode": "activity_round",
                "out": str(tmp / "r7.pt"),
                "out_bundle": None,
            }
        )
        pre_env = copy.deepcopy(env)
        replica_state, replica_env, replica_entry, replica_event, replica_error = run_round(
            model,
            state,
            pre_env,
            policy,
            list(entries),
            last_event=last_event,
            prediction_error_value=carried_error,
            gate_provider=provider,
            gate_index=0,
        )
        assert torch.equal(worker["h"], replica_state.h)
        assert worker["decision_entry"]["logits"] == replica_entry["logits"]
        state, env = replica_state, replica_env
        last_event, carried_error = replica_event, replica_error
        entries.append(dict(replica_entry))
        restored = load_subject(bundle)
        assert restored["revision"] == 1

        # 8. pending_outcome: resolution is guarded by the current journal.
        bundle = gate(
            "pending", model, state, pending={"outcome_id": "o-9", "reward": 0.5, "resolved": False}
        )
        worker = spawn_worker(
            {
                "bundle": str(bundle),
                "mode": "resolve",
                "out": str(tmp / "r8.pt"),
                "out_bundle": None,
                "current_entries": [],
            }
        )
        assert worker["resolved"] is True and worker["reward_delta"] == 0.5
        updated = worker["journal_entries_out"]
        second = spawn_worker(
            {
                "bundle": str(bundle),
                "mode": "resolve",
                "out": str(tmp / "r8b.pt"),
                "out_bundle": None,
                "current_entries": updated,
            }
        )
        assert second["resolved"] is False and second["reward_delta"] == 0.0


def build_birth(parent, book, fit_credits, verdicts):
    allocation = {}
    for group in range(parent.config.groups):
        alive, caps = {}, {}
        for name, verdict in verdicts.items():
            entry = verdict["groups"][group]
            if entry["enter_V"]:
                from taiji.growth_credit import rho_curve

                alive[name] = rho_curve(torch.tensor(entry["sigma"]), entry["frobenius"])
                caps[name] = entry["rank_num"]
        from taiji.growth_credit import allocate

        for name, units in allocate(alive, caps)["allocation"].items():
            allocation[name] = units
    directions = {group: [] for group in range(parent.config.groups)}
    unit_branches = {group: [] for group in range(parent.config.groups)}
    for name, units in sorted(allocation.items()):
        book.branch_ids.index(name)
        for group in range(parent.config.groups):
            _, _, vectors = torch.linalg.svd(fit_credits[name][group], full_matrices=False)
            directions[group].extend(vectors[:, j].detach().clone() for j in range(units))
            unit_branches[group].extend([name] * units)
    from taiji.growth_credit import GatedGrowthCore

    child = GatedGrowthCore(parent.config).to(dtype=parent.W.dtype)
    child.load_state_dict(parent.state_dict())
    parent_r, parent_p = parent.R.detach().clone(), parent.P.detach().clone()
    max_extra = max(len(value) for value in directions.values())
    grown_r, grown_p = [], []
    for group in range(parent.config.groups):
        zeros = torch.zeros(max_extra, parent_r.shape[2], dtype=parent_r.dtype)
        grown_r.append(torch.cat([parent_r[group], zeros], dim=0))
        columns = list(directions[group])
        columns.extend(
            torch.zeros(parent_p.shape[1], dtype=parent_p.dtype)
            for _ in range(max_extra - len(columns))
        )
        grown_p.append(torch.cat([parent_p[group], torch.stack(columns, dim=1)], dim=1))
    child.R = torch.nn.Parameter(torch.stack(grown_r, dim=0))
    child.P = torch.nn.Parameter(torch.stack(grown_p, dim=0))
    child.growth_parent_units = parent_r.shape[1]
    child.growth_unit_branches = [unit_branches[group] for group in range(parent.config.groups)]

    def provider(index):
        item = book.gates[index % len(book.gates)]
        return {"copy": item[0], "generate": 1.0 - item[1]}

    return child, provider, {"units_total": max_extra * parent.config.groups}


def test_r027_illegal_source_filtered_and_isolation():
    torch.set_num_threads(1)
    model = LivingCoreReference(CONFIG).double()
    state = model.initial_state("r027-subject")
    env = ActivityEnvironment(store=RECORDS, goal=b"growth-goal", now=10)
    foreign = RawSource("foreign", b"foreign bytes", version=1, observed_at=0, owner="other")
    env.store = tuple(env.store) + (foreign,)
    policy = ActivityPolicy(CONFIG.width).double()
    journal: list = []
    state, env, entry, _, _ = run_round(
        model,
        state,
        env,
        policy,
        journal,
        last_event=torch.zeros(CONFIG.width, dtype=torch.float64),
        prediction_error_value=0.0,
    )
    assert "foreign" in entry["note"]["rejected_sources"]
    assert all(record.source_id != "foreign" for record in env.store)
    # query-only isolation: a probe on a detached copy leaves the subject bitwise.
    snapshot = detach_state(state)
    probe_state, _ = resume_career(model, detach_state(state), FIT, 0, limit=5)
    assert probe_state.step == snapshot.step + 5
    assert torch.equal(state.h, snapshot.h)
    assert torch.equal(state.fast, snapshot.fast)
    assert state.coverage is snapshot.coverage or torch.equal(state.coverage, snapshot.coverage)
