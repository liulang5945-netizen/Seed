"""R024 subject restore oracles: bundle boundaries, independent-process continuation."""

from __future__ import annotations

import multiprocessing
import tempfile
from dataclasses import replace
from pathlib import Path

import torch

from taiji import growth_credit as gc
from taiji.living_core import LivingCoreConfig, LivingCoreReference, RawSource
from taiji.state_migration import (
    CareerJournal,
    append_journal,
    career_step_specs,
    eligibility_step,
    journal_has_resolution,
    load_subject,
    replay_transaction,
    resume_career,
    run_career,
    run_worker_job,
    save_subject,
)

CONFIG = LivingCoreConfig(groups=2, width=8, units=12, adapter_rank=2, key_width=4, value_width=4)
JOURNAL = CareerJournal(
    subject_id="r024-subject",
    prompt=b"observe-context-" * 4,
    target=(b"target-bytes-ab" * 7)[:100],
    goal=b"growth-goal",
    records=(
        RawSource("s1", b"abc target ab", version=1, observed_at=0),
        RawSource("s2", b"secondary ab", version=2, observed_at=4),
    ),
    raw_nu=(0.5, 2.0, 0.0, 0.8, 8.0, -0.4, 0.0, 0.0),
    known_nu=(True, True, False, True, True, True, False, True),
    now=10,
)
SPECS = career_step_specs(JOURNAL)
FEE_PER_STEP = 1e-3
LEARN_BOUNDARY = 54


def new_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=1e-2, betas=(0.9, 0.999), eps=1e-8)


def gate_provider(table):
    return lambda index: {"copy": table[index][0], "generate": table[index][1]}


def declared_update(model, state, optimizer):
    """One declared Adam step on the teacher-forced replay branch."""
    _, steps = resume_career(model, state, JOURNAL, 0)
    terms = [output.probabilities[byte] for output, byte in steps if byte is not None]
    loss = -torch.stack(terms).log().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def spawn_worker(spec: dict) -> dict:
    """Run the worker job in a fresh interpreter process (multiprocessing spawn).

    The target is the importable taiji.state_migration.run_worker_job itself, so
    the spawned interpreter resolves it by name. Results travel through a file:
    torch tensors cannot cross process queues on this platform's shared stack."""
    out = spec["out"]
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=run_worker_job, args=(spec,))
    process.start()
    process.join(timeout=600)
    if process.exitcode != 0:
        raise RuntimeError(f"worker failed with exitcode {process.exitcode}")
    return torch.load(out, weights_only=True)


def test_r024_no_input_boundary_preserves_idle_subject():
    torch.set_num_threads(1)
    model = LivingCoreReference(CONFIG).double()
    state, _ = run_career(model, JOURNAL)
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        bundle_path = tmp / "idle_bundle.pt"
        save_subject(
            bundle_path,
            model,
            state,
            career=JOURNAL,
            journal_cursor=len(SPECS),
            eligibility=torch.zeros(CONFIG.groups, dtype=torch.float64),
            journal_entries=[{"seq": 0, "kind": "fee", "amount": 0.25}],
            pending=None,
            extra={"fee_total": 0.25, "probe_exposures": 7},
        )
        spec = {
            "bundle": str(bundle_path),
            "mode": "microstep",
            "out": str(tmp / "idle_result.pt"),
            "out_bundle": None,
            "fee_per_step": FEE_PER_STEP,
        }
        worker = spawn_worker(spec)
        restored = load_subject(bundle_path)
    assert worker["consumed"] == 0 and worker["cursor_out"] == len(SPECS)
    assert worker["subject_id"] == "r024-subject" and worker["revision"] == 0
    assert torch.equal(restored["h"], state.h)
    assert worker["fee_total"] == 0.25
    torch.set_rng_state(restored["rng_state"])
    assert torch.equal(torch.rand(3), worker["rng_probe"])


def test_r024_first_learning_boundary_update_and_microstep():
    torch.set_num_threads(1)
    model = LivingCoreReference(CONFIG).double()
    optimizer = new_optimizer(model)
    state_mid, _ = resume_career(
        model, model.initial_state(JOURNAL.subject_id), JOURNAL, 0, limit=LEARN_BOUNDARY
    )
    loss_value = declared_update(model, state_mid, optimizer)
    assert loss_value > 0.0
    eligibility = torch.zeros(CONFIG.groups, dtype=torch.float64)
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        bundle_path = tmp / "learned_bundle.pt"
        save_subject(
            bundle_path,
            model,
            state_mid,
            career=JOURNAL,
            journal_cursor=LEARN_BOUNDARY,
            optimizer=optimizer,
            eligibility=eligibility,
            extra={"fee_total": 0.05},
        )
        update_result = spawn_worker(
            {
                "bundle": str(bundle_path),
                "mode": "update",
                "out": str(tmp / "update_result.pt"),
                "out_bundle": None,
                "fee_per_step": FEE_PER_STEP,
            }
        )
        micro_result = spawn_worker(
            {
                "bundle": str(bundle_path),
                "mode": "microstep",
                "out": str(tmp / "micro_result.pt"),
                "out_bundle": None,
                "fee_per_step": FEE_PER_STEP,
            }
        )
        # The worker ran the NEXT update from the restored bundle; the parent runs
        # the same next update locally and compares parameters bitwise. The saved
        # state is detached first: the probe career's graph was freed by the first
        # update. The expected microstep is captured against the post-first-update
        # model before the parent's own second update changes it.
        expected_state, expected_steps = resume_career(
            model, state_mid, JOURNAL, LEARN_BOUNDARY, limit=1
        )
        expected_probabilities = expected_steps[0][0].probabilities.detach().clone()
        state_saved = replace(
            state_mid,
            h=state_mid.h.detach(),
            fast=state_mid.fast.detach(),
            nu_ema=state_mid.nu_ema.detach(),
            coverage=state_mid.coverage.detach(),
            prior=state_mid.prior.detach(),
        )
        next_loss = declared_update(model, state_saved, optimizer)
        assert next_loss > 0.0
        assert update_result["post_update_consumed"] is True
        for name, parameter in model.named_parameters():
            assert torch.equal(
                parameter.detach().cpu(), update_result["parameters_after"][name]
            ), name
        bundle_data = load_subject(bundle_path)
        parent_moment = max(
            (
                float(moment["exp_avg"].abs().max())
                for moment in optimizer.state.values()
                if isinstance(moment, dict) and "exp_avg" in moment
            ),
            default=0.0,
        )
        assert parent_moment == update_result["moment_max"]
        torch.set_rng_state(bundle_data["rng_state"])
        assert torch.equal(torch.rand(3), update_result["rng_probe"])
        assert torch.equal(expected_state.h, micro_result["h"])
        assert torch.equal(expected_probabilities, micro_result["probabilities"])
        expected_eligibility = eligibility_step(
            eligibility, expected_steps[0][0].trace["x"].detach()
        )
        assert torch.equal(expected_eligibility, micro_result["eligibility"])
        assert micro_result["fee_total"] == 0.05 + FEE_PER_STEP


def test_r024_birth_boundary_gated_continuation():
    torch.set_num_threads(1)
    parent = LivingCoreReference(CONFIG).double()
    book, _ = gc.collect_credit(parent, JOURNAL)
    steps_total = len(book.deltas)
    fit, check = range(0, steps_total // 2), range(steps_total // 2, steps_total)
    full_credits = {name: gc.branch_credit(book, b) for b, name in enumerate(book.branch_ids)}
    fit_credits = {name: gc.branch_credit(book, b, fit) for b, name in enumerate(book.branch_ids)}
    verdicts = {}
    for b, name in enumerate(book.branch_ids):
        tau_noise = gc.calibrate_tau_noise(book, b, fit)
        verdicts[name] = gc.branch_verdict(book, b, tau_noise, fit, check)
    allocation = {}
    for group in range(CONFIG.groups):
        alive, caps = {}, {}
        for name, verdict in verdicts.items():
            entry = verdict["groups"][group]
            if entry["enter_V"]:
                alive[name] = gc.rho_curve(torch.tensor(entry["sigma"]), entry["frobenius"])
                caps[name] = entry["rank_num"]
        for name, units in gc.allocate(alive, caps)["allocation"].items():
            allocation[name] = units
    directions = {}
    for name, units in allocation.items():
        b = book.branch_ids.index(name)
        for group in range(CONFIG.groups):
            _, _, vectors = torch.linalg.svd(fit_credits[name][group], full_matrices=False)
            directions[(name, group)] = [vectors[:, j].detach().clone() for j in range(units)]
    child, plan = gc.birth(parent, directions, full_credits)
    provider = gate_provider(book.gates)
    state_mid, _ = resume_career(
        child, child.initial_state(JOURNAL.subject_id), JOURNAL, 0, provider, limit=LEARN_BOUNDARY
    )
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        bundle_path = tmp / "born_bundle.pt"
        save_subject(
            bundle_path,
            child,
            state_mid,
            career=JOURNAL,
            journal_cursor=LEARN_BOUNDARY,
            eligibility=torch.zeros(CONFIG.groups, dtype=torch.float64),
            growth_gates=[[m, 1.0 - m] for m, _ in book.gates],
        )
        worker = spawn_worker(
            {
                "bundle": str(bundle_path),
                "mode": "microstep",
                "out": str(tmp / "born_result.pt"),
                "out_bundle": None,
                "fee_per_step": FEE_PER_STEP,
            }
        )
        bundle_data = load_subject(bundle_path)
    assert plan["child_r_shape"][1] == plan["parent_r_shape"][1] + 2
    expected_state, expected_steps = resume_career(
        child, child.initial_state(JOURNAL.subject_id), JOURNAL, 0, provider, limit=55
    )
    assert torch.equal(expected_state.h, worker["h"])
    assert torch.equal(expected_steps[-1][0].probabilities.detach(), worker["probabilities"])
    assert bundle_data["growth_parent_units"] == plan["parent_r_shape"][1]
    assert bundle_data["growth_gates"] is not None


def test_r024_migration_boundary_and_rng():
    torch.set_num_threads(1)
    model = LivingCoreReference(CONFIG).double()
    state, _ = run_career(model, JOURNAL)

    def coordinate_change(shadow):
        generator = torch.Generator().manual_seed(20261031)
        noise = torch.randn(shadow.encoder.input.weight.shape, generator=generator) * 1e-3
        shadow.encoder.input.weight = torch.nn.Parameter(
            shadow.encoder.input.weight.detach() + noise
        )

    committed = replay_transaction(model, state, JOURNAL, coordinate_change)
    assert committed["outcome"] == "committed"
    new_model, new_state = committed["committed_model"], committed["committed_state"]
    optimizer = new_optimizer(new_model)
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        bundle_path = tmp / "migrated_bundle.pt"
        save_subject(
            bundle_path,
            new_model,
            new_state,
            career=JOURNAL,
            journal_cursor=len(SPECS),
            optimizer=optimizer,
            eligibility=torch.zeros(CONFIG.groups, dtype=torch.float64),
            extra={"fee_total": 0.4},
        )
        worker = spawn_worker(
            {
                "bundle": str(bundle_path),
                "mode": "update",
                "out": str(tmp / "migrated_result.pt"),
                "out_bundle": None,
                "fee_per_step": FEE_PER_STEP,
            }
        )
        bundle_data = load_subject(bundle_path)
    # The worker ran the first declared update from the restored bundle; the parent
    # runs the same update locally and compares parameters bitwise.
    parent_loss = declared_update(new_model, new_state, optimizer)
    assert parent_loss > 0.0
    assert worker["post_update_consumed"] is False
    for name, parameter in new_model.named_parameters():
        assert torch.equal(parameter.detach().cpu(), worker["parameters_after"][name]), name
    torch.set_rng_state(bundle_data["rng_state"])
    assert torch.equal(torch.rand(3), worker["rng_probe"])
    assert bundle_data["revision"] == 1


def test_r024_pending_outcome_idempotence():
    torch.set_num_threads(1)
    model = LivingCoreReference(CONFIG).double()
    state, _ = run_career(model, JOURNAL)
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        bundle_path = tmp / "pending_bundle.pt"
        entries = [{"seq": 0, "kind": "fee", "amount": 0.1}]
        save_subject(
            bundle_path,
            model,
            state,
            career=JOURNAL,
            journal_cursor=len(SPECS),
            eligibility=torch.zeros(CONFIG.groups, dtype=torch.float64),
            journal_entries=[dict(entry) for entry in entries],
            pending={"outcome_id": "o-1", "reward": 0.25, "resolved": False},
        )
        first = spawn_worker(
            {
                "bundle": str(bundle_path),
                "mode": "resolve",
                "out": str(tmp / "resolve1.pt"),
                "out_bundle": None,
                "current_entries": entries,
                "fee_per_step": FEE_PER_STEP,
            }
        )
        assert first["resolved"] is True and first["reward_delta"] == 0.25
        updated_entries = first["journal_entries_out"]
        assert [entry["seq"] for entry in updated_entries] == [0, 1]
        assert updated_entries[1]["kind"] == "resolve_outcome"
        second = spawn_worker(
            {
                "bundle": str(bundle_path),
                "mode": "resolve",
                "out": str(tmp / "resolve2.pt"),
                "out_bundle": None,
                "current_entries": updated_entries,
                "fee_per_step": FEE_PER_STEP,
            }
        )
        assert second["resolved"] is False and second["reward_delta"] == 0.0
        assert second["guard"] == "already_resolved"


def test_r024_journal_monotonicity_across_restore():
    torch.set_num_threads(1)
    model = LivingCoreReference(CONFIG).double()
    state_mid, _ = resume_career(
        model, model.initial_state(JOURNAL.subject_id), JOURNAL, 0, limit=20
    )
    entries = [{"seq": 3, "kind": "fee", "amount": 0.01}]
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        bundle_path = tmp / "mono_bundle.pt"
        save_subject(
            bundle_path,
            model,
            state_mid,
            career=JOURNAL,
            journal_cursor=20,
            eligibility=torch.zeros(CONFIG.groups, dtype=torch.float64),
            journal_entries=[dict(entry) for entry in entries],
        )
        worker = spawn_worker(
            {
                "bundle": str(bundle_path),
                "mode": "microstep",
                "out": str(tmp / "mono_result.pt"),
                "out_bundle": str(tmp / "mono_advanced.pt"),
                "fee_per_step": FEE_PER_STEP,
            }
        )
        advanced = load_subject(tmp / "mono_advanced.pt")
        restored = load_subject(bundle_path)
    assert advanced["journal_cursor"] == 21
    assert torch.equal(advanced["h"], worker["h"])
    assert advanced["step"] == state_mid.step + 1
    assert restored["journal_cursor"] == 20
    assert [entry["seq"] for entry in worker["journal_entries_out"]] == [3]
    appended = append_journal(restored["journal_entries"], "restore", at_cursor=20)
    assert appended["seq"] == 4
    assert journal_has_resolution(restored["journal_entries"], "missing") is False
