"""R023 growth-credit counterexample battery and engineering oracles.

Covers the R023 contract on the R021 reference graph: signal conditions, noise
calibration (including its measured non-discrimination on random gates), max-min
allocation, function-neutral gated birth, per-unit startup, lesion rollback, the
global-SVD baseline, the §24 counterexamples and the §10.3 replay transaction.
No trainer, no task data, no capability claim.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from taiji import growth_credit as gc
from taiji.living_core import LivingCoreConfig, LivingCoreReference, RawSource
from taiji.state_migration import (
    CareerJournal,
    MigrationBudget,
    replay_transaction,
    run_career,
    states_bitwise_equal,
)

CONFIG = LivingCoreConfig(groups=2, width=8, units=12, adapter_rank=2, key_width=4, value_width=4)
LONG_JOURNAL = CareerJournal(
    subject_id="r023-birth",
    prompt=b"observe-context-" * 4,
    target=(b"target-bytes-ab" * 7)[:100],
    goal=b"growth-goal",
    records=(
        RawSource("s1", b"abc target ab", version=1, observed_at=0),
        RawSource("s2", b"secondary ab", version=2, observed_at=4),
    ),
    raw_nu=(0.5, 2.0, 0.0, 0.8, 8.0, -0.4, 0.0, 0.3),
    known_nu=(True, True, False, True, True, True, False, True),
    now=10,
)
SHORT_JOURNAL = CareerJournal(
    subject_id="r023-migration",
    prompt=b"observe-context-",
    target=b"ab",
    goal=b"growth-goal",
    records=(RawSource("s1", b"abc target ab", version=1, observed_at=0),),
    raw_nu=(0.5, 2.0, 0.0, 0.8, 8.0, -0.4, 0.0, 0.3),
    known_nu=(True, True, False, True, True, True, False, True),
    now=10,
)
RANDOM_SEEDS = (1, 2, 3, 4, 5)

_CACHE = None


def pipeline():
    global _CACHE
    if _CACHE is None:
        torch.set_num_threads(1)
        model = LivingCoreReference(CONFIG).double()
        book, state = gc.collect_credit(model, LONG_JOURNAL)
        steps = len(book.deltas)
        fit, check = range(0, steps // 2), range(steps // 2, steps)
        verdicts, full_credits, fit_credits, tau_noises = {}, {}, {}, {}
        for branch, name in enumerate(book.branch_ids):
            tau_noises[name] = gc.calibrate_tau_noise(book, branch, fit)
            verdicts[name] = gc.branch_verdict(book, branch, tau_noises[name], fit, check)
            full_credits[name] = gc.branch_credit(book, branch)
            fit_credits[name] = gc.branch_credit(book, branch, fit)
        allocation, proposals = {}, {}
        for group in range(model.config.groups):
            alive, caps = {}, {}
            for name, verdict in verdicts.items():
                entry = verdict["groups"][group]
                if entry["enter_V"]:
                    alive[name] = gc.rho_curve(torch.tensor(entry["sigma"]), entry["frobenius"])
                    caps[name] = entry["rank_num"]
            proposals[group] = gc.allocate(alive, caps)
            for name, units in proposals[group]["allocation"].items():
                allocation[name] = units
        directions = {}
        for name, units in allocation.items():
            branch = book.branch_ids.index(name)
            for group in range(model.config.groups):
                _, _, vectors = torch.linalg.svd(fit_credits[name][group], full_matrices=False)
                directions[(name, group)] = [vectors[:, j].detach().clone() for j in range(units)]
        child, plan = gc.birth(model, directions, full_credits)
        _CACHE = {
            "model": model,
            "book": book,
            "state": state,
            "fit": fit,
            "check": check,
            "verdicts": verdicts,
            "full_credits": full_credits,
            "fit_credits": fit_credits,
            "tau_noises": tau_noises,
            "allocation": allocation,
            "proposals": proposals,
            "directions": directions,
            "child": child,
            "plan": plan,
        }
    return _CACHE


def gate_provider(book):
    return lambda index: {"copy": book.gates[index][0], "generate": book.gates[index][1]}


def test_r023_signal_conditions_on_reference_gates():
    data = pipeline()
    book = data["book"]
    assert len(book.deltas) == LONG_JOURNAL.event_count
    assert data["fit"].stop - data["fit"].start >= gc.WINDOW_MIN_POSITIONS
    assert data["check"].stop - data["check"].start >= gc.WINDOW_MIN_POSITIONS
    for verdict in data["verdicts"].values():
        for entry in verdict["groups"]:
            assert entry["norm_ok"] and entry["windows_ok"] and entry["noise_ok"]
            assert entry["enter_V"] is True
            assert entry["rank_num"] == 1
            alignment = entry["alignment"][0]
            assert alignment["cosine_ok"] and alignment["check_norm_ok"]
            assert alignment["cosine"] >= gc.COSINE_MIN
    credit_global = sum(data["full_credits"].values())
    kappa = gc.cancellation_index(data["full_credits"], credit_global)
    assert kappa is not None and 0.0 < kappa <= 1.0


def test_r023_birth_neutral_startup_and_lesion():
    data = pipeline()
    model, child, plan, book = data["model"], data["child"], data["plan"], data["book"]
    assert plan["child_r_shape"][1] == plan["parent_r_shape"][1] + 2
    assert plan["child_p_shape"][2] == plan["parent_p_shape"][2] + 2
    provider = gate_provider(book)
    with torch.no_grad():
        parent_state, parent_steps = run_career(model, LONG_JOURNAL)
        child_state, child_steps = run_career(child, LONG_JOURNAL, growth_gate_provider=provider)
    assert all(
        torch.equal(a.probabilities, b.probabilities)
        for (a, _), (b, _) in zip(parent_steps, child_steps, strict=True)
    )
    assert torch.equal(parent_state.h, child_state.h)
    assert torch.equal(parent_state.fast, child_state.fast)
    _, graph_steps = run_career(child, LONG_JOURNAL, growth_gate_provider=provider)
    terms = [output.probabilities[byte] for output, byte in graph_steps if byte is not None]
    loss = -torch.stack(terms).log().mean()
    row_grads, column_grads = torch.autograd.grad(loss, [child.R, child.P])
    parent_units = plan["parent_r_shape"][1]
    assert bool(column_grads[:, :, parent_units:].abs().max() == 0.0)
    report = gc.startup_report(plan)
    assert report["l_start"] == 1.0
    base = parent_units
    offsets = {}
    for group in range(CONFIG.groups):
        offset = 0
        for branch, entry_group in sorted(data["directions"]):
            if entry_group == group:
                offsets[(branch, group)] = offset
                offset += len(data["directions"][(branch, group)])
    for key, predicted in sorted(plan["predicted_incoming"].items()):
        branch, group, unit = key.split(".")
        row = row_grads[int(group), base + offsets[(branch, int(group))] + int(unit)]
        assert torch.allclose(row, predicted, atol=1e-12, rtol=1e-12)
        assert float(row.norm()) > gc.TAU_G
    independence = gc.independence_ratio(data["directions"], data["full_credits"])
    assert independence["ratio"] == 1.0
    gc.lesion(child, plan)
    assert torch.equal(child.R, model.R) and torch.equal(child.P, model.P)
    assert child.growth_parent_units is None and child.growth_unit_branches == []
    with torch.no_grad():
        restored_state, restored_steps = run_career(child, LONG_JOURNAL)
    assert torch.equal(restored_state.h, parent_state.h)
    assert all(
        torch.equal(a.probabilities, b.probabilities)
        for (a, _), (b, _) in zip(parent_steps, restored_steps, strict=True)
    )


def test_r023_admission_failure_rolls_back():
    data = pipeline()
    model = data["model"]
    credit = torch.zeros(8, 56, dtype=torch.float64)
    credit[0, 0] = 1.0
    _, _, vectors = torch.linalg.svd(credit, full_matrices=False)
    directions = {("dead", 0): [vectors[:, 0].detach(), vectors[:, 1].detach()]}
    child, plan = gc.birth(model, directions, {"dead": credit.unsqueeze(0)})
    report = gc.startup_report(plan)
    assert report["l_start"] == 0.5
    failed = [row["unit"] for row in report["units"] if not row["started"]]
    assert failed == ["dead.0.1"]
    gc.lesion(child, plan)
    assert torch.equal(child.R, model.R) and torch.equal(child.P, model.P)


def test_r023_calibration_non_discrimination_recorded():
    data = pipeline()
    book, fit = data["book"], data["fit"]
    steps = len(book.deltas)
    margins, first_random = [], None
    for seed in RANDOM_SEEDS:
        generator = torch.Generator().manual_seed(seed)
        values = torch.rand(steps, generator=generator).tolist()
        random_book = replace(book, gates=tuple((value, 1.0 - value) for value in values))
        if first_random is None:
            first_random = random_book
        tau_noise = gc.calibrate_tau_noise(random_book, 0, fit)
        verdict = gc.branch_verdict(random_book, 0, tau_noise, fit, range(steps // 2, steps))
        entry = verdict["groups"][0]
        margins.append(entry["sigma"][0] / entry["tau_noise"])
        assert entry["noise_ok"] is True
    # A random gate's own SVD direction clears the naive startup floor, so L_start
    # alone cannot separate alignment from noise; on this fixture the frozen 0.99
    # permutation bar does not either (temporal common-mode inflates the null).
    random_credit = gc.branch_credit(first_random, 0, fit)
    _, sigma, _ = gc.numerical_rank(random_credit[0])
    assert float(sigma[0]) > gc.TAU_G
    aligned = data["verdicts"]["copy"]["groups"][0]
    aligned_margin = aligned["sigma"][0] / aligned["tau_noise"]
    generator = torch.Generator().manual_seed(20261031)
    order = torch.randperm(steps, generator=generator).tolist()
    copy_values = [gate[0] for gate in book.gates]
    permuted_book = replace(
        book, gates=tuple((copy_values[index], 1.0 - copy_values[index]) for index in order)
    )
    tau_noise = gc.calibrate_tau_noise(permuted_book, 0, fit)
    verdict = gc.branch_verdict(permuted_book, 0, tau_noise, fit, range(steps // 2, steps))
    assert verdict["groups"][0]["noise_ok"] is True
    assert all(0.9 < margin < 1.5 for margin in margins + [aligned_margin])


def test_r023_allocation_optimality_ties_caps_and_budget():
    flat = [0.0, 1.0, 1.0]
    result = gc.allocate({"a": [0.0, 1.0], "b": [0.0, 1.0]}, {"a": 2, "b": 2}, r_max=2)
    assert result["allocation"] == {"a": 1, "b": 1}
    result = gc.allocate({"a": flat, "b": [0.0, 1.0]}, {"a": 2, "b": 1}, r_max=3)
    assert result["allocation"] == {"a": 1, "b": 1} and result["total_units"] == 2
    result = gc.allocate({"a": flat, "b": flat}, {"a": 2, "b": 2}, r_max=3)
    assert result["allocation"] == {"a": 1, "b": 1}
    result = gc.allocate(
        {"a": [0.0, 0.5, 1.0], "b": [0.0, 0.4, 0.9], "c": [0.0, 0.3, 0.6]},
        {"a": 2, "b": 2, "c": 2},
        r_max=2,
    )
    # With |V| > R_max every feasible allocation leaves some branch at rho 0, so
    # max-min is tied at zero and the total-rho tie-break concentrates on the
    # strongest branch; uncovered branches are reported, never fabricated.
    assert result["allocation"] == {"a": 2} and result["min_rho"] == 0.0
    assert result["uncovered"] == ["b", "c"]
    result = gc.allocate({"a": [0.0, 1.0, 1.0]}, {"a": 2}, r_max=1)
    assert result["allocation"] == {"a": 1}
    result = gc.allocate(
        {"a": [0.0, 1.0, 1.0]}, {"a": 2}, r_max=8, budget=gc.ProposalBudget(max_states=2)
    )
    assert result["status"] == "budget_stop"
    assert gc.allocate({}, {})["status"] == "no_candidates"
    result = gc.allocate({"a": [0.0, 1.0]}, {"a": 1}, r_max=8)
    assert result["uncovered"] == []


def test_r023_rank_capping_hadamard_and_soft_gate():
    hadamard = (
        torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, -1.0, 1.0, -1.0],
                [1.0, 1.0, -1.0, -1.0],
                [1.0, -1.0, -1.0, 1.0],
            ],
            dtype=torch.float64,
        )
        / 2.0
    )
    diag_credit = torch.diag(torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64))
    incoming = hadamard.transpose(0, 1) @ diag_credit
    rank, _, _ = gc.numerical_rank(incoming)
    assert rank == 1
    assert torch.allclose(incoming.norm(dim=1), torch.full((4,), 0.5, dtype=torch.float64))
    _, sigma, _ = gc.numerical_rank(diag_credit)
    assert gc.rho_curve(sigma, 1.0)[4] - gc.rho_curve(sigma, 1.0)[1] < 1e-12
    generator = torch.Generator().manual_seed(20261031)
    left = torch.randn(8, generator=generator, dtype=torch.float64)
    right = torch.randn(56, generator=generator, dtype=torch.float64)
    matrix = torch.outer(left, right)
    half = 0.5 * matrix
    ranks = [gc.numerical_rank(half)[0], gc.numerical_rank(half)[0]]
    assert sum(ranks) == 2 and gc.numerical_rank(matrix)[0] == 1
    cancellation = {"c1": matrix, "c2": -matrix}
    assert gc.cancellation_index(cancellation, matrix + (-matrix)) == 0.0
    baseline = gc.global_svd_baseline((matrix + (-matrix)).unsqueeze(0))
    assert all(group["refused"] for group in baseline["groups"])
    _, _, vectors = torch.linalg.svd(matrix, full_matrices=False)
    guard = gc.shared_direction_guard(vectors[:, :1], matrix + (-matrix))
    assert guard["accepted"] is False and guard["sigma_min_overall"] == 0.0


def test_r023_replay_non_monotonic_accepted():
    unit = torch.tensor([1.0, 1.0], dtype=torch.float64) / 2**0.5
    before = torch.eye(2, dtype=torch.float64)
    after = before - torch.outer(unit, unit)
    rank_before, _, _ = gc.numerical_rank(before)
    rank_after, sigma_after, _ = gc.numerical_rank(after)
    assert rank_before == 2 and rank_after == 1
    assert abs(float(sigma_after[0]) - 1.0) < 1e-12
    result = gc.allocate(
        {"only": gc.rho_curve(sigma_after, float(torch.linalg.vector_norm(after)))},
        {"only": rank_after},
    )
    assert result["status"] == "allocated" and result["allocation"] == {"only": 1}


def test_r023_migration_identity_commit_and_budget_refusal():
    torch.set_num_threads(1)
    model = LivingCoreReference(CONFIG).double()
    original_state, _ = run_career(model, SHORT_JOURNAL)
    replayed_state, _ = run_career(model, SHORT_JOURNAL)
    assert states_bitwise_equal(original_state, replayed_state)
    identity = replay_transaction(model, original_state, SHORT_JOURNAL)
    assert identity["outcome"] == "identity_checked" and identity["state_bitwise_equal"] is True

    def coordinate_change(shadow):
        generator = torch.Generator().manual_seed(20261031)
        noise = torch.randn(shadow.encoder.input.weight.shape, generator=generator) * 1e-3
        shadow.encoder.input.weight = torch.nn.Parameter(
            shadow.encoder.input.weight.detach() + noise
        )

    committed = replay_transaction(model, original_state, SHORT_JOURNAL, coordinate_change)
    assert committed["outcome"] == "committed"
    new_state, new_model = committed["committed_state"], committed["committed_model"]
    assert new_state.revision == original_state.revision + 1
    assert new_state.subject_id == original_state.subject_id
    assert committed["no_replay_parameter_mutation"] is True
    assert committed["reward_ledger_untouched"] is True
    assert committed["external_actions_executed"] == 0
    for name, parameter in new_model.named_parameters():
        parent_parameter = dict(model.named_parameters())[name]
        if name.startswith("encoder."):
            continue
        assert torch.equal(parameter.detach(), parent_parameter.detach()), name
    assert not torch.equal(
        new_model.encoder.input.weight.detach(), model.encoder.input.weight.detach()
    )
    refused = replay_transaction(
        model, original_state, SHORT_JOURNAL, budget=MigrationBudget(max_events=3)
    )
    assert refused["outcome"] == "refused_budget" and refused["reason"] == "event_cap"
    refused = replay_transaction(
        model, original_state, SHORT_JOURNAL, budget=MigrationBudget(max_journal_bytes=10)
    )
    assert refused["outcome"] == "refused_budget" and refused["reason"] == "byte_cap"
