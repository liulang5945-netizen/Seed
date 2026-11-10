"""09.4 plasticity arms on the same parent subject (R026 contract).

Six arms share one parent subject and one declared update schedule (8 Adam steps
on the teacher-forced replay branch over the fit journal). Birth arms differ only
in unit directions and gate tables. The preregistered endpoint and adjudication
live in the R026 contract and are evaluated verbatim. No trainer beyond the
declared expression; no task data.
"""

from __future__ import annotations

import time

import torch

from taiji.growth_credit import (
    R_MAX,
    GatedGrowthCore,
    branch_credit,
    numerical_rank,
)
from taiji.state_migration import run_career

UPDATE_LR = 1e-3
UPDATE_STEPS = 8
HOLDOUT_PROMPT = b"probe-holdout-" * 6
HOLDOUT_TARGET = b"probe target ab" * 4
RETENTION_TOLERANCE = 0.05
NET_MARGIN = 1e-3


def evaluate_career_nll(model, journal, provider=None):
    _, steps = run_career(model, journal, growth_gate_provider=provider)
    terms = [output.probabilities[byte] for output, byte in steps if byte is not None]
    return float((-torch.stack(terms).log().mean()).detach())


def declared_feedback_updates(model, journal, optimizer, provider=None, steps=UPDATE_STEPS):
    """The declared legal-feedback update: Adam on the replay-branch NLL."""
    losses = []
    for _ in range(steps):
        _, steps_out = run_career(model, journal, growth_gate_provider=provider)
        terms = [o.probabilities[b] for o, b in steps_out if b is not None]
        loss = -torch.stack(terms).log().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


def _gated_child(parent, directions, unit_branches, gates_table):
    child = GatedGrowthCore(parent.config).to(dtype=parent.W.dtype)
    child.load_state_dict(parent.state_dict())
    parent_r, parent_p = parent.R.detach().clone(), parent.P.detach().clone()
    max_extra = max(len(value) for value in directions.values())
    grown_r, grown_p = [], []
    for group in range(parent.config.groups):
        zeros = torch.zeros(max_extra, parent_r.shape[2], dtype=parent_r.dtype)
        grown_r.append(torch.cat([parent_r[group], zeros], dim=0))
        columns = list(directions.get(group, []))
        padding = max_extra - len(columns)
        columns.extend(torch.zeros(parent_p.shape[1], dtype=parent_p.dtype) for _ in range(padding))
        grown_p.append(torch.cat([parent_p[group], torch.stack(columns, dim=1)], dim=1))
    child.R = torch.nn.Parameter(torch.stack(grown_r, dim=0))
    child.P = torch.nn.Parameter(torch.stack(grown_p, dim=0))
    child.growth_parent_units = parent_r.shape[1]
    child.growth_unit_branches = unit_branches
    provider = (
        None if gates_table is None else lambda index: dict(gates_table[index % len(gates_table)])
    )
    return child, provider


def _top_directions(credit, count, seed=None):
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
        base = torch.randn(
            credit.shape[0], credit.shape[1], generator=generator, dtype=credit.dtype
        )
        q, _ = torch.linalg.qr(base)
        return [q[:, j].detach().clone() for j in range(min(count, q.shape[1]))]
    _, _, vectors = torch.linalg.svd(credit, full_matrices=False)
    return [vectors[:, j].detach().clone() for j in range(min(count, vectors.shape[1]))]


def coverage_birth_arm(parent, book, fit_credits, verdicts):
    """The R023 pipeline verbatim: signal conditions, max-min, fit SVD, mixed gates."""
    from taiji.growth_credit import rho_curve

    allocation = {}
    for group in range(parent.config.groups):
        alive, caps = {}, {}
        for name, verdict in verdicts.items():
            entry = verdict["groups"][group]
            if entry["enter_V"]:
                alive[name] = rho_curve(torch.tensor(entry["sigma"]), entry["frobenius"])
                caps[name] = entry["rank_num"]
        for name, units in _max_min_allocate(alive, caps)["allocation"].items():
            allocation[name] = units
    directions: dict[int, list] = {}
    unit_branches = []
    counts = {group: 0 for group in range(parent.config.groups)}
    for name, units in sorted(allocation.items()):
        for group in range(parent.config.groups):
            value = _top_directions(fit_credits[name][group], units)
            directions.setdefault(group, []).extend(value)
            counts[group] += len(value)
    for _ in range(parent.config.groups):
        unit_branches.append(
            [name for name, units in sorted(allocation.items()) for _ in range(units)]
        )
    child, provider = _gated_child(
        parent, directions, unit_branches, [{"copy": m, "generate": 1.0 - m} for m, _ in book.gates]
    )
    return child, provider, {"units_total": sum(counts.values()), "allocation": allocation}


def _max_min_allocate(rho_tables, caps, r_max=R_MAX):
    best_key, best = None, None
    ids = sorted(rho_tables)
    for combination in _feasible(caps, r_max):
        rhos = {name: rho_tables[name][combination[i]] for i, name in enumerate(ids)}
        key = (min(rhos.values()), sum(rhos.values()), -sum(combination), combination)
        if best_key is None or key > best_key:
            best_key, best = key, dict(zip(ids, combination, strict=True))
    return {"allocation": {k: v for k, v in (best or {}).items() if v > 0}}


def _feasible(caps, r_max):
    from itertools import product

    ids = sorted(caps)
    for combination in product(*(range(min(caps[name], r_max) + 1) for name in ids)):
        if sum(combination) <= r_max:
            yield combination


def gradmax_arm(parent, full_credits):
    """Fixed single-event schedule: global credit top-R_max directions, constant gates."""
    global_credit = sum(full_credits.values())
    per_group = R_MAX // parent.config.groups
    directions = {
        group: _top_directions(global_credit[group], per_group)
        for group in range(parent.config.groups)
    }
    unit_branches = [["const"] * len(directions[group]) for group in range(parent.config.groups)]
    child, provider = _gated_child(parent, directions, unit_branches, [{"const": 1.0}] * 109)
    return child, provider, {"units_total": per_group * parent.config.groups}


def conditional_svd_arm(parent, book, fit_credits, verdicts):
    """Per-branch top-r_c without max-min or the shared budget."""
    directions: dict[int, list] = {}
    unit_branches = []
    counts = {group: 0 for group in range(parent.config.groups)}
    for name in book.branch_ids:
        for group in range(parent.config.groups):
            rank = verdicts[name]["groups"][group]["rank_num"]
            value = _top_directions(fit_credits[name][group], min(rank, 4))
            directions.setdefault(group, []).extend(value)
            counts[group] += len(value)
    for group in range(parent.config.groups):
        unit_branches.append(
            [
                name
                for name in book.branch_ids
                for _ in range(min(verdicts[name]["groups"][group]["rank_num"], 4))
            ]
        )
    child, provider = _gated_child(
        parent, directions, unit_branches, [{"copy": m, "generate": 1.0 - m} for m, _ in book.gates]
    )
    return child, provider, {"units_total": sum(counts.values())}


def random_gate_arm(parent, book, unit_total, seed=20261031):
    """Birth directions from a random-gate credit; gates carry the random values.

    The credit is the gate-weighted gradient product under a seeded random gate
    (the R023 battery construction); the same values form the arm's gate table.
    Fit-journal gate tables wrap modulo on other careers (declared instrument)."""
    generator = torch.Generator().manual_seed(seed)
    from dataclasses import replace as dc_replace

    specs = len(book.deltas)
    values = torch.rand(specs, generator=generator).tolist()
    random_book = dc_replace(
        book, gates=tuple((value, 1.0 - value) for value in values), branch_ids=("rgate", "rgen")
    )
    credit = branch_credit(random_book, 0)
    per_group = unit_total // parent.config.groups
    directions = {
        group: _top_directions(credit[group], per_group) for group in range(parent.config.groups)
    }
    unit_branches = [["rgate"] * per_group for _ in range(parent.config.groups)]
    child, provider = _gated_child(
        parent, directions, unit_branches, [{"rgate": value} for value in values]
    )
    return child, provider, {"units_total": per_group * parent.config.groups}


def capacity_random_arm(parent, unit_total, seed=20261031):
    """Capacity-matched control: seeded random orthonormal directions, constant gates."""
    generator = torch.Generator().manual_seed(seed)
    per_group = unit_total // parent.config.groups
    directions = {}
    for group in range(parent.config.groups):
        base = torch.randn(parent.config.width, per_group, generator=generator, dtype=torch.float64)
        q, _ = torch.linalg.qr(base)
        directions[group] = [q[:, j].detach().clone() for j in range(per_group)]
    unit_branches = [["const"] * per_group for _ in range(parent.config.groups)]
    child, provider = _gated_child(parent, directions, unit_branches, [{"const": 1.0}] * 109)
    return child, provider, {"units_total": unit_total}


def credit_diagnostics(model, book_credits, fit_credits):
    """kappa and numerical ranks of X/Delta/Z stand-ins at the birth endpoint."""
    global_credit = sum(book_credits.values())
    kappa = float(global_credit.norm()) / sum(
        float(value.norm()) for value in book_credits.values()
    )
    ranks = {
        name: numerical_rank(value[group])[0]
        for name, value in fit_credits.items()
        for group in range(value.shape[0])
    }
    return {"kappa": kappa, "branch_group_ranks": ranks}


def run_arm(
    name, parent, journal, holdout_journal, retention_journal, build_child, optimizer_factory
):
    """One arm: optional birth, the declared updates, preregistered evaluations."""
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    provider = None
    plan = {"units_total": 0}
    model = parent
    if build_child is not None:
        model, provider, plan = build_child(parent)
    nll_fit_pre = evaluate_career_nll(model, journal, provider)
    nll_holdout_pre = evaluate_career_nll(model, holdout_journal, provider)
    nll_retention_pre = evaluate_career_nll(model, retention_journal, provider)
    optimizer = optimizer_factory(model)
    losses = declared_feedback_updates(model, journal, optimizer, provider)
    nll_fit_post = evaluate_career_nll(model, journal, provider)
    nll_holdout_post = evaluate_career_nll(model, holdout_journal, provider)
    nll_retention_post = evaluate_career_nll(model, retention_journal, provider)
    return {
        "arm": name,
        "units_total": plan["units_total"],
        "allocation": plan.get("allocation"),
        "nll_fit_pre": nll_fit_pre,
        "nll_fit_post": nll_fit_post,
        "drift_retention": abs(nll_retention_post - nll_retention_pre),
        "delta_nll_holdout": nll_holdout_post - nll_holdout_pre,
        "update_losses": losses,
        "cpu_seconds": time.process_time() - cpu_start,
        "wall_seconds": time.perf_counter() - wall_start,
    }


def adjudicate(arm_results):
    """The preregistered R026 rule, verbatim."""
    arms = {row["arm"]: row for row in arm_results}
    coverage = arms["coverage_birth"]
    controls = min(
        arms["replay_only"]["delta_nll_holdout"], arms["capacity_random"]["delta_nll_holdout"]
    )
    breaches = [row["arm"] for row in arm_results if row["drift_retention"] > RETENTION_TOLERANCE]
    if coverage["delta_nll_holdout"] < controls - NET_MARGIN and not breaches:
        verdict = "positive"
    elif abs(coverage["delta_nll_holdout"] - controls) <= NET_MARGIN:
        verdict = "indeterminable"
    else:
        verdict = "negative"
    return {"verdict": verdict, "retention_breaches": breaches}
