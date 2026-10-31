"""§24.4 credit coverage birth on the R021 reference graph residual path.

Implements the frozen engineering bounds of SPEC-TAIJI-CORE_REFERENCE_GRAPH_R021 §4
and the R023 contract: branch credit C_c = sum_s g_c(s) delta_s x_s^T, 128-permutation
noise calibration, signal conditions, max-min integer allocation under R_max, the
function-neutral birth operator on the P/R residual pair, the global-SVD baseline arm
and the shared-direction global-credit guard. Directions are selected on the fit
window; the actual startup gradient is cross-checked on the full career. Engineering
instruments only; no task claim, no trainer.
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F

from taiji.living_core import LivingCoreReference, LivingStep, state_normalize
from taiji.state_migration import CareerJournal, run_career

R_MAX = 8
TAU_G = 1e-8
FLOOR_FROBENIUS = 1e-6
COSINE_MIN = 0.1
WINDOW_MIN_POSITIONS = 32
PERMUTATIONS = 128
NOISE_QUANTILE = 0.99
C_SCALE = 1.0
ENUMERATION_CAP = 6561
PROPOSAL_CPU_SECONDS = 1.0
PROPOSAL_WALL_SECONDS = 2.0
NOISE_SEED = 20261031


class GatedGrowthCore(LivingCoreReference):
    """R021 step plus the §24.4 gated newborn residual term (v1.2 candidate).

    The forward mirrors LivingCoreReference.step exactly; the only addition is
    that residual rows beyond ``growth_parent_units`` contribute through their
    declared branch gate: ``pre += sum_c g_c(step) * B_c * tanh(R_c x)``. Gate
    values are recorded per career step with auditable provenance (the parent
    endpoint trace) and enter as step inputs, not learned parameters. Parent
    rows stay ungated. taiji/living_core.py itself is unchanged.
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        self.growth_parent_units: int | None = None
        self.growth_unit_branches: list[list[str]] = []

    def step(self, state, *, event, growth_gates=None, **kwargs):
        c, ref = self.config, self.W
        if state.h.shape != (c.groups, c.width) or event.shape != (c.width,):
            raise ValueError("State/event dimensions differ from R021")
        if state.fast.shape != (c.groups, c.fast_heads, c.key_width, c.value_width):
            raise ValueError("Fast state dimensions differ from R021")
        if (
            not 0 <= kwargs.get("previous_byte", 256) <= 256
            or not 0 <= kwargs.get("mode", 0) < 7
            or len(kwargs.get("goal", b"")) > c.max_event_bytes
        ):
            raise ValueError("Invalid preceding byte, mode or goal budget")
        if not bool(
            torch.isfinite(state.h).all()
            and torch.isfinite(state.fast).all()
            and torch.isfinite(event).all()
        ) or bool(state.h.abs().max() > 1):
            raise ValueError("Invalid neural state or event")
        raw_nu = ref.new_zeros(8) if kwargs.get("raw_nu") is None else kwargs["raw_nu"]
        known_nu = kwargs.get("known_nu")
        if known_nu is None:
            known_nu = torch.zeros(8, dtype=torch.bool, device=ref.device)
        allowed = kwargs.get("allowed")
        if allowed is None:
            allowed = torch.zeros((c.groups, c.fast_heads), dtype=torch.bool, device=ref.device)
        goal_vector = self.encoder.goal_vector(kwargs.get("goal", b""))
        ema, nu, nu_features = self.update_nu(state, raw_nu, known_nu)
        view = self.source_view(
            kwargs.get("records", ()),
            kwargs.get("owner", "owner"),
            kwargs.get("scope", "scope"),
            kwargs.get("now", 0),
        )
        coverage, prior = self.align_positions(state, view.keys)
        message, message_weights, mq, mk, mv = self.messages(state.h, goal_vector)
        fast, fast_queries, fast_values = self.read_fast(
            state.h, state.fast, event, goal_vector, nu
        )
        recalled, episodic_weights = self.episodic(state.h, goal_vector, view)
        x = torch.cat(
            [
                state_normalize(state.h),
                event.expand_as(state.h),
                message,
                fast,
                recalled,
                goal_vector.expand_as(state.h),
                nu.expand_as(state.h),
            ],
            -1,
        )
        effective = self.W + self.U @ self.V.transpose(1, 2)
        pre = torch.einsum("ndp,np->nd", effective, x)
        parent_units = (
            self.growth_parent_units if self.growth_parent_units is not None else self.R.shape[1]
        )
        pre = pre + torch.einsum(
            "ndm,nm->nd",
            self.P[:, :, :parent_units],
            torch.tanh(torch.einsum("nmp,np->nm", self.R[:, :parent_units, :], x)),
        )
        if parent_units < self.R.shape[1]:
            if growth_gates is None:
                raise ValueError("Gated newborn units require per-step branch gates")
            activated = torch.tanh(torch.einsum("nmp,np->nm", self.R[:, parent_units:, :], x))
            new_units = self.R.shape[1] - parent_units
            gate_rows = []
            for group in range(c.groups):
                branches = self.growth_unit_branches[group]
                values = [growth_gates[branch] for branch in branches]
                values.extend(0.0 for _ in range(new_units - len(branches)))
                gate_rows.append(torch.tensor(values, dtype=x.dtype, device=x.device))
            activated = activated * torch.stack(gate_rows, dim=0)
            pre = pre + torch.einsum("ndm,nm->nd", self.P[:, :, parent_units:], activated)
        tau = 0.5 + 127.5 * torch.sigmoid(self.tau_logit)
        leak = (1 - torch.exp(-1 / tau))[:, None]
        h = (1 - leak) * state.h + leak * torch.tanh(pre)
        mode_vector = F.one_hot(torch.tensor(kwargs.get("mode", 0), device=ref.device), 7).to(
            ref.dtype
        )
        previous = self.encoder.embedding(
            torch.tensor(kwargs.get("previous_byte", 256), device=ref.device)
        )
        xi = self.xi(torch.cat([goal_vector, previous, mode_vector, nu]))
        z, query_weights, joined = self.readout(h, xi)
        r_task = query_weights.mean(0) @ recalled
        probabilities, copied, mixing, continuation, generated = self.expression(
            z, r_task, goal_vector, previous, view, coverage, prior
        )
        action = ref.new_zeros(c.width) if kwargs.get("action") is None else kwargs["action"]
        predictions = torch.stack([head(torch.cat([z, action])) for head in self.world])
        memory, chi, write_keys = self.write_fast(
            h,
            state.fast,
            event,
            goal_vector,
            nu,
            mode_vector,
            ref.new_tensor(kwargs.get("past_error", 0.0)),
            kwargs.get("error_missing", True),
            allowed,
        )
        child_state = replace(
            state,
            step=state.step + 1,
            h=h,
            fast=memory,
            nu_ema=ema,
            keys=view.keys,
            coverage=coverage,
            prior=prior,
        )
        trace = {
            "event": event,
            "goal": goal_vector,
            "nu": nu,
            "nu_features": nu_features,
            "message": message,
            "message_weights": message_weights,
            "message_q": mq,
            "message_k": mk,
            "message_v": mv,
            "fast_read": fast,
            "fast_queries": fast_queries,
            "fast_values": fast_values,
            "recalled": recalled,
            "episodic_weights": episodic_weights,
            "x": x,
            "preactivation": pre,
            "tau": tau,
            "query": xi,
            "z": z,
            "joined_heads": joined,
            "query_weights": query_weights,
            "mixing": mixing,
            "continuation": continuation,
            "generated": generated,
            "chi": chi,
            "write_keys": write_keys,
        }
        return LivingStep(
            child_state, probabilities, copied, view, predictions[:, :-1], predictions[:, -1], trace
        )


@dataclass(frozen=True)
class CreditBook:
    """Per-microstep design matrices, deltas and branch gate values."""

    branch_ids: tuple[str, ...]
    deltas: tuple[torch.Tensor, ...]  # each [n, d]
    inputs: tuple[torch.Tensor, ...]  # each [n, 7d], detached design matrix
    gates: tuple[tuple[float, ...], ...]  # per step, K values summing to 1
    loss: float
    round_count: int


def collect_credit(model: LivingCoreReference, journal: CareerJournal):
    state, steps = run_career(model, journal)
    terms = [output.probabilities[byte] for output, byte in steps if byte is not None]
    loss = -torch.stack(terms).log().mean()
    pre_tensors = [output.trace["preactivation"] for output, _ in steps]
    deltas = torch.autograd.grad(loss, pre_tensors)
    inputs = [output.trace["x"].detach() for output, _ in steps]
    gates = []
    for output, _ in steps:
        mixing = float(output.trace["mixing"].detach())
        gates.append((mixing, 1.0 - mixing))
    book = CreditBook(
        branch_ids=("copy", "generate"),
        deltas=tuple(delta.detach() for delta in deltas),
        inputs=tuple(inputs),
        gates=tuple(gates),
        loss=float(loss.detach()),
        round_count=len(terms),
    )
    return book, state


def branch_credit(book: CreditBook, branch: int, window=None) -> torch.Tensor:
    first = book.deltas[0]
    credit = torch.zeros(
        first.shape[0], first.shape[1], book.inputs[0].shape[1], dtype=torch.float64
    )
    indices = range(len(book.deltas)) if window is None else window
    for s in indices:
        weight = book.gates[s][branch]
        if weight == 0.0:
            continue
        credit = credit + weight * torch.einsum("nd,np->ndp", book.deltas[s], book.inputs[s])
    return credit


def nonzero_positions(book: CreditBook, window: range) -> int:
    return sum(1 for s in window if float(book.deltas[s].abs().max()) > 0.0)


def numerical_rank(matrix: torch.Tensor, tau_noise: float = 0.0):
    sigma = torch.linalg.svdvals(matrix)
    tau = max(1e-8, 1e-5 * float(sigma[0]), tau_noise)
    return int((sigma > tau).sum()), sigma, tau


def _higher_quantile(values: list[float], quantile: float) -> float:
    ordered = sorted(values, reverse=True)
    index = min(len(ordered), max(1, math.ceil(quantile * len(ordered)))) - 1
    return ordered[index]


def calibrate_tau_noise(
    book: CreditBook, branch: int, window: range, permutations: int = PERMUTATIONS
) -> torch.Tensor:
    """Per-group 0.99 higher-order statistic of permuted branch max singular values.

    The gate stays with its delta; only the residual-input correspondence (X
    columns) is permuted, keeping the averaging and dimensions consistent. Each
    group's credit matrix gets its own null distribution.
    """
    generator = torch.Generator().manual_seed(NOISE_SEED)
    deltas = torch.stack([book.deltas[s] for s in window])
    inputs = torch.stack([book.inputs[s] for s in window])
    weights = torch.tensor([book.gates[s][branch] for s in window], dtype=torch.float64)
    weighted = weights[:, None, None] * deltas
    maxima: list[list[float]] = [[] for _ in range(deltas.shape[1])]
    for _ in range(permutations):
        order = torch.randperm(len(window), generator=generator)
        permuted = torch.einsum("snd,snp->ndp", weighted, inputs[order])
        sigmas = torch.linalg.svdvals(permuted)
        for group in range(deltas.shape[1]):
            maxima[group].append(float(sigmas[group].max()))
    return torch.tensor(
        [_higher_quantile(values, NOISE_QUANTILE) for values in maxima], dtype=torch.float64
    )


def window_alignment(direction: torch.Tensor, credit_fit: torch.Tensor, credit_check: torch.Tensor):
    projection_fit = direction.reshape(1, -1) @ credit_fit
    projection_check = direction.reshape(1, -1) @ credit_check
    norm_fit = float(torch.linalg.vector_norm(projection_fit))
    norm_check = float(torch.linalg.vector_norm(projection_check))
    if norm_fit == 0.0 or norm_check == 0.0:
        cosine = 0.0
    else:
        cosine = float(
            (projection_fit.flatten() @ projection_check.flatten()) / (norm_fit * norm_check)
        )
    return {
        "fit_norm": norm_fit,
        "check_norm": norm_check,
        "cosine": cosine,
        "check_norm_ok": norm_check > TAU_G,
        "cosine_ok": cosine >= COSINE_MIN,
    }


def branch_verdict(
    book: CreditBook, branch: int, tau_noise: torch.Tensor, fit: range, check: range
) -> dict:
    """Per-group signal verdicts for one branch; V entry is decided per group."""
    credit_fit = branch_credit(book, branch, fit)
    credit_check = branch_credit(book, branch, check)
    groups = []
    for group in range(credit_fit.shape[0]):
        rank, sigma, tau = numerical_rank(credit_fit[group], float(tau_noise[group]))
        frobenius = float(torch.linalg.vector_norm(credit_fit[group]))
        positions_fit = nonzero_positions(book, fit)
        positions_check = nonzero_positions(book, check)
        entry = {
            "group": group,
            "frobenius": frobenius,
            "sigma": [float(value) for value in sigma],
            "tau": tau,
            "tau_noise": float(tau_noise[group]),
            "rank_num": rank,
            "fit_positions": positions_fit,
            "check_positions": positions_check,
            "norm_ok": frobenius > FLOOR_FROBENIUS,
            "windows_ok": (
                positions_fit >= WINDOW_MIN_POSITIONS and positions_check >= WINDOW_MIN_POSITIONS
            ),
            "noise_ok": bool(sigma[0] > tau) if sigma.numel() else False,
        }
        entry["enter_V"] = entry["norm_ok"] and entry["windows_ok"] and entry["noise_ok"]
        alignments = []
        if rank > 0:
            _, _, vectors = torch.linalg.svd(credit_fit[group], full_matrices=False)
            for unit in range(rank):
                alignments.append(
                    {
                        "unit": unit,
                        **window_alignment(
                            vectors[:, unit], credit_fit[group], credit_check[group]
                        ),
                    }
                )
        entry["alignment"] = alignments
        groups.append(entry)
    return {"branch": book.branch_ids[branch], "groups": groups}


def rho_curve(sigma: torch.Tensor, frobenius: float) -> list[float]:
    """rho(r) = sum_{j<=r} sigma_j^2 / ||C||_F^2 for r = 0..len(sigma)."""
    if frobenius <= 0.0:
        return []
    squared = (sigma**2).cumsum(0) / frobenius**2
    return [0.0] + [float(value) for value in squared]


@dataclass(frozen=True)
class ProposalBudget:
    max_states: int = ENUMERATION_CAP
    max_cpu_seconds: float = PROPOSAL_CPU_SECONDS
    max_wall_seconds: float = PROPOSAL_WALL_SECONDS


def allocate(
    rho_tables: dict[str, list[float]],
    rank_caps: dict[str, int],
    r_max: int = R_MAX,
    budget: ProposalBudget | None = None,
) -> dict:
    """Exact max-min integer allocation over branches in V, with tie-breaks.

    Key: (min rho, total rho, -total units, r tuple over sorted branch ids), all
    maximized; fewer units then the stable branch-id order break ties.
    """
    budget = budget or ProposalBudget()
    ids = sorted(rho_tables)
    if not ids:
        return {"status": "no_candidates", "allocation": {}, "uncovered": []}
    caps = {branch: min(rank_caps[branch], r_max, len(rho_tables[branch]) - 1) for branch in ids}
    best_key, best = None, None
    states = 0
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    for combination in itertools.product(*(range(caps[branch] + 1) for branch in ids)):
        states += 1
        if (
            states > budget.max_states
            or time.process_time() - cpu_start > budget.max_cpu_seconds
            or time.perf_counter() - wall_start > budget.max_wall_seconds
        ):
            return {
                "status": "budget_stop",
                "states_checked": states,
                "allocation": {},
                "uncovered": [],
            }
        if sum(combination) > r_max:
            continue
        rhos = {branch: rho_tables[branch][combination[index]] for index, branch in enumerate(ids)}
        key = (min(rhos.values()), sum(rhos.values()), -sum(combination), combination)
        if best_key is None or key > best_key:
            best_key, best = key, dict(zip(ids, combination, strict=True))
    allocation = {branch: units for branch, units in best.items() if units > 0}
    uncovered = [branch for branch, units in best.items() if units == 0]
    return {
        "status": "allocated",
        "allocation": allocation,
        "uncovered": uncovered,
        "states_checked": states,
        "min_rho": best_key[0],
        "total_rho": best_key[1],
        "total_units": sum(best.values()),
    }


def fit_directions(
    credit_fit: torch.Tensor, allocation: dict[str, int], branch_ids: tuple[str, ...]
) -> dict:
    """Per (branch, group): the first r_c left singular vectors of the fit credit."""
    table = {}
    for branch, units in sorted(allocation.items()):
        index = branch_ids.index(branch)
        for group in range(credit_fit.shape[0]):
            _, _, vectors = torch.linalg.svd(credit_fit[index][group], full_matrices=False)
            table[(branch, group)] = [vectors[:, j].detach().clone() for j in range(units)]
    return table


def birth(
    parent: LivingCoreReference,
    directions: dict,
    credits_full: dict[str, torch.Tensor],
    c_scale: float = C_SCALE,
):
    """Build the gated child: parent rows plus §24.4 newborn gated residual units."""
    child = GatedGrowthCore(parent.config).to(dtype=parent.W.dtype)
    child.load_state_dict(parent.state_dict())
    groups = parent.config.groups
    parent_r, parent_p = parent.R.detach().clone(), parent.P.detach().clone()
    counts = {group: 0 for group in range(groups)}
    for branch, group in directions:
        counts[group] += len(directions[(branch, group)])
    # Dense [n, m, p] layout forces equal newborn-row counts across groups; groups
    # with fewer real units are zero-padded (R zero rows and P zero columns keep
    # the contribution exactly zero and padding units permanently gradient-dead).
    max_extra = max(counts.values()) if counts else 0
    grown_r, grown_p, unit_branches = [], [], []
    for group in range(groups):
        extra = counts[group]
        zeros = torch.zeros(max_extra, parent_r.shape[2], dtype=parent_r.dtype)
        grown_r.append(torch.cat([parent_r[group], zeros], dim=0))
        columns, branches = [], []
        for branch, entry_group in sorted(directions):
            if entry_group != group:
                continue
            for j in range(len(directions[(branch, group)])):
                columns.append(c_scale * directions[(branch, group)][j])
                branches.append(branch)
        padding = max_extra - extra
        if padding > 0:
            columns.extend(
                torch.zeros(parent_p.shape[1], dtype=parent_p.dtype) for _ in range(padding)
            )
            branches.extend(["pad"] * padding)
        block = (
            torch.stack(columns, dim=1)
            if columns
            else torch.zeros(parent_p.shape[1], 0, dtype=parent_p.dtype)
        )
        grown_p.append(torch.cat([parent_p[group], block], dim=1))
        unit_branches.append(branches)
    child.R = torch.nn.Parameter(torch.stack(grown_r, dim=0))
    child.P = torch.nn.Parameter(torch.stack(grown_p, dim=0))
    child.growth_parent_units = parent_r.shape[1]
    child.growth_unit_branches = unit_branches
    predicted = {}
    for (branch, group), unit_directions in directions.items():
        for j, direction in enumerate(unit_directions):
            incoming = (c_scale * direction).reshape(1, -1) @ credits_full[branch][group]
            predicted[f"{branch}.{group}.{j}"] = incoming.flatten().detach()
    plan = {
        "parent_r": parent_r,
        "parent_p": parent_p,
        "parent_r_shape": list(parent_r.shape),
        "child_r_shape": list(child.R.shape),
        "parent_p_shape": list(parent_p.shape),
        "child_p_shape": list(child.P.shape),
        "c_scale": c_scale,
        "unit_keys": sorted(predicted),
        "predicted_incoming": predicted,
        "units_per_group": counts,
        "padded_units_per_group": {group: max_extra - counts[group] for group in range(groups)},
        "unit_branches": unit_branches,
    }
    return child, plan


def lesion(child: GatedGrowthCore, plan: dict) -> None:
    """Restore the exact parent residual tensors and clear growth metadata."""
    child.R = torch.nn.Parameter(plan["parent_r"].clone())
    child.P = torch.nn.Parameter(plan["parent_p"].clone())
    child.growth_parent_units = None
    child.growth_unit_branches = []


def cancellation_index(
    credits: dict[str, torch.Tensor], credit_global: torch.Tensor
) -> float | None:
    """||C||_F / sum_c ||C_c||_F; undefined (None) when the denominator is zero."""
    denominator = sum(float(torch.linalg.vector_norm(value)) for value in credits.values())
    if denominator == 0.0:
        return None
    return float(torch.linalg.vector_norm(credit_global)) / denominator


def startup_report(plan: dict) -> dict:
    """L_start over the born units with per-unit incoming norms, failed rows kept."""
    rows = []
    for key in plan["unit_keys"]:
        norm = float(plan["predicted_incoming"][key].norm())
        rows.append({"unit": key, "incoming_norm": norm, "started": norm > TAU_G})
    l_start = sum(row["started"] for row in rows) / len(rows) if rows else None
    return {"units": rows, "l_start": l_start}


def independence_ratio(
    directions: dict, credits_full: dict[str, torch.Tensor], c_scale: float = C_SCALE
) -> dict:
    """sum_c rank_num(B_c^T C_c) / sum_c r_c with per-cluster blocks kept separate."""
    numerator, denominator, blocks = 0, 0, {}
    for (branch, group), unit_directions in directions.items():
        if not unit_directions:
            continue
        matrix = torch.stack([c_scale * value for value in unit_directions], dim=1)
        incoming = matrix.transpose(0, 1) @ credits_full[branch][group]
        rank, _, _ = numerical_rank(incoming)
        numerator += rank
        denominator += len(unit_directions)
        blocks[f"{branch}.{group}"] = rank
    ratio = numerator / denominator if denominator else None
    return {
        "ratio": ratio,
        "numerator": numerator,
        "denominator": denominator,
        "blocks": blocks,
    }


def global_svd_baseline(credit_global: torch.Tensor, r_max: int = R_MAX, c_scale: float = C_SCALE):
    """Pre-v1.2 comparison arm: directions from the global credit's SVD, same floors."""
    results = []
    for group in range(credit_global.shape[0]):
        rank, sigma, tau = numerical_rank(credit_global[group])
        units = min(rank, r_max)
        directions = []
        if units > 0:
            _, _, vectors = torch.linalg.svd(credit_global[group], full_matrices=False)
            directions = [c_scale * vectors[:, j].detach().clone() for j in range(units)]
        results.append(
            {
                "group": group,
                "sigma": [float(value) for value in sigma],
                "tau": tau,
                "rank_num": rank,
                "units": units,
                "directions": directions,
                "refused": units == 0,
            }
        )
    return {"groups": results}


def shared_direction_guard(
    directions: torch.Tensor, credit_global: torch.Tensor, tau_global: float = TAU_G
) -> dict:
    """A shared branch must hold real overall credit, not per-branch max-min."""
    projection = directions.transpose(0, 1) @ credit_global
    sigma_min = float(torch.linalg.svdvals(projection).min())
    return {
        "sigma_min_overall": sigma_min,
        "tau_global": tau_global,
        "accepted": sigma_min > tau_global,
    }
