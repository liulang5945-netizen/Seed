"""Is question-conditioned focus reachable at all by W_q alone? Zero optimizer steps.

The r2 scale calibration proved the logit range must come from the init, not from
384 Adam updates. It did NOT prove that any W_q can put the attention mass on the
queried colour's columns. Because the module's logits are exactly linear in W_q,
that question is a linear least-squares feasibility problem, not a training run:
solve for the best W_q on the labelled TRAIN scenes, then read focus on the unseen
FIT scenes with the module's own forward.

Column keys are reconstructed from the module's own unit_result and anchored
bit-wise against the logits captured at the module's softmax call. A deranged
target assignment is the negative control that gives the reading dynamic range.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training import probe_u1r2_scale_focus_20261116 as sf
from scripts.training import probe_u1r2_value_side_20261116 as vs
from scripts.training import run_u1_unit_read_20261116 as u1
from scripts.training import run_u1r1_position_20261116 as r1
from taiji.u2_history import value_digest
from taiji.u2_neural_units import UnitGraphSpec

REPORT = ROOT / "reports/taiji_u2_u1r2_focus_feasibility_20261116.json"
T = r1.T
CAP_CPU, CAP_WALL, RESERVE = 90.0, 180.0, 10.0
TARGET_FLOOR = 1e-4
ANCHOR_ATOL = 1e-12
NORM_SCAN_FACTORS = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625)
RESIDUAL_QUALITY_RATIO = 0.5
ROTATION_RELATIVE_DISPLACEMENT = 1.5
GRADIENT_COSINE_MIN = 0.99
OPTIMIZER_STEPS = [0]


def column_keys(graph: Any, item: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    data = u1.cast_item(item, graph.spec.torch_dtype)
    with torch.no_grad():
        out = graph(data["channels"], data["gates"], graph.initial_state())
        result = out["unit_result"]
        active = graph.units.active_gate(data["gates"])
        gated_a = result["a_trace"] * active
        state_view = graph.state_mix[0] * result["activity"] + graph.state_mix[1] * gated_a
        keys = torch.einsum("hn,tn->th", graph.units.p, state_view)
        question = data["channels"]["question"][0].to(keys.dtype)
    return keys.detach(), question.detach()


def design_row(graph: Any, keys: torch.Tensor, question: torch.Tensor) -> torch.Tensor:
    """logits[t] = sum_{d,h} attn_scale * keys[t,h] * question[d] * W_q[d,h]."""
    row = graph.attn_scale * torch.einsum("th,d->tdh", keys, question)
    return row.reshape(T, -1)


def target_logits(item: dict[str, Any]) -> torch.Tensor:
    mass = vs.column_mass(item)
    total = float(mass.sum())
    normalised = mass / total if total > 0 else mass
    return torch.log(normalised.clamp_min(TARGET_FLOOR))


def solve_focus(
    graph: Any,
    rows: list[torch.Tensor],
    targets: list[torch.Tensor],
) -> dict[str, Any]:
    """Least squares with one free logit offset per scene (softmax is shift invariant)."""
    n_scenes = len(rows)
    width = int(rows[0].shape[1])
    matrix = torch.zeros(n_scenes * T, width + n_scenes, dtype=torch.float64)
    rhs = torch.zeros(n_scenes * T, dtype=torch.float64)
    for index, (row, target) in enumerate(zip(rows, targets, strict=True)):
        matrix[index * T : (index + 1) * T, :width] = row
        matrix[index * T : (index + 1) * T, width + index] = 1.0
        rhs[index * T : (index + 1) * T] = target
    solution = torch.linalg.lstsq(matrix, rhs.unsqueeze(1), driver="gelsd").solution[:, 0]
    residual = matrix @ solution - rhs
    singular = torch.linalg.svdvals(matrix)
    w_q = solution[:width].reshape(graph.W_q.shape)
    return {
        "w_q": w_q,
        "equations": n_scenes * T,
        "unknowns": width + n_scenes,
        "w_q_unknowns": width,
        "offset_unknowns": n_scenes,
        "residual_rmse": float(torch.sqrt((residual.square()).mean())),
        "residual_max_abs": float(residual.abs().max()),
        "offset_abs_max": float(solution[width:].abs().max()),
        "singular_max": float(singular.max()),
        "singular_min": float(singular.min()),
        "rank": int(torch.linalg.matrix_rank(matrix)),
        "w_q_frobenius": float(w_q.norm()),
    }


def install(graph: Any, w_q: torch.Tensor) -> str:
    before = value_digest({n: t.detach().clone() for n, t in graph.state_dict().items()})
    with torch.no_grad():
        graph.W_q.copy_(w_q)
    after = value_digest({n: t.detach().clone() for n, t in graph.state_dict().items()})
    if before == after:
        raise ValueError("w_q_install_did_not_change_state")
    return after


def norm_scan(
    graph: Any,
    w_q: torch.Tensor,
    calibrated_norm: float,
    train: list[dict[str, Any]],
    fit: list[dict[str, Any]],
) -> dict[str, Any]:
    """Shrink the fitted query along its own direction; select on TRAIN focus only.

    The least-squares solution overshoots: it fits exact log targets including the
    log(floor) penalty on absent columns. focus_met only needs mass and spread, so
    the required norm is read off the smallest scanned factor that still meets the
    train criterion. Fit is reported at every factor but never used to select.
    """
    rows = []
    for factor in NORM_SCAN_FACTORS:
        install(graph, w_q * factor)
        train_stats = sf.focus_stats(graph, train)
        fit_stats = sf.focus_stats(graph, fit)
        rows.append(
            {
                "factor": factor,
                "w_q_frobenius": float(w_q.norm()) * factor,
                "norm_over_calibrated": float(w_q.norm()) * factor / calibrated_norm,
                "train_logit_range_mean": train_stats["logit_range_mean"],
                "train_answer_mass_mean": train_stats["answer_mass_mean"],
                "train_effective_columns_mean": train_stats["effective_columns_mean"],
                "train_focus_met": train_stats["focus_met"],
                "fit_answer_mass_mean": fit_stats["answer_mass_mean"],
                "fit_effective_columns_mean": fit_stats["effective_columns_mean"],
                "fit_focus_met": fit_stats["focus_met"],
            }
        )
    meeting = [row for row in rows if row["train_focus_met"]]
    smallest = min(meeting, key=lambda row: row["factor"]) if meeting else None
    return {
        "rows": rows,
        "selection_face": "train",
        "smallest_factor_meeting_train_focus": None if smallest is None else smallest["factor"],
        "required_norm_over_calibrated": (
            None if smallest is None else smallest["norm_over_calibrated"]
        ),
        "fit_focus_met_at_selected": None if smallest is None else smallest["fit_focus_met"],
        "calibrated_norm_already_sufficient": bool(rows[-1]["train_focus_met"]),
    }


def rotation_dose(
    init_norm: float, numel: int, updates: int, learning_rate: float
) -> dict[str, Any]:
    """Adam moves each coordinate by ~lr, so rotation is bounded by lr*updates*sqrt(n).

    A large init norm fixes the logit range but freezes the direction: the same
    absolute displacement is a smaller angle. This is arithmetic on the published
    optimizer settings, not a measurement; the pricing phase measures the step.
    """
    displacement = learning_rate * updates * math.sqrt(numel)
    needed = ROTATION_RELATIVE_DISPLACEMENT * init_norm / math.sqrt(numel) / updates
    return {
        "w_q_numel": numel,
        "updates": updates,
        "learning_rate": learning_rate,
        "init_w_q_frobenius": init_norm,
        "per_coordinate_rms": init_norm / math.sqrt(numel),
        "l2_displacement_upper_bound": displacement,
        "relative_displacement_upper_bound": displacement / init_norm,
        "rotation_target_relative_displacement": ROTATION_RELATIVE_DISPLACEMENT,
        "w_q_learning_rate_needed_for_rotation": needed,
        "rotation_reachable_at_uniform_lr": displacement / init_norm
        >= ROTATION_RELATIVE_DISPLACEMENT,
    }


def query_gradient(
    graph: Any, items: list[dict[str, Any]], kind: str
) -> tuple[torch.Tensor, float]:
    """W_q gradient vector under the module's own forward, plus the smallest attention."""
    graph.zero_grad(set_to_none=True)
    smallest = 1.0
    for item in items:
        data = u1.cast_item(item, graph.spec.torch_dtype)
        out = graph(data["channels"], data["gates"], graph.initial_state())
        attention = out["attention"]
        smallest = min(smallest, float(attention.detach().min()))
        if kind == "focus":
            target = vs.forced_distribution(item, "oracle_soft").to(attention.dtype)
            loss = -(target * torch.log(attention.clamp_min(sf.LOG_CLAMP))).sum() / len(items)
        else:
            label = torch.tensor(data["label"], dtype=torch.long)
            loss = torch.nn.functional.cross_entropy(
                out["readout"].unsqueeze(0), label.unsqueeze(0)
            ) / len(items)
        loss.backward()
    grad = graph.W_q.grad.detach().clone()
    graph.zero_grad(set_to_none=True)
    return grad, smallest


def init_operating_point(
    graph: Any,
    birth_direction: torch.Tensor,
    target_norm: float,
    calibrated_norm: float,
    train: list[dict[str, Any]],
) -> dict[str, Any]:
    """Does a sharp birth attention still carry a coherent query gradient?

    At the norm focus needs, a random birth direction saturates the softmax. Adam
    normalises magnitude, so saturation is only fatal if it corrupts the direction.
    Compare the gradient direction at the candidate init against the unsaturated
    calibrated scale; cosine near 1 means the signal survives.
    """
    labelled = [item for item in train if sf.labelled(item)]
    unit = birth_direction / float(birth_direction.norm())
    arms: dict[str, Any] = {}
    gradients: dict[str, dict[str, torch.Tensor]] = {}
    for name, norm in (("calibrated", calibrated_norm), ("candidate_init", target_norm)):
        install(graph, unit * norm)
        stats = sf.focus_stats(graph, train)
        focus_grad, focus_min = query_gradient(graph, labelled, "focus")
        task_grad, task_min = query_gradient(graph, labelled, "task")
        gradients[name] = {"focus": focus_grad, "task": task_grad}
        arms[name] = {
            "w_q_frobenius": norm,
            "train_logit_range_mean": stats["logit_range_mean"],
            "train_max_attention_mean": stats["max_attention_mean"],
            "train_attention_min_over_all": stats["attention_min_over_all"],
            "train_effective_columns_mean": stats["effective_columns_mean"],
            "train_answer_mass_mean": stats["answer_mass_mean"],
            "focus_gradient_l2": float(focus_grad.norm()),
            "task_gradient_l2": float(task_grad.norm()),
            "focus_attention_min_during_loss": focus_min,
            "task_attention_min_during_loss": task_min,
        }
    for kind in ("focus", "task"):
        low = gradients["calibrated"][kind]
        high = gradients["candidate_init"][kind]
        denominator = float(low.norm()) * float(high.norm())
        arms["candidate_init"][f"{kind}_gradient_cosine_vs_calibrated"] = (
            float((low * high).sum() / denominator) if denominator > 0 else None
        )
    return {"arms": arms, "gradient_face": "train_labelled", "gradient_denominator": len(labelled)}


def seed_row(
    seed: int,
    multiplier: float,
    train: list[dict[str, Any]],
    fit: list[dict[str, Any]],
    updates: int,
    learning_rate: float,
) -> dict[str, Any]:
    graph = sf.graph_at(seed, "birth", multiplier)
    calibrated_norm = float(graph.W_q.detach().norm())
    birth_direction = graph.W_q.detach().clone() / multiplier
    labelled_train = [item for item in train if sf.labelled(item)]
    labelled_fit = [item for item in fit if sf.labelled(item)]
    rows: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    anchor_max = 0.0
    for item in labelled_train:
        keys, question = column_keys(graph, item)
        row = design_row(graph, keys, question)
        rebuilt = row @ graph.W_q.detach().reshape(-1)
        captured = sf.capture_attention(graph, item)[1]
        anchor_max = max(anchor_max, float((rebuilt - captured).abs().max()))
        rows.append(row)
        targets.append(target_logits(item))
    solved = solve_focus(graph, rows, targets)
    deranged = targets[1:] + targets[:1]
    control = solve_focus(graph, rows, deranged)
    arms: dict[str, Any] = {}
    for name, candidate in (("lsq_train_targets", solved), ("lsq_deranged_targets", control)):
        digest = install(graph, candidate["w_q"])
        arms[name] = {
            "train": sf.focus_stats(graph, train),
            "fit": sf.focus_stats(graph, fit),
            "state_digest": digest,
            "w_q_frobenius": candidate["w_q_frobenius"],
            "w_q_norm_over_calibrated": candidate["w_q_frobenius"] / calibrated_norm,
            "residual_rmse": candidate["residual_rmse"],
            "residual_max_abs": candidate["residual_max_abs"],
            "offset_abs_max": candidate["offset_abs_max"],
        }
    return {
        "seed": int(seed),
        "multiplier": multiplier,
        "calibrated_w_q_frobenius": round(calibrated_norm, 9),
        "keys_anchor_max_abs": anchor_max,
        "labelled_train": len(labelled_train),
        "labelled_fit": len(labelled_fit),
        "system": {
            key: value for key, value in solved.items() if key not in ("w_q", "w_q_frobenius")
        },
        "control_system": {
            key: value for key, value in control.items() if key not in ("w_q", "w_q_frobenius")
        },
        "arms": arms,
        "norm_scan": norm_scan(graph, solved["w_q"], calibrated_norm, train, fit),
        "init_operating_point": init_operating_point(
            graph, birth_direction, solved["w_q_frobenius"], calibrated_norm, train
        ),
        "rotation_dose": rotation_dose(
            solved["w_q_frobenius"], int(graph.W_q.numel()), updates, learning_rate
        ),
    }


def classify(rows: list[dict[str, Any]], complete: bool) -> str:
    if not complete or not rows:
        return "not_established"
    if any(row["keys_anchor_max_abs"] > ANCHOR_ATOL for row in rows):
        return "instrument_invalid"
    if any(row["labelled_fit"] == 0 or row["labelled_train"] == 0 for row in rows):
        return "instrument_invalid"
    treated = [row["arms"]["lsq_train_targets"] for row in rows]
    control = [row["arms"]["lsq_deranged_targets"] for row in rows]
    if not all(
        c["fit"]["answer_mass_mean"] < t["fit"]["answer_mass_mean"]
        for t, c in zip(treated, control, strict=True)
    ):
        return "control_not_separated"
    if not all(a["train"]["focus_met"] for a in treated):
        return "focus_linearly_unreachable"
    if not any(a["fit"]["focus_met"] for a in treated):
        return "focus_reachable_in_sample_not_unseen"
    if not all(a["fit"]["focus_met"] for a in treated):
        return "focus_generalizes_on_some_seeds"
    return "focus_reachable_and_generalizes"


def main() -> int:
    torch.set_num_threads(1)
    started_cpu, started_wall = time.process_time(), time.monotonic()
    contract = json.loads(r1.CONTRACT.read_text(encoding="utf-8"))
    r1.guards(contract, "probe")
    calibration = json.loads(sf.REPORT.read_text(encoding="utf-8"))
    if calibration["criteria_sha256"] != contract["criteria_sha256"]:
        raise ValueError("criteria_sha256_mismatch_between_contract_and_calibration")
    multipliers = calibration["scale"]["per_seed_multiplier"]
    updates = int(contract["criteria"]["updates_per_seed"])
    learning_rate = float(contract["criteria"]["learning_rate"])
    data = u1.load_data(contract)
    episodes = u1.prepare_episodes(data, UnitGraphSpec().n_units)
    rows: list[dict[str, Any]] = []
    budget_stopped = False
    real_step = torch.optim.Optimizer.step

    def counted_step(self: Any, *args: Any, **kwargs: Any) -> Any:
        OPTIMIZER_STEPS[0] += 1
        return real_step(self, *args, **kwargs)

    torch.optim.Optimizer.step = counted_step  # type: ignore[method-assign]
    try:
        for seed in contract["criteria"]["seeds"]:
            if time.process_time() - started_cpu >= CAP_CPU - RESERVE:
                budget_stopped = True
                break
            if time.monotonic() - started_wall >= CAP_WALL - RESERVE:
                budget_stopped = True
                break
            rows.append(
                seed_row(
                    int(seed),
                    float(multipliers[str(seed)]),
                    episodes["train"],
                    episodes["fit"],
                    updates,
                    learning_rate,
                )
            )
    finally:
        torch.optim.Optimizer.step = real_step  # type: ignore[method-assign]
    complete = (not budget_stopped) and len(rows) == len(contract["criteria"]["seeds"])
    verdict = classify(rows, complete)
    target_span = math.log(1.0 / TARGET_FLOOR)
    checks = {
        "keys_reproduce_module_logits": all(r["keys_anchor_max_abs"] <= ANCHOR_ATOL for r in rows),
        "system_overdetermined": all(
            r["system"]["equations"] > r["system"]["unknowns"] for r in rows
        ),
        "lsq_residual_within_half_target_span": all(
            r["system"]["residual_rmse"] < RESIDUAL_QUALITY_RATIO * target_span for r in rows
        ),
        "labelled_fit_denominator_positive": all(r["labelled_fit"] > 0 for r in rows),
        "deranged_control_fits_train_worse": all(
            r["control_system"]["residual_rmse"] > r["system"]["residual_rmse"] for r in rows
        ),
        "deranged_control_loses_unseen_focus": all(
            r["arms"]["lsq_deranged_targets"]["fit"]["answer_mass_mean"]
            < r["arms"]["lsq_train_targets"]["fit"]["answer_mass_mean"]
            for r in rows
        ),
        "install_changed_state_every_arm": all(
            len({a["state_digest"] for a in r["arms"].values()}) == len(r["arms"]) for r in rows
        ),
        "norm_scan_brackets_the_criterion": all(
            any(row["train_focus_met"] for row in r["norm_scan"]["rows"])
            and any(not row["train_focus_met"] for row in r["norm_scan"]["rows"])
            for r in rows
        ),
        "init_focus_gradient_direction_survives": all(
            (
                r["init_operating_point"]["arms"]["candidate_init"][
                    "focus_gradient_cosine_vs_calibrated"
                ]
                or 0.0
            )
            >= GRADIENT_COSINE_MIN
            for r in rows
        ),
        "init_task_gradient_direction_survives": all(
            (
                r["init_operating_point"]["arms"]["candidate_init"][
                    "task_gradient_cosine_vs_calibrated"
                ]
                or 0.0
            )
            >= GRADIENT_COSINE_MIN
            for r in rows
        ),
        "init_gradients_nonzero": all(
            r["init_operating_point"]["arms"]["candidate_init"]["focus_gradient_l2"] > 0.0
            and r["init_operating_point"]["arms"]["candidate_init"]["task_gradient_l2"] > 0.0
            for r in rows
        ),
        "no_optimizer_steps": OPTIMIZER_STEPS[0] == 0,
    }
    payload = {
        "schema": "taiji-u2-u1r2-focus-feasibility-v3",
        "card": "22.5.U.1r2",
        "phase": "design_probe",
        "anchored_contract": "plans/manifests/taiji_u2_u1r1_position_20261116.json",
        "anchored_calibration": "reports/taiji_u2_u1r2_scale_focus_20261116.json",
        "criteria_sha256": contract["criteria_sha256"],
        "columns_T": T,
        "declared": {
            "target_floor": TARGET_FLOOR,
            "target_logit_span": target_span,
            "anchor_atol": ANCHOR_ATOL,
            "residual_quality_ratio": RESIDUAL_QUALITY_RATIO,
            "norm_scan_factors": list(NORM_SCAN_FACTORS),
            "rotation_relative_displacement": ROTATION_RELATIVE_DISPLACEMENT,
            "gradient_cosine_min": GRADIENT_COSINE_MIN,
            "focus_answer_mass_min": sf.FOCUS_ANSWER_MASS_MIN,
            "focus_effective_columns_max": sf.FOCUS_EFFECTIVE_COLUMNS_MAX,
            "method": "linear least squares on W_q with one free logit offset per scene",
            "supervision_face": "train",
            "generalization_face": "fit",
            "rank_deficiency_disclosure": (
                "the design matrix is rank deficient by construction: only 6 distinct question "
                "vectors drive 16 column keys, so the reachable logit space is far smaller than "
                "the equation count; gelsd returns the minimum-norm least-squares solution and "
                "the reported rank is a disclosure, not a pass condition"
            ),
        },
        "seeds": rows,
        "classification": verdict,
        "instrument_checks": checks,
        "instrument_green": sum(1 for v in checks.values() if v),
        "instrument_total": len(checks),
        "revision_history": {
            "version": 3,
            "v1_verdict": "focus_reachable_and_generalizes",
            "v1_instrument": "7/8",
            "v2_verdict": "focus_reachable_and_generalizes",
            "v2_instrument": "9/9",
            "changes": [
                "dropped system_full_column_rank: rank 66 of 112 unknowns is a property of the "
                "encoding (6 distinct questions), not an instrument defect; replaced by an "
                "absolute residual-quality threshold that can fail",
                "added norm_scan: the least-squares norm overshoots focus_met, so the required "
                "norm is read along the fitted direction on the TRAIN face only",
                "added rotation_dose: a large init norm freezes the query direction under a "
                "fixed Adam dose, which the v1 reading did not price",
                "no_optimizer_steps is now counted by patching Optimizer.step instead of "
                "being asserted as a constant",
                "v3 added init_operating_point: at the norm focus needs, a random birth "
                "direction saturates the softmax, so the query gradient direction is compared "
                "against the unsaturated scale before the init is declared usable",
                "v3 moved rotation_dose to the candidate init norm (v2 priced it at the "
                "range-calibrated norm, which is not the operating point)",
            ],
        },
        "capability_scope": "design feasibility for r2; no capability claim, no trained model",
        "discipline": {
            "optimizer_steps": OPTIMIZER_STEPS[0],
            "training_updates": 0,
            "data_file_reads": 1,
            "partitions_measured": ["train", "fit"],
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0,
            "sealed_partitions_read": [],
            "product_code_edits": 0,
            "capability_claims": 0,
        },
        "budget": {
            "cap_cpu": CAP_CPU,
            "cap_wall": CAP_WALL,
            "reserve": RESERVE,
            "budget_stopped": budget_stopped,
        },
        "cost": {
            "cpu": round(time.process_time() - started_cpu, 4),
            "wall": round(time.monotonic() - started_wall, 4),
        },
    }
    r1.write_report(REPORT, payload)
    print(
        "FOCUS_FEASIBILITY verdict="
        + verdict
        + " instrument="
        + str(payload["instrument_green"])
        + "/"
        + str(payload["instrument_total"])
        + " cpu="
        + str(payload["cost"]["cpu"])
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
