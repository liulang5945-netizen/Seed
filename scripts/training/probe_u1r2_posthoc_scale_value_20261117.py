"""22.5.U.1r2 post-hoc, zero-training localisation of the focus_not_met negative.

The frozen card closed `focus_not_met`: seed02 stayed below the focus gate and the
read gates were 0/3, while `log_tau` reached only 0.53-0.58 of the value the design
probe required. Three questions are still open and decide whether the wall is a
control gap or an architectural one:

  1. Would more scale alone rescue the read?  Sweep `log_tau` on the three trained
     capsules (weights frozen) and score the model's own position head.
  2. Did the learned query direction contribute anything?  Zero `W_q` on the same
     trained weights (exact mean-pool arm) and compare.
  3. Is the value side decodable at all under the trained attention?  Fit the same
     linear no-bias head class to convergence (convex CE, LBFGS) on the attended
     vectors and score fit; oracle-forced attention is the positive control.

A gradient census at the trained point explains why `log_tau` stopped where it did
(mean focus gradient, positive-scene count, cosine against the task gradient).

This is a diagnostic, not a revision: zero optimizer steps, zero weight writes,
only the run's own train/fit partitions, no check/final/confirm reads.
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
from scripts.training import run_u1r2_scale_supervision_20261116 as r2
from taiji.u2_history import value_digest
from taiji.u2_neural_units import UnitGraphSpec, apply_capsule, load_capsule
from taiji.u2_unit_temperature_readout import TemperaturePositionGraph

REPORT = ROOT / "reports/taiji_u2_u1r2_posthoc_scale_value_20261117.json"
CAP_CPU = 60.0
CAP_WALL = 120.0
LOG_CLAMP = sf.LOG_CLAMP
FIT_GATE = 0.80
TRAIN_GATE = 0.90
ORACLE_FIT_MIN = 0.80
DERANGED_FIT_MAX = 0.50
CHANCE_BAND = 0.50


def sha_of(path: Path) -> str:
    return r2.sha_of(path)


def param_digest(graph: torch.nn.Module) -> str:
    return value_digest(
        {name: tensor.detach().clone() for name, tensor in graph.state_dict().items()}
    )


def load_trained(seed: int) -> TemperaturePositionGraph:
    capsule = load_capsule(r2.final_capsule_path(seed))
    graph = TemperaturePositionGraph(UnitGraphSpec(**capsule["runtime"]["spec"]))
    apply_capsule(graph, capsule)
    return graph


def set_tau(graph: TemperaturePositionGraph, value: float) -> None:
    with torch.no_grad():
        graph.log_tau.data.fill_(float(value))


def attention_grad_census(
    graph: TemperaturePositionGraph, episodes: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    """One backward per scene at the trained point; no optimizer step."""
    focus_tau: list[torch.Tensor] = []
    task_tau: list[torch.Tensor] = []
    focus_wq: list[torch.Tensor] = []
    task_wq: list[torch.Tensor] = []
    for item in episodes["train"]:
        data = r2.u1.cast_item(item, graph.spec.torch_dtype)
        out = graph(data["channels"], data["gates"], graph.initial_state())
        label = torch.tensor(item["label"], dtype=torch.long)
        task = torch.nn.functional.cross_entropy(out["readout"].unsqueeze(0), label.unsqueeze(0))
        task_grads = torch.autograd.grad(task, (graph.log_tau, graph.W_q), retain_graph=True)
        task_tau.append(task_grads[0].detach().clone())
        task_wq.append(task_grads[1].detach().clone())
        if sf.labelled(item):
            target = vs.forced_distribution(item, "oracle_soft").to(out["attention"].dtype)
            focus = -(target * torch.log(out["attention"].clamp_min(LOG_CLAMP))).sum()
            focus_grads = torch.autograd.grad(focus, (graph.log_tau, graph.W_q), retain_graph=False)
            focus_tau.append(focus_grads[0].detach().clone())
            focus_wq.append(focus_grads[1].detach().clone())
        graph.zero_grad(set_to_none=True)
    tau_focus_sum = torch.stack(focus_tau).sum()
    tau_task_sum = torch.stack(task_tau).sum()
    wq_focus_sum = torch.stack(focus_wq).sum()
    wq_task_sum = torch.stack(task_wq).sum()
    return {
        "labelled_scenes": len(focus_tau),
        "task_scenes": len(task_tau),
        "focus_tau_sum": float(tau_focus_sum),
        "focus_tau_mean": float(tau_focus_sum) / len(focus_tau),
        "focus_tau_positive_scenes": int(sum(int(g > 0) for g in focus_tau)),
        "focus_tau_negative_scenes": int(sum(int(g < 0) for g in focus_tau)),
        "task_tau_sum": float(tau_task_sum),
        "cosine_tau_task_vs_focus": float(
            torch.nn.functional.cosine_similarity(
                tau_focus_sum.reshape(1), tau_task_sum.reshape(1), dim=0
            )
        ),
        "focus_wq_norm": float(torch.linalg.vector_norm(wq_focus_sum)),
        "task_wq_norm": float(torch.linalg.vector_norm(wq_task_sum)),
        "cosine_wq_task_vs_focus": float(
            torch.nn.functional.cosine_similarity(
                wq_focus_sum.reshape(-1), wq_task_sum.reshape(-1), dim=0
            )
        ),
    }


def features(
    graph: TemperaturePositionGraph, episodes: dict[str, list[dict[str, Any]]], arm: str
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int], list[dict[str, Any]]]:
    rows: dict[str, list[torch.Tensor]] = {"train": [], "fit": []}
    labels: dict[str, list[int]] = {"train": [], "fit": []}
    checks: list[dict[str, Any]] = []
    for split in ("train", "fit"):
        for item in episodes[split]:
            vector, one = vs.attended_under_focus(graph, item, arm)
            rows[split].append(vector)
            labels[split].append(int(item["label"]))
            checks.append(one)
    return (
        torch.stack(rows["train"]),
        torch.stack(rows["fit"]),
        labels["train"],
        labels["fit"],
        checks,
    )


def read_at(
    graph: TemperaturePositionGraph, episodes: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    fit = r2.r1.evaluate(graph, episodes["fit"])
    train = r2.r1.evaluate(graph, episodes["train"])
    fit_focus = sf.focus_stats(graph, episodes["fit"])
    return {
        "fit_accuracy": fit["accuracy"],
        "train_accuracy": train["accuracy"],
        "fit_distinct": fit["distinct"],
        "train_distinct": train["distinct"],
        "fit_focus": fit_focus,
    }


def sweep(
    graph: TemperaturePositionGraph, episodes: dict[str, list[dict[str, Any]]], values: list[float]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in values:
        set_tau(graph, value)
        row = read_at(graph, episodes)
        rows.append({"log_tau": float(value), "scale": math.exp(float(value)), **row})
    return rows


def zero_query_arm(
    graph: TemperaturePositionGraph, episodes: dict[str, list[dict[str, Any]]], tau: float
) -> dict[str, Any]:
    saved = graph.W_q.detach().clone()
    set_tau(graph, tau)
    with torch.no_grad():
        graph.W_q.zero_()
    row = read_at(graph, episodes)
    with torch.no_grad():
        graph.W_q.copy_(saved)
    return {"log_tau": float(tau), **row}


def probe_seed(
    contract: dict[str, Any],
    run_arms: dict[int, dict[str, Any]],
    seed: int,
    episodes: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    graph = load_trained(seed)
    before = param_digest(graph)
    trained_tau = float(graph.log_tau.detach())
    arm = run_arms[seed]
    criteria = contract["criteria"]
    needed = float(
        math.log(float(criteria["logit_temperature"]["scale_target_by_seed"][str(seed)]))
    )
    trained = read_at(graph, episodes)
    reproduction = {
        "fit_matches_run": trained["fit_accuracy"] == arm["fit_accuracy"],
        "train_matches_run": trained["train_accuracy"] == arm["train_accuracy"],
        "focus_matches_run": (
            trained["fit_focus"]["answer_mass_mean"] == arm["fit_focus"]["answer_mass_mean"]
            and trained["fit_focus"]["effective_columns_mean"]
            == arm["fit_focus"]["effective_columns_mean"]
            and trained["fit_focus"]["focus_met"] == arm["fit_focus"]["focus_met"]
        ),
        "log_tau_matches_run": trained_tau == arm["log_tau_final"],
    }
    grid = [0.0, 2.0, 4.0, trained_tau, 6.0, needed, 9.0, 12.0]
    grid = sorted(set(round(value, 10) for value in grid))
    sweep_rows = sweep(graph, episodes, grid)
    set_tau(graph, trained_tau)
    zero_arm = zero_query_arm(graph, episodes, trained_tau)
    set_tau(graph, trained_tau)
    trained_features = features(graph, episodes, "trained_attention")
    set_tau(graph, needed)
    needed_features = features(graph, episodes, "trained_attention")
    set_tau(graph, trained_tau)
    oracle_features = features(graph, episodes, "oracle_soft")
    deranged = list(reversed(trained_features[2]))
    head_trained = vs.ce_head_probe(
        trained_features[0], trained_features[2], trained_features[1], trained_features[3]
    )
    head_needed = vs.ce_head_probe(
        needed_features[0], needed_features[2], needed_features[1], needed_features[3]
    )
    head_oracle = vs.ce_head_probe(
        oracle_features[0], oracle_features[2], oracle_features[1], oracle_features[3]
    )
    head_deranged = vs.ce_head_probe(
        trained_features[0], deranged, trained_features[1], trained_features[3]
    )
    census = attention_grad_census(graph, episodes)
    after = param_digest(graph)
    checks = {
        **{f"reproduce_{key}": value for key, value in reproduction.items()},
        "params_unchanged": before == after,
        "oracle_attention_exact": all(one["attention_exact"] for one in oracle_features[4]),
        "trained_attention_capture_ok": all(one["attended_shape"] for one in trained_features[4]),
        "deranged_head_below_band": head_deranged["fit_accuracy"] <= DERANGED_FIT_MAX,
        "sweep_covers_trained_point": any(
            row["log_tau"] == round(trained_tau, 10) for row in sweep_rows
        ),
    }
    max_fit = max(row["fit_accuracy"] for row in sweep_rows)
    max_train = max(row["train_accuracy"] for row in sweep_rows)
    at_trained = next(row for row in sweep_rows if row["log_tau"] == round(trained_tau, 10))
    at_needed = next(row for row in sweep_rows if row["log_tau"] == round(needed, 10))
    scale_rescues = max_fit >= FIT_GATE and max_train >= TRAIN_GATE
    scale_gain_at_needed = {
        "fit_delta": at_needed["fit_accuracy"] - at_trained["fit_accuracy"],
        "train_delta": at_needed["train_accuracy"] - at_trained["train_accuracy"],
        "focus_delta": at_needed["fit_focus"]["answer_mass_mean"]
        - at_trained["fit_focus"]["answer_mass_mean"],
    }
    value_decodable = (
        head_trained["fit_accuracy"] >= FIT_GATE and head_trained["train_accuracy"] >= TRAIN_GATE
    )
    value_decodable_needed = (
        head_needed["fit_accuracy"] >= FIT_GATE and head_needed["train_accuracy"] >= TRAIN_GATE
    )
    oracle_value_side = (
        head_oracle["fit_accuracy"] >= FIT_GATE and head_oracle["train_accuracy"] >= TRAIN_GATE
    )
    return {
        "seed": seed,
        "trained_log_tau": trained_tau,
        "needed_log_tau": needed,
        "trained_point": trained,
        "reproduction": reproduction,
        "sweep": sweep_rows,
        "sweep_max_fit": max_fit,
        "sweep_max_train": max_train,
        "scale_rescues_read": scale_rescues,
        "scale_gain_at_needed": scale_gain_at_needed,
        "w_q_zero_arm": zero_arm,
        "w_q_zero_delta": {
            "fit_delta": trained["fit_accuracy"] - zero_arm["fit_accuracy"],
            "train_delta": trained["train_accuracy"] - zero_arm["train_accuracy"],
        },
        "converged_head": {
            "trained_attention": head_trained,
            "needed_attention": head_needed,
            "oracle_soft_attention": head_oracle,
            "deranged_labels": head_deranged,
        },
        "value_side_decodable_trained_attention": value_decodable,
        "value_side_decodable_needed_attention": value_decodable_needed,
        "oracle_value_side_reaches_both_gates": oracle_value_side,
        "gradient_census": census,
        "instrument_checks": checks,
        "uncovered": [key for key, value in checks.items() if not value],
    }


def main() -> int:
    torch.set_num_threads(1)
    started_cpu, started_wall = time.process_time(), time.monotonic()
    contract = json.loads(r2.CONTRACT.read_text(encoding="utf-8"))
    r2.guards(contract, "probe")
    run_report = json.loads(r2.RUN_REPORT.read_text(encoding="utf-8"))
    run_arms = {int(arm["seed"]): arm for arm in run_report["arms"]}
    data = r2.u1.load_data(contract)
    episodes = r2.u1.prepare_episodes(data, UnitGraphSpec().n_units)
    geometry = vs.geometry_positive_control(episodes)
    input_reference = vs.input_side_reference(episodes)
    instrument = {
        "geometry_positive_control": geometry,
        "input_side_reference": input_reference,
        "geometry_positive_control_reaches_gate": geometry["fit_accuracy"] >= ORACLE_FIT_MIN,
    }
    seeds: list[dict[str, Any]] = []
    budget_stopped = False
    for seed in contract["criteria"]["seeds"]:
        if time.process_time() - started_cpu > CAP_CPU:
            budget_stopped = True
            break
        if time.monotonic() - started_wall > CAP_WALL:
            budget_stopped = True
            break
        seeds.append(probe_seed(contract, run_arms, int(seed), episodes))
    uncovered = [f"seed{row['seed']}:{key}" for row in seeds for key in row["uncovered"]]
    if not instrument["geometry_positive_control_reaches_gate"]:
        uncovered.append("geometry_positive_control_reaches_gate")
    rescued = [row["seed"] for row in seeds if row["scale_rescues_read"]]
    decodable = [row["seed"] for row in seeds if row["value_side_decodable_trained_attention"]]
    decodable_needed = [
        row["seed"] for row in seeds if row["value_side_decodable_needed_attention"]
    ]
    oracle_below = [row["seed"] for row in seeds if not row["oracle_value_side_reaches_both_gates"]]
    if budget_stopped:
        verdict = "budget_stop"
    elif uncovered:
        verdict = "instrument_invalid"
    elif rescued:
        verdict = "scale_can_rescue_read"
    elif decodable or decodable_needed:
        verdict = "value_side_decodable_head_or_credit_gap"
    else:
        verdict = "scale_and_value_both_short"
    report = {
        "schema": "taiji-u2-u1r2-posthoc-scale-value-v1",
        "card": "22.5.U.1r2",
        "phase": "posthoc_zero_training",
        "criteria_sha256": contract["criteria_sha256"],
        "verdict": verdict,
        "instrument": instrument,
        "scale_rescues_seeds": rescued,
        "value_decodable_trained_seeds": decodable,
        "value_decodable_needed_seeds": decodable_needed,
        "oracle_value_side_below_both_gates_seeds": oracle_below,
        "seeds": seeds,
        "caps": {"cap_cpu": CAP_CPU, "cap_wall": CAP_WALL},
        "discipline": {
            "optimizer_steps": 0,
            "weight_writes": 0,
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0,
            "sealed_partitions_read": [],
        },
        "cost": {
            "cpu": round(time.process_time() - started_cpu, 4),
            "wall": round(time.monotonic() - started_wall, 4),
        },
    }
    r2.write_report(REPORT, report)
    print(
        "POSTHOC verdict="
        + verdict
        + " rescued="
        + str(rescued)
        + " decodable="
        + str(decodable)
        + " decodable_needed="
        + str(decodable_needed)
        + " cpu="
        + str(report["cost"]["cpu"])
    )
    return 0 if (not uncovered and not budget_stopped) else 1


if __name__ == "__main__":
    sys.exit(main())
