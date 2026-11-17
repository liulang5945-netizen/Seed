"""22.5.V.2 zero-training selection-mechanism review.

V.1 raised the complete-seed top-1-on-span to 0.84/0.52 but the gate is 0.95.
This probe asks whether any of three zero-training mechanisms can reach 0.95 on
the unseen fit face, with the value side frozen:

  * A baseline: the parent's trained W_q attention top-1;
  * B pointer head: linear (16 params) or bilinear (64 params) per-column score,
    trained on train with an answer-span CE, scored on fit;
  * C shared W_q least squares: the r2-feasibility linear system with a per-scene
    offset, target = uniform over the answer span.

Zero optimizer steps, zero weight writes, train/fit only, no sealed reads.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training import probe_u1r2_scale_focus_20261116 as sf
from scripts.training import probe_u1r2_value_side_review_20261117 as vr
from scripts.training import run_u1_unit_read_20261116 as u1
from scripts.training import run_u1r1_position_20261116 as r1
from scripts.training import run_u1r2_scale_supervision_20261116 as r2
from taiji.u2_neural_units import UnitGraphSpec

REPORT = ROOT / "reports/taiji_u2_v2_selection_review_20261117.json"
CAP_CPU = 120.0
CAP_WALL = 240.0
SELECTION_GATE = 0.95
FIT_GATE = 0.80
DERANGED_MAX = 0.50
LBFGS_ITER = 300
LOG_EPS = 1e-12


def selection_rate(attention: torch.Tensor, span: torch.Tensor) -> int:
    return int(bool(span[int(torch.argmax(attention).item())]))


def pointer_head(records: list[dict[str, Any]], mode: str, target_mode: str) -> dict[str, Any]:
    train = [r for r in records if r["split"] == "train" and bool(r["span"].any())]
    fit = [r for r in records if r["split"] == "fit" and bool(r["span"].any())]
    if mode == "linear":
        params = torch.zeros(16, dtype=torch.float64, requires_grad=True)
    else:
        params = torch.zeros(8, 8, dtype=torch.float64, requires_grad=True)

    def scores(record: dict[str, Any]) -> torch.Tensor:
        q = record["item"]["channels"]["question"][0].to(torch.float64)
        if mode == "linear":
            features = torch.cat([record["projected"], q.unsqueeze(0).expand(16, 8)], dim=1)
            return features @ params  # type: ignore[no-any-return]
        return q @ params @ record["projected"].T  # type: ignore[no-any-return]

    def target(record: dict[str, Any]) -> torch.Tensor:
        span = record["span"].to(torch.float64)
        if target_mode == "onehot":
            out = torch.zeros(16, dtype=torch.float64)
            out[record["answer_col"]] = 1.0
            return out
        return span / span.sum()

    optimizer = torch.optim.LBFGS(
        [params],
        max_iter=LBFGS_ITER,
        history_size=50,
        tolerance_grad=1e-10,
        tolerance_change=1e-14,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = torch.zeros((), dtype=torch.float64)
        for record in train:
            log_probs = torch.log_softmax(scores(record), dim=0)
            loss = loss - (target(record) * log_probs).sum()
        loss.backward()
        return loss

    optimizer.step(closure)

    def evaluate(selected: list[dict[str, Any]], query_mode: str) -> dict[str, Any]:
        hits = 0
        mass = 0.0
        for record in selected:
            q = record["item"]["channels"]["question"][0].to(torch.float64)
            if query_mode == "wrong":
                q = q.clone()
                q[:6] = torch.roll(q[:6], 1)
            elif query_mode == "fixed":
                q = q.clone()
                q[:6] = 0.0
            if mode == "linear":
                features = torch.cat([record["projected"], q.unsqueeze(0).expand(16, 8)], dim=1)
                logits = features @ params
            else:
                logits = q @ params @ record["projected"].T
            attention = torch.softmax(logits, dim=0)
            hits += selection_rate(attention, record["span"])
            mass += float(attention[record["span"]].sum().detach())
        total = len(selected)
        return {
            "selection": hits / total if total else 0.0,
            "hits": hits,
            "denominator": total,
            "answer_mass": mass / total if total else 0.0,
        }

    return {
        "mode": mode,
        "target": target_mode,
        "fit": evaluate(fit, "normal"),
        "train": evaluate(train, "normal"),
        "fit_wrong_query": evaluate(fit, "wrong")["selection"],
        "fit_fixed_query": evaluate(fit, "fixed")["selection"],
        "param_count": int(params.numel()),
        "param_norm": float(torch.linalg.vector_norm(params.detach())),
    }


def wq_least_squares(
    records: list[dict[str, Any]], scale: float, target_mode: str
) -> dict[str, Any] | None:
    train = [r for r in records if r["split"] == "train" and bool(r["span"].any())]
    fit = [r for r in records if r["split"] == "fit" and bool(r["span"].any())]
    if not train:
        return None
    n = len(train)
    rows = n * 16
    cols = 64 + n
    design = torch.zeros(rows, cols, dtype=torch.float64)
    target = torch.zeros(rows, dtype=torch.float64)
    for index, record in enumerate(train):
        q = record["item"]["channels"]["question"][0].to(torch.float64)
        block = scale * torch.einsum("th,d->tdh", record["projected"], q).reshape(16, 64)
        design[index * 16 : (index + 1) * 16, :64] = block
        design[index * 16 : (index + 1) * 16, 64 + index] = 1.0
        span = record["span"].to(torch.float64)
        weights = span / span.sum()
        if target_mode == "onehot":
            weights = torch.zeros(16, dtype=torch.float64)
            weights[record["answer_col"]] = 1.0
        target[index * 16 : (index + 1) * 16] = torch.log(weights + LOG_EPS)
    try:
        solution = torch.linalg.lstsq(design, target, driver="gelsd").solution
    except RuntimeError:
        return None
    w_q = solution[:64].reshape(8, 8)

    def evaluate(selected: list[dict[str, Any]]) -> dict[str, Any]:
        hits = 0
        mass = 0.0
        for record in selected:
            q = record["item"]["channels"]["question"][0].to(torch.float64)
            logits = scale * record["projected"] @ (q @ w_q)
            attention = torch.softmax(logits, dim=0)
            hits += selection_rate(attention, record["span"])
            mass += float(attention[record["span"]].sum().detach())
        total = len(selected)
        return {
            "selection": hits / total if total else 0.0,
            "hits": hits,
            "denominator": total,
            "answer_mass": mass / total if total else 0.0,
        }

    return {
        "target": target_mode,
        "fit": evaluate(fit),
        "train": evaluate(train),
        "rank": int(torch.linalg.matrix_rank(design).item()),
        "w_q_norm": float(torch.linalg.vector_norm(w_q)),
    }


def probe_seed(
    contract: dict[str, Any],
    run_arms: dict[int, dict[str, Any]],
    seed: int,
    episodes: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    graph = vr.load_trained(seed)
    before = vr.param_digest(graph)
    scale = float(graph.base_attn_scale * torch.exp(graph.log_tau.detach()))
    records: list[dict[str, Any]] = []
    for split in ("train", "fit"):
        for item in episodes[split]:
            record = vr.capture(graph, item)
            record["split"] = split
            record["item"] = item
            records.append(record)
    reconstruction_max = max(record["reconstruction_max_abs"] for record in records)
    fit_records = [r for r in records if r["split"] == "fit" and bool(r["span"].any())]
    baseline_hits = sum(
        selection_rate(record["attention"], record["span"]) for record in fit_records
    )
    baseline = {
        "selection": baseline_hits / len(fit_records),
        "hits": baseline_hits,
        "denominator": len(fit_records),
    }
    pointer_linear = pointer_head(records, "linear", "span")
    pointer_bilinear = pointer_head(records, "bilinear", "span")
    pointer_onehot = pointer_head(records, "bilinear", "onehot")
    wq_span = wq_least_squares(records, scale, "span")
    wq_onehot = wq_least_squares(records, scale, "onehot")
    fit = r1.evaluate(graph, episodes["fit"])
    train = r1.evaluate(graph, episodes["train"])
    focus = sf.focus_stats(graph, episodes["fit"])
    arm = run_arms[seed]
    checks = {
        "reconstruction_exact": reconstruction_max <= 1e-12,
        "params_unchanged": before == vr.param_digest(graph),
        "fit_matches_run": fit["accuracy"] == arm["fit_accuracy"],
        "train_matches_run": train["accuracy"] == arm["train_accuracy"],
        "focus_matches_run": focus["answer_mass_mean"] == arm["fit_focus"]["answer_mass_mean"],
        "baseline_denominator_positive": len(fit_records) > 0,
    }
    pointer_best = max(pointer_linear["fit"]["selection"], pointer_bilinear["fit"]["selection"])
    wq_best = max(
        wq_span["fit"]["selection"] if wq_span else 0.0,
        wq_onehot["fit"]["selection"] if wq_onehot else 0.0,
    )
    return {
        "seed": seed,
        "scale": scale,
        "baseline": baseline,
        "pointer_linear": pointer_linear,
        "pointer_bilinear": pointer_bilinear,
        "pointer_bilinear_onehot": pointer_onehot,
        "wq_span": wq_span,
        "wq_onehot": wq_onehot,
        "pointer_best_selection": pointer_best,
        "wq_best_selection": wq_best,
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
    data = u1.load_data(contract)
    episodes = u1.prepare_episodes(data, UnitGraphSpec().n_units)
    geometry = vr.vs.geometry_positive_control(episodes)
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
    if geometry["fit_accuracy"] < FIT_GATE:
        uncovered.append("geometry_positive_control")
    pointer_ok = bool(seeds) and all(
        row["pointer_best_selection"] >= SELECTION_GATE for row in seeds
    )
    wq_ok = bool(seeds) and all(row["wq_best_selection"] >= SELECTION_GATE for row in seeds)
    pointer_some = any(row["pointer_best_selection"] >= SELECTION_GATE for row in seeds)
    wq_some = any(row["wq_best_selection"] >= SELECTION_GATE for row in seeds)
    if budget_stopped:
        verdict = "budget_stop"
    elif uncovered:
        verdict = "instrument_invalid"
    elif pointer_ok:
        verdict = "pointer_reaches_gate"
    elif wq_ok:
        verdict = "wq_lsq_reaches_gate"
    elif pointer_some or wq_some:
        verdict = "mixed"
    else:
        verdict = "neither_reaches_gate"
    report = {
        "schema": "taiji-u2-v2-selection-review-v1",
        "card": "22.5.V.2",
        "phase": "selection_review_zero_training",
        "criteria_sha256": contract["criteria_sha256"],
        "verdict": verdict,
        "geometry_positive_control": geometry,
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
        "SELECTION_REVIEW verdict="
        + verdict
        + " baseline="
        + str([round(row["baseline"]["selection"], 4) for row in seeds])
        + " pointer="
        + str([round(row["pointer_best_selection"], 4) for row in seeds])
        + " wq="
        + str([round(row["wq_best_selection"], 4) for row in seeds])
        + " cpu="
        + str(report["cost"]["cpu"])
    )
    return 0 if (not uncovered and not budget_stopped) else 1


if __name__ == "__main__":
    sys.exit(main())
