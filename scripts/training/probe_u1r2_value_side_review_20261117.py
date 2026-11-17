"""22.5.V value-side review: where in the value chain is the answer lost?

Zero training. The U.1r2 posthoc probe showed that scale is not the lever, the
query direction contributes only a little, and a converged linear head on the
attended vector is below the fit gate even under oracle focus. This probe asks the
next question: is the information missing from the per-column value, from the
position axis, from the attention average (background dilution), or from the
readout transformation?

Method:
  * capture the module's own ``projected`` [T,H] with a single einsum patch, never
    recomputing the chain outside the module;
  * per-column colour head ``projected -> 7`` (6 palette colours + background) on
    train columns, scored on fit columns; input-side colour head is the ruler;
  * position heads on: attended, oracle-soft weighted columns, the answer column
    value, the answer-span mean, [attended, answer column, present], quadratic
    [attended, attended^2], and the top-1 attended column;
  * background dominance: attention mass and value-norm ratio;
  * controls: geometry positive control, deranged labels, exact attended
    reconstruction, unchanged parameters, exact run reproduction.

Zero optimizer steps, zero weight writes, train/fit only, no check/final/confirm.
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
from scripts.training import probe_u1r2_value_side_20261116 as vs
from scripts.training import run_u1r2_scale_supervision_20261116 as r2
from taiji.u2_history import value_digest
from taiji.u2_neural_units import UnitGraphSpec, apply_capsule, load_capsule
from taiji.u2_unit_input import _MASS_FLOOR
from taiji.u2_unit_temperature_readout import TemperaturePositionGraph

REPORT = ROOT / "reports/taiji_u2_value_side_review_20261117.json"
CAP_CPU = 120.0
CAP_WALL = 240.0
T = 16
COLOR_WIDTH = 7
FIT_GATE = 0.80
TRAIN_GATE = 0.90
DERANGED_FIT_MAX = 0.50
INPUT_COLOR_MIN = 0.80
LBFGS_ITER = 300


def param_digest(graph: torch.nn.Module) -> str:
    return value_digest(
        {name: tensor.detach().clone() for name, tensor in graph.state_dict().items()}
    )


def load_trained(seed: int) -> TemperaturePositionGraph:
    capsule = load_capsule(r2.final_capsule_path(seed))
    graph = TemperaturePositionGraph(UnitGraphSpec(**capsule["runtime"]["spec"]))
    apply_capsule(graph, capsule)
    return graph


def capture(graph: TemperaturePositionGraph, item: dict[str, Any]) -> dict[str, Any]:
    data = r2.u1.cast_item(item, graph.spec.torch_dtype)
    real_einsum = torch.einsum
    captured: list[torch.Tensor] = []

    def patched(equation: str, *operands: Any, **kwargs: Any) -> torch.Tensor:
        out = real_einsum(equation, *operands, **kwargs)
        if equation == "hn,tn->th":
            captured.append(out.detach().clone())
        return out

    torch.einsum = patched  # type: ignore[assignment]
    try:
        with torch.no_grad():
            out = graph(data["channels"], data["gates"], graph.initial_state())
    finally:
        torch.einsum = real_einsum  # type: ignore[assignment]
    if len(captured) != 1:
        raise ValueError(f"einsum_capture_count:{len(captured)}")
    projected = captured[0].to(torch.float64)
    attention = out["attention"].detach().to(torch.float64)
    attended = out["attended"].detach().to(torch.float64)
    reconstruction = attention @ projected
    rgb = item["channels"]["rgb"].to(torch.float64)
    mass = vs.column_mass(item)
    total = rgb[:, 7]
    span = mass > 0.0
    answer_col = int(mass.argmax().item()) if float(mass.sum()) > 0.0 else 0
    colour = rgb[:, :6].argmax(-1)
    colour_label = torch.where(total >= _MASS_FLOOR, colour, torch.full_like(colour, 6))
    return {
        "projected": projected,
        "attention": attention,
        "attended": attended,
        "mass": mass,
        "span": span,
        "answer_col": answer_col,
        "present": float(mass.sum() >= _MASS_FLOOR),
        "colour_label": colour_label,
        "rgb": rgb,
        "reconstruction_max_abs": float((reconstruction - attended).abs().max()),
    }


def fit_head(
    train_x: torch.Tensor,
    train_labels: list[int],
    fit_x: torch.Tensor,
    fit_labels: list[int],
    width: int,
) -> dict[str, Any]:
    weights = torch.zeros(train_x.shape[1], width, dtype=torch.float64, requires_grad=True)
    target = torch.tensor(train_labels, dtype=torch.long)
    fit_target = torch.tensor(fit_labels, dtype=torch.long)
    optimizer = torch.optim.LBFGS(
        [weights],
        max_iter=LBFGS_ITER,
        history_size=50,
        tolerance_grad=1e-10,
        tolerance_change=1e-14,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(train_x @ weights, target)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        train_logits = train_x @ weights
        fit_logits = fit_x @ weights
        train_pred = train_logits.argmax(-1)
        fit_pred = fit_logits.argmax(-1)
        train_loss = float(torch.nn.functional.cross_entropy(train_logits, target))
    return {
        "fit_accuracy": float((fit_pred == fit_target).to(torch.float64).mean()),
        "train_accuracy": float((train_pred == target).to(torch.float64).mean()),
        "fit_distinct": len(set(fit_pred.tolist())),
        "train_loss": train_loss,
        "fit_pred": fit_pred,
        "train_pred": train_pred,
        "fit_target": fit_target,
        "train_target": target,
    }


def head_summary(head: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in head.items()
        if key not in ("fit_pred", "train_pred", "fit_target", "train_target")
    }


def column_rows(
    records: list[dict[str, Any]], side: str
) -> tuple[torch.Tensor, list[int], torch.Tensor, list[int]]:
    train_x: list[torch.Tensor] = []
    train_y: list[int] = []
    fit_x: list[torch.Tensor] = []
    fit_y: list[int] = []
    for record in records:
        for column in range(T):
            if side == "value":
                row = torch.cat([record["projected"][column], torch.ones(1, dtype=torch.float64)])
            else:
                row = torch.cat([record["rgb"][column, :6], torch.ones(1, dtype=torch.float64)])
            if record["split"] == "train":
                train_x.append(row)
                train_y.append(int(record["colour_label"][column]))
            else:
                fit_x.append(row)
                fit_y.append(int(record["colour_label"][column]))
    return torch.stack(train_x), train_y, torch.stack(fit_x), fit_y


def scene_rows(
    records: list[dict[str, Any]], feature: str
) -> tuple[torch.Tensor, list[int], torch.Tensor, list[int]]:
    train_x: list[torch.Tensor] = []
    train_y: list[int] = []
    fit_x: list[torch.Tensor] = []
    fit_y: list[int] = []
    for record in records:
        if feature == "attended":
            row = record["attended"]
        elif feature == "oracle":
            target = vs.forced_distribution(record["item"], "oracle_soft").to(torch.float64)
            row = target @ record["projected"]
        elif feature == "answer":
            row = torch.cat(
                [
                    record["projected"][record["answer_col"]],
                    torch.tensor([record["present"]], dtype=torch.float64),
                ]
            )
        elif feature == "masked":
            span = record["span"]
            value = (
                record["projected"][span].mean(0)
                if bool(span.any())
                else torch.zeros(record["projected"].shape[1], dtype=torch.float64)
            )
            row = torch.cat([value, torch.tensor([record["present"]], dtype=torch.float64)])
        elif feature == "concat":
            row = torch.cat(
                [
                    record["attended"],
                    record["projected"][record["answer_col"]],
                    torch.tensor([record["present"]], dtype=torch.float64),
                ]
            )
        elif feature == "quadratic":
            row = torch.cat([record["attended"], record["attended"].square()])
        elif feature == "top1":
            column = int(record["attention"].argmax().item())
            row = torch.cat(
                [
                    record["projected"][column],
                    torch.tensor([record["present"]], dtype=torch.float64),
                ]
            )
        else:
            raise ValueError(f"unknown_feature:{feature}")
        if record["split"] == "train":
            train_x.append(row)
            train_y.append(int(record["item"]["label"]))
        else:
            fit_x.append(row)
            fit_y.append(int(record["item"]["label"]))
    return torch.stack(train_x), train_y, torch.stack(fit_x), fit_y


def painted_accuracy(head: dict[str, Any]) -> dict[str, float]:
    fit_pred = head["fit_pred"]
    fit_target = head["fit_target"]
    painted = fit_target != 6
    return {
        "fit_painted_accuracy": (
            float((fit_pred[painted] == fit_target[painted]).to(torch.float64).mean())
            if bool(painted.any())
            else 0.0
        ),
        "fit_painted_denominator": int(painted.sum()),
        "fit_background_accuracy": (
            float((fit_pred[~painted] == fit_target[~painted]).to(torch.float64).mean())
            if bool((~painted).any())
            else 0.0
        ),
    }


def background_stats(records: list[dict[str, Any]], split: str) -> dict[str, float]:
    masses: list[float] = []
    ratios: list[float] = []
    for record in records:
        if record["split"] != split:
            continue
        background = record["rgb"][:, 7] < _MASS_FLOOR
        masses.append(float(record["attention"][background].sum()))
        painted = ~background
        if bool(painted.any()) and bool(background.any()):
            bg_norm = float(torch.linalg.vector_norm(record["projected"][background].sum(0)))
            fg_norm = float(torch.linalg.vector_norm(record["projected"][painted].sum(0)))
            if fg_norm > 0.0:
                ratios.append(bg_norm / fg_norm)
    return {
        "background_attention_mean": sum(masses) / len(masses) if masses else 0.0,
        "background_value_norm_ratio_mean": sum(ratios) / len(ratios) if ratios else 0.0,
    }


def classify(
    column_painted: float,
    heads: dict[str, dict[str, Any]],
    input_colour: dict[str, Any],
) -> str:
    if input_colour["fit_accuracy"] < INPUT_COLOR_MIN:
        return "instrument_invalid"
    column_ok = column_painted >= FIT_GATE
    answered = all(
        heads[name]["fit_accuracy"] >= FIT_GATE and heads[name]["train_accuracy"] >= TRAIN_GATE
        for name in ("answer", "masked", "oracle", "concat")
    )
    any_answer = any(
        heads[name]["fit_accuracy"] >= FIT_GATE and heads[name]["train_accuracy"] >= TRAIN_GATE
        for name in ("answer", "masked", "oracle", "concat")
    )
    trained_ok = (
        heads["attended"]["fit_accuracy"] >= FIT_GATE
        and heads["attended"]["train_accuracy"] >= TRAIN_GATE
    )
    extra_ok = any(
        heads[name]["fit_accuracy"] >= FIT_GATE and heads[name]["train_accuracy"] >= TRAIN_GATE
        for name in ("quadratic", "concat")
    )
    if not column_ok and not any_answer:
        return "value_absent_in_columns"
    if column_ok and not any_answer:
        return "value_present_position_absent"
    if column_ok and extra_ok and not trained_ok and not answered:
        return "value_present_needs_extra_features"
    if (
        column_ok
        and any(
            heads[name]["fit_accuracy"] >= FIT_GATE and heads[name]["train_accuracy"] >= TRAIN_GATE
            for name in ("answer", "masked")
        )
        and not trained_ok
    ):
        return "value_present_readout_gap"
    if trained_ok:
        return "value_already_readable"
    return "mixed"


def probe_seed(
    contract: dict[str, Any],
    run_arms: dict[int, dict[str, Any]],
    seed: int,
    episodes: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    graph = load_trained(seed)
    before = param_digest(graph)
    records: list[dict[str, Any]] = []
    for split in ("train", "fit"):
        for item in episodes[split]:
            record = capture(graph, item)
            record["split"] = split
            record["item"] = item
            records.append(record)
    reconstruction_max = max(record["reconstruction_max_abs"] for record in records)
    fit = r2.r1.evaluate(graph, episodes["fit"])
    train = r2.r1.evaluate(graph, episodes["train"])
    focus = sf.focus_stats(graph, episodes["fit"])
    arm = run_arms[seed]
    colour_train_x, colour_train_y, colour_fit_x, colour_fit_y = column_rows(records, "value")
    input_train_x, input_train_y, input_fit_x, input_fit_y = column_rows(records, "rgb")
    colour = fit_head(colour_train_x, colour_train_y, colour_fit_x, colour_fit_y, COLOR_WIDTH)
    input_colour = fit_head(input_train_x, input_train_y, input_fit_x, input_fit_y, COLOR_WIDTH)
    heads: dict[str, dict[str, Any]] = {}
    for name in ("attended", "oracle", "answer", "masked", "concat", "quadratic", "top1"):
        train_x, train_y, fit_x, fit_y = scene_rows(records, name)
        heads[name] = head_summary(fit_head(train_x, train_y, fit_x, fit_y, 4))
    train_x, train_y, fit_x, fit_y = scene_rows(records, "concat")
    deranged = list(reversed(train_y))
    deranged_head = fit_head(train_x, deranged, fit_x, fit_y, 4)
    painted = painted_accuracy(colour)
    verdict = classify(painted["fit_painted_accuracy"], heads, input_colour)
    after = param_digest(graph)
    checks = {
        "reconstruction_exact": reconstruction_max <= 1e-12,
        "params_unchanged": before == after,
        "fit_matches_run": fit["accuracy"] == arm["fit_accuracy"],
        "train_matches_run": train["accuracy"] == arm["train_accuracy"],
        "focus_matches_run": (
            focus["answer_mass_mean"] == arm["fit_focus"]["answer_mass_mean"]
            and focus["focus_met"] == arm["fit_focus"]["focus_met"]
        ),
        "input_colour_reaches_gate": input_colour["fit_accuracy"] >= INPUT_COLOR_MIN,
        "deranged_head_below_band": deranged_head["fit_accuracy"] <= DERANGED_FIT_MAX,
    }
    return {
        "seed": seed,
        "verdict": verdict,
        "colour_head": {**head_summary(colour), **painted},
        "input_colour_head": head_summary(input_colour),
        "heads": heads,
        "deranged_head": {
            "fit_accuracy": deranged_head["fit_accuracy"],
            "train_accuracy": deranged_head["train_accuracy"],
        },
        "background": {
            "fit": background_stats(records, "fit"),
            "train": background_stats(records, "train"),
        },
        "attention_focus": focus,
        "reconstruction_max_abs": reconstruction_max,
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
    verdicts = [row["verdict"] for row in seeds]
    majority = max(set(verdicts), key=verdicts.count) if verdicts else "no_seed"
    verdict = majority if verdicts.count(majority) >= 2 else "mixed"
    if budget_stopped:
        verdict = "budget_stop"
    elif uncovered:
        verdict = "instrument_invalid"
    report = {
        "schema": "taiji-u2-value-side-review-v1",
        "card": "22.5.V",
        "phase": "value_side_review_zero_training",
        "criteria_sha256": contract["criteria_sha256"],
        "verdict": verdict,
        "seed_verdicts": verdicts,
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
        "VALUE_REVIEW verdict="
        + verdict
        + " seeds="
        + str(verdicts)
        + " cpu="
        + str(report["cost"]["cpu"])
    )
    return 0 if (not uncovered and not budget_stopped) else 1


if __name__ == "__main__":
    sys.exit(main())
