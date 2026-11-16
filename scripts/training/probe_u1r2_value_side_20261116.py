"""22.5.U.1r2 design-phase probe: is the value side decodable under perfect focus?

U.1r1 closed negative with a located cause (attention never focused) and left one
item explicitly untested: whether the attended column vector carries enough
information for the linear position head to decode the label. Owner ruled option B
for revision 2 (logit scale correction + training-label-only attention supervision)
on 2026-11-16. This probe measures the ceiling of that revision BEFORE any training.

Method (zero model training, no product-code edit):
  * reuse the U.1r1 runner's graph builder, episode preparation, guards and capsule
    loading;
  * force the attention distribution by patching ``torch.softmax`` for the duration
    of one forward, so the module's own chain computes ``attended`` from the forced
    selection (nothing is recomputed outside the module);
  * fit a linear, no-intercept head on the TRAIN attended vectors with the SAME
    cross-entropy objective the model uses, by LBFGS to convergence (the problem is
    convex, so this is the best head of that class, not one Adam trajectory), and
    score it on the FIT vectors. ``position_head`` is ``nn.Linear(H, 4, bias=False)``;
  * a ridge-to-one-hot least-squares fit is reported alongside, because it is a
    weaker fitting procedure and must not be mistaken for an information bound;
  * controls: deranged training labels, a constant wrong column, the unpatched
    trained attention, the raw encoded RGB row (input-side reference), a quadratic
    feature reference (is the information present but not linearly accessible?), a
    label-geometry positive control (can this instrument reach the gate at all?),
    and the majority-class baseline (what "chance" means on this label distribution).

Instrument validity is fail-closed: the patch must be called exactly once per
forward, the returned attention must equal the forced distribution bit-exactly,
every model parameter must be unchanged (digest before == after), and the geometry
positive control must itself reach the fit gate.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training import run_u1_unit_read_20261116 as u1
from scripts.training import run_u1r1_position_20261116 as r1
from taiji.u2_history import value_digest
from taiji.u2_neural_units import UnitGraphSpec, apply_capsule, load_capsule
from taiji.u2_task_episode import NAMES, position_word
from taiji.u2_unit_input import _MASS_FLOOR, _X_CENTER

REPORT = ROOT / "reports/taiji_u2_u1r2_value_side_20261116.json"
T = 16
RIDGE_LAMBDA = 1e-6
LBFGS_ITER = 300
FIT_GATE = 0.8
TRAIN_GATE = 0.9
CHANCE_BAND = 0.5
FOCUS_ARMS = ("oracle_soft", "oracle_onehot", "wrong_column", "trained_attention")
SEEDS = (2026111601, 2026111602, 2026111603)


def param_digest(graph: Any) -> str:
    return value_digest(
        {name: tensor.detach().clone() for name, tensor in graph.state_dict().items()}
    )


def column_mass(item: dict[str, Any]) -> torch.Tensor:
    """Queried-entity colour mass per column, read from the frozen encoding."""
    rgb = item["channels"]["rgb"]
    index = NAMES.index(item["entity"])
    return cast(torch.Tensor, rgb[:, index].to(torch.float64))


def weighted_x(item: dict[str, Any]) -> float:
    mass = column_mass(item)
    total = float(mass.sum())
    if total <= 0.0:
        return 0.0
    return float((mass * _X_CENTER).sum() / total)


def forced_distribution(item: dict[str, Any], arm: str) -> torch.Tensor:
    mass = column_mass(item)
    if arm == "wrong_column":
        out = torch.zeros(T, dtype=torch.float64)
        out[0] = 1.0
        return out
    total = float(mass.sum())
    if total <= 0.0:
        return torch.full((T,), 1.0 / T, dtype=torch.float64)
    if arm == "oracle_onehot":
        out = torch.zeros(T, dtype=torch.float64)
        out[int(torch.argmax(mass).item())] = 1.0
        return out
    return mass / total


def attended_under_focus(
    graph: Any, item: dict[str, Any], arm: str
) -> tuple[torch.Tensor, dict[str, Any]]:
    data = u1.cast_item(item, graph.spec.torch_dtype)
    target = forced_distribution(item, arm).to(graph.spec.torch_dtype)
    real_softmax = torch.softmax
    calls: list[int] = []

    def patched(input: torch.Tensor, dim: int | None = None, **kwargs: Any) -> torch.Tensor:
        if dim == 0 and tuple(input.shape) == (T,):
            calls.append(1)
            return target
        return real_softmax(input, dim=dim, **kwargs)  # type: ignore[arg-type]

    if arm == "trained_attention":
        with torch.no_grad():
            out = graph(data["channels"], data["gates"], graph.initial_state())
    else:
        torch.softmax = patched  # type: ignore[assignment]
        try:
            with torch.no_grad():
                out = graph(data["channels"], data["gates"], graph.initial_state())
        finally:
            torch.softmax = real_softmax  # type: ignore[assignment]
    attention = out["attention"]
    checks = {
        "softmax_calls": len(calls),
        "attention_exact": (
            bool(torch.equal(attention, target)) if arm != "trained_attention" else True
        ),
        "attended_shape": tuple(out["attended"].shape) == (graph.spec.hidden_dim,),
    }
    return cast(torch.Tensor, out["attended"]).detach().to(torch.float64), checks


def ce_head_probe(
    train_x: torch.Tensor, train_labels: list[int], fit_x: torch.Tensor, fit_labels: list[int]
) -> dict[str, Any]:
    """Best linear no-bias head under cross-entropy (convex ⇒ LBFGS to convergence)."""
    width = int(max(train_labels + fit_labels)) + 1
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
        "train_accuracy": float((train_pred == target).to(torch.float64).mean()),
        "fit_accuracy": float((fit_pred == fit_target).to(torch.float64).mean()),
        "fit_distinct": len(set(fit_pred.tolist())),
        "train_loss": train_loss,
        "head_weight_norm": float(torch.linalg.vector_norm(weights.detach())),
        "feature_dim": train_x.shape[1],
    }


def ridge_probe(
    train_x: torch.Tensor, train_y: torch.Tensor, fit_x: torch.Tensor, fit_y: torch.Tensor
) -> dict[str, Any]:
    eye = torch.eye(train_x.shape[1], dtype=torch.float64)
    gram = train_x.T @ train_x + RIDGE_LAMBDA * eye
    weights = torch.linalg.solve(gram, train_x.T @ train_y)
    train_pred = (train_x @ weights).argmax(-1)
    fit_pred = (fit_x @ weights).argmax(-1)
    return {
        "ridge_train_accuracy": float((train_pred == train_y.argmax(-1)).to(torch.float64).mean()),
        "ridge_fit_accuracy": float((fit_pred == fit_y.argmax(-1)).to(torch.float64).mean()),
    }


def dispersion(features: torch.Tensor) -> dict[str, float]:
    """Cross-scene variability of the value vector (C-line record-invariance check)."""
    centred = features - features.mean(0, keepdim=True)
    total = float(torch.linalg.vector_norm(features))
    return {
        "relative_dispersion": float(torch.linalg.vector_norm(centred)) / total if total else 0.0,
        "mean_abs": float(features.abs().mean()),
        "mean_std": float(features.std(0).mean()),
    }


def one_hot(labels: list[int], width: int) -> torch.Tensor:
    out = torch.zeros(len(labels), width, dtype=torch.float64)
    for row, label in enumerate(labels):
        out[row, label] = 1.0
    return out


def dataset_facts(episodes: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    spread: dict[str, int] = {}
    unknown = 0
    argmax_agrees = 0
    label_from_x_agrees = 0
    counts: dict[str, Counter] = {"train": Counter(), "fit": Counter()}
    for split in ("train", "fit"):
        for item in episodes[split]:
            mass = column_mass(item)
            present = int((mass > 0).sum())
            spread[str(present)] = spread.get(str(present), 0) + 1
            counts[split][int(item["label"])] += 1
            if float(mass.sum()) < _MASS_FLOOR:
                unknown += 1
                continue
            x = weighted_x(item)
            argmax_agrees += int(
                position_word(float(_X_CENTER[int(torch.argmax(mass).item())])) == position_word(x)
            )
            label_from_x_agrees += int(u1.LABELS.index(position_word(x)) == int(item["label"]))
    return {
        "train": len(episodes["train"]),
        "fit": len(episodes["fit"]),
        "below_mass_floor": unknown,
        "answer_column_spread": spread,
        "argmax_x_matches_weighted_x_label": argmax_agrees,
        "label_equals_position_word_of_weighted_x": label_from_x_agrees,
        "labelled_scenes": len(episodes["train"]) + len(episodes["fit"]) - unknown,
        "label_counts_train": {str(k): v for k, v in sorted(counts["train"].items())},
        "label_counts_fit": {str(k): v for k, v in sorted(counts["fit"].items())},
        "majority_class_fit_accuracy": max(counts["fit"].values()) / len(episodes["fit"]),
    }


def split_features(
    episodes: dict[str, list[dict[str, Any]]], row_of: Any
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int]]:
    features: dict[str, list[torch.Tensor]] = {"train": [], "fit": []}
    labels: dict[str, list[int]] = {"train": [], "fit": []}
    for split in ("train", "fit"):
        for item in episodes[split]:
            features[split].append(row_of(item))
            labels[split].append(int(item["label"]))
    return (
        torch.stack(features["train"]),
        torch.stack(features["fit"]),
        labels["train"],
        labels["fit"],
    )


def input_side_reference(episodes: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Same head on the mass-weighted raw encoded RGB row (8 features)."""

    def row_of(item: dict[str, Any]) -> torch.Tensor:
        mass = column_mass(item)
        total = float(mass.sum())
        rgb = item["channels"]["rgb"].to(torch.float64)
        return (mass[:, None] * rgb).sum(0) / total if total > 0 else rgb.mean(0)

    train_x, fit_x, train_labels, fit_labels = split_features(episodes, row_of)
    out = ce_head_probe(train_x, train_labels, fit_x, fit_labels)
    out["dispersion"] = dispersion(train_x)  # type: ignore[assignment]
    return out


def geometry_positive_control(episodes: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """[x, x^2, present, 1] of the oracle weighted centre: the label is an interval test.

    ``present`` separates below-floor scenes (label unknown) from genuine centre
    scenes, whose weighted centre is the same point x=0. The constant column is
    required because the head has no intercept: without it every score at an
    absent scene's all-zero feature vector ties, so the unknown class can never
    win. If this control cannot reach the fit gate, the instrument itself cannot
    demonstrate decodability and every negative reading below is void.
    """

    def row_of(item: dict[str, Any]) -> torch.Tensor:
        mass = column_mass(item)
        present = float(float(mass.sum()) >= _MASS_FLOOR)
        x = weighted_x(item) if present else 0.0
        return torch.tensor([x, x * x, present, 1.0], dtype=torch.float64)

    train_x, fit_x, train_labels, fit_labels = split_features(episodes, row_of)
    return ce_head_probe(train_x, train_labels, fit_x, fit_labels)


def state_breakdown(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-weight-state readings: a max over both states must not be read as both."""
    out: dict[str, Any] = {}
    for state in ("birth", "r1_trained"):
        selected = [r for r in rows if r["weight_state"] == state]
        soft = [r for r in selected if r["focus_arm"] == "oracle_soft"]
        out[state] = {
            "oracle_soft_fit_min": min(r["fit_accuracy"] for r in soft),
            "oracle_soft_fit_max": max(r["fit_accuracy"] for r in soft),
            "oracle_soft_train_min": min(r["train_accuracy"] for r in soft),
            "oracle_soft_seeds_at_fit_gate": [
                r["seed"] for r in soft if r["fit_accuracy"] >= FIT_GATE
            ],
            "oracle_soft_seeds_at_train_gate": [
                r["seed"] for r in soft if r["train_accuracy"] >= TRAIN_GATE
            ],
            "oracle_soft_seeds_at_both_gates": [
                r["seed"]
                for r in soft
                if r["fit_accuracy"] >= FIT_GATE and r["train_accuracy"] >= TRAIN_GATE
            ],
            "wrong_column_fit_max": max(
                r["fit_accuracy"] for r in selected if r["focus_arm"] == "wrong_column"
            ),
            "trained_attention_fit_max": max(
                r["fit_accuracy"] for r in selected if r["focus_arm"] == "trained_attention"
            ),
            "permuted_label_fit_max": max(r["permuted_label_fit_accuracy"] for r in soft),
            "relative_dispersion_min": min(r["dispersion"]["relative_dispersion"] for r in soft),
            "relative_dispersion_max": max(r["dispersion"]["relative_dispersion"] for r in soft),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-report", default=str(REPORT))
    args = parser.parse_args()
    torch.set_num_threads(1)
    started_cpu, started_wall = time.process_time(), time.monotonic()
    contract = json.loads(r1.CONTRACT.read_text(encoding="utf-8"))
    r1.guards(contract, "probe")
    data = u1.load_data(contract)
    episodes = u1.prepare_episodes(data, UnitGraphSpec().n_units)
    width = len(u1.LABELS)
    facts = dataset_facts(episodes)
    geometry = geometry_positive_control(episodes)
    rows: list[dict[str, Any]] = []
    instrument: dict[str, bool] = {}
    for weight_state in ("birth", "r1_trained"):
        for seed in SEEDS:
            if weight_state == "birth":
                graph = r1.build_graph(seed)
            else:
                capsule = load_capsule(r1.final_capsule_path(seed))
                graph = r1.PositionRouteBGraph(UnitGraphSpec(**capsule["runtime"]["spec"]))
                apply_capsule(graph, capsule)
            before = param_digest(graph)
            for arm in FOCUS_ARMS:
                features: dict[str, list[torch.Tensor]] = {"train": [], "fit": []}
                labels: dict[str, list[int]] = {"train": [], "fit": []}
                arm_checks: dict[str, bool] = {}
                for split in ("train", "fit"):
                    for index, item in enumerate(episodes[split]):
                        attended, checks = attended_under_focus(graph, item, arm)
                        features[split].append(attended)
                        labels[split].append(int(item["label"]))
                        for key, value in checks.items():
                            if key == "softmax_calls":
                                expected = 0 if arm == "trained_attention" else 1
                                arm_checks[f"{split}{index}_{key}"] = value == expected
                            else:
                                arm_checks[f"{split}{index}_{key}"] = bool(value)
                train_x = torch.stack(features["train"])
                fit_x = torch.stack(features["fit"])
                probe = ce_head_probe(train_x, labels["train"], fit_x, labels["fit"])
                ridge = ridge_probe(
                    train_x, one_hot(labels["train"], width), fit_x, one_hot(labels["fit"], width)
                )
                deranged = u1._derangement(len(labels["train"]))
                permuted = ce_head_probe(train_x, deranged, fit_x, labels["fit"])
                quadratic = ce_head_probe(
                    torch.cat([train_x, train_x**2], dim=-1),
                    labels["train"],
                    torch.cat([fit_x, fit_x**2], dim=-1),
                    labels["fit"],
                )
                rows.append(
                    {
                        "weight_state": weight_state,
                        "seed": seed,
                        "focus_arm": arm,
                        **probe,
                        **ridge,
                        "dispersion": dispersion(train_x),
                        "permuted_label_fit_accuracy": permuted["fit_accuracy"],
                        "quadratic_fit_accuracy": quadratic["fit_accuracy"],
                        "quadratic_train_accuracy": quadratic["train_accuracy"],
                        "quadratic_train_loss": quadratic["train_loss"],
                        "instrument_checks_green": sum(arm_checks.values()),
                        "instrument_checks_total": len(arm_checks),
                    }
                )
                instrument[f"{weight_state}_{seed}_{arm}"] = all(arm_checks.values())
            after = param_digest(graph)
            instrument[f"{weight_state}_{seed}_params_unchanged"] = before == after
    soft = [r["fit_accuracy"] for r in rows if r["focus_arm"] == "oracle_soft"]
    onehot = [r["fit_accuracy"] for r in rows if r["focus_arm"] == "oracle_onehot"]
    wrong = [r["fit_accuracy"] for r in rows if r["focus_arm"] == "wrong_column"]
    perm = [r["permuted_label_fit_accuracy"] for r in rows if r["focus_arm"] == "oracle_soft"]
    quad = [r["quadratic_fit_accuracy"] for r in rows if r["focus_arm"] == "oracle_soft"]
    ridge_soft = [r["ridge_fit_accuracy"] for r in rows if r["focus_arm"] == "oracle_soft"]
    reference = input_side_reference(episodes)
    best_soft = max(soft)
    best_quad = max(quad)
    breakdown = state_breakdown(rows)
    instrument["gates_match_frozen_r1_contract"] = (
        float(contract["criteria"]["fit_gate"]) == FIT_GATE
        and float(contract["criteria"]["train_gate"]) == TRAIN_GATE
        and float(contract["criteria"]["control_agreement_max"]) == CHANCE_BAND
        and tuple(SEEDS) == tuple(contract["criteria"]["seeds"])
    )
    instrument["geometry_positive_control_reaches_fit_gate"] = geometry["fit_accuracy"] >= FIT_GATE
    instrument["label_is_position_word_of_weighted_x"] = (
        facts["label_equals_position_word_of_weighted_x"] == facts["labelled_scenes"]
    )
    if not all(instrument.values()):
        verdict = "instrument_invalid"
    elif best_soft >= FIT_GATE and max(perm) <= CHANCE_BAND and max(wrong) < best_soft:
        verdict = "value_side_decodable"
    elif best_soft < FIT_GATE <= best_quad:
        verdict = "value_side_nonlinear_only"
    elif best_soft < FIT_GATE:
        verdict = "value_side_below_fit_gate"
    else:
        verdict = "value_side_partial"
    payload = {
        "schema": "taiji-u2-u1r2-value-side-probe-v4",
        "card": "22.5.U.1r2",
        "phase": "design_value_side_oracle_probe",
        "date": "2026-11-16",
        "authority": (
            "owner 2026-11-16 ruled option B (scale correction + attention supervision); "
            "this is a design-phase measurement with zero model training and zero product-code edit"
        ),
        "criteria": {
            "head_class": "linear_no_bias_same_as_position_head",
            "head_objective": "cross_entropy_same_as_task_loss",
            "head_fit": "LBFGS_convex_to_convergence",
            "lbfgs_max_iter": LBFGS_ITER,
            "ridge_lambda": RIDGE_LAMBDA,
            "probe_fit_split": "train",
            "probe_eval_split": "fit",
            "fit_gate": FIT_GATE,
            "chance_band": CHANCE_BAND,
            "focus_arms": list(FOCUS_ARMS),
            "seeds": list(SEEDS),
            "quadratic_reference": "same head on [v, v*v]; 2H features vs 64 train rows",
            "verdict_branches": [
                "instrument_invalid",
                "value_side_decodable",
                "value_side_nonlinear_only",
                "value_side_below_fit_gate",
                "value_side_partial",
            ],
        },
        "labels": list(u1.LABELS),
        "dataset": facts,
        "rows": rows,
        "input_side_reference": reference,
        "geometry_positive_control": geometry,
        "by_weight_state": breakdown,
        "verdict_scope": {
            "claim_form": "existence over the six (weight_state, seed) oracle_soft rows",
            "weight_states_reaching_fit_gate": [
                state
                for state, block in breakdown.items()
                if block["oracle_soft_fit_max"] >= FIT_GATE
            ],
            "weight_states_below_fit_gate": [
                state
                for state, block in breakdown.items()
                if block["oracle_soft_fit_max"] < FIT_GATE
            ],
        },
        "derived": {
            "oracle_soft_fit_min": min(soft),
            "oracle_soft_fit_max": best_soft,
            "oracle_soft_train_loss_max": max(
                r["train_loss"] for r in rows if r["focus_arm"] == "oracle_soft"
            ),
            "oracle_onehot_fit_min": min(onehot),
            "oracle_onehot_fit_max": max(onehot),
            "wrong_column_fit_max": max(wrong),
            "permuted_label_fit_max": max(perm),
            "quadratic_fit_min": min(quad),
            "quadratic_fit_max": best_quad,
            "ridge_oracle_soft_fit_min": min(ridge_soft),
            "ridge_oracle_soft_fit_max": max(ridge_soft),
            "oracle_soft_relative_dispersion_min": min(
                r["dispersion"]["relative_dispersion"]
                for r in rows
                if r["focus_arm"] == "oracle_soft"
            ),
            "oracle_soft_relative_dispersion_max": max(
                r["dispersion"]["relative_dispersion"]
                for r in rows
                if r["focus_arm"] == "oracle_soft"
            ),
            "instrument_checks_total": len(instrument),
            "instrument_checks_green": sum(instrument.values()),
            "geometry_control_fit_accuracy": geometry["fit_accuracy"],
            "birth_oracle_soft_fit_max": breakdown["birth"]["oracle_soft_fit_max"],
            "r1_trained_oracle_soft_fit_max": breakdown["r1_trained"]["oracle_soft_fit_max"],
        },
        "instrument": instrument,
        "verdict": verdict,
        "untested": (
            "whether 384 supervised updates can reach this ceiling: the probe freezes the graph "
            "weights and fits the head to convergence on the train split only, so it bounds the "
            "readout head at these weights, not what training the substrate could achieve"
        ),
        "cost": {
            "cpu": round(time.process_time() - started_cpu, 4),
            "wall": round(time.monotonic() - started_wall, 4),
        },
    }
    u1.write_report(Path(args.out_report), payload)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "oracle_soft_fit_max": best_soft,
                "oracle_soft_fit_min": min(soft),
                "birth_fit_max": breakdown["birth"]["oracle_soft_fit_max"],
                "r1_trained_fit_max": breakdown["r1_trained"]["oracle_soft_fit_max"],
                "r1_trained_seeds_both_gates": breakdown["r1_trained"][
                    "oracle_soft_seeds_at_both_gates"
                ],
                "quadratic_fit_max": best_quad,
                "geometry_control_fit": geometry["fit_accuracy"],
                "input_side_fit": reference["fit_accuracy"],
                "majority_fit": facts["majority_class_fit_accuracy"],
                "instrument_green": sum(instrument.values()),
                "instrument_total": len(instrument),
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=True,
        )
    )
    return 0 if verdict != "instrument_invalid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
