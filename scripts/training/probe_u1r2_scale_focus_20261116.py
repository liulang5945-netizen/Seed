"""22.5.U.1r2 design calibration: query-logit multiplier + focus ruler (TRAIN face only).

Owner ruling D-20261116-24 picked option 乙 (logit scale correction + attention
supervision) and explicitly deferred training: design and pricing first, zero
training, a fresh authorization request at the measured price. This probe is part
of that design phase. It takes ZERO optimizer steps and measures only the train
partition; the fit partition is encoded by ``prepare_episodes`` but never measured.

Three questions, all answered on train:

1. Scale. r1 located the gap as a flat query logit range (0.025293-0.203172 nats on
   fit). Logits are linear in W_q, so multiplying W_q by ``c`` multiplies the logit
   range by exactly ``c``. The probe measures each seed's base range at birth on
   train, derives the per-seed ``c`` that puts it at a declared target, and then
   VERIFIES the identity by running on a multiplier grid. A single global ``c`` is
   also measured and reported: v1 found the birth spread across seeds wider than
   the declared band, so the global option is rejected on a reading, not on taste.
   Folding sqrt(hidden_dim) into ``c`` is algebraically identical to removing the
   module's 1/sqrt(H) damping, so the pinned module is not edited and the knob lives
   entirely in W_q (hence in the capsule).

2. Focus ruler. A pre-registered ``focus_met`` must have dynamic range: false on
   birth attention, false on r1-trained attention, true on oracle-aligned attention,
   and FALSE on sharp-but-misaligned attention. Without the last arm the ruler would
   only measure sharpness, which is not the claim.

3. Supervision signal. Backward-only gradient credit of the soft-target focus loss
   versus the task cross entropy, on the SAME train scenes with the SAME denominator,
   at c=1 and at the calibrated c. r1's located defect was that the task loss gave
   W_q only 1/228 to 1/11 of the credit it gave position_head, so the question this
   probe gates on is whether supervision actually delivers more query credit.
   A one-step finite difference of answer mass is also reported, but DESCRIPTIVELY:
   v2 gated on its sign and that was withdrawn (no dynamic range, and softmax
   normalization does not force the sign positive). The analytic first-order
   derivative is reported alongside so the negative reading is explained.

4. Dose feasibility. Adam normalizes the gradient, so the larger credit from
   supervision does not buy a larger step. A generous upper bound on the W_q norm
   growth reachable within the frozen lr/update count is compared against the
   required multiplier, anchored to r1's measured relative W_q change under the same
   optimizer settings. If the dose could reach the required scale on its own, the
   scale fix would not be necessary and the probe says so.

The supervision target is the queried entity's colour-mass distribution over
columns, read from the frozen encoding. It is FINER than the 4-way position label
and is disclosed as such; it is a training signal, not unit pre-labelling (SPEC 7).
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training import probe_u1r1_attention_collapse_20261116 as ac
from scripts.training import probe_u1r2_value_side_20261116 as vs
from scripts.training import run_u1_unit_read_20261116 as u1
from scripts.training import run_u1r1_position_20261116 as r1
from taiji.u2_history import value_digest
from taiji.u2_neural_units import UnitGraphSpec, apply_capsule, load_capsule
from taiji.u2_unit_input import _MASS_FLOOR

REPORT = ROOT / "reports/taiji_u2_u1r2_scale_focus_20261116.json"
T = r1.T
UNIFORM = math.log(T)
CAP_CPU, CAP_WALL, RESERVE = 90.0, 180.0, 10.0

#: Declared before any reading is taken.
TARGET_LOGIT_RANGE = 3.0
RANGE_BAND = (2.0, 6.0)
FOCUS_ANSWER_MASS_MIN = 0.5
FOCUS_EFFECTIVE_COLUMNS_MAX = 8.0
GRID = (1.0, 4.0, 16.0, 64.0, 256.0)
FD_RELATIVE_STEP = 0.01
FD_FIRST_ORDER_RTOL = 0.25
IDENTITY_RTOL = 1e-9
ANCHOR_ATOL = 1e-6
LOG_CLAMP = 1e-12

WEIGHT_STATES = ("birth", "r1_trained")
RULER_ARMS = ("oracle_soft", "oracle_onehot", "wrong_column")


def graph_at(seed: int, state: str, multiplier: float) -> Any:
    graph = r1.build_graph(seed)
    if state == "r1_trained":
        apply_capsule(graph, load_capsule(r1.final_capsule_path(seed)))
    if multiplier != 1.0:
        with torch.no_grad():
            graph.W_q.mul_(multiplier)
    return graph


def labelled(item: dict[str, Any]) -> bool:
    return float(vs.column_mass(item).sum()) >= _MASS_FLOOR


def answer_span(item: dict[str, Any]) -> torch.Tensor:
    return vs.column_mass(item) > 0.0


def capture_attention(
    graph: Any, item: dict[str, Any], forced: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (attention, logits) from the module's own forward.

    Logits are CAPTURED at the module's softmax call, never recomputed outside it.
    ``forced`` replaces the distribution for one forward (oracle arms) using the
    same single-branch patch as the value-side probe.
    """
    data = u1.cast_item(item, graph.spec.torch_dtype)
    real_softmax = torch.softmax
    captured: list[torch.Tensor] = []

    def patched(input: torch.Tensor, dim: int | None = None, **kwargs: Any) -> torch.Tensor:
        if dim == 0 and tuple(input.shape) == (T,):
            captured.append(input.detach().clone())
            if forced is not None:
                return forced.to(input.dtype)
        return real_softmax(input, dim=dim, **kwargs)  # type: ignore[arg-type]

    torch.softmax = patched  # type: ignore[assignment]
    try:
        with torch.no_grad():
            out = graph(data["channels"], data["gates"], graph.initial_state())
    finally:
        torch.softmax = real_softmax  # type: ignore[assignment]
    if len(captured) != 1:
        raise ValueError(f"softmax_capture_expected_one_call_got_{len(captured)}")
    return cast(torch.Tensor, out["attention"]).detach(), captured[0]


def focus_stats(
    graph: Any, episodes: list[dict[str, Any]], forced_arm: str | None = None
) -> dict[str, Any]:
    entropies: list[float] = []
    ranges: list[float] = []
    maxima: list[float] = []
    minima: list[float] = []
    masses: list[float] = []
    spans: list[int] = []
    absent_maxima: list[float] = []
    for item in episodes:
        forced = None if forced_arm is None else vs.forced_distribution(item, forced_arm)
        attention, logits = capture_attention(graph, item, forced)
        if forced is not None and not bool(torch.equal(attention, forced.to(attention.dtype))):
            raise ValueError("forced_attention_not_exact")
        entropy = float(-(attention * torch.log(attention.clamp_min(LOG_CLAMP))).sum())
        entropies.append(entropy)
        ranges.append(float(logits.max() - logits.min()))
        maxima.append(float(attention.max()))
        minima.append(float(attention.min()))
        if labelled(item):
            span = answer_span(item)
            spans.append(int(span.sum()))
            masses.append(float(attention[span].sum()))
        else:
            absent_maxima.append(float(attention.max()))
    count = len(episodes)
    n_labelled = len(masses)
    mean_entropy = sum(entropies) / count
    answer_mass = sum(masses) / n_labelled if n_labelled else 0.0
    effective = sum(math.exp(e) for e in entropies) / count
    return {
        "denominator": count,
        "labelled_denominator": n_labelled,
        "absent_denominator": len(absent_maxima),
        "forced": forced_arm is not None,
        "entropy_mean": round(mean_entropy, 6),
        "entropy_fraction_of_uniform_ceiling": round(mean_entropy / UNIFORM, 6),
        "effective_columns_mean": round(effective, 6),
        "logit_range_mean": round(sum(ranges) / count, 6),
        "logit_range_mean_unrounded": sum(ranges) / count,
        "logit_range_min": round(min(ranges), 6),
        "logit_range_max": round(max(ranges), 6),
        "max_attention_mean": round(sum(maxima) / count, 6),
        "attention_min_over_all": min(minima),
        "answer_mass_mean": round(answer_mass, 6),
        "answer_span_size_mean": round(sum(spans) / n_labelled, 6) if n_labelled else None,
        "absent_max_attention_mean": (
            round(sum(absent_maxima) / len(absent_maxima), 6) if absent_maxima else None
        ),
        "focus_met": bool(
            n_labelled > 0
            and answer_mass >= FOCUS_ANSWER_MASS_MIN
            and effective <= FOCUS_EFFECTIVE_COLUMNS_MAX
        ),
    }


def credit_probe(graph: Any, items: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    """Mean loss over ``items`` and the gradient credit it delivers; backward only.

    ``kind="focus"`` is the soft-target attention cross entropy; ``kind="task"`` is
    the same 4-way cross entropy the runner trains on. Both are averaged by the same
    denominator so the credit comparison is apples to apples.
    """
    graph.zero_grad(set_to_none=True)
    total = 0.0
    smallest = 1.0
    for item in items:
        data = u1.cast_item(item, graph.spec.torch_dtype)
        out = graph(data["channels"], data["gates"], graph.initial_state())
        attention = out["attention"]
        smallest = min(smallest, float(attention.detach().min()))
        if kind == "focus":
            target = vs.forced_distribution(item, "oracle_soft").to(attention.dtype)
            loss = -(target * torch.log(attention.clamp_min(LOG_CLAMP))).sum() / len(items)
        else:
            label = torch.tensor(data["label"], dtype=torch.long)
            loss = torch.nn.functional.cross_entropy(
                out["readout"].unsqueeze(0), label.unsqueeze(0)
            ) / len(items)
        loss.backward()
        total += float(loss.detach())
    credit = {
        name: float(param.grad.detach().abs().sum())
        for name, param in graph.named_parameters()
        if param.grad is not None
    }
    return {
        "kind": kind,
        "loss": round(total, 9),
        "denominator": len(items),
        "attention_min_during_loss": smallest,
        "grad_abs_sum": credit,
        "parameters_with_gradient": len(credit),
    }


def finite_difference(graph: Any, episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """One gradient-sized relative step on W_q, forward-only, then exact restore.

    Also reports the analytic first-order prediction of the answer-mass change, so a
    negative reading can be attributed to the objective's geometry instead of being
    mistaken for a broken measurement. For softmax attention a and span indicator s,
    d(answer_mass)/d(logit_j) = a_j * (s_j - answer_mass): flattening a peak that sits
    INSIDE the span leaks mass outside it, so the sign is not forced positive.
    """
    targets = [item for item in episodes if labelled(item)]
    original = value_digest({k: v.detach().clone() for k, v in graph.state_dict().items()})
    before_pairs = [capture_attention(graph, item) for item in targets]
    before = focus_stats(graph, episodes)["answer_mass_mean"]
    signal = credit_probe(graph, targets, "focus")
    weight = graph.W_q.detach().clone()
    grad = graph.W_q.grad
    if grad is None:
        raise ValueError("focus_loss_produced_no_w_q_gradient")
    grad_norm = float(grad.detach().norm())
    weight_norm = float(weight.norm())
    if grad_norm <= 0.0 or weight_norm <= 0.0:
        raise ValueError("degenerate_finite_difference_direction")
    step = FD_RELATIVE_STEP * weight_norm
    with torch.no_grad():
        graph.W_q.sub_(grad.detach() * (step / grad_norm))
    after_pairs = [capture_attention(graph, item) for item in targets]
    after = focus_stats(graph, episodes)["answer_mass_mean"]
    predicted = 0.0
    for item, pair0, pair1 in zip(targets, before_pairs, after_pairs, strict=True):
        attention0, logits0 = pair0
        _, logits1 = pair1
        span = answer_span(item).to(attention0.dtype)
        mass0 = float(attention0[span > 0].sum())
        derivative = attention0 * (span - mass0)
        predicted += float((derivative * (logits1 - logits0)).sum())
    predicted /= len(targets)
    with torch.no_grad():
        graph.W_q.copy_(weight)
    graph.zero_grad(set_to_none=True)
    restored = value_digest({k: v.detach().clone() for k, v in graph.state_dict().items()})
    measured = after - before
    return {
        "answer_mass_before": before,
        "answer_mass_after": after,
        "answer_mass_delta": round(measured, 9),
        "first_order_predicted_delta": round(predicted, 9),
        "first_order_agrees_in_sign": (predicted > 0.0) == (measured > 0.0),
        "first_order_relative_error": (abs(predicted - measured) / max(abs(measured), 1e-12)),
        "restored_exactly": original == restored,
        "relative_step": FD_RELATIVE_STEP,
        "w_q_norm": round(weight_norm, 9),
        "w_q_grad_norm": round(grad_norm, 12),
        "focus_loss": signal["loss"],
        "grad_abs_sum_w_q": signal["grad_abs_sum"].get("W_q"),
        "grad_abs_sum_units_p": signal["grad_abs_sum"].get("units.p"),
        "grad_abs_sum_position_head": signal["grad_abs_sum"].get("position_head.weight"),
        "attention_min_during_loss": signal["attention_min_during_loss"],
    }


def dose_bound(
    contract: dict[str, Any], seeds: list[int], per_seed: dict[int, float]
) -> dict[str, Any]:
    """Can the frozen dose grow W_q enough on its own? Upper bound, no training.

    Adam normalizes the gradient, so a larger credit does NOT buy a larger step: the
    per-coordinate displacement is bounded by lr*updates regardless of gradient scale.
    The L2 bound below assumes every coordinate moves the full distance coherently,
    so it is generous; the shortfall it implies is a LOWER bound on the real gap.
    The r1 column is the empirical anchor: the same optimizer, lr and update count
    already ran, and its measured relative W_q change is read from that report.
    """
    learning_rate = float(contract["criteria"]["learning_rate"])
    updates = int(contract["criteria"]["updates_per_seed"])
    spec = UnitGraphSpec()
    numel = spec.input_dim * spec.hidden_dim
    per_coordinate = learning_rate * updates
    l2_displacement = per_coordinate * math.sqrt(numel)
    collapse = json.loads(ac.REPORT.read_text(encoding="utf-8"))
    published = {int(row["seed"]): row for row in collapse["seeds"]}
    rows = []
    for seed in seeds:
        birth_norm = float(graph_at(seed, "birth", 1.0).W_q.detach().norm())
        row = published[seed]
        r1_relative = abs(row["w_q_frobenius_trained"] / row["w_q_frobenius_init"] - 1.0)
        reachable = (birth_norm + l2_displacement) / birth_norm
        rows.append(
            {
                "seed": seed,
                "w_q_numel": numel,
                "birth_w_q_norm": round(birth_norm, 9),
                "reachable_multiplier_upper_bound": round(reachable, 6),
                "required_multiplier": per_seed[seed],
                "shortfall_factor_lower_bound": round(per_seed[seed] / reachable, 3),
                "r1_measured_relative_w_q_change": round(r1_relative, 9),
                "r1_measured_multiplier": round(1.0 + r1_relative, 9),
            }
        )
    return {
        "learning_rate": learning_rate,
        "updates_per_seed": updates,
        "per_coordinate_displacement_max": per_coordinate,
        "l2_displacement_max": round(l2_displacement, 9),
        "adam_scale_invariance": (
            "step size is set by lr, not by gradient magnitude, so the focus loss's larger "
            "W_q credit does not enlarge the reachable W_q norm"
        ),
        "bound_is_generous": True,
        "r1_anchor_report": "reports/taiji_u2_u1r1_attention_collapse_20261116.json",
        "r1_anchor_criteria_matches": collapse["criteria_sha256"] == contract["criteria_sha256"],
        "rows": rows,
        "scale_fix_required_by_dose": all(
            row["required_multiplier"] > row["reachable_multiplier_upper_bound"] for row in rows
        ),
    }


def dataset_facts(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    spans = [int(answer_span(item).sum()) for item in episodes]
    marks = [labelled(item) for item in episodes]
    return {
        "train_total": len(episodes),
        "train_labelled": sum(marks),
        "train_absent": len(episodes) - sum(marks),
        "labelled_with_empty_span": sum(
            1
            for item, mark in zip(episodes, marks, strict=True)
            if mark and not answer_span(item).any()
        ),
        "answer_span_histogram": {str(k): spans.count(k) for k in sorted(set(spans))},
        "uniform_answer_mass_reference": round(
            sum(s for s, m in zip(spans, marks, strict=True) if m) / sum(marks) / T, 6
        ),
    }


def classify(payload: dict[str, Any], complete: bool) -> str:
    if not complete:
        return "not_established"
    if not all(payload["instrument_checks"].values()):
        return "instrument_invalid"
    if not payload["scale"]["per_seed_all_in_band"]:
        return "scale_out_of_band"
    if not payload["dose_feasibility"]["scale_fix_required_by_dose"]:
        return "scale_fix_not_required_by_dose"
    at_chosen = payload["credit"]["at_chosen"]
    if not all(row["focus_credit_above_task"] for row in at_chosen):
        return "supervision_credit_not_above_task"
    return "scale_calibrated_supervision_stronger"


def main() -> int:
    torch.set_num_threads(1)
    started_cpu, started_wall = time.process_time(), time.monotonic()
    contract = json.loads(r1.CONTRACT.read_text(encoding="utf-8"))
    r1.guards(contract, "probe")
    data = u1.load_data(contract)
    episodes = u1.prepare_episodes(data, UnitGraphSpec().n_units)
    train = episodes["train"]
    seeds = [int(s) for s in contract["criteria"]["seeds"]]
    hidden = UnitGraphSpec().hidden_dim
    sqrt_hidden = math.sqrt(hidden)
    facts = dataset_facts(train)

    rows: list[dict[str, Any]] = []
    grid_rows: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []
    ruler_rows: list[dict[str, Any]] = []
    signal_one: list[dict[str, Any]] = []
    signal_chosen: list[dict[str, Any]] = []
    credit_one: list[dict[str, Any]] = []
    credit_chosen: list[dict[str, Any]] = []
    targets = [item for item in train if labelled(item)]
    anchor_rows: list[dict[str, Any]] = []
    digest_rows: list[dict[str, Any]] = []
    budget_stopped = False

    base_ranges: dict[int, float] = {}
    for seed in seeds:
        graph = graph_at(seed, "birth", 1.0)
        stats = focus_stats(graph, train)
        base_ranges[seed] = stats["logit_range_mean_unrounded"]
        rows.append({"seed": seed, "weight_state": "birth", "multiplier": 1.0, **stats})
        anchor = ac.attention_stats(graph, train)
        anchor_rows.append(
            {
                "seed": seed,
                "mine": {
                    "entropy_mean": stats["entropy_mean"],
                    "effective_columns_mean": stats["effective_columns_mean"],
                    "logit_range_mean": stats["logit_range_mean"],
                },
                "published": {
                    "entropy_mean": anchor["entropy_mean"],
                    "effective_columns_mean": anchor["effective_columns_mean"],
                    "logit_range_mean": anchor["logit_range_mean"],
                },
                "entropy_mean_matches_published_instrument": abs(
                    stats["entropy_mean"] - anchor["entropy_mean"]
                )
                <= ANCHOR_ATOL,
                "effective_columns_matches_published_instrument": abs(
                    stats["effective_columns_mean"] - anchor["effective_columns_mean"]
                )
                <= ANCHOR_ATOL,
                "logit_range_matches_published_instrument": abs(
                    stats["logit_range_mean"] - anchor["logit_range_mean"]
                )
                <= ANCHOR_ATOL,
            }
        )
    mean_base = sum(base_ranges.values()) / len(seeds)
    spread_ratio = max(base_ranges.values()) / min(base_ranges.values())
    per_seed = {seed: round(TARGET_LOGIT_RANGE / base_ranges[seed], 3) for seed in seeds}
    global_candidate = round(TARGET_LOGIT_RANGE / mean_base, 3) if mean_base > 0.0 else None
    global_ranges = (
        {seed: round(global_candidate * base_ranges[seed], 6) for seed in seeds}
        if global_candidate is not None
        else {}
    )
    global_in_band = bool(global_ranges) and all(
        RANGE_BAND[0] <= value <= RANGE_BAND[1] for value in global_ranges.values()
    )
    dose = dose_bound(contract, seeds, per_seed)

    for seed in seeds:
        for multiplier in GRID:
            if time.process_time() - started_cpu >= CAP_CPU - RESERVE:
                budget_stopped = True
                break
            graph = graph_at(seed, "birth", multiplier)
            stats = focus_stats(graph, train)
            grid_rows.append({"seed": seed, "multiplier": multiplier, **stats})
            identity_rows.append(
                {
                    "seed": seed,
                    "multiplier": multiplier,
                    "predicted_range": multiplier * base_ranges[seed],
                    "measured_range": stats["logit_range_mean_unrounded"],
                    "relative_error": (
                        abs(stats["logit_range_mean_unrounded"] - multiplier * base_ranges[seed])
                        / max(multiplier * base_ranges[seed], 1e-12)
                    ),
                }
            )
        if budget_stopped:
            break

    chosen_rows: dict[int, dict[str, Any]] = {}
    if not budget_stopped:
        for seed in seeds:
            stats = focus_stats(graph_at(seed, "birth", per_seed[seed]), train)
            chosen_rows[seed] = stats
            rows.append(
                {"seed": seed, "weight_state": "birth", "multiplier": per_seed[seed], **stats}
            )
    chosen_in_band = bool(chosen_rows) and all(
        RANGE_BAND[0] <= row["logit_range_mean"] <= RANGE_BAND[1] for row in chosen_rows.values()
    )

    for seed in seeds:
        if budget_stopped or time.process_time() - started_cpu >= CAP_CPU - RESERVE:
            budget_stopped = True
            break
        for state in WEIGHT_STATES:
            graph = graph_at(seed, state, 1.0)
            before = value_digest({k: v.detach().clone() for k, v in graph.state_dict().items()})
            stats = focus_stats(graph, train)
            ruler_rows.append({"seed": seed, "arm": f"{state}_c1", "weight_state": state, **stats})
            for arm in RULER_ARMS:
                ruler_rows.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "weight_state": state,
                        **focus_stats(graph, train, arm),
                    }
                )
            after = value_digest({k: v.detach().clone() for k, v in graph.state_dict().items()})
            digest_rows.append({"seed": seed, "weight_state": state, "unchanged": before == after})
        if chosen_in_band:
            graph = graph_at(seed, "birth", per_seed[seed])
            ruler_rows.append(
                {
                    "seed": seed,
                    "arm": "birth_chosen",
                    "weight_state": "birth",
                    "multiplier": per_seed[seed],
                    **focus_stats(graph, train),
                }
            )
            signal_one.append(
                {"seed": seed, **finite_difference(graph_at(seed, "birth", 1.0), train)}
            )
            signal_chosen.append(
                {"seed": seed, "multiplier": per_seed[seed], **finite_difference(graph, train)}
            )
            for tag, multiplier, sink in (
                ("at_multiplier_one", 1.0, credit_one),
                ("at_chosen", per_seed[seed], credit_chosen),
            ):
                probe_graph = graph_at(seed, "birth", multiplier)
                focus = credit_probe(probe_graph, targets, "focus")
                task_same = credit_probe(probe_graph, targets, "task")
                task_all = credit_probe(probe_graph, train, "task")
                focus_w_q = focus["grad_abs_sum"].get("W_q", 0.0)
                task_w_q = task_same["grad_abs_sum"].get("W_q", 0.0)
                sink.append(
                    {
                        "seed": seed,
                        "arm": tag,
                        "multiplier": multiplier,
                        "focus": focus,
                        "task_same_denominator": task_same,
                        "task_all_train": task_all,
                        "focus_w_q_credit": focus_w_q,
                        "task_w_q_credit_same_denominator": task_w_q,
                        "focus_credit_above_task": focus_w_q > task_w_q,
                        "focus_credit_ratio": (
                            round(focus_w_q / task_w_q, 6) if task_w_q > 0.0 else None
                        ),
                        "focus_gives_no_position_head_credit": (
                            focus["grad_abs_sum"].get("position_head.weight") is None
                        ),
                    }
                )

    def arm_rows(name: str) -> list[dict[str, Any]]:
        return [row for row in ruler_rows if row["arm"] == name]

    ruler_range = {
        "false_on_birth_c1": all(not row["focus_met"] for row in arm_rows("birth_c1")),
        "false_on_r1_trained_c1": all(not row["focus_met"] for row in arm_rows("r1_trained_c1")),
        "true_on_oracle_soft": all(row["focus_met"] for row in arm_rows("oracle_soft")),
        "true_on_oracle_onehot": all(row["focus_met"] for row in arm_rows("oracle_onehot")),
        "false_on_wrong_column": all(not row["focus_met"] for row in arm_rows("wrong_column")),
    }
    if chosen_in_band:
        ruler_range["false_on_birth_chosen"] = all(
            not row["focus_met"] for row in arm_rows("birth_chosen")
        )

    complete = (
        (not budget_stopped)
        and len(rows) > 0
        and len(signal_chosen) == len(seeds)
        and len(credit_chosen) == len(seeds)
    )
    instrument = {
        "train_face_only": (
            facts["train_total"] == len(data["train"]) and facts["train_total"] == len(train)
        ),
        "labelled_span_never_empty": facts["labelled_with_empty_span"] == 0,
        "labelled_denominator_positive": facts["train_labelled"] > 0,
        "gates_inherited_from_frozen_r1": (
            float(contract["criteria"]["fit_gate"]) == vs.FIT_GATE
            and float(contract["criteria"]["train_gate"]) == vs.TRAIN_GATE
            and tuple(seeds) == tuple(contract["criteria"]["seeds"])
        ),
        "published_instrument_anchor_matches": all(
            row[key]
            for row in anchor_rows
            for key in (
                "entropy_mean_matches_published_instrument",
                "effective_columns_matches_published_instrument",
                "logit_range_matches_published_instrument",
            )
        ),
        "logit_range_linear_in_multiplier": all(
            row["relative_error"] <= IDENTITY_RTOL for row in identity_rows
        ),
        "no_attention_underflow": all(
            row["attention_min_over_all"] > 0.0
            for row in grid_rows + rows + ruler_rows
            if not row["forced"]
        ),
        "forced_arms_are_exact_zeros_by_construction": all(
            row["attention_min_over_all"] == 0.0
            for row in ruler_rows
            if row["arm"] in ("oracle_onehot", "wrong_column")
        ),
        "focus_loss_attention_above_clamp": all(
            row["attention_min_during_loss"] > LOG_CLAMP for row in signal_one + signal_chosen
        ),
        "ruler_has_dynamic_range": all(ruler_range.values()) and len(ruler_range) >= 5,
        "measurements_did_not_train": all(row["unchanged"] for row in digest_rows),
        "finite_difference_restored_exactly": all(
            row["restored_exactly"] for row in signal_one + signal_chosen
        ),
        "finite_difference_matches_first_order": all(
            row["first_order_agrees_in_sign"]
            and row["first_order_relative_error"] <= FD_FIRST_ORDER_RTOL
            for row in signal_one + signal_chosen
        ),
        "focus_loss_never_credits_position_head": all(
            row["focus_gives_no_position_head_credit"] for row in credit_one + credit_chosen
        ),
        "credit_denominators_match": all(
            row["focus"]["denominator"]
            == row["task_same_denominator"]["denominator"]
            == len(targets)
            for row in credit_one + credit_chosen
        ),
        "dose_anchor_criteria_matches": dose["r1_anchor_criteria_matches"],
    }
    payload: dict[str, Any] = {
        "schema": "taiji-u2-u1r2-scale-focus-calibration-v4",
        "card": "22.5.U.1r2",
        "phase": "design_calibration",
        "anchored_contract": "plans/manifests/taiji_u2_u1r1_position_20261116.json",
        "criteria_sha256": contract["criteria_sha256"],
        "columns_T": T,
        "hidden_dim": hidden,
        "sqrt_hidden_dim": sqrt_hidden,
        "uniform_entropy_ceiling": round(UNIFORM, 6),
        "declared": {
            "target_logit_range": TARGET_LOGIT_RANGE,
            "range_band": list(RANGE_BAND),
            "focus_answer_mass_min": FOCUS_ANSWER_MASS_MIN,
            "focus_effective_columns_max": FOCUS_EFFECTIVE_COLUMNS_MAX,
            "grid": list(GRID),
            "finite_difference_relative_step": FD_RELATIVE_STEP,
            "finite_difference_first_order_rtol": FD_FIRST_ORDER_RTOL,
            "identity_rtol": IDENTITY_RTOL,
            "anchor_atol": ANCHOR_ATOL,
        },
        "dataset_facts": facts,
        "scale": {
            "base_logit_range_by_seed": base_ranges,
            "mean_base_logit_range": mean_base,
            "birth_seed_spread_ratio": round(spread_ratio, 6),
            "band_width_ratio": RANGE_BAND[1] / RANGE_BAND[0],
            "calibration_scope": "per_seed_train_only",
            "per_seed_multiplier": per_seed,
            "residual_after_removing_damping": {
                seed: round(per_seed[seed] / sqrt_hidden, 6) for seed in seeds
            },
            "per_seed_measured_range": {
                seed: chosen_rows[seed]["logit_range_mean"] for seed in chosen_rows
            },
            "per_seed_all_in_band": chosen_in_band,
            "global_alternative": {
                "multiplier": global_candidate,
                "per_seed_predicted_range": global_ranges,
                "all_in_band": global_in_band,
                "rejected_because": (
                    "birth seed spread exceeds the declared band width, so one global "
                    "multiplier cannot put every seed in band"
                    if not global_in_band
                    else None
                ),
            },
            "grid": grid_rows,
            "identity_checks": identity_rows,
            "birth_rows": rows,
        },
        "focus_ruler": {"arms": ruler_rows, "dynamic_range": ruler_range},
        "signal": {"at_multiplier_one": signal_one, "at_chosen": signal_chosen},
        "credit": {"at_multiplier_one": credit_one, "at_chosen": credit_chosen},
        "dose_feasibility": dose,
        "withdrawn_proxy": {
            "name": "one-step answer_mass delta sign",
            "used_in_schema_version": 2,
            "v2_verdict_it_produced": "focus_not_movable",
            "status": "withdrawn as a verdict criterion; readings kept as descriptive",
            "why_invalid": (
                "a single 1%-relative gradient step moves answer mass by ~1e-3 against a "
                "0.125 -> 1.0 gap that 384 updates must cover, so the ruler has no range to "
                "decide movability; and the sign is not forced positive because for softmax "
                "attention d(answer_mass)/d(logit_j) = a_j*(span_j - answer_mass), so "
                "flattening an in-span peak leaks mass outside the span"
            ),
            "evidence": (
                "the analytic first-order prediction agrees with the measured delta in sign "
                "and magnitude on every row (see instrument_checks."
                "finite_difference_matches_first_order), so the negative reading is the "
                "objective's geometry, not a broken measurement"
            ),
            "replaced_by": "same-face, same-denominator W_q gradient credit: focus loss vs task CE",
        },
        "supervision_disclosure": {
            "target": "queried-entity colour mass per column, normalised (oracle_soft)",
            "source": "frozen encoding unit-input-column-color-v1, train scene's own rgb channel",
            "finer_than_task_label": True,
            "task_label_width": 4,
            "target_width": T,
            "applied_to": "labelled train scenes only",
            "labelled_train_scenes": facts["train_labelled"],
            "train_scenes_without_focus_supervision": facts["train_absent"],
            "fit_or_check_used_as_supervision": False,
            "unit_prelabeling": "none: loss is a training signal, not an architectural label",
        },
        "revision_history": {
            "version": 4,
            "supersedes": (
                "taiji-u2-u1r2-scale-focus-calibration-v1, -v2 and -v3 "
                "(all untracked, self-produced)"
            ),
            "v1_verdict": "not_established",
            "v2_verdict": "focus_not_movable",
            "v3_verdict": "scale_calibrated_supervision_stronger",
            "v4_addition": (
                "dose-feasibility gate: a generous upper bound on the W_q norm growth the "
                "frozen lr/update count can reach, anchored to r1's measured relative change; "
                "this is an ADDITIONAL necessary condition, not a relaxation of any threshold"
            ),
            "declared_criteria_unchanged": [
                "target_logit_range",
                "range_band",
                "focus_answer_mass_min",
                "focus_effective_columns_max",
            ],
            "estimator_changed": "global multiplier -> per-seed multiplier, both train-only",
            "reason": (
                "v1 measured a birth logit-range spread across seeds wider than the declared "
                "band, so no single global multiplier can satisfy the band for every seed; "
                "the band and target were NOT relaxed"
            ),
            "instrument_defects_fixed": [
                "v1: linearity test used the 6-decimal rounded base range against a 1e-9 "
                "tolerance (rounding alone gave ~1e-5 relative error); now uses unrounded values",
                "v1: underflow test included forced one-hot arms whose zeros are by "
                "construction; now restricted to softmax-produced attention, with a separate "
                "check that the forced arms really are exact one-hot",
                "v2: the verdict gated on the sign of a one-step answer_mass delta, a proxy "
                "with no dynamic range and a sign the objective does not force; withdrawn and "
                "replaced by a same-denominator gradient-credit comparison, with the analytic "
                "first-order derivative added so the negative reading is explained rather "
                "than published as a capability claim",
                "v2: float() on a requires_grad attention tensor raised a UserWarning; now "
                "detached before conversion",
            ],
        },
        "instrument_checks": instrument,
        "anchors": anchor_rows,
        "digest_checks": digest_rows,
        "capability_scope": "design calibration for U.1r2; no capability claim, no training",
        "discipline": {
            "optimizer_steps": 0,
            "training_updates": 0,
            "data_file_reads": 1,
            "partitions_encoded": ["train", "fit"],
            "partitions_measured": ["train"],
            "fit_scenes_measured": 0,
            "backward_passes": (len(signal_one) + len(signal_chosen)) * len(targets)
            + sum(
                row["focus"]["denominator"]
                + row["task_same_denominator"]["denominator"]
                + row["task_all_train"]["denominator"]
                for row in credit_one + credit_chosen
            ),
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
    payload["verdict"] = classify(payload, complete)
    r1.write_report(REPORT, payload)
    print(
        "SCALE_FOCUS verdict="
        + payload["verdict"]
        + f" per_seed={per_seed} in_band={chosen_in_band}"
        + f" instrument={sum(instrument.values())}/{len(instrument)}"
        + f" cpu={payload['cost']['cpu']}"
    )
    return 0 if payload["verdict"] != "instrument_invalid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
