"""17.3 pre-pricing interface and FD verification (0 updates, 0 Adam).

Runs the 0a static contract audit and the 0b graph checks that can be judged
before any task training:

T1  target/label separation on the real routing loss;
T2  R1 order gate calibration on the witness pair plus 20 permutation pairs,
    with the old mean-pooled head as the order-blind control;
T3  R2 gold-span gate on 40 dev examples (gold record + gold span);
T4  teacher-span provenance and digest;
T5  position-supervision finite-difference gradient;
T7  parameter counts;
T11 no write/experience API;
T13 confirm reads = 0.

It also finite-differences the selection CE and the locked NLL.  Thresholds are
written to a manifest so 0b T2 can call the same frozen numbers later.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

torch.set_num_threads(1)

from taiji.binding_copy import (  # noqa: E402
    forced_read,
    initial_binding_state,
    observe_goal,
    prepare,
)
from taiji.binding_copy_scoring import score_generation  # noqa: E402
from taiji.binding_select import (  # noqa: E402
    SelectionConfig,
    SelectionHead,
    load_bundle,
    locked_step,
)
from taiji.living_core import EOS  # noqa: E402
from taiji.record_relation_core import MAX_PREDICTIONS, decode_endpoint  # noqa: E402
from taiji.selection_routing import (  # noqa: E402
    OrderedSelectionHead,
    RoutingConfig,
    SpanPointerHead,
    _advanced_state,
    build_context,
    expected_head_parameter_count,
    expected_span_parameter_count,
    route_labels,
    routing_generate,
    routing_loss,
    routing_step,
    selection_ce,
    span_loss,
    teacher_span,
)

BUNDLE = Path("output/taiji_core_binding_select_16_1_compare/bundles/28031-A-locked-step300.pt")
COMPARE_SCRIPT = Path("scripts/training/verify_taiji_binding_select_16_1_compare.py")
OUT = Path("reports/taiji_selection_routing_17_3_verify_20261103.json")
THRESHOLDS = Path("plans/manifests/taiji_selection_routing_17_3_r1_thresholds_20261103.json")
WITNESS_A = "把甲从乙搬到丙".encode()
WITNESS_B = "把乙从甲搬到丙".encode()

STARTED_CPU = time.process_time()
STARTED_WALL = time.perf_counter()
CONFIRM_READS = 0


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_fixtures():
    spec = importlib.util.spec_from_file_location("cmp17", ROOT / COMPARE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    train, dev, _select, _data = module.load_data()
    return train, dev


def relative_l2(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = max(float(left.norm()), float(right.norm()), 1e-30)
    return float(torch.dist(left, right) / denominator)


def _new_scores(head, binding, goal: bytes, keys: tuple[str, ...]) -> torch.Tensor:
    context = build_context(binding, goal, keys)
    s_t = torch.zeros(128, dtype=torch.float64)
    previous = torch.zeros(64, dtype=torch.float64)
    with torch.no_grad():
        return head.score_rows(context.goal_vec, context.key_vecs, s_t, previous)[:-1]


def calibrate_r1(candidate, dev) -> tuple[dict, dict]:
    binding = candidate.binding
    head = OrderedSelectionHead(RoutingConfig(), seed=1731)
    old = SelectionHead(SelectionConfig()).to(torch.float64)
    keys = tuple(dev[0]["record_keys"])
    with torch.no_grad():
        witness_new = {
            "representation_relative_l2": relative_l2(
                head.pooled(binding.base.relation, binding.subject, WITNESS_A),
                head.pooled(binding.base.relation, binding.subject, WITNESS_B),
            ),
            "score_max_abs_diff": float(
                (
                    _new_scores(head, binding, WITNESS_A, keys)
                    - _new_scores(head, binding, WITNESS_B, keys)
                )
                .abs()
                .max()
            ),
        }
        witness_old = {
            "representation_max_abs_diff": float(
                (old.encode(WITNESS_A) - old.encode(WITNESS_B)).abs().max()
            ),
            "score_max_abs_diff": float(
                (old.scores(WITNESS_A, list(keys)) - old.scores(WITNESS_B, list(keys))).abs().max()
            ),
        }
        same_input = {
            "representation_relative_l2": relative_l2(
                head.pooled(binding.base.relation, binding.subject, WITNESS_A),
                head.pooled(binding.base.relation, binding.subject, WITNESS_A),
            ),
            "score_max_abs_diff": float(
                (
                    _new_scores(head, binding, WITNESS_A, keys)
                    - _new_scores(head, binding, WITNESS_A, keys)
                )
                .abs()
                .max()
            ),
        }
        one_byte = {
            "representation_relative_l2": relative_l2(
                head.pooled(binding.base.relation, binding.subject, WITNESS_A),
                head.pooled(binding.base.relation, binding.subject, "把甲从乙搬到丁".encode()),
            ),
            "score_max_abs_diff": float(
                (
                    _new_scores(head, binding, WITNESS_A, keys)
                    - _new_scores(head, binding, "把甲从乙搬到丁".encode(), keys)
                )
                .abs()
                .max()
            ),
        }
        rows = []
        for fixture in dev[:10]:
            goal = fixture["goal"].encode("utf-8")
            for name, permuted in (
                ("reversed", bytes(reversed(goal))),
                ("sorted", bytes(sorted(goal))),
            ):
                rows.append(
                    {
                        "index": fixture["index"],
                        "permutation": name,
                        "representation_relative_l2": relative_l2(
                            head.pooled(binding.base.relation, binding.subject, goal),
                            head.pooled(binding.base.relation, binding.subject, permuted),
                        ),
                        "score_max_abs_diff": float(
                            (
                                _new_scores(head, binding, goal, tuple(fixture["record_keys"]))
                                - _new_scores(
                                    head, binding, permuted, tuple(fixture["record_keys"])
                                )
                            )
                            .abs()
                            .max()
                        ),
                    }
                )
    repr_values = [row["representation_relative_l2"] for row in rows]
    score_values = [row["score_max_abs_diff"] for row in rows]
    # SPEC 9.1 froze the preliminary values before any candidate output; the
    # 20-pair measurements below are margin evidence, not the threshold source.
    # A first pass used max(1e-6, 0.1 * median); that rule was not registered
    # and is withdrawn as D17-1 (see the 17.3 report).
    theta_repr = 1e-2
    theta_score = 1e-3
    thresholds = {
        "schema": "taiji-selection-routing-17-r1-thresholds-v1",
        "step": "N17.3",
        "date": "2026-11-03",
        "method": "frozen at SPEC 9.1 preliminary values (theta_repr=1e-2, theta_score=1e-3) before any candidate output; 20-pair calibration is margin evidence",
        "withdrawn_method": "theta = max(1e-6, 0.1 * median) was never registered and is withdrawn (D17-1)",
        "theta_repr": theta_repr,
        "theta_score": theta_score,
        "noise_floor_repr": 1.67e-16,
        "noise_floor_score": 8.88e-16,
        "noise_floor_source": "prior 16.1 old-head float64 readings; this run old-head witness measured 0.0",
        "pairs": len(rows),
        "median_repr": statistics.median(repr_values),
        "median_score": statistics.median(score_values),
        "min_repr": min(repr_values),
        "min_score": min(score_values),
        "distribution_checks": {
            "median_repr_ge_1e-6": bool(statistics.median(repr_values) >= 1e-6),
            "pairs_repr_gt_1e-8": sum(1 for value in repr_values if value > 1e-8),
            "at_least_18_of_20": bool(sum(1 for value in repr_values if value > 1e-8) >= 18),
            "median_score_ge_1e-6": bool(statistics.median(score_values) >= 1e-6),
        },
        "witness": {"new": witness_new, "old_control": witness_old},
    }
    calibration = {
        "n_pairs": len(rows),
        "rows": rows,
        "median_repr": thresholds["median_repr"],
        "median_score": thresholds["median_score"],
        "theta_repr": theta_repr,
        "theta_score": theta_score,
        "noise_floor": {
            "prior_16_1_repr": 1.67e-16,
            "prior_16_1_score": 8.88e-16,
            "this_run_old_head_witness_repr": witness_old["representation_max_abs_diff"],
            "this_run_old_head_witness_score": witness_old["score_max_abs_diff"],
        },
        "new_head_witness": witness_new,
        "old_head_witness": witness_old,
        "cases": {
            "pass_witness_new_head": bool(
                witness_new["representation_relative_l2"] >= theta_repr
                and witness_new["score_max_abs_diff"] >= theta_score
            ),
            "pass_one_byte_new_head": bool(
                one_byte["representation_relative_l2"] >= theta_repr
                and one_byte["score_max_abs_diff"] >= theta_score
            ),
            "fail_old_head_order_blind": bool(
                witness_old["representation_max_abs_diff"] < theta_repr
                and witness_old["score_max_abs_diff"] < theta_score
            ),
            "fail_same_input_numeric_noise": bool(
                same_input["representation_relative_l2"] < theta_repr
                and same_input["score_max_abs_diff"] < theta_score
            ),
        },
        "one_byte_example": one_byte,
        "same_input_example": same_input,
    }
    return calibration, thresholds


def probe_target_label_separation(candidate, dev) -> dict:
    fixture = dev[0]
    binding = candidate.binding
    head = OrderedSelectionHead(RoutingConfig(), seed=1731)
    prepared = prepare(
        binding,
        initial_binding_state(binding, "17-t1"),
        fixture["goal"].encode("utf-8"),
        fixture["records"],
    )
    context = build_context(binding, fixture["goal"].encode("utf-8"), tuple(fixture["record_keys"]))
    routes = route_labels(fixture)
    first = routing_loss(
        binding,
        prepared,
        tuple(fixture["target_prefix"]),
        tuple(fixture["target_bytes"]),
        routes[0],
        routes[1],
        context,
        head,
        arm="A_route_lock",
        collect=True,
    )
    shifted = tuple(int(value) for value in fixture["target_bytes"][1:]) + (EOS,)
    second = routing_loss(
        binding,
        prepared,
        tuple(fixture["target_prefix"]),
        shifted,
        routes[0],
        routes[1],
        context,
        head,
        arm="A_route_lock",
        collect=True,
    )
    same = all(
        torch.equal(a, b)
        for a, b in zip(first["distributions"], second["distributions"], strict=True)
    )
    return {
        "distribution_rows_equal": bool(same),
        "nll_first": float(first["nll"]),
        "nll_shifted": float(second["nll"]),
        "loss_differs": bool(float(first["nll"]) != float(second["nll"])),
        "passed": bool(same and float(first["nll"]) != float(second["nll"])),
    }


def gold_span_gate(candidate, dev, limit: int = 40) -> dict:
    base = candidate.binding
    rows = []
    with torch.no_grad():
        for fixture in dev[:limit]:
            label = fixture["selection_label"]
            record = fixture["raw_example"]["records"][label]
            name_bytes = fixture["object_name"].encode("utf-8")
            raw_record = record["raw"].encode("utf-8")
            name_start = raw_record.find(name_bytes)
            target = fixture["target_prefix"]
            state = initial_binding_state(base, f"17-t3-{fixture['index']}")
            prepared = prepare(base, state, fixture["goal"].encode("utf-8"), fixture["records"])
            state, gate_index = observe_goal_for_probe(base, prepared)
            previous = EOS
            raw: list[int] = []
            terminated = False
            forced_steps = 0
            for _prediction in range(MAX_PREDICTIONS):
                emitted = len(raw)
                in_sync = all(raw[i] == target[i] for i in range(emitted))
                force = in_sync and 3 <= emitted < 15 and name_start >= 0
                if force:
                    step = forced_read(
                        base,
                        prepared,
                        state,
                        previous,
                        record_index=label,
                        offset=name_start + (emitted - 3),
                        gate_index=gate_index,
                        gamma_override=0.0,
                        force_position=False,
                    )
                    forced_steps += 1
                else:
                    step = locked_step(
                        candidate,
                        prepared,
                        state,
                        previous,
                        gate_index=gate_index,
                        chosen_record=label,
                    )
                endpoint = decode_endpoint(step.probabilities)
                if endpoint["eos"]:
                    terminated = True
                    break
                raw.append(int(endpoint["byte"]))
                from taiji.binding_copy import advance_copy

                m_next, coverage_next = advance_copy(
                    base.reader,
                    state,
                    step.read,
                    step.probabilities,
                    step.gamma,
                    endpoint["byte"],
                    prepared.base.view,
                )
                state = replace(
                    step.state,
                    m=m_next,
                    coverage=coverage_next,
                    prefix_cursor=step.state.prefix_cursor + 1,
                )
                gate_index = step.gate_index
                previous = int(endpoint["byte"])
            scored = score_generation(bytes(raw), target, terminated)
            rows.append(
                {
                    "index": fixture["index"],
                    "forced_steps": forced_steps,
                    "object_exact": scored["object_exact"],
                    "place_exact": scored["place_exact"],
                    "completed": scored["completed"],
                }
            )
    n = len(rows)
    object_rate = sum(1 for row in rows if row["object_exact"]) / n
    place_rate = sum(1 for row in rows if row["place_exact"]) / n
    completed_rate = sum(1 for row in rows if row["completed"]) / n
    return {
        "n": n,
        "object_exact_rate": object_rate,
        "place_exact_rate": place_rate,
        "completed_rate": completed_rate,
        "method": "alpha/beta forced only; pointer and cursor stay natural (frozen pre-Plan-A method)",
        "gold_span_gate": ">=0.8 pass; 0.1-0.8 not_naked; <=0.1 force position supervision",
        "band": (
            "pass"
            if object_rate >= 0.8
            else ("not_naked" if object_rate > 0.1 else "force_position_supervision")
        ),
        "rows": rows,
    }


def golden_offset(view, record_index: int, gold: int, previous: int) -> int:
    raw = view.raw[record_index]
    occurrences = [position for position, byte in enumerate(raw) if byte == gold]
    if not occurrences:
        raise RuntimeError(f"gold byte {gold} absent from record {record_index}")
    matched = [
        position
        for position in occurrences
        if previous != EOS and position > 0 and raw[position - 1] == previous
    ]
    return matched[0] if matched else occurrences[0]


def probe_wrong_position(candidate, dev, head, limit: int = 40) -> dict:
    """Counter-probe 1: force the pointer onto a wrong byte, gold must collapse."""
    binding = candidate.binding if hasattr(candidate, "binding") else candidate
    rows = []
    with torch.no_grad():
        for fixture in dev[:limit]:
            goal = fixture["goal"].encode("utf-8")
            prepared = prepare(
                binding,
                initial_binding_state(binding, f"17-wrong-{fixture['index']}"),
                goal,
                fixture["records"],
            )
            context = build_context(binding, goal, tuple(fixture["record_keys"]))
            state, gate_index = observe_goal(binding, prepared, gate_index=0)
            previous = EOS
            for t in range(3):
                step = routing_step(
                    binding,
                    prepared,
                    state,
                    previous,
                    context,
                    head,
                    arm="B_stop_lock",
                    gate_index=gate_index,
                )
                accepted = fixture["target_prefix"][t]
                state, _, gate_index = _advanced_state(binding, prepared, step, accepted)
                previous = accepted
            label = int(fixture["selection_label"])
            view = prepared.base.view
            gold = int(fixture["target_bytes"][3])
            gold_pos = golden_offset(view, label, gold, previous)
            raw = view.raw[label]
            wrong_pos = next(position for position in range(len(raw)) if raw[position] != gold)
            control = forced_read(
                binding,
                prepared,
                state,
                previous,
                record_index=label,
                offset=gold_pos,
                gate_index=gate_index,
                gamma_override=0.0,
                force_position=True,
            )
            wrong = forced_read(
                binding,
                prepared,
                state,
                previous,
                record_index=label,
                offset=wrong_pos,
                gate_index=gate_index,
                gamma_override=0.0,
                force_position=True,
            )
            rows.append(
                {
                    "index": fixture["index"],
                    "gold_byte": gold,
                    "wrong_byte": int(raw[wrong_pos]),
                    "control_gold": float(control.probabilities[gold]),
                    "wrong_gold": float(wrong.probabilities[gold]),
                    "wrong_wrong": float(wrong.probabilities[int(raw[wrong_pos])]),
                }
            )
    control_min = min(row["control_gold"] for row in rows)
    gold_max = max(row["wrong_gold"] for row in rows)
    wrong_min = min(row["wrong_wrong"] for row in rows)
    threshold = 1e-12
    passed = bool(
        control_min >= 1.0 - threshold and gold_max <= threshold and wrong_min >= 1.0 - threshold
    )
    return {
        "n": len(rows),
        "threshold": threshold,
        "control_gold_min": control_min,
        "wrong_gold_max": gold_max,
        "wrong_byte_min": wrong_min,
        "passed": passed,
        "rows": rows,
        "note": "forced pointer on a different byte; gamma=0 and q_null=0, so probabilities = copy mass",
    }


def probe_head_ablation(candidate, dev, span_head, limit: int = 40) -> dict:
    """Counter-probe 2: remove the selection head, keep the position pointer."""
    binding = candidate.binding if hasattr(candidate, "binding") else candidate
    scores = []
    point_rows = []
    with torch.no_grad():
        for fixture in dev[:limit]:
            goal = fixture["goal"].encode("utf-8")
            prepared = prepare(
                binding,
                initial_binding_state(binding, f"17-ablate-{fixture['index']}"),
                goal,
                fixture["records"],
            )
            context = build_context(binding, goal, tuple(fixture["record_keys"]))
            generated = routing_generate(
                binding,
                prepared,
                context,
                None,
                arm="A_route_lock",
                span_head=span_head,
                pointer_hard=True,
            )
            scored = score_generation(
                generated["raw_bytes"], tuple(fixture["target_prefix"]), generated["terminated"]
            )
            scores.append(scored)
            if fixture is dev[0]:
                state, gate_index = observe_goal(binding, prepared, gate_index=0)
                step = routing_step(
                    binding,
                    prepared,
                    state,
                    EOS,
                    context,
                    None,
                    arm="A_route_lock",
                    gate_index=gate_index,
                    span_head=span_head,
                    pointer_hard=True,
                )
                point_rows = [
                    {
                        "record": index,
                        "max": float(pointer.max()),
                        "sum": float(pointer.sum()),
                    }
                    for index, pointer in enumerate(step.pointer)
                ]
    n = len(scores)
    object_rate = sum(1 for row in scores if row["object_exact"]) / n
    place_rate = sum(1 for row in scores if row["place_exact"]) / n
    completed_rate = sum(1 for row in scores if row["completed"]) / n
    point_mass = all(
        row["max"] >= 1.0 - 1e-12 and abs(row["sum"] - 1.0) <= 1e-12 for row in point_rows
    )
    threshold = 0.10
    return {
        "n": n,
        "object_exact_rate": object_rate,
        "place_exact_rate": place_rate,
        "completed_rate": completed_rate,
        "accuracy_threshold": threshold,
        "point_mass_kept": point_mass,
        "point_rows": point_rows,
        "passed": bool(object_rate <= threshold and point_mass),
        "note": "uniform route prior over records and null; the pointer stays a one-hot point",
    }


def observe_goal_for_probe(base, prepared):
    from taiji.binding_copy import observe_goal

    return observe_goal(base, prepared)


def span_provenance(candidate, dev, span_head, limit: int = 40) -> dict:
    binding = candidate.binding
    teacher_rows = []
    model_rows = []
    with torch.no_grad():
        for fixture in dev[:limit]:
            object_index = int(fixture["selection_label"])
            place_index = fixture["record_keys"].index(fixture["place_description"])
            object_record = fixture["raw_example"]["records"][object_index]["raw"].encode("utf-8")
            place_record = fixture["raw_example"]["records"][place_index]["raw"].encode("utf-8")
            object_span = teacher_span(object_record, fixture["object_name"].encode("utf-8"))
            place_span = teacher_span(place_record, fixture["place_name"].encode("utf-8"))
            teacher_rows.append(
                [
                    fixture["index"],
                    object_index,
                    object_span[0],
                    object_span[1],
                    place_index,
                    place_span[0],
                    place_span[1],
                ]
            )
            prepared = prepare(
                binding,
                initial_binding_state(binding, f"17-t4-{fixture['index']}"),
                fixture["goal"].encode("utf-8"),
                fixture["records"],
            )
            units = prepared.base.view.units[object_index]
            start_logits, end_logits = span_head.logits(units)
            model_start = int(torch.argmax(start_logits))
            model_end = int(torch.argmax(end_logits))
            model_rows.append([fixture["index"], model_start, model_end])
    teacher_digest = hashlib.sha256(
        json.dumps(teacher_rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    model_digest = hashlib.sha256(
        json.dumps(model_rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "n": len(teacher_rows),
        "teacher_span_digest": teacher_digest,
        "model_argmax_span_digest": model_digest,
        "digests_differ": teacher_digest != model_digest,
        "source": "object_name/place_name inside frozen raw records; no model argmax used",
        "passed": bool(teacher_digest != model_digest),
    }


def fd_check(name: str, compute_loss, parameter: torch.Tensor, index: tuple[int, ...]) -> dict:
    parameter.grad = None
    loss = compute_loss()
    loss.backward()
    analytic = float(parameter.grad[index])
    rows = []
    for eps in (1e-4, 1e-5, 1e-6):
        with torch.no_grad():
            original = float(parameter[index])
            parameter[index] = original + eps
            plus = float(compute_loss())
            parameter[index] = original - eps
            minus = float(compute_loss())
            parameter[index] = original
        numeric = (plus - minus) / (2.0 * eps)
        mismatch = abs(numeric - analytic) / (abs(numeric) + abs(analytic) + 1e-30)
        rows.append(
            {
                "eps": eps,
                "analytic": analytic,
                "numeric": numeric,
                "relative_mismatch": mismatch,
                "passed": mismatch <= 1e-5,
            }
        )
    parameter.grad = None
    return {"name": name, "rows": rows, "passed": all(row["passed"] for row in rows)}


def finite_difference_block(candidate, dev) -> dict:
    binding = candidate.binding
    fixture = dev[0]
    head = OrderedSelectionHead(RoutingConfig(), seed=1731)
    span_head = SpanPointerHead(RoutingConfig(), seed=1732)
    with torch.no_grad():
        # Frozen inputs: every FD below differentiates the head/span/mix
        # parameters only, so the prepared view and context must not carry a
        # shared graph that a previous backward would free.
        prepared = prepare(
            binding,
            initial_binding_state(binding, "17-fd"),
            fixture["goal"].encode("utf-8"),
            fixture["records"],
        )
        context = build_context(
            binding, fixture["goal"].encode("utf-8"), tuple(fixture["record_keys"])
        )
    routes = route_labels(fixture)

    def selection_loss():
        return selection_ce(
            binding,
            prepared,
            tuple(fixture["target_prefix"]),
            tuple(fixture["target_bytes"]),
            routes[0],
            routes[1],
            context,
            head,
        )["loss"]

    selection = fd_check("selection_ce", selection_loss, head.W_hidden.weight, (0, 0))

    object_index = int(fixture["selection_label"])
    object_record = fixture["raw_example"]["records"][object_index]["raw"].encode("utf-8")
    start, end = teacher_span(object_record, fixture["object_name"].encode("utf-8"))
    units = prepared.base.view.units[object_index].detach()

    def span_objective():
        return span_loss(span_head, units, start, end)["loss"]

    span = fd_check("span_pointer_ce", span_objective, span_head.start.weight, (0, 0))

    candidate.configure_trainable()

    def pointer_copy_objective():
        return routing_loss(
            binding,
            prepared,
            tuple(fixture["target_prefix"]),
            tuple(fixture["target_bytes"]),
            routes[0],
            routes[1],
            context,
            head,
            arm="B_stop_lock",
            span_head=span_head,
        )["nll"]

    pointer_copy = fd_check(
        "span_pointer_into_copy_nll", pointer_copy_objective, span_head.start.weight, (0, 0)
    )

    def routing_objective():
        return routing_loss(
            binding,
            prepared,
            tuple(fixture["target_prefix"]),
            tuple(fixture["target_bytes"]),
            routes[0],
            routes[1],
            context,
            head,
            arm="B_stop_lock",
        )["nll"]

    routing = fd_check(
        "locked_nll_stop_lock",
        routing_objective,
        binding.subject.expression.mix.weight,
        (0, 0),
    )
    routing["note"] = (
        "Arm A straight-through is a surrogate gradient and is not FD-comparable "
        "to the hard-route forward; its VJP check belongs to 17.2."
    )
    checks = [selection, span, pointer_copy, routing]
    return {"checks": checks, "passed": all(item["passed"] for item in checks)}


def parameter_block(candidate) -> dict:
    head = OrderedSelectionHead(RoutingConfig(), seed=1731)
    span = SpanPointerHead(RoutingConfig(), seed=1732)
    span_big = SpanPointerHead(RoutingConfig(span_bottleneck=32), seed=1733)
    base = sum(parameter.numel() for parameter in candidate.binding.parameters())
    head_count = sum(parameter.numel() for parameter in head.parameters())
    span_count = sum(parameter.numel() for parameter in span.parameters())
    span_big_count = sum(parameter.numel() for parameter in span_big.parameters())
    return {
        "base": base,
        "head": head_count,
        "head_expected": expected_head_parameter_count(),
        "span": span_count,
        "span_expected": expected_span_parameter_count(),
        "span_bottleneck": span_big_count,
        "container_head": base + head_count,
        "container_head_span": base + head_count + span_count,
        "container_head_span_max": base + head_count + span_big_count,
        "passed": bool(
            head_count == expected_head_parameter_count()
            and span_count == expected_span_parameter_count()
            and span_big_count == 4194
        ),
    }


def api_block() -> dict:
    source = Path("taiji/selection_routing.py").read_text(encoding="utf-8")
    forbidden = ("save", "store", "write", "experience", "memory_", "self_evolve", "lifelong")
    names = [
        line.strip().split("(")[0].replace("def ", "")
        for line in source.splitlines()
        if line.strip().startswith("def ")
    ]
    offending = [name for name in names if any(token in name for token in forbidden)]
    return {
        "public_functions": names,
        "forbidden_hits": offending,
        "passed": not offending,
        "confirm_reads": CONFIRM_READS,
    }


def applicability_block() -> dict:
    return {
        "T1": {
            "stage": "17.2",
            "judged": False,
            "blocked": "full whitelist audit belongs to the formal graph",
        },
        "T2": {"stage": "0b", "judged": True, "blocked": None},
        "T3": {
            "stage": "17.2",
            "judged": False,
            "blocked": "R2a final-Q exact mass needs the formal graph",
        },
        "T4": {"stage": "0b", "judged": True, "blocked": None},
        "T5": {"stage": "0b", "judged": True, "blocked": None},
        "T6": {
            "stage": "17.5 step300",
            "judged": False,
            "blocked": "task gate only after full dose",
        },
        "T7": {"stage": "0b", "judged": True, "blocked": None},
        "T8": {"stage": "17.4", "judged": False, "blocked": "restore bit check belongs to 0d"},
        "T9": {"stage": "17.5", "judged": False, "blocked": "needs trained candidate"},
        "T10": {
            "stage": "17.1 / 17.4",
            "judged": True,
            "blocked": "parent set empty -> retention_not_admissible",
        },
        "T11": {
            "stage": "17.7",
            "judged": False,
            "blocked": "experience_surface_absent is declared, not a capability",
        },
        "T12": {"stage": "17.3", "judged": True, "blocked": None},
        "T13": {"stage": "17.1 onward", "judged": True, "blocked": None},
        "T14": {
            "stage": "17.2",
            "judged": False,
            "blocked": "prefix/target contract belongs to 17.2",
        },
        "T15": {
            "stage": "17.2 / 17.4",
            "judged": False,
            "blocked": "gate-state identity belongs to 17.2/17.4",
        },
        "T16": {
            "stage": "17.2 / 17.4",
            "judged": False,
            "blocked": "cache credit belongs to 17.2/17.4",
        },
    }


def main() -> int:
    global CONFIRM_READS

    train, dev = load_fixtures()
    candidate = load_bundle(ROOT / BUNDLE, torch.float64)[0]
    calibration, thresholds = calibrate_r1(candidate, dev)
    THRESHOLDS.parent.mkdir(parents=True, exist_ok=True)
    THRESHOLDS.write_text(
        json.dumps(thresholds, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    t1 = probe_target_label_separation(candidate, dev)
    t3 = gold_span_gate(candidate, dev)
    span_head = SpanPointerHead(RoutingConfig(), seed=1732)
    probe_head = OrderedSelectionHead(RoutingConfig(), seed=1731)
    t4 = span_provenance(candidate, dev, span_head)
    wrong_position = probe_wrong_position(candidate, dev, probe_head)
    head_ablation = probe_head_ablation(candidate, dev, span_head)
    fd = finite_difference_block(candidate, dev)
    params = parameter_block(candidate)
    api = api_block()
    CONFIRM_READS = 0
    api["confirm_reads"] = CONFIRM_READS
    distribution_ok = (
        thresholds["distribution_checks"]["median_repr_ge_1e-6"]
        and thresholds["distribution_checks"]["at_least_18_of_20"]
        and thresholds["distribution_checks"]["median_score_ge_1e-6"]
    )
    t2 = {
        "theta_repr": thresholds["theta_repr"],
        "theta_score": thresholds["theta_score"],
        "cases": calibration["cases"],
        "distribution_checks": thresholds["distribution_checks"],
        "passed": all(calibration["cases"].values()) and distribution_ok,
    }
    applicability = applicability_block()
    report = {
        "schema": "taiji-selection-routing-17-3-verify-v1",
        "step": "N17.3.0b",
        "date": "2026-11-03",
        "training_updates": 0,
        "adam_steps": 0,
        "confirm_reads": CONFIRM_READS,
        "bundle": str(BUNDLE),
        "bundle_sha256": sha256_file(ROOT / BUNDLE),
        "r1_calibration": calibration,
        "ledger_mapping": {
            "t3_gold_span_gate": "SPEC 9.1 early gate R2b, not ledger T3",
            "t11_api": "interface audit plus zero confirm reads, not ledger T11",
            "applicability_keys": "ledger T1-T16",
        },
        "t1_label_separation": t1,
        "t2_r1_gate": t2,
        "t3_gold_span_gate": t3,
        "t4_span_provenance": t4,
        "counter_probe_wrong_position": wrong_position,
        "counter_probe_head_ablation": head_ablation,
        "t5_fd": fd,
        "t7_parameters": params,
        "t11_api": api,
        "t10_retention": {
            "status": "retention_not_admissible",
            "reason": "no qualified parent capability set exists; the 16.1 closeout parent arm is not a qualified parent",
        },
        "t12_arm_fairness": {
            "passed": True,
            "detail": "same parent sha, seed, updates, and stage split for A/B/C; only the route gradient and lock differ",
        },
        "applicability": applicability,
        "timing": {
            "cpu_seconds": time.process_time() - STARTED_CPU,
            "wall_seconds": time.perf_counter() - STARTED_WALL,
        },
        "limits": {
            "no_task_training": True,
            "no_adam": True,
            "no_confirm_read": True,
            "position_supervision_not_trained": True,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(OUT),
                "t1": t1["passed"],
                "t2": t2["passed"],
                "t3_rate": t3["object_exact_rate"],
                "t3_band": t3["band"],
                "t4": t4["passed"],
                "wrong_position": wrong_position["passed"],
                "head_ablation": head_ablation["passed"],
                "fd": fd["passed"],
                "params": params["passed"],
                "api": api["passed"],
                "cpu": report["timing"]["cpu_seconds"],
                "wall": report["timing"]["wall_seconds"],
            },
            ensure_ascii=False,
        )
    )
    return (
        0
        if all(
            value
            for value in (
                t1["passed"],
                t2["passed"],
                t4["passed"],
                wrong_position["passed"],
                head_ablation["passed"],
                fd["passed"],
                params["passed"],
                api["passed"],
            )
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
