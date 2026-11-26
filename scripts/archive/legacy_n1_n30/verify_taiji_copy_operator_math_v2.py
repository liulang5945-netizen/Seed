"""Independent OP2 mathematics checks. No candidate model or optimizer is used.

The Python/log-domain reference, rational latent-path enumeration, analytic
partials, and PyTorch tensor graph are deliberately separate formulations.
OP2 is a proposed record-first probability kernel, not a runtime replacement.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import itertools
import json
import math
import random
import subprocess
import sys
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
NEG_INF = float("-inf")
COUNTS = {"reference_calls": 0, "tensor_calls": 0, "runtime_helper_calls": 0}


def logsumexp(values):
    maximum = max(values, default=NEG_INF)
    if maximum == NEG_INF:
        return NEG_INF
    return maximum + math.log(math.fsum(math.exp(x - maximum) for x in values))


def log_probability(value):
    return math.log(value) if value > 0 else NEG_INF


def log_one_minus_exp(value):
    if value > 0:
        raise ArithmeticError("Negative restart probability")
    if value == 0:
        return NEG_INF
    if value < -math.log(2):
        return math.log1p(-math.exp(value))
    return math.log(-math.expm1(value))


def log_softmax(values, temperature=1.0):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    if not values or any(math.isnan(x) or x == math.inf for x in values):
        raise ValueError("Invalid logits")
    maximum = max(values)
    if maximum == NEG_INF:
        raise ValueError("All positions are masked")
    shifted = [(x - maximum) / temperature for x in values]
    if any(math.isfinite(x) and not math.isfinite(y) for x, y in zip(values, shifted, strict=True)):
        raise OverflowError("Scaled logit gap is outside the numeric range")
    normalizer = logsumexp(shifted)
    return [x - normalizer for x in shifted]


def probabilities(log_values):
    return [math.exp(x) for x in log_values]


def canonical_log_cursor(rows):
    """Shrink only an impossible roundoff excess; never normalize low mass."""
    flat = [x for row in rows for x in row]
    total_log = logsumexp(flat)
    probability_sum = math.fsum(math.exp(x) for x in flat)
    total_log = max(total_log, math.log(probability_sum) if probability_sum > 1 else NEG_INF)
    correction = 0.0
    if total_log > 0:
        size = max(1, sum(len(row) for row in rows))
        if total_log > 128 * sys.float_info.epsilon * size:
            raise ArithmeticError("Posterior copy mass exceeds its probability domain")
        correction = total_log + 8 * sys.float_info.epsilon * size
    return [[x - correction for x in row] for row in rows], correction


def validate(case):
    n = len(case["bytes"])
    if len(case["route"]) != n + 1 or len(case["m"]) != n:
        raise ValueError("Record/route/state shapes differ")
    if len(case["generator"]) < 2 or not 0 <= case["gamma"] <= 1:
        raise ValueError("Invalid output domain")
    for field in ("source", "transition", "bias", "lambda"):
        if len(case[field]) != n:
            raise ValueError("Per-record shapes differ")
    masses = []
    for i, byte_ids in enumerate(case["bytes"]):
        length = len(byte_ids)
        if not length:
            raise ValueError("Empty records must be excluded before routing")
        if any(not isinstance(b, int) or not 0 <= b < len(case["generator"]) - 1 for b in byte_ids):
            raise ValueError("Sources contain an invalid byte or EOS")
        if not (len(case["source"][i]) == len(case["bias"][i]) == len(case["m"][i]) == length):
            raise ValueError("Position shapes differ")
        if len(case["transition"][i]) != length or any(
            len(row) != length for row in case["transition"][i]
        ):
            raise ValueError("Transition shapes differ")
        if any(not math.isfinite(x) for x in case["bias"][i]):
            raise ValueError("Pointer biases must be finite")
        for row in case["transition"][i]:
            for value, bias in zip(row, case["bias"][i], strict=True):
                if math.isfinite(value) and not math.isfinite(value + bias):
                    raise OverflowError("Biased logit outside the numeric range")
        if not math.isfinite(case["lambda"][i]) or not 0 <= case["lambda"][i] <= 1:
            raise ValueError("Lambda is an effective probability")
        if any(not math.isfinite(x) or x < 0 for x in case["m"][i]) or case["m"][i][0] != 0:
            raise ValueError("Cursor is outside the normal reachable domain")
        masses.append(math.fsum(case["m"][i]))
    if math.fsum(masses) > 1:
        raise ValueError("Cursor mass exceeds one")
    if "log_m" in case:
        if len(case["log_m"]) != n:
            raise ValueError("Log cursor record shape differs")
        for row, ids, displayed in zip(case["log_m"], case["bytes"], case["m"], strict=True):
            if len(row) != len(ids) or row[0] != NEG_INF:
                raise ValueError("Log cursor is outside the reachable domain")
            if any(math.isnan(x) or x > 0 for x in row):
                raise ValueError("Invalid log cursor")
            if any(
                not math.isclose(math.exp(x), y, rel_tol=1e-12, abs_tol=1e-300)
                for x, y in zip(row, displayed, strict=True)
            ):
                raise ValueError("Probability view and authoritative log cursor disagree")
        canonical_log_cursor(case["log_m"])
    if "log_noncontinuation" in case:
        if len(case["log_noncontinuation"]) != n or "log_m" not in case:
            raise ValueError("Restart complement requires a matching log cursor")
        for row, complement in zip(case["log_m"], case["log_noncontinuation"], strict=True):
            if math.isnan(complement) or complement > 0:
                raise ValueError("Invalid restart complement")
            tolerance = 256 * sys.float_info.epsilon * max(1, sum(len(x) for x in case["log_m"]))
            if abs(logsumexp([logsumexp(row), complement])) > tolerance:
                raise ValueError("Cursor and restart complement disagree")


def reference(case):
    """Log-domain OP2, retaining finite log likelihoods after exp underflow."""
    COUNTS["reference_calls"] += 1
    validate(case)
    log_a = log_softmax(case["route"], case["tr"])
    log_g = log_softmax(case["generator"])
    log_m, _ = canonical_log_cursor(
        case.get("log_m", [[log_probability(x) for x in row] for row in case["m"]])
    )
    log_p, log_eta, log_q = [], [], []
    for i, byte_ids in enumerate(case["bytes"]):
        source = log_softmax(case["source"][i], case["ts"])
        transitions = [
            (
                log_softmax([x + b for x, b in zip(row, case["bias"][i], strict=True)], case["tt"])
                if source[r] != NEG_INF
                else [NEG_INF] * len(byte_ids)
            )
            for r, row in enumerate(case["transition"][i])
        ]
        fresh = [
            logsumexp([source[r] + transitions[r][j] for r in range(len(byte_ids))])
            for j in range(len(byte_ids))
        ]
        lam = case["lambda"][i]
        if "log_noncontinuation" in case:
            log_restart = logsumexp(
                [log_probability(1 - lam), log_probability(lam) + case["log_noncontinuation"][i]]
            )
        else:
            log_restart = log_one_minus_exp(log_probability(lam) + logsumexp(log_m[i]))
        eta = [
            logsumexp([log_probability(lam) + log_m[i][j], log_restart + fresh[j]])
            for j in range(len(byte_ids))
        ]
        eta_normalizer = logsumexp(eta)
        tolerance = 1024 * sys.float_info.epsilon * max(1, sum(len(row) for row in case["bytes"]))
        if not math.isfinite(eta_normalizer) or abs(eta_normalizer) > tolerance:
            raise ArithmeticError("Record conditional mass is inconsistent")
        eta = [value - eta_normalizer for value in eta]
        log_p.append(fresh)
        log_eta.append(eta)
        log_q.append([log_a[i] + x for x in eta])
    gamma = case["gamma"]
    log_gen_mass = logsumexp([log_probability(gamma), log_probability(1 - gamma) + log_a[-1]])
    log_output = []
    for byte in range(len(log_g)):
        terms = [log_gen_mass + log_g[byte]]
        terms.extend(
            log_probability(1 - gamma) + log_q[i][j]
            for i, ids in enumerate(case["bytes"])
            for j, source_byte in enumerate(ids)
            if source_byte == byte
        )
        log_output.append(logsumexp(terms))
    return MappingProxyType(
        {
            "log_a": tuple(log_a),
            "log_g": tuple(log_g),
            "log_p": tuple(map(tuple, log_p)),
            "log_eta": tuple(map(tuple, log_eta)),
            "log_q": tuple(map(tuple, log_q)),
            "log_gen_mass": log_gen_mass,
            "log_output": tuple(log_output),
            "gamma": gamma,
            "byte_ids": tuple(map(tuple, case["bytes"])),
            "vocabulary_size": len(log_g),
        }
    )


def accept(result, byte):
    """Use only the immutable forward packet, including its effective gamma."""
    if not isinstance(byte, int) or not 0 <= byte < result["vocabulary_size"]:
        raise ValueError("Invalid accepted token")
    denominator = result["log_output"][byte]
    if denominator == NEG_INF:
        raise ValueError("Conditioning on an impossible observation")
    copy_likelihoods = [
        [
            (
                log_probability(1 - result["gamma"]) + result["log_q"][i][j]
                if source_byte == byte
                else NEG_INF
            )
            for j, source_byte in enumerate(ids)
        ]
        for i, ids in enumerate(result["byte_ids"])
    ]
    latent = [result["log_gen_mass"] + result["log_g"][byte]] + [
        x for row in copy_likelihoods for x in row
    ]
    normalized = log_softmax(latent)
    log_r, offset = [], 1
    for ids in result["byte_ids"]:
        log_r.append(normalized[offset : offset + len(ids)])
        offset += len(ids)
    log_r, correction = canonical_log_cursor(log_r)
    responsibilities = [probabilities(row) for row in log_r]
    generated = math.exp(normalized[0])
    next_m = [[0.0] + row[:-1] for row in responsibilities]
    next_log_m = [[NEG_INF] + row[:-1] for row in log_r]
    complements, offset = [], 1
    for ids in result["byte_ids"]:
        continuable = set(range(offset, offset + len(ids) - 1))
        complement = logsumexp(
            [value for j, value in enumerate(normalized) if j not in continuable]
        )
        if complement > 128 * sys.float_info.epsilon * len(normalized):
            raise ArithmeticError("Restart complement exceeds its probability domain")
        complements.append(min(0.0, complement))
        offset += len(ids)
    return {
        "r": responsibilities,
        "r_gen": generated,
        "m": next_m,
        "log_m": next_log_m,
        "log_noncontinuation": complements,
        "roundoff_log_correction": correction,
    }


def example():
    return {
        "bytes": [[0, 0, 1], [0, 2]],
        "route": [0.3, -0.2, 0.1],
        "source": [[0.2, -0.4, 0.1], [-0.1, 0.3]],
        "transition": [
            [[0.3, -0.2, 0.1], [0.2, 0.4, -0.3], [-0.4, 0.2, 0.1]],
            [[0.1, -0.2], [0.4, 0.2]],
        ],
        "bias": [[0.1, -0.1, 0.2], [0.05, -0.05]],
        "m": [[0.0, 0.12, 0.08], [0.0, 0.16]],
        "lambda": [0.6, 0.4],
        "gamma": 0.35,
        "generator": [0.2, -0.3, 0.1, -0.1],
        "ts": 0.8,
        "tt": 0.9,
        "tr": 1.1,
    }


def assert_close(actual, expected, tolerance=1e-10):
    if not math.isfinite(actual) or abs(actual - expected) > tolerance:
        raise AssertionError((actual, expected, tolerance))


def probability_properties():
    rng = random.Random(20261104)
    worst = 0.0
    for _ in range(240):
        case = example()
        case["gamma"] = rng.choice([0.0, 0.2, 0.7, 1.0, rng.random()])
        case["lambda"] = [rng.choice([0.0, 1.0, rng.random()]) for _ in case["bytes"]]
        case["route"] = [rng.uniform(-6, 6) for _ in case["route"]]
        case["ts"], case["tt"] = rng.choice([0.05, 0.5, 1.0, 3.0]), rng.choice([0.05, 1.0, 2.0])
        case["m"] = [[0.0, rng.random() * 0.2, rng.random() * 0.2], [0.0, rng.random() * 0.2]]
        result = reference(case)
        worst = max(worst, abs(math.fsum(probabilities(result["log_output"])) - 1))
        shrunk = copy.deepcopy(case)
        shrunk["m"] = [[x * 0.9999 for x in row] for row in case["m"]]
        removed = math.fsum(x for row in case["m"] for x in row) * 0.0001
        changed = reference(shrunk)
        emission_error = math.fsum(
            abs(x - y)
            for x, y in zip(
                probabilities(result["log_output"]),
                probabilities(changed["log_output"]),
                strict=True,
            )
        )
        assert emission_error <= 2 * removed + 1e-12
        for eta in result["log_eta"]:
            assert_close(math.fsum(probabilities(eta)), 1.0)
        for byte in range(len(case["generator"])):
            child = accept(result, byte)
            total_copy = math.fsum(x for row in child["r"] for x in row)
            assert_close(total_copy + child["r_gen"], 1.0)
            assert math.fsum(x for row in child["m"] for x in row) <= total_copy + 1e-12
            assert all(row[0] == 0 for row in child["m"])
            for ids, row in zip(case["bytes"], child["m"], strict=True):
                assert all(value == 0 or ids[j - 1] == byte for j, value in enumerate(row) if j > 0)
            if byte == len(case["generator"]) - 1:
                assert total_copy == 0 and all(x == 0 for row in child["m"] for x in row)
    return {"cases": 240, "worst_mass_error": worst}


def exact_latent_enumeration():
    """Independent Fraction hidden-state model; does not use OP2 eta."""
    F = Fraction
    ids = [[0, 1], [1, 0]]
    a, fresh, lam, gen, gamma = (
        [F(2, 5), F(7, 20), F(1, 4)],
        [[F(7, 10), F(3, 10)], [F(2, 5), F(3, 5)]],
        [F(3, 5), F(4, 5)],
        [F(1, 5), F(3, 10), F(1, 2)],
        F(3, 10),
    )
    states = [None, (0, 0), (0, 1), (1, 0), (1, 1)]

    def transition(old, new):
        if new is None:
            return gamma + (1 - gamma) * a[-1]
        i, j = new
        can_continue = old is not None and old[0] == i and old[1] + 1 < len(ids[i])
        continued = lam[i] if can_continue and j == old[1] + 1 else F(0)
        restart = (1 - lam[i]) if can_continue else F(1)
        return (1 - gamma) * a[i] * (continued + restart * fresh[i][j])

    case = {
        "bytes": ids,
        "route": [math.log(float(x)) for x in a],
        "source": [[0.0, 0.0], [0.0, 0.0]],
        "transition": [[[math.log(float(x)) for x in row]] * 2 for row in fresh],
        "bias": [[0.0, 0.0], [0.0, 0.0]],
        "m": [[0.0, 0.0], [0.0, 0.0]],
        "lambda": [float(x) for x in lam],
        "gamma": float(gamma),
        "generator": [math.log(float(x)) for x in gen],
        "ts": 1.0,
        "tt": 1.0,
        "tr": 1.0,
    }
    tested, worst, smoothing_witness = 0, 0.0, None
    for length in (1, 2, 3):
        for prefix in itertools.product((0, 1), repeat=length - 1):
            for last in (0, 1, 2):
                sequence = prefix + (last,)
                paths = {(): F(1)}
                current = copy.deepcopy(case)
                log_joint, online_copy = 0.0, 0.0
                for byte in sequence:
                    next_paths = {}
                    for path, weight in paths.items():
                        old = path[-1] if path else None
                        for new in states:
                            emission = (
                                gen[byte] if new is None else F(int(ids[new[0]][new[1]] == byte))
                            )
                            probability = weight * transition(old, new) * emission
                            if probability:
                                next_paths[path + (new,)] = probability
                    paths = next_paths
                    evidence = sum(paths.values())
                    result = reference(current)
                    child = accept(result, byte)
                    log_joint += result["log_output"][byte]
                    for i, row in enumerate(child["r"]):
                        for j, value in enumerate(row):
                            exact = (
                                sum(weight for path, weight in paths.items() if path[-1] == (i, j))
                                / evidence
                            )
                            assert_close(value, float(exact))
                    online_copy += math.fsum(x for row in child["r"] for x in row)
                    current["m"] = child["m"]
                    current["log_m"] = child["log_m"]
                    current["log_noncontinuation"] = child["log_noncontinuation"]
                worst = max(worst, abs(math.exp(log_joint) - float(evidence)))
                smooth_count = (
                    sum(
                        weight * sum(state is not None for state in path)
                        for path, weight in paths.items()
                    )
                    / evidence
                )
                if (
                    len(sequence) > 1
                    and abs(online_copy - float(smooth_count)) > 1e-5
                    and smoothing_witness is None
                ):
                    smoothing_witness = {
                        "sequence": sequence,
                        "online_responsibility_count": online_copy,
                        "all_history_smoothed_count": float(smooth_count),
                    }
                tested += 1
    assert worst < 1e-12 and smoothing_witness is not None
    return {
        "prefixes": tested,
        "maximum_joint_error": worst,
        "coverage_is_not_smoothing": smoothing_witness,
    }


def counterexamples():
    F = Fraction
    old = []
    for alpha in (F(1, 10), F(9, 10)):
        raw = [F(3, 5) * alpha * F(99, 100), F(2, 5) + F(3, 5) * alpha * F(1, 100)]
        old.append([x / sum(raw) for x in raw])
    assert old[0][0] < old[0][1] and old[1][0] > old[1][1]
    v2 = [F(3, 5) * F(99, 100), F(2, 5) + F(3, 5) * F(1, 100)]
    assert v2 == [F(297, 500), F(203, 500)]
    previous = {
        "bytes": [[0, 1]],
        "route": [0.0, NEG_INF],
        "source": [[0.0, 0.0]],
        "transition": [[[math.log(0.8), math.log(0.2)], [math.log(0.8), math.log(0.2)]]],
        "bias": [[0.0, 0.0]],
        "m": [[0.0, 0.0]],
        "lambda": [0.8],
        "gamma": 0.5,
        "generator": [math.log(0.8), math.log(0.1), math.log(0.1)],
        "ts": 1.0,
        "tt": 1.0,
        "tr": 1.0,
    }
    reachable = accept(reference(previous), 0)["m"][0]
    assert_close(reachable[0], 0.0)
    assert_close(reachable[1], 0.5)
    arithmetic = 0.9 * 0.6 + 0.1 * 0.01
    geometry = probabilities(
        log_softmax(
            [0.9 * math.log(0.6) + 0.1 * math.log(0.01), 0.9 * math.log(0.4) + 0.1 * math.log(0.99)]
        )
    )[0]
    assert arithmetic > 0.5 > geometry
    # Sharp fresh location plus a different cursor is still a two-point eta.
    assert [0.8 * 0.0 + 0.6, 0.8 * 0.5] == [0.6, 0.4]
    # Legacy geometric pooling with tied sources differs from its hard tie-break.
    logits = [[2.0, 0.0], [0.0, 4.0]]
    soft_target = probabilities(
        log_softmax([0.5 * (row0 + row1) for row0, row1 in zip(*logits, strict=True)], 0.001)
    )
    hard_target = max(range(2), key=logits[0].__getitem__)
    soft_winner = max(range(2), key=soft_target.__getitem__)
    assert soft_winner != hard_target
    return {
        "reachable_previous_cursor": reachable,
        "reachable_cursor_old_alpha_flip": [[str(x) for x in row] for row in old],
        "record_first_eta": [str(x) for x in v2],
        "arithmetic_target0": arithmetic,
        "geometric_target0": geometry,
        "hard_fresh_not_final_delta": [0.6, 0.4],
        "legacy_source_tie_counterexample": {
            "beta": [0.5, 0.5],
            "logits": logits,
            "soft_logit_pool_winner": soft_winner,
            "hard_first_source_winner": hard_target,
        },
    }


def invariances_and_limits():
    case = example()
    baseline = reference(case)
    shifted = copy.deepcopy(case)
    shifted["route"] = [x + 7 for x in shifted["route"]]
    for i in range(len(shifted["source"])):
        shifted["source"][i] = [x + 3 for x in shifted["source"][i]]
        shifted["transition"][i] = [
            [x + 2 * (j + 1) for x in row] for j, row in enumerate(shifted["transition"][i])
        ]
    for x, y in zip(baseline["log_output"], reference(shifted)["log_output"], strict=True):
        assert_close(x, y)
    reordered = copy.deepcopy(case)
    for field in ("bytes", "source", "transition", "bias", "m", "lambda"):
        reordered[field] = list(reversed(reordered[field]))
    reordered["route"] = list(reversed(case["route"][:-1])) + [case["route"][-1]]
    for x, y in zip(baseline["log_output"], reference(reordered)["log_output"], strict=True):
        assert_close(x, y)
    source_reordered = copy.deepcopy(case)
    source_reordered["source"][0] = list(reversed(case["source"][0]))
    source_reordered["transition"][0] = list(reversed(case["transition"][0]))
    for x, y in zip(baseline["log_output"], reference(source_reordered)["log_output"], strict=True):
        assert_close(x, y)
    # Target relabelling fixes the first physical position to retain the normal m domain.
    perm = [0, 2, 1]
    target_reordered = copy.deepcopy(case)
    for field in ("bytes", "bias", "m"):
        target_reordered[field][0] = [case[field][0][j] for j in perm]
    target_reordered["transition"][0] = [[row[j] for j in perm] for row in case["transition"][0]]
    for x, y in zip(baseline["log_output"], reference(target_reordered)["log_output"], strict=True):
        assert_close(x, y)
    n = 3
    source_gap, target_gap, ts, tt = 0.7, 0.8, 0.1, 0.15
    weights = probabilities(log_softmax([source_gap, 0, -0.4], ts))
    rows = [probabilities(log_softmax([target_gap, 0, -0.2], tt)), [0.1, 0.7, 0.2], [0.2, 0.3, 0.5]]
    fresh0 = math.fsum(weights[r] * rows[r][0] for r in range(n))
    bound = (n - 1) * math.exp(-source_gap / ts) + (n - 1) * math.exp(-target_gap / tt)
    assert 1 - fresh0 <= bound
    return {
        "logit_shift_and_record_permutation": True,
        "unique_gap_fresh_error": 1 - fresh0,
        "proved_gap_bound": bound,
        "scope": "Fresh pointer bound only. Cursor mixing and hard projection need separate margins.",
    }


def edge_cases():
    case = example()
    packet = reference(case)
    original_posterior = accept(packet, 0)
    case["gamma"] = 0.99
    assert accept(packet, 0) == original_posterior
    try:
        packet["gamma"] = 0.0
    except TypeError:
        pass
    else:
        raise AssertionError("Forward packet is mutable")
    empty = {
        "bytes": [],
        "route": [0.0],
        "source": [],
        "transition": [],
        "bias": [],
        "m": [],
        "lambda": [],
        "gamma": 0.3,
        "generator": [0.2, -0.1, 0.4],
        "ts": 1.0,
        "tt": 1.0,
        "tr": 1.0,
    }
    result = reference(empty)
    assert result["log_output"] == result["log_g"]
    extreme = copy.deepcopy(empty)
    extreme["generator"] = [0.0, -2000.0, -1000.0]
    extreme_result = reference(extreme)
    assert (
        extreme_result["log_output"][1] == -2000 and math.exp(extreme_result["log_output"][1]) == 0
    )
    assert_close(accept(extreme_result, 1)["r_gen"], 1)
    invalid = []
    for field, value in [("ts", 0.0), ("tt", -1.0), ("gamma", 1.1), ("route", [NEG_INF] * 3)]:
        candidate = example()
        candidate[field] = value
        invalid.append(candidate)
    candidate = example()
    candidate["m"][0][0] = 0.1
    invalid.append(candidate)
    for candidate in invalid:
        try:
            reference(candidate)
        except ValueError:
            continue
        raise AssertionError("Invalid domain accepted")
    impossible = copy.deepcopy(empty)
    impossible["generator"] = [0.0, NEG_INF, 0.0]
    try:
        accept(reference(impossible), 1)
    except ValueError:
        pass
    else:
        raise AssertionError("Zero-probability Bayes conditioning accepted")
    assert probabilities(log_softmax([0.0, NEG_INF])) == [1.0, 0.0]
    disjoint = {
        "bytes": [[0, 1]],
        "route": [0.0, NEG_INF],
        "source": [[0.0, 0.0]],
        "transition": [[[0.0, NEG_INF], [NEG_INF, 0.0]]],
        "bias": [[0.0, 0.0]],
        "m": [[0.0, 0.0]],
        "lambda": [1.0],
        "gamma": 0.0,
        "generator": [0.0, 0.0, 0.0],
        "ts": 1.0,
        "tt": 1.0,
        "tr": 1.0,
    }
    assert probabilities(reference(disjoint)["log_p"][0]) == [0.5, 0.5]
    inactive = copy.deepcopy(disjoint)
    inactive["source"] = [[0.0, NEG_INF]]
    inactive["transition"] = [[[0.0, NEG_INF], [NEG_INF, NEG_INF]]]
    assert probabilities(reference(inactive)["log_p"][0]) == [1.0, 0.0]
    terminal = copy.deepcopy(disjoint)
    terminal["m"] = [[0.0, 1.0]]
    terminal_result = reference(terminal)
    assert probabilities(terminal_result["log_eta"][0]) == [0.0, 1.0]
    assert accept(terminal_result, 1)["m"] == [[0.0, 0.0]]
    singleton = {
        "bytes": [[0]],
        "route": [0.0, NEG_INF],
        "source": [[0.0]],
        "transition": [[[0.0]]],
        "bias": [[0.0]],
        "m": [[0.0]],
        "lambda": [1.0],
        "gamma": 0.0,
        "generator": [0.0, 0.0],
        "ts": 1.0,
        "tt": 1.0,
        "tr": 1.0,
    }
    assert accept(reference(singleton), 0)["m"] == [[0.0]]
    return {
        "immutable_forward_posterior_packet": True,
        "empty_is_generator": True,
        "underflow_log_likelihood": -2000.0,
        "invalid_domains_rejected": len(invalid),
        "impossible_observation_rejected": True,
        "disjoint_transition_masks": [0.5, 0.5],
        "last_position_and_singleton_restart": True,
    }


def tensor_forward(case, tensors, m, *, coupled=False, route_override=None):
    import torch

    COUNTS["tensor_calls"] += 1
    a = (
        torch.softmax(tensors["route"] / case["tr"], 0)
        if route_override is None
        else route_override
    )
    q, fresh, eta = [], [], []
    for i in range(len(case["bytes"])):
        u = torch.softmax(tensors[f"source{i}"] / case["ts"], 0)
        transition = torch.stack(
            [
                (
                    torch.softmax((row + tensors[f"bias{i}"]) / case["tt"], 0)
                    if bool(torch.isfinite(tensors[f"source{i}"][r]))
                    else row.new_zeros(len(row))
                )
                for r, row in enumerate(tensors[f"transition{i}"])
            ]
        )
        p = u @ transition
        d = tensors["lambda"][i] * m[i] + (1 - tensors["lambda"][i] * m[i].sum()) * p
        fresh.append(p)
        eta.append(d)
        q.append(a[i] * d)
    gamma = tensors["gamma"]
    logits = tensors["generator"]
    if coupled:
        content = sum((j + 1) * value for row in q for j, value in enumerate(row))
        logits = logits + content * logits.new_tensor([0.4, -0.2, 0.3, -0.1])
        gamma = 0.2 + 0.8 * torch.sigmoid(gamma + 0.3 * content)
    generated = torch.softmax(logits, 0)
    byte_mass = generated.new_zeros(len(case["generator"]))
    for i, ids in enumerate(case["bytes"]):
        byte_mass = byte_mass.index_add(0, torch.tensor(ids), q[i])
    output = (gamma + (1 - gamma) * a[-1]) * generated + (1 - gamma) * byte_mass
    return output, q, fresh, eta, generated, gamma, a


def tensor_parameters(case, dtype):
    import torch

    tensors = {
        key: torch.tensor(case[key], dtype=dtype, requires_grad=True)
        for key in ("route", "lambda", "generator", "gamma")
    }
    for i in range(len(case["bytes"])):
        for name in ("source", "transition", "bias"):
            tensors[f"{name}{i}"] = torch.tensor(case[name][i], dtype=dtype, requires_grad=True)
    for name, value in tensors.items():
        if name in ("route", "lambda", "generator", "gamma"):
            original = case[name]
        else:
            field, record = name.rstrip("0123456789"), int(name[len(name.rstrip("0123456789")) :])
            original = case[field][record]
        exact_range = torch.tensor(original, dtype=torch.float64)
        if bool((torch.isfinite(exact_range) & ~torch.isfinite(value)).any()):
            raise OverflowError("Tensor input conversion lost a finite logit")
    return tensors


def tensor_sequence_loss(case, tensors, sequence, *, coupled=False):
    import torch

    m = [torch.tensor(row, dtype=tensors["route"].dtype) for row in case["m"]]
    loss = tensors["route"].new_zeros(())
    for byte in sequence:
        output, q, _, _, _, gamma, _ = tensor_forward(case, tensors, m, coupled=coupled)
        loss = loss - torch.log(output[byte])
        r = [
            (1 - gamma) * row * torch.tensor([int(b == byte) for b in ids]) / output[byte]
            for row, ids in zip(q, case["bytes"], strict=True)
        ]
        m = [torch.cat([row.new_zeros(1), row[:-1]]) for row in r]
    return loss


def tensor_logsumexp(terms):
    import torch

    values = torch.stack(terms)
    if bool(torch.isnan(values).any()) or bool((values == math.inf).any()):
        raise ArithmeticError("Invalid log weight")
    mask = torch.isfinite(values)
    if not bool(mask.any()):
        return values.new_tensor(NEG_INF)
    return torch.logsumexp(values[mask], dim=0)


def tensor_log_probability(value):
    if float(value.detach()) < 0:
        raise ValueError("Negative probability")
    if float(value.detach()) == 0:
        return value.new_tensor(NEG_INF)
    return value.log()


def tensor_log_one_minus_exp(value):
    import torch

    scalar = float(value.detach())
    if scalar > 0:
        raise ArithmeticError("Negative restart probability")
    if scalar == 0:
        return value.new_tensor(NEG_INF)
    if scalar < -math.log(2):
        return torch.log1p(-torch.exp(value))
    return torch.log(-torch.expm1(value))


def tensor_stable_log_softmax(values, temperature=1.0):
    import torch

    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("Invalid temperature")
    if bool(torch.isnan(values).any()) or bool((values == math.inf).any()):
        raise ValueError("Invalid logits")
    if not bool(torch.isfinite(values).any()):
        raise ValueError("All logits masked")
    shifted = (values - values.max()) / temperature
    if bool((torch.isfinite(values) & ~torch.isfinite(shifted)).any()):
        raise OverflowError("Scaled tensor logit gap outside the numeric range")
    return shifted - torch.logsumexp(shifted, dim=0)


def tensor_biased_logits(row, bias):
    import torch

    values = row + bias
    if bool((torch.isfinite(row) & torch.isfinite(bias) & ~torch.isfinite(values)).any()):
        raise OverflowError("Biased tensor logit outside the numeric range")
    return values


def tensor_log_forward(case, tensors, log_m, log_noncontinuation=None):
    """Separate log-domain tensor graph, retaining log cursor across steps."""
    import torch

    COUNTS["tensor_calls"] += 1
    log_a = tensor_stable_log_softmax(tensors["route"], case["tr"])
    log_g = tensor_stable_log_softmax(tensors["generator"])
    log_q = []
    for i, ids in enumerate(case["bytes"]):
        u = tensor_stable_log_softmax(tensors[f"source{i}"], case["ts"])
        transition = [
            (
                tensor_stable_log_softmax(
                    tensor_biased_logits(row, tensors[f"bias{i}"]), case["tt"]
                )
                if bool(torch.isfinite(u[r]))
                else row.new_full(row.shape, NEG_INF)
            )
            for r, row in enumerate(tensors[f"transition{i}"])
        ]
        fresh = [
            tensor_logsumexp([u[r] + transition[r][j] for r in range(len(ids))])
            for j in range(len(ids))
        ]
        log_lambda = tensor_log_probability(tensors["lambda"][i])
        log_mass = tensor_logsumexp(list(log_m[i]))
        if log_noncontinuation is None:
            log_restart = tensor_log_one_minus_exp(log_lambda + log_mass)
        else:
            log_restart = tensor_logsumexp(
                [
                    tensor_log_probability(1 - tensors["lambda"][i]),
                    log_lambda + log_noncontinuation[i],
                ]
            )
        eta = [
            tensor_logsumexp([log_lambda + log_m[i][j], log_restart + fresh[j]])
            for j in range(len(ids))
        ]
        normalizer = tensor_logsumexp(eta)
        tolerance = (
            1024
            * torch.finfo(tensors["route"].dtype).eps
            * max(1, sum(len(row) for row in case["bytes"]))
        )
        if abs(float(normalizer.detach())) > tolerance:
            raise ArithmeticError("Record conditional mass is inconsistent")
        eta = [value - normalizer for value in eta]
        log_q.append(torch.stack([log_a[i] + value for value in eta]))
    log_gamma = tensor_log_probability(tensors["gamma"])
    log_copy_gate = tensor_log_probability(1 - tensors["gamma"])
    log_gen = tensor_logsumexp([log_gamma, log_copy_gate + log_a[-1]])
    output = []
    for byte in range(len(case["generator"])):
        terms = [log_gen + log_g[byte]]
        terms.extend(
            log_copy_gate + log_q[i][j]
            for i, ids in enumerate(case["bytes"])
            for j, b in enumerate(ids)
            if b == byte
        )
        output.append(tensor_logsumexp(terms))
    return torch.stack(output), log_q, log_copy_gate, log_gen + log_g


def tensor_log_sequence_loss(case, tensors, sequence):
    import torch

    initial_logs = case.get("log_m", [[log_probability(x) for x in row] for row in case["m"]])
    log_m = [torch.tensor(row, dtype=tensors["route"].dtype) for row in initial_logs]
    log_noncontinuation = (
        [torch.tensor(x, dtype=tensors["route"].dtype) for x in case["log_noncontinuation"]]
        if "log_noncontinuation" in case
        else None
    )
    loss = tensors["route"].new_zeros(())
    for byte in sequence:
        output, log_q, log_copy_gate, log_generated = tensor_log_forward(
            case, tensors, log_m, log_noncontinuation
        )
        if not bool(torch.isfinite(output[byte])):
            raise ValueError("Impossible observation")
        loss = loss - output[byte]
        likelihoods = [log_generated[byte]]
        likelihoods.extend(
            log_copy_gate + row[j] if b == byte else output.new_tensor(NEG_INF)
            for row, ids in zip(log_q, case["bytes"], strict=True)
            for j, b in enumerate(ids)
        )
        normalized = tensor_stable_log_softmax(torch.stack(likelihoods))
        log_noncontinuation, offset = [], 1
        for ids in case["bytes"]:
            continuable = set(range(offset, offset + len(ids) - 1))
            complement = tensor_logsumexp(
                [x for j, x in enumerate(normalized) if j not in continuable]
            )
            if float(complement.detach()) > 128 * torch.finfo(output.dtype).eps * len(normalized):
                raise ArithmeticError("Restart complement exceeds its probability domain")
            log_noncontinuation.append(torch.minimum(complement, complement.new_zeros(())))
            offset += len(ids)
        r, offset = [], 1
        for ids in case["bytes"]:
            r.append(normalized[offset : offset + len(ids)])
            offset += len(ids)
        log_total = (
            tensor_logsumexp([x for row in r for x in row]) if r else output.new_tensor(NEG_INF)
        )
        probability_total = sum((row.exp().sum() for row in r), output.new_zeros(()))
        epsilon = torch.finfo(output.dtype).eps
        excess = max(
            float(log_total.detach()),
            (
                math.log(float(probability_total.detach()))
                if float(probability_total.detach()) > 1
                else NEG_INF
            ),
        )
        if excess > 0:
            size = max(1, sum(len(row) for row in r))
            if excess > 128 * epsilon * size:
                raise ArithmeticError("Posterior mass exceeds tolerance")
            r = [row - (excess + 8 * epsilon * size) for row in r]
        log_m = [torch.cat([row.new_full((1,), NEG_INF), row[:-1]]) for row in r]
    return loss


def python_sequence_loss(case, sequence):
    current = copy.deepcopy(case)
    loss = 0.0
    for byte in sequence:
        result = reference(current)
        loss -= result["log_output"][byte]
        child = accept(result, byte)
        current["m"] = child["m"]
        current["log_m"] = child["log_m"]
        current["log_noncontinuation"] = child["log_noncontinuation"]
    return loss


def input_locations(case):
    for field in ("route", "lambda", "generator"):
        for j in range(len(case[field])):
            yield field, None, (j,)
    yield "gamma", None, ()
    for i in range(len(case["bytes"])):
        for field in ("source", "bias"):
            for j in range(len(case[field][i])):
                yield field, i, (j,)
        for j, row in enumerate(case["transition"][i]):
            for k in range(len(row)):
                yield "transition", i, (j, k)


def perturb(case, location, amount):
    field, record, indices = location
    obj = case if field == "gamma" else (case[field] if record is None else case[field][record])
    if field == "gamma":
        case[field] += amount
    elif len(indices) == 1:
        obj[indices[0]] += amount
    else:
        obj[indices[0]][indices[1]] += amount


def finite_difference_checks():
    import torch

    case = example()
    results = []
    for sequence in ([0], [0, 1, 3]):
        tensors = tensor_parameters(case, torch.float64)
        loss = tensor_sequence_loss(case, tensors, sequence)
        loss.backward()
        assert_close(float(loss.detach()), python_sequence_loss(case, sequence))
        log_tensors = tensor_parameters(case, torch.float64)
        log_loss = tensor_log_sequence_loss(case, log_tensors, sequence)
        log_loss.backward()
        assert_close(float(log_loss.detach()), float(loss.detach()))
        for name in tensors:
            assert torch.allclose(
                tensors[name].grad, log_tensors[name].grad, atol=1e-11, rtol=1e-10
            ), name
        worst, components = 0.0, 0
        for location in input_locations(case):
            field, record, indices = location
            key = field if record is None else f"{field}{record}"
            exact = float(tensors[key].grad[indices])
            passed = 0
            errors = []
            for eps in (1e-4, 1e-5, 1e-6):
                plus, minus = copy.deepcopy(case), copy.deepcopy(case)
                perturb(plus, location, eps)
                perturb(minus, location, -eps)
                fd = (
                    python_sequence_loss(plus, sequence) - python_sequence_loss(minus, sequence)
                ) / (2 * eps)
                error = abs(fd - exact)
                errors.append(error)
                passed += error <= 2e-7 + 2e-5 * abs(exact)
            if passed < 2:
                raise AssertionError((location, exact, errors))
            worst = max(worst, max(errors))
            components += 1
        results.append(
            {
                "sequence": sequence,
                "declared_parameter_components": components,
                "maximum_absolute_fd_error_all_eps": worst,
                "at_least_two_eps_pass_each": True,
            }
        )
    for dtype, tolerance in ((torch.float64, 1e-12), (torch.float32, 2e-6)):
        probe = tensor_parameters(case, dtype)
        output = tensor_forward(case, probe, [torch.tensor(row, dtype=dtype) for row in case["m"]])[
            0
        ]
        reference_output = probabilities(reference(case)["log_output"])
        assert_close(float(output.sum().detach()), 1.0, tolerance)
        assert (
            max(abs(float(x.detach()) - y) for x, y in zip(output, reference_output, strict=True))
            < tolerance
        )
        stable_loss = tensor_log_sequence_loss(case, probe, [0, 1, 3])
        stable_loss.backward()
        for name, value in probe.items():
            assert bool(torch.isfinite(value.grad).all()), name
            assert torch.allclose(
                value.grad.double(), tensors[name].grad, atol=2e-6, rtol=2e-5
            ), name
    return {
        "checks": results,
        "FD_precision": "float64",
        "normal_range_forward_and_three_step_gradient_precisions": ["float64", "float32"],
        "scope": "33 declared conditional input components at fixed initial cursor and temperatures; probability/log graphs agree. Boundary/state/temperature derivatives are not covered. No Taiji model instantiated.",
    }


def underflow_and_roundoff_regressions():
    import torch

    raw = [[math.log(0.08357455553872419), math.log(0.9164254444612759)]]
    repaired, correction = canonical_log_cursor(raw)
    assert correction > 0 and math.fsum(probabilities(repaired[0])) <= 1
    rng = random.Random(731104)
    corrections = 0
    for _ in range(64):
        case = {
            "bytes": [[0, 0, 1]],
            "route": [0.0, NEG_INF],
            "source": [[rng.uniform(-5, 5) for _ in range(3)]],
            "transition": [[[rng.uniform(-6, 6) for _ in range(3)] for _ in range(3)]],
            "bias": [[0.0, 0.0, 0.0]],
            "m": [[0.0, 0.0, 0.0]],
            "lambda": [rng.choice([0.0, 0.7, 1.0])],
            "gamma": 0.0,
            "generator": [0.0, 0.0, 0.0],
            "ts": 1.0,
            "tt": 1.0,
            "tr": 1.0,
        }
        for _step in range(16):
            result = reference(case)
            byte = max(range(3), key=result["log_output"].__getitem__)
            child = accept(result, byte)
            assert math.fsum(x for row in child["m"] for x in row) <= 1
            corrections += child["roundoff_log_correction"] > 0
            case["m"], case["log_m"] = child["m"], child["log_m"]
            case["log_noncontinuation"] = child["log_noncontinuation"]
    extremes = []
    for dtype, low in ((torch.float32, -110.0), (torch.float64, -800.0)):
        case = {
            "bytes": [],
            "route": [0.0],
            "source": [],
            "transition": [],
            "bias": [],
            "m": [],
            "lambda": [],
            "gamma": 0.3,
            "generator": [0.0, low, 0.0],
            "ts": 1.0,
            "tt": 1.0,
            "tr": 1.0,
        }
        old = tensor_parameters(case, dtype)
        old_loss = tensor_sequence_loss(case, old, [1])
        old_loss.backward()
        assert not bool(torch.isfinite(old_loss))
        assert not bool(torch.isfinite(old["generator"].grad).all())
        stable = tensor_parameters(case, dtype)
        stable_loss = tensor_log_sequence_loss(case, stable, [1])
        stable_loss.backward()
        assert bool(torch.isfinite(stable_loss)) and bool(
            torch.isfinite(stable["generator"].grad).all()
        )
        assert_close(float(stable_loss.detach()), python_sequence_loss(case, [1]), 2e-5)
        assert torch.allclose(
            stable["generator"].grad,
            stable["generator"].new_tensor([0.5, -1.0, 0.5]),
            atol=1e-6,
            rtol=0,
        )
        extremes.append(
            {
                "dtype": str(dtype),
                "target_logit": low,
                "probability_graph_failure_reproduced": True,
                "log_graph_loss": float(stable_loss.detach()),
                "generator_gradient": stable["generator"].grad.tolist(),
            }
        )
    first = {
        "bytes": [[0, 1]],
        "route": [0.0, NEG_INF],
        "source": [[0.0, 0.0]],
        "transition": [[[-110.0, 0.0], [-110.0, 0.0]]],
        "bias": [[0.0, 0.0]],
        "m": [[0.0, 0.0]],
        "lambda": [0.7],
        "gamma": 0.5,
        "generator": [0.0, -1000.0, -1000.0],
        "ts": 1.0,
        "tt": 1.0,
        "tr": 1.0,
    }
    second = copy.deepcopy(first)
    second["transition"] = [[[0.0, -220.0], [0.0, -220.0]]]
    second["generator"] = [0.0, -220.0, -220.0]
    first_tensors = tensor_parameters(first, torch.float32)
    second_tensors = tensor_parameters(second, torch.float32)
    log_out1, log_q1, log_copy1, log_gen1 = tensor_log_forward(
        first, first_tensors, [torch.tensor([NEG_INF, NEG_INF])]
    )
    first_posterior = tensor_stable_log_softmax(
        torch.stack([log_gen1[0], log_copy1 + log_q1[0][0], log_out1.new_tensor(NEG_INF)])
    )
    rare_copy_log = first_posterior[1]
    assert float(rare_copy_log.exp().detach()) == 0
    log_out2, _, _, _ = tensor_log_forward(
        second, second_tensors, [torch.stack([rare_copy_log.new_tensor(NEG_INF), rare_copy_log])]
    )
    prior = accept(reference(first), 0)
    second["m"], second["log_m"] = prior["m"], prior["log_m"]
    second["log_noncontinuation"] = prior["log_noncontinuation"]
    expected_log = reference(second)["log_output"][1]
    assert_close(float(log_out2[1].detach()), expected_log, 2e-5)
    (-log_out2[1]).backward()
    source_credit = first_tensors["transition0"].grad[:, 0]
    assert bool(torch.isfinite(source_credit).all())
    assert_close(float(source_credit.sum()), -1.0, 2e-5)
    balanced = copy.deepcopy(first)
    balanced["transition"] = [[[-10000.0, 0.0], [-10000.0, 0.0]]]
    balanced["generator"] = [-10000.0, 0.0, -10000.0]
    balanced_posterior = accept(reference(balanced), 0)
    assert_close(balanced_posterior["r_gen"], 0.5, 1e-14)
    assert_close(balanced_posterior["r"][0][0], 0.5, 1e-14)
    near_one = copy.deepcopy(first)
    near_one["m"], near_one["log_m"] = [[0.0, 1.0]], [[NEG_INF, -1e-20]]
    near_one["lambda"], near_one["gamma"] = [1.0], 0.0
    near_one["transition"] = [[[0.0, NEG_INF], [0.0, NEG_INF]]]
    expected_restart_log = log_one_minus_exp(-1e-20)
    assert_close(reference(near_one)["log_output"][0], expected_restart_log)
    near_tensors = tensor_parameters(near_one, torch.float64)
    assert_close(
        float(tensor_log_sequence_loss(near_one, near_tensors, [0]).detach()), -expected_restart_log
    )
    rare_restart = {
        "bytes": [[0, 1, 1]],
        "route": [0.0, NEG_INF],
        "source": [[0.0, 0.0, 0.0]],
        "transition": [[[0.0, NEG_INF, NEG_INF]] * 3],
        "bias": [[0.0, 0.0, 0.0]],
        "m": [[0.0, 0.0, 0.0]],
        "lambda": [1.0],
        "gamma": 0.5,
        "generator": [-1000.0, 0.0, -1000.0],
        "ts": 1.0,
        "tt": 1.0,
        "tr": 1.0,
    }
    prior_restart = accept(reference(rare_restart), 0)
    assert prior_restart["log_m"][0][1] == 0
    assert_close(prior_restart["log_noncontinuation"][0], -1000.0)
    resumed = copy.deepcopy(rare_restart)
    resumed["gamma"] = 0.0
    for field in ("m", "log_m", "log_noncontinuation"):
        resumed[field] = prior_restart[field]
    preserved_restart_log = reference(resumed)["log_output"][0]
    assert_close(preserved_restart_log, -1000.0)
    restart_tensors = tensor_parameters(resumed, torch.float64)
    assert_close(float(tensor_log_sequence_loss(resumed, restart_tensors, [0]).detach()), 1000.0)
    return {
        "posterior_roundoff_witness_repaired": True,
        "initial_repair_log_shift": correction,
        "recursive_closure_steps": 1024,
        "additional_roundoff_repairs": corrections,
        "underflow_cases": extremes,
        "balanced_rare_event_posterior": {
            "log_likelihood_scale": -10000.0,
            "generation": balanced_posterior["r_gen"],
            "copy": balanced_posterior["r"][0][0],
        },
        "near_unit_cursor_restart_log": expected_restart_log,
        "separate_noncontinuation_log_preserves_rare_restart": preserved_restart_log,
        "log_cursor_retains_underflowed_probability": {
            "first_copy_log": float(rare_copy_log.detach()),
            "second_log_likelihood": float(log_out2[1].detach()),
            "sum_first_transition_credit": float(source_credit.sum()),
        },
        "scope": "Roundoff-only copy-mass shrink; low valid mass is never normalized to one. Finite gradients are asserted for the listed profiles, not arbitrary ill-conditioned boundary controls.",
    }


def analytic_partial_checks():
    """Hand-derived partials, independent of autodiff and central FD."""
    import torch

    case = example()
    tensors = tensor_parameters(case, torch.float64)
    m = [torch.tensor(row, dtype=torch.float64) for row in case["m"]]
    output, _, _, _, _, _, _ = tensor_forward(case, tensors, m)
    (-torch.log(output[0])).backward()
    result = reference(case)
    a, generated = probabilities(result["log_a"]), probabilities(result["log_g"])
    fresh = [probabilities(row) for row in result["log_p"]]
    eta = [probabilities(row) for row in result["log_eta"]]
    py = math.exp(result["log_output"][0])
    gamma = case["gamma"]
    masks = [[float(b == 0) for b in ids] for ids in case["bytes"]]
    fresh_y = [
        math.fsum(x * y for x, y in zip(row, mask, strict=True))
        for row, mask in zip(fresh, masks, strict=True)
    ]
    conditional_y = [
        math.fsum(x * y for x, y in zip(row, mask, strict=True))
        for row, mask in zip(eta, masks, strict=True)
    ]
    copy_y = math.fsum(x * y for x, y in zip(a[:-1], conditional_y, strict=True))
    expected = {}
    route_v = [-(1 - gamma) * value / py for value in conditional_y + [generated[0]]]
    average_v = math.fsum(x * y for x, y in zip(a, route_v, strict=True))
    expected["route"] = [x * (y - average_v) / case["tr"] for x, y in zip(a, route_v, strict=True)]
    generator_responsibility = (gamma + (1 - gamma) * a[-1]) * generated[0] / py
    expected["generator"] = [
        generator_responsibility * (value - int(j == 0)) for j, value in enumerate(generated)
    ]
    expected["gamma"] = (copy_y - (1 - a[-1]) * generated[0]) / py
    expected["lambda"] = []
    for i, ids in enumerate(case["bytes"]):
        mass = math.fsum(case["m"][i])
        m_y = math.fsum(x * y for x, y in zip(case["m"][i], masks[i], strict=True))
        expected["lambda"].append(-(1 - gamma) * a[i] * (m_y - mass * fresh_y[i]) / py)
        u = probabilities(log_softmax(case["source"][i], case["ts"]))
        transition = [
            probabilities(
                log_softmax([x + b for x, b in zip(row, case["bias"][i], strict=True)], case["tt"])
            )
            for row in case["transition"][i]
        ]
        downstream = [
            -(1 - gamma) * a[i] * (1 - case["lambda"][i] * mass) * mask / py for mask in masks[i]
        ]
        row_v = [
            math.fsum(x * y for x, y in zip(row, downstream, strict=True)) for row in transition
        ]
        mean_v = math.fsum(x * y for x, y in zip(u, row_v, strict=True))
        expected[f"source{i}"] = [
            weight * (value - mean_v) / case["ts"] for weight, value in zip(u, row_v, strict=True)
        ]
        expected[f"transition{i}"] = [
            [
                u[r] * transition[r][j] * (downstream[j] - row_v[r]) / case["tt"]
                for j in range(len(ids))
            ]
            for r in range(len(ids))
        ]
        expected[f"bias{i}"] = [
            math.fsum(row[j] for row in expected[f"transition{i}"]) for j in range(len(ids))
        ]
    worst = 0.0
    for name, values in expected.items():
        actual = tensors[name].grad
        wanted = torch.tensor(values, dtype=torch.float64)
        error = float((actual - wanted).abs().max())
        worst = max(worst, error)
        assert error < 1e-11, (name, error)
    temperature_gradients = {
        "ts": -sum(
            case["source"][i][j] * expected[f"source{i}"][j]
            for i in range(len(case["bytes"]))
            for j in range(len(case["source"][i]))
        )
        / case["ts"],
        "tt": -sum(
            (value + case["bias"][i][j]) * expected[f"transition{i}"][r][j]
            for i, matrix in enumerate(case["transition"])
            for r, row in enumerate(matrix)
            for j, value in enumerate(row)
        )
        / case["tt"],
        "tr": -sum(x * y for x, y in zip(case["route"], expected["route"], strict=True))
        / case["tr"],
    }
    control_errors = []
    for field, derivative in temperature_gradients.items():
        plus, minus = copy.deepcopy(case), copy.deepcopy(case)
        plus[field] += 1e-6
        minus[field] -= 1e-6
        fd = (python_sequence_loss(plus, [0]) - python_sequence_loss(minus, [0])) / 2e-6
        control_errors.append(abs(fd - derivative))
    cursor_errors = []
    for i, ids in enumerate(case["bytes"]):
        for j in range(1, len(ids)):
            derivative = -(1 - gamma) * a[i] * case["lambda"][i] * (masks[i][j] - fresh_y[i]) / py
            plus, minus = copy.deepcopy(case), copy.deepcopy(case)
            plus["m"][i][j] += 1e-6
            minus["m"][i][j] -= 1e-6
            fd = (python_sequence_loss(plus, [0]) - python_sequence_loss(minus, [0])) / 2e-6
            cursor_errors.append(abs(fd - derivative))
    assert max(control_errors + cursor_errors) < 1e-8
    entropy_logits = torch.tensor([1.5, -0.5, 0.0], dtype=torch.float64, requires_grad=True)
    temperature, coefficient = 0.8, 0.01
    log_p = torch.log_softmax(entropy_logits / temperature, 0)
    p = log_p.exp()
    negative_entropy = coefficient * (p * log_p).sum()
    entropy_grad = torch.autograd.grad(negative_entropy, entropy_logits)[0]
    entropy_formula = coefficient / temperature * p * (log_p - (p * log_p).sum())
    entropy_error = float((entropy_grad - entropy_formula).abs().max().detach())
    assert entropy_error < 1e-12 and float(entropy_grad[0]) > 0
    shared_case = copy.deepcopy(case)
    shared_case["lambda"] = [0.55, 0.55]
    shared_tensors = tensor_parameters(shared_case, torch.float64)
    shared_lambda = torch.tensor(0.55, dtype=torch.float64, requires_grad=True)
    shared_tensors["lambda"] = shared_lambda.expand(2)
    shared_loss = tensor_log_sequence_loss(shared_case, shared_tensors, [0])
    shared_loss.backward()
    plus, minus = copy.deepcopy(shared_case), copy.deepcopy(shared_case)
    plus["lambda"] = [x + 1e-6 for x in plus["lambda"]]
    minus["lambda"] = [x - 1e-6 for x in minus["lambda"]]
    shared_fd = (python_sequence_loss(plus, [0]) - python_sequence_loss(minus, [0])) / 2e-6
    shared_error = abs(shared_fd - float(shared_lambda.grad))
    assert shared_error < 1e-8
    return {
        "input_groups": len(expected),
        "declared_parameter_components": sum(value.numel() for value in tensors.values()),
        "max_abs_error": worst,
        "single_step_temperature_partials_FD_max_error": max(control_errors),
        "initial_cursor_interior_partials": len(cursor_errors),
        "initial_cursor_interior_FD_max_error": max(cursor_errors),
        "negative_entropy_gradient_error": entropy_error,
        "shared_lambda_broadcast_FD_error": shared_error,
        "scope": "Conditional operator inputs. Initial cursor/temperature partials are single-step interior scalar checks; active repairs, support boundaries and full neural content/H chains are not universal gradient claims.",
    }


def gradient_scope_and_st():
    import torch

    case = example()
    # Byte2 is absent after relabelling sources, so direct copy responsibility is0.
    case["bytes"] = [[0, 0, 1], [0, 1]]
    tensors = tensor_parameters(case, torch.float64)
    loss = tensor_sequence_loss(case, tensors, [2], coupled=True)
    loss.backward()
    total = float(tensors["source0"].grad[0])
    assert abs(total) > 1e-5
    result = reference(case)
    a = probabilities(result["log_a"])
    content = math.fsum(
        (j + 1) * math.exp(value) for row in result["log_q"] for j, value in enumerate(row)
    )
    u = probabilities(log_softmax(case["source"][0], case["ts"]))
    transition = [
        probabilities(
            log_softmax([x + b for x, b in zip(row, case["bias"][0], strict=True)], case["tt"])
        )
        for row in case["transition"][0]
    ]
    row_content = [math.fsum((j + 1) * value for j, value in enumerate(row)) for row in transition]
    mean_content = math.fsum(x * y for x, y in zip(u, row_content, strict=True))
    dc = (
        a[0]
        * (1 - case["lambda"][0] * math.fsum(case["m"][0]))
        * u[0]
        * (row_content[0] - mean_content)
        / case["ts"]
    )
    direction = [0.4, -0.2, 0.3, -0.1]
    generated = probabilities(
        log_softmax([x + content * d for x, d in zip(case["generator"], direction, strict=True)])
    )
    raw_gamma = 1 / (1 + math.exp(-(case["gamma"] + 0.3 * content)))
    effective_gamma = 0.2 + 0.8 * raw_gamma
    mass_gen = effective_gamma + (1 - effective_gamma) * a[-1]
    derivative_content = (
        -(direction[2] - math.fsum(x * y for x, y in zip(generated, direction, strict=True)))
        - (1 - a[-1]) / mass_gen * 0.8 * raw_gamma * (1 - raw_gamma) * 0.3
    )
    assert_close(total, dc * derivative_content, 1e-11)
    # Same hard forward values, but ST's VJP differs from soft-forward loss.
    score = torch.tensor([0.2, -0.3, 0.1], dtype=torch.float64, requires_grad=True)
    soft = torch.softmax(score, 0)
    hard = torch.nn.functional.one_hot(torch.argmax(score), 3).to(score.dtype)
    straight = hard + soft - soft.detach()
    coefficients = score.new_tensor([0.8, 0.15, 0.3])
    hard_loss = -torch.log(straight @ coefficients)
    observed = torch.autograd.grad(hard_loss, score, retain_graph=True)[0]
    v = -coefficients / (hard @ coefficients)
    expected = soft * (v - (soft * v).sum())
    assert torch.allclose(observed, expected, atol=1e-12, rtol=0)
    soft_gradient = torch.autograd.grad(-torch.log(soft @ coefficients), score)[0]
    assert float((soft_gradient - observed).abs().max()) > 0.01
    fixed_v = v.detach().tolist()
    scores = score.detach().tolist()
    proxy_errors = []
    hard_differences = []
    for j in range(len(scores)):
        plus, minus = scores.copy(), scores.copy()
        plus[j] += 1e-6
        minus[j] -= 1e-6
        proxy_plus = math.fsum(
            x * y for x, y in zip(fixed_v, probabilities(log_softmax(plus)), strict=True)
        )
        proxy_minus = math.fsum(
            x * y for x, y in zip(fixed_v, probabilities(log_softmax(minus)), strict=True)
        )
        proxy_fd = (proxy_plus - proxy_minus) / 2e-6
        proxy_errors.append(abs(proxy_fd - float(observed[j])))
        plus_winner = max(range(len(plus)), key=plus.__getitem__)
        minus_winner = max(range(len(minus)), key=minus.__getitem__)
        hard_differences.append(
            (
                -math.log(float(coefficients[plus_winner]))
                + math.log(float(coefficients[minus_winner]))
            )
            / 2e-6
        )
        assert plus_winner == minus_winner
    assert max(proxy_errors) < 1e-8
    return {
        "zero_direct_copy_responsibility_nonzero_content_chain_pointer_gradient": total,
        "content_chain_analytic_error": abs(total - dc * derivative_content),
        "ST_local_VJP_max_error": float((observed - expected).abs().max().detach()),
        "ST_surrogate_FD_error": max(proxy_errors),
        "ST_hard_forward_FD": hard_differences,
        "ST_vs_soft_loss_gradient_difference": float((soft_gradient - observed).abs().max()),
        "scope": "ST validates a surrogate VJP at hard values, not a hard-objective derivative or a descent guarantee.",
    }


def actual_byte_alphabet_profile():
    import torch

    case = example()
    case["generator"] = [0.2, -0.3, 0.1] + [-4.0] * 253 + [0.05]
    result = reference(case)
    eos = accept(result, 256)
    assert all(x == 0 for row in eos["m"] for x in row)
    assert_close(eos["r_gen"], 1.0)
    errors = []
    for dtype, tolerance in ((torch.float64, 1e-10), (torch.float32, 2e-6)):
        tensors = tensor_parameters(case, dtype)
        loss = tensor_log_sequence_loss(case, tensors, [0])
        loss.backward()
        responsibility = accept(result, 0)["r_gen"]
        generated = probabilities(result["log_g"])
        expected = torch.tensor(
            [responsibility * (p - int(j == 0)) for j, p in enumerate(generated)], dtype=dtype
        )
        error = float((expected - tensors["generator"].grad).abs().max())
        assert error < tolerance
        assert_close(float(loss.detach()), -result["log_output"][0], tolerance)
        errors.append(
            {"dtype": str(dtype), "generator_components": 257, "analytic_gradient_max_error": error}
        )
        recurrent = tensor_parameters(case, dtype)
        sequence_loss = tensor_log_sequence_loss(case, recurrent, [0, 1, 256])
        sequence_loss.backward()
        assert bool(torch.isfinite(recurrent["generator"].grad).all())
        assert_close(float(sequence_loss.detach()), python_sequence_loss(case, [0, 1, 256]), 2e-5)
    return {
        "vocabulary": 257,
        "EOS": 256,
        "probability_mass_error": abs(math.fsum(probabilities(result["log_output"])) - 1),
        "checks": errors,
        "scope": "Actual byte/EOS alphabet applied to a small conditional operator graph, not a Taiji candidate.",
    }


def runtime_helper_crosscheck():
    """Compile only two small functions; import no model package."""
    import torch

    def extract(path, name):
        text = (ROOT / path).read_text(encoding="utf-8-sig")
        tree = ast.parse(text)
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
        namespace = {
            "torch": torch,
            "BindingRead": object,
            "Any": object,
            "MAX_COVERAGE": 257,
            "EOS": 256,
        }
        exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), namespace)
        return namespace[name], hashlib.sha256(text.encode()).hexdigest()

    locked, locked_sha = extract("taiji/selection_routing.py", "_locked_quantities")
    view = SimpleNamespace(record_count=1, slice=lambda _: slice(0, 2))
    old_outputs = []
    for alpha in (0.1, 0.9):
        read = SimpleNamespace(
            q_bytes=torch.tensor(
                [0.6 * alpha * 0.99, 0.4 + 0.6 * alpha * 0.01], dtype=torch.float64
            ),
            betas=(torch.tensor([0.99, 0.01], dtype=torch.float64),),
        )
        result = locked(
            read, view, torch.tensor([0.5, 0.5]), torch.tensor([1.0, 0.0]), "B_stop_lock"
        )
        old_outputs.append(result[0].tolist())
        COUNTS["runtime_helper_calls"] += 1
    assert old_outputs[0][0] < 0.5 < old_outputs[1][0]
    advance, advance_sha = extract("taiji/binding_copy.py", "advance_copy")
    state = SimpleNamespace(
        m=torch.zeros(2, dtype=torch.float64), coverage=torch.zeros(2, dtype=torch.float64)
    )
    read = SimpleNamespace(copy_weights=torch.tensor([1.0, 0.0], dtype=torch.float64))
    reader = SimpleNamespace(
        inherited=SimpleNamespace(
            V_position=SimpleNamespace(weight=torch.zeros(1, dtype=torch.float64))
        )
    )
    source_view = SimpleNamespace(byte_ids=torch.tensor([0, 1]), record_offsets=((0, 2),))
    output = torch.zeros(257, dtype=torch.float64)
    output[0], output[2] = 0.8, 0.2
    m, _ = advance(
        reader, state, read, output, torch.tensor(0.2, dtype=torch.float64), 0, source_view
    )
    assert torch.allclose(m, torch.tensor([0.0, 1.0], dtype=torch.float64), atol=1e-12, rtol=0)
    wrong, _ = advance(reader, state, read, output, torch.tensor(0.0), 0, source_view)
    assert float(wrong.sum()) > 1
    eos_output = torch.zeros(257, dtype=torch.float64)
    eos_output[0], eos_output[256] = 0.8, 0.2
    eos_state = SimpleNamespace(
        m=torch.tensor([0.0, 0.7], dtype=torch.float64),
        coverage=torch.tensor([0.2, 0.3], dtype=torch.float64),
    )
    eos_m, eos_coverage = advance(
        reader, eos_state, read, eos_output, torch.tensor(0.2), 256, source_view
    )
    assert torch.count_nonzero(eos_state.m) == 1 and torch.count_nonzero(eos_m) == 0
    assert torch.equal(eos_coverage, eos_state.coverage)
    COUNTS["runtime_helper_calls"] += 3
    return {
        "runtime_record_conditional_alpha_flip": old_outputs,
        "same_gamma_posterior_matches": True,
        "wrong_gamma_mass_counterexample": float(wrong.sum()),
        "EOS_helper_clears_cursor": True,
        "reviewed_function_sources": {
            "selection_routing.py": locked_sha,
            "binding_copy.py": advance_sha,
        },
        "scope": "Only extracted helpers called. Generation-loop EOS commit and full model remain outside this check.",
    }


def run_checks():
    checks = {}
    failures = []
    battery = [
        probability_properties,
        exact_latent_enumeration,
        counterexamples,
        invariances_and_limits,
        edge_cases,
        finite_difference_checks,
        analytic_partial_checks,
        underflow_and_roundoff_regressions,
        gradient_scope_and_st,
        actual_byte_alphabet_profile,
        runtime_helper_crosscheck,
    ]
    for check in battery:
        try:
            checks[check.__name__] = {"passed": True, "details": check()}
        except Exception as error:
            checks[check.__name__] = {"passed": False, "error": repr(error)}
            failures.append(check.__name__)
    return {
        "schema": "taiji-copy-operator-op2-math-v1",
        "passed": not failures,
        "failures": failures,
        "checks": checks,
        "accounting": {
            **COUNTS,
            "candidate_models_constructed": 0,
            "optimizer_steps": 0,
            "weights_loaded": False,
        },
        "scope": "Mathematical OP2 proposal and versioned helper checks, not task/model validation or adoption.",
        "adoption_status": "design_only",
        "objective_convention": "Sum token NLL for sequence gradients; model mean-loss scaling must be declared separately.",
        "full_model_validated": False,
        "counter_scope": "This verification process; development reruns are not a project-wide compute ledger.",
        "review_findings": [
            "Removing alpha from the fresh pointer alone does not remove alpha from legacy record conditioning.",
            "Probability-domain tensor loss/recurrence underflow is retained as an expected negative control.",
            "Roundoff can exceed cursor mass one; only tolerance-sized copy mass is shrunk and recorded.",
            "Separate log noncontinuation preserves a rare restart tail when displayed copy mass rounds to one.",
            "Direct responsibility-scaled gradients, full content chains and ST surrogate VJPs have distinct claims.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.report is not None:
        target = args.report.resolve()
        if target.parent != (ROOT / "reports").resolve() or target.exists():
            parser.error("Report must be a new file directly under reports")
    result = run_checks()
    result["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["protocol_date"] = "2026-11-04"
    result["generated_at_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    result["numeric_environment"] = {
        "python": sys.version,
        "torch": getattr(sys.modules.get("torch"), "__version__", None),
        "device": "CPU tensors",
        "random_case_seed": 20261104,
        "closure_seed": 731104,
    }
    result["baseline_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if args.report is not None:
        with args.report.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "failures": result["failures"],
                "checks": {key: value["passed"] for key, value in result["checks"].items()},
                "accounting": result["accounting"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
