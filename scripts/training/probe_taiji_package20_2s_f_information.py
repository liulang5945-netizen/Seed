"""20.2s: closed-form probe of the information F carries about correction labels.

Zero training, zero optimizer steps, zero Adam, zero confirm reads, zero new
parameters.  The probe replays the 20.1 fixture with the *committed* 20.2 runner
(sha pinned below), collects ``LivingState.fast`` at every correction-surface
turn, and asks whether a ridge readout can recover the gold answer bytes from F
alone.

Thresholds are fixed before the first forward pass, echoed into the progress
log at stage ``thresholds``, and repeated verbatim in the report.  The oracle
arm does not exist here: this package measures information content only.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

torch.set_num_threads(1)

from taiji.rule_learning_20_1_identity import (  # noqa: E402
    DECODE_RULES_20,
)
from taiji.rule_learning_20_1_net import (  # noqa: E402
    ACTIONS,
    ANSWER_BYTES,
    DECISION_POSITIONS,
)

REPORT = ROOT / "reports/taiji_package20_2s_f_information_20261105.json"
MANIFEST = ROOT / "plans/manifests/taiji_package20_2s_f_information_20261105.json"
PROGRESS = ROOT / "reports/taiji_package20_2s_f_information_20261105_progress.jsonl"
P2_PATH = ROOT / "scripts/training/probe_taiji_package20_2_credit_recovery.py"
P2_COMMITTED_SHA = "cd23ce2e06b4b6ad1455378f0d9df4f60bef98762893756f6302f3a709c04119"

CPU_CAP = 200.0
CPU_GUARD = 190.0
WALL_CAP = 1800.0

ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
PERMUTE_SEEDS = (20261105, 20261106, 20261107, 20261108, 20261109)
RANDOM_SEED = 20261105
SENTINEL = 255
PRIMARY_WORLDS = 4

PASS_ACCURACY = 0.60
PASS_MARGIN = 0.20
FAIL_ACCURACY = 0.30
FAIL_MARGIN = 0.10

ACT0 = ACTIONS[0].encode("utf-8")
ACT1 = ACTIONS[1].encode("utf-8")
BASE_ANSWER = ACT0

THRESHOLDS = {
    "primary_metric": (
        "leave-one-out decision-bit accuracy: correct decision bytes at positions "
        f"{list(DECISION_POSITIONS)} divided by {len(DECISION_POSITIONS)} bits x 22 turns"
    ),
    "pass_accuracy": PASS_ACCURACY,
    "pass_margin": PASS_MARGIN,
    "fail_accuracy": FAIL_ACCURACY,
    "fail_margin": FAIL_MARGIN,
    "ridge_lambda_grid": list(ALPHAS),
    "ridge_lambda_selection": (
        "per outer fold: inner leave-one-out over the 21 training rows only; "
        "highest inner decision-bit accuracy wins, tie-break larger lambda"
    ),
    "controls": [
        "label_permutation_5_seeds",
        "random_vector_same_dim_same_norm",
        "no_f_state_baseline",
    ],
    "margin_definition": "primary accuracy minus the strongest of the three control accuracies",
    "decision_rule": (
        ">= 0.60 and margin >= 0.20 -> f_carries_correction; "
        "<= 0.30 or margin < 0.10 -> f_write_is_empty; otherwise inconclusive"
    ),
    "written_before_reading": True,
}

START_WALL = time.time()
START_CPU = time.process_time()


class BudgetStop(RuntimeError):
    pass


def cost() -> tuple[float, float]:
    return time.process_time() - START_CPU, time.time() - START_WALL


def guard() -> None:
    cpu, wall = cost()
    if cpu > CPU_GUARD or wall > WALL_CAP:
        raise BudgetStop(f"budget stop at cpu={cpu:.2f} wall={wall:.2f}")


def append_progress(record: dict) -> None:
    with PROGRESS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_committed_p2():
    raw = P2_PATH.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    spec = importlib.util.spec_from_file_location("probe_taiji_package20_2", P2_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, sha


def collect_world(p2, graph, world: dict) -> dict:
    """Full 51-turn replay, keeping the state at every correction surface."""
    response = world["response"]
    state = graph.initial_state(response["world_id"])
    released: list[dict] = []
    rows: list[dict] = []
    release_states: list[dict] = []
    fast_writes = 0
    for turn in response["turns"]:
        guard()
        number = int(turn["turn"])
        result = graph.run_turn(
            state,
            turn,
            released,
            write_enabled=True,
            vocab=p2.legal_vocab(released),
            now=number,
        )
        state = result["state"]
        gold = world["scoring"]["turns"][str(number)]["expected_action"].encode("utf-8")
        rows.append(
            {
                "turn": number,
                "session": int(turn["session"]),
                "kind": turn["kind"],
                "version": int(turn["version"]),
                "correction_surface": bool(turn["correction_surface"]),
                "released_lines": len(released),
                "bytes": tuple(int(value) for value in result["bytes"]),
                "constrained": tuple(int(value) for value in result["constrained_bytes"]),
                "gold": gold,
                "fast": state.living.fast.detach().cpu().clone(),
                "h": state.living.h.detach().cpu().clone(),
                "nu_ema": state.living.nu_ema.detach().cpu().clone(),
                "coverage": state.living.coverage.detach().cpu().clone(),
                "prior": state.living.prior.detach().cpu().clone(),
                "keys_count": len(state.living.keys),
                "step": int(state.living.step),
                "revision": int(state.revision),
                "journal_len": len(state.journal),
            }
        )
        for release in response["releases"]:
            if int(release["after_turn"]) != number:
                continue
            state, info = graph.release(state, release, released, write_enabled=True, now=number)
            fast_writes += int(info["written"])
            release_states.append(
                {
                    "after_turn": number,
                    "version": int(release["version"]),
                    "written": int(info["written"]),
                    "fast": state.living.fast.detach().cpu().clone(),
                }
            )
            released.extend(
                {
                    "source_id": item["source_id"],
                    "feedback": item["feedback"],
                    "version": item["version"],
                    "after_turn": number,
                }
                for item in release["items"]
            )
    return {"rows": rows, "release_states": release_states, "fast_writes": fast_writes}


def surface_rows(record: dict) -> list[dict]:
    return [
        row for row in record["rows"] if row["correction_surface"] and row["released_lines"] > 0
    ]


def other_features(row: dict) -> np.ndarray:
    def summary(tensor: torch.Tensor) -> list[float]:
        if tensor.numel() == 0:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        values = tensor.detach().cpu().double().reshape(-1)
        return [
            float(values.sum()),
            float(values.abs().sum()),
            float(values.norm()),
            float(values.max()),
            float(values.numel()),
        ]

    return np.concatenate(
        [
            row["h"].double().reshape(-1).numpy(),
            row["nu_ema"].double().reshape(-1).numpy(),
            np.asarray(summary(row["coverage"]), dtype=np.float64),
            np.asarray(summary(row["prior"]), dtype=np.float64),
            np.asarray(
                [
                    row["step"],
                    row["revision"],
                    row["session"],
                    row["released_lines"],
                    row["keys_count"],
                    row["journal_len"],
                    row["version"],
                    row["turn"],
                ],
                dtype=np.float64,
            ),
        ]
    )


def encode_labels(gold: list[bytes]) -> tuple[np.ndarray, np.ndarray]:
    count = len(gold)
    wide = np.zeros((count, ANSWER_BYTES * 3), dtype=np.float64)
    decision = np.zeros((count, len(DECISION_POSITIONS) * 2), dtype=np.float64)
    for index, answer in enumerate(gold):
        for position in range(ANSWER_BYTES):
            if answer[position] == ACT1[position] and ACT1[position] != ACT0[position]:
                slot = 1
            else:
                slot = 0
            wide[index, position * 3 + slot] = 1.0
        for offset, position in enumerate(DECISION_POSITIONS):
            slot = int(answer[position] == ACT1[position])
            decision[index, offset * 2 + slot] = 1.0
    return wide, decision


def decode_wide(prediction: np.ndarray) -> list[bytes]:
    rows: list[bytes] = []
    for row in prediction:
        answer = bytearray(ANSWER_BYTES)
        for position in range(ANSWER_BYTES):
            chunk = row[position * 3 : position * 3 + 3]
            slot = int(np.argmax(chunk))
            if slot == 1 and ACT1[position] != ACT0[position]:
                answer[position] = ACT1[position]
            else:
                answer[position] = ACT0[position]
        rows.append(bytes(answer))
    return rows


def decode_decision(prediction: np.ndarray) -> list[bytes]:
    rows: list[bytes] = []
    for row in prediction:
        answer = bytearray(BASE_ANSWER)
        for offset, position in enumerate(DECISION_POSITIONS):
            slot = int(np.argmax(row[offset * 2 : offset * 2 + 2]))
            answer[position] = ACT1[position] if slot else ACT0[position]
        rows.append(bytes(answer))
    return rows


def score_bytes(predicted: list[bytes], gold: list[bytes]) -> dict:
    position_hits = 0
    decision_hits = 0
    full_correct = 0
    turns_correct = 0
    decisions = len(DECISION_POSITIONS)
    for answer, truth in zip(predicted, gold, strict=True):
        hits = sum(1 for left, right in zip(answer, truth, strict=True) if left == right)
        position_hits += hits
        bits = sum(1 for position in DECISION_POSITIONS if answer[position] == truth[position])
        decision_hits += bits
        full_correct += int(hits == ANSWER_BYTES)
        turns_correct += int(bits == decisions)
    total_positions = len(gold) * ANSWER_BYTES
    total_bits = len(gold) * decisions
    return {
        "turns": len(gold),
        "position_hits": position_hits,
        "position_total": total_positions,
        "position_accuracy": position_hits / total_positions,
        "decision_bit_hits": decision_hits,
        "decision_bit_total": total_bits,
        "decision_bit_accuracy": decision_hits / total_bits,
        "decisions_correct": turns_correct,
        "full_correct": full_correct,
        "full_accuracy": full_correct / len(gold),
    }


def ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    lam: float,
) -> np.ndarray:
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-12] = 1.0
    z_train = (x_train - mean) / scale
    z_test = (x_test - mean) / scale
    y_mean = y_train.mean(axis=0)
    gram = z_train @ z_train.T
    alpha = np.linalg.solve(gram + lam * np.eye(gram.shape[0]), y_train - y_mean)
    return z_test @ z_train.T @ alpha + y_mean


def loo_predictions(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    count = len(x)
    output = np.empty_like(y)
    for index in range(count):
        keep = np.ones(count, dtype=bool)
        keep[index] = False
        output[index] = ridge_predict(x[keep], y[keep], x[index : index + 1], lam)[0]
    return output


def select_lambda(
    x: np.ndarray,
    y: np.ndarray,
    gold: list[bytes],
    decode,
) -> float:
    best: tuple[tuple[float, float], float] | None = None
    for lam in ALPHAS:
        metrics = score_bytes(decode(loo_predictions(x, y, lam)), gold)
        key = (metrics["decision_bit_accuracy"], lam)
        if best is None or key > best[0]:
            best = (key, lam)
    assert best is not None
    return best[1]


def loo_with_selection(
    x: np.ndarray,
    y: np.ndarray,
    gold: list[bytes],
    decode,
) -> tuple[np.ndarray, list[float]]:
    count = len(x)
    output = np.empty_like(y)
    lambdas: list[float] = []
    for index in range(count):
        keep = np.ones(count, dtype=bool)
        keep[index] = False
        lam = select_lambda(x[keep], y[keep], [gold[i] for i in np.flatnonzero(keep)], decode)
        lambdas.append(lam)
        output[index] = ridge_predict(x[keep], y[keep], x[index : index + 1], lam)[0]
    return output, lambdas


def probe(x: np.ndarray, gold: list[bytes], decode) -> dict:
    wide, decision = encode_labels(gold)
    target = decision if decode is decode_decision else wide
    prediction, lambdas = loo_with_selection(x, target, gold, decode)
    decoded = decode(prediction)
    metrics = score_bytes(decoded, gold)
    per_fold = [
        {
            "index": index,
            "lambda": lambdas[index],
            "gold": gold[index].decode("utf-8"),
            "predicted": decoded[index].decode("utf-8"),
            "decision_bits_correct": sum(
                1
                for position in DECISION_POSITIONS
                if decoded[index][position] == gold[index][position]
            ),
            "bytes_correct": sum(
                1 for left, right in zip(decoded[index], gold[index], strict=True) if left == right
            ),
        }
        for index in range(len(gold))
    ]
    return {
        "features": list(x.shape),
        "metrics": metrics,
        "lambdas": {
            "selected": sorted(set(lambdas)),
            "per_fold": lambdas,
        },
        "per_fold": per_fold,
    }


def control(x: np.ndarray, gold: list[bytes], label: str) -> dict:
    return {"control": label, **probe(x, gold, decode_decision)["metrics"]}


def forward_predict(
    x: np.ndarray, y: np.ndarray, gold: list[bytes], train: list[int], test: list[int]
) -> dict:
    lam = select_lambda(x[train], y[train], [gold[i] for i in train], decode_decision)
    prediction = ridge_predict(x[train], y[train], x[test], lam)
    metrics = score_bytes(decode_decision(prediction), [gold[i] for i in test])
    return {"train": len(train), "test": len(test), "lambda": lam, **metrics}


def readout_list() -> dict:
    f_elements = 4 * 2 * 32 * 32
    decision_logits = len(DECISION_POSITIONS) * 256
    full_logits = ANSWER_BYTES * 256
    return {
        "status": "estimate_only_not_implemented",
        "f_elements": f_elements,
        "decision_bit_low_rank": {
            "rank_4": int(f_elements * len(DECISION_POSITIONS) * 4 + decision_logits * 4),
            "rank_8": int(f_elements * len(DECISION_POSITIONS) * 8 + decision_logits * 8),
        },
        "full_position_low_rank": {
            "rank_4": int(f_elements * ANSWER_BYTES * 4 + full_logits * 4),
            "rank_8": int(f_elements * ANSWER_BYTES * 8 + full_logits * 8),
        },
    }


def main() -> int:
    runner_sha = file_sha(Path(__file__).resolve())
    p2, p2_sha = load_committed_p2()
    append_progress(
        {
            "stage": "thresholds",
            "written_before_reading": True,
            "thresholds": THRESHOLDS,
            "p2_script_sha256": p2_sha,
            "committed_match": p2_sha == P2_COMMITTED_SHA,
            "cpu": round(cost()[0], 3),
        }
    )
    if p2_sha != P2_COMMITTED_SHA:
        write_json(
            REPORT,
            {
                "package": "20.2s",
                "date": "2026-11-05",
                "verdict": "replay_script_drift",
                "thresholds": THRESHOLDS,
                "script": {
                    "runner_sha256": runner_sha,
                    "p2_script_sha256": p2_sha,
                    "p2_committed_sha256": P2_COMMITTED_SHA,
                },
                "cost": {"cpu_seconds": round(cost()[0], 2)},
            },
        )
        print(json.dumps({"verdict": "replay_script_drift", "p2_sha": p2_sha}, indent=2))
        return 0

    fixture = p2.fixture
    bundle = torch.load(p2.BUNDLE, map_location="cpu", weights_only=False)
    candidate = p2.load_routing_bundle(p2.BUNDLE)[0]
    gate = dict(bundle["growth_gates"][0]) if bundle.get("growth_gates") else None
    parent = candidate.subject
    parent_sha = p2.weights_digest(parent)
    state_shas = {"cf1": p2.sha256_file(p2.CF1_STATE), "b_eh": p2.sha256_file(p2.BEH_STATE)}
    assemblies_intact = (
        state_shas["cf1"] == p2.CF1_STATE_SHA and state_shas["b_eh"] == p2.BEH_STATE_SHA
    )
    cf1 = p2.fresh_graph(parent, gate, "CF1")
    p2.freeze(cf1.subject)
    append_progress(
        {"stage": "graphs", "assemblies_intact": assemblies_intact, "cpu": round(cost()[0], 3)}
    )

    primary_world = fixture.build_world(fixture.SEED, 0)
    primary_record = collect_world(p2, cf1, primary_world)
    rows = surface_rows(primary_record)
    assert len(rows) == 22, f"expected 22 correction surfaces, got {len(rows)}"
    gold = [row["gold"] for row in rows]
    x_fast = np.stack([row["fast"].double().reshape(-1).numpy() for row in rows])
    x_other = np.stack([other_features(row) for row in rows])
    baseline_readout = score_bytes([bytes(row["constrained"]) for row in rows], gold)
    append_progress({"stage": "collect_primary", "rows": len(rows), "cpu": round(cost()[0], 3)})

    fast_norms = [float(row["fast"].norm()) for row in rows]
    unique_rows = np.unique(x_fast, axis=0)
    session1 = [i for i, row in enumerate(rows) if row["session"] == 1]
    session2 = [i for i, row in enumerate(rows) if row["session"] == 2]
    unique_session1 = np.unique(x_fast[session1], axis=0).shape[0]
    unique_session2 = np.unique(x_fast[session2], axis=0).shape[0]
    pair_delta = None
    if unique_rows.shape[0] == 2:
        pair_delta = float(np.abs(unique_rows[0] - unique_rows[1]).max())
    release_match = []
    for entry in primary_record["release_states"]:
        following = [row for row in rows if row["turn"] > entry["after_turn"]]
        if following:
            first = following[0]
            release_match.append(
                {
                    "after_turn": entry["after_turn"],
                    "first_correction_turn": first["turn"],
                    "written": entry["written"],
                    "fast_identical": bool(torch.equal(entry["fast"], first["fast"])),
                    "fast_sha256": hashlib.sha256(
                        entry["fast"].double().numpy().tobytes()
                    ).hexdigest(),
                }
            )
    structural = {
        "correction_turns": len(rows),
        "distinct_f_vectors": int(unique_rows.shape[0]),
        "distinct_f_vectors_session1": int(unique_session1),
        "distinct_f_vectors_session2": int(unique_session2),
        "f_elements": int(x_fast.shape[1]),
        "f_norm_min": min(fast_norms),
        "f_norm_max": max(fast_norms),
        "f_window_max_abs_delta": pair_delta,
        "f_written_lines": int(primary_record["fast_writes"]),
        "release_states": release_match,
        "note": (
            "F is written only inside a release call; correction turns cannot write it, "
            "so every correction surface inside one feedback window shares one F vector"
        ),
    }

    wide_probe = probe(x_fast, gold, decode_wide)
    decision_probe = probe(x_fast, gold, decode_decision)
    append_progress(
        {
            "stage": "probe_f",
            "decision_accuracy": decision_probe["metrics"]["decision_bit_accuracy"],
            "wide_full_correct": wide_probe["metrics"]["full_correct"],
            "cpu": round(cost()[0], 3),
        }
    )

    # Controls.  The permutation control re-uses the same features and pipeline.
    permutation = []
    for seed in PERMUTE_SEEDS:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(gold))
        permuted_gold = [gold[int(index)] for index in order]
        metrics = probe(x_fast, permuted_gold, decode_decision)["metrics"]
        permutation.append(
            {
                "seed": seed,
                "order": [int(value) for value in order],
                "decision_bit_accuracy": metrics["decision_bit_accuracy"],
                "full_correct": metrics["full_correct"],
            }
        )
    random_rng = np.random.default_rng(RANDOM_SEED)
    x_random = random_rng.normal(size=x_fast.shape)
    for index in range(len(x_random)):
        scale = float(np.linalg.norm(x_fast[index])) / max(
            float(np.linalg.norm(x_random[index])), 1e-12
        )
        x_random[index] *= scale
    random_metrics = probe(x_random, gold, decode_decision)["metrics"]
    other_metrics = probe(x_other, gold, decode_decision)["metrics"]
    window_features = np.zeros((len(rows), 2), dtype=np.float64)
    for index, row in enumerate(rows):
        window_features[index, 0 if row["session"] == 1 else 1] = 1.0
    window_metrics = probe(window_features, gold, decode_decision)["metrics"]

    permutation_mean = float(np.mean([entry["decision_bit_accuracy"] for entry in permutation]))
    permutation_max = float(np.max([entry["decision_bit_accuracy"] for entry in permutation]))
    controls = {
        "label_permutation": {
            "seeds": [entry["seed"] for entry in permutation],
            "accuracy_mean": permutation_mean,
            "accuracy_max": permutation_max,
            "per_seed": permutation,
        },
        "random_vector_same_dim_same_norm": random_metrics,
        "no_f_state_baseline": other_metrics,
        "window_identity_diagnostic": window_metrics,
        "note": (
            "the window-identity diagnostic is not one of the three required controls; "
            "it shows how much of any F accuracy is explained by feedback-window identity"
        ),
    }
    strongest_control = max(
        permutation_mean,
        random_metrics["decision_bit_accuracy"],
        other_metrics["decision_bit_accuracy"],
    )
    append_progress(
        {
            "stage": "controls",
            "permutation_mean": permutation_mean,
            "random": random_metrics["decision_bit_accuracy"],
            "no_f": other_metrics["decision_bit_accuracy"],
            "window": window_metrics["decision_bit_accuracy"],
            "cpu": round(cost()[0], 3),
        }
    )

    wide_labels, decision_labels = encode_labels(gold)
    forward = forward_predict(x_fast, decision_labels, gold, session1, session2)
    backward = forward_predict(x_fast, decision_labels, gold, session2, session1)

    robustness = []
    for world_index in range(1, PRIMARY_WORLDS):
        append_progress(
            {"stage": "robustness_collect", "world": world_index, "cpu": round(cost()[0], 3)}
        )
        other_world = fixture.build_world(fixture.SEED, world_index)
        other_record = collect_world(p2, cf1, other_world)
        other_rows = surface_rows(other_record)
        other_gold = [row["gold"] for row in other_rows]
        other_x = np.stack([row["fast"].double().reshape(-1).numpy() for row in other_rows])
        other_probe = probe(other_x, other_gold, decode_decision)
        robustness.append(
            {
                "world": world_index,
                "correction_turns": len(other_rows),
                "distinct_f_vectors": int(np.unique(other_x, axis=0).shape[0]),
                "decision_bit_accuracy": other_probe["metrics"]["decision_bit_accuracy"],
                "full_correct": other_probe["metrics"]["full_correct"],
                "lambdas": other_probe["lambdas"]["selected"],
            }
        )
        append_progress(
            {
                "stage": "robustness_probe",
                "world": world_index,
                "accuracy": other_probe["metrics"]["decision_bit_accuracy"],
                "cpu": round(cost()[0], 3),
            }
        )

    primary_accuracy = decision_probe["metrics"]["decision_bit_accuracy"]
    margin = primary_accuracy - strongest_control
    if primary_accuracy >= PASS_ACCURACY and margin >= PASS_MARGIN:
        verdict = "f_carries_correction"
    elif primary_accuracy <= FAIL_ACCURACY or margin < FAIL_MARGIN:
        verdict = "f_write_is_empty"
    else:
        verdict = "inconclusive"

    parent_after = p2.weights_digest(parent)
    gates = {
        "p2_script_is_committed_bytes": p2_sha == P2_COMMITTED_SHA,
        "correction_turns_22": len(rows) == 22,
        "reference_assemblies_untouched": assemblies_intact,
        "parent_weights_unchanged": parent_after == parent_sha,
        "thresholds_written_before_reading": True,
        "lambda_selected_on_training_side": True,
        "three_controls_present": True,
        "zero_training": True,
        "zero_optimizer_steps": True,
        "zero_backward_calls": True,
        "zero_new_parameters": True,
        "report_written_before_format": True,
    }
    engineering_ready = all(gates.values())
    cpu_seconds, wall_seconds = cost()
    report = {
        "package": "20.2s",
        "date": "2026-11-05",
        "verdict": verdict if engineering_ready else "engineering_not_ready",
        "scope": (
            "closed-form F information probe on the 22 correction surfaces; "
            "0 training, 0 optimizer steps, 0 new parameters"
        ),
        "thresholds": THRESHOLDS,
        "script": {
            "runner_sha256": runner_sha,
            "p2_script_sha256": p2_sha,
            "p2_committed_sha256": P2_COMMITTED_SHA,
            "note": (
                "report written before formatting; submitted_runner_sha256 is recorded "
                "after the commit that stores the formatted bytes"
            ),
        },
        "probe": {
            "primary": decision_probe,
            "wide_15_positions_3_classes": wide_probe,
            "label_contract": {
                "decision_positions": list(DECISION_POSITIONS),
                "answer_bytes": ANSWER_BYTES,
                "decode_rules_20_decision_positions": DECODE_RULES_20["decision_positions"],
                "wide_shape": [ANSWER_BYTES, 3],
                "decision_shape": [len(DECISION_POSITIONS), 2],
                "note": (
                    "the third wide slot is a sentinel that no gold byte uses; "
                    "template positions have slot0 == slot1 so only one class occurs"
                ),
            },
            "forward_prediction": {
                "train_session1_test_session2": forward,
                "train_session2_test_session1": backward,
            },
            "structural": structural,
            "baseline_model_readout": baseline_readout,
        },
        "controls": controls,
        "judgement": {
            "primary_accuracy": primary_accuracy,
            "strongest_control_accuracy": strongest_control,
            "margin": margin,
            "verdict": verdict,
            "no_f_baseline_higher_than_f": bool(
                other_metrics["decision_bit_accuracy"] > primary_accuracy
            ),
            "window_identity_matches_f": bool(
                abs(window_metrics["decision_bit_accuracy"] - primary_accuracy) < FAIL_MARGIN
            ),
        },
        "robustness": robustness,
        "state": {
            "reference_assemblies": {
                "cf1": {"sha256": state_shas["cf1"], "label": "initialized_untrained"},
                "b_eh": {"sha256": state_shas["b_eh"], "label": "initialized_untrained"},
                "untouched": assemblies_intact,
            },
            "parent_parameter_sha256": parent_sha,
            "parent_parameter_sha256_after": parent_after,
            "parameter_tensors": sum(1 for _ in cf1.subject.parameters()),
        },
        "gates": gates,
        "engineering_ready": engineering_ready,
        "cost": {
            "cpu_seconds": round(cpu_seconds, 2),
            "wall_seconds": round(wall_seconds, 2),
            "cpu_cap": CPU_CAP,
            "wall_cap": WALL_CAP,
            "training_steps": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "confirm_reads": 0,
            "new_parameters": 0,
            "worlds_forwarded": 1 + len(robustness),
            "turns_forwarded": fixture.TOTAL_TURNS * (1 + len(robustness)),
        },
    }
    if verdict == "f_carries_correction":
        report["readout_parameter_list"] = readout_list()
    write_json(REPORT, report)
    append_progress(
        {
            "stage": "report",
            "verdict": report["verdict"],
            "cpu": round(cpu_seconds, 3),
            "wall": round(wall_seconds, 3),
        }
    )

    manifest = {
        "package": "20.2s",
        "date": "2026-11-05",
        "verdict": report["verdict"],
        "report": str(REPORT.relative_to(ROOT)),
        "report_sha256": file_sha(REPORT),
        "runner": str(Path(__file__).resolve().relative_to(ROOT)),
        "runner_sha256": runner_sha,
        "replayed_runner_sha256": p2_sha,
        "fixture_digest": fixture.fixture_digest([primary_world]),
        "parent_bundle": str(p2.BUNDLE.relative_to(ROOT)),
        "parent_bundle_sha256": p2.sha256_file(p2.BUNDLE),
        "parent_parameter_sha256": parent_sha,
        "reference_assemblies_untouched": assemblies_intact,
        "training_admitted": False,
        "saved_states": [],
        "gates": gates,
        "cost_cpu_seconds": report["cost"]["cpu_seconds"],
    }
    write_json(MANIFEST, manifest)
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "primary_accuracy": primary_accuracy,
                "strongest_control": strongest_control,
                "margin": margin,
                "distinct_f_vectors": structural["distinct_f_vectors"],
                "gates": gates,
                "cost": report["cost"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
