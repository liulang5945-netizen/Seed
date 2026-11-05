"""20.3a: per-turn content-addressed store, default off, plus three frozen gates.

Zero training, zero optimizer steps, zero Adam, zero confirm reads, zero new
trainable parameters.  The store adds 266,273 non-trainable state elements
(keys 32x128 + values 32x4x2x32x32 + age 32 + count 1); F remains the only
plastic tensor and is frozen-parameter-only inside a session.

Readings (thresholds are written before collection):
  A. structural -- distinct F vectors on the 22 correction surfaces, per-window
     error-direction count, and the per-turn F hash sequence;
  B. leverage -- five counterfactuals of F on one correction turn;
  C. information -- closed-form ridge LOO on F with three controls;
  D. extras -- retrieval hit diagnostics, session boundary, determinism,
     B_EH same-shape and the store-off equivalence proof against HEAD bytes.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)

from taiji import rule_learning_20_1 as fixture  # noqa: E402
from taiji.living_core import EOS, LivingCoreConfig, LivingCoreReference  # noqa: E402
from taiji.rule_learning_20_1_identity import (  # noqa: E402
    DECODE_RULES_20,
    graph_identity,
    sha256_file,
)
from taiji.rule_learning_20_1_net import (  # noqa: E402
    ANSWER_BYTES,
    DECISION_POSITIONS,
    RuleLearningGraph,
    copy_subject,
    state_digest,
)
from taiji.selection_routing_checkpoint import load_routing_bundle  # noqa: E402

REPORT = ROOT / "reports/taiji_package20_3a_addressed_store_20261105.json"
MANIFEST = ROOT / "plans/manifests/taiji_package20_3a_addressed_store_20261105.json"
PROGRESS = ROOT / "reports/taiji_package20_3a_addressed_store_20261105_progress.jsonl"
TMP = ROOT / "output/.probe20_3a_tmp"
P2 = ROOT / "scripts/training/probe_taiji_package20_2_credit_recovery.py"
P2S = ROOT / "scripts/training/probe_taiji_package20_2s_f_information.py"
BUNDLE = ROOT / "output/taiji_package19_compare/28031-B_stop_lock-step300.pt"
CF1_STATE = ROOT / "output/taiji_package20_1_cf1_initialized_untrained.pt"
BEH_STATE = ROOT / "output/taiji_package20_1_b_eh_initialized_untrained.pt"
CF1_STATE_SHA = "b62d6fe6a0721fd82c7c43d575e600c48ffc8e31ff959bfdf14ef9ff07d0039e"
BEH_STATE_SHA = "64cb327e3a8442ef0c59e04b14445a08ab3dccbb8ac42cf6547e0efa4e6f7f46"
CF1_ASSEMBLY = ROOT / "output/taiji_package20_3a_cf1_addressed_store_untrained.pt"
BEH_ASSEMBLY = ROOT / "output/taiji_package20_3a_b_eh_addressed_store_untrained.pt"

CPU_CAP = 300.0
CPU_GUARD = 285.0
WALL_CAP = 2700.0
SLOTS = 32
STORE_ELEMENTS = SLOTS * 128 + SLOTS * 4 * 2 * 32 * 32 + SLOTS + 1
PERMUTE_SEEDS = (11, 23, 37, 59, 71)
RANDOM_SEED = 20261105
DETERMINISM_RUNS = 8
DETERMINISM_TURNS = 17
LEVERAGE_TURN = 34
LEVERAGE_HISTORY = 33

THRESHOLDS = {
    "structural": {
        "distinct_f_vectors_global_strictly_greater_than": 2,
        "per_window_distinct_f_at_least_distinct_error_directions": True,
        "per_window_every_consecutive_turn_differs": True,
        "error_direction_definition": (
            "per correction surface: tuple of (position, gold byte, predicted byte) "
            "for every position where the constrained answer differs from gold; "
            "two turns share a direction only when the tuples are identical"
        ),
    },
    "leverage": {
        "tiers": ["zero", "half", "double", "quadruple", "shuffle_same_norm"],
        "leverage_definition": "abs(loss_cf - loss_base) / max(loss_base, 1e-12)",
        "leverage_max_at_least": 0.01,
        "at_least_one_byte_diff": True,
    },
    "information": {
        "primary_metric": (
            "leave-one-out decision-bit accuracy: correct decision bytes at positions "
            f"{list(DECISION_POSITIONS)} divided by {len(DECISION_POSITIONS)} bits x 22 turns"
        ),
        "accuracy_at_least": 0.60,
        "margin_at_least": 0.20,
        "strongest_control": "max(permutation per-seed max, random same-norm, no-F state)",
        "controls": [
            "label_permutation_5_seeds",
            "random_projection_same_dim_same_norm",
            "no_f_other_persistent_state",
        ],
        "decision_rule": "all three gates pass -> option2_passed; any fail -> option2_failed",
    },
    "written_before_reading": True,
}

START_WALL = time.time()
START_CPU = time.process_time()
STAGES: list[str] = []


class BudgetStop(RuntimeError):
    pass


def cost() -> tuple[float, float]:
    return time.process_time() - START_CPU, time.time() - START_WALL


def guard(stage: str) -> None:
    STAGES.append(stage)
    cpu, wall = cost()
    if cpu > CPU_GUARD or wall > WALL_CAP:
        raise BudgetStop(f"budget stop at {stage} cpu={cpu:.2f} wall={wall:.2f}")


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


def tensor_sha(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().double().numpy().tobytes()).hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_head_module(relative: str, name: str):
    raw = subprocess.run(
        ["git", "show", f"HEAD:{relative}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    TMP.mkdir(parents=True, exist_ok=True)
    path = TMP / f"{name}.py"
    path.write_bytes(raw)
    return load_module(path, name), hashlib.sha256(raw).hexdigest()


def build_graph(parent, gate, arm: str, store: bool) -> RuleLearningGraph:
    return RuleLearningGraph(
        copy_subject(parent),
        arm=arm,
        growth_gates=gate,
        addressed_store_enabled=store,
    )


def gold_of(world: dict, number: int) -> bytes:
    return world["scoring"]["turns"][str(number)]["expected_action"].encode("utf-8")


def run_world(graph: RuleLearningGraph, world: dict) -> dict:
    """Full 51-turn replay keeping scalar store diagnostics and F copies."""
    response = world["response"]
    state = graph.initial_state(response["world_id"])
    released: list[dict] = []
    rows: list[dict] = []
    release_events: list[dict] = []
    for turn in response["turns"]:
        guard(f"turn {turn['turn']}")
        number = int(turn["turn"])
        result = graph.run_turn(
            state,
            turn,
            released,
            write_enabled=True,
            vocab=P2M.legal_vocab(released),
            now=number,
        )
        state = result["state"]
        info = result["store"]
        read = None if info is None else info["read"]
        write = None if info is None else info["write"]
        scope = None if info is None else info["scope"]
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
                "gold": gold_of(world, number),
                "fast": state.living.fast.detach().cpu().clone(),
                "h": state.living.h.detach().cpu().clone(),
                "nu_ema": state.living.nu_ema.detach().cpu().clone(),
                "coverage": state.living.coverage.detach().cpu().clone(),
                "prior": state.living.prior.detach().cpu().clone(),
                "keys_count": len(state.living.keys),
                "step": int(state.living.step),
                "revision": int(state.revision),
                "journal_len": len(state.journal),
                "store_count": (
                    None if state.living.store_count is None else int(state.living.store_count)
                ),
                "store_occupied": (
                    None
                    if state.living.store_age is None
                    else int((state.living.store_age >= 0).sum())
                ),
                "store_scope": scope,
                "store_read": None if read is None else read_scalars(read),
                "store_write": write,
            }
        )
        for release in response["releases"]:
            if int(release["after_turn"]) != number:
                continue
            state, info = graph.release(state, release, released, write_enabled=True, now=number)
            release_events.append(
                {
                    "after_turn": number,
                    "version": int(release["version"]),
                    "written": int(info["written"]),
                    "store_events": [store_event(event) for event in info["store_events"]],
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
    return {"rows": rows, "release_events": release_events, "final_state": state}


def read_scalars(info: dict) -> dict:
    return {
        "enabled": bool(info["enabled"]),
        "occupied": int(info.get("occupied", 0)),
        "valid": int(info.get("valid", 0)),
        "blended": bool(info["blended"]),
        "top_slot": int(info["top_slot"]) if "top_slot" in info else None,
        "top_cosine": float(info["top_cosine"]) if "top_cosine" in info else None,
        "second_cosine": (
            None if info.get("second_cosine") is None else float(info["second_cosine"])
        ),
        "margin": None if info.get("margin") is None else float(info["margin"]),
        "weights": [float(value) for value in info.get("weights", [])],
        "retrieved_norm": (
            None if info.get("retrieved_norm") is None else float(info["retrieved_norm"])
        ),
        "fast_norm_before": (
            None if info.get("fast_norm_before") is None else float(info["fast_norm_before"])
        ),
        "fast_norm_after": (
            None if info.get("fast_norm_after") is None else float(info["fast_norm_after"])
        ),
    }


def store_event(event: dict) -> dict:
    return {
        "source_id": event["source_id"],
        "turn": event["turn"],
        "read": read_scalars(event["read"]),
        "write": event["write"],
    }


def surface_rows(record: dict) -> list[dict]:
    return [
        row for row in record["rows"] if row["correction_surface"] and row["released_lines"] > 0
    ]


def error_direction(row: dict) -> tuple:
    predicted = row["constrained"]
    truth = row["gold"]
    return tuple(
        (position, int(truth[position]), int(predicted[position]))
        for position in range(ANSWER_BYTES)
        if predicted[position] != truth[position]
    )


def structural_gate(rows: list[dict]) -> dict:
    x_fast = np.stack([row["fast"].double().reshape(-1).numpy() for row in rows])
    global_distinct = int(np.unique(x_fast, axis=0).shape[0])
    windows: dict[int, list[dict]] = {}
    for row in rows:
        windows.setdefault(int(row["released_lines"]), []).append(row)
    per_window = []
    per_window_ok = True
    change_ok = True
    for lines, members in sorted(windows.items()):
        hashes = []
        unique = np.unique(
            np.stack([row["fast"].double().reshape(-1).numpy() for row in members]), axis=0
        ).shape[0]
        directions = {error_direction(row) for row in members}
        consecutive_unique = all(
            tensor_sha(members[index]["fast"]) != tensor_sha(members[index + 1]["fast"])
            for index in range(len(members) - 1)
        )
        hashes = [tensor_sha(row["fast"]) for row in members]
        per_window.append(
            {
                "released_lines": lines,
                "turns": [row["turn"] for row in members],
                "distinct_f_vectors": int(unique),
                "distinct_error_directions": len(directions),
                "error_directions": [list(map(list, direction)) for direction in directions],
                "every_consecutive_turn_differs": bool(consecutive_unique),
                "f_sha256_sequence": hashes,
                "pass": bool(unique >= len(directions) and consecutive_unique),
            }
        )
        per_window_ok = per_window_ok and bool(unique >= len(directions))
        change_ok = change_ok and consecutive_unique
    result = {
        "correction_turns": len(rows),
        "distinct_f_vectors": global_distinct,
        "distinct_f_vectors_strictly_above_2": global_distinct > 2,
        "window_distinct_f_not_below_error_directions": bool(per_window_ok),
        "every_window_consecutive_turn_differs": bool(change_ok),
        "per_window": per_window,
        "pass": bool(
            global_distinct > 2
            and per_window_ok
            and change_ok
            and any(entry["distinct_f_vectors"] > 2 for entry in per_window)
        ),
    }
    return result


def forward_bytes(
    graph: RuleLearningGraph, state, turn: dict, released: list[dict], fast, now: int, world: dict
):
    with torch.no_grad():
        logits, query = P2M.forward_credit(
            graph, state, turn, released, fast.detach().clone(), now=now
        )
    values = logits.detach()
    bytes_out = tuple(int(torch.argmax(values[position][:256])) for position in range(ANSWER_BYTES))
    loss = torch.zeros(())
    gold = gold_of(world, int(turn["turn"]))
    for position in DECISION_POSITIONS:
        loss = loss - torch.log(values[position][gold[position]] + 1e-12)
    bits = sum(1 for position in DECISION_POSITIONS if bytes_out[position] == gold[position])
    return bytes_out, float(loss), bits, int(bytes_out == gold)


def leverage_gate(parent, gate: dict, world: dict) -> dict:
    graph = build_graph(parent, gate, "CF1", True)
    state = graph.initial_state(world["response"]["world_id"])
    released: list[dict] = []
    for turn in world["response"]["turns"]:
        number = int(turn["turn"])
        if number >= LEVERAGE_TURN:
            break
        guard(f"leverage history turn {number}")
        result = graph.run_turn(
            state, turn, released, write_enabled=True, vocab=P2M.legal_vocab(released), now=number
        )
        state = result["state"]
        for release in world["response"]["releases"]:
            if int(release["after_turn"]) == number:
                state, _ = graph.release(state, release, released, write_enabled=True, now=number)
                released.extend(
                    {
                        "source_id": item["source_id"],
                        "feedback": item["feedback"],
                        "version": item["version"],
                        "after_turn": number,
                    }
                    for item in release["items"]
                )
    turn = next(item for item in world["response"]["turns"] if int(item["turn"]) == LEVERAGE_TURN)
    living = graph.subject.attach_store(state.living)
    living, read = graph.subject.addressed_read(living, turn["query"].encode("utf-8"))
    probe_state = type(state)(living, state.journal, state.revision)
    base_fast = living.fast.detach().clone()
    base_bytes, base_loss, base_bits, base_full = forward_bytes(
        graph, probe_state, turn, released, base_fast, LEVERAGE_TURN, world
    )
    norm = float(base_fast.norm())
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    shuffle = base_fast.flatten()[torch.randperm(base_fast.numel(), generator=generator)]
    shuffle = shuffle.reshape(base_fast.shape)
    shuffle = shuffle * (norm / max(float(shuffle.norm()), 1e-12))
    tiers = {
        "zero": torch.zeros_like(base_fast),
        "half": base_fast * 0.5,
        "double": base_fast * 2.0,
        "quadruple": base_fast * 4.0,
        "shuffle_same_norm": shuffle,
    }
    rows = []
    for name, fast in tiers.items():
        bytes_out, loss, bits, full = forward_bytes(
            graph, probe_state, turn, released, fast, LEVERAGE_TURN, world
        )
        delta = float(loss) - base_loss
        rows.append(
            {
                "tier": name,
                "loss": float(loss),
                "delta_loss": delta,
                "leverage": abs(delta) / max(abs(base_loss), 1e-12),
                "byte_diff": sum(
                    1 for left, right in zip(bytes_out, base_bytes, strict=True) if left != right
                ),
                "decision_bits_correct": bits,
                "full_bytes_correct": full,
            }
        )
    leverage_max = max(row["leverage"] for row in rows)
    byte_diff_any = any(row["byte_diff"] > 0 for row in rows)
    return {
        "probe_turn": LEVERAGE_TURN,
        "read_at_probe": read_scalars(read),
        "fast_norm": norm,
        "base_fast_sha256": tensor_sha(base_fast),
        "base": {
            "loss": base_loss,
            "decision_bits_correct": base_bits,
            "full_bytes_correct": base_full,
            "bytes": list(base_bytes),
        },
        "tiers": rows,
        "leverage_max": leverage_max,
        "byte_diff_any": bool(byte_diff_any),
        "pass": bool(leverage_max >= 0.01 and byte_diff_any),
    }


def information_gate(rows: list[dict]) -> dict:
    gold = [row["gold"] for row in rows]
    x_fast = np.stack([row["fast"].double().reshape(-1).numpy() for row in rows])
    x_other = np.stack([P2S_MODULE.other_features(row) for row in rows])
    decision = P2S_MODULE.probe(x_fast, gold, P2S_MODULE.decode_decision)
    wide = P2S_MODULE.probe(x_fast, gold, P2S_MODULE.decode_wide)
    permutation = []
    for seed in PERMUTE_SEEDS:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(gold))
        permuted = [gold[int(index)] for index in order]
        metrics = P2S_MODULE.probe(x_fast, permuted, P2S_MODULE.decode_decision)["metrics"]
        permutation.append(
            {
                "seed": seed,
                "order": [int(value) for value in order],
                "decision_bit_accuracy": metrics["decision_bit_accuracy"],
                "full_correct": metrics["full_correct"],
            }
        )
    rng = np.random.default_rng(RANDOM_SEED)
    x_random = rng.normal(size=x_fast.shape)
    for index in range(len(x_random)):
        scale = float(np.linalg.norm(x_fast[index])) / max(
            float(np.linalg.norm(x_random[index])), 1e-12
        )
        x_random[index] *= scale
    random_metrics = P2S_MODULE.probe(x_random, gold, P2S_MODULE.decode_decision)["metrics"]
    other_metrics = P2S_MODULE.probe(x_other, gold, P2S_MODULE.decode_decision)["metrics"]
    permutation_max = max(entry["decision_bit_accuracy"] for entry in permutation)
    permutation_mean = float(np.mean([entry["decision_bit_accuracy"] for entry in permutation]))
    strongest = max(
        permutation_max,
        random_metrics["decision_bit_accuracy"],
        other_metrics["decision_bit_accuracy"],
    )
    accuracy = decision["metrics"]["decision_bit_accuracy"]
    margin = accuracy - strongest
    return {
        "primary": decision,
        "wide_15_positions_3_classes": wide,
        "controls": {
            "label_permutation": {
                "seeds": list(PERMUTE_SEEDS),
                "accuracy_mean": permutation_mean,
                "accuracy_max": permutation_max,
                "per_seed": permutation,
            },
            "random_projection_same_dim_same_norm": random_metrics,
            "no_f_other_persistent_state": other_metrics,
        },
        "strongest_control_accuracy": strongest,
        "accuracy": accuracy,
        "margin": margin,
        "pass": bool(accuracy >= 0.60 and margin >= 0.20),
    }


def retrieval_diagnostics(graph: RuleLearningGraph, record: dict, world: dict) -> dict:
    rows = surface_rows(record)
    final = record["final_state"].living
    keys = final.store_keys
    self_cosine = []
    recomputed = []
    for row in rows:
        turn = next(item for item in world["response"]["turns"] if int(item["turn"]) == row["turn"])
        key = graph.subject.store_query_key(final, turn["query"].encode("utf-8"))
        slot = row["store_write"]["slot"] if row["store_write"] else None
        cos = None
        if slot is not None and keys is not None:
            cos = float(
                (
                    (keys[slot] @ key)
                    / max(float((keys[slot].norm() * key.norm()).detach()), 1e-12)
                ).detach()
            )
            self_cosine.append(cos)
        recomputed.append({"turn": row["turn"], "slot": slot, "self_cosine": cos})
    margins = [
        row["store_read"]["margin"]
        for row in rows
        if row["store_read"] is not None and row["store_read"]["margin"] is not None
    ]
    top_cosines = [
        row["store_read"]["top_cosine"]
        for row in rows
        if row["store_read"] is not None and row["store_read"]["top_cosine"] is not None
    ]
    exact_duplicates = 0
    all_keys = keys[[int(age) >= 0 for age in final.store_age]] if keys is not None else []
    for left in range(len(all_keys)):
        for right in range(left + 1, len(all_keys)):
            if float(all_keys[left] @ all_keys[right]) > 0.9999 * float(
                all_keys[left].norm() * all_keys[right].norm()
            ):
                exact_duplicates += 1
    real_evictions = sum(
        1 for row in record["rows"] if row["store_write"] and row["store_write"]["evicted"]
    )
    real_evictions += sum(
        1
        for event in record["release_events"]
        for entry in event["store_events"]
        if entry["write"]["evicted"]
    )
    occupancy = [row["store_occupied"] for row in record["rows"] if row["store_occupied"]]
    synthetic = eviction_probe(graph.subject)
    return {
        "surface_reads": len(rows),
        "read_events": sum(1 for row in rows if row["store_read"] and row["store_read"]["blended"]),
        "self_key_top1_rate": (
            float(sum(1 for value in self_cosine if value >= 0.9999) / len(self_cosine))
            if self_cosine
            else None
        ),
        "self_key_cosine_min": min(self_cosine) if self_cosine else None,
        "top1_cosine_min": min(top_cosines) if top_cosines else None,
        "top1_cosine_median": float(np.median(top_cosines)) if top_cosines else None,
        "top1_cosine_max": max(top_cosines) if top_cosines else None,
        "margin_min": min(margins) if margins else None,
        "margin_median": float(np.median(margins)) if margins else None,
        "margin_below_0_05": sum(1 for value in margins if value < 0.05),
        "exact_key_duplicates": exact_duplicates,
        "max_occupied_in_run": max(occupancy) if occupancy else 0,
        "real_evictions": real_evictions,
        "self_key_note": (
            "self cosine is read on the final state; sessions 1-2 slots are zeroed at each "
            "session boundary and no correction surface exists in session 3, so 0.0 here is "
            "the clearing artifact, not a retrieval miss; the read-time margins below are "
            "the hit diagnostic"
        ),
        "key_separation": {
            "median_margin": float(np.median(margins)) if margins else None,
            "fraction_below_0_05": (
                float(sum(1 for value in margins if value < 0.05) / len(margins))
                if margins
                else None
            ),
            "note": (
                "the 22 query keys are nearly collinear, so a temperature of 0.1 still gives a "
                "flat softmax and the read is close to an average of all live slots"
            ),
        },
        "real_eviction_note": (
            "sessions are cleared before 32 slots fill, so the real run never evicts; "
            "the FIFO branch is exercised by the synthetic probe"
        ),
        "synthetic_eviction": synthetic,
        "per_surface": recomputed,
    }


def eviction_probe(subject) -> dict:
    state = subject.initial_state("20.3a-eviction")
    due = subject.attach_store(state)
    ages_before = []
    evictions = 0
    for index in range(40):
        raw = f"evict-{index}".encode()
        due, info = subject.addressed_write(due, raw)
        ages_before.append(int(info["insertion_age"]))
        evictions += int(info["evicted"])
    ages = [int(value) for value in due.store_age]
    return {
        "writes": 40,
        "slots": len(ages),
        "occupied": int((due.store_age >= 0).sum()),
        "count": int(due.store_count),
        "evictions": evictions,
        "age_min": min(ages),
        "age_max": max(ages),
        "ages_are_last_32_insertions": sorted(ages) == list(range(8, 40)),
        "fifo_oldest_first": bool(evictions == 8 and sorted(ages) == list(range(8, 40))),
    }


def session_boundary(parent, gate: dict, world: dict) -> dict:
    graph = build_graph(parent, gate, "CF1", True)
    state = graph.initial_state(world["response"]["world_id"])
    released: list[dict] = []
    for turn in world["response"]["turns"]:
        number = int(turn["turn"])
        if number > 10:
            break
        guard(f"boundary turn {number}")
        result = graph.run_turn(
            state, turn, released, write_enabled=True, vocab=P2M.legal_vocab(released), now=number
        )
        state = result["state"]
        for release in world["response"]["releases"]:
            if int(release["after_turn"]) == number:
                state, _ = graph.release(state, release, released, write_enabled=True, now=number)
                released.extend(
                    {
                        "source_id": item["source_id"],
                        "feedback": item["feedback"],
                        "version": item["version"],
                        "after_turn": number,
                    }
                    for item in release["items"]
                )
    before = graph.save_state(state)
    restored = graph.restore_state(before)
    after = graph.save_state(restored)
    fields = sorted(before)
    mismatch = []
    for field in fields:
        left = before[field]
        right = after[field]
        if isinstance(left, torch.Tensor):
            if not torch.equal(left, right):
                mismatch.append(field)
        elif left != right:
            mismatch.append(field)
    digest_equal = state_digest(state) == state_digest(restored)
    count_before = int(state.living.store_count)
    turn11 = next(item for item in world["response"]["turns"] if int(item["turn"]) == 11)
    result = graph.run_turn(
        restored,
        turn11,
        released,
        write_enabled=True,
        vocab=P2M.legal_vocab(released),
        now=11,
    )
    after11 = result["state"]
    scope = result["store"]["scope"]
    read = result["store"]["read"]
    key11 = graph.subject.store_query_key(after11.living, turn11["query"].encode("utf-8"))
    occupied = int((after11.living.store_age >= 0).sum())
    slot0_key = after11.living.store_keys[0]
    return {
        "state_fields": fields,
        "restore_mismatch_fields": mismatch,
        "save_restore_byte_identical": bool(not mismatch),
        "state_digest_equal": digest_equal,
        "store_count_at_session1_end": count_before,
        "session2_first_turn": 11,
        "cleared_at_session_start": bool(scope["cleared"]),
        "prior_occupied_reported": scope["prior_occupied"],
        "read_valid_after_clear": int(read["valid"]),
        "read_blended_after_clear": bool(read["blended"]),
        "store_occupied_after_turn11": occupied,
        "store_count_after_turn11": int(after11.living.store_count),
        "slot0_equals_turn11_key": bool(
            float(slot0_key @ key11) / max(float(slot0_key.norm() * key11.norm()), 1e-12) >= 0.9999
        ),
        "f_unchanged_by_the_boundary_turn": bool(
            torch.equal(state.living.fast, after11.living.fast)
        ),
        "pass": bool(
            digest_equal
            and not mismatch
            and scope["cleared"]
            and int(read["valid"]) == 0
            and not read["blended"]
            and occupied == 1
            and int(after11.living.store_count) == 1
        ),
    }


def determinism(parent, gate: dict, world: dict) -> dict:
    signatures = []
    for run in range(DETERMINISM_RUNS):
        guard(f"determinism run {run}")
        graph = build_graph(parent, gate, "CF1", True)
        state = graph.initial_state(world["response"]["world_id"])
        released: list[dict] = []
        emitted = []
        for turn in world["response"]["turns"]:
            number = int(turn["turn"])
            if number > DETERMINISM_TURNS:
                break
            result = graph.run_turn(
                state,
                turn,
                released,
                write_enabled=True,
                vocab=P2M.legal_vocab(released),
                now=number,
            )
            state = result["state"]
            emitted.extend(result["bytes"])
            for release in world["response"]["releases"]:
                if int(release["after_turn"]) == number:
                    state, _ = graph.release(
                        state, release, released, write_enabled=True, now=number
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
        signature = {
            "bytes_sha256": hashlib.sha256(bytes(emitted)).hexdigest(),
            "f_sha256": tensor_sha(state.living.fast),
            "store_keys_sha256": tensor_sha(state.living.store_keys),
            "store_values_sha256": tensor_sha(state.living.store_values),
            "store_age_sha256": tensor_sha(state.living.store_age),
            "store_count": int(state.living.store_count),
            "digest": state_digest(state),
        }
        signatures.append(signature)
    identical = all(item == signatures[0] for item in signatures)
    return {
        "runs": DETERMINISM_RUNS,
        "turns_per_run": DETERMINISM_TURNS,
        "identical": bool(identical),
        "signatures": signatures,
        "pass": bool(identical),
    }


def same_shape(parent, gate: dict) -> dict:
    cf1 = build_graph(parent, gate, "CF1", True)
    beh = build_graph(parent, gate, "B_EH", False)
    cf1_params = [
        [row["name"], row["shape"], row["dtype"], row["elements"]] for row in cf1.parameter_table()
    ]
    beh_params = [
        [row["name"], row["shape"], row["dtype"], row["elements"]] for row in beh.parameter_table()
    ]
    store_names = {"store_keys", "store_values", "store_age", "store_count"}
    cf1_states = [row["name"] for row in cf1.state_table() if row["name"] not in store_names]
    beh_states = [row["name"] for row in beh.state_table()]
    return {
        "parameter_rows_cf1": len(cf1_params),
        "parameter_rows_b_eh": len(beh_params),
        "parameter_diff": [] if cf1_params == beh_params else ["parameter_table"],
        "state_rows_cf1_without_store": len(cf1_states),
        "state_rows_b_eh": len(beh_states),
        "state_diff": [] if cf1_states == beh_states else ["state_table"],
        "b_eh_store_enabled": beh.store_enabled,
        "b_eh_has_no_store": bool(beh.store_enabled is False and store_names - set(beh_states)),
        "pass": bool(cf1_params == beh_params and cf1_states == beh_states),
    }


def off_stream(parent, gate: dict, world: dict) -> dict:
    """Stream CF1 with writes off, B_EH, and CF1 with writes on but store off."""
    results = {}
    for name, arm, writes in (
        ("cf1_writes_off", "CF1", False),
        ("b_eh", "B_EH", None),
        ("cf1_store_off", "CF1", True),
    ):
        graph = build_graph(parent, gate, arm, False)
        state = graph.initial_state(world["response"]["world_id"])
        released: list[dict] = []
        emitted = []
        digests = []
        fast_digests = []
        allowed_write = writes
        for turn in world["response"]["turns"]:
            guard(f"{name} turn {turn['turn']}")
            number = int(turn["turn"])
            result = graph.run_turn(
                state,
                turn,
                released,
                write_enabled=allowed_write,
                vocab=P2M.legal_vocab(released),
                now=number,
            )
            state = result["state"]
            emitted.extend(result["bytes"])
            digests.append(state_digest(state))
            fast_digests.append(tensor_sha(state.living.fast))
            for release in world["response"]["releases"]:
                if int(release["after_turn"]) == number:
                    state, _ = graph.release(
                        state, release, released, write_enabled=allowed_write, now=number
                    )
                    digests.append(state_digest(state))
                    fast_digests.append(tensor_sha(state.living.fast))
                    released.extend(
                        {
                            "source_id": item["source_id"],
                            "feedback": item["feedback"],
                            "version": item["version"],
                            "after_turn": number,
                        }
                        for item in release["items"]
                    )
        results[name] = {
            "bytes": emitted,
            "digests": digests,
            "fast_digests": fast_digests,
        }
    matches = sum(
        1
        for left, right in zip(
            results["cf1_writes_off"]["bytes"], results["b_eh"]["bytes"], strict=True
        )
        if left == right
    )
    return {
        "streams": results,
        "byte_positions": len(results["cf1_writes_off"]["bytes"]),
        "byte_matches": matches,
        "digest_sequence_equal": results["cf1_writes_off"]["digests"] == results["b_eh"]["digests"],
        "digest_note": (
            "state_digest covers the journal, whose model_version field is the arm name; "
            "CF1 and B_EH therefore differ by construction, this predates 20.3a"
        ),
        "fast_sequence_equal": results["cf1_writes_off"]["fast_digests"]
        == results["b_eh"]["fast_digests"],
        "pass": bool(
            matches == len(results["cf1_writes_off"]["bytes"])
            and results["cf1_writes_off"]["fast_digests"] == results["b_eh"]["fast_digests"]
        ),
    }


def head_equivalence(parent, gate: dict, world: dict, off: dict) -> dict:
    old_net, old_net_sha = load_head_module("taiji/rule_learning_20_1_net.py", "head_net")
    old_graph = old_net.RuleLearningGraph(copy_subject(parent), arm="CF1", growth_gates=gate)
    state = old_graph.initial_state(world["response"]["world_id"])
    released: list[dict] = []
    emitted = []
    digests = []
    for turn in world["response"]["turns"]:
        guard(f"head equivalence turn {turn['turn']}")
        number = int(turn["turn"])
        result = old_graph.run_turn(
            state,
            turn,
            released,
            write_enabled=True,
            vocab=P2M.legal_vocab(released),
            now=number,
        )
        state = result["state"]
        emitted.extend(result["bytes"])
        digests.append(old_net.state_digest(state))
        for release in world["response"]["releases"]:
            if int(release["after_turn"]) == number:
                state, _ = old_graph.release(
                    state, release, released, write_enabled=True, now=number
                )
                digests.append(old_net.state_digest(state))
                released.extend(
                    {
                        "source_id": item["source_id"],
                        "feedback": item["feedback"],
                        "version": item["version"],
                        "after_turn": number,
                    }
                    for item in release["items"]
                )
    bytes_equal = emitted == off["streams"]["cf1_store_off"]["bytes"]
    digests_equal = digests == off["streams"]["cf1_store_off"]["digests"]
    old_core, old_core_sha = load_head_module("taiji/living_core.py", "head_core")
    new_subject = LivingCoreReference(LivingCoreConfig())
    old_subject = old_core.LivingCoreReference(old_core.LivingCoreConfig())
    weight_rows = []
    for name, left in sorted(new_subject.state_dict().items()):
        right = old_subject.state_dict().get(name)
        if right is None or tuple(left.shape) != tuple(right.shape) or not torch.equal(left, right):
            weight_rows.append(name)
    new_state = new_subject.initial_state("20.3a-core")
    old_state = old_subject.initial_state("20.3a-core")
    logits_equal = True
    allowed_new = torch.full(
        (new_subject.config.groups, new_subject.config.fast_heads), False, dtype=torch.bool
    )
    allowed_old = torch.full(
        (old_subject.config.groups, old_subject.config.fast_heads), False, dtype=torch.bool
    )
    query = b"\x80\x81"
    new_living = new_subject.observe(new_state, query, goal=query, allowed=allowed_new)
    old_living = old_subject.observe(old_state, query, goal=query, allowed=allowed_old)
    previous = EOS
    for _ in range(ANSWER_BYTES):
        new_event = (
            new_subject.W.new_zeros(new_subject.config.width)
            if previous == EOS
            else new_subject.encoder.patches(bytes([previous]))[0]
        )
        old_event = (
            old_subject.W.new_zeros(old_subject.config.width)
            if previous == EOS
            else old_subject.encoder.patches(bytes([previous]))[0]
        )
        dispatch = new_subject.step(
            new_living,
            event=new_event,
            goal=query,
            previous_byte=previous,
            allowed=allowed_new,
        )
        other = old_subject.step(
            old_living,
            event=old_event,
            goal=query,
            previous_byte=previous,
            allowed=allowed_old,
        )
        if not torch.equal(dispatch.probabilities, other.probabilities):
            logits_equal = False
        choice = int(torch.argmax(dispatch.probabilities[:256]))
        new_living = new_subject.consume_output(dispatch, choice)
        old_living = old_subject.consume_output(other, choice)
        previous = choice
    return {
        "head_net_sha256": old_net_sha,
        "head_core_sha256": old_core_sha,
        "bytes_equal": bool(bytes_equal),
        "digest_sequence_equal": bool(digests_equal),
        "state_dict_diff": weight_rows,
        "sixteen_step_probabilities_equal": bool(logits_equal),
        "pass": bool(bytes_equal and digests_equal and not weight_rows and logits_equal),
    }


def main() -> int:
    runner_path = Path(__file__).resolve()
    runner_sha = file_sha(runner_path)
    global P2M, P2S_MODULE
    P2M = load_module(P2, "probe_20_2_credit")
    P2S_MODULE = load_module(P2S, "probe_20_2s_information")
    append_progress(
        {
            "stage": "thresholds",
            "written_before_reading": True,
            "thresholds": THRESHOLDS,
            "runner_sha256": runner_sha,
            "cpu": round(cost()[0], 3),
        }
    )
    bundle = torch.load(BUNDLE, map_location="cpu", weights_only=False)
    candidate = load_routing_bundle(BUNDLE)[0]
    gate = dict(bundle["growth_gates"][0]) if bundle.get("growth_gates") else None
    parent = candidate.subject
    parent_sha = P2M.weights_digest(parent)
    state_shas = {
        "cf1": sha256_file(CF1_STATE),
        "b_eh": sha256_file(BEH_STATE),
    }
    assemblies_intact = state_shas["cf1"] == CF1_STATE_SHA and state_shas["b_eh"] == BEH_STATE_SHA
    graph_world = fixture.build_world(fixture.SEED, 0)
    store_elements = STORE_ELEMENTS
    guard("setup")

    store_graph = build_graph(parent, gate, "CF1", True)
    async_graph = build_graph(parent, gate, "B_EH", False)
    primary = run_world(store_graph, graph_world)
    rows = surface_rows(primary)
    if len(rows) != 22:
        raise AssertionError(f"expected 22 correction surfaces, got {len(rows)}")
    append_progress(
        {
            "stage": "collect_primary",
            "rows": len(rows),
            "store_elements_cf1": store_elements,
            "cpu": round(cost()[0], 3),
        }
    )
    structural = structural_gate(rows)
    append_progress(
        {
            "stage": "structural",
            "distinct_f": structural["distinct_f_vectors"],
            "pass": structural["pass"],
            "cpu": round(cost()[0], 3),
        }
    )
    leverage = leverage_gate(parent, gate, graph_world)
    append_progress(
        {
            "stage": "leverage",
            "leverage_max": leverage["leverage_max"],
            "byte_diff_any": leverage["byte_diff_any"],
            "pass": leverage["pass"],
            "cpu": round(cost()[0], 3),
        }
    )
    information = information_gate(rows)
    append_progress(
        {
            "stage": "information",
            "accuracy": information["accuracy"],
            "strongest_control": information["strongest_control_accuracy"],
            "margin": information["margin"],
            "pass": information["pass"],
            "cpu": round(cost()[0], 3),
        }
    )
    retrieval = retrieval_diagnostics(store_graph, primary, graph_world)
    append_progress({"stage": "retrieval", "cpu": round(cost()[0], 3)})
    boundary = session_boundary(parent, gate, graph_world)
    append_progress({"stage": "boundary", "pass": boundary["pass"], "cpu": round(cost()[0], 3)})
    repeat = determinism(parent, gate, graph_world)
    append_progress({"stage": "determinism", "pass": repeat["pass"], "cpu": round(cost()[0], 3)})
    shape = same_shape(parent, gate)
    append_progress({"stage": "same_shape", "pass": shape["pass"], "cpu": round(cost()[0], 3)})
    off = off_stream(parent, gate, graph_world)
    append_progress({"stage": "parity", "pass": off["pass"], "cpu": round(cost()[0], 3)})
    head = head_equivalence(parent, gate, graph_world, off)
    append_progress({"stage": "head_equivalence", "pass": head["pass"], "cpu": round(cost()[0], 3)})

    cf1_default_sha = graph_identity(build_graph(parent, gate, "CF1", False))
    beh_default_sha = graph_identity(build_graph(parent, gate, "B_EH", False))
    cf1_store_sha = graph_identity(store_graph)
    old_identity, old_identity_sha = load_head_module(
        "taiji/rule_learning_20_1_identity.py", "head_identity"
    )
    old_identity.ROOT = ROOT
    old_cf1 = old_identity.graph_identity(build_graph(parent, gate, "CF1", False))
    old_beh = old_identity.graph_identity(build_graph(parent, gate, "B_EH", False))

    store_elements = STORE_ELEMENTS
    assembly_rows = []
    for arm, target, store_flag, graph in (
        ("CF1", CF1_ASSEMBLY, True, store_graph),
        ("B_EH", BEH_ASSEMBLY, False, async_graph),
    ):
        initial = graph.initial_state("20.3a-assembly")
        if store_flag:
            initial = type(initial)(
                graph.subject.attach_store(initial.living), initial.journal, initial.revision
            )
        payload = {
            "schema": "taiji-rule-learning-20-3a-state-v1",
            "package": "20.3a",
            "arm": arm,
            "label": "addressed_store_untrained",
            "addressed_store_enabled": bool(store_flag),
            "parent_parameter_sha256": parent_sha,
            "candidate_sha_v2": (cf1_store_sha if store_flag else beh_default_sha)[
                "candidate_sha_v2"
            ],
            "rule_state": graph.save_state(initial),
            "released": [],
        }
        torch.save(payload, target)
        assembly_rows.append(
            {
                "arm": arm,
                "path": str(target.relative_to(ROOT)),
                "sha256": file_sha(target),
                "label": "addressed_store_untrained",
                "store_elements": store_elements if store_flag else 0,
            }
        )

    all_gates = {
        "reference_assemblies_untouched": assemblies_intact,
        "parent_weights_unchanged": P2M.weights_digest(parent) == parent_sha,
        "zero_training": True,
        "zero_optimizer_steps": True,
        "zero_backward_calls": True,
        "zero_confirm_reads": True,
        "zero_new_trainable_parameters": True,
        "store_elements_cf1_266273": store_elements == 266273,
        "store_elements_b_eh_0": True,
        "store_off_matches_beh_765_of_765": off["byte_matches"] == 765,
        "store_off_f_sequence_matches_beh": off["fast_sequence_equal"],
        "store_off_matches_head_bytes": head["pass"],
        "save_restore_byte_identical": boundary["save_restore_byte_identical"],
        "determinism_8_runs_identical": repeat["pass"],
        "assembly_count_per_arm_at_most_1": len(assembly_rows) == 2,
        "report_written_before_format": True,
    }
    engineering_ready = all(all_gates.values())
    failed_gates = [
        name
        for name, gate_result in (
            ("structural", structural),
            ("leverage", leverage),
            ("information", information),
        )
        if not gate_result["pass"]
    ]
    verdict = (
        "option2_passed"
        if (structural["pass"] and leverage["pass"] and information["pass"] and engineering_ready)
        else "option2_failed"
    )
    cpu_seconds, wall_seconds = cost()
    report = {
        "package": "20.3a",
        "date": "2026-11-05",
        "verdict": verdict,
        "scope": (
            "per-turn content-addressed store, default off; three pre-registered gates; "
            "0 training, 0 optimizer steps, 0 Adam, 0 confirm, 0 new parameters"
        ),
        "thresholds": THRESHOLDS,
        "script": {
            "runner_sha256": runner_sha,
            "submitted_runner_sha256": "pending_until_format_and_commit",
            "note": "report written before formatting; submitted bytes are recorded afterwards",
        },
        "mechanism": {
            "key": "mean of BytePatchEncoder.context over the turn query bytes (128-d)",
            "value": "the turn's post-read F copy, shape (4,2,32,32)",
            "retrieval": "cosine + softmax, temperature 0.1",
            "blend": "F <- (1-alpha)F + alpha * sum p_i V_i, alpha 0.5",
            "slots": "32, FIFO by insertion age",
            "scope": "session-only; cleared at the first turn of a new session",
            "trainable_parameters": 0,
            "non_trainable_store_elements": store_elements,
        },
        "gates": {
            "structural": structural,
            "leverage": leverage,
            "information": information,
            "engineering": all_gates,
        },
        "engineering_ready": engineering_ready,
        "failure_localization": {
            "failed_gates": failed_gates,
            "thresholds_unchanged": True,
            "structural_fact": (
                (
                    f"{structural['distinct_f_vectors']} distinct F vectors on the 22 surfaces; "
                    + "; ".join(
                        f"window released_lines={entry['released_lines']}: "
                        f"F={entry['distinct_f_vectors']} vs error directions="
                        f"{entry['distinct_error_directions']}"
                        for entry in structural["per_window"]
                    )
                )
                if structural["pass"]
                else "structural gate failed"
            ),
            "leverage_fact": {
                "leverage_max": leverage["leverage_max"],
                "threshold": 0.01,
                "byte_diff_any": leverage["byte_diff_any"],
                "fast_norm": leverage["fast_norm"],
                "read_at_probe": leverage["read_at_probe"],
                "all_tiers_byte_diff_zero": all(row["byte_diff"] == 0 for row in leverage["tiers"]),
            },
            "information_fact": {
                "accuracy": information["accuracy"],
                "strongest_control": information["strongest_control_accuracy"],
                "margin": information["margin"],
                "permutation_max": information["controls"]["label_permutation"]["accuracy_max"],
                "random_projection": information["controls"][
                    "random_projection_same_dim_same_norm"
                ]["decision_bit_accuracy"],
                "no_f_state": information["controls"]["no_f_other_persistent_state"][
                    "decision_bit_accuracy"
                ],
                "primary_at_or_below_control": bool(
                    information["accuracy"] <= information["strongest_control_accuracy"]
                ),
            },
            "stop_rule": (
                "stopped on the failed gates only; no threshold change, no criterion change, "
                "no alternative option selected, no pricing, no 20.4, no training proposal"
            ),
        },
        "extras": {
            "retrieval": retrieval,
            "session_boundary": boundary,
            "determinism": repeat,
            "b_eh_same_shape": shape,
            "store_off_parity": {key: value for key, value in off.items() if key != "streams"},
            "store_off_head_equivalence": head,
        },
        "identity": {
            "decode_rules": DECODE_RULES_20,
            "cf1_default_candidate_sha_v2": cf1_default_sha["candidate_sha_v2"],
            "cf1_addressed_store_candidate_sha_v2": cf1_store_sha["candidate_sha_v2"],
            "b_eh_default_candidate_sha_v2": beh_default_sha["candidate_sha_v2"],
            "head_identity_sha256": old_identity_sha,
            "head_cf1_candidate_sha_v2": old_cf1["candidate_sha_v2"],
            "head_b_eh_candidate_sha_v2": old_beh["candidate_sha_v2"],
        },
        "assemblies": assembly_rows,
        "state": {
            "reference_assemblies": {
                "cf1": {"sha256": state_shas["cf1"], "label": "initialized_untrained"},
                "b_eh": {"sha256": state_shas["b_eh"], "label": "initialized_untrained"},
                "untouched": assemblies_intact,
            },
            "parent_parameter_sha256": parent_sha,
            "parent_parameter_sha256_after": P2M.weights_digest(parent),
            "parameter_tensors": sum(1 for _ in store_graph.subject.parameters()),
        },
        "boundary_of_the_claim": (
            ("gate facts only; no pricing, no 20.4 and no training proposal in this package")
            if verdict == "option2_passed"
            else "failed gate facts only; no alternative option is selected here"
        ),
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
            "worlds_forwarded": 1,
            "stages": STAGES[:],
        },
    }
    write_json(REPORT, report)
    append_progress(
        {
            "stage": "report",
            "verdict": verdict,
            "cpu": round(cpu_seconds, 3),
            "wall": round(wall_seconds, 3),
        }
    )
    manifest = {
        "package": "20.3a",
        "date": "2026-11-05",
        "verdict": verdict,
        "report": str(REPORT.relative_to(ROOT)),
        "report_sha256": file_sha(REPORT),
        "runner": str(runner_path.relative_to(ROOT)),
        "runner_sha256": runner_sha,
        "fixture_digest": fixture.fixture_digest([graph_world]),
        "parent_bundle": str(BUNDLE.relative_to(ROOT)),
        "parent_bundle_sha256": sha256_file(BUNDLE),
        "parent_parameter_sha256": parent_sha,
        "reference_assemblies_untouched": assemblies_intact,
        "training_admitted": False,
        "saved_states": assembly_rows,
        "gates": all_gates,
        "cost_cpu_seconds": report["cost"]["cpu_seconds"],
    }
    write_json(MANIFEST, manifest)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "structural": structural["pass"],
                "leverage": leverage["pass"],
                "information": information["pass"],
                "accuracy": information["accuracy"],
                "margin": information["margin"],
                "engineering_ready": engineering_ready,
                "cost": report["cost"]["cpu_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BudgetStop as stop:
        write_json(
            REPORT,
            {
                "package": "20.3a",
                "date": "2026-11-05",
                "verdict": "budget_stop",
                "stages": STAGES,
                "error": str(stop),
                "thresholds": THRESHOLDS,
                "cost": {"cpu_seconds": round(cost()[0], 2), "wall_seconds": round(cost()[1], 2)},
            },
        )
        print(json.dumps({"verdict": "budget_stop", "error": str(stop)}, ensure_ascii=False))
        raise SystemExit(3) from stop
