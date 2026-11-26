"""20.3b: F authority gate -- can F push answer bytes at all?

Zero training, zero optimizer steps, zero Adam, zero confirm, zero new
parameters persisted, graphs unchanged.  The oracle F is constructed by a
teacher-forced full-score replay (gold bytes are consumed at every position):
no gradients are taken anywhere.  All three readouts are solved in closed form
in memory and never enter ``state_dict``.

Readings (thresholds are written before collection):
  A. gap -- per correction surface, the decision-bit logit gap to gold, the
     gold-class rank distribution, and the non-template positions needing a
     flip, plus the scale of the required shift against the local logit spread;
  B. three readouts on the oracle F --
     1. uniform gain ``logits += g * (F projection to decision positions)``
        with g in {1,10,100,1000,1e4};
     2. closed-form ridge ``W = Y^T X (XX^T + lambda I)^-1`` with leave-one-out
        over the 22 surfaces and lambda in {1e-4,1e-3,1e-2,1e-1};
     3. rank-constrained readout r in {1,2,4,8};
  C. verdict -- restorable / weak / not wired, by the pre-registered rule.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)

from taiji import rule_learning_20_1 as fixture  # noqa: E402
from taiji.living_core import EOS  # noqa: E402
from taiji.rule_learning_20_1_net import (  # noqa: E402
    ACTIONS,
    ANSWER_BYTES,
    DECISION_POSITIONS,
)

P2 = ROOT / "scripts/training/probe_taiji_package20_2_credit_recovery.py"
REPORT = ROOT / "reports/taiji_package20_3b_f_authority_20261105.json"
MANIFEST = ROOT / "plans/manifests/taiji_package20_3b_f_authority_20261105.json"
PROGRESS = ROOT / "reports/taiji_package20_3b_f_authority_20261105_progress.jsonl"
CF1_STATE = ROOT / "output/taiji_package20_1_cf1_initialized_untrained.pt"
CF1_STATE_SHA = "b62d6fe6a0721fd82c7c43d575e600c48ffc8e31ff959bfdf14ef9ff07d0039e"

CPU_CAP = 150.0
CPU_GUARD = 140.0
WALL_CAP = 1800.0
GAIN_GRID = (1.0, 10.0, 100.0, 1000.0, 10000.0)
LAMBDA_GRID = (1e-4, 1e-3, 1e-2, 1e-1)
RANKS = (1, 2, 4, 8)
TARGET_MARGIN = 1.0
LEFT_BYTES = tuple(ACTIONS[0].encode("utf-8")[position] for position in DECISION_POSITIONS)
RIGHT_BYTES = tuple(ACTIONS[1].encode("utf-8")[position] for position in DECISION_POSITIONS)

DECISION_RULES = {
    "restorable": (
        "some readout reaches decision_correct >= 11 of 22 and byte_diff > 0 "
        "(byte_diff counted against the actual-run constrained bytes)"
    ),
    "not_wired": (
        "every readout at its extreme setting -- uniform gain g=1e4, ridge at the "
        "selected lambda, rank readout r=8 -- still has byte_diff == 0"
    ),
    "weak": "flips occur but decision_correct stays below 11 of 22",
    "written_before_reading": True,
}

THRESHOLDS = {
    "decision_positions": list(DECISION_POSITIONS),
    "gain_grid": list(GAIN_GRID),
    "lambda_grid": list(LAMBDA_GRID),
    "ranks": list(RANKS),
    "target_margin": TARGET_MARGIN,
    "oracle_definition": (
        "teacher-forced full-score replay: every emitted byte is the gold byte, "
        "so F is the state accumulated under perfect answers; diagnostic only"
    ),
    "uniform_gain_definition": (
        "projection = oracle_logits - zero_logits at decision positions, both read "
        "from the same actual pre-turn state so only F differs; logits(g) = "
        "actual_logits + g * projection"
    ),
    "ridge_y_definition": (
        "per decision bit that the actual answer gets wrong, +1 on the fixed left "
        "action byte and -1 on the fixed right action byte (reversed when the gold "
        "side is right), 0 where the actual answer is already right; the columns are "
        "the two global action bytes, never the per-surface gold label, so no label "
        "is injected at decode time"
    ),
    "target_margin_note": (
        "the target magnitude is 1 while every measured surface deficit is <= 0.0087, "
        "so the readout is a sign test: any correctly signed increment above the "
        "deficit flips the byte"
    ),
    "readout_baseline": (
        "every readout is added to the actual-run decision scores at the same "
        "surface, so byte_diff counts movement away from what the model emits"
    ),
    "verdict_rule": DECISION_RULES,
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


def head_sha(relative: str) -> str:
    raw = subprocess.run(
        ["git", "show", f"HEAD:{relative}"], cwd=ROOT, check=True, capture_output=True
    ).stdout
    return hashlib.sha256(raw).hexdigest()


def load_committed_p2():
    raw = P2.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    spec = importlib.util.spec_from_file_location("probe_taiji_package20_3b_p2", P2)
    module = importlib.util.module_from_spec(spec)
    sys.modules["probe_taiji_package20_3b_p2"] = module
    spec.loader.exec_module(module)
    return module, sha


def clone_state(state):
    living = replace(
        state.living,
        h=state.living.h.clone(),
        fast=state.living.fast.clone(),
        nu_ema=state.living.nu_ema.clone(),
        coverage=state.living.coverage.clone(),
        prior=state.living.prior.clone(),
    )
    return replace(state, living=living, journal=tuple(dict(entry) for entry in state.journal))


def teacher_turn(graph, state, turn: dict, released: list[dict], gold: bytes, now: int):
    """One answer turn with every emitted byte forced to gold."""
    query = turn["query"].encode("utf-8")
    records = graph._records(released, now)
    living = graph.subject.observe(
        state.living,
        query,
        goal=query,
        records=records,
        allowed=graph._allowed(False),
        now=now,
        **graph.step_kwargs,
    )
    logits: list[torch.Tensor] = []
    emitted: list[int] = []
    previous = EOS
    with torch.no_grad():
        for position in range(ANSWER_BYTES):
            event = graph._patch(graph.subject, previous)
            output = graph.subject.step(
                living,
                event=event,
                goal=query,
                records=records,
                previous_byte=previous,
                allowed=graph._allowed(False),
                now=now,
                **graph.step_kwargs,
            )
            logits.append(output.probabilities.detach().clone())
            choice = int(gold[position])
            emitted.append(choice)
            living = graph.subject.consume_output(output, choice)
            previous = choice
    return replace(state, living=living), torch.stack(logits), tuple(emitted)


def actual_surface(p2, graph, world: dict) -> dict:
    """Baseline run keeping the pre-turn state of every correction surface."""
    response = world["response"]
    state = graph.initial_state(response["world_id"])
    released: list[dict] = []
    snaps: list[dict] = []
    rows: list[dict] = []
    for turn in response["turns"]:
        number = int(turn["turn"])
        guard(f"actual turn {number}")
        if bool(turn["correction_surface"]) and released:
            snaps.append(
                {
                    "number": number,
                    "session": int(turn["session"]),
                    "turn": turn,
                    "state": clone_state(state),
                    "released": list(released),
                }
            )
        result = graph.run_turn(
            state,
            turn,
            released,
            write_enabled=True,
            vocab=p2.legal_vocab(released),
            now=number,
        )
        state = result["state"]
        rows.append(
            {
                "turn": number,
                "constrained": tuple(int(value) for value in result["constrained_bytes"]),
                "logits": result["logits"].detach().clone(),
            }
        )
        for release in response["releases"]:
            if int(release["after_turn"]) != number:
                continue
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
    return {"snaps": snaps, "rows": rows, "final_state": state}


def teacher_surface(p2, graph, world: dict) -> dict:
    """Full-score replay: gold bytes consumed at every position."""
    response = world["response"]
    state = graph.initial_state(response["world_id"])
    released: list[dict] = []
    snaps: list[dict] = []
    rows: list[dict] = []
    for turn in response["turns"]:
        number = int(turn["turn"])
        guard(f"teacher turn {number}")
        gold = p2.gold_bytes(world, turn)
        if bool(turn["correction_surface"]) and released:
            snaps.append(
                {
                    "number": number,
                    "session": int(turn["session"]),
                    "turn": turn,
                    "state": clone_state(state),
                    "released": list(released),
                }
            )
        state, logits, emitted = teacher_turn(graph, state, turn, released, gold, number)
        rows.append({"turn": number, "emitted": emitted, "logits": logits})
        for release in response["releases"]:
            if int(release["after_turn"]) != number:
                continue
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
    return {"snaps": snaps, "rows": rows, "final_state": state}


def eval_logits(p2, graph, snap: dict, fast: torch.Tensor) -> torch.Tensor:
    living = replace(snap["state"].living, fast=fast)
    state = replace(snap["state"], living=living)
    result = graph.run_turn(
        state,
        snap["turn"],
        snap["released"],
        write_enabled=True,
        vocab=p2.legal_vocab(snap["released"]),
        now=snap["number"],
    )
    return result["logits"].detach().clone()


def decode(p2, logits: torch.Tensor, released: list[dict]) -> tuple[int, ...]:
    vocab = torch.tensor(p2.legal_vocab(released), dtype=torch.long)
    out: list[int] = []
    for position in range(ANSWER_BYTES):
        candidates = logits[position][vocab]
        out.append(int(vocab[int(torch.argmax(candidates))]))
    return tuple(out)


def decision_correct(answer: tuple[int, ...], gold: bytes) -> bool:
    return all(answer[position] == gold[position] for position in DECISION_POSITIONS)


def decision_bits(answer: tuple[int, ...], gold: bytes) -> int:
    return sum(int(answer[position] == gold[position]) for position in DECISION_POSITIONS)


def byte_diff(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    return sum(1 for a, b in zip(left, right, strict=True) if a != b)


def other_action_byte(gold: bytes, position: int) -> int:
    acts = [action.encode("utf-8")[position] for action in ("应送左箱。", "应送右箱。")]
    return acts[0] if gold[position] == acts[1] else acts[1]


def rank_of(logits: torch.Tensor, position: int, byte: int, vocab: list[int]) -> int:
    values = logits[position][torch.tensor(vocab, dtype=torch.long)]
    target = float(logits[position][byte])
    return int((values > target).sum()) + 1


def median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64))) if values else 0.0


def gap_section(p2, snaps, rows, world, teacher_rows) -> dict:
    decision_gaps: list[float] = []
    surface_min_gaps: list[float] = []
    surface_deficits: list[float] = []
    rank_counts = {"1": 0, "2": 0, "3": 0, ">3": 0}
    spreads: list[float] = []
    template_flip_counts = 0
    template_flip_gaps: list[float] = []
    details = []
    for snap in snaps:
        number = snap["number"]
        row = next(item for item in rows if item["turn"] == number)
        gold = p2.gold_bytes(world, {"turn": number})
        vocab = list(p2.legal_vocab(snap["released"]))
        logits = row["logits"]
        bits = []
        for position in DECISION_POSITIONS:
            other = other_action_byte(gold, position)
            gap = float(logits[position][gold[position]]) - float(logits[position][other])
            rank = rank_of(logits, position, gold[position], vocab)
            decision_gaps.append(gap)
            bits.append(
                {
                    "position": position,
                    "gold": int(gold[position]),
                    "other": int(other),
                    "gap": gap,
                    "rank": rank,
                    "argmax_is_gold": gap > 0,
                }
            )
            bucket = "1" if rank == 1 else "2" if rank == 2 else "3" if rank == 3 else ">3"
            rank_counts[bucket] += 1
            values = logits[position][torch.tensor(vocab, dtype=torch.long)]
            spreads.append(float(values.max()) - float(values.min()))
        template = [
            position
            for position in range(ANSWER_BYTES)
            if position not in DECISION_POSITIONS and row["constrained"][position] != gold[position]
        ]
        flip_gaps = []
        for position in template:
            others = [float(logits[position][byte]) for byte in vocab if byte != gold[position]]
            needed = float(logits[position][gold[position]]) - max(others) if others else 0.0
            template_flip_counts += 1
            template_flip_gaps.append(needed)
            flip_gaps.append(needed)
        min_gap = min(bit["gap"] for bit in bits)
        surface_min_gaps.append(min_gap)
        surface_deficits.append(max(0.0, -min_gap))
        teacher_row = next(item for item in teacher_rows if item["turn"] == number)
        details.append(
            {
                "turn": number,
                "decision_bits": bits,
                "min_gap": min_gap,
                "deficit": max(0.0, -min_gap),
                "template_flip_positions": template,
                "template_flip_gaps": flip_gaps,
                "teacher_decision_correct": decision_correct(teacher_row["emitted"], gold),
                "actual_decision_correct": decision_correct(row["constrained"], gold),
            }
        )
    ratio = median(surface_deficits) / max(median(spreads), 1e-12)
    return {
        "decision_bits_total": len(decision_gaps),
        "decision_gap_median": median(decision_gaps),
        "decision_gap_min": float(min(decision_gaps)) if decision_gaps else 0.0,
        "decision_gap_max": float(max(decision_gaps)) if decision_gaps else 0.0,
        "negative_gap_bits": sum(1 for gap in decision_gaps if gap < 0),
        "gold_rank_distribution": rank_counts,
        "surface_min_gap_median": median(surface_min_gaps),
        "surface_deficit_median": median(surface_deficits),
        "surface_deficit_max": float(max(surface_deficits)) if surface_deficits else 0.0,
        "needed_template_flips": template_flip_counts,
        "template_flip_gap_median": median(template_flip_gaps),
        "template_flip_gap_max": float(max(template_flip_gaps)) if template_flip_gaps else 0.0,
        "logit_spread_median": median(spreads),
        "scale_ratio_median": ratio,
        "conclusion": (
            f"deficit median {median(surface_deficits):.4f} vs spread median "
            f"{median(spreads):.4f} -> {ratio:.3f}x spread"
        ),
        "surfaces": details,
    }


def uniform_gain(p2, snaps, actual_rows, oracle_logits, zero_logits, golds) -> dict:
    rows = []
    first_flip = None
    for gain in GAIN_GRID:
        decision_ok = 0
        full_ok = 0
        diffs = 0
        bit_ok = 0
        for index, snap in enumerate(snaps):
            logits = actual_rows[index]["logits"].clone()
            delta = oracle_logits[index] - zero_logits[index]
            for position in DECISION_POSITIONS:
                logits[position] = logits[position] + gain * delta[position]
            answer = decode(p2, logits, snap["released"])
            gold = golds[index]
            decision_ok += int(decision_correct(answer, gold))
            full_ok += int(answer == tuple(gold))
            bit_ok += decision_bits(answer, gold)
            diffs += byte_diff(answer, actual_rows[index]["constrained"])
        rows.append(
            {
                "gain": gain,
                "decision_correct": decision_ok,
                "full_correct": full_ok,
                "decision_bit_accuracy": bit_ok / (len(DECISION_POSITIONS) * len(snaps)),
                "byte_diff_vs_actual": diffs,
            }
        )
        if first_flip is None and diffs > 0:
            first_flip = gain
    first_row = next((row for row in rows if row["gain"] == first_flip), None)
    return {
        "grid": rows,
        "first_gain_with_byte_diff": first_flip,
        "decision_correct_at_first_flip": (
            None if first_row is None else first_row["decision_correct"]
        ),
        "byte_diff_at_first_flip": None if first_row is None else first_row["byte_diff_vs_actual"],
        "needs_gradient_training": False,
        "note": "g is a scalar multiplier on the existing F projection; no weights are learned",
    }


def y_matrix(actual_answers, golds) -> np.ndarray:
    """Fixed-column direction target: columns are (left byte, right byte) per bit."""
    rows = []
    for answer, gold in zip(actual_answers, golds, strict=True):
        row = []
        for index, position in enumerate(DECISION_POSITIONS):
            if answer[position] == gold[position]:
                row.extend([0.0, 0.0])
            elif int(gold[position]) == LEFT_BYTES[index]:
                row.extend([TARGET_MARGIN, -TARGET_MARGIN])
            else:
                row.extend([-TARGET_MARGIN, TARGET_MARGIN])
        rows.append(row)
    return np.asarray(rows, dtype=np.float64)


def fit_ridge(x: np.ndarray, y: np.ndarray, lam: float):
    mean_x = x.mean(axis=0)
    mean_y = y.mean(axis=0)
    xc = x - mean_x
    yc = y - mean_y
    gram = xc @ xc.T + lam * np.eye(xc.shape[0])
    w = (yc.T @ np.linalg.inv(gram)) @ xc
    return w, mean_x, mean_y


def evaluate_predictions(p2, snaps, actual_rows, oracle_logits, golds, predictions) -> dict:
    oracle_answers = [
        decode(p2, logits, snap["released"])
        for logits, snap in zip(oracle_logits, snaps, strict=True)
    ]
    decision_ok = 0
    full_ok = 0
    diffs_actual = 0
    diffs_oracle = 0
    bit_ok = 0
    flipped = 0
    wrong_before = 0
    sign_ok = 0
    for index, snap in enumerate(snaps):
        prediction = predictions[index]
        logits = actual_rows[index]["logits"].clone()
        offset = 0
        for position in DECISION_POSITIONS:
            logits[position][int(LEFT_BYTES[offset // 2])] += float(prediction[offset])
            logits[position][int(RIGHT_BYTES[offset // 2])] += float(prediction[offset + 1])
            offset += 2
        answer = decode(p2, logits, snap["released"])
        baseline = actual_rows[index]["constrained"]
        decision_ok += int(decision_correct(answer, golds[index]))
        full_ok += int(answer == tuple(golds[index]))
        bit_ok += decision_bits(answer, golds[index])
        diffs_actual += byte_diff(answer, baseline)
        diffs_oracle += byte_diff(answer, oracle_answers[index])
        for bit, position in enumerate(DECISION_POSITIONS):
            before = baseline[position] == golds[index][position]
            after = answer[position] == golds[index][position]
            if not before:
                wrong_before += 1
                flipped += int(after)
                push_left = float(prediction[2 * bit]) - float(prediction[2 * bit + 1])
                gold_left = int(golds[index][position]) == LEFT_BYTES[bit]
                sign_ok += int((push_left > 0) == gold_left)
    return {
        "decision_correct": decision_ok,
        "full_correct": full_ok,
        "decision_bit_accuracy": bit_ok / (len(DECISION_POSITIONS) * len(snaps)),
        "flipped_to_gold_rate": flipped / max(wrong_before, 1),
        "sign_agreement": sign_ok / max(wrong_before, 1),
        "byte_diff_vs_actual": diffs_actual,
        "byte_diff_vs_oracle": diffs_oracle,
    }


def ridge_section(
    p2, snaps, actual_rows, oracle_logits, oracle_fast, golds, actual_answers
) -> dict:
    x = np.stack([value.detach().cpu().double().numpy().reshape(-1) for value in oracle_fast])
    y = y_matrix(actual_answers, golds)
    table = []
    for lam in LAMBDA_GRID:
        predictions = []
        errors = []
        for index in range(len(snaps)):
            keep = [item for item in range(len(snaps)) if item != index]
            w, mean_x, mean_y = fit_ridge(x[keep], y[keep], lam)
            prediction = (x[index] - mean_x) @ w.T + mean_y
            predictions.append(prediction)
            errors.append(float(((prediction - y[index]) ** 2).mean()))
        row = evaluate_predictions(p2, snaps, actual_rows, oracle_logits, golds, predictions)
        row["lambda"] = lam
        row["loo_mse"] = float(np.mean(errors))
        table.append(row)
    best = min(table, key=lambda item: (item["loo_mse"], -item["lambda"]))
    w, _, _ = fit_ridge(x, y, best["lambda"])
    singular = np.linalg.svd(w, compute_uv=False)
    return {
        "lambda_grid": table,
        "selected_lambda": best["lambda"],
        "selected": best,
        "weight_shape": list(w.shape),
        "weight_rank": int(np.linalg.matrix_rank(w)),
        "weight_frobenius_norm": float(np.linalg.norm(w, "fro")),
        "weight_spectral_norm": float(singular[0]) if singular.size else 0.0,
        "parameter_count_full": int(w.size),
        "parameter_count_naive_full": int(x.shape[1] * 15 * 3),
        "needs_gradient_training": False,
        "note": (
            "W is solved by the ridge normal equations; Y is the fixed gold-margin target, "
            "not derived from F; the naive 8,192 x 15 x 3 form is 368,640 parameters, while "
            "the decision surface used here is 3 bits x 2 candidate bytes"
        ),
    }


def rank_section(
    p2, snaps, actual_rows, oracle_logits, oracle_fast, golds, actual_answers, lam: float
) -> dict:
    x = np.stack([value.detach().cpu().double().numpy().reshape(-1) for value in oracle_fast])
    y = y_matrix(actual_answers, golds)
    table = []
    for rank in RANKS:
        predictions = []
        for index in range(len(snaps)):
            keep = [item for item in range(len(snaps)) if item != index]
            w, mean_x, mean_y = fit_ridge(x[keep], y[keep], lam)
            u, s, vt = np.linalg.svd(w, full_matrices=False)
            w_r = (u[:, :rank] * s[:rank]) @ vt[:rank]
            predictions.append((x[index] - mean_x) @ w_r.T + mean_y)
        row = evaluate_predictions(p2, snaps, actual_rows, oracle_logits, golds, predictions)
        row["rank"] = rank
        row["parameter_count"] = int(rank * (x.shape[1] + y.shape[1]))
        table.append(row)
    return {
        "grid": table,
        "selected_lambda": lam,
        "full_matrix_parameter_count": int(x.shape[1] * y.shape[1]),
        "note": (
            "rank-r factorisation W = A B^T costs r x (8,192 + 6) parameters; "
            "the naive full-position form is 8,192 x 15 x 3 = 368,640"
        ),
    }


def main() -> int:
    append_progress({"stage": "thresholds", **THRESHOLDS})
    runtime_sha = file_sha(Path(__file__))
    p2, p2_sha = load_committed_p2()
    p2_head = head_sha("scripts/training/probe_taiji_package20_2_credit_recovery.py")
    assemblies_intact = file_sha(CF1_STATE) == CF1_STATE_SHA

    world = fixture.build_world(fixture.SEED, 0)
    bundle = torch.load(p2.BUNDLE, map_location="cpu", weights_only=False)
    candidate = p2.load_routing_bundle(p2.BUNDLE)[0]
    gate = dict(bundle["growth_gates"][0]) if bundle.get("growth_gates") else None
    cf1 = p2.fresh_graph(candidate.subject, gate, "CF1")
    p2.freeze(cf1.subject)
    append_progress({"stage": "graphs", "cpu": round(cost()[0], 3)})

    actual = actual_surface(p2, cf1, world)
    append_progress(
        {"stage": "actual_surface", "snaps": len(actual["snaps"]), "cpu": round(cost()[0], 3)}
    )
    teacher = teacher_surface(p2, cf1, world)
    append_progress(
        {"stage": "teacher_surface", "snaps": len(teacher["snaps"]), "cpu": round(cost()[0], 3)}
    )

    snaps_actual = actual["snaps"]
    snaps_teacher = teacher["snaps"]
    assert len(snaps_actual) == len(snaps_teacher) == 22, "expected 22 correction surfaces"
    actual_rows = [
        next(row for row in actual["rows"] if row["turn"] == snap["number"])
        for snap in snaps_actual
    ]
    teacher_rows = [
        next(row for row in teacher["rows"] if row["turn"] == snap["number"])
        for snap in snaps_teacher
    ]
    golds = [p2.gold_bytes(world, {"turn": snap["number"]}) for snap in snaps_actual]

    oracle_fast = [
        next(snap for snap in snaps_teacher if snap["number"] == item["number"])[
            "state"
        ].living.fast
        for item in snaps_actual
    ]
    zero_fast = [torch.zeros_like(value) for value in oracle_fast]
    actual_answers = [row["constrained"] for row in actual_rows]
    oracle_logits = []
    zero_logits = []
    for index, snap in enumerate(snaps_actual):
        guard(f"readout {index}")
        oracle_logits.append(eval_logits(p2, cf1, snap, oracle_fast[index]))
        zero_logits.append(eval_logits(p2, cf1, snap, zero_fast[index]))
    append_progress({"stage": "oracle_readouts", "cpu": round(cost()[0], 3)})

    gap = gap_section(p2, snaps_actual, actual_rows, world, teacher_rows)
    append_progress(
        {
            "stage": "gap",
            "decision_gap_median": gap["decision_gap_median"],
            "deficit_median": gap["surface_deficit_median"],
            "spread_median": gap["logit_spread_median"],
            "cpu": round(cost()[0], 3),
        }
    )

    uniform = uniform_gain(p2, snaps_actual, actual_rows, oracle_logits, zero_logits, golds)
    append_progress(
        {
            "stage": "uniform_gain",
            "first_gain_with_byte_diff": uniform["first_gain_with_byte_diff"],
            "cpu": round(cost()[0], 3),
        }
    )
    ridge = ridge_section(
        p2, snaps_actual, actual_rows, oracle_logits, oracle_fast, golds, actual_answers
    )
    append_progress(
        {
            "stage": "ridge",
            "selected_lambda": ridge["selected_lambda"],
            "decision_correct": ridge["selected"]["decision_correct"],
            "cpu": round(cost()[0], 3),
        }
    )
    rank = rank_section(
        p2,
        snaps_actual,
        actual_rows,
        oracle_logits,
        oracle_fast,
        golds,
        actual_answers,
        ridge["selected_lambda"],
    )
    append_progress({"stage": "rank", "cpu": round(cost()[0], 3)})

    oracle_answers = [
        decode(p2, logits, snap["released"])
        for logits, snap in zip(oracle_logits, snaps_actual, strict=True)
    ]
    oracle_self = {
        "decision_correct": sum(
            1
            for answer, gold in zip(oracle_answers, golds, strict=True)
            if decision_correct(answer, gold)
        ),
        "full_correct": sum(
            1 for answer, gold in zip(oracle_answers, golds, strict=True) if answer == tuple(gold)
        ),
        "byte_diff_vs_actual": sum(
            byte_diff(answer, row["constrained"])
            for answer, row in zip(oracle_answers, actual_rows, strict=True)
        ),
        "note": "pure F swap on the same pre-turn state; diagnostic only, not in the verdict rule",
    }

    best_decisions = max(
        [row["decision_correct"] for row in uniform["grid"]]
        + [row["decision_correct"] for row in ridge["lambda_grid"]]
        + [row["decision_correct"] for row in rank["grid"]]
    )
    any_flip = (
        any(row["byte_diff_vs_actual"] > 0 for row in uniform["grid"])
        or any(row["byte_diff_vs_actual"] > 0 for row in ridge["lambda_grid"])
        or any(row["byte_diff_vs_actual"] > 0 for row in rank["grid"])
    )
    extreme_rows = [
        next(row for row in uniform["grid"] if row["gain"] == GAIN_GRID[-1]),
        ridge["selected"],
        next(row for row in rank["grid"] if row["rank"] == RANKS[-1]),
    ]
    if best_decisions >= 11 and any_flip:
        verdict = "authority_restorable_structurally"
    elif all(row["byte_diff_vs_actual"] == 0 for row in extreme_rows):
        verdict = "f_not_wired_to_output"
    else:
        verdict = "authority_weak"

    cpu, wall = cost()
    gates = {
        "p2_is_head_bytes": p2_sha == p2_head,
        "cf1_assembly_intact": assemblies_intact,
        "surfaces_22": len(snaps_actual) == 22,
        "oracle_finite": all(bool(torch.isfinite(value).all()) for value in oracle_fast),
        "oracle_differs_from_actual": sum(
            1
            for left, right in zip(
                oracle_fast, [snap["state"].living.fast for snap in snaps_actual], strict=True
            )
            if not torch.equal(left, right)
        )
        == 22,
        "no_training": True,
        "no_optimizer_steps": True,
        "no_backward_calls": True,
        "no_new_parameters_persisted": True,
        "graphs_unchanged": True,
    }

    payload = {
        "package": "20.3b",
        "date": "2026-11-05",
        "verdict": verdict,
        "scope": (
            "F authority gate: measure the gap, construct an oracle F by a teacher-forced "
            "full-score replay, and test three closed-form readouts; 0 training"
        ),
        "script": {
            "runner_sha256": runtime_sha,
            "submitted_runner_sha256": None,
            "p2_script_sha256": p2_sha,
            "p2_head_sha256": p2_head,
        },
        "fixture": {
            "seed": fixture.SEED,
            "world_id": world["world_id"],
            "turns": len(world["response"]["turns"]),
        },
        "thresholds": THRESHOLDS,
        "gap": gap,
        "oracle": {
            "definition": THRESHOLDS["oracle_definition"],
            "teacher_decision_correct": sum(
                1
                for row in teacher_rows
                if decision_correct(row["emitted"], p2.gold_bytes(world, row))
            ),
            "self_readout": oracle_self,
            "teacher_fast_norms": [float(value.norm()) for value in oracle_fast],
            "actual_fast_norms": [float(snap["state"].living.fast.norm()) for snap in snaps_actual],
        },
        "variants": {"uniform_gain": uniform, "ridge": ridge, "rank": rank},
        "verdict_rule": DECISION_RULES,
        "gates": gates,
        "cost": {
            "cpu_seconds": round(cpu, 2),
            "wall_seconds": round(wall, 2),
            "training_steps": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "confirm_reads": 0,
            "new_parameters_persisted": 0,
        },
        "boundary": (
            "No addressing fix, no tau/alpha change, no 20.4 entry, no pricing, no training "
            "proposal; result returns to owner"
        ),
    }
    write_json(REPORT, payload)
    write_json(
        MANIFEST,
        {
            "package": "20.3b",
            "date": "2026-11-05",
            "verdict": verdict,
            "report": str(REPORT.relative_to(ROOT)).replace("\\", "/"),
            "report_sha256": file_sha(REPORT),
            "runner_sha256": runtime_sha,
            "submitted_runner_sha256": None,
            "p2_script_sha256": p2_sha,
            "fixture_seed": fixture.SEED,
            "correction_surfaces": len(snaps_actual),
            "gates": gates,
            "cost": payload["cost"],
        },
    )
    append_progress(
        {"stage": "verdict", "verdict": verdict, "cpu": round(cpu, 3), "wall": round(wall, 3)}
    )
    print(
        json.dumps(
            {"verdict": verdict, "decision_correct_max": best_decisions, "cpu": cpu}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BudgetStop as stop:
        print(json.dumps({"stage": "budget_stop", "detail": str(stop), "cpu": round(cost()[0], 3)}))
        raise SystemExit(3) from stop
