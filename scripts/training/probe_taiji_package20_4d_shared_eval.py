"""20.4d: the zero-training shared evaluation, arm x batch decoupled.

The four dev readings of 20.4c_r2 are not directly comparable across arms: each
run evaluated worlds from its own seed (``world_at(seed, DEV_INDICES)``), so an
arm difference and a batch difference are the same number.  This probe spends no
training to separate them.

* A. Checkerboard: the four 200-update checkpoints are each scored on both
  existing dev batches (seed 1105 and seed 1106, indices 160-239).  The four
  diagonal cells are the dev evals already produced by 20.4c_r2 and are reused
  after verifying that the checkpoint's weight digest is byte-identical; the
  four off-diagonal cells are computed here.
* B. Shared batch: a new seed (20261107) with indices 320-399, never used for
  training, dev or confirm, scored by all four checkpoints on identical worlds
  in identical order.  Per-world pairing gives the CF1-B_EH delta with a
  world-level bootstrap 95% CI.

Nothing trains: no optimizer, no backward, no Adam, no checkpoint is written, no
confirm world is read, and the frozen graph modules are untouched.

The verdict is pre-registered before the first world is scored: the criteria
block is written to the report file first and the runner refuses to start if the
file's criteria no longer match the constants below.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for entry in (ROOT, HERE):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import torch  # noqa: E402

THREADS = 1
torch.set_num_threads(THREADS)
with contextlib.suppress(RuntimeError):
    torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True, warn_only=True)

import probe_taiji_package20_4c_r2_formal_comparison as r2  # noqa: E402

from taiji.rule_learning_20_1_net import DECISION_POSITIONS  # noqa: E402
from taiji.selection_routing_checkpoint import load_routing_bundle  # noqa: E402

REPORT = ROOT / "reports/taiji_package20_4d_shared_eval_20261106.json"
MANIFEST = ROOT / "plans/manifests/taiji_package20_4d_shared_eval_20261106.json"
PROGRESS = ROOT / "reports/taiji_package20_4d_shared_eval_20261106_progress.jsonl"
PREREG = ROOT / "reports/taiji_package20_4d_preregistration_20261106.json"
R2_PROGRESS = ROOT / "reports/taiji_package20_4c_r2_formal_comparison_20261105_progress.jsonl"

CPU_CAP = 6000.0
CPU_GUARD = 5920.0
CHECKPOINT_STEP = 200
DEV_INDICES = tuple(range(160, 240))
SHARED_SEED = 20261107
SHARED_INDICES = tuple(range(320, 400))
CONFIRM_INDICES = (240, 319)
BOOTSTRAP_ROUNDS = 2000
BOOTSTRAP_SEED = 20261105
DETERMINISM_SAMPLES = 8
ARMS = ("CF1", "B_EH")
SEEDS = (20261105, 20261106)
BATCH = 2

START_WALL = time.time()
START_CPU = time.process_time()

CRITERIA = {
    "arm_effective": (
        "the paired CF1-B_EH delta on the shared batch has a bootstrap 95% CI that "
        "excludes 0, |delta| >= 0.02 for decision accuracy, and the sign agrees on "
        "both existing dev batches"
    ),
    "arm_indistinguishable": (
        "the CI contains 0, or the two existing dev batches disagree in sign"
    ),
    "batch_dominant": (
        "the cross-batch same-arm spread is materially larger than the same-batch "
        "arm gap; the ratio is reported and 'materially' means > 2.0x"
    ),
    "metrics": (
        "decision accuracy is the pre-registered primary metric; template bits, "
        "content bits and full-byte accuracy are reported separately and do not "
        "override the primary verdict"
    ),
    "frozen": (
        "criteria fixed before the first world is scored; the runner aborts if the "
        "pre-registration file differs from this block"
    ),
}


class BudgetStop(RuntimeError):
    def __init__(self, verdict: str, detail: str) -> None:
        super().__init__(detail)
        self.verdict = verdict
        self.detail = detail


def cost() -> tuple[float, float]:
    return time.process_time() - START_CPU, time.time() - START_WALL


def guard(stage: str) -> None:
    cpu, wall = cost()
    if cpu > CPU_GUARD:
        raise BudgetStop("budget_cap_reached", f"CPU cap at {stage}: {cpu:.1f} of {CPU_CAP:.0f}")


def append_progress(row: dict) -> None:
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with PROGRESS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def file_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def world_sha(world: dict) -> str:
    return canonical_sha(world["response"])


def empty_world_accum() -> dict:
    return {
        "surfaces": 0,
        "decision_correct": 0,
        "template_bits": 0,
        "template_total": 0,
        "content_bits": 0,
        "content_total": 0,
        "full_bytes": 0,
        "turns": 0,
        "correction_correct": 0,
        "correction_surfaces": 0,
        "post_feedback_correct": 0,
        "post_feedback_first": 0,
    }


def evaluate(graph, worlds: list[dict]) -> list[dict]:
    """Score ``worlds`` in the given order, one row of raw counts per world."""
    core = graph.subject
    const = r2.update_constants(core)
    parent, gates = r2.growth_rows(core, graph.step_kwargs.get("growth_gates"), core.W)
    table = r2.patch_table(core, core.W)
    responses = [world["response"] for world in worlds]
    states = r2.initial_batched_state(core, tuple(response["world_id"] for response in responses))
    released = [[] for _ in responses]
    rows = [empty_world_accum() for _ in worlds]
    traces = [hashlib.sha256() for _ in worlds]
    with torch.no_grad():
        for index in range(r2.TOTAL_TURNS):
            turns = [response["turns"][index] for response in responses]
            states, _logits, emitted = r2.batched_turn(
                core, graph, states, turns, released, const, parent, gates, table
            )
            for position, world in enumerate(worlds):
                number = int(turns[position]["turn"])
                gold = r2.gold_bytes(world, number)
                answer = [int(value) for value in emitted[position]]
                decision = all(answer[p] == gold[p] for p in DECISION_POSITIONS)
                row = rows[position]
                row["surfaces"] += 1
                row["turns"] += 1
                row["decision_correct"] += int(decision)
                row["template_bits"] += sum(
                    int(answer[p] == gold[p]) for p in r2.TEMPLATE_POSITIONS
                )
                row["template_total"] += len(r2.TEMPLATE_POSITIONS)
                row["content_bits"] += sum(int(answer[p] == gold[p]) for p in DECISION_POSITIONS)
                row["content_total"] += len(DECISION_POSITIONS)
                row["full_bytes"] += int(tuple(answer) == tuple(gold))
                if turns[position]["correction_surface"]:
                    row["correction_surfaces"] += 1
                    row["correction_correct"] += int(decision)
                if turns[position]["first_answer"] and released[position]:
                    row["post_feedback_first"] += 1
                    row["post_feedback_correct"] += int(decision)
                traces[position].update(emitted[position].to(torch.int64).numpy().tobytes())
            now = int(turns[0]["turn"])
            release_rows = [
                next(
                    (
                        release
                        for release in response["releases"]
                        if int(release["after_turn"]) == now
                    ),
                    None,
                )
                for response in responses
            ]
            if release_rows[0] is not None:
                states, _count = r2.batched_release(
                    core,
                    graph,
                    states,
                    release_rows,
                    released,
                    now,
                    const,
                    parent,
                    gates,
                    table,
                    bool(graph.fast_enabled),
                )
                for position, _response in enumerate(responses):
                    released[position].extend(
                        {
                            "source_id": item["source_id"],
                            "feedback": item["feedback"],
                            "version": item["version"],
                            "after_turn": int(turns[position]["turn"]),
                        }
                        for item in release_rows[position]["items"]
                    )
    for row, trace in zip(rows, traces, strict=True):
        row["trace_sha256"] = trace.hexdigest()
    return rows


def summarise(rows: list[dict]) -> dict:
    surfaces = sum(row["surfaces"] for row in rows)
    turns = sum(row["turns"] for row in rows)
    template_total = sum(row["template_total"] for row in rows)
    content_total = sum(row["content_total"] for row in rows)
    correction = sum(row["correction_surfaces"] for row in rows)
    post_first = sum(row["post_feedback_first"] for row in rows)
    return {
        "worlds": len(rows),
        "surfaces": surfaces,
        "decision_correct": sum(row["decision_correct"] for row in rows),
        "decision_accuracy": sum(row["decision_correct"] for row in rows) / max(surfaces, 1),
        "template_bit_accuracy": sum(row["template_bits"] for row in rows) / max(template_total, 1),
        "content_bit_accuracy": sum(row["content_bits"] for row in rows) / max(content_total, 1),
        "full_byte_accuracy": sum(row["full_bytes"] for row in rows) / max(turns, 1),
        "correction_accuracy": sum(row["correction_correct"] for row in rows) / max(correction, 1),
        "correction_surfaces": correction,
        "post_feedback_accuracy": sum(row["post_feedback_correct"] for row in rows)
        / max(post_first, 1),
        "post_feedback_first": post_first,
        "trace_root": canonical_sha([row["trace_sha256"] for row in rows]),
    }


def per_world_series(rows: list[dict], key: str, denominator: str) -> list[float]:
    return [row[key] / max(row[denominator], 1) for row in rows]


def bootstrap_ci(deltas: list[float]) -> dict:
    import random

    rng = random.Random(BOOTSTRAP_SEED)
    count = len(deltas)
    means = []
    for _ in range(BOOTSTRAP_ROUNDS):
        total = 0.0
        for _ in range(count):
            total += deltas[rng.randrange(count)]
        means.append(total / count)
    means.sort()
    observed = sum(deltas) / max(count, 1)
    return {
        "rounds": BOOTSTRAP_ROUNDS,
        "worlds": count,
        "mean_delta": round(observed, 6),
        "ci_low": round(means[int(0.025 * BOOTSTRAP_ROUNDS)], 6),
        "ci_high": round(means[int(0.975 * BOOTSTRAP_ROUNDS) - 1], 6),
        "excludes_zero": not (
            means[int(0.025 * BOOTSTRAP_ROUNDS)] <= 0 <= means[int(0.975 * BOOTSTRAP_ROUNDS) - 1]
        ),
    }


def paired(rows_a: list[dict], rows_b: list[dict], key: str, denominator: str) -> dict:
    left = per_world_series(rows_a, key, denominator)
    right = per_world_series(rows_b, key, denominator)
    deltas = [a - b for a, b in zip(left, right, strict=True)]
    return bootstrap_ci(deltas)


def two_way_variance(cells: dict[tuple[str, int], float]) -> dict:
    """Crude SS split of the 2x2x2 checkerboard: arm, batch, interaction, residual."""
    values = list(cells.values())
    grand = sum(values) / len(values)
    ss_arm = 0.0
    ss_batch = 0.0
    ss_interaction = 0.0
    ss_total = sum((value - grand) ** 2 for value in values)
    for arm in ARMS:
        arm_values = [cells[(arm, seed)] for seed in SEEDS]
        ss_arm += len(arm_values) * (sum(arm_values) / len(arm_values) - grand) ** 2
    for seed in SEEDS:
        seed_values = [cells[(arm, seed)] for arm in ARMS]
        ss_batch += len(seed_values) * (sum(seed_values) / len(seed_values) - grand) ** 2
    for arm in ARMS:
        for seed in SEEDS:
            cell = cells[(arm, seed)]
            arm_mean = sum(cells[(arm, other)] for other in SEEDS) / len(SEEDS)
            batch_mean = sum(cells[(other, seed)] for other in ARMS) / len(ARMS)
            ss_interaction += (cell - arm_mean - batch_mean + grand) ** 2
    residual = max(ss_total - ss_arm - ss_batch - ss_interaction, 0.0)
    scale = ss_total if ss_total > 0 else 1.0
    return {
        "ss_total": round(ss_total, 8),
        "arm_share": round(ss_arm / scale, 4),
        "batch_share": round(ss_batch / scale, 4),
        "interaction_share": round(ss_interaction / scale, 4),
        "residual_share": round(residual / scale, 4),
        "note": "crude sum-of-squares split of the eight cell means, no model fitted",
    }


def load_model(candidate, gate, arm: str, seed: int):
    graph = r2.build_graph(candidate, gate, arm)
    path = r2.checkpoint_path(arm, seed, CHECKPOINT_STEP)
    if not path.exists():
        raise BudgetStop("missing_checkpoint", f"no checkpoint at {path}")
    payload = r2.load_checkpoint(path)
    if int(payload["updates"]) != CHECKPOINT_STEP:
        raise BudgetStop("missing_checkpoint", f"{path} is at update {payload['updates']}")
    graph.subject.load_state_dict(payload["subject"])
    digest = r2.weights_digest(graph.subject)
    del payload
    return graph, path, digest


def run() -> int:
    if not PREREG.exists():
        raise SystemExit(f"pre-registration missing: {PREREG}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    if canonical_sha(prereg["criteria"]) != canonical_sha(CRITERIA):
        raise SystemExit("pre-registration criteria changed; refusing to run")
    if not REPORT.exists():
        r2.write_json(
            REPORT,
            {
                "schema": "taiji-rule-learning-20-4d-shared-eval-v1",
                "evidence": "20_4d",
                "status": "pre_registered",
                "preregistration": {
                    "path": str(PREREG.relative_to(ROOT)),
                    "sha256": file_sha(PREREG),
                    "criteria_sha256": canonical_sha(CRITERIA),
                },
                "criteria": CRITERIA,
            },
        )
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text("", encoding="utf-8")
    bundle = torch.load(r2.BUNDLE, map_location="cpu", weights_only=False)
    candidate = load_routing_bundle(r2.BUNDLE)[0]
    gate = dict(bundle["growth_gates"][0]) if bundle.get("growth_gates") else None
    del bundle

    r2_rows = [
        json.loads(line)
        for line in R2_PROGRESS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    diagonal = {
        (row["arm"], int(row["seed"])): row for row in r2_rows if row.get("stage") == "dev_eval"
    }
    batches = {seed: [r2.world_at(seed, index) for index in DEV_INDICES] for seed in SEEDS}
    shared = [r2.world_at(SHARED_SEED, index) for index in SHARED_INDICES]
    shared_sha = canonical_sha([world_sha(world) for world in shared])

    append_progress(
        {
            "stage": "protocol",
            "threads": THREADS,
            "training_steps": 0,
            "optimizer_steps": 0,
            "checkpoint_step": CHECKPOINT_STEP,
            "dev_indices": [DEV_INDICES[0], DEV_INDICES[-1]],
            "shared_seed": SHARED_SEED,
            "shared_indices": [SHARED_INDICES[0], SHARED_INDICES[-1]],
            "shared_world_root": shared_sha,
            "confirm_read": False,
            "cpu_cap": CPU_CAP,
            "cpu": round(cost()[0], 3),
        }
    )

    checkerboard: dict[str, dict] = {}
    shared_rows: dict[str, list[dict]] = {}
    details = {}
    stop = {"verdict": "shared_eval_completed", "detail": "all cells scored"}
    try:
        for arm in ARMS:
            for seed in SEEDS:
                key = f"{arm}|{seed}"
                guard(f"{key}/load")
                graph, path, digest = load_model(candidate, gate, arm, seed)
                details[key] = {
                    "checkpoint": str(path.relative_to(ROOT)),
                    "checkpoint_sha256": file_sha(path),
                    "weights_sha256": digest,
                }
                cells = {}
                for batch_seed in SEEDS:
                    if batch_seed == seed:
                        row = diagonal.get((arm, seed))
                        match = bool(row) and row.get("weights") == digest
                        cells[str(batch_seed)] = {
                            "source": "reused_dev_eval_20_4c_r2" if match else "recomputed",
                            "weights_digest_match": match,
                            "metrics": row if match else None,
                        }
                        if match:
                            continue
                    guard(f"{key}/batch{batch_seed}")
                    started = time.process_time()
                    rows = evaluate(graph, batches[batch_seed])
                    summary = summarise(rows)
                    cells[str(batch_seed)] = {
                        "source": "computed_20_4d",
                        "weights_digest_match": True,
                        "metrics": summary,
                        "world_rows": {str(DEV_INDICES[i]): rows[i] for i in range(len(rows))},
                    }
                    append_progress(
                        {
                            "stage": "cell",
                            "arm": arm,
                            "seed": seed,
                            "batch_seed": batch_seed,
                            "decision_accuracy": summary["decision_accuracy"],
                            "template_bit_accuracy": summary["template_bit_accuracy"],
                            "content_bit_accuracy": summary["content_bit_accuracy"],
                            "correction_accuracy": summary["correction_accuracy"],
                            "post_feedback_accuracy": summary["post_feedback_accuracy"],
                            "cell_cpu": round(time.process_time() - started, 3),
                            "cpu": round(cost()[0], 3),
                        }
                    )
                guard(f"{key}/shared")
                started = time.process_time()
                shared_rows[key] = evaluate(graph, shared)
                shared_summary = summarise(shared_rows[key])
                append_progress(
                    {
                        "stage": "shared_cell",
                        "arm": arm,
                        "seed": seed,
                        "worlds": shared_summary["worlds"],
                        "decision_accuracy": shared_summary["decision_accuracy"],
                        "template_bit_accuracy": shared_summary["template_bit_accuracy"],
                        "content_bit_accuracy": shared_summary["content_bit_accuracy"],
                        "full_byte_accuracy": shared_summary["full_byte_accuracy"],
                        "correction_accuracy": shared_summary["correction_accuracy"],
                        "post_feedback_accuracy": shared_summary["post_feedback_accuracy"],
                        "cell_cpu": round(time.process_time() - started, 3),
                        "cpu": round(cost()[0], 3),
                    }
                )
                checkerboard[key] = {
                    "arm": arm,
                    "seed": seed,
                    "cells": cells,
                    "shared": shared_summary,
                }
                del graph
    except BudgetStop as fired:
        stop = {"verdict": fired.verdict, "detail": fired.detail}

    if stop["verdict"] == "shared_eval_completed":
        guard("determinism")
        deterministic = determinism_check(candidate, gate, shared)
        payload = assemble(
            checkerboard,
            shared_rows,
            details,
            deterministic,
            shared_sha,
            stop,
        )
    else:
        payload = assemble(checkerboard, shared_rows, details, None, shared_sha, stop)
    emit(payload, stop["verdict"], stop["detail"])
    print(json.dumps({"verdict": stop["verdict"], "detail": stop["detail"]}, ensure_ascii=False))
    return 0


def determinism_check(candidate, gate, shared: list[dict]) -> dict:
    """Repeat a sample of worlds twice; the emitted-byte traces must be identical."""
    samples = shared[:DETERMINISM_SAMPLES]
    rows = []
    for arm, seed in (("CF1", SEEDS[0]), ("B_EH", SEEDS[0])):
        graph, _path, _digest = load_model(candidate, gate, arm, seed)
        first = evaluate(graph, samples)
        guard("determinism/repeat")
        second = evaluate(graph, samples)
        for index, (left, right) in enumerate(zip(first, second, strict=True)):
            rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "world": samples[index]["world_id"],
                    "identical": left["trace_sha256"] == right["trace_sha256"],
                }
            )
        del graph
    return {
        "samples": len(rows),
        "all_identical": all(row["identical"] for row in rows),
        "rows": rows,
    }


def assemble(checkerboard, shared_rows, details, deterministic, shared_sha, stop) -> dict:
    def metric(cell: dict, name: str):
        metrics = cell.get("metrics") or {}
        return metrics.get(name)

    checkerboard_table = {}
    for key, entry in checkerboard.items():
        checkerboard_table[key] = {
            "arm": entry["arm"],
            "seed": entry["seed"],
            "by_batch": {
                batch: {
                    "source": entry["cells"][batch]["source"],
                    "weights_digest_match": entry["cells"][batch]["weights_digest_match"],
                    "decision_accuracy": metric(entry["cells"][batch], "decision_accuracy"),
                    "template_bit_accuracy": metric(entry["cells"][batch], "template_bit_accuracy"),
                    "content_bit_accuracy": metric(entry["cells"][batch], "content_bit_accuracy"),
                    "full_byte_accuracy": metric(entry["cells"][batch], "full_byte_accuracy"),
                    "correction_accuracy": metric(entry["cells"][batch], "correction_accuracy"),
                    "post_feedback_accuracy": metric(
                        entry["cells"][batch], "post_feedback_accuracy"
                    ),
                }
                for batch in ("20261105", "20261106")
            },
        }
    variance = {}
    cells = {}
    for entry in checkerboard.values():
        for batch, cell in entry["cells"].items():
            value = metric(cell, "decision_accuracy")
            if value is not None:
                cells[(entry["arm"], int(batch))] = value
    if len(cells) == 4:
        variance["decision_accuracy"] = two_way_variance(cells)

    paired_shared = {}
    paired_batch = {}
    if len(shared_rows) == 4:
        cf1 = shared_rows["CF1|20261105"]
        beh = shared_rows["B_EH|20261105"]
        paired_shared["seed_20261105"] = {
            "decision": paired(cf1, beh, "decision_correct", "surfaces"),
            "template": paired(cf1, beh, "template_bits", "template_total"),
            "content": paired(cf1, beh, "content_bits", "content_total"),
            "full_byte": paired(cf1, beh, "full_bytes", "turns"),
            "correction": paired(cf1, beh, "correction_correct", "correction_surfaces"),
            "post_feedback": paired(cf1, beh, "post_feedback_correct", "post_feedback_first"),
        }
        cf1b = shared_rows["CF1|20261106"]
        behb = shared_rows["B_EH|20261106"]
        paired_shared["seed_20261106"] = {
            "decision": paired(cf1b, behb, "decision_correct", "surfaces"),
            "template": paired(cf1b, behb, "template_bits", "template_total"),
            "content": paired(cf1b, behb, "content_bits", "content_total"),
            "full_byte": paired(cf1b, behb, "full_bytes", "turns"),
            "correction": paired(cf1b, behb, "correction_correct", "correction_surfaces"),
            "post_feedback": paired(cf1b, behb, "post_feedback_correct", "post_feedback_first"),
        }
        pooled_left = [row["decision_correct"] / max(row["surfaces"], 1) for row in cf1 + cf1b]
        pooled_right = [row["decision_correct"] / max(row["surfaces"], 1) for row in beh + behb]
        paired_shared["pooled"] = {
            "decision": bootstrap_ci(
                [a - b for a, b in zip(pooled_left, pooled_right, strict=True)]
            )
        }
        # same-batch arm gaps on the two existing dev batches (aggregate difference)
        for batch in ("20261105", "20261106"):
            values = {
                arm: checkerboard[f"{arm}|{batch}"]["cells"][batch].get("metrics") for arm in ARMS
            }
            if all(values.values()):
                paired_batch[batch] = {
                    "cf1_minus_b_eh_decision": round(
                        values["CF1"]["decision_accuracy"] - values["B_EH"]["decision_accuracy"],
                        6,
                    ),
                    "cf1_minus_b_eh_content": round(
                        values["CF1"]["content_bit_accuracy"]
                        - values["B_EH"]["content_bit_accuracy"],
                        6,
                    ),
                    "note": "aggregate difference on identical worlds, no per-world pairing",
                }

    cross_batch = {}
    for key, entry in checkerboard.items():
        first = entry["cells"]["20261105"].get("metrics") or {}
        second = entry["cells"]["20261106"].get("metrics") or {}
        if first and second:
            cross_batch[key] = {
                "decision_spread": round(
                    first["decision_accuracy"] - second["decision_accuracy"], 6
                ),
                "content_spread": round(
                    first["content_bit_accuracy"] - second["content_bit_accuracy"], 6
                ),
            }

    verdict = {
        "arm_effective": False,
        "arm_indistinguishable": True,
        "batch_dominant": False,
        "detail": "not evaluated",
    }
    if len(paired_shared) == 3 and len(paired_batch) == 2:
        pooled = paired_shared["pooled"]["decision"]
        signs = [paired_batch[batch]["cf1_minus_b_eh_decision"] > 0 for batch in paired_batch]
        effective = (
            pooled["excludes_zero"] and abs(pooled["mean_delta"]) >= 0.02 and len(set(signs)) == 1
        )
        indistinguishable = (not pooled["excludes_zero"]) or len(set(signs)) > 1
        arm_gap = abs(pooled["mean_delta"])
        batch_spread = max(abs(cross_batch[key]["decision_spread"]) for key in cross_batch)
        verdict = {
            "arm_effective": bool(effective),
            "arm_indistinguishable": bool(indistinguishable),
            "batch_dominant": bool(arm_gap > 0 and batch_spread / arm_gap > 2.0),
            "arm_gap": round(arm_gap, 6),
            "batch_spread": round(batch_spread, 6),
            "batch_over_arm_ratio": round(batch_spread / arm_gap, 4) if arm_gap else None,
            "sign_agreement_across_batches": len(set(signs)) == 1,
            "detail": (
                f"pooled delta {pooled['mean_delta']:+.4f} "
                f"CI [{pooled['ci_low']:+.4f}, {pooled['ci_high']:+.4f}]; "
                f"batch/arm spread ratio "
                f"{(batch_spread / arm_gap) if arm_gap else float('inf'):.2f}"
            ),
        }

    return {
        "schema": "taiji-rule-learning-20-4d-shared-eval-v1",
        "evidence": "20_4d",
        "status": "completed",
        "preregistration": {
            "path": str(PREREG.relative_to(ROOT)),
            "sha256": file_sha(PREREG),
            "criteria_sha256": canonical_sha(CRITERIA),
            "written_before_first_world": True,
        },
        "criteria": CRITERIA,
        "design": {
            "training_steps": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "checkpoints_written": 0,
            "checkpoint_step": CHECKPOINT_STEP,
            "dev_batches": {"20261105": [160, 239], "20261106": [160, 239]},
            "shared_batch": {
                "seed": SHARED_SEED,
                "indices": [SHARED_INDICES[0], SHARED_INDICES[-1]],
                "worlds": len(SHARED_INDICES),
                "world_root_sha256": shared_sha,
                "disjoint_from_train_dev_confirm": True,
            },
            "diagonal_reuse": (
                "the four diagonal cells are the 20.4c_r2 dev evals, reused only when "
                "the checkpoint weight digest matches the digest recorded in that row"
            ),
        },
        "checkerboard": checkerboard_table,
        "variance": variance,
        "paired_shared_batch": paired_shared,
        "paired_existing_batches": paired_batch,
        "cross_batch_spread": cross_batch,
        "determinism": deterministic,
        "verdict": verdict,
        "stop": stop,
        "confirm_read": False,
        "addressed_store_enabled": False,
        "cost": {
            "cpu_seconds": round(cost()[0], 3),
            "wall_seconds": round(cost()[1], 3),
            "cpu_cap": CPU_CAP,
            "remaining_cpu_seconds": round(CPU_CAP - cost()[0], 3),
        },
        "script": {
            "runner_sha256": file_sha(Path(__file__)),
            "submitted_runner_sha256": None,
            "reused_runner_sha256": file_sha(
                ROOT / "scripts/training/probe_taiji_package20_4c_r2_formal_comparison.py"
            ),
            "frozen_module_sha256": {
                "taiji/rule_learning_20_1_net.py": file_sha(
                    ROOT / "taiji/rule_learning_20_1_net.py"
                ),
                "taiji/living_core.py": file_sha(ROOT / "taiji/living_core.py"),
            },
        },
        "inputs": details,
    }


def emit(payload: dict, verdict: str, detail: str) -> None:
    payload["verdict_label"] = verdict
    payload["detail"] = detail
    r2.write_json(REPORT, payload)
    r2.write_json(
        MANIFEST,
        {
            "schema": "taiji-rule-learning-20-4d-shared-eval-manifest-v1",
            "evidence": "20_4d",
            "verdict": verdict,
            "detail": detail,
            "report_path": str(REPORT.relative_to(ROOT)),
            "report_sha256": file_sha(REPORT),
            "preregistration_path": str(PREREG.relative_to(ROOT)),
            "preregistration_sha256": file_sha(PREREG),
            "runner_sha256": payload["script"]["runner_sha256"],
            "submitted_runner_sha256": None,
            "cpu_seconds": payload["cost"]["cpu_seconds"],
            "cpu_cap": CPU_CAP,
            "completed": verdict == "shared_eval_completed",
        },
    )


if __name__ == "__main__":
    raise SystemExit(run())
