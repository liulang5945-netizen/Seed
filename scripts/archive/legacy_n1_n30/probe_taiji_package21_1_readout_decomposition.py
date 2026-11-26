"""Package 21.1: zero-training readout/emission decomposition.

Loads the four frozen 20.4c_r2 final checkpoints and evaluates three arms on
the same dev worlds:

  natural          -- unchanged decoding
  template_forced  -- feed gold back only at the 9 template positions
  content_forced   -- feed gold back only at the 18 content positions

No optimizer, no backward, no checkpoint write, no confirm world.
"""

from __future__ import annotations

import gc
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "training"))

import probe_taiji_package20_4c_r2_formal_comparison as r2  # noqa: E402
import torch  # noqa: E402

REPORT = ROOT / "reports/taiji_package21_1_readout_decomposition_20261107.json"
MANIFEST = ROOT / "plans/manifests/taiji_package21_1_readout_decomposition_20261107.json"
PROGRESS = ROOT / "output/taiji_package21_1_readout_decomposition/progress.jsonl"

DEV_INDICES = tuple(range(160, 240, 5))  # 160,165,...,235 -> 16 worlds
if max(DEV_INDICES) >= 240:
    raise RuntimeError("dev indices must not touch the sealed confirm range 240-319")
MODES = ("natural", "template_forced", "content_forced")
CHECKPOINT_ARM_SEEDS = (
    ("CF1", 20261105),
    ("CF1", 20261106),
    ("B_EH", 20261105),
    ("B_EH", 20261106),
)
CAP_CPU = 4800.0
STOP_CPU = 4320.0
MAX_WALL = 7200.0

START_CPU = time.process_time()
START_WALL = time.time()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_progress(row: dict) -> None:
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with PROGRESS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        import os

        os.fsync(handle.fileno())


def cost() -> tuple[float, float]:
    return time.process_time() - START_CPU, time.time() - START_WALL


def gold_matrix(worlds: list[dict], number: int) -> torch.Tensor:
    rows = []
    for world in worlds:
        raw = r2.gold_bytes(world, number)
        if len(raw) != r2.ANSWER_BYTES:
            raise RuntimeError(f"gold length {len(raw)} != {r2.ANSWER_BYTES}")
        rows.append([int(byte) for byte in raw])
    return torch.tensor(rows, dtype=torch.long)


def forced_turn(
    core, graph, states, turns, released_per_world, const, parent, gates, table, mode, gold
):
    reference = core.W
    batch = len(turns)
    numbers = {int(turn["turn"]) for turn in turns}
    if len(numbers) != 1:
        raise RuntimeError(f"turn numbers differ: {sorted(numbers)}")
    now = numbers.pop()
    queries = [turn["query"].encode("utf-8") for turn in turns]
    records = [graph._records(released, now) for released in released_per_world]
    allowed = graph._allowed(False).unsqueeze(0).expand(batch, -1, -1)
    states, view, goal = r2.batched_observe(
        core, graph, states, queries, queries, records, now, allowed, const, parent, gates
    )
    shift = r2.shift_index(view.keys, reference)
    forced_positions: tuple[int, ...] = ()
    if mode == "template_forced":
        forced_positions = r2.TEMPLATE_POSITIONS
    elif mode == "content_forced":
        forced_positions = r2.DECISION_POSITIONS

    previous = torch.full((batch,), r2.EOS, dtype=torch.long)
    emitted = []
    for position in range(r2.ANSWER_BYTES):
        output = r2.step_batch(
            core,
            states,
            event=table[previous],
            goal_vector=goal,
            view=view,
            align=None,
            previous_byte=previous,
            allowed=allowed,
            const=const,
            parent_units=parent,
            gate_rows=gates,
        )
        choice = torch.argmax(output.probabilities[:, :256], dim=1)
        emitted.append(choice)
        fed = choice.clone()
        if position in forced_positions:
            fed = gold[:, position].clone()
        states = r2.consume_batch(output.state, output, fed, shift)
        previous = fed
    return states, torch.stack(emitted, 1)


def run_pair(core, graph, worlds, const, parent, gates, table, mode):
    responses = [world["response"] for world in worlds]
    states = r2.initial_batched_state(core, tuple(response["world_id"] for response in responses))
    released = [[] for _ in responses]
    metrics = {
        "decision_correct": 0,
        "surfaces": 0,
        "template_bits": 0,
        "template_total": 0,
        "content_bits": 0,
        "content_total": 0,
        "full_bytes": 0,
        "turns": 0,
        "fast_writes": 0,
        "correction_correct": 0,
        "correction_surfaces": 0,
        "post_feedback_correct": 0,
        "post_feedback_first": 0,
    }
    for index in range(r2.TOTAL_TURNS):
        turns = [response["turns"][index] for response in responses]
        number = int(turns[0]["turn"])
        gold_rows = gold_matrix(worlds, number)
        states, emitted = forced_turn(
            core, graph, states, turns, released, const, parent, gates, table, mode, gold_rows
        )
        for world_index, world in enumerate(worlds):
            gold = r2.gold_bytes(world, number)
            row = [int(value) for value in emitted[world_index]]
            decision = all(row[p] == gold[p] for p in r2.DECISION_POSITIONS)
            metrics["decision_correct"] += int(decision)
            metrics["surfaces"] += 1
            if turns[world_index]["correction_surface"]:
                metrics["correction_surfaces"] += 1
                metrics["correction_correct"] += int(decision)
            if turns[world_index]["first_answer"] and released[world_index]:
                metrics["post_feedback_first"] += 1
                metrics["post_feedback_correct"] += int(decision)
            metrics["template_bits"] += sum(int(row[p] == gold[p]) for p in r2.TEMPLATE_POSITIONS)
            metrics["template_total"] += len(r2.TEMPLATE_POSITIONS)
            metrics["content_bits"] += sum(int(row[p] == gold[p]) for p in r2.DECISION_POSITIONS)
            metrics["content_total"] += len(r2.DECISION_POSITIONS)
            metrics["full_bytes"] += int(tuple(row) == tuple(gold))
            metrics["turns"] += 1
        now = int(turns[0]["turn"])
        release_rows = [
            next(
                (release for release in response["releases"] if int(release["after_turn"]) == now),
                None,
            )
            for response in responses
        ]
        if release_rows[0] is not None:
            states, count = r2.batched_release(
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
            metrics["fast_writes"] += int(count.sum())
            for world_index, _response in enumerate(responses):
                released[world_index].extend(
                    {
                        "source_id": item["source_id"],
                        "feedback": item["feedback"],
                        "version": item["version"],
                        "after_turn": int(turns[world_index]["turn"]),
                    }
                    for item in release_rows[world_index]["items"]
                )
    return metrics


def accuracy(accum: dict) -> dict:
    def ratio(numerator: int, denominator: int):
        return None if denominator == 0 else round(numerator / denominator, 6)

    return {
        "decision_accuracy": ratio(accum["decision_correct"], accum["surfaces"]),
        "template_bit_accuracy": ratio(accum["template_bits"], accum["template_total"]),
        "content_bit_accuracy": ratio(accum["content_bits"], accum["content_total"]),
        "full_byte_accuracy": ratio(accum["full_bytes"], accum["turns"]),
        "correction_accuracy": ratio(accum["correction_correct"], accum["correction_surfaces"]),
        "post_feedback_accuracy": ratio(
            accum["post_feedback_correct"], accum["post_feedback_first"]
        ),
        "surfaces": accum["surfaces"],
        "turns": accum["turns"],
        "fast_writes": accum["fast_writes"],
    }


def main() -> int:
    runner_sha = sha256_file(Path(__file__).resolve())
    bundle = torch.load(r2.BUNDLE, map_location="cpu", weights_only=False)
    candidate = r2.load_routing_bundle(r2.BUNDLE)[0]
    gate = dict(bundle["growth_gates"][0]) if bundle.get("growth_gates") else None
    append_progress(
        {
            "stage": "protocol",
            "runner_sha256": runner_sha,
            "dev_indices": list(DEV_INDICES),
            "modes": list(MODES),
            "cap_cpu": CAP_CPU,
            "worlds_per_cell": len(DEV_INDICES),
        }
    )

    cells = []
    for arm, seed in CHECKPOINT_ARM_SEEDS:
        path = r2.CHECKPOINTS / f"{arm}_{seed}_200.pt"
        if not path.exists():
            raise RuntimeError(f"missing checkpoint {path}")
        checkpoint_sha = sha256_file(path)
        payload = r2.load_checkpoint(path)
        if payload["arm"] != arm or int(payload["seed"]) != seed:
            raise RuntimeError("checkpoint identity mismatch")
        graph = r2.build_graph(candidate, gate, arm)
        graph.subject.load_state_dict(payload["subject"])
        graph.subject.eval()
        const = r2.update_constants(graph.subject)
        parent, gates = r2.growth_rows(
            graph.subject, graph.step_kwargs.get("growth_gates"), graph.subject.W
        )
        table = r2.patch_table(graph.subject, graph.subject.W)
        worlds = [r2.world_at(seed, index) for index in DEV_INDICES]
        for mode in MODES:
            started = time.process_time()
            accum = {
                "decision_correct": 0,
                "surfaces": 0,
                "template_bits": 0,
                "template_total": 0,
                "content_bits": 0,
                "content_total": 0,
                "full_bytes": 0,
                "turns": 0,
                "fast_writes": 0,
                "correction_correct": 0,
                "correction_surfaces": 0,
                "post_feedback_correct": 0,
                "post_feedback_first": 0,
            }
            with torch.no_grad():
                for start in range(0, len(worlds), r2.BATCH):
                    used, wall = cost()
                    if used > STOP_CPU or wall > MAX_WALL:
                        raise RuntimeError(f"budget_stop cpu={used:.3f} wall={wall:.1f}")
                    metrics = run_pair(
                        graph.subject,
                        graph,
                        worlds[start : start + r2.BATCH],
                        const,
                        parent,
                        gates,
                        table,
                        mode,
                    )
                    for key in accum:
                        accum[key] += metrics[key]
            cell = {
                "arm": arm,
                "seed": seed,
                "mode": mode,
                "checkpoint": str(path.relative_to(ROOT)),
                "checkpoint_sha256": checkpoint_sha,
                **accuracy(accum),
                "cpu": round(time.process_time() - started, 3),
            }
            cells.append(cell)
            append_progress({"stage": "cell", **cell, "total_cpu": round(cost()[0], 3)})
            del accum, metrics
            gc.collect()
        del graph, const, parent, gates, table, worlds, payload
        gc.collect()

    by_key = {(cell["arm"], cell["seed"], cell["mode"]): cell for cell in cells}
    summary = []
    r_cells = 0
    u_cells = 0
    for arm, seed in CHECKPOINT_ARM_SEEDS:
        natural = by_key[(arm, seed, "natural")]
        template = by_key[(arm, seed, "template_forced")]
        content = by_key[(arm, seed, "content_forced")]
        gain = (template["content_bit_accuracy"] or 0.0) - (natural["content_bit_accuracy"] or 0.0)
        is_r = bool(
            (template["content_bit_accuracy"] or 0.0) >= 0.60
            and (template["full_byte_accuracy"] or 0.0) >= 0.50
        )
        is_u = bool((template["content_bit_accuracy"] or 0.0) < 0.40 and gain < 0.05)
        r_cells += int(is_r)
        u_cells += int(is_u)
        summary.append(
            {
                "arm": arm,
                "seed": seed,
                "natural_content": natural["content_bit_accuracy"],
                "template_forced_content": template["content_bit_accuracy"],
                "template_forced_full": template["full_byte_accuracy"],
                "content_gain_from_template": round(gain, 6),
                "content_forced_template": content["template_bit_accuracy"],
                "content_forced_full": content["full_byte_accuracy"],
                "branch_r": is_r,
                "branch_u": is_u,
            }
        )
    if r_cells >= 3:
        branch = "R"
        verdict = "emission_readout_is_first_wall"
    elif u_cells >= 3:
        branch = "U"
        verdict = "upstream_representation_is_first_wall"
    else:
        branch = "I"
        verdict = "inconclusive_single_point_diagnosis_required"

    used_cpu, used_wall = cost()
    report = {
        "schema": "taiji-rule-learning-21-1-readout-decomposition-v1",
        "evidence": "21_1",
        "date": "2026-11-07",
        "status": "completed",
        "verdict": verdict,
        "branch": branch,
        "detail": (
            "three zero-training arms on four frozen 20.4c_r2 final checkpoints; "
            "template-forced feeds gold only at template positions, content-forced only at content positions"
        ),
        "discipline": {
            "training_steps": 0,
            "optimizer_steps": 0,
            "adam_steps": 0,
            "new_probes": 0,
            "evaluation_reruns": 0,
            "confirm_reads": 0,
            "checkpoints_written": 0,
            "frozen_modules_touched": False,
            "worlds_per_cell": len(DEV_INDICES),
            "dev_indices": list(DEV_INDICES),
            "dev_index_range": [DEV_INDICES[0], DEV_INDICES[-1]],
            "cpu_cap": CAP_CPU,
            "cpu_stop_threshold": STOP_CPU,
            "max_wall": MAX_WALL,
            "cpu_seconds": round(used_cpu, 3),
            "wall_seconds": round(used_wall, 3),
            "write_then_rename": True,
            "progress_append_fsync": True,
        },
        "periods": {
            "checkpoints": [str(r2.CHECKPOINTS.relative_to(ROOT))],
            "checkpoint_step": 200,
            "arms": [arm for arm, _ in CHECKPOINT_ARM_SEEDS],
            "seeds": sorted({seed for _, seed in CHECKPOINT_ARM_SEEDS}),
            "runner_sha256": runner_sha,
            "submitted_runner_sha256": runner_sha,
        },
        "criteria": {
            "R": "at least 3/4 cells: template_forced content_bit_accuracy >= 0.60 and full_byte_accuracy >= 0.50",
            "U": "at least 3/4 cells: template_forced content_bit_accuracy < 0.40 and gain over natural < 0.05",
            "I": "otherwise",
            "frozen_before_run": True,
        },
        "branch_counts": {"R": r_cells, "U": u_cells, "I": 4 - r_cells - u_cells},
        "cells": cells,
        "summary": summary,
        "confirm": {"released": False, "read_record": 0, "indices": [240, 319]},
    }
    write_json(REPORT, report)
    manifest = {
        "schema": "taiji-rule-learning-21-1-readout-decomposition-manifest-v1",
        "date": "2026-11-07",
        "report": str(REPORT.relative_to(ROOT)),
        "report_sha256": sha256_file(REPORT),
        "runner": str(Path(__file__).resolve().relative_to(ROOT)),
        "submitted_runner_sha256": runner_sha,
        "runner_sha256": runner_sha,
        "verdict": verdict,
        "branch": branch,
        "cpu_seconds": round(used_cpu, 3),
        "wall_seconds": round(used_wall, 3),
        "confirm_reads": 0,
        "checkpoints_written": 0,
    }
    write_json(MANIFEST, manifest)
    append_progress(
        {"stage": "report", "verdict": verdict, "branch": branch, "cpu": round(used_cpu, 3)}
    )
    print(json.dumps({"verdict": verdict, "branch": branch, "cpu": round(used_cpu, 3)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
