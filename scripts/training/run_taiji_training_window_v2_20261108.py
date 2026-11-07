"""Training window v2: the admitted bounded training run.

Executes the frozen pre-registration (plans/manifests/
taiji_training_window_v2_prereg_20261108.json, sha fdb975a7): 2 seeds x 2 arms
x 200 updates on the redesigned F write path (content keys + tied reads +
content-conditioned values + per-turn write timing), same parent checkpoint,
same data order per seed, arms differing only in the F feedback path.

Per run: checkpoints every 25 steps (save->load bitwise verified at each
save), one mid-run resume verification (update recomputed on a fresh graph
from the step-100 checkpoint, loss bitwise), dev evaluation (16 worlds,
deployed semantics) at step0 and step200, budget hard stop, NaN guard.

The pooled decision-bit delta (CF1_v2 - B_EH_v2) is judged against the
package-20 frozen contribution gates (>= 0.05 with paired world-level
bootstrap lower bound > 0, both seeds same direction).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "training"))

import torch  # noqa: E402

torch.set_num_threads(1)

import probe_taiji_fec1_w1a_retest_20261108 as w1a  # noqa: E402,F401  (installs V2Capture)
import probe_taiji_fec1_wv2_retest_20261108 as wv2  # noqa: E402,F401  (same side effect)
import probe_taiji_package20_2s_f_information as p2s  # noqa: E402
import probe_taiji_package20_4c_r2_formal_comparison as r2  # noqa: E402

from taiji.rule_learning_20_1_net import (  # noqa: E402
    enable_content_path_v2,
    enable_write_timing_per_turn,
)

PREREG = ROOT / "plans/manifests/taiji_training_window_v2_prereg_20261108.json"
PREREG_SHA256 = "b2db40018e89bffdd9b443b3b413cab5d1b479229c2d21834ee99d01b2366d12"
REPORT = ROOT / "reports/taiji_training_window_v2_run_20261108.json"
EXEC_MANIFEST = ROOT / "plans/manifests/taiji_training_window_v2_run_execution_20261108.json"
PROGRESS = ROOT / "reports/taiji_training_window_v2_run_20261108_progress.jsonl"
CHECKPOINTS = ROOT / "output/taiji_training_window_v2_checkpoints"
CENTROID_CACHE = ROOT / "output/taiji_training_window_v2_centroid.pt"

SEEDS = (20261105, 20261106)
ARMS = ("CF1", "B_EH")
CPU_CAP = 34816.0
CPU_GUARD = 34000.0
BOOTSTRAP_SEED = 20261108
BOOTSTRAP_RESAMPLES = 2000

START_CPU = time.process_time()
START_WALL = time.time()
candidate_ref: tuple = (None, None)


class BudgetStop(RuntimeError):
    pass


class EngineeringStop(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def append_progress(record: dict) -> None:
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with PROGRESS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def cost() -> tuple[float, float]:
    return time.process_time() - START_CPU, time.time() - START_WALL


def check_budget() -> None:
    cpu, wall = cost()
    if cpu >= CPU_GUARD:
        raise BudgetStop(f"budget guard at cpu={cpu:.2f} (cap {CPU_CAP})")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def training_centroid() -> torch.Tensor:

    counts = []
    for index in r2.TRAIN_INDICES:
        world = r2.world_at(20261105, index)
        for turn in world["response"]["turns"]:
            vector = torch.zeros(128, dtype=torch.float64)
            for value in turn["query"].encode("utf-8"):
                vector[int(value) % 128] += 1.0
            counts.append(vector)
    return torch.stack(counts).mean(0)


def build_v2_graph(candidate, gate, arm: str, centroid):
    graph = r2.build_graph(candidate, gate, arm)
    graph.subject.eval()
    enable_content_path_v2(graph, seed=20261031, centroid=centroid)
    enable_write_timing_per_turn(graph)
    graph.subject.eval()
    return graph


def dev_worlds():
    return [r2.world_at(20261105, index) for index in r2.DEV_INDICES[::5]]


def dev_eval(graph, worlds) -> dict:
    """Deployed-semantics inference over 16 dev worlds; decision-bit accuracy
    of the argmax answers on correction faces, plus UTF-8 legality."""
    predicted, golds, utf8_ok, utf8_total = [], [], 0, 0
    for world in worlds:
        state = graph.initial_state(world["response"]["world_id"])
        released: list[dict] = []
        for turn in world["response"]["turns"]:
            check_budget()
            number = int(turn["turn"])
            result = graph.run_turn(
                state, turn, released, write_enabled=True, vocab=None, now=number
            )
            state = result["state"]
            if bool(turn["correction_surface"]):
                answer = bytes(result["bytes"])
                gold = world["scoring"]["turns"][str(number)]["expected_action"].encode("utf-8")
                predicted.append(answer)
                golds.append(gold)
                try:
                    answer.decode("utf-8")
                    utf8_ok += 1
                except UnicodeDecodeError:
                    pass
                utf8_total += 1
            for release in world["response"]["releases"]:
                if int(release["after_turn"]) != number:
                    continue
                state, _info = graph.release(
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
    scored = p2s.score_bytes(predicted, golds)
    return {
        "decision_bit_accuracy": float(scored["decision_bit_accuracy"]),
        "decisions_correct": int(scored["decisions_correct"]),
        "faces": len(predicted),
        "utf8_legal_fraction": utf8_ok / max(utf8_total, 1),
        "per_face_bits": [
            sum(1 for p in p2s.DECISION_POSITIONS if a[p] == g[p])
            for a, g in zip(predicted, golds, strict=True)
        ],
    }


def run_training(graph, optimizer, seed: int, run_name: str, centroid) -> dict:
    """One run: 200 updates, checkpoints every 25 (bitwise save-verified),
    one mid-run resume verification at step 100, dev eval at step0/step200."""
    dev_worlds_list = dev_worlds()
    step0_dev = dev_eval(graph, dev_worlds_list)
    append_progress(
        {
            "stage": "dev0",
            "run": run_name,
            "acc": step0_dev["decision_bit_accuracy"],
            "cpu": round(cost()[0], 3),
        }
    )
    losses, grad_norms = [], []
    cursor = 0
    resume_check = None
    for update in range(1, r2.UPDATES_PER_RUN + 1):
        check_budget()
        worlds = [
            r2.world_at(seed, r2.TRAIN_INDICES[(cursor + offset) % len(r2.TRAIN_INDICES)])
            for offset in range(r2.BATCH)
        ]
        const = r2.update_constants(graph.subject)
        parent_rows, gates_rows = r2.growth_rows(
            graph.subject, graph.step_kwargs.get("growth_gates"), graph.subject.W
        )
        table = r2.patch_table(graph.subject, graph.subject.W)
        graph.subject.zero_grad(set_to_none=True)
        loss_value, metrics = r2.run_pair(
            graph.subject, graph, worlds, const, parent_rows, gates_rows, table, train=True
        )
        if not bool(torch.isfinite(loss_value)):
            raise EngineeringStop(f"non-finite loss at {run_name} update {update}")
        loss_value.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(r2.trainer_params(graph), r2.GRAD_CLIP))
        optimizer.step()
        cursor += r2.BATCH
        losses.append(float(loss_value.detach()))
        grad_norms.append(grad_norm)
        if update % 25 == 0:
            payload = {
                "schema": "taiji-training-window-v2-checkpoint-v1",
                "run": run_name,
                "arm": graph.arm,
                "seed": seed,
                "updates": update,
                "cursor": cursor,
                "subject": graph.subject.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng": torch.get_rng_state(),
            }
            path = CHECKPOINTS / f"{run_name}_step{update}.pt"
            torch.save(payload, path)
            reloaded = torch.load(path, map_location="cpu", weights_only=True)
            worst = 0.0
            for name, tensor in payload["subject"].items():
                back = reloaded["subject"][name]
                if tensor.numel():
                    worst = max(worst, float((tensor - back).abs().max()))
            if worst != 0.0:
                raise EngineeringStop(
                    f"checkpoint roundtrip diff {worst} at {run_name} step {update}"
                )
            # mid-run resume verification once, at step 100: a fresh graph
            # loaded from the checkpoint must reproduce the next update's loss
            if update == 100:
                fresh = r2.build_graph(candidate_ref[0], candidate_ref[1], graph.arm)
                r2.unfreeze_outer(fresh.subject)
                fresh.subject.load_state_dict(reloaded["subject"])
                fresh_optimizer = torch.optim.Adam(r2.trainer_params(fresh), lr=r2.LEARNING_RATE)
                fresh_optimizer.load_state_dict(reloaded["optimizer"])
                torch.set_rng_state(reloaded["rng"])
                fresh_graph_obj = fresh
                const = r2.update_constants(fresh.subject)
                parent_rows, gates_rows = r2.growth_rows(
                    fresh.subject, fresh.step_kwargs.get("growth_gates"), fresh.subject.W
                )
                table = r2.patch_table(fresh.subject, fresh.subject.W)
                fresh.subject.zero_grad(set_to_none=True)
                resume_worlds = [
                    r2.world_at(seed, r2.TRAIN_INDICES[(cursor + offset) % len(r2.TRAIN_INDICES)])
                    for offset in range(r2.BATCH)
                ]
                loss_resume, _m = r2.run_pair(
                    fresh.subject,
                    fresh_graph_obj,
                    resume_worlds,
                    const,
                    parent_rows,
                    gates_rows,
                    table,
                    train=True,
                )
                loss_resume.backward()
                torch.nn.utils.clip_grad_norm_(r2.trainer_params(fresh), r2.GRAD_CLIP)
                fresh_optimizer.step()
                # the in-session graph runs the SAME next update for comparison;
                # reset the global RNG to the checkpoint state first so both
                # sides start the update from identical RNG (the fresh side
                # already consumed from that state)
                torch.set_rng_state(reloaded["rng"])
                const = r2.update_constants(graph.subject)
                parent_rows, gates_rows = r2.growth_rows(
                    graph.subject, graph.step_kwargs.get("growth_gates"), graph.subject.W
                )
                table = r2.patch_table(graph.subject, graph.subject.W)
                graph.subject.zero_grad(set_to_none=True)
                loss_here, _m = r2.run_pair(
                    graph.subject,
                    graph,
                    resume_worlds,
                    const,
                    parent_rows,
                    gates_rows,
                    table,
                    train=True,
                )
                loss_here.backward()
                torch.nn.utils.clip_grad_norm_(r2.trainer_params(graph), r2.GRAD_CLIP)
                optimizer.step()
                resume_check = {
                    "at_step": 100,
                    "loss_resume": float(loss_resume.detach()),
                    "loss_here": float(loss_here.detach()),
                    "match": bool(
                        abs(float(loss_resume.detach()) - float(loss_here.detach())) == 0.0
                    ),
                }
                cursor += r2.BATCH
                if not resume_check["match"]:
                    raise EngineeringStop(f"resume verification failed: {resume_check}")
            append_progress(
                {
                    "stage": "checkpoint",
                    "run": run_name,
                    "step": update,
                    "roundtrip": worst,
                    "cpu": round(cost()[0], 3),
                }
            )
        if update % 10 == 0:
            append_progress(
                {
                    "stage": "update",
                    "run": run_name,
                    "step": update,
                    "loss": round(losses[-1], 6),
                    "grad_norm": round(grad_norm, 6),
                    "cpu": round(cost()[0], 3),
                }
            )
    step200_dev = dev_eval(graph, dev_worlds_list)
    append_progress(
        {
            "stage": "dev200",
            "run": run_name,
            "acc": step200_dev["decision_bit_accuracy"],
            "cpu": round(cost()[0], 3),
        }
    )
    return {
        "run": run_name,
        "arm": graph.arm,
        "seed": seed,
        "step0_dev": step0_dev,
        "step200_dev": step200_dev,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_mean": sum(losses) / len(losses),
        "grad_norm_mean": sum(grad_norms) / len(grad_norms),
        "resume_check": resume_check,
        "updates": r2.UPDATES_PER_RUN,
    }


def paired_bootstrap(deltas_by_world: list[list[float]], resamples: int) -> dict:
    """World-level paired bootstrap over the 16 dev worlds, pooling both seeds
    (each world contributes its per-seed delta; the statistic is the mean)."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    world_means = np.array([float(np.mean(d)) for d in deltas_by_world])
    point = float(world_means.mean())
    boots = []
    for _ in range(resamples):
        chosen = rng.integers(0, len(world_means), size=len(world_means))
        boots.append(float(world_means[chosen].mean()))
    return {
        "point": point,
        "mean": float(np.mean(boots)),
        "percentile_2_5": float(np.percentile(boots, 2.5)),
        "percentile_97_5": float(np.percentile(boots, 97.5)),
        "resamples": resamples,
        "seed": BOOTSTRAP_SEED,
        "resampling_unit": "dev worlds (pooled over seeds)",
    }


def main() -> int:
    runner_sha = sha256_file(Path(__file__).resolve())
    try:
        if sha256_file(PREREG) != PREREG_SHA256:
            raise EngineeringStop("prereg manifest sha mismatch")
    except EngineeringStop as error:
        append_progress({"stage": "preflight", "ok": False, "reason": str(error)})
        write_json(
            REPORT,
            {
                "package": "training-window-v2",
                "status": "engineering_stop",
                "verdict": "engineering_stop",
                "reason": str(error),
                "runner_sha256": runner_sha,
            },
        )
        print(json.dumps({"verdict": "engineering_stop", "reason": str(error)}))
        return 0

    append_progress(
        {
            "stage": "start",
            "prereg_sha256": PREREG_SHA256,
            "runner_sha256": runner_sha,
            "cpu": round(cost()[0], 3),
        }
    )
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    centroid = training_centroid()
    torch.save(centroid, CENTROID_CACHE)
    parent_payload = torch.load(r2.BUNDLE, map_location="cpu", weights_only=True)
    gates_list = parent_payload.get("growth_gates")
    gate = dict(gates_list[0]) if gates_list else None
    candidate = r2.load_routing_bundle(r2.BUNDLE)[0]
    global candidate_ref
    candidate_ref = (candidate, gate)

    runs: list[dict] = []
    try:
        for seed in SEEDS:
            for arm in ARMS:
                check_budget()
                run_name = f"{arm}_{seed}"
                graph = build_v2_graph(candidate, gate, arm, centroid)
                r2.unfreeze_outer(graph.subject)
                optimizer = torch.optim.Adam(r2.trainer_params(graph), lr=r2.LEARNING_RATE)
                result = run_training(graph, optimizer, seed, run_name, centroid)
                result["checkpoint_sha256"] = {
                    p.name: sha256_file(p) for p in sorted(CHECKPOINTS.glob(f"{run_name}_*.pt"))
                }
                runs.append(result)
                append_progress(
                    {
                        "stage": "run_done",
                        "run": run_name,
                        "dev0": result["step0_dev"]["decision_bit_accuracy"],
                        "dev200": result["step200_dev"]["decision_bit_accuracy"],
                        "cpu": round(cost()[0], 3),
                    }
                )
    except BudgetStop as error:
        append_progress({"stage": "budget_stop", "reason": str(error)})
        write_json(
            REPORT,
            {
                "package": "training-window-v2",
                "status": "budget_stop",
                "verdict": "inconclusive",
                "reason": str(error),
                "runs": runs,
                "cost": {"cpu_seconds": round(cost()[0], 2)},
            },
        )
        print(json.dumps({"verdict": "inconclusive", "status": "budget_stop"}))
        return 0
    except EngineeringStop as error:
        append_progress({"stage": "engineering_stop", "reason": str(error)})
        write_json(
            REPORT,
            {
                "package": "training-window-v2",
                "status": "engineering_stop",
                "verdict": "engineering_stop",
                "reason": str(error),
                "runner_sha256": runner_sha,
                "runs": runs,
            },
        )
        print(json.dumps({"verdict": "engineering_stop", "reason": str(error)}))
        return 0

    # ---- aggregation: pooled paired delta (CF1_v2 - B_EH_v2) over 16 worlds,
    # pooling both seeds; world-level paired bootstrap; direction consistency
    by_key = {(r["arm"], r["seed"]): r for r in runs}
    deltas_by_world = []
    per_seed_delta = {}
    for seed in SEEDS:
        cf1 = by_key[("CF1", seed)]["step200_dev"]["per_face_bits"]
        beh = by_key[("B_EH", seed)]["step200_dev"]["per_face_bits"]
        # face bits are pooled across the 16 dev worlds in file order; the
        # world index = face_index // 22 within each run's face list
        n_faces = min(len(cf1), len(beh))
        n_worlds = n_faces // 22
        seed_deltas = []
        for w in range(n_worlds):
            slice_cf = cf1[w * 22 : (w + 1) * 22]
            slice_be = beh[w * 22 : (w + 1) * 22]
            acc_cf = sum(slice_cf) / (22 * len(p2s.DECISION_POSITIONS))
            acc_be = sum(slice_be) / (22 * len(p2s.DECISION_POSITIONS))
            seed_deltas.append(acc_cf - acc_be)
        per_seed_delta[seed] = float(np.mean(seed_deltas))
        deltas_by_world.append(seed_deltas)
    bootstrap = paired_bootstrap(deltas_by_world, BOOTSTRAP_RESAMPLES)
    pooled_delta = bootstrap["point"]
    directions = [1 if d > 0 else (-1 if d < 0 else 0) for d in per_seed_delta.values()]
    same_direction = len(set(directions)) == 1
    utf8_ok = all(r["step200_dev"]["utf8_legal_fraction"] >= 1.0 for r in runs)
    if pooled_delta >= 0.05 and bootstrap["percentile_2_5"] > 0 and same_direction and utf8_ok:
        verdict = "f_feedback_learning_established"
    elif pooled_delta <= 0.0 and bootstrap["percentile_97_5"] <= 0.0:
        verdict = "f_feedback_negative"
    elif (
        abs(pooled_delta) <= 1e-9
        and bootstrap["percentile_2_5"] <= 0.0 <= bootstrap["percentile_97_5"]
    ):
        verdict = "zero_identifiability"
    else:
        verdict = "inconclusive"
    cpu, wall = cost()
    write_json(
        REPORT,
        {
            "schema": "taiji-training-window-v2-run-v1",
            "package": "training-window-v2",
            "date": "2026-11-08",
            "status": "completed",
            "verdict": verdict,
            "verdict_basis": "package-20 frozen contribution gates: pooled delta >= 0.05 AND paired bootstrap LB > 0 AND both seeds same direction; UTF-8/NaN safety",
            "pooled_delta": pooled_delta,
            "per_seed_delta": {str(k): v for k, v in per_seed_delta.items()},
            "bootstrap": bootstrap,
            "same_direction": same_direction,
            "gates": {
                "pooled_delta_ge_0.05": bool(pooled_delta >= 0.05),
                "bootstrap_lb_gt_0": bool(bootstrap["percentile_2_5"] > 0),
                "same_direction": same_direction,
                "utf8_legal": utf8_ok,
            },
            "runs": runs,
            "discipline": {
                "confirm_reads": 0,
                "optimizer_updates_total": 4 * r2.UPDATES_PER_RUN,
                "criteria_frozen_before": PREREG_SHA256,
            },
            "provenance": {
                "prereg_sha256": PREREG_SHA256,
                "runner_sha256": runner_sha,
                "centroid_sha256": sha256_file(CENTROID_CACHE),
            },
            "cost": {"cpu_seconds": round(cpu, 3), "wall_seconds": round(wall, 3), "cap": CPU_CAP},
            "scope": (
                "A pass here is a LEARNING-GAIN input on the fixture under the "
                "redesigned path; not a capability claim, not P2/L3."
            ),
        },
    )
    write_json(
        EXEC_MANIFEST,
        {
            "schema": "taiji-training-window-v2-run-execution-manifest-v1",
            "package": "training-window-v2",
            "date": "2026-11-08",
            "prereg": "plans/manifests/taiji_training_window_v2_prereg_20261108.json",
            "prereg_sha256": PREREG_SHA256,
            "report": "reports/taiji_training_window_v2_run_20261108.json",
            "report_sha256": sha256_file(REPORT),
            "runner": "scripts/training/run_taiji_training_window_v2_20261108.py",
            "runner_sha256": runner_sha,
            "submitted_runner_sha256": runner_sha,
            "verdict": verdict,
            "cost": {"cpu_seconds": round(cpu, 3), "wall_seconds": round(wall, 3)},
        },
    )
    print(
        json.dumps(
            {
                "verdict": verdict,
                "pooled_delta": round(pooled_delta, 6),
                "bootstrap_lb": round(bootstrap["percentile_2_5"], 6),
                "per_seed": {str(k): round(v, 6) for k, v in per_seed_delta.items()},
                "cpu": round(cpu, 2),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
