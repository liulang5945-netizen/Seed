"""Training-window v2: checkpoint save/load preflight + one-time pricing pilot.

Owner admitted establishing the bounded training window; per the established
discipline the actual training run starts only after (1) the checkpoint
save/load preflight passes (user standing rule: before any training, verify
checkpoints save correctly) and (2) a one-time price measured on the REAL v2
graph is on the table for owner admission.

This script does NOT run the training window.  Its pilot updates are real
optimizer steps on disposable graph copies, used solely to measure unit cost;
the training run itself restarts from the parent checkpoint under its own
frozen pre-registration.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "training"))

import torch  # noqa: E402

torch.set_num_threads(1)

import probe_taiji_fec1_w1a_retest_20261108 as w1a  # noqa: E402
import probe_taiji_fec1_wv2_retest_20261108 as wv2  # noqa: E402,F401  (import side effect: installs the v2-aware capture instrument into w1a)
import probe_taiji_package20_4c_r2_formal_comparison as r2  # noqa: E402

from taiji.living_core_batch import growth_rows, patch_table, update_constants  # noqa: E402
from taiji.rule_learning_20_1_net import (  # noqa: E402
    enable_content_path_v2,
    enable_write_timing_per_turn,
)

REPORT = ROOT / "reports/taiji_training_window_v2_pricing_20261108.json"
PROGRESS = ROOT / "reports/taiji_training_window_v2_pricing_20261108_progress.jsonl"
TEMP_DIR = ROOT / "output/taiji_training_window_v2_pricing"
PILOT_UPDATES = 5
DEV_REPLAYS = 2
BUFFER = 1.25

START_CPU = time.process_time()
START_WALL = time.time()


class PreflightFailure(RuntimeError):
    pass


def append_progress(record: dict) -> None:
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with PROGRESS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def cost() -> tuple[float, float]:
    return time.process_time() - START_CPU, time.time() - START_WALL


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def training_centroid() -> torch.Tensor:
    """Input-side centroid over the queries of the 160 training worlds (no
    gold, no answers) - the training-distribution analogue of the probe-face
    centroid used by the FEC-1 retests."""

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


def one_update(graph, optimizer, seed: int, cursor: int) -> float:
    """One real training update on the v2 graph (2 worlds, backward, step)."""
    worlds = [
        r2.world_at(seed, r2.TRAIN_INDICES[(cursor + offset) % len(r2.TRAIN_INDICES)])
        for offset in range(r2.BATCH)
    ]
    const = update_constants(graph.subject)
    parent, gates = growth_rows(
        graph.subject, graph.step_kwargs.get("growth_gates"), graph.subject.W
    )
    table = patch_table(graph.subject, graph.subject.W)
    graph.subject.zero_grad(set_to_none=True)
    loss_value, _metrics = r2.run_pair(
        graph.subject, graph, worlds, const, parent, gates, table, train=True
    )
    loss_value.backward()
    torch.nn.utils.clip_grad_norm_(r2.trainer_params(graph), r2.GRAD_CLIP)
    optimizer.step()
    return float(loss_value.detach())


def checkpoint_preflight(graph, optimizer, seed: int) -> dict:
    """User standing rule + P3 hard gate: before any training, prove
    checkpoints save and restore correctly."""

    checks: dict = {}
    payload = {
        "schema": "taiji-training-window-v2-preflight-v1",
        "arm": graph.arm,
        "seed": seed,
        "updates": 1,
        "cursor": r2.BATCH,
        "subject": graph.subject.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": torch.get_rng_state(),
    }
    path = TEMP_DIR / f"preflight_{graph.arm}.pt"
    torch.save(payload, path)
    size_bytes = path.stat().st_size
    checks["saved_bytes"] = size_bytes
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    graph.subject.load_state_dict(loaded["subject"])
    optimizer.load_state_dict(loaded["optimizer"])
    torch.set_rng_state(loaded["rng"])
    # bitwise roundtrip on every tensor of the subject
    worst = 0.0
    mismatched = []
    for name, tensor in payload["subject"].items():
        back = loaded["subject"][name]
        delta = float((tensor - back).abs().max()) if tensor.numel() else 0.0
        worst = max(worst, delta)
        if delta != 0.0:
            mismatched.append(name)
    checks["roundtrip_max_abs_diff"] = worst
    checks["roundtrip_mismatched_tensors"] = mismatched
    # resave from the loaded state and require a SECOND tensor-level roundtrip
    # (file-byte determinism is not a project requirement; the r2 lineage's
    # resume verification is loss/weights equality, and torch.save may embed
    # storage-order metadata that differs after an in-place load_state_dict)
    resave = TEMP_DIR / f"preflight_{graph.arm}_resave.pt"
    torch.save(payload, resave)
    reloaded2 = torch.load(resave, map_location="cpu", weights_only=True)
    second_worst = 0.0
    for name, tensor in payload["subject"].items():
        back = reloaded2["subject"][name]
        if tensor.numel():
            second_worst = max(second_worst, float((tensor - back).abs().max()))
    checks["resave_roundtrip_max_abs_diff"] = second_worst
    checks["resave_sha_equal"] = second_worst == 0.0
    resave.unlink()
    # truncated payload must be rejected
    truncated = TEMP_DIR / f"preflight_{graph.arm}_trunc.pt"
    raw = path.read_bytes()
    truncated.write_bytes(raw[: int(len(raw) * 0.6)])
    rejected = False
    try:
        torch.load(truncated, map_location="cpu", weights_only=True)
    except Exception:
        rejected = True
    truncated.unlink()
    checks["truncated_rejected"] = rejected
    checks["disk_free_mb"] = round(shutil.disk_usage(TEMP_DIR).free / 1e6, 1)
    path.unlink()
    ok = worst == 0.0 and not mismatched and checks["resave_sha_equal"] and rejected
    checks["ok"] = bool(ok)
    return checks


def main() -> int:
    started = cost()
    append_progress({"stage": "start", "cpu": round(started[0], 3)})
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    parent_payload = torch.load(r2.BUNDLE, map_location="cpu", weights_only=True)
    gates_list = parent_payload.get("growth_gates")
    gate = dict(gates_list[0]) if gates_list else None
    candidate = r2.load_routing_bundle(r2.BUNDLE)[0]
    centroid = training_centroid()

    per_update: dict = {}
    preflight: dict = {}
    dev_cpu_per_replay: list = []
    checkpoint_cost = None
    losses_seen: dict = {}
    try:
        for arm in ("CF1", "B_EH"):
            graph = build_v2_graph(candidate, gate, arm, centroid)
            r2.unfreeze_outer(graph.subject)
            optimizer = torch.optim.Adam(r2.trainer_params(graph), lr=r2.LEARNING_RATE)
            # checkpoint preflight AFTER one real update (non-trivial state)
            one_update(graph, optimizer, 20261105, 0)
            checks = checkpoint_preflight(graph, optimizer, 20261105)
            preflight[arm] = checks
            append_progress(
                {"stage": "preflight", "arm": arm, "ok": checks["ok"], "cpu": round(cost()[0], 3)}
            )
            if not checks["ok"]:
                raise PreflightFailure(f"checkpoint preflight failed for {arm}: {checks}")
            # pricing pilot: 5 real updates, fresh optimizer (restart-from-parent
            # semantics; the pilot measures COST, not learning)
            graph = build_v2_graph(candidate, gate, arm, centroid)
            r2.unfreeze_outer(graph.subject)
            optimizer = torch.optim.Adam(r2.trainer_params(graph), lr=r2.LEARNING_RATE)
            times = []
            for step in range(PILOT_UPDATES):
                marker = time.process_time()
                loss = one_update(graph, optimizer, 20261105, r2.BATCH * step)
                times.append(time.process_time() - marker)
                losses_seen.setdefault(arm, []).append(round(loss, 6))
            per_update[arm] = {
                "per_update_cpu": [round(t, 3) for t in times],
                "median": round(sorted(times)[len(times) // 2], 3),
                "mean": round(sum(times) / len(times), 3),
            }
            append_progress(
                {
                    "stage": "pilot",
                    "arm": arm,
                    "median": per_update[arm]["median"],
                    "cpu": round(cost()[0], 3),
                }
            )
            # checkpoint save+load cost (r2 checkpoints every 25 steps)
            payload = {
                "subject": graph.subject.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng": torch.get_rng_state(),
            }
            cpath = TEMP_DIR / f"cost_{arm}.pt"
            marker = time.process_time()
            torch.save(payload, cpath)
            torch.load(cpath, map_location="cpu", weights_only=True)
            checkpoint_cost = time.process_time() - marker
            cpath.unlink()

        # dev-world inference cost (v2 graph, no_grad replay)
        dev_graph = build_v2_graph(candidate, gate, "CF1", centroid)
        dev_graph.subject.eval()
        for index in range(DEV_REPLAYS):
            marker = time.process_time()
            w1a.replay(
                dev_graph,
                r2.world_at(20261105, 160 + index),
                torch.zeros((4, 2), dtype=torch.bool),
                torch.full((4, 2), True, dtype=torch.bool),
            )
            dev_cpu_per_replay.append(round(time.process_time() - marker, 3))
        append_progress({"stage": "dev_cost", "cpu": round(cost()[0], 3)})
    except PreflightFailure as error:
        write_json(
            REPORT,
            {
                "package": "training-window-v2",
                "status": "preflight_failed",
                "reason": str(error),
                "preflight": preflight,
            },
        )
        print(json.dumps({"status": "preflight_failed", "reason": str(error)}))
        return 0

    # extrapolation: 2 seeds x 2 arms x 200 updates + dev evals + checkpoints
    updates_total = 2 * 2 * r2.UPDATES_PER_RUN
    train_cpu = (
        2 * r2.UPDATES_PER_RUN * per_update["CF1"]["mean"]
        + 2 * r2.UPDATES_PER_RUN * per_update["B_EH"]["mean"]
    )
    dev_evals = 4 * 2 * 16
    dev_cpu = dev_evals * (sum(dev_cpu_per_replay) / len(dev_cpu_per_replay))
    saves = 4 * (r2.UPDATES_PER_RUN // r2.CHECKPOINT_EVERY)
    ckpt_cpu = saves * (checkpoint_cost or 0.0)
    resume_overhead = 4 * (sum(dev_cpu_per_replay) / len(dev_cpu_per_replay))
    base = train_cpu + dev_cpu + ckpt_cpu + resume_overhead
    projection = {
        "updates_total": updates_total,
        "train_cpu": round(train_cpu, 1),
        "dev_cpu": round(dev_cpu, 1),
        "checkpoint_cpu": round(ckpt_cpu, 1),
        "resume_overhead_cpu": round(resume_overhead, 1),
        "base_total_cpu": round(base, 1),
        "buffer_25pct": round(base * (BUFFER - 1), 1),
        "proposed_cap_cpu": round(base * BUFFER, 1),
        "comparisons": {
            "package20_proposal": 24000.0,
            "package20_measured": 50573.637,
        },
    }
    cpu, wall = cost()
    write_json(
        REPORT,
        {
            "schema": "taiji-training-window-v2-pricing-v1",
            "package": "training-window-v2",
            "date": "2026-11-08",
            "status": "completed",
            "preflight": preflight,
            "pilot": {
                "updates_per_arm": PILOT_UPDATES,
                "per_update": per_update,
                "losses_observed": losses_seen,
            },
            "dev_cpu_per_replay": dev_cpu_per_replay,
            "checkpoint_save_load_cpu": round(checkpoint_cost or 0.0, 3),
            "projection": projection,
            "discipline": {
                "training_run_updates": 0,
                "pilot_updates": 2 * PILOT_UPDATES,
                "confirm_reads": 0,
                "note": "pilot updates are cost probes on disposable graph copies; the training run restarts from the parent checkpoint under its own frozen prereg",
            },
            "cost": {"cpu_seconds": round(cpu, 3), "wall_seconds": round(wall, 3)},
        },
    )
    print(
        json.dumps(
            {
                "preflight_ok": {arm: preflight[arm]["ok"] for arm in preflight},
                "cpu_per_update": {arm: per_update[arm]["median"] for arm in per_update},
                "proposed_cap_cpu": projection["proposed_cap_cpu"],
                "total_cpu_this_script": round(cpu, 2),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
