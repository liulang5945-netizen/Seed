"""20.2r: replay the 20.2 credit probe byte-for-byte, then measure F leverage.

Zero training, zero optimizer steps, zero Adam, zero confirm reads.  The probe
imports the *committed* 20.2 runner (sha pinned below) and reuses its code, so
the replay executes the same bytes that produced the 20.2 readings.  The ladder
then perturbs only the fast tensor (F) at each correction-surface readout, one
forward per cell, and reports how much answer quality can move per unit of
loss.  The oracle arm is a diagnostic ceiling only: gold-direction F, never
persisted, never admitted to any training or evaluation metric.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

torch.set_num_threads(1)

REPORT = ROOT / "reports/taiji_package20_2r_f_leverage_20261105.json"
MANIFEST = ROOT / "plans/manifests/taiji_package20_2r_f_leverage_20261105.json"
PROGRESS = ROOT / "reports/taiji_package20_2r_f_leverage_20261105_progress.jsonl"
P2_PATH = ROOT / "scripts/training/probe_taiji_package20_2_credit_recovery.py"
P2_COMMITTED_SHA = "cd23ce2e06b4b6ad1455378f0d9df4f60bef98762893756f6302f3a709c04119"
REPORT_20_2 = ROOT / "reports/taiji_package20_2_credit_recovery_20261105.json"

CPU_CAP = 300.0
CPU_GUARD = 290.0
WALL_CAP = 2700.0
REPLAY_RTOL = 1e-6
LEVERAGE_THRESHOLD = 0.01
LADDER_ARMS = ("f_zero", "f_half", "f_double", "f_quad", "f_permuted")
ORACLE_SCALES = (0.5, 2.0, 8.0)
PERMUTE_SEED = 20261105

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


def load_committed_p2():
    raw = P2_PATH.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    spec = importlib.util.spec_from_file_location("probe_taiji_package20_2", P2_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, sha


def rel_diff(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), 1e-12)


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


def snapshot_surface(p2, graph, world: dict) -> dict:
    """Baseline sequence, keeping the pre-turn state of every correction turn."""
    response = world["response"]
    state = graph.initial_state(response["world_id"])
    released: list[dict] = []
    rows: list[dict] = []
    snaps: list[dict] = []
    writes = 0
    for turn in response["turns"]:
        number = int(turn["turn"])
        guard()
        if bool(turn["correction_surface"]) and released:
            snaps.append(
                {
                    "number": number,
                    "session": int(turn["session"]),
                    "kind": turn["kind"],
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
                "session": int(turn["session"]),
                "kind": turn["kind"],
                "correction_surface": bool(turn["correction_surface"]),
                "released_lines": len(released),
                "bytes": tuple(int(value) for value in result["bytes"]),
                "constrained": tuple(int(value) for value in result["constrained_bytes"]),
                "logits": result["logits"].detach().clone(),
            }
        )
        for release in response["releases"]:
            if release["after_turn"] != number:
                continue
            state, info = graph.release(state, release, released, write_enabled=True, now=number)
            writes += int(info["written"])
            released.extend(
                {
                    "source_id": item["source_id"],
                    "feedback": item["feedback"],
                    "version": item["version"],
                    "after_turn": number,
                }
                for item in release["items"]
            )
    return {
        "rows": rows,
        "snaps": snaps,
        "final_state": state,
        "fast_writes": writes,
    }


def readout(p2, graph, snap: dict, fast: torch.Tensor, gold: bytes) -> dict:
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
    logits = result["logits"]
    return {
        "turn": snap["number"],
        "constrained": tuple(int(value) for value in result["constrained_bytes"]),
        "bytes": tuple(int(value) for value in result["bytes"]),
        "loss": float(p2.task_loss(logits, gold).detach()),
    }


def variant_fast(kind: str, fast: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    if kind == "f_zero":
        return torch.zeros_like(fast)
    if kind == "f_half":
        return fast * 0.5
    if kind == "f_double":
        return fast * 2.0
    if kind == "f_quad":
        return fast * 4.0
    if kind == "f_permuted":
        flat = fast.flatten()
        order = torch.randperm(flat.numel(), generator=rng)
        shuffled = flat[order]
        norm = float(flat.norm())
        shuffled = shuffled * (norm / max(float(shuffled.norm()), 1e-12))
        return shuffled.reshape(fast.shape)
    raise KeyError(kind)


def summarize(p2, rows: list[dict], baseline: list[dict], world: dict) -> dict:
    correct_full = 0
    correct_decision = 0
    byte_diff = 0
    for row, base in zip(rows, baseline, strict=True):
        gold = p2.gold_bytes(world, {"turn": row["turn"]})
        correct_full += int(row["constrained"] == tuple(gold))
        correct_decision += int(
            all(
                row["constrained"][position] == gold[position] for position in p2.DECISION_POSITIONS
            )
        )
        byte_diff += sum(
            1 for a, b in zip(row["constrained"], base["constrained"], strict=True) if a != b
        )
    loss = sum(row["loss"] for row in rows) / len(rows)
    base_loss = sum(row["loss"] for row in baseline) / len(baseline)
    delta = loss - base_loss
    return {
        "correction_turns": len(rows),
        "correct_full": correct_full,
        "correct_decision": correct_decision,
        "loss": loss,
        "baseline_loss": base_loss,
        "delta_loss": delta,
        "byte_diff": byte_diff,
        "leverage": abs(delta) / max(abs(base_loss), 1e-12),
    }


def main() -> int:
    p2, p2_sha = load_committed_p2()
    append_progress(
        {
            "stage": "load_p2",
            "p2_script_sha256": p2_sha,
            "committed_match": p2_sha == P2_COMMITTED_SHA,
            "cpu": round(cost()[0], 3),
        }
    )
    if p2_sha != P2_COMMITTED_SHA:
        write_json(
            REPORT,
            {
                "package": "20.2r",
                "date": "2026-11-05",
                "verdict": "replay_script_drift",
                "script": {"p2_script_sha256": p2_sha, "pinned": P2_COMMITTED_SHA},
                "cost": {"cpu_seconds": round(cost()[0], 2)},
            },
        )
        print(json.dumps({"verdict": "replay_script_drift", "p2_sha": p2_sha}, indent=2))
        return 0

    world = p2.fixture.build_world(p2.fixture.SEED, 0)
    bundle = torch.load(p2.BUNDLE, map_location="cpu", weights_only=False)
    candidate = p2.load_routing_bundle(p2.BUNDLE)[0]
    gate = dict(bundle["growth_gates"][0]) if bundle.get("growth_gates") else None
    parent = candidate.subject
    state_shas = {"cf1": p2.sha256_file(p2.CF1_STATE), "b_eh": p2.sha256_file(p2.BEH_STATE)}
    assemblies_intact = (
        state_shas["cf1"] == p2.CF1_STATE_SHA and state_shas["b_eh"] == p2.BEH_STATE_SHA
    )
    cf1 = p2.fresh_graph(parent, gate, "CF1")
    beh = p2.fresh_graph(parent, gate, "B_EH")
    p2.freeze(cf1.subject)
    p2.freeze(beh.subject)
    assert assemblies_intact, "reference assemblies drifted"
    append_progress({"stage": "graphs", "cpu": round(cost()[0], 3)})

    # ---- A. replay the 20.2 credit probe on the committed bytes -------------
    report_20_2 = json.loads(REPORT_20_2.read_text(encoding="utf-8"))
    expect = report_20_2["credit"]["a1"]
    history = p2.run_sequence(cf1, world, write_enabled=True, end_turn=p2.CREDIT_PROBE_TURN - 1)
    probe_turn = next(
        turn for turn in world["response"]["turns"] if int(turn["turn"]) == p2.CREDIT_PROBE_TURN
    )
    released = p2.released_upto(world, p2.CREDIT_PROBE_TURN - 1)
    fast_leaf = history["final_state"].living.fast.detach().clone().requires_grad_(True)
    logits, _ = p2.forward_credit(
        cf1, history["final_state"], probe_turn, released, fast_leaf, now=p2.CREDIT_PROBE_TURN
    )
    gold_probe = p2.gold_bytes(world, probe_turn)
    loss = p2.task_loss(logits, gold_probe)
    p2.backward(loss)
    fast_norm = float(fast_leaf.grad.norm())
    by_head = p2.head_grad(fast_leaf.grad)
    share = fast_norm / (float(loss.detach()) + 1e-12)
    p2.clear_grads(cf1.subject)
    replay = {
        "probe_turn": p2.CREDIT_PROBE_TURN,
        "loss": float(loss.detach()),
        "fast_grad_norm": fast_norm,
        "fast_grad_by_head": by_head,
        "share_of_loss": share,
        "expected_loss": expect["loss"],
        "expected_fast_grad_norm": expect["fast_grad_norm"],
        "expected_share_of_loss": expect["share_of_loss"],
        "loss_rel_diff": rel_diff(float(loss.detach()), expect["loss"]),
        "grad_rel_diff": rel_diff(fast_norm, expect["fast_grad_norm"]),
        "share_rel_diff": rel_diff(share, expect["share_of_loss"]),
        "head_rel_diff": max(
            rel_diff(float(a), float(b))
            for row_a, row_b in zip(by_head, expect["fast_grad_by_head"], strict=True)
            for a, b in zip(row_a, row_b, strict=True)
        ),
    }
    append_progress({"stage": "replay_a1", "loss": replay["loss"], "cpu": round(cost()[0], 3)})

    # Dry forward parity: F off, CF1 vs B_EH, 51 turns.
    off_cf1 = p2.run_sequence(cf1, world, write_enabled=False)
    off_beh = p2.run_sequence(beh, world, write_enabled=False)
    parity = p2.compare_runs(off_cf1, off_beh)
    append_progress(
        {"stage": "parity", "bytes": parity["byte_identical"], "cpu": round(cost()[0], 3)}
    )

    replay_ok = (
        replay["loss_rel_diff"] <= REPLAY_RTOL
        and replay["grad_rel_diff"] <= REPLAY_RTOL
        and replay["share_rel_diff"] <= REPLAY_RTOL
        and replay["head_rel_diff"] <= REPLAY_RTOL
        and parity["bytes_compared"] == parity["byte_identical"]
    )
    if not replay_ok:
        payload = {
            "package": "20.2r",
            "date": "2026-11-05",
            "verdict": "replay_mismatch",
            "script": {
                "runner_sha256": p2_sha,
                "p2_script_sha256": p2_sha,
                "p2_committed_sha256": P2_COMMITTED_SHA,
                "20_2_report_script_sha256": None,
                "note": "20.2 report carries no script sha; replay runs the committed bytes",
            },
            "replay": replay,
            "parity": parity,
            "cost": {
                "cpu_seconds": round(cost()[0], 2),
                "wall_seconds": round(cost()[1], 2),
                "training_steps": 0,
                "optimizer_steps": 0,
                "backward_calls": p2.BACKWARD_CALLS,
            },
        }
        write_json(REPORT, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    # ---- B. F leverage ladder on the 22 correction-surface turns ------------
    surface = snapshot_surface(p2, cf1, world)
    snaps = surface["snaps"]
    baseline = [
        readout(
            p2, cf1, snap, snap["state"].living.fast, p2.gold_bytes(world, {"turn": snap["number"]})
        )
        for snap in snaps
    ]
    identity = sum(
        1
        for row, snap in zip(baseline, snaps, strict=True)
        if row["constrained"]
        == tuple(next(t for t in surface["rows"] if t["turn"] == snap["number"])["constrained"])
    )
    append_progress(
        {
            "stage": "surface",
            "corrections": len(snaps),
            "identity": identity,
            "cpu": round(cost()[0], 3),
        }
    )

    rng = torch.Generator().manual_seed(PERMUTE_SEED)
    arms: dict[str, dict] = {}
    for kind in LADDER_ARMS:
        guard()
        rows = []
        for snap in snaps:
            fast = variant_fast(kind, snap["state"].living.fast, rng)
            rows.append(
                readout(p2, cf1, snap, fast, p2.gold_bytes(world, {"turn": snap["number"]}))
            )
        arms[kind] = summarize(p2, rows, baseline, world)
        append_progress(
            {
                "stage": "ladder",
                "arm": kind,
                "delta_loss": arms[kind]["delta_loss"],
                "leverage": arms[kind]["leverage"],
                "cpu": round(cost()[0], 3),
            }
        )

    # Oracle diagnostic: F moved along the gold-descent direction, ceiling only.
    oracle_rows = []
    oracle_grads = []
    for snap in snaps:
        guard()
        base = snap["state"].living.fast.detach().clone()
        leaf = base.clone().requires_grad_(True)
        probe_logits, _ = p2.forward_credit(
            cf1, snap["state"], snap["turn"], snap["released"], leaf, now=snap["number"]
        )
        probe_loss = p2.task_loss(probe_logits, p2.gold_bytes(world, {"turn": snap["number"]}))
        p2.backward(probe_loss)
        gradient = leaf.grad.detach().clone()
        p2.clear_grads(cf1.subject)
        oracle_grads.append(float(gradient.norm()))
        direction = -gradient / max(float(gradient.norm()), 1e-12) * float(base.norm())
        best = None
        for scale in ORACLE_SCALES:
            fast = (base + direction * scale).detach()
            row = readout(p2, cf1, snap, fast, p2.gold_bytes(world, {"turn": snap["number"]}))
            if best is None or row["loss"] < best["loss"]:
                best = row
        oracle_rows.append(best)
    oracle = summarize(p2, oracle_rows, baseline, world)
    oracle["grad_norm_mean"] = sum(oracle_grads) / len(oracle_grads)
    oracle["scales"] = list(ORACLE_SCALES)
    oracle["oracle_diagnostic_only"] = True
    append_progress(
        {"stage": "oracle", "delta_loss": oracle["delta_loss"], "cpu": round(cost()[0], 3)}
    )

    leverage_max = max(arm["leverage"] for arm in arms.values())
    verdict = (
        "channel_capacity_sufficient"
        if leverage_max >= LEVERAGE_THRESHOLD
        else "under_powered_f_channel"
    )

    # ---- C. gates (thresholds fixed before the ladder was computed) ---------
    gates = {
        "script_is_committed_bytes": p2_sha == P2_COMMITTED_SHA,
        "replay_loss_within_1e_6": replay["loss_rel_diff"] <= REPLAY_RTOL,
        "replay_grad_within_1e_6": replay["grad_rel_diff"] <= REPLAY_RTOL,
        "replay_share_within_1e_6": replay["share_rel_diff"] <= REPLAY_RTOL,
        "replay_heads_within_1e_6": replay["head_rel_diff"] <= REPLAY_RTOL,
        "dry_forward_parity_765": parity["bytes_compared"] == parity["byte_identical"],
        "baseline_readout_identity": identity == len(snaps),
        "ladder_five_arms": len(arms) == len(LADDER_ARMS),
        "oracle_ceiling_measured": oracle["oracle_diagnostic_only"],
        "no_training": True,
        "assemblies_intact": assemblies_intact,
    }
    payload = {
        "package": "20.2r",
        "date": "2026-11-05",
        "verdict": verdict,
        "scope": "replay consistency and F leverage gate; 0 training, 0 optimizer steps",
        "script": {
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "p2_script_sha256": p2_sha,
            "p2_committed_sha256": P2_COMMITTED_SHA,
            "20_2_report_script_sha256": None,
            "note": "20.2 report carries no script sha; this run reuses the committed 20.2 bytes",
        },
        "replay": replay,
        "parity": parity,
        "leverage": {
            "threshold": LEVERAGE_THRESHOLD,
            "leverage_max": leverage_max,
            "verdict": verdict,
            "correction_turns": len(snaps),
            "baseline": {
                "loss": arms["f_zero"]["baseline_loss"],
                "correct_full": sum(
                    1
                    for row, snap in zip(baseline, snaps, strict=True)
                    if row["constrained"] == tuple(p2.gold_bytes(world, {"turn": snap["number"]}))
                ),
                "correct_decision": sum(
                    1
                    for row, snap in zip(baseline, snaps, strict=True)
                    if all(
                        row["constrained"][position]
                        == p2.gold_bytes(world, {"turn": snap["number"]})[position]
                        for position in p2.DECISION_POSITIONS
                    )
                ),
            },
            "arms": arms,
            "oracle": oracle,
            "oracle_ceiling_leverage": oracle["leverage"],
            "oracle_ceiling_below_1pct": oracle["leverage"] < LEVERAGE_THRESHOLD,
        },
        "state": {
            "fast_writes_baseline": surface["fast_writes"],
            "parent_weight_sha256": p2.weights_digest(parent),
            "assemblies": state_shas,
        },
        "gates": gates,
        "cost": {
            "cpu_cap": CPU_CAP,
            "cpu_seconds": round(cost()[0], 2),
            "wall_seconds": round(cost()[1], 2),
            "training_steps": 0,
            "optimizer_steps": 0,
            "adam_steps": 0,
            "confirm_reads": 0,
            "backward_calls": p2.BACKWARD_CALLS,
            "turns_forwarded": 51
            + 102
            + 17
            + 1
            + len(snaps) * (1 + len(LADDER_ARMS) + 1 + len(ORACLE_SCALES)),
        },
    }
    write_json(REPORT, payload)
    write_json(
        MANIFEST,
        {
            "package": "20.2r",
            "date": "2026-11-05",
            "report": str(REPORT.relative_to(ROOT)),
            "report_sha256": hashlib.sha256(REPORT.read_bytes()).hexdigest(),
            "runner": str(Path(__file__).relative_to(ROOT)),
            "runner_sha256": payload["script"]["runner_sha256"],
            "replayed_runner_sha256": p2_sha,
            "verdict": verdict,
            "reference_assemblies_untouched": True,
            "training_admitted": False,
            "saved_states": [],
            "cost_cpu_seconds": payload["cost"]["cpu_seconds"],
        },
    )
    append_progress({"stage": "report", "verdict": verdict, "cpu": round(cost()[0], 3)})
    print(
        json.dumps(
            {"verdict": verdict, "leverage_max": leverage_max, "arms": arms}, ensure_ascii=False
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
