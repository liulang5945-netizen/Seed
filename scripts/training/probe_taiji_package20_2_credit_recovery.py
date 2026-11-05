"""20.2 probe: credit chain, negative controls, full save and independent restore.

Zero training.  Backward passes are diagnostic only: no optimizer is created and
no parameter receives an update.  The 20.1 graph module is imported unchanged so
that both ``candidate_sha_v2`` values stay frozen.

Readings:
  A. credit chain -- dL/dF, per-tensor gradients (135), freeze check, F-only
     shaping, query<->release binding trace;
  B. controls -- B_EH (F never written), same-length scrambled feedback,
     F-off parity re-checked in this package;
  C. recovery -- full checkpoint then reload, cross-session resume, crash
     (truncated file) recovery, state x persistence x byte-identity table.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

torch.set_num_threads(1)

from taiji import rule_learning_20_1 as fixture  # noqa: E402
from taiji.living_core import EOS  # noqa: E402
from taiji.rule_learning_20_1_identity import sha256_file  # noqa: E402
from taiji.rule_learning_20_1_net import (  # noqa: E402
    ANSWER_BYTES,
    DECISION_POSITIONS,
    TEMPLATE_BYTES,
    RuleLearningGraph,
    _group_of,
    copy_subject,
    state_digest,
)
from taiji.selection_routing_checkpoint import load_routing_bundle  # noqa: E402

REPORT = ROOT / "reports/taiji_package20_2_credit_recovery_20261105.json"
MANIFEST = ROOT / "plans/manifests/taiji_package20_2_credit_recovery_20261105.json"
PROGRESS = ROOT / "reports/taiji_package20_2_credit_recovery_20261105_progress.jsonl"
BUNDLE = ROOT / "output/taiji_package19_compare/28031-B_stop_lock-step300.pt"
CF1_STATE = ROOT / "output/taiji_package20_1_cf1_initialized_untrained.pt"
BEH_STATE = ROOT / "output/taiji_package20_1_b_eh_initialized_untrained.pt"
RECOVERY = ROOT / "output/taiji_package20_2_recovery"
CF1_STATE_SHA = "b62d6fe6a0721fd82c7c43d575e600c48ffc8e31ff959bfdf14ef9ff07d0039e"
BEH_STATE_SHA = "64cb327e3a8442ef0c59e04b14445a08ab3dccbb8ac42cf6547e0efa4e6f7f46"

CPU_CAP = 400.0
CPU_GUARD = 390.0
WALL_CAP = 3600.0
CREDIT_TURNS = (6, 18)
CREDIT_PROBE_TURN = 18
RESUME_SPAN = 6

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


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def weights_digest(subject) -> str:
    rows = []
    for name, tensor in sorted(subject.state_dict().items()):
        data = tensor.detach().cpu().numpy().tobytes()
        rows.append((name, str(tensor.dtype), tuple(tensor.shape), sha256_bytes(data)))
    return sha256_bytes(repr(rows).encode("utf-8"))


def fresh_graph(parent, gate, arm: str) -> RuleLearningGraph:
    return RuleLearningGraph(copy_subject(parent), arm=arm, growth_gates=gate)


def released_upto(world: dict, turn: int) -> list[dict]:
    return [
        {
            "source_id": item["source_id"],
            "feedback": item["feedback"],
            "version": item["version"],
            "after_turn": release["after_turn"],
        }
        for release in world["response"]["releases"]
        if release["after_turn"] <= turn
        for item in release["items"]
    ]


def legal_vocab(released: list[dict]) -> tuple[int, ...]:
    ids = set(TEMPLATE_BYTES.values())
    for item in released:
        ids.update(item["feedback"].encode("utf-8"))
    return tuple(sorted(ids))


def scrambled_world(world: dict, rng: random.Random) -> dict:
    """Same response, same byte length per feedback line, characters permuted."""
    releases = []
    for release in world["response"]["releases"]:
        items = []
        for item in release["items"]:
            chars = list(item["feedback"])
            rng.shuffle(chars)
            text = "".join(chars)
            if len(text.encode()) != len(item["feedback"].encode()):
                raise AssertionError("scrambled feedback changed the byte length")
            items.append(dict(item, feedback=text))
        releases.append(dict(release, items=items))
    return {
        "world_id": world["world_id"],
        "index": world["index"],
        "seed": world["seed"],
        "response": dict(world["response"], releases=releases),
        "scoring": world["scoring"],
    }


def run_sequence(
    graph: RuleLearningGraph,
    world: dict,
    *,
    write_enabled: bool,
    start_state=None,
    start_turn: int = 0,
    end_turn: int = fixture.TOTAL_TURNS,
    released: list[dict] | None = None,
) -> dict:
    response = world["response"]
    state = graph.initial_state(response["world_id"]) if start_state is None else start_state
    released = [] if released is None else list(released)
    turns: list[dict] = []
    traces: list[dict] = []
    fast_writes = 0
    applied = 0
    for turn in response["turns"]:
        number = int(turn["turn"])
        if number <= start_turn or number > end_turn:
            continue
        guard()
        result = graph.run_turn(
            state,
            turn,
            released,
            write_enabled=write_enabled,
            vocab=legal_vocab(released),
            now=number,
        )
        state = result["state"]
        turns.append(
            {
                "turn": number,
                "session": int(turn["session"]),
                "kind": turn["kind"],
                "correction_surface": bool(turn["correction_surface"]),
                "released_lines": len(released),
                "bytes": tuple(int(value) for value in result["bytes"]),
                "constrained": tuple(int(value) for value in result["constrained_bytes"]),
                "logits": result["logits"],
            }
        )
        for release in response["releases"]:
            if release["after_turn"] != number or number > end_turn:
                continue
            before = state.living.fast.clone()
            state, info = graph.release(
                state, release, released, write_enabled=write_enabled, now=number
            )
            fast_writes += int(info["written"])
            applied += int(info["applied_receipts"])
            after = state.living.fast
            changed = []
            delta = 0.0
            for group in range(after.shape[0]):
                for head in range(after.shape[1]):
                    difference = (after[group, head] - before[group, head]).abs()
                    delta = max(delta, float(difference.max()))
                    if float(difference.max()) > 0:
                        changed.append([group, head])
            traces.append(
                {
                    "after_turn": int(release["after_turn"]),
                    "version": int(release["version"]),
                    "lines": len(release["items"]),
                    "source_ids": [item["source_id"] for item in release["items"]],
                    "referenced_turns": [int(item["turn"]) for item in release["items"]],
                    "write_enabled": bool(write_enabled),
                    "written": int(info["written"]),
                    "applied_receipts": int(info["applied_receipts"]),
                    "changed_heads": changed,
                    "fast_max_delta": delta,
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
    return {
        "turns": turns,
        "traces": traces,
        "final_state": state,
        "fast_writes": fast_writes,
        "applied_receipts": applied,
        "released": released,
    }


def compare_runs(left: dict, right: dict) -> dict:
    if len(left["turns"]) != len(right["turns"]):
        return {
            "turns_compared": min(len(left["turns"]), len(right["turns"])),
            "bytes_compared": 0,
            "byte_identical": 0,
            "max_abs_logit_delta": None,
            "turn_count_mismatch": True,
        }
    max_delta = 0.0
    identical = 0
    compared = 0
    mismatch_turns = 0
    for first, second in zip(left["turns"], right["turns"], strict=True):
        if first["turn"] != second["turn"] or first["bytes"] != second["bytes"]:
            mismatch_turns += 1
        delta = float((first["logits"] - second["logits"]).abs().max())
        max_delta = max(max_delta, delta)
        for a, b in zip(first["bytes"], second["bytes"], strict=True):
            compared += 1
            identical += int(a == b)
    return {
        "turns_compared": len(left["turns"]),
        "bytes_compared": compared,
        "byte_identical": identical,
        "max_abs_logit_delta": max_delta,
        "byte_mismatch_turns": mismatch_turns,
    }


def gold_bytes(world: dict, turn: dict) -> bytes:
    return world["scoring"]["turns"][str(turn["turn"])]["expected_action"].encode()


def accuracy(run: dict, world: dict) -> dict:
    rows = [
        turn for turn in run["turns"] if turn["correction_surface"] and turn["released_lines"] > 0
    ]
    full = 0
    decision = 0
    for row in rows:
        truth = gold_bytes(world, row)
        full += int(tuple(row["constrained"]) == tuple(truth))
        decision += int(
            all(row["constrained"][position] == truth[position] for position in DECISION_POSITIONS)
        )
    return {
        "correction_turns": len(rows),
        "full_match": full,
        "decision_match": decision,
        "full_rate": full / len(rows) if rows else 0.0,
    }


def forward_credit(
    graph: RuleLearningGraph,
    state,
    turn: dict,
    released: list[dict],
    fast_leaf: torch.Tensor,
    *,
    now: int,
) -> tuple[torch.Tensor, bytes]:
    """Grad-enabled mirror of ``run_turn`` for one answer turn."""
    query = turn["query"].encode("utf-8")
    records = graph._records(released, now)
    living = replace(state.living, fast=fast_leaf)
    living = graph.subject.observe(
        living,
        query,
        goal=query,
        records=records,
        allowed=graph._allowed(False),
        now=now,
        **graph.step_kwargs,
    )
    logits: list[torch.Tensor] = []
    previous = EOS
    for _ in range(ANSWER_BYTES):
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
        logits.append(output.probabilities)
        choice = int(torch.argmax(output.probabilities[:256].detach()))
        living = graph.subject.consume_output(output, choice)
        previous = choice
    return torch.stack(logits), query


def task_loss(logits: torch.Tensor, action: bytes) -> torch.Tensor:
    total = logits.new_zeros(())
    for position in DECISION_POSITIONS:
        total = total - torch.log(logits[position][action[position]] + 1e-12)
    return total


def grad_table(model, frozen: bool) -> list[dict]:
    rows = []
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        rows.append(
            {
                "group": _group_of(name),
                "name": f"subject.{name}",
                "elements": int(parameter.numel()),
                "requires_grad": bool(parameter.requires_grad),
                "grad_norm": None if gradient is None else float(gradient.norm()),
                "frozen_declared": frozen,
            }
        )
    return rows


def head_grad(fast_grad: torch.Tensor | None) -> list[list[float]]:
    if fast_grad is None:
        return []
    return [
        [float(fast_grad[group, head].norm()) for head in range(fast_grad.shape[1])]
        for group in range(fast_grad.shape[0])
    ]


BACKWARD_CALLS = 0


def backward(loss: torch.Tensor) -> None:
    global BACKWARD_CALLS
    loss.backward()
    BACKWARD_CALLS += 1


def freeze(subject) -> list[str]:
    frozen = []
    for name, parameter in subject.named_parameters():
        if parameter.requires_grad:
            parameter.requires_grad_(False)
            frozen.append(name)
    return frozen


def clear_grads(subject) -> None:
    for parameter in subject.parameters():
        parameter.grad = None


def checkpoint_payload(graph: RuleLearningGraph, state, released: list[dict], turn: int) -> dict:
    return {
        "schema": "taiji-rule-learning-20-2-checkpoint-v1",
        "arm": graph.arm,
        "turn": int(turn),
        "rule_state": graph.save_state(state),
        "session": {"previous_byte": int(EOS), "turn_cursor": 0},
        "released": copy.deepcopy(released),
        "parent_parameter_sha256": weights_digest(graph.subject),
    }


def load_checkpoint(path: Path, graph: RuleLearningGraph) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "taiji-rule-learning-20-2-checkpoint-v1":
        raise ValueError("unexpected checkpoint schema")
    if payload["arm"] != graph.arm:
        raise ValueError("checkpoint arm mismatch")
    if payload["parent_parameter_sha256"] != weights_digest(graph.subject):
        raise ValueError("checkpoint parent weights mismatch")
    return payload


def state_field_table(before, restored, payload: dict) -> list[dict]:
    living_before, living_after = before.living, restored.living
    rows = []
    for name, kind in (
        ("h", "persistent"),
        ("fast", "persistent"),
        ("nu_ema", "persistent"),
        ("coverage", "persistent"),
        ("prior", "persistent"),
    ):
        first = getattr(living_before, name).detach().cpu()
        second = getattr(living_after, name).detach().cpu()
        rows.append(
            {
                "name": name,
                "kind": kind,
                "serialized": True,
                "restored": True,
                "byte_identical": bool(torch.equal(first, second)),
            }
        )
    rows.append(
        {
            "name": "keys",
            "kind": "persistent",
            "serialized": True,
            "restored": True,
            "byte_identical": list(living_before.keys) == list(living_after.keys),
        }
    )
    rows.append(
        {
            "name": "journal",
            "kind": "persistent",
            "serialized": True,
            "restored": True,
            "byte_identical": list(before.journal) == list(restored.journal),
            "entries": len(restored.journal),
        }
    )
    session = payload["session"]
    rows.append(
        {
            "name": "previous_byte",
            "kind": "session",
            "serialized": True,
            "restored": True,
            "byte_identical": session["previous_byte"] == EOS,
            "restore_semantics": "reset to EOS",
        }
    )
    rows.append(
        {
            "name": "turn_cursor",
            "kind": "session",
            "serialized": True,
            "restored": True,
            "byte_identical": session["turn_cursor"] == 0,
            "restore_semantics": "reset to 0",
        }
    )
    return rows


def binding_audit(world: dict, run: dict) -> dict:
    """Every released line must point to an earlier turn with the same query."""
    turns = {int(turn["turn"]): turn for turn in world["response"]["turns"]}
    items = 0
    query_match = 0
    ordered = 0
    applied = 0
    for release in world["response"]["releases"]:
        for item in release["items"]:
            items += 1
            source = turns[int(item["turn"])]
            query_match += int(source["query"].encode("utf-8") == item["query"].encode("utf-8"))
            ordered += int(int(item["turn"]) <= int(release["after_turn"]))
    journal = run["final_state"].journal
    if journal:
        applied = sum(1 for entry in journal if entry["applied"])
    return {
        "released_lines": items,
        "query_matches_source_turn": query_match,
        "released_after_referenced_turn": ordered,
        "applied_receipts": applied,
        "binding_ok": items == query_match == ordered,
    }


def main() -> int:
    world = fixture.build_world(fixture.SEED, 0)
    bundle = torch.load(BUNDLE, map_location="cpu", weights_only=False)
    candidate = load_routing_bundle(BUNDLE)[0]
    gate = dict(bundle["growth_gates"][0]) if bundle.get("growth_gates") else None
    parent = candidate.subject
    parent_sha = weights_digest(parent)
    append_progress({"stage": "load", "parent_parameter_sha256": parent_sha, "cpu": cost()[0]})

    state_shas = {"cf1": sha256_file(CF1_STATE), "b_eh": sha256_file(BEH_STATE)}
    assemblies_intact = state_shas["cf1"] == CF1_STATE_SHA and state_shas["b_eh"] == BEH_STATE_SHA

    cf1 = fresh_graph(parent, gate, "CF1")
    beh = fresh_graph(parent, gate, "B_EH")
    frozen_names = freeze(cf1.subject)
    freeze(beh.subject)
    frozen_count = sum(1 for parameter in cf1.subject.parameters() if not parameter.requires_grad)
    append_progress({"stage": "freeze", "frozen": frozen_count, "cpu": cost()[0]})

    baseline = run_sequence(cf1, world, write_enabled=True)
    append_progress({"stage": "baseline", "turns": len(baseline["turns"]), "cpu": cost()[0]})

    # ---- A. credit chain -----------------------------------------------------
    history = run_sequence(cf1, world, write_enabled=True, end_turn=CREDIT_PROBE_TURN - 1)
    probe_turn = next(
        turn for turn in world["response"]["turns"] if int(turn["turn"]) == CREDIT_PROBE_TURN
    )
    released = released_upto(world, CREDIT_PROBE_TURN - 1)
    fast_leaf = history["final_state"].living.fast.detach().clone().requires_grad_(True)
    logits, _ = forward_credit(
        cf1, history["final_state"], probe_turn, released, fast_leaf, now=CREDIT_PROBE_TURN
    )
    action = gold_bytes(world, probe_turn)
    loss = task_loss(logits, action)
    backward(loss)
    fast_norm = None if fast_leaf.grad is None else float(fast_leaf.grad.norm())
    fast_by_head = head_grad(fast_leaf.grad)
    frozen_table = grad_table(cf1.subject, frozen=True)
    nonzero_frozen = [
        row for row in frozen_table if row["grad_norm"] is not None and row["grad_norm"] > 0
    ]
    a1 = {
        "probe_turn": CREDIT_PROBE_TURN,
        "probe_kind": probe_turn["kind"],
        "loss": float(loss.detach()),
        "fast_grad_norm": fast_norm,
        "fast_grad_by_head": fast_by_head,
        "finite": bool(
            fast_norm is not None and fast_norm > 0 and torch.isfinite(torch.tensor(fast_norm))
        ),
        "share_of_loss": None if fast_norm is None else fast_norm / (float(loss.detach()) + 1e-12),
        "verdict": "reachable" if fast_norm and fast_norm > 1e-12 else "credit_starvation",
    }
    append_progress({"stage": "credit_a1", "fast_grad_norm": fast_norm, "cpu": cost()[0]})
    clear_grads(cf1.subject)

    if a1["verdict"] != "reachable":
        payload = {
            "package": "20.2",
            "date": "2026-11-05",
            "verdict": "credit_starvation",
            "credit": {"a1": a1},
            "cost": {
                "cpu_seconds": round(cost()[0], 2),
                "wall_seconds": round(cost()[1], 2),
                "training_steps": 0,
                "optimizer_steps": 0,
                "backward_calls": BACKWARD_CALLS,
            },
            "gates": {"a1_credit_reachable": False},
        }
        write_json(REPORT, payload)
        print(json.dumps({"verdict": "credit_starvation", "a1": a1}, ensure_ascii=False, indent=2))
        return 0

    # A2: frozen tensors must all be None/zero; A3: F is the only shaped tensor.
    a2 = {
        "tensors": len(frozen_table),
        "grad_norms_present": sum(1 for row in frozen_table if row["grad_norm"] is not None),
        "nonzero": len(nonzero_frozen),
        "violations": nonzero_frozen,
        "pass": len(frozen_table) == 135 and not nonzero_frozen,
    }
    del frozen_table

    # Control: thaw the same 135 tensors (no optimizer, no step) to show the
    # graph is live and that the zeros above come from the declared freeze.
    thawed = []
    for name, parameter in cf1.subject.named_parameters():
        parameter.requires_grad_(True)
        thawed.append(name)
    thaw_fast = history["final_state"].living.fast.detach().clone().requires_grad_(True)
    thaw_logits, _ = forward_credit(
        cf1, history["final_state"], probe_turn, released, thaw_fast, now=CREDIT_PROBE_TURN
    )
    thaw_loss = task_loss(thaw_logits, action)
    backward(thaw_loss)
    thaw_fast_norm = None if thaw_fast.grad is None else float(thaw_fast.grad.norm())
    thaw_table = grad_table(cf1.subject, frozen=False)
    param_norms = [row["grad_norm"] for row in thaw_table if row["grad_norm"] is not None]
    param_total = sum(norm * norm for norm in param_norms) ** 0.5 if param_norms else 0.0
    live_tensors = sum(1 for norm in param_norms if norm > 0)
    share_thaw = (
        thaw_fast_norm / (thaw_fast_norm + param_total)
        if thaw_fast_norm is not None and param_total > 0
        else None
    )
    zero_logits, _ = forward_credit(
        cf1,
        history["final_state"],
        probe_turn,
        released,
        torch.zeros_like(fast_leaf),
        now=CREDIT_PROBE_TURN,
    )
    zero_loss = float(task_loss(zero_logits, action).detach())
    a3 = {
        "design_freeze_param_grads": 0,
        "design_freeze_fast_grad_norm": fast_norm,
        "design_freeze_share": 1.0,
        "thaw_control_forward_identical": float((thaw_logits - logits).detach().abs().max()),
        "thaw_control_param_grad_norm": param_total,
        "thaw_control_tensors_with_grad": live_tensors,
        "thaw_control_fast_grad_norm": thaw_fast_norm,
        "thaw_control_share": share_thaw,
        "zero_fast_loss": zero_loss,
        "f_influences_loss": abs(float(loss.detach()) - zero_loss),
        "pass": live_tensors > 0 and fast_norm is not None and fast_norm > 0,
    }
    for parameter in cf1.subject.parameters():
        parameter.requires_grad_(False)
    clear_grads(cf1.subject)
    weights_after_probe = weights_digest(cf1.subject)
    append_progress({"stage": "credit_a2_a3", "cpu": cost()[0]})

    # A4: binding trace for both releases.
    binding = binding_audit(world, baseline)
    followers = []
    for index, trace in enumerate(baseline["traces"]):
        following = []
        limit = (
            baseline["traces"][index + 1]["after_turn"]
            if index + 1 < len(baseline["traces"])
            else fixture.TOTAL_TURNS
        )
        for row in baseline["turns"]:
            if (
                row["turn"] > trace["after_turn"]
                and row["turn"] <= limit
                and row["correction_surface"]
            ):
                following.append({"turn": row["turn"], "kind": row["kind"]})
        followers.append(following)
    a4 = {
        "releases": [
            dict(trace, following_correction_turns=following)
            for trace, following in zip(baseline["traces"], followers, strict=True)
        ],
        "binding": binding,
        "pass": binding["binding_ok"] and len(baseline["traces"]) == 2,
    }
    append_progress({"stage": "credit_a4", "cpu": cost()[0]})

    # ---- B. controls ---------------------------------------------------------
    beh_run = run_sequence(beh, world, write_enabled=False)
    beh_accuracy = accuracy(beh_run, world)

    beh_fast = beh_run["final_state"].living.fast.detach().clone().requires_grad_(True)
    beh_history_state = run_sequence(
        beh, world, write_enabled=False, end_turn=CREDIT_PROBE_TURN - 1
    )["final_state"]
    beh_logits, _ = forward_credit(
        beh,
        beh_history_state,
        probe_turn,
        released_upto(world, CREDIT_PROBE_TURN - 1),
        beh_fast,
        now=CREDIT_PROBE_TURN,
    )
    beh_loss = task_loss(beh_logits, action)
    backward(beh_loss)
    beh_fast_norm = None if beh_fast.grad is None else float(beh_fast.grad.norm())
    clear_grads(beh.subject)
    for parameter in beh.subject.parameters():
        parameter.requires_grad_(True)
    beh_thaw_fast = beh_history_state.living.fast.detach().clone().requires_grad_(True)
    beh_thaw_logits, _ = forward_credit(
        beh,
        beh_history_state,
        probe_turn,
        released_upto(world, CREDIT_PROBE_TURN - 1),
        beh_thaw_fast,
        now=CREDIT_PROBE_TURN,
    )
    beh_thaw_loss = task_loss(beh_thaw_logits, action)
    backward(beh_thaw_loss)
    beh_thaw_fast_norm = None if beh_thaw_fast.grad is None else float(beh_thaw_fast.grad.norm())
    beh_thaw_table = grad_table(beh.subject, frozen=False)
    beh_param_norms = [row["grad_norm"] for row in beh_thaw_table if row["grad_norm"] is not None]
    beh_param_total = (
        sum(norm * norm for norm in beh_param_norms) ** 0.5 if beh_param_norms else 0.0
    )
    beh_share_thaw = (
        beh_thaw_fast_norm / (beh_thaw_fast_norm + beh_param_total)
        if beh_thaw_fast_norm is not None and beh_param_total > 0
        else None
    )
    del beh_thaw_table
    for parameter in beh.subject.parameters():
        parameter.requires_grad_(False)
    clear_grads(beh.subject)

    scrambled = scrambled_world(world, random.Random(fixture.SEED + 202))
    shuffled_run = run_sequence(cf1, scrambled, write_enabled=True)
    shuffled_accuracy = accuracy(shuffled_run, world)
    shuffled_history = run_sequence(
        cf1, scrambled, write_enabled=True, end_turn=CREDIT_PROBE_TURN - 1
    )
    shuffled_fast = (
        shuffled_history["final_state"].living.fast.detach().clone().requires_grad_(True)
    )
    shuffled_logits, _ = forward_credit(
        cf1,
        shuffled_history["final_state"],
        probe_turn,
        released_upto(scrambled, CREDIT_PROBE_TURN - 1),
        shuffled_fast,
        now=CREDIT_PROBE_TURN,
    )
    shuffled_loss = task_loss(shuffled_logits, action)
    backward(shuffled_loss)
    shuffled_fast_norm = None if shuffled_fast.grad is None else float(shuffled_fast.grad.norm())
    real_grad_vector = fast_leaf.grad.detach().reshape(-1) if fast_leaf.grad is not None else None
    shuffled_grad_vector = (
        shuffled_fast.grad.detach().reshape(-1) if shuffled_fast.grad is not None else None
    )
    cosine = None
    if real_grad_vector is not None and shuffled_grad_vector is not None:
        denominator = float(real_grad_vector.norm() * shuffled_grad_vector.norm())
        if denominator > 0:
            cosine = float(torch.dot(real_grad_vector, shuffled_grad_vector) / denominator)
    fast_divergence = float(
        (history["final_state"].living.fast - shuffled_history["final_state"].living.fast)
        .abs()
        .max()
    )
    clear_grads(cf1.subject)

    # Same "error share reaching F" under the thaw reference, so the frozen
    # share (=1.0 by construction) does not hide the comparison.
    for parameter in cf1.subject.parameters():
        parameter.requires_grad_(True)
    shuff_thaw_fast = (
        shuffled_history["final_state"].living.fast.detach().clone().requires_grad_(True)
    )
    shuff_thaw_logits, _ = forward_credit(
        cf1,
        shuffled_history["final_state"],
        probe_turn,
        released_upto(scrambled, CREDIT_PROBE_TURN - 1),
        shuff_thaw_fast,
        now=CREDIT_PROBE_TURN,
    )
    shuff_thaw_loss = task_loss(shuff_thaw_logits, action)
    backward(shuff_thaw_loss)
    shuff_thaw_fast_norm = (
        None if shuff_thaw_fast.grad is None else float(shuff_thaw_fast.grad.norm())
    )
    shuff_thaw_table = grad_table(cf1.subject, frozen=False)
    shuff_param_norms = [
        row["grad_norm"] for row in shuff_thaw_table if row["grad_norm"] is not None
    ]
    shuff_param_total = (
        sum(norm * norm for norm in shuff_param_norms) ** 0.5 if shuff_param_norms else 0.0
    )
    share_thaw_scrambled = (
        shuff_thaw_fast_norm / (shuff_thaw_fast_norm + shuff_param_total)
        if shuff_thaw_fast_norm is not None and shuff_param_total > 0
        else None
    )
    del shuff_thaw_table
    for parameter in cf1.subject.parameters():
        parameter.requires_grad_(False)
    clear_grads(cf1.subject)

    relative_change = (
        abs(shuffled_fast_norm - fast_norm) / max(abs(fast_norm), 1e-12)
        if fast_norm not in (None, 0.0) and shuffled_fast_norm is not None
        else None
    )
    share_change = (
        abs(share_thaw_scrambled - share_thaw) / max(abs(share_thaw), 1e-12)
        if share_thaw not in (None, 0.0) and share_thaw_scrambled is not None
        else None
    )
    b_eh_section = {
        "fast_writes": beh_run["fast_writes"],
        "correction_accuracy": beh_accuracy,
        "fast_grad_norm": beh_fast_norm,
        "share_of_loss": (
            None if beh_fast_norm is None else beh_fast_norm / (float(beh_loss.detach()) + 1e-12)
        ),
        "share_thaw": beh_share_thaw,
    }
    shuffled_section = {
        "fast_writes": shuffled_run["fast_writes"],
        "correction_accuracy": shuffled_accuracy,
        "fast_grad_norm": shuffled_fast_norm,
        "norm_change_vs_real": relative_change,
        "share_thaw": share_thaw_scrambled,
        "share_change_vs_real": share_change,
        "direction_cosine_vs_real": cosine,
        "fast_divergence_vs_real": fast_divergence,
        "probe_loss": float(shuffled_loss.detach()),
        "output_not_better": shuffled_accuracy["full_match"]
        <= accuracy(baseline, world)["full_match"],
    }
    transparent = run_sequence(cf1, world, write_enabled=False)
    parity = compare_runs(transparent, beh_run)
    controls = {
        "b_eh": b_eh_section,
        "feedback_scrambled": shuffled_section,
        "parity_fast_off": parity,
        "real_correction_accuracy": accuracy(baseline, world),
        "real_share_thaw": share_thaw,
        "pass": (
            beh_run["fast_writes"] == 0
            and share_change is not None
            and share_change >= 0.5
            and shuffled_section["output_not_better"]
            and parity["byte_identical"] == 765
        ),
    }
    append_progress({"stage": "controls", "cpu": cost()[0], "wall": cost()[1]})

    # ---- C. full save, cross-session resume, crash recovery ------------------
    RECOVERY.mkdir(parents=True, exist_ok=True)
    mid_path = RECOVERY / "mid_session2_turn18.pt"
    boundary_path = RECOVERY / "end_session1_turn10.pt"
    truncated_path = RECOVERY / "mid_session2_turn18_truncated.pt"

    mid_history = run_sequence(cf1, world, write_enabled=True, end_turn=CREDIT_PROBE_TURN)
    mid_payload = checkpoint_payload(
        cf1, mid_history["final_state"], mid_history["released"], CREDIT_PROBE_TURN
    )
    torch.save(mid_payload, mid_path)
    mid_sha = sha256_file(mid_path)

    resumed = fresh_graph(parent, gate, "CF1")
    loaded = load_checkpoint(mid_path, resumed)
    restored = resumed.restore_state(loaded["rule_state"])
    mid_exact = state_digest(restored) == state_digest(mid_history["final_state"])
    resumed_run = run_sequence(
        resumed,
        world,
        write_enabled=True,
        start_state=restored,
        start_turn=CREDIT_PROBE_TURN,
        end_turn=CREDIT_PROBE_TURN + RESUME_SPAN,
        released=loaded["released"],
    )
    baseline_slice = {
        "turns": [
            row
            for row in baseline["turns"]
            if CREDIT_PROBE_TURN < row["turn"] <= CREDIT_PROBE_TURN + RESUME_SPAN
        ]
    }
    full_compare = compare_runs(baseline_slice, resumed_run)
    fields = state_field_table(mid_history["final_state"], restored, loaded)

    boundary_history = run_sequence(cf1, world, write_enabled=True, end_turn=10)
    boundary_payload = checkpoint_payload(
        cf1, boundary_history["final_state"], boundary_history["released"], 10
    )
    torch.save(boundary_payload, boundary_path)
    fast_at_boundary = boundary_history["final_state"].living.fast.clone()
    initial_fast = cf1.initial_state("w000").living.fast.clone()
    fast_written_before_boundary = not torch.equal(fast_at_boundary, initial_fast)

    cross = fresh_graph(parent, gate, "CF1")
    cross_loaded = load_checkpoint(boundary_path, cross)
    cross_restored = cross.restore_state(cross_loaded["rule_state"])
    fast_after_load = cross_restored.living.fast
    fast_persisted = bool(torch.equal(fast_after_load, fast_at_boundary))
    journal_persisted = len(cross_restored.journal) == len(boundary_history["final_state"].journal)
    cross_run = run_sequence(
        cross,
        world,
        write_enabled=True,
        start_state=cross_restored,
        start_turn=10,
        end_turn=10 + RESUME_SPAN,
        released=cross_loaded["released"],
    )
    cross_slice = {
        "turns": [row for row in baseline["turns"] if 10 < row["turn"] <= 10 + RESUME_SPAN]
    }
    cross_compare = compare_runs(cross_slice, cross_run)

    # Crash: truncate a written file before any rename, then recover from the
    # last valid snapshot.
    data = mid_path.read_bytes()
    truncated_path.write_bytes(data[: max(1, len(data) // 3)])
    crash_rejected = False
    crash_error = None
    try:
        load_checkpoint(truncated_path, fresh_graph(parent, gate, "CF1"))
    except Exception as error:  # noqa: BLE001 - the rejection is the reading
        crash_rejected = True
        crash_error = type(error).__name__
    crash_recovered = fresh_graph(parent, gate, "CF1")
    crash_payload = load_checkpoint(mid_path, crash_recovered)
    crash_state = crash_recovered.restore_state(crash_payload["rule_state"])
    crash_run = run_sequence(
        crash_recovered,
        world,
        write_enabled=True,
        start_state=crash_state,
        start_turn=CREDIT_PROBE_TURN,
        end_turn=CREDIT_PROBE_TURN + RESUME_SPAN,
        released=crash_payload["released"],
    )
    crash_compare = compare_runs(baseline_slice, crash_run)
    canonical_sha_after_crash = sha256_file(mid_path)

    recovery = {
        "full": {
            "checkpoint": str(mid_path.relative_to(ROOT)),
            "sha256": mid_sha,
            "bytes": mid_path.stat().st_size,
            "state_restore_exact": mid_exact,
            "resumed_turns": len(resumed_run["turns"]),
            "compare": full_compare,
            "pass": mid_exact and full_compare["byte_identical"] == full_compare["bytes_compared"],
        },
        "cross_session": {
            "checkpoint": str(boundary_path.relative_to(ROOT)),
            "sha256": sha256_file(boundary_path),
            "fast_written_before_boundary": fast_written_before_boundary,
            "fast_persisted_after_load": fast_persisted,
            "journal_persisted": journal_persisted,
            "f_reset_at_boundary": False,
            "resumed_turns": len(cross_run["turns"]),
            "compare": cross_compare,
            "pass": (
                fast_persisted
                and journal_persisted
                and cross_compare["byte_identical"] == cross_compare["bytes_compared"]
            ),
        },
        "crash": {
            "artifact": str(truncated_path.relative_to(ROOT)),
            "bytes": truncated_path.stat().st_size,
            "load_rejected": crash_rejected,
            "error": crash_error,
            "canonical_unchanged": canonical_sha_after_crash == mid_sha,
            "recovered_turns": len(crash_run["turns"]),
            "compare": crash_compare,
            "pass": (
                crash_rejected
                and canonical_sha_after_crash == mid_sha
                and crash_compare["byte_identical"] == crash_compare["bytes_compared"]
            ),
        },
        "state_fields": fields,
        "pass": True,
    }
    recovery["pass"] = (
        recovery["full"]["pass"] and recovery["cross_session"]["pass"] and recovery["crash"]["pass"]
    )
    append_progress({"stage": "recovery", "cpu": cost()[0], "wall": cost()[1]})

    # ---- D. dry forward summary ---------------------------------------------
    shapes = sorted({tuple(row["logits"].shape) for row in baseline["turns"]})
    finite = all(bool(torch.isfinite(row["logits"]).all()) for row in baseline["turns"])
    dry_forward = {
        "turns_forwarded": len(baseline["turns"]),
        "logit_shape": list(shapes[0]) if len(shapes) == 1 else [list(shape) for shape in shapes],
        "answer_bytes": ANSWER_BYTES,
        "finite_logits": finite,
        "cf1_fast_writes": baseline["fast_writes"],
        "b_eh_fast_writes": beh_run["fast_writes"],
        "backward_calls": BACKWARD_CALLS,
        "turns_covered": sum(
            len(run["turns"])
            for run in (
                baseline,
                beh_run,
                shuffled_run,
                transparent,
                resumed_run,
                cross_run,
                crash_run,
            )
        )
        + 2,
    }

    cpu_seconds, wall_seconds = cost()
    gates = {
        "a1_credit_reachable": a1["verdict"] == "reachable",
        "a2_session_freeze": a2["pass"],
        "a3_f_only": a3["pass"],
        "a4_binding": a4["pass"],
        "b_eh_zero_writes": beh_run["fast_writes"] == 0,
        "feedback_shuffled_credit_changed": bool(share_change is not None and share_change >= 0.5),
        "feedback_shuffled_not_better": shuffled_section["output_not_better"],
        "parity_fast_off": parity["byte_identical"] == 765,
        "full_restore": recovery["full"]["pass"],
        "cross_session_restore": recovery["cross_session"]["pass"],
        "crash_recovery": recovery["crash"]["pass"],
        "zero_training": True,
        "state_assets_untouched": assemblies_intact,
        "parent_weights_unchanged": weights_after_probe == parent_sha,
    }
    verdict = "ready" if all(gates.values()) else "not_ready"
    report = {
        "package": "20.2",
        "date": "2026-11-05",
        "verdict": verdict,
        "scope": "credit chain, negative controls, full save and independent restore; 0 training",
        "credit": {"a1": a1, "a2": a2, "a3": a3, "a4": a4},
        "controls": controls,
        "recovery": recovery,
        "dry_forward": dry_forward,
        "gates": gates,
        "cost": {
            "cpu_seconds": round(cpu_seconds, 2),
            "wall_seconds": round(wall_seconds, 2),
            "cpu_cap": CPU_CAP,
            "wall_cap": WALL_CAP,
            "training_steps": 0,
            "optimizer_steps": 0,
            "backward_calls": BACKWARD_CALLS,
            "confirm_reads": 0,
            "new_parameters": 0,
            "frozen_parameters": frozen_count,
            "freeze_list": frozen_names,
        },
    }
    write_json(REPORT, report)

    manifest = {
        "package": "20.2",
        "date": "2026-11-05",
        "verdict": verdict,
        "report": str(REPORT.relative_to(ROOT)),
        "report_sha256": sha256_file(REPORT),
        "fixture_digest": fixture.fixture_digest([world]),
        "parent_bundle": str(BUNDLE.relative_to(ROOT)),
        "parent_bundle_sha256": sha256_file(BUNDLE),
        "parent_parameter_sha256": parent_sha,
        "reference_assemblies": {
            "cf1": {
                "path": str(CF1_STATE.relative_to(ROOT)),
                "sha256": state_shas["cf1"],
                "label": "initialized_untrained",
            },
            "b_eh": {
                "path": str(BEH_STATE.relative_to(ROOT)),
                "sha256": state_shas["b_eh"],
                "label": "initialized_untrained",
            },
            "untouched": assemblies_intact,
        },
        "saved_states": [
            {"path": str(mid_path.relative_to(ROOT)), "sha256": mid_sha, "label": "recovery_copy"},
            {
                "path": str(boundary_path.relative_to(ROOT)),
                "sha256": sha256_file(boundary_path),
                "label": "recovery_copy",
            },
            {
                "path": str(truncated_path.relative_to(ROOT)),
                "sha256": sha256_file(truncated_path),
                "label": "crash_artifact",
            },
        ],
        "saved_state_count": 3,
        "gates": gates,
        "cost": report["cost"],
        "training_admitted": False,
    }
    write_json(MANIFEST, manifest)
    print(
        json.dumps(
            {"verdict": verdict, "gates": gates, "cost": report["cost"]},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
