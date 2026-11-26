"""20.4 pricing and freeze: checkpoint roundtrip, one measured update, frozen protocol.

Zero formal training is started before this probe (1) proves that a save -> load ->
resume sequence is byte identical for weights, Adam moments, RNG state and the
data cursor, and (2) prices one real outer update from the actual graph rather
than from an estimate.  If the extrapolated formal cost exceeds the frozen
magnitude ceiling the probe stops and only reports numbers.

The graph module is imported unchanged: the differentiable trajectory mirror
lives in this probe because the shipped ``run_turn`` / ``release`` are
``torch.no_grad`` inference paths.  No confirm world is read, ``addressed_store``
stays at its frozen value (off), and no threshold is altered.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

torch.set_num_threads(1)

from taiji import rule_learning_20_1 as fixture  # noqa: E402
from taiji.living_core import EOS  # noqa: E402
from taiji.rule_learning_20_1_net import (  # noqa: E402
    ANSWER_BYTES,
    DECISION_POSITIONS,
    TEMPLATE_BYTES,
    RuleLearningGraph,
    _group_of,
    copy_subject,
)
from taiji.selection_routing_checkpoint import load_routing_bundle  # noqa: E402

REPORT = ROOT / "reports/taiji_package20_4_pricing_freeze_20261105.json"
MANIFEST = ROOT / "plans/manifests/taiji_package20_4_pricing_freeze_20261105.json"
FREEZE = ROOT / "plans/manifests/taiji_package20_4_freeze_20261105.json"
PROGRESS = ROOT / "reports/taiji_package20_4_pricing_freeze_20261105_progress.jsonl"
CHECKPOINT = ROOT / "output/taiji_package20_4_checkpoints/preflight.pt"
BUNDLE = ROOT / "output/taiji_package19_compare/28031-B_stop_lock-step300.pt"

CPU_CAP = 6000.0
CPU_GUARD = 5900.0
WALL_CAP = 14400.0
FROZEN_BUDGET_CPU = 24000.0
MAGNITUDE_CEILING_CPU = 2 * FROZEN_BUDGET_CPU
UPDATES_PER_SEED = 200
SEEDS = (20261105, 20261106)
ARMS = ("CF1", "B_EH")
BATCH = 2
LEARNING_RATE = 1e-3
GRAD_CLIP = 1.0
TRAIN_INDICES = tuple(range(0, 160))
DEV_INDICES = tuple(range(160, 240))
CONFIRM_INDICES = (240, 319)
ADDRESSED_STORE = False
EARLY_STOP_STEPS = (25, 50, 100)
EARLY_STOP_GAIN = 1e-4
PREFLIGHT_UPDATES = 3
RESUME_UPDATES = 3
TOTAL_TURNS = fixture.TOTAL_TURNS

START_WALL = time.time()
START_CPU = time.process_time()


class BudgetStop(RuntimeError):
    pass


def cost() -> tuple[float, float]:
    return time.process_time() - START_CPU, time.time() - START_WALL


def guard(stage: str) -> None:
    cpu, wall = cost()
    if cpu > CPU_GUARD or wall > WALL_CAP:
        raise BudgetStop(f"budget stop at {stage}: cpu={cpu:.2f} wall={wall:.2f}")


def append_progress(record: dict) -> None:
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
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


def tensor_digest(tensors: dict) -> str:
    rows = []
    for name, value in sorted(tensors.items()):
        data = value.detach().cpu().numpy().tobytes()
        rows.append((name, str(value.dtype), tuple(value.shape), sha256_bytes(data)))
    return sha256_bytes(repr(rows).encode("utf-8"))


def weights_digest(subject) -> str:
    return tensor_digest(dict(subject.state_dict()))


def optimizer_digest(optimizer) -> str:
    rows = []
    for group in optimizer.param_groups:
        for _index, parameter in enumerate(group["params"]):
            state = optimizer.state.get(parameter, {})
            for key in ("step", "exp_avg", "exp_avg_sq"):
                value = state.get(key)
                if value is None:
                    rows.append((key, "missing"))
                elif isinstance(value, torch.Tensor):
                    rows.append(
                        (
                            key,
                            str(value.dtype),
                            tuple(value.shape),
                            sha256_bytes(value.detach().cpu().numpy().tobytes()),
                        )
                    )
                else:
                    rows.append((key, float(value)))
    return sha256_bytes(repr(rows).encode("utf-8"))


def rng_digest() -> str:
    return sha256_bytes(torch.get_rng_state().numpy().tobytes())


def unfreeze_outer(subject) -> list[str]:
    """All declared groups except the frozen world head participate in outer updates."""
    trainable = []
    for name, parameter in subject.named_parameters():
        if _group_of(name) == "world":
            parameter.requires_grad_(False)
        else:
            parameter.requires_grad_(True)
            trainable.append(name)
    return trainable


def gold_bytes(world: dict, number: int) -> bytes:
    return world["scoring"]["turns"][str(number)]["expected_action"].encode("utf-8")


def legal_vocab(released: list[dict]) -> tuple[int, ...]:
    ids = set(TEMPLATE_BYTES.values())
    for item in released:
        ids.update(item["feedback"].encode("utf-8"))
    return tuple(sorted(ids))


def train_turn(graph, state, turn: dict, released: list[dict], now: int):
    """Grad-enabled mirror of ``run_turn`` with ``@torch.no_grad`` removed."""
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
    logits = []
    emitted = []
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
        emitted.append(choice)
        living = graph.subject.consume_output(output, choice)
        previous = choice
    child = type(state)(living, state.journal, state.revision)
    return torch.stack(logits), tuple(emitted), child


def train_release(graph, state, release: dict, released: list[dict], now: int, write: bool):
    records = graph._records(released, now)
    living = state.living
    written = 0
    for item in release["items"]:
        feedback = item["feedback"].encode("utf-8")
        query = item["query"].encode("utf-8")
        before = living.fast.detach().clone()
        living = graph.subject.observe(
            living,
            feedback,
            goal=query,
            records=records,
            allowed=graph._allowed(write),
            now=now,
            **graph.step_kwargs,
        )
        written += int(not torch.equal(before, living.fast.detach()))
    child = type(state)(living, state.journal, state.revision)
    return child, written


def run_world(graph, world: dict) -> dict:
    response = world["response"]
    state = graph.initial_state(response["world_id"])
    released: list[dict] = []
    losses = []
    decision_bits = 0
    template_bits = 0
    full_bytes = 0
    writes = 0
    surfaces = 0
    template_positions = [p for p in range(ANSWER_BYTES) if p not in DECISION_POSITIONS]
    for turn in response["turns"]:
        number = int(turn["turn"])
        logits, emitted, state = train_turn(graph, state, turn, released, number)
        gold = gold_bytes(world, number)
        decision_bits += sum(int(emitted[p] == gold[p]) for p in DECISION_POSITIONS)
        template_bits += sum(int(emitted[p] == gold[p]) for p in template_positions)
        full_bytes += int(tuple(emitted) == tuple(gold))
        if turn["correction_surface"] and released:
            surfaces += 1
            term = logits.new_zeros(())
            for position in DECISION_POSITIONS:
                term = term - torch.log(logits[position][gold[position]] + 1e-12)
            losses.append(term)
        for release in response["releases"]:
            if int(release["after_turn"]) != number:
                continue
            state, count = train_release(
                graph, state, release, released, number, bool(graph.fast_enabled)
            )
            writes += count
            released.extend(
                {
                    "source_id": item["source_id"],
                    "feedback": item["feedback"],
                    "version": item["version"],
                    "after_turn": number,
                }
                for item in release["items"]
            )
    loss = torch.stack(losses).sum() if losses else None
    return {
        "loss": loss,
        "surfaces": surfaces,
        "decision_bit_accuracy": decision_bits / (surfaces * len(DECISION_POSITIONS)),
        "template_bit_accuracy": template_bits / (TOTAL_TURNS * len(template_positions)),
        "full_byte_accuracy": full_bytes / TOTAL_TURNS,
        "fast_writes": writes,
    }


def update_once(graph, optimizer, worlds: list[dict], *, train: bool) -> dict:
    optimizer.zero_grad(set_to_none=True)
    totals = []
    rows = []
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for world in worlds:
            row = run_world(graph, world)
            rows.append(
                {
                    "world": world["world_id"],
                    "surfaces": row["surfaces"],
                    "decision_bit_accuracy": row["decision_bit_accuracy"],
                    "template_bit_accuracy": row["template_bit_accuracy"],
                    "full_byte_accuracy": row["full_byte_accuracy"],
                    "fast_writes": row["fast_writes"],
                }
            )
            totals.append(row["loss"])
        loss = torch.stack(totals).sum() if train else None
    if train:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in graph.subject.parameters() if p.requires_grad], GRAD_CLIP
        )
        optimizer.step()
    return {
        "loss": None if loss is None else float(loss.detach()),
        "worlds": rows,
    }


def save_checkpoint(path: Path, graph, optimizer, *, cursor: int, updates: int) -> str:
    payload = {
        "schema": "taiji-rule-learning-20-4-checkpoint-v1",
        "updates": int(updates),
        "cursor": int(cursor),
        "subject": {
            name: value.detach().cpu().clone() for name, value in graph.subject.state_dict().items()
        },
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "rng": torch.get_rng_state().cpu().clone(),
        "protocol": {
            "arms": list(ARMS),
            "seeds": list(SEEDS),
            "updates_per_seed": UPDATES_PER_SEED,
            "batch": BATCH,
            "learning_rate": LEARNING_RATE,
            "grad_clip": GRAD_CLIP,
            "addressed_store_enabled": ADDRESSED_STORE,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return sha256_bytes(path.read_bytes())


def load_checkpoint(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "taiji-rule-learning-20-4-checkpoint-v1":
        raise ValueError("unexpected checkpoint schema")
    return payload


def build_graph(candidate, gate, arm: str) -> RuleLearningGraph:
    return RuleLearningGraph(
        copy_subject(candidate.subject),
        arm=arm,
        growth_gates=gate,
        addressed_store_enabled=ADDRESSED_STORE,
    )


def main() -> int:
    try:
        return run()
    except BudgetStop as stop:
        append_progress({"stage": "budget_stop", "message": str(stop), "cpu": round(cost()[0], 3)})
        print(json.dumps({"verdict": "budget_stop", "message": str(stop)}, ensure_ascii=False))
        return 0


def run() -> int:
    if PROGRESS.exists():
        PROGRESS.unlink()
    append_progress(
        {
            "stage": "protocol",
            "arms": list(ARMS),
            "seeds": list(SEEDS),
            "updates_per_seed": UPDATES_PER_SEED,
            "batch": BATCH,
            "learning_rate": LEARNING_RATE,
            "train_indices": [TRAIN_INDICES[0], TRAIN_INDICES[-1]],
            "dev_indices": [DEV_INDICES[0], DEV_INDICES[-1]],
            "addressed_store_enabled": ADDRESSED_STORE,
            "cpu_cap": CPU_CAP,
        }
    )

    bundle = torch.load(BUNDLE, map_location="cpu", weights_only=False)
    candidate = load_routing_bundle(BUNDLE)[0]
    gate = dict(bundle["growth_gates"][0]) if bundle.get("growth_gates") else None
    cf1 = build_graph(candidate, gate, "CF1")
    trainable = unfreeze_outer(cf1.subject)
    optimizer = torch.optim.Adam(
        [p for p in cf1.subject.parameters() if p.requires_grad], lr=LEARNING_RATE
    )
    append_progress(
        {
            "stage": "build",
            "trainable_tensors": len(trainable),
            "trainable_elements": int(
                sum(p.numel() for p in cf1.subject.parameters() if p.requires_grad)
            ),
            "cpu": round(cost()[0], 3),
        }
    )

    # ---- preflight: save -> load -> resume must be byte identical -----------------
    needed_worlds = 2 * (PREFLIGHT_UPDATES + RESUME_UPDATES)
    train_worlds = [fixture.build_world(SEEDS[0], index) for index in TRAIN_INDICES[:needed_worlds]]
    continuous = []
    measured = []
    measured_wall = []
    cursor = 0
    for step in range(PREFLIGHT_UPDATES):
        guard("preflight")
        started = time.process_time()
        started_wall = time.time()
        row = update_once(cf1, optimizer, train_worlds[cursor : cursor + 2], train=True)
        row["update_cpu"] = round(time.process_time() - started, 3)
        row["update_wall"] = round(time.time() - started_wall, 3)
        row["step"] = step + 1
        row["cursor"] = cursor + 2
        row["weights"] = weights_digest(cf1.subject)
        row["optimizer"] = optimizer_digest(optimizer)
        row["rng"] = rng_digest()
        measured.append(row["update_cpu"])
        measured_wall.append(row["update_wall"])
        cursor += 2
        continuous.append(row)
        append_progress({"stage": "preflight_step", "step": row["step"], **row["worlds"][0]})
    checkpoint_sha = save_checkpoint(
        CHECKPOINT, cf1, optimizer, cursor=cursor, updates=PREFLIGHT_UPDATES
    )
    append_progress(
        {
            "stage": "preflight_saved",
            "sha": checkpoint_sha,
            "cursor": cursor,
            "cpu": round(cost()[0], 3),
        }
    )

    reference = []
    for step in range(RESUME_UPDATES):
        guard("continuation")
        started = time.process_time()
        started_wall = time.time()
        row = update_once(cf1, optimizer, train_worlds[cursor : cursor + 2], train=True)
        row["update_cpu"] = round(time.process_time() - started, 3)
        row["update_wall"] = round(time.time() - started_wall, 3)
        row["step"] = PREFLIGHT_UPDATES + step + 1
        row["cursor"] = cursor + 2
        row["weights"] = weights_digest(cf1.subject)
        row["optimizer"] = optimizer_digest(optimizer)
        row["rng"] = rng_digest()
        measured.append(row["update_cpu"])
        measured_wall.append(row["update_wall"])
        cursor += 2
        reference.append(row)

    payload = load_checkpoint(CHECKPOINT)
    resumed = build_graph(candidate, gate, "CF1")
    unfreeze_outer(resumed.subject)
    resumed.subject.load_state_dict(payload["subject"])
    resumed_opt = torch.optim.Adam(
        [p for p in resumed.subject.parameters() if p.requires_grad], lr=LEARNING_RATE
    )
    resumed_opt.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["rng"])
    resumed_rows = []
    resumed_cursor = int(payload["cursor"])
    for step in range(RESUME_UPDATES):
        guard("resume")
        started = time.process_time()
        started_wall = time.time()
        row = update_once(
            resumed,
            resumed_opt,
            train_worlds[resumed_cursor : resumed_cursor + 2],
            train=True,
        )
        row["update_cpu"] = round(time.process_time() - started, 3)
        row["update_wall"] = round(time.time() - started_wall, 3)
        row["step"] = int(payload["updates"]) + step + 1
        row["cursor"] = resumed_cursor + 2
        row["weights"] = weights_digest(resumed.subject)
        row["optimizer"] = optimizer_digest(resumed_opt)
        row["rng"] = rng_digest()
        measured.append(row["update_cpu"])
        measured_wall.append(row["update_wall"])
        resumed_cursor += 2
        resumed_rows.append(row)

    resume_checks = []
    for left, right in zip(reference, resumed_rows, strict=True):
        resume_checks.append(
            {
                "step": left["step"],
                "loss_byte_identical": left["loss"] == right["loss"],
                "weights_byte_identical": left["weights"] == right["weights"],
                "optimizer_byte_identical": left["optimizer"] == right["optimizer"],
                "rng_byte_identical": left["rng"] == right["rng"],
                "worlds_byte_identical": left["worlds"] == right["worlds"],
                "cursor_byte_identical": left["cursor"] == right["cursor"],
            }
        )
    preflight_ok = all(
        all(value for key, value in row.items() if key != "step") for row in resume_checks
    )
    append_progress(
        {
            "stage": "preflight_verified",
            "ok": preflight_ok,
            "checks": resume_checks,
            "cpu": round(cost()[0], 3),
        }
    )
    if not preflight_ok:
        verdict = "checkpoint_unreliable"
        payload_out = {
            "schema": "taiji-rule-learning-20-4-pricing-v1",
            "verdict": verdict,
            "preflight": {"ok": False, "checks": resume_checks, "sha": checkpoint_sha},
            "cost": {"cpu_seconds": round(cost()[0], 3), "wall_seconds": round(cost()[1], 3)},
            "script": {"runner_sha256": sha256_bytes(Path(__file__).read_bytes())},
        }
        write_json(REPORT, payload_out)
        print(json.dumps({"verdict": verdict}, ensure_ascii=False))
        return 0

    # ---- pricing: real updates measured in-process, one dev world eval -------------
    per_update_cpu = sum(measured) / len(measured)
    eval_started_cpu = time.process_time()
    eval_started_wall = time.time()
    eval_world = fixture.build_world(SEEDS[0], DEV_INDICES[0])
    eval_row = update_once(resumed, resumed_opt, [eval_world], train=False)
    eval_cpu = time.process_time() - eval_started_cpu
    eval_wall = time.time() - eval_started_wall
    wall_ratio = (sum(measured_wall) / len(measured_wall)) / per_update_cpu
    append_progress(
        {
            "stage": "pricing",
            "samples_cpu": measured,
            "samples_wall": measured_wall,
            "per_update_cpu": round(per_update_cpu, 3),
            "per_update_cpu_max": round(max(measured), 3),
            "per_eval_world_cpu": round(eval_cpu, 3),
            "per_eval_world_wall": round(eval_wall, 3),
            "cpu": round(cost()[0], 3),
        }
    )

    runs = len(ARMS) * len(SEEDS)
    train_cpu = per_update_cpu * UPDATES_PER_SEED * runs
    eval_cpu_total = eval_cpu * len(DEV_INDICES) * runs
    probe_cpu = eval_cpu * 5 * len(EARLY_STOP_STEPS) * runs
    total_cpu = train_cpu + eval_cpu_total + probe_cpu
    total_wall = total_cpu * wall_ratio
    projection = {
        "runs": runs,
        "updates_per_run": UPDATES_PER_SEED,
        "train_cpu": round(train_cpu, 1),
        "dev_eval_cpu": round(eval_cpu_total, 1),
        "early_stop_probe_cpu": round(probe_cpu, 1),
        "total_cpu": round(total_cpu, 1),
        "total_wall": round(total_wall, 1),
        "frozen_budget_cpu": FROZEN_BUDGET_CPU,
        "magnitude_ceiling_cpu": MAGNITUDE_CEILING_CPU,
        "round_cap_cpu": CPU_CAP,
        "ratio_to_frozen_budget": round(total_cpu / FROZEN_BUDGET_CPU, 2),
        "within_magnitude_ceiling": total_cpu <= MAGNITUDE_CEILING_CPU,
        "within_round_cap": total_cpu + cost()[0] <= CPU_CAP,
    }
    append_progress({"stage": "projection", **projection})

    protocol = {
        "data": {
            "train_worlds": [TRAIN_INDICES[0], TRAIN_INDICES[-1]],
            "dev_worlds": [DEV_INDICES[0], DEV_INDICES[-1]],
            "confirm_worlds_reserved": list(CONFIRM_INDICES),
            "confirm_read": False,
        },
        "training": {
            "arms": list(ARMS),
            "seeds": list(SEEDS),
            "updates_per_seed": UPDATES_PER_SEED,
            "batch_worlds": BATCH,
            "learning_rate": LEARNING_RATE,
            "grad_clip": GRAD_CLIP,
            "addressed_store_enabled": ADDRESSED_STORE,
            "checkpoint_every_updates": 100,
            "reading_every_updates": 25,
        },
        "early_stop": {
            "steps": list(EARLY_STOP_STEPS),
            "probe": "five counterfactual F arms, 0 optimizer steps",
            "criterion": (
                f"two consecutive probes with leverage_max < {EARLY_STOP_GAIN} "
                "and byte_diff == 0"
            ),
            "verdict": "f_channel_dead_under_training",
        },
        "error_gates": {
            "all_51_turns_kept": True,
            "correction_surfaces": "22 separate",
            "post_feedback_first_answer": "separate",
            "per_arm_judgement": True,
            "template_vs_content": "reported separately",
        },
        "pricing": projection,
    }

    if total_cpu > MAGNITUDE_CEILING_CPU:
        verdict = "pricing_over_budget"
        detail = (
            f"projected {total_cpu:.0f} CPU = {total_cpu / FROZEN_BUDGET_CPU:.1f}x the "
            f"frozen budget; the magnitude ceiling is {MAGNITUDE_CEILING_CPU:.0f} CPU, "
            "so the bounded comparison is not started and the protocol is not frozen"
        )
    elif not projection["within_round_cap"]:
        verdict = "budget_stop"
        detail = (
            f"projected {total_cpu:.0f} CPU exceeds the round cap of {CPU_CAP:.0f} CPU; "
            "the formal comparison is not started and the protocol is not frozen"
        )
    else:
        verdict = "ready_to_run"
        detail = (
            f"projected {total_cpu:.0f} CPU is inside the magnitude ceiling and the round cap; "
            "the bounded comparison is admitted but not started in this probe"
        )
    append_progress({"stage": "verdict", "verdict": verdict, "detail": detail})

    freeze_written = verdict == "ready_to_run"
    freeze_record = {"written": False, "path": str(FREEZE.relative_to(ROOT)), "sha256": None}
    if freeze_written:
        freeze_payload = {
            "schema": "taiji-rule-learning-20-4-freeze-v1",
            "frozen_at_cpu": round(cost()[0], 3),
            "status": "frozen",
            **protocol,
        }
        write_json(FREEZE, freeze_payload)
        freeze_record = {
            "written": True,
            "path": str(FREEZE.relative_to(ROOT)),
            "sha256": sha256_bytes(FREEZE.read_bytes()),
        }
        append_progress({"stage": "freeze_written", "cpu": round(cost()[0], 3)})

    baseline_rows = continuous[0]["worlds"]
    baseline = {
        "decision_bit_accuracy": round(
            sum(row["decision_bit_accuracy"] for row in baseline_rows) / len(baseline_rows), 6
        ),
        "template_bit_accuracy": round(
            sum(row["template_bit_accuracy"] for row in baseline_rows) / len(baseline_rows), 6
        ),
        "full_byte_accuracy": round(
            sum(row["full_byte_accuracy"] for row in baseline_rows) / len(baseline_rows), 6
        ),
        "fast_writes_per_world": [row["fast_writes"] for row in baseline_rows],
    }
    payload_out = {
        "schema": "taiji-rule-learning-20-4-pricing-v1",
        "evidence": "20_4",
        "verdict": verdict,
        "detail": detail,
        "preflight": {
            "ok": True,
            "sha": checkpoint_sha,
            "checks": resume_checks,
            "saved_updates": PREFLIGHT_UPDATES,
            "resumed_updates": RESUME_UPDATES,
            "cursor_after_save": 2 * PREFLIGHT_UPDATES,
        },
        "pricing": projection,
        "measured": {
            "per_update_cpu": round(per_update_cpu, 3),
            "per_update_cpu_samples": measured,
            "per_update_wall_samples": measured_wall,
            "per_eval_world_cpu": round(eval_cpu, 3),
            "per_eval_world_wall": round(eval_wall, 3),
            "wall_ratio": round(wall_ratio, 3),
            "updates_measured": PREFLIGHT_UPDATES + RESUME_UPDATES,
            "eval_worlds_measured": 1,
        },
        "baseline_before_training": baseline,
        "eval_first_dev_world": eval_row["worlds"][0],
        "training_started": False,
        "freeze": freeze_record,
        "proposed_protocol": protocol if not freeze_written else None,
        "cost": {"cpu_seconds": round(cost()[0], 3), "wall_seconds": round(cost()[1], 3)},
        "script": {
            "runner_sha256": sha256_bytes(Path(__file__).read_bytes()),
            "submitted_runner_sha256": None,
        },
        "invariants": {
            "confirm_read": False,
            "graph_modified": False,
            "addressed_store_enabled": ADDRESSED_STORE,
            "optimizer_steps_preflight_only": True,
            "formal_training_steps": 0,
        },
    }
    write_json(REPORT, payload_out)
    manifest = {
        "schema": "taiji-rule-learning-20-4-manifest-v1",
        "evidence": "20_4",
        "verdict": verdict,
        "report_path": str(REPORT.relative_to(ROOT)),
        "report_sha256": sha256_bytes(REPORT.read_bytes()),
        "freeze_written": freeze_written,
        "freeze_path": freeze_record["path"],
        "freeze_sha256": freeze_record["sha256"],
        "runner_sha256": payload_out["script"]["runner_sha256"],
        "submitted_runner_sha256": None,
        "pricing": projection,
        "preflight_ok": True,
        "training_started": False,
    }
    write_json(MANIFEST, manifest)
    print(json.dumps({"verdict": verdict, "detail": detail}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
