"""20.4c r2: the formal bounded comparison, 200 updates x 2 arms x 2 seeds.

Owner ruling on the first attempt: an absolute free-memory floor is the wrong
instrument on Windows.  Cached pages push ``available`` around by gigabytes
while the machine is perfectly healthy, and the thing the floor was meant to
protect against is exactly one: paging inflating the CPU price.  So this run
replaces the floor with a slowdown detector built on the measured unit price.

* Calibration: the first three real updates of a fresh run must average
  ``<= 1.10 x 56.906`` CPU seconds, otherwise ``cpu_inflation`` stops the run.
* Re-measurement: every reading block (25 updates) recomputes that block's mean
  update cost.  Above ``1.25 x`` the baseline the run stops on ``cpu_inflation``;
  between ``1.10 x`` and ``1.25 x`` it records a warning and continues.
* Memory is no longer a gate.  Every reading still records available, commit and
  page-fault counters, and the report carries a "CPU price vs available" table so
  the substitution can be audited afterwards.
* One hard floor survives: ``available < 200 MB`` stops the run as
  ``out_of_memory``.

Checkpoints are now every 25 updates, each with its sha256, so no run can lose
more than 25 updates.  A restart loads the newest checkpoint of that run and
continues, records ``resumed: true`` and compares the replayed segment against
the continuous one.  The aborted first attempt (CF1/20261105, 25 updates,
2,514.203 CPU) is kept untouched under the ``20_4c`` prefix as historical
evidence and its cost is charged against the same 52,000 CPU cap.

The experiment itself is unchanged: 200 updates, two arms, two seeds, 51 turns,
22 correction surfaces, batch 2 worlds, F probes at 25/50/100/200.

No confirm world is read, ``addressed_store`` stays off, the frozen graph
modules (``living_core.py``, ``rule_learning_20_1_net.py``) are not touched, and
the executor is the additive ``living_core_batch.py`` proven equivalent in 20.4b.
"""

from __future__ import annotations

import contextlib
import copy
import gc
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

THREADS = 1
torch.set_num_threads(THREADS)
with contextlib.suppress(RuntimeError):
    torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True, warn_only=True)

from taiji import rule_learning_20_1 as fixture  # noqa: E402
from taiji.living_core import EOS  # noqa: E402
from taiji.living_core_batch import (  # noqa: E402
    BatchedLiving,
    align_index,
    consume_batch,
    context_batch,
    goal_vector_batch,
    growth_rows,
    initial_batched_state,
    patch_events_batch,
    patch_table,
    shift_index,
    source_view_batch,
    step_batch,
    update_constants,
)
from taiji.rule_learning_20_1_identity import graph_identity  # noqa: E402
from taiji.rule_learning_20_1_net import (  # noqa: E402
    ANSWER_BYTES,
    DECISION_POSITIONS,
    RuleLearningGraph,
    _group_of,
    copy_subject,
)
from taiji.selection_routing_checkpoint import load_routing_bundle  # noqa: E402

REPORT = ROOT / "reports/taiji_package20_4c_r2_formal_comparison_20261105.json"
MANIFEST = ROOT / "plans/manifests/taiji_package20_4c_r2_formal_comparison_20261105.json"
PROGRESS = ROOT / "reports/taiji_package20_4c_r2_formal_comparison_20261105_progress.jsonl"
FREEZE = ROOT / "plans/manifests/taiji_package20_4c_r2_freeze_20261105.json"
CHECKPOINTS = ROOT / "output/taiji_package20_4c_r2_checkpoints"
BUNDLE = ROOT / "output/taiji_package19_compare/28031-B_stop_lock-step300.pt"
PRIOR_REPORT = ROOT / "reports/taiji_package20_4c_formal_comparison_20261105.json"

CPU_CAP = 52000.0
CPU_GUARD = 51500.0
WALL_CAP = 90000.0
MEM_HARD_FLOOR_MB = 200.0
PRIOR_CPU_SECONDS = 2514.203
BASELINE_UNIT = 56.906
CALIBRATION_UPDATES = 3
CALIBRATION_RATIO = 1.10
INFLATION_WARN_RATIO = 1.10
INFLATION_STOP_RATIO = 1.25
VERIFY_SEED = 20261105
UPDATES_PER_RUN = 200
SEEDS = (20261105, 20261106)
ARMS = ("CF1", "B_EH")
BATCH = 2
LEARNING_RATE = 1e-3
GRAD_CLIP = 1.0
TRAIN_INDICES = tuple(range(0, 160))
DEV_INDICES = tuple(range(160, 240))
CONFIRM_INDICES = (240, 319)
ADDRESSED_STORE = False
READ_EVERY = 25
CHECKPOINT_EVERY = 25
EARLY_STOP_STEPS = (25, 50, 100)
PROBE_STEPS = EARLY_STOP_STEPS + (UPDATES_PER_RUN,)
EARLY_STOP_GAIN = 1e-4
LADDER_ARMS = ("f_zero", "f_half", "f_double", "f_quad", "f_permuted")
PERMUTE_SEED = 20261105
DIVERGENCE_FACTOR = 10.0
PROBE_WORLD_INDEX = 0
TOTAL_TURNS = fixture.TOTAL_TURNS
TEMPLATE_POSITIONS = tuple(p for p in range(ANSWER_BYTES) if p not in DECISION_POSITIONS)
REFERENCE_UNIT_FROM_20_4B = 56.906
REFERENCE_TOTAL_FROM_20_4B = 48129.2

START_WALL = time.time()
START_CPU = time.process_time()


class BudgetStop(RuntimeError):
    """Raised when the CPU, wall or memory guard fires."""

    def __init__(self, verdict: str, detail: str) -> None:
        super().__init__(detail)
        self.verdict = verdict
        self.detail = detail


def cost() -> tuple[float, float]:
    return time.process_time() - START_CPU, time.time() - START_WALL


def free_mb() -> float:
    import psutil

    return psutil.virtual_memory().available / (1024.0 * 1024.0)


def total_mb() -> float:
    import psutil

    return psutil.virtual_memory().total / (1024.0 * 1024.0)


def commit_mb() -> tuple[float, float]:
    """Windows commit charge: (used, limit).  Falls back to physical when absent."""
    import ctypes

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong),
            ("load", ctypes.c_ulong),
            ("total_phys", ctypes.c_ulonglong),
            ("avail_phys", ctypes.c_ulonglong),
            ("total_pagefile", ctypes.c_ulonglong),
            ("avail_pagefile", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("avail_virtual", ctypes.c_ulonglong),
            ("avail_extended", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return float("nan"), float("nan")
    scale = 1024.0 * 1024.0
    limit = status.total_pagefile / scale
    used = (status.total_pagefile - status.avail_pagefile) / scale
    return used, limit


def memory_sample() -> dict:
    """Available, commit and page-fault counters: recorded, never used as a gate."""
    import psutil

    available = free_mb()
    used, limit = commit_mb()
    info = psutil.Process().memory_info()
    return {
        "host_available_mb": round(available, 1),
        "host_commit_used_mb": round(used, 1),
        "host_commit_limit_mb": round(limit, 1),
        "proc_rss_mb": round(info.rss / (1024.0 * 1024.0), 1),
        "proc_page_faults": int(getattr(info, "num_page_faults", -1)),
    }


def total_cpu() -> float:
    """CPU seconds charged to the evidence, including the aborted 20_4c attempt."""
    return cost()[0] + PRIOR_CPU_SECONDS


def guard(stage: str) -> None:
    cpu, wall = cost()
    if cpu + PRIOR_CPU_SECONDS > CPU_GUARD:
        raise BudgetStop(
            "budget_cap_reached",
            f"CPU cap reached at {stage}: {total_cpu():.1f} of {CPU_CAP:.0f}",
        )
    if wall > WALL_CAP:
        raise BudgetStop("budget_cap_reached", f"wall cap reached at {stage}: {wall:.0f}")
    available = free_mb()
    if available < MEM_HARD_FLOOR_MB:
        raise BudgetStop(
            "out_of_memory",
            f"only {available:.0f} MB available at {stage} (hard floor {MEM_HARD_FLOOR_MB:.0f} MB)",
        )


def append_progress(row: dict) -> None:
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with PROGRESS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_progress() -> list[dict]:
    if not PROGRESS.exists():
        return []
    rows = []
    for line in PROGRESS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


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
        for parameter in group["params"]:
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


def build_graph(candidate, gate, arm: str) -> RuleLearningGraph:
    return RuleLearningGraph(
        copy_subject(candidate.subject),
        arm=arm,
        growth_gates=gate,
        addressed_store_enabled=ADDRESSED_STORE,
    )


def unfreeze_outer(subject) -> None:
    for name, parameter in subject.named_parameters():
        parameter.requires_grad_(_group_of(name) != "world")


def trainer_params(graph) -> list:
    return [p for p in graph.subject.parameters() if p.requires_grad]


def gold_bytes(world: dict, number: int) -> bytes:
    return world["scoring"]["turns"][str(number)]["expected_action"].encode("utf-8")


def world_at(seed: int, index: int) -> dict:
    return fixture.build_world(seed, index)


def accumulator() -> dict:
    return {
        "updates": 0,
        "cursor": 0,
        "losses": [],
        "decision_correct": 0,
        "surfaces": 0,
        "template_bits": 0,
        "template_total": 0,
        "content_bits": 0,
        "content_total": 0,
        "full_bytes": 0,
        "turns": 0,
        "fast_writes": 0,
        "update_cpu": [],
    }


# ---------------------------------------------------------------------------
# batched execution (same arithmetic as 20.4b, proven equivalent there)
# ---------------------------------------------------------------------------


def batched_observe(core, graph, states, raws, goals, records, now, allowed, const, parent, gates):
    reference = core.W
    batch = len(raws)
    widths = {len(raw) for raw in raws}
    if len(widths) != 1:
        raise RuntimeError(f"query lengths differ: {sorted(widths)}")
    view = source_view_batch(core, records, "owner", "scope", now, reference)
    first = align_index(states.keys, view.keys, reference)
    goal = goal_vector_batch(core.encoder, goals, reference)
    events = patch_events_batch(core.encoder, context_batch(core.encoder, raws, reference))
    for position in range(events.shape[1]):
        output = step_batch(
            core,
            states,
            event=events[:, position],
            goal_vector=goal,
            view=view,
            align=first if position == 0 else None,
            previous_byte=torch.full((batch,), EOS, dtype=torch.long),
            allowed=allowed,
            const=const,
            parent_units=parent,
            gate_rows=gates,
        )
        states = output.state
    return states, view, goal


def batched_turn(core, graph, states, turns, released_per_world, const, parent, gates, table):
    reference = core.W
    batch = len(turns)
    numbers = {int(turn["turn"]) for turn in turns}
    if len(numbers) != 1:
        raise RuntimeError(f"turn numbers differ: {sorted(numbers)}")
    now = numbers.pop()
    queries = [turn["query"].encode("utf-8") for turn in turns]
    records = [graph._records(released, now) for released in released_per_world]
    allowed = graph._allowed(False).unsqueeze(0).expand(batch, -1, -1)
    states, view, goal = batched_observe(
        core, graph, states, queries, queries, records, now, allowed, const, parent, gates
    )
    shift = shift_index(view.keys, reference)
    previous = torch.full((batch,), EOS, dtype=torch.long)
    logits = []
    emitted = []
    for _ in range(ANSWER_BYTES):
        output = step_batch(
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
        logits.append(output.probabilities)
        choice = torch.argmax(output.probabilities[:, :256], dim=1)
        emitted.append(choice)
        states = consume_batch(output.state, output, choice, shift)
        previous = choice
    return states, torch.stack(logits, 1), torch.stack(emitted, 1)


def batched_release(
    core, graph, states, releases, released_per_world, now, const, parent, gates, table, write
):
    batch = len(released_per_world)
    records = [graph._records(released, now) for released in released_per_world]
    written = torch.zeros(batch, dtype=torch.long)
    for item_index in range(len(releases[0]["items"])):
        feedback = [release["items"][item_index]["feedback"] for release in releases]
        goals = [release["items"][item_index]["query"] for release in releases]
        raws = [text.encode("utf-8") for text in feedback]
        allowed = graph._allowed(write).unsqueeze(0).expand(batch, -1, -1)
        before = states.fast.detach().clone()
        states, _, _ = batched_observe(
            core,
            graph,
            states,
            raws,
            [text.encode("utf-8") for text in goals],
            records,
            now,
            allowed,
            const,
            parent,
            gates,
        )
        changed = ~(states.fast == before).flatten(1).all(1)
        written += changed.to(torch.long)
    return states, written


def run_pair(core, graph, worlds, const, parent, gates, table, train: bool):
    """One 51-turn pass over ``BATCH`` worlds; one loss, one backward outside."""
    responses = [world["response"] for world in worlds]
    states = initial_batched_state(core, tuple(response["world_id"] for response in responses))
    released = [[] for _ in responses]
    losses = []
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
    }
    for index in range(TOTAL_TURNS):
        turns = [response["turns"][index] for response in responses]
        states, logits, emitted = batched_turn(
            core, graph, states, turns, released, const, parent, gates, table
        )
        for world_index, world in enumerate(worlds):
            number = int(turns[world_index]["turn"])
            gold = gold_bytes(world, number)
            row = [int(value) for value in emitted[world_index]]
            decision = all(row[p] == gold[p] for p in DECISION_POSITIONS)
            metrics["decision_correct"] += int(decision)
            metrics["surfaces"] += 1
            metrics["template_bits"] += sum(int(row[p] == gold[p]) for p in TEMPLATE_POSITIONS)
            metrics["template_total"] += len(TEMPLATE_POSITIONS)
            metrics["content_bits"] += sum(int(row[p] == gold[p]) for p in DECISION_POSITIONS)
            metrics["content_total"] += len(DECISION_POSITIONS)
            metrics["full_bytes"] += int(tuple(row) == tuple(gold))
            metrics["turns"] += 1
        if turns[0]["correction_surface"] and all(released):
            golds = [
                gold_bytes(world, int(turns[world_index]["turn"]))
                for world_index, world in enumerate(worlds)
            ]
            term = logits.new_zeros(())
            for position in DECISION_POSITIONS:
                index_tensor = torch.tensor([int(gold[position]) for gold in golds])
                picked = logits[:, position].gather(1, index_tensor.unsqueeze(1)).squeeze(1)
                term = term - torch.log(picked + 1e-12).sum()
            losses.append(term)
        now = int(turns[0]["turn"])
        release_rows = [
            next(
                (release for release in response["releases"] if int(release["after_turn"]) == now),
                None,
            )
            for response in responses
        ]
        if release_rows[0] is not None:
            states, count = batched_release(
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
    loss = torch.stack(losses).sum() if losses else core.W.new_zeros(())
    return loss, metrics


def merge(accum: dict, metrics: dict, loss: float, update_cpu: float) -> None:
    accum["updates"] += 1
    accum["cursor"] += BATCH
    accum["losses"].append(loss)
    accum["update_cpu"].append(update_cpu)
    for key in (
        "decision_correct",
        "surfaces",
        "template_bits",
        "template_total",
        "content_bits",
        "content_total",
        "full_bytes",
        "turns",
        "fast_writes",
    ):
        accum[key] += metrics[key]


def accuracy(accum: dict) -> dict:
    return {
        "decision_correct": accum["decision_correct"],
        "surfaces": accum["surfaces"],
        "decision_accuracy": accum["decision_correct"] / max(accum["surfaces"], 1),
        "template_bit_accuracy": accum["template_bits"] / max(accum["template_total"], 1),
        "content_bit_accuracy": accum["content_bits"] / max(accum["content_total"], 1),
        "full_byte_accuracy": accum["full_bytes"] / max(accum["turns"], 1),
        "fast_writes": accum["fast_writes"],
        "turns": accum["turns"],
    }


# ---------------------------------------------------------------------------
# F leverage probe: five counterfactual arms, zero optimizer steps
# ---------------------------------------------------------------------------


def clone_state(state: BatchedLiving) -> BatchedLiving:
    return BatchedLiving(
        subject_id=state.subject_id,
        revision=state.revision,
        step=state.step,
        h=state.h.clone(),
        fast=state.fast.clone(),
        nu_ema=state.nu_ema.clone(),
        keys=state.keys,
        coverage=state.coverage.clone(),
        prior=state.prior.clone(),
    )


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


def surface_loss(logits: torch.Tensor, gold: bytes) -> torch.Tensor:
    term = logits.new_zeros(())
    for position in DECISION_POSITIONS:
        term = term - torch.log(logits[position][gold[position]] + 1e-12)
    return term


def f_leverage(core, graph, world, const, parent, gates, table) -> dict:
    """Run one world, snapshot the 22 correction surfaces, perturb F five ways."""
    response = world["response"]
    states = initial_batched_state(core, (response["world_id"],))
    released = [[]]
    snaps = []
    rows_by_turn = {}
    for index, turn in enumerate(response["turns"]):
        number = int(turn["turn"])
        if turn["correction_surface"] and released[0]:
            snaps.append(
                {
                    "index": index,
                    "turn": turn,
                    "number": number,
                    "state": clone_state(states),
                    "released": list(released[0]),
                }
            )
        states, logits, emitted = batched_turn(
            core, graph, states, [turn], released, const, parent, gates, table
        )
        gold = gold_bytes(world, number)
        rows_by_turn[index] = {
            "bytes": [int(value) for value in emitted[0]],
            "loss": float(surface_loss(logits[0], gold)),
        }
        now = number
        release = next(
            (item for item in response["releases"] if int(item["after_turn"]) == now), None
        )
        if release is not None:
            states, _ = batched_release(
                core,
                graph,
                states,
                [release],
                released,
                now,
                const,
                parent,
                gates,
                table,
                bool(graph.fast_enabled),
            )
            released[0].extend(
                {
                    "source_id": item["source_id"],
                    "feedback": item["feedback"],
                    "version": item["version"],
                    "after_turn": now,
                }
                for item in release["items"]
            )
    baseline = [rows_by_turn[snap["index"]] for snap in snaps]
    base_loss = sum(row["loss"] for row in baseline) / max(len(baseline), 1)
    rng = torch.Generator().manual_seed(PERMUTE_SEED)
    arms = {}
    for kind in LADDER_ARMS:
        rows = []
        for snap in snaps:
            state = snap["state"]
            counterfactual = BatchedLiving(
                subject_id=state.subject_id,
                revision=state.revision,
                step=state.step,
                h=state.h,
                fast=variant_fast(kind, state.fast, rng),
                nu_ema=state.nu_ema,
                keys=state.keys,
                coverage=state.coverage,
                prior=state.prior,
            )
            _, logits, emitted = batched_turn(
                core,
                graph,
                counterfactual,
                [snap["turn"]],
                [snap["released"]],
                const,
                parent,
                gates,
                table,
            )
            rows.append(
                {
                    "bytes": [int(value) for value in emitted[0]],
                    "loss": float(surface_loss(logits[0], gold_bytes(world, snap["number"]))),
                }
            )
        arm_loss = sum(row["loss"] for row in rows) / max(len(rows), 1)
        diff = sum(
            sum(1 for a, b in zip(row["bytes"], base["bytes"], strict=True) if a != b)
            for row, base in zip(rows, baseline, strict=True)
        )
        arms[kind] = {
            "loss": arm_loss,
            "delta_loss": arm_loss - base_loss,
            "leverage": abs(arm_loss - base_loss) / max(abs(base_loss), 1e-12),
            "byte_diff": int(diff),
        }
    return {
        "correction_turns": len(snaps),
        "baseline_loss": base_loss,
        "arms": arms,
        "leverage_max": max(arm["leverage"] for arm in arms.values()),
        "byte_diff_max": max(arm["byte_diff"] for arm in arms.values()),
    }


# ---------------------------------------------------------------------------
# checkpoints
# ---------------------------------------------------------------------------


def checkpoint_path(arm: str, seed: int, step: int) -> Path:
    return CHECKPOINTS / f"{arm}_{seed}_{step:03d}.pt"


def save_checkpoint(path: Path, graph, optimizer, arm, seed, accum) -> str:
    payload = {
        "schema": "taiji-rule-learning-20-4c-r2-checkpoint-v1",
        "arm": arm,
        "seed": int(seed),
        "updates": int(accum["updates"]),
        "cursor": int(accum["cursor"]),
        "subject": {
            name: value.detach().cpu().clone() for name, value in graph.subject.state_dict().items()
        },
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "rng": torch.get_rng_state().cpu().clone(),
        "accum": copy.deepcopy(accum),
        "protocol": {
            "arms": list(ARMS),
            "seeds": list(SEEDS),
            "updates_per_run": UPDATES_PER_RUN,
            "batch": BATCH,
            "learning_rate": LEARNING_RATE,
            "grad_clip": GRAD_CLIP,
            "addressed_store_enabled": ADDRESSED_STORE,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return file_sha(path)


def load_checkpoint(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "taiji-rule-learning-20-4c-r2-checkpoint-v1":
        raise ValueError("unexpected checkpoint schema")
    return payload


def latest_checkpoint(arm: str, seed: int) -> tuple[Path, int] | None:
    """Newest 25-step checkpoint of this run, or None when the run has none."""
    found = None
    for step in range(CHECKPOINT_EVERY, UPDATES_PER_RUN + 1, CHECKPOINT_EVERY):
        candidate = checkpoint_path(arm, seed, step)
        if candidate.exists():
            found = (candidate, step)
    return found


def replay_after_checkpoint(candidate, gate, arm: str, seed: int, payload: dict) -> dict:
    """One update rebuilt from a checkpoint, to prove a resume picks up exactly."""
    graph = build_graph(candidate, gate, arm)
    unfreeze_outer(graph.subject)
    graph.subject.load_state_dict(payload["subject"])
    optimizer = torch.optim.Adam(trainer_params(graph), lr=LEARNING_RATE)
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["rng"])
    cursor = int(payload["cursor"])
    worlds = [
        world_at(seed, TRAIN_INDICES[(cursor + offset) % len(TRAIN_INDICES)])
        for offset in range(BATCH)
    ]
    const = update_constants(graph.subject)
    parent, gates = growth_rows(
        graph.subject, graph.step_kwargs.get("growth_gates"), graph.subject.W
    )
    table = patch_table(graph.subject, graph.subject.W)
    graph.subject.zero_grad(set_to_none=True)
    loss_value, metrics = run_pair(
        graph.subject, graph, worlds, const, parent, gates, table, train=True
    )
    loss_value.backward()
    torch.nn.utils.clip_grad_norm_(trainer_params(graph), GRAD_CLIP)
    optimizer.step()
    result = {
        "step": int(payload["updates"]) + 1,
        "loss": float(loss_value.detach()),
        "weights": weights_digest(graph.subject),
        "optimizer": optimizer_digest(optimizer),
        "rng": rng_digest(),
    }
    del const, parent, gates, table, loss_value, metrics, graph, optimizer
    gc.collect()
    return result


# ---------------------------------------------------------------------------
# one run
# ---------------------------------------------------------------------------


def run_one(arm: str, seed: int, candidate, gate, log: list[dict]) -> dict:
    graph = build_graph(candidate, gate, arm)
    unfreeze_outer(graph.subject)
    optimizer = torch.optim.Adam(trainer_params(graph), lr=LEARNING_RATE)
    found = latest_checkpoint(arm, seed)
    accum = accumulator()
    resumed = False
    last_probe = None
    if found is not None:
        path, loaded_step = found
        payload = load_checkpoint(path)
        if payload["arm"] != arm or int(payload["seed"]) != seed:
            raise ValueError("checkpoint identity mismatch")
        if int(payload["updates"]) != loaded_step:
            raise ValueError("checkpoint step does not match its file name")
        graph.subject.load_state_dict(payload["subject"])
        optimizer.load_state_dict(payload["optimizer"])
        torch.set_rng_state(payload["rng"])
        accum = copy.deepcopy(payload["accum"])
        resumed = accum["updates"] > 0
        append_progress(
            {
                "stage": "resume_loaded",
                "arm": arm,
                "seed": seed,
                "from_update": accum["updates"],
                "checkpoint_sha256": file_sha(path),
                "cpu": round(total_cpu(), 3),
            }
        )
        del payload
        gc.collect()
        prior_probes = [
            row
            for row in log
            if row.get("stage") == "f_probe"
            and row.get("arm") == arm
            and row.get("seed") == seed
            and int(row.get("step", 0)) <= accum["updates"]
        ]
        if prior_probes:
            last_probe = prior_probes[-1]
    append_progress(
        {
            "stage": "run_start",
            "arm": arm,
            "seed": seed,
            "resumed": resumed,
            "from_update": accum["updates"],
            "host_free_mb": round(free_mb(), 1),
            "cpu": round(cost()[0], 3),
        }
    )
    if accum["updates"] == 0:
        # step-0 baseline on the first training batch, before any optimizer step
        baselines = [world_at(seed, TRAIN_INDICES[0]), world_at(seed, TRAIN_INDICES[1])]
        const = update_constants(graph.subject)
        parent, gates = growth_rows(
            graph.subject, graph.step_kwargs.get("growth_gates"), graph.subject.W
        )
        table = patch_table(graph.subject, graph.subject.W)
        with torch.no_grad():
            _, metrics = run_pair(
                graph.subject, graph, baselines, const, parent, gates, table, train=False
            )
        base_accum = accumulator()
        base_accum["surfaces"] = metrics["surfaces"]
        base_accum["turns"] = metrics["turns"]
        for key in (
            "decision_correct",
            "template_bits",
            "template_total",
            "content_bits",
            "content_total",
            "full_bytes",
            "fast_writes",
        ):
            base_accum[key] = metrics[key]
        append_progress(
            {
                "stage": "baseline",
                "arm": arm,
                "seed": seed,
                "worlds": [world["world_id"] for world in baselines],
                **accuracy(base_accum),
                "cpu": round(cost()[0], 3),
            }
        )
        del base_accum
        gc.collect()

    while accum["updates"] < UPDATES_PER_RUN:
        guard(f"{arm}/{seed}/update")
        step = accum["updates"] + 1
        cursor = accum["cursor"]
        worlds = [
            world_at(seed, TRAIN_INDICES[(cursor + offset) % len(TRAIN_INDICES)])
            for offset in range(BATCH)
        ]
        const = update_constants(graph.subject)
        parent, gates = growth_rows(
            graph.subject, graph.step_kwargs.get("growth_gates"), graph.subject.W
        )
        table = patch_table(graph.subject, graph.subject.W)
        started = time.process_time()
        graph.subject.zero_grad(set_to_none=True)
        loss_value, metrics = run_pair(
            graph.subject, graph, worlds, const, parent, gates, table, train=True
        )
        loss_value.backward()
        torch.nn.utils.clip_grad_norm_(trainer_params(graph), GRAD_CLIP)
        optimizer.step()
        update_cpu = time.process_time() - started
        value = float(loss_value.detach())
        if not (value == value) or value in (float("inf"), float("-inf")):
            raise BudgetStop("numerical_instability", f"non-finite loss at {arm}/{seed}/{step}")
        merge(accum, metrics, value, update_cpu)
        del const, parent, gates, table, loss_value, metrics
        if step == CALIBRATION_UPDATES and not resumed:
            mean_cpu = sum(accum["update_cpu"][:CALIBRATION_UPDATES]) / CALIBRATION_UPDATES
            ratio = mean_cpu / BASELINE_UNIT
            append_progress(
                {
                    "stage": "calibration",
                    "arm": arm,
                    "seed": seed,
                    "updates": CALIBRATION_UPDATES,
                    "cpu_mean": round(mean_cpu, 3),
                    "baseline_unit": BASELINE_UNIT,
                    "ratio": round(ratio, 4),
                    "ok": ratio <= CALIBRATION_RATIO,
                    **memory_sample(),
                    "cpu": round(cost()[0], 3),
                    "total_cpu": round(total_cpu(), 3),
                }
            )
            if ratio > CALIBRATION_RATIO:
                raise BudgetStop(
                    "cpu_inflation",
                    (
                        f"calibration cpu_mean {mean_cpu:.1f} s = {ratio:.2f}x "
                        f"baseline at {arm}/{seed}/update {CALIBRATION_UPDATES}"
                    ),
                )
        if step == CHECKPOINT_EVERY + 1 and not resumed:
            append_progress(
                {
                    "stage": "resume_continuous",
                    "arm": arm,
                    "seed": seed,
                    "step": step,
                    "loss": value,
                    "weights": weights_digest(graph.subject),
                    "optimizer": optimizer_digest(optimizer),
                    "rng": rng_digest(),
                    "cpu": round(cost()[0], 3),
                }
            )
        if step % READ_EVERY == 0:
            block = accum["losses"][-READ_EVERY:]
            recent = accum["update_cpu"][-READ_EVERY:]
            mean_cpu = sum(recent) / len(recent)
            ratio = mean_cpu / BASELINE_UNIT
            status = "ok"
            if ratio > INFLATION_STOP_RATIO:
                status = "stop"
            elif ratio > INFLATION_WARN_RATIO:
                status = "warning"
            append_progress(
                {
                    "stage": "reading",
                    "arm": arm,
                    "seed": seed,
                    "step": step,
                    "losses": [round(item, 6) for item in block],
                    "loss_mean": round(sum(block) / len(block), 6),
                    "update_cpu_mean": round(mean_cpu, 3),
                    "cpu_ratio": round(ratio, 4),
                    "cpu_status": status,
                    "host_free_mb": round(free_mb(), 1),
                    **memory_sample(),
                    **accuracy(accum),
                    "cpu": round(cost()[0], 3),
                    "total_cpu": round(total_cpu(), 3),
                }
            )
            if len(accum["losses"]) >= READ_EVERY:
                mean_first = sum(accum["losses"][:READ_EVERY]) / READ_EVERY
                if sum(block) / len(block) > DIVERGENCE_FACTOR * mean_first:
                    raise BudgetStop(
                        "numerical_instability",
                        f"loss blew up beyond {DIVERGENCE_FACTOR}x at {arm}/{seed}/{step}",
                    )
            if status == "stop":
                save_checkpoint(
                    checkpoint_path(arm, seed, step), graph, optimizer, arm, seed, accum
                )
                raise BudgetStop(
                    "cpu_inflation",
                    (
                        f"update cost {mean_cpu:.1f} s = {ratio:.2f}x baseline "
                        f"at {arm}/{seed}/{step}"
                    ),
                )
        if step in PROBE_STEPS:
            probe_started = time.process_time()
            const = update_constants(graph.subject)
            parent, gates = growth_rows(
                graph.subject, graph.step_kwargs.get("growth_gates"), graph.subject.W
            )
            table = patch_table(graph.subject, graph.subject.W)
            with torch.no_grad():
                probe = f_leverage(
                    graph.subject,
                    graph,
                    world_at(seed, TRAIN_INDICES[PROBE_WORLD_INDEX]),
                    const,
                    parent,
                    gates,
                    table,
                )
            probe["probe_cpu"] = round(time.process_time() - probe_started, 3)
            append_progress(
                {
                    "stage": "f_probe",
                    "arm": arm,
                    "seed": seed,
                    "step": step,
                    **probe,
                    "cpu": round(cost()[0], 3),
                }
            )
            dead_now = probe["leverage_max"] < EARLY_STOP_GAIN and probe["byte_diff_max"] == 0
            if (
                arm == "CF1"
                and step in EARLY_STOP_STEPS
                and dead_now
                and last_probe is not None
                and last_probe["leverage_max"] < EARLY_STOP_GAIN
                and last_probe["byte_diff_max"] == 0
            ):
                raise BudgetStop(
                    "f_channel_dead_under_training",
                    f"two consecutive dead probes at {arm}/{seed}/{step}",
                )
            last_probe = probe
            del const, parent, gates, table
        if step % CHECKPOINT_EVERY == 0:
            sha = save_checkpoint(
                checkpoint_path(arm, seed, step), graph, optimizer, arm, seed, accum
            )
            append_progress(
                {
                    "stage": "checkpoint",
                    "arm": arm,
                    "seed": seed,
                    "step": step,
                    "sha256": sha,
                    "cpu": round(cost()[0], 3),
                    "total_cpu": round(total_cpu(), 3),
                }
            )
        if step % READ_EVERY == 0:
            gc.collect()
    append_progress(
        {
            "stage": "run_trained",
            "arm": arm,
            "seed": seed,
            "updates": accum["updates"],
            "cpu": round(cost()[0], 3),
        }
    )

    rows = load_progress()
    continuous = next(
        (
            row
            for row in rows
            if row.get("stage") == "resume_continuous"
            and row.get("arm") == arm
            and row.get("seed") == seed
        ),
        None,
    )
    verified = any(
        row.get("stage") == "resume_verified" and row.get("arm") == arm and row.get("seed") == seed
        for row in rows
    )
    verify_path = checkpoint_path(arm, seed, CHECKPOINT_EVERY)
    # One replay per arm is the evidence that a resume picks up exactly; running it
    # on every seed would spend CPU the cap cannot spare.
    if seed == VERIFY_SEED and continuous is not None and not verified and verify_path.exists():
        payload = load_checkpoint(verify_path)
        if int(payload["updates"]) == CHECKPOINT_EVERY:
            guard(f"{arm}/{seed}/resume_verify")
            replayed = replay_after_checkpoint(candidate, gate, arm, seed, payload)
            keys = ("loss", "weights", "optimizer", "rng")
            append_progress(
                {
                    "stage": "resume_verified",
                    "arm": arm,
                    "seed": seed,
                    "step": replayed["step"],
                    "checkpoint_step": CHECKPOINT_EVERY,
                    "ok": all(replayed[key] == continuous[key] for key in keys),
                    "replay": {key: replayed[key] for key in keys},
                    "continuous": {key: continuous[key] for key in keys},
                    "cpu": round(cost()[0], 3),
                }
            )
        del payload

    eval_started = time.process_time()
    eval_accum = accumulator()
    const = update_constants(graph.subject)
    parent, gates = growth_rows(
        graph.subject, graph.step_kwargs.get("growth_gates"), graph.subject.W
    )
    table = patch_table(graph.subject, graph.subject.W)
    with torch.no_grad():
        dev_worlds = [world_at(seed, index) for index in DEV_INDICES]
        for start in range(0, len(dev_worlds), BATCH):
            guard(f"{arm}/{seed}/dev_eval")
            _, metrics = run_pair(
                graph.subject,
                graph,
                dev_worlds[start : start + BATCH],
                const,
                parent,
                gates,
                table,
                train=False,
            )
            eval_accum["surfaces"] += metrics["surfaces"]
            eval_accum["turns"] += metrics["turns"]
            for key in (
                "decision_correct",
                "template_bits",
                "template_total",
                "content_bits",
                "content_total",
                "full_bytes",
                "fast_writes",
            ):
                eval_accum[key] += metrics[key]
    eval_cpu = time.process_time() - eval_started
    append_progress(
        {
            "stage": "dev_eval",
            "arm": arm,
            "seed": seed,
            "worlds": len(DEV_INDICES),
            "indices": [DEV_INDICES[0], DEV_INDICES[-1]],
            **accuracy(eval_accum),
            "eval_cpu": round(eval_cpu, 3),
            **memory_sample(),
            "weights": weights_digest(graph.subject),
            "cpu": round(cost()[0], 3),
            "total_cpu": round(total_cpu(), 3),
        }
    )
    append_progress(
        {
            "stage": "run_completed",
            "arm": arm,
            "seed": seed,
            "updates": accum["updates"],
            "weights": weights_digest(graph.subject),
            "optimizer": optimizer_digest(optimizer),
            "rng": rng_digest(),
            "cpu": round(cost()[0], 3),
            "total_cpu": round(total_cpu(), 3),
        }
    )
    del graph, optimizer, eval_accum
    gc.collect()
    return {"arm": arm, "seed": seed, "updates": accum["updates"], "resumed": resumed}


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def assemble(
    log: list[dict], freeze_record: dict, identity: dict, resume: dict, stop: dict
) -> dict:
    runs: dict[str, dict] = {}
    for arm in ARMS:
        for seed in SEEDS:
            key = f"{arm}|{seed}"
            rows = [row for row in log if row.get("arm") == arm and row.get("seed") == seed]
            runs[key] = {
                "arm": arm,
                "seed": seed,
                "baseline": next((row for row in rows if row["stage"] == "baseline"), None),
                "readings": [row for row in rows if row["stage"] == "reading"],
                "probes": [row for row in rows if row["stage"] == "f_probe"],
                "checkpoints": [row for row in rows if row["stage"] == "checkpoint"],
                "dev_eval": next((row for row in rows if row["stage"] == "dev_eval"), None),
                "completed": any(row["stage"] == "run_completed" for row in rows),
                "resumed": any(row["stage"] == "run_start" and row.get("resumed") for row in rows),
            }
    comparison = {}
    for arm in ARMS:
        final = [runs[f"{arm}|{seed}"]["dev_eval"] for seed in SEEDS]
        finals = [row for row in final if row is not None]
        comparison[arm] = {
            "dev_runs": len(finals),
            "decision_accuracy": round(
                sum(row["decision_accuracy"] for row in finals) / max(len(finals), 1), 6
            ),
            "template_bit_accuracy": round(
                sum(row["template_bit_accuracy"] for row in finals) / max(len(finals), 1), 6
            ),
            "content_bit_accuracy": round(
                sum(row["content_bit_accuracy"] for row in finals) / max(len(finals), 1), 6
            ),
            "full_byte_accuracy": round(
                sum(row["full_byte_accuracy"] for row in finals) / max(len(finals), 1), 6
            ),
            "decision_correct": sum(row["decision_correct"] for row in finals),
            "surfaces": sum(row["surfaces"] for row in finals),
            "fast_writes": sum(row["fast_writes"] for row in finals),
        }
    completed = all(run["completed"] for run in runs.values())
    dead = [key for key, run in runs.items() if run["arm"] == "CF1" and not run["completed"]]
    calibration = [row for row in log if row.get("stage") == "calibration"]
    inflation = [
        row for row in log if row.get("stage") == "reading" and row.get("cpu_status") != "ok"
    ]
    cpu_vs_memory = [
        {
            "arm": row.get("arm"),
            "seed": row.get("seed"),
            "step": row.get("step"),
            "update_cpu_mean": row.get("update_cpu_mean"),
            "cpu_ratio": row.get("cpu_ratio"),
            "cpu_status": row.get("cpu_status"),
            "host_available_mb": row.get("host_available_mb"),
            "host_commit_used_mb": row.get("host_commit_used_mb"),
            "proc_page_faults": row.get("proc_page_faults"),
        }
        for row in log
        if row.get("stage") == "reading"
    ]
    return {
        "runs": runs,
        "comparison": comparison,
        "completed": completed,
        "interrupted_runs": dead,
        "freeze": freeze_record["path"],
        "freeze_sha256": freeze_record["sha256"],
        "identity": identity,
        "resume": {
            "note": resume.get("note"),
            "rows": [row for row in log if row.get("stage") == "resume_verified"],
            "loaded": [row for row in log if row.get("stage") == "resume_loaded"],
            "resumed": any(row.get("resumed") for row in log if row.get("stage") == "run_start"),
        },
        "stop": stop,
        "calibration": calibration,
        "inflation": inflation,
        "cpu_vs_memory": cpu_vs_memory,
        "wasted_prior_run": {
            "evidence": "20_4c",
            "report": "reports/taiji_package20_4c_formal_comparison_20261105.json",
            "run": "CF1|20261105",
            "updates_completed": 25,
            "cpu_seconds": PRIOR_CPU_SECONDS,
            "note": (
                "aborted on the absolute memory floor; readings kept untouched as historical "
                "evidence and the cost is charged against the same 52,000 CPU cap"
            ),
        },
    }


def emit_report(payload: dict, verdict: str, detail: str) -> None:
    payload = {
        "schema": "taiji-rule-learning-20-4c-r2-formal-comparison-v1",
        "evidence": "20_4c_r2",
        "verdict": verdict,
        "detail": detail,
        **payload,
        "formal_training_steps": sum(
            run["readings"][-1]["step"] if run["readings"] else 0
            for run in payload["runs"].values()
        ),
        "confirm_read": False,
        "addressed_store_enabled": ADDRESSED_STORE,
        "memory_policy": {
            "absolute_floor_removed": True,
            "replacement": "CPU price versus the 56.906 s baseline unit",
            "calibration_updates": CALIBRATION_UPDATES,
            "calibration_ratio_max": CALIBRATION_RATIO,
            "inflation_warn_ratio": INFLATION_WARN_RATIO,
            "inflation_stop_ratio": INFLATION_STOP_RATIO,
            "hard_floor_mb": MEM_HARD_FLOOR_MB,
            "hard_floor_verdict": "out_of_memory",
            "note": (
                "available/commit/page-fault counters are recorded on every reading; the "
                "cpu_vs_memory table is the audit trail for this substitution"
            ),
        },
        "cost": {
            "cpu_seconds": round(cost()[0], 3),
            "prior_cpu_seconds": PRIOR_CPU_SECONDS,
            "total_cpu_seconds": round(total_cpu(), 3),
            "remaining_cpu_seconds": round(CPU_CAP - total_cpu(), 3),
            "wall_seconds": round(cost()[1], 3),
            "cpu_cap": CPU_CAP,
        },
        "script": {
            "runner_sha256": file_sha(Path(__file__)),
            "submitted_runner_sha256": None,
            "executor_sha256": file_sha(ROOT / "taiji/living_core_batch.py"),
            "frozen_module_sha256": {
                "taiji/rule_learning_20_1_net.py": file_sha(
                    ROOT / "taiji/rule_learning_20_1_net.py"
                ),
                "taiji/living_core.py": file_sha(ROOT / "taiji/living_core.py"),
            },
        },
    }
    write_json(REPORT, payload)
    write_json(
        MANIFEST,
        {
            "schema": "taiji-rule-learning-20-4c-r2-manifest-v1",
            "evidence": "20_4c_r2",
            "verdict": verdict,
            "report_path": str(REPORT.relative_to(ROOT)),
            "report_sha256": file_sha(REPORT),
            "runner_sha256": payload["script"]["runner_sha256"],
            "submitted_runner_sha256": None,
            "executor_sha256": payload["script"]["executor_sha256"],
            "freeze_path": payload["freeze"],
            "freeze_sha256": payload["freeze_sha256"],
            "candidate_sha_v2": payload["identity"].get("candidate_sha_v2_after"),
            "formal_training_steps": payload["formal_training_steps"],
            "completed_runs": sum(1 for run in payload["runs"].values() if run["completed"]),
            "prior_cpu_seconds": PRIOR_CPU_SECONDS,
            "total_cpu_seconds": round(total_cpu(), 3),
            "training_started": True,
        },
    )


def write_freeze(identity: dict) -> dict:
    protocol = {
        "schema": "taiji-rule-learning-20-4c-r2-freeze-v1",
        "status": "frozen",
        "owner_ruling": (
            "absolute free-memory floor is the wrong instrument; gate on CPU price against "
            "the 56.906 s unit instead, keep a 200 MB hard floor for real OOM"
        ),
        "data": {
            "train_worlds": [TRAIN_INDICES[0], TRAIN_INDICES[-1]],
            "dev_worlds": [DEV_INDICES[0], DEV_INDICES[-1]],
            "confirm_worlds_reserved": list(CONFIRM_INDICES),
            "confirm_read": False,
        },
        "training": {
            "arms": list(ARMS),
            "seeds": list(SEEDS),
            "updates_per_run": UPDATES_PER_RUN,
            "runs": len(ARMS) * len(SEEDS),
            "batch_worlds": BATCH,
            "learning_rate": LEARNING_RATE,
            "grad_clip": GRAD_CLIP,
            "addressed_store_enabled": ADDRESSED_STORE,
            "optimizer": "Adam",
            "reading_every_updates": READ_EVERY,
            "checkpoint_every_updates": CHECKPOINT_EVERY,
        },
        "measurement": {
            "threads": THREADS,
            "dtype": "float64",
            "deterministic": True,
            "executor": "taiji/living_core_batch.py",
            "memory_is_not_a_gate": True,
            "cpu_ratio_calibration": {
                "updates": CALIBRATION_UPDATES,
                "ratio_max": CALIBRATION_RATIO,
                "reference_unit": BASELINE_UNIT,
                "verdict": "cpu_inflation",
            },
            "cpu_ratio_per_block": {
                "block_updates": READ_EVERY,
                "warn_ratio": INFLATION_WARN_RATIO,
                "stop_ratio": INFLATION_STOP_RATIO,
            },
            "memory_hard_floor_mb": MEM_HARD_FLOOR_MB,
            "memory_telemetry": "available, commit, process rss, page faults per reading",
            "cpu_cap": CPU_CAP,
            "cpu_guard": CPU_GUARD,
        },
        "early_stop": {
            "steps": list(EARLY_STOP_STEPS),
            "arms": list(LADDER_ARMS),
            "optimizer_steps_in_probe": 0,
            "threshold": EARLY_STOP_GAIN,
            "criterion": (
                "two consecutive probes with leverage_max < 1e-4 and byte_diff == 0 " "on CF1"
            ),
            "verdict": "f_channel_dead_under_training",
            "applies_to": "CF1 only; B_EH has no F writes and is recorded as a control",
        },
        "error_gates": {
            "all_51_turns_kept": True,
            "correction_surfaces": "22 separate",
            "post_feedback_first_answer": "separate",
            "template_vs_content": "reported separately",
            "per_arm_judgement": True,
        },
        "pricing": {
            "unit_from_20_4b": REFERENCE_UNIT_FROM_20_4B,
            "total_from_20_4b": REFERENCE_TOTAL_FROM_20_4B,
            "prior_cpu_seconds": PRIOR_CPU_SECONDS,
            "prior_source": "reports/taiji_package20_4c_formal_comparison_20261105.json",
            "hard_cap": CPU_CAP,
            "remaining_at_freeze": round(CPU_CAP - PRIOR_CPU_SECONDS, 3),
        },
        "identity": identity,
        "read_only_evidence": [
            "reports/taiji_package20_4_checkpoint_roundtrip",
            "reports/taiji_package20_4b_graph_optimization_20261105.json",
            "reports/taiji_package20_4a_unit_cost_20261105.json",
            "reports/taiji_package20_4c_formal_comparison_20261105.json",
        ],
    }
    write_json(FREEZE, protocol)
    return {"path": str(FREEZE.relative_to(ROOT)), "sha256": file_sha(FREEZE)}


def run() -> int:
    if not PROGRESS.exists():
        PROGRESS.parent.mkdir(parents=True, exist_ok=True)
        PROGRESS.write_text("", encoding="utf-8")
    bundle = torch.load(BUNDLE, map_location="cpu", weights_only=False)
    candidate = load_routing_bundle(BUNDLE)[0]
    gate = dict(bundle["growth_gates"][0]) if bundle.get("growth_gates") else None
    probe_graph = build_graph(candidate, gate, "CF1")
    identity_before = graph_identity(probe_graph)
    del probe_graph
    gc.collect()
    freeze_record = write_freeze(identity_before)
    available = free_mb()
    log = load_progress()
    append_progress(
        {
            "stage": "protocol",
            "threads": THREADS,
            "dtype": "float64",
            "deterministic": True,
            "batch_worlds": BATCH,
            "arms": list(ARMS),
            "seeds": list(SEEDS),
            "updates_per_run": UPDATES_PER_RUN,
            "cpu_cap": CPU_CAP,
            "prior_cpu_seconds": PRIOR_CPU_SECONDS,
            "baseline_unit": BASELINE_UNIT,
            "calibration_ratio_max": CALIBRATION_RATIO,
            "inflation_stop_ratio": INFLATION_STOP_RATIO,
            "memory_hard_floor_mb": MEM_HARD_FLOOR_MB,
            "host_free_mb": round(available, 1),
            "host_total_mb": round(total_mb(), 1),
            **memory_sample(),
            "freeze_sha256": freeze_record["sha256"],
            "candidate_sha_v2": identity_before["candidate_sha_v2"],
            "rng_state": rng_digest(),
            "cpu": round(cost()[0], 3),
            "total_cpu": round(total_cpu(), 3),
        }
    )
    log = load_progress()
    resume = {
        "stage": "resume_verified",
        "note": "the first seed of each arm replays one update from its 25-step checkpoint",
        "rows": [row for row in log if row.get("stage") == "resume_verified"],
    }

    stop = {"verdict": "comparison_completed", "detail": "all four runs and dev evals done"}
    try:
        for seed in SEEDS:
            for arm in ARMS:
                done = any(
                    row.get("stage") == "run_completed"
                    and row.get("arm") == arm
                    and row.get("seed") == seed
                    for row in log
                )
                if done:
                    append_progress(
                        {
                            "stage": "run_reused",
                            "arm": arm,
                            "seed": seed,
                            "cpu": round(cost()[0], 3),
                        }
                    )
                    continue
                run_one(arm, seed, candidate, gate, log)
                log = load_progress()
    except BudgetStop as fired:
        stop = {"verdict": fired.verdict, "detail": fired.detail}
    identity_after = graph_identity(build_graph(candidate, gate, "CF1"))
    log = load_progress()
    payload = assemble(
        log, freeze_record, {"before": identity_before, "after": identity_after}, resume, stop
    )
    payload["identity"] = {
        "candidate_sha_v2_before": identity_before["candidate_sha_v2"],
        "candidate_sha_v2_after": identity_after["candidate_sha_v2"],
        "unchanged": identity_before["candidate_sha_v2"] == identity_after["candidate_sha_v2"],
    }
    payload["host_memory"] = {
        "start_mb": round(available, 1),
        "end_mb": round(free_mb(), 1),
        "end_sample": memory_sample(),
    }
    payload["recovery_rerun"] = any(run["resumed"] for run in payload["runs"].values()) or any(
        row.get("stage") == "run_reused" for row in log
    )
    emit_report(payload, stop["verdict"], stop["detail"])
    print(json.dumps({"verdict": stop["verdict"], "detail": stop["detail"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
