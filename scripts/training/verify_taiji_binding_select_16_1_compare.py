#!/usr/bin/env python
"""N16.1.1 formal comparison: two seeds x two arms x 300 updates.

Arm A (``A-locked``) locks the record-level alpha to the selection head and
floors gamma at 0.2.  Arm B (``B-select``) trains the same head but leaves the
decoder unlocked, as the pre-registered control for the selection--decode
interface.

Frozen dose: 2 seeds x 2 arms x 300 updates, batch 8, 288 train / 240 dev,
stages selection (1-100, CE only), copy (101-200, locked NLL), joint
(201-300, locked NLL + CE).  The self-prefix curriculum is triggered per batch
by the teacher-forced completion rate (threshold 0.5), never by update index.

Gamma-floor activation is reported separately (model gamma, used gamma,
floored-step and floored-update fractions, and a floor-removed sensitivity
pass) so the engineering floor cannot be mistaken for model capability.

Runs are serial per arm.  Pricing measurements are not rerun.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import statistics
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

torch.set_num_threads(1)

from taiji.binding_copy import (  # noqa: E402
    advance_copy,
    answer_step,
    forced_read,
    generate_bytes,
    initial_binding_state,
    observe_goal,
    prepare,
    score_example,
)
from taiji.binding_copy_scoring import score_generation  # noqa: E402
from taiji.binding_select import (  # noqa: E402
    GAMMA_FLOOR,
    SelectionConfig,
    build_selection_candidate,
    load_bundle,
    locked_generate,
    locked_loss,
    locked_step,
    make_selection_optimizer,
    save_bundle,
    select,
    selection_ce,
)
from taiji.living_core import EOS, RawSource  # noqa: E402
from taiji.record_relation import RecordRelationConfig  # noqa: E402
from taiji.record_relation_core import (  # noqa: E402
    ARM_FULL,
    CLIP_NORM,
    MAX_OUTPUT_BYTES,
    MAX_PREDICTIONS,
    RecordRelationCandidate,
    build_subject_from_candidate_bundle,
    candidate_parameter_map,
    decode_endpoint,
)

PARENT_BUNDLE = "output/taiji_core_record_relation_14_6/27031_A_full_step300.pt"
PARENT_SHA256 = "b75a83a0a474c041c2a61bed8c4e0a51de11197d67a9b196262e7e039a9a4cef"
SELECT_PAYLOAD = "output/taiji_core_binding_select_16_1/select_v1.json"
DATA_PAYLOAD = "output/taiji_core_binding_copy_15_1/data_v2.json"

REPORT = "reports/taiji_binding_select_16_1_compare_20261103.json"
MANIFEST = "plans/manifests/taiji_binding_select_16_1_compare_20261103.json"
OUTPUT_DIR = Path("output/taiji_core_binding_select_16_1_compare")
RESULTS_DIR = OUTPUT_DIR / "results"
BUNDLE_DIR = OUTPUT_DIR / "bundles"
LEDGER = OUTPUT_DIR / "ledger.jsonl"
PRECHECK = OUTPUT_DIR / "precheck.json"
PARTIAL = OUTPUT_DIR / "partial-{seed}-{arm}.json"

SEEDS = (28031, 28032)
ORDER_SEEDS = {28031: 38031, 28032: 38032}
ARMS = ("A-locked", "B-select")
ARM_LOCKED = {"A-locked": True, "B-select": False}

BATCH = 8
UPDATES = 300
CHECKPOINT_UPDATE = 150
STAGES = (("selection", 1, 100), ("copy", 101, 200), ("joint", 201, 300))
CURRICULUM_THRESHOLD = 0.5

ADAM_CAP = 1200
PRODUCT_CAP = 2 * 1024**3
WATCH_CPU_SECONDS = 20000.0
WATCH_WALL_SECONDS = 30000.0
WORKER_TIMEOUT = 600

GATE_SELECTION = 0.90
GATE_UNSEEN = 0.80
GATE_NEW_DESCRIPTION = 0.80
GATE_OVERALL = 0.80
GATE_REVERSED = 0.95
GATE_RESCUE = 8
GATE_CONTENT_DROP = 0.10
GATE_GAMMA = 0.20
GATE_INCREMENT = 0.10

BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 58031
SENSITIVITY_EXAMPLES = 40

SOURCE_FILES = (
    "taiji/binding_select.py",
    "taiji/binding_copy.py",
    "taiji/binding_copy_scoring.py",
    "taiji/record_relation.py",
    "taiji/record_relation_core.py",
    "scripts/training/verify_taiji_binding_select_16_1_compare.py",
)

STARTED_CPU = time.process_time()
STARTED_WALL = time.perf_counter()


class CompareStop(RuntimeError):
    """Named stop for the comparison stage."""


class WatchLine(RuntimeError):
    """Observation line reached: report progress instead of running through."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def ledger_totals() -> dict:
    """Sum per-process costs.  Rows store the cumulative stage cost, so the
    authoritative per-process number is ``cpu_seconds - prior_cpu_seconds``."""
    totals = {"cpu_seconds": 0.0, "wall_seconds": 0.0, "adam_calls": 0}
    if not LEDGER.exists():
        return totals
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        totals["cpu_seconds"] += float(row.get("cpu_seconds", 0.0)) - float(
            row.get("prior_cpu_seconds", 0.0)
        )
        totals["wall_seconds"] += float(row.get("wall_seconds", 0.0)) - float(
            row.get("prior_wall_seconds", 0.0)
        )
        totals["adam_calls"] += int(row.get("adam_calls", 0))
    return totals


def append_ledger(row: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class Budget:
    """Cumulative observation line across the serial per-arm processes."""

    def __init__(self) -> None:
        prior = ledger_totals()
        self.prior_cpu = prior["cpu_seconds"]
        self.prior_wall = prior["wall_seconds"]
        self.prior_adam = prior["adam_calls"]
        self.adam = 0
        self.live = 0
        self.max_live = 0
        self.cumulative = 0

    def cpu(self) -> float:
        return self.prior_cpu + (time.process_time() - STARTED_CPU)

    def wall(self) -> float:
        return self.prior_wall + (time.perf_counter() - STARTED_WALL)

    def add_product(self, info: dict) -> None:
        size = int(info.get("bytes", 0))
        self.live += size
        self.cumulative += size
        self.max_live = max(self.max_live, self.live)

    def ensure(self, *, context: dict | None = None) -> None:
        cpu = self.cpu()
        wall = self.wall()
        if self.prior_adam + self.adam > ADAM_CAP:
            raise CompareStop(f"adam cap exceeded: {self.prior_adam + self.adam}")
        if self.max_live > PRODUCT_CAP:
            raise CompareStop(f"product cap exceeded: {self.max_live}")
        if cpu >= WATCH_CPU_SECONDS or wall >= WATCH_WALL_SECONDS:
            raise WatchLine(
                json.dumps(
                    {
                        "reason": "observation_line",
                        "cpu_seconds": cpu,
                        "wall_seconds": wall,
                        "adam_calls": self.prior_adam + self.adam,
                        "context": context or {},
                    },
                    ensure_ascii=False,
                )
            )

    def row(self, *, status: str, seed: int | None, arm: str | None) -> dict:
        return {
            "seed": seed,
            "arm": arm,
            "status": status,
            "cpu_seconds": self.cpu(),
            "wall_seconds": self.wall(),
            "adam_calls": self.adam,
            "prior_cpu_seconds": self.prior_cpu,
            "prior_wall_seconds": self.prior_wall,
            "prior_adam_calls": self.prior_adam,
        }


def stage_of(update: int) -> str:
    for name, low, high in STAGES:
        if low <= update <= high:
            return name
    raise CompareStop(f"update {update} outside the frozen stages")


def load_data() -> tuple[tuple[dict, ...], tuple[dict, ...], dict, dict]:
    select_payload = json.loads((ROOT / SELECT_PAYLOAD).read_text(encoding="utf-8"))
    data = json.loads((ROOT / DATA_PAYLOAD).read_text(encoding="utf-8"))
    # v2 indices are local to (split, world), so the key carries all three fields.
    by_key = {
        (example["split"], int(example["world"]), int(example["index"])): example
        for example in data["examples"]
    }

    def to_fixture(row: dict) -> dict:
        example = by_key[(row["split"], int(row["world"]), int(row["index"]))]
        records = tuple(
            RawSource(source_id=record["source_id"], raw=record["raw"].encode("utf-8"))
            for record in example["records"]
        )
        keys = list(row["record_keys"])
        if len(keys) != len(records):
            raise CompareStop(f"key/record length drift at index {row['index']}")
        for index, record in enumerate(example["records"]):
            if keys[index] != record["description"]:
                raise CompareStop(f"key fragment drift at index {row['index']}#{index}")
        object_index = next(
            (
                index
                for index, record in enumerate(example["records"])
                if record["kind"] == "object"
                and record["description"] == example["object_description"]
            ),
            None,
        )
        if object_index != int(row["selection_label"]):
            raise CompareStop(f"selection label drift at index {row['index']}")
        return {
            "index": int(row["index"]),
            "world": int(row["world"]),
            "goal": row["goal"],
            "records": records,
            "record_keys": keys,
            "selection_label": int(row["selection_label"]),
            "target_prefix": tuple(example["action"].encode("utf-8")),
            "target_bytes": tuple(example["action"].encode("utf-8")) + (EOS,),
            "flags": dict(row["flags"]),
            "object_description": example["object_description"],
            "place_description": example["place_description"],
            "object_name": example["object_name"],
            "place_name": example["place_name"],
            "raw_example": example,
        }

    train = tuple(to_fixture(row) for row in select_payload["train"])
    dev = tuple(to_fixture(row) for row in select_payload["dev"])
    return train, dev, select_payload, data


def build_candidate(payload: dict, profile: str, seed: int):
    config = RecordRelationConfig(**payload["relation_config"])
    subject = build_subject_from_candidate_bundle(payload)
    base = RecordRelationCandidate(subject, config, ARM_FULL, list(payload["growth_gates"]))
    named = dict(base.named_parameters())
    for name, value in payload["parameters"].items():
        if name.startswith("relation."):
            named[name].data.copy_(value)
    base.to(torch.float64)
    base.configure_trainable()
    torch.manual_seed(seed)
    candidate = build_selection_candidate(
        base, profile=profile, dtype=torch.float64, selection_config=SelectionConfig()
    )
    candidate.configure_trainable()
    return candidate, make_selection_optimizer(candidate)


def schedule(seed: int, train: tuple[dict, ...], updates: int) -> list[tuple[dict, ...]]:
    """Deterministic schedule with a fresh permutation per epoch."""
    batches: list[tuple[dict, ...]] = []
    epoch = 0
    generator = torch.Generator().manual_seed(ORDER_SEEDS[seed])
    permutation = torch.randperm(len(train) - len(train) % BATCH, generator=generator).tolist()
    cursor = 0
    while len(batches) < updates:
        if cursor + BATCH > len(permutation):
            epoch += 1
            generator = torch.Generator().manual_seed(ORDER_SEEDS[seed] + epoch)
            permutation = torch.randperm(
                len(train) - len(train) % BATCH, generator=generator
            ).tolist()
            cursor = 0
        batches.append(tuple(train[index] for index in permutation[cursor : cursor + BATCH]))
        cursor += BATCH
    return batches


def generate_unlocked_traced(
    candidate,
    goal: bytes,
    records: tuple,
    *,
    subject_id: str,
    max_predictions: int = MAX_PREDICTIONS,
    trace: list | None = None,
) -> dict:
    """Mirror of binding_copy.generate_bytes with a read-only gamma trace."""
    base = candidate.binding
    state = initial_binding_state(base, subject_id)
    prepared = prepare(base, state, goal, records)
    state, gate_index = observe_goal(base, prepared)
    previous = EOS
    raw: list[int] = []
    terminated = False
    gamma_values: list[float] = []
    for _prediction in range(max_predictions):
        step = answer_step(base, prepared, state, previous, gate_index=gate_index)
        if trace is not None:
            gamma = float(step.gamma.detach())
            trace.append({"gamma_model": gamma, "gamma_used": gamma, "floor": 0.0})
        endpoint = decode_endpoint(step.probabilities)
        if endpoint["eos"]:
            terminated = True
            break
        raw.append(int(endpoint["byte"]))
        if len(raw) > MAX_OUTPUT_BYTES:
            break
        m_next, coverage_next = advance_copy(
            base.reader,
            state,
            step.read,
            step.probabilities,
            step.gamma,
            endpoint["byte"],
            prepared.base.view,
        )
        state = replace(
            step.state,
            m=m_next,
            coverage=coverage_next,
            prefix_cursor=step.state.prefix_cursor + 1,
        )
        gamma_values.append(float(step.gamma))
        gate_index = step.gate_index
        previous = int(endpoint["byte"])
    return {
        "raw_bytes": bytes(raw),
        "terminated": terminated,
        "selection": None,
        "gamma_mean": (sum(gamma_values) / len(gamma_values)) if gamma_values else 0.0,
        "gamma_floor": None,
    }


def generate(
    candidate,
    fixture: dict,
    *,
    arm: str,
    records=None,
    keys=None,
    floor: float = GAMMA_FLOOR,
    trace: list | None = None,
    chosen_record: int | None = None,
    subject_id: str,
) -> dict:
    goal = fixture["goal"].encode("utf-8")
    recs = fixture["records"] if records is None else records
    fragments = fixture["record_keys"] if keys is None else keys
    with torch.no_grad():
        if ARM_LOCKED[arm]:
            return locked_generate(
                candidate,
                goal,
                recs,
                fragments,
                subject_id=subject_id,
                gamma_floor=floor,
                max_predictions=MAX_PREDICTIONS,
                trace=trace,
                chosen_record=chosen_record,
            )
        return generate_unlocked_traced(
            candidate,
            goal,
            recs,
            subject_id=subject_id,
            max_predictions=MAX_PREDICTIONS,
            trace=trace,
        )


def locked_generate_offset0(
    candidate,
    fixture: dict,
    *,
    record_index: int,
    subject_id: str,
    floor: float = GAMMA_FLOOR,
) -> dict:
    with torch.no_grad():
        return _locked_generate_offset0(
            candidate, fixture, record_index=record_index, subject_id=subject_id, floor=floor
        )


def _locked_generate_offset0(
    candidate,
    fixture: dict,
    *,
    record_index: int,
    subject_id: str,
    floor: float = GAMMA_FLOOR,
) -> dict:
    """Correct record alpha, beta pinned at record offset 0 on every step."""
    base = candidate.binding
    goal = fixture["goal"].encode("utf-8")
    state = initial_binding_state(base, subject_id)
    prepared = prepare(base, state, goal, fixture["records"])
    state, gate_index = observe_goal(base, prepared)
    previous = EOS
    raw: list[int] = []
    terminated = False
    for _prediction in range(MAX_PREDICTIONS):
        probe = answer_step(base, prepared, state, previous, gate_index=gate_index)
        gamma = torch.clamp(probe.gamma, min=floor)
        step = forced_read(
            base,
            prepared,
            state,
            previous,
            record_index=record_index,
            offset=0,
            gate_index=gate_index,
            gamma_override=gamma,
        )
        endpoint = decode_endpoint(step.probabilities)
        if endpoint["eos"]:
            terminated = True
            break
        raw.append(int(endpoint["byte"]))
        m_next, coverage_next = advance_copy(
            base.reader,
            state,
            step.read,
            step.probabilities,
            step.gamma,
            endpoint["byte"],
            prepared.base.view,
        )
        state = replace(
            step.state,
            m=m_next,
            coverage=coverage_next,
            prefix_cursor=step.state.prefix_cursor + 1,
        )
        gate_index = step.gate_index
        previous = int(endpoint["byte"])
    return {"raw_bytes": bytes(raw), "terminated": terminated}


def teacher_forced(candidate, fixture: dict, *, arm: str, subject_id: str) -> dict:
    with torch.no_grad():
        return _teacher_forced(candidate, fixture, arm=arm, subject_id=subject_id)


def _teacher_forced(candidate, fixture: dict, *, arm: str, subject_id: str) -> dict:
    """Teacher-forced argmax completion over the target prefix plus EOS."""
    if not ARM_LOCKED[arm]:
        result = score_example(
            candidate.binding,
            fixture["goal"].encode("utf-8"),
            fixture["records"],
            fixture["target_prefix"],
            subject_id=subject_id,
        )
        labels = fixture["target_prefix"] + (EOS,)
        hits = [
            bool(int(torch.argmax(row).item()) == int(label))
            for row, label in zip(result.distributions, labels, strict=True)
        ]
        return {"hits": hits, "completed": all(hits), "nll": float(result.nll)}

    base = candidate.binding
    goal = fixture["goal"].encode("utf-8")
    choice = select(candidate, goal, fixture["record_keys"])
    state = initial_binding_state(base, subject_id)
    prepared = prepare(base, state, goal, fixture["records"])
    state, gate_index = observe_goal(base, prepared)
    previous = EOS
    labels = fixture["target_prefix"] + (EOS,)
    hits = []
    nll = 0.0
    for position, label in enumerate(labels):
        step = locked_step(
            candidate,
            prepared,
            state,
            previous,
            gate_index=gate_index,
            chosen_record=int(choice["chosen"]),
        )
        probability = float(step.probabilities[label].clamp_min(1e-12))
        nll -= math.log(probability)
        hits.append(bool(int(torch.argmax(step.probabilities).item()) == int(label)))
        if position == len(labels) - 1:
            break
        m_next, coverage_next = advance_copy(
            base.reader,
            state,
            step.read,
            step.probabilities,
            step.gamma,
            label,
            prepared.base.view,
        )
        state = replace(
            step.state,
            m=m_next,
            coverage=coverage_next,
            prefix_cursor=step.state.prefix_cursor + 1,
        )
        gate_index = step.gate_index
        previous = int(label)
    return {"hits": hits, "completed": all(hits), "nll": nll / len(labels)}


def gamma_row(trace: list, emitted: int | None = None) -> dict:
    rows = trace if emitted is None else trace[:emitted]
    if not rows:
        return {
            "steps": 0,
            "gamma_model_mean": 0.0,
            "gamma_used_mean": 0.0,
            "floored_steps": 0,
            "floored_fraction": 0.0,
        }
    floored = sum(1 for item in rows if item["gamma_model"] < item["floor"])
    return {
        "steps": len(rows),
        "gamma_model_mean": sum(item["gamma_model"] for item in rows) / len(rows),
        "gamma_used_mean": sum(item["gamma_used"] for item in rows) / len(rows),
        "floored_steps": floored,
        "floored_fraction": floored / len(rows),
    }


def batch_objective(
    candidate,
    fixtures: tuple[dict, ...],
    *,
    arm: str,
    stage: str,
    self_prefix: bool,
    subject_tag: str,
) -> tuple[torch.Tensor, dict]:
    if stage == "selection":
        terms = [
            selection_ce(
                candidate,
                fixture["goal"].encode("utf-8"),
                fixture["record_keys"],
                fixture["selection_label"],
            )
            for fixture in fixtures
        ]
        total = sum(terms) / len(terms)
        return total, {
            "steps": 0,
            "gamma_model_mean": 0.0,
            "gamma_used_mean": 0.0,
            "floored_steps": 0,
            "floored_fraction": 0.0,
        }

    trace: list = []
    nll_terms = []
    for fixture in fixtures:
        prefix = fixture["target_prefix"]
        if self_prefix:
            with torch.no_grad():
                generated = generate(
                    candidate,
                    fixture,
                    arm=arm,
                    subject_id=f"{subject_tag}-self-{fixture['index']}",
                )
            own = tuple(generated["raw_bytes"])[: len(prefix)]
            if len(own) < len(prefix):
                own = own + (EOS,) * (len(prefix) - len(own))
            # Frozen 16.1.1 behavior: this self-prefix is passed to
            # locked_loss/score_example as target_prefix, so the labels become the
            # model's own bytes (self-imitation, not correction).  The N17 design
            # separates the input prefix from the fixed target bytes.
            prefix = own
        if ARM_LOCKED[arm]:
            nll_terms.append(
                locked_loss(
                    candidate,
                    fixture["goal"].encode("utf-8"),
                    fixture["records"],
                    fixture["record_keys"],
                    prefix,
                    subject_id=f"{subject_tag}-{fixture['index']}",
                    trace=trace,
                )
            )
        else:
            nll_terms.append(
                score_example(
                    candidate.binding,
                    fixture["goal"].encode("utf-8"),
                    fixture["records"],
                    prefix,
                    subject_id=f"{subject_tag}-{fixture['index']}",
                ).nll
            )
    nll = sum(nll_terms) / len(nll_terms)
    if stage == "copy":
        return nll, gamma_row(trace)
    terms = [
        selection_ce(
            candidate,
            fixture["goal"].encode("utf-8"),
            fixture["record_keys"],
            fixture["selection_label"],
        )
        for fixture in fixtures
    ]
    return nll + sum(terms) / len(terms), gamma_row(trace)


def summarize(rows) -> dict:
    def rate(pred) -> float:
        return sum(1 for row in rows if pred(row)) / len(rows) if rows else 0.0

    flag_keys = ("unseen_combination", "unseen_output_name", "new_description", "both_new")
    denominators = {key: sum(1 for row in rows if row["flags"][key]) for key in flag_keys}
    flags = {
        key: (
            sum(1 for row in rows if row["flags"][key] and row["score"]["completed"])
            / denominators[key]
            if denominators[key]
            else 0.0
        )
        for key in flag_keys
    }
    by_world: dict[str, dict] = {}
    for row in rows:
        entry = by_world.setdefault(str(row["world"]), {"n": 0, "completed": 0})
        entry["n"] += 1
        entry["completed"] += int(row["score"]["completed"])
    return {
        "completed": rate(lambda row: row["score"]["completed"]),
        "subsets": flags,
        "subset_denominators": denominators,
        "utf8": rate(lambda row: row["score"]["utf8"]),
        "eos": rate(lambda row: row["score"]["terminated"]),
        "full_sentence": rate(lambda row: row["score"]["full_sentence"]),
        "by_world": by_world,
        "examples": len(rows),
    }


def aggregate_gamma(per_example: list[dict]) -> dict:
    steps = sum(row["steps"] for row in per_example)
    floored = sum(row["floored_steps"] for row in per_example)
    model_mean = (
        sum(row["gamma_model_mean"] * row["steps"] for row in per_example) / steps if steps else 0.0
    )
    used_mean = (
        sum(row["gamma_used_mean"] * row["steps"] for row in per_example) / steps if steps else 0.0
    )
    any_floored = sum(1 for row in per_example if row["floored_steps"] > 0)
    return {
        "examples": len(per_example),
        "steps": steps,
        "gamma_model_mean": model_mean,
        "gamma_used_mean": used_mean,
        "copy_weight_mean": 1.0 - model_mean,
        "floored_steps": floored,
        "floored_step_fraction": floored / steps if steps else 0.0,
        "examples_with_any_floored": any_floored,
        "examples_with_any_floored_fraction": (
            any_floored / len(per_example) if per_example else 0.0
        ),
        "per_example": per_example,
    }


def selection_report(candidate, dev) -> dict:
    rows = []
    ce_total = 0.0
    for fixture in dev:
        goal = fixture["goal"].encode("utf-8")
        with torch.no_grad():
            outcome = select(
                candidate, goal, fixture["record_keys"], label=fixture["selection_label"]
            )
            ce_total += float(
                selection_ce(candidate, goal, fixture["record_keys"], fixture["selection_label"])
            )
        rows.append(
            {
                "index": fixture["index"],
                "world": fixture["world"],
                "flags": dict(fixture["flags"]),
                "chosen": int(outcome["chosen"]),
                "label": fixture["selection_label"],
                "correct": bool(outcome["correct"]),
                "confidence": float(outcome["confidence"]),
                "null_mass": float(outcome["null_mass"]),
            }
        )
    return {
        "top1": sum(1 for row in rows if row["correct"]) / len(rows) if rows else 0.0,
        "examples": len(rows),
        "ce_mean": ce_total / len(rows) if rows else 0.0,
        "rows": rows,
    }


def alpha_reading(candidate, dev) -> dict:
    base = candidate.binding
    rows = []
    for fixture in dev:
        state = initial_binding_state(base, f"alpha-{fixture['index']}")
        prepared = prepare(base, state, fixture["goal"].encode("utf-8"), fixture["records"])
        state, gate_index = observe_goal(base, prepared)
        with torch.no_grad():
            step = answer_step(base, prepared, state, EOS, gate_index=gate_index)
        alpha = step.read.alpha
        weights = step.read.copy_weights
        offsets = prepared.base.view.record_offsets
        alpha_record = int(torch.argmax(alpha[:-1]).item()) if alpha.numel() > 1 else -1
        chosen_byte = int(torch.argmax(weights).item()) if weights.numel() else -1
        byte_record = None
        for index, (start, end) in enumerate(offsets):
            if start <= chosen_byte < end:
                byte_record = index
                break
        rows.append(
            {
                "index": fixture["index"],
                "world": fixture["world"],
                "alpha_record": alpha_record,
                "byte_record": byte_record,
                "label": fixture["selection_label"],
                "alpha_correct": alpha_record == fixture["selection_label"],
                "byte_correct": byte_record == fixture["selection_label"],
            }
        )
    return {
        "alpha_top1": sum(1 for row in rows if row["alpha_correct"]) / len(rows),
        "byte_top1": sum(1 for row in rows if row["byte_correct"]) / len(rows),
        "examples": len(rows),
        "rows": rows,
    }


def record_template(style_name: str, description: str, name: str) -> str:
    if style_name == "stated":
        return f"{description}是{name}。"
    if style_name == "aka":
        return f"{name}又叫{description}。"
    if style_name == "mapped":
        return f"{description}对应{name}。"
    raise CompareStop(f"unknown record style {style_name}")


def object_records(example: dict) -> list[dict]:
    return [record for record in example["records"] if record["kind"] == "object"]


def alternate_object_index(fixture: dict) -> int | None:
    correct = fixture["selection_label"]
    names = {
        record["source_id"]: record["name"] for record in object_records(fixture["raw_example"])
    }
    target_name = names.get(fixture["raw_example"]["records"][correct]["source_id"])
    for index, record in enumerate(fixture["raw_example"]["records"]):
        if index == correct or record["kind"] != "object":
            continue
        if names.get(record["source_id"]) != target_name:
            return index
    return None


def swap_object_name(fixture: dict) -> dict | None:
    correct = fixture["selection_label"]
    objects = object_records(fixture["raw_example"])
    target = fixture["raw_example"]["records"][correct]
    alternate = next((record for record in objects if record["name"] != target["name"]), None)
    if alternate is None:
        return None
    new_name = alternate["name"]
    records = []
    for record in fixture["raw_example"]["records"]:
        name = new_name if record["source_id"] == target["source_id"] else record["name"]
        raw = record_template(record["style_name"], record["description"], name)
        records.append(RawSource(source_id=record["source_id"], raw=raw.encode("utf-8")))
    expected = f"把{new_name}放到{fixture['place_name']}"
    return {
        "records": tuple(records),
        "target_prefix": tuple(expected.encode("utf-8")),
    }


def evaluate(
    candidate, dev, *, arm: str, full: bool, budget: Budget, tag: str
) -> tuple[dict, dict]:
    rows = []
    true_gen: dict[tuple[int, int], dict] = {}
    per_example = []
    for fixture in dev:
        trace: list = []
        generated = generate(
            candidate,
            fixture,
            arm=arm,
            trace=trace,
            subject_id=f"{tag}-dev-{fixture['index']}",
        )
        scored = score_generation(
            generated["raw_bytes"], fixture["target_prefix"], generated["terminated"]
        )
        rows.append(
            {
                "index": fixture["index"],
                "world": fixture["world"],
                "flags": fixture["flags"],
                "score": scored,
            }
        )
        true_gen[(fixture["world"], fixture["index"])] = {
            "raw": generated["raw_bytes"],
            "terminated": generated["terminated"],
        }
        emitted = len(generated["raw_bytes"])
        stats = gamma_row(trace, emitted=min(emitted, len(trace)))
        per_example.append({"index": fixture["index"], **stats})
        budget.ensure(context={"phase": "dev_generate", "index": fixture["index"]})
    out: dict = {
        "overall": summarize(rows),
        "gamma": aggregate_gamma(per_example),
        "selection": selection_report(candidate, dev),
        "examples": len(dev),
    }
    if not full:
        out["rows"] = None
        return out, true_gen

    out["rows"] = rows
    out["reversed"] = evaluate_reversed(
        candidate, dev, arm=arm, true_gen=true_gen, budget=budget, tag=tag
    )
    out["counterfactual"] = evaluate_counterfactual(
        candidate, dev, arm=arm, rows=rows, budget=budget, tag=tag
    )
    out["alpha_reading"] = alpha_reading(candidate, dev)
    out["identity_rescue"] = evaluate_identity_rescue(candidate, dev, budget=budget, tag=tag)
    if ARM_LOCKED[arm]:
        out["content_interventions"] = evaluate_interventions(
            candidate, dev, arm=arm, true_gen=true_gen, budget=budget, tag=tag
        )
    else:
        out["content_interventions"] = {
            "applicable": False,
            "reason": "arm B has no selection lock; content interventions are not defined",
        }
    out["nll"] = evaluate_nll(candidate, dev, arm=arm, budget=budget, tag=tag)
    return out, true_gen


def evaluate_reversed(
    candidate, dev, *, arm: str, true_gen: dict, budget: Budget, tag: str
) -> dict:
    rows = []
    for fixture in dev:
        generated = generate(
            candidate,
            fixture,
            arm=arm,
            records=tuple(reversed(fixture["records"])),
            keys=list(reversed(fixture["record_keys"])),
            subject_id=f"{tag}-rev-{fixture['index']}",
        )
        scored = score_generation(
            generated["raw_bytes"], fixture["target_prefix"], generated["terminated"]
        )
        rows.append(
            {
                "index": fixture["index"],
                "world": fixture["world"],
                "score": scored,
                "paired_with_true": generated["raw_bytes"]
                == true_gen.get((fixture["world"], fixture["index"]), {}).get("raw", b""),
            }
        )
        budget.ensure(context={"phase": "reversed"})
    return {
        "accuracy": sum(1 for row in rows if row["score"]["completed"]) / len(rows),
        "paired_rate": sum(1 for row in rows if row["paired_with_true"]) / len(rows),
        "examples": len(rows),
        "rows": rows,
    }


def evaluate_counterfactual(candidate, dev, *, arm: str, rows, budget: Budget, tag: str) -> dict:
    true_by_key = {(row["world"], row["index"]): row["score"]["completed"] for row in rows}
    out = []
    for fixture in dev:
        swapped = swap_object_name(fixture)
        if swapped is None:
            continue
        generated = generate(
            candidate,
            fixture,
            arm=arm,
            records=swapped["records"],
            subject_id=f"{tag}-map-{fixture['index']}",
        )
        scored = score_generation(
            generated["raw_bytes"], swapped["target_prefix"], generated["terminated"]
        )
        out.append(
            {
                "index": fixture["index"],
                "world": fixture["world"],
                "counterfactual_completed": scored["completed"],
                "true_completed": true_by_key.get((fixture["world"], fixture["index"]), False),
            }
        )
        budget.ensure(context={"phase": "counterfactual"})
    both = sum(1 for row in out if row["counterfactual_completed"] and row["true_completed"])
    return {
        "applicable": len(out),
        "both_correct": both / len(out) if out else 0.0,
        "rows": out,
    }


def evaluate_identity_rescue(candidate, dev, *, budget: Budget, tag: str) -> dict:
    base = candidate.binding
    rows = []
    for fixture in dev[:8]:
        state = initial_binding_state(base, f"{tag}-rescue-{fixture['index']}")
        prepared = prepare(base, state, fixture["goal"].encode("utf-8"), fixture["records"])
        state, gate_index = observe_goal(base, prepared)
        with torch.no_grad():
            baseline = answer_step(base, prepared, state, EOS, gate_index=gate_index)
            intervened = answer_step(
                base, prepared, state, EOS, gate_index=gate_index, gamma_override=0.0
            )
            restored = answer_step(base, prepared, state, EOS, gate_index=gate_index)
        rows.append(
            {
                "index": fixture["index"],
                "intervention_diff": float(
                    (intervened.probabilities - baseline.probabilities).abs().max()
                ),
                "rescue_equal": bool(torch.equal(restored.probabilities, baseline.probabilities)),
                "rescue_max_abs_diff": float(
                    (restored.probabilities - baseline.probabilities).abs().max()
                ),
            }
        )
        budget.ensure(context={"phase": "identity_rescue"})
    return {
        "equal": sum(1 for row in rows if row["rescue_equal"]),
        "denominator": len(rows),
        "rows": rows,
    }


def evaluate_interventions(
    candidate, dev, *, arm: str, true_gen: dict, budget: Budget, tag: str
) -> dict:
    targets = [fixture for fixture in dev if fixture["index"] % 15 == 0]
    rows = []
    for fixture in targets:
        reference = true_gen.get(
            (fixture["world"], fixture["index"]), {"raw": b"", "terminated": False}
        )
        true_scored = score_generation(
            reference["raw"], fixture["target_prefix"], reference["terminated"]
        )
        no_source = generate(
            candidate,
            fixture,
            arm=arm,
            records=(),
            keys=(),
            subject_id=f"{tag}-int-nosrc-{fixture['index']}",
        )
        alternate = alternate_object_index(fixture)
        if alternate is None:
            continue
        wrong_record = generate(
            candidate,
            fixture,
            arm=arm,
            chosen_record=alternate,
            subject_id=f"{tag}-int-wrec-{fixture['index']}",
        )
        wrong_position = locked_generate_offset0(
            candidate,
            fixture,
            record_index=fixture["selection_label"],
            subject_id=f"{tag}-int-wpos-{fixture['index']}",
        )
        rows.append(
            {
                "index": fixture["index"],
                "world": fixture["world"],
                "true_completed": true_scored["completed"],
                "no_source_completed": score_generation(
                    no_source["raw_bytes"], fixture["target_prefix"], no_source["terminated"]
                )["completed"],
                "wrong_record_completed": score_generation(
                    wrong_record["raw_bytes"], fixture["target_prefix"], wrong_record["terminated"]
                )["completed"],
                "wrong_position_completed": score_generation(
                    wrong_position["raw_bytes"],
                    fixture["target_prefix"],
                    wrong_position["terminated"],
                )["completed"],
            }
        )
        budget.ensure(context={"phase": "content_interventions"})
    true_rate = sum(1 for row in rows if row["true_completed"]) / len(rows)
    drops = {
        name: {
            "completed_rate": sum(1 for row in rows if row[f"{name}_completed"]) / len(rows),
            "drop": true_rate - sum(1 for row in rows if row[f"{name}_completed"]) / len(rows),
        }
        for name in ("no_source", "wrong_record", "wrong_position")
    }
    return {
        "applicable": len(rows),
        "true_completed_rate": true_rate,
        "drops": drops,
        "rows": rows,
    }


def evaluate_nll(candidate, dev, *, arm: str, budget: Budget, tag: str) -> dict:
    total = 0.0
    for fixture in dev:
        with torch.no_grad():
            if ARM_LOCKED[arm]:
                total += float(
                    locked_loss(
                        candidate,
                        fixture["goal"].encode("utf-8"),
                        fixture["records"],
                        fixture["record_keys"],
                        fixture["target_prefix"],
                        subject_id=f"{tag}-nll-{fixture['index']}",
                    )
                )
            else:
                total += float(
                    score_example(
                        candidate.binding,
                        fixture["goal"].encode("utf-8"),
                        fixture["records"],
                        fixture["target_prefix"],
                        subject_id=f"{tag}-nll-{fixture['index']}",
                    ).nll
                )
        budget.ensure(context={"phase": "nll"})
    return {"mean_nll": total / len(dev), "examples": len(dev)}


def floor_sensitivity(bundle_path: Path, dev, *, budget: Budget) -> dict:
    """Reload the stored step-300 bundle and regenerate with gamma_floor=0.0."""
    candidate, _state, _optimizer, _profile, _update, _cursor, _payload = load_bundle(
        bundle_path, torch.float64
    )
    fixtures = dev[:SENSITIVITY_EXAMPLES]
    rows = []
    for fixture in fixtures:
        generated = generate(
            candidate,
            fixture,
            arm="A-locked",
            floor=0.0,
            subject_id=f"sens-{fixture['index']}",
        )
        scored = score_generation(
            generated["raw_bytes"], fixture["target_prefix"], generated["terminated"]
        )
        rows.append(
            {
                "index": fixture["index"],
                "completed": scored["completed"],
                "gamma_mean": generated["gamma_mean"],
            }
        )
        budget.ensure(context={"phase": "floor_sensitivity"})
    return {
        "examples": len(rows),
        "completed_rate": sum(1 for row in rows if row["completed"]) / len(rows),
        "gamma_mean": sum(row["gamma_mean"] for row in rows) / len(rows),
        "rows": rows,
    }


def train_config(payload, train, dev, *, seed: int, arm: str, updates: int, budget: Budget) -> dict:
    profile = f"{arm}-f64"
    candidate, optimizer = build_candidate(payload, profile, seed)
    tag = f"{seed}-{arm}"

    step0, _raw0 = evaluate(candidate, dev, arm=arm, full=False, budget=budget, tag=tag)
    checkpoint_path = BUNDLE_DIR / f"{seed}-{arm}-step{CHECKPOINT_UPDATE}.pt"
    resumed_from = 0
    if checkpoint_path.exists():
        loaded = load_bundle(checkpoint_path, torch.float64)
        candidate, _state, optimizer, _p, resumed_from, _c, _payload = loaded
        del loaded
        candidate.configure_trainable()

    batches = schedule(seed, train, updates)
    rows = []
    for update in range(resumed_from + 1, updates + 1):
        stage = stage_of(update)
        batch = batches[update - 1]
        completed = [
            teacher_forced(
                candidate,
                fixture,
                arm=arm,
                subject_id=f"{tag}-tf-{fixture['index']}",
            )["completed"]
            for fixture in batch
        ]
        rate = sum(1 for flag in completed if flag) / len(completed)
        self_prefix = rate >= CURRICULUM_THRESHOLD
        loss, gamma = batch_objective(
            candidate,
            batch,
            arm=arm,
            stage=stage,
            self_prefix=self_prefix,
            subject_tag=f"{tag}-train-{update}",
        )
        optimizer.zero_grad()
        loss.backward()
        clip = float(
            torch.nn.utils.clip_grad_norm_(list(candidate.trainable_parameters()), CLIP_NORM)
        )
        optimizer.step()
        budget.adam += 1
        rows.append(
            {
                "update": update,
                "stage": stage,
                "loss": float(loss.detach()),
                "clip_scale": clip,
                "curriculum_rate": rate,
                "self_prefix": bool(self_prefix),
                "gamma_steps": gamma["steps"],
                "gamma_model_mean": gamma["gamma_model_mean"],
                "gamma_used_mean": gamma["gamma_used_mean"],
                "floored_steps": gamma["floored_steps"],
                "floored_fraction": gamma["floored_fraction"],
            }
        )
        budget.ensure(context={"seed": seed, "arm": arm, "update": update, "phase": "train"})
        if update == CHECKPOINT_UPDATE:
            info = save_bundle(
                checkpoint_path,
                profile=profile,
                candidate=candidate,
                state=initial_binding_state(candidate.binding, f"ckpt-{tag}"),
                optimizer=optimizer,
                update_index=update,
                cursor={"stage": "N16.1.1", "seed": seed, "arm": arm, "update": update},
                parent_identity={"sha256": PARENT_SHA256},
            )
            budget.add_product(info)
        if update % 20 == 0:
            write_json(
                OUTPUT_DIR / f"progress-{tag}.json",
                {
                    "stage": "N16.1.1",
                    "seed": seed,
                    "arm": arm,
                    "update": update,
                    "updates_total": updates,
                    "last_loss": rows[-1]["loss"],
                    "curriculum_rate_max": max(row["curriculum_rate"] for row in rows),
                    "self_prefix_updates": sum(1 for row in rows if row["self_prefix"]),
                    "floored_step_fraction": (
                        sum(row["floored_steps"] for row in rows)
                        / sum(row["gamma_steps"] for row in rows)
                        if sum(row["gamma_steps"] for row in rows)
                        else 0.0
                    ),
                    "cost": {"cpu_seconds": budget.cpu(), "wall_seconds": budget.wall()},
                },
            )

    final, true_gen = evaluate(candidate, dev, arm=arm, full=True, budget=budget, tag=tag)
    final_path = BUNDLE_DIR / f"{seed}-{arm}-step{updates}.pt"
    final_info = save_bundle(
        final_path,
        profile=profile,
        candidate=candidate,
        state=initial_binding_state(candidate.binding, f"final-{tag}"),
        optimizer=optimizer,
        update_index=updates,
        cursor={"stage": "N16.1.1", "seed": seed, "arm": arm, "update": updates},
        parent_identity={"sha256": PARENT_SHA256},
    )
    budget.add_product(final_info)

    sensitivity = (
        floor_sensitivity(final_path, dev, budget=budget)
        if ARM_LOCKED[arm]
        else {
            "applicable": False,
            "reason": "arm B has no gamma floor",
        }
    )
    sensitivity_true = None
    if ARM_LOCKED[arm]:
        subset = [
            row
            for row in final["rows"]
            if row["index"] in {item["index"] for item in sensitivity["rows"]}
        ]
        sensitivity_true = sum(1 for row in subset if row["score"]["completed"]) / len(subset)

    deleted = []
    if checkpoint_path.exists():
        deleted.append({"path": str(checkpoint_path), "bytes": checkpoint_path.stat().st_size})
        checkpoint_path.unlink()
    for path in (OUTPUT_DIR / f"progress-{tag}.json",):
        if path.exists():
            deleted.append({"path": str(path), "bytes": path.stat().st_size})
            path.unlink()

    intermediate = [row for row in rows if row["update"] == 1 or row["update"] == updates]
    return {
        "schema": "taiji-binding-select-compare-config-v1",
        "step": "N16.1.1",
        "seed": seed,
        "arm": arm,
        "profile": profile,
        "updates": updates,
        "resumed_from": resumed_from,
        "step0": step0,
        "step300": final,
        "train": {
            "first_loss": rows[0]["loss"],
            "last_loss": rows[-1]["loss"],
            "loss_last20_mean": statistics.fmean(row["loss"] for row in rows[-20:]),
            "clip_min": min(row["clip_scale"] for row in rows),
            "clip_max": max(row["clip_scale"] for row in rows),
            "self_prefix_updates": sum(1 for row in rows if row["self_prefix"]),
            "curriculum_rate_max": max(row["curriculum_rate"] for row in rows),
            "curriculum_by_stage": {
                stage: {
                    "updates": sum(1 for row in rows if row["stage"] == stage),
                    "mean_rate": statistics.fmean(
                        row["curriculum_rate"] for row in rows if row["stage"] == stage
                    ),
                    "self_prefix_updates": sum(
                        1 for row in rows if row["stage"] == stage and row["self_prefix"]
                    ),
                }
                for stage, _low, _high in STAGES
            },
            "gamma_by_stage": {
                stage: {
                    "updates": sum(1 for row in rows if row["stage"] == stage),
                    "steps": sum(row["gamma_steps"] for row in rows if row["stage"] == stage),
                    "gamma_model_mean": (
                        sum(
                            row["gamma_model_mean"] * row["gamma_steps"]
                            for row in rows
                            if row["stage"] == stage
                        )
                        / sum(row["gamma_steps"] for row in rows if row["stage"] == stage)
                        if sum(row["gamma_steps"] for row in rows if row["stage"] == stage)
                        else 0.0
                    ),
                    "gamma_used_mean": (
                        sum(
                            row["gamma_used_mean"] * row["gamma_steps"]
                            for row in rows
                            if row["stage"] == stage
                        )
                        / sum(row["gamma_steps"] for row in rows if row["stage"] == stage)
                        if sum(row["gamma_steps"] for row in rows if row["stage"] == stage)
                        else 0.0
                    ),
                    "floored_steps": sum(
                        row["floored_steps"] for row in rows if row["stage"] == stage
                    ),
                    "floored_step_fraction": (
                        sum(row["floored_steps"] for row in rows if row["stage"] == stage)
                        / sum(row["gamma_steps"] for row in rows if row["stage"] == stage)
                        if sum(row["gamma_steps"] for row in rows if row["stage"] == stage)
                        else 0.0
                    ),
                    "updates_with_any_floored": sum(
                        1 for row in rows if row["stage"] == stage and row["floored_steps"] > 0
                    ),
                    "updates_with_any_floored_fraction": (
                        sum(1 for row in rows if row["stage"] == stage and row["floored_steps"] > 0)
                        / sum(1 for row in rows if row["stage"] == stage)
                    ),
                }
                for stage, _low, _high in STAGES
            },
            "rows": rows,
            "endpoint_losses": intermediate,
        },
        "sensitivity": {
            **sensitivity,
            "true_completed_rate_on_same_subset": sensitivity_true,
            "delta": (
                sensitivity_true - sensitivity["completed_rate"]
                if sensitivity_true is not None
                else None
            ),
        },
        "bundle": {
            "path": str(final_path),
            "bytes": int(final_info["bytes"]),
            "sha256": final_info["sha256"],
        },
        "deleted": deleted,
        "cost": {
            "cpu_seconds": budget.cpu() - budget.prior_cpu,
            "wall_seconds": budget.wall() - budget.prior_wall,
            "adam_calls": budget.adam,
        },
        "source_hashes": {path: sha256_file(ROOT / path) for path in SOURCE_FILES},
    }


def snapshot(candidate, state, optimizer) -> dict:
    tensors: dict[str, torch.Tensor] = {}
    for name, value in candidate_parameter_map(candidate.binding).items():
        tensors[f"param/{name}"] = value
    for name, value in candidate.head.state_dict().items():
        if isinstance(value, torch.Tensor):
            tensors[f"head/{name}"] = value
    for index, entry in optimizer.state_dict()["state"].items():
        for field, value in entry.items():
            if isinstance(value, torch.Tensor):
                tensors[f"opt/{index}/{field}"] = value
    for name, value in state.payload().items():
        if isinstance(value, torch.Tensor):
            tensors[f"state/{name}"] = value
    return {
        name: hashlib.sha256(
            f"{tuple(value.detach().cpu().contiguous().shape)}|"
            f"{value.detach().cpu().contiguous().dtype}".encode()
            + value.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest()
        for name, value in sorted(tensors.items())
    }


def probability_digest(candidate, fixture) -> str:
    base = candidate.binding
    state = initial_binding_state(base, "precheck-forward")
    prepared = prepare(base, state, fixture["goal"].encode("utf-8"), fixture["records"])
    state, gate_index = observe_goal(base, prepared)
    with torch.no_grad():
        step = answer_step(base, prepared, state, EOS, gate_index=gate_index)
    tensor = step.probabilities.detach().cpu().contiguous()
    return hashlib.sha256(
        f"{tuple(tensor.shape)}|{tensor.dtype}".encode() + tensor.numpy().tobytes()
    ).hexdigest()


def worker(spec: dict) -> None:
    start_cpu = time.process_time()
    out: dict = {"ok": False}
    try:
        candidate, state, optimizer, _profile, _update, _cursor, _payload = load_bundle(
            Path(spec["bundle"]), torch.float64
        )
        fixture = None
        if spec.get("fixture") is not None:
            row = spec["fixture"]
            fixture = {
                "goal": row["goal"],
                "records": tuple(
                    RawSource(source_id=item["source_id"], raw=item["raw"].encode("utf-8"))
                    for item in row["records"]
                ),
            }
        out = {
            "ok": True,
            "snapshot": snapshot(candidate, state, optimizer),
            "probability": probability_digest(candidate, fixture) if fixture else None,
        }
    except Exception:
        out = {"ok": False, "message": traceback.format_exc()}
    out["cpu_seconds"] = time.process_time() - start_cpu
    out["adam_calls"] = 0
    torch.save(out, spec["result"])


def spawn(spec: dict) -> dict:
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=worker, args=(spec,))
    process.start()
    process.join(timeout=WORKER_TIMEOUT)
    if process.is_alive():
        process.terminate()
        process.join()
        return {"ok": False, "message": "worker timed out", "cpu_seconds": 0.0}
    if process.exitcode != 0:
        return {"ok": False, "message": f"exit {process.exitcode}", "cpu_seconds": 0.0}
    path = Path(spec["result"])
    if not path.exists():
        return {"ok": False, "message": "no result", "cpu_seconds": 0.0}
    return torch.load(path, map_location="cpu", weights_only=False)


def run_precheck(payload, dev) -> int:
    """Save, reload in an independent process, and compare (0 Adam)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    budget = Budget()
    fixture = dev[0]
    rows = []
    instrument = []
    for arm in ARMS:
        profile = f"{arm}-f64"
        candidate, optimizer = build_candidate(payload, profile, SEEDS[0])
        with torch.no_grad():
            if ARM_LOCKED[arm]:
                plain_loss = float(
                    locked_loss(
                        candidate,
                        fixture["goal"].encode("utf-8"),
                        fixture["records"],
                        fixture["record_keys"],
                        fixture["target_prefix"],
                        subject_id="precheck-parity",
                    )
                )
                trace: list = []
                traced_loss = float(
                    locked_loss(
                        candidate,
                        fixture["goal"].encode("utf-8"),
                        fixture["records"],
                        fixture["record_keys"],
                        fixture["target_prefix"],
                        subject_id="precheck-parity",
                        trace=trace,
                    )
                )
                plain = locked_generate(
                    candidate,
                    fixture["goal"].encode("utf-8"),
                    fixture["records"],
                    fixture["record_keys"],
                    subject_id="precheck-parity",
                )
                trace_gen: list = []
                traced = locked_generate(
                    candidate,
                    fixture["goal"].encode("utf-8"),
                    fixture["records"],
                    fixture["record_keys"],
                    subject_id="precheck-parity",
                    trace=trace_gen,
                )
                forced = teacher_forced(candidate, fixture, arm=arm, subject_id="precheck-parity")
                instrument.append(
                    {
                        "arm": arm,
                        "loss_trace_equal": plain_loss == traced_loss,
                        "generation_trace_equal": plain["raw_bytes"] == traced["raw_bytes"]
                        and plain["terminated"] == traced["terminated"],
                        "teacher_forced_loss_equal": abs(forced["nll"] - traced_loss) < 1e-12,
                        "trace_rows": len(trace),
                        "gamma_model_mean": traced.get("gamma_model_mean"),
                        "gamma_floored_steps": traced.get("gamma_floored_steps"),
                    }
                )
            else:
                plain = generate_bytes(
                    candidate.binding,
                    fixture["goal"].encode("utf-8"),
                    fixture["records"],
                    subject_id="precheck-parity",
                )
                trace_gen: list = []
                traced = generate_unlocked_traced(
                    candidate,
                    fixture["goal"].encode("utf-8"),
                    fixture["records"],
                    subject_id="precheck-parity",
                    trace=trace_gen,
                )
                instrument.append(
                    {
                        "arm": arm,
                        "generation_trace_equal": plain["raw_bytes"] == traced["raw_bytes"]
                        and plain["terminated"] == traced["terminated"],
                        "trace_rows": len(trace_gen),
                    }
                )
        state = initial_binding_state(candidate.binding, f"pre-{arm}")
        path = OUTPUT_DIR / f"precheck-{arm}.pt"
        info = save_bundle(
            path,
            profile=profile,
            candidate=candidate,
            state=state,
            optimizer=optimizer,
            update_index=0,
            cursor={"stage": "N16.1.1-precheck", "arm": arm},
            parent_identity={"sha256": PARENT_SHA256},
        )
        budget.add_product(info)
        live = {
            "snapshot": snapshot(candidate, state, optimizer),
            "probability": probability_digest(candidate, fixture),
        }
        result_path = OUTPUT_DIR / f"precheck-{arm}-worker.pt"
        worker_out = spawn(
            {
                "bundle": str(path),
                "result": str(result_path),
                "fixture": {
                    "goal": fixture["goal"],
                    "records": [
                        {"source_id": record.source_id, "raw": record.raw.decode("utf-8")}
                        for record in fixture["records"]
                    ],
                },
            }
        )
        rows.append(
            {
                "arm": arm,
                "bundle_bytes": int(info["bytes"]),
                "save_ok": True,
                "load_worker_ok": bool(worker_out.get("ok")),
                "snapshot_equal": bool(worker_out.get("ok"))
                and worker_out["snapshot"] == live["snapshot"],
                "probability_equal": bool(worker_out.get("ok"))
                and worker_out["probability"] == live["probability"],
            }
        )
        path.unlink(missing_ok=True)
        result_path.unlink(missing_ok=True)
    checks = {
        "grid": len(rows) == len(ARMS),
        "save": all(row["save_ok"] for row in rows),
        "independent_load": all(row["load_worker_ok"] for row in rows),
        "snapshot": all(row["snapshot_equal"] for row in rows),
        "probability": all(row["probability_equal"] for row in rows),
        "instrument_parity": all(
            item.get("generation_trace_equal", False)
            and item.get("loss_trace_equal", True)
            and item.get("teacher_forced_loss_equal", True)
            for item in instrument
        ),
        "zero_adam": budget.adam == 0,
    }
    payload_out = {
        "schema": "taiji-binding-select-compare-precheck-v1",
        "step": "N16.1.1",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "arms": rows,
        "instrument": instrument,
        "cost": {
            "cpu_seconds": budget.cpu() - budget.prior_cpu,
            "wall_seconds": budget.wall() - budget.prior_wall,
            "adam_calls": budget.adam,
        },
    }
    write_json(PRECHECK, payload_out)
    print(f"precheck: {payload_out['status']} checks={checks}")
    print(
        f"  cpu {payload_out['cost']['cpu_seconds']:.2f}, "
        f"wall {payload_out['cost']['wall_seconds']:.2f}, adam 0"
    )
    return 0 if all(checks.values()) else 1


def run_config(args) -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    budget = Budget()
    status = "failed"
    try:
        if sha256_file(ROOT / PARENT_BUNDLE) != PARENT_SHA256:
            raise SystemExit("parent bundle sha drifted")
        payload = torch.load(ROOT / PARENT_BUNDLE, map_location="cpu", weights_only=False)
        train, dev, _select, _data = load_data()
        result = train_config(
            payload, train, dev, seed=args.seed, arm=args.arm, updates=args.updates, budget=budget
        )
        result["cost"] = {
            "cpu_seconds": budget.cpu() - budget.prior_cpu,
            "wall_seconds": budget.wall() - budget.prior_wall,
            "adam_calls": budget.adam,
            "prior_cpu_seconds": budget.prior_cpu,
            "prior_wall_seconds": budget.prior_wall,
        }
        write_json(RESULTS_DIR / f"{args.seed}-{args.arm}.json", result)
        status = "completed"
        print(
            f"{args.seed}-{args.arm}: step300 overall={result['step300']['overall']['completed']:.4f} "
            f"selection={result['step300']['selection']['top1']:.4f} "
            f"cpu={result['cost']['cpu_seconds']:.1f} adam={budget.adam}"
        )
        return 0
    except WatchLine as exc:
        info = json.loads(str(exc))
        context = info.get("context", {}) or {}
        update = int(context.get("update", 0) or 0)
        cpu = budget.cpu()
        per_update = cpu / update if update else 0.0
        completed = len(list(RESULTS_DIR.glob("*.json")))
        remaining_configs = max(0, len(SEEDS) * len(ARMS) - completed - 1)
        projection = (
            cpu
            + per_update * max(0, args.updates - update)
            + per_update * args.updates * remaining_configs
            if update
            else None
        )
        write_json(
            PARTIAL.format(seed=args.seed, arm=args.arm),
            {
                "stage": "N16.1.1",
                "status": "watch_line_reported",
                "detail": info,
                "seed": args.seed,
                "arm": args.arm,
                "progress": {
                    "updates_done": update,
                    "updates_total": args.updates,
                    "completed_configs": completed,
                    "cpu_per_update": per_update,
                    "projected_stage_cpu_seconds": projection,
                },
            },
        )
        print(f"WATCH LINE reached: {info}")
        print(f"  progress update={update}, completed configs={completed}, projection={projection}")
        status = "watch_line"
        return 2
    except CompareStop as exc:
        print(f"compare stop: {exc}")
        return 1
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        append_ledger(budget.row(status=status, seed=args.seed, arm=args.arm))


def bootstrap(deltas: list[float]) -> dict:
    generator = torch.Generator().manual_seed(BOOTSTRAP_SEED)
    n = len(deltas)
    means = []
    for _ in range(BOOTSTRAP_REPS):
        idx = torch.randint(0, n, (n,), generator=generator).tolist()
        means.append(sum(deltas[index] for index in idx) / n)
    means.sort()
    return {
        "mean": sum(deltas) / n,
        "lower": means[int(0.025 * BOOTSTRAP_REPS)],
        "upper": means[int(0.975 * BOOTSTRAP_REPS)],
        "reps": BOOTSTRAP_REPS,
        "seed": BOOTSTRAP_SEED,
    }


def world_rates(entry: dict) -> dict[str, float]:
    return {
        world: stats["completed"] / stats["n"]
        for world, stats in entry["step300"]["overall"]["by_world"].items()
    }


def contrast(entries: dict, base_arm: str, other_arm: str) -> dict:
    per_seed = {}
    for seed in SEEDS:
        base = world_rates(entries[(seed, base_arm)])
        other = world_rates(entries[(seed, other_arm)])
        deltas = [other[world] - base[world] for world in sorted(base)]
        per_seed[str(seed)] = {
            "delta_mean": sum(deltas) / len(deltas),
            "bootstrap": bootstrap(deltas),
            "world_deltas": deltas,
        }
    pooled = [stat["delta_mean"] for stat in per_seed.values()]
    return {
        "base_arm": base_arm,
        "other_arm": other_arm,
        "per_seed": per_seed,
        "pooled_mean": sum(pooled) / len(pooled),
    }


def task_checks(entry: dict) -> dict:
    overall = entry["step300"]["overall"]
    return {
        "unseen_output_name": overall["subsets"]["unseen_output_name"] >= GATE_UNSEEN,
        "new_description": overall["subsets"]["new_description"] >= GATE_NEW_DESCRIPTION,
        "overall": overall["completed"] >= GATE_OVERALL,
        "reversed_paired": entry["step300"]["reversed"]["paired_rate"] >= GATE_REVERSED,
    }


def mechanism_checks(entry: dict, increment: dict) -> dict:
    gamma = entry["step300"]["gamma"]
    rescue = entry["step300"]["identity_rescue"]
    interventions = entry["step300"]["content_interventions"]
    return {
        "selection_top1": entry["step300"]["selection"]["top1"] >= GATE_SELECTION,
        "gamma_model_mean": gamma["gamma_model_mean"] >= GATE_GAMMA,
        "identity_rescue": rescue["equal"] == rescue["denominator"] == GATE_RESCUE,
        "content_no_source": interventions["drops"]["no_source"]["drop"] >= GATE_CONTENT_DROP,
        "content_wrong_record": interventions["drops"]["wrong_record"]["drop"] >= GATE_CONTENT_DROP,
        "content_wrong_position": interventions["drops"]["wrong_position"]["drop"]
        >= GATE_CONTENT_DROP,
        "increment_pooled": increment["pooled_mean"] >= GATE_INCREMENT,
        "increment_bootstrap": all(
            stat["bootstrap"]["lower"] > 0 for stat in increment["per_seed"].values()
        ),
    }


def build_report(configs: list[dict], extra: dict | None = None) -> dict:
    entries = {(entry["seed"], entry["arm"]): entry for entry in configs}
    increment = contrast(entries, "B-select", "A-locked")
    task = {f"{seed}-A": task_checks(entries[(seed, "A-locked")]) for seed in SEEDS}
    mechanism = {
        f"{seed}-A": mechanism_checks(entries[(seed, "A-locked")], increment) for seed in SEEDS
    }
    gamma_rows = {}
    for seed in SEEDS:
        for arm in ARMS:
            entry = entries[(seed, arm)]
            gamma_rows[f"{seed}-{arm}"] = {
                "dev_step": {
                    key: entry["step300"]["gamma"].get(key)
                    for key in (
                        "steps",
                        "gamma_model_mean",
                        "gamma_used_mean",
                        "copy_weight_mean",
                        "floored_steps",
                        "floored_step_fraction",
                        "examples_with_any_floored",
                        "examples_with_any_floored_fraction",
                    )
                },
                "train": entry["train"]["gamma_by_stage"],
                "floor_removed": entry["sensitivity"],
                "step0_model_mean": entry["step0"]["gamma"]["gamma_model_mean"],
            }
    precheck = json.loads(PRECHECK.read_text(encoding="utf-8")) if PRECHECK.exists() else None
    task_ok = all(all(checks.values()) for checks in task.values())
    mechanism_ok = all(all(checks.values()) for checks in mechanism.values())
    if task_ok and mechanism_ok:
        status = "selection_locked_binding_supported"
    elif task_ok:
        status = "task_supported_component_gain_unresolved"
    else:
        status = "native_byte_task_failed"
    if precheck is None or precheck.get("status") != "passed":
        status = "engineering_failure"

    configs_out = []
    for entry in configs:
        slim = dict(entry)
        configs_out.append(slim)
    cost = {
        "cpu_seconds": sum(entry["cost"]["cpu_seconds"] for entry in configs),
        "wall_seconds": sum(entry["cost"]["wall_seconds"] for entry in configs),
        "adam_calls": sum(entry["cost"]["adam_calls"] for entry in configs),
    }
    extra = extra or {}
    cost_detail = {
        "configs_cpu_seconds": cost["cpu_seconds"],
        "configs_wall_seconds": cost["wall_seconds"],
        "precheck_cpu_seconds": (precheck or {}).get("cost", {}).get("cpu_seconds", 0.0),
        "precheck_wall_seconds": (precheck or {}).get("cost", {}).get("wall_seconds", 0.0),
        "aggregate_cpu_seconds": float(extra.get("cpu_seconds", 0.0)),
        "aggregate_wall_seconds": float(extra.get("wall_seconds", 0.0)),
        "total_cpu_seconds": cost["cpu_seconds"]
        + (precheck or {}).get("cost", {}).get("cpu_seconds", 0.0)
        + float(extra.get("cpu_seconds", 0.0)),
        "total_wall_seconds": cost["wall_seconds"]
        + (precheck or {}).get("cost", {}).get("wall_seconds", 0.0)
        + float(extra.get("wall_seconds", 0.0)),
    }
    deleted = [item for entry in configs for item in entry.get("deleted", [])]
    return {
        "schema": "taiji-binding-select-compare-v1",
        "step": "N16.1.1",
        "date": "2026-11-03",
        "status": status,
        "verdict": status,
        "gates": {"task": task, "mechanism": mechanism},
        "contrast_A_minus_B": increment,
        "gamma_floor": {
            "floor": GAMMA_FLOOR,
            "gate_reading": (
                "gamma multiplies the generated branch, so the 0.2 clamp is a generation "
                "floor, not a copy floor.  The capability reading is the model gamma "
                "before the clamp; the used gamma is >= floor by construction; the copy "
                "weight is 1 - model gamma and must be reported separately."
            ),
            "per_config": gamma_rows,
        },
        "precheck": precheck,
        "configs": configs_out,
        "release": {
            "N16.1.2": "released" if (task_ok and mechanism_ok) else "not_released",
            "confirm": "sealed" if not (task_ok and mechanism_ok) else "sealed_until_1.2_run",
        },
        "budget_ruling": {
            "date": "2026-11-02",
            "scope": "owner sufficient budget; observation line only",
            "observation_line_cpu": WATCH_CPU_SECONDS,
            "observation_line_wall": WATCH_WALL_SECONDS,
            "dose_unchanged": "2 seeds x 2 arms x 300 updates, batch 8, 288 train / 240 dev",
            "price_note": (
                "0c froze 8,390.9 CPU for 1,200 updates plus 2,400 dev generations only. "
                "1.1 additionally runs the per-update teacher-forced curriculum probe and the "
                "reversed/counterfactual/intervention/sensitivity generations; this is an "
                "arithmetic extension declared before the run, not a re-pricing."
            ),
        },
        "accounting": {
            "actual_adam_calls": cost["adam_calls"],
            "effective_adam_calls": cost["adam_calls"],
            "consumed_train_examples": cost["adam_calls"] * BATCH,
            "consumed_dev_generations": sum(
                240 * 2
                + 240
                + 240
                + (72 if entry["arm"] == "A-locked" else 0)
                + (SENSITIVITY_EXAMPLES if entry["arm"] == "A-locked" else 0)
                for entry in configs
            ),
            "consumed_confirm_examples": 0,
        },
        "cost": cost,
        "cost_detail": cost_detail,
        "products": {
            "bundles": [entry["bundle"] for entry in configs],
            "bundle_bytes": sum(entry["bundle"]["bytes"] for entry in configs),
            "deleted": deleted,
            "deleted_bytes": sum(item["bytes"] for item in deleted),
        },
        "source_hashes": {path: sha256_file(ROOT / path) for path in SOURCE_FILES},
        "limits": (
            "Formal bounded comparison on the declared native-byte task only. No confirm read, "
            "no N15.2, no step 12/13, no 14.7 release, no capability claim beyond this task."
        ),
    }


def run_aggregate() -> int:
    results = sorted(RESULTS_DIR.glob("*.json"))
    if len(results) != len(SEEDS) * len(ARMS):
        print(f"aggregate needs 4 configs, found {len(results)}")
        return 1
    configs = [json.loads(path.read_text(encoding="utf-8")) for path in results]
    extra = {
        "cpu_seconds": time.process_time() - STARTED_CPU,
        "wall_seconds": time.perf_counter() - STARTED_WALL,
    }
    report = build_report(configs, extra)
    write_json(ROOT / REPORT, report)
    manifest = {
        "schema": "taiji-binding-select-manifest-v1",
        "step": "N16.1.1",
        "status": report["status"],
        "report": REPORT,
        "parent_bundle": PARENT_BUNDLE,
        "parent_sha256": PARENT_SHA256,
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "updates": UPDATES,
        "batch": BATCH,
        "stages": {name: [low, high] for name, low, high in STAGES},
        "curriculum_threshold": CURRICULUM_THRESHOLD,
        "gamma_floor": GAMMA_FLOOR,
        "gates": report["gates"],
        "release": report["release"],
        "contrast": report["contrast_A_minus_B"],
        "gamma_floor_report": report["gamma_floor"],
        "precheck": report["precheck"],
        "cost": report["cost"],
        "accounting": report["accounting"],
        "products": report["products"],
        "source_hashes": report["source_hashes"],
        "notes": (
            "1.2 released only when both the task gate and the mechanism gate pass; otherwise "
            "confirm stays sealed."
        ),
    }
    write_json(ROOT / MANIFEST, manifest)
    cleanup_bytes = 0
    cleanup_files = []
    for path in results:
        cleanup_bytes += path.stat().st_size
        cleanup_files.append(str(path))
        path.unlink()
    if LEDGER.exists():
        cleanup_bytes += LEDGER.stat().st_size
        cleanup_files.append(str(LEDGER))
        LEDGER.unlink()
    manifest["cleanup"] = {"files": cleanup_files, "bytes": cleanup_bytes}
    write_json(ROOT / MANIFEST, manifest)
    print(f"1.1 status: {report['status']}")
    print(f"  task: {report['gates']['task']}")
    print(f"  mechanism: {report['gates']['mechanism']}")
    print(f"  A-B pooled: {report['contrast_A_minus_B']['pooled_mean']:.4f}")
    print(f"  cost: {report['cost']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="run", choices=("run", "precheck", "aggregate"))
    parser.add_argument("--seed", type=int, default=SEEDS[0])
    parser.add_argument("--arm", default=ARMS[0], choices=ARMS)
    parser.add_argument("--updates", type=int, default=UPDATES)
    args = parser.parse_args()

    if args.mode == "aggregate":
        return run_aggregate()
    if args.mode == "precheck":
        payload = torch.load(ROOT / PARENT_BUNDLE, map_location="cpu", weights_only=False)
        _train, dev, _select, _data = load_data()
        return run_precheck(payload, dev)
    return run_config(args)


if __name__ == "__main__":
    raise SystemExit(main())
