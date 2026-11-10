"""14.6 formal native Chinese comparison for the record-relation candidate.

Two incremental module seeds x three structural arms, 300 batch-8 updates per
arm, the full 192 train / 160 dev composition split, and the fixed step-0 /
step-300 evaluation endpoint.  The stage compares ``A_full`` against
``B_local`` and ``C_flat``, reads the declared counterfactual conditions, and
decides by the pre-registered task, component and content-path gates.

The script has two stages:

``smoke``
    one seed x three arms x two engineering updates plus a small evaluation
    sample; it proves the runner writes, reloads and independently restores its
    own checkpoints before the formal run, and that the traced generator
    matches ``generate_bytes`` bit for bit.

``compare``
    the formal 1,800-update run.  It is write-once, honours the frozen 14.5
    price and the declared 14,400 CPU / 28,800 wall / 1,800 Adam / 4 GiB
    product cap, and never reads the sealed confirm split.

The subject graph, the content subgraph, the fixture module and the frozen
14.4/14.5/14.6.0 cards are not modified here.  All training and evaluation
parameters are retained only where the manifest says so.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import random
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import contextlib

import torch  # noqa: E402

from taiji.candidate_manifest import build_manifest, parameter_digest  # noqa: E402
from taiji.core_09_bundle import BUNDLE_RELATIVE_PATH as CORE_09_BUNDLE_PATH  # noqa: E402
from taiji.core_09_bundle import EXPECTED_BYTES as CORE_09_BUNDLE_BYTES  # noqa: E402
from taiji.core_09_bundle import EXPECTED_SHA256 as CORE_09_BUNDLE_SHA  # noqa: E402
from taiji.living_core import EOS, RawSource  # noqa: E402
from taiji.native_composition_fixture import (  # noqa: E402
    FixtureExample,
    build_split,
    fixture_digest,
    fixture_payload,
    price_dev_examples,
)
from taiji.record_relation import RECORD_RELATION_SCHEMA, RecordRelationConfig  # noqa: E402
from taiji.record_relation_core import (  # noqa: E402
    ARM_FLAT,
    ARM_FULL,
    ARM_LOCAL,
    ARMS,
    CLIP_NORM,
    CONTENT_REVISION,
    MAX_OUTPUT_BYTES,
    MAX_PREDICTIONS,
    answer_step,
    build_candidate,
    candidate_digest,
    decode_endpoint,
    generate_bytes,
    load_bundle,
    make_optimizer,
    observe_goal,
    prepare,
    run_worker,
    save_bundle,
    score_example,
)

BUNDLE = CORE_09_BUNDLE_PATH
EXPECTED_PARENT = "feb0cbcb00e2e16bb66b77db530ae566ce40eecaf2e14852b7cc1661292139dd"
EXPECTED_BUNDLE_SHA = CORE_09_BUNDLE_SHA
EXPECTED_BUNDLE_BYTES = CORE_09_BUNDLE_BYTES
EXPECTED_14_4_CANDIDATE = "16f7e33cfc24101b3d702c5020e9b1aa1bacaacf4144de27acd8d25c98176352"
GRAPH_REPORT = "reports/taiji_core_record_relation_14_4_20261102.json"
GRAPH_MANIFEST = "plans/manifests/taiji_core_record_relation_14_4_20261102.json"
PRICE_REPORT = "reports/taiji_core_record_relation_14_5_20261102.json"
PRICE_MANIFEST = "plans/manifests/taiji_core_record_relation_14_5_20261102.json"
PREFLIGHT_REPORT = "reports/taiji_core_record_relation_14_6_preflight_20261102.json"
PREFLIGHT_MANIFEST = "plans/manifests/taiji_core_record_relation_14_6_preflight_20261102.json"
REPORT = "reports/taiji_core_record_relation_14_6_20261102.json"
MANIFEST = "plans/manifests/taiji_core_record_relation_14_6_20261102.json"
OUTPUT_DIR = "output/taiji_core_record_relation_14_6"

FORMAL_CPU_SECONDS = 14400
FORMAL_WALL_SECONDS = 28800
FORMAL_ADAM_CALLS = 1800
FORMAL_PRODUCT_BYTES = 4 * 1024**3
FORMAL_UPDATES_PER_ARM = 300
BATCH = 8
SEEDS = (27031, 27032)
ORDER_SEEDS = (37031, 37032)
CHECKPOINTS = (0, 100, 200, 300)
FIXED_WORLD = 0
BOOTSTRAP_SEED = 57031
BOOTSTRAP_ROUNDS = 2000
WORKER_TIMEOUT = 240

SMOKE_CPU_SECONDS = 300
SMOKE_WALL_SECONDS = 600
SMOKE_ADAM_CALLS = 6
SMOKE_PRODUCT_BYTES = 1 * 1024**3
SMOKE_UPDATES = 2
SMOKE_EXAMPLES = 8

TASK_TOTAL_MIN = 0.80
TASK_LAYER_MIN = 0.60
BOTH_SIDES_MIN = 0.60
REVERSE_MIN = 0.95
STRUCTURE_GAIN_MIN = 0.10
CONTENT_DROP_MIN = 0.10

EXPECTED_DEV = 160
EXPECTED_TRAIN = 192
EXPECTED_STEP0_REAL = 960
EXPECTED_STEP300_REAL = 960
EXPECTED_PERMUTED = 960
EXPECTED_REVERSE = 960
EXPECTED_STRUCTURAL_PER_CONDITION = 48
EXPECTED_NLL = 1920
EXPECTED_GENERATIONS = 4032

SOURCE_FILES = (
    "taiji/record_relation.py",
    "taiji/record_relation_core.py",
    "taiji/native_composition_fixture.py",
    "scripts/training/verify_taiji_record_relation_14_6.py",
)

STARTED_WALL = time.perf_counter()
STARTED_CPU = time.process_time()
STARTED_AT = time.strftime("%Y-%m-%dT%H:%M:%S%z")


class FormalStop(RuntimeError):
    """The declared stage budget was exhausted."""


class InstrumentError(RuntimeError):
    """A parent, data, seed, candidate or evaluation invariant failed."""


def sha256_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(
    rows: list,
    check_id: str,
    passed: bool,
    observed,
    note: str = "",
    *,
    required: bool = True,
) -> None:
    rows.append(
        {
            "id": check_id,
            "passed": bool(passed),
            "required": bool(required),
            "observed": observed,
            "note": note,
        }
    )


def max_abs_diff(first: torch.Tensor, second: torch.Tensor) -> float:
    return float((first.detach() - second.detach()).abs().max())


def tree_equal(first, second) -> bool:
    if torch.is_tensor(first) or torch.is_tensor(second):
        return bool(torch.equal(first, second))
    if isinstance(first, dict) and isinstance(second, dict):
        if set(first) != set(second):
            return False
        return all(tree_equal(first[key], second[key]) for key in first)
    if isinstance(first, (list, tuple)) and isinstance(second, (list, tuple)):
        if len(first) != len(second):
            return False
        return all(tree_equal(left, right) for left, right in zip(first, second, strict=True))
    return bool(first == second)


ARCHIVED_WRITE_PREFIXES = ("reports/", "plans/manifests/")


def refuse_archived_overwrite(path: Path) -> None:
    """默认拒写已归档的读数件。

    这些报告与清单是 2026-11-02 当年跑出来的原始读数；本脚本当前版本既缺当年的输入件
    （`output/taiji_core_09/subject_bundle.pt` 不在库），也没有记录运行条件，重跑即使同路径
    落盘也不构成复核，只会把原始读数覆盖掉。要出新读数，请用 `--report`／`--manifest`
    指向一个尚不存在的路径。
    """
    try:
        rel = path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return
    if path.exists() and rel.startswith(ARCHIVED_WRITE_PREFIXES):
        raise SystemExit(
            f"拒绝覆盖已归档读数 {rel}：本脚本缺当年输入件与运行条件，"
            "重写该文件会让原始读数不可追溯；请改用 --report／--manifest 指向新路径。"
        )


def write_json(path: Path, payload: dict) -> dict:
    refuse_archived_overwrite(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {"path": str(path), "sha256": sha256_text(path), "bytes": path.stat().st_size}


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def append_log(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_progress(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


class StageBudget:
    """CPU, wall, Adam and live-product accounting for one stage run."""

    def __init__(self, cpu_cap: int, wall_cap: int, adam_cap: int, product_cap: int) -> None:
        self.cpu_cap = cpu_cap
        self.wall_cap = wall_cap
        self.adam_cap = adam_cap
        self.product_cap = product_cap
        self.main_cpu = 0.0
        self.worker_cpu = 0.0
        self.worker_wall = 0.0
        self.adam_calls = 0
        self.recovered_adam_calls = 0
        self.live_products = 0
        self.max_products = 0
        self.cumulative_products = 0
        self.peak_ram_mb = 0.0

    def total_cpu(self) -> float:
        return self.main_cpu + self.worker_cpu

    def total_adam(self) -> int:
        return self.adam_calls + self.recovered_adam_calls

    def ensure(self) -> None:
        self.main_cpu = time.process_time() - STARTED_CPU
        if self.total_cpu() > self.cpu_cap:
            raise FormalStop(f"CPU budget exceeded: {self.total_cpu():.2f}")
        wall = time.perf_counter() - STARTED_WALL
        if wall > self.wall_cap:
            raise FormalStop(f"wall budget exceeded: {wall:.2f}")
        if self.total_adam() > self.adam_cap:
            raise FormalStop(f"Adam budget exceeded: {self.total_adam()}")
        if self.max_products > self.product_cap:
            raise FormalStop(f"product budget exceeded: {self.max_products}")

    def add_product(self, size: int) -> None:
        self.live_products += size
        self.cumulative_products += size
        self.max_products = max(self.max_products, self.live_products)

    def drop_product(self, size: int) -> None:
        self.live_products -= size

    def sample_ram(self) -> None:
        try:
            import psutil

            used = psutil.Process().memory_info().rss / 1024**2
            self.peak_ram_mb = max(self.peak_ram_mb, used)
        except Exception:  # noqa: BLE001
            return

    def cost(self) -> dict:
        self.main_cpu = time.process_time() - STARTED_CPU
        return {
            "main_cpu_seconds": self.main_cpu,
            "worker_cpu_seconds": self.worker_cpu,
            "total_cpu_seconds": self.total_cpu(),
            "wall_seconds": time.perf_counter() - STARTED_WALL,
            "worker_wall_seconds": self.worker_wall,
            "adam_calls_executed": self.adam_calls,
            "adam_calls_recovered": self.recovered_adam_calls,
            "adam_calls_total": self.total_adam(),
            "max_live_product_bytes": self.max_products,
            "cumulative_product_bytes": self.cumulative_products,
            "peak_ram_mb": self.peak_ram_mb,
        }

    def within(self) -> dict:
        self.main_cpu = time.process_time() - STARTED_CPU
        wall = time.perf_counter() - STARTED_WALL
        return {
            "cpu": self.total_cpu() <= self.cpu_cap,
            "wall": wall <= self.wall_cap,
            "adam": self.total_adam() <= self.adam_cap,
            "products": self.max_products <= self.product_cap,
            "limits": {
                "cpu_seconds": self.cpu_cap,
                "wall_seconds": self.wall_cap,
                "adam_calls": self.adam_cap,
                "product_bytes": self.product_cap,
            },
            "observed": {
                "cpu_seconds": self.total_cpu(),
                "wall_seconds": wall,
                "adam_calls": self.total_adam(),
                "max_live_product_bytes": self.max_products,
            },
        }


def example_fixture(example: FixtureExample, name: str) -> dict:
    expected = tuple(example.action.encode())
    if len(expected) != 27 or len(expected) + 1 != 28:
        raise InstrumentError("The composition fixture is not 27 bytes plus EOS")
    goal = example.goal.encode()
    if len(goal) != 36:
        raise InstrumentError("The composition goal is not 36 bytes")
    if not example.records:
        raise InstrumentError("A composition example has no records")
    return {
        "name": name,
        "index": example.index,
        "world": example.world_index,
        "layer": example.layer,
        "goal": example.goal,
        "goal_bytes": goal,
        "object_description": example.object_description,
        "place_description": example.place_description,
        "object_name": example.object_text,
        "place_name": example.place_text,
        "expected_text": example.action,
        "expected_bytes": expected,
        "target": expected + (EOS,),
        "records": example.records,
    }


def build_full_data() -> tuple:
    train_worlds = build_split("train")
    dev_worlds = build_split("dev")
    train = tuple(
        example_fixture(example, f"train-{world.world}-{example.index}")
        for world in train_worlds
        for example in world.examples
    )
    dev = tuple(
        example_fixture(example, f"dev-{world.world}-{example.index}")
        for world in dev_worlds
        for example in world.examples
    )
    if len(train) != EXPECTED_TRAIN or len(dev) != EXPECTED_DEV:
        raise InstrumentError(
            f"train/dev counts are {len(train)}/{len(dev)}, expected "
            f"{EXPECTED_TRAIN}/{EXPECTED_DEV}"
        )
    return train_worlds, dev_worlds, train, dev


def data_identity(train_worlds, dev_worlds, train, dev) -> dict:
    worlds = tuple(train_worlds) + tuple(dev_worlds)
    return {
        "schema": "taiji-composition-fixture-v1",
        "train_worlds": len(train_worlds),
        "dev_worlds": len(dev_worlds),
        "train_examples": len(train),
        "dev_examples": len(dev),
        "layers": {
            "train": {
                layer: sum(1 for row in train if row["layer"] == layer)
                for layer in ("seen_combination",)
            },
            "dev": {
                layer: sum(1 for row in dev if row["layer"] == layer)
                for layer in ("seen_combination", "unseen_combination", "unseen_entity")
            },
        },
        "digest": fixture_digest(worlds),
        "payload": fixture_payload(worlds),
        "confirm_read": False,
    }


def action_expected(object_name: str, place_name: str) -> bytes:
    value = f"把{object_name}放到{place_name}".encode()
    if len(value) != 27:
        raise InstrumentError(f"action is {len(value)} bytes, expected 27")
    return value


@dataclass(frozen=True)
class Case:
    """One declared stimulus: goal, records, expected bytes and the oracle name."""

    name: str
    seed: int
    world: int
    layer: str
    goal: bytes
    records: tuple[RawSource, ...]
    expected: bytes
    expected_text: str
    object_name: str
    source: str

    @property
    def goal_text(self) -> str:
        return self.goal.decode("utf-8")

    def row_base(self) -> dict:
        return {
            "name": self.name,
            "seed": self.seed,
            "world": self.world,
            "layer": self.layer,
            "source": self.source,
        }


def real_case(fixture: dict, seed: int) -> Case:
    return Case(
        name=fixture["name"],
        seed=seed,
        world=fixture["world"],
        layer=fixture["layer"],
        goal=fixture["goal_bytes"],
        records=fixture["records"],
        expected=fixture["expected_bytes"],
        expected_text=fixture["expected_text"],
        object_name=fixture["object_name"],
        source="real",
    )


def next_world(fixture: dict, dev_worlds) -> Any:
    world = dev_worlds[fixture["world"]]
    return dev_worlds[(world.world + 1) % len(dev_worlds)]


def permuted_case(fixture: dict, dev_worlds, seed: int) -> Case:
    nxt = next_world(fixture, dev_worlds)
    object_index = nxt.object_descriptions.index(fixture["object_description"])
    place_index = nxt.place_descriptions.index(fixture["place_description"])
    object_name = nxt.object_names[object_index]
    place_name = nxt.place_names[place_index]
    expected = action_expected(object_name, place_name)
    return Case(
        name=fixture["name"],
        seed=seed,
        world=fixture["world"],
        layer=fixture["layer"],
        goal=fixture["goal_bytes"],
        records=nxt.records,
        expected=expected,
        expected_text=f"把{object_name}放到{place_name}",
        object_name=object_name,
        source="permuted",
    )


def wrong_record_case(fixture: dict, dev_worlds, seed: int) -> Case:
    nxt = next_world(fixture, dev_worlds)
    return Case(
        name=fixture["name"],
        seed=seed,
        world=fixture["world"],
        layer=fixture["layer"],
        goal=fixture["goal_bytes"],
        records=nxt.records,
        expected=fixture["expected_bytes"],
        expected_text=fixture["expected_text"],
        object_name=fixture["object_name"],
        source="wrong_record",
    )


def no_source_case(fixture: dict, seed: int) -> Case:
    return Case(
        name=fixture["name"],
        seed=seed,
        world=fixture["world"],
        layer=fixture["layer"],
        goal=fixture["goal_bytes"],
        records=(),
        expected=fixture["expected_bytes"],
        expected_text=fixture["expected_text"],
        object_name=fixture["object_name"],
        source="no_source",
    )


def reverse_case(fixture: dict, seed: int) -> Case:
    return Case(
        name=fixture["name"],
        seed=seed,
        world=fixture["world"],
        layer=fixture["layer"],
        goal=fixture["goal_bytes"],
        records=tuple(reversed(fixture["records"])),
        expected=fixture["expected_bytes"],
        expected_text=fixture["expected_text"],
        object_name=fixture["object_name"],
        source="reverse",
    )


def correct_record_index(records: tuple[RawSource, ...], object_name: str) -> int | None:
    needle = object_name.encode()
    for index, record in enumerate(records):
        if needle in record.raw:
            return index
    return None


def wrong_record_index(records: tuple[RawSource, ...], object_name: str) -> int:
    needle = object_name.encode()
    for index, record in enumerate(records):
        if needle not in record.raw:
            return index
    raise InstrumentError("no record outside the correct object record")


def object_position(records: tuple[RawSource, ...], index: int, object_name: str) -> int:
    needle = object_name.encode()
    position = int(records[index].raw.index(needle))
    return position


def force_payload(
    candidate, records: tuple[RawSource, ...], index: int | None, offset: int
) -> dict:
    """Formal intervention entry for every arm, flat included.

    The 14.4 ``forced_read`` helper only builds the record/position force.  The
    flat arm consumes one position-level distribution, so the runner builds
    that declared force here instead of adding a second host oracle.
    """
    if index is None:
        if candidate.flat:
            total = sum(len(record.raw) for record in records)
            flat = torch.zeros(total + 1, dtype=torch.float64)
            flat[-1] = 1.0
            return {"flat": flat}
        alpha = torch.zeros(len(records) + 1, dtype=torch.float64)
        alpha[-1] = 1.0
        betas = tuple(torch.zeros(len(record.raw), dtype=torch.float64) for record in records)
        return {"alpha": alpha, "betas": betas}
    if candidate.flat:
        total = sum(len(record.raw) for record in records)
        flat = torch.zeros(total + 1, dtype=torch.float64)
        start = sum(len(record.raw) for record in records[:index])
        if offset < 0 or offset >= len(records[index].raw):
            raise InstrumentError("forced offset outside the record")
        flat[start + offset] = 1.0
        return {"flat": flat}
    alpha = torch.zeros(len(records) + 1, dtype=torch.float64)
    alpha[index] = 1.0
    betas = tuple(torch.zeros(len(record.raw), dtype=torch.float64) for record in records)
    if offset < 0 or offset >= len(records[index].raw):
        raise InstrumentError("forced offset outside the record")
    betas[index][offset] = 1.0
    return {"alpha": alpha, "betas": betas}


def first_source(read, view) -> tuple[int | None, str | None]:
    if view.record_count == 0:
        return None, None
    masses = read.alpha[: view.record_count] if read.alpha is not None else read.record_mass
    index = int(torch.argmax(masses))
    return index, view.source_ids[index]


def score_generation(
    raw: bytes,
    expected: bytes,
    terminated: bool,
    *,
    top_index: int | None,
    correct_index: int | None,
    position_correct: bool | None,
) -> dict:
    try:
        text = raw.decode("utf-8")
        utf8 = True
    except UnicodeDecodeError:
        text = "<invalid-utf8:" + raw.hex() + ">"
        utf8 = False
    full = raw == expected
    object_exact = len(raw) == 27 and raw[3:15] == expected[3:15]
    place_exact = len(raw) == 27 and raw[21:27] == expected[21:27]
    source_correct = (
        None if correct_index is None or top_index is None else top_index == correct_index
    )
    return {
        "generated_text": text,
        "utf8": utf8,
        "bytes": len(raw),
        "terminated": terminated,
        "completed": bool(full and terminated),
        "full_sentence": bool(full),
        "object_exact": bool(object_exact),
        "place_exact": bool(place_exact),
        "entity_exact": bool(object_exact and place_exact),
        "top_index": top_index,
        "correct_index": correct_index,
        "source_correct": source_correct,
        "position_correct": position_correct,
    }


def generate_trace(candidate, case: Case, *, tag: str) -> dict:
    """Free generation with the first declared read exposed for scoring."""
    with torch.no_grad():
        subject = candidate.subject
        state = subject.initial_state(
            f"14.6-{tag}-{case.seed}-{case.name}", revision=CONTENT_REVISION
        )
        prepared = prepare(candidate, state, case.goal, case.records)
        state, gate_index = observe_goal(candidate, prepared)
        coverage = prepared.coverage
        previous = EOS
        raw: list[int] = []
        terminated = False
        first_read = None
        for prediction in range(MAX_PREDICTIONS):
            step = answer_step(
                candidate,
                prepared,
                state,
                coverage,
                previous,
                gate_index=gate_index,
            )
            if prediction == 0:
                first_read = step.read
            endpoint = decode_endpoint(step.probabilities)
            if endpoint["eos"]:
                terminated = True
                break
            raw.append(endpoint["byte"])
            if len(raw) > MAX_OUTPUT_BYTES:
                break
            state = step.state
            coverage = step.coverage
            gate_index = step.gate_index
            previous = endpoint["byte"]
            if prediction == MAX_PREDICTIONS - 1:
                break
        top_index, top_source = first_source(first_read, prepared.view)
    return {
        "raw": bytes(raw),
        "terminated": terminated,
        "top_index": top_index,
        "top_source": top_source,
    }


def generate_forced_trace(
    candidate,
    case: Case,
    *,
    tag: str,
    record_index: int,
    offset: int,
    gamma: float,
) -> dict:
    """One declared forced first read, then ordinary free generation."""
    with torch.no_grad():
        subject = candidate.subject
        state = subject.initial_state(
            f"14.6-{tag}-{case.seed}-{case.name}", revision=CONTENT_REVISION
        )
        prepared = prepare(candidate, state, case.goal, case.records)
        state, gate_index = observe_goal(candidate, prepared)
        coverage = prepared.coverage
        force = force_payload(candidate, case.records, record_index, offset)
        step = answer_step(
            candidate,
            prepared,
            state,
            coverage,
            EOS,
            gate_index=gate_index,
            force=force,
            gamma_override=gamma,
        )
        raw: list[int] = []
        terminated = False
        endpoint = decode_endpoint(step.probabilities)
        if endpoint["eos"]:
            terminated = True
        else:
            raw.append(endpoint["byte"])
            state = step.state
            coverage = step.coverage
            gate_index = step.gate_index
            previous = endpoint["byte"]
            for prediction in range(1, MAX_PREDICTIONS):
                step = answer_step(
                    candidate,
                    prepared,
                    state,
                    coverage,
                    previous,
                    gate_index=gate_index,
                )
                endpoint = decode_endpoint(step.probabilities)
                if endpoint["eos"]:
                    terminated = True
                    break
                raw.append(endpoint["byte"])
                if len(raw) > MAX_OUTPUT_BYTES:
                    break
                state = step.state
                coverage = step.coverage
                gate_index = step.gate_index
                previous = endpoint["byte"]
                if prediction == MAX_PREDICTIONS - 1:
                    break
        top_source = case.records[record_index].source_id if record_index is not None else None
    return {
        "raw": bytes(raw),
        "terminated": terminated,
        "top_index": record_index,
        "top_source": top_source,
        "forced_position": offset,
    }


def run_case(
    candidate,
    case: Case,
    *,
    tag: str,
    record_index: int | None = None,
    offset: int | None = None,
    gamma: float | None = None,
) -> dict:
    if record_index is None:
        trace = generate_trace(candidate, case, tag=tag)
        position_correct = None
    else:
        if offset is None or gamma is None:
            raise InstrumentError("forced cases need an offset and a gamma value")
        trace = generate_forced_trace(
            candidate,
            case,
            tag=tag,
            record_index=record_index,
            offset=offset,
            gamma=gamma,
        )
        correct = correct_record_index(case.records, case.object_name)
        position_correct = (
            None
            if correct is None
            else bool(
                correct == record_index
                and offset == object_position(case.records, correct, case.object_name)
            )
        )
    correct_index = correct_record_index(case.records, case.object_name)
    row = score_generation(
        trace["raw"],
        case.expected,
        bool(trace["terminated"]),
        top_index=trace["top_index"],
        correct_index=correct_index,
        position_correct=position_correct,
    )
    row.update(case.row_base())
    row.update(
        {
            "goal": case.goal_text,
            "expected": case.expected_text,
            "records": [record.source_id for record in case.records],
            "top_source": trace["top_source"],
            "forced": record_index is not None,
            "forced_record": record_index,
            "forced_offset": offset,
            "gamma": gamma,
        }
    )
    return row


def summarize_rows(rows: list[dict]) -> dict:
    if not rows:
        return {
            "n": 0,
            "completed": 0.0,
            "full_sentence": 0.0,
            "entity_exact": 0.0,
            "eos": 0.0,
            "utf8": 0.0,
            "source_top1": None,
            "layers": {},
        }
    source_rows = [row for row in rows if row.get("source_correct") is not None]

    def rate(key: str) -> float:
        return sum(1 for row in rows if row.get(key)) / len(rows)

    layers = {}
    for layer in ("seen_combination", "unseen_combination", "unseen_entity"):
        selected = [row for row in rows if row["layer"] == layer]
        if selected:
            layers[layer] = {
                "n": len(selected),
                "completed": sum(1 for row in selected if row["completed"]) / len(selected),
            }
    return {
        "n": len(rows),
        "completed": rate("completed"),
        "full_sentence": rate("full_sentence"),
        "entity_exact": rate("entity_exact"),
        "eos": rate("terminated"),
        "utf8": rate("utf8"),
        "source_top1": (
            None
            if not source_rows
            else sum(1 for row in source_rows if row["source_correct"]) / len(source_rows)
        ),
        "layers": layers,
    }


def accuracy(rows: list[dict], key: str = "completed") -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row[key]) / len(rows)


def mean_nll(rows: list[dict]) -> float:
    if not rows:
        return float("nan")
    return float(statistics.fmean(row["nll"] for row in rows))


def mean_float(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(statistics.fmean(values))


def bootstrap_lower(diffs: list[float], *, seed: int, rounds: int) -> dict:
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(rounds):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return {
        "rounds": rounds,
        "seed": seed,
        "mean": statistics.fmean(diffs),
        "lower_95": means[int(0.025 * rounds)],
        "upper_95": means[int(0.975 * rounds) - 1],
    }


def evaluate_real(candidate, fixtures: tuple[dict, ...], *, seed: int, arm: str, tag: str) -> dict:
    rows = []
    nll_rows = []
    start = time.process_time()
    for index, fixture in enumerate(fixtures):
        case = real_case(fixture, seed)
        row = run_case(candidate, case, tag=f"{tag}-real")
        rows.append(row)
        with torch.no_grad():
            nll = score_example(
                candidate,
                case.goal,
                case.records,
                tuple(fixture["target"][:-1]),
                subject_id=f"14.6-{tag}-{seed}-{arm}-{fixture['name']}",
            ).nll
        nll_rows.append({"name": fixture["name"], "layer": fixture["layer"], "nll": float(nll)})
        if index % 8 == 7:
            STAGE_BUDGET.ensure()
    return {
        "cpu_seconds": time.process_time() - start,
        "rows": rows,
        "summary": summarize_rows(rows),
        "nll": {
            "mean": mean_nll(nll_rows),
            "rows": nll_rows,
            "by_layer": {
                layer: mean_nll([row for row in nll_rows if row["layer"] == layer])
                for layer in ("seen_combination", "unseen_combination", "unseen_entity")
            },
        },
    }


def evaluate_cases(
    candidate,
    cases: tuple[Case, ...],
    *,
    tag: str,
) -> dict:
    rows = []
    start = time.process_time()
    for index, case in enumerate(cases):
        rows.append(run_case(candidate, case, tag=f"{tag}-{case.source}"))
        if index % 8 == 7:
            STAGE_BUDGET.ensure()
    return {
        "cpu_seconds": time.process_time() - start,
        "rows": rows,
        "summary": summarize_rows(rows),
    }


def evaluate_structural(
    candidate,
    fixed_fixtures: tuple[dict, ...],
    dev_worlds,
    *,
    seed: int,
    arm: str,
) -> dict:
    rows: dict[str, list[dict]] = {}
    start = time.process_time()
    for fixture in fixed_fixtures:
        real = real_case(fixture, seed)
        no_source = no_source_case(fixture, seed)
        wrong = wrong_record_case(fixture, dev_worlds, seed)
        correct = correct_record_index(real.records, real.object_name)
        if correct is None:
            raise InstrumentError("the correct object record is missing")
        rescue_position = object_position(real.records, correct, real.object_name)
        rows.setdefault("no_source", []).append(
            run_case(candidate, no_source, tag=f"14.6-{arm}-no-source")
        )
        rows.setdefault("wrong_record", []).append(
            run_case(candidate, wrong, tag=f"14.6-{arm}-wrong-record")
        )
        rows.setdefault("wrong_position", []).append(
            run_case(
                candidate,
                replace(real, source="wrong_position"),
                tag=f"14.6-{arm}-wrong-position",
                record_index=correct,
                offset=0,
                gamma=0.0,
            )
        )
        rows.setdefault("rescue", []).append(
            run_case(
                candidate,
                replace(real, source="rescue"),
                tag=f"14.6-{arm}-rescue",
                record_index=correct,
                offset=rescue_position,
                gamma=1.0,
            )
        )
    return {
        "cpu_seconds": time.process_time() - start,
        "conditions": {
            name: {"rows": rows[name], "summary": summarize_rows(rows[name])}
            for name in ("no_source", "wrong_record", "wrong_position", "rescue")
        },
    }


def spawn_worker(spec: dict) -> dict:
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=run_worker, args=(spec,))
    process.start()
    process.join(timeout=WORKER_TIMEOUT)
    if process.is_alive():
        process.terminate()
        process.join()
        return {
            "ok": False,
            "message": "worker timed out",
            "cpu_seconds": 0.0,
            "wall_seconds": 0.0,
        }
    if process.exitcode != 0:
        return {
            "ok": False,
            "exitcode": process.exitcode,
            "message": "worker did not exit cleanly",
            "cpu_seconds": 0.0,
            "wall_seconds": 0.0,
        }
    path = Path(spec["result"])
    if not path.exists():
        return {
            "ok": False,
            "message": "worker wrote no result",
            "cpu_seconds": 0.0,
            "wall_seconds": 0.0,
        }
    loaded: dict = torch.load(path, map_location="cpu", weights_only=True)
    return loaded


def worker_spec(bundle: Path, result: Path, fixture: dict, *, gate_index: int) -> dict:
    return {
        "bundle": str(bundle),
        "result": str(result),
        "fixtures": (
            {
                "goal": fixture["goal"],
                "records": fixture["records"],
                "target_bytes": fixture["target"],
            },
        ),
        "dtype": "float64",
        "mode": "probe",
        "previous_byte": EOS,
        "fixture_index": 0,
        "force": None,
        "gamma_override": None,
        "gate_index": gate_index,
    }


def probe_state(candidate, fixture: dict, *, subject_id: str) -> tuple:
    with torch.no_grad():
        state = candidate.subject.initial_state(subject_id, revision=CONTENT_REVISION)
        prepared = prepare(candidate, state, fixture["goal_bytes"], fixture["records"])
        observation, gate_index = observe_goal(candidate, prepared)
        probe = answer_step(
            candidate,
            prepared,
            observation,
            observation.coverage,
            EOS,
            gate_index=gate_index,
        )
    return observation, gate_index, probe


def load_bundle_guard(path: Path) -> tuple:
    rng = torch.get_rng_state()
    try:
        with torch.no_grad():
            return load_bundle(path, torch.float64)
    finally:
        torch.set_rng_state(rng)


def relation_digest(candidate) -> str:
    return str(
        parameter_digest(
            {
                name: parameter.detach().cpu()
                for name, parameter in candidate.relation.named_parameters()
            }
        )
    )


def save_checkpoint(
    path: Path,
    *,
    candidate,
    optimizer,
    fixture: dict,
    seed: int,
    arm: str,
    order_seed: int,
    update_index: int,
    keep: bool,
    worker: bool,
) -> dict:
    observation, gate_index, probe = probe_state(
        candidate,
        fixture,
        subject_id=f"14.6-probe-{seed}-{arm}-{update_index}",
    )
    current_id = candidate_digest(candidate)
    info = save_bundle(
        path,
        profile=f"14.6-{seed}-{arm}-step{update_index}",
        candidate=candidate,
        state=observation,
        optimizer=optimizer,
        update_index=update_index,
        cursor={
            "seed": seed,
            "arm": arm,
            "order_seed": order_seed,
            "step": "14.6",
        },
        parent_identity={
            "parent_candidate": EXPECTED_PARENT,
            "seed": seed,
            "arm": arm,
            "order_seed": order_seed,
            "step": update_index,
            "candidate_at_save": current_id,
        },
    )
    STAGE_BUDGET.add_product(int(info["bytes"]))
    row: dict[str, Any] = {
        "seed": seed,
        "arm": arm,
        "update_index": update_index,
        "path": relative(path),
        "sha256": info["sha256"],
        "bytes": int(info["bytes"]),
        "candidate_at_save": current_id,
        "kept": keep,
        "probe_gate_index": gate_index,
    }
    loaded = load_bundle_guard(path)
    loaded_candidate = loaded[0]
    loaded_digest = candidate_digest(loaded_candidate)
    row["reload"] = {
        "ok": loaded_digest == current_id,
        "loaded_digest": loaded_digest,
        "update_index": int(loaded[5]),
        "arm": loaded[3],
    }
    del loaded_candidate
    if worker:
        result_path = path.with_name(path.stem + "_probe.pt")
        worker_result = spawn_worker(worker_spec(path, result_path, fixture, gate_index=gate_index))
        STAGE_BUDGET.worker_cpu += float(worker_result.get("cpu_seconds", 0.0))
        STAGE_BUDGET.worker_wall += float(worker_result.get("wall_seconds", 0.0))
        equal_prob = bool(worker_result.get("ok")) and bool(
            torch.equal(probe.probabilities, worker_result["probabilities"])
        )
        equal_h = bool(worker_result.get("ok")) and bool(
            torch.equal(probe.state.h, worker_result["h"])
        )
        equal_cov = bool(worker_result.get("ok")) and bool(
            torch.equal(probe.coverage, worker_result["coverage"])
        )
        row["worker_restore"] = {
            "ok": bool(worker_result.get("ok")),
            "equal_probabilities": equal_prob,
            "equal_h": equal_h,
            "equal_coverage": equal_cov,
            "max_abs_diff": (
                None
                if not worker_result.get("ok")
                else max_abs_diff(probe.probabilities, worker_result["probabilities"])
            ),
            "cpu_seconds": float(worker_result.get("cpu_seconds", 0.0)),
            "wall_seconds": float(worker_result.get("wall_seconds", 0.0)),
        }
        if result_path.exists():
            result_path.unlink()
    if not keep:
        STAGE_BUDGET.drop_product(int(info["bytes"]))
        path.unlink()
        row["deleted"] = True
    return row


def batch_plan(length: int, order_seed: int, updates: int) -> tuple:
    rng = random.Random(order_seed)
    order = list(range(length))
    batches: list[tuple[int, ...]] = []
    while len(batches) < updates:
        rng.shuffle(order)
        for start in range(0, length, BATCH):
            block = tuple(order[start : start + BATCH])
            if len(block) != BATCH:
                continue
            batches.append(block)
            if len(batches) == updates:
                break
    if len(batches) != updates:
        raise InstrumentError("batch plan is shorter than the declared updates")
    return tuple(batches)


def batch_loss(candidate, train: tuple[dict, ...], batch: tuple[int, ...]) -> torch.Tensor:
    losses = [
        score_example(
            candidate,
            train[index]["goal_bytes"],
            train[index]["records"],
            tuple(train[index]["target"][:-1]),
        ).nll
        for index in batch
    ]
    return sum(losses) / len(losses)


def train_candidate(
    candidate,
    optimizer,
    train: tuple[dict, ...],
    plan: tuple,
    *,
    seed: int,
    arm: str,
    order_seed: int,
    output_dir: Path,
    fixture: dict,
    checkpoints: tuple[int, ...],
    keep: tuple[int, ...],
    worker_steps: tuple[int, ...],
    progress_path: Path,
    log_path: Path,
) -> dict:
    rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    start = time.process_time()
    for index, batch in enumerate(plan, start=1):
        forward_start = time.process_time()
        loss = batch_loss(candidate, train, batch)
        forward_cpu = time.process_time() - forward_start
        if not bool(torch.isfinite(loss)):
            raise InstrumentError(f"non-finite loss at {seed}/{arm}/step {index}")
        bptt_start = time.process_time()
        optimizer.zero_grad()
        loss.backward()
        bptt_cpu = time.process_time() - bptt_start
        step_start = time.process_time()
        clip = float(torch.nn.utils.clip_grad_norm_(candidate.trainable_parameters(), CLIP_NORM))
        if not math.isfinite(clip):
            raise InstrumentError(f"non-finite clip at {seed}/{arm}/step {index}")
        optimizer.step()
        step_cpu = time.process_time() - step_start
        STAGE_BUDGET.adam_calls += 1
        row = {
            "index": index,
            "loss": float(loss.detach()),
            "clip_scale": clip,
            "forward_cpu": forward_cpu,
            "bptt_cpu": bptt_cpu,
            "step_cpu": step_cpu,
            "total_cpu": forward_cpu + bptt_cpu + step_cpu,
            "batch": list(batch),
            "batch_names": [train[position]["name"] for position in batch],
        }
        rows.append(row)
        if index in checkpoints and index != 0:
            checkpoint_rows.append(
                save_checkpoint(
                    output_dir / f"{seed}_{arm}_step{index}.pt",
                    candidate=candidate,
                    optimizer=optimizer,
                    fixture=fixture,
                    seed=seed,
                    arm=arm,
                    order_seed=order_seed,
                    update_index=index,
                    keep=index in keep,
                    worker=index in worker_steps,
                )
            )
        if index % 16 == 0 or index == len(plan):
            STAGE_BUDGET.ensure()
            STAGE_BUDGET.sample_ram()
            write_progress(
                progress_path,
                {
                    "phase": "train",
                    "seed": seed,
                    "arm": arm,
                    "update_index": index,
                    "updates": len(plan),
                    "last_loss": float(loss.detach()),
                    "cost": STAGE_BUDGET.cost(),
                },
            )
            append_log(
                log_path,
                {
                    "event": "train",
                    "seed": seed,
                    "arm": arm,
                    "update_index": index,
                    "loss": float(loss.detach()),
                    "cpu": STAGE_BUDGET.total_cpu(),
                    "wall": time.perf_counter() - STARTED_WALL,
                },
            )
    summary = {
        "updates": len(rows),
        "adam_calls": len(rows),
        "cpu_seconds": time.process_time() - start,
        "first_loss": rows[0]["loss"] if rows else None,
        "last_loss": rows[-1]["loss"] if rows else None,
        "mean_first_16": (
            statistics.fmean(float(row["loss"]) for row in rows[:16]) if rows else None
        ),
        "mean_last_16": (
            statistics.fmean(float(row["loss"]) for row in rows[-16:]) if rows else None
        ),
        "batch_digest": hashlib.sha256(repr(plan).encode()).hexdigest(),
    }
    return {"rows": rows, "checkpoints": checkpoint_rows, "summary": summary}


STAGE_BUDGET: StageBudget


def adopt_checkpoint(
    path: Path,
    *,
    candidate,
    fixture: dict,
    seed: int,
    arm: str,
    order_seed: int,
    update_index: int,
) -> dict:
    """Adopt a checkpoint written by a prior attempt after proving its identity."""
    size = path.stat().st_size
    STAGE_BUDGET.add_product(size)
    observation, gate_index, probe = probe_state(
        candidate,
        fixture,
        subject_id=f"14.6-adopt-{seed}-{arm}-{update_index}",
    )
    current_id = candidate_digest(candidate)
    loaded = load_bundle_guard(path)
    loaded_candidate, state, optimizer, loaded_arm, profile, loaded_index, cursor, payload = loaded
    _ = (optimizer, profile)
    if int(loaded_index) != update_index:
        raise InstrumentError("the adopted checkpoint has the wrong update index")
    if payload.get("content_schema") != RECORD_RELATION_SCHEMA:
        raise InstrumentError("the adopted checkpoint has the wrong content schema")
    if payload.get("relation_config", {}).get("seed") != seed:
        raise InstrumentError("the adopted checkpoint has the wrong relation seed")
    if payload.get("parent", {}).get("parent_candidate") != EXPECTED_PARENT:
        raise InstrumentError("the adopted checkpoint has the wrong parent")
    if payload["cursor"].get("seed") != seed or payload["cursor"].get("arm") != arm:
        raise InstrumentError("the adopted checkpoint has the wrong cursor")
    if payload["cursor"].get("order_seed") != order_seed:
        raise InstrumentError("the adopted checkpoint has the wrong order seed")
    if payload["arm"] != arm or loaded_arm != arm:
        raise InstrumentError("the adopted checkpoint has the wrong arm field")
    loaded_id = candidate_digest(loaded_candidate)
    if loaded_id != current_id or loaded_id != payload["parent"]["candidate_at_save"]:
        raise InstrumentError("the adopted checkpoint parameter identity drifted")
    state_reproduced = bool(
        torch.equal(observation.h, state.h)
        and torch.equal(observation.coverage, state.coverage)
        and int(state.step) == int(observation.step)
    )
    del loaded_candidate
    result_path = path.with_name(path.stem + "_adopt_probe.pt")
    worker_result = spawn_worker(worker_spec(path, result_path, fixture, gate_index=gate_index))
    STAGE_BUDGET.worker_cpu += float(worker_result.get("cpu_seconds", 0.0))
    STAGE_BUDGET.worker_wall += float(worker_result.get("wall_seconds", 0.0))
    equal_prob = bool(worker_result.get("ok")) and bool(
        torch.equal(probe.probabilities, worker_result["probabilities"])
    )
    equal_h = bool(worker_result.get("ok")) and bool(torch.equal(probe.state.h, worker_result["h"]))
    equal_cov = bool(worker_result.get("ok")) and bool(
        torch.equal(probe.coverage, worker_result["coverage"])
    )
    if result_path.exists():
        result_path.unlink()
    return {
        "seed": seed,
        "arm": arm,
        "update_index": update_index,
        "path": relative(path),
        "sha256": sha256_text(path),
        "bytes": size,
        "candidate_at_save": loaded_id,
        "kept": True,
        "adopted": True,
        "probe_gate_index": gate_index,
        "reload": {
            "ok": loaded_id == current_id,
            "loaded_digest": loaded_id,
            "update_index": int(loaded_index),
            "arm": loaded_arm,
        },
        "state_reproduced": state_reproduced,
        "worker_restore": {
            "ok": bool(worker_result.get("ok")),
            "equal_probabilities": equal_prob,
            "equal_h": equal_h,
            "equal_coverage": equal_cov,
            "max_abs_diff": (
                None
                if not worker_result.get("ok")
                else max_abs_diff(probe.probabilities, worker_result["probabilities"])
            ),
            "cpu_seconds": float(worker_result.get("cpu_seconds", 0.0)),
            "wall_seconds": float(worker_result.get("wall_seconds", 0.0)),
        },
    }


def run_candidate(
    parent: dict,
    *,
    seed: int,
    arm: str,
    order_seed: int,
    train: tuple[dict, ...],
    dev: tuple[dict, ...],
    dev_worlds,
    fixed_fixtures: tuple[dict, ...],
    plan: tuple,
    output_dir: Path,
    progress_path: Path,
    log_path: Path,
    checkpoints: tuple[int, ...],
    keep: tuple[int, ...],
    expected_seed_row: dict,
) -> dict:
    config = RecordRelationConfig(seed=seed)
    final_path = output_dir / f"{seed}_{arm}_step{max(checkpoints)}.pt"
    step0_path = output_dir / f"{seed}_{arm}_step{min(checkpoints)}.pt"
    row: dict[str, Any] = {
        "seed": seed,
        "arm": arm,
        "order_seed": order_seed,
        "status": "running",
        "candidate_initial": None,
        "candidate_final": None,
        "relation_initial": None,
        "relation_final": None,
        "parameter_count": None,
        "trainable_count": None,
        "disabled": None,
        "training": {},
        "step0": {},
        "step300": {},
        "checkpoints": [],
        "resumed_from_final": final_path.exists(),
    }
    initial = build_candidate(parent, arm=arm, config=config, dtype=torch.float64)
    initial_id = candidate_digest(initial)
    if initial_id != expected_seed_row["candidate_id"]:
        raise InstrumentError(
            f"{seed}/{arm} initial candidate {initial_id} drifted from the preflight identity"
        )
    initial_relation = relation_digest(initial)
    if initial_relation != expected_seed_row["relation_digest"]:
        raise InstrumentError(f"{seed}/{arm} relation digest drifted from the preflight identity")
    row["candidate_initial"] = initial_id
    row["relation_initial"] = initial_relation
    row["parameter_count"] = initial.parameter_count()
    row["trainable_count"] = int(
        sum(parameter.numel() for parameter in initial.trainable_parameters())
    )
    row["disabled"] = initial.disabled_rows()

    if final_path.exists():
        final_candidate, state, optimizer, loaded_arm, profile, update_index, cursor, payload = (
            load_bundle_guard(final_path)
        )
        _ = (state, optimizer, profile, cursor)
        if int(update_index) != max(checkpoints):
            raise InstrumentError("the retained final checkpoint has the wrong update index")
        if payload.get("content_schema") != RECORD_RELATION_SCHEMA:
            raise InstrumentError("the retained final checkpoint has the wrong content schema")
        if payload.get("relation_config", {}).get("seed") != seed:
            raise InstrumentError("the retained final checkpoint has the wrong relation seed")
        if payload.get("parent", {}).get("parent_candidate") != EXPECTED_PARENT:
            raise InstrumentError("the retained final checkpoint has the wrong parent")
        if payload["cursor"].get("seed") != seed or payload["cursor"].get("arm") != arm:
            raise InstrumentError("the retained final checkpoint has the wrong cursor")
        if payload["cursor"].get("order_seed") != order_seed:
            raise InstrumentError("the retained final checkpoint has the wrong order seed")
        if payload["arm"] != arm or loaded_arm != arm:
            raise InstrumentError("the retained final checkpoint has the wrong arm field")
        final_id = candidate_digest(final_candidate)
        if final_id != payload["parent"]["candidate_at_save"]:
            raise InstrumentError("the retained final checkpoint parameter identity drifted")
        row["candidate_final"] = final_id
        row["relation_final"] = relation_digest(final_candidate)
        row["training"] = {
            "recovered": True,
            "updates": int(update_index),
            "adam_calls": int(update_index),
            "summary": None,
            "rows": None,
            "note": "training was completed by a prior attempt of this stage",
        }
        STAGE_BUDGET.recovered_adam_calls += int(update_index)
        for intermediate in checkpoints:
            if intermediate not in (min(checkpoints), max(checkpoints)):
                stale = output_dir / f"{seed}_{arm}_step{intermediate}.pt"
                if stale.exists():
                    stale.unlink()
        row["checkpoints"].append(
            adopt_checkpoint(
                final_path,
                candidate=final_candidate,
                fixture=dev[0],
                seed=seed,
                arm=arm,
                order_seed=order_seed,
                update_index=max(checkpoints),
            )
        )
        if step0_path.exists():
            row["checkpoints"].append(
                adopt_checkpoint(
                    step0_path,
                    candidate=initial,
                    fixture=dev[0],
                    seed=seed,
                    arm=arm,
                    order_seed=order_seed,
                    update_index=min(checkpoints),
                )
            )
        else:
            row["checkpoints"].append(
                save_checkpoint(
                    step0_path,
                    candidate=initial,
                    optimizer=make_optimizer(initial),
                    fixture=dev[0],
                    seed=seed,
                    arm=arm,
                    order_seed=order_seed,
                    update_index=min(checkpoints),
                    keep=min(checkpoints) in keep,
                    worker=True,
                )
            )
        row["step0"] = evaluate_real(initial, dev, seed=seed, arm=arm, tag=f"{seed}-{arm}-step0")
        del initial
        row["step300"] = {
            "real": evaluate_real(
                final_candidate, dev, seed=seed, arm=arm, tag=f"{seed}-{arm}-step300"
            ),
            "permuted": evaluate_cases(
                final_candidate,
                tuple(permuted_case(fixture, dev_worlds, seed) for fixture in dev),
                tag=f"{seed}-{arm}-permuted",
            ),
            "reverse": evaluate_cases(
                final_candidate,
                tuple(reverse_case(fixture, seed) for fixture in dev),
                tag=f"{seed}-{arm}-reverse",
            ),
            "structural": evaluate_structural(
                final_candidate, fixed_fixtures, dev_worlds, seed=seed, arm=arm
            ),
        }
        del final_candidate
        row["status"] = "resumed"
        return row

    for index in checkpoints:
        stale = output_dir / f"{seed}_{arm}_step{index}.pt"
        if stale.exists():
            stale.unlink()
    optimizer = make_optimizer(initial)
    checkpoint0 = save_checkpoint(
        step0_path,
        candidate=initial,
        optimizer=optimizer,
        fixture=dev[0],
        seed=seed,
        arm=arm,
        order_seed=order_seed,
        update_index=min(checkpoints),
        keep=min(checkpoints) in keep,
        worker=True,
    )
    row["checkpoints"].append(checkpoint0)
    row["step0"] = evaluate_real(initial, dev, seed=seed, arm=arm, tag=f"{seed}-{arm}-step0")
    training = train_candidate(
        initial,
        optimizer,
        train,
        plan,
        seed=seed,
        arm=arm,
        order_seed=order_seed,
        output_dir=output_dir,
        fixture=dev[0],
        checkpoints=checkpoints,
        keep=keep,
        worker_steps=tuple(index for index in checkpoints if index in keep and index != 0),
        progress_path=progress_path,
        log_path=log_path,
    )
    row["training"] = {
        "recovered": False,
        "updates": training["summary"]["updates"],
        "adam_calls": training["summary"]["adam_calls"],
        "summary": training["summary"],
        "rows": training["rows"],
    }
    row["checkpoints"].extend(training["checkpoints"])
    row["candidate_final"] = candidate_digest(initial)
    row["relation_final"] = relation_digest(initial)
    row["step300"] = {
        "real": evaluate_real(initial, dev, seed=seed, arm=arm, tag=f"{seed}-{arm}-step300"),
        "permuted": evaluate_cases(
            initial,
            tuple(permuted_case(fixture, dev_worlds, seed) for fixture in dev),
            tag=f"{seed}-{arm}-permuted",
        ),
        "reverse": evaluate_cases(
            initial,
            tuple(reverse_case(fixture, seed) for fixture in dev),
            tag=f"{seed}-{arm}-reverse",
        ),
        "structural": evaluate_structural(initial, fixed_fixtures, dev_worlds, seed=seed, arm=arm),
    }
    if row["candidate_final"] != candidate_digest(initial):
        raise InstrumentError("the final candidate identity changed during evaluation")
    row["status"] = "complete"
    return row


def collect_rows(
    candidates: list[dict],
    *,
    phase: str,
    condition: str,
    arm: str | None = None,
    seed: int | None = None,
) -> list[dict]:
    rows = []
    for candidate in candidates:
        if arm is not None and candidate["arm"] != arm:
            continue
        if seed is not None and candidate["seed"] != seed:
            continue
        section = candidate.get(phase, {})
        if condition == "structural":
            for name in ("no_source", "wrong_record", "wrong_position", "rescue"):
                rows.extend(
                    section.get("structural", {})
                    .get("conditions", {})
                    .get(name, {})
                    .get("rows", [])
                )
        else:
            rows.extend(section.get(condition, {}).get("rows", []))
    return rows


def gate_task(candidates: list[dict]) -> dict:
    real = collect_rows(candidates, phase="step300", condition="real", arm=ARM_FULL)
    permuted = collect_rows(candidates, phase="step300", condition="permuted", arm=ARM_FULL)
    reverse = collect_rows(candidates, phase="step300", condition="reverse", arm=ARM_FULL)
    total = accuracy(real)
    unseen_combination = accuracy([row for row in real if row["layer"] == "unseen_combination"])
    unseen_entity = accuracy([row for row in real if row["layer"] == "unseen_entity"])
    real_by_key = {(row["seed"], row["name"]): row for row in real}
    perm_by_key = {(row["seed"], row["name"]): row for row in permuted}
    keys = sorted(set(real_by_key) & set(perm_by_key))
    both = [real_by_key[key]["completed"] and perm_by_key[key]["completed"] for key in keys]
    both_sides = sum(1 for value in both if value) / len(both) if both else 0.0
    reverse_rate = accuracy(reverse)
    passed = (
        total >= TASK_TOTAL_MIN
        and unseen_combination >= TASK_LAYER_MIN
        and unseen_entity >= TASK_LAYER_MIN
        and both_sides >= BOTH_SIDES_MIN
        and reverse_rate >= REVERSE_MIN
        and len(real) == 320
        and len(permuted) == 320
        and len(reverse) == 320
        and len(keys) == 320
    )
    return {
        "passed": passed,
        "bars": {
            "total": TASK_TOTAL_MIN,
            "unseen_combination": TASK_LAYER_MIN,
            "unseen_entity": TASK_LAYER_MIN,
            "both_sides": BOTH_SIDES_MIN,
            "reverse": REVERSE_MIN,
        },
        "observed": {
            "total": total,
            "total_n": len(real),
            "unseen_combination": unseen_combination,
            "unseen_entity": unseen_entity,
            "both_sides": both_sides,
            "both_sides_n": len(both),
            "reverse": reverse_rate,
            "reverse_n": len(reverse),
        },
    }


def gate_structure(candidates: list[dict]) -> dict:
    rows = []
    for seed in SEEDS:
        a_rows = collect_rows(
            candidates, phase="step300", condition="real", arm=ARM_FULL, seed=seed
        )
        b_rows = collect_rows(
            candidates, phase="step300", condition="real", arm=ARM_LOCAL, seed=seed
        )
        c_rows = collect_rows(
            candidates, phase="step300", condition="real", arm=ARM_FLAT, seed=seed
        )
        a_acc = accuracy(a_rows)
        b_acc = accuracy(b_rows)
        c_acc = accuracy(c_rows)
        saturated = a_acc >= 0.95 and b_acc >= 0.95 and c_acc >= 0.95
        worlds = sorted({row["world"] for row in a_rows})
        complete_worlds = worlds == list(range(8))
        for name, other, other_acc in (
            ("A_minus_B", ARM_LOCAL, b_acc),
            ("A_minus_C", ARM_FLAT, c_acc),
        ):
            diffs = []
            for world in worlds:
                a_world = [row for row in a_rows if row["world"] == world]
                other_world = [
                    row
                    for row in (b_rows if other == ARM_LOCAL else c_rows)
                    if row["world"] == world
                ]
                diffs.append(accuracy(a_world) - accuracy(other_world))
            bootstrap = (
                bootstrap_lower(diffs, seed=BOOTSTRAP_SEED, rounds=BOOTSTRAP_ROUNDS)
                if diffs
                else {
                    "rounds": 0,
                    "seed": BOOTSTRAP_SEED,
                    "mean": 0.0,
                    "lower_95": 0.0,
                    "upper_95": 0.0,
                }
            )
            gain = a_acc - other_acc
            passed = complete_worlds and gain >= STRUCTURE_GAIN_MIN and bootstrap["lower_95"] > 0.0
            rows.append(
                {
                    "seed": seed,
                    "comparison": name,
                    "gain": gain,
                    "bootstrap": bootstrap,
                    "passed": bool(passed),
                    "saturated": bool(saturated),
                    "complete_worlds": bool(complete_worlds),
                    "a_accuracy": a_acc,
                    "other_accuracy": other_acc,
                    "world_differences": diffs,
                }
            )
    passed = all(row["passed"] for row in rows) and not any(row["saturated"] for row in rows)
    return {
        "passed": passed,
        "bar": STRUCTURE_GAIN_MIN,
        "rows": rows,
        "reason": (
            "saturated_unresolved" if any(row["saturated"] for row in rows) and not passed else None
        ),
    }


def gate_content(candidates: list[dict], fixed_names: tuple[str, ...]) -> dict:
    real = [
        row
        for row in collect_rows(candidates, phase="step300", condition="real", arm=ARM_FULL)
        if row["name"] in fixed_names
    ]
    conditions = {}
    for name in ("no_source", "wrong_record", "wrong_position"):
        rows = [
            row
            for row in collect_rows(
                candidates, phase="step300", condition="structural", arm=ARM_FULL
            )
            if row["name"] in fixed_names and row["source"] == name
        ]
        real_acc = accuracy(real)
        condition_acc = accuracy(rows)
        drop = real_acc - condition_acc
        conditions[name] = {
            "n": len(rows),
            "real_accuracy": real_acc,
            "condition_accuracy": condition_acc,
            "drop": drop,
            "passed": drop >= CONTENT_DROP_MIN,
        }
    rescue = [
        row
        for row in collect_rows(candidates, phase="step300", condition="structural", arm=ARM_FULL)
        if row["name"] in fixed_names and row["source"] == "rescue"
    ]
    passed = all(condition["passed"] for condition in conditions.values())
    return {
        "passed": passed,
        "bar": CONTENT_DROP_MIN,
        "conditions": conditions,
        "rescue": {
            "n": len(rescue),
            "real_accuracy": accuracy(real),
            "rescue_accuracy": accuracy(rescue),
            "delta": accuracy(rescue) - accuracy(real),
        },
        "fixed_names": list(fixed_names),
    }


def gate_retention(candidates: list[dict]) -> dict:
    step0 = collect_rows(candidates, phase="step0", condition="real", arm=ARM_FULL)
    step300 = collect_rows(candidates, phase="step300", condition="real", arm=ARM_FULL)
    step0_nll = [
        item["nll"]
        for candidate in candidates
        if candidate["arm"] == ARM_FULL
        for item in candidate.get("step0", {}).get("real", {}).get("nll", {}).get("rows", [])
    ]
    step300_nll = [
        item["nll"]
        for candidate in candidates
        if candidate["arm"] == ARM_FULL
        for item in candidate.get("step300", {}).get("real", {}).get("nll", {}).get("rows", [])
    ]
    return {
        "status": "retention_not_admissible",
        "reason": "no_qualified_parent_capability_set",
        "step0_accuracy": accuracy(step0),
        "step300_accuracy": accuracy(step300),
        "step0_mean_nll": mean_float(step0_nll),
        "step300_mean_nll": mean_float(step300_nll),
        "signed_nll_change": (
            mean_float(step300_nll) - mean_float(step0_nll) if step0_nll and step300_nll else None
        ),
        "note": (
            "Old behaviour is measured at both endpoints but not claimed as retained; "
            "there is no qualified parent capability set in this package."
        ),
    }


def stage_declaration() -> dict:
    return {
        "task_scope": (
            "two incremental module seeds x three structural arms; 300 batch-8 updates "
            "per arm; the full 192 train / 160 dev composition split; fixed step-0 and "
            "step-300 evaluation endpoints."
        ),
        "data_boundary": (
            "public taiji-composition-fixture-v1 train/dev worlds only; the confirm split "
            "is sealed and not generated or read; dev examples are development evidence, "
            "not a new unseen confirmation."
        ),
        "feedback_boundary": (
            "the only training signal is the raw 27-byte canonical action plus EOS byte "
            "NLL; no entity span, action id, host repair or gold record is fed forward."
        ),
        "resource_budget": {
            "cpu_seconds": FORMAL_CPU_SECONDS,
            "wall_seconds": FORMAL_WALL_SECONDS,
            "adam_calls": FORMAL_ADAM_CALLS,
            "product_bytes": FORMAL_PRODUCT_BYTES,
            "single_thread": True,
        },
        "closeout": {
            "task_gate": (
                "A_full two-seed total completed_action_exact >= 0.80; unseen combination "
                "and unseen entity >= 0.60; real and permuted both correct >= 0.60; "
                "reversed records >= 0.95."
            ),
            "component_gate": (
                "A_full minus B_local and A_full minus C_flat >= 0.10 per seed with an "
                "8-world paired bootstrap 2000, 95% lower bound above zero."
            ),
            "content_gate": (
                "no-source, wrong-record and wrong-position each lower the fixed-8 A_full "
                "score by >= 0.10."
            ),
            "named_exits": [
                "engineering_failure",
                "instrument_invalid",
                "budget_not_admitted",
                "budget_stop",
                "native_byte_task_failed",
                "task_supported_component_gain_unresolved",
                "content_path_not_necessary",
                "record_relation_supported_on_declared_task",
            ],
            "confirm_reserved": (
                "record_relation_supported_on_declared_task requires the independent 14.7 "
                "confirmation; 14.6 can only release or refuse 14.7."
            ),
        },
    }


def run_type_check(paths: list[str]) -> dict:
    command = [
        sys.executable,
        "-m",
        "mypy",
        "--follow-imports=skip",
        "--ignore-missing-imports",
        *paths,
    ]
    try:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=240)
        errors = 0
        for line in result.stdout.splitlines():
            if line.startswith("Found ") and " error" in line:
                errors = int(line.split()[1])
        return {
            "returncode": result.returncode,
            "errors": errors,
            "output": result.stdout.strip(),
            "passed": result.returncode == 0 and errors == 0,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "returncode": None,
            "errors": None,
            "output": f"{type(exc).__name__}: {exc}",
            "passed": False,
        }


def run_lint(paths: list[str]) -> dict:
    ruff = subprocess.run(
        [sys.executable, "-m", "ruff", "check", *paths],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=240,
    )
    ruff_format = subprocess.run(
        [sys.executable, "-m", "ruff", "format", "--check", *paths],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=240,
    )
    return {
        "ruff": {"returncode": ruff.returncode, "output": ruff.stdout.strip()},
        "ruff_format": {
            "returncode": ruff_format.returncode,
            "output": ruff_format.stdout.strip(),
        },
        "black": {
            "returncode": None,
            "status": "not_run",
            "reason": (
                "the installed black 26.5.1 does not terminate on the multi-thousand-line "
                "stage runners in this environment; ruff format is the declared formatter "
                "check for this stage"
            ),
        },
        "passed": ruff.returncode == 0 and ruff_format.returncode == 0,
    }


def load_chain() -> dict:
    parent_manifest = build_manifest(ROOT, ROOT / BUNDLE)
    graph_report = json.loads((ROOT / GRAPH_REPORT).read_text(encoding="utf-8"))
    graph_manifest = json.loads((ROOT / GRAPH_MANIFEST).read_text(encoding="utf-8"))
    price_report = json.loads((ROOT / PRICE_REPORT).read_text(encoding="utf-8"))
    price_manifest = json.loads((ROOT / PRICE_MANIFEST).read_text(encoding="utf-8"))
    preflight_report = json.loads((ROOT / PREFLIGHT_REPORT).read_text(encoding="utf-8"))
    preflight_manifest = json.loads((ROOT / PREFLIGHT_MANIFEST).read_text(encoding="utf-8"))
    if (
        parent_manifest["candidate_id"] != EXPECTED_PARENT
        or parent_manifest["provenance"]["sha256"] != EXPECTED_BUNDLE_SHA
        or parent_manifest["provenance"]["bytes"] != EXPECTED_BUNDLE_BYTES
    ):
        raise InstrumentError("parent identity drifted from the fixed 09.8 asset")
    if graph_report["status"] != "passed_record_relation_graph_engineering_only":
        raise InstrumentError("14.4 did not pass the engineering gate")
    if graph_manifest["candidate"]["candidate_id"] != EXPECTED_14_4_CANDIDATE:
        raise InstrumentError("14.4 candidate identity drifted")
    if price_report["status"] != "priced_within_compare_cap":
        raise InstrumentError("14.5 pricing did not admit the comparison cap")
    if not price_report.get("extrapolation", {}).get("admitted", False):
        raise InstrumentError("14.5 price did not fit the 14.6 cap")
    if preflight_report["status"] != "passed_14_6_checkpoint_preflight":
        raise InstrumentError("14.6.0 checkpoint preflight did not pass")
    if preflight_report.get("formal_run", {}).get("started", True):
        raise InstrumentError("14.6.0 reports a started formal run")
    return {
        "parent_manifest": parent_manifest,
        "graph_report": graph_report,
        "graph_manifest": graph_manifest,
        "price_report": price_report,
        "price_manifest": price_manifest,
        "preflight_report": preflight_report,
        "preflight_manifest": preflight_manifest,
    }


def chain_identity(chain: dict, source_hashes: dict) -> dict:
    preflight_seeds = {(row["seed"], row["arm"]): row for row in chain["preflight_report"]["seeds"]}
    return {
        "parent_candidate_id": chain["parent_manifest"]["candidate_id"],
        "parent_bundle_sha256": chain["parent_manifest"]["provenance"]["sha256"],
        "parent_bundle_bytes": chain["parent_manifest"]["provenance"]["bytes"],
        "graph_report": GRAPH_REPORT,
        "graph_report_sha256": sha256_text(ROOT / GRAPH_REPORT),
        "graph_candidate_id": chain["graph_manifest"]["candidate"]["candidate_id"],
        "price_report": PRICE_REPORT,
        "price_report_sha256": sha256_text(ROOT / PRICE_REPORT),
        "price_candidate_id": chain["price_manifest"]["candidate"]["candidate_id"],
        "price_manifest": PRICE_MANIFEST,
        "price_manifest_sha256": sha256_text(ROOT / PRICE_MANIFEST),
        "preflight_report": PREFLIGHT_REPORT,
        "preflight_report_sha256": sha256_text(ROOT / PREFLIGHT_REPORT),
        "preflight_manifest": PREFLIGHT_MANIFEST,
        "preflight_manifest_sha256": sha256_text(ROOT / PREFLIGHT_MANIFEST),
        "preflight_seeds": [
            {
                "seed": row["seed"],
                "arm": row["arm"],
                "candidate_id": row["candidate_id"],
                "relation_digest": row["relation_digest"],
            }
            for row in chain["preflight_report"]["seeds"]
        ],
        "expected_seed_candidates": {
            f"{seed}:{arm}": {
                "candidate_id": row["candidate_id"],
                "relation_digest": row["relation_digest"],
            }
            for (seed, arm), row in preflight_seeds.items()
        },
        "frozen_train_update_batch8_cpu": chain["price_report"]["prices"][
            "train_update_batch8_cpu"
        ],
        "formal_estimate_cpu": chain["price_report"]["extrapolation"]["components"]["total_cpu"],
        "source_hashes": source_hashes,
    }


def check_data_binding(data: dict, chain: dict) -> bool:
    preflight = chain["preflight_report"]
    return (
        data["digest"] == preflight["data"]["digest"]
        and data["train_examples"] == EXPECTED_TRAIN
        and data["dev_examples"] == EXPECTED_DEV
        and data["confirm_read"] is False
    )


def closeout_status(
    *,
    engineering_ok: bool,
    task_gate: dict,
    structure_gate: dict,
    content_gate: dict,
    budget_within: dict,
    error: str | None,
    override: str | None,
) -> tuple[str, bool]:
    if override:
        return override, False
    if error is not None or not engineering_ok:
        return "engineering_failure", False
    if not (
        budget_within["cpu"]
        and budget_within["wall"]
        and budget_within["adam"]
        and budget_within["products"]
    ):
        return "budget_stop", False
    if not task_gate["passed"]:
        return "native_byte_task_failed", False
    if not content_gate["passed"]:
        return "content_path_not_necessary", False
    if not structure_gate["passed"]:
        return "task_supported_component_gain_unresolved", False
    return "task_and_component_supported_pending_confirm", True


def build_manifest_payload(
    chain: dict,
    identity: dict,
    data: dict,
    candidates: list[dict],
    gates: dict,
    status: str,
    cost: dict,
    source_hashes: dict,
) -> dict:
    retained = []
    for candidate in candidates:
        for checkpoint in candidate.get("checkpoints", []):
            if checkpoint.get("kept"):
                retained.append(
                    {
                        "seed": candidate["seed"],
                        "arm": candidate["arm"],
                        "update_index": checkpoint["update_index"],
                        "path": checkpoint["path"],
                        "sha256": checkpoint["sha256"],
                        "bytes": checkpoint["bytes"],
                        "candidate_at_save": checkpoint["candidate_at_save"],
                    }
                )
    return {
        "schema": "taiji-record-relation-14-6-manifest-v1",
        "step": "14.6",
        "date": "2026-11-02",
        "content_schema": RECORD_RELATION_SCHEMA,
        "content_revision": CONTENT_REVISION,
        "parent": {
            "candidate_id": chain["parent_manifest"]["candidate_id"],
            "bundle": BUNDLE,
            "sha256": chain["parent_manifest"]["provenance"]["sha256"],
            "bytes": chain["parent_manifest"]["provenance"]["bytes"],
        },
        "gate_chain": {
            "graph": {
                "report": GRAPH_REPORT,
                "sha256": identity["graph_report_sha256"],
                "candidate_id": identity["graph_candidate_id"],
            },
            "price": {
                "report": PRICE_REPORT,
                "sha256": identity["price_report_sha256"],
                "manifest": PRICE_MANIFEST,
                "manifest_sha256": identity["price_manifest_sha256"],
                "candidate_id": identity["price_candidate_id"],
                "train_update_batch8_cpu": identity["frozen_train_update_batch8_cpu"],
                "formal_estimate_cpu": identity["formal_estimate_cpu"],
            },
            "preflight": {
                "report": PREFLIGHT_REPORT,
                "sha256": identity["preflight_report_sha256"],
                "manifest": PREFLIGHT_MANIFEST,
                "manifest_sha256": identity["preflight_manifest_sha256"],
            },
        },
        "data": data,
        "seeds": list(SEEDS),
        "order_seeds": list(ORDER_SEEDS),
        "arms": list(ARMS),
        "updates_per_arm": FORMAL_UPDATES_PER_ARM,
        "batch": BATCH,
        "evaluation_endpoints": [0, 300],
        "candidates": [
            {
                "seed": candidate["seed"],
                "arm": candidate["arm"],
                "order_seed": candidate["order_seed"],
                "candidate_initial": candidate["candidate_initial"],
                "candidate_final": candidate["candidate_final"],
                "relation_initial": candidate["relation_initial"],
                "relation_final": candidate["relation_final"],
                "parameter_count": candidate["parameter_count"],
                "trainable_count": candidate["trainable_count"],
                "disabled": candidate["disabled"],
                "training": {
                    "recovered": candidate["training"].get("recovered"),
                    "updates": candidate["training"].get("updates"),
                    "adam_calls": candidate["training"].get("adam_calls"),
                    "summary": candidate["training"].get("summary"),
                },
                "step300_summary": candidate.get("step300", {}).get("real", {}).get("summary"),
            }
            for candidate in candidates
        ],
        "checkpoints": [
            checkpoint
            for candidate in candidates
            for checkpoint in candidate.get("checkpoints", [])
        ],
        "retained": retained,
        "retained_count": len(retained),
        "gates": gates,
        "status": status,
        "cost": cost,
        "source_hashes": source_hashes,
        "confirm_read": False,
        "next": (
            "14.7 independent confirmation"
            if status == "task_and_component_supported_pending_confirm"
            else "14.8 closeout of this package"
        ),
    }


def compare_main(args) -> int:
    report_path = Path(args.report) if args.report else ROOT / REPORT
    manifest_path = Path(args.manifest) if args.manifest else ROOT / MANIFEST
    output_dir = Path(args.payload_dir) if args.payload_dir else ROOT / OUTPUT_DIR
    if report_path.exists():
        raise FileExistsError("14.6 formal card is write-once")
    if manifest_path.exists():
        raise FileExistsError("14.6 formal manifest is write-once")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    global STAGE_BUDGET
    STAGE_BUDGET = StageBudget(
        FORMAL_CPU_SECONDS,
        FORMAL_WALL_SECONDS,
        FORMAL_ADAM_CALLS,
        FORMAL_PRODUCT_BYTES,
    )
    progress_path = output_dir / "progress.json"
    log_path = output_dir / "run_log.jsonl"
    if log_path.exists():
        log_path.unlink()
    checks: list[dict] = []
    report: dict[str, Any] = {
        "schema": "taiji-record-relation-14-6-formal-v1",
        "step": "14.6",
        "date": "2026-11-02",
        "started_at": STARTED_AT,
        "status": "engineering_failure",
        "checks": checks,
        "declaration": stage_declaration(),
        "identity": {},
        "data": {},
        "seeds_plan": [],
        "candidates": [],
        "gates": {},
        "retention": {},
        "cost": {},
        "within": {},
        "type_check": {},
        "lint": {},
        "confirm_read": False,
        "limits": (
            "One native Chinese comparison on the declared composition task. "
            "No general Chinese, L3/L4, open autonomy or growth claim; the confirm "
            "split is sealed."
        ),
    }
    error: str | None = None
    status_override: str | None = None
    source_hashes = {relative: sha256_text(ROOT / relative) for relative in SOURCE_FILES}
    parent_path = ROOT / BUNDLE
    parent_sha_before = sha256_text(parent_path)
    try:
        chain = load_chain()
        identity = chain_identity(chain, source_hashes)
        report["identity"] = identity
        check(
            checks,
            "parent_identity",
            True,
            {
                "candidate_id": chain["parent_manifest"]["candidate_id"],
                "sha256": chain["parent_manifest"]["provenance"]["sha256"],
                "bytes": chain["parent_manifest"]["provenance"]["bytes"],
            },
        )
        check(
            checks,
            "gate_chain",
            chain["graph_report"]["status"] == "passed_record_relation_graph_engineering_only"
            and chain["price_report"]["status"] == "priced_within_compare_cap"
            and chain["price_report"]["extrapolation"]["admitted"]
            and chain["preflight_report"]["status"] == "passed_14_6_checkpoint_preflight",
            {
                "graph": chain["graph_report"]["status"],
                "price": chain["price_report"]["status"],
                "admitted": chain["price_report"]["extrapolation"]["admitted"],
                "preflight": chain["preflight_report"]["status"],
            },
        )
        train_worlds, dev_worlds, train, dev = build_full_data()
        data = data_identity(train_worlds, dev_worlds, train, dev)
        report["data"] = data
        check(
            checks,
            "data_contract",
            check_data_binding(data, chain),
            {
                "train": len(train),
                "dev": len(dev),
                "digest": data["digest"],
                "preflight_digest": chain["preflight_report"]["data"]["digest"],
            },
        )
        fixed_examples = price_dev_examples(dev_worlds[FIXED_WORLD])
        fixed_fixtures = tuple(
            example_fixture(example, f"dev-{FIXED_WORLD}-{example.index}")
            for example in fixed_examples
        )
        fixed_names = tuple(fixture["name"] for fixture in fixed_fixtures)
        if len(fixed_names) != 8:
            raise InstrumentError("the fixed structural set is not 8 examples")
        expected_seeds = identity["expected_seed_candidates"]
        plans = {}
        for seed, order_seed in zip(SEEDS, ORDER_SEEDS, strict=True):
            plan = batch_plan(len(train), order_seed, FORMAL_UPDATES_PER_ARM)
            plans[seed] = plan
            report["seeds_plan"].append(
                {
                    "seed": seed,
                    "order_seed": order_seed,
                    "batch": BATCH,
                    "updates": len(plan),
                    "batch_digest": hashlib.sha256(repr(plan).encode()).hexdigest(),
                    "first_batch": list(plan[0]),
                    "last_batch": list(plan[-1]),
                }
            )
        parent = torch.load(parent_path, map_location="cpu", weights_only=True)
        for seed, order_seed in zip(SEEDS, ORDER_SEEDS, strict=True):
            for arm in ARMS:
                expected_row = expected_seeds[f"{seed}:{arm}"]
                candidate = run_candidate(
                    parent,
                    seed=seed,
                    arm=arm,
                    order_seed=order_seed,
                    train=train,
                    dev=dev,
                    dev_worlds=dev_worlds,
                    fixed_fixtures=fixed_fixtures,
                    plan=plans[seed],
                    output_dir=output_dir,
                    progress_path=progress_path,
                    log_path=log_path,
                    checkpoints=CHECKPOINTS,
                    keep=(0, 300),
                    expected_seed_row=expected_row,
                )
                report["candidates"].append(candidate)
                check(
                    checks,
                    f"candidate_{seed}_{arm}",
                    candidate["status"] in ("complete", "resumed")
                    and candidate["candidate_initial"] == expected_row["candidate_id"]
                    and candidate["relation_initial"] == expected_row["relation_digest"],
                    {
                        "status": candidate["status"],
                        "candidate_initial": candidate["candidate_initial"],
                        "candidate_final": candidate["candidate_final"],
                    },
                )
                STAGE_BUDGET.ensure()
                STAGE_BUDGET.sample_ram()
                write_progress(
                    progress_path,
                    {
                        "phase": "candidate_complete",
                        "seed": seed,
                        "arm": arm,
                        "cost": STAGE_BUDGET.cost(),
                    },
                )
        del parent
        total_updates = sum(
            int(candidate["training"].get("updates") or 0) for candidate in report["candidates"]
        )
        check(
            checks,
            "training_completed",
            len(report["candidates"]) == 6
            and total_updates == FORMAL_ADAM_CALLS
            and STAGE_BUDGET.total_adam() == FORMAL_ADAM_CALLS,
            {
                "candidates": len(report["candidates"]),
                "total_updates": total_updates,
                "adam_executed": STAGE_BUDGET.adam_calls,
                "adam_recovered": STAGE_BUDGET.recovered_adam_calls,
            },
        )
        finite = all(
            all(
                math.isfinite(float(row["loss"]))
                for row in (candidate["training"].get("rows") or [])
            )
            for candidate in report["candidates"]
        )
        check(
            checks,
            "training_losses_finite",
            finite,
            [
                {
                    "seed": candidate["seed"],
                    "arm": candidate["arm"],
                    "recovered": candidate["training"].get("recovered"),
                    "last_loss": (
                        candidate["training"].get("summary", {}).get("last_loss")
                        if candidate["training"].get("summary")
                        else None
                    ),
                }
                for candidate in report["candidates"]
            ],
        )
        checkpoint_rows = [
            checkpoint
            for candidate in report["candidates"]
            for checkpoint in candidate.get("checkpoints", [])
        ]
        expected_checkpoints = sum(
            2 if candidate.get("status") == "resumed" else len(CHECKPOINTS)
            for candidate in report["candidates"]
        )
        checkpoint_ok = (
            len(checkpoint_rows) == expected_checkpoints
            and all(checkpoint.get("reload", {}).get("ok") for checkpoint in checkpoint_rows)
            and all(
                checkpoint.get("worker_restore", {}).get("equal_probabilities")
                and checkpoint.get("worker_restore", {}).get("equal_h")
                and checkpoint.get("worker_restore", {}).get("equal_coverage")
                for checkpoint in checkpoint_rows
                if checkpoint["update_index"] in (0, 300)
            )
        )
        check(
            checks,
            "checkpoint_chain",
            checkpoint_ok,
            {
                "checkpoints": len(checkpoint_rows),
                "expected": expected_checkpoints,
                "worker_rows": [
                    {
                        "seed": checkpoint["seed"],
                        "arm": checkpoint["arm"],
                        "step": checkpoint["update_index"],
                        "ok": checkpoint.get("worker_restore", {}).get("ok"),
                        "prob": checkpoint.get("worker_restore", {}).get("equal_probabilities"),
                        "h": checkpoint.get("worker_restore", {}).get("equal_h"),
                        "coverage": checkpoint.get("worker_restore", {}).get("equal_coverage"),
                    }
                    for checkpoint in checkpoint_rows
                    if checkpoint["update_index"] in (0, 300)
                ],
            },
        )
        real_step0 = collect_rows(report["candidates"], phase="step0", condition="real")
        real_step300 = collect_rows(report["candidates"], phase="step300", condition="real")
        permuted = collect_rows(report["candidates"], phase="step300", condition="permuted")
        reverse = collect_rows(report["candidates"], phase="step300", condition="reverse")
        structural_rows = {
            name: collect_rows(report["candidates"], phase="step300", condition="structural")
            for name in ("no_source", "wrong_record", "wrong_position", "rescue")
        }
        nll_rows = [
            item
            for candidate in report["candidates"]
            for phase in ("step0", "step300")
            for item in candidate.get(phase, {}).get("real", {}).get("nll", {}).get("rows", [])
        ]
        counts = {
            "step0_real": len(real_step0),
            "step300_real": len(real_step300),
            "permuted": len(permuted),
            "reverse": len(reverse),
            "no_source": len(structural_rows["no_source"]),
            "wrong_record": len(structural_rows["wrong_record"]),
            "wrong_position": len(structural_rows["wrong_position"]),
            "rescue": len(structural_rows["rescue"]),
            "nll": len(nll_rows),
            "generations": (
                len(real_step0)
                + len(real_step300)
                + len(permuted)
                + len(reverse)
                + sum(len(rows) for rows in structural_rows.values())
            ),
        }
        evaluation_ok = (
            counts["step0_real"] == EXPECTED_STEP0_REAL
            and counts["step300_real"] == EXPECTED_STEP300_REAL
            and counts["permuted"] == EXPECTED_PERMUTED
            and counts["reverse"] == EXPECTED_REVERSE
            and all(
                counts[name] == EXPECTED_STRUCTURAL_PER_CONDITION
                for name in ("no_source", "wrong_record", "wrong_position", "rescue")
            )
            and counts["nll"] == EXPECTED_NLL
            and counts["generations"] == EXPECTED_GENERATIONS
        )
        check(checks, "evaluation_complete", evaluation_ok, counts)
        task_gate = gate_task(report["candidates"])
        structure_gate = gate_structure(report["candidates"])
        content_gate = gate_content(report["candidates"], fixed_names)
        retention = gate_retention(report["candidates"])
        report["gates"] = {
            "task": task_gate,
            "structure": structure_gate,
            "content": content_gate,
        }
        report["retention"] = retention
        check(checks, "task_gate", task_gate["passed"], task_gate["observed"], required=False)
        check(
            checks,
            "structure_gate",
            structure_gate["passed"],
            {
                "bar": structure_gate["bar"],
                "rows": [
                    {
                        "seed": row["seed"],
                        "comparison": row["comparison"],
                        "gain": row["gain"],
                        "lower_95": row["bootstrap"]["lower_95"],
                        "passed": row["passed"],
                        "saturated": row["saturated"],
                    }
                    for row in structure_gate["rows"]
                ],
                "reason": structure_gate["reason"],
            },
            required=False,
        )
        check(
            checks,
            "content_gate",
            content_gate["passed"],
            content_gate["conditions"],
            required=False,
        )
        check(
            checks,
            "retention",
            retention["status"] == "retention_not_admissible",
            retention,
            required=False,
        )
        parent_sha_after = sha256_text(parent_path)
        check(
            checks,
            "parent_bundle_unchanged",
            parent_sha_after == parent_sha_before,
            {"before": parent_sha_before, "after": parent_sha_after},
        )
        within = STAGE_BUDGET.within()
        report["within"] = within
        report["cost"] = STAGE_BUDGET.cost()
        check(
            checks,
            "stage_cost_within_budget",
            within["cpu"] and within["wall"] and within["adam"] and within["products"],
            within,
        )
        check(
            checks,
            "products_within_limit",
            STAGE_BUDGET.max_products <= FORMAL_PRODUCT_BYTES,
            {
                "max_live_product_bytes": STAGE_BUDGET.max_products,
                "cumulative_product_bytes": STAGE_BUDGET.cumulative_products,
                "product_bytes": FORMAL_PRODUCT_BYTES,
            },
        )
        report["type_check"] = run_type_check(list(SOURCE_FILES))
        check(checks, "type_check", report["type_check"]["passed"], report["type_check"])
        report["lint"] = run_lint(list(SOURCE_FILES))
        check(checks, "lint", report["lint"]["passed"], report["lint"])
        engineering_ok = all(row["passed"] for row in checks if row.get("required", True))
        status, release = closeout_status(
            engineering_ok=engineering_ok,
            task_gate=task_gate,
            structure_gate=structure_gate,
            content_gate=content_gate,
            budget_within=within,
            error=error,
            override=status_override,
        )
        report["status"] = status
        report["release_14_7"] = release
        report["passed"] = engineering_ok
        manifest_data = build_manifest_payload(
            chain,
            identity,
            data,
            report["candidates"],
            report["gates"],
            status,
            report["cost"],
            source_hashes,
        )
        manifest_info = write_json(manifest_path, manifest_data)
        report["manifest"] = {
            "path": relative(manifest_path),
            "sha256": manifest_info["sha256"],
            "bytes": manifest_info["bytes"],
            "retained_count": len(manifest_data["retained"]),
        }
        check(
            checks,
            "manifest_identity",
            manifest_info["sha256"] == sha256_text(manifest_path),
            {"path": report["manifest"]["path"], "sha256": manifest_info["sha256"]},
        )
    except FormalStop as exc:
        error = f"FormalStop: {exc}"
        status_override = "budget_stop"
    except InstrumentError as exc:
        error = f"InstrumentError: {exc}"
        status_override = "instrument_invalid"
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        status_override = "engineering_failure"
    finally:
        report["error"] = error
        report["required_checks"] = sum(1 for row in checks if row.get("required", True))
        report["failed_required"] = [
            row["id"] for row in checks if row.get("required", True) and not row["passed"]
        ]
        if status_override:
            report["status"] = status_override
            report["passed"] = False
            report["release_14_7"] = False
        elif error is not None or report["failed_required"]:
            report["status"] = "engineering_failure"
            report["passed"] = False
            report["release_14_7"] = False
        else:
            report["passed"] = True
            report["release_14_7"] = (
                report.get("status") == "task_and_component_supported_pending_confirm"
            )
        if not report.get("cost"):
            report["cost"] = STAGE_BUDGET.cost()
        if not report.get("within"):
            report["within"] = STAGE_BUDGET.within()
        report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        info = write_json(report_path, report)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "passed": report.get("passed"),
                    "release_14_7": report.get("release_14_7", False),
                    "failed_required": report["failed_required"],
                    "task_gate": report.get("gates", {}).get("task", {}).get("passed"),
                    "structure_gate": report.get("gates", {}).get("structure", {}).get("passed"),
                    "content_gate": report.get("gates", {}).get("content", {}).get("passed"),
                    "cost": report.get("cost"),
                    "report": info["path"],
                },
                ensure_ascii=False,
            )
        )
    return 0 if report.get("passed") and report.get("release_14_7") else 1


def smoke_main(args) -> int:
    report_path = Path(args.report) if args.report else ROOT / OUTPUT_DIR / "smoke_report.json"
    output_dir = Path(args.payload_dir) if args.payload_dir else ROOT / OUTPUT_DIR
    if report_path == ROOT / REPORT:
        raise SystemExit("the smoke stage may not write the formal 14.6 card")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    global STAGE_BUDGET
    STAGE_BUDGET = StageBudget(
        SMOKE_CPU_SECONDS,
        SMOKE_WALL_SECONDS,
        SMOKE_ADAM_CALLS,
        SMOKE_PRODUCT_BYTES,
    )
    checks: list[dict] = []
    report: dict[str, Any] = {
        "schema": "taiji-record-relation-14-6-smoke-v1",
        "step": "14.6.smoke",
        "date": "2026-11-02",
        "status": "engineering_failure",
        "checks": checks,
        "identity": {},
        "data": {},
        "trace": [],
        "candidates": [],
        "structural": {},
        "cost": {},
        "type_check": {},
        "lint": {},
        "limits": "Smoke only. No formal training, no capability claim, no confirm read.",
    }
    error: str | None = None
    try:
        chain = load_chain()
        report["identity"] = chain_identity(chain, {})
        if chain["preflight_report"]["data"]["digest"] is None:
            raise InstrumentError("the preflight data digest is missing")
        train_worlds, dev_worlds, train, dev = build_full_data()
        data = data_identity(train_worlds, dev_worlds, train, dev)
        report["data"] = {
            "train": len(train),
            "dev": len(dev),
            "digest": data["digest"],
            "matches_preflight": check_data_binding(data, chain),
        }
        check(
            checks,
            "data_contract",
            check_data_binding(data, chain),
            report["data"],
        )
        seed = SEEDS[0]
        expected = {(row["seed"], row["arm"]): row for row in chain["preflight_report"]["seeds"]}
        fixed = price_dev_examples(dev_worlds[FIXED_WORLD])
        fixed_fixtures = tuple(
            example_fixture(example, f"dev-{FIXED_WORLD}-{example.index}") for example in fixed
        )
        for arm in ARMS:
            initial = build_candidate(
                torch.load(ROOT / BUNDLE, map_location="cpu", weights_only=True),
                arm=arm,
                config=RecordRelationConfig(seed=seed),
                dtype=torch.float64,
            )
            expected_row = expected[(seed, arm)]
            check(
                checks,
                f"candidate_{seed}_{arm}",
                candidate_digest(initial) == expected_row["candidate_id"]
                and relation_digest(initial) == expected_row["relation_digest"],
                {
                    "candidate_id": candidate_digest(initial),
                    "relation_digest": relation_digest(initial),
                    "disabled": initial.disabled_rows(),
                },
            )
            trace_rows = []
            for fixture in fixed_fixtures:
                case = real_case(fixture, seed)
                direct = generate_bytes(
                    initial,
                    case.goal,
                    case.records,
                    subject_id=f"smoke-direct-{arm}-{fixture['name']}",
                )
                traced = generate_trace(initial, case, tag=f"smoke-trace-{arm}")
                trace_rows.append(
                    {
                        "name": fixture["name"],
                        "match": direct["raw_bytes"] == traced["raw"]
                        and bool(direct["terminated"]) == bool(traced["terminated"]),
                        "bytes": len(direct["raw_bytes"]),
                        "terminated": bool(direct["terminated"]),
                    }
                )
            report["trace"].append({"arm": arm, "rows": trace_rows})
            check(
                checks,
                f"trace_{arm}",
                all(row["match"] for row in trace_rows),
                trace_rows,
            )
            optimizer = make_optimizer(initial)
            checkpoint0 = save_checkpoint(
                output_dir / f"smoke_{seed}_{arm}_step0.pt",
                candidate=initial,
                optimizer=optimizer,
                fixture=dev[0],
                seed=seed,
                arm=arm,
                order_seed=ORDER_SEEDS[0],
                update_index=0,
                keep=False,
                worker=True,
            )
            plan = batch_plan(len(train), ORDER_SEEDS[0], SMOKE_UPDATES)
            training = train_candidate(
                initial,
                optimizer,
                train,
                plan,
                seed=seed,
                arm=arm,
                order_seed=ORDER_SEEDS[0],
                output_dir=output_dir,
                fixture=dev[0],
                checkpoints=(0, SMOKE_UPDATES),
                keep=(),
                worker_steps=(SMOKE_UPDATES,),
                progress_path=output_dir / "smoke_progress.json",
                log_path=output_dir / "smoke_log.jsonl",
            )
            final = output_dir / f"{seed}_{arm}_step{SMOKE_UPDATES}.pt"
            if final.exists():
                raise InstrumentError("the smoke final checkpoint should have been deleted")
            report["candidates"].append(
                {
                    "seed": seed,
                    "arm": arm,
                    "checkpoint0": checkpoint0,
                    "training": training["summary"],
                    "checkpoints": training["checkpoints"],
                }
            )
            del initial
        if STAGE_BUDGET.adam_calls != SMOKE_ADAM_CALLS:
            raise InstrumentError(
                f"expected {SMOKE_ADAM_CALLS} smoke Adam calls, observed {STAGE_BUDGET.adam_calls}"
            )
        a = build_candidate(
            torch.load(ROOT / BUNDLE, map_location="cpu", weights_only=True),
            arm=ARM_FULL,
            config=RecordRelationConfig(seed=seed),
            dtype=torch.float64,
        )
        structural = evaluate_structural(a, fixed_fixtures, dev_worlds, seed=seed, arm=ARM_FULL)
        report["structural"] = {
            name: data_row["summary"] for name, data_row in structural["conditions"].items()
        }
        wrong_count = sum(
            1
            for name in ("wrong_record", "wrong_position")
            for row in structural["conditions"][name]["rows"]
            if row["generated_text"].startswith("把")
        )
        check(
            checks,
            "structural_forced_reads",
            all(
                len(structural["conditions"][name]["rows"]) == len(fixed_fixtures)
                for name in ("no_source", "wrong_record", "wrong_position", "rescue")
            )
            and wrong_count == 0,
            {
                "counts": {
                    name: len(structural["conditions"][name]["rows"])
                    for name in ("no_source", "wrong_record", "wrong_position", "rescue")
                },
                "wrong_reads_starting_with_action_prefix": wrong_count,
            },
        )
        permuted = tuple(permuted_case(fixture, dev_worlds, seed) for fixture in fixed_fixtures)
        reverse = tuple(reverse_case(fixture, seed) for fixture in fixed_fixtures)
        permuted_rows = evaluate_cases(a, permuted, tag="smoke-permuted")
        reverse_rows = evaluate_cases(a, reverse, tag="smoke-reverse")
        changed = sum(
            1
            for case, fixture in zip(permuted, fixed_fixtures, strict=True)
            if case.expected != fixture["expected_bytes"]
        )
        check(
            checks,
            "counterfactual_contract",
            len(permuted_rows["rows"]) == len(fixed_fixtures)
            and len(reverse_rows["rows"]) == len(fixed_fixtures)
            and all(
                row["records"] == [record.source_id for record in reversed(fixture["records"])]
                for row, fixture in zip(reverse_rows["rows"], fixed_fixtures, strict=True)
            )
            and changed > 0,
            {
                "permuted_rows": len(permuted_rows["rows"]),
                "reverse_rows": len(reverse_rows["rows"]),
                "permuted_expected_changed": changed,
            },
        )
        del a
        report["cost"] = STAGE_BUDGET.cost()
        check(
            checks,
            "smoke_cost_within_budget",
            STAGE_BUDGET.total_cpu() <= SMOKE_CPU_SECONDS
            and time.perf_counter() - STARTED_WALL <= SMOKE_WALL_SECONDS
            and STAGE_BUDGET.adam_calls <= SMOKE_ADAM_CALLS,
            report["cost"],
        )
        report["type_check"] = run_type_check(list(SOURCE_FILES))
        check(checks, "type_check", report["type_check"]["passed"], report["type_check"])
        report["lint"] = run_lint(list(SOURCE_FILES))
        check(checks, "lint", report["lint"]["passed"], report["lint"])
        check(
            checks,
            "formal_cards_absent",
            not (ROOT / REPORT).exists() and not (ROOT / MANIFEST).exists(),
            {"report": (ROOT / REPORT).exists(), "manifest": (ROOT / MANIFEST).exists()},
        )
    except FormalStop as exc:
        error = f"FormalStop: {exc}"
    except InstrumentError as exc:
        error = f"InstrumentError: {exc}"
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    finally:
        for path in output_dir.glob("smoke_*.pt"):
            with contextlib.suppress(OSError):
                path.unlink()
        for arm in ARMS:
            path = output_dir / f"{SEEDS[0]}_{arm}_step{SMOKE_UPDATES}.pt"
            if path.exists():
                with contextlib.suppress(OSError):
                    path.unlink()
        report["error"] = error
        report["required_checks"] = sum(1 for row in checks if row.get("required", True))
        report["failed_required"] = [
            row["id"] for row in checks if row.get("required", True) and not row["passed"]
        ]
        passed = error is None and not report["failed_required"]
        report["passed"] = passed
        if not report.get("cost"):
            report["cost"] = STAGE_BUDGET.cost()
        if "status" not in report or report.get("status") == "engineering_failure":
            report["status"] = "passed_14_6_smoke" if passed else "engineering_failure"
        report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        info = write_json(report_path, report)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "passed": report["passed"],
                    "failed_required": report["failed_required"],
                    "adam_calls": STAGE_BUDGET.adam_calls,
                    "cost": report["cost"],
                    "report": info["path"],
                },
                ensure_ascii=False,
            )
        )
    return 0 if report["passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "compare"), default="smoke")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--payload-dir", type=Path)
    args = parser.parse_args()
    if args.stage == "smoke":
        return smoke_main(args)
    return compare_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
