"""14.5 same-form pricing and freeze for the record-relation candidate.

The stage measures the declared costs of the three fixed structural arms on the
public 10.21 composition split: batch-8 training updates with the full 27-byte
action plus EOS, record/goal encoding, BPTT, self-generation, forced
intervention and independent restore.  It then extrapolates the 14.6
two-seed x three-arm x 300-update run against the frozen 14400 CPU / 28800 wall
/ 1800 Adam / 4 GiB product cap.

The price stage does not train a candidate for adoption, does not claim task
capability and does not read the sealed confirm split.  All engineering
parameter copies and bundles are deleted after measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import contextlib

import torch  # noqa: E402

from taiji.candidate_manifest import build_manifest  # noqa: E402
from taiji.core_09_bundle import BUNDLE_RELATIVE_PATH as CORE_09_BUNDLE_PATH  # noqa: E402
from taiji.core_09_bundle import EXPECTED_BYTES as CORE_09_BUNDLE_BYTES  # noqa: E402
from taiji.core_09_bundle import EXPECTED_SHA256 as CORE_09_BUNDLE_SHA  # noqa: E402
from taiji.living_core import EOS  # noqa: E402
from taiji.native_composition_fixture import (  # noqa: E402
    SPLIT_WORLDS,
    build_split,
    fixture_digest,
    fixture_payload,
    price_dev_examples,
    price_train_examples,
)
from taiji.record_relation_core import (  # noqa: E402
    ARM_FLAT,
    ARM_FULL,
    ARM_LOCAL,
    ARMS,
    CLIP_NORM,
    CONTENT_REVISION,
    answer_step,
    build_candidate,
    candidate_digest,
    generate_bytes,
    make_optimizer,
    observe_goal,
    prepare,
    run_worker,
    save_bundle,
    score_example,
)

BUNDLE = CORE_09_BUNDLE_PATH
EXPECTED_CANDIDATE = "feb0cbcb00e2e16bb66b77db530ae566ce40eecaf2e14852b7cc1661292139dd"
EXPECTED_BUNDLE_SHA = CORE_09_BUNDLE_SHA
EXPECTED_BUNDLE_BYTES = CORE_09_BUNDLE_BYTES
EXPECTED_14_4_CANDIDATE = "16f7e33cfc24101b3d702c5020e9b1aa1bacaacf4144de27acd8d25c98176352"
GRAPH_REPORT = "reports/taiji_core_record_relation_14_4_20261102.json"
GRAPH_MANIFEST = "plans/manifests/taiji_core_record_relation_14_4_20261102.json"
REPORT = "reports/taiji_core_record_relation_14_5_20261102.json"
MANIFEST = "plans/manifests/taiji_core_record_relation_14_5_20261102.json"
OUTPUT_DIR = "output/taiji_core_record_relation_14_5"

PRICE_CPU_SECONDS = 600
PRICE_WALL_SECONDS = 1200
PRICE_ADAM_CALLS = 12
PRICE_PRODUCT_BYTES = 1 * 1024**3
UPDATES_PER_ARM = 4
BATCH = 8
WORKER_TIMEOUT = 240

COMPARE_CPU_SECONDS = 14400
COMPARE_WALL_SECONDS = 28800
COMPARE_ADAM_CALLS = 1800
COMPARE_PRODUCT_BYTES = 4 * 1024**3
COMPARE_SEEDS = 2
COMPARE_ARMS = 3
COMPARE_UPDATES_PER_ARM = 300
COMPARE_DEV_EXAMPLES = 160
COMPARE_EVAL_ENDPOINTS = 2
COMPARE_CONDITIONS = 7
COMPARE_CONDITION_EXAMPLES = COMPARE_DEV_EXAMPLES
OVERHEAD_FRACTION = 0.10
STARTUP_WALL_SECONDS = 60.0

SOURCE_FILES = (
    "taiji/record_relation.py",
    "taiji/record_relation_core.py",
    "taiji/native_composition_fixture.py",
    "scripts/training/verify_taiji_record_relation_14_5.py",
)

STARTED_WALL = time.perf_counter()
STARTED_CPU = time.process_time()


class PriceStop(RuntimeError):
    """The declared price-stage budget was exhausted."""


class InstrumentError(RuntimeError):
    """A parent, data or candidate identity invariant failed."""


def sha256_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


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


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def example_fixture(example, name: str) -> dict:
    target = tuple(example.action.encode()) + (EOS,)
    if len(example.action.encode()) != 27 or len(target) != 28:
        raise InstrumentError("The composition fixture is not the declared 27 bytes plus EOS")
    return {
        "name": name,
        "index": example.index,
        "layer": example.layer,
        "goal": example.goal,
        "records": example.records,
        "target": target,
    }


def build_price_data() -> tuple[Any, Any, tuple[dict, ...], tuple[dict, ...]]:
    """One train world and one dev world; 8 declared examples from each."""
    train_world = build_split("train", worlds=1)[0]
    dev_world = build_split("dev", worlds=1)[0]
    train_examples = price_train_examples(train_world)
    dev_examples = price_dev_examples(dev_world)
    if len(train_examples) != BATCH or len(dev_examples) != BATCH:
        raise InstrumentError("The declared price picks are not 8/8")
    train = tuple(example_fixture(example, f"train-{example.index}") for example in train_examples)
    dev = tuple(example_fixture(example, f"dev-{example.index}") for example in dev_examples)
    for fixture in (*train, *dev):
        if len(fixture["goal"].encode()) != 36:
            raise InstrumentError("The composition goal is not 36 bytes")
        if not fixture["records"]:
            raise InstrumentError("A composition example unexpectedly has no records")
    return train_world, dev_world, train, dev


def data_identity(train_world, dev_world, train, dev) -> dict:
    return {
        "schema": "taiji-composition-fixture-v1",
        "declared_worlds": dict(SPLIT_WORLDS),
        "price_worlds": {
            "train": train_world.summary(),
            "dev": dev_world.summary(),
        },
        "price_examples": {
            "train": [fixture_payload_row(fixture) for fixture in train],
            "dev": [fixture_payload_row(fixture) for fixture in dev],
        },
        "digest": fixture_digest((train_world, dev_world)),
        "payload": fixture_payload((train_world, dev_world)),
        "confirm_read": False,
    }


def fixture_payload_row(fixture: dict) -> dict:
    return {
        "name": fixture["name"],
        "index": fixture["index"],
        "layer": fixture["layer"],
        "goal": fixture["goal"],
        "records": [
            {"source_id": record.source_id, "raw": record.raw.decode("utf-8")}
            for record in fixture["records"]
        ],
        "action_bytes": len(fixture["target"]) - 1,
        "target_bytes": list(fixture["target"]),
    }


def spawn_worker(spec: dict) -> dict:
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=run_worker, args=(spec,))
    process.start()
    process.join(timeout=WORKER_TIMEOUT)
    if process.is_alive():
        process.terminate()
        process.join()
        return {"ok": False, "message": "worker timed out", "cpu_seconds": 0.0, "wall_seconds": 0.0}
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


def worker_spec(
    bundle: Path,
    result: Path,
    fixtures: tuple[dict, ...],
    *,
    previous_byte: int,
    gate_index: int,
) -> dict:
    return {
        "bundle": str(bundle),
        "result": str(result),
        "fixtures": tuple(
            {
                "goal": fixture["goal"],
                "records": fixture["records"],
                "target_bytes": fixture["target"],
            }
            for fixture in fixtures
        ),
        "dtype": "float64",
        "mode": "probe",
        "previous_byte": previous_byte,
        "fixture_index": 0,
        "force": None,
        "gamma_override": None,
        "gate_index": gate_index,
    }


class PriceBudget:
    def __init__(self) -> None:
        self.main_cpu = 0.0
        self.worker_cpu = 0.0
        self.wall = 0.0
        self.adam_calls = 0
        self.live_products = 0
        self.max_products = 0
        self.cumulative_products = 0
        self.peak_ram_mb = 0.0

    def total_cpu(self) -> float:
        return self.main_cpu + self.worker_cpu

    def ensure(self) -> None:
        self.main_cpu = time.process_time() - STARTED_CPU
        if self.total_cpu() > PRICE_CPU_SECONDS:
            raise PriceStop(f"CPU budget exceeded: {self.total_cpu():.2f}")
        wall = time.perf_counter() - STARTED_WALL
        if wall > PRICE_WALL_SECONDS:
            raise PriceStop(f"wall budget exceeded: {wall:.2f}")
        if self.adam_calls > PRICE_ADAM_CALLS:
            raise PriceStop(f"Adam budget exceeded: {self.adam_calls}")
        if self.max_products > PRICE_PRODUCT_BYTES:
            raise PriceStop(f"product budget exceeded: {self.max_products}")

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


def batch_loss(candidate, fixtures: tuple[dict, ...]) -> torch.Tensor:
    losses = [
        score_example(
            candidate,
            fixture["goal"].encode(),
            fixture["records"],
            fixture["target"][:-1],
        ).nll
        for fixture in fixtures
    ]
    return sum(losses) / len(losses)


def measure_update(candidate, optimizer, fixtures) -> dict:
    start = time.process_time()
    loss = batch_loss(candidate, fixtures)
    forward = time.process_time() - start
    start = time.process_time()
    optimizer.zero_grad()
    loss.backward()
    bptt = time.process_time() - start
    start = time.process_time()
    clip = float(torch.nn.utils.clip_grad_norm_(candidate.trainable_parameters(), CLIP_NORM))
    optimizer.step()
    step = time.process_time() - start
    return {
        "loss": float(loss.detach()),
        "forward_cpu": forward,
        "bptt_cpu": bptt,
        "step_cpu": step,
        "total_cpu": forward + bptt + step,
        "clip_scale": clip,
    }


def measure_encoding(candidate, fixtures) -> float:
    state = candidate.subject.initial_state("price-encode", revision=CONTENT_REVISION)
    start = time.process_time()
    for fixture in fixtures:
        prepare(candidate, state, fixture["goal"].encode(), fixture["records"])
    return (time.process_time() - start) / len(fixtures)


def measure_generation(candidate, fixtures) -> tuple[float, list[dict]]:
    results = []
    start = time.process_time()
    with torch.no_grad():
        for fixture in fixtures:
            results.append(
                generate_bytes(
                    candidate,
                    fixture["goal"].encode(),
                    fixture["records"],
                    subject_id=f"price-gen-{fixture['index']}",
                )
            )
    return (time.process_time() - start) / len(fixtures), [
        {
            "index": fixture["index"],
            "terminated": result["terminated"],
            "raw_bytes": len(result["raw_bytes"]),
            "raw_sha256": hashlib.sha256(result["raw_bytes"]).hexdigest(),
        }
        for fixture, result in zip(fixtures, results, strict=True)
    ]


def measure_eval_score(candidate, fixtures) -> float:
    start = time.process_time()
    with torch.no_grad():
        for fixture in fixtures:
            score_example(
                candidate,
                fixture["goal"].encode(),
                fixture["records"],
                fixture["target"][:-1],
                subject_id=f"price-eval-{fixture['index']}",
            )
    return (time.process_time() - start) / len(fixtures)


def measure_intervention(candidate, fixtures) -> float:
    start = time.process_time()
    with torch.no_grad():
        for fixture in fixtures:
            state = candidate.subject.initial_state(
                f"price-force-{fixture['index']}", revision=CONTENT_REVISION
            )
            prepared = prepare(candidate, state, fixture["goal"].encode(), fixture["records"])
            observation, gate_index = observe_goal(candidate, prepared)
            if prepared.view.record_count:
                if candidate.flat:
                    total = sum(prepared.view.lengths)
                    flat = torch.zeros(total + 1, dtype=torch.float64)
                    flat[0] = 1.0
                    force = {"flat": flat}
                else:
                    force = {
                        "alpha": torch.tensor(
                            [
                                1.0 if index == 0 else 0.0
                                for index in range(prepared.view.record_count)
                            ]
                            + [0.0],
                            dtype=torch.float64,
                        ),
                        "betas": tuple(
                            torch.tensor(
                                [
                                    1.0 if index == 0 else 0.0
                                    for index in range(prepared.view.lengths[record])
                                ],
                                dtype=torch.float64,
                            )
                            for record in range(prepared.view.record_count)
                        ),
                    }
                answer_step(
                    candidate,
                    prepared,
                    observation,
                    prepared.coverage,
                    EOS,
                    gate_index=gate_index,
                    force=force,
                    gamma_override=0.0,
                )
    return (time.process_time() - start) / len(fixtures)


def measure_restore(
    candidate,
    optimizer,
    fixtures,
    *,
    output_dir: Path,
    tag: str,
    update_index: int,
    budget: PriceBudget,
) -> dict:
    first = fixtures[0]
    state = candidate.subject.initial_state(f"price-restore-{tag}", revision=CONTENT_REVISION)
    prepared = prepare(candidate, state, first["goal"].encode(), first["records"])
    observation, gate_index = observe_goal(candidate, prepared)
    probe = answer_step(
        candidate,
        prepared,
        observation,
        observation.coverage,
        EOS,
        gate_index=gate_index,
    )
    bundle = output_dir / f"restore_{tag}.pt"
    result_path = output_dir / f"restore_{tag}_probe.pt"
    start = time.process_time()
    info = save_bundle(
        bundle,
        profile=f"price-restore-{tag}",
        candidate=candidate,
        state=observation,
        optimizer=optimizer,
        update_index=update_index,
        cursor={"tag": tag},
        parent_identity={"candidate": candidate_digest(candidate)},
    )
    save_cpu = time.process_time() - start
    budget.add_product(int(info["bytes"]))
    start = time.process_time()
    worker = spawn_worker(
        worker_spec(
            bundle,
            result_path,
            fixtures,
            previous_byte=EOS,
            gate_index=gate_index,
        )
    )
    elapsed = time.process_time() - start
    budget.worker_cpu += float(worker.get("cpu_seconds", 0.0))
    budget.wall += float(worker.get("wall_seconds", 0.0))
    budget.sample_ram()
    equal = bool(worker.get("ok")) and bool(
        torch.equal(probe.probabilities, worker["probabilities"])
    )
    diff = (
        None if not worker.get("ok") else max_abs_diff(probe.probabilities, worker["probabilities"])
    )
    size = int(info["bytes"])
    bundle.unlink()
    if result_path.exists():
        result_path.unlink()
    budget.drop_product(size)
    return {
        "ok": bool(worker.get("ok")),
        "equal": equal,
        "max_abs_diff": diff,
        "save_cpu": save_cpu,
        "worker_cpu": float(worker.get("cpu_seconds", 0.0)),
        "wall_seconds": float(worker.get("wall_seconds", 0.0)),
        "elapsed_cpu": save_cpu + float(worker.get("cpu_seconds", 0.0)),
        "parent_wait_cpu": elapsed,
        "bundle_bytes": size,
        "bundle_mib": size / 1024**2,
        "adam_calls": 0,
    }


def run_arm(
    parent: dict,
    train: tuple[dict, ...],
    dev: tuple[dict, ...],
    *,
    arm: str,
    output_dir: Path,
    budget: PriceBudget,
) -> dict:
    candidate = build_candidate(parent, arm=arm, dtype=torch.float64)
    optimizer = make_optimizer(candidate)
    initial_restore = measure_restore(
        candidate,
        optimizer,
        dev,
        output_dir=output_dir,
        tag=f"{arm}-initial",
        update_index=0,
        budget=budget,
    )
    encoding_cpu = measure_encoding(candidate, train)
    update_rows = []
    for index in range(UPDATES_PER_ARM):
        row = measure_update(candidate, optimizer, train)
        row["index"] = index
        update_rows.append(row)
        budget.adam_calls += 1
        budget.ensure()
    generation_cpu, generation_rows = measure_generation(candidate, dev)
    eval_cpu = measure_eval_score(candidate, dev)
    intervention_cpu = measure_intervention(candidate, dev)
    final_restore = measure_restore(
        candidate,
        optimizer,
        dev,
        output_dir=output_dir,
        tag=f"{arm}-final",
        update_index=UPDATES_PER_ARM,
        budget=budget,
    )
    result = {
        "arm": arm,
        "candidate_digest": candidate_digest(candidate),
        "parameter_count": candidate.parameter_count(),
        "trainable_count": int(
            sum(parameter.numel() for parameter in candidate.trainable_parameters())
        ),
        "disabled": candidate.disabled_rows(),
        "encoding_cpu": encoding_cpu,
        "updates": update_rows,
        "generation_cpu": generation_cpu,
        "generation": generation_rows,
        "eval_score_cpu": eval_cpu,
        "intervention_cpu": intervention_cpu,
        "initial_restore": initial_restore,
        "final_restore": final_restore,
        "adam_calls": UPDATES_PER_ARM,
    }
    del candidate
    return result


def frozen_prices(arms: list[dict]) -> dict:
    updates = [row for arm in arms for row in arm["updates"]]
    return {
        "train_update_batch8_cpu": median([row["total_cpu"] for row in updates]),
        "forward_batch8_cpu": median([row["forward_cpu"] for row in updates]),
        "bptt_batch8_cpu": median([row["bptt_cpu"] for row in updates]),
        "adam_step_batch8_cpu": median([row["step_cpu"] for row in updates]),
        "encode_example_cpu": median([arm["encoding_cpu"] for arm in arms]),
        "self_generate_example_cpu": median([arm["generation_cpu"] for arm in arms]),
        "eval_score_example_cpu": median([arm["eval_score_cpu"] for arm in arms]),
        "intervention_example_cpu": median([arm["intervention_cpu"] for arm in arms]),
        "restore_cpu": median([arm["final_restore"]["elapsed_cpu"] for arm in arms]),
        "bundle_mib": median([arm["final_restore"]["bundle_mib"] for arm in arms]),
    }


def extrapolate(prices: dict) -> dict:
    train_updates = COMPARE_SEEDS * COMPARE_ARMS * COMPARE_UPDATES_PER_ARM
    eval_generations = COMPARE_SEEDS * COMPARE_ARMS * COMPARE_EVAL_ENDPOINTS * COMPARE_DEV_EXAMPLES
    eval_scores = eval_generations
    interventions = (
        COMPARE_SEEDS
        * COMPARE_ARMS
        * COMPARE_EVAL_ENDPOINTS
        * COMPARE_CONDITION_EXAMPLES
        * COMPARE_CONDITIONS
    )
    restores = COMPARE_SEEDS * COMPARE_ARMS
    training_cpu = train_updates * prices["train_update_batch8_cpu"]
    generation_cpu = eval_generations * prices["self_generate_example_cpu"]
    score_cpu = eval_scores * prices["eval_score_example_cpu"]
    intervention_cpu = interventions * prices["intervention_example_cpu"]
    restore_cpu = restores * prices["restore_cpu"]
    subtotal = training_cpu + generation_cpu + score_cpu + intervention_cpu + restore_cpu
    overhead = OVERHEAD_FRACTION * subtotal
    total_cpu = subtotal + overhead
    wall_ratio = 1.0
    total_wall = total_cpu * wall_ratio + STARTUP_WALL_SECONDS
    products_mib = restores * prices["bundle_mib"] + 100.0
    return {
        "formula": {
            "train_updates": train_updates,
            "train_batch": BATCH,
            "eval_generations": eval_generations,
            "eval_scores": eval_scores,
            "interventions": interventions,
            "restores": restores,
            "overhead_fraction": OVERHEAD_FRACTION,
            "startup_wall_seconds": STARTUP_WALL_SECONDS,
        },
        "components": {
            "training_cpu": training_cpu,
            "generation_cpu": generation_cpu,
            "eval_score_cpu": score_cpu,
            "intervention_cpu": intervention_cpu,
            "restore_cpu": restore_cpu,
            "subtotal_cpu": subtotal,
            "overhead_cpu": overhead,
            "total_cpu": total_cpu,
            "estimated_wall_seconds": total_wall,
            "estimated_products_mib": products_mib,
        },
        "cap": {
            "cpu_seconds": COMPARE_CPU_SECONDS,
            "wall_seconds": COMPARE_WALL_SECONDS,
            "adam_calls": COMPARE_ADAM_CALLS,
            "product_bytes": COMPARE_PRODUCT_BYTES,
        },
        "within": {
            "cpu": total_cpu <= COMPARE_CPU_SECONDS,
            "wall": total_wall <= COMPARE_WALL_SECONDS,
            "adam": train_updates <= COMPARE_ADAM_CALLS,
            "products": products_mib * 1024**2 <= COMPARE_PRODUCT_BYTES,
        },
        "admitted": (
            total_cpu <= COMPARE_CPU_SECONDS
            and total_wall <= COMPARE_WALL_SECONDS
            and train_updates <= COMPARE_ADAM_CALLS
            and products_mib * 1024**2 <= COMPARE_PRODUCT_BYTES
        ),
    }


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


def run_type_check() -> dict:
    paths = list(SOURCE_FILES)
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


def run_lint() -> dict:
    ruff = subprocess.run(
        [sys.executable, "-m", "ruff", "check", *SOURCE_FILES],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    black = subprocess.run(
        [sys.executable, "-m", "black", "--no-cache", "--check", *SOURCE_FILES],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return {
        "ruff": {"returncode": ruff.returncode, "output": ruff.stdout.strip()},
        "black": {"returncode": black.returncode, "output": black.stdout.strip()},
        "passed": ruff.returncode == 0 and black.returncode == 0,
    }


def build_price_manifest(
    parent: dict,
    parent_manifest: dict,
    graph_manifest: dict,
    graph_report: dict,
    candidate,
    data: dict,
    prices: dict,
    extrapolation: dict,
    source_hashes: dict,
    graph_source_hashes: dict,
) -> dict:
    return {
        "schema": "taiji-record-relation-price-manifest-v1",
        "step": "14.5",
        "date": "2026-11-02",
        "content_schema": "taiji-record-relation-v1",
        "parent": {
            "candidate_id": parent_manifest["candidate_id"],
            "bundle": BUNDLE,
            "sha256": parent_manifest["provenance"]["sha256"],
            "bytes": parent_manifest["provenance"]["bytes"],
            "subject_id": parent["subject_id"],
            "revision": parent["revision"],
            "step": parent["step"],
        },
        "candidate": {
            "candidate_id": candidate_digest(candidate),
            "arm": ARM_FULL,
            "relation_config": {
                "width": candidate.config.width,
                "key_width": candidate.config.key_width,
                "hidden": candidate.config.hidden,
                "chunk": candidate.config.chunk,
            },
            "total_parameters": candidate.parameter_count(),
            "new_parameters": candidate.new_parameter_count(),
            "graph_manifest": GRAPH_MANIFEST,
            "graph_manifest_sha256": sha256_text(ROOT / GRAPH_MANIFEST),
            "graph_report": GRAPH_REPORT,
            "graph_report_sha256": sha256_text(ROOT / GRAPH_REPORT),
            "graph_candidate_id": graph_manifest["candidate"]["candidate_id"],
            "graph_status": graph_report["status"],
        },
        "data": data,
        "prices": prices,
        "extrapolation": extrapolation,
        "arms": list(ARMS),
        "updates_per_arm": UPDATES_PER_ARM,
        "batch": BATCH,
        "pricing_parameters_discarded": True,
        "retained_parameters": [],
        "source_hashes": source_hashes,
        "graph_source_hashes": graph_source_hashes,
        "limits": (
            "Same-form price measurement only. No task training, no capability claim, "
            "no adoption, no confirm read; engineering parameter copies discarded."
        ),
    }


def price_main(args) -> int:
    report_path = Path(args.report) if args.report else ROOT / REPORT
    manifest_path = Path(args.manifest) if args.manifest else ROOT / MANIFEST
    output_dir = Path(args.payload_dir) if args.payload_dir else ROOT / OUTPUT_DIR
    if report_path.exists():
        raise FileExistsError("14.5 price card is write-once")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    budget = PriceBudget()
    checks: list[dict] = []
    report: dict[str, Any] = {
        "schema": "taiji-record-relation-price-v1",
        "step": "14.5",
        "date": "2026-11-02",
        "status": "engineering_failure",
        "checks": checks,
        "identity": {},
        "data": {},
        "prices": {},
        "extrapolation": {},
        "arms": [],
        "cost": {},
        "type_check": {},
        "lint": {},
        "limits": (
            "Same-form price measurement only. No task training, no capability claim, "
            "no adoption, no confirm read; engineering parameter copies discarded."
        ),
    }
    error: str | None = None
    status_override: str | None = None
    source_hashes = {relative: sha256_text(ROOT / relative) for relative in SOURCE_FILES}
    parent_path = ROOT / BUNDLE
    parent_sha_before = sha256_text(parent_path)
    try:
        parent = torch.load(parent_path, map_location="cpu", weights_only=True)
        parent_manifest = build_manifest(ROOT, parent_path)
        if (
            parent_manifest["candidate_id"] != EXPECTED_CANDIDATE
            or parent_manifest["provenance"]["sha256"] != EXPECTED_BUNDLE_SHA
            or parent_manifest["provenance"]["bytes"] != EXPECTED_BUNDLE_BYTES
        ):
            raise InstrumentError("Parent identity drifted from the fixed 09.8 asset")
        graph_report = json.loads((ROOT / GRAPH_REPORT).read_text(encoding="utf-8"))
        graph_manifest = json.loads((ROOT / GRAPH_MANIFEST).read_text(encoding="utf-8"))
        if graph_report["status"] != "passed_record_relation_graph_engineering_only":
            raise InstrumentError("14.4 did not pass the engineering gate")
        if graph_manifest["candidate"]["candidate_id"] != EXPECTED_14_4_CANDIDATE:
            raise InstrumentError("14.4 candidate identity drifted")
        identity = {
            "parent_candidate_id": parent_manifest["candidate_id"],
            "parent_bundle_sha256": parent_manifest["provenance"]["sha256"],
            "parent_bundle_bytes": parent_manifest["provenance"]["bytes"],
            "parent_subject_id": parent["subject_id"],
            "parent_revision": parent["revision"],
            "parent_step": parent["step"],
            "graph_manifest": GRAPH_MANIFEST,
            "graph_manifest_sha256": sha256_text(ROOT / GRAPH_MANIFEST),
            "graph_report": GRAPH_REPORT,
            "graph_report_sha256": sha256_text(ROOT / GRAPH_REPORT),
            "graph_candidate_id": graph_manifest["candidate"]["candidate_id"],
            "source_hashes": source_hashes,
            "graph_source_hashes": graph_manifest["source_hashes"],
        }
        report["identity"] = identity
        check(
            checks,
            "parent_identity",
            True,
            {
                "candidate_id": parent_manifest["candidate_id"],
                "sha256": parent_manifest["provenance"]["sha256"],
                "bytes": parent_manifest["provenance"]["bytes"],
            },
        )
        check(
            checks,
            "graph_admission",
            True,
            {
                "status": graph_report["status"],
                "candidate_id": graph_manifest["candidate"]["candidate_id"],
            },
        )

        train_world, dev_world, train, dev = build_price_data()
        data = data_identity(train_world, dev_world, train, dev)
        report["data"] = data
        check(
            checks,
            "data_contract",
            len(train) == BATCH
            and len(dev) == BATCH
            and all(len(fixture["target"]) == 28 for fixture in (*train, *dev))
            and all(len(fixture["target"]) - 1 == 27 for fixture in (*train, *dev))
            and data["declared_worlds"]["train"] == 16
            and data["declared_worlds"]["dev"] == 8
            and data["confirm_read"] is False,
            {
                "train": len(train),
                "dev": len(dev),
                "declared_worlds": data["declared_worlds"],
                "digest": data["digest"],
            },
        )

        candidate = build_candidate(parent, dtype=torch.float64)
        check(
            checks,
            "candidate_pedigree",
            candidate_digest(candidate) == EXPECTED_14_4_CANDIDATE,
            {"candidate_id": candidate_digest(candidate)},
        )
        report["candidate"] = {
            "candidate_id": candidate_digest(candidate),
            "total_parameters": candidate.parameter_count(),
            "trainable_parameters": int(
                sum(parameter.numel() for parameter in candidate.trainable_parameters())
            ),
            "new_parameters": candidate.new_parameter_count(),
            "relation_config": asdict_config(candidate.config),
        }
        del candidate

        arms = []
        for arm in (ARM_FULL, ARM_LOCAL, ARM_FLAT):
            row = run_arm(
                parent,
                train,
                dev,
                arm=arm,
                output_dir=output_dir,
                budget=budget,
            )
            arms.append(row)
            report["arms"].append(row)
            budget.ensure()
        if budget.adam_calls != PRICE_ADAM_CALLS:
            raise InstrumentError(f"expected 12 Adam calls, observed {budget.adam_calls}")
        check(
            checks,
            "same_form_contract",
            all(row["updates"] for row in arms)
            and all(len(row["updates"]) == UPDATES_PER_ARM for row in arms)
            and all(row["initial_restore"]["equal"] for row in arms)
            and all(row["final_restore"]["equal"] for row in arms)
            and budget.adam_calls == PRICE_ADAM_CALLS,
            {
                "arms": [row["arm"] for row in arms],
                "updates": [len(row["updates"]) for row in arms],
                "adam_calls": budget.adam_calls,
                "initial_restore": [row["initial_restore"]["equal"] for row in arms],
                "final_restore": [row["final_restore"]["equal"] for row in arms],
            },
        )
        check(
            checks,
            "update_losses_finite",
            all(
                all(torch.isfinite(torch.tensor(row["loss"])) for row in arm["updates"])
                for arm in arms
            ),
            [[row["loss"] for row in arm["updates"]] for arm in arms],
        )

        prices = frozen_prices(arms)
        report["prices"] = prices
        extrapolation = extrapolate(prices)
        report["extrapolation"] = extrapolation
        check(
            checks,
            "price_formula_frozen",
            extrapolation["formula"]["train_updates"] == COMPARE_ADAM_CALLS
            and extrapolation["formula"]["train_batch"] == BATCH
            and extrapolation["formula"]["overhead_fraction"] == OVERHEAD_FRACTION,
            extrapolation["formula"],
        )
        check(
            checks,
            "compare_budget_admission",
            extrapolation["admitted"],
            extrapolation["within"],
            "If false the named exit is budget_not_admitted; the dose and denominator are not changed.",
            required=False,
        )
        report["decision"] = {
            "admitted": extrapolation["admitted"],
            "reason": None if extrapolation["admitted"] else "price_does_not_fit_compare_cap",
        }

        parent_sha_after = sha256_text(parent_path)
        check(
            checks,
            "parent_bundle_unchanged",
            parent_sha_after == parent_sha_before,
            {"before": parent_sha_before, "after": parent_sha_after},
        )
        check(
            checks,
            "products_within_limit",
            budget.max_products <= PRICE_PRODUCT_BYTES,
            {
                "max_live_bytes": budget.max_products,
                "cumulative_bytes": budget.cumulative_products,
                "cap": PRICE_PRODUCT_BYTES,
            },
        )
        report["cost"] = {
            "main_cpu_seconds": time.process_time() - STARTED_CPU,
            "worker_cpu_seconds": budget.worker_cpu,
            "total_cpu_seconds": budget.total_cpu(),
            "wall_seconds": time.perf_counter() - STARTED_WALL,
            "adam_calls": budget.adam_calls,
            "max_live_product_bytes": budget.max_products,
            "cumulative_product_bytes": budget.cumulative_products,
            "peak_ram_mb": budget.peak_ram_mb,
        }
        check(
            checks,
            "price_cost_within_budget",
            budget.total_cpu() <= PRICE_CPU_SECONDS
            and time.perf_counter() - STARTED_WALL <= PRICE_WALL_SECONDS
            and budget.adam_calls <= PRICE_ADAM_CALLS,
            report["cost"],
        )
        report["type_check"] = run_type_check()
        check(checks, "type_check", report["type_check"]["passed"], report["type_check"])
        report["lint"] = run_lint()
        check(checks, "lint", report["lint"]["passed"], report["lint"])

        manifest_data = build_price_manifest(
            parent,
            parent_manifest,
            graph_manifest,
            graph_report,
            _candidate_for_manifest(parent),
            data,
            prices,
            extrapolation,
            source_hashes,
            graph_manifest["source_hashes"],
        )
        if manifest_path.exists():
            stored = json.loads(manifest_path.read_text(encoding="utf-8"))
            drift = {
                "candidate_id": stored.get("candidate", {}).get("candidate_id")
                != manifest_data["candidate"]["candidate_id"],
                "prices": stored.get("prices") != manifest_data["prices"],
                "source_hashes": stored.get("source_hashes") != manifest_data["source_hashes"],
            }
            manifest_ok = not any(drift.values())
            manifest_written = False
        else:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(manifest_data, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            drift = {}
            manifest_ok = True
            manifest_written = True
        report["manifest"] = {
            "path": (
                str(manifest_path.relative_to(ROOT))
                if manifest_path.is_relative_to(ROOT)
                else str(manifest_path)
            ),
            "sha256": sha256_text(manifest_path),
            "written": manifest_written,
            "drift": drift,
        }
        check(
            checks, "manifest_identity", manifest_ok, {"written": manifest_written, "drift": drift}
        )
        if not extrapolation["admitted"]:
            status_override = "budget_not_admitted"
    except PriceStop as exc:
        error = f"PriceStop: {exc}"
        status_override = "budget_stop"
    except InstrumentError as exc:
        error = f"InstrumentError: {exc}"
        status_override = "instrument_invalid"
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        status_override = "engineering_failure"
    finally:
        for path in output_dir.glob("*"):
            with contextlib.suppress(OSError):
                path.unlink()
        if not report.get("cost"):
            report["cost"] = {
                "main_cpu_seconds": time.process_time() - STARTED_CPU,
                "worker_cpu_seconds": budget.worker_cpu,
                "total_cpu_seconds": budget.total_cpu(),
                "wall_seconds": time.perf_counter() - STARTED_WALL,
                "adam_calls": budget.adam_calls,
                "max_live_product_bytes": budget.max_products,
                "cumulative_product_bytes": budget.cumulative_products,
                "peak_ram_mb": budget.peak_ram_mb,
            }
        if error is not None or any(
            not row["passed"] for row in checks if row.get("required", True)
        ):
            passed = False
        else:
            passed = True
        report["error"] = error
        report["required_checks"] = sum(1 for row in checks if row.get("required", True))
        report["failed_required"] = [
            row["id"] for row in checks if row.get("required", True) and not row["passed"]
        ]
        report["passed"] = passed
        if status_override:
            report["status"] = status_override
        elif passed:
            report["status"] = (
                "priced_within_compare_cap"
                if report["extrapolation"].get("admitted")
                else "budget_not_admitted"
            )
        else:
            report["status"] = "engineering_failure"
        info = write_json(report_path, report)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "passed": report["passed"],
                    "failed_required": report["failed_required"],
                    "admitted": report["extrapolation"].get("admitted"),
                    "prices": report.get("prices"),
                    "total_cpu_estimate": report["extrapolation"]
                    .get("components", {})
                    .get("total_cpu"),
                    "cost": report.get("cost"),
                    "report": info["path"],
                },
                ensure_ascii=False,
            )
        )
    return 0 if report["passed"] else 1


def asdict_config(config) -> dict:
    return {
        "width": config.width,
        "key_width": config.key_width,
        "hidden": config.hidden,
        "chunk": config.chunk,
        "seed": config.seed,
    }


def _candidate_for_manifest(parent):
    return build_candidate(parent, dtype=torch.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="price")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--payload-dir", type=Path)
    args = parser.parse_args()
    if args.stage != "price":
        raise SystemExit("verify_taiji_record_relation_14_5 only implements the price stage")
    return price_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
