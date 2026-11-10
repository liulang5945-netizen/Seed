"""14.6.0 checkpoint preflight for the record-relation native Chinese comparison.

This package builds the full 192 train / 160 dev composition split, two
incremental module seeds x three structural arms, and verifies that every
candidate can be saved and restored in an independent process before the
formal 14.6 run:

* step-0 save / independent probe;
* one batch-8 engineering update, save, independent probe;
* the next batch-8 update on both the original and the restored side, then
  bitwise comparison of parameters, Adam moments and the post-update
  distribution.

The preflight does not start the formal 1,800-update run, does not evaluate
task capability and does not read the sealed confirm split.  All engineering
parameter copies and bundles are deleted after measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import random
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

from taiji.candidate_manifest import build_manifest, parameter_digest  # noqa: E402
from taiji.core_09_bundle import BUNDLE_RELATIVE_PATH as CORE_09_BUNDLE_PATH  # noqa: E402
from taiji.core_09_bundle import EXPECTED_BYTES as CORE_09_BUNDLE_BYTES  # noqa: E402
from taiji.core_09_bundle import EXPECTED_SHA256 as CORE_09_BUNDLE_SHA  # noqa: E402
from taiji.living_core import EOS  # noqa: E402
from taiji.native_composition_fixture import (  # noqa: E402
    build_split,
    fixture_digest,
    fixture_payload,
)
from taiji.record_relation import RecordRelationConfig  # noqa: E402
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
REPORT = "reports/taiji_core_record_relation_14_6_preflight_20261102.json"
MANIFEST = "plans/manifests/taiji_core_record_relation_14_6_preflight_20261102.json"
OUTPUT_DIR = "output/taiji_core_record_relation_14_6_preflight"

PREFLIGHT_CPU_SECONDS = 300
PREFLIGHT_WALL_SECONDS = 600
PREFLIGHT_PRODUCT_BYTES = 1 * 1024**3
ENGINEERING_ADAM_CALLS = 18
FORMAL_UPDATES = 1800
FORMAL_ADAM_CALLS = 1800
FORMAL_CPU_SECONDS = 14400
FORMAL_WALL_SECONDS = 28800
BATCH = 8
SEEDS = (27031, 27032)
ORDER_SEEDS = (37031, 37032)
WORKER_TIMEOUT = 240

SOURCE_FILES = (
    "taiji/record_relation.py",
    "taiji/record_relation_core.py",
    "taiji/native_composition_fixture.py",
    "scripts/training/verify_taiji_record_relation_14_6_preflight.py",
)

STARTED_WALL = time.perf_counter()
STARTED_CPU = time.process_time()


class PreflightStop(RuntimeError):
    """The declared preflight budget was exhausted."""


class InstrumentError(RuntimeError):
    """A parent, data, seed or candidate identity invariant failed."""


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


def example_fixture(example, name: str) -> dict:
    target = tuple(example.action.encode()) + (EOS,)
    if len(example.action.encode()) != 27 or len(target) != 28:
        raise InstrumentError("The composition fixture is not 27 bytes plus EOS")
    return {
        "name": name,
        "index": example.index,
        "world": example.world_index,
        "layer": example.layer,
        "goal": example.goal,
        "records": example.records,
        "target": target,
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
    if len(train) != 192 or len(dev) != 160:
        raise InstrumentError(f"train/dev counts are {len(train)}/{len(dev)}, expected 192/160")
    for fixture in (*train, *dev):
        if len(fixture["goal"].encode()) != 36:
            raise InstrumentError("A composition goal is not 36 bytes")
        if not fixture["records"]:
            raise InstrumentError("A composition example has no records")
    return train_worlds, dev_worlds, train, dev


def data_identity(train_worlds, dev_worlds) -> dict:
    return {
        "schema": "taiji-composition-fixture-v1",
        "train_worlds": len(train_worlds),
        "dev_worlds": len(dev_worlds),
        "train_examples": sum(len(world.examples) for world in train_worlds),
        "dev_examples": sum(len(world.examples) for world in dev_worlds),
        "digest": fixture_digest(tuple(train_worlds) + tuple(dev_worlds)),
        "payload": fixture_payload(tuple(train_worlds) + tuple(dev_worlds)),
        "confirm_read": False,
    }


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


def batch_order(length: int, seed: int) -> list[int]:
    order = list(range(length))
    random.Random(seed).shuffle(order)
    return order


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
    mode: str,
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
        "mode": mode,
        "previous_byte": previous_byte,
        "fixture_index": 0,
        "force": None,
        "gamma_override": None,
        "gate_index": gate_index,
    }


class PreflightBudget:
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
        if self.total_cpu() > PREFLIGHT_CPU_SECONDS:
            raise PreflightStop(f"CPU budget exceeded: {self.total_cpu():.2f}")
        wall = time.perf_counter() - STARTED_WALL
        if wall > PREFLIGHT_WALL_SECONDS:
            raise PreflightStop(f"wall budget exceeded: {wall:.2f}")
        if self.adam_calls > ENGINEERING_ADAM_CALLS:
            raise PreflightStop(f"engineering Adam budget exceeded: {self.adam_calls}")
        if self.max_products > PREFLIGHT_PRODUCT_BYTES:
            raise PreflightStop(f"product budget exceeded: {self.max_products}")

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


def relation_digest(candidate) -> str:
    return str(
        parameter_digest(
            {
                name: parameter.detach().cpu()
                for name, parameter in candidate.relation.named_parameters()
            }
        )
    )


def preflight_candidate(
    parent: dict,
    *,
    seed: int,
    arm: str,
    train: tuple[dict, ...],
    dev0: dict,
    order_seed: int,
    output_dir: Path,
    budget: PreflightBudget,
) -> dict:
    config = RecordRelationConfig(seed=seed)
    candidate = build_candidate(parent, arm=arm, config=config, dtype=torch.float64)
    optimizer = make_optimizer(candidate)
    state = candidate.subject.initial_state(f"preflight-{seed}-{arm}", revision=CONTENT_REVISION)
    prepared = prepare(candidate, state, dev0["goal"].encode(), dev0["records"])
    observation, gate_index = observe_goal(candidate, prepared)
    probe0 = answer_step(
        candidate,
        prepared,
        observation,
        observation.coverage,
        EOS,
        gate_index=gate_index,
    )
    bundle0 = output_dir / f"{seed}_{arm}_step0.pt"
    result0 = output_dir / f"{seed}_{arm}_step0_probe.pt"
    info0 = save_bundle(
        bundle0,
        profile=f"{seed}-{arm}-step0",
        candidate=candidate,
        state=observation,
        optimizer=optimizer,
        update_index=0,
        cursor={"seed": seed, "arm": arm},
        parent_identity={"candidate": candidate_digest(candidate)},
    )
    budget.add_product(int(info0["bytes"]))
    worker0 = spawn_worker(
        worker_spec(
            bundle0,
            result0,
            (dev0,),
            mode="probe",
            previous_byte=EOS,
            gate_index=gate_index,
        )
    )
    budget.worker_cpu += float(worker0.get("cpu_seconds", 0.0))
    budget.wall += float(worker0.get("wall_seconds", 0.0))
    step0_restore = {
        "ok": bool(worker0.get("ok")),
        "equal": bool(worker0.get("ok"))
        and bool(torch.equal(probe0.probabilities, worker0["probabilities"])),
        "max_abs_diff": (
            None
            if not worker0.get("ok")
            else max_abs_diff(probe0.probabilities, worker0["probabilities"])
        ),
    }

    order = batch_order(len(train), order_seed)
    batch = (dev0,) + tuple(train[index] for index in order[: BATCH - 1])
    if len(batch) != BATCH:
        raise InstrumentError("The preflight batch is not batch-8")
    loss1 = batch_loss(candidate, batch)
    optimizer.zero_grad()
    loss1.backward()
    clip1 = float(torch.nn.utils.clip_grad_norm_(candidate.trainable_parameters(), CLIP_NORM))
    optimizer.step()
    budget.adam_calls += 1
    budget.ensure()

    prepared1 = prepare(candidate, observation, dev0["goal"].encode(), dev0["records"])
    probe1 = answer_step(
        candidate,
        prepared1,
        observation,
        observation.coverage,
        EOS,
        gate_index=gate_index,
    )
    bundle1 = output_dir / f"{seed}_{arm}_step1.pt"
    result1 = output_dir / f"{seed}_{arm}_step1_probe.pt"
    info1 = save_bundle(
        bundle1,
        profile=f"{seed}-{arm}-step1",
        candidate=candidate,
        state=observation,
        optimizer=optimizer,
        update_index=1,
        cursor={"seed": seed, "arm": arm},
        parent_identity={"candidate": candidate_digest(candidate)},
    )
    budget.add_product(int(info1["bytes"]))
    worker1 = spawn_worker(
        worker_spec(
            bundle1,
            result1,
            batch,
            mode="update",
            previous_byte=EOS,
            gate_index=gate_index,
        )
    )
    budget.worker_cpu += float(worker1.get("cpu_seconds", 0.0))
    budget.wall += float(worker1.get("wall_seconds", 0.0))
    budget.adam_calls += int(worker1.get("adam_calls", 0))
    step1_restore = {
        "ok": bool(worker1.get("ok")),
        "equal": bool(worker1.get("ok"))
        and bool(torch.equal(probe1.probabilities, worker1["probabilities"])),
        "max_abs_diff": (
            None
            if not worker1.get("ok")
            else max_abs_diff(probe1.probabilities, worker1["probabilities"])
        ),
    }

    loss2 = batch_loss(candidate, batch)
    optimizer.zero_grad()
    loss2.backward()
    clip2 = float(torch.nn.utils.clip_grad_norm_(candidate.trainable_parameters(), CLIP_NORM))
    optimizer.step()
    budget.adam_calls += 1
    budget.ensure()

    post = score_example(
        candidate,
        dev0["goal"].encode(),
        dev0["records"],
        tuple(dev0["target"][:-1]),
    )
    post_probe = answer_step(
        candidate,
        post.prepared,
        post.prefix_state,
        post.prefix_coverage,
        dev0["target"][-2],
        gate_index=0,
    )
    if worker1.get("ok"):
        params_equal = tree_equal(dict(candidate.named_parameters()), worker1["parameters"])
        optimizer_equal = tree_equal(optimizer.state_dict(), worker1["optimizer"])
        post_equal = bool(torch.equal(post_probe.probabilities, worker1["post_probabilities"]))
        post_diff = max_abs_diff(post_probe.probabilities, worker1["post_probabilities"])
        worker_loss = float(worker1["loss"])
    else:
        params_equal = optimizer_equal = post_equal = False
        post_diff = None
        worker_loss = None

    for info in (info0, info1):
        budget.drop_product(int(info["bytes"]))
    for path in (bundle0, bundle1, result0, result1):
        if path.exists():
            path.unlink()
    budget.sample_ram()
    return {
        "seed": seed,
        "arm": arm,
        "candidate_id": candidate_digest(candidate),
        "relation_digest": relation_digest(candidate),
        "parameter_count": candidate.parameter_count(),
        "trainable_count": int(
            sum(parameter.numel() for parameter in candidate.trainable_parameters())
        ),
        "disabled": candidate.disabled_rows(),
        "batch_indices": [fixture["index"] for fixture in batch],
        "batch_includes_dev0": True,
        "losses": {
            "step1": float(loss1.detach()),
            "step2": float(loss2.detach()),
            "worker_step2": worker_loss,
        },
        "clip_scale": {"step1": clip1, "step2": clip2},
        "step0_restore": step0_restore,
        "step1_restore": step1_restore,
        "next_update": {
            "ok": bool(worker1.get("ok")),
            "parameters_equal": params_equal,
            "optimizer_equal": optimizer_equal,
            "post_probabilities_equal": post_equal,
            "post_max_abs_diff": post_diff,
        },
        "engineering_adam_calls": 2,
    }


def run_type_check() -> dict:
    command = [
        sys.executable,
        "-m",
        "mypy",
        "--follow-imports=skip",
        "--ignore-missing-imports",
        *SOURCE_FILES,
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


def preflight_main(args) -> int:
    report_path = Path(args.report) if args.report else ROOT / REPORT
    manifest_path = Path(args.manifest) if args.manifest else ROOT / MANIFEST
    output_dir = Path(args.payload_dir) if args.payload_dir else ROOT / OUTPUT_DIR
    if report_path.exists():
        raise FileExistsError("14.6.0 preflight card is write-once")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    budget = PreflightBudget()
    checks: list[dict] = []
    report: dict[str, Any] = {
        "schema": "taiji-record-relation-14-6-preflight-v1",
        "step": "14.6.0",
        "date": "2026-11-02",
        "status": "engineering_failure",
        "checks": checks,
        "identity": {},
        "data": {},
        "seeds": [],
        "preflight": [],
        "formal_run": {"started": False, "formal_adam_calls": 0},
        "cost": {},
        "type_check": {},
        "lint": {},
        "limits": (
            "Checkpoint preflight only. The formal 14.6 1,800-update run is not started; "
            "no task capability claim, no confirm read, no retained engineering parameters."
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
            parent_manifest["candidate_id"] != EXPECTED_PARENT
            or parent_manifest["provenance"]["sha256"] != EXPECTED_BUNDLE_SHA
            or parent_manifest["provenance"]["bytes"] != EXPECTED_BUNDLE_BYTES
        ):
            raise InstrumentError("Parent identity drifted from the fixed 09.8 asset")
        graph_report = json.loads((ROOT / GRAPH_REPORT).read_text(encoding="utf-8"))
        graph_manifest = json.loads((ROOT / GRAPH_MANIFEST).read_text(encoding="utf-8"))
        price_report = json.loads((ROOT / PRICE_REPORT).read_text(encoding="utf-8"))
        price_manifest = json.loads((ROOT / PRICE_MANIFEST).read_text(encoding="utf-8"))
        if graph_report["status"] != "passed_record_relation_graph_engineering_only":
            raise InstrumentError("14.4 did not pass the engineering gate")
        if price_report["status"] != "priced_within_compare_cap":
            raise InstrumentError("14.5 pricing did not admit the comparison cap")
        if not price_report.get("extrapolation", {}).get("admitted", False):
            raise InstrumentError("14.5 price did not fit the 14.6 cap")
        if graph_manifest["candidate"]["candidate_id"] != EXPECTED_14_4_CANDIDATE:
            raise InstrumentError("14.4 candidate identity drifted")
        identity = {
            "parent_candidate_id": parent_manifest["candidate_id"],
            "parent_bundle_sha256": parent_manifest["provenance"]["sha256"],
            "parent_bundle_bytes": parent_manifest["provenance"]["bytes"],
            "graph_report": GRAPH_REPORT,
            "graph_report_sha256": sha256_text(ROOT / GRAPH_REPORT),
            "graph_candidate_id": graph_manifest["candidate"]["candidate_id"],
            "price_report": PRICE_REPORT,
            "price_report_sha256": sha256_text(ROOT / PRICE_REPORT),
            "price_candidate_id": price_manifest["candidate"]["candidate_id"],
            "price_manifest": PRICE_MANIFEST,
            "price_manifest_sha256": sha256_text(ROOT / PRICE_MANIFEST),
            "frozen_train_update_batch8_cpu": price_report["prices"]["train_update_batch8_cpu"],
            "formal_estimate_cpu": price_report["extrapolation"]["components"]["total_cpu"],
            "source_hashes": source_hashes,
            "graph_source_hashes": graph_manifest["source_hashes"],
            "price_source_hashes": price_manifest["source_hashes"],
        }
        report["identity"] = identity
        check(
            checks,
            "parent_identity",
            True,
            {
                "candidate_id": parent_manifest["candidate_id"],
                "sha256": parent_manifest["provenance"]["sha256"],
            },
        )
        check(
            checks,
            "gate_chain",
            graph_report["status"] == "passed_record_relation_graph_engineering_only"
            and price_report["status"] == "priced_within_compare_cap"
            and price_report["extrapolation"]["admitted"],
            {
                "graph": graph_report["status"],
                "price": price_report["status"],
                "admitted": price_report["extrapolation"]["admitted"],
            },
        )

        train_worlds, dev_worlds, train, dev = build_full_data()
        data = data_identity(train_worlds, dev_worlds)
        report["data"] = data
        check(
            checks,
            "data_contract",
            len(train) == 192
            and len(dev) == 160
            and all(len(fixture["target"]) == 28 for fixture in (*train, *dev))
            and all(len(fixture["target"]) - 1 == 27 for fixture in (*train, *dev))
            and data["confirm_read"] is False,
            {
                "train": len(train),
                "dev": len(dev),
                "digest": data["digest"],
            },
        )

        seed_rows = []
        candidate_rows: dict[tuple[int, str], dict] = {}
        for seed in SEEDS:
            relation_digests = {}
            for arm in ARMS:
                candidate = build_candidate(
                    parent, arm=arm, config=RecordRelationConfig(seed=seed), dtype=torch.float64
                )
                candidate_rows[(seed, arm)] = {
                    "candidate_id": candidate_digest(candidate),
                    "relation_digest": relation_digest(candidate),
                    "new_parameters": candidate.new_parameter_count(),
                    "disabled": candidate.disabled_rows()["elements"],
                }
                relation_digests[arm] = relation_digest(candidate)
                seed_rows.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "candidate_id": candidate_digest(candidate),
                        "relation_digest": relation_digest(candidate),
                        "new_parameters": candidate.new_parameter_count(),
                        "disabled": candidate.disabled_rows()["elements"],
                    }
                )
                del candidate
        report["seeds"] = seed_rows
        a1 = candidate_rows[(SEEDS[0], ARM_FULL)]
        a2 = candidate_rows[(SEEDS[1], ARM_FULL)]
        check(
            checks,
            "seed_identity",
            a1["candidate_id"] == EXPECTED_14_4_CANDIDATE
            and a1["relation_digest"] != a2["relation_digest"]
            and a1["candidate_id"] != a2["candidate_id"],
            {
                "seed_27031": a1,
                "seed_27032": a2,
            },
        )
        check(
            checks,
            "arm_structure",
            all(
                candidate_rows[(seed, arm)]["new_parameters"] == 172801
                for seed in SEEDS
                for arm in ARMS
            )
            and all(
                candidate_rows[(seed, ARM_FULL)]["disabled"] == 0
                and candidate_rows[(seed, ARM_LOCAL)]["disabled"] == 91008
                and candidate_rows[(seed, ARM_FLAT)]["disabled"] == 9504
                for seed in SEEDS
            ),
            {
                str(seed): {arm: candidate_rows[(seed, arm)]["disabled"] for arm in ARMS}
                for seed in SEEDS
            },
        )
        check(
            checks,
            "formal_budget_alignment",
            price_report["extrapolation"]["within"]["cpu"]
            and price_report["extrapolation"]["within"]["wall"]
            and price_report["extrapolation"]["within"]["adam"]
            and price_report["extrapolation"]["within"]["products"]
            and price_report["extrapolation"]["components"]["total_cpu"] <= FORMAL_CPU_SECONDS,
            {
                "formal_updates": FORMAL_UPDATES,
                "formal_adam": FORMAL_ADAM_CALLS,
                "formal_estimate_cpu": price_report["extrapolation"]["components"]["total_cpu"],
                "cap": FORMAL_CPU_SECONDS,
            },
        )

        preflight_rows = []
        for seed, order_seed in zip(SEEDS, ORDER_SEEDS, strict=True):
            for arm in ARMS:
                row = preflight_candidate(
                    parent,
                    seed=seed,
                    arm=arm,
                    train=train,
                    dev0=dev[0],
                    order_seed=order_seed,
                    output_dir=output_dir,
                    budget=budget,
                )
                preflight_rows.append(row)
                report["preflight"].append(row)
                budget.ensure()
        if budget.adam_calls != ENGINEERING_ADAM_CALLS:
            raise InstrumentError(
                f"expected {ENGINEERING_ADAM_CALLS} engineering Adam calls, observed {budget.adam_calls}"
            )
        check(
            checks,
            "checkpoint_restore_chain",
            all(
                row["step0_restore"]["equal"]
                and row["step1_restore"]["equal"]
                and row["next_update"]["parameters_equal"]
                and row["next_update"]["optimizer_equal"]
                and row["next_update"]["post_probabilities_equal"]
                for row in preflight_rows
            ),
            [
                {
                    "seed": row["seed"],
                    "arm": row["arm"],
                    "step0": row["step0_restore"]["equal"],
                    "step1": row["step1_restore"]["equal"],
                    "parameters": row["next_update"]["parameters_equal"],
                    "optimizer": row["next_update"]["optimizer_equal"],
                    "post": row["next_update"]["post_probabilities_equal"],
                }
                for row in preflight_rows
            ],
        )
        check(
            checks,
            "engineering_update_sane",
            all(
                torch.isfinite(torch.tensor(row["losses"]["step1"]))
                and torch.isfinite(torch.tensor(row["losses"]["step2"]))
                for row in preflight_rows
            ),
            [
                {
                    "seed": row["seed"],
                    "arm": row["arm"],
                    "step1": row["losses"]["step1"],
                    "step2": row["losses"]["step2"],
                    "worker": row["losses"]["worker_step2"],
                }
                for row in preflight_rows
            ],
        )
        check(
            checks,
            "engineering_adam_separate_from_formal",
            budget.adam_calls == ENGINEERING_ADAM_CALLS
            and report["formal_run"]["formal_adam_calls"] == 0,
            {
                "engineering_adam": budget.adam_calls,
                "formal_adam": report["formal_run"]["formal_adam_calls"],
            },
        )

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
            budget.max_products <= PREFLIGHT_PRODUCT_BYTES,
            {
                "max_live_bytes": budget.max_products,
                "cumulative_bytes": budget.cumulative_products,
                "cap": PREFLIGHT_PRODUCT_BYTES,
            },
        )
        report["cost"] = {
            "main_cpu_seconds": time.process_time() - STARTED_CPU,
            "worker_cpu_seconds": budget.worker_cpu,
            "total_cpu_seconds": budget.total_cpu(),
            "wall_seconds": time.perf_counter() - STARTED_WALL,
            "engineering_adam_calls": budget.adam_calls,
            "max_live_product_bytes": budget.max_products,
            "cumulative_product_bytes": budget.cumulative_products,
            "peak_ram_mb": budget.peak_ram_mb,
        }
        check(
            checks,
            "preflight_cost_within_budget",
            budget.total_cpu() <= PREFLIGHT_CPU_SECONDS
            and time.perf_counter() - STARTED_WALL <= PREFLIGHT_WALL_SECONDS
            and budget.adam_calls <= ENGINEERING_ADAM_CALLS,
            report["cost"],
        )
        report["type_check"] = run_type_check()
        check(checks, "type_check", report["type_check"]["passed"], report["type_check"])
        report["lint"] = run_lint()
        check(checks, "lint", report["lint"]["passed"], report["lint"])

        manifest_data: dict[str, Any] = {
            "schema": "taiji-record-relation-14-6-preflight-manifest-v1",
            "step": "14.6.0",
            "date": "2026-11-02",
            "parent": {
                "candidate_id": parent_manifest["candidate_id"],
                "bundle": BUNDLE,
                "sha256": parent_manifest["provenance"]["sha256"],
                "bytes": parent_manifest["provenance"]["bytes"],
            },
            "gate_chain": {
                "graph": {
                    "report": GRAPH_REPORT,
                    "sha256": sha256_text(ROOT / GRAPH_REPORT),
                    "candidate_id": graph_manifest["candidate"]["candidate_id"],
                },
                "price": {
                    "report": PRICE_REPORT,
                    "sha256": sha256_text(ROOT / PRICE_REPORT),
                    "manifest": PRICE_MANIFEST,
                    "manifest_sha256": sha256_text(ROOT / PRICE_MANIFEST),
                    "candidate_id": price_manifest["candidate"]["candidate_id"],
                    "train_update_batch8_cpu": price_report["prices"]["train_update_batch8_cpu"],
                    "formal_estimate_cpu": price_report["extrapolation"]["components"]["total_cpu"],
                    "admitted": price_report["extrapolation"]["admitted"],
                },
            },
            "data": data,
            "seeds": seed_rows,
            "preflight": preflight_rows,
            "formal_run": {
                "started": False,
                "formal_updates": FORMAL_UPDATES,
                "formal_adam_calls": FORMAL_ADAM_CALLS,
                "formal_estimate_cpu": price_report["extrapolation"]["components"]["total_cpu"],
                "cap_cpu": FORMAL_CPU_SECONDS,
                "cap_wall": FORMAL_WALL_SECONDS,
            },
            "source_hashes": source_hashes,
            "pricing_parameters_discarded": True,
            "retained_parameters": [],
            "limits": (
                "Checkpoint preflight only. The formal 14.6 run is not started; no task "
                "capability claim, no confirm read, no retained engineering parameters."
            ),
        }
        if manifest_path.exists():
            stored = json.loads(manifest_path.read_text(encoding="utf-8"))
            drift = {
                "candidate_id": stored.get("gate_chain", {}).get("graph", {}).get("candidate_id")
                != manifest_data["gate_chain"]["graph"]["candidate_id"],
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
    except PreflightStop as exc:
        error = f"PreflightStop: {exc}"
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
                "engineering_adam_calls": budget.adam_calls,
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
            report["status"] = "passed_14_6_checkpoint_preflight"
        else:
            report["status"] = "engineering_failure"
        info = write_json(report_path, report)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "passed": report["passed"],
                    "failed_required": report["failed_required"],
                    "engineering_adam": budget.adam_calls,
                    "formal_started": False,
                    "cost": report.get("cost"),
                    "report": info["path"],
                },
                ensure_ascii=False,
            )
        )
    return 0 if report["passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="preflight")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--payload-dir", type=Path)
    args = parser.parse_args()
    if args.stage != "preflight":
        raise SystemExit("14.6.0 only implements the checkpoint preflight")
    return preflight_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
