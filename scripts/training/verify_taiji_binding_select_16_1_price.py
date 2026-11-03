#!/usr/bin/env python
"""N16.1.0c pricing for the selection-locked candidate.

Measures the three training stages (selection / copy / joint), the locked
decoder, evaluation and restore, then extrapolates the formal comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

torch.set_num_threads(1)

from taiji.binding_select import (  # noqa: E402
    SelectionConfig,
    build_selection_candidate,
    load_bundle,
    locked_generate,
    locked_loss,
    make_selection_optimizer,
    save_bundle,
    select,
    selection_ce,
)
from taiji.living_core import RawSource  # noqa: E402
from taiji.record_relation import RecordRelationConfig  # noqa: E402
from taiji.record_relation_core import (  # noqa: E402
    ARM_FULL,
    CLIP_NORM,
    RecordRelationCandidate,
    build_subject_from_candidate_bundle,
)

PARENT_BUNDLE = "output/taiji_core_record_relation_14_6/27031_A_full_step300.pt"
PARENT_SHA256 = "b75a83a0a474c041c2a61bed8c4e0a51de11197d67a9b196262e7e039a9a4cef"
SELECT_PAYLOAD = "output/taiji_core_binding_select_16_1/select_v1.json"
DATA_PAYLOAD = "output/taiji_core_binding_copy_15_1/data_v2.json"
REPORT = "reports/taiji_binding_select_16_1_price_20261103.json"
MANIFEST = "plans/manifests/taiji_binding_select_16_1_price_20261103.json"
OUTPUT_DIR = Path("output/taiji_core_binding_select_16_1_price")

STAGE_CPU_SECONDS = 600.0
STAGE_WALL_SECONDS = 1200.0
STAGE_ADAM_CALLS = 12
STAGE_PRODUCT_BYTES = 1 * 1024**3

FORMAL_SEEDS = 2
FORMAL_ARMS = 2
FORMAL_UPDATES = 300
STAGE_UPDATES = {"selection": 100, "copy": 100, "joint": 100}
DEV_EXAMPLES = 240
EVAL_ENDPOINTS = 2
OVERHEAD_FRACTION = 0.10
STARTUP_WALL_SECONDS = 60.0
WATCH_CPU_SECONDS = 20000.0
WATCH_WALL_SECONDS = 30000.0
BATCH = 8

SOURCE_FILES = (
    "taiji/binding_select.py",
    "scripts/training/verify_taiji_binding_select_16_1_price.py",
)

STARTED_CPU = time.process_time()
STARTED_WALL = time.perf_counter()


class PriceStop(RuntimeError):
    pass


class Budget:
    def __init__(self) -> None:
        self.adam = 0
        self.live = 0
        self.max_live = 0
        self.cumulative = 0

    def add_product(self, info: dict) -> None:
        size = int(info.get("bytes", 0))
        self.live += size
        self.cumulative += size
        self.max_live = max(self.max_live, size + self.live)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(split: str) -> list[dict]:
    payload = json.loads(Path(SELECT_PAYLOAD).read_text(encoding="utf-8"))
    return payload[split]


def fixtures() -> list[dict]:
    data = json.loads(Path(DATA_PAYLOAD).read_text(encoding="utf-8"))
    rows = load_rows("train")
    out = []
    for row in rows:
        example = next(e for e in data["examples"] if e["index"] == row["index"])
        out.append(
            {
                "index": int(row["index"]),
                "goal": row["goal"],
                "record_keys": row["record_keys"],
                "selection_label": int(row["selection_label"]),
                "records": tuple(
                    RawSource(source_id=r["source_id"], raw=r["raw"].encode("utf-8"))
                    for r in example["records"]
                ),
                "target_prefix": tuple(example["action"].encode("utf-8")),
            }
        )
    return out


def build(payload: dict):
    config = RecordRelationConfig(**payload["relation_config"])
    subject = build_subject_from_candidate_bundle(payload)
    base = RecordRelationCandidate(subject, config, ARM_FULL, list(payload["growth_gates"]))
    named = dict(base.named_parameters())
    for name, value in payload["parameters"].items():
        if name.startswith("relation."):
            named[name].data.copy_(value)
    base.to(torch.float64)
    base.configure_trainable()
    candidate = build_selection_candidate(
        base, profile="price-f64", dtype=torch.float64, selection_config=SelectionConfig()
    )
    candidate.configure_trainable()
    return candidate, make_selection_optimizer(candidate)


def batch_from(rows: list[dict], start: int) -> list[dict]:
    return rows[start % len(rows) : start % len(rows) + BATCH]


def measure_select(candidate, items: list[dict]) -> float:
    start = time.process_time()
    for item in items:
        select(candidate, item["goal"].encode(), item["record_keys"], label=item["selection_label"])
    return (time.process_time() - start) / len(items)


def measure_generate(candidate, items: list[dict]) -> float:
    start = time.process_time()
    with torch.no_grad():
        for item in items:
            locked_generate(
                candidate,
                item["goal"].encode(),
                item["records"],
                item["record_keys"],
                subject_id=f"price-gen-{item['index']}",
            )
    return (time.process_time() - start) / len(items)


def measure_update(candidate, optimizer, items: list[dict], *, stage: str) -> dict:
    start = time.process_time()
    losses = []
    for item in items:
        ce = selection_ce(
            candidate, item["goal"].encode(), item["record_keys"], item["selection_label"]
        )
        if stage == "selection":
            loss = ce
        else:
            locked = locked_loss(
                candidate,
                item["goal"].encode(),
                item["records"],
                item["record_keys"],
                item["target_prefix"],
                subject_id=f"price-{stage}-{item['index']}",
            )
            loss = locked if stage == "copy" else locked + ce
        losses.append(loss)
    loss = sum(losses) / len(losses)
    forward = time.process_time() - start
    start = time.process_time()
    optimizer.zero_grad()
    loss.backward()
    bptt = time.process_time() - start
    start = time.process_time()
    clip = float(torch.nn.utils.clip_grad_norm_(list(candidate.trainable_parameters()), CLIP_NORM))
    optimizer.step()
    step = time.process_time() - start
    return {
        "stage": stage,
        "loss": float(loss.detach()),
        "forward_cpu": forward,
        "bptt_cpu": bptt,
        "step_cpu": step,
        "total_cpu": forward + bptt + step,
        "clip_scale": clip,
    }


def measure_restore(candidate, optimizer, budget: Budget) -> dict:
    from taiji.binding_copy import initial_binding_state

    state = initial_binding_state(candidate.binding, "price-restore")
    path = OUTPUT_DIR / "price-restore.pt"
    start = time.process_time()
    info = save_bundle(
        path,
        profile="price-f64",
        candidate=candidate,
        state=state,
        optimizer=optimizer,
        update_index=0,
        cursor={"tag": "price"},
        parent_identity={"sha256": PARENT_SHA256},
    )
    save_cpu = time.process_time() - start
    budget.add_product(info)
    start = time.process_time()
    loaded = load_bundle(path, torch.float64)
    load_cpu = time.process_time() - start
    del loaded
    path.unlink(missing_ok=True)
    return {
        "save_cpu": save_cpu,
        "load_cpu": load_cpu,
        "elapsed_cpu": save_cpu + load_cpu,
        "bundle_mib": info["bytes"] / (1024 * 1024),
    }


def extrapolate(prices: dict) -> dict:
    updates = FORMAL_SEEDS * FORMAL_ARMS * FORMAL_UPDATES
    weights = {stage: STAGE_UPDATES[stage] / sum(STAGE_UPDATES.values()) for stage in STAGE_UPDATES}
    per_update = sum(prices[f"update_{stage}_cpu"] * weights[stage] for stage in STAGE_UPDATES)
    training_cpu = per_update * updates
    generations = FORMAL_SEEDS * FORMAL_ARMS * DEV_EXAMPLES * EVAL_ENDPOINTS
    generation_cpu = prices["locked_generate_cpu"] * generations
    selection_cpu = (
        prices["select_cpu"] * FORMAL_SEEDS * FORMAL_ARMS * DEV_EXAMPLES * EVAL_ENDPOINTS
    )
    restore_cpu = prices["restore_cpu"] * FORMAL_SEEDS * FORMAL_ARMS * 2
    subtotal = training_cpu + generation_cpu + selection_cpu + restore_cpu
    overhead = subtotal * OVERHEAD_FRACTION
    total = subtotal + overhead
    return {
        "formula": {
            "formal_updates": updates,
            "stage_weights": weights,
            "generations": generations,
            "eval_endpoints": EVAL_ENDPOINTS,
            "overhead_fraction": OVERHEAD_FRACTION,
            "startup_wall_seconds": STARTUP_WALL_SECONDS,
        },
        "components": {
            "training_cpu": training_cpu,
            "generation_cpu": generation_cpu,
            "selection_cpu": selection_cpu,
            "restore_cpu": restore_cpu,
            "subtotal_cpu": subtotal,
            "overhead_cpu": overhead,
            "total_cpu": total,
            "estimated_wall_seconds": total + STARTUP_WALL_SECONDS,
        },
        "watch_line": {"cpu_seconds": WATCH_CPU_SECONDS, "wall_seconds": WATCH_WALL_SECONDS},
        "within_watch_line": bool(
            total + STARTUP_WALL_SECONDS <= WATCH_WALL_SECONDS and total <= WATCH_CPU_SECONDS
        ),
        "cap": "owner sufficient budget; observation line only",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="price", choices=("price",))
    parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    budget = Budget()
    try:
        if sha256_file(ROOT / PARENT_BUNDLE) != PARENT_SHA256:
            raise SystemExit("parent bundle sha drifted")
        payload = torch.load(ROOT / PARENT_BUNDLE, map_location="cpu", weights_only=False)
        rows = fixtures()
        candidate, optimizer = build(payload)

        select_cpu = measure_select(candidate, rows[:5])
        generate_cpu = measure_generate(candidate, rows[:4])
        restore = measure_restore(candidate, optimizer, budget)

        updates = {}
        batch = batch_from(rows, 0)
        for stage in ("selection", "copy", "joint"):
            for offset in range(3):
                row = measure_update(
                    candidate, optimizer, batch_from(rows, offset * BATCH), stage=stage
                )
                budget.adam += 1
                rows_out = updates.setdefault(stage, [])
                rows_out.append(row)
                if budget.adam > STAGE_ADAM_CALLS:
                    raise PriceStop("adam cap exceeded")
                if time.process_time() - STARTED_CPU > STAGE_CPU_SECONDS:
                    raise PriceStop("cpu cap exceeded")
        del batch

        prices = {
            "select_cpu": select_cpu,
            "locked_generate_cpu": generate_cpu,
            "restore_cpu": restore["elapsed_cpu"],
            "bundle_mib": restore["bundle_mib"],
            **{
                f"update_{stage}_cpu": statistics.median(r["total_cpu"] for r in updates[stage])
                for stage in updates
            },
        }
        extrap = extrapolate(prices)
        total_cpu = time.process_time() - STARTED_CPU
        total_wall = time.perf_counter() - STARTED_WALL
        status = (
            "priced_within_watch_line" if extrap["within_watch_line"] else "priced_over_watch_line"
        )

        report = {
            "schema": "taiji-binding-select-price-v1",
            "step": "N16.1.0c",
            "date": "2026-11-03",
            "status": status,
            "prices": prices,
            "updates": updates,
            "restore": restore,
            "extrapolation": extrap,
            "cost": {
                "cpu_seconds": total_cpu,
                "wall_seconds": total_wall,
                "adam_calls": budget.adam,
                "peak_live_product_bytes": budget.max_live,
            },
            "budget": {
                "cpu_seconds": STAGE_CPU_SECONDS,
                "wall_seconds": STAGE_WALL_SECONDS,
                "adam_calls": STAGE_ADAM_CALLS,
                "product_bytes": STAGE_PRODUCT_BYTES,
            },
            "source_hashes": {path: sha256_text(ROOT / path) for path in SOURCE_FILES},
            "limits": "Pricing only. No adoption training, no confirm read, no capability claim.",
        }
        Path(REPORT).write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        manifest = {
            "schema": "taiji-binding-select-manifest-v1",
            "step": "N16.1.0c",
            "status": status,
            "report": REPORT,
            "prices": prices,
            "extrapolation": extrap,
            "cost": report["cost"],
            "source_hashes": report["source_hashes"],
        }
        Path(MANIFEST).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"16.1.0c status: {status}")
        print(f"  prices: {json.dumps(prices)}")
        print(
            f"  extrapolated cpu {extrap['components']['total_cpu']:.2f}, wall {extrap['components']['estimated_wall_seconds']:.2f}"
        )
        print(
            f"  cost cpu {total_cpu:.2f}/{STAGE_CPU_SECONDS}, adam {budget.adam}/{STAGE_ADAM_CALLS}"
        )
        return 0
    except PriceStop as exc:
        print(f"16.1.0c stop: {exc}")
        return 1
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
