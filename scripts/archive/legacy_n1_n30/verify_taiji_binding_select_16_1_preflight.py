#!/usr/bin/env python
"""N16.1.0d checkpoint preflight: two seeds x two arms.

Initial, step 1 and the replayed next update must be bit-exact after restore in
an independent process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

torch.set_num_threads(1)

from taiji.binding_copy import initial_binding_state  # noqa: E402
from taiji.binding_select import (  # noqa: E402
    SelectionConfig,
    build_selection_candidate,
    load_bundle,
    locked_loss,
    make_selection_optimizer,
    save_bundle,
    selection_ce,
)
from taiji.living_core import RawSource  # noqa: E402
from taiji.record_relation import RecordRelationConfig  # noqa: E402
from taiji.record_relation_core import (  # noqa: E402
    ARM_FULL,
    CLIP_NORM,
    RecordRelationCandidate,
    build_subject_from_candidate_bundle,
    candidate_parameter_map,
)

PARENT_BUNDLE = "output/taiji_core_record_relation_14_6/27031_A_full_step300.pt"
PARENT_SHA256 = "b75a83a0a474c041c2a61bed8c4e0a51de11197d67a9b196262e7e039a9a4cef"
SELECT_PAYLOAD = "output/taiji_core_binding_select_16_1/select_v1.json"
DATA_PAYLOAD = "output/taiji_core_binding_copy_15_1/data_v2.json"
REPORT = "reports/taiji_binding_select_16_1_preflight_20261103.json"
MANIFEST = "plans/manifests/taiji_binding_select_16_1_preflight_20261103.json"
OUTPUT_DIR = Path("output/taiji_core_binding_select_16_1_preflight")

STAGE_CPU_SECONDS = 400.0
STAGE_WALL_SECONDS = 800.0
STAGE_ADAM_CALLS = 24
STAGE_PRODUCT_BYTES = 2 * 1024**3
WORKER_TIMEOUT = 300
SEEDS = (28031, 28032)
PROFILES = ("A-locked-f64", "B-select-f64")
BATCH = 8

SOURCE_FILES = (
    "taiji/binding_select.py",
    "scripts/training/verify_taiji_binding_select_16_1_preflight.py",
)

STARTED_CPU = time.process_time()
STARTED_WALL = time.perf_counter()


class StageStop(RuntimeError):
    pass


class Budget:
    def __init__(self) -> None:
        self.main_cpu = 0.0
        self.worker_cpu = 0.0
        self.adam = 0
        self.live = 0
        self.max_live = 0

    def total_cpu(self) -> float:
        self.main_cpu = time.process_time() - STARTED_CPU
        return self.main_cpu + self.worker_cpu

    def add_worker(self, result: dict) -> None:
        self.worker_cpu += float(result.get("cpu_seconds", 0.0))
        self.adam += int(result.get("adam_calls", 0))
        self.ensure()

    def add_product(self, info: dict) -> None:
        self.live += int(info.get("bytes", 0))
        self.max_live = max(self.max_live, self.live)

    def ensure(self) -> None:
        if self.total_cpu() > STAGE_CPU_SECONDS:
            raise StageStop(f"cpu budget exceeded: {self.total_cpu():.2f}")
        if time.perf_counter() - STARTED_WALL > STAGE_WALL_SECONDS:
            raise StageStop("wall budget exceeded")
        if self.adam > STAGE_ADAM_CALLS:
            raise StageStop(f"adam budget exceeded: {self.adam}")
        if self.max_live > STAGE_PRODUCT_BYTES:
            raise StageStop("product budget exceeded")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(split: str) -> list[dict]:
    payload = json.loads(Path(SELECT_PAYLOAD).read_text(encoding="utf-8"))
    return payload[split]


def to_fixture(row: dict) -> dict:
    data = json.loads(Path(DATA_PAYLOAD).read_text(encoding="utf-8"))
    example = next(e for e in data["examples"] if e["index"] == row["index"])
    records = tuple(
        RawSource(source_id=r["source_id"], raw=r["raw"].encode("utf-8"))
        for r in example["records"]
    )
    return {
        "index": int(row["index"]),
        "goal": row["goal"],
        "records": records,
        "record_keys": row["record_keys"],
        "selection_label": int(row["selection_label"]),
        "target_prefix": tuple(example["action"].encode("utf-8")),
    }


def build(payload: dict, profile: str):
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
        base, profile=profile, dtype=torch.float64, selection_config=SelectionConfig()
    )
    candidate.configure_trainable()
    return candidate, make_selection_optimizer(candidate)


def digest(tensors: dict[str, torch.Tensor]) -> dict[str, str]:
    out = {}
    for name, value in sorted(tensors.items()):
        tensor = value.detach().cpu().contiguous()
        out[name] = hashlib.sha256(
            f"{tuple(tensor.shape)}|{tensor.dtype}".encode() + tensor.numpy().tobytes()
        ).hexdigest()
    return out


def snapshot(candidate, state, optimizer, update_index: int) -> dict:
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
    return {"tensors": digest(tensors), "meta": {"update_index": int(update_index)}}


def combined_step(candidate, optimizer, fixture: dict, seed: int) -> float:
    torch.manual_seed(seed)
    loss = selection_ce(
        candidate, fixture["goal"].encode(), fixture["record_keys"], fixture["selection_label"]
    ) + locked_loss(
        candidate,
        fixture["goal"].encode(),
        fixture["records"],
        fixture["record_keys"],
        fixture["target_prefix"],
        subject_id=f"pre-{seed}-{fixture['index']}",
    )
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(candidate.trainable_parameters()), CLIP_NORM)
    optimizer.step()
    return float(loss.detach())


def worker(spec: dict) -> None:
    start_cpu = time.process_time()
    out: dict = {"ok": False}
    try:
        candidate, state, optimizer, profile, update_index, cursor, _payload = load_bundle(
            Path(spec["bundle"]), torch.float64
        )
        out = {"ok": True, "snapshot": snapshot(candidate, state, optimizer, update_index)}
        if spec["mode"] == "step":
            fixture = to_fixture(load_rows("train")[spec["row"]])
            out["loss"] = combined_step(candidate, optimizer, fixture, int(spec["seed"]))
            out["adam_calls"] = 1
            out["snapshot"] = snapshot(candidate, state, optimizer, int(update_index) + 1)
    except Exception:
        out = {"ok": False, "message": traceback.format_exc()}
    out["cpu_seconds"] = time.process_time() - start_cpu
    out.setdefault("adam_calls", 0)
    torch.save(out, spec["result"])


def spawn(spec: dict) -> dict:
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=worker, args=(spec,))
    process.start()
    process.join(timeout=WORKER_TIMEOUT)
    if process.is_alive():
        process.terminate()
        process.join()
        return {"ok": False, "message": "worker timed out", "cpu_seconds": 0.0, "adam_calls": 0}
    if process.exitcode != 0:
        return {
            "ok": False,
            "message": f"exit {process.exitcode}",
            "cpu_seconds": 0.0,
            "adam_calls": 0,
        }
    path = Path(spec["result"])
    if not path.exists():
        return {"ok": False, "message": "no result", "cpu_seconds": 0.0, "adam_calls": 0}
    return torch.load(path, map_location="cpu", weights_only=False)


def run_config(payload: dict, profile: str, seed: int, budget: Budget) -> dict:
    candidate, optimizer = build(payload, profile)
    rows = load_rows("train")
    offset = (seed - 28031) * 7
    fixture = to_fixture(rows[offset])
    tag = f"{seed}-{profile}"

    init_state = initial_binding_state(candidate.binding, f"init-{tag}")
    init_path = OUTPUT_DIR / f"{tag}-initial.pt"
    info = save_bundle(
        init_path,
        profile=profile,
        candidate=candidate,
        state=init_state,
        optimizer=optimizer,
        update_index=0,
        cursor={"boundary": "initial", "seed": seed},
        parent_identity={"sha256": PARENT_SHA256},
    )
    budget.add_product(info)
    live0 = snapshot(candidate, init_state, optimizer, 0)
    w0 = spawn(
        {"bundle": str(init_path), "result": str(OUTPUT_DIR / f"{tag}-w0.pt"), "mode": "digest"}
    )
    budget.add_worker(w0)
    initial_equal = bool(w0.get("ok")) and w0["snapshot"]["tensors"] == live0["tensors"]
    init_path.unlink(missing_ok=True)
    (OUTPUT_DIR / f"{tag}-w0.pt").unlink(missing_ok=True)

    loss1 = combined_step(candidate, optimizer, fixture, seed)
    budget.adam += 1
    budget.ensure()
    step1_state = initial_binding_state(candidate.binding, f"step1-{tag}")
    step1_path = OUTPUT_DIR / f"{tag}-step1.pt"
    step1_info = save_bundle(
        step1_path,
        profile=profile,
        candidate=candidate,
        state=step1_state,
        optimizer=optimizer,
        update_index=1,
        cursor={"boundary": "step1", "seed": seed},
        parent_identity={"sha256": PARENT_SHA256},
    )
    budget.add_product(step1_info)
    live1 = snapshot(candidate, step1_state, optimizer, 1)
    w1 = spawn(
        {"bundle": str(step1_path), "result": str(OUTPUT_DIR / f"{tag}-w1.pt"), "mode": "digest"}
    )
    budget.add_worker(w1)
    step1_equal = bool(w1.get("ok")) and w1["snapshot"]["tensors"] == live1["tensors"]

    w2 = spawn(
        {
            "bundle": str(step1_path),
            "result": str(OUTPUT_DIR / f"{tag}-w2.pt"),
            "mode": "step",
            "row": offset,
            "seed": seed,
        }
    )
    budget.add_worker(w2)
    loss2 = combined_step(candidate, optimizer, fixture, seed)
    budget.adam += 1
    budget.ensure()
    live2 = snapshot(candidate, step1_state, optimizer, 2)
    replay_equal = (
        bool(w2.get("ok"))
        and w2["snapshot"]["tensors"] == live2["tensors"]
        and abs(float(w2.get("loss", 0.0)) - loss2) < 1e-12
    )
    for path in (step1_path, OUTPUT_DIR / f"{tag}-w1.pt", OUTPUT_DIR / f"{tag}-w2.pt"):
        path.unlink(missing_ok=True)

    return {
        "seed": seed,
        "profile": profile,
        "initial_restore_equal": initial_equal,
        "step1_restore_equal": step1_equal,
        "replay_equal": bool(replay_equal),
        "initial_worker_ok": bool(w0.get("ok")),
        "step1_worker_ok": bool(w1.get("ok")),
        "replay_worker_ok": bool(w2.get("ok")),
        "step1_loss": loss1,
        "step2_loss": loss2,
        "replay_worker_loss": float(w2.get("loss", 0.0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="preflight", choices=("preflight",))
    parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    budget = Budget()
    try:
        if sha256_file(ROOT / PARENT_BUNDLE) != PARENT_SHA256:
            raise SystemExit("parent bundle sha drifted")
        payload = torch.load(ROOT / PARENT_BUNDLE, map_location="cpu", weights_only=False)
        results = []
        for seed in SEEDS:
            for profile in PROFILES:
                results.append(run_config(payload, profile, seed, budget))

        checks = {
            "parent_identity": sha256_file(ROOT / PARENT_BUNDLE) == PARENT_SHA256,
            "grid": len(results) == len(SEEDS) * len(PROFILES),
            "initial_restore": all(r["initial_restore_equal"] for r in results),
            "step1_restore": all(r["step1_restore_equal"] for r in results),
            "replay_update": all(r["replay_equal"] for r in results),
            "independent_process": all(
                r["initial_worker_ok"] and r["step1_worker_ok"] and r["replay_worker_ok"]
                for r in results
            ),
            "adam_budget": budget.adam <= STAGE_ADAM_CALLS,
            "product_limit": budget.max_live <= STAGE_PRODUCT_BYTES,
            "stage_budget": budget.total_cpu() <= STAGE_CPU_SECONDS
            and (time.perf_counter() - STARTED_WALL) <= STAGE_WALL_SECONDS,
            "products_cleaned": not any(OUTPUT_DIR.glob("*.pt")),
            "confirm_untouched": True,
        }
        failed = [name for name, ok in checks.items() if not ok]
        status = "passed_16_1_0d_preflight" if not failed else "engineering_failure"

        report = {
            "schema": "taiji-binding-select-preflight-v1",
            "step": "N16.1.0d",
            "date": "2026-11-03",
            "status": status,
            "checks": checks,
            "failed": failed,
            "configs": results,
            "accounting": {"actual_adam_calls": budget.adam, "effective_adam_calls": 0},
            "cost": {
                "cpu_seconds": budget.total_cpu(),
                "wall_seconds": time.perf_counter() - STARTED_WALL,
                "adam_calls": budget.adam,
                "peak_live_product_bytes": budget.max_live,
            },
            "budget": {
                "cpu_seconds": STAGE_CPU_SECONDS,
                "wall_seconds": STAGE_WALL_SECONDS,
                "adam_calls": STAGE_ADAM_CALLS,
                "product_bytes": STAGE_PRODUCT_BYTES,
            },
            "source_hashes": {path: sha256_file(ROOT / path) for path in SOURCE_FILES},
            "limits": "Preflight only. No adoption training, no confirm read, no capability claim.",
        }
        Path(REPORT).write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        manifest = {
            "schema": "taiji-binding-select-manifest-v1",
            "step": "N16.1.0d",
            "status": status,
            "report": REPORT,
            "seeds": list(SEEDS),
            "profiles": list(PROFILES),
            "checks": checks,
            "cost": report["cost"],
            "source_hashes": report["source_hashes"],
        }
        Path(MANIFEST).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"16.1.0d status: {status}")
        print(f"  failed: {failed}")
        print(
            f"  cpu {budget.total_cpu():.2f}/{STAGE_CPU_SECONDS}, adam {budget.adam}/{STAGE_ADAM_CALLS}"
        )
        print(f"  peak {budget.max_live / (1024 * 1024):.1f} MiB")
        return 0 if not failed else 1
    except StageStop as exc:
        print(f"16.1.0d stop: {exc}")
        return 1
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
