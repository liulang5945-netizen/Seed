"""R-REBUILD-1 rebuildability probe for the frozen U.1r1 inputs (zero training).

The start bundles and the task dataset live under data/, which .gitignore excludes,
so the frozen contract pins bytes that are not in the repository. This probe rebuilds
them through the producing chain itself and compares digests, with a perturbation
negative control on each side so a passing readout can actually fail.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training import first_a_support_22_5 as support
from scripts.training import run_u1_unit_read_20261116 as u1
from scripts.training import run_u1r1_position_20261116 as r1
from taiji.u2_history import value_digest

REPORT = ROOT / "reports/taiji_u2_u1r1_rebuild_20261116.json"
SCRATCH = ROOT / "output/_u1r1_rebuild_scratch"
PARTITIONS = (("price", 8), ("train", 64), ("fit", 32))
WALL_ALLOWANCE = 600.0
WALL_RESERVE = 30.0


def bytes_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bundle_rows(contract: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key, rel in contract["start_bundles"].items():
        seed = int(key.removeprefix("seed"))
        frozen = contract["start_bundle_sha256"][key]
        rebuilt = SCRATCH / f"{key}_rebuilt.bin"
        r1.save_capsule(rebuilt, r1.build_capsule(r1.build_graph(seed)))
        perturbed_graph = r1.build_graph(seed)
        with torch.no_grad():
            dict(perturbed_graph.named_parameters())["position_head.weight"] += 1e-3
        perturbed = SCRATCH / f"{key}_perturbed.bin"
        r1.save_capsule(perturbed, r1.build_capsule(perturbed_graph))
        rebuilt_sha = bytes_sha(rebuilt)
        rows.append(
            {
                "key": key,
                "seed": seed,
                "path": rel,
                "frozen_sha256": frozen,
                "on_disk_sha256": bytes_sha(ROOT / rel),
                "rebuilt_sha256": rebuilt_sha,
                "rebuilt_bytes": rebuilt.stat().st_size,
                "exact": rebuilt_sha == frozen == bytes_sha(ROOT / rel),
                "perturbed_sha256": bytes_sha(perturbed),
                "negative_control_rejects": bytes_sha(perturbed) != frozen,
            }
        )
    return rows


def dataset_rows(
    contract: dict[str, Any], started_wall: float
) -> tuple[list[dict[str, Any]], bool]:
    on_disk = u1.load_data(contract)
    rows = []
    regenerated: dict[str, Any] = {}
    budget_stopped = False
    for part, count in PARTITIONS:
        if time.monotonic() - started_wall >= WALL_ALLOWANCE - WALL_RESERVE:
            budget_stopped = True
            rows.append({"partition": part, "count": count, "status": "not_measured"})
            continue
        cpu0, wall0 = time.process_time(), time.monotonic()
        examples = support.generate_examples(part, count)
        cpu, wall = time.process_time() - cpu0, time.monotonic() - wall0
        rebuilt_digest = value_digest(examples)
        disk_digest = value_digest(on_disk[part])
        tampered = copy.deepcopy(examples)
        tampered[0]["target"] = b"\xe5\xb7\xa6"
        rows.append(
            {
                "partition": part,
                "count": count,
                "status": "measured",
                "rebuilt_value_digest": rebuilt_digest,
                "on_disk_value_digest": disk_digest,
                "exact": rebuilt_digest == disk_digest,
                "ids_match": [e["id"] for e in examples] == [e["id"] for e in on_disk[part]],
                "negative_control_rejects": value_digest(tampered) != disk_digest,
                "cpu": round(cpu, 4),
                "wall": round(wall, 3),
            }
        )
        regenerated[part] = examples
    complete = len(regenerated) == len(PARTITIONS)
    if complete:
        whole = value_digest(regenerated)
        rows.append(
            {
                "partition": "whole_dataset",
                "count": sum(n for _, n in PARTITIONS),
                "status": "measured",
                "rebuilt_value_digest": whole,
                "on_disk_value_digest": contract["data_digest"],
                "exact": whole == contract["data_digest"],
            }
        )
    return rows, budget_stopped


def classify(bundle_exact: bool, dataset_exact: bool, complete: bool) -> str:
    if not complete:
        return "not_established"
    if bundle_exact and dataset_exact:
        return "regenerable_exact"
    return "not_regenerable"


def main() -> int:
    torch.set_num_threads(1)
    started_cpu, started_wall = time.process_time(), time.monotonic()
    contract = json.loads(r1.CONTRACT.read_text(encoding="utf-8"))
    r1.guards(contract, "smoke")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    try:
        bundles = bundle_rows(contract)
        datasets, budget_stopped = dataset_rows(contract, started_wall)
    finally:
        shutil.rmtree(SCRATCH, ignore_errors=True)
    scratch_clean = not SCRATCH.exists()
    partition_rows = [
        r for r in datasets if r.get("status") == "measured" and r["partition"] != "whole_dataset"
    ]
    whole_rows = [r for r in datasets if r["partition"] == "whole_dataset"]
    complete = len(partition_rows) == len(PARTITIONS)
    bundle_exact = all(r["exact"] and r["negative_control_rejects"] for r in bundles)
    dataset_exact = (
        complete
        and all(r["exact"] and r["negative_control_rejects"] for r in partition_rows)
        and all(r["exact"] for r in whole_rows)
    )
    classification = classify(bundle_exact, dataset_exact, complete)
    payload = {
        "schema": "taiji-u2-u1r1-rebuild-v1",
        "card": "22.5.U.1r1",
        "phase": "rebuild",
        "rule": "R-REBUILD-1",
        "criteria_sha256": contract["criteria_sha256"],
        "ignored_by": ".gitignore:160:data/",
        "classification": classification,
        "bundles": bundles,
        "dataset": datasets,
        "budget_stopped": budget_stopped,
        "scratch_removed": scratch_clean,
        "verdict": (
            "rebuild_exact"
            if classification == "regenerable_exact" and scratch_clean and not budget_stopped
            else "rebuild_not_established"
        ),
        "discipline": {
            "optimizer_steps": 0,
            "training_updates": 0,
            "task_data_reads": 1,
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0,
        },
        "cost": {
            "cpu": round(time.process_time() - started_cpu, 4),
            "wall": round(time.monotonic() - started_wall, 3),
        },
    }
    u1.write_report(REPORT, payload)
    bundle_hits = sum(1 for r in bundles if r["exact"])
    print(
        f"REBUILD verdict={payload['verdict']} classification={classification} "
        f"bundles={bundle_hits}/{len(bundles)} "
        f"dataset_partitions={len(partition_rows)}/{len(PARTITIONS)} "
        f"scratch_clean={scratch_clean}"
    )
    return 0 if payload["verdict"] == "rebuild_exact" else 1


if __name__ == "__main__":
    raise SystemExit(main())
