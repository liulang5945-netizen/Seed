"""R-REBUILD-1 rebuildability probe for the frozen U.1r2 inputs (zero training).

The birth capsules and the task dataset live under data/, which .gitignore:160
excludes, so the contract pins bytes that are not in the repository. This probe
rebuilds them through the producing chain and compares digests, with a perturbation
negative control on each side so a passing readout can actually fail.

The dataset half is delegated to the U.1r1 probe (same data_root, same partitions,
same generating functions); only the bundle half is r2-specific, because the birth
graph now carries log_tau. Rebuilding goes through the functions freeze() itself
calls, not through --freeze: that mode refuses a frozen contract and would rewrite it.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training import run_u1_unit_read_20261116 as u1
from scripts.training import run_u1r2_scale_supervision_20261116 as r2
from scripts.training import verify_u1r1_rebuild_20261116 as rb1

REPORT = ROOT / "reports/taiji_u2_u1r2_scale_supervision_rebuild_20261116.json"
SCRATCH = ROOT / "output/_u1r2_rebuild_scratch"
PERTURBATIONS = (("position_head.weight", 1e-3), ("log_tau", 1e-3))


def bundle_rows(contract: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key, rel in contract["start_bundles"].items():
        seed = int(key.removeprefix("seed"))
        frozen = contract["start_bundle_sha256"][key]
        rebuilt = SCRATCH / f"{key}_rebuilt.bin"
        r2.save_capsule(rebuilt, r2.build_capsule(r2.build_graph(seed)))
        rebuilt_sha = rb1.bytes_sha(rebuilt)
        controls = {}
        for name, delta in PERTURBATIONS:
            graph = r2.build_graph(seed)
            with torch.no_grad():
                dict(graph.named_parameters())[name] += delta
            path = SCRATCH / f"{key}_perturbed_{name.replace('.', '_')}.bin"
            r2.save_capsule(path, r2.build_capsule(graph))
            sha = rb1.bytes_sha(path)
            controls[name] = {"sha256": sha, "rejects": sha != frozen}
        rows.append(
            {
                "key": key,
                "seed": seed,
                "path": rel,
                "frozen_sha256": frozen,
                "on_disk_sha256": rb1.bytes_sha(ROOT / rel),
                "rebuilt_sha256": rebuilt_sha,
                "rebuilt_bytes": rebuilt.stat().st_size,
                "exact": rebuilt_sha == frozen == rb1.bytes_sha(ROOT / rel),
                "negative_controls": controls,
                "negative_control_rejects": all(c["rejects"] for c in controls.values()),
            }
        )
    return rows


def main() -> int:
    torch.set_num_threads(1)
    started_cpu, started_wall = time.process_time(), time.monotonic()
    contract = json.loads(r2.CONTRACT.read_text(encoding="utf-8"))
    r2.guards(contract, "smoke")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    try:
        bundles = bundle_rows(contract)
        datasets, budget_stopped = rb1.dataset_rows(contract, started_wall)
    finally:
        shutil.rmtree(SCRATCH, ignore_errors=True)
    scratch_clean = not SCRATCH.exists()
    partition_rows = [
        r for r in datasets if r.get("status") == "measured" and r["partition"] != "whole_dataset"
    ]
    whole_rows = [r for r in datasets if r["partition"] == "whole_dataset"]
    complete = len(partition_rows) == len(rb1.PARTITIONS)
    bundle_exact = all(r["exact"] and r["negative_control_rejects"] for r in bundles)
    dataset_exact = (
        complete
        and all(r["exact"] and r["negative_control_rejects"] for r in partition_rows)
        and all(r["exact"] for r in whole_rows)
    )
    classification = rb1.classify(bundle_exact, dataset_exact, complete)
    rebuilt_shas = [r["rebuilt_sha256"] for r in bundles]
    cross_seed_distinct = len(set(rebuilt_shas)) == len(rebuilt_shas)
    payload = {
        "schema": "taiji-u2-u1r2-rebuild-v1",
        "card": "22.5.U.1r2",
        "phase": "rebuild",
        "rule": "R-REBUILD-1",
        "criteria_sha256": contract["criteria_sha256"],
        "ignored_by": ".gitignore:160:data/",
        "classification": classification,
        "bundles": bundles,
        "dataset": datasets,
        "cross_seed_distinct": cross_seed_distinct,
        "rebuild_path": (
            "r2.build_graph(seed) -> r2.build_capsule -> r2.save_capsule, the same calls "
            "freeze() makes; --freeze itself refuses a frozen contract and rewrites it"
        ),
        "reused_instrument": {
            "module": "scripts/training/verify_u1r1_rebuild_20261116.py",
            "sha256": r2.sha_of(Path(rb1.__file__).resolve()),
            "reused": ["bytes_sha", "dataset_rows", "classify", "PARTITIONS"],
            "r2_specific": ["bundle_rows (log_tau negative control)"],
        },
        "budget_stopped": budget_stopped,
        "scratch_removed": scratch_clean,
        "verdict": (
            "rebuild_exact"
            if classification == "regenerable_exact"
            and scratch_clean
            and not budget_stopped
            and cross_seed_distinct
            else "rebuild_not_established"
        ),
        "discipline": {
            "optimizer_steps": 0,
            "training_updates": 0,
            "task_data_reads": 1,
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0,
            "sealed_partitions_read": [],
        },
        "cost": {
            "cpu": round(time.process_time() - started_cpu, 4),
            "wall": round(time.monotonic() - started_wall, 3),
        },
    }
    u1.write_report(REPORT, payload)
    bundle_hits = sum(1 for r in bundles if r["exact"])
    control_hits = sum(1 for r in bundles for c in r["negative_controls"].values() if c["rejects"])
    print(
        f"REBUILD verdict={payload['verdict']} classification={classification} "
        f"bundles={bundle_hits}/{len(bundles)} "
        f"negative_controls={control_hits}/{len(bundles) * len(PERTURBATIONS)} "
        f"dataset_partitions={len(partition_rows)}/{len(rb1.PARTITIONS)} "
        f"scratch_clean={scratch_clean}"
    )
    return 0 if payload["verdict"] == "rebuild_exact" else 1


if __name__ == "__main__":
    raise SystemExit(main())
