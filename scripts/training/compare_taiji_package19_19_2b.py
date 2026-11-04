"""19.2b six-cell comparison: same dose as 19.5, 32-d span pointer only.

Every cell trains from the frozen 16.1 A-locked parent exactly like 19.5
(2 seeds x 3 arms x 300 updates, batch 8, stages 1-100/101-200/201-300, LR 1e-3,
clip 1.0).  The only change is ``span_bottleneck=32`` on the teacher-span head:
4,194 parameters instead of 258, gold start/end from
``teacher_span(record_raw, entity_bytes)``, start logits injected into
``routing_step`` as ``pointer_bias``.  Alpha/beta, encoder, transport operator,
gate numbers and data are untouched.

Outputs go to ``output/taiji_package19_2b_compare`` so the 19.5 baseline in
``output/taiji_package19_compare`` is never overwritten.  Each cell runs in its
own process (Windows spawn would otherwise lose the span-shape patch), writes
its bundle before any evaluation, and keeps a step150 midpoint.

Monitoring: 20,000 / 30,000 CPU lines write ``watchline.json``; extrapolation
uses the measured 19.5 per-update slope (never a fixed start-up cost) and the
run continues with a recorded reason.

Run:
  python scripts/training/compare_taiji_package19_19_2b.py            # six cells
  python scripts/training/compare_taiji_package19_19_2b.py --report-only
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MODULE_PATH = ROOT / "scripts/training/compare_taiji_package19.py"
OUTPUT_DIR = ROOT / "output/taiji_package19_2b_compare"
REPORT = ROOT / "reports/taiji_package19_2b_compare_20261104.json"
MANIFEST = ROOT / "plans/manifests/taiji_package19_2b_compare_20261104.json"
PRICING = ROOT / "plans/manifests/taiji_pricing_19_2b_20261104.json"
PREREG = ROOT / "plans/manifests/taiji_package19_2b_prereg_20261104.json"
LOG_DIR = ROOT / "output/taiji_package19_2b_logs"
SEEDS = (28031, 28032)
ARMS = ("A_route_lock", "B_stop_lock", "C_unlocked")
SPAN_BOTTLENECK = 32
UPDATES = 300

MEASURED_19_5 = {
    "cpu_seconds": 29574.234375,
    "cells": 6,
    "updates_per_cell": 300,
    "per_update_all_in": 29574.234375 / 1800.0,
    "first_half_cpu_six": 8997.015625,
    "second_half_cpu_six": 20577.21875,
    "second_half_per_update_band": [20.8, 26.7],
}
WATCH_CPU = (20000.0, 30000.0)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def module():
    spec = importlib.util.spec_from_file_location("cmp19_2b", MODULE_PATH)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def patch(loaded) -> None:
    from taiji.selection_routing_checkpoint import (
        build_routing_candidate,
        save_routing_bundle,
    )

    loaded.OUTPUT_DIR = OUTPUT_DIR
    loaded.PARENTS = {
        28031: Path(
            "output/taiji_core_binding_select_16_1_compare/bundles/28031-A-locked-step300.pt"
        ),
        28032: Path(
            "output/taiji_core_binding_select_16_1_compare/bundles/28032-A-locked-step300.pt"
        ),
    }
    loaded.build_routing_candidate = lambda parent, seed, arm: build_routing_candidate(
        parent, seed, arm, span_bottleneck=SPAN_BOTTLENECK
    )

    def relabelled(path, candidate, optimizer, update, metadata, parent, initial_state):
        metadata = {**metadata, "stage": "N19.2b", "span_bottleneck": SPAN_BOTTLENECK}
        return save_routing_bundle(
            path, candidate, optimizer, update, metadata, parent, initial_state
        )

    loaded.save_routing_bundle = relabelled


def worker(seed: int, arm: str, resume: bool) -> int:
    loaded = module()
    patch(loaded)
    start_cpu = time.process_time()
    result = loaded.run_one(seed, arm, resume=resume)
    write_json(OUTPUT_DIR / f"run-{seed}-{arm}.json", result)
    payload = loaded.evaluate_step300_cell(seed, arm)
    print(
        json.dumps(
            {
                "cell": f"{seed}-{arm}",
                "overall": round(result["step300"]["overall"], 3),
                "copy": round(result["step300"]["new_name_copied"], 3),
                "unseen": round(result["step300"]["subsets"]["unseen_output_name"], 3),
                "cpu": round(result["timing"]["cpu_seconds"], 1),
                "eval_cpu": round(payload["timing"]["cpu_seconds"], 1),
                "driver_cpu": round(time.process_time() - start_cpu, 1),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


def watch_loop(processes: list[subprocess.Popen]) -> dict:
    hit = {}
    lines = list(WATCH_CPU)
    while any(process.poll() is None for process in processes):
        rows = []
        for seed in SEEDS:
            for arm in ARMS:
                path = OUTPUT_DIR / f"progress-{seed}-{arm}.json"
                if not path.exists():
                    continue
                try:
                    rows.append(json.loads(path.read_text(encoding="utf-8")))
                except (json.JSONDecodeError, OSError):
                    continue
        observed = sum(float(row["cost"]["cpu_seconds"]) for row in rows)
        finished = sum(
            1
            for seed in SEEDS
            for arm in ARMS
            if (OUTPUT_DIR / f"{seed}-{arm}-step300.pt").exists()
        )
        remaining_cells = 6 - finished
        for line in lines:
            if line in hit:
                continue
            extrapolated = observed + MEASURED_19_5["per_update_all_in"] * UPDATES * remaining_cells
            if observed >= line or extrapolated >= line:
                payload = {
                    "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "line_cpu": line,
                    "observed_cpu_seconds": observed,
                    "extrapolated_cpu_seconds": extrapolated,
                    "slope_source": "measured 19.5 all-in per-update "
                    f"{MEASURED_19_5['per_update_all_in']:.3f} CPU/update; "
                    "no fixed start-up cost amplification",
                    "cells_finished_step300": finished,
                    "cells_with_progress": [
                        f"{row['seed']}-{row['arm']}:{row['update']}" for row in rows
                    ],
                    "decision": "continue; reason recorded",
                }
                write_json(OUTPUT_DIR / f"watchline-{int(line)}.json", payload)
                hit[line] = payload
                print(json.dumps({"WATCHLINE_HIT": payload}, ensure_ascii=False), flush=True)
        time.sleep(45)
    return hit


def launch() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    processes = []
    for seed in SEEDS:
        for arm in ARMS:
            log_path = LOG_DIR / f"{seed}-{arm}.log"
            handle = log_path.open("w", encoding="utf-8")
            processes.append(
                (
                    subprocess.Popen(
                        [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "--worker",
                            "--seed",
                            str(seed),
                            "--arm",
                            arm,
                        ],
                        cwd=ROOT,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                    ),
                    f"{seed}-{arm}",
                    handle,
                )
            )
    watch = watch_loop([item[0] for item in processes])
    failures = []
    for process, tag, handle in processes:
        code = process.wait()
        handle.close()
        if code != 0:
            failures.append({"cell": tag, "exit_code": code})
    if failures:
        write_json(OUTPUT_DIR / "failures.json", {"failures": failures})
        print(json.dumps({"failures": failures}, ensure_ascii=False))
        return 1
    return report_only(watch)


def report_only(watch: dict | None = None) -> int:
    runs = {}
    for seed in SEEDS:
        for arm in ARMS:
            path = OUTPUT_DIR / f"run-{seed}-{arm}.json"
            if not path.exists():
                raise FileNotFoundError(path)
            runs[f"{seed}-{arm}"] = json.loads(path.read_text(encoding="utf-8"))
    cells = {}
    total_cpu = 0.0
    for key, run in sorted(runs.items()):
        step = run["step300"]
        cells[key] = {
            "seed": run["seed"],
            "arm": run["arm"],
            "candidate_sha": run["candidate"]["candidate_sha"],
            "overall": step["overall"],
            "new_name_copied": step["new_name_copied"],
            "unseen_output_name": step["subsets"]["unseen_output_name"],
            "new_description": step["subsets"]["new_description"],
            "utf8": step["utf8"],
            "eos": step["eos"],
            "gamma_mean": step["gamma_mean"],
            "one_minus_gamma": step["one_minus_gamma"],
            "selection_object": step["selection_object"],
            "selection_place": step["selection_place"],
            "step0": {
                "overall": run["step0"]["overall"],
                "new_name_copied": run["step0"]["new_name_copied"],
            },
            "step150": {
                "overall": run["step150"]["overall"],
                "new_name_copied": run["step150"]["new_name_copied"],
                "cost_cpu": run["step150"]["cost"]["cpu_seconds"],
            },
            "train": run["train"],
            "consistency": run["consistency"],
            "timing": run["timing"],
        }
        total_cpu += float(run["timing"]["cpu_seconds"])
    pooled = {
        "overall": statistics.fmean(cell["overall"] for cell in cells.values()),
        "new_name_copied": statistics.fmean(cell["new_name_copied"] for cell in cells.values()),
        "unseen_output_name": statistics.fmean(
            cell["unseen_output_name"] for cell in cells.values()
        ),
        "new_description": statistics.fmean(cell["new_description"] for cell in cells.values()),
        "utf8": statistics.fmean(float(cell["utf8"]) for cell in cells.values()),
        "eos": statistics.fmean(float(cell["eos"]) for cell in cells.values()),
    }
    gate_without_probe = {
        "new_name_copied_0.60": pooled["new_name_copied"] >= 0.60,
        "overall_0.80": pooled["overall"] >= 0.80,
        "unseen_0.80": pooled["unseen_output_name"] >= 0.80,
        "new_description_0.80": pooled["new_description"] >= 0.80,
        "utf8_0.98": pooled["utf8"] >= 0.98,
        "eos_0.95": pooled["eos"] >= 0.95,
    }
    pricing = json.loads(PRICING.read_text(encoding="utf-8"))
    report = {
        "schema": "taiji-package19-2b-compare-v1",
        "step": "N19.2b",
        "date": "2026-11-04",
        "design": {
            "dose": "2 seeds x 3 arms x 300 updates; batch 8; selection 1-100 / copy 101-200 / joint 201-300",
            "parent": "16.1 A-locked step300 (same as 19.5)",
            "change": "span_bottleneck 32 (4,194 params); alpha/beta/encoder/transport/gates unchanged",
            "baseline": "19.5 six step300 bundles in output/taiji_package19_compare",
        },
        "pricing": {
            "bare_six_cpu": pricing["budget"]["six_cells_bare_cpu"],
            "protected_six_cpu": pricing["budget"]["six_cells_protected_cpu"],
            "measured_19_5_six_cpu": pricing["budget"]["six_cells_19_5_measured_cpu"],
        },
        "measured_19_5_reference": MEASURED_19_5,
        "cells": cells,
        "pooled": pooled,
        "gate_without_probe": gate_without_probe,
        "cost": {
            "cpu_seconds": total_cpu,
            "per_update_all_in": total_cpu / (len(cells) * UPDATES),
            "budget_bare": pricing["budget"]["six_cells_bare_cpu"],
            "budget_protected": pricing["budget"]["six_cells_protected_cpu"],
            "over_budget_bare": total_cpu > pricing["budget"]["six_cells_bare_cpu"],
            "watchlines": list(WATCH_CPU),
            "watchline_hits": [int(line) for line in (watch or {})],
            "first_half_cpu_six": sum(cell["step150"]["cost_cpu"] for cell in cells.values()),
            "second_half_cpu_six": sum(
                float(cell["timing"]["cpu_seconds"]) - cell["step150"]["cost_cpu"]
                for cell in cells.values()
            ),
        },
        "limits": {"confirm_reads": 0, "read_only_probe": False},
        "note": "byte-level object/place and start_hit/conditional_byte live in the "
        "step300 ledger probe; the decision report merges both",
    }
    write_json(REPORT, report)
    write_json(
        MANIFEST,
        {
            "schema": "taiji-package19-2b-compare-manifest-v1",
            "step": "N19.2b",
            "report": str(REPORT.relative_to(ROOT)),
            "cells": list(cells),
            "cost_cpu": total_cpu,
            "pooled": pooled,
            "gate_without_probe": gate_without_probe,
        },
    )
    print(
        json.dumps(
            {
                "report": str(REPORT.relative_to(ROOT)),
                "pooled": {key: round(value, 3) for key, value in pooled.items()},
                "gate": gate_without_probe,
                "cpu": round(total_cpu, 1),
            },
            ensure_ascii=False,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--seed", type=int, default=SEEDS[0])
    parser.add_argument("--arm", default=ARMS[0])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.worker:
        return worker(arguments.seed, arguments.arm, arguments.resume)
    if arguments.report_only:
        return report_only()
    return launch()


if __name__ == "__main__":
    raise SystemExit(main())
