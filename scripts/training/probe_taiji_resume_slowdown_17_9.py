"""17.9 timing probe: why did the step150 resume cost ~2.2x more CPU per update?

Attempt 1 (fresh run, 300 updates x 6 cells) averaged 7.08 CPU s per update.
Attempt 3 (resume from step150, 150 updates x 6 cells) averaged 15.78.  This probe
re-measures the same code path on an idle machine from one persisted step150 bundle,
so the difference can be attributed to the code path or to external load instead of
guessed at.

Hard caps: at most 4 optimiser steps and 60 CPU seconds.  Nothing is persisted; the
updated weights are discarded.  No confirm read, no threshold change.
"""

from __future__ import annotations

import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

torch.set_num_threads(1)

COMPARE = ROOT / "scripts/training/compare_taiji_selection_routing_17_5.py"
OUTPUT_DIR = ROOT / "output/taiji_selection_routing_17_5"
REPORT = ROOT / "reports/taiji_resume_slowdown_probe_20261104.json"
MANIFEST = ROOT / "plans/manifests/taiji_resume_slowdown_probe_20261104.json"

SEED = 28031
ARM = "B_stop_lock"
UPDATES_TO_TIME = (201, 202)
ATTEMPT_ONE_PER_UPDATE = 12743.296875 / 1800
ATTEMPT_THREE_PER_UPDATE = 14204.7 / 900
CAP_ADAM = 4
CAP_CPU = 60.0


def load_compare():
    spec = importlib.util.spec_from_file_location("cmp17t", COMPARE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    import hashlib

    start_cpu = time.process_time()
    start_wall = time.perf_counter()
    compare = load_compare()
    train, _dev = compare.load_fixtures()
    batches = compare.schedule(SEED, train, compare.UPDATES)
    bundle_path = OUTPUT_DIR / f"{SEED}-{ARM}-step{compare.CHECKPOINT_UPDATE}.pt"
    candidate, optimizer, _state, resumed_from, _payload = compare.load_routing_bundle(bundle_path)
    candidate.configure_trainable()
    tag = f"timing-{SEED}-{ARM}"
    rows = []
    for update in UPDATES_TO_TIME:
        if time.process_time() - start_cpu > CAP_CPU:
            rows.append({"update": update, "skipped": "cpu cap reached"})
            break
        if len(rows) >= CAP_ADAM:
            break
        stage = compare.stage_of(update)
        batch = batches[update - 1]
        cpu_0 = time.process_time()
        completed = [
            compare.teacher_forced_completion(candidate, fixture, ARM, update, tag)
            for fixture in batch
        ]
        cpu_curriculum = time.process_time() - cpu_0
        rate = sum(1 for flag in completed if flag) / len(completed)
        use_self_prefix = rate >= compare.CURRICULUM_THRESHOLD
        parameters = compare.set_trainable(candidate, stage)
        cpu_1 = time.process_time()
        loss, _stats = compare.batch_loss(
            candidate, batch, ARM, stage, use_self_prefix, update, tag
        )
        cpu_forward = time.process_time() - cpu_1
        cpu_2 = time.process_time()
        optimizer.zero_grad()
        loss.backward()
        clip = float(torch.nn.utils.clip_grad_norm_(parameters, compare.CLIP))
        optimizer.step()
        cpu_backward = time.process_time() - cpu_2
        rows.append(
            {
                "update": update,
                "stage": stage,
                "curriculum_cpu": cpu_curriculum,
                "forward_cpu": cpu_forward,
                "backward_cpu": cpu_backward,
                "total_cpu": cpu_curriculum + cpu_forward + cpu_backward,
                "curriculum_rate": rate,
                "self_prefix": bool(use_self_prefix),
                "clip_scale": clip,
            }
        )
    measured = [row["total_cpu"] for row in rows if "total_cpu" in row]
    mean_cpu = statistics.fmean(measured) if measured else None
    verdict = "unexplained_slowdown"
    if mean_cpu is not None and mean_cpu <= ATTEMPT_ONE_PER_UPDATE * 1.25:
        verdict = "external_contention_suspected"
    if mean_cpu is not None and mean_cpu >= ATTEMPT_THREE_PER_UPDATE * 0.9:
        verdict = "resume_path_cost_reproduced"
    protection = (
        max(1.0, ATTEMPT_THREE_PER_UPDATE / mean_cpu)
        if mean_cpu
        else ATTEMPT_THREE_PER_UPDATE / ATTEMPT_ONE_PER_UPDATE
    )
    report = {
        "schema": "taiji-resume-slowdown-probe-17-9-v1",
        "step": "N17.9",
        "date": "2026-11-04",
        "status": "zero_persist_timing_probe",
        "cell": f"{SEED}-{ARM}",
        "bundle": {
            "path": str(bundle_path.relative_to(ROOT)),
            "sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
            "resumed_from": int(resumed_from),
        },
        "caps": {"adam_steps": CAP_ADAM, "cpu_seconds": CAP_CPU},
        "rows": rows,
        "measured_cpu_per_update": mean_cpu,
        "references": {
            "attempt1_per_update": ATTEMPT_ONE_PER_UPDATE,
            "attempt3_resume_per_update": ATTEMPT_THREE_PER_UPDATE,
            "attempt1_basis": "12743.296875 CPU over 300 updates x 6 cells",
            "attempt3_basis": "14204.7 CPU training over 150 updates x 6 cells",
        },
        "verdict": verdict,
        "protection_coefficient": protection,
        "cost": {
            "optimizer_steps": len(measured),
            "training_updates": len(measured),
            "confirm_reads": 0,
            "cpu_seconds": time.process_time() - start_cpu,
            "wall_seconds": time.perf_counter() - start_wall,
        },
    }
    write_json(REPORT, report)
    write_json(
        MANIFEST,
        {
            "schema": "taiji-resume-slowdown-probe-17-9-manifest-v1",
            "step": "N17.9",
            "date": "2026-11-04",
            "report": str(REPORT.relative_to(ROOT)),
            "report_sha256": hashlib.sha256(REPORT.read_bytes()).hexdigest(),
            "report_bytes": REPORT.stat().st_size,
            "scripts": {
                str(Path(__file__).relative_to(ROOT)): hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest()
            },
            "optimizer_steps": len(measured),
            "training_updates": len(measured),
            "confirm_reads": 0,
        },
    )
    print(
        json.dumps(
            {
                "report": str(REPORT.relative_to(ROOT)),
                "verdict": verdict,
                "measured_cpu_per_update": mean_cpu,
                "protection_coefficient": protection,
                "cpu": round(report["cost"]["cpu_seconds"], 2),
                "wall": round(report["cost"]["wall_seconds"], 1),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
