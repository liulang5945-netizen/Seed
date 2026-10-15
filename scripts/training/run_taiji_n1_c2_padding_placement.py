"""PLAN-N1-09: move equal-dose zero padding before the ordered trajectory."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

START = time.perf_counter()
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from api.seed_runtime import SeedRuntime  # noqa: E402
from scripts.training import run_taiji_n1_c2_calibration as c2  # noqa: E402
from taiji.stepwise_consumer import (  # noqa: E402
    StepwiseConsumerRequest,
    select_and_prepare,
    vector_digest,
)

REPORT = ROOT / "reports/taiji_n1_c2_padding_placement_20261015.json"
FINAL = ROOT / "reports/taiji_n1_c2_padding_placement_final_20261015.json"
AUDIT = ROOT / "reports/taiji_n1_c2_padding_placement_restore_20261015.json"
EVENTS = ROOT / "reports/taiji_n1_c2_padding_placement_20261015_events.jsonl"
BASELINE = c2.paths("20261015_r2")["final"]
BASELINE_SHA = "82cfde7d0f62606310a96f5a30db16ac13d6e52ff98e09b14d4b41a5f80b7184"
MODES = ("oracle-correct", "wrong-episode", "zero-replay")
LIMIT = 180.0


def timing():
    return {"cpu_seconds": time.process_time(), "wall_seconds": time.perf_counter() - START}


def prefix_padding(request, query, candidates):
    if request.mode not in MODES:
        raise ValueError("padding diagnostic only accepts the three frozen controls")
    replay, audit = select_and_prepare(request, query, candidates)
    padding = len(replay) - audit["original_steps"]
    if padding:
        replay = torch.cat((replay[-padding:], replay[:-padding]), dim=0)
    audit = dict(
        audit,
        padding_steps=padding,
        padding_placement="front",
        tail_replay_sha256=audit["replay_sha256"],
        replay_sha256=vector_digest(replay),
    )
    return replay, audit


@contextmanager
def front_padding():
    module = importlib.import_module("taiji.model")
    original = module.select_and_prepare
    module.select_and_prepare = prefix_padding
    try:
        yield
    finally:
        module.select_and_prepare = original


def query(prior):
    selected = c2.items()
    if c2.sha(BASELINE) != BASELINE_SHA:
        raise RuntimeError("frozen tail-padding baseline changed")
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    old = {(r["item_id"], r["arm"]): r for r in baseline["rows"]}
    files = c2.paths("20261015")
    support = json.loads(files["support"].read_text(encoding="utf-8"))
    restored = json.loads(AUDIT.read_text(encoding="utf-8"))
    if c2.sha(files["checkpoint"]) != restored["checkpoint_sha256"]:
        raise RuntimeError("audited support changed")
    rows = []
    status = "complete"
    for item in selected:
        ids = tuple(r["episode_id"] for r in support["rows"] if r["item_id"] == item["id"])
        for mode in MODES:
            used = prior + time.process_time()
            projected = used + (48 - len(rows)) * sum(r["arm_cpu_seconds"] for r in rows) / max(
                1, len(rows)
            )
            if used >= LIMIT or (used >= 144 and projected > LIMIT):
                status = "budget-stop"
                break
            started = time.process_time()
            runtime = SeedRuntime.load(files["checkpoint"])
            model = runtime.model.substrate
            if model.persistent_stepwise_carrier or model._stepwise_consumer_request is not None:
                raise RuntimeError("restore switches are enabled")
            before = c2.parameter_digest(model)
            if before != support["model_parameter_digest"]:
                raise RuntimeError("frozen weights changed")
            index = (
                item["answer_tell_position"]
                if mode == "oracle-correct"
                else 1 - item["answer_tell_position"]
            )
            model.configure_stepwise_consumer(
                StepwiseConsumerRequest(mode, ids, None if mode == "zero-replay" else ids[index])
            )
            with front_padding():
                result = c2.generated(runtime, item["turns"][2])
            if (
                c2.parameter_digest(model) != before
                or model._persistent_stepwise_carrier.episode_manifest()
                != support["episode_manifest"]
            ):
                raise RuntimeError("query changed weights or carrier")
            tail = old[item["id"], mode]
            audit = result["consumer"]
            unchanged = mode == "zero-replay" or audit["padding_steps"] == 0
            if audit["replay_steps"] != tail["consumer"]["replay_steps"]:
                raise RuntimeError("padding placement changed dose")
            if unchanged and (
                result["raw_hex"] != tail["raw_hex"]
                or result["output"] != tail["output"]
                or audit["first_probabilities_after"]
                != tail["consumer"]["first_probabilities_after"]
            ):
                raise RuntimeError("zero or unpadded control changed")
            row = {
                "item_id": item["id"],
                "arm": mode,
                "kind": item["kind"],
                "position": item["answer_tell_position"],
                "expected_values": item["expected_contains"],
                "strict_hit": c2.strict_value_hit(result["output"], item["expected_contains"]),
                "raw_strict_hit": c2.strict_value_hit(
                    result["raw_text"], item["expected_contains"]
                ),
                "raw_differs_from_tail": result["raw_hex"] != tail["raw_hex"],
                "first_l1_from_tail": sum(
                    abs(a - b)
                    for a, b in zip(
                        audit["first_probabilities_after"],
                        tail["consumer"]["first_probabilities_after"],
                        strict=True,
                    )
                ),
                "unchanged_control": unchanged,
                "parameters_before": before,
                "parameters_after": c2.parameter_digest(model),
                "arm_cpu_seconds": time.process_time() - started,
                **result,
            }
            rows.append(row)
            c2.event(
                EVENTS,
                {
                    "kind": "arm",
                    "item_id": item["id"],
                    "arm": mode,
                    "completed_arms": len(rows),
                    "cpu_seconds": prior + time.process_time(),
                },
            )
            print(
                f"{item['id']} {mode} {len(rows)}/48 CPU={prior+time.process_time():.2f}s",
                flush=True,
            )
        if status != "complete":
            break
    if (
        c2.sha(files["checkpoint"]) != support["checkpoint_sha256"]
        or c2.sha(c2.BASE) != c2.BASE_SHA
    ):
        raise RuntimeError("mother or support changed")
    c2.write_new(
        REPORT,
        {
            "format": "taiji-n1-c2-padding-placement-v1",
            "status": status,
            "command": [sys.executable, *sys.argv],
            "prereg": "plans/reference/PLAN-N1-09_C2_padding_placement_prereg_20261015.md",
            "baseline_sha256": BASELINE_SHA,
            "base_sha256": c2.BASE_SHA,
            "manifest_sha256": c2.MANIFEST_SHA,
            "support_checkpoint_sha256": support["checkpoint_sha256"],
            "item_ids": c2.ITEM_IDS,
            "source_sha256": {
                str(p.relative_to(ROOT)): c2.sha(p)
                for p in (
                    Path(__file__).resolve(),
                    ROOT / "taiji/model.py",
                    ROOT / "taiji/stepwise_consumer.py",
                    Path(c2.__file__),
                )
            },
            "rows": rows,
            "guards": {
                "complete_surface": len(rows) == 48,
                "equal_dose": True,
                "zero_and_unpadded_unchanged": True,
                "weights_unchanged": True,
                "support_unchanged": True,
                "independent_restore": True,
                "query_episodes_absent": True,
            },
            "timing": timing(),
            "reading_limit": "Descriptive padding-placement intervention; no semantic preservation, capability, credit or training verdict",
        },
    )


def run():
    for path in (REPORT, FINAL, AUDIT, EVENTS):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    with EVENTS.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({"kind": "start", "command": [sys.executable, *sys.argv]}) + "\n")
    child_cpu, phases = 0.0, {}
    for phase in ("audit", "query"):
        worker = subprocess.Popen(
            [
                sys.executable,
                "-X",
                "utf8",
                str(Path(__file__).resolve()),
                "--phase",
                phase,
                "--prior-cpu",
                str(child_cpu + time.process_time()),
            ],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        measured = None
        assert worker.stdout is not None
        for line in worker.stdout:
            print(line, end="", flush=True)
            if line.startswith("PAD_TIMING="):
                measured = json.loads(line.split("=", 1)[1])
        worker.wait()
        if measured is not None:
            phases[phase] = measured
            child_cpu += measured["cpu_seconds"]
        if worker.returncode or measured is None:
            c2.event(
                EVENTS,
                {
                    "kind": "tail",
                    "status": "instrument-error",
                    "phase": phase,
                    "returncode": worker.returncode,
                    "total_cpu_seconds": child_cpu + time.process_time(),
                },
            )
            return 1
    result = json.loads(REPORT.read_text(encoding="utf-8"))
    phases["orchestrator"] = timing()
    total = child_cpu + time.process_time()
    status = result["status"] if total <= LIMIT else "budget-overrun"
    c2.write_new(
        FINAL,
        {
            "format": "taiji-n1-c2-padding-placement-final-v1",
            "status": status,
            "command": [sys.executable, *sys.argv],
            "report": str(REPORT.relative_to(ROOT)),
            "report_sha256": c2.sha(REPORT),
            "cpu_limit_seconds": LIMIT,
            "total_cpu_seconds": total,
            "timing_by_process": phases,
            "guards": {**result["guards"], "cpu_within_limit": total <= LIMIT},
            "summary": {
                mode: {
                    "count": len(group := [r for r in result["rows"] if r["arm"] == mode]),
                    "strict_hits": sum(r["strict_hit"] for r in group),
                    "raw_strict_hits": sum(r["raw_strict_hit"] for r in group),
                    "raw_changes_from_tail": sum(r["raw_differs_from_tail"] for r in group),
                    "natural_stops": sum(r["natural_stop"] for r in group),
                }
                for mode in MODES
            },
            "reading_limit": result["reading_limit"],
        },
    )
    c2.event(
        EVENTS,
        {
            "kind": "tail",
            "status": status,
            "completed_arms": len(result["rows"]),
            "total_cpu_seconds": child_cpu + time.process_time(),
            "final_sha256": c2.sha(FINAL),
        },
    )
    print(FINAL.read_text(encoding="utf-8"), flush=True)
    return 0 if status == "complete" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("run", "audit", "query"), default="run")
    parser.add_argument("--prior-cpu", type=float, default=0.0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.phase == "run":
        return run()
    try:
        if args.phase == "audit":
            files = c2.paths("20261015")
            files["audit"] = AUDIT
            c2.items()
            c2.audit_phase(files)
        else:
            query(args.prior_cpu)
    finally:
        print("PAD_TIMING=" + json.dumps(timing()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
