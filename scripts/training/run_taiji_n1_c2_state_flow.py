"""Observe the frozen PLAN-N1-08 forward path without changing product code."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path
from typing import Any

START = time.perf_counter()
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from api.seed_runtime import SeedRuntime  # noqa: E402
from scripts.training import run_taiji_n1_c2_calibration as c2  # noqa: E402
from taiji.stepwise_consumer import StepwiseConsumerRequest  # noqa: E402

REPORT = ROOT / "reports/taiji_n1_c2_state_flow_20261015.json"
FINAL = ROOT / "reports/taiji_n1_c2_state_flow_final_20261015.json"
SNAPSHOTS = ROOT / "reports/taiji_n1_c2_state_flow_20261015_snapshots.json.gz"
AUDIT = ROOT / "reports/taiji_n1_c2_state_flow_restore_20261015.json"
EVENTS = ROOT / "reports/taiji_n1_c2_state_flow_20261015_events.jsonl"
MODES = ("oracle-correct", "wrong-episode", "zero-replay")
LIMIT = 300.0


def timing() -> dict[str, float]:
    return {"cpu_seconds": time.process_time(), "wall_seconds": time.perf_counter() - START}


def vector(value: torch.Tensor) -> list[float]:
    return value.detach().cpu().tolist()


def regions_snapshot(regions: Any) -> list[dict[str, Any]]:
    return [
        {field.name: vector(getattr(region, field.name)) for field in fields(region)}
        for region in regions
    ]


def state_snapshot(model: Any) -> dict[str, Any]:
    return {
        "tick": model.tick,
        "cue": vector(model.cortical_cue()),
        "regions": regions_snapshot(model._state.regions),
        "context": vector(model._state.motor_context),
        "probabilities": vector(model._state.motor_probabilities),
    }


@contextmanager
def observe_state_flow(model: Any):
    """Instance-only observers; call each original exactly once and return it intact."""
    trace: dict[str, Any] = {"replay": [], "snapshots": {}, "generation": []}
    replaying = False
    answering = False
    saved = []

    def bind(owner, name, wrapper):
        saved.append((owner, name, name in owner.__dict__, owner.__dict__.get(name)))
        setattr(owner, name, wrapper)

    consume = model._consume_stepwise_before_answer
    fabric_step = model.fabric.step
    encode = model.predictive_context.encode
    observe = model.observe

    def consume_observer(step):
        nonlocal replaying, answering
        trace["query"] = state_snapshot(model)
        request = model._stepwise_consumer_request
        total = max(
            len(model.read_persistent_stepwise_episode(key))
            for key in request.candidate_episode_ids
        )
        trace["snapshot_positions"] = sorted(
            {1, math.ceil(total / 4), math.ceil(total / 2), math.ceil(3 * total / 4), total}
        )
        replaying = True
        try:
            result = consume(step)
        finally:
            replaying = False
        trace["after_readout"] = state_snapshot(model)
        answering = True
        return result

    def fabric_observer(sensory, regions, **kwargs):
        if not replaying:
            return fabric_step(sensory, regions, **kwargs)
        before = [
            {name: getattr(region, name).clone() for name in ("activity", "trace", "membrane")}
            for region in regions
        ]
        result = fabric_step(sensory, regions, **kwargs)
        feedback = kwargs["episodic_feedback"]
        index = len(trace["replay"]) + 1
        trace["replay"].append(
            {
                "step": index,
                "feedback_norm": float(feedback.norm()),
                "feedback_sha256": hashlib.sha256(
                    feedback.detach().cpu().contiguous().numpy().tobytes()
                ).hexdigest(),
                "regions": [
                    {
                        name: {
                            "norm": float(getattr(region, name).norm()),
                            "delta_norm": float((getattr(region, name) - prior[name]).norm()),
                        }
                        for name in prior
                    }
                    for region, prior in zip(result[0], before, strict=True)
                ],
            }
        )
        if index in trace["snapshot_positions"]:
            trace["snapshots"][str(index)] = regions_snapshot(result[0])
        return result

    def encode_observer(*args, **kwargs):
        result = encode(*args, **kwargs)
        if replaying:
            trace["encoded_context"] = vector(result[0])
        return result

    def observe_observer(symbol, **kwargs):
        result = observe(symbol, **kwargs)
        if answering and len(trace["generation"]) < 16:
            trace["generation"].append(
                {
                    "emitted_index": len(trace["generation"]),
                    "input_symbol": int(symbol),
                    "context": vector(model._state.motor_context),
                    "probabilities": vector(result.probabilities),
                }
            )
        return result

    bind(model, "_consume_stepwise_before_answer", consume_observer)
    bind(model.fabric, "step", fabric_observer)
    bind(model.predictive_context, "encode", encode_observer)
    bind(model, "observe", observe_observer)
    try:
        yield trace
    finally:
        for owner, name, present, value in reversed(saved):
            if present:
                setattr(owner, name, value)
            else:
                delattr(owner, name)


def assert_noninterference(plain: dict[str, Any], traced: dict[str, Any]) -> None:
    for key in ("raw_hex", "output", "consumer", "emitted_steps", "natural_stop"):
        if plain[key] != traced[key]:
            raise RuntimeError(f"state-flow observer changed {key}")


def query(prior: float) -> None:
    files = c2.paths("20261015")
    support = json.loads(files["support"].read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if audit["checkpoint_sha256"] != c2.sha(files["checkpoint"]):
        raise RuntimeError("independently audited support changed")
    selected = c2.items()
    rows, snapshots = [], []
    status = "complete"
    for item in selected:
        ids = tuple(row["episode_id"] for row in support["rows"] if row["item_id"] == item["id"])
        for mode in MODES:
            used = prior + time.process_time()
            estimate = used + (48 - len(rows)) * sum(row["pair_cpu_seconds"] for row in rows) / max(
                1, len(rows)
            )
            if used >= LIMIT or (used >= 240 and estimate > LIMIT):
                status = "budget-stop"
                break
            started = time.process_time()
            outputs = []
            trace = None
            for instrumented in (False, True):
                runtime = SeedRuntime.load(files["checkpoint"])
                model = runtime.model.substrate
                if (
                    model.persistent_stepwise_carrier
                    or model._stepwise_consumer_request is not None
                ):
                    raise RuntimeError("checkpoint restored with live switches")
                if c2.parameter_digest(model) != support["model_parameter_digest"]:
                    raise RuntimeError("source weight identity changed")
                index = (
                    item["answer_tell_position"]
                    if mode == "oracle-correct"
                    else 1 - item["answer_tell_position"]
                )
                model.configure_stepwise_consumer(
                    StepwiseConsumerRequest(
                        mode, ids, None if mode == "zero-replay" else ids[index]
                    )
                )
                if instrumented:
                    with observe_state_flow(model) as trace:
                        output = c2.generated(runtime, item["turns"][2])
                else:
                    output = c2.generated(runtime, item["turns"][2])
                if (
                    c2.parameter_digest(model) != support["model_parameter_digest"]
                    or model._persistent_stepwise_carrier.episode_manifest()
                    != support["episode_manifest"]
                ):
                    raise RuntimeError("query changed weights or support")
                outputs.append(output)
            assert_noninterference(*outputs)
            assert trace is not None
            if len(trace["replay"]) != outputs[1]["consumer"]["replay_steps"]:
                raise RuntimeError("replay observation coverage incomplete")
            if len(trace["generation"]) != min(16, outputs[1]["emitted_steps"]):
                raise RuntimeError("generation observation coverage incomplete")
            row = {
                "item_id": item["id"],
                "kind": item["kind"],
                "position": item["answer_tell_position"],
                "arm": mode,
                "noninterference": True,
                "pair_cpu_seconds": time.process_time() - started,
                **outputs[1],
            }
            rows.append(row)
            snapshots.append({"item_id": item["id"], "arm": mode, **trace})
            c2.event(
                EVENTS,
                {
                    "kind": "pair",
                    "item_id": item["id"],
                    "arm": mode,
                    "completed_pairs": len(rows),
                    "total_cpu_seconds": prior + time.process_time(),
                },
            )
            print(
                f"{item['id']} {mode} pair={len(rows)}/48 CPU={prior + time.process_time():.2f}s",
                flush=True,
            )
        if status != "complete":
            break
    if (
        c2.sha(files["checkpoint"]) != support["checkpoint_sha256"]
        or c2.sha(c2.BASE) != c2.BASE_SHA
    ):
        raise RuntimeError("checkpoint identity changed during diagnostic")
    with (
        SNAPSHOTS.open("xb") as stream,
        gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as compressed,
    ):
        compressed.write(
            json.dumps(
                snapshots, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            ).encode("utf-8")
        )
    c2.write_new(
        REPORT,
        {
            "format": "taiji-n1-c2-state-flow-v1",
            "status": status,
            "command": [sys.executable, *sys.argv],
            "prereg": "plans/reference/PLAN-N1-08_C2_state_flow_diagnostic_prereg_20261015.md",
            "cpu_limit_seconds": LIMIT,
            "total_cpu_seconds": prior + time.process_time(),
            "item_ids": c2.ITEM_IDS,
            "base_sha256": c2.BASE_SHA,
            "manifest_sha256": c2.MANIFEST_SHA,
            "support_checkpoint_sha256": support["checkpoint_sha256"],
            "restore_audit": str(AUDIT.relative_to(ROOT)),
            "source_sha256": {
                str(p.relative_to(ROOT)): c2.sha(p)
                for p in (
                    Path(__file__).resolve(),
                    ROOT / "taiji/model.py",
                    ROOT / "taiji/stepwise_consumer.py",
                    Path(c2.__file__),
                )
            },
            "snapshots": str(SNAPSHOTS.relative_to(ROOT)),
            "snapshots_sha256": c2.sha(SNAPSHOTS),
            "guards": {
                "complete_surface": len(rows) == 48,
                "noninterference": True,
                "weights_unchanged": True,
                "support_unchanged": True,
                "query_episodes_absent": True,
                "independent_restore": True,
            },
            "rows": rows,
            "reading_limit": "numerical state transfer only; no semantic preservation, capability or credit verdict",
            "timing": timing(),
        },
    )


def run() -> int:
    for path in (REPORT, FINAL, SNAPSHOTS, AUDIT, EVENTS):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    with EVENTS.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({"kind": "start", "command": [sys.executable, *sys.argv]}) + "\n")
    child_cpu = 0.0
    phases = {}
    for phase in ("audit", "query"):
        command = [
            sys.executable,
            "-X",
            "utf8",
            str(Path(__file__).resolve()),
            "--phase",
            phase,
            "--prior-cpu",
            str(child_cpu + time.process_time()),
        ]
        worker = subprocess.Popen(
            command,
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
            if line.startswith("FLOW_TIMING="):
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
                    "timing_by_process": phases,
                    "total_cpu_seconds": child_cpu + time.process_time(),
                },
            )
            return 1
    phases["orchestrator"] = timing()
    total = child_cpu + time.process_time()
    result = json.loads(REPORT.read_text(encoding="utf-8"))
    status = result["status"] if total <= LIMIT else "budget-overrun"
    c2.write_new(
        FINAL,
        {
            "format": "taiji-n1-c2-state-flow-final-v1",
            "status": status,
            "command": [sys.executable, *sys.argv],
            "report": str(REPORT.relative_to(ROOT)),
            "report_sha256": c2.sha(REPORT),
            "snapshots_sha256": c2.sha(SNAPSHOTS),
            "completed_pairs": len(result["rows"]),
            "queries": 2 * len(result["rows"]),
            "cpu_limit_seconds": LIMIT,
            "total_cpu_seconds": total,
            "timing_by_process": phases,
            "guards": {**result["guards"], "cpu_within_limit": total <= LIMIT},
            "reading_limit": result["reading_limit"],
        },
    )
    c2.event(
        EVENTS,
        {
            "kind": "tail",
            "status": status,
            "completed_pairs": len(result["rows"]),
            "total_cpu_seconds": total,
            "timing_by_process": phases,
            "report_sha256": c2.sha(REPORT),
            "snapshot_sha256": c2.sha(SNAPSHOTS),
        },
    )
    print(json.dumps({"status": status, "pairs": len(result["rows"]), "total_cpu_seconds": total}))
    return 0 if status == "complete" else 2


def main() -> int:
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
        print("FLOW_TIMING=" + json.dumps(timing()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
