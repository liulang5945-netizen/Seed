"""Run the frozen, zero-training PLAN-N1-07 C2 calibration in separate processes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_WALL_START = time.perf_counter()

import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.seed_runtime import SeedRuntime  # noqa: E402
from taiji.stepwise_consumer import MODES, StepwiseConsumerRequest  # noqa: E402

BASE = ROOT / "output/a31_chunked_self/checkpoint.pt"
MANIFEST = ROOT / "plans/manifests/r2_copy_surface_extension_v3_position_random.json"
BASE_SHA = "ca2628077b21bc4c9a6f2fc06ff410d70311b1218a30b8794ecfd7e5a027d8bb"
MANIFEST_SHA = "b162ff091630f64a8cac986184729ff7894887f4806c923cc12b301ce5ae4fee"
ITEM_IDS = (
    "V023",
    "V007",
    "V002",
    "V010",
    "V008",
    "V016",
    "V005",
    "V013",
    "V009",
    "V001",
    "V027",
    "V003",
    "V006",
    "V014",
    "V004",
    "V012",
)
CPU_LIMIT = 600.0


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parameter_digest(model: Any) -> str:
    digest = hashlib.sha256()
    for tensor in model.parameter_tensors():
        value = tensor.detach().cpu().contiguous().clone()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def items() -> list[dict[str, Any]]:
    if sha(BASE) != BASE_SHA or sha(MANIFEST) != MANIFEST_SHA:
        raise RuntimeError("frozen base or manifest identity changed")
    source = json.loads(MANIFEST.read_text(encoding="utf-8"))["dimensions"]["X"]["items"]
    selected = [
        min(
            (
                item
                for item in source
                if item["kind"] == kind and item["answer_tell_position"] == position
            ),
            key=lambda item: item["id"],
        )
        for kind in sorted({item["kind"] for item in source})
        for position in (0, 1)
    ]
    core = [
        [str(item["id"]), list(item["turns"]), list(item["expected_contains"])] for item in selected
    ]
    fingerprint = hashlib.sha256(
        json.dumps(core, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    if tuple(item["id"] for item in selected) != ITEM_IDS or fingerprint != "f47dd69836fd4938":
        raise RuntimeError("frozen calibration selection changed")
    return selected


def paths(run_id: str) -> dict[str, Path]:
    if not re.fullmatch(r"[0-9]{8}(?:_[a-z0-9]+)?", run_id):
        raise ValueError("run id must be YYYYMMDD with an optional lowercase suffix")
    return {
        "checkpoint": ROOT / f"output/n1_c2_calibration_{run_id}/support.pt",
        "support": ROOT / f"reports/taiji_n1_c2_support_{run_id}.json",
        "audit": ROOT / f"reports/taiji_n1_c2_restore_{run_id}.json",
        "query": ROOT / f"reports/taiji_n1_c2_calibration_{run_id}.json",
        "final": ROOT / f"reports/taiji_n1_c2_final_{run_id}.json",
        "events": ROOT / f"reports/taiji_n1_c2_events_{run_id}.jsonl",
    }


def timing() -> dict[str, float]:
    return {
        "cpu_seconds": time.process_time(),
        "wall_seconds": time.perf_counter() - SCRIPT_WALL_START,
    }


def write_new(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def event(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")


def timed_event(path: Path, phase: str, **kwargs: Any) -> None:
    event(path, {"phase": phase, **kwargs, **timing()})


def generated(runtime: SeedRuntime, prompt: str) -> dict[str, Any]:
    model = runtime.model.substrate
    original = runtime.model.generate_input
    capture: dict[str, Any] = {}

    def wrapped(*args: Any, **kwargs: Any) -> bytes:
        raw = original(*args, **kwargs)
        capture.update(
            raw_hex=raw.hex(),
            raw_text=raw.decode("utf-8", errors="strict"),
            output_bytes=len(raw),
            final_tick=model.tick,
        )
        return raw

    runtime.model.generate_input = wrapped
    try:
        capture["output"] = runtime.chat(
            prompt,
            history=[],
            max_length=256,
            learn=False,
            repetition_penalty=2.0,
            structural_closure=False,
        )
    finally:
        runtime.model.generate_input = original
    audit = model.last_stepwise_consumer_audit
    replay_steps = 0 if audit is None else int(audit["replay_steps"])
    prompt_steps = len(runtime._serialize(prompt, []).encode("utf-8")) + 1
    emitted = capture["final_tick"] - prompt_steps - replay_steps
    if not 0 <= emitted <= 256:
        raise RuntimeError("cannot reconcile prompt, replay and emitted step counts")
    capture.update(
        emitted_steps=emitted,
        natural_stop=emitted < 256,
        exhausted_budget=emitted == 256,
        consumer=audit,
    )
    return capture


def support_phase(selected: list[dict[str, Any]], files: dict[str, Path], prior: float) -> None:
    runtime = SeedRuntime.load(BASE)
    model = runtime.model.substrate
    if runtime.chat_language_backend != "native-readable" or model.copy_circuit is not None:
        raise RuntimeError("calibration mother must use the native organ without a copy circuit")
    if model._persistent_stepwise_carrier.episode_count:
        raise RuntimeError("mother already contains support episodes")
    before = parameter_digest(model)
    model.persistent_stepwise_carrier = True
    rows = []
    for item in selected:
        for index, prompt in enumerate(item["turns"][:2]):
            if prior + timing()["cpu_seconds"] >= CPU_LIMIT:
                raise RuntimeError("CPU budget reached before all support episodes were written")
            result = generated(runtime, prompt)
            episode_id = model._persistent_stepwise_carrier.episode_ids[-1]
            row = {
                "item_id": item["id"],
                "support_index": index,
                "episode_id": episode_id,
                "prompt": prompt,
                **result,
            }
            rows.append(row)
            timed_event(files["events"], "support", kind="support", row=row)
    model.persistent_stepwise_carrier = False
    after = parameter_digest(model)
    if before != after or len(rows) != 32 or model._persistent_stepwise_carrier.episode_count != 32:
        raise RuntimeError("support write changed parameters or episode count")
    if files["checkpoint"].exists():
        raise FileExistsError(files["checkpoint"])
    runtime.save(files["checkpoint"])
    payload = {
        "format": "taiji-n1-c2-support-v1",
        "items": selected,
        "rows": rows,
        "checkpoint": str(files["checkpoint"]),
        "checkpoint_sha256": sha(files["checkpoint"]),
        "base_sha256": BASE_SHA,
        "manifest_sha256": MANIFEST_SHA,
        "model_parameter_digest": before,
        "episode_manifest": model._persistent_stepwise_carrier.episode_manifest(),
        "recording_enabled_after_save": model.persistent_stepwise_carrier,
        "timing": timing(),
    }
    write_new(files["support"], payload)


def audit_phase(files: dict[str, Path]) -> None:
    from scripts.training.audit_taiji_n1_persistent_stepwise_carrier import (
        _carrier_from_checkpoint,
        _load_weights_only,
        _payload_manifest,
    )

    support = json.loads(files["support"].read_text(encoding="utf-8"))
    if sha(files["checkpoint"]) != support["checkpoint_sha256"]:
        raise RuntimeError("support checkpoint identity changed")
    saved = _load_weights_only(files["checkpoint"])
    for location in ("native", "compatibility"):
        payload = _carrier_from_checkpoint(saved, location)
        if payload is None or _payload_manifest(payload) != support["episode_manifest"]:
            raise RuntimeError(f"{location} weights-only carrier differs from saved evidence")
    del saved
    runtime = SeedRuntime.load(files["checkpoint"])
    model = runtime.model.substrate
    carrier = model._persistent_stepwise_carrier
    if (
        carrier.episode_count != 32
        or carrier.context_dim != 1152
        or model.persistent_stepwise_carrier
        or model._stepwise_consumer_request is not None
        or carrier.episode_manifest() != support["episode_manifest"]
        or parameter_digest(model) != support["model_parameter_digest"]
    ):
        raise RuntimeError("independent restore failed carrier/weight/default-off guards")
    for row in support["episode_manifest"]:
        vectors = model.read_persistent_stepwise_episode(row["episode_id"])
        if (
            vectors.dtype != torch.float32
            or vectors.shape != (row["steps"], 1152)
            or not torch.isfinite(vectors).all().item()
        ):
            raise RuntimeError("restored support has an invalid vector trajectory")
    if tuple(model.read_persistent_stepwise_episode("absent-query").shape) != (0, 1152):
        raise RuntimeError("missing episode silently fell back")
    if len({row["episode_id"] for row in support["rows"]}) != 32:
        raise RuntimeError("support membership does not match item-local candidate pools")
    write_new(
        files["audit"],
        {
            "format": "taiji-n1-c2-restore-v1",
            "episodes": 32,
            "steps": carrier.total_steps,
            "context_dim": 1152,
            "checkpoint_sha256": sha(files["checkpoint"]),
            "weights_only": True,
            "source_and_content_equal": True,
            "recording_disabled": True,
            "consumer_disabled": True,
            "missing_episode_returns_empty": True,
            "query_episodes_absent": True,
            "timing": timing(),
        },
    )


def strict_value_hit(text: str, expected: list[str]) -> bool:
    return text.strip(" \t\r\n。.!！?？《》\"'“”‘’") in expected


def validate_resume(report: dict[str, Any], selected: list[dict[str, Any]], digest: str) -> None:
    if (
        report.get("status") != "budget-stop"
        or report.get("base_sha256") != BASE_SHA
        or report.get("manifest_sha256") != MANIFEST_SHA
        or tuple(report.get("item_ids", ())) != ITEM_IDS
    ):
        raise ValueError("only an identical budget-stopped frozen surface can resume")
    allowed = {(item["id"], mode) for item in selected for mode in MODES}
    seen = set()
    for row in report["rows"]:
        key = (row["item_id"], row["arm"])
        if key not in allowed or key in seen:
            raise ValueError("resume contains duplicate or out-of-surface measurements")
        if row["parameters_before"] != digest or row["parameters_after"] != digest:
            raise ValueError("resume contains a measurement from different weights")
        seen.add(key)
    if report.get("mechanical_repair"):
        raise ValueError("only one mechanical repair is permitted")


def query_phase(
    selected: list[dict[str, Any]],
    files: dict[str, Path],
    prior: float,
    resume_report: Path | None = None,
) -> None:
    support = json.loads(files["support"].read_text(encoding="utf-8"))
    restored = json.loads(files["audit"].read_text(encoding="utf-8"))
    if restored["checkpoint_sha256"] != sha(files["checkpoint"]):
        raise RuntimeError("audited support checkpoint changed before consumption")
    source = json.loads(MANIFEST.read_text(encoding="utf-8"))["dimensions"]["X"]["items"]
    known_values = sorted({value for item in source for value in item["expected_contains"]})
    rows: list[dict[str, Any]] = []
    if resume_report is not None:
        previous_report = json.loads(resume_report.read_text(encoding="utf-8"))
        validate_resume(previous_report, selected, support["model_parameter_digest"])
        rows = list(previous_report["rows"])
    prior_count = len(rows)
    completed = {(row["item_id"], row["arm"]) for row in rows}
    status = "complete"
    for item in selected:
        candidates = tuple(
            row["episode_id"] for row in support["rows"] if row["item_id"] == item["id"]
        )
        if len(candidates) != 2:
            raise RuntimeError("item-local support pool is incomplete")
        correct = candidates[item["answer_tell_position"]]
        wrong = candidates[1 - item["answer_tell_position"]]
        order = sorted(
            MODES,
            key=lambda mode: hashlib.sha256((item["id"] + ":" + mode).encode("utf-8")).hexdigest(),
        )
        for mode in order:
            if (item["id"], mode) in completed:
                continue
            used = prior + timing()["cpu_seconds"]
            remaining = 112 - len(rows)
            new_rows = rows[prior_count:]
            projected = used + remaining * sum(row["arm_cpu_seconds"] for row in new_rows) / max(
                1, len(new_rows)
            )
            if used >= CPU_LIMIT or (used >= 480 and projected > CPU_LIMIT):
                status = "budget-stop"
                break
            arm_start = time.process_time()
            runtime = SeedRuntime.load(files["checkpoint"])
            model = runtime.model.substrate
            if model.persistent_stepwise_carrier or model._stepwise_consumer_request is not None:
                raise RuntimeError("consumer restore did not start with both switches disabled")
            before = parameter_digest(model)
            if before != support["model_parameter_digest"]:
                raise RuntimeError("arm does not share the frozen mother weights")
            if mode == "no-replay":
                request = StepwiseConsumerRequest(mode)
            else:
                controlled = mode not in {"model-top1", "zero-replay"}
                episode_id = (wrong if mode == "wrong-episode" else correct) if controlled else None
                request = StepwiseConsumerRequest(
                    mode,
                    candidates,
                    episode_id,
                    int.from_bytes(hashlib.sha256(item["id"].encode()).digest()[:8], "big"),
                )
            model.configure_stepwise_consumer(request)
            result = generated(runtime, item["turns"][2])
            audit = result["consumer"]
            after = parameter_digest(model)
            if (
                before != after
                or model._persistent_stepwise_carrier.episode_manifest()
                != support["episode_manifest"]
            ):
                raise RuntimeError("consumer changed weights or wrote a query episode")
            row = {
                "item_id": item["id"],
                "kind": item["kind"],
                "answer_tell_position": item["answer_tell_position"],
                "arm": mode,
                "expected_values": item["expected_contains"],
                "selector_correct": (
                    None if mode == "no-replay" else audit["picked_episode_id"] == correct
                ),
                "strict_hit": strict_value_hit(result["output"], item["expected_contains"]),
                "raw_strict_hit": strict_value_hit(result["raw_text"], item["expected_contains"]),
                "known_wrong_or_extra_values": [
                    value
                    for value in known_values
                    if value not in item["expected_contains"] and value in result["output"]
                ],
                "parameters_before": before,
                "parameters_after": after,
                "arm_cpu_seconds": time.process_time() - arm_start,
                **result,
            }
            rows.append(row)
            timed_event(files["events"], "query", kind="arm", row=row)
            print(
                f"{item['id']} {mode} done cpu_total={prior + timing()['cpu_seconds']:.1f}s",
                flush=True,
            )
        if status != "complete":
            break
    if sha(BASE) != BASE_SHA or sha(files["checkpoint"]) != support["checkpoint_sha256"]:
        raise RuntimeError("base or support checkpoint changed during consumption")
    if status == "complete" and (
        len(rows) != 112
        or any(len([row for row in rows if row["item_id"] == item["id"]]) != 7 for item in selected)
    ):
        raise RuntimeError("complete verdict has an incomplete paired surface")
    summary = {
        mode: {
            "count": len(group := [row for row in rows if row["arm"] == mode]),
            "strict_hits": sum(row["strict_hit"] for row in group),
            "raw_strict_hits": sum(row["raw_strict_hit"] for row in group),
            "natural_stops": sum(row["natural_stop"] for row in group),
        }
        for mode in sorted(MODES)
    }
    strata = {
        f"{kind}:position{position}": [
            row for row in rows if row["kind"] == kind and row["answer_tell_position"] == position
        ]
        for kind in sorted({item["kind"] for item in selected})
        for position in (0, 1)
    }
    paired = []
    for item in selected:
        by_arm = {row["arm"]: row for row in rows if row["item_id"] == item["id"]}
        if len(by_arm) != 7:
            continue
        oracle = by_arm["oracle-correct"]
        paired.append(
            {
                "item_id": item["id"],
                "model_selector_correct": by_arm["model-top1"]["selector_correct"],
                "oracle_raw_differs": {
                    mode: oracle["raw_hex"] != by_arm[mode]["raw_hex"]
                    for mode in (
                        "no-replay",
                        "zero-replay",
                        "wrong-episode",
                        "shuffle-correct",
                        "pooled-correct",
                        "model-top1",
                    )
                },
            }
        )
    phase_timing = timing()
    if prior + phase_timing["cpu_seconds"] > CPU_LIMIT:
        status = "budget-overrun"
    payload = {
        "format": "taiji-n1-c2-calibration-v1",
        "status": status,
        "command": [sys.executable, *sys.argv],
        "prereg": "plans/reference/PLAN-N1-07_C2_consumer_calibration_prereg_20261014.md",
        "base_sha256": BASE_SHA,
        "manifest_sha256": MANIFEST_SHA,
        "item_ids": ITEM_IDS,
        "items_fingerprint": "f47dd69836fd4938",
        "threads": torch.get_num_threads(),
        "cpu_limit_seconds": CPU_LIMIT,
        "timing": phase_timing,
        "total_cpu_seconds": prior + phase_timing["cpu_seconds"],
        "cpu_measurement": "sum of process_time from separate scripts, including imports",
        "support": support,
        "restore_audit": restored,
        "rows": rows,
        "summary": summary,
        "strata": strata,
        "paired": paired,
        "strict_scoring": "whole emitted text equals expected value after trimming outer whitespace/punctuation; no substring credit",
        "extra_fact_reading_limit": "known value lexicon flags only; full raw and organ outputs are retained for semantic inspection",
        "replay_implementation": "zero sensory; learn/restructure/adapt_homeostasis false; live regions and native predictive readout refreshed before first byte",
        "guards": {
            "complete_surface": status == "complete",
            "weights_unchanged": True,
            "support_unchanged": True,
            "query_episodes_absent": True,
            "base_unchanged": True,
            "independent_restore": True,
            "cpu_within_limit": prior + phase_timing["cpu_seconds"] <= CPU_LIMIT,
        },
        "events": [{"kind": "tail", "status": status, "completed_arms": len(rows)}],
        "reading_limit": "descriptive single-organ calibration, no C2/S1/N1 capability verdict",
    }
    write_new(files["query"], payload)


def run(
    selected: list[dict[str, Any]],
    files: dict[str, Path],
    run_id: str,
    resume_run_id: str | None = None,
) -> int:
    resume_report = None
    prior_run_cpu = 0.0
    if resume_run_id is not None:
        original = paths(resume_run_id)
        resume_report = original["final"]
        previous = json.loads(resume_report.read_text(encoding="utf-8"))
        validate_resume(previous, selected, previous["support"]["model_parameter_digest"])
        prior_run_cpu = float(previous["total_cpu_seconds"])
        if not 0 <= prior_run_cpu < CPU_LIMIT:
            raise ValueError("resume must use the remaining original CPU budget")
        for key in ("support", "audit", "checkpoint"):
            files[key] = original[key]
    new_paths = (
        (files[key] for key in ("query", "final", "events")) if resume_report else files.values()
    )
    for path in new_paths:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite run artifact: {path}")
    with files["events"].open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps({"kind": "start", "command": [sys.executable, *sys.argv]}) + "\n")
    child_cpu = prior_run_cpu
    phases = {}
    for phase in (("query",) if resume_report else ("support", "audit", "query")):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--phase",
            phase,
            "--run-id",
            run_id,
            "--prior-cpu-seconds",
            str(child_cpu + timing()["cpu_seconds"]),
        ]
        if resume_report is not None:
            command.extend(
                ["--resume-report", str(resume_report), "--resume-run-id", resume_run_id]
            )
        worker = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        phase_measurement = None
        assert worker.stdout is not None
        for line in worker.stdout:
            print(line, end="", flush=True)
            if line.startswith("C2_PHASE_TIMING="):
                phase_measurement = json.loads(line.split("=", 1)[1])
        worker.wait()
        if phase_measurement is None:
            raise RuntimeError("worker exited without process CPU measurement")
        phases[phase] = phase_measurement
        child_cpu += phase_measurement["cpu_seconds"]
        if worker.returncode:
            timed_event(
                files["events"],
                phase,
                kind="tail",
                status="instrument-error",
                returncode=worker.returncode,
                total_cpu_seconds=child_cpu + timing()["cpu_seconds"],
            )
            write_new(
                files["final"],
                {
                    "format": "taiji-n1-c2-calibration-v1",
                    "status": "instrument-error",
                    "failed_phase": phase,
                    "timing_by_process": phases,
                    "events_path": str(files["events"]),
                    "total_cpu_seconds": child_cpu + timing()["cpu_seconds"],
                },
            )
            return worker.returncode
    final = json.loads(files["query"].read_text(encoding="utf-8"))
    phases["orchestrator"] = timing()
    final.update(
        command=[sys.executable, *sys.argv],
        source_sha256={
            str(path.relative_to(ROOT)): sha(path)
            for path in (
                ROOT / "taiji/model.py",
                ROOT / "taiji/stepwise_consumer.py",
                Path(__file__).resolve(),
            )
        },
        timing_by_process=phases,
        total_cpu_seconds=child_cpu + phases["orchestrator"]["cpu_seconds"],
        total_wall_seconds=phases["orchestrator"]["wall_seconds"],
        cpu_measurement="sum process_time of orchestrator and three phase processes; includes startup/imports and phase serialization; endpoint before final report serialization",
    )
    if resume_report is not None:
        final["mechanical_repair"] = {
            "count": 1,
            "reason": "replace Python storage-byte iteration with byte-identical native extraction",
            "prior_report": str(resume_report),
            "prior_report_sha256": sha(resume_report),
            "prior_run_cpu_seconds": prior_run_cpu,
            "prior_completed_arms": len(previous["rows"]),
            "duplicate_measurements": 0,
        }
    final["guards"]["cpu_within_limit"] = final["total_cpu_seconds"] <= CPU_LIMIT
    if not final["guards"]["cpu_within_limit"]:
        final["status"] = "budget-overrun"
    write_new(files["final"], final)
    timed_event(
        files["events"],
        "run",
        kind="tail",
        status=final["status"],
        completed_arms=len(final["rows"]),
        total_cpu_seconds=child_cpu + timing()["cpu_seconds"],
    )
    print(
        json.dumps(
            {
                "report": str(files["final"]),
                "status": final["status"],
                "total_cpu_seconds": child_cpu + timing()["cpu_seconds"],
                "wall_seconds": timing()["wall_seconds"],
                "summary": final["summary"],
            }
        )
    )
    return 0 if final["status"] == "complete" else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("run", "support", "audit", "query"), default="run")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--prior-cpu-seconds", type=float, default=0.0)
    parser.add_argument("--resume-run-id")
    parser.add_argument("--resume-report", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    files = paths(args.run_id)
    selected = items()
    if args.phase == "run":
        return run(selected, files, args.run_id, args.resume_run_id)
    if args.resume_report is not None:
        original = paths(args.resume_run_id)
        for key in ("support", "audit", "checkpoint"):
            files[key] = original[key]
    try:
        if args.phase == "support":
            support_phase(selected, files, args.prior_cpu_seconds)
        elif args.phase == "audit":
            audit_phase(files)
        else:
            query_phase(selected, files, args.prior_cpu_seconds, args.resume_report)
    finally:
        print("C2_PHASE_TIMING=" + json.dumps(timing()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
