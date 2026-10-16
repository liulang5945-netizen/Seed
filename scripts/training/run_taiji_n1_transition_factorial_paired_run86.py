"""Run the frozen N1-84 training protocol as an on-demand run86b task."""

from __future__ import annotations

import argparse
import builtins
import hashlib
import json
import os
import runpy
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/training/run_taiji_n1_transition_factorial_paired.py"
LAUNCHER = ROOT / "scripts/training/launch_taiji_n1_transition_factorial_run86.ps1"
FROZEN_ATTEMPT = ROOT / "reports/taiji_n1_transition_factorial_paired_train_20261016_attempt.json"
FROZEN_ATTEMPT_SHA256 = "6a3124a1293c33dcbdc45bc55fb34971200236cb4673e52c9c981335e0bc6234"

REPORT = ROOT / "reports/taiji_n1_transition_factorial_run86b.json"
ATTEMPT = ROOT / "reports/taiji_n1_transition_factorial_run86b_attempt.json"
LAUNCH_MANIFEST = ROOT / "reports/taiji_n1_transition_factorial_run86b_launcher.json"
LAUNCH_RETURNED = ROOT / "reports/taiji_n1_transition_factorial_run86b_launcher_returned.json"
WORKER_MANIFEST = ROOT / "reports/taiji_n1_transition_factorial_run86b_worker.json"
RUN86_STDOUT = ROOT / "reports/taiji_n1_transition_factorial_run86b.stdout.log"
RUN86_STDERR = ROOT / "reports/taiji_n1_transition_factorial_run86b.stderr.log"
COMPLETION = ROOT / "reports/taiji_n1_transition_factorial_run86b_completion.json"

PREFLIGHT_STEM = ROOT / "reports/taiji_n1_transition_factorial_run86_runtime_preflight_scheduled_retry2"
PREFLIGHT_REPORT = PREFLIGHT_STEM.with_suffix(".json")
PREFLIGHT_HEARTBEATS = PREFLIGHT_STEM.with_suffix(".jsonl")
PREFLIGHT_LAUNCH_MANIFEST = PREFLIGHT_STEM.with_name(
    PREFLIGHT_STEM.name + "_launcher.json"
)
PREFLIGHT_LAUNCH_RETURNED = PREFLIGHT_STEM.with_name(
    PREFLIGHT_STEM.name + "_returned.json"
)
PREFLIGHT_WORKER = PREFLIGHT_STEM.with_name(PREFLIGHT_STEM.name + "_worker.json")
PREFLIGHT_STDOUT = PREFLIGHT_STEM.with_name(PREFLIGHT_STEM.name + ".stdout.log")
PREFLIGHT_STDERR = PREFLIGHT_STEM.with_name(PREFLIGHT_STEM.name + ".stderr.log")
HANDOFF_STEM = ROOT / "reports/taiji_n1_transition_factorial_run86b_return_preflight"
HANDOFF_REPORT = HANDOFF_STEM.with_suffix(".json")
HANDOFF_LAUNCH_MANIFEST = HANDOFF_STEM.with_name(HANDOFF_STEM.name + "_launcher.json")
HANDOFF_LAUNCH_RETURNED = HANDOFF_STEM.with_name(HANDOFF_STEM.name + "_launcher_returned.json")
HANDOFF_WORKER = HANDOFF_STEM.with_name(HANDOFF_STEM.name + "_worker.json")
HANDOFF_STDOUT = HANDOFF_STEM.with_name(HANDOFF_STEM.name + ".stdout.log")
HANDOFF_STDERR = HANDOFF_STEM.with_name(HANDOFF_STEM.name + ".stderr.log")

TASK_PROBE = "Seed-N1-Run86-Probe-20261016"
TASK_HANDOFF = "Seed-N1-Run86B-Handoff-20261016"
TASK_RUN = "Seed-N1-Run86B-20261016"
CHECKPOINTS = ROOT / "data/checkpoints/run86b"
PREFLIGHT_CPU_LIMIT_SECONDS = 60


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new_json(path: Path, value: dict) -> None:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def configure_stdio(probe: bool, handoff: bool = False) -> None:
    paths = (
        (PREFLIGHT_STDOUT, PREFLIGHT_STDERR)
        if probe
        else (HANDOFF_STDOUT, HANDOFF_STDERR)
        if handoff
        else (RUN86_STDOUT, RUN86_STDERR)
    )
    for path in paths:
        if path.exists():
            raise FileExistsError(f"run86 log already exists: {path}")
    sys.stdout = paths[0].open("x", encoding="utf-8", buffering=1)
    sys.stderr = paths[1].open("x", encoding="utf-8", buffering=1)


def task_receipts(probe: bool, handoff: bool = False) -> tuple[dict, dict, Path]:
    if probe:
        manifest_path, returned_path, worker_path, task_name = (
            PREFLIGHT_LAUNCH_MANIFEST,
            PREFLIGHT_LAUNCH_RETURNED,
            PREFLIGHT_WORKER,
            TASK_PROBE,
        )
        expected_mode = "probe"
    elif handoff:
        manifest_path, returned_path, worker_path, task_name = (
            HANDOFF_LAUNCH_MANIFEST,
            HANDOFF_LAUNCH_RETURNED,
            HANDOFF_WORKER,
            TASK_HANDOFF,
        )
        expected_mode = "handoff"
    else:
        manifest_path, returned_path, worker_path, task_name = (
            LAUNCH_MANIFEST,
            LAUNCH_RETURNED,
            WORKER_MANIFEST,
            TASK_RUN,
        )
        expected_mode = "run"
    deadline = time.monotonic() + 60
    while (
        (not manifest_path.exists() or not returned_path.exists())
        and time.monotonic() < deadline
    ):
        time.sleep(0.05)
    if not manifest_path.exists() or not returned_path.exists():
        raise FileNotFoundError("run86 task registration/return receipts are missing")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    returned = json.loads(returned_path.read_text(encoding="utf-8"))
    if (
        manifest.get("mode") != expected_mode
        or manifest.get("task_name") != task_name
        or manifest.get("trigger_count") != 0
        or manifest.get("principal_logon_type") != "Interactive"
        or manifest.get("run_level") != "Limited"
        or returned.get("task_name") != task_name
        or not returned.get("start_succeeded")
    ):
        raise ValueError("run86 task receipts do not match the frozen local task settings")
    for key, path in (
        ("controller_sha256", Path(__file__).resolve()),
        ("runner_sha256", RUNNER),
        ("launcher_sha256", LAUNCHER),
    ):
        if manifest.get(key) != sha256(path):
            raise ValueError(f"run86 source hash changed: {key}")
    write_new_json(
        worker_path,
        dict(
            schema="n1-transition-factorial-run86-worker-v1",
            task_name=task_name,
            process_id=os.getpid(),
            started_utc=utc_now(),
            controller_sha256=sha256(Path(__file__).resolve()),
            runner_sha256=sha256(RUNNER),
            launcher_sha256=sha256(LAUNCHER),
        ),
    )
    return manifest, returned, worker_path


def synthetic_run_handoff_probe() -> int:
    for path in (HANDOFF_REPORT, HANDOFF_WORKER):
        if path.exists():
            raise FileExistsError(f"run86b handoff preflight artifact already exists: {path}")
    started = utc_now()
    cpu_start = time.process_time()
    _, returned, worker_path = task_receipts(probe=False, handoff=True)
    worker = json.loads(worker_path.read_text(encoding="utf-8"))
    report = dict(
        schema="n1-transition-factorial-run86b-return-preflight-v1",
        complete=True,
        synthetic_only=True,
        task_name=TASK_HANDOFF,
        process_id=os.getpid(),
        launcher_returning_utc=returned["launcher_returning_utc"],
        worker_started_utc=worker["started_utc"],
        started_utc=started,
        completed_utc=utc_now(),
        launcher_return_sha256=sha256(HANDOFF_LAUNCH_RETURNED),
        launcher_manifest_sha256=sha256(HANDOFF_LAUNCH_MANIFEST),
        controller_sha256=sha256(Path(__file__).resolve()),
        launcher_sha256=sha256(LAUNCHER),
        runner_sha256=sha256(RUNNER),
        candidate_loaded=False,
        training_material_loaded=False,
        optimizer_updates=0,
        curriculum_queries=0,
        dev_queries=0,
        holdout_queries=0,
        process_cpu_seconds=time.process_time() - cpu_start,
        handoff_only=True,
    )
    write_new_json(HANDOFF_REPORT, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


def synthetic_heartbeat() -> int:
    for path in (PREFLIGHT_REPORT, PREFLIGHT_HEARTBEATS, PREFLIGHT_WORKER):
        if path.exists():
            raise FileExistsError(f"run86 preflight artifact already exists: {path}")
    launcher, returned, worker_path = task_receipts(probe=True)
    returned_utc = returned["launcher_returning_utc"]
    samples = []
    with PREFLIGHT_HEARTBEATS.open("xb") as stream:
        started = utc_now()
        cpu_start = time.process_time()
        for index in range(6):
            sample = dict(index=index, pid=os.getpid(), utc=utc_now())
            samples.append(sample)
            stream.write((json.dumps(sample, sort_keys=True) + "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
            if index < 5:
                time.sleep(0.8)

    ended = utc_now()
    returning = datetime.fromisoformat(returned_utc)
    post_return = [
        sample
        for sample in samples
        if datetime.fromisoformat(sample["utc"]) > returning
    ]
    cpu_seconds = time.process_time() - cpu_start
    result = dict(
        schema="n1-transition-factorial-run86-scheduled-runtime-preflight-v1",
        complete=(len(samples) == 6 and len(post_return) >= 3 and cpu_seconds <= 60),
        synthetic_only=True,
        model_loaded=False,
        training_material_loaded=False,
        optimizer_updates=0,
        curriculum_queries=0,
        dev_queries=0,
        holdout_queries=0,
        task_name=TASK_PROBE,
        pid=os.getpid(),
        launcher_returning_utc=returned_utc,
        worker_started_utc=json.loads(worker_path.read_text(encoding="utf-8"))["started_utc"],
        started_utc=started,
        completed_utc=ended,
        heartbeat_count=len(samples),
        heartbeats_after_launcher_return=len(post_return),
        heartbeat_sha256=sha256(PREFLIGHT_HEARTBEATS),
        registration_sha256=sha256(PREFLIGHT_LAUNCH_MANIFEST),
        controller_sha256=sha256(Path(__file__).resolve()),
        launcher_sha256=sha256(LAUNCHER),
        process_cpu_seconds=cpu_seconds,
        cpu_limit_seconds=PREFLIGHT_CPU_LIMIT_SECONDS,
    )
    write_new_json(PREFLIGHT_REPORT, result)
    return 0 if result["complete"] else 1


def execute_run86() -> dict:
    for path in (REPORT, ATTEMPT, COMPLETION, WORKER_MANIFEST):
        if path.exists():
            raise FileExistsError(f"run86 artifact already exists: {path}")
    if CHECKPOINTS.exists() or list(REPORT.parent.glob(f"{REPORT.stem}_s*.json.gz")):
        raise FileExistsError("run86 checkpoint or shard family already exists")
    if sha256(FROZEN_ATTEMPT) != FROZEN_ATTEMPT_SHA256:
        raise ValueError("frozen incomplete84 attempt bytes changed")

    launch, returned, worker_path = task_receipts(probe=False)
    if (
        launch.get("controller_sha256") != sha256(Path(__file__).resolve())
        or launch.get("runner_sha256") != sha256(RUNNER)
        or launch.get("launcher_sha256") != sha256(LAUNCHER)
    ):
        raise ValueError("run86 source differs from registered task")

    frozen = json.loads(FROZEN_ATTEMPT.read_text(encoding="utf-8"))
    module = runpy.run_path(str(RUNNER), run_name="_taiji_n1_run86_frozen84")
    if module["PROTOCOL"] != frozen["protocol"] or module["identity"]() != frozen["code_identity"]:
        raise ValueError("run86 protocol or frozen84 code identity changed")

    module["REPORT"] = REPORT
    module["ATTEMPT"] = ATTEMPT
    original_external_path = module["get_external_path"]

    def external_path(path: str) -> Path:
        return Path(original_external_path(path)) / "run86"

    module["get_external_path"] = external_path
    original_readiness = module["readiness"]

    def guarded_readiness():
        guards, sources = original_readiness()
        for path in (
            Path(__file__).resolve(),
            LAUNCHER,
            FROZEN_ATTEMPT,
            LAUNCH_MANIFEST,
            LAUNCH_RETURNED,
            worker_path,
        ):
            guards[str(path.resolve())] = sha256(path)
        return guards, sources

    module["readiness"] = guarded_readiness
    original_print = builtins.print

    def print_run86(*args, **kwargs):
        if args and isinstance(args[0], str):
            args = (args[0].replace("factorial84", "factorial86"), *args[1:])
        original_print(*args, **kwargs)

    module["print"] = print_run86
    module["torch"].set_num_threads(1)
    CHECKPOINTS.mkdir(exist_ok=False)
    result = module["audit"]()
    return dict(result=result, worker_path=worker_path, returned=returned)


def complete_run86(result: dict, started_utc: str, cpu_start: float) -> int:
    report = result["result"]
    exit_code = 0 if report.get("complete") else 1
    completion = dict(
        schema="n1-transition-factorial-run86-completion-v1",
        task_name=TASK_RUN,
        process_id=os.getpid(),
        started_utc=started_utc,
        completed_utc=utc_now(),
        process_cpu_seconds=time.process_time() - cpu_start,
        engineering_complete=bool(report.get("complete")),
        training_sufficient=bool(report.get("training_sufficient")),
        report_sha256=sha256(REPORT) if REPORT.exists() else None,
        exit_code=exit_code,
    )
    write_new_json(COMPLETION, completion)
    print(json.dumps(completion, ensure_ascii=False, sort_keys=True), flush=True)
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--synthetic-heartbeat", action="store_true")
    mode.add_argument("--run-handoff-probe", action="store_true")
    mode.add_argument("--execute-approved", action="store_true")
    args = parser.parse_args()
    configure_stdio(probe=args.synthetic_heartbeat, handoff=args.run_handoff_probe)
    try:
        if args.synthetic_heartbeat:
            return synthetic_heartbeat()
        if args.run_handoff_probe:
            return synthetic_run_handoff_probe()
        started_utc = utc_now()
        cpu_start = time.process_time()
        try:
            result = execute_run86()
        except BaseException as error:
            if not COMPLETION.exists():
                write_new_json(
                    COMPLETION,
                    dict(
                        schema="n1-transition-factorial-run86-completion-v1",
                        task_name=TASK_RUN,
                        process_id=os.getpid(),
                        started_utc=started_utc,
                        completed_utc=utc_now(),
                        process_cpu_seconds=time.process_time() - cpu_start,
                        engineering_complete=False,
                        exit_code=1,
                        error=f"{type(error).__name__}: {error}",
                    ),
                )
            raise
        return complete_run86(result, started_utc, cpu_start)
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
