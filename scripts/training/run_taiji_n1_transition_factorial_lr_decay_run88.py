"""PLAN-N1-88: fresh position training with a smaller final-cycle Adam rate."""

from __future__ import annotations

import argparse
import builtins
import hashlib
import json
import os
import runpy
import subprocess
import sys
import time
import traceback
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import _taiji_n1_transition_factorial_training_receipt as receipt

FROZEN_RUNNER = ROOT / "scripts/training/run_taiji_n1_transition_factorial_paired.py"
LAUNCHER = ROOT / "scripts/training/launch_taiji_n1_transition_factorial_lr_decay_run88b.ps1"
PLAN = ROOT / "plans/reference/PLAN-N1-88_position_final_cycle_lr_decay_review_20261016.md"
BASE_REPORT = ROOT / "reports/taiji_n1_transition_factorial_run86c.json"
AUDIT_REPORT = ROOT / "reports/taiji_n1_transition_factorial_run86c_offline_trajectory_audit.json"
PREFLIGHT = ROOT / "reports/taiji_n1_transition_factorial_lr_decay_run88b_preflight.json"
PREFLIGHT_ATTEMPT = ROOT / "reports/taiji_n1_transition_factorial_lr_decay_run88b_preflight_attempt.json"
PREFLIGHT_ATTEMPT_2 = ROOT / "reports/taiji_n1_transition_factorial_lr_decay_run88b_preflight_attempt_02.json"
PREFLIGHT_FAILURE = ROOT / "reports/taiji_n1_transition_factorial_lr_decay_run88b_preflight_failure.json"
ATTEMPT = ROOT / "reports/taiji_n1_transition_factorial_lr_decay_run88b_attempt.json"
REPORT = ROOT / "reports/taiji_n1_transition_factorial_lr_decay_run88b.json"
COMPLETION = ROOT / "reports/taiji_n1_transition_factorial_lr_decay_run88b_completion.json"
LAUNCH_MANIFEST = ROOT / "reports/taiji_n1_transition_factorial_lr_decay_run88b_launcher.json"
LAUNCH_RETURNED = ROOT / "reports/taiji_n1_transition_factorial_lr_decay_run88b_launcher_returned.json"
WORKER_MANIFEST = ROOT / "reports/taiji_n1_transition_factorial_lr_decay_run88b_worker.json"
STDOUT = ROOT / "reports/taiji_n1_transition_factorial_lr_decay_run88b.stdout.log"
STDERR = ROOT / "reports/taiji_n1_transition_factorial_lr_decay_run88b.stderr.log"
CHECKPOINT_ROOT = (
    ROOT / "data/checkpoints/run88b/n1_transition_factorial_lr_decay_20261016"
)

TASK_RUN = "Seed-N1-LRDecay-Run88B-20261016"
BASE_REPORT_SHA256 = "0a8fc62e9aa00577d67001c70008773e6e5cee54cebad4115956e0360446d785"
AUDIT_REPORT_SHA256 = "c5b21641de584d6e871ad4747ce8e181205e790a886c7b58bbb1305c9e46b2a0"
FROZEN_RUNNER_SHA256 = "38018dbe08b9eb4c35d0b5c97196727b34f94efee68d68113e80269b8d3c9491"
FRESH_ATTEMPT = ROOT / "reports/taiji_n1_transition_factorial_paired_train_20261016_attempt.json"
FRESH_ATTEMPT_SHA256 = "6a3124a1293c33dcbdc45bc55fb34971200236cb4673e52c9c981335e0bc6234"
PRIOR_RUN88_FAILURES = (
    ROOT / "reports/taiji_n1_transition_factorial_lr_decay_run88_preflight_attempt.json",
    ROOT / "reports/taiji_n1_transition_factorial_lr_decay_run88_preflight_failure.json",
    ROOT / "reports/taiji_n1_transition_factorial_lr_decay_run88_preflight_attempt_02.json",
    ROOT / "reports/taiji_n1_transition_factorial_lr_decay_run88_preflight_failure_02.json",
    ROOT / "data/checkpoints/run88/n1_transition_factorial_lr_decay_20261016/s11_position_zero.pt",
)

SEEDS = (11, 12, 13)
ARM = "position"
UPDATES = 24576
MILESTONES = (0, 6144, 12288, 18432, 24576)
DECAY_START = 18432
BASE_LR = 0.001
FINAL_LR = 0.00025
CPU_LIMIT_SECONDS = 4800
PRIOR_CPU_RESERVE_SECONDS = 300
ACTIVE_RATE_LOG: dict[int, float] | None = None
ACTIVE_BASELINE: dict[tuple[int, str], dict] | None = None


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path: Path, value: dict) -> None:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def learning_rate_for(completed_updates: int) -> float:
    if not 0 <= completed_updates <= UPDATES + 2:
        raise ValueError("update count outside frozen run88 schedule")
    return BASE_LR if completed_updates < DECAY_START else FINAL_LR


def apply_learning_rate(optimizer, completed_updates: int) -> float:
    lr = learning_rate_for(completed_updates)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def verify_learning_rate_groups() -> bool:
    cases = (
        (0, BASE_LR),
        (18431, BASE_LR),
        (18432, FINAL_LR),
        (24576, FINAL_LR),
    )
    for completed, expected in cases:
        optimizer = type("OptimizerProbe", (), {})()
        optimizer.param_groups = [{"lr": -1.0}]
        if apply_learning_rate(optimizer, completed) != expected:
            return False
        if optimizer.param_groups[0]["lr"] != expected:
            return False
    return True


def run_protocol(base_protocol: dict) -> dict:
    protocol = deepcopy(base_protocol)
    protocol.update(
        schema="n1-transition-factorial-position-final-cycle-lr-decay-v1",
        seeds=list(SEEDS),
        arms=[ARM],
        total_updates=len(SEEDS) * UPDATES,
        total_qa_sequences=len(SEEDS) * 55296,
        total_training_sessions=len(SEEDS) * 43008,
        known_score_rows=len(SEEDS) * 4800,
        independent_restores=len(SEEDS) * 2,
        signature_clone_updates=len(SEEDS) * 8,
        synthetic_qa_sequences=len(SEEDS) * 8 * 4,
        cpu_limit_seconds=CPU_LIMIT_SECONDS,
        prior_cpu_reserve_seconds=PRIOR_CPU_RESERVE_SECONDS,
        learning_rate_schedule={
            "schema": "n1-final-cycle-step-decay-v1",
            "base_lr": BASE_LR,
            "final_cycle_start": DECAY_START,
            "final_cycle_lr": FINAL_LR,
            "adam_moments": "continuous_unmodified",
        },
        paired_control_report_sha256=BASE_REPORT_SHA256,
        offline_diagnostic_sha256=AUDIT_REPORT_SHA256,
        dev_queries=0,
        holdout_queries=0,
        confirmation_queries=0,
    )
    return protocol


def code_identity(protocol: dict) -> dict:
    if any(not path.is_file() for path in PRIOR_RUN88_FAILURES):
        raise FileNotFoundError("preserved run88 preflight history is incomplete")
    return {
        "schema": "n1-transition-factorial-lr-decay-run88b-code-identity-v1",
        "protocol": protocol,
        "controller_sha256": sha256(Path(__file__).resolve()),
        "launcher_sha256": sha256(LAUNCHER),
        "frozen_runner_sha256": sha256(FROZEN_RUNNER),
        "plan_sha256": sha256(PLAN),
        "control_report_sha256": sha256(BASE_REPORT),
        "offline_audit_sha256": sha256(AUDIT_REPORT),
        "prior_run88_failures": {str(path): sha256(path) for path in PRIOR_RUN88_FAILURES},
    }


def make_module() -> tuple[dict, dict, dict]:
    if sha256(FROZEN_RUNNER) != FROZEN_RUNNER_SHA256:
        raise ValueError("frozen run84 training runner hash changed")
    if sha256(BASE_REPORT) != BASE_REPORT_SHA256:
        raise ValueError("sealed run86c control report hash changed")
    if sha256(AUDIT_REPORT) != AUDIT_REPORT_SHA256:
        raise ValueError("sealed PLAN-N1-87 audit report hash changed")
    if sha256(FRESH_ATTEMPT) != FRESH_ATTEMPT_SHA256:
        raise ValueError("frozen run84 source attempt hash changed")

    loaded = runpy.run_path(str(FROZEN_RUNNER), run_name="_taiji_n1_run88b_frozen84")
    loaded["torch"].set_num_threads(1)
    audit_function = loaded.get("audit")
    globals_dict = getattr(audit_function, "__globals__", None)
    if not isinstance(globals_dict, dict):
        raise TypeError("frozen runner audit function has no globals")
    for name in ("identity", "readiness", "known_views", "train_one", "save", "restore"):
        function = loaded.get(name)
        if getattr(function, "__globals__", globals_dict) is not globals_dict:
            raise RuntimeError(f"frozen function {name} has a separate globals namespace")

    base_protocol = deepcopy(globals_dict["PROTOCOL"])
    protocol = run_protocol(base_protocol)
    original_external_path = globals_dict["get_external_path"]

    def external_path(relative_path: str) -> Path:
        return Path(original_external_path(relative_path)) / "run88b"

    def identity() -> dict:
        return code_identity(protocol)

    update_module = globals_dict["update"]
    original_train_step = update_module.train_step

    def scheduled_train_step(model, optimizer, *args, **kwargs):
        nonlocal original_train_step
        completed = max(
            (int(state["step"]) for state in optimizer.state.values() if "step" in state),
            default=0,
        )
        lr = apply_learning_rate(optimizer, completed)
        if ACTIVE_RATE_LOG is not None and kwargs.get("audit") is not True:
            if completed in ACTIVE_RATE_LOG:
                raise ValueError(f"duplicate run88b learning-rate step {completed}")
            ACTIVE_RATE_LOG[completed] = lr
        return original_train_step(model, optimizer, *args, **kwargs)

    update_module.train_step = scheduled_train_step
    globals_dict.update(
        {
            "REPORT": REPORT,
            "ATTEMPT": ATTEMPT,
            "PROTOCOL": protocol,
            "UPDATES": UPDATES,
            "MILESTONES": MILESTONES,
            "identity": identity,
            "get_external_path": external_path,
        }
    )
    original_save = globals_dict["save"]
    preflight_checkpoints: dict[str, dict] = {}

    def save_with_preflight(model, optimizer, path, completed, seed, arm):
        path = Path(path)
        if path.exists():
            if completed != 0 or path.name not in preflight_checkpoints:
                raise FileExistsError(f"unexpected existing run88b checkpoint: {path}")
            payload = globals_dict["torch"].load(path, map_location="cpu", weights_only=True)
            if (
                payload.get("protocol") != protocol
                or payload.get("seed") != seed
                or payload.get("arm") != arm
                or payload.get("phase") != "zero"
                or payload.get("completed_updates") != 0
                or globals_dict["paired"].fingerprint(payload["candidate"])
                != globals_dict["paired"].fingerprint(model.bundle(optimizer))
                or sha256(path) != preflight_checkpoints[path.name]["sha256"]
            ):
                raise ValueError("formal training zero checkpoint differs from its preflight")
            return payload
        return original_save(model, optimizer, path, completed, seed, arm)

    globals_dict["save"] = save_with_preflight
    original_seal = globals_dict["seal_record"]

    def seal_with_paired_control(record):
        if ACTIVE_BASELINE is None or ACTIVE_RATE_LOG is None:
            raise RuntimeError("run88b paired baseline and rate log must be active during training")
        seed = record["seed"]
        baseline = ACTIVE_BASELINE[seed, ARM]
        observed_schedule = {
            "base_updates": sum(rate == BASE_LR for rate in ACTIVE_RATE_LOG.values()),
            "final_cycle_updates": sum(rate == FINAL_LR for rate in ACTIVE_RATE_LOG.values()),
            "base_lr": BASE_LR,
            "final_cycle_lr": FINAL_LR,
        }
        record["learning_rate_schedule"] = observed_schedule
        problems = []
        expected_rates = {
            **{step: BASE_LR for step in range(DECAY_START)},
            **{step: FINAL_LR for step in range(DECAY_START, UPDATES)},
        }
        if ACTIVE_RATE_LOG != expected_rates:
            problems.append(f"actual learning-rate steps differ for seed {seed}")
        if (
            record["losses"][:DECAY_START] != baseline["losses"][:DECAY_START]
            or record["measurements"][:DECAY_START]
            != baseline["measurements"][:DECAY_START]
        ):
            problems.append(f"paired trajectory differs before step {DECAY_START}")
        current_rows = [row for row in record["rows"] if row["completed_updates"] < DECAY_START]
        baseline_rows = [row for row in baseline["rows"] if row["completed_updates"] < DECAY_START]
        if current_rows != baseline_rows:
            problems.append(f"frozen scores differ before step {DECAY_START}")
        if (
            record["restore_checks"][0]["signature"]["bundle"]
            != baseline["restore_checks"][0]["signature"]["bundle"]
        ):
            problems.append(f"starting bundle differs from run86c seed {seed}")
        summary = original_seal(record)
        summary["learning_rate_schedule"] = observed_schedule
        if problems:
            raise ValueError("; ".join(problems))
        return summary

    globals_dict["seal_record"] = seal_with_paired_control
    def independent(path, expected, budget):
        budget.guard(reserve=10)
        python = Path(sys.executable)
        if python.name.lower() == "pythonw.exe":
            python = python.with_name("python.exe")
        try:
            output = subprocess.run(
                [str(python), str(Path(__file__).resolve()), "--restore-worker", str(path)],
                check=True,
                capture_output=True,
                text=True,
                timeout=min(45, budget.limit - budget.used() - 2),
                env=dict(
                    os.environ,
                    OMP_NUM_THREADS="1",
                    MKL_NUM_THREADS="1",
                    OPENBLAS_NUM_THREADS="1",
                ),
            )
            restored = json.loads(output.stdout)
        except Exception:
            budget.accounting_complete = False
            raise
        budget.children += restored["cpu_seconds"]
        budget.guard()
        if restored["signature"] != expected or restored["signature_clone_updates"] != 2:
            raise ValueError("independent run88b checkpoint restore signature differs")
        return dict(path=str(path), sha256=sha256(Path(path)), equal=True, **restored)

    globals_dict["independent"] = independent
    original_print = builtins.print

    def print_run88(*args, **kwargs):
        if args and isinstance(args[0], str):
            args = (args[0].replace("factorial84", "factorial88_lr"), *args[1:])
        original_print(*args, **kwargs)

    globals_dict["print"] = print_run88
    return loaded, globals_dict, preflight_checkpoints


def source_guards(globals_dict: dict, readiness_guards: dict, extra: list[Path]) -> dict:
    guards = dict(readiness_guards)
    for path in extra:
        absolute = Path(path).resolve()
        guards[str(absolute)] = sha256(absolute)
    for raw_path, expected in guards.items():
        path = Path(raw_path)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"run88b guarded input changed: {raw_path}")
    if globals_dict["identity"]()["controller_sha256"] != sha256(Path(__file__).resolve()):
        raise ValueError("run88b controller identity changed")
    return guards


def control_records() -> tuple[dict, dict[tuple[int, str], dict]]:
    report = read_json(BASE_REPORT)
    if not report.get("complete") or report.get("training_sufficient") is not False:
        raise ValueError("run86c complete but insufficient control batch required")
    by_key = {(entry["seed"], entry["arm"]): entry for entry in report["models"]}
    selected = {}
    for seed in SEEDS:
        model = by_key.get((seed, ARM))
        if model is None:
            raise ValueError("position baseline missing for a frozen seed")
        record = receipt.read_model(model)
        if not record.get("complete") or record.get("arm") != ARM:
            raise ValueError("run86c position baseline shard failed its seal")
        selected[seed, ARM] = record
    return report, selected


def validate_preflight(report: dict, binding: dict, identity: dict) -> tuple[float, dict]:
    preflight_sha = sha256(PREFLIGHT)
    if (
        report.get("complete") is not True
        or report.get("cpu_limit_seconds") != CPU_LIMIT_SECONDS
        or report.get("code_identity") != identity
        or binding.get("preflight_report_sha256") != preflight_sha
        or binding.get("code_identity") != identity
        or len(report.get("zero_checkpoints", [])) != len(SEEDS)
        or any(not entry.get("restore_equal") for entry in report["zero_checkpoints"])
    ):
        raise ValueError("complete PLAN-N1-88 zero checkpoint preflight and binding required")
    preflight_cpu = float(report["cpu_seconds"])
    if not 0 <= preflight_cpu + PRIOR_CPU_RESERVE_SECONDS < CPU_LIMIT_SECONDS:
        raise ValueError("run88b preflight exhausted the complete CPU budget")
    known = {entry["path_name"]: entry for entry in report["zero_checkpoints"]}
    if set(known) != {f"s{seed}_{ARM}_zero.pt" for seed in SEEDS}:
        raise ValueError("one preflight zero checkpoint for each seed required")
    for name, entry in known.items():
        path = CHECKPOINT_ROOT / name
        if not path.is_file() or sha256(path) != entry["sha256"]:
            raise ValueError(f"run88b preflight checkpoint changed: {name}")
    return preflight_cpu, known


def checkpoint_preflight() -> dict:
    started = time.process_time()
    if PREFLIGHT.exists():
        raise FileExistsError("run88b checkpoint preflight report already exists")
    if PREFLIGHT_ATTEMPT.exists():
        if (
            not PREFLIGHT_FAILURE.is_file()
            or not CHECKPOINT_ROOT.is_dir()
            or any(CHECKPOINT_ROOT.iterdir())
            or PREFLIGHT_ATTEMPT_2.exists()
        ):
            raise FileExistsError("run88b preflight retry requires the recorded empty-root first failure")
        attempt_path = PREFLIGHT_ATTEMPT_2
        attempt_number = 2
    else:
        if CHECKPOINT_ROOT.exists() or PREFLIGHT_ATTEMPT_2.exists():
            raise FileExistsError("run88b checkpoint preflight path already exists")
        attempt_path = PREFLIGHT_ATTEMPT
        attempt_number = 1
    _, g, checkpoint_records = make_module()
    guards, sources = g["readiness"]()
    baseline_report, baseline_by_seed = control_records()
    extra = [
        Path(__file__).resolve(),
        LAUNCHER,
        FROZEN_RUNNER,
        BASE_REPORT,
        AUDIT_REPORT,
        *PRIOR_RUN88_FAILURES,
        *(Path(entry["shard"]["path"]) for entry in baseline_report["models"] if entry["arm"] == ARM),
    ]
    if attempt_number == 2:
        extra.extend((PREFLIGHT_ATTEMPT, PREFLIGHT_FAILURE))
    guards = source_guards(g, guards, extra)
    identity = g["identity"]()
    protocol = g["PROTOCOL"]
    write_new(
        attempt_path,
        {
            "schema": "n1-transition-factorial-lr-decay-run88b-preflight-attempt-v1",
            "status": "started",
            "attempt_number": attempt_number,
            "started_utc": utc_now(),
            "protocol": protocol,
            "code_identity": identity,
            "source_guards": guards,
            "training_updates": 0,
            "dev_queries": 0,
            "holdout_queries": 0,
            "confirmation_queries": 0,
        },
    )
    if attempt_number == 1:
        CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=False)
    budget = g["paired"].CpuBudget(CPU_LIMIT_SECONDS - PRIOR_CPU_RESERVE_SECONDS)
    zero_checkpoints = []
    for seed in SEEDS:
        source = g["torch"].load(sources[seed, ARM], map_location="cpu", weights_only=True)
        model, optimizer = g["engineering"].load(source)
        g["validate_state"](model, optimizer, 0, seed, ARM)
        zero_sha = g["paired"].fingerprint(model.bundle(optimizer))
        baseline_signature = baseline_by_seed[seed, ARM]["restore_checks"][0]["signature"]["bundle"]
        if zero_sha != baseline_signature:
            raise ValueError(f"fresh zero bundle differs from run86c seed {seed}")
        path = CHECKPOINT_ROOT / f"s{seed}_{ARM}_zero.pt"
        saved = g["save"](model, optimizer, path, 0, seed, ARM)
        restore = g["independent"](path, saved["expected_signature"], budget)
        if not restore["equal"]:
            raise ValueError(f"run88b zero checkpoint recovery failed for seed {seed}")
        entry = {
            "seed": seed,
            "arm": ARM,
            "path_name": path.name,
            "sha256": sha256(path),
            "bundle_sha256": zero_sha,
            "matches_run86c_position_zero": True,
            "restore_equal": restore["equal"],
            "restore_cpu_seconds": restore["cpu_seconds"],
            "signature_clone_updates": restore["signature_clone_updates"] + 2,
        }
        zero_checkpoints.append(entry)
        checkpoint_records[path.name] = entry
        budget.guard()

    if any(sha256(Path(path)) != digest for path, digest in guards.items()):
        raise ValueError("run88b preflight source guards changed")
    identity_after = g["identity"]()
    if identity_after != identity:
        raise ValueError("run88b preflight source identity changed")
    cpu_seconds = time.process_time() - started + budget.children
    report = {
        "schema": "n1-transition-factorial-lr-decay-run88b-checkpoint-preflight-v1",
        "complete": True,
        "attempt_number": attempt_number,
        "protocol": protocol,
        "code_identity": identity,
        "source_guards": guards,
        "source_report_sha256": BASE_REPORT_SHA256,
        "offline_audit_sha256": AUDIT_REPORT_SHA256,
        "zero_checkpoints": zero_checkpoints,
        "checkpoint_loads": len(SEEDS) * 2,
        "model_loads": len(SEEDS),
        "training_updates": 0,
        "signature_clone_updates": len(SEEDS) * 4,
        "synthetic_qa_sequences": len(SEEDS) * 4 * 4,
        "learning_rate_contract": {
            "update_0": learning_rate_for(0),
            "update_18431": learning_rate_for(18431),
            "update_18432": learning_rate_for(18432),
            "update_24576": learning_rate_for(24576),
            "optimizer_group_rates_verified": verify_learning_rate_groups(),
        },
        "cpu_seconds": round(cpu_seconds, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "prior_cpu_reserve_seconds": PRIOR_CPU_RESERVE_SECONDS,
        "dev_queries": 0,
        "holdout_queries": 0,
        "confirmation_queries": 0,
        "can_promote": False,
        "completed_utc": utc_now(),
    }
    if not report["learning_rate_contract"]["optimizer_group_rates_verified"] or cpu_seconds + PRIOR_CPU_RESERVE_SECONDS >= CPU_LIMIT_SECONDS:
        raise ValueError("run88b checkpoint preflight or schedule contract did not pass")
    write_new(PREFLIGHT, report)
    binding = {
        "schema": "n1-transition-factorial-lr-decay-run88b-preflight-binding-v1",
        "preflight_report_sha256": sha256(PREFLIGHT),
        "code_identity": identity,
        "protocol_sha256": hashlib.sha256(
            json.dumps(protocol, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    binding_path = PREFLIGHT.with_name(PREFLIGHT.stem + "_binding.json")
    write_new(binding_path, binding)
    # The report is immutable once written; the binding contains the report's raw SHA.
    print(json.dumps({"complete": True, "report": str(PREFLIGHT), "cpu_seconds": report["cpu_seconds"], "checkpoints": 3}))
    return report


def task_receipts(identity: dict) -> tuple[dict, Path]:
    deadline = time.monotonic() + 60
    while (not LAUNCH_MANIFEST.exists() or not LAUNCH_RETURNED.exists()) and time.monotonic() < deadline:
        time.sleep(0.05)
    if not LAUNCH_MANIFEST.exists() or not LAUNCH_RETURNED.exists():
        raise FileNotFoundError("run88b task manifest or launcher return is missing")
    launch = read_json(LAUNCH_MANIFEST)
    returned = read_json(LAUNCH_RETURNED)
    if (
        launch.get("mode") != "run"
        or launch.get("task_name") != TASK_RUN
        or launch.get("trigger_count") != 0
        or launch.get("principal_logon_type") != "Interactive"
        or launch.get("run_level") != "Limited"
        or returned.get("task_name") != TASK_RUN
        or returned.get("start_succeeded") is not True
        or launch.get("controller_sha256") != identity["controller_sha256"]
        or launch.get("launcher_sha256") != identity["launcher_sha256"]
        or launch.get("frozen_runner_sha256") != identity["frozen_runner_sha256"]
        or launch.get("preflight_report_sha256") != sha256(PREFLIGHT)
        or launch.get("preflight_binding_sha256")
        != sha256(PREFLIGHT.with_name(PREFLIGHT.stem + "_binding.json"))
    ):
        raise ValueError("run88b task receipts do not match frozen launch identity")
    if WORKER_MANIFEST.exists():
        raise FileExistsError("run88 worker manifest already exists")
    worker = {
        "schema": "n1-transition-factorial-lr-decay-run88b-worker-v1",
        "task_name": TASK_RUN,
        "process_id": os.getpid(),
        "started_utc": utc_now(),
        "code_identity": identity,
        "launcher_manifest_sha256": sha256(LAUNCH_MANIFEST),
        "launcher_returned_sha256": sha256(LAUNCH_RETURNED),
    }
    write_new(WORKER_MANIFEST, worker)
    return launch, WORKER_MANIFEST


def training_audit() -> dict:
    setup_started = time.process_time()
    if any(path.exists() for path in (REPORT, ATTEMPT, COMPLETION)):
        raise FileExistsError("run88b formal batch is exclusive and cannot restart")
    _, g, checkpoint_records = make_module()
    identity = g["identity"]()
    preflight = read_json(PREFLIGHT)
    binding_path = PREFLIGHT.with_name(PREFLIGHT.stem + "_binding.json")
    binding = read_json(binding_path)
    preflight_cpu, known_checkpoints = validate_preflight(preflight, binding, identity)
    checkpoint_records.update(known_checkpoints)
    setup_cpu = time.process_time() - setup_started
    remaining_cpu = CPU_LIMIT_SECONDS - PRIOR_CPU_RESERVE_SECONDS - preflight_cpu - setup_cpu
    if remaining_cpu <= 0:
        raise ValueError("run88b setup and checkpoint preflight exhausted the CPU budget")
    budget = g["paired"].CpuBudget(remaining_cpu)
    launch, worker_path = task_receipts(identity)

    readiness_guards, sources = g["readiness"]()
    baseline_report, baseline_by_seed = control_records()
    global ACTIVE_BASELINE
    ACTIVE_BASELINE = baseline_by_seed
    extra = [
        Path(__file__).resolve(),
        LAUNCHER,
        FROZEN_RUNNER,
        BASE_REPORT,
        AUDIT_REPORT,
        *PRIOR_RUN88_FAILURES,
        PREFLIGHT,
        PREFLIGHT_ATTEMPT,
        PREFLIGHT_ATTEMPT_2,
        PREFLIGHT_FAILURE,
        binding_path,
        LAUNCH_MANIFEST,
        LAUNCH_RETURNED,
        worker_path,
        *(Path(entry["shard"]["path"]) for entry in baseline_report["models"] if entry["arm"] == ARM),
        *(CHECKPOINT_ROOT / name for name in known_checkpoints),
    ]
    guards = source_guards(g, readiness_guards, extra)
    items, curriculum, views = g["known_views"]()
    if len(items) != 192 or any(item.get("split") != "train" for item in items):
        raise ValueError("run88b requires the unchanged complete train-only source")
    budget.guard()
    result = {
        "protocol": g["PROTOCOL"],
        "code_identity": identity,
        "source_guards": guards,
        "preflight_report_sha256": sha256(PREFLIGHT),
        "models": [],
        "dev_queries": 0,
        "holdout_queries": 0,
        "confirmation_queries": 0,
        "can_promote": False,
        "status": "running",
    }
    write_new(
        ATTEMPT,
        {
            "schema": "n1-transition-factorial-lr-decay-run88b-attempt-v1",
            "protocol": result["protocol"],
            "code_identity": identity,
            "source_guards": guards,
            "preflight_report_sha256": sha256(PREFLIGHT),
            "started_utc": utc_now(),
        },
    )
    error = None
    try:
        for seed in SEEDS:
            global ACTIVE_RATE_LOG
            ACTIVE_RATE_LOG = {}
            record = g["train_one"](
                seed,
                ARM,
                sources[seed, ARM],
                CHECKPOINT_ROOT,
                items,
                curriculum,
                views,
                budget,
            )
            schedule = record.get("learning_rate_schedule", {})
            if schedule != {
                "base_updates": DECAY_START,
                "final_cycle_updates": UPDATES - DECAY_START,
                "base_lr": BASE_LR,
                "final_cycle_lr": FINAL_LR,
            }:
                raise ValueError(f"run88b learning-rate schedule evidence differs for seed {seed}")
            ACTIVE_RATE_LOG = None
            if not record["complete"]:
                raise ValueError(f"run88b seed {seed} training record incomplete")
            result["models"].append(record)
            budget.guard(reserve=15)

        old_protocol = receipt.train.PROTOCOL
        receipt.train.PROTOCOL = g["PROTOCOL"]
        try:
            for summary in result["models"]:
                shard_record = receipt.read_model(summary)
                receipt.validate_model(summary, shard_record, views=views)
        finally:
            receipt.train.PROTOCOL = old_protocol
        result["status"] = "complete"
    except BaseException as exc:
        ACTIVE_RATE_LOG = None
        ACTIVE_BASELINE = None
        error = f"{type(exc).__name__}: {exc}"
        result.update(status="stopped", error=error)
    ACTIVE_BASELINE = None

    result.update(
        preflight_cpu_seconds=preflight_cpu,
        setup_cpu_seconds=round(setup_cpu, 6),
        training_cpu_seconds=budget.used(),
        prior_cpu_reserve_seconds=PRIOR_CPU_RESERVE_SECONDS,
        cpu_seconds=round(PRIOR_CPU_RESERVE_SECONDS + preflight_cpu + setup_cpu + budget.used(), 6),
        cpu_accounting_complete=budget.accounting_complete,
        sources_unchanged=all(sha256(Path(path)) == digest for path, digest in guards.items()),
        code_unchanged=g["identity"]() == identity,
        completed_updates=sum(model.get("completed_updates", 0) for model in result["models"]),
        qa_sequences=sum(model.get("qa_sequences", 0) for model in result["models"]),
        new_session_count=sum(model.get("new_session_count", 0) for model in result["models"]),
        score_queries=sum(model.get("score_queries", 0) for model in result["models"]),
        signature_clone_updates=sum(
            model.get("signature_clone_updates", 0) for model in result["models"]
        ),
        synthetic_qa_sequences=4
        * sum(model.get("signature_clone_updates", 0) for model in result["models"]),
        training_sufficient=(
            result["status"] == "complete"
            and len(result["models"]) == len(SEEDS)
            and all(model["training_sufficient"] for model in result["models"])
        ),
    )
    result["complete"] = (
        result["status"] == "complete"
        and len(result["models"]) == len(SEEDS)
        and result["completed_updates"] == len(SEEDS) * UPDATES
        and result["qa_sequences"] == len(SEEDS) * 55296
        and result["new_session_count"] == len(SEEDS) * 43008
        and result["score_queries"] == len(SEEDS) * 4800
        and result["signature_clone_updates"] == len(SEEDS) * 8
        and result["synthetic_qa_sequences"] == len(SEEDS) * 32
        and result["cpu_seconds"] <= CPU_LIMIT_SECONDS
        and result["sources_unchanged"]
        and result["code_unchanged"]
        and result["cpu_accounting_complete"]
        and not any(result[key] for key in ("dev_queries", "holdout_queries", "confirmation_queries"))
    )
    result["can_promote"] = False
    write_new(REPORT, result)
    if error:
        raise RuntimeError(error)
    return result


def configure_stdio() -> None:
    for path in (STDOUT, STDERR):
        if path.exists():
            raise FileExistsError(f"run88b log already exists: {path}")
    sys.stdout = STDOUT.open("x", encoding="utf-8", buffering=1)
    sys.stderr = STDERR.open("x", encoding="utf-8", buffering=1)


def complete(exit_code: int, started: str, cpu_start: float, report_sha: str | None, error: str | None = None) -> int:
    receipt_value = {
        "schema": "n1-transition-factorial-lr-decay-run88b-completion-v1",
        "task_name": TASK_RUN,
        "process_id": os.getpid(),
        "started_utc": started,
        "completed_utc": utc_now(),
        "process_cpu_seconds": time.process_time() - cpu_start,
        "report_sha256": report_sha,
        "exit_code": exit_code,
        "error": error,
    }
    write_new(COMPLETION, receipt_value)
    print(json.dumps(receipt_value, ensure_ascii=False, sort_keys=True), flush=True)
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--checkpoint-preflight", action="store_true")
    mode.add_argument("--execute-approved", action="store_true")
    mode.add_argument("--restore-worker", type=Path)
    args = parser.parse_args()
    if args.restore_worker:
        _, g, _ = make_module()
        print(json.dumps(g["restore"](args.restore_worker), ensure_ascii=False, sort_keys=True))
        return 0
    if args.checkpoint_preflight:
        value = checkpoint_preflight()
        print(json.dumps({"complete": value["complete"], "report": str(PREFLIGHT), "cpu_seconds": value["cpu_seconds"]}))
        return 0

    configure_stdio()
    started = utc_now()
    cpu_start = time.process_time()
    try:
        report = training_audit()
        return complete(0 if report["complete"] else 1, started, cpu_start, sha256(REPORT))
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        traceback.print_exc(file=sys.stderr)
        if not COMPLETION.exists():
            return complete(1, started, cpu_start, sha256(REPORT) if REPORT.exists() else None, failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
