"""PLAN-N1-100 nine-model hard-neighbor training controller."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import subprocess
import sys
import time
import types
from collections import Counter
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import (
    _taiji_n1_transition_factorial_training_receipt as training_receipt,
)
from scripts.training import (  # noqa: E402
    build_taiji_n1_hard_neighbor_training_schedule as schedule_builder,
)
from scripts.training import (
    run_taiji_n1_transition_factorial_paired as frozen_runner,
)
from seed_platform.paths import get_external_path  # noqa: E402

FROZEN_RUNNER = ROOT / "scripts/training/run_taiji_n1_transition_factorial_paired.py"
PLAN = ROOT / "plans/reference/PLAN-N1-100_balanced_hard_neighbor_training_and_v7_20261017.md"
SCHEDULE = schedule_builder.OUT
REPORT = ROOT / "reports/taiji_n1_hard_neighbor_training_20261017.json"
ATTEMPT = REPORT.with_name(REPORT.stem + "_attempt.json")
PROGRESS = REPORT.with_name(REPORT.stem + "_progress.jsonl")
PREFLIGHT = REPORT.with_name(REPORT.stem + "_checkpoint_preflight.json")
PREFLIGHT_ATTEMPT = PREFLIGHT.with_name(PREFLIGHT.stem + "_attempt.json")
PREFLIGHT_BINDING = PREFLIGHT.with_name(PREFLIGHT.stem + "_binding.json")
CHECKPOINT_ROOT = (
    Path(get_external_path("data/checkpoints")) / "n1_hard_neighbor_training_v1_20261017"
)
SEEDS = (11, 12, 13)
ARMS = ("position", "terminal", "cut")
UPDATES = 24576
MILESTONES = (0, 6144, 12288, 18432, 24576)
DECAY_START = 18432
BASE_LR = 0.001
FINAL_LR = 0.00025
CPU_LIMIT_SECONDS = 18000
PREFLIGHT_CPU_LIMIT_SECONDS = 300
FROZEN_RUNNER_SHA256 = "38018dbe08b9eb4c35d0b5c97196727b34f94efee68d68113e80269b8d3c9491"
SCHEDULE_SHA256 = "dc3a96ad17d43bd898e403ecd119c12c781317a71ab06ba39cdab6db24c40d50"
BUILDER_SHA256 = "efa81a0bbc87f84db11e46cc40bc33e986fcfb6185c12b0404e9fccea4489d79"
RUN88_ATTEMPT_SHA256 = "6a3124a1293c33dcbdc45bc55fb34971200236cb4673e52c9c981335e0bc6234"
MANIFEST_SHA256 = schedule_builder.MANIFEST_SHA256
RENAMING_SHA256 = schedule_builder.RENAMING_SHA256
PAIRS_SHA256 = schedule_builder.PAIRS_SHA256
HARD_SCHEDULE: list[dict] = []
ACTIVE_ITEMS: list[dict] | None = None
ACTIVE_HARD_COUNTS: Counter | None = None
ACTIVE_CYCLE_COUNTS: Counter | None = None
ACTIVE_RATE_LOG: dict[int, float] | None = None
PREFLIGHT_CHECKPOINTS: dict[str, dict] = {}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path: Path, value: dict) -> None:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def learning_rate(completed_updates: int) -> float:
    if type(completed_updates) is not int or not 0 <= completed_updates <= UPDATES + 2:
        raise ValueError("update count outside PLAN-N1-100 trajectory")
    return BASE_LR if completed_updates < DECAY_START else FINAL_LR


def verify_learning_rate_schedule() -> bool:
    return tuple(learning_rate(step) for step in (0, 18431, 18432, 24575, 24576)) == (
        BASE_LR,
        BASE_LR,
        FINAL_LR,
        FINAL_LR,
        FINAL_LR,
    )


def protocol(base_protocol: dict | None = None) -> dict:
    base_protocol = deepcopy(frozen_runner.PROTOCOL if base_protocol is None else base_protocol)
    base_protocol.update(
        schema="n1-hard-neighbor-fresh-paired-training-v1",
        seeds=list(SEEDS),
        arms=list(ARMS),
        updates_per_model=UPDATES,
        total_updates=len(SEEDS) * len(ARMS) * UPDATES,
        total_qa_sequences=len(SEEDS) * len(ARMS) * 55296,
        total_training_sessions=len(SEEDS) * len(ARMS) * 43008,
        known_score_rows=len(SEEDS) * len(ARMS) * 4800,
        independent_restores=len(SEEDS) * len(ARMS) * 2,
        signature_clone_updates=len(SEEDS) * len(ARMS) * 8,
        synthetic_qa_sequences=len(SEEDS) * len(ARMS) * 32,
        cpu_limit_seconds=CPU_LIMIT_SECONDS,
        hard_schedule_sha256=SCHEDULE_SHA256,
        hard_schedule_seed=schedule_builder.SEED,
        hard_views_per_model=12288,
        raw_views_per_model=12288,
        schedule_entries_per_cycle=3072,
        learning_rate_schedule={
            "schema": "n1-final-cycle-step-decay-v1",
            "base_lr": BASE_LR,
            "final_cycle_start": DECAY_START,
            "final_cycle_lr": FINAL_LR,
            "adam_moments": "continuous_unmodified",
        },
        initialization="run83-full-fresh-nine-zero-source-v1",
        dev_queries=0,
        holdout_queries=0,
        confirmation_queries=0,
    )
    return base_protocol


def _custom_identity(base_identity: dict, protocol_value: dict) -> dict:
    return {
        "schema": "n1-hard-neighbor-training-code-identity-v1",
        "protocol": protocol_value,
        "base_runner_identity": base_identity,
        "controller_sha256": sha256(Path(__file__).resolve()),
        "frozen_runner_sha256": sha256(FROZEN_RUNNER),
        "schedule_sha256": sha256(SCHEDULE),
        "schedule_builder_sha256": sha256(Path(schedule_builder.__file__).resolve()),
        "plan_path": PLAN.relative_to(ROOT).as_posix(),
    }


def _verify_schedule_artifact() -> dict:
    if sha256(SCHEDULE) != SCHEDULE_SHA256:
        raise ValueError("frozen PLAN-N1-100 schedule digest changed")
    payload = read_json(SCHEDULE)
    if (
        payload.get("schema") != "n1-hard-neighbor-training-schedule-v1"
        or payload.get("source_manifest_sha256") != MANIFEST_SHA256
        or payload.get("source_renaming_sha256") != RENAMING_SHA256
        or payload.get("source_pair_manifest_sha256") != PAIRS_SHA256
        or payload.get("builder_sha256") != BUILDER_SHA256
        or len(payload.get("items", [])) != 3072
        or payload.get("summary", {}).get("parameter_updates") != 0
    ):
        raise ValueError("complete frozen train-only hard schedule required")
    return payload


def load_training_inputs() -> tuple[list[dict], dict, dict, dict]:
    items, manifest_sha = schedule_builder.train_items_from_frozen_manifest()
    if manifest_sha != MANIFEST_SHA256:
        raise ValueError("train-only manifest identity changed")
    items, curriculum, pairs = schedule_builder.load_training_sources(items)
    payload = _verify_schedule_artifact()
    regenerated = schedule_builder.build_schedule(items, curriculum)
    if regenerated["items"] != payload["items"] or regenerated["summary"] != payload["summary"]:
        raise ValueError("schedule entries no longer match the frozen train-only recipe")
    probes = [
        schedule_builder.core.renaming.rename_item(item, entry["names"])
        for item, entry in zip(items, curriculum["probes"], strict=True)
    ]
    views = {
        "original": items,
        "pair": pairs,
        "rename": probes,
        "rename_pair": schedule_builder.core.build_pairs(probes),
    }
    return items, curriculum, views, payload


def source_paths() -> list[Path]:
    return [
        FROZEN_RUNNER,
        PLAN,
        SCHEDULE,
        Path(schedule_builder.__file__).resolve(),
        schedule_builder.MANIFEST,
        schedule_builder.RENAMING,
        schedule_builder.PAIRS,
        ROOT / "reports/taiji_n1_transition_factorial_paired_train_20261016_attempt.json",
        ROOT / "scripts/training/run_taiji_n1_transition_factorial_calibration.py",
        ROOT / "scripts/training/audit_taiji_n1_transition_factorial_paired.py",
        ROOT / "scripts/training/_taiji_n1_transition_factorial_learning.py",
        ROOT / "scripts/training/_taiji_n1_transition_factorial_training_receipt.py",
    ]


def collect_source_guards(
    readiness_guards: dict[str, str], *, extra: list[Path] = ()
) -> dict[str, str]:
    guards = {}
    for raw_path, expected in readiness_guards.items():
        path = Path(raw_path).resolve()
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"run83 frozen source chain changed: {path}")
        guards[str(path)] = expected
    for raw_path in extra:
        path = Path(raw_path).resolve()
        expected = sha256(path)
        prior = guards.get(str(path))
        if prior is not None and prior != expected:
            raise ValueError(f"overlapping source guard differs: {path}")
        guards[str(path)] = expected
    if sha256(FROZEN_RUNNER) != FROZEN_RUNNER_SHA256:
        raise ValueError("frozen84 training runner hash changed")
    if sha256(SCHEDULE) != SCHEDULE_SHA256:
        raise ValueError("hard schedule hash changed")
    if sha256(Path(schedule_builder.__file__).resolve()) != BUILDER_SHA256:
        raise ValueError("hard schedule builder hash changed")
    if (
        sha256(ROOT / "reports/taiji_n1_transition_factorial_paired_train_20261016_attempt.json")
        != RUN88_ATTEMPT_SHA256
    ):
        raise ValueError("run88 source attempt hash changed")
    return guards


def source_guards_unchanged(guards: dict[str, str]) -> bool:
    return all(
        Path(path).is_file() and sha256(Path(path)) == digest for path, digest in guards.items()
    )


def make_runner(*, load_inputs: bool) -> tuple[dict, dict, dict | None]:
    if sha256(FROZEN_RUNNER) != FROZEN_RUNNER_SHA256:
        raise ValueError("frozen run84 training runner hash changed")
    if sha256(Path(schedule_builder.__file__).resolve()) != BUILDER_SHA256:
        raise ValueError("frozen hard schedule builder hash changed")
    schedule_payload = _verify_schedule_artifact()
    loaded = runpy.run_path(str(FROZEN_RUNNER), run_name="_taiji_n1_run100_frozen84")
    loaded["torch"].set_num_threads(1)
    audit_function = loaded.get("audit")
    g = getattr(audit_function, "__globals__", None)
    if not isinstance(g, dict):
        raise TypeError("frozen run84 runner has no shared module globals")
    for name in (
        "identity",
        "readiness",
        "known_views",
        "train_one",
        "save",
        "restore",
        "seal_record",
    ):
        function = loaded.get(name)
        if getattr(function, "__globals__", g) is not g:
            raise RuntimeError(f"frozen function {name} has an isolated globals namespace")

    base_identity = g["identity"]()
    protocol_value = protocol(g["PROTOCOL"])
    identity_value = _custom_identity(base_identity, protocol_value)
    copied_update = types.ModuleType("_taiji_n1_run100_learning_copy")
    copied_update.__dict__.update(g["update"].__dict__)
    original_train_step = copied_update.train_step
    original_save = g["save"]
    original_train_one = g["train_one"]
    original_seal = g["seal_record"]

    def custom_identity() -> dict:
        return identity_value

    def scheduled(step: int, items: list[dict], curriculum: dict) -> tuple[dict, dict | None]:
        if type(step) is not int or not 0 <= step < UPDATES:
            raise ValueError("fixed PLAN-N1-100 update step required")
        local = step % 6144
        parent_index = (local // 2) % 192
        if (
            ACTIVE_ITEMS is None
            or len(items) != 192
            or items[parent_index]["id"] != ACTIVE_ITEMS[parent_index]["id"]
        ):
            raise ValueError("scheduled training items differ from the frozen train-only source")
        if local % 2 == 0:
            item = ACTIVE_ITEMS[parent_index]
            second = schedule_builder.core.renaming.second_question(item)
            return item, second
        schedule_index = local // 2
        row = HARD_SCHEDULE[schedule_index]
        if (
            row["schedule_index"] != schedule_index
            or row["parent_id"] != ACTIVE_ITEMS[parent_index]["id"]
        ):
            raise ValueError("hard schedule parent binding changed")
        if ACTIVE_HARD_COUNTS is not None:
            ACTIVE_HARD_COUNTS[
                (
                    row["parent_kind"],
                    row["observation_count"],
                    row["hamming_distance"],
                    row["condition"],
                )
            ] += 1
            ACTIVE_CYCLE_COUNTS[step // 6144] += 1
        return row["item"], row["second_question"]

    def scheduled_train_step(model, optimizer, *args, **kwargs):
        completed = max(
            (int(state["step"]) for state in optimizer.state.values() if "step" in state),
            default=0,
        )
        lr = learning_rate(completed)
        for group in optimizer.param_groups:
            group["lr"] = lr
        if ACTIVE_RATE_LOG is not None and kwargs.get("audit") is not True:
            if completed in ACTIVE_RATE_LOG:
                raise ValueError(f"duplicate PLAN-N1-100 learning-rate update {completed}")
            ACTIVE_RATE_LOG[completed] = lr
        return original_train_step(model, optimizer, *args, **kwargs)

    def expected_hard_counts() -> Counter:
        summary = schedule_payload["summary"]["condition_counts"]
        expected = Counter()
        for label, count in summary.items():
            kind, obs, distance = label.split("/")
            obs_count = int(obs.removeprefix("obs"))
            distance_value = None if distance == "dNone" else int(distance.removeprefix("d"))
            condition = (
                "frozen-rename" if distance_value is None else f"obs{obs_count}-d{distance_value}"
            )
            expected[(kind, obs_count, distance_value, condition)] = count * 4
        return expected

    def schedule_audit(record: dict) -> dict:
        current = ACTIVE_HARD_COUNTS or Counter()
        cycles = ACTIVE_CYCLE_COUNTS or Counter()
        expected = expected_hard_counts()
        complete = record.get("completed_updates") == UPDATES
        valid = (
            complete
            and sum(current.values()) == 12288
            and current == expected
            and cycles == Counter({0: 3072, 1: 3072, 2: 3072, 3: 3072})
        )
        return {
            "schedule_sha256": SCHEDULE_SHA256,
            "hard_view_updates": sum(current.values()),
            "cycle_counts": {str(k): v for k, v in sorted(cycles.items())},
            "condition_counts": {
                f"{kind}/obs{observations}/d{distance}/{condition}": count
                for (kind, observations, distance, condition), count in sorted(
                    current.items(), key=lambda entry: tuple(str(x) for x in entry[0])
                )
            },
            "valid": valid,
        }

    def schedule_rates(record: dict) -> dict:
        current = ACTIVE_RATE_LOG or {}
        expected = {
            step: (BASE_LR if step < DECAY_START else FINAL_LR)
            for step in range(record.get("completed_updates", 0))
        }
        valid = current == expected
        return {
            "base_updates": sum(value == BASE_LR for value in current.values()),
            "final_cycle_updates": sum(value == FINAL_LR for value in current.values()),
            "base_lr": BASE_LR,
            "final_cycle_lr": FINAL_LR,
            "valid": valid,
        }

    def save_with_preflight(model, optimizer, path, completed, seed, arm):
        path = Path(path)
        if path.exists():
            known = PREFLIGHT_CHECKPOINTS.get(path.name)
            if completed != 0 or known is None:
                raise FileExistsError(f"unexpected existing PLAN-N1-100 checkpoint: {path}")
            payload = g["torch"].load(path, map_location="cpu", weights_only=True)
            if (
                payload.get("protocol") != protocol_value
                or payload.get("seed") != seed
                or payload.get("arm") != arm
                or payload.get("phase") != "zero"
                or payload.get("completed_updates") != 0
                or g["paired"].fingerprint(payload["candidate"])
                != g["paired"].fingerprint(model.bundle(optimizer))
                or sha256(path) != known["sha256"]
            ):
                raise ValueError("formal training zero differs from its preflight bundle")
            return payload
        return original_save(model, optimizer, path, completed, seed, arm)

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
                    os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"
                ),
                cwd=ROOT,
            )
            restored = json.loads(output.stdout)
        except Exception:
            budget.accounting_complete = False
            raise
        budget.children += restored["cpu_seconds"]
        budget.guard()
        if restored["signature"] != expected or restored["signature_clone_updates"] != 2:
            raise ValueError("independent PLAN-N1-100 recovery signature differs")
        return dict(path=str(path), sha256=sha256(Path(path)), equal=True, **restored)

    def seal_record(record: dict) -> dict:
        record["hard_schedule_audit"] = schedule_audit(record)
        record["learning_rate_schedule"] = schedule_rates(record)
        summary = original_seal(record)
        summary["hard_schedule_audit"] = record["hard_schedule_audit"]
        summary["learning_rate_schedule"] = record["learning_rate_schedule"]
        return summary

    def train_one(*args, **kwargs):
        global ACTIVE_HARD_COUNTS, ACTIVE_CYCLE_COUNTS, ACTIVE_RATE_LOG
        ACTIVE_HARD_COUNTS = Counter()
        ACTIVE_CYCLE_COUNTS = Counter()
        ACTIVE_RATE_LOG = {}
        try:
            return original_train_one(*args, **kwargs)
        finally:
            pass

    g.update(
        {
            "REPORT": REPORT,
            "ATTEMPT": ATTEMPT,
            "PROTOCOL": protocol_value,
            "UPDATES": UPDATES,
            "MILESTONES": MILESTONES,
            "identity": custom_identity,
            "scheduled": scheduled,
            "train_one": train_one,
            "save": save_with_preflight,
            "independent": independent,
            "seal_record": seal_record,
        }
    )
    copied_update.train_step = scheduled_train_step
    g["update"] = copied_update

    if load_inputs:
        items, curriculum, views, loaded_schedule = load_training_inputs()
        if loaded_schedule != schedule_payload:
            raise ValueError("schedule payload changed during input validation")
        HARD_SCHEDULE[:] = schedule_payload["items"]
        ACTIVE_ITEMS = items

        def known_views() -> tuple[list[dict], dict, dict]:
            return items, curriculum, views

        g["known_views"] = known_views
        return loaded, g, {"items": items, "curriculum": curriculum, "views": views}
    HARD_SCHEDULE[:] = schedule_payload["items"]
    ACTIVE_ITEMS = None
    return loaded, g, None


def _restore_worker(path: Path) -> dict:
    _, g, _ = make_runner(load_inputs=False)
    return g["restore"](Path(path))


def checkpoint_preflight() -> dict:
    started = time.process_time()
    if (
        any(path.exists() for path in (PREFLIGHT, PREFLIGHT_ATTEMPT, PREFLIGHT_BINDING))
        or CHECKPOINT_ROOT.exists()
    ):
        raise FileExistsError("PLAN-N1-100 checkpoint preflight is one-shot")
    loaded, g, inputs = make_runner(load_inputs=True)
    if not verify_learning_rate_schedule():
        raise ValueError("fixed learning-rate boundary check failed")
    readiness_guards, sources = g["readiness"]()
    guards = collect_source_guards(readiness_guards, extra=source_paths())
    identity_value = g["identity"]()
    write_new(
        PREFLIGHT_ATTEMPT,
        {
            "schema": "n1-hard-neighbor-checkpoint-preflight-attempt-v1",
            "status": "started",
            "protocol": g["PROTOCOL"],
            "code_identity": identity_value,
            "source_guards": guards,
            "training_updates": 0,
            "known_model_queries": 0,
            "dev_queries": 0,
            "holdout_queries": 0,
            "confirmation_queries": 0,
        },
    )
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=False)
    budget = g["paired"].CpuBudget(PREFLIGHT_CPU_LIMIT_SECONDS)
    zeros = []
    error = None
    initial_by_seed = Counter()
    try:
        for seed in SEEDS:
            for arm in ARMS:
                budget.guard(reserve=10)
                source_payload = g["torch"].load(
                    sources[seed, arm], map_location="cpu", weights_only=True
                )
                model, optimizer = g["engineering"].load(source_payload)
                g["validate_state"](model, optimizer, 0, seed, arm)
                fingerprint = g["paired"].fingerprint(model.bundle(optimizer))
                if fingerprint != g["engineering"].initial_fingerprint(seed):
                    raise ValueError(
                        f"run83 source is not a complete fresh zero for seed {seed}/{arm}"
                    )
                initial_by_seed[seed] += 1
                path = CHECKPOINT_ROOT / f"s{seed}_{arm}_zero.pt"
                saved = g["save"](model, optimizer, path, 0, seed, arm)
                restored = g["independent"](path, saved["expected_signature"], budget)
                if not restored["equal"]:
                    raise ValueError(f"independent zero restore failed for seed {seed}/{arm}")
                entry = {
                    "seed": seed,
                    "arm": arm,
                    "path_name": path.name,
                    "sha256": sha256(path),
                    "bundle_sha256": fingerprint,
                    "matches_run83_full_fresh_zero": True,
                    "restore_equal": restored["equal"],
                    "restore_cpu_seconds": restored["cpu_seconds"],
                    "signature_clone_updates": 4,
                }
                zeros.append(entry)
                PREFLIGHT_CHECKPOINTS[path.name] = entry
        if any(initial_by_seed[seed] != len(ARMS) for seed in SEEDS):
            raise ValueError("all three same-seed fresh-arm source bundles are required")
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"

    cpu_seconds = round(time.process_time() - started + budget.children, 6)
    source_unchanged = source_guards_unchanged(guards)
    complete = (
        error is None
        and len(zeros) == 9
        and all(entry["restore_equal"] for entry in zeros)
        and cpu_seconds <= PREFLIGHT_CPU_LIMIT_SECONDS
        and budget.accounting_complete
        and source_unchanged
    )
    report = {
        "schema": "n1-hard-neighbor-nine-zero-checkpoint-preflight-v1",
        "complete": complete,
        "status": "complete" if complete else "stopped",
        "protocol": g["PROTOCOL"],
        "code_identity": identity_value,
        "source_guards": guards,
        "zero_checkpoints": zeros,
        "model_loads": len(zeros),
        "checkpoint_loads": len(zeros) * 2,
        "training_updates": 0,
        "signature_clone_updates": len(zeros) * 4,
        "independent_restores": len(zeros),
        "cpu_seconds": cpu_seconds,
        "cpu_limit_seconds": PREFLIGHT_CPU_LIMIT_SECONDS,
        "cpu_accounting_complete": budget.accounting_complete,
        "sources_unchanged": source_unchanged,
        "learning_rate_contract_verified": verify_learning_rate_schedule(),
        "known_model_queries": 0,
        "dev_queries": 0,
        "holdout_queries": 0,
        "confirmation_queries": 0,
        "can_promote": False,
        "error": error,
    }
    write_new(PREFLIGHT, report)
    write_new(
        PREFLIGHT_BINDING,
        {
            "schema": "n1-hard-neighbor-preflight-binding-v1",
            "preflight_report_sha256": sha256(PREFLIGHT),
            "code_identity": identity_value,
            "schedule_sha256": SCHEDULE_SHA256,
        },
    )
    if not complete:
        raise RuntimeError(error or "PLAN-N1-100 checkpoint preflight did not pass")
    return report


def _validate_preflight(g: dict, identity_value: dict) -> tuple[float, dict[str, dict]]:
    report = read_json(PREFLIGHT)
    binding = read_json(PREFLIGHT_BINDING)
    if (
        report.get("complete") is not True
        or report.get("status") != "complete"
        or report.get("code_identity") != identity_value
        or report.get("protocol") != g["PROTOCOL"]
        or binding.get("preflight_report_sha256") != sha256(PREFLIGHT)
        or binding.get("code_identity") != identity_value
        or binding.get("schedule_sha256") != SCHEDULE_SHA256
        or len(report.get("zero_checkpoints", [])) != 9
        or any(not entry.get("restore_equal") for entry in report["zero_checkpoints"])
    ):
        raise ValueError("complete bound nine-model zero checkpoint preflight required")
    checkpoints = {entry["path_name"]: entry for entry in report["zero_checkpoints"]}
    expected_names = {f"s{seed}_{arm}_zero.pt" for seed in SEEDS for arm in ARMS}
    if set(checkpoints) != expected_names:
        raise ValueError("exact run83 zero path for every seed/arm required")
    for name, entry in checkpoints.items():
        path = CHECKPOINT_ROOT / name
        if not path.is_file() or sha256(path) != entry["sha256"]:
            raise ValueError(f"preflight zero checkpoint changed: {name}")
    return float(report["cpu_seconds"]), checkpoints


def _load_shard(model: dict, g: dict) -> dict:
    seal = model["shard"]
    identity_value = {
        "seed": model["seed"],
        "arm": model["arm"],
        "protocol": g["PROTOCOL"],
        "phase": "complete" if model["complete"] else "partial",
    }
    return g["shard"].read_checked(Path(seal["path"]), seal, identity=identity_value)


def training_audit() -> dict:
    if any(path.exists() for path in (REPORT, ATTEMPT, PROGRESS)):
        raise FileExistsError("PLAN-N1-100 formal nine-model batch is one-shot")
    started = time.process_time()
    loaded, g, inputs = make_runner(load_inputs=True)
    items = inputs["items"]
    curriculum = inputs["curriculum"]
    views = inputs["views"]
    identity_value = g["identity"]()
    preflight_cpu, checkpoints = _validate_preflight(g, identity_value)
    readiness_guards, sources = g["readiness"]()
    guards = collect_source_guards(
        readiness_guards,
        extra=[
            *source_paths(),
            PREFLIGHT,
            PREFLIGHT_BINDING,
            PREFLIGHT_ATTEMPT,
            *(CHECKPOINT_ROOT / name for name in checkpoints),
        ],
    )
    if set(sources) != {(seed, arm) for seed in SEEDS for arm in ARMS}:
        raise ValueError("complete run83 nine-zero source map required")
    if any(REPORT.parent.glob(f"{REPORT.stem}_s*.json.gz")):
        raise FileExistsError("PLAN-N1-100 model shards already exist; no restart")
    write_new(
        ATTEMPT,
        {
            "schema": "n1-hard-neighbor-training-attempt-v1",
            "status": "started",
            "started_cpu_clock": started,
            "preflight_report_sha256": sha256(PREFLIGHT),
            "protocol": g["PROTOCOL"],
            "code_identity": identity_value,
            "source_guards": guards,
            "schedule_sha256": SCHEDULE_SHA256,
            "training_updates": 0,
            "dev_queries": 0,
            "holdout_queries": 0,
            "confirmation_queries": 0,
            "can_promote": False,
        },
    )
    with PROGRESS.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps({"status": "started", "schedule_sha256": SCHEDULE_SHA256}) + "\n")
        stream.flush()
        os.fsync(stream.fileno())

    PREFLIGHT_CHECKPOINTS.update(checkpoints)
    budget = g["paired"].CpuBudget(CPU_LIMIT_SECONDS - preflight_cpu)
    result = {
        "schema": "n1-hard-neighbor-training-result-v1",
        "protocol": g["PROTOCOL"],
        "code_identity": identity_value,
        "source_guards": guards,
        "preflight_report_sha256": sha256(PREFLIGHT),
        "schedule_sha256": SCHEDULE_SHA256,
        "models": [],
        "dev_queries": 0,
        "holdout_queries": 0,
        "confirmation_queries": 0,
        "can_promote": False,
        "status": "running",
    }
    error = None
    try:
        for seed in SEEDS:
            for arm in ARMS:
                record = g["train_one"](
                    seed,
                    arm,
                    sources[seed, arm],
                    CHECKPOINT_ROOT,
                    items,
                    curriculum,
                    views,
                    budget,
                )
                result["models"].append(record)
                with PROGRESS.open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write(
                        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                    )
                    stream.flush()
                    os.fsync(stream.fileno())
                if (
                    not record.get("complete")
                    or not record.get("hard_schedule_audit", {}).get("valid")
                    or not record.get("learning_rate_schedule", {}).get("valid")
                ):
                    raise ValueError(
                        f"model trajectory or hard schedule contract failed for {seed}/{arm}"
                    )
                budget.guard(reserve=20)
        old_protocol = training_receipt.train.PROTOCOL
        old_identity = training_receipt.train.identity
        try:
            training_receipt.train.PROTOCOL = g["PROTOCOL"]
            training_receipt.train.identity = g["identity"]
            for model in result["models"]:
                shard_record = training_receipt.read_model(model)
                training_receipt.validate_model(model, shard_record, views=views)
        finally:
            training_receipt.train.PROTOCOL = old_protocol
            training_receipt.train.identity = old_identity
        result["status"] = "complete"
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        result.update(status="stopped", error=error)

    result.update(
        preflight_cpu_seconds=preflight_cpu,
        training_cpu_seconds=round(budget.used(), 6),
        cpu_seconds=round(preflight_cpu + budget.used(), 6),
        cpu_limit_seconds=CPU_LIMIT_SECONDS,
        cpu_accounting_complete=budget.accounting_complete,
        sources_unchanged=source_guards_unchanged(guards),
        code_unchanged=g["identity"]() == identity_value,
        completed_updates=sum(model.get("completed_updates", 0) for model in result["models"]),
        qa_sequences=sum(model.get("qa_sequences", 0) for model in result["models"]),
        new_session_count=sum(model.get("new_session_count") or 0 for model in result["models"]),
        score_queries=sum(model.get("score_queries", 0) for model in result["models"]),
        signature_clone_updates=sum(
            model.get("signature_clone_updates", 0) for model in result["models"]
        ),
        synthetic_qa_sequences=4
        * sum(model.get("signature_clone_updates", 0) for model in result["models"]),
        hard_view_updates=sum(
            model.get("hard_schedule_audit", {}).get("hard_view_updates", 0)
            for model in result["models"]
        ),
        training_sufficient=(
            result["status"] == "complete"
            and all(
                model.get("training_sufficient")
                for model in result["models"]
                if model.get("arm") == "position"
            )
            and len([model for model in result["models"] if model.get("arm") == "position"]) == 3
        ),
    )
    result["complete"] = (
        result["status"] == "complete"
        and len(result["models"]) == 9
        and result["completed_updates"] == 9 * UPDATES
        and result["qa_sequences"] == 9 * 55296
        and result["new_session_count"] == 9 * 43008
        and result["score_queries"] == 9 * 4800
        and result["signature_clone_updates"] == 9 * 8
        and result["synthetic_qa_sequences"] == 9 * 32
        and result["hard_view_updates"] == 9 * 12288
        and result["cpu_seconds"] <= CPU_LIMIT_SECONDS
        and result["cpu_accounting_complete"]
        and result["sources_unchanged"]
        and result["code_unchanged"]
        and not any(
            result[key] for key in ("dev_queries", "holdout_queries", "confirmation_queries")
        )
    )
    result["can_promote"] = False
    write_new(REPORT, result)
    if error:
        raise RuntimeError(error)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--checkpoint-preflight", action="store_true")
    mode.add_argument("--execute-approved", action="store_true")
    mode.add_argument("--restore-worker", type=Path)
    args = parser.parse_args()
    if args.restore_worker:
        result = _restore_worker(args.restore_worker)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    if args.checkpoint_preflight:
        result = checkpoint_preflight()
        print(
            json.dumps(
                {
                    "complete": result["complete"],
                    "report": str(PREFLIGHT),
                    "cpu_seconds": result["cpu_seconds"],
                }
            )
        )
        return 0
    result = training_audit()
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "complete",
                    "training_sufficient",
                    "completed_updates",
                    "cpu_seconds",
                    "error",
                )
            }
        )
    )
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
