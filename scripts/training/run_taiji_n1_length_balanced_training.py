"""PLAN-N1-110 candidate runner with frozen PLAN-N1-100 training machinery."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import (  # noqa: E402
    build_taiji_n1_length_balanced_training_schedule as candidate_schedule,
)
from scripts.training import seal_ast  # noqa: E402
from seed_platform.paths import get_external_path  # noqa: E402

PLAN = (
    ROOT
    / "plans/reference/PLAN-N1-110_length_balanced_training_and_v11_confirmation_draft_20261017.md"
)
BASE_PLAN = ROOT / "plans/reference/PLAN-N1-100_balanced_hard_neighbor_training_and_v7_20261017.md"
BASE_CONTROLLER = ROOT / "scripts/training/run_taiji_n1_hard_neighbor_training.py"
BASE_TRAIN_REPORT = ROOT / "reports/taiji_n1_hard_neighbor_training_20261017.json"
BASELINE_CHECKPOINT_ROOT = (
    Path(get_external_path("data/checkpoints")) / "n1_hard_neighbor_training_v1_20261017"
).resolve()
TRAIN_SCHEDULE = candidate_schedule.TRAIN_SCHEDULE
SCHEDULE_CHECKS = candidate_schedule.CHECKS
CONTRACT_TEST = candidate_schedule.CONTRACT_TEST
RUNNER_TEST = ROOT / "tests/taiji_native/test_n1_plan110_length_balanced_runner_contract.py"

REPORT = ROOT / "reports/taiji_n1_length_balanced_training_20261018.json"
ATTEMPT = ROOT / "reports/taiji_n1_length_balanced_training_20261018_attempt.json"
PROGRESS = ROOT / "reports/taiji_n1_length_balanced_training_20261018_progress.jsonl"
PREFLIGHT = ROOT / "reports/taiji_n1_length_balanced_training_20261018_checkpoint_preflight.json"
PREFLIGHT_ATTEMPT = (
    ROOT / "reports/taiji_n1_length_balanced_training_20261018_checkpoint_preflight_attempt.json"
)
PREFLIGHT_BINDING = (
    ROOT / "reports/taiji_n1_length_balanced_training_20261018_checkpoint_preflight_binding.json"
)
RUNNER_READINESS = ROOT / "reports/taiji_n1_length_balanced_training_20261018_runner_readiness.json"
CHECKPOINT_ROOT = (
    Path(get_external_path("data/checkpoints"))
    / "n1_hard_neighbor_length_balanced_training_20261018"
).resolve()

PLAN_SHA256 = "16d0cecb325af6463c107973fc0e741478140134a7642cd64c86e045c6069783"
SCHEDULE_SHA256 = "39edaf58020f6cf18680838a6d7576eafef5c7c9286cfd1ea40b93a581851380"
CHECKS_SHA256 = "51cb205b17a1319b86c7f73c4f0158957cbeaf73b933aa9a942aba52cfabf0d9"
BUILDER_SHA256 = "94d5abc2f53660d5d8f8e15f41b7c2ad87ca76849d9d55d911b68d9b8238c600"
CONTRACT_TEST_SHA256 = "41c408aa290edcf8cc3e6006fe71cde7746eb9dd89e1eb174d7f9f9f286cc37b"
RUNNER_TEST_SHA256 = "8ba902d9d613b7c755ee19796443f2c15e3e0b21299d6a01c983171f658d4060"
BASE_CONTROLLER_SHA256 = "8d161c0f2d6c5b275d0aaa88ac5709bbef4db6087359594f65515d868636857b"
BASE_SCHEDULE_BUILDER_SHA256 = "efa81a0bbc87f84db11e46cc40bc33e986fcfb6185c12b0404e9fccea4489d79"
BASE_PLAN_SHA256 = "3a02e7395f2e91c4ea654cb37e076472c697efb09d2a0a2c8d534d3c6975b773"
BASE_TRAIN_REPORT_SHA256 = "dad10ec72c605367f99da035bab07d522ce1793154915ebcc3c7f1022f411909"
BASELINE_FINAL_CHECKPOINTS = {
    "s11_position_final.pt": "294b12e3a2dd7cee17536a2d8f322ee15c6843f6861c7138d3fd89621b271533",
    "s11_terminal_final.pt": "2531c678c0b1bfadca9b81e5f67d52f037b69705313a629674d0c401e725b41a",
    "s11_cut_final.pt": "ab4e54c040e887e10cc55c15c3a6bb34b27c3e17828e4ab8d7660a98b72190af",
    "s12_position_final.pt": "97024a97c8d0e5f79af532af44e81e603716eee090ab738c112f06d94d9458e5",
    "s12_terminal_final.pt": "70613239de596046704304b9fefbe38e8195ed0bdf1ec62eb365606d95baf803",
    "s12_cut_final.pt": "a96972dc276ca33d049fc402e9299f287482c89782ca7851daa55d858e5300a8",
    "s13_position_final.pt": "09df1d47eb490ef9afc5d563d08b40eca6b51951096875d016a67f133f7ed653",
    "s13_terminal_final.pt": "d191e1d6855942524d9164a390b253b58b09a7ebd4eb5ca2570b5eb1c5f6472f",
    "s13_cut_final.pt": "04dced31494a4e69e8c7516fbf74540a813fc0866f0f1a82ec5fd3459329e9dd",
}
LENGTH_ASSIGNMENT_SEED = 11010
VALUE_MAP_SEED = 11011
CPU_LIMIT_SECONDS = 18_000
PREFLIGHT_CPU_LIMIT_SECONDS = 300
UPDATES_PER_MODEL = 24_576
MATERIAL_LENGTHS = (6, 7)

EXPECTED_PER_PATH = {
    (kind, length): 1_536 for kind in ("lookup", "revision", "unknown") for length in (6, 7)
} | {("transition", None): 3_072}

_BASE_MODULE = None
_LENGTH_AUDITS: dict[tuple[int, str], dict] = {}
_ISOLATED_BASE_MODULE_NAME = "_seed_plan110_isolated_run100_controller"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def restore_worker_command(checkpoint: Path, *, python: Path | None = None) -> list[str]:
    interpreter = Path(sys.executable if python is None else python)
    if interpreter.name.lower() == "pythonw.exe":
        interpreter = interpreter.with_name("python.exe")
    return [str(interpreter), str(Path(__file__).resolve()), "--restore-worker", str(checkpoint)]


def verify_candidate_sources() -> tuple[dict, dict]:
    expected = {
        PLAN: PLAN_SHA256,
        BASE_PLAN: BASE_PLAN_SHA256,
        BASE_CONTROLLER: BASE_CONTROLLER_SHA256,
        Path(candidate_schedule.base_builder.__file__).resolve(): BASE_SCHEDULE_BUILDER_SHA256,
        BASE_TRAIN_REPORT: BASE_TRAIN_REPORT_SHA256,
        TRAIN_SCHEDULE: SCHEDULE_SHA256,
        SCHEDULE_CHECKS: CHECKS_SHA256,
        Path(candidate_schedule.__file__).resolve(): BUILDER_SHA256,
        CONTRACT_TEST: CONTRACT_TEST_SHA256,
        RUNNER_TEST: RUNNER_TEST_SHA256,
    }
    expected.update(
        {
            BASELINE_CHECKPOINT_ROOT / name: digest
            for name, digest in BASELINE_FINAL_CHECKPOINTS.items()
        }
    )
    for path, digest in expected.items():
        if not Path(path).is_file():
            raise FileNotFoundError(f"candidate source missing: {path}")
        # S6: .py 钉点为历史事实——在位字节或 git 历史 blob 任一核验通过即可（当前文件解冻）
        if not seal_ast.verify_path_pin(path, digest):
            raise ValueError(f"candidate source hash mismatch: {path}")
    if not CHECKPOINT_ROOT.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("candidate checkpoint root escaped the authorized workspace")
    schedule = read_json(TRAIN_SCHEDULE)
    checks = read_json(SCHEDULE_CHECKS)
    baseline = read_json(BASE_TRAIN_REPORT)
    baseline_models = baseline.get("models", [])
    if (
        baseline.get("status") != "complete"
        or baseline.get("complete") is not True
        or baseline.get("training_sufficient") is not True
        or len(baseline_models) != 9
        or any(model.get("completed_updates") != UPDATES_PER_MODEL for model in baseline_models)
        or {(model.get("seed"), model.get("arm")) for model in baseline_models}
        != {(seed, arm) for seed in (11, 12, 13) for arm in ("position", "terminal", "cut")}
    ):
        raise ValueError("PLAN-N1-100 baseline is not a complete frozen nine-endpoint cohort")
    if (
        schedule.get("schema") != candidate_schedule.SCHEMA
        or schedule.get("plan_sha256") != PLAN_SHA256
        or schedule.get("builder_sha256") != BUILDER_SHA256
        or checks.get("accepted") is not True
        or checks.get("errors") != []
        or checks.get("plan_sha256") != PLAN_SHA256
        or checks.get("schedule_file_sha256") != SCHEDULE_SHA256
        or checks.get("source_guards") != schedule.get("source_guards")
        or checks.get("model_queries") != 0
        or checks.get("parameter_updates") != 0
        or checks.get("checkpoint_loads") != 0
    ):
        raise ValueError("candidate schedule/check receipt differs from frozen PLAN-N1-110")
    guards_ok, guard_failures = seal_ast.verify_hash_dict(
        schedule.get("source_guards", {}), allow_git_history=True
    )
    if not guards_ok:
        raise ValueError(f"candidate schedule source guard changed: {guard_failures[:3]}")
    return schedule, checks


def expected_length_ledger() -> dict[tuple[str, str, int | None], int]:
    return {
        (path, kind, length): count
        for path in ("raw", "hard")
        for (kind, length), count in EXPECTED_PER_PATH.items()
    }


def summarize_length_ledger(ledger: Counter) -> dict:
    expected = expected_length_ledger()
    observed = {key: ledger[key] for key in expected}
    extras = {key: count for key, count in ledger.items() if key not in expected and count}
    return {
        "valid": observed == expected and not extras,
        "expected": {
            f"{path}/{kind}/len{length}": count for (path, kind, length), count in expected.items()
        },
        "observed": {
            f"{path}/{kind}/len{length}": count for (path, kind, length), count in observed.items()
        },
        "extra": {"/".join(map(str, key)): count for key, count in extras.items()},
        "raw_slots": sum(count for (path, _, _), count in observed.items() if path == "raw"),
        "hard_slots": sum(count for (path, _, _), count in observed.items() if path == "hard"),
    }


def make_candidate_scheduler(
    base_module, schedule: dict, items: list[dict] | None, ledger: Counter
):
    raw_rows = {(row["parent_id"], row["raw_slot"]): row for row in schedule["raw_views"]}
    hard_rows = {row["schedule_index"]: row for row in schedule["items"]}
    if len(raw_rows) != 3072 or len(hard_rows) != 3072:
        raise ValueError("candidate raw/hard schedule key coverage is incomplete")

    def scheduled(step: int, active_items: list[dict], curriculum: dict):
        if type(step) is not int or not 0 <= step < UPDATES_PER_MODEL:
            raise ValueError("candidate update step outside PLAN-N1-110 trajectory")
        if items is None or len(items) != 192 or len(active_items) != 192:
            raise ValueError("complete candidate raw train-only parent set is required")
        local = step % 6144
        parent_index = (local // 2) % 192
        parent = items[parent_index]
        if active_items[parent_index]["id"] != parent["id"]:
            raise ValueError("candidate training parent order differs from frozen train source")
        if local % 2 == 0:
            raw_slot = local // 384
            row = raw_rows[(parent["id"], raw_slot)]
            if row["parent_kind"] != parent["kind"]:
                raise ValueError("raw-view length schedule parent kind mismatch")
            ledger[("raw", row["parent_kind"], row["material_length_chars"])] += 1
            return row["item"], row["second_question"]

        schedule_index = local // 2
        row = hard_rows[schedule_index]
        if (
            row["schedule_index"] != schedule_index
            or row["parent_id"] != parent["id"]
            or row["parent_kind"] != parent["kind"]
        ):
            raise ValueError("hard-view parent/order binding changed")
        if base_module.ACTIVE_HARD_COUNTS is not None:
            base_module.ACTIVE_HARD_COUNTS[
                (
                    row["parent_kind"],
                    row["observation_count"],
                    row["hamming_distance"],
                    row["condition"],
                )
            ] += 1
            base_module.ACTIVE_CYCLE_COUNTS[step // 6144] += 1
        ledger[("hard", row["parent_kind"], row["material_length_chars"])] += 1
        return row["item"], row["second_question"]

    return scheduled


def _source_paths(base_module, original_source_paths) -> list[Path]:
    paths = list(original_source_paths())
    paths.extend(
        [
            Path(__file__).resolve(),
            PLAN,
            BASE_PLAN,
            BASE_CONTROLLER,
            Path(candidate_schedule.base_builder.__file__).resolve(),
            Path(candidate_schedule.__file__).resolve(),
            TRAIN_SCHEDULE,
            SCHEDULE_CHECKS,
            CONTRACT_TEST,
            RUNNER_TEST,
            BASE_TRAIN_REPORT,
            *[BASELINE_CHECKPOINT_ROOT / name for name in sorted(BASELINE_FINAL_CHECKPOINTS)],
            *([RUNNER_READINESS] if RUNNER_READINESS.is_file() else []),
        ]
    )
    return list(dict.fromkeys(Path(path).resolve() for path in paths))


def _load_isolated_base_controller():
    existing = sys.modules.get(_ISOLATED_BASE_MODULE_NAME)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(_ISOLATED_BASE_MODULE_NAME, BASE_CONTROLLER)
    if spec is None or spec.loader is None:
        raise ImportError("could not create an isolated PLAN-N1-100 controller module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_ISOLATED_BASE_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(_ISOLATED_BASE_MODULE_NAME, None)
        raise
    for name in ("make_runner", "training_audit", "checkpoint_preflight"):
        function = getattr(module, name, None)
        if getattr(function, "__globals__", None) is not module.__dict__:
            raise RuntimeError(
                f"isolated controller function {name} has a shared globals namespace"
            )
    return module


def _build_training_inputs():
    items, manifest_sha = candidate_schedule.train_items_from_frozen_manifest()
    if manifest_sha != candidate_schedule.MANIFEST_SHA256:
        raise ValueError("candidate source manifest hash mismatch")
    items, curriculum, pairs = candidate_schedule.load_training_sources(items)
    payload, _ = verify_candidate_sources()
    rebuilt, checks = candidate_schedule.build_from_frozen_sources()
    if rebuilt != payload or checks.get("accepted") is not True:
        raise ValueError("candidate schedule does not regenerate byte-for-byte")
    probes = [
        candidate_schedule.core.renaming.rename_item(item, entry["names"])
        for item, entry in zip(items, curriculum["probes"], strict=True)
    ]
    views = {
        "original": items,
        "pair": pairs,
        "rename": probes,
        "rename_pair": candidate_schedule.core.build_pairs(probes),
    }
    return items, curriculum, views, payload


def configure_candidate_runner():
    global _BASE_MODULE
    if _BASE_MODULE is not None:
        return _BASE_MODULE
    verify_candidate_sources()
    base = _load_isolated_base_controller()
    original_make_runner = base.make_runner
    original_source_paths = base.source_paths
    original_protocol = base.protocol
    original_write_new = base.write_new

    candidate_schedule.train_items_from_frozen_manifest = (
        candidate_schedule.base_builder.train_items_from_frozen_manifest
    )
    candidate_schedule.load_training_sources = candidate_schedule.base_builder.load_training_sources
    candidate_schedule.validate_train_items = candidate_schedule.base_builder.validate_train_items
    candidate_schedule.core = candidate_schedule.base_builder.core
    candidate_schedule.MANIFEST = candidate_schedule.base_builder.MANIFEST
    candidate_schedule.MANIFEST_SHA256 = candidate_schedule.base_builder.MANIFEST_SHA256
    candidate_schedule.RENAMING = candidate_schedule.base_builder.RENAMING
    candidate_schedule.RENAMING_SHA256 = candidate_schedule.base_builder.RENAMING_SHA256
    candidate_schedule.PAIRS = candidate_schedule.base_builder.PAIRS
    candidate_schedule.PAIRS_SHA256 = candidate_schedule.base_builder.PAIRS_SHA256
    base.schedule_builder = candidate_schedule

    base.PLAN = PLAN
    base.SCHEDULE = TRAIN_SCHEDULE
    base.SCHEDULE_SHA256 = SCHEDULE_SHA256
    base.BUILDER_SHA256 = BUILDER_SHA256
    base.REPORT = REPORT
    base.ATTEMPT = ATTEMPT
    base.PROGRESS = PROGRESS
    base.PREFLIGHT = PREFLIGHT
    base.PREFLIGHT_ATTEMPT = PREFLIGHT_ATTEMPT
    base.PREFLIGHT_BINDING = PREFLIGHT_BINDING
    base.CHECKPOINT_ROOT = CHECKPOINT_ROOT
    base.CPU_LIMIT_SECONDS = CPU_LIMIT_SECONDS
    base.PREFLIGHT_CPU_LIMIT_SECONDS = PREFLIGHT_CPU_LIMIT_SECONDS

    def verify_schedule_artifact() -> dict:
        schedule, _ = verify_candidate_sources()
        return schedule

    def load_training_inputs():
        return _build_training_inputs()

    def protocol(base_protocol: dict | None = None) -> dict:
        value = original_protocol(base_protocol)
        schedule, _ = verify_candidate_sources()
        value.update(
            schema="n1-hard-neighbor-length-balanced-training-v1",
            candidate_schedule_sha256=SCHEDULE_SHA256,
            candidate_checks_sha256=CHECKS_SHA256,
            material_value_map_sha256=schedule["value_map_sha256"],
            length_assignment_seed=LENGTH_ASSIGNMENT_SEED,
            value_map_seed=VALUE_MAP_SEED,
            raw_length_slots_per_parent={"6": 8, "7": 8},
            hard_length_balance="per-parent/per-condition-50-50",
            baseline_training_report_sha256=BASE_TRAIN_REPORT_SHA256,
            baseline_final_checkpoint_hashes=dict(sorted(BASELINE_FINAL_CHECKPOINTS.items())),
            cpu_limit_seconds=CPU_LIMIT_SECONDS,
        )
        return value

    def source_paths() -> list[Path]:
        return _source_paths(base, original_source_paths)

    def write_new(path: Path, value: dict) -> None:
        if Path(path).resolve() == REPORT.resolve():
            audit_map = {
                f"{seed}/{arm}": audit for (seed, arm), audit in sorted(_LENGTH_AUDITS.items())
            }
            length_valid = len(audit_map) == 9 and all(
                audit.get("valid") is True for audit in audit_map.values()
            )
            value["length_balance_audit"] = {
                "valid": length_valid,
                "per_endpoint": audit_map,
                "raw_length_slots_per_parent": {"6": 8, "7": 8},
                "schedule_sha256": SCHEDULE_SHA256,
            }
            value["length_balance_valid"] = length_valid
            value["complete"] = value.get("complete") is True and length_valid
            if not length_valid:
                value["status"] = "stopped"
                value["can_promote"] = False
        original_write_new(path, value)

    def wrapped_make_runner(*, load_inputs: bool):
        loaded, globals_dict, inputs = original_make_runner(load_inputs=load_inputs)
        schedule, _ = verify_candidate_sources()
        parent_items = inputs["items"] if inputs is not None else None
        ledger: Counter = Counter()
        globals_dict["length_ledger"] = ledger
        globals_dict["scheduled"] = make_candidate_scheduler(base, schedule, parent_items, ledger)
        if inputs is not None:
            original_train_one = globals_dict["train_one"]

            def train_one(*args, **kwargs):
                ledger.clear()
                seed, arm = args[0], args[1]
                try:
                    record = original_train_one(*args, **kwargs)
                except BaseException:
                    _LENGTH_AUDITS[(seed, arm)] = summarize_length_ledger(ledger)
                    raise
                audit = summarize_length_ledger(ledger)
                _LENGTH_AUDITS[(seed, arm)] = audit
                if not audit["valid"]:
                    raise ValueError(f"length ledger failed for {seed}/{arm}: {audit}")
                return record

            globals_dict["train_one"] = train_one
        globals_dict["independent"] = _candidate_independent
        return loaded, globals_dict, inputs

    base._verify_schedule_artifact = verify_schedule_artifact
    base.load_training_inputs = load_training_inputs
    base.protocol = protocol
    base.source_paths = source_paths
    base.make_runner = wrapped_make_runner
    base.write_new = write_new
    _BASE_MODULE = base
    return base


def _candidate_independent(path: Path, expected: str, budget) -> dict:
    if _BASE_MODULE is None:
        raise RuntimeError("candidate runner is not configured")
    budget.guard(reserve=10)
    python = Path(sys.executable)
    if python.name.lower() == "pythonw.exe":
        python = python.with_name("python.exe")
    try:
        output = subprocess.run(
            restore_worker_command(path, python=python),
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
            cwd=ROOT,
        )
        restored = json.loads(output.stdout)
    except Exception:
        budget.accounting_complete = False
        raise
    budget.children += restored["cpu_seconds"]
    budget.guard()
    if restored.get("signature") != expected or restored.get("signature_clone_updates") != 2:
        raise ValueError("candidate checkpoint independent restore signature mismatch")
    return dict(path=str(path), sha256=sha256_file(Path(path)), equal=True, **restored)


def validate_candidate_runner() -> dict:
    """Simulate every fixed schedule step without loading weights or updating parameters."""
    import time

    started = time.process_time()
    if RUNNER_READINESS.exists():
        raise FileExistsError("candidate runner readiness report is write-once")
    if any(
        path.exists() for path in (PREFLIGHT, PREFLIGHT_ATTEMPT, PREFLIGHT_BINDING, REPORT, ATTEMPT)
    ):
        raise FileExistsError("candidate preflight/training artifacts already exist")
    if CHECKPOINT_ROOT.exists():
        raise FileExistsError("candidate checkpoint root already exists before validation")

    schedule, _ = verify_candidate_sources()
    source_paths = [
        PLAN,
        BASE_PLAN,
        BASE_CONTROLLER,
        Path(candidate_schedule.base_builder.__file__).resolve(),
        Path(candidate_schedule.__file__).resolve(),
        TRAIN_SCHEDULE,
        SCHEDULE_CHECKS,
        CONTRACT_TEST,
        RUNNER_TEST,
        BASE_TRAIN_REPORT,
        *[BASELINE_CHECKPOINT_ROOT / name for name in sorted(BASELINE_FINAL_CHECKPOINTS)],
    ]
    before = {str(Path(path).resolve()): sha256_file(Path(path)) for path in source_paths}
    base = configure_candidate_runner()
    loaded, globals_dict, inputs = base.make_runner(load_inputs=True)
    if inputs is None or len(inputs["items"]) != 192:
        raise ValueError("candidate runner did not load all 192 train-only parents")
    readiness_guards, readiness_sources = globals_dict["readiness"]()
    guards = base.collect_source_guards(readiness_guards, extra=base.source_paths())
    if not base.verify_learning_rate_schedule():
        raise ValueError("frozen 24,576-update learning-rate schedule failed verification")

    base.ACTIVE_HARD_COUNTS = Counter()
    base.ACTIVE_CYCLE_COUNTS = Counter()
    for step in range(UPDATES_PER_MODEL):
        item, second = globals_dict["scheduled"](step, inputs["items"], inputs["curriculum"])
        if item.get("split") != "train":
            raise ValueError("candidate scheduler emitted a non-train view")
        if (
            item.get("kind") in {"lookup", "revision"}
            and len(item.get("answer", "")) not in MATERIAL_LENGTHS
        ):
            raise ValueError("candidate known material answer is outside lengths 6/7")
        if item.get("kind") == "unknown" and item.get("answer") != "unknown":
            raise ValueError("candidate unknown view changed its literal abstention label")
        if item.get("kind") in {"lookup", "unknown"} and second is None:
            raise ValueError("candidate paired question is missing")
        if item.get("kind") in {"revision", "transition"} and second is not None:
            raise ValueError("candidate revision/transition unexpectedly has a second question")

    length_audit = summarize_length_ledger(globals_dict["length_ledger"])
    expected_hard = Counter()
    for label, count in schedule["summary"]["condition_counts"].items():
        kind, obs, distance = label.split("/")
        observations = int(obs.removeprefix("obs"))
        distance_value = None if distance == "dNone" else int(distance.removeprefix("d"))
        condition = (
            "frozen-rename" if distance_value is None else f"obs{observations}-d{distance_value}"
        )
        expected_hard[(kind, observations, distance_value, condition)] = count * 4
    if (
        not length_audit["valid"]
        or base.ACTIVE_HARD_COUNTS != expected_hard
        or base.ACTIVE_CYCLE_COUNTS != Counter({0: 3072, 1: 3072, 2: 3072, 3: 3072})
    ):
        raise ValueError("candidate scheduler failed raw/hard length or condition ledger")
    if not base.source_guards_unchanged(guards):
        raise ValueError("a candidate training source changed during runner validation")
    after = {str(Path(path).resolve()): sha256_file(Path(path)) for path in source_paths}
    if before != after:
        raise ValueError("a source hash changed during static runner validation")
    if CHECKPOINT_ROOT.exists() or any(
        path.exists() for path in (PREFLIGHT, PREFLIGHT_ATTEMPT, PREFLIGHT_BINDING, REPORT, ATTEMPT)
    ):
        raise ValueError("static validation wrote a checkpoint or training artifact")

    cpu_seconds = time.process_time() - started
    if cpu_seconds > PREFLIGHT_CPU_LIMIT_SECONDS:
        raise ValueError("static candidate runner validation exceeded its CPU ceiling")
    report = {
        "schema": "n1-length-balanced-runner-readiness-v1",
        "status": "complete",
        "complete": True,
        "plan_sha256": PLAN_SHA256,
        "schedule_sha256": SCHEDULE_SHA256,
        "checks_sha256": CHECKS_SHA256,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "source_hashes": before,
        "inputs_unchanged": True,
        "schedule_length_audit": length_audit,
        "hard_condition_counts": {
            f"{kind}/obs{obs}/d{distance}/{condition}": count
            for (kind, obs, distance, condition), count in sorted(
                base.ACTIVE_HARD_COUNTS.items(), key=lambda pair: tuple(str(v) for v in pair[0])
            )
        },
        "cycle_counts": {
            str(key): value for key, value in sorted(base.ACTIVE_CYCLE_COUNTS.items())
        },
        "steps_simulated": UPDATES_PER_MODEL,
        "base_readiness_source_count": len(readiness_sources),
        "base_code_identity": globals_dict["identity"](),
        "cpu_seconds": cpu_seconds,
        "cpu_limit_seconds": PREFLIGHT_CPU_LIMIT_SECONDS,
        "model_loads": 0,
        "checkpoint_loads": 0,
        "model_queries": 0,
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "dev_queries": 0,
        "holdout_queries": 0,
        "confirmation_queries": 0,
        "can_promote": False,
    }
    encoded = json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    with RUNNER_READINESS.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded + "\n")
    return report


def verify_runner_readiness() -> dict:
    if not RUNNER_READINESS.is_file():
        raise FileNotFoundError("complete candidate runner readiness report is required")
    report = read_json(RUNNER_READINESS)
    if (
        report.get("schema") != "n1-length-balanced-runner-readiness-v1"
        or report.get("status") != "complete"
        or report.get("complete") is not True
        or report.get("plan_sha256") != PLAN_SHA256
        or report.get("schedule_sha256") != SCHEDULE_SHA256
        or report.get("checks_sha256") != CHECKS_SHA256
        or report.get("runner_sha256") != sha256_file(Path(__file__).resolve())
        or report.get("schedule_length_audit", {}).get("valid") is not True
        or any(
            report.get(key) != 0
            for key in (
                "model_loads",
                "checkpoint_loads",
                "model_queries",
                "parameter_updates",
                "optimizer_steps",
            )
        )
    ):
        raise ValueError("candidate runner readiness report is stale or incomplete")
    for path, expected in report.get("source_hashes", {}).items():
        source = Path(path)
        if not source.is_file() or sha256_file(source) != expected:
            raise ValueError(f"candidate runner readiness source changed: {source}")
    return report


def run_checkpoint_preflight() -> dict:
    verify_runner_readiness()
    if any(path.exists() for path in (PREFLIGHT, PREFLIGHT_ATTEMPT, PREFLIGHT_BINDING)):
        raise FileExistsError("candidate checkpoint preflight is write-once")
    if CHECKPOINT_ROOT.exists():
        raise FileExistsError(f"candidate checkpoint root already exists: {CHECKPOINT_ROOT}")
    base = configure_candidate_runner()
    return base.checkpoint_preflight()


def run_training() -> dict:
    verify_runner_readiness()
    if any(path.exists() for path in (REPORT, ATTEMPT, PROGRESS)):
        raise FileExistsError("candidate training report/attempt already exists")
    if not PREFLIGHT.is_file() or not PREFLIGHT_BINDING.is_file():
        raise FileNotFoundError("successful candidate zero checkpoint preflight is required")
    base = configure_candidate_runner()
    return base.training_audit()


def run_restore_worker(path: Path) -> dict:
    base = configure_candidate_runner()
    return base._restore_worker(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--checkpoint-preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--restore-worker", type=Path)
    args = parser.parse_args()
    if args.restore_worker is not None:
        result = run_restore_worker(args.restore_worker)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    if args.validate_only:
        result = validate_candidate_runner()
        print(
            json.dumps(
                {
                    "complete": result["complete"],
                    "runner_readiness": str(RUNNER_READINESS.relative_to(ROOT)),
                    "steps_simulated": result["steps_simulated"],
                    "cpu_seconds": result["cpu_seconds"],
                    "model_loads": result["model_loads"],
                    "checkpoint_loads": result["checkpoint_loads"],
                    "parameter_updates": result["parameter_updates"],
                },
                sort_keys=True,
            )
        )
        return 0
    result = run_checkpoint_preflight() if args.checkpoint_preflight else run_training()
    print(
        json.dumps(
            {
                "complete": result.get("complete"),
                "status": result.get("status"),
                "report": str(
                    (PREFLIGHT if args.checkpoint_preflight else REPORT).relative_to(ROOT)
                ),
                "cpu_seconds": result.get("cpu_seconds"),
                "completed_updates": result.get("completed_updates", 0),
                "checkpoint_loads": result.get("checkpoint_loads"),
                "dev_queries": result.get("dev_queries", 0),
                "holdout_queries": result.get("holdout_queries", 0),
                "confirmation_queries": result.get("confirmation_queries", 0),
                "can_promote": result.get("can_promote", False),
            },
            sort_keys=True,
        )
    )
    return 0 if result.get("complete") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
