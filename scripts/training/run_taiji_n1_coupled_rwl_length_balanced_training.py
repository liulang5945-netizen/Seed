"""PLAN-N1-116: length-balanced training with one shared reader/writer/local trust region."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import (  # noqa: E402
    _taiji_n1_coupled_rwl_group_adam as coupled,
)
from scripts.training import (
    _taiji_n1_group_relative_adam as base_bounded,
)
from scripts.training import (
    run_taiji_n1_length_balanced_training as length_balanced,
)
from seed_platform.paths import get_external_path  # noqa: E402

PLAN = ROOT / "plans/reference/PLAN-N1-116_coupled_rwl_trust_region_training_draft_20261018.md"
PLAN_SHA256 = "d97b340a18fc68188d0e43b67c63e0d325d1dbb314ebaa14bb2621224b65e721"
REPORT = ROOT / "reports/taiji_n1_coupled_rwl_length_balanced_training_20261018.json"
ATTEMPT = REPORT.with_name(REPORT.stem + "_attempt.json")
PROGRESS = REPORT.with_name(REPORT.stem + "_progress.jsonl")
PREFLIGHT = REPORT.with_name(REPORT.stem + "_checkpoint_preflight.json")
PREFLIGHT_ATTEMPT = REPORT.with_name(REPORT.stem + "_checkpoint_preflight_attempt.json")
PREFLIGHT_BINDING = REPORT.with_name(REPORT.stem + "_checkpoint_preflight_binding.json")
RUNNER_READINESS = REPORT.with_name(REPORT.stem + "_runner_readiness.json")
RUNNER_SOURCE_BINDING = REPORT.with_name(REPORT.stem + "_runner_source_binding.json")
CONTRACT_TEST = ROOT / "tests/taiji_native/test_n1_plan116_coupled_rwl_adam_contract.py"
CHECKPOINT_ROOT = (
    Path(get_external_path("data/checkpoints")) / "n1_coupled_rwl_length_balanced_20261018"
).resolve()
CPU_LIMIT_SECONDS = 18_000
PREFLIGHT_CPU_LIMIT_SECONDS = 300

_BASE = None
_CONFIGURED = False
_ORIGINAL_POLICY = dict(base_bounded.POLICY)
_ORIGINAL_BOUNDED_STEP = base_bounded.bounded_step


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path: Path, value: dict) -> None:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded + "\n")


def _source_paths(existing) -> list[Path]:
    paths = list(existing())
    paths.extend((PLAN, Path(__file__).resolve(), Path(coupled.__file__).resolve(), CONTRACT_TEST))
    if RUNNER_SOURCE_BINDING.is_file():
        paths.append(RUNNER_SOURCE_BINDING)
    return list(dict.fromkeys(Path(path).resolve() for path in paths))


def _restore_worker_command(checkpoint: Path, *, python: Path | None = None) -> list[str]:
    interpreter = Path(sys.executable if python is None else python)
    if interpreter.name.lower() == "pythonw.exe":
        interpreter = interpreter.with_name("python.exe")
    return [str(interpreter), str(Path(__file__).resolve()), "--restore-worker", str(checkpoint)]


def configure() -> None:
    global _BASE, _CONFIGURED
    if _CONFIGURED:
        return
    if sha256(PLAN) != PLAN_SHA256:
        raise ValueError("frozen PLAN-N1-116 plan hash mismatch")
    if not CHECKPOINT_ROOT.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("PLAN-N1-116 checkpoint root escaped the workspace")
    # The frozen N1-110 machinery keeps its data, schedule and recovery protocol.
    # Only redirect its write-once outputs and replace the post-Adam projection.
    length_balanced.REPORT = REPORT
    length_balanced.ATTEMPT = ATTEMPT
    length_balanced.PROGRESS = PROGRESS
    length_balanced.PREFLIGHT = PREFLIGHT
    length_balanced.PREFLIGHT_ATTEMPT = PREFLIGHT_ATTEMPT
    length_balanced.PREFLIGHT_BINDING = PREFLIGHT_BINDING
    length_balanced.RUNNER_READINESS = RUNNER_READINESS
    length_balanced.CHECKPOINT_ROOT = CHECKPOINT_ROOT
    length_balanced.CPU_LIMIT_SECONDS = CPU_LIMIT_SECONDS
    length_balanced.PREFLIGHT_CPU_LIMIT_SECONDS = PREFLIGHT_CPU_LIMIT_SECONDS
    length_balanced.restore_worker_command = _restore_worker_command

    base_bounded.POLICY.clear()
    base_bounded.POLICY.update(coupled.POLICY)
    base_bounded.bounded_step = coupled.bounded_step

    _BASE = length_balanced.configure_candidate_runner()
    original_protocol = _BASE.protocol
    original_source_paths = _BASE.source_paths
    original_make_runner = _BASE.make_runner

    def protocol(base_protocol=None):
        value = original_protocol(base_protocol)
        value.update(
            schema="n1-coupled-rwl-length-balanced-training-v1",
            optimizer_update_policy=dict(coupled.POLICY),
            coupled_update_plan_sha256=PLAN_SHA256,
            coupled_update_contract_test_sha256=sha256(CONTRACT_TEST),
            checkpoint_root_relative=CHECKPOINT_ROOT.relative_to(ROOT).as_posix(),
            cpu_limit_seconds=CPU_LIMIT_SECONDS,
        )
        return value

    _BASE.protocol = protocol
    _BASE.PLAN = PLAN
    _BASE.CHECKPOINT_ROOT = CHECKPOINT_ROOT
    _BASE.REPORT = REPORT
    _BASE.ATTEMPT = ATTEMPT
    _BASE.PROGRESS = PROGRESS
    _BASE.PREFLIGHT = PREFLIGHT
    _BASE.PREFLIGHT_ATTEMPT = PREFLIGHT_ATTEMPT
    _BASE.PREFLIGHT_BINDING = PREFLIGHT_BINDING
    _BASE.CPU_LIMIT_SECONDS = CPU_LIMIT_SECONDS
    _BASE.PREFLIGHT_CPU_LIMIT_SECONDS = PREFLIGHT_CPU_LIMIT_SECONDS
    _BASE.source_paths = lambda: _source_paths(original_source_paths)

    def make_runner_with_preflight_policy_bridge(*, load_inputs: bool):
        loaded, globals_dict, inputs = original_make_runner(load_inputs=load_inputs)
        original_readiness = globals_dict.get("readiness")
        if original_readiness is not None:

            def readiness_with_original_optimizer_policy():
                current_policy = dict(base_bounded.POLICY)
                current_step = base_bounded.bounded_step
                base_bounded.POLICY.clear()
                base_bounded.POLICY.update(_ORIGINAL_POLICY)
                base_bounded.bounded_step = _ORIGINAL_BOUNDED_STEP
                try:
                    return original_readiness()
                finally:
                    base_bounded.POLICY.clear()
                    base_bounded.POLICY.update(current_policy)
                    base_bounded.bounded_step = current_step

            globals_dict["readiness"] = readiness_with_original_optimizer_policy
        return loaded, globals_dict, inputs

    _BASE.make_runner = make_runner_with_preflight_policy_bridge
    _CONFIGURED = True


def validate_only() -> dict:
    configure()
    result = (
        length_balanced.verify_runner_readiness()
        if RUNNER_READINESS.is_file()
        else length_balanced.validate_candidate_runner()
    )
    if result.get("complete") is not True or any(
        result.get(key) != 0
        for key in (
            "model_loads",
            "checkpoint_loads",
            "model_queries",
            "parameter_updates",
            "optimizer_steps",
        )
    ):
        raise ValueError("zero-query PLAN-N1-116 runner readiness required")
    if RUNNER_SOURCE_BINDING.exists():
        raise FileExistsError("PLAN-N1-116 runner source binding is write-once")
    source_hashes = {
        str(path): sha256(path) for path in _BASE.source_paths() if Path(path).is_file()
    }
    if not source_hashes or sha256(PLAN) != PLAN_SHA256:
        raise ValueError("PLAN-N1-116 source binding is incomplete")
    binding = {
        "schema": "n1-coupled-rwl-runner-source-binding-v1",
        "status": "complete",
        "plan_sha256": PLAN_SHA256,
        "runner_readiness_sha256": sha256(RUNNER_READINESS),
        "source_hashes": source_hashes,
        "model_loads": 0,
        "checkpoint_loads": 0,
        "model_queries": 0,
        "parameter_updates": 0,
        "optimizer_steps": 0,
    }
    write_new(RUNNER_SOURCE_BINDING, binding)
    result["runner_source_binding"] = str(RUNNER_SOURCE_BINDING.relative_to(ROOT))
    return result


def verify_runner_source_binding() -> dict:
    if not RUNNER_SOURCE_BINDING.is_file():
        raise FileNotFoundError("complete PLAN-N1-116 source binding is required")
    binding = json.loads(RUNNER_SOURCE_BINDING.read_text(encoding="utf-8"))
    if (
        binding.get("schema") != "n1-coupled-rwl-runner-source-binding-v1"
        or binding.get("status") != "complete"
        or binding.get("plan_sha256") != PLAN_SHA256
        or binding.get("runner_readiness_sha256") != sha256(RUNNER_READINESS)
        or any(
            binding.get(key) != 0
            for key in (
                "model_loads",
                "checkpoint_loads",
                "model_queries",
                "parameter_updates",
                "optimizer_steps",
            )
        )
    ):
        raise ValueError("PLAN-N1-116 source binding is stale or incomplete")
    for raw_path, digest in binding.get("source_hashes", {}).items():
        path = Path(raw_path)
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"PLAN-N1-116 source changed after runner validation: {path}")
    return binding


def checkpoint_preflight() -> dict:
    configure()
    verify_runner_source_binding()
    return length_balanced.run_checkpoint_preflight()


def execute() -> dict:
    configure()
    verify_runner_source_binding()
    return length_balanced.run_training()


def restore_worker(path: Path) -> dict:
    configure()
    verify_runner_source_binding()
    return length_balanced.run_restore_worker(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--checkpoint-preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--restore-worker", type=Path)
    args = parser.parse_args()
    if args.validate_only:
        result = validate_only()
        output = {
            "complete": result.get("complete"),
            "steps_simulated": result.get("steps_simulated"),
            "cpu_seconds": result.get("cpu_seconds"),
            "model_loads": result.get("model_loads"),
            "checkpoint_loads": result.get("checkpoint_loads"),
            "parameter_updates": result.get("parameter_updates"),
            "readiness": str(RUNNER_READINESS.relative_to(ROOT)),
        }
    elif args.restore_worker is not None:
        result = restore_worker(args.restore_worker)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    elif args.checkpoint_preflight:
        result = checkpoint_preflight()
        output = {
            key: result.get(key)
            for key in (
                "complete",
                "status",
                "cpu_seconds",
                "checkpoint_loads",
                "independent_restores",
            )
        }
        output["report"] = str(PREFLIGHT.relative_to(ROOT))
    else:
        result = execute()
        output = {
            key: result.get(key)
            for key in (
                "complete",
                "status",
                "cpu_seconds",
                "completed_updates",
                "training_sufficient",
                "checkpoint_loads",
                "dev_queries",
                "holdout_queries",
                "confirmation_queries",
                "can_promote",
            )
        }
        output["report"] = str(REPORT.relative_to(ROOT))
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0 if result.get("complete") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
