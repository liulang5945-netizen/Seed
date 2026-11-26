"""PLAN-N1-117: retry the coupled RWL protocol with a legacy-zero loader bridge."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import (
    _taiji_n1_coupled_rwl_group_adam as coupled,
)
from scripts.training import (  # noqa: E402
    _taiji_n1_coupled_rwl_legacy_source_loader as legacy_loader,
)
from scripts.training import (
    _taiji_n1_group_relative_adam as bounded,
)
from scripts.training import (
    run_taiji_n1_coupled_rwl_length_balanced_training as inherited,
)
from seed_platform.paths import get_external_path  # noqa: E402

PLAN = ROOT / "plans/reference/PLAN-N1-117_coupled_rwl_source_loader_revision_draft_20261018.md"
PLAN_SHA256 = "07d32a1f9056198b81b2a6fb55ad3f4dda23f5512fde169997c17eaad429932c"
FAILURE_REPORT = (
    ROOT
    / "reports/taiji_n1_coupled_rwl_length_balanced_training_20261018_checkpoint_preflight_failure.json"
)
FAILURE_SHA256 = "731cedcdfe8e0ccd7be97d22cc81eefab20be6bb3f5d7d1ac78cd54f2d20914e"
REPORT = ROOT / "reports/taiji_n1_coupled_rwl_length_balanced_r1_20261018.json"
ATTEMPT = REPORT.with_name(REPORT.stem + "_attempt.json")
PROGRESS = REPORT.with_name(REPORT.stem + "_progress.jsonl")
PREFLIGHT = REPORT.with_name(REPORT.stem + "_checkpoint_preflight.json")
PREFLIGHT_ATTEMPT = REPORT.with_name(REPORT.stem + "_checkpoint_preflight_attempt.json")
PREFLIGHT_BINDING = REPORT.with_name(REPORT.stem + "_checkpoint_preflight_binding.json")
RUNNER_READINESS = REPORT.with_name(REPORT.stem + "_runner_readiness.json")
RUNNER_SOURCE_BINDING = REPORT.with_name(REPORT.stem + "_runner_source_binding.json")
CONTRACT_TEST = ROOT / "tests/taiji_native/test_n1_plan117_legacy_policy_bridge_contract.py"
CHECKPOINT_ROOT = (
    Path(get_external_path("data/checkpoints")) / "n1_coupled_rwl_length_balanced_r1_20261018"
).resolve()
CPU_LIMIT_SECONDS = 18_000
PREFLIGHT_CPU_LIMIT_SECONDS = 300
LEGACY_POLICY = dict(inherited._ORIGINAL_POLICY)
_CONFIGURED = False
_BASE = None


def sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _restore_worker_command(checkpoint: Path, *, python: Path | None = None) -> list[str]:
    interpreter = Path(sys.executable if python is None else python)
    if interpreter.name.lower() == "pythonw.exe":
        interpreter = interpreter.with_name("python.exe")
    return [str(interpreter), str(Path(__file__).resolve()), "--restore-worker", str(checkpoint)]


def configure() -> None:
    global _CONFIGURED, _BASE
    if _CONFIGURED:
        return
    if sha256(PLAN) != PLAN_SHA256 or sha256(FAILURE_REPORT) != FAILURE_SHA256:
        raise ValueError("frozen PLAN-N1-117 or failure evidence hash mismatch")
    if not CHECKPOINT_ROOT.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("PLAN-N1-117 checkpoint root escaped the workspace")

    inherited.PLAN = PLAN
    inherited.PLAN_SHA256 = PLAN_SHA256
    inherited.REPORT = REPORT
    inherited.ATTEMPT = ATTEMPT
    inherited.PROGRESS = PROGRESS
    inherited.PREFLIGHT = PREFLIGHT
    inherited.PREFLIGHT_ATTEMPT = PREFLIGHT_ATTEMPT
    inherited.PREFLIGHT_BINDING = PREFLIGHT_BINDING
    inherited.RUNNER_READINESS = RUNNER_READINESS
    inherited.RUNNER_SOURCE_BINDING = RUNNER_SOURCE_BINDING
    inherited.CONTRACT_TEST = CONTRACT_TEST
    inherited.CHECKPOINT_ROOT = CHECKPOINT_ROOT
    inherited.CPU_LIMIT_SECONDS = CPU_LIMIT_SECONDS
    inherited.PREFLIGHT_CPU_LIMIT_SECONDS = PREFLIGHT_CPU_LIMIT_SECONDS
    inherited._CONFIGURED = False
    inherited._BASE = None
    inherited._restore_worker_command = _restore_worker_command
    inherited.configure()

    _BASE = inherited._BASE
    original_make_runner = _BASE.make_runner
    original_source_paths = _BASE.source_paths

    def make_runner_with_legacy_source_adapter(*, load_inputs: bool):
        loaded, globals_dict, inputs = original_make_runner(load_inputs=load_inputs)
        original_load = globals_dict["engineering"].load

        def load_source(payload):
            return legacy_loader.load_source(
                payload,
                original_load,
                bounded.POLICY,
                LEGACY_POLICY,
            )

        globals_dict["engineering"].load = load_source
        return loaded, globals_dict, inputs

    def source_paths():
        paths = list(original_source_paths())
        paths.extend(
            (
                PLAN,
                FAILURE_REPORT,
                Path(__file__).resolve(),
                Path(legacy_loader.__file__).resolve(),
                Path(coupled.__file__).resolve(),
                CONTRACT_TEST,
            )
        )
        return list(dict.fromkeys(Path(path).resolve() for path in paths))

    _BASE.make_runner = make_runner_with_legacy_source_adapter
    _BASE.source_paths = source_paths
    _CONFIGURED = True


def validate_only() -> dict:
    configure()
    return inherited.validate_only()


def checkpoint_preflight() -> dict:
    configure()
    return inherited.checkpoint_preflight()


def execute() -> dict:
    configure()
    return inherited.execute()


def restore_worker(path: Path) -> dict:
    configure()
    return inherited.restore_worker(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--checkpoint-preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--restore-worker", type=Path)
    args = parser.parse_args()
    if args.restore_worker is not None:
        result = restore_worker(args.restore_worker)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    if args.validate_only:
        result = validate_only()
    elif args.checkpoint_preflight:
        result = checkpoint_preflight()
    else:
        result = execute()
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "complete",
                    "status",
                    "cpu_seconds",
                    "steps_simulated",
                    "completed_updates",
                    "training_sufficient",
                    "checkpoint_loads",
                    "independent_restores",
                    "model_queries",
                    "parameter_updates",
                    "dev_queries",
                    "holdout_queries",
                    "confirmation_queries",
                    "can_promote",
                )
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if result.get("complete") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
