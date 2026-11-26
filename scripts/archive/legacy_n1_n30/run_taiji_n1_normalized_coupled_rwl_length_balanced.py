"""PLAN-N1-118: normalized shared trust region for the coupled RWL Adam proposal."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import (  # noqa: E402
    _taiji_n1_coupled_rwl_legacy_source_loader as legacy_loader,
)
from scripts.training import (
    _taiji_n1_group_relative_adam as bounded,
)
from scripts.training import (
    _taiji_n1_normalized_coupled_rwl_adam as normalized,
)
from scripts.training import (
    run_taiji_n1_coupled_rwl_length_balanced_training as inherited,
)
from seed_platform.paths import get_external_path  # noqa: E402

PLAN = ROOT / "plans/reference/PLAN-N1-118_normalized_coupled_rwl_trust_region_draft_20261018.md"
PLAN_SHA256 = "4e7624c385256b41df2ae62de4a632292ea753c4412dea9f53ff712084e435d4"
N116_FAILURE = (
    ROOT
    / "reports/taiji_n1_coupled_rwl_length_balanced_training_20261018_checkpoint_preflight_failure.json"
)
N116_FAILURE_SHA256 = "731cedcdfe8e0ccd7be97d22cc81eefab20be6bb3f5d7d1ac78cd54f2d20914e"
N110_REPORT = ROOT / "reports/taiji_n1_length_balanced_training_20261018.json"
N110_REPORT_SHA256 = "e848433de597d378e9e1dc52bf488f296f90c5ce67f3855e19bcfe49268aaa0c"
N117_REPORT = ROOT / "reports/taiji_n1_coupled_rwl_length_balanced_r1_20261018.json"
N117_REPORT_SHA256 = "1d4f43519790a1e36cb7dfbc580506d697815dccbdecb95fd576ec3a7bc2e391"
N117_ATTRIBUTION = ROOT / "reports/taiji_n1_coupled_rwl_trajectory_attribution_20261018.json"
N117_ATTRIBUTION_SHA256 = "a6de24cb407bab157991084f1c9914628db9267fffa7155db68cfc0e5ab784f6"
REPORT = ROOT / "reports/taiji_n1_normalized_coupled_rwl_length_balanced_20261018.json"
ATTEMPT = REPORT.with_name(REPORT.stem + "_attempt.json")
PROGRESS = REPORT.with_name(REPORT.stem + "_progress.jsonl")
PREFLIGHT = REPORT.with_name(REPORT.stem + "_checkpoint_preflight.json")
PREFLIGHT_ATTEMPT = REPORT.with_name(REPORT.stem + "_checkpoint_preflight_attempt.json")
PREFLIGHT_BINDING = REPORT.with_name(REPORT.stem + "_checkpoint_preflight_binding.json")
RUNNER_READINESS = REPORT.with_name(REPORT.stem + "_runner_readiness.json")
RUNNER_SOURCE_BINDING = REPORT.with_name(REPORT.stem + "_runner_source_binding.json")
CONTRACT_TEST = ROOT / "tests/taiji_native/test_n1_plan118_normalized_coupled_rwl_contract.py"
CHECKPOINT_ROOT = (
    Path(get_external_path("data/checkpoints"))
    / "n1_normalized_coupled_rwl_length_balanced_20261018"
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


def _position_shard_paths(report_path: Path) -> list[Path]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    result = []
    for seed in (11, 12, 13):
        models = [
            model
            for model in report["models"]
            if (model.get("seed"), model.get("arm")) == (seed, "position")
        ]
        if len(models) != 1 or models[0].get("complete") is not True:
            raise ValueError(f"one complete seed{seed}/position source required: {report_path}")
        shard = models[0]["shard"]
        path = Path(shard["path"])
        if sha256(path) != shard["compressed_sha256"]:
            raise ValueError(f"position source shard changed: {path}")
        result.append(path)
    return result


def _validate_evidence() -> list[Path]:
    expected = {
        N116_FAILURE: N116_FAILURE_SHA256,
        N110_REPORT: N110_REPORT_SHA256,
        N117_REPORT: N117_REPORT_SHA256,
        N117_ATTRIBUTION: N117_ATTRIBUTION_SHA256,
    }
    for path, digest in expected.items():
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"frozen N1-118 evidence hash mismatch: {path}")
    n110 = json.loads(N110_REPORT.read_text(encoding="utf-8"))
    n117 = json.loads(N117_REPORT.read_text(encoding="utf-8"))
    attribution = json.loads(N117_ATTRIBUTION.read_text(encoding="utf-8"))
    if (
        n110.get("complete") is not True
        or n117.get("complete") is not True
        or n110.get("schedule_sha256") != n117.get("schedule_sha256")
        or attribution.get("status") != "complete"
        or attribution.get("source_shards_read") != 6
        or attribution.get("model_queries") != 0
    ):
        raise ValueError("matched N1-110/N1-117 train-only evidence required")
    return [*_position_shard_paths(N110_REPORT), *_position_shard_paths(N117_REPORT)]


def configure() -> None:
    global _CONFIGURED, _BASE
    if _CONFIGURED:
        return
    if sha256(PLAN) != PLAN_SHA256:
        raise ValueError("frozen PLAN-N1-118 SHA mismatch")
    if not CHECKPOINT_ROOT.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("PLAN-N1-118 checkpoint root escaped workspace")
    evidence_shards = _validate_evidence()

    inherited.PLAN = PLAN
    inherited.PLAN_SHA256 = PLAN_SHA256
    inherited.coupled = normalized
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
            return legacy_loader.load_source(payload, original_load, bounded.POLICY, LEGACY_POLICY)

        globals_dict["engineering"].load = load_source
        return loaded, globals_dict, inputs

    def source_paths():
        paths = list(original_source_paths())
        paths.extend(
            (
                PLAN,
                N116_FAILURE,
                N110_REPORT,
                N117_REPORT,
                N117_ATTRIBUTION,
                *evidence_shards,
                Path(__file__).resolve(),
                Path(normalized.__file__).resolve(),
                Path(legacy_loader.__file__).resolve(),
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
