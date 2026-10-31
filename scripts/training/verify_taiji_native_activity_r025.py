"""R025 activity-dynamics receipt; binds contract, code, checks and the write-once report."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from taiji import living_core as core  # noqa: E402
from taiji import native_activity  # noqa: E402

TEST_FILE = ROOT / "tests/taiji_native/test_native_activity_r025.py"
CONTRACT = ROOT / "plans/reference/SPEC-TAIJI-CORE_ACTIVITY_R025_20261031.md"
R024_RECEIPT = ROOT / "reports/taiji_core_r024_subject_restore_20261031.json"
BASELINE_COMMIT = "6194a984a"
BUDGET_CPU_SECONDS = 900
BUDGET_WALL_SECONDS = 1200


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_checks():
    spec = importlib.util.spec_from_file_location("r025_activity_checks", TEST_FILE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def location(symbol) -> dict:
    lines, start = inspect.getsourcelines(symbol)
    return {
        "file": str(Path(inspect.getfile(symbol)).relative_to(ROOT).as_posix()),
        "symbol": symbol.__qualname__,
        "line": start,
        "last_line": start + len(lines) - 1,
    }


def verify() -> dict:
    start, cpu = time.perf_counter(), time.process_time()
    checks = load_checks()
    results = []
    journal = []
    for name in sorted(name for name in vars(checks) if name.startswith("test_r025_")):
        function = getattr(checks, name)
        try:
            function()
            result = {"id": name, "passed": True}
        except Exception as exc:  # Evidence records the actual failure, never a fake PASS.
            result = {"id": name, "passed": False, "error": f"{type(exc).__name__}: {exc}"}
        result["test"] = location(function)
        results.append(result)
    # One reference phase for the journal sample in the report.
    model = core.LivingCoreReference(checks.CONFIG).double()
    state = model.initial_state("r025-report")
    env = checks.ActivityEnvironment(store=checks.RECORDS, goal=checks.GOAL, now=10)
    policy = checks.ActivityPolicy(checks.CONFIG.width).double()
    last_event = torch.zeros(checks.CONFIG.width, dtype=torch.float64)
    error_value = 0.0
    for _ in range(5):
        state, env, entry, last_event, error_value = checks.run_round(
            model,
            state,
            env,
            policy,
            journal,
            last_event=last_event,
            prediction_error_value=error_value,
        )
    cpu_seconds, wall_seconds = time.process_time() - cpu, time.perf_counter() - start
    budget_stop = cpu_seconds > BUDGET_CPU_SECONDS or wall_seconds > BUDGET_WALL_SECONDS
    all_checks = all(row["passed"] for row in results)
    if budget_stop:
        status = "budget_stop"
    elif all_checks:
        status = "passed_r025_activity_engineering_only"
    else:
        status = "failed_" + "_".join(
            row["id"].replace("test_r025_", "") for row in results if not row["passed"]
        )
    r024 = json.loads(R024_RECEIPT.read_text(encoding="utf-8"))
    living_digest = digest(Path(core.__file__))
    return {
        "schema": "taiji-r025-activity-dynamics-v1",
        "date": "2026-10-31",
        "baseline_commit": BASELINE_COMMIT,
        "contract": CONTRACT.relative_to(ROOT).as_posix(),
        "status": status,
        "config": asdict(checks.CONFIG),
        "dtype": "torch.float64",
        "torch_version": core.torch.__version__,
        "lineage_binding": {
            "living_core_sha256": living_digest,
            "matches_r024_receipt": r024["source_hashes"]["taiji/living_core.py"] == living_digest,
            "r024_status": r024["status"],
        },
        "activities": list(native_activity.ACTIVITIES),
        "policy_head": {
            "input_dim": 2 * checks.CONFIG.width + 19,
            "hidden": 16,
            "init": "seeded 20261031",
            "trained": False,
            "selection": "deterministic argmax over masked softmax; probs journaled",
        },
        "control_arm": "fixed ladder: queue>0 -> interact; error>0.10 and store -> recall; pending>0 -> maintain; else wait",
        "checks": results,
        "reference_journal_sample": journal,
        "data_splits_read": [],
        "activity_learned_claimed": False,
        "autonomy_claimed": False,
        "task_capability_claimed": False,
        "training_run": False,
        "default_model_replaced": False,
        "cpu_seconds_checks_only": cpu_seconds,
        "wall_seconds_checks_only": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "source_hashes": {
            path.relative_to(ROOT).as_posix(): digest(path)
            for path in (
                CONTRACT,
                Path(core.__file__),
                ROOT / "taiji/native_activity.py",
                TEST_FILE,
                Path(__file__),
            )
        },
        "limits": "Activity loop, traceable model-owned selection, imagined-event isolation, "
        "wait/maintain semantics, the control-arm differentiation check and bundle-carried "
        "policy restore on one synthetic fixture. No claim that the model learned to choose, "
        "no activity-policy quality claim, no autonomy, task capability, training or pricing. "
        "World-EMA targets and learnable activity credit belong to 09.4.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("R025 receipt is write-once")
    result = verify()
    if args.output:
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False, default=str)
            stream.write("\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "checks": len(result["checks"]),
                "failed": [row["id"] for row in result["checks"] if not row["passed"]],
                "cpu_seconds": round(result["cpu_seconds_checks_only"], 2),
            }
        )
    )
    return int(result["status"] != "passed_r025_activity_engineering_only")


if __name__ == "__main__":
    raise SystemExit(main())
