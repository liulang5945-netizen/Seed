"""R027 subject-gates receipt; binds the 09.5 checklist consolidation to its report."""

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


from taiji import living_core as core  # noqa: E402

TEST_FILE = ROOT / "tests/taiji_native/test_subject_gates_r027.py"
CONTRACT = ROOT / "plans/reference/SPEC-TAIJI-CORE_GATES_R027_20261031.md"
R026_RECEIPT = ROOT / "reports/taiji_core_r026_plasticity_arms_20261031.json"
BASELINE_COMMIT = "a237e6dac"
BUDGET_CPU_SECONDS = 900
BUDGET_WALL_SECONDS = 1200

CHECKLIST = [
    "parameters and topology (R/P shapes with growth structure) carried and restored",
    "optimizer state and eligibility stand-in carried across every boundary",
    "RNG state carried (probe drawn after restore matches the parent)",
    "cursors: career journal_cursor plus monotonic activity/ledger seq",
    "memory: store records serialized per byte, illegal sources filtered with provenance",
    "encoder revision: migration commit advances revision and rides the bundle",
    "budget: fee_total/probe_exposures continue and are never cleared",
    "receipt: pending-outcome resolution idempotent against the current journal",
    "no unauthorized or wrong-source records enter memory or updates",
    "no ledger clearing, no replay double-claim, no external-action re-send",
    "query-only probes run on detached copies and leave hidden/cache bitwise",
    "external triggers and endogenous policy choices accounted separately",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_checks():
    spec = importlib.util.spec_from_file_location("r027_gates_checks", TEST_FILE)
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
    for name in sorted(name for name in vars(checks) if name.startswith("test_r027_")):
        function = getattr(checks, name)
        try:
            function()
            result = {"id": name, "passed": True}
        except Exception as exc:  # Evidence records the actual failure, never a fake PASS.
            result = {"id": name, "passed": False, "error": f"{type(exc).__name__}: {exc}"}
        result["test"] = location(function)
        results.append(result)
    cpu_seconds, wall_seconds = time.process_time() - cpu, time.perf_counter() - start
    budget_stop = cpu_seconds > BUDGET_CPU_SECONDS or wall_seconds > BUDGET_WALL_SECONDS
    all_checks = all(row["passed"] for row in results)
    if budget_stop:
        status = "budget_stop"
    elif all_checks:
        status = "passed_r027_gates_engineering_only"
    else:
        status = "failed_" + "_".join(
            row["id"].replace("test_r027_", "") for row in results if not row["passed"]
        )
    r026 = json.loads(R026_RECEIPT.read_text(encoding="utf-8"))
    living_digest = digest(Path(core.__file__))
    return {
        "schema": "taiji-r027-subject-gates-v1",
        "date": "2026-11-01",
        "baseline_commit": BASELINE_COMMIT,
        "contract": CONTRACT.relative_to(ROOT).as_posix(),
        "status": status,
        "config": asdict(checks.CONFIG),
        "dtype": "torch.float64",
        "torch_version": core.torch.__version__,
        "lineage_binding": {
            "living_core_sha256": living_digest,
            "matches_r026_receipt": r026["source_hashes"]["taiji/living_core.py"] == living_digest,
            "r026_status": r026["status"],
        },
        "boundaries": [
            "after_interaction (host-injected message consumed by a host-driven round)",
            "no_prompt_activity (two endogenous model rounds, empty queue)",
            "sleep_wait (forced wait: no forward step, clock advance, H bitwise)",
            "first_learning (declared update; worker repeats it bitwise)",
            "before_birth / after_birth (R023 gated birth with the carried gate table)",
            "after_migration (encoder revision+1 committed on the gated child)",
            "pending_outcome (idempotent resolution against the current journal)",
        ],
        "checklist": CHECKLIST,
        "checks": results,
        "worker_modes": ["microstep", "update", "resolve", "activity_round"],
        "data_splits_read": [],
        "semantic_preservation_claimed": False,
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
                ROOT / "taiji/state_migration.py",
                TEST_FILE,
                Path(__file__),
            )
        },
        "limits": "Gate consolidation of the R024 bundle and R025 activity loop on one "
        "synthetic life sequence. Passing does NOT sign semantic preservation, autonomy, "
        "task capability, training or pricing admission.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("R027 receipt is write-once")
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
    return int(result["status"] != "passed_r027_gates_engineering_only")


if __name__ == "__main__":
    raise SystemExit(main())
