"""R024 subject-restore receipt; binds contract, code, checks and the write-once report."""

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

TEST_FILE = ROOT / "tests/taiji_native/test_subject_restore_r024.py"
CONTRACT = ROOT / "plans/reference/SPEC-TAIJI-CORE_RESTORE_R024_20261031.md"
R023_RECEIPT = ROOT / "reports/taiji_core_r023_growth_birth_20261031.json"
BASELINE_COMMIT = "5137597a1"
BUDGET_CPU_SECONDS = 900
BUDGET_WALL_SECONDS = 1200


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_checks():
    spec = importlib.util.spec_from_file_location("r024_restore_checks", TEST_FILE)
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
    grad_counter = {"grad": 0, "backward": 0}
    original_grad, original_backward = torch.autograd.grad, torch.Tensor.backward

    def counting_grad(*args, **kwargs):
        grad_counter["grad"] += 1
        return original_grad(*args, **kwargs)

    def counting_backward(self, *args, **kwargs):
        grad_counter["backward"] += 1
        return original_backward(self, *args, **kwargs)

    torch.autograd.grad = counting_grad
    torch.Tensor.backward = counting_backward
    try:
        checks = load_checks()
        results = []
        for name in sorted(name for name in vars(checks) if name.startswith("test_r024_")):
            function = getattr(checks, name)
            try:
                function()
                result = {"id": name, "passed": True}
            except Exception as exc:  # Evidence records the actual failure, never a fake PASS.
                result = {"id": name, "passed": False, "error": f"{type(exc).__name__}: {exc}"}
            result["test"] = location(function)
            results.append(result)
    finally:
        torch.autograd.grad = original_grad
        torch.Tensor.backward = original_backward
    cpu_seconds, wall_seconds = time.process_time() - cpu, time.perf_counter() - start
    budget_stop = cpu_seconds > BUDGET_CPU_SECONDS or wall_seconds > BUDGET_WALL_SECONDS
    all_checks = all(row["passed"] for row in results)
    if budget_stop:
        status = "budget_stop"
    elif all_checks:
        status = "passed_r024_restore_engineering_only"
    else:
        status = "failed_" + "_".join(
            row["id"].replace("test_r024_", "") for row in results if not row["passed"]
        )
    r023 = json.loads(R023_RECEIPT.read_text(encoding="utf-8"))
    living_digest = digest(Path(core.__file__))
    return {
        "schema": "taiji-r024-subject-restore-v1",
        "date": "2026-10-31",
        "baseline_commit": BASELINE_COMMIT,
        "contract": CONTRACT.relative_to(ROOT).as_posix(),
        "status": status,
        "config": asdict(checks.CONFIG),
        "dtype": "torch.float64",
        "torch_version": core.torch.__version__,
        "lineage_binding": {
            "living_core_sha256": living_digest,
            "matches_r023_receipt": r023["source_hashes"]["taiji/living_core.py"] == living_digest,
            "r023_status": r023["status"],
        },
        "bundle_schema": "taiji-subject-bundle-r024-v1 (weights_only loadable)",
        "boundaries": [
            "no-input idle (cursor at end, nothing consumed, nothing mutated)",
            "after first declared learning update (next update and microstep bitwise)",
            "after gated birth (gated continuation bitwise via carried gate table)",
            "after encoder migration commit (revision+1, first declared update bitwise)",
            "pending outcome (idempotent resolution against the current journal)",
        ],
        "checks": results,
        "worker": {
            "job": "taiji/state_migration.py:run_worker_job",
            "cli": "scripts/training/r024_restore_worker.py",
            "independent_process": "multiprocessing spawn, fresh interpreter per boundary",
            "transport": "file-based (tensors do not cross process queues here)",
        },
        "declared_instruments": {
            "update": "one Adam(lr=1e-2, betas=(0.9,0.999), eps=1e-8) step on the "
            "teacher-forced replay branch NLL; synthetic expression, not training",
            "eligibility": "e <- 0.99*e + mean_i ||x_i||_2/sqrt(p) per consumed step; "
            "recorded quantity, no forward change; real rules belong to 09.4",
        },
        "gradient_calls": grad_counter["grad"],
        "backward_calls": grad_counter["grad"] + grad_counter["backward"],
        "optimizer_steps_in_checks": 3,
        "data_splits_read": [],
        "semantic_preservation_claimed": False,
        "task_capability_claimed": False,
        "full_lifecycle_claimed": False,
        "default_model_replaced": False,
        "cpu_seconds_checks_only": cpu_seconds,
        "wall_seconds_checks_only": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "source_hashes": {
            path.relative_to(ROOT).as_posix(): digest(path)
            for path in (
                CONTRACT,
                Path(core.__file__),
                ROOT / "taiji/growth_credit.py",
                ROOT / "taiji/state_migration.py",
                TEST_FILE,
                ROOT / "scripts/training/r024_restore_worker.py",
                Path(__file__),
            )
        },
        "limits": "Bundle save/restore, fresh-process bitwise continuation, RNG/eligibility/fee "
        "continuity, the declared update expression and resolution idempotence on one synthetic "
        "fixture. Migration-restore passing does NOT sign semantic preservation, old-capability "
        "retention, task capability or subject autonomy. Pricing/bounded-comparison contract "
        "decisions come after 09.2.1-09.2.4 are all closed.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("R024 receipt is write-once")
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
    return int(result["status"] != "passed_r024_restore_engineering_only")


if __name__ == "__main__":
    raise SystemExit(main())
