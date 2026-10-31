"""R026 plasticity-arms receipt; binds the preregistered comparison to its report."""

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

from taiji import growth_credit as growth_credit_module  # noqa: E402
from taiji import living_core as core  # noqa: E402
from taiji import native_development as development  # noqa: E402

TEST_FILE = ROOT / "tests/taiji_native/test_plasticity_arms_r026.py"
CONTRACT = ROOT / "plans/reference/SPEC-TAIJI-CORE_PLASTICITY_R026_20261031.md"
R025_RECEIPT = ROOT / "reports/taiji_core_r025_activity_dynamics_20261031.json"
BASELINE_COMMIT = "4cfc3125e"
BUDGET_CPU_SECONDS = 900
BUDGET_WALL_SECONDS = 1200


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_checks():
    spec = importlib.util.spec_from_file_location("r026_plasticity_checks", TEST_FILE)
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
        for name in sorted(name for name in vars(checks) if name.startswith("test_r026_")):
            function = getattr(checks, name)
            try:
                function()
                result = {"id": name, "passed": True}
            except Exception as exc:  # Evidence records the actual failure, never a fake PASS.
                result = {"id": name, "passed": False, "error": f"{type(exc).__name__}: {exc}"}
            result["test"] = location(function)
            results.append(result)
        data = checks.comparison()
    finally:
        torch.autograd.grad = original_grad
        torch.Tensor.backward = original_backward
    cpu_seconds, wall_seconds = time.process_time() - cpu, time.perf_counter() - start
    budget_stop = cpu_seconds > BUDGET_CPU_SECONDS or wall_seconds > BUDGET_WALL_SECONDS
    all_checks = all(row["passed"] for row in results)
    execution_failures = [
        row["arm"]
        for row in data["ordered"]
        if any(value != value for value in (row["delta_nll_holdout"], row["drift_retention"]))
    ]
    if budget_stop:
        status = "budget_stop"
    elif not all_checks or execution_failures:
        status = "failed_arm_execution"
    else:
        status = "passed_r026_preregistered_comparison"
    r025 = json.loads(R025_RECEIPT.read_text(encoding="utf-8"))
    living_digest = digest(Path(core.__file__))
    diagnostics = development.credit_diagnostics(
        data["parent"],
        {
            n: growth_credit_module.branch_credit(data["book"], b)
            for b, n in enumerate(data["book"].branch_ids)
        },
        {
            n: growth_credit_module.branch_credit(
                data["book"], b, range(len(data["book"].deltas) // 2)
            )
            for b, n in enumerate(data["book"].branch_ids)
        },
    )
    return {
        "schema": "taiji-r026-plasticity-arms-v1",
        "date": "2026-10-31",
        "baseline_commit": BASELINE_COMMIT,
        "contract": CONTRACT.relative_to(ROOT).as_posix(),
        "status": status,
        "config": asdict(checks.CONFIG),
        "dtype": "torch.float64",
        "torch_version": core.torch.__version__,
        "lineage_binding": {
            "living_core_sha256": living_digest,
            "matches_r025_receipt": r025["source_hashes"]["taiji/living_core.py"] == living_digest,
            "r025_status": r025["status"],
        },
        "preregistration": {
            "endpoint": "delta_nll_holdout = NLL_holdout(post) - NLL_holdout(parent), "
            "teacher-forced from the canonical initial state",
            "retention_probe": "third declared career, tolerance 0.05",
            "net_margin": development.NET_MARGIN,
            "update_schedule": "8 Adam(lr=1e-3) steps per arm on the replay branch, identical arms",
        },
        "arms": data["ordered"],
        "adjudication": data["adjudication"],
        "preregistered_verdict": data["adjudication"]["verdict"],
        "credit_diagnostics": diagnostics,
        "checks": results,
        "gradient_calls": grad_counter["grad"],
        "backward_calls": grad_counter["grad"] + grad_counter["backward"],
        "data_splits_read": [],
        "task_capability_claimed": False,
        "real_world_benefit_claimed": False,
        "training_admitted_afterwards": False,
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
                ROOT / "taiji/native_development.py",
                TEST_FILE,
                Path(__file__),
            )
        },
        "limits": "Six-arm preregistered comparison on one synthetic parent with the declared "
        "update expression. A positive verdict does NOT sign task capability, real-world "
        "benefit or training admission; a negative verdict does not falsify the v1.2 "
        "candidate under real conditions. Retention breaches are recorded verbatim.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("R026 receipt is write-once")
    result = verify()
    if args.output:
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False, default=str)
            stream.write("\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "verdict": result["preregistered_verdict"],
                "retention_breaches": result["adjudication"]["retention_breaches"],
                "failed_checks": [row["id"] for row in result["checks"] if not row["passed"]],
                "cpu_seconds": round(result["cpu_seconds_checks_only"], 2),
            }
        )
    )
    return int(result["status"] != "passed_r026_preregistered_comparison")


if __name__ == "__main__":
    raise SystemExit(main())
