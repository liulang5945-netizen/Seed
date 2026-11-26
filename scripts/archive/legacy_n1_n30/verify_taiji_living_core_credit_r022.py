"""R022 credit validation receipt; binds contract, code, checks and the write-once report."""

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

TEST_FILE = ROOT / "tests/taiji_native/test_living_core_credit_r022.py"
CONTRACT = ROOT / "plans/reference/SPEC-TAIJI-CORE_CREDIT_R022_20261031.md"
R021_RECEIPT = ROOT / "reports/taiji_core_reference_r021_forward_coverage_20261031.json"
BASELINE_COMMIT = "e0d788a0f"
BUDGET_CPU_SECONDS = 900
BUDGET_WALL_SECONDS = 1200


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_checks():
    spec = importlib.util.spec_from_file_location("r022_credit_checks", TEST_FILE)
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
    for name in sorted(name for name in vars(checks) if name.startswith("test_r022_")):
        function = getattr(checks, name)
        try:
            function()
            result = {"id": name, "passed": True}
        except Exception as exc:  # Evidence records the actual failure, never a fake PASS.
            result = {"id": name, "passed": False, "error": f"{type(exc).__name__}: {exc}"}
        result["test"] = location(function)
        results.append(result)
    payload = checks.compute_all()
    gate = payload["gate"]
    cpu_seconds, wall_seconds = time.process_time() - cpu, time.perf_counter() - start
    budget_stop = cpu_seconds > BUDGET_CPU_SECONDS or wall_seconds > BUDGET_WALL_SECONDS
    all_checks = all(row["passed"] for row in results)
    passed = all_checks and gate["passed"] and not budget_stop
    if budget_stop:
        status = "budget_stop"
    elif passed:
        status = "passed_r022_credit_only"
    elif not gate["passed"]:
        failing = [
            key
            for key in (
                "unexpected",
                "below_noise",
                "fd_disagreements",
                "directional_disagreements",
                "second_disagreements",
                "fp32_violations",
                "probe_failures",
                "control_failures",
            )
            if gate[key]
        ]
        status = f"failed_{'_and_'.join(failing)}" if failing else "failed_update_or_clamp"
    else:
        status = "failed_check"
    r021 = json.loads(R021_RECEIPT.read_text(encoding="utf-8"))
    return {
        "schema": "taiji-r022-credit-validation-v1",
        "date": "2026-10-31",
        "baseline_commit": BASELINE_COMMIT,
        "contract": CONTRACT.relative_to(ROOT).as_posix(),
        "status": status,
        "config": asdict(
            core.LivingCoreConfig(
                groups=2, width=8, units=12, adapter_rank=2, key_width=4, value_width=4
            )
        ),
        "dtypes": ["torch.float64", "torch.float32"],
        "torch_version": core.torch.__version__,
        "device": str(core.torch.get_default_dtype()),
        "subject_id": "r022-credit",
        "fixture": {
            "prompt": "prompt",
            "goal": "goal",
            "target": "ab",
            "records": [
                {"source_id": r.source_id, "raw": r.raw.decode("ascii"), "version": r.version}
                for r in checks.RECORDS
            ],
            "raw_nu": list(checks.RAW_NU),
            "known_nu": list(checks.KNOWN_NU),
            "now": checks.NOW,
        },
        "fixture_parameter_sha256_pre_update": payload["fixture_parameter_sha256_pre_update"],
        "fixture_parameter_sha256_post_update": payload["fixture_parameter_sha256_post_update"],
        "r021_receipt_binding": {
            "report": R021_RECEIPT.relative_to(ROOT).as_posix(),
            "living_core_sha256_match": r021["source_hashes"]["taiji/living_core.py"]
            == digest(Path(core.__file__)),
            "r021_status": r021["status"],
        },
        "targets": {
            "T_EXPR": {
                "definition": "teacher-forced prediction NLL over score_targets on the synthetic fixture",
                "owner_families": [
                    "encoder",
                    "nu_projection",
                    "message_q",
                    "message_k",
                    "message_v",
                    "message_goal",
                    "fast_queries",
                    "fast_outputs",
                    "episodic",
                    "W",
                    "U",
                    "V",
                    "R",
                    "P",
                    "tau_logit",
                    "xi",
                    "readout",
                    "expression",
                    "write_heads (cross-step F chain)",
                ],
                "loss_float64": payload["losses_float64"]["T_EXPR"],
                "loss_float32": payload["losses_float32"]["T_EXPR"],
            },
            "T_WRITE2": {
                "definition": "two-step chain: allowed write into F, then read F and score prediction NLL",
                "owner_families": ["write_heads"],
                "loss_float64": payload["losses_float64"]["T_WRITE2"],
                "loss_float32": payload["losses_float32"]["T_WRITE2"],
            },
            "T_WORLD_PROXY": {
                "definition": "MSE of three world heads against declared constant synthetic targets; proxy only",
                "owner_families": ["world"],
                "loss_float64": payload["losses_float64"]["T_WORLD_PROXY"],
                "loss_float32": payload["losses_float32"]["T_WORLD_PROXY"],
                "real_target_stage": "09.3/09.4 (world learning not implemented in R021)",
            },
        },
        "checks": results,
        "gate": gate,
        "parameter_classification": payload["tensors"],
        "zero_class_probes": payload["probes"],
        "negative_controls": payload["controls"],
        "finite_difference": payload["finite_difference"],
        "directional_first_derivative": payload["directional"],
        "second_directional_derivative": payload["second_derivative"],
        "float32_float64_comparison": payload["fp32"],
        "declared_update": payload["update"],
        "backward_calls": checks.BACKWARD_CALLS,
        "optimizer_steps": 2,
        "data_splits_read": [],
        "full_subject_restore": False,
        "default_model_replaced": False,
        "persistent_subject": False,
        "training_run": False,
        "updated_parameters_discarded": True,
        "cpu_seconds_checks_only": cpu_seconds,
        "wall_seconds_checks_only": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "source_hashes": {
            path.relative_to(ROOT).as_posix(): digest(path)
            for path in (CONTRACT, Path(core.__file__), TEST_FILE, Path(__file__))
        },
        "limits": "Credit ownership, derivative agreement, negative controls and the declared "
        "synthetic update expression on one fixture. No task capability, growth, retention, "
        "recovery, pricing or training approval. The inert episodic.score.bias finding awaits "
        "an R021 interface revision decision.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("R022 receipt is write-once")
    result = verify()
    if args.output:
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "checks": len(result["checks"]),
                "failed": [row["id"] for row in result["checks"] if not row["passed"]],
                "gate": result["gate"],
                "backward_calls": result["backward_calls"],
                "cpu_seconds": round(result["cpu_seconds_checks_only"], 2),
            }
        )
    )
    return int(result["status"] != "passed_r022_credit_only")


if __name__ == "__main__":
    raise SystemExit(main())
