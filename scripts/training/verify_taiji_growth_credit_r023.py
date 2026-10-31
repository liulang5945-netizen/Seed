"""R023 growth-birth receipt; binds contract, code, checks and the write-once report."""

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

from taiji import growth_credit as growth  # noqa: E402
from taiji import living_core as core  # noqa: E402

TEST_FILE = ROOT / "tests/taiji_native/test_growth_credit_r023.py"
CONTRACT = ROOT / "plans/reference/SPEC-TAIJI-CORE_GROWTH_R023_20261031.md"
R022_RECEIPT = ROOT / "reports/taiji_core_r022_credit_validation_20261031.json"
BASELINE_COMMIT = "47b0d1bb3"
BUDGET_CPU_SECONDS = 900
BUDGET_WALL_SECONDS = 1200


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_checks():
    spec = importlib.util.spec_from_file_location("r023_growth_checks", TEST_FILE)
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


def summarize(data) -> dict:
    book = data["book"]
    verdicts = []
    for name, verdict in data["verdicts"].items():
        for entry in verdict["groups"]:
            verdicts.append(
                {
                    "branch": name,
                    "group": entry["group"],
                    "frobenius": entry["frobenius"],
                    "sigma": entry["sigma"],
                    "tau": entry["tau"],
                    "tau_noise": entry["tau_noise"],
                    "noise_margin": entry["sigma"][0] / entry["tau_noise"],
                    "rank_num": entry["rank_num"],
                    "enter_V": entry["enter_V"],
                    "alignment": entry["alignment"],
                }
            )
    credit_global = sum(data["full_credits"].values())
    startup = growth.startup_report(data["plan"])
    independence = growth.independence_ratio(data["directions"], data["full_credits"])
    baseline = growth.global_svd_baseline(
        torch.stack(
            [sum(data["fit_credits"][name][g] for name in data["fit_credits"]) for g in range(2)]
        )
    )
    return {
        "career_steps": len(book.deltas),
        "round_count": book.round_count,
        "loss_aggregate": book.loss,
        "branch_gate_definition": "K=2: copy=trace mixing m_s, generate=1-m_s (auditable graph quantity)",
        "branch_verdicts": verdicts,
        "allocation_per_group": {
            str(group): proposal for group, proposal in data["proposals"].items()
        },
        "allocation_merged": data["allocation"],
        "birth_plan": {
            key: value
            for key, value in data["plan"].items()
            if key not in {"predicted_incoming", "parent_r", "parent_p"}
        },
        "startup_report": startup,
        "independence_ratio": independence,
        "kappa": growth.cancellation_index(data["full_credits"], credit_global),
        "global_svd_baseline": {
            "groups": [
                {
                    "group": group["group"],
                    "units": group["units"],
                    "refused": group["refused"],
                    "tau": group["tau"],
                    "sigma_top3": group["sigma"][:3],
                }
                for group in baseline["groups"]
            ]
        },
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
        for name in sorted(name for name in vars(checks) if name.startswith("test_r023_")):
            function = getattr(checks, name)
            try:
                function()
                result = {"id": name, "passed": True}
            except Exception as exc:  # Evidence records the actual failure, never a fake PASS.
                result = {"id": name, "passed": False, "error": f"{type(exc).__name__}: {exc}"}
            result["test"] = location(function)
            results.append(result)
        data = checks.pipeline()
    finally:
        torch.autograd.grad = original_grad
        torch.Tensor.backward = original_backward
    payload = summarize(data)
    cpu_seconds, wall_seconds = time.process_time() - cpu, time.perf_counter() - start
    budget_stop = cpu_seconds > BUDGET_CPU_SECONDS or wall_seconds > BUDGET_WALL_SECONDS
    all_checks = all(row["passed"] for row in results)
    passed = all_checks and not budget_stop
    if budget_stop:
        status = "budget_stop"
    elif passed:
        status = "passed_r023_birth_engineering_only"
    else:
        status = "failed_" + "_".join(
            row["id"].replace("test_r023_", "") for row in results if not row["passed"]
        )
    r022 = json.loads(R022_RECEIPT.read_text(encoding="utf-8"))
    living_digest = digest(Path(core.__file__))
    return {
        "schema": "taiji-r023-growth-birth-v1",
        "date": "2026-10-31",
        "baseline_commit": BASELINE_COMMIT,
        "contract": CONTRACT.relative_to(ROOT).as_posix(),
        "status": status,
        "config": asdict(checks.CONFIG),
        "dtype": "torch.float64",
        "torch_version": core.torch.__version__,
        "lineage_binding": {
            "living_core_sha256": living_digest,
            "matches_r022_receipt": r022["source_hashes"]["taiji/living_core.py"] == living_digest,
            "r022_status": r022["status"],
        },
        "frozen_bounds": {
            "r_max": growth.R_MAX,
            "tau_g": growth.TAU_G,
            "floor_frobenius": growth.FLOOR_FROBENIUS,
            "cosine_min": growth.COSINE_MIN,
            "window_min_positions": growth.WINDOW_MIN_POSITIONS,
            "permutations": growth.PERMUTATIONS,
            "noise_quantile": growth.NOISE_QUANTILE,
            "c_scale": growth.C_SCALE,
            "enumeration_cap": growth.ENUMERATION_CAP,
            "proposal_cpu_seconds": growth.PROPOSAL_CPU_SECONDS,
            "proposal_wall_seconds": growth.PROPOSAL_WALL_SECONDS,
        },
        "checks": results,
        "summary": payload,
        "calibration_finding": {
            "headline": "frozen 0.99 permutation bar does not reject random gates on this fixture",
            "cause_hypothesis": "temporal common-mode structure inflates the permuted null",
            "disposition": "recorded for the R021 §4 calibration-protocol owner decision; thresholds unchanged",
        },
        "backward_calls": grad_counter["grad"] + grad_counter["backward"],
        "gradient_calls": grad_counter["grad"],
        "backward_calls_direct": grad_counter["backward"],
        "optimizer_steps": 0,
        "data_splits_read": [],
        "full_subject_restore": False,
        "default_model_replaced": False,
        "persistent_subject": False,
        "training_run": False,
        "task_capability_claimed": False,
        "growth_net_benefit_claimed": False,
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
                Path(__file__),
            )
        },
        "limits": "Coverage allocation, gated birth, noise calibration, diagnostics and the "
        "bounded replay migration on one synthetic fixture. No task benefit, retention, "
        "recovery, pricing or growth-net-benefit approval. The calibration "
        "non-discrimination finding awaits an R021 §4 owner decision.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("R023 receipt is write-once")
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
                "allocation": result["summary"]["allocation_merged"],
                "l_start": result["summary"]["startup_report"]["l_start"],
                "kappa": result["summary"]["kappa"],
                "cpu_seconds": round(result["cpu_seconds_checks_only"], 2),
            }
        )
    )
    return int(result["status"] != "passed_r023_birth_engineering_only")


if __name__ == "__main__":
    raise SystemExit(main())
