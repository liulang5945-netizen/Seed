"""Price the same native three-session stream after output-result reuse."""

# ruff: noqa: E402 -- include tensor import startup in the measured wall interval
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

STARTED = time.monotonic()
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training.check_report_run_conditions import write_report
from scripts.training.verify_taiji_u2_metadata_groups_22_4 import StepGuard
from taiji.u2_recovery import U2Runtime
from taiji.u2_subject import U2Config, U2Subject
from taiji.u2_task_episode import structured_world
from taiji.u2_task_executor import PhysicalExecutor
from taiji.u2_task_session import SessionBudgetStop, run_joint_session

CONTRACT = ROOT / "plans/manifests/taiji_u2_22_4_2_joint_price_20261112_v2.json"
REPORT = ROOT / "reports/taiji_u2_joint_session_22_4_2_20261112_v2.json"


def verify():
    torch.set_num_threads(1)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    for path, pin in contract["source_pins"].items():
        assert (
            hashlib.sha256((ROOT / path).read_text(encoding="utf-8-sig").encode()).hexdigest()
            == pin
        ), path
    if contract.get("prior_measured_cpu", 0) > contract["cpu_cap"]:
        if REPORT.exists():
            raise FileExistsError(REPORT)
        write_report(
            REPORT,
            {
                "schema": "taiji-u2-joint-price-precheck-v1",
                "verdict": "budget_stop_before_run",
                "prior_measured_cpu": contract["prior_measured_cpu"],
                "cpu_cap": contract["cpu_cap"],
                "model_forwards": 0,
                "optimizer_steps": 0,
                "project_data_reads": 0,
                "run_conditions": {"device": "cpu", "threads": 1},
                "reason": "Measured current cold profile exceeds the frozen cap. Reprice a new source/candidate before another full run.",
            },
        )
        print("budget_stop_before_run")
        return
    previous = json.loads((ROOT / contract["previous_report"]).read_text(encoding="utf-8"))
    with (
        tempfile.TemporaryDirectory(
            dir=ROOT / ".seed_test_tmp", prefix="u2-joint-price-"
        ) as directory,
        StepGuard() as guard,
    ):
        root = Path(directory)
        world, names = structured_world(124, [(-0.55, -0.55), (0.0, 0.0), (0.55, 0.55)])
        executor = PhysicalExecutor.create(world, root / "world.bin")
        runtime = U2Runtime(
            U2Subject(U2Config.small()).metadata_candidate(),
            executor=executor,
            subject_id="joint-cold-124",
        )
        runtime.set_goal("记住三段会话中的可核验观测。")
        try:
            runtime, result = run_joint_session(runtime, names, root, cpu_cap=contract["cpu_cap"])
        except SessionBudgetStop as error:
            error.runtime.save(root / "budget-stop.bin")
            result = {
                "answers": error.answers,
                "sessions": error.sessions,
                "correct": sum(r["correct"] for r in error.answers),
                "denominator": 18,
                "verdict": "budget_stop",
                "unanswered": 18 - len(error.answers),
            }
        else:
            assert result["answers"] == previous["answers"], "output/score semantics changed"
            result["verdict"] = "passed_joint_price_and_semantic_equivalence"
            result["outputs_equal_to_previous"] = True
        result.update(
            schema="taiji-u2-joint-session-price-v1",
            contract_sha256=hashlib.sha256(
                CONTRACT.read_text(encoding="utf-8-sig").encode()
            ).hexdigest(),
            optimizer_steps=guard.calls,
            cpu_seconds=time.process_time(),
            wall_seconds=time.monotonic() - STARTED,
            cpu_cap=contract["cpu_cap"],
            run_conditions={
                "torch": torch.__version__,
                "threads": 1,
                "device": "cpu",
                "dtype": "float32",
            },
            project_data_reads=0,
            capability_qualifications=[],
        )
        if (
            result["wall_seconds"] > contract["wall_cap"]
            or result["cpu_seconds"] > contract["cpu_cap"]
        ):
            result["verdict"] = "budget_stop"
        if REPORT.exists():
            raise FileExistsError(REPORT)
        write_report(REPORT, result)
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in (
                        "verdict",
                        "correct",
                        "denominator",
                        "cpu_seconds",
                        "wall_seconds",
                        "optimizer_steps",
                    )
                }
            )
        )


if __name__ == "__main__":
    verify()
