"""P5-3 canary: keep Workbench execution boundary outside SeedRuntime."""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.seed_runtime import SeedRuntime  # noqa: E402
from api.workbench_execution import WorkbenchExecutionBoundary  # noqa: E402
from taiji import ActionIntent  # noqa: E402

REPORT_FORMAT = "taiji-w7-p5-3-natural-language-workbench-execution-modularization-v1"


class _Descriptor:
    risk = "read_only"


class _Snapshot:
    snapshot_id = "snapshot-execution"

    def get(self, capability_id):
        return _Descriptor() if capability_id == "workspace.read" else None


class _Registry:
    snapshot_id = "registry-execution"


class _Environment:
    capability_snapshot = _Snapshot()
    capability_registry = _Registry()
    mcp_registry = _Registry()


class _Runtime:
    class _Model:
        tick = 7

    model = _Model()

    def __init__(self):
        self.calls: list[tuple[str, object, dict[str, object]]] = []

    def preflight_workbench_loop(self, requests, **kwargs):
        self.calls.append(("preflight", requests, kwargs))
        return {"accepted": True, "preflight_id": "preflight-execution"}

    def execute_preflighted_workbench_loop(self, intents, requests, **kwargs):
        self.calls.append(("execute", intents, kwargs))
        return {
            "status": "completed",
            "steps": [{"success": True, "capability_id": "workspace.read"}],
        }


def evaluate() -> dict[str, object]:
    runtime_source = inspect.getsource(SeedRuntime._execute_natural_language_workbench_task_impl)
    module = sys.modules[WorkbenchExecutionBoundary.__module__]
    module_source = inspect.getsource(module)

    runtime = _Runtime()
    intent = ActionIntent(
        "intent:execution-boundary",
        "workspace.read",
        parameters={"path": "README.md"},
        confidence=0.95,
        tick=7,
    )
    result = WorkbenchExecutionBoundary(runtime).run(
        base={"execution": {"status": "not_executed", "side_effects": False}},
        planning_steps=({"index": 0, "step_id": "step-0"},),
        intents=(intent,),
        environment=_Environment(),
        loop_id="loop-execution-boundary",
        max_steps=1,
        max_budget_units=1.0,
        learn=False,
        prepare_only=False,
    )

    metrics = {
        "dedicated_execution_module_owns_request_preflight_execution": all(
            marker in module_source
            for marker in (
                "WorkbenchActionRequest.from_action_intent",
                "preflight_workbench_loop",
                "execute_preflighted_workbench_loop",
                '"side_effects"',
            )
        ),
        "runtime_execution_path_delegates_to_boundary": (
            "WorkbenchExecutionBoundary(self).run" in runtime_source
            and "preflight_workbench_loop(" not in runtime_source
            and "execute_preflighted_workbench_loop(" not in runtime_source
            and "WorkbenchActionRequest.from_action_intent" not in runtime_source
        ),
        "execution_module_has_no_runtime_or_provider_dependency": (
            "from .seed_runtime import" not in module_source
            and "from api.seed_runtime import" not in module_source
            and "from seed.language_provider import" not in module_source
        ),
        "boundary_preserves_current_snapshot_and_execution_contract": (
            result["status"] == "completed"
            and result["execution"]["status"] == "completed"
            and result["execution"]["side_effects"] is False
            and len(runtime.calls) == 2
            and runtime.calls[0][0] == "preflight"
            and runtime.calls[1][0] == "execute"
            and runtime.calls[0][1][0].snapshot_id == "snapshot-execution"
            and runtime.calls[0][1][0].capability_registry_snapshot_id
            == "registry-execution"
        ),
        "previous_protocol_and_grounding_gates_remain_green": all(
            json.loads(
                (PROJECT_ROOT / "reports" / report_name).read_text(encoding="utf-8")
            )["gate"]["passed"]
            for report_name in (
                "taiji_w7_p2_13_natural_language_workbench_api_20260831.json",
                "taiji_w7_p5_1_natural_language_workbench_modularization_20260831.json",
                "taiji_w7_p5_2_natural_language_workbench_grounding_modularization_20260831.json",
            )
        ),
    }
    return {
        "format": REPORT_FORMAT,
        "task": "Workbench request, preflight, and execution boundary modularization",
        "metrics": metrics,
        "gate": {
            "passed": all(metrics.values()),
            "criterion": (
                "The Workbench execution boundary must own current-snapshot request binding, "
                "preflight, execution, and side-effect projection, while SeedRuntime remains a "
                "cognitive/runtime facade and previous protocol gates stay green."
            ),
        },
        "boundary": (
            "This Gate proves behavior-preserving ownership modularization only. It does not "
            "claim real provider quality, unrestricted execution, CUDA, CI, or open-domain autonomy."
        ),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    report = evaluate()
    report_path = (
        PROJECT_ROOT
        / "reports"
        / "taiji_w7_p5_3_natural_language_workbench_execution_modularization_20260831.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
