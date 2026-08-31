"""P5-1 canary: keep Workbench protocol orchestration outside the runtime facade."""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.natural_language_workbench import NaturalLanguageWorkbenchOrchestrator  # noqa: E402
from api.seed_runtime import SeedRuntime  # noqa: E402

REPORT_FORMAT = "taiji-w7-p5-1-natural-language-workbench-modularization-v1"


class _SpyRuntime:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def _execute_natural_language_workbench_task_impl(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return {"status": "planned"}


def evaluate() -> dict[str, object]:
    runtime_source = inspect.getsource(SeedRuntime)
    orchestrator_source = inspect.getsource(NaturalLanguageWorkbenchOrchestrator)
    module_source = (
        PROJECT_ROOT / "api" / "natural_language_workbench.py"
    ).read_text(encoding="utf-8")

    spy = _SpyRuntime()
    orchestrator = NaturalLanguageWorkbenchOrchestrator(spy)
    orchestrator.plan(
        "edit",
        {"semantic_steps": []},
        snapshot_id="snapshot-1",
        loop_id="loop-1",
    )
    orchestrator.execute(
        "edit",
        {"semantic_steps": []},
        snapshot_id="snapshot-1",
        loop_id="loop-2",
    )

    metrics = {
        "dedicated_protocol_module_owns_plan_approval_execute": all(
            name in orchestrator_source
            for name in ("def plan(", "def approve(", "def execute_planned(")
        ),
        "runtime_public_facade_delegates_to_orchestrator": (
            "NaturalLanguageWorkbenchOrchestrator(self)" in runtime_source
            and "def plan_natural_language_workbench_task(" in runtime_source
            and "def execute_planned_natural_language_workbench_task(" in runtime_source
        ),
        "module_does_not_import_runtime_or_provider": (
            "from .seed_runtime import" not in module_source
            and "from seed.language_provider import" not in module_source
        ),
        "legacy_execute_and_plan_forward_without_semantic_mutation": (
            len(spy.calls) == 2
            and spy.calls[0][1].get("prepare_only") is True
            and "parameter_bindings" not in spy.calls[0][1]
            and "parameter_bindings" not in spy.calls[1][1]
        ),
        "previous_protocol_gates_remain_green": (
            json.loads(
                (
                    PROJECT_ROOT
                    / "reports"
                    / "taiji_w7_p2_12_natural_language_write_20260831.json"
                ).read_text(encoding="utf-8")
            )["gate"]["passed"]
            and json.loads(
                (
                    PROJECT_ROOT
                    / "reports"
                    / "taiji_w7_p2_13_natural_language_workbench_api_20260831.json"
                ).read_text(encoding="utf-8")
            )["gate"]["passed"]
        ),
    }
    return {
        "format": REPORT_FORMAT,
        "task": "Natural-language Workbench orchestration modularization",
        "metrics": metrics,
        "gate": {
            "passed": all(metrics.values()),
            "criterion": (
                "The protocol owner must be a dedicated module, SeedRuntime must remain a "
                "compatibility facade, no provider/runtime circular dependency may be added, "
                "and P2-12/P2-13 behavior must remain green."
            ),
        },
        "boundary": (
            "This Gate proves ownership and behavior-preserving modularization only. It does not "
            "claim real provider quality, a complete chat UI journey, CUDA, CI, or open-domain autonomy."
        ),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    report = evaluate()
    report_path = (
        PROJECT_ROOT
        / "reports"
        / "taiji_w7_p5_1_natural_language_workbench_modularization_20260831.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
