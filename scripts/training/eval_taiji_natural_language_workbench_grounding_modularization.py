"""P5-2 canary: keep semantic Workbench grounding outside SeedRuntime."""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.seed_runtime import SeedRuntime  # noqa: E402
from api.workbench_grounding import (  # noqa: E402
    ground_natural_language_workbench_step,
)

REPORT_FORMAT = "taiji-w7-p5-2-natural-language-workbench-grounding-modularization-v1"


class _LanguageEnvironment:
    class _Snapshot:
        def ground_semantic_step(self, slots, **kwargs):
            assert kwargs == {
                "allow_reversible_ui": True,
                "allow_controlled_write": True,
            }
            return {"editor.set_language": ({"path": slots["path"]},)}

    capability_snapshot = _Snapshot()

    def resolve_programming_language_evidence(self, parameters):
        return {
            "path": parameters["path"],
            "programming_language_id": "python",
            "selection_state": "inferred",
        }


class _PatchEnvironment:
    class _Snapshot:
        def ground_semantic_step(self, slots, **kwargs):
            assert kwargs == {
                "allow_reversible_ui": True,
                "allow_controlled_write": True,
            }
            return {
                "workspace.apply_patch": (
                    {"path": slots["path"], "operation": "patch"},
                )
            }

    capability_snapshot = _Snapshot()

    def read_workspace_evidence(self, parameters):
        return {
            "path": parameters["path"],
            "content": "Seed editor source\n",
            "digest": "fixture-before",
            "truncated": False,
        }


def evaluate() -> dict[str, object]:
    runtime_source = inspect.getsource(SeedRuntime._ground_natural_language_workbench_step)
    module = sys.modules[ground_natural_language_workbench_step.__module__]
    module_source = inspect.getsource(module)

    language_step = SimpleNamespace(
        semantic_slots={"operation": "set_language", "path": "main.py"}
    )
    language_bindings, language_error, language_evidence, language_key = (
        ground_natural_language_workbench_step(_LanguageEnvironment(), language_step)
    )
    patch_step = SimpleNamespace(
        semantic_slots={
            "operation": "patch",
            "path": "reports/fixture.txt",
            "edit": {"kind": "replace_text", "find": "Seed", "replace": "Taiji"},
        }
    )
    patch_bindings, patch_error, patch_evidence, patch_key = (
        ground_natural_language_workbench_step(_PatchEnvironment(), patch_step)
    )

    metrics = {
        "dedicated_grounding_module_owns_live_language_and_patch_derivation": all(
            marker in module_source
            for marker in (
                "resolve_programming_language_evidence",
                "read_workspace_evidence",
                "expected_after_digest",
            )
        ),
        "runtime_grounding_method_is_a_thin_facade": (
            "ground_natural_language_workbench_step(environment, step)" in runtime_source
            and len(runtime_source.splitlines()) <= 16
        ),
        "grounding_module_has_no_runtime_or_provider_dependency": (
            "from .seed_runtime import" not in module_source
            and "from api.seed_runtime import" not in module_source
            and "from seed.language_provider import" not in module_source
        ),
        "language_binding_still_comes_from_live_workbench_evidence": (
            language_error == ""
            and language_key == "language_evidence"
            and language_evidence["programming_language_id"] == "python"
            and language_bindings["editor.set_language"][0]["programming_language_id"]
            == "python"
        ),
        "patch_binding_still_has_digest_checked_declarative_operation": (
            patch_error == ""
            and patch_key == "patch_evidence"
            and patch_evidence["before_digest"] == "fixture-before"
            and patch_bindings["workspace.apply_patch"][0]["patch"]["kind"]
            == "text_replace"
            and bool(
                patch_bindings["workspace.apply_patch"][0]["expected_after_digest"]
            )
        ),
        "previous_workbench_gates_remain_green": all(
            json.loads(
                (
                    PROJECT_ROOT / "reports" / report_name
                ).read_text(encoding="utf-8")
            )["gate"]["passed"]
            for report_name in (
                "taiji_w7_p2_11_ide_language_chain_20260831.json",
                "taiji_w7_p2_12_natural_language_write_20260831.json",
                "taiji_w7_p2_13_natural_language_workbench_api_20260831.json",
                "taiji_w7_p5_1_natural_language_workbench_modularization_20260831.json",
            )
        ),
    }
    return {
        "format": REPORT_FORMAT,
        "task": "Semantic Workbench grounding engine modularization",
        "metrics": metrics,
        "gate": {
            "passed": all(metrics.values()),
            "criterion": (
                "The live language and digest-checked patch grounding engine must have a "
                "runtime-agnostic owner, while SeedRuntime remains a compatibility facade and "
                "the existing Workbench protocol gates stay green."
            ),
        },
        "boundary": (
            "This Gate proves behavior-preserving ownership modularization only. It does not "
            "claim real provider quality, unconstrained semantic interpretation, CUDA, CI, or "
            "open-domain autonomy."
        ),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    report = evaluate()
    report_path = (
        PROJECT_ROOT
        / "reports"
        / "taiji_w7_p5_2_natural_language_workbench_grounding_modularization_20260831.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
