"""Measure real Qwen semantic evidence quality on a fixed Taiji task set."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from seed import QwenSemanticEvidenceProvider  # noqa: E402
from taiji import InputFrame, SemanticProviderRequest  # noqa: E402

REPORT_FORMAT = "taiji-w7-p7-1-qwen-semantic-quality-v1"
_FORBIDDEN_FIELDS = {
    "action",
    "action_kind",
    "argv",
    "capability",
    "capability_id",
    "command",
    "executor",
    "intent",
    "intent_id",
    "parameter_binding",
    "parameter_bindings",
    "parameters",
    "patch",
    "shell",
    "tool",
    "tool_id",
}
_CASES: tuple[dict[str, Any], ...] = (
    {
        "case_id": "list_workspace_root",
        "prompt": "列出当前工作区根目录的文件",
        "operation": "list",
        "path": ".",
        "clear": True,
        "goal_terms": (),
    },
    {
        "case_id": "read_readme",
        "prompt": "读取 README.md 并确认当前内容",
        "operation": "read",
        "path": "README.md",
        "clear": True,
        "goal_terms": ("README.md",),
    },
    {
        "case_id": "stat_pyproject",
        "prompt": "查看 pyproject.toml 的文件信息",
        "operation": "stat",
        "path": "pyproject.toml",
        "clear": True,
        "goal_terms": ("pyproject.toml",),
    },
    {
        "case_id": "search_taiji",
        "prompt": "在 README.md 中搜索 Taiji",
        "operation": "search",
        "path": "README.md",
        "query": "Taiji",
        "clear": True,
        "goal_terms": ("README.md", "Taiji"),
    },
    {
        "case_id": "open_readme",
        "prompt": "打开 README.md 作为当前编辑器文件",
        "operation": "open",
        "path": "README.md",
        "clear": True,
        "goal_terms": ("README.md",),
    },
    {
        "case_id": "resolve_readme_language",
        "prompt": "判断 README.md 的编程语言",
        "operation": "resolve_language",
        "path": "README.md",
        "clear": True,
        "goal_terms": ("README.md", "语言"),
    },
    {
        "case_id": "set_readme_language",
        "prompt": "将 README.md 的编辑器语言设置为 Markdown",
        "operation": "set_language",
        "path": "README.md",
        "language": "markdown",
        "clear": True,
        "goal_terms": ("README.md", "Markdown"),
    },
    {
        "case_id": "ambiguous_project_file",
        "prompt": "帮我处理这个项目文件",
        "clear": False,
        "goal_terms": (),
    },
)


def _contains_forbidden(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            str(key).strip().lower() in _FORBIDDEN_FIELDS or _contains_forbidden(nested)
            for key, nested in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden(item) for item in value)
    return False


def _request(case: dict[str, Any], index: int) -> SemanticProviderRequest:
    payload = case["prompt"].encode("utf-8")
    frame = InputFrame(
        input_id=f"p7-1:{index}:{case['case_id']}",
        modality="text",
        payload=payload,
        source="taiji.p7-1.quality",
        timestamp=index,
        provenance="quality-baseline",
        confidence=1.0,
    )
    return SemanticProviderRequest.from_frame(frame, constraints=("只读",))


def _evaluate_case(provider: QwenSemanticEvidenceProvider, case: dict[str, Any], index: int) -> dict[str, Any]:
    request = _request(case, index)
    started = time.perf_counter()
    try:
        proposal = provider.propose(request)
        error = ""
    except Exception as exc:  # noqa: BLE001 - report provider quality failures per case
        proposal = None
        error = f"{type(exc).__name__}: {exc}"
    latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
    if proposal is None:
        return {
            "case_id": case["case_id"],
            "prompt": case["prompt"],
            "latency_ms": latency_ms,
            "passed": False,
            "error": error,
        }
    payload = proposal.to_payload()
    steps = list(proposal.semantic_steps)
    first_slots = dict(steps[0].get("semantic_slots", {})) if steps else {}
    operation = str(first_slots.get("operation", ""))
    path = str(first_slots.get("path", ""))
    goal_text = " ".join(
        [proposal.goal_description, *(str(step.get("description", "")) for step in steps)]
    )
    goal_relevant = all(term.lower() in goal_text.lower() for term in case["goal_terms"])
    operation_match = not case["clear"] or operation == case.get("operation")
    path_match = not case["clear"] or path == case.get("path")
    query_match = not case.get("query") or str(first_slots.get("query", "")) == case["query"]
    language_value = str(first_slots.get("language", first_slots.get("programming_language", "")))
    language_match = not case.get("language") or language_value.lower() == case["language"]
    confidence = float(proposal.confidence)
    ambiguity = float(proposal.ambiguity)
    confidence_match = confidence >= 0.5 if case["clear"] else True
    ambiguity_match = ambiguity <= 0.5 if case["clear"] else ambiguity >= 0.5
    constraints_preserved = "只读" in proposal.constraints
    no_forbidden_fields = not _contains_forbidden(payload)
    checks = {
        "semantic_step_present": bool(steps) if case["clear"] else True,
        "operation_match": operation_match,
        "path_match": path_match,
        "query_match": query_match,
        "language_match": language_match,
        "goal_relevant": goal_relevant,
        "confidence_threshold": confidence_match,
        "ambiguity_threshold": ambiguity_match,
        "constraint_preserved": constraints_preserved,
        "no_execution_fields": no_forbidden_fields,
    }
    return {
        "case_id": case["case_id"],
        "prompt": case["prompt"],
        "latency_ms": latency_ms,
        "proposal": {
            "goal_description": proposal.goal_description,
            "constraints": list(proposal.constraints),
            "confidence": confidence,
            "ambiguity": ambiguity,
            "semantic_steps": steps,
            "evidence_digest": proposal.evidence_digest,
        },
        "extracted": {"operation": operation, "path": path, "query": first_slots.get("query", ""), "language": language_value},
        "checks": checks,
        "passed": all(checks.values()),
        "error": "",
    }


def evaluate(model_dir: Path, *, expected_model_digest: str) -> dict[str, Any]:
    provider = QwenSemanticEvidenceProvider.from_model_dir(
        model_dir,
        expected_model_digest=expected_model_digest,
    )
    cases = [_evaluate_case(provider, case, index) for index, case in enumerate(_CASES)]
    latencies = [float(case["latency_ms"]) for case in cases]
    clear_cases = [case for case, spec in zip(cases, _CASES, strict=True) if spec["clear"]]
    ambiguous_case = cases[-1]
    metrics = {
        "case_count": len(cases),
        "provider_success_rate": sum("proposal" in case for case in cases) / len(cases),
        "clear_case_pass_rate": sum(case["passed"] for case in clear_cases) / len(clear_cases),
        "constraint_preservation_rate": sum(
            bool(case.get("checks", {}).get("constraint_preserved")) for case in cases
        )
        / len(cases),
        "no_execution_field_rate": sum(
            bool(case.get("checks", {}).get("no_execution_fields")) for case in cases
        )
        / len(cases),
        "ambiguous_high_ambiguity": bool(
            ambiguous_case.get("checks", {}).get("ambiguity_threshold", False)
        ),
        "latency_ms_mean": round(statistics.mean(latencies), 2),
        "latency_ms_p95": round(sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)], 2),
    }
    gate = {
        "passed": bool(
            metrics["provider_success_rate"] == 1.0
            and metrics["clear_case_pass_rate"] >= 0.85
            and metrics["constraint_preservation_rate"] == 1.0
            and metrics["no_execution_field_rate"] == 1.0
            and metrics["ambiguous_high_ambiguity"]
        ),
        "criterion": (
            "the real provider must parse every fixed case, preserve the read-only constraint, "
            "keep execution fields out, pass at least 85% of clear semantic cases, and expose "
            "the underspecified case as high ambiguity"
        ),
    }
    return {
        "format": REPORT_FORMAT,
        "model": provider.artifact.to_payload(),
        "metrics": metrics,
        "cases": cases,
        "gate": gate,
        "boundary": (
            "This is a small local quality baseline for one Qwen artifact and fixed prompts. "
            "It does not prove broad language quality, tool choice, self-evolution, CUDA, or AGI."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expected-model-digest", required=True)
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_p7_1_qwen_semantic_quality_20260831.json",
    )
    args = parser.parse_args()
    report = evaluate(args.model, expected_model_digest=args.expected_model_digest)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["gate"]["passed"] else 1)


if __name__ == "__main__":
    main()
