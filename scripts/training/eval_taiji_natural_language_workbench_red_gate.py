"""Run the P2 red Gate for the missing prose-to-Workbench planner path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.app import create_app  # noqa: E402

REPORT_FORMAT = "taiji-w7-p2-natural-language-workbench-red-gate-v1"


def evaluate() -> dict[str, object]:
    app = create_app(startup_tasks=False)
    with TestClient(app) as client:
        response = client.post(
            "/api/chat/workbench/stream",
            json={
                "prompt": "请读取 README.md 并告诉我项目如何启动",
                "history": [],
            },
        )
    payload = response.json()
    detail = payload.get("detail", ()) if isinstance(payload, dict) else ()
    missing_intent = any(
        item.get("loc", ())[-1:] == ["intent"]
        for item in detail
        if isinstance(item, dict)
    )
    metrics = {
        "ordinary_prompt_is_not_silently_executed": response.status_code == 422,
        "request_boundary_identifies_missing_intent": missing_intent,
    }
    return {
        "format": REPORT_FORMAT,
        "endpoint": "/api/chat/workbench/stream",
        "status_code": response.status_code,
        "detail": detail,
        "metrics": metrics,
        "gate": {
            "passed": all(metrics.values()),
            "criterion": (
                "before P2 planning is wired, a prose-only Workbench request must be "
                "rejected at the explicit ActionIntent boundary rather than selecting a tool"
            ),
        },
        "gap": {
            "current": "ChatWorkbenchRequest requires a pre-formed intent",
            "next": "Taiji-owned TaskInterpretation and Goal evidence before planner selection",
        },
        "boundary": (
            "This red Gate proves the current missing capability and intentionally does not "
            "claim autonomous planning, IDE execution, provider quality, CUDA, CI, or AGI."
        ),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "taiji_w7_p2_natural_language_workbench_red_gate_20260831.json",
    )
    args = parser.parse_args()
    report = evaluate()
    report_path = args.report if args.report.is_absolute() else PROJECT_ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
