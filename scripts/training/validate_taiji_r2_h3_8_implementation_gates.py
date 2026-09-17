"""R2-H3.8 implementation-gate evidence runner.

Contract: plans/reference/M5_R2_H3_8_ISOLATED_PROTOTYPE_CONTRACT_20260917.md
section 2.  Runs the five gate groups (isolation/inventory, gradients, causal
mask, save-restore, wiring) plus the scoped static checks and records the
evidence.  It performs no capability training and reads no evaluation split.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from taiji.internalization import content_digest  # noqa: E402
from taiji.sequence_workspace import (  # noqa: E402
    SEQUENCE_WORKSPACE_PARAMETERS,
    SequenceWorkspaceConfig,
    SequenceWorkspacePrototype,
)

REPORT_FORMAT = "taiji-r2-h3-8-implementation-gates-v1"
CONTRACT = "plans/reference/M5_R2_H3_8_ISOLATED_PROTOTYPE_CONTRACT_20260917.md"
GATE_TESTS = "tests/taiji_native/test_sequence_workspace_contract.py"
STATIC_FILES = (
    "taiji/sequence_workspace.py",
    "tests/taiji_native/test_sequence_workspace_contract.py",
)
GATE_PREFIXES = {
    "gate1_isolation_and_inventory": ("test_gate1",),
    "gate2_gradients": ("test_gate2",),
    "gate3_causal_mask": ("test_gate3",),
    "gate4_save_and_restore": ("test_gate4",),
    "gate5_wiring": ("test_gate5",),
}


def _run(command: list[str], *, timeout: float = 1200.0) -> dict[str, Any]:
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout,
    )
    return {
        "command": command,
        "returncode": result.returncode,
        "passed": result.returncode == 0,
        "tail": (result.stdout or result.stderr or "").strip().splitlines()[-5:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/r2_h3_8_implementation_gates_20260917.json"),
    )
    args = parser.parse_args()

    prototype = SequenceWorkspacePrototype(SequenceWorkspaceConfig())
    inventory = {
        "declared_parameters": list(SEQUENCE_WORKSPACE_PARAMETERS),
        "parameter_count": prototype.parameter_count(),
        "initialization_digest": content_digest(prototype.parameter_payload()),
    }

    static_checks = {
        "ruff": _run([sys.executable, "-m", "ruff", "check", *STATIC_FILES]),
        "black": _run([sys.executable, "-m", "black", "--no-cache", "--check", *STATIC_FILES]),
        "mypy": _run([sys.executable, "-m", "mypy", "--follow-imports=silent", STATIC_FILES[0]]),
    }

    with tempfile.TemporaryDirectory(prefix="h38-gates-") as temporary:
        junit = Path(temporary) / "gates.xml"
        pytest_run = _run(
            [
                sys.executable,
                "-m",
                "pytest",
                GATE_TESTS,
                "-q",
                "--no-header",
                f"--junitxml={junit}",
            ]
        )
        cases: list[dict[str, Any]] = []
        if junit.exists():
            for case in ElementTree.parse(junit).getroot().iter("testcase"):
                cases.append(
                    {
                        "name": case.attrib.get("name", ""),
                        "classname": case.attrib.get("classname", ""),
                        "outcome": (
                            "failed"
                            if case.find("failure") is not None
                            else "skipped" if case.find("skipped") is not None else "passed"
                        ),
                    }
                )
    gates: dict[str, Any] = {}
    for gate, prefixes in GATE_PREFIXES.items():
        group = [case for case in cases if case["name"].startswith(prefixes)]
        gates[gate] = {
            "tests": [case["name"] for case in group],
            "passed": bool(group) and all(case["outcome"] == "passed" for case in group),
        }

    passed = (
        all(item["passed"] for item in static_checks.values())
        and pytest_run["passed"]
        and all(item["passed"] for item in gates.values())
    )
    payload = {
        "format": REPORT_FORMAT,
        "version": 1,
        "contract": CONTRACT,
        "scope": (
            "implementation gates only: no capability training, no evaluation "
            "split read, no default-entry or existing-checkpoint change"
        ),
        "inventory": inventory,
        "static_checks": static_checks,
        "pytest": pytest_run,
        "gates": gates,
        "outcome": "passed" if passed else "failed",
        "growth_admitted": False,
        "can_promote": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "outcome": payload["outcome"],
                "gates": {name: item["passed"] for name, item in gates.items()},
                "static": {name: item["passed"] for name, item in static_checks.items()},
            },
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
