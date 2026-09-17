"""R2-D2 implementation-gate evidence runner (graph v3, no training).

Contract:
plans/reference/M5_R2_D2_PER_POSITION_EVIDENCE_PREREGISTRATION_FROZEN_20260918.md
section 5.1.  Runs the seven gate groups: inventory/counts, single-channel
structure, per-position identity, gradients/causality, lesion wiring,
checkpoint restoration, and the context-removal construction over the whole
D1 fixture.  It performs no capability training and reads no evaluation
results; the fixture is used structurally for gate 7 only.
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

from scripts.training.build_taiji_r2_d1_measurement_fixture import (  # noqa: E402
    MATERIAL_CLAUSE,
    TAIL_MARKERS,
    remove_material_clause,
)
from taiji.internalization import content_digest  # noqa: E402
from taiji.sequence_workspace import (  # noqa: E402
    EVIDENCE_BROADCAST_FINAL,
    EVIDENCE_FINAL_STATE_SLOTS,
    EVIDENCE_PER_POSITION,
    SequenceWorkspaceConfig,
    SequenceWorkspacePrototype,
)

REPORT_FORMAT = "taiji-r2-d2-implementation-gates-v1"
CONTRACT = "plans/reference/M5_R2_D2_PER_POSITION_EVIDENCE_PREREGISTRATION_FROZEN_20260918.md"
FIXTURE = Path("tests/fixtures/r2_d1_measurement_v1.jsonl")
GATE_TESTS = (
    "tests/taiji_native/test_sequence_workspace_contract.py",
    "tests/taiji_native/test_sequence_workspace_v3_contract.py",
)
STATIC_FILES = (
    "taiji/sequence_workspace.py",
    "scripts/training/build_taiji_r2_d1_measurement_fixture.py",
    "tests/taiji_native/test_sequence_workspace_v3_contract.py",
)
GATE_PREFIXES = {
    "gate1_inventory_counts": ("test_gate1",),
    "gate2_single_channel": ("test_gate2",),
    "gate3_per_position_identity": ("test_gate3",),
    "gate4_gradients_causality": ("test_gate4",),
    "gate5_lesion_wiring": ("test_gate5",),
    "gate6_restoration": ("test_gate6",),
}
# v2 gates that must stay green unchanged while v3 lands
LEGACY_PREFIXES = {
    "legacy_v2_gates": ("test_gate1", "test_gate2", "test_gate3", "test_gate4", "test_gate5"),
}
EXPECTED_COUNTS = {
    EVIDENCE_PER_POSITION: 82_593,
    EVIDENCE_BROADCAST_FINAL: 82_593,
    EVIDENCE_FINAL_STATE_SLOTS: 101_025,
}
SYNTHETIC_REMOVAL_CASES = (
    # (split, rendered prefix, expected stem-plus-tail)
    ("train", "问：天空是什么颜色？背景：天空是蓝。答：", "问：天空是什么颜色？答："),
    (
        "dev",
        "提问：雪与太阳的颜色一致吗？线索：雪是白，太阳是灰白；回答：",
        "提问：雪与太阳的颜色一致吗？回答：",
    ),
    (
        "dev",
        "提问：雪的颜色？线索：雪是乳白。另有资料称湖水是米色；回答：",
        "提问：雪的颜色？回答：",
    ),
    ("final", "查询：夜晚属于黑吗。已知：夜晚不是黑。输出：", "查询：夜晚属于黑吗。输出："),
)


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


def _gate7_context_removal() -> dict[str, Any]:
    synthetic = []
    for _split, rendered, expected in SYNTHETIC_REMOVAL_CASES:
        got = remove_material_clause(rendered)
        synthetic.append(
            {
                "rendered": rendered,
                "removed": got,
                "expected": expected,
                "passed": got == expected
                and MATERIAL_CLAUSE.search(got) is None
                and any(got.endswith(tail) for tail in TAIL_MARKERS.values()),
            }
        )
    fixture_path = PROJECT_ROOT / FIXTURE
    records = [
        json.loads(line)
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pair_cuts: dict[str, list[str]] = {}
    violations: list[str] = []
    for record in records:
        cut = remove_material_clause(record["prefix"])
        if MATERIAL_CLAUSE.search(cut) is not None:
            violations.append(f"material marker survived: {record['id']}")
        if not any(cut.endswith(tail) for tail in TAIL_MARKERS.values()):
            violations.append(f"tail marker lost: {record['id']}")
        if record["pair_id"]:
            pair_cuts.setdefault(record["pair_id"], []).append(cut)
    diverging_pairs = [pid for pid, cuts in pair_cuts.items() if len(set(cuts)) != 1]
    return {
        "synthetic_cases": synthetic,
        "fixture_records_checked": len(records),
        "paired_prefixes_checked": len(pair_cuts),
        "pairs_with_diverging_stems": diverging_pairs,
        "violations": violations,
        "passed": (
            bool(records)
            and all(case["passed"] for case in synthetic)
            and not violations
            and not diverging_pairs
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/r2_d2_implementation_gates_20260918.json"),
    )
    args = parser.parse_args()

    counts = {}
    init_digests: dict[str, str] = {}
    for source, expected in EXPECTED_COUNTS.items():
        prototype = SequenceWorkspacePrototype(
            SequenceWorkspaceConfig(seed=20260917, evidence_source=source)
        )
        counts[source] = {
            "parameter_count": prototype.parameter_count(),
            "expected": expected,
            "passed": prototype.parameter_count() == expected,
        }
        init_digests[source] = content_digest(prototype.parameter_payload())
    defaulted = SequenceWorkspacePrototype(SequenceWorkspaceConfig())
    default_is_v2 = (
        defaulted.config.evidence_source == EVIDENCE_FINAL_STATE_SLOTS
        and defaulted.parameter_count() == 101_025
    )
    inventory = {
        "counts": counts,
        "default_graph_is_v2": default_is_v2,
        "initialization_digests": init_digests,
        "passed": default_is_v2 and all(item["passed"] for item in counts.values()),
    }

    static_checks = {
        "ruff": _run([sys.executable, "-m", "ruff", "check", *STATIC_FILES]),
        "black": _run([sys.executable, "-m", "black", "--no-cache", "--check", *STATIC_FILES]),
        "mypy": _run([sys.executable, "-m", "mypy", "--follow-imports=silent", STATIC_FILES[0]]),
    }

    with tempfile.TemporaryDirectory(prefix="r2d2-gates-") as temporary:
        junit = Path(temporary) / "gates.xml"
        pytest_run = _run(
            [
                sys.executable,
                "-m",
                "pytest",
                *GATE_TESTS,
                "-q",
                "--no-header",
                f"--junitxml={junit}",
            ]
        )
        cases: list[dict[str, Any]] = []
        if junit.exists():
            tree = ElementTree.parse(junit)
            for case in tree.getroot().iter("testcase"):
                cases.append(
                    {
                        "name": case.attrib.get("name", ""),
                        "file": Path(case.attrib.get("classname", "")).name,
                        "outcome": (
                            "failed"
                            if case.find("failure") is not None
                            else "skipped" if case.find("skipped") is not None else "passed"
                        ),
                    }
                )

    def _select(prefixes: tuple[str, ...], file_stem: str | None = None) -> list[dict[str, Any]]:
        return [
            case
            for case in cases
            if case["name"].startswith(prefixes)
            and (file_stem is None or case["file"].rsplit(".", 1)[-1] == file_stem)
        ]

    gates: dict[str, Any] = {}
    for gate, prefixes in GATE_PREFIXES.items():
        group = _select(prefixes, file_stem="test_sequence_workspace_v3_contract")
        gates[gate] = {
            "tests": [case["name"] for case in group],
            "passed": bool(group) and all(case["outcome"] == "passed" for case in group),
        }
    legacy_group = _select(
        LEGACY_PREFIXES["legacy_v2_gates"], file_stem="test_sequence_workspace_contract"
    )
    gates["legacy_v2_unchanged"] = {
        "tests": [case["name"] for case in legacy_group],
        "passed": bool(legacy_group) and all(case["outcome"] == "passed" for case in legacy_group),
    }
    gates["gate7_context_removal"] = _gate7_context_removal()
    gates["gate0_inventory"] = inventory

    passed = (
        all(item["passed"] for item in static_checks.values())
        and pytest_run["passed"]
        and all(item["passed"] for item in gates.values())
    )
    payload = {
        "format": REPORT_FORMAT,
        "version": 1,
        "contract": CONTRACT,
        "fixture": str(FIXTURE),
        "scope": (
            "implementation gates only: no capability training, no dev results, "
            "no default-entry change, final stays sealed"
        ),
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
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
