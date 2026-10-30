"""Audit the single-mainline plan contract.

The project previously had several documents that each looked like the current
mainline.  This guard makes the convergence enforceable:

* ``plans/MAINLINE.md`` exists and is the only direction source;
* only ``03_CURRENT_EXECUTION.md`` may declare a "current next step" heading;
* every document under ``plans/active`` carries a status tag in its first 10
  lines, and every one is listed in ``PLAN_INDEX.md``;
* the legacy reference documents that look like plans carry a legacy status
  banner, so a reader cannot mistake them for current instructions.

Usage:
    python scripts/training/audit_plan_single_mainline.py
    python scripts/training/audit_plan_single_mainline.py --json
Exit code: 0 = contract holds; 1 = at least one violation.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLANS = ROOT / "plans"
CURRENT_EXECUTION = "03_CURRENT_EXECUTION.md"
ALLOWED_STATUS = {
    "主线",
    "状态",
    "工作包",
    "合同",
    "运营",
    "规范",
    "组件",
    "历史",
    "证据",
    "索引",
}
STATUS_RE = re.compile(r">\s*文档状态：\[([^\]]+)\]")
QUEUE_RE = re.compile(r"^#+\s*当前唯一下一步", re.MULTILINE)

# Reference documents that historically read like the current plan.  They must
# say what they are, or a future reader will follow a stale queue again.
LEGACY_BANNER_REQUIRED = (
    "plans/reference/VISION_FUTURE_TECHNOLOGY.md",
    "plans/reference/PLAN-N1-177_joint_cognition_slice_20261026.md",
    "plans/reference/SPEC-N1-177_E1_state_consumption_contract_20261029.md",
    "plans/reference/SPEC-N1-177_F_value_encoder_20261029.md",
    "plans/reference/SPEC-N1-177_F1_freeze_proposal_20261029.md",
    "plans/reference/SPEC-N1-178_parse_coverage_extension_proposal_20261030.md",
    "plans/reference/PLAN-N1-176_foundation_curriculum_20261026.md",
    "plans/reference/SPEC-N1-176_F0_F1_pilot_20261026.md",
    "plans/reference/PLAN-N1-175_evaluation_handoff_20261026.md",
    "plans/reference/delivery-runs/20261021-n-mainline/artifacts/14-memory-gated-fast-resume.md",
)

CURRENT_CONTRACTS = (
    "plans/reference/PLAN-NATIVE-CONTENT-CORE_REDESIGN_20261030.md",
    "plans/reference/SPEC-NATIVE-CONTENT-CORE_R1_SEMANTIC_CONTRACT_20261030.md",
    "plans/reference/SPEC-NATIVE-CONTENT-CORE_R2_CANDIDATE_COMPARISON_20261030.md",
    "plans/reference/SPEC-NATIVE-CONTENT-CORE_R3_ISOLATED_IMPLEMENTATION_20261030.md",
)


def _head(text: str, limit: int = 10) -> str:
    return "\n".join(text.splitlines()[:limit])


def audit(root: Path = ROOT) -> dict:
    plans = root / "plans"
    failures: list[str] = []
    checks: list[str] = []

    mainline = plans / "MAINLINE.md"
    if not mainline.exists():
        failures.append("plans/MAINLINE.md is missing")
    else:
        text = mainline.read_text(encoding="utf-8")
        for needle in ("唯一主线", "R3b", "唯一执行队列"):
            if needle not in text:
                failures.append(f"MAINLINE.md lacks required marker: {needle}")
        checks.append("MAINLINE.md present")

    readme = plans / "README.md"
    if not readme.exists():
        failures.append("plans/README.md is missing")
    else:
        text = readme.read_text(encoding="utf-8")
        if "MAINLINE.md" not in text:
            failures.append("plans/README.md does not link to MAINLINE.md")
        checks.append("README entry present")

    index = plans / "active/PLAN_INDEX.md"
    index_text = index.read_text(encoding="utf-8") if index.exists() else ""
    if not index.exists():
        failures.append("plans/active/PLAN_INDEX.md is missing")

    active_docs = sorted((plans / "active").rglob("*.md"))
    queue_docs: list[str] = []
    for doc in active_docs:
        relative = doc.relative_to(root).as_posix()
        text = doc.read_text(encoding="utf-8")
        match = STATUS_RE.search(_head(text))
        if not match:
            failures.append(f"{relative}: missing `> 文档状态：[...]` in first 10 lines")
        elif match.group(1).strip() not in ALLOWED_STATUS:
            failures.append(f"{relative}: status tag `{match.group(1)}` is not allowed")
        if QUEUE_RE.search(text):
            queue_docs.append(relative)
            if doc.name != CURRENT_EXECUTION:
                failures.append(f"{relative}: only {CURRENT_EXECUTION} may declare the queue head")
        if doc.name != "PLAN_INDEX.md" and doc.name not in index_text:
            failures.append(f"{relative}: not listed in PLAN_INDEX.md")
    checks.append(f"active docs checked: {len(active_docs)}")

    for relative in CURRENT_CONTRACTS:
        doc = root / relative
        if not doc.exists():
            failures.append(f"{relative}: current contract is missing")
            continue
        if QUEUE_RE.search(doc.read_text(encoding="utf-8")):
            failures.append(f"{relative}: current contract must not declare a queue head")
    checks.append(f"current contracts checked: {len(CURRENT_CONTRACTS)}")

    if len(queue_docs) != 1 or Path(queue_docs[0]).name != CURRENT_EXECUTION:
        failures.append(f"queue head count is {len(queue_docs)}: {queue_docs}")

    reference_readme = plans / "reference/README.md"
    legacy_index = plans / "reference/LEGACY_ASSET_INDEX.md"
    if not legacy_index.exists():
        failures.append("plans/reference/LEGACY_ASSET_INDEX.md is missing")
    if reference_readme.exists() and "LEGACY_ASSET_INDEX" not in reference_readme.read_text(
        encoding="utf-8"
    ):
        failures.append("reference/README.md does not link to LEGACY_ASSET_INDEX.md")

    for relative in LEGACY_BANNER_REQUIRED:
        doc = root / relative
        if not doc.exists():
            failures.append(f"{relative}: legacy document is missing")
            continue
        match = STATUS_RE.search(_head(doc.read_text(encoding="utf-8")))
        if not match:
            failures.append(f"{relative}: legacy document lacks a status banner")
        elif match.group(1).strip() not in ALLOWED_STATUS:
            failures.append(f"{relative}: legacy banner `{match.group(1)}` is not allowed")
    checks.append(f"legacy banners checked: {len(LEGACY_BANNER_REQUIRED)}")

    return {
        "schema": "taiji-plan-single-mainline-audit-v1",
        "status": "ok" if not failures else "failed",
        "checks": checks,
        "failures": failures,
        "rc": 0 if not failures else 1,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()
    result = audit()
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"status: {result['status']}")
        for failure in result["failures"]:
            print(f"- {failure}")
    return int(result["rc"])


if __name__ == "__main__":
    raise SystemExit(main())
