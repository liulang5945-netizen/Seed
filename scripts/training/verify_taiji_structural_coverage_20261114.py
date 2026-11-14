"""Guard for the structural goal coverage review (D-20261114-05).

Re-reads the coverage receipt, the review document and the three headline
documents. Recomputes every count from the rows instead of trusting the summary.
No model, forward pass, optimizer step or sealed face.
"""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import pathlib
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
RECEIPT = ROOT / "reports/taiji_u2_structural_coverage_20261114.json"
DOC = ROOT / "plans/reference/REVIEW-TAIJI_STRUCTURAL_GOAL_COVERAGE_20261114.md"
OUT = ROOT / "reports/taiji_u2_structural_coverage_guard_20261114.json"
HEADLINE_MARK = "**队首（唯一）**："
HEADLINE_FILES = (
    ROOT / "plans/MAINLINE.md",
    ROOT / "plans/MASTER-PLAN.md",
    ROOT / "plans/reference/ROADMAP-NEXT_20261104.md",
)
HEAD_MARKERS = ("P3.31", "22.6")  # navigation probe only; update when the head advances
STATUSES = ("capability", "engineering", "structural", "blocked", "absent")
EXPECTED_TOTAL = 19
FORBIDDEN = ("B 胜出", "能力成立 1／", "已切换默认图", "路线B已证明")
DOSE_MARKERS = ("24", "256", "1844", "7.2")


def headline_of(path: pathlib.Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        index = line.find(HEADLINE_MARK)
        if index >= 0:
            return line[index + len(HEADLINE_MARK) :].strip()
    return ""


def summary_from(counts: dict[str, int]) -> str:
    return (
        f"覆盖合计（{counts['total']} 项）：能力成立 {counts['capability']}／"
        f"工程可运行 {counts['engineering']}／仅结构 {counts['structural']}／"
        f"阻塞于能力父 {counts['blocked']}／缺席 {counts['absent']}。"
    )


def evaluate(receipt: dict, doc: str, headlines: list[str]) -> dict[str, bool]:
    rows = receipt.get("rows", [])
    tally = collections.Counter(row.get("status") for row in rows)
    computed = {status: int(tally.get(status, 0)) for status in STATUSES}
    computed["total"] = len(rows)
    declared = receipt.get("counts", {})

    checks: dict[str, bool] = {}
    checks["rows_have_unique_ids"] = len({row.get("id") for row in rows}) == len(rows)
    checks["row_status_vocabulary"] = all(row.get("status") in STATUSES for row in rows)
    checks["total_is_19"] = computed["total"] == EXPECTED_TOTAL
    checks["declared_counts_match_rows"] = all(
        declared.get(key) == value for key, value in computed.items()
    )
    checks["capability_count_is_zero"] = computed["capability"] == 0
    checks["summary_line_matches_counts"] = (
        receipt.get("summary_line") == summary_from(computed) and summary_from(computed) in doc
    )
    checks["document_contains_every_row_id"] = all(f"| {row.get('id')} |" in doc for row in rows)
    checks["verdicts_are_expected"] = (
        receipt.get("verdict") == "coverage_review_recorded"
        and receipt.get("route_verdict") == "unproven_within_scope"
        and receipt.get("parent_verdict") == "capability_parent_absent"
    )
    parent = receipt.get("parent_qualification", {})
    checks["capability_parent_absent"] = (
        parent.get("capability_parent") is None and parent.get("native_parent_qualified") is False
    )
    gates = parent.get("gates_released", {})
    checks["downstream_gates_not_released"] = (
        gates.get("22.6") is False and gates.get("22.7") is False
    )
    checks["default_graph_not_switched"] = parent.get("production_default_graph_switched") is False
    checks["dose_ladder_documented"] = all(marker in doc for marker in DOSE_MARKERS)
    checks["no_forbidden_claims"] = not any(text in doc for text in FORBIDDEN)
    checks["headlines_agree_and_name_head"] = (
        len(headlines) == len(HEADLINE_FILES)
        and len(set(headlines)) == 1
        and all(marker in headlines[0] for marker in HEAD_MARKERS)
    )
    return checks


def run_real_file_drill(receipt: dict) -> dict:
    """Mutate the document, expect the summary check to ring, restore byte-exact."""
    original = DOC.read_bytes()
    before_sha = hashlib.sha256(original).hexdigest()
    text = original.decode("utf-8")
    target = "能力成立 0／工程可运行 5"
    if target not in text:
        return {"performed": False, "reason": "anchor_missing"}
    mutated = text.replace(target, "能力成立 1／工程可运行 5", 1).encode("utf-8")
    try:
        DOC.write_bytes(mutated)
        red = evaluate(
            receipt, DOC.read_text(encoding="utf-8"), [headline_of(path) for path in HEADLINE_FILES]
        )
    finally:
        DOC.write_bytes(original)
    after_sha = hashlib.sha256(DOC.read_bytes()).hexdigest()
    green = evaluate(
        receipt, DOC.read_text(encoding="utf-8"), [headline_of(path) for path in HEADLINE_FILES]
    )
    return {
        "performed": True,
        "red_checks": sorted(name for name, ok in red.items() if not ok),
        "restored_byte_exact": before_sha == after_sha,
        "green_after_restore": all(green.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-drills", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    started, wall = time.process_time(), time.perf_counter()
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    doc = DOC.read_text(encoding="utf-8")
    headlines = [headline_of(path) for path in HEADLINE_FILES]
    checks = evaluate(receipt, doc, headlines)

    drills: dict[str, bool] = {}
    if not args.skip_drills:
        mutated = copy.deepcopy(receipt)
        mutated["rows"].append(
            {
                "id": "FAKE-1",
                "clause": "injected",
                "owner": "injected",
                "status": "capability",
                "evidence": "injected",
            }
        )
        drills["inject_capability_claim"] = not evaluate(mutated, doc, headlines)[
            "capability_count_is_zero"
        ]
        mutated = copy.deepcopy(receipt)
        mutated["rows"].pop()
        drills["drop_one_row"] = not evaluate(mutated, doc, headlines)["total_is_19"]
        mutated = copy.deepcopy(receipt)
        mutated["parent_qualification"]["gates_released"]["22.6"] = True
        drills["release_22_6_gate"] = not evaluate(mutated, doc, headlines)[
            "downstream_gates_not_released"
        ]
        mutated = copy.deepcopy(receipt)
        mutated["parent_qualification"]["native_parent_qualified"] = True
        drills["claim_capability_parent"] = not evaluate(mutated, doc, headlines)[
            "capability_parent_absent"
        ]
        mutated = copy.deepcopy(receipt)
        mutated["summary_line"] = "覆盖合计（19 项）：能力成立 0／工程可运行 19。"
        drills["tamper_summary_line"] = not evaluate(mutated, doc, headlines)[
            "summary_line_matches_counts"
        ]

    real = run_real_file_drill(receipt) if not args.skip_drills else {"performed": False}
    drills_ok = all(drills.values()) and (real.get("performed") is False or real["performed"])
    if real.get("performed"):
        drills_ok = (
            drills_ok
            and bool(real["red_checks"])
            and real["restored_byte_exact"]
            and real["green_after_restore"]
        )

    verdict = "pass" if all(checks.values()) and drills_ok else "fail"
    payload = {
        "schema": "taiji-u2-structural-coverage-guard-v1",
        "date": "2026-11-14",
        "card": "22.5.P3.9.5",
        "checks": checks,
        "red_drills": drills,
        "real_file_drill": real,
        "verdict": verdict,
        "failed": sorted(name for name, ok in checks.items() if not ok),
        "measured": {
            "cpu_seconds": round(time.process_time() - started, 6),
            "wall_seconds": round(time.perf_counter() - wall, 6),
        },
    }
    if not args.no_write:
        OUT.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    print(
        json.dumps(
            {
                "verdict": verdict,
                "failed": payload["failed"],
                "drills": drills,
                "real": real,
                "measured": payload["measured"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
