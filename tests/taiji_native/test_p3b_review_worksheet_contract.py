"""Contract guard for the CAP-0 human-review worksheet exporter.

J3's "B 经人工复核达标" branch cannot be computed by any instrument: B/G items come back
``score = null`` with ``pending_human_review = true``.  The exporter's whole job is therefore to
produce a sheet that makes the *absence* of review visible, so the honest properties tested here
are about completeness and about not judging:

* only genuinely pending items are exported, and every exported item has a blank score slot;
* items skipped because their answer did not change are still listed by id -- "not re-reviewed"
  must never be readable as "reviewed and unchanged";
* a machine-scored item never leaks into the sheet;
* the sheet itself assigns no grade, no verdict and no criterion outcome;
* the default output is a worksheet name, never a sealed report.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
WORKSHEET = REPO / "scripts" / "training" / "make_p3b_review_worksheet.py"


def _load(name: str, path: Path) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    for entry in (str(path.parent), str(REPO)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sheet() -> Any:
    return _load("_p3b_review_worksheet_under_test", WORKSHEET)


def _item(item_id: str, answer: str, *, pending: bool = True) -> dict[str, Any]:
    return {
        "id": item_id,
        "family": "self_description" if item_id.startswith("B") else "unknown_info",
        "tick": 16_000_000,
        "turns": [{"prompt": f"问 {item_id}", "raw_output": answer}],
        "raw_last_output": answer,
        "score": None if pending else 1,
        "pending_human_review": pending,
        "machine_precheck": {"machine_verdict": "precheck_skipped", "reason": "无对应机检规则"},
    }


def _report(b_items: list[dict[str, Any]], g_items: list[dict[str, Any]]) -> dict[str, Any]:
    def dimension(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "items": items,
            "tally": {"pending_human_review_items": sum(i["pending_human_review"] for i in items)},
        }

    return {
        "checkpoint": "checkpoints/p3b/seed_aligned.pt",
        "dimensions": {"B": dimension(b_items), "G": dimension(g_items)},
    }


def test_only_pending_items_are_exported_with_a_score_slot(sheet: Any) -> None:
    report = _report(
        [_item("B01", "甲"), _item("B02", "乙", pending=False)],
        [_item("G01", "丙")],
    )
    text = sheet.build_worksheet(report, label="unit")
    assert "### B01" in text and "### G01" in text
    assert "### B02" not in text, "a machine-scored item is not for human review"
    assert text.count("**评分（0/1/2）**") == 2
    assert "本题表需打分：**2** 题" in text


def test_unchanged_answers_are_skipped_but_still_listed(sheet: Any) -> None:
    before = _report([_item("B01", "甲"), _item("B02", "乙")], [_item("G01", "丙")])
    after = _report([_item("B01", "甲"), _item("B02", "乙变了")], [_item("G01", "丙变了")])
    text = sheet.build_worksheet(after, label="tick 17M", changed_vs=before)
    assert "### B01" not in text, "same answer, nothing new to review"
    assert "### B02" in text and "### G01" in text
    assert "B/B01" in text, "a skipped item must be listed, not silently dropped"
    assert "不等于已复核" in text
    assert "本题表需打分：**2** 题" in text
    assert "待复核总数（含被跳过的）：**3** 题" in text


def test_the_sheet_never_grades_anything(sheet: Any) -> None:
    report = _report([_item("B01", "甲")], [])
    text = sheet.build_worksheet(report, label="unit")
    for invented in ("PASS", "FAIL", "达标", "J3", "判据通过"):
        assert invented not in text, invented
    assert "未复核的题不得记为通过" in text, "the sheet must state its own limit"


def test_a_dimension_without_items_is_reported_as_such(sheet: Any) -> None:
    text = sheet.build_worksheet(_report([], []), label="empty")
    assert "本题表需打分：**0** 题" in text
    assert "待复核维度" in text


def test_default_output_is_a_worksheet_not_a_sealed_report(sheet: Any) -> None:
    assert sheet.DEFAULT_OUTPUT.parent == REPO / "reports"
    name = sheet.DEFAULT_OUTPUT.name
    assert name.startswith("p3b_review_worksheet")
    assert "baseline" not in name, "the sealed P3a report must never be the destination"
