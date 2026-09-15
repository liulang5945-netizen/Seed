"""Contract guard for the CAP-0 human-review worksheet exporter.

Why this exists: J3's "B 经人工复核达标" branch cannot be computed by any instrument -- B/G items
come back ``score = null`` with ``pending_human_review = true``.  The exporter's whole job is
therefore to make the **absence** of review visible, so the honest properties tested here are
about completeness, about never pre-filling a grade, and about not judging:

* only genuinely pending items are exported, and every exported item's score slot is **empty**;
* a machine precheck verdict may be echoed as a hint, but never lands in a score position;
* items skipped because their answer did not change are still listed by id -- "not re-reviewed"
  must never be readable as "reviewed and unchanged";
* per-dimension pending counts are printed, so an empty dimension cannot look like a full sheet;
* the sheet itself assigns no grade, no verdict and no criterion outcome;
* the default output is a worksheet name, never the sealed baseline it is derived from.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
WORKSHEET = REPO / "scripts" / "training" / "make_p3b_review_worksheet.py"
P3A_BASELINE = REPO / "reports" / "taiji_cap0_baseline_constrained_20260915.json"

#: The four values `_machine_precheck` in eval_taiji_cap0_baseline.py can actually emit.
REAL_PRECHECK_VERDICTS = ("precheck_skipped", "precheck_pass", "precheck_fail", "hard_safety_risk")
SCORE_SLOT = re.compile(r"\*\*评分（0/1/2）\*\*：(.*)")


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


DIALOGUE_TEXT = "老师：甲\n乙：乙\n"
META_ONLY_TEXT = "作者：佚名\n正文一段\n"
PLAIN_TEXT = "这是一段没有角色名的叙述文字\n"


def _row(text: str) -> str:
    return json.dumps({"text": text}, ensure_ascii=False) + "\n"


def _item(
    item_id: str,
    answer: str,
    *,
    pending: bool = True,
    verdict: str = "precheck_skipped",
) -> dict[str, Any]:
    return {
        "id": item_id,
        "family": "self_description" if item_id.startswith("B") else "unknown_info",
        "tick": 16_000_000,
        "turns": [{"prompt": f"问 {item_id}", "raw_output": answer}],
        "raw_last_output": answer,
        "score": None if pending else 1,
        "pending_human_review": pending,
        "machine_precheck": {"machine_verdict": verdict, "reason": "无对应机检规则"},
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


def _slots(text: str) -> list[str]:
    """Whatever the sheet put after each score label -- "" is the only acceptable answer."""

    return SCORE_SLOT.findall(text)


# --------------------------------------------------------------------------- #
# Completeness
# --------------------------------------------------------------------------- #


def test_only_pending_items_are_exported_with_an_empty_score_slot(sheet: Any) -> None:
    report = _report(
        [_item("B01", "甲"), _item("B02", "乙", pending=False)],
        [_item("G01", "丙")],
    )
    text = sheet.build_worksheet(report, label="unit")
    assert "### B01" in text and "### G01" in text
    assert "### B02" not in text, "a machine-scored item is not for human review"
    assert len(_slots(text)) == 2
    assert _slots(text) == ["", ""], "a pre-filled score is a fake review"
    assert "本题表需打分：**2** 题" in text


def test_per_dimension_counts_disclose_an_empty_dimension(sheet: Any) -> None:
    """Without counts the sheet looks the same whether a dimension has 20 items or none."""

    both = sheet.build_worksheet(
        _report([_item("B01", "甲"), _item("B02", "乙")], [_item("G01", "丙")]), label="x"
    )
    assert "每维待复核题数：B：2 题；G：1 题" in both
    empty = sheet.build_worksheet(_report([], []), label="empty")
    assert "每维待复核题数：B：0 题；G：0 题" in empty
    assert "本题表需打分：**0** 题" in empty


def test_unchanged_answers_are_skipped_but_still_listed(sheet: Any) -> None:
    before = _report([_item("B01", "甲"), _item("B02", "乙")], [_item("G01", "丙")])
    after = _report(
        [_item("B01", "甲"), _item("B02", "乙变了")],
        [_item("G01", "丙变了")],
    )
    text = sheet.build_worksheet(after, label="tick 17M", changed_vs=before)
    assert "### B01" not in text, "same answer, nothing new to review"
    assert "### B02" in text and "### G01" in text
    assert "B/B01" in text, "a skipped item must be listed, not silently dropped"
    assert "不等于已复核" in text
    assert "本题表需打分：**2** 题" in text
    assert "待复核总数（含被跳过的）：**3** 题" in text


# --------------------------------------------------------------------------- #
# The sheet must not grade, hint or otherwise decide
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("verdict", REAL_PRECHECK_VERDICTS)
def test_a_machine_verdict_is_echoed_as_a_hint_never_as_a_grade(sheet: Any, verdict: str) -> None:
    """The exporter prints `机检预判：<verdict>`; a ban list of words cannot catch the real values.

    So the discriminating assertion is positional: the verdict line carries the value, and the
    score line carries nothing, for every value the evaluator can produce.
    """

    text = sheet.build_worksheet(_report([_item("B01", "甲", verdict=verdict)], []), label="unit")
    assert f"机检预判：{verdict}" in text
    assert _slots(text) == [""], "the score position stays blank whatever the precheck says"
    assert f"评分（0/1/2）：{verdict}" not in text


def test_the_sheet_never_grades_anything(sheet: Any) -> None:
    text = sheet.build_worksheet(_report([_item("B01", "甲")], [_item("G01", "乙")]), label="unit")
    for invented in ("PASS", "FAIL", "达标", "J3", "判据通过", "合格"):
        assert invented not in text, invented
    assert _slots(text) == ["", ""]
    assert "未复核的题不得记为通过" in text, "the sheet must state its own limit"


def test_default_output_is_a_worksheet_not_a_sealed_report(sheet: Any) -> None:
    """Compare against the real paths, instead of banning one word in a filename."""

    forbidden = {
        P3A_BASELINE.resolve(),
        (REPO / "reports" / "taiji_p3b_campaign_treatment_20260915.json").resolve(),
        (REPO / "reports" / "taiji_p3b_campaign_control_20260915.json").resolve(),
        (REPO / "reports" / "taiji_p3b_criteria_check_20260915.json").resolve(),
        (REPO / "reports" / "taiji_p3b_training_run_treatment_20260915.json").resolve(),
    }
    assert sheet.DEFAULT_OUTPUT.parent == REPO / "reports"
    assert sheet.DEFAULT_OUTPUT.name.startswith("p3b_review_worksheet")
    assert sheet.DEFAULT_OUTPUT.resolve() not in forbidden
