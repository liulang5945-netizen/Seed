"""VISION 条款承载「存在性」红门（DEBT-G97 修法③，owner #25 乙裁：接成门）。

这条门只有一种实质主张：**每一条"现在必须可核验"的条款都必须有承载件指到它**——
要么按条款号被引用（`V-17` 字样），要么按 VISION 章节锚被引用（`§5.3` 字样）。缺任一档就是红。

**它故意不守的两件事**（写在这里，防止后来者把门越修越像"覆盖率"）：

* **不守命中数**：一份件里多写几次条款号不会让门更绿，少写几次也不会让门更红。
  命中数是自证的产物——㊵-675/679/680 三次都看到"补登记"直接把 id／锚命中往上推，
  门若守数量就会被登记文本自己喂饱（同族教训＝"已有门守着的关系别自己造裁判"）。
* **不守 id 级**：只有章节锚命中（`section_only`）也算通过。本仓今天 id 级只有 7/19，
  把 id 级写进门等于宣布一条不存在的进展。

**第 6 支钉的是仪器既有的一条拒绝，也是我这轮判断错的地方**：我先以为零条款行的表会被
空集合读成「零违例＝绿」，还给仪器补了一条 raise；单跑时那支测红了，报的是仪器自己的消息——
`_parse_table` 末尾本来就会为「一行条款都没有」抛错，我补的那条永远走不到、已删。
本册把这条既有拒绝钉成契约（此前没有任何测钉过它），教训写在这里：读函数要读到 return，
别凭半段就断言它缺守卫。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "audit_taiji_vision_clause_traceability.py"
PLAN_DOC = REPO / "plans" / "active" / "roadmap" / "09_NEXT_MAINLINE_PLAN.md"
REFERENCE_DIR = REPO / "plans" / "reference"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("n6_05_clause_existence_gate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = _load()

DOC_TWO_KINDS = """# 对账表

| 条款 | 内容（关键词） | 章节 | 归属 | 当前承载 |
|---|---|---|---|---|
| V-01 | 按条款号有登记节 | §3.1 | **N1** | 已展开 |
| V-02 | 只有章节锚 | §5.2 | **N1** | 已展开 |
| V-03 | 两种都没有 | §7.1 | **N1** | 待展开 |
"""

PLAN_WITH_ALL = "# PLAN-N1-00\n\n这条把 V-01 展开了，另有整节对着 §5.2 与 §7.1 写实现设计。\n"
PLAN_MISSING_ONE = "# PLAN-N1-00\n\n这条把 V-01 展开了，另有整节对着 §5.2 写实现设计。\n"


def _tree(tmp_path: Path, plan_text: str) -> tuple[Path, Path]:
    doc = tmp_path / "09.md"
    doc.write_text(DOC_TWO_KINDS, encoding="utf-8", newline="\n")
    ref = tmp_path / "reference"
    ref.mkdir(parents=True, exist_ok=True)
    (ref / "PLAN-N1-00_x_20261011.md").write_text(plan_text, encoding="utf-8", newline="\n")
    return doc, ref


def _violations(payload: dict[str, Any]) -> list[str]:
    """门唯一的判定：必须项里有没有"两种引用都没有"的条款。"""
    return sorted(
        str(one["clause"])
        for one in payload["rows"]
        if one.get("required_now") and one.get("traceability") == "unmapped"
    )


def test_the_real_tree_has_no_uncovered_required_clause() -> None:
    payload = MODULE.audit(PLAN_DOC, REFERENCE_DIR)
    assert payload["clause_rows_parsed"] >= 1, payload
    assert payload["clauses_required_now"] >= 1, payload
    assert payload["verdicts"]["table_parsed_every_row_has_clause_id"] is True, payload["verdicts"]
    assert _violations(payload) == [], _violations(payload)
    assert payload["clauses_unmapped_both"] == len(_violations(payload))


def test_a_clause_losing_both_citations_turns_the_gate_red(tmp_path: Path) -> None:
    doc, ref = _tree(tmp_path, PLAN_MISSING_ONE)
    payload = MODULE.audit(doc, ref)
    assert _violations(payload) == ["V-03"], payload["unmapped_both_list"]
    assert payload["verdicts"]["every_required_clause_has_any_citation"] is False


def test_a_section_only_clause_passes_so_the_gate_is_not_id_level(tmp_path: Path) -> None:
    doc, ref = _tree(tmp_path, PLAN_WITH_ALL)
    payload = MODULE.audit(doc, ref)
    rows = {one["clause"]: one for one in payload["rows"]}
    assert rows["V-02"]["traceability"] == "section_only", rows["V-02"]
    assert _violations(payload) == [], _violations(payload)


def test_extra_citations_do_not_change_the_verdict(tmp_path: Path) -> None:
    """同一棵树里把 `V-01` 多写几次，门的结果必须一字不变（守存在性，不守数量）。"""
    doc, ref = _tree(tmp_path, PLAN_MISSING_ONE)
    before = MODULE.audit(doc, ref)
    plan = next(ref.glob("PLAN-N1-00_*.md"))
    plan.write_text(
        PLAN_MISSING_ONE + "\nV-01 V-01 V-01 V-01\n",
        encoding="utf-8",
        newline="\n",
    )
    after = MODULE.audit(doc, ref)
    assert _violations(before) == _violations(after) == ["V-03"]
    assert before["clauses_with_id_citation"] == after["clauses_with_id_citation"] == 1


def test_short_row_is_refused(tmp_path: Path) -> None:
    doc = tmp_path / "09.md"
    doc.write_text(
        "# 对账表\n\n| V-01 | 竖线不足 | §3.1 | **N1** |\n",
        encoding="utf-8",
        newline="\n",
    )
    ref = tmp_path / "reference"
    ref.mkdir(parents=True, exist_ok=True)
    try:
        MODULE.audit(doc, ref)
    except ValueError as error:
        assert "竖线不足" in str(error), str(error)
    else:
        raise AssertionError("a short row must be refused loudly")


def test_an_empty_table_is_refused_instead_of_reading_green(tmp_path: Path) -> None:
    """零条款行不是"零违例"。这一支钉的是仪器补上的 fail-closed 分支。"""
    doc = tmp_path / "09.md"
    doc.write_text("# 对账表\n\n（还没有行）\n", encoding="utf-8", newline="\n")
    ref = tmp_path / "reference"
    ref.mkdir(parents=True, exist_ok=True)
    try:
        MODULE.audit(doc, ref)
    except ValueError as error:
        assert "对账表为空" in str(error), str(error)
    else:
        raise AssertionError("an empty table must be refused, not read as an empty violation list")
