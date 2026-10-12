"""VISION 条款对账表的可核验性普查仪（`audit_taiji_vision_clause_traceability.py`，DEBT-G97）的契约测。

这台仪器只有一处实质主张：**三档可核验性要分得开，而"现在必须可核验"的分母不能被备选条款污染**。
（后者不是假想——第一版我就是用 `归属里含 N5` 当分母，把 `备选（N5 后若行动线开启再评）` 两行
算进了"必须现在覆盖"，于是 unmapped 从 7 被虚报成 9。本册第 1 支就是这个错的负对照。）

* `id`／`section_only`／`unmapped`／`n/a` 四档必须由同一条读数分开，且各能单独为假；
* 紧凑写法 `V-08/11/13` 里的第 3 条要能被认成命中（否则分母会假装少一条）；
* 表解析不了（竖线不足／一行条款号都没有）必须 rc=2 **且不落读数件**；
* 真树那一支只钉结构性事实（行数、每行有条款号、`id 命中数 ≤ 必须数`），
  **不钉覆盖率数值**——覆盖率是要往上爬的，钉死它就是把一次读数冒充成契约。
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "audit_taiji_vision_clause_traceability.py"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("n6_04_vision_clause_traceability", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = _load()

DOC_FOUR_CLASSES = """# 对账表

| 条款 | 内容（关键词） | 章节 | 归属 | 当前承载 |
|---|---|---|---|---|
| V-01 | 被条款号引用 | §3.1 | **N1** | 已展开 |
| V-02 | 只被章节锚引用 | §5.2 | **N1** | 已展开 |
| V-03 | 两种引用都没有 | §7.1 | **N1** | 待展开 |
| V-04 | 行动线的备选条款 | §9.9 | 备选（N1 后若行动线开启再评） | — |
"""

PLAN_CITING = """# PLAN-N1-00

这条把 V-01 展开了。另有一节对着 §5.2 写实现设计。
"""


def _tree(tmp_path: Path, doc_text: str, plan_text: str = PLAN_CITING) -> tuple[Path, Path]:
    doc = tmp_path / "09.md"
    doc.write_text(doc_text, encoding="utf-8", newline="\n")
    ref = tmp_path / "reference"
    ref.mkdir(parents=True, exist_ok=True)
    (ref / "PLAN-N1-00_x_20261011.md").write_text(plan_text, encoding="utf-8", newline="\n")
    return doc, ref


def _rows(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {one["clause"]: one for one in payload["rows"]}


def test_the_four_traceability_classes_are_distinguished(tmp_path: Path) -> None:
    doc, ref = _tree(tmp_path, DOC_FOUR_CLASSES)
    payload = MODULE.audit(doc, ref)
    rows = _rows(payload)
    assert payload["clause_rows_parsed"] == 4, payload
    assert payload["clauses_required_now"] == 3, payload
    assert payload["clauses_with_id_citation"] == 1, payload
    assert payload["clauses_with_section_citation"] == 1, payload
    assert rows["V-01"]["traceability"] == "id", rows["V-01"]
    assert rows["V-02"]["traceability"] == "section_only", rows["V-02"]
    assert rows["V-03"]["traceability"] == "unmapped", rows["V-03"]
    assert payload["unmapped_both_list"] == ["V-03"], payload["unmapped_both_list"]
    #: 这一行就是第一版的错：归属里写着 N1，但它是**备选**。分母必须不含它。
    assert rows["V-04"]["required_now"] is False, rows["V-04"]
    assert rows["V-04"]["traceability"] == "n/a", rows["V-04"]
    assert payload["verdicts"]["every_required_clause_has_any_citation"] is False


def test_a_compact_clause_list_still_counts_as_an_id_citation(tmp_path: Path) -> None:
    """`V-08/11/13` 这种紧凑写法里，第 3 条也是命中——否则分母会假装少一条。"""

    doc_text = (
        "# 对账表\n\n| 条款 | 内容（关键词） | 章节 | 归属 | 当前承载 |\n"
        "|---|---|---|---|---|\n| V-13 | 紧凑列表末位 | §4.4 | **N2** | 已展开 |\n"
    )
    doc = tmp_path / "09.md"
    doc.write_text(doc_text, encoding="utf-8", newline="\n")
    ref = tmp_path / "reference"
    ref.mkdir(parents=True, exist_ok=True)
    (ref / "PLAN-N2-01_y_20261011.md").write_text(
        "本件消化 V-08/11/13 三条。\n", encoding="utf-8", newline="\n"
    )
    payload = MODULE.audit(doc, ref)
    row = _rows(payload)["V-13"]
    assert row["id_citations"] == 1, row
    assert row["traceability"] == "id", row


def test_a_malformed_row_refuses_and_writes_no_artifact(tmp_path: Path) -> None:
    doc, ref = _tree(tmp_path, "| V-05 | 只有三格\n")
    out = tmp_path / "out.json"
    rc = MODULE.main(["--doc", str(doc), "--reference-dir", str(ref), "--out-report", str(out)])
    assert rc == 2, rc
    assert not out.exists(), "拒绝路径不许留下读数件"


def test_an_empty_table_refuses_instead_of_reporting_zero_rows(tmp_path: Path) -> None:
    """零行不是"全覆盖"：表里一枚条款号都没有 ⇒ 响亮失败（同 `扫面为空` 那一族）。"""

    doc, ref = _tree(tmp_path, "# 这里一张表都没有\n")
    out = tmp_path / "out.json"
    rc = MODULE.main(["--doc", str(doc), "--reference-dir", str(ref), "--out-report", str(out)])
    assert rc == 2, rc
    assert not out.exists(), "拒绝路径不许留下读数件"


def test_the_real_table_is_parsed_without_pinning_coverage(tmp_path: Path) -> None:
    """真树只钉结构性事实，覆盖率留给改进——但必须**扫到东西**，零命中不算读数。"""

    payload = MODULE.audit(REPO / "plans/active/roadmap/09_NEXT_MAINLINE_PLAN.md")
    assert payload["status"] == "ok", payload
    assert payload["clause_rows_parsed"] == len(payload["rows"]), payload
    assert payload["clause_rows_parsed"] >= 30, payload["clause_rows_parsed"]
    assert payload["clauses_required_now"] >= 10, payload["clauses_required_now"]
    assert all(one["clause"].startswith("V-") for one in payload["rows"]), payload
    assert payload["clauses_with_id_citation"] <= payload["clauses_required_now"], payload
    assert payload["verdicts"]["table_parsed_every_row_has_clause_id"] is True
    #: 未映射清单必须是**必须格**的子集，备选/出口后那类不许混进来。
    required = {one["clause"] for one in payload["rows"] if one["required_now"]}
    assert set(payload["unmapped_both_list"]) <= required, payload["unmapped_both_list"]


def test_help_names_every_flag() -> None:
    proc = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, check=False)
    text = proc.stdout.decode("utf-8", errors="replace")
    assert proc.stdout is not None, proc.returncode
    assert proc.returncode == 0, (proc.returncode, text[-300:])
    for flag in ("--doc", "--reference-dir", "--out-report"):
        assert flag in text, (flag, text[-300:])
