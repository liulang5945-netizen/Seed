"""`audit_taiji_debt_table_column_shape.py` 的契约册：分布要能为真也能为假。

四支要紧的：
* 混合列形必须被**分开数**（一行 5 列、一行 4 列时两者都要出现，不许平均成一个数）；
* 表头声明**不是多数形状**时那个布尔必须为假（现状就是假：表头四列、多数是五列）；
* 行内含额外竖线（把一列切成两列）要能被抓出来，且**行首竖线不足**的行不能冒充合法行；
* 零行必须响亮失败——"普查仪报了 0 行"与"债表已规范化"是两件事。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "training" / "audit_taiji_debt_table_column_shape.py"

TABLE = """# 债表

| 编号 | 内容 | 严重性 | 处置约束 |
|---|---|---|---|
| DEBT-G1 | 四列行 | 低 | 处置 |
| DEBT-G2 | 五列行 | 低 | 处置 | 状态 |
| DEBT-G3 | 行内多一枚竖线 | 低 | 含 `a|b` 的处置 | 状态 |
| DEBT-G3 | 重复编号 | 低 | 处置 | 状态 |
"""


def _load():
    spec = importlib.util.spec_from_file_location("n5_26_column_shape", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_mixed_shapes_are_counted_separately(tmp_path: Path) -> None:
    module = _load()
    table = tmp_path / "debt.md"
    table.write_text(TABLE, encoding="utf-8")
    report = module.measure(table)
    assert report["rows"] == 4
    assert report["header_declared_cells"] == 4
    #: 含额外竖线的那一行被数成 6 格——这正是本仪器要暴露的形状（`a|b` 把一列切成两列）。
    assert report["cells_per_row"] == {"4": 1, "5": 2, "6": 1}
    assert report["rows_matching_declaration"] == 1
    assert report["rows_with_fewer_cells"] == 0
    assert report["rows_with_more_cells"] == 3
    assert report["max_debt_id"] == 3
    assert report["id_duplicates"] == {"3": 2}
    assert report["declared_is_the_majority_shape"] is False


def test_a_uniform_table_would_report_the_declaration_as_majority(tmp_path: Path) -> None:
    module = _load()
    table = tmp_path / "flat.md"
    table.write_text(
        "| 编号 | 内容 | 严重性 | 处置约束 |\n|---|---|---|---|\n"
        "| DEBT-G1 | a | b | c |\n| DEBT-G2 | d | e | f |\n",
        encoding="utf-8",
    )
    report = module.measure(table)
    assert report["declared_is_the_majority_shape"] is True
    assert report["rows_with_more_cells"] == 0
    assert report["rows_not_closed"] == 0


def test_rows_not_closed_with_a_pipe_are_counted(tmp_path: Path) -> None:
    module = _load()
    table = tmp_path / "unclosed.md"
    table.write_text(
        "| 编号 | 内容 | 严重性 | 处置约束 |\n|---|---|---|---|\n"
        "| DEBT-G1 | a | b | c |\n| DEBT-G2 | d | e | f （收尾竖线漏了\n",
        encoding="utf-8",
    )
    report = module.measure(table)
    assert report["rows_not_closed"] == 1
    assert report["rows_closed_with_pipe"] == 1


def test_zero_rows_fail_loudly(tmp_path: Path) -> None:
    module = _load()
    table = tmp_path / "empty.md"
    table.write_text("# 只有标题\n\n没有债行。\n", encoding="utf-8")
    try:
        module.measure(table)
    except ValueError as exc:
        assert "零行" in str(exc)
    else:
        raise AssertionError("零行必须响亮失败，不许读成『债表已规范化』")


def test_missing_header_is_refused(tmp_path: Path) -> None:
    module = _load()
    table = tmp_path / "noheader.md"
    table.write_text("| DEBT-G1 | a | b | c |\n", encoding="utf-8")
    try:
        module.measure(table)
    except ValueError as exc:
        assert "表头" in str(exc)
    else:
        raise AssertionError("没有表头就无法对照声明列数，必须拒绝出值")


def test_real_register_reports_a_shape_that_differs_from_the_header() -> None:
    module = _load()
    report = module.measure(PROJECT_ROOT / "plans/active/roadmap/05_TECH_DEBT_REGISTER.md")
    assert report["rows"] > 50
    assert report["max_debt_id"] >= 109
    assert report["distinct_ids"] == report["rows"], report["id_duplicates"]
    assert report["cells_per_row"].keys() >= {"4", "5"}
    # 这条现在为假是真实现状（表头四列、多数行五列）；将来若 owner 并成一型，它会翻真。
    assert isinstance(report["declared_is_the_majority_shape"], bool)
