"""`audit_taiji_debt_table_column_shape.py` 的契约册：分布要能为真也能为假。

四支要紧的：
* 混合列形必须被**分开数**（一行 5 列、一行 4 列时两者都要出现，不许平均成一个数）；
* 表头声明**不是多数形状**时那个布尔必须为假（现状就是假：表头四列、多数是五列）；
* 行内含额外竖线（把一列切成两列）要能被抓出来，且**行首竖线不足**的行不能冒充合法行；
* 零行必须响亮失败——"普查仪报了 0 行"与"债表已规范化"是两件事。
"""

from __future__ import annotations

import importlib.util
import json
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


UNIFORM_TABLE = (
    "| 编号 | 缺陷或事项 | 影响 | 修法或处置约束 | 状态 |\n"
    "|---|---|---|---|---|\n"
    "| DEBT-G1 | a | b | c | d |\n"
    "| DEBT-G2 | e | f | g | h |\n"
)


def _write(tmp_path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def test_main_greens_on_a_uniform_five_cell_table(tmp_path: Path) -> None:
    module = _load()
    table = _write(tmp_path, "uniform.md", UNIFORM_TABLE)
    report = tmp_path / "uniform.json"
    assert module.main(["--table", str(table), "--out-report", str(report)]) == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert all(value is True for value in payload["verdicts"].values()), payload["verdicts"]


def test_main_reds_when_one_row_breaks_the_shape(tmp_path: Path) -> None:
    """DEBT-G119 的红支：以前这台仪器第 108 行恒 `return 0`，红不出来。"""

    module = _load()
    broken = UNIFORM_TABLE.replace("| DEBT-G2 | e | f | g | h |\n", "| DEBT-G2 | e | f | g |\n")
    table = _write(tmp_path, "broken.md", broken)
    report = tmp_path / "broken.json"
    #: 形状那条是**分级红**：不开 `--enforce-shape` 时它只出版读数，不开红。
    assert module.main(["--table", str(table), "--out-report", str(report)]) == 0
    assert json.loads(report.read_text(encoding="utf-8"))["rows_not_expected_cells"] == 1
    report2 = tmp_path / "broken2.json"
    assert (
        module.main(
            [
                "--table",
                str(table),
                "--out-report",
                str(report2),
                "--expect-cells",
                "5",
                "--enforce-shape",
            ]
        )
        == 1
    )
    payload = json.loads(report.read_text(encoding="utf-8"))
    #: 红了也照样出版读数——不许"红就不留证据"。
    assert payload["verdicts"]["all_rows_are_the_expected_shape"] is False
    assert payload["rows_not_expected_cells"] == 1


def test_main_reds_on_a_duplicated_debt_id(tmp_path: Path) -> None:
    module = _load()
    dup = UNIFORM_TABLE.replace("DEBT-G2", "DEBT-G1")
    table = _write(tmp_path, "dup.md", dup)
    report = tmp_path / "dup.json"
    assert module.main(["--table", str(table), "--out-report", str(report)]) == 1
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["verdicts"]["ids_unique"] is False
    assert payload["id_duplicates"] == {"1": 2}


def test_main_refuses_a_nonpositive_target_shape(tmp_path: Path) -> None:
    module = _load()
    table = _write(tmp_path, "flat.md", UNIFORM_TABLE)
    report = tmp_path / "flat.json"
    assert (
        module.main(["--table", str(table), "--out-report", str(report), "--expect-cells", "0"])
        == 2
    )


def test_real_register_still_needs_the_shape_ruling_executed() -> None:
    """债表今天还没并成一型——这台仪器必须把这件事**报出来**而不是放过。

    ㊵-748 的实测：122 行里转义敏感列数为 5 的只有 81 行，其余仍是旧四列形；
    重排器 `unify_taiji_debt_table_columns.py` 能机械解决其中绝大多数，
    但 G41／G42／G43 三行的格子里有裸竖线，需要人工决定归哪一列（登 DEBT-G123）。
    所以这条测的是"分布被如实出版"，不是"已经统一"。
    """

    module = _load()
    report = module.measure(PROJECT_ROOT / "plans/active/roadmap/05_TECH_DEBT_REGISTER.md")
    assert report["rows"] >= 100
    assert report["distinct_ids"] == report["rows"], report["id_duplicates"]
    assert report["rows_not_expected_cells"] > 0, report["cells_per_row_escape_aware"]
    assert report["verdicts"]["all_rows_are_the_expected_shape"] is False
    assert report["cells_per_row_escape_aware"].keys() >= {"4", "5"}
