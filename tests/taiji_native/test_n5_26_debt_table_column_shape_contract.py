"""`audit_taiji_debt_table_column_shape.py` 的契约册：分布要能为真也能为假。

五支要紧的：
* 混合列形必须被**分开数**（一行 5 列、一行 4 列时两者都要出现，不许平均成一个数）；
* 表头声明**是不是多数形状**都要能为假也能为真——㊵-748 之前真面是假（表头四列、多数五列），
  ㊵-751 并形之后真面是真；两种都由同一枚布尔如实出版；
* 行内含额外竖线（把一列切成两列）要能被抓出来，且**行首竖线不足**的行不能冒充合法行；
* 形状红面默认开着：坏形状不传任何旗标就必须 rc=1，而 `--no-enforce-shape` 必须 rc=0 **且照样出读数**；
* 零行必须响亮失败——"普查仪报了 0 行"与"债表已规范化"是两件事。
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
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
    """DEBT-G119 的红支：以前这台仪器第 108 行恒 `return 0`，红不出来。

    ㊵-751 之后形状那条是**默认红**（债表已并成一型）；要只出版分布得显式传 `--no-enforce-shape`。
    两支都跑：默认必须红、关掉必须绿——反过来写就是钉住一个不存在的旧形状。
    """

    module = _load()
    broken = UNIFORM_TABLE.replace("| DEBT-G2 | e | f | g | h |\n", "| DEBT-G2 | e | f | g |\n")
    table = _write(tmp_path, "broken.md", broken)
    report = tmp_path / "broken.json"
    assert module.main(["--table", str(table), "--out-report", str(report)]) == 1
    payload = json.loads(report.read_text(encoding="utf-8"))
    #: 红了也照样出版读数——不许"红就不留证据"。
    assert payload["verdicts"]["all_rows_are_the_expected_shape"] is False
    assert payload["rows_not_expected_cells"] == 1
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
                "--no-enforce-shape",
            ]
        )
        == 0
    )
    quiet = json.loads(report2.read_text(encoding="utf-8"))
    #: 关掉红面不等于关掉读数：分布那条照样如实出版。
    assert quiet["rows_not_expected_cells"] == 1
    assert quiet["verdicts"]["all_rows_are_the_expected_shape"] is False


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


def test_real_register_is_one_shape_and_greens_under_default_enforcement(tmp_path: Path) -> None:
    """七裁⑶ 的执行结果由这条守：债表只有一型，且默认红面下必须为绿。

    ㊵-748 时这条测的是反面（`rows_not_expected_cells > 0`、形状判定为假），因为当时 122 行里
    还有 41 行旧四列形与 3 行需人工归列（DEBT-G123）。2026-10-12 ㊵-751 并形落地＋三行手工归列之后
    真面变成 123 行全五列，断言按新状态**重推**而不是在旧值上打补丁。
    本条走 `main()`（不是只调 `measure()`），所以 CI 的 pytest 腿就是这道门的执行点：
    谁往格子里写裸竖线、或把表头改回四列，这条就红。
    """

    module = _load()
    table = PROJECT_ROOT / "plans/active/roadmap/05_TECH_DEBT_REGISTER.md"
    report = module.measure(table)
    assert report["rows"] >= 100
    assert report["distinct_ids"] == report["rows"], report["id_duplicates"]
    #: 一型的定义就是这一行：转义敏感列数只有 5 这一个键，其计数等于行数。
    assert report["cells_per_row_escape_aware"] == {"5": report["rows"]}
    assert report["header_declared_cells"] == 5
    assert report["rows_not_expected_cells"] == 0
    assert all(value is True for value in report["verdicts"].values()), report["verdicts"]
    #: 并形器把每一行都收成了闭合行；"收尾竖线漏写"这一族从此不许再出现。
    assert report["rows_not_closed"] == 0
    assert report["rows_closed_with_pipe"] == report["rows"]

    out = tmp_path / "register.json"
    assert module.main(["--out-report", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["verdicts"] == report["verdicts"]


def test_the_help_screen_survives_a_gbk_console() -> None:
    """DEBT-G81 修法①：`description=__doc__` 会把带 ㊵（U+32B5）的正文送上用法屏，GBK 下抛 `UnicodeEncodeError`。

    2026-10-12 实测：加守卫之前本件的 `--help` 在这台机器上 rc=1（崩溃掩盖了设计值），加守卫之后 rc=0。
    必须起**真进程**——capsys 恒绿看不见这一支（同一课已落在 `test_a30_probe_help_runs.py`）。
    """

    env = dict(os.environ, PYTHONIOENCODING="gbk")
    env.pop("PYTHONUTF8", None)
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        cwd=str(PROJECT_ROOT),
    )
    assert proc.returncode == 0, proc.stderr[-400:]
    #: DEBT-G85 那一族的坏形状：rc=0 而 stdout 是 None，会被读成"帮助里没有这条旗标"。
    assert proc.stdout is not None
    assert "usage" in proc.stdout.lower()
    assert "--no-enforce-shape" in proc.stdout
