"""`unify_taiji_debt_table_columns.py` 的契约册（DEBT-G125 修法②的钉子）。

这台仪器**已经造成过一次真实数据丢失**：第一版把整行按解析结果重铺，`DEBT-G22` 从 919 字符变成
578，「已修与实测」那一整格没了，而它自己的断言当时说"可写"。所以本册不是围着统计转，
而是逐条钉住那次的形状：

* 旧字节必须在场——给四列行追加留空标记时，前面四格一个字符都不许动；
* 省略了收尾竖线的行不能被切成少一格（那正是 `[1:-1]` 那次丢格的成因）；
* 数不动的行必须**拒绝处理**，而且拒绝的时候**什么都不写**（宁可整件不动，也不猜列边界）；
* 干跑与写盘是两件事：不给 `--apply` 不许碰文件。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "training" / "unify_taiji_debt_table_columns.py"

HEADER = "| 编号 | 缺陷或事项 | 影响 | 修法或处置约束 |\n|---|---|---|---|\n"
ROW_FOUR = "| DEBT-G2 | 四列行 | 低 | 处置 |\n"
ROW_FIVE = "| DEBT-G1 | 五列行 | 低 | 处置 | 状态 |\n"
ROW_UNCLOSED = "| DEBT-G3 | 收尾竖线漏了的一行 | 低 | 处置里最后一格绝对不能丢"


def _load():
    spec = importlib.util.spec_from_file_location("n5_32_unifier", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _transform(module, text: str):
    lines = text.split("\n")
    out, stats, problems = module.transform(lines)
    return "\n".join(out), stats, problems


def test_appending_the_status_marker_preserves_every_original_byte(tmp_path: Path) -> None:
    """丢格那次的直接反面：追加只发生在行尾，前面四格的字节必须逐字符还在。"""

    module = _load()
    new, stats, problems = _transform(module, HEADER + ROW_FOUR)
    assert problems == []
    assert stats["appended_status"] == 1
    assert new.split("\n")[2].startswith(ROW_FOUR.rstrip("\n")[:40])
    assert "处置里" not in new  # 这张表里没有这串字，出现即说明拼接走偏
    assert ROW_FOUR.rstrip("\n").split("|")[2].strip() in new
    assert new.split("\n")[2].endswith(module.STATUS_MARKER + " |")


def test_an_unclosed_row_is_closed_before_the_cells_are_counted(tmp_path: Path) -> None:
    """省略收尾竖线的行以前被 `[1:-1]` 切掉最后一格；现在先补竖线，最后一格必须原样在场。"""

    module = _load()
    text = HEADER + ROW_UNCLOSED
    new, stats, problems = _transform(module, text)
    assert problems == []
    assert stats["closed_missing_tail"] >= 1
    line = new.split("\n")[2]
    assert "处置里最后一格绝对不能丢" in line
    assert len(module.cells_of(line)) >= 4


def test_a_five_cell_row_is_left_byte_identical(tmp_path: Path) -> None:
    module = _load()
    new, stats, problems = _transform(module, HEADER + ROW_FIVE)
    assert problems == []
    assert stats["kept_five"] == 1
    assert new.split("\n")[2] == ROW_FIVE.rstrip("\n")


def test_pipes_inside_code_spans_are_escaped_not_treated_as_columns() -> None:
    module = _load()
    backslash = chr(92)
    row = "| DEBT-G4 | 日志 `Tests 1 failed " + backslash + "| 487 passed` | 低 | 处置 | 状态 |"
    # 码段内的裸竖线（未转义）会被切成第六格；重排器应转义它而不是搬动文字。
    raw_row = row.replace(backslash + "|", "|")
    new, stats, problems = _transform(module, HEADER + raw_row + "\n")
    assert problems == []
    assert stats["escaped"] == 1
    assert "`Tests 1 failed " in new and "487 passed`" in new
    assert len(module.cells_of(new.split("\n")[2])) == 5


def test_an_unresolvable_row_is_refused_and_nothing_is_written(tmp_path: Path) -> None:
    """行中间的裸竖线不在码段里 ⇒ 工具不解释、不吸收，`problems` 非空且 `--apply` 必须拒写。"""

    module = _load()
    row = "| DEBT-G5 | 叙述里有一枚裸竖线 左|右 需要人工归列 | 低 | 处置 | 状态 |\n"
    table = tmp_path / "register.md"
    table.write_text(HEADER + row, encoding="utf-8", newline="\n")
    before = table.read_bytes()
    rc = module.main(["--table", str(table)])
    assert rc == 2, "数不动的行必须响亮拒绝，而不是挑一个解释吞下去"
    assert table.read_bytes() == before


def test_dry_run_never_touches_the_file(tmp_path: Path) -> None:
    module = _load()
    table = tmp_path / "register.md"
    table.write_text(HEADER + ROW_FOUR, encoding="utf-8", newline="\n")
    before = table.read_bytes()
    rc = module.main(["--table", str(table)])
    assert rc == 0
    assert table.read_bytes() == before, "不给 --apply 就写盘＝把干跑当验收"


def test_apply_writes_and_normalises_the_header(tmp_path: Path) -> None:
    module = _load()
    table = tmp_path / "register.md"
    table.write_text(HEADER + ROW_FOUR + ROW_FIVE, encoding="utf-8", newline="\n")
    rc = module.main(["--table", str(table), "--apply"])
    assert rc == 0
    lines = table.read_bytes().decode("utf-8").split("\n")
    assert lines[0] == module.NEW_HEADER
    assert lines[1] == module.NEW_SEPARATOR
    assert all(len(module.cells_of(line)) == 5 for line in lines[2:] if module.ROW_RE.match(line))
    assert table.read_bytes().count(b"\r") == 0, "写盘必须纯 LF，否则整档行尾翻牌"


def test_the_transform_is_idempotent_on_an_already_normalised_table() -> None:
    """再干跑一次必须零改动——这是「批量重排已收敛」的第四条验收。"""

    module = _load()
    once, _, _ = _transform(module, HEADER + ROW_FOUR + ROW_UNCLOSED + "\n")
    again, stats, problems = _transform(module, once)
    assert problems == []
    assert again == once
    assert stats["appended_status"] == 0 and stats["closed_missing_tail"] == 0
