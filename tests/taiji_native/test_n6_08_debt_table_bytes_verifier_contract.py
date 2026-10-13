"""`verify_taiji_debt_table_bytes.py` 的契约册（DEBT-G125 修法①的钉子）。

这台仪器存在的理由就是它要能红：㊵-747 那次 `DEBT-G22` 被吃掉一整格（919 字符变 578），
**重排器自己的断言当时是错的却全绿**，抓到我的是另一份独立复核。所以本册的主角不是"绿的那支"，
而是四支必须为假的：删正文、尾巴代填、改了非 DEBT 行、行数变了。

一支专门钉"不许整体反转义"：HEAD 里作者本来就写着的转义竖线，如果被复核器摊平后去比旧行，
就会把"作者原有转义"误报成"仪器改了字节"——这正是 ㊵-751 第一次复核在 `DEBT-G22` 上误报的形状。
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "training" / "verify_taiji_debt_table_bytes.py"

HEADER = "| 编号 | 缺陷或事项 | 影响 | 修法或处置约束 | 状态 |\n|---|---|---|---|---|\n"
ROW_FIVE = "| DEBT-G1 | 五列行 | 低 | 处置 | 状态 |\n"
ROW_FOUR = "| DEBT-G2 | 四列行 | 低 | 处置 |\n"
STATUS_MARKER = "状态未登记（原文按四列登记，㊵-748 统一列形时留空不代填）"


def _load():
    spec = importlib.util.spec_from_file_location("n6_08_bytes_verifier", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _pair(tmp_path: Path, old: str, new: str) -> tuple[Path, Path]:
    old_file = tmp_path / "old.md"
    new_file = tmp_path / "new.md"
    old_file.write_text(old, encoding="utf-8", newline="\n")
    new_file.write_text(new, encoding="utf-8", newline="\n")
    return old_file, new_file


def test_identical_faces_are_green_with_every_verdict_true(tmp_path: Path) -> None:
    module = _load()
    old_file, new_file = _pair(tmp_path, HEADER + ROW_FIVE, HEADER + ROW_FIVE)
    report = module.compare(
        old_file.read_text(encoding="utf-8"), new_file.read_text(encoding="utf-8"), table="t.md"
    )
    assert report["rows"] == 1
    assert report["rows_unchanged"] == 1
    assert all(value is True for value in report["verdicts"].values()), report["verdicts"]


def test_byte_insertions_and_registered_marker_tails_pass(tmp_path: Path) -> None:
    """真正的"合法改写"长这样：旧字节全在，尾巴只多一条已登记的留空标记。"""

    module = _load()
    old = HEADER + ROW_FOUR
    new = HEADER + ROW_FOUR.rstrip("\n") + " | " + STATUS_MARKER + " |\n"
    old_file, new_file = _pair(tmp_path, old, new)
    report = module.compare(
        old_file.read_text(encoding="utf-8"),
        new_file.read_text(encoding="utf-8"),
        table="t.md",
        strict=True,
    )
    assert report["profile"] == "strict_rewrite"
    assert report["verdicts"]["all_row_bytes_preserved_in_order"] is True
    assert report["verdicts"]["tails_are_only_registered_markers"] is True
    assert report["rows_rewritten"] == 1
    assert report["rewrite_kinds"]["appended_marker"] == 1


def test_deleting_body_bytes_must_go_red(tmp_path: Path) -> None:
    """㊵-747 的形状：一整格被吃掉。删掉正文若干个字符，字节判定必须为假、rc 必须非零。"""

    module = _load()
    old = HEADER + "| DEBT-G1 | 这一格里有很长的正文绝对不能丢 | 低 | 处置 | 状态 |\n"
    new = HEADER + "| DEBT-G1 | 这一格里有正文绝对不能丢 | 低 | 处置 | 状态 |\n"
    old_file, new_file = _pair(tmp_path, old, new)
    report = module.compare(
        old_file.read_text(encoding="utf-8"), new_file.read_text(encoding="utf-8"), table="t.md"
    )
    assert report["verdicts"]["all_row_bytes_preserved_in_order"] is False
    assert report["problems"]
    assert (
        module.main(["--old-file", str(old_file), "--new", str(new_file), "--table", "t.md"]) == 1
    )


def test_a_fabricated_tail_must_go_red(tmp_path: Path) -> None:
    """尾巴只许是留空标记；替任何人"补一句结论"都必须当场为假。"""

    module = _load()
    old = HEADER + ROW_FOUR
    new = HEADER + ROW_FOUR.rstrip("\n") + " | 我自己补的一句结论 |\n"
    old_file, new_file = _pair(tmp_path, old, new)
    report = module.compare(
        old_file.read_text(encoding="utf-8"),
        new_file.read_text(encoding="utf-8"),
        table="t.md",
        strict=True,
    )
    assert report["verdicts"]["all_row_bytes_preserved_in_order"] is True
    assert report["verdicts"]["tails_are_only_registered_markers"] is False
    assert (
        module.main(
            [
                "--old-file",
                str(old_file),
                "--new",
                str(new_file),
                "--table",
                "t.md",
                "--strict-rewrite",
            ]
        )
        == 1
    )
    #: 同一份输入在**缺档**下必须为绿：往状态格尾巴里就地追加一条带日期的注是本仓的常规动作，
    #: 禁它的只能是"验证某一次重写"的严档。两档各守各的话，别把严档规则当每批守门。
    assert (
        module.main(["--old-file", str(old_file), "--new", str(new_file), "--table", "t.md"]) == 0
    )


def test_touching_a_non_row_line_must_go_red(tmp_path: Path) -> None:
    """表头与分隔行可以整体替换，正文段落里别人的一个字都不许动。"""

    module = _load()
    old = "# 债表\n\n解释段落，作者写的句子。\n\n" + HEADER + ROW_FIVE
    new = "# 债表\n\n解释段落，作者写的句子（被我改过了）。\n\n" + HEADER + ROW_FIVE
    old_file, new_file = _pair(tmp_path, old, new)
    report = module.compare(
        old_file.read_text(encoding="utf-8"), new_file.read_text(encoding="utf-8"), table="t.md"
    )
    assert report["verdicts"]["non_row_lines_untouched"] is False


def test_adding_a_row_is_reported_but_a_lost_row_goes_red(tmp_path: Path) -> None:
    """对齐按**债号**而不是按行号：正常加一行不许把其后每行都判成篡改。

    这台仪器首跑就撞过这件事——它比的是"HEAD 与工作树"，而那一轮之间正常新增了两行债，
    按行号 zip 于是生出 195 条假红，等于把每一次合法加行都堵死（与 `--enforce-shape`
    当初不能提前打开是同一族理由）。反过来，把已存在的一行整行删掉**必须**为红：那才是真丢数据。
    """

    module = _load()
    added = HEADER + ROW_FIVE + ROW_FOUR
    baseline = HEADER + ROW_FIVE

    old_file, new_file = _pair(tmp_path, baseline, added)
    report = module.compare(
        old_file.read_text(encoding="utf-8"),
        new_file.read_text(encoding="utf-8"),
        table="t.md",
    )
    assert report["rows_added"] == 1
    assert report["rows_lost"] == 0
    assert all(value is True for value in report["verdicts"].values())
    assert (
        module.main(["--old-file", str(old_file), "--new", str(new_file), "--table", "t.md"]) == 0
    )

    old_file2, new_file2 = _pair(tmp_path, added, baseline)
    report2 = module.compare(
        old_file2.read_text(encoding="utf-8"),
        new_file2.read_text(encoding="utf-8"),
        table="t.md",
    )
    assert report2["rows_lost"] == 1
    assert report2["lost_debt_ids"] == ["2"]
    assert report2["verdicts"]["no_rows_lost"] is False
    assert (
        module.main(["--old-file", str(old_file2), "--new", str(new_file2), "--table", "t.md"]) == 1
    )


def test_author_existing_escapes_are_not_reported_as_tampering(tmp_path: Path) -> None:
    """不许整体反转义：旧行里作者本来就写着的 `\\|` 必须照样逐字节对得上。

    第一版复核器做了整行反转义，把作者原有转义一起摊平，于是在 `DEBT-G22` 上把
    "没改动"误报成"改了字节"——那是一次假红，比漏报更难被发现，因为它看起来像抓到了什么。
    """

    module = _load()
    backslash = chr(92)
    row = "| DEBT-G1 | 日志原文 Tests 1 failed " + backslash + "| 487 passed | 低 | 处置 | 状态 |\n"
    old_file, new_file = _pair(tmp_path, HEADER + row, HEADER + row)
    report = module.compare(
        old_file.read_text(encoding="utf-8"), new_file.read_text(encoding="utf-8"), table="t.md"
    )
    assert report["verdicts"]["all_row_bytes_preserved_in_order"] is True
    assert report["problems"] == []


def test_zero_rows_fail_loudly_rather_than_reading_green(tmp_path: Path) -> None:
    module = _load()
    old_file, new_file = _pair(tmp_path, "# 只有标题\n\n没有债行。\n", "# 只有标题\n\n没有债行。\n")
    assert (
        module.main(["--old-file", str(old_file), "--new", str(new_file), "--table", "t.md"]) == 2
    )


def test_the_report_lands_only_where_it_is_told(tmp_path: Path) -> None:
    """默认落点必须是"没有默认落点"：不给 --out-report 就只印 stdout，不覆写在库件。"""

    module = _load()
    old = HEADER + ROW_FIVE
    old_file, new_file = _pair(tmp_path, old, old)
    out = tmp_path / "nested" / "report.json"
    rc = module.main(
        [
            "--old-file",
            str(old_file),
            "--new",
            str(new_file),
            "--table",
            "t.md",
            "--out-report",
            str(out),
        ]
    )
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["format"] == "taiji-debt-table-bytes-verification-v2"
    assert b"\r" not in out.read_bytes()
    assert (
        module.main(
            [
                "--old-file",
                str(old_file),
                "--new",
                str(new_file),
                "--table",
                "t.md",
                "--out-report",
                str(out),
            ]
        )
        == 2
    )


def test_it_judges_the_real_register_against_head_without_a_gate_running() -> None:
    """现行面：HEAD 与工作树此刻应当逐行相同（本批已提交），所以必须为绿且零行改写。"""

    module = _load()
    report = module.compare(
        module.load_old(table=module.DEFAULT_TABLE, old_ref="HEAD", old_file=None),
        (PROJECT_ROOT / module.DEFAULT_TABLE).read_text(encoding="utf-8"),
        table=module.DEFAULT_TABLE,
    )
    assert report["rows"] >= 100
    assert report["profile"] == "no_deletion"
    #: 缺档不禁插入也不禁加行：HEAD 与工作树之间通常就隔着"就地追加注"与新登记行，
    #: 所以这里守的是"没有一行被丢掉、没有一行的旧字节被替换掉"。
    assert report["rows_lost"] == 0, report["lost_debt_ids"]
    assert all(value is True for value in report["verdicts"].values()), report["problems"]


def test_the_help_screen_survives_a_gbk_console() -> None:
    """DEBT-G81 修法①的同一课：拒绝消息与用法屏走 stdout，未守卫则 GBK 把 rc 掩盖成 traceback。"""

    env = dict(os_environ_for_gbk())
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
    assert proc.stdout is not None
    assert "usage" in proc.stdout.lower()


def os_environ_for_gbk() -> dict:
    import os

    values = dict(os.environ, PYTHONIOENCODING="gbk")
    values.pop("PYTHONUTF8", None)
    return values


def test_a_cleaned_file_tail_is_not_mistaken_for_deleted_prose() -> None:
    """DEBT-G135 修法②：尾随空行不是作者的字，但**清理与累积要分方向看**。

    v1 把末尾空行当 prose 行比较，于是"把写盘工具累积的空行清掉"这种纯尾形改动会被判成
    "吞了别人的字"（㊵-776 实测 rc=1、problem＝prose line missing in the new face）。
    v2 先按同量剥掉尾随空行再比 prose，同时把剥掉的枚数出版，并加一条只在**变多**方向为假的判据：
    清理＝另一笔提交，不算违规；累积＝幂等缺失的指纹，必须红。
    """

    verify = _load()
    #: 不用反斜杠转义——外层传输会把换行落成真字符，把字面量劈成语法错（本仓记过的坑，这次又踩）。
    nl = chr(10)
    base = HEADER + ROW_FIVE + "一句作者写的说明。" + nl
    payload = verify.compare(base + nl + nl, base, table="t.md")
    assert payload["verdicts"]["non_row_lines_untouched"] is True
    assert payload["verdicts"]["file_tail_not_grown"] is True
    assert payload["trailing_blank_lines"]["old"] > payload["trailing_blank_lines"]["new"]
    grown = verify.compare(base, base + nl + nl + nl + nl, table="t.md")
    assert grown["verdicts"]["file_tail_not_grown"] is False
    #: 反面对照（这一支最重要）：**真删**一条作者写的说明行时 prose 判据必须仍然为假——
    #: 少了它，"剥尾随空行"就变成一条让删正文也能蒙过去的宽松化。
    deleted = verify.compare(base, HEADER + ROW_FIVE, table="t.md")
    assert deleted["verdicts"]["non_row_lines_untouched"] is False
