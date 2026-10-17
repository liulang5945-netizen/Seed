"""债表"有效状态"普查仪的合同测：三档分类各自要能为假，且词表外的情形必须显式落 unknown。

仪器存在的理由是我自己把已结清的 DEBT-G125 当成"仅剩的未修格"引用过（08 ㊵-1041），
所以这里钉死四件事：①头部"还欠"＋行尾戳"已做完"必须被抓；②结清之后又追加**与状态无关**的注
不得把结论冲掉（我第一版就是按字面最后一枚戳，漏了 G125）；③表外措辞必须进 `unclassified_stamp`
而不是被猜成某个状态；④严行与追加行的计数差必须点名到编号。
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.training import census_taiji_debt_row_effective_status as census  # noqa: E402

HEAD_OPEN = "**只登记，未实现**；相关：DEBT-G101"
STAMP_CLOSED = "〔2026-10-13 ㊵-755 · 修法①②③全部做完，本行结清〕①入库仪器已建"
NEUTRAL_STAMP = "〔2026-10-14 ㊵-800 · 同日就地再修：这台仪器首跑撞出自己的适用面缺陷〕"


def row(identifier: str, status_cell: str) -> str:
    return f"| {identifier} | 缺陷与复现 | 影响 | 修法与验收 | {status_cell} |"


def analyse_cell(status_cell: str, identifier: str = "DEBT-G900") -> dict:
    return census.analyse(row(identifier, status_cell))


def test_stale_head_is_caught_and_a_clean_row_is_not() -> None:
    stale = analyse_cell(HEAD_OPEN + "；" + STAMP_CLOSED)
    assert stale["stale_head"] is True, stale
    assert stale["head_state"] == "open" and stale["stamp_state"] == "closed", stale

    honest = analyse_cell("**只登记，未修**；由 ㊵-1 量出。〔2026-10-17 ㊵-2 登记〕")
    assert honest["stale_head"] is False, honest
    assert honest["reopened_by_stamp"] is False, honest


def test_a_state_free_later_note_does_not_erase_the_closure() -> None:
    """我第一版按"字面最后一枚戳"分类，于是 G125 那种"结清后又追加自述注"被读成 unknown。"""
    reading = analyse_cell(HEAD_OPEN + "；" + STAMP_CLOSED + "；" + NEUTRAL_STAMP)
    assert reading["stamp_state"] == "closed", reading
    assert reading["stale_head"] is True, reading
    assert reading["deciding_stamp_text"].startswith("2026-10-13"), reading
    assert reading["last_stamp_text"].startswith("2026-10-14"), "字面最后一枚仍要原样出版"


def test_vocabulary_outside_the_table_lands_in_unclassified_not_a_guess() -> None:
    reading = analyse_cell(
        "**待 owner 定形状**〔2026-10-13 ㊵-776 · 十裁⑶ 批准实施受体通道重映射〕"
    )
    assert reading["stamp_state"] == "unknown", reading
    assert reading["unclassified_stamp"] is True, reading
    assert reading["stale_head"] is False, reading
    assert reading["reopened_by_stamp"] is False, reading


def test_a_closed_head_reopened_by_a_later_stamp_is_flagged_the_other_way() -> None:
    reading = analyse_cell("**已修（㊵-9）**〔2026-10-15 ㊵-10 · 上一句作废，仍待 owner 裁定〕")
    assert reading["head_state"] == "closed" and reading["stamp_state"] == "open", reading
    assert reading["reopened_by_stamp"] is True and reading["stale_head"] is False, reading


def test_a_row_without_any_bracket_is_neither_caught_nor_invented() -> None:
    reading = analyse_cell("**只登记，未实现**；无注。")
    assert reading["stamp_state"] == "none" and reading["stamp_count"] == 0, reading
    assert reading["unclassified_stamp"] is False and reading["stale_head"] is False, reading


def test_negative_prefix_rows_are_not_read_as_status_words() -> None:
    """`未结清` 里含着 `结清`：不加否定前缀守卫就会把"还欠着"读成"已做完"。"""
    state, token = census.classify("**本行未结清**")
    assert state == "open", (state, token)
    state, token = census.classify("**本行结清**")
    assert state == "closed", (state, token)


def test_the_real_register_parses_and_the_two_counts_reconcile() -> None:
    text = census.TABLE.read_text(encoding="utf-8")
    result = census.measure(text)
    counts = result["counts"]
    assert counts["strict_rows"] == counts["distinct_ids"], counts
    assert counts["prefix_lines"] > counts["strict_rows"], counts
    assert counts["extra_lines_with_a_repeated_id"] == ["DEBT-G48", "DEBT-G49", "DEBT-G50"], counts
    #: 棘轮（㊵-1104 改写）：债表自 08 ㊵-1097 起 stale_head 为空，今后再出现
    #: "状态列比本行行尾戳旧"的行，这条测必须红；判别力本身改由下面的合成夹具钉住。
    assert result["stale_head"] == [], result["stale_head"][:5]
    assert counts["rows_with_stamps"] > 0 and counts["max_stamps_in_one_row"] >= 5, counts
    flagged = (
        set(result["stale_head"])
        | set(result["reopened_by_stamp"])
        | set(result["unclassified_stamp"])
    )
    assert len(flagged) > 0, "no row needed a human look: the classifier is probably blind"


def test_cli_writes_once_and_has_no_default_repo_target(tmp_path: Path) -> None:
    out = tmp_path / "census.json"
    assert census.main(["--out-report", str(out)]) == 0
    first = out.read_bytes()
    assert census.main(["--out-report", str(out)]) == 2
    assert out.read_bytes() == first, "refusal must not rewrite the sealed report"
    assert census.main([]) == 0, "无旗标只印 stdout，不得有默认写靶"


def test_a_stale_head_against_its_own_row_end_stamp_is_detected_on_a_synthetic_row() -> None:
    """判别力改由合成人行钉：DEBT-G125 那一形（头部说还欠、行尾戳说已结清）必须在夹具上被抓到。

    〔㊵-1104〕本测的存在理由原先写在"真表里 G125 必须出现在 stale_head"上；08 ㊵-1097 把
    那 27 行清完之后，真表读数变成空 ⇒ 拿现场读数当判别力证据是错的形状（DEBT-G148 同族：
    盘面状态不许当判据）。判据不变，取证方式换成夹具。
    """
    text = (
        "| 编号 | 缺陷 | 影响 | 修法 | 状态 |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| DEBT-G900 | 示例缺陷 | 示例影响 | 示例修法 | 待 owner 裁定（㊵-1） "
        "〔㊵-2 裁定落地：本行结清〕 |\n"
        "| DEBT-G901 | 示例缺陷二 | 示例影响 | 示例修法 | 只登记，未修（㊵-3） |\n"
    )
    result = census.measure(text)
    assert result["stale_head"] == ["DEBT-G900"], result["stale_head"]
    assert result["reopened_by_stamp"] == [], result["reopened_by_stamp"]
    assert "DEBT-G901" not in result["stale_head"]
