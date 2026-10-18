"""DEBT-G174 的合同测：`--emit-per-item` 是 opt-in，默认出版形状必须与今天逐字节相同。

这台复述仪在 `:227` 一直把算好的逐条列剥掉，所以保持侧那三条跌（`replay_strict_hits:1.0/2.0`、
`replay_well_formed:2.0`）点不到题。改它的两条硬要求都能为假：

* 不给旗标 ⇒ `arms` 的字节与旧的字典推导**逐字相同**（否则本轮之前封存的两张件就复现不了）；
* 给旗标 ⇒ 逐条列出门，且这张表必须**恰好铺满**文本表（`assert_rows_tile` 两支：铺满通过／错位响亮拒）。

第三组钉的是"别把推导改回内联"：件面必须调用 `published_arms(...)`，源码里出现旧的内联剥列写法就红。
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "measure_taiji_a30_repetition_penalty.py"


def _module():
    spec = importlib.util.spec_from_file_location("a30_replay_reader_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _arm() -> dict:
    return {
        "repetition_penalty": 0.0,
        "items": 2,
        "texts": 4,
        "strict_hits": 1,
        "per_item": [
            {
                "id": "V001",
                "hit": True,
                "cyclic": False,
                "expected_contains": ["42"],
                "text_range": [0, 2],
                "texts": 2,
                "formed_texts": 2,
                "cyclic_tail_texts": 0,
            },
            {
                "id": "V002",
                "hit": False,
                "cyclic": True,
                "expected_contains": ["17"],
                "text_range": [2, 4],
                "texts": 2,
                "formed_texts": 1,
                "cyclic_tail_texts": 1,
            },
        ],
    }


def test_the_default_publication_is_byte_identical_to_the_old_expression() -> None:
    module = _module()
    arms = [_arm(), _arm()]
    legacy = [{k: v for k, v in arm.items() if k != "per_item"} for arm in arms]
    published = module.published_arms(arms, False)
    assert json.dumps(published, sort_keys=True) == json.dumps(legacy, sort_keys=True)
    assert all("per_item" not in arm for arm in published)


def test_the_flag_publishes_the_rows_without_mutating_the_source_arms() -> None:
    module = _module()
    arms = [_arm()]
    published = module.published_arms(arms, True)
    assert published[0]["per_item"][0]["id"] == "V001"
    published[0]["injected"] = True
    assert "injected" not in arms[0], "出版用的顶层字典不许与被测量对象共享"
    assert "per_item" in arms[0] and len(arms[0]["per_item"]) == 2


def test_rows_that_tile_the_text_list_pass_and_misaligned_rows_are_refused() -> None:
    module = _module()
    module.assert_rows_tile(_arm()["per_item"], items=2, texts=4)
    gap = [
        {"id": "V001", "text_range": [0, 1]},
        {"id": "V002", "text_range": [2, 4]},
    ]
    try:
        module.assert_rows_tile(gap, items=2, texts=4)
    except RuntimeError as exc:
        assert "do not tile" in str(exc)
    else:
        raise AssertionError("漏掉一条文本的逐条列必须被拒（分子分母不同集合）")
    duplicate = [
        {"id": "V001", "text_range": [0, 2]},
        {"id": "V002", "text_range": [1, 4]},
    ]
    try:
        module.assert_rows_tile(duplicate, items=2, texts=4)
    except RuntimeError:
        pass
    else:
        raise AssertionError("同一条文本被两个 item 认领必须被拒")


def test_a_row_count_that_does_not_match_the_item_count_is_refused() -> None:
    module = _module()
    rows = [{"id": "V001", "text_range": [0, 4]}]
    try:
        module.assert_rows_tile(rows, items=2, texts=4)
    except RuntimeError as exc:
        assert "rows=1 items=2" in str(exc)
    else:
        raise AssertionError("行数与题数不符必须被拒")


def test_the_report_assembly_calls_the_function_instead_of_re_inlining_the_strip() -> None:
    """源码钉：有人把 `arms` 改回内联剥列，旗标就会静默失效，这条测必须红。"""

    source = SCRIPT.read_text(encoding="utf-8")
    assert '"arms": published_arms(arms, args.emit_per_item)' in source
    assert (
        '"arms": [{k: v for k, v in arm.items() if k != "per_item"} for arm in arms]' not in source
    )
    assert '"--emit-per-item"' in source


def test_the_per_item_rows_carry_the_fields_the_mechanism_question_needs() -> None:
    """点到题 requires 每题的期望子串与成句条数；光有 hit 只答得出"没答出"那一半。"""

    module = _module()
    fields = {
        "id",
        "hit",
        "cyclic",
        "expected_contains",
        "text_range",
        "texts",
        "formed_texts",
        "cyclic_tail_texts",
    }
    row_fields = set(module.published_arms([_arm()], True)[0]["per_item"][0])
    assert fields <= row_fields, f"缺字段: {sorted(fields - row_fields)}"
