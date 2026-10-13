"""`PLAN-N4-08` §4 批次 B 的契约册：VISION §3.3 四类验收的执行体必须每类出一个值，且每支判级式都能为假。

本册不重跑判据本身（判据冻在 `PLAN-N4-08` §3），它钉四件事：
①四类各有且仅有一个 verdict，且没有哪类被写成 `holds` 来冒充通过；
②三条纯判级式（干扰／属性互换／槽满）的**每一支**都被实际调用过——含"恰好等于噪声带"那一支；
③引用式的源码钉在真件上为真、在空件上为假（钉本身能为假）；
④读数件落盘是 LF，且没有 `--overwrite` 时拒绝覆盖已入库件（DEBT-G115 同族）。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INSTRUMENT = PROJECT_ROOT / "scripts" / "training" / "adjudicate_taiji_n4_four_class_acceptance.py"
CLASSES = ("rename", "attribute_swap", "distractor", "slot_full")


def _module():
    spec = importlib.util.spec_from_file_location("n4_33_instrument", INSTRUMENT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def instrument():
    return _module()


@pytest.fixture(scope="module")
def payload(tmp_path_factory) -> dict[str, Any]:
    module = _module()
    out = tmp_path_factory.mktemp("n433") / "four_classes.json"
    assert module.main(["--out-report", str(out)]) == 0
    return json.loads(out.read_text(encoding="utf-8"))


def test_every_class_publishes_exactly_one_verdict(payload: dict[str, Any]) -> None:
    assert sorted(payload["classes"]) == sorted(CLASSES), sorted(payload["classes"])
    assert payload["every_class_has_a_value"] is True
    assert payload["all_pinned"] is True, payload["cited_formula_pins"]
    for name in CLASSES:
        verdict = payload["classes"][name]["verdict"]
        assert isinstance(verdict, str) and verdict, name


def test_the_two_carrierless_classes_are_not_reported_as_passes(payload: dict[str, Any]) -> None:
    """重命名今天仍不可判——本册钉的是「它不许被写成 holds」；属性那一格㊵-757 起换了口径。

    原来这支冻的是 `swap["verdict"] == "no_field_to_swap"`。属性位（PLAN-N4-12）落地后这一格**第一次
    出真值**，"不许是 holds"这句在它身上变成了"要 holds 就必须把两枚探针一起出版"——
    所以改成下面三条各自能红的实测：它必须离开 no_field_to_swap；holds 只能带着
    「换 cue 顶位变」与「不带属性那张面不动」两枚探针出现；缺探针时的 holds 一律算伪绿。
    """

    rename = payload["classes"]["rename"]
    swap = payload["classes"]["attribute_swap"]
    assert rename["verdict"] != "holds"
    assert swap["verdict"] != "no_field_to_swap"
    #: 两枚探针必须**无条件在场**（冻件 §2 要求同时出版）；缺任何一枚就不许出 verdict。
    assert {"cue_probe_moves_top1", "cue_only_face_unchanged_after_swap"} <= set(swap["probe"])
    if swap["verdict"] == "holds":
        assert swap["probe"]["cue_probe_moves_top1"] is True
        assert swap["probe"]["cue_only_face_unchanged_after_swap"] is True
    #: 伪绿的证据链必须自己在件里，不是靠我口头说。
    assert rename["frozen_shape_would_read_green"] is True
    assert rename["paired_queries_after_record_only_rename"] == 0
    assert rename["unpaired_queries_after_record_only_rename"] > 0
    assert rename["cue_carrier_positive_control"] is True


def test_retrieve_face_reads_cue_episode_and_the_binding_key(payload: dict[str, Any]) -> None:
    """㊵-745 换载体后的重钉：retrieve 可以看 memory_id（绑定通道用它过滤），但仍不认对象字段本身。

    原来这支冻的是 `["cue", "episode_id"]`。绑定通道（`PLAN-N4-11` §1）落地后 retrieve 必须引用
    `memory_id` 才能按对象收窄候选集——所以这一支改成三条各自能红的实测，而不是放松成"看几个都行"：
    字段集合恰好是这三枚、提取器仍能认出 cue（它没瞎）、`object_ids` 仍不被引用（打分还是只看相似度）。
    """

    swap = payload["classes"]["attribute_swap"]
    assert swap["fields_referenced_by_retrieve"] == ["cue", "episode_id", "memory_id"], swap
    assert swap["extractor_detects_cue_in_retrieve"] is True
    assert swap["extractor_detects_memory_id_in_write"] is True
    assert swap["object_ids_referenced_by_retrieve"] is False
    assert "cue" in swap["fields_referenced_by_write"]


def test_interference_formula_covers_the_exact_band_branch(instrument) -> None:
    assert instrument._interference_verdict(0.1, 0.2, 0.25) == "holds"
    #: 恰好等于带：冻的原话是"必须大于带才许出结论"，所以这一支是答不了，不是不成立。
    assert instrument._interference_verdict(0.1, 0.1, 0.1) == "ruler_unusable"
    assert instrument._interference_verdict(0.1, 0.1, 0.05) == "ruler_unusable"
    #: 带为零＝这把尺没有动态范围，也不许出结论（哪怕差很大）。
    assert instrument._interference_verdict(0.0, 0.0, 0.9) == "ruler_unusable"


def test_swap_formula_covers_every_branch_and_the_absent_field_one_still_fires(instrument) -> None:
    """㊵-757 换签名的重钉：原判据式只有两支，属性位落地后变成四出口。

    原来这本册冻的是 `_swap_verdict(False) == "no_field_to_swap"` 与 `(True) == "swap_branch_missing"`。
    `swap_branch_missing` 那一支已被真互换判定取代（冻件 PLAN-N4-12 §2 的 J-N4g-3c），全分支表住在
    `test_n4_11::test_swap_verdict_can_fire_every_branch`；这里只保留本册自己该守的那一条——
    **字段不在时仍然必须落在 no_field_to_swap**，不许哪天字段被删掉却出版一个更轻的出口。
    """

    assert instrument._swap_verdict(False, True, True, False) == "no_field_to_swap"
    assert instrument._swap_verdict(True, True, True, False) == "holds"


def test_slot_verdict_can_fire_every_branch(instrument) -> None:
    assert instrument._slot_verdict(0, 0, False, False) == "not_full_no_eviction"
    assert instrument._slot_verdict(12, 12, False, False) == "holds_recorded_eviction"
    assert instrument._slot_verdict(12, 12, True, False) == "violated_returns_empty"
    assert instrument._slot_verdict(12, 3, False, False) == "violated_silent_or_stale"
    assert instrument._slot_verdict(12, 12, False, True) == "violated_silent_or_stale"


def test_slot_lane_ships_a_not_full_control_that_cannot_pass(
    payload: dict[str, Any],
) -> None:
    block = payload["classes"]["slot_full"]
    assert block["verdict"] == "holds_recorded_eviction"
    assert block["control_lane"]["verdict"] == "not_full_no_eviction"
    assert block["control_must_not_report_eviction"] is True
    assert block["lane"]["evicted_count_in_checkpoint"] == block["lane"]["evicted_expected"]


def test_the_cited_formula_pin_can_be_false(instrument) -> None:
    pins = instrument._cited_formula_pins("def judge_metric_lane(): return None\n")
    assert set(pins.values()) == {False}, pins
    real = (PROJECT_ROOT / instrument.SHADOW_GATE).read_text(encoding="utf-8")
    assert set(instrument._cited_formula_pins(real).values()) == {True}


def test_the_instrument_refuses_to_overwrite_an_existing_report(tmp_path) -> None:
    module = _module()
    out = tmp_path / "already.json"
    out.write_text("{}\n", encoding="utf-8", newline="\n")
    assert module.main(["--out-report", str(out)]) == 2
    assert out.read_text(encoding="utf-8") == "{}\n"


def test_the_report_is_written_with_lf_endings(tmp_path, instrument) -> None:
    out = tmp_path / "lf_check.json"
    assert instrument.main(["--out-report", str(out)]) == 0
    raw = out.read_bytes()
    assert raw.count(b"\r") == 0
    assert json.loads(raw.decode("utf-8"))["format"] == instrument.REPORT_FORMAT
