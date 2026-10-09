"""PLAN-N5-05 §3 G-N5e-1：把「位置输入×发育 F1 通路」那道拒绝的**调用点与守卫引用**钉成测。

为什么先钉再改：甲路线要动的是 `taiji/model.py:1123` 那道响亮拒绝。本仓的老规矩是
"改动前先把影响面变成会红的断言"——否则改完只剩两个风险：
① 有人新增了第三处调用而没人知道；② 钉着"该抛"的既有测被顺手删掉，于是"没抛"变成默认正确。

本册只钉**清单本身**（数量与在场性），不预判改动结果：接通之后 §J-N5e-2 的正向证据
（`position_path_delta > 0 ∧ n_changed_units ≥ 1`）在下一格补进本册，届时这里的"该抛"支会重钉成新形状。
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODEL = REPO / "taiji" / "model.py"
HELPER = "_reject_position_input_without_learning_path"

#: ㊵-634 现读：两处调用（:1284 migrate、:3821 恢复发育态）。数量是钉值，新增第三处必红。
EXPECTED_CALL_SITES = 2
#: 现读：行为被这两册钉着（test_n3_04 断"该抛"、test_readout_utf8_position 也调用一次）。
GUARD_FILES = (
    "tests/taiji_native/test_n3_04_developmental_flags_contract.py",
    "tests/taiji_native/test_readout_utf8_position.py",
)
#: migrate 的既有调用方（改动后都要跑；少一个就说明清单过期）。
MIGRATE_CALLERS = (
    "taiji/language_alignment.py",
    "scripts/training/check_taiji_m4v2_checkpoint_preflight.py",
    "scripts/training/eval_taiji_m4v2_r2_canary.py",
    "tests/taiji_native/test_developmental_synapse.py",
)


def _model_text() -> str:
    return MODEL.read_text(encoding="utf-8")


def _call_sites(text: str) -> list[int]:
    return [i for i, line in enumerate(text.split(chr(10)), start=1) if f"self.{HELPER}()" in line]


def test_the_rejection_is_called_from_exactly_the_inventoried_sites() -> None:
    sites = _call_sites(_model_text())
    assert len(sites) == EXPECTED_CALL_SITES, sites
    #: 钉"是哪两处"而不只是"几个"：位置漂移说明有人把它挪进了别的分支。
    assert sites == [1284, 3821], sites


def test_the_helper_is_still_defined_once_and_still_reads_the_config_flag() -> None:
    text = _model_text()
    tree = ast.parse(text, filename=str(MODEL))
    defs = [
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == HELPER
    ]
    assert len(defs) == 1, [d.lineno for d in defs]
    body = ast.dump(defs[0])
    assert "readout_utf8_position_input" in body, "拒绝器不再看配置位＝它已经不是那道闸"
    assert "ValueError" in body, "拒绝器不再抛＝清单过期，须重钉而不是让本测静默放行"


def test_the_pinned_guards_still_reference_the_rejection() -> None:
    """钉着"该抛"的测不许被悄悄删——删了就等于让"没抛"成为默认正确。"""

    for rel in GUARD_FILES:
        path = REPO / rel
        assert path.is_file(), rel
        text = path.read_text(encoding="utf-8")
        assert HELPER in text or "not wired to the developmental F1" in text, rel
        assert re.search(r"pytest\.raises|raises\(", text), rel


def test_every_inventoried_migrate_caller_still_exists() -> None:
    """migrate 的调用方清单：任何一处改名/删除都要先更新本件，而不是让甲的实施跳过它。"""

    for rel in MIGRATE_CALLERS:
        path = REPO / rel
        assert path.is_file(), rel
        assert "migrate_f1_to_developmental_synapses" in path.read_text(encoding="utf-8"), rel
