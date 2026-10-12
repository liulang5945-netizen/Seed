"""DEBT-G113 修法① 的契约册：参数面规划公式里的情节场来源宽度必须**跟着枚数走**，且现值不许动。

为什么这册值得存在（㊵-715 实测）：`taiji/config.py` 里那句 `readout_outputs = … + 4` 是枚数的**副本**，
而枚数住在 `EpisodicField.PROVENANCE_KINDS`。批次 B 预检把枚数从 4 加到 5 时，实参涨了 32 而规划数纹丝不动
⇒ `test_architecture_contract.py` 两支红。那两支持红了是**好事**（参数面变大被响亮报出），但真正的债是
`PLAN-N5-14` 的 `J-N5m-1` 只检查 encoder／readout 的入参，看不见这枚公式——不变量会假绿。

四支各自的用处：
①钉"改派生之后 4 枚档的数值逐位不变"（＝这针是重排不是涨价）；
②钉"枚数＋1 ⇒ 规划数跟着涨，涨幅恰＝一枚读出行的宽度"，并要规划数与实参**再次相等**；
③钉"这一格里不许出现字面量 4 或 5"；
④是③的负对照——同一把尺必须能在**旧形状**上数出那枚 4，否则③是恒真式（[[guard-must-be-able-to-fail]]）。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import taiji.config as config_module
from taiji import Taiji, TaijiConfig
from taiji.memory import EpisodicField

CONFIG_SOURCE = Path(config_module.__file__).read_text(encoding="utf-8")
PLAN_FUNCTION = "planned_active_parameter_count"
#: ㊵-716 立这条钉子时枚数仍是 4，规划数与实参都＝99122；㊵-725 把来源词表加到 5 枚之后
#: 按同一把尺现取＝**99154**，差值 32 恰＝一枚读出行的宽度（㊵-715 实测过的同一个恒等式）。
#: 这一格是"随枚举重钉"（`G-N5n-2`），不是放宽成区间——旧值留着当出处。
PLAN_AT_FIVE_KINDS = 99154
FOUR_KIND_PLAN = 99122

BASE_KWARGS = {
    "region_sizes": (64, 48),
    "synapse_fan_in": 16,
    "motor_fan_in": 48,
    "memory_units": 128,
    "memory_fan_in": 32,
    "memory_meta_dim": 32,
    "memory_readout_fan_in": 32,
    "memory_iterations": 3,
    "seed": 23,
    "identity_organ_enabled": False,
}
KINDS4 = ("experienced", "imagined", "replayed", "external")
KINDS5 = KINDS4 + ("generated",)


def _literal_width_hits(source: str) -> list[int]:
    """在规划公式这一格里找整型字面量 4／5——按 AST 取，不按字符串撞（注释与文档不算数）。"""

    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == PLAN_FUNCTION:
            return sorted(
                {
                    constant.value
                    for constant in ast.walk(node)
                    if isinstance(constant, ast.Constant)
                    and isinstance(constant.value, int)
                    and constant.value in (4, 5)
                }
            )
    raise AssertionError(f"找不到 {PLAN_FUNCTION} 的定义，扫描面不能静默为空")


def test_plan_value_follows_the_vocabulary_by_exactly_one_readout_row() -> None:
    config = TaijiConfig(**BASE_KWARGS)
    assert config.planned_active_parameter_count == PLAN_AT_FIVE_KINDS
    assert Taiji(config).parameter_count() == PLAN_AT_FIVE_KINDS
    # 两档历史读数之间的差必须能被"一枚读出行的宽度"解释——这是恒等式，不是新的魔法数。
    readout_width = min(config.memory_readout_fan_in, config.memory_meta_dim)
    assert readout_width == PLAN_AT_FIVE_KINDS - FOUR_KIND_PLAN, (
        PLAN_AT_FIVE_KINDS,
        FOUR_KIND_PLAN,
        readout_width,
    )


def test_plan_follows_the_vocabulary_and_still_equals_the_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """再加一枚（合成词表）⇒ 规划数涨幅恰＝一枚读出行宽度，且规划数与实参仍相等。

    这一支刻意**不写死枚数**：它取当前默认词表再加一个探针用的假来源，
    所以无论词表以后是 5 枚还是 6 枚，它测的都是"公式跟着枚数走"这件事本身。
    """

    config = TaijiConfig(**BASE_KWARGS)
    readout_width = min(config.memory_readout_fan_in, config.memory_meta_dim)
    before = config.planned_active_parameter_count
    wider = (*config_module.EPISODIC_PROVENANCE_KINDS, "probe_one_more_source")

    monkeypatch.setattr(config_module, "EPISODIC_PROVENANCE_KINDS", wider)
    monkeypatch.setattr(EpisodicField, "PROVENANCE_KINDS", wider)

    widened_config = TaijiConfig(**BASE_KWARGS)
    assert widened_config.planned_active_parameter_count - before == readout_width, (
        widened_config.planned_active_parameter_count,
        before,
        readout_width,
    )
    assert Taiji(widened_config).parameter_count() == widened_config.planned_active_parameter_count


def test_the_plan_formula_holds_no_literal_width() -> None:
    assert _literal_width_hits(CONFIG_SOURCE) == []


def test_the_finder_can_still_see_the_old_shape() -> None:
    """③的负对照：旧形状必须被数出来，否则"没有字面量"这句谁都验不了。"""

    legacy = f"def {PLAN_FUNCTION}(self):\n    return 2 * self.alphabet_size + 4\n"
    assert _literal_width_hits(legacy) == [4]
    with pytest.raises(AssertionError, match="扫描面不能静默为空"):
        _literal_width_hits("def unrelated(self):\n    return 4\n")
