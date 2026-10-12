"""`PLAN-N5-16` 的掩码契约册（J-N5o-1／2／3＋守卫）：空白来源行不许拿概率质量，也不许被假误差拉走。

本册落盘时**枚数仍是 4**——掩码与加宽是两次单变量改动（㊵-715 的教训：一次动两个变量的读数无法归因）。
`J-N5o-4`（旧件端到端中性）与 `J-N5o-5`（`generated` 可用）要等枚数真的变 5，落在那一格的提交里。

四支各自的用处：
①`J-N5o-1` 空白位必须恰为 0.0、已写位之和为 1.0；
②`J-N5o-2` 位置不许消失——掩码是"给 0"，不是"删掉一位"（删掉会让按 `PROVENANCE_KINDS` 下标取来源的消费者错位）；
③`J-N5o-3` 全空白必须回退成今天的 `1/k` 且不出 NaN——这一支是**回归锚**：它证明"未训模型的观测量没被我改动"；
④`G-N5o-1` 按 AST 数：空白谓词在 `taiji/memory.py` 里只能出现 **1 次**，而三处消费点各自必须**调用**辅助函数
  （改前两处 softmax＋一处先验＝3 处调用）；这一支防的是"第二份空白判定"，它比"softmax 处数"更对，
  因为把三处收敛进一个辅助函数**必然**让 softmax 变少——少不等于对。
"""

from __future__ import annotations

import ast
import inspect
import math
from pathlib import Path

import pytest
import torch

from taiji import Taiji, TaijiConfig
from taiji.memory import EpisodicField

MEMORY_SOURCE = Path(inspect.getsourcefile(EpisodicField) or "").read_text(encoding="utf-8")
BLANK_PREDICATE = "edge_weight.abs().amax(dim=1)"
MASK_CALL = "self._normalized_provenance("


def _field() -> EpisodicField:
    config = TaijiConfig(
        region_sizes=(16, 12),
        memory_units=24,
        memory_fan_in=4,
        memory_meta_dim=8,
        memory_readout_fan_in=4,
        seed=7,
        identity_organ_enabled=False,
    )
    return Taiji(config).memory


def _make_row_live(field: EpisodicField, row: int) -> None:
    field.provenance_readout.edge_weight[row] = 0.25 * (row + 1)


def test_blank_source_row_takes_no_probability_mass() -> None:
    """J-N5o-1：只有被写过的行进分母，空白行的位必须恰为 0.0。"""

    field = _field()
    _make_row_live(field, 2)
    probabilities = field._normalized_provenance(
        torch.zeros(len(field.PROVENANCE_KINDS), device=field.device)
    )
    assert float(probabilities[2]) == pytest.approx(1.0)
    assert [float(probabilities[i]) for i in (0, 1, 3)] == [0.0, 0.0, 0.0]
    assert float(probabilities.sum()) == pytest.approx(1.0)


def test_mask_zeroes_the_row_but_keeps_its_position() -> None:
    """J-N5o-2：掩码给 0 而不是删掉一位——长度必须仍等于枚数。"""

    field = _field()
    width = len(field.PROVENANCE_KINDS)
    _make_row_live(field, 0)
    _make_row_live(field, 3)
    probabilities = field._normalized_provenance(torch.zeros(width, device=field.device))
    assert probabilities.shape == (width,)
    assert float(probabilities[1]) == 0.0 and float(probabilities[2]) == 0.0
    assert float(probabilities[0]) == pytest.approx(0.5)
    # 再点亮一行：原来那两行的质量必须按比例缩小，而不是维持原值（那说明分母没跟着变）。
    _make_row_live(field, 1)
    after = field._normalized_provenance(torch.zeros(width, device=field.device))
    assert float(after[0]) == pytest.approx(1.0 / 3.0)
    assert float(after.sum()) == pytest.approx(1.0)


def test_all_blank_falls_back_to_today_behaviour_without_nan() -> None:
    """J-N5o-3（回退支＝回归锚）：全空白时逐位等于今天——`1/k` 均匀且没有 NaN。"""

    field = _field()
    width = len(field.PROVENANCE_KINDS)
    probabilities = field._normalized_provenance(torch.zeros(width, device=field.device))
    assert not bool(torch.isnan(probabilities).any())
    assert torch.equal(probabilities, torch.full((width,), 1.0 / width, device=field.device))


def test_blank_rows_are_not_pulled_by_a_fake_error() -> None:
    """写路径的形状（`PLAN-N5-16` §2 第三处消费点）：写一个**已存在**的来源时，空白位的误差必须恰为 0.0。

    改前它是 `0 - 1/k`＝一个把还没用过的来源往负方向拉的**假误差**；这条测同时把改前的值算出来对照，
    免得"掩码生效"这句话只由我自己说了算。
    """

    field = _field()
    width = len(field.PROVENANCE_KINDS)
    blank_row = width - 1
    for row in range(blank_row):
        _make_row_live(field, row)
    live_rows = blank_row
    zeros = torch.zeros(width, device=field.device)
    code = torch.zeros(width, device=field.device)
    code[0] = 1.0  # 写的是第 0 个来源，它已经存在

    masked_error = code - field._normalized_provenance(zeros)
    unmasked_error = code - torch.softmax(zeros, dim=0)

    assert float(masked_error[blank_row]) == pytest.approx(0.0)
    assert float(unmasked_error[blank_row]) == pytest.approx(-1.0 / width)
    assert float(masked_error[0]) == pytest.approx(1.0 - 1.0 / live_rows)
    assert all(
        float(masked_error[row]) == pytest.approx(-1.0 / live_rows) for row in range(1, live_rows)
    )
    assert math.isclose(float(masked_error.sum()), 0.0, abs_tol=1e-6)


def test_the_blank_predicate_exists_only_once() -> None:
    """G-N5o-1：判定"什么算空白行"的逻辑只许有一处，三处消费点必须都走辅助函数。"""

    tree = ast.parse(MEMORY_SOURCE)
    predicate_sites = [
        node for node in ast.walk(tree) if isinstance(node, ast.Attribute) and node.attr == "amax"
    ]
    assert len(predicate_sites) == 1, predicate_sites
    assert MEMORY_SOURCE.count(BLANK_PREDICATE) == 1
    assert MEMORY_SOURCE.count(MASK_CALL) == 3, "读侧、无读先验、写路径三处必须各自调用"
