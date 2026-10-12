"""PLAN-N5-15 §1 的迁移原语契约册（B′ 第一步）：`SparseSynapses.expand_presynaptic_slots`。

这个原语不是新发明的：它原本是 `taiji/identity_organ.py` 私有的 `_expand_slot_synapses`，
情节场加宽要用同一件事（把 presynaptic 宽度加长、**不重写任何已学边**、新列权重为零），
所以把它提到 `taiji/sparse.py` 由两边共用（`G-N5n-1`＝不许第二份生成链）。

六支各自能为假：
①旧边逐位保留；②**新列置零 ⇒ `forward` 逐位不变**——这一支直接量 `PLAN-N5-15` §7 点名的
  首查失败面（`(edge_weight * presynaptic[pre_index]).sum(dim=1)` 在 4 列与 5 列上的归约顺序），
  所以它不是"设计上应该相等"，是**测出来相等**；
③`fan_in != in_features`（行没覆盖全部坐标）必须拒绝——否则"新列是哪些坐标"无从得知；
④宽度不增必须拒绝；
⑤按 AST 验 identity 那一侧真的只剩一次转发（重抄就红）；
⑥签名里没有 `generator` 参数 ⇒ 取数流独立性是**结构**保证，不靠调用方小心。
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
import torch

from taiji import identity_organ as identity_module
from taiji.sparse import SparseSynapses

IDENTITY_SOURCE = Path(identity_module.__file__).read_text(encoding="utf-8")


def _full_slot_synapses(out_features: int = 6, width: int = 4, seed: int = 11) -> SparseSynapses:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return SparseSynapses(
        out_features,
        width,
        width,
        generator=generator,
        init_scale=0.3,
        max_weight_norm=2.0,
        device="cpu",
    )


def _partial_fan_in_synapses() -> SparseSynapses:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(5)
    return SparseSynapses(
        6,
        8,
        3,
        generator=generator,
        init_scale=0.3,
        max_weight_norm=2.0,
        device="cpu",
    )


def test_expansion_keeps_every_existing_edge_bitwise() -> None:
    source = _full_slot_synapses()
    expanded = source.expand_presynaptic_slots(5)

    assert expanded.in_features == 5 and expanded.row_fan_in == 5
    assert torch.equal(expanded.pre_index[:, :4], source.pre_index)
    assert torch.equal(expanded.edge_weight[:, :4], source.edge_weight)
    # 追加那一列的坐标是"旧群体没有的那些"，从件自己的宽度读出，不写常数。
    assert torch.equal(
        expanded.pre_index[:, 4:],
        torch.full((6, 1), 4, dtype=torch.int32),
    )
    assert float(expanded.edge_weight[:, 4:].abs().max()) == 0.0


def test_widened_forward_is_bitwise_identical_when_new_column_is_zero() -> None:
    """`J-N5n-2` 的最小面：新列权重为零 ⇒ 同一条前向逐位不变，哪怕新坐标本身不为零。"""

    source = _full_slot_synapses()
    expanded = source.expand_presynaptic_slots(5)
    old_input = torch.tensor([0.25, -1.5, 3.0, 0.125], dtype=torch.float32)
    new_input = torch.cat([old_input, torch.tensor([7.3], dtype=torch.float32)])

    before = source.forward(old_input)
    after = expanded.forward(new_input)

    assert torch.equal(before, after), (before.tolist(), after.tolist())


def test_expansion_refuses_rows_that_do_not_cover_the_population() -> None:
    partial = _partial_fan_in_synapses()
    with pytest.raises(ValueError, match="full slot fan-in"):
        partial.expand_presynaptic_slots(9)


def test_expansion_refuses_a_non_increasing_width() -> None:
    source = _full_slot_synapses()
    with pytest.raises(ValueError, match="must increase width"):
        source.expand_presynaptic_slots(4)
    with pytest.raises(ValueError, match="must increase width"):
        source.expand_presynaptic_slots(2)


def test_identity_organ_no_longer_carries_a_second_copy() -> None:
    """`G-N5n-1`：共用一份实现。委托函数体里**不许再有赋值**——那意味着第二份补列逻辑。"""

    for node in ast.walk(ast.parse(IDENTITY_SOURCE)):
        if isinstance(node, ast.FunctionDef) and node.name == "_expand_slot_synapses":
            assignments = [q for q in ast.walk(node) if isinstance(q, (ast.Assign, ast.AugAssign))]
            calls = [q for q in ast.walk(node) if isinstance(q, ast.Call)]
            assert not assignments, ast.dump(node)
            assert len(calls) == 1, calls
            return
    raise AssertionError("identity_organ 里找不到 _expand_slot_synapses，委托关系已断")


def test_expansion_is_independent_of_the_callers_random_stream() -> None:
    """结构性保证：原语不接受 `generator`，所以加宽不会推后调用方的取数位置。"""

    parameters = list(inspect.signature(SparseSynapses.expand_presynaptic_slots).parameters)
    assert parameters == ["self", "new_in_features"], parameters
