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

#: 加宽前／加宽后的来源词表；`generated` 必须加在**末尾**，否则旧坐标会整体移位。
KINDS4 = ("experienced", "imagined", "replayed", "external")
KINDS5 = KINDS4 + ("generated",)


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


def _field(kinds: tuple[str, ...], monkeypatch: pytest.MonkeyPatch):
    import taiji.config as config_module
    from taiji.memory import EpisodicField

    monkeypatch.setattr(config_module, "EPISODIC_PROVENANCE_KINDS", kinds)
    monkeypatch.setattr(EpisodicField, "PROVENANCE_KINDS", kinds)
    from taiji import Taiji, TaijiConfig

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


def test_narrower_provenance_payload_is_adopted_with_silent_new_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`J-N5n-2` 的端到端接线：旧 payload 进新架构 ⇒ 旧边逐位保留、新列权重为零。"""

    field4 = _field(KINDS4, monkeypatch)
    context = torch.randn(
        field4.provenance_readout.in_features, generator=torch.Generator().manual_seed(3)
    )
    baseline = field4.provenance_readout.forward(context)
    payload = {
        "provenance_encoder": field4.provenance_encoder.to_payload(),
        "provenance_readout": field4.provenance_readout.to_payload(),
    }

    field5 = _field(KINDS5, monkeypatch)
    field5.load_payload({**field5.to_payload(), **payload})

    encoder = field5.provenance_encoder
    assert encoder.in_features == 5 and encoder.row_fan_in == 5
    assert torch.equal(
        encoder.edge_weight[:, 4], torch.zeros(encoder.out_features, dtype=torch.float32)
    )
    assert torch.equal(
        encoder.pre_index[:, 4], torch.full((encoder.out_features,), 4, dtype=torch.int32)
    )
    assert torch.equal(
        field5.provenance_readout.edge_weight[4], torch.zeros(4, dtype=torch.float32)
    )
    assert torch.equal(field5.provenance_readout.forward(context)[:4], baseline)


def test_adopt_refuses_a_row_that_was_already_written(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """负对照：多出来的行不空白就必须拒绝——否则"采纳旧件"会覆盖已学权重。"""

    field4 = _field(KINDS4, monkeypatch)
    field5 = _field(KINDS5, monkeypatch)
    field5.provenance_readout.edge_weight[4] = 0.5
    with pytest.raises(ValueError, match="extra output rows to be blank"):
        field5.provenance_readout.adopt_output_rows_from_payload(
            field4.provenance_readout.to_payload()
        )


def test_adopt_refuses_a_payload_that_is_not_a_shorter_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """等宽与更宽都不许走迁移路——那条路只在枚数真的变大时存在。"""

    field4 = _field(KINDS4, monkeypatch)
    same = _field(KINDS4, monkeypatch)
    with pytest.raises(ValueError, match="shorter prefix"):
        same.provenance_readout.adopt_output_rows_from_payload(
            field4.provenance_readout.to_payload()
        )
    wider = _field(KINDS5, monkeypatch)
    with pytest.raises(ValueError, match="shorter prefix"):
        field4.provenance_readout.adopt_output_rows_from_payload(
            wider.provenance_readout.to_payload()
        )


def test_migrated_readout_can_still_learn_the_new_source_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`J-N5n-3` 的"活槽位"那一支：空白行经过一次真实三因子写就能变非零。

    这是**原语面**的证据（直接调 `local_update`）；端到端写链上"generated 的表征真能长出来"
    要等带回答级来源记录的那一档，本格不替它作证。
    """

    field4 = _field(KINDS4, monkeypatch)
    field5 = _field(KINDS5, monkeypatch)
    field5.load_payload(
        {
            **field5.to_payload(),
            "provenance_encoder": field4.provenance_encoder.to_payload(),
            "provenance_readout": field4.provenance_readout.to_payload(),
        }
    )
    readout = field5.provenance_readout
    error = torch.zeros(readout.out_features, dtype=torch.float32)
    error[4] = 1.0
    trace = torch.ones(readout.in_features, dtype=torch.float32)

    readout.local_update(error, trace, learning_rate=0.5, weight_decay=0.0)

    assert float(readout.edge_weight[4].abs().max()) > 0.0
    assert float(readout.edge_weight[:4].abs().max()) == 0.0
