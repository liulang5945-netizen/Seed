"""`PLAN-N5-16` 的掩码契约册（J-N5o-1／2／3＋守卫）：空白来源行不许拿概率质量，也不许被假误差拉走。

本册落盘时**产品默认枚数仍是 4**——掩码与加宽是两次单变量改动（㊵-715 的教训：一次动两个变量的读数无法归因）。
`J-N5o-4`（旧件端到端中性）在本册里**先用打补丁的词表**测出来（架构 5 枚、产品默认仍 4 枚），
这样"掩码是否真的换回逐位不变"在动默认位之前就有凭据；`J-N5o-5`（`generated` 可用）仍留给枚数真的变 5 那一格。

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
#: 加宽一档的词表（`generated` 必须加在末尾，否则旧坐标整体移位）。
KINDS5 = ("experienced", "imagined", "replayed", "external", "generated")


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


def test_a_written_source_immediately_enters_the_denominator() -> None:
    """`J-N5o-5`：`generated` 被写一次之后必须拿到正质量——否则"中性"就是把它做成了不可用。

    起点不是全新模型（那档全部行空白，走的是回退，1/5 人人有份），而是"四个老来源都写过、
    第五个还没写过"这一档——正是旧件迁移进来时的形状。
    """

    field = _field()
    width = len(field.PROVENANCE_KINDS)
    generated = field.PROVENANCE_KINDS.index("generated")
    readout = field.provenance_readout
    for row in range(width):
        if row == generated:
            continue
        error = torch.zeros(width, device=field.device)
        error[row] = 1.0
        readout.local_update(
            error,
            torch.ones(readout.in_features, device=field.device),
            learning_rate=0.05,
            weight_decay=0.0,
        )
    zeros = torch.zeros(width, device=field.device)
    before = field._normalized_provenance(zeros)
    assert float(before[generated]) == 0.0
    assert float(before.sum()) == pytest.approx(1.0)

    write_generated = torch.zeros(width, device=field.device)
    write_generated[generated] = 1.0
    readout.local_update(
        write_generated,
        torch.ones(readout.in_features, device=field.device),
        learning_rate=0.05,
        weight_decay=0.0,
    )
    after = field._normalized_provenance(zeros)
    assert float(after[generated]) > 0.0, "写过一次还不进分母＝这条来源被做成了不可用"
    assert float(after.sum()) == pytest.approx(1.0)
    assert float(after[generated]) == pytest.approx(1.0 / width)


def test_the_blank_predicate_exists_only_once() -> None:
    """G-N5o-1：判定"什么算空白行"的逻辑只许有一处，三处消费点必须都走辅助函数。"""

    tree = ast.parse(MEMORY_SOURCE)
    predicate_sites = [
        node for node in ast.walk(tree) if isinstance(node, ast.Attribute) and node.attr == "amax"
    ]
    assert len(predicate_sites) == 1, predicate_sites
    assert MEMORY_SOURCE.count(BLANK_PREDICATE) == 1
    assert MEMORY_SOURCE.count(MASK_CALL) == 3, "读侧、无读先验、写路径三处必须各自调用"


def test_a_real_four_kind_artifact_reads_back_bitwise_identically_under_five_kinds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`J-N5o-4`＝㊵-720 那条否证的正脸：同一枚**真实**旧 payload，两档架构观测量逐位相同。

    架构宽度靠打补丁的词表得到（产品默认仍是 4 枚），所以这条不是在验证我自己造的宽档，
    而是在验证载入路径 `migrate_presynaptic_payload`／`adopt_output_rows_from_payload` 加掩码之后
    到底换回了什么。夹具用 `to_payload()` 的真实产物，不是手搓形状。
    """

    import taiji.config as config_module

    # 旧件＝比当前默认少一枚来源的词表（不写死 4，而是"少最后一枚"，默认以后再变它也成立）。
    monkeypatch.setattr(config_module, "EPISODIC_PROVENANCE_KINDS", KINDS5[:-1])
    monkeypatch.setattr(EpisodicField, "PROVENANCE_KINDS", KINDS5[:-1])
    narrow = _field()
    # 用真实写路径把每一行点亮（而不是 normal_：那会绕过 `_bound_rows` 的行长上限，
    # 让"复制过去再夹一次"与原件不等——第一次跑就是这么误报的）。
    rows = narrow.provenance_readout.out_features
    for row in range(rows):
        error = torch.zeros(rows, device=narrow.device)
        error[row] = 1.0
        trace = torch.ones(narrow.provenance_readout.in_features, device=narrow.device)
        narrow.provenance_readout.local_update(
            error,
            trace,
            learning_rate=0.05,
            weight_decay=0.0,
        )
    payload = narrow.to_payload()
    width = len(narrow.PROVENANCE_KINDS)

    monkeypatch.setattr(config_module, "EPISODIC_PROVENANCE_KINDS", KINDS5)
    monkeypatch.setattr(EpisodicField, "PROVENANCE_KINDS", KINDS5)
    widened = _field()
    widened.load_payload(payload)

    assert widened.provenance_readout.out_features == width + 1
    assert widened.provenance_encoder.in_features == width + 1
    # 补进去的那一行必须真的空白（这是掩码生效的前提，也是"没被用过"的定义）。
    assert float(widened.provenance_readout.edge_weight[width].abs().max()) == 0.0

    context = torch.randn(
        widened.provenance_readout.in_features, generator=torch.Generator().manual_seed(9)
    )
    before = narrow._normalized_provenance(narrow.provenance_readout.forward(context))
    after = widened._normalized_provenance(widened.provenance_readout.forward(context))
    assert torch.equal(before, after[:width]), (before.tolist(), after.tolist())
    assert float(after[width]) == 0.0
    assert float(after.sum()) == pytest.approx(1.0)
    # 无读那一支也要逐位：它以前是 1/4，加宽后必须是 1/4＋一个 0.0 位，而不是 1/5。
    prior_before = narrow._normalized_provenance(torch.zeros(width, device=narrow.device))
    prior_after = widened._normalized_provenance(torch.zeros(width + 1, device=widened.device))
    assert torch.equal(prior_before, prior_after[:width]), (prior_before, prior_after)
    assert float(prior_after[width]) == 0.0


def test_the_migration_path_still_validates_the_receptor_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """迁移不是放行：改走 `transplant_payload` 之后，留空一个通道的受体图仍必须被响亮拒绝。

    这一支就是 `J-N5n-4` 的反支——那句 `receptor payload leaves a context channel unconnected`
    得真的能被触发，否则"复用现成迁移入口"就等于把校验换成了信任。
    """

    import taiji.config as config_module

    # 旧件＝比当前默认少一枚来源的词表（不写死 4，而是"少最后一枚"，默认以后再变它也成立）。
    monkeypatch.setattr(config_module, "EPISODIC_PROVENANCE_KINDS", KINDS5[:-1])
    monkeypatch.setattr(EpisodicField, "PROVENANCE_KINDS", KINDS5[:-1])
    narrow = _field()
    payload = narrow.to_payload()
    hole = dict(payload["readout_receptors"])
    hole["channel"] = torch.zeros_like(hole["channel"])

    monkeypatch.setattr(config_module, "EPISODIC_PROVENANCE_KINDS", KINDS5)
    monkeypatch.setattr(EpisodicField, "PROVENANCE_KINDS", KINDS5)
    widened = _field()
    with pytest.raises(ValueError, match="unconnected"):
        widened.load_payload({**payload, "readout_receptors": hole})


def test_a_migrated_checkpoint_can_be_resumed_by_a_second_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """迁移过的件必须能被第二个进程再载入——㊵-725 整族抓到的那条红就是这个形状。

    第一手迁移把**旧**受体图继承进来并存了盘；第二手再按"必须等于本机所抽"去比就会拒。
    受体图是状态不是架构，所以这里断言：第二次载入不抛，且 `receptors_inherited` 把这件事披露出来。
    """

    import taiji.config as config_module

    monkeypatch.setattr(config_module, "EPISODIC_PROVENANCE_KINDS", KINDS5[:-1])
    monkeypatch.setattr(EpisodicField, "PROVENANCE_KINDS", KINDS5[:-1])
    narrow = _field()
    payload = narrow.to_payload()

    monkeypatch.setattr(config_module, "EPISODIC_PROVENANCE_KINDS", KINDS5)
    monkeypatch.setattr(EpisodicField, "PROVENANCE_KINDS", KINDS5)
    first = _field()
    first.load_payload(payload)
    resumed_payload = first.to_payload()

    second = _field()
    second.load_payload(resumed_payload)
    assert bool(second.receptors_inherited)
    assert torch.equal(second.readout_receptors.channel, first.readout_receptors.channel)
    assert second.provenance_encoder.in_features == len(KINDS5)
