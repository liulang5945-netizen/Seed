"""R2 复制回路合同（A2.1+A2.2）——立项 §4 判据①「结构存在性」的可提交形态。

钉死三件事：

1. **位级惰性**：挂载（gate 零初始化、store 空）后 predictive 输出与未挂载**逐位相同**；
2. **oracle 寻址驱动发射**：写入告知字节后，以诊断覆写（one-hot 寻址 + gate 幅度）验证
   「内容 → F1 字节发射」的接线真实存在——这正是 A0 判 (c) 有罪所缺的那条通路；
3. **持久化**：circuit 随 checkpoint 往返；旧 checkpoint（无键）直载 ⇒ 未挂载。
"""

from __future__ import annotations

import pytest
import torch

from taiji import CopyCircuit, Taiji, TaijiConfig

TELL = "我叫阿岩。".encode()
ASK = "我的名字是什么？".encode()
# 阿 = E9 98 BF；TELL 中首字节位置 6（我E6 88 91｜叫E5 8F AB｜阿…）。
A_FIRST_BYTE = 0xE9
A_FIRST_POSITION = 6


def _model() -> Taiji:
    return Taiji(
        TaijiConfig(
            region_sizes=(64, 48),
            synapse_fan_in=16,
            motor_fan_in=48,
            seed=11,
        )
    )


def _feed_predictive(model: Taiji, symbols: bytes) -> torch.Tensor:
    model.reset_dynamics(episode_id="copy-circuit-contract")
    step = model.observe(model.config.boundary_symbol, learn=False, readout="predictive")
    last = step
    for symbol in symbols:
        last = model.observe(int(symbol), learn=False, readout="predictive")
    return last.probabilities.detach().cpu().clone()


def test_mount_is_bitwise_inert() -> None:
    plain = _model()
    mounted = _model()
    mounted.mount_copy_circuit()
    assert isinstance(mounted.copy_circuit, CopyCircuit)
    plain_probs = _feed_predictive(plain, ASK)
    mounted_probs = _feed_predictive(mounted, ASK)
    assert torch.equal(plain_probs, mounted_probs)


def test_zero_init_gate_makes_evidence_exactly_zero() -> None:
    model = _model()
    model.mount_copy_circuit()
    circuit = model.copy_circuit
    assert circuit is not None
    circuit.store.record(TELL, torch.ones(model.config.cortical_context_dim))
    cue = torch.zeros(model.config.cortical_context_dim)
    cue[3] = 1.0
    f1_context = torch.zeros(model.config.motor_context_dim)
    f1_context[5] = 0.5
    evidence = circuit.evidence(cue=cue, f1_context=f1_context)
    assert torch.equal(evidence, torch.zeros(model.config.alphabet_size))


def test_oracle_address_drives_emission() -> None:
    model = _model()
    model.mount_copy_circuit()
    # 写入段：让皮质态先经过告知内容，再走 A2.1 语言写入门。
    _feed_predictive(model, TELL)
    event_id = model.record_told_content(TELL)
    assert event_id == 0
    circuit = model.copy_circuit
    assert circuit is not None
    weights = torch.zeros(len(TELL))
    weights[A_FIRST_POSITION] = 1.0
    circuit.address_override = weights
    circuit.gate_override = 20.0
    probabilities = _feed_predictive(model, ASK)
    assert int(probabilities.argmax()) == A_FIRST_BYTE


def test_store_capacity_and_cosine_match() -> None:
    model = _model()
    model.mount_copy_circuit(max_events=2)
    circuit = model.copy_circuit
    assert circuit is not None
    dim = model.config.cortical_context_dim
    first = torch.zeros(dim)
    first[0] = 1.0
    second = torch.zeros(dim)
    second[1] = 1.0
    third = torch.zeros(dim)
    third[2] = 1.0
    circuit.store.record(b"aa", first)
    circuit.store.record(b"bb", second)
    assert circuit.store.count == 2
    circuit.store.record(b"cc", third)
    assert circuit.store.count == 2  # FIFO 淘汰最早一条
    matched = circuit.store.best_match(second)
    assert matched is not None and matched.content == b"bb"
    evicted = circuit.store.best_match(first)
    assert evicted is not None and evicted.content != b"aa"


def test_successor_bonus_shifts_copy_mass_to_next_byte() -> None:
    """rev3：「前一字节＝刚发出的字节」的行加 induce 偏置 ⇒ 复制质量集中到后继位置。"""
    model = _model()
    model.mount_copy_circuit()
    circuit = model.copy_circuit
    assert circuit is not None
    cue = torch.zeros(model.config.cortical_context_dim)
    cue[3] = 1.0
    circuit.store.record(TELL, cue.clone())
    f1 = torch.zeros(model.config.motor_context_dim)
    f1[5] = 0.5
    before = circuit.addressing(cue=cue, f1_context=f1)["copy_distribution"]
    circuit.parameters()["copy_induce_bias"].data.fill_(8.0)
    after = circuit.addressing(cue=cue, f1_context=f1, prev_byte=TELL[0])["copy_distribution"]
    # TELL[0]＝0xE6（「我」首字节）⇒ 后继位置 1（TELL[1]＝0x88）吸走质量
    assert float(after[TELL[1]]) > 0.5
    assert float(after[TELL[1]]) > float(before[TELL[1]])


def test_checkpoint_roundtrip_and_legacy_absence() -> None:
    model = _model()
    model.mount_copy_circuit()
    circuit = model.copy_circuit
    assert circuit is not None
    dim = model.config.cortical_context_dim
    cue = torch.zeros(dim)
    cue[7] = 1.0
    circuit.store.record(TELL, cue)
    circuit.parameters()["gate_bias"].data.fill_(0.25)
    payload = model.checkpoint()
    assert payload["copy_circuit"]["format"] == CopyCircuit.PAYLOAD_FORMAT

    restored = Taiji.from_checkpoint(payload)
    restored_circuit = restored.copy_circuit
    assert restored_circuit is not None
    assert torch.equal(
        restored_circuit.parameters()["gate_bias"],
        circuit.parameters()["gate_bias"],
    )
    assert restored_circuit.store.count == 1
    event = restored_circuit.store.best_match(cue)
    assert event is not None and event.content == TELL

    # 防篡改（fail-closed）：从带 circuit 写入的 payload 里摘掉 circuit 键，
    # identity lineage 校验必须拒绝——lineage digest 覆盖 circuit，键集不可漂移。
    tampered = model.checkpoint()
    tampered.pop("copy_circuit")
    with pytest.raises(ValueError, match="lineage"):
        Taiji.from_checkpoint(tampered)

    # 旧 checkpoint（从未挂载 ⇒ 天然无键）直载 ⇒ 未挂载，行为零变化。
    plain_payload = _model().checkpoint()
    assert "copy_circuit" not in plain_payload
    plain = Taiji.from_checkpoint(plain_payload)
    assert plain.copy_circuit is None
