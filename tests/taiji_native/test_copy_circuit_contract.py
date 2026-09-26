"""R2 复制回路合同（A2.1+A2.2）——立项 §4 判据①「结构存在性」的可提交形态。

钉死三件事：

1. **位级惰性**：挂载（gate 零初始化、store 空）后 predictive 输出与未挂载**逐位相同**；
2. **oracle 寻址驱动发射**：写入告知字节后，以诊断覆写（one-hot 寻址 + gate 幅度）验证
   「内容 → F1 字节发射」的接线真实存在——这正是 A0 判 (c) 有罪所缺的那条通路；
3. **持久化**：circuit 随 checkpoint 往返；旧 checkpoint（无键）直载 ⇒ 未挂载。

A2.3b 追加（文件后半段）：格式对齐重训臂的三条守卫——**裸格式臂逐位不变**、
**对齐臂落在产品生成链上**、**寻址抢错事件时一步都不更新**。
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch

from taiji import CopyCircuit, Taiji, TaijiConfig

TELL = "我叫阿岩。".encode()
ASK = "我的名字是什么？".encode()
# 阿 = E9 98 BF；TELL 中首字节位置 6（我E6 88 91｜叫E5 8F AB｜阿…）。
A_FIRST_BYTE = 0xE9
A_FIRST_POSITION = 6

REPO_ROOT = Path(__file__).resolve().parents[2]
for _entry in (REPO_ROOT, REPO_ROOT / "scripts" / "training"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))


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


def test_pool_default_is_bitwise_the_existing_sum() -> None:
    """`pool_override` 不设时，字节聚合必须与直接 `index_add` **逐位相同**。

    这条守卫的意义：器官已经上线过"挂载必须位级不变"的规矩（§4.1），
    加一个诊断分支不能悄悄改变生产路径。
    """
    circuit = CopyCircuit(_model().config, max_events=4)
    codes = torch.tensor([229, 139, 162, 229, 147, 175], dtype=torch.long)
    weights = torch.tensor([0.1, 0.2, 0.3, 0.35, 0.02, 0.03])
    zeros = torch.zeros(int(circuit.config.alphabet_size), dtype=torch.float32)
    got = circuit._pool_into(zeros.clone(), codes, weights)
    want = torch.zeros_like(zeros).index_add(0, codes, weights)
    assert bool(torch.equal(got, want))


def test_pool_max_reports_the_strongest_position_not_the_count() -> None:
    """ "取最大"必须只看最强位置——这正是要检验的那件事（位置数不该变成质量）。"""
    circuit = CopyCircuit(_model().config, max_events=4)
    codes = torch.tensor([229, 139, 162, 229, 147, 175], dtype=torch.long)
    weights = torch.tensor([0.1, 0.2, 0.3, 0.35, 0.02, 0.03])
    zeros = torch.zeros(int(circuit.config.alphabet_size), dtype=torch.float32)
    circuit.pool_override = "max"
    got = circuit._pool_into(zeros.clone(), codes, weights)
    assert float(got[229]) == pytest.approx(0.35), got[229]  # 位置多但都不是最强
    assert float(got[162]) == pytest.approx(0.3)
    summed = torch.zeros_like(zeros).index_add(0, codes, weights)
    assert float(summed[229]) == pytest.approx(0.45)  # 求和那侧才是"数票"


def test_pool_mean_divides_by_position_count_and_keeps_multiplicity() -> None:
    """ "取平均"是对"数被当成质量"的公平检验：重数保留、计数放大去掉。"""
    circuit = CopyCircuit(_model().config, max_events=4)
    codes = torch.tensor([229, 139, 162, 229, 147, 175], dtype=torch.long)
    weights = torch.tensor([0.1, 0.2, 0.3, 0.35, 0.02, 0.03])
    zeros = torch.zeros(int(circuit.config.alphabet_size), dtype=torch.float32)
    circuit.pool_override = "mean"
    got = circuit._pool_into(zeros.clone(), codes, weights)
    assert float(got[229]) == pytest.approx(0.45 / 2.0, abs=1e-6), got[229]
    assert float(got[162]) == pytest.approx(0.3)  # 只出现一次的位置不受影响
    summed = torch.zeros_like(zeros).index_add(0, codes, weights)
    assert float(got[229]) < float(summed[229])  # 计数放大被除掉，但质量次序仍留着


def test_pool_unknown_mode_is_loud() -> None:
    circuit = CopyCircuit(_model().config, max_events=4)
    circuit.pool_override = "median"
    zeros = torch.zeros(int(circuit.config.alphabet_size), dtype=torch.float32)
    with pytest.raises(ValueError):
        circuit._pool_into(zeros, torch.tensor([1, 2]), torch.tensor([0.5, 0.5]))


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


# ---------------------------------------------------------------------------
# A2.3b：格式对齐重训臂的守卫（预注册 SPEC-A-17 §5）
# ---------------------------------------------------------------------------

#: A2.3 冻结臂（本次抽取重构**之前**的 `train_taiji_r2_copy_circuit.py`）在
#: stage=smoke／episodes=5／seed=20260925／lr 默认下的 circuit 参数 sha256，
#: 用 `git show HEAD:` 取出的未改动副本实测钉下（不是推算值）。
BARE_ARM_PINNED_DIGEST = "d55bf2ef5f13f998f220d575d6cbe9352d40c4223f11f6f49f102ae2d2e9f48d"


def _parameters_digest(parameters: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps({k: v.tolist() for k, v in parameters.items()}, sort_keys=True).encode()
    ).hexdigest()


#: 这两份 digest 是在 rev5（A2.5 选择头）**之前**按当时的全部 7 个张量实测的。
#: 加了新参数就不能再拿"整套参数"去比——那会把"新头是零"这个新事实混进"旧张量没动"这个老断言。
#: 所以：按下面这份字面名单逐个取，名单本身被断言钉住（少一个/多一个都红），
#: 新增的两项则单独断言"恰好为零"。
PRE_SELECTOR_PARAMETER_NAMES = (
    "content_embed",
    "query_state",
    "query_content",
    "gate_state",
    "gate_content",
    "gate_bias",
    "copy_induce_bias",
)


def _pre_selector_digest(parameters: dict[str, Any]) -> str:
    selected = {name: parameters[name] for name in PRE_SELECTOR_PARAMETER_NAMES}
    assert set(parameters) - set(selected) == set(
        CopyCircuit.OPTIONAL_ZERO_PARAMETERS
    ), "参数面变了：先重推这两份 digest 的口径，别改名单迁就现状"
    for name in CopyCircuit.OPTIONAL_ZERO_PARAMETERS:
        assert float(parameters[name].abs().sum()) == 0.0, name
    return _parameters_digest(selected)


def _run_trainer(tmp_path: Path, protocol: str, episodes: int) -> dict[str, Any]:
    import train_taiji_r2_copy_circuit as trainer

    argv = [
        "train_taiji_r2_copy_circuit.py",
        "--stage",
        "smoke",
        "--protocol",
        protocol,
        "--episodes",
        str(episodes),
        "--max-minutes",
        "5",
        "--out-dir",
        str(tmp_path / "out" / protocol),
        "--report-dir",
        str(tmp_path / "rep"),
    ]
    with patch.object(sys, "argv", argv):
        assert trainer.main() == 0
    return torch.load(
        tmp_path / "out" / protocol / "smoke" / "circuit-final.pt", weights_only=False
    )


def test_bare_arm_survives_the_extraction_untouched(tmp_path: Path) -> None:
    """抽取学习规则为共用函数**没有**改动 A2.3 冻结臂：同参数同种子 ⇒ 参数逐位相同。

    这条钉子是 A2.3b 全部结论的前提——两臂必须只差"怎么喂"，否则格式对齐的读数差
    可能被"顺手改了规则"解释掉。
    """
    payload = _run_trainer(tmp_path, "bare", 5)
    assert payload["protocol"] == "bare"
    assert payload["prereg"].endswith("M5_R2_A2_3_PREREG_20260925.md")
    assert _pre_selector_digest(payload["copy_circuit"]["parameters"]) == BARE_ARM_PINNED_DIGEST


def test_record_primitive_keeps_told_turns_unwrapped() -> None:
    """入库语义＝产品语义：存**用户轮原文**（不套「问：」壳），cue＝「问：轮」读完的皮质态。"""
    from api.seed_runtime import record_told_history

    model = _model()
    model.mount_copy_circuit(max_events=4)
    circuit = model.copy_circuit
    assert circuit is not None
    history = [("我叫阿蒙。", "上一轮答复"), ("水沸点是多少？", "另一轮答复")]
    record_told_history(model, circuit, history, episode_id="contract")
    events = circuit.store.events()
    assert [event.content.decode() for event in events] == ["我叫阿蒙。", "水沸点是多少？"]

    # cue 逐位核对：另一台同种子模型独立重放「问：{轮}」，皮质态必须与入库时那条相同。
    replay = _model()
    replay.reset_dynamics(episode_id="contract-replay")
    replay.observe(replay.config.boundary_symbol, learn=False, readout="predictive")
    for symbol in "问：我叫阿蒙。\n".encode():
        replay.observe(int(symbol), learn=False, readout="predictive")
    assert torch.equal(events[0].cue, replay.cortical_cue())


def test_chat_arm_reproduces_the_product_conversation() -> None:
    """对齐臂的历史与提问文本必须**就是**产品协议铺出来的那一份。"""
    import train_taiji_r2_copy_circuit as trainer

    from api.seed_runtime import SeedRuntime

    model = _model()
    model.mount_copy_circuit(max_events=4)
    circuit = model.copy_circuit
    assert circuit is not None
    turns = ["我叫阿蒙。", "水沸点是多少？", "我的名字是什么？"]
    feed = trainer._run_chat_episode(model, circuit, turns, episode_id="contract-chat")

    assert [user for user, _ in feed.history] == turns[:-1]  # 每轮告知都进了历史
    assert feed.prompt_bytes == SeedRuntime._serialize(turns[-1], list(feed.history)).encode()
    assert feed.tell_bytes == turns[0].encode("utf-8")
    assert feed.prev_byte == feed.prompt_bytes[-1]
    # 提问轮不入库（它是问，不是被告知内容）⇒ 库里的条数＝历史轮数。
    assert [event.content.decode() for event in circuit.store.events()] == turns[:-1]


def test_chat_arm_feed_lands_on_the_generation_chain() -> None:
    """**链路同一性**：训练侧"喂完提问段"的那个状态，必须就是产品生成时的状态。

    否则训出来的寻址/gate 是在另一条链上——A2.4 把能力压住的正是这类分布差。
    判据＝喂完 prompt 的末位 argmax 与产品 `generate(prompt, 1)` 实际吐出的首字节相同。
    """
    import train_taiji_r2_copy_circuit as trainer

    model = _model()
    model.mount_copy_circuit(max_events=4)
    circuit = model.copy_circuit
    assert circuit is not None
    turns = ["我叫阿蒙。", "我的名字是什么？"]
    feed = trainer._run_chat_episode(model, circuit, turns, episode_id="contract-chain")
    trained_byte = int(feed.prompt_probs.detach().cpu().argmax())
    product_byte = int(model.generate(feed.prompt_bytes, 1, stop_at_boundary=False)[0])
    assert trained_byte == product_byte


def test_wrong_event_steps_are_never_trained_against() -> None:
    """多事件库里寻址抢到别条告知 ⇒ **整段不更新**（不喂错标签——rev4 的教训钉成守卫）。

    正反两断言：抢到含答案的那条 ⇒ 必须照常更新（否则这条守卫会变成静默不学习）。
    """
    import train_taiji_r2_copy_circuit as trainer

    from taiji.copy_circuit import ToldEvent

    model = _model()
    model.mount_copy_circuit(max_events=4)
    circuit = model.copy_circuit
    assert circuit is not None
    tell = "我叫阿蒙。".encode()
    other = "今天天气很好。".encode()
    cue = torch.zeros(model.config.cortical_context_dim)
    cue[1] = 1.0
    circuit.store.record(tell, cue.clone())
    model.reset_dynamics(episode_id="contract-wrong")
    model.observe(model.config.boundary_symbol, learn=False, readout="predictive")
    for symbol in "问：我叫阿蒙。\n答：。\n问：我的名字是什么？\n答：".encode():
        model.observe(int(symbol), learn=False, readout="predictive")

    def _train_with(matched: bytes) -> dict[str, Any]:
        win = {"hits": 0, "steps": 0, "gate": [], "adv": [], "wrong_event": 0}
        fake = ToldEvent(event_id=99, content=matched, cue=cue.clone())
        circuit.store.best_match = lambda _query: fake  # type: ignore[method-assign]
        before = _parameters_digest(dict(circuit.parameters()))
        trainer._train_answer(
            model,
            circuit,
            model.config,
            tell_bytes=tell,
            answer="阿蒙",
            prev_byte=other[-1],
            lr_address=0.15,
            lr_gate=0.002,
            win=win,
        )
        win["changed"] = _parameters_digest(dict(circuit.parameters())) != before
        return win

    wrong = _train_with(other)
    assert wrong["steps"] == 0 and wrong["wrong_event"] >= 1 and not wrong["changed"]
    right = _train_with(tell)
    assert right["steps"] > 0 and right["wrong_event"] == 0 and right["changed"]


#: 挂载结果的**基线 digest**，在加 `init_seed` 参数**之前**用当时的代码实测钉下
#: （`Taiji(TaijiConfig(region_sizes=(64,48), synapse_fan_in=16, motor_fan_in=48, seed=S))`
#: ＋ `mount_copy_circuit(max_events=4)`）。用它证明"加种子旋钮没动默认路径"。
MOUNT_BASELINE_DIGEST_BY_CONFIG_SEED = {
    1: "8e0662215c5e61548a4c7f1ea3dbbcb2f68c5325156e520b047be46cc9a7fcba",
    11: "15d07e262f01fb8a1f758f96e19b6818736ac905b3b535171e75d8e74b70929a",
    20260925: "a74b4558705d7155f03bb4cc9441bc31c5db7ea811982ee1a104a57f4cfb65e7",
}


def _small_model(seed: int) -> Taiji:
    return Taiji(TaijiConfig(region_sizes=(64, 48), synapse_fan_in=16, motor_fan_in=48, seed=seed))


def test_init_seed_default_leaves_the_mount_bitwise_identical() -> None:
    """不传 `init_seed` ⇒ 挂载结果逐位等于加参数之前的实测值（三个种子各测一次）。

    口径＝rev5 之前的 7 个张量（`_pre_selector_digest`），新头另行断言恒零。
    """

    for config_seed, expected in MOUNT_BASELINE_DIGEST_BY_CONFIG_SEED.items():
        model = _small_model(config_seed)
        model.mount_copy_circuit(max_events=4)
        circuit = model.copy_circuit
        assert circuit is not None
        assert _pre_selector_digest(dict(circuit.parameters())) == expected, config_seed


def test_circuit_init_seed_moves_only_the_random_projections() -> None:
    """换 `init_seed` 只该动三个随机投影张量；四个零初始化参数必须仍恒零，发射证据仍恒零。

    反向断言同样重要：如果种子偷偷进了门参数，"挂载即位级不变"就被绕开了，
    而这条不变性正是 A2 全部"加性新参数"分账的地基。
    """

    left, right = _small_model(7), _small_model(7)
    left.mount_copy_circuit(max_events=4, init_seed=101)
    right.mount_copy_circuit(max_events=4, init_seed=202)
    params_left = left.copy_circuit.parameters()
    params_right = right.copy_circuit.parameters()
    for name in ("content_embed", "query_state", "query_content"):
        assert not torch.equal(params_left[name], params_right[name]), name
    for name in ("gate_state", "gate_content", "gate_bias", "copy_induce_bias"):
        assert int(torch.count_nonzero(params_left[name])) == 0, name
        assert int(torch.count_nonzero(params_right[name])) == 0, name
    for name in CopyCircuit.OPTIONAL_ZERO_PARAMETERS:
        assert int(torch.count_nonzero(params_left[name])) == 0, name
        assert int(torch.count_nonzero(params_right[name])) == 0, name
    cue = torch.zeros(left.config.cortical_context_dim)
    cue[2] = 1.0
    f1 = torch.zeros(left.config.motor_context_dim)
    f1[3] = 0.5
    left.copy_circuit.store.record(b"abc", cue)
    evidence = left.copy_circuit.evidence(cue=cue, f1_context=f1)
    assert torch.equal(evidence, torch.zeros(left.config.alphabet_size))


def _two_event_store(model: Taiji) -> tuple[CopyCircuit, torch.Tensor, torch.Tensor, torch.Tensor]:
    """挂上电路、喂两段告知＋一个提问，返回 (circuit, cue_第一条, cue_第二条, cue_提问)。"""
    model.reset_dynamics(episode_id="selector")
    model.observe(model.config.boundary_symbol, learn=False, readout="predictive")
    cues: list[torch.Tensor] = []
    for turn in ("我叫明轩。", "我家住在苏州。"):
        for symbol in turn.encode():
            model.observe(int(symbol), learn=False, readout="predictive")
        cues.append(model.cortical_cue())
    for symbol in "你住在哪里？".encode():
        model.observe(int(symbol), learn=False, readout="predictive")
    model.mount_copy_circuit(max_events=4)
    circuit = model.copy_circuit
    assert circuit is not None
    for turn, cue in zip(("我叫明轩。", "我家住在苏州。"), cues):
        circuit.store.record(turn.encode(), cue)
    return circuit, cues[0], cues[1], model.cortical_cue()


def test_zero_head_scores_bitwise_equal_the_legacy_cue_cosine() -> None:
    """§5 守卫②：头零初始化时分数**逐位等于** `best_match` 那条余弦，argmax 同现状。

    这是"挂载不改行为"在选择侧的版本——比 `evidence ≡ 0` 更强：gate 开之后（训练中）
    仍要能保证"没学到的选择器＝原来的选择器"。
    """
    model = _model()
    circuit, cue_a, _cue_b, query_cue = _two_event_store(model)
    f1 = model._state.motor_context.detach().cpu().clone()
    state = circuit.selection(cue=query_cue, f1_context=f1, query_bytes="你住在哪里？".encode())
    assert state is not None
    assert torch.equal(state["scores"], state["features"][:, 0])
    assert float(state["head_values"].abs().sum()) == 0.0
    assert state["event"].event_id == circuit.store.best_match(query_cue).event_id

    # 平手裁决也必须一致：两条告知挂同一个 cue ⇒ 分数真相同，严格大于才换 ⇒ 取更前面那条。
    tied = _model()
    tied.mount_copy_circuit(max_events=4)
    tied_circuit = tied.copy_circuit
    shared = torch.zeros(tied.config.cortical_context_dim)
    shared[1] = 0.7
    shared[3] = -0.4
    tied_circuit.store.record(b"first", shared)
    tied_circuit.store.record(b"second", shared)
    tie_state = tied_circuit.selection(
        cue=shared, f1_context=torch.zeros(tied.config.motor_context_dim), query_bytes=b""
    )
    assert float(tie_state["scores"][0]) == float(tie_state["scores"][1])
    assert tie_state["event"].event_id == 0
    assert tie_state["event"].content == b"first"


def test_selection_lock_freezes_the_event_across_steps() -> None:
    """§5 守卫③：锁上之后换一个偏向另一条告知的 cue，取到的仍是锁住的那一条。

    未锁时同一个换 cue 必须改主意——否则这条守卫是在测一件本来就成立的事。
    这里用"每条告知自己的段末 cue"造对照：cue_A⇒A、cue_B⇒B 是 §1.2 那种逐步漂移的干净形态。
    """
    model = _model()
    circuit, cue_a, cue_b, query_cue = _two_event_store(model)
    f1 = model._state.motor_context.detach().cpu().clone()
    assert circuit.addressing(cue=cue_a, f1_context=f1)["event"].event_id == 0
    assert circuit.addressing(cue=cue_b, f1_context=f1)["event"].event_id == 1

    lock = circuit.lock_selection(cue=query_cue, f1_context=f1, query_bytes="你住在哪里？".encode())
    assert lock is not None
    assert circuit.locked_event_id == lock["event"].event_id
    frozen = lock["event"].event_id
    for cue in (cue_a, cue_b, query_cue):
        assert circuit.addressing(cue=cue, f1_context=f1)["event"].event_id == frozen
        assert torch.equal(
            circuit.evidence(cue=cue, f1_context=f1),
            circuit.evidence(cue=query_cue, f1_context=f1),
        )
    assert circuit.selection_lock_dropped == 0

    # 锁指向的事件被淘汰：必须**响亮**地计数并就地弃锁（不静默换路）。
    circuit.store.clear()
    circuit.store.record("我叫明轩。".encode(), cue_a)
    assert circuit.addressing(cue=cue_b, f1_context=f1) is not None
    assert circuit.selection_lock_dropped == 1
    assert circuit.locked_event_id is None


def test_selector_head_learns_toward_the_told_event_and_loads_legacy_payload() -> None:
    """标签来自题面那条告知；一步更新后头离开零，旧 payload（无 rev5 两项）仍可载入。"""
    model = _model()
    circuit, cue_a, _cue_b, query_cue = _two_event_store(model)
    f1 = model._state.motor_context.detach().cpu().clone()
    state = circuit.selection(cue=query_cue, f1_context=f1, query_bytes="你住在哪里？".encode())
    before = circuit.parameters()["selector_weight"].detach().cpu().clone()
    target = 1 - int(state["event"].event_id)
    circuit.learn_selection(state, target_event_id=target, advantage=1.0, lr_selector=0.5)
    moved = circuit.parameters()["selector_weight"].detach().cpu().clone()
    assert not torch.equal(before, moved)
    assert float(moved.abs().sum()) > 0.0
    # 训练后头必须能把选择翻到标签那条（分数不再逐位等于余弦行）。
    after = circuit.selection(cue=query_cue, f1_context=f1, query_bytes="你住在哪里？".encode())
    assert not torch.equal(after["scores"], after["features"][:, 0])
    assert after["event"].event_id == target

    payload = circuit.to_payload()
    legacy_parameters = {
        name: value
        for name, value in payload["parameters"].items()
        if name not in CopyCircuit.OPTIONAL_ZERO_PARAMETERS
    }
    fresh = _model()
    fresh.mount_copy_circuit(max_events=4)
    fresh.copy_circuit.load_payload({**payload, "parameters": legacy_parameters})
    assert float(fresh.copy_circuit.parameters()["selector_weight"].abs().sum()) == 0.0
    fresh.copy_circuit.load_payload(payload)
    assert torch.equal(fresh.copy_circuit.parameters()["selector_weight"], moved)
    assert fresh.copy_circuit.locked_event_id is None
    with pytest.raises(ValueError, match="missed parameters"):
        fresh.copy_circuit.load_payload(
            {
                **payload,
                "parameters": {
                    name: value
                    for name, value in payload["parameters"].items()
                    if name != "query_state"
                },
            }
        )
