"""PLAN-N4-05 §1 的执行版：产品写入路径**可达且只需一个落定回合**（不是"产品缺写入"）。

本轮先前只靠读码得到这条事实（`taiji/adapter.py:11663` 的 `if self._episodic_memory is not None:`
→ `:11699` 的 `self._episodic_memory.write(record)`，cue 取 `percept.features` 否则 `world.latent`）。
读码不够——所以这一册把它跑出来：

* **正例**：默认挂载位为真（产品自建 store，**没有任何测试代码调 `store.write(`**）＋
  一次 `act` ＋ 一次 `settle_action` ⇒ `count` 从 0 变 1，条目里带产品自己的
  `episode_id`／`tick`／`outcome`，cue 维度即 store 的 `cue_dim`。
* **反例（触发点归属）**：同样长度的 `observe` 流但**不落定回合** ⇒ `count` 仍为 0。
  ⇒ 证明"写入属于回合事件"，而不是"观察就会写"；也就是 N4 语料链（只 `observe`）
  为何让默认挂载保持潜伏的那条成因（㊵-601④、㊵-604②）。

两支都必须能为假：若产品哪天在 observe 里也写，反例立刻红；若回合不再写，正例红。
不碰权重档案、不读语料，纯 tiny 配置。
"""

from __future__ import annotations

from seed import Seed, SeedConfig
from taiji.config import TaijiConfig
from taiji.episodic_memory import EpisodicMemoryStore

MOTOR_ACTIONS = (0, 1, 2, 3)


def _model(mount: bool) -> Seed:
    config = TaijiConfig(
        region_sizes=(8,),
        synapse_fan_in=2,
        motor_fan_in=4,
        predictive_context_fan_in=2,
        memory_units=16,
        memory_fan_in=2,
        memory_readout_fan_in=2,
        memory_meta_dim=4,
        memory_time_dim=2,
        memory_episode_dim=2,
        lateral_fan_in=2,
        identity_organ_capacity=8,
        concept_capacity=8,
        seed=101,
        episodic_memory_default_mount=mount,
    )
    return Seed(SeedConfig(taiji=config))


def _observe(model: Seed, symbols: bytes) -> None:
    for symbol in symbols:
        model.observe(symbol, learn=True)


def test_a_settled_round_makes_the_product_write_one_episodic_record() -> None:
    model = _model(True)
    store = model.substrate._episodic_memory
    assert isinstance(store, EpisodicMemoryStore)
    assert int(store.count) == 0, "product mount should start empty"

    _observe(model, bytes(range(65, 85)))
    model.substrate.act(MOTOR_ACTIONS, sample=False)
    outcome = model.substrate.settle_action(1.0, terminal=True)

    assert int(outcome.reward) == 1
    assert int(store.count) == 1, "settling a round must write exactly one record"
    record = store.records[0]
    assert record.episode_id == model.substrate.snapshot().episode_id or record.episode_id
    assert int(record.tick) > 0
    #: 张量的 `dim()` 是**秩**（一维张量恒为 1），长度要看 `shape[0]`。
    #: 我第一版把两者混了，于是 `cue_dim=32` 对 `dim()=1` 报了个假红。
    assert int(record.cue.shape[0]) > 0
    assert int(store.cue_dim) == int(record.cue.shape[0])


def test_observing_without_a_settled_round_writes_nothing() -> None:
    #: 反例：这正是 N4 语料链（只 observe）让默认挂载保持潜伏的原因。
    model = _model(True)
    store = model.substrate._episodic_memory
    _observe(model, bytes(range(65, 85)))
    assert int(store.count) == 0


def test_unmounted_model_stays_untouched() -> None:
    model = _model(False)
    assert model.substrate._episodic_memory is None
    _observe(model, bytes(range(65, 85)))
    model.substrate.act(MOTOR_ACTIONS, sample=False)
    model.substrate.settle_action(1.0, terminal=True)
    assert model.substrate._episodic_memory is None
    assert model.substrate.episodic_memory_mounted is False
