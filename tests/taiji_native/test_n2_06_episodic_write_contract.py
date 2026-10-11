"""DEBT-G99／G98／G100 的写入合同契约测（㊵-681，owner #25 丙裁"三笔一起批"）。

本册钉的是 VISION §5.3「过期、错误和重复经历有撤销/去重规则」与 §19.6「未经核验的生成内容
保留 generated 来源，不混进事实/奖励」在仓内的**最小可判形状**。
每条断言都必须能为假，且写清它钉的是哪一句——不是"测很多所以好"。

先冻在 `PLAN-N2-06` §3（J-N2f-1..5）与 `PLAN-N5-11` §3（J-N5k-1..5），实施顺序＝
G99（纯新增）→ G100（枚举加末位取值＋宽度迁移）→ G98（取消隐式来源）。
"""

from __future__ import annotations

import hashlib

import pytest
import torch

from taiji import Taiji, TaijiConfig
from taiji.contracts import EpisodicMemoryRecord, Outcome
from taiji.episodic_memory import EpisodicMemoryStore


def _config() -> TaijiConfig:
    return TaijiConfig(region_sizes=(32, 24), synapse_fan_in=8, motor_fan_in=16, seed=19)


def _record(
    memory_id: str, *, cue: list[float] | None = None, provenance: str = "experienced"
) -> EpisodicMemoryRecord:
    return EpisodicMemoryRecord(
        memory_id=memory_id,
        episode_id=f"episode-{memory_id}",
        tick=1,
        cue=torch.tensor(cue if cue is not None else [1.0, 0.0, 0.0, 0.0]),
        provenance=provenance,
    )


def _ids(store: EpisodicMemoryStore) -> list[str]:
    return [
        hit.record.memory_id for hit in store.retrieve(torch.tensor([1.0, 0.0, 0.0, 0.0]), limit=8)
    ]


def _weight_digest(model: Taiji) -> dict[str, str]:
    """按序列化产物取摘要：递归走 `checkpoint()` 里的每一枚张量。

    `Taiji` 不是 `nn.Module`，没有 `state_dict()`；而 `checkpoint()` 就是权重真正落盘的那一份，
    拿它做逐位对照比自造枚举器更贴近"装机后会被读到的东西"。
    """
    found: dict[str, str] = {}

    def walk(node: object, path: str) -> None:
        if isinstance(node, torch.Tensor):
            found[path] = hashlib.sha256(
                node.detach().cpu().contiguous().numpy().tobytes()
            ).hexdigest()
        elif isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{path}.{key}")
        elif isinstance(node, (list, tuple)):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")

    walk(model.checkpoint(), "root")
    assert found, "checkpoint() 里没有张量，这台对照器就没在量东西"
    return found


# --------------------------------------------------------------- G99：按 id 撤销 E


def test_expire_removes_the_named_record_from_retrieval() -> None:
    store = EpisodicMemoryStore(capacity=4, cue_dim=4)
    store.write(_record("keep"))
    store.write(_record("drop", cue=[0.0, 1.0, 0.0, 0.0]))
    assert "drop" in _ids(store), _ids(store)
    assert store.expire("drop") == 1
    assert "drop" not in _ids(store), _ids(store)
    assert store.expired_memory_ids == ("drop",)


def test_expire_reports_zero_when_the_id_is_absent_rather_than_silently_succeeding() -> None:
    """ "没删到"必须是可读出来的数，不许和"删到了"共用一个返回值。"""
    store = EpisodicMemoryStore(capacity=4, cue_dim=4)
    store.write(_record("keep"))
    assert store.expire("never-stored") == 0
    assert store.expired_memory_ids == ("never-stored",)
    assert store.count == 1


def test_expire_rejects_an_empty_identifier() -> None:
    store = EpisodicMemoryStore(capacity=4, cue_dim=4)
    with pytest.raises(ValueError, match="memory_id"):
        store.expire("")


def test_capacity_eviction_is_counted_separately_from_expiry() -> None:
    store = EpisodicMemoryStore(capacity=2, cue_dim=4)
    for index in range(3):
        store.write(_record(f"m-{index}", cue=[float(index + 1), 0.0, 0.0, 0.0]))
    assert store.evicted_count == 1, store.evicted_count
    assert store.expired_memory_ids == (), store.expired_memory_ids


def test_checkpoint_round_trip_carries_both_counters_and_survives_a_legacy_payload() -> None:
    store = EpisodicMemoryStore(capacity=4, cue_dim=4)
    store.write(_record("a"))
    store.write(_record("b"))
    store.expire("a")
    payload = store.checkpoint()
    assert payload["expired_memory_ids"] == ["a"], payload
    assert payload["evicted_count"] == 0, payload
    revived = EpisodicMemoryStore.from_checkpoint(payload)
    assert revived.expired_memory_ids == ("a",)
    assert revived.count == 1
    # 旧形状（没有这两枚键）必须照样读得动，读不到就等于把历史检查点判成坏件。
    legacy = {k: v for k, v in payload.items() if k not in ("expired_memory_ids", "evicted_count")}
    old_shape = EpisodicMemoryStore.from_checkpoint(legacy)
    assert old_shape.expired_memory_ids == ()
    assert old_shape.evicted_count == 0


def test_revoking_episodic_records_does_not_touch_weights() -> None:
    """E 删除 != θ 删除：撤销一条经历之后，权重摘要必须逐位相同（也反向证"不许承诺自动遗忘"）。"""
    model = Taiji(_config())
    before = _weight_digest(model)
    store = EpisodicMemoryStore(capacity=4, cue_dim=4)
    store.write(_record("fact"))
    assert store.expire("fact") == 1
    assert _weight_digest(model) == before


# ------------------------------------------------- G98：来源必须显式给出且落在白名单内


def test_provenance_has_no_implicit_default_anymore() -> None:
    """不给来源就构造不出来。这条为假的那一面是"沉默地当成 experienced"——就是被修掉的缺陷。"""
    with pytest.raises(TypeError):
        EpisodicMemoryRecord(
            memory_id="m-2",
            episode_id="e-2",
            tick=1,
            cue=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        )


def test_source_labels_are_an_open_vocabulary_not_a_closed_list() -> None:
    """来源标签是**开放词表**——这一支是把 12 枚红测出来的那次改判。

    我起初按 config 里的 EPISODIC_PROVENANCE_KINDS 做闭集校验，实测撞掉三族：
    scripts/training/measure_taiji_n4_addressing_surface.py:144 送 n4-01-* 标签、
    taiji/artifact_internalization.py:537/539 送 artifact-outcome 与 artifact-internalized
    （全仓 provenance 实参共 90 处）。⇒ 闭集是我把"情节场的 4 个编码通道种类"误当成
    "来源标签"：通道侧校验仍归 EpisodicField.PROVENANCE_KINDS（taiji/model.py:2612 原样在），
    记录层只要求**显式声明**。这条测钉的就是"外部标签必须照样收"，防止将来有人再收窄。
    """
    label = "artifact-outcome"
    assert _record("m-open", cue=[1.0, 0.0, 0.0, 0.0], provenance=label).provenance == label


def test_legacy_payload_without_provenance_still_loads_but_is_counted() -> None:
    """历史件不许被追认成坏件；但"这条来源是补出来的"必须出版成一枚读数。"""
    store = EpisodicMemoryStore(capacity=4, cue_dim=4)
    store.write(_record("a"))
    payload = store.checkpoint()
    for item in payload["records"]:
        item.pop("provenance", None)
    revived = EpisodicMemoryStore.from_checkpoint(payload)
    assert revived.count == 1
    assert revived.implicit_provenance_count == 1
    assert revived.checkpoint()["implicit_provenance_count"] == 1
    assert EpisodicMemoryStore.from_checkpoint(store.checkpoint()).implicit_provenance_count == 0


# --------------------------------------------- G100：generated 在记录层成立且不进奖励账


def test_generated_is_a_source_value_a_record_can_carry() -> None:
    record = _record("self-answer", cue=[1.0, 0.0, 0.0, 0.0])
    generated = EpisodicMemoryRecord(
        memory_id="g-1",
        episode_id="e-g",
        tick=1,
        cue=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        provenance="generated",
    )
    assert generated.provenance == "generated"
    assert record.provenance == "experienced"


def test_generated_record_cannot_book_a_nonzero_reward() -> None:
    """VISION §19.6「不混进事实/奖励」的可判形式：generated + 非零 outcome.reward = 构造期拒绝。"""
    from taiji.contracts import Outcome

    with pytest.raises(ValueError, match="non-zero reward"):
        EpisodicMemoryRecord(
            memory_id="g-2",
            episode_id="e-g2",
            tick=3,
            cue=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            outcome=Outcome(intent_id="i-2", reward=1.0, success=True, tick=3),
            provenance="generated",
        )
    # 反向那一支也要走：零奖励的 generated 记录必须能构造出来，否则禁令成了"禁止该来源存在"。
    allowed = EpisodicMemoryRecord(
        memory_id="g-3",
        episode_id="e-g3",
        tick=3,
        cue=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        outcome=Outcome(intent_id="i-3", reward=0.0, success=False, tick=3),
        provenance="generated",
    )
    assert allowed.provenance == "generated"


def test_the_associative_field_channel_vocabulary_is_still_four_channels() -> None:
    """把"张量层没跟着加宽"这件事钉成断言，防止有人日后偷偷改通道数。

    `generated` 今天只活在记录层：情节场的编码通路仍是 4 通道，接它需要一次受体通道重映射
    迁移（会作废盘上 230 枚检查点与槽位钉子）⇒ 那是要另批的结构变更，记 DEBT-G104。
    """
    model = Taiji(_config())
    assert model.memory.PROVENANCE_KINDS == ("experienced", "imagined", "replayed", "external")
    assert model.memory.provenance_encoder.in_features == 4


def test_contradictory_facts_about_one_cue_stay_two_records_not_a_silent_merge() -> None:
    """矛盾事实（§5.3 测试清单）：同一条线索上两条互斥结果必须各自在库、各自可读。"""
    cue = [1.0, 0.5, 0.0]
    store = EpisodicMemoryStore(capacity=4, cue_dim=3)
    for memory_id, value in (("fact-yes", 1.0), ("fact-no", -1.0)):
        store.write(
            EpisodicMemoryRecord(
                memory_id=memory_id,
                episode_id=f"episode-{memory_id}",
                tick=1,
                cue=torch.tensor(cue),
                outcome=Outcome(
                    intent_id=f"intent-{memory_id}", reward=value, success=value > 0, tick=1
                ),
                provenance="experienced",
            )
        )
    assert store.count == 2, store.count
    rewards = {
        hit.record.memory_id: float(hit.record.outcome.reward)
        for hit in store.retrieve(torch.tensor(cue), limit=4)
    }
    assert rewards == {"fact-yes": 1.0, "fact-no": -1.0}, rewards


def test_the_same_record_written_twice_does_not_double_count() -> None:
    """重复输入：只声明"不重复计数"，不声明内容级去重（后者今天不存在，见 DEBT-G106）。"""
    cue = [0.5, 0.5, 0.0]
    store = EpisodicMemoryStore(capacity=4, cue_dim=3)
    for _ in range(2):
        store.write(
            EpisodicMemoryRecord(
                memory_id="dup",
                episode_id="episode-dup",
                tick=1,
                cue=torch.tensor(cue),
                provenance="experienced",
            )
        )
    assert store.count == 1, store.count


def test_orthogonal_and_zero_cues_score_zero_and_cannot_win_a_ranking() -> None:
    """噪声：零范数线索走 `query_norm <= 1e-8` 那一支，所有分数记 0（仪器性质，非能力）。"""
    store = EpisodicMemoryStore(capacity=4, cue_dim=3)
    store.write(
        EpisodicMemoryRecord(
            memory_id="signal",
            episode_id="episode-signal",
            tick=1,
            cue=torch.tensor([0.0, 1.0, 0.0]),
            provenance="experienced",
        )
    )
    for noise in ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]):
        hits = store.retrieve(torch.tensor(noise), limit=4)
        assert hits, noise
        assert all(hit.score == 0.0 for hit in hits), (noise, [h.score for h in hits])


def test_retrieval_ignores_episode_boundaries_documented_not_endorsed() -> None:
    """跨会话不泄漏今天**不成立**：`retrieve` 不按 `episode_id` 过滤。

    这支刻意钉"当前缺陷形状"而不是声称隔离成立——将来接了隔离它会红，那时按 DEBT-G106
    的结案方向改判据并升版；现在假装通过就是把判据写成愿望。
    """
    cue = [0.0, 0.0, 1.0]
    store = EpisodicMemoryStore(capacity=4, cue_dim=3)
    store.write(
        EpisodicMemoryRecord(
            memory_id="s1",
            episode_id="session-1",
            tick=1,
            cue=torch.tensor(cue),
            provenance="experienced",
        )
    )
    assert [hit.record.episode_id for hit in store.retrieve(torch.tensor(cue), limit=4)] == [
        "session-1"
    ]
