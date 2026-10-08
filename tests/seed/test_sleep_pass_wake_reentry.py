"""DEBT-G47 修法甲的契约测：睡眠巩固跑完之后，organism 必须**醒得来**，且醒这一步不动权重。

缺陷本体（本轮 N2 通电当场实测，`reports/taiji_n2_postcheck_20261008.json`）：
`SeedSleepScheduler.night` 里每次 experience／observation 都把 episode 换成 `sleep-*`，
而 `native_checkpoint()` 的 `cognitive_state` 半边只跟到最后一次同步 ⇒ 信封两半自相矛盾
（实测 cognitive tick 66／kernel tick 92），`TaijiKernel.restore_native` 末尾那条守卫
（taiji/adapter.py:12631-12632）据此把整枚候选档**拒收**——产品自己存的巩固档产品自己装不回来。

三支测各有各的用处：
1. **反支（今天就是红的）**：直接调 `night` 而不收束 ⇒ 装载必须被拒。这条把缺陷本体钉住；
2. **正支**：走产品路径（`sleep_pass.run(organs=True, learn=True)`）⇒ 必须醒得来、信封两半一致、
   `Seed.from_checkpoint` 必须装得回来；
3. **反向守卫（owner 批文随修法一起定的那条）**：收束这一步**不得改动任何张量**——
   否则"能装回来"是靠抹掉有效状态换来的。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from seed import Seed, SeedConfig
from seed.judge import SeedJudge
from seed.sleep import SeedSleepScheduler
from seed_platform import memory_store, sleep_pass, turn_records
from taiji.config import TaijiConfig


@pytest.fixture()
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """把外部数据根指到一棵临时树（与 `test_sleep_pass.py` 同一形状）。"""

    from api.training import datasets as training_datasets

    for module in (sleep_pass, turn_records, memory_store, training_datasets):
        monkeypatch.setattr(module, "get_external_path", lambda rel: str(tmp_path / rel))
    turn_records._SEEN_CONSTRAINTS.clear()
    memory_store._SEEN_DIGESTS.clear()
    return tmp_path


def _small_config() -> SeedConfig:
    return SeedConfig(
        taiji=TaijiConfig(
            region_sizes=(12, 8),
            synapse_fan_in=4,
            motor_fan_in=6,
            memory_units=16,
            memory_fan_in=4,
            memory_readout_fan_in=6,
            memory_meta_dim=6,
            memory_iterations=2,
            memory_time_dim=4,
            memory_episode_dim=4,
            lateral_fan_in=4,
            seed=45,
        )
    )


class _StubRuntime:
    """只带 `model` 的最小 runtime——`sleep_organs` 要的就是这一个成员。"""

    def __init__(self, model: Seed) -> None:
        self.model = model


def _learned_seed(episode_id: str) -> Seed:
    model = Seed(_small_config(), episode_id=episode_id)
    model.learn_bytes(b"ababcdcdabcd efgh", epochs=6)
    return model


def _halves(checkpoint: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    taiji = checkpoint.get("taiji") or {}
    cognitive = taiji.get("cognitive_state") or {}
    kernel_state = (taiji.get("kernel") or {}).get("state") or {}
    return (
        cognitive.get("tick"),
        cognitive.get("episode_id"),
        kernel_state.get("tick"),
        kernel_state.get("episode_id"),
    )


def _tensors(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    found: dict[str, torch.Tensor] = {}

    def walk(node: Any, prefix: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{prefix}.{key}")
        elif torch.is_tensor(node):
            found[prefix] = node

    walk(checkpoint, "")
    return found


def test_night_alone_leaves_an_envelope_the_loader_refuses() -> None:
    """反支：不收束就是装不回来。这条今天必须红——它钉的是缺陷本体，不是我的写法。"""

    model = _learned_seed("g47-negative")
    scheduler = SeedSleepScheduler(model, SeedJudge(model))
    scheduler.night([b"ababcdcdabcd ababcdcdabcd"], cycles_per_text=1, learn=True, max_symbols=24)

    cognitive_tick, cognitive_episode, kernel_tick, kernel_episode = _halves(model.checkpoint())
    assert cognitive_episode == kernel_episode == "sleep-experience"
    assert cognitive_tick != kernel_tick, "两半已一致 ⇒ 缺陷被修在更深处，这条守卫该换形状"
    with pytest.raises(ValueError, match="out of sync with kernel state"):
        Seed.from_checkpoint(model.checkpoint())


def test_product_sleep_pass_returns_to_a_loadable_waking_state(workspace: Path) -> None:
    """正支：走产品路径的巩固跑完之后，同一枚信封必须能被自己的装载器读回来。"""

    memory_store.record(
        "interaction",
        "问：一\n答：一 ababcdcdabcd",
        session_id="s1",
        turn=1,
        source="taiji-harness",
    )
    model = _learned_seed("g47-positive")

    report = sleep_pass.run(
        reason="g47",
        organs=True,
        learn=True,
        runtime=_StubRuntime(model),
        max_texts=1,
        cycles_per_text=1,
        max_symbols=24,
    )

    organs = report["organs"]
    assert organs["ran"] is True, organs
    assert organs["learn"] is True
    assert organs["wake_error"] == "", organs
    assert organs["wake_episode"] == sleep_pass.WAKE_EPISODE_ID

    checkpoint = model.checkpoint()
    cognitive_tick, cognitive_episode, kernel_tick, kernel_episode = _halves(checkpoint)
    assert (cognitive_tick, cognitive_episode) == (kernel_tick, kernel_episode)
    restored = Seed.from_checkpoint(checkpoint)
    assert isinstance(restored, Seed)


def test_wake_reentry_changes_no_learned_weight() -> None:
    """反向守卫（owner 批文随修法一起定的那条）：收束只准动**回合态**，准不得动已学权重。

    实测形状（本机小模型）：`reset_dynamics` 之后有 24 枚张量变了，全部落在
    `state`／`cognitive_state`／`components`／`perception` 这些**这一回合的活动态**里
    （docstring 自己就写着 "Clear activity while preserving all learned synapses"）。
    ⇒ 所以这条守卫不能拿"整个信封一字不动"去判（那是假断言，一判就红），
    而要先**证明选择器非空**，再要求被选中的那些已学突触载荷逐位不变。
    """

    model = _learned_seed("g47-reverse")
    scheduler = SeedSleepScheduler(model, SeedJudge(model))
    scheduler.night([b"ababcdcdabcd ababcdcdabcd"], cycles_per_text=1, learn=False, max_symbols=24)

    before = _tensors(model.checkpoint())
    model.reset_dynamics(episode_id=sleep_pass.WAKE_EPISODE_ID)
    after = _tensors(model.checkpoint())

    #: 已学突触的载荷形状＝稀疏银行的 `edge_weight` + `pre_index`（+ `post_index`，若该器官有）。
    learned = sorted(
        key
        for key in before
        if "edge_weight" in key or key.endswith(".pre_index") or key.endswith(".post_index")
    )
    assert len(learned) >= 10, f"选择器只挑到 {len(learned)} 枚 ⇒ 这条守卫已经变成空判，先修测"
    #: 回合态里"字段消失"是合法的（percept 的 features 会被置 None），**已学权重那一组不许消失**。
    lost_learned = sorted(key for key in learned if key not in after)
    assert lost_learned == [], f"收束步骤让已学权重载荷消失：{lost_learned[:5]}"

    touched = [
        key
        for key in learned
        if before[key].shape != after[key].shape or not torch.equal(before[key], after[key])
    ]
    assert touched == [], f"收束步骤改动了已学权重载荷：{touched[:5]}"

    #: 同一次调用必须真的把回合态动了——否则"权重没动"可能只是因为这一步什么都没做。
    #: 回合态里"字段消失"是合法的一种"动了"（percept 的 features 会被置 None），所以两条都算。
    state_keys = sorted(key for key in before if key not in learned)
    dropped = sorted(key for key in state_keys if key not in after)
    cleared = [
        key
        for key in state_keys
        if key in after
        and before[key].shape == after[key].shape
        and not torch.equal(before[key], after[key])
    ]
    assert cleared or dropped, "收束步骤对整枚信封零改动 ⇒ 它没在做事，正支的'醒得来'要重新验"
