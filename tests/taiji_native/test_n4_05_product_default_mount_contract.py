"""PLAN-N4-04 §3 步骤 3 的实施契约测：产品默认挂载位（缺省关）＋四条硬断言。

前置：反例探针 4 支全绿（`test_n4_04_pre_implementation_probe_contract.py`）才允许动产品码；
本篇与实现同批提交（不拆"先加字段后补消费"）。

基线摘要 `71afabbc9b7dc0990da6b10bfc036bee049f4a18170596c1a427b41eb696a126` 的**出处**：
`git archive HEAD taiji seed` 解到仓外后，用同一 tiny 配置与同一 64 字节输入
（`bytes(range(32,96))`、`readout="predictive"`、`learn=True`）跑出的观测摘要；
改动后的工作树在同一算法下给出**同一个值** ⇒ J-N4d-2「默认关时逐位不变」是被真跑对表过的，
不是我的一句承诺。若有人改动默认关路径，本册两支摘要断言都会当场红。

另钉一条冒烟才摊得到的事实：**挂载一个空库不改变观测路径**（同一摘要）。
这不是"挂载没生效"——store 在位、容量生效；而是本架构的检索要求 `count > 0`
（"寻址失败≡空库"那条债），所以默认挂载在有人写入之前是**潜伏**效果。
后继格必须自己证明写入发生，不许拿"挂载为真"当行为改变的证据。
"""

from __future__ import annotations

import dataclasses
import subprocess
from pathlib import Path

import pytest

from seed import Seed, SeedConfig
from taiji.config import TaijiConfig
from taiji.internalization import content_digest

REPO = Path(__file__).resolve().parents[2]
BASELINE_DIGEST = "71afabbc9b7dc0990da6b10bfc036bee049f4a18170596c1a427b41eb696a126"
PROBE_INPUT = bytes(range(32, 96))


def _tiny_config(**over: object) -> TaijiConfig:
    kwargs: dict[str, object] = dict(
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
    )
    kwargs.update(over)
    return TaijiConfig(**kwargs)  # type: ignore[arg-type]


def _digest(model: Seed, symbols: bytes) -> str:
    rows = []
    for symbol in symbols:
        step = model.observe(symbol, learn=True, readout="predictive")
        prior = getattr(step, "prior_prediction", None)
        surprise = getattr(step, "surprise", None)
        rows.append(
            [
                None if prior is None else int(prior),
                None if surprise is None else round(float(surprise), 12),
            ]
        )
    return str(content_digest({"rows": rows}))


def _substrate(**over: object):
    return Seed(SeedConfig(taiji=_tiny_config(**over))).substrate


def test_default_is_off_and_nothing_is_mounted() -> None:
    config = _tiny_config()
    assert config.episodic_memory_default_mount is False
    assert config.episodic_memory_capacity == 1024
    substrate = _substrate()
    assert substrate.episodic_memory_default_mount is False
    assert substrate.episodic_memory_mounted is False


def test_default_on_mounts_a_store_with_the_configured_capacity() -> None:
    substrate = _substrate(episodic_memory_default_mount=True, episodic_memory_capacity=7)
    #: J-N4d-1：默认位与实际值同源相等，且两枚都由产品一侧自述（不是 harness 补的）。
    assert substrate.episodic_memory_default_mount is True
    assert substrate.episodic_memory_mounted is True
    assert substrate._episodic_memory is not None
    assert int(substrate._episodic_memory.capacity) == 7
    assert int(substrate._episodic_memory.count) == 0


def test_default_off_observation_digest_equals_the_head_baseline() -> None:
    #: J-N4d-2：值抄自仓外 HEAD 树真跑；这条能为假（任何动到默认关路径的改动都会撞它）。
    model = Seed(SeedConfig(taiji=_tiny_config()))
    assert _digest(model, PROBE_INPUT) == BASELINE_DIGEST


def test_mounting_an_empty_store_is_a_latent_effect_not_a_behaviour_change() -> None:
    model = Seed(
        SeedConfig(
            taiji=_tiny_config(episodic_memory_default_mount=True, episodic_memory_capacity=9)
        )
    )
    assert model.substrate.episodic_memory_mounted is True
    assert _digest(model, PROBE_INPUT) == BASELINE_DIGEST


def test_invalid_mount_knobs_are_rejected_loudly() -> None:
    with pytest.raises(ValueError, match="episodic_memory_capacity must be positive"):
        _tiny_config(episodic_memory_capacity=0)
    with pytest.raises(TypeError, match="episodic_memory_default_mount must be a bool"):
        _tiny_config(episodic_memory_default_mount="yes")


def test_attach_call_sites_count_seven_after_the_product_assembly() -> None:
    #: J-N4d-4：数的是 `git grep` 的真实命中行（原 6 枚 harness/训练调用点＋产品装配链 1 枚）。
    listing = subprocess.run(  # noqa: S603 - 仓内固定命令，参数不含用户输入
        [
            "git",
            "grep",
            "-c",
            "-e",
            r"\.attach_episodic_memory(",
            "--",
            "taiji",
            "scripts",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        check=False,
    )
    assert listing.returncode == 0, listing.stderr[-300:]
    counts = {}
    for line in listing.stdout.splitlines():
        path, _, value = line.rpartition(":")
        counts[path] = int(value)
    assert sum(counts.values()) == 7, counts
    assert counts.get("taiji/adapter.py", 0) == 1, counts


def test_python_explains_the_mount_layer_choice_not_the_seed_wrapper() -> None:
    #: 探针实测过 attach 只在底座；实施沿用同一层 ⇒ Seed 仍不新增该方法（暴露层次是自述的一部分）。
    model = Seed(SeedConfig(taiji=_tiny_config()))
    assert not hasattr(model, "attach_episodic_memory")
    assert hasattr(model.substrate, "attach_episodic_memory")
    names = {field.name for field in dataclasses.fields(TaijiConfig)}
    assert "episodic_memory_default_mount" in names
