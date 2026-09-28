"""A-4 守卫：主训练线的实验开关必须"默认关 ⇒ 逐键不变"，且各自只翻自己那一键。

来历：A 支线把三处**默认关**的实验开关（`receptors_factored`／`predictive_context_region0_only`
／`readout_utf8_position_input`）接给主训练线 `train_seed_corpus.py`，让主线能直接跑
"用它已经证过的部件"这一档。接线抽成纯函数 `apply_experiment_flags` 就是为了能被这里钉住——
否则一次手滑就能让"默认关"变成名义上的关。

三条：① 全 False ⇒ config **逐键等于入参**；② 单个 True 只翻自己那一键；
③ 三个字段在 `TaijiConfig` 上的默认值都是 False（防线在源头）。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from seed import SeedConfig
from taiji import TaijiConfig

RUNNER = Path(__file__).resolve().parents[2] / "scripts" / "training" / "train_seed_corpus.py"

FLAGS = {
    "receptors_factored": "receptors_factored",
    "predictive_context_region0_only": "predictive_context_region0_only",
    "readout_position": "readout_utf8_position_input",
}


def _module():
    spec = importlib.util.spec_from_file_location("train_seed_corpus_under_test", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def trainer():
    return _module()


def _small_config() -> SeedConfig:
    return SeedConfig(
        taiji=TaijiConfig(region_sizes=(16,), synapse_fan_in=4, motor_fan_in=8)
    )


def test_source_fields_default_to_false() -> None:
    config = TaijiConfig()
    for field in FLAGS.values():
        assert getattr(config, field) is False, field


def test_all_flags_off_returns_the_same_config_key_by_key(trainer) -> None:
    config = _small_config()
    returned = trainer.apply_experiment_flags(config)
    assert returned.to_dict() == config.to_dict()


@pytest.mark.parametrize("flag,field", sorted(FLAGS.items()))
def test_single_flag_flips_exactly_its_own_key(trainer, flag, field) -> None:
    config = _small_config()
    returned = trainer.apply_experiment_flags(config, **{flag: True})
    before = config.to_dict()["taiji"]
    after = returned.to_dict()["taiji"]
    assert after[field] is True
    changed = {key for key in before if before[key] != after[key]}
    assert changed == {field}, changed


def test_flags_compose_without_touching_other_keys(trainer) -> None:
    config = _small_config()
    returned = trainer.apply_experiment_flags(
        config, predictive_context_region0_only=True, readout_position=True
    )
    before = config.to_dict()["taiji"]
    after = returned.to_dict()["taiji"]
    changed = {key for key in before if before[key] != after[key]}
    assert changed == {
        "predictive_context_region0_only",
        "readout_utf8_position_input",
    }, changed