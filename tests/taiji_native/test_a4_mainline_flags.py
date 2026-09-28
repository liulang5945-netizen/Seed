"""A-4 守卫：主训练线的实验开关必须"默认关 ⇒ 逐键不变"，且各自只翻自己那一键。

来历：A 支线把三处**默认关**的实验开关（`receptors_factored`／`predictive_context_region0_only`
／`readout_utf8_position_input`）接给主训练线 `train_seed_corpus.py`，让主线能直接跑
"用它已经证过的部件"这一档。接线抽成纯函数 `apply_experiment_flags` 就是为了能被这里钉住——
否则一次手滑就能让"默认关"变成名义上的关。

三条：① 全 False ⇒ config **逐键等于入参**；② 单个 True 只翻自己那一键；
③ 三个字段在 `TaijiConfig` 上的默认值都是 False（防线在源头）。

追加（PLAN-A-26 热启动）：`patch_envelope_config_flags` 必须翻遍受载档里**三处** config 副本
（`config.taiji`／`substrate.config`／`taiji.kernel.config`）、只翻点名键，并且**正是它**解开了
`热启动→restore` 的"档配比架构"守卫——不加补丁则 restore 必须响亮失败（否则这层守卫形同虚设）。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from seed import Seed, SeedConfig
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


# --- PLAN-A-26 热启动：信封补丁 -------------------------------------------------


def _envelope() -> dict:
    return Seed(_small_config()).checkpoint()


def test_patch_touches_all_three_config_copies_and_only_named_keys(trainer) -> None:
    envelope = _envelope()
    before = {
        "seed": dict(envelope["config"]["taiji"]),
        "substrate": dict(envelope["substrate"]["config"]),
        "kernel": dict(envelope["taiji"]["kernel"]["config"]),
    }
    patched = trainer.patch_envelope_config_flags(
        envelope, {"readout_utf8_position_input": True}
    )
    assert patched == 3
    for label, node in (
        ("seed", envelope["config"]["taiji"]),
        ("substrate", envelope["substrate"]["config"]),
        ("kernel", envelope["taiji"]["kernel"]["config"]),
    ):
        assert node["readout_utf8_position_input"] is True, label
        changed = {key for key in before[label] if before[label][key] != node[key]}
        assert changed == {"readout_utf8_position_input"}, (label, changed)


def _hotstart_config() -> SeedConfig:
    """关掉身份器官：与热启动的真目标 `seed_beta` 同形（它不带身份器官载荷）。"""

    return SeedConfig(
        taiji=TaijiConfig(
            region_sizes=(16,),
            synapse_fan_in=4,
            motor_fan_in=8,
            identity_organ_enabled=False,
        )
    )


def _v1_envelope() -> dict:
    """`seed-native-v1` 旧档形状：没有 `taiji` 原生副本，只有两处 config（seed_beta 即此形）。"""

    envelope = Seed(_hotstart_config()).checkpoint()
    envelope.pop("taiji", None)
    return envelope


def test_patch_covers_v1_envelope_two_copies(trainer) -> None:
    envelope = _v1_envelope()
    assert trainer.patch_envelope_config_flags(
        envelope, {"readout_utf8_position_input": True}
    ) == 2
    assert envelope["config"]["taiji"]["readout_utf8_position_input"] is True
    assert envelope["substrate"]["config"]["readout_utf8_position_input"] is True


def test_hot_start_restore_needs_the_patch_and_then_succeeds(trainer) -> None:
    """补丁是解锁那次有意配方切换的**唯一**手段：不加 ⇒ 守卫拦下；加了 ⇒ 载入成功。"""

    target_config = trainer.apply_experiment_flags(
        _hotstart_config(), readout_position=True
    )

    # 不加补丁：档里的配置仍是关着的，与开启位置输入的架构不符 ⇒ 必须响亮失败。
    with pytest.raises(ValueError):
        Seed(target_config).restore(_v1_envelope())

    # 加了补丁：两处副本对齐 ⇒ 载入成功，且实际架构确实带着位置输入。
    envelope = _v1_envelope()
    trainer.patch_envelope_config_flags(envelope, {"readout_utf8_position_input": True})
    restored = Seed(target_config)
    restored.restore(envelope)
    assert restored.config.taiji.readout_utf8_position_input is True


def test_v10_envelope_with_identity_organ_fails_closed(trainer) -> None:
    """钉住已知边界：v10 档带身份器官时，改 config 会让血缘校验响亮失败（**故意**）。

    血缘把身份器官绑在它当时的核心上，改配方等于换核心；本流程不替它重发血缘。
    PLAN-A-26 的热启动目标是 `seed_beta`（v8/v1、不带身份器官），不受此限。
    """

    target_config = trainer.apply_experiment_flags(_small_config(), readout_position=True)
    envelope = _envelope()
    trainer.patch_envelope_config_flags(envelope, {"readout_utf8_position_input": True})
    with pytest.raises(ValueError, match="lineage"):
        Seed(target_config).restore(envelope)


# --- A-4 读出链：开关必须挂在**真正在学**的那条链上（2026-09-28 实测补钉） ---------


def _position_config() -> SeedConfig:
    return SeedConfig(
        taiji=TaijiConfig(
            region_sizes=(16,),
            synapse_fan_in=4,
            motor_fan_in=8,
            readout_utf8_position_input=True,
        )
    )


def test_predictive_readout_is_what_feeds_the_f1_position_columns() -> None:
    """机制钉：位置列只在 `readout="predictive"` 时被走到，且真被训到（非零）。"""

    model = Seed(_position_config())
    readout = model.substrate.predictive_readout
    for step in range(10):
        model.observe((65 + step) % 256, learn=True, readout="predictive", learn_motor=False)
    assert readout.position_probability_steps > 0
    assert readout.position_learn_steps > 0
    assert float(readout.position_weight.abs().sum()) > 0.0


def test_action_readout_leaves_the_f1_position_columns_at_zero() -> None:
    """反面钉：主训练线默认的 `action` 档**根本不走**F1 读出 ⇒ 位置列恒零。

    这正是 2026-09-28 那对"逐位相同的两臂"的成因：`--readout-position` 单开是**静默空转**。
    """

    model = Seed(_position_config())
    readout = model.substrate.predictive_readout
    for step in range(10):
        model.observe((65 + step) % 256, learn=True)  # 默认 readout="action"
    assert readout.position_probability_steps == 0
    assert readout.position_learn_steps == 0
    assert float(readout.position_weight.abs().sum()) == 0.0


def test_cli_refuses_position_flag_on_the_action_readout(trainer, monkeypatch) -> None:
    """响亮失败而不是静默空转：`--readout-position` 必须配 `--readout predictive`。"""

    monkeypatch.setattr(sys, "argv", ["train_seed_corpus.py", "--readout-position"])
    with pytest.raises(SystemExit):
        trainer.main()


def test_smoke_default_output_never_lands_on_the_product_checkpoint(trainer) -> None:
    """冒烟缺省必须落在 `output/`，**永不**落到产品件 `checkpoints/seed_corpus.pt` 上。

    来历：2026-09-28 一次 `--smoke`（只改预算、不改输出路径）把产品件覆盖成了 5000-tick
    的冒烟模型 —— 靠 `dist/Seed/_internal/checkpoints/` 的打包副本按 sha256 `c8025db44c65…`
    才复原。正式跑的缺省（＝产品件）一字不变，只有 `--smoke` 改道。
    """

    root = Path("/repo")
    smoke_checkpoint, smoke_progress = trainer.default_output_paths(smoke=True, project_root=root)
    assert smoke_checkpoint == root / "output" / "seed_corpus_smoke.pt"
    assert smoke_progress == root / "reports" / "seed_corpus_smoke_progress.jsonl"
    assert "checkpoints" not in smoke_checkpoint.parts, smoke_checkpoint
    assert "checkpoints" not in smoke_progress.parts, smoke_progress

    checkpoint, progress = trainer.default_output_paths(smoke=False, project_root=root)
    assert checkpoint == root / "checkpoints" / "seed_corpus.pt"
    assert progress == root / "reports" / "seed_corpus_progress.jsonl"