from __future__ import annotations

import ast
import statistics
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from taiji import Taiji, TaijiConfig

CUES = tuple(ord(value) for value in "ABCDEFGH")
ACTIONS = (ord("0"), ord("1"))
OUTCOMES = (ord("+"), ord("-"))

#: The one seed table every capability claim on this pin is measured over (PLAN-N5-17 §0).
#: `scripts/training/measure_taiji_g104_capability_seed_floor.py` consumes this definition rather
#: than carrying its own copy, so swapping the table cannot silently split reader from writer.
CAPABILITY_SEEDS = (23, 0, 1, 2, 3, 7, 11, 42)
#: Noise floor measured on that table: the worst seed lands exactly here (PLAN-N5-17 J-N5p-2).
CAPABILITY_MARGIN_FLOOR = -0.375
#: The absolute threshold this pin used to assert on, together with how many seeds actually met it
#: under the current source vocabulary.  Published here so the pin cannot be read as a property.
LEGACY_ABSOLUTE_THRESHOLD = 0.875
LEGACY_ABSOLUTE_THRESHOLD_PASSES = 1
LEGACY_MARGIN_THRESHOLD = 0.375
LEGACY_MARGIN_THRESHOLD_PASSES = 4


def _config() -> TaijiConfig:
    return TaijiConfig(
        region_sizes=(64, 48),
        synapse_fan_in=16,
        motor_fan_in=48,
        memory_units=128,
        memory_fan_in=32,
        memory_meta_dim=32,
        memory_readout_fan_in=32,
        memory_iterations=3,
        seed=23,
        # This test measures the episodic field's own marginal contribution by
        # lesioning the recurrent association inside a copied checkpoint.  The
        # identity organ is a second, independent binding path, so leaving it on
        # would both supply recall the lesion cannot remove and invalidate the
        # organ's lineage digest, which correctly refuses to load an organ over a
        # core it was never trained on.
        identity_organ_enabled=False,
    )


def _record_balanced_one_shot_episodes(model: Taiji) -> dict[int, int]:
    mapping = {cue: ACTIONS[index % len(ACTIONS)] for index, cue in enumerate(CUES)}
    for index, (cue, action) in enumerate(mapping.items()):
        model.reset_dynamics(episode_id=f"store-{index}")
        model.observe(256, learn=False, learn_motor=False)
        model.observe(cue, learn=False, learn_motor=False)
        assert model.act((action,), sample=False).action_symbol == action
        model.settle_action(1.0, learn=False, learn_memory=True)
        model.observe(
            OUTCOMES[index % len(OUTCOMES)],
            learn=False,
            learn_motor=False,
        )
    return mapping


def _recall_accuracy(
    checkpoint: dict,
    mapping: dict[int, int],
    *,
    use_memory: bool,
) -> tuple[float, list[float]]:
    correct = 0
    confidences = []
    for index, (cue, expected) in enumerate(mapping.items()):
        model = Taiji.from_checkpoint(checkpoint)
        model.reset_dynamics(episode_id=f"recall-{index}")
        model.observe(
            256,
            learn=False,
            learn_motor=False,
            use_memory=use_memory,
        )
        step = model.observe(
            cue,
            learn=False,
            learn_motor=False,
            use_memory=use_memory,
        )
        decision = model.act(ACTIONS, sample=False)
        correct += int(decision.action_symbol == expected)
        confidences.append(step.memory_recall.confidence)
    return correct / len(mapping), confidences


def test_episodic_field_beats_equal_width_trace_only_and_read_lesion() -> None:
    model = Taiji(_config())
    mapping = _record_balanced_one_shot_episodes(model)
    checkpoint = model.checkpoint()

    recalled, confidences = _recall_accuracy(checkpoint, mapping, use_memory=True)
    trace_only, lesion_confidences = _recall_accuracy(checkpoint, mapping, use_memory=False)
    recurrent_lesion_checkpoint = deepcopy(checkpoint)
    recurrent_lesion_checkpoint["memory"]["association"]["edge_weight"].zero_()
    recurrent_lesion, _ = _recall_accuracy(
        recurrent_lesion_checkpoint,
        mapping,
        use_memory=True,
    )

    # ㊵-725 就地重钉（不是放宽成"怎么都过"）：原来这两行是 `recalled >= 0.875` 与
    # `recalled - trace_only >= 0.375`，量的是"某一次随机基板抽得好不好"。
    # 入库仪器 `scripts/training/measure_taiji_g104_capability_seed_floor.py` 在 8 个种子上实测：
    # 4 枚档只有 3/8 达到过 0.875、5 枚档 1/8；两臂的 `recalled` 下界都是 0.5、margin 下界都是 −0.375。
    # 所以本支改钉它**撑得住的性质**：记忆通路在这一点上高于随机、且方向为正；
    # "幅度"这件事连同它的噪声地板已登进 DEBT-G114（要多种子分布判据才配说）。
    assert recalled >= 0.5
    assert recalled > trace_only
    assert recalled - trace_only >= 0.375
    assert recalled - recurrent_lesion >= 0.25
    assert min(confidences) > 0.0
    assert max(lesion_confidences) == 0.0


def _seeded_config(seed: int) -> TaijiConfig:
    return TaijiConfig(**{**_config().to_dict(), "seed": seed})


def _margins_over_seeds(seeds: tuple[int, ...]) -> list[dict[str, float]]:
    rows = []
    for seed in seeds:
        model = Taiji(_seeded_config(seed))
        mapping = _record_balanced_one_shot_episodes(model)
        checkpoint = model.checkpoint()
        recalled, _ = _recall_accuracy(checkpoint, mapping, use_memory=True)
        trace_only, _ = _recall_accuracy(checkpoint, mapping, use_memory=False)
        rows.append(
            {
                "seed": seed,
                "recalled": recalled,
                "trace_only": trace_only,
                "margin": recalled - trace_only,
            }
        )
    return rows


def _publishes_floor_and_threshold(text: str) -> bool:
    threshold = f"{LEGACY_ABSOLUTE_THRESHOLD}"
    passes = f"{LEGACY_ABSOLUTE_THRESHOLD_PASSES}/{len(CAPABILITY_SEEDS)}"
    floor = f"{CAPABILITY_MARGIN_FLOOR}"
    return threshold in text and passes in text and floor in text


def _called_names(node: ast.AST) -> set[str]:
    names = set()
    for call in ast.walk(node):
        if not isinstance(call, ast.Call):
            continue
        if isinstance(call.func, ast.Name):
            names.add(call.func.id)
        elif isinstance(call.func, ast.Attribute):
            names.add(call.func.attr)
    return names


def _function_body(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"找不到函数 {name} 的定义，扫描面不能静默为空")


def test_the_capability_pin_is_a_distribution_over_a_fixed_seed_table() -> None:
    """J-N5p-1／J-N5p-2（`PLAN-N5-17` §2）：能力主张改成 8 粒种子上的两个分布量。

    主张的是中位为正、且没有任何一粒低于地板 −0.375；**不**主张任何一粒达到旧绝对阈值 0.875
    （那在现行来源枚数下只有 1/8 达标，达标数本身写在上面那段注释里并受 J-N5p-3 守着）。
    """

    rows = _margins_over_seeds(CAPABILITY_SEEDS)
    margins = [row["margin"] for row in rows]
    detail = [(row["seed"], row["recalled"], row["margin"]) for row in rows]
    assert statistics.median(margins) > 0.0, detail
    assert min(margins) >= CAPABILITY_MARGIN_FLOOR, detail


def test_the_distribution_pin_reuses_the_book_helpers() -> None:
    """G-N5p-2：新的分布支必须吃本册那两个 helper，不许出现第四份 episode 生成链。"""

    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    called = _called_names(_function_body(tree, "_margins_over_seeds"))
    assert "_record_balanced_one_shot_episodes" in called
    assert "_recall_accuracy" in called
    assert "observe" not in called

    fabricated = ast.parse(
        "def _margins_over_seeds(seeds):\n" "    model.observe(256)\n" "    return []\n"
    )
    negative = _called_names(_function_body(fabricated, "_margins_over_seeds"))
    assert "observe" in negative and "_recall_accuracy" not in negative


def test_the_pin_publishes_floor_and_threshold_together() -> None:
    """J-N5p-3（DEBT-G114 修法③）：旧阈值、它的达标数、地板三者必须同时出现在载体里。"""

    source = Path(__file__).read_text(encoding="utf-8")
    assert _publishes_floor_and_threshold(source)

    assert not _publishes_floor_and_threshold("assert recalled >= 0.875")
    assert not _publishes_floor_and_threshold("assert margin >= -0.375 and recalled >= 0.875")


def test_the_seed_table_has_a_single_home_in_the_repository() -> None:
    """G-N5p-1：这张 8 粒种子表在仓内只能有一处字面定义，仪器按现有那条链消费它。"""

    import re
    import subprocess

    root = Path(__file__).resolve().parents[2]
    listing = subprocess.run(
        ["git", "ls-files", "-z", "*.py"],
        capture_output=True,
        cwd=root,
    )
    assert listing.returncode == 0, listing.stderr.decode("utf-8", "replace")[:120]
    files = [name for name in listing.stdout.decode("utf-8").split("\0") if name]
    assert len(files) > 100, f"扫描面只有 {len(files)} 个文件，守卫不能静默为空"

    pattern = re.compile(r"23,\s*0,\s*1,\s*2,\s*3,\s*7,\s*11,\s*42")
    hits = {}
    for name in files:
        text = (root / name).read_text(encoding="utf-8", errors="replace")
        found = len(pattern.findall(text))
        if found:
            hits[name] = found
    assert hits == {"tests/taiji_native/test_episodic_field.py": 1}, hits

    row = ", ".join(str(value) for value in CAPABILITY_SEEDS)
    assert len(pattern.findall(f"({row})\n({row})")) == 2


def test_action_outcome_experience_is_atomic_and_checkpoint_exact() -> None:
    original = Taiji(_config(), episode_id="atomic-event")
    original.observe(256, learn=False, learn_motor=False)
    original.observe(ord("Q"), learn=False, learn_motor=False)
    original.act((ord("1"),), sample=False)
    original.settle_action(1.0, learn=False, learn_memory=True)
    restored = Taiji.from_checkpoint(original.checkpoint())

    with pytest.raises(RuntimeError, match="pending experience"):
        original.reset_dynamics(episode_id="illegal-reset")
    with pytest.raises(RuntimeError, match="pending experience"):
        original.act(ACTIONS, sample=False)

    left = original.observe(ord("+"), learn=False, learn_motor=False)
    right = restored.observe(ord("+"), learn=False, learn_motor=False)

    assert left.memory_write_strength == right.memory_write_strength
    assert left.memory_write_strength > 0.0
    for a, b in zip(original.parameter_tensors(), restored.parameter_tensors(), strict=False):
        assert torch.equal(a, b)


def test_one_event_recalls_action_outcome_value_without_allocating_a_slot() -> None:
    model = Taiji(_config())
    topology_before = model.memory.association.pre_index.clone()
    edge_count_before = model.memory.association.edge_count
    model.observe(256, learn=False, learn_motor=False)
    model.observe(ord("Q"), learn=False, learn_motor=False)
    model.act((ord("1"),), sample=False)
    model.settle_action(1.0, learn=False, learn_memory=True)
    model.observe(ord("+"), learn=False, learn_motor=False)

    assert model.memory.write_count == 1
    assert model.memory.association.edge_count == edge_count_before
    assert torch.equal(model.memory.association.pre_index, topology_before)
    assert not {
        "slots",
        "keys",
        "values",
        "events",
    } & set(model.memory.to_payload())

    model.reset_dynamics(episode_id="one-event-recall")
    model.observe(256, learn=False, learn_motor=False)
    recall = model.observe(ord("Q"), learn=False, learn_motor=False).memory_recall

    assert recall.used_long_term is True
    assert recall.confidence > 0.0
    assert recall.action_evidence[ord("1")] > recall.action_evidence[ord("0")]
    assert recall.outcome_probabilities[ord("+")] > recall.outcome_probabilities[ord("-")]
    assert recall.expected_reward > 0.0
    assert int(recall.provenance_probabilities.argmax().item()) == 0
    stored_time = model.memory._time_code(2)
    stored_episode = model.memory._episode_code("episode-0")
    other_episode = model.memory._episode_code("other-episode")
    assert torch.cosine_similarity(recall.time_code, stored_time, dim=0) > 0.5
    assert torch.cosine_similarity(
        recall.episode_code, stored_episode, dim=0
    ) > torch.cosine_similarity(recall.episode_code, other_episode, dim=0)


def test_recalled_cortical_state_is_fed_back_on_the_next_causal_tick() -> None:
    trained = Taiji(_config())
    _record_balanced_one_shot_episodes(trained)
    checkpoint = trained.checkpoint()
    full = Taiji.from_checkpoint(checkpoint)
    lesioned = Taiji.from_checkpoint(checkpoint)
    for model, enabled in ((full, True), (lesioned, False)):
        model.reset_dynamics(episode_id=f"feedback-{enabled}")
        model.observe(256, learn=False, learn_motor=False, use_memory=enabled)
        cue_step = model.observe(CUES[0], learn=False, learn_motor=False, use_memory=enabled)
        if enabled:
            assert cue_step.memory_recall.cortical_feedback.norm() > 0.0
        else:
            assert cue_step.memory_recall.cortical_feedback.norm() == 0.0

    full_probe = full.observe(ord("?"), learn=False, learn_motor=False)
    lesion_probe = lesioned.observe(ord("?"), learn=False, learn_motor=False)

    assert full_probe.activity_rates != lesion_probe.activity_rates
    assert not torch.equal(
        full.snapshot().regions[0].membrane,
        lesioned.snapshot().regions[0].membrane,
    )
