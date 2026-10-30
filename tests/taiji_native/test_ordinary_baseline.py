"""R3b-2 contracts for the ordinary end-to-end baseline."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import torch

from taiji.conditional_language_v3 import TaskCoreConfigV3
from taiji.ordinary_baseline import OrdinaryBundleV1, build_prompt

ROOT = Path(__file__).resolve().parents[2]
QUERY = "雨伞的颜色是什么？".encode()


def _config() -> TaskCoreConfigV3:
    return TaskCoreConfigV3(
        width=64,
        embedding_width=32,
        key_width=32,
        slots=8,
        chunk_bytes=8,
        max_observation_bytes=512,
        max_input_evidence_bytes=512,
        max_output_position=128,
        max_generation_bytes=64,
        seed=31,
        position_extension_seed=32,
    )


def _bundle(history: tuple[str, ...]) -> OrdinaryBundleV1:
    bundle = OrdinaryBundleV1.create(_config())
    for text in history:
        bundle.observe(text)
    return bundle


def test_prompt_contains_raw_history_and_time_without_a_hard_gate():
    prompt = build_prompt(["雨伞的颜色是红色。", "未解析的一段话"], QUERY, "previous")
    assert b"<|history|>" in prompt
    assert "雨伞的颜色是红色。".encode() in prompt
    assert "未解析的一段话".encode() in prompt
    assert b"<|query|>" in prompt
    assert b"<|time|>previous" in prompt
    assert b"<|answer|>" in prompt


def test_every_parameter_is_trainable():
    bundle = OrdinaryBundleV1.create(_config())
    names = {name for name, _ in bundle.model.named_parameters()}
    assert set(bundle.model.trainable_parameters()) == names


def test_first_target_byte_is_supervised_and_prefix_advances():
    bundle = _bundle(("雨伞的颜色是红色。",))
    left = bundle.score_target(query=QUERY, target=b"ab")
    right = bundle.score_target(query=QUERY, target=b"cb")
    assert left.shape == (2, 257)
    assert torch.allclose(left[0], right[0])
    assert not torch.allclose(left[1], right[1])


def test_removing_the_relevant_history_changes_output_and_rescue_restores():
    history = ("雨伞的颜色是红色。", "雨伞的颜色是蓝色。")
    bundle = _bundle(history)
    baseline = bundle.score_target(query=QUERY, target=b"ab")
    saved = bundle.history
    bundle.history = (history[0],)
    cut = bundle.score_target(query=QUERY, target=b"ab")
    assert not torch.equal(baseline, cut)
    bundle.history = saved
    assert torch.equal(baseline, bundle.score_target(query=QUERY, target=b"ab"))


def test_same_query_with_different_current_history_changes_output():
    left = _bundle(("雨伞的颜色是红色。", "雨伞的颜色是蓝色。")).score_target(
        query=QUERY, target=b"ab"
    )
    right = _bundle(("雨伞的颜色是蓝色。", "雨伞的颜色是绿色。")).score_target(
        query=QUERY, target=b"ab"
    )
    assert not torch.allclose(left, right)


def test_same_current_value_with_different_history_changes_output():
    left = _bundle(("雨伞的颜色是红色。", "雨伞的颜色是蓝色。")).score_target(
        query=QUERY, target=b"ab"
    )
    right = _bundle(("雨伞的颜色是绿色。", "雨伞的颜色是蓝色。")).score_target(
        query=QUERY, target=b"ab"
    )
    assert not torch.allclose(left, right)


def test_zero_step_bundle_restores_in_process_and_fresh_process(tmp_path):
    bundle = _bundle(("雨伞的颜色是红色。", "雨伞的颜色是蓝色。"))
    expected = bundle.probe_digest(query=QUERY)
    path = tmp_path / "ordinary.pt"
    bundle.save(path)
    restored = OrdinaryBundleV1.load(path)
    assert restored.probe_digest(query=QUERY) == expected
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    code = (
        "import sys; "
        "from taiji.ordinary_baseline import OrdinaryBundleV1; "
        "b=OrdinaryBundleV1.load(sys.argv[1]); "
        "print(b.probe_digest(query=sys.argv[2].encode()))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(path), QUERY.decode()],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == expected
