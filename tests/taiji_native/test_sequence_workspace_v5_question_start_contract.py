"""R2-D2 H-A3 implementation gates for graph v5 (question-conditioned start).

Contract:
plans/reference/M5_R2_D2_QUESTION_START_AMENDMENT_FROZEN_20260918.md
section 2.  Zero capability training: inventory, causal isolation of the
question start (it varies with the stem and is invariant to the material),
the single-channel proof that same-stem material changes act only through the
read, marker splitting for all three splits, gradient reach and versioned
restoration.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from taiji.internalization import content_digest
from taiji.sequence_workspace import (
    EVIDENCE_BROADCAST_FINAL,
    EVIDENCE_PER_POSITION,
    MATERIAL_MARKERS,
    SEQUENCE_WORKSPACE_VERSION,
    SequenceWorkspaceConfig,
    SequenceWorkspacePrototype,
    SequenceWorkspaceTrainer,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _arm(source: str = EVIDENCE_PER_POSITION, seed: int = 20260917) -> SequenceWorkspacePrototype:
    return SequenceWorkspacePrototype(
        SequenceWorkspaceConfig(
            seed=seed,
            evidence_source=source,
            copy_mixture=True,
            question_conditioned_start=True,
        )
    )


def _state(prototype: SequenceWorkspacePrototype, prefix: str) -> torch.Tensor:
    return prototype.begin_episode(prefix.encode("utf-8")).renderer_state


# --------------------------------------------------------------------------- #
# Gate 1: inventory and combinations
# --------------------------------------------------------------------------- #


def test_gate1_question_start_inventory_and_shared_init() -> None:
    a5 = _arm(EVIDENCE_PER_POSITION)
    c5 = _arm(EVIDENCE_BROADCAST_FINAL)
    assert a5.parameter_count() == 86_818
    assert c5.parameter_count() == 86_818
    names = tuple(name for name, _ in a5.named_parameters())
    assert "answer_start_weight" in names and "answer_start_bias" in names
    for name in names:
        torch.testing.assert_close(
            a5.named_parameter(name), c5.named_parameter(name), rtol=0, atol=0
        )
    v4 = SequenceWorkspacePrototype(
        SequenceWorkspaceConfig(
            seed=20260917, evidence_source=EVIDENCE_PER_POSITION, copy_mixture=True
        )
    )
    assert v4.parameter_count() == 82_658
    for invalid in (
        dict(workspace_enabled=False, question_conditioned_start=True),
        dict(evidence_source="final_state_slots", question_conditioned_start=True),
    ):
        try:
            SequenceWorkspaceConfig(**invalid)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"invalid start combination accepted: {invalid}")


# --------------------------------------------------------------------------- #
# Gate 2: causal isolation of the question start
# --------------------------------------------------------------------------- #


def test_gate2_start_varies_with_stem_and_is_material_invariant() -> None:
    prototype = _arm()
    white_fact = "提问：雪的颜色？线索：雪是白。回答："
    cyan_fact = "提问：雪的颜色？线索：雪是青。回答："
    other_question = "提问：湖水的颜色？线索：湖水是青。回答："
    same_stem_white = _state(prototype, white_fact)
    same_stem_cyan = _state(prototype, cyan_fact)
    other_stem = _state(prototype, other_question)
    torch.testing.assert_close(same_stem_white, same_stem_cyan, rtol=0, atol=0)
    assert not torch.allclose(same_stem_white, other_stem)


def test_gate2_material_acts_only_through_the_read_at_first_step() -> None:
    prototype = _arm()
    white = "提问：雪的颜色？线索：雪是白。回答：".encode()
    cyan = "提问：雪的颜色？线索：雪是青。回答：".encode()
    # with the evidence pathways removed, first-step distributions must be
    # bit-identical across same-stem materials
    state_w = prototype.begin_episode(white)
    state_c = prototype.begin_episode(cyan)
    _s, probs_w = prototype.step_distribution(state_w, 256, zero_read=True)
    _s, probs_c = prototype.step_distribution(state_c, 256, zero_read=True)
    torch.testing.assert_close(probs_w, probs_c, rtol=0, atol=0)
    # while the intact graph is material-sensitive at the same step
    _s, full_w = prototype.step_distribution(prototype.begin_episode(white), 256)
    _s, full_c = prototype.step_distribution(prototype.begin_episode(cyan), 256)
    assert not torch.allclose(full_w, full_c, atol=1e-7)


def test_gate2_marker_split_for_every_split_and_synthetic_fallback() -> None:
    prototype = _arm()
    cases = (
        "问：天空是什么颜色？背景：天空是蓝。答：",
        "提问：雪是否为乳白？线索：雪不是乳白。回答：",
        "查询：夜晚属于黑吗。已知：夜晚不是黑。输出：",
    )
    # states must be question-only: equal to a prefix truncated at the marker
    for rendered in cases:
        encoded = rendered.encode("utf-8")
        state = prototype.begin_episode(encoded).renderer_state
        marker = next(m for m in MATERIAL_MARKERS if encoded.find(m) >= 0)
        stem = encoded[: encoded.find(marker)]
        assert len(stem) > 0
        # recompute the projected stem state directly from scan states
        _h0, states = prototype._scan_prefix(stem)
        expected = torch.tanh(
            states[-1] @ prototype.named_parameter("answer_start_weight")
            + prototype.named_parameter("answer_start_bias")
        )
        torch.testing.assert_close(state, expected, rtol=0, atol=0)
    # prefix without any marker falls back to the whole-prefix state
    fallback = prototype.begin_episode(b"synthetic prefix no marker").renderer_state
    _h0, states = prototype._scan_prefix(b"synthetic prefix no marker")
    fallback_expected = torch.tanh(
        states[-1] @ prototype.named_parameter("answer_start_weight")
        + prototype.named_parameter("answer_start_bias")
    )
    torch.testing.assert_close(fallback, fallback_expected, rtol=0, atol=0)


# --------------------------------------------------------------------------- #
# Gate 3: gradients
# --------------------------------------------------------------------------- #


def test_gate3_answer_start_receives_finite_nonzero_gradient() -> None:
    prototype = _arm()
    prototype.zero_grads_for_test()
    prefix = "提问：雪是否为白？线索：雪不是白。回答：".encode()
    response = "不是白".encode()
    loss, _ = prototype.sequence_loss(prefix, response)
    loss.backward()
    for name in ("answer_start_weight", "answer_start_bias", "evidence_key", "copy_gate_weight"):
        gradient = prototype.named_parameter(name).grad
        assert gradient is not None and bool(torch.isfinite(gradient).all()), name
        assert float(gradient.abs().sum()) > 0.0, name


# --------------------------------------------------------------------------- #
# Gate 4: restoration v5/v4
# --------------------------------------------------------------------------- #


def _logits_digest(prototype: SequenceWorkspacePrototype) -> str:
    prefix = "提问：雪的颜色？线索：雪是白。回答：".encode()
    response = "白".encode()
    return content_digest(prototype.teacher_forced_logits(prefix, response).detach())


def test_gate4_v5_roundtrip_and_fresh_process(tmp_path: Path) -> None:
    prototype = _arm()
    batch = (("提问：雪的颜色？线索：雪是白。回答：".encode(), "白".encode()),)
    trainer = SequenceWorkspaceTrainer(prototype, learning_rate=0.01, code_revision="d2v5")
    trainer.set_episodes(batch)
    expected = _logits_digest(prototype)
    path = trainer.save(tmp_path / "v5.pt")
    restored = SequenceWorkspaceTrainer.from_checkpoint(trainer.checkpoint())
    assert restored.prototype.config.question_conditioned_start is True
    assert _logits_digest(restored.prototype) == expected
    uninterrupted = SequenceWorkspaceTrainer.from_checkpoint(trainer.checkpoint())
    uninterrupted.train_step(batch)
    continued = str(uninterrupted.checkpoint()["checkpoint_digest"])
    restored.train_step(batch)
    assert str(restored.checkpoint()["checkpoint_digest"]) == continued

    prefix, response = batch[0]
    script = (
        f"import sys, json; sys.path.insert(0, r'{str(PROJECT_ROOT)}');"
        "import torch;"
        "from taiji.internalization import content_digest;"
        "from taiji.sequence_workspace import SequenceWorkspaceTrainer;"
        f"payload = torch.load(r'{str(path)}', map_location='cpu', weights_only=False);"
        "trainer = SequenceWorkspaceTrainer.from_checkpoint(payload);"
        f"logits = trainer.prototype.teacher_forced_logits({prefix!r}, {response!r});"
        f"trainer.train_step({list(batch)!r});"
        "print(json.dumps({'logits': content_digest(logits.detach()),"
        " 'continued': str(trainer.checkpoint()['checkpoint_digest']),"
        " 'version': int(trainer.checkpoint()['version']),"
        " 'qstart': bool(trainer.prototype.config.question_conditioned_start)}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout.strip().splitlines()[-1])
    assert observed["logits"] == expected
    assert observed["continued"] == continued
    assert observed["version"] == SEQUENCE_WORKSPACE_VERSION
    assert observed["qstart"] is True


def test_gate4_v4_payload_loads_without_question_start() -> None:
    v4_trainer = SequenceWorkspaceTrainer(
        SequenceWorkspacePrototype(
            SequenceWorkspaceConfig(
                seed=20260917,
                evidence_source=EVIDENCE_PER_POSITION,
                copy_mixture=True,
            )
        ),
        code_revision="v4graph",
    )
    v4_trainer.set_episodes(
        (("提问：雪的颜色？线索：雪是白。回答：".encode(), "白".encode()),)
    )
    historical = dict(v4_trainer.checkpoint())
    historical["version"] = 4
    historical["checkpoint_digest"] = content_digest(
        {key: value for key, value in historical.items() if key != "checkpoint_digest"}
    )
    restored = SequenceWorkspaceTrainer.from_checkpoint(historical)
    assert restored.prototype.config.question_conditioned_start is False
    assert _logits_digest(restored.prototype) == _logits_digest(v4_trainer.prototype)
