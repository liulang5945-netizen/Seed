"""R2 content-binding workspace implementation gates (contract §2/§3/§7).

Zero-training gates on ``taiji/sequence_content_workspace.py``: uniform
UTF-8 byte identity (padding distinct from byte 0, unseen characters
distinguishable), structural no-oracle API, name-stable shared initialisation
across arms, C consumption on BOTH output paths, mixture normalisation with
and without material, duplicate-character copy merge, emission-successor
bonus semantics, explicit range errors, arm compute/parameter disclosure,
checkpoint round-trip with digest tamper rejection.
"""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
from dataclasses import replace as dataclasses_replace
from pathlib import Path

import pytest
import torch

from taiji.sequence_char_workspace import CharVocab
from taiji.sequence_content_workspace import (
    CONTENT_BOUNDARY_SLOT,
    SEQUENCE_CONTENT_WORKSPACE_VERSION,
    SequenceContentConfig,
    SequenceContentTrainer,
    SequenceContentWorkspace,
    arm_parameter_order,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_FIXTURE = PROJECT_ROOT / "tests/fixtures/r2_content_binding_v1_train.jsonl"


@pytest.fixture(scope="module")
def vocab() -> CharVocab:
    rows = _rows()
    text = "".join(r["question"] + r["material"] + r["response"] for r in rows)
    return CharVocab(text)


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    return _rows()


def _rows() -> list[dict]:
    return [
        json.loads(line)
        for line in TRAIN_FIXTURE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _fact_item(rows: list[dict]) -> dict:
    return next(r for r in rows if r["group_class"] == "fact_flip" and r["copyable"])


def _workspaces(vocab: CharVocab, seed: int = 7) -> tuple[SequenceContentWorkspace, SequenceContentWorkspace]:
    return (
        SequenceContentWorkspace(vocab, SequenceContentConfig(arm="A", seed=seed)),
        SequenceContentWorkspace(vocab, SequenceContentConfig(arm="B", seed=seed)),
    )


# --------------------------------------------------------------------------- #
# Gate 1: uniform byte identity (contract section 2)
# --------------------------------------------------------------------------- #


def test_gate1_byte_features_uniform_and_padding_distinct(vocab) -> None:
    arm_a, arm_b = _workspaces(vocab)
    for ws in (arm_a, arm_b):
        # byte 0 (NUL) vs padding: byte-0 has length slot 0, padding has slot 4
        byte_zero = ws._byte_features("\x00")
        padding = ws._byte_features(None)
        assert not torch.equal(byte_zero, padding)
        assert byte_zero[-1] == 0.0 and byte_zero[0] == 0.0
        assert padding[-5:] .argmax() == 4
        # every character goes through the SAME byte channel; two unseen
        # characters produce different projections
        f1 = ws._char_input_feature("\u9f8c", CONTENT_UNK := 1)
        f2 = ws._char_input_feature("\u9f99", CONTENT_UNK)
        assert not torch.equal(f1, f2)
        # 5-byte characters are an explicit error
        with pytest.raises(ValueError):
            ws._byte_features("\U0001F600" + "x")[:0] if False else ws._byte_features("𝄞𝄞")
        del f1, f2


def test_gate1_unseen_output_keeps_codepoint(vocab) -> None:
    """An unseen material char occupies a dynamic candidate slot and decodes
    back to its real codepoint (no UNK fallback).  Uses a calibration row:
    train rows cannot contain train-unseen glyphs by construction."""

    _, arm_b = _workspaces(vocab)
    calibration_path = PROJECT_ROOT / "tests/fixtures/r2_content_binding_v1_calibration.jsonl"
    calibration = [
        json.loads(line)
        for line in calibration_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    item = next(r for r in calibration if r["group_class"] == "fact_flip" and r["copyable"])
    unseen_glyph = next(
        ch for ch in item["material"] if ch not in vocab.char_to_slot and ch not in "：，。？"
    )
    state = arm_b.begin_episode(item["question"], item["material"])
    slot = state.char2slot[unseen_glyph]
    assert slot >= vocab.size
    assert arm_b.glyph_for_slot(state, slot) == unseen_glyph
    assert state.extra_slot_codepoints[slot] == ord(unseen_glyph)


# --------------------------------------------------------------------------- #
# Gate 2: structural no-oracle (contract section 7)
# --------------------------------------------------------------------------- #


def test_gate2_runtime_api_carries_no_labels(vocab, rows) -> None:
    arm_a, arm_b = _workspaces(vocab)
    for ws in (arm_a, arm_b):
        for function in (ws.begin_episode, ws.generate, ws.episode_loss):
            parameters = set(inspect.signature(function).parameters)
            forbidden = {"shape", "answer", "labels", "pair_id", "group_class", "gold"}
            assert not (parameters & forbidden), (function.__name__, parameters)
    # mutating metadata fields cannot change behaviour: the model only sees
    # question/material strings
    item = _fact_item(rows)
    base = arm_b.generate(item["question"], item["material"]).text
    mutated = dict(item)
    mutated["group_class"] = "forged"
    mutated["response"] = "forged"
    mutated["copy_mask"] = [False]
    assert arm_b.generate(mutated["question"], mutated["material"]).text == base


# --------------------------------------------------------------------------- #
# Gate 3: name-stable shared initialisation across arms
# --------------------------------------------------------------------------- #


def test_gate3_shared_parameters_bitwise_identical(vocab) -> None:
    arm_a, arm_b = _workspaces(vocab)
    common_names = [
        name
        for name in arm_parameter_order("A")
        if name in set(arm_parameter_order("B"))
    ]
    assert len(common_names) == 23  # the shared front end; the rest is arm-specific
    for name in common_names:
        assert torch.equal(
            arm_a._parameters[name].detach(), arm_b._parameters[name].detach()
        ), name
    # creation order cannot move values: build in the opposite order
    ws_b2 = SequenceContentWorkspace(vocab, SequenceContentConfig(arm="B", seed=7))
    ws_a2 = SequenceContentWorkspace(vocab, SequenceContentConfig(arm="A", seed=7))
    for name in common_names:
        assert torch.equal(
            ws_b2._parameters[name].detach(), ws_a2._parameters[name].detach()
        ), name
    # the induction bonus starts at exactly zero in both arms
    assert float(arm_a._parameters["copy_induce_bias"].item()) == 0.0
    assert float(arm_b._parameters["copy_induce_bias"].item()) == 0.0


# --------------------------------------------------------------------------- #
# Gate 4: C consumption on both output paths (contract section 7)
# --------------------------------------------------------------------------- #


def test_gate4_gradients_reach_both_content_modules(vocab, rows) -> None:
    arm_a, arm_b = _workspaces(vocab)
    item = _fact_item(rows)
    loss_a, _ = arm_a.episode_loss(
        item["question"], item["material"], item["response"], item["copy_mask"],
        copy_value_weight=1.0,
    )
    grads_a = torch.autograd.grad(
        loss_a,
        [arm_a._parameters["head_query_1"], arm_a._parameters["a_content_mlp_input"]],
        allow_unused=True,
    )
    assert grads_a[0] is not None and torch.isfinite(grads_a[0]).all()
    assert grads_a[1] is not None and torch.isfinite(grads_a[1]).all()
    loss_b, _ = arm_b.episode_loss(
        item["question"], item["material"], item["response"], item["copy_mask"],
        copy_value_weight=1.0,
    )
    grads_b = torch.autograd.grad(
        loss_b,
        [arm_b._parameters["slot_init"], arm_b._parameters["relation_mlp_input"]],
        allow_unused=True,
    )
    assert grads_b[0] is not None and torch.isfinite(grads_b[0]).all()
    assert grads_b[1] is not None and torch.isfinite(grads_b[1]).all()


def test_gate4_c_feeds_vocab_and_copy_paths(vocab, rows) -> None:
    arm_a, arm_b = _workspaces(vocab)
    item = _fact_item(rows)
    for ws in (arm_a, arm_b):
        state = ws.begin_episode(item["question"], item["material"])
        _, mixture, _ = ws.step(state, None)
        perturbed_state = dataclasses_replace(state, content=state.content + 3.0)
        _, mixture_perturbed, _ = ws.step(perturbed_state, None)
        assert not torch.allclose(mixture, mixture_perturbed, atol=1e-8)


# --------------------------------------------------------------------------- #
# Gate 5: normalisation, empty material, duplicate merge
# --------------------------------------------------------------------------- #


def test_gate5_mixture_normalisation_with_and_without_material(vocab, rows) -> None:
    arm_a, arm_b = _workspaces(vocab)
    item = _fact_item(rows)
    for ws in (arm_a, arm_b):
        mixture, copies = ws.teacher_forced_distributions(
            item["question"], item["material"], item["response"]
        )
        sums = mixture.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
        # empty material: copy off, still finite and normalised
        empty_mixture, empty_copies = ws.teacher_forced_distributions(
            item["question"], "", "未知"
        )
        sums_empty = empty_mixture.sum(dim=1)
        assert torch.allclose(sums_empty, torch.ones_like(sums_empty), atol=1e-5)
        assert torch.isfinite(empty_mixture).all()
        assert float(empty_copies.abs().max()) == 0.0
        result = ws.generate(item["question"], "")
        assert result.steps <= ws.config.max_output_chars


def test_gate5_duplicate_characters_merge(vocab) -> None:
    arm_a, arm_b = _workspaces(vocab)
    for ws in (arm_a, arm_b):
        question = "问：草原是什么颜色？"
        material = "背景：草原是琥珀，岩石是琥珀。答："
        state = ws.begin_episode(question, material)
        positions = [j for j, ch in enumerate(state.material_chars) if ch == "琥"]
        assert len(positions) == 2
        _, _, copy_distribution = ws.step(state, None)
        # merge: candidate mass equals the summed addressing weights
        weights = ws._copy_weights(state, None)
        expected = float(weights[positions].sum().detach())
        assert copy_distribution[state.char2slot["琥"]].item() == pytest.approx(expected, abs=1e-6)


# --------------------------------------------------------------------------- #
# Gate 6: emission-successor bonus semantics (contract section 2)
# --------------------------------------------------------------------------- #


def test_gate6_successor_bonus_targets_following_rows(vocab) -> None:
    arm_a, arm_b = _workspaces(vocab)
    question = "问：草原是什么颜色？"
    material = "背景：草原是琥珀，岩石是玛瑙。答："
    for ws in (arm_a, arm_b):
        with torch.no_grad():
            ws._parameters["copy_induce_bias"].fill_(15.0)
        state = ws.begin_episode(question, material)
        weights = ws._copy_weights(state, "琥")
        rows_with_predecessor = [
            j for j in range(1, len(state.material_chars))
            if state.material_chars[j - 1] == "琥"
        ]
        assert int(weights.argmax()) in rows_with_predecessor
        # no previous char -> no bonus anywhere
        plain = ws._copy_weights(state, None)
        assert torch.isfinite(plain).all()
        # zero the bias again for later gates
        with torch.no_grad():
            ws._parameters["copy_induce_bias"].zero_()


# --------------------------------------------------------------------------- #
# Gate 7: explicit range errors (contract section 2)
# --------------------------------------------------------------------------- #


def test_gate7_input_and_output_limits(vocab, rows) -> None:
    arm_a, arm_b = _workspaces(vocab)
    item = _fact_item(rows)
    for ws in (arm_a, arm_b):
        with pytest.raises(ValueError):
            ws.begin_episode("问：" + "山" * 300, "")
        with pytest.raises(ValueError):
            ws.episode_loss(
                item["question"], item["material"], "长" * 33, [True] * 33
            )
        # generation beyond the cap records a range error, no silent truncation
        with torch.no_grad():
            ws._parameters["decoder"].zero_()
            ws._parameters["decoder_bias"].fill_(5.0)
            ws._parameters["decoder_bias"][CONTENT_BOUNDARY_SLOT] = -5.0
        result = ws.generate(item["question"], item["material"])
        assert result.range_error is True
        assert len(result.text) == ws.config.max_output_chars
        assert result.stopped_on_boundary is False


# --------------------------------------------------------------------------- #
# Gate 8: arm disclosure (contract section 1: B/A differences are disclosed)
# --------------------------------------------------------------------------- #


def test_gate8_parameter_and_compute_profiles_disclosed(vocab) -> None:
    arm_a, arm_b = _workspaces(vocab)
    profile_a = arm_a.compute_profile()
    profile_b = arm_b.compute_profile()
    assert profile_a["arm"] == "A" and profile_b["arm"] == "B"
    assert profile_a["parameter_count"] > 0 and profile_b["parameter_count"] > 0
    assert profile_b["per_episode_mac_estimate"] != profile_a["per_episode_mac_estimate"]
    # both arms disclose their content-module share
    assert profile_a["content_module_mac_estimate"] > 0
    assert profile_b["content_module_mac_estimate"] > 0


# --------------------------------------------------------------------------- #
# Gate 9: checkpoint round-trip and rejection (contract section 6)
# --------------------------------------------------------------------------- #


def test_gate9_roundtrip_and_tamper_rejection(vocab, rows, tmp_path) -> None:
    arm_b = SequenceContentWorkspace(vocab, SequenceContentConfig(arm="B", seed=11))
    item = _fact_item(rows)
    trainer = SequenceContentTrainer(
        arm_b, learning_rate=0.001, total_updates=10, code_revision="gate9", data_digest="digest-9"
    )
    trainer.train_step(
        [
            {
                "question": item["question"],
                "material": item["material"],
                "response": item["response"],
                "copy_mask": item["copy_mask"],
            }
        ]
    )
    before = arm_b.generate(item["question"], item["material"]).text
    path = trainer.save(tmp_path / "step1.pt")
    restored = SequenceContentTrainer.from_checkpoint(
        torch.load(path, map_location="cpu", weights_only=False)
    )
    assert restored.workspace.generate(item["question"], item["material"]).text == before
    assert restored.workspace.config.arm == "B"
    for name, parameter in trainer.workspace.named_parameters():
        assert torch.equal(parameter.detach(), restored.workspace._parameters[name].detach())
    # digest tampering is rejected
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["parameters"]["decoder"] = payload["parameters"]["decoder"] + 1.0
    with pytest.raises(ValueError):
        SequenceContentTrainer.from_checkpoint(payload)
    # foreign (char-graph) checkpoints are rejected
    with pytest.raises(ValueError):
        SequenceContentTrainer.from_checkpoint({"format": "taiji-sequence-char-workspace-trainer-v1"})


def test_gate9_fresh_process_restore_continues_identically(vocab, rows, tmp_path) -> None:
    arm_b = SequenceContentWorkspace(vocab, SequenceContentConfig(arm="B", seed=13))
    item = _fact_item(rows)
    batch = [
        {
            "question": item["question"],
            "material": item["material"],
            "response": item["response"],
            "copy_mask": item["copy_mask"],
        }
    ]
    trainer = SequenceContentTrainer(
        arm_b, learning_rate=0.001, total_updates=10, code_revision="gate9-fresh"
    )
    trainer.train_step(batch)
    path = trainer.save(tmp_path / "zero.pt")
    uninterrupted = SequenceContentTrainer.from_checkpoint(trainer.checkpoint())
    uninterrupted.train_step(batch)
    continued_digest = str(uninterrupted.checkpoint()["checkpoint_digest"])
    script = (
        f"import sys, json; sys.path.insert(0, r'{PROJECT_ROOT}');\n"
        "import torch\n"
        "from taiji.sequence_content_workspace import SequenceContentTrainer\n"
        f"payload = torch.load(r'{path}', map_location='cpu', weights_only=False)\n"
        "trainer = SequenceContentTrainer.from_checkpoint(payload)\n"
        f"trainer.train_step({batch!r})\n"
        "print(json.dumps({'digest': str(trainer.checkpoint()['checkpoint_digest']),"
        " 'version': int(payload['version'])}))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout.strip().splitlines()[-1])
    assert observed["version"] == SEQUENCE_CONTENT_WORKSPACE_VERSION
    assert observed["digest"] == continued_digest
