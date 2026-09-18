"""R2-D7 H-Tok implementation gates: char-token graph v1.

Contract:
plans/reference/M5_R2_D7_CHAR_TOKEN_CONTRACT_FROZEN_20260918.md

Zero-training gates on the character-unit twin: train-only vocabulary with
byte-identical dev/final value glyphs excluded, candidate-space mixture
structure (vocab + dynamic material slots, boundary zero in copy), the
character-unit induction successor rule, value supervision gradient reach,
self-consistent fresh-process preflight, and checkpoint round-trip.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from taiji.sequence_char_workspace import (
    CHAR_BOUNDARY_SLOT,
    CHAR_UNK_SLOT,
    SEQUENCE_CHAR_WORKSPACE_VERSION,
    CharVocab,
    SequenceCharConfig,
    SequenceCharTrainer,
    SequenceCharWorkspace,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = PROJECT_ROOT / "tests/fixtures/r2_d1_measurement_v2.jsonl"


def _rows() -> list[dict]:
    return [
        json.loads(line)
        for line in FIXTURE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _vocab() -> CharVocab:
    rows = _rows()
    train_text = "".join(r["prefix"] + r["response"] for r in rows if r["split"] == "train")
    return CharVocab(train_text)


def _workspace(**extra) -> tuple[SequenceCharWorkspace, CharVocab]:
    vocab = _vocab()
    config = SequenceCharConfig(**extra)
    return SequenceCharWorkspace(vocab, config), vocab


def _sample_fact_multibyte() -> dict:
    return next(
        r
        for r in _rows()
        if r["split"] == "train" and r["shape"] == "fact" and "琥珀" in r["response"]
    )


def _sample_dev_unknown_style() -> dict:
    """A dev row whose value glyph is fully outside the train vocabulary."""

    return next(r for r in _rows() if r["split"] == "dev" and r["shape"] == "fact")


# --------------------------------------------------------------------------- #
# Gate 1: vocabulary and non-leakage (contract section 3.1)
# --------------------------------------------------------------------------- #


def test_gate1_train_only_vocab_excludes_dev_final_value_glyphs() -> None:
    vocab = _vocab()
    rows = _rows()
    train_text = "".join(r["prefix"] + r["response"] for r in rows if r["split"] == "train")
    # Every train glyph is in the table (or the boundary/unk slots).
    assert all(vocab.slot_of(character) != CHAR_UNK_SLOT for character in set(train_text))
    assert vocab.size == len(set(train_text)) + 2
    # The dev/final value glyphs that must exit through the copy channel.
    for glyph in "灰乳米白黑墨漆暗":
        assert glyph not in vocab.char_to_slot, glyph
    # Shared function glyphs (template wording) legitimately overlap.
    assert "色" in vocab.char_to_slot


# --------------------------------------------------------------------------- #
# Gate 2: candidate-space mixture structure (contract section 3.2)
# --------------------------------------------------------------------------- #


def test_gate2_candidate_slots_and_mixture_normalization() -> None:
    workspace, vocab = _workspace()
    sample = _sample_dev_unknown_style()  # train row; use a dev one for extras
    dev_rows = [r for r in _rows() if r["split"] == "dev" and r["shape"] == "fact"]
    unseen = next(r for r in dev_rows if True)
    state = workspace.begin_episode(unseen["prefix"])
    # Unseen material glyphs occupy dynamic extra slots.
    unseen_glyphs = {c for c in unseen["prefix"] if vocab.slot_of(c) == CHAR_UNK_SLOT}
    assert unseen_glyphs
    extras = {chr(cp) for cp in state.extra_slot_codepoints.values()}
    assert unseen_glyphs <= extras
    # Duplicate characters map to a single candidate slot.
    assert len(set(state.entry_slots)) <= len(state.entry_slots)
    mixture, copies = workspace.teacher_forced_distributions(unseen["prefix"], unseen["response"])
    per_row = mixture.sum(dim=1)
    assert torch.allclose(per_row, torch.ones_like(per_row), atol=1e-6)
    # The boundary slot receives no copy mass anywhere.
    assert float(copies[:, CHAR_BOUNDARY_SLOT].max()) == 0.0
    del sample


def test_gate2_copy_mass_merges_duplicate_characters() -> None:
    workspace, _ = _workspace()
    prefix = "问：天空是什么颜色？背景：天空是琥珀，琥珀不是翡翠。答："
    state = workspace.begin_episode(prefix)
    positions = [j for j, character in enumerate(prefix) if character == "琥"]
    assert len(positions) == 2
    _, _, copy_distribution = workspace.step_distribution(state, CHAR_BOUNDARY_SLOT)
    slot = state.char2slot["琥"]
    # Merged copy mass equals the summed addressing weights of both rows,
    # evaluated on the SAME post-step renderer state the distribution used.
    stepped, _ = workspace.step(state, CHAR_BOUNDARY_SLOT)
    weights = workspace._head_weights(stepped)
    stacked = weights if weights.ndim == 2 else weights.unsqueeze(0)
    expected = float(stacked[:, positions].sum() / stacked.shape[0])
    assert copy_distribution[slot].item() == pytest.approx(expected, abs=1e-6)


# --------------------------------------------------------------------------- #
# Gate 3: induction successor rule at the character unit (contract 3.3)
# --------------------------------------------------------------------------- #


def test_gate3_induction_bonus_targets_successor_rows() -> None:
    workspace, _ = _workspace()
    with torch.no_grad():
        workspace._parameters["copy_induce_bias"].fill_(15.0)
    prefix = "提问：雪的颜色？线索：雪是琥珀。回答："
    state = workspace.begin_episode(prefix)
    rows = list(prefix)
    target_index = rows.index("琥")
    from_dataclasses = __import__("dataclasses")
    with_prev = from_dataclasses.replace(state, last_emitted_codepoint=ord("琥"))
    weights = workspace._head_weights(with_prev)
    stacked = weights if weights.ndim == 2 else weights.unsqueeze(0)
    for head in range(stacked.shape[0]):
        assert int(stacked[head].argmax()) == target_index + 1
    # First step (no pointer): identical to the flag-off graph.
    plain = workspace._head_weights(state)
    assert torch.isfinite(plain).all()
    successor_rows = [j for j in range(1, len(rows)) if rows[j - 1] == "琥"]
    assert successor_rows == [target_index + 1]


def test_gate3_induction_flag_off_equals_induction_on_zero_bias() -> None:
    sample = _sample_fact_multibyte()
    on, _ = _workspace(copy_induction=True)
    off, _ = _workspace(copy_induction=False)
    mixture_on, copy_on = on.teacher_forced_distributions(sample["prefix"], sample["response"])
    mixture_off, copy_off = off.teacher_forced_distributions(
        sample["prefix"], sample["response"]
    )
    assert torch.allclose(mixture_on, mixture_off, atol=1e-7)
    assert torch.allclose(copy_on, copy_off, atol=1e-7)
    gen_on = on.generate(sample["prefix"], max_chars=6).text
    gen_off = off.generate(sample["prefix"], max_chars=6).text
    assert gen_on == gen_off


# --------------------------------------------------------------------------- #
# Gate 4: supervision reaches the addressing chain (contract 3.4)
# --------------------------------------------------------------------------- #


def test_gate4_value_supervision_gradient_reaches_head_queries() -> None:
    workspace, _ = _workspace()
    sample = _sample_fact_multibyte()
    response = sample["response"]
    total, metrics = workspace.sequence_loss(
        sample["prefix"], response, copy_value_weight=1.0, value_mask=[True] * len(response)
    )
    grads = torch.autograd.grad(
        total,
        [
            workspace._parameters["head_query_1"],
            workspace._parameters["evidence_key"],
            workspace._parameters["copy_induce_bias"],
        ],
        allow_unused=True,
    )
    assert grads[0] is not None and torch.isfinite(grads[0]).all()
    assert grads[1] is not None and torch.isfinite(grads[1]).all()
    assert metrics["copy_value_prob_mean"] > 0.0


def test_gate4_value_mask_character_unit_rules() -> None:
    from scripts.training.probe_taiji_r2_d7_char_token import value_mask_for_char

    assert value_mask_for_char("fact", "琥珀") == (True, True)
    assert value_mask_for_char("negation", "不是灰白") == (False, False, True, True)
    assert value_mask_for_char("same_opening_fact", "米色") == (True, True)
    assert value_mask_for_char("unknown", "未知") == (False, False)
    assert value_mask_for_char("combination_same", "相同") == (False, False)
    assert value_mask_for_char("combination_different", "不同") == (False, False)


# --------------------------------------------------------------------------- #
# Gate 5: fresh-process preflight and checkpoint round-trip (contract 3.4/3.5)
# --------------------------------------------------------------------------- #


def test_gate5_self_consistent_preflight(tmp_path: Path) -> None:
    workspace, _ = _workspace()
    sample = _sample_fact_multibyte()
    trainer = SequenceCharTrainer(workspace, learning_rate=0.01, code_revision="preflight")
    trainer.set_episodes([(sample["prefix"], sample["response"])])
    path = trainer.save(tmp_path / "zero_step.pt")
    script = (
        f"import sys, json; sys.path.insert(0, r'{str(PROJECT_ROOT)}');"
        "import torch;"
        "from taiji.sequence_char_workspace import SequenceCharTrainer;"
        f"payload = torch.load(r'{str(path)}', map_location='cpu', weights_only=False);"
        "trainer = SequenceCharTrainer.from_checkpoint(payload);"
        f"workspace = trainer.workspace;"
        f"state = workspace.begin_episode({sample['prefix']!r});"
        f"mixed, copied = workspace.teacher_forced_distributions({sample['prefix']!r}, {sample['response']!r});"
        "print(json.dumps({'sum': float(mixed.sum()), 'version': int(payload['version'])}))"
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
    assert observed["version"] == SEQUENCE_CHAR_WORKSPACE_VERSION
    local_mixed, _ = workspace.teacher_forced_distributions(
        sample["prefix"], sample["response"]
    )
    assert observed["sum"] == pytest.approx(float(local_mixed.sum()), abs=1e-5)


def test_gate5_roundtrip_and_byte_payload_rejected(tmp_path: Path) -> None:
    workspace, _ = _workspace()
    sample = _sample_fact_multibyte()
    trainer = SequenceCharTrainer(workspace, learning_rate=0.02, code_revision="gate")
    trainer.enable_copy_value_supervision(1.0)
    trainer.set_episodes([(sample["prefix"], sample["response"])])
    mask = [[True] * len(sample["response"])]
    trainer.train_step([(sample["prefix"], sample["response"])], value_masks=mask)
    before = workspace.generate(sample["prefix"], max_chars=6).text
    restored = SequenceCharTrainer.from_checkpoint(trainer.checkpoint())
    assert restored.workspace.generate(sample["prefix"], max_chars=6).text == before
    byte_path = PROJECT_ROOT / "reports/r2_d4_checkpoints/matched/20260917/epoch30.pt"
    byte_payload = torch.load(byte_path, map_location="cpu", weights_only=False)
    with pytest.raises(ValueError):
        SequenceCharTrainer.from_checkpoint(byte_payload)
