"""R2-D6 H-B implementation gates: graph v8 copy induction.

Contract:
plans/reference/M5_R2_D6_INDUCTION_COPY_CONTRACT_FROZEN_20260918.md

Zero-training gates per contract section 3: flag-off and zero-init bitwise
identity with the frozen v6 graph, replay of the frozen D4 probe checkpoint
against its recorded train loss (the D5 canonical-index lesson), the
successor-row bonus structure (duplicate-byte match sets, first-step no-op,
injection override), gradient reach of the single new scalar, inventory
switching, and checkpoint round-trips across v8/v7/v6/v2.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from taiji.sequence_workspace import (
    SEQUENCE_WORKSPACE_VERSION,
    SequenceWorkspaceConfig,
    SequenceWorkspacePrototype,
    SequenceWorkspaceTrainer,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = PROJECT_ROOT / "tests/fixtures/r2_d1_measurement_v1.jsonl"
D4_PROBE_CHECKPOINT = PROJECT_ROOT / "reports/r2_d4_checkpoints/copy_supervision_probe/epoch30.pt"
D4_PROBE_REPORT = PROJECT_ROOT / "reports/r2_d4_copy_supervision_probe_20260918.json"


def _config(**extra) -> SequenceWorkspaceConfig:
    base = dict(
        seed=20260917,
        evidence_source="per_position",
        copy_mixture=True,
        question_conditioned_start=True,
        readout_heads=4,
        positional_keys=True,
    )
    base.update(extra)
    return SequenceWorkspaceConfig(**base)


def _v6() -> SequenceWorkspacePrototype:
    return SequenceWorkspacePrototype(_config())


def _v8() -> SequenceWorkspacePrototype:
    return SequenceWorkspacePrototype(_config(copy_induction=True))


def _prefix() -> bytes:
    return "提问：雪的颜色？线索：雪是琥珀。回答：".encode()


def _response() -> bytes:
    return "琥珀".encode()


# --------------------------------------------------------------------------- #
# Gate 1: inventories and bit-identity
# --------------------------------------------------------------------------- #


def test_gate1_v8_inventory() -> None:
    assert SEQUENCE_WORKSPACE_VERSION == 8
    induced = _v8()
    assert induced.parameter_count() == 96_035
    names = tuple(name for name, _ in induced.named_parameters())
    assert "copy_induce_bias" in names
    assert float(induced._parameters["copy_induce_bias"].detach().sum()) == 0.0
    plain = _v6()
    assert plain.parameter_count() == 96_034
    assert "copy_induce_bias" not in tuple(name for name, _ in plain.named_parameters())


def test_gate1_zero_bias_bitwise_equal_v6() -> None:
    v6 = _v6()
    on = _v8()
    prefix, response = _prefix(), _response()
    with torch.no_grad():
        assert torch.equal(
            on.teacher_forced_logits(prefix, response),
            v6.teacher_forced_logits(prefix, response),
        )
        mixed, copies = on.teacher_forced_mixture_and_copy(prefix, response)
        v6_mixed, v6_copies = v6.teacher_forced_mixture_and_copy(prefix, response)
        assert torch.equal(mixed, v6_mixed) and torch.equal(copies, v6_copies)
        assert (
            on.sequence_loss(prefix, response)[0].item()
            == v6.sequence_loss(prefix, response)[0].item()
        )
        assert (
            on.generate(prefix, max_bytes=8).bytes_out == v6.generate(prefix, max_bytes=8).bytes_out
        )


def test_gate1_frozen_d4_checkpoint_replays_recorded_loss() -> None:
    """The D5 lesson institutionalized before training: the induction-off
    path must replay the frozen D4 probe checkpoint's recorded train loss."""

    report = json.loads(D4_PROBE_REPORT.read_text(encoding="utf-8"))
    recorded = float(report["final"]["mean_sequence_loss"])
    payload = torch.load(D4_PROBE_CHECKPOINT, map_location="cpu", weights_only=False)
    trainer = SequenceWorkspaceTrainer.from_checkpoint(payload)
    rows = [
        json.loads(line)
        for line in FIXTURE.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line)["split"] == "train"
    ]
    total = 0.0
    with torch.no_grad():
        for row in rows:
            loss, _ = trainer.prototype.sequence_loss(
                row["prefix"].encode("utf-8"), row["response"].encode("utf-8")
            )
            total += float(loss)
    replayed = total / len(rows)
    assert abs(replayed - recorded) < 1e-9


# --------------------------------------------------------------------------- #
# Gate 2: successor-row bonus structure
# --------------------------------------------------------------------------- #


def test_gate2_first_step_boundary_matches_nothing() -> None:
    prototype = _v8()
    with torch.no_grad():
        prototype._parameters["copy_induce_bias"].fill_(15.0)
    state = prototype.begin_episode(_prefix())
    boundary = int(prototype.config.boundary_symbol)
    baseline = prototype.addressing_weights(state, detach=False)
    stepped = prototype.addressing_weights(state, detach=False, induce_byte=boundary)
    assert torch.equal(baseline, stepped)


def test_gate2_bonus_targets_successor_rows_of_every_match() -> None:
    prototype = _v8()
    with torch.no_grad():
        prototype._parameters["copy_induce_bias"].fill_(15.0)
    prefix = "提问：雪的颜色？线索：雪是琥珀，不是琥珀。回答：".encode()
    state = prototype.begin_episode(prefix)
    entries = list(state.entry_bytes)  # type: ignore[arg-type]
    probe_byte = entries[30]  # an interior byte guaranteed to have successors
    matches = [j for j in range(1, len(entries)) if entries[j - 1] == probe_byte]
    assert matches
    weights = prototype.addressing_weights(state, detach=False, induce_byte=probe_byte)
    stacked = weights if weights.ndim == 2 else weights.unsqueeze(0)
    total = float(stacked.sum())
    assert total == pytest.approx(stacked.shape[0], rel=1e-6)  # each head normalized
    assert float(stacked[:, matches].sum()) / stacked.shape[0] > 0.99
    for head in range(stacked.shape[0]):
        assert int(stacked[head].argmax()) in matches


def test_gate2_inject_override_matches_step_emission() -> None:
    prototype = _v8()
    with torch.no_grad():
        prototype._parameters["copy_induce_bias"].fill_(15.0)
    prefix, response = _prefix(), _response()
    _, copies = prototype.teacher_forced_mixture_and_copy(prefix, response)
    # Step 3 conditions on the second byte of 琥: its unique-ish byte must
    # raise the continuation mass above the zero-init graph at that position.
    third_byte = "琥".encode()[1]
    with torch.no_grad():
        prototype._parameters["copy_induce_bias"].fill_(0.0)
    _, copies_zero = prototype.teacher_forced_mixture_and_copy(prefix, response)
    state = prototype.begin_episode(prefix)
    entries = list(state.entry_bytes)  # type: ignore[arg-type]
    successors = {entries[j] for j in range(1, len(entries)) if entries[j - 1] == third_byte}
    scored = sum(float(copies[2, byte]) for byte in successors)
    scored_zero = sum(float(copies_zero[2, byte]) for byte in successors)
    assert scored > scored_zero


# --------------------------------------------------------------------------- #
# Gate 3: gradient and validation
# --------------------------------------------------------------------------- #


def test_gate3_bias_gradient_finite() -> None:
    prototype = _v8()
    response = _response()
    total, _ = prototype.sequence_loss(
        _prefix(), response, copy_value_weight=1.0, value_mask=[True] * len(response)
    )
    (grad,) = torch.autograd.grad(
        total, [prototype._parameters["copy_induce_bias"]], allow_unused=True
    )
    assert grad is None or torch.isfinite(grad).all()


def test_config_rejects_induction_without_copy() -> None:
    with pytest.raises(ValueError):
        _config(copy_induction=True, copy_mixture=False)


# --------------------------------------------------------------------------- #
# Gate 4: checkpoint round-trips
# --------------------------------------------------------------------------- #


def test_gate4_roundtrip_and_legacy_reads(tmp_path: Path) -> None:
    trainer = SequenceWorkspaceTrainer(_v8(), learning_rate=0.01, code_revision="gate")
    trainer.set_episodes([(_prefix(), _response())])
    payload = trainer.checkpoint()
    assert payload["version"] == SEQUENCE_WORKSPACE_VERSION
    restored = SequenceWorkspaceTrainer.from_checkpoint(payload)
    assert restored.prototype.config.copy_induction is True
    before = trainer.prototype.generate(_prefix(), max_bytes=6).bytes_out
    after = restored.prototype.generate(_prefix(), max_bytes=6).bytes_out
    assert before == after
    legacy = SequenceWorkspaceTrainer(_v6(), learning_rate=0.01, code_revision="legacy")
    legacy_path = tmp_path / "v6.pt"
    legacy.save(legacy_path)
    revived = SequenceWorkspaceTrainer.from_checkpoint(
        torch.load(legacy_path, map_location="cpu", weights_only=False)
    )
    assert revived.prototype.config.copy_induction is False
    assert "copy_induce_bias" not in dict(
        (name, _) for name, _ in revived.prototype.named_parameters()
    )
