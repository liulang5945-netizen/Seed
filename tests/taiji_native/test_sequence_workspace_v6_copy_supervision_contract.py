"""R2-D4 H-T implementation gates: copy-component value supervision.

Contract:
plans/reference/M5_R2_D4_COPY_SUPERVISION_CONTRACT_FROZEN_20260918.md

Zero-capability gates on the unchanged v6 graph: bitwise identity of the
lambda=0 path (including replay of the frozen D3 probe checkpoint against its
recorded loss), mixture/copy structural decomposition, the value-mask
derivation rules over the full sealed fixture, gradient reach of the auxiliary
term (addressing chain yes, decoder no), and trainer integration with
checkpoint restoration.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from taiji.sequence_workspace import (
    SequenceWorkspaceConfig,
    SequenceWorkspacePrototype,
    SequenceWorkspaceTrainer,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = PROJECT_ROOT / "tests/fixtures/r2_d1_measurement_v1.jsonl"
D3_CHECKPOINT = PROJECT_ROOT / "reports/r2_d3_checkpoints/multihead_probe/hg_h4_seed20260917_epoch30.pt"
D3_REPORT = PROJECT_ROOT / "reports/r2_d3_multihead_probe_20260918.json"
D3_FINAL_LOSS = 0.00807605644927364

COPY_SUPPORTED_SHAPES = ("fact", "negation", "same_opening_fact")
NO_VALUE_SHAPES = ("unknown", "same_opening_unknown", "combination_same", "combination_different")

sys_path_scripts = str(PROJECT_ROOT)


def _prototype(seed: int = 20260917) -> SequenceWorkspacePrototype:
    return SequenceWorkspacePrototype(
        SequenceWorkspaceConfig(
            seed=seed,
            evidence_source="per_position",
            copy_mixture=True,
            question_conditioned_start=True,
            readout_heads=4,
            positional_keys=True,
        )
    )


def _prefix() -> bytes:
    return "问：雪是什么颜色？背景：雪是白。答：".encode("utf-8")


def _response() -> bytes:
    return "白".encode("utf-8")


def _rows() -> list[dict]:
    return [
        json.loads(line)
        for line in FIXTURE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# --------------------------------------------------------------------------- #
# Gate 1: lambda=0 bitwise identity (incl. frozen D3 checkpoint replay)
# --------------------------------------------------------------------------- #


def test_gate1_zero_weight_loss_identity() -> None:
    prototype = _prototype()
    with torch.no_grad():
        plain, metrics_plain = prototype.sequence_loss(_prefix(), _response())
        explicit, metrics_explicit = prototype.sequence_loss(
            _prefix(),
            _response(),
            copy_value_weight=0.0,
            value_mask=[True] * len(_response()),
        )
    assert float(plain) == float(explicit)
    for key in ("positions", "correct", "accuracy"):
        assert metrics_plain[key] == metrics_explicit[key]
    assert metrics_plain["copy_value_prob_mean"] == 0.0
    # And the loss equals the hand-computed teacher-forced mixture NLL.
    distributions = prototype.teacher_forced_distributions(_prefix(), _response())
    targets = torch.tensor(
        [int(symbol) for symbol in _response()] + [int(prototype.config.boundary_symbol)],
        dtype=torch.long,
    )
    per_position = -distributions.gather(1, targets.unsqueeze(1)).squeeze(1).clamp_min(1e-12).log()
    assert torch.allclose(plain, per_position.mean(), rtol=1e-6, atol=1e-7)


def test_gate1_frozen_d3_checkpoint_replays_recorded_loss() -> None:
    """The lambda=0 path must reproduce the D3 probe's recorded final loss."""

    payload = torch.load(D3_CHECKPOINT, map_location="cpu", weights_only=False)
    trainer = SequenceWorkspaceTrainer.from_checkpoint(payload)
    prototype = trainer.prototype
    rows = [row for row in _rows() if row["split"] == "train"]
    total = 0.0
    with torch.no_grad():
        for row in rows:
            loss, _ = prototype.sequence_loss(
                row["prefix"].encode("utf-8"), row["response"].encode("utf-8")
            )
            total += float(loss)
    replayed = total / len(rows)
    assert abs(replayed - D3_FINAL_LOSS) < 1e-9


# --------------------------------------------------------------------------- #
# Gate 2: mixture/copy structural decomposition
# --------------------------------------------------------------------------- #


def test_gate2_mixture_is_gated_vocab_plus_copy() -> None:
    prototype = _prototype()
    prefix, response = _prefix(), _response()
    mixtures, copies = prototype.teacher_forced_mixture_and_copy(prefix, response)
    assert mixtures.shape == copies.shape
    boundary = int(prototype.config.boundary_symbol)
    # The copy component never assigns mass to the boundary symbol.
    assert torch.all(copies[:, boundary] == 0.0)
    # Each copy row is a normalized distribution over visible prefix bytes.
    assert torch.allclose(copies.sum(dim=1), torch.ones(copies.shape[0]), rtol=1e-6, atol=1e-6)
    # Rebuild the mixture by hand from the same components: gate * vocab + (1-gate) * copy.
    state = prototype.begin_episode(prefix)
    previous = int(prototype.config.boundary_symbol)
    for index, symbol in enumerate(response):
        new_state, vocab_logits = prototype.step(state, previous)
        p_vocab = torch.softmax(vocab_logits, dim=0)
        weights = prototype.addressing_weights(new_state, detach=False)
        p_copy = prototype._copy_distribution(new_state, weights)
        gate = torch.sigmoid(
            new_state.renderer_state @ prototype._parameters["copy_gate_weight"]
            + prototype._parameters["copy_gate_bias"]
        )
        assert torch.allclose(mixtures[index], gate * p_vocab + (1.0 - gate) * p_copy)
        assert torch.allclose(copies[index], p_copy)
        state = new_state
        previous = int(symbol)


# --------------------------------------------------------------------------- #
# Gate 3: value-mask derivation rules over the sealed fixture
# --------------------------------------------------------------------------- #


def test_gate3_value_masks_over_full_fixture() -> None:
    from scripts.training.probe_taiji_r2_d4_copy_supervision import value_mask_for

    rows = _rows()
    supervised = 0
    for row in rows:
        mask = value_mask_for(row["shape"], row["response"])
        encoded = row["response"].encode("utf-8")
        assert len(mask) == len(encoded)
        if row["shape"] in ("fact", "same_opening_fact"):
            assert all(mask) and len(mask) > 0
        elif row["shape"] == "negation":
            lead = "不是".encode("utf-8")
            assert encoded[: len(lead)] == lead
            assert mask[: len(lead)] == (False,) * len(lead)
            assert any(mask[len(lead) :])
        else:
            assert row["shape"] in NO_VALUE_SHAPES
            assert not any(mask)
        if row["split"] == "train" and any(mask):
            supervised += 1
    # Exactly the copy-supported train episodes carry supervised value positions.
    assert supervised == 144


def test_gate3_value_mask_rejects_malformed_negation() -> None:
    from scripts.training.probe_taiji_r2_d4_copy_supervision import value_mask_for

    with pytest.raises(ValueError):
        value_mask_for("negation", "白色")


# --------------------------------------------------------------------------- #
# Gate 4: gradient reach of the auxiliary term
# --------------------------------------------------------------------------- #


def test_gate4_auxiliary_gradient_reaches_addressing_not_decoder() -> None:
    prototype = _prototype()
    mask = [True] * len(_response())
    total = prototype.sequence_loss(_prefix(), _response(), copy_value_weight=1.0, value_mask=mask)[0]
    (decoder_grad, head_grad) = torch.autograd.grad(
        total,
        [prototype._parameters["decoder"], prototype._parameters["head_query_1"]],
        allow_unused=True,
    )
    answer_only = prototype.sequence_loss(_prefix(), _response())[0]
    (answer_decoder_grad, answer_head_grad) = torch.autograd.grad(
        answer_only,
        [prototype._parameters["decoder"], prototype._parameters["head_query_1"]],
        allow_unused=True,
    )
    # Decoder gradients come solely from the answer term: the auxiliary copy
    # NLL never touches the vocab branch.
    assert torch.equal(decoder_grad - answer_decoder_grad, torch.zeros_like(decoder_grad))
    # The addressing chain must receive auxiliary gradient.
    assert not torch.equal(head_grad - answer_head_grad, torch.zeros_like(head_grad))


# --------------------------------------------------------------------------- #
# Gate 5: trainer integration, alignment errors, checkpoint restoration
# --------------------------------------------------------------------------- #


def test_gate5_trainer_requires_mask_when_enabled() -> None:
    trainer = SequenceWorkspaceTrainer(_prototype(), learning_rate=0.01, code_revision="test")
    trainer.enable_copy_value_supervision(1.0)
    with pytest.raises(ValueError):
        trainer.train_step([(_prefix(), _response())])


def test_gate5_trainer_rejects_misaligned_masks() -> None:
    trainer = SequenceWorkspaceTrainer(_prototype(), learning_rate=0.01, code_revision="test")
    trainer.enable_copy_value_supervision(1.0)
    with pytest.raises(ValueError):
        trainer.train_step(
            [(_prefix(), _response())],
            value_masks=[[True, False]],
        )


def test_gate5_training_reduces_copy_loss_and_restores() -> None:
    torch.manual_seed(20260917)
    trainer = SequenceWorkspaceTrainer(_prototype(), learning_rate=0.05, code_revision="test")
    trainer.enable_copy_value_supervision(1.0)
    batch = [(_prefix(), _response()), ("问：草是什么颜色？背景：草是绿。答：".encode(), "绿".encode())]
    masks = [[True] * 3, [True] * 3]
    first = trainer.train_step(batch, value_masks=masks)
    assert first["loss"] == first["loss"]  # finite
    assert first["copy_value_prob_mean"] > 0.0
    for _ in range(5):
        record = trainer.train_step(batch, value_masks=masks)
    assert record["loss"] < first["loss"]
    # Checkpoint round-trip: parameters identical, training-time flag re-enabled.
    restored = SequenceWorkspaceTrainer.from_checkpoint(trainer.checkpoint())
    assert restored.copy_value_weight == 0.0  # training-time setting, not state
    restored.enable_copy_value_supervision(1.0)
    with torch.no_grad():
        baseline = trainer.prototype.sequence_loss(_prefix(), _response())[0]
        after = restored.prototype.sequence_loss(_prefix(), _response())[0]
    assert float(baseline) == float(after)
