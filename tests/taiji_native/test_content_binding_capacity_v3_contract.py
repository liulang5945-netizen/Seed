"""R2 content-binding v3 implementation gates: capacity extension.

Contract: plans/reference/M5_R2_CONTENT_BINDING_CONTRACT_V3_FROZEN_20260919.md §E1/§E2/§E3

Zero-training gates on the single v3 design variable (question-side capacity:
question_hidden_width 64 -> 128, relation_hidden 96 -> 128): parameter shapes
move with the config, the margin diagnostic reproduces v2-era values on a v2
checkpoint (instrument continuity), train-only margins are finite for every
class, the runner exposes the capacity knobs with an isolated smoke run, and
pre-v3 configs/checkpoints keep loading unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts.training.train_taiji_r2_content_binding import (
    GROUP_CLASSES,
    load_train_fixture,
    run,
    train_pair_margin_diagnostic,
)
from taiji.sequence_char_workspace import CharVocab
from taiji.sequence_content_workspace import (
    SequenceContentConfig,
    SequenceContentTrainer,
    SequenceContentWorkspace,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def vocab() -> CharVocab:
    rows = _rows()
    return CharVocab("".join(r["question"] + r["material"] + r["response"] for r in rows))


def _rows() -> list[dict]:
    return [
        json.loads(line)
        for line in (PROJECT_ROOT / "tests/fixtures/r2_content_binding_v1_train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]


# --------------------------------------------------------------------------- #
# Gate 1: capacity moves the question-side shapes (contract section E1)
# --------------------------------------------------------------------------- #


def test_gate1_v3_shapes_and_counts(vocab) -> None:
    v2 = SequenceContentWorkspace(vocab, SequenceContentConfig(arm="B", seed=3))
    v3 = SequenceContentWorkspace(
        vocab,
        SequenceContentConfig(arm="B", seed=3, question_hidden_width=128, relation_hidden=128),
    )
    assert v3._parameters["question_recur"].shape == (128, 128)
    assert v3._parameters["question_input"].shape == (48, 128)
    assert v3._parameters["start_weight"].shape == (128, 64)
    assert v3._parameters["relation_mlp_input"].shape == (96 + 128, 128)
    assert v3._parameters["slot_pool_query"].shape == (128, 48)
    # material/renderer side untouched
    assert v3._parameters["material_recur"].shape == (64, 64)
    assert v3._parameters["renderer_recur"].shape == (64, 64)
    assert v2._parameters["question_recur"].shape == (64, 64)
    assert v3.parameter_count() > v2.parameter_count()
    profile = v3.compute_profile()
    assert profile["parameter_count"] == v3.parameter_count()
    # forward pass works at the widened shape
    rows = _rows()
    item = next(r for r in rows if r["group_class"] == "fact_flip")
    loss, metrics = v3.episode_loss(
        item["question"], item["material"], item["response"], item["copy_mask"],
        copy_value_weight=1.0,
    )
    assert torch.isfinite(loss)


def test_gate1_backward_compat_defaults(vocab) -> None:
    """A pre-v3 config payload without the new field resolves to width 64."""

    legacy_payload = {"arm": "B", "seed": 3}
    config = SequenceContentConfig(**legacy_payload)
    assert config.question_hidden_width == 64
    workspace = SequenceContentWorkspace(vocab, config)
    assert workspace._parameters["question_recur"].shape == (64, 64)


# --------------------------------------------------------------------------- #
# Gate 2: margin diagnostic instrument continuity (contract section E3)
# --------------------------------------------------------------------------- #


def test_gate2_margin_diagnostic_on_v2_checkpoint() -> None:
    """Reproduces the v2 terminal hinge structure (fact/missing ~0,
    object/relation/negation near softplus(gamma)) from the committed v2
    checkpoint, proving the diagnostic measures the same quantity."""

    checkpoint = (
        PROJECT_ROOT / "reports/r2_content_binding_v2/B/20260920/cal_lr1/update_002000.pt"
    )
    if not checkpoint.is_file():
        pytest.skip("v2 checkpoint not present on this machine")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    trainer = SequenceContentTrainer.from_checkpoint(payload)
    diagnostic = train_pair_margin_diagnostic(trainer.workspace, load_train_fixture())
    assert set(diagnostic) == set(GROUP_CLASSES)
    softplus_gamma = float(torch.nn.functional.softplus(torch.tensor(1.0)))
    # margin satisfied on material-internal contrasts
    assert diagnostic["fact_flip"]["mean_hinge_a"] < 0.01
    assert diagnostic["missing_to_filled"]["mean_hinge_b"] < 0.01
    # structural constants on equal-answer classes
    assert diagnostic["distractor_invariant"]["mean_hinge_a"] == pytest.approx(softplus_gamma, abs=1e-5)
    # the failing classes sit near the untouched margin
    assert diagnostic["object_swap"]["mean_hinge_a"] == pytest.approx(softplus_gamma, abs=0.05)
    for cls in GROUP_CLASSES:
        assert diagnostic[cls]["groups"] > 0


def test_gate2_margins_finite_on_v3_workspace(vocab) -> None:
    v3 = SequenceContentWorkspace(
        vocab,
        SequenceContentConfig(arm="B", seed=5, question_hidden_width=128, relation_hidden=128),
    )
    diagnostic = train_pair_margin_diagnostic(
        v3, load_train_fixture(), per_class_limit=4
    )
    for cls in GROUP_CLASSES:
        for key in ("mean_hinge_a", "mean_hinge_b"):
            value = diagnostic[cls][key]
            assert value is not None and value >= 0.0 and value < 50.0


# --------------------------------------------------------------------------- #
# Gate 3: runner capacity knobs with isolated smoke run (contract section E3)
# --------------------------------------------------------------------------- #


def test_gate3_v3_smoke_run(tmp_path) -> None:
    report = run(
        arm="B",
        config_name="cal_lr1",
        seed=20260920,
        total_updates=2,
        output_root=tmp_path,
        mode="smoke",
        wall_cap_seconds=600,
        trainer_revision="v2",
        question_hidden_width=128,
        relation_hidden=128,
    )
    assert report["status"] == "completed"
    assert report["capacity"]["question_hidden_width"] == 128
    assert report["capacity"]["relation_hidden"] == 128
    assert report["capacity"]["parameter_count"] > 108963  # v2 B-arm count
    assert report["pair_contrastive_weight"] == 1.0
