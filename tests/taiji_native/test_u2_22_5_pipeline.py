"""Smoke & contract tests for the unified 22.5.0 -> 22.5.1 pipeline."""

from __future__ import annotations

import torch

from scripts.training.first_a_support_22_5 import generate_examples
from taiji.u2_22_5_pipeline import (
    ATrainConfig,
    BOrganTrainConfig,
    build_b_items,
    cosine_warmup_lr,
    evaluate_22_5_1_suite,
    train_record_organ_on_parent,
    train_task_a,
)


def test_cosine_warmup_lr_schedule_bounds():
    assert cosine_warmup_lr(0, 100, 1e-2, 1e-4, 10) == 1e-3
    assert cosine_warmup_lr(9, 100, 1e-2, 1e-4, 10) == 1e-2
    end_lr = cosine_warmup_lr(100, 100, 1e-2, 1e-4, 10)
    assert abs(end_lr - 1e-4) < 1e-9


def test_unified_pipeline_smoke_freezes_parent_and_recovers_bit_exact():
    torch.set_num_threads(1)
    examples = generate_examples("train", 8)
    b_items = build_b_items(examples)
    assert len(b_items) == 6  # 2 of 8 are "未知"

    parent, hist = train_task_a(
        examples,
        ATrainConfig(seed=20261110, d=8, updates=2, batch_size=2, warmup_steps=1),
        fit_examples=examples[:4],
        eval_every=2,
    )
    assert len(hist) == 1

    child, info = train_record_organ_on_parent(
        parent,
        examples,
        BOrganTrainConfig(seed=20261110, steps=2, batch_size=2),
    )
    assert info["parent_bit_identical"] is True

    suite = evaluate_22_5_1_suite(child, parent, examples)
    assert suite["a_retention_drop"] == 0.0
    assert suite["recovery_exact"] is True
