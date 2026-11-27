"""Tests for Taiji Native Non-Transformer Cognitive Backbone & Autonomous Shadow Growth."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import torch

from taiji.native_cognitive_backbone import (
    NATIVE_PRESETS,
    NativeNeuralGroupBlock,
    TaijiNativeCognitiveBackbone,
)
from taiji.native_growth_controller import NativeGrowthController, NativeGrowthPolicy


def test_native_backbone_is_strictly_non_transformer() -> None:
    src = Path("taiji/native_cognitive_backbone.py").read_text(encoding="utf-8")
    assert "class GQAAttention" not in src
    assert "class TransformerBlock" not in src
    assert "F.scaled_dot_product_attention" not in src


def test_native_presets_and_causality() -> None:
    small = TaijiNativeCognitiveBackbone(NATIVE_PRESETS["native_small_8m"])
    mid = TaijiNativeCognitiveBackbone(NATIVE_PRESETS["native_mid_18m"])
    assert 4_000_000 <= small.count_parameters() <= 11_000_000
    assert 12_000_000 <= mid.count_parameters() <= 22_000_000

    torch.manual_seed(7)
    tiny = TaijiNativeCognitiveBackbone(NATIVE_PRESETS["native_tiny_2m"])
    tiny.eval()
    x = torch.randint(1, 8192, (2, 16))
    x_mod = x.clone()
    x_mod[:, -1] = (x_mod[:, -1] + 1) % 8191 + 1
    with torch.no_grad():
        logits_a, _, _ = tiny(x)
        logits_b, _, _ = tiny(x_mod)
    assert torch.allclose(logits_a[:, :-1, :], logits_b[:, :-1, :], atol=1e-5)


def test_proximal_erase_freeze_and_neutral_growth_and_capsule() -> None:
    torch.manual_seed(11)
    model = TaijiNativeCognitiveBackbone(NATIVE_PRESETS["native_tiny_2m"])
    model.eval()
    x = torch.randint(1, 8192, (2, 12))

    # p_allowed=0.0 must leave fast weights bit-identical
    st0 = model.init_state(2)
    with torch.no_grad():
        _, _, st_frozen = model(x, state=st0, p_allowed=0.0)
    for f0, f1 in zip(st0.fast_weights, st_frozen.fast_weights, strict=True):
        assert torch.equal(f0, f1)

    # Neutral growth primitives must preserve forward output (< 1e-5)
    with torch.no_grad():
        logits0, _, _ = model(x)
        model.grow_group_units_neutral_(layer_idx=0, delta_units=16, seed=1)
        logits1, _, _ = model(x)
        model.grow_fast_heads_neutral_(layer_idx=0, delta_heads=1, seed=2)
        logits2, _, _ = model(x)
        model.grow_depth_neutral_(seed=3)
        logits3, _, _ = model(x)

    assert float((logits1 - logits0).abs().max().item()) < 1e-5
    assert float((logits2 - logits1).abs().max().item()) < 1e-5
    assert float((logits3 - logits2).abs().max().item()) < 1e-5
    assert len(model.growth_lineage) == 3

    # Selective plasticity grad mask zeroes mature units & w_slow
    model.mark_all_mature_()
    model.grow_group_units_neutral_(layer_idx=0, delta_units=8, seed=4)
    model.train()
    y = torch.randint(1, 8192, (2, 12))
    _, loss, _ = model(x, labels=y)
    assert loss is not None
    loss.backward()
    masked = model.apply_selective_plasticity_grad_mask(freeze_mature_slow=True)
    assert masked["masked_units"] > 0
    blk0 = cast(NativeNeuralGroupBlock, model.blocks[0])
    assert blk0.w_slow.weight.grad is not None
    assert float(blk0.w_slow.weight.grad.abs().max().item()) == 0.0

    # Capsule roundtrip after structural growth
    model.eval()
    cap = model.export_capsule("native_zh")
    restored = TaijiNativeCognitiveBackbone.from_capsule(cap, "native_zh")
    restored.eval()
    with torch.no_grad():
        la, _, _ = model(x)
        lb, _, _ = restored(x)
    assert torch.equal(la, lb)


def test_autonomous_shadow_growth_controller_accepts_and_rejects() -> None:
    torch.manual_seed(19)
    model = TaijiNativeCognitiveBackbone(NATIVE_PRESETS["native_tiny_2m"])
    ctrl = NativeGrowthController(
        NativeGrowthPolicy(auto_grow_enabled=True, min_gain=0.01, max_retention_drop=0.02),
        error_threshold=0.25,
        patience=2,
        min_delta=0.01,
        cooldown=1,
    )
    # Trigger plateau pressure
    assert ctrl.propose_action(model, 0.50, step=1)["action"] == "no_grow"
    prop = ctrl.propose_action(model, 0.50, step=2)
    assert prop["action"] == "grow_group_units"

    x = torch.randint(1, 8192, (1, 8))
    # Beneficial shadow growth -> accepted
    grown, rec_ok = ctrl.run_shadow_growth_cycle(
        model,
        prop,
        adapt_fn=lambda _m: None,
        eval_old_fn=lambda _m: 0.90,
        eval_new_fn=lambda m: 0.50 if m.count_parameters() == model.count_parameters() else 0.70,
        sample_input=x,
        seed=5,
    )
    assert rec_ok["accepted"] is True
    assert grown.count_parameters() > model.count_parameters()

    # Harmful retention breach -> rejected, parent untouched
    kept, rec_bad = ctrl.run_shadow_growth_cycle(
        model,
        prop,
        adapt_fn=lambda _m: None,
        eval_old_fn=lambda m: 0.90 if m.count_parameters() == model.count_parameters() else 0.50,
        eval_new_fn=lambda _m: 0.70,
        sample_input=x,
        seed=6,
    )
    assert rec_bad["accepted"] is False
    assert "retention_breach" in rec_bad["reject_reason"]
    assert kept is model
