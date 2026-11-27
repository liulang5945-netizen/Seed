"""Tests for 态极 Native Byte-Group Backbone (`taiji-native-byte-group-v2`)."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import torch

from taiji.native_cognitive_backbone import (
    NATIVE_PRESETS,
    NativeBackboneConfig,
    StatefulNeuralGroup,
    TaijiNativeCognitiveBackbone,
    bits_per_byte,
)
from taiji.native_growth_controller import NativeGrowthController, NativeGrowthPolicy


def test_native_backbone_is_strictly_non_transformer_and_zero_bpe() -> None:
    src = Path("taiji/native_cognitive_backbone.py").read_text(encoding="utf-8")
    assert "class GQAAttention" not in src
    assert "class TransformerBlock" not in src
    assert "F.scaled_dot_product_attention" not in src
    assert "tokenizers" not in src


def test_init_loss_sanity_and_byte_causality() -> None:
    torch.manual_seed(7)
    tiny = TaijiNativeCognitiveBackbone(NATIVE_PRESETS["native_byte_tiny"])
    tiny.eval()
    x = torch.randint(0, 256, (2, 32))
    y = torch.randint(0, 256, (2, 32))
    with torch.no_grad():
        logits_a, loss, _ = tiny(x, labels=y)
    assert loss is not None
    assert 4.5 < float(loss.item()) < 6.5
    assert 6.5 < bits_per_byte(float(loss.item())) < 9.5

    # Strict byte causality: changing last byte must not change earlier logits
    x_mod = x.clone()
    x_mod[:, -1] = (x_mod[:, -1] + 17) % 256
    with torch.no_grad():
        logits_b, _, _ = tiny(x_mod)
    assert torch.allclose(logits_a[:, :-1, :], logits_b[:, :-1, :], atol=1e-5)


def test_fp1_kkt_freeze_and_no_f_ablation() -> None:
    torch.manual_seed(11)
    model = TaijiNativeCognitiveBackbone(NATIVE_PRESETS["native_byte_tiny"])
    model.eval()
    x = torch.randint(0, 256, (2, 24))
    st0 = model.init_state(2)
    with torch.no_grad():
        _, _, st_frozen = model(x, state=st0, p_allowed=0.0)
    for f0, f1 in zip(st0.fast_memories, st_frozen.fast_memories, strict=True):
        assert torch.equal(f0, f1)

    # Explicit no-F strong baseline ablation switch
    cfg_no_f = NativeBackboneConfig(d_model=128, n_groups=4, group_units=64, use_fast_memory=False)
    m_no_f = TaijiNativeCognitiveBackbone(cfg_no_f)
    logits, loss, _ = m_no_f(x, labels=x)
    assert logits.shape == (2, 24, 257)
    assert loss is not None and torch.isfinite(loss)


def test_svd_zero_in_birth_proliferation_residual_group_and_capsule() -> None:
    torch.manual_seed(13)
    model = TaijiNativeCognitiveBackbone(NATIVE_PRESETS["native_byte_tiny"])
    model.eval()
    x = torch.randint(0, 256, (2, 24))
    y = torch.randint(0, 256, (2, 24))

    grp0 = cast(StatefulNeuralGroup, model.groups[0])
    old_n = grp0.n_units

    with torch.no_grad():
        logits0, _, _ = model(x)
        ev1 = model.grow_stateful_units_svd_neutral_(group_idx=0, budget=8, seed=1)
        assert ev1["count"] == 8
        logits1, _, _ = model(x)
        model.proliferate_units_neutral_(group_idx=0, parent_indices=[0, 1])
        logits2, _, _ = model(x)
        model.grow_group_neutral_(seed=2)
        logits3, _, _ = model(x)

    assert float((logits1 - logits0).abs().max().item()) < 1e-5
    assert float((logits2 - logits1).abs().max().item()) < 1e-5
    assert float((logits3 - logits2).abs().max().item()) < 1e-5

    # First backward step must give non-zero analytical startup credit on new unit_r
    model.train()
    model.zero_grad(set_to_none=True)
    _, loss, _ = model(x, labels=y)
    assert loss is not None
    loss.backward()
    assert grp0.unit_r.grad is not None
    assert float(grp0.unit_r.grad[old_n : old_n + 8].norm().item()) > 0.0

    # Selective plasticity grad mask zeroes mature units & w_slow
    model.mark_all_mature_()
    model.grow_stateful_units_svd_neutral_(group_idx=0, budget=4, seed=3)
    model.zero_grad(set_to_none=True)
    _, loss2, _ = model(x, labels=y)
    assert loss2 is not None
    loss2.backward()
    masked = model.apply_selective_plasticity_grad_mask(freeze_mature_slow=True)
    assert masked["masked_units"] > 0
    assert grp0.w_slow.weight.grad is not None
    assert float(grp0.w_slow.weight.grad.abs().max().item()) == 0.0

    # Capsule roundtrip after structural growth
    model.eval()
    cap = model.export_capsule("native_byte")
    restored = TaijiNativeCognitiveBackbone.from_capsule(cap, "native_byte")
    restored.eval()
    with torch.no_grad():
        la, _, _ = model(x)
        lb, _, _ = restored(x)
    assert torch.equal(la, lb)


def test_utf8_legal_generation_and_growth_controller() -> None:
    torch.manual_seed(17)
    model = TaijiNativeCognitiveBackbone(NATIVE_PRESETS["native_byte_tiny"])
    text, _ = model.generate_utf8("今天", max_new_bytes=24)
    assert isinstance(text, str)

    ctrl = NativeGrowthController(
        NativeGrowthPolicy(auto_grow_enabled=True, min_gain=0.01, max_retention_drop=0.02),
        error_threshold=0.25,
        patience=2,
        min_delta=0.01,
        cooldown=1,
    )
    assert ctrl.propose_action(model, 0.50, step=1)["action"] == "no_grow"
    prop = ctrl.propose_action(model, 0.50, step=2)
    assert prop["action"] == "grow_stateful_units_svd"
    assert prop["trigger"] == "engineering_prototype"

    x = torch.randint(0, 256, (1, 16))
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
    assert rec_ok["trigger"] == "engineering_prototype"
    assert grown.count_parameters() > model.count_parameters()
