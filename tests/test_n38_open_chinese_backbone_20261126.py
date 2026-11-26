"""Unit tests for N38 Open-Chinese Unified Cognitive Backbone."""

from __future__ import annotations

import torch

from taiji.open_chinese_backbone import PRESETS, TaijiOpenChineseLM


def test_presets_instantiate_and_param_counts() -> None:
    expected = {
        "small_8m": (6_000_000, 11_000_000),
        "mid_18m": (14_000_000, 22_000_000),
        "large_30m": (25_000_000, 36_000_000),
    }
    for name, (lo, hi) in expected.items():
        model = TaijiOpenChineseLM(PRESETS[name])
        n = model.count_parameters()
        assert lo <= n <= hi, f"{name} params={n} out of [{lo}, {hi}]"


def test_forward_backward_memory_and_capsule_roundtrip() -> None:
    torch.manual_seed(42)
    model = TaijiOpenChineseLM(PRESETS["small_8m"])
    x = torch.randint(1, 8192, (2, 32))
    y = torch.randint(1, 8192, (2, 32))

    mem0 = model.init_mem_state(2)
    logits0, loss0, mem1 = model(x, mem_state=mem0, labels=y)
    assert logits0.shape == (2, 32, 8192)
    assert loss0 is not None and torch.isfinite(loss0)
    loss0.backward()
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()

    model.eval()
    with torch.no_grad():
        logits_a, _, _ = model(x, mem_state=mem0)
        logits_b, _, _ = model(x, mem_state=mem1)
        assert not torch.allclose(logits_a, logits_b)

        # Causal invariance: changing last token does not change earlier logits
        x_mod = x.clone()
        x_mod[:, -1] = (x_mod[:, -1] + 1) % 8191 + 1
        logits_mod, _, _ = model(x_mod, mem_state=mem0)
        assert torch.allclose(logits_a[:, :-1, :], logits_mod[:, :-1, :], atol=1e-5)

        # Capsule roundtrip
        cap = model.export_capsule("open_zh")
        model2 = TaijiOpenChineseLM(PRESETS["small_8m"])
        model2.load_capsule(cap, "open_zh")
        model2.eval()
        logits_c, _, _ = model2(x, mem_state=mem0)
        assert torch.equal(logits_a, logits_c)
