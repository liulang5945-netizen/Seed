from __future__ import annotations

import torch
import torch.nn.functional as F

from taiji.slot_memory_gru import INPUT_DIM, SlotHopGRU


def test_deterministic_init_and_capsule_roundtrip() -> None:
    m1 = SlotHopGRU(10, d_model=48, n_heads=4, n_hops=2, seed=42)
    m2 = SlotHopGRU(10, d_model=48, n_heads=4, n_hops=2, seed=42)
    m1.eval()
    m2.eval()

    x = torch.randn(4, 15, INPUT_DIM)
    with torch.no_grad():
        y1 = m1(x)
        y2 = m2(x)
        restored = SlotHopGRU.from_capsule(m1.capsule())
        restored.eval()
        y_restored = restored(x)

    assert torch.equal(y1, y2)
    assert torch.equal(y1, y_restored)


def test_padding_invariance_and_fast_vs_masked_gru_equivalence() -> None:
    model = SlotHopGRU(10, d_model=48, n_heads=4, n_hops=2, seed=7)
    model.eval()

    x_valid = torch.randn(5, 12, INPUT_DIM)
    x_padded = torch.zeros(5, 20, INPUT_DIM)
    x_padded[:, :12, :] = x_valid

    with torch.no_grad():
        out_unpadded = model(x_valid)
        out_padded = model(x_padded)

    assert torch.allclose(out_unpadded, out_padded, atol=1e-5)


def test_gradient_flow_to_all_components() -> None:
    model = SlotHopGRU(10, d_model=48, n_heads=4, n_hops=2, seed=13)
    x = torch.randn(4, 16, INPUT_DIM)
    x[:, 14:, :] = 0.0
    y = torch.tensor([0, 3, 5, 9], dtype=torch.long)

    loss = F.cross_entropy(model(x), y)
    loss.backward()

    for prefix in ("gru", "rel_qkv", "hop_q", "hop_kv", "null_slot", "hop_cell", "head"):
        grads = [p.grad for n, p in model.named_parameters() if prefix in n]
        assert grads, f"missing parameter matching {prefix}"
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert any(float(g.abs().sum()) > 0.0 for g in grads if g is not None)


def test_parameter_budget_comparable_to_gru96() -> None:
    model = SlotHopGRU(10, d_model=48, n_heads=4, n_hops=2, seed=0)
    assert model.n_params() <= 80_000
