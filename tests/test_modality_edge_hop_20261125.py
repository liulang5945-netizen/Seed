from __future__ import annotations

import torch
import torch.nn.functional as F

from taiji.modality_edge_hop import (
    DEFAULT_QUERY_FLAG,
    DEFAULT_SCENE_FLAG,
    INPUT_DIM,
    ModalityEdgeHop,
    extract_modality_masks,
)


def _make_sample(batch: int = 4, n_obj: int = 3, n_q: int = 6) -> tuple[torch.Tensor, torch.Tensor]:
    objs = torch.randn(batch, n_obj, INPUT_DIM)
    objs[:, :, DEFAULT_SCENE_FLAG] = 1.0
    objs[:, :, DEFAULT_QUERY_FLAG] = 0.0
    q = torch.randn(batch, n_q, INPUT_DIM)
    q[:, :, DEFAULT_SCENE_FLAG] = 0.0
    q[:, :, DEFAULT_QUERY_FLAG] = 1.0
    return objs, q


def test_deterministic_init_and_capsule_roundtrip() -> None:
    m1 = ModalityEdgeHop(10, d_model=64, d_edge=64, n_edge_layers=2, n_hops=2, seed=99)
    m2 = ModalityEdgeHop(10, d_model=64, d_edge=64, n_edge_layers=2, n_hops=2, seed=99)
    m1.eval()
    m2.eval()

    objs, q = _make_sample()
    raw = torch.cat([objs, q], dim=1)
    sm, qm = extract_modality_masks(raw)
    with torch.no_grad():
        y1 = m1(raw, sm, qm)
        y2 = m2(raw, sm, qm)
        restored = ModalityEdgeHop.from_capsule(m1.capsule())
        restored.eval()
        y_restored = restored(raw, sm, qm)

    assert torch.equal(y1, y2)
    assert torch.equal(y1, y_restored)


def test_exact_object_permutation_invariance() -> None:
    model = ModalityEdgeHop(10, seed=7)
    model.eval()
    objs, q = _make_sample(batch=5, n_obj=3, n_q=8)
    raw = torch.cat([objs, q], dim=1)
    sm, qm = extract_modality_masks(raw)

    perm = [2, 0, 1]
    raw_perm = torch.cat([objs[:, perm], q], dim=1)
    sm_perm, qm_perm = extract_modality_masks(raw_perm)

    with torch.no_grad():
        out1 = model(raw, sm, qm)
        out2 = model(raw_perm, sm_perm, qm_perm)

    assert torch.allclose(out1, out2, atol=1e-5)


def test_sf_vs_qf_and_padding_invariance() -> None:
    model = ModalityEdgeHop(10, seed=11)
    model.eval()
    objs, q = _make_sample(batch=4, n_obj=3, n_q=7)

    sf = torch.cat([objs, q], dim=1)
    sf_sm, sf_qm = extract_modality_masks(sf)

    qf = torch.cat([q, objs], dim=1)
    qf_sm, qf_qm = extract_modality_masks(qf)

    padded = torch.zeros(4, 15, INPUT_DIM)
    padded[:, :3] = objs
    padded[:, 5:12] = q  # non-contiguous gap forces masked-GRU fallback
    pad_sm, pad_qm = extract_modality_masks(padded)

    with torch.no_grad():
        out_sf = model(sf, sf_sm, sf_qm)
        out_qf = model(qf, qf_sm, qf_qm)
        out_pad = model(padded, pad_sm, pad_qm)

    assert torch.allclose(out_sf, out_qf, atol=1e-5)
    assert torch.allclose(out_sf, out_pad, atol=1e-5)


def test_gradient_flow_and_param_budget() -> None:
    model = ModalityEdgeHop(10, d_model=64, d_edge=64, n_edge_layers=2, n_hops=2, seed=0)
    objs, q = _make_sample(batch=4, n_obj=3, n_q=8)
    raw = torch.cat([objs, q], dim=1)
    sm, qm = extract_modality_masks(raw)
    y = torch.tensor([0, 2, 5, 8], dtype=torch.long)

    loss = F.cross_entropy(model(raw, sm, qm), y)
    loss.backward()

    for prefix in ("q_gru", "obj_in", "edge_mlps", "null_obj", "hop_q", "head"):
        grads = [p.grad for n, p in model.named_parameters() if prefix in n]
        assert grads, f"missing parameter matching {prefix}"
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert any(float(g.abs().sum()) > 0.0 for g in grads if g is not None)

    assert model.n_params() <= 140_000
