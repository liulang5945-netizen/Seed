from __future__ import annotations

import torch

from scripts.training.n32_5_01_q3_gru_learnability import (
    AUG_SEED,
    Q3_LEN,
    encode_q3_fast,
    moments_dim,
    qf_order,
    relabel,
)
from taiji import composition_encoding as ce
from taiji import native_composition_v2 as nc2
from taiji.native_composition import NAMES


def _encode_q3_reference(scenes: list[dict], copies: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(AUG_SEED)
    ticks: list[torch.Tensor] = []
    labels: list[int] = []
    for scene in scenes:
        variants = [scene]
        for _ in range(copies):
            perm = [NAMES[i] for i in torch.randperm(len(NAMES), generator=gen).tolist()]
            variants.append(relabel(scene, perm))
        for v in variants:
            for q in nc2.questions(v):
                if q["kind"] != "q3":
                    continue
                ticks.append(ce.encode(v, q["text"])[:Q3_LEN].to(torch.float16))
                labels.append(ce.ANSWER_INDEX[str(q["answer"])])
    return torch.stack(ticks), torch.tensor(labels, dtype=torch.long)


def test_encode_q3_fast_is_bit_identical_to_reference() -> None:
    scenes = nc2.load_scenes("val")[:10]
    fast_x, fast_y = encode_q3_fast(scenes, copies=3)
    ref_x, ref_y = _encode_q3_reference(scenes, copies=3)

    assert fast_x.shape == ref_x.shape
    assert fast_x.dtype == torch.float16
    assert torch.equal(fast_x, ref_x)
    assert torch.equal(fast_y, ref_y)


def test_qf_order_preserves_normalization_equivalence() -> None:
    scenes = nc2.load_scenes("val")[:5]
    raw_x, _ = encode_q3_fast(scenes, copies=1)
    mu, sd = moments_dim(raw_x)
    order = qf_order(Q3_LEN)

    reordered_then_norm = (raw_x[:, order].float() - mu) / sd
    norm_then_reordered = ((raw_x.float() - mu) / sd)[:, order]
    assert torch.equal(reordered_then_norm, norm_then_reordered)
