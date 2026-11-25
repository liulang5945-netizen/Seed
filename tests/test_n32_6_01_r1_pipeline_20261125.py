from __future__ import annotations

import torch

from scripts.training.n32_5_01_q3_gru_learnability import AUG_SEED, Q3_LEN, relabel
from scripts.training.n32_6_01_r1_edge_hop import (
    compute_r1_verdict,
    encode_all_kinds_fast,
    make_column,
)
from taiji import composition_encoding as ce
from taiji import native_composition_v2 as nc2
from taiji.modality_edge_hop import extract_modality_masks
from taiji.native_composition import NAMES


def _encode_all_reference(
    scenes: list[dict], copies: int
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    gen = torch.Generator().manual_seed(AUG_SEED)
    ticks: dict[str, list[torch.Tensor]] = {"q1": [], "q2": [], "q3": []}
    labels: dict[str, list[int]] = {"q1": [], "q2": [], "q3": []}
    for scene in scenes:
        variants = [scene]
        for _ in range(copies):
            perm = [NAMES[i] for i in torch.randperm(len(NAMES), generator=gen).tolist()]
            variants.append(relabel(scene, perm))
        for v in variants:
            for q in nc2.questions(v):
                kind = str(q["kind"])
                ticks[kind].append(ce.encode(v, q["text"])[:Q3_LEN].to(torch.float16))
                labels[kind].append(ce.ANSWER_INDEX[str(q["answer"])])
    return {
        k: (torch.stack(ticks[k]), torch.tensor(labels[k], dtype=torch.long))
        for k in ("q1", "q2", "q3")
    }


def test_encode_all_kinds_fast_bit_identical_to_reference() -> None:
    scenes = nc2.load_scenes("val")[:10]
    fast = encode_all_kinds_fast(scenes, copies=3)
    ref = _encode_all_reference(scenes, copies=3)
    for kind in ("q1", "q2", "q3"):
        fx, fy = fast[kind]
        rx, ry = ref[kind]
        assert fx.shape == rx.shape
        assert torch.equal(fx, rx), f"{kind} tensor mismatch"
        assert torch.equal(fy, ry), f"{kind} label mismatch"


def test_mixed_task_batch_invariance() -> None:
    scenes = nc2.load_scenes("val")[:8]
    raw = encode_all_kinds_fast(scenes, copies=0)
    model = make_column(20261120, "test")
    model.eval()

    xs, sms, qms, sep_logits = [], [], [], []
    with torch.no_grad():
        for kind in ("q1", "q2", "q3"):
            x = raw[kind][0].float()
            sm, qm = extract_modality_masks(raw[kind][0])
            xs.append(x)
            sms.append(sm)
            qms.append(qm)
            sep_logits.append(model(x, sm, qm))
        x_cat = torch.cat(xs, dim=0)
        sm_cat = torch.cat(sms, dim=0)
        qm_cat = torch.cat(qms, dim=0)
        sep_cat = torch.cat(sep_logits, dim=0)
        perm = torch.randperm(x_cat.shape[0], generator=torch.Generator().manual_seed(17))
        mix_logits = model(x_cat[perm], sm_cat[perm], qm_cat[perm])

    assert torch.allclose(mix_logits, sep_cat[perm], atol=1e-5)


def test_r1_metrics_g2_and_g3() -> None:
    s_pos = {
        "replay_batches": 10,
        "mixed_equivalence": {"exact": True},
        "final_accs": {"q1": 0.96, "q2": 0.94, "q3": 0.98},
        "peaks_before_q3": {"q1": 0.97, "q2": 0.95},
        "q3_ckpts": {"1000": 0.72, "2000": 0.90, "4000": 0.97, "6000": 0.98},
    }
    rt_pos = {
        "q3_ckpts": {"1000": 0.60, "2000": 0.78, "4000": 0.95, "6000": 0.98},
    }
    assert compute_r1_verdict(s_pos, rt_pos)["exit"] == "pilot_signal_positive"

    s_ceil = {**s_pos, "q3_ckpts": {"1000": 0.95, "2000": 0.97, "4000": 0.98, "6000": 0.98}}
    rt_ceil = {"q3_ckpts": {"1000": 0.96, "2000": 0.97, "4000": 0.98, "6000": 0.98}}
    assert compute_r1_verdict(s_ceil, rt_ceil)["exit"] == "pilot_inconclusive"

    s_forget = {**s_pos, "final_accs": {"q1": 0.80, "q2": 0.94, "q3": 0.98}}
    assert compute_r1_verdict(s_forget, rt_pos)["exit"] == "pilot_signal_negative"
