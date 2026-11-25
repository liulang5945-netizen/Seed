"""Unit tests for the N32.2.03 v2 fixes (D-20261125-513/514).

Covers the three review findings:
  1. S replay actually happens (replay_batches > 0 when order has >1 kind).
  2. G3 uses each kind's post-learning peak as the BWT baseline.
  3. Same seed reproduces bitwise (deterministic generators, no ``hash()``).

Also covers RT Q3 checkpoint recording (the original engineering bug).
"""

from __future__ import annotations

import sys

sys.path.insert(0, r"E:\Seed")
import torch

import scripts.training.n32_2_03_s_vs_rt as m


def _tiny_setup(monkeypatch) -> tuple[dict, dict]:
    monkeypatch.setattr(m, "BATCH", 4)
    monkeypatch.setattr(m, "REPLAY", 1)
    monkeypatch.setattr(m, "EVAL_PER_KIND", 16)
    monkeypatch.setattr(m, "CKPT", (2,))
    rows = m.load_rows("train")
    for kind in m.KINDS:
        rows[kind] = rows[kind][:32]
    eval_data = m.build_eval(m.load_rows("val"))
    return rows, eval_data


def test_replay_happens(monkeypatch) -> None:
    rows, eval_data = _tiny_setup(monkeypatch)
    model = m.SharedGRU(units=8, seed=7)
    stats = m.train_shared(
        model, rows, ("q1", "q2", "q3"), steps=2, replay=True, eval_data=eval_data
    )
    assert stats["replay_batches"] > 0


def test_same_seed_bitwise(monkeypatch) -> None:
    rows, eval_data = _tiny_setup(monkeypatch)
    a = m.SharedGRU(units=8, seed=11)
    b = m.SharedGRU(units=8, seed=11)
    m.train_shared(a, rows, ("q1", "q2"), steps=2, replay=True, eval_data=eval_data)
    m.train_shared(b, rows, ("q1", "q2"), steps=2, replay=True, eval_data=eval_data)
    for (na, pa), (nb, pb) in zip(a.state_dict().items(), b.state_dict().items(), strict=True):
        assert na == nb
        assert torch.equal(pa, pb)


def test_g3_uses_peak_baseline() -> None:
    final = {"q1": 0.50, "q2": 0.60, "q3": 0.90}
    peaks = {"q1": 0.60, "q2": 0.60, "q3": 0.0}
    metrics = m.compute_metrics(final, peaks, {150: 0.9}, {150: 0.5})
    assert abs(metrics["g3_bwt_min"] - (-0.10)) < 1e-9
    assert abs(metrics["g2_shared_gain"] - 0.40) < 1e-9


def test_rt_q3_checkpoint_recorded(monkeypatch) -> None:
    rows, eval_data = _tiny_setup(monkeypatch)
    rt = m.build_rt(5)
    ckpt: dict[int, float] = {}
    m.train_rt_column(rt, rows, "q3", steps=2, eval_data=eval_data, ckpt=ckpt, seed=5)
    assert 2 in ckpt


if __name__ == "__main__":
    import _pytest.monkeypatch as mp

    monkeypatch = mp.MonkeyPatch()
    test_replay_happens(monkeypatch)
    test_same_seed_bitwise(monkeypatch)
    test_g3_uses_peak_baseline()
    monkeypatch.undo()
    print("N32.2.03 fixes: PASS")
