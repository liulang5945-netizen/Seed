"""级1 不变量单测：GRU72 骨干（N31.4 REPLACE-STANDARD 候选）。

覆盖机制标准 §3 级1 中可离线验证的部分：
  不变量01 批内样本互不影响；05 评估确定性；09 同一 seed 重跑逐位一致；
  生命周期掩码可逆（休眠槽位读出置零）；read_gate 生效（非死参数）。

零 check/final。
"""

from __future__ import annotations

import sys

sys.path.insert(0, r"E:\Seed")
import torch

from taiji.gru72_backbone import GRU72Backbone


def _ticks(seed: int = 20261120, batch: int = 8, steps: int = 14) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(batch, steps, 48)


def test_forward_shapes() -> None:
    net = GRU72Backbone(seed=1)
    state, activity = net(_ticks())
    assert state.shape == (8, 72), state.shape
    assert activity.shape == (8, 72), activity.shape


def test_batch_independence() -> None:
    net = GRU72Backbone(seed=2)
    assert net.batch_independent(_ticks())


def test_eval_determinism() -> None:
    net = GRU72Backbone(seed=3)
    assert net.eval_deterministic(_ticks())


def test_same_seed_bitwise() -> None:
    ticks = _ticks()
    a = GRU72Backbone(seed=4)
    b = GRU72Backbone(seed=4)
    a.eval()
    b.eval()
    with torch.no_grad():
        assert torch.equal(a(ticks)[0], b(ticks)[0])
        assert torch.equal(a(ticks)[1], b(ticks)[1])


def test_lifecycle_dormant_reversible() -> None:
    net = GRU72Backbone(seed=5)
    ticks = _ticks()
    net.eval()
    with torch.no_grad():
        before = net(ticks)[0]
    net.set_lifecycle_(range(24, 48), active=False)
    assert net.active_units() == 48
    with torch.no_grad():
        after = net(ticks)[0]
    assert torch.count_nonzero(after[:, 24:48]) == 0
    assert torch.allclose(after[:, :24], before[:, :24], atol=1e-6)
    net.set_lifecycle_(range(24, 48), active=True)
    with torch.no_grad():
        restored = net(ticks)[0]
    assert torch.equal(restored, before)


def test_read_gate_non_dead() -> None:
    net = GRU72Backbone(seed=6)
    assert net.read_gate_effect(_ticks()) > 1e-3


if __name__ == "__main__":
    test_forward_shapes()
    test_batch_independence()
    test_eval_determinism()
    test_same_seed_bitwise()
    test_lifecycle_dormant_reversible()
    test_read_gate_non_dead()
    print("GRU72 backbone level-1 invariants: PASS")
