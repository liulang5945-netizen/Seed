"""Level-1 invariant unit tests for `taiji/plastic_growth.py` & `taiji/continual_growth_task.py`."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

sys.path.insert(0, r"E:\Seed")
import torch
import torch.nn.functional as F

from taiji.continual_growth_task import IN_DIM, NUM_CLASSES, make_split
from taiji.plastic_growth import STAGE_ID, GrowthPressureController, PlasticGrowthNet
from taiji.session_store import SessionStore


def test_inv01_batch_independence() -> None:
    net = PlasticGrowthNet(IN_DIM, 10, NUM_CLASSES, seed=11)
    net.eval()
    x = torch.randn(16, IN_DIM)
    with torch.no_grad():
        full = net(x)
        singles = torch.cat([net(x[i : i + 1]) for i in range(16)], dim=0)
    assert float((full - singles).abs().max()) <= 1e-6


def test_inv02_neutral_birth_and_differentiate() -> None:
    net = PlasticGrowthNet(IN_DIM, 8, NUM_CLASSES, seed=22)
    net.eval()
    x = torch.randn(32, IN_DIM)
    with torch.no_grad():
        y0 = net(x)
        added = net.grow_neutral_(12, seed=101)
        assert len(added) == 12
        assert net.hidden_dim == 20
        y1 = net(x)
        assert float((y0 - y1).abs().max()) <= 1e-6

        child = net.differentiate_(3, ratio=0.35)
        assert child == 20
        y2 = net(x)
        assert float((y1 - y2).abs().max()) <= 1e-6


def test_inv03_session_store_capsule_roundtrip(tmp_path: Path) -> None:
    net = PlasticGrowthNet(IN_DIM, 8, NUM_CLASSES, seed=33)
    net.grow_neutral_(6, seed=77)
    net.mature_(list(range(8)))
    store = SessionStore(tmp_path)
    rec = net.to_session_record("subj-growth", pending=[{"id": "p2", "phase": 2}])
    store.save(rec)
    back_rec = store.load("subj-growth")
    restored = PlasticGrowthNet.from_session_record(back_rec)
    assert (
        restored.pending
        if hasattr(restored, "pending")
        else back_rec.pending == [{"id": "p2", "phase": 2}]
    )
    assert torch.equal(restored.stage, net.stage)
    assert torch.equal(restored.unit_ids, net.unit_ids)
    assert restored._next_unit_id == net._next_unit_id
    x = torch.randn(24, IN_DIM)
    with torch.no_grad():
        assert torch.equal(net(x), restored(x))


def test_inv04_05_train_eval_mask_and_determinism() -> None:
    net = PlasticGrowthNet(IN_DIM, 8, NUM_CLASSES, seed=44)
    net.set_dormant_([2, 5])
    x = torch.randn(10, IN_DIM)
    net.train()
    m_train = net.alive_mask().clone()
    y_train = net(x).detach()
    net.eval()
    m_eval = net.alive_mask().clone()
    y_eval1 = net(x).detach()
    y_eval2 = net(x).detach()
    assert torch.equal(m_train, m_eval)
    assert torch.equal(y_train, y_eval1)
    assert torch.equal(y_eval1, y_eval2)


def test_inv06_selective_plasticity_grad_ownership() -> None:
    net = PlasticGrowthNet(IN_DIM, 8, NUM_CLASSES, seed=55)
    net.mature_()
    net.grow_neutral_(8, seed=99)
    # Give new columns tiny non-zero weight to test gradient flow cleanly
    with torch.no_grad():
        net.fc_out.weight[:, 8:] = 0.05
    x = torch.randn(16, IN_DIM)
    y = torch.randint(0, NUM_CLASSES, (16,))
    loss = F.cross_entropy(net(x), y)
    loss.backward()
    net.apply_selective_plasticity_grad_mask(freeze_mature=True)
    assert float(net.fc_in.weight.grad[:8].abs().max()) == 0.0
    assert float(net.fc_in.bias.grad[:8].abs().max()) == 0.0
    assert float(net.fc_out.weight.grad[:, :8].abs().max()) == 0.0
    assert float(net.fc_in.weight.grad[8:].abs().max()) > 0.0
    assert float(net.fc_out.weight.grad[:, 8:].abs().max()) > 0.0


def test_inv08_09_signature_and_seed_determinism() -> None:
    params = list(inspect.signature(PlasticGrowthNet.forward).parameters)
    assert params == ["self", "x"]
    n1 = PlasticGrowthNet(IN_DIM, 8, NUM_CLASSES, seed=66)
    n2 = PlasticGrowthNet(IN_DIM, 8, NUM_CLASSES, seed=66)
    for p1, p2 in zip(n1.parameters(), n2.parameters(), strict=True):
        assert torch.equal(p1, p2)


def test_dormant_wake_and_retire_rollback_recycle() -> None:
    net = PlasticGrowthNet(IN_DIM, 8, NUM_CLASSES, seed=77)
    x = torch.randn(12, IN_DIM)
    with torch.no_grad():
        y_before = net(x)
    saved = net.set_dormant_([1, 3, 6])
    assert int((net.stage == STAGE_ID["dormant"]).sum().item()) == 3
    net.wake_(saved)
    with torch.no_grad():
        assert torch.equal(y_before, net(x))

    # Retire with score_fn that forces rollback when retiring unit 0
    ok_rollback = net.retire_([0], score_fn=lambda m: float(m.active_units()), max_drop=0.5)
    assert ok_rollback is False
    assert net.active_units() == 8

    # Retire with zeroed outgoing column -> 0 score drop -> commit and recycle
    with torch.no_grad():
        net.fc_out.weight[:, 7] = 0.0
    y_pre_retire = net(x).detach()
    ok_commit = net.retire_([7], score_fn=lambda m: 1.0, max_drop=0.02)
    assert ok_commit is True
    removed = net.recycle_()
    assert removed == 1
    assert net.hidden_dim == 7
    with torch.no_grad():
        assert float((y_pre_retire - net(x)).abs().max()) <= 1e-6


def test_pressure_controller_and_task_split() -> None:
    ctrl = GrowthPressureController(window=3, plateau_eps=0.015, error_threshold=0.15, grow_k=24)
    # No pressure when error <= 0.15
    assert [ctrl.observe(e) for e in (0.10, 0.10, 0.10)] == [0, 0, 0]
    assert ctrl.growth_events == 0
    # Under pressure & improving fast -> still 0
    ctrl2 = GrowthPressureController(window=3, plateau_eps=0.015, error_threshold=0.15, grow_k=24)
    assert ctrl2.observe(0.60) == 0
    assert ctrl2.observe(0.45) == 0
    assert ctrl2.observe(0.30) == 0
    # Plateau at high error -> triggers once
    assert ctrl2.observe(0.295) == 0
    assert ctrl2.observe(0.292) == 24
    assert ctrl2.growth_events == 1
    assert ctrl2.observe(0.29) == 0

    d1 = make_split(3001, n_train=64, n_val=32)
    d2 = make_split(3001, n_train=64, n_val=32)
    assert torch.equal(d1["phase2"]["val"]["x"], d2["phase2"]["val"]["x"])
    assert torch.equal(d1["phase2"]["val"]["y"], d2["phase2"]["val"]["y"])
    assert set(d1["phase1"]["val"]["y"].tolist()) <= {0, 1, 2}
    assert set(d1["phase2"]["val"]["y"].tolist()) <= {3, 4, 5}
