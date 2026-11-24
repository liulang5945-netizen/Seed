"""Level-1 invariants for the RT backbone ``taiji/gru_rt.py`` (N32.0.01 prerequisite).

Mechanism standard §3 level 1, applied to the N32 candidate (D-20261124-495):
  01 no cross-sample leakage (bitwise)       02 neutral birth (bitwise)
  03 capsule round trip                      04 lifecycle: train mask == eval mask
  05 eval determinism                        06 gradient attribution / old-task isolation
  07 no dead parameters                      08 routing argument declared (scope label)
  09 same seed -> bitwise identical training
plus padding invariance (GRU72 issue 1) and per-tick dormancy (GRU72 issue 3).

Run: python -m pytest tests/test_gru_rt_level1_20261124.py -q
"""

from __future__ import annotations

import inspect
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from taiji.gru_rt import ARCH_VERSION, SCOPE, RTSubject  # noqa: E402

TASKS = {"cn": 8, "p24": 5, "p27": 4}
SEED = 20261124


def _ticks(seed: int, batch: int = 6, steps: int = 12, pad: int = 3) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, steps, 48, generator=g)
    if pad:
        x[:, steps - pad :] = 0.0
    return x


def _model(units: int = 8) -> RTSubject:
    return RTSubject(TASKS, units, SEED)


def _logits(model: RTSubject, ticks: torch.Tensor) -> dict[str, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        return {t: model(ticks, t).clone() for t in TASKS}


def _train(model: RTSubject, task: str, steps: int, params: list | None = None) -> None:
    model.train()
    opt = torch.optim.Adam(params if params is not None else model.task_parameters(task), lr=1e-2)
    for k in range(steps):
        x = _ticks(1000 + k)
        y = torch.arange(x.shape[0]) % 4
        loss = F.cross_entropy(model(x, task)[:, :4], y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


def test_inv01_no_cross_sample_leakage() -> None:
    model = _model()
    x = _ticks(1)
    y = x.clone()
    y[1:] = _ticks(2)[1:]
    a, b = _logits(model, x), _logits(model, y)
    for t in TASKS:
        assert torch.equal(a[t][0], b[t][0]), t


def test_padding_invariance() -> None:
    model = _model()
    x = _ticks(3, pad=0)
    padded = torch.cat([x, torch.zeros(x.shape[0], 5, 48)], dim=1)
    a, b = _logits(model, x), _logits(model, padded)
    for t in TASKS:
        assert torch.equal(a[t], b[t]), t


def test_inv02_neutral_birth_bitwise() -> None:
    model = _model()
    _train(model, "cn", 5)
    x = _ticks(4)
    before = _logits(model, x)
    model.grow_("cn", 6, seed=7)
    model.grow_("cn", 4, seed=8)
    after = _logits(model, x)
    for t in TASKS:
        assert torch.equal(before[t], after[t]), t
    assert model.column("cn").n_units == 8 + 6 + 4


def test_inv03_capsule_round_trip() -> None:
    model = _model()
    _train(model, "p24", 3)
    model.grow_("p27", 5, seed=9)
    model.set_dormant_("cn", 0, [1, 3], True)
    x = _ticks(5)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cap.pt"
        torch.save(model.capsule(), path)
        restored = RTSubject.from_capsule(torch.load(path))
    assert restored.spec() == model.spec()
    assert restored.spec()["arch_version"] == ARCH_VERSION
    a, b = _logits(model, x), _logits(restored, x)
    for t in TASKS:
        assert torch.equal(a[t], b[t]), t


def test_inv04_lifecycle_train_equals_eval_and_reversible() -> None:
    model = _model()
    x = _ticks(6)
    base = _logits(model, x)
    model.set_dormant_("cn", 0, [0, 2, 5], True)
    model.train()
    with torch.no_grad():
        train_out = model(x, "cn").clone()
    dormant = _logits(model, x)
    assert torch.equal(train_out, dormant["cn"])
    assert not torch.equal(dormant["cn"], base["cn"])
    states = model.column("cn").states(x)[0]
    assert torch.count_nonzero(states[:, [0, 2, 5]]) == 0
    model.train()
    loss = model(x, "cn").pow(2).sum()
    loss.backward()
    cell = model.column("cn").block_list()[0].cell
    for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
        grad = getattr(cell, name).grad.view(3, 8, -1)
        assert torch.count_nonzero(grad[:, [0, 2, 5]]) == 0, name
    model.set_dormant_("cn", 0, [0, 2, 5], False)
    assert torch.equal(_logits(model, x)["cn"], base["cn"])


def test_inv05_eval_determinism() -> None:
    model = _model()
    x = _ticks(7)
    a, b = _logits(model, x), _logits(model, x)
    for t in TASKS:
        assert torch.equal(a[t], b[t]), t


@pytest.mark.parametrize("all_params", [False, True])
def test_inv06_gradient_attribution_and_isolation(all_params: bool) -> None:
    model = _model()
    _train(model, "cn", 5)
    x = _ticks(8)
    before = _logits(model, x)
    params = list(model.parameters()) if all_params else None
    _train(model, "p24", 6, params)
    _train(model, "p27", 6, params)
    after = _logits(model, x)
    assert torch.equal(before["cn"], after["cn"])
    assert not torch.equal(before["p24"], after["p24"])
    model.zero_grad(set_to_none=True)
    model.train()
    model(x, "p27").sum().backward()
    for t in ("cn", "p24"):
        assert all(p.grad is None for p in model.task_parameters(t)), t
    assert all(p.grad is not None for p in model.task_parameters("p27"))


def test_inv07_no_dead_parameters() -> None:
    model = _model()
    x = _ticks(9)
    base = _logits(model, x)
    g = torch.Generator().manual_seed(11)
    for task in TASKS:
        col = model.column(task)
        for name, p in col.named_parameters():
            saved = p.detach().clone()
            with torch.no_grad():
                p.add_(0.05 * torch.randn(p.shape, generator=g))
            changed = not torch.equal(_logits(model, x)[task], base[task])
            with torch.no_grad():
                p.copy_(saved)
            assert changed, f"{task}.{name}"


def test_inv08_routing_is_declared() -> None:
    assert "task" in inspect.signature(RTSubject.forward).parameters
    assert "routing=external_task_id" in SCOPE
    with pytest.raises(KeyError):
        _model()(_ticks(10), "unknown")


def test_inv09_same_seed_bitwise() -> None:
    a, b = _model(), _model()
    for m in (a, b):
        _train(m, "cn", 4)
        _train(m, "p27", 4)
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k


def test_newborn_block_learns_after_training() -> None:
    model = _model()
    model.grow_("cn", 4, seed=12)
    head = model.column("cn").head_list()[1]
    assert torch.count_nonzero(head.weight) == 0
    _train(model, "cn", 3)
    assert torch.count_nonzero(head.weight) > 0
