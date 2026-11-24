"""Level-1 tests for ``taiji-gru-rt-v2`` lifecycle operations (D-20261124-496).

Release-01 prerequisite: differentiation, integration, retention-verified retirement and
recycling on the N32 backbone, plus v2 == v1 bitwise when no lifecycle op is used (so the
N32.0 evidence obtained on v1 carries over).

Run: python -m pytest tests/test_gru_rt_lifecycle_20261124.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from taiji.gru_rt import RTSubject  # noqa: E402
from taiji.gru_rt_lifecycle import ARCH_VERSION_V2, LifecycleRTSubject  # noqa: E402

TASKS = {"cn": 8, "p24": 5, "p27": 4}
SEED = 20261124
TOL = 1e-5


def _ticks(seed: int, batch: int = 6, steps: int = 12, pad: int = 3) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, steps, 48, generator=g)
    if pad:
        x[:, steps - pad :] = 0.0
    return x


def _out(model: RTSubject, ticks: torch.Tensor) -> dict[str, torch.Tensor]:
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


def _trained(grow: bool = True) -> LifecycleRTSubject:
    m = LifecycleRTSubject(TASKS, 8, SEED)
    _train(m, "cn", 5)
    if grow:
        m.grow_("cn", 4, seed=7)
        _train(m, "cn", 3)
    _train(m, "p24", 3)
    return m


def _maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max())


def test_v2_equals_v1_bitwise_without_lifecycle_ops() -> None:
    a = RTSubject(TASKS, 8, SEED)
    b = LifecycleRTSubject(TASKS, 8, SEED)
    for m in (a, b):
        _train(m, "cn", 4)
        m.grow_("cn", 4, seed=7)
        _train(m, "cn", 2)
        _train(m, "p27", 4)
    sa, sb = a.state_dict(), b.state_dict()
    assert set(sb) - set(sa) == {k for k in sb if k.endswith(".stage")}
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k
    x = _ticks(1)
    oa, ob = _out(a, x), _out(b, x)
    for t in TASKS:
        assert torch.equal(oa[t], ob[t]), t


def test_stage_lifecycle() -> None:
    m = LifecycleRTSubject(TASKS, 8, SEED)
    m.grow_("p27", 4, seed=3)
    assert m.stage_names("p27")[0] == ["mature"] * 8
    assert m.stage_names("p27")[1] == ["immature"] * 4
    m.promote_("p27")
    assert m.stage_names("p27")[1] == ["mature"] * 4
    m.set_dormant_("p27", 1, [0, 2], True)
    assert m.stage_names("p27")[1][:3] == ["dormant", "mature", "dormant"]
    m.set_dormant_("p27", 1, [0, 2], False)
    assert m.stage_names("p27")[1] == ["mature"] * 4


@pytest.mark.parametrize("block", [0, 1])
def test_differentiate_preserves_function_and_breaks_symmetry(block: int) -> None:
    m = _trained()
    x = _ticks(4)
    before = _out(m, x)
    units = m.column("cn").block_list()[block].units
    new = m.differentiate_("cn", block, 2)
    after = _out(m, x)
    assert new == units
    assert m.column("cn").block_list()[block].units == units + 1
    assert m.stage_names("cn")[block][new] == "immature"
    assert _maxdiff(before["cn"], after["cn"]) <= TOL
    for t in ("p24", "p27"):
        assert torch.equal(before[t], after[t]), t
    if block == 0:
        later = m.column("cn").block_list()[1]
        assert later.lateral == units + 1
        assert later.cell.input_size == 48 + units + 1
    cell = m.column("cn").block_list()[block].cell
    u = units + 1
    assert torch.equal(cell.weight_ih[2], cell.weight_ih[new])
    _train(m, "cn", 4)
    assert not torch.equal(cell.weight_ih[2], cell.weight_ih[new])
    assert not torch.equal(cell.weight_hh[2], cell.weight_hh[new])
    assert cell.weight_hh.shape == (3 * u, u)


def test_integrate_inverts_differentiate() -> None:
    m = _trained()
    x = _ticks(5)
    before = _out(m, x)
    new = m.differentiate_("cn", 0, 3)
    rep = m.integrate_("cn", 0, keep=3, drop=new, probe=x, tol=TOL)
    assert rep["committed"]
    assert m.column("cn").block_list()[0].units == 8
    assert _maxdiff(before["cn"], _out(m, x)["cn"]) <= TOL


def test_integrate_rolls_back_distinct_units() -> None:
    m = _trained()
    x = _ticks(6)
    before = _out(m, x)
    state = {k: v.clone() for k, v in m.state_dict().items()}
    rep = m.integrate_("cn", 0, keep=0, drop=1, probe=x, tol=1e-6)
    assert not rep["committed"]
    assert m.column("cn").block_list()[0].units == 8
    after_state = m.state_dict()
    assert after_state.keys() == state.keys()
    for k in state:
        assert torch.equal(state[k], after_state[k]), k
    for t in TASKS:
        assert torch.equal(before[t], _out(m, x)[t]), t


def _make_useless(m: LifecycleRTSubject, unit: int) -> None:
    col = m.column("cn")
    with torch.no_grad():
        col.block_list()[0].cell.weight_hh[:, unit] = 0.0
        col.head_list()[0].weight[:, unit] = 0.0
        col.block_list()[1].cell.weight_ih[:, 48 + unit] = 0.0


def test_retire_is_retention_verified() -> None:
    m = _trained()
    _make_useless(m, 5)
    x = _ticks(7)
    base = _out(m, x)

    def score(mm: LifecycleRTSubject) -> dict[str, float]:
        return {"cn": -_maxdiff(_out(mm, x)["cn"], base["cn"])}

    rep = m.retire_("cn", 0, [1], score, max_drop=0.0)
    assert not rep["committed"]
    assert m.stage_names("cn")[0][1] == "mature"
    assert float(m.column("cn").block_list()[0].alive[1]) == 1.0
    for t in TASKS:
        assert torch.equal(base[t], _out(m, x)[t]), t
    rep = m.retire_("cn", 0, [5], score, max_drop=0.0)
    assert rep["committed"]
    assert m.stage_names("cn")[0][5] == "retired"
    for t in TASKS:
        assert torch.equal(base[t], _out(m, x)[t]), t
    with pytest.raises(ValueError):
        m.set_dormant_("cn", 0, [5], False)


def test_recycle_removes_retired_units_and_round_trips() -> None:
    m = _trained()
    _make_useless(m, 5)
    x = _ticks(8)
    base = _out(m, x)
    rep = m.retire_("cn", 0, [5], lambda mm: {"cn": 0.0}, max_drop=0.0)
    assert rep["committed"]
    assert m.recycle_("cn", 0) == 1
    blocks = m.column("cn").block_list()
    assert blocks[0].units == 7
    assert blocks[1].lateral == 7
    assert blocks[1].cell.input_size == 48 + 7
    assert "retired" not in m.stage_names("cn")[0]
    after = _out(m, x)
    assert _maxdiff(base["cn"], after["cn"]) <= 1e-6
    for t in ("p24", "p27"):
        assert torch.equal(base[t], after[t]), t
    restored = LifecycleRTSubject.from_capsule(m.capsule())
    assert restored.spec() == m.spec()
    assert restored.spec()["arch_version"] == ARCH_VERSION_V2
    for t in TASKS:
        assert torch.equal(after[t], _out(restored, x)[t]), t
    assert restored.stage_names("cn") == m.stage_names("cn")
    with pytest.raises(ValueError):
        RTSubject.from_capsule(m.capsule())


def test_isolation_after_structural_ops() -> None:
    m = _trained()
    m.differentiate_("cn", 0, 1)
    x = _ticks(9)
    before = _out(m, x)
    _train(m, "p24", 4, list(m.parameters()))
    _train(m, "p27", 4, list(m.parameters()))
    after = _out(m, x)
    assert torch.equal(before["cn"], after["cn"])
    m.zero_grad(set_to_none=True)
    m.train()
    m(x, "p27").sum().backward()
    for t in ("cn", "p24"):
        assert all(p.grad is None for p in m.task_parameters(t)), t


def test_no_dead_parameters_after_differentiate() -> None:
    m = _trained()
    m.differentiate_("cn", 1, 0)
    x = _ticks(10)
    base = _out(m, x)["cn"]
    g = torch.Generator().manual_seed(11)
    for name, p in m.column("cn").named_parameters():
        saved = p.detach().clone()
        with torch.no_grad():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
        changed = not torch.equal(_out(m, x)["cn"], base)
        with torch.no_grad():
            p.copy_(saved)
        assert changed, name


def test_invalid_operations_raise() -> None:
    m = _trained(grow=False)
    with pytest.raises(ValueError):
        m.differentiate_("cn", 0, 0, ratio=0.5)
    with pytest.raises(ValueError):
        m.integrate_("cn", 0, keep=2, drop=2)
    m.set_dormant_("cn", 0, [4], True)
    with pytest.raises(ValueError):
        m.differentiate_("cn", 0, 4)
    rep = m.retire_("cn", 0, list(range(8)), lambda mm: {"cn": 0.0}, max_drop=0.0)
    assert rep["committed"]
    with pytest.raises(ValueError):
        m.recycle_("cn", 0)
