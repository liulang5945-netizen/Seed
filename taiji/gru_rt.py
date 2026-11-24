"""GRU+read_gate backbone with external task routing (RT columns) for N32 admission.

N31 closed on the contraction branch (D-20261124-493 / D-494): N32 trains with external
routing and must carry the scope label below.  This module is the frozen candidate for
release condition 06; level-1 invariants live in ``tests/test_gru_rt_level1_20261124.py``.

Structure
  * one *column* per task name; the task name is an explicit forward argument (external
    routing, invariant 08 is answered by declaring it, not by hiding it);
  * a column is a list of *blocks*; block k is a ``GRUCell`` whose input is the tick plus
    the current states of blocks 0..k-1 of the same column (lateral, one direction only);
  * each block has the final-tick multiplicative read_gate validated in N31.4
    (``Linear(48 -> units)`` on the last valid input, bias 1.0);
  * the readout first layer is a sum of per-block linears, so a newborn block whose linear
    is zero leaves every old output bitwise unchanged (neutral birth, invariant 02);
  * all-zero ticks are padding: the state is held, and the read_gate uses the last valid
    tick (fixes GRU72 issue 1);
  * every unit has an ``alive`` flag applied at every tick (dormancy is local, reversible
    and gradient-free while dormant; fixes GRU72 issue 3).

Old-task protection is by construction (disjoint columns), not a learned property.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn

ARCH_VERSION = "taiji-gru-rt-v1"
SCOPE = (
    "routing=external_task_id; protection=disjoint_columns(construction); heads=per_task; "
    "state=episode_reset; scenario=task_incremental"
)
INPUT_DIM = 48
READOUT_HIDDEN = 64


def valid_steps(ticks: torch.Tensor) -> torch.Tensor:
    """(B, T) float mask; a tick whose features are all zero is padding."""
    return (ticks.abs().sum(-1) > 0).to(ticks.dtype)


class Block(nn.Module):
    alive: torch.Tensor

    def __init__(self, units: int, lateral: int) -> None:
        super().__init__()
        self.units = int(units)
        self.lateral = int(lateral)
        self.cell = nn.GRUCell(INPUT_DIM + self.lateral, self.units)
        self.read_gate = nn.Linear(INPUT_DIM, self.units)
        nn.init.constant_(self.read_gate.bias, 1.0)
        self.register_buffer("alive", torch.ones(self.units))


class Column(nn.Module):
    def __init__(self, units: int, out_dim: int) -> None:
        super().__init__()
        self.out_dim = int(out_dim)
        self.blocks = nn.ModuleList([Block(units, 0)])
        self.head_in = nn.ModuleList([nn.Linear(units, READOUT_HIDDEN)])
        self.head_out = nn.Linear(READOUT_HIDDEN, self.out_dim)

    def block_list(self) -> list[Block]:
        return [cast(Block, b) for b in self.blocks]

    def head_list(self) -> list[nn.Linear]:
        return [cast(nn.Linear, h) for h in self.head_in]

    @property
    def n_units(self) -> int:
        return sum(b.units for b in self.block_list())

    def grow_(self, units: int) -> None:
        """Append a block reading the existing blocks; its readout weight starts at zero."""
        block = Block(units, self.n_units)
        head = nn.Linear(int(units), READOUT_HIDDEN, bias=False)
        nn.init.zeros_(head.weight)
        self.blocks.append(block)
        self.head_in.append(head)

    def states(self, ticks: torch.Tensor) -> list[torch.Tensor]:
        if ticks.dim() != 3 or ticks.shape[-1] != INPUT_DIM:
            raise ValueError(f"ticks must be (B, T, {INPUT_DIM}); got {tuple(ticks.shape)}")
        batch, horizon, _ = ticks.shape
        valid = valid_steps(ticks)
        blocks = self.block_list()
        hs = [ticks.new_zeros(batch, b.units) for b in blocks]
        for t in range(horizon):
            x = ticks[:, t]
            v = valid[:, t : t + 1]
            new: list[torch.Tensor] = []
            for k, block in enumerate(blocks):
                inp = x if k == 0 else torch.cat([x, *new], dim=-1)
                h = block.cell(inp, hs[k]) * block.alive
                new.append(v * h + (1.0 - v) * hs[k])
            hs = new
        return hs

    def forward(self, ticks: torch.Tensor) -> torch.Tensor:
        hs = self.states(ticks)
        batch, horizon, _ = ticks.shape
        valid = valid_steps(ticks)
        last = (valid * torch.arange(horizon, dtype=valid.dtype)).argmax(-1)
        x_last = ticks[torch.arange(batch), last]
        pre: torch.Tensor | None = None
        for block, head, h in zip(self.block_list(), self.head_list(), hs, strict=True):
            out = h * torch.sigmoid(block.read_gate(x_last))
            term = cast(torch.Tensor, head(out))
            pre = term if pre is None else pre + term
        assert pre is not None
        return cast(torch.Tensor, self.head_out(torch.tanh(pre)))


class RTSubject(nn.Module):
    """Task-routed subject: ``forward(ticks, task)``; one column per task."""

    def __init__(self, tasks: dict[str, int], units: int, seed: int) -> None:
        super().__init__()
        self.columns = nn.ModuleDict()
        for i, (name, out_dim) in enumerate(tasks.items()):
            self.add_task_(name, out_dim, units, seed=seed + 101 * i)

    # ------------------------------------------------------------------ structure
    def column(self, task: str) -> Column:
        if task not in self.columns:
            raise KeyError(f"unknown task {task!r}; routes: {sorted(self.columns.keys())}")
        return cast(Column, self.columns[task])

    def add_task_(self, name: str, out_dim: int, units: int, *, seed: int) -> None:
        if name in self.columns:
            raise ValueError(f"task {name!r} already routed")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.columns[name] = Column(units, out_dim)

    def grow_(self, task: str, units: int, *, seed: int) -> None:
        col = self.column(task)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            col.grow_(units)

    def set_dormant_(self, task: str, block: int, units: list[int], dormant: bool) -> None:
        b = self.column(task).block_list()[block]
        b.alive[torch.as_tensor(units, dtype=torch.long)] = 0.0 if dormant else 1.0

    def task_parameters(self, task: str) -> list[nn.Parameter]:
        return list(self.column(task).parameters())

    def forward(self, ticks: torch.Tensor, task: str) -> torch.Tensor:
        return cast(torch.Tensor, self.column(task)(ticks))

    # ------------------------------------------------------------------- capsule
    def spec(self) -> dict[str, Any]:
        cols: dict[str, Any] = {}
        for name in self.columns:
            col = self.column(name)
            cols[name] = {"out_dim": col.out_dim, "blocks": [b.units for b in col.block_list()]}
        return {"arch_version": ARCH_VERSION, "scope": SCOPE, "columns": cols}

    def capsule(self) -> dict[str, Any]:
        state = {k: v.detach().clone() for k, v in self.state_dict().items()}
        return {"spec": self.spec(), "state": state}

    @classmethod
    def from_capsule(cls, capsule: dict[str, Any]) -> RTSubject:
        spec = capsule["spec"]
        if spec.get("arch_version") != ARCH_VERSION:
            raise ValueError(f"capsule arch {spec.get('arch_version')!r} != {ARCH_VERSION}")
        model = cls({}, 1, 0)
        for name, col in spec["columns"].items():
            blocks = [int(u) for u in col["blocks"]]
            model.add_task_(name, int(col["out_dim"]), blocks[0], seed=0)
            for units in blocks[1:]:
                model.grow_(name, units, seed=0)
        model.load_state_dict(capsule["state"], strict=True)
        return model

    def n_params(self, task: str | None = None) -> int:
        params = self.task_parameters(task) if task is not None else list(self.parameters())
        return sum(p.numel() for p in params)
