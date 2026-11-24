"""Lifecycle operations for the RT backbone: ``taiji-gru-rt-v2`` (D-20261124-496).

Release-01 requires the full lifecycle of the core requirements on the N32 backbone:
inheritance birth, differentiation, integration, reversible dormancy and retention-verified
retirement / recycling.  ``LifecycleRTSubject`` subclasses the frozen ``taiji-gru-rt-v1``
``RTSubject`` without touching its forward pass or its RNG use, so for the same seed and
no lifecycle operation it is bitwise identical to v1 (tested) and the N32.0 evidence
obtained on v1 carries over.

Per-unit stage buffer ``stage`` in every block, codes in ``STAGES``; a unit is alive iff
its stage is immature or mature.  All structural operations rebuild modules, so callers
must rebuild their optimisers afterwards.

  differentiate_  Net2Net-style split of one unit inside a block: the child copies the
                  parent's incoming parameters; every outgoing weight is split
                  ratio / (1 - ratio) (ratio != 0.5 breaks the symmetry).  Function
                  preserved up to float rounding.
  integrate_      merge ``drop`` into ``keep`` (outgoing weights summed, ``drop`` removed);
                  with a probe it is rolled back when the output changes by more than tol.
  retire_         take units offline, score with a caller-supplied retention check, commit
                  only if no score drops by more than ``max_drop``; otherwise roll back.
  recycle_        physically remove retired units (their state is identically zero).
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any, cast

import torch
from torch import nn

from taiji.gru_rt import INPUT_DIM, Block, Column, RTSubject

ARCH_VERSION_V2 = "taiji-gru-rt-v2"
STAGES = ("proposed", "immature", "mature", "dormant", "retired")
STAGE_ID = {name: i for i, name in enumerate(STAGES)}
ALIVE = (STAGE_ID["immature"], STAGE_ID["mature"])
ScoreFn = Callable[["LifecycleRTSubject"], dict[str, float]]


def stage_of(block: Block) -> torch.Tensor:
    return block.get_buffer("stage")


def _set_stage(block: Block, name: str) -> None:
    block.register_buffer("stage", torch.full((block.units,), STAGE_ID[name], dtype=torch.long))


def _gru_cell(
    in_dim: int, w_ih: torch.Tensor, w_hh: torch.Tensor, b_ih: torch.Tensor, b_hh: torch.Tensor
) -> nn.GRUCell:
    with torch.random.fork_rng(devices=[]):
        cell = nn.GRUCell(in_dim, w_hh.shape[1])
    with torch.no_grad():
        cell.weight_ih.copy_(w_ih)
        cell.weight_hh.copy_(w_hh)
        cast(torch.Tensor, cell.bias_ih).copy_(b_ih)
        cast(torch.Tensor, cell.bias_hh).copy_(b_hh)
    return cell


def _linear(weight: torch.Tensor, bias: torch.Tensor | None) -> nn.Linear:
    with torch.random.fork_rng(devices=[]):
        lin = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None)
    with torch.no_grad():
        lin.weight.copy_(weight)
        if bias is not None:
            cast(torch.Tensor, lin.bias).copy_(bias)
    return lin


def _cell_parts(cell: nn.GRUCell) -> tuple[torch.Tensor, ...]:
    return (
        cell.weight_ih.detach().clone(),
        cell.weight_hh.detach().clone(),
        cast(torch.Tensor, cell.bias_ih).detach().clone(),
        cast(torch.Tensor, cell.bias_hh).detach().clone(),
    )


def _transform_block(
    col: Column, b: int, src: list[int], out: torch.Tensor, new_stage: torch.Tensor
) -> None:
    """New unit i takes the incoming parameters of old unit ``src[i]``; every weight that
    reads this block's states (columns = old units) becomes ``W @ out``."""
    blocks = col.block_list()
    heads = col.head_list()
    blk = blocks[b]
    u_old, u_new = blk.units, len(src)
    if out.shape != (u_old, u_new):
        raise ValueError(f"out must be ({u_old}, {u_new}); got {tuple(out.shape)}")
    idx = torch.as_tensor(src, dtype=torch.long)
    rows = torch.cat([idx + g * u_old for g in range(3)])
    w_ih, w_hh, b_ih, b_hh = _cell_parts(blk.cell)
    out = out.to(w_hh.dtype)
    with torch.no_grad():
        blk.cell = _gru_cell(
            blk.cell.input_size, w_ih[rows], w_hh[rows] @ out, b_ih[rows], b_hh[rows]
        )
        rg_bias = cast(torch.Tensor, blk.read_gate.bias).detach()
        blk.read_gate = _linear(blk.read_gate.weight.detach()[idx], rg_bias[idx])
        alive = blk.alive.detach()[idx].clone()
        head = heads[b]
        col.head_in[b] = _linear(head.weight.detach() @ out, head.bias)
        offset = INPUT_DIM + sum(x.units for x in blocks[:b])
        for later in blocks[b + 1 :]:
            l_ih, l_hh, l_bih, l_bhh = _cell_parts(later.cell)
            seg = l_ih[:, offset : offset + u_old] @ out
            new_ih = torch.cat([l_ih[:, :offset], seg, l_ih[:, offset + u_old :]], dim=1)
            later.cell = _gru_cell(new_ih.shape[1], new_ih, l_hh, l_bih, l_bhh)
            later.lateral += u_new - u_old
    blk.units = u_new
    blk.register_buffer("alive", alive)
    blk.register_buffer("stage", new_stage)


class LifecycleRTSubject(RTSubject):
    """``taiji-gru-rt-v2``: v1 forward plus per-unit lifecycle and structural operations."""

    # ------------------------------------------------------------------ structure
    def add_task_(self, name: str, out_dim: int, units: int, *, seed: int) -> None:
        super().add_task_(name, out_dim, units, seed=seed)
        for blk in self.column(name).block_list():
            _set_stage(blk, "mature")

    def grow_(self, task: str, units: int, *, seed: int) -> None:
        super().grow_(task, units, seed=seed)
        _set_stage(self.column(task).block_list()[-1], "immature")

    def _block(self, task: str, block: int) -> Block:
        return self.column(task).block_list()[block]

    def stage_names(self, task: str) -> list[list[str]]:
        return [[STAGES[int(s)] for s in stage_of(b)] for b in self.column(task).block_list()]

    def promote_(self, task: str) -> None:
        for blk in self.column(task).block_list():
            st = stage_of(blk)
            st[st == STAGE_ID["immature"]] = STAGE_ID["mature"]

    def set_dormant_(self, task: str, block: int, units: list[int], dormant: bool) -> None:
        blk = self._block(task, block)
        st = stage_of(blk)
        idx = torch.as_tensor(units, dtype=torch.long)
        if bool((st[idx] == STAGE_ID["retired"]).any()):
            raise ValueError("retired units cannot change dormancy; recycle them instead")
        st[idx] = STAGE_ID["dormant"] if dormant else STAGE_ID["mature"]
        blk.alive[idx] = 0.0 if dormant else 1.0

    # ------------------------------------------------------------------ probes
    def _probe(self, task: str, ticks: torch.Tensor) -> torch.Tensor:
        mode = self.training
        self.eval()
        with torch.no_grad():
            out = self(ticks, task).clone()
        self.train(mode)
        return cast(torch.Tensor, out)

    def _require_alive(self, blk: Block, units: list[int]) -> None:
        st = stage_of(blk)
        for u in units:
            if int(st[u]) not in ALIVE:
                raise ValueError(f"unit {u} is {STAGES[int(st[u])]}, not alive")

    # --------------------------------------------------------- differentiation
    def differentiate_(self, task: str, block: int, unit: int, ratio: float = 0.75) -> int:
        if not 0.0 < ratio < 1.0 or ratio == 0.5:
            raise ValueError("ratio must be in (0, 1) and != 0.5 (symmetry breaking)")
        blk = self._block(task, block)
        self._require_alive(blk, [unit])
        u = blk.units
        out = torch.eye(u, u + 1)
        out[unit, unit] = ratio
        out[unit, u] = 1.0 - ratio
        stage = torch.cat([stage_of(blk), torch.tensor([STAGE_ID["immature"]])])
        _transform_block(self.column(task), block, [*range(u), unit], out, stage)
        return u

    # -------------------------------------------------------------- integration
    def integrate_(
        self,
        task: str,
        block: int,
        keep: int,
        drop: int,
        *,
        probe: torch.Tensor | None = None,
        tol: float = 1e-5,
    ) -> dict[str, Any]:
        blk = self._block(task, block)
        if keep == drop:
            raise ValueError("keep and drop must differ")
        self._require_alive(blk, [keep, drop])
        saved = copy.deepcopy(self.column(task))
        before = self._probe(task, probe) if probe is not None else None
        u = blk.units
        kept = [i for i in range(u) if i != drop]
        out = torch.zeros(u, u - 1)
        for pos, i in enumerate(kept):
            out[i, pos] = 1.0
        out[drop, kept.index(keep)] = 1.0
        _transform_block(self.column(task), block, kept, out, stage_of(blk)[kept].clone())
        err = None
        if before is not None and probe is not None:
            err = float((self._probe(task, probe) - before).abs().max())
            if err > tol:
                self.columns[task] = saved
                return {"committed": False, "error": err}
        return {"committed": True, "error": err}

    # ------------------------------------------------------ retirement / recycling
    def retire_(
        self, task: str, block: int, units: list[int], score_fn: ScoreFn, max_drop: float = 0.02
    ) -> dict[str, Any]:
        blk = self._block(task, block)
        st = stage_of(blk)
        for u in units:
            if int(st[u]) not in (*ALIVE, STAGE_ID["dormant"]):
                raise ValueError(f"unit {u} is {STAGES[int(st[u])]}; cannot retire")
        idx = torch.as_tensor(units, dtype=torch.long)
        mode = self.training
        before = score_fn(self)
        saved_alive = blk.alive.detach().clone()
        blk.alive[idx] = 0.0
        after = score_fn(self)
        self.train(mode)
        drops = {k: before[k] - after[k] for k in before}
        if any(d > max_drop for d in drops.values()):
            blk.alive.copy_(saved_alive)
            return {"committed": False, "before": before, "after": after, "drops": drops}
        stage_of(blk)[idx] = STAGE_ID["retired"]
        return {"committed": True, "before": before, "after": after, "drops": drops}

    def recycle_(self, task: str, block: int) -> int:
        blk = self._block(task, block)
        st = stage_of(blk)
        retired = [i for i in range(blk.units) if int(st[i]) == STAGE_ID["retired"]]
        if not retired:
            return 0
        kept = [i for i in range(blk.units) if i not in retired]
        if not kept:
            raise ValueError("a block must keep at least one unit")
        out = torch.zeros(blk.units, len(kept))
        for pos, i in enumerate(kept):
            out[i, pos] = 1.0
        _transform_block(self.column(task), block, kept, out, st[kept].clone())
        return len(retired)

    # ------------------------------------------------------------------- capsule
    def spec(self) -> dict[str, Any]:
        spec = super().spec()
        spec["arch_version"] = ARCH_VERSION_V2
        spec["stages"] = list(STAGES)
        return spec

    @classmethod
    def from_capsule(cls, capsule: dict[str, Any]) -> LifecycleRTSubject:
        spec = capsule["spec"]
        if spec.get("arch_version") != ARCH_VERSION_V2:
            raise ValueError(f"capsule arch {spec.get('arch_version')!r} != {ARCH_VERSION_V2}")
        model = cls({}, 1, 0)
        for name, col in spec["columns"].items():
            blocks = [int(u) for u in col["blocks"]]
            model.add_task_(name, int(col["out_dim"]), blocks[0], seed=0)
            for units in blocks[1:]:
                model.grow_(name, units, seed=0)
        model.load_state_dict(capsule["state"], strict=True)
        return model
