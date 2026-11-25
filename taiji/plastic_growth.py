"""Task-agnostic plastic & growing neural network for N33.3 (`taiji-plastic-growth-v1`).

Implements all core-requirement lifecycle & continual plasticity primitives without external
``task_id`` arguments:
  * Inheritance/neutral birth (`grow_neutral_`): adds ``immature`` units with zero outgoing
    weights so pre-training forward output changes by identically 0.0.
  * Net2Net differentiation (`differentiate_`): splits a unit across incoming weights and
    proportional outgoing columns, preserving function up to float rounding (<= 1e-6).
  * Reversible dormancy (`set_dormant_` / `wake_`) & retention-checked retirement (`retire_`).
  * Physical recycling (`recycle_`) of retired units without perturbing live forward outputs.
  * Selective plasticity (`apply_selective_plasticity_grad_mask`): zeroes gradients on ``mature``
    unit parameters while leaving ``immature`` units plastic.
  * Conservative internal pressure controller (`GrowthPressureController`): triggers growth only
    when validation error exceeds threshold and hits a plateau; never grows without pressure.
  * Native `SessionStore` capsule serialization (`to_session_record` / `from_session_record`),
    preserving tensor dtypes (`float32` weights + `int64` lifecycle/id buffers) bit-exactly.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from taiji.session_store import SessionRecord

ARCH_VERSION = "taiji-plastic-growth-v1"
STAGES = ("immature", "mature", "dormant", "retired")
STAGE_ID = {name: i for i, name in enumerate(STAGES)}


class PlasticGrowthNet(nn.Module):
    """Single-hidden-layer tanh network supporting unit lifecycle, growth, and plasticity."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, seed: int = 0) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self._next_unit_id = self.hidden_dim

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.fc_in = nn.Linear(self.in_dim, self.hidden_dim)
            self.fc_out = nn.Linear(self.hidden_dim, self.out_dim)

        self.register_buffer(
            "stage",
            torch.full((self.hidden_dim,), STAGE_ID["immature"], dtype=torch.long),
        )
        self.register_buffer(
            "unit_ids",
            torch.arange(self.hidden_dim, dtype=torch.long),
        )

    def alive_mask(self) -> torch.Tensor:
        """Mask of active units (`immature` or `mature`), identical in train and eval."""
        alive = (self.stage == STAGE_ID["immature"]) | (self.stage == STAGE_ID["mature"])
        return alive.to(dtype=self.fc_in.weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Task-agnostic forward pass (`x` only, zero `task_id` routing)."""
        h = torch.tanh(self.fc_in(x)) * self.alive_mask()
        return self.fc_out(h)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def active_units(self) -> int:
        return int(self.alive_mask().sum().item())

    def mature_indices(self) -> list[int]:
        return [
            int(i)
            for i in torch.nonzero(self.stage == STAGE_ID["mature"], as_tuple=False).flatten()
        ]

    def immature_indices(self) -> list[int]:
        return [
            int(i)
            for i in torch.nonzero(self.stage == STAGE_ID["immature"], as_tuple=False).flatten()
        ]

    def _rebuild_layers(
        self,
        w_in: torch.Tensor,
        b_in: torch.Tensor,
        w_out: torch.Tensor,
        b_out: torch.Tensor,
        stage: torch.Tensor,
        unit_ids: torch.Tensor,
    ) -> None:
        new_h = int(w_in.shape[0])
        fc_in = nn.Linear(self.in_dim, new_h)
        fc_out = nn.Linear(new_h, self.out_dim)
        with torch.no_grad():
            fc_in.weight.copy_(w_in)
            fc_in.bias.copy_(b_in)
            fc_out.weight.copy_(w_out)
            fc_out.bias.copy_(b_out)
        self.fc_in = fc_in
        self.fc_out = fc_out
        self.hidden_dim = new_h
        self.register_buffer("stage", stage.to(dtype=torch.long))
        self.register_buffer("unit_ids", unit_ids.to(dtype=torch.long))

    def grow_neutral_(self, k: int, seed: int, init_scale: float = 1.0) -> list[int]:
        """Add ``k`` new immature units with zero outgoing weights (<= 1e-6 output perturbation)."""
        if k <= 0:
            return []
        old_h = self.hidden_dim
        bound = init_scale / math.sqrt(self.in_dim)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            new_w_in = torch.empty(k, self.in_dim).uniform_(-bound, bound)
            new_b_in = torch.empty(k, dtype=self.fc_in.bias.dtype).uniform_(-bound, bound)
        new_w_out = torch.zeros(self.out_dim, k, dtype=self.fc_out.weight.dtype)

        w_in = torch.cat([self.fc_in.weight.detach(), new_w_in], dim=0)
        b_in = torch.cat([self.fc_in.bias.detach(), new_b_in], dim=0)
        w_out = torch.cat([self.fc_out.weight.detach(), new_w_out], dim=1)
        b_out = self.fc_out.bias.detach().clone()

        new_stage = torch.cat(
            [self.stage, torch.full((k,), STAGE_ID["immature"], dtype=torch.long)], dim=0
        )
        new_ids = torch.arange(self._next_unit_id, self._next_unit_id + k, dtype=torch.long)
        self._next_unit_id += k
        self._rebuild_layers(
            w_in, b_in, w_out, b_out, new_stage, torch.cat([self.unit_ids, new_ids])
        )
        return list(range(old_h, old_h + k))

    def differentiate_(self, unit_idx: int, ratio: float = 0.35) -> int:
        """Net2Net-style split of ``unit_idx`` preserving function within float rounding."""
        if not (0 <= unit_idx < self.hidden_dim):
            raise IndexError(f"unit_idx {unit_idx} out of range")
        if not (0.0 < ratio < 1.0):
            raise ValueError(f"ratio must be in (0, 1), got {ratio}")

        child_idx = self.hidden_dim
        w_in = torch.cat(
            [self.fc_in.weight.detach(), self.fc_in.weight.detach()[unit_idx : unit_idx + 1]],
            dim=0,
        )
        b_in = torch.cat(
            [self.fc_in.bias.detach(), self.fc_in.bias.detach()[unit_idx : unit_idx + 1]],
            dim=0,
        )
        w_out_old = self.fc_out.weight.detach().clone()
        parent_col = w_out_old[:, unit_idx : unit_idx + 1].clone()
        w_out_old[:, unit_idx : unit_idx + 1] = parent_col * ratio
        w_out = torch.cat([w_out_old, parent_col * (1.0 - ratio)], dim=1)
        b_out = self.fc_out.bias.detach().clone()

        new_stage = torch.cat([self.stage, torch.tensor([STAGE_ID["immature"]], dtype=torch.long)])
        child_id = torch.tensor([self._next_unit_id], dtype=torch.long)
        self._next_unit_id += 1
        self._rebuild_layers(
            w_in, b_in, w_out, b_out, new_stage, torch.cat([self.unit_ids, child_id])
        )
        return child_idx

    def mature_(self, indices: list[int] | None = None) -> None:
        """Promote specified (or all immature) units to `mature` stage."""
        targets = self.immature_indices() if indices is None else indices
        with torch.no_grad():
            for idx in targets:
                self.stage[idx] = STAGE_ID["mature"]

    def set_dormant_(self, indices: list[int]) -> dict[int, int]:
        """Take units offline into `dormant` stage, returning previous stage codes for wake."""
        prev = {int(i): int(self.stage[i].item()) for i in indices}
        with torch.no_grad():
            for idx in indices:
                self.stage[idx] = STAGE_ID["dormant"]
        return prev

    def wake_(self, saved_stages: dict[int, int]) -> None:
        """Restore previously dormant units back to their saved lifecycle stages."""
        with torch.no_grad():
            for idx, st in saved_stages.items():
                self.stage[idx] = int(st)

    def retire_(
        self,
        indices: list[int],
        score_fn: Callable[[PlasticGrowthNet], float] | None = None,
        max_drop: float = 0.02,
    ) -> bool:
        """Mark units retired; roll back if ``score_fn`` drops more than ``max_drop``."""
        snap = self.stage.clone()
        before = score_fn(self) if score_fn is not None else None
        with torch.no_grad():
            for idx in indices:
                self.stage[idx] = STAGE_ID["retired"]
        if before is not None and score_fn is not None:
            after = score_fn(self)
            if before - after > max_drop:
                self.register_buffer("stage", snap)
                return False
        return True

    def recycle_(self) -> int:
        """Physically remove retired units without changing live forward outputs."""
        keep = self.stage != STAGE_ID["retired"]
        removed = int((~keep).sum().item())
        if removed == 0:
            return 0
        w_in = self.fc_in.weight.detach()[keep].clone()
        b_in = self.fc_in.bias.detach()[keep].clone()
        w_out = self.fc_out.weight.detach()[:, keep].clone()
        b_out = self.fc_out.bias.detach().clone()
        self._rebuild_layers(
            w_in, b_in, w_out, b_out, self.stage[keep].clone(), self.unit_ids[keep].clone()
        )
        return removed

    def apply_selective_plasticity_grad_mask(self, freeze_mature: bool = True) -> None:
        """Zero parameter gradients belonging to `mature` units, leaving `immature` plastic."""
        if not freeze_mature:
            return
        mature = self.stage == STAGE_ID["mature"]
        if not bool(mature.any()):
            return
        if self.fc_in.weight.grad is not None:
            self.fc_in.weight.grad[mature] = 0.0
        if self.fc_in.bias.grad is not None:
            self.fc_in.bias.grad[mature] = 0.0
        if self.fc_out.weight.grad is not None:
            self.fc_out.weight.grad[:, mature] = 0.0

    def to_session_record(
        self,
        subject_id: str,
        pending: list[dict[str, Any]] | None = None,
        experience: list[dict[str, Any]] | None = None,
    ) -> SessionRecord:
        """Pack parameters + lifecycle buffers into a ``SessionRecord`` capsule."""
        rec = SessionRecord(
            subject_id=subject_id,
            pending=list(pending) if pending else [],
            experience=list(experience) if experience else [],
        )
        rec.capsule = {
            "fc_in.weight": self.fc_in.weight.detach().clone(),
            "fc_in.bias": self.fc_in.bias.detach().clone(),
            "fc_out.weight": self.fc_out.weight.detach().clone(),
            "fc_out.bias": self.fc_out.bias.detach().clone(),
            "stage": self.stage.detach().clone(),
            "unit_ids": self.unit_ids.detach().clone(),
            "meta": torch.tensor(
                [self.in_dim, self.hidden_dim, self.out_dim, self._next_unit_id], dtype=torch.long
            ),
        }
        rec.append_experience(
            "capsule_snapshot",
            {
                "arch_version": ARCH_VERSION,
                "hidden_dim": self.hidden_dim,
                "active_units": self.active_units(),
            },
            source="plastic_growth",
        )
        return rec

    @classmethod
    def from_session_record(cls, record: SessionRecord) -> PlasticGrowthNet:
        """Reconstruct a ``PlasticGrowthNet`` bit-exactly from a ``SessionRecord`` capsule."""
        cap = record.capsule
        in_dim, hidden_dim, out_dim, next_uid = (int(v) for v in cap["meta"].tolist())
        net = cls(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, seed=0)
        net._next_unit_id = next_uid
        net._rebuild_layers(
            cap["fc_in.weight"],
            cap["fc_in.bias"],
            cap["fc_out.weight"],
            cap["fc_out.bias"],
            cap["stage"],
            cap["unit_ids"],
        )
        return net


@dataclass
class GrowthPressureController:
    """Conservative on-demand growth trigger: grows only on genuine plateau pressure."""

    window: int = 3
    plateau_eps: float = 0.015
    error_threshold: float = 0.15
    max_growth_events: int = 1
    grow_k: int = 24
    history: list[float] = field(default_factory=list)
    growth_events: int = 0

    def observe(self, val_error: float) -> int:
        self.history.append(float(val_error))
        if self.growth_events >= self.max_growth_events:
            return 0
        if len(self.history) < self.window or val_error <= self.error_threshold:
            return 0
        recent = self.history[-self.window :]
        improvement = recent[0] - recent[-1]
        if improvement < self.plateau_eps:
            self.growth_events += 1
            self.history.clear()
            return self.grow_k
        return 0
