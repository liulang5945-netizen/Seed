"""GRU72：N31.4 终态（REPLACE-STANDARD）对应的标准门控骨干候选。

每个「计算单位」对应一个 GRU 隐单元槽位，保留单位身份（槽位索引）、持续活动状态、
生命周期掩码语义（休眠/退役槽位状态置零且不参与读出），末步乘性 read_gate 作为
标准组件（登记表 ADOPT）。

本模块只做 N31.1.02 所需的骨干实现与级1 不变量单测；尚未进入正式训练配置。
零 check/final。
"""

from __future__ import annotations

import torch
from torch import nn


class GRU72Backbone(nn.Module):
    """72 个 GRU 槽位 + 末步乘性 read_gate。

    Args:
        input_dim: 每 tick 输入维度（当前任务生成器为 48）。
        hidden_dim: 槽位隐状态宽度（每槽位一维；默认 1 -> 状态 72 维）。
        n_units: 槽位数（默认 72）。
        gate_hidden: read_gate 的隐藏层宽度。
    """

    def __init__(
        self,
        input_dim: int = 48,
        hidden_dim: int = 1,
        n_units: int = 72,
        gate_hidden: int = 32,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_units = int(n_units)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.gru = nn.GRU(self.input_dim, self.n_units * self.hidden_dim, batch_first=True)
            # per-unit shared gate: dormant masking stays local to its own slot
            self.read_gate = nn.Sequential(
                nn.Linear(self.hidden_dim, gate_hidden),
                nn.Tanh(),
                nn.Linear(gate_hidden, self.hidden_dim),
            )
        self.register_buffer("lifecycle_mask", torch.ones(self.n_units), persistent=True)

    # ---- lifecycle -------------------------------------------------------
    def set_lifecycle_(self, indices, active: bool) -> None:
        """dormant/retired 槽位: 状态读出置零（掩码可逆）。"""
        idx = torch.as_tensor(list(indices), dtype=torch.long)
        self.lifecycle_mask[idx] = 1.0 if active else 0.0

    def active_units(self) -> int:
        return int((self.lifecycle_mask > 0).sum().item())

    # ---- forward ---------------------------------------------------------
    def forward(
        self, ticks: torch.Tensor, step_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """ticks: (B, T, input_dim)。返回 (state, activity)，各 (B, n_units*hidden_dim)。

        step_mask: (B, T) 可选；被 mask 的 tick 不推进状态（用上一 tick 状态）。
        """
        if ticks.dim() != 3 or ticks.shape[-1] != self.input_dim:
            raise ValueError(f"ticks must be (B,T,{self.input_dim}), got {tuple(ticks.shape)}")
        out, _ = self.gru(ticks)  # (B,T,D)
        if step_mask is not None:
            keep = step_mask.to(out.dtype)  # (B,T)
            out = out * keep.unsqueeze(-1)
        # 持续状态：最后一个有效 tick 的隐状态
        state = out[:, -1, :]
        # 活动量：相邻 tick 状态差的绝对值（末步）
        if out.shape[1] >= 2:
            activity = (out[:, -1, :] - out[:, -2, :]).abs()
        else:
            activity = torch.zeros_like(state)
        mask = self.lifecycle_mask.repeat_interleave(self.hidden_dim).to(state.dtype)
        state = state * mask
        b, d = state.shape
        per_unit = state.view(b, self.n_units, self.hidden_dim)
        gate = torch.sigmoid(self.read_gate(per_unit)).reshape(b, d)
        return state * gate, activity * mask

    # ---- level-1 invariants ---------------------------------------------
    def eval_deterministic(self, ticks: torch.Tensor, reps: int = 2) -> bool:
        self.eval()
        with torch.no_grad():
            first = self.forward(ticks)[0]
            for _ in range(reps - 1):
                if not torch.equal(first, self.forward(ticks)[0]):
                    return False
        return True

    def batch_independent(self, ticks: torch.Tensor, atol: float = 1e-6) -> bool:
        self.eval()
        with torch.no_grad():
            single = self.forward(ticks[:1])[0]
            batched = self.forward(ticks)[0][:1]
        return bool(torch.allclose(single, batched, atol=atol))

    def read_gate_effect(self, ticks: torch.Tensor) -> float:
        """关闭 read_gate（乘以 0）后输出差的绝对均值，用于不变量06/07 冒烟。"""
        self.eval()
        with torch.no_grad():
            on = self.forward(ticks)[0]
            zeros = torch.zeros_like(on)
            return float((on - zeros).abs().mean())
