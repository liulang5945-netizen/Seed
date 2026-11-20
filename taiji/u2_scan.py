"""Batch × time separated scan for the unit graph (new module; core untouched).

``UnitScan.run`` runs ``T`` ticks for each of ``B`` independent samples, updating the
unit state ``{z,a}`` with the same equations as ``NeuralUnits.forward`` but keeping a
separate state per sample (the core forward treats the leading dimension as time and
carries one shared state, so it cannot express per-sample recurrence).

See plans/reference/DESIGN-TAIJI_U2_SCAN_INTERFACE_20261120.md.
"""

from __future__ import annotations

from typing import Any

import torch

from taiji.u2_neural_units import LEGAL_INPUT_CHANNELS


class UnitScan:
    """Per-sample multi-tick scan over an existing RouteBGraph (read-only)."""

    def __init__(self, graph: Any) -> None:
        self.graph = graph
        self.units = graph.units

    def _encode_tick(self, channels: dict[str, torch.Tensor]) -> torch.Tensor:
        columns = []
        for channel in LEGAL_INPUT_CHANNELS:
            columns.append(
                self.graph.input_encoders[channel](channels[channel]) @ self.graph.input_projection
            )
        return torch.stack(columns, dim=1)  # [B,G,D]

    def run(
        self,
        channels: dict[str, torch.Tensor],
        gates: torch.Tensor,
        state: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        batch, ticks = gates.shape[0], gates.shape[1]
        units = self.units
        if state is None:
            initial = self.graph.initial_state()
            z = initial["z"].unsqueeze(0).expand(batch, -1).clone()
            a = initial["a"].unsqueeze(0).expand(batch, -1).clone()
        else:
            z = state["z"]
            a = state["a"]
        weights = (units.w * units.edge_mask).T
        group_index = units.group_index
        activities = []
        readouts = []
        for tick in range(ticks):
            tick_channels = {name: channels[name][:, tick] for name in LEGAL_INPUT_CHANNELS}
            inputs = self._encode_tick(tick_channels)  # [B,G,D]
            x = inputs[:, group_index, :]  # [B,N,D]
            drive = (x * units.r.unsqueeze(0)).sum(-1)  # [B,N]
            gate = gates[:, tick]
            incoming = (gate * z) @ weights  # [B,N]
            v = drive + incoming - units.b.unsqueeze(0) * a
            z = units.c.unsqueeze(0) * z + (1.0 - units.c.unsqueeze(0)) * torch.tanh(v)
            a = (1.0 - units.eta.unsqueeze(0)) * a + units.eta.unsqueeze(0) * z
            activity = gate * z
            state_view = self.graph.state_mix[0] * activity + self.graph.state_mix[1] * (a * gate)
            projected = state_view @ units.p.T  # [B,hidden]
            direct = projected @ self.graph.query.T  # [B,query]
            readouts.append(self.graph.readout(direct))
            activities.append(activity)
        return {
            "activity": torch.stack(activities, dim=1),  # [B,T,N]
            "readout": torch.stack(readouts, dim=1),  # [B,T,out]
            "state": {"z": z, "a": a},
        }
