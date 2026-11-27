"""Autonomous Shadow Growth Controller for `TaijiNativeCognitiveBackbone`.

Combines `GrowthPressureController` (pressure-triggered proposal, never grows
without pressure or above budget) with isolated N37-style shadow verification
(neutral birth check + selective plasticity adaptation + retention <= 0.02 &
net-gain gate + atomic commit or lossless rollback).
"""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from taiji.native_cognitive_backbone import TaijiNativeCognitiveBackbone
from taiji.plastic_growth import GrowthPressureController

SCHEMA = "taiji-native-growth-controller-v1"


@dataclass
class NativeGrowthPolicy:
    auto_grow_enabled: bool = True
    max_params: int = 40_000_000
    max_layers: int = 16
    max_group_units: int = 2048
    max_fast_heads: int = 8
    max_retention_drop: float = 0.02
    min_gain: float = 0.001
    delta_units: int = 32
    delta_fast_heads: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NativeGrowthPolicy:
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


def save_growth_policy(path: str | Path, policy: NativeGrowthPolicy) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(policy.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(tmp, p)


def load_growth_policy(path: str | Path) -> NativeGrowthPolicy:
    p = Path(path)
    if not p.exists():
        return NativeGrowthPolicy()
    try:
        return NativeGrowthPolicy.from_dict(json.loads(p.read_text(encoding="utf-8")))
    except Exception:
        return NativeGrowthPolicy()


def save_native_backbone_status(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def load_native_backbone_status(path: str | Path) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return dict(json.loads(p.read_text(encoding="utf-8")))
    except Exception:
        return None


class NativeGrowthController:
    """Autonomous pressure-driven + shadow-verified growth controller."""

    def __init__(
        self,
        policy: NativeGrowthPolicy | None = None,
        *,
        error_threshold: float = 0.25,
        patience: int = 3,
        min_delta: float = 0.005,
        cooldown: int = 2,
    ) -> None:
        self.policy = policy or NativeGrowthPolicy()
        self.pressure = GrowthPressureController(
            window=patience,
            plateau_eps=min_delta,
            error_threshold=error_threshold,
            max_growth_events=max(1, cooldown),
        )
        self.history: list[dict[str, Any]] = []

    def propose_action(
        self,
        model: TaijiNativeCognitiveBackbone,
        val_loss: float,
        step: int,
    ) -> dict[str, Any]:
        """Decide whether to propose growth based on pressure and budget caps."""
        if not self.policy.auto_grow_enabled:
            return {"action": "no_grow", "reason": "auto_grow_disabled", "step": int(step)}

        should_grow = self.pressure.observe(float(val_loss)) > 0
        if not should_grow:
            return {"action": "no_grow", "reason": "no_pressure", "step": int(step)}

        cur_params = model.count_parameters()
        if cur_params >= self.policy.max_params:
            return {"action": "no_grow", "reason": "max_params_reached", "step": int(step)}

        topo = model.topology_summary()
        max_u = max(topo["group_units_per_layer"])
        max_h = max(topo["fast_heads_per_layer"])
        n_layers = int(topo["n_layers"])

        if max_u + self.policy.delta_units <= self.policy.max_group_units:
            return {
                "action": "grow_group_units",
                "delta_units": self.policy.delta_units,
                "reason": "pressure_plateau",
                "step": int(step),
            }
        if max_h + self.policy.delta_fast_heads <= self.policy.max_fast_heads:
            return {
                "action": "grow_fast_heads",
                "delta_heads": self.policy.delta_fast_heads,
                "reason": "pressure_plateau",
                "step": int(step),
            }
        if n_layers + 1 <= self.policy.max_layers:
            return {
                "action": "grow_depth",
                "reason": "pressure_plateau",
                "step": int(step),
            }
        return {"action": "no_grow", "reason": "all_topology_caps_reached", "step": int(step)}

    def run_shadow_growth_cycle(
        self,
        model: TaijiNativeCognitiveBackbone,
        proposal: dict[str, Any],
        *,
        adapt_fn: Callable[[TaijiNativeCognitiveBackbone], None],
        eval_old_fn: Callable[[TaijiNativeCognitiveBackbone], float],
        eval_new_fn: Callable[[TaijiNativeCognitiveBackbone], float],
        sample_input: torch.Tensor | None = None,
        seed: int = 0,
    ) -> tuple[TaijiNativeCognitiveBackbone, dict[str, Any]]:
        """Execute neutral growth in an isolated shadow copy and commit iff gates pass."""
        action = str(proposal.get("action", "no_grow"))
        if action == "no_grow":
            return model, {
                "schema": SCHEMA,
                "action": action,
                "accepted": False,
                "reason": str(proposal.get("reason", "no_grow")),
            }

        old_before = float(eval_old_fn(model))
        new_before = float(eval_new_fn(model))

        shadow = copy.deepcopy(model)
        before_logits: torch.Tensor | None = None
        if sample_input is not None:
            shadow.eval()
            with torch.no_grad():
                before_logits, _, _ = shadow(sample_input)

        if action == "grow_group_units":
            event = shadow.grow_group_units_neutral_(
                delta_units=int(proposal.get("delta_units", self.policy.delta_units)), seed=seed
            )
        elif action == "grow_fast_heads":
            event = shadow.grow_fast_heads_neutral_(
                delta_heads=int(proposal.get("delta_heads", self.policy.delta_fast_heads)),
                seed=seed,
            )
        elif action == "grow_depth":
            event = shadow.grow_depth_neutral_(seed=seed)
        else:
            return model, {
                "schema": SCHEMA,
                "action": action,
                "accepted": False,
                "reason": f"unknown_action:{action}",
            }

        birth_max_abs_diff = 0.0
        if sample_input is not None and before_logits is not None:
            shadow.eval()
            with torch.no_grad():
                after_logits, _, _ = shadow(sample_input)
            birth_max_abs_diff = float((after_logits - before_logits).abs().max().item())

        shadow.mark_all_mature_()
        shadow.train()
        adapt_fn(shadow)
        shadow.eval()

        all_finite = all(bool(torch.isfinite(p).all()) for p in shadow.parameters())
        old_after = float(eval_old_fn(shadow))
        new_after = float(eval_new_fn(shadow))
        retention_drop = round(max(0.0, old_before - old_after), 6)
        gain = round(new_after - new_before, 6)

        accepted = False
        reject_reason = ""
        if not all_finite:
            reject_reason = "non_finite_parameters"
        elif birth_max_abs_diff > 1e-4:
            reject_reason = f"non_neutral_birth:{birth_max_abs_diff:.6f}"
        elif retention_drop > self.policy.max_retention_drop:
            reject_reason = (
                f"retention_breach:{retention_drop:.4f}>{self.policy.max_retention_drop:.4f}"
            )
        elif gain < self.policy.min_gain:
            reject_reason = f"insufficient_gain:{gain:.4f}<{self.policy.min_gain:.4f}"
        else:
            accepted = True

        receipt = {
            "schema": SCHEMA,
            "ts_utc": datetime.now(UTC).isoformat(),
            "action": action,
            "accepted": accepted,
            "reject_reason": reject_reason,
            "birth_max_abs_diff": birth_max_abs_diff,
            "retention_drop": retention_drop,
            "gain": gain,
            "growth_event": event,
        }
        self.history.append(receipt)
        return (shadow if accepted else model), receipt
