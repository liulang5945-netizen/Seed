"""Run the bounded M4.V2.R3 main-path structural bridge canary.

R3 is a wiring and causality gate, not a promotion experiment.  It attaches
one zero-gated adaptive population to the native predictive path, verifies
that the closed candidate is equivalent to its parent, then opens it for one
small local-credit probe while freezing the mature F1 owners.  The report
always keeps ``can_promote=false``.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji import (  # noqa: E402
    ADAPTIVE_RESIDUAL_BRIDGE_FORMAT,
    ADAPTIVE_RESIDUAL_BRIDGE_VERSION,
    Taiji,
    TaijiConfig,
)
from taiji.internalization import content_digest  # noqa: E402

CANARY_FORMAT = "taiji-m4v2-r3-bridge-canary-v1"
CANARY_VERSION = 1
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m4v2_r3_bridge_canary_20260909.json"


def _tiny_config() -> TaijiConfig:
    return TaijiConfig(
        region_sizes=(24,),
        synapse_fan_in=6,
        motor_fan_in=12,
        predictive_context_fan_in=6,
        memory_units=24,
        memory_fan_in=6,
        memory_meta_dim=16,
        memory_readout_fan_in=12,
        identity_organ_enabled=False,
        seed=71,
    )


def _parent_checkpoint() -> tuple[dict[str, Any], bool]:
    """Create the R2 parent and verify checkpoint recovery before the probe."""

    model = Taiji(_tiny_config(), episode_id="r3-canary-parent")
    checkpoint = model.checkpoint()
    restored = Taiji.from_checkpoint(copy.deepcopy(checkpoint))
    return checkpoint, content_digest(restored.checkpoint()) == content_digest(checkpoint)


def _predictive_probe(model: Taiji, symbol: int) -> torch.Tensor:
    model.reset_dynamics(episode_id="r3-canary-probe")
    return model.observe(symbol, learn=False, readout="predictive").probabilities.detach().clone()


def run_canary(parent_checkpoint: dict[str, Any] | None = None) -> dict[str, Any]:
    if parent_checkpoint is None:
        parent, checkpoint_preflight = _parent_checkpoint()
    else:
        parent = copy.deepcopy(parent_checkpoint)
        restored = Taiji.from_checkpoint(copy.deepcopy(parent))
        checkpoint_preflight = content_digest(restored.checkpoint()) == content_digest(parent)
    if Taiji.ADAPTIVE_RESIDUAL_BRIDGE_KEY in parent:
        raise ValueError("R3 canary parent must not already contain an adaptive residual bridge")

    parent_digest = content_digest(parent)
    parent_model = Taiji.from_checkpoint(copy.deepcopy(parent))
    zero_gate = Taiji.from_checkpoint(copy.deepcopy(parent))
    zero_metadata = zero_gate.enable_adaptive_residual_bridge(gate=0.0)
    zero_before = content_digest(zero_gate.adaptive_residual_bridge.to_payload())
    parent_output = parent_model.generate(b"ab", 8)
    zero_output = zero_gate.generate(b"ab", 8)
    zero_after = content_digest(zero_gate.adaptive_residual_bridge.to_payload())

    baseline = Taiji.from_checkpoint(copy.deepcopy(parent))
    active = Taiji.from_checkpoint(copy.deepcopy(parent))
    active_metadata = active.enable_adaptive_residual_bridge(gate=1.0, residual_gain=1.0)
    baseline_probabilities = _predictive_probe(baseline, 97)
    active.reset_dynamics(episode_id="r3-canary-probe")
    active_step = active.observe(97, learn=False, readout="predictive")
    active_probabilities = active_step.probabilities.detach().clone()
    residual_activity = active.adaptive_residual_bridge.region.activity.detach().clone()

    context_before = content_digest(active.predictive_context.to_payload())
    readout_before = content_digest(active.predictive_readout.to_payload())
    bridge_before = content_digest(active.adaptive_residual_bridge.to_payload())
    active.observe(
        98,
        learn=True,
        learn_fabric=False,
        learn_predictive_context=False,
        learn_predictive_readout=False,
        learn_adaptive_residual_bridge=True,
        readout="predictive",
    )
    bridge_after = content_digest(active.adaptive_residual_bridge.to_payload())
    active_checkpoint = active.checkpoint()
    active_checkpoint_digest = content_digest(active_checkpoint)
    restored = Taiji.from_checkpoint(copy.deepcopy(active_checkpoint))

    no_bridge = Taiji.from_checkpoint(copy.deepcopy(active_checkpoint))
    no_bridge.disable_adaptive_residual_bridge()
    lesioned = Taiji.from_checkpoint(copy.deepcopy(active_checkpoint))
    lesioned.lesion_adaptive_residual_bridge()
    no_bridge_output = _predictive_probe(no_bridge, 97)
    lesioned_output = _predictive_probe(lesioned, 97)

    rollback = Taiji.from_checkpoint(copy.deepcopy(parent))
    rollback_output = rollback.generate(b"ab", 8)

    old_owner_unchanged = context_before == content_digest(
        parent["predictive_context"]
    ) and readout_before == content_digest(parent["predictive_readout"])
    gates = {
        "checkpoint_preflight": checkpoint_preflight,
        "parent_has_no_bridge": Taiji.ADAPTIVE_RESIDUAL_BRIDGE_KEY not in parent,
        "bridge_payload_contract": (
            zero_metadata["format"] == ADAPTIVE_RESIDUAL_BRIDGE_FORMAT
            and zero_metadata["version"] == ADAPTIVE_RESIDUAL_BRIDGE_VERSION
            and active_metadata["unit_count"] > 0
            and active_metadata["edge_count"] > 0
        ),
        "gate_zero_output_equivalence": zero_output == parent_output,
        "gate_zero_no_state_mutation": zero_before == zero_after,
        "active_residual_activity": bool(residual_activity.abs().any()),
        "active_output_delta": not torch.allclose(
            baseline_probabilities,
            active_probabilities,
        ),
        "active_local_credit": bridge_before != bridge_after,
        "mature_f1_owners_unchanged": old_owner_unchanged,
        "fresh_restore_exact": (
            restored.adaptive_residual_bridge_enabled
            and content_digest(restored.checkpoint()) == active_checkpoint_digest
        ),
        "lesion_equivalence": torch.equal(no_bridge_output, lesioned_output),
        "rollback_matches_parent": rollback_output == parent_output,
    }
    return {
        "format": CANARY_FORMAT,
        "version": CANARY_VERSION,
        "status": "passed" if all(gates.values()) else "failed",
        "can_promote": False,
        "promotion_reason": "R3 is a single CPU wiring canary; structural promotion remains prohibited",
        "parent": {
            "checkpoint_digest": parent_digest,
            "owner_graph": "pre-adaptive-residual-bridge",
            "bridge_key": Taiji.ADAPTIVE_RESIDUAL_BRIDGE_KEY,
        },
        "zero_gate": {
            "metadata": zero_metadata,
            "parent_output_hex": parent_output.hex(),
            "candidate_output_hex": zero_output.hex(),
            "bridge_digest_before": zero_before,
            "bridge_digest_after": zero_after,
        },
        "active_probe": {
            "metadata": active_metadata,
            "residual_activity_l1": float(residual_activity.abs().sum().item()),
            "probability_max_delta": float(
                (active_probabilities - baseline_probabilities).abs().max().item()
            ),
            "bridge_digest_before_credit": bridge_before,
            "bridge_digest_after_credit": bridge_after,
            "checkpoint_digest": active_checkpoint_digest,
        },
        "gates": gates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = run_canary()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
