"""Run the CPU canary for the M4.R1 protected-readout consolidation candidate.

The canary is intentionally small and does not use a provider, terminal,
MCP, or a real workspace.  It verifies that the candidate is an opt-in local
update on an active predictive readout, while the protected readout, shared
fabric, and predictive context remain unchanged and checkpointable.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji import (  # noqa: E402
    Taiji,
    TaijiConfig,
    WorkbenchBoundaryAuthorization,
    WorkbenchTaskBoundary,
)
from taiji.internalization import content_digest  # noqa: E402

FORMAT = "taiji-m4r1-consolidation-canary-v1"
REPORT_DEFAULT = PROJECT_ROOT / "reports" / "taiji_m4r1_consolidation_canary_20260908.json"


def _config() -> TaijiConfig:
    return TaijiConfig(
        region_sizes=(32,),
        synapse_fan_in=8,
        motor_fan_in=16,
        memory_units=32,
        memory_fan_in=8,
        memory_readout_fan_in=16,
        memory_meta_dim=16,
        seed=71,
    )


def _boundary() -> WorkbenchTaskBoundary:
    return WorkbenchTaskBoundary.issue(
        project_id="project:seed",
        task_id="task:m4r1-canary",
        session_id="session:m4r1-canary",
        language_id="python",
        capability_snapshot_id="capability:snapshot:1",
        capability_ids=("workspace.read",),
        generation_scope="active",
        issued_tick=10,
        ttl_ticks=20,
    )


def _authorization(boundary: WorkbenchTaskBoundary) -> WorkbenchBoundaryAuthorization:
    return WorkbenchBoundaryAuthorization(
        project_id=boundary.project_id,
        task_id=boundary.task_id,
        session_id=boundary.session_id,
        capability_snapshot_id=boundary.capability_snapshot_id,
        authorized_capability_ids=("workspace.read",),
        active_boundary_digest=boundary.token_digest,
        current_tick=12,
        usage="execute",
    )


def evaluate() -> dict[str, object]:
    old_data = b"old-old-old"
    new_data = b"new-new-new"
    source = Taiji(_config(), episode_id="m4r1-source")
    source.learn_bytes(
        old_data,
        epochs=2,
        learn_fabric=False,
        learn_predictive_context=False,
    )
    source_checkpoint = source.checkpoint()
    source_digest = content_digest(source_checkpoint)
    active = _boundary()
    authorization = _authorization(active)

    baseline = Taiji.from_checkpoint(copy.deepcopy(source_checkpoint))
    zero_strength = Taiji.from_checkpoint(copy.deepcopy(source_checkpoint))
    for model in (baseline, zero_strength):
        model.clone_protected_predictive_readout_as_active(boundary_digest=active.token_digest)
    baseline.learn_bytes(
        new_data,
        learn_fabric=False,
        learn_predictive_context=False,
        boundary=active,
        authorization=authorization,
    )
    zero_strength.learn_bytes(
        new_data,
        learn_fabric=False,
        learn_predictive_context=False,
        consolidation_strength=0.0,
        boundary=active,
        authorization=authorization,
    )

    positive = Taiji.from_checkpoint(copy.deepcopy(source_checkpoint))
    positive.clone_protected_predictive_readout_as_active(boundary_digest=active.token_digest)
    protected_before = positive.readout_registry_status()["protected"]["readout_digest"]
    active_before = positive.active_predictive_readout_metadata
    if active_before is None:
        raise RuntimeError("M4.R1 canary active owner did not attach")
    fabric_before = content_digest(positive.fabric.to_payload())
    context_before = content_digest(positive.predictive_context.to_payload())
    positive.learn_bytes(
        new_data,
        epochs=2,
        learn_fabric=False,
        learn_predictive_context=False,
        consolidation_strength=0.5,
        boundary=active,
        authorization=authorization,
    )
    active_after = positive.active_predictive_readout_metadata
    if active_after is None:
        raise RuntimeError("M4.R1 canary active owner disappeared")
    checkpoint = positive.checkpoint()
    checkpoint_digest = content_digest(checkpoint)
    checkpoint_path = PROJECT_ROOT / "output" / "taiji_m4r1_consolidation_canary.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.save(checkpoint, checkpoint_path)
        restored = Taiji.from_checkpoint(
            torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        )
        restored_digest = content_digest(restored.checkpoint())
        live_score = positive.score_bytes(
            new_data,
            boundary=active,
            authorization=authorization,
        )
        restored_score = restored.score_bytes(
            new_data,
            boundary=active,
            authorization=authorization,
        )
    finally:
        checkpoint_path.unlink(missing_ok=True)

    gate = {
        "zero_strength_checkpoint_compatible": content_digest(baseline.checkpoint())
        == content_digest(zero_strength.checkpoint()),
        "active_owner_changes": active_before["readout_digest"] != active_after["readout_digest"],
        "protected_owner_unchanged": positive.readout_registry_status()["protected"][
            "readout_digest"
        ]
        == protected_before,
        "shared_fabric_unchanged": content_digest(positive.fabric.to_payload()) == fabric_before,
        "predictive_context_unchanged": content_digest(positive.predictive_context.to_payload())
        == context_before,
        "checkpoint_round_trip": restored_digest == checkpoint_digest,
        "registry_metadata_round_trip": restored.active_predictive_readout_metadata == active_after,
        "score_round_trip": live_score == restored_score,
        "source_checkpoint_unchanged": content_digest(source.checkpoint()) == source_digest,
        "positive_strength_is_finite": math.isfinite(0.5),
    }
    return {
        "format": FORMAT,
        "version": 1,
        "status": "passed" if all(gate.values()) else "failed",
        "can_promote": False,
        "candidate": {
            "owner": "predictive_readout.active",
            "reference_owner": "predictive_readout",
            "consolidation_strength": 0.5,
            "shared_fabric_learned": False,
            "predictive_context_learned": False,
        },
        "gate": gate,
        "source_checkpoint_digest": source_digest,
        "trained_checkpoint_digest": checkpoint_digest,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=REPORT_DEFAULT)
    args = parser.parse_args(argv)
    payload = evaluate()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "report": str(args.report),
                "status": payload["status"],
                "can_promote": False,
                "gate": payload["gate"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
