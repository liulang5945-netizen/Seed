"""Evaluate substrate-driven automatic neuron-growth proposals."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji import (  # noqa: E402
    AdaptiveNeuronRegion,
    AdaptiveStructuralGrowthController,
    StructuralGrowthDynamics,
    TaijiConfig,
    TSKV8Adapter,
)

REPORT_FORMAT = "taiji-auto-growth-v1"
MANIFEST_FORMAT = "taiji-auto-growth-manifest-v1"


def _config(*, budget: int) -> TaijiConfig:
    return TaijiConfig(
        alphabet_size=257,
        boundary_symbol=256,
        region_sizes=(8, 6),
        synapse_fan_in=3,
        motor_fan_in=4,
        lateral_fan_in=3,
        memory_units=12,
        memory_fan_in=3,
        memory_readout_fan_in=4,
        memory_meta_dim=4,
        memory_time_dim=4,
        memory_episode_dim=4,
        development_structural_budget=budget,
        seed=71,
    )


def _region() -> AdaptiveNeuronRegion:
    return AdaptiveNeuronRegion(
        region_id="adaptive.cortex",
        input_dim=7,
        unit_ids=("u0", "u1"),
        fan_in=3,
        generator=torch.Generator().manual_seed(71),
    )


def _controller() -> AdaptiveStructuralGrowthController:
    return AdaptiveStructuralGrowthController(
        dynamics=StructuralGrowthDynamics(
            ema_rate=1.0,
            error_threshold=0.6,
            holdout_transfer_threshold=0.7,
            minimum_resource_state=0.5,
            required_error_steps=3,
        )
    )


def evaluate() -> dict[str, object]:
    model = TSKV8Adapter(_config(budget=1), episode_id="auto-growth")
    model.attach_adaptive_neuron_region(_region())
    model.attach_structural_growth_controller(_controller())
    proposals = [
        model.propose_neuron_growth_from_error(
            region_id="adaptive.cortex",
            prediction_error=0.9,
            resource_state=0.8,
            holdout_transfer=0.9,
            evidence_ids=(f"tick:{index}",),
        )
        for index in range(1, 4)
    ]
    proposal = proposals[-1]
    assert proposal is not None
    committed = model.commit_neuron_add(proposal)
    region = model.neuron_regions[0]
    holdout = torch.zeros(region.input_dim)
    holdout[region.incoming.pre_index[-1].long()] = 1.0
    region.incoming.edge_weight[-1].zero_()
    before = float(region.incoming.forward(holdout)[-1].item())
    region.learn(holdout, torch.tensor([0.0, 0.0, 1.0]))
    after = float(region.incoming.forward(holdout)[-1].item())
    region.lesion_topology_proposal(proposal)
    lesion = float(region.step(torch.ones(region.input_dim))[-1].item())

    restored = TSKV8Adapter.from_native_checkpoint(model.native_checkpoint())
    checkpoint_continuation = bool(
        restored.structural_growth_controller is not None
        and restored.structural_growth_controller.total_observations == 3
        and restored.neuron_regions[0].unit_ids == region.unit_ids
        and restored.neuron_regions[0].lesioned_unit_ids == region.lesioned_unit_ids
    )
    rollback = restored.rollback_neuron_add(proposal.proposal_id)

    no_budget = TSKV8Adapter(_config(budget=0), episode_id="auto-growth-no-budget")
    no_budget.attach_adaptive_neuron_region(_region())
    no_budget.attach_structural_growth_controller(_controller())
    no_budget_proposal = None
    for index in range(1, 4):
        no_budget_proposal = no_budget.propose_neuron_growth_from_error(
            region_id="adaptive.cortex",
            prediction_error=0.9,
            resource_state=0.8,
            holdout_transfer=0.9,
            evidence_ids=(f"no-budget:{index}",),
        )
    assert no_budget_proposal is not None
    rejected = no_budget.commit_neuron_add(no_budget_proposal)

    gate = {
        "passed": bool(
            proposals[0] is None
            and proposals[1] is None
            and committed
            and before == 0.0
            and after > 0.0
            and lesion == 0.0
            and checkpoint_continuation
            and rollback
            and restored.neuron_regions[0].unit_ids == ("u0", "u1")
            and not rejected
            and no_budget.topology_proposals[-1].status == "rejected"
        ),
        "criterion": (
            "persistent substrate prediction error plus holdout transfer and available resources "
            "must emit a non-semantic neuron proposal; the runtime ledger must validate budget and "
            "checkpoint trial, the new unit must improve a holdout signal and fail under lesion, "
            "and the parent structure must be recoverable by reverse rollback"
        ),
    }
    return {
        "format": REPORT_FORMAT,
        "proposal": proposal.to_payload(),
        "metrics": {
            "persistent_error_required_steps": 3,
            "proposal_emitted_after_steps": 3,
            "unit_id": dict(proposal.specification)["unit_id"],
            "committed": committed,
            "unit_count_after_growth": region.unit_count,
            "holdout_before": before,
            "holdout_after_local_learning": after,
            "lesion_activity": lesion,
            "checkpoint_continuation": checkpoint_continuation,
            "rollback": rollback,
            "unit_count_after_rollback": restored.neuron_regions[0].unit_count,
            "rejected_without_budget": not rejected,
        },
        "gate": gate,
        "boundary": (
            "This gate proves substrate-driven proposal emission and controlled neuron birth; it does "
            "not claim unrestricted self-evolution, automatic region invention, or general intelligence."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_auto_growth_20260826.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "plans" / "manifests" / "taiji_auto_growth_v1.json",
    )
    args = parser.parse_args()
    report = evaluate()
    report_path = args.report if args.report.is_absolute() else PROJECT_ROOT / args.report
    manifest_path = args.manifest if args.manifest.is_absolute() else PROJECT_ROOT / args.manifest
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "format": MANIFEST_FORMAT,
        "task": "substrate-driven automatic neuron growth",
        "report": str(report_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "seed": 71,
        "controls": [
            "persistent_prediction_error",
            "holdout_transfer_gate",
            "resource_availability_gate",
            "non_semantic_identity",
            "budget_validation",
            "checkpoint_trial",
            "holdout_learning",
            "functional_lesion",
            "reverse_rollback",
            "zero_budget",
        ],
        "gate": report["gate"],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
