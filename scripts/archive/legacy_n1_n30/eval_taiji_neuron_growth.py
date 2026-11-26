"""Evaluate identity-preserving neuron birth in the native Taiji runtime."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji import AdaptiveNeuronRegion, TaijiConfig, TSKV8Adapter  # noqa: E402

REPORT_FORMAT = "taiji-neuron-growth-v1"
MANIFEST_FORMAT = "taiji-neuron-growth-manifest-v1"


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
        input_source_id="fabric.region.0",
        generator=torch.Generator().manual_seed(71),
    )


def evaluate() -> dict[str, object]:
    region = _region()
    old_ids = region.unit_ids
    old_incoming_index = region.incoming.pre_index.clone()
    old_incoming_weight = region.incoming.edge_weight.clone()
    old_recurrent_index = region.recurrent.pre_index.clone() if region.recurrent else None
    old_recurrent_weight = region.recurrent.edge_weight.clone() if region.recurrent else None
    proposal = region.propose_unit_add(
        unit_id="u2",
        evidence_ids=("holdout:novel-unit",),
        parent_checkpoint_id="parent:region",
    )
    region.apply_topology_proposal(
        proposal,
        generator=torch.Generator().manual_seed(99),
    )
    identity_preserved = region.unit_ids[:2] == old_ids
    support_preserved = bool(
        torch.equal(region.incoming.pre_index[:2], old_incoming_index)
        and torch.equal(region.incoming.edge_weight[:2], old_incoming_weight)
    )
    if region.recurrent is not None and old_recurrent_index is not None:
        support_preserved = bool(
            support_preserved
            and torch.equal(region.recurrent.pre_index[:2], old_recurrent_index)
            and torch.equal(region.recurrent.edge_weight[:2], old_recurrent_weight)
        )

    holdout = torch.zeros(region.input_dim)
    holdout[region.incoming.pre_index[2].long()] = 1.0
    region.incoming.edge_weight[2].zero_()
    before = float(region.incoming.forward(holdout)[2].item())
    region.learn(holdout, torch.tensor([0.0, 0.0, 1.0]))
    after = float(region.incoming.forward(holdout)[2].item())
    region.lesion_topology_proposal(proposal)
    lesion_activity = float(region.step(torch.ones(region.input_dim))[2].item())
    region_roundtrip = AdaptiveNeuronRegion.from_payload(
        region.to_payload(),
        generator=torch.Generator().manual_seed(123),
    )
    checkpoint_roundtrip = bool(
        region_roundtrip.unit_ids == region.unit_ids
        and region_roundtrip.lesioned_unit_ids == region.lesioned_unit_ids
        and torch.equal(region_roundtrip.incoming.pre_index, region.incoming.pre_index)
    )

    model = TSKV8Adapter(_config(budget=1), episode_id="neuron-ledger")
    runtime_region = _region()
    model.attach_adaptive_neuron_region(runtime_region)
    runtime_proposal = model.propose_neuron_add(
        region_id=runtime_region.region_id,
        unit_id="u2",
        evidence_ids=("runtime:neuron-holdout",),
    )
    accepted = model.commit_neuron_add(runtime_proposal)
    accepted_entry = model.topology_proposals[-1]
    budget_after_accept = model.cognitive_snapshot().development.structural_budget
    restored = TSKV8Adapter.from_native_checkpoint(model.native_checkpoint())
    checkpoint_status = restored.topology_proposals[-1].status
    rollback = restored.rollback_neuron_add(runtime_proposal.proposal_id)
    budget_after_rollback = restored.cognitive_snapshot().development.structural_budget
    no_budget = TSKV8Adapter(_config(budget=0), episode_id="neuron-no-budget")
    no_budget_region = _region()
    no_budget.attach_adaptive_neuron_region(no_budget_region)
    rejected_proposal = no_budget.propose_neuron_add(
        region_id=no_budget_region.region_id,
        unit_id="u2",
        evidence_ids=("runtime:neuron-holdout",),
    )
    rejected = no_budget.commit_neuron_add(rejected_proposal)
    rejected_entry = no_budget.topology_proposals[-1]
    gate = {
        "passed": bool(
            identity_preserved
            and support_preserved
            and before == 0.0
            and after > 0.0
            and lesion_activity == 0.0
            and checkpoint_roundtrip
            and accepted
            and accepted_entry.status == "accepted"
            and budget_after_accept == 0
            and checkpoint_status == "accepted"
            and rollback
            and budget_after_rollback == 1
            and not rejected
            and rejected_entry.status == "rejected"
        ),
        "criterion": (
            "neuron birth must append a stable identity, preserve existing sparse support and "
            "state coordinates, learn a holdout signal locally, fail under functional lesion, "
            "survive checkpoint continuation, consume ledger budget, and rollback in reverse order"
        ),
    }
    return {
        "format": REPORT_FORMAT,
        "proposal": proposal.to_payload(),
        "metrics": {
            "identity_preserved": identity_preserved,
            "support_preserved": support_preserved,
            "unit_count_after_growth": region.unit_count,
            "holdout_before": before,
            "holdout_after_local_learning": after,
            "lesion_activity": lesion_activity,
            "checkpoint_roundtrip": checkpoint_roundtrip,
            "runtime_accepted": accepted,
            "runtime_accepted_status": accepted_entry.status,
            "budget_after_accept": budget_after_accept,
            "checkpoint_status": checkpoint_status,
            "runtime_rollback": rollback,
            "budget_after_rollback": budget_after_rollback,
            "rejected_without_budget": not rejected,
            "rejected_status": rejected_entry.status,
        },
        "gate": gate,
        "boundary": (
            "This gate proves one adaptive neuron organ and its runtime ledger; it does not yet "
            "claim automatic multi-region routing, open-domain self-evolution, or general intelligence."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_neuron_growth_20260826.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "plans" / "manifests" / "taiji_neuron_growth_v1.json",
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
        "task": "identity-preserving neuron growth",
        "report": str(report_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "seed": 71,
        "controls": [
            "stable_identity",
            "support_preservation",
            "local_holdout_learning",
            "functional_lesion",
            "checkpoint_continuation",
            "budget_consumption",
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
