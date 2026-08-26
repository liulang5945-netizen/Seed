"""Evaluate runtime ownership of topology proposals and structural budget."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji import TaijiConfig, TSKV8Adapter  # noqa: E402

REPORT_FORMAT = "taiji-topology-runtime-ledger-v1"
MANIFEST_FORMAT = "taiji-topology-runtime-ledger-manifest-v1"


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


def _proposal(model: TSKV8Adapter):
    bank = model.fabric.decoders[0]
    row = bank.pre_index[0].long()
    replacement = next(index for index in range(bank.in_features) if index not in row)
    return model.propose_synapse_rewire(
        substrate_id="fabric.decoder.0",
        post_index=0,
        slot_index=0,
        replacement_pre_index=replacement,
        evidence_ids=("runtime:topology-holdout",),
    )


def evaluate() -> dict[str, object]:
    model = TSKV8Adapter(_config(budget=1), episode_id="topology-ledger")
    original = model.fabric.decoders[0].pre_index.clone()
    proposal = _proposal(model)
    accepted = model.commit_synapse_rewire(proposal)
    accepted_entry = model.topology_proposals[-1]
    budget_after_accept = model.cognitive_snapshot().development.structural_budget

    restored = TSKV8Adapter.from_native_checkpoint(model.native_checkpoint())
    checkpoint_status = restored.topology_proposals[-1].status
    rollback = restored.rollback_synapse_rewire(accepted_entry.proposal_id)
    budget_after_rollback = restored.cognitive_snapshot().development.structural_budget
    rollback_topology = restored.fabric.decoders[0].pre_index.equal(original)

    no_budget = TSKV8Adapter(_config(budget=0), episode_id="topology-no-budget")
    rejected = _proposal(no_budget)
    rejected_result = no_budget.commit_synapse_rewire(rejected)
    rejected_entry = no_budget.topology_proposals[-1]
    gate = {
        "passed": bool(
            accepted
            and accepted_entry.status == "accepted"
            and budget_after_accept == 0
            and checkpoint_status == "accepted"
            and rollback
            and budget_after_rollback == 1
            and rollback_topology
            and not rejected_result
            and rejected_entry.status == "rejected"
            and no_budget.cognitive_snapshot().development.structural_budget == 0
        ),
        "criterion": (
            "runtime topology proposals must be ledger-owned, consume structural budget only "
            "after checkpoint validation, survive native checkpoint continuation, rollback in "
            "reverse order, and fail closed when budget is exhausted"
        ),
    }
    return {
        "format": REPORT_FORMAT,
        "proposal": proposal.to_payload(),
        "metrics": {
            "accepted": accepted,
            "accepted_status": accepted_entry.status,
            "budget_after_accept": budget_after_accept,
            "checkpoint_status": checkpoint_status,
            "rollback": rollback,
            "budget_after_rollback": budget_after_rollback,
            "rollback_topology": rollback_topology,
            "rejected_without_budget": not rejected_result,
            "rejected_status": rejected_entry.status,
        },
        "gate": gate,
        "boundary": (
            "This gate proves runtime budget and rollback ownership for synapse rewiring only; "
            "it does not claim neuron/region birth or open-domain self-evolution."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_topology_runtime_ledger_20260826.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "plans" / "manifests" / "taiji_topology_runtime_ledger_v1.json",
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
        "task": "runtime topology proposal ledger",
        "report": str(report_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "seed": 71,
        "controls": ["budget_consumption", "native_checkpoint", "reverse_rollback", "zero_budget"],
        "gate": report["gate"],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
