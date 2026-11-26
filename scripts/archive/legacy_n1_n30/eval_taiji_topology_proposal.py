"""Evaluate a substrate-level Taiji synapse topology proposal."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji import StructuralTopologyProposal, TaijiConfig, TaijiFabric  # noqa: E402

REPORT_FORMAT = "taiji-topology-proposal-v1"
MANIFEST_FORMAT = "taiji-topology-proposal-manifest-v1"


def _fabric() -> TaijiFabric:
    config = TaijiConfig(
        alphabet_size=16,
        boundary_symbol=15,
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
        seed=71,
    )
    return TaijiFabric(
        config,
        generator=torch.Generator().manual_seed(71),
    )


def evaluate() -> dict[str, object]:
    fabric = _fabric()
    decoder = fabric.decoders[0]
    row = decoder.pre_index[0].long()
    replacement = next(index for index in range(decoder.in_features) if index not in row)
    proposal = fabric.propose_synapse_rewire(
        substrate_id="fabric.decoder.0",
        post_index=0,
        slot_index=0,
        replacement_pre_index=replacement,
        evidence_ids=("holdout:donor-response",),
        parent_checkpoint_id="fabric-parent:0",
    )
    roundtripped = StructuralTopologyProposal.from_payload(proposal.to_payload())
    parent = fabric.to_payload()
    holdout = torch.zeros(decoder.in_features)
    holdout[replacement] = 0.8
    before = float(decoder.forward(holdout)[0].item())

    applied = fabric.apply_synapse_rewire(roundtripped)
    decoder.local_update(
        torch.ones(decoder.out_features),
        holdout,
        learning_rate=0.8,
        weight_decay=0.0,
    )
    after = float(decoder.forward(holdout)[0].item())
    learned = fabric.to_payload()

    checkpoint = _fabric()
    checkpoint.load_payload(learned)
    checkpoint_score = float(checkpoint.decoders[0].forward(holdout)[0].item())
    lesioned = checkpoint.lesion_synapse_rewire(roundtripped)
    lesion_score = float(checkpoint.decoders[0].forward(holdout)[0].item())

    fabric.load_payload(parent)
    rollback_score = float(fabric.decoders[0].forward(holdout)[0].item())
    gate = {
        "passed": bool(
            applied
            and after > before
            and checkpoint_score == after
            and lesioned
            and lesion_score < checkpoint_score
            and rollback_score == before
        ),
        "criterion": (
            "a substrate proposal must change only an admissible synapse, improve a held-out "
            "donor response after local learning, survive topology checkpoint roundtrip, "
            "fail under functional lesion, and restore its parent topology"
        ),
    }
    return {
        "format": REPORT_FORMAT,
        "proposal": proposal.to_payload(),
        "metrics": {
            "applied": applied,
            "holdout_before": before,
            "holdout_after_learning": after,
            "checkpoint_score": checkpoint_score,
            "lesion_score": lesion_score,
            "rollback_score": rollback_score,
            "proposal_roundtrip": roundtripped.to_payload() == proposal.to_payload(),
        },
        "gate": gate,
        "boundary": (
            "This gate proves auditable fixed-fan-in synapse rewiring only; it does not claim "
            "neuron birth, region birth, open-domain self-evolution or general intelligence."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_topology_proposal_20260826.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "plans" / "manifests" / "taiji_topology_proposal_v1.json",
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
        "task": "substrate-level synapse topology proposal",
        "report": str(report_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "seed": 71,
        "controls": [
            "proposal_roundtrip",
            "holdout_response",
            "functional_lesion",
            "parent_rollback",
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
