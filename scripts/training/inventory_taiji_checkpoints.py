"""Inventory Taiji joint child checkpoints with lineage, owner and data metadata.

R0.6: produce a short, versioned checkpoint/owner inventory so every score can
answer "which checkpoint, which owner, what data, which capability".  Each
entry records the direct parent, authenticated ancestor, training phases,
data digests, organ training coverage, model format and a digest-integrity
check.  It also applies the R0.6 sequence-derived admission rule so a
memory/identity-phase child that legitimately continues a sequence child is
not excluded by a phase-label-only loader gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji.internalization import content_digest  # noqa: E402

INVENTORY_FORMAT = "taiji-checkpoint-owner-inventory-v1"
INVENTORY_VERSION = 1
JOINT_FORMAT = "taiji-native-joint-training-v1"


def inspect_checkpoint(path: Path) -> dict[str, Any]:
    """Return one owner/lineage entry for a checkpoint file."""

    entry: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return entry
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 - report load failures, do not abort
        entry["load_error"] = f"{type(exc).__name__}: {exc}"
        return entry
    if not isinstance(payload, Mapping):
        entry["load_error"] = "checkpoint payload is not a mapping"
        return entry

    is_pure_taiji = str(payload.get("format")) == "taiji-native-v10"
    if is_pure_taiji:
        # A bare Taiji checkpoint (e.g. a persisted active readout branch from
        # the R1/R2 phase-C arms) has no joint-training wrapper.  Its content
        # digest is the content-addressed identifier; registry/boundary owner
        # audit lives in the "predictive_readout_registry" payload.
        entry["checkpoint_digest"] = content_digest(payload)
        entry["digest_verified"] = True
        entry["format"] = "taiji-native-v10"
        entry["version"] = payload.get("state_version", payload.get("version"))
        registry = payload.get("predictive_readout_registry")
        if isinstance(registry, Mapping):
            generations = registry.get("generations")
            entry["active_readout_generation"] = (
                len(generations) if isinstance(generations, list) else 0
            )
        entry["model_organs"] = sorted(
            key
            for key in (
                "fabric",
                "motor",
                "predictive_context",
                "predictive_readout",
                "memory",
                "identity_organ",
            )
            if key in payload
        )
        return entry

    expected_digest = content_digest(
        {key: value for key, value in payload.items() if key != "checkpoint_digest"}
    )
    recorded_digest = str(payload.get("checkpoint_digest", ""))
    entry["checkpoint_digest"] = recorded_digest
    entry["digest_verified"] = recorded_digest == expected_digest
    entry["format"] = payload.get("format")
    entry["version"] = payload.get("version")
    entry["model_tier"] = payload.get("model_tier")
    entry["parent_checkpoint_digest"] = payload.get("parent_checkpoint_digest")
    entry["ancestor_checkpoint_digest"] = payload.get("continuation_source_checkpoint_digest")
    entry["training_phases"] = payload.get("training_phases")
    entry["current_phase"] = payload.get("phase")
    entry["global_step"] = payload.get("global_step")
    entry["code_revision"] = payload.get("code_revision")

    data_digests = {
        "corpus_digest": payload.get("corpus_digest"),
        "dataset_digest": payload.get("dataset_digest"),
        "protected_dataset_digest": payload.get("protected_dataset_digest"),
        "memory_digest": payload.get("memory_digest"),
        "world_action_digest": payload.get("world_action_digest"),
        "replay_dataset_digest": payload.get("replay_dataset_digest"),
        "replay_memory_digest": payload.get("replay_memory_digest"),
    }
    entry["data_digests"] = {
        key: value for key, value in data_digests.items() if value is not None
    }

    phases = payload.get("training_phases")
    phase_list = phases if isinstance(phases, list) else []
    identity_growth = payload.get("identity_growth")
    entry["organ_coverage"] = {
        "fabric": {
            "trained": "sequence" in phase_list,
            "sequence_fabric_learning": payload.get("sequence_fabric_learning"),
            "phase_checks": len(payload.get("sequence_fabric_phase_checks") or []),
        },
        "predictive_context": {
            "trained": "sequence" in phase_list,
            "mode": payload.get("sequence_predictive_context_mode"),
            "phase_checks": len(payload.get("sequence_predictive_context_phase_checks") or []),
        },
        "predictive_readout": {
            "trained": "sequence" in phase_list,
            "mode": payload.get("sequence_readout_mode"),
            "phase_checks": len(payload.get("sequence_readout_phase_checks") or []),
        },
        "memory": {"trained": "memory" in phase_list},
        "world": {"trained": "world" in phase_list},
        "goal": {"trained": "goal" in phase_list},
        "identity": {
            "trained": bool(identity_growth),
            "growth_events": len(identity_growth) if isinstance(identity_growth, list) else 0,
        },
    }

    model_payload = payload.get("model")
    if isinstance(model_payload, Mapping):
        entry["model_organs"] = sorted(
            key
            for key in (
                "fabric",
                "motor",
                "predictive_context",
                "predictive_readout",
                "memory",
                "identity_organ",
            )
            if key in model_payload
        )

    # R0.6 admission: a child is sequence-derived when it trained sequence at
    # least once, or when its sequence owners were frozen on a private
    # substrate.  Phase labels alone must not exclude a memory/identity child.
    entry["sequence_derived_admission"] = bool(
        phase_list and ("sequence" in phase_list or payload.get("sequence_fabric_learning") is False)
    )
    return entry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path, help="checkpoint files or directories")
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / f"taiji_m2_r0_checkpoint_inventory_{datetime.now(timezone.utc):%Y%m%d}.json",
    )
    args = parser.parse_args(argv)

    files: list[Path] = []
    for candidate in args.checkpoints:
        if candidate.is_dir():
            files.extend(sorted(candidate.glob("*.pt")))
        elif candidate.is_file():
            files.append(candidate)
        else:
            print(f"warning: no such path: {candidate}", file=sys.stderr)

    entries = [inspect_checkpoint(path) for path in files]
    report = {
        "format": INVENTORY_FORMAT,
        "version": INVENTORY_VERSION,
        "count": len(entries),
        "entries": entries,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    for entry in entries:
        digest = str(entry.get("checkpoint_digest") or "-")[:16]
        verified = entry.get("digest_verified", False)
        phases = ",".join(entry.get("training_phases") or [])
        admitted = entry.get("sequence_derived_admission")
        ckpt_format = entry.get("format")
        print(
            f"{Path(entry['path']).name:44s} fmt={ckpt_format} "
            f"phases=[{phases}] digest={digest} ok={verified} seq-admit={admitted}"
        )
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
