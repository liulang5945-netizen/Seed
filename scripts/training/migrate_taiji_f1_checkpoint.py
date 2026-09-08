"""Migrate an existing Taiji F1 checkpoint into the read-only R1 overlay.

The source checkpoint is never overwritten.  The command validates the source,
mounts ``slow_weight=old`` and ``fast_delta=0``, saves a new checkpoint, then
loads it again in the same process to verify the disk round trip.  It does not
train, change the source payload, or claim promotion.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji import Taiji  # noqa: E402
from taiji.internalization import content_digest  # noqa: E402

FORMAT = "taiji-m4v2-r1-checkpoint-migration-v1"
VERSION = 1


def _stored_checkpoint_digest_matches(payload: Mapping[str, Any]) -> bool:
    stored = payload.get("checkpoint_digest")
    if stored is None:
        return True
    expected = content_digest(
        {key: value for key, value in payload.items() if key != "checkpoint_digest"}
    )
    return str(stored) == expected


def migrate_checkpoint(source: Path, output: Path) -> dict[str, Any]:
    source_payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(source_payload, dict):
        raise ValueError("source checkpoint must contain a mapping")
    source_checkpoint_digest_valid = _stored_checkpoint_digest_matches(source_payload)
    if not source_checkpoint_digest_valid:
        raise ValueError("source checkpoint digest mismatch")
    source_digest = content_digest(source_payload)
    model_payload = source_payload
    wrapped_training_checkpoint = False
    nested_model = source_payload.get("model")
    if isinstance(nested_model, Mapping) and str(nested_model.get("format", "")).startswith(
        "taiji-native-v"
    ):
        model_payload = dict(nested_model)
        wrapped_training_checkpoint = True
    model = Taiji.from_checkpoint(model_payload)
    migration = model.migrate_f1_to_developmental_synapses()
    migrated_model_payload = model.checkpoint()
    if wrapped_training_checkpoint:
        migrated_payload = dict(source_payload)
        migrated_payload["model"] = migrated_model_payload
        if "checkpoint_digest" in migrated_payload:
            migrated_payload["checkpoint_digest"] = content_digest(
                {
                    key: value
                    for key, value in migrated_payload.items()
                    if key != "checkpoint_digest"
                }
            )
    else:
        migrated_payload = migrated_model_payload
    migrated_digest = content_digest(migrated_payload)
    source_after_digest = content_digest(source_payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(migrated_payload, output)

    restored_payload = torch.load(output, map_location="cpu", weights_only=False)
    if not isinstance(restored_payload, dict):
        raise ValueError("migrated checkpoint did not load as a mapping")
    restored_outer_digest_valid = _stored_checkpoint_digest_matches(restored_payload)
    if not restored_outer_digest_valid:
        raise ValueError("migrated checkpoint digest mismatch")
    restored_model_payload = restored_payload
    if wrapped_training_checkpoint:
        nested_restored_model = restored_payload.get("model")
        if not isinstance(nested_restored_model, Mapping):
            raise ValueError("migrated training checkpoint is missing model payload")
        restored_model_payload = dict(nested_restored_model)
    restored = Taiji.from_checkpoint(restored_model_payload)
    restored_digest = content_digest(restored.checkpoint())
    fresh_restore_matches = restored_digest == content_digest(migrated_model_payload)
    source_unchanged = source_after_digest == source_digest
    fast_is_zero = bool(restored.developmental_f1_bundle) and restored.developmental_f1_bundle.fast_is_zero
    status = bool(
        fresh_restore_matches
        and source_unchanged
        and fast_is_zero
        and source_checkpoint_digest_valid
        and restored_outer_digest_valid
    )
    return {
        "format": FORMAT,
        "version": VERSION,
        "status": "passed" if status else "failed",
        "can_promote": False,
        "source_checkpoint": str(source),
        "output_checkpoint": str(output),
        "wrapped_training_checkpoint": wrapped_training_checkpoint,
        "source_checkpoint_digest": source_digest,
        "migrated_checkpoint_digest": migrated_digest,
        "restored_checkpoint_digest": restored_digest,
        "source_unchanged": source_unchanged,
        "source_checkpoint_digest_valid": source_checkpoint_digest_valid,
        "restored_outer_digest_valid": restored_outer_digest_valid,
        "fresh_restore_matches": fresh_restore_matches,
        "fast_is_zero": fast_is_zero,
        "effective_parameter_count": migration["effective_parameter_count"],
        "developmental_state_scalar_count": migration["state_scalar_count"],
        "owner_graph_digest": migration["owner_graph_digest"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)

    report = migrate_checkpoint(args.source, args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
