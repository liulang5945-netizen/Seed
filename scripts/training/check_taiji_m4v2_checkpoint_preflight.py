"""Verify checkpoint persistence before starting a continual-learning pilot.

This is a bounded CPU preflight, not a capability experiment.  It performs a
small developmental update only to prove that the model state, replay buffer,
RNG, owner lineage, course cursor and training-state envelope survive an
atomic save/fresh restore/continued-update cycle.  The temporary checkpoint is
removed after verification; the JSON report is the durable audit artifact.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji import Taiji, content_digest  # noqa: E402

DEFAULT_PARENT = PROJECT_ROOT / "checkpoints" / "taiji_r6_parents" / "model_17.pt"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m4v2_checkpoint_preflight_20260910.json"
REPORT_FORMAT = "taiji-m4v2-checkpoint-preflight-v1"
VERSION = 1
COURSE_ID = "m4v2-b2-checkpoint-preflight"
COURSE_CURSOR = {"phase": "G", "step": 1}
PREFIX = b"b2-preflight-old-new"
SUFFIX = b"b2-preflight-continuation"


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError(f"checkpoint at {path} must be a mapping")
    return {str(key): value for key, value in payload.items()}


def _save_atomic(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _envelope(
    *,
    parent_digest: str,
    checkpoint: Mapping[str, Any],
    owner_graph_digest: str,
    replay_event_count: int,
) -> dict[str, Any]:
    model_digest = content_digest(checkpoint)
    training_state = {
        "format": "taiji-m4v2-training-state-v1",
        "course_id": COURSE_ID,
        "course_cursor": copy.deepcopy(COURSE_CURSOR),
        "learning_mode_before_restore": "fast_slow",
        "optimizer": {"kind": "none-local-developmental-update", "state": None},
        "replay_event_count": int(replay_event_count),
        "rng_state_digest": content_digest({"rng_state": checkpoint["rng_state"]}),
        "owner_graph_digest": owner_graph_digest,
    }
    envelope: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "version": VERSION,
        "parent_checkpoint_digest": parent_digest,
        "model_checkpoint_digest": model_digest,
        "model_checkpoint": dict(checkpoint),
        "training_state": training_state,
    }
    envelope["envelope_digest"] = content_digest(envelope)
    return envelope


def _verify_envelope(
    payload: Mapping[str, Any],
    *,
    expected_parent_digest: str,
    expected_owner_graph_digest: str,
) -> dict[str, Any]:
    if payload.get("format") != REPORT_FORMAT or int(payload.get("version", -1)) != VERSION:
        raise ValueError("checkpoint envelope format/version is invalid")
    unsigned = dict(payload)
    envelope_digest = unsigned.pop("envelope_digest", None)
    if envelope_digest != content_digest(unsigned):
        raise ValueError("checkpoint envelope digest is invalid")
    if payload.get("parent_checkpoint_digest") != expected_parent_digest:
        raise ValueError("checkpoint envelope parent digest differs")
    checkpoint = payload.get("model_checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint envelope has no model checkpoint")
    if payload.get("model_checkpoint_digest") != content_digest(checkpoint):
        raise ValueError("checkpoint envelope model digest is invalid")
    training_state = payload.get("training_state")
    if not isinstance(training_state, Mapping):
        raise ValueError("checkpoint envelope has no training state")
    if training_state.get("course_id") != COURSE_ID:
        raise ValueError("checkpoint envelope course id is invalid")
    if training_state.get("course_cursor") != COURSE_CURSOR:
        raise ValueError("checkpoint envelope course cursor is invalid")
    if training_state.get("optimizer") != {
        "kind": "none-local-developmental-update",
        "state": None,
    }:
        raise ValueError("checkpoint envelope optimizer state is invalid")
    if training_state.get("owner_graph_digest") != expected_owner_graph_digest:
        raise ValueError("checkpoint envelope owner graph differs")
    if training_state.get("rng_state_digest") != content_digest(
        {"rng_state": checkpoint["rng_state"]}
    ):
        raise ValueError("checkpoint envelope RNG digest differs")
    return {str(key): value for key, value in checkpoint.items()}


def run_preflight(
    *,
    parent_path: Path = DEFAULT_PARENT,
    report_path: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    started = time.perf_counter()
    checks: dict[str, bool] = {}
    parent_digest = ""
    temporary_checkpoint_path: Path | None = None
    try:
        parent = _load_mapping(parent_path)
        parent_digest = content_digest(parent)
        if Taiji.DEVELOPMENTAL_F1_KEY in parent:
            raise ValueError("B2 parent must not already contain developmental F1 state")
        parent_model = Taiji.from_checkpoint(copy.deepcopy(parent))
        model = Taiji.from_checkpoint(copy.deepcopy(parent))
        migration = model.migrate_f1_to_developmental_synapses()
        model.set_developmental_f1_learning_mode("fast_slow")
        model.learn_bytes(PREFIX, epochs=1, learn_fabric=False)
        bundle = model.developmental_f1_bundle
        if bundle is None:
            raise ValueError("developmental bundle was not mounted")
        mid_checkpoint = model.checkpoint()
        envelope = _envelope(
            parent_digest=parent_digest,
            checkpoint=mid_checkpoint,
            owner_graph_digest=bundle.owner_graph_digest,
            replay_event_count=model.developmental_f1_replay_count,
        )
        workspace_temp_root = PROJECT_ROOT / "output"
        workspace_temp_root.mkdir(parents=True, exist_ok=True)
        temporary_checkpoint_path = workspace_temp_root / ".taiji_m4v2_b2_preflight.pt"
        temporary_checkpoint_path.unlink(missing_ok=True)
        try:
            _save_atomic(envelope, temporary_checkpoint_path)
            loaded_envelope = _load_mapping(temporary_checkpoint_path)
            restored_checkpoint = _verify_envelope(
                loaded_envelope,
                expected_parent_digest=parent_digest,
                expected_owner_graph_digest=bundle.owner_graph_digest,
            )
            restored = Taiji.from_checkpoint(copy.deepcopy(restored_checkpoint))

            checks["parent_fresh_restore"] = (
                content_digest(parent_model.checkpoint()) == parent_digest
            )
            checks["migration_fast_zero"] = bool(migration["fast_is_zero"])
            checks["developmental_update_wrote_fast"] = (
                restored.developmental_f1_bundle is not None
                and not restored.developmental_f1_bundle.fast_is_zero
            )
            checks["replay_persisted"] = (
                restored.developmental_f1_replay_count == model.developmental_f1_replay_count > 0
            )
            checks["fresh_restore_digest_matches"] = content_digest(
                restored.checkpoint()
            ) == content_digest(mid_checkpoint)
            checks["fresh_restore_is_read_only"] = (
                restored.developmental_f1_learning_mode == "read_only"
            )
            checks["owner_lineage_persisted"] = (
                restored.developmental_f1_bundle is not None
                and restored.developmental_f1_bundle.owner_graph_digest == bundle.owner_graph_digest
            )
            checks["old_f1_owners_unchanged"] = all(
                content_digest(restored_checkpoint[key]) == content_digest(parent[key])
                for key in ("predictive_context", "predictive_readout")
            )

            uninterrupted = Taiji.from_checkpoint(copy.deepcopy(mid_checkpoint))
            uninterrupted.set_developmental_f1_learning_mode("fast_slow")
            uninterrupted.learn_bytes(SUFFIX, epochs=1, learn_fabric=False)
            resumed = Taiji.from_checkpoint(copy.deepcopy(restored_checkpoint))
            resumed.set_developmental_f1_learning_mode("fast_slow")
            resumed.learn_bytes(SUFFIX, epochs=1, learn_fabric=False)
            checks["rng_and_state_continuation_matches"] = content_digest(
                resumed.checkpoint()
            ) == content_digest(uninterrupted.checkpoint())

            rollback = Taiji.from_checkpoint(copy.deepcopy(parent))
            checks["rollback_matches_parent"] = (
                content_digest(rollback.checkpoint()) == parent_digest
            )
            checks["atomic_checkpoint_exists_before_cleanup"] = temporary_checkpoint_path.is_file()
        finally:
            temporary_checkpoint_path.unlink(missing_ok=True)
        checks["temporary_checkpoint_cleaned"] = not temporary_checkpoint_path.exists()
        status = "passed" if all(checks.values()) else "failed"
        report: dict[str, Any] = {
            "report_format": REPORT_FORMAT,
            "version": VERSION,
            "status": status,
            "preflight_only": True,
            "research_course_executed": False,
            "training_performed": True,
            "preflight_only_training_step": True,
            "candidate_promoted": False,
            "device": "cpu",
            "parent_path": str(parent_path),
            "parent_checkpoint_digest": parent_digest,
            "checks": checks,
            "migration": {
                "source_checkpoint_digest": migration["source_checkpoint_digest"],
                "owner_graph_digest": migration["owner_graph_digest"],
                "effective_parameter_count": migration["effective_parameter_count"],
                "state_scalar_count": migration["state_scalar_count"],
            },
            "temporary_checkpoint_removed": checks.get("temporary_checkpoint_cleaned", False),
            "can_start_training_pilot": status == "passed",
            "can_promote": False,
            "elapsed_seconds": time.perf_counter() - started,
        }
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        report = {
            "report_format": REPORT_FORMAT,
            "version": VERSION,
            "status": "blocked",
            "preflight_only": True,
            "research_course_executed": False,
            "training_performed": False,
            "preflight_only_training_step": False,
            "candidate_promoted": False,
            "device": "cpu",
            "parent_path": str(parent_path),
            "parent_checkpoint_digest": parent_digest or None,
            "checks": checks,
            "blocking_reason": f"{type(exc).__name__}: {exc}",
            "can_start_training_pilot": False,
            "can_promote": False,
            "elapsed_seconds": time.perf_counter() - started,
        }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    parent_path = args.parent if args.parent.is_absolute() else PROJECT_ROOT / args.parent
    report_path = args.report if args.report.is_absolute() else PROJECT_ROOT / args.report
    report = run_preflight(parent_path=parent_path, report_path=report_path)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
