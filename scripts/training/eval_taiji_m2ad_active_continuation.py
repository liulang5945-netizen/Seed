"""Evaluate an active readout branch on the real seed11 continuation corpus."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji import (  # noqa: E402
    FoundationTrainingDataset,
    Taiji,
    WorkbenchBoundaryAuthorization,
    WorkbenchTaskBoundary,
)
from taiji.internalization import content_digest  # noqa: E402

FORMAT = "taiji-active-readout-continuation-canary-v1"


def _load_joint_child(path: Path, *, expected_seed: int) -> tuple[dict[str, Any], Taiji]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("joint child checkpoint must contain a mapping")
    expected_digest = content_digest(
        {key: value for key, value in payload.items() if key != "checkpoint_digest"}
    )
    if str(payload.get("checkpoint_digest", "")) != expected_digest:
        raise ValueError("joint child checkpoint digest mismatch")
    if payload.get("format") != "taiji-native-joint-training-v1":
        raise ValueError("active continuation requires a joint training checkpoint")
    if int(payload.get("version", -1)) < 4:
        raise ValueError("active continuation requires a v4 private-context child")
    # R0.6: the old check required training_phases == ["sequence"], which
    # excluded memory/identity-phase children that legitimately continue a
    # sequence child with the shared fabric frozen.  A child is
    # sequence-derived when it trained sequence at least once, or when its
    # sequence owners were frozen on a private substrate; pure phase labels
    # must not decide loadability.
    phases = payload.get("training_phases")
    if not isinstance(phases, list) or not phases:
        raise ValueError("active continuation requires a joint-trained child")
    if "sequence" not in phases and payload.get("sequence_fabric_learning") is not False:
        raise ValueError(
            "active continuation requires a sequence-trained or sequence-derived child"
        )
    if payload.get("sequence_fabric_learning") is not False:
        raise ValueError("active continuation requires frozen shared fabric")
    if payload.get("sequence_predictive_context_mode") != "private-plastic-temporal-v1":
        raise ValueError("active continuation requires private predictive context")
    model_payload = payload.get("model")
    if not isinstance(model_payload, Mapping):
        raise ValueError("joint child checkpoint is missing model payload")
    model = Taiji.from_checkpoint(model_payload)
    if int(model.config.seed) != int(expected_seed):
        raise ValueError("joint child seed does not match evaluator seed")
    return dict(payload), model


def _protected_owner_digest(model: Taiji) -> str:
    owners: dict[str, Any] = {
        "fabric": model.fabric.to_payload(),
        "motor": model.motor.to_payload(),
        "predictive_context": model.predictive_context.to_payload(),
        "predictive_readout": model.predictive_readout.to_payload(),
        "memory": model.memory.to_payload(),
    }
    if model.identity_organ is not None:
        owners["identity"] = model.identity_organ.to_payload(
            parent_checkpoint_digest="m2ad-active-continuation-canary"
        )
    return content_digest(owners)


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


def _score_read_only(model: Taiji, data: bytes) -> tuple[float, bool]:
    before = content_digest(model.checkpoint())
    score = model.score_bytes(data)
    after = content_digest(model.checkpoint())
    return float(score["mean_surprise"]) / 0.6931471805599453, before == after


def _fresh_process_digest(path: Path) -> tuple[str, float]:
    probe = (
        "import sys, torch; "
        "from taiji import Taiji; "
        "from taiji.internalization import content_digest; "
        "payload=torch.load(sys.argv[1], map_location='cpu', weights_only=False); "
        "print(content_digest(Taiji.from_checkpoint(payload).checkpoint()))"
    )
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-c", probe, str(path)],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip(), time.perf_counter() - started


def run_canary(
    checkpoint: Path,
    *,
    corpus_paths: Sequence[Path],
    partition_seed: int,
    protected_partition_seed: int,
    epochs: int,
    train_bytes: int,
    eval_bytes: int | None,
) -> dict[str, Any]:
    joint_payload, source_model = _load_joint_child(checkpoint, expected_seed=11)
    protected_dataset = FoundationTrainingDataset.from_jsonl(
        corpus_paths,
        profile="foundation",
        partition_seed=protected_partition_seed,
    )
    active_dataset = FoundationTrainingDataset.from_jsonl(
        corpus_paths,
        profile="foundation",
        partition_seed=partition_seed,
        exclude_dataset=protected_dataset,
    )
    if str(joint_payload.get("dataset_digest")) != active_dataset.digest:
        raise ValueError("child dataset digest does not match active continuation corpus")
    if str(joint_payload.get("protected_dataset_digest")) != protected_dataset.digest:
        raise ValueError("child protected dataset digest does not match protected corpus")
    if train_bytes <= 0 or train_bytes > len(active_dataset.train):
        raise ValueError("train_bytes must be within the active training partition")
    if eval_bytes is not None and (eval_bytes <= 0 or eval_bytes > len(active_dataset.holdout)):
        raise ValueError("eval_bytes must be within the active holdout partition")
    active_holdout = active_dataset.holdout[:eval_bytes]
    protected_retention = protected_dataset.retention[:eval_bytes]

    source_model.clear_active_predictive_readout()
    protected_model = Taiji.from_checkpoint(source_model.checkpoint())
    active_model = Taiji.from_checkpoint(source_model.checkpoint())
    protected_before = _protected_owner_digest(active_model)
    protected_readout_digest = active_model.readout_registry_status()["protected"][
        "readout_digest"
    ]
    boundary = WorkbenchTaskBoundary.issue(
        project_id="project:seed",
        task_id="task:active-continuation",
        session_id="session:seed11",
        language_id="python",
        capability_snapshot_id="capability:snapshot:1",
        capability_ids=("workspace.read",),
        generation_scope="active",
        issued_tick=10,
        ttl_ticks=40,
    )
    authorization = _authorization(boundary)
    active_model.clone_protected_predictive_readout_as_active(
        boundary_digest=boundary.token_digest,
    )
    active_before = active_model.active_predictive_readout_metadata
    if active_before is None:
        raise RuntimeError("active readout was not registered")
    active_model.learn_bytes(
        active_dataset.train[:train_bytes],
        epochs=epochs,
        include_boundary=True,
        use_memory=False,
        learn_fabric=False,
        learn_predictive_context=False,
        learn_predictive_readout=True,
        boundary=boundary,
        authorization=authorization,
    )
    active_after = active_model.active_predictive_readout_metadata
    if active_after is None:
        raise RuntimeError("active readout disappeared after continuation")
    protected_after = _protected_owner_digest(active_model)
    active_checkpoint = active_model.checkpoint()
    active_checkpoint_digest = content_digest(active_checkpoint)

    protected_bpb, protected_holdout_read_only = _score_read_only(
        protected_model,
        active_holdout,
    )
    protected_retention_bpb, protected_retention_read_only = _score_read_only(
        protected_model,
        protected_retention,
    )
    active_bpb, active_holdout_read_only = _score_read_only(
        active_model,
        active_holdout,
    )
    active_retention_bpb, active_retention_read_only = _score_read_only(
        active_model,
        protected_retention,
    )

    restored = Taiji.from_checkpoint(copy.deepcopy(active_checkpoint))
    restored_metadata = restored.active_predictive_readout_metadata
    restored_checkpoint_digest = content_digest(restored.checkpoint())
    active_output = active_model.generate(
        b"Taiji",
        16,
        boundary=boundary,
        authorization=authorization,
    )
    restored_output = restored.generate(
        b"Taiji",
        16,
        boundary=boundary,
        authorization=authorization,
    )

    disk_path = PROJECT_ROOT / "output" / "taiji_m2ad_active_probe.pt"
    try:
        torch.save(active_checkpoint, disk_path)
        fresh_digest, restore_seconds = _fresh_process_digest(disk_path)
        checkpoint_bytes = disk_path.stat().st_size
    finally:
        disk_path.unlink(missing_ok=True)

    protected_only = Taiji.from_checkpoint(source_model.checkpoint())
    try:
        protected_only.generate(
            b"Taiji",
            4,
            boundary=boundary,
            authorization=authorization,
        )
    except RuntimeError as exc:
        missing_registry_rejection = str(exc)
    else:
        missing_registry_rejection = ""

    checks = {
        "child_lineage_matches_real_phase_a_and_phase_b_data": True,
        "active_registry_has_parent_and_boundary": all(
            bool(active_before.get(key))
            for key in (
                "parent_checkpoint_digest",
                "boundary_digest",
                "readout_digest",
            )
        ),
        "active_readout_changes": active_before["readout_digest"]
        != active_after["readout_digest"],
        "protected_owners_unchanged": protected_before == protected_after,
        "protected_readout_digest_unchanged": protected_readout_digest
        == active_model.readout_registry_status()["protected"]["readout_digest"],
        "holdout_and_retention_are_read_only": all(
            (
                protected_holdout_read_only,
                protected_retention_read_only,
                active_holdout_read_only,
                active_retention_read_only,
            )
        ),
        "active_registry_round_trips_in_process": (
            restored_metadata == active_after
            and restored_checkpoint_digest == active_checkpoint_digest
        ),
        "active_registry_round_trips_in_fresh_process": fresh_digest
        == active_checkpoint_digest,
        "active_generation_uses_active_owner": (
            active_model.last_generation_route is not None
            and active_model.last_generation_route["generation_scope"] == "active"
            and active_model.last_generation_route["readout_owner"] == "predictive_readout.active"
        ),
        "active_generation_output_round_trips": active_output == restored_output,
        "missing_registry_fails_closed": "not attached" in missing_registry_rejection,
    }
    capability = {
        "active_phase_b_bpb": active_bpb,
        "protected_phase_b_bpb": protected_bpb,
        "phase_b_gain_bpb": protected_bpb - active_bpb,
        "active_protected_retention_bpb": active_retention_bpb,
        "protected_retention_bpb": protected_retention_bpb,
        "retention_delta_bpb": active_retention_bpb - protected_retention_bpb,
        "active_phase_b_improved": active_bpb < protected_bpb,
        "retention_within_0_05_bpb": active_retention_bpb <= protected_retention_bpb + 0.05,
    }
    return {
        "format": FORMAT,
        "version": 1,
        "status": "passed" if all(checks.values()) else "failed",
        "can_promote": False,
        "checkpoint": str(checkpoint),
        "seed": 11,
        "epochs": int(epochs),
        "train_bytes": int(train_bytes),
        "eval_bytes": int(eval_bytes or len(active_dataset.holdout)),
        "datasets": {
            "active": {
                "digest": active_dataset.digest,
                "partition_seed": int(partition_seed),
                "sample_counts": active_dataset.sample_counts,
            },
            "protected": {
                "digest": protected_dataset.digest,
                "partition_seed": int(protected_partition_seed),
                "sample_counts": protected_dataset.sample_counts,
            },
        },
        "checks": checks,
        "capability": capability,
        "registry": {
            "before_training": active_before,
            "after_training": active_after,
            "restored": restored_metadata,
        },
        "checkpoint_audit": {
            "active_checkpoint_digest": active_checkpoint_digest,
            "fresh_process_digest": fresh_digest,
            "checkpoint_bytes": int(checkpoint_bytes),
            "fresh_process_restore_seconds": restore_seconds,
            "holdout_updates": 0,
            "retention_updates": 0,
        },
        "owner_audit": {
            "protected_owner_digest_before": protected_before,
            "protected_owner_digest_after": protected_after,
            "protected_owners_unchanged": protected_before == protected_after,
            "missing_registry_rejection": missing_registry_rejection,
        },
        "boundary": boundary.to_payload(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, nargs="+", required=True)
    parser.add_argument("--partition-seed", type=int, default=10011)
    parser.add_argument("--protected-partition-seed", type=int, default=11)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--train-bytes",
        type=int,
        default=262_144,
        help="Real phase-B train prefix used for the CPU continuation (default: 256 KiB).",
    )
    parser.add_argument(
        "--eval-bytes",
        type=int,
        help="Optional holdout/retention prefix for a faster local canary; default uses all bytes.",
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.train_bytes <= 0:
        parser.error("--train-bytes must be positive")
    if args.eval_bytes is not None and args.eval_bytes <= 0:
        parser.error("--eval-bytes must be positive")
    result = run_canary(
        args.checkpoint,
        corpus_paths=args.corpus,
        partition_seed=args.partition_seed,
        protected_partition_seed=args.protected_partition_seed,
        epochs=args.epochs,
        train_bytes=args.train_bytes,
        eval_bytes=args.eval_bytes,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    json.loads(args.report.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "report": str(args.report),
                "status": result["status"],
                "checks_passed": sum(int(value) for value in result["checks"].values()),
                "checks_total": len(result["checks"]),
                "phase_b_gain_bpb": result["capability"]["phase_b_gain_bpb"],
                "protected_owners_unchanged": result["owner_audit"][
                    "protected_owners_unchanged"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
