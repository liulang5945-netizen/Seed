"""Run the bounded M4.V2.R2 S/G fast-slow canary.

This is a contract canary, not the R6 promotion experiment.  It keeps one
parent and one owner graph, compares slow-only, fast-only and fast-plus-real-
replay arms, and always reports ``can_promote=false`` because a single CPU
course cannot establish continual-growth promotion.
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
    ContinualCourseManifest,
    ContinualEvaluationSnapshot,
    ContinualScorecard,
    CoursePhase,
    MetricObservation,
    MetricSpec,
    Taiji,
    TaijiConfig,
)
from taiji.internalization import content_digest  # noqa: E402

CANARY_FORMAT = "taiji-m4v2-r2-sg-canary-v1"
CANARY_VERSION = 1
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m4v2_r2_sg_canary.json"

S_TRAIN = b"abba-caba-abba-caba"
S_HOLDOUT = b"caba-bbaa-caba"
G_OLD = b"abba-caba"
G_NEW = b"xyz-zyx-uvw"
G_HOLDOUT = b"zyx-uvw-xyz"


def _tiny_config() -> TaijiConfig:
    return TaijiConfig(
        region_sizes=(24,),
        synapse_fan_in=6,
        motor_fan_in=12,
        memory_units=24,
        memory_fan_in=6,
        memory_meta_dim=16,
        memory_readout_fan_in=12,
        seed=71,
    )


def _load_parent(path: Path | None) -> dict[str, Any]:
    if path is None:
        model = Taiji(_tiny_config(), episode_id="r2-canary-parent")
        model.learn_bytes(S_TRAIN, epochs=1, learn_fabric=False)
        return model.checkpoint()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("parent checkpoint must be a mapping")
    if payload.get("format") == Taiji.CHECKPOINT_FORMAT:
        return payload
    nested = payload.get("model")
    if isinstance(nested, dict) and nested.get("format") == Taiji.CHECKPOINT_FORMAT:
        return nested
    raise ValueError("parent path must contain a bare Taiji v10 checkpoint or a model field")


def _course_manifest() -> ContinualCourseManifest:
    phases = [
        CoursePhase(
            phase_id="S",
            source_digest=content_digest({"source": "synthetic-old"}),
            dataset_digest=content_digest({"train": S_TRAIN, "holdout": S_HOLDOUT}),
            train_budget_bytes=len(S_TRAIN),
            holdout_budget_bytes=len(S_HOLDOUT),
            course_seed=71,
            order=1,
        )
    ]
    for order, (old_count, new_count) in enumerate(
        ((8, 2), (6, 4), (4, 6), (2, 8)),
        start=2,
    ):
        train = G_OLD * old_count + G_NEW * new_count
        phases.append(
            CoursePhase(
                phase_id=f"G-{old_count:02d}-{new_count:02d}",
                source_digest=content_digest(
                    {"old": "synthetic-old", "new": "synthetic-new"}
                ),
                dataset_digest=content_digest({"train": train, "holdout": G_HOLDOUT}),
                train_budget_bytes=len(train),
                holdout_budget_bytes=len(G_HOLDOUT),
                course_seed=71,
                order=order,
            )
        )
    return ContinualCourseManifest(course_id="m4v2-r2-sg-smoke", phases=tuple(phases))


def _blended_train(old_count: int, new_count: int) -> bytes:
    return G_OLD * old_count + G_NEW * new_count


def _score(model: Taiji, data: bytes) -> float:
    result = model.score_bytes(data, include_boundary=True, use_memory=False)
    return float(result["mean_surprise"])


def _arm(
    parent: dict[str, Any],
    arm_name: str,
    parent_scores: dict[str, float],
) -> dict[str, Any]:
    model = Taiji.from_checkpoint(copy.deepcopy(parent))
    migration = model.migrate_f1_to_developmental_synapses()
    mode = {
        "slow_only": "slow",
        "fast_only": "fast",
        "fast_replay": "fast_slow",
    }[arm_name]
    model.set_developmental_f1_learning_mode(mode)
    total_train_bytes = 0
    model.learn_bytes(S_TRAIN, epochs=1, learn_fabric=False)
    total_train_bytes += len(S_TRAIN)
    for old_count, new_count in ((8, 2), (6, 4), (4, 6), (2, 8)):
        train = _blended_train(old_count, new_count)
        model.learn_bytes(train, epochs=1, learn_fabric=False)
        total_train_bytes += len(train)

    replay_before = model.developmental_f1_replay_count
    replay_result = None
    if arm_name == "fast_replay":
        replay_result = model.replay_developmental_f1(
            learning_rate_scale=0.25,
            consolidate=True,
            consolidation_rate=1.0,
            clear_fast=True,
            clear_replay=True,
        )

    scores = {
        "S": _score(model, S_HOLDOUT),
        "G": _score(model, G_HOLDOUT),
    }
    checkpoint = model.checkpoint()
    checkpoint_digest = content_digest(checkpoint)
    restored = Taiji.from_checkpoint(copy.deepcopy(checkpoint))
    fresh_restore_matches = content_digest(restored.checkpoint()) == checkpoint_digest
    old_owner_unchanged = (
        content_digest(checkpoint["predictive_context"])
        == content_digest(parent["predictive_context"])
        and content_digest(checkpoint["predictive_readout"])
        == content_digest(parent["predictive_readout"])
    )
    rollback = Taiji.from_checkpoint(copy.deepcopy(parent))
    rollback_scores = {
        "S": _score(rollback, S_HOLDOUT),
        "G": _score(rollback, G_HOLDOUT),
    }
    rollback_matches_parent = rollback_scores == parent_scores
    bundle = model.developmental_f1_bundle
    assert bundle is not None
    return {
        "arm": arm_name,
        "mode": mode,
        "migration": migration,
        "scores": scores,
        "parent_delta": {domain: scores[domain] - parent_scores[domain] for domain in scores},
        "resources": {
            "train_bytes": total_train_bytes,
            "parameter_count": model.parameter_count(),
            "developmental_state_scalars": bundle.state_scalar_count,
            "replay_events_before_sleep": replay_before,
            "replay_events_after_sleep": model.developmental_f1_replay_count,
        },
        "state": {
            "fast_is_zero": bundle.fast_is_zero,
            "fresh_restore_matches": fresh_restore_matches,
            "restored_learning_mode": restored.developmental_f1_learning_mode,
            "old_owner_unchanged": old_owner_unchanged,
            "rollback_matches_parent": rollback_matches_parent,
            "checkpoint_digest": checkpoint_digest,
            "owner_graph_digest": bundle.owner_graph_digest,
        },
        "replay": replay_result,
    }


def run_canary(parent_checkpoint: dict[str, Any] | None = None) -> dict[str, Any]:
    parent = _load_parent(None) if parent_checkpoint is None else copy.deepcopy(parent_checkpoint)
    if Taiji.DEVELOPMENTAL_F1_KEY in parent:
        raise ValueError("R2 canary parent must be the pre-developmental F1 checkpoint")
    parent_model = Taiji.from_checkpoint(copy.deepcopy(parent))
    parent_digest = content_digest(parent)
    parent_scores = {
        "S": _score(parent_model, S_HOLDOUT),
        "G": _score(parent_model, G_HOLDOUT),
    }
    manifest = _course_manifest()
    arms = {
        name: _arm(parent, name, parent_scores)
        for name in ("slow_only", "fast_only", "fast_replay")
    }
    metric_specs = (
        MetricSpec(
            name="mean_surprise",
            direction="lower",
            unit="nats_per_byte",
            baseline_kind="parent",
            domain_id="*",
            critical=True,
            catastrophic_forgetting_threshold=2.0,
        ),
    )
    snapshots = []
    read_only_digest = content_digest({"S": S_HOLDOUT, "G": G_HOLDOUT})
    for arm in arms.values():
        observations = tuple(
            MetricObservation(
                metric_name="mean_surprise",
                domain_id=domain,
                checkpoint_digest=arm["state"]["checkpoint_digest"],
                absolute_value=arm["scores"][domain],
                owner_id=arm["state"]["owner_graph_digest"],
                read_only_input_digest=read_only_digest,
                parent_delta=arm["parent_delta"][domain],
                parent_checkpoint_digest=parent_digest,
            )
            for domain in ("S", "G")
        )
        snapshots.append(
            ContinualEvaluationSnapshot(
                checkpoint_digest=arm["state"]["checkpoint_digest"],
                phase_id="R2-final",
                owner_graph_digest=arm["state"]["owner_graph_digest"],
                read_only_input_digest=read_only_digest,
                observations=observations,
                parent_checkpoint_digest=parent_digest,
                resource_cost=float(arm["resources"]["train_bytes"]),
            )
        )
    scorecard = ContinualScorecard(
        metric_specs=metric_specs,
        snapshots=tuple(snapshots),
        epsilon=0.05,
    )
    scorecard_retention = {
        domain: scorecard.parent_retention("mean_surprise", domain)
        for domain in ("S", "G")
    }
    gates = {
        "course_manifest_round_trip": (
            ContinualCourseManifest.from_payload(manifest.to_payload()).to_payload()
            == manifest.to_payload()
        ),
        "scorecard_round_trip": (
            ContinualScorecard.from_payload(scorecard.to_payload()).to_payload()
            == scorecard.to_payload()
        ),
        "scorecard_non_inferiority_and_catastrophe": all(
            result["non_inferiority"] and not result["catastrophic_forgetting"]
            for result in scorecard_retention.values()
        ),
        "slow_only_fast_zero": arms["slow_only"]["state"]["fast_is_zero"],
        "fast_only_wrote_fast": not arms["fast_only"]["state"]["fast_is_zero"],
        "replay_captured_real_events": arms["fast_replay"]["resources"][
            "replay_events_before_sleep"
        ]
        > 0,
        "replay_consolidated_and_cleared_fast": arms["fast_replay"]["state"][
            "fast_is_zero"
        ],
        "old_f1_owners_unchanged": all(
            arm["state"]["old_owner_unchanged"] for arm in arms.values()
        ),
        "fresh_restore_matches": all(
            arm["state"]["fresh_restore_matches"] for arm in arms.values()
        ),
        "fresh_restore_is_read_only": all(
            arm["state"]["restored_learning_mode"] == "read_only" for arm in arms.values()
        ),
        "rollback_matches_parent": all(
            arm["state"]["rollback_matches_parent"] for arm in arms.values()
        ),
    }
    return {
        "format": CANARY_FORMAT,
        "version": CANARY_VERSION,
        "status": "passed" if all(gates.values()) else "failed",
        "can_promote": False,
        "promotion_reason": "R2 smoke is one CPU course; formal multi-seed promotion is prohibited",
        "parent": {
            "checkpoint_digest": parent_digest,
            "scores": parent_scores,
            "owner_graph": "pre-developmental-f1",
        },
        "course_manifest": manifest.to_payload(),
        "arms": arms,
        "scorecard": scorecard.to_payload(),
        "scorecard_retention": scorecard_retention,
        "gates": gates,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    report = run_canary(_load_parent(args.checkpoint) if args.checkpoint else None)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.report.relative_to(PROJECT_ROOT))
                if args.report.is_relative_to(PROJECT_ROOT)
                else str(args.report),
                "status": report["status"],
                "can_promote": report["can_promote"],
                "gates": report["gates"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
