"""M2.R2.R0 fixed-structure learning curves and owner attribution.

This preflight keeps the current Taiji structure unchanged.  It compares a
frozen parent with three explicitly scoped learning paths over the same
record-disjoint phase-C prefix:

* ``active_readout`` writes only the active predictive readout;
* ``predictive_context`` writes only the protected predictive context;
* ``joint_predictive`` writes the protected context and protected readout.

Every point records train/holdout BPB, persistent owner digests, read-only
score checks and an in-process checkpoint round trip.  The tool is diagnostic:
it does not promote a checkpoint or propose a new architecture.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m2r1_phase_c_canary import (  # noqa: E402
    _authorization,
    _process_working_set_bytes,
    build_disjoint_phase_chain,
    load_joint_child,
)
from taiji import (  # noqa: E402
    Taiji,
    WorkbenchBoundaryAuthorization,
    WorkbenchTaskBoundary,
)
from taiji.internalization import content_digest  # noqa: E402

FORMAT = "taiji-r2-learning-curve-v1"
LOG_TWO = 0.6931471805599453
DEFAULT_BUDGETS = (4096, 16384, 65536)
DEFAULT_ARMS = ("frozen", "active_readout", "predictive_context", "joint_predictive")
ALL_ARMS = frozenset(DEFAULT_ARMS)
# Keep the lineage spelling used when the identity-generation children were
# created.  Foundation dataset digests intentionally include the supplied
# corpus path, so changing this to PROJECT_ROOT / ... would silently make an
# existing child appear unrelated even when the file bytes are identical.
DEFAULT_CORPUS = Path("data") / "simple_zh" / "dialogue_extended_clean.jsonl"


def _owner_digests(model: Taiji) -> dict[str, str]:
    """Return stable digests for every owner relevant to R2 attribution."""

    registry = model.readout_registry_status()
    active = registry.get("active")
    return {
        "fabric": content_digest(model.fabric.to_payload()),
        "motor": content_digest(model.motor.to_payload()),
        "memory": content_digest(model.memory.to_payload()),
        "predictive_context": content_digest(model.predictive_context.to_payload()),
        "protected_predictive_readout": str(
            registry["protected"]["readout_digest"]
        ),
        "active_predictive_readout": ""
        if not isinstance(active, Mapping)
        else str(active.get("readout_digest", "")),
    }


def owner_attribution_check(
    arm: str,
    before: Mapping[str, str],
    after: Mapping[str, str],
) -> dict[str, Any]:
    """Classify expected and unexpected persistent owner changes.

    A point passes when no owner outside the declared write set changes.  The
    ``required_owner_changed`` flag is reported separately: a tiny budget may
    legitimately produce no effective update, and that is evidence for the
    learning curve rather than a broken ownership contract.
    """

    write_sets = {
        "frozen": set(),
        "active_readout": {"active_predictive_readout"},
        "predictive_context": {"predictive_context"},
        "joint_predictive": {
            "predictive_context",
            "protected_predictive_readout",
        },
    }
    if arm not in write_sets:
        raise ValueError(f"unsupported R2 arm: {arm}")
    changed = {
        owner: str(before.get(owner, "")) != str(after.get(owner, ""))
        for owner in sorted(set(before) | set(after))
    }
    expected = write_sets[arm]
    unexpected = sorted(owner for owner, value in changed.items() if value and owner not in expected)
    expected_changed = sorted(owner for owner in expected if changed.get(owner, False))
    return {
        "changed": changed,
        "expected_writable_owners": sorted(expected),
        "changed_expected_owners": expected_changed,
        "unexpected_changed_owners": unexpected,
        "contract_passed": not unexpected,
        "required_owner_changed": bool(expected) and bool(expected_changed),
    }


def _issue_boundary(*, arm: str, scope: str, seed: int, budget: int) -> WorkbenchTaskBoundary:
    return WorkbenchTaskBoundary.issue(
        project_id="project:seed",
        task_id=f"task:r2-curve-{arm}-{budget}",
        session_id=f"session:r2-seed{seed}",
        language_id="python",
        capability_snapshot_id="capability:snapshot:r2",
        capability_ids=("workspace.read",),
        generation_scope=scope,
        issued_tick=10,
        ttl_ticks=80,
    )


def _score(
    model: Taiji,
    data: bytes,
    *,
    boundary: WorkbenchTaskBoundary,
    authorization: WorkbenchBoundaryAuthorization,
) -> dict[str, Any]:
    before = content_digest(model.checkpoint())
    started = time.perf_counter()
    result = model.score_bytes(
        data,
        boundary=boundary,
        authorization=authorization,
    )
    elapsed = time.perf_counter() - started
    after = content_digest(model.checkpoint())
    return {
        "bpb": float(result["mean_surprise"]) / LOG_TWO,
        "accuracy": float(result["accuracy"]),
        "observations": int(result["observations"]),
        "owner": str(result["owner"]),
        "scope": str(result["scope"]),
        "persistent_digest_before": before,
        "persistent_digest_after": after,
        "read_only": before == after,
        "seconds": float(elapsed),
    }


def _validate_source_lineage(
    payload: Mapping[str, Any],
    *,
    phase_a_digest: str,
    phase_b_digest: str,
) -> None:
    if str(payload.get("protected_dataset_digest")) != phase_a_digest:
        raise ValueError("source protected_dataset_digest does not match phase-A lineage")
    if str(payload.get("dataset_digest")) != phase_b_digest:
        raise ValueError("source dataset_digest does not match phase-B lineage")


def _preflight_checkpoint(model: Taiji, path: Path) -> dict[str, Any]:
    """Prove disk save/restore before the first curve update."""

    path.parent.mkdir(parents=True, exist_ok=True)
    source_payload = model.checkpoint()
    source_digest = content_digest(source_payload)
    torch.save(source_payload, path)
    try:
        restored_payload = torch.load(path, map_location="cpu", weights_only=False)
        restored = Taiji.from_checkpoint(restored_payload)
        restored_digest = content_digest(restored.checkpoint())
        return {
            "saved_before_training": True,
            "path": str(path),
            "source_digest": source_digest,
            "restored_digest": restored_digest,
            "matching_digest": source_digest == restored_digest,
            "bytes": int(path.stat().st_size),
        }
    finally:
        path.unlink(missing_ok=True)


def _run_point(
    *,
    source_payload: Mapping[str, Any],
    data: bytes,
    holdout: bytes,
    arm: str,
    budget: int,
    seed: int,
    baseline_holdout_bpb: float,
) -> dict[str, Any]:
    model = Taiji.from_checkpoint(source_payload)
    before = _owner_digests(model)
    active_boundary: WorkbenchTaskBoundary | None = None
    active_authorization: WorkbenchBoundaryAuthorization | None = None
    if arm == "active_readout":
        active_boundary = _issue_boundary(
            arm=arm,
            scope="active",
            seed=seed,
            budget=budget,
        )
        active_authorization = _authorization(active_boundary)
        model.clone_protected_predictive_readout_as_active(
            boundary_digest=active_boundary.token_digest,
        )
        score_boundary = active_boundary
        score_authorization = active_authorization
    else:
        score_boundary = _issue_boundary(
            arm=arm,
            scope="protected",
            seed=seed,
            budget=budget,
        )
        score_authorization = _authorization(score_boundary)

    # Active readout registration is a routing operation, not a learning
    # update.  Start owner attribution after registration so the curve does
    # not mistake the cloned active slot for evidence that training wrote it.
    training_before = _owner_digests(model)

    started = time.perf_counter()
    train_metrics: dict[str, float] | None = None
    if arm != "frozen":
        train_metrics = model.learn_bytes(
            data[:budget],
            epochs=1,
            include_start_boundary=True,
            include_end_boundary=True,
            reset=True,
            use_memory=False,
            learn_fabric=False,
            learn_predictive_context=arm in {"predictive_context", "joint_predictive"},
            learn_predictive_readout=arm in {"active_readout", "joint_predictive"},
            boundary=active_boundary,
            authorization=active_authorization,
        )
    train_seconds = time.perf_counter() - started
    after = _owner_digests(model)
    attribution = owner_attribution_check(arm, training_before, after)

    train_score = _score(
        model,
        data[:budget],
        boundary=score_boundary,
        authorization=score_authorization,
    )
    holdout_score = _score(
        model,
        holdout,
        boundary=score_boundary,
        authorization=score_authorization,
    )
    protected_holdout: dict[str, Any] | None = None
    if arm == "active_readout":
        protected_boundary = _issue_boundary(
            arm=arm,
            scope="protected",
            seed=seed,
            budget=budget,
        )
        protected_holdout = _score(
            model,
            holdout,
            boundary=protected_boundary,
            authorization=_authorization(protected_boundary),
        )

    payload = model.checkpoint()
    checkpoint_digest = content_digest(payload)
    restored = Taiji.from_checkpoint(payload)
    restored_digest = content_digest(restored.checkpoint())
    checkpoint_path = _temporary_checkpoint_path(model)
    checkpoint_bytes = checkpoint_path.stat().st_size
    score_checks = [
        bool(train_score["read_only"]),
        bool(holdout_score["read_only"]),
    ]
    if protected_holdout is not None:
        score_checks.append(bool(protected_holdout["read_only"]))
    checks = {
        "owner_contract": bool(attribution["contract_passed"]),
        "scores_are_read_only": all(score_checks),
        "checkpoint_round_trip": checkpoint_digest == restored_digest,
        "active_route_is_active": arm != "active_readout"
        or holdout_score["scope"] == "active",
        "protected_route_is_protected": arm == "active_readout"
        or holdout_score["scope"] == "protected",
    }
    current_working_set = _process_working_set_bytes()
    return {
        "arm": arm,
        "budget_bytes": int(budget),
        "train": {
            "metrics": train_metrics,
            "seconds": float(train_seconds),
            "working_set_bytes": current_working_set,
        },
        "train_score": train_score,
        "holdout_score": holdout_score,
        "protected_holdout_score": protected_holdout,
        "baseline_holdout_bpb": float(baseline_holdout_bpb),
        "holdout_gain_bpb": float(baseline_holdout_bpb - holdout_score["bpb"]),
        "owners": {
            "before": before,
            "training_before": training_before,
            "after": after,
            "attribution": attribution,
        },
        "checkpoint": {
            "digest": checkpoint_digest,
            "restored_digest": restored_digest,
            "bytes": int(checkpoint_bytes),
        },
        "checks": checks,
    }


def _temporary_checkpoint_path(model: Taiji) -> Path:
    """Return a short-lived path for measuring serialized checkpoint size."""

    path = PROJECT_ROOT / "output" / "taiji-m2r2-curve-point.pt"
    torch.save(model.checkpoint(), path)
    return path


def run_learning_curve(
    checkpoint: Path,
    *,
    corpus_paths: Sequence[Path],
    seed: int,
    budgets: Sequence[int] = DEFAULT_BUDGETS,
    eval_bytes: int = 4096,
    arms: Sequence[str] = DEFAULT_ARMS,
) -> dict[str, Any]:
    normalized_budgets = tuple(sorted(dict.fromkeys(int(value) for value in budgets)))
    if not normalized_budgets or any(value <= 0 for value in normalized_budgets):
        raise ValueError("budgets must contain positive values")
    normalized_arms = tuple(dict.fromkeys(str(value) for value in arms))
    if not normalized_arms or any(value not in ALL_ARMS for value in normalized_arms):
        raise ValueError(f"arms must be drawn from {sorted(ALL_ARMS)}")

    payload, source_model = load_joint_child(checkpoint, expected_seed=seed)
    chain = build_disjoint_phase_chain(corpus_paths, cohort_seeds=(11, 29, 47))
    phase_a = chain.phase_a_by_seed[int(seed)]
    phase_b = chain.phase_b_by_seed[int(seed)]
    _validate_source_lineage(
        payload,
        phase_a_digest=phase_a.digest,
        phase_b_digest=phase_b.digest,
    )
    if max(normalized_budgets) > len(chain.phase_c.train):
        raise ValueError("largest budget exceeds phase-C training bytes")
    if eval_bytes <= 0 or eval_bytes > len(chain.phase_c.holdout):
        raise ValueError("eval_bytes must fit the phase-C holdout prefix")

    source_digest = content_digest(source_model.checkpoint())
    preflight = _preflight_checkpoint(
        source_model,
        PROJECT_ROOT / "output" / "taiji-m2r2-curve-preflight.pt",
    )
    if not preflight["matching_digest"]:
        raise RuntimeError("source checkpoint save/restore preflight failed")

    holdout = chain.phase_c.holdout[:eval_bytes]
    baseline_model = Taiji.from_checkpoint(source_model.checkpoint())
    baseline_boundary = _issue_boundary(
        arm="baseline",
        scope="protected",
        seed=seed,
        budget=0,
    )
    baseline_score = _score(
        baseline_model,
        holdout,
        boundary=baseline_boundary,
        authorization=_authorization(baseline_boundary),
    )

    points: list[dict[str, Any]] = []
    for arm in normalized_arms:
        for budget in normalized_budgets:
            point = _run_point(
                source_payload=source_model.checkpoint(),
                data=chain.phase_c.train,
                holdout=holdout,
                arm=arm,
                budget=budget,
                seed=seed,
                baseline_holdout_bpb=float(baseline_score["bpb"]),
            )
            temporary = PROJECT_ROOT / "output" / "taiji-m2r2-curve-point.pt"
            temporary.unlink(missing_ok=True)
            points.append(point)

    all_passed = all(
        bool(value) for point in points for value in point["checks"].values()
    )
    return {
        "format": FORMAT,
        "version": 1,
        "status": "passed" if all_passed else "failed",
        "can_promote": False,
        "source_checkpoint": str(checkpoint),
        "source_digest": source_digest,
        "seed": int(seed),
        "budgets": list(normalized_budgets),
        "eval_bytes": int(eval_bytes),
        "arms": list(normalized_arms),
        "preflight": preflight,
        "datasets": {
            "phase_a": {"digest": phase_a.digest, "selected_record_count": len(phase_a.selected_record_digests)},
            "phase_b": {"digest": phase_b.digest, "selected_record_count": len(phase_b.selected_record_digests)},
            "phase_c": {"digest": chain.phase_c.digest, "selected_record_count": len(chain.phase_c.selected_record_digests)},
            "overlap_counts": dict(chain.overlap_counts),
            "record_disjoint": not any(chain.overlap_counts.values()),
        },
        "baseline": {
            "holdout_score": baseline_score,
            "holdout_bpb": float(baseline_score["bpb"]),
        },
        "points": points,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "output" / "taiji-m2s-seed11-identity-generation-20260905" / "last.pt",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        nargs="+",
        default=[DEFAULT_CORPUS],
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    parser.add_argument("--eval-bytes", type=int, default=4096)
    parser.add_argument("--arms", nargs="+", choices=sorted(ALL_ARMS), default=list(DEFAULT_ARMS))
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="JSON report path",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = run_learning_curve(
        args.checkpoint,
        corpus_paths=args.corpus,
        seed=args.seed,
        budgets=args.budgets,
        eval_bytes=args.eval_bytes,
        arms=args.arms,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "report": str(args.report),
                "status": report["status"],
                "points": len(report["points"]),
                "technical_checks_passed": sum(
                    bool(value)
                    for point in report["points"]
                    for value in point["checks"].values()
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
