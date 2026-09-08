"""M4.R3 fixed-capacity update-interference attribution canary.

This is an experiment artifact, not a default Taiji runtime mode.  It keeps
the inherited model and the record-disjoint C -> C' -> C'' course fixed while
separating three existing F1 owner paths:

* ``readout_only``: isolated active predictive readout plasticity;
* ``context_only``: protected predictive-context plasticity with the readout
  frozen;
* ``joint``: protected predictive-context and predictive-readout plasticity.

No new capacity, provider, MCP, client write path, or gated temporal candidate
is attached by this canary.
"""

from __future__ import annotations

import argparse
import json
import math
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
    build_disjoint_phase_chain,
)
from seed.persistence import atomic_save  # noqa: E402
from taiji import (  # noqa: E402
    BytePredictiveReadout,
    Taiji,
    WorkbenchBoundaryAuthorization,
    WorkbenchTaskBoundary,
)
from taiji.internalization import content_digest  # noqa: E402

FORMAT = "taiji-m4r3-update-interference-canary-v1"
VERSION = 1
EXPECTED_SEED = 11
EXPECTED_COHORT_SEEDS = (11, 29, 47)
SCORE_NORMALIZER = math.log(2.0)
ARM_NAMES = ("readout_only", "context_only", "joint")


def _load_checkpoint(path: Path) -> tuple[dict[str, Any], Taiji]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("source checkpoint must contain a mapping")
    if "config" not in payload:
        model_payload = payload.get("model")
        if not isinstance(model_payload, Mapping):
            raise ValueError("source checkpoint envelope is missing model payload")
        payload = model_payload
    model = Taiji.from_checkpoint(payload)
    if int(model.config.seed) != EXPECTED_SEED:
        raise ValueError(f"canary expects seed {EXPECTED_SEED}, got {model.config.seed}")
    return dict(payload), model


def _boundary(label: str) -> tuple[WorkbenchTaskBoundary, WorkbenchBoundaryAuthorization]:
    boundary = WorkbenchTaskBoundary.issue(
        project_id="taiji-m4r3-update-interference-canary",
        task_id=f"{label}-task",
        session_id=f"{label}-session",
        language_id="binary-stream",
        capability_snapshot_id="m4r3-update-interference-canary-v1",
        capability_ids=("workspace.read",),
        generation_scope="active",
        issued_tick=0,
        ttl_ticks=100_000,
    )
    authorization = WorkbenchBoundaryAuthorization(
        project_id=boundary.project_id,
        task_id=boundary.task_id,
        session_id=boundary.session_id,
        capability_snapshot_id=boundary.capability_snapshot_id,
        authorized_capability_ids=("workspace.read",),
        active_boundary_digest=boundary.token_digest,
        current_tick=0,
        usage="execute",
    )
    return boundary, authorization


def _read_only_authorization(
    boundary: WorkbenchTaskBoundary,
) -> WorkbenchBoundaryAuthorization:
    # An active branch remains mounted on an open boundary during this
    # experiment.  ``read_only_replay`` is intentionally reserved for closed
    # boundaries; score_bytes is still read-only because it snapshots and
    # restores the complete model around every score.
    return WorkbenchBoundaryAuthorization(
        project_id=boundary.project_id,
        task_id=boundary.task_id,
        session_id=boundary.session_id,
        capability_snapshot_id=boundary.capability_snapshot_id,
        authorized_capability_ids=("workspace.read",),
        active_boundary_digest=boundary.token_digest,
        current_tick=0,
        usage="execute",
    )


def _owner_digests(model: Taiji) -> dict[str, str]:
    return {
        "fabric": content_digest(model.fabric.to_payload()),
        "predictive_context": content_digest(model.predictive_context.to_payload()),
        "predictive_readout": content_digest(model.predictive_readout.to_payload()),
        "memory": content_digest(model.memory.to_payload()),
    }


def _active_readout(model: Taiji) -> BytePredictiveReadout:
    readout = model._active_predictive_readout  # experiment-only introspection
    if not isinstance(readout, BytePredictiveReadout):
        raise RuntimeError("readout-only arm requires an active predictive readout")
    return readout


def _score(
    model: Taiji,
    data: bytes,
    *,
    boundary: WorkbenchTaskBoundary | None = None,
    authorization: WorkbenchBoundaryAuthorization | None = None,
) -> dict[str, Any]:
    before = content_digest(model.checkpoint())
    score = model.score_bytes(
        data,
        boundary=boundary,
        authorization=authorization,
    )
    after = content_digest(model.checkpoint())
    return {
        "bpb": float(score["mean_surprise"]) / SCORE_NORMALIZER,
        "accuracy": float(score["accuracy"]),
        "observations": int(score["observations"]),
        "scope": str(score["scope"]),
        "owner": str(score["owner"]),
        "read_only": before == after,
    }


def _train_phase(
    model: Taiji,
    arm: str,
    data: bytes,
    *,
    boundary: WorkbenchTaskBoundary | None,
    authorization: WorkbenchBoundaryAuthorization | None,
) -> dict[str, float]:
    if arm == "readout_only":
        learn_context = False
        learn_readout = True
    elif arm == "context_only":
        learn_context = True
        learn_readout = False
    elif arm == "joint":
        learn_context = True
        learn_readout = True
    else:
        raise ValueError(f"unsupported arm: {arm}")
    started = time.perf_counter()
    metrics = model.learn_bytes(
        data,
        epochs=1,
        include_boundary=True,
        reset=True,
        use_memory=False,
        learn_fabric=False,
        learn_predictive_context=learn_context,
        learn_predictive_readout=learn_readout,
        boundary=boundary,
        authorization=authorization,
    )
    return {
        "observations": float(metrics["observations"]),
        "online_accuracy": float(metrics["online_accuracy"]),
        "mean_surprise": float(metrics["mean_surprise"]),
        "elapsed_seconds": time.perf_counter() - started,
    }


def _metadata(dataset: Any) -> dict[str, Any]:
    return {
        "digest": dataset.digest,
        "partition_seed": int(dataset.partition_seed),
        "profile": dataset.profile,
        "sample_counts": dataset.sample_counts,
        "source_files": [list(item) for item in dataset.source_files],
        "excluded_dataset_digest": dataset.excluded_dataset_digest,
        "excluded_dataset_digests": list(dataset.excluded_dataset_digests),
        "selected_record_count": len(dataset.selected_record_digests),
    }


def _cycle_metrics(
    scores: Sequence[dict[str, float]],
    *,
    protected_c3_bpb: float,
    protected_c_bpb: float,
) -> dict[str, float]:
    c1, c2, c3 = (float(item["c_bpb"]) for item in scores)
    c2_cycle2, c2_cycle3 = (
        float(scores[1]["c2_bpb"]),
        float(scores[2]["c2_bpb"]),
    )
    return {
        "c3_holdout_bpb": float(scores[2]["c3_bpb"]),
        "c3_holdout_gain_bpb": protected_c3_bpb - float(scores[2]["c3_bpb"]),
        "c_holdout_cycle1_bpb": c1,
        "c_holdout_cycle2_bpb": c2,
        "c_holdout_cycle3_bpb": c3,
        "c_cycle2_delta_bpb": c2 - c1,
        "c_cycle3_delta_bpb": c3 - c2,
        "c2_holdout_cycle2_bpb": c2_cycle2,
        "c2_holdout_cycle3_bpb": c2_cycle3,
        "c2_cycle3_delta_bpb": c2_cycle3 - c2_cycle2,
        "c_retention_gain_bpb": protected_c_bpb - c3,
    }


def _score_phase_triplet(
    model: Taiji,
    chain: Any,
    *,
    eval_bytes: int,
    boundary: WorkbenchTaskBoundary | None,
    authorization: WorkbenchBoundaryAuthorization | None,
) -> dict[str, Any]:
    c_score = _score(
        model,
        chain.phase_c.holdout[:eval_bytes],
        boundary=boundary,
        authorization=authorization,
    )
    c2_score = _score(
        model,
        chain.phase_c2.holdout[:eval_bytes],
        boundary=boundary,
        authorization=authorization,
    )
    c3_score = _score(
        model,
        chain.phase_c3.holdout[:eval_bytes],
        boundary=boundary,
        authorization=authorization,
    )
    return {
        "c_bpb": c_score["bpb"],
        "c2_bpb": c2_score["bpb"],
        "c3_bpb": c3_score["bpb"],
        "read_only": bool(
            c_score["read_only"] and c2_score["read_only"] and c3_score["read_only"]
        ),
        "scope": c_score["scope"],
        "owner": c_score["owner"],
    }


def _run_arm(
    source_payload: Mapping[str, Any],
    chain: Any,
    arm: str,
    *,
    protected_scores: Mapping[str, float],
    train_bytes: int,
    eval_bytes: int,
) -> tuple[Taiji, dict[str, Any]]:
    model = Taiji.from_checkpoint(source_payload)
    boundary: WorkbenchTaskBoundary | None = None
    authorization: WorkbenchBoundaryAuthorization | None = None
    if arm == "readout_only":
        boundary, authorization = _boundary("m4r3-readout")
        model.clone_protected_predictive_readout_as_active(
            boundary_digest=boundary.token_digest
        )
    owner_before = _owner_digests(model)
    active_before = (
        content_digest(_active_readout(model).to_payload())
        if arm == "readout_only"
        else None
    )
    scores: list[dict[str, Any]] = []
    training: list[dict[str, float]] = []
    phases = (chain.phase_c, chain.phase_c2, chain.phase_c3)
    for phase in phases:
        training.append(
            _train_phase(
                model,
                arm,
                phase.train[:train_bytes],
                boundary=boundary,
                authorization=authorization,
            )
        )
        scores.append(
            _score_phase_triplet(
                model,
                chain,
                eval_bytes=eval_bytes,
                boundary=boundary,
                authorization=(
                    None
                    if boundary is None
                    else _read_only_authorization(boundary)
                ),
            )
        )
    owner_after = _owner_digests(model)
    active_after = (
        content_digest(_active_readout(model).to_payload())
        if arm == "readout_only"
        else None
    )
    return model, {
        "arm": arm,
        "training": training,
        "scores": scores,
        "metrics": _cycle_metrics(
            scores,
            protected_c3_bpb=float(protected_scores["c3_bpb"]),
            protected_c_bpb=float(protected_scores["c_bpb"]),
        ),
        "owner_before": owner_before,
        "owner_after": owner_after,
        "active_readout_before": active_before,
        "active_readout_after": active_after,
        "boundary_digest": None if boundary is None else boundary.token_digest,
        "boundary": boundary,
        "authorization": authorization,
        "read_only_scoring": all(bool(item["read_only"]) for item in scores),
    }


def _lesion_score(
    model: Taiji,
    arm: str,
    probe: bytes,
    *,
    boundary: WorkbenchTaskBoundary | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    authorization = None if boundary is None else _read_only_authorization(boundary)
    baseline = _score(model, probe, boundary=boundary, authorization=authorization)
    lesioned = Taiji.from_checkpoint(model.checkpoint())
    if arm == "readout_only":
        readout = _active_readout(lesioned)
        with torch.no_grad():
            readout.synapses.edge_weight.zero_()
            readout.bias.zero_()
    elif arm in {"context_only", "joint"}:
        with torch.no_grad():
            lesioned.predictive_context.recurrent.edge_weight.zero_()
    else:
        raise ValueError(f"unsupported arm: {arm}")
    lesion = _score(lesioned, probe, boundary=boundary, authorization=authorization)
    return baseline, lesion


def _save_arm_artifact(
    artifact_dir: Path,
    arm: str,
    model: Taiji,
    *,
    source_digest: str,
) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / f"{arm}_seed{EXPECTED_SEED}.pt"
    atomic_save(
        {
            "format": FORMAT,
            "version": VERSION,
            "arm": arm,
            "source_checkpoint_digest": source_digest,
            "model": model.checkpoint(),
        },
        path,
    )
    return path


def _fresh_restore_score(
    path: Path,
    probe: bytes,
    *,
    boundary: WorkbenchTaskBoundary | None,
) -> dict[str, Any]:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    model = Taiji.from_checkpoint(artifact["model"])
    authorization = None if boundary is None else _read_only_authorization(boundary)
    return _score(model, probe, boundary=boundary, authorization=authorization)


def run_canary(
    *,
    checkpoint_path: Path,
    corpus_path: Path,
    artifact_dir: Path,
    train_bytes: int,
    eval_bytes: int,
    report_path: Path,
) -> dict[str, Any]:
    if train_bytes <= 0 or eval_bytes <= 0:
        raise ValueError("train_bytes and eval_bytes must be positive")
    source_payload, source_model = _load_checkpoint(checkpoint_path)
    source_digest = content_digest(source_payload)
    chain = build_disjoint_phase_chain(
        [corpus_path],
        cohort_seeds=EXPECTED_COHORT_SEEDS,
        profile="smoke",
    )
    phases = (chain.phase_c, chain.phase_c2, chain.phase_c3)
    for phase in phases:
        if len(phase.train) < train_bytes or len(phase.holdout) < eval_bytes:
            raise ValueError("smoke dataset does not satisfy requested byte budgets")
    protected_c = _score(source_model, chain.phase_c.holdout[:eval_bytes])
    protected_c3 = _score(source_model, chain.phase_c3.holdout[:eval_bytes])
    protected_scores = {
        "c_bpb": protected_c["bpb"],
        "c3_bpb": protected_c3["bpb"],
    }
    started = time.perf_counter()
    models: dict[str, Taiji] = {}
    variants: dict[str, dict[str, Any]] = {}
    for arm in ARM_NAMES:
        model, variant = _run_arm(
            source_payload,
            chain,
            arm,
            protected_scores=protected_scores,
            train_bytes=train_bytes,
            eval_bytes=eval_bytes,
        )
        models[arm] = model
        variants[arm] = variant
    artifacts = {
        arm: _save_arm_artifact(
            artifact_dir,
            arm,
            models[arm],
            source_digest=source_digest,
        )
        for arm in ARM_NAMES
    }
    probe = chain.phase_c3.holdout[:eval_bytes]
    fresh_scores = {
        arm: _fresh_restore_score(
            artifacts[arm],
            probe,
            boundary=variants[arm]["boundary"],
        )
        for arm in ARM_NAMES
    }
    lesion_results: dict[str, dict[str, Any]] = {}
    for arm in ARM_NAMES:
        baseline, lesion = _lesion_score(
            models[arm],
            arm,
            probe,
            boundary=variants[arm]["boundary"],
        )
        lesion_results[arm] = {
            "baseline_bpb": baseline["bpb"],
            "lesion_bpb": lesion["bpb"],
            "changed": abs(float(lesion["bpb"]) - float(baseline["bpb"])) > 1e-9,
        }
    checks: dict[str, bool] = {
        "record_disjoint_chain": all(value == 0 for value in chain.overlap_counts.values()),
        "source_checkpoint_unchanged": content_digest(source_model.checkpoint()) == source_digest,
        "same_train_budget": all(
            len(phase.train) == train_bytes for phase in phases
        ),
        "same_eval_budget": all(
            len(phase.holdout) >= eval_bytes for phase in phases
        ),
    }
    checks.update(
        {
            f"{arm}_read_only_scoring": variants[arm]["read_only_scoring"]
            for arm in ARM_NAMES
        }
    )
    checks.update(
        {
            f"{arm}_owner_lesion_changes": lesion_results[arm]["changed"]
            for arm in ARM_NAMES
        }
    )
    checks.update(
        {
            "readout_only_active_changes": (
                variants["readout_only"]["active_readout_before"]
                != variants["readout_only"]["active_readout_after"]
            ),
            "readout_only_context_unchanged": (
                variants["readout_only"]["owner_before"]["predictive_context"]
                == variants["readout_only"]["owner_after"]["predictive_context"]
            ),
            "readout_only_protected_readout_unchanged": (
                variants["readout_only"]["owner_before"]["predictive_readout"]
                == variants["readout_only"]["owner_after"]["predictive_readout"]
            ),
            "context_only_context_changes": (
                variants["context_only"]["owner_before"]["predictive_context"]
                != variants["context_only"]["owner_after"]["predictive_context"]
            ),
            "context_only_readout_unchanged": (
                variants["context_only"]["owner_before"]["predictive_readout"]
                == variants["context_only"]["owner_after"]["predictive_readout"]
            ),
            "joint_context_changes": (
                variants["joint"]["owner_before"]["predictive_context"]
                != variants["joint"]["owner_after"]["predictive_context"]
            ),
            "joint_readout_changes": (
                variants["joint"]["owner_before"]["predictive_readout"]
                != variants["joint"]["owner_after"]["predictive_readout"]
            ),
        }
    )
    for arm in ARM_NAMES:
        checks[f"{arm}_fabric_unchanged"] = (
            variants[arm]["owner_before"]["fabric"]
            == variants[arm]["owner_after"]["fabric"]
        )
        checks[f"{arm}_memory_unchanged"] = (
            variants[arm]["owner_before"]["memory"]
            == variants[arm]["owner_after"]["memory"]
        )
        final_bpb = float(variants[arm]["metrics"]["c3_holdout_bpb"])
        fresh_bpb = float(fresh_scores[arm]["bpb"])
        checks[f"{arm}_checkpoint_round_trip"] = abs(final_bpb - fresh_bpb) < 1e-9
    technical_gate = all(checks.values())
    report = {
        "format": FORMAT,
        "version": VERSION,
        "generated_at_epoch": time.time(),
        "status": "passed" if technical_gate else "failed",
        "can_promote": False,
        "seed": EXPECTED_SEED,
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_digest": source_digest,
        "corpus": str(corpus_path),
        "configuration": {
            "train_bytes": int(train_bytes),
            "eval_bytes": int(eval_bytes),
            "cohort_seeds": list(EXPECTED_COHORT_SEEDS),
            "profile": "smoke",
            "arms": list(ARM_NAMES),
            "fixed_capacity": True,
            "gated_temporal_candidate": False,
        },
        "data_chain": {
            "phase_c": _metadata(chain.phase_c),
            "phase_c2": _metadata(chain.phase_c2),
            "phase_c3": _metadata(chain.phase_c3),
            "overlap_counts": dict(chain.overlap_counts),
        },
        "protected_baseline": protected_scores,
        "variants": {
            arm: {
                key: value
                for key, value in variant.items()
                if key not in {"boundary", "authorization"}
            }
            for arm, variant in variants.items()
        },
        "fresh_restore": {
            arm: {
                "bpb": fresh_scores[arm]["bpb"],
                "owner": fresh_scores[arm]["owner"],
                "scope": fresh_scores[arm]["scope"],
            }
            for arm in ARM_NAMES
        },
        "lesion": lesion_results,
        "artifacts": {arm: str(path) for arm, path in artifacts.items()},
        "checks": checks,
        "technical_gate_all_passed": technical_gate,
        "diagnosis": {
            "fixed_capacity_attribution_ready": technical_gate,
            "interpretation": (
                "technical attribution canary passed; use the three-arm metrics to "
                "select the next fixed-capacity update rule, but do not promote any arm"
                if technical_gate
                else "technical attribution canary failed; do not interpret arm metrics "
                "until owner and checkpoint contracts are repaired"
            ),
        },
        "resources": {
            "elapsed_seconds": time.perf_counter() - started,
            "train_bytes_per_arm": int(train_bytes * len(phases)),
            "artifact_bytes": {
                arm: int(path.stat().st_size) for arm, path in artifacts.items()
            },
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "taiji-m4r3-update-interference-canary",
    )
    parser.add_argument("--train-bytes", type=int, default=4_096)
    parser.add_argument("--eval-bytes", type=int, default=1_024)
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "taiji_m4r3_update_interference_canary_seed11_20260908.json",
    )
    args = parser.parse_args(argv)
    report = run_canary(
        checkpoint_path=args.checkpoint,
        corpus_path=args.corpus,
        artifact_dir=args.artifact_dir,
        train_bytes=args.train_bytes,
        eval_bytes=args.eval_bytes,
        report_path=args.report,
    )
    print(
        json.dumps(
            {
                "report": str(args.report),
                "status": report["status"],
                "technical_gate_all_passed": report["technical_gate_all_passed"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
