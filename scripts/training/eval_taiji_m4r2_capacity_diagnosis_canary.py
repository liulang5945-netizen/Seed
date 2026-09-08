"""M4.R2 fixed-capacity versus incremental-capacity diagnosis canary.

The canary is deliberately an experiment artifact, not a default Taiji
runtime feature.  Variant A adapts one active readout from the protected
parent.  Variant B keeps an old readout slot frozen and trains a second,
zero-initialized slot; read-only scoring mixes the two slots using the same
readout-only scorer.  This separates "one slot was overwritten" from "a new
slot can preserve the old surface" before any production routing change.
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
    FoundationTrainingDataset,
    Taiji,
    WorkbenchBoundaryAuthorization,
    WorkbenchTaskBoundary,
)
from taiji.internalization import content_digest  # noqa: E402

FORMAT = "taiji-m4r2-capacity-diagnosis-canary-v1"
VERSION = 1
EXPECTED_SEED = 11
EXPECTED_COHORT_SEEDS = (11, 29, 47)
SCORE_NORMALIZER = math.log(2.0)
ARTIFACT_FORMAT = "taiji-m4r2-capacity-diagnosis-artifact-v1"


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


def _authorization(boundary: WorkbenchTaskBoundary) -> WorkbenchBoundaryAuthorization:
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


def _boundary(label: str) -> tuple[WorkbenchTaskBoundary, WorkbenchBoundaryAuthorization]:
    boundary = WorkbenchTaskBoundary.issue(
        project_id="taiji-m4r2-capacity-canary",
        task_id=f"{label}-task",
        session_id=f"{label}-session",
        language_id="binary-stream",
        capability_snapshot_id="m4r2-capacity-canary-v1",
        capability_ids=("workspace.read",),
        generation_scope="active",
        issued_tick=0,
        ttl_ticks=100_000,
    )
    return boundary, _authorization(boundary)


def _new_slot(model: Taiji, *, seed_offset: int, zero: bool) -> BytePredictiveReadout:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(model.config.seed) + int(seed_offset))
    slot = BytePredictiveReadout(model.config, generator=generator, device=model.device)
    if zero:
        with torch.no_grad():
            slot.synapses.edge_weight.zero_()
            slot.bias.zero_()
    return slot


def _slot_from_payload(model: Taiji, payload: Mapping[str, Any], *, seed_offset: int) -> BytePredictiveReadout:
    slot = _new_slot(model, seed_offset=seed_offset, zero=False)
    slot.load_payload(payload)
    return slot


def _active_slot(model: Taiji) -> BytePredictiveReadout:
    slot = model._active_predictive_readout  # experiment-only introspection
    if not isinstance(slot, BytePredictiveReadout):
        raise RuntimeError("capacity canary requires one mounted active slot")
    return slot


def _slot_digest(slot: BytePredictiveReadout) -> str:
    return content_digest(slot.to_payload())


def _slot_bytes(slot: BytePredictiveReadout) -> int:
    return int(
        slot.synapses.edge_weight.numel() * slot.synapses.edge_weight.element_size()
        + slot.bias.numel() * slot.bias.element_size()
    )


def _owner_digests(model: Taiji) -> dict[str, str]:
    return {
        "fabric": content_digest(model.fabric.to_payload()),
        "predictive_context": content_digest(model.predictive_context.to_payload()),
        "predictive_readout": content_digest(model.predictive_readout.to_payload()),
        "memory": content_digest(model.memory.to_payload()),
    }


def _metadata(dataset: FoundationTrainingDataset) -> dict[str, Any]:
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


def _train_cycle(
    model: Taiji,
    data: bytes,
    *,
    boundary: WorkbenchTaskBoundary,
    authorization: WorkbenchBoundaryAuthorization,
) -> dict[str, float]:
    return model.learn_bytes(
        data,
        epochs=1,
        include_boundary=True,
        reset=True,
        use_memory=False,
        learn_fabric=False,
        learn_predictive_context=False,
        learn_predictive_readout=True,
        boundary=boundary,
        authorization=authorization,
    )


def _mix_probabilities(
    first: torch.Tensor,
    second: torch.Tensor,
) -> torch.Tensor:
    mixed = (first + second) * 0.5
    return mixed / mixed.sum().clamp_min(1e-12)


def _score_readout_only(
    model: Taiji,
    slot: BytePredictiveReadout,
    data: bytes,
) -> dict[str, Any]:
    """Score a slot without episodic/fabric evidence and restore all state."""

    checkpoint = model.checkpoint()
    before_digest = content_digest(checkpoint)
    result: dict[str, Any] | None = None
    try:
        model.reset_dynamics(episode_id="m4r2-score")
        symbols = tuple(model.sensor.symbols(data, include_boundary=True))
        if len(symbols) < 2:
            raise ValueError("capacity canary score needs at least two symbols")
        model.observe(
            symbols[0],
            learn=False,
            readout="predictive",
            use_memory=False,
            use_identity=False,
            _predictive_readout=slot,
        )
        current = slot.probabilities(model.snapshot().motor_context)
        surprise_sum = 0.0
        correct = 0
        for symbol in symbols[1:]:
            target = int(symbol)
            probability = float(current[target].item())
            surprise_sum += -math.log(max(probability, 1e-12))
            correct += int(int(current.argmax().item()) == target)
            model.observe(
                target,
                learn=False,
                readout="predictive",
                use_memory=False,
                use_identity=False,
                _predictive_readout=slot,
            )
            current = slot.probabilities(model.snapshot().motor_context)
        result = {
            "mean_surprise": surprise_sum / max(1, len(symbols) - 1),
            "bpb": surprise_sum / max(1, len(symbols) - 1) / SCORE_NORMALIZER,
            "accuracy": correct / max(1, len(symbols) - 1),
            "observations": len(symbols) - 1,
            "owner": "predictive_readout.slot",
        }
    finally:
        model.restore(checkpoint)
    if result is None:
        raise RuntimeError("readout-only score did not produce a result")
    result["read_only"] = content_digest(model.checkpoint()) == before_digest
    return result


def _score_composite(
    model: Taiji,
    old_slot: BytePredictiveReadout,
    new_slot: BytePredictiveReadout,
    data: bytes,
) -> dict[str, Any]:
    """Score the frozen old slot plus the new slot with a fixed equal mixture."""

    checkpoint = model.checkpoint()
    before_digest = content_digest(checkpoint)
    result: dict[str, Any] | None = None
    try:
        model.reset_dynamics(episode_id="m4r2-composite-score")
        symbols = tuple(model.sensor.symbols(data, include_boundary=True))
        if len(symbols) < 2:
            raise ValueError("capacity canary score needs at least two symbols")
        model.observe(
            symbols[0],
            learn=False,
            readout="predictive",
            use_memory=False,
            use_identity=False,
            _predictive_readout=old_slot,
        )
        context = model.snapshot().motor_context
        current = _mix_probabilities(
            old_slot.probabilities(context),
            new_slot.probabilities(context),
        )
        surprise_sum = 0.0
        correct = 0
        for symbol in symbols[1:]:
            target = int(symbol)
            probability = float(current[target].item())
            surprise_sum += -math.log(max(probability, 1e-12))
            correct += int(int(current.argmax().item()) == target)
            model.observe(
                target,
                learn=False,
                readout="predictive",
                use_memory=False,
                use_identity=False,
                _predictive_readout=old_slot,
            )
            context = model.snapshot().motor_context
            current = _mix_probabilities(
                old_slot.probabilities(context),
                new_slot.probabilities(context),
            )
        result = {
            "mean_surprise": surprise_sum / max(1, len(symbols) - 1),
            "bpb": surprise_sum / max(1, len(symbols) - 1) / SCORE_NORMALIZER,
            "accuracy": correct / max(1, len(symbols) - 1),
            "observations": len(symbols) - 1,
            "owner": "predictive_readout.old_plus_incremental_equal_mix",
        }
    finally:
        model.restore(checkpoint)
    if result is None:
        raise RuntimeError("composite score did not produce a result")
    result["read_only"] = content_digest(model.checkpoint()) == before_digest
    return result


def _cycle_metrics(
    scores: Sequence[dict[str, Any]],
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


def _run_variant_a(
    source_payload: Mapping[str, Any],
    chain: Any,
    *,
    protected_scores: Mapping[str, float],
    boundary: WorkbenchTaskBoundary,
    authorization: WorkbenchBoundaryAuthorization,
) -> tuple[Taiji, dict[str, Any]]:
    model = Taiji.from_checkpoint(source_payload)
    model.clone_protected_predictive_readout_as_active(boundary_digest=boundary.token_digest)
    active_before = _slot_digest(_active_slot(model))
    scores: list[dict[str, Any]] = []
    train_metrics: list[dict[str, float]] = []
    phases = (chain.phase_c, chain.phase_c2, chain.phase_c3)
    for phase in phases:
        train_metrics.append(_train_cycle(model, phase.train, boundary=boundary, authorization=authorization))
        active = _active_slot(model)
        c_score = _score_readout_only(model, active, chain.phase_c.holdout)
        c2_score = _score_readout_only(model, _active_slot(model), chain.phase_c2.holdout)
        c3_score = _score_readout_only(model, _active_slot(model), chain.phase_c3.holdout)
        scores.append(
            {
                "c_bpb": c_score["bpb"],
                "c2_bpb": c2_score["bpb"],
                "c3_bpb": c3_score["bpb"],
                "read_only": all(
                    bool(item["read_only"]) for item in (c_score, c2_score, c3_score)
                ),
            }
        )
    active_after = _slot_digest(_active_slot(model))
    return model, {
        "kind": "fixed_capacity_active_readout",
        "train_metrics": train_metrics,
        "metrics": _cycle_metrics(
            scores,
            protected_c3_bpb=float(protected_scores["c3_bpb"]),
            protected_c_bpb=float(protected_scores["c_bpb"]),
        ),
        "scores": scores,
        "read_only_scoring": all(bool(item["read_only"]) for item in scores),
        "active_slot_before_digest": active_before,
        "active_slot_after_digest": active_after,
        "active_slot_bytes": _slot_bytes(_active_slot(model)),
    }


def _run_variant_b(
    source_payload: Mapping[str, Any],
    chain: Any,
    *,
    protected_scores: Mapping[str, float],
    boundary: WorkbenchTaskBoundary,
    authorization: WorkbenchBoundaryAuthorization,
) -> tuple[Taiji, BytePredictiveReadout, dict[str, Any]]:
    model = Taiji.from_checkpoint(source_payload)
    protected_digest = _slot_digest(model.predictive_readout)
    old_slot = _new_slot(model, seed_offset=7_301, zero=False)
    old_slot.load_payload(model.predictive_readout.to_payload())
    old_before = _slot_digest(old_slot)
    new_slot = _new_slot(model, seed_offset=7_302, zero=True)
    new_before = _slot_digest(new_slot)
    uniform = torch.full(
        (model.config.alphabet_size,),
        1.0 / model.config.alphabet_size,
        device=model.device,
    )
    zero_probabilities = new_slot.probabilities(
        torch.zeros(model.config.motor_context_dim, device=model.device)
    )
    zero_init_uniform = bool(torch.allclose(zero_probabilities, uniform, atol=1e-7, rtol=0.0))
    model.register_active_predictive_readout(new_slot, boundary_digest=boundary.token_digest)
    before_owners = _owner_digests(model)
    scores: list[dict[str, Any]] = []
    train_metrics: list[dict[str, float]] = []
    phases = (chain.phase_c, chain.phase_c2, chain.phase_c3)
    for phase in phases:
        train_metrics.append(_train_cycle(model, phase.train, boundary=boundary, authorization=authorization))
        new_slot = _active_slot(model)
        c_score = _score_composite(model, old_slot, new_slot, chain.phase_c.holdout)
        new_slot = _active_slot(model)
        c2_score = _score_composite(model, old_slot, new_slot, chain.phase_c2.holdout)
        new_slot = _active_slot(model)
        c3_score = _score_composite(model, old_slot, new_slot, chain.phase_c3.holdout)
        scores.append(
            {
                "c_bpb": c_score["bpb"],
                "c2_bpb": c2_score["bpb"],
                "c3_bpb": c3_score["bpb"],
                "read_only": all(
                    bool(item["read_only"]) for item in (c_score, c2_score, c3_score)
                ),
            }
        )
    new_slot = _active_slot(model)
    after_owners = _owner_digests(model)
    return model, old_slot, {
        "kind": "incremental_old_plus_zero_initialized_slot",
        "train_metrics": train_metrics,
        "metrics": _cycle_metrics(
            scores,
            protected_c3_bpb=float(protected_scores["c3_bpb"]),
            protected_c_bpb=float(protected_scores["c_bpb"]),
        ),
        "scores": scores,
        "read_only_scoring": all(bool(item["read_only"]) for item in scores),
        "protected_readout_digest": protected_digest,
        "protected_readout_after_digest": after_owners["predictive_readout"],
        "old_slot_before_digest": old_before,
        "old_slot_after_digest": _slot_digest(old_slot),
        "new_slot_before_digest": new_before,
        "new_slot_after_digest": _slot_digest(new_slot),
        "old_slot_bytes": _slot_bytes(old_slot),
        "new_slot_bytes": _slot_bytes(new_slot),
        "zero_init_uniform": zero_init_uniform,
        "before_owners": before_owners,
        "after_owners": after_owners,
    }


def _save_artifacts(
    *,
    model_a: Taiji,
    model_b: Taiji,
    old_slot: BytePredictiveReadout,
    source_digest: str,
    artifact_dir: Path,
) -> tuple[Path, Path]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    a_path = artifact_dir / "variant_a_seed11.pt"
    b_path = artifact_dir / "variant_b_seed11.pt"
    atomic_save(
        {
            "format": ARTIFACT_FORMAT,
            "version": VERSION,
            "variant": "A",
            "source_checkpoint_digest": source_digest,
            "model": model_a.checkpoint(),
        },
        a_path,
    )
    atomic_save(
        {
            "format": ARTIFACT_FORMAT,
            "version": VERSION,
            "variant": "B",
            "source_checkpoint_digest": source_digest,
            "model": model_b.checkpoint(),
            "old_slot": old_slot.to_payload(),
        },
        b_path,
    )
    return a_path, b_path


def _fresh_restore_scores(
    a_path: Path,
    b_path: Path,
    *,
    probe: bytes,
) -> tuple[dict[str, Any], dict[str, Any], bool, bool]:
    a_artifact = torch.load(a_path, map_location="cpu", weights_only=False)
    b_artifact = torch.load(b_path, map_location="cpu", weights_only=False)
    fresh_a = Taiji.from_checkpoint(a_artifact["model"])
    fresh_b = Taiji.from_checkpoint(b_artifact["model"])
    fresh_a_score = _score_readout_only(fresh_a, _active_slot(fresh_a), probe)
    old_slot = _slot_from_payload(fresh_b, b_artifact["old_slot"], seed_offset=7_301)
    fresh_b_score = _score_composite(fresh_b, old_slot, _active_slot(fresh_b), probe)
    a_round_trip = bool(fresh_a_score["read_only"])
    b_round_trip = bool(fresh_b_score["read_only"])
    return fresh_a_score, fresh_b_score, a_round_trip, b_round_trip


def _lesion_changes(
    model_b: Taiji,
    old_slot: BytePredictiveReadout,
    probe: bytes,
) -> tuple[bool, bool]:
    baseline = _score_composite(model_b, old_slot, _active_slot(model_b), probe)
    old_lesion = _slot_from_payload(model_b, old_slot.to_payload(), seed_offset=7_301)
    with torch.no_grad():
        old_lesion.synapses.edge_weight.zero_()
        old_lesion.bias.zero_()
    old_score = _score_composite(model_b, old_lesion, _active_slot(model_b), probe)
    new_lesion = _slot_from_payload(model_b, _active_slot(model_b).to_payload(), seed_offset=7_302)
    with torch.no_grad():
        new_lesion.synapses.edge_weight.zero_()
        new_lesion.bias.zero_()
    new_score = _score_composite(model_b, old_slot, new_lesion, probe)
    return (
        abs(float(old_score["bpb"]) - float(baseline["bpb"])) > 1e-9,
        abs(float(new_score["bpb"]) - float(baseline["bpb"])) > 1e-9,
    )


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
    protected_scores = {
        "c_bpb": _score_readout_only(source_model, source_model.predictive_readout, chain.phase_c.holdout[:eval_bytes])["bpb"],
        "c3_bpb": _score_readout_only(source_model, source_model.predictive_readout, chain.phase_c3.holdout[:eval_bytes])["bpb"],
    }
    boundary_a, authorization_a = _boundary("m4r2-a")
    boundary_b, authorization_b = _boundary("m4r2-b")
    started = time.perf_counter()
    model_a, variant_a = _run_variant_a(
        source_payload,
        chain,
        protected_scores=protected_scores,
        boundary=boundary_a,
        authorization=authorization_a,
    )
    model_b, old_slot, variant_b = _run_variant_b(
        source_payload,
        chain,
        protected_scores=protected_scores,
        boundary=boundary_b,
        authorization=authorization_b,
    )
    artifact_a, artifact_b = _save_artifacts(
        model_a=model_a,
        model_b=model_b,
        old_slot=old_slot,
        source_digest=source_digest,
        artifact_dir=artifact_dir,
    )
    fresh_a, fresh_b, a_round_trip, b_round_trip = _fresh_restore_scores(
        artifact_a,
        artifact_b,
        probe=chain.phase_c3.holdout[:eval_bytes],
    )
    old_lesion_changed, new_lesion_changed = _lesion_changes(
        model_b,
        old_slot,
        chain.phase_c3.holdout[:eval_bytes],
    )
    detached = Taiji.from_checkpoint(model_b.checkpoint())
    detached.clear_active_predictive_readout()
    rollback = Taiji.from_checkpoint(model_b.checkpoint())
    rollback_active_digest = _slot_digest(_active_slot(rollback))
    final_active_digest = _slot_digest(_active_slot(model_b))
    owner_checks = {
        "a_active_changes": variant_a["active_slot_before_digest"]
        != variant_a["active_slot_after_digest"],
        "b_new_slot_changes": variant_b["new_slot_before_digest"]
        != variant_b["new_slot_after_digest"],
        "b_old_slot_unchanged": variant_b["old_slot_before_digest"]
        == variant_b["old_slot_after_digest"],
        "b_protected_readout_unchanged": variant_b["protected_readout_digest"]
        == variant_b["protected_readout_after_digest"],
        "b_fabric_unchanged": variant_b["before_owners"]["fabric"]
        == variant_b["after_owners"]["fabric"],
        "b_predictive_context_unchanged": variant_b["before_owners"]["predictive_context"]
        == variant_b["after_owners"]["predictive_context"],
        "b_memory_unchanged": variant_b["before_owners"]["memory"]
        == variant_b["after_owners"]["memory"],
    }
    checks = {
        "record_disjoint_chain": all(value == 0 for value in chain.overlap_counts.values()),
        "source_checkpoint_unchanged": content_digest(source_model.checkpoint()) == source_digest,
        "zero_init_uniform": variant_b["zero_init_uniform"],
        **owner_checks,
        "a_read_only_scoring": variant_a["read_only_scoring"],
        "b_read_only_scoring": variant_b["read_only_scoring"],
        "a_checkpoint_round_trip": a_round_trip,
        "b_checkpoint_round_trip": b_round_trip,
        "owner_lesion_changes": old_lesion_changed and new_lesion_changed,
        "slot_detach": detached.active_predictive_readout_metadata is None,
        "slot_rollback": rollback_active_digest == final_active_digest,
        "capacity_bytes_recorded": variant_b["new_slot_bytes"] > 0,
    }
    technical_gate = all(bool(value) for value in checks.values())
    a_metrics = variant_a["metrics"]
    b_metrics = variant_b["metrics"]
    capacity_pressure_supported = bool(
        b_metrics["c3_holdout_gain_bpb"] > 0.0
        and b_metrics["c_cycle2_delta_bpb"] <= 0.0
        and b_metrics["c_cycle3_delta_bpb"] <= 0.0
    )
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
            "scorer": "readout-only equal probability mixture",
        },
        "data_chain": {
            "phase_c": _metadata(chain.phase_c),
            "phase_c2": _metadata(chain.phase_c2),
            "phase_c3": _metadata(chain.phase_c3),
            "overlap_counts": dict(chain.overlap_counts),
        },
        "protected_baseline": protected_scores,
        "variant_a": variant_a,
        "variant_b": variant_b,
        "fresh_restore": {
            "a_probe_bpb": fresh_a["bpb"],
            "b_probe_bpb": fresh_b["bpb"],
            "a_score_round_trip": abs(float(fresh_a["bpb"]) - float(a_metrics["c3_holdout_bpb"])) < 1e-9,
            "b_score_round_trip": abs(float(fresh_b["bpb"]) - float(b_metrics["c3_holdout_bpb"])) < 1e-9,
        },
        "artifacts": {
            "variant_a": str(artifact_a),
            "variant_b": str(artifact_b),
            "variant_b_old_slot_digest": _slot_digest(old_slot),
            "variant_b_new_slot_digest": variant_b["new_slot_after_digest"],
        },
        "checks": checks,
        "technical_gate_all_passed": technical_gate,
        "diagnosis": {
            "capacity_pressure_supported": capacity_pressure_supported,
            "interpretation": (
                "incremental capacity is not supported by this canary: the diagnostic "
                "requires positive new holdout gain and no C retention degradation; "
                "formal validation remains blocked until both conditions hold"
                if not capacity_pressure_supported
                else "canary supports a formal incremental-capacity comparison, but does not promote it"
            ),
        },
        "resources": {
            "elapsed_seconds": time.perf_counter() - started,
            "variant_a_active_slot_bytes": variant_a["active_slot_bytes"],
            "variant_b_old_slot_bytes": variant_b["old_slot_bytes"],
            "variant_b_new_slot_bytes": variant_b["new_slot_bytes"],
            "variant_b_incremental_slot_bytes": variant_b["new_slot_bytes"],
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, default=PROJECT_ROOT / "output" / "taiji-m4r2-capacity-canary")
    parser.add_argument("--train-bytes", type=int, default=4_096)
    parser.add_argument("--eval-bytes", type=int, default=1_024)
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "reports" / "taiji_m4r2_capacity_diagnosis_canary_seed11_20260908.json")
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
                "capacity_pressure_supported": report["diagnosis"]["capacity_pressure_supported"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
