"""M2.R1 phase-C continuation arms on the confirmed B2-growth child (m2s seed11).

Runs the four pre-registered treatment arms over the same parent, the same
phase-C data slice and the same evaluation slice:

- ``no_update``: frozen parent baseline, never trained;
- ``protected_only``: F1 predictive readout + private temporal context plastic
  on phase-C, shared fabric frozen (no boundary);
- ``active_only``: an isolated active readout branch trained on phase-C under
  a Workbench boundary, shared fabric/context/protected readout frozen;
- ``replay``: active_only followed by an equal-bytes exact train-only replay of
  the phase-A training prefix.

Every arm re-verifies read-only scoring, owner audits and (for the branch
arms) fresh-process checkpoint round-trips, so a negative or zero capability
number cannot be blamed on a broken measurement pipe.  Capability numbers are
recorded per arm; the R1 gate treats reliable negative results as acceptable
and does not re-roll the data seed to force a pass.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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

FORMAT = "taiji-r1-phase-c-arms-v2"
C_PARTITION_SEED = 43
C2_PARTITION_SEED = 44
PHASE_B_SEED_OFFSET = 10_000
COHORT_SEEDS = (11, 29, 47)
ALL_ARMS = ("no_update", "protected_only", "active_only", "replay", "cascade")
CHECKPOINT_OUTPUT_DIR = PROJECT_ROOT / "output" / "taiji-m2r1-phase-c-actives-v2"


@dataclass(frozen=True)
class M2R1PhaseChain:
    """Content-addressed, record-disjoint continuation courses.

    Each evaluated source has its own phase-A/phase-B lineage because the
    existing children were trained with seed-specific partition addresses.
    Phase C and C' are shared across the cohort, but exclude the union of all
    source lineages so no child can receive a record it already consumed.
    """

    lineage_seeds: tuple[int, ...]
    phase_a_by_seed: dict[int, FoundationTrainingDataset]
    phase_b_by_seed: dict[int, FoundationTrainingDataset]
    phase_c: FoundationTrainingDataset
    phase_c2: FoundationTrainingDataset
    overlap_counts: dict[str, int]


def _dataset_metadata(dataset: FoundationTrainingDataset) -> dict[str, Any]:
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


def _record_set(dataset: FoundationTrainingDataset) -> set[str]:
    if not dataset.selected_record_digests:
        raise ValueError(
            "M2.R1 data contract requires record provenance on every phase dataset"
        )
    return set(dataset.selected_record_digests)


def build_disjoint_phase_chain(
    corpus_paths: Sequence[Path],
    *,
    cohort_seeds: Sequence[int] = COHORT_SEEDS,
    phase_c_seed: int = C_PARTITION_SEED,
    phase_c2_seed: int = C2_PARTITION_SEED,
    profile: str = "foundation",
) -> M2R1PhaseChain:
    """Build the A/B lineage and record-disjoint C/C' continuation chain.

    Different partition seeds alone are not a novelty guarantee.  A and B are
    rebuilt with their real source lineage, then C/C' exclude the selected
    record digests from every cohort lineage.  The function fails before any
    model training if the record-level boundary is not closed.
    """

    normalized_seeds = tuple(dict.fromkeys(int(seed) for seed in cohort_seeds))
    if not normalized_seeds or any(seed <= 0 for seed in normalized_seeds):
        raise ValueError("cohort_seeds must contain positive values")
    if int(phase_c_seed) <= 0 or int(phase_c2_seed) <= 0:
        raise ValueError("phase C partition seeds must be positive")
    if not corpus_paths:
        raise ValueError("M2.R1 data contract needs at least one corpus path")

    phase_a_by_seed: dict[int, FoundationTrainingDataset] = {}
    phase_b_by_seed: dict[int, FoundationTrainingDataset] = {}
    for seed in normalized_seeds:
        phase_a = FoundationTrainingDataset.from_jsonl(
            corpus_paths,
            profile=profile,
            partition_seed=seed,
            track_record_digests=True,
        )
        phase_b = FoundationTrainingDataset.from_jsonl(
            corpus_paths,
            profile=profile,
            partition_seed=PHASE_B_SEED_OFFSET + seed,
            exclude_dataset=phase_a,
            track_record_digests=True,
        )
        phase_a_by_seed[seed] = phase_a
        phase_b_by_seed[seed] = phase_b

    lineage_datasets: list[tuple[str, FoundationTrainingDataset]] = []
    for seed in normalized_seeds:
        lineage_datasets.extend(
            (
                (f"phase_a_seed{seed}", phase_a_by_seed[seed]),
                (f"phase_b_seed{seed}", phase_b_by_seed[seed]),
            )
        )
    lineage_exclusions = tuple(dataset for _name, dataset in lineage_datasets)
    phase_c = FoundationTrainingDataset.from_jsonl(
        corpus_paths,
        profile=profile,
        partition_seed=phase_c_seed,
        exclude_datasets=lineage_exclusions,
        track_record_digests=True,
    )
    phase_c2 = FoundationTrainingDataset.from_jsonl(
        corpus_paths,
        profile=profile,
        partition_seed=phase_c2_seed,
        exclude_datasets=lineage_exclusions + (phase_c,),
        track_record_digests=True,
    )

    overlap_counts: dict[str, int] = {}
    record_sets = {
        name: _record_set(dataset)
        for name, dataset in (*lineage_datasets, ("phase_c", phase_c), ("phase_c2", phase_c2))
    }
    for seed in normalized_seeds:
        key = f"phase_a_seed{seed}__vs__phase_b_seed{seed}"
        overlap_counts[key] = len(
            record_sets[f"phase_a_seed{seed}"] & record_sets[f"phase_b_seed{seed}"]
        )
    for course_name in ("phase_c", "phase_c2"):
        for lineage_name, _dataset in lineage_datasets:
            key = f"{lineage_name}__vs__{course_name}"
            overlap_counts[key] = len(record_sets[lineage_name] & record_sets[course_name])
    overlap_counts["phase_c__vs__phase_c2"] = len(
        record_sets["phase_c"] & record_sets["phase_c2"]
    )
    overlap_failures = {key: value for key, value in overlap_counts.items() if value}
    if overlap_failures:
        raise ValueError(f"M2.R1 record-disjoint Gate failed: {overlap_failures}")

    return M2R1PhaseChain(
        lineage_seeds=normalized_seeds,
        phase_a_by_seed=phase_a_by_seed,
        phase_b_by_seed=phase_b_by_seed,
        phase_c=phase_c,
        phase_c2=phase_c2,
        overlap_counts=overlap_counts,
    )


def _phase_chain_metadata(chain: M2R1PhaseChain) -> dict[str, Any]:
    return {
        "cohort_seeds": list(chain.lineage_seeds),
        "phase_b_seed_offset": PHASE_B_SEED_OFFSET,
        "phase_a_by_seed": {
            str(seed): _dataset_metadata(chain.phase_a_by_seed[seed])
            for seed in chain.lineage_seeds
        },
        "phase_b_by_seed": {
            str(seed): _dataset_metadata(chain.phase_b_by_seed[seed])
            for seed in chain.lineage_seeds
        },
        "phase_c": _dataset_metadata(chain.phase_c),
        "phase_c2": _dataset_metadata(chain.phase_c2),
        "overlap_counts": dict(chain.overlap_counts),
        "record_disjoint": not any(chain.overlap_counts.values()),
    }


def load_joint_child(path: Path, *, expected_seed: int) -> tuple[dict[str, Any], Taiji]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("joint child checkpoint must contain a mapping")
    expected_digest = content_digest(
        {key: value for key, value in payload.items() if key != "checkpoint_digest"}
    )
    if str(payload.get("checkpoint_digest", "")) != expected_digest:
        raise ValueError("joint child checkpoint digest mismatch")
    if payload.get("format") != "taiji-native-joint-training-v1":
        raise ValueError("R1 continuation requires a joint training checkpoint")
    if int(payload.get("version", -1)) < 4:
        raise ValueError("R1 continuation requires a v4 private-context child")
    phases = payload.get("training_phases")
    if not isinstance(phases, list) or not phases:
        raise ValueError("R1 continuation requires a joint-trained child")
    if "sequence" not in phases and payload.get("sequence_fabric_learning") is not False:
        raise ValueError("R1 continuation requires a sequence-trained or sequence-derived child")
    if payload.get("sequence_fabric_learning") is not False:
        raise ValueError("R1 continuation requires frozen shared fabric")
    if payload.get("sequence_predictive_context_mode") != "private-plastic-temporal-v1":
        raise ValueError("R1 continuation requires private predictive context")
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
            parent_checkpoint_digest="r1-phase-c-arms"
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


def _score_read_only(
    model: Taiji,
    data: bytes,
    *,
    boundary: WorkbenchTaskBoundary | None = None,
    authorization: WorkbenchBoundaryAuthorization | None = None,
) -> tuple[float, bool, str, str]:
    before = content_digest(model.checkpoint())
    score = model.score_bytes(
        data,
        boundary=boundary,
        authorization=authorization,
    )
    after = content_digest(model.checkpoint())
    return (
        float(score["mean_surprise"]) / 0.6931471805599453,
        before == after,
        str(score["owner"]),
        str(score["scope"]),
    )


def _fresh_process_digest(
    path: Path,
    probe: bytes,
    length: int,
    *,
    boundary_payload: Mapping[str, Any] | None = None,
    authorization_payload: Mapping[str, Any] | None = None,
) -> tuple[str, str, float]:
    boundary_arg = (
        "None"
        if boundary_payload is None
        else f"WorkbenchTaskBoundary.from_payload({boundary_payload!r})"
    )
    auth_arg = (
        "None"
        if authorization_payload is None
        else f"WorkbenchBoundaryAuthorization(**{authorization_payload!r})"
    )
    probe_code = (
        "import sys, torch; "
        "from taiji import Taiji, WorkbenchTaskBoundary, WorkbenchBoundaryAuthorization; "
        "from taiji.internalization import content_digest; "
        "payload=torch.load(sys.argv[1], map_location='cpu', weights_only=False); "
        "m=Taiji.from_checkpoint(payload); "
        f"boundary={boundary_arg}; authorization={auth_arg}; "
        "print(content_digest(m.checkpoint())); "
        f"print(m.generate({probe!r}, {int(length)}, boundary=boundary, authorization=authorization).hex())"
    )
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-c", probe_code, str(path)],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    stdout = completed.stdout.strip().splitlines()
    return stdout[0].strip(), stdout[1].strip(), time.perf_counter() - started


def _run_branch_arm(
    *,
    arm: str,
    source_model: Taiji,
    c_train: bytes,
    a_train: bytes,
    c_holdout: bytes,
    a_retention: bytes,
    epochs: int,
    replay_epochs: int,
    replay_bytes: int | None = None,
    seed: int = 11,
) -> dict[str, Any]:
    """Train one branch arm and return its capability + technical checks."""

    boundary = WorkbenchTaskBoundary.issue(
        project_id="project:seed",
        task_id=f"task:r1-phase-c-{arm}",
        session_id="session:seed11",
        language_id="python",
        capability_snapshot_id="capability:snapshot:1",
        capability_ids=("workspace.read",),
        generation_scope="active",
        issued_tick=10,
        ttl_ticks=40,
    )
    authorization = _authorization(boundary)
    protected_boundary = WorkbenchTaskBoundary.issue(
        project_id="project:seed",
        task_id=f"task:r1-phase-c-{arm}-protected",
        session_id="session:seed11",
        language_id="python",
        capability_snapshot_id="capability:snapshot:1",
        capability_ids=("workspace.read",),
        generation_scope="protected",
        issued_tick=10,
        ttl_ticks=40,
    )
    protected_authorization = _authorization(protected_boundary)

    model = Taiji.from_checkpoint(source_model.checkpoint())
    shared_before = content_digest(model.fabric.to_payload())
    protected_owners_before = _protected_owner_digest(model)
    protected_readout_before = model.readout_registry_status()["protected"]["readout_digest"]
    model.clone_protected_predictive_readout_as_active(
        boundary_digest=boundary.token_digest,
    )
    active_before = model.active_predictive_readout_metadata
    if active_before is None:
        raise RuntimeError("active readout was not registered")

    model.learn_bytes(
        c_train,
        epochs=epochs,
        include_boundary=True,
        use_memory=False,
        learn_fabric=False,
        learn_predictive_context=False,
        learn_predictive_readout=True,
        boundary=boundary,
        authorization=authorization,
    )
    if arm == "replay":
        replay_slice = a_train if replay_bytes is None else a_train[:replay_bytes]
        if not replay_slice:
            raise ValueError("replay byte slice is empty")
        model.learn_bytes(
            replay_slice,
            epochs=replay_epochs,
            include_boundary=True,
            use_memory=False,
            learn_fabric=False,
            learn_predictive_context=False,
            learn_predictive_readout=True,
            boundary=boundary,
            authorization=authorization,
        )
    active_after = model.active_predictive_readout_metadata
    if active_after is None:
        raise RuntimeError("active readout disappeared after continuation")

    protected_c_bpb, protected_c_ro, p_owner, p_scope = _score_read_only(
        model,
        c_holdout,
        boundary=protected_boundary,
        authorization=protected_authorization,
    )
    active_c_bpb, active_c_ro, a_owner, a_scope = _score_read_only(
        model,
        c_holdout,
        boundary=boundary,
        authorization=authorization,
    )
    active_a_bpb, active_a_ro, _, _ = _score_read_only(
        model,
        a_retention,
        boundary=boundary,
        authorization=authorization,
    )
    route = model.last_generation_route

    active_checkpoint = model.checkpoint()
    active_checkpoint_digest = content_digest(active_checkpoint)
    restored = Taiji.from_checkpoint(copy.deepcopy(active_checkpoint))
    restored_digest = content_digest(restored.checkpoint())
    restored_metadata = restored.active_predictive_readout_metadata
    active_canary = model.generate(
        b"Taiji", 16, boundary=boundary, authorization=authorization
    ).hex()
    restored_canary = restored.generate(
        b"Taiji", 16, boundary=boundary, authorization=authorization
    ).hex()
    disk_path = PROJECT_ROOT / "output" / f"taiji_r1_{arm}_probe.pt"
    try:
        torch.save(active_checkpoint, disk_path)
        fresh_digest, fresh_canary, restore_seconds = _fresh_process_digest(
            disk_path,
            b"Taiji",
            16,
            boundary_payload=boundary.to_payload(),
            authorization_payload=authorization.to_payload(),
        )
        checkpoint_bytes = disk_path.stat().st_size
    finally:
        disk_path.unlink(missing_ok=True)

    # The accepted active checkpoint is a primary artifact, not just a probe:
    # persist it under the run output directory so later inventory/re-evaluation
    # (R0.6/R0.7) can audit it from disk.
    CHECKPOINT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_path = CHECKPOINT_OUTPUT_DIR / f"seed{seed}_{arm}_c{len(c_train)}_r{replay_bytes or 0}.pt"
    torch.save(active_checkpoint, final_path)
    saved_final_digest = content_digest(
        torch.load(final_path, map_location="cpu", weights_only=False)
    )

    checks = {
        "active_readout_changes": active_before["readout_digest"]
        != active_after["readout_digest"],
        "active_readout_changes_during_replay": (
            arm != "replay" or active_after["readout_digest"] != active_before["readout_digest"]
        ),
        "shared_fabric_unchanged": content_digest(model.fabric.to_payload()) == shared_before,
        "protected_owners_unchanged": _protected_owner_digest(model)
        == protected_owners_before,
        "protected_readout_unchanged": protected_readout_before
        == model.readout_registry_status()["protected"]["readout_digest"],
        "scores_are_read_only": all(
            (protected_c_ro, active_c_ro, active_a_ro)
        ),
        "active_score_uses_active_owner": a_owner == "predictive_readout.active"
        and a_scope == "active",
        "protected_score_uses_protected_owner": p_owner == "predictive_readout"
        and p_scope == "protected",
        "registry_round_trips_in_process": (
            restored_metadata == active_after and restored_digest == active_checkpoint_digest
        ),
        "registry_round_trips_in_fresh_process": fresh_digest == active_checkpoint_digest,
        "generation_canary_round_trips": active_canary == restored_canary,
        "fresh_process_generation_canary_matches": fresh_canary == active_canary,
        "generation_route_records_active_owner": route is not None
        and route["generation_scope"] == "active"
        and route["readout_owner"] == "predictive_readout.active",
        "final_checkpoint_persisted_with_matching_digest": saved_final_digest
        == active_checkpoint_digest,
    }
    return {
        "arm": arm,
        "checks": checks,
        "capability": {
            "protected_c_holdout_bpb": protected_c_bpb,
            "active_c_holdout_bpb": active_c_bpb,
            "active_a_retention_bpb": active_a_bpb,
            "c_holdout_gain_bpb": protected_c_bpb - active_c_bpb,
        },
        "round_trip": {
            "active_checkpoint_digest": active_checkpoint_digest,
            "fresh_process_digest": fresh_digest,
            "final_checkpoint_path": str(final_path),
            "saved_final_digest": saved_final_digest,
            "checkpoint_bytes": int(checkpoint_bytes),
            "fresh_process_restore_seconds": restore_seconds,
        },
    }


def _run_cascade_arm(
    *,
    source_model: Taiji,
    c_train: bytes,
    c2_train: bytes,
    c_holdout: bytes,
    c2_holdout: bytes,
    a_retention: bytes,
    epochs: int,
    seed: int = 11,
) -> dict[str, Any]:
    """Two-cycle cascade: one active branch trains on phase-C then phase-C'.

    Measures the second-cycle gain on C' holdout, the retention of cycle-1 (C
    holdout after cycle-2) and the long-run A retention, all on the same
    isolated active owner with shared fabric/context/protected readout frozen.
    """

    boundary = WorkbenchTaskBoundary.issue(
        project_id="project:seed",
        task_id="task:r1-phase-c-cascade",
        session_id="session:seed11",
        language_id="python",
        capability_snapshot_id="capability:snapshot:1",
        capability_ids=("workspace.read",),
        generation_scope="active",
        issued_tick=10,
        ttl_ticks=60,
    )
    authorization = _authorization(boundary)
    protected_boundary = WorkbenchTaskBoundary.issue(
        project_id="project:seed",
        task_id="task:r1-phase-c-cascade-protected",
        session_id="session:seed11",
        language_id="python",
        capability_snapshot_id="capability:snapshot:1",
        capability_ids=("workspace.read",),
        generation_scope="protected",
        issued_tick=10,
        ttl_ticks=60,
    )
    protected_authorization = _authorization(protected_boundary)

    model = Taiji.from_checkpoint(source_model.checkpoint())
    shared_before = content_digest(model.fabric.to_payload())
    protected_before = _protected_owner_digest(model)
    protected_readout_before = model.readout_registry_status()["protected"]["readout_digest"]
    model.clone_protected_predictive_readout_as_active(
        boundary_digest=boundary.token_digest,
    )
    active_before = model.active_predictive_readout_metadata
    if active_before is None:
        raise RuntimeError("active readout was not registered")

    model.learn_bytes(
        c_train,
        epochs=epochs,
        include_boundary=True,
        use_memory=False,
        learn_fabric=False,
        learn_predictive_context=False,
        learn_predictive_readout=True,
        boundary=boundary,
        authorization=authorization,
    )
    after_cycle1 = model.active_predictive_readout_metadata
    if after_cycle1 is None:
        raise RuntimeError("active readout disappeared after cycle 1")
    c_after_cycle1 = float(
        _score_read_only(
            model,
            c_holdout,
            boundary=boundary,
            authorization=authorization,
        )[0]
    )
    model.learn_bytes(
        c2_train,
        epochs=epochs,
        include_boundary=True,
        use_memory=False,
        learn_fabric=False,
        learn_predictive_context=False,
        learn_predictive_readout=True,
        boundary=boundary,
        authorization=authorization,
    )
    active_after = model.active_predictive_readout_metadata
    if active_after is None:
        raise RuntimeError("active readout disappeared after cycle 2")

    c2_bpb, c2_ro, _, _ = _score_read_only(
        model,
        c2_holdout,
        boundary=boundary,
        authorization=authorization,
    )
    c_final_bpb, c_ro, _, _ = _score_read_only(
        model,
        c_holdout,
        boundary=boundary,
        authorization=authorization,
    )
    a_final_bpb, a_ro, _, _ = _score_read_only(
        model,
        a_retention,
        boundary=boundary,
        authorization=authorization,
    )
    p_c2_bpb, p_c2_ro, _, _ = _score_read_only(
        model,
        c2_holdout,
        boundary=protected_boundary,
        authorization=protected_authorization,
    )

    active_checkpoint = model.checkpoint()
    active_checkpoint_digest = content_digest(active_checkpoint)
    restored = Taiji.from_checkpoint(copy.deepcopy(active_checkpoint))
    restored_digest = content_digest(restored.checkpoint())
    restored_metadata = restored.active_predictive_readout_metadata
    active_canary = model.generate(
        b"Taiji", 16, boundary=boundary, authorization=authorization
    ).hex()
    restored_canary = restored.generate(
        b"Taiji", 16, boundary=boundary, authorization=authorization
    ).hex()
    route = model.last_generation_route
    disk_path = PROJECT_ROOT / "output" / "taiji_r1_cascade_probe.pt"
    try:
        torch.save(active_checkpoint, disk_path)
        fresh_digest, fresh_canary, restore_seconds = _fresh_process_digest(
            disk_path,
            b"Taiji",
            16,
            boundary_payload=boundary.to_payload(),
            authorization_payload=authorization.to_payload(),
        )
        checkpoint_bytes = disk_path.stat().st_size
    finally:
        disk_path.unlink(missing_ok=True)

    CHECKPOINT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_path = CHECKPOINT_OUTPUT_DIR / f"seed{seed}_cascade_c{len(c_train)}.pt"
    torch.save(active_checkpoint, final_path)
    saved_final_digest = content_digest(
        torch.load(final_path, map_location="cpu", weights_only=False)
    )

    checks = {
        "active_readout_changes": active_before["readout_digest"]
        != active_after["readout_digest"],
        "active_readout_changes_during_cycle2": after_cycle1["readout_digest"]
        != active_after["readout_digest"],
        "shared_fabric_unchanged": content_digest(model.fabric.to_payload()) == shared_before,
        "protected_owners_unchanged": _protected_owner_digest(model) == protected_before,
        "protected_readout_unchanged": protected_readout_before
        == model.readout_registry_status()["protected"]["readout_digest"],
        "scores_are_read_only": all((c2_ro, c_ro, a_ro, p_c2_ro)),
        "registry_round_trips_in_process": (
            restored_metadata == active_after and restored_digest == active_checkpoint_digest
        ),
        "registry_round_trips_in_fresh_process": fresh_digest == active_checkpoint_digest,
        "generation_canary_round_trips": active_canary == restored_canary,
        "fresh_process_generation_canary_matches": fresh_canary == active_canary,
        "generation_route_records_active_owner": route is not None
        and route["generation_scope"] == "active"
        and route["readout_owner"] == "predictive_readout.active",
        "final_checkpoint_persisted_with_matching_digest": saved_final_digest
        == active_checkpoint_digest,
    }
    return {
        "arm": "cascade",
        "checks": checks,
        "capability": {
            "protected_c2_holdout_bpb": p_c2_bpb,
            "active_c2_holdout_bpb": c2_bpb,
            "c2_holdout_gain_bpb": p_c2_bpb - c2_bpb,
            "active_c_holdout_bpb_after_cycle2": c_final_bpb,
            "c_holdout_after_cycle1_bpb": c_after_cycle1,
            "c_cycle2_delta_bpb": c_final_bpb - c_after_cycle1,
            "active_a_retention_bpb": a_final_bpb,
        },
        "round_trip": {
            "active_checkpoint_digest": active_checkpoint_digest,
            "fresh_process_digest": fresh_digest,
            "final_checkpoint_path": str(final_path),
            "saved_final_digest": saved_final_digest,
            "checkpoint_bytes": int(checkpoint_bytes),
            "fresh_process_restore_seconds": restore_seconds,
        },
    }


def _run_protected_only_arm(
    *,
    source_model: Taiji,
    c_train: bytes,
    c_holdout: bytes,
    a_retention: bytes,
    epochs: int,
) -> dict[str, Any]:
    """Single-branch plasticity: the protected F1 readout and private temporal
    context learn phase-C while the shared fabric stays frozen."""

    model = Taiji.from_checkpoint(source_model.checkpoint())
    shared_before = content_digest(model.fabric.to_payload())
    protected_readout_before = model.readout_registry_status()["protected"]["readout_digest"]
    model.learn_bytes(
        c_train,
        epochs=epochs,
        include_boundary=True,
        use_memory=False,
        learn_fabric=False,
        learn_predictive_context=True,
        learn_predictive_readout=True,
    )
    protected_readout_after = model.readout_registry_status()["protected"]["readout_digest"]

    protected_boundary = WorkbenchTaskBoundary.issue(
        project_id="project:seed",
        task_id="task:r1-phase-c-protected-only",
        session_id="session:seed11",
        language_id="python",
        capability_snapshot_id="capability:snapshot:1",
        capability_ids=("workspace.read",),
        generation_scope="protected",
        issued_tick=10,
        ttl_ticks=40,
    )
    protected_authorization = _authorization(protected_boundary)
    c_bpb, c_ro, owner, scope = _score_read_only(
        model,
        c_holdout,
        boundary=protected_boundary,
        authorization=protected_authorization,
    )
    a_bpb, a_ro, _, _ = _score_read_only(
        model,
        a_retention,
        boundary=protected_boundary,
        authorization=protected_authorization,
    )
    checks = {
        "predictive_readout_changes": protected_readout_before != protected_readout_after,
        "shared_fabric_unchanged": content_digest(model.fabric.to_payload()) == shared_before,
        "scores_are_read_only": c_ro and a_ro,
        "score_uses_protected_owner": owner == "predictive_readout" and scope == "protected",
    }
    return {
        "arm": "protected_only",
        "checks": checks,
        "capability": {
            "protected_c_holdout_bpb": c_bpb,
            "protected_a_retention_bpb": a_bpb,
            "active_c_holdout_bpb": c_bpb,
            "active_a_retention_bpb": a_bpb,
            "c_holdout_gain_bpb": 0.0,
        },
    }


def _run_no_update_arm(
    *,
    source_model: Taiji,
    c_holdout: bytes,
    a_retention: bytes,
) -> dict[str, Any]:
    """Frozen parent baseline: score without any training."""

    protected_boundary = WorkbenchTaskBoundary.issue(
        project_id="project:seed",
        task_id="task:r1-phase-c-no-update",
        session_id="session:seed11",
        language_id="python",
        capability_snapshot_id="capability:snapshot:1",
        capability_ids=("workspace.read",),
        generation_scope="protected",
        issued_tick=10,
        ttl_ticks=40,
    )
    protected_authorization = _authorization(protected_boundary)
    c_bpb, c_ro, owner, scope = _score_read_only(
        source_model,
        c_holdout,
        boundary=protected_boundary,
        authorization=protected_authorization,
    )
    a_bpb, a_ro, _, _ = _score_read_only(
        source_model,
        a_retention,
        boundary=protected_boundary,
        authorization=protected_authorization,
    )
    checks = {
        "scores_are_read_only": c_ro and a_ro,
        "score_uses_protected_owner": owner == "predictive_readout" and scope == "protected",
    }
    return {
        "arm": "no_update",
        "checks": checks,
        "capability": {
            "protected_c_holdout_bpb": c_bpb,
            "protected_a_retention_bpb": a_bpb,
            "active_c_holdout_bpb": c_bpb,
            "active_a_retention_bpb": a_bpb,
            "c_holdout_gain_bpb": 0.0,
        },
    }


def run_arms(
    checkpoint: Path,
    *,
    corpus_paths: Sequence[Path],
    epochs: int,
    replay_epochs: int,
    train_bytes: int,
    eval_bytes: int,
    arms: Sequence[str],
    seed: int = 11,
    replay_bytes: int | None = None,
) -> dict[str, Any]:
    joint_payload, source_model = load_joint_child(checkpoint, expected_seed=seed)
    lineage_seeds = tuple(dict.fromkeys((*COHORT_SEEDS, int(seed))))
    phase_chain = build_disjoint_phase_chain(
        corpus_paths,
        cohort_seeds=lineage_seeds,
    )
    a_dataset = phase_chain.phase_a_by_seed[int(seed)]
    b_dataset = phase_chain.phase_b_by_seed[int(seed)]
    if str(joint_payload.get("protected_dataset_digest")) != a_dataset.digest:
        raise ValueError(
            "source checkpoint protected_dataset_digest does not match the rebuilt "
            "phase-A lineage; use the exact corpus path used to create the child"
        )
    if str(joint_payload.get("dataset_digest")) != b_dataset.digest:
        raise ValueError(
            "source checkpoint dataset_digest does not match the rebuilt phase-B "
            "lineage; use the exact corpus path used to create the child"
        )
    c_dataset = phase_chain.phase_c
    if train_bytes <= 0 or train_bytes > len(c_dataset.train):
        raise ValueError("train_bytes must be within the phase-C training partition")
    if eval_bytes <= 0 or eval_bytes > min(len(a_dataset.retention), len(c_dataset.holdout)):
        raise ValueError("eval_bytes must fit both A retention and C holdout prefixes")
    c_train = c_dataset.train[:train_bytes]
    c_holdout = c_dataset.holdout[:eval_bytes]
    a_retention = a_dataset.retention[:eval_bytes]
    for arm in arms:
        if arm not in ALL_ARMS:
            raise ValueError(f"unsupported arm: {arm}")

    # --- preflight: fresh-process digest + generation canary of the source ---
    probe_disk = PROJECT_ROOT / "output" / "taiji_r1_source_probe.pt"
    try:
        torch.save(source_model.checkpoint(), probe_disk)
        source_digest, source_canary, source_seconds = _fresh_process_digest(
            probe_disk, b"Taiji", 16
        )
        source_bytes = probe_disk.stat().st_size
    finally:
        probe_disk.unlink(missing_ok=True)
    live_source_digest = content_digest(source_model.checkpoint())
    source_canary_live = source_model.generate(b"Taiji", 16).hex()

    results: list[dict[str, Any]] = []
    if "no_update" in arms:
        results.append(
            _run_no_update_arm(
                source_model=source_model,
                c_holdout=c_holdout,
                a_retention=a_retention,
            )
        )
    if "protected_only" in arms:
        results.append(
            _run_protected_only_arm(
                source_model=source_model,
                c_train=c_train,
                c_holdout=c_holdout,
                a_retention=a_retention,
                epochs=epochs,
            )
        )
    for arm in ("active_only", "replay"):
        if arm not in arms:
            continue
        results.append(
            _run_branch_arm(
                arm=arm,
                source_model=source_model,
                c_train=c_train,
                a_train=a_dataset.train[:train_bytes],
                c_holdout=c_holdout,
                a_retention=a_retention,
                epochs=epochs,
                replay_epochs=replay_epochs,
                replay_bytes=replay_bytes,
                seed=seed,
            )
        )
    if "cascade" in arms:
        c2_dataset = phase_chain.phase_c2
        if train_bytes <= 0 or train_bytes > len(c2_dataset.train):
            raise ValueError("cascade train_bytes must be within the phase-C' partition")
        results.append(
            _run_cascade_arm(
                source_model=source_model,
                c_train=c_train,
                c2_train=c2_dataset.train[:train_bytes],
                c_holdout=c_holdout,
                c2_holdout=c2_dataset.holdout[:eval_bytes],
                a_retention=a_retention,
                epochs=epochs,
                seed=seed,
            )
        )
    else:
        c2_dataset = None

    all_passed = all(
        bool(result["checks"]) and all(bool(value) for value in result["checks"].values())
        for result in results
    )
    return {
        "format": FORMAT,
        "version": 1,
        "status": "passed" if all_passed else "failed",
        "can_promote": False,
        "source_checkpoint": str(checkpoint),
        "source_digest": live_source_digest,
        "epochs": int(epochs),
        "replay_epochs": int(replay_epochs),
        "c_train_bytes": int(train_bytes),
        "eval_bytes": int(eval_bytes),
        "replay_bytes": None if replay_bytes is None else int(replay_bytes),
        "datasets": {
            "phase_a": _dataset_metadata(a_dataset),
            "phase_b": _dataset_metadata(b_dataset),
            "phase_c": _dataset_metadata(c_dataset),
            "phase_c2": (
                None if c2_dataset is None else _dataset_metadata(c2_dataset)
            ),
        },
        "data_contract": {
            "format": "taiji-m2r1-record-disjoint-course-v1",
            "version": 1,
            "current_seed": int(seed),
            "source_lineage": {
                "phase_a_expected_digest": str(joint_payload["protected_dataset_digest"]),
                "phase_a_actual_digest": a_dataset.digest,
                "phase_b_expected_digest": str(joint_payload["dataset_digest"]),
                "phase_b_actual_digest": b_dataset.digest,
                "matches_source_checkpoint": True,
            },
            "phase_chain": _phase_chain_metadata(phase_chain),
            "gate": {
                "record_disjoint": not any(phase_chain.overlap_counts.values()),
                "source_lineage_matches": True,
            },
        },
        "preflight": {
            "source_digest": live_source_digest,
            "source_fresh_digest": source_digest,
            "source_generation_canary": source_canary_live,
            "source_fresh_canary": source_canary,
            "source_checkpoint_bytes": int(source_bytes),
            "source_fresh_restore_seconds": source_seconds,
        },
        "arms": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--replay-epochs", type=int, default=1)
    parser.add_argument(
        "--replay-bytes",
        type=int,
        help="Replay byte slice from the phase-A train partition; default equals train-bytes.",
    )
    parser.add_argument("--train-bytes", type=int, default=64 * 1024)
    parser.add_argument("--eval-bytes", type=int, default=32 * 1024)
    parser.add_argument("--arms", nargs="+", choices=ALL_ARMS, default=list(ALL_ARMS))
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.epochs <= 0 or args.replay_epochs <= 0:
        parser.error("epochs and replay_epochs must be positive")
    if args.train_bytes <= 0 or args.eval_bytes <= 0:
        parser.error("train_bytes and eval_bytes must be positive")
    if args.replay_bytes is not None and (
        args.replay_bytes <= 0 or args.replay_bytes > args.train_bytes
    ):
        parser.error("--replay-bytes must be positive and within train-bytes")
    result = run_arms(
        args.checkpoint,
        corpus_paths=args.corpus,
        epochs=args.epochs,
        replay_epochs=args.replay_epochs,
        train_bytes=args.train_bytes,
        eval_bytes=args.eval_bytes,
        arms=args.arms,
        seed=args.seed,
        replay_bytes=args.replay_bytes,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "report": str(args.report),
        "status": result["status"],
        "preflight_ok": result["preflight"]["source_digest"]
        == result["preflight"]["source_fresh_digest"],
        "arms": [
            {
                "arm": arm["arm"],
                "checks_passed": sum(int(v) for v in arm["checks"].values()),
                "checks_total": len(arm["checks"]),
                "c_holdout_gain_bpb": arm["capability"].get(
                    "c_holdout_gain_bpb",
                    arm["capability"].get("c2_holdout_gain_bpb"),
                ),
            }
            for arm in result["arms"]
        ],
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
