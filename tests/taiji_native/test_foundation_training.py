from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from scripts.training.train_taiji_memory import build_corpus as build_memory_corpus
from scripts.training.train_taiji_world_action import (
    build_goal_corpus,
    build_world_corpus,
    build_world_learner,
)
from taiji import (
    JOINT_SEQUENCE_PREDICTIVE_CONTEXT_MODE,
    JOINT_SEQUENCE_READOUT_MODE,
    JOINT_TRAINING_VERSION,
    DelayedMemoryCorpus,
    DelayedMemoryQuery,
    FoundationTrainingDataset,
    FoundationTrainingRun,
    JointTrainingRun,
    MemoryEpisode,
    MemoryTrainingRun,
    Taiji,
    TaijiConfig,
    WorldActionTrainingRun,
    content_digest,
)
from taiji.foundation_training import _sequence_fabric_digest, _train_goal_episode


def _dataset() -> FoundationTrainingDataset:
    return FoundationTrainingDataset(
        train=b"ABCD1234-" * 128,
        holdout=b"ABCD1234+" * 16,
        retention=b"ABCD1234?" * 16,
        source_files=(("inline-test", "dataset-source"),),
        partition_seed=11,
        profile="smoke",
    )


def _config() -> TaijiConfig:
    return TaijiConfig(
        region_sizes=(8,),
        synapse_fan_in=2,
        motor_fan_in=4,
        memory_units=16,
        memory_fan_in=2,
        memory_readout_fan_in=2,
        memory_meta_dim=4,
        memory_time_dim=2,
        memory_episode_dim=2,
        lateral_fan_in=2,
        concept_capacity=8,
        seed=11,
    )


def _phase_datasets() -> tuple[FoundationTrainingDataset, FoundationTrainingDataset]:
    """Small content-addressed stand-ins for an M2 F5 phase-A/phase-B pair."""

    source_files = (("inline-m2-phase-course", "phase-course-source"),)
    phase_a = FoundationTrainingDataset(
        train=b"phase-a-old-train|" * 16,
        holdout=b"phase-a-old-holdout|" * 8,
        retention=b"phase-a-old-retention|" * 8,
        source_files=source_files,
        partition_seed=11,
        profile="smoke",
    )
    phase_b = FoundationTrainingDataset(
        train=b"phase-b-new-train|" * 16,
        holdout=b"phase-b-new-holdout|" * 8,
        retention=b"phase-b-new-retention|" * 8,
        source_files=source_files,
        partition_seed=10011,
        profile="smoke",
        excluded_dataset_digest=phase_a.digest,
    )
    return phase_a, phase_b


def test_phase_b_dataset_excludes_every_selected_phase_a_record() -> None:
    """F5 cannot call a reused source stream a new language course."""

    corpus = Path(".seed_test_tmp") / "m2-phase-exclusion.jsonl"
    corpus.parent.mkdir(parents=True, exist_ok=True)

    def record_for_phase_a_partition(
        partition: str,
        label: str,
        *,
        late_marker: str = "",
    ) -> dict[str, str]:
        for attempt in range(10_000):
            prefix = f"<{label}:{attempt:04d}>"
            text = prefix + ("x" * 256) + late_marker
            text += "x" * (1_000 - len(text))
            if FoundationTrainingDataset._partition_for_text(text, 11) == partition:
                return {"text": text}
        raise AssertionError(f"could not find phase-A {partition} record for {label}")

    # Four 1,000-byte train records leave only 96 bytes in phase A. The fifth
    # selected record therefore contributes a prefix only; its late marker
    # must not leak into B through the unused suffix of that record.
    phase_a_only_suffix = "<phase-a-only-suffix>"
    records = [record_for_phase_a_partition("train", f"train-{index}") for index in range(4)]
    records.append(
        record_for_phase_a_partition(
            "train",
            "partial-train",
            late_marker=phase_a_only_suffix,
        )
    )
    records.extend(
        record_for_phase_a_partition("holdout", f"holdout-{index}") for index in range(2)
    )
    records.extend(
        record_for_phase_a_partition("retention", f"retention-{index}") for index in range(2)
    )
    records.extend(
        {"text": f"<phase-b-candidate:{index:04d}>" + ("z" * 968)} for index in range(256)
    )
    corpus.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    phase_a = FoundationTrainingDataset.from_jsonl(
        (corpus,),
        profile="smoke",
        partition_seed=11,
    )
    phase_b = FoundationTrainingDataset.from_jsonl(
        (corpus,),
        profile="smoke",
        partition_seed=29,
        exclude_dataset=phase_a,
    )

    phase_a_bytes = phase_a.train + phase_a.holdout + phase_a.retention
    phase_b_bytes = phase_b.train + phase_b.holdout + phase_b.retention

    assert phase_b.excluded_dataset_digest == phase_a.digest
    assert b"<partial-train:" in phase_a_bytes
    assert b"<partial-train:" not in phase_b_bytes
    assert phase_a_only_suffix.encode() not in phase_a_bytes
    assert phase_a_only_suffix.encode() not in phase_b_bytes


def test_foundation_training_saves_and_resumes_from_disk_checkpoint() -> None:
    dataset = _dataset()
    output_dir = Path(".seed_test_tmp") / "m1-foundation-training"
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("parent.pt", "last.pt", "best-holdout.pt"):
        (output_dir / filename).unlink(missing_ok=True)

    run = FoundationTrainingRun(
        Taiji(TaijiConfig.from_dict(_config().to_dict()), episode_id="training-test"),
        dataset,
        output_dir=output_dir,
        profile="smoke",
        model_tier="micro",
        epochs=1,
        chunk_bytes=64,
        checkpoint_interval=2,
    )
    report = run.run()

    assert report["status"] == "completed"
    assert report["cursor"] == 0
    assert report["global_step"] > 0
    assert Path(report["checkpoint_paths"]["parent"]).is_file()
    assert Path(report["checkpoint_paths"]["last"]).is_file()
    assert Path(report["checkpoint_paths"]["best_holdout"]).is_file()

    restored = FoundationTrainingRun.from_checkpoint(
        output_dir / "last.pt",
        dataset,
        output_dir=output_dir,
        epochs=2,
    )
    evaluation = restored.evaluate_only()
    assert evaluation["checkpoint_read_only"] is True
    continued = restored.run()
    assert continued["global_step"] > report["global_step"]

    code = (
        "from pathlib import Path; "
        "from taiji import FoundationTrainingDataset, FoundationTrainingRun; "
        f"d=FoundationTrainingDataset(train={dataset.train!r}, holdout={dataset.holdout!r}, "
        f"retention={dataset.retention!r}, source_files={dataset.source_files!r}, "
        f"partition_seed={dataset.partition_seed}, profile={dataset.profile!r}); "
        f"r=FoundationTrainingRun.from_checkpoint(Path({str(output_dir / 'last.pt')!r}), d); "
        "assert r.evaluate_only()['checkpoint_read_only']"
    )
    subprocess.run((sys.executable, "-c", code), cwd=Path.cwd(), check=True)


def test_memory_training_saves_a_read_only_recall_checkpoint() -> None:
    train = tuple(
        MemoryEpisode(
            memory_id=f"m1-f2-test-{index}",
            cue=65 + index,
            action=48 + index % 2,
            outcome=43 + index % 2,
        )
        for index in range(4)
    )
    corpus = DelayedMemoryCorpus(
        train=train,
        holdout=tuple(
            DelayedMemoryQuery(f"holdout-{index}", item.cue, item.action)
            for index, item in enumerate(train)
        ),
        retention=tuple(
            DelayedMemoryQuery(f"retention-{index}", item.cue, item.action)
            for index, item in enumerate(train)
        ),
    )
    output_dir = Path(".seed_test_tmp") / "m1-memory-training"
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("parent.pt", "last.pt", "best-holdout.pt"):
        (output_dir / filename).unlink(missing_ok=True)

    run = MemoryTrainingRun(
        Taiji(_config(), episode_id="memory-training-test"),
        corpus,
        output_dir=output_dir,
        epochs=1,
        checkpoint_interval=1,
    )
    report = run.run()

    assert report["status"] == "completed"
    assert report["holdout_updates"] == 0
    assert report["corpus_sample_counts"] == {"train": 4, "holdout": 4, "retention": 4}
    assert Path(report["checkpoint_paths"]["last"]).is_file()
    restored = MemoryTrainingRun.from_checkpoint(
        output_dir / "last.pt",
        corpus,
        output_dir=output_dir,
    )
    evaluation = restored.evaluate_only()
    assert evaluation["checkpoint_read_only"] is True


def test_memory_corpus_foundation_scale_stays_inside_byte_alphabet() -> None:
    corpus = build_memory_corpus(count=1_000)

    assert min(item.cue for item in corpus.train) >= 0
    assert max(item.cue for item in corpus.train) <= 255
    assert max(item.action for item in corpus.train) <= 255
    assert max(item.outcome for item in corpus.train) <= 255


def test_world_action_training_saves_and_resumes_atomic_checkpoint() -> None:
    world_corpus = build_world_corpus(count=4)
    goal_corpus = build_goal_corpus(count=4)
    output_dir = Path(".seed_test_tmp") / "m1-world-action-training"
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("parent.pt", "last.pt", "best-holdout.pt"):
        (output_dir / filename).unlink(missing_ok=True)

    run = WorldActionTrainingRun(
        Taiji(_config(), episode_id="world-action-training-test"),
        build_world_learner(world_corpus, seed=11),
        world_corpus,
        goal_corpus,
        output_dir=output_dir,
        epochs=1,
        checkpoint_interval=2,
        world_repeats=1,
    )
    report = run.run()

    assert report["status"] == "completed"
    assert report["world_cursor"] == 0
    assert report["goal_cursor"] == 0
    assert report["global_step"] == 8
    assert Path(report["checkpoint_paths"]["parent"]).is_file()
    assert Path(report["checkpoint_paths"]["last"]).is_file()
    assert Path(report["checkpoint_paths"]["best_holdout"]).is_file()

    restored = WorldActionTrainingRun.from_checkpoint(
        output_dir / "last.pt",
        world_corpus,
        goal_corpus,
        output_dir=output_dir,
        epochs=2,
    )
    evaluation = restored.evaluate_only()
    assert evaluation["checkpoint_read_only"] is True
    continued = restored.run()
    assert continued["global_step"] > report["global_step"]


def test_joint_training_preserves_three_organs_in_one_checkpoint() -> None:
    dataset = FoundationTrainingDataset(
        train=b"ABCD1234-" * 32,
        holdout=b"ABCD1234+" * 8,
        retention=b"ABCD1234?" * 8,
        source_files=(("inline-joint", "dataset-source"),),
        partition_seed=11,
        profile="smoke",
    )
    memory_corpus = build_memory_corpus(count=4)
    world_corpus = build_world_corpus(count=4)
    goal_corpus = build_goal_corpus(count=4)
    output_dir = Path(".seed_test_tmp") / "m1-joint-training"
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("parent.pt", "last.pt", "best-holdout.pt"):
        (output_dir / filename).unlink(missing_ok=True)

    run = JointTrainingRun(
        Taiji(_config(), episode_id="joint-training-test"),
        build_world_learner(world_corpus, seed=11),
        dataset,
        memory_corpus,
        world_corpus,
        goal_corpus,
        output_dir=output_dir,
        epochs=1,
        chunk_bytes=32,
        checkpoint_interval=2,
        world_repeats=1,
    )
    report = run.run()

    assert report["status"] == "completed"
    assert report["holdout_updates"] == 0
    assert (
        report["final_metrics"]["sequence_holdout_bpb"]
        < report["parent_metrics"]["sequence_holdout_bpb"]
    )
    assert Path(report["checkpoint_paths"]["last"]).is_file()
    restored = JointTrainingRun.from_checkpoint(
        output_dir / "last.pt",
        dataset,
        memory_corpus,
        world_corpus,
        goal_corpus,
        output_dir=output_dir,
        epochs=2,
    )
    evaluation = restored.evaluate_only()
    assert evaluation["checkpoint_read_only"] is True
    continued = restored.run()
    assert continued["global_step"] > report["global_step"]


def test_joint_training_retries_transient_checkpoint_replace_lock(monkeypatch) -> None:
    dataset = _dataset()
    memory_corpus = build_memory_corpus(count=4)
    world_corpus = build_world_corpus(count=4)
    goal_corpus = build_goal_corpus(count=4)
    output_dir = Path(".seed_test_tmp") / "m1-joint-save-retry"
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "last.pt"
    target.unlink(missing_ok=True)
    (output_dir / "last.pt.tmp").unlink(missing_ok=True)

    run = JointTrainingRun(
        Taiji(_config(), episode_id="joint-save-retry-test"),
        build_world_learner(world_corpus, seed=11),
        dataset,
        memory_corpus,
        world_corpus,
        goal_corpus,
        output_dir=output_dir,
        epochs=1,
        chunk_bytes=32,
        checkpoint_interval=2,
        world_repeats=1,
    )
    original_replace = Path.replace
    attempts = {"count": 0}

    def flaky_replace(self: Path, destination: str | Path) -> Path:
        if self.name == "last.pt.tmp" and attempts["count"] < 2:
            attempts["count"] += 1
            raise PermissionError("transient reader lock")
        return original_replace(self, destination)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    assert run.save(target) == target
    assert attempts["count"] == 2
    assert target.is_file()


def test_joint_sequence_only_continuation_protects_phase_a_metrics(monkeypatch) -> None:
    phase_a, phase_b = _phase_datasets()
    memory_corpus = build_memory_corpus(count=4)
    world_corpus = build_world_corpus(count=4)
    goal_corpus = build_goal_corpus(count=4)
    output_dir = Path(".seed_test_tmp") / "m2-sequence-only-continuation"
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("parent.pt", "last.pt", "best-holdout.pt"):
        (output_dir / filename).unlink(missing_ok=True)

    world_learner = build_world_learner(world_corpus, seed=11)
    initial_world_updates = world_learner.online_updates
    run = JointTrainingRun(
        Taiji(_config(), episode_id="m2-sequence-only-test"),
        world_learner,
        phase_b,
        memory_corpus,
        world_corpus,
        goal_corpus,
        output_dir=output_dir,
        epochs=1,
        chunk_bytes=32,
        checkpoint_interval=2,
        metric_interval=1_000,
        world_repeats=1,
        protected_dataset=phase_a,
        training_phases=("sequence",),
    )
    original_measure = run._measure_metrics
    measurements = {"count": 0}

    def count_terminal_measurement() -> dict[str, float]:
        measurements["count"] += 1
        return original_measure()

    monkeypatch.setattr(run, "_measure_metrics", count_terminal_measurement)

    report = run.run()

    assert report["training_phases"] == ["sequence"]
    assert report["protected_dataset_digest"] == phase_a.digest
    assert report["sequence_readout_mode"] == JOINT_SEQUENCE_READOUT_MODE
    assert report["sequence_predictive_context_mode"] == JOINT_SEQUENCE_PREDICTIVE_CONTEXT_MODE
    assert report["world_online_updates"] == initial_world_updates
    assert report["global_step"] == len(phase_b.train) // 32
    assert measurements["count"] == 1
    assert all(item.get("train_kind") in {None, "sequence"} for item in report["history"])
    for key in (
        "protected_sequence_holdout_bpb",
        "protected_sequence_retention_bpb",
    ):
        assert key in report["parent_metrics"]
        assert key in report["final_metrics"]

    readout_contract = report["sequence_readout_contract"]
    assert set(readout_contract) == {"parent", "final", "phase_checks"}
    assert (
        readout_contract["parent"]["predictive_readout_digest"]
        != readout_contract["final"]["predictive_readout_digest"]
    )
    assert (
        readout_contract["parent"]["predictive_context_digest"]
        != readout_contract["final"]["predictive_context_digest"]
    )
    assert len(readout_contract["phase_checks"]) == 1
    assert all(readout_contract["phase_checks"][0]["preserved"].values())
    context_contract = report["sequence_predictive_context_contract"]
    assert context_contract["parent"] != context_contract["final"]
    assert len(context_contract["phase_checks"]) == 1
    assert context_contract["phase_checks"][0]["changed"] is True
    keep_gate = report["sequence_only_keep_gate"]
    assert keep_gate["applies"] is True
    assert all(keep_gate["readout_preserved"].values())
    assert keep_gate["predictive_context_grew"] is True

    payload = torch.load(output_dir / "last.pt", map_location="cpu", weights_only=False)
    assert payload["training_phases"] == ["sequence"]
    assert payload["protected_dataset_digest"] == phase_a.digest
    assert payload["sequence_readout_mode"] == JOINT_SEQUENCE_READOUT_MODE
    assert payload["sequence_readout_parent"] == readout_contract["parent"]
    assert payload["sequence_readout_phase_checks"] == readout_contract["phase_checks"]
    assert payload["sequence_predictive_context_mode"] == JOINT_SEQUENCE_PREDICTIVE_CONTEXT_MODE
    assert payload["sequence_predictive_context_parent"] == context_contract["parent"]
    assert payload["sequence_predictive_context_phase_checks"] == context_contract["phase_checks"]
    restored = JointTrainingRun.from_checkpoint(
        output_dir / "last.pt",
        phase_b,
        memory_corpus,
        world_corpus,
        goal_corpus,
        output_dir=output_dir,
        protected_dataset=phase_a,
    )
    assert restored.training_phases == ("sequence",)
    restored_evaluation = restored.evaluate_only()
    assert restored_evaluation["checkpoint_read_only"] is True
    assert restored_evaluation["sequence_readout_mode"] == JOINT_SEQUENCE_READOUT_MODE
    assert (
        restored_evaluation["sequence_predictive_context_mode"]
        == JOINT_SEQUENCE_PREDICTIVE_CONTEXT_MODE
    )
    assert restored_evaluation["sequence_readout_contract"]["parent"] == readout_contract["parent"]
    assert all(restored_evaluation["sequence_only_keep_gate"]["readout_preserved"].values())
    assert restored_evaluation["sequence_only_keep_gate"]["predictive_context_grew"] is True
    with pytest.raises(ValueError, match="phase plan"):
        JointTrainingRun.from_checkpoint(
            output_dir / "last.pt",
            phase_b,
            memory_corpus,
            world_corpus,
            goal_corpus,
            output_dir=output_dir,
            protected_dataset=phase_a,
            training_phases=("sequence", "memory"),
        )


def test_joint_sequence_fabric_mode_is_content_addressed_and_resumable() -> None:
    """A predictor-only F1 course must not silently resume as fabric-plastic."""

    phase_a, phase_b = _phase_datasets()
    memory_corpus = build_memory_corpus(count=4)
    world_corpus = build_world_corpus(count=4)
    goal_corpus = build_goal_corpus(count=4)
    output_dir = Path(".seed_test_tmp") / "m2-predictor-only-sequence"
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("parent.pt", "last.pt", "best-holdout.pt"):
        (output_dir / filename).unlink(missing_ok=True)

    run = JointTrainingRun(
        Taiji(_config(), episode_id="m2-predictor-only-test"),
        build_world_learner(world_corpus, seed=11),
        phase_b,
        memory_corpus,
        world_corpus,
        goal_corpus,
        output_dir=output_dir,
        epochs=1,
        chunk_bytes=32,
        checkpoint_interval=2,
        metric_interval=1_000,
        world_repeats=1,
        protected_dataset=phase_a,
        training_phases=("sequence",),
        sequence_fabric_learning=False,
    )
    report = run.run()

    assert report["sequence_fabric_learning"] is False
    fabric_contract = report["sequence_fabric_contract"]
    assert fabric_contract["parent"] == fabric_contract["final"]
    assert len(fabric_contract["phase_checks"]) == 1
    assert fabric_contract["phase_checks"][0]["preserved"] is True
    context_contract = report["sequence_predictive_context_contract"]
    assert context_contract["parent"] != context_contract["final"]
    assert context_contract["phase_checks"][0]["changed"] is True

    payload = torch.load(output_dir / "last.pt", map_location="cpu", weights_only=False)
    assert payload["version"] == JOINT_TRAINING_VERSION
    assert payload["sequence_fabric_learning"] is False
    assert payload["sequence_fabric_parent"] == fabric_contract["parent"]
    assert payload["sequence_fabric_phase_checks"] == fabric_contract["phase_checks"]
    assert payload["sequence_predictive_context_mode"] == JOINT_SEQUENCE_PREDICTIVE_CONTEXT_MODE
    assert payload["sequence_predictive_context_parent"] == context_contract["parent"]

    restored = JointTrainingRun.from_checkpoint(
        output_dir / "last.pt",
        phase_b,
        memory_corpus,
        world_corpus,
        goal_corpus,
        output_dir=output_dir,
        protected_dataset=phase_a,
    )
    assert restored.sequence_fabric_learning is False
    evaluation = restored.evaluate_only()
    assert evaluation["sequence_fabric_learning"] is False
    assert evaluation["sequence_fabric_contract"]["parent"] == fabric_contract["parent"]
    assert (
        evaluation["sequence_predictive_context_contract"]["parent"] == context_contract["parent"]
    )
    with pytest.raises(ValueError, match="sequence fabric learning"):
        JointTrainingRun.from_checkpoint(
            output_dir / "last.pt",
            phase_b,
            memory_corpus,
            world_corpus,
            goal_corpus,
            output_dir=output_dir,
            protected_dataset=phase_a,
            sequence_fabric_learning=True,
        )


def test_joint_protected_replay_requires_exact_phase_a_dataset() -> None:
    phase_a, phase_b = _phase_datasets()
    other_old_course = FoundationTrainingDataset(
        train=b"different-old-train|" * 16,
        holdout=b"different-old-holdout|" * 8,
        retention=b"different-old-retention|" * 8,
        source_files=(("inline-other-old-course", "other-source"),),
        partition_seed=47,
        profile="smoke",
    )
    memory_corpus = build_memory_corpus(count=4)
    world_corpus = build_world_corpus(count=4)
    goal_corpus = build_goal_corpus(count=4)

    with pytest.raises(ValueError, match="exact protected phase-A dataset"):
        JointTrainingRun(
            Taiji(_config(), episode_id="m2-replay-contract-test"),
            build_world_learner(world_corpus, seed=11),
            phase_b,
            memory_corpus,
            world_corpus,
            goal_corpus,
            output_dir=Path(".seed_test_tmp") / "m2-replay-contract",
            epochs=1,
            chunk_bytes=32,
            checkpoint_interval=2,
            world_repeats=1,
            protected_dataset=phase_a,
            replay_dataset=other_old_course,
            training_phases=("sequence", "replay"),
        )


def test_joint_legacy_shared_readout_checkpoint_requires_explicit_continuation() -> None:
    """A pre-M2-2f sequence course may be inspected, never silently resumed."""

    phase_a, phase_b = _phase_datasets()
    memory_corpus = build_memory_corpus(count=4)
    world_corpus = build_world_corpus(count=4)
    goal_corpus = build_goal_corpus(count=4)
    output_dir = Path(".seed_test_tmp") / "m2-legacy-shared-readout"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "legacy-last.pt"
    checkpoint_path.unlink(missing_ok=True)

    original = JointTrainingRun(
        Taiji(_config(), episode_id="m2-legacy-shared-readout"),
        build_world_learner(world_corpus, seed=11),
        phase_b,
        memory_corpus,
        world_corpus,
        goal_corpus,
        output_dir=output_dir,
        epochs=1,
        chunk_bytes=32,
        checkpoint_interval=2,
        world_repeats=1,
        protected_dataset=phase_a,
        training_phases=("sequence",),
    )
    original.save(checkpoint_path)
    legacy = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    legacy.pop("sequence_readout_mode")
    legacy.pop("sequence_readout_parent")
    legacy.pop("sequence_readout_phase_checks")
    legacy["checkpoint_digest"] = content_digest(
        {key: value for key, value in legacy.items() if key != "checkpoint_digest"}
    )
    torch.save(legacy, checkpoint_path)

    restored = JointTrainingRun.from_checkpoint(
        checkpoint_path,
        phase_b,
        memory_corpus,
        world_corpus,
        goal_corpus,
        output_dir=output_dir,
        protected_dataset=phase_a,
    )
    assert restored.sequence_readout_mode == "legacy-shared-readout-v0"
    with pytest.raises(RuntimeError, match="explicit continuation course"):
        restored.run()

    continuation = JointTrainingRun.from_continuation_checkpoint(
        checkpoint_path,
        phase_b,
        memory_corpus,
        world_corpus,
        goal_corpus,
        output_dir=output_dir / "m2f-child",
        epochs=1,
        chunk_bytes=32,
        checkpoint_interval=2,
        world_repeats=1,
        protected_dataset=phase_a,
        training_phases=("sequence",),
    )
    assert continuation.sequence_readout_mode == JOINT_SEQUENCE_READOUT_MODE


def test_organ_only_goal_training_does_not_write_shared_fabric() -> None:
    model = Taiji(_config(), episode_id="organ-only-goal-test")
    episode = build_goal_corpus(count=1).train[0]
    before = _sequence_fabric_digest(model)

    _train_goal_episode(model, episode, learn=True, learn_fabric=False)

    assert _sequence_fabric_digest(model) == before


def test_joint_v3_fixed_basis_checkpoint_requires_explicit_m2_2h_continuation() -> None:
    """A v3 predictor-only course cannot silently gain a plastic F1 context."""

    phase_a, phase_b = _phase_datasets()
    memory_corpus = build_memory_corpus(count=4)
    world_corpus = build_world_corpus(count=4)
    goal_corpus = build_goal_corpus(count=4)
    output_dir = Path(".seed_test_tmp") / "m2-legacy-fixed-basis"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "legacy-v3-last.pt"
    checkpoint_path.unlink(missing_ok=True)

    original = JointTrainingRun(
        Taiji(_config(), episode_id="m2-legacy-fixed-basis"),
        build_world_learner(world_corpus, seed=11),
        phase_b,
        memory_corpus,
        world_corpus,
        goal_corpus,
        output_dir=output_dir,
        epochs=1,
        chunk_bytes=32,
        checkpoint_interval=2,
        world_repeats=1,
        protected_dataset=phase_a,
        training_phases=("sequence",),
        sequence_fabric_learning=False,
    )
    original.save(checkpoint_path)
    legacy = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    def v9_model_payload(payload: dict) -> dict:
        migrated = deepcopy(payload)
        migrated["format"] = "taiji-native-v9"
        migrated.pop("predictive_context")
        for field in (
            "predictive_context_seed_offset",
            "predictive_context_fan_in",
            "predictive_context_learning_rate",
            "predictive_context_recurrent_gain",
        ):
            migrated["config"].pop(field)
        migrated["state"]["version"] = 6
        migrated["state"].pop("predictive_context_trace")
        identity = migrated.get("identity_organ")
        if isinstance(identity, dict):
            identity["lineage"]["parent_checkpoint_digest"] = content_digest(
                {
                    key: migrated[key]
                    for key in Taiji._checkpoint_core_keys(
                        include_predictive=True,
                        include_predictive_context=False,
                    )
                }
            )
        return migrated

    legacy["model"] = v9_model_payload(legacy["model"])
    legacy["parent_model"] = v9_model_payload(legacy["parent_model"])
    legacy["version"] = 3
    legacy.pop("sequence_predictive_context_mode")
    legacy.pop("sequence_predictive_context_parent")
    legacy.pop("sequence_predictive_context_phase_checks")
    legacy["corpus_digest"] = content_digest(
        {
            "format": legacy["format"],
            "version": 3,
            "dataset_digest": phase_b.digest,
            "memory_digest": legacy["memory_digest"],
            "world_action_digest": legacy["world_action_digest"],
            "protected_dataset_digest": phase_a.digest,
            "training_phases": ["sequence"],
            "sequence_fabric_learning": False,
        }
    )
    legacy["checkpoint_digest"] = content_digest(
        {key: value for key, value in legacy.items() if key != "checkpoint_digest"}
    )
    torch.save(legacy, checkpoint_path)

    restored = JointTrainingRun.from_checkpoint(
        checkpoint_path,
        phase_b,
        memory_corpus,
        world_corpus,
        goal_corpus,
        output_dir=output_dir,
        protected_dataset=phase_a,
    )
    assert restored.sequence_predictive_context_mode == "legacy-fixed-basis-v0"
    with pytest.raises(RuntimeError, match="M2-2h migration"):
        restored.run()

    continuation = JointTrainingRun.from_continuation_checkpoint(
        checkpoint_path,
        phase_b,
        memory_corpus,
        world_corpus,
        goal_corpus,
        output_dir=output_dir / "m2h-child",
        epochs=1,
        chunk_bytes=32,
        checkpoint_interval=2,
        world_repeats=1,
        protected_dataset=phase_a,
        training_phases=("sequence",),
    )
    assert continuation.sequence_predictive_context_mode == JOINT_SEQUENCE_PREDICTIVE_CONTEXT_MODE


def test_joint_training_starts_an_explicit_continuation_from_child_checkpoint() -> None:
    dataset = FoundationTrainingDataset(
        train=b"ABCD1234-" * 16,
        holdout=b"ABCD1234+" * 4,
        retention=b"ABCD1234?" * 4,
        source_files=(("inline-continuation", "dataset-source"),),
        partition_seed=11,
        profile="smoke",
    )
    memory_corpus = build_memory_corpus(count=4)
    replay_memory_corpus = build_memory_corpus(count=4)
    world_corpus = build_world_corpus(count=4)
    goal_corpus = build_goal_corpus(count=4)
    output_dir = Path(".seed_test_tmp") / "m1-continuation-parent"
    continuation_dir = Path(".seed_test_tmp") / "m1-continuation-child"
    for directory in (output_dir, continuation_dir):
        directory.mkdir(parents=True, exist_ok=True)
        for filename in ("parent.pt", "last.pt", "best-holdout.pt"):
            (directory / filename).unlink(missing_ok=True)

    original = JointTrainingRun(
        Taiji(_config(), episode_id="continuation-parent"),
        build_world_learner(world_corpus, seed=11),
        dataset,
        memory_corpus,
        world_corpus,
        goal_corpus,
        output_dir=output_dir,
        epochs=1,
        chunk_bytes=32,
        checkpoint_interval=2,
        world_repeats=1,
    )
    original_report = original.run()
    source_payload = torch.load(
        output_dir / "best-holdout.pt", map_location="cpu", weights_only=False
    )
    continuation = JointTrainingRun.from_continuation_checkpoint(
        output_dir / "best-holdout.pt",
        FoundationTrainingDataset(
            train=dataset.train + b"EFGH5678-" * 16,
            holdout=dataset.holdout + b"EFGH5678+" * 4,
            retention=dataset.retention + b"EFGH5678?" * 4,
            source_files=(("inline-continuation-expanded", "dataset-source"),),
            partition_seed=11,
            profile="smoke",
        ),
        build_memory_corpus(count=6),
        build_world_corpus(count=6),
        build_goal_corpus(count=6),
        output_dir=continuation_dir,
        epochs=1,
        chunk_bytes=32,
        checkpoint_interval=2,
        metric_interval=100,
        world_repeats=1,
        replay_dataset=dataset,
        replay_epochs=1,
        replay_memory_corpus=replay_memory_corpus,
        replay_memory_epochs=1,
    )

    assert continuation.continuation_source_checkpoint_digest
    assert continuation.parent_checkpoint_path.is_file()
    report = continuation.run()
    assert report["status"] == "completed"
    assert report["continuation_source_checkpoint_digest"]
    assert report["continuation_source_checkpoint_digest"] == source_payload["checkpoint_digest"]
    assert report["parent_checkpoint_digest"]
    assert report["parent_checkpoint_digest"] != original_report["child_checkpoint_digest"]
    assert report["replay_dataset_digest"] == dataset.digest
    assert report["replay_memory_digest"]
    assert report["replay_memory_relation"] == "different-explicit-corpus"
    assert report["metric_interval"] == 100
    assert any(item.get("train_kind") == "replay" for item in report["history"])
    assert any(item.get("train_kind") == "replay-memory" for item in report["history"])

    restored = JointTrainingRun.from_checkpoint(
        continuation_dir / "last.pt",
        continuation.dataset,
        continuation.memory_corpus,
        continuation.world_corpus,
        continuation.goal_corpus,
        output_dir=continuation_dir,
        replay_dataset=dataset,
        replay_memory_corpus=replay_memory_corpus,
    )
    assert restored.evaluate_only()["checkpoint_read_only"] is True
    assert restored.evaluate_only()["replay_memory_relation"] == "different-explicit-corpus"
