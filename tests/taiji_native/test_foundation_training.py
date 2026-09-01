from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from taiji import (
    DelayedMemoryCorpus,
    DelayedMemoryQuery,
    FoundationTrainingDataset,
    FoundationTrainingRun,
    MemoryEpisode,
    MemoryTrainingRun,
    Taiji,
    TaijiConfig,
)


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
