from __future__ import annotations

import pytest

from scripts.training import run_taiji_n1_hard_neighbor_training as training


def test_training_protocol_preserves_fixed_trajectory_and_zero_eval_budget():
    protocol = training.protocol()
    assert protocol["schema"] == "n1-hard-neighbor-fresh-paired-training-v1"
    assert protocol["seeds"] == [11, 12, 13]
    assert protocol["arms"] == ["position", "terminal", "cut"]
    assert protocol["updates_per_model"] == 24576
    assert protocol["total_updates"] == 9 * 24576
    assert protocol["hard_views_per_model"] == 12288
    assert protocol["raw_views_per_model"] == 12288
    assert protocol["hard_schedule_sha256"] == training.SCHEDULE_SHA256
    assert protocol["learning_rate_schedule"] == {
        "schema": "n1-final-cycle-step-decay-v1",
        "base_lr": 0.001,
        "final_cycle_start": 18432,
        "final_cycle_lr": 0.00025,
        "adam_moments": "continuous_unmodified",
    }
    assert protocol["dev_queries"] == 0
    assert protocol["holdout_queries"] == 0
    assert protocol["confirmation_queries"] == 0


def test_fixed_learning_rate_boundary():
    assert training.verify_learning_rate_schedule()
    assert training.learning_rate(0) == 0.001
    assert training.learning_rate(18431) == 0.001
    assert training.learning_rate(18432) == 0.00025
    assert training.learning_rate(24576) == 0.00025
    with pytest.raises(ValueError):
        training.learning_rate(-1)
    with pytest.raises(ValueError):
        training.learning_rate(24579)


def test_runtime_schedule_alternates_raw_and_frozen_hard_view_without_updates():
    _, g, inputs = training.make_runner(load_inputs=True)
    items = inputs["items"]
    curriculum = inputs["curriculum"]
    schedule = training._verify_schedule_artifact()["items"]
    raw, raw_second = g["scheduled"](0, items, curriculum)
    hard, hard_second = g["scheduled"](1, items, curriculum)
    repeated_hard, repeated_second = g["scheduled"](6145, items, curriculum)
    assert raw == items[0]
    assert raw_second["parent_id"] == raw["id"]
    assert hard == schedule[0]["item"]
    assert hard_second == schedule[0]["second_question"]
    assert repeated_hard == hard
    assert repeated_second == hard_second
    assert hard["split"] == "train"


def test_restore_worker_setup_does_not_parse_v3_manifest(monkeypatch):
    def refuse_manifest_read(*args, **kwargs):
        raise AssertionError("restore worker must not read training or holdout source rows")

    monkeypatch.setattr(
        training.schedule_builder, "train_items_from_frozen_manifest", refuse_manifest_read
    )
    _, _, inputs = training.make_runner(load_inputs=False)
    assert inputs is None


def test_frozen_sources_and_checkpoints_are_unique_and_pinned():
    assert training.sha256(training.FROZEN_RUNNER) == training.FROZEN_RUNNER_SHA256
    assert training.sha256(training.SCHEDULE) == training.SCHEDULE_SHA256
    assert training.sha256(training.schedule_builder.__file__) == training.BUILDER_SHA256
    assert (
        training.sha256(
            training.ROOT
            / "reports/taiji_n1_transition_factorial_paired_train_20261016_attempt.json"
        )
        == training.RUN88_ATTEMPT_SHA256
    )
    assert training.CHECKPOINT_ROOT.name == "n1_hard_neighbor_training_v1_20261017"
    assert (
        training.PREFLIGHT.name
        == "taiji_n1_hard_neighbor_training_20261017_checkpoint_preflight.json"
    )
    assert training.REPORT.name == "taiji_n1_hard_neighbor_training_20261017.json"
