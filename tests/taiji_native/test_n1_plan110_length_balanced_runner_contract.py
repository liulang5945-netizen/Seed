from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from scripts.training import build_taiji_n1_hard_neighbor_training_schedule as base_builder
from scripts.training import build_taiji_n1_length_balanced_training_schedule as schedule_builder
from scripts.training import run_taiji_n1_length_balanced_training as runner


def test_expected_material_ledger_is_exact_for_each_training_path():
    expected = runner.expected_length_ledger()
    assert len(expected) == 14
    assert sum(v for (path, _, _), v in expected.items() if path == "raw") == 12_288
    assert sum(v for (path, _, _), v in expected.items() if path == "hard") == 12_288
    assert expected[("raw", "lookup", 6)] == 1536
    assert expected[("raw", "lookup", 7)] == 1536
    assert expected[("hard", "revision", 6)] == 1536
    assert expected[("hard", "revision", 7)] == 1536
    assert expected[("raw", "transition", None)] == 3072


def test_length_ledger_accepts_only_complete_raw_hard_counts():
    expected = runner.expected_length_ledger()
    assert runner.summarize_length_ledger(Counter(expected))["valid"] is True
    bad = Counter(expected)
    bad[("raw", "lookup", 7)] -= 1
    bad[("raw", "lookup", 6)] += 1
    summary = runner.summarize_length_ledger(bad)
    assert summary["valid"] is False
    assert summary["raw_slots"] == 12_288
    assert summary["hard_slots"] == 12_288


def test_scheduler_consumes_every_raw_and_hard_slot_four_times_without_model_calls():
    schedule = schedule_builder.read_json(runner.TRAIN_SCHEDULE)
    items, manifest_sha = base_builder.train_items_from_frozen_manifest()
    assert manifest_sha == base_builder.MANIFEST_SHA256
    base_state = SimpleNamespace(
        ACTIVE_HARD_COUNTS=Counter(),
        ACTIVE_CYCLE_COUNTS=Counter(),
    )
    ledger = Counter()
    scheduled = runner.make_candidate_scheduler(base_state, schedule, items, ledger)
    for step in range(runner.UPDATES_PER_MODEL):
        item, second = scheduled(step, items, {})
        assert isinstance(item, dict)
        if item["kind"] in {"lookup", "unknown"}:
            assert isinstance(second, dict)
            assert second["observations"] == item["observations"]
        else:
            assert second is None
    audit = runner.summarize_length_ledger(ledger)
    assert audit["valid"] is True
    assert audit["raw_slots"] == audit["hard_slots"] == 12_288
    assert base_state.ACTIVE_CYCLE_COUNTS == Counter({0: 3072, 1: 3072, 2: 3072, 3: 3072})
    assert sum(base_state.ACTIVE_HARD_COUNTS.values()) == 12_288


def test_frozen_candidate_inputs_and_checkpoint_root_are_safe_before_preflight():
    schedule, checks = runner.verify_candidate_sources()
    assert checks["accepted"] is True
    assert len(schedule["items"]) == len(schedule["raw_views"]) == 3072
    assert runner.CHECKPOINT_ROOT.is_relative_to(runner.ROOT)
    assert not runner.CHECKPOINT_ROOT.exists()
    assert not runner.PREFLIGHT.exists()
    assert not runner.ATTEMPT.exists()
    assert not runner.REPORT.exists()


def test_restore_worker_command_uses_candidate_runner_process(tmp_path):
    checkpoint = tmp_path / "s11_position_zero.pt"
    command = runner.restore_worker_command(checkpoint, python=Path(sys.executable))
    assert command == [
        str(Path(sys.executable)),
        str(Path(runner.__file__).resolve()),
        "--restore-worker",
        str(checkpoint),
    ]


def test_candidate_plan_and_frozen_length_seeds_match_protocol():
    plan = runner.PLAN.read_text(encoding="utf-8")
    assert "owner-authorized and frozen" in plan
    assert runner.LENGTH_ASSIGNMENT_SEED == schedule_builder.LENGTH_ASSIGNMENT_SEED == 11010
    assert runner.VALUE_MAP_SEED == schedule_builder.VALUE_MAP_SEED == 11011
    assert runner.UPDATES_PER_MODEL == 24_576
    assert runner.CPU_LIMIT_SECONDS == 18_000
