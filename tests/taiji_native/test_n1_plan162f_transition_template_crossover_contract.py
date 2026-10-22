from __future__ import annotations

import hashlib
from pathlib import Path

from scripts.training import (
    run_taiji_n1_plan162f_transition_template_crossover_diagnostic as runner,
)


def test_plan_digest_and_query_denominator_are_frozen() -> None:
    assert hashlib.sha256(runner.PLAN.read_bytes()).hexdigest() == runner.PLAN_SHA256
    assert runner.CASES_PER_ENDPOINT == 384
    assert runner.QUERIES_PER_ENDPOINT == 768
    assert runner.TOTAL_QUERIES == 6_912
    assert len(runner.ENDPOINTS) == 9


def test_templates_are_distinct_and_render_same_entity() -> None:
    entity = "erifmmmlgg"
    rendered = {key: value.format(entity=entity) for key, value in runner.TEMPLATES.items()}
    assert set(rendered.values()) == {
        "which direction does erifmmmlgg face after turning?",
        "what direction is erifmmmlgg facing after its turn?",
    }
    assert len(set(rendered.values())) == 2


def test_baseline_key_normalizes_factorial_original_to_its_view() -> None:
    row = {
        "seed": 11,
        "arm": "surface_control",
        "evaluation_set": "factorial_rename",
        "id": "train-002",
        "condition": "turn_flip",
        "case_id": "train-002:turn_flip",
        "raw_hex": "77657374",
        "stop_reason": "boundary",
        "free_exact_stored": False,
    }
    base_view = row["evaluation_set"].removeprefix("factorial_")
    key = (row["seed"], row["arm"], base_view, row["id"], row["condition"], row["case_id"])
    assert key == (11, "surface_control", "rename", "train-002", "turn_flip", "train-002:turn_flip")


def test_readonly_result_paths_are_repository_local() -> None:
    for path in (
        runner.PLAN,
        runner.N155A_RESULT,
        runner.N162B_RESULT,
        runner.PROGRESS,
        runner.RESULT,
    ):
        assert Path(path).resolve().is_relative_to(runner.ROOT)
