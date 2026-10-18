from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "analyze_taiji_n1_length_balance_failure_from_shards.py"


def _module():
    spec = importlib.util.spec_from_file_location("n1_length_balance_shard_attribution", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _rows():
    module = _module()
    rows = []
    baseline_cells = (
        [("south", "left")] * 6
        + [("south", "right")] * 5
        + [("east", "right")] * 4
        + [("west", "left")] * 9
        + [("east", "left")] * 6
        + [("north", "left")] * 5
        + [("north", "right")] * 8
        + [("west", "right")] * 5
    )
    assert len(baseline_cells) == 48
    for parent in range(48):
        direction, turn = baseline_cells[parent]
        for condition in module.CONDITIONS:
            free_exact = not (
                (parent == 1 and condition == "turn_flip") or (parent == 2 and condition == "both")
            )
            teacher_exact = not (parent == 1 and condition == "turn_flip")
            rows.append(
                {
                    "parent_id": f"train-{parent:03d}",
                    "condition": condition,
                    "kind": "transition",
                    "initial_direction": direction,
                    "turn": turn,
                    "free_exact": free_exact,
                    "teacher_exact": teacher_exact,
                    "output_hex": ("north" if free_exact else "south").encode().hex(),
                }
            )
    return rows


def test_four_condition_attribution_reconstructs_joint_and_cell_counts() -> None:
    module = _module()
    summary = module.summarize_factorial_view(_rows())

    assert summary["items"] == 192
    assert summary["parents"] == 48
    assert summary["joint_four_exact"]["free"] == 46
    assert summary["joint_four_exact"]["teacher"] == 47
    assert summary["condition_counts"]["turn_flip"]["free_exact"] == 47
    assert summary["condition_counts"]["both"]["free_exact"] == 47
    assert summary["joint_four_exact"]["free_correct_count_distribution"] == {
        "0": 0,
        "1": 0,
        "2": 0,
        "3": 2,
        "4": 46,
    }
    assert summary["baseline_direction_turn_cells"]["south/left"]["parents"] == 6
    assert summary["baseline_direction_turn_cells"]["west/left"]["parents"] == 9
    assert all(
        values["items"] == summary["baseline_direction_turn_cells"][cell]["parents"]
        for cell, cell_values in summary["baseline_direction_turn_cells"].items()
        for values in cell_values.values()
        if isinstance(values, dict)
    )


def test_incomplete_or_duplicate_parent_condition_groups_are_refused() -> None:
    module = _module()
    rows = _rows()
    try:
        module.summarize_factorial_view(rows[:-1])
    except ValueError as exc:
        assert "expected 192" in str(exc)
    else:
        raise AssertionError("incomplete factorial score rows must be rejected")

    duplicate = rows[:-1] + [rows[0]]
    try:
        module.summarize_factorial_view(duplicate)
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate parent-condition rows must be rejected")


def test_the_offline_analyzer_does_not_import_model_runtime_or_torch() -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any(
        name == "torch" or name.startswith("api.seed_runtime") or "training_runner" in name
        for name in imported
    )
    assert '"model_queries": 0' in SCRIPT.read_text(encoding="utf-8")
