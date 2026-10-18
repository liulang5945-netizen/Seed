from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "diagnose_taiji_n1_position_weight_history_crossover.py"


def _module():
    spec = importlib.util.spec_from_file_location("n1_plan111_weight_history_crossover", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _rows(*, failing_parent: int | None = None):
    module = _module()
    directions = ("north", "east", "south", "west")
    turns = ("left", "right")
    rows = []
    for view in module.VIEWS:
        for parent in range(module.PARENTS):
            parent_id = f"train-parent-{parent:03d}"
            direction = directions[parent % len(directions)]
            turn = turns[(parent // len(directions)) % len(turns)]
            for condition in module.CONDITIONS:
                exact = not (parent == failing_parent and condition == "both")
                rows.append(
                    {
                        "view": view,
                        "case_id": f"{parent_id}:{condition}",
                        "parent_id": parent_id,
                        "condition": condition,
                        "initial_direction": direction,
                        "turn": turn,
                        "free_exact": exact,
                        "teacher_exact": exact,
                        "output_hex": ("north" if exact else "south").encode().hex(),
                    }
                )
    return rows


def test_four_condition_summary_preserves_views_cells_and_joint_parent_counts() -> None:
    module = _module()
    rows = _rows(failing_parent=0)

    summary = module.summarize_combination(rows)

    assert len(rows) == 384
    assert set(summary) == {"original", "rename"}
    assert summary["original"]["joint_four_exact"] == 47
    assert summary["rename"]["joint_four_exact"] == 47
    assert summary["original"]["by_condition"]["both"] == 47
    assert summary["original"]["cells"]["north/left"]["parents"] == 6
    assert summary["original"]["cells"]["north/left"]["items"] == 24
    assert summary["original"]["cells"]["north/left"]["joint_four_exact"] == 5


def test_incomplete_duplicate_and_mixed_parent_rows_are_refused() -> None:
    module = _module()
    rows = _rows()
    try:
        module.summarize_combination(rows[:-1])
    except ValueError as exc:
        assert "384" in str(exc)
    else:
        raise AssertionError("incomplete crossover rows must be refused")

    duplicate = rows[:-1] + [rows[0]]
    try:
        module.summarize_combination(duplicate)
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("duplicate crossover cases must be refused")


def test_interpretation_only_attributes_one_sided_mixed_checkpoint_rescue() -> None:
    module = _module()
    passing = module.summarize_combination(_rows())
    failing = module.summarize_combination(_rows(failing_parent=0))
    below_gate = {view: {**summary, "joint_four_exact": 43} for view, summary in failing.items()}
    summaries = {
        "baseline_weights+baseline_history": passing,
        "baseline_weights+candidate_history": passing,
        "candidate_weights+baseline_history": below_gate,
        "candidate_weights+candidate_history": below_gate,
    }

    assert module.interpretation(summaries) == "evidence_toward_candidate_network_weights"
    summaries["baseline_weights+candidate_history"] = below_gate
    assert module.interpretation(summaries) == "interaction_or_2x2_not_separating_sources"


def test_contract_freezes_scope_and_has_no_training_or_checkpoint_write_calls() -> None:
    module = _module()
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert module.PROTOCOL["model_queries"] == 1_536
    assert module.PROTOCOL["checkpoint_loads"] == 2
    assert module.PROTOCOL["parameter_updates"] == 0
    assert module.PROTOCOL["optimizer_steps"] == 0
    assert module.PROTOCOL["checkpoint_saves"] == 0
    assert module.PROTOCOL["dev_queries"] == module.PROTOCOL["holdout_queries"] == 0
    assert module.PROTOCOL["confirmation_queries"] == 0
    assert module.ANCHOR_COMBINATIONS[0] == ("baseline_weights", "baseline_history")
    assert module.ANCHOR_COMBINATIONS[1] == ("candidate_weights", "candidate_history")
    assert "measure" in calls
    assert not calls.intersection({"train_step", "backward", "step", "save", "save_pretrained"})
    assert "torch.save" not in source
