from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPO / "scripts" / "training" / "diagnose_taiji_n1_position_five_group_parameter_factorial.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("n1_plan115_parameter_factorial", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_full_binary_group_space_and_anchor_reuse_match_query_budget() -> None:
    module = _module()

    assert len(module.MASKS) == 32
    assert module.HYBRID_MASKS == tuple(range(1, 31))
    assert module.REUSED_MASKS == (0, 31)
    assert module.QUERIES_PER_STATE == 384
    assert module.NEW_QUERIES == 11_520
    assert module.TOTAL_STATE_ROWS == 12_288
    assert set(module.group_sources(0).values()) == {"baseline"}
    assert set(module.group_sources(31).values()) == {"candidate"}
    assert all(len(set(module.group_sources(mask).values())) == 2 for mask in module.HYBRID_MASKS)


def test_state_composition_selects_each_named_group_independently() -> None:
    import torch

    module = _module()
    baseline = {f"{group}.weight": torch.tensor([0.0]) for group in module.GROUPS}
    candidate = {f"{group}.weight": torch.tensor([1.0]) for group in module.GROUPS}
    group_by_name = {f"{group}.weight": group for group in module.GROUPS}

    reader_only = module.compose_state(baseline, candidate, group_by_name, 1)

    assert all(
        reader_only[f"{group}.weight"].item() == (1.0 if group == "reader" else 0.0)
        for group in module.GROUPS
    )
    all_candidate = module.compose_state(baseline, candidate, group_by_name, 31)
    assert all(value.item() == 1.0 for value in all_candidate.values())


def test_factorial_shapley_recovers_additive_effects_and_interaction() -> None:
    module = _module()
    summaries = {}
    for mask in module.MASKS:
        reader = bool(mask & 1)
        writer = bool(mask & 2)
        score = 10 + 2 * reader + 3 * writer + 4 * reader * writer
        summaries[mask] = {
            view: {
                "joint_four_exact": score,
                "by_condition": {condition: score for condition in module.CONDITIONS},
            }
            for view in module.VIEWS
        }

    result = module.factorial_attribution(summaries)
    selected = result["original"]["joint_four_exact"]
    assert selected["total_delta"] == 9
    assert selected["groups"]["reader"]["mean_edge_delta"] == 4
    assert abs(selected["groups"]["reader"]["shapley_delta"] - 4) < 1e-9
    assert selected["groups"]["writer"]["mean_edge_delta"] == 5
    assert abs(selected["groups"]["writer"]["shapley_delta"] - 5) < 1e-9
    assert abs(sum(value["shapley_delta"] for value in selected["groups"].values()) - 9) < 1e-9


def test_contract_has_no_training_optimizer_or_checkpoint_write_path() -> None:
    module = _module()
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert module.PROTOCOL["model_queries"] == 11_520
    assert module.PROTOCOL["parameter_updates"] == 0
    assert module.PROTOCOL["optimizer_steps"] == 0
    assert module.PROTOCOL["checkpoint_saves"] == 0
    assert module.PROTOCOL["dev_queries"] == module.PROTOCOL["holdout_queries"] == 0
    assert module.PROTOCOL["confirmation_queries"] == 0
    assert not calls.intersection({"train_step", "backward", "step", "answer", "generate", "save"})
    assert "torch.save" not in source
