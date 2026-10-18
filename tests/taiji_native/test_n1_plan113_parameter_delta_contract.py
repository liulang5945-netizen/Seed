from __future__ import annotations

import ast
import importlib.util
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPO / "scripts" / "training" / "analyze_taiji_n1_position_weight_checkpoint_parameter_delta.py"
)
TRAJECTORY_SCRIPT = (
    REPO / "scripts" / "training" / "analyze_taiji_n1_seed13_update_group_trajectory.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("n1_plan113_parameter_delta", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _trajectory():
    spec = importlib.util.spec_from_file_location("n1_plan114_update_trajectory", TRAJECTORY_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_five_group_delta_norms_close_over_the_full_tensor_difference() -> None:
    import torch

    module = _module()
    baseline = {
        "reader.weight": torch.tensor([1.0, 2.0]),
        "writer.bias": torch.tensor([0.0]),
        "local.weight": torch.tensor([1.0]),
        "context.weight": torch.tensor([1.0]),
        "relation.weight": torch.tensor([1.0]),
    }
    candidate = {
        "reader.weight": torch.tensor([2.0, 4.0]),
        "writer.bias": torch.tensor([1.0]),
        "local.weight": torch.tensor([1.0]),
        "context.weight": torch.tensor([3.0]),
        "relation.weight": torch.tensor([0.0]),
    }
    groups = {
        "reader": ["reader.weight"],
        "writer": ["writer.bias"],
        "local": ["local.weight"],
        "context": ["context.weight"],
        "relation": ["relation.weight"],
    }

    result = module.compute_group_deltas(
        baseline, candidate, groups, torch, expected_parameters=6, expected_tensor_count=5
    )

    assert math.isclose(result["groups"]["reader"]["delta_l2"], math.sqrt(5.0))
    assert result["groups"]["local"]["changed_scalars"] == 0
    assert result["groups"]["writer"]["changed_scalars"] == 1
    assert result["aggregate"]["scalars"] == 6
    assert result["aggregate"]["tensors"] == 5
    assert math.isclose(
        result["aggregate"]["delta_l2"] ** 2,
        sum(group["delta_l2"] ** 2 for group in result["groups"].values()),
    )


def test_mismatched_state_shapes_are_refused() -> None:
    import torch

    module = _module()
    state = {f"{group}.weight": torch.zeros(2) for group in module.GROUPS}
    candidate = {name: value.clone() for name, value in state.items()}
    candidate["reader.weight"] = torch.zeros(3)
    groups = {group: [f"{group}.weight"] for group in module.GROUPS}

    try:
        module.compute_group_deltas(state, candidate, groups, torch)
    except ValueError as exc:
        assert "shape/dtype" in str(exc)
    else:
        raise AssertionError("mismatched checkpoint tensor shapes must be rejected")


def test_contract_freezes_read_only_checkpoint_scope() -> None:
    module = _module()
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert module.PROTOCOL["checkpoint_payload_reads"] == 2
    assert module.PROTOCOL["model_loads"] == 0
    assert module.PROTOCOL["model_constructions"] == 0
    assert module.PROTOCOL["model_queries"] == 0
    assert module.PROTOCOL["parameter_updates"] == 0
    assert module.PROTOCOL["optimizer_steps"] == 0
    assert module.PROTOCOL["checkpoint_saves"] == 0
    assert module.PROTOCOL["dev_queries"] == module.PROTOCOL["holdout_queries"] == 0
    assert module.PROTOCOL["confirmation_queries"] == 0
    assert "load" in calls
    assert not calls.intersection(
        {"measure", "forward", "train_step", "answer", "generate", "step", "save"}
    )
    assert "torch.save" not in source


def test_step_labeler_covers_four_cycles_and_preserves_transition_inputs() -> None:
    module = _trajectory()
    items = []
    for index in range(192):
        kind = module.KINDS[index // 48]
        parent_id = f"parent-{index:03d}"
        if kind == "transition":
            observations = ["subject faces north.", "subject turns left."]
            question, answer = "direction of subject?", "west"
        else:
            observations = [f"subject uses {'a' * 6}."]
            question, answer = "material of subject?", "a" * 6
        items.append(
            {
                "id": parent_id,
                "split": "train",
                "kind": kind,
                "observations": observations,
                "question": question,
                "answer": answer,
            }
        )

    def resized(item, length):
        if item["kind"] == "transition":
            return dict(item)
        return {**item, "observations": [f"subject uses {'a' * length}."]}

    baseline_hard = []
    candidate_hard = []
    candidate_raw = []
    for schedule_index in range(3072):
        parent = items[schedule_index % 192]
        baseline_hard.append(
            {
                "schedule_index": schedule_index,
                "parent_id": parent["id"],
                "parent_kind": parent["kind"],
                "condition": "original",
                "observation_count": 2,
                "hamming_distance": 1,
                "target_positions": [],
                "item": dict(parent),
                "second_question": None if parent["kind"] == "transition" else "second?",
            }
        )
        length = None if parent["kind"] == "transition" else (6 if schedule_index % 2 == 0 else 7)
        candidate_hard.append(
            {
                **baseline_hard[-1],
                "item": resized(parent, length or 6),
                "material_length_chars": length,
            }
        )
    for parent in items:
        for slot in range(16):
            length = None if parent["kind"] == "transition" else (6 if slot % 2 == 0 else 7)
            candidate_raw.append(
                {
                    "parent_id": parent["id"],
                    "parent_kind": parent["kind"],
                    "raw_slot": slot,
                    "item": resized(parent, length or 6),
                    "material_length_chars": length,
                    "second_question": "second?",
                }
            )

    baseline_schedule = {"items": baseline_hard}
    candidate_schedule = {"items": candidate_hard, "raw_views": candidate_raw}
    labels = module.build_step_labels(
        items, baseline_schedule, candidate_schedule, second_question_fn=lambda _item: "second?"
    )

    assert labels["transition_updates_identical"] == 6_144
    assert len(labels["labels"]["baseline"]) == len(labels["labels"]["candidate"]) == 24_576
