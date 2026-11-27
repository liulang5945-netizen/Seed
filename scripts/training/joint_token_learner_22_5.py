"""Typed fusion learner persistence and explicit source-only parent migration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from scripts.training import evidence_a_learner_22_5 as base
from scripts.training.task_baselines_22_4_a import TaskSequenceBaseline
from scripts.training.task_goal_learner_22_5 import TYPES as OLD_TYPES
from taiji.u2_history import assert_values_equal, atomic_load, detached_tree, value_digest
from taiji.u2_joint_subject import JointGoalSubject, JointQuerySubject
from taiji.u2_task_goal import ActiveQuerySubject

ROOT = Path(__file__).resolve().parents[2]
TYPES = {
    **OLD_TYPES,
    "native-joint-query": JointQuerySubject,
    "native-joint-control": JointGoalSubject,
}
EXTRA = (
    "scripts/training/task_goal_learner_22_5.py",
    "scripts/training/joint_token_learner_22_5.py",
)


def save_learner(path, model, optimizer, architecture, lineage, **metadata):
    return base.save_learner(
        path,
        model,
        optimizer,
        architecture,
        lineage,
        native_types=TYPES,
        extra_sources=EXTRA,
        **metadata,
    )


def load_learner(path):
    return base.load_learner(path, native_types=TYPES, extra_sources=EXTRA)


def load_parent(name):
    contract = json.loads(
        (ROOT / "plans/manifests/taiji_u2_joint_subject_contract_20261113.json").read_text(
            encoding="utf-8"
        )
    )
    recipe = contract["parents"][name]
    path = Path(recipe["checkpoint"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == recipe["sha256"]
    payload = atomic_load(path)
    current = base.source_pins(("scripts/training/task_goal_learner_22_5.py",))
    assert set(current) - set(payload["source_pins"]) == set(contract["new_cognitive_sources"])
    assert not set(payload["source_pins"]) - set(current)
    overlays = {row["source"]: row for row in contract["source_overlays"]}
    for source, pin in payload["source_pins"].items():
        if current[source] != pin:
            archived = overlays[source]
            assert archived["sha256"] == pin
            assert (
                hashlib.sha256(
                    (ROOT / archived["archive"]).read_text(encoding="utf-8-sig").encode()
                ).hexdigest()
                == pin
            )
    if name == "D":
        assert payload["architecture"] == "native-goal-query"
        model, _ = ActiveQuerySubject.from_bundle(payload["model"])
    else:
        assert name == "TF-actual" and payload["architecture"] == "transformer"
        bundle = payload["model"]
        model = TaskSequenceBaseline(**bundle["description"])
        model = model.to(next(iter(bundle["parameters"].values())))
        model.load_state_dict(bundle["parameters"])
        model.parameter_revision = bundle["revision"]
    model.train(payload["training"])
    assert value_digest(model.state_dict()) == recipe["parameter_digest"]
    optimizer = base.fresh_optimizer(model)
    optimizer.load_state_dict(detached_tree(payload["optimizer"]))
    torch.set_rng_state(payload["torch_rng"])
    return model, optimizer, payload


def fork_with_optimizer(parent, optimizer, cls):
    candidate = cls.from_parent(parent)
    old_count = len(list(parent.parameters()))
    new_count = len(list(candidate.parameters()))
    inherited = detached_tree(optimizer.state_dict())
    assert len(inherited["param_groups"]) == 1
    assert inherited["param_groups"][0]["params"] == list(range(old_count))
    before = value_digest(inherited["state"])
    inherited["param_groups"][0]["params"].extend(range(old_count, new_count))
    other = base.fresh_optimizer(candidate)
    other.load_state_dict(inherited)
    assert value_digest(other.state_dict()["state"]) == before
    for index in range(old_count, new_count):
        assert index not in other.state_dict()["state"]
    assert_values_equal(
        parent.state_dict(),
        {name: candidate.state_dict()[name] for name in parent.state_dict()},
        "joint.parent_parameters",
    )
    return candidate, other
