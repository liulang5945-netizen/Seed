"""Typed task-goal learners and explicit parameter/moment-preserving parent forks."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from scripts.training import evidence_a_learner_22_5 as base
from taiji.u2_evidence_subject import EvidenceSubject
from taiji.u2_history import assert_values_equal, atomic_load, detached_tree, value_digest
from taiji.u2_task_goal import ActiveQuerySubject, PersistentGoalSubject

ROOT = Path(__file__).resolve().parents[2]
TYPES = {
    **base.NATIVE_TYPES,
    "native-goal-query": ActiveQuerySubject,
    "native-goal-control": PersistentGoalSubject,
}
EXTRA = ("scripts/training/task_goal_learner_22_5.py",)


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


def load_verified_parent():
    contract = json.loads(
        (ROOT / "plans/manifests/taiji_u2_p3_6_interface_contract_20261113.json").read_text(
            encoding="utf-8"
        )
    )
    path = Path(contract["parent"]["checkpoint"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != contract["parent"]["sha256"]:
        raise ValueError("task_goal_parent_file_mismatch")
    payload = atomic_load(path)
    if payload["architecture"] != "native-query":
        raise ValueError("task_goal_parent_architecture_mismatch")
    current = base.source_pins()
    overlays = {row["source"]: row for row in contract["source_overlays"]}
    if set(current) - set(payload["source_pins"]) != {"taiji/u2_task_goal.py"}:
        raise ValueError("unregistered_task_goal_source_additions")
    if set(payload["source_pins"]) - set(current):
        raise ValueError("task_goal_source_removed")
    for name, pin in payload["source_pins"].items():
        if pin == current[name]:
            continue
        row = overlays.get(name)
        if row is None or row["sha256"] != pin:
            raise ValueError("task_goal_source_migration_not_registered")
        archived = hashlib.sha256(
            (ROOT / row["archive"]).read_text(encoding="utf-8-sig").encode()
        ).hexdigest()
        if archived != pin:
            raise ValueError("task_goal_archive_mismatch")
    parent, _ = EvidenceSubject.from_bundle(payload["model"])
    parent.train(payload["training"])
    if value_digest(parent.state_dict()) != contract["parent"]["parameter_digest"]:
        raise ValueError("task_goal_parent_parameters_mismatch")
    optimizer = base.fresh_optimizer(parent)
    optimizer.load_state_dict(payload["optimizer"])
    return parent, optimizer, payload


def fork_with_optimizer(parent, optimizer, cls):
    candidate = cls.from_parent(parent)
    old_names = list(dict(parent.named_parameters()))
    new_names = list(dict(candidate.named_parameters()))
    if new_names[:-1] != old_names:
        raise ValueError("task_goal_optimizer_order_mismatch")
    inherited = detached_tree(optimizer.state_dict())
    if len(inherited["param_groups"]) != 1 or inherited["param_groups"][0]["params"] != list(
        range(len(old_names))
    ):
        raise ValueError("unsupported_task_goal_optimizer_layout")
    old_state = copy.deepcopy(inherited["state"])
    inherited["param_groups"][0]["params"].append(len(old_names))
    child_optimizer = base.fresh_optimizer(candidate)
    child_optimizer.load_state_dict(inherited)
    assert_values_equal(
        old_state, child_optimizer.state_dict()["state"], "task_goal.inherited_moments"
    )
    if candidate.native_task_projection.weight in child_optimizer.state:
        raise ValueError("new_task_goal_moment_must_start_absent")
    return candidate, child_optimizer
