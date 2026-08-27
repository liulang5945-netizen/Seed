"""Evaluate the versioned world-schema registry lifecycle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_p3_open_set import (  # noqa: E402
    _config,
    _training_corpus,
    _world,
)
from taiji import (  # noqa: E402
    TSKV8Adapter,
    WorldAction,
    WorldDynamicsLearner,
    WorldSchema,
    WorldSchemaBudgetError,
    WorldSchemaConflictError,
    WorldSchemaRegistry,
)

MANIFEST_FORMAT = "taiji-p3-schema-registry-manifest-v1"
REPORT_FORMAT = "taiji-p3-schema-registry-v1"


def _fit(seed: int) -> WorldDynamicsLearner:
    corpus = _training_corpus()
    learner = WorldDynamicsLearner(
        WorldSchema.from_corpus(corpus), hidden_dim=24, seed=int(seed) + 9100
    )
    learner.fit(corpus.train, epochs=80, learning_rate=0.01)
    return learner


def evaluate_seed(seed: int) -> dict[str, object]:
    learner = _fit(seed)
    registry = learner.schema_registry
    old_schema = learner.schema
    old_input = learner._input_layer.weight.detach().clone()
    old_output = learner._output_layer.weight.detach().clone()
    old_input_keys = old_schema.input_feature_keys
    old_output_keys = old_schema.state_feature_keys

    alias_state = _world("registry", 0, target_id="legacy-target")
    alias_action = WorldAction(
        "registry:alias",
        "assemble",
        0,
        actor_id="agent",
        target_id="legacy-target",
        parameters={"workspace_count": 2.0},
    )
    alias_registered = learner.register_schema_alias("legacy-target", "target")
    alias_prediction = learner.predict(alias_state, alias_action)
    alias_stable = bool(
        alias_registered
        and alias_prediction.state.objects[1].object_id == "target"
        and registry.active_version == 0
    )
    alias_conflict = False
    try:
        registry.register_alias("legacy-target", "other-target")
    except WorldSchemaConflictError:
        alias_conflict = True

    target_id = "target:registry:holdout"
    new_state = _world("registry", 1, target_id=target_id, phase=2)
    new_action = WorldAction(
        "registry:secure",
        "secure",
        1,
        actor_id="agent",
        target_id=target_id,
        parameters={"workspace_count": 2.0, "strength": 0.75},
    )
    learner.register_open_set(new_state, action=new_action)
    expanded_version = registry.active_version
    expanded_schema = learner.schema
    new_input_keys = expanded_schema.input_feature_keys
    new_output_keys = expanded_schema.state_feature_keys
    old_weights_preserved = bool(
        all(
            torch.equal(
                learner._input_layer.weight[:, new_input_keys.index(key)], old_input[:, index]
            )
            for index, key in enumerate(old_input_keys)
        )
        and all(
            torch.equal(learner._output_layer.weight[new_output_keys.index(key)], old_output[index])
            for index, key in enumerate(old_output_keys)
        )
    )
    new_prediction = learner.predict(new_state, new_action)
    old_prediction = learner.predict(_world("registry", 0, target_id="target"), alias_action)
    mixed_schema = bool(
        expanded_version == 1
        and target_id in learner.schema.object_ids
        and {"secure"} <= set(learner.schema.action_kinds)
        and old_prediction.state.tick == 1
        and new_prediction.state.tick == 2
    )

    feedback_key = ("relation", "agent", "secured", target_id)
    feedback_accept = learner.record_schema_feedback(feedback_key, 1.0)
    contradiction_rejected = not learner.record_schema_feedback(feedback_key, 0.0)
    conflict_stable = bool(
        feedback_accept
        and contradiction_rejected
        and registry.contradiction_count == 1
        and registry.feature_confidence[feedback_key] > 0.0
    )

    strength_key = ("parameter", "strength")
    prune_version_before = registry.active_version
    learner.prune_schema(strength_key, evidence_ids=("resource-pressure",))
    pruned_version = registry.active_version
    prune_tombstone = bool(
        pruned_version == prune_version_before + 1
        and strength_key in registry.tombstones
        and strength_key not in learner.schema.parameter_names
    )
    readd_blocked = False
    try:
        registry.propose_open_set(states=(new_state,), actions=(new_action,))
    except WorldSchemaConflictError:
        readd_blocked = True
    rollback = learner.rollback_schema(expanded_version)
    rollback_restored = bool(
        rollback
        and registry.active_version == expanded_version
        and strength_key[1] in learner.schema.parameter_names
    )

    budget_registry = WorldSchemaRegistry(
        old_schema,
        max_feature_count=old_schema.input_dim + old_schema.state_dim,
    )
    budget_blocked = False
    try:
        budget_registry.propose_open_set(states=(new_state,), actions=(new_action,))
    except WorldSchemaBudgetError:
        budget_blocked = True

    model = TSKV8Adapter(_config(seed), episode_id=f"registry:{seed}:a")
    model.attach_world_dynamics(learner)
    model.begin_episode(f"registry:{seed}:b")
    restored = TSKV8Adapter.from_native_checkpoint(model.native_checkpoint())
    restored_learner = restored._world_dynamics
    checkpoint = bool(
        restored_learner is not None
        and restored_learner.schema_registry.active_version == expanded_version
        and restored_learner.schema_registry.revision_versions == (0, 1, 2)
        and restored_learner.schema_registry.aliases == (("legacy-target", "target"),)
        and restored_learner.schema_registry.contradiction_count == 2
        and restored.begin_episode(f"registry:{seed}:c") is None
    )
    checkpoint_rollback = bool(
        restored_learner is not None
        and restored_learner.rollback_schema(0)
        and restored_learner.schema_registry.active_version == 0
    )
    return {
        "seed": int(seed),
        "alias_stable": alias_stable,
        "alias_conflict": alias_conflict,
        "old_weights_preserved": old_weights_preserved,
        "mixed_schema": mixed_schema,
        "conflict_stable": conflict_stable,
        "prune_tombstone": prune_tombstone,
        "readd_blocked": readd_blocked,
        "rollback_restored": rollback_restored,
        "budget_blocked": budget_blocked,
        "checkpoint": checkpoint,
        "checkpoint_rollback": checkpoint_rollback,
        "active_version": registry.active_version,
        "lineage_events": len(registry.lineage),
    }


def build_manifest() -> dict[str, object]:
    return {
        "format": MANIFEST_FORMAT,
        "task": "cross-episode world schema registry lifecycle",
        "controls": [
            "canonical object alias merge",
            "mixed old/new schema prediction",
            "revision proposal and checkpoint lineage",
            "contradictory outcome feedback fail-closed",
            "resource-budget fail-closed growth",
            "prune and tombstone re-add blocking",
            "network schema rollback",
            "adapter checkpoint continuation",
        ],
        "boundary": "schema lifecycle safety; not open-domain semantics or general intelligence",
    }


def evaluate(*, seeds: tuple[int, ...] = (11, 29, 47)) -> dict[str, object]:
    runs = [evaluate_seed(seed) for seed in seeds]
    metrics = (
        "alias_stable",
        "alias_conflict",
        "old_weights_preserved",
        "mixed_schema",
        "conflict_stable",
        "prune_tombstone",
        "readd_blocked",
        "rollback_restored",
        "budget_blocked",
        "checkpoint",
        "checkpoint_rollback",
    )
    aggregate = {f"{name}_min": min(float(bool(run[name])) for run in runs) for name in metrics}
    passed = all(aggregate[f"{name}_min"] >= 1.0 for name in metrics)
    aggregate["passed"] = passed
    return {
        "format": REPORT_FORMAT,
        "seeds": runs,
        "aggregate": aggregate,
        "gate": {
            "passed": passed,
            "criterion": "all registry identity, conflict, budget, lifecycle and checkpoint controls must pass for every seed",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p3_schema_registry_manifest_20260827.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p3_schema_registry_20260827.json",
    )
    args = parser.parse_args()
    report = evaluate()
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(build_manifest(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
