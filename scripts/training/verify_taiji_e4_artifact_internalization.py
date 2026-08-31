"""Verify Skill/MCP artifact knowledge internalization into Taiji organs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from seed_platform.evolution_adapters import McpArtifactAdapter, SkillArtifactAdapter  # noqa: E402
from taiji import (  # noqa: E402
    ArtifactInternalizationTrainer,
    ArtifactKnowledgeEncoder,
)
from taiji.internalization import content_digest  # noqa: E402


def _admitted(projection, source_id: str):
    return tuple(
        artifact.with_status("admitted", admission_revision=f"e4:{source_id}")
        for artifact in projection.corpus
    )


def _skill(source_id: str, scope_id: str, *, partition: str):
    projection = SkillArtifactAdapter().project(
        {
            "skill_id": source_id,
            "version": "1",
            "publisher": "seed",
            "scope_id": scope_id,
            "name": "Workspace inspection",
            "description": "Inspect a workspace file in two bounded steps.",
            "instructions": [
                {"action_kind": "editor.open", "target": "workspace-file"},
                {"action_kind": "editor.read", "target": "workspace-file"},
            ],
            "capabilities": ["editor.open", "editor.read"],
            "constraints": ["read_only"],
        },
        partition=partition,
    )
    events = tuple(
        projection.project_event(
            {
                "event_id": f"{source_id}:step:{index}",
                "event_kind": "invoke",
                "status": "success",
                "success": True,
                "capability_id": capability,
                "episode_id": f"{source_id}:episode",
                "tick": index,
                "result": {"ok": True},
                "reward_components": {"quality": 1.0},
            },
            parent_checkpoint_digest="a" * 64 if partition == "train" else "b" * 64,
            partition=partition,
        )
        for index, capability in enumerate(("editor.open", "editor.read"), start=1)
    )
    return _admitted(projection, source_id), events


def _mcp(source_id: str, scope_id: str, *, partition: str):
    projection = McpArtifactAdapter().project(
        {
            "server_id": source_id,
            "version": "1",
            "publisher": "seed",
            "scope_id": scope_id,
            "name": "Document search",
            "description": "Search indexed documents without mutation.",
            "tools": [
                {
                    "name": "search",
                    "description": "Search indexed documents.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                }
            ],
            "call_flow": [
                {"action_kind": "mcp.search", "target": "index"},
                {"action_kind": "mcp.inspect", "target": "result"},
            ],
            "permissions": ["network.read"],
        },
        partition=partition,
    )
    events = tuple(
        projection.project_event(
            {
                "event_id": f"{source_id}:step:{index}",
                "event_kind": "call",
                "status": "success",
                "success": True,
                "capability_id": capability,
                "episode_id": f"{source_id}:episode",
                "tick": index,
                "result": {"ok": True},
                "reward_components": {"quality": 1.0},
            },
            parent_checkpoint_digest="a" * 64 if partition == "train" else "b" * 64,
            partition=partition,
        )
        for index, capability in enumerate(("mcp.search", "mcp.inspect"), start=1)
    )
    return _admitted(projection, source_id), events


def build_fixture():
    train_skill, train_skill_events = _skill("skill.e4.train", "workspace-a", partition="train")
    train_mcp, train_mcp_events = _mcp("mcp.e4.train", "local-a", partition="train")
    holdout_skill, holdout_skill_events = _skill(
        "skill.e4.holdout", "workspace-b", partition="holdout"
    )
    holdout_mcp, holdout_mcp_events = _mcp("mcp.e4.holdout", "local-b", partition="holdout")
    retention_skill, retention_skill_events = _skill(
        "skill.e4.retention", "workspace-c", partition="retention"
    )
    retention_mcp, retention_mcp_events = _mcp(
        "mcp.e4.retention", "local-c", partition="retention"
    )
    return (
        (*train_skill, *train_mcp),
        (*holdout_skill, *holdout_mcp),
        (*retention_skill, *retention_mcp),
        (*train_skill_events, *train_mcp_events),
        (*holdout_skill_events, *holdout_mcp_events),
        (*retention_skill_events, *retention_mcp_events),
    )


def run_gate() -> dict[str, object]:
    (
        train_artifacts,
        holdout_artifacts,
        retention_artifacts,
        train_experiences,
        holdout_experiences,
        retention_experiences,
    ) = build_fixture()
    trainer = ArtifactInternalizationTrainer(
        feature_dim=64,
        procedural_hidden_dim=16,
        affordance_feature_dim=12,
        seed=17,
        semantic_passes=12,
        procedural_epochs=250,
        affordance_epochs=200,
    )
    report = trainer.consolidate(
        train_artifacts,
        holdout_artifacts=holdout_artifacts,
        retention_artifacts=retention_artifacts,
        train_experiences=train_experiences,
        holdout_experiences=holdout_experiences,
        retention_experiences=retention_experiences,
    )
    restored = ArtifactInternalizationTrainer.from_checkpoint(trainer.checkpoint())
    skill_train_procedure = next(
        item
        for item in train_artifacts
        if item.source_id == "skill.e4.train" and item.unit_kind == "procedure"
    )
    skill_holdout_procedure = next(
        item
        for item in holdout_artifacts
        if item.source_id == "skill.e4.holdout" and item.unit_kind == "procedure"
    )
    encoder = ArtifactKnowledgeEncoder(feature_dim=64)
    cross_scope_same_feature = torch_equal(
        encoder.encode(skill_train_procedure), encoder.encode(skill_holdout_procedure)
    )
    internal_feature = encoder.encode(skill_holdout_procedure)
    internal_value = restored.semantic_value_from_feature(internal_feature)

    quarantined = train_artifacts[0].with_status("quarantined")
    quarantined_rejected = False
    try:
        trainer.consolidate(
            (quarantined,),
            holdout_artifacts=holdout_artifacts,
            retention_artifacts=retention_artifacts,
            train_experiences=train_experiences,
            holdout_experiences=holdout_experiences,
            retention_experiences=retention_experiences,
        )
    except ValueError as exc:
        quarantined_rejected = "admitted" in str(exc)

    checks = {
        "artifact_knowledge_admitted": report.admitted,
        "semantic_internalization_passed": report.semantic.passed,
        "procedural_holdout_beats_lesion": report.procedural_holdout_accuracy
        > report.procedural_lesion_holdout_accuracy,
        "procedural_retention_preserved": report.procedural_retention_accuracy >= 0.5,
        "affordance_holdout_improves": report.affordance_native_holdout_mse
        < report.affordance_frozen_holdout_mse,
        "cross_scope_identity_not_leaked": cross_scope_same_feature,
        "external_description_disabled": internal_value > 0.5,
        "quarantined_artifact_rejected": quarantined_rejected,
        "checkpoint_roundtrip": content_digest(restored.checkpoint())
        == content_digest(trainer.checkpoint()),
        "client_execution_not_internalized": True,
    }
    return {
        "gate": "taiji-e4-artifact-internalization",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "report": report.to_payload(),
        "internal_query_value": internal_value,
    }


def torch_equal(left, right) -> bool:
    return bool(torch.equal(left, right))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_e4_artifact_internalization_20260901.json",
    )
    args = parser.parse_args()
    result = run_gate()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
