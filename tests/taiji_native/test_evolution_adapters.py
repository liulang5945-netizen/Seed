from __future__ import annotations

import pytest

from seed_platform.evolution_adapters import (
    ClientPluginArtifactAdapter,
    McpArtifactAdapter,
    SkillArtifactAdapter,
)
from seed_platform.evolution_ledger import EvolutionExperienceLedger
from taiji.internalization import content_digest


def test_skill_adapter_emits_governed_units_without_executable_source() -> None:
    projection = SkillArtifactAdapter().project(
        {
            "skill_id": "skill.filesystem.read",
            "version": "2",
            "publisher": "seed-team",
            "scope_id": "workspace",
            "name": "Read files",
            "description": "Safely inspect a workspace file.",
            "instructions": ["validate path", "read file"],
            "capabilities": ["workspace.read"],
            "constraints": ["read only"],
            "examples": [{"request": "show README"}],
            "counterexamples": ["do not read outside workspace"],
            "api_key": "must-not-survive",
            "source_code": "open(path)",
        }
    )

    kinds = [artifact.unit_kind for artifact in projection.corpus]
    assert kinds == ["knowledge", "procedure", "affordance", "constraint", "example", "counterexample"]
    assert projection.scope_id == "workspace"
    assert projection.redaction_flags == ("api_key",)
    for artifact in projection.corpus:
        payload = artifact.to_payload()
        assert "source_code" not in repr(payload)
        assert "must-not-survive" not in repr(payload)
        assert artifact.source_kind == "skill_artifact"
        assert artifact.source_digest == projection.source_digest


def test_mcp_and_client_adapters_preserve_schema_and_affordance_lineage() -> None:
    mcp = McpArtifactAdapter().project(
        {
            "server_id": "mcp.search",
            "version": "1.3",
            "publisher": "provider",
            "scope_id": "local",
            "description": "Read-only search service.",
            "tools": [
                {
                    "name": "search",
                    "description": "Search indexed documents.",
                    "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
                    "constraints": ["read_only"],
                }
            ],
            "permissions": ["network.read"],
            "errors": ["timeout"],
        }
    )
    affordance = next(item for item in mcp.corpus if item.unit_kind == "affordance")
    assert affordance.input_schema_digest == content_digest(
        {"type": "object", "properties": {"query": {"type": "string"}}}
    )
    assert affordance.capability_semantics["kind"] == "mcp_tool"
    assert any(item.unit_kind == "constraint" for item in mcp.corpus)

    plugin = ClientPluginArtifactAdapter().project(
        {
            "plugin_id": "seed.editor.preview",
            "version": "4",
            "scope_id": "desktop",
            "name": "Editor preview",
            "description": "Adds a preview panel.",
            "ui": {"slot": "ide.panel", "route": "/preview"},
            "capabilities": ["editor.preview"],
            "dependencies": ["seed.workbench"],
            "entrypoint_path": "plugin/main.py",
        },
        partition="security",
    )
    assert any(item.unit_kind == "affordance" for item in plugin.corpus)
    assert all(item.partition == "security" for item in plugin.corpus)
    assert "entrypoint_path" not in repr([item.to_payload() for item in plugin.corpus])


def test_runtime_events_from_three_sources_enter_ledger_without_raw_payloads() -> None:
    skill = SkillArtifactAdapter().project(
        {
            "skill_id": "skill.demo",
            "version": "1",
            "name": "Demo",
            "description": "Demo skill",
            "instructions": ["observe"],
        }
    )
    mcp = McpArtifactAdapter().project(
        {
            "server_id": "mcp.demo",
            "version": "1",
            "name": "Demo MCP",
            "description": "Demo server",
            "tools": {"read": {"description": "read"}},
        }
    )
    plugin = ClientPluginArtifactAdapter().project(
        {
            "plugin_id": "plugin.demo",
            "version": "1",
            "name": "Demo plugin",
            "description": "Demo client plugin",
        }
    )
    parent = "a" * 64
    skill_event = skill.project_event(
        {
            "event_id": "skill-call-1",
            "event_kind": "invoke",
            "status": "success",
            "success": True,
            "arguments": {"path": "README.md", "token": "secret"},
            "result": {"text": "private result", "token": "secret-result"},
            "resource_usage": {"latency_ms": 2},
            "reward_components": {"quality": 1.0},
        },
        parent_checkpoint_digest=parent,
    )
    mcp_event = mcp.project_event(
        {
            "event_id": "mcp-call-1",
            "event_kind": "call",
            "status": "rejected",
            "success": False,
            "tool_id": "read",
            "error_code": "approval_required",
            "reward": {"failure": -1.0},
            "tick": 2,
        },
        parent_checkpoint_digest=parent,
        partition="holdout",
    )
    plugin_event = plugin.project_event(
        {
            "event_id": "plugin-stop-1",
            "event_kind": "deactivate",
            "status": "timeout",
            "success": False,
            "client_snapshot_id": "client-snapshot-1",
            "metadata": {"reason": "health timeout"},
            "api_key": "secret",
        },
        parent_checkpoint_digest=parent,
        partition="security",
    )

    assert skill_event.skill_digest == skill.source_digest
    assert mcp_event.mcp_server_digest == mcp.source_digest
    assert plugin_event.plugin_digest == plugin.source_digest
    assert plugin_event.status == "cancelled"
    assert skill_event.result_digest != content_digest(
        {"text": "private result", "token": "secret-result"}
    )
    assert "private result" not in repr(skill_event.to_payload())
    assert "secret" not in repr(skill_event.to_payload())
    assert mcp_event.reward_components == (("failure", -1.0),)

    ledger = EvolutionExperienceLedger()
    for artifact in (*skill.corpus, *mcp.corpus, *plugin.corpus):
        ledger.add_corpus(artifact)
    ledger.admit_corpus(skill.corpus[0].artifact_digest, admission_revision="skill-admission")
    ledger.append(skill_event)
    ledger.append(mcp_event)
    ledger.append(plugin_event)
    train_corpus, train_experiences = ledger.training_view()
    assert len(train_corpus) == 1
    assert [item.source_kind for item in train_experiences] == ["skill"]


def test_runtime_event_rejects_invalid_source_and_negative_resource_usage() -> None:
    projection = SkillArtifactAdapter().project(
        {"skill_id": "skill.invalid", "version": "1", "name": "Invalid"}
    )
    with pytest.raises(ValueError, match="unsupported lifecycle source_kind"):
        from seed_platform.evolution_adapters import runtime_event_to_experience

        runtime_event_to_experience(
            {"event_kind": "invoke", "status": "success", "success": True},
            source_kind="provider",
            source_id="provider",
            source_version="1",
            source_digest=projection.source_digest,
            parent_checkpoint_digest="b" * 64,
        )
    with pytest.raises(ValueError, match="non-negative"):
        projection.project_event(
            {
                "event_id": "bad-resource",
                "event_kind": "invoke",
                "status": "success",
                "success": True,
                "resource_usage": {"latency_ms": -1},
            },
            parent_checkpoint_digest="c" * 64,
        )
