from __future__ import annotations

from copy import deepcopy

import pytest

from seed_platform.client_plugins import ClientPluginRegistry
from seed_platform.evolution_ledger import EvolutionExperienceLedger
from seed_platform.mcp_registry import McpArtifactRegistry
from seed_platform.skill_registry import SkillRegistry


def test_skill_registry_tracks_versioned_lifecycle_and_projects_idempotently() -> None:
    registry = SkillRegistry()
    skill = {
        "skill_id": "skill.registry.read",
        "version": "1",
        "publisher": "seed",
        "scope_id": "workspace",
        "name": "Read",
        "description": "Read a file after validation.",
        "instructions": ["validate", "read"],
        "source_code": "open(path)",
    }
    projection = registry.register(skill)
    assert registry.get("skill.registry.read", "1").state == "discovered"
    assert registry.register(deepcopy(skill)) == projection
    with pytest.raises(ValueError, match="another digest"):
        changed = deepcopy(skill)
        changed["description"] = "changed"
        registry.register(changed)
    with pytest.raises(ValueError, match="invalid source lifecycle"):
        registry.transition("skill.registry.read", "1", "active")
    registry.transition("skill.registry.read", "1", "staged")
    registry.transition("skill.registry.read", "1", "shadow")
    active = registry.transition("skill.registry.read", "1", "active")
    assert active.state == "active"
    assert len(active.events) == 4
    assert all("source_code" not in repr(event) for event in active.events)

    ledger = EvolutionExperienceLedger()
    first_results = registry.project_to_ledger(ledger, parent_checkpoint_digest="a" * 64)
    second_results = registry.project_to_ledger(ledger, parent_checkpoint_digest="a" * 64)
    assert len(first_results) == 4
    assert all(result.accepted for result in first_results)
    assert all(result.duplicate for result in second_results)
    assert {item.source_kind for item in ledger.experiences} == {"skill"}

    restored = SkillRegistry.from_checkpoint(registry.checkpoint())
    assert restored.snapshot_id == registry.snapshot_id
    assert restored.get("skill.registry.read", "1").state == "active"


def test_mcp_registry_preserves_schema_lineage_and_quarantines_failed_source() -> None:
    registry = McpArtifactRegistry()
    projection = registry.register(
        {
            "server_id": "mcp.registry.search",
            "version": "1",
            "scope_id": "local",
            "name": "Search",
            "description": "Read-only search.",
            "tools": [
                {
                    "name": "search",
                    "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
                }
            ],
        },
        partition="holdout",
    )
    assert projection.source_kind == "mcp"
    failed = registry.transition(
        "mcp.registry.search",
        "1",
        "failed",
        error_code="health_timeout",
    )
    quarantined = registry.transition("mcp.registry.search", "1", "quarantined")
    assert failed.state == "failed"
    assert quarantined.state == "quarantined"
    assert quarantined.events[-2]["status"] == "error"
    assert quarantined.events[-1]["event_kind"] == "lifecycle.quarantined"

    restored = McpArtifactRegistry.from_checkpoint(registry.checkpoint())
    assert restored.get("mcp.registry.search", "1").state == "quarantined"
    assert restored.snapshot_id == registry.snapshot_id


def test_client_plugin_registry_is_declarative_and_version_changes_are_distinct() -> None:
    registry = ClientPluginRegistry()
    first = registry.register(
        {
            "plugin_id": "plugin.registry.preview",
            "version": "1",
            "scope_id": "desktop",
            "name": "Preview",
            "description": "Adds an IDE preview panel.",
            "ui": {"slot": "ide.panel", "route": "/preview"},
            "capabilities": ["editor.preview"],
            "entrypoint_path": "plugin.py",
        },
        partition="security",
    )
    second = registry.register(
        {
            "plugin_id": "plugin.registry.preview",
            "version": "2",
            "scope_id": "desktop",
            "name": "Preview",
            "description": "Adds an IDE preview panel with diagnostics.",
            "ui": {"slot": "ide.panel", "route": "/preview"},
            "capabilities": ["editor.preview", "editor.diagnostics"],
        },
        partition="security",
    )
    assert first.source_version == "1"
    assert second.source_version == "2"
    registry.transition("plugin.registry.preview", "1", "staged")
    registry.transition("plugin.registry.preview", "1", "shadow")
    registry.transition("plugin.registry.preview", "1", "active")
    assert registry.get("plugin.registry.preview", "2").state == "discovered"
    assert "entrypoint_path" not in repr(registry.checkpoint())

    ledger = EvolutionExperienceLedger()
    registry.project_to_ledger(ledger, parent_checkpoint_digest="b" * 64)
    assert {item.source_kind for item in ledger.experiences} == {"client_plugin"}
    assert all(item.partition == "security" for item in ledger.experiences)
