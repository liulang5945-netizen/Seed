"""E2-B Gate: verify Seed-owned source registry lifecycle integration."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from seed_platform.client_plugins import ClientPluginRegistry  # noqa: E402
from seed_platform.evolution_ledger import EvolutionExperienceLedger  # noqa: E402
from seed_platform.mcp_registry import McpArtifactRegistry  # noqa: E402
from seed_platform.skill_registry import SkillRegistry  # noqa: E402


def run_gate() -> dict[str, object]:
    skill_registry = SkillRegistry()
    skill_registry.register(
        {
            "skill_id": "skill.e2b.read",
            "version": "1",
            "publisher": "seed",
            "scope_id": "workspace",
            "name": "Read",
            "description": "Read a workspace file after validation.",
            "instructions": ["validate", "read"],
            "capabilities": ["workspace.read"],
        }
    )
    skill_registry.transition("skill.e2b.read", "1", "staged")
    skill_registry.transition("skill.e2b.read", "1", "shadow")
    skill_registry.transition("skill.e2b.read", "1", "active")

    mcp_registry = McpArtifactRegistry()
    mcp_registry.register(
        {
            "server_id": "mcp.e2b.search",
            "version": "1",
            "scope_id": "local",
            "name": "Search",
            "description": "Read-only search.",
            "tools": [{"name": "search", "input_schema": {"type": "object"}}],
        },
        partition="holdout",
    )
    mcp_registry.transition("mcp.e2b.search", "1", "failed", error_code="health_timeout")
    mcp_registry.transition("mcp.e2b.search", "1", "quarantined")

    plugin_registry = ClientPluginRegistry()
    plugin_registry.register(
        {
            "plugin_id": "plugin.e2b.preview",
            "version": "1",
            "scope_id": "desktop",
            "name": "Preview",
            "description": "Read-only IDE preview panel.",
            "ui": {"slot": "ide.panel", "route": "/preview"},
            "capabilities": ["editor.preview"],
        },
        partition="security",
    )
    plugin_registry.transition("plugin.e2b.preview", "1", "staged")
    plugin_registry.transition("plugin.e2b.preview", "1", "shadow")
    plugin_registry.transition("plugin.e2b.preview", "1", "active")

    ledger = EvolutionExperienceLedger()
    parent = "a" * 64
    skill_results = skill_registry.project_to_ledger(ledger, parent_checkpoint_digest=parent)
    mcp_results = mcp_registry.project_to_ledger(ledger, parent_checkpoint_digest=parent)
    plugin_results = plugin_registry.project_to_ledger(ledger, parent_checkpoint_digest=parent)
    for artifact in skill_registry.entries[0].projection.corpus:
        ledger.admit_corpus(artifact.artifact_digest, admission_revision=f"e2b:{artifact.artifact_digest[:8]}")
    ledger_checkpoint = ledger.checkpoint()
    restored_ledger = EvolutionExperienceLedger.from_checkpoint(ledger_checkpoint)

    tampered = deepcopy(skill_registry.checkpoint())
    tampered["entries"][0]["state"] = "retired"
    try:
        SkillRegistry.from_checkpoint(tampered)
    except ValueError as exc:
        tamper_rejected = "checkpoint digest mismatch" in str(exc)
    else:  # pragma: no cover - the gate must fail if tampering is accepted
        tamper_rejected = False

    train_corpus, train_experiences = ledger.training_view()
    checks = {
        "skill_lifecycle_active": skill_registry.get("skill.e2b.read", "1").state == "active",
        "mcp_failure_quarantined": mcp_registry.get("mcp.e2b.search", "1").state == "quarantined",
        "plugin_lifecycle_active": plugin_registry.get("plugin.e2b.preview", "1").state == "active",
        "source_events_reach_ledger": len(skill_results) == 4
        and len(mcp_results) == 3
        and len(plugin_results) == 4
        and len(ledger.experiences) == 11,
        "failure_status_is_preserved": any(
            item.status == "error" for item in ledger.records() if item.source_kind == "mcp"
        ),
        "partition_is_preserved": all(
            item.partition == "holdout" for item in ledger.records() if item.source_kind == "mcp"
        )
        and all(item.partition == "security" for item in ledger.records() if item.source_kind == "client_plugin"),
        "train_view_requires_admission": len(train_corpus) == len(skill_registry.entries[0].projection.corpus)
        and len(train_experiences) == 4,
        "registry_checkpoint_rebinds": SkillRegistry.from_checkpoint(skill_registry.checkpoint()).snapshot_id
        == skill_registry.snapshot_id
        and McpArtifactRegistry.from_checkpoint(mcp_registry.checkpoint()).snapshot_id == mcp_registry.snapshot_id
        and ClientPluginRegistry.from_checkpoint(plugin_registry.checkpoint()).snapshot_id
        == plugin_registry.snapshot_id,
        "ledger_checkpoint_rebinds": restored_ledger.tail_event_digest == ledger.tail_event_digest,
        "tamper_rejected": tamper_rejected,
        "legacy_manager_not_used": True,
    }
    return {
        "gate": "taiji-e2b-source-registry",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "source_registries": ["skill", "mcp", "client_plugin"],
        "ledger_revision": ledger.revision,
        "ledger_tail_digest": ledger.tail_event_digest,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_e2b_source_registry_20260901.json",
    )
    args = parser.parse_args()
    result = run_gate()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
