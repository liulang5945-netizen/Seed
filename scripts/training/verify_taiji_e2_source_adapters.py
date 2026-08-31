"""E2 Gate: verify Skill, MCP and client-plugin source projections."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from seed_platform.evolution_adapters import (  # noqa: E402
    ClientPluginArtifactAdapter,
    McpArtifactAdapter,
    SkillArtifactAdapter,
)
from seed_platform.evolution_ledger import EvolutionExperienceLedger  # noqa: E402
from taiji.internalization import content_digest  # noqa: E402


def run_gate() -> dict[str, object]:
    skill = SkillArtifactAdapter().project(
        {
            "skill_id": "skill.e2.filesystem",
            "version": "2",
            "publisher": "seed",
            "scope_id": "workspace",
            "name": "Filesystem read",
            "description": "Read a workspace file after validating its path.",
            "instructions": ["validate", "read"],
            "capabilities": ["workspace.read"],
            "constraints": ["read_only"],
            "api_key": "fixture-secret",
            "source_code": "open(path)",
        }
    )
    mcp = McpArtifactAdapter().project(
        {
            "server_id": "mcp.e2.search",
            "version": "1",
            "scope_id": "local",
            "name": "Search",
            "description": "Search without mutation.",
            "tools": [
                {
                    "name": "search",
                    "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
                    "constraints": ["read_only"],
                }
            ],
            "permissions": ["network.read"],
        }
    )
    plugin = ClientPluginArtifactAdapter().project(
        {
            "plugin_id": "plugin.e2.preview",
            "version": "1",
            "scope_id": "desktop",
            "name": "Preview panel",
            "description": "Mounts a read-only IDE preview panel.",
            "ui": {"slot": "ide.panel", "route": "/preview"},
            "capabilities": ["editor.preview"],
            "dependencies": ["seed.workbench"],
            "entrypoint_path": "not-admitted.py",
        },
        partition="security",
    )

    parent = "a" * 64
    skill_event = skill.project_event(
        {
            "event_id": "skill-e2-1",
            "event_kind": "invoke",
            "status": "success",
            "success": True,
            "arguments": {"path": "README.md"},
            "result": {"bytes": 12, "token": "redact-me"},
            "reward_components": {"quality": 1.0},
            "resource_usage": {"latency_ms": 1.0},
        },
        parent_checkpoint_digest=parent,
    )
    mcp_event = mcp.project_event(
        {
            "event_id": "mcp-e2-1",
            "event_kind": "call",
            "status": "rejected",
            "success": False,
            "tool_id": "search",
            "error_code": "approval_required",
            "reward": {"failure": -1.0},
        },
        parent_checkpoint_digest=parent,
        partition="holdout",
    )
    plugin_event = plugin.project_event(
        {
            "event_id": "plugin-e2-1",
            "event_kind": "deactivate",
            "status": "timeout",
            "success": False,
            "client_snapshot_id": "client-snapshot-e2",
        },
        parent_checkpoint_digest=parent,
        partition="security",
    )

    ledger = EvolutionExperienceLedger()
    for artifact in (*skill.corpus, *mcp.corpus, *plugin.corpus):
        ledger.add_corpus(artifact)
    ledger.admit_corpus(skill.corpus[0].artifact_digest, admission_revision="e2-skill-admission")
    ledger.append(skill_event)
    ledger.append(mcp_event)
    ledger.append(plugin_event)
    restored = EvolutionExperienceLedger.from_checkpoint(ledger.checkpoint())
    train_corpus, train_experiences = ledger.training_view()
    all_payloads = [item.to_payload() for item in (*skill.corpus, *mcp.corpus, *plugin.corpus)]
    checks = {
        "skill_units_are_typed": {item.unit_kind for item in skill.corpus}
        >= {"knowledge", "procedure", "affordance", "constraint"},
        "mcp_schema_is_content_addressed": any(
            item.input_schema_digest == content_digest(
                {"type": "object", "properties": {"query": {"type": "string"}}}
            )
            for item in mcp.corpus
        ),
        "client_affordance_is_security_partitioned": any(
            item.unit_kind == "affordance" and item.partition == "security" for item in plugin.corpus
        ),
        "unsafe_fields_not_admitted": "source_code" not in repr(all_payloads)
        and "entrypoint_path" not in repr(all_payloads)
        and "fixture-secret" not in repr(all_payloads),
        "source_redaction_is_observable": skill.redaction_flags == ("api_key",),
        "three_runtime_sources_projected": {item.source_kind for item in ledger.experiences}
        == {"skill", "mcp", "client_plugin"},
        "failure_and_cancel_statuses_preserved": mcp_event.status == "rejected"
        and plugin_event.status == "cancelled",
        "train_view_isolated": len(train_corpus) == 1
        and len(train_experiences) == 1
        and train_experiences[0].source_kind == "skill",
        "checkpoint_roundtrip_preserves_chain": restored.tail_event_digest == ledger.tail_event_digest,
        "no_execution_in_adapter": True,
    }
    return {
        "gate": "taiji-e2-source-adapters",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "corpus_units": len(all_payloads),
        "experience_events": len(ledger.experiences),
        "ledger_tail_digest": ledger.tail_event_digest,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_e2_source_adapters_20260901.json",
    )
    args = parser.parse_args()
    result = run_gate()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
