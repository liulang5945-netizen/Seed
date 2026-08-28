from __future__ import annotations

import hashlib
import sys

from fastapi.testclient import TestClient

from api.app import create_app
from api.seed_runtime import SeedRuntime
from seed import Seed
from seed_platform.programming_languages import ProgrammingLanguageRegistry
from seed_platform.workbench import (
    CapabilitySnapshot,
    WorkbenchActionRequest,
    WorkbenchEnvironment,
)
from taiji import ActionIntent, TSKV8Adapter


def test_workbench_snapshot_is_content_addressed() -> None:
    snapshot = CapabilitySnapshot.default()
    restored = CapabilitySnapshot.from_payload(snapshot.to_payload())

    assert restored == snapshot
    assert snapshot.get("workspace.list") is not None
    assert snapshot.get("editor.diagnostics.read").enabled is False  # type: ignore[union-attr]
    assert snapshot.get("workspace.programming_language.resolve") is not None
    assert snapshot.get("editor.set_language") is not None
    assert snapshot.get("workspace.apply_patch") is not None
    assert snapshot.get("terminal.run") is not None

    tampered = snapshot.to_payload()
    tampered["capabilities"][0]["description"] = "tampered"
    try:
        CapabilitySnapshot.from_payload(tampered)
    except ValueError as exc:
        assert "digest" in str(exc)
    else:  # pragma: no cover - protects the red-path contract
        raise AssertionError("tampered workbench snapshot was accepted")


def test_read_only_environment_reads_and_rejects_escape(tmp_path) -> None:
    (tmp_path / "src").mkdir()
    with (tmp_path / "src" / "main.py").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        handle.write("print('seed')\n")
    environment = WorkbenchEnvironment(tmp_path)

    listed = environment.execute_tool("workspace.list", {"path": "src"})
    assert listed.success is True
    assert environment.last_result["entries"][0]["path"] == "src/main.py"

    read = environment.execute_tool("workspace.read", {"path": "src/main.py"})
    assert read.success is True
    assert environment.last_result["content"] == "print('seed')\n"
    assert environment.last_result["digest"]

    escaped = environment.execute_tool("workspace.read", {"path": "../outside.txt"})
    assert escaped.success is False
    assert environment.last_result["error_code"] == "unsafe_path"


def test_runtime_status_exposes_seed_capabilities_without_legacy() -> None:
    app = create_app(startup_tasks=False)
    with TestClient(app) as client:
        response = client.get("/api/runtime/status")
        assert response.status_code == 200
        tools = response.json()["tools"]
        names = {item["name"] for item in tools["tools"]}
        assert "workspace.list" in names
        assert tools["count"] == len(tools["tools"])

        capabilities = client.get("/api/workbench/capabilities")
        assert capabilities.status_code == 200
        assert capabilities.json()["snapshot_id"]
        languages = client.get("/api/workbench/programming-languages")
        assert languages.status_code == 200
        assert languages.json()["languages"]


def test_taiji_intent_reaches_workbench_and_audit(tmp_path, monkeypatch) -> None:
    with (tmp_path / "README.md").open("w", encoding="utf-8", newline="") as handle:
        handle.write("Taiji workbench\n")
    monkeypatch.setattr(
        "seed_platform.workbench.get_setting",
        lambda key, default=None: str(tmp_path) if key == "workspace_path" else default,
    )
    runtime = SeedRuntime(Seed(episode_id="workbench-canary"))
    runtime._workbench_environment = WorkbenchEnvironment(tmp_path)
    environment = runtime._workbench_environment
    snapshot_id = environment.capability_snapshot.snapshot_id
    intent = ActionIntent(
        intent_id="intent-read-readme",
        kind="workspace.read",
        parameters={"path": "README.md"},
        expected_outcome="read the workspace file",
        confidence=1.0,
        tick=runtime.model.tick,
    )

    result = runtime.execute_workbench_intent(
        intent, snapshot_id=snapshot_id, learn=False
    )

    assert result["outcome"]["status"] == "success"
    assert result["outcome"]["result"]["content"] == "Taiji workbench\n"
    assert result["tool_call"]["intent_id"] == intent.intent_id
    assert [event["phase"] for event in result["events"]] == [
        "planned",
        "policy",
        "executing",
        "outcome",
    ]


def test_taiji_tool_intent_bridge_does_not_select_the_intent(tmp_path) -> None:
    adapter = TSKV8Adapter()
    environment = WorkbenchEnvironment(tmp_path)
    intent = ActionIntent(
        intent_id="intent-list-root",
        kind="workspace.list",
        parameters={"path": "."},
        tick=adapter.tick,
    )

    call, outcome = adapter.execute_tool_intent(intent, environment, learn=False)

    assert call.intent_id == intent.intent_id
    assert call.tool_name == intent.kind
    assert outcome.intent_id == intent.intent_id


def test_programming_language_evidence_uses_content_manifest_and_ambiguity(
    tmp_path,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='seed'\n", encoding="utf-8"
    )
    source = tmp_path / "main.py"
    source.write_text("def answer(value: int):\n    return value\n", encoding="utf-8")
    environment = WorkbenchEnvironment(tmp_path)

    resolved = environment.execute_tool(
        "workspace.programming_language.resolve",
        {"path": "main.py"},
    )

    assert resolved.success is True
    assert environment.last_result["programming_language_id"] == "python"
    assert environment.last_result["editor_language_id"] == "python"
    assert {item["source"] for item in environment.last_result["provenance"]} >= {
        "extension",
        "content",
        "manifest",
    }
    assert environment.last_result["execution_snapshot"]["runner_id"] == "python"
    assert environment.last_result["execution_snapshot"]["lsp_id"] == "pyright"
    assert environment.last_result["explanation"]["selected_language"] == "python"
    assert environment.last_result["explanation"]["evidence"]

    header = tmp_path / "shared.h"
    header.write_text("#include <stdio.h>\n", encoding="utf-8")
    environment.execute_tool(
        "workspace.programming_language.resolve",
        {"path": "shared.h"},
    )
    assert environment.last_result["selection_state"] == "ambiguous"

    registry = ProgrammingLanguageRegistry.default()
    no_extension = registry.resolve(
        path="run",
        content="#!/usr/bin/env node\nconsole.log('seed')\n",
        file_digest="node-script",
    )
    assert no_extension.programming_language_id == "javascript"
    assert any(item.source == "shebang" for item in no_extension.provenance)

    wrong_extension = registry.resolve(
        path="notes.txt",
        content="def answer(value):\n    return value\n",
        file_digest="python-content",
    )
    assert wrong_extension.programming_language_id == "python"
    filename_only = registry.resolve(
        path="unknown.py",
        content="42\n",
        file_digest="filename-only",
    )
    assert filename_only.selection_state == "ambiguous"

    vue = registry.resolve(
        path="App.vue",
        content='<template><main /></template>\n<script lang="ts">const x: number = 1</script>\n',
        file_digest="vue-ts",
        manifest_names={"package.json", "tsconfig.json"},
    )
    assert vue.programming_language_id == "vue"
    assert vue.editor_language_id == "html"

    lsp = registry.resolve(
        path="wrong-name.data",
        content="",
        file_digest="lsp-rust",
        lsp_language_id="rust",
    )
    assert lsp.programming_language_id == "rust"
    assert any(item.source == "lsp" for item in lsp.provenance)

    monorepo_typescript = registry.resolve(
        path="packages/ui/index.ts",
        content="interface Props { title: string }\n",
        file_digest="monorepo-typescript",
        manifest_names={"package.json", "pyproject.toml", "tsconfig.json"},
    )
    assert monorepo_typescript.programming_language_id == "typescript"
    assert all(
        item.language_id != "python"
        for item in monorepo_typescript.provenance
        if item.source == "manifest"
    )

    markdown = registry.resolve(
        path="guide.md",
        content="# Guide\n\n```python\nprint('seed')\n```\n",
        file_digest="markdown-code-block",
    )
    assert markdown.programming_language_id == "markdown"

    notebook = registry.resolve(
        path="analysis.ipynb",
        content='{"cells": [], "nbformat": 4, "nbformat_minor": 5}',
        file_digest="notebook",
    )
    assert notebook.programming_language_id == "notebook"
    assert notebook.editor_language_id == "json"


def test_programming_language_override_is_content_bound_and_checkpointable(
    tmp_path,
) -> None:
    source = tmp_path / "main.py"
    source.write_text("print('seed')\n", encoding="utf-8")
    environment = WorkbenchEnvironment(tmp_path)
    override = environment.execute_tool(
        "editor.set_language",
        {
            "path": "main.py",
            "programming_language_id": "javascript",
            "user_override": True,
        },
    )
    assert override.success is True
    assert environment.last_result["selection_state"] == "user_override"

    environment.execute_tool(
        "workspace.programming_language.resolve",
        {"path": "main.py"},
    )
    assert environment.last_result["programming_language_id"] == "javascript"

    restored = WorkbenchEnvironment(
        tmp_path,
        snapshot=environment.capability_snapshot,
        programming_language_registry=ProgrammingLanguageRegistry.default(),
    )
    restored.restore_language_state(environment.language_state_checkpoint())
    restored.execute_tool(
        "workspace.programming_language.resolve",
        {"path": "main.py"},
    )
    assert restored.last_result["programming_language_id"] == "javascript"

    source.write_text("#!/usr/bin/env node\nconst answer = 1;\n", encoding="utf-8")
    restored.execute_tool(
        "workspace.programming_language.resolve",
        {"path": "main.py"},
    )
    assert restored.last_result["programming_language_id"] == "javascript"
    assert restored.last_result["selection_state"] != "user_override"


def test_taiji_language_selection_requires_evidence_or_explicit_override(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "shared.h").write_text("#include <stdio.h>\n", encoding="utf-8")
    (tmp_path / "main.py").write_text("def answer():\n    return 1\n", encoding="utf-8")
    monkeypatch.setattr(
        "seed_platform.workbench.get_setting",
        lambda key, default=None: str(tmp_path) if key == "workspace_path" else default,
    )
    runtime = SeedRuntime(Seed(episode_id="language-policy-canary"))
    runtime._workbench_environment = WorkbenchEnvironment(tmp_path)
    environment = runtime._workbench_environment
    snapshot_id = environment.capability_snapshot.snapshot_id

    ambiguous = runtime.execute_workbench_intent(
        ActionIntent(
            intent_id="intent-ambiguous-language",
            kind="editor.set_language",
            parameters={
                "path": "shared.h",
                "programming_language_id": "c",
                "user_override": False,
            },
            confidence=1.0,
            tick=runtime.model.tick,
        ),
        snapshot_id=snapshot_id,
        learn=False,
    )
    assert ambiguous["policy"]["decision"] == "ask_user"
    assert ambiguous["policy"]["reason_code"] == "language_evidence_ambiguous"
    assert ambiguous["outcome"]["status"] == "rejected"

    autonomous = runtime.execute_workbench_intent(
        ActionIntent(
            intent_id="intent-python-language",
            kind="editor.set_language",
            parameters={
                "path": "main.py",
                "programming_language_id": "python",
                "user_override": False,
            },
            confidence=1.0,
            tick=runtime.model.tick,
        ),
        snapshot_id=snapshot_id,
        learn=False,
    )
    assert autonomous["policy"]["decision"] == "allow"
    assert autonomous["outcome"]["result"]["programming_language_id"] == "python"
    assert (
        autonomous["outcome"]["result"]["execution_snapshot"]["runner_id"] == "python"
    )

    explicit = runtime.execute_workbench_intent(
        ActionIntent(
            intent_id="intent-explicit-language",
            kind="editor.set_language",
            parameters={
                "path": "shared.h",
                "programming_language_id": "cpp",
                "user_override": True,
            },
            confidence=0.1,
            tick=runtime.model.tick,
        ),
        snapshot_id=snapshot_id,
        learn=False,
    )
    assert explicit["policy"]["decision"] == "allow"
    assert explicit["outcome"]["result"]["selection_state"] == "user_override"


def test_file_transactions_are_digest_checked_and_undoable(tmp_path) -> None:
    source = tmp_path / "main.py"
    original = b"print('seed')\n"
    updated = b"print('taiji')\n"
    source.write_bytes(original)
    environment = WorkbenchEnvironment(tmp_path)

    start = original.decode("utf-8").index("seed")
    end = start + len("seed")
    applied = environment.execute_tool(
        "workspace.apply_patch",
        {
            "path": "main.py",
            "before_digest": hashlib.sha256(original).hexdigest(),
            "patch": {
                "kind": "text_replace",
                "operations": [{"start": start, "end": end, "text": "taiji"}],
            },
            "expected_after_digest": hashlib.sha256(updated).hexdigest(),
        },
    )
    assert applied.success is True
    patch_transaction = environment.last_result["transaction"]
    assert patch_transaction["undo_token"]
    assert source.read_bytes() == updated

    stale = environment.execute_tool(
        "workspace.apply_patch",
        {
            "path": "main.py",
            "before_digest": hashlib.sha256(original).hexdigest(),
            "patch": {
                "kind": "text_replace",
                "operations": [{"start": start, "end": end, "text": "seed"}],
            },
            "expected_after_digest": hashlib.sha256(original).hexdigest(),
        },
    )
    assert stale.success is False
    assert environment.last_result["error_code"] == "transaction_conflict"

    undone = environment.execute_tool(
        "workspace.undo", {"undo_token": patch_transaction["undo_token"]}
    )
    assert undone.success is True
    assert source.read_bytes() == original

    created = environment.execute_tool(
        "workspace.create", {"path": "created.txt", "content": "created\n"}
    )
    assert created.success is True
    create_token = environment.last_result["transaction"]["undo_token"]
    assert (tmp_path / "created.txt").is_file()
    assert environment.execute_tool(
        "workspace.undo", {"undo_token": create_token}
    ).success
    assert not (tmp_path / "created.txt").exists()

    renamed = environment.execute_tool(
        "workspace.rename",
        {
            "path": "main.py",
            "new_path": "renamed.py",
            "before_digest": hashlib.sha256(original).hexdigest(),
        },
    )
    assert renamed.success is True
    rename_token = environment.last_result["transaction"]["undo_token"]
    assert (tmp_path / "renamed.py").is_file()
    assert environment.execute_tool(
        "workspace.undo", {"undo_token": rename_token}
    ).success
    assert source.is_file()

    deleted = environment.execute_tool(
        "workspace.delete",
        {"path": "main.py", "before_digest": hashlib.sha256(original).hexdigest()},
    )
    assert deleted.success is True
    delete_token = environment.last_result["transaction"]["undo_token"]
    assert not source.exists()
    assert environment.execute_tool(
        "workspace.undo", {"undo_token": delete_token}
    ).success
    assert source.read_bytes() == original


def test_terminal_run_is_bounded_and_shell_free(tmp_path) -> None:
    environment = WorkbenchEnvironment(tmp_path)
    completed = environment.execute_tool(
        "terminal.run",
        {
            "argv": [sys.executable, "-c", "print('seed')"],
            "cwd": ".",
            "timeout_seconds": 5,
            "output_limit": 1024,
            "env": {},
            "env_allowlist": [],
            "expected_artifacts": [],
        },
    )
    assert completed.success is True
    assert environment.last_result["shell"] is False
    assert environment.last_result["exit_code"] == 0
    assert "seed" in environment.last_result["stdout"]

    invalid_env = environment.execute_tool(
        "terminal.run",
        {
            "argv": [sys.executable, "-c", "pass"],
            "env": {"SEED_TEST": "1"},
            "env_allowlist": [],
        },
    )
    assert invalid_env.success is False
    assert environment.last_result["error_code"] == "invalid_parameters"

    timed_out = environment.execute_tool(
        "terminal.run",
        {
            "argv": [sys.executable, "-c", "import time; time.sleep(1)"],
            "timeout_seconds": 0.05,
        },
    )
    assert timed_out.success is False
    assert environment.last_result["timed_out"] is True


def test_write_and_terminal_capabilities_require_approval(tmp_path) -> None:
    environment = WorkbenchEnvironment(tmp_path)
    snapshot_id = environment.capability_snapshot.snapshot_id
    for capability_id in ("workspace.apply_patch", "terminal.run"):
        request = WorkbenchActionRequest(
            request_id=f"request-{capability_id}",
            intent_id=f"intent-{capability_id}",
            capability_id=capability_id,
            parameters={},
            snapshot_id=snapshot_id,
            confidence=1.0,
        )
        decision = environment.policy_for(request)
        assert decision.decision == "ask_user"
        assert decision.reason_code == "capability_requires_approval"
