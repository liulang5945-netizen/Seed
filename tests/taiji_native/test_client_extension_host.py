from __future__ import annotations

from copy import deepcopy

import pytest

from seed_platform.client_extension_host import (
    ClientExtensionHost,
    ClientPluginManifest,
    DisposerNode,
    ExtensionDependencyError,
    ExtensionDisposalError,
    ExtensionHostBusyError,
    ExtensionHostError,
)


def _manifest(
    version: str = "1.0.0",
    *,
    service_dependencies: tuple[tuple[str, str], ...] = (),
    state_schema_version: int = 1,
    migration_id: str = "",
    capability_ids: tuple[str, ...] = ("editor.preview",),
) -> ClientPluginManifest:
    return ClientPluginManifest(
        plugin_id="seed.preview",
        version=version,
        scope="workspace",
        slots=("ide.panel", "route"),
        capability_ids=capability_ids,
        service_dependencies=service_dependencies,
        state_schema_version=state_schema_version,
        migration_id=migration_id,
        migration_version="1.0.0" if migration_id else "",
        disposer_id="seed.preview.dispose",
        disposer_version="1.0.0",
        metadata={"kind": "read_only"},
    )


def test_prepare_commit_is_atomic_and_content_addressed() -> None:
    host = ClientExtensionHost(capability_snapshot_id="capability:1")
    before = host.snapshot
    prepared = host.prepare(
        (_manifest(),),
        capability_snapshot_id="capability:1",
        available_capabilities=("editor.preview",),
        dependency_health={},
        states={"seed.preview": {"open_count": 0}},
    )

    assert host.snapshot == before
    committed = host.commit(prepared)
    assert committed.snapshot_id != before.snapshot_id
    assert host.state("seed.preview") == {"open_count": 0}
    assert committed.snapshot_id == host.snapshot.snapshot_id
    assert {item.state for item in host.lifecycle_records} >= {"prepared", "active"}


def test_state_migration_is_trusted_host_callback_and_preserves_old_state() -> None:
    host = ClientExtensionHost()
    v1 = _manifest()
    host.mount(
        v1,
        capability_snapshot_id="capability:1",
        available_capabilities=("editor.preview",),
        state={"open_count": 2},
    )
    host.register_state_migrator(
        "seed.preview",
        1,
        2,
        "seed.preview.state.v2",
        lambda state: {**state, "schema": 2, "open_count": int(state["open_count"]) + 1},
    )
    v2 = _manifest(version="2.0.0", state_schema_version=2, migration_id="seed.preview.state.v2")
    prepared = host.prepare(
        (v2,),
        capability_snapshot_id="capability:1",
        available_capabilities=("editor.preview",),
    )
    host.commit(prepared)

    assert host.state("seed.preview") == {"open_count": 3, "schema": 2}
    assert host.snapshot.manifests == (v2,)


def test_dependency_loss_quarantines_and_recovery_requires_explicit_remount() -> None:
    host = ClientExtensionHost()
    manifest = _manifest(service_dependencies=(("workbench", "1.0"),))
    host.mount(
        manifest,
        capability_snapshot_id="capability:1",
        available_capabilities=("editor.preview",),
        dependency_health={"workbench": True},
    )
    assert host.report_dependency("workbench", False) == ("seed.preview",)
    assert host.active_manifests == ()
    assert host.dependency_health == {"workbench": False}

    with pytest.raises(ExtensionDependencyError):
        host.prepare(
            (manifest,),
            capability_snapshot_id="capability:1",
            available_capabilities=("editor.preview",),
        )
    host.report_dependency("workbench", True)
    host.mount(
        manifest,
        capability_snapshot_id="capability:1",
        available_capabilities=("editor.preview",),
    )
    assert host.active_manifests == (manifest,)


def test_inflight_calls_block_snapshot_change_until_drained() -> None:
    host = ClientExtensionHost()
    manifest = _manifest()
    host.mount(
        manifest,
        capability_snapshot_id="capability:1",
        available_capabilities=("editor.preview",),
    )
    host.begin_call("seed.preview")
    prepared = host.prepare(
        (),
        capability_snapshot_id="capability:1",
        available_capabilities=(),
    )

    assert host.drain("seed.preview") is False
    with pytest.raises(ExtensionHostBusyError):
        host.commit(prepared)
    assert host.active_manifests == (manifest,)
    host.end_call("seed.preview")
    assert host.drain("seed.preview") is True
    host.commit(prepared)
    assert host.active_manifests == ()


def test_recursive_disposer_is_reverse_order_and_idempotent() -> None:
    host = ClientExtensionHost()
    manifest = _manifest()
    host.mount(
        manifest,
        capability_snapshot_id="capability:1",
        available_capabilities=("editor.preview",),
    )
    calls: list[str] = []
    root = DisposerNode("root", lambda: calls.append("root"))
    root.add(DisposerNode("first", lambda: calls.append("first")))
    nested = root.add(DisposerNode("nested"))
    nested.add(DisposerNode("nested-child", lambda: calls.append("nested-child")))
    root.add(DisposerNode("last", lambda: calls.append("last")))
    host.attach_disposer("seed.preview", root)

    host.retire("seed.preview")
    assert calls == ["last", "nested-child", "first", "root"]
    assert host.release("seed.preview") == ()


def test_disposer_failure_is_audited_without_reactivating_extension() -> None:
    host = ClientExtensionHost()
    manifest = _manifest()
    host.mount(
        manifest,
        capability_snapshot_id="capability:1",
        available_capabilities=("editor.preview",),
    )
    root = DisposerNode("root", lambda: (_ for _ in ()).throw(RuntimeError("cleanup failed")))
    host.attach_disposer("seed.preview", root)

    with pytest.raises(ExtensionDisposalError, match="cleanup failed"):
        host.retire("seed.preview")
    assert host.active_manifests == ()
    assert any(item.state == "failed" for item in host.lifecycle_records)


def test_rollback_checkpoint_and_manifest_boundary_are_fail_closed() -> None:
    host = ClientExtensionHost()
    v1 = _manifest()
    host.mount(
        v1,
        capability_snapshot_id="capability:1",
        available_capabilities=("editor.preview",),
        state={"version": 1},
    )
    parent_id = host.snapshot.snapshot_id
    v2 = _manifest(version="2.0.0")
    host.commit(
        host.prepare(
            (v2,),
            capability_snapshot_id="capability:2",
            available_capabilities=("editor.preview",),
        )
    )
    assert host.snapshot.capability_snapshot_id == "capability:2"
    host.rollback(parent_id)
    assert host.snapshot.capability_snapshot_id == "capability:1"
    assert host.snapshot.manifests == (v1,)
    assert host.state("seed.preview") == {"version": 1}

    checkpoint = host.checkpoint()
    restored = ClientExtensionHost.from_checkpoint(checkpoint)
    assert restored.checkpoint() == checkpoint
    assert "executor" not in repr(checkpoint).lower()
    assert "source_path" not in repr(checkpoint).lower()

    tampered = deepcopy(checkpoint)
    tampered["states"]["seed.preview"]["version"] = 99
    with pytest.raises(ValueError, match="checkpoint digest mismatch"):
        ClientExtensionHost.from_checkpoint(tampered)

    with pytest.raises(ValueError, match="executable-source"):
        ClientPluginManifest.from_payload(
            {
                **v1.to_payload(),
                "module": "untrusted.plugin",
            }
        )


def test_protected_root_shell_cannot_be_claimed_by_a_manifest() -> None:
    manifest = ClientPluginManifest(
        plugin_id="seed.shell.escape",
        version="1.0.0",
        scope="workspace",
        slots=("desktop.root_shell",),
        disposer_id="seed.shell.dispose",
        disposer_version="1.0.0",
    )
    with pytest.raises(ExtensionHostError, match="unsupported slots"):
        ClientExtensionHost().prepare(
            (manifest,),
            capability_snapshot_id="capability:1",
            available_capabilities=(),
        )
