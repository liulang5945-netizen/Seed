"""Verify the Seed-owned client extension host lifecycle Gate."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from seed_platform.client_extension_host import (  # noqa: E402
    ClientExtensionHost,
    ClientPluginManifest,
    DisposerNode,
    ExtensionDisposalError,
    ExtensionHostBusyError,
    ExtensionHostError,
)


def _manifest(
    version: str = "1.0.0",
    *,
    state_schema_version: int = 1,
    migration_id: str = "",
    service_dependencies: tuple[tuple[str, str], ...] = (),
) -> ClientPluginManifest:
    return ClientPluginManifest(
        plugin_id="seed.e5.preview",
        version=version,
        scope="workspace",
        slots=("ide.panel", "route"),
        capability_ids=("editor.preview",),
        service_dependencies=service_dependencies,
        state_schema_version=state_schema_version,
        migration_id=migration_id,
        migration_version="1.0.0" if migration_id else "",
        disposer_id="seed.e5.preview.dispose",
        disposer_version="1.0.0",
        metadata={"effect": "read_only"},
    )


def _mount(
    host: ClientExtensionHost,
    manifest: ClientPluginManifest | None = None,
    *,
    state: dict[str, object] | None = None,
    dependency_health: dict[str, bool] | None = None,
) -> None:
    host.mount(
        manifest or _manifest(),
        capability_snapshot_id="capability:e5",
        available_capabilities=("editor.preview",),
        dependency_health=dependency_health,
        state=state,
    )


def run_gate() -> dict[str, object]:
    host = ClientExtensionHost(capability_snapshot_id="capability:e5")
    before = host.snapshot
    prepared = host.prepare(
        (_manifest(),),
        capability_snapshot_id="capability:e5",
        available_capabilities=("editor.preview",),
        states={"seed.e5.preview": {"open_count": 1}},
    )
    prepare_is_non_mutating = host.snapshot == before
    first_snapshot = host.commit(prepared)

    host.register_state_migrator(
        "seed.e5.preview",
        1,
        2,
        "seed.e5.preview.state.v2",
        lambda state: {**state, "schema": 2, "open_count": int(state["open_count"]) + 1},
    )
    upgraded = _manifest(
        "2.0.0",
        state_schema_version=2,
        migration_id="seed.e5.preview.state.v2",
    )
    blue_green = host.commit(
        host.prepare(
            (upgraded,),
            capability_snapshot_id="capability:e5-v2",
            available_capabilities=("editor.preview",),
        )
    )
    migration_is_preserved = host.state("seed.e5.preview") == {"open_count": 2, "schema": 2}

    dependency_host = ClientExtensionHost()
    dependency_manifest = _manifest(service_dependencies=(("workbench", "1.0"),))
    _mount(dependency_host, dependency_manifest, dependency_health={"workbench": True})
    affected = dependency_host.report_dependency("workbench", False)
    dependency_quarantine = affected == ("seed.e5.preview",) and not dependency_host.active_manifests
    dependency_host.report_dependency("workbench", True)
    dependency_recovery_requires_remount = False
    try:
        dependency_host.prepare(
            (dependency_manifest,),
            capability_snapshot_id="capability:1",
            available_capabilities=("editor.preview",),
        )
    except ExtensionHostError:
        dependency_recovery_requires_remount = False
    else:
        dependency_recovery_requires_remount = True
    _mount(dependency_host, dependency_manifest)
    dependency_recovery_requires_remount = (
        dependency_recovery_requires_remount and dependency_host.active_manifests == (dependency_manifest,)
    )

    inflight_host = ClientExtensionHost()
    _mount(inflight_host)
    inflight_host.begin_call("seed.e5.preview")
    draining_prepared = inflight_host.prepare(
        (),
        capability_snapshot_id=inflight_host.snapshot.capability_snapshot_id,
        available_capabilities=(),
    )
    draining_blocks_commit = False
    try:
        inflight_host.commit(draining_prepared)
    except ExtensionHostBusyError:
        draining_blocks_commit = inflight_host.active_manifests == (_manifest(),)
    inflight_host.end_call("seed.e5.preview")
    inflight_host.commit(draining_prepared)
    draining_recovers = not inflight_host.active_manifests

    disposer_host = ClientExtensionHost()
    _mount(disposer_host)
    dispose_order: list[str] = []
    root = DisposerNode("root", lambda: dispose_order.append("root"))
    root.add(DisposerNode("child-a", lambda: dispose_order.append("child-a")))
    nested = root.add(DisposerNode("nested"))
    nested.add(DisposerNode("nested-child", lambda: dispose_order.append("nested-child")))
    root.add(DisposerNode("child-b", lambda: dispose_order.append("child-b")))
    disposer_host.attach_disposer("seed.e5.preview", root)
    disposer_host.retire("seed.e5.preview")
    recursive_disposer = dispose_order == ["child-b", "nested-child", "child-a", "root"]
    idempotent_disposer = disposer_host.release("seed.e5.preview") == ()

    failure_host = ClientExtensionHost()
    _mount(failure_host)
    failure_root = DisposerNode(
        "failure-root", lambda: (_ for _ in ()).throw(RuntimeError("cleanup failed"))
    )
    failure_host.attach_disposer("seed.e5.preview", failure_root)
    disposal_failure_audited = False
    try:
        failure_host.retire("seed.e5.preview")
    except ExtensionDisposalError:
        disposal_failure_audited = (
            not failure_host.active_manifests
            and any(item.state == "failed" for item in failure_host.lifecycle_records)
        )

    rollback_host = ClientExtensionHost(capability_snapshot_id="capability:e5")
    _mount(rollback_host, state={"generation": 1})
    parent_snapshot_id = rollback_host.snapshot.snapshot_id
    rollback_host.commit(
        rollback_host.prepare(
            (_manifest("2.0.0"),),
            capability_snapshot_id="capability:e5-v2",
            available_capabilities=("editor.preview",),
            states={"seed.e5.preview": {"generation": 2}},
        )
    )
    rollback_host.rollback(parent_snapshot_id)
    rollback_is_exact = (
        rollback_host.snapshot.snapshot_id == parent_snapshot_id
        and rollback_host.state("seed.e5.preview") == {"generation": 1}
    )
    checkpoint = rollback_host.checkpoint()
    restored = ClientExtensionHost.from_checkpoint(checkpoint)
    checkpoint_roundtrip = restored.checkpoint() == checkpoint
    tampered = deepcopy(checkpoint)
    tampered["states"]["seed.e5.preview"]["generation"] = 99
    checkpoint_tamper_rejected = False
    try:
        ClientExtensionHost.from_checkpoint(tampered)
    except ValueError as exc:
        checkpoint_tamper_rejected = "checkpoint digest mismatch" in str(exc)

    manifest_boundary = False
    try:
        ClientPluginManifest.from_payload({**_manifest().to_payload(), "module": "untrusted"})
    except ValueError as exc:
        manifest_boundary = "executable-source" in str(exc)

    checks = {
        "prepare_is_non_mutating": prepare_is_non_mutating,
        "blue_green_state_migration": blue_green.snapshot_id != first_snapshot.snapshot_id
        and migration_is_preserved,
        "dependency_loss_quarantine": dependency_quarantine,
        "dependency_recovery_requires_explicit_remount": dependency_recovery_requires_remount,
        "inflight_draining": draining_blocks_commit and draining_recovers,
        "recursive_disposer": recursive_disposer and idempotent_disposer,
        "disposer_failure_audited": disposal_failure_audited,
        "rollback_is_exact": rollback_is_exact,
        "checkpoint_roundtrip": checkpoint_roundtrip,
        "checkpoint_tamper_rejected": checkpoint_tamper_rejected,
        "manifest_executable_boundary": manifest_boundary,
        "taiji_cognition_checkpoint_is_not_client_state": all(
            marker not in repr(checkpoint).lower()
            for marker in ("model_state", "weights", "optimizer", "provider")
        ),
    }
    return {
        "gate": "taiji-e5-client-extension-host",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "metrics": {
            "lifecycle_records": len(host.lifecycle_records),
            "active_snapshot_revision": host.snapshot.revision,
            "blue_green_snapshot_revision": blue_green.revision,
            "dependency_quarantined_plugin_count": len(affected),
            "recursive_disposer_order": dispose_order,
            "rollback_snapshot_revision": rollback_host.snapshot.revision,
        },
        "scope": {
            "client_snapshot_only": True,
            "protected_root_shell_slots": True,
            "plugin_source_execution": False,
            "taiji_cognition_mutation": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_e5_client_extension_host_20260901.json",
    )
    args = parser.parse_args()
    result = run_gate()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
