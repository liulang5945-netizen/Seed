"""Unit tests for bounded N36 delivery runtime & user-takeover CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from taiji.bounded_delivery_runtime import (
    SCHEMA,
    SCOPE,
    BoundedDeliveryRuntime,
    BundleIntegrityError,
    LifecycleError,
    build_delivery_bundle,
)
from taiji.long_run_harness import CrashInjected, SealViolation
from taiji.long_run_trained_subject import RunConfig

FAST_CFG = RunConfig(
    n_sessions=12,
    events_per_session=15,
    crash_sessions=(3, 8),
    p1_end_session=4,
    p2_end_session=10,
    dormancy_session=11,
    ch_train_end_session=6,
    ch_updates_per_session=2,
    phy_steps_early=5,
    phy_steps_mid=5,
    grow_p1_steps=5,
    grow_p2_steps=5,
    grow_maint_steps=2,
)


def test_build_open_interact_and_oov_refusal(tmp_path: Path) -> None:
    bdir = tmp_path / "bundle"
    manifest = build_delivery_bundle(bdir, seed=6801, cfg=FAST_CFG)
    assert manifest["schema"] == SCHEMA
    assert manifest["scope"] == SCOPE
    assert manifest["boundaries"]["check_reads"] == 0
    assert manifest["boundaries"]["final_reads"] == 0

    rt = BoundedDeliveryRuntime.open(bdir)
    st = rt.status()
    assert st["chain_valid"] is True
    assert st["sessions"] == 12
    assert st["lifecycle_state"] == "Running"

    res = rt.interact_chinese(["请把红色杯子放到桌上。", "再去检查一下它。"], persist=True)
    assert len(res["turns"]) == 2
    assert res["sessions"] == 13
    assert rt.status()["sessions"] == 13

    with pytest.raises(ValueError, match="out_of_bounded_vocab"):
        rt.interact_chinese(["紫砂壶放在凉亭石凳。"], persist=False)


def test_fail_closed_integrity_checks(tmp_path: Path) -> None:
    with pytest.raises(BundleIntegrityError):
        BoundedDeliveryRuntime.open(tmp_path / "missing_dir")

    bdir = tmp_path / "tamper"
    build_delivery_bundle(bdir, seed=6801, cfg=FAST_CFG)

    # Tamper audit chain
    audit = bdir / "taiji-n36-bounded.audit.jsonl"
    orig_audit = audit.read_text(encoding="utf-8")
    lines = orig_audit.splitlines()
    lines[0] = lines[0].replace('"record_digest":"', '"record_digest":"f')[:-1] + '"'
    audit.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(BundleIntegrityError):
        BoundedDeliveryRuntime.open(bdir)
    audit.write_text(orig_audit, encoding="utf-8")

    # Tamper session file
    sess = bdir / "taiji-n36-bounded.session"
    sess.write_bytes(b"corrupted-bytes")
    with pytest.raises(BundleIntegrityError):
        BoundedDeliveryRuntime.open(bdir)


def test_crash_recovery_lifecycle_rollback_and_sealed_guard(tmp_path: Path) -> None:
    bdir = tmp_path / "lifecycle"
    build_delivery_bundle(bdir, seed=6801, cfg=FAST_CFG)
    rt = BoundedDeliveryRuntime.open(bdir)

    with pytest.raises(SealViolation):
        rt.read_split("check")
    with pytest.raises(SealViolation):
        rt.read_split("final")

    with pytest.raises(CrashInjected):
        rt.update_credibility("forum", 1, crash_before_commit=True)
    rt_rec = BoundedDeliveryRuntime.open(bdir)
    assert rt_rec.status()["chain_valid"] is True
    assert rt_rec.status()["sessions"] == 12

    snap_digest = rt_rec.status()["record_digest"]
    rt_rec.snapshot("before_mut")
    rt_rec.add_pending("p_n36", "check bounded pending")
    rt_rec.update_credibility("forum", 0)
    assert rt_rec.status()["record_digest"] != snap_digest

    rt_rec.pause()
    with pytest.raises(LifecycleError):
        rt_rec.update_credibility("forum", 1)
    rt_rec.resume()

    rb = rt_rec.rollback("before_mut")
    assert rb["record_digest"] == snap_digest
    assert rb["sessions"] == 12

    rt_rec.archive()
    with pytest.raises(LifecycleError):
        rt_rec.resume()
    with pytest.raises(LifecycleError):
        rt_rec.interact_chinese(["请把红色杯子放到桌上。"], persist=True)


def test_cli_status_and_chat_subprocess(tmp_path: Path) -> None:
    bdir = tmp_path / "cli_bundle"
    build_delivery_bundle(bdir, seed=6801, cfg=FAST_CFG)
    cmd = [
        sys.executable,
        "scripts/taiji_n36_takeover.py",
        "status",
        "--bundle-dir",
        str(bdir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(proc.stdout)
    assert data["schema"] == SCHEMA
    assert data["chain_valid"] is True
