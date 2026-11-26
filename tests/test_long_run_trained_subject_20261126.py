"""Unit tests for single-subject long-run trained durability (`taiji-long-run-trained-v1`)."""

from __future__ import annotations

import sys
import tempfile
import time

sys.path.insert(0, r"E:\Seed")

from taiji.long_run_trained_subject import RunConfig, run_long_run_subject


def test_capsule_recovery_sealed_blocks_and_determinism() -> None:
    cfg = RunConfig(
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
    dl = time.process_time() + 60.0
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        r1 = run_long_run_subject(d1, 91, dl, cfg=cfg)
        r2 = run_long_run_subject(d2, 91, dl, cfg=cfg)

    dur1 = r1["durability"]
    dur2 = r2["durability"]
    assert dur1["sessions"] == 12
    assert dur1["events"] >= 12 * 15
    assert dur1["crash_recoveries"] == 2
    assert dur1["sealed_blocks"] == 2
    assert dur1["chain_valid"] is True
    assert dur1["chain_head"] == dur2["chain_head"]
    assert dur1["record_digest"] == dur2["record_digest"]
