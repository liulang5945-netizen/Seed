"""Tests for bounded N37 Operational State Machine (§15)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from taiji.bounded_delivery_runtime import (
    BoundedDeliveryRuntime,
    LifecycleError,
    build_delivery_bundle,
)
from taiji.long_run_trained_subject import RunConfig
from taiji.operational_state_machine import (
    OperationalStateMachine,
    OperationalTransitionError,
    OutOfContractChangeError,
    classify_change,
)

SMOKE_CFG = RunConfig(
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


def _build_smoke(tmp_path: Path) -> Path:
    bdir = tmp_path / "n37_bundle"
    build_delivery_bundle(bdir, seed=6801, cfg=SMOKE_CFG, cap_cpu=60.0)
    return bdir


def test_o1_o2_fsm_transitions_and_three_tier_routing(tmp_path: Path) -> None:
    bdir = _build_smoke(tmp_path)
    fsm = OperationalStateMachine.initialize(bdir, initial_state="Qualified")
    assert fsm.state == "Qualified"

    with pytest.raises(OperationalTransitionError):
        fsm.transition("Running")

    fsm.transition("Trial", reason="start_trial")
    fsm.transition("Paused", reason="pause_before_pass")
    with pytest.raises(OperationalTransitionError):
        fsm.transition("Running", reason="must_not_bypass_trial")

    fsm.transition("Trial", reason="resume_trial")
    fsm.transition("Running", reason="trial_passed")
    assert fsm.status()["trial_passed"] is True

    assert classify_change("ops_refactor")["tier"] == "tier1_ops_only"
    assert classify_change("ecl_calibration")["tier"] == "tier2_in_contract_shadow"
    t3 = classify_change("readout_class")
    assert t3["tier"] == "tier3_contract_or_architecture"
    assert t3["reopen_research_required"] is True

    with pytest.raises(OutOfContractChangeError):
        fsm.run_shadow_upgrade("readout_class")


def test_o3_o4_shadow_accept_reject_waiting_and_wake(tmp_path: Path) -> None:
    bdir = _build_smoke(tmp_path)
    fsm = OperationalStateMachine.initialize(bdir, initial_state="Running", shadow_budget=4)

    acc = fsm.run_shadow_upgrade("ecl_calibration", min_gain=0.0005, max_retention_drop=0.02)
    assert acc["accepted"] is True
    assert acc["gain"] >= 0.0005
    assert acc["retention_drop"] <= 0.02
    assert fsm.state == "Running"

    parent_digest = fsm.rt.manifest["record_digest"]
    parent_chain = fsm.rt.manifest["chain_head"]

    rej1 = fsm.run_shadow_upgrade(
        "harmful_phy_corrupt",
        max_retention_drop=0.02,
        external_actions_taken=["sent_status_report"],
    )
    assert rej1["accepted"] is False
    assert "retention_breach" in rej1["reject_reason"]
    assert fsm.rt.manifest["record_digest"] == parent_digest
    assert fsm.rt.manifest["chain_head"] == parent_chain
    assert fsm.state == "Running"

    rej2 = fsm.run_shadow_upgrade("harmful_phy_corrupt", max_retention_drop=0.02)
    assert rej2["accepted"] is False
    assert rej2["auto_waited"] is True
    assert fsm.state == "Waiting"

    with pytest.raises(LifecycleError):
        fsm.rt.update_credibility("forum", 1)

    fsm.wake_event("new_user_input")
    assert fsm.state == "Running"


def test_o5_o6_restricted_recovery_retire_and_cli(tmp_path: Path) -> None:
    bdir = _build_smoke(tmp_path)
    fsm = OperationalStateMachine.initialize(bdir, initial_state="Running")
    fsm.rt.add_pending("p_retire_1", "open task before retire")

    fsm.enter_restricted("low_disk_space")
    assert fsm.state == "Restricted"
    with pytest.raises(LifecycleError):
        fsm.rt.update_credibility("forum", 1)

    fsm.recover_from_restricted(health_ok=False)
    assert fsm.state == "Paused"
    fsm.transition("Running", reason="resume_after_pause")
    fsm.enter_restricted("transient_cpu_spike")
    fsm.recover_from_restricted(health_ok=True)
    assert fsm.state == "Running"

    arch_dir = tmp_path / "n37_archive"
    arch = fsm.retire_and_archive(arch_dir, reason="end_of_service")
    assert arch["state"] == "Archived"
    assert arch["verified_readable"] is True
    assert arch["pending_count_exported"] == 1

    with pytest.raises(OperationalTransitionError):
        fsm.transition("Running")
    with pytest.raises(LifecycleError):
        fsm.rt.resume()

    opened_arch = BoundedDeliveryRuntime.open(arch_dir)
    assert opened_arch.manifest["record_digest"] == arch["record_digest"]

    proc = subprocess.run(
        [sys.executable, "scripts/taiji_n37_ops.py", "classify-change", "--kind", "cli"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(proc.stdout)["tier"] == "tier1_ops_only"


def test_o3_growth_proposal_executes_real_phase2_adaptation(tmp_path: Path) -> None:
    bdir = _build_smoke(tmp_path)
    fsm = OperationalStateMachine.initialize(bdir, initial_state="Running", shadow_budget=2)
    before_units = fsm.rt.grow_net.active_units()
    res = fsm.run_shadow_upgrade("growth_proposal", min_gain=0.01, max_retention_drop=0.05)
    assert res["accepted"] is True
    assert res["gain"] >= 0.01
    assert res["retention_drop"] <= 0.05
    assert fsm.rt.grow_net.active_units() > before_units
    assert fsm.rt.harness.verify_chain() is True
