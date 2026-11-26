"""N37.1.01 (ops FSM drills): 8-state FSM, 6-step shadow upgrade, 3-tier routing & retirement.

Pre-registered: plans/reference/DESIGN-N37-1-01_20261126.md (D-20261126-686).
Builds a full 100-session bounded bundle on delivery seed 6801 (zero confirmation
seeds, zero check/final reads) and executes all 6 operational drills (O1..O6).

Usage:
  python scripts/training/n37_1_01_ops_fsm_drills.py [--smoke]
Output:
  reports/n37_1_01_ops_fsm_drills.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from taiji.bounded_delivery_runtime import (  # noqa: E402
    SCOPE,
    BoundedDeliveryRuntime,
    LifecycleError,
    build_delivery_bundle,
)
from taiji.long_run_trained_subject import RunConfig  # noqa: E402
from taiji.operational_state_machine import (  # noqa: E402
    OperationalStateMachine,
    OperationalTransitionError,
    OutOfContractChangeError,
    classify_change,
)

CARD = "N37.1.01"
DESIGN = "DESIGN-N37-1-01_20261126.md"
DELIVERY_SEED = 6801
CAP_CPU = 100.0
OUT = ROOT / "reports" / "n37_1_01_ops_fsm_drills.json"


def _expect_exc(fn, exc_type: type[BaseException]) -> bool:
    try:
        fn()
    except exc_type:
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(1)
    started = time.process_time()
    cfg = (
        RunConfig(
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
        if args.smoke
        else RunConfig()
    )

    with tempfile.TemporaryDirectory() as tmp:
        bdir = Path(tmp) / "n37_bundle"
        manifest = build_delivery_bundle(bdir, seed=DELIVERY_SEED, cfg=cfg, cap_cpu=CAP_CPU)
        fsm = OperationalStateMachine.initialize(bdir, initial_state="Qualified", shadow_budget=4)

        # O1: FSM transition graph
        o1_bad = _expect_exc(lambda: fsm.transition("Running"), OperationalTransitionError)
        fsm.transition("Trial", reason="o1_trial")
        fsm.transition("Paused", reason="o1_pause_before_pass")
        o1_bypass = _expect_exc(lambda: fsm.transition("Running"), OperationalTransitionError)
        fsm.transition("Trial", reason="o1_resume_trial")
        fsm.transition("Running", reason="o1_trial_passed")
        o1 = bool(o1_bad and o1_bypass and fsm.state == "Running")

        # O2: 3-tier change routing
        t1 = classify_change("ops_refactor")
        t2 = classify_change("ecl_calibration")
        t3 = classify_change("readout_class")
        o2_refuse = _expect_exc(
            lambda: fsm.run_shadow_upgrade("readout_class"), OutOfContractChangeError
        )
        o2 = bool(
            t1["tier"] == "tier1_ops_only"
            and t2["tier"] == "tier2_in_contract_shadow"
            and t3["reopen_research_required"] is True
            and o2_refuse
        )

        # O3: 6-step shadow upgrade accept & recalibrate
        acc = fsm.run_shadow_upgrade("ecl_calibration", min_gain=0.0005, max_retention_drop=0.02)
        o3 = bool(
            acc["accepted"] is True
            and acc["gain"] >= 0.0005
            and acc["retention_drop"] <= 0.02
            and fsm.state == "Running"
            and fsm.rt.harness.verify_chain() is True
        )

        # O4: harmful candidate reject, bit-identical parent, 2-reject auto-Waiting & wake
        p_dig = fsm.rt.manifest["record_digest"]
        p_chn = fsm.rt.manifest["chain_head"]
        rej1 = fsm.run_shadow_upgrade(
            "harmful_phy_corrupt",
            max_retention_drop=0.02,
            external_actions_taken=["emitted_telemetry"],
        )
        rej2 = fsm.run_shadow_upgrade("harmful_phy_corrupt", max_retention_drop=0.02)
        o4_wait_block = _expect_exc(lambda: fsm.rt.update_credibility("forum", 1), LifecycleError)
        fsm.wake_event("o4_wake")
        o4 = bool(
            rej1["accepted"] is False
            and rej2["accepted"] is False
            and rej2["auto_waited"] is True
            and fsm.rt.manifest["record_digest"] == p_dig
            and fsm.rt.manifest["chain_head"] == p_chn
            and o4_wait_block
            and fsm.state == "Running"
        )

        # O5: Restricted guard & recovery
        fsm.enter_restricted("o5_disk_warning")
        o5_block = _expect_exc(lambda: fsm.rt.update_credibility("forum", 1), LifecycleError)
        fsm.recover_from_restricted(health_ok=False)
        o5_paused = fsm.state == "Paused"
        fsm.transition("Running", reason="o5_resume")
        fsm.enter_restricted("o5_transient")
        fsm.recover_from_restricted(health_ok=True)
        o5 = bool(o5_block and o5_paused and fsm.state == "Running")

        # O6: Retirement archival verification
        fsm.rt.add_pending("o6_pending", "settle at retirement")
        arch_dir = Path(tmp) / "n37_archive"
        arch = fsm.retire_and_archive(arch_dir, reason="o6_retirement")
        opened_arch = BoundedDeliveryRuntime.open(arch_dir)
        o6_terminal = _expect_exc(lambda: fsm.transition("Running"), OperationalTransitionError)
        o6 = bool(
            arch["state"] == "Archived"
            and arch["verified_readable"] is True
            and opened_arch.manifest["record_digest"] == arch["record_digest"]
            and o6_terminal
        )

        final_status = fsm.status()

    checks = {
        "o1_fsm_transition_graph": o1,
        "o2_three_tier_change_routing": o2,
        "o3_shadow_upgrade_accept_and_recalibrate": o3,
        "o4_shadow_reject_rollback_and_waiting": o4,
        "o5_restricted_guard_and_recovery": o5,
        "o6_retirement_archival_verification": o6,
    }
    cpu = round(time.process_time() - started, 2)
    exit_ = (
        "smoke"
        if args.smoke
        else ("ops_state_machine_ready" if all(checks.values()) else "drill_failed")
    )
    payload = {
        "schema": "taiji-n37-1-01-v1",
        "card": CARD,
        "design": DESIGN,
        "scope": SCOPE,
        "created_utc": datetime.now(UTC).isoformat(),
        "smoke": args.smoke,
        "delivery_seed": DELIVERY_SEED,
        "bundle_summary": {
            "initial_sessions": manifest["sessions"],
            "parameter_count": manifest["parameter_count"],
            "shadow_accept_gain": acc["gain"],
            "shadow_accept_retention_drop": acc["retention_drop"],
            "shadow_reject_retention_drop": rej1["retention_drop"],
            "transitions_count": final_status["transitions_count"],
            "consequence_ledger_count": final_status["consequence_ledger_count"],
        },
        "checks": checks,
        "exit": exit_,
        "cost": {"cpu": cpu, "cap_cpu": CAP_CPU},
        "discipline": {
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0,
            "oracle_used_by_subject": False,
        },
    }
    if not args.smoke:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"exit": exit_, "checks": checks, "cpu": cpu}, ensure_ascii=False))
    return 0 if exit_ in ("ops_state_machine_ready", "smoke") else 3


if __name__ == "__main__":
    raise SystemExit(main())
