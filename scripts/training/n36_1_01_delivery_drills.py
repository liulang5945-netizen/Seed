"""N36.1.01 (delivery drills): 5-class fault, recovery, cross-process & authenticity drills.

Pre-registered: plans/reference/DESIGN-N36-1-01_20261126.md (D-20261126-683).
Builds a full 100-session bounded delivery bundle on delivery seed 6801 (zero confirmation
seeds, zero check/final reads) and executes all 5 takeover & fault drills (D1..D5).

Usage:
  python scripts/training/n36_1_01_delivery_drills.py [--smoke]
Output:
  reports/n36_1_01_delivery_drills.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
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
    BundleIntegrityError,
    LifecycleError,
    build_delivery_bundle,
)
from taiji.long_run_harness import CrashInjected, SealViolation  # noqa: E402
from taiji.long_run_trained_subject import RunConfig  # noqa: E402

CARD = "N36.1.01"
DESIGN = "DESIGN-N36-1-01_20261126.md"
DELIVERY_SEED = 6801
CAP_CPU = 100.0
OUT = ROOT / "reports" / "n36_1_01_delivery_drills.json"


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
        bdir = Path(tmp) / "n36_bundle"
        manifest = build_delivery_bundle(bdir, seed=DELIVERY_SEED, cfg=cfg, cap_cpu=CAP_CPU)
        rt = BoundedDeliveryRuntime.open(bdir)

        # D1: fail-closed integrity
        d1_missing = _expect_exc(
            lambda: BoundedDeliveryRuntime.open(Path(tmp) / "no_such_bundle"),
            BundleIntegrityError,
        )
        tamper_dir = Path(tmp) / "tamper_bundle"
        shutil.copytree(bdir, tamper_dir)
        (tamper_dir / "taiji-n36-bounded.session").write_bytes(b"bad")
        d1_session_tamper = _expect_exc(
            lambda: BoundedDeliveryRuntime.open(tamper_dir),
            BundleIntegrityError,
        )
        tamper_audit = Path(tmp) / "tamper_audit"
        shutil.copytree(bdir, tamper_audit)
        ap = tamper_audit / "taiji-n36-bounded.audit.jsonl"
        ap.write_text(
            ap.read_text(encoding="utf-8").replace('"chain":"', '"chain":"0'), encoding="utf-8"
        )
        d1_audit_tamper = _expect_exc(
            lambda: BoundedDeliveryRuntime.open(tamper_audit),
            BundleIntegrityError,
        )
        d1 = bool(d1_missing and d1_session_tamper and d1_audit_tamper)

        # D2: cross-process fresh load & inference equivalence
        in_proc_chat = rt.interact_chinese(
            ["请把红色杯子放到桌上。", "再去检查一下它。"], persist=False
        )
        in_proc_phy = rt.predict_physics("rough_collision", seed=101)
        sub_code = (
            "import json, sys; sys.path.insert(0, r'E:\\Seed'); "
            "from taiji.bounded_delivery_runtime import BoundedDeliveryRuntime; "
            f"rt = BoundedDeliveryRuntime.open(r'{bdir}'); "
            "print(json.dumps({'st': rt.status(), "
            "'chat': rt.interact_chinese(['请把红色杯子放到桌上。', '再去检查一下它。'], persist=False), "
            "'phy': rt.predict_physics('rough_collision', seed=101)}, ensure_ascii=False))"
        )
        proc = subprocess.run(
            [sys.executable, "-c", sub_code],
            capture_output=True,
            text=True,
            check=True,
        )
        sub_out = json.loads(proc.stdout)
        d2 = bool(
            sub_out["st"]["record_digest"] == manifest["record_digest"]
            and sub_out["st"]["chain_head"] == manifest["chain_head"]
            and sub_out["chat"] == in_proc_chat
            and sub_out["phy"] == in_proc_phy
        )

        # D3: pre-commit crash recovery
        d3_crash = _expect_exc(
            lambda: rt.update_credibility("forum", 1, crash_before_commit=True),
            CrashInjected,
        )
        rt_after_crash = BoundedDeliveryRuntime.open(bdir)
        d3 = bool(
            d3_crash
            and rt_after_crash.status()["chain_valid"] is True
            and rt_after_crash.status()["sessions"] == manifest["sessions"]
        )

        # D4: lifecycle pause/resume/snapshot/rollback/archive
        snap_digest = rt_after_crash.status()["record_digest"]
        rt_after_crash.snapshot("drill_snap")
        d4_pause = False
        rt_after_crash.pause()
        d4_pause = _expect_exc(
            lambda: rt_after_crash.update_credibility("forum", 0),
            LifecycleError,
        )
        rt_after_crash.resume()
        rt_after_crash.add_pending("drill_p1", "bounded drill pending")
        rt_after_crash.update_credibility("forum", 0)
        rb = rt_after_crash.rollback("drill_snap")
        rt_after_crash.archive()
        d4_arch = _expect_exc(
            lambda: rt_after_crash.resume(),
            LifecycleError,
        )
        d4 = bool(d4_pause and rb["record_digest"] == snap_digest and d4_arch)

        # D5: sealed guard + isolated weight-zeroing authenticity ablation
        d5_check = _expect_exc(lambda: rt.read_split("check"), SealViolation)
        d5_final = _expect_exc(lambda: rt.read_split("final"), SealViolation)
        phy_full = in_proc_phy["metrics"]["collision_recall"]
        phy_abl = rt.predict_physics("rough_collision", seed=101, ablate_barrier=True)[
            "barrier_ablation"
        ]["collision_recall_drop"]
        d5 = bool(d5_check and d5_final and phy_full >= 0.90 and phy_abl >= 0.40)

    checks = {
        "d1_fail_closed_integrity": d1,
        "d2_cross_process_fresh_load": d2,
        "d3_crash_recovery": d3,
        "d4_lifecycle_and_rollback": d4,
        "d5_sealed_guard_and_authenticity": d5,
    }
    cpu = round(time.process_time() - started, 2)
    exit_ = (
        "smoke"
        if args.smoke
        else ("delivery_package_ready" if all(checks.values()) else "drill_failed")
    )
    payload = {
        "schema": "taiji-n36-1-01-v1",
        "card": CARD,
        "design": DESIGN,
        "scope": SCOPE,
        "created_utc": datetime.now(UTC).isoformat(),
        "smoke": args.smoke,
        "delivery_seed": DELIVERY_SEED,
        "bundle_summary": {
            "sessions": manifest["sessions"],
            "parameter_count": manifest["parameter_count"],
            "record_digest": manifest["record_digest"],
            "chain_head": manifest["chain_head"],
            "phy_collision_recall": in_proc_phy["metrics"]["collision_recall"],
            "phy_barrier_ablation_drop": phy_abl,
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
    return 0 if exit_ in ("delivery_package_ready", "smoke") else 3


if __name__ == "__main__":
    raise SystemExit(main())
