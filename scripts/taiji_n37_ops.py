"""User-operable CLI for the bounded N37 Operational State Machine (§15).

Usage:
  python scripts/taiji_n37_ops.py init-ops --bundle-dir <dir> [--initial-state Qualified]
  python scripts/taiji_n37_ops.py status --bundle-dir <dir>
  python scripts/taiji_n37_ops.py transition --bundle-dir <dir> --to Trial [--reason ...]
  python scripts/taiji_n37_ops.py classify-change --kind ecl_calibration
  python scripts/taiji_n37_ops.py shadow-upgrade --bundle-dir <dir> --kind ecl_calibration
  python scripts/taiji_n37_ops.py wake --bundle-dir <dir> --note timer
  python scripts/taiji_n37_ops.py restrict --bundle-dir <dir> --reason low_disk
  python scripts/taiji_n37_ops.py recover --bundle-dir <dir> [--health-ok]
  python scripts/taiji_n37_ops.py retire --bundle-dir <dir> --archive-dir <arch_dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taiji.operational_state_machine import (  # noqa: E402
    OperationalStateMachine,
    classify_change,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init-ops")
    p_init.add_argument("--bundle-dir", required=True)
    p_init.add_argument("--initial-state", default="Qualified")
    p_init.add_argument("--shadow-budget", type=int, default=4)

    p_st = sub.add_parser("status")
    p_st.add_argument("--bundle-dir", required=True)

    p_tr = sub.add_parser("transition")
    p_tr.add_argument("--bundle-dir", required=True)
    p_tr.add_argument("--to", required=True)
    p_tr.add_argument("--reason", default="")

    p_cc = sub.add_parser("classify-change")
    p_cc.add_argument("--kind", required=True)

    p_sh = sub.add_parser("shadow-upgrade")
    p_sh.add_argument("--bundle-dir", required=True)
    p_sh.add_argument("--kind", default="ecl_calibration")

    p_wk = sub.add_parser("wake")
    p_wk.add_argument("--bundle-dir", required=True)
    p_wk.add_argument("--note", default="user_event")

    p_rs = sub.add_parser("restrict")
    p_rs.add_argument("--bundle-dir", required=True)
    p_rs.add_argument("--reason", required=True)

    p_rc = sub.add_parser("recover")
    p_rc.add_argument("--bundle-dir", required=True)
    p_rc.add_argument("--health-ok", action="store_true")

    p_rt = sub.add_parser("retire")
    p_rt.add_argument("--bundle-dir", required=True)
    p_rt.add_argument("--archive-dir", required=True)
    p_rt.add_argument("--reason", default="explicit_retirement")

    args = parser.parse_args()
    out: dict[str, Any]
    if args.cmd == "classify-change":
        out = classify_change(args.kind)
    elif args.cmd == "init-ops":
        fsm = OperationalStateMachine.initialize(
            args.bundle_dir,
            initial_state=args.initial_state,
            shadow_budget=args.shadow_budget,
        )
        out = fsm.status()
    else:
        fsm = OperationalStateMachine.open(args.bundle_dir)
        if args.cmd == "status":
            out = fsm.status()
        elif args.cmd == "transition":
            out = fsm.transition(args.to, reason=args.reason)
        elif args.cmd == "shadow-upgrade":
            out = fsm.run_shadow_upgrade(args.kind)
        elif args.cmd == "wake":
            out = fsm.wake_event(args.note)
        elif args.cmd == "restrict":
            out = fsm.enter_restricted(args.reason)
        elif args.cmd == "recover":
            out = fsm.recover_from_restricted(health_ok=bool(args.health_ok))
        else:
            out = fsm.retire_and_archive(args.archive_dir, reason=args.reason)

    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
