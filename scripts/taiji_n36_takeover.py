"""User-operable CLI for the bounded N36 delivery & takeover package.

Usage:
  python scripts/taiji_n36_takeover.py build --bundle-dir <dir> [--seed 6801]
  python scripts/taiji_n36_takeover.py status --bundle-dir <dir>
  python scripts/taiji_n36_takeover.py chat --bundle-dir <dir> --utterance "请把红色杯子放到桌上。"
  python scripts/taiji_n36_takeover.py predict-physics --bundle-dir <dir> [--family rough_collision]
  python scripts/taiji_n36_takeover.py credibility --bundle-dir <dir> --source forum --correct 1
  python scripts/taiji_n36_takeover.py pause|resume|archive --bundle-dir <dir>
  python scripts/taiji_n36_takeover.py snapshot|rollback --bundle-dir <dir> --name <snap>
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

from taiji.bounded_delivery_runtime import (
    BoundedDeliveryRuntime,
    build_delivery_bundle,
)  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build")
    p_build.add_argument("--bundle-dir", required=True)
    p_build.add_argument("--seed", type=int, default=6801)

    p_status = sub.add_parser("status")
    p_status.add_argument("--bundle-dir", required=True)

    p_chat = sub.add_parser("chat")
    p_chat.add_argument("--bundle-dir", required=True)
    p_chat.add_argument("--utterance", action="append", required=True)
    p_chat.add_argument("--no-persist", action="store_true")

    p_phy = sub.add_parser("predict-physics")
    p_phy.add_argument("--bundle-dir", required=True)
    p_phy.add_argument("--family", default="rough_collision")
    p_phy.add_argument("--seed", type=int, default=101)
    p_phy.add_argument("--ablate-barrier", action="store_true")

    p_cred = sub.add_parser("credibility")
    p_cred.add_argument("--bundle-dir", required=True)
    p_cred.add_argument("--source", required=True)
    p_cred.add_argument("--correct", type=int, choices=(0, 1), required=True)

    for lc in ("pause", "resume", "archive"):
        p_lc = sub.add_parser(lc)
        p_lc.add_argument("--bundle-dir", required=True)

    for sn in ("snapshot", "rollback"):
        p_sn = sub.add_parser(sn)
        p_sn.add_argument("--bundle-dir", required=True)
        p_sn.add_argument("--name", required=True)

    args = parser.parse_args()
    out: dict[str, Any]
    if args.cmd == "build":
        out = build_delivery_bundle(args.bundle_dir, seed=args.seed)
    else:
        rt = BoundedDeliveryRuntime.open(args.bundle_dir)
        if args.cmd == "status":
            out = rt.status()
        elif args.cmd == "chat":
            out = rt.interact_chinese(args.utterance, persist=not args.no_persist)
        elif args.cmd == "predict-physics":
            out = rt.predict_physics(
                args.family, seed=args.seed, ablate_barrier=args.ablate_barrier
            )
        elif args.cmd == "credibility":
            out = rt.update_credibility(args.source, args.correct)
        elif args.cmd == "pause":
            out = rt.pause()
        elif args.cmd == "resume":
            out = rt.resume()
        elif args.cmd == "archive":
            out = rt.archive()
        elif args.cmd == "snapshot":
            out = rt.snapshot(args.name)
        else:
            out = rt.rollback(args.name)

    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
