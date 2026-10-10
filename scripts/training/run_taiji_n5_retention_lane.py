"""N5 保持侧配对的**跑道器**（PLAN-N5-04 §3 G-N5d-1 的机械执行面）。

**为什么要有它**：双臂跑完之后，"出四张 after 面＋逐臂判读"这件事一直是**我手拼命令**——
而本仓为手拼命令付过学费：㊵-593 那份"冻结命令"把带参旗标写成裸旗标，照抄即被 argparse 拒，
台账里那条过期处方比过期结论更贵。跑道器把这六条评测／判读命令（外加一次预检）钉成一份可复算的计划，并且**先过预检再花钱**：
预检说"题集内容不可声称同源"或"基座摘要对不上"，就不该去花那 13 分钟评测。

**硬前置（fail-closed，缺任何一条整道拒绝跑）**：
① 两枚 after 检查点都必须存在；② `check_taiji_n5_pairing_preflight` 对**将要配对的那两张 before 面**
必须给出 `cap0_identity=verified`、`cap0_checkpoint_identity=verified`、`replay_identity=content_hash_available`
——第三项就是 DEBT-G79 的来由：旧面只有偏移与首件，配对能跑但"题集逐字相同"这句话不能说。

`--dry-run` 只打印将要执行的计划（含预检本身），不跑任何东西；`--only` 用于单独重跑某一侧的面。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CAP0_EVALUATOR = "scripts/training/eval_taiji_cap0_baseline.py"
REPLAY_EVALUATOR = "scripts/training/measure_taiji_a30_repetition_penalty.py"
ADJUDICATOR = "scripts/training/adjudicate_taiji_n2_04_retention_pair.py"
PREFLIGHT = "scripts/training/check_taiji_n5_pairing_preflight.py"

CAP0_BEFORE = Path("reports/taiji_n2_cap0_before_20261008.json")
#: 默认指向**带内容哈希**的那张新 before 面（㊵-645 出路①）；旧面仍可指名，但预检会拒绝配对。
REPLAY_BEFORE = Path("reports/taiji_n2_replay24_before_20261010.json")
RETENTION_MANIFEST = "plans/manifests/cap0_eval_set_v2.json"

ARMS = ("treated", "control")
REQUIRED_PREFLIGHT = {
    ("cap0", "cap0_identity"): "verified",
    ("base_checkpoint", "cap0_checkpoint_identity"): "verified",
    ("replay", "replay_identity"): "content_hash_available",
}


def preflight_refusals(payload: dict[str, Any]) -> list[str]:
    """预检读数 ⇒ 拒绝清单（空表才允许花钱跑）。"""

    refusals: list[str] = []
    for (block, key), expected in REQUIRED_PREFLIGHT.items():
        observed = payload.get(block, {}).get(key)
        if observed != expected:
            refusals.append(f"{block}.{key}={observed!r}（需要 {expected!r}）")
    return refusals


def build_plan(
    *,
    treated: Path,
    control: Path,
    out_dir: Path,
    cap0_before: Path = CAP0_BEFORE,
    replay_before: Path = REPLAY_BEFORE,
    replay_limit: int,
    only: str | None,
) -> list[list[str]]:
    """返回将要执行的命令（ argv 列表，第一项是可执行程序名）。"""

    python = [sys.executable]
    plan: list[list[str]] = []
    for arm in ARMS:
        checkpoint = treated if arm == "treated" else control
        if only in (None, "cap0"):
            plan.append(
                python
                + [
                    str(PROJECT_ROOT / CAP0_EVALUATOR),
                    "--checkpoint",
                    str(checkpoint),
                    "--report",
                    str(out_dir / f"{arm}_cap0_after.json"),
                ]
            )
        if only in (None, "replay"):
            plan.append(
                python
                + [
                    str(PROJECT_ROOT / REPLAY_EVALUATOR),
                    "--checkpoint",
                    str(checkpoint),
                    "--no-circuit",
                    "--limit",
                    str(replay_limit),
                    "--out-report",
                    str(out_dir / f"{arm}_replay_after.json"),
                ]
            )
        if only in (None, "adjudicate"):
            plan.append(
                python
                + [
                    str(PROJECT_ROOT / ADJUDICATOR),
                    "--cap0-before",
                    str(cap0_before),
                    "--cap0-after",
                    str(out_dir / f"{arm}_cap0_after.json"),
                    "--replay-before",
                    str(replay_before),
                    "--replay-after",
                    str(out_dir / f"{arm}_replay_after.json"),
                    "--retention-manifest",
                    RETENTION_MANIFEST,
                    "--out",
                    str(out_dir / f"{arm}_retention_pair.json"),
                ]
            )
    return plan


def _run(argv: list[str]) -> int:
    print("$ " + " ".join(argv), flush=True)
    return subprocess.run(argv, check=False).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="N5 retention-lane runner (static pre-flight first)"
    )
    parser.add_argument("--treated-checkpoint", type=Path, required=True)
    parser.add_argument("--control-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cap0-before", type=Path, default=CAP0_BEFORE)
    parser.add_argument("--replay-before", type=Path, default=REPLAY_BEFORE)
    parser.add_argument("--replay-limit", type=int, default=24)
    parser.add_argument("--only", choices=("cap0", "replay", "adjudicate"), default=None)
    parser.add_argument("--dry-run", action="store_true", help="print the plan and run nothing")
    args = parser.parse_args(argv)

    for label, path in (
        ("treated", args.treated_checkpoint),
        ("control", args.control_checkpoint),
    ):
        if not path.is_file():
            print(f"REFUSE: {label} checkpoint 不在场：{path}")
            return 2
    for label, path in (("cap0-before", args.cap0_before), ("replay-before", args.replay_before)):
        if not (path if path.is_absolute() else PROJECT_ROOT / path).is_file():
            print(f"REFUSE: {label} 不在场：{path}")
            return 2

    plan = build_plan(
        treated=args.treated_checkpoint,
        control=args.control_checkpoint,
        out_dir=args.out_dir,
        cap0_before=args.cap0_before,
        replay_before=args.replay_before,
        replay_limit=args.replay_limit,
        only=args.only,
    )
    preflight_argv = [
        sys.executable,
        str(PROJECT_ROOT / PREFLIGHT),
        "--replay-before",
        str(args.replay_before),
        "--out-report",
        str(args.out_dir / "preflight.json"),
    ]
    print("preflight: $ " + " ".join(preflight_argv))
    for row in plan:
        print("plan: $ " + " ".join(row))
    if args.dry_run:
        print(f"DRY_RUN commands={len(plan)}")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rc = _run(preflight_argv)
    if rc != 0:
        print(f"REFUSE: 预检 rc={rc} ⇒ 没算成，不花评测的时间")
        return 2
    payload = json.loads((args.out_dir / "preflight.json").read_text(encoding="utf-8"))
    refusals = preflight_refusals(payload)
    if refusals:
        print("REFUSE: 配对的硬前置未满足：")
        for row in refusals:
            print("  - " + row)
        return 2
    for row in plan:
        rc = _run(row)
        if rc != 0:
            print(f"REFUSE: 步骤失败 rc={rc}：{' '.join(row[1:3])} ⇒ 后续步骤不跑（半套面不可判）")
            return rc
    print(f"LANE_OK out_dir={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
