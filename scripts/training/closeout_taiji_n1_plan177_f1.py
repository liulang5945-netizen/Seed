"""N177 F1 结案件：把 12 个 run 的资源账、旧保持探针、holdout 判定合成一份收口。

只做三件事，全部 fail-closed：
1. 12 个 run 都 `completed`，且端点文件 sha256 与 run_state 记录一致；
2. 旧保持探针件在且 `worst_drop_pp ≤ 3`；
3. holdout 件在，`evaluation_attempts` 唯一（attempt 文件只一份），且其 `decision.verdict`
   与按同一份 `scores` 重算的判定一致（不允许手改结论）。

任一项不满足就拒绝出版，不生成"半份结案"。输出写一次。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.training.evaluate_taiji_n1_plan177_f_f1_holdout import (  # noqa: E402
    ARMS,
    verdict,
)

F1_ROOT = ROOT / "data/checkpoints/n1_plan177_f_f1_20261029"
RETENTION = ROOT / "reports/taiji_n1_plan177_f1_old_retention_20261029.json"
HOLDOUT = ROOT / "reports/taiji_n1_plan177_f_f1_holdout_20261029.json"
REPORT = ROOT / "reports/taiji_n1_plan177_f1_closeout_20261029.json"
MAX_DROP_PP = 3.0


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_runs(run_root: Path) -> list[dict[str, Any]]:
    runs = []
    for seed in (0, 1, 2):
        for arm in ARMS:
            state_path = run_root / f"{arm}__seed_{seed}" / "run_state.json"
            if not state_path.is_file():
                raise FileNotFoundError(f"missing F1 run state: {state_path}")
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("status") != "completed":
                raise ValueError(f"{arm}__seed_{seed} not completed: {state.get('status')}")
            endpoint = Path(state["endpoint"])
            if not endpoint.is_absolute():
                endpoint = ROOT / endpoint
            if not endpoint.is_file():
                raise FileNotFoundError(f"missing endpoint: {endpoint}")
            actual = _sha256_file(endpoint)
            if actual != state.get("endpoint_sha256"):
                raise ValueError(f"{arm}__seed_{seed} endpoint sha differs from its ledger")
            runs.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "updates": state["updates"],
                    "process_cpu_seconds": state["process_cpu_seconds"],
                    "wall_seconds": state.get("wall_seconds"),
                    "endpoint": state["endpoint"],
                    "endpoint_sha256": actual,
                    "resumes": state.get("resumes", 0),
                }
            )
    return runs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="N177 F1 结案")
    parser.add_argument("--run-root", type=Path, default=F1_ROOT)
    parser.add_argument("--retention", type=Path, default=RETENTION)
    parser.add_argument("--holdout", type=Path, default=HOLDOUT)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument(
        "--attempt",
        type=Path,
        default=None,
        help="一次性 attempt 文件；缺省按 holdout 同名查找",
    )
    args = parser.parse_args(argv)
    if args.report.exists():
        raise ValueError("F1 closeout is write-once")

    runs = collect_runs(args.run_root)
    if not args.retention.is_file():
        raise FileNotFoundError("old-retention probe report is missing")
    retention = json.loads(args.retention.read_text(encoding="utf-8"))
    retention_pp = float(retention.get("main_worst_drop_pp", retention["worst_drop_pp"]))
    if not args.holdout.is_file():
        raise FileNotFoundError("holdout report is missing")
    holdout = json.loads(args.holdout.read_text(encoding="utf-8"))
    attempts = (
        [args.attempt]
        if args.attempt is not None
        else list(args.holdout.parent.glob(args.holdout.stem + ".attempt.json"))
    )
    if len(attempts) != 1 or not attempts[0].is_file():
        raise ValueError(f"holdout must have exactly one attempt file, found {len(attempts)}")
    recomputed = verdict(holdout["scores"], retention_pp)
    recorded = holdout.get("decision", {})
    if recorded.get("verdict") != recomputed["verdict"]:
        raise ValueError(
            f"recorded verdict {recorded.get('verdict')} != recomputed {recomputed['verdict']}"
        )

    total_cpu = sum(run["process_cpu_seconds"] for run in runs)
    report = {
        "format": "taiji-n177-f1-closeout-v1",
        "date": "2026-10-29",
        "candidate": {
            "trainable": "value encoder + byte bias",
            "frozen": "producer, V4 decoder",
            "train_entities": ["钥匙", "杯子", "书", "盒子"],
            "updates_per_run": 1280,
            "runs": 12,
        },
        "verdict": recomputed["verdict"],
        "reasons": recomputed["reasons"],
        "old_retention_main_worst_drop_pp": retention_pp,
        "old_retention_all_arms_worst_drop_pp": float(retention["worst_drop_pp"]),
        "scores": holdout["scores"],
        "gates": holdout["gates"],
        "runs": runs,
        "cost": {
            "process_cpu_seconds": total_cpu,
            "budget_cap": 121500.0,
            "within_cap": total_cpu <= 121500.0,
        },
        "evidence": {
            "holdout": _rel(args.holdout),
            "retention": _rel(args.retention),
            "runs_root": _rel(args.run_root),
        },
        "next_direction": {
            "PASS_LEARNED_VALUE": "状态消费在 holdout 上成立：进入 F2 的跨实体/跨会话泛化复验。",
            "FAIL_COPY_ONLY": "主路径没超出复制下界：先查连续特征的可分性，再谈信用。",
            "FAIL_NO_STATE_EFFECT": "特征未被消费：回到 selector/解码器接入面，不调阈值。",
            "FAIL_NO_LEARNING": "回答信用没到负责模块：先证梯度与参数责任，再谈收益。",
            "FAIL_NO_GENERALIZATION": "训练侧饱和但 holdout 不过：查过拟合与实体覆盖。",
            "INCONCLUSIVE": "工程有效但效应不足或缺项：按缺失项逐个补证，不重跑同一配置。",
        }.get(recomputed["verdict"], "未预置的判定：人工复核后再定后继。"),
    }
    temporary = args.report.with_name(args.report.name + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(args.report)
    print(
        f"VERDICT={report['verdict']} cpu={total_cpu:.1f} within_cap={report['cost']['within_cap']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
