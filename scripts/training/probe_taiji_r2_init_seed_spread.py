"""R2：初始化可分离性的**跨种子分布**——用来判定"训练把它花掉了"是不是真的（零训练、分钟级）。

## 为什么要这一件（这是自我证伪的一件）

`probe_taiji_r2_separability_before_training.py` 给出"tick0 0.7256 → 16M 0.5766（−0.149）"。
**但那个对比有一个没堵住的洞**：我无法核实训练器当初**恰好**从这一份第 0 tick 配置起步。
只要训练器当时另设过 seed，这个差就有一部分是**种子差异**，不是训练造成的。

⇒ 本件把**初始化状态本身的分布**量出来（同一 config、只换 seed，取多个），
再看训练后的值落在分布里的什么位置：

* 训练后的值**远在分布之外** ⇒ "训练把结构花掉了"站得住。
* 训练后的值**落在分布之内** ⇒ 上面那个差是噪声，**那条结论撤回**。

**判读线先写死**：以初始化的 `min`/`max` 为界；训练值在界外记 `beyond_init_range`，
在界内记 `within_init_range`，并同时给出它相对初始化的 z 分数（样本少，只作旁注）。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts" / "training") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))

from probe_taiji_r2_compositionality import FAMILIES  # noqa: E402
from probe_taiji_r2_separability_before_training import measure  # noqa: E402

#: 预注册的种子集合（先定后跑）
SEEDS: tuple[int, ...] = (20260822, 20260823, 20260824, 20260825, 20260826, 20260827, 20260828, 20260901)
#: 训练后（`seed_beta.pt`，16M）的读数，取自冻结的 Step 0 分块剖面
TRAINED = {"1_activity_block": 0.4586, "2_trace_block": 0.5766, "3_context": 0.4197}


def _stats(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)
    n = len(ordered)
    mean = sum(ordered) / n
    var = sum((v - mean) ** 2 for v in ordered) / (n - 1) if n > 1 else 0.0
    return {
        "n": n,
        "min": round(ordered[0], 6),
        "median": round(ordered[n // 2], 6),
        "max": round(ordered[-1], 6),
        "mean": round(mean, 6),
        "stdev": round(var**0.5, 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-from", default=str(PROJECT_ROOT / "checkpoints" / "seed_beta.pt"))
    parser.add_argument(
        "--families-file",
        default=None,
        help="给一个族集 JSON（如确认集）就量那个集；缺省用筛选集 6 族",
    )
    parser.add_argument("--out-report", default=None)
    args = parser.parse_args()
    from seed import Seed, SeedConfig
    from taiji import TaijiConfig

    envelope = torch.load(Path(args.config_from), map_location="cpu", weights_only=False)
    values = dict(envelope["config"]["taiji"])
    values.pop("receptors_factored", None)
    base = TaijiConfig.from_dict(values)
    families = FAMILIES
    if args.families_file:
        payload = json.loads(Path(args.families_file).read_text(encoding="utf-8"))
        families = payload["families"]

    per_seed: list[dict[str, Any]] = []
    for seed in SEEDS:
        model = Seed(SeedConfig(taiji=replace(base, seed=seed)))
        per_seed.append({"seed": seed, **measure(model.architecture, f"seed{seed}", families)})

    stages = sorted(TRAINED)
    summary: dict[str, Any] = {}
    for stage in stages:
        values_at_init = [row[stage]["consistency_median"] for row in per_seed]
        stats = _stats(values_at_init)
        trained = TRAINED[stage]
        beyond = trained < stats["min"] or trained > stats["max"]
        z = (trained - stats["mean"]) / stats["stdev"] if stats["stdev"] > 0 else None
        summary[stage] = {
            "init_stats": stats,
            "init_values": [round(v, 6) for v in values_at_init],
            "trained_16m": trained,
            "verdict": "beyond_init_range" if beyond else "within_init_range",
            "trained_z_score": None if z is None else round(z, 3),
        }

    payload = {
        "format": "taiji-r2-init-seed-spread-v1",
        "written_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "question": "训练后 16M 的可分离性是否落在初始化（跨 seed）分布之外——即'训练把它花掉了'是否成立",
        "preregistered_seeds": list(SEEDS),
        "preregistered_rule": "训练值 < init min 或 > init max ⇒ beyond_init_range；否则 within_init_range",
        "disclosures": [
            f"样本只有 {len(SEEDS)} 个 seed，min/max 作为界很脆；z 分数也只作旁注",
            "本件只判'训练前后之差是否超出初始化波动'，不判因果机制",
            "刺激集仍是那 6 个冻结族",
        ],
        "per_seed": per_seed,
        "summary": summary,
    }
    report_path = Path(
        args.out_report
        or PROJECT_ROOT / "reports" / "taiji_r2_init_seed_spread_20260923.json"
    )
    if report_path.exists():
        parser.error(f"{report_path} already exists; 判决件不覆写")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"report": str(report_path)}))
    print(f"  {'stage':20} {'init min':>9} {'median':>9} {'max':>9} {'trained':>9} {'z':>7}  verdict")
    for stage in stages:
        row = summary[stage]
        stats = row["init_stats"]
        print(
            f"  {stage:20} {stats['min']:+9.4f} {stats['median']:+9.4f} {stats['max']:+9.4f}"
            f" {row['trained_16m']:+9.4f} {str(row['trained_z_score']):>7}  {row['verdict']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
