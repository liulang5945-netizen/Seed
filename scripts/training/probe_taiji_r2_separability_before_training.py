"""R2：把"训练前"那一端的可分离性剖面量出来，与训练后的**同一把尺子**对齐（零训练、分钟级）。

## 为什么要这一件

Step 0 的剖面是在**训练过 16M** 的 `seed_beta.pt` 上量的（trace 块 0.5766）。
初始化筛选器顺手量到**同架构**的 trace 块是 0.7256 ——两者差 **−0.149**。
如果这真的是"训练把槽结构花掉了"，那整个方向要改：**缺的不是某个零件的形状，而是训练目标
本身不奖励"按角色分解"**。

**但在下这个结论之前必须先验可比性**：新构造的 `Taiji(config)` 与训练器用的
`Seed(SeedConfig(...))` **未必是同一条构造路径**（多一个器官、多一次抽样，随机流就错开了，
量出来的差就不是"训练造成的"）。所以本件**用训练器那条路径**（`Seed`）从 `seed_beta.pt` 的 config
重建**同一份模型的第 0 tick 状态**，再用**同一个** `_cells` 仪器量一遍。

⇒ 只有这条"同一构造路径、同一把尺子"的对比，才允许说"训练让它掉了多少"。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts" / "training") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))

from probe_taiji_r2_compositionality import (  # noqa: E402
    ENTANGLED_CEILING,
    FAMILIES,
    SEPARABLE_FLOOR,
)
from probe_taiji_r2_separability_profile import STAGES, _cells, _cos, _median  # noqa: E402


def measure(
    substrate: Any, label: str, families: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    per_stage: dict[str, list[float]] = {stage: [] for stage in STAGES}
    us: dict[str, list[torch.Tensor]] = {stage: [] for stage in STAGES}
    for family in families if families is not None else FAMILIES:
        cells = _cells(substrate, None, family, label)
        for stage in STAGES:
            c00, c10 = cells["00"][stage], cells["10"][stage]
            c01, c11 = cells["01"][stage], cells["11"][stage]
            per_stage[stage].append(_cos(c10 - c00, c11 - c01))
            us[stage].append(c10 - c00)
    profile: dict[str, Any] = {}
    for stage in STAGES:
        baseline = [
            _cos(us[stage][i], us[stage][j])
            for i in range(len(FAMILIES))
            for j in range(i + 1, len(FAMILIES))
        ]
        median = _median(per_stage[stage])
        profile[stage] = {
            "consistency_median": round(median, 6),
            "baseline_median": round(_median(baseline), 6),
            "verdict": (
                "separable"
                if median >= SEPARABLE_FLOOR
                else ("partial" if median >= ENTANGLED_CEILING else "not_separable")
            ),
        }
    return profile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trained", default=str(PROJECT_ROOT / "checkpoints" / "seed_beta.pt"))
    parser.add_argument("--out-report", default=None)
    args = parser.parse_args()
    from seed import Seed, SeedConfig
    from taiji import TaijiConfig

    checkpoint = Path(args.trained)
    envelope = torch.load(checkpoint, map_location="cpu", weights_only=False)
    values = dict(envelope["config"]["taiji"])
    values.pop("receptors_factored", None)  # 训练前那一端必须是**未分块**的现行架构

    fresh = Seed(SeedConfig(taiji=TaijiConfig.from_dict(values)))
    before = measure(fresh.architecture, "before")

    # 训练后那一端：同一个构造路径读回 checkpoint（而不是另起 Taiji）
    trained_seed = Seed.from_checkpoint(envelope)
    after = measure(trained_seed.architecture, "after")

    report_path = Path(
        args.out_report
        or PROJECT_ROOT / "reports" / "taiji_r2_separability_before_training_20260923.json"
    )
    if report_path.exists():
        parser.error(f"{report_path} already exists; 判决件不覆写")

    delta = {
        stage: {
            "before_tick0": before[stage]["consistency_median"],
            "after_16m": after[stage]["consistency_median"],
            "delta": round(after[stage]["consistency_median"] - before[stage]["consistency_median"], 6),
        }
        for stage in STAGES
    }
    payload = {
        "format": "taiji-r2-separability-before-vs-after-training-v1",
        "written_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "question": "同一构造路径、同一把尺子：训练 16M 让各级可分离性变了多少",
        "comparability": (
            "两端都走 `Seed(SeedConfig(...))` 这条训练器路径；前端是同一 config 的第 0 tick，"
            "后端是 `Seed.from_checkpoint` 读回的同一份 checkpoint"
        ),
        "disclosures": [
            "未能核实训练器当初是否**恰好**从这一份第 0 tick 配置起步；"
            "若训练器另加过器官或另设过 seed，本对比就名不副实——数字要按此打折读",
            "刺激集仍是那 6 个冻结族",
        ],
        "profile_before_tick0": before,
        "profile_after_16m": after,
        "per_stage_delta": delta,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"report": str(report_path)}))
    print(f"  {'stage':22} {'tick0':>10} {'16M':>10} {'delta':>10}")
    for stage in STAGES:
        row = delta[stage]
        print(f"  {stage:22} {row['before_tick0']:+10.4f} {row['after_16m']:+10.4f} {row['delta']:+10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
