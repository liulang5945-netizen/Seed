"""R2 Step 1 的**快速筛选器**：在初始化状态下扫"两个半块各分多少通道"（零训练、分钟级）。

## 为什么能当筛选器

`receptors` 是固定的随机稀疏压缩、不训练 ⇒ 这次改动的效果**在初始化时就已完整存在**。
所以"哪种分法能把最终 cue 的可分离性抬起来"可以在**训练之前**就量出来，
把 22–38 h 的长跑留给**唯一值得跑的那一种**。

## 筛选设计（写在代码里，先定后跑）

* 把 1152 的两半映射到 96 个通道：activity 半拿 **k** 个、trace 半拿 **96−k** 个。
* 扫 `k ∈ {24, 32, 40, 48, 56, 64, 72}`，另加 `k=0/96`（**单边独占**）作边界参考。
* 每档都用**独立固定种子**的 generator 建表（与 `organs.py` 里那条路同一个种子常量），
  保证"同一档在不同时刻重跑得到同一张表"。
* 读数＝Step 0 分块口径下 **`3_context`** 与 `5_readout_dist` 的 `consistency` 中位
  （外加第 2 级作对照，它不该随 k 变）。

## 必须一起声明的风险（不许省）

刺激集只有 6 个族，**在这个小集合上挑最好的 k 就是在挑**。⇒ 本件的产出**只是候选**，
真正的判据仍是在**训练后的配对臂**上按 Step 0 口径量；本件**不构成任何结论**。
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

#: 预注册的扫描档位（先定后跑，跑完不许加档）
SPLITS: tuple[int, ...] = (0, 24, 32, 40, 48, 56, 64, 72, 96)


def measure(substrate: Any, label: str) -> dict[str, float]:
    per_stage: dict[str, list[float]] = {stage: [] for stage in STAGES}
    for family in FAMILIES:
        cells = _cells(substrate, None, family, label)
        for stage in STAGES:
            c00, c10 = cells["00"][stage], cells["10"][stage]
            c01, c11 = cells["01"][stage], cells["11"][stage]
            per_stage[stage].append(_cos(c10 - c00, c11 - c01))
    return {stage: round(_median(values), 6) for stage, values in per_stage.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-from", default=str(PROJECT_ROOT / "checkpoints" / "seed_beta.pt"))
    parser.add_argument("--out-report", default=None)
    args = parser.parse_args()
    from taiji import Taiji, TaijiConfig
    from taiji.organs import RECEPTOR_FACTOR_SEED, SparseReceptorBank

    env = torch.load(Path(args.config_from), map_location="cpu", weights_only=False)
    values = dict(env["config"]["taiji"])
    values["receptors_factored"] = False
    config = TaijiConfig.from_dict(values)
    split = sum(config.region_sizes)
    width = config.motor_context_dim

    rows: list[dict[str, Any]] = []
    for k in SPLITS:
        if k == width // 2:
            # 48/48 ＝ organs.py 里实现的那一档，走内置构造（不靠属性手术）
            values_on = dict(values)
            values_on["receptors_factored"] = True
            model = Taiji(TaijiConfig.from_dict(values_on))
        elif 0 < k < width:
            # 其它配比靠属性手术建表；`encode` 读的就是这两个属性，所以能表达任意 (k, 96-k)
            model = Taiji(config)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(RECEPTOR_FACTOR_SEED)
            model.predictive_context.receptors_factor = (
                SparseReceptorBank(
                    split, k, generator=generator, context_norm=config.motor_context_norm
                ),
                SparseReceptorBank(
                    config.cortical_context_dim - split,
                    width - k,
                    generator=generator,
                    context_norm=config.motor_context_norm,
                ),
            )
        else:
            # 单边独占（k=0 或 k=96）表达不了：`SparseReceptorBank` 不接受零宽输出。
            # 如实跳过，不拿近似值冒充那一档。
            continue

        rows.append({"k_activity": k, "k_trace": width - k, **measure(model, f"k{k}")})

    ranked = sorted(rows, key=lambda row: row["3_context"], reverse=True)
    payload = {
        "format": "taiji-r2-factorization-split-screen-v1",
        "written_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "question": "1152 的两半各分多少通道，能把最终 cue 的可分离性抬得最高（初始化口径）",
        "preregistered_splits": list(SPLITS),
        "ladder": {"separable_floor": SEPARABLE_FLOOR, "not_separable_below": ENTANGLED_CEILING},
        "disclosures": [
            "刺激集只有 6 个族，在小集合上挑最好的 k 就是在挑 ⇒ 本件产出只是候选，不是结论",
            "真正的判据仍是在训练后的配对臂上按 Step 0 口径量",
            "k=0/96 的独占档未实现（SparseReceptorBank 不接受零宽），已在代码里跳过",
        ],
        "rows": rows,
        "ranked_by_context_consistency": [row["k_activity"] for row in ranked],
        "best_k_activity": ranked[0]["k_activity"] if ranked else None,
    }
    report_path = Path(
        args.out_report
        or PROJECT_ROOT / "reports" / "taiji_r2_factorization_split_screen_20260923.json"
    )
    if report_path.exists():
        parser.error(f"{report_path} already exists; 判决件不覆写")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps({"report": str(report_path)}))
    print(f"  {'k_act':>6} {'k_trace':>8} {'2_trace':>10} {'3_context':>11} {'5_readout':>11}")
    for row in rows:
        print(
            f"  {row['k_activity']:>6} {row['k_trace']:>8} {row['2_trace_block']:>10.4f}"
            f" {row['3_context']:>11.4f} {row['5_readout_dist']:>11.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
