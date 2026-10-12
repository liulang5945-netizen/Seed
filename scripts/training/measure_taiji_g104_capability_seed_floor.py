"""DEBT-G104 的噪声地板读数：`test_episodic_field` 那条能力钉子到底钉在分布的哪个位置。

为什么需要这台仪器（不是临时探针）：㊵-715 的五种子对照已经显示 `recalled >= 0.875` 在
**加宽之前**就只有 2/5 的种子达标；㊵-725 把来源词表加到 5 枚之后该档又掉到 0.625。
两次都指向同一件事：这个阈值量的是"某一次随机基板抽得好不好"，不是"记忆通路有没有用"。
要把钉子改到它真正能声称的东西上，就必须有一份**可复算**的地板读数，而不是我在提交信息里说一句。

面（face）：复用 `tests/taiji_native/test_episodic_field.py` 自己的 helper，一行生成链都不重抄；
两臂＝当前默认词表 vs 少一枚的词表（`EPISODIC_PROVENANCE_KINDS` 打补丁），同一批种子。
零训练（同一 tiny 配置、只跑那支测本来就跑的 8 个 episode），不落任何模型文件。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
REPORT_FORMAT = "taiji-g104-capability-seed-floor-v1"
TEST_MODULE = PROJECT_ROOT / "tests" / "taiji_native" / "test_episodic_field.py"
#: 种子表与两个旧阈值、噪声地板全部住在测册（`PLAN-N5-17` G-N5p-1），本仪器按那条 `importlib` 链现取，
#: 这样换种子或换阈值只会改一处，读数与判据不可能各自漂移。


def _load_test_helpers():
    spec = importlib.util.spec_from_file_location("g104_seed_floor_source", TEST_MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sweep(seeds: list[int] | None) -> dict[str, object]:
    from taiji import Taiji, TaijiConfig
    from taiji.config import EPISODIC_PROVENANCE_KINDS
    from taiji.memory import EpisodicField

    helpers = _load_test_helpers()
    if not seeds:
        seeds = list(helpers.CAPABILITY_SEEDS)
    absolute_floor = helpers.LEGACY_ABSOLUTE_THRESHOLD
    margin_floor = helpers.LEGACY_MARGIN_THRESHOLD
    noise_floor = helpers.CAPABILITY_MARGIN_FLOOR
    arms = {
        "default_vocabulary": EPISODIC_PROVENANCE_KINDS,
        "one_narrower_vocabulary": EPISODIC_PROVENANCE_KINDS[:-1],
    }
    per_arm: dict[str, object] = {}
    for name, kinds in arms.items():
        EpisodicField.PROVENANCE_KINDS = kinds
        rows = []
        for seed in seeds:
            config = TaijiConfig(**{**helpers._config().to_dict(), "seed": seed})
            model = Taiji(config)
            mapping = helpers._record_balanced_one_shot_episodes(model)
            checkpoint = model.checkpoint()
            recalled, _ = helpers._recall_accuracy(checkpoint, mapping, use_memory=True)
            trace_only, _ = helpers._recall_accuracy(checkpoint, mapping, use_memory=False)
            rows.append(
                {
                    "seed": seed,
                    "recalled": round(recalled, 6),
                    "trace_only": round(trace_only, 6),
                    "margin": round(recalled - trace_only, 6),
                    "meets_absolute_floor": recalled >= absolute_floor,
                    "meets_margin_floor": recalled - trace_only >= margin_floor,
                }
            )
        per_arm[name] = {
            "kinds": list(kinds),
            "rows": rows,
            "recalled_min": min(row["recalled"] for row in rows),
            "recalled_median": statistics.median([row["recalled"] for row in rows]),
            "recalled_max": max(row["recalled"] for row in rows),
            "absolute_floor_passes": sum(row["meets_absolute_floor"] for row in rows),
            "margin_floor_passes": sum(row["meets_margin_floor"] for row in rows),
            "margin_min": min(row["margin"] for row in rows),
            "margin_median": statistics.median([row["margin"] for row in rows]),
            "margin_below_noise_floor": sum(row["margin"] < noise_floor for row in rows),
        }
    EpisodicField.PROVENANCE_KINDS = EPISODIC_PROVENANCE_KINDS
    return {
        "format": REPORT_FORMAT,
        "prereg": "plans/reference/PLAN-N5-17_capability_seed_distribution_prereg_20261012.md",
        "debt": "DEBT-G104",
        "seeds": seeds,
        "absolute_floor": absolute_floor,
        "margin_floor": margin_floor,
        "noise_floor": noise_floor,
        "arms": per_arm,
        "what_this_does_not_prove": [
            "这不证明加宽提高了能力，也不证明它降低了能力——两臂都是同一套随机抽样下的分布读数",
            "这不证明 `recalled` 的绝对值是机制性质：达标数本身就是这条读数的产出",
            "种子数只有 len(seeds) 个，地板的下沿可能更低",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-report",
        default="reports/taiji_g104_capability_seed_floor_20261012.json",
        help="落盘路径（件已存在则拒覆盖 rc=2，除非 --overwrite）",
    )
    parser.add_argument("--seeds", action="append", type=int, default=[])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = _sweep(args.seeds)
    out = Path(args.out_report)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    if out.exists() and not args.overwrite:
        print(f"refusing to overwrite an existing report: {out}", file=sys.stderr)
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "seeds": len(report["seeds"]),
                "arms": {
                    name: {
                        "absolute_floor_passes": arm["absolute_floor_passes"],
                        "margin_floor_passes": arm["margin_floor_passes"],
                        "recalled_min": arm["recalled_min"],
                        "margin_min": arm["margin_min"],
                        "margin_median": arm["margin_median"],
                        "margin_below_noise_floor": arm["margin_below_noise_floor"],
                    }
                    for name, arm in report["arms"].items()
                },
                "out": out.as_posix(),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
