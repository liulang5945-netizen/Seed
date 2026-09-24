"""零训练审计：退化是**三个区均匀**发生，还是**集中在某一个区**？（只读 `t1` 既有存档）

## 为什么这一刀可能有信息

写入块里有四笔，它们的**作用范围不同**：

* `lateral.anti_hebbian_update` 是**区内**竞争（同区的单元之间）⇒ 只能抹**区内**的方向；
* `decoder.local_update` / `transition.local_update` 用 **`old.trace`（上一 tick）** 当资格迹
  ⇒ 抹的是**跨时间**的方向；
* `trace_baselines` 是**每区一个共享零点**（`t1` 里区2 的 ‖baseline‖ 涨了 4.4 倍、区1 涨 2.2 倍、区0 基本平）。

如果退化**集中在一个区**，"哪一笔"就有了第一条线索（例如退化最重的区恰好是基线漂得最凶的那个）。
如果**三个区同步退化**，那更可能是**共享机制**（例如所有笔都吃同一个 `old.trace`），
逐笔分离的优先级就要重排。

## 做法

逐个读 `t1` 的既有存档（默认每 500k 一个），对每个区 **r** 单独算

`consistency_r = cos( 换A方向在B=b0 , 换A方向在B=b1 )`

"换A方向"取**只属于区 r 的切片**（`trace` 块与 `activity` 块各取该段）。
与全区拼在一起的读数并排。**判读线**：某区显著先破 ⇒ 有空间结构；三区同步 ⇒ 指向共享机制。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts" / "training") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))

from probe_taiji_r2_compositionality import FAMILIES  # noqa: E402
from probe_taiji_r2_separability_profile import _cells, _cos, _median  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-dir", default="output/taiji_r2_t1t2/t1")
    parser.add_argument("--stride", type=int, default=500000)
    parser.add_argument("--out-report", default=None)
    args = parser.parse_args()

    from seed import Seed, SeedConfig
    from taiji import TaijiConfig

    arm_dir = Path(args.arm_dir)
    if not arm_dir.is_absolute():
        arm_dir = PROJECT_ROOT / arm_dir

    def tick_of(path: Path) -> int:
        return int(path.stem.split("_")[-1])

    selected = sorted(
        (p for p in arm_dir.glob("checkpoint_*.pt") if tick_of(p) % args.stride == 0),
        key=tick_of,
    )
    if not selected:
        raise SystemExit("没有可选的存档")

    first = torch.load(selected[0], map_location="cpu", weights_only=False)
    values = dict(first["config"]["taiji"])
    values.pop("receptors_factored", None)
    config = SeedConfig(taiji=TaijiConfig.from_dict(values))
    sizes = tuple(config.taiji.region_sizes)

    # 每区的切片（trace 块与 activity 块都是按区顺序拼接的）
    bounds: list[tuple[int, int]] = []
    start = 0
    for size in sizes:
        bounds.append((start, start + size))
        start += size
    total = start

    curves: dict[str, list[float]] = {}
    ticks_seen: list[int] = []
    for path in selected:
        tick = tick_of(path)
        envelope = torch.load(path, map_location="cpu", weights_only=False)
        substrate = Seed.from_checkpoint(envelope).architecture
        ticks_seen.append(tick)

        per_key: dict[str, list[float]] = {}
        for family in FAMILIES:
            cells = _cells(substrate, None, family, f"pr-{tick}")
            # 逐区 + 全区，一次算完
            scopes: list[tuple[str, int, int]] = [
                (f"r{index}", lo, hi) for index, (lo, hi) in enumerate(bounds)
            ]
            scopes.append(("all", 0, total))
            for name, lo, hi in scopes:
                for block in ("2_trace_block", "1_activity_block"):
                    a0 = cells["00"][block][lo:hi]
                    a1 = cells["10"][block][lo:hi]
                    b0 = cells["01"][block][lo:hi]
                    b1 = cells["11"][block][lo:hi]
                    per_key.setdefault(f"{name}_{block}", []).append(_cos(a1 - a0, b1 - b0))

        for key, values_list in per_key.items():
            curves.setdefault(key, []).append(round(_median(values_list), 6))

    payload = {
        "format": "taiji-r2-per-region-degradation-v1",
        "arm_dir": str(arm_dir),
        "region_sizes": list(sizes),
        "ticks": ticks_seen,
        "curves": curves,
        "note": "零训练、只读既有存档；判读线见模块文档（某区先破 ⇒ 有空间结构；三区同步 ⇒ 指向共享机制）",
    }
    out = Path(args.out_report) if args.out_report else (
        PROJECT_ROOT / "reports" / "taiji_r2_per_region_degradation_20260924.json"
    )
    if out.exists():
        parser.error(f"{out} already exists; 判决件不覆写")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"report": str(out), "points": len(ticks_seen)}))
    keys = list(curves)
    print("  tick   " + "".join(f" {k:>18}" for k in keys))
    for index, tick in enumerate(ticks_seen):
        print(f"  {tick//1000:>5}k " + "".join(f" {curves[k][index]:>18.4f}" for k in keys))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
