"""零训练审计：`trace_baselines`（对手基点/稳态）在 `t1` 的既有存档里是怎么漂的。

## 为什么查它

读 `taiji/fabric.py:392-470` 的写入块得到一件事：那里有**四笔权重写入 + 一笔状态写入**——

| 写入 | 是否被 `learn_scale` 缩放 |
|---|---|
| `decoder.local_update`（互预测） | 是 |
| `consolidation_decoders[…]` | 是（走 `consolidation_learn_scale`，本配方为 0 ⇒ 不写） |
| `transition.local_update`（自预测） | 是 |
| `lateral.anti_hebbian_update` | 是 |
| **`trace_baselines[index].lerp_(trace, cortical_baseline_rate)`** | **否** |

而 `fabric.step` 的 `adapt_homeostasis` **默认 True**，醒来路径（`model.py:1993`）**没有覆盖它**
⇒ 训练时它**每 tick 都在动**。**这就解释了 T3 为什么三臂全不中**：我扫的每一个旋钮
（`predictive_learning_rate` / `synapse_decay` / `lateral_learning_rate` / `transition_learning_rate`）
**都碰不到这一笔**。

更要紧的是它的**结构地位**：`opponent_trace(i, trace) = trace − trace_baselines[i]` —— 一个**每区共享的零点**，
对区内**所有单元做同一次减法**。它**无法编码"哪个槽"**，却会**改变偏差的尺度** ⇒ 结构上正是会把
"槽方向"揉掉的那类操作。

## 本件做什么（**零训练**，只读 `t1` 已有存档）

`t1` 每个 250k 都留了带 tick 的存档（保号存档的收益）。逐个读 `fabric.trace_baselines`，量：

1. 每个区的 `‖baseline‖` 随 tick 的曲线；
2. 相邻快照之间 baseline 的**位移量**（它是不是一直在动、动多快）；
3. 与已知的结构读数（`t1` 的 `3_context` 曲线）**并排**，看 baseline 的行为在 **4.25M 那个下台阶**附近有没有变化。

**判读（先写死）**：

* baseline **在 4.25M 附近出现行为变化**（位移量跃变/单调累积到某量级）⇒ **假说增强**，
  值得花 5.5 h 跑"关掉 homeostasis"的判别臂。
* baseline **一直在平稳漂移、与 4.25M 无关** ⇒ **假说减弱**，回到 fabric 的权重写入里继续找。
* ⚠️ 本件**只是相关**，不是因果；因果要靠那条判别臂。
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-dir", default="output/taiji_r2_t1t2/t1")
    parser.add_argument("--out-report", default=None)
    args = parser.parse_args()

    arm_dir = Path(args.arm_dir)
    if not arm_dir.is_absolute():
        arm_dir = PROJECT_ROOT / arm_dir
    # ⚠️ 按 **tick 数值** 排序，不能按文件名——驱动写的是**不补零**的 `checkpoint_<ticks>.pt`，
    # 字典序会把 `250000` 排到 `2500000` 后面（我首版就是这么错的，"位移"整列成了跟错前驱的差）。
    def _tick_of(path: Path) -> int:
        return int(path.stem.split("_")[-1])

    snapshots = sorted(arm_dir.glob("checkpoint_*.pt"), key=_tick_of)
    if not snapshots:
        raise SystemExit(f"{arm_dir} 里没有 checkpoint_*.pt；拒跑")
    ticks = [_tick_of(p) for p in snapshots]
    if ticks != sorted(ticks):  # pragma: no cover - 防御性
        raise SystemExit("快照 tick 不是单调的，拒跑")

    # 结构读数（同一条轨迹），用来并排看
    trajectory = arm_dir / "trajectory.jsonl"
    structure: dict[int, float] = {}
    if trajectory.is_file():
        for line in trajectory.read_text(encoding="utf-8").strip().splitlines():
            row = json.loads(line)
            if "screen" in row:
                structure[int(row["ticks"])] = row["screen"]["3_context"]

    previous: list[torch.Tensor] | None = None
    rows: list[dict[str, object]] = []
    for path in snapshots:
        envelope = torch.load(path, map_location="cpu", weights_only=False)
        tick = int(envelope["metadata"]["tick"])
        baselines = [b.detach().double() for b in envelope["substrate"]["fabric"]["trace_baselines"]]
        norms = [round(float(b.norm()), 6) for b in baselines]
        shift = None
        if previous is not None:
            shift = round(sum(float((a - b).norm()) for a, b in zip(baselines, previous, strict=True)), 6)
        rows.append(
            {
                "tick": tick,
                "baseline_norms": norms,
                "baseline_shift_from_prev": shift,
                "structure_context": structure.get(tick),
            }
        )
        previous = baselines

    first = rows[0]["baseline_norms"]
    last = rows[-1]["baseline_norms"]
    payload = {
        "format": "taiji-r2-baseline-drift-audit-v1",
        "question": "trace_baselines 的漂移是否与 4.25M 那个结构性下台阶相关",
        "arm_dir": str(arm_dir),
        "n_snapshots": len(rows),
        "first_norms": first,
        "last_norms": last,
        "readings": rows,
        "note": "零训练、只读既有存档；本件只给相关性，因果要靠 adapt_homeostasis=False 的判别臂",
    }
    out = Path(args.out_report) if args.out_report else (
        PROJECT_ROOT / "reports" / "taiji_r2_baseline_drift_audit_20260924.json"
    )
    if out.exists():
        parser.error(f"{out} already exists; 判决件不覆写")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"report": str(out), "n": len(rows)}))
    print(f"  {'tick':>10} {'‖base‖':>28} {'位移':>10} {'结构':>8}")
    for row in rows:
        shift = row["baseline_shift_from_prev"]
        print(
            f"  {row['tick']:>10} {str(row['baseline_norms']):>28} "
            f"{'-' if shift is None else format(shift, '.4f'):>10} "
            f"{'-' if row['structure_context'] is None else format(row['structure_context'], '.4f'):>8}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
