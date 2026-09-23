"""R2 T1/T2 驱动：短跑 + **沿途就地测量**可分离性（判读线见预注册）。

预注册：`plans/reference/M5_R2_T1_T2_PREREG_20260923.md`（判读线在开跑前已冻结）。

## 两条设计约束（为什么不能直接用现成训练器分段跑）

1. **数据不能重放**：`train_seed_corpus.run_training` 的 `--resume` 只恢复模型与计数器，
   **语料仍从头迭代** ⇒ 分段续跑会重看已见数据，退化曲线就不可信。
   所以本件在**一次**运行里沿途自存快照。
2. **配方要逐字一致**：语料迭代**直接复用训练器的** `iter_corpus_symbols`
   （同一个 `boundary`、同一个顺序），主循环照抄训练器：
   `model.observe(symbol, learn=True)`，只额外挂上消融开关。

## 测量

每个测量点用**分块口径**（Step 0 剖面仪器的 `_cells`）量 `context` 与 `trace 块` 的可分离性，
**两个族集都量**：筛选集（6 族，旧）与**确认集**（6 族，先封后看）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
from probe_taiji_r2_separability_profile import _cells, _cos, _median  # noqa: E402

CONFIRMATION = PROJECT_ROOT / "plans" / "manifests" / "r2_separability_confirmation_families_20260923.json"

#: 只量这两级：`context` 是受改动影响的那级，`trace` 块是上游天花板
STAGES = ("2_trace_block", "3_context")


def _measure(substrate: Any, families: list[dict[str, Any]], tag: str) -> dict[str, float]:
    per_stage: dict[str, list[float]] = {stage: [] for stage in STAGES}
    for family in families:
        cells = _cells(substrate, None, family, tag)
        for stage in STAGES:
            c00, c10 = cells["00"][stage], cells["10"][stage]
            c01, c11 = cells["01"][stage], cells["11"][stage]
            per_stage[stage].append(_cos(c10 - c00, c11 - c01))
    return {stage: round(_median(values), 6) for stage, values in per_stage.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-name", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--measure-every", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=20260822)
    #: ⚠️ 必须显式给语料：`train_seed_corpus.DEFAULT_CORPUS` 是 `dialogue_extended_clean.jsonl`，
    #: 而 `seed_beta.pt` 的 `corpus_fingerprint` 写的是 **`simple_zh_texts.jsonl`**——两者不是同一个文件。
    #: 沿用训练器默认值会把两臂训在**另一份语料**上，"tick0 → 16M"的对比立刻作废。
    parser.add_argument("--corpus", default="data/simple_zh/simple_zh_texts.jsonl")
    parser.add_argument("--out-dir", required=True)
    #: 被外部杀掉后接着跑：从该 checkpoint 恢复，并按 `model.tick` **自动跳过**已消费的前缀
    #: （语料迭代是纯流式、不打乱，所以跳过是精确的）。不提供这个开关时，一次中断就白跑整臂。
    parser.add_argument("--resume-from", default=None)
    # 消融开关（默认全开 = 现行配方）
    parser.add_argument("--learn", choices=("on", "off"), default="on", help="总闸；off = 正对照臂（什么都不学）")
    parser.add_argument("--learn-fabric", choices=("on", "off"), default="on")
    parser.add_argument("--learn-motor", choices=("on", "off"), default="on")
    # 下面两个在**这条配方里**是空操作（守卫生效：`tests/taiji_native/test_t1t2_ablation_switches_bite.py`）
    parser.add_argument("--learn-readout", choices=("on", "off"), default="on")
    parser.add_argument("--use-memory", choices=("on", "off"), default="on")
    args = parser.parse_args()

    # 宁可**报错**也不让一条"以为自己消融了"的臂跑完 1M 符号：这类空操作臂会给出**假阴性**
    # （"关掉它结构却没变"），而且从读数上看不出来。
    if args.learn_readout == "off":
        parser.error(
            "--learn-readout off 在本配方里是空操作：readout=\"action\" 下 predictive_context/"
            "predictive_readout 根本不参与前向（tests/taiji_native/test_t1t2_ablation_switches_bite.py 钉住）"
        )
    if args.use_memory == "off":
        parser.error(
            "--use-memory off 在本配方里是空操作：这条配方不写 memory"
            "（同上守卫测试钉住）。要测情景侧得先换配方并另立预注册。"
        )

    from train_seed_corpus import iter_corpus_symbols

    from seed import Seed, SeedConfig
    from seed.persistence import atomic_save, attach_metadata
    from taiji import TaijiConfig

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    if (out_dir / "trajectory.jsonl").exists() and not args.resume_from:
        parser.error(f"{out_dir}/trajectory.jsonl already exists; 不覆写（换目录、换臂名，或用 --resume-from 接着跑）")
    out_dir.mkdir(parents=True, exist_ok=True)

    confirm_families = json.loads(CONFIRMATION.read_text(encoding="utf-8"))["families"]
    values = dict(torch.load(PROJECT_ROOT / "checkpoints" / "seed_beta.pt", map_location="cpu", weights_only=False)["config"]["taiji"])
    values.pop("receptors_factored", None)
    config = SeedConfig(taiji=TaijiConfig.from_dict({**values, "seed": args.seed}))

    # 消融：`learn=False` 是总闸，`None` 表示跟随总闸。关掉某一项就显式传 False。
    observe_kwargs: dict[str, Any] = {"learn": args.learn == "on"}
    if args.learn_fabric == "off":
        observe_kwargs["learn_fabric"] = False
    if args.learn_motor == "off":
        observe_kwargs["learn_motor"] = False
    if args.learn_readout == "off":
        observe_kwargs["learn_predictive_context"] = False
        observe_kwargs["learn_predictive_readout"] = False
    if args.use_memory == "off":
        observe_kwargs["use_memory"] = False
        observe_kwargs["use_identity"] = False

    model = Seed(config, episode_id=f"t1t2-{args.arm_name}")
    skip = 0
    if args.resume_from:
        model.restore(torch.load(args.resume_from, map_location="cpu", weights_only=False))
        skip = int(model.tick)
    boundary = config.taiji.boundary_symbol
    corpus_paths = [str(PROJECT_ROOT / args.corpus)]

    trajectory = out_dir / "trajectory.jsonl"
    started = time.perf_counter()
    ticks = skip  # 续跑时从恢复点的 tick 起算，进度与测量节奏才对得上
    window_ticks = 0
    window_correct = 0
    window_surprise = 0.0
    measurements: list[dict[str, Any]] = []

    def _measure_now(step: int) -> dict[str, Any]:
        # 在**checkpoint 往返出来的副本**上测，不碰正在训练的那个模型：
        # ① 测量走的是 predictive episode，直接在同一实例上测会与训练循环的 readout 撞守卫
        #    （实测报 `readout changed inside an active dynamics episode`）；
        # ② 即使重置能绕开，也会在每个测量点给训练插一个 episode 边界，凭空扰动被比较的两臂。
        # 用副本 ⇒ 训练流一点不动，且顺带每次都走一遍存档往返。
        probe = Seed.from_checkpoint(model.checkpoint())
        screen = _measure(probe.architecture, FAMILIES, f"{args.arm_name}-s{step}")
        confirm = _measure(probe.architecture, confirm_families, f"{args.arm_name}-c{step}")
        row = {
            "arm": args.arm_name,
            "step": step,
            "ticks": ticks,
            "screen": screen,
            "confirmation": confirm,
            "elapsed_seconds": round(time.perf_counter() - started, 2),
        }
        measurements.append(row)
        with trajectory.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)
        return row

    _measure_now(ticks)  # 第 0 点：这一臂自己的初始化（续跑时是恢复点的读数）
    stream = iter_corpus_symbols(corpus_paths, boundary=boundary)
    for _ in range(skip):  # 接着跑：精确跳过已消费的前缀（流式、不打乱）
        next(stream)
    for symbol in stream:
        # ⚠️ 必须直接调 substrate：`Seed.observe` **只转发** learn / learn_motor / use_memory /
        # use_identity（`seed/model.py:78`），四个消融开关传进去会直接 TypeError（已实测）。
        # 这里显式给 `readout="action"`，与 `Seed.observe`（不转发 readout ⇒ 用 substrate 默认）
        # 以及训练器保持逐字一致。
        step = model.substrate.observe(symbol, readout="action", **observe_kwargs)
        ticks += 1
        if step.prior_prediction is not None:
            window_ticks += 1
            window_correct += int(step.prior_prediction == symbol)
            window_surprise += float(step.surprise)
        if ticks % args.measure_every == 0:
            _measure_now(ticks)
            atomic_save(
                attach_metadata(
                    model.checkpoint(),
                    tick=model.tick,
                    corpus_fingerprint=json.dumps([{"name": Path(p).name} for p in corpus_paths]),
                    extra={"trainer": "run_taiji_r2_t1_t2_short_runs", "arm": args.arm_name},
                ),
                out_dir / f"checkpoint_{ticks}.pt",
            )
            # §5.3：每臂的在线准确率/惊讶度必须一起报，让"是不是根本没学"可见。
            # **必须落盘**，不能只 print：首轮只打 stdout，结果三臂的准确率只能从后台任务缓冲里捞。
            progress = {
                "arm": args.arm_name,
                "step": ticks,
                "online_accuracy": round(window_correct / max(1, window_ticks), 4),
                "mean_surprise": round(window_surprise / max(1, window_ticks), 4),
                "elapsed_seconds": round(time.perf_counter() - started, 2),
            }
            with trajectory.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(progress, ensure_ascii=False) + "\n")
            print(json.dumps(progress, ensure_ascii=False), flush=True)
            window_ticks = 0
            window_correct = 0
            window_surprise = 0.0
        if ticks >= args.budget:
            break

    summary = {
        "format": "taiji-r2-t1t2-trajectory-v1",
        "arm": args.arm_name,
        "written_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "prereg": "plans/reference/M5_R2_T1_T2_PREREG_20260923.md",
        "budget": args.budget,
        "observe_kwargs": observe_kwargs,
        "measurements": measurements,
        "wall_seconds": round(time.perf_counter() - started, 2),
        "confirmation_families_sha256_file": str(CONFIRMATION),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"arm": args.arm_name, "done": True, "ticks": ticks}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
