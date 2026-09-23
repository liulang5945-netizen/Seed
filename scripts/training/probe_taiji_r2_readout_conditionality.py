"""R2 P3：读出器的条件性（"内容有没有进到读出器"）——零训练探针。

所属：`plans/reference/M5_R2_SURFACE_DECODING_CONTRACT_DRAFT_20260920.md` §2 的 **P3**。
该条判据早已冻结，但 P3 一直被 **P0 前置门**挡着，理由是**"会给一个没训过的读出打分"**：
P0-a′ 查明那份基座的读出面（`predictive_readout`）在 M1 的 legacy 训练路径里**从未被写入**。
2026-09-23 起这个前提消失了——受控重训的 **A 臂写入面实测为 `['predictive_readout']`**，
读出头**第一次是真被训过的**。所以 P3 现在有了有意义的对象。

## 探针做什么

对每一对题面（**同骨架、只换一个内容词**）分别跑一遍预填充，取两件东西：

* **cue** = 读出器输入侧表示（`Taiji.observe` 里喂给 `predictive_readout.probabilities(context, …)`
  的那个 `context`）。它不在公开的 `TaijiStep` 上，只能从 `substrate._state.motor_context` 取
  ——**这是私有的**，本件如实披露：公开接口没有暴露 cue，而 P3 的判据明确要求测 cue。
* **readout 分布** = `step.probabilities`（257 维，`softmax(synapses·context + bias)`）。

## 判据（沿用已 FROZEN 的 P3 定义，阈值先写死在这里）

* **C1（内容进场）**：cue **随提问改变** **且** 读出分布 **KL ≥ 0.01**。两条都要。
  - "cue 随提问改变" 的操作化（本件补写，执行前定）：`cue_l1_relative = ||a−b||₁ / (||a||₁+||b||₁)`
    在**至少 90% 的最小对**上 **≥ 1e-3**。补这条的理由与 K2 同族：原定义只写了"改变"，没给判读线。
    1e-3 的正当性不靠拍——见下面的**零假设对照**，同一条题面跑两次必须给出**逐位相同**的 cue
    （差为 0），所以任何非零差都不是浮点噪声；1e-3 只是把"可忽略的极小差"挡在外面。
* **C2（内容未进场）**：cue 或读出分布对提问**无稳定差异** ⇒ 缺口在「问题→读出」的接线。
* 两条都不满足时如实记 **inconclusive**，不硬套结论。

## 零假设对照（仪器的可用性前置）

同一条题面在**两个全新 episode** 里各跑一遍：cue 必须**逐位相同**、KL 必须**恰好 0**。
不满足 ⇒ 这个探针有噪声底，**整件拒跑**（否则下面所有非零差都可能是噪声）。

## 无关对照（特异性）

另跑一组"无关联对"（骨架也不同的两条题面），给出差异的**上界参考**。
若最小对的差异与无关联对**同量级**，那"cue 随内容改变"就退化成"任何输入都让它变"，
可归因性要打折——报告里必须一起给这三组数。
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

from diag_taiji_r2_surface_decode import _serialize_prompt  # noqa: E402

CONTRACT = "plans/reference/M5_R2_SURFACE_DECODING_CONTRACT_DRAFT_20260920.md"
KL_THRESHOLD = 0.01  # 冻结合同 §2 P3 给的值（自然对数口径）
CUE_L1_RELATIVE_FLOOR = 1e-3
CUE_CHANGE_RATE_FLOOR = 0.9

#: 冻死的刺激集。**同骨架、只换一个内容词**是最小对；`unrelated` 是特异性对照。
#: 本表由我撰写（合同只规定"同骨架两题"，未给来源）——**这是本件的一处已披露弱点**：
#: 表在打分前随本文件一起提交，之后不得增删改。
PROBE_SET: list[dict[str, str]] = [
    {"kind": "minimal", "skeleton": "我想知道{s}是什么", "a": "猫", "b": "狗"},
    {"kind": "minimal", "skeleton": "我想知道{s}是什么", "a": "苹果", "b": "香蕉"},
    {"kind": "minimal", "skeleton": "我想知道{s}是什么", "a": "北京", "b": "上海"},
    {"kind": "minimal", "skeleton": "我想知道{s}是什么", "a": "水", "b": "火"},
    {"kind": "minimal", "skeleton": "{s}这个词是什么意思", "a": "场所", "b": "时间"},
    {"kind": "minimal", "skeleton": "{s}这个词是什么意思", "a": "语言", "b": "文字"},
    {"kind": "minimal", "skeleton": "{s}这个词是什么意思", "a": "音乐", "b": "美术"},
    {"kind": "minimal", "skeleton": "请解释一下{s}", "a": "学习", "b": "工作"},
    {"kind": "minimal", "skeleton": "请解释一下{s}", "a": "春天", "b": "冬天"},
    {"kind": "minimal", "skeleton": "他不喜欢{s}", "a": "下雨", "b": "刮风"},
    {"kind": "minimal", "skeleton": "他不喜欢{s}", "a": "早起", "b": "熬夜"},
    {"kind": "minimal", "skeleton": "这本书讲的是{s}", "a": "历史", "b": "数学"},
    {"kind": "unrelated", "skeleton": None, "a": "我想知道猫是什么", "b": "他不喜欢早起"},
    {"kind": "unrelated", "skeleton": None, "a": "请解释一下学习", "b": "这本书讲的是历史"},
    {"kind": "unrelated", "skeleton": None, "a": "场所这个词是什么意思", "b": "我想知道苹果是什么"},
    {"kind": "unrelated", "skeleton": None, "a": "他不喜欢刮风", "b": "请解释一下冬天"},
]


def _prompts(item: dict[str, str]) -> tuple[str, str]:
    if item.get("skeleton"):
        return item["skeleton"].format(s=item["a"]), item["skeleton"].format(s=item["b"])
    return item["a"], item["b"]


def _prefill(substrate: Any, runtime: Any, prompt: str, episode: str) -> dict[str, Any]:
    """Prefill one prompt in a fresh episode and return the readout's cue and distribution."""

    substrate.reset_dynamics(episode_id=episode)
    step = substrate.observe(
        substrate.config.boundary_symbol,
        learn=False,
        readout="predictive",
        use_memory=False,
        use_identity=False,
    )
    # ``_serialize_prompt`` 返回的就是 bytes（P1/P2 的 generate 也按 bytes 用）
    for symbol in _serialize_prompt(runtime, prompt):
        step = substrate.observe(
            int(symbol), learn=False, readout="predictive", use_memory=False, use_identity=False
        )
    # cue 只在私有 state 上（公开 step 没暴露它）——本件的披露点
    cue = substrate._state.motor_context.detach().clone()
    return {"cue": cue, "probabilities": step.probabilities.detach().clone()}


def _cue_l1_relative(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = float(a.abs().sum() + b.abs().sum())
    if denom == 0.0:
        return 0.0
    return float((a - b).abs().sum()) / denom


def _kl_nats(a: torch.Tensor, b: torch.Tensor) -> float:
    eps = 1e-12
    pa = a.clamp_min(eps)
    pb = b.clamp_min(eps)
    return float((pa * (pa / pb).log()).sum())


def probe_checkpoint(label: str, checkpoint: Path, runtime_cls: Any) -> dict[str, Any]:
    runtime = runtime_cls.load(checkpoint)
    substrate = runtime.model.substrate

    # ① 零假设对照：同一条题面跑两次，必须逐位相同
    first = _prefill(substrate, runtime, "我想知道猫是什么", f"p3-null-a-{label}")
    second = _prefill(substrate, runtime, "我想知道猫是什么", f"p3-null-b-{label}")
    null_cue_diff = float((first["cue"] - second["cue"]).abs().sum())
    null_kl = _kl_nats(first["probabilities"], second["probabilities"])
    null_ok = null_cue_diff == 0.0 and null_kl == 0.0

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, item in enumerate(PROBE_SET):
        prompt_a, prompt_b = _prompts(item)
        left = _prefill(substrate, runtime, prompt_a, f"p3-{label}-{index}-a")
        right = _prefill(substrate, runtime, prompt_b, f"p3-{label}-{index}-b")
        rows.append(
            {
                "kind": item["kind"],
                "prompt_a": prompt_a,
                "prompt_b": prompt_b,
                "cue_l1_relative": round(_cue_l1_relative(left["cue"], right["cue"]), 10),
                "kl_nats": round(_kl_nats(left["probabilities"], right["probabilities"]), 10),
            }
        )
    minimal = [r for r in rows if r["kind"] == "minimal"]
    unrelated = [r for r in rows if r["kind"] == "unrelated"]

    def _median(values: list[float]) -> float:
        ordered = sorted(values)
        return ordered[len(ordered) // 2] if ordered else 0.0

    minimal_kl = [r["kl_nats"] for r in minimal]
    minimal_cue = [r["cue_l1_relative"] for r in minimal]
    changed = [c for c in minimal_cue if c >= CUE_L1_RELATIVE_FLOOR]
    cue_changes = bool(minimal_cue and len(changed) / len(minimal_cue) >= CUE_CHANGE_RATE_FLOOR)
    kl_holds = _median(minimal_kl) >= KL_THRESHOLD
    c1 = bool(cue_changes and kl_holds)
    return {
        "label": label,
        "checkpoint": str(checkpoint),
        "null_control": {
            "cue_l1_diff": null_cue_diff,
            "kl_nats": null_kl,
            "ok": null_ok,
        },
        "rows": rows,
        "minimal": {
            "n": len(minimal),
            "cue_l1_relative_median": round(_median(minimal_cue), 10),
            "cue_l1_relative_max": round(max(minimal_cue), 10),
            "cue_change_rate": round(len(changed) / len(minimal_cue), 4) if minimal_cue else 0.0,
            "kl_nats_median": round(_median(minimal_kl), 10),
            "kl_nats_max": round(max(minimal_kl), 10),
            "kl_nats_min": round(min(minimal_kl), 10),
        },
        "unrelated_control": {
            "n": len(unrelated),
            "cue_l1_relative_median": round(_median([r["cue_l1_relative"] for r in unrelated]), 10),
            "kl_nats_median": round(_median([r["kl_nats"] for r in unrelated]), 10),
        },
        "criterion": {
            "cue_changes": cue_changes,
            "kl_median_ge_threshold": kl_holds,
            "kl_threshold": KL_THRESHOLD,
            "cue_l1_relative_floor": CUE_L1_RELATIVE_FLOOR,
            "cue_change_rate_floor": CUE_CHANGE_RATE_FLOOR,
            "C1_content_enters": c1,
            "C2_content_absent": bool(not c1 and null_ok),
        },
        "seconds": round(time.perf_counter() - started, 2),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", default=str(PROJECT_ROOT / "checkpoints" / "seed_beta.pt"))
    parser.add_argument("--arm-a", default=str(PROJECT_ROOT / "output" / "taiji_r2_readout_retrain" / "A" / "checkpoint.pt"))
    parser.add_argument("--out-report", default=None)
    parser.add_argument("--dry-run", action="store_true", help="只跑基座，验证管线")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    from api.seed_runtime import SeedRuntime

    report_path = Path(args.out_report or PROJECT_ROOT / "reports" / "taiji_r2_p3_readout_conditionality_20260923.json")
    if report_path.exists():
        parser.error(f"{report_path} already exists; 判决件不覆写")

    base = Path(args.base_checkpoint)
    arm_a = Path(args.arm_a)
    if not base.is_file():
        parser.error(f"base checkpoint not found: {base}")
    targets = [("base", base)]
    if not args.dry_run:
        if not arm_a.is_file():
            parser.error(f"arm A checkpoint not found: {arm_a}（缺臂则 P3 没有对象，拒跑）")
        targets.append(("arm_A", arm_a))

    results = {}
    for label, path in targets:
        results[label] = probe_checkpoint(label, path, SeedRuntime)

    if not all(entry["null_control"]["ok"] for entry in results.values()):
        # 噪声底不为零 ⇒ 所有非零差都可能只是噪声，整件不成立
        payload_status = "failed"
        failure = "零假设对照不干净（同一题面两次给出不同的 cue 或 KL≠0），探针有噪声底，拒判"
    else:
        payload_status = "completed"
        failure = None
    payload = {
        "format": "taiji-r2-p3-readout-conditionality-v1",
        "status": payload_status,
        "written_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "contract": CONTRACT,
        "probe": "P3 读出器的条件性（cue 是否随提问改变 + 读出分布 KL）",
        "preregistered_thresholds": {
            "kl_nats_threshold": KL_THRESHOLD,
            "cue_l1_relative_floor": CUE_L1_RELATIVE_FLOOR,
            "cue_change_rate_floor": CUE_CHANGE_RATE_FLOOR,
            "cue_change_operationalisation_written_by": "本件（合同只写了'改变'，没给判读线）",
        },
        "disclosures": [
            "cue 取自 substrate._state.motor_context（私有）——公开 step 没暴露 cue，而判据要求测 cue",
            "刺激集由我撰写（合同只规定'同骨架两题'）；表随执行件在打分前提交，之后不得改",
        ],
        "failure": failure,
        "checkpoints": results,
        "reading": (
            "C1 ⇒ 内容确实进到了读出器，缺口在更下游（表征/组合绑定）"
            if all(entry["criterion"]["C1_content_enters"] for entry in results.values())
            else "见各 checkpoint 的 criterion；C2 ⇒ 缺口在「问题→读出」的接线"
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload_status, "report": str(report_path)}))
    for label, entry in results.items():
        print(
            f"  {label}: null_ok={entry['null_control']['ok']} "
            f"C1={entry['criterion']['C1_content_enters']} "
            f"kl_median={entry['minimal']['kl_nats_median']} "
            f"cue_rel_median={entry['minimal']['cue_l1_relative_median']}"
        )
    return 0 if payload_status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
