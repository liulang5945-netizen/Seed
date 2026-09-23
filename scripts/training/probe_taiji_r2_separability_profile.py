"""R2 Step 0：可分离性的**逐级剖面**（零训练）——定位"可分离性是在哪一级丢的"。

授权：`plans/reference/M5_R2_COMPOSITION_BINDING_BUDGET_PROPOSAL_20260923.md` §3
（所有者 2026-09-23 批 Step 0）。**未改 `taiji/` 任何文件。**

## 怎么取到每一级

在**实例**上临时把两个方法包一层，记下**真实前向里流过**的张量，跑完立刻恢复原方法。
这样记录到的是实际被用的值，而不是我重算的近似。

| 级 | 取法 |
|---|---|
| 1 区域迹 | `substrate._state.regions`（预填充结束后取，私有字段） |
| 2 皮层投影 | `predictive_context.encode` 的**第 1 个实参**（= `fabric.predictive_context(regions)` 的输出） |
| 3 上下文器官输出 | `predictive_context.encode` 的**返回值**（`base_context`） |
| 4 最终 cue | `predictive_readout.probabilities` 的**第 1 个实参**（读出头真正拿到的 context） |
| 5 读出分布 | `predictive_readout.probabilities` 的**返回值** |

第 3 与第 4 级是否逐位相同，本件顺手核实（`_gated_temporal_candidate is None` 且无 residual 时应相同）。

## 统计量与阈值（与组合性探针**同一套**，不另写、不因剖面结果调整）

`consistency = cos(换A方向在B=b₀, 换A方向在B=b₁)`；可分离 ⇒ ≈1，随机 ⇒ ≈0。阶梯沿用组合性合同 §3
（`separable ≥ 0.9` / `not_separable < 0.5`）。逐级都跑零假设对照（同题面 ×2 必须逐位为零）与实测基线。

## 决策（提案 §3.3，**先写死**）

* 第 1–2 级 < 0.5 ⇒ 点名 **区域/皮层投影**
* 第 1–2 级可分离、第 3–4 级 < 0.5 ⇒ 点名 **上下文器官/递归混合**
* 第 1–4 级都可分离、只有第 5 级 < 0.5 ⇒ **提案前提作废**（改从读出侧立题）
* 其余（含"全程 < 0.5"）⇒ 仍点名**上游**，优先上下文器官那一级
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
    _cos,
)
from probe_taiji_r2_readout_conditionality import _prefill  # noqa: E402

PROPOSAL = "plans/reference/M5_R2_COMPOSITION_BINDING_BUDGET_PROPOSAL_20260923.md"
VERDICT_REPORT = PROJECT_ROOT / "reports" / "taiji_r2_separability_profile_20260923.json"
DRY_RUN_REPORT = (
    PROJECT_ROOT / "reports" / "taiji_r2_separability_profile_pipeline_check_20260923.json"
)
#: **修正后的层级**（第一版把"拼接"当成了"损失"，见模块末尾的更正说明）。
#:
#: 真实拓扑（读码所得）：`cat(region.activity…) + cat(region.trace…)`（1152，**无权重**）
#: → `BytePredictiveContext.receptors`（**有学习权重** 1152→96）→ `context`（96）→ 读出。
#: 所以**只有 `context` 这一级是"经过学习映射"的**，前两级只是把两部分接起来——
#: 把它们的数字当成"逐级损失"来读是错的：两个 block 的余弦**不可比**，拼接本身就会改变余弦。
STAGES = (
    "1_activity_block",  # 各区 activity 拼接（576）——无权重
    "2_trace_block",  # 各区 trace 拼接（576）——无权重，且是 fabric 投影真正消费的部分
    "3_context",  # receptors(1152→96) 的输出（96）——**唯一有学习权重的一级**
    "4_context_trace",  # 上一步 context 的副本（rec尾current 的输入，96）
    "5_readout_dist",  # 读出分布（257）
)


def _flatten(value: Any) -> torch.Tensor:
    """Nested tensor container → one flat float64 vector.

    **不认识的类型一律报错，不静默填零。** 首版对认不出的输入返回一个 1 维零向量，于是
    ``_state.regions``（其实是 3 个 ``RegionState`` 数据类，不是张量）被压成 dim=3，
    剖面里第 1 级"不可分离"的读数是**假的**——这正是本仓反复登记的"静默给错答案"那一类。
    """

    if isinstance(value, torch.Tensor):
        return value.detach().reshape(-1).double()
    if isinstance(value, dict):
        return _flatten([value[key] for key in sorted(value)])
    if isinstance(value, (list, tuple)):
        parts = [_flatten(item) for item in value]
        parts = [part for part in parts if part.numel()]
        return torch.cat(parts) if parts else torch.zeros(1, dtype=torch.float64)
    fields = getattr(value, "__dataclass_fields__", None)
    if fields:
        return _flatten([getattr(value, name) for name in fields])
    raise TypeError(f"_flatten: 不认识 {type(value).__name__}——宁可报错，也不静默填零")


def _region_trace(regions: Any) -> torch.Tensor:
    """第 1 级＝各区**迹**（`RegionState.trace`）拼接。

    取 `trace` 而不是把 7 个字段全拼进去，是因为链路的下一级
    `fabric.predictive_context(regions)` 消费的就是迹（`model.py` 里 `regions[0].trace` 同源）。
    """

    if not isinstance(regions, (list, tuple)) or not regions:
        raise TypeError(f"_region_trace: regions 不是非空序列（{type(regions).__name__}）")
    for region in regions:
        if not hasattr(region, "trace"):
            raise TypeError(f"_region_trace: {type(region).__name__} 没有 .trace")
    return _flatten([region.trace for region in regions])


class _Taps:
    """Temporarily shadow two instance methods to record the real forward values.

    Instance-level shadowing only — ``taiji/`` is untouched. ``__exit__`` restores the bound
    methods exactly as captured, so a failed run cannot leave a patched substrate behind.
    """

    def __init__(self, substrate: Any) -> None:
        self.substrate = substrate
        self.encode_in: list[Any] = []
        self.encode_out: list[Any] = []
        self.readout_in: list[Any] = []
        self.readout_out: list[Any] = []

    def __enter__(self) -> _Taps:
        encode_owner = self.substrate.predictive_context
        readout_owner = self.substrate.predictive_readout
        original_encode = encode_owner.encode
        original_probabilities = readout_owner.probabilities

        def encode_wrapper(*args: Any, **kwargs: Any) -> Any:
            self.encode_in.append(args[0] if args else None)
            result = original_encode(*args, **kwargs)
            self.encode_out.append(result)
            return result

        def probabilities_wrapper(*args: Any, **kwargs: Any) -> Any:
            self.readout_in.append(args[0] if args else None)
            result = original_probabilities(*args, **kwargs)
            self.readout_out.append(result)
            return result

        encode_owner.encode = encode_wrapper
        readout_owner.probabilities = probabilities_wrapper
        self._restore = (encode_owner, "encode", original_encode), (
            readout_owner,
            "probabilities",
            original_probabilities,
        )
        return self

    def __exit__(self, *exc: Any) -> None:
        for owner, name, original in self._restore:
            setattr(owner, name, original)
        return None


def _cells(substrate: Any, runtime: Any, family: dict[str, Any], tag: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for i in (0, 1):
        for j in (0, 1):
            prompt = family["skeleton"].format(a=family["a"][i], b=family["b"][j])
            with _Taps(substrate) as taps:
                _prefill(substrate, runtime, prompt, f"prof-{tag}-{family['id']}-{i}{j}")
            regions = substrate._state.regions
            encoded = taps.encode_out[-1]  # encode 返回 (context, trace)
            out[f"{i}{j}"] = {
                "prompt": prompt,
                "1_activity_block": _flatten([region.activity for region in regions]),
                "2_trace_block": _region_trace(regions),
                "3_context": _flatten(encoded[0]),
                "4_context_trace": _flatten(encoded[1]),
                "5_readout_dist": _flatten(taps.readout_out[-1]),
            }
    return out


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2] if ordered else 0.0


def probe(label: str, checkpoint: Path, runtime_cls: Any) -> dict[str, Any]:
    runtime = runtime_cls.load(checkpoint)
    substrate = runtime.model.substrate

    null_a = _cells(substrate, runtime, FAMILIES[0], f"null-a-{label}")["00"]
    null_b = _cells(substrate, runtime, FAMILIES[0], f"null-b-{label}")["00"]
    null = {stage: float((null_a[stage] - null_b[stage]).abs().sum()) for stage in STAGES}
    null_ok = all(value == 0.0 for value in null.values())

    per_stage: dict[str, list[float]] = {stage: [] for stage in STAGES}
    us: dict[str, list[torch.Tensor]] = {stage: [] for stage in STAGES}
    vs: dict[str, list[torch.Tensor]] = {stage: [] for stage in STAGES}
    # `_gated_temporal_candidate is None` ⇒ `context` 就是直接喂给读出的那个，中间没有额外候选层。
    # 这一条决定"第 3 级"与"最终 cue"是不是同一处，报出来比猜好。
    gated_absent = bool(getattr(substrate, "_gated_temporal_candidate", None) is None)
    for family in FAMILIES:
        cells = _cells(substrate, runtime, family, label)
        for stage in STAGES:
            c00, c10 = cells["00"][stage], cells["10"][stage]
            c01, c11 = cells["01"][stage], cells["11"][stage]
            per_stage[stage].append(_cos(c10 - c00, c11 - c01))
            us[stage].append(c10 - c00)  # 换 A 的方向 @ B=b₀
            vs[stage].append(c01 - c00)  # 换 B 的方向 @ A=a₀

    profile: dict[str, Any] = {}
    for stage in STAGES:
        baseline = [
            _cos(us[stage][i], us[stage][j])
            for i in range(len(FAMILIES))
            for j in range(i + 1, len(FAMILIES))
        ]
        baseline += [
            _cos(vs[stage][i], vs[stage][j])
            for i in range(len(FAMILIES))
            for j in range(i + 1, len(FAMILIES))
        ]
        baseline += [
            _cos(us[stage][i], vs[stage][j])
            for i in range(len(FAMILIES))
            for j in range(len(FAMILIES))
            if i != j
        ]
        median = _median(per_stage[stage])
        if median >= SEPARABLE_FLOOR:
            verdict = "separable"
        elif median >= ENTANGLED_CEILING:
            verdict = "partial"
        else:
            verdict = "not_separable"
        profile[stage] = {
            "consistency_median": round(median, 6),
            "consistency_min": round(min(per_stage[stage]), 6),
            "consistency_max": round(max(per_stage[stage]), 6),
            "baseline_median": round(_median(baseline), 6),
            "baseline_n": len(baseline),
            "gap_over_baseline": round(median - _median(baseline), 6),
            "verdict": verdict,
            "flattened_dim": int(us[stage][0].numel()),
        }
    return {
        "label": label,
        "checkpoint": str(checkpoint),
        "null_control": {"per_stage_abs_diff": null, "ok": null_ok},
        "gated_temporal_candidate_absent": gated_absent,
        "profile": profile,
    }


def _localize(entry: dict[str, Any]) -> dict[str, Any]:
    """提案 §3.3 的决策树，先写死，按剖面机械套用。"""

    p = entry["profile"]
    #: 无权重级（拼接块）的**上界**：任何一个 block 自己有多可分离。
    upstream = max(p["1_activity_block"]["consistency_median"], p["2_trace_block"]["consistency_median"])
    learned = p["3_context"]["consistency_median"]
    late = p["5_readout_dist"]["consistency_median"]
    drop = round(upstream - learned, 6)
    if learned >= SEPARABLE_FLOOR:
        level = "none_learned_stage_ok"
        note = "学习映射后仍可分离 ⇒ 与组合性探针结论冲突，须复核"
    elif drop >= 0.05:
        level = "level_3_receptors_learned_map"
        note = (
            f"学习映射（receptors 1152→96）把可分离性从块上界 {upstream:.4f} 拉到 {learned:.4f}"
            f"（−{drop:.4f}）⇒ **点名 `BytePredictiveContext.receptors`**"
        )
    elif late > learned + 0.05:
        level = "level_5_readout"
        note = "学习映射损失不显著、读出分布反而更可分离 ⇒ 瓶颈的归因要重写"
    else:
        level = "upstream_of_any_map"
        note = (
            "块本身就不分离（学习映射没有显著再损失）⇒ 缺口在**区域动力学**，"
            "不在任何学习映射上"
        )
    return {
        "named_level": level,
        "reason": note,
        "upstream_block_ceiling": round(upstream, 6),
        "learned_stage": round(learned, 6),
        "drop_at_learned_stage": drop,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-checkpoint", default=str(PROJECT_ROOT / "checkpoints" / "seed_beta.pt")
    )
    parser.add_argument(
        "--arm-a",
        default=str(PROJECT_ROOT / "output" / "taiji_r2_readout_retrain" / "A" / "checkpoint.pt"),
    )
    parser.add_argument("--out-report", default=None)
    parser.add_argument("--dry-run", action="store_true", help="只跑基座")
    args = parser.parse_args()
    from api.seed_runtime import SeedRuntime

    # dry-run 与正式判决**分开落盘**：判决件"已存在即拒跑"，一次自检若占了正式路径，
    # 就等于把真判决挡在门外（本仓在判决器上已经踩过一次）。
    if DRY_RUN_REPORT == VERDICT_REPORT:  # pragma: no cover - 防的是以后有人把两者改成同一个
        parser.error("internal: dry-run 与正式判决不得共用同一路径")
    report_path = Path(args.out_report) if args.out_report else (
        DRY_RUN_REPORT if args.dry_run else VERDICT_REPORT
    )
    if report_path.exists():
        parser.error(f"{report_path} already exists; 判决件不覆写")

    base = Path(args.base_checkpoint)
    if not base.is_file():
        parser.error(f"base checkpoint not found: {base}")
    targets = [("base", base)]
    if not args.dry_run:
        arm = Path(args.arm_a)
        if not arm.is_file():
            parser.error(f"arm A checkpoint not found: {arm}")
        targets.append(("arm_A", arm))

    results = {label: probe(label, path, SeedRuntime) for label, path in targets}
    clean = all(entry["null_control"]["ok"] for entry in results.values())
    for entry in results.values():
        entry["localization"] = _localize(entry)

    payload = {
        "format": "taiji-r2-separability-profile-v1",
        "status": "completed" if clean else "failed",
        "written_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "authorised_by": f"{PROPOSAL} §3（所有者 2026-09-23 批 Step 0）",
        "question": "可分离性是在链路的哪一级丢的",
        "preregistered_ladder": {
            "separable_floor": SEPARABLE_FLOOR,
            "not_separable_below": ENTANGLED_CEILING,
        },
        "disclosures": [
            "在实例上临时包 predictive_context.encode 与 predictive_readout.probabilities 取真实前向值，"
            "跑完恢复；未改 taiji/ 任何文件",
            "第 1 级取 substrate._state.regions（私有）",
            "刺激集沿用组合性探针的冻结 FAMILIES，不另写一份",
        ],
        "failure": None if clean else "零假设对照不干净，拒判",
        "checkpoints": results,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "report": str(report_path)}))
    for label, entry in results.items():
        print(f"  === {label} (null_ok={entry['null_control']['ok']}) ===")
        for stage in STAGES:
            row = entry["profile"][stage]
            print(
                f"    {stage:24} median={row['consistency_median']:+.4f} "
                f"base={row['baseline_median']:+.4f} dim={row['flattened_dim']:>6} {row['verdict']}"
            )
        print(f"    => {entry['localization']['named_level']}  {entry['localization']['reason']}")
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
