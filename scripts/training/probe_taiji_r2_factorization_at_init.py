"""R2 Step 1 的**快速否证**：分块受力面在**初始化**时到底有没有用（零训练、秒级）。

## 为什么这一读有效

`BytePredictiveContext.receptors` 是**固定的随机稀疏压缩**——连接表构造时抽定，之后没有任何路径能改它；
`recurrent` 在本基座里 abs_sum 为 0。所以 `context = bound_norm(receptors(cat(activity, trace)))` 里，
**唯一由这次改动决定的那一步，在训练开始前就已经是最终形态**。

⇒ 拿两套配置（同一架构、同一 seed，只有开关不同）在初始化状态下跑同一批冻结刺激，
直接量最终 cue 的可分离性：

* **若分块在初始化上就没把可分离性抬起来** ⇒ 这一处不训练、长跑也改不动它，
  16M×2 的长跑**几乎不可能**给出机制上的改善 —— **可以据此省掉整轮长跑**。
* **若明显抬起来了** ⇒ 是正向信号，但**还不算证据**（上游区域动力学也在训），长跑仍要做。

本件只做前者：**它能否证，不能确认。** 判读线沿用组合性合同 §3 的先注册阶梯（0.9 / 0.5），不新拍。

## 怎么复用已有仪器

测量路径**直接用** Step 0 剖面仪器里的 `_cells`（在**实例**上临时包
`predictive_context.encode` 与 `predictive_readout.probabilities` 取真实前向值），
以及组合性探针的冻结 `FAMILIES`。这里不重写任何测量代码。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
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

PROPOSAL = "plans/reference/M5_R2_COMPOSITION_BINDING_BUDGET_PROPOSAL_20260923.md"


def measure(label: str, model: Any) -> dict[str, Any]:
    substrate = model
    per_stage: dict[str, list[float]] = {stage: [] for stage in STAGES}
    us: dict[str, list[torch.Tensor]] = {stage: [] for stage in STAGES}
    vs: dict[str, list[torch.Tensor]] = {stage: [] for stage in STAGES}
    for family in FAMILIES:
        cells = _cells(substrate, None, family, f"init-{label}")
        for stage in STAGES:
            c00, c10 = cells["00"][stage], cells["10"][stage]
            c01, c11 = cells["01"][stage], cells["11"][stage]
            per_stage[stage].append(_cos(c10 - c00, c11 - c01))
            us[stage].append(c10 - c00)
            vs[stage].append(c01 - c00)

    profile: dict[str, Any] = {}
    for stage in STAGES:
        baseline = [
            _cos(us[stage][i], us[stage][j])
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
        verdict = (
            "separable"
            if median >= SEPARABLE_FLOOR
            else ("partial" if median >= ENTANGLED_CEILING else "not_separable")
        )
        profile[stage] = {
            "consistency_median": round(median, 6),
            "baseline_median": round(_median(baseline), 6),
            "gap_over_baseline": round(median - _median(baseline), 6),
            "verdict": verdict,
        }
    return {"label": label, "profile": profile}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-from",
        default=str(PROJECT_ROOT / "checkpoints" / "seed_beta.pt"),
        help="从这份 checkpoint 的 config 取架构与 seed（保证两套配置只差开关）",
    )
    parser.add_argument("--out-report", default=None)
    args = parser.parse_args()
    from taiji import Taiji, TaijiConfig

    env = torch.load(Path(args.config_from), map_location="cpu", weights_only=False)
    values = dict(env["config"]["taiji"])
    values["receptors_factored"] = False
    plain = Taiji(TaijiConfig.from_dict(values))
    values["receptors_factored"] = True
    factored = Taiji(TaijiConfig.from_dict(values))

    results = {"off_at_init": measure("off", plain), "on_at_init": measure("on", factored)}
    report_path = Path(
        args.out_report or PROJECT_ROOT / "reports" / "taiji_r2_factorization_at_init_20260923.json"
    )
    if report_path.exists():
        parser.error(f"{report_path} already exists; 判决件不覆写")

    verdict: dict[str, Any] = {}
    for stage in STAGES:
        left = results["off_at_init"]["profile"][stage]["consistency_median"]
        right = results["on_at_init"]["profile"][stage]["consistency_median"]
        verdict[stage] = {"off": left, "on": right, "delta": round(right - left, 6)}
    payload = {
        "format": "taiji-r2-factorization-at-init-v1",
        "written_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "contract": PROPOSAL,
        "question": "分块受力面在**初始化**时是否已经提高最终 cue 的可分离性",
        "why_this_is_valid": (
            "receptors 是固定随机稀疏压缩、不训练，故这次改动的效果在初始化时就已完整存在；"
            "本件**只能否证、不能确认**"
        ),
        "preregistered_ladder": {
            "separable_floor": SEPARABLE_FLOOR,
            "not_separable_below": ENTANGLED_CEILING,
        },
        "checkpoints": results,
        "per_stage_delta": verdict,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps({"report": str(report_path)}))
    print(f"  {'stage':22} {'off':>10} {'on':>10} {'delta':>10}  verdict(off -> on)")
    for stage in STAGES:
        row = verdict[stage]
        print(
            f"  {stage:22} {row['off']:+10.4f} {row['on']:+10.4f} {row['delta']:+10.4f}"
            f"  {results['off_at_init']['profile'][stage]['verdict']}"
            f" -> {results['on_at_init']['profile'][stage]['verdict']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
