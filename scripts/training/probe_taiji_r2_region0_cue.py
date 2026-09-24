"""零训练预检：把 F1 的输入**掩码成只吃区 0**之后，16M 基座的 cue 可分离性是多少？

## 为什么有这一步

逐区审计（`audit_taiji_r2_per_region_degradation.py`）显示：槽结构只存在于**区 0**
（16M 时 trace 0.729 / activity 0.709），而三区全拼接的 cue 只有 **0.4197** ——
稀释发生在"读法"上，不在结构上。

⇒ 若把 F1 的输入掩码成只吃区 0，**已训练 16M 的模型**应当在**不重训**的情况下
把 cue 的可分离性从 0.4197 拉回区 0 的水平（~0.73）。**这一步就是验证它**——
若拉不回来，说明稀释不（只）发生在读法上，"只吃区 0"这条修法**当场否掉**，不用跑训练。

## 实现

`fabric.predictive_context` 加了 `predictive_context_region0_only`（默认 False、逐位不变）：
**掩码**而非改宽度 ⇒ `receptors` 形状与既有 checkpoint 完全不动。
本件同时核三件：① 掩码真的咬住（区1/2 段必须为 0）；② 既有 checkpoint 能在掩码开/关两种配置下读回；
③ 两种配置下的 cue 可分离性并排。
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

from probe_taiji_r2_separability_before_training import measure  # noqa: E402

STAGES = ("1_activity_block", "2_trace_block", "3_context")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(PROJECT_ROOT / "checkpoints" / "seed_beta.pt"))
    parser.add_argument("--out-report", default=None)
    args = parser.parse_args()
    from seed import Seed, SeedConfig
    from taiji import TaijiConfig

    checkpoint = Path(args.checkpoint)
    envelope = torch.load(checkpoint, map_location="cpu", weights_only=False)
    values = dict(envelope["config"]["taiji"])
    values.pop("receptors_factored", None)

    current = Seed.from_checkpoint(envelope)
    cfg_on = SeedConfig(taiji=TaijiConfig.from_dict({**values, "predictive_context_region0_only": True}))
    masked = Seed.from_checkpoint(envelope)
    # `Seed.restore` 校验配置一致（不能把不同 config 的状态塞进来），而测量需要**同一个已训练模型**
    # 在两种读法下各量一遍 ⇒ 读回后在**活对象**上翻转这一个布尔（不改任何张量、不落盘）。
    # 训练用的是 config 字段本身（`predictive_context_region0_only` 会随 checkpoint 走），不是这个技巧。
    # `TaijiConfig` 是 frozen dataclass，且**所有器官持有同一个 config 对象**
    # ⇒ 用 object.__setattr__ 在共享对象上原地翻转一次，处处生效（fabric/predictive_context 同源）。
    shared_cfg = masked.architecture.config
    assert shared_cfg.predictive_context_region0_only is False
    object.__setattr__(shared_cfg, "predictive_context_region0_only", True)
    assert masked.architecture.fabric.config.predictive_context_region0_only is True

    # ① 掩码必须咬住：区1/2 的段必须恒为 0
    regions = masked.architecture._state.regions
    probe_vec = masked.architecture.fabric.predictive_context(regions)
    n0 = regions[0].activity.shape[0]
    total = sum(cfg_on.taiji.region_sizes)
    regions12 = torch.cat([probe_vec[n0:total], probe_vec[total + n0 :]])
    mask_bites = bool((regions12 == 0).all())
    assert mask_bites, "掩码没有咬住：区1/2 的段不是 0"
    assert probe_vec.shape == (2 * total,), "宽度不该变"

    results = {
        "current_readout": measure(current.architecture, "current"),
        "region0_only_readout": measure(masked.architecture, "region0"),
    }
    # 确认集（先封后看的那一套）也要量：T4 的教训是只报一个集会被质疑挑集
    manifest = PROJECT_ROOT / "plans" / "manifests" / "r2_separability_confirmation_families_20260923.json"
    confirm_families = json.loads(manifest.read_text(encoding="utf-8"))["families"]
    results["current_readout_confirmation"] = measure(
        current.architecture, "current-conf", confirm_families
    )
    results["region0_only_readout_confirmation"] = measure(
        masked.architecture, "region0-conf", confirm_families
    )
    delta = {
        stage: round(
            results["region0_only_readout"][stage]["consistency_median"]
            - results["current_readout"][stage]["consistency_median"],
            6,
        )
        for stage in STAGES
    }

    payload = {
        "format": "taiji-r2-region0-cue-precheck-v1",
        "checkpoint": str(checkpoint),
        "question": "掩码成只吃区 0 后，**已训练 16M** 的 cue 可分离性是否回到区 0 的水平",
        "mask_bites": mask_bites,
        "context_width": int(probe_vec.shape[0]),
        "profiles": results,
        "delta_region0_minus_current": delta,
        "note": "零训练、只读既有 checkpoint；若拉不回来 ⇒ '只吃区0'当场否掉，不跑训练",
    }
    out = Path(args.out_report) if args.out_report else (
        PROJECT_ROOT / "reports" / "taiji_r2_region0_cue_precheck_20260924.json"
    )
    if out.exists():
        parser.error(f"{out} already exists; 判决件不覆写")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"report": str(out), "mask_bites": mask_bites}))
    print(f"  {'stage':20} {'现行':>10} {'只吃区0':>10} {'delta':>10}")
    for stage in STAGES:
        print(
            f"  {stage:20} {results['current_readout'][stage]['consistency_median']:>10.4f} "
            f"{results['region0_only_readout'][stage]['consistency_median']:>10.4f} {delta[stage]:>+10.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
