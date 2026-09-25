"""T7：**F1 已训练的模型 × 区0 掩码 cue** —— 零训练判别（判读线见预注册，先冻结后跑）。

被评分对象：
* `A`：M5_R2 读出重训 A 臂（16M，F1 读出已训练），**现行读法**（三区全拼接）；
* `A_masked`：由 `A` **派生**——张量逐位相同，只有 config 的
  `predictive_context_region0_only` 翻成 True（掩码是读法，不碰任何张量）。

仪器：与 T6 同一台（`eval_taiji_cap0_baseline.run_baseline(dimensions=("D","E"))`，默认参数）。
自检：`S_A == 0`（A 臂已知 0/36）。
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

M2_DIMENSIONS = ("D", "E")


def _cap(checkpoint: Path) -> dict[str, Any]:
    from eval_taiji_cap0_baseline import run_baseline

    report = run_baseline(checkpoint=checkpoint, dimensions=M2_DIMENSIONS)
    tally = {key: report["dimensions"][key]["tally"] for key in M2_DIMENSIONS}
    return {
        "sum": sum(int(tally[key]["machine_scored_correct"] or 0) for key in M2_DIMENSIONS),
        "per_dimension": tally,
        "identity": report.get("identity"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-a", default="output/taiji_r2_readout_retrain/A/checkpoint.pt")
    parser.add_argument("--out-dir", default="output/taiji_r2_readout_retrain/A_masked")
    parser.add_argument("--out-report", default=None)
    args = parser.parse_args()

    from eval_taiji_cap0_baseline import run_baseline  # noqa: F401  (确保模块可导入)

    from seed import Seed

    arm_a = Path(args.arm_a)
    if not arm_a.is_absolute():
        arm_a = PROJECT_ROOT / arm_a
    if not arm_a.is_file():
        raise SystemExit(f"A 臂 checkpoint 不存在：{arm_a}")

    # ---- 派生 A_masked（张量逐位相同，只翻 config 的一个布尔）----
    envelope = torch.load(arm_a, map_location="cpu", weights_only=False)
    model = Seed.from_checkpoint(envelope)
    shared_cfg = model.architecture.config
    assert shared_cfg.predictive_context_region0_only is False
    object.__setattr__(shared_cfg, "predictive_context_region0_only", True)
    assert model.architecture.fabric.config.predictive_context_region0_only is True

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    staged = out_dir / "checkpoint.pt"
    if staged.exists():
        parser.error(f"{staged} already exists; 派生件不覆盖（删掉再跑）")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.checkpoint()  # 只为确认序列化可用；真正落盘在下面
    payload = model.checkpoint()
    from seed.persistence import atomic_save, attach_metadata

    atomic_save(
        attach_metadata(
            payload,
            tick=int(envelope.get("metadata", {}).get("tick", 0)),
            corpus_fingerprint=envelope.get("metadata", {}).get("corpus_fingerprint", ""),
            extra={
                "trainer": "derived_masked_readout",
                "derived_from": str(arm_a),
                "derivation": "config.predictive_context_region0_only=True; 张量逐位同 A",
            },
        ),
        staged,
    )
    print(json.dumps({"staged": str(staged)}), flush=True)

    # ---- 评分（同一台仪器）----
    scores = {"A": _cap(arm_a), "A_masked": _cap(staged)}

    # ---- 自检 + 判读线（预注册 §3）----
    sanity_ok = scores["A"]["sum"] == 0
    s_a = scores["A"]["sum"]
    s_m = scores["A_masked"]["sum"]
    if not sanity_ok:
        verdict = {
            "verdict": "INSTRUMENT_INVALID",
            "note": "S_A ≠ 0（A 臂已知 0/36）⇒ 仪器矛盾，全部读数作废",
            "scores": scores,
        }
    elif s_m >= 2 and s_m > s_a:
        verdict = {
            "verdict": "READ_FIX_BUYS_ANSWERS",
            "note": "读法修正在 F1 已训练的模型上换来了答对（端到端信号成立）",
        }
    elif s_m == 0:
        verdict = {
            "verdict": "READ_FIX_INSUFFICIENT",
            "note": "F1 已训练 + 可分离 cue 仍 0 ⇒ 问题在更深处；先读逐题生成样本",
        }
    else:
        verdict = {"verdict": "SIGNAL_BELOW_MARGIN", "note": "有信号未过 2 题门槛", "S_A_masked": s_m}

    payload_out = {
        "format": "taiji-r2-masked-arm-a-v1",
        "prereg": "plans/reference/M5_R2_T7_MASKED_A_PREREG_20260924.md",
        "staged_checkpoint": str(staged),
        "scores": scores,
        "sanity_S_A_zero": sanity_ok,
        "verdict": verdict,
    }
    out = Path(args.out_report) if args.out_report else (
        PROJECT_ROOT / "reports" / "taiji_r2_masked_arm_a_20260924.json"
    )
    if out.exists():
        parser.error(f"{out} already exists; 判决件不覆写")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"S_A": s_a, "S_A_masked": s_m, "verdict": verdict}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
