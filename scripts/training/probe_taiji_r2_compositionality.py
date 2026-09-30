"""R2 组合性探针：2×2 因子，问"换一个槽的效应是否跨另一个槽保持一致"——零训练。

预注册（判据、阈值、刺激表、零假设对照）见
``plans/reference/M5_R2_COMPOSITIONALITY_PROBE_CONTRACT_20260923.md``，随本文件在打分前提交。

**测量路径直接复用 P3 的执行件**（`_prefill` / `_kl_nats` / `_cue_l1_relative`）：同一段代码取 cue 与
读出分布，两个仪器不会各写一份慢慢漂移。本文件只加"四格因子 + 可分离性统计"这一层。

核心量：`consistency_A = cos(c₁₀−c₀₀, c₁₁−c₀₁)`（换 A 的方向在 B 的两个水平上是否同一个方向），
`consistency_B` 对称。96 维 cue 里两条随机方向的余弦 ≈ 0（sd≈0.10），所以"接近 1"是远离随机基线的
强预测；而"两个方向正交"在 96 维里是**废判据**（随机就正交），设计时就排除了。
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

from probe_taiji_r2_readout_conditionality import _kl_nats, _prefill  # noqa: E402

CONTRACT = "plans/reference/M5_R2_COMPOSITIONALITY_PROBE_CONTRACT_20260923.md"

#: 执行前冻结的判读线（合同 §3）
SEPARABLE_FLOOR = 0.9
ENTANGLED_CEILING = 0.5
BASELINE_GAP_FLOOR = 0.1

#: 冻死的刺激集：每族一个含两个槽的骨架，槽各取两水平 ⇒ 2×2 四格。
#: **由我撰写**（合同未规定来源），随本文件在打分前提交，之后不得增删改。
FAMILIES: list[dict[str, Any]] = [
    {"id": "f1", "skeleton": "我在{a}看见了一只{b}", "a": ["北京", "上海"], "b": ["猫", "狗"]},
    {"id": "f2", "skeleton": "{a}的时候我常常{b}", "a": ["早上", "晚上"], "b": ["读书", "写字"]},
    {"id": "f3", "skeleton": "他喜欢{a}还是{b}", "a": ["春天", "冬天"], "b": ["音乐", "美术"]},
    {"id": "f4", "skeleton": "{a}的味道和{b}很像", "a": ["苹果", "香蕉"], "b": ["蜂蜜", "柠檬"]},
    {
        "id": "f5",
        "skeleton": "请你告诉我{a}和{b}的区别",
        "a": ["学习", "工作"],
        "b": ["时间", "空间"],
    },
    {
        "id": "f6",
        "skeleton": "我昨天{a}所以今天{b}",
        "a": ["熬夜", "早起"],
        "b": ["很累", "很精神"],
    },
]


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    na, nb = float(a.norm()), float(b.norm())
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(torch.dot(a, b) / (na * nb))


def _cells(substrate: Any, runtime: Any, family: dict[str, Any], label: str) -> dict[str, Any]:
    """四个格：(a_i, b_j) i,j ∈ {0,1}；每格一个全新 episode。"""

    out: dict[str, Any] = {}
    for i in (0, 1):
        for j in (0, 1):
            prompt = family["skeleton"].format(a=family["a"][i], b=family["b"][j])
            out[f"{i}{j}"] = {
                "prompt": prompt,
                **_prefill(substrate, runtime, prompt, f"comp-{label}-{family['id']}-{i}{j}"),
            }
    return out


def probe_checkpoint(label: str, checkpoint: Path, runtime_cls: Any) -> dict[str, Any]:
    runtime = runtime_cls.load(checkpoint)
    substrate = runtime.model.substrate

    # 零假设对照（硬前置）：同一条题面在两个全新 episode 必须逐位相同
    first = _prefill(substrate, runtime, "我在北京看见了一只猫", f"comp-null-a-{label}")
    second = _prefill(substrate, runtime, "我在北京看见了一只猫", f"comp-null-b-{label}")
    null_cue_diff = float((first["cue"] - second["cue"]).abs().sum())
    null_kl = _kl_nats(first["probabilities"], second["probabilities"])
    null_ok = null_cue_diff == 0.0 and null_kl == 0.0

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    dirs: dict[str, dict[str, torch.Tensor]] = {}
    for family in FAMILIES:
        cells = _cells(substrate, runtime, family, label)
        c00, c10 = cells["00"]["cue"], cells["10"]["cue"]
        c01, c11 = cells["01"]["cue"], cells["11"]["cue"]
        p00 = cells["00"]["probabilities"]
        kl_a = _kl_nats(p00, cells["10"]["probabilities"])
        kl_b = _kl_nats(p00, cells["01"]["probabilities"])
        kl_ab = _kl_nats(p00, cells["11"]["probabilities"])
        dirs[family["id"]] = {"u": c10 - c00, "v": c01 - c00}
        rows.append(
            {
                "family": family["id"],
                "consistency_A": round(_cos(c10 - c00, c11 - c01), 6),
                "consistency_B": round(_cos(c01 - c00, c11 - c10), 6),
                "kl_a_to_a1": round(kl_a, 8),
                "kl_b_to_b1": round(kl_b, 8),
                "kl_both": round(kl_ab, 8),
                "kl_interaction_ratio": (
                    round(kl_ab / (kl_a + kl_b), 6) if (kl_a + kl_b) > 0 else None
                ),
                "prompts": {k: cells[k]["prompt"] for k in ("00", "10", "01", "11")},
            }
        )

    # 实测基线：不同族的方向互配（"不相干方向"的经验分布）
    ids = [f["id"] for f in FAMILIES]
    baseline: list[float] = []
    for i, left in enumerate(ids):
        for right in ids[i + 1 :]:
            baseline.append(_cos(dirs[left]["u"], dirs[right]["u"]))
            baseline.append(_cos(dirs[left]["v"], dirs[right]["v"]))

    consistencies = [r["consistency_A"] for r in rows] + [r["consistency_B"] for r in rows]
    ratios = [r["kl_interaction_ratio"] for r in rows if r["kl_interaction_ratio"] is not None]

    def _median(values: list[float]) -> float:
        ordered = sorted(values)
        return ordered[len(ordered) // 2] if ordered else 0.0

    median_consistency = _median(consistencies)
    median_baseline = _median(baseline)
    gap = median_consistency - median_baseline
    if gap < BASELINE_GAP_FLOOR:
        verdict = "inconclusive"
    elif median_consistency >= SEPARABLE_FLOOR:
        verdict = "separable"
    elif median_consistency >= ENTANGLED_CEILING:
        verdict = "partial"
    else:
        verdict = "entangled"

    return {
        "label": label,
        "checkpoint": str(checkpoint),
        "null_control": {"cue_l1_diff": null_cue_diff, "kl_nats": null_kl, "ok": null_ok},
        "rows": rows,
        "summary": {
            "consistency_median": round(median_consistency, 6),
            "consistency_min": round(min(consistencies), 6),
            "consistency_max": round(max(consistencies), 6),
            "baseline_median": round(median_baseline, 6),
            "baseline_n": len(baseline),
            "gap_over_baseline": round(gap, 6),
            "kl_interaction_ratio_median": round(_median(ratios), 6) if ratios else None,
        },
        "verdict": verdict,
        "seconds": round(time.perf_counter() - started, 2),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-checkpoint", default=str(PROJECT_ROOT / "checkpoints" / "seed_beta.pt")
    )
    parser.add_argument(
        "--arm-a",
        default=str(PROJECT_ROOT / "output" / "taiji_r2_readout_retrain" / "A" / "checkpoint.pt"),
    )
    parser.add_argument("--out-report", default=None)
    parser.add_argument("--dry-run", action="store_true", help="只跑基座，验证管线")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    from api.seed_runtime import SeedRuntime

    report_path = Path(
        args.out_report
        or PROJECT_ROOT / "reports" / "taiji_r2_compositionality_probe_20260923.json"
    )
    if report_path.exists():
        parser.error(f"{report_path} already exists; 判决件不覆写")

    base = Path(args.base_checkpoint)
    arm_a = Path(args.arm_a)
    if not base.is_file():
        parser.error(f"base checkpoint not found: {base}")
    targets = [("base", base)]
    if not args.dry_run:
        if not arm_a.is_file():
            parser.error(f"arm A checkpoint not found: {arm_a}")
        targets.append(("arm_A", arm_a))

    results = {label: probe_checkpoint(label, path, SeedRuntime) for label, path in targets}
    clean = all(entry["null_control"]["ok"] for entry in results.values())
    payload = {
        "format": "taiji-r2-compositionality-probe-v1",
        "status": "completed" if clean else "failed",
        "written_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "contract": CONTRACT,
        "question": "换一个槽的效应，是否跨另一个槽保持一致（表征可否按槽分解）",
        "preregistered_thresholds": {
            "separable_floor": SEPARABLE_FLOOR,
            "entangled_ceiling": ENTANGLED_CEILING,
            "baseline_gap_floor": BASELINE_GAP_FLOOR,
            "note": "96 维 cue 里随机方向的余弦 ≈ 0（sd≈0.10）；同时报实测基线",
        },
        "disclosures": [
            "cue 取自 substrate._state.motor_context（私有）——公开 TaijiStep 没暴露 cue",
            "刺激集由执行者撰写（合同未规定来源），表与阈值随执行件在打分前提交",
            "测量路径复用 P3 执行件（_prefill/_kl_nats），两个仪器不各自实现",
        ],
        "failure": None if clean else "零假设对照不干净（同题面两次给出不同的 cue 或 KL≠0），拒判",
        "checkpoints": results,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": payload["status"], "report": str(report_path)}))
    for label, entry in results.items():
        s = entry["summary"]
        print(
            f"  {label}: null_ok={entry['null_control']['ok']} verdict={entry['verdict']} "
            f"consistency={s['consistency_median']} baseline={s['baseline_median']} "
            f"gap={s['gap_over_baseline']} kl_ratio={s['kl_interaction_ratio_median']}"
        )
    return 0 if payload["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
