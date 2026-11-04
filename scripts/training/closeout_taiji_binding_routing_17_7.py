"""N17.7 project-level closeout: four ledgers from the frozen 17.5 evidence chain.

Read-only aggregation.  No training update, no confirm read, no dose/seed change.
It only re-states what the 17.1-17.5 receipts already measured, hashes the
persisted bundles, and lists the stop conditions and the competing explanations
that the 17.5 reading did not eliminate.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

COMPARE = ROOT / "reports/taiji_selection_routing_17_5_compare_20261104.json"
COMPARE_MANIFEST = ROOT / "plans/manifests/taiji_selection_routing_17_5_compare_20261104.json"
IDENTITY = ROOT / "reports/taiji_selection_routing_17_2r_identity_20261104.json"
FORMAT_EQ = ROOT / "reports/taiji_selection_routing_17_2r_format_equivalence_20261104.json"
PRICE = ROOT / "reports/taiji_selection_routing_17_3_price_20261103.json"
PREFLIGHT = ROOT / "reports/taiji_selection_routing_17_4_preflight_20261103_v2.json"
OUTPUT_DIR = ROOT / "output/taiji_selection_routing_17_5"
REPORT = ROOT / "reports/taiji_selection_routing_17_7_closeout_20261104.json"
MANIFEST = ROOT / "plans/manifests/taiji_selection_routing_17_7_closeout_20261104.json"

ARMS = ("A_route_lock", "B_stop_lock", "C_unlocked")
SEEDS = (28031, 28032)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def bundle_index() -> list[dict]:
    rows = []
    for seed in SEEDS:
        for arm in ARMS:
            for step in (150, 300):
                path = OUTPUT_DIR / f"{seed}-{arm}-step{step}.pt"
                rows.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "step": step,
                        "path": str(path.relative_to(ROOT)),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
    return rows


def reliable_range(compare: dict) -> dict:
    runs = compare["runs"]
    identity = compare["candidate"]
    readings = {}
    for tag, run in sorted(runs.items()):
        step0 = run["step0"]["overall"]["completed"]
        step300 = run["step300"]["overall"]["completed"]
        readings[tag] = {
            "step0_overall": step0,
            "step300_overall": step300,
            "delta": step300 - step0,
            "step300_selection_object": run["step300"]["selection"]["object_top1"],
            "step300_selection_place": run["step300"]["selection"]["place_top1"],
            "step300_nll": run["step300"]["nll"]["mean_nll"],
            "reversed_paired_rate": run["step300"]["reversed"]["paired_rate"],
            "identity_rescue_equal": run["step300"]["identity_rescue"]["equal"],
        }
    best = max(value["step300_overall"] for value in readings.values())
    selection = [value["step300_selection_object"] for value in readings.values()]
    unseen = [run["step300"]["overall"]["subsets"]["unseen_output_name"] for run in runs.values()]
    newdesc = [run["step300"]["overall"]["subsets"]["new_description"] for run in runs.values()]
    return {
        "identity": {
            "candidate_sha_v2": identity["candidate_sha_v2"],
            "identity_basis": identity["identity_basis"],
            "parameter_counts": identity["parameter_counts"],
            "consistent_across_cells": identity["consistent_across_cells"],
        },
        "per_cell": readings,
        "reliable": [
            "候选身份与参数清单逐格一致（语义哈希，六格同一个 candidate_sha_v2）",
            "step300 包、进度与包内 step 三者一致；六格均可重放评估，权重不再丢失",
            "identity_rescue 8/8 逐位成立：无干预的续推与干预后恢复完全一致",
            f"reversed 反序配对率 {min(value['reversed_paired_rate'] for value in readings.values()):.3f}-"
            f"{max(value['reversed_paired_rate'] for value in readings.values()):.3f}"
            "（在 overall 接近 0 时属退化通过，不构成复制不变性证据）",
        ],
        "not_reliable": [
            f"自由运行整句复制：六格最好 {best:.3f}，门 0.80",
            f"有序选择：物件 top1 区间 {min(selection):.3f}-{max(selection):.3f}，门 0.90",
            f"新输出名字 {min(unseen):.3f}-{max(unseen):.3f} / 新描述 "
            f"{min(newdesc):.3f}-{max(newdesc):.3f}，门 0.80",
            "跨 seed 稳定性：28032-A 的物件/地点选择 0.338/0.100、dev NLL 5.225，明显发散",
        ],
    }


def mechanism(compare: dict) -> dict:
    seed_gates = compare["seed_gates"]
    gamma = {
        tag: run["step300"]["gamma"]["gamma_effective_mean"]
        for tag, run in sorted(compare["runs"].items())
    }
    drops = {
        tag: run["step300"]["content_interventions"]["drops"]
        for tag, run in sorted(compare["runs"].items())
    }
    dependence = {
        arm: max(
            max(abs(value["drop"]) for value in drops[tag].values()) for tag in drops if arm in tag
        )
        for arm in ARMS
    }
    completion = max(run["step300"]["overall"]["completed"] for run in compare["runs"].values())
    return {
        "selection_object_top1": {
            tag: compare["runs"][tag]["step300"]["selection"]["object_top1"]
            for tag in sorted(compare["runs"])
        },
        "gamma_effective_mean": gamma,
        "gamma_floored_fraction": {
            tag: compare["runs"][tag]["step300"]["gamma"]["floored_fraction"]
            for tag in sorted(compare["runs"])
        },
        "content_drops": drops,
        "max_abs_content_drop_per_cell": {
            tag: max(abs(value["drop"]) for value in drops[tag].values()) for tag in drops
        },
        "max_abs_content_drop": max(
            abs(value["drop"])
            for drops_per_cell in drops.values()
            for value in drops_per_cell.values()
        ),
        "arm_content_dependence": dependence,
        "a_minus_c": {seed: seed_gates[seed]["a_minus_c"] for seed in seed_gates},
        "a_minus_b_report_only": {
            seed: seed_gates[seed]["a_minus_b_report_only"] for seed in seed_gates
        },
        "self_prefix_updates": {
            tag: compare["runs"][tag]["train"]["self_prefix_updates"]
            for tag in sorted(compare["runs"])
        },
        "train_last_loss": {
            tag: compare["runs"][tag]["train"]["last_loss"] for tag in sorted(compare["runs"])
        },
        "verdict": (
            f"copy is not load-bearing (completion stays <= {completion:.3f}); the route-credit arm "
            f"depends on records less than detach (A max drop {dependence['A_route_lock']:.3f} "
            f"vs B {dependence['B_stop_lock']:.3f})"
        ),
    }


def old_capability(compare: dict) -> dict:
    return {
        "parent": {seed: compare["runs"][f"{seed}-A_route_lock"]["parent"] for seed in SEEDS},
        "parent_task_status": "N16.1.3 native_byte_task_failed; no eligible task parent",
        "reusable": [
            "binding 图（3,526,481 参数）与复制/生成两路",
            "语义身份、bundle schema 与 step150 断点恢复（Adam 矩 + RNG + m/coverage）",
            "17.3 冻结阈值与 17.4 恢复预检结论",
            "三臂定义、阶段划分与冻结价格口径",
        ],
        "not_reusable_as_passed": [
            "任务级复制能力（父代与 17.5 均未达标）",
            "选择精度（0.75 < 0.90）作为 18 阶段输入",
        ],
    }


def next_task(compare: dict) -> dict:
    returns = {
        tag: compare["runs"][tag]["step300"]["overall"]["completed"]
        for tag in sorted(compare["runs"])
    }
    return {
        "best_arm": max(returns, key=returns.get),
        "best_overall": max(returns.values()),
        "reusable_for_18_22": [
            "候选容器、语义身份、bundle schema、恢复与回放工具链",
            "冻结阈值、评分器与内容干预三件套（可直接复用到新算子对照）",
            "三臂/阶段/价格合同（换算子时只需重定价，不重建流程）",
        ],
        "must_be_replaced": [
            "复制算子：当前 anchor@transport 的软扩散不足以让复制承重，OP2 候选要在结案后单独定价/对照",
            "课程判据：阈值挂在自由运行完成率上，本剂量下从未触发（self_prefix_updates=0），需改为教师强制完成率或按位置触发",
            "信用路径：ST 路由信用（A）未优于 detach（B），A−B 两 seed 均为负",
            "剂量：300 更新 / batch 8 不足；训练末 loss 0.10-0.15 而 dev NLL 1.2-2.1，存在记忆化",
        ],
        "not_allowed_this_window": "不自动追加 seed、剂量或新头；17 窗口截至本步",
    }


def main() -> int:
    start_cpu = time.process_time()
    start_wall = time.perf_counter()
    compare = read_json(COMPARE)
    if compare is None:
        raise FileNotFoundError(COMPARE)
    inputs = {
        "compare_report": {"path": str(COMPARE.relative_to(ROOT)), "sha256": sha256_file(COMPARE)},
        "compare_manifest": {
            "path": str(COMPARE_MANIFEST.relative_to(ROOT)),
            "sha256": sha256_file(COMPARE_MANIFEST),
        },
        "identity": read_json(IDENTITY) is not None,
        "format_equivalence": read_json(FORMAT_EQ) is not None,
        "price": read_json(PRICE) is not None,
        "preflight": read_json(PREFLIGHT) is not None,
    }
    dev_gate = compare["development_gate"]
    mech = mechanism(compare)
    last_losses = [value for value in mech["train_last_loss"].values() if value is not None]
    nlls = [run["step300"]["nll"]["mean_nll"] for run in compare["runs"].values()]
    prefix_by_arm = {
        arm: [value for tag, value in mech["self_prefix_updates"].items() if arm in tag]
        for arm in ARMS
    }
    stops = [
        "17.6 专属 confirm 不释放：dev 任务门 false，confirm 保持封存（读取次数 0）",
        "17.7 为 17 窗口终点：不追加 seed、不追加剂量、不新开头；后续 18–22 从设计卡重新准入",
        "17.5 的六份 step300 包与 step150 包保留为证据资产，不因负结果删除",
        "旧读数（17.1-17.4）保持原字节有效；本报告不改变任何冻结阈值",
    ]
    competing = [
        {
            "explanation": "剂量不足/欠训练（300 更新、LR 1e-3、batch 8）",
            "evidence": (
                f"训练末批 loss {min(last_losses):.2f}-{max(last_losses):.2f}，而 dev NLL "
                f"{min(nlls):.2f}-{max(nlls):.2f}；选择 top1 最高 "
                f"{max(run['step300']['selection']['object_top1'] for run in compare['runs'].values()):.3f}"
                " 未达 0.90"
            ),
            "status": "not_excluded",
        },
        {
            "explanation": "架构：goal→字串的参数化捷径使复制非必需",
            "evidence": (
                "三类内容干预的单格最大绝对降幅 "
                f"{min(mech['max_abs_content_drop_per_cell'].values()):.3f}-"
                f"{mech['max_abs_content_drop']:.3f}；A 臂（ST 信用）"
                f"{mech['arm_content_dependence']['A_route_lock']:.3f} 低于 B 臂（detach）"
                f"{mech['arm_content_dependence']['B_stop_lock']:.3f}，路由信用臂反而更不看记录"
            ),
            "status": "supported",
        },
        {
            "explanation": "训练/生成口径差（纯教师强制导致 exposure bias）",
            "evidence": (
                "课程按自由运行完成率触发：A/C 臂 self_prefix_updates 合计 "
                f"{sum(prefix_by_arm['A_route_lock']) + sum(prefix_by_arm['C_unlocked'])}（从未触发，"
                f"全程教师强制）；B 臂 {sum(prefix_by_arm['B_stop_lock'])} 次触发"
            ),
            "status": "supported",
        },
        {
            "explanation": "信用路径问题：ST 路由信用方向不佳",
            "evidence": (
                "A−B（参考读数）两 seed 为 "
                f"{mech['a_minus_b_report_only']['28031']['delta']:+.3f} / "
                f"{mech['a_minus_b_report_only']['28032']['delta']:+.3f}；"
                f"A−C 为 {mech['a_minus_c']['28031']['delta']:+.3f} / "
                f"{mech['a_minus_c']['28032']['delta']:+.3f}，门 +0.10"
            ),
            "status": "supported",
        },
        {
            "explanation": "评分/分母错误（历史 N16 曾出现分母与局部缓存缺陷）",
            "evidence": "本脚本沿用 17.5 已冻结评分器与 17.3 阈值；退化 reversed 通过已在报告中标注",
            "status": "not_excluded_for_reversed_only",
        },
    ]
    report = {
        "schema": "taiji-selection-routing-17-7-closeout-v1",
        "step": "N17.7",
        "date": "2026-11-04",
        "status": "n17_negative_closeout",
        "verdict": {
            "development_gate": dev_gate,
            "release_confirm_n17": dev_gate["release_confirm_n17"],
            "n17_6": "not_released",
            "n17_7": "closed_negative",
        },
        "inputs": inputs,
        "ledgers": {
            "reliable_range_same_candidate": reliable_range(compare),
            "mechanism_actual_contribution": mechanism(compare),
            "applicable_old_capability_state_resources": old_capability(compare),
            "next_task_applicability": next_task(compare),
        },
        "stop_conditions": stops,
        "remaining_competing_explanations": competing,
        "assets": {"bundles": bundle_index()},
        "cost": {
            "optimizer_steps": 0,
            "training_updates": 0,
            "confirm_reads": 0,
            "cpu_seconds": time.process_time() - start_cpu,
            "wall_seconds": time.perf_counter() - start_wall,
        },
        "totals_reference": {
            "n17_5_cumulative_cpu_seconds": compare["totals"]["cumulative"]["cpu_seconds"],
            "n17_5_cumulative_wall_seconds": compare["totals"]["cumulative"]["wall_seconds"],
            "watch_cpu": compare["totals"]["cumulative"]["watch_cpu"],
            "watch_wall": compare["totals"]["cumulative"]["watch_wall"],
            "watch_ratio_cpu": (
                compare["totals"]["cumulative"]["cpu_seconds"]
                / compare["totals"]["cumulative"]["watch_cpu"]
            ),
        },
    }
    write_json(REPORT, report)
    write_json(
        MANIFEST,
        {
            "schema": "taiji-selection-routing-17-7-closeout-manifest-v1",
            "step": "N17.7",
            "date": "2026-11-04",
            "report": str(REPORT.relative_to(ROOT)),
            "report_sha256": sha256_file(REPORT),
            "report_bytes": REPORT.stat().st_size,
            "compare_manifest_sha256": inputs["compare_manifest"]["sha256"],
            "bundles": {row["path"]: row["sha256"] for row in report["assets"]["bundles"]},
            "optimizer_steps": 0,
            "training_updates": 0,
            "confirm_reads": 0,
        },
    )
    print(
        json.dumps(
            {
                "report": str(REPORT.relative_to(ROOT)),
                "status": report["status"],
                "dev_gate": dev_gate["passed"],
                "best_overall": report["ledgers"]["next_task_applicability"]["best_overall"],
                "cpu": round(report["cost"]["cpu_seconds"], 2),
                "wall": round(report["cost"]["wall_seconds"], 2),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
