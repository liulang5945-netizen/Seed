"""R2 受控重训语言读出：A/B/C 三臂判决器（M1 + M2 + K2）。

合同：``plans/reference/M5_R2_READOUT_RETRAIN_CONTRACT_DRAFT_20260920.md`` §5。三条门：

* **M1（主门）** A 臂在**未见** dev 上的 n 元成句率 ≥ B、C 两臂的**最好者** + 15 pp，
  且逐题配对符号检验 `p < 0.05`；
* **M2（主门）** A 臂 CAP **D+E** 机器计分之和 ≥ B、C 的最好者 + 2 题（同题面同链路）；
* **K2（反事实）** 若 B 与 A 无差别 ⇒ 判"更多训练"即可解释，**读出侧不得记功**。

两条自我约束（都是为了不把本件变成"重写判据凑结果"）：

1. **判据一个字都不重写**。成句判据、n 元模型、反向对照门、配对符号检验全部**直接 import**
   已 FROZEN 的 P1/P2 仪器 ``diag_taiji_r2_surface_decode``；解码策略固定为 ``greedy``
   （P1 已实测它是四臂里最好的，本件问的是训练侧，不是解码侧）。这里只加"三臂配对"这一层。
2. **`--dry-run` 不产出判决**。为了让仪器在训练跑完之前就能被验证，dry-run 用同一份基座冒充三臂，
   报告里 ``dry_run=true`` 且 ``judgment`` 块被显式抑制（``suppressed``）——它只证明管线能跑通，
   **任何数字都不得被引用为读数**。

拒跑条件（缺一条即整件不执行，缺臂即作废配对）：

* 三臂 checkpoint 必须全在；
* 三臂必须来自**同一**基底 sha256、**同一** ``skip_symbols``，且都**跑满**同一预算
  （``symbols_consumed == symbols_budget == --symbols``）；
* 反向对照门（真中文须过 ≥0.5、打乱中文须挂 ≥0.9）不过 ⇒ 判据无判别力，拒跑；
* 报告已存在即拒跑（封存件不覆写）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts" / "training") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))

from diag_taiji_r2_surface_decode import (  # noqa: E402
    assert_criterion_discriminates,
    build_ngram_model,
    generate,
    sign_test,
    well_formed,
)
from eval_taiji_cap0_baseline import run_baseline  # noqa: E402
from readout_retrain_spec import APPROVED_SYMBOL_CEILING, TRAINER_NAME  # noqa: E402
from readout_retrain_spec import ARMS as TRAIN_ARMS  # noqa: E402

CONTRACT = "plans/reference/M5_R2_READOUT_RETRAIN_CONTRACT_DRAFT_20260920.md"
EVAL_SET = PROJECT_ROOT / "plans" / "manifests" / "cap0_eval_set_v2.json"
DEFAULT_RUN_DIR = PROJECT_ROOT / "output" / "taiji_r2_readout_retrain"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_r2_readout_retrain_verdict_20260920.json"
#: dry-run 的落盘路径**必须与正式判决不同**：判决报告"已存在即拒跑"，一次自检若占了正式路径，
#: 就等于把真判决挡在门外。
DRY_RUN_REPORT = (
    PROJECT_ROOT / "reports" / "taiji_r2_readout_retrain_pipeline_check_20260920.json"
)
#: M1 的题面集：与已 FROZEN 的 P1/P2 仪器同源（CAP 现行集合的 B/G 两维，逐轮取题面）。
M1_DIMENSIONS = ("B", "G")
#: M2 的维度与增量。
M2_DIMENSIONS = ("D", "E")
M1_MARGIN_PP = 15.0
M2_MARGIN_ITEMS = 2


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def arm_checkpoint(run_dir: Path, arm: str) -> Path:
    return run_dir / arm / "checkpoint.pt"


def verify_pairing(run_dir: Path, symbols: int, dry_run: bool) -> dict[str, Any]:
    """Fail closed unless the three arms form a valid matched triple."""

    import torch

    seen: dict[str, Any] = {}
    for arm in sorted(TRAIN_ARMS):
        path = arm_checkpoint(run_dir, arm)
        if not path.is_file():
            if dry_run:
                continue
            raise SystemExit(f"arm {arm} has no checkpoint at {path}; 缺臂即作废配对，拒判")
        envelope = torch.load(path, map_location="cpu", weights_only=False)
        metadata = envelope.get("metadata") if isinstance(envelope.get("metadata"), dict) else {}
        seen[arm] = {
            "path": str(path),
            "trainer": metadata.get("trainer"),
            "symbols_consumed": metadata.get("symbols_consumed"),
            "symbols_budget": metadata.get("symbols_budget"),
            "skip_symbols": metadata.get("skip_symbols"),
            "base_checkpoint_sha256": metadata.get("base_checkpoint_sha256"),
        }
    if dry_run:
        return seen

    for arm, entry in seen.items():
        if entry["trainer"] != TRAINER_NAME:
            raise SystemExit(f"arm {arm} was not produced by {TRAINER_NAME}")
        if entry["symbols_consumed"] != symbols or entry["symbols_budget"] != symbols:
            raise SystemExit(
                f"arm {arm} did not finish the approved budget "
                f"(consumed={entry['symbols_consumed']} budget={entry['symbols_budget']} "
                f"requested={symbols}); 未跑满的臂不得进入配对"
            )
    for key in ("skip_symbols", "base_checkpoint_sha256"):
        values = {arm: entry[key] for arm, entry in seen.items()}
        if len(set(values.values())) != 1:
            raise SystemExit(f"三臂的 {key} 不一致 ⇒ 不是同一段符号流/同一基底，拒判: {values}")
    return seen


def m1_tasks() -> list[tuple[str, str]]:
    """The frozen P1/P2 prompt set: every turn of every B/G item, in collection order."""

    payload = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    tasks: list[tuple[str, str]] = []
    for dim in M1_DIMENSIONS:
        for item in payload["dimensions"][dim]["items"]:
            for turn in item.get("turns") or []:
                prompt = turn.get("prompt") if isinstance(turn, dict) else turn
                if prompt:
                    tasks.append((str(item["id"]), str(prompt)))
    if not tasks:
        raise SystemExit("评价集里没取到任何题面：字段名与预期不符，拒跑而不是静默产出空报告")
    return tasks


def run_m1(run_dir: Path, dry_run: bool) -> dict[str, Any]:
    from diag_taiji_r2_surface_decode import _serialize_prompt  # noqa: PLC2701

    from api.seed_runtime import SeedRuntime

    base = SeedRuntime.load(PROJECT_ROOT / "checkpoints" / "seed_beta.pt")
    tasks = m1_tasks()
    ngram = build_ngram_model()
    controls = assert_criterion_discriminates(ngram)

    per_arm: dict[str, dict[str, Any]] = {}
    for arm in sorted(TRAIN_ARMS):
        checkpoint = arm_checkpoint(run_dir, arm) if not dry_run else None
        # dry-run 时三臂共用同一份基座实例——只证明管线通，不产生可比读数。
        runtime = base if checkpoint is None else SeedRuntime.load(checkpoint)
        taiji = runtime.model.substrate
        texts: list[str] = []
        started = time.perf_counter()
        for _, prompt in tasks:
            texts.append(
                generate(taiji, _serialize_prompt(runtime, prompt), "greedy").decode(
                    "utf-8", errors="replace"
                )
            )
        flags = [1 if well_formed(text, ngram) else 0 for text in texts]
        per_arm[arm] = {
            "usable": checkpoint is not None,
            "checkpoint": None if checkpoint is None else str(checkpoint),
            "n": len(texts),
            "well_formed_rate": round(sum(flags) / len(flags), 4),
            "flags": flags,
            "seconds": round(time.perf_counter() - started, 2),
            "samples": texts[:3],
            "mean_len_chars": round(sum(len(t) for t in texts) / len(texts), 2),
        }
    return {"tasks": len(tasks), "controls": controls, "per_arm": per_arm}


def judge_m1(per_arm: dict[str, Any], dry_run: bool) -> dict[str, Any]:
    a = per_arm["A"]
    rivals = [per_arm[arm] for arm in ("B", "C") if per_arm[arm]["usable"]]
    if dry_run or not rivals or not a["usable"]:
        return {"suppressed": "dry-run 或臂不齐：本块不是判决，任何数字不得引用"}
    best = max(rivals, key=lambda entry: entry["well_formed_rate"])
    best_arm = next(arm for arm in ("B", "C") if per_arm[arm] is best)
    pairs = list(zip(a["flags"], best["flags"], strict=True))
    delta_pp = round((a["well_formed_rate"] - best["well_formed_rate"]) * 100, 2)
    test = sign_test([(rival, ours) for ours, rival in pairs])
    return {
        "best_rival": best_arm,
        "a_rate": a["well_formed_rate"],
        "best_rival_rate": best["well_formed_rate"],
        "delta_percentage_points": delta_pp,
        "margin_required_pp": M1_MARGIN_PP,
        "sign_test_a_vs_best_rival": test,
        "holds": bool(delta_pp >= M1_MARGIN_PP and test["p_two_sided"] < 0.05),
    }


def judge_k2(m1_per_arm: dict[str, Any], dry_run: bool) -> dict[str, Any]:
    """K2 反事实门：B 与 A 无差别 ⇒ 判"更多训练"即可解释。

    合同 §5 只写了 K2 的**语义**（无差别则读出侧不得记功）没写操作化阈值。
    本件在执行前把它预注册为：B 相对 A 的成句率差 **< +15 pp**（同一个"可分辨最小步进"）
    **且** 配对符号检验 **p ≥ 0.05**（即两者不可分辨）⇒ 判无差别。
    """

    if dry_run:
        return {"suppressed": "dry-run：本块不是判决"}
    a, b = m1_per_arm["A"], m1_per_arm["B"]
    if not (a["usable"] and b["usable"]):
        return {"status": "arm_missing", "note": "缺臂，K2 无法判"}
    pairs = list(zip(a["flags"], b["flags"], strict=True))
    delta_pp = round((b["well_formed_rate"] - a["well_formed_rate"]) * 100, 2)
    test = sign_test([(ours, rival) for ours, rival in pairs])
    indistinguishable = bool(delta_pp < M1_MARGIN_PP and test["p_two_sided"] >= 0.05)
    return {
        "preregistered_operationalisation": {
            "delta_pp_threshold": M1_MARGIN_PP,
            "significance": "paired sign test p >= 0.05",
            "registered_before_any_arm_finished": True,
        },
        "b_minus_a_delta_pp": delta_pp,
        "sign_test_b_vs_a": test,
        "B_explains_it_by_more_training": indistinguishable,
        "consequence": (
            "判无差别 ⇒ 读出侧不得记功，M1 的改善归'更多训练'"
            if indistinguishable
            else "A 与 B 可分辨 ⇒ 读出侧可归因（仍须过 M1/M2 才算过）"
        ),
    }


def run_m2(run_dir: Path, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return {
            "status": "skipped_in_dry_run",
            "note": f"正式判定时会对每臂跑 run_baseline(dimensions={M2_DIMENSIONS})",
        }
    per_arm: dict[str, Any] = {}
    for arm in sorted(TRAIN_ARMS):
        checkpoint = arm_checkpoint(run_dir, arm)
        report = run_baseline(checkpoint=checkpoint, dimensions=M2_DIMENSIONS)
        per_arm[arm] = {
            "machine_scored_correct": sum(
                int(report["dimensions"][key]["tally"]["machine_scored_correct"] or 0)
                for key in M2_DIMENSIONS
            ),
            "per_dimension": {
                key: report["dimensions"][key]["tally"] for key in M2_DIMENSIONS
            },
            "identity": report.get("identity"),
            "chain": report.get("chain"),
        }
    rivals = [per_arm[arm]["machine_scored_correct"] for arm in ("B", "C")]
    best = max(rivals)
    best_arm = next(arm for arm in ("B", "C") if per_arm[arm]["machine_scored_correct"] == best)
    ours = per_arm["A"]["machine_scored_correct"]
    return {
        "per_arm": per_arm,
        "best_rival": best_arm,
        "a_machine_scored_correct": ours,
        "best_rival_machine_scored_correct": best,
        "delta_items": ours - best,
        "margin_required_items": M2_MARGIN_ITEMS,
        "holds": bool(ours >= best + M2_MARGIN_ITEMS),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--symbols", type=int, default=APPROVED_SYMBOL_CEILING)
    parser.add_argument(
        "--out-report",
        default=None,
        help="缺省：正式跑落 verdict 报告，--dry-run 落 pipeline_check 报告"
        "（两者分开，免得一次自检把正式报告的路径占掉）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="用同一份基座冒充三臂，只验证管线；报告里 judgment 被显式抑制",
    )
    parser.add_argument("--skip-m2", action="store_true", help="只跑 M1/K2（M2 较慢）")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    if DRY_RUN_REPORT == DEFAULT_REPORT:
        parser.error("dry-run 与正式判决不得共用同一路径；否则一次自检会把真判决挡在门外")
    default_report = DRY_RUN_REPORT if args.dry_run else DEFAULT_REPORT
    report_path = Path(args.out_report) if args.out_report else default_report
    if not report_path.is_absolute():
        report_path = PROJECT_ROOT / report_path
    if report_path.exists():
        parser.error(f"{report_path} already exists; verdicts are never overwritten")

    pairing = verify_pairing(run_dir, args.symbols, args.dry_run)
    m1 = run_m1(run_dir, args.dry_run)
    judgment: dict[str, Any] = {
        "M1_n_gram_well_formed": judge_m1(m1["per_arm"], args.dry_run),
        "K2_more_training_counterfactual": judge_k2(m1["per_arm"], args.dry_run),
    }
    judgment["M2_cap_d_plus_e"] = (
        {"status": "skipped_by_request"} if args.skip_m2 else run_m2(run_dir, args.dry_run)
    )
    if not args.dry_run:
        judgment["scope"] = "以未跑满预算/血缘不一致的臂为依据的判决作废；本件已在入口拒跑这类输入"

    payload = {
        "format": "taiji-r2-readout-retrain-verdict-v1",
        "dry_run": bool(args.dry_run),
        "contract": CONTRACT,
        "written_at_utc": _utc_now(),
        "run_dir": str(run_dir),
        "symbols": args.symbols,
        "arm_pairing": pairing,
        "evaluation_material": {
            "m1": {
                "eval_set": str(EVAL_SET.relative_to(PROJECT_ROOT).as_posix()),
                "dimensions": list(M1_DIMENSIONS),
                "instrument": "scripts/training/diag_taiji_r2_surface_decode.py (FROZEN)",
                "decoding": "greedy（P1 已实测为四臂最好；本件不动解码）",
                "tasks": m1["tasks"],
            },
            "m2": {
                "dimensions": list(M2_DIMENSIONS),
                "instrument": "scripts/training/eval_taiji_cap0_baseline.py::run_baseline",
                "chain": "产品链路（relax_legacy_guard 默认），与既有 CAP 报告同口径",
            },
        },
        "criterion_controls": {
            **m1["controls"],
            "note": "反向对照门不过则整件拒跑；成句判据本身自 FROZEN 的 P1/P2 仪器 import，未重写",
        },
        "per_arm": m1["per_arm"],
        "judgment": judgment,
        "claim_limits": [
            "成句率只说明表层形态，不构成语言回答能力主张",
            "M1 的题面集是训练前就已冻结的 cap0_eval_set_v2；未新增题面（新增属评价集 v3）",
            "K2 的操作化阈值在**任何一臂跑完之前**登记在本文件与合同 §5.1，不随结果调整",
        ],
    }
    if args.dry_run:
        payload["judgment"] = {"suppressed": "dry-run：本报告不含判决，任何数字不得引用"}
        payload["dry_run_warning"] = (
            "三臂共用同一份基座 ⇒ 逐臂读数必然相同，这不是读数，只是管线自检"
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dry_run": payload["dry_run"], "report": str(report_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
