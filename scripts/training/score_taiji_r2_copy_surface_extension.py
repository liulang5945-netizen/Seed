"""§4.3a 表层子判据的**扩展分母**跑法（题集 `r2_copy_surface_extension_v1.json`，104 题≈260 条文本）。

预注册：`plans/reference/SPEC-A-21_r2_surface_extension_prereg_20260925.md`。

**这条仪器只为解决一件事**：原 §4.3a 的表层判据在 36 题（≈72 条文本）上**结构性不可判**——
裁定要求"成句差 ≥3 且两次独立取数同向"，而那个分母上 ±1 条≈2.8pp，等于要求 8% 的摆动。
本件把分母抬到约 260 条文本（±1 条≈0.4pp），并把"两次独立取数"落到**两个独立初始化的电路**上
（seed-A＝A2.3b 已落地的那份；seed-B＝`--circuit-seed 20260926` 重训的那份）。

链路：与 A2.3b 的 chat CAP 完全同一条（`SeedRuntime._serialize` 铺文本、`_record_told_history` 入库、
`generate_input(reset=True)` 取基底原始字节、`_TURN_MARKERS` 折字、n 元口径成句判定）——
**换分母不换链**，否则读数不可比（本仓多次实测：换链路会让同一判据反向）。

判读（冻结在预注册里，这里只实现）：
* 每个电路各自与对照臂比成句文本数；
* **两个电路同向**且成句差 ≥3 条 ⇒ 判"表层劣化/改善"；
* 否则一律 `not_resolved`（既不判劣化也不判无代价）。
严格真命中与可解码率一并报出，但**不参与**这条门的判定。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "training"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

MANIFEST = PROJECT_ROOT / "plans/manifests/r2_copy_surface_extension_v1.json"
MIN_GAP_TEXTS = 3


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_items(path: Path = MANIFEST) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload["dimensions"]["X"]["items"])


def run_arm(
    items: list[dict[str, Any]],
    checkpoint: Path,
    circuit: str | None,
    *,
    evidence_utf8_gate: bool = False,
) -> dict[str, Any]:
    """一臂：跑完 104 题，按产品 chat 协议取基底原始答复，统计表层三率。

    `evidence_utf8_gate`（PLAN-A-25）：只在评测期把复制回路的加性证据按 UTF-8 位置状态门控
    ——**默认 False ⇒ 与冻结链逐位相同**；开启走 `Taiji.set_copy_evidence_utf8_gate` 的运行时覆写
    （不改 config、不进 payload，所以"唯一变量＝门开/关"这条归因干净）。
    """

    from eval_taiji_r2_readout_retrain import build_ngram_model, well_formed
    from score_taiji_r2_copy_circuit_chat_cap import _answer_raw

    from api.seed_runtime import SeedRuntime

    runtime = SeedRuntime.load(checkpoint)
    if circuit:
        runtime.enable_copy_circuit(PROJECT_ROOT / circuit)
    if evidence_utf8_gate:
        runtime.model.substrate.set_copy_evidence_utf8_gate(True)
    ngram = build_ngram_model()

    texts: list[str] = []
    hits = 0
    rows: list[dict[str, Any]] = []
    for item in items:
        history: list[tuple[str, str]] = []
        turns = [str(turn) for turn in item["turns"]]
        answer = ""
        for index, turn in enumerate(turns):
            answer = _answer_raw(runtime, turn, history)
            texts.append(answer)
            if index + 1 < len(turns):
                history.append((turn, answer))
        tokens = [str(token) for token in item["expected_contains"]]
        hit = any(token in answer for token in tokens)
        hits += int(hit)
        rows.append(
            {
                "id": item["id"],
                "family": item["family"],
                "hit": hit,
                "answer": answer[:60],
                "well_formed": bool(well_formed(answer, ngram)),
            }
        )
    decodable = sum(1 for text in texts if "\ufffd" not in text)
    formed = sum(1 for text in texts if well_formed(text, ngram))
    #: **切尾感知**的诊断列（不改判定）：固定字节预算会把"缓冲切在字中间"也算成不可解码。
    #: 去掉末尾连续的 `\ufffd` 后仍含 `\ufffd` 才算**真的吐过非法字节**。
    #: 本仓已在两处独立踩过同一坑（F0 的 ON 臂 64 字节、本件 24 字节）——
    #: 只报 `utf8_decodable_rate` 会把度量缺陷读成模型缺陷。
    trimmed_clean = sum(1 for text in texts if "\ufffd" not in text.rstrip("\ufffd"))
    return {
        "circuit": circuit,
        "items": len(rows),
        "texts": len(texts),
        "well_formed_texts": formed,
        "well_formed_rate": round(formed / max(len(texts), 1), 4),
        "utf8_decodable_rate": round(decodable / max(len(texts), 1), 4),
        "utf8_decodable_trimmed_rate": round(trimmed_clean / max(len(texts), 1), 4),
        "illegal_after_tail_trim_rate": round(
            1.0 - trimmed_clean / max(len(texts), 1), 4
        ),
        "strict_hits": hits,
        "rows": rows,
    }


def rule_verdict(control: dict[str, Any], treated: list[dict[str, Any]]) -> dict[str, Any]:
    """判定：每个电路与对照比成句**文本条数**；两个电路同向且差 ≥3 条才下结论。"""
    per_circuit = []
    directions = []
    for arm in treated:
        gap = arm["well_formed_texts"] - control["well_formed_texts"]
        direction = "worse" if gap < 0 else ("better" if gap > 0 else "flat")
        directions.append(direction)
        per_circuit.append(
            {
                "circuit": arm["circuit"],
                "gap_texts": gap,
                "direction": direction,
                "meets_min_gap": abs(gap) >= MIN_GAP_TEXTS,
                "control_rate": control["well_formed_rate"],
                "treated_rate": arm["well_formed_rate"],
            }
        )
    worse = [row for row, direction in zip(per_circuit, directions) if direction == "worse"]
    better = [row for row, direction in zip(per_circuit, directions) if direction == "better"]
    if len(per_circuit) < 2:
        # "两电路同向"在只有一个电路时会**空洞地成立**（len(worse)==len(per_circuit)==1），
        # 那就等于用一次取数下判据——正是 §3 禁止的事，这里显式堵住。
        return {
            "min_gap_texts": MIN_GAP_TEXTS,
            "per_circuit": per_circuit,
            "status": "not_resolved",
            "verdict": f"not_resolved（独立取数只有 {len(per_circuit)} 次，判据要求 ≥2 次）",
        }
    if len(worse) == len(per_circuit) and all(row["meets_min_gap"] for row in worse):
        verdict = "表层劣化成立（两电路同向、每条差 ≥3）"
        status = "degraded"
    elif len(better) == len(per_circuit) and all(row["meets_min_gap"] for row in better):
        verdict = "表层改善（两电路同向、每条差 ≥3）"
        status = "improved"
    else:
        verdict = f"not_resolved（未同时满足『两电路同向』与『差 ≥{MIN_GAP_TEXTS} 条』）"
        status = "not_resolved"
    return {
        "min_gap_texts": MIN_GAP_TEXTS,
        "per_circuit": per_circuit,
        "status": status,
        "verdict": verdict,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/seed_beta.pt")
    parser.add_argument(
        "--circuit",
        action="append",
        default=[],
        help="治疗臂电路 payload，可给多次（多次＝多次独立取数）",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="题集 JSON（默认 v1 软干扰）。给 v2 时读数必须与 v1 **分开报**，"
        "合并总分就是把两个难度档摊平成一个（SPEC-A-21 §6、SPEC-A-22 §3）。",
    )
    parser.add_argument("--out-report", default=None)
    parser.add_argument(
        "--copy-evidence-utf8-gate",
        action="store_true",
        help="PLAN-A-25：把复制回路的加性证据按 UTF-8 位置状态门控（默认关 ⇒ 与冻结链逐位相同）",
    )
    args = parser.parse_args()

    checkpoint = PROJECT_ROOT / args.checkpoint
    sha_before = _sha256(checkpoint)
    manifest = MANIFEST if not args.manifest else Path(args.manifest)
    if not manifest.is_absolute():
        manifest = PROJECT_ROOT / manifest
    items = load_items(manifest)
    control = run_arm(items, checkpoint, None)
    treated = [
        run_arm(
            items,
            checkpoint,
            circuit,
            evidence_utf8_gate=bool(args.copy_evidence_utf8_gate),
        )
        for circuit in args.circuit
    ]
    report = {
        "format": "taiji-r2-copy-surface-extension-v1",
        "prereg": "plans/reference/SPEC-A-21_r2_surface_extension_prereg_20260925.md",
        #: 本轮（A2.5）的判读线与三档归因矩阵钉在 SPEC-A-22 §3/§9；仪器本身仍是 §A-21 那台。
        "judge_prereg": "plans/reference/SPEC-A-22_r2_a2_5_query_conditioned_selector_prereg_20260926.md",
        "manifest": manifest.relative_to(PROJECT_ROOT).as_posix(),
        "manifest_sha256": _sha256(manifest),
        "checkpoint": args.checkpoint,
        #: PLAN-A-25：门开/关必须落在件上，否则两份读数看起来像同一次实验。
        "copy_evidence_utf8_gate": bool(args.copy_evidence_utf8_gate),
        "control_no_circuit": control,
        "treated_arms": treated,
        "surface_verdict": rule_verdict(control, treated),
        "base_sha256_unchanged": _sha256(checkpoint) == sha_before,
    }
    #: 默认名跟着题集走——v2 的读数撞进 v1 那个文件名，比多打一个参数贵得多
    #: （§3 要求两集**分开报**，文件名是最容易被误读的那一层）。
    manifest_tag = manifest.stem.rsplit("_", 1)[-1]  # ..._v1 / ..._v2
    out = (
        Path(args.out_report)
        if args.out_report
        else PROJECT_ROOT / f"reports/taiji_r2_copy_surface_extension_{manifest_tag}_20260925.json"
    )
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    if out.exists():
        out = out.with_name(f"{out.stem}-{datetime.now(timezone.utc).strftime('%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "texts_per_arm": control["texts"],
                "control_wf": control["well_formed_texts"],
                "treated": [(arm["circuit"], arm["well_formed_texts"]) for arm in treated],
                "status": report["surface_verdict"]["status"],
                "base_unchanged": report["base_sha256_unchanged"],
                "out": out.relative_to(PROJECT_ROOT).as_posix(),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
