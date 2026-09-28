"""PLAN-A-30 乙档定价：解码侧的重复惩罚能不能"杀掉循环，又留住被记住的实体"。

`PLAN-A-30` §2b 把病定在**掩码 × 证据质量集中 ⇒ 同一个字连发**（`raw_masked` 带电路循环率 0.458）。
本件把新参数 `repetition_penalty` 拉通到那条链上，扫四个取值，量三件事一起看——
**循环率**（要它降）、**严格命中**（要它不降，否则惩罚只是把话换成了别的废话）、
**成句/可读性**（要它升）。只报"循环少了"是不合格的：把输出打成乱码也能让循环消失。

每条 penalty 用一个**新建载的 runtime**（挂载＝产品入口那条自动挂载之外的显式 opt-in，
证据门按裁定 (b) 开着）——同一条链上换惩罚值会共享生成状态，那就不是同一题面的对照。

用法：
    python scripts/training/measure_taiji_a30_repetition_penalty.py \
        --circuit output/taiji_r2_copy_circuit_chat/judge/circuit-final.pt --limit 24
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "training"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

MANIFEST = PROJECT_ROOT / "plans/manifests/r2_copy_surface_extension_v3_position_random.json"
PENALTIES = (0.0, 0.5, 1.0, 2.0)


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_arm(
    items: list[dict[str, Any]],
    checkpoint: Path,
    circuit: str,
    penalty: float,
) -> dict[str, Any]:
    from eval_taiji_r2_readout_retrain import build_ngram_model, well_formed
    from probe_taiji_a30_position_cycling import best_partial_period
    from score_taiji_r2_copy_circuit_chat_cap import _answer_raw

    from api.seed_runtime import SeedRuntime

    runtime = SeedRuntime.load(checkpoint)
    runtime.enable_copy_circuit(PROJECT_ROOT / circuit)
    substrate = runtime.model.substrate
    override = getattr(substrate, "_copy_evidence_utf8_gate_override", None)
    ngram = build_ngram_model()

    texts: list[str] = []
    rows: list[dict[str, Any]] = []
    hits = 0
    for item in items:
        history: list[tuple[str, str]] = []
        answer = ""
        turn_list = [str(t) for t in item["turns"]]
        for index, turn in enumerate(turn_list):
            answer = _answer_raw(
                runtime, turn, history, utf8_strict=True, repetition_penalty=penalty
            )
            texts.append(answer)
            if index + 1 < len(turn_list):
                history.append((turn, answer))
        tokens = [str(t) for t in item["expected_contains"]]
        hit = any(token in answer for token in tokens)
        hits += int(hit)
        period = best_partial_period(answer)
        rows.append({"id": item["id"], "hit": hit, "cyclic": bool(period["cyclic_tail"])})

    formed = sum(1 for t in texts if well_formed(t, ngram))
    cyclic = sum(1 for t in texts if best_partial_period(t)["cyclic_tail"])
    runs = []
    for text in texts:
        chars = [c for c in text if not c.isspace()]
        best = run = 1
        for a, b in zip(chars, chars[1:], strict=False):
            run = run + 1 if a == b else 1
            best = max(best, run)
        runs.append(best if chars else 0)
    return {
        "repetition_penalty": penalty,
        "gate_effective": bool(override),
        "items": len(items),
        "texts": len(texts),
        "strict_hits": hits,
        "well_formed_texts": formed,
        "cyclic_texts": cyclic,
        "cyclic_rate": round(cyclic / max(len(texts), 1), 4),
        "mean_longest_run": round(sum(runs) / max(len(runs), 1), 3),
        "max_longest_run": max(runs) if runs else 0,
        "per_item": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/seed_beta.pt")
    parser.add_argument("--circuit", required=True)
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--out-report", default=None)
    args = parser.parse_args()

    checkpoint = PROJECT_ROOT / args.checkpoint
    sha_before = _sha256(checkpoint)
    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = PROJECT_ROOT / manifest
    items = json.loads(manifest.read_text(encoding="utf-8"))["dimensions"]["X"]["items"][
        : args.limit
    ]

    arms = [run_arm(items, checkpoint, args.circuit, penalty) for penalty in PENALTIES]
    base = arms[0]
    report = {
        "format": "taiji-a30-repetition-penalty-v1",
        "prereg": "plans/reference/PLAN-A-30_surface_repetition_localization_20260928.md 乙档",
        "chain": "base_raw_bytes_with_product_mask",
        "checkpoint": args.checkpoint,
        "circuit": args.circuit,
        "items": len(items),
        "base_sha256_unchanged": _sha256(checkpoint) == sha_before,
        "arms": [{k: v for k, v in arm.items() if k != "per_item"} for arm in arms],
        #: 判读线（先看线再看数）：循环率要降，**且**严格命中不许掉过 ≥3 条的分辨率线；
        #: 只满足前者＝把输出打成另一种东西，不算修。
        "verdict": {
            arm["repetition_penalty"]: {
                "cyclic_delta_vs_base": arm["cyclic_texts"] - base["cyclic_texts"],
                "hits_delta_vs_base": arm["strict_hits"] - base["strict_hits"],
                "formed_delta_vs_base": arm["well_formed_texts"] - base["well_formed_texts"],
            }
            for arm in arms
        },
        "started_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    out = (
        Path(args.out_report)
        if args.out_report
        else PROJECT_ROOT / "reports/taiji_a30_repetition_penalty_20260928.json"
    )
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(report["arms"], ensure_ascii=False, indent=1))
    print("verdict:", json.dumps(report["verdict"], ensure_ascii=False))
    print("base_unchanged:", report["base_sha256_unchanged"], "| out:", out.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
