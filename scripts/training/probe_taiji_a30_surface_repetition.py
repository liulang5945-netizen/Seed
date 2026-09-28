"""PLAN-A-30 战线一：为什么"挂上复制回路"之后，**表层成句反而掉**（48/312 → 20/312）。

`PLAN-A-28` §4b 的量是聚合的（命中 0→35 而成句 48→20），但它说不出**掉在哪一层**：
①回路把基底原始输出打成复读（那表层只是继承了坏输入），还是
②基底输出没变坏，是**语言器官的成句计划**在电路在场时救不回它？
本探针把两条链**分开各跑一臂**（同题面、同协议、同电路，唯一变量＝电路在场/不在场），
每条文本除 `well_formed` 外再量三件**重复度**指标，并把逐题面摆出来供人读。

四臂设计（不重抄生成链，全部走产品入口 `SeedRuntime`）：
`raw`＝`_answer_raw`（基底原始字节链，SPEC-A-21 那条冻结链）；
`surface`＝`runtime.chat(..., learn=False, repetition_penalty=0.0)`（产品表层链，过语言器官＋SPEC-R2-02 掩码）。
每臂一个新载入的 runtime——同一条链上先 `_answer_raw` 再 `chat` 会让表层多走一步生成，
那不是"两条面的对照"，是第三条面。

用法：
    python scripts/training/probe_taiji_a30_surface_repetition.py \
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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))

MANIFEST = PROJECT_ROOT / "plans/manifests/r2_copy_surface_extension_v3_position_random.json"
#: 三条链＝两端的原始/表层，再加**只把掩码挪到原始链上**那一手——这样"掩码"与"器官"才分得开
CHAINS = ("raw", "raw_masked", "surface")


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload_sha(value) -> str | None:
    """回路 payload 的摘要：件里只记路径不够（PLAN-A-24 rev22 那条复现性债）。"""

    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return _sha256(path)


def _repetition(text: str) -> dict[str, Any]:
    """三个重复度指标：唯一字符比、最长连写段、2-gram 重复率。

    只用"是不是合法 UTF-8"分不开"复读"与"乱码"——`哥哥哥哥…` 每个字都合法，
    而本症状要的正是把这两种坏分开（`PLAN-A-28` §4b 的抽样看着就是前者）。
    """
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return {"distinct_ratio": None, "longest_run": 0, "bigram_repeat": None, "length": 0}
    runs = 1
    best_run = 1
    for a, b in zip(chars[:-1], chars[1:], strict=True):
        if a == b:
            runs += 1
            best_run = max(best_run, runs)
        else:
            runs = 1
    bigrams = ["".join(pair) for pair in zip(chars[:-1], chars[1:], strict=True)]
    repeat = 1.0 - len(set(bigrams)) / len(bigrams) if bigrams else 0.0
    return {
        "distinct_ratio": round(len(set(chars)) / len(chars), 4),
        "longest_run": best_run,
        "bigram_repeat": round(repeat, 4),
        "length": len(chars),
    }


def run_arm(
    items: list[dict[str, Any]],
    checkpoint: Path,
    circuit: str | None,
    chain: str,
    *,
    max_bytes: int | None = None,
    penalty: float = 0.0,
) -> dict:
    from eval_taiji_r2_readout_retrain import build_ngram_model, well_formed
    from score_taiji_r2_copy_circuit_chat_cap import MAX_ANSWER_BYTES, _answer_raw

    from api.seed_runtime import SeedRuntime

    runtime = SeedRuntime.load(checkpoint)
    mount = "none"
    if circuit:
        runtime.enable_copy_circuit(PROJECT_ROOT / circuit)
        mount = "enable_copy_circuit"
    elif runtime.model.substrate.copy_circuit is not None:
        mount = "envelope_auto_mount"
    ngram = build_ngram_model()

    #: 两条链各自的**历史预算**：原始链 64（SPEC-A-21 冻结面）、表层链 256（`chat()` 默认）。
    #: §2h 查出的口径缺陷就在这儿——跨链比较时"链"与"预算"两个变量一起动了。
    #: 给了 `--max-bytes` 就把两条链拨到同一个预算，那才是"只差一条链"的对照。
    budget = (
        max_bytes if max_bytes is not None else (256 if chain == "surface" else MAX_ANSWER_BYTES)
    )

    texts: list[str] = []
    rows: list[dict[str, Any]] = []
    hits = 0
    for item in items:
        history: list[tuple[str, str]] = []
        answer = ""
        for index, turn in enumerate([str(t) for t in item["turns"]]):
            answer = (
                # 钉旧默认位（产品默认 2026-09-28 起 2.0）：本探针的六臂读数全是在 0.0 上取的。
                runtime.chat(
                    turn,
                    history=history,
                    learn=False,
                    repetition_penalty=penalty,
                    max_length=budget,
                )
                if chain == "surface"
                else _answer_raw(
                    runtime,
                    turn,
                    history,
                    utf8_strict=(chain == "raw_masked"),
                    repetition_penalty=penalty,
                    max_bytes=budget,
                )
            )
            texts.append(answer)
            if index + 1 < len(item["turns"]):
                history.append((turn, answer))
        tokens = [str(t) for t in item["expected_contains"]]
        hit = any(token in answer for token in tokens)
        hits += int(hit)
        rows.append({"id": item["id"], "hit": hit, "answer": answer[:80], **_repetition(answer)})
    formed = sum(1 for t in texts if well_formed(t, ngram))
    measured = [m for m in (_repetition(t) for t in texts) if m["distinct_ratio"] is not None]
    return {
        "chain": chain,
        "mount": mount,
        "max_bytes": budget,
        "repetition_penalty": penalty,
        "items": len(rows),
        "texts": len(texts),
        "strict_hits": hits,
        "well_formed_texts": formed,
        "mean_distinct_ratio": round(sum(m["distinct_ratio"] for m in measured) / len(measured), 4),
        "mean_bigram_repeat": round(sum(m["bigram_repeat"] for m in measured) / len(measured), 4),
        "texts_with_run_ge_4": sum(1 for m in measured if m["longest_run"] >= 4),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/seed_beta.pt")
    parser.add_argument("--circuit", default=None)
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=None,
        help="把两条链的生成预算拨成同一个值（不给＝各按历史预算：原始 64／表层 256）",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=0.0,
        help="钉在 0.0 才能复现已入库读数；产品默认位（2.0）要显式点名",
    )
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

    arms = [
        run_arm(
            items,
            checkpoint,
            args.circuit,
            chain,
            max_bytes=args.max_bytes,
            penalty=args.repetition_penalty,
        )
        for chain in CHAINS
    ]
    no_circuit = [
        run_arm(
            items,
            checkpoint,
            None,
            chain,
            max_bytes=args.max_bytes,
            penalty=args.repetition_penalty,
        )
        for chain in CHAINS
    ]
    report = {
        "format": "taiji-a30-surface-repetition-v1",
        "prereg": "PLAN-A-30 战线一（表层成句为何被回路打下去）",
        "manifest": manifest.name,
        "items": len(items),
        "checkpoint": args.checkpoint,
        "circuit": args.circuit,
        "circuit_sha256": _payload_sha(args.circuit),
        "max_bytes": args.max_bytes,
        "repetition_penalty": args.repetition_penalty,
        "base_sha256_unchanged": _sha256(checkpoint) == sha_before,
        "arms": [{k: v for k, v in arm.items() if k != "rows"} for arm in arms + no_circuit],
        "per_arm_rows": {
            f"{arm['chain']}|{arm['mount']}": arm["rows"] for arm in arms + no_circuit
        },
        "started_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    out = (
        Path(args.out_report)
        if args.out_report
        else PROJECT_ROOT / "reports/taiji_a30_surface_repetition_20260928.json"
    )
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "items": len(items),
                "arms": [
                    {
                        "chain": a["chain"],
                        "mount": a["mount"],
                        "budget": a["max_bytes"],
                        "penalty": a["repetition_penalty"],
                        "hits": a["strict_hits"],
                        "formed": a["well_formed_texts"],
                        "distinct": a["mean_distinct_ratio"],
                        "bigram_repeat": a["mean_bigram_repeat"],
                        "run>=4": a["texts_with_run_ge_4"],
                    }
                    for a in report["arms"]
                ],
                "base_unchanged": report["base_sha256_unchanged"],
                "out": (
                    out.relative_to(PROJECT_ROOT).as_posix()
                    if out.is_relative_to(PROJECT_ROOT)
                    else out.as_posix()
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
