"""A30 §2bf-自写档：为语料问句生成**模型自己的短答**缓存（供主线分块喂法 `--answer-source self`）。

为什么需要它：§2bg 三线表里，**唯一三线全过的是自写答案形状**（formal self 48g3x6e），
而语料侧三种形状都卡在 L1 挂回路那一格（2 对参考 5）⇒ "重出默认基座"的候选必须能在主线规模上
复现自写形状。生成与训练解耦：本件只写缓存 jsonl（`{"question": …, "answer": …}`），
训练侧按问句查表喂入，缓存可复查、可重放。

生成口径与 on-policy 仪器**逐同**（`probe_taiji_a30_onpolicy_shape_pilot.generate_self_answers`）：
`SeedRuntime.load(base).chat(question, history=[], learn=False, max_length=96, repetition_penalty=2.0)`
——不挂回路、贪心、产品掩码默认（`chat` 默认 `utf8_strict=True`）。

守卫：①问句非空否则响亮停；②每条答案落盘前做 UTF-8 可解检查（含 U+FFFD 的记录在案但**照存**，
判读侧要点名面）；③幂等：输出文件已存在且 `--resume` 未给时拒绝覆盖；④计数自述
（请求 pairs、写出 lines、空答条数）随件落盘，供训练侧核对。

用法：
    python scripts/training/build_taiji_a30_self_answers.py \
        --base output/a31_chunked_self/checkpoint.pt --pairs 12000 \
        --out output/a30_self_answers/ca262807_self.jsonl \
        --out-report reports/taiji_a30_self_answers_ca262807_20261003.json

落盘规矩（2026-10-03 加，抄 `probe_taiji_a30_writeback_gate_shipping_face.py` 的同名修法）：
`--out-report` **必给**，且目标已存在 ⇒ 打 `[拒绝落盘]` 并返回 2。原先它是可选参数且缺省值硬钉在一份
**已入库**的读数文件上（`reports/taiji_a30_self_answers_build_20261001.json`）⇒ 任何人重跑而不带旗标，
就会在跑完那一刻无声覆盖那张表的唯一一次实测。重取必须换新文件名，让两件并存可比。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "training"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

DEFAULT_CORPUS = PROJECT_ROOT / "data" / "simple_zh" / "dialogue_extended_clean.jsonl"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rel(path: Path) -> str:
    """件内自述：仓内给相对 posix，仓外给绝对 posix。

    原来这里直接 `path.relative_to(PROJECT_ROOT)`，所以把 `--out` 指到仓外（scratch 落点的常规做法）
    会在**整轮生成跑完之后的报告步**抛 ValueError ⇒ 成果只在最后一步丢掉。
    """

    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def read_questions(path: Path, *, pairs: int) -> list[str]:
    """按语料顺序取前 `pairs` 条的**问句**（拆 `\\n答：`；拆不开响亮停）。"""

    questions: list[str] = []
    marker = "\n答："
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            text = raw.strip()
            if not text:
                continue
            payload = json.loads(text)
            content = payload["text"] if isinstance(payload, dict) and "text" in payload else text
            question, sep, _answer = content.partition(marker)
            if not sep:
                raise SystemExit(f"语料行里没有 {marker!r} 接缝：{content[:24]!r}")
            question = question.removeprefix("问：").strip()
            if not question:
                raise SystemExit(f"空问句：{content[:24]!r}")
            questions.append(question)
            if len(questions) >= pairs:
                break
    if len(questions) < pairs:
        raise SystemExit(f"语料不足：要 {pairs} 条问句，只有 {len(questions)} 条")
    return questions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="output/a26_p1/checkpoint.pt")
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--pairs", type=int, default=12000)
    parser.add_argument("--gen-max", type=int, default=96)
    parser.add_argument("--penalty", type=float, default=2.0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resume", action="store_true", help="已存在的输出里已有的问句跳过")
    parser.add_argument(
        "--out-report",
        required=True,
        help="读数件落点（必给；已存在即拒绝落盘，重取请换新文件名让两件并存可比）",
    )
    args = parser.parse_args()

    report_path = Path(args.out_report)
    if not report_path.is_absolute():
        report_path = PROJECT_ROOT / report_path
    if report_path.exists():
        print(f"[拒绝落盘] 报告目标已存在：{report_path}（换新文件名重取）", file=sys.stderr)
        return 2

    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    if out.exists() and not args.resume:
        raise SystemExit(f"refusing to overwrite {out}; pass --resume to append/skip existing")

    base = PROJECT_ROOT / args.base
    sha_base = _sha256(base)
    corpus = Path(args.corpus)
    if not corpus.is_absolute():
        corpus = PROJECT_ROOT / corpus

    from api.seed_runtime import SeedRuntime

    existing: dict[str, str] = {}
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                existing[str(row["question"])] = str(row["answer"])

    questions = read_questions(corpus, pairs=args.pairs)
    todo = [q for q in questions if q not in existing]
    runtime = SeedRuntime.load(base)

    started = time.perf_counter()
    written = 0
    empty = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        for index, question in enumerate(todo, start=1):
            answer = str(
                runtime.chat(
                    question,
                    history=[],
                    learn=False,
                    max_length=args.gen_max,
                    repetition_penalty=args.penalty,
                )
            )
            if not answer.strip():
                empty += 1
            handle.write(
                json.dumps({"question": question, "answer": answer}, ensure_ascii=False) + "\n"
            )
            handle.flush()
            written += 1
            if index % 500 == 0:
                print(
                    json.dumps(
                        {
                            "written": written,
                            "remaining": len(todo) - index,
                            "elapsed_seconds": round(time.perf_counter() - started, 1),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    del runtime

    report = {
        "format": "taiji-a30-self-answers-v1",
        "base": _rel(base),
        "base_sha256": sha_base,
        "base_sha256_unchanged": _sha256(base) == sha_base,
        "corpus": corpus.name,
        "pairs_requested": args.pairs,
        "lines_total": len(existing) + written,
        "lines_written_this_run": written,
        "empty_answers": empty,
        "gen_max": args.gen_max,
        "penalty": args.penalty,
        "out": _rel(out),
        "out_sha256_16": hashlib.sha256(out.read_bytes()).hexdigest()[:16],
        "started_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "guard": {
            "base_sha256_unchanged": _sha256(base) == sha_base,
            "question_count_matches": len(existing) + written <= args.pairs,
        },
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
