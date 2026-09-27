"""F0 语言地板（PLAN-A-24 §5d 前置阶梯第 0 阶；零训练、零记忆、L0 层）。

**它回答的问题**：这条被所有 A 支线表层判据用来计分的**原始字节通道**，能不能输出合法中文？
不涉任何记忆检索——无电路、空库、`learn=False`、`use_memory=False`、贪心解码，
与既有全部表层读数同一条链（`_serialize` 之外直接 `substrate.generate`，64 字节、边界即停）。

**四类任务**（N 各 32）：
* T1a 语料同形问答续写：`问：<常见问题>\n答：`——被计分通道自己的格式；
* T1b 纯文本续写：常见简单陈述句后续写（QA 框架外的基础 LM 能力）；
* T2 听写拷贝：`问：请一字不差地重复以下内容：<X>\n答：`（指令跟随拷贝）；
* T3 上下文即时复述：`我的名字是明轩。我的名字是`（同提示词内的近距拷贝，无记忆部件）；
* T4 单轮即时问答：`问：我叫明轩。我的名字是什么？\n答：`（告知与提问同轮——F2 记忆任务
  去掉延迟与记忆件后的纯上下文版；它的成绩是任何记忆机制能加分的上限参照）。

**判读线（§5d 冻结）**：T1a 或 T1b 的**整句可解码率 ≥0.5** ⇒ 地板过线，F1 才准开跑；
否则 **冻结全部表层能力判据，问题退回 M5 主线（数据/目标/规模/读出）**。
成句率（n-gram）随表报出，不入判（rev7 未给它定线，如实报不硬凑）。

**诊断分解**（让"不过线"变成主线可执行的输入，而不是一句"不行"）：
* 首个非法字节的位置分布（0 位就断 vs 走了一段再断）；
* 前 8/16/32 字节前缀可解码率（地板是"全程碎"还是"走 N 步后碎"）；
* 每条输出里合法 CJK 字符数（内容里到底有多少真字）；
* 首字节类别（合法 UTF-8 首字节＝0x00-0x7F 或合法 lead byte 的比例）。
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

from taiji.utf8_state import trim_partial_tail  # noqa: E402

MAX_ANSWER_BYTES = 64
#: §5d 冻结的过线：整句可解码率 ≥0.5（T1a 或 T1b 任一）。
PASS_LINE = 0.5
N_PER_TASK = 32

COMMON_QUESTIONS = [
    "你叫什么名字？",
    "你今年多大？",
    "今天天气怎么样？",
    "你住在哪里？",
    "你喜欢吃什么？",
    "你的爱好是什么？",
    "你有几个兄弟姐妹？",
    "你做什么工作？",
    "你平时喜欢做什么？",
    "你家有几口人？",
    "你几点起床？",
    "你会上哪里去玩？",
    "你最喜欢的颜色是什么？",
    "你养宠物吗？",
    "你会说什么语言？",
    "你觉得这本书怎么样？",
    "今晚吃什么？",
    "周末你干什么？",
    "那条河叫什么名字？",
    "这家店几点开门？",
    "你去过北京吗？",
    "这个多少钱？",
    "明天会下雨吗？",
    "他为什么没来？",
    "你的老师是谁？",
    "你的家乡在哪里？",
    "你喜欢读书吗？",
    "你们学校大吗？",
    "这道题怎么做？",
    "那辆车是谁的？",
    "你昨晚睡得好吗？",
    "今天星期几？",
]

STATEMENTS = [
    "我叫天磊。",
    "我表哥叫静怡。",
    "我平时爱喝大麦茶。",
    "我家住在城南。",
    "今天是晴天。",
    "我哥哥是老师。",
    "我爱吃苹果。",
    "我们学校很大。",
    "这条河叫清江。",
    "我养了一只猫。",
    "妈妈做的饭很好吃。",
    "晚上我早点睡。",
    "弟弟今年七岁。",
    "爷爷喜欢下棋。",
    "这家店早上八点开门。",
    "我的家乡在江北。",
]

PEOPLE = ["天磊", "静怡", "志远", "小雨", "建华", "文博", "海燕", "国强"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _first_invalid_position(raw: bytes) -> int | None:
    """首个使 UTF-8 解码失败的位置；全程合法返回 None。"""
    for index in range(len(raw)):
        try:
            raw[: index + 1].decode("utf-8")
        except UnicodeDecodeError:
            # 单字节前缀解码失败可能只是"多字节字符切一半"，要按完整序列判定：
            # 从头解码到 index+1 失败，且把 index 处的字节当作截断起点重试仍失败，
            # 才算真非法。简化：用 errors="replace" 找第一个 \ufffd 的位置。
            break
    text = raw.decode("utf-8", errors="replace")
    position = text.find("\ufffd")
    return None if position < 0 else position


def _prefix_decodable(raw: bytes, n: int) -> bool:
    try:
        raw[:n].decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def _tail_trimmed_decode(raw: bytes) -> tuple[str, bool]:
    """丢掉"末尾被截断的多字节序列"之后再解码——把两类失败分开。

    动机（2026-09-27 PLAN-R2-01 两臂实测）：本仪器在 64 字节硬上限处截断，而纯 CJK 输出
    每 21 个汉字＝63 字节，第 64 字节必然是**下一个汉字的引导字节** ⇒ 整串解码失败。
    那是"缓冲切在字中间"，不是模型吐了非法字节。`decodable_whole`（§5d 冻结读数）
    **不改**，本函数只多给一列诊断，让"合法但被切断"与"真非法"分开报。
    """

    if not raw:
        return "", True
    trimmed = trim_partial_tail(raw)
    if not trimmed:
        #: 无任何前缀可解码 ⇒ **体内有非法字节**（不是尾部残缺）。此处不许返回空串当"干净"：
        #: 首版正是这么写的，把 OFF 臂的"整串非法"读成了 `illegal_trim=0.0`，与事实相反。
        return raw.decode("utf-8", errors="replace"), False
    try:
        return trimmed.decode("utf-8"), True
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace"), False


def _diagnose(raw: bytes) -> dict[str, Any]:
    text = raw.decode("utf-8", errors="replace")
    first_bad = _first_invalid_position(raw)
    trimmed_text, trimmed_ok = _tail_trimmed_decode(raw)
    trimmed_clean = trimmed_ok and "\ufffd" not in trimmed_text
    return {
        "decodable_whole": "\ufffd" not in text,
        #: 诊断列（不入判）：切掉末尾残缺序列后仍非法 ⇒ 模型真的吐过非法字节。
        "decodable_after_tail_trim": trimmed_clean,
        "tail_truncated_only": bool(trimmed_clean and "\ufffd" in text),
        "illegal_chars_after_tail_trim": trimmed_text.count("\ufffd"),
        "prefix8_ok": _prefix_decodable(raw, 8),
        "prefix16_ok": _prefix_decodable(raw, 16),
        "prefix32_ok": _prefix_decodable(raw, 32),
        "first_invalid_position": first_bad,
        "valid_cjk_chars": sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff"),
        "len_bytes": len(raw),
    }


def _generate_raw(runtime: Any, prompt: str, *, utf8_strict: bool = False) -> bytes:
    """与全部既有表层读数同链的原始字节获取（无电路、无记忆、贪心、边界停）。

    `utf8_strict=True`（SPEC-R2-02）＝**产品通道线**：解码带 UTF-8 硬约束。
    两线读数不许互换（§6 制度 4）：默认位＝模型地板；strict 位掩码是构造保证，
    该位的 verdict 只报"产品通道测量"，不参与地板过/不过的判定。
    """
    substrate = runtime.model.substrate
    return substrate.generate(
        prompt.encode("utf-8"),
        MAX_ANSWER_BYTES,
        stop_at_boundary=True,
        sample=False,
        use_memory=False,
        utf8_strict=utf8_strict,
    )


def _run_task(runtime: Any, task: str, *, utf8_strict: bool = False) -> dict[str, Any]:
    from eval_taiji_r2_readout_retrain import build_ngram_model, well_formed

    ngram = build_ngram_model()
    rows: list[dict[str, Any]] = []
    for index in range(N_PER_TASK):
        if task == "T1a_qa_continuation":
            prompt = f"问：{COMMON_QUESTIONS[index % len(COMMON_QUESTIONS)]}\n答："
            expected = None
        elif task == "T1b_plain_continuation":
            prompt = STATEMENTS[index % len(STATEMENTS)]
            expected = None
        elif task == "T2_dictation_copy":
            token = PEOPLE[index % len(PEOPLE)]
            prompt = f"问：请一字不差地重复以下内容：{token}\n答："
            expected = token
        elif task == "T3_incontext_immediate":
            token = PEOPLE[index % len(PEOPLE)]
            prompt = f"我的名字是{token}。我的名字是"
            expected = token
        elif task == "T4_single_turn_qa":
            token = PEOPLE[index % len(PEOPLE)]
            prompt = f"问：我叫{token}。我的名字是什么？\n答："
            expected = token
        else:
            raise ValueError(f"unknown task {task}")
        raw = _generate_raw(runtime, prompt, utf8_strict=utf8_strict)
        answer = raw.decode("utf-8", errors="replace")
        row = {"id": index, "prompt": prompt[:40], "answer": answer[:60]}
        row.update(_diagnose(raw))
        row["well_formed"] = bool(well_formed(answer, ngram))
        row["expected_hit"] = bool(expected in answer) if expected else None
        rows.append(row)

    def _rate(key: str) -> float:
        values = [row[key] for row in rows if row.get(key) is not None]
        return round(sum(1 for value in values if value) / max(len(values), 1), 4)

    first_bad_positions = [row["first_invalid_position"] for row in rows if row["first_invalid_position"] is not None]
    return {
        "task": task,
        "items": len(rows),
        "decodable_whole_rate": _rate("decodable_whole"),
        #: 诊断列（不入判；见 `_tail_trimmed_decode`）：把"被 64 字节上限切断"与
        #: "模型真吐非法字节"分开。判据仍只看 `decodable_whole_rate`。
        "decodable_after_tail_trim_rate": _rate("decodable_after_tail_trim"),
        "tail_truncated_only_rate": _rate("tail_truncated_only"),
        "illegal_after_tail_trim_rate": _rate("illegal_chars_after_tail_trim"),
        "prefix8_rate": _rate("prefix8_ok"),
        "prefix16_rate": _rate("prefix16_ok"),
        "prefix32_rate": _rate("prefix32_ok"),
        "well_formed_rate": _rate("well_formed"),
        "mean_valid_cjk_chars": round(
            sum(row["valid_cjk_chars"] for row in rows) / max(len(rows), 1), 2
        ),
        "first_invalid_position_median": (
            sorted(first_bad_positions)[len(first_bad_positions) // 2]
            if first_bad_positions
            else None
        ),
        "fully_valid_count": sum(1 for row in rows if row["decodable_whole"]),
        "expected_hit_rate": _rate("expected_hit") if rows[0].get("expected_hit") is not None else None,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/seed_beta.pt")
    parser.add_argument("--out-report", required=True)
    parser.add_argument(
        "--utf8-strict",
        action="store_true",
        help="产品通道线（带解码掩码，SPEC-R2-02）：可解码率按构造为 1，verdict 改报产品通道测量",
    )
    args = parser.parse_args()

    from api.seed_runtime import SeedRuntime

    checkpoint = PROJECT_ROOT / args.checkpoint
    sha_before = _sha256(checkpoint)
    runtime = SeedRuntime.load(checkpoint)
    substrate = runtime.model.substrate
    if substrate.copy_circuit is not None:
        print(json.dumps({"guard_ok": False, "error": "copy circuit mounted; F0 requires bare base"}, ensure_ascii=False))
        return 2

    tasks = [
        "T1a_qa_continuation",
        "T1b_plain_continuation",
        "T2_dictation_copy",
        "T3_incontext_immediate",
        "T4_single_turn_qa",
    ]
    results = [_run_task(runtime, task, utf8_strict=args.utf8_strict) for task in tasks]
    for result in results:
        first_bad = [
            row["first_invalid_position"]
            for row in result["rows"]
            if row["first_invalid_position"] is not None
        ]
        result["first_invalid_at_0_rate"] = round(
            sum(1 for p in first_bad if p == 0) / max(len(result["rows"]), 1), 4
        )

    basic = [r for r in results if r["task"] in ("T1a_qa_continuation", "T1b_plain_continuation")]
    best_basic = max(r["decodable_whole_rate"] for r in basic)
    verdict = "floor_pass" if best_basic >= PASS_LINE else "floor_fail"
    report = {
        "format": "taiji-f0-language-floor-v1",
        "prereg": "plans/reference/M5_R2_A_BRANCH_PLAN_REV2_20260926.md §5d（rev7 前置阶梯第 0 阶）",
        "checkpoint": args.checkpoint,
        "channel": (
            "masked product channel (utf8_strict=True) — 可解码按构造保证，本件 verdict 不是模型地板"
            if args.utf8_strict
            else "raw generate_input, no circuit, empty store, learn=False, use_memory=False, greedy, 64 bytes"
        ),
        "pass_line": {"decodable_whole": PASS_LINE, "scope": "T1a 或 T1b 任一"},
        "verdict": "product_channel_measurement" if args.utf8_strict else verdict,
        "masked_channel_note": (
            "掩码位下 prefix8/16/32 是「在固定字节位截断」的读数——掩码保证整串按字符对齐，"
            "截断点常落字符中间 ⇒ 前缀率失真为 0；判合法只看 decodable_whole（按构造=1）。"
            "有意义的增量读数是 wf/真汉字/命中：掩码修合法不修成句。"
            if args.utf8_strict
            else None
        ),
        "what_would_overturn": (
            "换可解码判定口径（如按字节而非整句）会改变过线判定——本件按 §5d 建议的整句口径冻结；"
            "采样换非贪心、或经语言器官通道，结果另计（器官通道当前 chat_enabled=False 替换为占位句，另行登记）"
        ),
        "tasks": results,
        "base_sha256_unchanged": _sha256(checkpoint) == sha_before,
    }
    report["instrument_guard"] = {
        "items_nonzero": all(r["items"] > 0 for r in results),
        "no_circuit": substrate.copy_circuit is None,
        "memory_untouched": getattr(substrate.memory, "write_count", 0) == 0,
        "base_unchanged": report["base_sha256_unchanged"],
    }

    out = Path(args.out_report)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    if out.exists():
        out = out.with_name(f"{out.stem}-{datetime.now(timezone.utc).strftime('%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ok = all(report["instrument_guard"].values())
    print(
        json.dumps(
            {
                "guard_ok": ok,
                "verdict": report["verdict"],
                "channel": "masked-product" if args.utf8_strict else "unmasked-model-floor",
                "tasks": [
                    {
                        "task": r["task"],
                        "decodable": r["decodable_whole_rate"],
                        "decodable_trim": r["decodable_after_tail_trim_rate"],
                        "tail_cut_only": r["tail_truncated_only_rate"],
                        "illegal_trim": r["illegal_after_tail_trim_rate"],
                        "prefix16": r["prefix16_rate"],
                        "wf": r["well_formed_rate"],
                        "cjk": r["mean_valid_cjk_chars"],
                        "hit": r["expected_hit_rate"],
                    }
                    for r in results
                ],
                "out": (
                    out.relative_to(PROJECT_ROOT).as_posix()
                    if out.is_relative_to(PROJECT_ROOT)
                    else str(out)
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
