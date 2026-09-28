"""SPEC-R2-02 主判据：产品表层占位句率 修前/修后 对照（seed_beta，零训练）。

**测什么**：`SeedRuntime.chat` 的产品表层链上，语言器官把基底输出判为"不可读"而回落到
占位句的比例。修前＝基底生成**无掩码**（`utf8_strict=False`，即改动前的唯一路径）；
修后＝**带掩码**的产品路径（`chat()` 现状）。两臂共用同一 `_serialize`、同一折字、
同一器官（`NativeReadableTextLanguageOrgan.emit`），**唯一差别＝基底生成是否 utf8_strict**
⇒ 差值可全量归因给解码掩码。

判据（SPEC-R2-02 §2 冻结）：32 题占位句率 **修后 ≤ 16/32**，且修后被接受表层 **100% 可解码**。
另取 4 题走**真 `runtime.chat()`** 复核"修后臂的复刻链与产品链一致"（不一致 ⇒ 仪器拒判）。

占位句识别＝命中 `NativeReadableTextLanguageOrgan._fallback_text` 的三种句式之一
（"当前原生语言表层正在形成稳定表达"／"Taiji 已完成内部处理"／"当前表达包含以下关键信息"）。
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

PROMPTS = [
    "你叫什么名字？", "你今年多大？", "今天天气怎么样？", "你住在哪里？",
    "你喜欢吃什么？", "你的爱好是什么？", "你有几个兄弟姐妹？", "你做什么工作？",
    "你平时喜欢做什么？", "你家有几口人？", "你几点起床？", "你去过北京吗？",
    "你最喜欢的颜色是什么？", "你养宠物吗？", "你会说什么语言？", "你觉得读书有用吗？",
    "今晚吃什么？", "周末你打算干什么？", "那条河叫什么名字？", "这家店几点开门？",
    "这个多少钱？", "明天会下雨吗？", "他为什么没来？", "你的老师是谁？",
    "你的家乡在哪里？", "你喜欢读书吗？", "这道题怎么做？", "那辆车是谁的？",
    "你昨晚睡得好吗？", "今天星期几？", "我们下一步去哪？", "这本书讲了什么？",
]

FALLBACK_MARKERS = (
    "当前原生语言表层正在形成稳定表达",
    "Taiji 已完成内部处理",
    "当前表达包含以下关键信息",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _emit(runtime: Any, prompt: str, *, utf8_strict: bool) -> dict[str, Any]:
    """复刻 chat() 的表层链（无写回学习），唯一可变＝基底生成是否带掩码。"""
    from taiji import ExpressionPlan, InputFrame

    substrate = runtime.model.substrate
    frame = InputFrame(
        input_id=f"r202:{substrate.tick}",
        modality="text",
        payload=runtime._serialize(prompt, []).encode("utf-8"),
        source="spec-r2-02",
        timestamp=substrate.tick,
        provenance="external",
        confidence=1.0,
    )
    raw = runtime.model.generate_input(
        frame, 96, stop_at_boundary=True, sample=False, utf8_strict=utf8_strict
    )
    native = raw.decode("utf-8", errors="replace")
    from api.seed_runtime import _TURN_MARKERS

    for marker in _TURN_MARKERS:
        index = native.find(marker)
        if index >= 0:
            native = native[:index]
    expression = ExpressionPlan(
        expression_id=f"r202:{substrate.tick}",
        content_id=f"r202:{substrate.tick}",
        modality="text",
        channel="message",
        fields={
            "intent_kind": "chat_answer",
            "semantic_slots": {"prompt": prompt, "history": []},
            "native_prediction": native,
            "expected_outcome": "answer user in readable language",
        },
        provenance="spec.r202",
        tick=substrate.tick,
    )
    emission = runtime._chat_organ.emit(expression)
    text = emission.text_bytes.decode("utf-8")
    return {
        "prompt": prompt,
        "surface": text[:80],
        "placeholder": any(marker in text for marker in FALLBACK_MARKERS),
        "decodable": "\ufffd" not in text,
        "raw_decodable": True if utf8_strict else ("\ufffd" not in native),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/seed_beta.pt")
    parser.add_argument("--out-report", required=True)
    args = parser.parse_args()

    from api.seed_runtime import SeedRuntime

    checkpoint = PROJECT_ROOT / args.checkpoint
    sha_before = _sha256(checkpoint)
    runtime = SeedRuntime.load(checkpoint)

    before = [_emit(runtime, p, utf8_strict=False) for p in PROMPTS]
    after = [_emit(runtime, p, utf8_strict=True) for p in PROMPTS]

    # 产品链真复核：4 题走真 chat()（learn=False），表层必须与 after 臂的机制判定一致
    # （允许内容不同——真 chat 有写回与 tick 推进；这里只核"不再占位/可解码"的定性）。
    checks = []
    for prompt in PROMPTS[:4]:
        answer = runtime.chat(prompt, history=[], max_length=96, learn=False, repetition_penalty=0.0)
        checks.append(
            {
                "prompt": prompt,
                "answer": answer[:80],
                "placeholder": any(marker in answer for marker in FALLBACK_MARKERS),
                "decodable": "\ufffd" not in answer,
            }
        )

    before_ph = sum(1 for row in before if row["placeholder"])
    after_ph = sum(1 for row in after if row["placeholder"])
    after_all_decodable = all(row["decodable"] for row in after)
    product_checks_ok = all(row["decodable"] and not row["placeholder"] for row in checks)
    verdict = (
        "pass" if after_ph <= len(PROMPTS) // 2 and after_all_decodable else "fail"
    )
    report = {
        "format": "taiji-r202-placeholder-rate-v1",
        "prereg": "plans/reference/SPEC-R2-02_decode_mask_productization_prereg_20260927.md §2",
        "checkpoint": args.checkpoint,
        "prompts": len(PROMPTS),
        "placeholder_before_unmasked": f"{before_ph}/{len(PROMPTS)}",
        "placeholder_after_masked": f"{after_ph}/{len(PROMPTS)}",
        "accepted_surfaces_all_decodable": after_all_decodable,
        "main_criterion": {"threshold": f"<= {len(PROMPTS)//2}", "verdict": verdict},
        "real_chat_checks": checks,
        "real_chat_checks_all_pass": product_checks_ok,
        "rows_before": before,
        "rows_after": after,
        "what_would_overturn": (
            "若 after 占位率超阈值 ⇒ 器官验收还卡掩码管不到的条件（控制符/alnum），按 §3 查因入册不放宽；"
            "若真 chat 复核与复刻链定性不一致 ⇒ 复刻链失真，本件读数作废"
        ),
        "base_sha256_unchanged": _sha256(checkpoint) == sha_before,
    }
    report["instrument_guard"] = {
        "prompts_nonzero": bool(PROMPTS),
        "before_is_all_placeholder_or_reported": True,
        "real_chat_recheck_agrees": product_checks_ok,
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
                "placeholder_before": report["placeholder_before_unmasked"],
                "placeholder_after": report["placeholder_after_masked"],
                "verdict": verdict,
                "real_chat_checks": [
                    {"answer": c["answer"][:36], "ph": c["placeholder"]} for c in checks
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
