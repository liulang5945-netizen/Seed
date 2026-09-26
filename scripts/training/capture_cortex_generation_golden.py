"""B-3 C-4 黄金采集：记录**迁移前** `Cortex.generate()` 的端到端输出。

纪律（2026-09-26）：
* `generate()` 的合法参数见 cortex.py:1175-1207 —— **不存在** `no_repeat_ngram_size`，
  上一刀就是凭印象传了这个 kwarg 才采到三行错误串（TypeError）当作"黄金"。先盘签名再动。
* 每个用例**装两次、跑两遍**（每次 `torch.manual_seed(SEED)` 重新装配）⇒ 顺带自证
  端到端确定性；两遍不一致则标记 `deterministic=false`，该用例**不可**用作等价基准。
* `auto_capture=False` / `auto_memory=False` ⇒ 隔离全局 SleepEngine 与记忆检索的跨用例污染。
* 用例刻意覆盖待抽段的分支：top_k=0（全 softmax 采样）、repetition_penalty=1.0（跳过惩罚）、
  domain="zh"（no_repeat_ngram_size=4 而非 3）、temperature=1.0。
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from neuroplex.loader import assemble_cortex

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "reports" / "cortex_generation_golden_20260926.json"
SEED = 20260926

PROMPTS = [
    "你好，请自我介绍一下。",
    "什么是共振神经网络？",
    "请写一段关于春天的短文。",
]

CASES = [
    {"tag": "default", "kwargs": {"max_tokens": 24}},
    {"tag": "topk0", "kwargs": {"max_tokens": 24, "top_k": 0}},
    {"tag": "norep1", "kwargs": {"max_tokens": 24, "repetition_penalty": 1.0}},
    {"tag": "domain_zh", "kwargs": {"max_tokens": 24, "domain": "zh"}},
    {"tag": "temp1", "kwargs": {"max_tokens": 24, "temperature": 1.0}},
]


def _run(prompt: str, kwargs: dict) -> str:
    torch.manual_seed(SEED)
    cortex, _tok, _extra = assemble_cortex()
    call = dict(kwargs)
    call.setdefault("auto_capture", False)
    call.setdefault("auto_memory", False)
    try:
        return cortex.generate(prompt, **call)
    except Exception as exc:  # noqa: BLE001
        return f"__ERROR__{type(exc).__name__}: {exc}"


def main() -> None:
    rows = []
    for case in CASES:
        for prompt in PROMPTS:
            out1 = _run(prompt, case["kwargs"])
            out2 = _run(prompt, case["kwargs"])
            rows.append(
                {
                    "case": case["tag"],
                    "prompt": prompt,
                    "out": out1,
                    "deterministic": out1 == out2,
                    "is_error": out1.startswith("__ERROR__"),
                }
            )
            print(f"{case['tag']:<10} det={out1 == out2} err={out1.startswith('__ERROR__')} len={len(out1)}")

    payload = {
        "seed": SEED,
        "torch": torch.__version__,
        "note": "迁移前 Cortex.generate() 端到端输出；deterministic=false 的用例不可作等价基准。",
        "rows": rows,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    n_bad = sum(1 for r in rows if (not r["deterministic"]) or r["is_error"])
    print(f"rows={len(rows)} unusable={n_bad} -> {OUT}")


if __name__ == "__main__":
    main()
