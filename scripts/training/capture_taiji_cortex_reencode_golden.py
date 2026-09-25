"""B-2 前置：从**迁移前的当前实现**采集 `_reencode_domain_generation_context` 的黄金向量。

零训练；用基座的真 tokenizer（general + zh 域），对代表性 (prefix, generated_ids) 组合
记录输出 ID 列表。迁移后 `helpers.reencode_domain_generation_context(general_sp, ...)`
必须逐组复现。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts" / "training") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))

from api.seed_runtime import SeedRuntime  # noqa: E402

CHECKPOINT = PROJECT_ROOT / "checkpoints" / "seed_beta.pt"

CASES = [
    {"name": "zh_name", "prefix": "我叫阿岩。", "generated_text": "阿岩"},
    {"name": "zh_empty_gen", "prefix": "我叫阿岩。", "generated_text": ""},
    {"name": "en_hello", "prefix": "Hello world. ", "generated_text": "你好"},
    {"name": "empty_prefix", "prefix": "", "generated_text": "回答内容"},
    {"name": "long_prefix", "prefix": "第一段话。第二段话说了很多内容，包括一些细节描述。", "generated_text": "这是续写"},
]


def main() -> int:
    runtime = SeedRuntime.load(CHECKPOINT)
    taiji = runtime.model.substrate
    general_sp = taiji._general_sp
    zh_sp = taiji._tokenizer_hub.get_tokenizer("zh")

    rows = []
    for case in CASES:
        generated_ids = list(zh_sp.encode(case["generated_text"])) if case["generated_text"] else []
        out = taiji._reencode_domain_generation_context(
            case["prefix"], generated_ids, zh_sp
        )
        rows.append(
            {
                "name": case["name"],
                "prefix": case["prefix"],
                "generated_ids": generated_ids,
                "output_general_ids": list(out),
            }
        )
        print(json.dumps({"name": case["name"], "out_len": len(out)}, ensure_ascii=False), flush=True)

    payload = {
        "format": "cortex-reencode-golden-v1",
        "captured_from": "迁移前的当前实现（Cortex._reencode_domain_generation_context，基座 tokenizer）",
        "note": "迁移后 helpers.reencode_domain_generation_context(general_sp, prefix, generated_ids, decode_sp) 必须逐组复现",
        "rows": rows,
    }
    out = PROJECT_ROOT / "reports" / "cortex_reencode_golden_20260925.json"
    if out.exists():
        raise SystemExit(f"{out} already exists; 判决件不覆写")
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
