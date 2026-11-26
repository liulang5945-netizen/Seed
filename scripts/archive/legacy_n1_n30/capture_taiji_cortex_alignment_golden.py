"""C-2 前置：从**迁移前的当前实现**采集 tokenizer/对齐簇两成员的黄金向量（零训练）。

成员：`_reencode_domain_generation_context` 与 `_get_domain_to_general_alignment`。
夹具：`neuroplex.loader.create_cortex()`（生产构造路径，带 tokenizer）。
迁移后 `_cortex_alignment.py` 的同名纯函数必须逐组复现。
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

from neuroplex.loader import create_cortex  # noqa: E402


def main() -> int:
    cortex, _tokenizer = create_cortex()
    # `create_cortex` **不设 hub**（hub 由生产调用方设置，本机 fallback 模式下 hub=None）
    # ⇒ 夹具自装：general/zh 的 SentencePiece 模型都在 neuroplex/domains/ 下（已实测存在）。
    import sentencepiece as spm

    from neuroplex.resonance.translator import TokenizerHub

    def _load(rel: str):
        sp = spm.SentencePieceProcessor()
        sp.Load(str(PROJECT_ROOT / "neuroplex" / "domains" / rel))
        return sp

    general_sp = _load("general/sp_general.model")
    zh_sp = _load("zh/sp_zh.model")
    hub = TokenizerHub(general_tokenizer=general_sp)
    hub.register_domain("general", general_sp)
    hub.register_domain("zh", zh_sp)
    cortex.set_tokenizer_hub(hub)
    cortex.set_general_tokenizer(general_sp)

    reencode_cases = [
        {"name": "zh_name", "prefix": "我叫阿岩。", "generated_ids": list(zh_sp.encode("阿岩"))},
        {"name": "zh_empty_gen", "prefix": "我叫阿岩。", "generated_ids": []},
        {
            "name": "en_hello",
            "prefix": "Hello world. ",
            "generated_ids": list(zh_sp.encode("你好")),
        },
        {"name": "empty_prefix", "prefix": "", "generated_ids": list(zh_sp.encode("回答内容"))},
    ]

    reencode_rows = []
    for case in reencode_cases:
        out = cortex._reencode_domain_generation_context(
            case["prefix"], case["generated_ids"], zh_sp
        )
        reencode_rows.append(
            {
                "name": case["name"],
                "prefix": case["prefix"],
                "generated_ids": case["generated_ids"],
                "output_general_ids": list(out),
            }
        )
        print(
            json.dumps({"reencode": case["name"], "out_len": len(out)}, ensure_ascii=False),
            flush=True,
        )

    alignment = cortex._get_domain_to_general_alignment("zh", zh_sp)
    alignment_summary = {
        "n_entries": len(alignment),
        "sample": {str(k): alignment[k] for k in list(alignment)[:6]},
    }
    print(json.dumps({"alignment_entries": len(alignment)}, ensure_ascii=False), flush=True)

    payload = {
        "format": "cortex-alignment-golden-v1",
        "captured_from": "迁移前的当前实现（create_cortex 生产构造路径）",
        "note": "迁移后 _cortex_alignment.py 的同名纯函数必须逐组复现（alignment 需全表一致）",
        "reencode_rows": reencode_rows,
        "alignment_summary": alignment_summary,
        "alignment_full": alignment,
    }
    out = PROJECT_ROOT / "reports" / "cortex_alignment_golden_20260925.json"
    if out.exists():
        raise SystemExit(f"{out} already exists; 判决件不覆写")
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
