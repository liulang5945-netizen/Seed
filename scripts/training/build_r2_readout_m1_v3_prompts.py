"""构建 M1 功效补足用的**未见题面集 v3**（机械抽取，不靠人选）。

为什么用"机械抽取"而不是"我来写题"：本轮已经看过 CAP-43 上的结果。若由我来撰写新题面，
就有**无意中挑到对 A 臂有利的题**的空间——那正是"看到结果再挑样本"。所以题面由一条**冻结的规则**
从语料里抽出来：谁都能重跑这条规则得到同一份集合，我没有任何挑选余地。

**规则（写进预注册，执行前冻结）**：

1. 源：``data/p3b_all_fresh.jsonl``（29,876 行 / 52,280,150 符号，与其血缘清单逐位相符）。
2. **可用区间**：行内累计字节偏移（每行 = ``len(text.encode()) + 1``，与清单口径一致）
   **≥ 16,000,000** 的行 ⇒ 第 4026 行起、共 25,851 行。该区间在**三臂训练窗口
   [11,199,800, 27,199,800) 之外**（本集取自 16,000,000 符号之后），所以**三臂都没见过这些字节**。
3. **等距抽取**：``stride = max(1, 25851 // 160) = 161``，从第 4026 行起每隔 ``stride`` 行取一行，
   取满 **160** 条。
4. **题面** = 该行 ``text`` 的**前 32 个字符**（不是前若干字节——字符口径，避免切碎多字节）。

产物落 ``plans/manifests/r2_readout_m1_v3_prompts.json``，**已存在即拒跑**；封存 digest 写进预注册。
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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CORPUS = PROJECT_ROOT / "data" / "p3b_all_fresh.jsonl"
MANIFEST = PROJECT_ROOT / "plans" / "manifests" / "p3b_all_fresh_manifest.json"
OUT = PROJECT_ROOT / "plans" / "manifests" / "r2_readout_m1_v3_prompts.json"

#: 三臂训练窗口的末尾（符号数，p3b 口径）。抽题只取这之后再往后。
TRAINING_WINDOW_END = 16_000_000
#: 预注册的题面条数。
PROMPT_COUNT = 160
#: 预注册的题面长度（字符）。
PROMPT_CHARS = 32


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract() -> dict[str, Any]:
    rows: list[tuple[int, int, str]] = []  # (row_index, byte_offset, text)
    offset = 0
    with CORPUS.open("r", encoding="utf-8") as handle:
        for index, raw in enumerate(handle):
            try:
                text = json.loads(raw).get("text") or ""
            except ValueError:
                continue
            if offset >= TRAINING_WINDOW_END:
                rows.append((index, offset, str(text)))
            offset += len(str(text).encode("utf-8")) + 1

    if not rows:
        raise SystemExit("可用区间为空：抽题规则与语料不匹配，拒跑而不是产出空集")
    stride = max(1, len(rows) // PROMPT_COUNT)
    picked = rows[::stride][:PROMPT_COUNT]
    if len(picked) < PROMPT_COUNT:
        raise SystemExit(f"只抽到 {len(picked)} 条（要 {PROMPT_COUNT} 条），规则不成立，拒跑")

    prompts = [
        {
            "id": f"v3-{n:03d}",
            "row_index": row_index,
            "byte_offset": byte_offset,
            "prompt": text[:PROMPT_CHARS],
            "prompt_chars": len(text[:PROMPT_CHARS]),
        }
        for n, (row_index, byte_offset, text) in enumerate(picked)
    ]
    return {
        "format": "r2-readout-m1-v3-prompts-v1",
        "built_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "purpose": "M1 功效补足的未见题面集（机械抽取，无人工挑选）",
        "source": {
            "corpus": str(CORPUS.relative_to(PROJECT_ROOT).as_posix()),
            "corpus_sha256": sha256_of(CORPUS),
            "lineage_manifest": str(MANIFEST.relative_to(PROJECT_ROOT).as_posix()),
        },
        "rule": {
            "eligible_offset_min": TRAINING_WINDOW_END,
            "first_eligible_row_index": rows[0][0],
            "eligible_rows": len(rows),
            "stride": stride,
            "count": PROMPT_COUNT,
            "prompt_chars": PROMPT_CHARS,
            "note": (
                "行内累计偏移 = Σ(len(text.encode())+1)，与 p3b 血缘清单的 emitted_symbols 口径一致；"
                "题面取前 N 个字符（字符口径，不切多字节）"
            ),
        },
        "outside_training_window": True,
        "prompts": prompts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    if out.exists():
        parser.error(f"{out} already exists; 封存的题面集不覆写（要换就换文件名并另立预注册）")

    payload = extract()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(out),
                "sha256": sha256_of(out),
                "count": len(payload["prompts"]),
                "first_eligible_row_index": payload["rule"]["first_eligible_row_index"],
                "stride": payload["rule"]["stride"],
                "sample": [item["prompt"] for item in payload["prompts"][:3]],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
