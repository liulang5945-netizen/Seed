"""构建 P3b 的**对话侧重**训练子集（只读原始语料，产物另存 + manifest 归档）。

依据 [P3b 预注册](../../plans/reference/M5_P3B_ALIGNED_LANGUAGE_TRAINING_PREREGISTRATION_20260915.md) §2：
数据 = `data/simple_zh/simple_zh_texts.jsonl` 中**多角色文档**构成的子集，
且**筛选规则、命中行数与产物 sha256 必须随报告归档**。

判据（实测语料是「角色名：内容」的多角色脚本，不是问答对）：
- 行首形如 `角色名：`（1–12 个字符，含中英文/括号/间隔号）；
- **不同角色名 ≥ 2** 才算多角色；
- 排除元数据角色（作者 / 来源 / 标题 / 译者 / 出处）。

只读原语料；输出写入被 gitignore 的 `data/`，manifest 写入 `plans/manifests/`。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "simple_zh" / "simple_zh_texts.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "p3b_dialogue_subset.jsonl"
DEFAULT_MANIFEST = PROJECT_ROOT / "plans" / "manifests" / "p3b_dialogue_subset_manifest.json"

SPEAKER_PATTERN = r"(?:^|\n)\s*([\u4e00-\u9fffA-Za-z0-9_（）()·]{1,12})："
SPEAKER = re.compile(SPEAKER_PATTERN)
META_SPEAKERS = frozenset({"作者", "来源", "标题", "译者", "出处", "编辑", "校对"})
MIN_SPEAKERS = 2


def _rel(path: Path) -> str:
    """仓库内用相对路径（便于归档），仓库外（如临时试跑）退回绝对路径。"""

    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _speakers(text: str) -> set[str]:
    """返回文档里出现过的、**排除元数据**的角色名。"""

    return {match.group(1) for match in SPEAKER.finditer(text)} - META_SPEAKERS


def build(
    source: Path = DEFAULT_SOURCE,
    output: Path = DEFAULT_OUTPUT,
    manifest: Path = DEFAULT_MANIFEST,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    rows = kept = 0
    kept_bytes = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        source.open(encoding="utf-8") as handle,
        output.open("w", encoding="utf-8", newline="\n") as sink,
    ):
        for line in handle:
            rows += 1
            if limit is not None and rows > limit:
                break
            stripped = line.rstrip("\n")
            if not stripped.strip():
                continue
            try:
                document = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            text = document.get("text")
            if not isinstance(text, str):
                continue
            if len(_speakers(text)) < MIN_SPEAKERS:
                continue
            payload = json.dumps({"text": text}, ensure_ascii=False) + "\n"
            sink.write(payload)
            kept += 1
            kept_bytes += len(payload.encode("utf-8"))

    digest = hashlib.sha256()
    with output.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)

    meta: dict[str, Any] = {
        "format": "p3b-dialogue-subset-manifest-v1",
        "source": _rel(source),
        "source_bytes": source.stat().st_size,
        "source_rows_scanned": rows,
        "truncated_by_limit": limit is not None and rows > limit,
        "rule": {
            "min_distinct_speakers": MIN_SPEAKERS,
            "speaker_pattern": SPEAKER_PATTERN,
            "excluded_speakers": sorted(META_SPEAKERS),
        },
        "kept_rows": kept,
        "kept_bytes": kept_bytes,
        "output": _rel(output),
        "output_sha256": digest.hexdigest(),
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="构建 P3b 对话侧重训练子集")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--limit", type=int, default=None, help="只扫描前 N 行（试跑）")
    args = parser.parse_args(argv)

    meta = build(args.source, args.output, args.manifest, limit=args.limit)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
