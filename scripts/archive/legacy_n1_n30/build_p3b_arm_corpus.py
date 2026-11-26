"""构建 P3b **两臂**的训练语料：同一个未见过源窗口，只差行筛选规则。

依据 [P3b 预注册](../../plans/reference/M5_P3B_ALIGNED_LANGUAGE_TRAINING_PREREGISTRATION_20260915.md)
§2 / §2.2，以及 [双臂新颖度配对修订](../../plans/reference/M5_P3B_NOVELTY_MATCHED_ARMS_AMENDMENT_20260915.md)。

**为什么两臂都要切片**：起点 `seed_beta.pt`（`tick = 16,000,000`）的上一环
`resumed_seed_corpus.pt`（`tick = 4,800,200`）指纹指向**另一个**语料
（`dialogue_extended_clean.jsonl`）⇒ 状态在 `simple_zh_texts.jsonl` 上已经顺序消耗过
**前 11,199,800 个符号**。实测（2026-09-15，见修订文档 §2）：

- 按预注册原文用"整条原始流"作对照臂 ⇒ 对照臂 **23.69%** 的预算在重放已见过的数据；
- 而"对话子集"**也有 22.27%** 来自同一段已见区域（源前 2,893 行里 **93.7%** 都合格）；
- 只切对照臂反而打断配对（0% 对 22.27%）⇒ **两臂必须从同一处未见过窗口开始**。

本脚本因此用**同一份代码、同一个窗口起点**产出两臂语料，只有 `--rule` 不同：

- `dialogue`：保留多角色文档（**≥2** 个 `角色：` 前缀，排除元数据角色）——实验臂；
- `all`：保留全部行——对照臂（原始分布）。

符号计数与训练器一致（每行 1 个 `boundary_symbol` + 文档 UTF-8 字节数），因为"已见前缀"
就是按训练器的口径消耗的。只读原语料；输出写入被 gitignore 的 `data/`，manifest 写入
`plans/manifests/`。构建完成后核对两臂共享前缀，超过预算的 1% 即拒绝产出。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "simple_zh" / "simple_zh_texts.jsonl"
MANIFEST_DIR = PROJECT_ROOT / "plans" / "manifests"

#: 48 h 档预算（`train_p3b_aligned.BUDGET_TIERS`），多留 5M 符号余量以免 epochs=1 提前耗尽。
DEFAULT_MAX_SYMBOLS = 52_280_000
#: 两臂共享前缀的硬上限：预算的 1%。超过即说明两臂其实读的是同一条流。
MAX_SHARED_PREFIX_SYMBOLS = 472_800
BOUNDARY = 256  # TaijiConfig().boundary_symbol，每行额外一个符号

SPEAKER_PATTERN = r"(?:^|\n)\s*([\u4e00-\u9fffA-Za-z0-9_（）()·]{1,12})："
SPEAKER = re.compile(SPEAKER_PATTERN)
META_SPEAKERS = frozenset({"作者", "来源", "标题", "译者", "出处", "编辑", "校对"})
MIN_SPEAKERS = 2
RULES = ("dialogue", "all")


def _rel(path: Path) -> str:
    """仓库内用相对路径（便于归档），仓库外（如临时试跑）退回绝对路径。"""

    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def row_symbol_count(text: str) -> int:
    """一个 jsonl 行喂给训练器会消耗多少符号（1 个边界符 + UTF-8 字节数）。"""

    return 1 + len(text.encode("utf-8"))


def speakers(text: str) -> set[str]:
    """文档里出现过的、**排除元数据**的角色名。"""

    return {match.group(1) for match in SPEAKER.finditer(text)} - META_SPEAKERS


def qualifies(text: str, rule: str) -> bool:
    """两臂唯一的差别：这一行进不进这一臂的流。"""

    if rule == "all":
        return True
    if rule == "dialogue":
        return len(speakers(text)) >= MIN_SPEAKERS
    raise SystemExit(f"unknown rule {rule!r}, expected one of {RULES}")


def seen_prefix_symbols(
    *,
    resume_tick: int,
    prior_tick: int,
    resume_corpus_name: str,
    prior_corpus_name: str,
    corpus_name: str,
) -> int:
    """起点检查点在 ``corpus_name`` 上已经顺序消耗了多少个 leading 符号。

    只有"上一环指纹是**别的**语料"时切换点才等于 ``prior_tick``；否则已见前缀比这更长，
    而信封里只记 (name, bytes) 不记偏移，无法从这里定界 ⇒ 拒绝推断。
    """

    if corpus_name == prior_corpus_name:
        raise SystemExit(
            f"cannot bound the seen prefix: the prior checkpoint already trained on {corpus_name}; "
            "its envelope records no offset, so more lineage is required before slicing"
        )
    if resume_corpus_name != corpus_name:
        raise SystemExit(
            f"start checkpoint was trained on {resume_corpus_name}, not {corpus_name}; "
            "the slice would not match what the state has already seen"
        )
    if prior_tick > resume_tick:
        raise SystemExit(f"prior tick {prior_tick} is ahead of resume tick {resume_tick}")
    return int(resume_tick) - int(prior_tick)


def _documents(path: Path) -> Iterator[tuple[int, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        for index, line in enumerate(handle):
            stripped = line.rstrip("\r\n")
            if not stripped.strip():
                continue
            try:
                document = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            text = document.get("text")
            if isinstance(text, str):
                yield index, text


def shared_prefix_symbols(left: Path, right: Path) -> int:
    """两臂语料**按顺序**共享的 leading 符号数（训练器的计数口径）。"""

    shared = 0
    for (_, a), (_, b) in zip(_documents(left), _documents(right), strict=False):
        if a != b:
            break
        shared += row_symbol_count(a)
    return shared


def build(
    source: Path,
    output: Path,
    *,
    rule: str,
    skip_symbols: int,
    max_symbols: int = DEFAULT_MAX_SYMBOLS,
    other_arm: Path | None = None,
) -> dict[str, Any]:
    if skip_symbols < 0:
        raise SystemExit("skip_symbols must be non-negative")
    if max_symbols <= 0:
        raise SystemExit("max_symbols must be positive")

    scanned = emitted = dropped = dialogue_rows = 0
    offset = 0
    emitted_symbols = 0
    emitted_bytes = 0
    first_row: int | None = None
    last_row: int | None = None
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as sink:
        for index, text in _documents(source):
            scanned += 1
            symbols = row_symbol_count(text)
            if offset >= skip_symbols:
                if qualifies(text, rule):
                    if first_row is None:
                        first_row = index
                    payload = json.dumps({"text": text}, ensure_ascii=False) + "\n"
                    sink.write(payload)
                    emitted += 1
                    last_row = index
                    emitted_symbols += symbols
                    emitted_bytes += len(payload.encode("utf-8"))
                    if len(speakers(text)) >= MIN_SPEAKERS:
                        dialogue_rows += 1
                    if emitted_symbols >= max_symbols:
                        break
                else:
                    dropped += 1
            offset += symbols
    if emitted_symbols < max_symbols:
        output.unlink(missing_ok=True)
        raise SystemExit(
            f"source exhausted after {emitted_symbols} emitted symbols (< {max_symbols}); "
            "the slice cannot cover the budget, so the arm would silently train on less data"
        )

    digest = hashlib.sha256()
    with output.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)

    overlap: dict[str, Any] | None = None
    if other_arm is not None and other_arm.exists():
        shared = shared_prefix_symbols(output, other_arm)
        overlap = {
            "against": _rel(other_arm),
            "shared_prefix_symbols": shared,
            "cap": MAX_SHARED_PREFIX_SYMBOLS,
            "passed": shared <= MAX_SHARED_PREFIX_SYMBOLS,
        }
        if not overlap["passed"]:
            output.unlink(missing_ok=True)
            raise SystemExit(
                f"the two arms still share {shared} leading symbols "
                f"(cap {MAX_SHARED_PREFIX_SYMBOLS}): the row filter did not separate the streams, "
                "so there is no distribution contrast left to read"
            )

    return {
        "format": "p3b-arm-corpus-manifest-v1",
        "purpose": f"P3b {rule} arm corpus: source rows past the already-seen prefix",
        "rule": rule,
        "source": _rel(source),
        "source_bytes": source.stat().st_size,
        "source_rows_scanned": scanned,
        "boundary_symbol": BOUNDARY,
        "symbol_accounting": "1 boundary symbol + len(text.encode('utf-8')) per jsonl row",
        "skip_symbols": skip_symbols,
        "first_emitted_row": first_row,
        "last_emitted_row": last_row,
        "emitted_rows": emitted,
        "dropped_rows_in_window": dropped,
        "emitted_symbols": emitted_symbols,
        "emitted_bytes": emitted_bytes,
        "max_symbols": max_symbols,
        "replay_symbols_from_seen_region": 0,
        #: 操作检验：这条流的对话密度到底是多少（对照臂应显著低于实验臂）。
        "dialogue_qualified_rows": dialogue_rows,
        "dialogue_density_of_emitted_rows": (
            round(dialogue_rows / emitted, 6) if emitted else None
        ),
        "arm_overlap_check": overlap,
        "dialogue_rule": {
            "min_distinct_speakers": MIN_SPEAKERS,
            "speaker_pattern": SPEAKER_PATTERN,
            "excluded_speakers": sorted(META_SPEAKERS),
        },
        "output": _rel(output),
        "output_sha256": digest.hexdigest(),
    }


def lineage_ticks(beta: Path, prior: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """读两环信封，返回 (起点, 上一环) 的 tick 与指纹名。"""

    import torch

    out = []
    for path in (beta, prior):
        envelope = torch.load(path, map_location="cpu", weights_only=False)
        metadata = envelope.get("metadata", {}) or {}
        records = json.loads(metadata.get("corpus_fingerprint") or "[]")
        out.append(
            {
                "checkpoint": _rel(path),
                "tick": int(metadata.get("tick", -1)),
                "corpus_names": [str(item.get("name")) for item in records],
            }
        )
    return out[0], out[1]


def write_manifest(manifest: Path, meta: dict[str, Any]) -> None:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="构建 P3b 某一臂的训练语料（未见过源窗口）")
    parser.add_argument("--rule", choices=RULES, required=True)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--other-arm", type=Path, default=None, help="配对语料，用于共享前缀核对")
    parser.add_argument("--skip-symbols", type=int, default=None, help="缺省则按检查点血缘推导")
    parser.add_argument("--max-symbols", type=int, default=DEFAULT_MAX_SYMBOLS)
    parser.add_argument("--beta", type=Path, default=PROJECT_ROOT / "checkpoints" / "seed_beta.pt")
    parser.add_argument(
        "--prior", type=Path, default=PROJECT_ROOT / "checkpoints" / "resumed_seed_corpus.pt"
    )
    args = parser.parse_args(argv)

    output = args.output or PROJECT_ROOT / "data" / f"p3b_{args.rule}_fresh.jsonl"
    manifest = args.manifest or MANIFEST_DIR / f"p3b_{args.rule}_fresh_manifest.json"

    derivation: dict[str, Any]
    if args.skip_symbols is not None:
        skip = args.skip_symbols
        derivation = {"skip_symbols_given_directly": skip}
    else:
        beta_meta, prior_meta = lineage_ticks(args.beta, args.prior)
        skip = seen_prefix_symbols(
            resume_tick=beta_meta["tick"],
            prior_tick=prior_meta["tick"],
            resume_corpus_name=beta_meta["corpus_names"][0] if beta_meta["corpus_names"] else "",
            prior_corpus_name=prior_meta["corpus_names"][0] if prior_meta["corpus_names"] else "",
            corpus_name=args.source.name,
        )
        derivation = {"beta": beta_meta, "prior": prior_meta, "method": "tick difference"}

    meta = build(
        args.source,
        output,
        rule=args.rule,
        skip_symbols=skip,
        max_symbols=args.max_symbols,
        other_arm=args.other_arm,
    )
    meta["skip_derivation"] = derivation
    write_manifest(manifest, meta)
    print(re.sub(r"[^\x00-\x7f]", "?", json.dumps(meta, indent=2)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
