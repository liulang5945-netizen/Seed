"""M5.S4: convert UltraData Code_Agent + Math into a heterogeneous holdout
corpus in the simple_zh dialogue contract.

S3's domain-level holdout (Chinese-general) turned out to be only a mild
topical variant of the training domains, so generalizing to it was trivial
and the causal probes lost discrimination.  This script builds a genuinely
heterogeneous holdout: code-agent task summaries (Code_Agent) and math word
problems (RL-2609 Math), which sit far from Chinese QA in embedding space.
Output is deterministic and manifest-anchored, like M4.R12.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Iterator
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FORMAT = "taiji-m5-s4-hetero-holdout-manifest-v1"
DEFAULT_QUOTAS = {"Math": 2000, "Code_Agent": 2000}
DEFAULT_MAX_RECORD_BYTES = 16 * 1024
MIN_USER_CHARS = 8
MIN_ASSISTANT_CHARS = 8


def _iter_math(domain_dir: Path, quota: int, max_bytes: int) -> Iterator[tuple[str, str]]:
    selected = 0
    seen: set[str] = set()
    for source_file in sorted(domain_dir.glob("*.jsonl")):
        if selected >= quota:
            break
        with source_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                if selected >= quota:
                    break
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                query = str(record.get("query", "")).strip()
                answer = str(record.get("ground_truth", "")).strip()
                if len(query) < MIN_USER_CHARS or len(answer) < 1:
                    continue
                text = f"问：{query}\n答：{answer}"
                if len(text.encode("utf-8")) > max_bytes:
                    continue
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)
                selected += 1
                yield "Math", text


def _iter_code(domain_dir: Path, quota: int, max_bytes: int) -> Iterator[tuple[str, str]]:
    selected = 0
    seen: set[str] = set()
    for source_file in sorted(domain_dir.glob("*.jsonl")):
        if selected >= quota:
            break
        with source_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                if selected >= quota:
                    break
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                messages = record.get("messages")
                if not isinstance(messages, list):
                    continue
                user = next(
                    (
                        m["content"].strip()
                        for m in messages
                        if m.get("role") == "user" and isinstance(m.get("content"), str)
                    ),
                    "",
                )
                assistants = [
                    m["content"].strip()
                    for m in messages
                    if m.get("role") == "assistant"
                    and isinstance(m.get("content"), str)
                    and m["content"].strip()
                ]
                if len(user) < MIN_USER_CHARS or not assistants:
                    continue
                text = f"问：{user}\n答：{assistants[-1]}"
                if len(text.encode("utf-8")) > max_bytes:
                    continue
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)
                selected += 1
                yield "Code_Agent", text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--math-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "ultradata" / "RL-2609" / "data" / "Math",
    )
    parser.add_argument(
        "--code-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "ultradata" / "SFT-Agent-2609" / "data" / "Code_Agent",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "ultradata"
        / "derived"
        / "ultradata_hetero_holdout_simplezh.jsonl",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_m5_s4_hetero_holdout_manifest_20260909.json",
    )
    parser.add_argument("--max-record-bytes", type=int, default=DEFAULT_MAX_RECORD_BYTES)
    args = parser.parse_args()

    started = time.perf_counter()
    math_items = list(_iter_math(args.math_dir, DEFAULT_QUOTAS["Math"], args.max_record_bytes))
    code_items = list(
        _iter_code(args.code_dir, DEFAULT_QUOTAS["Code_Agent"], args.max_record_bytes)
    )

    # Deterministic interleave so the two heterogeneous domains do not cluster.
    rows: list[tuple[str, str]] = []
    mi = ci = 0
    while mi < len(math_items) or ci < len(code_items):
        if mi < len(math_items):
            rows.append(math_items[mi])
            mi += 1
        if ci < len(code_items):
            rows.append(code_items[ci])
            ci += 1

    lines = [
        json.dumps({"text": text, "domain": domain}, ensure_ascii=False) for domain, text in rows
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for line in lines:
            handle.write(line + "\n")

    output_digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    manifest = {
        "format": FORMAT,
        "version": 1,
        "generated_at_epoch": int(time.time()),
        "quotas": DEFAULT_QUOTAS,
        "records_per_domain": {
            "Math": len(math_items),
            "Code_Agent": len(code_items),
        },
        "total_records": len(rows),
        "max_record_bytes": args.max_record_bytes,
        "conversion_style": "问：{query/request}\\n答：{answer/final-assistant}",
        "output": str(args.output.as_posix()),
        "output_sha256": output_digest,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "records_per_domain": manifest["records_per_domain"],
                "total_records": manifest["total_records"],
                "output": manifest["output"],
                "output_sha256": output_digest,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
