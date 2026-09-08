"""M4.R12 preflight: convert UltraData SFT-2605 no_think records to the
simple_zh dialogue contract.

The foundation dataset loader (``FoundationTrainingDataset.from_jsonl``)
only reads a single ``text``/``content``/``input`` string field per JSONL
line.  UltraData-SFT-2605 no_think records use a ``messages`` array, so this
script converts them into ``{"text": "问：...\\n答：..."}`` records with the
same prefix style as ``data/simple_zh/dialogue_extended_clean.jsonl``.

The conversion is deterministic (fixed per-domain quotas, file order, and
round-robin interleaving).  It never trains; it only produces a fixed data
file and a conversion manifest for the later pre-registered data-source
comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FORMAT = "taiji-m4r12-ultradata-conversion-manifest-v1"
DEFAULT_QUOTAS = {"Knowledge": 8000, "IF": 6000, "Chinese-general": 6000}
DEFAULT_MAX_RECORD_BYTES = 16 * 1024
MIN_USER_CHARS = 8
MIN_ASSISTANT_CHARS = 8


def _dialogue_text(messages: Sequence[dict[str, Any]]) -> str | None:
    """Map a no_think messages array onto the simple_zh 问/答 prefix style."""
    user = next(
        (
            m["content"].strip()
            for m in messages
            if m.get("role") == "user" and isinstance(m.get("content"), str)
        ),
        None,
    )
    assistant = next(
        (
            m["content"].strip()
            for m in messages
            if m.get("role") == "assistant" and isinstance(m.get("content"), str)
        ),
        None,
    )
    if not user or not assistant:
        return None
    if len(user) < MIN_USER_CHARS or len(assistant) < MIN_ASSISTANT_CHARS:
        return None
    return f"问：{user}\n答：{assistant}"


def _iter_domain_records(
    domain_dir: Path,
    quota: int,
    max_record_bytes: int,
) -> Iterator[str]:
    """Stream one domain in deterministic file order up to the quota."""
    selected = 0
    seen_digests: set[str] = set()
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
                messages = record.get("messages") if isinstance(record, dict) else None
                if not isinstance(messages, list) or len(messages) < 2:
                    continue
                text = _dialogue_text(messages)
                if text is None:
                    continue
                encoded = text.encode("utf-8")
                if len(encoded) > max_record_bytes:
                    continue
                digest = hashlib.sha256(encoded).hexdigest()
                if digest in seen_digests:
                    continue
                seen_digests.add(digest)
                selected += 1
                yield json.dumps({"text": text}, ensure_ascii=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "ultradata" / "SFT-2605" / "data" / "no_think",
    )
    parser.add_argument("--domains", nargs="+", default=sorted(DEFAULT_QUOTAS))
    parser.add_argument(
        "--quota",
        type=int,
        nargs="+",
        default=None,
        help="Per-domain record quotas, aligned with --domains order.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "ultradata"
        / "derived"
        / "ultradata_sft_nothink_simplezh.jsonl",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "taiji_m4r12_ultradata_conversion_manifest_20260909.json",
    )
    parser.add_argument("--max-record-bytes", type=int, default=DEFAULT_MAX_RECORD_BYTES)
    args = parser.parse_args()

    quotas = (
        dict(zip(args.domains, args.quota, strict=True))
        if args.quota
        else {d: DEFAULT_QUOTAS[d] for d in args.domains}
    )
    unknown = set(args.domains) - set(DEFAULT_QUOTAS)
    if unknown:
        raise ValueError(f"unknown domains: {sorted(unknown)}")

    started = time.perf_counter()
    per_domain: dict[str, int] = {}
    domain_lines: dict[str, list[str]] = {d: [] for d in args.domains}
    for domain in args.domains:
        domain_dir = args.source_root / domain
        if not domain_dir.is_dir():
            raise FileNotFoundError(domain_dir)
        for line in _iter_domain_records(domain_dir, quotas[domain], args.max_record_bytes):
            domain_lines[domain].append(line)
        per_domain[domain] = len(domain_lines[domain])

    # Deterministic round-robin interleave so domains do not cluster.
    output_lines: list[str] = []
    cursors = {d: 0 for d in args.domains}
    while any(cursors[d] < per_domain[d] for d in args.domains):
        for domain in args.domains:
            if cursors[domain] < per_domain[domain]:
                output_lines.append(domain_lines[domain][cursors[domain]])
                cursors[domain] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for line in output_lines:
            handle.write(line + "\n")

    output_digest = hashlib.sha256("\n".join(output_lines).encode("utf-8")).hexdigest()
    manifest = {
        "format": FORMAT,
        "version": 1,
        "generated_at_epoch": int(time.time()),
        "source_root": str(args.source_root.as_posix()),
        "domains": args.domains,
        "quotas": quotas,
        "records_per_domain": per_domain,
        "total_records": len(output_lines),
        "max_record_bytes": args.max_record_bytes,
        "conversion_style": "问：{user}\\n答：{assistant} (simple_zh prefix style)",
        "output": str(args.output.as_posix()),
        "output_sha256": output_digest,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in ("records_per_domain", "total_records", "output", "output_sha256")
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
