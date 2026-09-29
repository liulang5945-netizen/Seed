"""构建随包表面门槛工件（A30 owner 裁定 §7-1 的门槛支撑件，2026-09-29 批 +5.5 MB）。

从 ``diag_taiji_r2_surface_decode.build_ngram_model``（200k 行，判据标定所用的同一模型）
构建 uni/bi 计数，压成 lzma+pickle 工件供产品门槛在无 1.4 GB 语料的环境里逐位复算
``mean_nll``。自检：工件载回后，对语料抽样行与直建模型的 ``well_formed`` 判定必须逐条相同，
否则整件作废（宁可不发工件，也不发一个判据变味的工件）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "training"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

DEFAULT_OUT = PROJECT_ROOT / "checkpoints" / "seed_surface_ngram.lzma"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--limit", type=int, default=200000, help="语料行数上限；阈值 -7.8 按 200k 行标定"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--out-report", default=None)
    args = parser.parse_args()

    from diag_taiji_r2_surface_decode import build_ngram_model, well_formed

    from seed import surface_gate

    out = Path(args.out)
    if out.exists() and not args.overwrite:
        raise SystemExit(
            f"refusing to overwrite existing artifact {out}; pass --overwrite explicitly"
        )

    started = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    uni, bi, vocab, total = build_ngram_model(limit=args.limit)
    build_seconds = round(time.perf_counter() - t0, 1)

    blob = surface_gate.build_artifact_blob(uni, bi, vocab, total)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(blob)

    # 自检：工件载回 ⇒ 与直建模型在语料抽样上的 well_formed 判定逐条相同。
    loaded = surface_gate.load_surface_ngram(out)
    samples: list[str] = []
    with (PROJECT_ROOT / "data" / "simple_zh" / "simple_zh_texts.jsonl").open(
        "r", encoding="utf-8", errors="replace"
    ) as handle:
        for index, raw in enumerate(handle):
            if index >= args.limit:
                break
            try:
                text = "".join((json.loads(raw).get("text") or "").split())
            except ValueError:
                continue
            if len(text) >= 12:
                samples.append(text[:32])
    rng = random.Random(7)
    sample = rng.sample(samples, 40)
    verdicts_direct = [well_formed(t, (uni, bi, vocab, total)) for t in sample]
    verdicts_loaded = [well_formed(t, loaded) for t in sample]
    drift_guard = verdicts_direct == verdicts_loaded

    report = {
        "format": "taiji-a30-surface-ngram-artifact-v1",
        "started_utc": started,
        "corpus_limit_lines": args.limit,
        "uni_entries": len(uni),
        "bi_entries": len(bi),
        "vocab": vocab,
        "total_counted": total,
        "build_seconds": build_seconds,
        "artifact": out.relative_to(PROJECT_ROOT).as_posix(),
        "artifact_bytes": out.stat().st_size,
        "artifact_sha256": _sha256(out),
        "self_check": {
            "sample_size": len(sample),
            "verdicts_match_direct_model": drift_guard,
        },
        "guard": {"artifact_self_check_passed": drift_guard},
    }
    report_path = (
        Path(args.out_report)
        if args.out_report
        else PROJECT_ROOT / "reports" / "taiji_a30_surface_ngram_artifact_20260929.json"
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if drift_guard else 1


if __name__ == "__main__":
    raise SystemExit(main())
