"""Build a leakage-aware A1 manifest and evaluate Taiji perception.

The manifest is intentionally separate from model checkpoints.  It records
which raw sequences entered each evaluation role, while ``taiji.evaluation``
owns the metric and gate contract.

Example::

    python scripts/training/eval_taiji_perception_a1.py \
        --corpus data/simple_zh/dialogue_extended_clean.jsonl \
        --manifest-out reports/taiji_a1_manifest.json \
        --report-out reports/taiji_a1_perception.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from taiji import PerceptionCorpus, PerceptionEvaluator, TaijiConfig  # noqa: E402

MANIFEST_FORMAT = "taiji-a1-manifest-v1"
BOUNDARY_SYMBOL = 256


def _text_from_record(record: Any) -> str | None:
    if isinstance(record, str):
        return record
    if not isinstance(record, dict):
        return None
    for key in ("text", "content", "prompt", "input"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _bucket(text: str, *, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}\0{text}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % 10_000


def _bounded_symbols(text: str, *, max_symbols: int) -> tuple[int, ...]:
    symbols = tuple(text.encode()[: int(max_symbols)])
    if len(symbols) < 4:
        raise ValueError("A1 sequence must contain at least four UTF-8 bytes")
    return symbols


def _compose(left: tuple[int, ...], right: tuple[int, ...], *, max_symbols: int) -> tuple[int, ...]:
    left_cut = max(1, len(left) // 2)
    right_cut = max(1, len(right) // 2)
    composed = (*left[:left_cut], *right[-right_cut:])
    return composed[: int(max_symbols)]


def _insert_marker(sequence: tuple[int, ...], *, seed: int, max_symbols: int) -> tuple[int, ...]:
    position = 1 + int(seed) % max(1, len(sequence) - 1)
    perturbed = (*sequence[:position], BOUNDARY_SYMBOL, *sequence[position:])
    return perturbed[: int(max_symbols)]


def _random_chunk_control(
    sequence: tuple[int, ...], *, seed: int, max_symbols: int
) -> tuple[int, ...]:
    generator = random.Random(int(seed))
    candidate_positions = list(range(1, len(sequence)))
    count = min(3, max(1, len(candidate_positions) // 4))
    positions = sorted(generator.sample(candidate_positions, count))
    output: list[int] = []
    previous = 0
    for position in positions:
        output.extend(sequence[previous:position])
        output.append(BOUNDARY_SYMBOL)
        previous = position
    output.extend(sequence[previous:])
    return tuple(output[: int(max_symbols)])


def build_manifest(
    corpus_paths: list[Path | str],
    *,
    train_count: int = 32,
    holdout_count: int = 16,
    max_symbols: int = 256,
    seed: int = 20260825,
) -> dict[str, Any]:
    """Stream JSONL corpora into deterministic, disjoint A1 roles."""

    if train_count <= 0 or holdout_count <= 0:
        raise ValueError("A1 train_count and holdout_count must be positive")
    if max_symbols < 4:
        raise ValueError("A1 max_symbols must be at least four")
    train: list[tuple[int, ...]] = []
    holdout: list[tuple[int, ...]] = []
    scanned = 0
    for raw_path in corpus_paths:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                scanned += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = _text_from_record(record)
                if text is None:
                    continue
                bucket = _bucket(text, seed=seed)
                if bucket < 8_000 and len(train) < train_count:
                    train.append(_bounded_symbols(text, max_symbols=max_symbols))
                elif 8_000 <= bucket < 9_000 and len(holdout) < holdout_count:
                    holdout.append(_bounded_symbols(text, max_symbols=max_symbols))
                if len(train) >= train_count and len(holdout) >= holdout_count:
                    break
        if len(train) >= train_count and len(holdout) >= holdout_count:
            break
    if len(train) < train_count or len(holdout) < holdout_count:
        raise ValueError(
            "corpus did not provide enough hash-bucketed A1 rows: "
            f"train={len(train)}/{train_count}, holdout={len(holdout)}/{holdout_count}"
        )

    unseen = tuple(
        _compose(left, right, max_symbols=max_symbols)
        for left, right in zip(holdout[::2], holdout[1::2], strict=False)
    )
    if not unseen:
        unseen = tuple(holdout)
    boundary = tuple(
        _insert_marker(sequence, seed=seed + index, max_symbols=max_symbols)
        for index, sequence in enumerate(unseen)
    )
    random_chunk = tuple(
        _random_chunk_control(sequence, seed=seed + 97 * index, max_symbols=max_symbols)
        for index, sequence in enumerate(unseen)
    )
    return {
        "format": MANIFEST_FORMAT,
        "source": [str(Path(path)) for path in corpus_paths],
        "selection": {
            "method": "sha256-text-bucket",
            "seed": int(seed),
            "train_bucket": "[0, 8000)",
            "holdout_bucket": "[8000, 9000)",
            "scanned_rows": scanned,
            "unseen_composition": "pairwise half-composition of hash holdout rows",
            "boundary_symbol": BOUNDARY_SYMBOL,
        },
        "train": [list(sequence) for sequence in train],
        "unseen_composition": [list(sequence) for sequence in unseen],
        "boundary_perturbed": [list(sequence) for sequence in boundary],
        "random_chunk": [list(sequence) for sequence in random_chunk],
    }


def corpus_from_manifest(payload: dict[str, Any]) -> PerceptionCorpus:
    if payload.get("format") != MANIFEST_FORMAT:
        raise ValueError("unsupported Taiji A1 manifest format")

    def sequences(name: str) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(int(symbol) for symbol in sequence) for sequence in payload[name])

    return PerceptionCorpus(
        train=sequences("train"),
        unseen_composition=sequences("unseen_composition"),
        boundary_perturbed=sequences("boundary_perturbed"),
        random_chunk=sequences("random_chunk"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", nargs="+", required=True, type=Path)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=32)
    parser.add_argument("--holdout-count", type=int, default=16)
    parser.add_argument("--max-symbols", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()

    manifest = build_manifest(
        args.corpus,
        train_count=args.train_count,
        holdout_count=args.holdout_count,
        max_symbols=args.max_symbols,
        seed=args.seed,
    )
    corpus = corpus_from_manifest(manifest)
    report = PerceptionEvaluator(TaijiConfig()).evaluate(corpus)
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
