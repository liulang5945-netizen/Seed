from __future__ import annotations

import json
from pathlib import Path

from scripts.training.eval_taiji_perception_a1 import (
    MANIFEST_FORMAT,
    build_manifest,
    corpus_from_manifest,
)


def test_a1_manifest_uses_hash_buckets_and_explicit_controls(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    with corpus_path.open("w", encoding="utf-8") as handle:
        for index in range(500):
            handle.write(json.dumps({"text": f"样本 {index} 具有不同的组合结构。"}) + "\n")

    first = build_manifest([corpus_path], train_count=8, holdout_count=8, max_symbols=64)
    second = build_manifest([corpus_path], train_count=8, holdout_count=8, max_symbols=64)

    assert first == second
    assert first["format"] == MANIFEST_FORMAT
    assert first["selection"]["method"] == "sha256-text-bucket"
    assert first["selection"]["scanned_rows"] > 0
    assert len(first["train"]) == 8
    assert len(first["unseen_composition"]) == 4
    assert len(first["boundary_perturbed"]) == 4
    assert len(first["random_chunk"]) == 4
    assert all(256 in sequence for sequence in first["boundary_perturbed"])
    assert all(256 in sequence for sequence in first["random_chunk"])

    restored = corpus_from_manifest(first)
    assert len(restored.train) == 8
    assert len(restored.unseen_composition) == 4
