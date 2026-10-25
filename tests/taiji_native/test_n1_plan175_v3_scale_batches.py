"""The frozen length buckets keep each training row exactly once per epoch."""

from dataclasses import dataclass

import pytest

from scripts.training.run_taiji_n1_plan175_v3_scale import bucket, epoch_batches


@dataclass
class Episode:
    target_bytes: bytes


def test_bucket_edges_cover_the_complete_v3_target_window():
    assert [bucket(value) for value in (1, 128, 129, 512, 513, 1024, 1536, 2048)] == [
        128,
        128,
        512,
        512,
        1024,
        1024,
        1536,
        2048,
    ]
    with pytest.raises(ValueError, match="bound"):
        bucket(2049)


def test_epoch_batching_is_deterministic_complete_and_keeps_remainders():
    rows = [
        ({"target_id": f"id-{index}"}, Episode(b"x" * length))
        for index, length in enumerate((12, 100, 129, 500, 700, 900, 1100, 1400, 1800))
    ]
    first = epoch_batches(rows, seed=169, epoch=3)
    assert first == epoch_batches(rows, seed=169, epoch=3)
    flat = [candidate["target_id"] for batch in first for candidate, _ in batch]
    assert len(flat) == len(rows) and set(flat) == {row[0]["target_id"] for row in rows}
    assert all(1 <= len(batch) <= 4 for batch in first)
    assert any(len(batch) < 4 for batch in first)
