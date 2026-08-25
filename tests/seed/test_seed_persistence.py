"""M0 persistence hardening: atomic save + envelope metadata contracts."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from seed import Seed, SeedConfig  # noqa: E402
from seed.persistence import (  # noqa: E402
    atomic_save,
    attach_metadata,
    corpus_fingerprint,
)


def _tiny_seed() -> Seed:
    return Seed(SeedConfig(), episode_id="persistence-test")


def test_atomic_save_roundtrip(tmp_path: Path) -> None:
    model = _tiny_seed()
    model.learn_bytes("问：你好。\n答：你好。".encode())
    target = tmp_path / "ckpt.pt"
    atomic_save(model.checkpoint(), target)

    restored = Seed.from_checkpoint(torch.load(target, weights_only=False))
    assert restored.tick == model.tick
    probe = "你好".encode()
    assert restored.score_bytes(probe) == model.score_bytes(probe)


def test_atomic_save_leaves_no_tmp_residue(tmp_path: Path) -> None:
    target = tmp_path / "ckpt.pt"
    atomic_save(_tiny_seed().checkpoint(), target)
    leftovers = [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []


def test_atomic_save_failure_keeps_previous_version(tmp_path: Path) -> None:
    target = tmp_path / "ckpt.pt"
    first = _tiny_seed().checkpoint()
    atomic_save(first, target)
    before = target.read_bytes()

    class _Unpicklable:
        def __reduce__(self):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        atomic_save({"broken": _Unpicklable()}, target)

    assert target.read_bytes() == before
    leftovers = [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []


def test_metadata_attached_and_load_ignores_it(tmp_path: Path) -> None:
    model = _tiny_seed()
    model.learn_bytes("你好".encode())
    envelope = attach_metadata(
        model.checkpoint(),
        tick=model.tick,
        corpus_fingerprint="test-fingerprint",
        extra={"trainer": "unit-test"},
    )
    metadata = envelope["metadata"]
    assert metadata["tick"] == model.tick
    assert metadata["corpus_fingerprint"] == "test-fingerprint"
    assert metadata["trainer"] == "unit-test"
    assert "saved_at_utc" in metadata
    assert "profile" in metadata

    target = tmp_path / "meta.pt"
    atomic_save(envelope, target)
    restored = Seed.from_checkpoint(torch.load(target, weights_only=False))
    assert restored.tick == model.tick


def test_legacy_envelope_without_metadata_still_loads(tmp_path: Path) -> None:
    # Old checkpoints predate M0 and carry no metadata key; restore must not
    # require it.
    model = _tiny_seed()
    target = tmp_path / "legacy.pt"
    torch.save(model.checkpoint(), target)
    restored = Seed.from_checkpoint(torch.load(target, weights_only=False))
    assert restored.tick == model.tick


def test_corpus_fingerprint_detects_size_drift(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"text": "abc"}\n', encoding="utf-8")
    first = corpus_fingerprint([corpus])
    corpus.write_text('{"text": "abcdef"}\n', encoding="utf-8")
    second = corpus_fingerprint([corpus])
    assert first != second
    missing = corpus_fingerprint([tmp_path / "absent.jsonl"])
    assert '"bytes":-1' in missing
