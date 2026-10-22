"""Synthetic-only recovery equivalence and transaction contracts."""

from pathlib import Path

import pytest
import torch

from scripts.training.taiji_r2_dialogue_recovery import Cursor, RecoveryStore, train_next
from taiji import LanguageAlignmentTrainer, LanguageEpisode, LanguageEpisodeCorpus, Taiji


def trainer():
    torch.set_num_threads(1)
    corpus = LanguageEpisodeCorpus(
        tuple(
            LanguageEpisode(
                episode_id=f"e{i}",
                family_id=f"f{i}",
                task_family="fact",
                split="train",
                context="",
                user_input="a",
                response="b",
            )
            for i in range(2)
        ),
        require_eval_splits=False,
    )
    return LanguageAlignmentTrainer(
        Taiji(episode_id="recovery-test"), corpus, code_revision="synthetic"
    )


@pytest.mark.parametrize("stop", [0, 1, 2, 3, 4])
def test_exact_equivalence_across_boundaries(tmp_path: Path, stop):
    original = trainer()
    initial = original.checkpoint()
    original.train(epochs=2)
    candidate = LanguageAlignmentTrainer.from_checkpoint(initial, original.corpus)
    cursor = Cursor()
    for _ in range(stop):
        _, cursor = train_next(candidate, cursor, 2)
    with RecoveryStore(tmp_path, {"arm": "synthetic"}, original.corpus) as store:
        receipt = store.commit(candidate, cursor, 2, {"cpu_upper": 0})
        assert receipt["verification"]["cpu_seconds"] > 0
        candidate, cursor, envelope = store.load_latest()
        assert envelope["ledger"] == {"cpu_upper": 0}
        while cursor.epoch < 2:
            _, cursor = train_next(candidate, cursor, 2)
    assert candidate.checkpoint()["checkpoint_digest"] == original.checkpoint()["checkpoint_digest"]


def test_orphan_ignored_and_corruption_refused(tmp_path):
    candidate = trainer()
    bindings = {"arm": "synthetic"}
    with RecoveryStore(tmp_path, bindings, candidate.corpus) as store:
        receipt = store.commit(candidate, Cursor(), 2, {})
        (tmp_path / "orphan.pt").write_bytes(b"incomplete")
        _, cursor, _ = store.load_latest()
        assert cursor == Cursor()
        (tmp_path / receipt["bundle"]).write_bytes(b"corrupt")
        with pytest.raises(ValueError, match="bundle hash"):
            store.load_latest()


def test_lock_and_binding_refusal(tmp_path):
    candidate = trainer()
    with RecoveryStore(tmp_path, {"arm": "a"}, candidate.corpus) as store:
        with (
            pytest.raises(RuntimeError, match="live_identity=True"),
            RecoveryStore(tmp_path, {"arm": "a"}, candidate.corpus),
        ):
            pass
        store.commit(candidate, Cursor(), 2, {})
        with pytest.raises(ValueError, match="bindings"):
            RecoveryStore(tmp_path, {"arm": "b"}, candidate.corpus).load_latest()
    with pytest.raises(ValueError, match="cursor/history"):
        train_next(candidate, Cursor(0, 1), 2)


def test_multiple_commits_keep_rng_and_chain(tmp_path):
    import random

    candidate = trainer()
    with RecoveryStore(tmp_path, {"arm": "synthetic"}, candidate.corpus) as store:
        first = store.commit(candidate, Cursor(), 2, {})
        _, cursor = train_next(candidate, Cursor(), 2)
        random.seed(521)
        before_python = random.getstate()
        before_torch = torch.get_rng_state().clone()
        second = store.commit(candidate, cursor, 2, {"sessions": [1, 2]})
        assert random.getstate() == before_python
        assert torch.equal(torch.get_rng_state(), before_torch)
        assert second["previous_receipt_sha256"] is not None
        assert first["path"] != second["path"]
        _, restored, payload = store.load_latest()
        assert restored == cursor
        assert payload["ledger"]["sessions"] == [1, 2]
        (tmp_path / "latest.json").write_text("{}", encoding="utf-8")
        with pytest.raises(KeyError):
            store.load_latest()


def test_reclaim_dead_owner_and_guard(tmp_path, monkeypatch):
    import json

    import psutil

    from scripts.training.taiji_r2_dialogue_recovery import reclaim_writer_lock

    candidate = trainer()
    raw = json.dumps({"pid": 987654, "create_time": 1.0, "token": "dead"}).encode()
    (tmp_path / "writer.lock").write_bytes(raw)
    real_process = psutil.Process

    def process(pid=None):
        if pid == 987654:
            raise psutil.NoSuchProcess(pid)
        return real_process(pid)

    monkeypatch.setattr(psutil, "Process", process)
    with reclaim_writer_lock(tmp_path) as token:
        assert list((tmp_path / "stale-locks").iterdir())[0].read_bytes() == raw
        with (
            pytest.raises(RuntimeError, match="guard blocks"),
            RecoveryStore(tmp_path, {"arm": "a"}, candidate.corpus),
        ):
            pass
        with (
            pytest.raises(RuntimeError, match="already in progress"),
            reclaim_writer_lock(tmp_path),
        ):
            pass
        with RecoveryStore(tmp_path, {"arm": "a"}, candidate.corpus, reclaim_token=token):
            assert (tmp_path / "writer.lock").is_file()
    assert not (tmp_path / "reclaim.guard").exists()


def test_reclaim_alive_reused_invalid(tmp_path):
    import json
    import os

    import psutil

    from scripts.training.taiji_r2_dialogue_recovery import reclaim_writer_lock

    lock = tmp_path / "writer.lock"
    for identity, message in [
        (
            {"pid": os.getpid(), "create_time": psutil.Process().create_time(), "token": "x"},
            "still alive",
        ),
        ({"pid": os.getpid(), "create_time": 1.0, "token": "x"}, "pid reused"),
        ({"pid": "bad", "create_time": 1.0, "token": "x"}, "invalid writer"),
    ]:
        raw = json.dumps(identity).encode()
        lock.write_bytes(raw)
        with (
            pytest.raises((RuntimeError, ValueError), match=message),
            reclaim_writer_lock(tmp_path),
        ):
            pass
        assert lock.read_bytes() == raw
        assert not (tmp_path / "reclaim.guard").exists()
