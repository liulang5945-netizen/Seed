"""Train-only, transactional R2 recovery primitives; no formal-run entrypoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil
import torch

from taiji import LanguageAlignmentTrainer, LanguageEpisode, LanguageEpisodeCorpus
from taiji.internalization import content_digest


@dataclass(frozen=True)
class Cursor:
    epoch: int = 0
    index: int = 0


def _validate(trainer, cursor, epochs):
    if trainer.config.developmental_mode != "static":
        raise ValueError("recovery supports static mode only")
    episodes = trainer.corpus.for_split("train")
    if any(e.split != "train" for e in trainer.corpus.episodes):
        raise ValueError("recovery accepts train-only corpus")
    if epochs <= 0 or not 0 <= cursor.epoch <= epochs:
        raise ValueError("invalid epoch cursor")
    if not 0 <= cursor.index < len(episodes) or (cursor.epoch == epochs and cursor.index):
        raise ValueError("invalid episode cursor")
    count = cursor.epoch * len(episodes) + cursor.index
    if trainer.episode_count != count or len(trainer.history) != count:
        raise ValueError("cursor/history/count mismatch")
    if len(trainer.developmental_history) != cursor.epoch:
        raise ValueError("epoch finish history mismatch")
    if trainer.global_step != sum(int(r["observations"]) for r in trainer.history):
        raise ValueError("history/update mismatch")
    for i, record in enumerate(trainer.history):
        if (
            record["epoch"] != i // len(episodes)
            or record["episode_id"] != episodes[i % len(episodes)].episode_id
        ):
            raise ValueError("history order mismatch")
    return episodes


def train_next(trainer, cursor: Cursor, epochs: int):
    """Complete one episode and its epoch finish before exposing a save point."""
    episodes = _validate(trainer, cursor, epochs)
    if cursor.epoch == epochs:
        raise StopIteration
    episode = episodes[cursor.index]
    metrics = {"observations": 0.0, "accuracy": 0.0, "mean_surprise": 0.0}
    for _ in range(trainer.config.response_repeats):
        current = trainer._target_pass(episode, learn=True)
        for key in metrics:
            metrics[key] += current[key]
    for key in metrics:
        metrics[key] /= trainer.config.response_repeats
    trainer.global_step += int(metrics["observations"])
    trainer.episode_count += 1
    record = {
        "epoch": cursor.epoch,
        "episode_id": episode.episode_id,
        "family_id": episode.family_id,
        "target_bytes": len(episode.target_bytes),
        "prefix_bytes": len(episode.prompt_bytes),
        "observations": int(metrics["observations"]),
        "response_accuracy": metrics["accuracy"],
        "response_mean_surprise": metrics["mean_surprise"],
    }
    trainer.history.append(record)
    next_cursor = Cursor(cursor.epoch, cursor.index + 1)
    if next_cursor.index == len(episodes):
        trainer._finish_developmental_epoch(cursor.epoch)
        next_cursor = Cursor(cursor.epoch + 1, 0)
    return record, next_cursor


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _publish(path, writer, *, replace=False):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            if path.exists():
                raise FileExistsError(path)
            os.rename(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_json(path, payload, *, replace=False):
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    _publish(path, lambda stream: stream.write(raw), replace=replace)


def _restore_bundle(path, bindings):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "taiji-r2-recovery-v1" or payload["bindings"] != bindings:
        raise ValueError("recovery source bindings mismatch")
    expected = content_digest({k: v for k, v in payload.items() if k != "digest"})
    if payload.get("digest") != expected:
        raise ValueError("recovery envelope digest mismatch")
    corpus = LanguageEpisodeCorpus(
        tuple(LanguageEpisode.from_payload(p) for p in payload["episodes"]),
        tuple(tuple(p) for p in payload["source_files"]),
        require_eval_splits=False,
    )
    trainer = LanguageAlignmentTrainer.from_checkpoint(payload["trainer"], corpus)
    if trainer.checkpoint()["checkpoint_digest"] != payload["trainer"]["checkpoint_digest"]:
        raise ValueError("restored trainer differs")
    cursor = Cursor(**payload["cursor"])
    _validate(trainer, cursor, payload["epochs"])
    if payload["runtime"] != _runtime():
        raise ValueError("recovery runtime mismatch")
    return trainer, cursor, payload


def _runtime():
    return {
        "python": sys.version,
        "torch": str(torch.__version__),
        "threads": torch.get_num_threads(),
    }


@contextmanager
def reclaim_writer_lock(directory: Path):
    """Archive a proven-dead owner under an exclusive guard; never steal by age."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    guard = directory / "reclaim.guard"
    token = uuid.uuid4().hex
    try:
        with guard.open("x", encoding="utf-8") as stream:
            json.dump(
                {"token": token, "pid": os.getpid(), "create_time": psutil.Process().create_time()},
                stream,
            )
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        raise RuntimeError("reclaim already in progress") from None
    try:
        lock = directory / "writer.lock"
        raw = lock.read_bytes()
        identity = json.loads(raw)
        if (
            type(identity.get("pid")) is not int
            or identity["pid"] <= 0
            or type(identity.get("create_time")) not in (int, float)
            or identity["create_time"] <= 0
            or not identity.get("token")
        ):
            raise ValueError("invalid writer identity")
        try:
            owner = psutil.Process(identity["pid"])
            if owner.create_time() != identity["create_time"]:
                raise RuntimeError("writer pid reused; identity mismatch")
            if owner.is_running():
                raise RuntimeError("writer owner still alive")
        except psutil.NoSuchProcess:
            pass
        archive = directory / "stale-locks"
        archive.mkdir(exist_ok=True)
        if lock.read_bytes() != raw:
            raise RuntimeError("writer identity changed during reclaim")
        os.rename(lock, archive / (uuid.uuid4().hex + ".json"))
        yield token
    finally:
        if _json(guard).get("token") == token:
            guard.unlink()


class RecoveryStore:
    """A single-writer store. Orphans are retained and never inferred as committed."""

    def __init__(self, directory: Path, bindings: dict, corpus, *, reclaim_token=None):
        if not bindings:
            raise ValueError("nonempty source bindings required")
        if any(e.split != "train" for e in corpus.episodes):
            raise ValueError("train-only corpus required")
        self.directory = Path(directory)
        self.bindings = dict(bindings)
        self.corpus = corpus
        self.token = uuid.uuid4().hex
        self.locked = False
        self.reclaim_token = reclaim_token

    def _check_reclaim_guard(self):
        guard = self.directory / "reclaim.guard"
        if guard.exists():
            if self.reclaim_token is None or _json(guard).get("token") != self.reclaim_token:
                raise RuntimeError("reclaim guard blocks writer acquisition")
        elif self.reclaim_token is not None:
            raise RuntimeError("reclaim authorization expired")

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self._check_reclaim_guard()
        lock = self.directory / "writer.lock"
        # A stale lock requires explicit operator adjudication; never race to steal it.
        try:
            with lock.open("x", encoding="utf-8") as stream:
                json.dump(
                    {
                        "pid": os.getpid(),
                        "create_time": psutil.Process().create_time(),
                        "token": self.token,
                    },
                    stream,
                )
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError:
            identity = _json(lock)
            try:
                owner = psutil.Process(identity["pid"])
                live = owner.create_time() == identity["create_time"]
            except psutil.NoSuchProcess:
                live = False
            raise RuntimeError(f"writer lock exists; live_identity={live}") from None
        try:
            self._check_reclaim_guard()
        except BaseException:
            if _json(lock).get("token") == self.token:
                lock.unlink()
            raise
        self.locked = True
        return self

    def __exit__(self, *args):
        lock = self.directory / "writer.lock"
        if self.locked and _json(lock).get("token") == self.token:
            lock.unlink()
        self.locked = False

    def commit(self, trainer, cursor: Cursor, epochs: int, ledger: dict):
        if not self.locked or _json(self.directory / "writer.lock").get("token") != self.token:
            raise RuntimeError("writer lock required")
        _validate(trainer, cursor, epochs)
        if trainer.corpus.digest != self.corpus.digest:
            raise ValueError("store corpus mismatch")
        previous = None
        latest = self.directory / "latest.json"
        if latest.exists():
            self.load_latest(restore_rng=False)
            previous = _json(latest)["receipt_sha256"]
        payload = {
            "schema": "taiji-r2-recovery-v1",
            "bindings": self.bindings,
            "trainer": trainer.checkpoint(),
            "cursor": asdict(cursor),
            "epochs": epochs,
            "episodes": [e.to_payload() for e in self.corpus.episodes],
            "source_files": list(self.corpus.source_files),
            "runtime": _runtime(),
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "ledger": ledger,
            "previous_receipt_sha256": previous,
        }
        payload["digest"] = content_digest(payload)
        name = f"commit-{trainer.episode_count:08d}-{uuid.uuid4().hex}"
        bundle = self.directory / (name + ".pt")
        _publish(bundle, lambda stream: torch.save(payload, stream))
        env = dict(os.environ, OMP_NUM_THREADS=str(torch.get_num_threads()))
        started = time.monotonic()
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.training.taiji_r2_dialogue_recovery",
                "--verify",
                str(bundle),
                "--bindings",
                json.dumps(self.bindings),
                "--threads",
                str(torch.get_num_threads()),
            ],
            cwd=Path(__file__).resolve().parents[2],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if process.returncode:
            raise RuntimeError("fresh-process recovery failed: " + process.stderr[-2000:])
        verified = json.loads(process.stdout)
        if verified["digest"] != payload["digest"]:
            raise ValueError("fresh-process receipt mismatch")
        receipt = {
            "bundle": bundle.name,
            "sha256": _sha(bundle),
            "digest": payload["digest"],
            "previous_receipt_sha256": previous,
            "verification": verified,
            "verification_wall_seconds": time.monotonic() - started,
            "path": str(bundle.resolve()),
            "checkpoint_digest": payload["trainer"]["checkpoint_digest"],
        }
        receipt_path = self.directory / (name + ".json")
        _publish_json(receipt_path, receipt)
        _publish_json(
            latest,
            {"receipt": receipt_path.name, "receipt_sha256": _sha(receipt_path)},
            replace=True,
        )
        return receipt

    def load_latest(self, *, restore_rng=True):
        manifest = _json(self.directory / "latest.json")
        receipt_name = manifest["receipt"]
        if Path(receipt_name).name != receipt_name:
            raise ValueError("invalid receipt path")
        receipt_path = self.directory / receipt_name
        if _sha(receipt_path) != manifest["receipt_sha256"]:
            raise ValueError("receipt hash mismatch")
        receipt = _json(receipt_path)
        if Path(receipt["bundle"]).name != receipt["bundle"]:
            raise ValueError("invalid bundle path")
        bundle = self.directory / receipt["bundle"]
        if _sha(bundle) != receipt["sha256"]:
            raise ValueError("bundle hash mismatch")
        trainer, cursor, payload = _restore_bundle(bundle, self.bindings)
        if trainer.corpus.digest != self.corpus.digest or receipt["digest"] != payload["digest"]:
            raise ValueError("corpus or receipt binding mismatch")
        if receipt["previous_receipt_sha256"] != payload["previous_receipt_sha256"]:
            raise ValueError("receipt chain mismatch")
        if restore_rng:
            random.setstate(payload["python_rng"])
            torch.set_rng_state(payload["torch_rng"])
        return trainer, cursor, payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", type=Path, required=True)
    parser.add_argument("--bindings", required=True)
    parser.add_argument("--threads", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    _, _, payload = _restore_bundle(args.verify, json.loads(args.bindings))
    print(json.dumps({"digest": payload["digest"], "cpu_seconds": time.process_time()}))


if __name__ == "__main__":
    main()
