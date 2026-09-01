"""Checkpointed CPU pilot training for the native Taiji foundation model.

The pilot intentionally trains the existing native predictive fabric in small
cursor-addressed chunks.  It is a development-period learner, not a provider
adapter: every update is made by :meth:`Taiji.learn_bytes`, and every
evaluation is read-only through :meth:`Taiji.score_bytes`.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .config import TaijiConfig
from .foundation_tasks import (
    DelayedMemoryCorpus,
    DelayedMemoryQuery,
    MemoryEpisode,
    SequencePredictionCorpus,
)
from .internalization import content_digest
from .model import Taiji

FOUNDATION_TRAINING_FORMAT = "taiji-native-foundation-training-v1"
FOUNDATION_TRAINING_VERSION = 1
FOUNDATION_TRAINING_PROFILES = ("smoke", "pilot", "foundation")
FOUNDATION_TRAINING_PROFILE_BUDGETS = {
    "smoke": (4_096, 1_024, 1_024),
    "pilot": (16_384, 4_096, 4_096),
    "foundation": (1_048_576, 131_072, 131_072),
}
MEMORY_TRAINING_FORMAT = "taiji-native-memory-training-v1"
MEMORY_TRAINING_VERSION = 1


def _text_from_record(record: Any) -> str | None:
    if isinstance(record, str):
        return record.strip() or None
    if not isinstance(record, Mapping):
        return None
    for key in ("text", "content", "input"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FoundationTrainingDataset:
    """Content-addressed train/holdout/retention bytes for one pilot."""

    train: bytes
    holdout: bytes
    retention: bytes
    source_files: tuple[tuple[str, str], ...] = ()
    partition_seed: int = 0
    profile: str = "smoke"

    def __post_init__(self) -> None:
        for field_name in ("train", "holdout", "retention"):
            value = getattr(self, field_name)
            if not isinstance(value, bytes) or not value:
                raise ValueError(f"{field_name} training data must contain non-empty bytes")
        if self.profile not in FOUNDATION_TRAINING_PROFILES:
            raise ValueError("unsupported foundation training profile")
        if int(self.partition_seed) <= 0:
            raise ValueError("partition_seed must be positive")
        normalized_sources = tuple(
            (str(path).strip(), str(digest).strip()) for path, digest in self.source_files
        )
        if any(not path or not digest for path, digest in normalized_sources):
            raise ValueError("source_files must contain non-empty path/digest pairs")
        object.__setattr__(self, "source_files", normalized_sources)
        object.__setattr__(self, "partition_seed", int(self.partition_seed))

    @property
    def sample_counts(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "holdout": len(self.holdout),
            "retention": len(self.retention),
        }

    @property
    def digest(self) -> str:
        return content_digest(
            {
                "format": FOUNDATION_TRAINING_FORMAT,
                "version": FOUNDATION_TRAINING_VERSION,
                "profile": self.profile,
                "partition_seed": self.partition_seed,
                "source_files": [list(item) for item in self.source_files],
                "train": self.train,
                "holdout": self.holdout,
                "retention": self.retention,
            }
        )

    @classmethod
    def from_jsonl(
        cls,
        paths: Iterable[str | Path],
        *,
        profile: str = "pilot",
        partition_seed: int = 11,
    ) -> FoundationTrainingDataset:
        profile = str(profile)
        if profile not in FOUNDATION_TRAINING_PROFILE_BUDGETS:
            raise ValueError("unsupported foundation training profile")
        train_budget, holdout_budget, retention_budget = FOUNDATION_TRAINING_PROFILE_BUDGETS[
            profile
        ]
        budgets = {
            "train": int(train_budget),
            "holdout": int(holdout_budget),
            "retention": int(retention_budget),
        }
        buffers = {partition: bytearray() for partition in budgets}
        seen_text_digests: set[str] = set()
        normalized_paths = tuple(Path(path) for path in paths)
        if not normalized_paths:
            raise ValueError("training dataset needs at least one JSONL path")
        for path in normalized_paths:
            if not path.is_file():
                raise FileNotFoundError(path)
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = _text_from_record(record)
                    if text is None:
                        continue
                    text_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                    if text_digest in seen_text_digests:
                        continue
                    seen_text_digests.add(text_digest)
                    bucket = int.from_bytes(
                        hashlib.sha256(f"{int(partition_seed)}\0{text}".encode()).digest()[:4],
                        "big",
                    ) % 10_000
                    partition = (
                        "train"
                        if bucket < 8_000
                        else "holdout"
                        if bucket < 9_000
                        else "retention"
                    )
                    remaining = budgets[partition] - len(buffers[partition])
                    if remaining > 0:
                        buffers[partition].extend(text.encode("utf-8")[:remaining])
                    if all(len(buffers[name]) >= budgets[name] for name in budgets):
                        break
            if all(len(buffers[name]) >= budgets[name] for name in budgets):
                break
        missing = {
            name: f"{len(buffers[name])}/{budgets[name]}"
            for name in budgets
            if len(buffers[name]) < budgets[name]
        }
        if missing:
            raise ValueError("training dataset did not meet byte budgets: " + json.dumps(missing))
        source_files = tuple((str(path), _file_digest(path)) for path in normalized_paths)
        return cls(
            train=bytes(buffers["train"]),
            holdout=bytes(buffers["holdout"]),
            retention=bytes(buffers["retention"]),
            source_files=source_files,
            partition_seed=int(partition_seed),
            profile=profile,
        )

    def as_sequence_corpus(self) -> SequencePredictionCorpus:
        return SequencePredictionCorpus(
            train=self.train,
            holdout=self.holdout,
            retention=self.retention,
        )


def _code_revision() -> str:
    try:
        return subprocess.check_output(
            ("git", "rev-parse", "HEAD"),
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "working-tree"


class FoundationTrainingRun:
    """A resumable native training run with atomic checkpoint writes."""

    def __init__(
        self,
        model: Taiji,
        dataset: FoundationTrainingDataset,
        *,
        output_dir: str | Path,
        profile: str | None = None,
        model_tier: str = "micro",
        epochs: int = 1,
        chunk_bytes: int = 1_024,
        checkpoint_interval: int = 1,
        code_revision: str | None = None,
        parent_checkpoint_digest: str | None = None,
        parent_holdout_bpb: float | None = None,
    ) -> None:
        if profile is not None and profile not in FOUNDATION_TRAINING_PROFILES:
            raise ValueError("unsupported foundation training profile")
        if int(epochs) <= 0 or int(chunk_bytes) <= 0 or int(checkpoint_interval) <= 0:
            raise ValueError("training epochs, chunk_bytes, and checkpoint_interval must be positive")
        self.model = model
        self.dataset = dataset
        self.output_dir = Path(output_dir)
        self.profile = profile or dataset.profile
        self.model_tier = str(model_tier)
        self.total_epochs = int(epochs)
        self.chunk_bytes = int(chunk_bytes)
        self.checkpoint_interval = int(checkpoint_interval)
        self.code_revision = str(code_revision or _code_revision())
        self.parent_checkpoint_digest = parent_checkpoint_digest or content_digest(
            model.checkpoint()
        )
        self.parent_holdout_bpb = (
            float(parent_holdout_bpb)
            if parent_holdout_bpb is not None
            else self._bpb(model, dataset.holdout)
        )
        if not math.isfinite(self.parent_holdout_bpb):
            raise ValueError("parent_holdout_bpb must be finite")
        self.epoch = 0
        self.cursor = 0
        self.global_step = 0
        self.history: list[dict[str, Any]] = []
        self.best_holdout_bpb = 1e30
        self.started_from_checkpoint = False

    @property
    def last_checkpoint_path(self) -> Path:
        return self.output_dir / "last.pt"

    @property
    def best_checkpoint_path(self) -> Path:
        return self.output_dir / "best-holdout.pt"

    @property
    def parent_checkpoint_path(self) -> Path:
        return self.output_dir / "parent.pt"

    @staticmethod
    def _bpb(model: Taiji, data: bytes) -> float:
        before = content_digest(model.checkpoint())
        score = model.score_bytes(data)
        after = content_digest(model.checkpoint())
        if before != after:
            raise RuntimeError("Taiji score_bytes mutated the checkpoint")
        return float(score["mean_surprise"]) / math.log(2.0)

    def _checkpoint_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": FOUNDATION_TRAINING_FORMAT,
            "version": FOUNDATION_TRAINING_VERSION,
            "profile": self.profile,
            "model_tier": self.model_tier,
            "model": self.model.checkpoint(),
            "dataset_digest": self.dataset.digest,
            "dataset_sample_counts": self.dataset.sample_counts,
            "parent_checkpoint_digest": self.parent_checkpoint_digest,
            "parent_holdout_bpb": self.parent_holdout_bpb,
            "epoch": self.epoch,
            "cursor": self.cursor,
            "global_step": self.global_step,
            "total_epochs": self.total_epochs,
            "chunk_bytes": self.chunk_bytes,
            "checkpoint_interval": self.checkpoint_interval,
            "history": list(self.history),
            "best_holdout_bpb": self.best_holdout_bpb,
            "code_revision": self.code_revision,
        }
        payload["checkpoint_digest"] = content_digest(payload)
        return payload

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        torch.save(self._checkpoint_payload(), temporary)
        temporary.replace(target)
        return target

    def _save_progress(
        self,
        *,
        evaluate: bool,
        train_metrics: Mapping[str, float] | None = None,
    ) -> None:
        holdout_bpb = None
        retention_bpb = None
        if evaluate:
            holdout_bpb = self._bpb(self.model, self.dataset.holdout)
            retention_bpb = self._bpb(self.model, self.dataset.retention)
            record: dict[str, Any] = {
                "epoch": self.epoch,
                "cursor": self.cursor,
                "global_step": self.global_step,
                "holdout_bpb": holdout_bpb,
                "retention_bpb": retention_bpb,
            }
            if train_metrics is not None:
                record["train_accuracy"] = float(train_metrics["online_accuracy"])
                record["train_observations"] = int(train_metrics["observations"])
            self.history.append(record)
            if holdout_bpb < self.best_holdout_bpb:
                self.best_holdout_bpb = holdout_bpb
                self.save(self.best_checkpoint_path)
        self.save(self.last_checkpoint_path)

    def run(self) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not self.started_from_checkpoint and not self.parent_checkpoint_path.exists():
            self.save(self.parent_checkpoint_path)
        while self.epoch < self.total_epochs:
            while self.cursor < len(self.dataset.train):
                end = min(self.cursor + self.chunk_bytes, len(self.dataset.train))
                metrics = self.model.learn_bytes(self.dataset.train[self.cursor:end], epochs=1)
                self.cursor = end
                self.global_step += 1
                if (
                    self.global_step % self.checkpoint_interval == 0
                    or self.cursor == len(self.dataset.train)
                ):
                    self._save_progress(evaluate=True, train_metrics=metrics)
                else:
                    self.save(self.last_checkpoint_path)
            self.epoch += 1
            self.cursor = 0
            self._save_progress(evaluate=True)
        final_holdout = self._bpb(self.model, self.dataset.holdout)
        final_retention = self._bpb(self.model, self.dataset.retention)
        child_digest = content_digest(self.model.checkpoint())
        report = {
            "format": FOUNDATION_TRAINING_FORMAT,
            "version": FOUNDATION_TRAINING_VERSION,
            "status": "completed",
            "profile": self.profile,
            "model_tier": self.model_tier,
            "dataset_digest": self.dataset.digest,
            "dataset_sample_counts": self.dataset.sample_counts,
            "parent_checkpoint_digest": self.parent_checkpoint_digest,
            "parent_holdout_bpb": self.parent_holdout_bpb,
            "child_checkpoint_digest": child_digest,
            "epoch": self.epoch,
            "cursor": self.cursor,
            "global_step": self.global_step,
            "total_epochs": self.total_epochs,
            "best_holdout_bpb": self.best_holdout_bpb,
            "final_holdout_bpb": final_holdout,
            "final_retention_bpb": final_retention,
            "history": list(self.history),
            "checkpoint_paths": {
                "parent": str(self.parent_checkpoint_path),
                "last": str(self.last_checkpoint_path),
                "best_holdout": str(self.best_checkpoint_path),
            },
            "code_revision": self.code_revision,
            "started_from_checkpoint": self.started_from_checkpoint,
        }
        return report

    def evaluate_only(self) -> dict[str, Any]:
        before = content_digest(self.model.checkpoint())
        holdout = self._bpb(self.model, self.dataset.holdout)
        retention = self._bpb(self.model, self.dataset.retention)
        after = content_digest(self.model.checkpoint())
        if before != after:
            raise RuntimeError("eval-only scoring mutated the checkpoint")
        return {
            "format": FOUNDATION_TRAINING_FORMAT,
            "version": FOUNDATION_TRAINING_VERSION,
            "status": "evaluated",
            "profile": self.profile,
            "model_tier": self.model_tier,
            "dataset_digest": self.dataset.digest,
            "dataset_sample_counts": self.dataset.sample_counts,
            "checkpoint_digest": before,
            "holdout_bpb": holdout,
            "retention_bpb": retention,
            "code_revision": self.code_revision,
            "checkpoint_read_only": True,
        }

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        dataset: FoundationTrainingDataset,
        *,
        output_dir: str | Path | None = None,
        epochs: int | None = None,
        code_revision: str | None = None,
    ) -> FoundationTrainingRun:
        checkpoint_path = Path(path)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise ValueError("training checkpoint must contain a mapping")
        if payload.get("format") != FOUNDATION_TRAINING_FORMAT:
            raise ValueError("unsupported foundation training checkpoint format")
        if int(payload.get("version", -1)) != FOUNDATION_TRAINING_VERSION:
            raise ValueError("unsupported foundation training checkpoint version")
        expected = content_digest(
            {key: value for key, value in payload.items() if key != "checkpoint_digest"}
        )
        if str(payload.get("checkpoint_digest", "")) != expected:
            raise ValueError("foundation training checkpoint digest mismatch")
        if str(payload.get("dataset_digest")) != dataset.digest:
            raise ValueError("foundation training dataset digest mismatch")
        model_payload = payload.get("model")
        if not isinstance(model_payload, Mapping):
            raise ValueError("foundation training checkpoint is missing model payload")
        config = TaijiConfig.from_dict(dict(model_payload["config"]))
        model = Taiji(config, episode_id="foundation-resume")
        model.restore(model_payload)
        run = cls(
            model,
            dataset,
            output_dir=output_dir or checkpoint_path.parent,
            profile=str(payload["profile"]),
            model_tier=str(payload["model_tier"]),
            epochs=int(epochs if epochs is not None else payload["total_epochs"]),
            chunk_bytes=int(payload["chunk_bytes"]),
            checkpoint_interval=int(payload["checkpoint_interval"]),
            code_revision=code_revision or str(payload.get("code_revision", "working-tree")),
            parent_checkpoint_digest=str(payload["parent_checkpoint_digest"]),
            parent_holdout_bpb=float(payload["parent_holdout_bpb"]),
        )
        run.epoch = int(payload["epoch"])
        run.cursor = int(payload["cursor"])
        run.global_step = int(payload["global_step"])
        run.history = [dict(item) for item in payload.get("history", ())]
        run.best_holdout_bpb = float(payload.get("best_holdout_bpb", 1e30))
        run.started_from_checkpoint = True
        return run


def _memory_corpus_digest(corpus: DelayedMemoryCorpus) -> str:
    return content_digest(
        {
            "format": MEMORY_TRAINING_FORMAT,
            "version": MEMORY_TRAINING_VERSION,
            "train": [
                {
                    "memory_id": item.memory_id,
                    "cue": item.cue,
                    "action": item.action,
                    "outcome": item.outcome,
                }
                for item in corpus.train
            ],
            "holdout": [
                {
                    "query_id": item.query_id,
                    "cue": item.cue,
                    "expected_action": item.expected_action,
                }
                for item in corpus.holdout
            ],
            "retention": [
                {
                    "query_id": item.query_id,
                    "cue": item.cue,
                    "expected_action": item.expected_action,
                }
                for item in corpus.retention
            ],
        }
    )


class MemoryTrainingRun:
    """A resumable cue/episode memory pilot using only native memory writes."""

    def __init__(
        self,
        model: Taiji,
        corpus: DelayedMemoryCorpus,
        *,
        output_dir: str | Path,
        model_tier: str = "memory",
        epochs: int = 1,
        checkpoint_interval: int = 1,
        parent_checkpoint_digest: str | None = None,
        parent_holdout_recall: float | None = None,
        code_revision: str | None = None,
    ) -> None:
        if not isinstance(corpus, DelayedMemoryCorpus):
            raise TypeError("memory training corpus must be DelayedMemoryCorpus")
        if int(epochs) <= 0 or int(checkpoint_interval) <= 0:
            raise ValueError("memory training epochs and checkpoint_interval must be positive")
        self.model = model
        self.corpus = corpus
        self.output_dir = Path(output_dir)
        self.model_tier = str(model_tier)
        self.total_epochs = int(epochs)
        self.checkpoint_interval = int(checkpoint_interval)
        self.code_revision = str(code_revision or _code_revision())
        self.parent_checkpoint_digest = parent_checkpoint_digest or content_digest(
            model.checkpoint()
        )
        self.actions = tuple(dict.fromkeys(item.action for item in corpus.train))
        if len(self.actions) < 2:
            raise ValueError("memory training needs at least two action classes")
        self.parent_holdout_recall = (
            float(parent_holdout_recall)
            if parent_holdout_recall is not None
            else self._recall_accuracy(corpus.holdout, use_memory=True)
        )
        if not math.isfinite(self.parent_holdout_recall):
            raise ValueError("parent_holdout_recall must be finite")
        self.epoch = 0
        self.cursor = 0
        self.global_step = 0
        self.history: list[dict[str, Any]] = []
        self.best_holdout_recall = 0.0
        self.started_from_checkpoint = False

    @property
    def parent_checkpoint_path(self) -> Path:
        return self.output_dir / "parent.pt"

    @property
    def last_checkpoint_path(self) -> Path:
        return self.output_dir / "last.pt"

    @property
    def best_checkpoint_path(self) -> Path:
        return self.output_dir / "best-holdout.pt"

    @property
    def corpus_digest(self) -> str:
        return _memory_corpus_digest(self.corpus)

    @staticmethod
    def _persistent_digest(model: Taiji) -> str:
        checkpoint = model.checkpoint()
        return content_digest(
            {
                "fabric": checkpoint["fabric"],
                "motor": checkpoint["motor"],
                "memory": checkpoint["memory"],
            }
        )

    def _recall_accuracy(
        self,
        queries: Sequence[DelayedMemoryQuery],
        *,
        use_memory: bool,
    ) -> float:
        persistent_before = self._persistent_digest(self.model)
        correct = 0
        for query in queries:
            self.model.reset_dynamics(episode_id=f"m1-f2-query-{query.query_id}")
            self.model.observe(
                self.model.config.boundary_symbol,
                learn=False,
                learn_motor=False,
                use_memory=use_memory,
            )
            self.model.observe(
                query.cue,
                learn=False,
                learn_motor=False,
                use_memory=use_memory,
            )
            probabilities = self.model.snapshot().motor_probabilities
            prediction = max(
                self.actions,
                key=lambda action: float(probabilities[action].item()),
            )
            correct += int(prediction == query.expected_action)
        persistent_after = self._persistent_digest(self.model)
        if persistent_before != persistent_after:
            raise RuntimeError("memory holdout evaluation mutated persistent state")
        return correct / len(queries)

    def _train_episode(self, episode: MemoryEpisode) -> None:
        self.model.reset_dynamics(episode_id=f"m1-f2-train-{episode.memory_id}")
        self.model.observe(
            self.model.config.boundary_symbol,
            learn=False,
            learn_motor=False,
            use_memory=False,
        )
        self.model.observe(episode.cue, learn=False, learn_motor=False, use_memory=False)
        self.model.act((episode.action,), sample=False)
        self.model.settle_action(1.0, learn=False, learn_memory=True)
        self.model.observe(episode.outcome, learn=False, learn_motor=False, use_memory=False)

    def _checkpoint_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": MEMORY_TRAINING_FORMAT,
            "version": MEMORY_TRAINING_VERSION,
            "model_tier": self.model_tier,
            "model": self.model.checkpoint(),
            "corpus_digest": self.corpus_digest,
            "corpus_sample_counts": {
                "train": len(self.corpus.train),
                "holdout": len(self.corpus.holdout),
                "retention": len(self.corpus.retention),
            },
            "parent_checkpoint_digest": self.parent_checkpoint_digest,
            "parent_holdout_recall": self.parent_holdout_recall,
            "epoch": self.epoch,
            "cursor": self.cursor,
            "global_step": self.global_step,
            "total_epochs": self.total_epochs,
            "checkpoint_interval": self.checkpoint_interval,
            "history": list(self.history),
            "best_holdout_recall": self.best_holdout_recall,
            "code_revision": self.code_revision,
        }
        payload["checkpoint_digest"] = content_digest(payload)
        return payload

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        torch.save(self._checkpoint_payload(), temporary)
        temporary.replace(target)
        return target

    def _save_progress(self, *, train_episode: MemoryEpisode | None = None) -> None:
        holdout = self._recall_accuracy(self.corpus.holdout, use_memory=True)
        retention = self._recall_accuracy(self.corpus.retention, use_memory=True)
        lesion = self._recall_accuracy(self.corpus.holdout, use_memory=False)
        record: dict[str, Any] = {
            "epoch": self.epoch,
            "cursor": self.cursor,
            "global_step": self.global_step,
            "holdout_recall": holdout,
            "retention_recall": retention,
            "memory_lesion_recall": lesion,
        }
        if train_episode is not None:
            record["train_memory_id"] = train_episode.memory_id
        self.history.append(record)
        if holdout > self.best_holdout_recall:
            self.best_holdout_recall = holdout
            self.save(self.best_checkpoint_path)
        self.save(self.last_checkpoint_path)

    def run(self) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not self.started_from_checkpoint and not self.parent_checkpoint_path.exists():
            self.save(self.parent_checkpoint_path)
        while self.epoch < self.total_epochs:
            while self.cursor < len(self.corpus.train):
                episode = self.corpus.train[self.cursor]
                self._train_episode(episode)
                self.cursor += 1
                self.global_step += 1
                if (
                    self.global_step % self.checkpoint_interval == 0
                    or self.cursor == len(self.corpus.train)
                ):
                    self._save_progress(train_episode=episode)
                else:
                    self.save(self.last_checkpoint_path)
            self.epoch += 1
            self.cursor = 0
            self._save_progress()
        final_holdout = self._recall_accuracy(self.corpus.holdout, use_memory=True)
        final_retention = self._recall_accuracy(self.corpus.retention, use_memory=True)
        final_lesion = self._recall_accuracy(self.corpus.holdout, use_memory=False)
        return {
            "format": MEMORY_TRAINING_FORMAT,
            "version": MEMORY_TRAINING_VERSION,
            "status": "completed",
            "model_tier": self.model_tier,
            "corpus_digest": self.corpus_digest,
            "corpus_sample_counts": {
                "train": len(self.corpus.train),
                "holdout": len(self.corpus.holdout),
                "retention": len(self.corpus.retention),
            },
            "parent_checkpoint_digest": self.parent_checkpoint_digest,
            "parent_holdout_recall": self.parent_holdout_recall,
            "child_checkpoint_digest": content_digest(self.model.checkpoint()),
            "epoch": self.epoch,
            "cursor": self.cursor,
            "global_step": self.global_step,
            "total_epochs": self.total_epochs,
            "best_holdout_recall": self.best_holdout_recall,
            "final_holdout_recall": final_holdout,
            "final_retention_recall": final_retention,
            "final_memory_lesion_recall": final_lesion,
            "holdout_updates": 0,
            "history": list(self.history),
            "checkpoint_paths": {
                "parent": str(self.parent_checkpoint_path),
                "last": str(self.last_checkpoint_path),
                "best_holdout": str(self.best_checkpoint_path),
            },
            "code_revision": self.code_revision,
            "started_from_checkpoint": self.started_from_checkpoint,
        }

    def evaluate_only(self) -> dict[str, Any]:
        before = self._persistent_digest(self.model)
        holdout = self._recall_accuracy(self.corpus.holdout, use_memory=True)
        retention = self._recall_accuracy(self.corpus.retention, use_memory=True)
        after = self._persistent_digest(self.model)
        if before != after:
            raise RuntimeError("memory eval-only mutated persistent state")
        return {
            "format": MEMORY_TRAINING_FORMAT,
            "version": MEMORY_TRAINING_VERSION,
            "status": "evaluated",
            "model_tier": self.model_tier,
            "corpus_digest": self.corpus_digest,
            "corpus_sample_counts": {
                "train": len(self.corpus.train),
                "holdout": len(self.corpus.holdout),
                "retention": len(self.corpus.retention),
            },
            "checkpoint_digest": content_digest(self.model.checkpoint()),
            "holdout_recall": holdout,
            "retention_recall": retention,
            "checkpoint_read_only": True,
            "code_revision": self.code_revision,
        }

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        corpus: DelayedMemoryCorpus,
        *,
        output_dir: str | Path | None = None,
        epochs: int | None = None,
        code_revision: str | None = None,
    ) -> MemoryTrainingRun:
        checkpoint_path = Path(path)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise ValueError("memory training checkpoint must contain a mapping")
        if payload.get("format") != MEMORY_TRAINING_FORMAT:
            raise ValueError("unsupported memory training checkpoint format")
        if int(payload.get("version", -1)) != MEMORY_TRAINING_VERSION:
            raise ValueError("unsupported memory training checkpoint version")
        expected = content_digest(
            {key: value for key, value in payload.items() if key != "checkpoint_digest"}
        )
        if str(payload.get("checkpoint_digest", "")) != expected:
            raise ValueError("memory training checkpoint digest mismatch")
        if str(payload.get("corpus_digest")) != _memory_corpus_digest(corpus):
            raise ValueError("memory training corpus digest mismatch")
        model_payload = payload.get("model")
        if not isinstance(model_payload, Mapping):
            raise ValueError("memory training checkpoint is missing model payload")
        model = Taiji(
            TaijiConfig.from_dict(dict(model_payload["config"])),
            episode_id="memory-resume",
        )
        model.restore(model_payload)
        run = cls(
            model,
            corpus,
            output_dir=output_dir or checkpoint_path.parent,
            model_tier=str(payload["model_tier"]),
            epochs=int(epochs if epochs is not None else payload["total_epochs"]),
            checkpoint_interval=int(payload["checkpoint_interval"]),
            parent_checkpoint_digest=str(payload["parent_checkpoint_digest"]),
            parent_holdout_recall=float(payload["parent_holdout_recall"]),
            code_revision=code_revision or str(payload.get("code_revision", "working-tree")),
        )
        run.epoch = int(payload["epoch"])
        run.cursor = int(payload["cursor"])
        run.global_step = int(payload["global_step"])
        run.history = [dict(item) for item in payload.get("history", ())]
        run.best_holdout_recall = float(payload.get("best_holdout_recall", 0.0))
        run.started_from_checkpoint = True
        return run


__all__ = [
    "MEMORY_TRAINING_FORMAT",
    "MEMORY_TRAINING_VERSION",
    "FOUNDATION_TRAINING_FORMAT",
    "FOUNDATION_TRAINING_PROFILE_BUDGETS",
    "FOUNDATION_TRAINING_PROFILES",
    "FOUNDATION_TRAINING_VERSION",
    "FoundationTrainingDataset",
    "FoundationTrainingRun",
    "MemoryTrainingRun",
]
