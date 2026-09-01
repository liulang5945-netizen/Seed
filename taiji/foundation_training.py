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
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .config import TaijiConfig
from .contracts import WorldTransition
from .foundation_tasks import (
    DelayedMemoryCorpus,
    DelayedMemoryQuery,
    GoalActionCorpus,
    GoalActionEpisode,
    MemoryEpisode,
    SequencePredictionCorpus,
    WorldTransitionCorpus,
)
from .internalization import content_digest
from .model import Taiji
from .world_learning import WorldDynamicsLearner, WorldSchema, WorldSchemaRegistry

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
WORLD_ACTION_TRAINING_FORMAT = "taiji-native-world-action-training-v1"
WORLD_ACTION_TRAINING_VERSION = 1
JOINT_TRAINING_FORMAT = "taiji-native-joint-training-v1"
JOINT_TRAINING_VERSION = 1


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


def _world_action_corpus_digest(
    world_corpus: WorldTransitionCorpus,
    goal_corpus: GoalActionCorpus,
) -> str:
    return content_digest(
        {
            "format": WORLD_ACTION_TRAINING_FORMAT,
            "version": WORLD_ACTION_TRAINING_VERSION,
            "world": {
                partition: [case.to_payload() for case in getattr(world_corpus, partition)]
                for partition in ("train", "holdout", "retention")
            },
            "goal": {
                partition: [episode.__dict__ for episode in getattr(goal_corpus, partition)]
                for partition in ("train", "holdout", "retention")
            },
        }
    )


def _world_learner_payload(learner: WorldDynamicsLearner) -> dict[str, Any]:
    return {
        "schema": learner.schema.payload(),
        "schema_registry": learner.schema_registry.checkpoint(),
        "hidden_dim": learner.hidden_dim,
        "online_updates": learner.online_updates,
        "transition_acceptances": learner.transition_acceptances,
        "transition_rejections": learner.transition_rejections,
        "schema_evolution_count": learner.schema_evolution_count,
        "state_dict": {
            name: tensor.detach().cpu().clone() for name, tensor in learner.state_dict().items()
        },
        "schema_snapshots": {
            str(version): {
                name: tensor.detach().cpu().clone() for name, tensor in snapshot.items()
            }
            for version, snapshot in learner._schema_snapshots.items()
        },
    }


def _world_learner_from_payload(payload: Mapping[str, Any]) -> WorldDynamicsLearner:
    schema = WorldSchema.from_payload(dict(payload["schema"]))
    registry = WorldSchemaRegistry.from_checkpoint(dict(payload["schema_registry"]))
    if registry.schema != schema:
        raise ValueError("world action checkpoint schema registry does not match learner schema")
    learner = WorldDynamicsLearner(
        schema,
        hidden_dim=int(payload["hidden_dim"]),
        seed=0,
        schema_registry=registry,
    )
    learner.load_state_dict(payload["state_dict"])
    learner.online_updates = int(payload.get("online_updates", 0))
    learner.transition_acceptances = int(payload.get("transition_acceptances", 0))
    learner.transition_rejections = int(payload.get("transition_rejections", 0))
    learner.schema_evolution_count = int(payload.get("schema_evolution_count", 0))
    snapshots = payload.get("schema_snapshots")
    if isinstance(snapshots, Mapping):
        learner._schema_snapshots = {
            int(version): {
                str(name): tensor.detach().cpu().clone() for name, tensor in snapshot.items()
            }
            for version, snapshot in snapshots.items()
            if isinstance(snapshot, Mapping)
        }
        learner._schema_snapshots.setdefault(
            learner.schema_registry.active_version,
            learner._snapshot_state_dict(),
        )
    return learner


def _world_action_persistent_digest(
    model: Taiji,
    world_learner: WorldDynamicsLearner,
) -> str:
    checkpoint = model.checkpoint()
    return content_digest(
        {
            "model": {
                "fabric": checkpoint["fabric"],
                "motor": checkpoint["motor"],
                "memory": checkpoint["memory"],
            },
            "world": _world_learner_payload(world_learner),
        }
    )


def _world_action_error(
    learner: WorldDynamicsLearner,
    cases: Sequence[Any],
) -> float:
    errors: list[float] = []
    schema = learner.schema
    for case in cases:
        prediction = learner.predict(case.initial, case.action, register_parameters=False)
        expected_success = float(
            case.expected_outcome.success
            if case.expected_outcome.success is not None
            else case.expected_outcome.reward > 0.0
        )
        errors.append(
            schema.normalized_state_error(prediction.state, case.expected_state)
            + (prediction.reward - float(case.expected_outcome.reward)) ** 2
            + (prediction.success_probability - expected_success) ** 2
        )
    return sum(errors) / len(errors)


def _cold_start_action_organ(model: Taiji) -> None:
    with torch.no_grad():
        model.motor.synapses.edge_weight.zero_()
        model.motor.bias.zero_()
        model.motor.reward_baseline = 0.0
        model.motor.reward_updates = 0


def _train_goal_episode(
    model: Taiji,
    episode: GoalActionEpisode,
    *,
    learn: bool,
) -> bool:
    model.reset_dynamics(episode_id=f"m1-f3-train-{episode.episode_id}")
    model.observe(model.config.boundary_symbol, learn=False, learn_motor=False, use_memory=False)
    model.observe(episode.cue, learn=learn, learn_motor=False, use_memory=False)
    decision = model.act(
        tuple(sorted((episode.preferred_action, episode.alternate_action))),
        sample=True,
    )
    success = decision.action_symbol == episode.preferred_action
    model.settle_action(1.0 if success else -1.0, learn=learn, learn_memory=False)
    model.observe(model.config.boundary_symbol, learn=False, learn_motor=False, use_memory=False)
    return success


def _goal_action_accuracy(model: Taiji, episodes: Sequence[GoalActionEpisode]) -> float:
    correct = 0
    for episode in episodes:
        model.reset_dynamics(episode_id=f"m1-f3-eval-{episode.episode_id}")
        model.observe(model.config.boundary_symbol, learn=False, learn_motor=False, use_memory=False)
        model.observe(episode.cue, learn=False, learn_motor=False, use_memory=False)
        decision = model.act(
            tuple(sorted((episode.preferred_action, episode.alternate_action))),
            sample=False,
        )
        correct += int(decision.action_symbol == episode.preferred_action)
        model.settle_action(0.0, learn=False, learn_memory=False)
        model.observe(model.config.boundary_symbol, learn=False, learn_motor=False, use_memory=False)
    return correct / len(episodes)


class WorldActionTrainingRun:
    """A resumable native world-transition and goal-credit training run."""

    def __init__(
        self,
        model: Taiji,
        world_learner: WorldDynamicsLearner,
        world_corpus: WorldTransitionCorpus,
        goal_corpus: GoalActionCorpus,
        *,
        output_dir: str | Path,
        model_tier: str = "world-action",
        epochs: int = 1,
        checkpoint_interval: int = 1,
        world_learning_rate: float = 0.01,
        world_repeats: int = 4,
        parent_checkpoint_digest: str | None = None,
        parent_world_error: float | None = None,
        parent_goal_success: float | None = None,
        code_revision: str | None = None,
    ) -> None:
        if not isinstance(world_corpus, WorldTransitionCorpus):
            raise TypeError("world action training requires WorldTransitionCorpus")
        if not isinstance(goal_corpus, GoalActionCorpus):
            raise TypeError("world action training requires GoalActionCorpus")
        if not isinstance(world_learner, WorldDynamicsLearner):
            raise TypeError("world action training requires WorldDynamicsLearner")
        if int(epochs) <= 0 or int(checkpoint_interval) <= 0:
            raise ValueError("world action training epochs and checkpoint_interval must be positive")
        if float(world_learning_rate) <= 0.0 or int(world_repeats) <= 0:
            raise ValueError("world action learning settings must be positive")
        self.model = model
        self.world_learner = world_learner
        self.world_corpus = world_corpus
        self.goal_corpus = goal_corpus
        self.output_dir = Path(output_dir)
        self.model_tier = str(model_tier)
        self.total_epochs = int(epochs)
        self.checkpoint_interval = int(checkpoint_interval)
        self.world_learning_rate = float(world_learning_rate)
        self.world_repeats = int(world_repeats)
        self.code_revision = str(code_revision or _code_revision())
        self.parent_model_payload = deepcopy(model.checkpoint())
        self.parent_world_payload = deepcopy(_world_learner_payload(world_learner))
        self.parent_checkpoint_digest = parent_checkpoint_digest or content_digest(
            {"model": self.parent_model_payload, "world": self.parent_world_payload}
        )
        self.parent_world_error = (
            float(parent_world_error)
            if parent_world_error is not None
            else _world_action_error(world_learner, world_corpus.holdout)
        )
        self.parent_goal_success = (
            float(parent_goal_success)
            if parent_goal_success is not None
            else _goal_action_accuracy(model, goal_corpus.holdout)
        )
        self.epoch = 0
        self.phase = "world"
        self.world_cursor = 0
        self.goal_cursor = 0
        self.global_step = 0
        self.history: list[dict[str, Any]] = []
        self.best_holdout_score = 0.0
        self.started_from_checkpoint = False

    @property
    def corpus_digest(self) -> str:
        return _world_action_corpus_digest(self.world_corpus, self.goal_corpus)

    @property
    def parent_checkpoint_path(self) -> Path:
        return self.output_dir / "parent.pt"

    @property
    def last_checkpoint_path(self) -> Path:
        return self.output_dir / "last.pt"

    @property
    def best_checkpoint_path(self) -> Path:
        return self.output_dir / "best-holdout.pt"

    def _checkpoint_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": WORLD_ACTION_TRAINING_FORMAT,
            "version": WORLD_ACTION_TRAINING_VERSION,
            "model_tier": self.model_tier,
            "model": self.model.checkpoint(),
            "world_learner": _world_learner_payload(self.world_learner),
            "corpus_digest": self.corpus_digest,
            "corpus_sample_counts": {
                "world": self.world_corpus.sample_counts,
                "goal": self.goal_corpus.sample_counts,
            },
            "parent_checkpoint_digest": self.parent_checkpoint_digest,
            "parent_model": self.parent_model_payload,
            "parent_world_learner": self.parent_world_payload,
            "parent_world_error": self.parent_world_error,
            "parent_goal_success": self.parent_goal_success,
            "epoch": self.epoch,
            "phase": self.phase,
            "world_cursor": self.world_cursor,
            "goal_cursor": self.goal_cursor,
            "global_step": self.global_step,
            "total_epochs": self.total_epochs,
            "checkpoint_interval": self.checkpoint_interval,
            "world_learning_rate": self.world_learning_rate,
            "world_repeats": self.world_repeats,
            "history": list(self.history),
            "best_holdout_score": self.best_holdout_score,
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

    def _holdout_metrics(self) -> tuple[float, float, float, bool]:
        persistent_before = _world_action_persistent_digest(self.model, self.world_learner)
        world_error = _world_action_error(self.world_learner, self.world_corpus.holdout)
        goal_success = _goal_action_accuracy(self.model, self.goal_corpus.holdout)
        persistent_after = _world_action_persistent_digest(self.model, self.world_learner)
        return world_error, goal_success, persistent_before, persistent_before == persistent_after

    def _save_progress(self, *, train_kind: str | None = None, train_success: bool | None = None) -> None:
        world_error, goal_success, _persistent, read_only = self._holdout_metrics()
        if not read_only:
            raise RuntimeError("world action holdout evaluation mutated persistent state")
        score = (self.parent_world_error - world_error) + (goal_success - self.parent_goal_success)
        record: dict[str, Any] = {
            "epoch": self.epoch,
            "phase": self.phase,
            "world_cursor": self.world_cursor,
            "goal_cursor": self.goal_cursor,
            "global_step": self.global_step,
            "world_holdout_error": world_error,
            "goal_holdout_success": goal_success,
            "joint_holdout_gain": score,
        }
        if train_kind is not None:
            record["train_kind"] = train_kind
        if train_success is not None:
            record["train_success"] = bool(train_success)
        self.history.append(record)
        if score > self.best_holdout_score:
            self.best_holdout_score = score
            self.save(self.best_checkpoint_path)
        self.save(self.last_checkpoint_path)

    def run(self) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not self.started_from_checkpoint and not self.parent_checkpoint_path.exists():
            self.save(self.parent_checkpoint_path)
            self.save(self.best_checkpoint_path)
        while self.epoch < self.total_epochs:
            self.phase = "world"
            while self.world_cursor < len(self.world_corpus.train):
                case = self.world_corpus.train[self.world_cursor]
                transition = WorldTransition(
                    before=case.initial,
                    action=case.action,
                    after=case.expected_state,
                    outcome=case.expected_outcome,
                )
                self.world_learner.online_update(
                    transition,
                    learning_rate=self.world_learning_rate,
                    repeats=self.world_repeats,
                    register_parameters=True,
                )
                self.world_cursor += 1
                self.global_step += 1
                if (
                    self.global_step % self.checkpoint_interval == 0
                    or self.world_cursor == len(self.world_corpus.train)
                ):
                    self._save_progress(train_kind="world")
                else:
                    self.save(self.last_checkpoint_path)
            self.phase = "goal"
            while self.goal_cursor < len(self.goal_corpus.train):
                episode = self.goal_corpus.train[self.goal_cursor]
                success = _train_goal_episode(self.model, episode, learn=True)
                self.goal_cursor += 1
                self.global_step += 1
                if (
                    self.global_step % self.checkpoint_interval == 0
                    or self.goal_cursor == len(self.goal_corpus.train)
                ):
                    self._save_progress(train_kind="goal", train_success=success)
                else:
                    self.save(self.last_checkpoint_path)
            self.epoch += 1
            self.phase = "world"
            self.world_cursor = 0
            self.goal_cursor = 0
            self._save_progress()
        final_world_error, final_goal_success, _persistent, read_only = self._holdout_metrics()
        if not read_only:
            raise RuntimeError("world action final holdout evaluation mutated persistent state")
        lesion = Taiji.from_checkpoint(self.parent_model_payload)
        _cold_start_action_organ(lesion)
        for episode in self.goal_corpus.train:
            _train_goal_episode(lesion, episode, learn=False)
        credit_lesion = _goal_action_accuracy(lesion, self.goal_corpus.holdout)
        return {
            "format": WORLD_ACTION_TRAINING_FORMAT,
            "version": WORLD_ACTION_TRAINING_VERSION,
            "status": "completed",
            "model_tier": self.model_tier,
            "corpus_digest": self.corpus_digest,
            "corpus_sample_counts": {
                "world": self.world_corpus.sample_counts,
                "goal": self.goal_corpus.sample_counts,
            },
            "parent_checkpoint_digest": self.parent_checkpoint_digest,
            "parent_world_error": self.parent_world_error,
            "final_world_error": final_world_error,
            "parent_goal_success": self.parent_goal_success,
            "final_goal_success": final_goal_success,
            "final_credit_lesion_success": credit_lesion,
            "joint_holdout_gain": (
                self.parent_world_error
                - final_world_error
                + final_goal_success
                - self.parent_goal_success
            ),
            "world_online_updates": self.world_learner.online_updates,
            "world_transition_rejections": self.world_learner.transition_rejections,
            "holdout_updates": 0,
            "epoch": self.epoch,
            "phase": self.phase,
            "world_cursor": self.world_cursor,
            "goal_cursor": self.goal_cursor,
            "global_step": self.global_step,
            "total_epochs": self.total_epochs,
            "best_holdout_score": self.best_holdout_score,
            "child_checkpoint_digest": _world_action_persistent_digest(
                self.model, self.world_learner
            ),
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
        world_error, goal_success, persistent_before, read_only = self._holdout_metrics()
        if not read_only:
            raise RuntimeError("world action eval-only mutated persistent state")
        return {
            "format": WORLD_ACTION_TRAINING_FORMAT,
            "version": WORLD_ACTION_TRAINING_VERSION,
            "status": "evaluated",
            "model_tier": self.model_tier,
            "corpus_digest": self.corpus_digest,
            "corpus_sample_counts": {
                "world": self.world_corpus.sample_counts,
                "goal": self.goal_corpus.sample_counts,
            },
            "checkpoint_digest": persistent_before,
            "world_holdout_error": world_error,
            "goal_holdout_success": goal_success,
            "checkpoint_read_only": True,
            "code_revision": self.code_revision,
        }

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        world_corpus: WorldTransitionCorpus,
        goal_corpus: GoalActionCorpus,
        *,
        output_dir: str | Path | None = None,
        epochs: int | None = None,
        code_revision: str | None = None,
    ) -> WorldActionTrainingRun:
        checkpoint_path = Path(path)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise ValueError("world action training checkpoint must contain a mapping")
        if payload.get("format") != WORLD_ACTION_TRAINING_FORMAT:
            raise ValueError("unsupported world action training checkpoint format")
        if int(payload.get("version", -1)) != WORLD_ACTION_TRAINING_VERSION:
            raise ValueError("unsupported world action training checkpoint version")
        expected = content_digest(
            {key: value for key, value in payload.items() if key != "checkpoint_digest"}
        )
        if str(payload.get("checkpoint_digest", "")) != expected:
            raise ValueError("world action training checkpoint digest mismatch")
        if str(payload.get("corpus_digest")) != _world_action_corpus_digest(
            world_corpus, goal_corpus
        ):
            raise ValueError("world action training corpus digest mismatch")
        model_payload = payload.get("model")
        learner_payload = payload.get("world_learner")
        if not isinstance(model_payload, Mapping) or not isinstance(learner_payload, Mapping):
            raise ValueError("world action training checkpoint is missing model or world learner")
        model = Taiji(
            TaijiConfig.from_dict(dict(model_payload["config"])),
            episode_id="world-action-resume",
        )
        model.restore(model_payload)
        run = cls(
            model,
            _world_learner_from_payload(learner_payload),
            world_corpus,
            goal_corpus,
            output_dir=output_dir or checkpoint_path.parent,
            model_tier=str(payload["model_tier"]),
            epochs=int(epochs if epochs is not None else payload["total_epochs"]),
            checkpoint_interval=int(payload["checkpoint_interval"]),
            world_learning_rate=float(payload["world_learning_rate"]),
            world_repeats=int(payload["world_repeats"]),
            parent_checkpoint_digest=str(payload["parent_checkpoint_digest"]),
            parent_world_error=float(payload["parent_world_error"]),
            parent_goal_success=float(payload["parent_goal_success"]),
            code_revision=code_revision or str(payload.get("code_revision", "working-tree")),
        )
        parent_model = payload.get("parent_model")
        parent_world = payload.get("parent_world_learner")
        if not isinstance(parent_model, Mapping) or not isinstance(parent_world, Mapping):
            raise ValueError("world action checkpoint is missing parent lineage")
        run.parent_model_payload = deepcopy(parent_model)
        run.parent_world_payload = deepcopy(parent_world)
        run.epoch = int(payload["epoch"])
        run.phase = str(payload.get("phase", "world"))
        run.world_cursor = int(payload.get("world_cursor", 0))
        run.goal_cursor = int(payload.get("goal_cursor", 0))
        run.global_step = int(payload.get("global_step", 0))
        run.history = [dict(item) for item in payload.get("history", ())]
        run.best_holdout_score = float(payload.get("best_holdout_score", 0.0))
        run.started_from_checkpoint = True
        return run


def _joint_sequence_bpb(model: Taiji, data: bytes) -> float:
    before = content_digest(model.checkpoint())
    score = model.score_bytes(data)
    after = content_digest(model.checkpoint())
    if before != after:
        raise RuntimeError("joint sequence holdout evaluation mutated persistent state")
    return float(score["mean_surprise"]) / math.log(2.0)


def _joint_memory_recall(
    model: Taiji,
    corpus: DelayedMemoryCorpus,
    *,
    use_memory: bool,
) -> float:
    actions = tuple(dict.fromkeys(item.action for item in corpus.train))
    if len(actions) < 2:
        raise ValueError("joint memory corpus needs at least two action classes")
    correct = 0
    for query in corpus.holdout:
        model.reset_dynamics(episode_id=f"m1-f4-query-{query.query_id}")
        model.observe(
            model.config.boundary_symbol,
            learn=False,
            learn_motor=False,
            use_memory=use_memory,
        )
        model.observe(
            query.cue,
            learn=False,
            learn_motor=False,
            use_memory=use_memory,
        )
        probabilities = model.snapshot().motor_probabilities
        prediction = max(actions, key=lambda action: float(probabilities[action].item()))
        correct += int(prediction == query.expected_action)
    return correct / len(corpus.holdout)


def _joint_train_memory_episode(model: Taiji, episode: MemoryEpisode) -> None:
    model.reset_dynamics(episode_id=f"m1-f4-train-{episode.memory_id}")
    model.observe(model.config.boundary_symbol, learn=False, learn_motor=False, use_memory=False)
    model.observe(episode.cue, learn=False, learn_motor=False, use_memory=False)
    model.act((episode.action,), sample=False)
    model.settle_action(1.0, learn=False, learn_memory=True)
    model.observe(episode.outcome, learn=False, learn_motor=False, use_memory=False)


class JointTrainingRun:
    """A resumable short course combining F1, F2 and F3 on one lineage."""

    def __init__(
        self,
        model: Taiji,
        world_learner: WorldDynamicsLearner,
        dataset: FoundationTrainingDataset,
        memory_corpus: DelayedMemoryCorpus,
        world_corpus: WorldTransitionCorpus,
        goal_corpus: GoalActionCorpus,
        *,
        output_dir: str | Path,
        model_tier: str = "joint",
        epochs: int = 1,
        chunk_bytes: int = 1_024,
        checkpoint_interval: int = 1,
        world_learning_rate: float = 0.02,
        world_repeats: int = 8,
        parent_checkpoint_digest: str | None = None,
        parent_metrics: Mapping[str, float] | None = None,
        code_revision: str | None = None,
    ) -> None:
        if not isinstance(dataset, FoundationTrainingDataset):
            raise TypeError("joint training requires FoundationTrainingDataset")
        if not isinstance(memory_corpus, DelayedMemoryCorpus):
            raise TypeError("joint training requires DelayedMemoryCorpus")
        if not isinstance(world_corpus, WorldTransitionCorpus):
            raise TypeError("joint training requires WorldTransitionCorpus")
        if not isinstance(goal_corpus, GoalActionCorpus):
            raise TypeError("joint training requires GoalActionCorpus")
        if not isinstance(world_learner, WorldDynamicsLearner):
            raise TypeError("joint training requires WorldDynamicsLearner")
        if int(epochs) <= 0 or int(chunk_bytes) <= 0 or int(checkpoint_interval) <= 0:
            raise ValueError("joint training epochs, chunk_bytes, and checkpoint_interval must be positive")
        if float(world_learning_rate) <= 0.0 or int(world_repeats) <= 0:
            raise ValueError("joint world learning settings must be positive")
        self.model = model
        self.world_learner = world_learner
        self.dataset = dataset
        self.memory_corpus = memory_corpus
        self.world_corpus = world_corpus
        self.goal_corpus = goal_corpus
        self.output_dir = Path(output_dir)
        self.model_tier = str(model_tier)
        self.total_epochs = int(epochs)
        self.chunk_bytes = int(chunk_bytes)
        self.checkpoint_interval = int(checkpoint_interval)
        self.world_learning_rate = float(world_learning_rate)
        self.world_repeats = int(world_repeats)
        self.code_revision = str(code_revision or _code_revision())
        self.parent_model_payload = deepcopy(model.checkpoint())
        self.parent_world_payload = deepcopy(_world_learner_payload(world_learner))
        self.parent_checkpoint_digest = parent_checkpoint_digest or content_digest(
            {"model": self.parent_model_payload, "world": self.parent_world_payload}
        )
        measured_parent = self._measure_metrics()
        self.parent_metrics = {
            key: float(value) for key, value in (parent_metrics or measured_parent).items()
        }
        self.epoch = 0
        self.phase = "sequence"
        self.sequence_cursor = 0
        self.memory_cursor = 0
        self.world_cursor = 0
        self.goal_cursor = 0
        self.global_step = 0
        self.history: list[dict[str, Any]] = []
        self.best_holdout_score = 0.0
        self.started_from_checkpoint = False
        self.continuation_source_checkpoint_digest: str | None = None

    @property
    def corpus_digest(self) -> str:
        return content_digest(
            {
                "format": JOINT_TRAINING_FORMAT,
                "version": JOINT_TRAINING_VERSION,
                "dataset_digest": self.dataset.digest,
                "memory_digest": _memory_corpus_digest(self.memory_corpus),
                "world_action_digest": _world_action_corpus_digest(
                    self.world_corpus, self.goal_corpus
                ),
            }
        )

    @property
    def parent_checkpoint_path(self) -> Path:
        return self.output_dir / "parent.pt"

    @property
    def last_checkpoint_path(self) -> Path:
        return self.output_dir / "last.pt"

    @property
    def best_checkpoint_path(self) -> Path:
        return self.output_dir / "best-holdout.pt"

    def _measure_metrics(self) -> dict[str, float]:
        persistent_before = _world_action_persistent_digest(self.model, self.world_learner)
        metrics = {
            "sequence_holdout_bpb": _joint_sequence_bpb(self.model, self.dataset.holdout),
            "sequence_retention_bpb": _joint_sequence_bpb(self.model, self.dataset.retention),
            "memory_holdout_recall": _joint_memory_recall(
                self.model, self.memory_corpus, use_memory=True
            ),
            "memory_retention_recall": _joint_memory_recall(
                self.model, self.memory_corpus, use_memory=True
            ),
            "world_holdout_error": _world_action_error(
                self.world_learner, self.world_corpus.holdout
            ),
            "world_retention_error": _world_action_error(
                self.world_learner, self.world_corpus.retention
            ),
            "goal_holdout_success": _goal_action_accuracy(
                self.model, self.goal_corpus.holdout
            ),
            "goal_retention_success": _goal_action_accuracy(
                self.model, self.goal_corpus.retention
            ),
        }
        persistent_after = _world_action_persistent_digest(self.model, self.world_learner)
        if persistent_before != persistent_after:
            raise RuntimeError("joint holdout evaluation mutated persistent state")
        return metrics

    def _joint_holdout_score(self, metrics: Mapping[str, float]) -> float:
        return (
            self.parent_metrics["sequence_holdout_bpb"] - metrics["sequence_holdout_bpb"]
            + metrics["memory_holdout_recall"] - self.parent_metrics["memory_holdout_recall"]
            + self.parent_metrics["world_holdout_error"] - metrics["world_holdout_error"]
            + metrics["goal_holdout_success"] - self.parent_metrics["goal_holdout_success"]
        )

    def _checkpoint_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": JOINT_TRAINING_FORMAT,
            "version": JOINT_TRAINING_VERSION,
            "model_tier": self.model_tier,
            "model": self.model.checkpoint(),
            "world_learner": _world_learner_payload(self.world_learner),
            "dataset_digest": self.dataset.digest,
            "memory_digest": _memory_corpus_digest(self.memory_corpus),
            "world_action_digest": _world_action_corpus_digest(
                self.world_corpus, self.goal_corpus
            ),
            "corpus_digest": self.corpus_digest,
            "parent_checkpoint_digest": self.parent_checkpoint_digest,
            "parent_model": self.parent_model_payload,
            "parent_world_learner": self.parent_world_payload,
            "parent_metrics": dict(self.parent_metrics),
            "epoch": self.epoch,
            "phase": self.phase,
            "sequence_cursor": self.sequence_cursor,
            "memory_cursor": self.memory_cursor,
            "world_cursor": self.world_cursor,
            "goal_cursor": self.goal_cursor,
            "global_step": self.global_step,
            "total_epochs": self.total_epochs,
            "chunk_bytes": self.chunk_bytes,
            "checkpoint_interval": self.checkpoint_interval,
            "world_learning_rate": self.world_learning_rate,
            "world_repeats": self.world_repeats,
            "history": list(self.history),
            "best_holdout_score": self.best_holdout_score,
            "code_revision": self.code_revision,
            "continuation_source_checkpoint_digest": self.continuation_source_checkpoint_digest,
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
        train_kind: str | None = None,
        train_success: bool | None = None,
    ) -> None:
        metrics = self._measure_metrics()
        score = self._joint_holdout_score(metrics)
        record: dict[str, Any] = {
            "epoch": self.epoch,
            "phase": self.phase,
            "sequence_cursor": self.sequence_cursor,
            "memory_cursor": self.memory_cursor,
            "world_cursor": self.world_cursor,
            "goal_cursor": self.goal_cursor,
            "global_step": self.global_step,
            **metrics,
            "joint_holdout_gain": score,
        }
        if train_kind is not None:
            record["train_kind"] = train_kind
        if train_success is not None:
            record["train_success"] = bool(train_success)
        self.history.append(record)
        if score > self.best_holdout_score:
            self.best_holdout_score = score
            self.save(self.best_checkpoint_path)
        self.save(self.last_checkpoint_path)

    def run(self) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not self.started_from_checkpoint and not self.parent_checkpoint_path.exists():
            self.save(self.parent_checkpoint_path)
            self.save(self.best_checkpoint_path)
        while self.epoch < self.total_epochs:
            self.phase = "sequence"
            while self.sequence_cursor < len(self.dataset.train):
                end = min(self.sequence_cursor + self.chunk_bytes, len(self.dataset.train))
                self.model.learn_bytes(self.dataset.train[self.sequence_cursor:end], epochs=1)
                self.sequence_cursor = end
                self.global_step += 1
                if (
                    self.global_step % self.checkpoint_interval == 0
                    or self.sequence_cursor == len(self.dataset.train)
                ):
                    self._save_progress(train_kind="sequence")
                else:
                    self.save(self.last_checkpoint_path)

            self.phase = "memory"
            while self.memory_cursor < len(self.memory_corpus.train):
                _joint_train_memory_episode(
                    self.model, self.memory_corpus.train[self.memory_cursor]
                )
                self.memory_cursor += 1
                self.global_step += 1
                if (
                    self.global_step % self.checkpoint_interval == 0
                    or self.memory_cursor == len(self.memory_corpus.train)
                ):
                    self._save_progress(train_kind="memory")
                else:
                    self.save(self.last_checkpoint_path)

            self.phase = "world"
            while self.world_cursor < len(self.world_corpus.train):
                case = self.world_corpus.train[self.world_cursor]
                self.world_learner.online_update(
                    WorldTransition(
                        before=case.initial,
                        action=case.action,
                        after=case.expected_state,
                        outcome=case.expected_outcome,
                    ),
                    learning_rate=self.world_learning_rate,
                    repeats=self.world_repeats,
                    register_parameters=True,
                )
                self.world_cursor += 1
                self.global_step += 1
                if (
                    self.global_step % self.checkpoint_interval == 0
                    or self.world_cursor == len(self.world_corpus.train)
                ):
                    self._save_progress(train_kind="world")
                else:
                    self.save(self.last_checkpoint_path)

            self.phase = "goal"
            while self.goal_cursor < len(self.goal_corpus.train):
                success = _train_goal_episode(
                    self.model, self.goal_corpus.train[self.goal_cursor], learn=True
                )
                self.goal_cursor += 1
                self.global_step += 1
                if (
                    self.global_step % self.checkpoint_interval == 0
                    or self.goal_cursor == len(self.goal_corpus.train)
                ):
                    self._save_progress(train_kind="goal", train_success=success)
                else:
                    self.save(self.last_checkpoint_path)

            self.epoch += 1
            self.phase = "sequence"
            self.sequence_cursor = 0
            self.memory_cursor = 0
            self.world_cursor = 0
            self.goal_cursor = 0
            self._save_progress()

        final_metrics = self._measure_metrics()
        lesion = Taiji.from_checkpoint(self.parent_model_payload)
        _cold_start_action_organ(lesion)
        for episode in self.goal_corpus.train:
            _train_goal_episode(lesion, episode, learn=False)
        credit_lesion = _goal_action_accuracy(lesion, self.goal_corpus.holdout)
        return {
            "format": JOINT_TRAINING_FORMAT,
            "version": JOINT_TRAINING_VERSION,
            "status": "completed",
            "model_tier": self.model_tier,
            "corpus_digest": self.corpus_digest,
            "dataset_digest": self.dataset.digest,
            "memory_digest": _memory_corpus_digest(self.memory_corpus),
            "world_action_digest": _world_action_corpus_digest(
                self.world_corpus, self.goal_corpus
            ),
            "parent_checkpoint_digest": self.parent_checkpoint_digest,
            "parent_metrics": dict(self.parent_metrics),
            "final_metrics": final_metrics,
            "final_credit_lesion_success": credit_lesion,
            "joint_holdout_gain": self._joint_holdout_score(final_metrics),
            "world_online_updates": self.world_learner.online_updates,
            "world_transition_rejections": self.world_learner.transition_rejections,
            "holdout_updates": 0,
            "epoch": self.epoch,
            "phase": self.phase,
            "sequence_cursor": self.sequence_cursor,
            "memory_cursor": self.memory_cursor,
            "world_cursor": self.world_cursor,
            "goal_cursor": self.goal_cursor,
            "global_step": self.global_step,
            "total_epochs": self.total_epochs,
            "best_holdout_score": self.best_holdout_score,
            "child_checkpoint_digest": _world_action_persistent_digest(
                self.model, self.world_learner
            ),
            "history": list(self.history),
            "checkpoint_paths": {
                "parent": str(self.parent_checkpoint_path),
                "last": str(self.last_checkpoint_path),
                "best_holdout": str(self.best_checkpoint_path),
            },
            "code_revision": self.code_revision,
            "started_from_checkpoint": self.started_from_checkpoint,
            "continuation_source_checkpoint_digest": self.continuation_source_checkpoint_digest,
        }

    def evaluate_only(self) -> dict[str, Any]:
        before = _world_action_persistent_digest(self.model, self.world_learner)
        metrics = self._measure_metrics()
        after = _world_action_persistent_digest(self.model, self.world_learner)
        if before != after:
            raise RuntimeError("joint eval-only mutated persistent state")
        return {
            "format": JOINT_TRAINING_FORMAT,
            "version": JOINT_TRAINING_VERSION,
            "status": "evaluated",
            "model_tier": self.model_tier,
            "corpus_digest": self.corpus_digest,
            "checkpoint_digest": before,
            "metrics": metrics,
            "checkpoint_read_only": True,
            "code_revision": self.code_revision,
            "continuation_source_checkpoint_digest": self.continuation_source_checkpoint_digest,
        }

    @classmethod
    def from_continuation_checkpoint(
        cls,
        path: str | Path,
        dataset: FoundationTrainingDataset,
        memory_corpus: DelayedMemoryCorpus,
        world_corpus: WorldTransitionCorpus,
        goal_corpus: GoalActionCorpus,
        *,
        output_dir: str | Path,
        epochs: int | None = None,
        chunk_bytes: int | None = None,
        checkpoint_interval: int | None = None,
        world_learning_rate: float | None = None,
        world_repeats: int | None = None,
        code_revision: str | None = None,
    ) -> JointTrainingRun:
        """Start a new course from an existing child with an explicit data extension.

        Ordinary ``from_checkpoint`` remains strict about corpus identity so a
        resume cannot silently skip or replay data.  F5 needs a different,
        auditable operation: an already-trained F4 child becomes the parent of
        a new course whose expanded corpora and parent metrics are measured
        afresh.  The original checkpoint digest is retained as lineage, while
        all new cursors start at zero because this is a new course, not a
        resume of the old pilot.
        """

        checkpoint_path = Path(path)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise ValueError("joint continuation checkpoint must contain a mapping")
        if payload.get("format") != JOINT_TRAINING_FORMAT:
            raise ValueError("unsupported joint continuation checkpoint format")
        if int(payload.get("version", -1)) != JOINT_TRAINING_VERSION:
            raise ValueError("unsupported joint continuation checkpoint version")
        expected = content_digest(
            {key: value for key, value in payload.items() if key != "checkpoint_digest"}
        )
        if str(payload.get("checkpoint_digest", "")) != expected:
            raise ValueError("joint continuation checkpoint digest mismatch")
        model_payload = payload.get("model")
        learner_payload = payload.get("world_learner")
        if not isinstance(model_payload, Mapping) or not isinstance(learner_payload, Mapping):
            raise ValueError("joint continuation checkpoint is missing model or world learner")

        model = Taiji(
            TaijiConfig.from_dict(dict(model_payload["config"])),
            episode_id="joint-continuation",
        )
        model.restore(model_payload)
        run = cls(
            model,
            _world_learner_from_payload(learner_payload),
            dataset,
            memory_corpus,
            world_corpus,
            goal_corpus,
            output_dir=output_dir,
            model_tier=str(payload["model_tier"]),
            epochs=int(epochs if epochs is not None else payload["total_epochs"]),
            chunk_bytes=int(chunk_bytes if chunk_bytes is not None else payload["chunk_bytes"]),
            checkpoint_interval=int(
                checkpoint_interval
                if checkpoint_interval is not None
                else payload["checkpoint_interval"]
            ),
            world_learning_rate=float(
                world_learning_rate
                if world_learning_rate is not None
                else payload["world_learning_rate"]
            ),
            world_repeats=int(
                world_repeats if world_repeats is not None else payload["world_repeats"]
            ),
            code_revision=code_revision or _code_revision(),
        )
        run.continuation_source_checkpoint_digest = str(payload["checkpoint_digest"])
        run.started_from_checkpoint = True
        run.save(run.parent_checkpoint_path)
        return run

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        dataset: FoundationTrainingDataset,
        memory_corpus: DelayedMemoryCorpus,
        world_corpus: WorldTransitionCorpus,
        goal_corpus: GoalActionCorpus,
        *,
        output_dir: str | Path | None = None,
        epochs: int | None = None,
        code_revision: str | None = None,
    ) -> JointTrainingRun:
        checkpoint_path = Path(path)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise ValueError("joint training checkpoint must contain a mapping")
        if payload.get("format") != JOINT_TRAINING_FORMAT:
            raise ValueError("unsupported joint training checkpoint format")
        if int(payload.get("version", -1)) != JOINT_TRAINING_VERSION:
            raise ValueError("unsupported joint training checkpoint version")
        expected = content_digest(
            {key: value for key, value in payload.items() if key != "checkpoint_digest"}
        )
        if str(payload.get("checkpoint_digest", "")) != expected:
            raise ValueError("joint training checkpoint digest mismatch")
        expected_corpus = content_digest(
            {
                "format": JOINT_TRAINING_FORMAT,
                "version": JOINT_TRAINING_VERSION,
                "dataset_digest": dataset.digest,
                "memory_digest": _memory_corpus_digest(memory_corpus),
                "world_action_digest": _world_action_corpus_digest(world_corpus, goal_corpus),
            }
        )
        if str(payload.get("corpus_digest")) != expected_corpus:
            raise ValueError("joint training corpus digest mismatch")
        model_payload = payload.get("model")
        learner_payload = payload.get("world_learner")
        if not isinstance(model_payload, Mapping) or not isinstance(learner_payload, Mapping):
            raise ValueError("joint training checkpoint is missing model or world learner")
        model = Taiji(
            TaijiConfig.from_dict(dict(model_payload["config"])),
            episode_id="joint-resume",
        )
        model.restore(model_payload)
        run = cls(
            model,
            _world_learner_from_payload(learner_payload),
            dataset,
            memory_corpus,
            world_corpus,
            goal_corpus,
            output_dir=output_dir or checkpoint_path.parent,
            model_tier=str(payload["model_tier"]),
            epochs=int(epochs if epochs is not None else payload["total_epochs"]),
            chunk_bytes=int(payload["chunk_bytes"]),
            checkpoint_interval=int(payload["checkpoint_interval"]),
            world_learning_rate=float(payload["world_learning_rate"]),
            world_repeats=int(payload["world_repeats"]),
            parent_checkpoint_digest=str(payload["parent_checkpoint_digest"]),
            parent_metrics=dict(payload["parent_metrics"]),
            code_revision=code_revision or str(payload.get("code_revision", "working-tree")),
        )
        parent_model = payload.get("parent_model")
        parent_world = payload.get("parent_world_learner")
        if not isinstance(parent_model, Mapping) or not isinstance(parent_world, Mapping):
            raise ValueError("joint training checkpoint is missing parent lineage")
        run.parent_model_payload = deepcopy(parent_model)
        run.parent_world_payload = deepcopy(parent_world)
        run.epoch = int(payload["epoch"])
        run.phase = str(payload.get("phase", "sequence"))
        run.sequence_cursor = int(payload.get("sequence_cursor", 0))
        run.memory_cursor = int(payload.get("memory_cursor", 0))
        run.world_cursor = int(payload.get("world_cursor", 0))
        run.goal_cursor = int(payload.get("goal_cursor", 0))
        run.global_step = int(payload.get("global_step", 0))
        run.history = [dict(item) for item in payload.get("history", ())]
        run.best_holdout_score = float(payload.get("best_holdout_score", 0.0))
        run.started_from_checkpoint = True
        source_digest = payload.get("continuation_source_checkpoint_digest")
        run.continuation_source_checkpoint_digest = (
            str(source_digest) if source_digest is not None else None
        )
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
    "WORLD_ACTION_TRAINING_FORMAT",
    "WORLD_ACTION_TRAINING_VERSION",
    "WorldActionTrainingRun",
    "JOINT_TRAINING_FORMAT",
    "JOINT_TRAINING_VERSION",
    "JointTrainingRun",
]
