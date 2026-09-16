"""Native language-alignment training for structured answer episodes.

This module is the first executable R2 seam.  It keeps Taiji as the native
language owner while making the training unit explicit:

    structured prompt/context -> native predictive state -> response target

The prefix is consumed without learning.  Only the response target is used
for predictive updates, so an episode has an auditable answer-credit boundary
instead of being an undifferentiated raw-byte stream.  This is deliberately a
small, content-addressed pilot seam; it does not change the default Seed
entrypoint, attach an external provider, or claim that a trained checkpoint
has passed the later Mini/L2/L3 evaluation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch

from .internalization import content_digest
from .model import Taiji
from .organs import BytePredictiveReadout
from .response_plan_target import (
    FACTOR_RESPONSE_PLAN_TARGET_PHASE_STRIDE,
    FACTOR_RESPONSE_PLAN_TARGET_SLOTS,
    FactorizedResponsePlanTargetEncoder,
    ResponsePlanTargetEncoder,
)

LANGUAGE_ALIGNMENT_FORMAT = "taiji-native-language-alignment-v2"
LANGUAGE_ALIGNMENT_VERSION = 2
LANGUAGE_ALIGNMENT_SERIALIZATION = "r2-conditional-response-v2"
LANGUAGE_ALIGNMENT_SEQUENCE_EVALUATION = "r2-sequence-criteria-v1"
LANGUAGE_ALIGNMENT_GENERALIZATION_EVALUATION = "r2-generalization-profile-v1"
LANGUAGE_ALIGNMENT_CREDIT_EVALUATION = "r2-conditional-credit-profile-v1"
LANGUAGE_ALIGNMENT_SPLITS = frozenset({"train", "dev", "final", "retention"})
LANGUAGE_ALIGNMENT_MARKERS = (
    "<|system|>",
    "<|policy|>",
    "<|context|>",
    "<|history_user|>",
    "<|history_assistant|>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
    "<|none|>",
)
LANGUAGE_RESPONSE_PLAN_TARGET_GEOMETRIES = frozenset(
    {
        "signed_hash_span",
        "h3_6_whitened_native_compositional",
        "h3_7_factorized_response_chunks",
    }
)
LANGUAGE_RESPONSE_PLAN_TARGET_H36 = "h3_6_whitened_native_compositional"
LANGUAGE_RESPONSE_PLAN_TARGET_H37 = "h3_7_factorized_response_chunks"


def _text(value: Any, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    if not allow_empty and not normalized.strip():
        raise ValueError(f"{name} cannot be empty")
    if any(marker in normalized for marker in LANGUAGE_ALIGNMENT_MARKERS):
        raise ValueError(f"{name} contains a reserved language-alignment marker")
    return normalized


def _identifier(value: Any, name: str) -> str:
    normalized = _text(value, name)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}", normalized):
        raise ValueError(f"{name} has an invalid stable identifier")
    return normalized


def _history(value: Any) -> tuple[tuple[str, str], ...]:
    if value in (None, (), []):
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("history must be a sequence of user/assistant pairs")
    result: list[tuple[str, str]] = []
    for index, item in enumerate(value):
        if isinstance(item, Mapping):
            user = item.get("user")
            assistant = item.get("assistant")
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) == 2:
            user, assistant = item
        else:
            raise TypeError(f"history item {index} must contain user and assistant text")
        result.append(
            (
                _text(user, f"history[{index}].user"),
                _text(assistant, f"history[{index}].assistant"),
            )
        )
    return tuple(result)


def _terms(value: Any, name: str) -> tuple[str, ...]:
    if value in (None, (), []):
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of text terms")
    terms = tuple(_text(item, f"{name}[{index}]") for index, item in enumerate(value))
    if len(set(terms)) != len(terms):
        raise ValueError(f"{name} must not contain duplicate terms")
    return terms


@dataclass(frozen=True)
class LanguageEpisode:
    """One explicit prompt/context/response training unit."""

    episode_id: str
    family_id: str
    task_family: str
    split: str
    user_input: str
    response: str
    context: str = ""
    system: str = "回答问题；信息不足时明确说明不知道。"
    history: tuple[tuple[str, str], ...] = ()
    unknown_policy: str = "answer"
    required_terms: tuple[str, ...] = ()
    forbidden_terms: tuple[str, ...] = ()
    unknown_markers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "episode_id", _identifier(self.episode_id, "episode_id"))
        object.__setattr__(self, "family_id", _identifier(self.family_id, "family_id"))
        object.__setattr__(self, "task_family", _identifier(self.task_family, "task_family"))
        split = _text(self.split, "split")
        if split not in LANGUAGE_ALIGNMENT_SPLITS:
            raise ValueError(f"unsupported language episode split: {split}")
        object.__setattr__(self, "split", split)
        object.__setattr__(self, "user_input", _text(self.user_input, "user_input"))
        object.__setattr__(self, "response", _text(self.response, "response"))
        object.__setattr__(self, "context", _text(self.context, "context", allow_empty=True))
        object.__setattr__(self, "system", _text(self.system, "system"))
        object.__setattr__(self, "history", _history(self.history))
        policy = _text(self.unknown_policy, "unknown_policy")
        if policy not in {"answer", "say_unknown", "clarify", "refuse"}:
            raise ValueError("unknown_policy must be answer, say_unknown, clarify, or refuse")
        object.__setattr__(self, "unknown_policy", policy)
        object.__setattr__(self, "required_terms", _terms(self.required_terms, "required_terms"))
        object.__setattr__(self, "forbidden_terms", _terms(self.forbidden_terms, "forbidden_terms"))
        object.__setattr__(self, "unknown_markers", _terms(self.unknown_markers, "unknown_markers"))

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> LanguageEpisode:
        if not isinstance(payload, Mapping):
            raise TypeError("language episode payload must be a mapping")
        response = payload.get("response", payload.get("target"))
        if response is None:
            raise ValueError("language episode requires response or target")
        return cls(
            episode_id=payload["episode_id"],
            family_id=payload.get("family_id", payload["episode_id"]),
            task_family=payload.get("task_family", "general"),
            split=payload.get("split", "train"),
            user_input=payload.get("user_input", payload.get("user", "")),
            response=response,
            context=payload.get("context", ""),
            system=payload.get("system", "回答问题；信息不足时明确说明不知道。"),
            history=payload.get("history", ()),
            unknown_policy=payload.get("unknown_policy", "answer"),
            required_terms=payload.get("required_terms", ()),
            forbidden_terms=payload.get("forbidden_terms", ()),
            unknown_markers=payload.get("unknown_markers", ()),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "family_id": self.family_id,
            "task_family": self.task_family,
            "split": self.split,
            "system": self.system,
            "context": self.context,
            "history": [{"user": user, "assistant": assistant} for user, assistant in self.history],
            "user_input": self.user_input,
            "response": self.response,
            "unknown_policy": self.unknown_policy,
            "required_terms": list(self.required_terms),
            "forbidden_terms": list(self.forbidden_terms),
            "unknown_markers": list(self.unknown_markers),
        }

    @property
    def prefix_text(self) -> str:
        parts = [
            "<|system|>",
            self.system,
            "<|policy|>",
            self.unknown_policy,
            "<|context|>",
            self.context if self.context else "<|none|>",
        ]
        for user, assistant in self.history:
            parts.extend(("<|history_user|>", user, "<|history_assistant|>", assistant))
        parts.extend(("<|user|>", self.user_input, "<|assistant|>"))
        return "\n".join(parts) + "\n"

    @property
    def target_text(self) -> str:
        return self.response + "\n<|end|>\n"

    @property
    def prompt_bytes(self) -> bytes:
        return self.prefix_text.encode("utf-8")

    @property
    def target_bytes(self) -> bytes:
        return self.target_text.encode("utf-8")

    @property
    def full_bytes(self) -> bytes:
        return self.prompt_bytes + self.target_bytes

    @property
    def digest(self) -> str:
        return content_digest(
            {
                "format": LANGUAGE_ALIGNMENT_FORMAT,
                "version": LANGUAGE_ALIGNMENT_VERSION,
                "serialization": LANGUAGE_ALIGNMENT_SERIALIZATION,
                "episode": self.to_payload(),
            }
        )


@dataclass(frozen=True)
class LanguageEpisodeCorpus:
    """Content-addressed, family-disjoint corpus for R2 language alignment."""

    episodes: tuple[LanguageEpisode, ...]
    source_files: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        episodes = tuple(self.episodes)
        if not episodes:
            raise ValueError("language alignment corpus cannot be empty")
        if any(not isinstance(item, LanguageEpisode) for item in episodes):
            raise TypeError("language alignment corpus contains an invalid episode")
        ids = [item.episode_id for item in episodes]
        if len(set(ids)) != len(ids):
            raise ValueError("language episode ids must be unique")
        family_splits: dict[str, set[str]] = defaultdict(set)
        for item in episodes:
            family_splits[item.family_id].add(item.split)
        leaking = {
            family: sorted(splits) for family, splits in family_splits.items() if len(splits) > 1
        }
        if leaking:
            raise ValueError(f"language episode family crosses splits: {leaking}")
        if not any(item.split == "train" for item in episodes):
            raise ValueError("language alignment corpus needs a train split")
        if not any(item.split == "dev" for item in episodes):
            raise ValueError("language alignment corpus needs a dev split")
        if not any(item.split == "final" for item in episodes):
            raise ValueError("language alignment corpus needs a final split")
        normalized_sources = tuple(
            (str(path).strip(), str(digest).strip()) for path, digest in self.source_files
        )
        if any(not path or not digest for path, digest in normalized_sources):
            raise ValueError("source_files must contain path and digest")
        object.__setattr__(self, "episodes", episodes)
        object.__setattr__(self, "source_files", normalized_sources)

    @classmethod
    def from_jsonl(cls, paths: Iterable[str | Path]) -> LanguageEpisodeCorpus:
        normalized_paths = tuple(Path(path) for path in paths)
        if not normalized_paths:
            raise ValueError("language alignment corpus needs at least one JSONL path")
        episodes: list[LanguageEpisode] = []
        sources: list[tuple[str, str]] = []
        for path in normalized_paths:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            sources.append((str(path), digest))
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"invalid episode JSON at {path}:{line_number}") from exc
                    try:
                        episodes.append(LanguageEpisode.from_payload(payload))
                    except (TypeError, ValueError, KeyError) as exc:
                        raise ValueError(
                            f"invalid language episode at {path}:{line_number}: {exc}"
                        ) from exc
        return cls(tuple(episodes), tuple(sources))

    @property
    def digest(self) -> str:
        return content_digest(
            {
                "format": LANGUAGE_ALIGNMENT_FORMAT,
                "version": LANGUAGE_ALIGNMENT_VERSION,
                "serialization": LANGUAGE_ALIGNMENT_SERIALIZATION,
                "source_files": [list(item) for item in self.source_files],
                "episodes": [item.to_payload() for item in self.episodes],
            }
        )

    @property
    def sample_counts(self) -> dict[str, int]:
        return {
            split: sum(item.split == split for item in self.episodes)
            for split in ("train", "dev", "final", "retention")
        }

    def for_split(self, split: str) -> tuple[LanguageEpisode, ...]:
        if split not in LANGUAGE_ALIGNMENT_SPLITS:
            raise ValueError(f"unsupported language episode split: {split}")
        return tuple(item for item in self.episodes if item.split == split)

    def manifest(self) -> dict[str, Any]:
        return {
            "format": LANGUAGE_ALIGNMENT_FORMAT,
            "version": LANGUAGE_ALIGNMENT_VERSION,
            "serialization": LANGUAGE_ALIGNMENT_SERIALIZATION,
            "digest": self.digest,
            "source_files": [list(item) for item in self.source_files],
            "sample_counts": self.sample_counts,
            "episode_ids": [item.episode_id for item in self.episodes],
        }


@dataclass(frozen=True)
class LanguageAlignmentConfig:
    """Frozen controls for the first native response-credit pilot."""

    response_repeats: int = 1
    max_generation_bytes: int = 512
    constrained_decode: bool = True
    use_memory: bool = False
    learn_fabric: bool = True
    learn_predictive_context: bool = True
    learn_predictive_readout: bool = True
    response_start_readout: bool = False
    response_phase_readout: bool = False
    response_plan_readout: bool = False
    response_plan_width: int = 32
    response_plan_variant: str = "single"
    response_plan_slots: int = FACTOR_RESPONSE_PLAN_TARGET_SLOTS
    response_plan_phase_stride: int = FACTOR_RESPONSE_PLAN_TARGET_PHASE_STRIDE
    response_plan_bridge_learning_rate_scale: float = 1.0
    response_plan_slot_credit_scale: float = 0.25
    response_plan_target_geometry: str = "signed_hash_span"
    developmental_mode: str = "static"
    developmental_replay_learning_rate_scale: float = 0.25
    developmental_consolidation_rate: float = 1.0
    developmental_clear_fast: bool = True
    developmental_clear_replay: bool = True

    def __post_init__(self) -> None:
        if int(self.response_repeats) <= 0:
            raise ValueError("response_repeats must be positive")
        if int(self.max_generation_bytes) <= 0:
            raise ValueError("max_generation_bytes must be positive")
        if self.developmental_mode not in {"static", "slow", "fast", "fast_slow"}:
            raise ValueError("developmental_mode must be static, slow, fast, or fast_slow")
        for name in (
            "developmental_replay_learning_rate_scale",
            "developmental_consolidation_rate",
            "response_plan_bridge_learning_rate_scale",
            "response_plan_slot_credit_scale",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in (
            "use_memory",
            "constrained_decode",
            "learn_fabric",
            "learn_predictive_context",
            "learn_predictive_readout",
            "response_start_readout",
            "response_phase_readout",
            "response_plan_readout",
            "developmental_clear_fast",
            "developmental_clear_replay",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        if int(self.response_plan_width) <= 0:
            raise ValueError("response_plan_width must be positive")
        if self.response_plan_variant not in {"single", "factorized_v1"}:
            raise ValueError("response_plan_variant must be single or factorized_v1")
        if int(self.response_plan_slots) <= 0:
            raise ValueError("response_plan_slots must be positive")
        if int(self.response_plan_phase_stride) <= 0:
            raise ValueError("response_plan_phase_stride must be positive")
        if self.response_plan_variant == "factorized_v1":
            if int(self.response_plan_slots) <= 1:
                raise ValueError("factorized response plans require multiple slots")
            if self.response_plan_width % int(self.response_plan_slots) != 0:
                raise ValueError("factorized response plan width must divide into slots")
        elif int(self.response_plan_slots) != FACTOR_RESPONSE_PLAN_TARGET_SLOTS:
            raise ValueError("single response plans cannot declare factorized slot count")
        if self.response_plan_target_geometry not in LANGUAGE_RESPONSE_PLAN_TARGET_GEOMETRIES:
            raise ValueError(
                "response_plan_target_geometry must be signed_hash_span or "
                "h3_6_whitened_native_compositional"
            )
        if (
            self.response_plan_target_geometry != "signed_hash_span"
            and not self.response_plan_readout
        ):
            raise ValueError("a non-legacy response-plan target requires response_plan_readout")
        if (
            self.response_plan_target_geometry == LANGUAGE_RESPONSE_PLAN_TARGET_H36
            and self.response_plan_variant != "single"
        ):
            raise ValueError("H3.6 target geometry requires the single response plan variant")
        if (
            self.response_plan_target_geometry == LANGUAGE_RESPONSE_PLAN_TARGET_H37
            and self.response_plan_variant != "factorized_v1"
        ):
            raise ValueError("H3.7 target geometry requires the factorized response plan variant")
        if (
            sum(
                bool(item)
                for item in (
                    self.response_start_readout,
                    self.response_phase_readout,
                    self.response_plan_readout,
                )
            )
            > 1
        ):
            raise ValueError("response candidate readouts are mutually exclusive")

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


def _code_revision() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[1],
        )
    except (OSError, subprocess.CalledProcessError):
        return "working-tree"
    return result.stdout.strip() or "working-tree"


def _atomic_save(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Load Seed persistence lazily so Taiji top-level imports stay acyclic."""

    from seed.persistence import atomic_save

    return atomic_save(payload, path)


class LanguageAlignmentTrainer:
    """Train and score explicit response episodes on Taiji's native readout."""

    CHECKPOINT_FORMAT = LANGUAGE_ALIGNMENT_FORMAT
    CHECKPOINT_VERSION = LANGUAGE_ALIGNMENT_VERSION

    def __init__(
        self,
        model: Taiji,
        corpus: LanguageEpisodeCorpus,
        *,
        config: LanguageAlignmentConfig | None = None,
        code_revision: str | None = None,
        response_plan_target_encoder: ResponsePlanTargetEncoder | None = None,
        response_plan_targets: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        if not isinstance(model, Taiji):
            raise TypeError("language alignment trainer requires a Taiji model")
        if not isinstance(corpus, LanguageEpisodeCorpus):
            raise TypeError("language alignment trainer requires a LanguageEpisodeCorpus")
        self.model = model
        self.corpus = corpus
        self.corpus_digest = corpus.digest
        self.config = config or LanguageAlignmentConfig()
        self.code_revision = str(code_revision or _code_revision())
        self.global_step = 0
        self.episode_count = 0
        self.history: list[dict[str, Any]] = []
        self.developmental_history: list[dict[str, Any]] = []
        self.response_plan_target_encoder = response_plan_target_encoder
        self.response_plan_targets: dict[str, torch.Tensor] = {
            str(key): value.detach().cpu().to(dtype=torch.float32).clone()
            for key, value in (response_plan_targets or {}).items()
        }
        self._configure_developmental_mode()
        self._configure_response_start_readout()
        self._configure_response_phase_readout()
        self._configure_response_plan_readout()
        self._configure_response_plan_target()

    def _configure_response_start_readout(self) -> None:
        if not self.config.response_start_readout:
            return
        if not self.model.response_start_readout_enabled:
            self.model.enable_response_start_readout(seed_from_protected=True)

    def _configure_response_phase_readout(self) -> None:
        if not self.config.response_phase_readout:
            return
        if not self.model.response_phase_readout_enabled:
            self.model.enable_response_phase_readout(seed_from_protected=True)

    def _configure_response_plan_readout(self) -> None:
        if not self.config.response_plan_readout:
            return
        if not self.model.response_plan_readout_enabled:
            self.model.enable_response_plan_readout(
                plan_width=self.config.response_plan_width,
                variant=self.config.response_plan_variant,
                plan_slots=self.config.response_plan_slots,
                phase_stride=self.config.response_plan_phase_stride,
                bridge_learning_rate_scale=self.config.response_plan_bridge_learning_rate_scale,
                slot_credit_scale=self.config.response_plan_slot_credit_scale,
            )
        readout = self.model.response_plan_readout
        if readout.variant != self.config.response_plan_variant:
            raise ValueError("response plan readout variant does not match trainer config")
        if readout.variant == "factorized_v1" and (
            readout.plan_slots != self.config.response_plan_slots
            or readout.phase_stride != self.config.response_plan_phase_stride
        ):
            raise ValueError("factorized response plan schedule does not match trainer config")

    def _configure_response_plan_target(self) -> None:
        geometry = self.config.response_plan_target_geometry
        if geometry == "signed_hash_span":
            if self.response_plan_target_encoder is not None or self.response_plan_targets:
                raise ValueError(
                    "legacy response-plan geometry cannot carry an H3.6 target encoder"
                )
            return
        if geometry not in {
            LANGUAGE_RESPONSE_PLAN_TARGET_H36,
            LANGUAGE_RESPONSE_PLAN_TARGET_H37,
        }:
            raise ValueError("unsupported response-plan target geometry")
        if self.response_plan_target_encoder is None:
            raise ValueError("non-legacy response-plan geometry requires a target encoder")
        if not self.model.response_plan_readout_enabled:
            raise ValueError("non-legacy response-plan geometry requires the plan readout")
        if self.response_plan_target_encoder.width != self.config.response_plan_width:
            raise ValueError("target encoder width does not match response-plan width")
        if geometry == LANGUAGE_RESPONSE_PLAN_TARGET_H37:
            if not isinstance(self.response_plan_target_encoder, FactorizedResponsePlanTargetEncoder):
                raise ValueError("H3.7 target geometry requires its factorized encoder")
            if (
                self.response_plan_target_encoder.slots != self.config.response_plan_slots
                or self.response_plan_target_encoder.phase_stride
                != self.config.response_plan_phase_stride
            ):
                raise ValueError("H3.7 target encoder schedule does not match trainer config")
        self.response_plan_target_encoder.assert_compatible(corpus_digest=self.corpus_digest)
        train_ids = tuple(item.episode_id for item in self.corpus.for_split("train"))
        if self.response_plan_targets:
            if set(self.response_plan_targets) != set(train_ids):
                raise ValueError("checkpointed response-plan targets do not match the train split")
            targets = self.response_plan_targets
        else:
            targets = {
                episode.episode_id: self.response_plan_target_encoder.encode_episode(
                    self.model, episode
                )
                for episode in self.corpus.for_split("train")
            }
        self.response_plan_targets = {}
        for episode_id in train_ids:
            target = targets[episode_id].detach().cpu().to(dtype=torch.float32).clone()
            if target.shape != (self.config.response_plan_width,):
                raise ValueError("response-plan target dimension mismatch")
            if geometry == LANGUAGE_RESPONSE_PLAN_TARGET_H37:
                slot_width = self.config.response_plan_width // self.config.response_plan_slots
                slot_norms = torch.linalg.vector_norm(
                    target.reshape(self.config.response_plan_slots, slot_width), dim=1
                )
                if not bool(torch.isfinite(slot_norms).all()) or bool(
                    (slot_norms - 1.0).abs().gt(1e-4).any()
                ):
                    raise ValueError("H3.7 response-plan target slots must be unit normalized")
            else:
                norm = float(torch.linalg.vector_norm(target))
                if not math.isfinite(norm) or abs(norm - 1.0) > 1e-4:
                    raise ValueError("H3.6 response-plan targets must be unit normalized")
            if not bool(torch.isfinite(target).all()):
                raise ValueError("response-plan target contains non-finite values")
            self.response_plan_targets[episode_id] = target

    @property
    def response_plan_target_digest(self) -> str | None:
        if self.response_plan_target_encoder is None:
            return None
        return self.response_plan_target_encoder.target_digest

    def _response_plan_target(self, episode: LanguageEpisode) -> torch.Tensor:
        """Return the selected training-only response-plan target."""

        if self.config.response_plan_target_geometry in {
            LANGUAGE_RESPONSE_PLAN_TARGET_H36,
            LANGUAGE_RESPONSE_PLAN_TARGET_H37,
        }:
            try:
                return self.response_plan_targets[episode.episode_id].clone()
            except KeyError as exc:
                raise ValueError("response-plan target is missing for the episode") from exc

        characters = list(episode.response)
        span_count = min(8, max(1, len(characters)))
        width = int(self.config.response_plan_width)
        target = torch.zeros(width, dtype=torch.float32)
        for index in range(span_count):
            start = (index * len(characters)) // span_count
            end = ((index + 1) * len(characters)) // span_count
            span = "".join(characters[start:end]).encode("utf-8")
            digest = hashlib.sha256(b"r2-h3-5a-response-plan-v1\x00" + span).digest()
            for offset in range(width):
                byte = digest[offset % len(digest)]
                target[offset] += 1.0 if byte & 1 else -1.0
        norm = torch.linalg.vector_norm(target)
        if float(norm) == 0.0:
            raise RuntimeError("response plan target unexpectedly has zero norm")
        return target / norm

    def _configure_developmental_mode(self) -> None:
        mode = self.config.developmental_mode
        if mode == "static":
            if self.model.developmental_f1_enabled:
                raise ValueError(
                    "static language alignment mode cannot own a developmental F1 bundle"
                )
            return
        if not self.model.developmental_f1_enabled:
            self.model.migrate_f1_to_developmental_synapses()
        self.model.set_developmental_f1_learning_mode(mode)

    def _developmental_status(self) -> dict[str, Any]:
        enabled = self.model.developmental_f1_enabled
        bundle = self.model.developmental_f1_bundle
        return {
            "mode": self.config.developmental_mode,
            "enabled": enabled,
            "learning_mode": (
                "static" if not enabled else self.model.developmental_f1_learning_mode
            ),
            "replay_count": 0 if not enabled else self.model.developmental_f1_replay_count,
            "bundle_digest": None if bundle is None else content_digest(bundle.to_payload()),
        }

    def _finish_developmental_epoch(self, epoch: int) -> dict[str, Any]:
        replay = None
        if self.config.developmental_mode == "fast_slow":
            replay = self.model.replay_developmental_f1(
                learning_rate_scale=self.config.developmental_replay_learning_rate_scale,
                consolidate=True,
                consolidation_rate=self.config.developmental_consolidation_rate,
                clear_fast=self.config.developmental_clear_fast,
                clear_replay=self.config.developmental_clear_replay,
            )
        record = {
            "epoch": int(epoch),
            "replay": replay,
            "status": self._developmental_status(),
        }
        self.developmental_history.append(record)
        return record

    def _observe(
        self,
        symbol: int,
        *,
        learn: bool,
        predictive_readout: BytePredictiveReadout | None = None,
    ):
        if not learn:
            return self.model.observe(
                int(symbol),
                learn=False,
                readout="predictive",
                use_memory=self.config.use_memory,
                use_identity=False,
                _predictive_readout=predictive_readout,
            )
        return self.model.observe(
            int(symbol),
            learn=True,
            learn_fabric=self.config.learn_fabric,
            learn_predictive_context=self.config.learn_predictive_context,
            learn_predictive_readout=self.config.learn_predictive_readout,
            readout="predictive",
            use_memory=self.config.use_memory,
            use_identity=False,
            _predictive_readout=predictive_readout,
        )

    def _prime(self, episode: LanguageEpisode) -> None:
        self.model.reset_dynamics(episode_id=f"r2:{episode.episode_id}")
        self._observe(self.model.config.boundary_symbol, learn=False)
        for symbol in episode.prompt_bytes:
            self._observe(symbol, learn=False)

    def _target_pass(self, episode: LanguageEpisode, *, learn: bool) -> dict[str, float]:
        self._prime(episode)
        if self.config.response_start_readout and episode.target_bytes:
            start_context = self.model.snapshot().motor_context
            start_probabilities = self.model.response_start_probabilities()
            if learn:
                self.model.learn_response_start(
                    start_context,
                    start_probabilities,
                    episode.target_bytes[0],
                )
        response_phase_readout = None
        if self.config.response_phase_readout:
            self.model.begin_response_phase()
            response_phase_readout = self.model.response_phase_readout
        if self.config.response_plan_readout:
            self.model.begin_response_plan()
            response_phase_readout = self.model.response_plan_readout
            if learn:
                response_phase_readout.learn_plan_target(self._response_plan_target(episode))
        observations = 0
        correct = 0
        surprise_sum = 0.0
        for symbol in episode.target_bytes:
            step = self._observe(
                symbol,
                learn=learn,
                predictive_readout=response_phase_readout,
            )
            if (
                self.config.response_plan_readout
                and self.config.response_plan_variant == "factorized_v1"
            ):
                self.model.advance_response_plan_phase()
            if step.prior_prediction is not None:
                observations += 1
                correct += int(step.prior_prediction == symbol)
                surprise_sum += float(step.surprise or 0.0)
        return {
            "observations": float(observations),
            "accuracy": correct / max(1, observations),
            "mean_surprise": surprise_sum / max(1, observations),
        }

    @staticmethod
    def _utf8_allowed(remaining: int, lead: int) -> list[int]:
        """Return the exact legal next-byte set for one UTF-8 code point."""

        if remaining == 0:
            return list(range(0x00, 0x80)) + list(range(0xC2, 0xF5))
        low, high = 0x80, 0xBF
        # A 3-byte sequence has two continuation bytes remaining after its
        # lead; the first one must enforce the E0/ED scalar-value bounds.
        if remaining == 2 and lead == 0xE0:
            low = 0xA0
        elif remaining == 2 and lead == 0xED:
            high = 0x9F
        # A 4-byte sequence has three continuation bytes remaining after its
        # lead; the first one must enforce the F0/F4 scalar-value bounds.
        elif remaining == 3 and lead == 0xF0:
            low = 0x90
        elif remaining == 3 and lead == 0xF4:
            high = 0x8F
        return list(range(low, high + 1))

    @staticmethod
    def _advance_utf8(remaining: int, lead: int, symbol: int) -> tuple[int, int]:
        if remaining == 0:
            if symbol < 0x80:
                return 0, 0
            if symbol < 0xE0:
                return 1, symbol
            if symbol < 0xF0:
                return 2, symbol
            return 3, symbol
        return remaining - 1, lead

    def _constrained_generate(
        self,
        episode: LanguageEpisode,
        *,
        max_generation_bytes: int | None = None,
    ) -> bytes:
        limit = (
            self.config.max_generation_bytes
            if max_generation_bytes is None
            else int(max_generation_bytes)
        )
        if limit <= 0:
            raise ValueError("max_generation_bytes must be positive")
        self.model.reset_dynamics(episode_id=f"r2-generation:{episode.episode_id}")
        step = self._observe(self.model.config.boundary_symbol, learn=False)
        for symbol in episode.prompt_bytes:
            step = self._observe(symbol, learn=False)
        generated = bytearray()
        remaining = 0
        lead = 0
        response_phase_readout = None
        if self.config.response_phase_readout:
            self.model.begin_response_phase()
            response_phase_readout = self.model.response_phase_readout
        if self.config.response_plan_readout:
            self.model.begin_response_plan()
            response_phase_readout = self.model.response_plan_readout
        for _ in range(limit):
            allowed = self._utf8_allowed(remaining, lead)
            probabilities = (
                self.model.response_start_probabilities().detach().cpu()
                if self.config.response_start_readout and not generated
                else (
                    self.model.response_phase_probabilities().detach().cpu()
                    if self.config.response_phase_readout
                    else step.probabilities.detach().cpu()
                )
            )
            if self.config.response_plan_readout:
                probabilities = (
                    self.model.response_plan_readout.probabilities(
                        self.model.snapshot().motor_context
                    )
                    .detach()
                    .cpu()
                )
            mask = torch.zeros_like(probabilities, dtype=torch.bool)
            mask[torch.tensor(allowed, dtype=torch.long)] = True
            masked = probabilities.clone()
            masked[~mask] = -1.0
            symbol = int(masked.argmax().item())
            generated.append(symbol)
            remaining, lead = self._advance_utf8(remaining, lead, symbol)
            step = self._observe(
                symbol,
                learn=False,
                predictive_readout=response_phase_readout,
            )
            if (
                self.config.response_plan_readout
                and self.config.response_plan_variant == "factorized_v1"
            ):
                self.model.advance_response_plan_phase()
            if generated.endswith(b"<|end|>"):
                break
        raw = bytes(generated)
        for cut in range(len(raw), max(0, len(raw) - 4), -1):
            try:
                raw[:cut].decode("utf-8")
            except UnicodeDecodeError:
                continue
            return raw[:cut]
        return raw

    def _beam_generate(
        self,
        episode: LanguageEpisode,
        *,
        beam_width: int,
        top_k: int,
        max_generation_bytes: int,
    ) -> bytes:
        """Run a bounded native sequence candidate without learning.

        Every beam carries only a cloned dynamics state.  Learned synapses,
        RNG streams and readout ownership remain shared and untouched; this
        is a decoder comparison, not a second model or a training shortcut.
        """

        if not self.config.constrained_decode:
            raise RuntimeError("native beam diagnostic requires constrained decoding")
        if int(beam_width) <= 0 or int(top_k) <= 0:
            raise ValueError("beam_width and top_k must be positive")
        if int(max_generation_bytes) <= 0:
            raise ValueError("max_generation_bytes must be positive")
        original = self.model.checkpoint()
        try:
            self.model.reset_dynamics(episode_id=f"r2-beam:{episode.episode_id}")
            self._observe(self.model.config.boundary_symbol, learn=False)
            for symbol in episode.prompt_bytes:
                self._observe(symbol, learn=False)
            response_phase_readout = None
            if self.config.response_phase_readout:
                self.model.begin_response_phase()
                response_phase_readout = self.model.response_phase_readout
            beams: list[tuple[bytes, float, int, int, Any]] = [
                (b"", 0.0, 0, 0, self.model.snapshot())
            ]
            completed: list[tuple[float, bytes]] = []
            for _ in range(int(max_generation_bytes)):
                candidates: list[tuple[bytes, float, int, int, Any]] = []
                for generated, score, remaining, lead, state in beams:
                    self.model.restore_dynamics(state)
                    probabilities = (
                        self.model.response_start_probabilities().detach().cpu()
                        if self.config.response_start_readout and not generated
                        else (
                            self.model.response_phase_probabilities().detach().cpu()
                            if self.config.response_phase_readout
                            else self.model.snapshot().motor_probabilities.detach().cpu()
                        )
                    )
                    allowed = torch.tensor(self._utf8_allowed(remaining, lead), dtype=torch.long)
                    values = probabilities[allowed].clamp_min(1e-12).log()
                    count = min(int(top_k), int(allowed.numel()))
                    top_values, top_indices = torch.topk(values, count)
                    for value, offset in zip(
                        top_values.tolist(), top_indices.tolist(), strict=True
                    ):
                        symbol = int(allowed[offset])
                        next_generated = generated + bytes((symbol,))
                        next_score = score + float(value)
                        if next_generated.endswith(b"<|end|>"):
                            completed.append((next_score, next_generated))
                            continue
                        next_remaining, next_lead = self._advance_utf8(remaining, lead, symbol)
                        self.model.restore_dynamics(state)
                        self._observe(
                            symbol,
                            learn=False,
                            predictive_readout=response_phase_readout,
                        )
                        candidates.append(
                            (
                                next_generated,
                                next_score,
                                next_remaining,
                                next_lead,
                                self.model.snapshot(),
                            )
                        )
                if not candidates:
                    break
                candidates.sort(key=lambda item: item[1], reverse=True)
                beams = candidates[: int(beam_width)]
            if completed:
                return max(completed, key=lambda item: item[0])[1]
            return max(beams, key=lambda item: item[1])[0] if beams else b""
        finally:
            self.model.restore(original)
            self._configure_developmental_mode()

    def sequence_decode_diagnostic(
        self,
        split: str = "train",
        *,
        beam_width: int = 4,
        top_k: int = 8,
        max_generation_bytes: int = 64,
    ) -> dict[str, Any]:
        """Compare greedy and bounded native sequence readout paths.

        The candidate is intentionally diagnostic-only.  It uses the same
        predictive probabilities and prefix state as greedy decoding, and
        reports whether sequence search can expose an existing conditional
        margin without adding target or label information.
        """

        episodes = self.corpus.for_split(split)
        if not episodes:
            raise ValueError(f"language alignment split is empty: {split}")
        if int(beam_width) <= 0 or int(top_k) <= 0 or int(max_generation_bytes) <= 0:
            raise ValueError("sequence decoder parameters must be positive")
        checkpoint = self.checkpoint()
        checkpoint_digest = str(checkpoint["checkpoint_digest"])

        def run(decoder: LanguageAlignmentTrainer) -> list[dict[str, Any]]:
            records: list[dict[str, Any]] = []
            for episode in episodes:
                greedy = decoder._constrained_generate(
                    episode,
                    max_generation_bytes=int(max_generation_bytes),
                )
                beam = decoder._beam_generate(
                    episode,
                    beam_width=int(beam_width),
                    top_k=int(top_k),
                    max_generation_bytes=int(max_generation_bytes),
                )

                def summarize(raw: bytes) -> dict[str, Any]:
                    text, valid_utf8, no_replacement, boundary, stop_reason = (
                        decoder._generated_text(raw)
                    )
                    sequence = decoder._sequence_evaluation(
                        episode,
                        text,
                        valid_utf8=valid_utf8,
                        no_replacement=no_replacement,
                        boundary_present=boundary,
                        stop_reason=stop_reason,
                    )
                    return {
                        "generated_text": text,
                        "generated_bytes_hex": raw.hex(),
                        "generated_bytes_length": len(raw),
                        "utf8_valid": valid_utf8,
                        "no_replacement": no_replacement,
                        "response_boundary_present": boundary,
                        "generation_stop_reason": stop_reason,
                        "exact_response": text == episode.response.strip(),
                        "sequence_criterion_pass": sequence["sequence_criterion_pass"],
                    }

                records.append(
                    {
                        "episode_id": episode.episode_id,
                        "reference_response": episode.response,
                        "greedy": summarize(greedy),
                        "beam": summarize(beam),
                        "beam_changed_output": greedy != beam,
                    }
                )
            return records

        try:
            records = run(self)
            restored = self.from_checkpoint(checkpoint, self.corpus)
            repeated = run(restored)
            recovery_repeatable = [
                (item["greedy"]["generated_bytes_hex"], item["beam"]["generated_bytes_hex"])
                for item in records
            ] == [
                (item["greedy"]["generated_bytes_hex"], item["beam"]["generated_bytes_hex"])
                for item in repeated
            ]
            if not recovery_repeatable:
                raise RuntimeError("sequence decoder diagnostic was not repeatable after restore")
            greedy_texts = [item["greedy"]["generated_text"] for item in records]
            beam_texts = [item["beam"]["generated_text"] for item in records]
            return {
                "format": LANGUAGE_ALIGNMENT_FORMAT,
                "status": "completed",
                "split": split,
                "readout_owner": "predictive_readout",
                "readout_scope": "protected",
                "decoder": "native-constrained-greedy-vs-beam-v1",
                "response_start_readout": self.config.response_start_readout,
                "response_phase_readout": self.config.response_phase_readout,
                "beam_width": int(beam_width),
                "top_k": int(top_k),
                "max_generation_bytes": int(max_generation_bytes),
                "episodes": len(records),
                "greedy_unique_generated_texts": len(set(greedy_texts)),
                "greedy_collision_rate": 1.0 - (len(set(greedy_texts)) / len(records)),
                "beam_unique_generated_texts": len(set(beam_texts)),
                "beam_collision_rate": 1.0 - (len(set(beam_texts)) / len(records)),
                "beam_changed_output_rate": sum(item["beam_changed_output"] for item in records)
                / len(records),
                "beam_exact_response_rate": sum(item["beam"]["exact_response"] for item in records)
                / len(records),
                "beam_sequence_criterion_pass_rate": sum(
                    item["beam"]["sequence_criterion_pass"] for item in records
                )
                / len(records),
                "checkpoint_read_only": True,
                "recovery_repeatable": recovery_repeatable,
                "native_mode_only": True,
                "records": records,
            }
        finally:
            self.model.restore(checkpoint["model"])
            self._configure_developmental_mode()
            if str(self.checkpoint()["checkpoint_digest"]) != checkpoint_digest:
                raise RuntimeError("sequence decoder diagnostic mutated the trainer checkpoint")

    def _generate(self, episode: LanguageEpisode) -> bytes:
        if self.config.constrained_decode:
            return self._constrained_generate(episode)
        return self.model.generate(
            episode.prompt_bytes,
            self.config.max_generation_bytes,
            stop_at_boundary=False,
            sample=False,
            reset=True,
            use_memory=self.config.use_memory,
            response_start=self.config.response_start_readout,
            response_phase=self.config.response_phase_readout,
        )

    @staticmethod
    def _distribution_js_divergence(left: torch.Tensor, right: torch.Tensor) -> float:
        epsilon = torch.finfo(torch.float32).tiny
        p = left.detach().to(dtype=torch.float32).clamp_min(epsilon)
        q = right.detach().to(dtype=torch.float32).clamp_min(epsilon)
        p = p / p.sum()
        q = q / q.sum()
        midpoint = 0.5 * (p + q)
        return float(
            0.5 * (p * (p.log() - midpoint.log())).sum()
            + 0.5 * (q * (q.log() - midpoint.log())).sum()
        )

    def _prefix_condition_snapshot(self, episode: LanguageEpisode) -> dict[str, Any]:
        checkpoint = self.model.checkpoint()
        before_digest = content_digest(checkpoint)
        try:
            self._prime(episode)
            state = self.model.checkpoint()["state"]
            context = state["motor_context"].detach().to("cpu", dtype=torch.float32)
            probabilities = state["motor_probabilities"].detach().to("cpu", dtype=torch.float32)
            return {
                "episode_id": episode.episode_id,
                "context": context,
                "probabilities": probabilities,
                "context_digest": content_digest(context.tolist()),
                "probabilities_digest": content_digest(probabilities.tolist()),
            }
        finally:
            self.model.restore(checkpoint)
            self._configure_developmental_mode()
            after_digest = content_digest(self.model.checkpoint())
            if after_digest != before_digest:
                raise RuntimeError("prefix condition diagnostic mutated the model checkpoint")

    def condition_route_diagnostic(self, split: str = "train") -> dict[str, Any]:
        """Measure whether distinct prompts survive into the native readout.

        This is deliberately read-only.  It compares the state immediately
        after each structured prefix, before any response target is consumed,
        so an output collision can be separated into prompt-state compression
        versus a later readout/decoding collapse.
        """

        episodes = self.corpus.for_split(split)
        if not episodes:
            raise ValueError(f"language alignment split is empty: {split}")
        trainer_checkpoint = self.checkpoint()
        trainer_checkpoint_digest = str(trainer_checkpoint["checkpoint_digest"])
        snapshots = [self._prefix_condition_snapshot(episode) for episode in episodes]
        restored = self.from_checkpoint(trainer_checkpoint, self.corpus)
        restored_snapshots = [restored._prefix_condition_snapshot(episode) for episode in episodes]
        recovery_repeatable = all(
            left["context_digest"] == right["context_digest"]
            and left["probabilities_digest"] == right["probabilities_digest"]
            for left, right in zip(snapshots, restored_snapshots, strict=True)
        )
        if not recovery_repeatable:
            raise RuntimeError("prefix condition diagnostic was not repeatable after restore")
        pair_context_l2: list[float] = []
        pair_context_cosine: list[float] = []
        pair_probability_l1: list[float] = []
        pair_probability_js: list[float] = []
        pair_argmax_difference: list[bool] = []
        for index, left in enumerate(snapshots):
            for right in snapshots[index + 1 :]:
                pair_context_l2.append(
                    float(torch.linalg.vector_norm(left["context"] - right["context"]))
                )
                left_norm = float(torch.linalg.vector_norm(left["context"]))
                right_norm = float(torch.linalg.vector_norm(right["context"]))
                if left_norm == 0.0 or right_norm == 0.0:
                    cosine_distance = float(left["context"].equal(right["context"]) is False)
                else:
                    cosine = torch.dot(left["context"], right["context"]) / (left_norm * right_norm)
                    cosine_distance = float(1.0 - cosine.clamp(-1.0, 1.0))
                pair_context_cosine.append(cosine_distance)
                pair_probability_l1.append(
                    float(0.5 * torch.abs(left["probabilities"] - right["probabilities"]).sum())
                )
                pair_probability_js.append(
                    self._distribution_js_divergence(left["probabilities"], right["probabilities"])
                )
                pair_argmax_difference.append(
                    int(left["probabilities"].argmax()) != int(right["probabilities"].argmax())
                )

        perturbation_context_l2: list[float] = []
        perturbation_probability_l1: list[float] = []
        for episode, original in zip(episodes, snapshots):
            perturbed = replace(
                episode,
                episode_id=f"{episode.episode_id}:input-perturbed",
                user_input=f"{episode.user_input}\n请换一种表达理解同一问题。",
            )
            changed = self._prefix_condition_snapshot(perturbed)
            perturbation_context_l2.append(
                float(torch.linalg.vector_norm(original["context"] - changed["context"]))
            )
            perturbation_probability_l1.append(
                float(0.5 * torch.abs(original["probabilities"] - changed["probabilities"]).sum())
            )

        context_digests = [item["context_digest"] for item in snapshots]
        if str(self.checkpoint()["checkpoint_digest"]) != trainer_checkpoint_digest:
            raise RuntimeError("condition route diagnostic mutated the trainer checkpoint")
        return {
            "format": LANGUAGE_ALIGNMENT_FORMAT,
            "status": "completed",
            "split": split,
            "readout_owner": "predictive_readout",
            "readout_scope": "protected",
            "state_component": "motor_context",
            "distribution_component": "motor_probabilities",
            "episodes": len(episodes),
            "pair_count": len(pair_context_l2),
            "distinct_prefix_contexts": len(set(context_digests)),
            "prefix_context_collision_rate": 1.0 - (len(set(context_digests)) / len(episodes)),
            "context_pair_l2_mean": sum(pair_context_l2) / max(1, len(pair_context_l2)),
            "context_pair_cosine_distance_mean": sum(pair_context_cosine)
            / max(1, len(pair_context_cosine)),
            "next_byte_probability_l1_mean": sum(pair_probability_l1)
            / max(1, len(pair_probability_l1)),
            "next_byte_probability_js_mean": sum(pair_probability_js)
            / max(1, len(pair_probability_js)),
            "next_byte_argmax_difference_rate": sum(pair_argmax_difference)
            / max(1, len(pair_argmax_difference)),
            "perturbation_count": len(perturbation_context_l2),
            "perturbation_context_l2_mean": sum(perturbation_context_l2)
            / max(1, len(perturbation_context_l2)),
            "perturbation_probability_l1_mean": sum(perturbation_probability_l1)
            / max(1, len(perturbation_probability_l1)),
            "checkpoint_read_only": True,
            "recovery_repeatable": recovery_repeatable,
            "native_mode_only": True,
            "records": [
                {
                    "episode_id": item["episode_id"],
                    "context_digest": item["context_digest"],
                    "probabilities_digest": item["probabilities_digest"],
                }
                for item in snapshots
            ],
        }

    def response_start_margin_diagnostic(self, split: str = "train") -> dict[str, Any]:
        """Measure the first-response-byte margin of the optional owner.

        This is a read-only diagnostic for separating a conditional-start
        margin from later sequence behaviour.  It reports legal UTF-8 start
        bytes only and never uses task-family labels or reference text to
        change the model.
        """

        if not self.config.response_start_readout:
            raise RuntimeError("response-start margin requires the optional readout")
        episodes = self.corpus.for_split(split)
        if not episodes:
            raise ValueError(f"language alignment split is empty: {split}")
        checkpoint = self.checkpoint()
        checkpoint_digest = str(checkpoint["checkpoint_digest"])
        allowed = torch.tensor(self._utf8_allowed(0, 0), dtype=torch.long)

        def run(decoder: LanguageAlignmentTrainer) -> list[dict[str, Any]]:
            records: list[dict[str, Any]] = []
            for episode in episodes:
                decoder._prime(episode)
                state = decoder.model.snapshot()
                probabilities = decoder.model.response_start_probabilities().detach().cpu()
                target_symbol = int(episode.target_bytes[0])
                legal_probabilities = probabilities[allowed]
                target_probability = float(probabilities[target_symbol].item())
                target_legal_probability = float(
                    legal_probabilities[allowed == target_symbol].item()
                )
                target_rank = 1 + int((legal_probabilities > target_legal_probability).sum().item())
                top_index = int(legal_probabilities.argmax().item())
                top_symbol = int(allowed[top_index].item())
                top_probability = float(legal_probabilities[top_index].item())
                other = legal_probabilities.clone()
                other[top_index] = -float("inf")
                second_probability = float(other.max().item())
                records.append(
                    {
                        "episode_id": episode.episode_id,
                        "target_first_byte": target_symbol,
                        "target_first_byte_hex": f"{target_symbol:02x}",
                        "target_probability": target_probability,
                        "target_legal_probability": target_legal_probability,
                        "target_rank_among_legal_starts": target_rank,
                        "argmax_legal_start": top_symbol,
                        "argmax_legal_start_hex": f"{top_symbol:02x}",
                        "argmax_legal_probability": top_probability,
                        "target_margin_vs_second": target_legal_probability - second_probability,
                        "context_digest": content_digest(
                            state.motor_context.detach().cpu().tolist()
                        ),
                        "probabilities_digest": content_digest(probabilities.tolist()),
                    }
                )
            return records

        try:
            records = run(self)
            restored = self.from_checkpoint(checkpoint, self.corpus)
            repeated = run(restored)
            recovery_repeatable = [
                (
                    item["target_first_byte"],
                    item["argmax_legal_start"],
                    item["probabilities_digest"],
                )
                for item in records
            ] == [
                (
                    item["target_first_byte"],
                    item["argmax_legal_start"],
                    item["probabilities_digest"],
                )
                for item in repeated
            ]
            if not recovery_repeatable:
                raise RuntimeError(
                    "response-start margin diagnostic was not repeatable after restore"
                )
            return {
                "format": LANGUAGE_ALIGNMENT_FORMAT,
                "status": "completed",
                "split": split,
                "readout_owner": "predictive_readout.response_start",
                "readout_scope": "candidate",
                "legal_start_count": int(allowed.numel()),
                "episodes": len(records),
                "target_top1_rate": sum(
                    item["target_rank_among_legal_starts"] == 1 for item in records
                )
                / len(records),
                "mean_target_rank_among_legal_starts": sum(
                    item["target_rank_among_legal_starts"] for item in records
                )
                / len(records),
                "mean_target_legal_probability": sum(
                    item["target_legal_probability"] for item in records
                )
                / len(records),
                "mean_target_margin_vs_second": sum(
                    item["target_margin_vs_second"] for item in records
                )
                / len(records),
                "checkpoint_read_only": True,
                "recovery_repeatable": recovery_repeatable,
                "native_mode_only": True,
                "records": records,
            }
        finally:
            self.model.restore(checkpoint["model"])
            self._configure_developmental_mode()
            self._configure_response_start_readout()
            if str(self.checkpoint()["checkpoint_digest"]) != checkpoint_digest:
                raise RuntimeError(
                    "response-start margin diagnostic mutated the trainer checkpoint"
                )

    def response_phase_margin_diagnostic(self, split: str = "train") -> dict[str, Any]:
        """Measure the first-byte margin of the continuous response owner.

        The diagnostic enters the response phase without learning, then reads
        the same isolated owner that receives every response-byte update.  It
        is intentionally read-only and keeps the first-byte margin visible so
        H3.3-C can be compared with the response-start candidate on identical
        splits.
        """

        if not self.config.response_phase_readout:
            raise RuntimeError("response-phase margin requires the optional readout")
        episodes = self.corpus.for_split(split)
        if not episodes:
            raise ValueError(f"language alignment split is empty: {split}")
        checkpoint = self.checkpoint()
        checkpoint_digest = str(checkpoint["checkpoint_digest"])
        allowed = torch.tensor(self._utf8_allowed(0, 0), dtype=torch.long)

        def run(decoder: LanguageAlignmentTrainer) -> list[dict[str, Any]]:
            records: list[dict[str, Any]] = []
            for episode in episodes:
                decoder._prime(episode)
                decoder.model.begin_response_phase()
                state = decoder.model.snapshot()
                probabilities = decoder.model.response_phase_probabilities().detach().cpu()
                target_symbol = int(episode.target_bytes[0])
                legal_probabilities = probabilities[allowed]
                target_probability = float(probabilities[target_symbol].item())
                target_legal_probability = float(
                    legal_probabilities[allowed == target_symbol].item()
                )
                target_rank = 1 + int((legal_probabilities > target_legal_probability).sum().item())
                top_index = int(legal_probabilities.argmax().item())
                top_symbol = int(allowed[top_index].item())
                top_probability = float(legal_probabilities[top_index].item())
                other = legal_probabilities.clone()
                other[top_index] = -float("inf")
                second_probability = float(other.max().item())
                records.append(
                    {
                        "episode_id": episode.episode_id,
                        "target_first_byte": target_symbol,
                        "target_first_byte_hex": f"{target_symbol:02x}",
                        "target_probability": target_probability,
                        "target_legal_probability": target_legal_probability,
                        "target_rank_among_legal_starts": target_rank,
                        "argmax_legal_start": top_symbol,
                        "argmax_legal_start_hex": f"{top_symbol:02x}",
                        "argmax_legal_probability": top_probability,
                        "target_margin_vs_second": target_legal_probability - second_probability,
                        "context_digest": content_digest(
                            state.motor_context.detach().cpu().tolist()
                        ),
                        "probabilities_digest": content_digest(probabilities.tolist()),
                    }
                )
            return records

        try:
            records = run(self)
            restored = self.from_checkpoint(checkpoint, self.corpus)
            repeated = run(restored)
            recovery_repeatable = [
                (
                    item["target_first_byte"],
                    item["argmax_legal_start"],
                    item["probabilities_digest"],
                )
                for item in records
            ] == [
                (
                    item["target_first_byte"],
                    item["argmax_legal_start"],
                    item["probabilities_digest"],
                )
                for item in repeated
            ]
            if not recovery_repeatable:
                raise RuntimeError(
                    "response-phase margin diagnostic was not repeatable after restore"
                )
            return {
                "format": LANGUAGE_ALIGNMENT_FORMAT,
                "status": "completed",
                "split": split,
                "readout_owner": "predictive_readout.response_phase",
                "readout_scope": "candidate",
                "legal_start_count": int(allowed.numel()),
                "episodes": len(records),
                "target_top1_rate": sum(
                    item["target_rank_among_legal_starts"] == 1 for item in records
                )
                / len(records),
                "mean_target_rank_among_legal_starts": sum(
                    item["target_rank_among_legal_starts"] for item in records
                )
                / len(records),
                "mean_target_legal_probability": sum(
                    item["target_legal_probability"] for item in records
                )
                / len(records),
                "mean_target_margin_vs_second": sum(
                    item["target_margin_vs_second"] for item in records
                )
                / len(records),
                "checkpoint_read_only": True,
                "recovery_repeatable": recovery_repeatable,
                "native_mode_only": True,
                "records": records,
            }
        finally:
            self.model.restore(checkpoint["model"])
            self._configure_developmental_mode()
            self._configure_response_phase_readout()
            if str(self.checkpoint()["checkpoint_digest"]) != checkpoint_digest:
                raise RuntimeError(
                    "response-phase margin diagnostic mutated the trainer checkpoint"
                )

    def generalization_diagnostic(
        self,
        splits: Sequence[str] = ("train", "dev", "final"),
    ) -> dict[str, Any]:
        """Profile conditional transfer without changing the checkpoint.

        H3.3 needs to distinguish a model that memorizes a response surface
        from one whose prefix state transfers to a new question.  This
        diagnostic therefore compares each split with the training response
        surface and reports first-byte overlap, response-prefix overlap,
        policy/family overlap, and the native first-byte margin.  The family
        and policy fields are analysis-only; they are never passed back into
        the model or used to alter a score.
        """

        requested = tuple(str(split) for split in splits)
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("generalization diagnostic needs unique non-empty splits")
        episodes_by_split = {split: self.corpus.for_split(split) for split in requested}
        if any(not episodes for episodes in episodes_by_split.values()):
            raise ValueError("generalization diagnostic cannot profile an empty split")
        train_episodes = self.corpus.for_split("train")
        train_response_bytes = {
            episode.episode_id: episode.response.encode("utf-8") for episode in train_episodes
        }
        train_user_bytes = {
            episode.episode_id: episode.user_input.encode("utf-8") for episode in train_episodes
        }
        train_first_bytes = {int(episode.target_bytes[0]) for episode in train_episodes}
        train_families = {episode.task_family for episode in train_episodes}
        train_policies = {episode.unknown_policy for episode in train_episodes}
        allowed = torch.tensor(self._utf8_allowed(0, 0), dtype=torch.long)

        def longest_common_prefix(left: bytes, right: bytes) -> int:
            count = 0
            for first, second in zip(left, right):
                if first != second:
                    break
                count += 1
            return count

        checkpoint = self.checkpoint()
        checkpoint_digest = str(checkpoint["checkpoint_digest"])

        def run(decoder: LanguageAlignmentTrainer) -> dict[str, Any]:
            profiles: dict[str, Any] = {}
            for split, episodes in episodes_by_split.items():
                records: list[dict[str, Any]] = []
                for episode in episodes:
                    decoder._prime(episode)
                    state = decoder.model.snapshot()
                    if decoder.config.response_phase_readout:
                        decoder.model.begin_response_phase()
                        probabilities = decoder.model.response_phase_probabilities().detach().cpu()
                        readout_owner = "predictive_readout.response_phase"
                    elif decoder.config.response_start_readout:
                        probabilities = decoder.model.response_start_probabilities().detach().cpu()
                        readout_owner = "predictive_readout.response_start"
                    else:
                        probabilities = state.motor_probabilities.detach().cpu()
                        readout_owner = "predictive_readout"
                    target_first_byte = int(episode.target_bytes[0])
                    legal_probabilities = probabilities[allowed]
                    target_probability = float(probabilities[target_first_byte].item())
                    target_legal_probability = float(
                        legal_probabilities[allowed == target_first_byte].item()
                    )
                    target_rank = 1 + int(
                        (legal_probabilities > target_legal_probability).sum().item()
                    )
                    top_index = int(legal_probabilities.argmax().item())
                    top_symbol = int(allowed[top_index].item())
                    top_probability = float(legal_probabilities[top_index].item())
                    other = legal_probabilities.clone()
                    other[top_index] = -float("inf")
                    second_probability = float(other.max().item())
                    response_bytes = episode.response.encode("utf-8")
                    comparable_responses = [
                        value
                        for episode_id, value in train_response_bytes.items()
                        if episode_id != episode.episode_id
                    ]
                    comparable_users = [
                        value
                        for episode_id, value in train_user_bytes.items()
                        if episode_id != episode.episode_id
                    ]
                    response_prefix_overlap = max(
                        (
                            longest_common_prefix(response_bytes, value)
                            for value in comparable_responses
                        ),
                        default=0,
                    )
                    user_prefix_overlap = max(
                        (
                            longest_common_prefix(episode.user_input.encode("utf-8"), value)
                            for value in comparable_users
                        ),
                        default=0,
                    )
                    records.append(
                        {
                            "episode_id": episode.episode_id,
                            "split": episode.split,
                            "task_family": episode.task_family,
                            "unknown_policy": episode.unknown_policy,
                            "target_first_byte": target_first_byte,
                            "target_first_byte_hex": f"{target_first_byte:02x}",
                            "target_first_byte_seen_in_train": target_first_byte
                            in train_first_bytes,
                            "target_response_exact_seen_in_train": response_bytes
                            in train_response_bytes.values(),
                            "target_response_exact_seen_in_other_train": response_bytes
                            in comparable_responses,
                            "target_response_prefix_max_lcp_bytes": response_prefix_overlap,
                            "target_user_input_max_lcp_bytes": user_prefix_overlap,
                            "task_family_seen_in_train": episode.task_family in train_families,
                            "unknown_policy_seen_in_train": episode.unknown_policy
                            in train_policies,
                            "target_response_byte_length": len(response_bytes),
                            "target_probability": target_probability,
                            "target_legal_probability": target_legal_probability,
                            "target_rank_among_legal_starts": target_rank,
                            "argmax_legal_start": top_symbol,
                            "argmax_legal_start_hex": f"{top_symbol:02x}",
                            "argmax_legal_probability": top_probability,
                            "target_margin_vs_second": target_legal_probability
                            - second_probability,
                            "context_digest": content_digest(
                                state.motor_context.detach().cpu().tolist()
                            ),
                            "probabilities_digest": content_digest(probabilities.tolist()),
                        }
                    )

                def mean(name: str) -> float:
                    return sum(float(item[name]) for item in records) / len(records)

                profiles[split] = {
                    "split": split,
                    "episodes": len(records),
                    "target_first_byte_seen_in_train_rate": mean("target_first_byte_seen_in_train"),
                    "target_response_exact_seen_in_train_rate": mean(
                        "target_response_exact_seen_in_train"
                    ),
                    "target_response_exact_seen_in_other_train_rate": mean(
                        "target_response_exact_seen_in_other_train"
                    ),
                    "target_response_prefix_max_lcp_bytes_mean": mean(
                        "target_response_prefix_max_lcp_bytes"
                    ),
                    "target_user_input_max_lcp_bytes_mean": mean("target_user_input_max_lcp_bytes"),
                    "task_family_seen_in_train_rate": mean("task_family_seen_in_train"),
                    "unknown_policy_seen_in_train_rate": mean("unknown_policy_seen_in_train"),
                    "target_first_byte_top1_rate": sum(
                        item["target_rank_among_legal_starts"] == 1 for item in records
                    )
                    / len(records),
                    "mean_target_rank_among_legal_starts": mean("target_rank_among_legal_starts"),
                    "mean_target_legal_probability": mean("target_legal_probability"),
                    "mean_target_margin_vs_second": mean("target_margin_vs_second"),
                    "readout_owner": readout_owner,
                    "records": records,
                }
            return profiles

        try:
            profiles = run(self)
            restored = self.from_checkpoint(checkpoint, self.corpus)
            repeated = run(restored)

            def recovery_signature(value: Mapping[str, Any]) -> list[tuple[Any, ...]]:
                return [
                    (
                        split,
                        item["episode_id"],
                        item["target_first_byte"],
                        item["argmax_legal_start"],
                        item["context_digest"],
                        item["probabilities_digest"],
                    )
                    for split, profile in value.items()
                    for item in profile["records"]
                ]

            recovery_repeatable = recovery_signature(profiles) == recovery_signature(repeated)
            if not recovery_repeatable:
                raise RuntimeError("generalization diagnostic was not repeatable after restore")
            return {
                "format": LANGUAGE_ALIGNMENT_GENERALIZATION_EVALUATION,
                "status": "completed",
                "corpus_digest": self.corpus_digest,
                "splits": list(requested),
                "train_target_first_bytes_hex": [
                    f"{value:02x}" for value in sorted(train_first_bytes)
                ],
                "train_task_families": sorted(train_families),
                "train_unknown_policies": sorted(train_policies),
                "response_start_readout": self.config.response_start_readout,
                "response_start_readout_digest": self.model.response_start_readout_digest,
                "response_phase_readout": self.config.response_phase_readout,
                "response_phase_readout_digest": self.model.response_phase_readout_digest,
                "checkpoint_read_only": True,
                "recovery_repeatable": recovery_repeatable,
                "native_mode_only": True,
                "external_provider": False,
                "profiles": profiles,
            }
        finally:
            self.model.restore(checkpoint["model"])
            self._configure_developmental_mode()
            self._configure_response_start_readout()
            self._configure_response_phase_readout()
            if str(self.checkpoint()["checkpoint_digest"]) != checkpoint_digest:
                raise RuntimeError("generalization diagnostic mutated the trainer checkpoint")

    def conditional_credit_diagnostic(
        self,
        splits: Sequence[str] = ("train", "dev", "final"),
        *,
        max_generation_bytes: int = 64,
    ) -> dict[str, Any]:
        """Audit response credit at every target position without learning.

        H3.4 must distinguish a first-byte margin from continuation credit,
        boundary/stop behaviour, and an unreadable free-generation branch.
        This diagnostic therefore records the exact native probability surface
        immediately before every teacher-forced target byte, then runs the
        ordinary read-only generator on the same episode.  It never forwards
        ``task_family`` or ``unknown_policy`` as model inputs, never applies
        result credit, and never changes the checkpoint.
        """

        requested = tuple(str(split) for split in splits)
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("conditional credit diagnostic needs unique non-empty splits")
        if int(max_generation_bytes) <= 0:
            raise ValueError("max_generation_bytes must be positive")
        episodes_by_split = {split: self.corpus.for_split(split) for split in requested}
        if any(not episodes for episodes in episodes_by_split.values()):
            raise ValueError("conditional credit diagnostic cannot profile an empty split")

        checkpoint = self.checkpoint()
        checkpoint_digest = str(checkpoint["checkpoint_digest"])
        end_marker = b"<|end|>"

        def position_role(index: int, response_length: int) -> str:
            boundary_start = response_length
            boundary_stop = boundary_start + len(end_marker)
            if index == 0:
                return "response_first_byte"
            if index < boundary_start:
                return "response_continuation_byte"
            if index < boundary_stop:
                return "response_end_marker_byte"
            return "response_post_end_marker_byte"

        def entropy(probabilities: torch.Tensor) -> float:
            epsilon = torch.finfo(probabilities.dtype).tiny
            safe = probabilities.clamp_min(epsilon)
            return float(-(probabilities * safe.log()).sum().item())

        def run(decoder: LanguageAlignmentTrainer) -> dict[str, Any]:
            profiles: dict[str, Any] = {}
            for split, episodes in episodes_by_split.items():
                records: list[dict[str, Any]] = []
                for episode in episodes:
                    decoder._prime(episode)
                    phase_owner = None
                    if decoder.config.response_phase_readout:
                        decoder.model.begin_response_phase()
                        phase_owner = decoder.model.response_phase_readout

                    remaining = 0
                    lead = 0
                    cumulative_log_probability = 0.0
                    cumulative_legal_log_probability = 0.0
                    position_records: list[dict[str, Any]] = []
                    response_length = len(episode.response.encode("utf-8"))

                    for index, symbol in enumerate(episode.target_bytes):
                        allowed_symbols = decoder._utf8_allowed(remaining, lead)
                        allowed = torch.tensor(allowed_symbols, dtype=torch.long)
                        if decoder.config.response_phase_readout:
                            probabilities = (
                                decoder.model.response_phase_probabilities()
                                .detach()
                                .to("cpu", dtype=torch.float32)
                            )
                            readout_owner = "predictive_readout.response_phase"
                            predictive_readout = phase_owner
                        elif decoder.config.response_start_readout and index == 0:
                            probabilities = (
                                decoder.model.response_start_probabilities()
                                .detach()
                                .to("cpu", dtype=torch.float32)
                            )
                            readout_owner = "predictive_readout.response_start"
                            predictive_readout = None
                        else:
                            probabilities = (
                                decoder.model.snapshot()
                                .motor_probabilities.detach()
                                .to("cpu", dtype=torch.float32)
                            )
                            readout_owner = "predictive_readout"
                            predictive_readout = None

                        target_probability = float(probabilities[int(symbol)].item())
                        target_rank = 1 + int(
                            (probabilities > probabilities[int(symbol)]).sum().item()
                        )
                        legal_probabilities = probabilities[allowed]
                        legal_mass = float(legal_probabilities.sum().item())
                        target_is_legal = int(symbol) in allowed_symbols
                        if target_is_legal:
                            target_legal_probability = target_probability / max(legal_mass, 1e-12)
                            target_legal_rank = 1 + int(
                                (legal_probabilities > probabilities[int(symbol)]).sum().item()
                            )
                        else:
                            target_legal_probability = 0.0
                            target_legal_rank = None
                        normalized_legal = legal_probabilities / max(legal_mass, 1e-12)
                        legal_top_index = int(legal_probabilities.argmax().item())
                        target_log_probability = math.log(max(target_probability, 1e-12))
                        target_legal_log_probability = math.log(
                            max(target_legal_probability, 1e-12)
                        )
                        cumulative_log_probability += target_log_probability
                        cumulative_legal_log_probability += target_legal_log_probability
                        state = decoder.model.snapshot()
                        position_records.append(
                            {
                                "position_index": int(index),
                                "position_role": position_role(index, response_length),
                                "target_symbol": int(symbol),
                                "target_symbol_hex": f"{int(symbol):02x}",
                                "target_is_legal_utf8": target_is_legal,
                                "readout_owner": readout_owner,
                                "target_probability": target_probability,
                                "target_log_probability": target_log_probability,
                                "target_rank": target_rank,
                                "target_top1": target_rank == 1,
                                "legal_mass": legal_mass,
                                "target_legal_probability": target_legal_probability,
                                "target_legal_log_probability": target_legal_log_probability,
                                "target_legal_rank": target_legal_rank,
                                "target_legal_top1": target_legal_rank == 1,
                                "legal_argmax_symbol": int(allowed[legal_top_index].item()),
                                "entropy": entropy(probabilities),
                                "legal_entropy": entropy(normalized_legal),
                                "cumulative_log_probability": cumulative_log_probability,
                                "cumulative_legal_log_probability": (
                                    cumulative_legal_log_probability
                                ),
                                "utf8_remaining_before": int(remaining),
                                "utf8_lead_before": int(lead),
                                "context_digest": content_digest(
                                    state.motor_context.detach().cpu().tolist()
                                ),
                                "probabilities_digest": content_digest(probabilities.tolist()),
                            }
                        )
                        decoder._observe(
                            int(symbol),
                            learn=False,
                            predictive_readout=predictive_readout,
                        )
                        remaining, lead = decoder._advance_utf8(remaining, lead, int(symbol))

                    generated = decoder._generate(episode)
                    (
                        generated_text,
                        valid_utf8,
                        no_replacement,
                        boundary_present,
                        stop_reason,
                    ) = decoder._generated_text(generated)
                    sequence = decoder._sequence_evaluation(
                        episode,
                        generated_text,
                        valid_utf8=valid_utf8,
                        no_replacement=no_replacement,
                        boundary_present=boundary_present,
                        stop_reason=stop_reason,
                    )
                    records.append(
                        {
                            "episode_id": episode.episode_id,
                            "split": episode.split,
                            "target_byte_count": len(episode.target_bytes),
                            "response_byte_count": response_length,
                            "positions": position_records,
                            "final_cumulative_log_probability": cumulative_log_probability,
                            "final_cumulative_legal_log_probability": (
                                cumulative_legal_log_probability
                            ),
                            "free_generation": {
                                "generated_text": generated_text,
                                "generated_bytes_hex": generated.hex(),
                                "generated_bytes_length": len(generated),
                                "utf8_valid": valid_utf8,
                                "no_replacement": no_replacement,
                                "response_boundary_present": boundary_present,
                                "generation_stop_reason": stop_reason,
                                "exact_response": generated_text == episode.response.strip(),
                                "sequence_criterion_pass": sequence["sequence_criterion_pass"],
                            },
                        }
                    )

                all_positions = [position for record in records for position in record["positions"]]

                def mean(items: Sequence[Mapping[str, Any]], name: str) -> float:
                    return sum(float(item[name]) for item in items) / max(1, len(items))

                def summarize(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
                    return {
                        "observations": len(items),
                        "target_top1_rate": sum(bool(item["target_top1"]) for item in items)
                        / max(1, len(items)),
                        "target_legal_top1_rate": sum(
                            bool(item["target_legal_top1"]) for item in items
                        )
                        / max(1, len(items)),
                        "mean_target_rank": mean(items, "target_rank"),
                        "mean_target_legal_rank": sum(
                            float(item["target_legal_rank"])
                            for item in items
                            if item["target_legal_rank"] is not None
                        )
                        / max(
                            1,
                            sum(item["target_legal_rank"] is not None for item in items),
                        ),
                        "mean_target_probability": mean(items, "target_probability"),
                        "mean_target_legal_probability": mean(items, "target_legal_probability"),
                        "mean_target_log_probability": mean(items, "target_log_probability"),
                        "mean_target_legal_log_probability": mean(
                            items, "target_legal_log_probability"
                        ),
                        "mean_entropy": mean(items, "entropy"),
                        "mean_legal_entropy": mean(items, "legal_entropy"),
                    }

                roles = sorted({str(item["position_role"]) for item in all_positions})
                by_role = {
                    role: summarize(
                        [item for item in all_positions if item["position_role"] == role]
                    )
                    for role in roles
                }
                indices = sorted({int(item["position_index"]) for item in all_positions})
                by_position = {
                    str(index): summarize(
                        [item for item in all_positions if item["position_index"] == index]
                    )
                    for index in indices
                }
                generated = [record["free_generation"] for record in records]
                unique_generated = len({str(item["generated_text"]) for item in generated})
                profiles[split] = {
                    "split": split,
                    "episodes": len(records),
                    "readout_owner": (
                        "predictive_readout.response_phase"
                        if decoder.config.response_phase_readout
                        else (
                            "predictive_readout.response_start+predictive_readout"
                            if decoder.config.response_start_readout
                            else "predictive_readout"
                        )
                    ),
                    "records": records,
                    "by_role": by_role,
                    "by_position": by_position,
                    "free_generation": {
                        "unique_generated_texts": unique_generated,
                        "generated_text_collision_rate": 1.0
                        - (unique_generated / max(1, len(generated))),
                        "utf8_valid_rate": sum(bool(item["utf8_valid"]) for item in generated)
                        / max(1, len(generated)),
                        "no_replacement_rate": sum(
                            bool(item["no_replacement"]) for item in generated
                        )
                        / max(1, len(generated)),
                        "response_boundary_rate": sum(
                            bool(item["response_boundary_present"]) for item in generated
                        )
                        / max(1, len(generated)),
                        "sequence_criterion_pass_rate": sum(
                            bool(item["sequence_criterion_pass"]) for item in generated
                        )
                        / max(1, len(generated)),
                        "exact_response_rate": sum(
                            bool(item["exact_response"]) for item in generated
                        )
                        / max(1, len(generated)),
                        "stop_reasons": sorted(
                            {str(item["generation_stop_reason"]) for item in generated}
                        ),
                    },
                }
            return profiles

        try:
            profiles = run(self)
            restored = self.from_checkpoint(checkpoint, self.corpus)
            repeated = run(restored)

            def recovery_signature(value: Mapping[str, Any]) -> list[tuple[Any, ...]]:
                return [
                    (
                        split,
                        record["episode_id"],
                        tuple(
                            (
                                position["position_index"],
                                position["target_symbol"],
                                position["readout_owner"],
                                position["probabilities_digest"],
                            )
                            for position in record["positions"]
                        ),
                        record["free_generation"]["generated_bytes_hex"],
                    )
                    for split, profile in value.items()
                    for record in profile["records"]
                ]

            recovery_repeatable = recovery_signature(profiles) == recovery_signature(repeated)
            if not recovery_repeatable:
                raise RuntimeError("conditional credit diagnostic was not repeatable after restore")
            return {
                "format": LANGUAGE_ALIGNMENT_CREDIT_EVALUATION,
                "status": "completed",
                "corpus_digest": self.corpus_digest,
                "splits": list(requested),
                "max_generation_bytes": int(max_generation_bytes),
                "response_start_readout": self.config.response_start_readout,
                "response_start_readout_digest": self.model.response_start_readout_digest,
                "response_phase_readout": self.config.response_phase_readout,
                "response_phase_readout_digest": self.model.response_phase_readout_digest,
                "checkpoint_read_only": True,
                "recovery_repeatable": recovery_repeatable,
                "native_mode_only": True,
                "external_provider": False,
                "result_credit_applied": False,
                "task_family_and_unknown_policy_forwarded": False,
                "profiles": profiles,
            }
        finally:
            self.model.restore(checkpoint["model"])
            self._configure_developmental_mode()
            self._configure_response_start_readout()
            self._configure_response_phase_readout()
            if str(self.checkpoint()["checkpoint_digest"]) != checkpoint_digest:
                raise RuntimeError("conditional credit diagnostic mutated the trainer checkpoint")

    def train(
        self,
        *,
        epochs: int = 1,
        max_episodes: int | None = None,
    ) -> dict[str, Any]:
        if int(epochs) <= 0:
            raise ValueError("language alignment epochs must be positive")
        if max_episodes is not None and int(max_episodes) <= 0:
            raise ValueError("max_episodes must be positive when provided")
        episodes = self.corpus.for_split("train")
        if max_episodes is not None:
            episodes = episodes[: int(max_episodes)]
        if not episodes:
            raise ValueError("language alignment train split is empty")
        records: list[dict[str, Any]] = []
        for epoch in range(int(epochs)):
            for episode in episodes:
                metrics = {"observations": 0.0, "accuracy": 0.0, "mean_surprise": 0.0}
                for _ in range(self.config.response_repeats):
                    current = self._target_pass(episode, learn=True)
                    for key in metrics:
                        metrics[key] += current[key]
                for key in metrics:
                    metrics[key] /= self.config.response_repeats
                self.global_step += int(metrics["observations"])
                self.episode_count += 1
                record = {
                    "epoch": epoch,
                    "episode_id": episode.episode_id,
                    "family_id": episode.family_id,
                    "target_bytes": len(episode.target_bytes),
                    "prefix_bytes": len(episode.prompt_bytes),
                    "observations": int(metrics["observations"]),
                    "response_accuracy": metrics["accuracy"],
                    "response_mean_surprise": metrics["mean_surprise"],
                }
                self.history.append(record)
                records.append(record)
            self._finish_developmental_epoch(epoch)
        return {
            "status": "completed",
            "epochs": int(epochs),
            "episodes": len(records),
            "global_step": self.global_step,
            "response_accuracy": sum(item["response_accuracy"] for item in records)
            / max(1, len(records)),
            "response_mean_surprise": sum(item["response_mean_surprise"] for item in records)
            / max(1, len(records)),
            "records": records,
            "developmental_mode": self.config.developmental_mode,
            "developmental_history": list(self.developmental_history),
        }

    @staticmethod
    def _generated_text(raw: bytes) -> tuple[str, bool, bool, bool, str]:
        end_marker = b"<|end|>"
        boundary_present = end_marker in raw
        clipped = raw.split(end_marker, 1)[0]
        valid_utf8 = True
        try:
            text = clipped.decode("utf-8")
        except UnicodeDecodeError:
            valid_utf8 = False
            text = clipped.decode("utf-8", errors="replace")
        if not valid_utf8:
            stop_reason = "invalid_utf8"
        elif boundary_present:
            stop_reason = "end_marker"
        else:
            stop_reason = "max_generation_bytes"
        return text.strip(), valid_utf8, "\ufffd" not in text, boundary_present, stop_reason

    @staticmethod
    def _sequence_evaluation(
        episode: LanguageEpisode,
        generated_text: str,
        *,
        valid_utf8: bool,
        no_replacement: bool,
        boundary_present: bool,
        stop_reason: str,
    ) -> dict[str, Any]:
        required_hits = tuple(term for term in episode.required_terms if term in generated_text)
        forbidden_hits = tuple(term for term in episode.forbidden_terms if term in generated_text)
        required_coverage = len(required_hits) / max(1, len(episode.required_terms))
        if episode.unknown_policy == "answer":
            unknown_policy_satisfied: bool | None = True
            unknown_policy_evaluated = False
        elif episode.unknown_policy == "say_unknown":
            markers = episode.unknown_markers or ("不知道", "不清楚", "无法确定", "资料不足")
            unknown_policy_satisfied = any(marker in generated_text for marker in markers)
            unknown_policy_evaluated = True
        else:
            unknown_policy_satisfied = None
            unknown_policy_evaluated = False
        semantic_evaluated = bool(
            episode.required_terms or episode.forbidden_terms or unknown_policy_evaluated
        )
        required_terms_satisfied = not episode.required_terms or required_coverage >= 1.0
        semantic_pass = (
            bool(valid_utf8 and no_replacement)
            and required_terms_satisfied
            and not forbidden_hits
            and unknown_policy_satisfied is not False
            if semantic_evaluated
            else None
        )
        return {
            "sequence_evaluation": LANGUAGE_ALIGNMENT_SEQUENCE_EVALUATION,
            "response_boundary_present": boundary_present,
            "generation_stop_reason": stop_reason,
            "sequence_valid": bool(valid_utf8 and no_replacement and boundary_present),
            "required_terms": list(episode.required_terms),
            "required_terms_hit": list(required_hits),
            "required_term_coverage": required_coverage,
            "forbidden_terms": list(episode.forbidden_terms),
            "forbidden_terms_hit": list(forbidden_hits),
            "unknown_policy_satisfied": unknown_policy_satisfied,
            "semantic_criteria_evaluated": semantic_evaluated,
            "semantic_criteria_pass": semantic_pass,
            "sequence_criterion_pass": bool(
                valid_utf8 and no_replacement and boundary_present and (semantic_pass is not False)
            ),
        }

    def score_episode(self, episode: LanguageEpisode) -> dict[str, Any]:
        if not isinstance(episode, LanguageEpisode):
            raise TypeError("score_episode requires a LanguageEpisode")
        checkpoint = self.model.checkpoint()
        before_digest = content_digest(checkpoint)
        try:
            teacher = self._target_pass(episode, learn=False)
            generated = self._generate(episode)
            (
                generated_text,
                valid_utf8,
                no_replacement,
                boundary_present,
                stop_reason,
            ) = self._generated_text(generated)
            sequence = self._sequence_evaluation(
                episode,
                generated_text,
                valid_utf8=valid_utf8,
                no_replacement=no_replacement,
                boundary_present=boundary_present,
                stop_reason=stop_reason,
            )
            return {
                "episode_id": episode.episode_id,
                "split": episode.split,
                "task_family": episode.task_family,
                "reference_response": episode.response,
                "unknown_policy": episode.unknown_policy,
                "unknown_markers": list(episode.unknown_markers),
                "response_start_readout": self.config.response_start_readout,
                "response_start_readout_digest": self.model.response_start_readout_digest,
                "response_phase_readout": self.config.response_phase_readout,
                "response_phase_readout_digest": self.model.response_phase_readout_digest,
                "teacher_forced_accuracy": teacher["accuracy"],
                "teacher_forced_mean_surprise": teacher["mean_surprise"],
                "generated_text": generated_text,
                "generated_bytes_hex": generated.hex(),
                "generated_bytes_length": len(generated),
                "utf8_valid": valid_utf8,
                "no_replacement": no_replacement,
                "exact_response": generated_text == episode.response.strip(),
                "native_mode": True,
                "external_provider": False,
                **sequence,
            }
        finally:
            self.model.restore(checkpoint)
            # Taiji restores developmental F1 in read-only mode by contract.
            # A trainer-owned score is read-only, but it must return the
            # explicitly selected mode before a subsequent controlled update.
            self._configure_developmental_mode()
            after_digest = content_digest(self.model.checkpoint())
            if after_digest != before_digest:
                raise RuntimeError("language alignment score mutated the model checkpoint")

    def evaluate(self, split: str = "dev") -> dict[str, Any]:
        episodes = self.corpus.for_split(split)
        if not episodes:
            raise ValueError(f"language alignment split is empty: {split}")
        records = [self.score_episode(episode) for episode in episodes]
        generated_texts = [str(item["generated_text"]) for item in records]
        unique_generated_texts = len(set(generated_texts))
        return {
            "format": LANGUAGE_ALIGNMENT_FORMAT,
            "split": split,
            "episodes": len(records),
            "corpus_digest": self.corpus_digest,
            "response_start_readout": self.config.response_start_readout,
            "response_start_readout_digest": self.model.response_start_readout_digest,
            "response_phase_readout": self.config.response_phase_readout,
            "response_phase_readout_digest": self.model.response_phase_readout_digest,
            "teacher_forced_accuracy": sum(
                float(item["teacher_forced_accuracy"]) for item in records
            )
            / len(records),
            "teacher_forced_mean_surprise": sum(
                float(item["teacher_forced_mean_surprise"]) for item in records
            )
            / len(records),
            "exact_response_rate": sum(bool(item["exact_response"]) for item in records)
            / len(records),
            "utf8_valid_rate": sum(bool(item["utf8_valid"]) for item in records) / len(records),
            "no_replacement_rate": sum(bool(item["no_replacement"]) for item in records)
            / len(records),
            "response_boundary_rate": sum(
                bool(item["response_boundary_present"]) for item in records
            )
            / len(records),
            "sequence_valid_rate": sum(bool(item["sequence_valid"]) for item in records)
            / len(records),
            "sequence_criterion_pass_rate": sum(
                bool(item["sequence_criterion_pass"]) for item in records
            )
            / len(records),
            "required_term_coverage": sum(float(item["required_term_coverage"]) for item in records)
            / len(records),
            "semantic_criteria_evaluated": sum(
                bool(item["semantic_criteria_evaluated"]) for item in records
            ),
            "semantic_criteria_pass_rate": (
                sum(
                    bool(item["semantic_criteria_pass"])
                    for item in records
                    if item["semantic_criteria_evaluated"]
                )
                / max(
                    1,
                    sum(bool(item["semantic_criteria_evaluated"]) for item in records),
                )
            ),
            "unique_generated_texts": unique_generated_texts,
            "generated_text_collision_rate": 1.0 - (unique_generated_texts / len(records)),
            "records": records,
            "checkpoint_read_only": True,
            "native_mode_only": True,
        }

    def checkpoint(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": self.CHECKPOINT_FORMAT,
            "version": self.CHECKPOINT_VERSION,
            "serialization": LANGUAGE_ALIGNMENT_SERIALIZATION,
            "corpus_digest": self.corpus_digest,
            "config": self.config.to_payload(),
            "code_revision": self.code_revision,
            "global_step": self.global_step,
            "episode_count": self.episode_count,
            "history": list(self.history),
            "developmental_history": list(self.developmental_history),
            "model": self.model.checkpoint(),
        }
        if self.config.response_plan_target_geometry in {
            LANGUAGE_RESPONSE_PLAN_TARGET_H36,
            LANGUAGE_RESPONSE_PLAN_TARGET_H37,
        }:
            if self.response_plan_target_encoder is None:
                raise RuntimeError("response-plan target encoder is not configured")
            payload["response_plan_target_encoder"] = self.response_plan_target_encoder.to_payload()
            payload["response_plan_targets"] = {
                episode_id: target.detach().cpu().clone()
                for episode_id, target in sorted(self.response_plan_targets.items())
            }
        payload["checkpoint_digest"] = content_digest(payload)
        return payload

    def save(self, path: str | Path) -> Path:
        return _atomic_save(self.checkpoint(), path)

    @classmethod
    def from_checkpoint(
        cls,
        payload: Mapping[str, Any],
        corpus: LanguageEpisodeCorpus,
        *,
        device: torch.device | str = "cpu",
    ) -> LanguageAlignmentTrainer:
        if payload.get("format") != cls.CHECKPOINT_FORMAT:
            raise ValueError("unsupported language alignment checkpoint format")
        if int(payload.get("version", -1)) != cls.CHECKPOINT_VERSION:
            raise ValueError("unsupported language alignment checkpoint version")
        expected_digest = content_digest(
            {key: value for key, value in payload.items() if key != "checkpoint_digest"}
        )
        if payload.get("checkpoint_digest") != expected_digest:
            raise ValueError("language alignment checkpoint digest mismatch")
        if payload.get("corpus_digest") != corpus.digest:
            raise ValueError("language alignment checkpoint corpus digest mismatch")
        model_payload = payload.get("model")
        if not isinstance(model_payload, Mapping):
            raise ValueError("language alignment checkpoint is missing model")
        model = Taiji.from_checkpoint(model_payload, device=device)
        config_payload = payload.get("config")
        if not isinstance(config_payload, Mapping):
            raise ValueError("language alignment checkpoint is missing config")
        config = LanguageAlignmentConfig(**dict(config_payload))
        target_encoder = None
        target_payload = payload.get("response_plan_target_encoder")
        target_targets_payload = payload.get("response_plan_targets")
        if config.response_plan_target_geometry == LANGUAGE_RESPONSE_PLAN_TARGET_H36:
            if not isinstance(target_payload, Mapping):
                raise ValueError("H3.6 checkpoint is missing its target encoder")
            target_encoder = ResponsePlanTargetEncoder.from_payload(
                target_payload,
                expected_corpus_digest=corpus.digest,
            )
            if not isinstance(target_targets_payload, Mapping):
                raise ValueError("H3.6 checkpoint is missing its train targets")
            target_targets = {
                str(key): value.detach().cpu().to(dtype=torch.float32).clone()
                for key, value in target_targets_payload.items()
            }
        elif config.response_plan_target_geometry == LANGUAGE_RESPONSE_PLAN_TARGET_H37:
            if not isinstance(target_payload, Mapping):
                raise ValueError("H3.7 checkpoint is missing its target encoder")
            target_encoder = FactorizedResponsePlanTargetEncoder.from_payload(
                target_payload,
                expected_corpus_digest=corpus.digest,
            )
            if not isinstance(target_targets_payload, Mapping):
                raise ValueError("H3.7 checkpoint is missing its train targets")
            target_targets = {
                str(key): value.detach().cpu().to(dtype=torch.float32).clone()
                for key, value in target_targets_payload.items()
            }
        else:
            if target_payload is not None or target_targets_payload is not None:
                raise ValueError("legacy response-plan checkpoint cannot contain target state")
            target_targets = None
        trainer = cls(
            model,
            corpus,
            config=config,
            code_revision=str(payload.get("code_revision", "working-tree")),
            response_plan_target_encoder=target_encoder,
            response_plan_targets=target_targets,
        )
        trainer.global_step = int(payload.get("global_step", 0))
        trainer.episode_count = int(payload.get("episode_count", 0))
        history = payload.get("history", ())
        if not isinstance(history, Sequence) or isinstance(history, (str, bytes)):
            raise ValueError("language alignment checkpoint history is invalid")
        trainer.history = [dict(item) for item in history]
        developmental_history = payload.get("developmental_history", ())
        if not isinstance(developmental_history, Sequence) or isinstance(
            developmental_history, (str, bytes)
        ):
            raise ValueError("language alignment developmental history is invalid")
        trainer.developmental_history = [dict(item) for item in developmental_history]
        return trainer


def paired_checkpoint_diagnostic(
    trainer: LanguageAlignmentTrainer,
    baseline_payload: Mapping[str, Any],
    *,
    splits: Sequence[str] = ("dev", "final"),
) -> dict[str, Any]:
    """Compare a zero-step checkpoint with the trained child on original and
    minimally perturbed prompts.

    This is a transfer diagnostic, not an ability score.  It separates a
    local response proxy change from evidence that the trained child still
    conditions its native output on an unseen prompt.  Both trainers score in
    read-only mode and the caller's child model must remain byte-for-byte
    checkpoint stable.
    """

    if not isinstance(trainer, LanguageAlignmentTrainer):
        raise TypeError("paired checkpoint diagnostic requires a LanguageAlignmentTrainer")
    if not isinstance(baseline_payload, Mapping):
        raise TypeError("baseline_payload must be a checkpoint mapping")
    baseline = LanguageAlignmentTrainer.from_checkpoint(baseline_payload, trainer.corpus)
    baseline_model_digest = content_digest(baseline.model.checkpoint())
    child_model_digest = content_digest(trainer.model.checkpoint())
    records: list[dict[str, Any]] = []
    for split in splits:
        for episode in trainer.corpus.for_split(split):
            perturbed = replace(
                episode,
                episode_id=f"{episode.episode_id}:input-perturbed",
                user_input=f"{episode.user_input}\n请换一种表达理解同一问题。",
            )
            baseline_original = baseline.score_episode(episode)
            child_original = trainer.score_episode(episode)
            baseline_perturbed = baseline.score_episode(perturbed)
            child_perturbed = trainer.score_episode(perturbed)
            records.append(
                {
                    "episode_id": episode.episode_id,
                    "split": split,
                    "baseline": baseline_original,
                    "child": child_original,
                    "baseline_perturbed": baseline_perturbed,
                    "child_perturbed": child_perturbed,
                    "baseline_prompt_sensitive": (
                        baseline_original["generated_text"] != baseline_perturbed["generated_text"]
                    ),
                    "child_prompt_sensitive": (
                        child_original["generated_text"] != child_perturbed["generated_text"]
                    ),
                    "child_output_changed_after_update": (
                        baseline_original["generated_text"] != child_original["generated_text"]
                    ),
                }
            )
    if content_digest(baseline.model.checkpoint()) != baseline_model_digest:
        raise RuntimeError("paired baseline diagnostic mutated the baseline model")
    if content_digest(trainer.model.checkpoint()) != child_model_digest:
        raise RuntimeError("paired child diagnostic mutated the child model")
    if not records:
        raise ValueError("paired checkpoint diagnostic needs at least one episode")
    return {
        "format": LANGUAGE_ALIGNMENT_FORMAT,
        "status": "completed",
        "splits": list(splits),
        "episodes": len(records),
        "teacher_forced_mean_surprise_delta": sum(
            float(item["child"]["teacher_forced_mean_surprise"])
            - float(item["baseline"]["teacher_forced_mean_surprise"])
            for item in records
        )
        / len(records),
        "exact_response_delta": sum(
            int(bool(item["child"]["exact_response"]))
            - int(bool(item["baseline"]["exact_response"]))
            for item in records
        )
        / len(records),
        "baseline_prompt_sensitivity_rate": sum(
            bool(item["baseline_prompt_sensitive"]) for item in records
        )
        / len(records),
        "child_prompt_sensitivity_rate": sum(
            bool(item["child_prompt_sensitive"]) for item in records
        )
        / len(records),
        "child_output_changed_after_update_rate": sum(
            bool(item["child_output_changed_after_update"]) for item in records
        )
        / len(records),
        "checkpoint_read_only": True,
        "native_mode_only": True,
        "external_provider": False,
        "records": records,
    }


def checkpoint_roundtrip_preflight(
    trainer: LanguageAlignmentTrainer,
    *,
    episode: LanguageEpisode | None = None,
    directory: str | Path | None = None,
) -> dict[str, Any]:
    """Run zero-step and one-response-update disk round-trip checks.

    The one update happens only in a temporary child checkpoint.  The caller's
    trainer and its parent model are restored to the exact zero-step payload
    before this function returns.
    """

    if not isinstance(trainer, LanguageAlignmentTrainer):
        raise TypeError("checkpoint preflight requires a LanguageAlignmentTrainer")
    selected = episode or trainer.corpus.for_split("train")[0]
    parent_payload = trainer.checkpoint()
    parent_digest = str(parent_payload["checkpoint_digest"])
    original_model_payload = trainer.model.checkpoint()
    original_model_digest = content_digest(original_model_payload)
    if directory is None:
        context = tempfile.TemporaryDirectory(prefix="r2-language-preflight-")
        root = Path(context.name)
    else:
        context = None
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
    try:
        zero_path = root / "zero.pt"
        child_path = root / "child.pt"
        _atomic_save(parent_payload, zero_path)
        zero_payload = torch.load(zero_path, map_location="cpu", weights_only=False)
        zero_trainer = LanguageAlignmentTrainer.from_checkpoint(zero_payload, trainer.corpus)
        zero_output = zero_trainer.score_episode(selected)
        zero_restored_digest = str(zero_trainer.checkpoint()["checkpoint_digest"])
        if zero_restored_digest != parent_digest:
            raise RuntimeError("zero-step checkpoint digest changed after restore")
        child_result = zero_trainer.train(epochs=1, max_episodes=1)
        child_payload = zero_trainer.checkpoint()
        child_digest = str(child_payload["checkpoint_digest"])
        if child_digest == parent_digest:
            raise RuntimeError("one response update did not change the child checkpoint")
        _atomic_save(child_payload, child_path)
        child_payload_loaded = torch.load(child_path, map_location="cpu", weights_only=False)
        child_restored = LanguageAlignmentTrainer.from_checkpoint(
            child_payload_loaded, trainer.corpus
        )
        child_output = child_restored.score_episode(selected)
        if str(child_restored.checkpoint()["checkpoint_digest"]) != child_digest:
            raise RuntimeError("child checkpoint digest changed after restore")
        if content_digest(trainer.model.checkpoint()) != original_model_digest:
            raise RuntimeError("preflight mutated the caller model")
        return {
            "format": LANGUAGE_ALIGNMENT_FORMAT,
            "status": "passed",
            "parent_checkpoint_digest": parent_digest,
            "zero_step_roundtrip": True,
            "child_checkpoint_digest": child_digest,
            "one_response_update": child_result,
            "zero_output": zero_output,
            "child_output": child_output,
            "caller_model_unchanged": True,
            "atomic_save": True,
        }
    finally:
        if context is not None:
            context.cleanup()


def factorized_response_plan_preflight(
    trainer: LanguageAlignmentTrainer,
    *,
    episode: LanguageEpisode | None = None,
) -> dict[str, Any]:
    """Verify H3.7 slot lifecycle and causal candidate-only credit.

    The ordinary checkpoint preflight covers disk round-trips.  H3.7 adds
    this smaller in-memory contract so a formal run cannot start when a byte
    update changes only the renderer while leaving its declared plan bridge
    and slot planner untouched.
    """

    if not isinstance(trainer, LanguageAlignmentTrainer):
        raise TypeError("factorized response-plan preflight requires a language trainer")
    if not trainer.config.response_plan_readout:
        raise ValueError("factorized response-plan preflight requires the plan readout")
    if trainer.config.response_plan_variant != "factorized_v1":
        raise ValueError("factorized response-plan preflight requires factorized_v1")
    selected = episode or trainer.corpus.for_split("train")[0]
    baseline = trainer.checkpoint()
    baseline_digest = str(baseline["checkpoint_digest"])
    protected_digest = content_digest(trainer.model.predictive_readout.to_payload())
    readout = trainer.model.response_plan_readout
    try:
        trainer._prime(selected)
        trainer.model.begin_response_plan()
        if readout.plan_slots != trainer.config.response_plan_slots:
            raise RuntimeError("factorized plan slot count is not configured")
        if readout.plan_step != 0 or readout.plan_phase != 0:
            raise RuntimeError("factorized plan did not begin at phase zero")
        before_target_digest = content_digest(readout.to_payload())
        readout.learn_plan_target(trainer._response_plan_target(selected))
        after_target_digest = content_digest(readout.to_payload())
        if before_target_digest == after_target_digest:
            raise RuntimeError("H3.7 plan target update did not change the candidate")
        before_bridge_digest = content_digest(readout.plan_bridge.detach().cpu().tolist())
        before_planner_digest = content_digest(
            readout.planner_weight.detach().cpu().tolist()
        )
        trainer._observe(
            selected.target_bytes[0],
            learn=True,
            predictive_readout=readout,
        )
        after_bridge_digest = content_digest(readout.plan_bridge.detach().cpu().tolist())
        after_planner_digest = content_digest(
            readout.planner_weight.detach().cpu().tolist()
        )
        bridge_changed = before_bridge_digest != after_bridge_digest
        planner_changed = before_planner_digest != after_planner_digest
        if not bridge_changed and not planner_changed:
            raise RuntimeError("H3.7 byte update did not change bridge or slot planner credit")
        phase_after_advance = trainer.model.advance_response_plan_phase()
        if readout.plan_step != 1 or phase_after_advance != 0:
            raise RuntimeError("H3.7 phase advance did not preserve phase-zero slot at step one")
        trainer.model.reset_dynamics(episode_id="h3-7-preflight-reset")
        if readout.plan_state is not None or readout.plan_step != 0:
            raise RuntimeError("H3.7 reset did not clear factorized plan state")
        if content_digest(trainer.model.predictive_readout.to_payload()) != protected_digest:
            raise RuntimeError("H3.7 candidate credit changed the protected readout")
        return {
            "format": "taiji-r2-h3-7-factorized-preflight-v1",
            "status": "passed",
            "episode_id": selected.episode_id,
            "plan_target_candidate_changed": True,
            "bridge_changed_after_byte_update": bridge_changed,
            "slot_planner_changed_after_byte_update": planner_changed,
            "phase_after_advance": int(phase_after_advance),
            "reset_cleared_plan": True,
            "protected_readout_unchanged": True,
            "caller_restore_verified_in_finally": True,
        }
    finally:
        trainer.model.restore(baseline["model"])
        if str(trainer.checkpoint()["checkpoint_digest"]) != baseline_digest:
            raise RuntimeError("H3.7 preflight failed to restore the caller trainer")


__all__ = [
    "LANGUAGE_ALIGNMENT_CREDIT_EVALUATION",
    "LANGUAGE_ALIGNMENT_FORMAT",
    "LANGUAGE_ALIGNMENT_GENERALIZATION_EVALUATION",
    "LANGUAGE_ALIGNMENT_MARKERS",
    "LANGUAGE_ALIGNMENT_SERIALIZATION",
    "LANGUAGE_ALIGNMENT_SEQUENCE_EVALUATION",
    "LANGUAGE_ALIGNMENT_SPLITS",
    "LANGUAGE_ALIGNMENT_VERSION",
    "LanguageAlignmentConfig",
    "LanguageAlignmentTrainer",
    "LanguageEpisode",
    "LanguageEpisodeCorpus",
    "checkpoint_roundtrip_preflight",
    "factorized_response_plan_preflight",
    "paired_checkpoint_diagnostic",
]
