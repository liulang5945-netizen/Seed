"""Versioned response-plan target geometry for the R2 H3.6 candidate.

The target is deliberately separate from the response-plan readout.  A
teacher model supplies a read-only native response state, while this module
fits the native whitening transform on the training split only and combines
it with a deterministic compositional character n-gram projection.

This is a plumbing artifact, not a training policy.  It is content-addressed
by the corpus and by the exact parent model checkpoint so a later trainer
cannot silently regenerate targets from a different model or from a holdout
split.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import torch

from .internalization import content_digest

RESPONSE_PLAN_TARGET_FORMAT = "taiji-r2-h3-6-response-plan-target-v1"
RESPONSE_PLAN_TARGET_VERSION = 1
RESPONSE_PLAN_TARGET_WIDTH = 32
RESPONSE_PLAN_TARGET_FIT_SPLIT = "train"
RESPONSE_PLAN_TARGET_NATIVE_SOURCE = "mean_teacher_forced_motor_context"
RESPONSE_PLAN_TARGET_NATIVE_TRANSFORM = "train_covariance_eigendecomposition_whitened"
RESPONSE_PLAN_TARGET_COMPOSITION = "equal_normalized_native_and_signed_char_ngram"
RESPONSE_PLAN_TARGET_NGRAM_SALT = b"r2-h3-6-plan-geometry-v1\x00"
FACTOR_RESPONSE_PLAN_TARGET_FORMAT = "taiji-r2-h3-7-factorized-response-target-v1"
FACTOR_RESPONSE_PLAN_TARGET_VERSION = 1
FACTOR_RESPONSE_PLAN_TARGET_SLOTS = 4
FACTOR_RESPONSE_PLAN_TARGET_SLOT_WIDTH = 12
FACTOR_RESPONSE_PLAN_TARGET_PHASE_STRIDE = 16
FACTOR_RESPONSE_PLAN_TARGET_PROJECTION = "train_bound_count_sketch_signed_ngram"
FACTOR_RESPONSE_PLAN_TARGET_NORMALIZATION = "per_slot_l2"
FACTOR_RESPONSE_PLAN_TARGET_SALT = b"r2-h3-7-factorized-response-target-v1\x00"


def _normalize(vector: torch.Tensor, *, name: str) -> torch.Tensor:
    value = vector.detach().to(device="cpu", dtype=torch.float32).contiguous()
    norm = torch.linalg.vector_norm(value)
    if not bool(torch.isfinite(norm)) or float(norm) <= 1e-12:
        raise ValueError(f"{name} produced a zero or non-finite vector")
    return value / norm


def _required_digest(value: Any, name: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64:
        raise ValueError(f"{name} must be a 64-character SHA-256 digest")
    try:
        int(normalized, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a hexadecimal SHA-256 digest") from exc
    return normalized


def _signed_hash(token: bytes, *, width: int) -> torch.Tensor:
    digest = hashlib.sha256(RESPONSE_PLAN_TARGET_NGRAM_SALT + token).digest()
    return torch.tensor(
        [1.0 if digest[index % len(digest)] & 1 else -1.0 for index in range(width)],
        dtype=torch.float32,
    )


def _compositional_char_ngram(episode: Any, *, width: int) -> torch.Tensor:
    text = str(episode.response)
    vector = torch.zeros(width, dtype=torch.float32)
    total_weight = 0.0
    for size, weight in ((1, 1.0), (2, 1.5), (3, 2.0)):
        for index in range(max(0, len(text) - size + 1)):
            token = text[index : index + size].encode("utf-8")
            vector += weight * _signed_hash(bytes([size]) + token, width=width)
            total_weight += weight
    if total_weight == 0.0:
        vector += _signed_hash(text.encode("utf-8"), width=width)
    return _normalize(vector, name="compositional character n-gram")


def _native_teacher_raw(model: Any, episode: Any) -> torch.Tensor:
    """Read one response state without changing persistent model state."""

    response = str(episode.response)
    if not response:
        raise ValueError(f"episode {episode.episode_id} has an empty response")
    model.reset_dynamics(episode_id=f"h3.6-target:{episode.episode_id}")
    contexts: list[torch.Tensor] = []
    model.observe(
        model.config.boundary_symbol,
        learn=False,
        readout="predictive",
        use_memory=False,
        use_identity=False,
    )
    for symbol in response.encode("utf-8"):
        model.observe(
            symbol,
            learn=False,
            readout="predictive",
            use_memory=False,
            use_identity=False,
        )
        contexts.append(model.snapshot().motor_context.detach().cpu())
    return torch.stack(contexts).mean(dim=0).to(dtype=torch.float32)


def _corpus_digest(corpus: Any) -> str:
    return _required_digest(getattr(corpus, "digest"), "corpus_digest")


def _episodes_for_split(corpus: Any, split: str) -> tuple[Any, ...]:
    episodes = tuple(corpus.for_split(split))
    if not episodes:
        raise ValueError(f"corpus has no episodes in {split!r} split")
    return episodes


@contextmanager
def _preserve_model(model: Any) -> Iterator[str]:
    baseline = model.checkpoint()
    baseline_digest = content_digest(baseline)
    try:
        yield baseline_digest
    finally:
        model.restore(baseline)
        restored_digest = content_digest(model.checkpoint())
        if restored_digest != baseline_digest:
            raise RuntimeError("read-only response-plan teacher failed checkpoint restoration")


class ResponsePlanTargetEncoder:
    """Frozen, content-addressed H3.6 response-plan target encoder."""

    def __init__(
        self,
        *,
        width: int,
        model_context_dim: int,
        corpus_digest: str,
        parent_checkpoint_digest: str,
        fit_episode_ids: Sequence[str],
        effective_rank: int,
        native_mean: torch.Tensor,
        native_basis: torch.Tensor,
        native_scale: torch.Tensor,
    ) -> None:
        self.width = int(width)
        self.model_context_dim = int(model_context_dim)
        self.corpus_digest = _required_digest(corpus_digest, "corpus_digest")
        self.parent_checkpoint_digest = _required_digest(
            parent_checkpoint_digest, "parent_checkpoint_digest"
        )
        if isinstance(fit_episode_ids, (str, bytes)) or not isinstance(fit_episode_ids, Sequence):
            raise TypeError("fit_episode_ids must be a sequence of episode ids")
        self.fit_episode_ids = tuple(str(item) for item in fit_episode_ids)
        if not self.fit_episode_ids or any(not item for item in self.fit_episode_ids):
            raise ValueError("fit_episode_ids cannot be empty")
        if len(set(self.fit_episode_ids)) != len(self.fit_episode_ids):
            raise ValueError("fit_episode_ids must be unique")
        self.effective_rank = int(effective_rank)
        if self.width <= 0:
            raise ValueError("width must be positive")
        if self.model_context_dim <= 0:
            raise ValueError("model_context_dim must be positive")
        if not 1 <= self.effective_rank <= min(self.width, self.model_context_dim):
            raise ValueError("effective_rank is outside the encoder dimensions")

        self.native_mean = native_mean.detach().cpu().to(dtype=torch.float32).contiguous()
        self.native_basis = native_basis.detach().cpu().to(dtype=torch.float32).contiguous()
        self.native_scale = native_scale.detach().cpu().to(dtype=torch.float32).contiguous()
        if self.native_mean.shape != (self.model_context_dim,):
            raise ValueError("native_mean shape does not match model_context_dim")
        if self.native_basis.shape != (self.model_context_dim, self.effective_rank):
            raise ValueError("native_basis shape does not match encoder dimensions")
        if self.native_scale.shape != (self.effective_rank,):
            raise ValueError("native_scale shape does not match effective_rank")
        for name, value in (
            ("native_mean", self.native_mean),
            ("native_basis", self.native_basis),
            ("native_scale", self.native_scale),
        ):
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} contains non-finite values")
        if bool((self.native_scale <= 0).any()):
            raise ValueError("native_scale must be strictly positive")

    @classmethod
    def fit(
        cls,
        model: Any,
        corpus: Any,
        *,
        width: int = RESPONSE_PLAN_TARGET_WIDTH,
        parent_checkpoint_digest: str | None = None,
    ) -> ResponsePlanTargetEncoder:
        """Fit native whitening on train only and restore the teacher exactly."""

        if int(width) <= 0:
            raise ValueError("width must be positive")
        corpus_digest = _corpus_digest(corpus)
        train = _episodes_for_split(corpus, RESPONSE_PLAN_TARGET_FIT_SPLIT)
        raw: list[torch.Tensor] = []
        with _preserve_model(model) as actual_parent_digest:
            if parent_checkpoint_digest is not None:
                expected_parent_digest = _required_digest(
                    parent_checkpoint_digest, "parent_checkpoint_digest"
                )
                if expected_parent_digest != actual_parent_digest:
                    raise ValueError(
                        "parent_checkpoint_digest does not match the current teacher checkpoint"
                    )
            for episode in train:
                raw.append(_native_teacher_raw(model, episode))

        matrix = torch.stack(raw).to(dtype=torch.float32)
        context_dim = int(matrix.shape[1])
        mean = matrix.mean(dim=0)
        centered = matrix - mean
        covariance = centered.T @ centered / max(1, matrix.shape[0] - 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        order = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        effective_rank = min(
            int(width),
            max(1, int((eigenvalues > 1e-6).sum().item())),
        )
        basis = eigenvectors[:, :effective_rank]
        scale = torch.sqrt(eigenvalues[:effective_rank].clamp_min(1e-6))
        encoder = cls(
            width=int(width),
            model_context_dim=context_dim,
            corpus_digest=corpus_digest,
            parent_checkpoint_digest=actual_parent_digest,
            fit_episode_ids=tuple(str(item.episode_id) for item in train),
            effective_rank=effective_rank,
            native_mean=mean,
            native_basis=basis,
            native_scale=scale,
        )
        # Fail at fit time if a training example would produce an unusable
        # native component.  This keeps invalid target geometry out of a run.
        for episode, native in zip(train, raw):
            encoder.apply_native_state(native, episode)
        return encoder

    def assert_compatible(
        self,
        *,
        corpus_digest: str | None = None,
        parent_checkpoint_digest: str | None = None,
    ) -> None:
        if (
            corpus_digest is not None
            and _required_digest(corpus_digest, "corpus_digest") != self.corpus_digest
        ):
            raise ValueError("response-plan target corpus digest is incompatible")
        if (
            parent_checkpoint_digest is not None
            and _required_digest(parent_checkpoint_digest, "parent_checkpoint_digest")
            != self.parent_checkpoint_digest
        ):
            raise ValueError("response-plan target parent checkpoint is incompatible")

    def _native_component(self, native_state: torch.Tensor) -> torch.Tensor:
        raw = native_state.detach().cpu().to(dtype=torch.float32).contiguous()
        if raw.shape != (self.model_context_dim,):
            raise ValueError("native response state dimension does not match encoder")
        coordinates = ((raw - self.native_mean) @ self.native_basis) / self.native_scale
        padded = torch.zeros(self.width, dtype=torch.float32)
        padded[: self.effective_rank] = coordinates
        return _normalize(padded, name="native whitened response state")

    def apply_native_state(self, native_state: torch.Tensor, episode: Any) -> torch.Tensor:
        """Apply frozen native whitening and compose it with the text geometry."""

        native = self._native_component(native_state)
        compositional = _compositional_char_ngram(episode, width=self.width)
        return _normalize(native + compositional, name="response-plan target")

    def encode_episode(self, model: Any, episode: Any) -> torch.Tensor:
        """Encode one episode using the bound parent model as a read-only teacher."""

        with _preserve_model(model) as current_digest:
            self.assert_compatible(parent_checkpoint_digest=current_digest)
            native = _native_teacher_raw(model, episode)
            return self.apply_native_state(native, episode)

    def encode_corpus(self, model: Any, corpus: Any) -> dict[str, torch.Tensor]:
        """Encode a corpus while keeping the teacher checkpoint unchanged."""

        self.assert_compatible(corpus_digest=_corpus_digest(corpus))
        with _preserve_model(model) as current_digest:
            self.assert_compatible(parent_checkpoint_digest=current_digest)
            result: dict[str, torch.Tensor] = {}
            for episode in tuple(corpus.episodes):
                native = _native_teacher_raw(model, episode)
                result[str(episode.episode_id)] = self.apply_native_state(native, episode)
            return result

    @property
    def target_digest(self) -> str:
        return content_digest(self.to_payload())

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": RESPONSE_PLAN_TARGET_FORMAT,
            "version": RESPONSE_PLAN_TARGET_VERSION,
            "width": self.width,
            "model_context_dim": self.model_context_dim,
            "corpus_digest": self.corpus_digest,
            "parent_checkpoint_digest": self.parent_checkpoint_digest,
            "fitted_split": RESPONSE_PLAN_TARGET_FIT_SPLIT,
            "fit_episode_ids": list(self.fit_episode_ids),
            "effective_rank": self.effective_rank,
            "native_source": RESPONSE_PLAN_TARGET_NATIVE_SOURCE,
            "native_transform": RESPONSE_PLAN_TARGET_NATIVE_TRANSFORM,
            "composition": RESPONSE_PLAN_TARGET_COMPOSITION,
            "ngram_salt": RESPONSE_PLAN_TARGET_NGRAM_SALT,
            "native_mean": self.native_mean.detach().cpu().clone(),
            "native_basis": self.native_basis.detach().cpu().clone(),
            "native_scale": self.native_scale.detach().cpu().clone(),
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        expected_corpus_digest: str | None = None,
        expected_parent_checkpoint_digest: str | None = None,
    ) -> ResponsePlanTargetEncoder:
        if not isinstance(payload, Mapping):
            raise TypeError("response-plan target payload must be a mapping")
        if payload.get("format") != RESPONSE_PLAN_TARGET_FORMAT:
            raise ValueError("unsupported response-plan target format")
        if int(payload.get("version", -1)) != RESPONSE_PLAN_TARGET_VERSION:
            raise ValueError("unsupported response-plan target version")
        if payload.get("fitted_split") != RESPONSE_PLAN_TARGET_FIT_SPLIT:
            raise ValueError("response-plan target must be fitted on the train split")
        if payload.get("native_source") != RESPONSE_PLAN_TARGET_NATIVE_SOURCE:
            raise ValueError("response-plan target native source is incompatible")
        if payload.get("native_transform") != RESPONSE_PLAN_TARGET_NATIVE_TRANSFORM:
            raise ValueError("response-plan target native transform is incompatible")
        if payload.get("composition") != RESPONSE_PLAN_TARGET_COMPOSITION:
            raise ValueError("response-plan target composition is incompatible")
        if payload.get("ngram_salt") != RESPONSE_PLAN_TARGET_NGRAM_SALT:
            raise ValueError("response-plan target n-gram salt is incompatible")
        encoder = cls(
            width=int(payload["width"]),
            model_context_dim=int(payload["model_context_dim"]),
            corpus_digest=payload["corpus_digest"],
            parent_checkpoint_digest=payload["parent_checkpoint_digest"],
            fit_episode_ids=payload["fit_episode_ids"],
            effective_rank=int(payload["effective_rank"]),
            native_mean=payload["native_mean"],
            native_basis=payload["native_basis"],
            native_scale=payload["native_scale"],
        )
        encoder.assert_compatible(
            corpus_digest=expected_corpus_digest,
            parent_checkpoint_digest=expected_parent_checkpoint_digest,
        )
        return encoder


def _count_sketch_chunk(chunk: bytes, *, slot: int, width: int) -> torch.Tensor:
    """Build a deterministic additive sketch for one response chunk.

    Unlike H3.6's dense signed projection, each n-gram contributes to one
    bucket.  This keeps shared response prefixes and local fragments in a
    factorized slot instead of mixing every fragment into one global vector.
    The sketch is a training target only; no runtime decoder can call it.
    """

    if int(width) <= 0:
        raise ValueError("factorized target width must be positive")
    value = bytes(chunk) or b"<empty>"
    result = torch.zeros(int(width), dtype=torch.float32)
    for size, weight in ((1, 1.0), (2, 1.5), (3, 2.0)):
        count = max(1, len(value) - size + 1)
        for index in range(count):
            token = bytes((int(slot) & 0xFF, int(size) & 0xFF)) + value[
                index : index + size
            ]
            digest = hashlib.sha256(FACTOR_RESPONSE_PLAN_TARGET_SALT + token).digest()
            bucket = int.from_bytes(digest[:4], "little") % int(width)
            sign = 1.0 if digest[4] & 1 else -1.0
            result[bucket] += float(weight) * sign
    if not bool(torch.isfinite(result).all()) or not bool(result.abs().any()):
        raise ValueError("factorized target sketch is zero or non-finite")
    return result


def _response_chunks(
    response: str,
    *,
    slots: int,
    phase_stride: int,
) -> tuple[bytes, ...]:
    """Split a response at UTF-8 codepoint boundaries for ordered plan slots."""

    if not isinstance(response, str) or not response:
        raise ValueError("factorized response target requires a non-empty response")
    if int(slots) <= 0 or int(phase_stride) <= 0:
        raise ValueError("factorized response target slots and stride must be positive")
    raw = response.encode("utf-8")
    chunks: list[bytes] = []
    cursor = 0
    for slot in range(int(slots)):
        if slot == int(slots) - 1:
            end = len(raw)
        else:
            end = min(len(raw), cursor + int(phase_stride))
            while end > cursor:
                try:
                    raw[cursor:end].decode("utf-8")
                    break
                except UnicodeDecodeError:
                    end -= 1
        chunks.append(raw[cursor:end])
        cursor = end
    if cursor < len(raw):
        chunks[-1] += raw[cursor:]
    return tuple(chunks)


class FactorizedResponsePlanTargetEncoder:
    """Train-bound ordered response-chunk target geometry for H3.7."""

    def __init__(
        self,
        *,
        slots: int,
        slot_width: int,
        phase_stride: int,
        corpus_digest: str,
        parent_checkpoint_digest: str,
        fit_episode_ids: Sequence[str],
        slot_scale: torch.Tensor,
    ) -> None:
        self.slots = int(slots)
        self.slot_width = int(slot_width)
        self.phase_stride = int(phase_stride)
        self.width = self.slots * self.slot_width
        self.corpus_digest = _required_digest(corpus_digest, "corpus_digest")
        self.parent_checkpoint_digest = _required_digest(
            parent_checkpoint_digest, "parent_checkpoint_digest"
        )
        if self.slots <= 0 or self.slot_width <= 0 or self.phase_stride <= 0:
            raise ValueError("factorized target dimensions and stride must be positive")
        if isinstance(fit_episode_ids, (str, bytes)) or not isinstance(
            fit_episode_ids, Sequence
        ):
            raise TypeError("fit_episode_ids must be a sequence of episode ids")
        self.fit_episode_ids = tuple(str(item) for item in fit_episode_ids)
        if not self.fit_episode_ids or any(not item for item in self.fit_episode_ids):
            raise ValueError("fit_episode_ids cannot be empty")
        if len(set(self.fit_episode_ids)) != len(self.fit_episode_ids):
            raise ValueError("fit_episode_ids must be unique")
        self.slot_scale = slot_scale.detach().cpu().to(dtype=torch.float32).contiguous()
        if self.slot_scale.shape != (self.slots,):
            raise ValueError("factorized target slot_scale shape is invalid")
        if not bool(torch.isfinite(self.slot_scale).all()) or bool(
            (self.slot_scale <= 0).any()
        ):
            raise ValueError("factorized target slot_scale must be finite and positive")

    @classmethod
    def fit(
        cls,
        model: Any,
        corpus: Any,
        *,
        slots: int = FACTOR_RESPONSE_PLAN_TARGET_SLOTS,
        slot_width: int = FACTOR_RESPONSE_PLAN_TARGET_SLOT_WIDTH,
        phase_stride: int = FACTOR_RESPONSE_PLAN_TARGET_PHASE_STRIDE,
        parent_checkpoint_digest: str | None = None,
    ) -> FactorizedResponsePlanTargetEncoder:
        """Fit only per-slot scale on train responses and bind the parent."""

        corpus_digest = _corpus_digest(corpus)
        train = _episodes_for_split(corpus, RESPONSE_PLAN_TARGET_FIT_SPLIT)
        actual_parent_digest = content_digest(model.checkpoint())
        if parent_checkpoint_digest is not None:
            expected = _required_digest(parent_checkpoint_digest, "parent_checkpoint_digest")
            if expected != actual_parent_digest:
                raise ValueError(
                    "parent_checkpoint_digest does not match the current teacher checkpoint"
                )
        raw: list[torch.Tensor] = []
        for episode in train:
            chunks = _response_chunks(
                str(episode.response), slots=int(slots), phase_stride=int(phase_stride)
            )
            raw.append(
                torch.stack(
                    [
                        _count_sketch_chunk(
                            chunk, slot=index, width=int(slot_width)
                        )
                        for index, chunk in enumerate(chunks)
                    ]
                )
            )
        matrix = torch.stack(raw)
        scale = torch.sqrt(matrix.square().mean(dim=(0, 2)).clamp_min(1e-6))
        encoder = cls(
            slots=int(slots),
            slot_width=int(slot_width),
            phase_stride=int(phase_stride),
            corpus_digest=corpus_digest,
            parent_checkpoint_digest=actual_parent_digest,
            fit_episode_ids=tuple(str(item.episode_id) for item in train),
            slot_scale=scale,
        )
        for episode in train:
            encoder.encode_response(str(episode.response))
        return encoder

    def assert_compatible(
        self,
        *,
        corpus_digest: str | None = None,
        parent_checkpoint_digest: str | None = None,
    ) -> None:
        if (
            corpus_digest is not None
            and _required_digest(corpus_digest, "corpus_digest") != self.corpus_digest
        ):
            raise ValueError("factorized response target corpus digest is incompatible")
        if (
            parent_checkpoint_digest is not None
            and _required_digest(parent_checkpoint_digest, "parent_checkpoint_digest")
            != self.parent_checkpoint_digest
        ):
            raise ValueError("factorized response target parent checkpoint is incompatible")

    def encode_response(self, response: str) -> torch.Tensor:
        chunks = _response_chunks(
            response, slots=self.slots, phase_stride=self.phase_stride
        )
        slots: list[torch.Tensor] = []
        for index, chunk in enumerate(chunks):
            value = _count_sketch_chunk(chunk, slot=index, width=self.slot_width)
            value = value / self.slot_scale[index]
            slots.append(_normalize(value, name=f"factorized response target slot {index}"))
        return torch.cat(slots, dim=0)

    def encode_episode(self, model: Any, episode: Any) -> torch.Tensor:
        self.assert_compatible(parent_checkpoint_digest=content_digest(model.checkpoint()))
        return self.encode_response(str(episode.response))

    def encode_corpus(self, model: Any, corpus: Any) -> dict[str, torch.Tensor]:
        self.assert_compatible(corpus_digest=_corpus_digest(corpus))
        self.assert_compatible(parent_checkpoint_digest=content_digest(model.checkpoint()))
        return {
            str(episode.episode_id): self.encode_response(str(episode.response))
            for episode in tuple(corpus.episodes)
        }

    @property
    def target_digest(self) -> str:
        return content_digest(self.to_payload())

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": FACTOR_RESPONSE_PLAN_TARGET_FORMAT,
            "version": FACTOR_RESPONSE_PLAN_TARGET_VERSION,
            "slots": self.slots,
            "slot_width": self.slot_width,
            "width": self.width,
            "phase_stride": self.phase_stride,
            "corpus_digest": self.corpus_digest,
            "parent_checkpoint_digest": self.parent_checkpoint_digest,
            "fitted_split": RESPONSE_PLAN_TARGET_FIT_SPLIT,
            "fit_episode_ids": list(self.fit_episode_ids),
            "projection": FACTOR_RESPONSE_PLAN_TARGET_PROJECTION,
            "normalization": FACTOR_RESPONSE_PLAN_TARGET_NORMALIZATION,
            "salt": FACTOR_RESPONSE_PLAN_TARGET_SALT,
            "slot_scale": self.slot_scale.detach().cpu().clone(),
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        expected_corpus_digest: str | None = None,
        expected_parent_checkpoint_digest: str | None = None,
    ) -> FactorizedResponsePlanTargetEncoder:
        if not isinstance(payload, Mapping):
            raise TypeError("factorized response target payload must be a mapping")
        if payload.get("format") != FACTOR_RESPONSE_PLAN_TARGET_FORMAT:
            raise ValueError("unsupported factorized response target format")
        if int(payload.get("version", -1)) != FACTOR_RESPONSE_PLAN_TARGET_VERSION:
            raise ValueError("unsupported factorized response target version")
        if payload.get("fitted_split") != RESPONSE_PLAN_TARGET_FIT_SPLIT:
            raise ValueError("factorized response target must be fitted on train")
        if payload.get("projection") != FACTOR_RESPONSE_PLAN_TARGET_PROJECTION:
            raise ValueError("factorized response target projection is incompatible")
        if payload.get("normalization") != FACTOR_RESPONSE_PLAN_TARGET_NORMALIZATION:
            raise ValueError("factorized response target normalization is incompatible")
        if payload.get("salt") != FACTOR_RESPONSE_PLAN_TARGET_SALT:
            raise ValueError("factorized response target salt is incompatible")
        encoder = cls(
            slots=int(payload["slots"]),
            slot_width=int(payload["slot_width"]),
            phase_stride=int(payload["phase_stride"]),
            corpus_digest=payload["corpus_digest"],
            parent_checkpoint_digest=payload["parent_checkpoint_digest"],
            fit_episode_ids=payload["fit_episode_ids"],
            slot_scale=payload["slot_scale"],
        )
        if int(payload.get("width", -1)) != encoder.width:
            raise ValueError("factorized response target width is inconsistent")
        encoder.assert_compatible(
            corpus_digest=expected_corpus_digest,
            parent_checkpoint_digest=expected_parent_checkpoint_digest,
        )
        return encoder


__all__ = [
    "RESPONSE_PLAN_TARGET_COMPOSITION",
    "RESPONSE_PLAN_TARGET_FIT_SPLIT",
    "RESPONSE_PLAN_TARGET_FORMAT",
    "RESPONSE_PLAN_TARGET_NATIVE_SOURCE",
    "RESPONSE_PLAN_TARGET_NATIVE_TRANSFORM",
    "RESPONSE_PLAN_TARGET_NGRAM_SALT",
    "RESPONSE_PLAN_TARGET_VERSION",
    "RESPONSE_PLAN_TARGET_WIDTH",
    "FACTOR_RESPONSE_PLAN_TARGET_FORMAT",
    "FACTOR_RESPONSE_PLAN_TARGET_NORMALIZATION",
    "FACTOR_RESPONSE_PLAN_TARGET_PHASE_STRIDE",
    "FACTOR_RESPONSE_PLAN_TARGET_PROJECTION",
    "FACTOR_RESPONSE_PLAN_TARGET_SALT",
    "FACTOR_RESPONSE_PLAN_TARGET_SLOT_WIDTH",
    "FACTOR_RESPONSE_PLAN_TARGET_SLOTS",
    "FACTOR_RESPONSE_PLAN_TARGET_VERSION",
    "FactorizedResponsePlanTargetEncoder",
    "ResponsePlanTargetEncoder",
]
