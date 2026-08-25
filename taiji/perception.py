"""Learned local perception and variable-duration assemblies for Taiji P2."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from .config import TaijiConfig
from .contracts import PerceptEvent

PERCEPTION_STATE_VERSION = 2


class LearnedPerception(nn.Module):
    """A small learned perceptual hierarchy with an adaptive assembly clock.

    The module keeps the byte codec at the organ boundary, then maps bytes to
    continuous local features.  A recurrent predictive projection supplies a
    local error signal; change and error jointly decide when an assembly ends.
    No vocabulary, token table or Transformer block is involved.
    """

    def __init__(self, config: TaijiConfig, *, device: torch.device | str = "cpu") -> None:
        super().__init__()
        self.architecture_config = config
        self.config = config.perception
        self.alphabet_size = int(config.alphabet_size)
        self.device = torch.device(device)
        feature_dim = int(self.config.feature_dim)
        self.embedding = nn.Embedding(self.alphabet_size, feature_dim)
        self.local_projection = nn.Linear(feature_dim * int(self.config.local_window), feature_dim)
        self.context_projection = nn.Linear(feature_dim, feature_dim, bias=False)
        self.transition = nn.Linear(feature_dim, feature_dim, bias=False)
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(int(config.seed + self.config.seed_offset) % (2**63 - 1))
        self._initialize_parameters()
        self.to(self.device)
        self.reset_dynamics()

    def _initialize_parameters(self) -> None:
        scale = 1.0 / math.sqrt(float(self.config.feature_dim))
        with torch.no_grad():
            for parameter in (
                self.embedding.weight,
                self.local_projection.weight,
                self.context_projection.weight,
                self.transition.weight,
            ):
                values = torch.randn(
                    parameter.shape,
                    generator=self._rng,
                    dtype=parameter.dtype,
                )
                parameter.copy_(values * scale)
            self.local_projection.bias.zero_()

    @property
    def feature_dim(self) -> int:
        return int(self.config.feature_dim)

    def reset_dynamics(self) -> None:
        """Clear temporal activity while preserving learned parameters."""

        self._previous_feature = torch.zeros(self.feature_dim, device=self.device)
        self._recurrent_state = torch.zeros(self.feature_dim, device=self.device)
        self._assembly_sum = torch.zeros(self.feature_dim, device=self.device)
        self._assembly_duration = 0
        self._assembly_index = 0
        self._history: list[int] = []
        self._last_prediction_error = 0.0
        self._surprise_baseline = 0.0
        self._boundary_threshold_state = float(self.config.boundary_threshold)

    def _local_feature(self, symbol: int) -> torch.Tensor:
        window = self._history[-(int(self.config.local_window) - 1) :]
        window = [*window, int(symbol)]
        missing = int(self.config.local_window) - len(window)
        vectors = [
            torch.zeros(self.feature_dim, device=self.device) for _ in range(max(0, missing))
        ]
        vectors.extend(self.embedding.weight[index] for index in window)
        local = self.local_projection(torch.cat(vectors, dim=0))
        context = self.context_projection(self._recurrent_state)
        return torch.nn.functional.normalize(torch.tanh(local + context), dim=0)

    def _training_feature(
        self,
        symbol: int,
        history: Sequence[int],
        recurrent_state: torch.Tensor,
    ) -> torch.Tensor:
        """Build a differentiable local feature for predictive fitting."""

        window = list(history[-(int(self.config.local_window) - 1) :])
        window.append(int(symbol))
        missing = int(self.config.local_window) - len(window)
        vectors = [
            torch.zeros(self.feature_dim, device=self.device) for _ in range(max(0, missing))
        ]
        vectors.extend(self.embedding.weight[index] for index in window)
        local = self.local_projection(torch.cat(vectors, dim=0))
        context = self.context_projection(recurrent_state)
        return torch.nn.functional.normalize(torch.tanh(local + context), dim=0)

    def fit_predictive(
        self,
        sequences: Sequence[Sequence[int]],
        *,
        epochs: int = 1,
        learning_rate: float | None = None,
        temperature: float = 0.15,
        assembly_prediction_weight: float = 0.5,
        contrastive_weight: float = 0.1,
        contrastive_temperature: float = 0.2,
    ) -> list[float]:
        """Fit local features against the next-symbol predictive objective.

        The target is the next observed symbol's learned embedding, not a
        hand-authored vocabulary or a fixed assembly table.  Cosine-style
        logits through ``transition`` make local context and recurrent state
        useful only when they improve prediction of the next observation.
        """

        if int(epochs) <= 0:
            raise ValueError("predictive epochs must be positive")
        rate = float(self.config.learning_rate if learning_rate is None else learning_rate)
        if rate <= 0.0:
            raise ValueError("predictive learning_rate must be positive")
        if float(temperature) <= 0.0:
            raise ValueError("predictive temperature must be positive")
        if float(assembly_prediction_weight) < 0.0:
            raise ValueError("assembly_prediction_weight cannot be negative")
        if float(contrastive_weight) < 0.0:
            raise ValueError("contrastive_weight cannot be negative")
        if float(contrastive_temperature) <= 0.0:
            raise ValueError("contrastive temperature must be positive")
        training_sequences = tuple(
            tuple(int(symbol) for symbol in sequence) for sequence in sequences
        )
        if not training_sequences or any(len(sequence) < 2 for sequence in training_sequences):
            raise ValueError(
                "predictive fitting requires non-empty sequences of at least two symbols"
            )
        if any(
            any(not 0 <= symbol < self.alphabet_size for symbol in sequence)
            for sequence in training_sequences
        ):
            raise ValueError("predictive training symbol is outside the configured alphabet")

        optimizer = torch.optim.Adam(self.parameters(), lr=rate)
        losses: list[float] = []
        self.train()
        for _ in range(int(epochs)):
            epoch_losses: list[float] = []
            for sequence in training_sequences:
                history: list[int] = []
                recurrent_state = torch.zeros(self.feature_dim, device=self.device)
                features: list[torch.Tensor] = []
                for symbol in sequence:
                    feature = self._training_feature(symbol, history, recurrent_state)
                    features.append(feature)
                    history.append(symbol)
                    history = history[-int(self.config.local_window) :]
                    recurrent_state = feature
                pool_window = min(
                    int(self.config.local_window),
                    int(self.config.maximum_assembly_duration),
                )
                pooled = torch.stack(
                    [
                        torch.stack(features[max(0, index + 1 - pool_window) : index + 1]).mean(
                            dim=0
                        )
                        for index in range(len(features) - 1)
                    ]
                )
                predicted = torch.stack([self.transition(feature) for feature in pooled])
                target = torch.tensor(sequence[1:], dtype=torch.long, device=self.device)
                logits = predicted @ self.embedding.weight.T
                next_symbol_loss = torch.nn.functional.cross_entropy(
                    logits / float(temperature), target
                )
                future_targets = torch.stack(
                    [
                        self.embedding.weight[
                            list(sequence[index + 1 : index + 1 + pool_window])
                        ].mean(dim=0)
                        for index in range(len(features) - 1)
                    ]
                )
                assembly_loss = (
                    1.0
                    - torch.nn.functional.cosine_similarity(predicted, future_targets, dim=1).mean()
                )
                short_window = max(1, pool_window - 1)
                positive = torch.stack(
                    [
                        torch.stack(features[max(0, index + 1 - short_window) : index + 1]).mean(
                            dim=0
                        )
                        for index in range(len(features) - 1)
                    ]
                )
                anchor = torch.nn.functional.normalize(pooled, dim=1)
                positive = torch.nn.functional.normalize(positive, dim=1)
                negative = torch.nn.functional.normalize(pooled.roll(1, dims=0), dim=1)
                positive_logits = (anchor * positive).sum(dim=1, keepdim=True)
                negative_logits = anchor @ negative.T
                contrastive_logits = torch.cat((positive_logits, negative_logits), dim=1)
                contrastive_targets = torch.zeros(
                    contrastive_logits.shape[0], dtype=torch.long, device=self.device
                )
                contrastive_loss = torch.nn.functional.cross_entropy(
                    contrastive_logits / float(contrastive_temperature), contrastive_targets
                )
                loss = (
                    next_symbol_loss
                    + float(assembly_prediction_weight) * assembly_loss
                    + float(contrastive_weight) * contrastive_loss
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))
            losses.append(sum(epoch_losses) / len(epoch_losses))
        self.eval()
        self.reset_dynamics()
        return losses

    @torch.no_grad()
    def observe(
        self,
        symbol: int,
        *,
        tick: int,
        stream_id: str,
        learn: bool = True,
    ) -> PerceptEvent:
        """Encode one raw symbol and emit the current variable-duration assembly."""

        symbol = int(symbol)
        if not 0 <= symbol < self.alphabet_size:
            raise ValueError("perception symbol is outside the configured alphabet")
        if int(tick) < 0:
            raise ValueError("perception tick cannot be negative")
        if not stream_id:
            raise ValueError("perception stream_id cannot be empty")

        feature = self._local_feature(symbol)
        has_previous = bool(self._history)
        predicted = self.transition(self._previous_feature)
        if has_previous:
            feature_prediction_error = float(
                (feature - predicted).norm().item() / (feature.norm().item() + 1e-6)
            )
            change = float(
                0.5
                * (
                    1.0
                    - torch.nn.functional.cosine_similarity(
                        feature.unsqueeze(0), self._previous_feature.unsqueeze(0)
                    ).item()
                )
            )
            logits = predicted @ self.embedding.weight.T
            symbol_probability = float(torch.nn.functional.softmax(logits, dim=0)[symbol].item())
            prediction_error = max(feature_prediction_error, 1.0 - symbol_probability)
        else:
            prediction_error = 0.0
            change = 0.0
        surprise = max(0.0, min(1.0, prediction_error / 2.0))
        baseline = float(self._surprise_baseline)
        calibrated_surprise = max(0.0, min(1.0, (surprise - baseline) / max(1e-6, 1.0 - baseline)))
        boundary_score = max(
            0.0,
            min(
                1.0,
                float(self.config.change_gain) * change
                + float(self.config.surprise_gain) * calibrated_surprise,
            ),
        )

        self._assembly_sum.add_(feature)
        self._assembly_duration += 1
        duration = int(self._assembly_duration)
        reaches_maximum = duration >= int(self.config.maximum_assembly_duration)
        reaches_signal = duration >= int(
            self.config.minimum_assembly_duration
        ) and boundary_score >= float(self._boundary_threshold_state)
        boundary = bool(reaches_maximum or reaches_signal)
        pooled = self._assembly_sum / float(duration)
        event = PerceptEvent(
            event_id=f"{stream_id}:assembly:{self._assembly_index}",
            assembly_id=f"{stream_id}:assembly:{self._assembly_index}",
            observation_tick=int(tick),
            modality="text-byte",
            features=pooled.detach().clone(),
            duration=duration,
            boundary_score=boundary_score,
            prediction_error=surprise,
            boundary=boundary,
            confidence=1.0 - surprise,
        )

        if learn and has_previous:
            error = feature - predicted
            update = float(self.config.learning_rate) * torch.outer(error, self._previous_feature)
            self.transition.weight.add_(update.clamp(-0.05, 0.05))
            self.embedding.weight[symbol].lerp_(feature, float(self.config.learning_rate))
        if has_previous:
            baseline_rate = float(self.config.surprise_baseline_rate)
            self._surprise_baseline = (1.0 - baseline_rate) * baseline + baseline_rate * surprise
        if boundary:
            hysteresis = float(self.config.boundary_hysteresis)
            self._boundary_threshold_state = min(
                1.0,
                float(self._boundary_threshold_state)
                + hysteresis * (1.0 - float(self._boundary_threshold_state)),
            )
        else:
            self._boundary_threshold_state = max(
                float(self.config.boundary_threshold),
                float(self._boundary_threshold_state) - float(self.config.boundary_hysteresis),
            )
        self._previous_feature.copy_(feature)
        self._recurrent_state.copy_(feature)
        self._history.append(symbol)
        self._history = self._history[-int(self.config.local_window) :]
        self._last_prediction_error = surprise
        if boundary:
            self._assembly_sum.zero_()
            self._assembly_duration = 0
            self._assembly_index += 1
        return event

    def parameter_tensors(self) -> tuple[torch.Tensor, ...]:
        return tuple(parameter.detach() for parameter in self.parameters())

    def checkpoint(self) -> dict[str, Any]:
        return {
            "format": "taiji-perception-v1",
            "version": PERCEPTION_STATE_VERSION,
            "state_dict": {
                name: value.detach().cpu().clone() for name, value in self.state_dict().items()
            },
            "dynamic": {
                "previous_feature": self._previous_feature.detach().cpu().clone(),
                "recurrent_state": self._recurrent_state.detach().cpu().clone(),
                "assembly_sum": self._assembly_sum.detach().cpu().clone(),
                "assembly_duration": self._assembly_duration,
                "assembly_index": self._assembly_index,
                "history": list(self._history),
                "last_prediction_error": self._last_prediction_error,
                "surprise_baseline": self._surprise_baseline,
                "boundary_threshold_state": self._boundary_threshold_state,
            },
            "rng_state": self._rng.get_state().clone(),
        }

    def restore(self, payload: Mapping[str, Any]) -> None:
        if payload.get("format") != "taiji-perception-v1":
            raise ValueError("unsupported perception checkpoint format")
        version = int(payload.get("version"))
        if version not in (1, PERCEPTION_STATE_VERSION):
            raise ValueError("unsupported perception checkpoint version")
        state_dict = {
            name: value.detach().to(self.device).clone()
            for name, value in payload["state_dict"].items()
        }
        self.load_state_dict(state_dict)
        dynamic = payload["dynamic"]
        self._previous_feature = dynamic["previous_feature"].detach().to(self.device).clone()
        self._recurrent_state = dynamic["recurrent_state"].detach().to(self.device).clone()
        self._assembly_sum = dynamic["assembly_sum"].detach().to(self.device).clone()
        self._assembly_duration = int(dynamic["assembly_duration"])
        self._assembly_index = int(dynamic["assembly_index"])
        self._history = [int(symbol) for symbol in dynamic["history"]]
        self._last_prediction_error = float(dynamic["last_prediction_error"])
        self._surprise_baseline = float(dynamic.get("surprise_baseline", 0.0))
        self._boundary_threshold_state = float(
            dynamic.get("boundary_threshold_state", self.config.boundary_threshold)
        )
        self._rng.set_state(payload["rng_state"].detach().cpu())
