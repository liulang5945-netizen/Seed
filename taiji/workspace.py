"""Capacity-limited workspace routing for the Taiji native architecture."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn

from .contracts import WorkspaceCandidate, WorkspaceSelection


@dataclass(frozen=True)
class WorkspaceRoutingExample:
    """Supervision for which candidates deserve shared workspace access."""

    candidates: tuple[WorkspaceCandidate, ...]
    relevant_ids: tuple[str, ...]
    tick: int = 0

    def __post_init__(self) -> None:
        candidate_ids = {candidate.candidate_id for candidate in self.candidates}
        if len(candidate_ids) != len(self.candidates):
            raise ValueError("workspace example candidate ids must be unique")
        if not set(self.relevant_ids).issubset(candidate_ids):
            raise ValueError("workspace example relevance contains an unknown candidate")
        if self.tick < 0:
            raise ValueError("workspace example tick cannot be negative")


class WorkspaceRouter(nn.Module):
    """Learn a scalar access score, then select at most ``capacity`` items.

    This is deliberately a small content-addressed router.  It has no global
    mixing block: each candidate is scored independently and only the selected
    candidates are averaged into the broadcast vector.  ``none`` and
    ``random`` are explicit lesions used to measure whether routing matters.
    """

    def __init__(self, feature_dim: int, *, capacity: int = 4, seed: int = 0) -> None:
        super().__init__()
        if int(feature_dim) <= 0:
            raise ValueError("workspace feature_dim must be positive")
        if int(capacity) <= 0:
            raise ValueError("workspace capacity must be positive")
        self.feature_dim = int(feature_dim)
        self.capacity = int(capacity)
        self.fit_updates = 0
        self.scorer = nn.Linear(self.feature_dim, 1)
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        with torch.no_grad():
            self.scorer.weight.copy_(torch.randn(self.scorer.weight.shape, generator=generator) * 0.02)
            self.scorer.bias.zero_()

    def _features(self, candidates: tuple[WorkspaceCandidate, ...]) -> torch.Tensor:
        if not candidates:
            return torch.empty((0, self.feature_dim), device=self.scorer.weight.device)
        features = torch.stack([candidate.features for candidate in candidates]).to(
            self.scorer.weight.device
        )
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError("workspace candidate feature dimensions must match the router")
        return features

    def fit(
        self,
        examples: Iterable[WorkspaceRoutingExample],
        *,
        epochs: int = 80,
        learning_rate: float = 0.1,
    ) -> float:
        """Fit access scores from relevance labels and return final BCE loss."""

        examples = tuple(examples)
        if not examples:
            raise ValueError("workspace router needs at least one training example")
        if int(epochs) <= 0 or float(learning_rate) <= 0.0:
            raise ValueError("workspace fit epochs and learning_rate must be positive")
        optimizer = torch.optim.SGD(self.parameters(), lr=float(learning_rate))
        loss = torch.tensor(0.0, device=self.scorer.weight.device)
        for _ in range(int(epochs)):
            optimizer.zero_grad()
            losses: list[torch.Tensor] = []
            for example in examples:
                features = self._features(example.candidates)
                targets = torch.tensor(
                    [candidate.candidate_id in example.relevant_ids for candidate in example.candidates],
                    dtype=features.dtype,
                    device=features.device,
                )
                logits = self.scorer(features).squeeze(-1)
                losses.append(nn.functional.binary_cross_entropy_with_logits(logits, targets))
            loss = torch.stack(losses).mean()
            loss.backward()
            optimizer.step()
        self.fit_updates += 1
        return float(loss.detach())

    @torch.no_grad()
    def route(
        self,
        candidates: Iterable[WorkspaceCandidate],
        *,
        tick: int,
        mode: str = "learned",
        random_seed: int | None = None,
    ) -> WorkspaceSelection:
        """Select candidates and produce the only shared workspace broadcast."""

        candidates = tuple(candidates)
        if mode not in {"learned", "none", "random"}:
            raise ValueError(f"unsupported workspace routing mode: {mode}")
        features = self._features(candidates)
        candidate_ids = tuple(candidate.candidate_id for candidate in candidates)
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("workspace candidate ids must be unique")
        if mode == "learned" and candidates:
            scores_tensor = torch.sigmoid(self.scorer(features).squeeze(-1))
            ranked = sorted(
                range(len(candidates)),
                key=lambda index: (-float(scores_tensor[index]), index),
            )
            selected_indices = ranked[: self.capacity]
            scores = tuple(float(score) for score in scores_tensor)
        elif mode == "random" and candidates:
            generator = torch.Generator(device="cpu").manual_seed(
                int(random_seed) if random_seed is not None else 0
            )
            selected_indices = torch.randperm(len(candidates), generator=generator).tolist()[: self.capacity]
            scores = tuple(0.0 for _ in candidates)
        else:
            selected_indices = []
            scores = tuple(0.0 for _ in candidates)
        selected_ids = tuple(candidate_ids[index] for index in selected_indices)
        if selected_indices:
            broadcast = features[selected_indices].mean(dim=0)
        else:
            broadcast = torch.zeros(self.feature_dim, device=features.device)
        return WorkspaceSelection(
            tick=int(tick),
            mode=mode,
            candidate_ids=candidate_ids,
            selected_ids=selected_ids,
            scores=scores,
            broadcast=broadcast.detach().clone(),
            capacity=self.capacity,
        )

    def checkpoint(self) -> dict[str, object]:
        return {
            "feature_dim": self.feature_dim,
            "capacity": self.capacity,
            "fit_updates": self.fit_updates,
            "state_dict": {
                name: tensor.detach().cpu().clone() for name, tensor in self.state_dict().items()
            },
        }

    @classmethod
    def from_checkpoint(
        cls, payload: dict[str, object], *, device: torch.device | str = "cpu"
    ) -> WorkspaceRouter:
        router = cls(int(payload["feature_dim"]), capacity=int(payload["capacity"]), seed=0)
        router.load_state_dict(payload["state_dict"])
        router.to(device)
        router.fit_updates = int(payload.get("fit_updates", 0))
        return router
