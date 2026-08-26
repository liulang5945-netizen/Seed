"""Capacity-limited workspace routing for the Taiji native architecture."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn

from .contracts import WorkspaceCandidate, WorkspaceSelection
from .local_learning import apply_linear_delta, freeze_parameters, logistic_error_delta


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


@dataclass(frozen=True)
class WorkspaceCompositionSample:
    """A holdout sample whose target is composed from multiple candidates."""

    candidates: tuple[WorkspaceCandidate, ...]
    target: torch.Tensor
    relevant_ids: tuple[str, ...]
    tick: int = 0

    def __post_init__(self) -> None:
        candidate_ids = {candidate.candidate_id for candidate in self.candidates}
        if len(candidate_ids) != len(self.candidates):
            raise ValueError("workspace composition candidate ids must be unique")
        if not set(self.relevant_ids).issubset(candidate_ids):
            raise ValueError("workspace composition relevance contains an unknown candidate")
        if not self.relevant_ids:
            raise ValueError("workspace composition needs at least one relevant candidate")
        if self.target.ndim != 1:
            raise ValueError("workspace composition target must be a vector")
        if self.tick < 0:
            raise ValueError("workspace composition tick cannot be negative")


class WorkspaceCollaborationEvaluator:
    """Evaluate whether selective routing preserves a multi-source composition.

    The evaluator scores a fixed target reconstruction contract rather than a
    trainable answer head.  This keeps the result about workspace information
    selection: the target is the mean content of the registered relevant
    candidates, while distractors add independent content noise.
    """

    def __init__(self, *, content_dim: int, seeds: tuple[int, ...] = (11, 29, 47)) -> None:
        if int(content_dim) <= 0:
            raise ValueError("workspace content_dim must be positive")
        if not seeds:
            raise ValueError("workspace collaboration evaluator needs at least one seed")
        self.content_dim = int(content_dim)
        self.seeds = tuple(int(seed) for seed in seeds)

    def _feature_dim(self, samples: tuple[WorkspaceCompositionSample, ...]) -> int:
        if not samples:
            raise ValueError("workspace collaboration evaluation needs samples")
        dimensions = {
            candidate.features.numel() for sample in samples for candidate in sample.candidates
        }
        if len(dimensions) != 1:
            raise ValueError("workspace composition candidate dimensions must be consistent")
        feature_dim = next(iter(dimensions))
        if feature_dim < self.content_dim:
            raise ValueError("workspace feature dimension must cover content_dim")
        return feature_dim

    def _mse(self, value: torch.Tensor, target: torch.Tensor) -> float:
        return float(torch.mean((value[: self.content_dim] - target) ** 2))

    def _mean_content(
        self, candidates: tuple[WorkspaceCandidate, ...], target: torch.Tensor
    ) -> float:
        if not candidates:
            return self._mse(torch.zeros(self.content_dim), target)
        return self._mse(
            torch.stack([candidate.features for candidate in candidates]).mean(dim=0), target
        )

    def evaluate(
        self,
        train: Iterable[WorkspaceCompositionSample],
        holdout: Iterable[WorkspaceCompositionSample],
        *,
        capacity: int,
        epochs: int = 100,
        learning_rate: float = 0.2,
    ) -> dict[str, object]:
        train = tuple(train)
        holdout = tuple(holdout)
        if not train or not holdout:
            raise ValueError("workspace collaboration evaluation needs train and holdout samples")
        if int(capacity) <= 0:
            raise ValueError("workspace collaboration capacity must be positive")
        feature_dim = self._feature_dim(train + holdout)
        if any(len(sample.relevant_ids) > capacity for sample in train + holdout):
            raise ValueError("workspace capacity cannot fit the registered composition")
        seed_reports: list[dict[str, float]] = []
        for seed in self.seeds:
            router = WorkspaceRouter(feature_dim, capacity=capacity, seed=seed)
            router.fit(
                tuple(
                    WorkspaceRoutingExample(
                        candidates=sample.candidates,
                        relevant_ids=sample.relevant_ids,
                        tick=sample.tick,
                    )
                    for sample in train
                ),
                epochs=epochs,
                learning_rate=learning_rate,
            )
            totals: dict[str, float] = {
                "learned": 0.0,
                "random": 0.0,
                "none": 0.0,
                "fixed": 0.0,
                "dense": 0.0,
                "strongest_single": 0.0,
            }
            exact_routes = 0
            for index, sample in enumerate(holdout):
                learned = router.route(sample.candidates, tick=sample.tick, mode="learned")
                random_route = router.route(
                    sample.candidates,
                    tick=sample.tick,
                    mode="random",
                    random_seed=seed + index,
                )
                none_route = router.route(sample.candidates, tick=sample.tick, mode="none")
                totals["learned"] += self._mse(learned.broadcast, sample.target)
                totals["random"] += self._mse(random_route.broadcast, sample.target)
                totals["none"] += self._mse(none_route.broadcast, sample.target)
                fixed_candidates = sample.candidates[:capacity]
                totals["fixed"] += self._mean_content(fixed_candidates, sample.target)
                totals["dense"] += self._mean_content(sample.candidates, sample.target)
                totals["strongest_single"] += min(
                    self._mse(candidate.features, sample.target) for candidate in sample.candidates
                )
                if set(learned.selected_ids) == set(sample.relevant_ids):
                    exact_routes += 1
            count = float(len(holdout))
            metrics: dict[str, float] = {
                name: value / count for name, value in totals.items()
            }
            metrics["learned_gain_vs_strongest_single"] = (
                metrics["strongest_single"] - metrics["learned"]
            )
            metrics["learned_gain_vs_dense"] = metrics["dense"] - metrics["learned"]
            metrics["exact_route_rate"] = exact_routes / count
            metrics["router_fit_updates"] = router.fit_updates
            seed_reports.append(metrics)
        aggregate: dict[str, float] = {
            name: sum(float(report[name]) for report in seed_reports) / len(seed_reports)
            for name in seed_reports[0]
            if name != "router_fit_updates"
        }
        aggregate["learned_gain_vs_strongest_single_min"] = min(
            float(report["learned_gain_vs_strongest_single"]) for report in seed_reports
        )
        aggregate["exact_route_rate_min"] = min(
            float(report["exact_route_rate"]) for report in seed_reports
        )
        passed = bool(
            aggregate["learned_gain_vs_strongest_single_min"] > 0.05
            and aggregate["learned_gain_vs_dense"] > 0.05
            and aggregate["exact_route_rate_min"] >= 0.9
        )
        return {
            "format": "taiji-a3-workspace-composition-v1",
            "content_dim": self.content_dim,
            "capacity": int(capacity),
            "train_samples": len(train),
            "holdout_samples": len(holdout),
            "seeds": seed_reports,
            "aggregate": {**aggregate, "passed": passed},
            "gate": {
                "passed": passed,
                "criterion": "learned workspace beats strongest single and dense mean by >0.05 MSE; exact route rate >= 0.90",
            },
        }


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
            self.scorer.weight.copy_(
                torch.randn(self.scorer.weight.shape, generator=generator) * 0.02
            )
            self.scorer.bias.zero_()
        freeze_parameters(self)

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
        scale = 1.0 / len(examples)
        final_loss = 0.0
        for _ in range(int(epochs)):
            feature_rows: list[torch.Tensor] = []
            error_rows: list[torch.Tensor] = []
            losses: list[float] = []
            for example in examples:
                features = self._features(example.candidates)
                if features.shape[0] == 0:
                    raise ValueError("workspace routing example needs at least one candidate")
                targets = torch.tensor(
                    [
                        candidate.candidate_id in example.relevant_ids
                        for candidate in example.candidates
                    ],
                    dtype=features.dtype,
                    device=features.device,
                ).unsqueeze(-1)
                with torch.no_grad():
                    logits = self.scorer(features)
                    losses.append(
                        float(
                            nn.functional.binary_cross_entropy_with_logits(
                                logits.squeeze(-1), targets.squeeze(-1)
                            )
                        )
                    )
                feature_rows.append(features)
                error_rows.append(logistic_error_delta(logits, targets) * scale)
            final_loss = sum(losses) * scale
            apply_linear_delta(
                self.scorer,
                torch.cat(feature_rows, dim=0),
                torch.cat(error_rows, dim=0),
                float(learning_rate),
            )
        self.fit_updates += 1
        return final_loss

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
            selected_indices = torch.randperm(len(candidates), generator=generator).tolist()[
                : self.capacity
            ]
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
        feature_dim = payload.get("feature_dim")
        capacity = payload.get("capacity")
        state_dict_payload = payload.get("state_dict")
        if not isinstance(feature_dim, int) or not isinstance(capacity, int):
            raise ValueError("workspace checkpoint dimensions must be integers")
        if not isinstance(state_dict_payload, dict):
            raise ValueError("workspace checkpoint state_dict must be a mapping")
        state_dict: dict[str, torch.Tensor] = {}
        for name, value in state_dict_payload.items():
            if not isinstance(name, str) or not isinstance(value, torch.Tensor):
                raise ValueError("workspace checkpoint state_dict must contain tensors")
            state_dict[name] = value
        router = cls(feature_dim, capacity=capacity, seed=0)
        router.load_state_dict(state_dict)
        router.to(device)
        fit_updates = payload.get("fit_updates", 0)
        if not isinstance(fit_updates, int):
            raise ValueError("workspace checkpoint fit_updates must be an integer")
        router.fit_updates = fit_updates
        return router
