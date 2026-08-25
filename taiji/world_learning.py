"""Learned world-dynamics and intervention evaluation for the P3 slice."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .contracts import (
    WorldEpisode,
    WorldEpisodeCorpus,
    WorldInterventionCase,
    WorldInterventionCorpus,
    WorldObject,
    WorldState,
    WorldTransition,
)
from .world import TaijiWorldState, _world_state_equal


def _numeric(value: Any, name: str) -> float:
    if isinstance(value, bool):
        return float(value)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite numeric value")
    return float(value)


@dataclass(frozen=True)
class WorldSchema:
    """A data-derived numeric interface for a world intervention corpus."""

    object_ids: tuple[str, ...]
    state_slots: tuple[tuple[str, str], ...]
    relation_slots: tuple[tuple[str, str, str], ...]
    action_kinds: tuple[str, ...]
    actor_ids: tuple[str, ...]
    target_ids: tuple[str, ...]
    parameter_names: tuple[str, ...]

    @classmethod
    def from_corpus(cls, corpus: WorldInterventionCorpus) -> WorldSchema:
        if not corpus.train:
            raise ValueError("world dynamics requires at least one training case")
        object_ids: set[str] = set()
        state_slots: set[tuple[str, str]] = set()
        relation_slots: set[tuple[str, str, str]] = set()
        action_kinds: set[str] = set()
        actor_ids: set[str] = set()
        target_ids: set[str] = set()
        parameter_names: set[str] = set()
        for case in corpus.train:
            for obj in case.initial.objects:
                object_ids.add(obj.object_id)
                for name, value in obj.attributes:
                    _numeric(value, f"{obj.object_id}.{name}")
                    state_slots.add((obj.object_id, name))
            relation_slots.update(case.initial.relations)
            relation_slots.update(case.expected_state.relations)
            action_kinds.add(case.action.kind)
            actor_ids.add(case.action.actor_id)
            target_ids.add(case.action.target_id)
            for name, value in case.action.parameters:
                try:
                    _numeric(value, f"action parameter {name}")
                except ValueError:
                    continue
                parameter_names.add(name)
        if not state_slots:
            raise ValueError("world dynamics requires numeric object attributes")
        return cls(
            object_ids=tuple(sorted(object_ids)),
            state_slots=tuple(sorted(state_slots)),
            relation_slots=tuple(sorted(relation_slots)),
            action_kinds=tuple(sorted(action_kinds)),
            actor_ids=tuple(sorted(actor_ids)),
            target_ids=tuple(sorted(target_ids)),
            parameter_names=tuple(sorted(parameter_names)),
        )

    @property
    def state_dim(self) -> int:
        return len(self.state_slots) + len(self.relation_slots)

    @property
    def object_state_dim(self) -> int:
        return len(self.state_slots)

    @property
    def action_dim(self) -> int:
        return (
            len(self.action_kinds)
            + len(self.actor_ids)
            + len(self.target_ids)
            + len(self.parameter_names)
            + len(self.target_ids) * len(self.parameter_names)
        )

    @property
    def input_dim(self) -> int:
        return self.state_dim + self.action_dim

    def state_values(self, state: WorldState) -> torch.Tensor:
        objects = {item.object_id: item for item in state.objects}
        values = []
        for object_id, name in self.state_slots:
            if object_id not in objects:
                raise ValueError(f"world state is missing object: {object_id}")
            values.append(_numeric(objects[object_id].attribute(name, 0.0), f"{object_id}.{name}"))
        relation_set = set(state.relations)
        values.extend(float(relation in relation_set) for relation in self.relation_slots)
        return torch.tensor(values, dtype=torch.float32)

    def encode(self, state: WorldState, action: Any, *, bind_target: bool = True) -> torch.Tensor:
        state_values = self.state_values(state)
        action_values = torch.zeros(self.action_dim, dtype=torch.float32)
        offset = 0

        if action.kind not in self.action_kinds:
            raise ValueError(f"unknown action kind: {action.kind}")
        action_values[offset + self.action_kinds.index(action.kind)] = 1.0
        offset += len(self.action_kinds)

        if action.actor_id not in self.actor_ids:
            raise ValueError(f"unknown action actor: {action.actor_id}")
        action_values[offset + self.actor_ids.index(action.actor_id)] = 1.0
        offset += len(self.actor_ids)

        if bind_target:
            if action.target_id not in self.target_ids:
                raise ValueError(f"unknown action target: {action.target_id}")
            action_values[offset + self.target_ids.index(action.target_id)] = 1.0
        offset += len(self.target_ids)

        parameters = dict(action.parameters)
        parameter_values = []
        for name in self.parameter_names:
            value = _numeric(parameters.get(name, 0.0), f"action parameter {name}")
            action_values[offset] = value
            parameter_values.append(value)
            offset += 1
        if bind_target:
            target_index = self.target_ids.index(action.target_id)
            interaction_offset = offset + target_index * len(self.parameter_names)
            for parameter_index, value in enumerate(parameter_values):
                action_values[interaction_offset + parameter_index] = value
        return torch.cat((state_values, action_values))

    def delta_target(self, case: WorldInterventionCase) -> torch.Tensor:
        return self.state_values(case.expected_state) - self.state_values(case.initial)

    def payload(self) -> dict[str, Any]:
        return {
            "object_ids": list(self.object_ids),
            "state_slots": [list(item) for item in self.state_slots],
            "relation_slots": [list(item) for item in self.relation_slots],
            "action_kinds": list(self.action_kinds),
            "actor_ids": list(self.actor_ids),
            "target_ids": list(self.target_ids),
            "parameter_names": list(self.parameter_names),
            "target_parameter_interactions": len(self.target_ids) * len(self.parameter_names),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "input_dim": self.input_dim,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> WorldSchema:
        return cls(
            object_ids=tuple(str(item) for item in payload["object_ids"]),
            state_slots=tuple((str(item[0]), str(item[1])) for item in payload["state_slots"]),
            relation_slots=tuple(
                (str(item[0]), str(item[1]), str(item[2]))
                for item in payload.get("relation_slots", ())
            ),
            action_kinds=tuple(str(item) for item in payload["action_kinds"]),
            actor_ids=tuple(str(item) for item in payload["actor_ids"]),
            target_ids=tuple(str(item) for item in payload["target_ids"]),
            parameter_names=tuple(str(item) for item in payload["parameter_names"]),
        )


@dataclass(frozen=True)
class WorldPrediction:
    state: WorldState
    reward: float
    success_probability: float


@dataclass(frozen=True)
class WorldEpisodeRollout:
    episode_id: str
    predictions: tuple[WorldPrediction, ...]

    @property
    def final_state(self) -> WorldState:
        return self.predictions[-1].state


def _replace_numeric_state(state: WorldState, schema: WorldSchema, values: torch.Tensor) -> WorldState:
    updates: dict[str, dict[str, float]] = {}
    for index, (object_id, name) in enumerate(schema.state_slots):
        updates.setdefault(object_id, {})[name] = float(values[index].detach().cpu())
    objects = []
    for obj in state.objects:
        attributes = dict(obj.attributes)
        attributes.update(updates.get(obj.object_id, {}))
        objects.append(WorldObject(obj.object_id, attributes=attributes, tags=obj.tags))
    known_relations = set(schema.relation_slots)
    relations = {relation for relation in state.relations if relation not in known_relations}
    relations.update(
        relation
        for index, relation in enumerate(schema.relation_slots)
        if float(values[schema.object_state_dim + index].detach().cpu()) >= 0.5
    )
    return WorldState(
        tick=state.tick + 1,
        latent=state.latent.detach().clone(),
        entities=state.entities,
        relations=tuple(sorted(relations)),
        objects=tuple(objects),
        events=state.events,
        affordances=state.affordances,
        uncertainty=state.uncertainty,
    )


class WorldDynamicsLearner(nn.Module):
    """A compact learned transition model with no sequence-model dependency."""

    def __init__(self, schema: WorldSchema, *, hidden_dim: int = 64, seed: int = 0) -> None:
        super().__init__()
        if int(hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        torch.manual_seed(int(seed))
        self.schema = schema
        self.hidden_dim = int(hidden_dim)
        self.online_updates = 0
        self.network = nn.Sequential(
            nn.Linear(schema.input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, schema.state_dim + 2),
        )

    def predict(
        self, state: WorldState, action: Any, *, bind_target: bool = True
    ) -> WorldPrediction:
        self.eval()
        with torch.no_grad():
            output = self.network(self.schema.encode(state, action, bind_target=bind_target))
        delta = output[: self.schema.state_dim]
        values = self.schema.state_values(state) + delta
        return WorldPrediction(
            state=_replace_numeric_state(state, self.schema, values),
            reward=float(output[-2]),
            success_probability=float(torch.sigmoid(output[-1])),
        )

    def fit(
        self,
        cases: tuple[WorldInterventionCase, ...],
        *,
        epochs: int = 250,
        learning_rate: float = 0.01,
    ) -> list[float]:
        if not cases:
            raise ValueError("world dynamics requires training cases")
        if int(epochs) <= 0 or float(learning_rate) <= 0.0:
            raise ValueError("epochs and learning_rate must be positive")
        features = torch.stack([self.schema.encode(case.initial, case.action) for case in cases])
        targets = torch.stack(
            [
                torch.cat(
                    (
                        self.schema.delta_target(case),
                        torch.tensor(
                            [
                                float(case.expected_outcome.reward),
                                float(
                                    case.expected_outcome.success
                                    if case.expected_outcome.success is not None
                                    else case.expected_outcome.reward > 0.0
                                ),
                            ],
                            dtype=torch.float32,
                        ),
                    )
                )
                for case in cases
            ]
        )
        optimizer = torch.optim.Adam(self.parameters(), lr=float(learning_rate))
        losses = []
        self.train()
        for _ in range(int(epochs)):
            optimizer.zero_grad(set_to_none=True)
            output = self.network(features)
            delta_loss = torch.nn.functional.mse_loss(output[:, : self.schema.state_dim], targets[:, : self.schema.state_dim])
            reward_loss = torch.nn.functional.mse_loss(output[:, -2], targets[:, -2])
            success_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                output[:, -1], targets[:, -1]
            )
            loss = delta_loss + reward_loss + success_loss
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        return losses

    def online_update(
        self,
        transition: WorldTransition,
        *,
        learning_rate: float = 0.005,
        repeats: int = 1,
    ) -> list[float]:
        """Apply local error-driven correction from one experienced transition."""

        if float(learning_rate) <= 0.0 or int(repeats) <= 0:
            raise ValueError("learning_rate and repeats must be positive")
        features = self.schema.encode(transition.before, transition.action).unsqueeze(0)
        target = torch.cat(
            (
                self.schema.state_values(transition.after)
                - self.schema.state_values(transition.before),
                torch.tensor(
                    [
                        float(transition.outcome.reward),
                        float(
                            transition.outcome.success
                            if transition.outcome.success is not None
                            else transition.outcome.reward > 0.0
                        ),
                    ],
                    dtype=torch.float32,
                ),
            )
        ).unsqueeze(0)
        optimizer = torch.optim.SGD(self.parameters(), lr=float(learning_rate))
        losses = []
        self.train()
        for _ in range(int(repeats)):
            optimizer.zero_grad(set_to_none=True)
            output = self.network(features)
            delta_loss = torch.nn.functional.mse_loss(
                output[:, : self.schema.state_dim], target[:, : self.schema.state_dim]
            )
            reward_loss = torch.nn.functional.mse_loss(output[:, -2], target[:, -2])
            success_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                output[:, -1], target[:, -1]
            )
            loss = delta_loss + reward_loss + success_loss
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        self.online_updates += 1
        return losses


@dataclass(frozen=True)
class WorldInterventionEvaluationConfig:
    seeds: tuple[int, ...] = (11, 29, 47)
    hidden_dim: int = 64
    epochs: int = 250
    learning_rate: float = 0.01
    minimum_state_gain: float = 0.05
    minimum_outcome_gain: float = 0.02
    minimum_binding_drop: float = 0.05
    maximum_state_mse: float = 0.25
    maximum_outcome_error: float = 1.0
    maximum_cross_seed_std: float = 0.2


def _outcome_success(case: WorldInterventionCase) -> float:
    return float(
        case.expected_outcome.success
        if case.expected_outcome.success is not None
        else case.expected_outcome.reward > 0.0
    )


def _metrics(
    predictions: tuple[WorldPrediction, ...],
    cases: tuple[WorldInterventionCase, ...],
    schema: WorldSchema,
) -> dict[str, float]:
    state_errors = []
    reward_errors = []
    success_losses = []
    success_correct = []
    for prediction, case in zip(predictions, cases, strict=True):
        expected_values = schema.state_values(case.expected_state)
        actual_values = schema.state_values(prediction.state)
        state_errors.append(float(torch.mean((actual_values - expected_values) ** 2)))
        reward_errors.append((prediction.reward - float(case.expected_outcome.reward)) ** 2)
        expected_success = _outcome_success(case)
        probability = min(max(prediction.success_probability, 1e-6), 1.0 - 1e-6)
        success_losses.append(
            -(expected_success * math.log(probability) + (1.0 - expected_success) * math.log(1.0 - probability))
        )
        success_correct.append(float((prediction.success_probability >= 0.5) == bool(expected_success)))
    state_mse = sum(state_errors) / len(state_errors)
    reward_mse = sum(reward_errors) / len(reward_errors)
    success_bce = sum(success_losses) / len(success_losses)
    return {
        "state_mse": state_mse,
        "reward_mse": reward_mse,
        "success_bce": success_bce,
        "outcome_error": reward_mse + success_bce,
        "success_accuracy": sum(success_correct) / len(success_correct),
    }


def _mean_targets(
    cases: tuple[WorldInterventionCase, ...], schema: WorldSchema
) -> tuple[torch.Tensor, float, float]:
    deltas = torch.stack([schema.delta_target(case) for case in cases])
    rewards = sum(float(case.expected_outcome.reward) for case in cases) / len(cases)
    successes = sum(_outcome_success(case) for case in cases) / len(cases)
    return deltas.mean(dim=0), rewards, successes


def _baseline_predictions(
    cases: tuple[WorldInterventionCase, ...],
    schema: WorldSchema,
    *,
    deltas: dict[str, torch.Tensor] | None,
    rewards: dict[str, float] | None,
    successes: dict[str, float] | None,
) -> tuple[WorldPrediction, ...]:
    global_delta, global_reward, global_success = _mean_targets(cases, schema)
    predictions = []
    for case in cases:
        key = case.action.kind
        delta = global_delta if deltas is None else deltas.get(key, global_delta)
        reward = global_reward if rewards is None else rewards.get(key, global_reward)
        success = global_success if successes is None else successes.get(key, global_success)
        values = schema.state_values(case.initial) + delta
        predictions.append(
            WorldPrediction(
                state=_replace_numeric_state(case.initial, schema, values),
                reward=reward,
                success_probability=success,
            )
        )
    return tuple(predictions)


def _action_only_statistics(
    cases: tuple[WorldInterventionCase, ...], schema: WorldSchema
) -> tuple[dict[str, torch.Tensor], dict[str, float], dict[str, float]]:
    grouped: dict[str, list[WorldInterventionCase]] = {}
    for case in cases:
        grouped.setdefault(case.action.kind, []).append(case)
    deltas = {}
    rewards = {}
    successes = {}
    for kind, group in grouped.items():
        delta, reward, success = _mean_targets(tuple(group), schema)
        deltas[kind] = delta
        rewards[kind] = reward
        successes[kind] = success
    return deltas, rewards, successes


class WorldInterventionEvaluator:
    """Train and score A2's first causal intervention gate."""

    FORMAT = "taiji-a2-world-intervention-v1"

    def __init__(self, config: WorldInterventionEvaluationConfig | None = None) -> None:
        self.config = config or WorldInterventionEvaluationConfig()

    def evaluate(self, corpus: WorldInterventionCorpus) -> dict[str, Any]:
        if not corpus.train or not corpus.holdout:
            raise ValueError("A2 evaluation requires non-empty train and holdout splits")
        schema = WorldSchema.from_corpus(corpus)
        action_deltas, action_rewards, action_successes = _action_only_statistics(
            corpus.train, schema
        )
        seed_results = []
        for seed in self.config.seeds:
            learner = WorldDynamicsLearner(
                schema,
                hidden_dim=self.config.hidden_dim,
                seed=int(seed),
            )
            losses = learner.fit(
                corpus.train,
                epochs=self.config.epochs,
                learning_rate=self.config.learning_rate,
            )
            learned = tuple(
                learner.predict(case.initial, case.action) for case in corpus.holdout
            )
            lesion = tuple(
                learner.predict(case.initial, case.action, bind_target=False)
                for case in corpus.holdout
            )
            frequency = _baseline_predictions(
                corpus.holdout,
                schema,
                deltas=None,
                rewards=None,
                successes=None,
            )
            action_only = _baseline_predictions(
                corpus.holdout,
                schema,
                deltas=action_deltas,
                rewards=action_rewards,
                successes=action_successes,
            )
            seed_result = {
                "seed": int(seed),
                "training_loss": float(losses[-1]),
                "learned": _metrics(learned, corpus.holdout, schema),
                "target_binding_lesion": _metrics(lesion, corpus.holdout, schema),
                "frequency_baseline": _metrics(frequency, corpus.holdout, schema),
                "action_only_baseline": _metrics(action_only, corpus.holdout, schema),
            }
            if corpus.time_shuffled:
                shuffled = tuple(
                    learner.predict(case.initial, case.action) for case in corpus.time_shuffled
                )
                seed_result["time_shuffled"] = _metrics(
                    shuffled, corpus.time_shuffled, schema
                )
            seed_results.append(seed_result)
        state_gains = [
            item["frequency_baseline"]["state_mse"] - item["learned"]["state_mse"]
            for item in seed_results
        ]
        outcome_gains = [
            item["frequency_baseline"]["outcome_error"] - item["learned"]["outcome_error"]
            for item in seed_results
        ]
        binding_drops = [
            item["target_binding_lesion"]["state_mse"] - item["learned"]["state_mse"]
            for item in seed_results
        ]
        state_errors = [item["learned"]["state_mse"] for item in seed_results]
        outcome_errors = [item["learned"]["outcome_error"] for item in seed_results]
        state_error_max = max(state_errors)
        outcome_error_max = max(outcome_errors)
        gate = {
            "state_gain_min": min(state_gains),
            "outcome_gain_min": min(outcome_gains),
            "binding_drop_min": min(binding_drops),
            "state_error_max": state_error_max,
            "outcome_error_max": outcome_error_max,
            "state_gain_std": float(torch.tensor(state_gains).std(unbiased=False)),
            "passed": (
                min(state_gains) >= self.config.minimum_state_gain
                and min(outcome_gains) >= self.config.minimum_outcome_gain
                and min(binding_drops) >= self.config.minimum_binding_drop
                and state_error_max <= self.config.maximum_state_mse
                and outcome_error_max <= self.config.maximum_outcome_error
                and float(torch.tensor(state_gains).std(unbiased=False))
                <= self.config.maximum_cross_seed_std
            ),
        }
        return {
            "format": self.FORMAT,
            "config": {
                "seeds": list(self.config.seeds),
                "hidden_dim": self.config.hidden_dim,
                "epochs": self.config.epochs,
                "learning_rate": self.config.learning_rate,
                "minimum_state_gain": self.config.minimum_state_gain,
                "minimum_outcome_gain": self.config.minimum_outcome_gain,
                "minimum_binding_drop": self.config.minimum_binding_drop,
                "maximum_state_mse": self.config.maximum_state_mse,
                "maximum_outcome_error": self.config.maximum_outcome_error,
                "maximum_cross_seed_std": self.config.maximum_cross_seed_std,
            },
            "schema": schema.payload(),
            "train_cases": len(corpus.train),
            "holdout_cases": len(corpus.holdout),
            "time_shuffled_cases": len(corpus.time_shuffled),
            "seeds": seed_results,
            "gate": gate,
        }


def _case_from_transition(transition: Any, case_id: str) -> WorldInterventionCase:
    return WorldInterventionCase(
        case_id=case_id,
        initial=transition.before,
        action=transition.action,
        expected_state=transition.after,
        expected_outcome=transition.outcome,
    )


def rollout_episode(
    learner: WorldDynamicsLearner,
    episode: WorldEpisode,
    *,
    bind_target: bool = True,
) -> WorldEpisodeRollout:
    current = episode.initial
    predictions = []
    for transition in episode.transitions:
        prediction = learner.predict(current, transition.action, bind_target=bind_target)
        predictions.append(prediction)
        current = prediction.state
    return WorldEpisodeRollout(episode_id=episode.episode_id, predictions=tuple(predictions))


def _episode_metrics(
    rollouts: tuple[WorldEpisodeRollout, ...],
    episodes: tuple[WorldEpisode, ...],
    schema: WorldSchema,
) -> dict[str, float]:
    state_errors = []
    final_errors = []
    reward_errors = []
    success_correct = []
    for rollout, episode in zip(rollouts, episodes, strict=True):
        episode_final_error = None
        for prediction, transition in zip(rollout.predictions, episode.transitions, strict=True):
            state_error = float(
                torch.mean(
                    (schema.state_values(prediction.state) - schema.state_values(transition.after))
                    ** 2
                )
            )
            state_errors.append(state_error)
            episode_final_error = state_error
            reward_errors.append((prediction.reward - transition.outcome.reward) ** 2)
            expected_success = bool(
                transition.outcome.success
                if transition.outcome.success is not None
                else transition.outcome.reward > 0.0
            )
            success_correct.append(
                float((prediction.success_probability >= 0.5) == expected_success)
            )
        final_errors.append(float(episode_final_error))
    return {
        "rollout_state_mse": sum(state_errors) / len(state_errors),
        "final_state_mse": sum(final_errors) / len(final_errors),
        "reward_mse": sum(reward_errors) / len(reward_errors),
        "success_accuracy": sum(success_correct) / len(success_correct),
    }


def _checkpoint_recovery(episodes: tuple[WorldEpisode, ...]) -> bool:
    for episode in episodes:
        store = TaijiWorldState(episode.initial)
        midpoint = max(1, len(episode.transitions) // 2)
        restored = None
        checkpoint_state = None
        for index, transition in enumerate(episode.transitions):
            store.apply(transition)
            if index + 1 == midpoint:
                restored = TaijiWorldState.from_checkpoint(store.checkpoint())
                checkpoint_state = transition.after
        if restored is None:
            return False
        if checkpoint_state is None or not _world_state_equal(restored.state, checkpoint_state):
            return False
        for transition in episode.transitions[midpoint:]:
            restored.apply(transition)
        if not _world_state_equal(restored.state, episode.final_state):
            return False
    return True


@dataclass(frozen=True)
class WorldEpisodeEvaluationConfig:
    seeds: tuple[int, ...] = (11, 29, 47)
    hidden_dim: int = 64
    epochs: int = 250
    learning_rate: float = 0.01
    maximum_rollout_state_mse: float = 0.25
    maximum_final_state_mse: float = 0.5


class WorldEpisodeEvaluator:
    """Evaluate multi-step rollout, episode-ID independence and recovery."""

    FORMAT = "taiji-a2-world-episode-v1"

    def __init__(self, config: WorldEpisodeEvaluationConfig | None = None) -> None:
        self.config = config or WorldEpisodeEvaluationConfig()

    def evaluate(self, corpus: WorldEpisodeCorpus) -> dict[str, Any]:
        if not corpus.train or not corpus.holdout:
            raise ValueError("episode evaluation requires train and holdout episodes")
        train_cases = tuple(
            _case_from_transition(transition, f"{episode.episode_id}:{index}")
            for episode in corpus.train
            for index, transition in enumerate(episode.transitions)
        )
        holdout_cases = tuple(
            _case_from_transition(transition, f"{episode.episode_id}:{index}")
            for episode in corpus.holdout
            for index, transition in enumerate(episode.transitions)
        )
        intervention_corpus = WorldInterventionCorpus(train=train_cases, holdout=holdout_cases)
        schema = WorldSchema.from_corpus(intervention_corpus)
        seed_results = []
        for seed in self.config.seeds:
            learner = WorldDynamicsLearner(
                schema,
                hidden_dim=self.config.hidden_dim,
                seed=int(seed),
            )
            losses = learner.fit(
                train_cases,
                epochs=self.config.epochs,
                learning_rate=self.config.learning_rate,
            )
            rollouts = tuple(rollout_episode(learner, episode) for episode in corpus.holdout)
            seed_results.append(
                {
                    "seed": int(seed),
                    "training_loss": float(losses[-1]),
                    "rollout": _episode_metrics(rollouts, corpus.holdout, schema),
                    "schema_uses_episode_id": False,
                }
            )
        rollout_errors = [item["rollout"]["rollout_state_mse"] for item in seed_results]
        final_errors = [item["rollout"]["final_state_mse"] for item in seed_results]
        checkpoint_passed = _checkpoint_recovery(corpus.holdout)
        gate = {
            "rollout_state_mse_max": max(rollout_errors),
            "final_state_mse_max": max(final_errors),
            "checkpoint_recovery": checkpoint_passed,
            "passed": (
                max(rollout_errors) <= self.config.maximum_rollout_state_mse
                and max(final_errors) <= self.config.maximum_final_state_mse
                and checkpoint_passed
            ),
        }
        return {
            "format": self.FORMAT,
            "config": {
                "seeds": list(self.config.seeds),
                "hidden_dim": self.config.hidden_dim,
                "epochs": self.config.epochs,
                "learning_rate": self.config.learning_rate,
                "maximum_rollout_state_mse": self.config.maximum_rollout_state_mse,
                "maximum_final_state_mse": self.config.maximum_final_state_mse,
            },
            "schema": schema.payload(),
            "train_episodes": len(corpus.train),
            "holdout_episodes": len(corpus.holdout),
            "seeds": seed_results,
            "gate": gate,
        }
