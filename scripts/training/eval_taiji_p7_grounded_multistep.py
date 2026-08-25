"""Evaluate grounded multi-step executive transfer across train/holdout and seeds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji import (  # noqa: E402
    ActionIntent,
    AffordanceFeatureTrainingExample,
    ContentPlan,
    EnvironmentOutcome,
    ExecutiveCandidate,
    ExecutiveContext,
    ExecutiveController,
    ExecutiveTrainingExample,
    LearnedAffordanceFeatures,
    Observation,
    TaijiConfig,
    TSKV8Adapter,
    WorldAffordance,
    WorldAffordanceGroundingProducer,
    WorldObject,
    WorldState,
)

MANIFEST_FORMAT = "taiji-p7-grounded-multistep-manifest-v1"
REPORT_FORMAT = "taiji-p7-grounded-multistep-v1"
GROUNDING_DIM = 8
FEATURE_DIM = 4
SEEDS = (11, 23, 37)


def _config(seed: int) -> TaijiConfig:
    return TaijiConfig(
        region_sizes=(24, 16),
        synapse_fan_in=6,
        motor_fan_in=8,
        seed=seed,
    )


def _world(
    tick: int,
    *,
    actor: str,
    positive: str,
    negative: str,
    predicate: str,
) -> WorldState:
    return WorldState(
        tick=tick,
        relations=((actor, predicate, positive),),
        objects=(
            WorldObject(actor, attributes={"energy": 1.0}),
            WorldObject(positive, attributes={"position": 1.0}),
            WorldObject(negative, attributes={"position": 0.0}),
        ),
        affordances=(
            WorldAffordance(
                affordance_id=f"positive-{positive}",
                action_kind="holdout-positive-action",
                actor_id=actor,
                target_id=positive,
                parameters={"action_symbol": 10, "available_actions": (10, 11)},
            ),
            WorldAffordance(
                affordance_id=f"negative-{negative}",
                action_kind="holdout-negative-action",
                actor_id=actor,
                target_id=negative,
                parameters={"action_symbol": 11, "available_actions": (10, 11)},
            ),
        ),
    )


def _train_cases() -> tuple[tuple[WorldState, str, float], ...]:
    return (
        (
            _world(0, actor="agent-a", positive="red", negative="blue", predicate="near"),
            "positive-red",
            1.0,
        ),
        (
            _world(0, actor="agent-a", positive="blue", negative="red", predicate="near"),
            "negative-red",
            -1.0,
        ),
        (
            _world(0, actor="agent-b", positive="green", negative="yellow", predicate="touches"),
            "positive-green",
            1.0,
        ),
        (
            _world(0, actor="agent-b", positive="yellow", negative="green", predicate="touches"),
            "negative-green",
            -1.0,
        ),
    )


class _MultiStepEnvironment:
    def __init__(
        self,
        after_states: tuple[WorldState, ...],
        *,
        start_index: int = 0,
        success_indices: frozenset[int] | None = None,
    ) -> None:
        self.after_states = after_states
        self.actions: list[int] = []
        self._step_index = int(start_index)
        self.success_indices = (
            frozenset({len(after_states) - 1})
            if success_indices is None
            else frozenset(int(item) for item in success_indices)
        )

    def reset(self) -> tuple[int, tuple[int, ...]]:
        return 65, (10, 11)

    def step(self, action_symbol: int) -> EnvironmentOutcome:
        index = self._step_index
        self._step_index += 1
        self.actions.append(int(action_symbol))
        success = index in self.success_indices
        return EnvironmentOutcome(
            sensation=43 + index,
            reward=1.0 if success else -1.0,
            terminal=index == len(self.after_states) - 1,
            success=success,
            world_state=self.after_states[index],
        )


def _candidate(
    affordance: WorldAffordance,
    *,
    tick: int,
    features: torch.Tensor,
) -> ExecutiveCandidate:
    intent_id = f"train:{affordance.affordance_id}:intent"
    intent = affordance.action_kind
    action_intent = ActionIntent(
        intent_id=intent_id,
        kind=intent,
        parameters=dict(affordance.parameters),
        confidence=affordance.confidence,
        tick=tick,
    )
    return ExecutiveCandidate(
        candidate_id=f"candidate:{affordance.affordance_id}",
        action_intent=action_intent,
        content_plan=ContentPlan(
            content_id=f"content:{affordance.affordance_id}",
            intent_id=intent_id,
            intent_kind=intent,
            semantic_slots={"target_id": affordance.target_id},
            confidence=affordance.confidence,
            tick=tick,
        ),
        features=tuple(float(item) for item in features),
        source_affordance_id=affordance.affordance_id,
        provenance="evaluation-grounded",
    )


def evaluate_seed(seed: int) -> dict[str, object]:
    producer = WorldAffordanceGroundingProducer(GROUNDING_DIM)
    adapter = TSKV8Adapter(_config(seed))
    adapter.observe(65, learn=False)
    holdout = _world(
        adapter.tick + 1,
        actor="robot-holdout",
        positive="green-holdout",
        negative="yellow-holdout",
        predicate="supports-holdout",
    )
    adapter.observe_event(
        Observation(
            modality="text-byte",
            value=66,
            timestamp=holdout.tick,
            source="p7-grounded-multistep-eval",
        ),
        learn=False,
        world_state=holdout,
    )
    context = adapter.cognitive_snapshot().percept
    if context is None:
        raise RuntimeError("evaluation requires a current percept")
    context_features = context.features.detach().clone()
    source = LearnedAffordanceFeatures(
        input_dim=GROUNDING_DIM,
        feature_dim=FEATURE_DIM,
        context_dim=context_features.numel(),
        seed=seed + 1000,
    )
    train_examples: list[AffordanceFeatureTrainingExample] = []
    for index, (state, affordance_id, reward) in enumerate(_train_cases()):
        affordance = next(
            item for item in state.affordances if item.affordance_id == affordance_id
        )
        grounded = producer.ground(state, affordance)
        train_examples.append(
            AffordanceFeatureTrainingExample(
                example_id=f"seed-{seed}-train-{index}",
                affordance_id=grounded.affordance_id,
                action_kind=grounded.action_kind,
                grounding=grounded.features,
                reward=reward,
                percept_features=context_features,
                world_latent=context_features,
            )
        )
    source.fit(tuple(train_examples), epochs=180)
    adapter.attach_affordance_features(source)
    grounded_holdout = adapter.cognitive_snapshot().world.affordances
    runtime_context = ExecutiveContext.from_state(adapter.cognitive_snapshot())
    controller = ExecutiveController(candidate_feature_dim=FEATURE_DIM)
    training_candidates: list[ExecutiveTrainingExample] = []
    for state, affordance_id, reward in _train_cases():
        affordance = next(
            item for item in state.affordances if item.affordance_id == affordance_id
        )
        grounded = producer.ground(state, affordance)
        features = source.features_for(
            grounded,
            percept_features=context_features,
            world_latent=context_features,
        )
        training_candidates.append(
            ExecutiveTrainingExample(
                candidate=_candidate(grounded, tick=runtime_context.tick, features=features),
                context=runtime_context,
                reward=reward,
            )
        )
    controller.fit(tuple(training_candidates), epochs=180)
    executive_fit_steps = controller.training_steps
    adapter.attach_executive(controller)
    candidates = adapter.synthesize_executive_candidates()
    decision = adapter.select_executive(candidates)
    holdout_selected_positive = decision.selected.source_affordance_id == "positive-green-holdout"
    base_checkpoint = adapter.native_checkpoint()
    holdout_by_id = {item.affordance_id: item for item in grounded_holdout}
    lesioned_features = source.encode(
        torch.zeros(GROUNDING_DIM),
        percept_features=context_features,
        world_latent=context_features,
    ).detach()
    producer_lesion_candidates = tuple(
        _candidate(
            holdout_by_id[affordance_id],
            tick=runtime_context.tick,
            features=lesioned_features,
        )
        for affordance_id in ("negative-yellow-holdout", "positive-green-holdout")
    )
    producer_lesion_decision = controller.select(
        producer_lesion_candidates,
        runtime_context,
    )
    producer_lesion_degrades = (
        producer_lesion_decision.selected.source_affordance_id != "positive-green-holdout"
    )
    feature_lesion_adapter = TSKV8Adapter(_config(seed + 2000))
    feature_lesion_adapter.observe(65, learn=False)
    feature_lesion_adapter.attach_affordance_features(None)
    try:
        feature_lesion_adapter.synthesize_executive_candidates()
    except RuntimeError as error:
        feature_source_lesion_blocks_synthesis = "learned affordance feature source" in str(error)
    else:
        feature_source_lesion_blocks_synthesis = False

    after_states = tuple(
        _world(
            holdout.tick + offset,
            actor="robot-holdout",
            positive="green-holdout",
            negative="yellow-holdout",
            predicate=predicate,
        )
        for offset, predicate in enumerate(
            ("supports-after", "reaches-after", "tracks-after", "arrives-after"),
            start=1,
        )
    )
    environment = _MultiStepEnvironment(
        after_states,
        success_indices=frozenset({3}),
    )
    first = adapter.execute_executive_action(environment, learn=True)
    first_transition = adapter.cognitive_snapshot().world_transition
    if first_transition is None:
        raise RuntimeError("grounded multi-step evaluation lost first transition")
    first_lineage = all(
        item.grounding_lineage
        for item in (*first_transition.before.affordances, *first_transition.after.affordances)
    )
    checkpoint = adapter.native_checkpoint()
    restored = TSKV8Adapter.from_native_checkpoint(checkpoint)
    checkpoint_pending = restored._pending_executive_credit is not None
    restored.replan_executive_after_failure(candidates)
    restored.record_delayed_executive_credit(0.5)
    transitions = [first_transition]
    outcomes = [first]
    for _ in range(1, len(after_states)):
        outcome = restored.execute_executive_action(environment, learn=True)
        transition = restored.cognitive_snapshot().world_transition
        if transition is None:
            raise RuntimeError("grounded multi-step evaluation lost a later transition")
        transitions.append(transition)
        outcomes.append(outcome)
        if len(outcomes) < len(after_states):
            restored.replan_executive_after_failure(candidates)
            restored.record_delayed_executive_credit(0.5)
    horizon_lineage = all(
        item.grounding_lineage
        for transition in transitions
        for item in (*transition.before.affordances, *transition.after.affordances)
    )
    delayed_credit = bool(
        restored._affordance_features is not None
        and restored._affordance_features.online_updates == 7
        and restored._executive is not None
        and restored._executive.training_steps == executive_fit_steps + 7
    )
    delayed_lesion_environment = _MultiStepEnvironment(
        after_states,
        start_index=1,
    )
    delayed_lesion = TSKV8Adapter.from_native_checkpoint(checkpoint)
    delayed_lesion.replan_executive_after_failure(candidates)
    delayed_lesion.execute_executive_action(delayed_lesion_environment, learn=True)
    delayed_credit_lesion_effective = bool(
        delayed_lesion._affordance_features is not None
        and delayed_lesion._affordance_features.online_updates == 2
        and delayed_lesion._executive is not None
        and delayed_lesion._executive.training_steps == executive_fit_steps + 2
    )
    variable_episode_specs = (
        ("short-single-failure", 3, frozenset({0})),
        ("mid-late-failures", 4, frozenset({1, 2})),
        ("long-interleaved-failures", 5, frozenset({0, 2, 3})),
    )
    variable_episode_runs: list[dict[str, object]] = []
    for variant_name, length, failure_indices in variable_episode_specs:
        variant = TSKV8Adapter.from_native_checkpoint(base_checkpoint)
        variant.select_executive(candidates)
        predicates = tuple(f"{variant_name}-relation-{index}" for index in range(length))
        variant_after_states = tuple(
            _world(
                holdout.tick + index + 1,
                actor="robot-holdout",
                positive="green-holdout",
                negative="yellow-holdout",
                predicate=predicate,
            )
            for index, predicate in enumerate(predicates)
        )
        variant_environment = _MultiStepEnvironment(
            variant_after_states,
            success_indices=frozenset(set(range(length)) - set(failure_indices)),
        )
        variant_outcomes: list[EnvironmentOutcome] = []
        variant_transitions = []
        replan_count = 0
        for index in range(length):
            outcome = variant.execute_executive_action(
                variant_environment,
                learn=True,
            )
            transition = variant.cognitive_snapshot().world_transition
            if transition is None:
                raise RuntimeError(f"variable episode lost transition: {variant_name}")
            variant_outcomes.append(outcome)
            variant_transitions.append(transition)
            if index < length - 1:
                if outcome.success is False:
                    replan_count += 1
                    variant.replan_executive_after_failure(candidates)
                    variant.record_delayed_executive_credit(0.25)
                else:
                    variant.record_delayed_executive_credit(0.25)
                    variant.select_executive(candidates)
        expected_updates = (2 * length) - 1
        observed_failures = tuple(
            index for index, outcome in enumerate(variant_outcomes) if outcome.success is False
        )
        lineage_complete = all(
            item.grounding_lineage
            for transition in variant_transitions
            for item in (*transition.before.affordances, *transition.after.affordances)
        )
        variable_episode_runs.append(
            {
                "name": variant_name,
                "length": length,
                "failure_indices": sorted(failure_indices),
                "observed_failure_indices": list(observed_failures),
                "replans": replan_count,
                "lineage_complete": lineage_complete,
                "final_success": bool(
                    variant_outcomes[-1].success and variant_outcomes[-1].terminal
                ),
                "credit_updates_complete": bool(
                    variant._affordance_features is not None
                    and variant._affordance_features.online_updates == expected_updates
                    and variant._executive is not None
                    and variant._executive.training_steps == executive_fit_steps + expected_updates
                ),
            }
        )
    variable_episode_gate = all(
        run["failure_indices"] == run["observed_failure_indices"]
        and run["replans"] == len(run["failure_indices"])
        and run["lineage_complete"]
        and run["final_success"]
        and run["credit_updates_complete"]
        for run in variable_episode_runs
    )
    return {
        "seed": seed,
        "train_examples": len(train_examples),
        "holdout_affordances": len(grounded_holdout),
        "holdout_selected_positive": holdout_selected_positive,
        "producer_lesion_degrades": producer_lesion_degrades,
        "feature_source_lesion_blocks_synthesis": feature_source_lesion_blocks_synthesis,
        "first_failed": first.success is False,
        "all_intermediate_failed": all(
            outcome.success is False for outcome in outcomes[:-1]
        ),
        "final_succeeded": outcomes[-1].success is True and outcomes[-1].terminal,
        "checkpoint_pending_credit": checkpoint_pending,
        "first_lineage_complete": first_lineage,
        "horizon_lineage_complete": horizon_lineage,
        "delayed_credit_complete": delayed_credit,
        "delayed_credit_lesion_effective": delayed_credit_lesion_effective,
        "source_online_updates": (
            0 if restored._affordance_features is None else restored._affordance_features.online_updates
        ),
        "executive_training_steps": (
            0 if restored._executive is None else restored._executive.training_steps
        ),
        "executive_fit_steps": executive_fit_steps,
        "continuous_replan_complete": (
            not restored.replan_required
            and len(environment.actions) == len(after_states)
            and len(set(environment.actions)) > 1
        ),
        "variable_episode_gate": variable_episode_gate,
        "variable_episode_runs": variable_episode_runs,
        "action_trace": environment.actions,
    }


def build_manifest(seeds: tuple[int, ...] = SEEDS) -> dict[str, object]:
    return {
        "format": MANIFEST_FORMAT,
        "task": "four-step grounded executive transfer with continuous failure replan and delayed credit",
        "train": {
            "examples": 4,
            "actor_ids": ["agent-a", "agent-b"],
            "target_ids": ["red", "blue", "green", "yellow"],
            "relation_predicates": ["near", "touches"],
        },
        "holdout": {
            "actor_ids": ["robot-holdout"],
            "target_ids": ["green-holdout", "yellow-holdout"],
            "relation_predicates": [
                "supports-holdout",
                "supports-after",
                "reaches-after",
                "tracks-after",
                "arrives-after",
            ],
            "action_kinds": ["holdout-positive-action", "holdout-negative-action"],
        },
        "seeds": list(seeds),
        "controls": [
            "native-checkpoint-continuation",
            "producer-lesion",
            "feature-source-lesion",
            "delayed-credit-lesion",
        ],
        "variable_episodes": [
            {"name": "short-single-failure", "length": 3, "failure_indices": [0]},
            {"name": "mid-late-failures", "length": 4, "failure_indices": [1, 2]},
            {"name": "long-interleaved-failures", "length": 5, "failure_indices": [0, 2, 3]},
        ],
        "boundary": "numeric world grounding and executive credit; not general semantics or intelligence",
    }


def evaluate(seeds: tuple[int, ...] = SEEDS) -> dict[str, object]:
    runs = [evaluate_seed(seed) for seed in seeds]
    metric_names = (
        "holdout_selected_positive",
        "producer_lesion_degrades",
        "feature_source_lesion_blocks_synthesis",
        "first_failed",
        "all_intermediate_failed",
        "final_succeeded",
        "checkpoint_pending_credit",
        "first_lineage_complete",
        "horizon_lineage_complete",
        "delayed_credit_complete",
        "delayed_credit_lesion_effective",
        "continuous_replan_complete",
        "variable_episode_gate",
    )
    rates = {
        name: sum(bool(run[name]) for run in runs) / len(runs)
        for name in metric_names
    }
    passed = all(rate == 1.0 for rate in rates.values())
    return {
        "format": REPORT_FORMAT,
        "manifest_format": MANIFEST_FORMAT,
        "metrics": {"cross_seed_rates": rates, "runs": runs},
        "gate": {
            "passed": passed,
            "criterion": "all seeds must transfer the holdout selection, preserve lineage and credit across four transitions and variable 3/4/5-step episodes, complete the declared replans, and show the declared lesions",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "reports" / "taiji_p7_grounded_multistep_manifest_20260825.json")
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "reports" / "taiji_p7_grounded_multistep_report_20260825.json")
    args = parser.parse_args()
    report = evaluate()
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(build_manifest(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
