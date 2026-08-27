"""Evaluate ledger-aware risk handling in a real stepwise environment loop."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_p3_open_set import (  # noqa: E402
    _config,
    _fit_world_learner,
    _transition_state,
    _world,
)
from taiji import (  # noqa: E402
    EnvironmentCapability,
    EnvironmentOutcome,
    Goal,
    GoalPlanner,
    Observation,
    Outcome,
    PlanningCandidate,
    PlanningConfig,
    TSKV8Adapter,
    WorldAction,
    WorldAffordance,
    WorldDynamicsLearner,
    WorldState,
    WorldTransition,
)

SEEDS = (11, 23, 37)
MANIFEST_FORMAT = "taiji-p3-risk-sensitive-execution-manifest-v1"
REPORT_FORMAT = "taiji-p3-risk-sensitive-execution-v1"


class _ScriptedEnvironment:
    """A deterministic environment that still crosses the real adapter boundary."""

    def __init__(
        self,
        states: tuple[WorldState, ...],
        *,
        rewards: tuple[float, ...],
        successes: tuple[bool, ...],
        terminals: tuple[bool, ...],
        available_actions: tuple[int, ...] = (10, 11),
        action_kinds: tuple[str, ...] = ("assemble", "idle"),
    ) -> None:
        if not (len(states) == len(rewards) == len(successes) == len(terminals)):
            raise ValueError("scripted environment sequences must have equal lengths")
        self.states = states
        self.rewards = rewards
        self.successes = successes
        self.terminals = terminals
        self.available_actions = available_actions
        self.action_kinds = action_kinds
        self.index = 0
        self.actions: list[int] = []

    def reset(self) -> tuple[int, tuple[int, ...]]:
        self.index = 0
        self.actions.clear()
        return 97, self.available_actions

    def step(self, action_symbol: int) -> EnvironmentOutcome:
        if self.index >= len(self.states):
            raise RuntimeError("scripted environment received too many actions")
        index = self.index
        self.index += 1
        self.actions.append(int(action_symbol))
        return EnvironmentOutcome(
            sensation=97 + self.index,
            reward=float(self.rewards[index]),
            success=self.successes[index],
            terminal=self.terminals[index],
            world_state=self.states[index],
            available_actions=self.available_actions,
            action_kinds=self.action_kinds,
        )


def _action(
    episode_id: str,
    tick: int,
    *,
    action_symbol: int,
    kind: str = "assemble",
    resource_cost: float | None = None,
) -> WorldAction:
    parameters = {"workspace_count": 2.0, "action_symbol": action_symbol}
    if resource_cost is not None:
        parameters["resource_cost"] = resource_cost
    return WorldAction(
        f"{episode_id}:assemble",
        kind,
        tick,
        actor_id="agent",
        target_id="target",
        parameters=parameters,
        provenance="risk-sensitive-execution",
    )


def _run_ledger_transition(
    learner: WorldDynamicsLearner,
    *,
    seed: int,
    episode_id: str,
    phase: int,
    reward: float,
    success: bool,
    action_symbol: int = 10,
    kind: str = "assemble",
) -> TSKV8Adapter:
    model = TSKV8Adapter(_config(seed), episode_id=episode_id)
    model.attach_world_dynamics(learner)
    initial = _world(episode_id, 1, target_id="target", phase=0)
    model.observe_event(
        Observation(
            modality="text-byte",
            value=97,
            timestamp=0,
            source="p3-risk-sensitive-execution",
        ),
        learn=False,
        world_state=initial,
    )
    before = model.cognitive_snapshot().world
    world_action = _action(
        episode_id,
        before.tick,
        action_symbol=action_symbol,
        kind=kind,
    )
    model.act((10, 11), sample=False, world_action=world_action)
    model.settle_action(
        reward,
        learn=False,
        learn_world=True,
        world_state=_transition_state(
            before,
            sample_id=episode_id,
            target_id="target",
            phase=phase,
            success=success,
        ),
        success=success,
    )
    return model


def _seed_ledger(seed: int) -> WorldDynamicsLearner:
    learner = _fit_world_learner(seed)
    for index in range(2):
        _run_ledger_transition(
            learner,
            seed=seed,
            episode_id=f"risk:{seed}:a{index + 1}",
            phase=1,
            reward=1.0,
            success=True,
        )
    for index in range(3):
        _run_ledger_transition(
            learner,
            seed=seed,
            episode_id=f"risk:{seed}:b{index + 1}",
            phase=0,
            reward=-1.0,
            success=False,
        )
    return learner


def _seed_recovery_alternative(learner: WorldDynamicsLearner, *, seed: int) -> WorldDynamicsLearner:
    before = _world(f"recovery-seed:{seed}", 2, target_id="target", phase=1)
    action = _action(
        f"recovery-seed:{seed}",
        before.tick,
        action_symbol=11,
        kind="idle",
        resource_cost=0.2,
    )
    after = _transition_state(
        before,
        sample_id=f"recovery-seed:{seed}",
        target_id="target",
        phase=1,
        success=True,
    )
    learner.online_update(
        WorldTransition(
            before=before,
            action=action,
            after=after,
            outcome=Outcome(
                intent_id=action.action_id,
                reward=1.0,
                success=True,
                terminal=False,
                tick=after.tick,
            ),
        )
    )
    return learner


def _seed_conflicted_ledger(seed: int) -> WorldDynamicsLearner:
    learner = _fit_world_learner(seed)
    for index in range(2):
        _run_ledger_transition(
            learner,
            seed=seed,
            episode_id=f"conflicted:{seed}:a{index + 1}",
            phase=1,
            reward=1.0,
            success=True,
        )
    _run_ledger_transition(
        learner,
        seed=seed,
        episode_id=f"conflicted:{seed}:b1",
        phase=0,
        reward=-1.0,
        success=False,
    )
    return learner


def _recovery_affordances() -> tuple[WorldAffordance, ...]:
    return (
        WorldAffordance(
            "assemble-again",
            "assemble",
            actor_id="agent",
            target_id="target",
            parameters={"workspace_count": 2.0},
            confidence=0.9,
        ),
        WorldAffordance(
            "idle-alternative",
            "idle",
            actor_id="agent",
            target_id="target",
            parameters={"workspace_count": 2.0, "resource_cost": 0.2},
            confidence=0.8,
        ),
        WorldAffordance(
            "secure-alternative",
            "secure",
            actor_id="agent",
            target_id="target",
            parameters={"workspace_count": 2.0, "resource_cost": 0.8},
            confidence=0.7,
        ),
        WorldAffordance(
            "archive-over-budget",
            "archive",
            actor_id="agent",
            target_id="target",
            parameters={"workspace_count": 2.0, "resource_cost": 1.2},
            confidence=0.6,
        ),
    )


def _rollout_steps(
    adapter: TSKV8Adapter,
    *,
    prefix: str,
    action_symbol: int,
    kind: str = "assemble",
) -> tuple[PlanningCandidate, ...]:
    start_tick = adapter.cognitive_snapshot().world.tick
    return tuple(
        PlanningCandidate(
            candidate_id=f"{prefix}-step-{index}",
            action=_action(
                f"{prefix}-{index}",
                start_tick + index,
                action_symbol=action_symbol,
                kind=kind,
            ),
            predicted_reward=0.0,
            success_probability=0.0,
            expected_progress=(index + 1) / 2.0,
        )
        for index in range(2)
    )


def _ledger_snapshot(
    learner: WorldDynamicsLearner,
    key: tuple[str, ...],
) -> tuple[tuple[int, ...], str, float, int, int, int]:
    hypotheses = learner.schema_registry.transition_hypotheses.get(key, ())
    return (
        tuple(sorted(int(item["evidence_count"]) for item in hypotheses)),
        ("unseen" if not hypotheses else learner.schema_registry.transition_outcome_mode(key)),
        (1.0 if not hypotheses else learner.schema_registry.transition_uncertainty(key)[0]),
        learner.online_updates,
        learner.transition_acceptances,
        learner.transition_rejections,
    )


def _attach_runtime(adapter: TSKV8Adapter, learner: WorldDynamicsLearner, seed: int) -> None:
    adapter.attach_world_dynamics(learner)
    adapter.attach_goal_planner(GoalPlanner(PlanningConfig(replan_error_threshold=100.0)))
    adapter.set_goals((Goal("reach-world", "reach the world target", priority=1.0),))
    adapter.observe_event(
        Observation(
            modality="text-byte",
            value=97,
            timestamp=0,
            source="p3-risk-sensitive-execution",
        ),
        learn=False,
        world_state=_world(f"runtime:{seed}", 1, target_id="target", phase=0),
    )


def _run_ambiguity_case(seed: int) -> dict[str, object]:
    learner = _seed_ledger(seed)
    _seed_recovery_alternative(learner, seed=seed)
    adapter = TSKV8Adapter(_config(seed), episode_id=f"risk-runtime:{seed}")
    _attach_runtime(adapter, learner, seed)
    steps = _rollout_steps(adapter, prefix=f"ambiguous:{seed}", action_symbol=10)
    rollout = adapter.imagine_world_rollout(f"ambiguous-rollout:{seed}", "reach-world", steps)
    adapter.plan_rollouts((rollout,))
    initial = adapter.cognitive_snapshot().world
    evidence_key = learner.schema_registry.transition_context_key(initial, steps[0].action)
    before = _ledger_snapshot(learner, evidence_key)
    environment = _ScriptedEnvironment(
        (
            replace(
                _transition_state(
                    initial,
                    sample_id=f"ambiguous:{seed}",
                    target_id="target",
                    phase=1,
                    success=True,
                ),
                affordances=_recovery_affordances(),
            ),
        ),
        rewards=(1.0,),
        successes=(True,),
        terminals=(False,),
        available_actions=(10, 11, 12, 13),
        action_kinds=("assemble", "idle", "secure", "archive"),
    )
    first = adapter.execute_imagined_rollout_step(
        environment,
        available_actions=(10, 11, 12, 13),
        action_kinds=("assemble", "idle", "secure", "archive"),
        learn=False,
        learn_world=True,
    )
    interrupted = adapter.cognitive_snapshot()
    trace = interrupted.world_calibration_trace[-1]
    checkpoint = adapter.native_checkpoint()
    restored = TSKV8Adapter.from_native_checkpoint(checkpoint)
    restored_learner = restored._world_dynamics
    if restored_learner is None:
        raise RuntimeError("risk execution checkpoint lost world dynamics")
    after_checkpoint = _ledger_snapshot(restored_learner, evidence_key)
    checkpoint_no_replay = after_checkpoint == _ledger_snapshot(learner, evidence_key)
    checkpoint_branch = restored.recovery_branch
    checkpoint_capability = restored.environment_capability

    recovery_before = restored.cognitive_snapshot().world
    synthesized_recovery = restored.synthesize_recovery_rollouts(
        goal_id="reach-world",
        resource_budget=1.0,
    )
    capability = restored.environment_capability
    if capability is None:
        raise RuntimeError("risk execution recovery lost environment capability")
    lineage_recorded = bool(
        synthesized_recovery
        and all(
            rollout.recovery_lineage is not None
            and rollout.recovery_lineage.capability_tick == capability.tick
            and rollout.recovery_lineage.capability_actions == capability.actions
            and rollout.recovery_lineage.capability_action_kinds == capability.action_kinds
            and rollout.recovery_lineage.affordance_content_identity
            == next(
                affordance
                for affordance in restored.cognitive_snapshot().world.affordances
                if affordance.affordance_id == rollout.recovery_lineage.affordance_id
            ).content_identity
            and rollout.recovery_lineage.action_semantic_key
            == restored_learner.schema_registry.action_semantic_key(rollout.steps[0].action)
            and rollout.recovery_lineage.schema_revision
            == restored_learner.schema_registry.active_version
            for rollout in synthesized_recovery
        )
    )
    lineage = synthesized_recovery[0].recovery_lineage
    if lineage is None:
        raise RuntimeError("recovery synthesis did not record rollout lineage")
    stale_planning_rejected = False
    stale_rollout = replace(
        synthesized_recovery[0],
        recovery_lineage=replace(
            lineage,
            capability_actions=(11,),
            capability_action_kinds=("idle",),
        ),
    )
    try:
        restored.plan_rollouts((stale_rollout,))
    except RuntimeError as error:
        stale_planning_rejected = "stale" in str(error)
    stale_content_rejected = False
    stale_content_rollout = replace(
        synthesized_recovery[0],
        recovery_lineage=replace(lineage, affordance_content_identity="replaced-content"),
    )
    try:
        restored.plan_rollouts((stale_content_rollout,))
    except RuntimeError as error:
        stale_content_rejected = "stale" in str(error)
    stale_action_rejected = False
    stale_action = replace(
        synthesized_recovery[0].steps[0].action,
        parameters=tuple(
            sorted(
                {
                    **dict(synthesized_recovery[0].steps[0].action.parameters),
                    "action_symbol": 12,
                }.items()
            )
        ),
    )
    stale_action_rollout = replace(
        synthesized_recovery[0],
        steps=(replace(synthesized_recovery[0].steps[0], action=stale_action),),
    )
    try:
        restored.plan_rollouts((stale_action_rollout,))
    except RuntimeError as error:
        stale_action_rejected = "stale" in str(error)
    recovery_rollout = next(
        rollout for rollout in synthesized_recovery if rollout.steps[0].action.kind == "idle"
    )
    risky_recovery = next(
        rollout for rollout in synthesized_recovery if rollout.steps[0].action.kind == "secure"
    )
    restored.plan_rollouts(synthesized_recovery)
    planned_recovery_branch = restored.recovery_branch
    recovery_checkpoint = restored.native_checkpoint()
    checkpointed_recovery = TSKV8Adapter.from_native_checkpoint(recovery_checkpoint)
    checkpoint_lineage_preserved = bool(
        checkpointed_recovery._planned_rollout is not None
        and checkpointed_recovery._planned_rollout.recovery_lineage
        == restored._planned_rollout.recovery_lineage
    )
    restored = checkpointed_recovery
    recovery_environment = _ScriptedEnvironment(
        (
            replace(
                _transition_state(
                    recovery_before,
                    sample_id=f"recovery:{seed}",
                    target_id="target",
                    phase=1,
                    success=True,
                ),
                affordances=_recovery_affordances(),
            ),
        ),
        rewards=(1.0,),
        successes=(True,),
        terminals=(True,),
        available_actions=(11,),
        action_kinds=("idle",),
    )
    restored._cognitive_state = replace(
        restored._cognitive_state,
        environment_capability=EnvironmentCapability(
            actions=(11,),
            action_kinds=("idle",),
            tick=restored.cognitive_snapshot().world.tick,
        ),
    )
    stale_execution_rejected = False
    try:
        restored.execute_imagined_rollout_step(
            recovery_environment,
            available_actions=(11,),
            action_kinds=("idle",),
            learn=False,
            learn_world=True,
        )
    except RuntimeError as error:
        stale_execution_rejected = "stale" in str(error)
    if not stale_execution_rejected or recovery_environment.actions:
        raise RuntimeError("stale recovery execution was not rejected before the environment step")
    restored._cognitive_state = replace(
        restored._cognitive_state,
        environment_capability=capability,
    )
    synthesized_recovery = restored.synthesize_recovery_rollouts(
        goal_id="reach-world",
        resource_budget=1.0,
    )
    recovery_rollout = next(
        rollout for rollout in synthesized_recovery if rollout.steps[0].action.kind == "idle"
    )
    risky_recovery = next(
        rollout for rollout in synthesized_recovery if rollout.steps[0].action.kind == "secure"
    )
    restored.plan_rollouts(synthesized_recovery)
    planned_recovery_branch = restored.recovery_branch
    recovery = restored.execute_imagined_rollout_step(
        recovery_environment,
        available_actions=(11,),
        action_kinds=("idle",),
        learn=False,
        learn_world=True,
    )
    final = restored.cognitive_snapshot()
    recovery_learner = restored._world_dynamics
    if recovery_learner is None:
        raise RuntimeError("risk execution recovery lost world dynamics")
    post_recovery_rollouts = restored.synthesize_recovery_rollouts(
        goal_id="reach-world",
        resource_budget=1.0,
    )
    final_checkpoint = TSKV8Adapter.from_native_checkpoint(restored.native_checkpoint())
    final_checkpoint_state = final_checkpoint.cognitive_snapshot()
    return {
        "risk_mode_before": rollout.steps[0].uncertainty_mode,
        "risk_uncertainty_before": rollout.steps[0].uncertainty,
        "first_success": first.success,
        "first_adjudication": trace.adjudication,
        "first_ledger_mode": trace.ledger_uncertainty_mode,
        "first_ledger_uncertainty": trace.ledger_uncertainty,
        "first_evidence_count": trace.ledger_evidence_count,
        "ambiguity_replan": bool(
            first.success is True
            and trace.adjudication == "rejected"
            and trace.ledger_uncertainty_mode == "stochastic"
            and adapter.replan_required
            and adapter._planned_rollout is None
        ),
        "checkpoint_no_replay": checkpoint_no_replay,
        "checkpoint_branch_preserved": bool(
            checkpoint_branch is not None
            and checkpoint_branch.reason == "outcome-adjudication"
            and checkpoint_branch.source_rollout_id == rollout.rollout_id
            and checkpoint_branch.remaining_rollout_steps == 1
        ),
        "recovery_branch_filters_rejected_action": bool(
            planned_recovery_branch is not None
            and planned_recovery_branch.replacement_rollout_id == recovery_rollout.rollout_id
        ),
        "recovery_branch_selects_low_risk_counterfactual": bool(
            risky_recovery.steps[0].uncertainty_mode == "unseen"
            and recovery_rollout.steps[0].uncertainty_mode == "deterministic"
            and planned_recovery_branch is not None
            and planned_recovery_branch.replacement_rollout_id == recovery_rollout.rollout_id
        ),
        "recovery_candidates_generated_from_affordances": bool(
            len(synthesized_recovery) == 2
            and all(
                rollout.steps[0].action.kind in {"idle", "secure"}
                for rollout in synthesized_recovery
            )
            and all(
                float(dict(rollout.steps[0].action.parameters)["resource_cost"]) <= 1.0
                for rollout in synthesized_recovery
            )
        ),
        "recovery_lineage_recorded": lineage_recorded,
        "stale_planning_rejected": stale_planning_rejected,
        "stale_content_rejected": stale_content_rejected,
        "stale_action_rejected": stale_action_rejected,
        "stale_execution_rejected": stale_execution_rejected,
        "checkpoint_lineage_preserved": checkpoint_lineage_preserved,
        "recovery_synthesized_candidates": [
            {
                "kind": rollout.steps[0].action.kind,
                "resource_cost": dict(rollout.steps[0].action.parameters).get("resource_cost"),
                "uncertainty_mode": rollout.steps[0].uncertainty_mode,
            }
            for rollout in synthesized_recovery
        ],
        "ledger_before": before[0],
        "ledger_after_checkpoint": after_checkpoint[0],
        "recovery_success": recovery.success,
        "recovery_terminal": recovery.terminal,
        "recovery_complete": bool(
            recovery.success is True
            and recovery.terminal
            and not restored.replan_required
            and restored._planned_rollout is None
            and restored.recovery_branch is None
        ),
        "trace_complete": bool(
            len(final.world_calibration_trace) == 2
            and final.world_calibration_trace[0].adjudication == "rejected"
            and final.world_calibration_trace[1].adjudication == "accepted"
        ),
        "checkpoint_trace_complete": len(final_checkpoint_state.world_calibration_trace) == 2,
        "checkpoint_capability_preserved": bool(
            checkpoint_capability is not None
            and checkpoint_capability.actions == (10, 11, 12, 13)
            and checkpoint_capability.action_kinds == ("assemble", "idle", "secure", "archive")
            and final_checkpoint_state.environment_capability == final.environment_capability
        ),
        "capability_refresh_filters_next_candidates": bool(
            final.environment_capability is not None
            and final.environment_capability.actions == (11,)
            and final.environment_capability.action_kinds == ("idle",)
            and len(post_recovery_rollouts) == 1
            and post_recovery_rollouts[0].steps[0].action.kind == "idle"
        ),
        "outcome_count": recovery_learner.schema_registry.transition_outcome_count,
        "online_updates": recovery_learner.online_updates,
        "transition_acceptances": recovery_learner.transition_acceptances,
        "transition_rejections": recovery_learner.transition_rejections,
    }


def _run_failure_case(seed: int) -> dict[str, object]:
    learner = _seed_ledger(seed + 1000)
    adapter = TSKV8Adapter(_config(seed), episode_id=f"risk-failure:{seed}")
    _attach_runtime(adapter, learner, seed + 1000)
    steps = _rollout_steps(
        adapter,
        prefix=f"failure:{seed}",
        action_symbol=11,
        kind="idle",
    )
    adapter.plan_rollouts(
        (adapter.imagine_world_rollout(f"failure-rollout:{seed}", "reach-world", steps),)
    )
    initial = adapter.cognitive_snapshot().world
    environment = _ScriptedEnvironment(
        (
            _transition_state(
                initial,
                sample_id=f"failure:{seed}",
                target_id="target",
                phase=0,
                success=False,
            ),
        ),
        rewards=(-1.0,),
        successes=(False,),
        terminals=(False,),
    )
    outcome = adapter.execute_imagined_rollout_step(
        environment,
        available_actions=(10, 11),
        action_kinds=("assemble", "idle"),
        learn=False,
        learn_world=True,
    )
    trace = adapter.cognitive_snapshot().world_calibration_trace[-1]
    return {
        "failure_success": outcome.success,
        "failure_adjudication": trace.adjudication,
        "failure_replan": bool(
            outcome.success is False
            and trace.adjudication == "accepted"
            and adapter.replan_required
            and adapter._planned_rollout is None
        ),
    }


def _run_conflicted_case(seed: int) -> dict[str, object]:
    learner = _seed_conflicted_ledger(seed + 2000)
    adapter = TSKV8Adapter(_config(seed), episode_id=f"risk-conflicted:{seed}")
    _attach_runtime(adapter, learner, seed + 2000)
    steps = _rollout_steps(adapter, prefix=f"conflicted:{seed}", action_symbol=10)
    rollout = adapter.imagine_world_rollout(f"conflicted-rollout:{seed}", "reach-world", steps)
    adapter.plan_rollouts((rollout,))
    initial = adapter.cognitive_snapshot().world
    environment = _ScriptedEnvironment(
        (
            _transition_state(
                initial,
                sample_id=f"conflicted:{seed}",
                target_id="target",
                phase=0,
                success=True,
            ),
        ),
        rewards=(1.0,),
        successes=(True,),
        terminals=(False,),
    )
    outcome = adapter.execute_imagined_rollout_step(
        environment,
        available_actions=(10, 11),
        action_kinds=("assemble", "idle"),
        learn=False,
        learn_world=True,
    )
    trace = adapter.cognitive_snapshot().world_calibration_trace[-1]
    return {
        "conflicted_mode_before": rollout.steps[0].uncertainty_mode,
        "conflicted_uncertainty_before": rollout.steps[0].uncertainty,
        "conflicted_adjudication": trace.adjudication,
        "conflicted_ledger_mode_after": trace.ledger_uncertainty_mode,
        "conflicted_replan": bool(
            outcome.success is True
            and rollout.steps[0].uncertainty_mode == "conflicted"
            and trace.adjudication == "rejected"
            and trace.ledger_uncertainty_mode == "conflicted"
            and adapter.replan_required
            and adapter._planned_rollout is None
        ),
    }


def evaluate_seed(seed: int) -> dict[str, object]:
    ambiguity = _run_ambiguity_case(seed)
    failure = _run_failure_case(seed)
    conflicted = _run_conflicted_case(seed)
    booleans = (
        "ambiguity_replan",
        "checkpoint_no_replay",
        "checkpoint_branch_preserved",
        "recovery_branch_filters_rejected_action",
        "recovery_branch_selects_low_risk_counterfactual",
        "recovery_candidates_generated_from_affordances",
        "recovery_lineage_recorded",
        "stale_planning_rejected",
        "stale_content_rejected",
        "stale_action_rejected",
        "stale_execution_rejected",
        "checkpoint_lineage_preserved",
        "recovery_complete",
        "trace_complete",
        "checkpoint_trace_complete",
        "checkpoint_capability_preserved",
        "capability_refresh_filters_next_candidates",
        "failure_replan",
        "conflicted_replan",
    )
    return {
        "seed": int(seed),
        **ambiguity,
        **failure,
        **conflicted,
        "gate_passed": all(
            bool(
                ambiguity[name]
                if name in ambiguity
                else failure[name] if name in failure else conflicted[name]
            )
            for name in booleans
        ),
    }


def build_manifest(seeds: tuple[int, ...] = SEEDS) -> dict[str, object]:
    return {
        "format": MANIFEST_FORMAT,
        "task": "execute ledger-aware imagined actions through a real environment step by step",
        "seeds": list(seeds),
        "controls": [
            "stochastic-ledger-ambiguity",
            "conflicted-ledger-ambiguity",
            "real-environment-after-state-feedback",
            "failed-action-replan",
            "taiji-owned-affordance-synthesis",
            "motor-capability-filter",
            "resource-budget-filter",
            "environment-capability-discovery",
            "capability-refresh-after-step",
            "recovery-lineage-freshness",
            "stale-plan-rejection",
            "stale-affordance-content-rejection",
            "stale-action-semantic-rejection",
            "stale-execution-rejection",
            "recovery-lineage-checkpoint",
            "checkpoint-no-replay",
            "recovery-rollout-continuation",
        ],
        "boundary": "risk-sensitive execution and evidence accounting; not open-world intelligence",
    }


def evaluate(seeds: tuple[int, ...] = SEEDS) -> dict[str, object]:
    runs = [evaluate_seed(seed) for seed in seeds]
    passed = all(bool(run["gate_passed"]) for run in runs)
    return {
        "format": REPORT_FORMAT,
        "manifest_format": MANIFEST_FORMAT,
        "metrics": {
            "cross_seed_gate_rate": sum(bool(run["gate_passed"]) for run in runs) / len(runs),
            "runs": runs,
        },
        "gate": {
            "passed": passed,
            "criterion": "all seeds must replan on stochastic and conflicted ledger ambiguity plus failed non-terminal action, synthesize alternatives from affordances, enforce motor capability and resource budget, consume the current environment-reported capability, record capability and schema lineage on each recovery rollout, reject stale plans before planning and execution, preserve and restore that lineage through checkpoint, refresh capability after a step so next candidates cannot exceed the new boundary, filter the rejected branch, choose the lower-risk deterministic alternative over an unseen counterfactual, record both adjudications in the trace, and complete an explicit recovery rollout",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "taiji_p3_risk_sensitive_execution_manifest_20260827.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p3_risk_sensitive_execution_20260827.json",
    )
    args = parser.parse_args()
    report = evaluate()
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(build_manifest(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
