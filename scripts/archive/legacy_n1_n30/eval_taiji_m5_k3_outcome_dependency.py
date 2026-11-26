"""M5.K3 canary: real outcome feedback drives a dependent task chain.

The preregistered contract lives in
``plans/reference/M5_K3_OUTCOME_WORLD_DEPENDENCY_PREREGISTRATION_20260909.md``.
K2's semantic and transition assets are reused as training primitives, but the
transition corpus here adds only stable outcome-class facts.  Dynamic event and
dependency digests remain lineage evidence and are never learned as facts.

This is a shadow canary.  It never changes the default runtime and always
reports ``can_promote=false``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.seed_runtime import SeedRuntime  # noqa: E402
from scripts.training.eval_taiji_m5_k1_skill_composition import (  # noqa: E402
    EXTENSIONS,
    FILES_PER_LANG,
    LANGS,
    READ_ONLY_ROUTES,
    _build_workspace,
    _goal_content,
    _observe_all,
    _schema,
    _typed_fact_feature_masks,
    _world,
)
from scripts.training.eval_taiji_m5_k2_multistep_composition import (  # noqa: E402
    _semantic_and_transition_corpora,
)
from scripts.training.eval_taiji_m5_s6b_failure_admission import (  # noqa: E402
    _graded_reward,
)
from seed import Seed  # noqa: E402
from seed.config import SeedConfig  # noqa: E402
from seed_platform.programming_languages import ProgrammingLanguageRegistry  # noqa: E402
from seed_platform.workbench import WorkbenchEnvironment  # noqa: E402
from taiji import (  # noqa: E402
    ContentPlan,
    Goal,
    InternalizationConverter,
    InternalizationLedger,
    NativeReadOnlyIntentPlanner,
    OutcomeDependencyProjector,
    OutcomeDependencySpec,
    ReadOnlyIntentPolicy,
    StructuredSemanticCorpus,
    StructuredSemanticLearner,
    StructuredSemanticTransitionCorpus,
    StructuredSemanticTransitionExample,
    StructuredSemanticTransitionLearner,
    TaijiConfig,
    WorkbenchObservation,
    WorldEvent,
    WorldState,
)

REPORT_FORMAT = "taiji-m5-k3-outcome-dependency-v1"
VERSION = 1
SEMANTIC_EPOCHS = 160
SEMANTIC_LR = 2.0
# K3 adds typed outcome rows to the K2 transition head.  The local SGD rule
# normalizes the delta over the complete sparse output, so rare language rows
# need a longer fixed budget to cross the learner's 0.55 materialization gate.
# This is a training-budget correction, not a prediction-threshold change.
TRANSITION_EPOCHS = 1280
TRANSITION_LR = 0.2
FEEDBACK_SUBJECTS = frozenset({"outcome", "dependency"})
OUTCOME_FACTS = frozenset(
    {
        ("outcome", "class", "success"),
        ("outcome", "class", "failure"),
    }
)


def _create_isolated_temp_root() -> tuple[Path, Path, bool]:
    parent = PROJECT_ROOT / ".tmp-m5-k3"
    configured = os.environ.get("SEED_M5_K3_TMPDIR")
    if configured:
        parent = Path(configured)
    parent_created = not parent.exists()
    parent.mkdir(parents=True, exist_ok=True)
    root = parent / f"taiji_m5_k3_{uuid4().hex}"
    root.mkdir()
    return root, parent, parent_created


def _all_paths() -> list[str]:
    paths = ["missing_00.txt"]
    for index in range(FILES_PER_LANG):
        paths.extend(f"{language}_{index:02d}{EXTENSIONS[language]}" for language in LANGS)
    return paths


def _k3_registry() -> ProgrammingLanguageRegistry:
    """Expose every fixture language so K3 isolates outcome dependency."""

    command = Path(sys.executable).name
    return ProgrammingLanguageRegistry(
        tuple(
            (
                replace(definition, toolchain_commands=(command,))
                if definition.language_id in LANGS
                else definition
            )
            for definition in ProgrammingLanguageRegistry.default().definitions
        )
    )


def _model_world(world: WorldState) -> WorldState:
    """Remove nonsemantic dependency metadata before the strict planner gate."""

    return replace(
        world,
        relations=tuple(
            relation for relation in world.relations if relation[0] not in FEEDBACK_SUBJECTS
        ),
    )


def _world_with_feedback(
    world: WorldState,
    outcome_class: str,
) -> WorldState:
    if outcome_class not in {"success", "failure"}:
        raise ValueError("outcome_class must be success or failure")
    return replace(
        world,
        relations=(
            *tuple(
                relation for relation in world.relations if relation[0] not in FEEDBACK_SUBJECTS
            ),
            ("outcome", "class", outcome_class),
        ),
    )


def _branch_goal_content(
    observation: WorkbenchObservation,
    *,
    branch: str,
    tick: int,
) -> tuple[Goal, ContentPlan]:
    if branch == "success":
        return _goal_content(observation, tick=tick)
    if branch != "failure":
        raise ValueError("unsupported K3 branch")
    goal = Goal(
        goal_id="goal:dependency-recovery",
        description="Recover through the failure-dependent read-only follow-up.",
        priority=0.9,
    )
    content = ContentPlan(
        content_id="content:clarify-toolchain",
        intent_id="intent:dependency-recovery",
        intent_kind="request_information",
        semantic_slots={
            "dependency_outcome": "failure",
            "observed_language": observation.language_id,
            "selection_state": observation.selection_state,
        },
        source_goal_id=goal.goal_id,
        expected_outcome="failure-dependent read-only recovery",
        confidence=0.95,
        tick=tick,
    )
    return goal, content


def _transition_masks(
    fact_keys: tuple[str, ...],
    schema: Any,
    *,
    event_dim: int,
) -> dict[str, tuple[int, ...]]:
    """Bind base facts by typed event features and outcome facts by persistence."""

    fact_count = len(fact_keys)
    context_dim = int(event_dim) + 4
    masks: dict[str, tuple[int, ...]] = {}
    for row, key in enumerate(fact_keys):
        if key.startswith("workbench::"):
            feature_index = _typed_fact_feature_masks((key,), schema)[key][0]
            masks[key] = (
                row,
                fact_count + feature_index,
                fact_count + context_dim + row * context_dim + feature_index,
            )
        elif key in {
            "outcome::class::success",
            "outcome::class::failure",
        }:
            # The real outcome is projected between actions.  The transition
            # owner may preserve that typed state, but cannot invent it from a
            # current percept event.
            masks[key] = (row,)
        else:
            raise ValueError(f"unexpected K3 transition fact: {key}")
    return masks


def _transition_example(
    *,
    example_id: str,
    family_id: str,
    before: WorldState,
    observation: WorkbenchObservation,
    after: WorldState,
    goal: Goal,
    content: ContentPlan,
    tick: int,
) -> StructuredSemanticTransitionExample:
    return StructuredSemanticTransitionExample(
        example_id=example_id,
        family_id=family_id,
        before=before,
        event=observation.to_percept_event(tick=tick),
        after=after,
        goal=goal,
        content=content,
    )


def _build_transition_corpus(
    observations: dict[str, WorkbenchObservation],
    anchor: WorkbenchObservation,
    *,
    train_sequences: tuple[tuple[str, str, str], ...],
    dev_sequence: tuple[str, str, str],
    test_sequence: tuple[str, str, str],
    schema: Any,
) -> StructuredSemanticTransitionCorpus:
    def build(
        sequences: tuple[tuple[str, str, str], ...],
        *,
        split: str,
    ) -> tuple[StructuredSemanticTransitionExample, ...]:
        examples: list[StructuredSemanticTransitionExample] = []
        for sequence_index, paths in enumerate(sequences):
            probe, followup, verification = (observations[path] for path in paths)
            examples.append(
                _transition_example(
                    example_id=f"m5k3:{split}:{sequence_index}:probe",
                    family_id=f"m5k3:{split}:{sequence_index}:probe",
                    before=_world(anchor, tick=0),
                    observation=probe,
                    after=_world(probe, tick=1),
                    goal=_goal_content(probe, tick=1)[0],
                    content=_goal_content(probe, tick=1)[1],
                    tick=1,
                )
            )
            for branch in ("success", "failure"):
                before_followup = _world_with_feedback(_world(probe, tick=1), branch)
                after_followup = _world_with_feedback(_world(followup, tick=2), branch)
                follow_goal, follow_content = _branch_goal_content(
                    followup,
                    branch=branch,
                    tick=2,
                )
                examples.append(
                    _transition_example(
                        example_id=f"m5k3:{split}:{sequence_index}:{branch}:followup",
                        family_id=f"m5k3:{split}:{sequence_index}:{branch}",
                        before=before_followup,
                        observation=followup,
                        after=after_followup,
                        goal=follow_goal,
                        content=follow_content,
                        tick=2,
                    )
                )
                verify_goal, verify_content = _branch_goal_content(
                    verification,
                    branch=branch,
                    tick=3,
                )
                examples.append(
                    _transition_example(
                        example_id=f"m5k3:{split}:{sequence_index}:{branch}:verify",
                        family_id=f"m5k3:{split}:{sequence_index}:{branch}",
                        before=after_followup,
                        observation=verification,
                        after=_world_with_feedback(_world(verification, tick=3), branch),
                        goal=verify_goal,
                        content=verify_content,
                        tick=3,
                    )
                )
        return tuple(examples)

    return StructuredSemanticTransitionCorpus.from_splits(
        train=build(train_sequences, split="train"),
        dev=build((dev_sequence,), split="dev"),
        test=build((test_sequence,), split="test"),
    )


def _spec(branch: str, *, step: str) -> OutcomeDependencySpec:
    return OutcomeDependencySpec(
        dependency_id=f"m5k3:dependency:{step}",
        next_task_id=f"m5k3:{branch}:{step}",
        capability_id=(
            "workspace.read" if branch == "success" else "workspace.programming_language.resolve"
        ),
        required_outcome=branch,
    )


def _latest_workbench_event(runtime: SeedRuntime) -> WorldEvent:
    events = [
        event
        for event in runtime.model.architecture.cognitive_snapshot().world.events
        if event.kind == "workbench.evidence"
    ]
    if not events:
        raise RuntimeError("K3 execution did not record a Workbench evidence event")
    latest = events[-1]
    if not isinstance(latest, WorldEvent):
        raise TypeError("K3 Workbench evidence event is not a Taiji WorldEvent")
    return latest


def _admit_read_outcome(
    runtime: SeedRuntime,
    execution: dict[str, Any],
    *,
    parent_checkpoint_id: str,
) -> tuple[bool, float]:
    outcome = execution.get("outcome") or {}
    if outcome.get("capability_id") != "workspace.read":
        return False, 0.0
    result = outcome.get("result") or {}
    reward = float(_graded_reward(result))
    if not bool(outcome.get("success")):
        reward = min(-1e-6, reward)
    try:
        if bool(outcome.get("success")):
            reprojected = runtime.reproject_workbench_from_latest_evidence(
                snapshot_id=str(outcome["snapshot_id"])
            )
            affordance_id = str(reprojected["affordances"][0]["affordance_id"])
        else:
            affordance_id = "workbench-failed:auto"
        projected = runtime.project_workbench_outcome_for_internalization(
            snapshot_id=str(outcome["snapshot_id"]),
            affordance_id=affordance_id,
            reward=reward,
            reward_terms={"task_success": reward},
            parent_checkpoint_id=parent_checkpoint_id,
        )
        admitted = InternalizationLedger(
            converter=InternalizationConverter(seed=17, replay_budget=64)
        ).ingest(projected)
        return bool(admitted.accepted), reward
    except (KeyError, ValueError, RuntimeError):
        return False, reward


def _runtime(
    *,
    temp_root: Path,
    registry: ProgrammingLanguageRegistry,
    learner_seed: int,
    episode_id: str,
) -> SeedRuntime:
    runtime = SeedRuntime(
        Seed(
            SeedConfig(taiji=TaijiConfig(seed=learner_seed)),
            episode_id=episode_id,
        )
    )
    runtime._workbench_environment = WorkbenchEnvironment(
        root=temp_root,
        programming_language_registry=registry,
    )
    return runtime


def _run_episode(
    *,
    arm: str,
    branch: str,
    sequence: tuple[WorkbenchObservation, WorkbenchObservation, WorkbenchObservation],
    initial_world: WorldState,
    semantic_learner: StructuredSemanticLearner,
    transition_learner: StructuredSemanticTransitionLearner,
    planner: NativeReadOnlyIntentPlanner,
    runtime: SeedRuntime,
    projector: OutcomeDependencyProjector | None,
    temp_root: Path,
    parent_checkpoint_id: str,
) -> dict[str, Any]:
    probe, followup, verification = sequence
    rows: list[dict[str, Any]] = []
    current_world = initial_world
    projection = None
    dependency_token = ""
    expected_content = (
        "content:inspect-language" if branch == "success" else "content:clarify-toolchain"
    )
    expected_kind = (
        "workspace.read" if branch == "success" else "workspace.programming_language.resolve"
    )

    # Step 1: Stage-1 semantic output still chooses the first read-only intent.
    probe_event = probe.to_percept_event(tick=1)
    probe_transition = transition_learner.predict(current_world, probe_event)
    probe_semantic = semantic_learner.predict(probe_event)
    probe_row: dict[str, Any] = {
        "step": "probe",
        "path": probe.path,
        "branch": branch,
        "transition_status": probe_transition.status,
        "semantic_status": probe_semantic.status,
        "predicted_content_id": (
            None if probe_semantic.content_plan is None else probe_semantic.content_plan.content_id
        ),
        "dependency_gate": True,
        "outcome_admitted": False,
    }
    if (
        probe_transition.world is None
        or probe_semantic.goal is None
        or probe_semantic.content_plan is None
    ):
        probe_row["step_success"] = False
        rows.append(probe_row)
        return {"rows": rows, "episode_success": False}
    probe_decision = planner.propose(
        observation=probe,
        world=_model_world(probe_transition.world),
        goal=probe_semantic.goal,
        content=probe_semantic.content_plan,
        capability_snapshot=runtime.workbench_environment.capability_snapshot,
        tick=1,
    )
    probe_row["accepted"] = bool(probe_decision.accepted)
    probe_row["decision_reason"] = probe_decision.reason_code
    probe_row["intent_kind"] = (
        None if probe_decision.action_intent is None else probe_decision.action_intent.kind
    )
    if not probe_decision.accepted or probe_decision.action_intent is None:
        probe_row["step_success"] = False
        rows.append(probe_row)
        return {"rows": rows, "episode_success": False}

    probe_file = temp_root / probe.path
    original_probe_bytes = probe_file.read_bytes()
    if branch == "failure":
        probe_file.unlink()
    try:
        execution = runtime.execute_workbench_intent(
            probe_decision.action_intent,
            snapshot_id=runtime.workbench_environment.capability_snapshot.snapshot_id,
            learn=False,
        )
    finally:
        if branch == "failure":
            probe_file.write_bytes(original_probe_bytes)
    real_success = bool((execution.get("outcome") or {}).get("success"))
    admitted, reward = _admit_read_outcome(
        runtime,
        execution,
        parent_checkpoint_id=parent_checkpoint_id,
    )
    expected_probe_success = branch == "success"
    probe_row.update(
        {
            "real_success": real_success,
            "expected_probe_success": expected_probe_success,
            "outcome_matches_expected": real_success == expected_probe_success,
            "outcome_admitted": admitted,
            "real_reward": reward,
            "event_id": _latest_workbench_event(runtime).event_id,
        }
    )
    probe_row["step_success"] = bool(real_success == expected_probe_success and admitted)
    rows.append(probe_row)
    if not probe_row["step_success"] or probe_transition.world is None:
        return {"rows": rows, "episode_success": False}

    # The observed Workbench event is now the sole input to the feedback owner.
    event = _latest_workbench_event(runtime)
    if projector is not None:
        projection = projector.project(
            probe_transition.world,
            event,
            _spec(branch, step="followup"),
        )
        probe_row["projection_reason"] = projection.reason_code
        probe_row["projection_accepted"] = projection.accepted
        if projection.accepted:
            enriched = projector.apply(probe_transition.world, projection)
            current_world = replace(
                enriched,
                relations=tuple(
                    relation
                    for relation in enriched.relations
                    if relation in OUTCOME_FACTS or relation[0] == "workbench"
                ),
            )
            dependency_token = projection.dependency_digest
    else:
        probe_row["projection_reason"] = "no_feedback"
        probe_row["projection_accepted"] = False
        current_world = probe_transition.world

    for step_number, observation in ((2, followup), (3, verification)):
        event_input = observation.to_percept_event(tick=step_number)
        transition = transition_learner.predict(current_world, event_input)
        feedback_input = any(relation in OUTCOME_FACTS for relation in current_world.relations)
        row: dict[str, Any] = {
            "step": "followup" if step_number == 2 else "verification",
            "path": observation.path,
            "branch": branch,
            "transition_status": transition.status,
            "feedback_input_present": feedback_input,
            "predicted_content_id": (
                None if transition.content_plan is None else transition.content_plan.content_id
            ),
            "expected_content_id": expected_content,
            "expected_intent_kind": expected_kind,
            "dependency_projection_accepted": bool(projection is not None and projection.accepted),
        }
        if transition.world is None or transition.goal is None or transition.content_plan is None:
            row["step_success"] = False
            rows.append(row)
            break
        decision = planner.propose(
            observation=observation,
            world=_model_world(transition.world),
            goal=transition.goal,
            content=transition.content_plan,
            capability_snapshot=runtime.workbench_environment.capability_snapshot,
            tick=step_number,
        )
        row["accepted"] = bool(decision.accepted)
        row["decision_reason"] = decision.reason_code
        row["intent_kind"] = None if decision.action_intent is None else decision.action_intent.kind
        dependency_gate = bool(
            projector is not None
            and projection is not None
            and projection.accepted
            and feedback_input
            and transition.content_plan.content_id == expected_content
            and decision.accepted
            and decision.action_intent is not None
            and decision.action_intent.kind == expected_kind
            and dependency_token
        )
        row["dependency_gate"] = dependency_gate
        if not dependency_gate or decision.action_intent is None:
            row["step_success"] = False
            rows.append(row)
            break
        execution = runtime.execute_workbench_intent(
            decision.action_intent,
            snapshot_id=runtime.workbench_environment.capability_snapshot.snapshot_id,
            learn=False,
            boundary_token_digest=dependency_token,
        )
        outcome = execution.get("outcome") or {}
        boundary_echo = str((execution.get("request") or {}).get("boundary_token_digest", ""))
        row.update(
            {
                "real_success": bool(outcome.get("success")),
                "boundary_token_echo": boundary_echo,
                "lineage_matches": boundary_echo == dependency_token,
                "real_reward": float((execution.get("taiji_outcome") or {}).get("reward", 0.0)),
            }
        )
        row["step_success"] = bool(outcome.get("success") and row["lineage_matches"])
        rows.append(row)
        if not row["step_success"]:
            break
        current_world = transition.world

    return {
        "rows": rows,
        "episode_success": bool(len(rows) == 3 and all(row.get("step_success") for row in rows)),
    }


def _run_arm(
    *,
    arm: str,
    sequences: tuple[tuple[str, tuple[str, str, str]], ...],
    observation_map: dict[str, WorkbenchObservation],
    anchor: WorkbenchObservation,
    semantic_learner: StructuredSemanticLearner,
    transition_learner: StructuredSemanticTransitionLearner,
    planner: NativeReadOnlyIntentPlanner,
    temp_root: Path,
    registry: ProgrammingLanguageRegistry,
    learner_seed: int,
    task_seed: int,
    workspace_root_holder: dict[str, Path] | None = None,
) -> dict[str, Any]:
    if workspace_root_holder is not None:
        workspace_root_holder["path"] = temp_root
    episodes: list[dict[str, Any]] = []
    for episode_index, (branch, paths) in enumerate(sequences):
        runtime = _runtime(
            temp_root=temp_root,
            registry=registry,
            learner_seed=learner_seed,
            episode_id=f"m5-k3-{arm}-{task_seed}-{learner_seed}-{episode_index}",
        )
        projector: OutcomeDependencyProjector | None
        if arm == "A-full-feedback":
            projector = OutcomeDependencyProjector(
                f"m5-k3-episode-{arm}-{task_seed}-{learner_seed}-{episode_index}"
            )
        elif arm == "C-outcome-lesion":
            projector = OutcomeDependencyProjector(
                f"m5-k3-episode-{arm}-{task_seed}-{learner_seed}-{episode_index}",
                lesioned=True,
            )
        else:
            projector = None
        episodes.append(
            _run_episode(
                arm=arm,
                branch=branch,
                sequence=(
                    observation_map[paths[0]],
                    observation_map[paths[1]],
                    observation_map[paths[2]],
                ),
                initial_world=_world(anchor, tick=0),
                semantic_learner=semantic_learner,
                transition_learner=transition_learner,
                planner=planner,
                runtime=runtime,
                projector=projector,
                temp_root=temp_root,
                parent_checkpoint_id="checkpoint:k3-parent",
            )
        )
    probe_rows = [episode["rows"][0] if episode["rows"] else {} for episode in episodes]
    feedback_rows = [
        row
        for episode in episodes
        for row in episode["rows"]
        if row.get("step") in {"followup", "verification"} and row.get("dependency_gate")
    ]
    return {
        "episodes": episodes,
        "episode_success_rate": sum(1 for episode in episodes if episode["episode_success"])
        / max(1, len(episodes)),
        "probe_outcome_admission_rate": sum(
            1 for row in probe_rows if row.get("outcome_admitted", False)
        )
        / max(1, len(episodes)),
        "all_probe_outcomes_admitted": bool(
            episodes
            and len(probe_rows) == len(episodes)
            and all(row.get("outcome_admitted", False) for row in probe_rows)
        ),
        "feedback_lineage_admission_rate": sum(
            1 for row in feedback_rows if row.get("lineage_matches", False)
        )
        / max(1, len(feedback_rows)),
        "all_feedback_lineages_admitted": bool(
            feedback_rows and all(row.get("lineage_matches", False) for row in feedback_rows)
        ),
        "feedback_row_count": len(feedback_rows),
    }


def _checkpoint_gate(
    semantic: StructuredSemanticLearner,
    semantic_corpus: StructuredSemanticCorpus,
    transition: StructuredSemanticTransitionLearner,
    transition_corpus: StructuredSemanticTransitionCorpus,
    projector: OutcomeDependencyProjector,
) -> dict[str, Any]:
    semantic_restored = StructuredSemanticLearner.from_checkpoint(
        semantic.checkpoint(), semantic_corpus
    )
    transition_restored = StructuredSemanticTransitionLearner.from_checkpoint(
        transition.checkpoint(), transition_corpus
    )
    projector_restored = OutcomeDependencyProjector.from_checkpoint(projector.checkpoint())
    return {
        "semantic_owner_digest_equal": semantic.owner_digests()
        == semantic_restored.owner_digests(),
        "transition_owner_digest_equal": transition.owner_digests()
        == transition_restored.owner_digests(),
        "projector_checkpoint_equal": projector.checkpoint() == projector_restored.checkpoint(),
        "transition_masks_restored": transition.transition_input_masks
        == transition_restored.transition_input_masks,
    }


def _prefit_checkpoint_gate(
    semantic: StructuredSemanticLearner,
    semantic_corpus: StructuredSemanticCorpus,
    transition: StructuredSemanticTransitionLearner,
    transition_corpus: StructuredSemanticTransitionCorpus,
) -> dict[str, bool]:
    semantic_restored = StructuredSemanticLearner.from_checkpoint(
        semantic.checkpoint(), semantic_corpus
    )
    transition_restored = StructuredSemanticTransitionLearner.from_checkpoint(
        transition.checkpoint(), transition_corpus
    )
    return {
        "semantic_prefit_roundtrip": semantic.owner_digests() == semantic_restored.owner_digests(),
        "transition_prefit_roundtrip": transition.owner_digests()
        == transition_restored.owner_digests(),
    }


def run_cell(*, task_seed: int, learner_seed: int) -> dict[str, Any]:
    started = time.perf_counter()
    temp_root, temp_parent, temp_parent_created = _create_isolated_temp_root()
    try:
        import seed_platform.workbench as workbench_module

        original_get_setting = workbench_module.get_setting
        active_workspace_root = {"path": temp_root}
        workbench_module.get_setting = lambda key, default=None: (
            str(active_workspace_root["path"]) if key == "workspace_path" else default
        )
        try:
            train_root = temp_root / "train"
            holdout_root = temp_root / "holdout"
            train_root.mkdir()
            holdout_root.mkdir()
            holdout_task_seed = int(task_seed) + 1000
            _build_workspace(train_root, task_seed=task_seed)
            _build_workspace(holdout_root, task_seed=holdout_task_seed)
            schema = _schema()
            # K3 measures outcome/dependency feedback, not open-set language
            # discovery.  Every language in the holdout must therefore exist
            # in the transition vocabulary; otherwise an unknown language
            # failure would be misreported as a projection failure.
            train_registry = _k3_registry()
            holdout_registry = _k3_registry()
            paths = _all_paths()
            train_observations = {
                observation.path: observation
                for observation in _observe_all(
                    train_root,
                    registry=train_registry,
                    split="k3-train",
                    paths=paths,
                    schema=schema,
                )
            }
            anchor = train_observations["missing_00.txt"]
            semantic_corpus = _semantic_and_transition_corpora(
                train_observations,
                anchor,
                schema,
            )[0]
            fact_masks = _typed_fact_feature_masks(semantic_corpus.fact_keys, schema)
            readout_excluded = tuple(
                key for key in semantic_corpus.fact_keys if key.split("::")[1] == "language"
            )
            semantic_learner = StructuredSemanticLearner(
                semantic_corpus,
                fact_feature_masks=fact_masks,
                readout_excluded_facts=readout_excluded,
            )

            train_sequences = (
                ("success", ("python_00.py", "rust_00.rs", "python_01.py")),
                ("failure", ("rust_01.rs", "python_02.py", "typescript_00.ts")),
                ("success", ("typescript_01.ts", "rust_02.rs", "python_03.py")),
                ("failure", ("python_02.py", "typescript_02.ts", "rust_03.rs")),
                ("success", ("rust_00.rs", "typescript_03.ts", "python_00.py")),
                ("failure", ("typescript_02.ts", "python_01.py", "rust_01.rs")),
            )
            dev_sequence = ("python_04.py", "python_05.py", "python_06.py")
            test_sequence = ("rust_04.rs", "python_06.py", "python_07.py")
            transition_corpus = _build_transition_corpus(
                train_observations,
                anchor,
                train_sequences=tuple(paths for _, paths in train_sequences),
                dev_sequence=dev_sequence,
                test_sequence=test_sequence,
                schema=schema,
            )
            transition_masks = _transition_masks(
                transition_corpus.fact_keys,
                schema,
                event_dim=transition_corpus.event_dim,
            )
            transition_learner = StructuredSemanticTransitionLearner(
                transition_corpus,
                transition_input_masks=transition_masks,
            )
            prefit_checkpoint = _prefit_checkpoint_gate(
                semantic_learner,
                semantic_corpus,
                transition_learner,
                transition_corpus,
            )
            semantic_losses = semantic_learner.fit(
                semantic_corpus.train,
                epochs=SEMANTIC_EPOCHS,
                learning_rate=SEMANTIC_LR,
            )
            transition_losses = transition_learner.fit(
                transition_corpus.train,
                epochs=TRANSITION_EPOCHS,
                learning_rate=TRANSITION_LR,
            )
            holdout_paths = sorted(
                {
                    path
                    for _, sequence in (
                        ("success", ("typescript_05.ts", "python_05.py", "rust_05.rs")),
                        ("failure", ("rust_06.rs", "typescript_06.ts", "python_06.py")),
                        ("success", ("python_07.py", "rust_07.rs", "typescript_07.ts")),
                        ("failure", ("typescript_05.ts", "rust_06.rs", "python_07.py")),
                    )
                    for path in sequence
                }
            )
            holdout_observations = {
                observation.path: observation
                for observation in _observe_all(
                    holdout_root,
                    registry=holdout_registry,
                    split="k3-holdout",
                    paths=holdout_paths,
                    schema=schema,
                )
            }
            holdout_anchor = _observe_all(
                holdout_root,
                registry=holdout_registry,
                split="k3-holdout-anchor",
                paths=["missing_00.txt"],
                schema=schema,
            )[0]
            holdout_sequences = (
                ("success", ("typescript_05.ts", "python_05.py", "rust_05.rs")),
                ("failure", ("rust_06.rs", "typescript_06.ts", "python_06.py")),
                ("success", ("python_07.py", "rust_07.rs", "typescript_07.ts")),
                ("failure", ("typescript_05.ts", "rust_06.rs", "python_07.py")),
            )
            planner = NativeReadOnlyIntentPlanner(ReadOnlyIntentPolicy(routes=READ_ONLY_ROUTES))
            train_arm = _run_arm(
                arm="A-full-feedback",
                sequences=train_sequences,
                observation_map=train_observations,
                anchor=anchor,
                semantic_learner=semantic_learner,
                transition_learner=transition_learner,
                planner=planner,
                temp_root=train_root,
                registry=train_registry,
                learner_seed=learner_seed,
                task_seed=task_seed,
                workspace_root_holder=active_workspace_root,
            )
            arms = {
                arm: _run_arm(
                    arm=arm,
                    sequences=holdout_sequences,
                    observation_map=holdout_observations,
                    anchor=holdout_anchor,
                    semantic_learner=semantic_learner,
                    transition_learner=transition_learner,
                    planner=planner,
                    temp_root=holdout_root,
                    registry=holdout_registry,
                    learner_seed=learner_seed,
                    task_seed=task_seed,
                    workspace_root_holder=active_workspace_root,
                )
                for arm in ("A-full-feedback", "B-no-feedback", "C-outcome-lesion")
            }
            checkpoint_gate = _checkpoint_gate(
                semantic_learner,
                semantic_corpus,
                transition_learner,
                transition_corpus,
                OutcomeDependencyProjector("checkpoint-k3"),
            )
            probe_rewards = [
                float(row.get("real_reward", 0.0))
                for episode in train_arm["episodes"] + arms["A-full-feedback"]["episodes"]
                for row in episode["rows"]
                if row.get("step") == "probe"
            ]
            reward_mean = sum(probe_rewards) / max(1, len(probe_rewards))
            reward_variance = sum((value - reward_mean) ** 2 for value in probe_rewards) / max(
                1, len(probe_rewards)
            )
            a_rate = arms["A-full-feedback"]["episode_success_rate"]
            b_rate = arms["B-no-feedback"]["episode_success_rate"]
            c_rate = arms["C-outcome-lesion"]["episode_success_rate"]
            checks = {
                "a_holdout_success_at_least_0p75": a_rate >= 0.75,
                "a_minus_b_at_least_0p25": (a_rate - b_rate) >= 0.25,
                "a_minus_c_at_least_0p25": (a_rate - c_rate) >= 0.25,
                "a_train_success_is_1": train_arm["episode_success_rate"] == 1.0,
                "train_episode_count_at_least_6": len(train_arm["episodes"]) >= 6,
                "a_probe_outcomes_admitted": arms["A-full-feedback"]["all_probe_outcomes_admitted"],
                "a_feedback_lineages_admitted": arms["A-full-feedback"][
                    "all_feedback_lineages_admitted"
                ],
                "feedback_reward_variance_positive": reward_variance > 1e-12,
                "prefit_checkpoint_gate": all(prefit_checkpoint.values()),
                "postfit_checkpoint_gate": all(checkpoint_gate.values()),
                "a_model_feedback_facts_consumed": all(
                    row.get("feedback_input_present", False)
                    and row.get("predicted_content_id") == row.get("expected_content_id")
                    for episode in arms["A-full-feedback"]["episodes"]
                    for row in episode["rows"]
                    if row.get("step") in {"followup", "verification"}
                ),
            }
            return {
                "task_seed": int(task_seed),
                "learner_seed": int(learner_seed),
                "arms": arms,
                "train_arm": train_arm,
                "prefit_checkpoint": prefit_checkpoint,
                "checkpoint_gate": checkpoint_gate,
                "typed_binding": {
                    "transition_input_masks": {
                        key: list(indices) for key, indices in transition_masks.items()
                    },
                    "outcome_fact_keys": sorted(OUTCOME_FACTS),
                },
                "stage1_semantic_fit_losses": {
                    key: round(float(value), 8) for key, value in semantic_losses.items()
                },
                "stage2_transition_fit_losses": {
                    key: round(float(value), 8) for key, value in transition_losses.items()
                },
                "metrics": {
                    "a_holdout_success": a_rate,
                    "b_holdout_success": b_rate,
                    "c_holdout_success": c_rate,
                    "a_minus_b": a_rate - b_rate,
                    "a_minus_c": a_rate - c_rate,
                    "a_train_success": train_arm["episode_success_rate"],
                    "feedback_reward_variance": reward_variance,
                },
                "checks": checks,
                "technical_gate_all_passed": all(bool(value) for value in checks.values()),
                "elapsed_seconds": time.perf_counter() - started,
            }
        finally:
            workbench_module.get_setting = original_get_setting
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
        if temp_parent_created:
            with contextlib.suppress(OSError):
                temp_parent.rmdir()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-seed", type=int, default=0)
    parser.add_argument("--learner-seed", type=int, default=17)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    cell = run_cell(task_seed=args.task_seed, learner_seed=args.learner_seed)
    payload = {
        "format": REPORT_FORMAT,
        "version": VERSION,
        "generated_at_epoch": int(time.time()),
        "status": "passed" if cell["technical_gate_all_passed"] else "failed",
        "can_promote": False,
        "composition": "real Workbench outcome -> typed feedback projection -> transition-dependent read-only chain (S6B)",
        "holdout": "4 unseen branch x file combinations",
        "cell": cell,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": payload["status"], "report": str(args.report)}, ensure_ascii=False))
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
