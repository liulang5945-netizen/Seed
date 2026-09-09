"""M5.K2 canary: autoregressive multi-step skill composition.

This canary is the first course that makes the native transition learner part
of the execution path.  The semantic learner supplies the goal and content
plan; the transition learner must predict the next world from the previous
predicted world and the current percept; the read-only planner then consumes
that predicted world before the real isolated Workbench execution.

The contract is preregistered in
``plans/reference/M5_K2_MULTISTEP_COMPOSITION_PREREGISTRATION_20260909.md``.
This file intentionally remains a canary harness: it never promotes the
architecture and never changes the K1 thresholds.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import time
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
    _observe_all,
    _registry,
    _schema,
    _semantic_example,
    _transition_examples,
    _typed_fact_feature_masks,
    _world,
)
from scripts.training.eval_taiji_m5_s6b_failure_admission import (  # noqa: E402
    _graded_reward as _graded_read_reward,
)
from seed import Seed  # noqa: E402
from seed.config import SeedConfig  # noqa: E402
from seed_platform.programming_languages import (  # noqa: E402
    ProgrammingLanguageRegistry,
)
from seed_platform.workbench import WorkbenchEnvironment  # noqa: E402
from taiji import (  # noqa: E402
    InternalizationConverter,
    InternalizationLedger,
    NativeReadOnlyIntentPlanner,
    ReadOnlyIntentPolicy,
    StructuredSemanticCorpus,
    StructuredSemanticLearner,
    StructuredSemanticTransitionCorpus,
    StructuredSemanticTransitionLearner,
    TaijiConfig,
    WorkbenchObservation,
    WorldState,
)

REPORT_FORMAT = "taiji-m5-k2-multistep-composition-v1"
VERSION = 1
TRAIN_EPOCHS = 280
TRAIN_LR = 0.2
SEMANTIC_EPOCHS = 160
SEMANTIC_LR = 2.0


def _path(language: str, index: int) -> str:
    return f"{language}_{index:02d}{EXTENSIONS[language]}"


def _create_isolated_temp_root() -> tuple[Path, Path, bool]:
    """Create a writable process-private root on the managed Windows runner."""

    default_parent = PROJECT_ROOT / ".tmp-m5-k2"
    parent_value = os.environ.get("SEED_M5_K2_TMPDIR")
    parent = Path(parent_value) if parent_value else default_parent
    parent_created = not parent.exists()
    parent.mkdir(parents=True, exist_ok=True)
    root = parent / f"taiji_m5_k2_{uuid4().hex}"
    root.mkdir()
    return root, parent, parent_created


def _episode(anchor: WorkbenchObservation, observations: dict[str, WorkbenchObservation], paths: tuple[str, ...]) -> tuple[WorkbenchObservation, ...]:
    if len(paths) != 3:
        raise ValueError("K2 episodes must contain exactly three steps")
    return (anchor, *(observations[path] for path in paths))


def _train_episode_paths() -> tuple[tuple[str, ...], ...]:
    return (
        (_path("python", 0), _path("rust", 0), _path("python", 1)),
        (_path("rust", 1), _path("python", 2), _path("typescript", 0)),
        (_path("typescript", 1), _path("rust", 2), _path("python", 3)),
        (_path("python", 2), _path("typescript", 2), _path("rust", 3)),
        (_path("rust", 0), _path("typescript", 3), _path("python", 0)),
        (_path("typescript", 2), _path("python", 1), _path("rust", 1)),
    )


def _dev_episode_paths() -> tuple[str, ...]:
    return (_path("python", 4), _path("rust", 4), _path("typescript", 4))


def _test_episode_paths() -> tuple[str, ...]:
    return (_path("rust", 5), _path("typescript", 5), _path("python", 5))


def _holdout_episode_paths() -> tuple[tuple[str, ...], ...]:
    return (
        (_path("typescript", 5), _path("python", 5), _path("rust", 5)),
        (_path("rust", 6), _path("typescript", 6), _path("python", 6)),
        (_path("python", 7), _path("rust", 7), _path("typescript", 7)),
        (_path("typescript", 5), _path("rust", 6), _path("python", 7)),
    )


def _typed_transition_input_masks(
    fact_keys: tuple[str, ...], schema, *, event_dim: int
) -> dict[str, tuple[int, ...]]:
    """Bind each delta row to itself + its event attribute + their product.

    The first block is ``before``; the second is the event context; the third
    is ``before x event``.  This is the exact K2.1 contract, not a heuristic
    learned from the current course.
    """

    fact_count = len(fact_keys)
    context_dim = int(event_dim) + 4  # event features plus PerceptEvent metadata
    masks: dict[str, tuple[int, ...]] = {}
    for row, key in enumerate(fact_keys):
        feature_index = _typed_fact_feature_masks((key,), schema)[key][0]
        event_index = fact_count + feature_index
        interaction_index = (
            fact_count
            + context_dim
            + row * context_dim
            + feature_index
        )
        masks[key] = (row, event_index, interaction_index)
    return masks


def _mask_violations(
    learner: StructuredSemanticTransitionLearner,
) -> list[dict[str, int]]:
    masks = learner.transition_input_masks
    if masks is None:
        return []
    allowed = {key: set(indices) for key, indices in masks.items()}
    weight = learner.transition_head.weight.detach().cpu()
    violations: list[dict[str, int]] = []
    for row, key in enumerate(learner.fact_keys):
        for column, value in enumerate(weight[row]):
            if column not in allowed[key] and abs(float(value)) > 1e-9:
                violations.append({"row": row, "column": column})
    return violations


def _checkpoint_gate(
    learner: StructuredSemanticTransitionLearner,
    corpus: StructuredSemanticTransitionCorpus,
) -> dict[str, Any]:
    payload = learner.checkpoint()
    restored = StructuredSemanticTransitionLearner.from_checkpoint(payload, corpus)
    roundtrip_same = learner.owner_digests() == restored.owner_digests()
    masks_same = learner.transition_input_masks == restored.transition_input_masks

    tampered = copy.deepcopy(payload)
    weight = tampered["state_dict"]["transition_head.weight"]
    masks = learner.transition_input_masks or {}
    for row, key in enumerate(learner.fact_keys):
        allowed = set(masks[key])
        for column in range(weight.shape[1]):
            if column not in allowed:
                weight[row, column] = 7.0
    hardened = StructuredSemanticTransitionLearner.from_checkpoint(tampered, corpus)
    tamper_rejected_by_mask = not _mask_violations(hardened)
    return {
        "format": payload["format"],
        "version": payload["version"],
        "roundtrip_owner_digest_equal": roundtrip_same,
        "roundtrip_masks_equal": masks_same,
        "tampered_forbidden_weights_zeroed": tamper_rejected_by_mask,
        "post_fit_mask_violations": _mask_violations(learner),
        "passed": bool(
            roundtrip_same
            and masks_same
            and tamper_rejected_by_mask
            and not _mask_violations(learner)
        ),
    }


def _semantic_and_transition_corpora(
    observations: dict[str, WorkbenchObservation],
    anchor: WorkbenchObservation,
    schema,
) -> tuple[
    StructuredSemanticCorpus,
    StructuredSemanticTransitionCorpus,
    tuple[tuple[WorkbenchObservation, ...], ...],
    tuple[WorkbenchObservation, ...],
    tuple[WorkbenchObservation, ...],
]:
    train_paths: list[str] = [anchor.path]
    for episode_paths in _train_episode_paths():
        train_paths.extend(episode_paths)
    train_paths = list(dict.fromkeys(train_paths))
    dev_paths = list(_dev_episode_paths())
    test_paths = list(_test_episode_paths())

    semantic_corpus = StructuredSemanticCorpus.from_splits(
        train=tuple(
            _semantic_example(observations[path], split="train", tick=index + 1)
            for index, path in enumerate(train_paths)
        ),
        dev=tuple(
            _semantic_example(observations[path], split="dev", tick=index + 1)
            for index, path in enumerate(dev_paths)
        ),
        test=tuple(
            _semantic_example(observations[path], split="test", tick=index + 1)
            for index, path in enumerate(test_paths)
        ),
    )

    train_episodes = tuple(
        _episode(anchor, observations, paths) for paths in _train_episode_paths()
    )
    dev_episode = _episode(anchor, observations, _dev_episode_paths())
    test_episode = _episode(anchor, observations, _test_episode_paths())
    transition_corpus = StructuredSemanticTransitionCorpus.from_splits(
        train=tuple(
            item
            for index, sequence in enumerate(train_episodes)
            for item in _transition_examples(sequence, split=f"train-{index}")
        ),
        dev=_transition_examples(dev_episode, split="dev-0"),
        test=_transition_examples(test_episode, split="test-0"),
    )
    return (
        semantic_corpus,
        transition_corpus,
        train_episodes,
        dev_episode,
        test_episode,
    )


def _run_episode(
    *,
    semantic_learner: StructuredSemanticLearner,
    transition_learner: StructuredSemanticTransitionLearner,
    sequence: tuple[WorkbenchObservation, ...],
    initial_world: WorldState,
    planner: NativeReadOnlyIntentPlanner,
    runtime: SeedRuntime,
    parent_checkpoint_id: str,
) -> dict[str, Any]:
    def _runtime_freshness() -> dict[str, Any]:
        world = runtime.model.architecture.cognitive_snapshot().world
        workbench_events = [
            event
            for event in world.events
            if event.kind == "workbench.evidence"
        ]
        latest = workbench_events[-1] if workbench_events else None
        return {
            "world_tick": int(world.tick),
            "latest_workbench_event_tick": None if latest is None else int(latest.tick),
        }

    current_world = initial_world
    rows: list[dict[str, Any]] = []
    for tick, observation in enumerate(sequence[1:], start=1):
        event = observation.to_percept_event(tick=tick)
        semantic = semantic_learner.predict(event)
        transition = transition_learner.predict(current_world, event)
        decision = None
        if (
            transition.world is not None
            and semantic.goal is not None
            and semantic.content_plan is not None
        ):
            decision = planner.propose(
                observation=observation,
                world=transition.world,
                goal=semantic.goal,
                content=semantic.content_plan,
                capability_snapshot=runtime.workbench_environment.capability_snapshot,
                tick=tick,
            )
        row: dict[str, Any] = {
            "tick": tick,
            "path": observation.path,
            "transition_status": transition.status,
            "transition_world_relations": (
                []
                if transition.world is None
                else [list(relation) for relation in transition.world.relations]
            ),
            "semantic_status": semantic.status,
            "predicted_goal": None if semantic.goal is None else semantic.goal.goal_id,
            "accepted": bool(decision is not None and decision.accepted),
            "decision_reason": None if decision is None else decision.reason_code,
            "intent_kind": (
                None
                if decision is None or decision.action_intent is None
                else decision.action_intent.kind
            ),
            "admission_required": False,
            "real_success": False,
            "real_reward": 0.0,
            "outcome_admitted": False,
        }
        if decision is None or not decision.accepted or decision.action_intent is None:
            rows.append(row)
            break

        snapshot_id = runtime.workbench_environment.capability_snapshot.snapshot_id
        row["admission_required"] = decision.action_intent.kind == "workspace.read"
        outcome = runtime.execute_workbench_intent(
            decision.action_intent,
            snapshot_id=snapshot_id,
            learn=False,
        )
        row["real_success"] = bool(outcome["outcome"]["success"])
        taiji_outcome_payload = outcome.get("taiji_outcome") or {}
        row["real_reward"] = float(taiji_outcome_payload.get("reward", 0.0))
        if row["admission_required"] and row["real_success"]:
            # The executor's success reward is intentionally constant.  For
            # the preregistered variance gate, use the fixed S6/S6B outcome
            # function over this execution's real read result instead of
            # inventing a label outside the Workbench boundary.
            result_payload = outcome["outcome"].get("result") or {}
            row["real_reward"] = float(_graded_read_reward(result_payload))
            row["reward_source"] = "workbench.read.result:s6-graded-v1"
        else:
            row["reward_source"] = "executor"
        row["freshness_after_execute"] = _runtime_freshness()
        if row["admission_required"]:
            try:
                if row["real_success"]:
                    reprojected = runtime.reproject_workbench_from_latest_evidence(
                        snapshot_id=snapshot_id
                    )
                    affordance_id = reprojected["affordances"][0]["affordance_id"]
                else:
                    affordance_id = "workbench-failed:auto"
                projected = runtime.project_workbench_outcome_for_internalization(
                    snapshot_id=snapshot_id,
                    affordance_id=affordance_id,
                    reward=row["real_reward"],
                    reward_terms={"task_success": row["real_reward"]},
                    parent_checkpoint_id=parent_checkpoint_id,
                )
                admitted = InternalizationLedger(
                    converter=InternalizationConverter(seed=17, replay_budget=64)
                ).ingest(projected)
                row["outcome_admitted"] = bool(admitted.accepted)
            except (KeyError, ValueError) as exc:
                row["outcome_admission_error"] = str(exc)[:160]
                row["freshness_on_admission_error"] = _runtime_freshness()
        rows.append(row)
        if not row["real_success"] or (
            row["admission_required"] and not row["outcome_admitted"]
        ):
            break
        if transition.world is None:
            break
        current_world = transition.world

    return {
        "rows": rows,
        "step_count": len(rows),
            "episode_success": bool(
            len(rows) == 3
            and all(
                row["accepted"] and row["real_success"]
                for row in rows
            )
        ),
    }


def _run_arm(
    *,
    arm_name: str,
    semantic_learner: StructuredSemanticLearner,
    transition_learner: StructuredSemanticTransitionLearner,
    holdout_episodes: tuple[tuple[WorkbenchObservation, ...], ...],
    anchor: WorkbenchObservation,
    planner: NativeReadOnlyIntentPlanner,
    temp_root: Path,
    holdout_registry: ProgrammingLanguageRegistry,
    learner_seed: int,
    task_seed: int,
) -> dict[str, Any]:
    runtime = SeedRuntime(
        Seed(
            SeedConfig(taiji=TaijiConfig(seed=learner_seed)),
            episode_id=f"m5-k2-{arm_name}-{task_seed}-{learner_seed}",
        )
    )
    runtime._workbench_environment = WorkbenchEnvironment(
        root=temp_root, programming_language_registry=holdout_registry
    )
    initial_world = _world(anchor, tick=0)
    episodes = [
        _run_episode(
            semantic_learner=semantic_learner,
            transition_learner=transition_learner,
            sequence=sequence,
            initial_world=initial_world,
            planner=planner,
            runtime=runtime,
            parent_checkpoint_id="checkpoint:k2-parent",
        )
        for sequence in holdout_episodes
    ]
    return {
        "episodes": episodes,
        "episode_success_rate": (
            sum(1 for episode in episodes if episode["episode_success"])
            / len(episodes)
        ),
        "all_successful_outcomes_admitted": all(
            row["outcome_admitted"]
            for episode in episodes
            for row in episode["rows"]
            if row["real_success"] and row["admission_required"]
        ),
    }


def _population_variance(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def run_cell(*, task_seed: int, learner_seed: int) -> dict[str, Any]:
    started = time.perf_counter()
    temp_root, temp_parent, temp_parent_created = _create_isolated_temp_root()
    try:
        import seed_platform.workbench as workbench_module

        original_get_setting = workbench_module.get_setting
        workbench_module.get_setting = lambda key, default=None: (
            str(temp_root) if key == "workspace_path" else default
        )
        try:
            _build_workspace(temp_root, task_seed=task_seed)
            schema = _schema()
            train_registry = _registry(typescript_available=False)
            holdout_registry = _registry(typescript_available=True)
            all_paths = ["missing_00.txt"]
            for index in range(FILES_PER_LANG):
                all_paths.extend(_path(language, index) for language in LANGS)
            observations = {
                observation.path: observation
                for observation in _observe_all(
                    temp_root,
                    registry=train_registry,
                    split="course",
                    paths=all_paths,
                    schema=schema,
                )
            }
            anchor = observations["missing_00.txt"]
            (
                semantic_corpus,
                transition_corpus,
                train_episodes,
                dev_episode,
                test_episode,
            ) = _semantic_and_transition_corpora(observations, anchor, schema)

            fact_masks = _typed_fact_feature_masks(semantic_corpus.fact_keys, schema)
            readout_excluded = tuple(
                key for key in semantic_corpus.fact_keys if key.split("::")[1] == "language"
            )
            trained_semantic = StructuredSemanticLearner(
                semantic_corpus,
                fact_feature_masks=fact_masks,
                readout_excluded_facts=readout_excluded,
            )
            semantic_losses = trained_semantic.fit(
                semantic_corpus.train,
                epochs=SEMANTIC_EPOCHS,
                learning_rate=SEMANTIC_LR,
            )
            frozen_semantic = StructuredSemanticLearner(
                semantic_corpus,
                fact_feature_masks=fact_masks,
                readout_excluded_facts=readout_excluded,
            )
            semantic_lesion = StructuredSemanticLearner.from_checkpoint(
                trained_semantic.checkpoint(), semantic_corpus
            )
            semantic_lesion.zero_fact_head()

            transition_masks = _typed_transition_input_masks(
                transition_corpus.fact_keys,
                schema,
                event_dim=transition_corpus.event_dim,
            )
            trained_transition = StructuredSemanticTransitionLearner(
                transition_corpus,
                transition_input_masks=transition_masks,
            )
            transition_losses = trained_transition.fit(
                transition_corpus.train,
                epochs=TRAIN_EPOCHS,
                learning_rate=TRAIN_LR,
            )
            frozen_transition = StructuredSemanticTransitionLearner(
                transition_corpus,
                transition_input_masks=transition_masks,
            )
            transition_lesion = StructuredSemanticTransitionLearner.from_checkpoint(
                trained_transition.checkpoint(), transition_corpus
            )
            transition_lesion.zero_transition_head()

            holdout_paths = sorted(
                {path for paths in _holdout_episode_paths() for path in paths}
            )
            holdout_observations = {
                observation.path: observation
                for observation in _observe_all(
                    temp_root,
                    registry=holdout_registry,
                    split="holdout",
                    paths=holdout_paths,
                    schema=schema,
                )
            }
            holdout_episodes = tuple(
                _episode(
                    _observe_all(
                        temp_root,
                        registry=holdout_registry,
                        split="holdout-anchor",
                        paths=["missing_00.txt"],
                        schema=schema,
                    )[0],
                    holdout_observations,
                    paths,
                )
                for paths in _holdout_episode_paths()
            )
            holdout_anchor = holdout_episodes[0][0]
            planner = NativeReadOnlyIntentPlanner(
                ReadOnlyIntentPolicy(routes=READ_ONLY_ROUTES)
            )

            arms = {
                "A-full-chain": _run_arm(
                    arm_name="A-full-chain",
                    semantic_learner=trained_semantic,
                    transition_learner=trained_transition,
                    holdout_episodes=holdout_episodes,
                    anchor=holdout_anchor,
                    planner=planner,
                    temp_root=temp_root,
                    holdout_registry=holdout_registry,
                    learner_seed=learner_seed,
                    task_seed=task_seed,
                ),
                "B-frozen": _run_arm(
                    arm_name="B-frozen",
                    semantic_learner=frozen_semantic,
                    transition_learner=frozen_transition,
                    holdout_episodes=holdout_episodes,
                    anchor=holdout_anchor,
                    planner=planner,
                    temp_root=temp_root,
                    holdout_registry=holdout_registry,
                    learner_seed=learner_seed,
                    task_seed=task_seed,
                ),
                "C-transition-lesion": _run_arm(
                    arm_name="C-transition-lesion",
                    semantic_learner=trained_semantic,
                    transition_learner=transition_lesion,
                    holdout_episodes=holdout_episodes,
                    anchor=holdout_anchor,
                    planner=planner,
                    temp_root=temp_root,
                    holdout_registry=holdout_registry,
                    learner_seed=learner_seed,
                    task_seed=task_seed,
                ),
            }

            # The technical training gate is deliberately measured on the
            # observed course before any holdout result is read.
            course_planner = planner
            course_runtime = SeedRuntime(
                Seed(
                    SeedConfig(taiji=TaijiConfig(seed=learner_seed)),
                    episode_id=f"m5-k2-course-{task_seed}-{learner_seed}",
                )
            )
            course_runtime._workbench_environment = WorkbenchEnvironment(
                root=temp_root, programming_language_registry=train_registry
            )
            train_course_results = [
                _run_episode(
                    semantic_learner=trained_semantic,
                    transition_learner=trained_transition,
                    sequence=sequence,
                    initial_world=_world(anchor, tick=0),
                    planner=course_planner,
                    runtime=course_runtime,
                    parent_checkpoint_id="checkpoint:k2-parent",
                )
                for sequence in train_episodes
            ]
            dev_result = _run_episode(
                semantic_learner=trained_semantic,
                transition_learner=trained_transition,
                sequence=(anchor, *tuple(observations[path] for path in _dev_episode_paths())),
                initial_world=_world(anchor, tick=0),
                planner=course_planner,
                runtime=course_runtime,
                parent_checkpoint_id="checkpoint:k2-parent",
            )
            test_result = _run_episode(
                semantic_learner=trained_semantic,
                transition_learner=trained_transition,
                sequence=(anchor, *tuple(observations[path] for path in _test_episode_paths())),
                initial_world=_world(anchor, tick=0),
                planner=course_planner,
                runtime=course_runtime,
                parent_checkpoint_id="checkpoint:k2-parent",
            )
            checkpoint_gate = _checkpoint_gate(trained_transition, transition_corpus)

            a_rate = arms["A-full-chain"]["episode_success_rate"]
            b_rate = arms["B-frozen"]["episode_success_rate"]
            c_rate = arms["C-transition-lesion"]["episode_success_rate"]
            a_rewards = [
                float(row["real_reward"])
                for episode in arms["A-full-chain"]["episodes"]
                for row in episode["rows"]
            ]
            reward_variance = _population_variance(a_rewards)
            checks = {
                "a_holdout_success_at_least_0p75": a_rate >= 0.75,
                "a_minus_b_at_least_0p25": (a_rate - b_rate) >= 0.25,
                "a_minus_c_at_least_0p25": (a_rate - c_rate) >= 0.25,
                "a_train_success_is_1": all(
                    result["episode_success"] for result in train_course_results
                ),
                "train_episode_count_at_least_6": len(train_course_results) >= 6,
                "dev_episode_completed": dev_result["episode_success"],
                "test_episode_completed": test_result["episode_success"],
                "a_successful_outcomes_admitted": arms["A-full-chain"][
                    "all_successful_outcomes_admitted"
                ],
                "reward_variance_positive": reward_variance > 1e-12,
                "checkpoint_mask_gate": checkpoint_gate["passed"],
            }
            technical_gate_all_passed = all(bool(value) for value in checks.values())
            return {
                "task_seed": int(task_seed),
                "learner_seed": int(learner_seed),
                "arms": arms,
                "train_course": train_course_results,
                "dev_course": dev_result,
                "test_course": test_result,
                "typed_binding": {
                    "transition_input_masks": {
                        key: list(indices) for key, indices in transition_masks.items()
                    },
                    "semantic_fact_feature_masks": {
                        key: list(indices) for key, indices in fact_masks.items()
                    },
                    "readout_excluded_facts": list(readout_excluded),
                },
                "stage1_semantic_fit_losses": {
                    key: round(float(value), 8) for key, value in semantic_losses.items()
                },
                "stage2_transition_fit_losses": {
                    key: round(float(value), 8) for key, value in transition_losses.items()
                },
                "checkpoint_gate": checkpoint_gate,
                "metrics": {
                    "a_holdout_success": a_rate,
                    "b_holdout_success": b_rate,
                    "c_holdout_success": c_rate,
                    "a_minus_b": a_rate - b_rate,
                    "a_minus_c": a_rate - c_rate,
                    "a_train_success": sum(
                        1 for result in train_course_results if result["episode_success"]
                    )
                    / len(train_course_results),
                    "a_reward_variance": reward_variance,
                },
                "checks": checks,
                "technical_gate_all_passed": technical_gate_all_passed,
                "elapsed_seconds": time.perf_counter() - started,
            }
        finally:
            workbench_module.get_setting = original_get_setting
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
        if temp_parent_created:
            try:
                temp_parent.rmdir()
            except OSError:
                pass


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
        "composition": "autoregressive transition world -> stage1 goal/content -> read-only intent -> REAL isolated execution (S6B)",
        "holdout": "4 unseen three-step sequence x language x file combinations",
        "cell": cell,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "report": str(args.report)}, ensure_ascii=False))
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
