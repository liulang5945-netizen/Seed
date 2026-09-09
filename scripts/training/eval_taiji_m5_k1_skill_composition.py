"""M5.K1 canary: skill-composition course (A8 main evidence, first step).

Preregistration: ``plans/reference/M5_K1_SKILL_COMPOSITION_PREREGISTRATION
_20260909.md``.  One coherent path on real Workbench workspaces:

  stage 1  structured semantics  (StructuredSemanticLearner: percept ->
           facts / Goal / ContentPlan; the arms vary THIS learner)
  stage 2  world transition      (StructuredSemanticTransitionLearner,
           minimized per preregistration to an interface check)
  stage 3  read-only intent      (NativeReadOnlyIntentPlanner)
  stage 4  isolated execution    (SeedRuntime real execute, S6B outcomes)

Compositional holdout: the training environment has an unavailable
TypeScript toolchain, so TypeScript files only ever appear in
clarify-toolchain tasks; the holdout environment makes TypeScript available,
so ``inspect x typescript x file`` is a genuinely unseen triple whose
elements (inspect tasks, typescript files) were both seen in training.

Arms: A trained semantic learner, B untrained learner, C trained with a
zeroed fact head.  Primary metric: real execution success on the unseen
triples.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.seed_runtime import SeedRuntime  # noqa: E402
from scripts.training.eval_taiji_m3r1_native_observation import (  # noqa: E402
    _semantic_kind,
)
from seed import Seed  # noqa: E402
from seed.config import SeedConfig  # noqa: E402
from seed_platform.programming_languages import (  # noqa: E402
    ProgrammingLanguageRegistry,
)
from seed_platform.workbench import WorkbenchEnvironment  # noqa: E402
from taiji import (  # noqa: E402
    ContentPlan,
    Goal,
    InternalizationConverter,
    InternalizationLedger,
    NativeReadOnlyIntentPlanner,
    ReadOnlyIntentPolicy,
    StructuredSemanticCorpus,
    StructuredSemanticExample,
    StructuredSemanticLearner,
    StructuredSemanticTransitionCorpus,
    StructuredSemanticTransitionExample,
    StructuredSemanticTransitionLearner,
    TaijiConfig,
    WorkbenchObservation,
    WorldState,
)

REPORT_FORMAT = "taiji-m5-k1-skill-composition-v1"
VERSION = 1
GROUNDING_MARGIN_FLOOR = 0.05
TRAIN_EPOCHS = 280
TRAIN_LR = 0.2
SEMANTIC_EPOCHS = 160
SEMANTIC_LR = 2.0

FILES_PER_LANG = 8
TRAIN_FILES = 4  # _00.._03 train, _04 dev, _05.._07 holdout
LANGS = ("python", "typescript", "rust")
EXTENSIONS = {"python": ".py", "typescript": ".ts", "rust": ".rs"}
READ_ONLY_ROUTES = (
    ("content:inspect-language", "workspace.read"),
    ("content:clarify-toolchain", "workspace.programming_language.resolve"),
    ("content:clarify-language", "workspace.programming_language.resolve"),
)

KIND_GOALS = {
    "inspect-language": "Report verified language and toolchain facts.",
    "clarify-toolchain": "Clarify the unavailable toolchain before execution.",
    "clarify-language": "Clarify an ambiguous programming language selection.",
    "recover-target": "Recover a missing Workbench target before proceeding.",
}


def _registry(*, typescript_available: bool) -> ProgrammingLanguageRegistry:
    python_command = Path(sys.executable).name
    definitions = []
    for definition in ProgrammingLanguageRegistry.default().definitions:
        if definition.language_id == "python":
            definitions.append(replace(definition, toolchain_commands=(python_command,)))
        elif definition.language_id == "typescript":
            command = python_command if typescript_available else "m5k1-unavailable-tsc"
            definitions.append(replace(definition, toolchain_commands=(command,)))
        elif definition.language_id == "rust":
            definitions.append(replace(definition, toolchain_commands=("m5k1-unavailable-rustc",)))
        else:
            definitions.append(definition)
    return ProgrammingLanguageRegistry(definitions)


def _schema():
    from taiji import WorkbenchObservationSchema

    return WorkbenchObservationSchema(
        language_ids=("python", "rust", "typescript", "unknown"),
        selection_states=("ambiguous", "resolved", "unknown"),
        task_kinds=("inspect-language",),
        extensions=(".h", ".py", ".rs", ".ts", "<none>"),
    )


def _observation(
    environment: WorkbenchEnvironment,
    *,
    tag: str,
    path: str,
    schema,
    tick: int,
) -> WorkbenchObservation:
    read_result, language_result = _read_evidence(environment, path)
    snapshot = environment.capability_snapshot
    return WorkbenchObservation.from_workbench_evidence(
        observation_id=f"m5k1:{tag}:{path}",
        project_id="m5k1-project",
        task_id=f"m5k1:read-language:{path}",
        path=path,
        capability_snapshot_id=snapshot.snapshot_id,
        capability_revision=snapshot.revision,
        read_result=read_result,
        language_result=language_result,
        task_kind="inspect-language",
        schema=schema,
    )


def _read_evidence(environment: WorkbenchEnvironment, path: str):
    try:
        read_result = environment.read_workspace_evidence({"path": path})
    except FileNotFoundError:
        return {"success": False, "error_code": "not_found", "path": path}, None
    except (IsADirectoryError, NotADirectoryError, ValueError) as exc:
        return {"success": False, "error_code": type(exc).__name__, "path": path}, None
    read_result = {"success": True, **read_result}
    language_result = environment.resolve_programming_language_evidence({"path": path})
    return read_result, language_result


def _world(observation: WorkbenchObservation, *, tick: int) -> WorldState:
    relations = (
        ("workbench", "read", "success" if observation.read_success else "failure"),
        ("workbench", "target", "file" if observation.file_is_file else "missing"),
        ("workbench", "language", observation.language_id),
        ("workbench", "language_state", observation.selection_state),
        (
            "workbench",
            "toolchain",
            "available" if observation.toolchain_available else "missing",
        ),
        (
            "workbench",
            "diagnostics",
            "connected" if observation.diagnostics_connected else "disconnected",
        ),
    )
    event = observation.to_percept_event(tick=tick)
    return WorldState(
        tick=tick,
        latent=torch.empty(0),
        entities=("workbench",),
        relations=relations,
        uncertainty=0.0,
        percept_event_id=event.event_id,
        percept_assembly_id=event.assembly_id,
    )


def _goal_content(observation: WorkbenchObservation, *, tick: int) -> tuple[Goal, ContentPlan]:
    kind = _semantic_kind(observation)
    request = kind != "inspect-language"
    goal = Goal(
        goal_id=f"goal:{kind}",
        description=KIND_GOALS[kind],
        priority=0.9,
    )
    content = ContentPlan(
        content_id=f"content:{kind}",
        intent_id=f"intent:{kind}",
        intent_kind="request_information" if request else "report",
        semantic_slots={
            "observed_language": observation.language_id,
            "selection_state": observation.selection_state,
            "toolchain_available": observation.toolchain_available,
        },
        source_goal_id=goal.goal_id,
        expected_outcome={
            "inspect-language": "verified read-only report",
            "clarify-toolchain": "toolchain clarification",
            "clarify-language": "language clarification",
            "recover-target": "valid target path",
        }[kind],
        tick=tick,
    )
    return goal, content


def _semantic_example(
    observation: WorkbenchObservation, *, split: str, tick: int
) -> StructuredSemanticExample:
    goal, content = _goal_content(observation, tick=tick)
    return StructuredSemanticExample(
        example_id=f"m5k1:{split}:{observation.path}",
        family_id=f"m5k1:{split}",
        percept=observation.to_percept_event(tick=tick),
        world=_world(observation, tick=tick),
        goal=goal,
        content=content,
    )


def _transition_examples(
    sequence: tuple[WorkbenchObservation, ...],
    *,
    split: str,
) -> tuple[StructuredSemanticTransitionExample, ...]:
    result = []
    for index in range(1, len(sequence)):
        before = sequence[index - 1]
        current = sequence[index]
        goal, content = _goal_content(current, tick=index)
        result.append(
            StructuredSemanticTransitionExample(
                example_id=f"m5k1:{split}:step-{index}:{current.path}",
                family_id=f"m5k1:{split}:sequence",
                before=_world(before, tick=index - 1),
                event=current.to_percept_event(tick=index),
                after=_world(current, tick=index),
                goal=goal,
                content=content,
            )
        )
    return tuple(result)


def _expected_capability(observation: WorkbenchObservation) -> str | None:
    kind = _semantic_kind(observation)
    if kind == "inspect-language":
        return "workspace.read"
    if kind in {"clarify-toolchain", "clarify-language"}:
        return "workspace.programming_language.resolve"
    return None


def _build_workspace(root: Path, *, task_seed: int) -> None:
    # Project manifests supply the manifest evidence that pushes language
    # confidence past the resolved threshold (extension 0.14 + content 0.55
    # alone stays below it), matching the M3.R1 fixture design.
    (root / "pyproject.toml").write_text("[project]\nname = 'm5k1'\n", encoding="utf-8")
    (root / "Cargo.toml").write_text(
        "[package]\nname = 'm5k1'\nversion = '0.1.0'\n", encoding="utf-8"
    )
    (root / "tsconfig.json").write_text('{"compilerOptions": {"strict": true}}\n', encoding="utf-8")
    (root / "package.json").write_text('{"name": "m5k1", "version": "0.1.0"}\n', encoding="utf-8")
    for lang in LANGS:
        extension = EXTENSIONS[lang]
        for index in range(FILES_PER_LANG):
            variant = (index + task_seed) % 8
            # Language-idiomatic content so only the intended language's
            # content pattern matches (generic fillers like "record x" collide
            # with other registries' patterns), plus an index-length pad so
            # every file has a distinct byte_length and therefore a distinct
            # observation feature vector.
            pad = "p" * index
            if lang == "python":
                body = (
                    f"def answer_{index}(value: int) -> int:\n"
                    f"    return value + {index + variant}\n"
                    f"# m5k1 {pad}\n"
                )
            elif lang == "typescript":
                body = (
                    f"interface Answer{index} {{ value: number; }}\n"
                    f"export const answer{index} = (input: Answer{index}): number => "
                    f"input.value + {index + variant};\n"
                    f"// m5k1 {pad}\n"
                )
            else:
                body = (
                    "fn main() {\n"
                    f'    println!("m5k1-{index}-{variant}");\n'
                    "}\n"
                    f"// m5k1 {pad}\n"
                )
            (root / f"{lang}_{index:02d}{extension}").write_text(body, encoding="utf-8")


def _observe_all(
    root: Path,
    *,
    registry: ProgrammingLanguageRegistry,
    split: str,
    paths: list[str],
    schema,
) -> tuple[WorkbenchObservation, ...]:
    environment = WorkbenchEnvironment(root=root, programming_language_registry=registry)
    return tuple(
        _observation(
            environment,
            tag=split,
            path=path,
            schema=schema,
            tick=index + 1,
        )
        for index, path in enumerate(paths)
    )


def _arm_outcomes(
    *,
    semantic_learner: StructuredSemanticLearner,
    holdout: tuple[WorkbenchObservation, ...],
    planner: NativeReadOnlyIntentPlanner,
    runtime: SeedRuntime,
) -> dict[str, Any]:
    """Stage-1 semantics -> stage-3 intent -> stage-4 REAL execute, per step.

    The arms vary ONLY the semantic learner (preregistration section 3), so
    the planner consumes the stage-1 event-grounded world, goal, and content
    plan.  Stage 2 (world transition) is minimized to an interface check and
    is measured separately by ``_stage2_interface``.
    """
    rows: list[dict[str, Any]] = []
    for tick, observation in enumerate(holdout, start=1):
        expected = _expected_capability(observation)
        result = semantic_learner.predict(observation.to_percept_event(tick=tick))
        goal = result.goal
        content = result.content_plan
        world = result.world
        decision = None
        if goal is not None and content is not None and world is not None:
            decision = planner.propose(
                observation=observation,
                world=world,
                goal=goal,
                content=content,
                capability_snapshot=runtime.workbench_environment.capability_snapshot,
                tick=tick,
            )
        row: dict[str, Any] = {
            "path": observation.path,
            "semantic_kind": _semantic_kind(observation),
            "expected_capability": expected,
            "semantic_status": result.status,
            "predicted_goal": None if result.goal is None else result.goal.goal_id,
            "accepted": bool(decision is not None and decision.accepted),
            "intent_kind": (
                None
                if decision is None or decision.action_intent is None
                else decision.action_intent.kind
            ),
        }
        if decision is not None and decision.accepted and decision.action_intent is not None:
            intent = decision.action_intent
            snapshot_id = runtime.workbench_environment.capability_snapshot.snapshot_id
            outcome = runtime.execute_workbench_intent(
                intent,
                snapshot_id=snapshot_id,
                learn=False,
            )
            row["real_success"] = bool(outcome["outcome"]["success"])
            taiji_outcome_payload = outcome.get("taiji_outcome") or {}
            row["real_reward"] = float(taiji_outcome_payload.get("reward", 0.0))
            # Real outcome enters the internalization ledger under S6B policy:
            # successful executions reproject from the latest read evidence to
            # get the grounded affordance; failures use the auto failure
            # affordance (same split as eval_taiji_m5_s6b_failure_admission).
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
                    parent_checkpoint_id="checkpoint:k1-parent",
                )
                admitted = InternalizationLedger(
                    converter=InternalizationConverter(seed=17, replay_budget=64)
                ).ingest(projected)
                row["outcome_admitted"] = bool(admitted.accepted)
            except ValueError as exc:
                row["outcome_admitted"] = False
                row["outcome_admission_error"] = str(exc)[:120]
        else:
            row["real_success"] = False
            row["real_reward"] = 0.0
            row["outcome_admitted"] = False
        rows.append(row)
    unseen = [
        r
        for r in rows
        if r["path"].startswith("typescript_05")
        or r["path"].startswith("typescript_06")
        or r["path"].startswith("typescript_07")
    ]
    unseen_success = sum(1 for r in unseen if r["real_success"]) / len(unseen) if unseen else 0.0
    return {
        "rows": rows,
        "unseen_triple_success": unseen_success,
        "unseen_triple_count": len(unseen),
        "all_real_success": (
            sum(1 for r in rows if r["real_success"]) / len(rows) if rows else 0.0
        ),
    }


def _stage2_interface(
    *,
    transition_learner: StructuredSemanticTransitionLearner,
    holdout: tuple[WorkbenchObservation, ...],
    initial_world: WorldState,
) -> dict[str, Any]:
    """K1 minimizes stage 2 to an interface check (preregistration section 2).

    The transition learner consumes the same holdout percepts autoregressively
    and must return well-formed transition outputs at every step.  Its world
    does not gate the planner here: the arms vary only the semantic learner,
    so a stage-2-gated metric would be blind to the arm contrast.
    """
    steps: list[dict[str, Any]] = []
    current = initial_world
    for tick, observation in enumerate(holdout, start=1):
        result = transition_learner.predict(
            current, observation.to_percept_event(tick=tick)
        )
        world_wellformed = (
            result.world is not None
            and all(len(relation) == 3 for relation in result.world.relations)
        )
        goal_contract_ok = result.goal is None or isinstance(result.goal, Goal)
        content_contract_ok = result.content_plan is None or isinstance(
            result.content_plan, ContentPlan
        )
        steps.append(
            {
                "path": observation.path,
                "status": result.status,
                "world_wellformed": bool(world_wellformed),
                "goal_contract_ok": bool(goal_contract_ok),
                "content_contract_ok": bool(content_contract_ok),
            }
        )
        if result.world is not None:
            current = result.world
    return {
        "steps": steps,
        "step_count": len(steps),
        "all_worlds_wellformed": all(step["world_wellformed"] for step in steps),
        "all_contracts_ok": all(
            step["goal_contract_ok"] and step["content_contract_ok"] for step in steps
        ),
    }


def run_cell(*, task_seed: int, learner_seed: int) -> dict[str, Any]:
    started = time.perf_counter()
    temp_root = Path(tempfile.mkdtemp(prefix="taiji_m5_k1_"))
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

            # Stage 1+2 training: observations in the train environment.
            train_paths = [
                f"{lang}_{i:02d}{EXTENSIONS[lang]}" for lang in LANGS for i in range(TRAIN_FILES)
            ]
            train_paths += ["missing_00.txt", "missing_01.txt"]
            train_obs = _observe_all(
                temp_root,
                registry=train_registry,
                split="train",
                paths=train_paths,
                schema=schema,
            )
            sequence: tuple[WorkbenchObservation, ...] = (
                train_obs[-2],  # missing_00 (recover anchor)
                *train_obs[0:4],  # python inspect (recover -> inspect)
                train_obs[-1],  # missing_01 (inspect -> recover)
                *train_obs[4:8],  # typescript clarify (recover -> clarify)
                *train_obs[8:12],  # rust clarify (clarify -> clarify)
            )
            # Two recover->file transitions with DIFFERENT event languages
            # (missing_00->python, missing_01->typescript) force the
            # transition head to bind the language fact to the event's
            # language one-hot instead of a before-block shortcut; a single
            # recover->python transition let "M x toolchain=1.0 -> python"
            # win and broke compositional language binding at holdout.
            transitions = _transition_examples(sequence, split="train")

            # Dev/test sequences reuse the trained task kinds on unseen files,
            # ordered so that every (before-block, event-file) pair is disjoint
            # from the other splits.  before-fact keys encode only the block
            # (missing / python / typescript / rust), so a naive reversal of
            # the train sequence would duplicate input digests inside the
            # typescript and rust blocks.  All observations still use the
            # train registry (typescript toolchain unavailable).
            dev_sequence = _observe_all(
                temp_root,
                registry=train_registry,
                split="dev",
                paths=[
                    "missing_02.txt",  # recover (anchor)
                    f"typescript_04{EXTENSIONS['typescript']}",  # (missing, ts)
                    f"rust_04{EXTENSIONS['rust']}",  # (ts, rs)
                    f"python_04{EXTENSIONS['python']}",  # (rs, py)
                    f"typescript_05{EXTENSIONS['typescript']}",  # (py, ts)
                ],
                schema=schema,
            )
            test_sequence = _observe_all(
                temp_root,
                registry=train_registry,
                split="test",
                paths=[
                    "missing_03.txt",  # recover (anchor)
                    f"rust_05{EXTENSIONS['rust']}",  # (missing, rs)
                    f"python_05{EXTENSIONS['python']}",  # (rs, py)
                    f"typescript_06{EXTENSIONS['typescript']}",  # (py, ts)
                    f"rust_06{EXTENSIONS['rust']}",  # (ts, rs)
                ],
                schema=schema,
            )
            # Stage 2 corpus: fact-delta transitions over the train sequence.
            transition_corpus = StructuredSemanticTransitionCorpus.from_splits(
                train=transitions,
                dev=_transition_examples(dev_sequence, split="dev"),
                test=_transition_examples(test_sequence, split="test"),
            )

            # Stage 1 corpus: one semantic example per observation.  All
            # missing-file percepts share identical feature vectors (failed
            # reads carry no content), so only missing_00 enters the stage-1
            # corpus; a second one would duplicate an input digest across
            # splits.  Dev/test use the unseen file observations only.
            semantic_corpus = StructuredSemanticCorpus.from_splits(
                train=tuple(
                    _semantic_example(observation, split="train", tick=index + 1)
                    for index, observation in enumerate(train_obs[0:13])
                ),
                dev=tuple(
                    _semantic_example(observation, split="dev", tick=index + 1)
                    for index, observation in enumerate(dev_sequence[1:])
                ),
                test=tuple(
                    _semantic_example(observation, split="test", tick=index + 1)
                    for index, observation in enumerate(test_sequence[1:])
                ),
            )

            planner = NativeReadOnlyIntentPlanner(ReadOnlyIntentPolicy(routes=READ_ONLY_ROUTES))
            trained_semantic = StructuredSemanticLearner(semantic_corpus)
            semantic_losses = trained_semantic.fit(
                semantic_corpus.train, epochs=SEMANTIC_EPOCHS, learning_rate=SEMANTIC_LR
            )
            frozen_semantic = StructuredSemanticLearner(semantic_corpus)
            lesion_payload = trained_semantic.checkpoint()
            lesioned_semantic = StructuredSemanticLearner.from_checkpoint(
                lesion_payload, semantic_corpus
            )
            lesioned_semantic.zero_fact_head()

            trained_transition = StructuredSemanticTransitionLearner(transition_corpus)
            transition_losses = trained_transition.fit(
                transition_corpus.train, epochs=TRAIN_EPOCHS, learning_rate=TRAIN_LR
            )

            # Holdout environment: TypeScript toolchain now available.
            holdout_paths = [
                f"typescript_05{EXTENSIONS['typescript']}",
                f"typescript_06{EXTENSIONS['typescript']}",
                f"typescript_07{EXTENSIONS['typescript']}",
                f"python_05{EXTENSIONS['python']}",
                f"rust_05{EXTENSIONS['rust']}",
                "missing_03.txt",
            ]
            holdout = _observe_all(
                temp_root,
                registry=holdout_registry,
                split="holdout",
                paths=holdout_paths,
                schema=schema,
            )

            arms: dict[str, Any] = {}
            runtime_by_arm: dict[str, SeedRuntime] = {}
            for arm_name, semantic_learner in (
                ("A-full-chain", trained_semantic),
                ("B-frozen-semantic", frozen_semantic),
                ("C-semantic-lesion", lesioned_semantic),
            ):
                runtime = SeedRuntime(
                    Seed(
                        SeedConfig(taiji=TaijiConfig(seed=learner_seed)),
                        episode_id=f"m5-k1-{arm_name}-{task_seed}-{learner_seed}",
                    )
                )
                runtime._workbench_environment = WorkbenchEnvironment(
                    root=temp_root, programming_language_registry=holdout_registry
                )
                runtime_by_arm[arm_name] = runtime
                arms[arm_name] = _arm_outcomes(
                    semantic_learner=semantic_learner,
                    holdout=holdout,
                    planner=planner,
                    runtime=runtime,
                )
            del runtime_by_arm

            stage2_interface = _stage2_interface(
                transition_learner=trained_transition,
                holdout=holdout,
                initial_world=_world(sequence[-1], tick=0),
            )

            a_unseen = arms["A-full-chain"]["unseen_triple_success"]
            b_unseen = arms["B-frozen-semantic"]["unseen_triple_success"]
            c_unseen = arms["C-semantic-lesion"]["unseen_triple_success"]
            checks = {
                "a_unseen_success_at_least_0p8": a_unseen >= 0.8,
                "a_minus_b_at_least_0p3": (a_unseen - b_unseen) >= 0.3,
                "a_minus_c_at_least_0p3": (a_unseen - c_unseen) >= 0.3,
                "all_arms_produced_rows": all(
                    arms[name]["unseen_triple_count"] == 3 for name in arms
                ),
            }
            technical_gate_all_passed = all(bool(v) for v in checks.values())

            return {
                "task_seed": int(task_seed),
                "learner_seed": int(learner_seed),
                "arms": arms,
                "stage1_semantic_fit_losses": {
                    key: round(float(value), 8) for key, value in semantic_losses.items()
                },
                "stage2_transition_fit_losses": {
                    key: round(float(value), 8) for key, value in transition_losses.items()
                },
                "stage2_interface": stage2_interface,
                "metrics": {
                    "a_unseen_success": a_unseen,
                    "b_unseen_success": b_unseen,
                    "c_unseen_success": c_unseen,
                    "a_minus_b": a_unseen - b_unseen,
                    "a_minus_c": a_unseen - c_unseen,
                },
                "checks": checks,
                "technical_gate_all_passed": technical_gate_all_passed,
                "elapsed_seconds": time.perf_counter() - started,
            }
        finally:
            workbench_module.get_setting = original_get_setting
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


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
        "composition": "stage1 semantics (arms) -> stage2 transition interface -> read-only intent -> REAL isolated execution (S6B outcome policy)",
        "unseen_triple": "inspect x typescript x file (toolchain availability flips between train and holdout environments)",
        "cell": cell,
        "boundary": (
            "M5.K1 canary only; real read-only executions in a process-owned "
            "temporary workspace; no provider, network, or real client write"
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.report),
                "technical_gate_all_passed": cell["technical_gate_all_passed"],
                "failed_checks": [k for k, v in cell["checks"].items() if not v],
                "metrics": {k: round(v, 4) for k, v in cell["metrics"].items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if cell["technical_gate_all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
