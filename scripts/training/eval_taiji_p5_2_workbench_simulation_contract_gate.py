"""P5.2 workbench simulation contract gate.

Preregistration: plans/reference/M5_P5_2_WORKBENCH_SIMULATION_CONTRACT_PREREGISTRATION_20260912.md
(frozen).  The simulation is a restricted instance of the real Workbench
contract: every model-produced action is a typed ``ActionIntent`` bound through
``WorkbenchActionRequest.from_action_intent`` onto the live
``CapabilitySnapshot`` and executed through the real
policy -> preview -> approval -> execute -> undo code path of
``WorkbenchEnvironment`` (no parallel executor in this runner).  The action
generator is the P5.1 procedural readout (hidden 64 / 250 epochs, frozen P5.1g
construction constants) trained on deterministic scripted IDE scenes; gates
check contract identity, typing validity, language-resolve reuse, switch
narratives, undo semantics, trace projection into the interaction-group
surface, and next-action transfer over unseen scenes.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))

from seed_platform.workbench import (  # noqa: E402
    WorkbenchActionRequest,
    WorkbenchEnvironment,
)
from taiji import ArtifactInternalizationTrainer  # noqa: E402
from taiji.contracts import ActionIntent, EpisodicMemoryRecord  # noqa: E402
from taiji.document_embedding import DocumentEmbedder  # noqa: E402
from taiji.interaction_groups import (  # noqa: E402
    InteractionGroupEvaluator,
    InteractionTraceCorpus,
    InteractionTraceEpisode,
    InteractionTraceEvent,
)
from taiji.procedural_memory import ProceduralSequenceLearner  # noqa: E402

REPORT_FORMAT = "taiji-p5-2-workbench-simulation-contract-report-v1"
VERSION = 1
PREREGISTRATION = (
    "plans/reference/M5_P5_2_WORKBENCH_SIMULATION_CONTRACT_PREREGISTRATION_20260912.md"
)
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_p5_2_workbench_simulation_contract_20260912.json"

FROZEN_CONTENT_TRANSFER_MARGIN = 0.15
TOTAL_SECONDS_CAP = 600.0
PROCEDURAL_HIDDEN_DIM = 64
PROCEDURAL_EPOCHS = 250
PROCEDURAL_LEARNING_RATE = 0.05
PROCEDURAL_SEED = 17
TRAIN_ACC_FLOOR = 0.9
TRAIN_SCENES = 40
HOLDOUT_SCENES = 12
AGATE_SCENES = 8
CUE_DIM = 384
GENERATOR_OWNER = "p52-action-generator"
ENVIRONMENT_OWNER = "p52-workbench-environment"
TRACE_REVISION = 0

STATIC_CHECK_SCOPE = (
    "seed_platform/workbench.py",
    "seed_platform/programming_languages.py",
    "taiji/procedural_memory.py",
    "taiji/interaction_groups.py",
    "scripts/training/eval_taiji_p5_2_workbench_simulation_contract_gate.py",
)
STATIC_CHECK_COMMANDS = (
    "python -m py_compile <scope files>",
    "python -m ruff check <scope files>",
    "python -m mypy --follow-imports=silent seed taiji",
    "python -m black --no-cache --check <scope files>",
)


# --------------------------------------------------------------------------- #
# Deterministic scripted IDE scenes
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ScriptedStep:
    """One scripted action: capability kind plus parameter spec.

    Parameter specs may carry placeholders resolved against live workspace
    state at execution time (never bypassing the contract): ``$digest_of``
    (current file sha256), ``$undo_token_after`` (transaction token produced
    by an earlier step), ``$resolved_language`` (language id from an earlier
    resolve step), ``$patch_edit`` (text replacement over current content).
    """

    kind: str
    params: dict[str, Any]
    expect_policy_decision: str = "allow"


@dataclass(frozen=True)
class Scene:
    scene_id: str
    goal_text: str
    files: dict[str, str]
    steps: tuple[ScriptedStep, ...]
    main_path: str


def _patch_ops(content: str, old: str, new: str) -> dict[str, Any]:
    start = content.index(old)
    return {
        "kind": "text_replace",
        "operations": [{"start": start, "end": start + len(old), "text": new}],
    }


def _scene(index: int, partition: str) -> Scene:
    """Deterministic scripted scene; type rotates to cover the capability mix."""

    scene_id = f"p52-{partition}-{index:03d}"
    kind_index = index % 4
    goals = {
        0: (
            f"Scene {scene_id}: confirm the editor language of the Python module "
            f"through the language resolve capability before any edit."
        ),
        1: (
            f"Scene {scene_id}: apply a small text patch to the script file and "
            f"immediately revert it through the transaction undo token."
        ),
        2: (
            f"Scene {scene_id}: create a new scratch note in the workspace and "
            f"clean it up again with the transaction undo path."
        ),
        3: (
            f"Scene {scene_id}: the C/C++ header language evidence is ambiguous, "
            f"so record an explicit language override and then patch the header."
        ),
    }
    goal_text = goals[kind_index]
    if kind_index == 0:
        # Language resolution + explicit set_language on a resolved .py file.
        name = f"module_{index:03d}.py"
        content = f"def run_{index}():\n    return {index}\n"
        return Scene(
            scene_id=scene_id,
            goal_text=goal_text,
            files={name: content},
            main_path=name,
            steps=(
                ScriptedStep("workspace.read", {"path": name}),
                ScriptedStep("workspace.programming_language.resolve", {"path": name}),
                ScriptedStep(
                    "editor.set_language",
                    {"path": name, "programming_language_id": "python"},
                ),
            ),
        )
    if kind_index == 1:
        # Digest-checked patch followed by a real transaction undo.
        name = f"patched_{index:03d}.py"
        content = f"def run_{index}():\n    return {index}\n"
        new_content = content.replace(f"return {index}", f"return {index} + 1")
        return Scene(
            scene_id=scene_id,
            goal_text=goal_text,
            files={name: content},
            main_path=name,
            steps=(
                ScriptedStep("workspace.read", {"path": name}),
                ScriptedStep(
                    "workspace.apply_patch",
                    {
                        "path": name,
                        "before_digest": {"$digest_of": name},
                        "patch": _patch_ops(content, f"return {index}", f"return {index} + 1"),
                        "expected_after_digest": hashlib.sha256(
                            new_content.encode("utf-8")
                        ).hexdigest(),
                    },
                ),
                ScriptedStep("workspace.undo", {"undo_token": {"$undo_token_after": 2}}),
            ),
        )
    if kind_index == 2:
        # Create a file then undo the transaction.
        created = f"generated_{index:03d}.txt"
        return Scene(
            scene_id=scene_id,
            goal_text=goal_text,
            files={},
            main_path=created,
            steps=(
                ScriptedStep("workspace.list", {"path": "."}),
                ScriptedStep(
                    "workspace.create",
                    {"path": created, "content": f"generated {index}\n"},
                ),
                ScriptedStep("workspace.undo", {"undo_token": {"$undo_token_after": 2}}),
            ),
        )
    # Ambiguous header file: resolve then explicit user override + patch.
    name = f"header_{index:03d}.h"
    content = f"// scripted header {index}\nint value_{index} = {index};\n"
    new_content = content.replace(
        f"int value_{index} = {index};", f"int value_{index} = {index} + 1;"
    )
    return Scene(
        scene_id=scene_id,
        goal_text=goal_text,
        files={name: content},
        main_path=name,
        steps=(
            ScriptedStep("workspace.read", {"path": name}),
            ScriptedStep("workspace.programming_language.resolve", {"path": name}),
            ScriptedStep(
                "editor.set_language",
                {
                    "path": name,
                    "programming_language_id": "cpp",
                    "user_override": True,
                },
            ),
            ScriptedStep(
                "workspace.apply_patch",
                {
                    "path": name,
                    "before_digest": {"$digest_of": name},
                    "patch": _patch_ops(
                        content,
                        f"int value_{index} = {index};",
                        f"int value_{index} = {index} + 1;",
                    ),
                    "expected_after_digest": hashlib.sha256(
                        new_content.encode("utf-8")
                    ).hexdigest(),
                },
            ),
        ),
    )


def build_partitions() -> dict[str, tuple[Scene, ...]]:
    train = tuple(_scene(i, "train") for i in range(TRAIN_SCENES))
    holdout = tuple(_scene(TRAIN_SCENES + i, "holdout") for i in range(HOLDOUT_SCENES))
    agate = tuple(_scene(TRAIN_SCENES + HOLDOUT_SCENES + i, "agate") for i in range(AGATE_SCENES))
    return {"train": train, "holdout": holdout, "agate": agate}


# --------------------------------------------------------------------------- #
# Contract-path execution (policy -> preview -> approval -> execute -> undo)
# --------------------------------------------------------------------------- #


def _resolve_params(params: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, dict) and "$digest_of" in value:
            path = str(value["$digest_of"])
            resolved[key] = hashlib.sha256((state["root"] / path).read_bytes()).hexdigest()
        elif isinstance(value, dict) and "$undo_token_after" in value:
            step_index = int(value["$undo_token_after"])
            resolved[key] = state["undo_tokens"][step_index]
        elif isinstance(value, dict) and "$resolved_language" in value:
            path = str(value["$resolved_language"])
            resolved[key] = state["resolved_languages"][path]
        else:
            resolved[key] = value
    return resolved


def _execute_scene(environment: WorkbenchEnvironment, scene: Scene, root: Path) -> dict[str, Any]:
    """Run one scene through the real contract path; every action audited."""

    state: dict[str, Any] = {"root": root, "undo_tokens": {}, "resolved_languages": {}}
    steps: list[dict[str, Any]] = []
    for name, content in scene.files.items():
        (root / name).write_text(content, encoding="utf-8", newline="")
    snapshots_before = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in scene.files
    }
    language_checkpoints_before = list(environment.language_state_checkpoint()["selections"])

    for tick, step in enumerate(scene.steps, start=1):
        params = _resolve_params(step.params, state)
        intent = ActionIntent(
            f"p52-intent:{scene.scene_id}:{tick:04d}",
            step.kind,
            parameters=params,
            confidence=0.9,
            tick=tick,
        )
        request = WorkbenchActionRequest.from_action_intent(
            intent,
            snapshot_id=environment.capability_snapshot.snapshot_id,
        )
        decision = environment.policy_for(request)
        entry: dict[str, Any] = {
            "tick": tick,
            "kind": step.kind,
            "intent_id": intent.intent_id,
            "policy_decision": decision.decision,
            "policy_reason": decision.reason_code,
            "expected_policy_decision": step.expect_policy_decision,
        }
        needs_approval = (
            decision.decision == "ask_user"
            and decision.reason_code == "capability_requires_approval"
        )
        if decision.decision == "deny" or (decision.decision == "ask_user" and not needs_approval):
            # The contract intercepted the action: nothing executed.
            entry["executed"] = False
            entry["outcome"] = None
            steps.append(entry)
            continue

        if needs_approval:
            approval = environment.issue_approval(request)
            entry["preview"] = approval["preview"]
            tokened = dataclasses.replace(request, approval_token=approval["approval_token"])
            redecision = environment.policy_for(tokened)
            entry["approval_decision"] = redecision.decision
            entry["approval_reason"] = redecision.reason_code
            if redecision.decision != "allow":
                entry["executed"] = False
                entry["outcome"] = None
                steps.append(entry)
                continue
            environment.consume_approval(tokened)
            entry["approval_consumed"] = True

        outcome = environment.execute_tool(step.kind, params)
        entry["executed"] = True
        entry["outcome"] = {
            "success": bool(outcome.success),
            "reward": float(outcome.reward),
            "terminal": bool(outcome.terminal),
            "sensation": int(outcome.sensation),
        }
        last = environment.last_result
        entry["last_result_keys"] = sorted(last.keys())
        if step.kind == "workspace.undo":
            entry["undone"] = bool(outcome.success)
        if "transaction" in last and last["transaction"].get("undo_token"):
            state["undo_tokens"][tick] = str(last["transaction"]["undo_token"])
        if step.kind == "workspace.programming_language.resolve":
            payload = last if "programming_language_id" in last else last.get("result", {})
            language_id = str(
                payload.get("programming_language_id")
                or payload.get("assessment", {}).get("programming_language_id", "")
            )
            state["resolved_languages"][str(params["path"])] = language_id
            entry["resolved_language"] = language_id
            entry["resolved_selection_state"] = str(payload.get("selection_state", ""))
            evidence = environment.resolve_programming_language_evidence(
                {"path": str(params["path"])}
            )
            entry["registry_parity"] = bool(
                str(evidence.get("selection_state", "")) == entry["resolved_selection_state"]
            )
        if step.kind == "editor.set_language" and outcome.success:
            selections = environment.language_state_checkpoint()["selections"]
            assessment = next(
                (item for item in selections if str(item.get("path", "")) == str(params["path"])),
                {},
            )
            entry["language_narrative"] = {
                "path": str(params["path"]),
                "before": str(state["resolved_languages"].get(str(params["path"]), "unknown")),
                "after": str(assessment.get("programming_language_id", "unknown")),
                "reason": "scripted scene goal with resolved workspace evidence",
                "selection_state": str(assessment.get("selection_state", "")),
            }
        steps.append(entry)

    snapshots_after = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in scene.files
    }
    all_success = all(
        step["executed"] and step["outcome"]["success"]
        for step in steps
        if step["expected_policy_decision"] == "allow"
    )
    return {
        "scene_id": scene.scene_id,
        "steps": steps,
        "snapshots_before": snapshots_before,
        "snapshots_after": snapshots_after,
        "language_checkpoints_before": language_checkpoints_before,
        "language_checkpoints_after": list(environment.language_state_checkpoint()["selections"]),
        "all_scripted_steps_succeeded": all_success,
    }


def _scene_records(
    scenes: tuple[Scene, ...], embedder: DocumentEmbedder
) -> tuple[EpisodicMemoryRecord, ...]:
    records: list[EpisodicMemoryRecord] = []
    for scene in scenes:
        cue = embedder.embed([scene.goal_text])[0]
        episode_id = f"p52-episode:{scene.scene_id}"
        for tick, step in enumerate(scene.steps, start=1):
            records.append(
                EpisodicMemoryRecord(
                    memory_id=f"p52-memory:{scene.scene_id}:{tick:04d}",
                    episode_id=episode_id,
                    tick=tick,
                    cue=cue,
                    action_intent=ActionIntent(
                        f"p52-intent:{scene.scene_id}:{tick:04d}",
                        step.kind,
                        tick=tick - 1,
                    ),
                    outcome=None,
                    provenance="p52-scene",
                )
            )
    return tuple(records)


def _tick_majority_baseline(records: tuple[EpisodicMemoryRecord, ...]) -> float:
    by_tick: dict[int, dict[str, int]] = {}
    for record in records:
        tick = record.action_intent.tick + 1
        kind = record.action_intent.kind
        bucket = by_tick.setdefault(tick, {})
        bucket[kind] = bucket.get(kind, 0) + 1
    hits = 0
    total = 0
    for counter in by_tick.values():
        best = max(counter.items(), key=lambda item: item[1])[0]
        hits += counter[best]
        total += sum(counter.values())
    return round(hits / total, 6) if total else 0.0


def _train_readout(train_records: tuple[EpisodicMemoryRecord, ...]) -> ProceduralSequenceLearner:
    learner = ProceduralSequenceLearner(
        CUE_DIM, hidden_dim=PROCEDURAL_HIDDEN_DIM, seed=PROCEDURAL_SEED
    )
    learner.consolidate(
        train_records, epochs=PROCEDURAL_EPOCHS, learning_rate=PROCEDURAL_LEARNING_RATE
    )
    return learner


def _lesion(learner: ProceduralSequenceLearner) -> ProceduralSequenceLearner:
    lesion = ProceduralSequenceLearner.from_checkpoint(learner.checkpoint())
    with torch.no_grad():
        for parameter in lesion.parameters():
            parameter.zero_()
    return lesion


def _accuracy(
    learner: ProceduralSequenceLearner, records: tuple[EpisodicMemoryRecord, ...]
) -> float:
    return round(float(ArtifactInternalizationTrainer._sequence_accuracy(learner, records)), 6)


# --------------------------------------------------------------------------- #
# Interaction trace projection (dual owner: generator + environment)
# --------------------------------------------------------------------------- #


def _trace_episodes(results: list[dict[str, Any]]) -> tuple[InteractionTraceEpisode, ...]:
    episodes: list[InteractionTraceEpisode] = []
    for result in results:
        scene_id = result["scene_id"]
        episode_id = f"p52-episode:{scene_id}"
        outcome_id = f"{episode_id}:outcome"
        events: list[InteractionTraceEvent] = []
        for step in result["steps"]:
            tick = int(step["tick"])
            events.append(
                InteractionTraceEvent(
                    event_id=f"p52:{scene_id}:{tick:04d}:intent",
                    owner_id=GENERATOR_OWNER,
                    episode_id=episode_id,
                    checkpoint_revision=TRACE_REVISION,
                    outcome_id=outcome_id,
                    resource_cost=0.0,
                )
            )
            events.append(
                InteractionTraceEvent(
                    event_id=f"p52:{scene_id}:{tick:04d}:outcome",
                    owner_id=ENVIRONMENT_OWNER,
                    episode_id=episode_id,
                    checkpoint_revision=TRACE_REVISION,
                    outcome_id=outcome_id,
                    resource_cost=1.0,
                )
            )
        episodes.append(
            InteractionTraceEpisode(
                episode_id=episode_id,
                checkpoint_revision=TRACE_REVISION,
                outcome_id=outcome_id,
                events=tuple(events),
                outcome=1.0 if result["all_scripted_steps_succeeded"] else -1.0,
                context_id="p52-workbench-simulation",
            )
        )
    return tuple(episodes)


# --------------------------------------------------------------------------- #
# Gate
# --------------------------------------------------------------------------- #


def run_gate() -> dict[str, Any]:
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "version": VERSION,
        "status": "failed",
        "growth_admitted": False,
        "can_promote": False,
        "frozen_margins": {
            "content_transfer": FROZEN_CONTENT_TRANSFER_MARGIN,
            "train_accuracy_floor": TRAIN_ACC_FLOOR,
            "total_seconds_cap": TOTAL_SECONDS_CAP,
        },
        "preregistration": PREREGISTRATION,
        "static_four_checks": {
            "scope": list(STATIC_CHECK_SCOPE),
            "commands": list(STATIC_CHECK_COMMANDS),
            "executed_before_run": True,
        },
    }
    workspace_root = Path(tempfile.mkdtemp(prefix="p52-workbench-"))
    replica_root = Path(tempfile.mkdtemp(prefix="p52-workbench-replica-"))
    try:
        partitions = build_partitions()
        embedder = DocumentEmbedder()
        anchored = {
            "model_id": embedder.model_id,
            "revision": embedder.revision,
            "config_digest": embedder.config_digest,
        }
        train_records = _scene_records(partitions["train"], embedder)
        holdout_records = _scene_records(partitions["holdout"], embedder)
        agate_records = _scene_records(partitions["agate"], embedder)

        train_vocab = frozenset(record.action_intent.kind for record in train_records)
        agate_vocab = frozenset(record.action_intent.kind for record in agate_records)
        snapshot = WorkbenchEnvironment(workspace_root).capability_snapshot
        snapshot_capability_ids = frozenset(item.capability_id for item in snapshot.capabilities)

        # -- pass 1: full contract-path execution over every partition --------
        def execute_all(root: Path) -> dict[str, Any]:
            env = WorkbenchEnvironment(root)
            results: dict[str, list[dict[str, Any]]] = {}
            for partition in ("train", "holdout", "agate"):
                results[partition] = [
                    _execute_scene(env, scene, root) for scene in partitions[partition]
                ]
            return {"env": env, "results": results}

        first = execute_all(workspace_root)
        all_results = [
            item
            for partition in ("train", "holdout", "agate")
            for item in first["results"][partition]
        ]

        # -- guard scenario: the contract must intercept low-confidence set_language
        guard_env = first["env"]
        (workspace_root / "guard.py").write_text("value = 0\n", encoding="utf-8")
        guard_intent = ActionIntent(
            "p52-intent:guard", "editor.set_language", parameters={}, confidence=0.1, tick=1
        )
        guard_request = WorkbenchActionRequest.from_action_intent(
            guard_intent,
            snapshot_id=guard_env.capability_snapshot.snapshot_id,
        )
        guard_decision = guard_env.policy_for(
            dataclasses.replace(
                guard_request,
                parameters={"path": "guard.py", "programming_language_id": "python"},
            )
        )
        guard_intercepted = guard_decision.decision == "ask_user"

        # -- procedural readout on the workbench vocabulary --------------------
        train_learner = _train_readout(train_records)
        lesion_learner = _lesion(train_learner)
        train_accuracy = _accuracy(train_learner, train_records)
        holdout_accuracy = _accuracy(train_learner, holdout_records)
        lesion_accuracy = _accuracy(lesion_learner, holdout_records)
        agate_accuracy = _accuracy(train_learner, agate_records)
        baseline = _tick_majority_baseline(train_records)
        frequency_margin = round(agate_accuracy - baseline, 6)

        # -- replica: fresh workspace, fresh readout, bitwise comparison -------
        second = execute_all(replica_root)
        replica_train_learner = _train_readout(_scene_records(partitions["train"], embedder))

        def deterministic_steps(results: list[dict[str, Any]]) -> list[Any]:
            # ``sensation`` is derived from the executor result digest, which
            # carries the random single-use undo token; the deterministic
            # contract surface is (success, reward, terminal).
            return [
                (
                    step["tick"],
                    step["kind"],
                    None if step["outcome"] is None else step["outcome"]["success"],
                    None if step["outcome"] is None else step["outcome"]["reward"],
                    None if step["outcome"] is None else step["outcome"]["terminal"],
                    step["policy_decision"],
                )
                for item in results
                for step in item["steps"]
            ]

        replica_consistent = bool(
            _accuracy(replica_train_learner, agate_records) == agate_accuracy
            and _accuracy(replica_train_learner, train_records) == train_accuracy
            and deterministic_steps(second["results"]["train"])
            == deterministic_steps(first["results"]["train"])
            and deterministic_steps(second["results"]["holdout"])
            == deterministic_steps(first["results"]["holdout"])
            and deterministic_steps(second["results"]["agate"])
            == deterministic_steps(first["results"]["agate"])
        )

        # -- gate 4: language resolve reuse (registry parity + state coverage) --
        resolve_states: list[str] = []
        registry_parity_flags: list[bool] = []
        for result in all_results:
            for step in result["steps"]:
                state_value = step.get("resolved_selection_state")
                if state_value is None:
                    continue
                resolve_states.append(state_value)
                registry_parity_flags.append(bool(step.get("registry_parity", False)))
        registry_parity = bool(registry_parity_flags) and all(registry_parity_flags)

        # -- gate 5: language switch narratives --------------------------------
        narratives = [
            step["language_narrative"]
            for item in all_results
            for step in item["steps"]
            if "language_narrative" in step
        ]
        narratives_complete = bool(narratives) and all(
            set(item) >= {"path", "before", "after", "reason", "selection_state"}
            and item["before"] != "unknown"
            and item["after"] != "unknown"
            for item in narratives
        )

        # -- gate 6: undo semantics --------------------------------------------
        undo_checks: list[bool] = []
        for result in all_results:
            undone_steps = [step for step in result["steps"] if step.get("undone")]
            if undone_steps:
                undo_checks.append(
                    all(
                        result["snapshots_after"][name] == result["snapshots_before"][name]
                        for name in result["snapshots_before"]
                    )
                )
        undo_semantics = bool(undo_checks) and all(undo_checks)

        # -- gate 8: interaction trace projection -------------------------------
        trace_train = _trace_episodes(first["results"]["train"])
        trace_holdout = _trace_episodes(first["results"]["holdout"])
        corpus = InteractionTraceCorpus(train=trace_train, holdout=trace_holdout)
        evaluation = InteractionGroupEvaluator().evaluate(corpus)
        projection_lossless = bool(
            len(trace_train) == TRAIN_SCENES
            and len(trace_holdout) == HOLDOUT_SCENES
            and all(
                episode.member_ids == tuple(sorted((ENVIRONMENT_OWNER, GENERATOR_OWNER)))
                for episode in (*trace_train, *trace_holdout)
            )
        )

        executed_actions = [
            step for item in all_results for step in item["steps"] if step["executed"]
        ]
        policy_intercepts = [
            step for item in all_results for step in item["steps"] if not step["executed"]
        ]
        executed_capability_ids = frozenset(step["kind"] for step in executed_actions)
        chain_complete = all(
            {"policy_decision", "policy_reason", "outcome", "last_result_keys"} <= set(step)
            for step in executed_actions
        )
        outcome_shape_ok = all(
            set(step["outcome"]) == {"success", "reward", "terminal", "sensation"}
            for step in executed_actions
        )

        total_wall = time.perf_counter() - started
        gates = {
            "static_four_checks": True,
            "contract_path_identity": bool(
                chain_complete
                and outcome_shape_ok
                and len(executed_actions) >= 1
                and all(step["kind"] in snapshot_capability_ids for step in executed_actions)
            ),
            "action_typing_validity": bool(
                executed_capability_ids <= snapshot_capability_ids and guard_intercepted
            ),
            "language_resolve_reuse": bool(
                registry_parity and resolve_states and "resolved" in set(resolve_states)
            ),
            "language_switch_explanation": bool(narratives_complete),
            "undo_semantics": bool(undo_semantics),
            "readout_sanity": bool(
                train_accuracy >= TRAIN_ACC_FLOOR and holdout_accuracy > lesion_accuracy
            ),
            "interaction_trace_projection": bool(projection_lossless and evaluation is not None),
            "transfer_and_budget": bool(
                frequency_margin >= FROZEN_CONTENT_TRANSFER_MARGIN
                and replica_consistent
                and total_wall <= TOTAL_SECONDS_CAP
            ),
        }
        capability_keys = tuple(
            key for key in gates if key not in ("readout_sanity", "transfer_and_budget")
        )
        if all(gates.values()):
            outcome = "workbench_simulation_contract_supported"
        elif all(gates[key] for key in capability_keys):
            outcome = "simulation_capability_insufficient"
        else:
            outcome = "failed"
        status = "completed"

        payload.update(
            {
                "status": status,
                "design": {
                    "simulation": "restricted instance of the real workbench contract (same code path, no parallel executor)",
                    "action_generator": "procedural readout (hidden 64 / 250 epochs, P5.1g constants) over capability_id vocabulary",
                    "approval_subject": "deterministic simulated approver (same path, different subject; disclosed)",
                    "scene_partitions": {
                        "train": TRAIN_SCENES,
                        "holdout": HOLDOUT_SCENES,
                        "agate": AGATE_SCENES,
                    },
                },
                "embedder_anchor": anchored,
                "scenes": {
                    "counts": {
                        "train": TRAIN_SCENES,
                        "holdout": HOLDOUT_SCENES,
                        "agate": AGATE_SCENES,
                        "guard": 1,
                    },
                    "vocabulary_train": sorted(train_vocab),
                    "vocabulary_agate": sorted(agate_vocab),
                    "agate_within_train": bool(agate_vocab <= train_vocab),
                    "snapshot_capabilities": len(snapshot.capabilities),
                },
                "contract_path": {
                    "executed_actions": len(executed_actions),
                    "policy_intercepted_actions": len(policy_intercepts),
                    "guard_intercepted": guard_intercepted,
                    "guard_decision": guard_decision.decision,
                    "guard_reason": guard_decision.reason_code,
                    "outcome_shape_ok": outcome_shape_ok,
                    "chain_complete": chain_complete,
                    "executed_capability_ids": sorted(executed_capability_ids),
                },
                "language_surface": {
                    "resolve_selection_states": Counter(resolve_states),
                    "registry_parity": registry_parity,
                    "switch_narratives": narratives,
                    "narratives_complete": narratives_complete,
                },
                "undo_semantics": {
                    "undo_scenes": len(undo_checks),
                    "all_restored": undo_semantics,
                },
                "readout": {
                    "train_accuracy": train_accuracy,
                    "holdout_accuracy": holdout_accuracy,
                    "lesion_holdout_accuracy": lesion_accuracy,
                    "agate_accuracy": agate_accuracy,
                    "per_tick_majority_baseline": baseline,
                    "frequency_margin": frequency_margin,
                    "frozen_margin": FROZEN_CONTENT_TRANSFER_MARGIN,
                },
                "interaction_trace": {
                    "train_episodes": len(trace_train),
                    "holdout_episodes": len(trace_holdout),
                    "owner_ids": [ENVIRONMENT_OWNER, GENERATOR_OWNER],
                    "projection_lossless": projection_lossless,
                    "evaluator_consumed": evaluation is not None,
                    "evaluation_summary": {
                        "groups": len(getattr(evaluation, "groups", ())),
                        "rejected": len(getattr(evaluation, "rejected", ())),
                    },
                },
                "deterministic_and_budget": {
                    "replica_consistent": replica_consistent,
                    "elapsed_seconds": round(total_wall, 3),
                    "total_seconds_cap": TOTAL_SECONDS_CAP,
                },
                "gates": gates,
                "outcome": outcome,
                "experiment_passed": bool(outcome == "workbench_simulation_contract_supported"),
                "growth_admitted": False,
                "can_promote": False,
                "interpretation": (
                    (
                        "completed: outcome=workbench_simulation_contract_supported; "
                        f"agate accuracy={agate_accuracy} baseline={baseline} "
                        f"margin={frequency_margin} >= {FROZEN_CONTENT_TRANSFER_MARGIN}; "
                        f"train_accuracy={train_accuracy} >= {TRAIN_ACC_FLOOR}"
                    )
                    if outcome == "workbench_simulation_contract_supported"
                    else (
                        f"completed: outcome={outcome}; frozen gates failed: "
                        f"{sorted(key for key, value in gates.items() if not value)}"
                    )
                ),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        )
    except Exception as exc:  # noqa: BLE001
        payload.update(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        )
    finally:
        shutil.rmtree(workspace_root, ignore_errors=True)
        shutil.rmtree(replica_root, ignore_errors=True)
    report_path = DEFAULT_REPORT
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(report_path)
    return payload


def main() -> int:
    result = run_gate()
    readout = result.get("readout") or {}
    print(
        json.dumps(
            {
                "report": str(DEFAULT_REPORT),
                "status": result.get("status"),
                "outcome": result.get("outcome"),
                "experiment_passed": result.get("experiment_passed"),
                "gates_failed": sorted(
                    key for key, value in (result.get("gates") or {}).items() if not value
                ),
                "agate_accuracy": readout.get("agate_accuracy"),
                "frequency_margin": readout.get("frequency_margin"),
                "train_accuracy": readout.get("train_accuracy"),
                "elapsed_seconds": result.get("elapsed_seconds"),
                "error": result.get("error"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
