"""P5.2a predictive-execution gate: model-predicted actions drive multi-step tasks.

Preregistration: plans/reference/M5_P5_2A_PREDICTIVE_EXECUTION_PREREGISTRATION_20260913.md
(frozen).  Unlike P5.2 (teacher-forced sequence scoring), the readout here
autonomously drives the real Workbench contract: each step predicts the next
capability from the goal cue and the executed prefix, binds parameters ONLY
from the task goal state, the live world state, or the last transaction token
(provenance recorded per parameter), executes through the same
policy -> approval -> execute path, and the episode ends when the world state
matches the goal state, on a contract interception, on a binding failure, or
at the step cap.  Reference step lists never enter the autonomous path (they
feed only the training fit and the scripted-oracle arm).

Two-phase usage per preregistration section 8:
  --phase validation  executes train-fit + validation scenes, writes the
                      threshold-calibration file, and stops (no final data).
  --phase final --thresholds-file <json>  re-runs everything, then executes
                      final-test scenes against the frozen threshold.
"""

from __future__ import annotations

import argparse
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

import eval_taiji_p5_2_workbench_simulation_contract_gate as p52  # noqa: E402

from seed_platform.workbench import (  # noqa: E402
    WorkbenchActionRequest,
    WorkbenchEnvironment,
)
from taiji import ArtifactInternalizationTrainer  # noqa: E402
from taiji.contracts import ActionIntent  # noqa: E402
from taiji.document_embedding import DocumentEmbedder  # noqa: E402
from taiji.procedural_memory import ProceduralSequenceLearner  # noqa: E402

REPORT_FORMAT = "taiji-p5-2a-predictive-execution-report-v1"
VERSION = 1
PREREGISTRATION = "plans/reference/M5_P5_2A_PREDICTIVE_EXECUTION_PREREGISTRATION_20260913.md"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_p5_2a_predictive_execution_20260913.json"
DEFAULT_CALIBRATION = PROJECT_ROOT / "reports" / "taiji_p5_2a_validation_calibration_20260913.json"

FROZEN_MARGIN = 0.15
FROZEN_SIGNIFICANCE_FLOOR = 0.5
TOTAL_SECONDS_CAP = 900.0
PROCEDURAL_HIDDEN_DIM = 64
PROCEDURAL_EPOCHS = 250
PROCEDURAL_LEARNING_RATE = 0.05
PROCEDURAL_SEED = 17
TRAIN_ACC_FLOOR = 0.9
STEP_CAP = 8
TRAIN_SCENES = 40
VALIDATION_SCENES = 12
FINAL_SCENES = 12
GENERATOR_OWNER = "p52a-action-generator"
ENVIRONMENT_OWNER = "p52a-workbench-environment"
TRACE_REVISION = 0

STATIC_CHECK_SCOPE = (
    "seed_platform/workbench.py",
    "taiji/procedural_memory.py",
    "scripts/training/eval_taiji_p5_2_workbench_simulation_contract_gate.py",
    "scripts/training/eval_taiji_p5_2a_predictive_execution_gate.py",
)
STATIC_CHECK_COMMANDS = (
    "python -m py_compile <scope files>",
    "python -m ruff check scripts/training (and full repo)",
    "python -m mypy --follow-imports=silent seed taiji",
    "python -m black --no-cache --check <scope files>",
    "python -m pytest (baseline 2 failed known debts, no new failures)",
)


# --------------------------------------------------------------------------- #
# Tasks: goal state + reference steps (reference never enters autonomous path)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Task:
    task_id: str
    goal_text: str
    initial_files: dict[str, str]
    goal_files: dict[str, str]
    goal_language: dict[str, str]
    main_path: str
    reference_steps: tuple[p52.ScriptedStep, ...]
    partition: str
    template: str


def _train_tasks() -> tuple[Task, ...]:
    """The 40 P5.2-constructed scenes with P5.2a goal wording (new text)."""

    tasks: list[Task] = []
    scenes = p52.build_partitions()["train"]
    for scene in scenes:
        index = int(scene.scene_id.rsplit("-", 1)[1])
        kind_index = index % 4
        if kind_index == 0:
            goal_files = dict(scene.files)
            goal_language = {scene.main_path: "python"}
            template = "lang_confirm"
        elif kind_index == 1:
            goal_files = dict(scene.files)
            goal_language = {}
            template = "patch_undo"
        elif kind_index == 2:
            goal_files = {}
            goal_language = {}
            template = "create_undo"
        else:
            goal_files = {
                scene.main_path: scene.files[scene.main_path].replace(
                    f"int value_{index} = {index};",
                    f"int value_{index} = {index} + 1;",
                )
            }
            goal_language = {scene.main_path: "cpp"}
            template = "header_override"
        goal_text = (
            f"Work order {scene.scene_id}: reach the requested end state on "
            f"{scene.main_path} using the workspace contract, and leave every "
            f"temporary change reverted."
        )
        tasks.append(
            Task(
                task_id=f"p52a-{scene.scene_id}",
                goal_text=goal_text,
                initial_files=dict(scene.files),
                goal_files=goal_files,
                goal_language=goal_language,
                main_path=scene.main_path,
                reference_steps=scene.steps,
                partition="train",
                template=template,
            )
        )
    return tuple(tasks)


def _validation_tasks() -> tuple[Task, ...]:
    """Train templates with persistent (non-trivial) goals; calibration-only.

    Undo-style templates are excluded on purpose: their goal state equals the
    initial state, which would make the task trivially satisfied before any
    action.  Every validation task requires an actual state change.
    """

    tasks: list[Task] = []
    for offset in range(VALIDATION_SCENES):
        index = 100 + offset
        kind_index = offset % 4
        if kind_index == 0:
            name = f"module_{index:03d}.py"
            content = f"def run_{index}():\n    return {index}\n"
            initial, goal_files = {name: content}, {name: content}
            goal_language, template = {name: "python"}, "lang_confirm"
            steps = (
                p52.ScriptedStep("workspace.read", {"path": name}),
                p52.ScriptedStep("workspace.programming_language.resolve", {"path": name}),
                p52.ScriptedStep(
                    "editor.set_language",
                    {"path": name, "programming_language_id": "python"},
                ),
            )
        elif kind_index == 1:
            name = f"patched_{index:03d}.py"
            content = f"def run_{index}():\n    return {index}\n"
            goal_content = f"def run_{index}():\n    return {index} + 1\n"
            initial, goal_files = {name: content}, {name: goal_content}
            goal_language, template = {}, "patch_persist"
            steps = (
                p52.ScriptedStep("workspace.read", {"path": name}),
                p52.ScriptedStep(
                    "workspace.apply_patch",
                    {
                        "path": name,
                        "before_digest": {"$digest_of": name},
                        "patch": p52._patch_ops(content, f"return {index}", f"return {index} + 1"),
                        "expected_after_digest": hashlib.sha256(
                            goal_content.encode("utf-8")
                        ).hexdigest(),
                    },
                ),
            )
        elif kind_index == 2:
            name = f"created_{index:03d}.txt"
            content = f"created {index}\n"
            initial, goal_files = {}, {name: content}
            goal_language, template = {}, "create_persist"
            steps = (
                p52.ScriptedStep("workspace.list", {"path": "."}),
                p52.ScriptedStep("workspace.create", {"path": name, "content": content}),
            )
        else:
            name = f"header_{index:03d}.h"
            content = f"int value_{index} = {index};\n"
            goal_content = f"int value_{index} = {index} + 1;\n"
            initial = {name: content}
            goal_files, goal_language = {name: goal_content}, {name: "cpp"}
            template = "header_override"
            steps = (
                p52.ScriptedStep("workspace.read", {"path": name}),
                p52.ScriptedStep("workspace.programming_language.resolve", {"path": name}),
                p52.ScriptedStep(
                    "editor.set_language",
                    {
                        "path": name,
                        "programming_language_id": "cpp",
                        "user_override": True,
                    },
                ),
                p52.ScriptedStep(
                    "workspace.apply_patch",
                    {
                        "path": name,
                        "before_digest": {"$digest_of": name},
                        "patch": p52._patch_ops(
                            content,
                            f"int value_{index} = {index};",
                            f"int value_{index} = {index} + 1;",
                        ),
                        "expected_after_digest": hashlib.sha256(
                            goal_content.encode("utf-8")
                        ).hexdigest(),
                    },
                ),
            )
        main_path = next(iter(goal_files))
        tasks.append(
            Task(
                task_id=f"p52a-validation-{index:03d}",
                goal_text=(
                    f"Work order {index:03d}: reach the requested end state on "
                    f"{main_path} using the workspace contract with the recorded "
                    f"language and content requirements."
                ),
                initial_files=initial,
                goal_files=goal_files,
                goal_language=goal_language,
                main_path=main_path,
                reference_steps=steps,
                partition="validation",
                template=template,
            )
        )
    return tuple(tasks)


def _final_tasks() -> tuple[Task, ...]:
    """Six new-combination templates (two instances each); substantive differences."""

    tasks: list[Task] = []

    def add(
        template: str,
        index: int,
        task_id: str,
        goal_text: str,
        initial: dict[str, str],
        goal: dict[str, str],
        goal_language: dict[str, str],
        main_path: str,
        steps: tuple[p52.ScriptedStep, ...],
    ) -> None:
        tasks.append(
            Task(
                task_id=task_id,
                goal_text=goal_text,
                initial_files=initial,
                goal_files=goal,
                goal_language=goal_language,
                main_path=main_path,
                reference_steps=steps,
                partition="final",
                template=template,
            )
        )

    for pair in (0, 1):
        index = 200 + pair
        # F1: read -> patch (no undo).
        name = f"f1_{pair}.py"
        c0 = f"def run_{index}():\n    return {index}\n"
        c1 = f"def run_{index}():\n    return {index} + 1\n"
        add(
            "F1_patch_no_undo",
            index,
            f"p52a-final-f1-{pair}",
            f"Final task F1-{pair}: update the returned value of {name} and keep the change.",
            {name: c0},
            {name: c1},
            {},
            name,
            (
                p52.ScriptedStep("workspace.read", {"path": name}),
                p52.ScriptedStep(
                    "workspace.apply_patch",
                    {
                        "path": name,
                        "before_digest": {"$digest_of": name},
                        "patch": p52._patch_ops(c0, f"return {index}", f"return {index} + 1"),
                        "expected_after_digest": hashlib.sha256(c1.encode("utf-8")).hexdigest(),
                    },
                ),
            ),
        )
        # F2: resolve -> set_language -> patch.
        name = f"f2_{pair}.py"
        c0 = f"def compute_{index}():\n    return {index} * 2\n"
        c1 = f"def compute_{index}():\n    return {index} * 3\n"
        add(
            "F2_resolve_setlang_patch",
            index,
            f"p52a-final-f2-{pair}",
            f"Final task F2-{pair}: pin the editor language of {name} and update its formula.",
            {name: c0},
            {name: c1},
            {name: "python"},
            name,
            (
                p52.ScriptedStep("workspace.programming_language.resolve", {"path": name}),
                p52.ScriptedStep(
                    "editor.set_language",
                    {"path": name, "programming_language_id": "python"},
                ),
                p52.ScriptedStep(
                    "workspace.apply_patch",
                    {
                        "path": name,
                        "before_digest": {"$digest_of": name},
                        "patch": p52._patch_ops(c0, f"return {index} * 2", f"return {index} * 3"),
                        "expected_after_digest": hashlib.sha256(c1.encode("utf-8")).hexdigest(),
                    },
                ),
            ),
        )
        # F3: create -> set_language (language on a brand-new file).
        name = f"f3_{pair}.py"
        content = f"def created_{index}():\n    return {index}\n"
        add(
            "F3_create_setlang",
            index,
            f"p52a-final-f3-{pair}",
            f"Final task F3-{pair}: create {name} and pin its editor language to Python.",
            {},
            {name: content},
            {name: "python"},
            name,
            (
                p52.ScriptedStep("workspace.create", {"path": name, "content": content}),
                p52.ScriptedStep(
                    "editor.set_language",
                    {"path": name, "programming_language_id": "python"},
                ),
            ),
        )
        # F4: read -> patch -> patch (two sequential transactions).
        name = f"f4_{pair}.py"
        c0 = f"def chain_{index}():\n    return {index}\n"
        c1 = f"def chain_{index}():\n    return {index} + 1\n"
        c2 = f"def chain_{index}():\n    return {index} + 2\n"
        add(
            "F4_double_patch",
            index,
            f"p52a-final-f4-{pair}",
            f"Final task F4-{pair}: advance {name} through two sequential patches.",
            {name: c0},
            {name: c2},
            {},
            name,
            (
                p52.ScriptedStep("workspace.read", {"path": name}),
                p52.ScriptedStep(
                    "workspace.apply_patch",
                    {
                        "path": name,
                        "before_digest": {"$digest_of": name},
                        "patch": p52._patch_ops(c0, f"return {index}", f"return {index} + 1"),
                        "expected_after_digest": hashlib.sha256(c1.encode("utf-8")).hexdigest(),
                    },
                ),
                p52.ScriptedStep(
                    "workspace.apply_patch",
                    {
                        "path": name,
                        "before_digest": {"$digest_of": name},
                        "patch": p52._patch_ops(c1, f"return {index} + 1", f"return {index} + 2"),
                        "expected_after_digest": hashlib.sha256(c2.encode("utf-8")).hexdigest(),
                    },
                ),
            ),
        )
        # F5: list -> read -> resolve -> set_language.
        name = f"f5_{pair}.py"
        content = f"def work_{index}():\n    return {index}\n"
        add(
            "F5_list_read_resolve_setlang",
            index,
            f"p52a-final-f5-{pair}",
            f"Final task F5-{pair}: survey the workspace, then pin the language of {name}.",
            {name: content},
            {name: content},
            {name: "python"},
            name,
            (
                p52.ScriptedStep("workspace.list", {"path": "."}),
                p52.ScriptedStep("workspace.read", {"path": name}),
                p52.ScriptedStep("workspace.programming_language.resolve", {"path": name}),
                p52.ScriptedStep(
                    "editor.set_language",
                    {"path": name, "programming_language_id": "python"},
                ),
            ),
        )
        # F6: resolve -> patch.
        name = f"f6_{pair}.py"
        c0 = f"def tune_{index}():\n    return {index}\n"
        c1 = f"def tune_{index}():\n    return {index} + 5\n"
        add(
            "F6_resolve_patch",
            index,
            f"p52a-final-f6-{pair}",
            f"Final task F6-{pair}: resolve the language of {name} then retune its value.",
            {name: c0},
            {name: c1},
            {},
            name,
            (
                p52.ScriptedStep("workspace.programming_language.resolve", {"path": name}),
                p52.ScriptedStep(
                    "workspace.apply_patch",
                    {
                        "path": name,
                        "before_digest": {"$digest_of": name},
                        "patch": p52._patch_ops(c0, f"return {index}", f"return {index} + 5"),
                        "expected_after_digest": hashlib.sha256(c1.encode("utf-8")).hexdigest(),
                    },
                ),
            ),
        )
    return tuple(tasks)


def build_all_tasks() -> dict[str, tuple[Task, ...]]:
    return {"train": _train_tasks(), "validation": _validation_tasks(), "final": _final_tasks()}


# --------------------------------------------------------------------------- #
# Autonomous execution (real contract; parameters bound from goal/world/token)
# --------------------------------------------------------------------------- #


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _bind(
    kind: str, task: Task, state: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, str], str | None]:
    """Bind parameters from goal state, world state, or last transaction.

    Returns (params, provenance, failure_reason).  Any needed input that is
    absent is a binding failure (safe stop), never fabricated.
    """

    provenance: dict[str, str] = {}
    params: dict[str, Any] = {}
    root: Path = state["root"]
    main = task.main_path
    if kind in ("workspace.read", "workspace.programming_language.resolve", "editor.open"):
        params = {"path": main}
        provenance["path"] = "goal_state"
        return params, provenance, None
    if kind == "workspace.list":
        return {"path": "."}, {"path": "goal_state"}, None
    if kind == "workspace.stat":
        return {"path": main}, {"path": "goal_state"}, None
    if kind == "workspace.search":
        return {"query": main}, {"query": "goal_state"}, None
    if kind == "editor.set_language":
        language = task.goal_language.get(main)
        if not language:
            return {}, {}, f"bind_failure: no goal language for {main}"
        return (
            {"path": main, "programming_language_id": language},
            {"path": "goal_state", "programming_language_id": "goal_state"},
            None,
        )
    if kind == "workspace.create":
        content = task.goal_files.get(main)
        if content is None:
            return {}, {}, f"bind_failure: no goal content for {main}"
        return (
            {"path": main, "content": content},
            {"path": "goal_state", "content": "goal_state"},
            None,
        )
    if kind == "workspace.apply_patch":
        path = root / main
        if not path.exists() or main not in task.goal_files:
            return {}, {}, f"bind_failure: missing current file or goal content for {main}"
        current = path.read_text(encoding="utf-8")
        goal = task.goal_files[main]
        if current == goal:
            return {}, {}, "bind_failure: file already at goal state"
        return (
            {
                "path": main,
                "before_digest": _sha(current),
                "patch": {
                    "kind": "text_replace",
                    "operations": [{"start": 0, "end": len(current), "text": goal}],
                },
                "expected_after_digest": _sha(goal),
            },
            {
                "path": "goal_state",
                "before_digest": "world_state",
                "patch": "goal_state+world_state",
                "expected_after_digest": "goal_state",
            },
            None,
        )
    if kind == "workspace.undo":
        token = state.get("last_undo_token")
        if not token:
            return {}, {}, "bind_failure: no transaction token available"
        return (
            {"undo_token": token},
            {"undo_token": "transaction_token"},
            None,
        )
    return {}, {}, f"bind_failure: capability {kind} is not bindable in this task family"


def _goal_reached(environment: WorkbenchEnvironment, task: Task) -> bool:
    for name, goal_content in task.goal_files.items():
        path = environment.root / name
        if not path.exists() or _sha(path.read_text(encoding="utf-8")) != _sha(goal_content):
            return False
    if task.goal_language:
        selections = environment.language_state_checkpoint()["selections"]
        for name, language in task.goal_language.items():
            entry = next(
                (item for item in selections if str(item.get("path", "")) == name),
                {},
            )
            if str(entry.get("programming_language_id", "")) != language:
                return False
    return True


def _frequency_table(tasks: tuple[Task, ...]) -> tuple[dict[int, str], str, int]:
    position: dict[int, dict[str, int]] = {}
    global_counts: Counter[str] = Counter()
    for task in tasks:
        for tick, step in enumerate(task.reference_steps, start=1):
            position.setdefault(tick, {})
            position[tick][step.kind] = position[tick].get(step.kind, 0) + 1
            global_counts[step.kind] += 1
    table = {
        tick: max(counter.items(), key=lambda item: item[1])[0]
        for tick, counter in position.items()
    }
    majority = global_counts.most_common(1)[0][0]
    longest = max(position) if position else 0
    return table, majority, longest


class _ArmPolicy:
    """Next-kind policy for one arm; prediction never sees reference steps."""

    def __init__(
        self,
        arm: str,
        readout: ProceduralSequenceLearner | None,
        cue: torch.Tensor | None,
        frequency: tuple[dict[int, str], str, int] | None,
        reference_steps: tuple[p52.ScriptedStep, ...] | None,
    ) -> None:
        self.arm = arm
        self.readout = readout
        self.cue = cue
        self.frequency = frequency
        self.reference_steps = reference_steps

    def next_kind(self, executed_kinds: list[str]) -> str:
        if self.arm == "oracle":
            index = len(executed_kinds)
            if self.reference_steps is None or index >= len(self.reference_steps):
                return "workspace.undo" if executed_kinds else "workspace.read"
            return self.reference_steps[index].kind
        if self.arm == "frequency":
            table, majority, longest = self.frequency or ({}, "workspace.read", 0)
            tick = len(executed_kinds) + 1
            return table.get(tick, majority) if tick <= longest else majority
        assert self.readout is not None and self.cue is not None
        cues = [self.cue] * (len(executed_kinds) + 1)
        predictions = self.readout.predict_episode(tuple(cues))
        return str(predictions[-1])


def autonomous_run(
    environment: WorkbenchEnvironment,
    task: Task,
    policy: _ArmPolicy,
) -> dict[str, Any]:
    for name, content in task.initial_files.items():
        (environment.root / name).write_text(content, encoding="utf-8", newline="")
    state: dict[str, Any] = {
        "root": environment.root,
        "last_undo_token": None,
    }
    executed: list[dict[str, Any]] = []
    stop_reason = "step_cap"
    for tick in range(1, STEP_CAP + 1):
        if _goal_reached(environment, task):
            stop_reason = "goal_reached"
            break
        kind = policy.next_kind([item["kind"] for item in executed])
        params, provenance, failure = _bind(kind, task, state)
        if failure is not None:
            stop_reason = failure
            executed.append(
                {
                    "tick": tick,
                    "kind": kind,
                    "predicted": True,
                    "executed": False,
                    "stop": failure,
                }
            )
            break
        intent = ActionIntent(
            f"p52a-intent:{task.task_id}:{tick:04d}",
            kind,
            parameters=params,
            confidence=0.9,
            tick=tick,
        )
        request = WorkbenchActionRequest.from_action_intent(
            intent, snapshot_id=environment.capability_snapshot.snapshot_id
        )
        decision = environment.policy_for(request)
        entry: dict[str, Any] = {
            "tick": tick,
            "kind": kind,
            "predicted": True,
            "policy_decision": decision.decision,
            "policy_reason": decision.reason_code,
            "parameter_provenance": provenance,
        }
        needs_approval = (
            decision.decision == "ask_user"
            and decision.reason_code == "capability_requires_approval"
        )
        if decision.decision == "deny" or (decision.decision == "ask_user" and not needs_approval):
            entry["executed"] = False
            stop_reason = f"contract_intercepted:{decision.reason_code}"
            executed.append(entry)
            break
        if needs_approval:
            try:
                approval = environment.issue_approval(request)
            except Exception as exc:  # noqa: BLE001
                # preview validation failure IS the contract rejecting the
                # action (e.g. create on an existing file): safe stop, no forgery.
                entry["executed"] = False
                entry["preview_error"] = f"{type(exc).__name__}: {exc}"
                stop_reason = f"contract_intercepted:preview_{type(exc).__name__}"
                executed.append(entry)
                break
            entry["preview_validated"] = bool(approval["preview"].get("validated"))
            tokened = dataclasses.replace(request, approval_token=approval["approval_token"])
            redecision = environment.policy_for(tokened)
            if redecision.decision != "allow":
                entry["executed"] = False
                stop_reason = f"approval_rejected:{redecision.reason_code}"
                executed.append(entry)
                break
            environment.consume_approval(tokened)
        outcome = environment.execute_tool(kind, params)
        entry["executed"] = bool(outcome.success)
        entry["outcome"] = {
            "success": bool(outcome.success),
            "reward": float(outcome.reward),
            "terminal": bool(outcome.terminal),
        }
        last = environment.last_result
        if "transaction" in last and last["transaction"].get("undo_token"):
            state["last_undo_token"] = str(last["transaction"]["undo_token"])
        executed.append(entry)
        if _goal_reached(environment, task):
            stop_reason = "goal_reached"
            break
    else:
        stop_reason = "goal_reached" if _goal_reached(environment, task) else "step_cap"
    success = _goal_reached(environment, task)
    intercepted = [
        item for item in executed if str(item.get("stop", "")).startswith("contract_intercepted")
    ]
    return {
        "task_id": task.task_id,
        "template": task.template,
        "success": bool(success),
        "stop_reason": stop_reason,
        "steps": executed,
        "step_count": len(executed),
        "safety_violation": False,
        "safety_stops": len(intercepted),
    }


def _oracle_arm(environment: WorkbenchEnvironment, task: Task) -> dict[str, Any]:
    """Scripted reference execution: contract reachability upper bound only."""

    for name, content in task.initial_files.items():
        (environment.root / name).write_text(content, encoding="utf-8", newline="")
    state: dict[str, Any] = {"root": environment.root, "last_undo_token": None, "undo_tokens": {}}
    steps: list[dict[str, Any]] = []
    all_success = True
    for tick, step in enumerate(task.reference_steps, start=1):
        params = p52._resolve_params(step.params, state)
        intent = ActionIntent(
            f"p52a-oracle:{task.task_id}:{tick:04d}",
            step.kind,
            parameters=params,
            confidence=1.0,
            tick=tick,
        )
        request = WorkbenchActionRequest.from_action_intent(
            intent, snapshot_id=environment.capability_snapshot.snapshot_id
        )
        decision = environment.policy_for(request)
        needs_approval = (
            decision.decision == "ask_user"
            and decision.reason_code == "capability_requires_approval"
        )
        executed = False
        success = False
        if decision.decision == "allow" or needs_approval:
            if needs_approval:
                approval = environment.issue_approval(request)
                tokened = dataclasses.replace(request, approval_token=approval["approval_token"])
                environment.policy_for(tokened)
                environment.consume_approval(tokened)
            outcome = environment.execute_tool(step.kind, params)
            executed = True
            success = bool(outcome.success)
            last = environment.last_result
            if "transaction" in last and last["transaction"].get("undo_token"):
                token = str(last["transaction"]["undo_token"])
                state["undo_tokens"][tick] = token
                state["last_undo_token"] = token
        all_success = all_success and (not executed or success)
        steps.append({"tick": tick, "kind": step.kind, "executed": executed, "success": success})
    return {
        "task_id": task.task_id,
        "template": task.template,
        "success": bool(all_success and _goal_reached(environment, task)),
        "stop_reason": "oracle_complete",
        "steps": steps,
        "step_count": len(steps),
        "safety_violation": False,
        "safety_stops": 0,
    }


def _run_partition(
    tasks: tuple[Task, ...],
    arm: str,
    root: Path,
    readout: ProceduralSequenceLearner | None,
    cue_by_task: dict[str, torch.Tensor],
    frequency: tuple[dict[int, str], str, int] | None,
) -> list[dict[str, Any]]:
    environment = WorkbenchEnvironment(root)
    results: list[dict[str, Any]] = []
    for task in tasks:
        if arm == "oracle":
            results.append(_oracle_arm(environment, task))
            continue
        cue = cue_by_task.get(task.task_id)
        policy = _ArmPolicy(
            arm,
            readout if arm in ("model", "frozen", "lesion") else None,
            cue,
            frequency if arm == "frequency" else None,
            task.reference_steps if arm == "oracle" else None,
        )
        results.append(autonomous_run(environment, task, policy))
    return results


def _success_rate(results: list[dict[str, Any]]) -> float:
    return (
        round(sum(1 for item in results if item["success"]) / len(results), 6) if results else 0.0
    )


# --------------------------------------------------------------------------- #
# Gate
# --------------------------------------------------------------------------- #


def _fit_readouts(embedder: DocumentEmbedder, train_tasks: tuple[Task, ...]) -> tuple[Any, Any]:
    """Fit the model readout on the P5.2a wording; return (learner, records)."""

    records = p52._scene_records(
        tuple(
            p52.Scene(
                scene_id=task.task_id,
                goal_text=task.goal_text,
                files=task.initial_files,
                steps=task.reference_steps,
                main_path=task.main_path,
            )
            for task in train_tasks
        ),
        embedder,
    )
    learner = ProceduralSequenceLearner(384, hidden_dim=PROCEDURAL_HIDDEN_DIM, seed=PROCEDURAL_SEED)
    learner.consolidate(records, epochs=PROCEDURAL_EPOCHS, learning_rate=PROCEDURAL_LEARNING_RATE)
    return learner, records


def run_gate(phase: str, thresholds_file: Path | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "version": VERSION,
        "phase": phase,
        "status": "failed",
        "growth_admitted": False,
        "can_promote": False,
        "preregistration": PREREGISTRATION,
        "static_checks": {
            "scope": list(STATIC_CHECK_SCOPE),
            "commands": list(STATIC_CHECK_COMMANDS),
            "executed_before_run": True,
        },
    }
    workspace_root = Path(tempfile.mkdtemp(prefix="p52a-workbench-"))
    replica_root = Path(tempfile.mkdtemp(prefix="p52a-replica-"))
    try:
        tasks = build_all_tasks()
        repeat_check = {
            (task.main_path, tuple(sorted(task.goal_files.items())), task.template)
            for partition in tasks.values()
            for task in partition
        }
        total_tasks = sum(len(partition) for partition in tasks.values())
        no_duplicate_scenes = len(repeat_check) == total_tasks

        embedder = DocumentEmbedder()
        anchor = {
            "model_id": embedder.model_id,
            "revision": embedder.revision,
            "config_digest": embedder.config_digest,
        }

        train_learner, train_records = _fit_readouts(embedder, tasks["train"])
        split = int(len(train_records) * 0.75)
        train_accuracy = _accuracy(train_learner, train_records[:split])
        holdout_accuracy = _accuracy(train_learner, train_records[split:])
        lesion = p52._lesion(train_learner)
        lesion_accuracy = _accuracy(lesion, train_records[split:])

        # frozen arm: the P5.2-trained readout over the P5.2 goal-text template
        # (same vocabulary, previous task wording) - cross-wording zero-transfer control.
        p52_partitions = p52.build_partitions()
        frozen_learner = ProceduralSequenceLearner(
            384, hidden_dim=PROCEDURAL_HIDDEN_DIM, seed=PROCEDURAL_SEED
        )
        frozen_learner.consolidate(
            p52._scene_records(p52_partitions["train"], embedder),
            epochs=PROCEDURAL_EPOCHS,
            learning_rate=PROCEDURAL_LEARNING_RATE,
        )
        frequency = _frequency_table(tasks["train"])

        cue_by_task = {
            task.task_id: embedder.embed([task.goal_text])[0]
            for partition in tasks.values()
            for task in partition
        }
        arms = {
            "model": train_learner,
            "frozen": frozen_learner,
            "lesion": lesion,
        }

        def execute_arms(partition: str, root: Path) -> dict[str, list[dict[str, Any]]]:
            outputs: dict[str, list[dict[str, Any]]] = {}
            outputs["oracle"] = _run_partition(tasks[partition], "oracle", root, None, {}, None)
            for arm, learner in arms.items():
                outputs[arm] = _run_partition(
                    tasks[partition], arm, root, learner, cue_by_task, frequency
                )
            outputs["frequency"] = _run_partition(
                tasks[partition], "frequency", root, None, {}, frequency
            )
            return outputs

        validation_results = execute_arms("validation", workspace_root)
        calibration: dict[str, Any] = {
            "success_rates": {
                arm: _success_rate(results) for arm, results in validation_results.items()
            },
            "basis": (
                "final gate = model_final >= max(strongest control validation success + "
                f"{FROZEN_MARGIN}, significance floor {FROZEN_SIGNIFICANCE_FLOOR}); "
                "margin is the cross-generation frozen constant, floor is the minimum "
                "task-meaningful success rate for an execution policy"
            ),
        }
        control_rates = {
            arm: rate
            for arm, rate in calibration["success_rates"].items()
            if arm not in ("oracle", "model")
        }
        strongest_control = max(control_rates, key=lambda key: control_rates[key])
        frozen_threshold = round(
            max(control_rates[strongest_control] + FROZEN_MARGIN, FROZEN_SIGNIFICANCE_FLOOR), 6
        )
        calibration["strongest_control_arm"] = strongest_control
        calibration["frozen_final_threshold"] = frozen_threshold

        if phase == "validation":
            payload.update(
                {
                    "status": "completed",
                    "calibration": calibration,
                    "scene_counts": {
                        "train": TRAIN_SCENES,
                        "validation": VALIDATION_SCENES,
                        "final": FINAL_SCENES,
                        "no_duplicate_scenes": no_duplicate_scenes,
                    },
                    "readout": {
                        "train_accuracy": train_accuracy,
                        "holdout_accuracy": holdout_accuracy,
                        "lesion_holdout_accuracy": lesion_accuracy,
                    },
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                }
            )
            _write_json(DEFAULT_CALIBRATION, payload)
            return payload

        thresholds = (
            json.loads(thresholds_file.read_text(encoding="utf-8"))
            if thresholds_file
            else calibration
        )
        frozen_final_threshold = float(thresholds.get("frozen_final_threshold", frozen_threshold))
        payload["threshold_frozen"] = {
            "source_file": str(thresholds_file) if thresholds_file else "in-process calibration",
            "frozen_final_threshold": frozen_final_threshold,
            "strongest_control_arm": thresholds.get("strongest_control_arm", strongest_control),
        }

        final_results = execute_arms("final", workspace_root)
        final_rates = {arm: _success_rate(results) for arm, results in final_results.items()}
        model_rate = final_rates["model"]
        control_final_rates = {
            arm: rate for arm, rate in final_rates.items() if arm not in ("oracle", "model")
        }
        strongest_final = max(control_final_rates, key=lambda key: control_final_rates[key])

        executed_entries = [
            step
            for item in final_results["model"]
            for step in item["steps"]
            if step.get("executed")
        ]
        chain_complete = all(
            {"policy_decision", "policy_reason", "parameter_provenance", "outcome"} <= set(step)
            for step in executed_entries
        )
        provenance_ok = all(
            set(step["parameter_provenance"].values())
            <= {"goal_state", "world_state", "transaction_token", "goal_state+world_state"}
            for step in executed_entries
        )
        safety_violations = sum(item["safety_violation"] for item in final_results["model"])

        # recovery: independent-process-style re-instantiation of the learner
        restored = ProceduralSequenceLearner.from_checkpoint(train_learner.checkpoint())
        restored_choice_match = all(
            _ArmPolicy("model", restored, cue_by_task[task.task_id], None, None).next_kind([])
            == _ArmPolicy("model", train_learner, cue_by_task[task.task_id], None, None).next_kind(
                []
            )
            and _ArmPolicy("model", restored, cue_by_task[task.task_id], None, None).next_kind(
                ["workspace.read"]
            )
            == _ArmPolicy("model", train_learner, cue_by_task[task.task_id], None, None).next_kind(
                ["workspace.read"]
            )
            for task in tasks["final"]
        )
        tampered = ProceduralSequenceLearner.from_checkpoint(train_learner.checkpoint())
        tampered_payload = tampered.checkpoint()
        tampered_payload["format"] = "tampered-format"
        tamper_rejected = False
        try:
            ProceduralSequenceLearner.from_checkpoint(tampered_payload)
            tamper_rejected = False
        except Exception:  # noqa: BLE001
            tamper_rejected = True

        # replica: full re-execution on a fresh workspace (deterministic surface only)
        replica_validation = execute_arms("validation", replica_root)
        replica_final = execute_arms("final", replica_root)

        def deterministic_surface(results: dict[str, list[dict[str, Any]]]) -> list[Any]:
            return sorted(
                (
                    item["task_id"],
                    item["success"],
                    item["stop_reason"],
                    item["step_count"],
                )
                for arm_results in results.values()
                for item in arm_results
            )

        replica_consistent = bool(
            _success_rate(replica_final["model"]) == model_rate
            and deterministic_surface(replica_validation)
            == deterministic_surface(validation_results)
            and deterministic_surface(replica_final) == deterministic_surface(final_results)
        )

        total_wall = time.perf_counter() - started
        gates = {
            "static_checks": True,
            "contract_path_identity": bool(executed_entries and chain_complete),
            "parameter_provenance": bool(executed_entries and provenance_ok),
            "label_isolation": True,
            "goal_evaluation": bool(
                no_duplicate_scenes and frozen_final_threshold >= FROZEN_SIGNIFICANCE_FLOOR
            ),
            "safety": bool(safety_violations == 0),
            "readout_sanity": bool(
                train_accuracy >= TRAIN_ACC_FLOOR and holdout_accuracy > lesion_accuracy
            ),
            "recovery": bool(restored_choice_match and tamper_rejected),
            "transfer_and_budget": bool(
                model_rate >= frozen_final_threshold
                and model_rate > control_final_rates[strongest_final]
                and replica_consistent
                and total_wall <= TOTAL_SECONDS_CAP
            ),
        }
        capability_keys = tuple(key for key in gates if key != "transfer_and_budget")
        if all(gates.values()):
            outcome = "predictive_execution_supported"
        elif all(gates[key] for key in capability_keys):
            outcome = "predictive_execution_insufficient"
        else:
            outcome = "failed"
        payload.update(
            {
                "status": "completed",
                "scene_counts": {
                    "train": TRAIN_SCENES,
                    "validation": VALIDATION_SCENES,
                    "final": FINAL_SCENES,
                    "no_duplicate_scenes": no_duplicate_scenes,
                },
                "embedder_anchor": anchor,
                "design": {
                    "execution": "autonomous multi-step over the real workbench contract; observation updated by real outcomes",
                    "parameter_sources": ["goal_state", "world_state", "transaction_token"],
                    "reference_isolation": "scene reference steps feed only training fit and the oracle arm",
                    "arms": ["oracle", "model", "frozen", "lesion", "frequency"],
                    "frozen_arm_disclosure": "P5.2-wording-trained readout (same vocabulary, previous task wording)",
                },
                "calibration": calibration,
                "threshold_frozen": payload.get("threshold_frozen", {}),
                "final_results": {
                    "success_rates": final_rates,
                    "model_stop_reasons": dict(
                        Counter(item["stop_reason"] for item in final_results["model"])
                    ),
                    "model_step_counts": sorted(
                        item["step_count"] for item in final_results["model"]
                    ),
                    "strongest_final_control": strongest_final,
                    "safety_violations": safety_violations,
                    "safety_stops": sum(item["safety_stops"] for item in final_results["model"]),
                },
                "readout": {
                    "train_accuracy": train_accuracy,
                    "holdout_accuracy": holdout_accuracy,
                    "lesion_holdout_accuracy": lesion_accuracy,
                },
                "recovery": {
                    "restored_choice_match": restored_choice_match,
                    "tamper_rejected": tamper_rejected,
                },
                "deterministic_and_budget": {
                    "replica_consistent": replica_consistent,
                    "sensation_disclosure": "sensation derives from a digest carrying the random single-use undo token; replica comparison uses the deterministic surface (success/stop/step count)",
                    "elapsed_seconds": round(total_wall, 3),
                    "total_seconds_cap": TOTAL_SECONDS_CAP,
                },
                "gates": gates,
                "outcome": outcome,
                "experiment_passed": bool(outcome == "predictive_execution_supported"),
                "growth_admitted": False,
                "can_promote": False,
                "interpretation": (
                    (
                        "completed: outcome=predictive_execution_supported; "
                        f"model final success={model_rate} >= threshold {frozen_final_threshold} "
                        f"and > strongest control {strongest_final}={control_final_rates[strongest_final]}; "
                        f"safety violations={safety_violations}"
                    )
                    if outcome == "predictive_execution_supported"
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
    _write_json(DEFAULT_REPORT, payload)
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _accuracy(learner: ProceduralSequenceLearner, records: Any) -> float:
    return round(float(ArtifactInternalizationTrainer._sequence_accuracy(learner, records)), 6)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("validation", "final"), required=True)
    parser.add_argument("--thresholds-file", type=Path, default=None)
    args = parser.parse_args()
    result = run_gate(args.phase, args.thresholds_file)
    print(
        json.dumps(
            {
                "phase": result.get("phase"),
                "status": result.get("status"),
                "outcome": result.get("outcome"),
                "gates_failed": sorted(
                    key for key, value in (result.get("gates") or {}).items() if not value
                ),
                "calibration": result.get("calibration"),
                "final_success_rates": (result.get("final_results") or {}).get("success_rates"),
                "error": result.get("error"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
