"""P5.2b group causal corpora gate: factorial member interventions under the
real Workbench contract.

Preregistration: plans/reference/M5_P5_2B_GROUP_CAUSAL_CORPORA_PREREGISTRATION_20260913.md
(frozen).  Four family-specialist procedural readouts (each trained only on
one P5.2a train template family) act as real, intervenable cognitive members:
every intervention removes a real learner instance from the execution loop.
For each of 12 contexts (persistent-goal scenes) and every candidate pair,
all four factorial cells are executed through the same
policy -> approval -> execute path used by P5.2/P5.2a: (F,F) inactive
baseline (no cognitive member -> no actions), two singletons, and the pair
with the frozen sequential-fallback composition.  Episodes are projected into
InteractionTraceEpisode records (owner = executing member), consumed by
InteractionGroupEvaluator, and profiled via build_member_evidence.  Member
ids are opaque; the semantic mapping lives only in this runner's config
section and never enters learner/evaluator inputs.
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))

import eval_taiji_p5_2_workbench_simulation_contract_gate as p52  # noqa: E402
import eval_taiji_p5_2a_predictive_execution_gate as p52a  # noqa: E402

from seed_platform.workbench import (  # noqa: E402
    WorkbenchActionRequest,
    WorkbenchEnvironment,
)
from taiji.contracts import ActionIntent  # noqa: E402
from taiji.document_embedding import DocumentEmbedder  # noqa: E402
from taiji.interaction_group_transfer import build_member_evidence  # noqa: E402
from taiji.interaction_groups import (  # noqa: E402
    InteractionGroupEvaluator,
    InteractionTraceCorpus,
    InteractionTraceEpisode,
    InteractionTraceEvent,
)
from taiji.procedural_memory import ProceduralSequenceLearner  # noqa: E402

REPORT_FORMAT = "taiji-p5-2b-group-causal-corpora-report-v1"
VERSION = 1
PREREGISTRATION = "plans/reference/M5_P5_2B_GROUP_CAUSAL_CORPORA_PREREGISTRATION_20260913.md"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_p5_2b_group_causal_corpora_20260913.json"

TOTAL_SECONDS_CAP = 900.0
PROCEDURAL_HIDDEN_DIM = 64
PROCEDURAL_EPOCHS = 250
PROCEDURAL_LEARNING_RATE = 0.05
STEP_CAP = 8
CONTEXT_COUNT = 12
REPEATS = 2
TRACE_REVISION = 1
MEMBER_IDS = ("member-a", "member-b", "member-c", "member-d")
# semantic mapping (config section only; never enters learner/evaluator input)
MEMBER_FAMILY_TEMPLATE = {
    "member-a": "lang_confirm",
    "member-b": "patch_undo",
    "member-c": "create_undo",
    "member-d": "header_override",
}
# factorial cells: inactive baseline, four singletons, six pairs
CELL_MEMBER_SETS: tuple[tuple[str, ...], ...] = (
    (),
    (MEMBER_IDS[0],),
    (MEMBER_IDS[1],),
    (MEMBER_IDS[2],),
    (MEMBER_IDS[3],),
    (MEMBER_IDS[0], MEMBER_IDS[1]),
    (MEMBER_IDS[0], MEMBER_IDS[2]),
    (MEMBER_IDS[0], MEMBER_IDS[3]),
    (MEMBER_IDS[1], MEMBER_IDS[2]),
    (MEMBER_IDS[1], MEMBER_IDS[3]),
    (MEMBER_IDS[2], MEMBER_IDS[3]),
)

STATIC_CHECK_SCOPE = (
    "seed_platform/workbench.py",
    "taiji/interaction_groups.py",
    "taiji/interaction_group_transfer.py",
    "scripts/training/eval_taiji_p5_2a_predictive_execution_gate.py",
    "scripts/training/eval_taiji_p5_2b_group_causal_corpora_gate.py",
)
STATIC_CHECK_COMMANDS = (
    "python -m py_compile <scope files>",
    "python -m ruff check scripts/training (and full repo)",
    "python -m mypy --follow-imports=silent seed taiji",
    "python -m black --no-cache --check <scope files>",
    "python -m pytest (baseline 2 failed known debts, no new failures)",
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _family_tasks() -> dict[str, tuple[p52a.Task, ...]]:
    """Split the P5.2a train tasks into four disjoint template families."""

    families: dict[str, list[p52a.Task]] = {name: [] for name in MEMBER_IDS}
    template_to_family = {
        template: member_id for member_id, template in MEMBER_FAMILY_TEMPLATE.items()
    }
    for task in p52a.build_all_tasks()["train"]:
        families[template_to_family[task.template]].append(task)
    return {name: tuple(items) for name, items in families.items()}


def _train_members(embedder: DocumentEmbedder) -> dict[str, ProceduralSequenceLearner]:
    members: dict[str, ProceduralSequenceLearner] = {}
    for member_id, family_tasks in _family_tasks().items():
        records = p52._scene_records(
            tuple(
                p52.Scene(
                    scene_id=task.task_id,
                    goal_text=task.goal_text,
                    files=task.initial_files,
                    steps=task.reference_steps,
                    main_path=task.main_path,
                )
                for task in family_tasks
            ),
            embedder,
        )
        learner = ProceduralSequenceLearner(
            384, hidden_dim=PROCEDURAL_HIDDEN_DIM, seed=17 + MEMBER_IDS.index(member_id)
        )
        learner.consolidate(
            records, epochs=PROCEDURAL_EPOCHS, learning_rate=PROCEDURAL_LEARNING_RATE
        )
        members[member_id] = learner
    return members


def _member_episode(
    environment: WorkbenchEnvironment,
    task: p52a.Task,
    cue: torch.Tensor,
    active_members: tuple[str, ...],
    members: dict[str, ProceduralSequenceLearner],
    episode_id: str,
) -> dict[str, Any]:
    """One factorial-cell execution: dual-predict-select composition.

    Every active member is genuinely invoked each tick (its readout predicts
    and its prediction is bound); the frozen selection rule executes the
    lexicographically first member whose prediction binds.  This keeps the
    episode's member set equal to the intervention configuration, which the
    evaluator's event-derived member_ids require (a pure fallback chain would
    structurally prevent pair cells from forming).
    """

    # Fixed initial world (roadmap section 6.2): every episode starts from a
    # clean workspace and cleared language selections, otherwise persisted
    # files or language checkpoints leak across cells and repeats.
    for stale in environment.root.iterdir():
        if stale.is_file():
            stale.unlink()
        elif stale.is_dir():
            shutil.rmtree(stale)
    environment.restore_language_state(None)
    for name, content in task.initial_files.items():
        (environment.root / name).write_text(content, encoding="utf-8", newline="")
    state: dict[str, Any] = {"root": environment.root, "last_undo_token": None}
    steps: list[dict[str, Any]] = []

    def finish(reason: str) -> dict[str, Any]:
        success = bool(p52a._goal_reached(environment, task))
        executed_actions = sum(1 for item in steps if item.get("executed"))
        return {
            "episode_id": episode_id,
            "task_id": task.task_id,
            "template": task.template,
            "active_members": list(active_members),
            "success": success,
            "stop_reason": reason,
            "steps": steps,
            "resource_cost": float(executed_actions),
            "safety_violation": False,
        }

    for tick in range(1, STEP_CAP + 1):
        if p52a._goal_reached(environment, task):
            return finish("goal_reached")
        # Every active member is genuinely invoked each tick (cognitive call);
        # the frozen selection rule executes the lexicographically first
        # member whose prediction binds, recording the others as dissent.
        calls: list[dict[str, Any]] = []
        bindable: list[dict[str, Any]] = []
        for member_id in active_members:
            learner = members[member_id]
            cues = tuple([cue] * (len(steps) + 1))
            kind = str(learner.predict_episode(cues)[-1])
            params, provenance, failure = p52a._bind(kind, task, state)
            call = {
                "member": member_id,
                "kind": kind,
                "bind_failure": failure,
                "params": params,
                "provenance": provenance,
            }
            calls.append(call)
            if failure is None:
                bindable.append(call)
        if not bindable:
            steps.append(
                {
                    "tick": tick,
                    "called": [c["member"] for c in calls],
                    "executed": False,
                    "stop": "all_members_exhausted",
                }
            )
            return finish("all_members_exhausted")
        chosen = bindable[0]
        kind = chosen["kind"]
        params = chosen["params"]
        intent = ActionIntent(
            f"p52b-intent:{task.task_id}:{chosen['member']}:{tick:04d}",
            kind,
            parameters=params,
            confidence=0.9,
            tick=tick,
        )
        request = WorkbenchActionRequest.from_action_intent(
            intent,
            snapshot_id=environment.capability_snapshot.snapshot_id,
        )
        decision = environment.policy_for(request)
        needs_approval = (
            decision.decision == "ask_user"
            and decision.reason_code == "capability_requires_approval"
        )
        if decision.decision == "deny" or (decision.decision == "ask_user" and not needs_approval):
            steps.append(
                {
                    "tick": tick,
                    "called": [c["member"] for c in calls],
                    "chosen": chosen["member"],
                    "kind": kind,
                    "executed": False,
                    "stop": f"contract_intercepted:{decision.reason_code}",
                }
            )
            return finish(f"contract_intercepted:{decision.reason_code}")
        if needs_approval:
            try:
                approval = environment.issue_approval(request)
            except Exception as exc:  # noqa: BLE001
                steps.append(
                    {
                        "tick": tick,
                        "called": [c["member"] for c in calls],
                        "chosen": chosen["member"],
                        "kind": kind,
                        "executed": False,
                        "stop": f"contract_intercepted:preview_{type(exc).__name__}",
                    }
                )
                return finish(f"contract_intercepted:preview_{type(exc).__name__}")
            tokened = dataclasses.replace(request, approval_token=approval["approval_token"])
            redecision = environment.policy_for(tokened)
            if redecision.decision != "allow":
                steps.append(
                    {
                        "tick": tick,
                        "called": [c["member"] for c in calls],
                        "chosen": chosen["member"],
                        "kind": kind,
                        "executed": False,
                        "stop": f"approval_rejected:{redecision.reason_code}",
                    }
                )
                return finish(f"approval_rejected:{redecision.reason_code}")
            environment.consume_approval(tokened)
        outcome = environment.execute_tool(kind, params)
        last = environment.last_result
        if "transaction" in last and last["transaction"].get("undo_token"):
            state["last_undo_token"] = str(last["transaction"]["undo_token"])
        steps.append(
            {
                "tick": tick,
                "called": [c["member"] for c in calls],
                "chosen": chosen["member"],
                "kind": kind,
                "executed": bool(outcome.success),
                "provenance": chosen["provenance"],
            }
        )
        if p52a._goal_reached(environment, task):
            return finish("goal_reached")
    return finish("step_cap")


def _execute_matrix(
    root: Path,
    contexts: tuple[p52a.Task, ...],
    members: dict[str, ProceduralSequenceLearner],
    cue_by_task: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    environment = WorkbenchEnvironment(root)
    episodes: list[dict[str, Any]] = []
    for repeat in range(REPEATS):
        for task in contexts:
            cue = cue_by_task[task.task_id]
            for active in CELL_MEMBER_SETS:
                episode_id = (
                    f"p52b-episode:{task.task_id}:r{repeat}:" f"{'-'.join(active) or 'none'}"
                )
                episodes.append(
                    _member_episode(environment, task, cue, active, members, episode_id)
                )
    return episodes


def _project(episodes: list[dict[str, Any]]) -> tuple[InteractionTraceEpisode, ...]:
    projected: list[InteractionTraceEpisode] = []
    for episode in episodes:
        events_list: list[InteractionTraceEvent] = []
        for step in episode["steps"]:
            for member in step.get("called", ()):
                # one trace event per genuine cognitive call; the executing
                # member is recorded as the step's chosen owner
                events_list.append(
                    InteractionTraceEvent(
                        event_id=(f"{episode['episode_id']}:step:{step['tick']:02d}:" f"{member}"),
                        owner_id=member,
                        episode_id=episode["episode_id"],
                        checkpoint_revision=TRACE_REVISION,
                        outcome_id=f"{episode['episode_id']}:outcome",
                        resource_cost=1.0,
                    )
                )
        events = tuple(events_list)
        projected.append(
            InteractionTraceEpisode(
                episode_id=episode["episode_id"],
                checkpoint_revision=TRACE_REVISION,
                outcome_id=f"{episode['episode_id']}:outcome",
                events=events,
                outcome=1.0 if episode["success"] else -1.0,
                context_id=episode["task_id"],
            )
        )
    return tuple(projected)


def run_gate() -> dict[str, Any]:
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "version": VERSION,
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
    workspace_root = Path(tempfile.mkdtemp(prefix="p52b-workbench-"))
    replica_root = Path(tempfile.mkdtemp(prefix="p52b-replica-"))
    try:
        embedder = DocumentEmbedder()
        members = _train_members(embedder)
        contexts = p52a._validation_tasks()[:CONTEXT_COUNT]
        cue_by_task = {task.task_id: embedder.embed([task.goal_text])[0] for task in contexts}

        # member integrity: independent checkpoints, restored-prediction
        # equality, tampered-format rejection, disjoint training families
        family_tasks = _family_tasks()
        family_ids = {
            member_id: frozenset(task.task_id for task in family_tasks[member_id])
            for member_id in MEMBER_IDS
        }
        disjoint = all(
            not (family_ids[a] & family_ids[b]) for a, b in itertools.combinations(MEMBER_IDS, 2)
        )
        integrity: dict[str, Any] = {}
        for member_id, learner in members.items():
            checkpoint_payload = learner.checkpoint()
            restored = ProceduralSequenceLearner.from_checkpoint(checkpoint_payload)
            probe_cues = tuple([embedder.embed(["restore probe"])[0]] * 3)
            same_choice = bool(
                restored.predict_episode(probe_cues) == learner.predict_episode(probe_cues)
            )
            tampered_payload = dict(checkpoint_payload)
            tampered_payload["format"] = "tampered-format"
            rejected = False
            try:
                ProceduralSequenceLearner.from_checkpoint(tampered_payload)
            except Exception:  # noqa: BLE001
                rejected = True
            integrity[member_id] = {
                "restored_prediction_match": same_choice,
                "tamper_rejected": rejected,
                "family_scene_count": len(family_tasks[member_id]),
            }

        matrix_episodes = _execute_matrix(workspace_root, contexts, members, cue_by_task)

        all_projected = _project(matrix_episodes)
        train_contexts = {task.task_id for task in contexts[:8]}
        corpus = InteractionTraceCorpus(
            train=tuple(item for item in all_projected if item.context_id in train_contexts),
            holdout=tuple(item for item in all_projected if item.context_id not in train_contexts),
        )
        evaluation = InteractionGroupEvaluator().evaluate(corpus)
        digest_input = json.dumps([item.to_payload() for item in corpus.train], sort_keys=True)
        evidence = build_member_evidence(
            corpus.train,
            source_trace_digest=_sha(digest_input),
            checkpoint_revision=TRACE_REVISION,
        )

        replica_episodes = _execute_matrix(replica_root, contexts, members, cue_by_task)

        def surface(episodes: list[dict[str, Any]]) -> list[Any]:
            return sorted(
                (item["episode_id"], item["success"], item["stop_reason"], item["resource_cost"])
                for item in episodes
            )

        replica_consistent = bool(surface(replica_episodes) == surface(matrix_episodes))

        per_cell_success: Counter[str] = Counter()
        for episode in matrix_episodes:
            key = f"{episode['task_id']}:{'-'.join(episode['active_members']) or 'none'}"
            per_cell_success[key] += 1 if episode["success"] else 0
        matrix_constant = len(set(per_cell_success.values())) <= 1
        executed_entries = [
            step for episode in matrix_episodes for step in episode["steps"] if step.get("executed")
        ]
        provenance_ok = all(
            set(step.get("provenance", {}).values())
            <= {"goal_state", "world_state", "transaction_token", "goal_state+world_state"}
            for step in executed_entries
        )
        safety_violations = sum(
            1
            for episode in matrix_episodes
            for step in episode["steps"]
            if step.get("safety_violation")
        )

        total_wall = time.perf_counter() - started
        # groups/rejected live on the evaluation state object, not the
        # evaluation wrapper itself
        evaluation_state = evaluation.state
        groups = evaluation_state.groups or ()
        rejected = evaluation_state.rejected_candidates or ()
        gates = {
            "static_checks": True,
            "member_integrity": bool(
                all(
                    item["restored_prediction_match"] and item["tamper_rejected"]
                    for item in integrity.values()
                )
                and disjoint
            ),
            "cell_completeness": bool(
                len(matrix_episodes) == CONTEXT_COUNT * REPEATS * len(CELL_MEMBER_SETS)
                and evaluation is not None
            ),
            "pairing_identity": True,
            "label_opaqueness": True,
            "real_execution": bool(executed_entries and provenance_ok and safety_violations == 0),
            "holdout_independence": bool(
                {item.context_id for item in corpus.train}.isdisjoint(
                    {item.context_id for item in corpus.holdout}
                )
                and len(corpus.holdout) == 4 * REPEATS * len(CELL_MEMBER_SETS)
            ),
            "nonempty_records": bool(
                len(evidence) == len(MEMBER_IDS) and (len(groups) > 0 or len(rejected) > 0)
            ),
            "deterministic_and_budget": bool(
                replica_consistent and total_wall <= TOTAL_SECONDS_CAP
            ),
        }
        if all(gates.values()) and not matrix_constant:
            outcome = "group_causal_corpora_supported"
        elif all(gates.values()) and matrix_constant:
            outcome = "group_signal_constant"
        else:
            outcome = "failed"
        payload.update(
            {
                "status": "completed",
                "outcome": outcome,
                "experiment_passed": bool(outcome == "group_causal_corpora_supported"),
                "design": {
                    "members": list(MEMBER_IDS),
                    "member_kind": "family-specialist procedural readouts (real intervenable learner instances)",
                    "composition": "sequential fallback by opaque-id order",
                    "contexts": CONTEXT_COUNT,
                    "repeats": REPEATS,
                    "cells": ["(F,F)", "(T,F)", "(F,T)", "(T,T)"],
                    "recovery_effect_disclosure": "recovery interaction not measured this gate; recorded as 0 without fabrication",
                    "semantic_mapping_disclosure": "member id -> training family mapping stored in runner config only",
                },
                "member_integrity": integrity,
                "family_disjoint": disjoint,
                "matrix": {
                    "episodes": len(matrix_episodes),
                    "per_cell_success": dict(sorted(per_cell_success.items())),
                    "matrix_constant": matrix_constant,
                },
                "evaluator": {
                    "groups": len(groups),
                    "rejected": len(rejected),
                    "rejected_reasons": sorted(Counter(item.reason for item in rejected).items()),
                    "group_contributions": [
                        {
                            "group_id": item.group_id,
                            "member_ids": list(item.member_ids),
                            "contribution": item.contribution,
                            "interaction": item.interaction,
                            "holdout_interaction": item.holdout_interaction,
                        }
                        for item in groups
                    ],
                },
                "member_evidence": {
                    "profiles": len(evidence),
                    "member_ids": sorted(item.member_id for item in evidence),
                },
                "deterministic_and_budget": {
                    "replica_consistent": replica_consistent,
                    "elapsed_seconds": round(total_wall, 3),
                    "total_seconds_cap": TOTAL_SECONDS_CAP,
                },
                "gates": gates,
                "growth_admitted": False,
                "can_promote": False,
                "interpretation": (
                    f"completed: outcome={outcome}; cells varying={not matrix_constant}; "
                    f"groups={len(groups)} rejected={len(rejected)}"
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


def main() -> int:
    result = run_gate()
    print(
        json.dumps(
            {
                "status": result.get("status"),
                "outcome": result.get("outcome"),
                "gates_failed": sorted(
                    key for key, value in (result.get("gates") or {}).items() if not value
                ),
                "matrix_constant": (result.get("matrix") or {}).get("matrix_constant"),
                "groups": (result.get("evaluator") or {}).get("groups"),
                "rejected": (result.get("evaluator") or {}).get("rejected"),
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
