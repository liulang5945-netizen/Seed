"""Fixed-history state reencoding and atomic slow-parameter migration for U2."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

import torch

from .u2_events import eligibility_carry
from .u2_history import CausalHistory, assert_values_equal, detached_tree, value_digest
from .u2_recovery import U2Runtime, _optimizer_layout, _restore_optimizer, validate_subject_state
from .u2_subject import U2Subject

FACT_FIELDS = (
    "subject_id",
    "goal_raw",
    "goal_status",
    "goal_version",
    "records",
    "released_feedback_records",
    "evidence",
    "journal",
    "seen_packets",
    "settled",
    "world_time",
    "process_time",
    "processing_epoch",
    "rng",
    "wake_condition",
    "pending",
    "utterance_active",
    "previous_byte",
    "utf8_lead",
    "utf8_remaining",
    "data_bytes",
)


def _check_budget(deadline: float) -> None:
    if time.process_time() > deadline:
        raise RuntimeError("migration_unaffordable: CPU budget exhausted")


def _deadline(seconds: float) -> float:
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Migration requires a finite positive CPU budget")
    return time.process_time() + seconds


def _fact_view(state: dict[str, Any]) -> dict[str, Any]:
    view = {name: state[name] for name in FACT_FIELDS}
    for name in ("answer_request", "utterance_request"):
        if name in state:
            view[name] = state[name]
    # The original parameter version remains in the factual evidence. Its shadow
    # counterpart was calculated by the current model during reencoding.
    view["evidence"] = [
        {key: value for key, value in row.items() if key != "subject_version"}
        for row in state["evidence"]
    ]
    return view


class _HistoryReplay:
    def __init__(self, model: U2Subject, subject_id: str, deadline: float):
        self.model = model
        self.state = model.initial(subject_id)
        self.decision: dict[str, Any] | None = None
        self.acknowledged: set[str] = set()
        self.deadline = deadline
        self.counts = {
            "entries": 0,
            "observations": 0,
            "duplicates": 0,
            "bytes": 0,
            "internal": 0,
            "births": 0,
            "updates": 0,
        }

    def _issue(self, arguments: Any, identity: str, behavior_version: int) -> None:
        if self.decision is None:
            raise ValueError("Raw issue lacks its fixed behavior decision")
        self.state, _ = self.model.issue(self.state, self.decision, **arguments)
        self.state["pending"] = {
            **self.state["pending"],
            "id": identity,
            "behavior_version": behavior_version,
        }
        self.state["journal"] = [
            *self.state["journal"][:-1],
            {**self.state["journal"][-1], "id": identity, "status": "intent"},
        ]
        self.decision = None

    def _acknowledge(self, identity: str) -> None:
        if self.state["pending"] is None or self.state["pending"]["id"] != identity:
            raise ValueError("Raw acknowledgment has no matching intent")
        self.acknowledged.add(identity)
        self.state["journal"] = [
            (
                {**row, "status": "executed"}
                if row.get("id") == identity and row.get("status") == "intent"
                else row
            )
            for row in self.state["journal"]
        ]

    def _receive(self, packet: Any) -> None:
        receipt = self.state["pending"]
        duplicate = packet.packet_id in self.state["seen_packets"]
        if not duplicate and (receipt is None or receipt["id"] not in self.acknowledged):
            raise ValueError("Raw observation lacks an acknowledged actual action")
        self.state, result = self.model.assimilate(self.state, packet)
        if duplicate:
            self.counts["duplicates"] += 1
        elif self.state["pending"] is None:
            self.state["eligibility"] = eligibility_carry(receipt, self.state["goal_version"])
            self.counts["observations"] += 1

    def _birth(self, arguments: Any, expected: Any) -> None:
        past = expected["structure"]
        current = self.model.structure()
        if past["units"] > current["units"] or past["generation"] > current["generation"]:
            raise ValueError("Raw birth is absent from the accepted structure")
        for key in ("unit_ids", "unit_branches"):
            if len(past[key]) != len(current[key]) or any(
                row != current_row[: len(row)]
                for row, current_row in zip(past[key], current[key], strict=True)
            ):
                raise ValueError("Raw birth lineage differs from the accepted structure")
        self.state["history"] = [
            {**row, "neutral_birth_rebase": arguments["birth_id"]} for row in self.state["history"]
        ]
        self.counts["births"] += 1

    def _update(self, arguments: Any) -> None:
        if arguments["revision"] != arguments["parent_revision"] + 1:
            raise ValueError("Invalid historical parameter transition")
        if arguments["revision"] > self.model.parameter_revision:
            raise ValueError("Raw update is newer than the reencoding candidate")
        self.state["eligibility"] = {
            name: torch.zeros_like(parameter)
            for name, parameter in self.model.policy_named_parameters()
        }
        self.state["resources"] = {
            **self.state["resources"],
            "updates": self.state["resources"]["updates"] + 1,
        }
        self.state["journal"] = [*self.state["journal"], arguments["journal_entry"]]
        self.counts["updates"] += 1

    def apply(self, payload: Any) -> None:
        _check_budget(self.deadline)
        operation = payload["operation"]
        kind, arguments = operation["kind"], operation["arguments"]
        if kind == "goal":
            self.state = self.model.set_goal(self.state, **arguments)
        elif kind == "records":
            self.state = self.model.set_records(self.state, **arguments)
        elif kind == "request":
            self.state = self.model.set_request(self.state, **arguments)
        elif kind == "cancel_request":
            self.state = self.model.cancel_request(self.state, **arguments)
        elif kind == "choose":
            self.decision = detached_tree(operation["expected"])
        elif kind == "issue":
            self._issue(arguments, operation["expected"], payload["parameter_revision"])
        elif kind == "observe_intent":
            from .u2_events import observation_intent

            self.state = observation_intent(self.model, self.state, **arguments)
        elif kind == "ack":
            self._acknowledge(arguments["identity"])
        elif kind == "process_time":
            value = arguments["value"]
            if not math.isfinite(value) or value < self.state["process_time"]:
                raise ValueError("Raw processing clock regressed")
            self.state["process_time"] = value
        elif kind == "receive":
            self._receive(arguments["packet"])
        elif kind == "begin_utterance":
            self.state = self.model.begin_utterance(self.state)
        elif kind == "emit":
            for symbol in arguments["symbols"]:
                goal = self.model.language_goal(self.state)
                _, _, output = self.model.output(self.state, goal, self.state["prediction_action"])
                self.state = self.model.accept_byte(self.state, symbol, output)
                self.counts["bytes"] += int(symbol != 256)
        elif kind == "internal":
            if self.decision is None:
                raise ValueError("Raw internal activity lacks a fixed decision")
            self.state = self.model.internal(self.state, self.decision)
            self.decision = None
            self.counts["internal"] += 1
        elif kind == "birth":
            self._birth(arguments, operation["expected"])
        elif kind == "update":
            self._update(arguments)
        elif kind != "window":
            raise ValueError("Unsupported raw reencoding operation")
        self.counts["entries"] += 1


def rebuild_state(
    model: U2Subject,
    history: CausalHistory,
    factual_state: dict[str, Any],
    *,
    cpu_budget_seconds: float = 30,
    reset_eligibility: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reinterpret a complete prefix with current parameters and fixed behavior."""
    started, deadline = time.process_time(), _deadline(cpu_budget_seconds)
    history.require_complete()
    verified = CausalHistory.restore(history.export())
    if verified.root["subject_id"] != factual_state["subject_id"]:
        raise ValueError("Raw history belongs to another subject")
    if model.structure() != factual_state["structure"]:
        raise ValueError("Reencoding requires the accepted structure")
    from .u2_group_structure import ancestry_compatible

    if not ancestry_compatible(verified.root["config"], model.c):
        raise ValueError("Reencoding configuration mismatch")
    with torch.no_grad():
        replay = _HistoryReplay(model, factual_state["subject_id"], deadline)
        for node in verified.iter_entries():
            replay.apply(node["payload"])
    state = replay.state
    if state["pending"] is not None or state["utterance_active"] or replay.decision is not None:
        raise ValueError("Reencoding requires a committed history boundary")
    state["rng"] = detached_tree(factual_state["rng"])
    assert_values_equal(_fact_view(factual_state), _fact_view(state), "migration.facts")
    assert_values_equal(factual_state["resources"], state["resources"], "migration.resources")
    assert_values_equal(
        factual_state["nu_measurement_schema"],
        state["nu_measurement_schema"],
        "migration.measurements",
    )
    for name in FACT_FIELDS:
        state[name] = detached_tree(factual_state[name])
    if reset_eligibility:
        state["eligibility"] = {
            name: torch.zeros_like(parameter) for name, parameter in model.policy_named_parameters()
        }
    state["nu"][4:8] = factual_state["nu"][4:8]
    state["nu"][12:] = factual_state["nu"][12:]
    validate_subject_state(model, state)
    _check_budget(deadline)
    return state, {
        **replay.counts,
        "cpu_seconds": time.process_time() - started,
        "history_reference": history.reference(),
        "eligibility_reset": reset_eligibility,
    }


def _state_token(model: U2Subject, state: Any, optimizer: Any) -> Any:
    return {
        "parameters": value_digest(model.state_dict()),
        "state": value_digest(state),
        "revision": model.parameter_revision,
        "structure": model.structure(),
        "config": asdict(model.c),
        "training": model.training,
        "optimizer": value_digest(None if optimizer is None else optimizer.state_dict()),
        "optimizer_layout": _optimizer_layout(model, optimizer),
    }


def _parent_token(runtime: U2Runtime) -> Any:
    return {
        **_state_token(runtime.model, runtime.state, runtime.optimizer),
        "history": runtime.history.reference(),
        "execution": value_digest(runtime.execution),
        "data_cursor": value_digest(runtime.data_cursor),
    }


def migrate_parameters(
    runtime: U2Runtime,
    candidate: U2Subject,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    acceptance: Callable[[U2Subject, dict[str, Any], torch.optim.Optimizer | None], bool],
    cpu_budget_seconds: float = 30,
    fault: str | None = None,
) -> dict[str, Any]:
    """Commit a forked candidate and its reencoded state as one local transaction."""
    started, deadline = time.process_time(), _deadline(cpu_budget_seconds)
    if fault not in {None, "after_rebuild", "before_commit"}:
        raise ValueError("Unsupported migration fault point")
    if not callable(acceptance):
        raise ValueError("Migration requires an acceptance callback")
    if (
        runtime.state["pending"] is not None
        or runtime.state["utterance_active"]
        or runtime.last_decision is not None
    ):
        raise ValueError("Parameter migration requires a committed subject boundary")
    runtime._check_window_for_save()
    validate_subject_state(runtime.model, runtime.state)
    runtime.history.require_complete()
    if (
        candidate is runtime.model
        or candidate.parameter_revision != runtime.model.parameter_revision + 1
    ):
        raise ValueError("Candidate must be a fork at parent revision plus one")
    if candidate.c != runtime.model.c or candidate.structure() != runtime.model.structure():
        raise ValueError("Parameter migration requires the accepted structure and configuration")
    expected = runtime.model.state_dict()
    if set(candidate.state_dict()) != set(expected):
        raise ValueError("Candidate parameter registry mismatch")
    for name, value in candidate.state_dict().items():
        original = expected[name]
        if (
            value.shape != original.shape
            or value.dtype != original.dtype
            or value.device != original.device
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"Invalid candidate parameter {name}")
    if runtime.optimizer is not None and optimizer is None:
        raise ValueError("Candidate optimizer is required to preserve existing moments")
    token = _parent_token(runtime)
    values = None if optimizer is None else optimizer.state_dict()
    bundle = candidate.export_bundle(candidate.initial("migration-shadow"), values)
    model, _ = U2Subject.from_bundle(bundle)
    model.train(candidate.training)
    restored_optimizer = _restore_optimizer(model, _optimizer_layout(candidate, optimizer), values)
    _check_budget(deadline)
    state, metrics = rebuild_state(
        model,
        runtime.history,
        runtime.state,
        cpu_budget_seconds=max(1e-12, deadline - time.process_time()),
    )
    if fault == "after_rebuild":
        raise RuntimeError("injected after_rebuild")
    acceptance_token = _state_token(model, state, restored_optimizer)
    accepted = bool(acceptance(model, state, restored_optimizer))
    _check_budget(deadline)
    if not accepted:
        return {
            "status": "rejected",
            "rebuild": metrics,
            "cpu_seconds": time.process_time() - started,
        }
    assert_values_equal(
        acceptance_token,
        _state_token(model, state, restored_optimizer),
        "acceptance_callback_mutated_candidate",
    )
    validate_subject_state(model, state)
    _check_budget(deadline)
    assert_values_equal(token, _parent_token(runtime), "parent_changed_during_migration")
    journal_entry = {
        "operation": "parameter_update",
        "parent_revision": runtime.model.parameter_revision,
        "revision": model.parameter_revision,
    }
    state["journal"] = [*state["journal"], journal_entry]
    state["resources"] = {**state["resources"], "updates": state["resources"]["updates"] + 1}
    digest = value_digest(model.state_dict())
    arguments = {
        "parent_revision": runtime.model.parameter_revision,
        "revision": model.parameter_revision,
        "parent_parameter_digest": runtime._parameter_digest,
        "parameter_digest": digest,
        "history_prefix": runtime.history.reference(),
        "journal_entry": journal_entry,
    }
    history = runtime.history.append(
        {
            "operation": {
                "kind": "update",
                "arguments": arguments,
                "expected": {"structure": model.structure()},
            },
            "parameter_revision": runtime.model.parameter_revision,
            "parameter_digest": runtime._parameter_digest,
            "segment": runtime._history_segment + 1,
        }
    )
    child = U2Runtime(
        model,
        state=state,
        optimizer=restored_optimizer,
        executor=runtime.executor,
        max_operations=runtime.max_operations,
    )
    child.history = history
    child._history_anchor = history.reference()
    child._history_segment = runtime._history_segment + 1
    child._anchor_runtime = {
        "data_cursor": detached_tree(runtime.data_cursor),
        "execution": detached_tree(runtime.execution),
    }
    child.data_cursor = detached_tree(runtime.data_cursor)
    child.execution = detached_tree(runtime.execution)
    child.host_clock = detached_tree(runtime.host_clock)
    child.audit = [
        *runtime.audit,
        {
            "operation": "parameter_migration",
            "status": "committed",
            "revision": model.parameter_revision,
            "rebuild": metrics,
        },
    ]
    child._check_window_for_save()
    _check_budget(deadline)
    runtime._check_window_for_save()
    assert_values_equal(token, _parent_token(runtime), "parent_changed_during_migration")
    if fault == "before_commit":
        raise RuntimeError("injected before_commit")
    runtime.__dict__.update(child.__dict__)
    return {
        "status": "committed",
        "revision": model.parameter_revision,
        "parameter_digest": digest,
        "rebuild": metrics,
        "cpu_seconds": time.process_time() - started,
    }
