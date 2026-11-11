"""Finite-window U2 recovery and a controlled idempotent reference executor.

The envelope stores values plus raw operations. Restoring replays the declared
learning window under the saved parameters, rebuilding its tensor graph without
calling an external executor. Real execution remains behind a durable intent.
"""

from __future__ import annotations

import hashlib
import math
import os
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from functools import wraps
from pathlib import Path
from typing import Any, cast

import torch

from .u2_history import (
    CausalHistory,
    assert_values_equal,
    atomic_load,
    atomic_save,
    detached_tree,
    value_digest,
)
from .u2_operators import _validate_cursor
from .u2_subject import U2Packet, U2Record, U2Subject

ENVELOPE_SCHEMA = "taiji-u2-runtime-envelope-v3"
EXECUTOR_SCHEMA = "taiji-u2-reference-executor-v1"


def run_conditions(model: U2Subject) -> dict[str, Any]:
    return {
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "device": str(model.native_w.device),
        "dtype": str(model.native_w.dtype),
        "python": sys.version,
        "torch": torch.__version__,
        "environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
    }


def _source_pins() -> dict[str, str]:
    directory = Path(__file__).parent
    names = (
        "u2_recovery.py",
        "u2_history.py",
        "u2_events.py",
        "u2_learning_transaction.py",
        "u2_record_metadata.py",
        "u2_group_structure.py",
        "u2_subject.py",
        "u2_operators.py",
        "utf8_state.py",
    )
    return {
        name: hashlib.sha256(
            (directory / name).read_text(encoding="utf-8-sig").encode()
        ).hexdigest()
        for name in names
    }


def validate_subject_state(model: U2Subject, state: Mapping[str, Any]) -> None:
    """Validate legal anchors as well as reconstructed snapshots."""
    if set(state) != set(model.state_registry()):
        raise ValueError("Subject state registry mismatch")
    model.export_bundle(state)
    config, reference = model.c, model.native_w
    shapes = {
        "h": (config.branches, config.objects, config.d),
        "p": (config.branches, config.objects, 2),
        "v": (config.branches, config.objects, 2),
        "s": (config.branches, config.objects, 4),
        "l": (config.branches, config.objects, 1),
        "H": (config.branches, config.groups, config.d),
        "F": (config.branches, config.key, config.key),
        "nu": (16,),
    }
    tolerance = 1e-5 if reference.dtype == torch.float32 else 1e-10
    for name, shape in shapes.items():
        value = state[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != shape
            or value.dtype != reference.dtype
            or value.device != reference.device
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"Invalid subject tensor {name}")
    for name in ("h", "H"):
        if bool((state[name].abs() > 1 + tolerance).any()):
            raise ValueError(f"Subject {name} lies outside its bounded domain")
    if bool((state["F"].flatten(1).norm(dim=-1) > 4 + tolerance).any()):
        raise ValueError("Subject fast memory exceeds its ball")
    for name in ("p", "v"):
        if bool((state[name].norm(dim=-1) > 4 + tolerance).any()):
            raise ValueError(f"Subject {name} exceeds its geometry domain")
    if bool(((state["s"] < 1e-4 - tolerance) | (state["s"] > 4 + tolerance)).any()):
        raise ValueError("Invalid subject uncertainty scale")
    if bool((state["l"].abs() > 8 + tolerance).any()):
        raise ValueError("Invalid subject existence logit")
    weights = state["logw"]
    if (
        not isinstance(weights, torch.Tensor)
        or weights.shape != (config.branches,)
        or weights.dtype != reference.dtype
        or weights.device != reference.device
        or bool(torch.isnan(weights).any())
        or bool(torch.isposinf(weights).any())
        or not torch.allclose(
            torch.logsumexp(weights, 0), reference.new_zeros(()), atol=tolerance, rtol=0
        )
    ):
        raise ValueError("Invalid subject branch weights")
    for name in ("world_time", "process_time"):
        if not math.isfinite(state[name]) or state[name] < 0:
            raise ValueError("Subject clocks must be finite and nonnegative")
    records = state["records"]
    if not isinstance(records, (tuple, list)) or any(
        not isinstance(record, U2Record) for record in records
    ):
        raise ValueError("Invalid subject record registry")
    positions = max((len(record.raw) for record in records), default=0)
    valid = torch.zeros((len(records), positions), dtype=torch.bool, device=reference.device)
    for index, record in enumerate(records):
        if (
            not isinstance(record, U2Record)
            or not record.raw
            or len(record.raw) > config.max_record_bytes
            or not math.isfinite(record.observed_at)
            or record.observed_at > state["world_time"]
            or record.source != "sensor"
            or record.scope != "world"
        ):
            raise ValueError("Invalid subject record metadata")
        valid[index, : len(record.raw)] = True
    _validate_cursor(state["cursor"], valid)
    if (
        type(state["utf8_remaining"]) is not int
        or not 0 <= state["utf8_remaining"] <= 3
        or type(state["utf8_lead"]) is not int
        or not 0 <= state["utf8_lead"] <= 255
        or type(state["previous_byte"]) is not int
        or not 0 <= state["previous_byte"] <= 256
        or type(state["data_bytes"]) is not int
        or not 0 <= state["data_bytes"] <= config.max_output_bytes
    ):
        raise ValueError("Invalid subject utterance state")
    expected = dict(model.policy_named_parameters())
    if set(state["eligibility"]) != set(expected):
        raise ValueError("Eligibility parameter registry mismatch")
    for name, parameter in expected.items():
        value = state["eligibility"][name]
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != parameter.shape
            or value.dtype != parameter.dtype
            or value.device != parameter.device
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError("Invalid subject eligibility tensor")
    if set(state["resources"]) != {"events", "operations", "updates"} or any(
        type(value) is not int or value < 0 for value in state["resources"].values()
    ):
        raise ValueError("Invalid subject resource counters")
    try:
        torch.Generator(device="cpu").set_state(state["rng"])
    except (RuntimeError, TypeError) as error:
        raise ValueError("Invalid subject RNG state") from error


def _optimizer_layout(model: U2Subject, optimizer: torch.optim.Optimizer | None) -> Any:
    if optimizer is None:
        return None
    if type(optimizer) not in (torch.optim.Adam, torch.optim.SGD):
        raise ValueError("Recovery supports the declared Adam and SGD optimizers")
    if optimizer.defaults.get("differentiable", False):
        raise ValueError("Cross-window differentiable optimizer graphs are outside this contract")
    named = {id(parameter): name for name, parameter in model.named_parameters()}
    groups = []
    used = []
    for group in optimizer.param_groups:
        names = []
        for parameter in group["params"]:
            if id(parameter) not in named:
                raise ValueError("Optimizer contains an unregistered model parameter")
            names.append(named[id(parameter)])
        used.extend(names)
        groups.append(
            {
                "names": names,
                "options": detached_tree(
                    {key: value for key, value in group.items() if key != "params"}
                ),
            }
        )
    if len(used) != len(set(used)):
        raise ValueError("Optimizer parameter appears in multiple groups")
    return {
        "type": type(optimizer).__name__,
        "defaults": detached_tree(optimizer.defaults),
        "groups": groups,
    }


def _restore_optimizer(model: U2Subject, layout: Any, values: Any) -> torch.optim.Optimizer | None:
    if layout is None:
        if values is not None:
            raise ValueError("Optimizer values have no declared parameter layout")
        return None
    classes = {"Adam": torch.optim.Adam, "SGD": torch.optim.SGD}
    if layout.get("type") not in classes or values is None:
        raise ValueError("Unsupported optimizer recovery layout")
    named = dict(model.named_parameters())
    groups = []
    used = []
    for group in layout["groups"]:
        if any(name not in named for name in group["names"]):
            raise ValueError("Optimizer recovery refers to an unknown parameter")
        used.extend(group["names"])
        groups.append(
            {**detached_tree(group["options"]), "params": [named[name] for name in group["names"]]}
        )
    if len(used) != len(set(used)):
        raise ValueError("Duplicate recovered optimizer parameter")
    optimizer = classes[layout["type"]](groups, **detached_tree(layout["defaults"]))
    optimizer.load_state_dict(detached_tree(values))
    for parameter, entries in optimizer.state.items():
        for key, value in entries.items():
            if isinstance(value, torch.Tensor):
                if not bool(torch.isfinite(value).all()):
                    raise ValueError("Optimizer recovery contains a nonfinite moment")
                if key != "step" and (
                    value.shape != parameter.shape
                    or value.dtype != parameter.dtype
                    or value.device != parameter.device
                ):
                    raise ValueError("Optimizer moment layout differs from its parameter")
    return cast(torch.optim.Optimizer, optimizer)


class ReferenceExecutor:
    """A controlled simulator whose only effect is one atomic ledger transition."""

    def __init__(
        self, path: str | Path | None = None, initial_environment: dict[str, Any] | None = None
    ):
        self.path = None if path is None else Path(path)
        self._authoritative_absence = True
        self._ledger = {
            "schema": EXECUTOR_SCHEMA,
            "actions": {},
            "effect_count": 0,
            "environment": detached_tree(initial_environment or {"position": [0.0, 0.0]}),
        }
        if self.path is not None:
            if self.path.exists():
                self._ledger = self._read()
            else:
                atomic_save(self._ledger, self.path)

    def _read(self) -> dict[str, Any]:
        ledger = atomic_load(self.path) if self.path is not None else self._ledger
        if ledger.get("schema") != EXECUTOR_SCHEMA or not isinstance(ledger.get("actions"), dict):
            raise ValueError("Invalid reference executor ledger")
        return ledger

    @property
    def effect_count(self) -> int:
        return int(self._read()["effect_count"])

    @property
    def environment(self) -> dict[str, Any]:
        return cast(dict[str, Any], detached_tree(self._read()["environment"]))

    def snapshot(self) -> dict[str, Any]:
        return {"persistent": self.path is not None, "ledger": detached_tree(self._read())}

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, Any]) -> ReferenceExecutor:
        result = cls()
        result._ledger = detached_tree(snapshot["ledger"])
        result._authoritative_absence = not bool(snapshot["persistent"])
        result._read()
        return result

    def query(self, action_id: str) -> dict[str, Any]:
        ledger = self._read()
        if action_id in ledger["actions"]:
            return cast(dict[str, Any], detached_tree(ledger["actions"][action_id]))
        return {
            "status": "not_seen" if self._authoritative_absence else "unknown",
            "action_id": action_id,
        }

    def execute(self, action_id: str, action: torch.Tensor) -> dict[str, Any]:
        if not action_id or action.shape != (15,) or not bool(torch.isfinite(action).all()):
            raise ValueError("Invalid reference execution command")
        if not self._authoritative_absence:
            raise RuntimeError("Executor status must be queried from the live durable ledger")
        lock = None if self.path is None else self.path.with_name(self.path.name + ".lock")
        if lock is not None:
            with lock.open("x", encoding="utf-8") as stream:
                stream.write(str(os.getpid()))
        try:
            ledger = detached_tree(self._read())
            command = detached_tree(action.cpu())
            if action_id in ledger["actions"]:
                existing = ledger["actions"][action_id]
                assert_values_equal(existing["action"], command, "executor.command")
                return cast(dict[str, Any], detached_tree(existing))
            position = ledger["environment"].get("position", [0.0, 0.0])
            duration = float(command[14])
            ledger["environment"]["position"] = [
                float(position[index]) + float(command[11 + index]) * duration for index in range(2)
            ]
            ledger["effect_count"] += 1
            ack = {
                "status": "applied",
                "action_id": action_id,
                "action": command,
                "effect_index": ledger["effect_count"],
            }
            ledger["actions"][action_id] = ack
            if self.path is not None:
                atomic_save(ledger, self.path)
            self._ledger = ledger
            return cast(dict[str, Any], detached_tree(ack))
        finally:
            if lock is not None:
                lock.unlink(missing_ok=True)


def _history_atomic(method: Callable[..., Any]) -> Callable[..., Any]:
    """Roll back local operations if recording their raw history fails."""

    @wraps(method)
    def apply(runtime: U2Runtime, *args: Any, **kwargs: Any) -> Any:
        before = runtime.__dict__.copy()
        runtime._operations = detached_tree(runtime._operations)
        runtime.data_cursor = dict(runtime.data_cursor)
        runtime.execution = detached_tree(runtime.execution)
        runtime.audit = list(runtime.audit)
        try:
            return method(runtime, *args, **kwargs)
        except BaseException:
            runtime.__dict__.clear()
            runtime.__dict__.update(before)
            raise

    return apply


class U2Runtime:
    """Single-writer subject operations with an explicit finite learning window."""

    def __init__(
        self,
        model: U2Subject,
        state: dict[str, Any] | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        executor: ReferenceExecutor | None = None,
        *,
        subject_id: str = "u2-runtime",
        max_operations: int = 32,
    ):
        if type(max_operations) is not int or not 1 <= max_operations <= 32:
            raise ValueError("Finite learning window requires 1..32 operations")
        self.model = model
        self.optimizer = optimizer
        self.executor = executor if executor is not None else ReferenceExecutor()
        self.max_operations = max_operations
        self.state: dict[str, Any] = cast(
            dict[str, Any], model.initial(subject_id) if state is None else detached_tree(state)
        )
        validate_subject_state(model, self.state)
        if state is not None and (
            self.state["pending"] is not None or self.state["utterance_active"]
        ):
            raise ValueError("A values anchor requires a committed subject boundary")
        self.last_decision: dict[str, Any] | None = None
        self.data_cursor: dict[str, Any] = {"received_packets": 0}
        self.execution: dict[str, dict[str, Any]] = {}
        self.audit: list[dict[str, Any]] = []
        self._operations: list[dict[str, Any]] = []
        self._revision = model.parameter_revision
        self._parameter_digest = value_digest(model.state_dict())
        self._sources = _source_pins()
        self._anchor = (
            {"kind": "genesis", "subject_id": subject_id}
            if state is None
            else {"kind": "values", "state": detached_tree(self.state)}
        )
        self._anchor_runtime = {
            "data_cursor": detached_tree(self.data_cursor),
            "execution": detached_tree(self.execution),
        }
        self.history = CausalHistory(
            {
                "kind": "genesis" if state is None else "values",
                "subject_id": self.state["subject_id"],
                "config": asdict(model.c),
                "parameter_revision": self._revision,
                "parameter_digest": self._parameter_digest,
                "source_pins": self._sources,
                "anchor_state": None if state is None else self.state,
            }
        )
        self._history_segment = 0
        self._history_anchor = self.history.reference()
        self.host_clock: dict[str, Any] = {
            "epoch": uuid.uuid4().hex,
            "origin": time.monotonic() - float(self.state["process_time"]),
        }
        self._shadow = False

    @property
    def operation_count(self) -> int:
        return len(self._operations)

    def _check_window(self, *, coalesce_emit: bool = False) -> None:
        if self.model.parameter_revision != self._revision:
            raise ValueError("Parameters changed inside a frozen learning window")
        coalesced = coalesce_emit and self._operations and self._operations[-1]["kind"] == "emit"
        if len(self._operations) >= self.max_operations and not coalesced:
            raise ValueError("learning_window_budget_stop")

    def _history_payload(self, kind: str, arguments: dict[str, Any], expected: Any = None) -> Any:
        return {
            "operation": {"kind": kind, "arguments": arguments, "expected": expected},
            "parameter_revision": self._revision,
            "parameter_digest": self._parameter_digest,
            "segment": self._history_segment,
        }

    def _append_history(self, kind: str, arguments: dict[str, Any], expected: Any = None) -> None:
        if not self._shadow:
            self.history = self.history.append(self._history_payload(kind, arguments, expected))

    def _record(
        self, kind: str, arguments: dict[str, Any], expected: Any = None, *, archive: bool = True
    ) -> None:
        operation = detached_tree({"kind": kind, "arguments": arguments, "expected": expected})
        if archive:
            self._append_history(kind, arguments, expected)
        self._operations.append(operation)

    @_history_atomic
    def set_goal(self, raw: str | bytes, version: int | None = None) -> dict[str, Any]:
        self._check_window()
        self.state = self.model.set_goal(self.state, raw, version)
        self._record("goal", {"raw": raw, "version": version})
        return self.state

    @_history_atomic
    def set_records(self, records: Sequence[U2Record]) -> dict[str, Any]:
        self._check_window()
        self.state = self.model.set_records(self.state, records)
        self._record("records", {"records": tuple(records)})
        return self.state

    @_history_atomic
    def choose(self, legal_mask: torch.Tensor | None = None) -> dict[str, Any]:
        self._check_window()
        self.state, decision = self.model.choose_operation(self.state, legal_mask)
        self.last_decision = decision
        self._record("choose", {"legal_mask": legal_mask}, decision)
        return cast(dict[str, Any], decision)

    @_history_atomic
    def issue(self, decision: dict[str, Any] | None = None, dt: float = 0.05) -> str:
        self._check_window()
        chosen = self.last_decision if decision is None else decision
        if chosen is None or self.last_decision is None:
            raise ValueError("A sampled decision must precede a durable intent")
        assert_values_equal(chosen, self.last_decision, "issued_decision")
        self.state, identity = self.model.issue(self.state, chosen, dt)
        self.state["journal"] = [
            *self.state["journal"][:-1],
            {**self.state["journal"][-1], "status": "intent"},
        ]
        self.execution[identity] = {"status": "intent"}
        self.last_decision = None
        self._record("issue", {"dt": dt}, identity)
        return str(identity)

    def _execution_status(self, identity: str) -> dict[str, Any]:
        found = self.executor.query(identity)
        return {"status": "unknown", "action_id": identity} if found is None else found

    def execute_pending(
        self, checkpoint: str | Path | None = None, fault: str | None = None
    ) -> dict[str, Any]:
        if self._shadow:
            raise ValueError("A shadow replay cannot execute external effects")
        pending = self.state["pending"]
        if pending is None:
            raise ValueError("No pending intent")
        identity = pending["id"]
        status = self._execution_status(identity)
        if status["status"] == "applied":
            return cast(dict[str, Any], self.acknowledge_pending())
        if status["status"] != "not_seen":
            return {"status": "awaiting_execution_status", "action_id": identity}
        # An acknowledgment must fit before any irreversible executor effect.
        self._check_window()
        self.history.append(self._history_payload("ack", {"identity": identity}))
        if checkpoint is None:
            raise ValueError("A durable checkpoint is required before execution")
        self.save(checkpoint)
        if fault == "after_intent":
            raise RuntimeError("injected after_intent")
        self.execution[identity] = {"status": "sent"}
        self.save(checkpoint)
        self.executor.execute(identity, pending["action"])
        if fault == "after_execute_before_ack":
            raise RuntimeError("injected after_execute_before_ack")
        result = cast(dict[str, Any], self.acknowledge_pending())
        self.save(checkpoint)
        return result

    @_history_atomic
    def acknowledge_pending(self) -> dict[str, Any]:
        pending = self.state["pending"]
        if pending is None:
            raise ValueError("No pending intent to acknowledge")
        identity = pending["id"]
        status = self._execution_status(identity)
        if status["status"] != "applied":
            return {
                "status": (
                    "not_executed"
                    if status["status"] == "not_seen"
                    else "awaiting_execution_status"
                ),
                "action_id": identity,
            }
        assert_values_equal(
            status["action"], pending["action"].detach().cpu(), "execution_ack.action"
        )
        if self.execution.get(identity, {}).get("status") == "acknowledged":
            return {"status": "acknowledged", "action_id": identity}
        self._check_window()
        self._apply_ack(identity)
        self._record("ack", {"identity": identity})
        return {"status": "acknowledged", "action_id": identity}

    def _apply_ack(self, identity: str) -> None:
        if self.state["pending"] is None or self.state["pending"]["id"] != identity:
            raise ValueError("Acknowledgment does not bind the pending intent")
        self.execution[identity] = {"status": "acknowledged"}
        rows = [dict(row) for row in self.state["journal"]]
        for row in rows:
            if row.get("id") == identity and row.get("status") == "intent":
                row["status"] = "executed"
        self.state = {**self.state, "journal": rows}

    @_history_atomic
    def set_process_time(self, value: float) -> None:
        self._check_window()
        if not math.isfinite(value) or value < self.state["process_time"]:
            raise ValueError("Processing time must be finite and monotonic")
        self.state = {**self.state, "process_time": float(value)}
        self._record("process_time", {"value": float(value)})

    def advance_process(self, now: float | None = None) -> None:
        current = time.monotonic() if now is None else now
        self.set_process_time(
            max(float(self.state["process_time"]), current - float(self.host_clock["origin"]))
        )

    @_history_atomic
    def receive(self, packet: U2Packet) -> dict[str, Any]:
        self._check_window()
        pending = self.state["pending"]
        if packet.packet_id not in self.state["seen_packets"] and (
            pending is None or self.execution.get(pending["id"], {}).get("status") != "acknowledged"
        ):
            raise ValueError("Observation settlement requires a confirmed execution acknowledgment")
        before = len(self.state["evidence"])
        self.state, result = self.model.receive(self.state, packet)
        self.data_cursor["received_packets"] += len(self.state["evidence"]) - before
        if self.state["pending"] is None and pending is not None:
            self.execution[pending["id"]] = {"status": "settled"}
        self._record("receive", {"packet": packet})
        return cast(dict[str, Any], result)

    @_history_atomic
    def begin_utterance(self) -> dict[str, Any]:
        self._check_window()
        self.state = self.model.begin_utterance(self.state)
        self._record("begin_utterance", {})
        return self.state

    def output(self) -> tuple[Any, Any, Any]:
        goal = self.model.goal(self.state["goal_raw"])
        return cast(
            tuple[Any, Any, Any],
            self.model.output(self.state, goal, self.state["prediction_action"]),
        )

    @_history_atomic
    def accept_byte(self, symbol: int, output: Any = None) -> dict[str, Any]:
        self._check_window(coalesce_emit=True)
        if not self.state["utterance_active"]:
            raise ValueError("Actual self-output requires an active utterance")
        if output is None:
            _, _, output = self.output()
        self.state = self.model.accept_byte(self.state, symbol, output)
        if self._operations and self._operations[-1]["kind"] == "emit":
            self._operations[-1]["arguments"]["symbols"].append(int(symbol))
        else:
            self._record("emit", {"symbols": [int(symbol)]}, archive=False)
        self._append_history("emit", {"symbols": [int(symbol)]})
        return self.state

    @_history_atomic
    def internal(self, decision: dict[str, Any] | None = None) -> dict[str, Any]:
        self._check_window()
        chosen = self.last_decision if decision is None else decision
        if chosen is None or self.last_decision is None:
            raise ValueError("A sampled internal decision is required")
        assert_values_equal(chosen, self.last_decision, "internal_decision")
        self.state = self.model.internal(self.state, chosen)
        self.last_decision = None
        self._record("internal", {})
        return self.state

    @_history_atomic
    def begin_new_window(self) -> None:
        if (
            self.state["pending"] is not None
            or self.state["utterance_active"]
            or self.last_decision is not None
        ):
            raise ValueError("A new window requires a committed subject boundary")
        self._check_window_for_save()
        validate_subject_state(self.model, self.state)
        self._history_segment += 1
        self._append_history("window", {"reason": "truncate"})
        self._history_anchor = self.history.reference()
        self.state = detached_tree(self.state)
        self._anchor = {"kind": "values", "state": detached_tree(self.state)}
        self._anchor_runtime = {
            "data_cursor": detached_tree(self.data_cursor),
            "execution": detached_tree(self.execution),
        }
        self._operations = []
        self._revision = self.model.parameter_revision
        self._parameter_digest = value_digest(self.model.state_dict())

    @_history_atomic
    def birth_transaction(
        self,
        credits: torch.Tensor,
        *,
        count: int = 1,
        branch: int = 0,
        birth_id: str | None = None,
        groups: tuple[int, ...] | None = None,
        validate: (
            Callable[[U2Subject, dict[str, Any], torch.optim.Optimizer | None], bool] | None
        ) = None,
        fault: str | None = None,
    ) -> dict[str, Any]:
        if (
            self.state["pending"] is not None
            or self.state["utterance_active"]
            or self.last_decision is not None
        ):
            raise ValueError("Birth transaction requires a committed boundary")
        self._check_window_for_save()
        validate_subject_state(self.model, self.state)
        optimizer_state = None if self.optimizer is None else self.optimizer.state_dict()
        layout = _optimizer_layout(self.model, self.optimizer)
        model, state = U2Subject.from_bundle(self.model.export_bundle(self.state, optimizer_state))
        optimizer = _restore_optimizer(model, layout, optimizer_state)
        result = model.birth_units(
            credits,
            count=count,
            branch=branch,
            birth_id=birth_id,
            state=state,
            optimizer=optimizer,
            groups=groups,
        )
        if fault in {"after_birth", "after_optimizer"}:
            raise RuntimeError(f"injected {fault}")
        if validate is not None and not bool(validate(model, state, optimizer)):
            self.audit.append({"operation": "birth", "status": "rejected"})
            return {"status": "rejected", **result}
        model.export_bundle(state, None if optimizer is None else optimizer.state_dict())
        validate_subject_state(model, state)
        next_state = detached_tree(state)
        next_anchor = {"kind": "values", "state": detached_tree(next_state)}
        next_anchor_runtime = {
            "data_cursor": detached_tree(self.data_cursor),
            "execution": detached_tree(self.execution),
        }
        next_digest = value_digest(model.state_dict())
        self._check_window_for_save()
        self._history_segment += 1
        self._append_history(
            "birth",
            {
                "credits": credits,
                "count": count,
                "branch": branch,
                "birth_id": result["birth_id"],
                "groups": groups,
            },
            {"result": result, "structure": model.structure(), "parameter_digest": next_digest},
        )
        self._history_anchor = self.history.reference()
        # All fallible shadow work precedes the single-writer commit.
        self.model, self.state, self.optimizer = model, next_state, optimizer
        self._anchor, self._anchor_runtime = next_anchor, next_anchor_runtime
        self._operations = []
        self._revision = model.parameter_revision
        self._parameter_digest = next_digest
        self.audit.append(
            {"operation": "birth", "status": "committed", "birth_id": result["birth_id"]}
        )
        return {"status": "committed", **result}

    def migrate_parameters(
        self,
        candidate: U2Subject,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        acceptance: Callable[[U2Subject, dict[str, Any], torch.optim.Optimizer | None], bool],
        cpu_budget_seconds: float = 30,
        fault: str | None = None,
    ) -> dict[str, Any]:
        from .u2_learning_transaction import migrate_parameters

        return migrate_parameters(
            self,
            candidate,
            optimizer=optimizer,
            acceptance=acceptance,
            cpu_budget_seconds=cpu_budget_seconds,
            fault=fault,
        )

    def to_envelope(self) -> dict[str, Any]:
        self._check_window_for_save()
        optimizer_state = None if self.optimizer is None else self.optimizer.state_dict()
        return {
            "schema": ENVELOPE_SCHEMA,
            "conditions": run_conditions(self.model),
            "source_pins": dict(self._sources),
            "subject_bundle": self.model.export_bundle(self.state, optimizer_state),
            "optimizer_layout": _optimizer_layout(self.model, self.optimizer),
            "history": self.history.export(),
            "window": {
                "anchor": detached_tree(self._anchor),
                "anchor_runtime": detached_tree(self._anchor_runtime),
                "operations": detached_tree(self._operations),
                "parameter_digest": self._parameter_digest,
                "parameter_revision": self._revision,
                "max_operations": self.max_operations,
                "history_anchor": detached_tree(self._history_anchor),
                "history_segment": self._history_segment,
            },
            "runtime": {
                "last_decision": detached_tree(self.last_decision),
                "data_cursor": detached_tree(self.data_cursor),
                "execution": detached_tree(self.execution),
                "audit": detached_tree(self.audit),
            },
            "executor_snapshot": self.executor.snapshot(),
            "host_clock": detached_tree(self.host_clock),
            "credit_scope": "complete_declared_finite_window",
        }

    def _check_window_for_save(self) -> None:
        if self._sources != _source_pins():
            raise ValueError("Runtime source changed inside the learning window")
        if (
            self.model.parameter_revision != self._revision
            or value_digest(self.model.state_dict()) != self._parameter_digest
        ):
            raise ValueError("Frozen window parameters changed without a declared boundary")
        if len(self._operations) > self.max_operations:
            raise ValueError("Learning window exceeds its declared operation budget")
        assert_values_equal(self.history.operations_since(self._history_anchor), self._operations)

    def save(self, path: str | Path) -> Path:
        return atomic_save(self.to_envelope(), path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        executor: ReferenceExecutor | None = None,
        strict_conditions: bool = True,
    ) -> U2Runtime:
        return cls.from_envelope(
            atomic_load(path), executor=executor, strict_conditions=strict_conditions
        )

    @classmethod
    def from_envelope(
        cls,
        envelope: dict[str, Any],
        *,
        executor: ReferenceExecutor | None = None,
        strict_conditions: bool = True,
    ) -> U2Runtime:
        required = {
            "schema",
            "conditions",
            "source_pins",
            "subject_bundle",
            "optimizer_layout",
            "history",
            "window",
            "runtime",
            "executor_snapshot",
            "host_clock",
            "credit_scope",
        }
        if set(envelope) != required or envelope["schema"] != ENVELOPE_SCHEMA:
            raise ValueError("Unsupported or incomplete runtime envelope schema")
        if envelope["source_pins"] != _source_pins():
            raise ValueError("Runtime recovery source version mismatch")
        if envelope["credit_scope"] != "complete_declared_finite_window":
            raise ValueError("Unsupported runtime credit scope")
        model, saved_state = U2Subject.from_bundle(envelope["subject_bundle"])
        validate_subject_state(model, saved_state)
        if strict_conditions and envelope["conditions"] != run_conditions(model):
            raise ValueError("Runtime recovery conditions mismatch")
        window = envelope["window"]
        if set(window) != {
            "anchor",
            "anchor_runtime",
            "operations",
            "parameter_digest",
            "parameter_revision",
            "max_operations",
            "history_anchor",
            "history_segment",
        }:
            raise ValueError("Incomplete causal learning window")
        history = CausalHistory.restore(envelope["history"])
        assert_values_equal(
            history.operations_since(window["history_anchor"]), window["operations"]
        )
        root = history.root
        if root["subject_id"] != saved_state["subject_id"] or root["source_pins"] != _source_pins():
            raise ValueError("Raw history belongs to another subject or source")
        from .u2_group_structure import ancestry_compatible

        if not ancestry_compatible(root["config"], model.c):
            raise ValueError("Raw history root configuration mismatch")
        entries = envelope["history"]["entries"]
        segment = entries[-1]["payload"]["segment"] if entries else 0
        if window["history_segment"] != segment:
            raise ValueError("Raw history window segment mismatch")
        if (
            value_digest(model.state_dict()) != window["parameter_digest"]
            or model.parameter_revision != window["parameter_revision"]
        ):
            raise ValueError("Learning window parameter identity mismatch")
        anchor = window["anchor"]
        anchor_fields = (
            {"kind", "subject_id"} if anchor.get("kind") == "genesis" else {"kind", "state"}
        )
        if set(anchor) != anchor_fields:
            raise ValueError("Unregistered learning-window anchor fields")
        if set(window["anchor_runtime"]) != {"data_cursor", "execution"} or set(
            envelope["runtime"]
        ) != {"last_decision", "data_cursor", "execution", "audit"}:
            raise ValueError("Unregistered runtime recovery fields")
        if set(envelope["host_clock"]) != {"epoch", "origin"}:
            raise ValueError("Unregistered host clock fields")
        if anchor.get("kind") == "genesis":
            result = cls(
                model, subject_id=anchor["subject_id"], max_operations=window["max_operations"]
            )
        elif anchor.get("kind") == "values":
            result = cls(model, state=anchor["state"], max_operations=window["max_operations"])
        else:
            raise ValueError("Unknown learning-window anchor")
        result.executor = (
            executor
            if executor is not None
            else ReferenceExecutor.from_snapshot(envelope["executor_snapshot"])
        )
        result.optimizer = _restore_optimizer(
            model, envelope["optimizer_layout"], envelope["subject_bundle"]["optimizer"]
        )
        result.data_cursor = detached_tree(window["anchor_runtime"]["data_cursor"])
        result.execution = detached_tree(window["anchor_runtime"]["execution"])
        result._anchor_runtime = detached_tree(window["anchor_runtime"])
        result._shadow = True
        try:
            for operation in window["operations"]:
                result._replay_operation(operation)
        finally:
            result._shadow = False
        assert_values_equal(saved_state, result.state)
        assert_values_equal(
            envelope["runtime"]["last_decision"], result.last_decision, "runtime.last_decision"
        )
        assert_values_equal(
            envelope["runtime"]["data_cursor"], result.data_cursor, "runtime.data_cursor"
        )
        # Sent is host transport metadata between intent and acknowledgment.
        expected_execution = envelope["runtime"]["execution"]
        if set(expected_execution) != set(result.execution):
            raise ValueError("Execution recovery identity registry mismatch")
        for identity, status in expected_execution.items():
            actual = result.execution.get(identity)
            if status.get("status") == "sent" and actual == {"status": "intent"}:
                continue
            assert_values_equal(status, actual, f"runtime.execution.{identity}")
        result.execution = detached_tree(expected_execution)
        result.audit = detached_tree(envelope["runtime"]["audit"])
        result._operations = detached_tree(window["operations"])
        result._anchor = detached_tree(anchor)
        result.history = history
        result._history_anchor = detached_tree(window["history_anchor"])
        result._history_segment = segment
        validate_subject_state(result.model, result.state)
        result.host_clock = {
            "epoch": uuid.uuid4().hex,
            "origin": time.monotonic() - float(result.state["process_time"]),
        }
        result.audit.append(
            {
                "operation": "restore",
                "old_host_epoch": envelope["host_clock"]["epoch"],
                "new_host_epoch": result.host_clock["epoch"],
                "logical_process_time": result.state["process_time"],
            }
        )
        return result

    def _replay_operation(self, operation: dict[str, Any]) -> None:
        if set(operation) != {"kind", "arguments", "expected"}:
            raise ValueError("Unregistered causal operation fields")
        kind, args = operation["kind"], operation["arguments"]
        if kind == "goal":
            self.set_goal(**args)
        elif kind == "records":
            self.set_records(**args)
        elif kind == "choose":
            decision = self.choose(**args)
            assert_values_equal(operation["expected"], decision, "replayed_decision")
        elif kind == "issue":
            identity = self.issue(**args)
            assert_values_equal(operation["expected"], identity, "replayed_intent")
        elif kind == "ack":
            self._check_window()
            self._apply_ack(args["identity"])
            self._record("ack", args)
        elif kind == "process_time":
            self.set_process_time(**args)
        elif kind == "receive":
            self.receive(**args)
        elif kind == "begin_utterance":
            self.begin_utterance()
        elif kind == "emit":
            for symbol in args["symbols"]:
                self.accept_byte(symbol)
        elif kind == "internal":
            self.internal()
        else:
            raise ValueError(f"Unregistered causal operation {kind}")
