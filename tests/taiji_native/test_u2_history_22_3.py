"""Raw-history preservation through the U2 runtime's normal entry points."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from scripts.training.verify_taiji_u2_runtime_22_3 import load_checks
from taiji.u2_history import CausalHistory, value_digest
from taiji.u2_recovery import U2Runtime

helpers = load_checks()


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def bounded_history(runtime, *, max_entries=96, max_bytes=16 * 1024 * 1024):
    bounded = CausalHistory(runtime.history.root, max_entries=max_entries, max_bytes=max_bytes)
    for node in runtime.history.export()["entries"]:
        bounded = bounded.append(node["payload"])
    runtime.history = bounded


def test_history_survives_truncation_save_restore_and_next_window(tmp_path):
    runtime, fixture = helpers.settings(tmp_path)
    runtime.receive(helpers.issue_round(runtime, fixture, tmp_path, 0))
    prefix = runtime.history.reference()
    previous_entries = runtime.history.export()["entries"]
    runtime.begin_new_window()
    assert runtime.operation_count == 0
    assert runtime.history.entry_count == prefix["entries"] + 1
    assert runtime.history.validate_reference(prefix) == prefix["entries"]
    assert value_digest(runtime.history.export()["entries"][:-1]) == value_digest(previous_entries)
    checkpoint = runtime.save(tmp_path / "full-history.bin")
    restored = U2Runtime.load(checkpoint, executor=runtime.executor)
    assert restored.history.reference() == runtime.history.reference()
    assert restored.history.complete
    restored.history.require_complete()
    restored.set_goal(fixture["replacement_goal"])
    assert restored.history.entry_count == runtime.history.entry_count + 1
    assert value_digest(previous_entries) == value_digest(runtime.history.export()["entries"][:-1])


def test_each_actual_byte_is_archived_and_window_emit_is_coalesced(tmp_path):
    runtime, fixture = helpers.settings(tmp_path)
    runtime.begin_utterance()
    prefix_count = runtime.history.entry_count
    for symbol in [*fixture["target"], 256]:
        runtime.accept_byte(symbol)
    nodes = runtime.history.export()["entries"][prefix_count:]
    assert [node["payload"]["operation"]["arguments"]["symbols"] for node in nodes] == [
        [symbol] for symbol in [*fixture["target"], 256]
    ]
    assert runtime._operations[-1]["arguments"]["symbols"] == [*fixture["target"], 256]
    restored = U2Runtime.from_envelope(
        helpers.serialized(runtime.to_envelope()), executor=runtime.executor
    )
    assert restored.history.reference() == runtime.history.reference()
    assert len(restored.state["evidence"]) == 0


def test_values_anchor_is_explicitly_partial_for_future_migration(tmp_path):
    runtime, _ = helpers.settings(tmp_path)
    partial = U2Runtime(runtime.model, state=runtime.state, executor=runtime.executor)
    assert not partial.history.complete
    with pytest.raises(ValueError, match="coverage_incomplete"):
        partial.history.require_complete()
    restored = U2Runtime.from_envelope(
        helpers.serialized(partial.to_envelope()), executor=runtime.executor
    )
    assert not restored.history.complete


@pytest.mark.parametrize(
    "mutation", ["order", "missing", "duplicate", "payload", "parent", "size", "head", "extra"]
)
def test_history_corruption_is_rejected(tmp_path, mutation):
    runtime, _ = helpers.settings(tmp_path)
    envelope = helpers.serialized(runtime.to_envelope())
    history = envelope["history"]
    entries = list(history["entries"])
    if mutation == "order":
        entries.reverse()
    elif mutation == "missing":
        entries.pop(0)
    elif mutation == "duplicate":
        entries.append(copy.deepcopy(entries[-1]))
    elif mutation == "payload":
        entries[0]["payload"]["operation"]["arguments"]["raw"] = b"changed"
    elif mutation == "parent":
        entries[-1]["parent"] = "0" * 64
    elif mutation == "size":
        entries[-1]["payload_bytes"] += 1
    elif mutation == "head":
        history["head"] = "0" * 64
    else:
        entries[-1]["unregistered"] = True
    history["entries"] = tuple(entries)
    with pytest.raises(ValueError):
        U2Runtime.from_envelope(envelope, executor=runtime.executor)


@pytest.mark.parametrize("mutation", ["prefix", "segment", "window", "subject"])
def test_history_and_runtime_binding_are_verified(tmp_path, mutation):
    runtime, _ = helpers.settings(tmp_path)
    runtime.begin_new_window()
    envelope = helpers.serialized(runtime.to_envelope())
    if mutation == "prefix":
        envelope["window"]["history_anchor"]["head"] = "0" * 64
    elif mutation == "segment":
        envelope["window"]["history_segment"] += 1
    elif mutation == "window":
        envelope["window"]["history_anchor"] = {
            "entries": 0,
            "head": value_digest(runtime.history.root),
        }
    else:
        root = runtime.history.root
        root["subject_id"] = "other-subject"
        replacement = CausalHistory(root)
        for node in runtime.history.export()["entries"]:
            replacement = replacement.append(node["payload"])
        envelope["history"] = replacement.export()
        envelope["window"]["history_anchor"] = replacement.reference()
    with pytest.raises(ValueError):
        U2Runtime.from_envelope(envelope, executor=runtime.executor)


def test_history_quota_rolls_back_sampled_rng_and_decision(tmp_path):
    runtime, _ = helpers.settings(tmp_path)
    bounded_history(runtime, max_entries=runtime.history.entry_count)
    before_state, before_history = value_digest(runtime.state), runtime.history.reference()
    with pytest.raises(ValueError, match="raw_history_budget_stop"):
        runtime.choose(torch.ones(9, dtype=torch.bool))
    assert value_digest(runtime.state) == before_state
    assert runtime.last_decision is None
    assert runtime.operation_count == 2
    assert runtime.history.reference() == before_history


def test_history_byte_quota_is_atomic_for_new_content(tmp_path):
    runtime, _ = helpers.settings(tmp_path)
    bounded_history(runtime, max_bytes=runtime.history.total_bytes)
    state, history = runtime.state, runtime.history
    with pytest.raises(ValueError, match="raw_history_budget_stop"):
        runtime.set_goal(b"a new goal with insufficient history capacity")
    assert runtime.state is state and runtime.history is history


def test_raw_recording_exception_rolls_back_local_operation(tmp_path, monkeypatch):
    runtime, _ = helpers.settings(tmp_path)
    before = value_digest(runtime.to_envelope())

    def fail_append(self, payload):
        raise OSError("injected raw recording failure")

    monkeypatch.setattr(CausalHistory, "append", fail_append)
    with pytest.raises(OSError, match="injected"):
        runtime.set_goal(b"new goal")
    assert value_digest(runtime.to_envelope()) == before


def test_history_quota_rolls_back_coalesced_utf8_prefix(tmp_path):
    runtime, _ = helpers.settings(tmp_path)
    runtime.begin_utterance()
    runtime.accept_byte(0xE7)
    bounded_history(runtime, max_entries=runtime.history.entry_count)
    before = value_digest(runtime.to_envelope())
    with pytest.raises(ValueError, match="raw_history_budget_stop"):
        runtime.accept_byte(0xBA)
    assert value_digest(runtime.to_envelope()) == before
    assert runtime.state["utf8_remaining"] == 2


def test_history_capacity_checked_before_external_execution(tmp_path):
    runtime, _ = helpers.settings(tmp_path)
    mask = torch.zeros(9, dtype=torch.bool)
    mask[0] = True
    identity = runtime.issue(runtime.choose(mask))
    bounded_history(runtime, max_entries=runtime.history.entry_count)
    ledger = (tmp_path / "executor.bin").read_bytes()
    with pytest.raises(ValueError, match="raw_history_budget_stop"):
        runtime.execute_pending(checkpoint=tmp_path / "intent.bin")
    assert runtime.executor.effect_count == 0
    assert (tmp_path / "executor.bin").read_bytes() == ledger
    assert runtime.execution[identity] == {"status": "intent"}


def test_history_quota_rolls_back_receipt_and_fast_memory(tmp_path):
    runtime, fixture = helpers.settings(tmp_path)
    packet = helpers.issue_round(runtime, fixture, tmp_path, 0)
    bounded_history(runtime, max_entries=runtime.history.entry_count)
    before = value_digest(runtime.to_envelope())
    with pytest.raises(ValueError, match="raw_history_budget_stop"):
        runtime.receive(packet)
    assert value_digest(runtime.to_envelope()) == before
    assert runtime.state["pending"] is not None
    assert runtime.executor.effect_count == 1


@pytest.mark.parametrize("operation", ["window", "birth"])
def test_history_quota_preserves_parent_at_boundary(tmp_path, operation):
    runtime, _ = helpers.settings(tmp_path)
    bounded_history(runtime, max_entries=runtime.history.entry_count)
    before = value_digest(runtime.to_envelope())
    model = runtime.model
    with pytest.raises(ValueError, match="raw_history_budget_stop"):
        if operation == "window":
            runtime.begin_new_window()
        else:
            runtime.birth_transaction(helpers.birth_credit(model))
    assert runtime.model is model
    assert value_digest(runtime.to_envelope()) == before


def test_history_keeps_two_births_and_original_prefix(tmp_path):
    runtime, _ = helpers.settings(tmp_path)
    prefix = runtime.history.reference()
    for generation in (1, 2):
        runtime.birth_transaction(
            helpers.birth_credit(runtime.model), birth_id=f"history-{generation}"
        )
    nodes = runtime.history.export()["entries"]
    assert [node["payload"]["operation"]["kind"] for node in nodes] == [
        "goal",
        "records",
        "birth",
        "birth",
    ]
    assert runtime.history.validate_reference(prefix) == 2
    restored = U2Runtime.from_envelope(
        helpers.serialized(runtime.to_envelope()), executor=runtime.executor
    )
    assert restored.model.c.units == runtime.model.c.units
    assert restored.history.reference() == runtime.history.reference()


def test_shadow_restore_preserves_history_without_appending_replayed_facts(tmp_path):
    runtime, _, _ = helpers.pending_after_one_observation(tmp_path)
    reference = runtime.history.reference()
    restored = U2Runtime.from_envelope(
        helpers.serialized(runtime.to_envelope()), executor=runtime.executor
    )
    assert restored.history.reference() == reference
    assert restored.executor.effect_count == runtime.executor.effect_count
    assert restored.history.total_bytes == runtime.history.total_bytes


def test_legacy_runtime_without_raw_prefix_is_explicitly_rejected(tmp_path):
    runtime, _ = helpers.settings(tmp_path)
    envelope = helpers.serialized(runtime.to_envelope())
    envelope["schema"] = "taiji-u2-runtime-envelope-v1"
    del envelope["history"]
    with pytest.raises(ValueError, match="schema"):
        U2Runtime.from_envelope(envelope, executor=runtime.executor)


def test_failed_bundle_replace_preserves_previous_complete_raw_prefix(tmp_path, monkeypatch):
    runtime, _ = helpers.settings(tmp_path)
    checkpoint = runtime.save(tmp_path / "history.bin")
    previous = checkpoint.read_bytes()
    previous_reference = runtime.history.reference()
    runtime.begin_new_window()

    def fail_replace(self, target):
        raise OSError("injected raw history bundle failure")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", fail_replace)
        with pytest.raises(OSError, match="injected"):
            runtime.save(checkpoint)
    assert checkpoint.read_bytes() == previous
    restored = U2Runtime.load(checkpoint, executor=runtime.executor)
    assert restored.history.reference() == previous_reference
    assert restored.history.complete


def test_raw_history_parameter_revisions_cannot_regress(tmp_path):
    runtime, _ = helpers.settings(tmp_path)
    for generation in (1, 2):
        runtime.birth_transaction(
            helpers.birth_credit(runtime.model), birth_id=f"revision-{generation}"
        )
    candidate = runtime._history_payload("goal", {"raw": b"new", "version": None})
    candidate["parameter_revision"] = 0
    previous = runtime.history.reference()
    with pytest.raises(ValueError, match="regressed"):
        runtime.history.append(candidate)
    assert runtime.history.reference() == previous
