"""Zero-optimizer tests for fixed-history U2 state reencoding and migration."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from scripts.training.verify_taiji_u2_runtime_22_3 import load_checks
from taiji.u2_history import CausalHistory, assert_values_equal, value_digest
from taiji.u2_learning_transaction import FACT_FIELDS, rebuild_state
from taiji.u2_recovery import ReferenceExecutor, U2Runtime
from taiji.u2_subject import U2Config, U2Subject, synthetic_fixture

helpers = load_checks()


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def settled(tmp_path, precision="float64", *, rich=False):
    dtype = getattr(torch, precision)
    model = U2Subject(U2Config.small()).to(dtype=dtype)
    runtime = U2Runtime(model, executor=ReferenceExecutor(tmp_path / "executor.bin"))
    fixture = synthetic_fixture(model.c, dtype)
    runtime.set_goal(fixture["goal"])
    runtime.set_records(fixture["records"])
    runtime.receive(helpers.issue_round(runtime, fixture, tmp_path, 0))
    if rich:
        packet = helpers.issue_round(runtime, fixture, tmp_path, 1)
        runtime.set_goal(fixture["replacement_goal"])
        unknown = packet.outcome.clone()
        unknown[4:] = 0
        runtime.receive(replace(packet, packet_id="deferred", outcome=unknown))
        runtime.receive(packet)
        runtime.receive(packet)
        runtime.begin_utterance()
        for symbol in [*fixture["target"], 256]:
            runtime.accept_byte(symbol)
        for index in (4, 8):
            mask = torch.zeros(9, dtype=torch.bool)
            mask[index] = True
            runtime.internal(runtime.choose(mask))
        runtime.begin_new_window()
        for generation in (1, 2):
            credit = torch.zeros(model.c.groups, model.c.d, 8 * model.c.d, dtype=dtype)
            credit[:, 0, 0], credit[:, 1, 1] = 3, 2
            runtime.birth_transaction(credit, birth_id=f"inherited-{generation}")
    return runtime, fixture


def candidate_from(runtime, *, changed=True):
    candidate, _ = U2Subject.from_bundle(runtime.model.export_bundle(runtime.state))
    if changed:
        with torch.no_grad():
            candidate.text_input.bias.add_(0.15)
            candidate.critic.last.bias.add_(0.1)
    candidate.mark_parameter_update()
    return candidate


def close_values(left, right, precision):
    if isinstance(left, torch.Tensor):
        tolerance = (
            {"atol": 1e-10, "rtol": 1e-8}
            if precision == "float64"
            else {"atol": 1e-5, "rtol": 1e-4}
        )
        assert torch.allclose(left, right, **tolerance)
    elif isinstance(left, dict):
        assert set(left) == set(right)
        for name in left:
            close_values(left[name], right[name], precision)
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            close_values(a, b, precision)
    else:
        assert_values_equal(left, right)


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_same_parameter_full_history_reconstructs_all_state_fields(tmp_path, precision):
    runtime, _ = settled(tmp_path, precision, rich=True)
    state, metrics = rebuild_state(
        runtime.model, runtime.history, runtime.state, reset_eligibility=False
    )
    close_values(runtime.state, state, precision)
    assert metrics["births"] == 2
    assert metrics["internal"] == 2
    assert metrics["duplicates"] == 1
    assert all(not value.requires_grad for value in (state["H"], state["h"], state["F"]))


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_new_encoder_migration_changes_representation_preserves_facts_and_rng(tmp_path, precision):
    runtime, _ = settled(tmp_path, precision)
    old_state, old_model = runtime.state, runtime.model
    old_prefix = runtime.history.reference()
    effects = runtime.executor.effect_count
    result = runtime.migrate_parameters(
        candidate_from(runtime), acceptance=lambda model, state, opt: True
    )
    assert result["status"] == "committed"
    assert runtime.model is not old_model and runtime.model.parameter_revision == 1
    assert not torch.equal(runtime.state["H"], old_state["H"])
    assert not torch.equal(runtime.state["F"], old_state["F"])
    assert not torch.equal(runtime.state["history"][0]["keys"], old_state["history"][0]["keys"])
    for name in FACT_FIELDS:
        if name != "journal":
            assert_values_equal(old_state[name], runtime.state[name], name)
    assert runtime.state["journal"][:-1] == old_state["journal"]
    assert all(row["parameter_revision"] == 1 for row in runtime.state["history"])
    assert all(row["source_subject_version"] == 0 for row in runtime.state["history"])
    assert all(not bool(value.any()) for value in runtime.state["eligibility"].values())
    assert runtime.history.validate_reference(old_prefix) == old_prefix["entries"]
    assert runtime.operation_count == 0
    assert runtime.executor.effect_count == effects


def test_reencoding_uses_no_policy_sampling_or_critic_credit(tmp_path, monkeypatch):
    runtime, _ = settled(tmp_path)
    candidate = candidate_from(runtime)

    def forbidden(*args, **kwargs):
        raise AssertionError("historical policy or external execution was called")

    monkeypatch.setattr(U2Subject, "policy", forbidden)
    monkeypatch.setattr(U2Subject, "choose_operation", forbidden)
    monkeypatch.setattr(U2Subject, "pending_value", forbidden)
    monkeypatch.setattr("taiji.u2_events.behavior_credit", forbidden)
    monkeypatch.setattr(ReferenceExecutor, "execute", forbidden)
    original_init = U2Subject.__init__

    def guarded_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.critic.register_forward_pre_hook(forbidden)

    monkeypatch.setattr(U2Subject, "__init__", guarded_init)
    receipt = runtime.migrate_parameters(candidate, acceptance=lambda model, state, opt: True)
    assert receipt["status"] == "committed"


def test_assimilation_is_independent_of_stale_behavior_critic(tmp_path):
    runtime, fixture = helpers.settings(tmp_path)
    packet = helpers.issue_round(runtime, fixture, tmp_path, 0)
    state = helpers.serialized(runtime.model.export_bundle(runtime.state))["state"]
    state["pending"]["decision"]["psi_version"] = -1
    assimilated, observation = runtime.model.assimilate(state, packet)
    assert assimilated["pending"] is None and "predictive_logp" in observation
    assert "actor" not in observation and "critic" not in observation
    with pytest.raises(ValueError, match="Psi version"):
        runtime.model.receive(state, packet)


def test_deferred_observation_provenance_survives_reencoding(tmp_path):
    runtime, fixture = helpers.settings(tmp_path)
    packet = helpers.issue_round(runtime, fixture, tmp_path, 0)
    unknown = packet.outcome.clone()
    unknown[4:] = 0
    runtime.receive(replace(packet, packet_id="partial-observation", outcome=unknown))
    complete = replace(packet, packet_id="complete-outcome", observations={}, coordinates={})
    runtime.receive(complete)
    rebuilt, _ = rebuild_state(
        runtime.model, runtime.history, runtime.state, reset_eligibility=False
    )
    close_values(runtime.state, rebuilt, "float64")
    row = rebuilt["history"][-1]
    assert row["source_packet_id"] == "complete-outcome"
    assert row["source_observation_packet_id"] == "partial-observation"


def test_recorded_reconsidered_decisions_remain_fixed(tmp_path):
    runtime, fixture = settled(tmp_path)
    mask = torch.zeros(9, dtype=torch.bool)
    mask[0] = True
    runtime.choose(mask)
    runtime.choose(mask)
    runtime.receive(helpers.issue_round(runtime, fixture, tmp_path, 1))
    rebuilt, _ = rebuild_state(
        runtime.model, runtime.history, runtime.state, reset_eligibility=False
    )
    close_values(runtime.state, rebuilt, "float64")


def test_two_migrations_keep_complete_history_and_roundtrip_current_state(tmp_path):
    runtime, _ = settled(tmp_path, rich=True)
    for index in (1, 2):
        runtime.migrate_parameters(
            candidate_from(runtime), acceptance=lambda model, state, opt: True
        )
        assert runtime.state["resources"]["updates"] == index
    state, _ = rebuild_state(runtime.model, runtime.history, runtime.state)
    close_values(runtime.state, state, "float64")
    path = runtime.save(tmp_path / "accepted.bin")
    restored = U2Runtime.load(path, executor=runtime.executor)
    assert restored.history.complete
    assert restored.history.reference() == runtime.history.reference()
    assert_values_equal(runtime.state, restored.state)


@pytest.mark.parametrize("fault", ["after_rebuild", "before_commit"])
def test_injected_migration_fault_keeps_parent_intact(tmp_path, fault):
    runtime, _ = settled(tmp_path)
    before = value_digest(runtime.to_envelope())
    model = runtime.model
    with pytest.raises(RuntimeError, match="injected"):
        runtime.migrate_parameters(
            candidate_from(runtime), acceptance=lambda model, state, opt: True, fault=fault
        )
    assert runtime.model is model
    assert value_digest(runtime.to_envelope()) == before


def test_retention_rejection_preserves_parent_and_history(tmp_path):
    runtime, _ = settled(tmp_path)
    before = value_digest(runtime.to_envelope())
    result = runtime.migrate_parameters(
        candidate_from(runtime), acceptance=lambda model, state, opt: False
    )
    assert result["status"] == "rejected"
    assert value_digest(runtime.to_envelope()) == before


def test_budget_exhaustion_preserves_parent(tmp_path):
    runtime, _ = settled(tmp_path)
    candidate = candidate_from(runtime)
    before = value_digest(runtime.to_envelope())
    with pytest.raises(RuntimeError, match="migration_unaffordable"):
        runtime.migrate_parameters(
            candidate, acceptance=lambda model, state, opt: True, cpu_budget_seconds=1e-12
        )
    assert value_digest(runtime.to_envelope()) == before


def test_partial_history_cannot_be_reencoded_as_complete(tmp_path):
    runtime, _ = settled(tmp_path)
    partial = U2Runtime(runtime.model, state=runtime.state, executor=runtime.executor)
    with pytest.raises(ValueError, match="coverage_incomplete"):
        partial.migrate_parameters(
            candidate_from(partial), acceptance=lambda model, state, opt: True
        )


@pytest.mark.parametrize(
    "mutation", ["version", "structure", "dtype", "nonfinite", "same_instance"]
)
def test_invalid_candidate_is_rejected_before_commit(tmp_path, mutation):
    runtime, _ = settled(tmp_path)
    candidate = candidate_from(runtime)
    if mutation == "version":
        candidate.parameter_revision += 1
    elif mutation == "structure":
        candidate.unit_ids[0][0] = "foreign-lineage"
    elif mutation == "dtype":
        candidate = candidate.float()
    elif mutation == "nonfinite":
        with torch.no_grad():
            candidate.native_w.fill_(torch.nan)
    else:
        candidate = runtime.model
    before = value_digest(runtime.to_envelope())
    with pytest.raises(ValueError):
        runtime.migrate_parameters(candidate, acceptance=lambda model, state, opt: True)
    assert value_digest(runtime.to_envelope()) == before


def test_pending_action_blocks_slow_parameter_migration(tmp_path):
    runtime, fixture = helpers.settings(tmp_path)
    helpers.issue_round(runtime, fixture, tmp_path, 0)
    with pytest.raises(ValueError, match="committed subject boundary"):
        runtime.migrate_parameters(
            candidate_from(runtime), acceptance=lambda model, state, opt: True
        )


def test_unrecorded_fact_change_is_detected(tmp_path):
    runtime, _ = settled(tmp_path)
    runtime.state["goal_raw"] = b"not in the causal history"
    with pytest.raises(ValueError, match="migration.facts"):
        rebuild_state(runtime.model, runtime.history, runtime.state)


def test_parent_change_during_acceptance_prevents_overwrite(tmp_path):
    runtime, _ = settled(tmp_path)
    model = runtime.model

    def acceptance(candidate, state, optimizer):
        runtime.set_goal(b"concurrent legitimate goal")
        return True

    with pytest.raises(ValueError, match="parent_changed"):
        runtime.migrate_parameters(candidate_from(runtime), acceptance=acceptance)
    assert runtime.model is model
    assert runtime.state["goal_raw"] == b"concurrent legitimate goal"


@pytest.mark.parametrize("target", ["parameters", "state", "configuration"])
def test_acceptance_cannot_mutate_the_reencoded_candidate(tmp_path, target):
    runtime, _ = settled(tmp_path)
    before = value_digest(runtime.to_envelope())

    def acceptance(candidate, state, optimizer):
        if target == "parameters":
            with torch.no_grad():
                candidate.text_input.bias.add_(0.1)
        elif target == "state":
            state["F"].zero_()
        else:
            candidate.c = replace(candidate.c, seed=candidate.c.seed + 1)
        return True

    with pytest.raises(ValueError, match="acceptance_callback_mutated_candidate"):
        runtime.migrate_parameters(candidate_from(runtime), acceptance=acceptance)
    assert value_digest(runtime.to_envelope()) == before


def test_source_change_at_acceptance_is_rejected(tmp_path, monkeypatch):
    runtime, _ = settled(tmp_path)
    candidate = candidate_from(runtime)
    model = runtime.model

    def acceptance(candidate, state, optimizer):
        monkeypatch.setattr("taiji.u2_recovery._source_pins", lambda: {"changed": "0" * 64})
        return True

    with pytest.raises(ValueError, match="source changed"):
        runtime.migrate_parameters(candidate, acceptance=acceptance)
    assert runtime.model is model


def test_history_capacity_failure_preserves_parent(tmp_path):
    runtime, _ = settled(tmp_path)
    history = CausalHistory(runtime.history.root, max_entries=runtime.history.entry_count)
    for node in runtime.history.export()["entries"]:
        history = history.append(node["payload"])
    runtime.history = history
    before = value_digest(runtime.to_envelope())
    with pytest.raises(ValueError, match="raw_history_budget_stop"):
        runtime.migrate_parameters(
            candidate_from(runtime), acceptance=lambda model, state, opt: True
        )
    assert value_digest(runtime.to_envelope()) == before


def test_optimizer_moments_follow_the_candidate_mapping(tmp_path):
    runtime, _ = settled(tmp_path)
    runtime.optimizer = torch.optim.Adam(runtime.model.parameters(), lr=1e-3)
    candidate = candidate_from(runtime)
    optimizer = torch.optim.Adam(candidate.parameters(), lr=1e-3)
    for model, opt in ((runtime.model, runtime.optimizer), (candidate, optimizer)):
        for parameter in model.parameters():
            opt.state[parameter] = {
                "step": torch.tensor(2.0),
                "exp_avg": torch.full_like(parameter, 0.2),
                "exp_avg_sq": torch.full_like(parameter, 0.3),
            }
    before = value_digest(optimizer.state_dict())
    runtime.migrate_parameters(
        candidate, optimizer=optimizer, acceptance=lambda model, state, opt: True
    )
    assert runtime.optimizer is not optimizer
    assert value_digest(runtime.optimizer.state_dict()) == before
    assert {id(p) for group in runtime.optimizer.param_groups for p in group["params"]} == {
        id(p) for p in runtime.model.parameters()
    }


def test_existing_optimizer_cannot_be_silently_dropped(tmp_path):
    runtime, _ = settled(tmp_path)
    runtime.optimizer = torch.optim.Adam(runtime.model.parameters(), lr=1e-3)
    with pytest.raises(ValueError, match="Candidate optimizer"):
        runtime.migrate_parameters(
            candidate_from(runtime), acceptance=lambda model, state, opt: True
        )
