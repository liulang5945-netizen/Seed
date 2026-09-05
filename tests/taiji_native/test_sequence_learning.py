from copy import deepcopy

import pytest
import torch

from taiji import Taiji, TaijiConfig
from taiji.internalization import content_digest


def test_native_taiji_learns_a_raw_byte_cycle_online() -> None:
    data = b"abcdabcdabcdabcd"
    model = Taiji(
        TaijiConfig(
            region_sizes=(64, 48),
            synapse_fan_in=16,
            motor_fan_in=48,
            seed=7,
        )
    )

    before = model.score_bytes(data)
    model.learn_bytes(data, epochs=200)
    after = model.score_bytes(data)

    assert after["mean_surprise"] < before["mean_surprise"]
    assert after["accuracy"] >= 0.75
    assert model.generate(b"a", 8) == b"bcdabcda"


def test_byte_interfaces_isolate_long_term_memory_by_default(monkeypatch) -> None:
    """F1 byte prediction must not silently inherit F2 episodic feedback."""

    model = Taiji(
        TaijiConfig(
            region_sizes=(16,),
            synapse_fan_in=4,
            motor_fan_in=8,
            memory_units=16,
            memory_fan_in=4,
            memory_readout_fan_in=8,
            memory_meta_dim=8,
            seed=17,
        )
    )
    observed = model.observe
    memory_flags: list[bool] = []

    def observe_spy(symbol: int, **kwargs):
        memory_flags.append(bool(kwargs.get("use_memory", True)))
        return observed(symbol, **kwargs)

    monkeypatch.setattr(model, "observe", observe_spy)

    model.learn_bytes(b"abcd")
    model.score_bytes(b"abcd")
    model.generate(b"a", 2)

    assert memory_flags
    assert not any(memory_flags)

    memory_flags.clear()
    model.score_bytes(b"abcd", use_memory=True)
    assert memory_flags
    assert all(memory_flags)


def test_byte_learning_has_a_dedicated_predictive_readout_and_preserves_action_memory_reads() -> (
    None
):
    """F1 may improve byte prediction without rewriting F2/F4 value readouts.

    M2-2d/e established that the old shared ``ByteMotor`` was trained both as
    a next-byte decoder and as the action policy.  This fixture deliberately
    enables the identity organ, then verifies the narrower contract required
    for continuous sequence learning: byte updates belong to a predictive
    readout; action policy and identity key/value state are read-only.
    """

    data = b"abcd" * 16
    model = Taiji(
        TaijiConfig(
            region_sizes=(32,),
            synapse_fan_in=8,
            motor_fan_in=16,
            memory_units=32,
            memory_fan_in=8,
            memory_readout_fan_in=16,
            memory_meta_dim=16,
            identity_organ_capacity=16,
            identity_organ_value_router_max_keys=8,
            seed=29,
        )
    )
    assert model.identity_organ is not None

    action_before = content_digest(model.motor.to_payload())
    identity_before = content_digest(
        model.identity_organ.to_payload(parent_checkpoint_digest="sequence-readout-test")
    )
    predictive_before = model.predictive_readout.synapses.edge_weight.clone()
    predictive_context_before = content_digest(model.predictive_context.to_payload())
    before = model.score_bytes(data)

    model.learn_bytes(data, epochs=80)

    after = model.score_bytes(data)
    assert content_digest(model.motor.to_payload()) == action_before
    assert (
        content_digest(
            model.identity_organ.to_payload(parent_checkpoint_digest="sequence-readout-test")
        )
        == identity_before
    )
    assert not torch.equal(model.predictive_readout.synapses.edge_weight, predictive_before)
    assert content_digest(model.predictive_context.to_payload()) != predictive_context_before
    assert after["mean_surprise"] < before["mean_surprise"]


def test_byte_learning_can_freeze_shared_fabric_without_freezing_predictive_readout() -> None:
    """M2-2g needs a causal F1-only plasticity control, not a global freeze.

    The live fabric dynamics still provide the predictor's context, but the
    sequence update must be able to leave its persistent substrate untouched.
    This is deliberately narrower than ``learn=False``: the predictive decoder
    must still improve from the same byte evidence.
    """

    data = b"abcd" * 16
    model = Taiji(
        TaijiConfig(
            region_sizes=(32,),
            synapse_fan_in=8,
            motor_fan_in=16,
            memory_units=32,
            memory_fan_in=8,
            memory_readout_fan_in=16,
            memory_meta_dim=16,
            seed=41,
        )
    )
    fabric_before = content_digest(model.fabric.to_payload())
    action_before = content_digest(model.motor.to_payload())
    predictive_before = model.predictive_readout.synapses.edge_weight.clone()
    predictive_context_before = content_digest(model.predictive_context.to_payload())
    before = model.score_bytes(data)

    model.learn_bytes(data, epochs=80, learn_fabric=False)

    after = model.score_bytes(data)
    assert content_digest(model.fabric.to_payload()) == fabric_before
    assert content_digest(model.motor.to_payload()) == action_before
    assert not torch.equal(model.predictive_readout.synapses.edge_weight, predictive_before)
    assert content_digest(model.predictive_context.to_payload()) != predictive_context_before
    assert after["mean_surprise"] < before["mean_surprise"]


def test_byte_learning_can_isolate_predictive_context_and_readout_owners() -> None:
    """B5 diagnosis must be able to freeze either F1 owner independently."""

    data = b"abcd" * 16
    config = TaijiConfig(
        region_sizes=(32,),
        synapse_fan_in=8,
        motor_fan_in=16,
        memory_units=32,
        memory_fan_in=8,
        memory_readout_fan_in=16,
        memory_meta_dim=16,
        seed=53,
    )

    readout_only = Taiji(config)
    context_before = content_digest(readout_only.predictive_context.to_payload())
    readout_before = content_digest(readout_only.predictive_readout.to_payload())
    readout_only.learn_bytes(
        data,
        epochs=20,
        learn_fabric=False,
        learn_predictive_context=False,
    )
    assert content_digest(readout_only.predictive_context.to_payload()) == context_before
    assert content_digest(readout_only.predictive_readout.to_payload()) != readout_before

    context_only = Taiji(config)
    context_before = content_digest(context_only.predictive_context.to_payload())
    readout_before = content_digest(context_only.predictive_readout.to_payload())
    context_only.learn_bytes(
        data,
        epochs=20,
        learn_fabric=False,
        learn_predictive_readout=False,
    )
    assert content_digest(context_only.predictive_context.to_payload()) != context_before
    assert content_digest(context_only.predictive_readout.to_payload()) == readout_before


def test_legacy_shared_motor_checkpoint_migrates_to_a_separate_predictive_readout() -> None:
    """An existing F1/F4 checkpoint must remain loadable without relearning."""

    model = Taiji(
        TaijiConfig(
            region_sizes=(32,),
            synapse_fan_in=8,
            motor_fan_in=16,
            memory_units=32,
            memory_fan_in=8,
            memory_readout_fan_in=16,
            memory_meta_dim=16,
            identity_organ_capacity=16,
            identity_organ_value_router_max_keys=8,
            seed=31,
        )
    )
    for symbol in b"action-owner":
        model.observe(symbol, learn=True)

    legacy = deepcopy(model.checkpoint())
    legacy["format"] = "taiji-native-v8"
    legacy.pop("predictive_context")
    legacy.pop("predictive_readout")
    legacy["config"].pop("predictive_context_seed_offset")
    legacy["config"].pop("predictive_context_fan_in")
    legacy["config"].pop("predictive_context_learning_rate")
    legacy["config"].pop("predictive_context_recurrent_gain")
    legacy["config"].pop("predictive_readout_seed_offset")
    legacy["state"]["version"] = 5
    legacy["state"].pop("predictive_context_trace")
    legacy["state"].pop("readout_kind")
    assert "identity_organ" in legacy
    legacy["identity_organ"]["lineage"]["parent_checkpoint_digest"] = content_digest(
        {key: legacy[key] for key in Taiji._checkpoint_core_keys(include_predictive=False)}
    )

    restored = Taiji.from_checkpoint(legacy)

    assert restored.snapshot().readout_kind == "action"
    migrated_payload = restored.checkpoint()
    assert migrated_payload["format"] == "taiji-native-v10"
    assert migrated_payload["state"]["version"] == 7
    assert torch.equal(restored.motor.synapses.edge_weight, model.motor.synapses.edge_weight)
    assert torch.equal(restored.motor.bias, model.motor.bias)
    assert torch.equal(
        restored.predictive_readout.synapses.edge_weight,
        model.motor.synapses.edge_weight,
    )
    assert torch.equal(restored.predictive_readout.bias, model.motor.bias)
    assert torch.equal(
        restored.predictive_context.receptors.channel, restored.motor.receptors.channel
    )
    assert torch.equal(
        restored.predictive_context.receptors.polarity, restored.motor.receptors.polarity
    )
    assert torch.count_nonzero(restored.predictive_context.recurrent.edge_weight) == 0
    assert torch.count_nonzero(restored.snapshot().predictive_context_trace) == 0

    migrated = Taiji.from_checkpoint(restored.checkpoint())
    assert torch.equal(
        migrated.predictive_readout.synapses.edge_weight,
        restored.predictive_readout.synapses.edge_weight,
    )
    assert torch.equal(
        migrated.predictive_context.recurrent.edge_weight,
        restored.predictive_context.recurrent.edge_weight,
    )


def test_v9_checkpoint_migrates_a_neutral_private_predictive_context() -> None:
    """The v9 F1/F4 split remains inspectable without semantic rewriting."""

    model = Taiji(
        TaijiConfig(
            region_sizes=(32,),
            synapse_fan_in=8,
            motor_fan_in=16,
            memory_units=32,
            memory_fan_in=8,
            memory_readout_fan_in=16,
            memory_meta_dim=16,
            identity_organ_capacity=16,
            identity_organ_value_router_max_keys=8,
            seed=43,
        )
    )
    legacy = deepcopy(model.checkpoint())
    legacy["format"] = "taiji-native-v9"
    legacy.pop("predictive_context")
    for field in (
        "predictive_context_seed_offset",
        "predictive_context_fan_in",
        "predictive_context_learning_rate",
        "predictive_context_recurrent_gain",
    ):
        legacy["config"].pop(field)
    legacy["state"]["version"] = 6
    legacy["state"].pop("predictive_context_trace")
    assert "identity_organ" in legacy
    legacy["identity_organ"]["lineage"]["parent_checkpoint_digest"] = content_digest(
        {
            key: legacy[key]
            for key in Taiji._checkpoint_core_keys(
                include_predictive=True,
                include_predictive_context=False,
            )
        }
    )

    restored = Taiji.from_checkpoint(legacy)

    assert restored.checkpoint()["format"] == "taiji-native-v10"
    assert restored.snapshot().version == 7
    assert torch.equal(restored.predictive_readout.bias, model.predictive_readout.bias)
    assert torch.equal(
        restored.predictive_context.receptors.channel, restored.motor.receptors.channel
    )
    assert torch.equal(
        restored.predictive_context.receptors.polarity, restored.motor.receptors.polarity
    )
    assert torch.count_nonzero(restored.predictive_context.recurrent.edge_weight) == 0


def test_cross_readout_rejection_cannot_commit_a_pending_action_memory_write() -> None:
    """A rejected F4→F1 switch must fail before any delayed write commits."""

    model = Taiji(
        TaijiConfig(
            region_sizes=(32,),
            synapse_fan_in=8,
            motor_fan_in=16,
            memory_units=32,
            memory_fan_in=8,
            memory_readout_fan_in=16,
            memory_meta_dim=16,
            seed=37,
        )
    )
    model.observe(ord("a"), learn=True, readout="action")
    model.act((ord("b"), ord("c")), sample=False)
    model.settle_action(1.0, learn=True)
    memory_before = content_digest(model.memory.to_payload())
    identity_before = (
        None
        if model.identity_organ is None
        else content_digest(
            model.identity_organ.to_payload(parent_checkpoint_digest="readout-switch")
        )
    )

    with pytest.raises(RuntimeError, match="readout changed"):
        model.observe(ord("d"), learn=True, readout="predictive")

    assert content_digest(model.memory.to_payload()) == memory_before
    if model.identity_organ is not None:
        assert (
            content_digest(
                model.identity_organ.to_payload(parent_checkpoint_digest="readout-switch")
            )
            == identity_before
        )
