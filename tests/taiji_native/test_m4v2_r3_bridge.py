from __future__ import annotations

from copy import deepcopy

import torch

from taiji import Taiji, TaijiConfig
from taiji.internalization import content_digest


def _config() -> TaijiConfig:
    return TaijiConfig(
        region_sizes=(16,),
        synapse_fan_in=4,
        motor_fan_in=8,
        predictive_context_fan_in=4,
        memory_units=16,
        memory_fan_in=4,
        memory_meta_dim=8,
        memory_readout_fan_in=4,
        identity_organ_enabled=False,
        seed=123,
    )


def test_r3_zero_gate_is_a_main_path_equivalence_and_noop() -> None:
    parent = Taiji(_config(), episode_id="r3-parent")
    parent_checkpoint = deepcopy(parent.checkpoint())
    candidate = Taiji.from_checkpoint(deepcopy(parent_checkpoint))
    candidate.enable_adaptive_residual_bridge(gate=0.0)

    bridge_before = content_digest(candidate.adaptive_residual_bridge.to_payload())
    assert candidate.adaptive_residual_bridge.gate == 0.0
    assert candidate.adaptive_residual_bridge.region.recurrent.row_fan_in == 4
    assert candidate.generate(b"ab", 8) == parent.generate(b"ab", 8)
    assert content_digest(candidate.adaptive_residual_bridge.to_payload()) == bridge_before
    assert content_digest(candidate.predictive_context.to_payload()) == content_digest(
        parent_checkpoint["predictive_context"]
    )
    assert content_digest(candidate.predictive_readout.to_payload()) == content_digest(
        parent_checkpoint["predictive_readout"]
    )


def test_r3_open_gate_is_on_predictive_path_and_receives_local_credit() -> None:
    parent = Taiji(_config(), episode_id="r3-active-parent")
    checkpoint = parent.checkpoint()
    baseline = Taiji.from_checkpoint(deepcopy(checkpoint))
    candidate = Taiji.from_checkpoint(deepcopy(checkpoint))
    candidate.enable_adaptive_residual_bridge(gate=1.0, residual_gain=1.0)

    baseline.reset_dynamics(episode_id="r3-probe")
    candidate.reset_dynamics(episode_id="r3-probe")
    baseline_step = baseline.observe(97, learn=False, readout="predictive")
    candidate_step = candidate.observe(97, learn=False, readout="predictive")

    assert bool(candidate.adaptive_residual_bridge.region.activity.abs().any())
    assert not torch.allclose(baseline_step.probabilities, candidate_step.probabilities)

    context_before = content_digest(candidate.predictive_context.to_payload())
    readout_before = content_digest(candidate.predictive_readout.to_payload())
    bridge_before = content_digest(candidate.adaptive_residual_bridge.to_payload())
    candidate.observe(
        98,
        learn=True,
        learn_fabric=False,
        learn_predictive_context=False,
        learn_predictive_readout=False,
        learn_adaptive_residual_bridge=True,
        readout="predictive",
    )
    bridge_after = content_digest(candidate.adaptive_residual_bridge.to_payload())
    assert bridge_after != bridge_before
    assert content_digest(candidate.predictive_context.to_payload()) == context_before
    assert content_digest(candidate.predictive_readout.to_payload()) == readout_before


def test_r3_lesion_is_equivalent_to_absent_bridge_and_restore_is_exact() -> None:
    source = Taiji(_config(), episode_id="r3-restore")
    source.enable_adaptive_residual_bridge(gate=1.0, residual_gain=1.0)
    source.learn_bytes(b"abba-caba", epochs=1, learn_fabric=False)
    checkpoint = source.checkpoint()
    restored = Taiji.from_checkpoint(deepcopy(checkpoint))

    assert restored.adaptive_residual_bridge_enabled
    assert content_digest(restored.checkpoint()) == content_digest(checkpoint)

    parent = Taiji.from_checkpoint(deepcopy(checkpoint))
    parent.disable_adaptive_residual_bridge()
    lesioned = Taiji.from_checkpoint(deepcopy(checkpoint))
    lesioned.lesion_adaptive_residual_bridge()
    parent.reset_dynamics(episode_id="r3-lesion")
    lesioned.reset_dynamics(episode_id="r3-lesion")
    parent_step = parent.observe(97, learn=False, readout="predictive")
    lesioned_step = lesioned.observe(97, learn=False, readout="predictive")

    assert torch.equal(parent_step.probabilities, lesioned_step.probabilities)
    assert lesioned.adaptive_residual_bridge.lesioned is True

    rolled_back = Taiji.from_checkpoint(deepcopy(checkpoint))
    rolled_back.restore(source.checkpoint())
    assert rolled_back.adaptive_residual_bridge_enabled
    assert content_digest(rolled_back.checkpoint()) == content_digest(checkpoint)
