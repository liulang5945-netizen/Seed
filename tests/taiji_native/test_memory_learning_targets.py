from copy import deepcopy

import torch

from taiji import (
    EPISODIC_LEARNING_TARGETS,
    DelayedMemoryTask,
    MemoryEpisode,
    Taiji,
    TaijiConfig,
)
from taiji.internalization import content_digest

EPISODE = MemoryEpisode(
    memory_id="target-contract-episode",
    cue=17,
    action=48,
    outcome=43,
)


def _memory_weights(model: Taiji) -> dict[str, torch.Tensor]:
    memory = model.memory
    return {
        "association": memory.association.edge_weight.detach().clone(),
        "action_readout": memory.action_readout.edge_weight.detach().clone(),
        "outcome_readout": memory.outcome_readout.edge_weight.detach().clone(),
        "cortical_readout": memory.cortical_readout.edge_weight.detach().clone(),
    }


def _write(targets: str | None) -> tuple[Taiji, dict[str, torch.Tensor]]:
    model = Taiji(TaijiConfig(seed=11), episode_id="target-contract")
    before = _memory_weights(model)
    kwargs = {} if targets is None else {"memory_learning_targets": targets}
    DelayedMemoryTask._write_episode(model, EPISODE, **kwargs)
    return model, before


def test_learning_target_contract_exposes_legacy_and_isolated_targets() -> None:
    assert EPISODIC_LEARNING_TARGETS == (
        "all",
        "association",
        "readout",
        "action_readout",
        "outcome_readout",
    )


def test_default_all_is_checkpoint_compatible_with_explicit_all() -> None:
    default_model, _ = _write(None)
    explicit_model, _ = _write("all")
    assert content_digest(default_model.checkpoint()) == content_digest(
        explicit_model.checkpoint()
    )


def test_action_readout_target_only_updates_action_path() -> None:
    model, before = _write("action_readout")
    after = _memory_weights(model)
    assert not torch.equal(after["action_readout"], before["action_readout"])
    assert torch.equal(after["association"], before["association"])
    assert torch.equal(after["outcome_readout"], before["outcome_readout"])
    assert torch.equal(after["cortical_readout"], before["cortical_readout"])


def test_outcome_readout_target_only_updates_outcome_path() -> None:
    model, before = _write("outcome_readout")
    after = _memory_weights(model)
    assert not torch.equal(after["outcome_readout"], before["outcome_readout"])
    assert torch.equal(after["association"], before["association"])
    assert torch.equal(after["action_readout"], before["action_readout"])
    assert torch.equal(after["cortical_readout"], before["cortical_readout"])


def test_isolated_target_survives_checkpoint_roundtrip() -> None:
    model, _ = _write("action_readout")
    checkpoint = deepcopy(model.checkpoint())
    restored = Taiji.from_checkpoint(checkpoint)
    assert content_digest(restored.checkpoint()) == content_digest(checkpoint)
