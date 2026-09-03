"""M1-66b: per-slot value router keeps folded keys decisive.

A cue slot occasionally folds several distinct keys at the write threshold and
the single blended value head then contradicts itself on half the reads.  The
router keeps, per slot, a bounded table of the keys written there and the
rewarded action each was bound under; a read's verdict follows the nearest
stored key instead of the blended average.  Punished writes drop their entry
(reward-modulated canary) and slot allocation stays reward-orthogonal (M1-63).
"""

from __future__ import annotations

import torch

from scripts.training.eval_taiji_foundation_baseline import _memory_config
from scripts.training.eval_taiji_m1_64_foundation_memory import (
    build_foundation_delayed_memory_corpus,
)
from taiji import Taiji
from taiji.config import TaijiConfig
from taiji.foundation_tasks import DelayedMemoryTask


def _model(*, router_off: bool = False) -> Taiji:
    values = _memory_config(11).to_dict()
    values["identity_organ_capacity"] = 256
    if router_off:
        values["identity_organ_value_router_enabled"] = False
    return Taiji(TaijiConfig.from_dict(values), episode_id="m1-66b")


def test_router_records_rewarded_keys_and_their_actions() -> None:
    model = _model()
    corpus = build_foundation_delayed_memory_corpus(
        train_units=8, holdout_units=4, retention_units=4
    )
    for episode in corpus.train:
        DelayedMemoryTask._write_episode(model, episode)
    total = int(model.identity_organ._value_counts.sum().item())
    assert total == len(corpus.train)
    for episode in corpus.train[:4]:
        # replay the write key and check the router voted for its action
        model.reset_dynamics(episode_id=f"r-{episode.memory_id}")
        for symbol in (
            model.config.boundary_symbol,
            *episode.context,
            episode.cue,
        ):
            model.observe(symbol, learn=False, learn_motor=False)
        ctx = model.fabric.cortical_context(model.snapshot().regions)
        slot = model.identity_organ.bank.route(ctx, learn=False).slot_index
        assert slot is not None
        voted = model.identity_organ._nearest_value_action(slot, ctx)
        assert voted == episode.action


def test_punished_write_drops_router_entry() -> None:
    model = _model()
    pattern = torch.randn(model.config.cortical_context_dim)
    pattern = pattern / pattern.norm().clamp_min(1e-12)
    slot = model.identity_organ.bank.route(pattern, learn=True).slot_index
    assert slot is not None
    model.identity_organ._record_value(slot, pattern, 48)
    assert model.identity_organ._value_counts[slot].item() == 1
    model.identity_organ._remove_value(slot, pattern)
    assert model.identity_organ._value_counts[slot].item() == 0


def test_router_checkpoint_roundtrip() -> None:
    model = _model()
    corpus = build_foundation_delayed_memory_corpus(
        train_units=8, holdout_units=4, retention_units=4
    )
    for episode in corpus.train:
        DelayedMemoryTask._write_episode(model, episode)
    payload = model.identity_organ.to_payload(parent_checkpoint_digest="m1-66b-parent")
    restored = model.identity_organ
    restored.load_payload(dict(payload))
    assert torch.equal(restored._value_counts, model.identity_organ._value_counts)


def test_router_off_falls_back_to_blended_head() -> None:
    model = _model(router_off=True)
    assert model.identity_organ._value_router_enabled is False
    assert int(model.identity_organ._value_counts.sum().item()) == 0
