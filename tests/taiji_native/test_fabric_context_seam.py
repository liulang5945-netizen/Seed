"""M1-65: the addressing key freezes on a routed cue; prediction stays live.

A successfully routed cue ``stamp_cue_snapshot`` freezes the cortical context
as the addressing key, so symbols observed afterwards (interference) cannot
wash it away -- the identity organ and the episodic field keep routing on the
cue they bound.  The motor's prediction input is the *live* two-segment context
and never sees the snapshot, so B1 stays bit-identical.  Probe-level acceptance
lives in ``test_m1_65_addressing_survives_interference.py``; this module tests
the seam between the frozen addressing context and the live prediction context.
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

SEAM_SEED = 11


def _primed_model() -> Taiji:
    values = _memory_config(SEAM_SEED).to_dict()
    model = Taiji(TaijiConfig.from_dict(values), episode_id="m1-65-seam")
    corpus = build_foundation_delayed_memory_corpus(
        train_units=8, holdout_units=4, retention_units=4
    )
    for episode in corpus.train:
        DelayedMemoryTask._write_episode(model, episode)
    return model


def _activity_and_trace(model: Taiji) -> tuple[torch.Tensor, torch.Tensor]:
    activity = torch.cat(tuple(region.activity for region in model.snapshot().regions))
    trace = torch.cat(tuple(region.trace for region in model.snapshot().regions))
    return activity, trace


def test_predictive_context_is_always_the_live_two_segment_vector() -> None:
    """The motor's byte-prediction input stays the live activity+trace."""
    model = _primed_model()
    activity, trace = _activity_and_trace(model)
    predictive = model.fabric.predictive_context(model.snapshot().regions)
    assert torch.equal(predictive, torch.cat((activity, trace)))


def test_addressing_key_freeze() -> None:
    """Interference does not move the addressing key once a cue is routed."""
    model = _primed_model()
    corpus = build_foundation_delayed_memory_corpus(
        train_units=8, holdout_units=4, retention_units=4
    )
    query = corpus.holdout[0]
    model.reset_dynamics(episode_id="q")
    for symbol in (
        model.config.boundary_symbol,
        *query.context,
        query.cue,
    ):
        model.observe(symbol, learn=False, learn_motor=False)
    key_after_cue = model.fabric.addressing_key()
    assert key_after_cue is not None
    for symbol in corpus.interference_symbols:
        model.observe(symbol, learn=False, learn_motor=False)
    assert torch.equal(model.fabric.addressing_key(), key_after_cue)


def test_reset_clears_the_frozen_key() -> None:
    """One episode cannot leak its frozen cue into the next."""
    model = _primed_model()
    corpus = build_foundation_delayed_memory_corpus(
        train_units=8, holdout_units=4, retention_units=4
    )
    model.reset_dynamics(episode_id="q")
    for symbol in (model.config.boundary_symbol, *corpus.holdout[0].context, corpus.holdout[0].cue):
        model.observe(symbol, learn=False, learn_motor=False)
    assert model.fabric.addressing_key() is not None
    model.reset_dynamics(episode_id="next")
    assert model.fabric.addressing_key() is None


def test_snapshot_key_equals_written_context() -> None:
    """The frozen key equals the cortical context the write side would see.

    The write/read cosine acceptance probe measures exactly this equality: the
    addressing key a delayed read routes on is the same vector the organ bound
    at write time, not a live state disturbed by interference.
    """
    from tests.taiji_native.test_m1_65_addressing_survives_interference import (  # noqa: PLC0415
        _primed_cosines,
    )

    corpus = build_foundation_delayed_memory_corpus(
        train_units=40, holdout_units=20, retention_units=20
    )
    cosines, threshold = _primed_cosines(interference=corpus.interference_symbols)
    assert min(cosines) >= float(threshold)
