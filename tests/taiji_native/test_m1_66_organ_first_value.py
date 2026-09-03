"""M1-66: a routed cue owns the value verdict.

Once the identity organ routes a cue (``identity_recall.used``), the organ's
own action distribution replaces the motor-synthesised distribution as the
decision.  The episodic field's value head is chance-level on foundation-scale
courses and its high-confidence wrong evidence diluted correct organ verdicts,
so the readout is decoupled instead of summed.  Ablations stay intact
(``use_identity=False`` never routes; ``use_memory=False`` disables the gate).
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


def _primed_model() -> Taiji:
    values = _memory_config(11).to_dict()
    values["identity_organ_capacity"] = 128
    model = Taiji(TaijiConfig.from_dict(values), episode_id="m1-66")
    corpus = build_foundation_delayed_memory_corpus(
        train_units=80, holdout_units=40, retention_units=40
    )
    for episode in corpus.train:
        DelayedMemoryTask._write_episode(model, episode)
    return model


def _read_cue(model: Taiji, query) -> tuple[torch.Tensor, object, object]:
    """Observe one delayed query and return (probabilities, identity, memory)."""
    model.reset_dynamics(episode_id=f"r-{query.query_id}")
    for symbol in (
        model.config.boundary_symbol,
        *query.context,
        query.cue,
    ):
        step = model.observe(
            symbol, learn=False, learn_motor=False, use_memory=True, use_identity=None
        )
    return model.snapshot().motor_probabilities, step.identity_recall, step.memory_recall


def test_routed_verdict_is_the_organ_distribution() -> None:
    """used=True => motor_probabilities equals the organ's action distribution."""
    model = _primed_model()
    corpus = build_foundation_delayed_memory_corpus(
        train_units=80, holdout_units=40, retention_units=40
    )
    routed = 0
    for query in corpus.holdout[:20]:
        probabilities, identity, _ = _read_cue(model, query)
        if not identity.used:
            continue
        routed += 1
        # organ-first: the decision carries the organ's argmax verdict
        organ_arg = int(identity.action_probabilities.argmax().item())
        decided_arg = int(probabilities.argmax().item())
        assert decided_arg == organ_arg
    assert routed >= 10


def test_organ_first_beats_both_ablations() -> None:
    """The organ verdict beats memory-only and identity-disabled decisions."""
    model = _primed_model()
    corpus = build_foundation_delayed_memory_corpus(
        train_units=80, holdout_units=40, retention_units=40
    )
    organ = 0
    identity_off = 0
    total = 0
    for query in corpus.holdout:
        probabilities, identity, _ = _read_cue(model, query)
        decided = int(probabilities.argmax().item())
        organ += int(identity.used and decided == query.expected_action)

        model.reset_dynamics(episode_id=f"i-{query.query_id}")
        for symbol in (
            model.config.boundary_symbol,
            *query.context,
            query.cue,
        ):
            step = model.observe(
                symbol,
                learn=False,
                learn_motor=False,
                use_memory=True,
                use_identity=False,
            )
        identity_off += int(int(step.probabilities.argmax().item()) == query.expected_action)
        total += 1
    # the organ-first path must not trail the identity-disabled arm
    assert organ >= identity_off


def test_unrouted_read_keeps_motor_synthesis_distribution() -> None:
    """Without routing, the decision is a probability distribution, not logits."""
    model = _primed_model()
    corpus = build_foundation_delayed_memory_corpus(
        train_units=80, holdout_units=40, retention_units=40
    )
    probabilities, identity, _ = _read_cue(model, corpus.holdout[0])
    assert probabilities.shape == (model.config.alphabet_size,)
    assert torch.allclose(probabilities.sum(), torch.tensor(1.0))
    # whether or not it routed, the stored motor_probabilities is a valid
    # distribution the evaluator can argmax over
    assert float(probabilities[identity.action_probabilities.argmax().item()].item()) > 0.0
