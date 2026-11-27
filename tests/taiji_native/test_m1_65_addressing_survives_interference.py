"""M1-65 acceptance probe: cue addressing must survive interference.

M1-64's falsification probes pinned the B2 gap to addressing, not capacity or
course difficulty: the organ keys on the fabric's instantaneous cortical
activity, and one interference symbol collapses the write/read cosine from
~0.99 to ~0.098 against the ``identity_organ_match_threshold`` of 0.9.  A read
then routes to ``slot_index=None`` and decays to ``IDENTITY_ORGAN_UNBOUND``,
making the read path mathematically identical to absent.

This test is the standalone, accuracy-independent gate the M1-65 mechanism must
turn green.  It asserts, on the production observation paths, that the cosine
between the write basis (the cortical context a settled episode bound) and the
read basis (the cortical context after at least one interference symbol) stays
at or above the organ's own match threshold.  Written red-first: today it
measures ~0.04, far below 0.9.

Control discipline: the memory field is fully primed before any basis is
captured -- the write is replayed inside the same late-stage memory state the
read will run in -- so `interference` is the *only* variable between the two
arms.  A naive probe that captures the write basis early and reads late mixes
in the memory-feedback drift from engram accumulation (measured min 0.8964 on
an otherwise clean read), which would make the control arm red for the wrong
reason.
"""

from __future__ import annotations

import torch

from scripts.training.eval_taiji_foundation_baseline import _memory_config
from scripts.training.eval_taiji_m1_64_foundation_memory import (
    build_foundation_delayed_memory_corpus,
)
from taiji import Taiji
from taiji.foundation_tasks import (
    DelayedMemoryQuery,
    DelayedMemoryTask,
    MemoryEpisode,
)

PROBE_SEED = 11


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a @ b) / (a.norm() * b.norm().clamp_min(1e-12)))


def _capture_write_basis(model: Taiji, episode: MemoryEpisode) -> torch.Tensor:
    """Replay a write and return the cortical context the organ binds.

    Mirrors ``DelayedMemoryTask._write_episode`` step for step.  The basis is
    captured at the same moment the organ learns -- after ``settle_action``,
    before ``observe(outcome)`` -- so it is exactly the key the identity organ
    stored, not a reconstruction.  The write is idempotent for the measured
    fabric activity: every observation passes ``learn=False``, so replaying a
    train episode merely re-captures the basis inside the current memory state.
    """

    model.reset_dynamics(episode_id=f"m1-65-train-{episode.memory_id}")
    model.observe(model.config.boundary_symbol, learn=False, learn_motor=False)
    for symbol in episode.context:
        model.observe(symbol, learn=False, learn_motor=False)
    model.observe(episode.cue, learn=False, learn_motor=False)
    model.act((episode.action,), sample=False)
    model.settle_action(
        1.0,
        learn=False,
        learn_memory=True,
        provenance="experienced",
    )
    basis = model.fabric.cortical_context(model.snapshot().regions).detach().clone()
    model.observe(episode.outcome, learn=False, learn_motor=False)
    return basis


def _capture_read_basis(
    model: Taiji,
    query: DelayedMemoryQuery,
    *,
    interference: tuple[int, ...],
) -> torch.Tensor:
    """Return the cortical context routed during a delayed query read.

    The observation sequence mirrors ``DelayedMemoryTask._recall_accuracy``:
    boundary, context, cue, then each interference symbol.  The basis is the
    activity field the final observation leaves behind -- the very field the
    identity organ routes against on a real delayed read.
    """

    model.reset_dynamics(episode_id=f"m1-65-query-{query.query_id}")
    for symbol in (
        model.config.boundary_symbol,
        *query.context,
        query.cue,
        *interference,
    ):
        model.observe(symbol, learn=False, learn_motor=False)
    return model.fabric.cortical_context(model.snapshot().regions).detach().clone()


def _primed_cosines(
    *,
    interference: tuple[int, ...],
) -> tuple[list[float], float]:
    """Write/read cosine pairs with the memory field primed on both sides.

    All train episodes are written first (the memory field thus holds every
    engram, exactly like a real holdout read).  Each probe row then replays its
    own train write to capture the *late-stage* write basis, and reads its
    holdout query -- so write and read share one memory state and only
    ``interference`` differs between the clean and interfered arms.
    """

    corpus = build_foundation_delayed_memory_corpus(
        train_units=40, holdout_units=20, retention_units=20
    )
    model = Taiji(_memory_config(PROBE_SEED), episode_id="m1-65-probe")
    for episode in corpus.train:
        DelayedMemoryTask._write_episode(model, episode)

    cosines: list[float] = []
    for index, query in enumerate(corpus.holdout):
        basis = _capture_write_basis(model, corpus.train[index])
        read_basis = _capture_read_basis(model, query, interference=interference)
        cosines.append(_cosine(basis, read_basis))
    threshold = float(model.config.identity_organ_match_threshold)
    return cosines, threshold


def test_clean_read_basis_cosine_meets_match_threshold() -> None:
    """Without interference the read finds the write; the probe is sensitive.

    This is the control arm.  With the memory state held constant across write
    and read, a read of the exact cue sequence routes back to the write basis
    with a cosine near 1.0.  If this ever fails, the probe itself is broken (a
    read cannot route to a key it shares almost no cosine with) and the
    interference arm below is uninterpretable.
    """

    cosines, threshold = _primed_cosines(interference=())
    assert min(cosines) >= threshold


def test_interfered_read_basis_cosine_meets_match_threshold() -> None:
    """After at least one interference symbol the write/read cosine must hold.

    M1-65's fix must push ``cos(write_basis, read_basis_with_interference)``
    from the measured ~0.04 back above ``identity_organ_match_threshold``.
    Today this is red: a single interference symbol already collapses the
    cosine below 0.9 on every row.
    """

    corpus = build_foundation_delayed_memory_corpus(
        train_units=40, holdout_units=20, retention_units=20
    )
    assert len(corpus.interference_symbols) >= 1
    cosines, threshold = _primed_cosines(interference=corpus.interference_symbols)
    assert min(cosines) >= threshold
