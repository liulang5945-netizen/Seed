"""R0.3: scoring must fully restore model state, RNG and learning counters.

The digest equality already checked by foundation training is not enough:
scoring must also leave the shared RNGs, dynamics tick, development ledger and
the active-readout registry untouched, including on an exceptional path.
"""

from __future__ import annotations

import pytest

from taiji import Taiji, TaijiConfig, WorkbenchBoundaryAuthorization, WorkbenchTaskBoundary
from taiji.internalization import content_digest


def _boundary(*, task_id: str = "task:alpha") -> WorkbenchTaskBoundary:
    return WorkbenchTaskBoundary.issue(
        project_id="project:seed",
        task_id=task_id,
        session_id="session:one",
        language_id="python",
        capability_snapshot_id="capability:snapshot:1",
        capability_ids=("workspace.read",),
        generation_scope="protected" if task_id == "task:alpha" else "active",
        issued_tick=10,
        ttl_ticks=20,
    )


def _authorization(
    boundary: WorkbenchTaskBoundary,
    *,
    current_tick: int = 12,
    usage: str = "execute",
) -> WorkbenchBoundaryAuthorization:
    return WorkbenchBoundaryAuthorization(
        project_id=boundary.project_id,
        task_id=boundary.task_id,
        session_id=boundary.session_id,
        capability_snapshot_id=boundary.capability_snapshot_id,
        authorized_capability_ids=("workspace.read",),
        active_boundary_digest=boundary.token_digest,
        current_tick=current_tick,
        usage=usage,
    )


def _state_snapshot(model: Taiji) -> dict[str, object]:
    return {
        "checkpoint_digest": content_digest(model.checkpoint()),
        "rng_state": model._rng.get_state().clone().cpu().numpy().tobytes(),
        "memory_rng_state": model._memory_rng.get_state().clone().cpu().numpy().tobytes(),
        "tick": model.tick,
        "development_ticks": model._development_ticks,
        "memory_write_count": model.memory.write_count,
        "active_readout_digest": (
            None
            if model.active_predictive_readout_metadata is None
            else model.active_predictive_readout_metadata["readout_digest"]
        ),
    }


def test_scoring_restores_full_model_state_after_protected_and_active_scores() -> None:
    model = Taiji(TaijiConfig(seed=41), episode_id="score-restore")
    protected = _boundary()
    active = protected.successor(
        task_id="task:beta",
        generation_scope="active",
        issued_tick=12,
        ttl_ticks=20,
    )
    active_context = _authorization(active)
    data = b"abcdabcdabcd"

    model.learn_bytes(data, epochs=3)
    model.clone_protected_predictive_readout_as_active(
        boundary_digest=active.token_digest,
    )
    model.learn_bytes(
        data,
        epochs=2,
        learn_fabric=False,
        learn_predictive_context=False,
        boundary=active,
        authorization=active_context,
    )
    before = _state_snapshot(model)
    assert before["tick"] > 0

    model.score_bytes(data)
    model.score_bytes(
        data,
        boundary=protected,
        authorization=_authorization(protected, current_tick=10),
    )
    model.score_bytes(
        data,
        boundary=active,
        authorization=active_context,
        use_memory=True,
    )

    assert _state_snapshot(model) == before


def test_scoring_restores_full_model_state_on_exception() -> None:
    model = Taiji(TaijiConfig(seed=43), episode_id="score-restore-exception")
    data = b"abcdabcdabcd"
    model.learn_bytes(data, epochs=2)
    before = _state_snapshot(model)

    real_observe = model.observe
    calls = {"count": 0}

    def exploding_observe(*args: object, **kwargs: object) -> object:
        calls["count"] += 1
        if calls["count"] == 3:
            raise ValueError("synthetic scoring failure")
        return real_observe(*args, **kwargs)

    model.observe = exploding_observe  # type: ignore[method-assign]
    try:
        with pytest.raises(ValueError, match="synthetic scoring failure"):
            model.score_bytes(data)
    finally:
        model.observe = real_observe  # type: ignore[method-assign]

    assert _state_snapshot(model) == before


def test_scoring_records_real_owner_calls_not_empty_round_trip() -> None:
    model = Taiji(TaijiConfig(seed=47), episode_id="score-owner-calls")
    protected = _boundary()
    active = protected.successor(
        task_id="task:beta",
        generation_scope="active",
        issued_tick=12,
        ttl_ticks=20,
    )
    data = b"abcdabcdabcd"
    model.learn_bytes(data, epochs=1)
    model.clone_protected_predictive_readout_as_active(
        boundary_digest=active.token_digest,
    )

    expected_symbols = len(list(model.sensor.symbols(data, include_boundary=True)))
    # The first scored symbol carries no prior prediction, so the observations
    # gauge records one fewer real owner calls than the symbol count.
    expected_observations = expected_symbols - 1
    real_observe = model.observe
    calls = {"count": 0}

    def counting_observe(*args: object, **kwargs: object) -> object:
        calls["count"] += 1
        return real_observe(*args, **kwargs)

    model.observe = counting_observe  # type: ignore[method-assign]
    try:
        scored = model.score_bytes(
            data,
            boundary=active,
            authorization=_authorization(active),
        )
    finally:
        model.observe = real_observe  # type: ignore[method-assign]

    # The observations counter is the real owner-call gauge: it must equal the
    # number of scored symbols, proving the active owner was actually invoked
    # rather than measured through an empty `holdout_updates=0` round trip.
    assert calls["count"] == expected_symbols
    assert scored["observations"] == expected_observations
    assert scored["scope"] == "active"
    assert scored["owner"] == "predictive_readout.active"
