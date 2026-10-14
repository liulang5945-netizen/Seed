from __future__ import annotations

import pytest
import torch

from taiji import InputFrame, TaijiConfig, TSKV8Adapter
from taiji.internalization import content_digest
from taiji.stepwise_consumer import StepwiseConsumerRequest, select_and_prepare


def _model() -> TSKV8Adapter:
    return TSKV8Adapter(
        TaijiConfig(region_sizes=(24, 16), synapse_fan_in=6, motor_fan_in=8, seed=41)
    )


def _support(model: TSKV8Adapter) -> tuple[str, str]:
    model.persistent_stepwise_carrier = True
    for index, length in enumerate((3, 5)):
        frame = InputFrame(
            input_id=f"support:{index}",
            modality="text",
            payload=b"fact",
            source="test",
            timestamp=index,
            provenance="external",
            confidence=1.0,
        )
        model.generate_input(frame, length, reset=True, stop_at_boundary=False)
    model.persistent_stepwise_carrier = False
    return model._persistent_stepwise_carrier.episode_ids


def test_no_replay_is_bitwise_identical_and_never_checkpointed() -> None:
    default, explicit = _model(), _model()
    explicit.configure_stepwise_consumer(StepwiseConsumerRequest("no-replay"))
    assert default.generate(b"question", 9) == explicit.generate(b"question", 9)
    assert content_digest(default.checkpoint()) == content_digest(explicit.checkpoint())
    audit = explicit.last_stepwise_consumer_audit
    assert audit["replay_steps"] == 0
    assert audit["first_probabilities_before"] == audit["first_probabilities_after"]
    explicit.restore(explicit.checkpoint())
    assert explicit._stepwise_consumer_request is None
    assert explicit.last_stepwise_consumer_audit is None


def test_selector_uses_content_for_ties_and_is_invariant_to_ids_and_order() -> None:
    candidates = {"old": torch.tensor([[1.0, 0.0], [1.0, 0.0]]), "new": torch.tensor([[0.0, 1.0]])}
    query = torch.tensor([1.0, 1.0])
    first, a = select_and_prepare(
        StepwiseConsumerRequest("model-top1", ("old", "new")), query, candidates
    )
    second, b = select_and_prepare(
        StepwiseConsumerRequest("model-top1", ("z", "a")),
        query,
        {"z": candidates["new"], "a": candidates["old"]},
    )
    assert torch.equal(first, second)
    assert a["ranking"][0]["vectors_sha256"] == b["ranking"][0]["vectors_sha256"]
    assert a["ranking"][0]["score"] == a["ranking"][1]["score"]


def test_controls_preserve_dose_and_ordered_replay_is_not_pooled() -> None:
    correct = torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    other = torch.tensor([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [0.0, 4.0]])
    pool = {"correct": correct, "other": other}
    traces = {}
    for mode in (
        "oracle-correct",
        "wrong-episode",
        "shuffle-correct",
        "pooled-correct",
        "model-top1",
        "zero-replay",
    ):
        control = "other" if mode == "wrong-episode" else "correct"
        if mode in {"model-top1", "zero-replay"}:
            control = None
        traces[mode], audit = select_and_prepare(
            StepwiseConsumerRequest(mode, ("correct", "other"), control, 77),
            torch.tensor([1.0, 0.0]),
            pool,
        )
        assert audit["replay_steps"] == 4
        assert traces[mode].shape == (4, 2)
    assert torch.equal(traces["oracle-correct"][:3], correct)
    assert torch.equal(traces["model-top1"], traces["oracle-correct"])
    assert torch.equal(traces["wrong-episode"], other)
    assert torch.equal(traces["pooled-correct"][:3], correct.mean(0).expand_as(correct))
    assert not torch.equal(traces["shuffle-correct"], traces["oracle-correct"])
    assert sorted(traces["shuffle-correct"][:3, 0].tolist()) == [1.0, 2.0, 3.0]
    assert torch.count_nonzero(traces["zero-replay"]) == 0
    assert torch.count_nonzero(traces["oracle-correct"][-1]) == 0


def test_replay_reaches_live_state_and_readout_without_weight_or_carrier_writes(
    monkeypatch,
) -> None:
    model = _model()
    ids = _support(model)
    parameters = [t.clone() for t in model.parameter_tensors()]
    before_manifest = model._persistent_stepwise_carrier.episode_manifest()
    seen = []
    original = model.fabric.step

    def capture(sensory, regions, **kwargs):
        if kwargs.get("adapt_homeostasis") is False:
            assert kwargs["learn"] is False
            assert kwargs["restructure"] is False
            assert torch.count_nonzero(sensory) == 0
            seen.append(kwargs["episodic_feedback"].clone())
        return original(sensory, regions, **kwargs)

    monkeypatch.setattr(model.fabric, "step", capture)
    model.configure_stepwise_consumer(StepwiseConsumerRequest("oracle-correct", ids, ids[0]))
    model.generate(b"question", 7)
    assert len(seen) == 5
    assert torch.equal(torch.stack(seen)[:3], model.read_persistent_stepwise_episode(ids[0]))
    assert model._persistent_stepwise_carrier.episode_manifest() == before_manifest
    assert all(
        torch.equal(a, b) for a, b in zip(parameters, model.parameter_tensors(), strict=True)
    )
    audit = model.last_stepwise_consumer_audit
    assert audit["first_distribution_l1"] > 0
    assert audit["first_probabilities_before"] != audit["first_probabilities_after"]


def test_missing_episode_and_query_recording_fail_closed() -> None:
    model = _model()
    ids = _support(model)
    model.configure_stepwise_consumer(StepwiseConsumerRequest("model-top1", (ids[0], "missing")))
    with pytest.raises(ValueError, match="missing or invalid"):
        model.generate(b"question", 1)
    model.persistent_stepwise_carrier = True
    with pytest.raises(ValueError, match="must not record"):
        model.generate(b"question", 1)


def test_model_selector_cannot_receive_an_oracle_or_duplicate_candidates() -> None:
    with pytest.raises(ValueError, match="oracle"):
        StepwiseConsumerRequest("model-top1", ("x", "y"), "x")
    with pytest.raises(ValueError, match="distinct"):
        StepwiseConsumerRequest("model-top1", ("x", "x"))
    with pytest.raises(ValueError, match="must not receive"):
        StepwiseConsumerRequest("no-replay", ("x", "y"))
