from __future__ import annotations

from dataclasses import replace

import torch

from scripts.training.eval_taiji_m2r3_r0_structured_semantics import (
    build_corpus,
    evaluate,
)
from taiji import StructuredSemanticLearner, semantic_input_digest


def test_m2r3_structured_semantic_canary_passes() -> None:
    report = evaluate(epochs=240, learning_rate=2.0)

    assert report["format"] == "taiji-m2r3-r0-structured-semantics-v1"
    assert report["status"] == "passed"
    assert report["can_promote"] is False
    assert report["gate"]["passed"] is True
    assert report["metrics"]["dev"]["fact_f1"] == 1.0
    assert report["metrics"]["test"]["content_accuracy"] == 1.0
    assert report["metrics"]["unknown_status"] == "unknown"
    assert report["metrics"]["conflict_status"] == "conflict"
    assert report["metrics"]["clarification_status"] == "clarify"


def test_semantic_input_digest_ignores_transport_identity() -> None:
    example = build_corpus().train[0]
    changed = replace(
        example.percept,
        event_id="transport-renamed",
        assembly_id="transport-assembly-renamed",
        observation_tick=999,
    )

    assert semantic_input_digest(example.percept) == semantic_input_digest(changed)


def test_structured_semantic_checkpoint_restores_native_outputs() -> None:
    corpus = build_corpus()
    learner = StructuredSemanticLearner(corpus)
    learner.fit(corpus.train, epochs=160, learning_rate=2.0)
    restored = StructuredSemanticLearner.from_checkpoint(learner.checkpoint(), corpus)

    for example in corpus.test:
        left = learner.predict(example.percept)
        right = restored.predict(example.percept)
        assert left.status == right.status
        assert left.goal == right.goal
        assert left.content_plan == right.content_plan
        assert left.world is not None and right.world is not None
        assert torch.equal(left.world.latent, right.world.latent)
        assert left.world.relations == right.world.relations
