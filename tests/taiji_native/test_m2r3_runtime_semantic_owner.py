from __future__ import annotations

from scripts.training.eval_taiji_m2r3_r0_structured_semantics import build_corpus
from taiji import StructuredSemanticLearner, TSKV8Adapter, content_digest


def _trained_learner() -> tuple[StructuredSemanticLearner, object]:
    corpus = build_corpus()
    learner = StructuredSemanticLearner(corpus)
    learner.fit(corpus.train, epochs=160, learning_rate=2.0)
    return learner, corpus


def test_adapter_owns_structured_semantic_snapshot_and_checkpoint() -> None:
    learner, corpus = _trained_learner()
    adapter = TSKV8Adapter()
    adapter.attach_structured_semantic_learner(learner)
    before_state = content_digest(adapter.cognitive_snapshot().to_payload())

    result = adapter.infer_structured_semantics(corpus.test[0].percept)
    checkpoint = adapter.native_checkpoint()
    restored = TSKV8Adapter.from_native_checkpoint(checkpoint)

    assert content_digest(adapter.cognitive_snapshot().to_payload()) == before_state
    assert result.content_plan is not None
    assert restored.structured_semantic_learner is not None
    assert restored.last_structured_semantic_result is not None
    assert content_digest(restored.last_structured_semantic_result.to_payload()) == content_digest(
        result.to_payload()
    )
    assert (
        content_digest(restored.infer_structured_semantics(corpus.test[0].percept).to_payload())
        == content_digest(result.to_payload())
    )


def test_structured_semantic_owner_is_optional_and_detachable() -> None:
    learner, corpus = _trained_learner()
    adapter = TSKV8Adapter()
    plain_checkpoint = adapter.native_checkpoint()
    assert "structured_semantic" not in plain_checkpoint["components"]

    adapter.attach_structured_semantic_learner(learner)
    adapter.infer_structured_semantics(corpus.test[0].percept)
    adapter.attach_structured_semantic_learner(None)

    assert adapter.structured_semantic_learner is None
    assert adapter.last_structured_semantic_result is None
    assert "structured_semantic" not in adapter.native_checkpoint()["components"]
