from __future__ import annotations

from scripts.training.eval_taiji_m2r3_r0_structured_semantics import (
    build_corpus as build_semantic_corpus,
)
from scripts.training.eval_taiji_m2r3_r3_multistep import (
    build_corpus as build_transition_corpus,
)
from taiji import (
    StructuredSemanticLearner,
    StructuredSemanticTransitionLearner,
    content_digest,
)
from taiji.k_fixed_large import NativeKFixedLargeEnsemble


def _transition_masks(corpus):
    fact_count = len(corpus.fact_keys)
    context_dim = corpus.event_dim + 4
    return {
        key: (
            row,
            fact_count + (row % corpus.event_dim),
            fact_count + context_dim + row * context_dim + (row % corpus.event_dim),
        )
        for row, key in enumerate(corpus.fact_keys)
    }


def test_fixed_large_ensemble_roundtrips_and_preserves_typed_results() -> None:
    semantic_corpus = build_semantic_corpus()
    semantic = StructuredSemanticLearner(semantic_corpus)
    semantic.fit(semantic_corpus.train, epochs=4, learning_rate=2.0)
    semantic_copy = StructuredSemanticLearner.from_checkpoint(semantic.checkpoint())

    transition_corpus = build_transition_corpus(seed=7)
    transition = StructuredSemanticTransitionLearner(
        transition_corpus,
        transition_input_masks=_transition_masks(transition_corpus),
    )
    transition.fit(transition_corpus.train, epochs=4, learning_rate=0.2)
    transition_copy = StructuredSemanticTransitionLearner.from_checkpoint(transition.checkpoint())

    ensemble = NativeKFixedLargeEnsemble(
        (semantic, semantic_copy),
        (transition, transition_copy),
    )
    restored = NativeKFixedLargeEnsemble.from_checkpoint(ensemble.checkpoint())

    semantic_example = semantic_corpus.test[0]
    assert content_digest(restored.predict_semantic(semantic_example.percept).to_payload()) == (
        content_digest(ensemble.predict_semantic(semantic_example.percept).to_payload())
    )
    transition_example = transition_corpus.test[0]
    assert content_digest(
        restored.predict_transition(
            transition_example.before, transition_example.event
        ).to_payload()
    ) == content_digest(
        ensemble.predict_transition(
            transition_example.before, transition_example.event
        ).to_payload()
    )
    assert restored.ensemble_width == 2
    assert restored.owner_digests == ensemble.owner_digests
