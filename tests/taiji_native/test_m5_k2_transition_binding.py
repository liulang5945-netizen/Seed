from __future__ import annotations

import copy

from scripts.training.eval_taiji_m2r3_r3_multistep import build_corpus
from taiji.semantic_transition import StructuredSemanticTransitionLearner


def _masks(corpus) -> dict[str, tuple[int, ...]]:
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


def _violations(learner: StructuredSemanticTransitionLearner) -> list[tuple[int, int]]:
    masks = learner.transition_input_masks
    assert masks is not None
    weight = learner.transition_head.weight.detach().cpu()
    return [
        (row, column)
        for row, key in enumerate(learner.fact_keys)
        for column, value in enumerate(weight[row])
        if column not in set(masks[key]) and abs(float(value)) > 1e-9
    ]


def test_k2_transition_mask_survives_fit_checkpoint_and_tampering() -> None:
    corpus = build_corpus(seed=7)
    masks = _masks(corpus)
    learner = StructuredSemanticTransitionLearner(
        corpus,
        transition_input_masks=masks,
    )
    learner.fit(corpus.train, epochs=2, learning_rate=0.2)
    assert _violations(learner) == []

    payload = learner.checkpoint()
    restored = StructuredSemanticTransitionLearner.from_checkpoint(payload, corpus)
    assert restored.transition_input_masks == learner.transition_input_masks
    assert restored.owner_digests() == learner.owner_digests()

    tampered = copy.deepcopy(payload)
    weight = tampered["state_dict"]["transition_head.weight"]
    for row, key in enumerate(learner.fact_keys):
        for column in range(weight.shape[1]):
            if column not in set(masks[key]):
                weight[row, column] = 9.0
    hardened = StructuredSemanticTransitionLearner.from_checkpoint(tampered, corpus)
    assert _violations(hardened) == []
