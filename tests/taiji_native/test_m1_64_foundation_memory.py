from __future__ import annotations

from scripts.training.eval_taiji_m1_64_foundation_memory import (
    ACTION_SYMBOLS,
    _marginal_predictability,
    build_foundation_delayed_memory_corpus,
)


def test_foundation_corpus_keys_are_mutually_distinct_beyond_the_alphabet() -> None:
    corpus = build_foundation_delayed_memory_corpus(
        train_units=1000, holdout_units=200, retention_units=200
    )

    assert corpus.sample_counts == {"train": 1000, "holdout": 200, "retention": 200}
    assert len({episode.recall_key for episode in corpus.train}) == 1000
    assert len({episode.cue for episode in corpus.train}) < 1000


def test_foundation_corpus_declares_a_real_delay() -> None:
    corpus = build_foundation_delayed_memory_corpus(
        train_units=80, holdout_units=20, retention_units=20
    )

    assert len(corpus.interference_symbols) > 0


def test_marginal_guard_clears_an_unbiased_course() -> None:
    corpus = build_foundation_delayed_memory_corpus(
        train_units=1000, holdout_units=200, retention_units=200
    )

    marginals = _marginal_predictability(corpus)

    assert marginals["tail_only_exceeds_null"] is False
    assert marginals["context_only_exceeds_null"] is False


def test_marginal_guard_catches_a_tail_only_shortcut() -> None:
    corpus = build_foundation_delayed_memory_corpus(
        train_units=1000, holdout_units=200, retention_units=200
    )
    leaked = tuple(
        type(episode)(
            memory_id=episode.memory_id,
            cue=episode.cue,
            action=ACTION_SYMBOLS[episode.cue % len(ACTION_SYMBOLS)],
            outcome=episode.outcome,
            context=episode.context,
        )
        for episode in corpus.train
    )

    marginals = _marginal_predictability(
        type(corpus)(
            train=leaked,
            holdout=corpus.holdout,
            retention=corpus.retention,
            interference_symbols=corpus.interference_symbols,
        )
    )

    assert marginals["tail_only"] == 1.0
    assert marginals["tail_only_exceeds_null"] is True
