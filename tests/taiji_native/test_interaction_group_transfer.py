from __future__ import annotations

from dataclasses import replace

import pytest

from scripts.training.eval_taiji_interaction_group_transfer import (
    MEMBER_IDS,
    _evaluator,
    build_corpus,
)
from taiji import (
    InteractionGroupTransferLearner,
    build_member_evidence,
)


def test_interaction_group_transfer_gate() -> None:
    from scripts.training.eval_taiji_interaction_group_transfer import evaluate

    report = evaluate()

    assert report["format"] == "taiji-w7-p4-6-interaction-group-transfer-v1"
    assert report["gate"]["passed"] is True
    assert all(report["metrics"].values())


def test_transfer_rejects_unknown_member_without_singleton_evidence() -> None:
    corpus, _ = build_corpus()
    revision = next(iter(corpus.train_checkpoint_revisions))
    learner = InteractionGroupTransferLearner(maximum_uncertainty=2.0)
    learner.observe_members(
        build_member_evidence(
            corpus.train,
            source_trace_digest=corpus.train_trace_digest,
            checkpoint_revision=revision,
        )
    )
    learner.observe_records(_evaluator().train_only_candidates(corpus))

    known = MEMBER_IDS["a"]
    assert learner.candidate((known, "workbench-owner:unknown")) is None


def test_transfer_rejects_holdout_derived_record() -> None:
    corpus, _ = build_corpus()
    revision = next(iter(corpus.train_checkpoint_revisions))
    learner = InteractionGroupTransferLearner(maximum_uncertainty=2.0)
    learner.observe_members(
        build_member_evidence(
            corpus.train,
            source_trace_digest=corpus.train_trace_digest,
            checkpoint_revision=revision,
        )
    )
    record = _evaluator().train_only_candidates(corpus)[0]

    with pytest.raises(ValueError, match="holdout-derived"):
        learner.observe_records((replace(record, holdout_interaction=0.5),))
