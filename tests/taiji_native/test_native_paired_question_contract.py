"""Pair semantics, loss weights and zero-update restore contract."""

import pytest
import torch

from scripts.training import run_taiji_n1_paired_question as runner
from taiji.native_evidence_fidelity import learning_loss
from taiji.native_paired_question_learning import paired_question_loss


def test_pair_semantics_and_scope():
    train = runner.checked_train()
    pairs = runner.checked_pairs(train)
    originals = {x["id"]: x for x in train}
    assert len(pairs) == 96
    for pair in pairs:
        original = originals[pair["parent_id"]]
        assert original["observations"] == pair["observations"]
        assert original["question"] != pair["question"]
        assert original["answer"] != pair["answer"]
        entity = pair["question"].split()[-1].removesuffix("?")
        assert (
            next(
                o.split()[-1].removesuffix(".")
                for o in original["observations"]
                if o.startswith(entity + " ")
            )
            == pair["answer"]
        )
        assert runner.paired.project(pair).question == pair["question"].encode()
    with pytest.raises(ValueError, match="train"):
        runner.build_pairs([dict(train[0], split="holdout")])


def test_two_questions_half_each_auxiliary_once_and_cut():
    torch.set_num_threads(1)
    train = runner.checked_train()
    model, optimizer = runner.fresh(train)
    item = train[0]
    pair = runner.checked_pairs(train)[0]
    before = runner.paired.fingerprint(model.bundle(optimizer))
    q, a, q2, a2 = [
        x.encode() for x in (item["question"], item["answer"], pair["question"], pair["answer"])
    ]
    first = learning_loss(model, "zero-probe", q, a)
    second = learning_loss(model, "zero-probe", q2, a2)
    loss = paired_question_loss(model, "zero-probe", q, a, second_question=q2, second_target=a2)
    assert torch.allclose(loss.qa, (first.qa + second.qa) / 2)
    assert torch.equal(loss.observation, first.observation)
    assert torch.equal(loss.total, loss.qa + 0.25 * first.observation)
    cut = paired_question_loss(
        model, "zero-probe", q, a, second_question=q2, second_target=a2, arm="cut"
    )
    assert torch.equal(loss.total, cut.total)
    cut.total.backward()
    assert all(p.grad is None for p in model.writer.parameters())
    assert before == runner.paired.fingerprint(model.bundle(optimizer))
    with pytest.raises(ValueError, match="complete optional"):
        paired_question_loss(model, "zero-probe", q, a, second_question=q2)
    with pytest.raises(ValueError, match="paired question required"):
        runner.train_step(model, optimizer, item, "no-pair")


def test_recipe_and_zero_restore(tmp_path):
    assert runner.PROTOCOL["updates"] == 9216
    assert runner.PROTOCOL["qa_sequences"] == 13824
    assert runner.PROTOCOL["pair_gate"] == 87
    assert runner.PROTOCOL["milestones"] == [3072, 6144, 9216]
    model, optimizer = runner.fresh(runner.checked_train())
    assert runner.baseline_initial_match(model)["passed"]
    saved = runner.save(model, optimizer, tmp_path / "zero.pt", 0)
    assert runner.restore(tmp_path / "zero.pt")["signature"] == saved["expected_signature"]
