"""Internal address teacher detachment, credit, loss recipe and full restore."""

import pytest
import torch

from scripts.training import run_taiji_n1_address_consistency as runner
from taiji.native_address_consistency_learning import address_consistency_loss


def test_frozen_cache_loss_credit_and_restore(tmp_path):
    torch.set_num_threads(1)
    train = runner.checked_train()
    model, optimizer = runner.fresh(train)
    record, _, _ = runner.transfer_source()
    before = runner.paired.fingerprint(model.bundle(optimizer))
    cache = runner.build_cache(model, train[:1], record, runner.paired.CpuBudget(300))
    assert before == runner.paired.fingerprint(model.bundle(optimizer))
    assert all(not cache[train[0]["id"]][k].requires_grad for k in ("logits", "addresses"))
    curriculum = runner.checked_curriculum(train)
    item = runner.renaming.rename_item(train[0], curriculum["stream"][0]["names"])
    runner.paired.observe(model, runner.paired.project(item), "loss-check")
    reference = cache[item["id"]]
    losses = address_consistency_loss(
        model,
        "loss-check",
        [(item["question"].encode(), item["answer"].encode())],
        [reference],
        preserve_logits=False,
    )
    assert losses["preservation"] == 0 and losses["address"] > 0
    assert torch.allclose(
        losses["total"], losses["qa"] + 0.25 * losses["observation"] + losses["address"]
    )
    losses["address"].backward()
    for parameter in (model.query.weight, model.writer.weight_ih, model.question_fusion.weight):
        assert (
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and parameter.grad.abs().sum() > 0
        )
    assert reference["addresses"].grad is None
    with pytest.raises(ValueError, match="detached"):
        address_consistency_loss(
            model,
            "loss-check",
            [(item["question"].encode(), item["answer"].encode())],
            [dict(reference, logits=reference["logits"].clone().requires_grad_())],
            preserve_logits=True,
        )
    model._observations.pop("loss-check")
    optimizer.zero_grad(set_to_none=True)
    saved = runner.save(model, optimizer, tmp_path / "zero.pt", 0)
    assert runner.restore(tmp_path / "zero.pt")["signature"] == saved["expected_signature"]
    assert all(int(v["step"]) == 6144 for v in optimizer.state.values())


def test_original_preservation_identity_and_train_only():
    train = runner.checked_train()
    model, optimizer = runner.fresh(train)
    record, _, _ = runner.transfer_source()
    cache = runner.build_cache(model, train[:1], record, runner.paired.CpuBudget(300))
    item = train[0]
    runner.paired.observe(model, runner.paired.project(item), "original-loss")
    losses = address_consistency_loss(
        model,
        "original-loss",
        [(item["question"].encode(), item["answer"].encode())],
        [cache[item["id"]]],
        preserve_logits=True,
    )
    assert abs(float(losses["address"].detach())) < 1e-6
    assert abs(float(losses["preservation"].detach())) < 1e-6
    with pytest.raises(ValueError, match="train"):
        runner.build_cache(model, [dict(item, split="dev")], record, runner.paired.CpuBudget(300))
    with pytest.raises(ValueError, match="train"):
        runner.train_step(model, optimizer, dict(item, split="holdout"), "bad", cache, True)
