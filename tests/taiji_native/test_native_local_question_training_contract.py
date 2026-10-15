"""PLAN49 initialization and exact optimizer accounting on synthetic byte input."""

import pytest

from scripts.training import run_taiji_n1_local_question as runner


def test_engineering_zero_identity_and_shared_new_adam_accounting(tmp_path):
    runner.torch.set_num_threads(1)
    model, optimizer = runner.fresh([])
    assert runner.baseline_initial_match(model, optimizer)["passed"]
    saved = runner.save(model, optimizer, tmp_path / "zero.pt", 0)
    assert runner.restore(tmp_path / "zero.pt")["signature"] == saved["expected_signature"]
    item = dict(
        id="synthetic-contract",
        kind="revision",
        split="train",
        observations=["alpha uses quartz.", "bravo uses copper."],
        question="material of alpha?",
        answer="quartz",
    )
    for step in (1, 2):
        loss = runner.train_step(model, optimizer, item, f"local49-contract-{step}")
        assert loss["qa_sequences"] == 1
        for name, parameter in model.named_parameters():
            expected = step if name in runner.transfer.NEW_PARAMETERS else 6144 + step
            assert int(optimizer.state[parameter]["step"]) == expected
    saved = runner.save(model, optimizer, tmp_path / "synthetic.pt", 2)
    assert runner.restore(tmp_path / "synthetic.pt")["signature"] == saved["expected_signature"]
    with pytest.raises(ValueError, match="train"):
        runner.train_step(model, optimizer, dict(item, split="dev"), "forbidden")


def test_frozen_recipe_and_data_scope():
    assert runner.PROTOCOL["parameter_count"] == 47480
    assert runner.PROTOCOL["updates"] == 6144
    assert runner.PROTOCOL["new_parameter_final_adam_step"] == 6144
    assert runner.PROTOCOL["shared_parameter_final_adam_step"] == 12288
    assert "address_loss_contract" not in runner.PROTOCOL
    train = runner.checked_train()
    assert len(train) == 192 and len(runner.checked_pairs(train)) == 96
    assert runner.checked_curriculum(train)["updates"] == 6144
