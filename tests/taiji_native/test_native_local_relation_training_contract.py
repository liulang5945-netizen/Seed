"""PLAN61 zero checkpoint and exact three-group Adam advancement on synthetic input."""

import pytest

from scripts.training import run_taiji_n1_local_relation as runner


def test_initial_bundle_three_adam_groups_and_restore(tmp_path):
    runner.torch.set_num_threads(1)
    model, optimizer = runner.fresh([])
    assert runner.baseline_initial_match(model, optimizer)["passed"]
    saved = runner.save(model, optimizer, tmp_path / "zero.pt", 0)
    assert runner.restore(tmp_path / "zero.pt")["signature"] == saved["expected_signature"]
    item = dict(
        id="synthetic-context52",
        kind="revision",
        split="train",
        observations=["alpha uses quartz.", "bravo uses copper."],
        question="material of alpha?",
        answer="quartz",
    )
    bases = {"reader": 18432, "local": 12288, "context": 6144, "relation": 0}
    for step in (1, 2):
        loss = runner.train_step(model, optimizer, item, f"context52-contract-{step}")
        assert loss["qa_sequences"] == 1
        for name, parameter in model.named_parameters():
            assert (
                int(optimizer.state[parameter]["step"])
                == bases[runner.parameter_group(name)] + step
            )
    saved = runner.save(model, optimizer, tmp_path / "synthetic.pt", 2)
    assert runner.restore(tmp_path / "synthetic.pt")["signature"] == saved["expected_signature"]
    with pytest.raises(ValueError, match="train"):
        runner.train_step(model, optimizer, dict(item, split="dev"), "forbidden")


def test_frozen_recipe_and_source_final_control():
    assert runner.PROTOCOL["updates"] == 6144
    assert runner.PROTOCOL["parameter_count"] == 62056
    assert runner.PROTOCOL["reader_final_adam_step"] == 24576
    assert runner.PROTOCOL["local_final_adam_step"] == 18432
    assert runner.PROTOCOL["context_final_adam_step"] == 12288
    assert runner.PROTOCOL["source_historical_updates"] == 27648
    assert runner.PROTOCOL["final_historical_updates"] == 33792
    train = runner.checked_train()
    assert len(train) == 192 and len(runner.checked_pairs(train)) == 96
    assert runner.checked_curriculum(train)["updates"] == 6144
    source, payload, _ = runner.transfer_source()
    assert source["complete"] and not source["training_sufficient"]
    assert payload["protocol"] == runner.prior.PROTOCOL
