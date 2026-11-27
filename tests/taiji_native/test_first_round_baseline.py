"""Reference calibration preserves public inputs, byte semantics and recovery."""

import copy

import pytest
import torch

from scripts.training.calibrate_first_round_22_5 import (
    measure,
    optimizer_for,
    payload_for,
    persist,
    restore,
)
from scripts.training.first_round_baseline_22_5 import (
    FirstRoundBaseline,
    generate,
    loss_and_logits,
    make_batch,
)
from scripts.training.paired_world_22_5 import build_partition, public_image
from taiji.u2_history import assert_values_equal


@pytest.fixture(scope="module")
def batch():
    torch.set_num_threads(1)
    return make_batch(build_partition("train"))


def test_model_size_and_public_pixel_layout(batch):
    model = FirstRoundBaseline()
    assert sum(parameter.numel() for parameter in model.parameters()) == 65169
    assert batch["images"].shape == (18, 3, 16, 16)
    assert batch["targets"].shape == (180, 7)
    assert bool(batch["teacher_masks"].gather(-1, batch["targets"][..., None]).all())
    partition = build_partition("train")
    episode = partition["episodes"][0]
    expected = public_image(partition["worlds"][episode["world_id"]]["public"])
    actual = (
        (batch["images"][batch["image_index"][0]].permute(1, 2, 0) * 255).round().to(torch.uint8)
    )
    assert torch.equal(actual, expected)


def test_private_fields_and_future_gold_do_not_change_context():
    partition = build_partition("train")
    changed = copy.deepcopy(partition)
    for episode in changed["episodes"]:
        episode["audit"] = {"private": "ignored"}
        episode["targets"] = (b"abcdef", b"uvwxyz")
    first, second = make_batch(partition), make_batch(changed)
    for key in ("images", "image_index", "question_ids", "question_valid"):
        assert torch.equal(first[key], second[key])
    model = FirstRoundBaseline()
    assert torch.equal(model.teacher_logits(first)[:, 0], model.teacher_logits(second)[:, 0])


def test_shared_image_bank_matches_expanded_images_and_full_gradients(batch):
    model = FirstRoundBaseline()
    expanded = dict(batch)
    expanded["images"] = batch["images"][batch["image_index"]]
    expanded["image_index"] = torch.arange(180)
    loss, logits = loss_and_logits(model, batch)
    gradients = torch.autograd.grad(loss, list(model.parameters()))
    other_loss, other_logits = loss_and_logits(model, expanded)
    other_gradients = torch.autograd.grad(other_loss, list(model.parameters()))
    torch.testing.assert_close(logits, other_logits, atol=1e-6, rtol=1e-5)
    for first, second in zip(gradients, other_gradients, strict=True):
        torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-5)


def test_teacher_sequence_matches_normal_step_entry(batch):
    model = FirstRoundBaseline()
    context = model.context(
        batch["images"], batch["image_index"], batch["question_ids"], batch["question_valid"]
    )
    hidden = context[None]
    logits = []
    for step in range(7):
        current, hidden = model.step(batch["previous"][:, step], context, hidden)
        logits.append(current)
    torch.testing.assert_close(
        torch.stack(logits, 1), model.teacher_logits(batch), atol=1e-6, rtol=1e-5
    )


def test_eos_generation_keeps_all_failures_in_denominator(batch):
    model = FirstRoundBaseline()
    with torch.no_grad():
        model.output.weight.zero_()
        model.output.bias.fill_(-100)
        model.output.bias[256] = 100
    result = generate(model, batch)
    assert result["denominator"] == 180 and result["correct"] == 0
    assert all(
        row["eos"] and row["valid_utf8"] and row["output_hex"] == "" for row in result["rows"]
    )
    assert len(result["by_question"]) == 6


def test_checkpoint_zero_and_nonempty_adam_restore_next_loss_and_gradients(tmp_path, batch):
    model = FirstRoundBaseline(31)
    optimizer = optimizer_for(model)
    payload = payload_for(model, optimizer, 0, "fixture", batch["episode_ids"], 0.0)
    persist(tmp_path / "zero.bin", payload, model, optimizer, batch)
    loss_and_logits(model, batch)[0].backward()
    optimizer.step()
    payload = payload_for(model, optimizer, 1, "fixture", batch["episode_ids"], 0.1)
    restored, other = restore(payload, "fixture")
    assert other.state
    assert_values_equal(optimizer.state_dict(), other.state_dict())
    assert_values_equal(measure(model, batch), measure(restored, batch))
    wrong = {**payload, "examples_seen": 1}
    with pytest.raises(AssertionError):
        restore(wrong, "fixture")
