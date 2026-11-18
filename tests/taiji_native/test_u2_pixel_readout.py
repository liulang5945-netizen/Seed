"""22.5.0.A 像素显著度读出：接线、出生中性、可学性与恢复回归。"""

from dataclasses import replace

import pytest
import torch

from scripts.training.first_a_support_22_5 import (
    generate_examples,
    load_learner,
    output,
    prepare,
    save_learner,
    task_loss,
)
from taiji.u2_history import value_digest
from taiji.u2_pixel_read import PixelSalienceReadout, pixel_grid, raster
from taiji.u2_subject import U2Config, U2Subject


def make(readout="pixel-salience"):
    config = replace(U2Config.small(), seed=20261110, readout=readout)
    return U2Subject(config).metadata_candidate().request_candidate()


def first_example():
    return generate_examples("train", 4)[0]


def finite_delta(left, right):
    valid = torch.isfinite(left) & torch.isfinite(right)
    assert torch.equal(torch.isfinite(left), torch.isfinite(right))
    return float((left[valid] - right[valid]).detach().abs().max())


def test_pixel_raster_and_grid_match_sensor_layout():
    blocks = torch.arange(16 * 48, dtype=torch.float32).reshape(16, 48)
    image = raster(blocks)
    assert image.shape == (16, 16, 3)
    assert torch.equal(image.reshape(4, 4, 4, 4, 3).permute(0, 2, 1, 3, 4).reshape(16, 48), blocks)
    coordinates = torch.zeros(16, 8)
    coordinates[:, 0] = torch.arange(4).repeat(4) / 4 - 0.75
    coordinates[:, 1] = torch.arange(4).repeat_interleave(4) / 4 - 0.75
    grid = pixel_grid(coordinates)
    assert grid.shape == (16, 16, 2)
    offsets = torch.tensor([-3 / 32, -1 / 32, 1 / 32, 3 / 32])
    expected_x = (coordinates[:, 0, None, None] + offsets[None, None, :]).expand(-1, 4, 4)
    expected_y = (coordinates[:, 1, None, None] + offsets[None, :, None]).expand(-1, 4, 4)
    expected = torch.stack([expected_x, expected_y], -1)
    expected = expected.reshape(4, 4, 4, 4, 2).permute(0, 2, 1, 3, 4).reshape(16, 16, 2)
    assert torch.allclose(grid, expected)


def test_pixel_readout_is_born_neutral_and_red_direction_is_live():
    parent = make("patch")
    child = make("pixel-salience")
    example = first_example()
    parent_state = prepare(parent, example["public"], example["id"])
    child_state = prepare(child, example["public"], example["id"])
    parent_probability, _, _ = output(parent, parent_state)
    child_probability, _, _ = output(child, child_state)
    assert finite_delta(parent_probability, child_probability) == 0.0
    with torch.no_grad():
        child.pixel_readout.out.weight.add_(1e-3)
    shifted_state = prepare(child, example["public"], example["id"])
    shifted_probability, _, _ = output(child, shifted_state)
    assert finite_delta(parent_probability, shifted_probability) > 0.0


def test_pixel_delta_is_bound_to_request_and_cached_recompute_is_exact():
    model = make()
    with torch.no_grad():
        model.pixel_readout.out.weight.fill_(0.01)
    example = first_example()
    state = prepare(model, example["public"], example["id"])
    assert "pixel_delta" in state
    cached = state["pixel_delta"].detach().clone()
    uncached = dict(state)
    del uncached["pixel_delta"]
    cached_probability, _, _ = output(model, state)
    recomputed_probability, _, _ = output(model, uncached)
    assert finite_delta(cached_probability, recomputed_probability) == 0.0
    other = dict(example["public"])
    other["request"] = "蓝色圆片现在在哪里？".encode()
    other_state = prepare(model, other, example["id"])
    assert not torch.equal(cached, other_state["pixel_delta"])


def test_zero_initialized_injection_learns_upstream_after_first_credit():
    model = make()
    example = first_example()
    parameters = dict(model.pixel_readout.named_parameters())
    optimizer = torch.optim.Adam(model.pixel_readout.parameters(), lr=0.01)
    loss = task_loss(model, [example])
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert parameters["out.weight"].grad is not None
    assert parameters["out.weight"].grad.abs().sum() > 0
    assert (
        parameters["pixel.weight"].grad is None or parameters["pixel.weight"].grad.abs().sum() == 0
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    task_loss(model, [example]).backward()
    assert parameters["pixel.weight"].grad is not None
    assert parameters["pixel.weight"].grad.abs().sum() > 0
    assert parameters["byte.weight"].grad is not None
    assert parameters["byte.weight"].grad.abs().sum() > 0


def test_pixel_readout_save_load_preserves_rows(tmp_path):
    model = make()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    examples = generate_examples("fit", 4)
    rows = []
    for example in examples:
        state = prepare(model, example["public"], example["id"])
        rows.append(output(model, state)[0].detach())
    checkpoint = tmp_path / "pixel.bin"
    save_learner(checkpoint, model, optimizer, "native", 0, 0, [], value_digest(examples))
    restored, _, payload = load_learner(checkpoint)
    restored_rows = []
    for example in examples:
        state = prepare(restored, example["public"], example["id"])
        restored_rows.append(output(restored, state)[0].detach())
    assert value_digest(rows) == value_digest(restored_rows)
    assert payload["description"]["config"]["readout"] == "pixel-salience"


def test_pixel_operator_rejects_empty_request():
    operator = PixelSalienceReadout()
    blocks = torch.zeros(16, 48)
    coordinates = torch.zeros(16, 8)
    with pytest.raises(ValueError, match="empty_request"):
        operator(blocks, pixel_grid(coordinates), torch.tensor([], dtype=torch.long))


def test_utterance_derived_state_can_be_exported_and_restored():
    model = make()
    example = first_example()
    state = prepare(model, example["public"], example["id"])
    bundle = model.export_bundle(state)
    restored, restored_state = U2Subject.from_bundle(bundle)
    assert value_digest(restored_state) == value_digest(state)
    assert restored_state["pixel_delta"].shape == state["pixel_delta"].shape
    assert value_digest(output(restored, restored_state)[0]) == value_digest(
        output(model, state)[0]
    )
