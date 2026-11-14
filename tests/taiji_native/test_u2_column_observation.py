"""Contract tests for the column-structured observation pathway."""

from __future__ import annotations

import torch

from scripts.training.first_a_support_22_5 import make_model
from taiji.u2_observation_columns import (
    IDENTITY,
    class_of_column,
    column_coordinates,
    column_index_of,
    column_tokens,
    column_values,
)


def _image(patches: torch.Tensor) -> torch.Tensor:
    """Independent reconstruction of the 16x16x3 image from the patches."""
    return patches.reshape(4, 4, 4, 4, 3).permute(0, 2, 1, 3, 4).reshape(16, 16, 3)


def _sample(seed: int = 7) -> dict:
    generator = torch.Generator().manual_seed(seed)
    patches = torch.rand(16, 48, generator=generator)
    coordinates = torch.zeros(16, 8)
    coordinates[:, 0] = torch.tensor([2 * ((index % 4) + 0.5) / 4 - 1 for index in range(16)])
    coordinates[:, 1] = torch.tensor([2 * ((index // 4) + 0.5) / 4 - 1 for index in range(16)])
    coordinates[:, 5] = 1
    coordinates[:, 7] = 1
    return {"observations": {"rgb": patches}, "coordinates": {"rgb": coordinates}}


def test_identity_is_declared() -> None:
    assert IDENTITY == "column-observation-v1"


def test_column_values_are_a_permutation_of_the_patches() -> None:
    sample = _sample()
    columns = column_values(sample["observations"]["rgb"])
    assert columns.shape == (16, 48)
    assert torch.equal(
        columns.flatten().sort().values, sample["observations"]["rgb"].flatten().sort().values
    )


def test_column_k_equals_image_column_k() -> None:
    sample = _sample()
    image = _image(sample["observations"]["rgb"])
    columns = column_values(sample["observations"]["rgb"])
    for column in range(16):
        expected = image[:, column, :].reshape(-1)
        assert torch.equal(columns[column], expected), column


def test_wrong_permutation_is_caught() -> None:
    """The equality test above must fail on a plausible near-miss regrouping."""
    sample = _sample()
    image = _image(sample["observations"]["rgb"])
    correct = column_values(sample["observations"]["rgb"])
    wrong = correct.flip(0)
    assert not torch.equal(wrong[0], image[:, 0, :].reshape(-1))
    assert torch.equal(wrong[15], image[:, 0, :].reshape(-1))


def test_column_coordinates_put_the_column_centre_on_the_x_axis() -> None:
    sample = _sample()
    frame = column_coordinates(sample["coordinates"]["rgb"])
    assert frame.shape == (16, 8)
    for column in range(16):
        assert abs(float(frame[column, 0]) - column_index_of(column)) < 1e-6
        assert float(frame[column, 1]) == 0.0
    assert torch.equal(frame[:, 2:], sample["coordinates"]["rgb"][:, 2:])


def test_column_classes_match_position_word() -> None:
    from taiji.u2_task_episode import position_word

    for column in range(16):
        assert class_of_column(column) == position_word(column_index_of(column))


def test_column_tokens_use_the_body_without_new_parameters() -> None:
    sample = _sample()
    model = make_model("native", 20261110)
    model.eval()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    tokens = column_tokens(model, sample["observations"], sample["coordinates"])
    assert tokens.shape == (16, 8)
    after = model.state_dict()
    assert set(before) == set(after)
    assert all(torch.equal(before[name], after[name]) for name in before)


def test_column_tokens_are_deterministic_but_not_constant() -> None:
    sample = _sample()
    model = make_model("native", 20261110)
    model.eval()
    first = column_tokens(model, sample["observations"], sample["coordinates"])
    second = column_tokens(model, sample["observations"], sample["coordinates"])
    assert torch.equal(first, second)
    assert float(first.detach().std(0).mean()) > 0


def test_shape_guard_rejects_the_wrong_layout() -> None:
    try:
        column_values(torch.zeros(16, 49))
    except ValueError as error:
        assert "column_values_expects_16x48" in str(error)
    else:
        raise AssertionError("expected a ValueError")
