"""22.5.0.A / S0 像素显著度读出：双身份接线、出生中性、可学性与恢复回归。"""

from dataclasses import replace
from functools import cache
from pathlib import Path

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
from taiji.u2_pixel_read import (
    READOUT_V1,
    READOUT_V2,
    PixelSalienceReadout,
    PixelSalienceReadoutV2,
    build_readout,
    pixel_grid,
    raster,
    unit_registry,
    unit_registry_v2,
)
from taiji.u2_subject import U2Config, U2Subject

IDENTITIES = (READOUT_V1, READOUT_V2)
ACCEPTED_PARENT = Path(__file__).resolve().parents[2] / (
    "output/a_pixel_parallel_20261119T063705Z/segment.bin"
)


def make(readout=READOUT_V2):
    config = replace(U2Config.small(), seed=20261110, readout=readout)
    return U2Subject(config).metadata_candidate().request_candidate()


@cache
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


def test_identity_isolation_keeps_v1_byte_layout_frozen():
    """The accepted 22.5.0 A parent keeps its exact v1 layout (S0/S1 discipline)."""
    config = replace(U2Config.small(), seed=20261110, readout=READOUT_V1, d=32)
    model = U2Subject(config).metadata_candidate().request_candidate()
    shape = {name: tuple(t.shape) for name, t in model.pixel_readout.state_dict().items()}
    assert shape == {
        "byte.weight": (256, 16),
        "position": (1, 64, 16),
        "text.weight": (16, 32),
        "text.bias": (16,),
        "pixel.weight": (16, 3),
        "pixel.bias": (16,),
        "null": (16,),
        "log_temp": (),
        "out.weight": (32, 6),
        "out.bias": (32,),
    }
    assert model.pixel_unit_owners == unit_registry()


def test_v1_identity_keeps_the_pre_s0_delta_head_layout():
    """The accepted parent's strict load requires the v1 delta-head bias vectors.

    The S0 revision removed both biases; that removal belongs to the new v2
    identity only.  Rebuilding the v1 bias vectors here guards the accepted
    checkpoint layout (see S1 / the accepted-parent load probe).
    """
    v1_parameters = set(dict(make(READOUT_V1).named_parameters()))
    v2_parameters = set(dict(make(READOUT_V2).named_parameters()))
    assert {
        "delta_head.0.weight",
        "delta_head.0.bias",
        "delta_head.2.weight",
        "delta_head.2.bias",
    } <= v1_parameters
    assert {"delta_head.0.bias", "delta_head.2.bias"}.isdisjoint(v2_parameters)


@pytest.mark.skipif(
    not ACCEPTED_PARENT.is_file(),
    reason="local accepted A parent checkpoint is gitignored by policy",
)
def test_accepted_a_parent_loads_strictly_under_the_v1_identity():
    from taiji.u2_history import atomic_load

    payload = atomic_load(ACCEPTED_PARENT)
    assert payload["description"]["config"]["readout"] == READOUT_V1
    model = U2Subject(U2Config(**payload["description"]["config"]))
    first = next(iter(payload["parameters"].values()))
    model = model.to(dtype=first.dtype, device=first.device)
    model.load_state_dict(payload["parameters"])
    model.parameter_revision = payload["revision"]
    assert model.parameter_revision == payload["revision"]


def test_identity_isolation_registers_the_s0_revision_separately():
    """The S0 upgrade is a new identity with its own ledger and no shared checkpoint."""
    v1 = make(READOUT_V1).pixel_readout
    v2 = make(READOUT_V2).pixel_readout
    assert isinstance(v1, PixelSalienceReadout)
    assert isinstance(v2, PixelSalienceReadoutV2)
    assert v2.state_dict().keys() != v1.state_dict().keys()
    with pytest.raises(RuntimeError):
        v1.load_state_dict(v2.state_dict())
    assert make(READOUT_V2).pixel_unit_owners == unit_registry_v2()
    assert all(key.startswith(f"{READOUT_V1}:") for key in unit_registry())
    assert all(key.startswith(f"{READOUT_V2}:") for key in unit_registry_v2())
    with pytest.raises(ValueError, match="Invalid readout identity"):
        build_readout("pixel-salience-v3", 8)
    with pytest.raises(ValueError, match="Invalid readout identity"):
        replace(U2Config.small(), readout="unknown")


@pytest.mark.parametrize("identity", IDENTITIES)
def test_pixel_readout_is_born_neutral_and_red_direction_is_live(identity):
    parent = make("patch")
    child = make(identity)
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


@pytest.mark.parametrize("identity", IDENTITIES)
def test_pixel_delta_is_bound_to_request_and_cached_recompute_is_exact(identity):
    model = make(identity)
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


@pytest.mark.parametrize("identity", IDENTITIES)
def test_zero_initialized_injection_learns_upstream_after_first_credit(identity):
    model = make(identity)
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


@pytest.mark.parametrize("identity", IDENTITIES)
def test_pixel_readout_save_load_preserves_rows(identity, tmp_path):
    model = make(identity)
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
    assert payload["description"]["config"]["readout"] == identity


@pytest.mark.parametrize(
    "operator_class", (PixelSalienceReadout, PixelSalienceReadoutV2), ids=["v1", "v2"]
)
def test_pixel_operator_rejects_empty_request(operator_class):
    operator = operator_class()
    blocks = torch.zeros(16, 48)
    coordinates = torch.zeros(16, 8)
    with pytest.raises(ValueError, match="empty_request"):
        operator(blocks, pixel_grid(coordinates), torch.tensor([], dtype=torch.long))


@pytest.mark.parametrize("identity", IDENTITIES)
def test_utterance_derived_state_can_be_exported_and_restored(identity):
    model = make(identity)
    example = first_example()
    state = prepare(model, example["public"], example["id"])
    bundle = model.export_bundle(state)
    restored, restored_state = U2Subject.from_bundle(bundle)
    assert value_digest(restored_state) == value_digest(state)
    assert restored_state["pixel_delta"].shape == state["pixel_delta"].shape
    assert value_digest(output(restored, restored_state)[0]) == value_digest(
        output(model, state)[0]
    )
