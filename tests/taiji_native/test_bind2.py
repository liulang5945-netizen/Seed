"""Structural tests for the object-centric binding v2 candidate."""

from __future__ import annotations

import copy
from dataclasses import replace

import torch

from taiji.u2_subject import U2Config, U2Subject

SEED = 20261110


def _v1() -> U2Subject:
    return U2Subject(U2Config(seed=SEED))


def _v2() -> U2Subject:
    return U2Subject(replace(U2Config(seed=SEED), binding="v2"))


def test_v2_adds_exactly_seven_tensors_and_keeps_v1_initialization() -> None:
    first = dict(_v1().named_parameters())
    second = dict(_v2().named_parameters())
    assert (sum(p.numel() for p in first.values()), len(first)) == (22638, 191)
    assert (sum(p.numel() for p in second.values()), len(second)) == (23118, 198)
    assert set(second) - set(first) == {
        "slot_anchor",
        "slot_norm.bias",
        "slot_norm.weight",
        "slot_update.first.bias",
        "slot_update.first.weight",
        "slot_update.last.bias",
        "slot_update.last.weight",
    }
    assert all(torch.equal(first[name], second[name]) for name in first)
    assert len(_v2().state_registry()) == 40


def test_v2_bundle_schema_and_cross_identity_rejection() -> None:
    model = _v2()
    bundle = model.export_bundle(model.initial("bundle"))
    assert bundle["schema"] == "taiji-u2-subject-v5"
    restored, state = U2Subject.from_bundle(bundle)
    assert restored.c.binding == "v2"
    assert set(state) == set(restored.state_registry())
    with_old_schema = copy.deepcopy(bundle)
    with_old_schema["schema"] = "taiji-u2-subject-v4"
    try:
        U2Subject.from_bundle(with_old_schema)
    except ValueError as error:
        assert "binding_profile_schema_mismatch" in str(error)
    else:
        raise AssertionError("v5 tensors loaded under a v4 schema")
    with_old_binding = copy.deepcopy(bundle)
    with_old_binding["schema"] = "taiji-u2-subject-v4"
    with_old_binding["config"] = dict(with_old_binding["config"])
    with_old_binding["config"]["binding"] = "v1"
    try:
        U2Subject.from_bundle(with_old_binding)
    except ValueError:
        pass
    else:
        raise AssertionError("v4 schema accepted with v1 binding")


def test_v2_binding_shapes_and_ablation_switches() -> None:
    model = _v2()
    state = model.initial("binding")
    tokens = {"rgb": torch.randn(16, 8)}
    coordinates = {"rgb": torch.zeros(16, 8)}
    content, mass, spatial = model.binding(state, tokens, coordinates)
    assert content.shape == (2, 3, 8)
    assert mass.shape == (2, 3)
    assert spatial.shape == (2, 3)
    assert bool(torch.isfinite(content).all())
    assert bool(((mass >= 0) & (mass <= 1)).all())
    model._binding_steps = 1
    single, _, _ = model.binding(state, tokens, coordinates)
    assert not torch.allclose(content, single)
    model._binding_steps = 2
    model._binding_anchor = False
    without_anchor, _, _ = model.binding(state, tokens, coordinates)
    assert not torch.allclose(content, without_anchor)
