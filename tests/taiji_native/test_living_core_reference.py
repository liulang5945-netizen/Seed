"""R021 forward oracles; no backward, optimizer, corpus or persisted subject."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from taiji.living_core import (
    EOS,
    LivingCoreConfig,
    LivingCoreReference,
    RawSource,
    state_normalize,
)


def assert_close(actual, expected, *, atol=1e-12, rtol=1e-12):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def model_case():
    torch.set_num_threads(1)
    model = LivingCoreReference(
        LivingCoreConfig(groups=2, width=8, units=12, adapter_rank=2, key_width=4, value_width=4)
    ).double()
    state = model.initial_state("r021-subject")
    state = replace(
        state,
        h=torch.linspace(-0.4, 0.6, 16, dtype=torch.float64).reshape(2, 8),
        fast=torch.linspace(-0.2, 0.3, 64, dtype=torch.float64).reshape(2, 2, 4, 4),
    )
    records = (RawSource("a", b"AA"), RawSource("b", b"AB", version=1))
    event = model.encoder.patches(b"input")[0]
    return model, state, records, event


def test_r021_i1_patch_pooling_and_causality():
    model, state, _, _ = model_case()
    raw = b"0123456789"
    context = model.encoder.context(raw)
    patches = model.encoder.patches(raw)
    expected = F.linear(
        torch.stack([context[:8].mean(0), context[8:].mean(0)]),
        model.encoder.event.weight,
        model.encoder.event.bias,
    )
    assert_close(patches, expected, atol=1e-12, rtol=1e-12)
    assert_close(context[:8], model.encoder.context(b"01234567XX")[:8], atol=0, rtol=0)
    assert model.encoder.patches(b"").shape == (0, 8)
    assert model.observe(state, b"", goal=b"").step == state.step
    assert model.observe(state, raw, goal=b"").step == state.step + 2


def test_r021_i2_nu_missing_and_projection():
    model, state, _, _ = model_case()
    raw = torch.tensor([0.5, 2, float("nan"), 0.8, 8, -0.4, 0.2, 0.3], dtype=torch.float64)
    known = torch.tensor([True, True, False, True, True, True, True, True])
    ema, projected, features = model.update_nu(state, raw, known)
    expected = torch.tensor([0.5, 0.1, 0, 0.04, 8, -0.02, 0.01, 0.015], dtype=torch.float64)
    assert_close(ema, expected)
    scale = raw.new_tensor([1, 1, 1, 1, 16, 1, 1, 1])
    expected_features = torch.cat([torch.tanh(expected / scale), (~known).double()])
    assert_close(features, expected_features)
    assert_close(
        projected, F.linear(expected_features, model.nu_projection.weight, model.nu_projection.bias)
    )
    assert features[10] == 1 and torch.isfinite(projected).all()


def test_r021_i3_legal_sources_null_and_two_tanh():
    model, state, records, _ = model_case()
    illegal = (
        RawSource("foreign", b"x" * 1024, owner="other"),
        RawSource("future", b"future", observed_at=100),
    )
    view = model.source_view(records + illegal, "owner", "scope", 0)
    assert len(view.keys) == 4
    goal = model.encoder.goal_vector(b"query")
    recalled, weights = model.episodic(state.h, goal, view)
    e = model.episodic
    a = torch.cat([view.positions, e.null_key[None]])
    metadata = torch.cat([view.metadata, view.metadata.new_zeros(1, 4)])
    first = F.linear(a, e.a.weight)[None] + F.linear(metadata, e.meta.weight, e.meta.bias)[None]
    first = first + F.linear(state_normalize(state.h), e.h.weight)[:, None]
    first = first + F.linear(goal, e.g.weight)[None, None]
    scores = F.linear(
        torch.tanh(F.linear(torch.tanh(first), e.second.weight, e.second.bias)),
        e.score.weight,
        e.score.bias,
    ).squeeze(-1)
    assert_close(weights, scores.softmax(-1), atol=1e-12, rtol=1e-12)
    empty = model.source_view((), "owner", "scope", 0)
    null_recall, null_weights = e(state.h, goal, empty)
    assert_close(null_recall, e.null_value.expand_as(null_recall))
    assert torch.equal(null_weights, torch.ones(2, 1, dtype=torch.float64))
    assert recalled.shape == (2, 8)


def test_r021_i4_formal_group_qkv():
    model, state, _, _ = model_case()
    goal = model.encoder.goal_vector(b"goal")
    message, weights, q, k, v = model.messages(state.h, goal)
    expected_q = torch.stack([layer.weight @ state.h[i] for i, layer in enumerate(model.message_q)])
    expected_q = expected_q + model.message_goal.weight @ goal
    expected_k = torch.stack([layer.weight @ state.h[i] for i, layer in enumerate(model.message_k)])
    expected_v = torch.stack([layer.weight @ state.h[i] for i, layer in enumerate(model.message_v)])
    assert_close(q, expected_q)
    assert_close(k, expected_k)
    assert_close(v, expected_v)
    expected_weights = (expected_q @ expected_k.T / (8**0.5)).softmax(-1)
    assert_close(weights, expected_weights)
    assert_close(message, expected_weights @ expected_v)


def test_r021_fast_read_and_dynamics():
    model, state, records, event = model_case()
    original_h, original_f = state.h.clone(), state.fast.clone()
    out = model.step(state, event=event, goal=b"goal", records=records)
    t = out.trace
    u = torch.cat([state_normalize(state.h[0]), event, t["goal"], t["nu"]])
    raw_q = model.fast_queries[0].weight @ u + model.fast_queries[0].bias
    query = raw_q / torch.sqrt(1 + raw_q.square().sum())
    assert_close(t["fast_queries"][0, 0], query)
    assert_close(t["fast_values"][0, 0], state.fast[0, 0].T @ query)
    merged = (
        model.fast_outputs[0].weight @ t["fast_values"][0].flatten() + model.fast_outputs[0].bias
    )
    assert_close(t["fast_read"][0], merged)
    x = torch.cat(
        [
            state_normalize(state.h),
            event.expand_as(state.h),
            t["message"],
            t["fast_read"],
            t["recalled"],
            t["goal"].expand_as(state.h),
            t["nu"].expand_as(state.h),
        ],
        -1,
    )
    assert_close(t["x"], x)
    effective = model.W + model.U @ model.V.transpose(1, 2)
    pre = torch.stack(
        [effective[i] @ x[i] + model.P[i] @ torch.tanh(model.R[i] @ x[i]) for i in range(2)]
    )
    leak = (1 - torch.exp(-1 / t["tau"]))[:, None]
    assert_close(out.state.h, (1 - leak) * state.h + leak * torch.tanh(pre))
    assert torch.equal(state.h, original_h) and torch.equal(state.fast, original_f)
    assert out.state.subject_id == state.subject_id and out.state.step == state.step + 1


def test_r021_continuous_fast_update_and_permissions():
    model, state, records, event = model_case()
    allowed = torch.ones(2, 2, dtype=torch.bool)
    out = model.step(state, event=event, goal=b"goal", records=records, allowed=allowed)
    mode = torch.tensor([1, 0, 0, 0, 0, 0, 0], dtype=torch.float64)
    context = torch.cat(
        [
            state_normalize(out.state.h[0]),
            event,
            out.trace["goal"],
            out.trace["nu"],
            event.new_tensor([0, 1]),
            mode,
        ]
    )
    raw = model.write_heads[0](context).reshape(2, -1)[0]
    k, v, decay, erase, write, strength, chi = raw.split((4, 4, 4, 4, 4, 1, 1))
    k = k / torch.sqrt(1 + k.square().sum())
    v = v.tanh() / torch.sqrt(1 + v.tanh().square().sum())
    direction = erase.sigmoid() * k
    eta = 16 * strength.sigmoid()[0]
    a = torch.eye(4, dtype=torch.float64) - eta * torch.outer(direction, direction) / (
        1 + eta * direction.square().sum()
    )
    edited = a @ (decay.sigmoid()[:, None] * state.fast[0, 0]) + torch.outer(k, write.sigmoid() * v)
    expected = (1 - chi.sigmoid()[0]) * state.fast[0, 0] + chi.sigmoid()[0] * edited
    assert_close(out.state.fast[0, 0], expected)
    assert bool(((out.trace["chi"] > 0) & (out.trace["chi"] < 1)).all())
    blocked = model.step(state, event=event, goal=b"goal", records=records)
    assert torch.equal(blocked.state.fast, state.fast)
    for fraction in (0.0, 0.2, 0.7, 1.0):
        value = model.edit_fast(
            state.fast[0, 0],
            k,
            v,
            decay.sigmoid(),
            erase.sigmoid(),
            write.sigmoid(),
            strength[0],
            fraction,
            True,
            16,
        )
        assert_close(value, (1 - fraction) * state.fast[0, 0] + fraction * edited)


def test_r021_i5_query_read_and_head_order():
    model, state, records, event = model_case()
    out = model.step(state, event=event, goal=b"query", records=records)
    xi, h, read = out.trace["query"], out.state.h, model.readout
    expected, all_weights = [], []
    for head in range(2):
        q = read.query[head].weight @ xi
        keys = torch.stack([read.keys[head * 2 + i].weight @ h[i] for i in range(2)])
        values = torch.stack([read.values[head * 2 + i].weight @ h[i] for i in range(2)])
        weights = (keys @ q / (8**0.5)).softmax(0)
        expected.append(weights @ values)
        all_weights.append(weights)
    joined = torch.cat(expected)
    assert_close(out.trace["joined_heads"], joined)
    assert_close(out.trace["z"], read.output.weight @ joined)
    assert_close(out.trace["query_weights"], torch.stack(all_weights))
    z1, w1, _ = read(h, xi.new_ones(8))
    z2, w2, _ = read(h, -xi.new_ones(8))
    assert not torch.allclose(w1, w2, atol=1e-12, rtol=0)
    assert not torch.allclose(z1, z2, atol=1e-12, rtol=0)


def test_r021_i6_mixture_and_no_source():
    model, state, records, event = model_case()
    out = model.step(state, event=event, goal=b"query", records=records)
    copied = torch.zeros(257, dtype=torch.float64).scatter_add(
        0, out.sources.byte_ids, out.copy_weights
    )
    mixing = out.trace["mixing"]
    assert_close(out.probabilities, mixing * out.trace["generated"] + (1 - mixing) * copied)
    assert_close(out.probabilities.sum(), mixing.new_tensor(1))
    assert copied[EOS] == 0 and out.copy_weights.shape == (4,)
    empty = model.step(state, event=event, goal=b"query")
    assert empty.trace["mixing"] == 1
    assert torch.equal(empty.probabilities, empty.trace["generated"])


def test_r021_i6_coverage_stays_inside_source():
    model, state, records, event = model_case()
    out = model.step(state, event=event, goal=b"query", records=records)
    out = replace(out, copy_weights=torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64))
    child = model.consume_output(out, ord("A"))
    assert_close(child.coverage, event.new_tensor([1 / 6, 2 / 6, 3 / 6, 0]))
    assert_close(child.prior, event.new_tensor([0, 0.25, 0, 0.75]))
    unknown = model.consume_output(out, 255)
    assert torch.equal(unknown.coverage, out.state.coverage) and unknown.prior.sum() == 0
    reordered = tuple(reversed(child.keys)) + (("new", 0),)
    cov, _ = model.align_positions(child, reordered)
    assert_close(cov, torch.cat([child.coverage.flip(0), child.coverage.new_zeros(1)]))
    continued = model.step(child, event=event, goal=b"query", records=records)
    assert continued.trace["continuation"] > 0


def test_r021_target_is_consumed_after_prediction():
    model, state, records, _ = model_case()
    first, _ = model.score_targets(
        state, prompt=b"prompt", target=b"AB", goal=b"query", records=records
    )
    second, _ = model.score_targets(
        state, prompt=b"prompt", target=b"AC", goal=b"query", records=records
    )
    assert_close(first[:2], second[:2], atol=0, rtol=0)
    assert not torch.allclose(first[2], second[2], atol=1e-12, rtol=0)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_r021_current_encoding_and_world_heads():
    model, state, records, event = model_case()
    old = model.source_view(records, "owner", "scope", 0).positions
    # A declared parameter intervention checks cache freshness; it is not learning.
    with torch.no_grad():
        model.encoder.embedding.weight[ord("A")].add_(0.1)
    current = model.source_view(records, "owner", "scope", 0).positions
    assert not torch.equal(old, current)
    out = model.step(state, event=event, goal=b"query", records=records)
    expected = torch.stack(
        [head(torch.cat([out.trace["z"], event.new_zeros(8)])) for head in model.world]
    )
    assert_close(out.prediction, expected[:, :8])
    assert_close(out.predicted_reward, expected[:, 8])
    assert out.prediction.shape == (3, 8)
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_r021_budgets_and_malformed_interfaces_fail():
    model, state, records, event = model_case()
    with pytest.raises(ValueError, match="event-byte"):
        model.observe(state, b"x" * 513, goal=b"")
    with pytest.raises(ValueError, match="target-byte"):
        model.score_targets(state, prompt=b"", target=b"x" * 129, goal=b"")
    with pytest.raises(ValueError, match="unique"):
        model.source_view(records + records, "owner", "scope", 0)
    with pytest.raises(ValueError, match="availability"):
        model.update_nu(state, event, event)
    with pytest.raises(ValueError, match="dimensions"):
        model.step(state, event=event[:2], goal=b"")
