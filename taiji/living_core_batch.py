"""20.4b graph-level batched executor for the R021 reference graph.

The frozen single-world ``LivingCoreReference`` / ``GatedGrowthCore`` path stays
untouched (owner ruling O2): this module only *adds* an executor that runs a
leading world dimension, reuses static per-update constants, and replaces the
per-key Python ``select`` loops with ``gather``.  Same parameters, same
arithmetic, same order of operations; the reference path remains available for
the 1e-6 equivalence gate.

No rule, signature, truth table or solver can reach this file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from taiji.living_core import (
    EOS,
    position_features,
    state_normalize,
    unit_ball,
)

BLOCK = 8
ANSWER_BYTES = 15


class BatchShapeError(RuntimeError):
    """The two worlds are not shape-uniform; the batched path cannot be used."""


@dataclass(frozen=True)
class BatchedView:
    keys: tuple[tuple[tuple[str, int], ...], ...]
    byte_ids: torch.Tensor
    positions: torch.Tensor
    positional: torch.Tensor
    metadata: torch.Tensor


@dataclass(frozen=True)
class BatchedLiving:
    subject_id: tuple[str, ...]
    revision: int
    step: int
    h: torch.Tensor
    fast: torch.Tensor
    nu_ema: torch.Tensor
    keys: tuple[tuple[tuple[str, int], ...], ...]
    coverage: torch.Tensor
    prior: torch.Tensor


@dataclass(frozen=True)
class BatchedStep:
    state: BatchedLiving
    probabilities: torch.Tensor
    copy_weights: torch.Tensor
    sources: BatchedView


def initial_batched_state(core, subject_ids: tuple[str, ...]) -> BatchedLiving:
    c, ref = core.config, core.W
    batch = len(subject_ids)
    return BatchedLiving(
        subject_id=subject_ids,
        revision=0,
        step=0,
        h=ref.new_zeros((batch, c.groups, c.width)),
        fast=ref.new_zeros((batch, c.groups, c.fast_heads, c.key_width, c.value_width)),
        nu_ema=ref.new_zeros((batch, 8)),
        keys=tuple(() for _ in range(batch)),
        coverage=ref.new_zeros((batch, 0)),
        prior=ref.new_zeros((batch, 0)),
    )


def update_constants(core) -> dict[str, torch.Tensor]:
    """Tensors that are constant for a whole forward pass (weights do not move)."""
    tau = 0.5 + 127.5 * torch.sigmoid(core.tau_logit)
    leak = 1 - torch.exp(-1 / tau)
    return {
        "effective": core.W + core.U @ core.V.transpose(1, 2),
        "leak": leak,
    }


def growth_rows(core, gates: dict[str, float] | None, reference: torch.Tensor):
    parent = core.growth_parent_units if core.growth_parent_units is not None else core.R.shape[1]
    if parent >= core.R.shape[1]:
        return parent, None
    if gates is None:
        raise ValueError("Gated newborn units require per-step branch gates")
    new_units = core.R.shape[1] - parent
    rows = []
    for group in range(core.config.groups):
        branches = core.growth_unit_branches[group]
        values = [float(gates[branch]) for branch in branches]
        values.extend(0.0 for _ in range(new_units - len(branches)))
        rows.append(values)
    return parent, torch.tensor(rows, dtype=reference.dtype, device=reference.device)


def context_batch(encoder, raws, reference: torch.Tensor) -> torch.Tensor:
    """(B,L,W) contexts for equal-length byte strings (one batched convolution)."""
    lengths = {len(raw) for raw in raws}
    if len(lengths) != 1:
        raise BatchShapeError(f"batched context needs uniform lengths, saw {sorted(lengths)}")
    length = lengths.pop()
    if length == 0:
        width = encoder.input.out_features
        return reference.new_empty((len(raws), 0, width))
    ids = torch.tensor([list(raw) for raw in raws], dtype=torch.long, device=reference.device)
    value = encoder.input(encoder.embedding(ids)).transpose(1, 2)
    for convolution in encoder.convolutions:
        value = value + torch.tanh(convolution(F.pad(value, (2 * convolution.dilation[0], 0))))
    return value.transpose(1, 2)


def patch_events_batch(encoder, context: torch.Tensor) -> torch.Tensor:
    """(B,ceil(L/8),W) patch events for padded-mean blocks of ``BLOCK``."""
    batch, length, width = context.shape
    if length == 0:
        return context
    blocks = (length + BLOCK - 1) // BLOCK
    pad = blocks * BLOCK - length
    if pad:
        context = F.pad(context, (0, 0, 0, pad))
    summed = context.reshape(batch, blocks, BLOCK, width).sum(2)
    counts = torch.full((blocks,), float(BLOCK), dtype=context.dtype, device=context.device)
    if pad:
        counts[-1] = float(BLOCK - pad)
    means = summed / counts[None, :, None]
    return encoder.event(means)


def goal_vector_batch(encoder, raws, reference: torch.Tensor) -> torch.Tensor:
    width = encoder.goal.out_features
    if not raws:
        return reference.new_zeros((0, width))
    if any(not raw for raw in raws):
        return reference.new_zeros((len(raws), width))
    context = context_batch(encoder, raws, reference)
    return encoder.goal(context.mean(1))


def patch_table(core, reference: torch.Tensor) -> torch.Tensor:
    """(257,W) precomputed patch events; index 256 (EOS) is the zero vector."""
    raws = [bytes([byte]) for byte in range(256)]
    events = patch_events_batch(core.encoder, context_batch(core.encoder, raws, reference))
    table = events.new_zeros((257, core.config.width))
    table[:256] = events[:, 0]
    return table


def _empty_view(batch: int, width: int, reference: torch.Tensor) -> BatchedView:
    return BatchedView(
        keys=tuple(() for _ in range(batch)),
        byte_ids=torch.empty((batch, 0), dtype=torch.long, device=reference.device),
        positions=reference.new_empty((batch, 0, width)),
        positional=reference.new_empty((batch, 0, width)),
        metadata=reference.new_empty((batch, 0, 4)),
    )


def source_view_batch(
    core, records_per_world, owner: str, scope: str, now: int, reference: torch.Tensor
) -> BatchedView:
    c = core.config
    batch = len(records_per_world)
    legal_per_world = []
    for records in records_per_world:
        if len({record.source_id for record in records}) != len(records):
            raise ValueError("Source identities must be unique")
        if sum(len(record.raw) for record in records if record.raw) > c.max_source_bytes:
            raise ValueError("R021 source-byte budget exceeded")
        legal = [
            record
            for record in records
            if record.owner == owner
            and record.scope == scope
            and record.observed_at <= now
            and record.real
        ]
        for record in legal:
            if record.observed_at < 0 or record.version < 0 or not record.source_id:
                raise ValueError("Malformed source metadata")
        legal = [record for record in legal if record.raw]
        legal_per_world.append(legal)
    counts = {len(legal) for legal in legal_per_world}
    if len(counts) != 1:
        raise BatchShapeError(f"record counts differ across worlds: {sorted(counts)}")
    per_world = counts.pop()
    if per_world == 0:
        return _empty_view(batch, c.width, reference)
    raws = tuple(record.raw for legal in legal_per_world for record in legal)
    lengths = {len(raw) for raw in raws}
    if len(lengths) != 1:
        raise BatchShapeError(f"record byte lengths differ: {sorted(lengths)}")
    length = lengths.pop()
    contexts = context_batch(core.encoder, raws, reference).view(batch, per_world, length, c.width)
    positional = position_features(length, c.width, reference)
    repeat = positional.unsqueeze(0).expand(batch, per_world, length, c.width)
    repeat = repeat.reshape(batch, per_world * length, c.width)
    positions = (contexts + positional).reshape(batch, per_world * length, c.width)
    times = [
        [(now - record.observed_at) / 64, record.observed_at / 64, record.version / 16]
        for legal in legal_per_world
        for record in legal
    ]
    stamp = reference.new_tensor(times).tanh()
    meta = torch.cat([stamp, reference.new_ones((len(times), 1))], -1)
    metadata = meta.view(batch, per_world, 1, 4).expand(batch, per_world, length, 4)
    metadata = metadata.reshape(batch, per_world * length, 4)
    ids = torch.tensor(
        [[byte for record in legal for byte in record.raw] for legal in legal_per_world],
        dtype=torch.long,
        device=reference.device,
    )
    keys = tuple(
        tuple((record.source_id, offset) for record in legal for offset in range(len(record.raw)))
        for legal in legal_per_world
    )
    return BatchedView(
        keys=keys,
        byte_ids=ids,
        positions=positions,
        positional=repeat,
        metadata=metadata,
    )


def align_index(old_keys_per_world, new_keys_per_world, reference: torch.Tensor) -> torch.Tensor:
    rows = []
    for old_keys, new_keys in zip(old_keys_per_world, new_keys_per_world, strict=True):
        mapping = {key: index for index, key in enumerate(old_keys)}
        rows.append([mapping.get(key, -1) for key in new_keys])
    return torch.tensor(rows, dtype=torch.long, device=reference.device)


def identity_index(batch: int, length: int, reference: torch.Tensor) -> torch.Tensor:
    return torch.arange(length, device=reference.device).expand(batch, length)


def shift_index(keys_per_world, reference: torch.Tensor) -> torch.Tensor:
    rows = []
    for keys in keys_per_world:
        mapping = {key: index for index, key in enumerate(keys)}
        rows.append([mapping.get((source, offset - 1), -1) for source, offset in keys])
    return torch.tensor(rows, dtype=torch.long, device=reference.device)


def align_gather(coverage: torch.Tensor, prior: torch.Tensor, index: torch.Tensor):
    batch, length = index.shape
    current = coverage.shape[1]
    if length > current:
        pad = length - current
        coverage = torch.cat([coverage, coverage.new_zeros((batch, pad))], 1)
        prior = torch.cat([prior, prior.new_zeros((batch, pad))], 1)
    elif length < current:
        raise BatchShapeError("the source view shrank across steps")
    safe = index.clamp_min(0)
    gathered = torch.gather(coverage, 1, safe)
    gathered_prior = torch.gather(prior, 1, safe)
    mask = index >= 0
    coverage = torch.where(mask, gathered, torch.zeros_like(gathered))
    prior = torch.where(mask, gathered_prior, torch.zeros_like(gathered_prior))
    total = prior.sum(-1)
    has = total > 0
    denom = torch.where(has, total, torch.ones_like(total))
    prior = torch.where(has.unsqueeze(-1), prior / denom.unsqueeze(-1), prior)
    return coverage, prior


def messages_batch(core, h: torch.Tensor, goal: torch.Tensor):
    stack = []
    for networks in (core.message_q, core.message_k, core.message_v):
        stack.append(torch.stack([layer(h[:, row, :]) for row, layer in enumerate(networks)], 1))
    q, k, v = stack
    q = q + core.message_goal(goal).unsqueeze(1)
    attention = torch.softmax(q @ k.transpose(1, 2) / math.sqrt(core.config.width), -1)
    return attention @ v, attention, q, k, v


def read_fast_batch(core, h, memory, event, goal, nu):
    c = core.config
    inputs = torch.cat(
        [
            state_normalize(h),
            event.unsqueeze(1).expand_as(h),
            goal.unsqueeze(1).expand_as(h),
            nu.unsqueeze(1).expand_as(h),
        ],
        -1,
    )
    queries, values, merged = [], [], []
    for group in range(c.groups):
        q = torch.stack(
            [
                unit_ball(core.fast_queries[group * c.fast_heads + head](inputs[:, group, :]))
                for head in range(c.fast_heads)
            ],
            1,
        )
        value = torch.einsum("bhkv,bhk->bhv", memory[:, group], q)
        queries.append(q)
        values.append(value)
        merged.append(core.fast_outputs[group](value.reshape(value.shape[0], -1)))
    return (
        torch.stack(merged, 1),
        torch.stack(queries, 1),
        torch.stack(values, 1),
    )


def episodic_batch(core, h, goal, view: BatchedView):
    c = core.config
    batch = h.shape[0]
    key = torch.cat(
        [
            view.positions,
            core.episodic.null_key.view(1, 1, c.width).expand(batch, 1, c.width),
        ],
        1,
    )
    metadata = torch.cat([view.metadata, view.metadata.new_zeros((batch, 1, 4))], 1)
    first = core.episodic.a(key).unsqueeze(1) + core.episodic.meta(metadata).unsqueeze(1)
    first = first + core.episodic.h(state_normalize(h)).unsqueeze(2)
    first = first + core.episodic.g(goal).unsqueeze(1).unsqueeze(1)
    scores = core.episodic.score(torch.tanh(core.episodic.second(torch.tanh(first))))
    attention = torch.softmax(scores.squeeze(-1), -1)
    values = torch.cat(
        [
            core.episodic.value(view.positions) + core.episodic.value_meta(view.metadata),
            core.episodic.null_value.view(1, 1, c.width).expand(batch, 1, c.width),
        ],
        1,
    )
    recalled = torch.einsum("bgl,bld->bgd", attention, values)
    return recalled, attention


def readout_batch(core, h, xi):
    c = core.config
    heads, weights = [], []
    for head, query in enumerate(core.readout.query):
        keys = torch.stack(
            [core.readout.keys[head * c.groups + i](h[:, i, :]) for i in range(c.groups)], 1
        )
        values = torch.stack(
            [core.readout.values[head * c.groups + i](h[:, i, :]) for i in range(c.groups)], 1
        )
        projected = query(xi)
        attention = torch.softmax((keys * projected.unsqueeze(1)).sum(-1) / math.sqrt(c.width), 1)
        heads.append(torch.einsum("bn,bnd->bd", attention, values))
        weights.append(attention)
    joined = torch.cat(heads, -1)
    return core.readout.output(joined), torch.stack(weights, 1), joined


def expression_batch(core, z, task, goal, previous, view, coverage, prior):
    exp = core.expression
    batch = z.shape[0]
    generated = torch.softmax(exp.generate(torch.cat([z, previous], -1)), -1)
    mixing = torch.sigmoid(exp.mix(torch.cat([z, task, previous], -1))).squeeze(-1)
    continuation = torch.sigmoid(exp.continuation(torch.cat([z, previous], -1))).squeeze(-1)
    length = view.positions.shape[1]
    if length == 0:
        return (
            generated,
            generated.new_empty((batch, 0)),
            mixing,
            continuation,
            generated,
        )
    score = exp.z(z).unsqueeze(1) + exp.a(view.positions) + exp.meta(view.metadata)
    score = score + exp.position(view.positional) + exp.goal(goal).unsqueeze(1)
    score = score + coverage.unsqueeze(-1) * exp.coverage
    base = torch.softmax(exp.score(torch.tanh(score)).squeeze(-1), -1)
    total = prior.sum(-1)
    has = total > 0
    shifted = torch.where(has.unsqueeze(-1), prior / total.clamp_min(1e-300).unsqueeze(-1), prior)
    weights = (1 - continuation).unsqueeze(-1) * base + continuation.unsqueeze(-1) * shifted
    weights = torch.where(has.unsqueeze(-1), weights, base)
    copied = generated.new_zeros((batch, 257)).scatter_add(1, view.byte_ids, weights)
    probabilities = mixing.unsqueeze(-1) * generated + (1 - mixing).unsqueeze(-1) * copied
    return probabilities, weights, mixing, continuation, generated


def write_fast_batch(core, h, memory, event, goal, nu, mode, past_error, error_missing, allowed):
    c = core.config
    batch = h.shape[0]
    result, all_chi, all_keys = [], [], []
    for group, network in enumerate(core.write_heads):
        context = torch.cat(
            [
                state_normalize(h[:, group, :]),
                event,
                goal,
                nu,
                past_error.reshape(batch, 1),
                error_missing.to(event.dtype).reshape(batch, 1),
                mode,
            ],
            -1,
        )
        raw = network(context).view(batch, c.fast_heads, -1)
        splits = (c.key_width, c.value_width, c.key_width, c.key_width, c.value_width, 1, 1)
        key, value, decay, erase, write, strength, chi = raw.split(splits, dim=-1)
        key = unit_ball(key)
        value = unit_ball(torch.tanh(value))
        chi = torch.sigmoid(chi)
        heads = []
        for head in range(c.fast_heads):
            slot = memory[:, group, head]
            decayed = torch.sigmoid(decay[:, head, :]).unsqueeze(-1) * slot
            direction = torch.sigmoid(erase[:, head, :]) * key[:, head, :]
            eta = 16 * torch.sigmoid(strength[:, head, 0])
            factor = eta / (1 + eta * direction.square().sum(-1))
            projected = (direction.unsqueeze(-1) * decayed).sum(1)
            erased = decayed - factor.unsqueeze(-1).unsqueeze(-1) * direction.unsqueeze(
                -1
            ) * projected.unsqueeze(-2)
            weight = torch.sigmoid(write[:, head, :]) * value[:, head, :]
            edited = erased + key[:, head, :].unsqueeze(-1) * weight.unsqueeze(-2)
            gate = chi[:, head, 0]
            candidate = (1 - gate).unsqueeze(-1).unsqueeze(-1) * slot + gate.unsqueeze(
                -1
            ).unsqueeze(-1) * edited
            norm = torch.linalg.vector_norm(candidate.reshape(batch, -1), dim=1)
            scale = torch.where(
                norm <= c.fast_limit,
                torch.ones_like(norm),
                c.fast_limit / norm.clamp_min(1e-300),
            )
            scaled = candidate * scale.unsqueeze(-1).unsqueeze(-1)
            permit = allowed[:, group, head]
            heads.append(torch.where(permit.unsqueeze(-1).unsqueeze(-1), scaled, slot))
        result.append(torch.stack(heads, 1))
        all_chi.append(chi.squeeze(-1))
        all_keys.append(key)
    return torch.stack(result, 1), torch.stack(all_chi, 1), torch.stack(all_keys, 1)


def step_batch(
    core,
    states: BatchedLiving,
    *,
    event: torch.Tensor,
    goal_vector: torch.Tensor,
    view: BatchedView,
    align: torch.Tensor,
    previous_byte: torch.Tensor,
    allowed: torch.Tensor,
    const: dict[str, torch.Tensor],
    parent_units: int,
    gate_rows: torch.Tensor | None,
    mode: torch.Tensor | None = None,
    past_error: torch.Tensor | None = None,
    error_missing: torch.Tensor | None = None,
    raw_nu: torch.Tensor | None = None,
    known_nu: torch.Tensor | None = None,
) -> BatchedStep:
    reference = core.W
    batch = states.h.shape[0]
    if raw_nu is None:
        raw_nu = reference.new_zeros((batch, 8))
    if known_nu is None:
        known_nu = torch.zeros((batch, 8), dtype=torch.bool, device=reference.device)
    if mode is None:
        mode = torch.zeros(batch, dtype=torch.long, device=reference.device)
    if past_error is None:
        past_error = reference.new_zeros(batch)
    if error_missing is None:
        error_missing = torch.ones(batch, dtype=torch.bool, device=reference.device)
    if align is None:
        align = identity_index(batch, view.byte_ids.shape[1], reference)

    safe = torch.where(known_nu, raw_nu, torch.zeros_like(raw_nu))
    smoothed = 0.95 * states.nu_ema + 0.05 * safe
    direct = torch.tensor(
        [True, False, False, False, True, False, False, False], device=reference.device
    )
    ema = torch.where(known_nu, torch.where(direct, safe, smoothed), states.nu_ema)
    scale = reference.new_tensor([1, 1, 1, 1, 16, 1, 1, 1])
    value = torch.where(known_nu, torch.tanh(ema / scale), torch.zeros_like(ema))
    features = torch.cat([value, (~known_nu).to(reference.dtype)], -1)
    nu = core.nu_projection(features)

    coverage, prior = align_gather(states.coverage, states.prior, align)
    h0 = states.h
    message = messages_batch(core, h0, goal_vector)[0]
    fast = read_fast_batch(core, h0, states.fast, event, goal_vector, nu)[0]
    recalled = episodic_batch(core, h0, goal_vector, view)[0]
    x = torch.cat(
        [
            state_normalize(h0),
            event.unsqueeze(1).expand_as(h0),
            message,
            fast,
            recalled,
            goal_vector.unsqueeze(1).expand_as(h0),
            nu.unsqueeze(1).expand_as(h0),
        ],
        -1,
    )
    pre = torch.einsum("ndp,bnp->bnd", const["effective"], x)
    pre = pre + torch.einsum(
        "ndm,bnm->bnd",
        core.P[:, :, :parent_units],
        torch.tanh(torch.einsum("nmp,bnp->bnm", core.R[:, :parent_units, :], x)),
    )
    if parent_units < core.R.shape[1]:
        activated = torch.tanh(torch.einsum("nmp,bnp->bnm", core.R[:, parent_units:, :], x))
        activated = activated * gate_rows.unsqueeze(0)
        pre = pre + torch.einsum("ndm,bnm->bnd", core.P[:, :, parent_units:], activated)
    leak = const["leak"].unsqueeze(0).unsqueeze(-1)
    h = (1 - leak) * h0 + leak * torch.tanh(pre)

    mode_vector = F.one_hot(mode, 7).to(reference.dtype)
    previous = core.encoder.embedding(previous_byte)
    xi = core.xi(torch.cat([goal_vector, previous, mode_vector, nu], -1))
    z, query_weights, _ = readout_batch(core, h, xi)
    r_task = torch.einsum("bn,bnd->bd", query_weights.mean(1), recalled)
    probabilities, copied, _, _, _ = expression_batch(
        core, z, r_task, goal_vector, previous, view, coverage, prior
    )
    memory, _, _ = write_fast_batch(
        core,
        h,
        states.fast,
        event,
        goal_vector,
        nu,
        mode_vector,
        past_error,
        error_missing,
        allowed,
    )
    child = BatchedLiving(
        subject_id=states.subject_id,
        revision=states.revision,
        step=states.step + 1,
        h=h,
        fast=memory,
        nu_ema=ema,
        keys=view.keys,
        coverage=coverage,
        prior=prior,
    )
    return BatchedStep(child, probabilities, copied, view)


def consume_batch(
    states: BatchedLiving, output: BatchedStep, byte: torch.Tensor, shift: torch.Tensor
):
    view = output.sources
    matched = output.copy_weights * (view.byte_ids == byte.unsqueeze(1))
    mass = matched.sum(-1)
    has = (byte != EOS) & (mass > 0)
    posterior = matched / mass.clamp_min(1e-300).unsqueeze(1)
    gathered = torch.gather(posterior, 1, shift.clamp_min(0))
    shifted = torch.where(shift >= 0, gathered, torch.zeros_like(gathered))
    total = shifted.sum(-1)
    keep = total > 0
    denom = torch.where(keep, total, torch.ones_like(total))
    shifted = torch.where(keep.unsqueeze(1), shifted / denom.unsqueeze(1), shifted)
    coverage = torch.where(has.unsqueeze(1), states.coverage + posterior, states.coverage)
    prior = torch.where(has.unsqueeze(1), shifted, torch.zeros_like(shifted))
    return BatchedLiving(
        subject_id=states.subject_id,
        revision=states.revision,
        step=states.step,
        h=states.h,
        fast=states.fast,
        nu_ema=states.nu_ema,
        keys=states.keys,
        coverage=coverage,
        prior=prior,
    )
