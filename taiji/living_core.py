"""R021 formal neural forward reference; no trainer, growth or lifecycle service.

The engineering interfaces are frozen in SPEC-TAIJI-CORE_REFERENCE_GRAPH_R021.
Persistent tensors are caller-owned. No serialized latent or parser selects an answer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
from torch import nn
from torch.nn import functional as F

from taiji.sc_sfk import AddressTable

R021_SCHEMA = "taiji-living-core-r021-v1"
EOS = 256
ADDRESSED_SLOTS = 32
ADDRESSED_TEMPERATURE = 0.1
ADDRESSED_ALPHA = 0.5


def _pack_fast(fast: torch.Tensor) -> bytes:
    """Byte-exact snapshot of ``fast``; the SC-SFK store's value payload."""
    return fast.detach().to("cpu").contiguous().numpy().tobytes()


def _unpack_fast(blob: bytes, reference: torch.Tensor) -> torch.Tensor:
    """Rebuild a ``fast`` snapshot and reject a shape/dtype mismatch explicitly."""
    expected = reference.numel() * reference.element_size()
    if len(blob) != expected:
        raise ValueError(f"SC-SFK value carries {len(blob)} bytes, the live fast needs {expected}")
    flat = torch.frombuffer(bytearray(blob), dtype=reference.dtype)
    return flat.view(reference.shape).to(reference.device)


@dataclass(frozen=True)
class LivingCoreConfig:
    groups: int = 4
    width: int = 128
    units: int = 256
    adapter_rank: int = 16
    fast_heads: int = 2
    key_width: int = 32
    value_width: int = 32
    output_heads: int = 2
    fast_limit: float = 16.0
    max_event_bytes: int = 512
    max_source_bytes: int = 512
    max_target_bytes: int = 128
    seed: int = 20261031
    # Content-conditioned write keys (FEC-1/W1(a), default off).  With the flag
    # off the reference is bit-identical to the pre-change graph; the projection
    # parameter exists only when enabled (registered post construction).
    content_keys_enabled: bool = False
    content_gain: float = 1.0
    content_centered: bool = False
    content_path_v2: bool = False
    # Read gate (default off).  ``None`` keeps ``addressed_read`` bit-identical
    # to the pre-change graph: every occupied store returns its top slot and is
    # blended into ``fast``.  A float enables the explicit abstention branch --
    # a query whose top cosine is below the gate returns no slot and leaves
    # ``fast`` untouched.  The value must be anchored on the positive control
    # (the lowest cosine of an already-written read), never on negative labels.
    read_gate_cosine: float | None = None
    # Shared-carrier sparse-feature addressing (SC-SFK, default off).  With the flag
    # off, ``attach_store``/``clear_store``/``addressed_read``/``addressed_write``
    # take the legacy path verbatim and no table is allocated.  With it on, the
    # address layer is the pure-Python ``AddressTable`` (no tensor, no parameter);
    # the FEC-1 value path is not blended into ``fast`` in this round.
    sc_sfk_enabled: bool = False

    def __post_init__(self) -> None:
        if not 1 <= self.groups <= 4 or self.width < 2 or self.width % 2:
            raise ValueError("R021 requires 1..4 groups and an even communication width")
        if (
            min(
                self.units,
                self.adapter_rank,
                self.fast_heads,
                self.key_width,
                self.value_width,
                self.output_heads,
                self.max_event_bytes,
                self.max_source_bytes,
                self.max_target_bytes,
            )
            < 1
            or self.fast_limit <= 0
        ):
            raise ValueError("R021 dimensions and bounds must be positive")
        if self.read_gate_cosine is not None and not (
            math.isfinite(self.read_gate_cosine) and -1.0 <= self.read_gate_cosine <= 1.0
        ):
            raise ValueError("read gate must be a finite cosine in [-1, 1]")


@dataclass(frozen=True)
class RawSource:
    source_id: str
    raw: bytes
    observed_at: int = 0
    version: int = 0
    owner: str = "owner"
    scope: str = "scope"
    real: bool = True


@dataclass(frozen=True)
class SourceView:
    keys: tuple[tuple[str, int], ...]
    byte_ids: torch.Tensor
    positions: torch.Tensor
    positional: torch.Tensor
    metadata: torch.Tensor


@dataclass(frozen=True)
class LivingState:
    subject_id: str
    revision: int
    step: int
    h: torch.Tensor
    fast: torch.Tensor
    nu_ema: torch.Tensor
    keys: tuple[tuple[str, int], ...]
    coverage: torch.Tensor
    prior: torch.Tensor
    # Content-addressed per-turn store (20.3a).  ``None`` for every 19.x and
    # 20.1/20.2 assembly; only ``addressed_store_enabled`` allocates it.
    store_keys: torch.Tensor | None = None
    store_values: torch.Tensor | None = None
    store_age: torch.Tensor | None = None
    store_count: torch.Tensor | None = None
    # SC-SFK address table (pure Python, no tensor).  Only
    # ``sc_sfk_enabled`` allocates it; every other assembly keeps ``None``.
    sc_sfk: object | None = None


@dataclass(frozen=True)
class LivingStep:
    state: LivingState
    probabilities: torch.Tensor
    copy_weights: torch.Tensor
    sources: SourceView
    prediction: torch.Tensor
    predicted_reward: torch.Tensor
    trace: dict[str, torch.Tensor]


def state_normalize(value: torch.Tensor) -> torch.Tensor:
    return value / torch.sqrt(1 + value.square().mean(dim=-1, keepdim=True))


def unit_ball(value: torch.Tensor) -> torch.Tensor:
    return value / torch.sqrt(1 + value.square().sum(dim=-1, keepdim=True))


def position_features(length: int, width: int, reference: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(length, device=reference.device, dtype=reference.dtype)[:, None]
    rates = torch.exp(
        -math.log(10000)
        * torch.arange(0, width, 2, device=reference.device, dtype=reference.dtype)
        / width
    )
    return torch.stack([torch.sin(positions * rates), torch.cos(positions * rates)], -1).flatten(1)


class BytePatchEncoder(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(257, 64)
        self.input = nn.Linear(64, width)
        self.convolutions = nn.ModuleList(
            nn.Conv1d(width, width, 3, dilation=dilation) for dilation in (1, 2, 4)
        )
        self.event = nn.Linear(width, width)
        self.goal = nn.Linear(width, width)

    def context(self, raw: bytes) -> torch.Tensor:
        ids = torch.tensor(list(raw), dtype=torch.long, device=self.embedding.weight.device)
        if not raw:
            return self.input.weight.new_empty((0, self.input.out_features))
        value = self.input(self.embedding(ids)).T.unsqueeze(0)
        for convolution in self.convolutions:
            value = value + torch.tanh(convolution(F.pad(value, (2 * convolution.dilation[0], 0))))
        return value.squeeze(0).T

    def patches(self, raw: bytes) -> torch.Tensor:
        context = self.context(raw)
        if not raw:
            return context
        return self.event(torch.stack([block.mean(0) for block in context.split(8)]))

    def goal_vector(self, raw: bytes) -> torch.Tensor:
        context = self.context(raw)
        return (
            self.goal(context.mean(0))
            if raw
            else self.goal.weight.new_zeros(self.goal.out_features)
        )


class EpisodicRead(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.a = nn.Linear(width, width, bias=False)
        self.h = nn.Linear(width, width, bias=False)
        self.g = nn.Linear(width, width, bias=False)
        self.meta = nn.Linear(4, width)
        self.second = nn.Linear(width, width)
        self.score = nn.Linear(width, 1)
        self.value = nn.Linear(width, width)
        self.value_meta = nn.Linear(4, width, bias=False)
        self.null_key = nn.Parameter(torch.randn(width) / math.sqrt(width))
        self.null_value = nn.Parameter(torch.randn(width) / math.sqrt(width))

    def forward(self, h: torch.Tensor, goal: torch.Tensor, view: SourceView):
        keys = torch.cat([view.positions, self.null_key[None]], dim=0)
        metadata = torch.cat([view.metadata, view.metadata.new_zeros((1, 4))], dim=0)
        first = self.a(keys)[None] + self.meta(metadata)[None]
        first = first + self.h(state_normalize(h))[:, None] + self.g(goal)[None, None]
        scores = self.score(torch.tanh(self.second(torch.tanh(first)))).squeeze(-1)
        attention = torch.softmax(scores, dim=-1)
        values = torch.cat(
            [self.value(view.positions) + self.value_meta(view.metadata), self.null_value[None]], 0
        )
        return attention @ values, attention


class QueryRead(nn.Module):
    def __init__(self, groups: int, width: int, heads: int) -> None:
        super().__init__()
        self.query = nn.ModuleList(nn.Linear(width, width, bias=False) for _ in range(heads))
        self.keys = nn.ModuleList(
            nn.Linear(width, width, bias=False) for _ in range(groups * heads)
        )
        self.values = nn.ModuleList(
            nn.Linear(width, width, bias=False) for _ in range(groups * heads)
        )
        self.output = nn.Linear(heads * width, width, bias=False)
        self.groups, self.width = groups, width

    def forward(self, h: torch.Tensor, xi: torch.Tensor):
        heads, weights = [], []
        for head, query in enumerate(self.query):
            keys = torch.stack(
                [self.keys[head * self.groups + i](h[i]) for i in range(self.groups)]
            )
            values = torch.stack(
                [self.values[head * self.groups + i](h[i]) for i in range(self.groups)]
            )
            attention = torch.softmax(keys @ query(xi) / math.sqrt(self.width), dim=0)
            heads.append(attention @ values)
            weights.append(attention)
        joined = torch.cat(heads, dim=-1)
        return self.output(joined), torch.stack(weights), joined


class CopyExpression(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.z = nn.Linear(width, width, bias=False)
        self.a = nn.Linear(width, width, bias=False)
        self.meta = nn.Linear(4, width, bias=False)
        self.position = nn.Linear(width, width, bias=False)
        self.goal = nn.Linear(width, width)
        self.coverage = nn.Parameter(torch.randn(width) / math.sqrt(width))
        self.score = nn.Linear(width, 1, bias=False)
        self.generate = nn.Linear(width + 64, 257)
        self.mix = nn.Linear(2 * width + 64, 1)
        self.continuation = nn.Linear(width + 64, 1)

    def forward(self, z, recalled, goal, previous, view, coverage, prior):
        generated = torch.softmax(self.generate(torch.cat([z, previous])), -1)
        mixing = torch.sigmoid(self.mix(torch.cat([z, recalled, previous]))).squeeze(0)
        continuation = torch.sigmoid(self.continuation(torch.cat([z, previous]))).squeeze(0)
        if not view.keys:
            return (
                generated,
                generated.new_empty(0),
                generated.new_ones(()),
                continuation,
                generated,
            )
        score_input = self.z(z) + self.a(view.positions) + self.meta(view.metadata)
        score_input = score_input + self.position(view.positional) + self.goal(goal)
        score_input = score_input + coverage[:, None] * self.coverage
        base = torch.softmax(self.score(torch.tanh(score_input)).squeeze(-1), 0)
        weights = (
            (1 - continuation) * base + continuation * prior if bool(prior.sum() > 0) else base
        )
        copied = generated.new_zeros(257).scatter_add(0, view.byte_ids, weights)
        return mixing * generated + (1 - mixing) * copied, weights, mixing, continuation, generated


class LivingCoreReference(nn.Module):
    def __init__(self, config: LivingCoreConfig = LivingCoreConfig()) -> None:
        super().__init__()
        self.config = config
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            self._build()

    def _build(self) -> None:
        c = self.config
        n, d, p = c.groups, c.width, 7 * c.width
        self.encoder = BytePatchEncoder(d)
        self.nu_projection = nn.Linear(16, d)
        self.message_q = nn.ModuleList(nn.Linear(d, d, bias=False) for _ in range(n))
        self.message_k = nn.ModuleList(nn.Linear(d, d, bias=False) for _ in range(n))
        self.message_v = nn.ModuleList(nn.Linear(d, d, bias=False) for _ in range(n))
        self.message_goal = nn.Linear(d, d, bias=False)
        self.fast_queries = nn.ModuleList(
            nn.Linear(4 * d, c.key_width) for _ in range(n * c.fast_heads)
        )
        self.fast_outputs = nn.ModuleList(
            nn.Linear(c.fast_heads * c.value_width, d) for _ in range(n)
        )
        self.episodic = EpisodicRead(d)
        self.W = nn.Parameter(torch.randn(n, d, p) / math.sqrt(p))
        self.U = nn.Parameter(torch.zeros(n, d, c.adapter_rank))
        self.V = nn.Parameter(torch.randn(n, p, c.adapter_rank) / math.sqrt(p))
        self.R = nn.Parameter(torch.randn(n, c.units, p) / math.sqrt(p))
        self.P = nn.Parameter(torch.randn(n, d, c.units) / math.sqrt(c.units))
        initial_tau = torch.tensor([1.0, 4.0, 16.0, 64.0][:n])
        self.tau_logit = nn.Parameter(torch.logit((initial_tau - 0.5) / 127.5))
        self.xi = nn.Linear(2 * d + 64 + 7, d)
        self.readout = QueryRead(n, d, c.output_heads)
        self.expression = CopyExpression(d)
        outputs = c.fast_heads * (3 * c.key_width + 2 * c.value_width + 2)
        self.write_heads = nn.ModuleList(
            nn.Sequential(nn.Linear(4 * d + 9, d), nn.Tanh(), nn.Linear(d, outputs))
            for _ in range(n)
        )
        with torch.no_grad():
            for network in self.write_heads:
                biases = network[-1].bias.view(c.fast_heads, -1)
                biases[:, c.key_width + c.value_width : 2 * c.key_width + c.value_width] = 4
                biases[:, -1] = -2
        self.world = nn.ModuleList(
            nn.Sequential(nn.Linear(2 * d, d), nn.Tanh(), nn.Linear(d, d + 1)) for _ in range(3)
        )

    def initial_state(self, subject_id: str, revision: int = 0) -> LivingState:
        if not subject_id:
            raise ValueError("The reference requires a subject identity")
        c, ref = self.config, self.W
        return LivingState(
            subject_id,
            revision,
            0,
            ref.new_zeros((c.groups, c.width)),
            ref.new_zeros((c.groups, c.fast_heads, c.key_width, c.value_width)),
            ref.new_zeros(8),
            (),
            ref.new_empty(0),
            ref.new_empty(0),
        )

    def source_view(
        self, records: tuple[RawSource, ...], owner: str, scope: str, now: int
    ) -> SourceView:
        c, ref = self.config, self.W
        if len({record.source_id for record in records}) != len(records):
            raise ValueError("Source identities must be unique")
        legal = [
            r
            for r in records
            if r.owner == owner and r.scope == scope and r.observed_at <= now and r.real
        ]
        if sum(len(r.raw) for r in legal) > c.max_source_bytes:
            raise ValueError("R021 source-byte budget exceeded")
        keys, ids, positions, positional, metadata = [], [], [], [], []
        for record in legal:
            if record.observed_at < 0 or record.version < 0 or not record.source_id:
                raise ValueError("Malformed source metadata")
            if not record.raw:
                continue
            context = self.encoder.context(record.raw)
            pe = position_features(len(record.raw), c.width, ref)
            positions.append(context + pe)
            positional.append(pe)
            time = ref.new_tensor(
                [(now - record.observed_at) / 64, record.observed_at / 64, record.version / 16]
            ).tanh()
            metadata.append(torch.cat([time, ref.new_ones(1)]).expand(len(record.raw), -1))
            ids.extend(record.raw)
            keys.extend((record.source_id, p) for p in range(len(record.raw)))
        return SourceView(
            tuple(keys),
            torch.tensor(ids, dtype=torch.long, device=ref.device),
            torch.cat(positions) if positions else ref.new_empty((0, c.width)),
            torch.cat(positional) if positional else ref.new_empty((0, c.width)),
            torch.cat(metadata) if metadata else ref.new_empty((0, 4)),
        )

    def update_nu(self, state: LivingState, raw: torch.Tensor, known: torch.Tensor):
        if raw.shape != (8,) or known.shape != (8,) or known.dtype != torch.bool:
            raise ValueError("nu requires eight readings and explicit boolean availability")
        if not bool(torch.isfinite(raw[known]).all()):
            raise ValueError("Known readings must be finite")
        safe = torch.where(known, raw, torch.zeros_like(raw))
        smoothed = 0.95 * state.nu_ema + 0.05 * safe
        direct = torch.tensor(
            [True, False, False, False, True, False, False, False], device=raw.device
        )
        ema = torch.where(known, torch.where(direct, safe, smoothed), state.nu_ema)
        scale = raw.new_tensor([1, 1, 1, 1, 16, 1, 1, 1])
        value = torch.where(known, torch.tanh(ema / scale), torch.zeros_like(ema))
        features = torch.cat([value, (~known).to(raw.dtype)])
        return ema, self.nu_projection(features), features

    def messages(self, h: torch.Tensor, goal: torch.Tensor):
        q = torch.stack([layer(row) for layer, row in zip(self.message_q, h, strict=True)])
        q = q + self.message_goal(goal)
        k = torch.stack([layer(row) for layer, row in zip(self.message_k, h, strict=True)])
        v = torch.stack([layer(row) for layer, row in zip(self.message_v, h, strict=True)])
        attention = torch.softmax(q @ k.T / math.sqrt(self.config.width), dim=-1)
        return attention @ v, attention, q, k, v

    def tied_read_keys(self, h, event, goal, nu, past_error, error_missing, mode):
        """Component 2 (v2 window): the read query reuses the write-addressing
        function - the write-head key segment evaluated at the read's own
        context.  Returns (groups, fast_heads, key_width), boost NOT applied."""
        c = self.config
        keys = []
        for group, network in enumerate(self.write_heads):
            context = torch.cat(
                [
                    state_normalize(h[group]),
                    event,
                    goal,
                    nu,
                    past_error.reshape(1),
                    event.new_tensor([error_missing]),
                    mode,
                ]
            )
            raw = network(context).view(c.fast_heads, -1)
            keys.append(unit_ball(raw[:, : c.key_width]))
        return torch.stack(keys)

    def read_fast(self, h, memory, event, goal, nu, content_feature=None, query_override=None):
        c = self.config
        inputs = torch.cat(
            [state_normalize(h), event.expand_as(h), goal.expand_as(h), nu.expand_as(h)], dim=-1
        )
        content_boost = None
        if content_feature is not None:
            projection = getattr(self, "key_content_proj", None)
            if projection is None:
                raise ValueError("Content-conditioned keys require the registered projection")
            content_boost = unit_ball(projection @ content_feature)
        queries, values, merged = [], [], []
        for group in range(c.groups):
            if query_override is not None:
                q = query_override[group]
            else:
                q = torch.stack(
                    [
                        unit_ball(self.fast_queries[group * c.fast_heads + head](inputs[group]))
                        for head in range(c.fast_heads)
                    ]
                )
            if content_boost is not None:
                q = unit_ball(q + c.content_gain * content_boost)
            value = torch.einsum("hkv,hk->hv", memory[group], q)
            queries.append(q)
            values.append(value)
            merged.append(self.fast_outputs[group](value.flatten()))
        return torch.stack(merged), torch.stack(queries), torch.stack(values)

    @staticmethod
    def edit_fast(memory, key, value, decay, erase, write, strength, chi, allowed, limit):
        if not allowed:
            return memory
        decayed = decay[:, None] * memory
        direction = erase * key
        eta = 16 * torch.sigmoid(strength)
        factor = eta / (1 + eta * direction.square().sum())
        erased = decayed - factor * torch.outer(direction, direction @ decayed)
        edited = erased + torch.outer(key, write * value)
        candidate = (1 - chi) * memory + chi * edited
        norm = torch.linalg.vector_norm(candidate)
        return candidate if bool(norm <= limit) else candidate * (limit / norm)

    def write_fast(
        self,
        h,
        memory,
        event,
        goal,
        nu,
        mode,
        past_error,
        error_missing,
        allowed,
        content_feature=None,
    ):
        c = self.config
        if allowed.shape != (c.groups, c.fast_heads) or allowed.dtype != torch.bool:
            raise ValueError("Fast write permissions require a per-group/per-head boolean mask")
        content_boost = None
        if content_feature is not None:
            projection = getattr(self, "key_content_proj", None)
            if projection is None:
                raise ValueError("Content-conditioned keys require the registered projection")
            content_boost = unit_ball(projection @ content_feature)
        result, all_chi, all_keys = [], [], []
        for group, network in enumerate(self.write_heads):
            context = torch.cat(
                [
                    state_normalize(h[group]),
                    event,
                    goal,
                    nu,
                    past_error.reshape(1),
                    event.new_tensor([error_missing]),
                    mode,
                ]
            )
            raw = network(context).view(c.fast_heads, -1)
            splits = (c.key_width, c.value_width, c.key_width, c.key_width, c.value_width, 1, 1)
            key, value, decay, erase, write, strength, chi = raw.split(splits, dim=-1)
            key, value, chi = unit_ball(key), unit_ball(torch.tanh(value)), torch.sigmoid(chi)
            if content_boost is not None:
                key = unit_ball(key + c.content_gain * content_boost)
                if getattr(c, "content_path_v2", False):
                    value = unit_ball(value + c.content_gain * content_boost)
            heads = [
                self.edit_fast(
                    memory[group, head],
                    key[head],
                    value[head],
                    torch.sigmoid(decay[head]),
                    torch.sigmoid(erase[head]),
                    torch.sigmoid(write[head]),
                    strength[head, 0],
                    chi[head, 0],
                    bool(allowed[group, head]),
                    c.fast_limit,
                )
                for head in range(c.fast_heads)
            ]
            result.append(torch.stack(heads))
            all_chi.append(chi.squeeze(-1))
            all_keys.append(key)
        return torch.stack(result), torch.stack(all_chi), torch.stack(all_keys)

    # ---- content-addressed per-turn store (20.3a, default off) --------------
    def empty_store(self, reference: torch.Tensor | None = None):
        ref = self.W if reference is None else reference
        c = self.config
        return (
            ref.new_zeros((ADDRESSED_SLOTS, c.width)),
            ref.new_zeros((ADDRESSED_SLOTS, c.groups, c.fast_heads, c.key_width, c.value_width)),
            torch.full((ADDRESSED_SLOTS,), -1, dtype=torch.long, device=ref.device),
            torch.zeros((), dtype=torch.long, device=ref.device),
        )

    def attach_store(self, state: LivingState) -> LivingState:
        if self.config.sc_sfk_enabled:
            if state.sc_sfk is None:
                return replace(state, sc_sfk=AddressTable())
            return state
        if state.store_keys is not None:
            return state
        keys, values, age, count = self.empty_store()
        return replace(
            state, store_keys=keys, store_values=values, store_age=age, store_count=count
        )

    def clear_store(self, state: LivingState) -> LivingState:
        if self.config.sc_sfk_enabled:
            if state.sc_sfk is None:
                return state
            return replace(state, sc_sfk=AddressTable())
        if state.store_keys is None:
            return state
        keys, values, age, count = self.empty_store()
        return replace(
            state, store_keys=keys, store_values=values, store_age=age, store_count=count
        )

    def store_query_key(self, state: LivingState, raw: bytes) -> torch.Tensor:
        context = self.encoder.context(raw)
        if context.numel() == 0:
            return state.fast.new_zeros(self.config.width)
        return context.mean(0)

    def addressed_read(self, state: LivingState, raw: bytes, key: torch.Tensor | None = None):
        """Read the content-addressed store.

        ``key`` is a diagnostic entry only: an already-encoded query key may be
        handed in instead of being derived from ``raw``.  Production callers
        never pass it, so the two-argument call is unchanged.

        ``matched`` is always present.  With the gate off (``None``) it is
        ``True`` in every branch and the returned state is bit-identical to the
        pre-gate graph.  With the gate on, an occupied store whose top cosine is
        below ``read_gate_cosine`` returns ``matched=False``, no ``top_slot``
        and the *unmodified* state -- the explicit "no match" exit.
        """
        if self.config.sc_sfk_enabled:
            return self._sc_sfk_read(state, raw)
        key = self.store_query_key(state, raw) if key is None else key
        gate = self.config.read_gate_cosine
        gate_info = {} if gate is None else {"gate_threshold": float(gate)}
        if state.store_keys is None:
            return state, {
                "enabled": False,
                "blended": False,
                "key": key,
                "matched": True,
                **gate_info,
            }
        valid = torch.nonzero(state.store_age >= 0, as_tuple=False).flatten()
        if valid.numel() == 0:
            info = {
                "enabled": True,
                "occupied": 0,
                "valid": 0,
                "blended": False,
                "key": key,
                "matched": gate is None,
                **gate_info,
            }
            if gate is not None:
                info.update({"abstained": True, "reason": "empty_store"})
            return state, info
        keys = state.store_keys[valid]
        cosine = (keys @ key) / torch.clamp(keys.norm(dim=-1) * key.norm(), min=1e-12)
        weights = torch.softmax(cosine / ADDRESSED_TEMPERATURE, dim=0)
        ordered = torch.sort(cosine.detach(), descending=True).values
        top = int(torch.argmax(cosine.detach()))
        top_cosine = float(cosine[top].detach())
        margin = float(ordered[0] - ordered[1]) if ordered.numel() > 1 else None
        info = {
            "enabled": True,
            "occupied": int(valid.numel()),
            "valid": int(valid.numel()),
            "blended": True,
            "top_slot": int(valid[top]),
            "top_cosine": top_cosine,
            "second_cosine": float(ordered[1]) if ordered.numel() > 1 else None,
            "margin": margin,
            "weights": [float(value.detach()) for value in weights],
            "retrieved_norm": None,
            "fast_norm_before": float(state.fast.detach().norm()),
            "fast_norm_after": None,
            "key": key,
            "matched": True,
            **gate_info,
        }
        if gate is not None and top_cosine < gate:
            info.pop("top_slot")
            for field in ("retrieved_norm", "fast_norm_after"):
                info.pop(field)
            info.update(
                {"blended": False, "matched": False, "abstained": True, "reason": "below_gate"}
            )
            return state, info
        retrieved = torch.einsum("i,i...->...", weights, state.store_values[valid])
        fast = (1 - ADDRESSED_ALPHA) * state.fast + ADDRESSED_ALPHA * retrieved
        info["retrieved_norm"] = float(retrieved.detach().norm())
        info["fast_norm_after"] = float(fast.detach().norm())
        return replace(state, fast=fast), info

    def addressed_write(self, state: LivingState, raw: bytes, key: torch.Tensor | None = None):
        if self.config.sc_sfk_enabled:
            return self._sc_sfk_write(state, raw)
        if state.store_keys is None:
            state = self.attach_store(state)
        if key is None:
            key = self.store_query_key(state, raw)
        slots = int(state.store_keys.shape[0])
        occupied = int((state.store_age >= 0).sum())
        evicted = occupied >= slots
        slot = int(torch.argmin(state.store_age)) if evicted else occupied
        keys = state.store_keys.clone()
        values = state.store_values.clone()
        age = state.store_age.clone()
        replaced = int(age[slot])
        keys[slot] = key
        values[slot] = state.fast
        age[slot] = int(state.store_count)
        count = state.store_count + 1
        info = {
            "slot": slot,
            "insertion_age": int(age[slot]),
            "occupied_after": min(int(count), slots),
            "evicted": bool(evicted),
            "evicted_age": replaced,
        }
        return (
            replace(state, store_keys=keys, store_values=values, store_age=age, store_count=count),
            info,
        )

    def _sc_sfk_read(self, state: LivingState, raw: bytes):
        """SC-SFK read: the address decides, the stored value is blended on a hit.

        ``matched`` is the only branch that touches ``fast``; ``no_match`` and
        ``ambiguous_multiple_anchors`` return the *same* state object, so an
        abstention is bit-identical to not having read at all.
        """
        table = state.sc_sfk
        if table is None:
            return state, {
                "enabled": False,
                "blended": False,
                "matched": False,
                "decision": "no_match",
                "address": None,
                "reason": "no_store",
                "top1": 0.0,
                "margin": 0.0,
            }
        result = table.read(raw)
        info = {
            "enabled": True,
            "blended": False,
            "matched": result["decision"] == "matched",
            "value_bytes": 0,
            "retrieved_norm": None,
            "fast_norm_before": float(state.fast.detach().norm()),
            "fast_norm_after": None,
            **result,
        }
        if result["decision"] != "matched":
            return state, info
        value = result["value"]
        if not isinstance(value, (bytes, bytearray)):
            raise ValueError("SC-SFK store value is not a packed fast snapshot")
        retrieved = _unpack_fast(bytes(value), state.fast)
        fast = (1 - ADDRESSED_ALPHA) * state.fast + ADDRESSED_ALPHA * retrieved
        info.update(
            {
                "blended": True,
                "value_bytes": len(value),
                "retrieved_norm": float(retrieved.detach().norm()),
                "fast_norm_after": float(fast.detach().norm()),
            }
        )
        return replace(state, fast=fast), info

    def _sc_sfk_write(self, state: LivingState, raw: bytes):
        """SC-SFK write: one record, structural, no gradient and no eviction.

        The stored value is the *current fast snapshot*, not the record bytes: that
        is what makes the read path able to feed the next step.  A re-observation
        of the same record with a different fast state is a replace (the old value
        moves into the address history); an identical value is a merge.
        """
        if state.sc_sfk is None:
            state = self.attach_store(state)
        table = state.sc_sfk
        value = _pack_fast(state.fast)
        result = table.write_batch_checked([(raw, value)])
        address = table.target_of(raw)[0]
        return state, {
            "slot": address,
            "address": address,
            "action": result["action"],
            "occupied_after": len(table.addresses),
            "capacity": table.capacity,
            "generation": table.generation,
            "value_bytes": len(value),
            "evicted": False,
            "global_check": result.get("global_check"),
        }

    @staticmethod
    def align_positions(state: LivingState, keys):
        old = {key: index for index, key in enumerate(state.keys)}
        zero = state.h.new_zeros(())
        coverage = (
            torch.stack([state.coverage[old[k]] if k in old else zero for k in keys])
            if keys
            else zero.new_empty(0)
        )
        prior = (
            torch.stack([state.prior[old[k]] if k in old else zero for k in keys])
            if keys
            else zero.new_empty(0)
        )
        if bool(prior.sum() > 0):
            prior = prior / prior.sum()
        return coverage, prior

    def step(
        self,
        state: LivingState,
        *,
        event: torch.Tensor,
        goal: bytes,
        records: tuple[RawSource, ...] = (),
        previous_byte: int = EOS,
        raw_nu: torch.Tensor | None = None,
        known_nu: torch.Tensor | None = None,
        mode: int = 0,
        past_error: float = 0.0,
        error_missing: bool = True,
        allowed: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        owner: str = "owner",
        scope: str = "scope",
        now: int = 0,
    ) -> LivingStep:
        c, ref = self.config, self.W
        if state.h.shape != (c.groups, c.width) or event.shape != (c.width,):
            raise ValueError("State/event dimensions differ from R021")
        if state.fast.shape != (c.groups, c.fast_heads, c.key_width, c.value_width):
            raise ValueError("Fast state dimensions differ from R021")
        if not 0 <= previous_byte <= EOS or not 0 <= mode < 7 or len(goal) > c.max_event_bytes:
            raise ValueError("Invalid preceding byte, mode or goal budget")
        if not bool(
            torch.isfinite(state.h).all()
            and torch.isfinite(state.fast).all()
            and torch.isfinite(event).all()
        ) or bool(state.h.abs().max() > 1):
            raise ValueError("Invalid neural state or event")
        raw_nu = ref.new_zeros(8) if raw_nu is None else raw_nu
        known_nu = (
            torch.zeros(8, dtype=torch.bool, device=ref.device) if known_nu is None else known_nu
        )
        allowed = (
            torch.zeros((c.groups, c.fast_heads), dtype=torch.bool, device=ref.device)
            if allowed is None
            else allowed
        )
        goal_vector = self.encoder.goal_vector(goal)
        content_feature = (
            self.content_feature(goal) if getattr(c, "content_keys_enabled", False) else None
        )
        ema, nu, nu_features = self.update_nu(state, raw_nu, known_nu)
        view = self.source_view(records, owner, scope, now)
        coverage, prior = self.align_positions(state, view.keys)
        message, message_weights, mq, mk, mv = self.messages(state.h, goal_vector)
        read_override = (
            self.tied_read_keys(
                state.h,
                event,
                goal_vector,
                nu,
                past_error,
                error_missing,
                F.one_hot(torch.tensor(mode, device=ref.device), 7).to(ref.dtype),
            )
            if getattr(c, "content_path_v2", False)
            else None
        )
        fast, fast_queries, fast_values = self.read_fast(
            state.h,
            state.fast,
            event,
            goal_vector,
            nu,
            content_feature=content_feature,
            query_override=read_override,
        )
        recalled, episodic_weights = self.episodic(state.h, goal_vector, view)
        x = torch.cat(
            [
                state_normalize(state.h),
                event.expand_as(state.h),
                message,
                fast,
                recalled,
                goal_vector.expand_as(state.h),
                nu.expand_as(state.h),
            ],
            -1,
        )
        effective = self.W + self.U @ self.V.transpose(1, 2)
        pre = torch.einsum("ndp,np->nd", effective, x)
        pre = pre + torch.einsum(
            "ndm,nm->nd", self.P, torch.tanh(torch.einsum("nmp,np->nm", self.R, x))
        )
        tau = 0.5 + 127.5 * torch.sigmoid(self.tau_logit)
        leak = (1 - torch.exp(-1 / tau))[:, None]
        h = (1 - leak) * state.h + leak * torch.tanh(pre)
        mode_vector = F.one_hot(torch.tensor(mode, device=ref.device), 7).to(ref.dtype)
        previous = self.encoder.embedding(torch.tensor(previous_byte, device=ref.device))
        xi = self.xi(torch.cat([goal_vector, previous, mode_vector, nu]))
        z, query_weights, joined = self.readout(h, xi)
        r_task = query_weights.mean(0) @ recalled
        probabilities, copied, mixing, continuation, generated = self.expression(
            z, r_task, goal_vector, previous, view, coverage, prior
        )
        action = ref.new_zeros(c.width) if action is None else action
        predictions = torch.stack([head(torch.cat([z, action])) for head in self.world])
        memory, chi, write_keys = self.write_fast(
            h,
            state.fast,
            event,
            goal_vector,
            nu,
            mode_vector,
            ref.new_tensor(past_error),
            error_missing,
            allowed,
            content_feature=content_feature,
        )
        child = replace(
            state,
            step=state.step + 1,
            h=h,
            fast=memory,
            nu_ema=ema,
            keys=view.keys,
            coverage=coverage,
            prior=prior,
        )
        trace = {
            "event": event,
            "goal": goal_vector,
            "nu": nu,
            "nu_features": nu_features,
            "message": message,
            "message_weights": message_weights,
            "message_q": mq,
            "message_k": mk,
            "message_v": mv,
            "fast_read": fast,
            "fast_queries": fast_queries,
            "fast_values": fast_values,
            "recalled": recalled,
            "episodic_weights": episodic_weights,
            "x": x,
            "preactivation": pre,
            "tau": tau,
            "query": xi,
            "z": z,
            "joined_heads": joined,
            "query_weights": query_weights,
            "mixing": mixing,
            "continuation": continuation,
            "generated": generated,
            "chi": chi,
            "write_keys": write_keys,
        }
        return LivingStep(
            child, probabilities, copied, view, predictions[:, :-1], predictions[:, -1], trace
        )

    def content_feature(self, goal) -> torch.Tensor:
        """Content feature of the goal text (W1(a)); centered mode subtracts the
        registered centroid (W1 round 2) so the feature carries the text's
        distinct bytes rather than the shared template mass."""
        if isinstance(goal, str):
            goal = goal.encode("utf-8")
        raw = goal if isinstance(goal, (bytes, bytearray)) else bytes(goal)
        feature = torch.zeros(128, dtype=torch.float64)
        for value in raw:
            feature[int(value) % 128] += 1.0
        if getattr(self.config, "content_centered", False):
            centroid = getattr(self, "content_centroid", None)
            if centroid is not None:
                delta = feature - centroid
                norm = float(torch.linalg.vector_norm(delta))
                if norm > 0:
                    return delta / norm
        norm = float(torch.linalg.vector_norm(feature))
        return feature / norm if norm > 0 else feature

    @staticmethod
    def consume_output(output: LivingStep, byte: int) -> LivingState:
        if not 0 <= byte <= EOS:
            raise ValueError("Output must be a byte or EOS")
        state, view = output.state, output.sources
        matched = output.copy_weights * (view.byte_ids == byte)
        mass = matched.sum()
        if byte == EOS or not bool(mass > 0):
            return replace(state, prior=torch.zeros_like(state.prior))
        posterior = matched / mass
        old = {key: index for index, key in enumerate(view.keys)}
        zero = posterior.new_zeros(())
        shifted = torch.stack(
            [
                posterior[old[(source, position - 1)]] if (source, position - 1) in old else zero
                for source, position in view.keys
            ]
        )
        if bool(shifted.sum() > 0):
            shifted = shifted / shifted.sum()
        return replace(state, coverage=state.coverage + posterior, prior=shifted)

    def observe(self, state: LivingState, raw: bytes, **kwargs) -> LivingState:
        if len(raw) > self.config.max_event_bytes:
            raise ValueError("R021 event-byte budget exceeded")
        for event in self.encoder.patches(raw):
            state = self.step(state, event=event, **kwargs).state
        return state

    def score_targets(
        self,
        state: LivingState,
        *,
        prompt: bytes,
        target: bytes,
        goal: bytes,
        records: tuple[RawSource, ...] = (),
        **kwargs,
    ):
        if len(target) > self.config.max_target_bytes:
            raise ValueError("R021 target-byte budget exceeded")
        state = self.observe(state, prompt, goal=goal, records=records, **kwargs)
        previous, distributions = EOS, []
        for byte in (*target, EOS):
            event = (
                self.W.new_zeros(self.config.width)
                if previous == EOS
                else self.encoder.patches(bytes([previous]))[0]
            )
            output = self.step(
                state, event=event, goal=goal, records=records, previous_byte=previous, **kwargs
            )
            distributions.append(output.probabilities)
            state = self.consume_output(output, byte)
            previous = byte
        return torch.stack(distributions), state
