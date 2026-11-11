"""U2 same-subject reference graph, promoted from the MGW1 mathematical core.

This isolated candidate shares learned modalities, native state, record reading,
fast memory, expression and development credit. It does not change defaults or
load existing project weights. Frozen MGW1 and U2 22.1 remain independent oracles.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, replace
from typing import Any, cast

import torch
from torch import nn
from torch.nn import functional as functional

from taiji.u2_operators import (
    accept,
    choose_left_directions,
    emission,
    fp1,
    initial_cursor,
    record_inputs,
    utf8_mask,
)


def bounded(x: Any) -> Any:
    return x / (1 + x.square().sum(dim=-1, keepdim=True)).sqrt()


def normal_cdf(x: Any) -> Any:
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))


def truncated_log_prob(y: Any, mean: Any, scale: Any, lower: Any, upper: Any) -> Any:
    normalizer = normal_cdf((upper - mean) / scale) - normal_cdf((lower - mean) / scale)
    density = -0.5 * ((y - mean) / scale).square() - scale.log() - 0.5 * math.log(2 * math.pi)
    return torch.where(
        (y >= lower) & (y <= upper),
        density - normalizer.log(),
        torch.full_like(density, -torch.inf),
    )


@dataclass(frozen=True)
class U2Config:
    d: int = 8
    groups: int = 2
    units: int = 12
    objects: int = 3
    branches: int = 2
    key: int = 3
    side: int = 16
    patch: int = 4
    audio_samples: int = 256
    fft: int = 64
    hop: int = 32
    embedding: int = 4
    max_record_bytes: int = 512
    max_output_bytes: int = 64
    seed: int = 20261110

    def __post_init__(self) -> None:
        if (
            min(
                self.d,
                self.groups,
                self.units,
                self.objects,
                self.branches,
                self.key,
                self.embedding,
                self.side,
                self.patch,
                self.audio_samples,
                self.fft,
                self.hop,
                self.max_record_bytes,
                self.max_output_bytes,
            )
            < 1
        ):
            raise ValueError("positive U2 dimensions required")
        if self.groups > 4 or self.side % self.patch or self.audio_samples < self.fft:
            raise ValueError("invalid modality or group profile")

    @property
    def dimensions(self) -> dict[str, int]:
        return {"rgb": 3 * self.patch**2, "audio": self.fft // 2 + 1, "proprio": 8}

    @classmethod
    def small(cls) -> U2Config:
        return cls()

    @classmethod
    def nominal(cls) -> U2Config:
        return cls(
            d=128,
            groups=4,
            units=256,
            objects=4,
            branches=3,
            key=32,
            side=64,
            patch=8,
            audio_samples=2560,
            fft=256,
            hop=128,
            embedding=64,
        )


DEFAULT_CONFIG = U2Config()


class MLP(nn.Module):
    def __init__(self, inputs: Any, outputs: Any, d: Any) -> None:
        super().__init__()
        self.first = nn.Linear(inputs, 2 * d)
        self.last = nn.Linear(2 * d, outputs)

    def forward(self, x: Any) -> Any:
        return self.last(torch.tanh(self.first(x)))


class BiasQuery(nn.Module):
    """Proprio coordinates are fixed zero in this profile, so only a query is learned."""

    def __init__(self, d: Any) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.zeros(d))

    def forward(self, coordinates: Any) -> Any:
        return self.value.expand(*coordinates.shape[:-1], self.value.numel())


@dataclass(frozen=True)
class U2Record:
    record_id: str
    raw: bytes
    version: int = 0
    source: str = "sensor"
    scope: str = "world"
    observed_at: float = 0.0
    writable: bool = True


@dataclass(frozen=True)
class U2Packet:
    packet_id: str
    source: str
    scope: str
    t_start: float
    t_end: float
    t_arrival: float
    observations: dict[str, torch.Tensor]
    coordinates: dict[str, torch.Tensor]
    outcome: torch.Tensor
    receipt_id: str
    valid: bool = True
    write_allowed: bool = True
    feedback_record_ids: tuple[str, ...] = ()


U2State = dict[str, Any]


def clone_tree(value: Any, *, detach: Any = False) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone() if detach else value.clone()
    if isinstance(value, dict):
        return {key: clone_tree(item, detach=detach) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_tree(item, detach=detach) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_tree(item, detach=detach) for item in value)
    return copy.deepcopy(value)


class _WorldKernel(nn.Module):
    def __init__(self, profile: Any = DEFAULT_CONFIG) -> None:
        super().__init__()
        self.c: U2Config = profile
        c, d = profile, profile.d
        self.encoder = nn.ModuleDict(
            {
                m: nn.Linear(size + (8 if m == "proprio" else 0), d)
                for m, size in c.dimensions.items()
            }
        )
        self.metadata = nn.Linear(8, d)
        self.modality = nn.Parameter(torch.randn(5, d) * 0.05)
        self.embedding = nn.Embedding(256, c.embedding)
        self.text_input = nn.Linear(c.embedding, d)
        self.convolutions = nn.ModuleList(nn.Conv1d(d, d, 3, dilation=x) for x in (1, 2, 4))
        self.goal_score = nn.Linear(d, 1, bias=False)
        self.goal_value = nn.Linear(d, d)
        self.initial_h = nn.Parameter(torch.randn(c.branches, c.objects, d) * 0.1)
        self.initial_p = nn.Parameter(torch.randn(c.branches, c.objects, 2) * 0.2)
        self.object_projection = nn.Linear(d + 9, d)
        self.h_query = nn.Linear(d, d)
        self.h_key = nn.Linear(d, d, bias=False)
        self.h_value = nn.Linear(d, d)
        self.o_query = nn.Linear(d, d)
        self.o_key = nn.Linear(d, d, bias=False)
        self.o_value = nn.Linear(d, d)
        self.action = nn.Linear(15, d)
        self.f_query = nn.Linear(4 * d, c.key)
        self.f_output = nn.Linear(c.key, d)
        self.relation_query = nn.Linear(d, d)
        self.relation_key = nn.Linear(d, d, bias=False)
        self.relation_value = nn.Linear(d + 4, d)
        self.relation_geometry = MLP(4, 1, d)
        self.motion = MLP(5 * d + 5, d + 6, d)
        self.native_w = nn.Parameter(torch.randn(c.groups, d, 8 * d) * 0.1 / math.sqrt(8 * d))
        self.native_r = nn.Parameter(torch.randn(c.groups, c.units, 8 * d) * 0.1 / math.sqrt(8 * d))
        self.native_p = nn.Parameter(torch.randn(c.groups, d, c.units) * 0.1 / math.sqrt(c.units))
        self.native_q = nn.Parameter(torch.randn(c.groups, d, d) * 0.2 / math.sqrt(d))
        self.native_k = nn.Parameter(torch.randn(c.groups, d, d) * 0.2 / math.sqrt(d))
        self.native_v = nn.Parameter(torch.randn(c.groups, d, d) * 0.2 / math.sqrt(d))
        self.native_goal = nn.Linear(d, d)
        self.nu_projection = nn.Linear(16, d)
        tau = torch.tensor([1.0, 4.0, 16.0, 64.0][: c.groups])
        ratio = (tau - 0.5) / 127.5
        self.tau_logit = nn.Parameter(torch.logit(ratio))
        self.bind_query = nn.Linear(d, d)
        binding_modes = [*c.dimensions, "outcome"]
        self.bind_key = nn.ModuleDict({m: nn.Linear(d, d) for m in binding_modes})
        self.bind_value = nn.ModuleDict({m: nn.Linear(d, d) for m in binding_modes})
        self.null_score = nn.ModuleDict({m: nn.Linear(d, 1) for m in binding_modes})
        self.reliability = nn.ModuleDict({m: nn.Linear(d, 1) for m in binding_modes})
        self.corrector = MLP(3 * d + 9, d + 9, d)
        self.f_key = nn.Linear(4 * d, c.key)
        self.outcome = nn.Linear(6, d)
        self.f_value = MLP(2 * d, c.key, d)
        self.f_beta = MLP(2 * d, 1, d)
        self.f_alpha = MLP(2 * d, c.key, d)
        self.f_chi = MLP(2 * d, 1, d)
        self.post_memory = MLP(3 * d, d, d)
        self.background = nn.Parameter(torch.randn(d) * 0.05)
        self.decoder_query = nn.ModuleDict(
            {m: (BiasQuery(d) if m == "proprio" else nn.Linear(8, d)) for m in c.dimensions}
        )
        self.decoder_key = nn.Linear(d, d, bias=False)
        self.decoder_value = nn.Linear(d, d)
        self.decoder = nn.ModuleDict(
            {m: MLP(d + 8, 2 * size, d) for m, size in c.dimensions.items()}
        )
        self.joint_readout = MLP(3 * d, d, d)
        self.byte_head = nn.Linear(d + c.embedding, 257)
        self.reward = MLP(d + 15, 4, d)
        self.actor = MLP(2 * d + 18, 1, d)
        self.actor.last.register_parameter("bias", None)
        self.relation_geometry.last.register_parameter("bias", None)
        self.critic = MLP(2 * d, 1, d)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.normal_(module.weight, std=0.3 / math.sqrt(module.weight[0].numel()))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)

    def initial(self) -> U2State:
        c = self.c
        factory = self.initial_h
        return {
            "h": self.initial_h.tanh(),
            "p": self.initial_p.tanh(),
            "v": factory.new_zeros(c.branches, c.objects, 2),
            "s": factory.new_full((c.branches, c.objects, 4), 0.1),
            "l": factory.new_zeros(c.branches, c.objects, 1),
            "H": factory.new_zeros(c.branches, c.groups, c.d),
            "F": factory.new_zeros(c.branches, c.key, c.key),
            "logw": factory.new_full((c.branches,), -math.log(c.branches)),
        }

    def packed(self, state: Any) -> Any:
        return torch.cat([state[x] for x in ("h", "p", "v")] + [state["s"].log(), state["l"]], -1)

    def object_z(self, state: Any) -> Any:
        return bounded(self.object_projection(self.packed(state)))

    def read_h(self, state: Any, goal: Any) -> Any:
        score = (self.h_key(state["H"]) * self.h_query(goal)).sum(-1) / math.sqrt(self.c.d)
        return (score.softmax(-1)[..., None] * self.h_value(state["H"])).sum(-2)

    def read_o(self, state: Any, query: Any) -> Any:
        z = torch.cat(
            [
                self.object_z(state),
                bounded(self.background)[None, None].expand(self.c.branches, 1, -1),
            ],
            1,
        )
        score = (self.o_key(z) * self.o_query(query)[:, None]).sum(-1) / math.sqrt(self.c.d)
        score = score + torch.cat(
            [
                functional.logsigmoid(state["l"]).squeeze(-1),
                state["l"].new_zeros(self.c.branches, 1),
            ],
            -1,
        )
        return (score.softmax(-1)[..., None] * self.o_value(z)).sum(-2)

    def views(self, state: Any, goal: Any, action: Any) -> Any:
        c = self.c
        return torch.cat(
            [
                self.object_z(state),
                self.read_h(state, goal)[:, None].expand(-1, c.objects, -1),
                goal[None, None].expand(c.branches, c.objects, -1),
                self.action(action)[None, None].expand(c.branches, c.objects, -1),
            ],
            -1,
        )

    def read_f(self, state: Any, goal: Any, action: Any) -> Any:
        query = bounded(self.f_query(self.views(state, goal, action)))
        return self.f_output(torch.einsum("bkv,bmk->bmv", state["F"], query))

    def predict(self, state: Any, goal: Any, action: Any, dt: Any = 0.05) -> Any:
        c = self.c
        result = dict(state)
        z = self.object_z(state)
        relative = torch.cat(
            [
                state["p"][:, None] - state["p"][:, :, None],
                state["v"][:, None] - state["v"][:, :, None],
            ],
            -1,
        )
        scores = self.relation_query(z) @ self.relation_key(z).transpose(-1, -2) / math.sqrt(c.d)
        scores = scores + self.relation_geometry(relative).squeeze(-1)
        scores = scores + functional.logsigmoid(state["l"]).transpose(-1, -2)
        scores = scores.masked_fill(torch.eye(c.objects, dtype=torch.bool)[None], -torch.inf)
        messages = bounded(
            self.relation_value(
                torch.cat([state["h"][:, None].expand(-1, c.objects, -1, -1), relative], -1)
            )
        )
        message = (
            (scores.softmax(-1)[..., None] * messages).sum(-2)
            if c.objects > 1
            else torch.zeros_like(z)
        )
        u = torch.cat(
            [
                state["h"],
                message,
                self.read_f(state, goal, action),
                goal[None, None].expand(c.branches, c.objects, -1),
                self.action(action)[None, None].expand(c.branches, c.objects, -1),
                state["p"],
                state["v"],
                z.new_full((c.branches, c.objects, 1), dt),
            ],
            -1,
        )
        output = self.motion(u)
        result["h"] = (
            math.exp(-0.25) * state["h"] + (1 - math.exp(-0.25)) * output[..., : c.d].tanh()
        )
        result["v"] = state["v"] + dt * 2 * output[..., c.d : c.d + 2].tanh()
        result["p"] = state["p"] + dt * result["v"]
        for field in ("p", "v"):
            result[field] = result[field] * (
                4 / result[field].norm(dim=-1, keepdim=True).clamp_min(4)
            )
        result["s"] = (state["s"] + dt * (1e-4 + 0.1 * output[..., c.d + 2 :].sigmoid())).clamp(
            1e-4, 4
        )
        result["H"] = self.native(result, goal, action, z.new_zeros(c.branches, c.d))
        result["prediction_goal"] = goal.clone()
        result["prediction_action"] = action.clone()
        return result

    def native(self, state: Any, goal: Any, action: Any, event: Any) -> Any:
        raise NotImplementedError("The subject supplies the native state transition")

    def encode(self, observations: Any, coordinates: Any, outcome: Any = None) -> Any:
        result = {}
        for m, value in observations.items():
            encoded_input = value
            if m == "proprio":
                if not bool(value[..., 8:].any()):
                    continue
                encoded_input = torch.cat(
                    [
                        torch.where(
                            value[..., 8:].bool(), value[..., :8], torch.zeros_like(value[..., :8])
                        ),
                        value[..., 8:],
                    ],
                    -1,
                )
            result[m] = bounded(
                self.encoder[m](encoded_input)
                + self.metadata(coordinates[m])
                + self.modality[("rgb", "audio", "proprio").index(m)]
            )
        if outcome is not None and bool(outcome[4:].any()):
            result["outcome"] = bounded(self.outcome(outcome) + self.modality[4])[None]
        return result

    def binding(self, state: Any, tokens: Any, coordinates: Any) -> Any:
        if not tokens:
            return None
        c = self.c
        z = self.object_z(state)
        reliability = torch.stack(
            [self.reliability[m](value.mean(0)).squeeze() for m, value in tokens.items()]
        ).softmax(0)
        masses, contents, spatial = [], [], []
        for m, value in tokens.items():
            score = torch.einsum(
                "bmd,nd->bnm", self.bind_query(z), self.bind_key[m](value)
            ) / math.sqrt(c.d)
            score = score + functional.logsigmoid(state["l"]).transpose(-1, -2)
            if m == "rgb":
                distance = (
                    (coordinates[m][None, :, None, :2] - state["p"][:, None]).square().sum(-1)
                )
                score = score - distance / (0.01 + state["s"][..., :2].mean(-1)[:, None])
            null = self.null_score[m](value)[None].expand(c.branches, -1, -1)
            assignment = torch.cat([score, null], -1).softmax(-1)[..., :-1]
            mass = assignment.mean(1)
            content = (
                torch.einsum("bnm,nd->bmd", assignment, bounded(self.bind_value[m](value)))
                / value.shape[0]
            )
            masses.append(mass)
            contents.append(content)
            spatial.append(mass if m == "rgb" else torch.zeros_like(mass))
        mass = torch.einsum("m,mbi->bi", reliability, torch.stack(masses))
        content = torch.einsum("m,mbid->bid", reliability, torch.stack(contents))
        spatial_mass = torch.einsum("m,mbi->bi", reliability, torch.stack(spatial))
        return content / (mass[..., None] + 0.01), mass, spatial_mass

    def decode(self, state: Any, goal: Any, coordinates: Any) -> Any:
        c = self.c
        objects = torch.cat(
            [self.object_z(state), bounded(self.background)[None, None].expand(c.branches, 1, -1)],
            1,
        )
        prior_mass = torch.cat(
            [functional.logsigmoid(state["l"]).squeeze(-1), state["l"].new_zeros(c.branches, 1)], -1
        )
        result = {}
        for m, coord in coordinates.items():
            query = bounded(self.decoder_query[m](coord))
            scores = torch.einsum("nd,bmd->bnm", query, self.decoder_key(objects)) / math.sqrt(c.d)
            context = (scores + prior_mass[:, None]).softmax(-1) @ self.decoder_value(objects)
            if m == "proprio":
                context = context + self.read_h(state, goal)[:, None]
            raw = self.decoder[m](torch.cat([context, coord[None].expand(c.branches, -1, -1)], -1))
            mean_raw, scale_raw = raw.chunk(2, -1)
            lower, upper = {"rgb": (0.0, 1.0), "audio": (0.0, 6.0), "proprio": (-1.0, 1.0)}[m]
            result[m] = (
                lower + (upper - lower) * mean_raw.sigmoid(),
                (upper - lower) * (0.02 + 0.18 * scale_raw.sigmoid()),
                lower,
                upper,
            )
        return result

    def likelihood(
        self,
        state: Any,
        goal: Any,
        observations: Any,
        coordinates: Any,
        action: Any = None,
        outcome: Any = None,
    ) -> Any:
        parts = self.decode(state, goal, coordinates)
        logp = state["logw"].new_zeros(self.c.branches)
        for m, target in observations.items():
            mu, sd, lo, hi = parts[m]
            if m == "proprio":
                valid = target[..., 8:]
                target = torch.where(
                    valid.bool(), target[..., :8], torch.zeros_like(target[..., :8])
                )
                contributions = truncated_log_prob(target[None], mu, sd, lo, hi) * valid[None]
            else:
                contributions = truncated_log_prob(target[None], mu, sd, lo, hi)
            logp = logp + contributions.sum((-1, -2))
        if outcome is not None and bool(outcome[4:].any()):
            logits = self.reward(
                torch.cat([self.read_h(state, goal), action[None].expand(self.c.branches, -1)], -1)
            )
            if bool(outcome[4]):
                logp = logp + logits[:, :3].log_softmax(-1)[:, int(outcome[:3].argmax())]
            if bool(outcome[5]):
                logp = logp + functional.logsigmoid(logits[:, 3] * (2 * outcome[3] - 1))
        return logp


class U2Subject(_WorldKernel):
    """One differentiable candidate with explicit event and development state."""

    _capture_credit: bool
    _credit_tape: list[dict[str, torch.Tensor]]
    _verification_cuts: frozenset[str]

    def __init__(self, config: U2Config = DEFAULT_CONFIG) -> None:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            super().__init__(config)
            d = config.d
            self.start_embedding = nn.Parameter(torch.randn(config.embedding) * 0.05)
            self.record_goal_query = nn.Linear(d, d, bias=False)
            self.record_query = nn.Linear(2 * d + config.embedding, d)
            self.record_key = nn.Linear(d, d, bias=False)
            self.record_position_key = nn.Linear(d, d, bias=False)
            self.record_position_query = nn.Linear(2 * d, d)
            self.record_null = nn.Linear(d, 1)
            self.record_value = nn.Linear(d, d)
            self.record_output = nn.Linear(d, d)
            self.record_null_value = nn.Parameter(torch.randn(d) * 0.05)
            self.coverage_logit = nn.Parameter(torch.tensor(-2.0))
            self.copy_gate = nn.Linear(2 * d + config.embedding, 2)
            self.actor_parameters = MLP(2 * d, 6, d)
            self.actor_log_std = nn.Parameter(torch.full((6,), -1.0))
            self.development = MLP(2 * d + 15, 3, d)
            for module in (self.record_output,):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)
        self.generation = 0
        self.parameter_revision = 0
        self.unit_ids = [
            [f"g{g}:base:{u}" for u in range(config.units)] for g in range(config.groups)
        ]
        self.unit_branches = [[-1] * config.units for _ in range(config.groups)]

    @property
    def config(self) -> U2Config:
        return self.c

    def structure(self) -> Any:
        return {
            "generation": self.generation,
            "unit_ids": copy.deepcopy(self.unit_ids),
            "unit_branches": copy.deepcopy(self.unit_branches),
            "units": self.c.units,
        }

    def text_units(self, raw: Any) -> Any:
        raw = raw.encode() if isinstance(raw, str) else raw
        ids = torch.tensor(list(raw), dtype=torch.long, device=self.native_w.device)
        if not ids.numel():
            return self.native_w.new_zeros((0, self.c.d))
        value = self.text_input(self.embedding(ids)).T.unsqueeze(0)
        for layer, dilation in zip(self.convolutions, (1, 2, 4), strict=True):
            value = value + torch.tanh(layer(functional.pad(value, (2 * dilation, 0))))
        coordinates = value.new_zeros(len(raw), 8)
        coordinates[:, 2] = torch.linspace(-1, 1, len(raw), dtype=value.dtype, device=value.device)
        coordinates[:, 6] = 1
        return bounded(value.squeeze(0).T + self.metadata(coordinates) + self.modality[3])

    def goal(self, raw: Any) -> Any:
        if isinstance(raw, torch.Tensor):
            raw = bytes(raw.detach().cpu().tolist())
        units = self.text_units(raw)
        if not units.numel():
            return self.native_w.new_zeros(self.c.d)
        return (
            self.goal_score(units).squeeze(-1).softmax(0)[:, None] * self.goal_value(units)
        ).sum(0)

    def initial(self, subject_id: Any = "u2-reference") -> U2State:
        state = super().initial()
        ref = self.native_w
        generator = torch.Generator(device="cpu").manual_seed(self.c.seed + 1)
        state.update(
            {
                "subject_id": subject_id,
                "revision": self.parameter_revision,
                "goal_raw": b"",
                "goal_version": 0,
                "goal_status": "active",
                "records": (),
                "cursor": initial_cursor(
                    self.c.branches, 0, 0, ref.dtype, ref.device, state["logw"]
                ),
                "nu": torch.cat([ref.new_zeros(8), ref.new_ones(8)]),
                "nu_measurement_schema": (
                    "branch_entropy",
                    "predictive_surprisal",
                    "fast_norm_ratio",
                    "event_age",
                    "unknown4",
                    "unknown5",
                    "unknown6",
                    "unknown7",
                ),
                "world_time": 0.0,
                "process_time": 0.0,
                "processing_epoch": "u2-reference-epoch",
                "pending": None,
                "journal": [],
                "settled": [],
                "seen_packets": [],
                "history": [],
                "evidence": [],
                "rng": generator.get_state(),
                "resources": {"events": 0, "operations": 0, "updates": 0},
                "structure": self.structure(),
                "utterance_active": False,
                "previous_byte": 256,
                "utf8_remaining": 0,
                "utf8_lead": 0,
                "data_bytes": 0,
                "wake_condition": None,
                "eligibility": {
                    name: torch.zeros_like(p) for name, p in self.policy_named_parameters()
                },
                "prediction_goal": ref.new_zeros(self.c.d),
                "prediction_action": ref.new_zeros(15),
                "prediction_key_view": ref.new_zeros(self.c.branches, self.c.objects, 4 * self.c.d),
                "released_feedback_records": (),
            }
        )
        return state

    def set_goal(self, state: Any, raw: Any, version: Any = None) -> Any:
        child = dict(state)
        child["goal_raw"] = raw.encode() if isinstance(raw, str) else bytes(raw)
        child["goal_version"] = state["goal_version"] + 1 if version is None else int(version)
        child["goal_status"] = "active"
        child["eligibility"] = {
            name: torch.zeros_like(p) for name, p in self.policy_named_parameters()
        }
        return child

    def set_records(self, state: Any, records: Any) -> Any:
        if state["utterance_active"]:
            raise ValueError("record changes require an utterance boundary")
        records = tuple(records)
        if any(len(r.raw) > self.c.max_record_bytes or not r.raw for r in records):
            raise ValueError("record outside profile")
        if any(
            not math.isfinite(r.observed_at)
            or r.observed_at > state["world_time"]
            or r.source != "sensor"
            or r.scope != "world"
            for r in records
        ):
            raise ValueError("future or unauthorized record")
        if len({(r.record_id, r.version) for r in records}) != len(records):
            raise ValueError("duplicate record identity")
        child = dict(state)
        child["records"] = records
        size = max((len(r.raw) for r in records), default=0)
        child["cursor"] = initial_cursor(
            self.c.branches,
            len(records),
            size,
            self.native_w.dtype,
            self.native_w.device,
            state["logw"],
        )
        return child

    def record_units(self, state: Any) -> Any:
        records = state["records"]
        positions = max((len(r.raw) for r in records), default=0)
        tokens = self.native_w.new_zeros(len(records), positions, self.c.d)
        ids = torch.zeros((len(records), positions), dtype=torch.long, device=self.native_w.device)
        valid = torch.zeros_like(ids, dtype=torch.bool)
        for index, record in enumerate(records):
            size = len(record.raw)
            tokens[index, :size] = self.text_units(record.raw)
            ids[index, :size] = torch.tensor(list(record.raw), device=ids.device)
            valid[index, :size] = True
        return tokens, ids, valid

    def previous_embedding(self, previous: Any) -> Any:
        if previous == 256:
            return self.start_embedding
        return self.embedding(torch.tensor(previous, dtype=torch.long, device=self.native_w.device))

    def record_read(self, state: Any, goal: Any, previous: Any = 256, queries: Any = None) -> Any:
        if state["revision"] != self.parameter_revision:
            raise ValueError("Record state requires causal rebuilding under current parameters")
        tokens, ids, valid = self.record_units(state)
        c = self.c
        queries = state["H"] if queries is None else queries
        count = queries.shape[1]
        goal_units = self.text_units(state["goal_raw"])
        if goal_units.numel():
            scores = self.record_goal_query(queries) @ goal_units.T / math.sqrt(c.d)
            goal_context = scores.softmax(-1) @ goal_units
        else:
            goal_context = goal[None, None].expand(c.branches, count, -1)
        previous_value = self.previous_embedding(previous)[None, None].expand(c.branches, count, -1)
        query = self.record_query(torch.cat([queries, goal_context, previous_value], -1))
        pools = tokens.sum(1) / valid.sum(1).clamp_min(1)[:, None]
        route_score = query @ self.record_key(pools).T / math.sqrt(c.d)
        null = self.record_null(query)
        log_alpha_rows, log_beta_rows, read_rows = [], [], []
        for group in range(count):
            position_query = self.record_position_query(
                torch.cat(
                    [
                        query[:, group, None].expand(-1, len(pools), -1),
                        pools[None].expand(c.branches, -1, -1),
                    ],
                    -1,
                )
            )
            position_score = torch.einsum(
                "brd,rpd->brp", position_query, self.record_position_key(tokens)
            ) / math.sqrt(c.d)
            position_score = (
                position_score
                - functional.softplus(self.coverage_logit) * state["cursor"]["coverage"]
            )
            log_alpha, log_beta = record_inputs(
                torch.cat([route_score[:, group], null[:, group]], -1), position_score, valid
            )
            weights = (log_alpha[:, :-1, None] + log_beta).exp()
            context = torch.einsum("brp,rpd->bd", weights, self.record_value(tokens))
            context = context + log_alpha[:, -1:].exp() * self.record_null_value
            read_rows.append(self.record_output(context))
            log_alpha_rows.append(log_alpha)
            log_beta_rows.append(log_beta)
        return (
            torch.stack(read_rows, 1),
            torch.stack(log_alpha_rows, 1),
            torch.stack(log_beta_rows, 1),
            ids,
            valid,
        )

    def native(self, state: Any, goal: Any, action: Any, event: Any) -> Any:
        c = self.c
        old = state["H"]
        key = torch.einsum("bni,noi->bno", old, self.native_k)
        query = torch.einsum("bni,noi->bno", old, self.native_q) + self.native_goal(goal)
        value = torch.einsum("bni,noi->bno", old, self.native_v)
        message = (query @ key.transpose(-1, -2) / math.sqrt(c.d)).softmax(-1) @ value
        obj = self.read_o(state, self.read_h(state, goal))
        fview = torch.cat(
            [
                old,
                obj[:, None].expand(-1, c.groups, -1),
                goal[None, None].expand(c.branches, c.groups, -1),
                self.action(action)[None, None].expand(c.branches, c.groups, -1),
            ],
            -1,
        )
        read_f_value = self.f_output(
            torch.einsum("bkv,bnk->bnv", state["F"], bounded(self.f_query(fview)))
        )
        if "fast" in getattr(self, "_verification_cuts", ()):
            read_f_value = self.f_output(
                torch.zeros_like(
                    torch.einsum("bkv,bnk->bnv", state["F"], bounded(self.f_query(fview)))
                )
            )
        read_record, *_ = self.record_read(state, goal, state.get("previous_byte", 256))
        if "record_injection" in getattr(self, "_verification_cuts", ()):
            read_record = torch.zeros_like(read_record)
        x = torch.cat(
            [
                old / (1 + old.square().mean(-1, keepdim=True)).sqrt(),
                event[:, None].expand(-1, c.groups, -1),
                message,
                read_f_value,
                obj[:, None].expand(-1, c.groups, -1),
                read_record,
                goal[None, None].expand(c.branches, c.groups, -1),
                self.nu_projection(state["nu"])[None, None].expand(c.branches, c.groups, -1),
            ],
            -1,
        )
        hidden = torch.einsum("nmi,bni->bnm", self.native_r, x).tanh()
        content_gates = x[..., :2].softmax(-1)
        gate_rows = []
        for group in range(c.groups):
            gate_rows.append(
                torch.stack(
                    [
                        (
                            torch.ones_like(content_gates[:, group, 0])
                            if branch < 0
                            else content_gates[:, group, branch]
                        )
                        for branch in self.unit_branches[group]
                    ],
                    -1,
                )
            )
        gates = torch.stack(gate_rows, 1)
        pre = torch.einsum("bni,noi->bno", x, self.native_w) + torch.einsum(
            "nom,bnm->bno", self.native_p, hidden * gates
        )
        if getattr(self, "_capture_credit", False):
            self._credit_tape.append(
                {"preactivation": pre, "input": x, "content_gates": content_gates}
            )
        tau = 0.5 + 127.5 * self.tau_logit.sigmoid()
        leak = 1 - (-1 / tau).exp()
        return (1 - leak[None, :, None]) * old + leak[None, :, None] * pre.tanh()

    def predict(self, state: Any, goal: Any, action: Any, dt: Any = 0.05) -> Any:
        if not math.isfinite(dt) or dt < 0:
            raise ValueError("invalid physical duration")
        result = super().predict(state, goal, action, dt)
        result["prediction_key_view"] = self.views(result, goal, action).clone()
        return result

    def correct(
        self,
        prior: Any,
        goal: Any,
        action: Any,
        observations: Any,
        coordinates: Any,
        outcome: Any,
        write: Any = True,
    ) -> Any:
        observations = {
            m: v for m, v in observations.items() if m != "proprio" or bool(v[..., 8:].any())
        }
        coordinates = {m: coordinates[m] for m in observations}
        if not observations and not bool(outcome[4:].any()):
            return prior
        c = self.c
        result = dict(prior)
        logp = self.likelihood(
            prior,
            prior["prediction_goal"],
            observations,
            coordinates,
            prior["prediction_action"],
            outcome,
        )
        result["logw"] = (prior["logw"] + logp).log_softmax(0)
        tokens = self.encode(observations, coordinates, outcome)
        for _ in range(2):
            obs, mass, spatial = self.binding(result, tokens, coordinates)
            gate = mass / (mass + 0.01)
            pgate = spatial / (spatial + 0.01)
            packed = self.packed(result)
            raw = self.corrector(
                torch.cat(
                    [result["h"], obs, self.read_f(result, goal, action), packed[..., c.d :]], -1
                )
            )
            result["h"] = (1 - gate[..., None]) * result["h"] + gate[..., None] * raw[
                ..., : c.d
            ].tanh()
            result["p"] = result["p"] + pgate[..., None] * raw[..., c.d : c.d + 2].tanh()
            result["v"] = result["v"] + 2 * pgate[..., None] * raw[..., c.d + 2 : c.d + 4].tanh()
            result["s"] = (1 - pgate[..., None]) * result["s"] + pgate[..., None] * (
                1e-4 + (4 - 1e-4) * raw[..., c.d + 4 : c.d + 8].sigmoid()
            )
            result["l"] = (1 - gate[..., None]) * result["l"] + gate[..., None] * 8 * raw[
                ..., -1:
            ].tanh()
            for field in ("p", "v"):
                result[field] = result[field] * (
                    4 / result[field].norm(dim=-1, keepdim=True).clamp_min(4)
                )
            event = torch.stack([value.mean(0) for value in tokens.values()]).mean(0)
            result["H"] = self.native(result, goal, action, event[None].expand(c.branches, -1))
        if write:
            pooled = torch.cat([obs.mean(1), goal[None].expand(c.branches, -1)], -1)
            chi = self.f_chi(pooled).sigmoid().squeeze(-1)
            alpha = self.f_alpha(pooled).sigmoid()
            keys = bounded(self.f_key(prior["prediction_key_view"]))
            outcome_view = self.outcome(outcome)
            values = bounded(
                self.f_value(
                    torch.cat([obs, outcome_view[None, None].expand(c.branches, c.objects, -1)], -1)
                )
            )
            gains = (
                gate
                * 16
                * self.f_beta(torch.cat([self.object_z(result), obs], -1)).sigmoid().squeeze(-1)
            )
            record_tokens, _, record_valid = self.record_units(result)
            record_pools = record_tokens.sum(1) / record_valid.sum(1).clamp_min(1)[:, None]
            released = tuple(result["released_feedback_records"])
            if released:
                selected = [
                    index
                    for index, record in enumerate(result["records"])
                    if record.record_id in released and record.writable
                ]
                if len(selected) != len(released):
                    raise ValueError("feedback record not legally released")
                record_pools = record_pools[selected]
                # One feedback event is one association. Its key was known before
                # the response arrived, while the released content supplies its value.
                record_pools = record_pools.mean(0, keepdim=True)
                context = self.read_h(prior, prior["prediction_goal"])
                rec_view = torch.cat(
                    [
                        context[:, None],
                        self.read_o(prior, context)[:, None],
                        prior["prediction_goal"][None, None].expand(c.branches, 1, -1),
                        self.action(prior["prediction_action"])[None, None].expand(
                            c.branches, 1, -1
                        ),
                    ],
                    -1,
                )
                rec_keys = bounded(self.f_key(rec_view))
                rec_obs = record_pools[None].expand(c.branches, -1, -1)
                rec_values = bounded(
                    self.f_value(
                        torch.cat(
                            [
                                rec_obs,
                                outcome_view[None, None].expand(c.branches, len(record_pools), -1),
                            ],
                            -1,
                        )
                    )
                )
                rec_gains = 16 * self.f_beta(torch.cat([rec_obs, rec_obs], -1)).sigmoid().squeeze(
                    -1
                )
                keys = torch.cat([keys, rec_keys], 1)
                values = torch.cat([values, rec_values], 1)
                gains = torch.cat([gains, rec_gains], 1)
            result["F"] = torch.stack(
                [
                    fp1(
                        prior["F"][b],
                        keys[b].T,
                        values[b].T,
                        alpha[b],
                        gains[b],
                        chi[b],
                        chi.new_tensor(4.0),
                    )
                    for b in range(c.branches)
                ]
            )
            result["history"] = [
                *prior["history"],
                {
                    "keys": clone_tree(keys, detach=True),
                    "values": clone_tree(values, detach=True),
                    "gains": clone_tree(gains, detach=True),
                    "alpha": clone_tree(alpha, detach=True),
                    "chi": clone_tree(chi, detach=True),
                    "parameter_revision": self.parameter_revision,
                },
            ]
        read = self.read_f(result, goal, action)
        result["h"] = (1 - gate[..., None]) * result["h"] + gate[..., None] * self.post_memory(
            torch.cat([result["h"], read, goal[None, None].expand(c.branches, c.objects, -1)], -1)
        ).tanh()
        result["H"] = self.native(result, goal, action, event[None].expand(c.branches, -1))
        return result

    def read_f(self, state: Any, goal: Any, action: Any) -> Any:
        if "fast" not in getattr(self, "_verification_cuts", ()):
            return super().read_f(state, goal, action)
        query = bounded(self.f_query(self.views(state, goal, action)))
        return self.f_output(torch.zeros_like(torch.einsum("bkv,bmk->bmv", state["F"], query)))

    def branch_context(self, state: Any, goal: Any, action: Any) -> Any:
        h = self.read_h(state, goal)
        o = self.read_o(state, h)
        query = bounded(
            self.f_query(
                torch.cat(
                    [
                        h,
                        o,
                        goal[None].expand(self.c.branches, -1),
                        self.action(action)[None].expand(self.c.branches, -1),
                    ],
                    -1,
                )
            )
        )
        fast = self.f_output(torch.einsum("bkv,bk->bv", state["F"], query))
        if "fast" in getattr(self, "_verification_cuts", ()):
            fast = self.f_output(torch.zeros_like(torch.einsum("bkv,bk->bv", state["F"], query)))
        return self.joint_readout(torch.cat([h, o, fast], -1))

    def begin_utterance(self, state: Any) -> Any:
        child = dict(state)
        records = state["records"]
        positions = max((len(r.raw) for r in records), default=0)
        child["cursor"] = initial_cursor(
            self.c.branches,
            len(records),
            positions,
            self.native_w.dtype,
            self.native_w.device,
            state["logw"],
        )
        child.update(
            utterance_active=True, previous_byte=256, utf8_remaining=0, utf8_lead=0, data_bytes=0
        )
        return child

    def output(self, state: Any, goal: Any, action: Any, previous: Any = None) -> Any:
        previous = state["previous_byte"] if previous is None else previous
        context = self.branch_context(state, goal, action)
        record, alpha, beta, ids, valid = self.record_read(
            state, goal, previous, self.read_h(state, goal)[:, None]
        )
        record = record[:, 0]
        alpha = alpha[:, 0]
        beta = beta[:, 0]
        previous_value = self.previous_embedding(previous)[None].expand(self.c.branches, -1)
        branch_logits = self.byte_head(torch.cat([context, previous_value], -1)).log_softmax(-1)
        gate_input = torch.cat(
            [context, goal[None].expand(self.c.branches, -1), previous_value], -1
        )
        lam, gamma = self.copy_gate(gate_input).sigmoid().unbind(-1)
        mask = utf8_mask(
            state["utf8_remaining"],
            state["utf8_lead"],
            self.c.max_output_bytes - state["data_bytes"],
            device=self.native_w.device,
        )
        out = emission(state["cursor"], alpha, beta, lam, gamma, branch_logits, ids, valid, mask)
        return out["log_output"], context, {"mixture": out, "ids": ids, "valid": valid}

    def accept_byte(self, state: Any, symbol: Any, output: Any) -> Any:
        child = dict(state)
        child["cursor"] = accept(
            state["cursor"], output["mixture"], symbol, output["ids"], output["valid"]
        )
        if symbol == 256:
            child["utterance_active"] = False
            return child
        child["previous_byte"] = int(symbol)
        child["data_bytes"] += 1
        from taiji.utf8_state import advance_utf8

        child["utf8_remaining"], child["utf8_lead"] = advance_utf8(
            state["utf8_remaining"], state["utf8_lead"], symbol
        )
        event = bounded(
            self.text_input(self.embedding(torch.tensor(symbol, device=self.native_w.device)))
        )
        child["H"] = self.native(
            child,
            self.goal(child["goal_raw"]),
            self.native_w.new_zeros(15),
            event[None].expand(self.c.branches, -1),
        )
        return child

    def policy_named_parameters(self) -> Any:
        return [
            (name, p)
            for name, p in self.named_parameters()
            if name.startswith(("actor.", "actor_parameters.")) or name == "actor_log_std"
        ]

    def policy(self, state: Any, legal_mask: Any = None) -> Any:
        goal = self.goal(state["goal_raw"])
        context = self.branch_context(state, goal, self.native_w.new_zeros(15))
        features = torch.cat([context, goal[None].expand(self.c.branches, -1)], -1)
        proposals = torch.zeros((9, 15), dtype=features.dtype, device=features.device)
        proposals[:, :9] = torch.eye(9, dtype=features.dtype, device=features.device)
        development = self.development(
            torch.cat(
                [
                    features[:, None].expand(-1, 9, -1),
                    proposals[None].expand(self.c.branches, -1, -1),
                ],
                -1,
            )
        )
        actor_input = torch.cat(
            [
                features.detach()[:, None].expand(-1, 9, -1),
                proposals[None].expand(self.c.branches, -1, -1),
                development.detach(),
            ],
            -1,
        )
        logw = state["logw"].detach()
        weights = logw.exp()
        scores = (weights[:, None] * self.actor(actor_input).squeeze(-1)).sum(0)
        legal = (
            torch.ones(9, dtype=torch.bool, device=scores.device)
            if legal_mask is None
            else legal_mask
        )
        if not bool(legal.any()):
            raise ValueError("no legal operation")
        log_policy = scores.masked_fill(~legal, -torch.inf).log_softmax(0)
        flat = (weights[:, None] * features.detach()).sum(0)
        value = (weights * 2 * self.critic(features.detach()).tanh().squeeze(-1)).sum()
        return {
            "log_policy": log_policy,
            "value": value,
            "features": flat,
            "critic_features": features.detach().clone(),
            "critic_weights": weights.detach().clone(),
            "psi_version": self.parameter_revision,
            "development": (weights[:, None, None] * development).sum(0),
            "proposals": proposals,
        }

    def pending_value(self, decision: Any) -> torch.Tensor:
        """Rebuild the conditional old V from behavior-time inputs and current frozen Psi."""
        required = {"critic_features", "critic_weights", "psi_version", "value"}
        if not required.issubset(decision):
            raise ValueError("pending critic credit inputs are missing")
        features, weights = decision["critic_features"], decision["critic_weights"]
        cached = decision["value"]
        if decision["psi_version"] != self.parameter_revision:
            raise ValueError("pending critic Psi version differs from subject")
        if (
            not isinstance(features, torch.Tensor)
            or not isinstance(weights, torch.Tensor)
            or not isinstance(cached, torch.Tensor)
            or features.shape != (self.c.branches, 2 * self.c.d)
            or weights.shape != (self.c.branches,)
            or cached.ndim != 0
        ):
            raise ValueError("pending critic input shape differs from profile")
        for value in (features, weights, cached):
            if value.dtype != self.native_w.dtype or value.device != self.native_w.device:
                raise ValueError("pending critic input layout differs from subject")
            if not bool(torch.isfinite(value).all()):
                raise ValueError("pending critic inputs must be finite")
        tolerance = 1e-10 if features.dtype == torch.float64 else 1e-5
        if bool((weights < 0).any()) or not torch.allclose(
            weights.sum(), weights.new_tensor(1.0), atol=tolerance, rtol=0
        ):
            raise ValueError("pending critic weights must be normalized")
        # Behavior inputs stay fixed. Critic parameters retain their learning graph.
        rebuilt = (weights.detach() * 2 * self.critic(features.detach()).tanh().squeeze(-1)).sum()
        if not torch.allclose(rebuilt.detach(), cached.detach(), atol=tolerance, rtol=tolerance):
            raise ValueError("pending critic cached value is inconsistent")
        return cast(torch.Tensor, rebuilt)

    def choose_operation(self, state: Any, legal_mask: Any = None) -> Any:
        policy = self.policy(state, legal_mask)
        generator = torch.Generator(device="cpu")
        generator.set_state(state["rng"])
        index = int(
            torch.multinomial(policy["log_policy"].exp().detach().cpu(), 1, generator=generator)
        )
        mean = self.actor_parameters(policy["features"])
        noise = torch.randn(6, dtype=mean.dtype, generator=generator).to(mean.device)
        latent = (mean + self.actor_log_std.exp() * noise).detach()
        log_density_parts = (
            -0.5 * ((latent - mean) / self.actor_log_std.exp()).square()
            - self.actor_log_std
            - 0.5 * math.log(2 * math.pi)
        )
        target = latent[:2].tanh()
        force = latent[2:4] / (1 + latent[2:4].square().sum()).sqrt()
        intensity = latent[4].sigmoid()
        duration = 0.05 * latent[5].sigmoid()
        sample = torch.cat([target, force, intensity[None], duration[None]])
        # Density of the invertible actuator transforms, holding the sampled
        # latent fixed for the conditional policy score.
        target_jac = 2 * (math.log(2) - latent[:2] - functional.softplus(-2 * latent[:2]))
        force_logdet = -2 * torch.log1p(latent[2:4].square().sum())
        intensity_logdet = functional.logsigmoid(latent[4]) + functional.logsigmoid(-latent[4])
        duration_logdet = (
            math.log(0.05) + functional.logsigmoid(latent[5]) + functional.logsigmoid(-latent[5])
        )
        target_density = log_density_parts[:2].sum() - target_jac.sum()
        sensing_density = log_density_parts[4:].sum() - intensity_logdet - duration_logdet
        log_density = target_density + sensing_density
        if index == 0:
            log_density = log_density + log_density_parts[2:4].sum() - force_logdet
        elif index == 2:
            sample = sample.clone()
            sample[2:4] = 0
        else:
            sample = torch.zeros_like(sample)
        action = torch.cat([policy["proposals"][index, :9], sample]).detach()
        log_behavior = policy["log_policy"][index] + (log_density if index in (0, 2) else 0.0)
        named = self.policy_named_parameters()
        scores = torch.autograd.grad(
            log_behavior, [p for _, p in named], retain_graph=True, allow_unused=True
        )
        eligibility = {
            name: state["eligibility"].get(name, torch.zeros_like(p))
            + (torch.zeros_like(p) if score is None else score.detach())
            for (name, p), score in zip(named, scores, strict=True)
        }
        decision = {
            **policy,
            "index": index,
            "operation": (
                "act",
                "express",
                "inspect",
                "ask",
                "replay",
                "consolidate",
                "grow",
                "rest",
                "wait",
            )[index],
            "action": action,
            "log_behavior": log_behavior,
            "eligibility": eligibility,
            "action_parameter_schema": (
                "target_x",
                "target_y",
                "unit_disk_force_x",
                "unit_disk_force_y",
                "sense_intensity",
                "requested_duration",
            ),
        }
        child = dict(state)
        child["rng"] = generator.get_state()
        return child, decision

    def issue(self, state: Any, decision: Any, dt: Any = 0.05) -> Any:
        if state["revision"] != self.parameter_revision:
            raise ValueError("Action state requires causal rebuilding under current parameters")
        if state["pending"] is not None:
            raise ValueError("pending operation already exists")
        if decision["operation"] not in {"act", "inspect", "ask"}:
            raise ValueError("external issue requires external operation")
        goal = self.goal(state["goal_raw"])
        action = decision["action"]
        if decision["operation"] == "ask":
            dt = 0.0
            action = action.clone()
            action[9:] = 0
            prior = dict(state)
            prior["prediction_goal"] = goal.clone()
            prior["prediction_action"] = action.clone()
            prior["prediction_key_view"] = self.views(state, goal, action).clone()
        else:
            prior = self.predict(state, goal, action, dt)
        receipt_id = f"{state['subject_id']}:{state['revision']}:{len(state['journal'])}"
        receipt = {
            "id": receipt_id,
            "subject_version": state["revision"],
            "goal_version": state["goal_version"],
            "goal_raw": state["goal_raw"],
            "operation": decision["operation"],
            "action": action,
            "dt": dt,
            "prior": prior,
            "decision": decision,
            "world_end": state["world_time"] + dt,
        }
        child = dict(state)
        child["pending"] = receipt
        child["journal"] = [
            *state["journal"],
            {"id": receipt_id, "status": "executed", "operation": decision["operation"]},
        ]
        return child, receipt_id

    def receive(self, state: Any, packet: Any) -> Any:
        if packet.packet_id in state["seen_packets"]:
            return state, {}
        if (
            not all(math.isfinite(t) for t in (packet.t_start, packet.t_end, packet.t_arrival))
            or packet.t_start > packet.t_end
        ):
            raise ValueError("invalid event time")
        if packet.source != "sensor" or packet.scope != "world" or not packet.valid:
            raise ValueError("illegal event source/scope")
        receipt = state["pending"]
        if receipt is None or receipt["id"] != packet.receipt_id:
            raise ValueError("receipt mismatch")
        if receipt["subject_version"] != state["revision"]:
            raise ValueError("parameters changed while pending")
        old_value = self.pending_value(receipt["decision"])
        if packet.t_end > receipt["world_end"] or packet.t_arrival > state["process_time"]:
            raise ValueError("event not arrived")
        if (
            not math.isclose(packet.t_end, receipt["world_end"], rel_tol=0.0, abs_tol=1e-10)
            or packet.t_start < receipt["world_end"] - receipt["dt"] - 1e-10
        ):
            raise ValueError("historical observation requires shadow replay")
        deferred = receipt.get("deferred_packet")
        if deferred is not None and deferred.packet_id == packet.packet_id:
            if bool(packet.outcome[4:].all()):
                raise ValueError("Completed result must have a fresh immutable packet identity")
            return state, {"status": "awaiting_outcome"}
        evidence = {
            "packet_id": packet.packet_id,
            "source": packet.source,
            "scope": packet.scope,
            "t_start": packet.t_start,
            "t_end": packet.t_end,
            "t_arrival": packet.t_arrival,
            "observations": clone_tree(packet.observations, detach=True),
            "coordinates": clone_tree(packet.coordinates, detach=True),
            "outcome": packet.outcome.detach().clone(),
            "old_goal": receipt["goal_raw"],
            "action": receipt["action"].detach().clone(),
            "subject_version": receipt["subject_version"],
            "receipt_id": receipt["id"],
        }
        if not bool(packet.outcome[4:].all()):
            child = dict(state)
            child["pending"] = {**receipt, "deferred_packet": packet}
            child["evidence"] = [*state["evidence"], evidence]
            child["journal"] = [
                *state["journal"],
                {"id": receipt["id"], "status": "awaiting_outcome", "packet": packet.packet_id},
            ]
            return child, {"status": "awaiting_outcome"}
        if not packet.observations and "deferred_packet" in receipt:
            cached = receipt["deferred_packet"]
            packet = replace(
                packet,
                observations=cached.observations,
                coordinates=cached.coordinates,
                feedback_record_ids=cached.feedback_record_ids,
            )
        old_goal = self.goal(receipt["goal_raw"])
        goal = self.goal(state["goal_raw"])
        prior = receipt["prior"]
        predictive = self.likelihood(
            prior,
            old_goal,
            packet.observations,
            packet.coordinates,
            receipt["action"],
            packet.outcome,
        )
        current = dict(prior)
        current["goal_raw"] = state["goal_raw"]
        current["goal_version"] = state["goal_version"]
        current["released_feedback_records"] = packet.feedback_record_ids
        result = self.correct(
            current,
            goal,
            receipt["action"],
            packet.observations,
            packet.coordinates,
            packet.outcome,
            write=packet.write_allowed,
        )
        result["world_time"] = receipt["world_end"]
        result["process_time"] = state["process_time"]
        result["pending"] = None
        result["seen_packets"] = [*state["seen_packets"], packet.packet_id]
        if "deferred_packet" in receipt:
            result["seen_packets"] = list(
                dict.fromkeys([*result["seen_packets"], receipt["deferred_packet"].packet_id])
            )
        result["evidence"] = [*state["evidence"], evidence]
        result["journal"] = [
            *state["journal"],
            {"id": receipt["id"], "packet": packet.packet_id, "status": "settled"},
        ]
        result["settled"] = [*state["settled"], receipt["id"]]
        result["resources"] = {**state["resources"], "events": state["resources"]["events"] + 1}
        decision = receipt["decision"]
        utility = packet.outcome.new_tensor([-1.0, 0.0, 1.0]) @ packet.outcome[:3]
        terminal = packet.outcome[3]
        next_context = self.branch_context(result, old_goal, receipt["action"])
        next_features = torch.cat(
            [next_context.detach(), old_goal.detach()[None].expand(self.c.branches, -1)], -1
        )
        next_value = (
            result["logw"].exp().detach() * 2 * self.critic(next_features).tanh().squeeze(-1)
        ).sum()
        discount = 0.99 ** max(1.0, receipt["dt"] / 0.05)
        delta = (
            (1 - discount) * utility + discount * (1 - terminal) * next_value.detach() - old_value
        )
        actor_loss = -delta.detach() * sum(
            (p * decision["eligibility"][name]).sum() for name, p in self.policy_named_parameters()
        )
        critic_loss = 0.5 * delta.square()
        result["eligibility"] = {
            name: (
                (0.99 * 0.95) ** max(1.0, receipt["dt"] / 0.05) * value
                if receipt["goal_version"] == state["goal_version"]
                else torch.zeros_like(value)
            )
            for name, value in decision["eligibility"].items()
        }
        entropy = -(result["logw"].exp() * result["logw"]).sum() / math.log(max(2, self.c.branches))
        measured = torch.stack(
            [
                entropy.detach(),
                predictive.detach().abs().mean().sigmoid(),
                result["F"].detach().norm() / (4 * math.sqrt(self.c.branches)),
                result["F"].new_tensor(math.log1p(result["resources"]["events"])),
            ]
        )
        nu = result["nu"].clone()
        nu[:4] = 0.95 * nu[:4] + 0.05 * measured
        nu[8:12] = 0
        result["nu"] = nu
        return result, {
            "predictive_logp": predictive,
            "actor": actor_loss,
            "critic": critic_loss,
            "prior": prior,
            "decision": decision,
        }

    def replay(self, state: Any) -> Any:
        if state["pending"] is not None:
            raise ValueError("cannot replay during pending external operation")
        if not state["history"]:
            raise ValueError("no released historical association")
        child = dict(state)
        rows = state["history"][-4:]
        if state["revision"] != self.parameter_revision or any(
            row["parameter_revision"] != self.parameter_revision for row in rows
        ):
            raise ValueError("Historical associations require causal rebuilding")
        keys = torch.cat([row["keys"] for row in rows], 1)
        values = torch.cat([row["values"] for row in rows], 1)
        gains = torch.cat([row["gains"] for row in rows], 1)
        alpha = torch.stack([row["alpha"] for row in rows]).mean(0)
        chi = torch.stack([row["chi"] for row in rows]).mean(0)
        child["F"] = torch.stack(
            [
                fp1(
                    state["F"][b],
                    keys[b].T,
                    values[b].T,
                    alpha[b],
                    gains[b],
                    chi[b],
                    chi.new_tensor(4.0),
                )
                for b in range(self.c.branches)
            ]
        )
        child["journal"] = [*state["journal"], {"operation": "replay", "history_rows": len(rows)}]
        child["resources"] = {
            **state["resources"],
            "operations": state["resources"]["operations"] + 1,
        }
        return child

    def internal(self, state: Any, decision: Any) -> Any:
        operation = decision["operation"]
        if operation == "replay":
            return self.replay(state)
        if operation in {"wait", "rest"}:
            child = dict(state)
            child["wake_condition"] = {
                "event": "observation_or_goal",
                "deadline": state["world_time"] + 1.0,
            }
            child["journal"] = [*state["journal"], {"operation": operation, "status": "waiting"}]
            return child
        raise ValueError(
            "operation requires explicit expression, growth or consolidation transaction"
        )

    def consolidate(
        self,
        state: Any,
        optimizer: Any,
        loss: Any,
        acceptance: Any = None,
        rebuild: Any = None,
        budget_ticket: Any = None,
    ) -> Any:
        if state["pending"] is not None or state["utterance_active"]:
            raise ValueError("consolidation requires boundary")
        if (
            acceptance is None
            or rebuild is None
            or budget_ticket is None
            or budget_ticket.get("updates", 0) < 1
        ):
            raise ValueError(
                "consolidation not admitted without retention, causal rebuild and budget"
            )
        parent = {name: value.detach().clone() for name, value in self.state_dict().items()}
        old_optimizer = copy.deepcopy(optimizer.state_dict())
        old_revision = self.parameter_revision
        budget_ticket["updates"] -= 1
        budget_ticket["attempts"] = budget_ticket.get("attempts", 0) + 1
        try:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            accepted = bool(acceptance())
            if accepted:
                self.parameter_revision = old_revision + 1
                child = rebuild()
                if child["revision"] != self.parameter_revision or any(
                    row["parameter_revision"] != self.parameter_revision for row in child["history"]
                ):
                    raise ValueError("Consolidation callback returned stale encoded state")
            else:
                child = clone_tree(state, detach=True)
        except Exception:
            self.load_state_dict(parent)
            optimizer.load_state_dict(old_optimizer)
            optimizer.zero_grad(set_to_none=True)
            self.parameter_revision = old_revision
            raise
        child["resources"]["updates"] += 1
        child["journal"].append({"operation": "consolidate", "accepted": accepted})
        if not accepted:
            self.load_state_dict(parent)
            optimizer.load_state_dict(old_optimizer)
            optimizer.zero_grad(set_to_none=True)
        return child

    def birth_units(
        self,
        credits: Any,
        *,
        count: Any = 1,
        branch: Any = 0,
        state: Any = None,
        optimizer: Any = None,
        birth_id: Any = None,
    ) -> Any:
        if type(count) is not int or count < 1 or type(branch) is not int or branch not in (0, 1):
            raise ValueError("Invalid birth count or conditional branch")
        if (
            not isinstance(credits, torch.Tensor)
            or credits.dtype != self.native_w.dtype
            or credits.device != self.native_w.device
            or not bool(torch.isfinite(credits).all())
        ):
            raise ValueError("Birth credit must be finite and share the parent layout")
        if birth_id is not None and (not isinstance(birth_id, str) or not birth_id):
            raise ValueError("Birth identity must be a nonempty string")
        next_birth_id = birth_id or f"birth-{self.generation + 1}"
        if any(
            f"{next_birth_id}:g{group}:u{unit}" in self.unit_ids[group]
            for group in range(self.c.groups)
            for unit in range(count)
        ):
            raise ValueError("Birth identity has already been committed")
        if state is not None and (state["pending"] is not None or state["utterance_active"]):
            raise ValueError("birth requires boundary")
        if state is not None and (
            state["revision"] != self.parameter_revision or state["structure"] != self.structure()
        ):
            raise ValueError("Birth state belongs to another parameter or structure version")
        if credits.shape != (self.c.groups, self.c.d, 8 * self.c.d):
            raise ValueError("credit shape differs from full native input")
        selected = [choose_left_directions(c, count) for c in credits]
        if any(item["effective_budget"] != count for item in selected):
            raise ValueError("insufficient independent credit")
        old_r, old_p = self.native_r, self.native_p
        out = torch.stack([item["directions"] for item in selected]) * 0.1
        new_r = nn.Parameter(
            torch.cat([old_r.detach(), old_r.new_zeros(self.c.groups, count, 8 * self.c.d)], 1)
        )
        new_p = nn.Parameter(torch.cat([old_p.detach(), out], -1))
        self.native_r = new_r
        self.native_p = new_p
        for old, new, axis in ((old_r, new_r, 1), (old_p, new_p, 2)):
            if optimizer is not None:
                for group in optimizer.param_groups:
                    group["params"] = [new if p is old else p for p in group["params"]]
                stored = optimizer.state.pop(old, {})
                migrated = {}
                for key, value in stored.items():
                    if isinstance(value, torch.Tensor) and value.shape == old.shape:
                        target = torch.zeros_like(new)
                        slices = [slice(None)] * new.ndim
                        slices[axis] = slice(0, old.shape[axis])
                        target[tuple(slices)] = value
                        migrated[key] = target
                    else:
                        migrated[key] = clone_tree(value, detach=True)
                optimizer.state[new] = migrated
        self.generation += 1
        birth_id = next_birth_id
        for group in range(self.c.groups):
            self.unit_ids[group].extend(f"{birth_id}:g{group}:u{u}" for u in range(count))
            self.unit_branches[group].extend([branch] * count)
        self.c = replace(self.c, units=self.c.units + count)
        self.parameter_revision += 1
        if state is not None:
            state["structure"] = self.structure()
            state["revision"] = self.parameter_revision
            state["history"] = [
                {
                    **row,
                    "parameter_revision": self.parameter_revision,
                    "neutral_birth_rebase": birth_id,
                }
                for row in state["history"]
            ]
        return {
            "birth_id": birth_id,
            "generation": self.generation,
            "parent_units": old_r.shape[1],
            "units": self.c.units,
            "count": count,
        }

    def parameter_owners(self) -> dict[str, list[str]]:
        result = {}
        for name, _ in self.named_parameters():
            if name.startswith(("actor.", "actor_parameters.")) or name == "actor_log_std":
                owners = ["actor"]
            elif name.startswith("critic."):
                owners = ["critic"]
            elif name.startswith("development."):
                owners = ["development"]
            elif name.startswith(("decoder", "background")):
                owners = ["next_sensor", "reconstruction", "masked"]
            elif name.startswith("reward."):
                owners = ["reward", "done", "next_sensor"]
            elif name.startswith(
                ("record_", "copy_gate.", "coverage_logit", "start_embedding", "byte_head.")
            ):
                owners = ["task", "next_sensor"]
            else:
                owners = [
                    "next_sensor",
                    "reconstruction",
                    "masked",
                    "task",
                    "reward",
                    "done",
                    "development",
                ]
            result[name] = owners
        return result

    def collect_birth_credit(self, task_loss: Any, trace: Any, branch: Any = 0) -> Any:
        """Task-owned recurrent preactivation credit, with no second averaging."""
        tape = trace["credit_tape"]
        derivatives = torch.autograd.grad(
            task_loss, [row["preactivation"] for row in tape], retain_graph=True, allow_unused=True
        )
        credit = self.native_w.new_zeros(self.c.groups, self.c.d, 8 * self.c.d)
        count = 0
        for row, delta in zip(tape, derivatives, strict=True):
            if delta is None:
                continue
            if branch is not None:
                delta = delta * row["content_gates"][..., branch, None]
            credit = credit + torch.einsum("bgd,bgp->gdp", delta, row["input"])
            count += 1
        return {
            "credit": credit.detach(),
            "legal_samples": count,
            "branch": branch,
            "loss_denominator": "already included in task_loss",
        }

    def state_registry(self) -> tuple[str, ...]:
        return tuple(sorted(self.initial().keys()))

    def mark_parameter_update(self) -> int:
        """Declare an externally owned engineering update before causal rebuild."""
        self.parameter_revision += 1
        return self.parameter_revision

    def export_bundle(self, state: Any, optimizer_state: Any = None) -> Any:
        if set(state) != set(self.state_registry()):
            raise ValueError("unregistered subject state")
        if state["structure"] != self.structure():
            raise ValueError("state/model structure mismatch")
        if state["revision"] != self.parameter_revision:
            raise ValueError("state must be rebuilt under the current parameter version")
        if any(row["parameter_revision"] != self.parameter_revision for row in state["history"]):
            raise ValueError("encoded history version differs from subject parameters")
        if state["pending"] is not None:
            if state["pending"]["subject_version"] != self.parameter_revision:
                raise ValueError("pending version differs from subject parameters")
            self.pending_value(state["pending"]["decision"])
        return {
            "schema": "taiji-u2-subject-v2",
            "credit_contract": {
                "critic": "behavior_features_under_frozen_Psi",
                "body_graph": "values_only_requires_window_replay",
            },
            "config": asdict(self.c),
            "structure": self.structure(),
            "parameter_revision": self.parameter_revision,
            "parameters": clone_tree(self.state_dict(), detach=True),
            "state": clone_tree(state, detach=True),
            "optimizer": clone_tree(optimizer_state, detach=True),
        }

    @classmethod
    def from_bundle(cls, bundle: Any) -> Any:
        if bundle["schema"] not in {"taiji-u2-subject-v1", "taiji-u2-subject-v2"}:
            raise ValueError("unsupported bundle schema")
        if bundle["schema"] == "taiji-u2-subject-v1" and bundle["state"].get("pending") is not None:
            raise ValueError("pending v1 bundle lacks recoverable critic/window credit inputs")
        if bundle["schema"] == "taiji-u2-subject-v2" and bundle.get("credit_contract") != {
            "critic": "behavior_features_under_frozen_Psi",
            "body_graph": "values_only_requires_window_replay",
        }:
            raise ValueError("bundle credit contract differs from supported recovery scope")
        model = cls(U2Config(**bundle["config"]))
        first = next(iter(bundle["parameters"].values()))
        model = model.to(dtype=first.dtype, device=first.device)
        model.load_state_dict(bundle["parameters"])
        model.parameter_revision = bundle["parameter_revision"]
        structure = bundle["structure"]
        model.generation = structure["generation"]
        model.unit_ids = copy.deepcopy(structure["unit_ids"])
        model.unit_branches = copy.deepcopy(structure["unit_branches"])
        state = clone_tree(bundle["state"], detach=True)
        if set(state) != set(model.state_registry()) or state["structure"] != model.structure():
            raise ValueError("bundle state registry mismatch")
        if state["revision"] != model.parameter_revision or any(
            row["parameter_revision"] != model.parameter_revision for row in state["history"]
        ):
            raise ValueError("bundle state/history parameter version mismatch")
        if state["pending"] is not None:
            pending = state["pending"]
            if pending["subject_version"] != model.parameter_revision:
                raise ValueError("pending bundle parameter version mismatch")
            model.pending_value(pending["decision"])
        return model, state


def synthetic_fixture(config: U2Config, dtype: torch.dtype) -> dict[str, Any]:
    """Deterministic generated observations. No project corpus or hidden world IDs."""
    generator = torch.Generator().manual_seed(config.seed + 17)

    def coordinates(count: Any, spatial: Any = False) -> Any:
        value = torch.zeros(count, 8, dtype=dtype)
        value[:, 2] = torch.linspace(-1, 1, count, dtype=dtype)
        value[:, 3] = torch.linspace(-0.16, -0.01, count, dtype=dtype)
        value[:, 4] = torch.linspace(-0.01, 0, count, dtype=dtype)
        value[:, 6] = 1
        if spatial:
            grid = config.side // config.patch
            yy, xx = torch.meshgrid(
                torch.linspace(-1, 1, grid, dtype=dtype),
                torch.linspace(-1, 1, grid, dtype=dtype),
                indexing="ij",
            )
            value[:, :2] = torch.stack([xx.flatten(), yy.flatten()], -1)
            value[:, 5] = 1
        return value

    image = 0.4 + 0.2 * torch.rand(config.side, config.side, 3, dtype=dtype, generator=generator)
    grid = config.side // config.patch
    rgb = (
        image.reshape(grid, config.patch, grid, config.patch, 3)
        .permute(0, 2, 1, 3, 4)
        .reshape(grid**2, config.dimensions["rgb"])
    )
    wave = 0.3 * torch.sin(torch.arange(config.audio_samples, dtype=dtype) * 0.17)
    audio = (
        torch.stft(
            wave,
            config.fft,
            config.hop,
            window=torch.hann_window(config.fft, dtype=dtype),
            center=False,
            return_complex=True,
        )
        .abs()
        .log1p()
        .T
    )
    proprio = torch.cat(
        [torch.rand(1, 8, dtype=dtype, generator=generator) * 0.5, torch.ones(1, 8, dtype=dtype)],
        -1,
    )
    observations = {"rgb": rgb, "audio": audio, "proprio": proprio}
    coords = {m: coordinates(value.shape[0], m == "rgb") for m, value in observations.items()}
    coords["rgb"][:, 2:5] = 0
    coords["rgb"][:, 6] = 0
    coords["proprio"][:, 2:5] = 0
    coords["proprio"][:, 6] = 0
    starts = (torch.arange(audio.shape[0], dtype=dtype) * config.hop - config.audio_samples) / 16000
    coords["audio"][:, 3] = torch.asinh(starts)
    coords["audio"][:, 4] = torch.asinh(starts + config.fft / 16000)
    coords["audio"][:, 7] = 1
    events = []
    for index in range(3):
        factor = 0.98 - index * 0.01
        obs = {
            m: (
                v * factor + 0.002 * (index + 1)
                if m != "proprio"
                else torch.cat([v[..., :8] * factor + 0.002 * (index + 1), v[..., 8:]], -1)
            )
            for m, v in observations.items()
        }
        events.append(
            {
                "observations": obs,
                "coordinates": coords,
                "outcome": torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0, 1.0], dtype=dtype),
            }
        )
    future = {
        m: (
            v * 0.94 + 0.01
            if m != "proprio"
            else torch.cat([v[..., :8] * 0.94 + 0.01, v[..., 8:]], -1)
        )
        for m, v in observations.items()
    }
    return {
        "events": events,
        "future": future,
        "coordinates": coords,
        "goal": "把红盒推到左边。".encode(),
        "replacement_goal": "先核对蓝盒的位置。".encode(),
        "records": (
            U2Record("red", "红盒在左边。".encode()),
            U2Record("blue", "蓝盒在右边。".encode()),
        ),
        "target": "红盒".encode(),
        "old_target": "蓝盒".encode(),
    }


def run_reference_sequence(
    model: U2Subject, fixture: dict[str, Any], cuts: tuple[str, ...] = ()
) -> tuple[dict[str, torch.Tensor], U2State, dict[str, Any], dict[str, Any]]:
    """One subject's synthetic events, expression, policy and real internal result.

    All targets are generated observations, requested bytes or measured effects
    of an actually executed reference operation. No parameter norm is a target.
    """
    previous_cuts: frozenset[str] = getattr(model, "_verification_cuts", frozenset())
    previous_capture = getattr(model, "_capture_credit", False)
    model._capture_credit = True
    model._credit_tape = []
    model._verification_cuts = frozenset(cuts)
    try:
        state = model.initial("u2-reference-sequence")
        state = model.set_goal(state, fixture["goal"])
        state = model.set_records(state, fixture["records"])
        policy_losses = []
        critic_losses = []
        predictive_losses = []
        receipts = []
        for index, event in enumerate(fixture["events"]):
            mask = torch.zeros(9, dtype=torch.bool, device=model.native_w.device)
            mask[0] = True
            mask[2] = True
            state, decision = model.choose_operation(state, mask)
            state, receipt_id = model.issue(state, decision, dt=0.05)
            if index == 1:
                state = model.set_goal(state, fixture["replacement_goal"])
            state["process_time"] += 0.05
            packet = U2Packet(
                f"observation-{index}",
                "sensor",
                "world",
                index * 0.05,
                (index + 1) * 0.05,
                state["process_time"],
                event["observations"],
                event["coordinates"],
                event["outcome"],
                receipt_id,
                feedback_record_ids=(("red",) if index < 2 else ("blue",)),
            )
            state, receipt = model.receive(state, packet)
            count = sum(
                v.numel() if m != "proprio" else int(v[..., 8:].sum())
                for m, v in event["observations"].items()
            )
            predictive_losses.append(
                -torch.logsumexp(receipt["prior"]["logw"] + receipt["predictive_logp"], 0) / count
            )
            policy_losses.append(receipt["actor"])
            critic_losses.append(receipt["critic"])
            receipts.append(receipt)
        goal = model.goal(state["goal_raw"])
        action = state["prediction_action"]
        next_prior = model.predict(state, goal, action, 0.05)
        future_count = sum(
            v.numel() if m != "proprio" else int(v[..., 8:].sum())
            for m, v in fixture["future"].items()
        )
        next_sensor = (
            -torch.logsumexp(
                state["logw"]
                + model.likelihood(next_prior, goal, fixture["future"], fixture["coordinates"]),
                0,
            )
            / future_count
        )
        last = fixture["events"][-1]
        reconstruction = (
            -torch.logsumexp(
                state["logw"]
                + model.likelihood(state, goal, last["observations"], last["coordinates"]),
                0,
            )
            / future_count
        )
        reduced = {m: v for m, v in last["observations"].items() if m != "audio"}
        reduced_coords = {m: last["coordinates"][m] for m in reduced}
        masked_state = model.correct(
            receipts[-1]["prior"],
            goal,
            action,
            reduced,
            reduced_coords,
            last["outcome"],
            write=False,
        )
        masked = (
            -torch.logsumexp(
                masked_state["logw"]
                + model.likelihood(
                    masked_state,
                    goal,
                    {"audio": last["observations"]["audio"]},
                    {"audio": last["coordinates"]["audio"]},
                ),
                0,
            )
            / last["observations"]["audio"].numel()
        )
        reward_logits = model.reward(
            torch.cat(
                [
                    model.read_h(receipts[-1]["prior"], receipts[-1]["prior"]["prediction_goal"]),
                    action[None].expand(model.c.branches, -1),
                ],
                -1,
            )
        )
        reward = -torch.logsumexp(
            receipts[-1]["prior"]["logw"] + reward_logits[:, :3].log_softmax(-1)[:, 2], 0
        )
        done = -torch.logsumexp(
            receipts[-1]["prior"]["logw"] + functional.logsigmoid(-reward_logits[:, 3]), 0
        )
        utterance = model.begin_utterance(state)
        before_world = {
            key: state[key].detach().clone() for key in ("p", "v", "s", "l", "logw", "F")
        }
        before_h = state["H"].detach().clone()
        task_terms = []
        for symbol in [*fixture["target"], 256]:
            distribution, _, output = model.output(utterance, goal, action)
            task_terms.append(-distribution[symbol])
            utterance = model.accept_byte(utterance, symbol, output)
        assert all(torch.equal(state[key].detach(), value) for key, value in before_world.items())
        assert torch.equal(state["H"].detach(), before_h)
        task = torch.stack(task_terms).mean()
        probe = model.begin_utterance(state)
        distribution, _, _ = model.output(probe, goal, action)
        before_task = -distribution[fixture["target"][0]]
        before_old = -distribution[fixture["old_target"][0]]
        internal_mask = torch.zeros(9, dtype=torch.bool, device=model.native_w.device)
        internal_mask[4] = True
        internal_mask[8] = True
        state, decision = model.choose_operation(state, internal_mask)
        after = model.internal(state, decision)
        after_probe = model.begin_utterance(after)
        after_distribution, _, _ = model.output(after_probe, goal, action)
        after_task = -after_distribution[fixture["target"][0]]
        after_old = -after_distribution[fixture["old_target"][0]]
        cost = 0.01 if decision["operation"] == "replay" else 0.001
        target = torch.stack(
            [before_task - after_task, after_old - before_old, after_task.new_tensor(cost)]
        ).detach()
        development = functional.huber_loss(
            decision["development"][decision["index"]], target, reduction="sum"
        )
        internal_utility = target[0] - target[1].clamp_min(0) - cost
        delta = 0.01 * internal_utility - decision["value"]
        actor_internal = -delta.detach() * sum(
            (p * decision["eligibility"][name]).sum() for name, p in model.policy_named_parameters()
        )
        critic_internal = 0.5 * delta.square()
        losses = {
            "next_sensor": next_sensor,
            "reconstruction": reconstruction,
            "masked": masked,
            "task": task,
            "reward": reward,
            "done": done,
            "actor": torch.stack(policy_losses).sum() + actor_internal,
            "critic": torch.stack(critic_losses).sum() + critic_internal,
            "development": development,
            "predictive": torch.stack(predictive_losses).mean(),
        }
        losses["total"] = (
            losses["next_sensor"]
            + 0.1 * losses["reconstruction"]
            + 0.1 * losses["masked"]
            + losses["task"]
            + 0.5 * losses["reward"]
            + 0.1 * losses["done"]
            + losses["actor"]
            + losses["critic"]
            + 0.1 * losses["development"]
            + 0.1 * losses["predictive"]
        )
        trace = {
            "read_R_injection_loss": next_sensor + 0.1 * reconstruction,
            "record_value_parameter": "record_value.weight",
            "record_output_parameter": "record_output.weight",
            "events": len(fixture["events"]),
            "accepted_bytes": len(fixture["target"]),
            "world_unchanged_by_expression": True,
            "teacher_forcing_committed": False,
            "internal_operation": decision["operation"],
            "development_target": target,
            "receipts": receipts,
            "parameter_revision": getattr(model, "parameter_revision", 0),
            "verification_cuts": tuple(cuts),
            "credit_tape": list(model._credit_tape),
        }
        credit = {
            "native_input_width": 8 * model.c.d,
            "owner_map": model.parameter_owners(),
            "state_fields": tuple(sorted(after)),
            "birth_structure": model.structure(),
        }
        return losses, after, trace, credit
    finally:
        model._verification_cuts = previous_cuts
        model._capture_credit = previous_capture
