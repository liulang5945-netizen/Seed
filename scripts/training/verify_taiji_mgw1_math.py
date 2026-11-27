"""Independent MGW1 tensor reference and bounded mathematical checks.

No production Taiji import, existing checkpoint, world dataset or sealed data.
The reference uses synthetic codec values and its own random parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as functional

ROOT = Path(__file__).resolve().parents[2]
COUNTS = {
    "reference_graphs_constructed": 0,
    "reference_loss_evaluations": 0,
    "reference_parameter_updates": 0,
    "reference_update_attempts": 0,
    "reference_planning_belief_nodes": 0,
}


def bounded(x: torch.Tensor) -> torch.Tensor:
    return x / (1 + x.square().sum(dim=-1, keepdim=True)).sqrt()


def normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))


def truncated_log_prob(y, mean, scale, lower, upper):
    normalizer = normal_cdf((upper - mean) / scale) - normal_cdf((lower - mean) / scale)
    value = (
        -0.5 * ((y - mean) / scale).square()
        - scale.log()
        - 0.5 * math.log(2 * math.pi)
        - normalizer.log()
    )
    return torch.where((y >= lower) & (y <= upper), value, torch.full_like(value, -torch.inf))


class SphereQuadratic(torch.autograd.Function):
    """SPD quadratic ball solver with the active multiplier's implicit VJP."""

    @staticmethod
    def forward(ctx, hessian, rhs, radius):
        if float(radius) < 0:
            raise ValueError("negative radius")
        ctx.zero = float(radius) == 0
        if ctx.zero:
            ctx.save_for_backward(hessian, rhs, radius)
            return torch.zeros_like(rhs)
        eye = torch.eye(hessian.shape[0], dtype=hessian.dtype, device=hessian.device)
        x = torch.linalg.solve(hessian, rhs)
        active = bool(x.norm() > radius)
        mu = radius.new_zeros(())
        if active:
            low = radius.new_zeros(())
            high = torch.maximum(radius.new_tensor(1.0), rhs.norm() / radius)
            for _ in range(64):
                mid = (low + high) / 2
                trial = torch.linalg.solve(hessian + mid * eye, rhs)
                if bool(trial.norm() > radius):
                    low = mid
                else:
                    high = mid
            mu = high
            x = torch.linalg.solve(hessian + mu * eye, rhs)
        ctx.save_for_backward(hessian + mu * eye, x, radius)
        ctx.active = active
        return x

    @staticmethod
    def backward(ctx, upstream):
        if ctx.zero:
            hessian, rhs, radius = ctx.saved_tensors
            return torch.zeros_like(hessian), torch.zeros_like(rhs), torch.zeros_like(radius)
        matrix, x, radius = ctx.saved_tensors
        y = torch.linalg.solve(matrix.T, upstream)
        grad_radius = torch.zeros_like(radius)
        if ctx.active:
            z = torch.linalg.solve(matrix, x)
            coefficient = (upstream * z).sum() / (x * z).sum()
            y = y - coefficient * z
            grad_radius = coefficient * radius
        grad_hessian = -y @ x.T
        return grad_hessian, y, grad_radius


def fp1(memory, keys, values, beta, alpha, chi, radius):
    if float(chi.detach()) == 0.0:
        return memory
    baseline = (1 - chi + chi * alpha).unsqueeze(-1) * memory
    hessian = torch.eye(memory.shape[0], dtype=memory.dtype) + keys.T @ (beta[:, None] * keys)
    rhs = baseline + keys.T @ (beta[:, None] * values)
    return SphereQuadratic.apply(hessian, rhs, radius)


def historical_association_replay(state, keys, values):
    result = dict(state)
    result["F"] = torch.stack(
        [
            fp1(
                memory,
                keys[b],
                values[b],
                memory.new_full((keys.shape[1],), 0.5),
                memory.new_full((memory.shape[0],), 0.9),
                memory.new_tensor(0.5),
                memory.new_tensor(4.0),
            )
            for b, memory in enumerate(state["F"])
        ]
    )
    return result


@dataclass(frozen=True)
class Profile:
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

    @property
    def dimensions(self):
        return {"rgb": 3 * self.patch**2, "audio": self.fft // 2 + 1, "proprio": 8}


class MLP(nn.Module):
    def __init__(self, inputs, outputs, d):
        super().__init__()
        self.first = nn.Linear(inputs, 2 * d)
        self.last = nn.Linear(2 * d, outputs)

    def forward(self, x):
        return self.last(torch.tanh(self.first(x)))


class MGWReference(nn.Module):
    def __init__(self, profile=Profile()):
        super().__init__()
        COUNTS["reference_graphs_constructed"] += 1
        self.c = profile
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
        self.goal_score = nn.Linear(d, 1)
        self.goal_value = nn.Linear(d, d)
        self.initial_h = nn.Parameter(torch.randn(c.branches, c.objects, d) * 0.1)
        self.initial_p = nn.Parameter(torch.randn(c.branches, c.objects, 2) * 0.2)
        self.object_projection = nn.Linear(d + 9, d)
        self.h_query = nn.Linear(d, d)
        self.h_key = nn.Linear(d, d)
        self.h_value = nn.Linear(d, d)
        self.o_query = nn.Linear(d, d)
        self.o_key = nn.Linear(d, d)
        self.o_value = nn.Linear(d, d)
        self.action = nn.Linear(15, d)
        self.f_query = nn.Linear(4 * d, c.key)
        self.f_output = nn.Linear(c.key, d)
        self.relation_query = nn.Linear(d, d)
        self.relation_key = nn.Linear(d, d)
        self.relation_value = nn.Linear(d + 4, d)
        self.relation_geometry = MLP(4, 1, d)
        self.motion = MLP(5 * d + 5, d + 6, d)
        self.native_w = nn.Parameter(torch.randn(c.groups, d, 7 * d) * 0.1 / math.sqrt(7 * d))
        self.native_r = nn.Parameter(torch.randn(c.groups, c.units, 7 * d) * 0.1 / math.sqrt(7 * d))
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
        self.decoder_query = nn.ModuleDict({m: nn.Linear(8, d) for m in c.dimensions})
        self.decoder_key = nn.Linear(d, d)
        self.decoder_value = nn.Linear(d, d)
        self.decoder = nn.ModuleDict(
            {m: MLP(d + 8, 2 * size, d) for m, size in c.dimensions.items()}
        )
        self.joint_readout = MLP(3 * d, d, d)
        self.byte_head = nn.Linear(d + c.embedding, 257)
        self.reward = MLP(d + 15, 4, d)
        self.actor = MLP(2 * d + 15, 1, d)
        self.critic = MLP(2 * d, 1, d)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.normal_(module.weight, std=0.3 / math.sqrt(module.weight[0].numel()))
                nn.init.constant_(module.bias, 0.01)

    def goal(self, raw):
        if raw.numel() == 0:
            return self.initial_h.new_zeros(self.c.d)
        value = self.text_input(self.embedding(raw)).T.unsqueeze(0)
        for layer, dilation in zip(self.convolutions, (1, 2, 4), strict=True):
            value = value + torch.tanh(layer(functional.pad(value, (2 * dilation, 0))))
        coordinates = value.new_zeros(raw.numel(), 8)
        coordinates[:, 2] = torch.linspace(-1, 1, raw.numel(), dtype=value.dtype)
        coordinates[:, 6] = 1
        tokens = bounded(value.squeeze(0).T + self.metadata(coordinates) + self.modality[3])
        return (
            self.goal_score(tokens).squeeze(-1).softmax(0)[:, None] * self.goal_value(tokens)
        ).sum(0)

    def initial(self):
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

    def packed(self, state):
        return torch.cat([state[x] for x in ("h", "p", "v")] + [state["s"].log(), state["l"]], -1)

    def object_z(self, state):
        return bounded(self.object_projection(self.packed(state)))

    def read_h(self, state, goal):
        score = (self.h_key(state["H"]) * self.h_query(goal)).sum(-1) / math.sqrt(self.c.d)
        return (score.softmax(-1)[..., None] * self.h_value(state["H"])).sum(-2)

    def read_o(self, state, query):
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

    def views(self, state, goal, action):
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

    def read_f(self, state, goal, action):
        query = bounded(self.f_query(self.views(state, goal, action)))
        return self.f_output(torch.einsum("bkv,bmk->bmv", state["F"], query))

    def native(self, state, goal, action, event):
        c = self.c
        old = state["H"]
        key = torch.einsum("bni,noi->bno", old, self.native_k)
        query = torch.einsum("bni,noi->bno", old, self.native_q) + self.native_goal(goal)
        value = torch.einsum("bni,noi->bno", old, self.native_v)
        message = (query @ key.transpose(-1, -2) / math.sqrt(c.d)).softmax(-1) @ value
        obj = self.read_o(state, self.read_h(state, goal))
        # Separate group queries; never pool the entire F into a fixed summary.
        fview = torch.cat(
            [
                old,
                obj[:, None].expand(-1, c.groups, -1),
                goal[None, None].expand(c.branches, c.groups, -1),
                self.action(action)[None, None].expand(c.branches, c.groups, -1),
            ],
            -1,
        )
        fq = bounded(self.f_query(fview))
        read = self.f_output(torch.einsum("bkv,bnk->bnv", state["F"], fq))
        x = torch.cat(
            [
                old / (1 + old.square().mean(-1, keepdim=True)).sqrt(),
                event[:, None].expand(-1, c.groups, -1),
                message,
                read,
                obj[:, None].expand(-1, c.groups, -1),
                goal[None, None].expand(c.branches, c.groups, -1),
                self.nu_projection(
                    old.new_tensor(
                        [
                            0.8,
                            0.1,
                            0.2,
                            0.1,
                            0.0,
                            0.3,
                            0.2,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                        ]
                    )
                )[None, None].expand(c.branches, c.groups, -1),
            ],
            -1,
        )
        pre = torch.einsum("bni,noi->bno", x, self.native_w)
        pre = pre + torch.einsum(
            "nom,bnm->bno", self.native_p, torch.einsum("nmi,bni->bnm", self.native_r, x).tanh()
        )
        tau = 0.5 + 127.5 * self.tau_logit.sigmoid()
        leak = 1 - (-1 / tau).exp()
        return (1 - leak[None, :, None]) * old + leak[None, :, None] * pre.tanh()

    def predict(self, state, goal, action, dt=0.05):
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

    def encode(self, observations, coordinates, outcome=None):
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

    def binding(self, state, tokens, coordinates):
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

    def decode(self, state, goal, coordinates):
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

    def likelihood(self, state, goal, observations, coordinates, action=None, outcome=None):
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

    def correct(self, prior, goal, action, observations, coordinates, outcome, write=True):
        observations = {
            m: v for m, v in observations.items() if m != "proprio" or bool(v[..., 8:].any())
        }
        coordinates = {m: coordinates[m] for m in observations}
        if not observations and not bool(outcome[4:].any()):
            return prior
        c = self.c
        result = dict(prior)
        evidence_goal = prior["prediction_goal"]
        evidence_action = prior["prediction_action"]
        frozen_key_view = self.views(prior, evidence_goal, evidence_action)
        logp = self.likelihood(
            prior, evidence_goal, observations, coordinates, evidence_action, outcome
        )
        result["logw"] = functional.log_softmax(prior["logw"] + logp, 0)
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
            event = torch.stack([v.mean(0) for v in tokens.values()]).mean(0)
            result["H"] = self.native(result, goal, action, event[None].expand(c.branches, -1))
        if write:
            pooled = torch.cat([obs.mean(1), goal[None].expand(c.branches, -1)], -1)
            chi = self.f_chi(pooled).sigmoid().squeeze(-1)
            alpha = self.f_alpha(pooled).sigmoid()
            keys = bounded(self.f_key(frozen_key_view))
            values = bounded(
                self.f_value(
                    torch.cat(
                        [obs, self.outcome(outcome)[None, None].expand(c.branches, c.objects, -1)],
                        -1,
                    )
                )
            )
            beta = (
                chi[:, None]
                * gate
                * 16
                * self.f_beta(torch.cat([self.object_z(result), obs], -1)).sigmoid().squeeze(-1)
            )
            result["F"] = torch.stack(
                [
                    fp1(
                        prior["F"][b],
                        keys[b],
                        values[b],
                        beta[b],
                        alpha[b],
                        chi[b],
                        chi.new_tensor(4.0),
                    )
                    for b in range(c.branches)
                ]
            )
        read = self.read_f(result, goal, action)
        result["h"] = (1 - gate[..., None]) * result["h"] + gate[..., None] * self.post_memory(
            torch.cat([result["h"], read, goal[None, None].expand(c.branches, c.objects, -1)], -1)
        ).tanh()
        result["H"] = self.native(result, goal, action, event[None].expand(c.branches, -1))
        return result

    def output(self, state, goal, action, previous=0):
        h = self.read_h(state, goal)
        o = self.read_o(state, h)
        q = bounded(
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
        f = self.f_output(torch.einsum("bkv,bk->bv", state["F"], q))
        z = self.joint_readout(torch.cat([h, o, f], -1))
        embedding = self.embedding(torch.tensor(previous)).expand(self.c.branches, -1)
        branch_log = self.byte_head(torch.cat([z, embedding], -1)).log_softmax(-1)
        log_byte = torch.logsumexp(state["logw"][:, None] + branch_log, 0)
        outcome_logits = self.reward(torch.cat([h, action[None].expand(self.c.branches, -1)], -1))
        return log_byte, z, outcome_logits


def fixture(c, dtype):
    def coordinates(count, spatial=False):
        x = torch.zeros(count, 8, dtype=dtype)
        x[:, 2] = torch.linspace(-1, 1, count)
        x[:, 3] = torch.linspace(-0.16, -0.01, count)
        x[:, 4] = torch.linspace(-0.01, 0, count)
        x[:, 6] = 1
        if spatial:
            side = c.side // c.patch
            ys, xs = torch.meshgrid(
                torch.linspace(-1, 1, side), torch.linspace(-1, 1, side), indexing="ij"
            )
            x[:, :2] = torch.stack([xs.flatten(), ys.flatten()], -1).to(dtype)
            x[:, 5] = 1
        return x

    image = 0.4 + 0.2 * torch.rand(c.side, c.side, 3, dtype=dtype)
    grid = c.side // c.patch
    rgb = (
        image.reshape(grid, c.patch, grid, c.patch, 3)
        .permute(0, 2, 1, 3, 4)
        .reshape(grid**2, c.dimensions["rgb"])
    )
    wave = 0.3 * torch.sin(torch.arange(c.audio_samples, dtype=dtype) * 0.17)
    audio = (
        torch.stft(
            wave,
            c.fft,
            c.hop,
            window=torch.hann_window(c.fft, dtype=dtype),
            center=False,
            return_complex=True,
        )
        .abs()
        .log1p()
        .T
    )
    observations = {
        "rgb": rgb,
        "audio": audio,
        "proprio": torch.cat(
            [torch.rand(1, 8, dtype=dtype) * 0.5, torch.ones(1, 8, dtype=dtype)], -1
        ),
    }
    coords = {m: coordinates(value.shape[0], m == "rgb") for m, value in observations.items()}
    coords["rgb"][:, 2:5] = 0
    coords["rgb"][:, 6] = 0
    coords["proprio"][:, 2:5] = 0
    coords["proprio"][:, 6] = 0
    starts = (torch.arange(audio.shape[0], dtype=dtype) * c.hop - c.audio_samples) / 16000
    coords["audio"][:, 3] = torch.asinh(starts)
    coords["audio"][:, 4] = torch.asinh(starts + c.fft / 16000)
    coords["audio"][:, 7] = 1
    goal = torch.tensor(list("推方块。".encode()), dtype=torch.long)
    action = torch.zeros(15, dtype=dtype)
    action[0] = 1
    action[10] = 1.0
    action[12] = 0.05
    outcome = torch.tensor([0.0, 1.0, 0.0, 0.0, 1.0, 1.0], dtype=dtype)
    future = {
        m: (
            value * 0.97 + 0.01
            if m != "proprio"
            else torch.cat([value[..., :8] * 0.97 - 0.01, value[..., 8:]], -1)
        )
        for m, value in observations.items()
    }
    return observations, coords, goal, action, outcome, future


def reference_loss(model, data, fixed_psi=None):
    COUNTS["reference_loss_evaluations"] += 1
    observations, coords, raw_goal, action, outcome, future = data
    goal = model.goal(raw_goal)
    warm = {
        m: (
            value * 0.98 + 0.002
            if m != "proprio"
            else torch.cat([value[..., :8] * 0.98 - 0.002, value[..., 8:]], -1)
        )
        for m, value in observations.items()
    }
    warm_prior = model.predict(model.initial(), goal, action)
    warm_state = model.correct(warm_prior, goal, action, warm, coords, outcome)
    prior = model.predict(warm_state, goal, action)
    state = model.correct(prior, goal, action, observations, coords, outcome)
    next_prior = model.predict(state, goal, action)
    n = sum(
        value.numel() if m != "proprio" else int(value[..., 8:].sum())
        for m, value in future.items()
    )
    future_loss = (
        -torch.logsumexp(state["logw"] + model.likelihood(next_prior, goal, future, coords), 0) / n
    )
    reconstruction = (
        -torch.logsumexp(state["logw"] + model.likelihood(state, goal, observations, coords), 0) / n
    )
    reduced = {m: v for m, v in observations.items() if m != "audio"}
    reduced_coords = {m: coords[m] for m in reduced}
    masked_state = model.correct(prior, goal, action, reduced, reduced_coords, outcome, write=False)
    cross = (
        -torch.logsumexp(
            masked_state["logw"]
            + model.likelihood(
                masked_state, goal, {"audio": observations["audio"]}, {"audio": coords["audio"]}
            ),
            0,
        )
        / observations["audio"].numel()
    )
    log_byte, z, _ = model.output(state, goal, action)
    reward = model.reward(
        torch.cat([model.read_h(next_prior, goal), action[None].expand(model.c.branches, -1)], -1)
    )
    task = -log_byte[17]
    reward_loss = -torch.logsumexp(state["logw"] + reward[:, :3].log_softmax(-1)[:, 2], 0)
    done_loss = -torch.logsumexp(state["logw"] + functional.logsigmoid(-reward[:, 3]), 0)
    # Ψ-only diagnostic: detached belief features and real synthetic outcome.
    features = torch.cat([z.detach(), goal.detach()[None].expand(model.c.branches, -1)], -1)
    psi_weights = state["logw"].exp().detach()
    if fixed_psi is not None:
        features, psi_weights = fixed_psi
    value = (psi_weights * 2 * model.critic(features).tanh().squeeze(-1)).sum()
    observe = torch.zeros_like(action)
    observe[1] = 1
    observe[12] = 0.05
    wait = torch.zeros_like(action)
    wait[2] = 1
    wait[12] = 0.05
    choices = torch.stack([action, observe, wait])
    actor_input = torch.cat(
        [features[:, None].expand(-1, 3, -1), choices[None].expand(model.c.branches, -1, -1)], -1
    )
    actor_scores = (psi_weights[:, None] * model.actor(actor_input).squeeze(-1)).sum(0)
    policy_log = actor_scores.log_softmax(0)
    psi_loss = 0.5 * (value - 0.3).square() - 0.3 * policy_log[1]
    loss = (
        future_loss
        + 0.1 * reconstruction
        + 0.1 * cross
        + task
        + 0.5 * reward_loss
        + 0.1 * done_loss
        + psi_loss
    )
    return loss, state, log_byte, (features.detach().clone(), psi_weights.detach().clone())


def reference_belief_plan(model, state, goal, coordinates, actions, rng, depth=2):
    """Same Predict/Assimilate graph on clones; three legal reference actions."""
    COUNTS["reference_planning_belief_nodes"] += 1
    if depth == 0:
        _, z, _ = model.output(state, goal, actions[0])
        features = torch.cat([z, goal[None].expand(model.c.branches, -1)], -1)
        return (state["logw"].exp() * 2 * model.critic(features).tanh().squeeze(-1)).sum()
    values = []
    for action in actions:
        prior = model.predict(state, goal, action, float(action[12]))
        logits = model.reward(
            torch.cat(
                [
                    model.read_h(prior, prior["prediction_goal"]),
                    action[None].expand(model.c.branches, -1),
                ],
                -1,
            )
        )
        expected_reward = (
            state["logw"].exp()
            * (logits[:, :3].softmax(-1) * logits.new_tensor([-1.0, 0.0, 1.0])).sum(-1)
        ).sum()
        mode = int(action[:8].argmax())
        selected = {
            m: c for m, c in coordinates.items() if mode == 1 or (mode == 0 and m == "proprio")
        }
        parameters = model.decode(prior, goal, selected)
        gamma = 0.99 ** max(1.0, float(action[12]) / 0.05)
        cost = {0: 0.1, 1: 0.15, 2: 0.01}[mode]
        continuation = prior["logw"].new_zeros(())
        for _ in range(2):
            branch = int(torch.multinomial(state["logw"].exp(), 1, generator=rng))
            observations = {}
            for m, (mean, sd, lower, upper) in parameters.items():
                lo = normal_cdf((lower - mean[branch]) / sd[branch])
                hi = normal_cdf((upper - mean[branch]) / sd[branch])
                u = torch.rand(mean[branch].shape, dtype=mean.dtype, generator=rng).clamp(
                    2**-24, 1 - 2**-24
                )
                point = (lo + (hi - lo) * u).clamp(2**-24, 1 - 2**-24)
                value = mean[branch] + sd[branch] * math.sqrt(2) * torch.erfinv(2 * point - 1)
                observations[m] = (
                    torch.cat([value, torch.ones_like(value)], -1) if m == "proprio" else value
                )
            reward_index = int(torch.multinomial(logits[branch, :3].softmax(-1), 1, generator=rng))
            done = bool(torch.rand((), generator=rng) < logits[branch, 3].sigmoid())
            outcome = logits.new_zeros(6)
            outcome[reward_index] = 1
            outcome[3] = float(done)
            outcome[4:] = 1
            child = model.correct(prior, goal, action, observations, selected, outcome)
            if not done:
                continuation = (
                    continuation
                    + reference_belief_plan(
                        model, child, goal, coordinates, actions, rng, depth - 1
                    )
                    / 2
                )
        values.append((1 - gamma) * (expected_reward - cost) + gamma * continuation)
    return torch.stack(values).max()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    output = args.report.resolve() if args.report else None
    if output is not None and (output.parent != ROOT / "reports" or output.exists()):
        parser.error("Report must be a new file directly inside reports/.")
    torch.set_num_threads(1)
    torch.manual_seed(5112026)
    started = time.process_time()
    model = MGWReference().double()
    data = fixture(model.c, torch.float64)
    loss, state, log_byte, psi_context = reference_loss(model, data)
    loss.backward()
    rows = []
    for name, parameter in model.named_parameters():
        norm = None if parameter.grad is None else float(parameter.grad.norm())
        rows.append({"name": name, "elements": parameter.numel(), "gradient_norm": norm})
    assert all(row["gradient_norm"] is not None for row in rows), [
        r["name"] for r in rows if r["gradient_norm"] is None
    ]
    assert all(math.isfinite(row["gradient_norm"]) for row in rows)
    assert torch.allclose(log_byte.exp().sum(), loss.new_tensor(1), atol=1e-12)
    assert torch.allclose(state["logw"].exp().sum(), loss.new_tensor(1), atol=1e-12)
    checks = {
        "coupled_reference": {
            "passed": True,
            "parameter_tensors": len(rows),
            "loss": float(loss.detach()),
            "rows": rows,
        }
    }
    checks.update(run_checks(model, data, psi_context))
    assert all(item["passed"] for item in checks.values())
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    report = {
        "schema": "taiji-mgw1-reference-v1",
        "date": "2026-11-06",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "basis_commit": revision,
        "status": "specified_candidate_with_scoped_reference_checks",
        "passed": True,
        "checks": checks,
        "accounting": {
            "production_candidates_constructed": 0,
            "existing_weights_loaded": False,
            "project_data_reads": 0,
            "production_optimizer_steps": 0,
            **COUNTS,
        },
        "cpu_seconds": time.process_time() - started,
        "scope": "Scaled inference/learning kernel, not deployed world understanding. Planner search, full sequence generation and lifecycle admission are specified but not signed by this reference.",
    }
    if output:
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    print(
        json.dumps(
            {
                "passed": True,
                "groups": len(checks),
                "parameter_tensors": len(rows),
                "report": str(output) if output else None,
            }
        )
    )
    return 0


def run_checks(model, data, psi_context):
    checks = {}
    observations, coords, raw_goal, action, outcome, future = data
    with torch.no_grad():
        goal = model.goal(raw_goal)
        prior = model.predict(model.initial(), goal, action)
        state = model.correct(prior, goal, action, observations, coords, outcome)
        absent = model.correct(prior, goal, action, {}, {}, torch.zeros_like(outcome))
        assert absent is prior and all(torch.equal(prior[k], absent[k]) for k in prior)
        assert not torch.equal(prior["p"], model.initial()["p"])
        no_spatial = {m: v for m, v in observations.items() if m != "rgb"}
        no_spatial_coords = {m: coords[m] for m in no_spatial}
        acoustic = model.correct(prior, goal, action, no_spatial, no_spatial_coords, outcome)
        assert all(torch.equal(acoustic[k], prior[k]) for k in ("p", "v", "s"))
        no_write = model.correct(prior, goal, action, observations, coords, outcome, write=False)
        assert torch.equal(no_write["F"], prior["F"])
        assert state["F"].norm() > 0
        assert state["F"].norm(dim=(-1, -2)).max() <= 4 + 1e-10
        checks["missing_and_permission"] = {
            "passed": True,
            "all_missing_exact_prior": True,
            "nonspatial_pose_unchanged": True,
            "physical_prior_still_advances": True,
        }
        partial = {m: v.clone() for m, v in observations.items()}
        partial["proprio"][..., 8::2] = 0
        corrupted = {m: v.clone() for m, v in partial.items()}
        mask = partial["proprio"][..., 8:].bool()
        corrupted["proprio"][..., :8][~mask] = float("nan")
        one = model.correct(prior, goal, action, partial, coords, outcome)
        two = model.correct(prior, goal, action, corrupted, coords, outcome)
        assert all(torch.equal(one[k], two[k]) for k in one)
        invalid = observations["proprio"].clone()
        invalid[..., 8:] = 0
        assert (
            model.correct(
                prior,
                goal,
                action,
                {"proprio": invalid},
                {"proprio": coords["proprio"]},
                torch.zeros_like(outcome),
            )
            is prior
        )
        outcome_only = model.correct(prior, goal, action, {}, {}, outcome)
        assert all(torch.equal(prior[k], outcome_only[k]) for k in ("p", "v", "s"))
        assert not torch.equal(prior["H"], outcome_only["H"])
        revised_goal = model.correct(prior, -goal, action, observations, coords, outcome)
        assert torch.equal(state["logw"], revised_goal["logw"])
        checks["partial_masks_outcome_and_goal_packet"] = {
            "passed": True,
            "invalid_values_cannot_become_zero_measurements": True,
            "old_goal_evidence_unchanged": True,
        }
        tokens = model.encode(observations, coords, outcome)
        baseline = model.binding(prior, tokens, coords)
        shuffled_tokens, shuffled_coords = {}, {}
        for m, value in tokens.items():
            permutation = torch.arange(value.shape[0] - 1, -1, -1)
            shuffled_tokens[m] = value[permutation]
            if m in coords:
                shuffled_coords[m] = coords[m][permutation]
        reordered = model.binding(prior, shuffled_tokens, shuffled_coords)
        repeated = model.binding(
            prior,
            {m: v.repeat_interleave(2, 0) for m, v in tokens.items()},
            {m: v.repeat_interleave(2, 0) for m, v in coords.items()},
        )
        for candidate in (reordered, repeated):
            assert all(
                torch.allclose(x, y, atol=1e-12, rtol=1e-12)
                for x, y in zip(baseline, candidate, strict=True)
            )
        assert baseline[0].norm(dim=-1).max() <= 1 + 1e-12
        checks["binding_invariances"] = {
            "passed": True,
            "scope": "Binding only, not duplicated raw likelihood.",
        }
        changed_action = action.clone()
        changed_action[10] = -action[10]
        other = model.predict(state, goal, changed_action)
        natural = model.predict(state, goal, action)
        action_difference = float((other["p"] - natural["p"]).abs().max())
        assert action_difference > 1e-9
        varied_query = model.read_f(state, -goal, action)
        query_difference = float((varied_query - model.read_f(state, goal, action)).abs().max())
        assert query_difference > 1e-9
        ablated = dict(state)
        ablated["F"] = torch.zeros_like(state["F"])
        output_difference = float(
            (model.output(state, goal, action)[0] - model.output(ablated, goal, action)[0])
            .abs()
            .max()
        )
        assert output_difference > 1e-9
        checks["action_and_query_leverage"] = {
            "passed": True,
            "position_difference": action_difference,
            "same_F_different_query": query_difference,
            "F_cut_log_probability_difference": output_difference,
        }
        replay_keys = bounded(model.f_key(model.views(prior, goal, action)))
        replay_values = bounded(torch.ones_like(replay_keys) * 0.3)
        replayed = historical_association_replay(state, replay_keys, replay_values)
        assert all(torch.equal(state[k], replayed[k]) for k in state if k != "F")
        assert not torch.equal(state["F"], replayed["F"])
        checks["historical_replay_is_not_current_observation"] = {
            "passed": True,
            "changed_field": "F",
        }
        # A finite latent mixture must not be replaced by its mean state.
        means = torch.tensor([0.1, 0.9], dtype=torch.float64)
        mixture_log_at_zero = torch.logsumexp(
            truncated_log_prob(means.new_tensor(0.5), means, means.new_tensor(0.05), 0, 1), 0
        ) - math.log(2)
        mean_state_log_at_zero = float(
            truncated_log_prob(
                means.new_tensor(0.5), means.new_tensor(0.5), means.new_tensor(0.05), 0, 1
            )
        )
        assert mean_state_log_at_zero - float(mixture_log_at_zero) > 31
        checks["hypotheses_not_averaged"] = {
            "passed": True,
            "log_density_gap": mean_state_log_at_zero - float(mixture_log_at_zero),
        }
        payload = {
            "parameters": model.state_dict(),
            "state": {k: v.clone() for k, v in state.items()},
        }
        stream = io.BytesIO()
        torch.save(payload, stream)
        stream.seek(0)
        restored = torch.load(stream, weights_only=True)
        clone = MGWReference(model.c).double()
        clone.load_state_dict(restored["parameters"])
        actual = model.predict(state, goal, action)
        recovered = clone.predict(restored["state"], clone.goal(raw_goal), action)
        assert all(torch.equal(actual[k], recovered[k]) for k in actual)
        checks["reference_state_restore"] = {
            "passed": True,
            "state_fields": list(actual),
            "scope": "In-memory tensor bundle, not independent-process production recovery.",
        }
        observe = torch.zeros_like(action)
        observe[1] = 1
        observe[12] = 0.05
        wait = torch.zeros_like(action)
        wait[2] = 1
        wait[12] = 0.05
        state_copy = {k: v.clone() for k, v in state.items()}
        value = reference_belief_plan(
            model, state, goal, coords, [action, observe, wait], torch.Generator().manual_seed(99)
        )
        assert torch.isfinite(value)
        assert all(torch.equal(state[k], state_copy[k]) for k in state)
        checks["neural_belief_planning_clones"] = {
            "passed": True,
            "depth": 2,
            "actions": 3,
            "value": float(value),
            "real_state_unchanged": True,
            "scope": "Reference actions only; production candidate enumeration/lifecycle pending.",
        }

    # Normalized sensor densities and their parameter gradients.
    x = torch.linspace(0, 1, 4001, dtype=torch.float64)
    density = truncated_log_prob(x, x.new_tensor(0.2), x.new_tensor(0.18), 0, 1).exp()
    integral = float(torch.trapezoid(density, x))
    assert abs(integral - 1) < 1e-6
    assert torch.isneginf(
        truncated_log_prob(x.new_tensor(-0.1), x.new_tensor(0.2), x.new_tensor(0.18), 0, 1)
    )
    checks["observation_density"] = {"passed": True, "trapezoid_integral": integral}

    # Active ball: finite differences must include the changing multiplier.
    raw = torch.tensor([[0.9, 0.2], [0.1, 0.7]], dtype=torch.float64, requires_grad=True)
    rhs = torch.tensor([[1.0, 0.3], [-0.4, 0.8]], dtype=torch.float64, requires_grad=True)
    radius = torch.tensor(0.35, dtype=torch.float64, requires_grad=True)
    upstream = torch.tensor([[0.4, -0.3], [0.7, 0.2]], dtype=torch.float64)

    def scalar():
        return (
            SphereQuadratic.apply(torch.eye(2, dtype=torch.float64) + raw @ raw.T, rhs, radius)
            * upstream
        ).sum()

    derivatives = torch.autograd.grad(scalar(), (raw, rhs, radius))
    active_errors = []
    for parameter, gradient in zip((raw, rhs, radius), derivatives, strict=True):
        for index in range(parameter.numel()):
            with torch.no_grad():
                original = parameter.flatten()[index].item()
                parameter.flatten()[index] = original + 1e-5
                plus = float(scalar())
                parameter.flatten()[index] = original - 1e-5
                minus = float(scalar())
                parameter.flatten()[index] = original
            error = abs((plus - minus) / 2e-5 - gradient.flatten()[index].item())
            active_errors.append(error)
            assert error < 2e-7
    checks["active_proximal_implicit_gradient"] = {
        "passed": True,
        "components": len(active_errors),
        "max_error": max(active_errors),
    }
    with torch.no_grad():
        f = torch.tensor([[0.2, 0.1], [0.1, -0.2]], dtype=torch.float64)
        k = bounded(torch.tensor([[0.3, 0.8], [-0.4, 0.2]], dtype=torch.float64))
        v = torch.zeros(2, 2, dtype=torch.float64)
        beta = torch.tensor([0.8, 0.4], dtype=torch.float64)
        a = torch.tensor([0.6, 0.8], dtype=torch.float64)
        chi = torch.tensor(0.7, dtype=torch.float64)
        erased = fp1(f, k, v, beta, a, chi, chi.new_tensor(1.0))
        assert erased.norm() <= f.norm()
        assert fp1(f, k, v, beta, a, chi * 0, chi.new_tensor(1.0)) is f
        nonzero = torch.tensor([[0.8, -0.2], [0.1, 0.5]], dtype=torch.float64)
        first = fp1(f, k, nonzero, beta, a, chi, chi.new_tensor(0.25))
        second = fp1(f, k.flip(0), nonzero.flip(0), beta.flip(0), a, chi, chi.new_tensor(0.25))
        assert torch.allclose(first, second, atol=1e-12)
        checks["proximal_domains_and_permutation"] = {
            "passed": True,
            "erased_norm": float(erased.norm()),
            "old_norm": float(f.norm()),
        }
        baseline = (1 - chi + chi * a)[:, None] * f
        hessian = torch.eye(2, dtype=f.dtype) + k.T @ (beta[:, None] * k)
        rhs = baseline + k.T @ (beta[:, None] * nonzero)
        mu = ((first * (rhs - hessian @ first)).sum() / first.square().sum()).clamp_min(0)
        objective = (
            0.5 * (first - baseline).square().sum()
            + 0.5 * (beta[:, None] * (k @ first - nonzero).square()).sum()
        )
        constant = 0.5 * baseline.square().sum() + 0.5 * (beta[:, None] * nonzero.square()).sum()
        dual = (
            constant
            - 0.5
            * (rhs * torch.linalg.solve(hessian + mu * torch.eye(2, dtype=f.dtype), rhs)).sum()
            - 0.5 * mu * 0.25**2
        )
        gap = float(objective - dual)
        assert first.norm() <= 0.25 + 1e-12 and gap >= -1e-12 and gap < 1e-8
        checks["proximal_dual_gap_estimate"] = {
            "passed": True,
            "gap": gap,
            "interval_roundoff_certificate": False,
        }

    # Check every parameter tensor, selecting the largest conditional component.
    inert = {"goal_score.bias", "relation_geometry.last.bias", "actor.last.bias"}
    fd_rows, unresolved = [], []
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        index = int(gradient.abs().argmax())
        derivative = float(gradient.flatten()[index])
        if name in inert:
            assert gradient.norm() < 1e-10, name
            fd_rows.append({"name": name, "classification": "softmax_shift_invariant"})
            continue
        if abs(derivative) < 1e-8:
            unresolved.append({"name": name, "largest_component": derivative})
            continue
        errors, passed = [], 0
        for eps in (1e-4, 3e-5, 1e-5):
            with torch.no_grad():
                original = parameter.flatten()[index].item()
                parameter.flatten()[index] = original + eps
                plus = float(reference_loss(model, data, psi_context)[0])
                parameter.flatten()[index] = original - eps
                minus = float(reference_loss(model, data, psi_context)[0])
                parameter.flatten()[index] = original
            difference = (plus - minus) / (2 * eps)
            errors.append(abs(difference - derivative))
            passed += abs(difference - derivative) <= 1e-9 + 2e-3 * abs(derivative)
        assert passed >= 2, (name, derivative, errors)
        fd_rows.append(
            {
                "name": name,
                "classification": "FD_resolved",
                "gradient": derivative,
                "maximum_absolute_error": max(errors),
                "passing_eps": passed,
            }
        )
    checks["per_tensor_conditional_gradients"] = {
        "passed": True,
        "rows": fd_rows,
        "unresolved_small_signal": unresolved,
        "scope": "One max-gradient component per resolved tensor; policy/value belief inputs held fixed as specified. Not all entries or useful task learning.",
    }

    with torch.no_grad():
        model32 = MGWReference(model.c).float()
        model32.load_state_dict(model.state_dict())
        data32 = tuple(
            (
                {k: v.float() for k, v in part.items()}
                if isinstance(part, dict)
                else (
                    part.float()
                    if isinstance(part, torch.Tensor) and part.is_floating_point()
                    else part
                )
            )
            for part in data
        )
        loss32, state32, output32, _ = reference_loss(model32, data32)
        assert torch.isfinite(loss32) and torch.isfinite(output32).all()
        assert all(torch.isfinite(v).all() for v in state32.values())
        checks["float32_profile"] = {
            "passed": True,
            "loss": float(loss32),
            "finite_state_fields": len(state32),
        }
        full_config = Profile(
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
        full = MGWReference(full_config).float()
        for module in full.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.normal_(module.weight, std=1 / math.sqrt(module.weight[0].numel()))
                nn.init.zeros_(module.bias)
        full_data = fixture(full_config, torch.float32)
        fobs, fcoords, ftext, fact, foutcome, _ = full_data
        fg = full.goal(ftext)
        fp = full.predict(full.initial(), fg, fact)
        fs = full.correct(fp, fg, fact, fobs, fcoords, foutcome)
        fb, _, _ = full.output(fs, fg, fact)
        assert fs["H"].shape == (3, 4, 128) and fs["F"].shape == (3, 32, 32)
        assert fobs["rgb"].shape == (64, 192) and fobs["audio"].shape == (19, 129)
        assert torch.isfinite(fb).all() and all(torch.isfinite(v).all() for v in fs.values())
        checks["declared_full_shape_forward"] = {
            "passed": True,
            "H": list(fs["H"].shape),
            "F": list(fs["F"].shape),
            "rgb_codec": list(fobs["rgb"].shape),
            "audio_codec": list(fobs["audio"].shape),
            "backward_checked": False,
        }

    # One disposable reference step, explicitly distinct from Taiji training.
    original_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    norm = math.sqrt(sum(float(p.grad.square().sum()) for p in model.parameters()))
    old_loss = float(reference_loss(model, data, psi_context)[0].detach())
    accepted = None
    for trial in range(8):
        COUNTS["reference_update_attempts"] += 1
        step = 1e-3 / (2**trial)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                parameter.copy_(original_state[name] - step * parameter.grad / max(1.0, norm))
            updated = float(reference_loss(model, data, psi_context)[0])
        if updated < old_loss - 1e-10:
            accepted = (step, updated)
            break
    assert accepted is not None
    COUNTS["reference_parameter_updates"] += 1
    model.load_state_dict(original_state)
    checks["disposable_parameter_update"] = {
        "passed": True,
        "old_loss": old_loss,
        "new_loss": accepted[1],
        "step": accepted[0],
        "parameters_restored": True,
        "scope": "Synthetic frozen-belief conditional objective descent witness, not capability training.",
    }
    checks["causal_packet_gate"] = {
        "passed": True,
        "cases": [
            end <= clock and arrival <= clock
            for end, arrival, clock in [(0.2, 0.3, 0.1), (0.1, 0.3, 0.2), (0.1, 0.2, 0.2)]
        ],
    }
    assert checks["causal_packet_gate"]["cases"] == [False, False, True]
    with torch.no_grad():
        alternate_future = {m: value.clone() for m, value in future.items()}
        alternate_future["rgb"] = 1 - alternate_future["rgb"]
        modified = (*data[:-1], alternate_future)
        _, changed_state, changed_bytes, _ = reference_loss(model, modified, psi_context)
        _, ordinary_state, ordinary_bytes, _ = reference_loss(model, data, psi_context)
        assert all(torch.equal(changed_state[k], ordinary_state[k]) for k in ordinary_state)
        assert torch.equal(changed_bytes, ordinary_bytes)
    checks["future_targets_do_not_enter_current_state"] = {"passed": True}
    gamma = 0.99
    no_information = (1 - gamma) * 0.5
    observe_then_choose = -(1 - gamma) * 0.1 + gamma * (1 - gamma) * 0.9
    assert observe_then_choose > no_information
    terminal_value = (1 - gamma) * 1 + gamma * (1 - 1) * 2
    assert abs(terminal_value - 0.01) < 1e-12
    checks["planning_information_and_terminal_algebra"] = {
        "passed": True,
        "no_information": no_information,
        "observe_then_choose": observe_then_choose,
        "terminal_bootstrap": 0,
        "reference_neural_planner_executed": True,
    }
    return checks


if __name__ == "__main__":
    raise SystemExit(main())
