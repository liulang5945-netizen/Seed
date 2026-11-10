"""10.21.2: one query/binding candidate owned by the subject.

The module replaces neither the R021 graph nor the byte encoder. It reads the full
goal with two role queries, reads the legal records with a goal-conditioned source
query, binds object/place content through one shared nonlinear map, feeds that
binding as the mode-6 internal event of the same R021/growth step, and scores the
public candidate actions with a rank-4 bilinear form. Parent parameters stay
frozen; every new parameter belongs to the subject.

Engineering only: no task training, no pricing, no candidate adoption.
"""

from __future__ import annotations

import hashlib
import math
import traceback
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from taiji.candidate_manifest import parameter_digest
from taiji.living_core import EOS, LivingState, RawSource, position_features
from taiji.state_migration import resume_subject

BINDING_MODE = 6
BINDING_RANK = 4
BINDING_ROLES = 2
BINDING_SEED = 17021
DEFAULT_LR = 1e-3
DEFAULT_BETAS = (0.9, 0.999)
DEFAULT_EPS = 1e-8
CLIP_NORM = 1.0

PARAMETER_GROUPS: dict[str, tuple[str, ...]] = {
    "K_goal": ("K_goal",),
    "V_goal": ("V_goal", "V_goal_bias"),
    "role": ("role",),
    "W_state": ("W_state", "W_state_bias"),
    "Q_source": ("Q_source", "Q_source_bias"),
    "K_source": ("K_source",),
    "V_source": ("V_source", "V_source_bias"),
    "source_null": ("source_null_key", "source_null_value"),
    "W_bind": ("W_bind", "W_bind_bias"),
    "U": ("U",),
    "V": ("V",),
}
S_TRAINABLE_GROUPS: tuple[str, ...] = ("W_bind", "U", "V")


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _dtype_from_name(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def expected_parameter_count(width: int, query_width: int, roles: int, rank: int) -> int:
    """The exact count in SPEC §4.3, generalised only in d and k."""
    return (
        query_width * width
        + width * width
        + width
        + roles * query_width
        + query_width * width
        + query_width
        + query_width * (2 * width + query_width)
        + query_width
        + query_width * (width + 4)
        + width * (width + 4)
        + width
        + query_width
        + width
        + width * 3 * width
        + width
        + 2 * width * rank
    )


@dataclass(frozen=True)
class BindingConfig:
    width: int = 128
    query_width: int = 32
    roles: int = BINDING_ROLES
    rank: int = BINDING_RANK
    seed: int = BINDING_SEED
    mode: int = BINDING_MODE

    def parameter_count(self) -> int:
        return expected_parameter_count(self.width, self.query_width, self.roles, self.rank)


@dataclass(frozen=True)
class BindingFeatures:
    goal_bytes: bytes
    records: tuple[RawSource, ...]
    state: LivingState
    goal_features: torch.Tensor
    s: torch.Tensor
    source_positions: torch.Tensor
    source_metadata: torch.Tensor
    source_mask: torch.Tensor
    action_objects: torch.Tensor
    action_places: torch.Tensor
    cache_key: dict
    gate_index: int = 0


class QueryBindingCandidate(nn.Module):
    """Two role queries, one goal-conditioned source read, one shared binding map."""

    def __init__(self, config: BindingConfig) -> None:
        super().__init__()
        self.config = config
        width, query, roles, rank = (
            config.width,
            config.query_width,
            config.roles,
            config.rank,
        )
        generator = torch.Generator().manual_seed(config.seed)
        dtype = torch.float64

        def matrix(rows: int, columns: int) -> nn.Parameter:
            value = torch.empty(rows, columns, dtype=dtype)
            value.normal_(mean=0.0, std=1.0 / math.sqrt(columns), generator=generator)
            return nn.Parameter(value)

        def vector(size: int, std: float) -> nn.Parameter:
            value = torch.empty(size, dtype=dtype)
            value.normal_(mean=0.0, std=std, generator=generator)
            return nn.Parameter(value)

        self.K_goal = matrix(query, width)
        self.V_goal = matrix(width, width)
        self.V_goal_bias = nn.Parameter(torch.zeros(width, dtype=dtype))
        role_value = torch.empty(roles, query, dtype=dtype)
        role_value.normal_(mean=0.0, std=0.1, generator=generator)
        self.role = nn.Parameter(role_value)
        self.W_state = matrix(query, width)
        self.W_state_bias = nn.Parameter(torch.zeros(query, dtype=dtype))
        self.Q_source = matrix(query, 2 * width + query)
        self.Q_source_bias = nn.Parameter(torch.zeros(query, dtype=dtype))
        self.K_source = matrix(query, width + 4)
        self.V_source = matrix(width, width + 4)
        self.V_source_bias = nn.Parameter(torch.zeros(width, dtype=dtype))
        self.source_null_key = vector(query, 1.0 / math.sqrt(query))
        self.source_null_value = vector(width, 1.0 / math.sqrt(width))
        self.W_bind = matrix(width, 3 * width)
        self.W_bind_bias = nn.Parameter(torch.zeros(width, dtype=dtype))
        self.U = matrix_small(width, rank, generator, dtype)
        self.V = matrix_small(width, rank, generator, dtype)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def group_parameters(self, group: str) -> list[nn.Parameter]:
        if group not in PARAMETER_GROUPS:
            raise KeyError(f"unknown parameter group {group!r}")
        return [getattr(self, name) for name in PARAMETER_GROUPS[group]]

    def all_group_parameters(self) -> list[nn.Parameter]:
        return [
            parameter for group in PARAMETER_GROUPS for parameter in self.group_parameters(group)
        ]

    def goal_read(self, goal_features: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        c = self.config
        query = self.role + s.unsqueeze(0) @ self.W_state.T + self.W_state_bias
        keys = goal_features @ self.K_goal.T
        values = goal_features @ self.V_goal.T + self.V_goal_bias
        null_key = goal_features.new_zeros((1, c.query_width))
        null_value = self.V_goal_bias.unsqueeze(0)
        all_keys = torch.cat([keys, null_key], dim=0)
        all_values = torch.cat([values, null_value], dim=0)
        weights = torch.softmax(query @ all_keys.T / math.sqrt(c.query_width), dim=-1)
        return weights @ all_values

    def source_read(
        self,
        m: torch.Tensor,
        s: torch.Tensor,
        source_positions: torch.Tensor,
        source_metadata: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> torch.Tensor:
        c = self.config
        query_input = torch.cat([m, s.unsqueeze(0).expand(c.roles, -1), self.role], dim=-1)
        query = query_input @ self.Q_source.T + self.Q_source_bias
        key_value_input = torch.cat([source_positions, source_metadata], dim=-1)
        keys = key_value_input @ self.K_source.T
        values = key_value_input @ self.V_source.T + self.V_source_bias
        all_keys = torch.cat([keys, self.source_null_key.unsqueeze(0)], dim=0)
        all_values = torch.cat([values, self.source_null_value.unsqueeze(0)], dim=0)
        scores = query @ all_keys.T / math.sqrt(c.query_width)
        legal = torch.cat([source_mask.to(torch.bool), source_mask.new_ones(1, dtype=torch.bool)])
        scores = scores.masked_fill(~legal.unsqueeze(0), float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        return weights @ all_values

    def bind_goal(self, c_obj: torch.Tensor, c_place: torch.Tensor) -> torch.Tensor:
        joined = torch.cat([c_obj, c_place, c_obj * c_place], dim=-1)
        return torch.tanh(joined @ self.W_bind.T + self.W_bind_bias)

    def bind_actions(self, e_obj: torch.Tensor, e_place: torch.Tensor) -> torch.Tensor:
        joined = torch.cat([e_obj, e_place, e_obj * e_place], dim=-1)
        return torch.tanh(joined @ self.W_bind.T + self.W_bind_bias)

    def score(self, z_bind: torch.Tensor, b_actions: torch.Tensor) -> torch.Tensor:
        left = z_bind @ self.U
        right = b_actions @ self.V
        return right @ left

    def forward_binding(
        self,
        features: BindingFeatures,
        source_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = features.source_mask if source_mask is None else source_mask
        m = self.goal_read(features.goal_features, features.s)
        content = self.source_read(
            m, features.s, features.source_positions, features.source_metadata, mask
        )
        b_goal = self.bind_goal(content[0], content[1])
        b_actions = self.bind_actions(features.action_objects, features.action_places)
        return b_goal, b_actions, content


def matrix_small(
    rows: int, columns: int, generator: torch.Generator, dtype: torch.dtype
) -> nn.Parameter:
    value = torch.empty(rows, columns, dtype=dtype)
    value.normal_(mean=0.0, std=0.01, generator=generator)
    return nn.Parameter(value)


def build_binding(
    config: BindingConfig, dtype: torch.dtype = torch.float64
) -> QueryBindingCandidate:
    module = QueryBindingCandidate(config)
    module.to(dtype)
    return module


def freeze_parent(subject: nn.Module) -> None:
    for parameter in subject.parameters():
        parameter.requires_grad_(False)


def set_trainable(module: QueryBindingCandidate, groups: Iterable[str]) -> None:
    allowed = set(groups)
    for group in PARAMETER_GROUPS:
        for parameter in module.group_parameters(group):
            parameter.requires_grad_(group in allowed)


def trainable_parameters(
    module: QueryBindingCandidate, groups: Iterable[str]
) -> list[nn.Parameter]:
    return [
        parameter
        for group in groups
        for parameter in module.group_parameters(group)
        if parameter.requires_grad
    ]


def make_optimizer(module: QueryBindingCandidate, groups: Iterable[str]) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        trainable_parameters(module, groups),
        lr=DEFAULT_LR,
        betas=DEFAULT_BETAS,
        eps=DEFAULT_EPS,
        weight_decay=0.0,
    )


def gates_provider(table):
    def provider(index: int) -> dict:
        item = table[index % len(table)]
        if isinstance(item, dict):
            return dict(item)
        return {"copy": item[0], "generate": item[1]}

    return provider


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def records_sha(records: tuple[RawSource, ...]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(record.source_id.encode())
        digest.update(record.raw)
    return digest.hexdigest()


def state_digest(state: LivingState) -> str:
    digest = hashlib.sha256()
    for name, tensor in (
        ("h", state.h),
        ("fast", state.fast),
        ("nu_ema", state.nu_ema),
        ("coverage", state.coverage),
        ("prior", state.prior),
    ):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    digest.update(repr(state.keys).encode())
    return digest.hexdigest()


def features_sha(features: BindingFeatures) -> str:
    digest = hashlib.sha256()
    digest.update(features.goal_bytes)
    digest.update(records_sha(features.records).encode())
    digest.update(state_digest(features.state).encode())
    for name, tensor in (
        ("goal", features.goal_features),
        ("s", features.s),
        ("positions", features.source_positions),
        ("metadata", features.source_metadata),
        ("actions_obj", features.action_objects),
        ("actions_place", features.action_places),
    ):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def binding_step(
    subject,
    features: BindingFeatures,
    b_goal: torch.Tensor,
    gate_provider,
) -> tuple[torch.Tensor, torch.Tensor, Any]:
    c = subject.config
    allowed = torch.zeros((c.groups, c.fast_heads), dtype=torch.bool, device=b_goal.device)
    output = subject.step(
        features.state,
        event=b_goal,
        goal=features.goal_bytes,
        records=features.records,
        previous_byte=EOS,
        mode=BINDING_MODE,
        allowed=allowed,
        growth_gates=gate_provider(features.gate_index),
    )
    return output.trace["z"], output.state.h, output


def score_actions(
    module: QueryBindingCandidate, z_bind: torch.Tensor, b_actions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = module.score(z_bind, b_actions)
    return scores, torch.softmax(scores, dim=0)


def binding_loss(scores: torch.Tensor, target: int) -> torch.Tensor:
    return -torch.log_softmax(scores, dim=0)[target]


def build_binding_features(
    subject,
    state: LivingState,
    goal: bytes,
    records: tuple[RawSource, ...],
    object_texts: tuple[str, ...],
    place_texts: tuple[str, ...],
    gate_provider,
    identity: dict,
) -> BindingFeatures:
    """Freeze the parent features once; the binding step itself stays live."""
    with torch.no_grad():
        s = subject.W.new_zeros(subject.config.width)
        consumed = state
        gate_index = 0
        for patch in subject.encoder.patches(goal):
            output = subject.step(
                consumed,
                event=patch,
                goal=goal,
                records=records,
                growth_gates=gate_provider(gate_index),
            )
            consumed = output.state
            s = output.trace["z"]
            gate_index += 1
        reference = subject.W
        goal_features = subject.encoder.context(goal)
        goal_features = goal_features + position_features(
            len(goal), subject.config.width, reference
        )
        view = subject.source_view(records, "owner", "scope", 0)
        action_objects = torch.stack(
            [subject.encoder.goal_vector(text.encode()) for text in object_texts]
        )
        action_places = torch.stack(
            [subject.encoder.goal_vector(text.encode()) for text in place_texts]
        )
        mask = torch.ones(len(view.keys), dtype=torch.bool, device=reference.device)
    cache_key = {
        "parent_candidate_id": identity.get("parent_candidate_id"),
        "parent_bundle_sha256": identity.get("parent_bundle_sha256"),
        "parent_revision": state.revision,
        "parent_step": state.step,
        "goal_sha256": sha256_bytes(goal),
        "records_sha256": records_sha(records),
        "state_digest": state_digest(state),
        "source_hashes": identity.get("source_hashes"),
        "action_sha256": sha256_bytes(
            ("|".join(object_texts) + "#" + "|".join(place_texts)).encode()
        ),
    }
    return BindingFeatures(
        goal_bytes=goal,
        records=records,
        state=consumed,
        goal_features=goal_features,
        s=s,
        source_positions=view.positions,
        source_metadata=view.metadata,
        source_mask=mask,
        action_objects=action_objects,
        action_places=action_places,
        cache_key=cache_key,
        gate_index=gate_index,
    )


def parent_bundle_payload(subject, state: LivingState) -> dict:
    return {
        "schema": "taiji-subject-bundle-r024-v1",
        "subject_id": state.subject_id,
        "revision": state.revision,
        "step": state.step,
        "config": dict(vars(subject.config)),
        "parameters": {
            name: parameter.detach().cpu().clone() for name, parameter in subject.named_parameters()
        },
        "growth_parent_units": subject.growth_parent_units,
        "growth_unit_branches": subject.growth_unit_branches,
        "h": state.h.detach().cpu().clone(),
        "fast": state.fast.detach().cpu().clone(),
        "nu_ema": state.nu_ema.detach().cpu().clone(),
        "keys": tuple(state.keys),
        "coverage": state.coverage.detach().cpu().clone(),
        "prior": state.prior.detach().cpu().clone(),
        "optimizer_state": None,
        "optimizer_config": {
            "lr": DEFAULT_LR,
            "betas": list(DEFAULT_BETAS),
            "eps": DEFAULT_EPS,
        },
        "rng_state": torch.get_rng_state(),
    }


def save_binding_bundle(
    path: Path,
    profile: str,
    subject,
    state: LivingState,
    module: QueryBindingCandidate,
    optimizer: torch.optim.Adam,
    cursor: dict,
    allowed_groups: tuple[str, ...],
    growth_gates,
) -> dict:
    bundle = {
        "schema": "taiji-query-binding-bundle-v1",
        "step": "10.21.2",
        "profile": profile,
        "dtype": _dtype_name(next(module.parameters()).dtype),
        "parent": parent_bundle_payload(subject, state),
        "binding": {
            "config": asdict(module.config),
            "parameters": {
                name: parameter.detach().cpu().clone()
                for name, parameter in module.named_parameters()
            },
        },
        "optimizer": optimizer.state_dict(),
        "cursor": dict(cursor),
        "allowed_groups": list(allowed_groups),
        "growth_gates": growth_gates,
        "rng_state": torch.get_rng_state(),
    }
    torch.save(bundle, path)
    return {
        "path": str(path),
        "bytes": Path(path).stat().st_size,
        "sha256": sha256_bytes(Path(path).read_bytes()),
    }


def load_binding_bundle(path: Path, dtype: torch.dtype):
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    parent, parent_state, _ = resume_subject(bundle["parent"])
    parent = parent.to(dtype)
    parent.eval()
    freeze_parent(parent)
    module = QueryBindingCandidate(BindingConfig(**bundle["binding"]["config"]))
    module.load_state_dict(bundle["binding"]["parameters"], strict=True)
    module = module.to(dtype)
    set_trainable(module, bundle["allowed_groups"])
    module.train()
    optimizer = make_optimizer(module, bundle["allowed_groups"])
    optimizer.load_state_dict(bundle["optimizer"])
    torch.set_rng_state(bundle["rng_state"])
    return (
        parent,
        parent_state,
        module,
        optimizer,
        bundle["cursor"],
        gates_provider(bundle["growth_gates"]),
        bundle["profile"],
        _dtype_from_name(bundle["dtype"]),
        tuple(bundle["allowed_groups"]),
    )


def run_binding_worker(spec: dict) -> dict:
    """Independent-process restore: forward, or one declared synthetic Adam step."""
    import time

    start_cpu, start_wall = time.process_time(), time.perf_counter()
    torch.set_num_threads(1)
    result_path = Path(spec["result"])
    try:
        (
            parent,
            state,
            module,
            optimizer,
            cursor,
            gates,
            profile,
            _dtype,
            allowed_groups,
        ) = load_binding_bundle(Path(spec["bundle"]), _dtype_from_name(spec["dtype"]))
        fixture = spec["fixtures"][cursor["fixture_index"]]
        records = tuple(
            RawSource(source_id=item["source_id"], raw=item["raw"].encode())
            for item in fixture["records"]
        )
        features = build_binding_features(
            parent,
            state,
            fixture["goal"].encode(),
            records,
            tuple(item[0] for item in fixture["actions"]),
            tuple(item[1] for item in fixture["actions"]),
            gates,
            spec["identity"],
        )
        b_goal, b_actions, _ = module.forward_binding(features)
        z_bind, h_state, output = binding_step(parent, features, b_goal, gates)
        scores, probabilities = score_actions(module, z_bind, b_actions)
        loss = binding_loss(scores, fixture["target"])
        result = {
            "ok": True,
            "profile": profile,
            "features_sha": features_sha(features),
            "scores": scores.detach(),
            "probabilities": probabilities.detach(),
            "h": h_state.detach(),
            "fast_unchanged": bool(torch.equal(output.state.fast, features.state.fast)),
            "loss": float(loss.detach()),
        }
        if spec["mode"] == "update":
            optimizer.zero_grad()
            loss.backward()
            trainable = trainable_parameters(module, allowed_groups)
            clip = float(torch.nn.utils.clip_grad_norm_(trainable, CLIP_NORM))
            optimizer.step()
            post_features = features
            b_goal_post, b_actions_post, _ = module.forward_binding(post_features)
            z_post, h_post, _ = binding_step(parent, post_features, b_goal_post, gates)
            post_scores, post_probabilities = score_actions(module, z_post, b_actions_post)
            result.update(
                {
                    "parameters": {
                        name: parameter.detach().clone()
                        for name, parameter in module.named_parameters()
                    },
                    "optimizer": optimizer.state_dict(),
                    "post_scores": post_scores.detach(),
                    "post_probabilities": post_probabilities.detach(),
                    "post_h": h_post.detach(),
                    "grad_norm": float(
                        torch.sqrt(
                            cast(
                                torch.Tensor,
                                sum(
                                    parameter.grad.detach().square().sum()
                                    for parameter in trainable
                                    if parameter.grad is not None
                                ),
                            )
                        )
                    ),
                    "clip_scale": clip,
                }
            )
        result["cpu_seconds"] = time.process_time() - start_cpu
        result["wall_seconds"] = time.perf_counter() - start_wall
        torch.save(result, result_path)
        return {"ok": True, "result": str(result_path)}
    except Exception as exc:  # noqa: BLE001
        failure = {
            "ok": False,
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "cpu_seconds": time.process_time() - start_cpu,
            "wall_seconds": time.perf_counter() - start_wall,
        }
        torch.save(failure, result_path)
        return {"ok": False, "result": str(result_path)}


def module_bundle_payload(
    profile: str,
    config: BindingConfig,
    module: QueryBindingCandidate,
    optimizer: torch.optim.Adam,
    features: BindingFeatures,
    z_bind: torch.Tensor | None,
    target: int,
    allowed_groups: tuple[str, ...],
    cursor: dict,
) -> dict:
    return {
        "schema": "taiji-query-binding-module-bundle-v1",
        "step": "10.21.2",
        "profile": profile,
        "dtype": _dtype_name(next(module.parameters()).dtype),
        "config": asdict(config),
        "parameters": {
            name: parameter.detach().cpu().clone() for name, parameter in module.named_parameters()
        },
        "optimizer": optimizer.state_dict(),
        "features": {
            "goal": features.goal_bytes,
            "goal_features": features.goal_features.detach().cpu().clone(),
            "s": features.s.detach().cpu().clone(),
            "source_positions": features.source_positions.detach().cpu().clone(),
            "source_metadata": features.source_metadata.detach().cpu().clone(),
            "source_mask": features.source_mask.detach().cpu().clone(),
            "action_objects": features.action_objects.detach().cpu().clone(),
            "action_places": features.action_places.detach().cpu().clone(),
        },
        "z_bind": None if z_bind is None else z_bind.detach().cpu().clone(),
        "target": target,
        "allowed_groups": list(allowed_groups),
        "cursor": dict(cursor),
    }


def save_module_bundle(path: Path, payload: dict) -> dict:
    torch.save(payload, path)
    return {
        "path": str(path),
        "bytes": Path(path).stat().st_size,
        "sha256": sha256_bytes(Path(path).read_bytes()),
    }


def module_fixture_state() -> LivingState:
    zero = torch.zeros(1, dtype=torch.float64)
    return LivingState(
        "module-fixture",
        0,
        0,
        zero,
        zero,
        torch.zeros(8, dtype=torch.float64),
        (),
        torch.zeros(0, dtype=torch.float64),
        torch.zeros(0, dtype=torch.float64),
    )


def run_module_worker(spec: dict) -> dict:
    """Independent-process restore for the d=8 isomorphic module fixture."""
    import time

    start_cpu, start_wall = time.process_time(), time.perf_counter()
    torch.set_num_threads(1)
    result_path = Path(spec["result"])
    try:
        bundle = torch.load(Path(spec["bundle"]), map_location="cpu", weights_only=True)
        config = BindingConfig(**bundle["config"])
        module = QueryBindingCandidate(config).to(_dtype_from_name(bundle["dtype"]))
        module.load_state_dict(bundle["parameters"], strict=True)
        set_trainable(module, bundle["allowed_groups"])
        module.train()
        optimizer = make_optimizer(module, bundle["allowed_groups"])
        optimizer.load_state_dict(bundle["optimizer"])
        features = BindingFeatures(
            goal_bytes=bundle["features"]["goal"],
            records=(),
            state=module_fixture_state(),
            goal_features=bundle["features"]["goal_features"],
            s=bundle["features"]["s"],
            source_positions=bundle["features"]["source_positions"],
            source_metadata=bundle["features"]["source_metadata"],
            source_mask=bundle["features"]["source_mask"],
            action_objects=bundle["features"]["action_objects"],
            action_places=bundle["features"]["action_places"],
            cache_key={},
        )
        b_goal, b_actions, _ = module.forward_binding(features)
        z_bind = b_goal if bundle["z_bind"] is None else bundle["z_bind"]
        scores, probabilities = score_actions(module, z_bind, b_actions)
        loss = binding_loss(scores, bundle["target"])
        result = {
            "ok": True,
            "profile": bundle["profile"],
            "features_sha": features_sha(features),
            "scores": scores.detach(),
            "probabilities": probabilities.detach(),
            "loss": float(loss.detach()),
        }
        if spec["mode"] == "update":
            optimizer.zero_grad()
            loss.backward()
            trainable = trainable_parameters(module, bundle["allowed_groups"])
            clip = float(torch.nn.utils.clip_grad_norm_(trainable, CLIP_NORM))
            optimizer.step()
            post_b_goal, post_actions, _ = module.forward_binding(features)
            post_z = post_b_goal if bundle["z_bind"] is None else bundle["z_bind"]
            post_scores, post_probabilities = score_actions(module, post_z, post_actions)
            result.update(
                {
                    "parameters": {
                        name: parameter.detach().clone()
                        for name, parameter in module.named_parameters()
                    },
                    "optimizer": optimizer.state_dict(),
                    "post_scores": post_scores.detach(),
                    "post_probabilities": post_probabilities.detach(),
                    "grad_norm": float(
                        torch.sqrt(
                            cast(
                                torch.Tensor,
                                sum(
                                    parameter.grad.detach().square().sum()
                                    for parameter in trainable
                                    if parameter.grad is not None
                                ),
                            )
                        )
                    ),
                    "clip_scale": clip,
                }
            )
        result["cpu_seconds"] = time.process_time() - start_cpu
        result["wall_seconds"] = time.perf_counter() - start_wall
        torch.save(result, result_path)
        return {"ok": True, "result": str(result_path)}
    except Exception as exc:  # noqa: BLE001
        failure = {
            "ok": False,
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "cpu_seconds": time.process_time() - start_cpu,
            "wall_seconds": time.perf_counter() - start_wall,
        }
        torch.save(failure, result_path)
        return {"ok": False, "result": str(result_path)}


def module_parameter_digest(module: QueryBindingCandidate) -> str:
    return str(
        parameter_digest(
            {name: parameter.detach() for name, parameter in module.named_parameters()}
        )
    )


def optimizer_moment_norms(optimizer: torch.optim.Adam) -> dict:
    rows = {}
    for group in optimizer.param_groups:
        for parameter, state in optimizer.state.items():
            if not any(parameter is item for item in group["params"]):
                continue
            rows[str(id(parameter))] = {
                "step": int(state["step"]),
                "exp_avg_norm": float(state["exp_avg"].norm()),
                "exp_avg_sq_norm": float(state["exp_avg_sq"].norm()),
            }
    return rows
