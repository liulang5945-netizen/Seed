"""N17.4 checkpoint preflight: routing candidate container, bundle I/O, digests.

This module is the restore gate between the frozen 17.0 graph and the formal
17.5 dose.  It does not change any 17 selection-routing judgement or equation;
it only wraps the inherited binding graph with a fresh ordered selection head,
persists that pair with Adam moments and RNG state, and produces ordered f64
digests so two processes can be compared bit for bit.

The bundle schema is its own (``taiji-selection-routing-bundle-17-v1``); a
17.4 artefact must never be mistaken for a 16.1 selection bundle.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from taiji.binding_copy import BindingCopyCandidate, BindingState
from taiji.binding_select import (
    candidate_parameter_map,
    load_bundle,
    render_view_bundle,
)
from taiji.record_relation_core import state_from_payload
from taiji.selection_routing import (
    ARMS,
    OrderedSelectionHead,
    RoutingConfig,
    SpanPointerHead,
    expected_span_parameter_count,
)

ROUTING_BUNDLE_SCHEMA = "taiji-selection-routing-bundle-17-v1"
ROUTING_STEP = "N17.4"
ADAM_LR = 1e-3
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
ADAM_WEIGHT_DECAY = 0.0


class RoutingCandidate:
    """Binding-copy candidate plus an independent ordered selection head.

    The inherited graph is kept as one plain reference (``binding``); the head
    is separate.  Nothing is registered twice: parameter iteration walks the
    binding's own modules once and then the head's own modules once.
    """

    def __init__(
        self,
        binding: BindingCopyCandidate,
        head: OrderedSelectionHead,
        *,
        arm: str,
        seed: int,
        span_head: SpanPointerHead | None = None,
    ) -> None:
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}")
        self.binding = binding
        self.head = head
        self.span_head = span_head
        self.arm = arm
        self.seed = int(seed)

    @property
    def base(self) -> Any:
        return self.binding.base

    @property
    def subject(self) -> Any:
        return self.binding.subject

    @property
    def reader(self) -> Any:
        return self.binding.reader

    @property
    def growth_gates(self) -> list[Any]:
        return self.binding.growth_gates

    def parameters(self) -> Iterable[nn.Parameter]:
        yield from self.binding.parameters()
        yield from self.head.parameters()
        if self.span_head is not None:
            yield from self.span_head.parameters()

    def trainable_parameters(self) -> list[nn.Parameter]:
        parameters = list(self.binding.trainable_parameters()) + list(self.head.parameters())
        if self.span_head is not None:
            parameters += list(self.span_head.parameters())
        return parameters

    def configure_trainable(self) -> None:
        self.binding.configure_trainable()
        for parameter in self.head.parameters():
            parameter.requires_grad_(True)
        if self.span_head is not None:
            for parameter in self.span_head.parameters():
                parameter.requires_grad_(True)

    def to(self, dtype: torch.dtype) -> RoutingCandidate:
        self.binding.to(dtype)
        self.head.to(dtype)
        if self.span_head is not None:
            self.span_head.to(dtype)
        return self

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def make_routing_optimizer(candidate: RoutingCandidate) -> torch.optim.Adam:
    return torch.optim.Adam(
        list(candidate.trainable_parameters()),
        lr=ADAM_LR,
        betas=ADAM_BETAS,
        eps=ADAM_EPS,
        weight_decay=ADAM_WEIGHT_DECAY,
    )


def build_routing_candidate(
    parent_path: str | Path, seed: int, arm: str
) -> tuple[RoutingCandidate, torch.optim.Adam]:
    """Load the 16.1 parent, keep only its binding, attach a fresh 17 head.

    The parent's legacy ``SelectionHead`` is discarded.  Module registration
    order is fixed by the class definitions, so two calls with the same parent
    produce the same parameter order.
    """
    parent = Path(parent_path)
    loaded = load_bundle(parent, torch.float64)
    binding = loaded[0].binding
    head = OrderedSelectionHead(RoutingConfig(), seed=int(seed))
    span_head = SpanPointerHead(RoutingConfig(), seed=int(seed))
    candidate = RoutingCandidate(binding, head, arm=arm, seed=int(seed), span_head=span_head)
    candidate.configure_trainable()
    optimizer = make_routing_optimizer(candidate)
    return candidate, optimizer


def _trainable_names(candidate: RoutingCandidate) -> list[str]:
    names = [
        f"binding.{name}"
        for name, parameter in candidate.binding.named_parameters()
        if parameter.requires_grad
    ]
    names += [f"head.{name}" for name, _parameter in candidate.head.named_parameters()]
    if candidate.span_head is not None:
        names += [f"span.{name}" for name, _parameter in candidate.span_head.named_parameters()]
    return names


def binding_state_from_payload(
    payload: dict[str, Any], binding: BindingCopyCandidate
) -> BindingState:
    """Rebuild a ``BindingState`` (including the mid-sequence m/coverage)."""
    living = state_from_payload(payload["living"], binding.subject.W)
    reference = binding.subject.W
    return BindingState(
        living=living,
        m=payload["m"].to(reference.dtype),
        coverage=payload["coverage"].to(reference.dtype),
        keys=tuple((source, version, offset) for source, version, offset in payload["keys"]),
        record_offsets=tuple((int(start), int(end)) for start, end in payload["record_offsets"]),
        prefix_cursor=int(payload["prefix_cursor"]),
    )


def save_routing_bundle(
    path: str | Path,
    candidate: RoutingCandidate,
    optimizer: torch.optim.Adam,
    update_index: int,
    cursor: dict[str, Any],
    parent_identity: dict[str, Any],
    state: BindingState,
) -> dict[str, Any]:
    """Persist binding + head + Adam + RNG + state + parent summary."""
    target = Path(path)
    payload = {
        "schema": ROUTING_BUNDLE_SCHEMA,
        "step": ROUTING_STEP,
        "profile": candidate.binding.profile,
        "dtype": str(next(candidate.binding.parameters()).dtype).replace("torch.", ""),
        "relation_config": asdict(candidate.binding.base.config),
        "binding_config": asdict(candidate.binding.config),
        "routing_config": asdict(candidate.head.config),
        "subject_config": dict(vars(candidate.binding.subject.config)),
        "growth_parent_units": getattr(candidate.binding.subject, "growth_parent_units", None),
        "growth_unit_branches": getattr(candidate.binding.subject, "growth_unit_branches", []),
        "growth_gates": list(candidate.binding.growth_gates),
        "base_arm": candidate.binding.base.arm,
        "arm": candidate.arm,
        "seed": candidate.seed,
        "parent": dict(parent_identity),
        "parameters": candidate_parameter_map(candidate.binding),
        "head_parameters": dict(candidate.head.state_dict()),
        "head_parameter_count": sum(parameter.numel() for parameter in candidate.head.parameters()),
        "span_head_parameters": (
            dict(candidate.span_head.state_dict()) if candidate.span_head is not None else {}
        ),
        "span_head_parameter_count": (
            sum(parameter.numel() for parameter in candidate.span_head.parameters())
            if candidate.span_head is not None
            else 0
        ),
        "trainable_names": _trainable_names(candidate),
        "state": state.payload(),
        "optimizer": optimizer.state_dict(),
        "update_index": int(update_index),
        "cursor": dict(cursor),
        "rng_state": torch.get_rng_state().clone(),
    }
    torch.save(payload, target)
    return {
        "path": str(target),
        "bytes": target.stat().st_size,
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }


def load_routing_bundle(path: str | Path) -> tuple[
    RoutingCandidate,
    torch.optim.Adam,
    BindingState,
    int,
    dict[str, Any],
]:
    """Rebuild the exact architecture, then load every persisted surface."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("schema") != ROUTING_BUNDLE_SCHEMA:
        raise ValueError(f"routing bundle schema drift: {payload.get('schema')!r}")
    binding = render_view_bundle(payload)
    head = OrderedSelectionHead(
        RoutingConfig(**payload["routing_config"]), seed=int(payload["seed"])
    )
    head.load_state_dict(payload["head_parameters"])
    span_head = None
    if payload.get("span_head_parameters"):
        span_head = SpanPointerHead(
            RoutingConfig(**payload["routing_config"]), seed=int(payload["seed"])
        )
        span_head.load_state_dict(payload["span_head_parameters"])
        if int(payload["span_head_parameter_count"]) != expected_span_parameter_count(
            RoutingConfig(**payload["routing_config"])
        ):
            raise ValueError("routing bundle span head parameter count drifted")
    candidate = RoutingCandidate(
        binding,
        head,
        arm=str(payload["arm"]),
        seed=int(payload["seed"]),
        span_head=span_head,
    )
    candidate.configure_trainable()
    names = _trainable_names(candidate)
    if names != list(payload["trainable_names"]):
        raise ValueError("routing bundle trainable parameter order drifted")
    optimizer = make_routing_optimizer(candidate)
    optimizer.load_state_dict(payload["optimizer"])
    state = binding_state_from_payload(payload["state"], binding)
    torch.set_rng_state(payload["rng_state"])
    return (
        candidate,
        optimizer,
        state,
        int(payload["update_index"]),
        dict(payload["cursor"]),
    )


def parameter_tensors(candidate: RoutingCandidate) -> dict[str, torch.Tensor]:
    """Ordered name -> f64 tensor map, binding first then head."""
    tensors: dict[str, torch.Tensor] = {}
    for name, parameter in candidate.binding.named_parameters():
        tensors[f"binding.{name}"] = parameter.detach().to(torch.float64).cpu().clone()
    for name, parameter in candidate.head.named_parameters():
        tensors[f"head.{name}"] = parameter.detach().to(torch.float64).cpu().clone()
    if candidate.span_head is not None:
        for name, parameter in candidate.span_head.named_parameters():
            tensors[f"span.{name}"] = parameter.detach().to(torch.float64).cpu().clone()
    return tensors


def optimizer_moment_tensors(optimizer: torch.optim.Adam) -> dict[str, torch.Tensor]:
    """Ordered ``param_index/field`` -> f64 Adam moment map."""
    state = optimizer.state_dict()["state"]
    tensors: dict[str, torch.Tensor] = {}
    for index in sorted(state, key=lambda value: int(value)):
        entry = state[index]
        for field in ("exp_avg", "exp_avg_sq"):
            value = entry.get(field)
            if isinstance(value, torch.Tensor):
                tensors[f"{index}/{field}"] = value.detach().to(torch.float64).cpu().clone()
    return tensors


def _digest_map(tensors: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in tensors.items():
        array = value.detach().to(torch.float64).cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(f"|{tuple(array.shape)}|f64|".encode())
        digest.update(array.numpy().tobytes())
    return digest.hexdigest()


def _max_abs_diff(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> float:
    if set(left) != set(right):
        return float("inf")
    worst = 0.0
    for name, value in left.items():
        other = right[name]
        if tuple(value.shape) != tuple(other.shape):
            return float("inf")
        if value.numel() == 0:
            continue
        worst = max(worst, float((value - other).abs().max()))
    return worst


def _reference_map(reference: Any) -> dict[str, torch.Tensor] | None:
    if reference is None:
        return None
    if isinstance(reference, dict):
        return {
            str(name): value.detach().to(torch.float64).cpu() for name, value in reference.items()
        }
    raise TypeError(f"reference must be a tensor map, got {type(reference)!r}")


def digest_parameters(
    candidate: RoutingCandidate, reference: dict[str, torch.Tensor] | None = None
) -> dict[str, Any]:
    """Ordered f64 parameter digest plus max_abs_diff against a reference map."""
    tensors = parameter_tensors(candidate)
    result = {
        "sha256": _digest_map(tensors),
        "max_abs_diff": 0.0,
        "has_reference": reference is not None,
        "elements": int(sum(value.numel() for value in tensors.values())),
        "tensors": len(tensors),
    }
    other = _reference_map(reference)
    if other is not None:
        result["max_abs_diff"] = _max_abs_diff(tensors, other)
    return result


def digest_optimizer_moments(
    optimizer: torch.optim.Adam, reference: dict[str, torch.Tensor] | None = None
) -> dict[str, Any]:
    """Ordered f64 Adam moment digest plus max_abs_diff against a reference map."""
    tensors = optimizer_moment_tensors(optimizer)
    result = {
        "sha256": _digest_map(tensors),
        "max_abs_diff": 0.0,
        "has_reference": reference is not None,
        "elements": int(sum(value.numel() for value in tensors.values())),
        "tensors": len(tensors),
    }
    other = _reference_map(reference)
    if other is not None:
        result["max_abs_diff"] = _max_abs_diff(tensors, other)
    return result


def digest_probabilities(
    fixture_probs: torch.Tensor | list[torch.Tensor],
    reference: list[torch.Tensor] | torch.Tensor | None = None,
) -> dict[str, Any]:
    """Ordered f64 probability digest plus max_abs_diff against a reference list."""
    rows = [fixture_probs] if isinstance(fixture_probs, torch.Tensor) else list(fixture_probs)
    arrays: dict[str, torch.Tensor] = {
        f"{index:03d}": row.detach().to(torch.float64).cpu().clone()
        for index, row in enumerate(rows)
    }
    result = {
        "sha256": _digest_map(arrays),
        "max_abs_diff": 0.0,
        "has_reference": reference is not None,
        "rows": len(rows),
    }
    if reference is not None:
        other_rows = [reference] if isinstance(reference, torch.Tensor) else list(reference)
        other = {
            f"{index:03d}": row.detach().to(torch.float64).cpu().clone()
            for index, row in enumerate(other_rows)
        }
        result["max_abs_diff"] = _max_abs_diff(arrays, other)
    return result
