"""Single identity for the 17 selection-routing candidate graph.

Every 17 report and manifest must carry the same ``candidate_sha`` and the same
declared-parameter list.  The digest covers behaviour-relevant elements only:
the graph contract (schema, config, arms, stage plan), the ordered parameter
names/shapes/dtypes, the declared trainable ownership, the routing-step input
whitelist and the routing/decode rule version.

Source-file bytes and parameter values are recorded next to the digest but do
**not** enter it: formatting or comments must not change candidate identity.
Weights are runtime state (the parent bundle and the head initialisation differ
per cell), so they are tracked as ``weights_sha256`` / ``candidate_state_sha256``
instead of being folded into the graph identity.

The module must stay free of graph code so hashing it never appears in its own
payload.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from taiji.selection_routing import ARMS, ROUTING_SCHEMA, RoutingConfig
from taiji.selection_routing_checkpoint import ROUTING_BUNDLE_SCHEMA, parameter_tensors

ROOT = Path(__file__).resolve().parents[1]

GRAPH_MODULES = (
    "taiji/selection_routing.py",
    "taiji/binding_copy.py",
    "taiji/binding_select.py",
    "taiji/record_relation_core.py",
    "taiji/living_core.py",
    "taiji/selection_routing_checkpoint.py",
    "taiji/byte_match_route.py",
    "taiji/copy_stop_boundary.py",
)

STAGE_PLAN = (
    ("selection", 1, 100),
    ("copy", 101, 200),
    ("joint", 201, 300),
)

IDENTITY_BASIS = "semantic"
ROUTING_RULE_VERSION = "17-r3"

#: Keyword arguments ``routing_step`` accepts.  Any change here is behavioural:
#: adding, removing or renaming an input changes what the graph can read.
INPUT_WHITELIST = (
    "arm",
    "forced_pointer_bias",
    "gate_index",
    "head",
    "pointer_hard",
    "pointer_temperature",
    "previous",
    "span_head",
    "state",
)

DECODE_RULES = {
    "copy_transport": "point_transport",
    "pointer_temperature": {
        "schedule": "T(u)=1-0.95*(u-1)/99 for u<=100, then 0.05",
        "floor": 0.05,
    },
    "pointer_inference": "argmax",
    "pointer_entropy_coef": 0.01,
    "span_supervision_weight": 0.5,
    "gamma_floor": 0.2,
    "sequence_stop": "decode terminates one byte past the final hop's field end",
    "credit": {
        "A_route_lock": "straight-through hard route",
        "B_stop_lock": "detached hard route",
        "C_unlocked": "inherited R1 unlocked read",
    },
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def graph_sources() -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in GRAPH_MODULES}


def declared_parameters(candidate: Any) -> list[tuple[str, tuple[int, ...], str]]:
    """Ordered (name, shape, dtype) of every registered candidate tensor."""
    rows = []
    for name, tensor in parameter_tensors(candidate).items():
        rows.append((name, tuple(int(size) for size in tensor.shape), str(tensor.dtype)))
    return rows


def trainable_declarations(candidate: Any) -> dict[str, list[str]]:
    """Stage-independent trainable ownership, not the current requires_grad."""
    binding = [
        name for name, parameter in candidate.binding.named_parameters() if parameter.requires_grad
    ]
    head = [f"head.{name}" for name, _parameter in candidate.head.named_parameters()]
    span = (
        [f"span.{name}" for name, _parameter in candidate.span_head.named_parameters()]
        if candidate.span_head is not None
        else []
    )
    return {"binding": binding, "head": head, "span": span}


def _config_payload(alignment: bool = False) -> dict[str, Any]:
    config = RoutingConfig()
    payload = {
        "width": config.width,
        "prev_width": config.prev_width,
        "hidden": config.hidden,
        "gamma_floor": config.gamma_floor,
        "span_bottleneck": config.span_bottleneck,
    }
    # The 19.2h alignment block is declared only when it is present, so every
    # pre-19.2h candidate keeps its exact ``candidate_sha_v2``.
    if alignment:
        payload["alignment"] = True
    return payload


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    with torch.no_grad():
        return tensor.detach().cpu().contiguous().numpy().tobytes()


def weights_sha256(candidate: Any) -> dict[str, str]:
    """Per-tensor value digests; runtime state, not part of the graph identity."""
    digest = {}
    for name, tensor in parameter_tensors(candidate).items():
        payload = (
            name.encode("utf-8")
            + str(tuple(tensor.shape)).encode("utf-8")
            + str(tensor.dtype).encode("utf-8")
            + _tensor_bytes(tensor)
        )
        digest[name] = hashlib.sha256(payload).hexdigest()
    return digest


def _state_digest(digests: dict[str, str]) -> str:
    payload = json.dumps(digests, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _semantic_payload(
    parameters: list[tuple[str, tuple[int, ...], str]],
    trainable: dict[str, list[str]],
    alignment: bool = False,
) -> dict[str, Any]:
    return {
        "graph": "taiji-selection-routing-17",
        "identity_basis": IDENTITY_BASIS,
        "contract_version": ROUTING_SCHEMA,
        "bundle_schema": ROUTING_BUNDLE_SCHEMA,
        "config": _config_payload(alignment),
        "arms": list(ARMS),
        "stage_plan": [list(row) for row in STAGE_PLAN],
        "modules": list(GRAPH_MODULES),
        "routing_rule_version": ROUTING_RULE_VERSION,
        "decode_rules": DECODE_RULES,
        "input_whitelist": list(INPUT_WHITELIST),
        "parameters": [[name, list(shape), dtype] for name, shape, dtype in parameters],
        "declared_trainable": trainable,
    }


def _digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _legacy_payload(
    parameters: list[tuple[str, tuple[int, ...], str]],
    trainable: dict[str, list[str]],
    sources: dict[str, str],
    alignment: bool = False,
) -> dict[str, Any]:
    """Pre-2026-11-04 identity: source bytes were part of the digest."""
    payload = _semantic_payload(parameters, trainable, alignment)
    payload.pop("identity_basis")
    payload.pop("modules")
    payload.pop("routing_rule_version")
    payload.pop("decode_rules")
    payload.pop("input_whitelist")
    payload["source_files"] = sources
    return payload


def candidate_identity(candidate: Any) -> dict[str, Any]:
    """Return the shared identity block; call it once per process and reuse it."""
    parameters = declared_parameters(candidate)
    trainable = trainable_declarations(candidate)
    sources = graph_sources()
    weights = weights_sha256(candidate)
    binding_elements = sum(
        parameter.numel() for _name, parameter in candidate.binding.named_parameters()
    )
    head_elements = sum(parameter.numel() for parameter in candidate.head.parameters())
    span_elements = (
        sum(parameter.numel() for parameter in candidate.span_head.parameters())
        if candidate.span_head is not None
        else 0
    )
    parameter_names = [row[0] for row in parameters]
    names_digest = hashlib.sha256(
        json.dumps(parameter_names, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    alignment = bool(getattr(candidate.head.config, "alignment", False))
    semantic = _semantic_payload(parameters, trainable, alignment)
    candidate_sha = _digest(semantic)
    return {
        "candidate_sha": candidate_sha,
        "candidate_sha_v2": candidate_sha,
        "identity_basis": IDENTITY_BASIS,
        "candidate_sha_v1_byte_bound": _digest(
            _legacy_payload(parameters, trainable, sources, alignment)
        ),
        "contract_version": ROUTING_SCHEMA,
        "bundle_schema": ROUTING_BUNDLE_SCHEMA,
        "routing_rule_version": ROUTING_RULE_VERSION,
        "config": semantic["config"],
        "arms": semantic["arms"],
        "stage_plan": semantic["stage_plan"],
        "modules": semantic["modules"],
        "decode_rules": semantic["decode_rules"],
        "input_whitelist": semantic["input_whitelist"],
        "parameter_counts": {
            "binding": binding_elements,
            "head": head_elements,
            "span": span_elements,
            "container_head_span": binding_elements + head_elements + span_elements,
            "declared_trainable_elements": sum(
                tensor.numel()
                for name, tensor in parameter_tensors(candidate).items()
                if name in set(trainable["binding"] + trainable["head"] + trainable["span"])
            ),
        },
        "parameter_names_sha256": names_digest,
        "parameter_count": len(parameters),
        "declared_trainable": trainable,
        "source_files": sources,
        "source_sha": sources,
        "weights_sha256": weights,
        "candidate_state_sha256": _state_digest(weights),
    }
