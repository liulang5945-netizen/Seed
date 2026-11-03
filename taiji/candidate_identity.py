"""Single identity for the 17 selection-routing candidate graph.

Every 17 report and manifest must carry the same ``candidate_sha`` and the same
declared-parameter list.  The digest covers the graph contract (schema, config,
arms, stage plan), the ordered parameter names/shapes/dtypes, and the source of
the graph modules.  Runner scripts are deliberately excluded: they may change
between stages without changing the graph.

The module must stay free of graph code so hashing it never appears in its own
payload.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

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
)

STAGE_PLAN = (
    ("selection", 1, 100),
    ("copy", 101, 200),
    ("joint", 201, 300),
)


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


def _config_payload() -> dict[str, Any]:
    config = RoutingConfig()
    return {
        "width": config.width,
        "prev_width": config.prev_width,
        "hidden": config.hidden,
        "gamma_floor": config.gamma_floor,
        "span_bottleneck": config.span_bottleneck,
    }


def candidate_identity(candidate: Any) -> dict[str, Any]:
    """Return the shared identity block; call it once per process and reuse it."""
    parameters = declared_parameters(candidate)
    trainable = trainable_declarations(candidate)
    sources = graph_sources()
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
    payload = {
        "graph": "taiji-selection-routing-17",
        "contract_version": ROUTING_SCHEMA,
        "bundle_schema": ROUTING_BUNDLE_SCHEMA,
        "config": _config_payload(),
        "arms": list(ARMS),
        "stage_plan": [list(row) for row in STAGE_PLAN],
        "parameters": [[name, list(shape), dtype] for name, shape, dtype in parameters],
        "declared_trainable": trainable,
        "source_files": sources,
    }
    candidate_sha = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return {
        "candidate_sha": candidate_sha,
        "contract_version": ROUTING_SCHEMA,
        "bundle_schema": ROUTING_BUNDLE_SCHEMA,
        "config": payload["config"],
        "arms": payload["arms"],
        "stage_plan": payload["stage_plan"],
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
    }
