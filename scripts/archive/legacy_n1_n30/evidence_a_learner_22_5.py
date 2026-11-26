"""Typed episodic learner checkpoints for the four-arm evidence experiment."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from scripts.training.first_a_support_22_5 import model_description
from scripts.training.task_baselines_22_4_a import TaskSequenceBaseline
from taiji.u2_capacity_control import CapacitySubject
from taiji.u2_evidence_runtime import verify_legacy_sources
from taiji.u2_evidence_subject import EvidenceSubject
from taiji.u2_history import (
    assert_values_equal,
    atomic_load,
    atomic_save,
    detached_tree,
    value_digest,
)
from taiji.u2_recovery import _source_pins
from taiji.u2_subject import U2Subject

ROOT = Path(__file__).resolve().parents[2]
NATIVE_TYPES: dict[str, type[U2Subject]] = {
    "native-old": U2Subject,
    "native-query": EvidenceSubject,
    "native-capacity": CapacitySubject,
}
SCHEMA = "taiji-u2-evidence-A-learner-v1"


def source_pins(extra_sources=()):
    pins = {f"taiji/{name}": digest for name, digest in _source_pins().items()}
    names = (
        "scripts/training/evidence_a_learner_22_5.py",
        "scripts/training/first_a_support_22_5.py",
        "scripts/training/task_baselines_22_4_a.py",
    )
    for name in names + tuple(extra_sources):
        pins[name] = hashlib.sha256(
            (ROOT / name).read_text(encoding="utf-8-sig").encode()
        ).hexdigest()
    return pins


def fresh_optimizer(model):
    return torch.optim.Adam(
        model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0
    )


def save_learner(
    path,
    model,
    optimizer,
    architecture,
    lineage,
    *,
    update,
    cursor,
    order,
    data_digest,
    native_types=None,
    extra_sources=(),
):
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(destination)
    factories = NATIVE_TYPES if native_types is None else native_types
    if architecture in factories:
        if not isinstance(model, U2Subject) or type(model) is not factories[architecture]:
            raise ValueError("learner_native_type_mismatch")
        bundle = model.export_bundle(model.initial("episodic-learner-boundary"))
    elif (
        architecture == "transformer"
        and isinstance(model, TaskSequenceBaseline)
        and model.architecture == "transformer"
    ):
        bundle = {
            "description": model_description(model),
            "parameters": detached_tree(model.state_dict()),
            "revision": model.parameter_revision,
        }
    else:
        raise ValueError("unsupported_evidence_learner_architecture")
    payload = {
        "schema": SCHEMA,
        "architecture": architecture,
        "model": bundle,
        "optimizer": detached_tree(optimizer.state_dict()),
        "training": model.training,
        "lineage": detached_tree(lineage),
        "update": update,
        "cursor": cursor,
        "order": list(order),
        "data_digest": data_digest,
        "torch_rng": torch.get_rng_state(),
        "source_pins": source_pins(extra_sources),
        "state_scope": "episodic public-input replay; native bundle state is the empty boundary",
    }
    atomic_save(payload, destination)
    restored, other, loaded = load_learner(
        destination, native_types=native_types, extra_sources=extra_sources
    )
    assert_values_equal(model.state_dict(), restored.state_dict(), "learner.parameters")
    assert_values_equal(optimizer.state_dict(), other.state_dict(), "learner.optimizer")
    for name in ("training", "lineage", "update", "cursor", "order", "data_digest", "torch_rng"):
        assert_values_equal(payload[name], loaded[name], f"learner.{name}")
    return value_digest(payload)


def load_learner(path, *, native_types=None, extra_sources=()):
    payload = atomic_load(path)
    expected = {
        "schema",
        "architecture",
        "model",
        "optimizer",
        "training",
        "lineage",
        "update",
        "cursor",
        "order",
        "data_digest",
        "torch_rng",
        "source_pins",
        "state_scope",
    }
    if set(payload) != expected or payload["schema"] != SCHEMA:
        raise ValueError("unsupported_evidence_learner_schema")
    if payload["source_pins"] != source_pins(extra_sources):
        raise ValueError("evidence_learner_source_identity_mismatch")
    architecture, bundle = payload["architecture"], payload["model"]
    model: U2Subject | TaskSequenceBaseline
    factories = NATIVE_TYPES if native_types is None else native_types
    if architecture in factories:
        model, _ = factories[architecture].from_bundle(bundle)
    elif architecture == "transformer":
        if bundle["description"]["architecture"] != "transformer":
            raise ValueError("baseline_learner_type_mismatch")
        model = TaskSequenceBaseline(**bundle["description"])
        first = next(iter(bundle["parameters"].values()))
        model = model.to(dtype=first.dtype, device=first.device)
        model.load_state_dict(bundle["parameters"])
        model.parameter_revision = bundle["revision"]
    else:
        raise ValueError("unsupported_evidence_learner_architecture")
    model.train(payload["training"])
    optimizer = fresh_optimizer(model)
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["torch_rng"])
    return model, optimizer, payload


def load_legacy_transformer(path):
    payload = atomic_load(path)
    if (
        payload["schema"] != "taiji-first-A-learner-v1"
        or payload["description"]["architecture"] != "transformer"
    ):
        raise ValueError("unsupported_legacy_transformer_parent")
    checked = verify_legacy_sources(payload)
    model = TaskSequenceBaseline(**payload["description"])
    first = next(iter(payload["parameters"].values()))
    model = model.to(dtype=first.dtype, device=first.device)
    model.load_state_dict(payload["parameters"])
    model.parameter_revision = payload["revision"]
    return model, {
        "source_pins_verified": checked,
        "parameter_digest": value_digest(model.state_dict()),
        "update": payload["update"],
    }
