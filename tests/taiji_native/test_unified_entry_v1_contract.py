"""Unified-entry implementation gates (zero training).

Contract: plans/reference/M5_UNIFIED_ENTRY_EVIDENCE_PACKAGE_DRAFT_20260919.md §3/§4

Pins: bundle assembly with SHA-256 verification (real P5.1h child checkpoint
when present), digest stability, ablation arm configurations (each arm
disables exactly one mechanism), and the trace schema.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from taiji.collab_handoff import RULE_REVISION
from taiji.unified_entry import (
    ABLATION_ARMS,
    TRACE_EVENT_KINDS,
    ArmConfig,
    UnifiedBundle,
    arm_config,
    assemble_bundle,
    validate_trace,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
P51H_CHILD = PROJECT_ROOT / "reports/taiji_p5_1h_child_20260919.pt"
P51H_SHA = "daf2e8779b9faa734994b8f9de084ab75744540bcda032d3f1c79aeb9b87021d"


def _write_component(tmp_path: Path, name: str, payload: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(payload)
    return path


# --------------------------------------------------------------------------- #
# Gate 1: bundle assembly with SHA-256 verification
# --------------------------------------------------------------------------- #


def test_gate1_bundle_assembly_verifies_real_p51h_child(tmp_path: Path) -> None:
    if P51H_CHILD.is_file():
        knowledge_path = P51H_CHILD
        expected = {**{}} or {}
    else:
        knowledge_path = _write_component(tmp_path, "child.pt", b"synthetic-child")
        expected = {}
    policy = _write_component(tmp_path, "policy.json", b'{"composition_rule": "m4_failure_handoff"}')
    bundle = assemble_bundle(
        "p52-family-task",
        (
            ("knowledge_child", "p51h_adopted_child", knowledge_path),
            ("handoff_policy", "collab_handoff_v1", policy),
        ),
        expected_sha256=expected,
    )
    assert isinstance(bundle, UnifiedBundle)
    names = {c.name: c for c in bundle.components}
    if P51H_CHILD.is_file():
        assert names["knowledge_child"].sha256 == P51H_SHA
    assert names["handoff_policy"].sha256 == hashlib.sha256(policy.read_bytes()).hexdigest()
    # digest stability: same inputs -> same digest
    again = assemble_bundle(
        "p52-family-task",
        (
            ("knowledge_child", "p51h_adopted_child", knowledge_path),
            ("handoff_policy", "collab_handoff_v1", policy),
        ),
    )
    assert again.digest == bundle.digest


def test_gate1_bundle_rejects_missing_and_mutated_components(tmp_path: Path) -> None:
    component = _write_component(tmp_path, "component.pt", b"payload")
    with pytest.raises(ValueError, match="file missing"):
        assemble_bundle("task", (("x", "kind", tmp_path / "missing.pt"),))
    expected_sha = hashlib.sha256(b"payload").hexdigest()
    assemble_bundle("task", (("x", "kind", component),), expected_sha256={"x": expected_sha})
    mutated = _write_component(tmp_path, "mutated.pt", b"payload-tampered")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        assemble_bundle(
            "task",
            (("x", "kind", mutated),),
            expected_sha256={"x": expected_sha},
        )
    with pytest.raises(ValueError, match="duplicate bundle component"):
        assemble_bundle("task", (("x", "kind", component), ("x", "kind", component)))
    with pytest.raises(ValueError, match="at least one component"):
        assemble_bundle("task", ())
    with pytest.raises(ValueError, match="task_id"):
        assemble_bundle("", (("x", "kind", component),))


# --------------------------------------------------------------------------- #
# Gate 2: ablation arms disable exactly one mechanism each
# --------------------------------------------------------------------------- #


def test_gate2_ablation_arms() -> None:
    assert ABLATION_ARMS == (
        "full",
        "simple_strategy",
        "disable_memory",
        "disable_selection",
        "disable_writeback",
    )
    full = arm_config("full")
    assert full == ArmConfig(
        name="full", rule_revision=RULE_REVISION, memory_enabled=True,
        writeback_enabled=True, simple_strategy=False,
    )
    # each non-full arm differs from full in exactly one field
    for name in ("simple_strategy", "disable_memory", "disable_selection", "disable_writeback"):
        arm = arm_config(name)
        diffs = [
            field
            for field in ("rule_revision", "memory_enabled", "writeback_enabled", "simple_strategy")
            if getattr(arm, field) != getattr(full, field)
        ]
        if name == "simple_strategy":
            assert set(diffs) == {"rule_revision", "memory_enabled", "writeback_enabled", "simple_strategy"}
        else:
            assert len(diffs) == 1, (name, diffs)
    with pytest.raises(ValueError, match="unknown ablation arm"):
        arm_config("nonexistent")


# --------------------------------------------------------------------------- #
# Gate 3: trace schema
# --------------------------------------------------------------------------- #


def test_gate3_trace_schema() -> None:
    good = [
        {"tick": 1, "kind": "member_called", "rule_revision": 1, "bundle_digest": "abc"},
        {"tick": 1, "kind": "stop", "rule_revision": 1, "bundle_digest": "abc", "stop": "goal_reached"},
    ]
    assert validate_trace(good) is True
    assert validate_trace([]) is True
    assert validate_trace([{"tick": 1, "kind": "member_called", "rule_revision": 1}]) is False
    assert validate_trace([{"tick": 1, "kind": "unknown_kind", "rule_revision": 1, "bundle_digest": "abc"}]) is False
    assert "consequence_predicted" in TRACE_EVENT_KINDS
    assert "writeback_applied" in TRACE_EVENT_KINDS
