"""Unified execution entry: one bundle, one real task, three mechanisms.

Contract: plans/reference/M5_UNIFIED_ENTRY_EVIDENCE_PACKAGE_DRAFT_20260919.md

The M5 common gate's single remaining gap is a unified-entry evidence run:
ONE fixed bundle binding the three closed axes' components (P5.1h adopted
knowledge child, HANDOFF-M4 selection policy, P5.2d online writeback), ONE
frozen real task, a full per-tick trace (input -> memory retrieval ->
consequence prediction -> selection -> execution -> real outcome), and a
pre-registered ablation matrix against a simple deployable strategy.

This module owns the bundle identity (file-backed component refs verified by
SHA-256), the ablation arm configurations, and the trace schema.  The task
selection, thresholds and stop lines are frozen in the package's
pre-registration BEFORE the evidence run (contract section 4).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .collab_handoff import RULE_REVISION
from .contracts import _check_text
from .internalization import content_digest

BUNDLE_FORMAT = "taiji-unified-entry-bundle-v1"

#: The pre-registered ablation arms (contract section 3.4).  ``full`` is the
#: complete system; every other arm disables exactly one mechanism (or swaps
#: the policy for a simple deployable strategy) so each mechanism's
#: contribution is attributable.
ABLATION_ARMS = (
    "full",
    "simple_strategy",
    "disable_memory",
    "disable_selection",
    "disable_writeback",
)

#: Trace event kinds (schema, contract section 3.3).
TRACE_EVENT_KINDS = (
    "member_called",
    "member_chosen",
    "member_executed",
    "attempt_failed",
    "consequence_predicted",
    "writeback_applied",
    "writeback_skipped",
    "goal_reached",
    "stop",
)
TRACE_REQUIRED_KEYS = ("tick", "kind", "rule_revision", "bundle_digest")


@dataclass(frozen=True)
class ComponentRef:
    """A file-backed component of the unified bundle, verified by SHA-256."""

    name: str
    kind: str
    path: str
    sha256: str


@dataclass(frozen=True)
class UnifiedBundle:
    """The fixed identity every evidence run must cite."""

    task_id: str
    components: tuple[ComponentRef, ...]
    digest: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": BUNDLE_FORMAT,
            "task_id": self.task_id,
            "components": [
                {"name": c.name, "kind": c.kind, "path": c.path, "sha256": c.sha256}
                for c in self.components
            ],
            "digest": self.digest,
        }


def assemble_bundle(
    task_id: str,
    components: Sequence[tuple[str, str, Path]],
    *,
    expected_sha256: Mapping[str, str] | None = None,
) -> UnifiedBundle:
    """Assemble and verify the fixed bundle.

    Every component path must exist; its SHA-256 is computed and (when an
    expectation is given) must match, so a run cannot silently cite a moved
    or mutated checkpoint.  The bundle digest covers task id and all
    component hashes.
    """

    _check_text(task_id, "unified entry task_id")
    if not components:
        raise ValueError("unified entry bundle needs at least one component")
    refs: list[ComponentRef] = []
    seen: set[str] = set()
    for name, kind, path in components:
        if name in seen:
            raise ValueError(f"duplicate bundle component: {name}")
        seen.add(name)
        _check_text(name, "component name")
        _check_text(kind, "component kind")
        file_path = Path(path)
        if not file_path.is_file():
            raise ValueError(f"bundle component {name!r} file missing: {path}")
        sha = hashlib.sha256(file_path.read_bytes()).hexdigest()
        expected = (expected_sha256 or {}).get(name)
        if expected is not None and sha != expected:
            raise ValueError(
                f"bundle component {name!r} sha256 mismatch: {sha} != {expected}"
            )
        refs.append(ComponentRef(name=name, kind=kind, path=str(path), sha256=sha))
    digest = content_digest(
        {
            "task_id": task_id,
            "components": [
                {"name": r.name, "kind": r.kind, "sha256": r.sha256} for r in refs
            ],
        }
    )
    return UnifiedBundle(task_id=task_id, components=tuple(refs), digest=digest)


@dataclass(frozen=True)
class ArmConfig:
    """One ablation arm: exactly one mechanism swapped or disabled."""

    name: str
    rule_revision: int
    memory_enabled: bool
    writeback_enabled: bool
    simple_strategy: bool


def arm_config(name: str) -> ArmConfig:
    """The pre-registered ablation arms (contract section 3.4)."""

    if name == "full":
        return ArmConfig(name, rule_revision=RULE_REVISION, memory_enabled=True, writeback_enabled=True, simple_strategy=False)
    if name == "simple_strategy":
        return ArmConfig(name, rule_revision=0, memory_enabled=False, writeback_enabled=False, simple_strategy=True)
    if name == "disable_memory":
        return ArmConfig(name, rule_revision=RULE_REVISION, memory_enabled=False, writeback_enabled=True, simple_strategy=False)
    if name == "disable_selection":
        return ArmConfig(name, rule_revision=0, memory_enabled=True, writeback_enabled=True, simple_strategy=False)
    if name == "disable_writeback":
        return ArmConfig(name, rule_revision=RULE_REVISION, memory_enabled=True, writeback_enabled=False, simple_strategy=False)
    raise ValueError(f"unknown ablation arm: {name}")


def validate_trace(events: Sequence[Mapping[str, Any]]) -> bool:
    """Trace schema (contract section 3.3): every event carries the required
    keys, a known kind, and the bundle digest of its run."""

    for event in events:
        for key in TRACE_REQUIRED_KEYS:
            if key not in event:
                return False
        if event["kind"] not in TRACE_EVENT_KINDS:
            return False
    return True
