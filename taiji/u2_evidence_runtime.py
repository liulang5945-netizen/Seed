"""Typed runtime and explicit legacy learner migration for evidence candidates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import ClassVar, cast

import torch

from .u2_evidence_subject import EvidenceSubject
from .u2_history import atomic_load, value_digest
from .u2_recovery import U2Runtime, _history_atomic
from .u2_subject import U2Config, U2Subject


class EvidenceRuntime(U2Runtime):
    model: EvidenceSubject
    subject_class: ClassVar[type[U2Subject]] = EvidenceSubject

    def __init__(self, model, *args, **kwargs):
        if not isinstance(model, EvidenceSubject):
            raise ValueError("evidence_runtime_requires_evidence_subject")
        super().__init__(model, *args, **kwargs)

    @_history_atomic
    def revoke_evidence(self, packet_id):
        self._check_window()
        self.state = cast(EvidenceSubject, self.model).revoke_evidence(self.state, packet_id)
        self._record("revoke_evidence", {"packet_id": packet_id})

    def _replay_operation(self, operation):
        if operation["kind"] == "revoke_evidence":
            if (
                set(operation) != {"kind", "arguments", "expected"}
                or operation["expected"] is not None
            ):
                raise ValueError("invalid_evidence_revocation_operation")
            self.revoke_evidence(**operation["arguments"])
        else:
            super()._replay_operation(operation)


def load_legacy_learner(path):
    """Verify every frozen source pin against current or explicitly archived bytes.

    The cognitive source must still match. Archived runtime files explain the
    source-only migration; their old runtime envelopes are not executed here.
    """
    root = Path(__file__).resolve().parents[1]
    catalog = json.loads(
        (root / "plans/manifests/taiji_u2_p3_source_migration_20261112.json").read_text(
            encoding="utf-8"
        )
    )
    archives = {row["source"]: row for row in catalog["archives"]}
    payload = atomic_load(path)
    if (
        payload["schema"] != "taiji-first-A-learner-v1"
        or payload["description"]["architecture"] != "native"
    ):
        raise ValueError("unsupported_evidence_migration_parent")
    checked = {}
    for name, pin in payload["source_pins"].items():
        relative = name if "/" in name else f"taiji/{name}"
        current = root / relative
        actual = hashlib.sha256(current.read_text(encoding="utf-8-sig").encode()).hexdigest()
        if actual != pin:
            row = archives.get(relative)
            if row is None or row["sha256"] != pin:
                raise ValueError("legacy_source_migration_not_registered")
            actual = hashlib.sha256(
                (root / row["archive"]).read_text(encoding="utf-8-sig").encode()
            ).hexdigest()
            if actual != pin:
                raise ValueError("legacy_archive_identity_mismatch")
        checked[relative] = pin
    if "taiji/u2_subject.py" not in checked:
        raise ValueError("legacy_cognitive_source_pin_missing")
    model = U2Subject(U2Config(**payload["description"]["config"]))
    first = next(iter(payload["parameters"].values()))
    model = model.to(dtype=first.dtype, device=first.device)
    model.load_state_dict(payload["parameters"])
    model.parameter_revision = payload["revision"]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.load_state_dict(payload["optimizer"])
    return (
        model,
        optimizer,
        {
            "source_pins_verified": checked,
            "parameter_digest": value_digest(model.state_dict()),
            "update": payload["update"],
        },
    )
