"""Cross-session persistent store for N33.2 (zero-training harness).

The N33 stage needs a same-identity, cross-session experience record: the subject's capsule
(tensor state), pending items, and an append-only experience log must survive between
sessions and be recoverable bit-exactly.

This module is deliberately small and dependency-free beyond torch:

* ``SessionRecord``: subject_id, capsule tensors, pending items, append-only experience log,
  logical clock, schema version.
* ``SessionStore``: file-per-subject store under a directory; each file carries a canonical
  sha256 digest over metadata + tensor bytes, verified on load; corrupted files raise.
* Cross-process round-trip is covered by tests; no training happens here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

SCHEMA = "taiji-session-store-v1"


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _tensor_digest(tensors: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name in sorted(tensors):
        t = tensors[name].detach().to(torch.float32).contiguous()
        h.update(name.encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


@dataclass
class SessionRecord:
    subject_id: str
    capsule: dict[str, torch.Tensor] = field(default_factory=dict)
    pending: list[dict[str, Any]] = field(default_factory=list)
    experience: list[dict[str, Any]] = field(default_factory=list)
    clock: int = 0
    schema: str = SCHEMA

    # -- pending items -------------------------------------------------------
    def add_pending(self, item: dict[str, Any]) -> None:
        if not isinstance(item.get("id"), str) or not item["id"]:
            raise ValueError("pending item needs a non-empty string id")
        if any(p["id"] == item["id"] for p in self.pending):
            raise ValueError(f"duplicate pending id {item['id']!r}")
        self.pending.append(dict(item))
        self.clock += 1

    def resolve_pending(self, item_id: str, outcome: str) -> None:
        for i, p in enumerate(self.pending):
            if p["id"] == item_id:
                done = dict(p)
                done["outcome"] = outcome
                self.experience.append(
                    {"kind": "resolved_pending", "payload": done, "clock": self.clock}
                )
                self.pending.pop(i)
                self.clock += 1
                return
        raise KeyError(f"unknown pending id {item_id!r}")

    # -- experience log ------------------------------------------------------
    def append_experience(
        self, kind: str, payload: dict[str, Any], source: str = "session"
    ) -> None:
        self.experience.append(
            {
                "kind": str(kind),
                "payload": dict(payload),
                "source": str(source),
                "clock": self.clock,
            }
        )
        self.clock += 1

    # -- metadata / digest ---------------------------------------------------
    def metadata(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "subject_id": self.subject_id,
            "clock": self.clock,
            "pending": self.pending,
            "experience": self.experience,
        }

    def digest(self) -> str:
        meta = _canonical(self.metadata())
        return hashlib.sha256(
            meta.encode("utf-8") + _tensor_digest(self.capsule).encode("utf-8")
        ).hexdigest()


class SessionStore:
    """File-per-subject cross-session store with integrity checks."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, subject_id: str) -> Path:
        if not subject_id or any(c in subject_id for c in ("/", "\\", "..")):
            raise ValueError(f"invalid subject id {subject_id!r}")
        return self.root / f"{subject_id}.session"

    def save(self, record: SessionRecord) -> str:
        payload = {
            "metadata": record.metadata(),
            "digest": record.digest(),
            "tensors": {
                k: v.detach().to(torch.float32).contiguous() for k, v in record.capsule.items()
            },
        }
        path = self._path(record.subject_id)
        torch.save(payload, path)
        return str(path)

    def load(self, subject_id: str) -> SessionRecord:
        path = self._path(subject_id)
        if not path.exists():
            raise FileNotFoundError(f"no session for {subject_id!r}")
        payload = torch.load(path, weights_only=False)
        meta = payload["metadata"]
        if meta.get("schema") != SCHEMA:
            raise ValueError(f"unsupported schema {meta.get('schema')!r}")
        if meta.get("subject_id") != subject_id:
            raise ValueError("subject id mismatch in stored session")
        record = SessionRecord(
            subject_id=subject_id,
            capsule=dict(payload["tensors"]),
            pending=list(meta["pending"]),
            experience=list(meta["experience"]),
            clock=int(meta["clock"]),
        )
        if record.digest() != payload["digest"]:
            raise ValueError("session digest mismatch (corrupted store)")
        return record

    def subjects(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.session"))
