"""Durable, append-only attempt accounting for the bounded U2 engineering card."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class LedgerError(RuntimeError):
    exit_code = 3


class AuthorizationError(LedgerError):
    exit_code = 5


class AttemptAlreadyConsumed(LedgerError):
    pass


class AttemptOrderError(LedgerError):
    exit_code = 6


class TailTruncated(LedgerError):
    pass


class AttemptLedger:
    def __init__(
        self,
        path: Path,
        run: dict[str, Any],
        source_pins: dict[str, str],
        precontract: dict[str, Any],
    ):
        if (
            run.get("authorized") is not True
            or not run.get("owner_approval_ref")
            or not run.get("manifest_sha256")
        ):
            raise AuthorizationError("A recorded owner authorization is required")
        self.path, self.run, self.source_pins = path, run, source_pins
        self.order = precontract["run"]["attempt_order"]
        self.parents = precontract["run"]["attempt_parents"]
        self.posts = precontract["run"]["attempt_posts"]
        self.maximum = precontract["run"]["budget_max"]

    def read(self) -> dict[str, Any]:
        if not self.path.exists() or not self.path.stat().st_size:
            raise AuthorizationError("Ledger authorization header is missing")
        raw = self.path.read_bytes()
        if not raw.endswith(b"\n"):
            raise TailTruncated("tail_truncated=true")
        rows, opened, closed, reconciled = [], {}, {}, set()
        previous = "0" * 64
        for index, line in enumerate(raw.splitlines(keepends=True)):
            if len(line) > 4096 or b"\r" in line or line.startswith(b"\xef\xbb\xbf"):
                raise LedgerError("Ledger line encoding/size is invalid")
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError) as error:
                raise LedgerError("Ledger JSON is invalid") from error
            if not isinstance(row, dict):
                raise LedgerError("Ledger row must be an object")
            event_body = {key: value for key, value in row.items() if key != "record_sha256"}
            event_bytes = json.dumps(
                event_body, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            if row.get("record_sha256") != hashlib.sha256(event_bytes).hexdigest():
                raise LedgerError("Ledger event digest is invalid")
            if row.get("seq") != index or row.get("prev_line_sha256") != previous:
                raise LedgerError("Ledger hash chain is invalid")
            if (
                row.get("run_id") != self.run["run_id"]
                or row.get("source_pins") != self.source_pins
            ):
                raise LedgerError("Ledger run/source identity mismatch")
            event, identity = row.get("event"), row.get("attempt_id")
            before, after = row.get("budget_before"), row.get("budget_after")
            if index == 0:
                if (
                    event != "run_start"
                    or row.get("owner_approval_ref") != self.run["owner_approval_ref"]
                    or row.get("manifest_sha256") != self.run["manifest_sha256"]
                ):
                    raise AuthorizationError("Ledger authorization header is missing")
                if before != 0 or after != 0:
                    raise LedgerError("run_start cannot consume budget")
            elif event == "open":
                if identity in opened:
                    raise AttemptAlreadyConsumed(str(identity))
                count = len(opened)
                if count >= self.maximum:
                    raise LedgerError("Attempt budget exhausted")
                if identity != self.order[count]:
                    raise AttemptOrderError("Attempt order violated")
                if (
                    before != count
                    or after != count + 1
                    or row.get("parent_checkpoint") != self.parents[identity]
                ):
                    raise AttemptOrderError("Attempt parent/budget mapping violated")
                opened[identity] = row
            elif event == "close":
                if identity not in opened or identity in closed or identity in reconciled:
                    raise LedgerError("Invalid duplicate or unmatched close")
                if before != opened[identity]["budget_after"] or after != before:
                    raise LedgerError("close changed the consumed budget")
                if row.get("outcome") not in {
                    "accepted",
                    "rejected",
                    "exception",
                    "readback_failed",
                }:
                    raise LedgerError("Invalid close outcome")
                if (
                    row.get("outcome") == "accepted"
                    and row.get("post_checkpoint") != self.posts[identity]
                ):
                    raise AttemptOrderError("Attempt post mapping violated")
                closed[identity] = row
            elif event == "reconcile":
                if identity not in opened or identity in closed or identity in reconciled:
                    raise LedgerError("Invalid reconciliation")
                if before != len(opened) or after != before:
                    raise LedgerError("reconcile changed budget")
                if row.get("outcome") not in {
                    "indeterminate_crash",
                    "crash_reconciled_after_readback",
                }:
                    raise LedgerError("Invalid reconciliation outcome")
                reconciled.add(identity)
            elif event == "annotation":
                if identity not in opened or identity not in closed:
                    raise LedgerError("Only a closed attempt can be annotated")
                if before != len(opened) or after != before:
                    raise LedgerError("annotation changed budget")
            else:
                raise LedgerError("Unexpected ledger event")
            rows.append(row)
            previous = hashlib.sha256(line).hexdigest()
        if not rows:
            raise AuthorizationError("Ledger authorization header is missing")
        return {
            "rows": rows,
            "opened": opened,
            "closed": closed,
            "reconciled": reconciled,
            "head": previous,
            "consumed": len(opened),
        }

    def _append(
        self,
        event: str,
        identity: str | None,
        before: int,
        after: int,
        fields: dict[str, Any],
        *,
        new=False,
    ) -> dict[str, Any]:
        state = None if new else self.read()
        common = {
            "seq": 0 if new else len(state["rows"]),
            "run_id": self.run["run_id"],
            "attempt_id": identity,
            "event": event,
            "utc": datetime.now(UTC).isoformat(),
            "pid": os.getpid(),
            "host": hashlib.sha256(platform.node().encode()).hexdigest(),
            "budget_before": before,
            "budget_after": after,
            "prev_line_sha256": "0" * 64 if new else state["head"],
            "source_pins": self.source_pins,
        }
        if set(common) & set(fields):
            raise LedgerError("Event fields override ledger identity")
        row = {**common, **fields}
        event_bytes = json.dumps(
            row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        row["record_sha256"] = hashlib.sha256(event_bytes).hexdigest()
        data = (
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        if len(data) > 4096:
            raise LedgerError("Ledger line exceeds 4096 bytes")
        flags = (
            os.O_WRONLY
            | getattr(os, "O_BINARY", 0)
            | (os.O_CREAT | os.O_EXCL if new else os.O_APPEND)
        )
        descriptor = os.open(self.path, flags)
        try:
            if os.write(descriptor, data) != len(data):
                raise LedgerError("Partial ledger write")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return row

    def start(self) -> dict[str, Any]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return self._append(
            "run_start",
            None,
            0,
            0,
            {
                "owner_approval_ref": self.run["owner_approval_ref"],
                "manifest_sha256": self.run["manifest_sha256"],
                "budget_max": self.maximum,
            },
            new=True,
        )

    def open(self, identity: str, **fields: Any) -> dict[str, Any]:
        state = self.read()
        if identity in state["opened"]:
            raise AttemptAlreadyConsumed(identity)
        count = state["consumed"]
        if count >= self.maximum:
            raise LedgerError("Attempt budget exhausted")
        if identity != self.order[count]:
            raise AttemptOrderError("Attempt order violated")
        if any(
            key not in state["closed"] and key not in state["reconciled"] for key in state["opened"]
        ):
            raise LedgerError("Unreconciled open attempt exists")
        if fields.get("parent_checkpoint") != self.parents[identity]:
            raise AttemptOrderError("Attempt parent mapping violated")
        return self._append("open", identity, count, count + 1, fields)

    def close(self, identity: str, **fields: Any) -> dict[str, Any]:
        state = self.read()
        if (
            identity not in state["opened"]
            or identity in state["closed"]
            or identity in state["reconciled"]
        ):
            raise LedgerError("Attempt cannot be closed")
        if fields.get("outcome") not in {"accepted", "rejected", "exception", "readback_failed"}:
            raise LedgerError("Invalid close outcome")
        if (
            fields.get("outcome") == "accepted"
            and fields.get("post_checkpoint") != self.posts[identity]
        ):
            raise AttemptOrderError("Attempt post mapping violated")
        budget = state["opened"][identity]["budget_after"]
        return self._append("close", identity, budget, budget, fields)

    def reconcile(self) -> list[dict[str, Any]]:
        state = self.read()
        rows = []
        for identity in state["opened"]:
            if identity not in state["closed"] and identity not in state["reconciled"]:
                rows.append(
                    self._append(
                        "reconcile",
                        identity,
                        state["consumed"],
                        state["consumed"],
                        {"outcome": "indeterminate_crash", "note": "open without durable close"},
                    )
                )
        return rows

    def annotate(self, identity: str, **fields: Any) -> dict[str, Any]:
        """Append a post-close, non-financial note row.

        The state machine has no other slot for "after the settlement was written we
        discovered something about it". An annotation may only follow an existing close,
        never change the consumed budget, and is therefore not a settlement event.
        """
        state = self.read()
        if identity not in state["closed"]:
            raise LedgerError("Only a closed attempt can be annotated")
        budget = state["consumed"]
        return self._append("annotation", identity, budget, budget, fields)

    def repair_tail(self, destination: Path) -> dict[str, Any]:
        raw = self.path.read_bytes()
        if raw.endswith(b"\n"):
            raise LedgerError("Ledger has no truncated tail")
        split = raw.rfind(b"\n") + 1
        removed = raw[split:]
        proof = {
            "schema": "taiji-u2-ledger-tail-repair-v1",
            "removed_bytes": len(removed),
            "removed_sha256": hashlib.sha256(removed).hexdigest(),
        }
        # Preserve the repair proof before truncating only the incomplete tail.
        with destination.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(proof, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        with self.path.open("r+b") as stream:
            stream.truncate(split)
            stream.flush()
            os.fsync(stream.fileno())
        self.read()
        return proof
