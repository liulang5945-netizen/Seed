"""Recovery v2: engineering admission is separate from one formal face access.

The frozen evaluator is imported as a library.  This module owns the v2 contract,
the cross-attempt window ledger, per-phase namespaces, the single durable
``face_access_started`` marker and the one prompt seam the frozen ``execute()``
accepts (``prompt_selector``).  No frozen global is patched.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.training import check_taiji_r2_evaluation_recovery as audit
from scripts.training import evaluate_taiji_r2_daily_dialogue_recovery_v2 as frozen
from scripts.training import run_taiji_r2_daily_dialogue_full_turn_pretraining as base
from scripts.training import taiji_r2_continuation_evaluation_adapter as adapter

ROOT = Path(__file__).resolve().parents[2]
RECOVERY_ID = "taiji-r2-evaluation-recovery-v2-20261025"
REPORT_PREFIX = "taiji_r2_daily_dialogue_recovery_v2_20261021_extension_v1_paired_eval_recovery_v2"
CONTRACT_GLOB = "taiji_r2_evaluation_recovery_v2_contract_r*.json"
CONTRACT_PATTERN = re.compile(r"taiji_r2_evaluation_recovery_v2_contract_r(\d+)_\d{8}\.json")
WINDOW_LEDGER = ROOT / "reports" / f"{REPORT_PREFIX}_window_ledger.jsonl"
MARKER = ROOT / "reports" / f"{REPORT_PREFIX}_face_access.json"

WINDOW_CPU_LIMIT_SECONDS = 14_400.0
WINDOW_WALL_LIMIT_SECONDS = 14_400.0
HISTORICAL_CPU_RESERVE_SECONDS = 3_400.0
HISTORICAL_WALL_RESERVE_SECONDS = 180.0
V1_MEASURED_CPU_SECONDS = 84.484375
V1_MEASURED_WALL_SECONDS = 94.236
ADMISSION_CPU_LIMIT_SECONDS = 500.0
ADMISSION_WALL_LIMIT_SECONDS = 500.0
ADMISSION_RESTORE_RESERVE_SECONDS = 60.0
FORMAL_CPU_LIMIT_SECONDS = 10_330.0
FORMAL_WALL_LIMIT_SECONDS = 12_500.0
GENERATION_OUTPUTS = 156
REVIEW_ANSWERS = 48

MARKER_SCHEMA = "taiji-r2-evaluation-recovery-v2-face-access-marker-v1"
ATTEMPT_SCHEMA = "taiji-r2-dialogue-recovery-v2-paired-evaluation-attempt-v1"
PHASES = ("admission", "formal")

CC1_PREFIX = ROOT / "reports" / audit.PREFIX
V1_PREFIX = (
    ROOT
    / "reports"
    / "taiji_r2_daily_dialogue_recovery_v2_20261021_extension_v1_paired_eval_recovery_v1"
)

REQUIRED_SOURCES = (
    "scripts/training/taiji_r2_evaluation_recovery_v2.py",
    "scripts/training/evaluate_taiji_r2_evaluation_recovery_v2.py",
    "scripts/training/evaluate_taiji_r2_daily_dialogue_continuation_extension_worker_v1.py",
    "scripts/training/taiji_r2_continuation_evaluation_adapter.py",
    "scripts/training/evaluate_taiji_r2_daily_dialogue_continuation_extension_v1.py",
    "scripts/training/check_taiji_r2_evaluation_recovery.py",
    "tests/taiji_native/test_r2_evaluation_recovery_v2_contract.py",
    "plans/reference/delivery-runs/20261021-n-mainline/artifacts/"
    "04-evaluation-consumption-boundary-v2.md",
    "plans/reference/delivery-runs/20261021-n-mainline/artifacts/"
    "05-evaluation-recovery-v2-plan.md",
)

REQUIRED_ADOPTION_FLAGS = (
    "adopts_consumption_boundary_v2",
    "authorizes_engineering_admission",
    "authorizes_one_formal_face_consumption",
    "preserves_v1_failure",
    "preserves_cc1_failure",
    "same_dev_face_reuse_approved",
    "historical_tail_reserve_accepted",
)


def _sha(path: Path) -> str:
    return audit.sha256(Path(path))


def canonical_sha(payload: Mapping[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def _number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ValueError(f"{label} must be a finite non-negative number")
    return float(value)


def _sha_field(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase sha256")
    return value


def default_old_evidence() -> dict[str, Any]:
    return {
        "cc1": audit.audit_failed_attempt(CC1_PREFIX),
        "v1": audit.audit_failed_attempt(V1_PREFIX),
    }


def require_authorization(contract: Mapping[str, Any]) -> None:
    """One adopted v2 contract, with every fixed boundary checked arithmetically."""
    if contract.get("recovery_id") != RECOVERY_ID:
        raise ValueError("unknown recovery contract identity")
    if contract.get("status") != "adopted":
        raise ValueError("recovery v2 contract is not adopted")
    for name in REQUIRED_ADOPTION_FLAGS:
        if contract.get(name) is not True:
            raise ValueError(f"recovery requires explicit protocol decision: {name}")
    if contract.get("cpu_limit_seconds") != ADMISSION_CPU_LIMIT_SECONDS:
        raise ValueError("admission CPU limit differs from the versioned contract")
    if contract.get("wall_limit_seconds") != ADMISSION_WALL_LIMIT_SECONDS:
        raise ValueError("admission wall limit differs from the versioned contract")
    if contract.get("formal_cpu_limit_seconds") != FORMAL_CPU_LIMIT_SECONDS:
        raise ValueError("formal CPU limit differs from the versioned contract")
    if contract.get("formal_wall_limit_seconds") != FORMAL_WALL_LIMIT_SECONDS:
        raise ValueError("formal wall limit differs from the versioned contract")
    if contract.get("historical_cpu_reserve_seconds") != HISTORICAL_CPU_RESERVE_SECONDS:
        raise ValueError("historical CPU reserve differs from the inherited window")
    if contract.get("historical_wall_reserve_seconds") != HISTORICAL_WALL_RESERVE_SECONDS:
        raise ValueError("historical wall reserve differs from the inherited window")
    if contract.get("generation_outputs") != GENERATION_OUTPUTS:
        raise ValueError("generation denominator differs")
    if contract.get("review_answers") != REVIEW_ANSWERS:
        raise ValueError("review denominator differs")
    cpu_upper = (
        HISTORICAL_CPU_RESERVE_SECONDS
        + V1_MEASURED_CPU_SECONDS
        + ADMISSION_CPU_LIMIT_SECONDS
        + FORMAL_CPU_LIMIT_SECONDS
    )
    wall_upper = (
        HISTORICAL_WALL_RESERVE_SECONDS
        + V1_MEASURED_WALL_SECONDS
        + ADMISSION_WALL_LIMIT_SECONDS
        + FORMAL_WALL_LIMIT_SECONDS
    )
    if cpu_upper > WINDOW_CPU_LIMIT_SECONDS or wall_upper > WINDOW_WALL_LIMIT_SECONDS:
        raise ValueError("v2 allocations exceed the inherited evaluation window")
    if (
        frozen._whole_run_reserve()[0] + 60.0 > FORMAL_CPU_LIMIT_SECONDS
        or frozen._whole_run_reserve()[1] > FORMAL_WALL_LIMIT_SECONDS
    ):
        raise ValueError("formal limit cannot cover the frozen whole-run reserve")
    endpoints = contract.get("endpoint_checkpoint_sha256")
    if not isinstance(endpoints, Mapping) or set(endpoints) != set(frozen.ENDPOINT_ORDER):
        raise ValueError("endpoint identity map differs from the frozen four endpoints")
    for name, digest in endpoints.items():
        _sha_field(digest, f"{name} checkpoint sha256")
    for key in ("old_cc1", "old_v1"):
        evidence = contract.get(key)
        if not isinstance(evidence, Mapping):
            raise ValueError(f"{key} evidence binding is missing")
        if not isinstance(evidence.get("attempt_id"), str) or not evidence["attempt_id"]:
            raise ValueError(f"{key} attempt identity is missing")
        raw = evidence.get("raw_sha256")
        if not isinstance(raw, Mapping) or set(raw) != {"attempt", "failure", "result"}:
            raise ValueError(f"{key} raw evidence map is incomplete")
        for name, digest in raw.items():
            _sha_field(digest, f"{key} {name} sha256")
        _sha_field(evidence.get("ledger_sha256"), f"{key} ledger sha256")
    sources = contract.get("approved_source_sha256")
    if not isinstance(sources, Mapping) or not set(REQUIRED_SOURCES).issubset(sources):
        raise ValueError("recovery source snapshot is incomplete")
    for value in sources.values():
        _sha_field(value, "recovery source")


def active_contract_path(reports_dir: Path | None = None) -> Path:
    """Highest published contract revision; a fix publishes a new revision."""
    directory = Path(reports_dir or (ROOT / "reports"))
    revisions: list[tuple[int, Path]] = []
    for path in directory.glob(CONTRACT_GLOB):
        match = CONTRACT_PATTERN.fullmatch(path.name)
        if match:
            revisions.append((int(match.group(1)), path))
    if not revisions:
        raise ValueError("no recovery v2 contract revision is published")
    return max(revisions)[1]


def read_marker(marker_path: Path | None = None) -> dict[str, Any] | None:
    """``None`` = absent; a valid marker; or ``{"state": "unknown"}`` = consumed."""
    path = Path(marker_path or MARKER)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"state": "unknown"}
    if not isinstance(payload, dict) or payload.get("schema") != MARKER_SCHEMA:
        return {"state": "unknown"}
    if not isinstance(payload.get("attempt_id"), str) or not payload["attempt_id"]:
        return {"state": "unknown"}
    endpoints = payload.get("endpoint_checkpoint_sha256")
    if not isinstance(endpoints, dict) or set(endpoints) != set(frozen.ENDPOINT_ORDER):
        return {"state": "unknown"}
    if not isinstance(payload.get("source_hashes_sha256"), str):
        return {"state": "unknown"}
    if not isinstance(payload.get("contract_sha256"), str):
        return {"state": "unknown"}
    return payload


def read_window_events(ledger_path: Path | None = None) -> list[dict[str, Any]]:
    path = Path(ledger_path or WINDOW_LEDGER)
    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"window ledger is corrupt: {exc}") from exc
        if not isinstance(event, dict):
            raise ValueError("window ledger contains a non-object event")
        events.append(event)
    return events


def window_summary(ledger_path: Path | None = None) -> dict[str, Any]:
    events = read_window_events(ledger_path)
    measured_cpu = measured_wall = 0.0
    unknown_tail_events = 0
    for event in events:
        if event.get("event") not in {
            "admission_completed",
            "admission_failed",
            "formal_completed",
            "formal_failed",
        }:
            continue
        measured_cpu += _number(event.get("cpu_seconds_measured"), "window CPU charge")
        measured_wall += _number(event.get("wall_seconds_measured"), "window wall charge")
        if event.get("unknown_tail") is True:
            unknown_tail_events += 1
    remaining_cpu = (
        WINDOW_CPU_LIMIT_SECONDS
        - HISTORICAL_CPU_RESERVE_SECONDS
        - V1_MEASURED_CPU_SECONDS
        - measured_cpu
    )
    remaining_wall = (
        WINDOW_WALL_LIMIT_SECONDS
        - HISTORICAL_WALL_RESERVE_SECONDS
        - V1_MEASURED_WALL_SECONDS
        - measured_wall
    )
    return {
        "window_cpu_limit_seconds": WINDOW_CPU_LIMIT_SECONDS,
        "window_wall_limit_seconds": WINDOW_WALL_LIMIT_SECONDS,
        "historical_cpu_reserve_seconds": HISTORICAL_CPU_RESERVE_SECONDS,
        "historical_wall_reserve_seconds": HISTORICAL_WALL_RESERVE_SECONDS,
        "v1_measured_cpu_seconds": V1_MEASURED_CPU_SECONDS,
        "v1_measured_wall_seconds": V1_MEASURED_WALL_SECONDS,
        "v2_measured_cpu_seconds": measured_cpu,
        "v2_measured_wall_seconds": measured_wall,
        "remaining_cpu_seconds": remaining_cpu,
        "remaining_wall_seconds": remaining_wall,
        "unknown_tail_events": unknown_tail_events,
        "event_count": len(events),
    }


def next_sequence(kind: str, ledger_path: Path | None = None) -> int:
    if kind not in PHASES:
        raise ValueError("unknown evaluation phase")
    sequences = [
        int(event["sequence"])
        for event in read_window_events(ledger_path)
        if event.get("phase") == kind and isinstance(event.get("sequence"), int)
    ]
    return max(sequences, default=0) + 1


def prepared_formal_sequence(
    *, reports_dir: Path, prefix_base: str, ledger_path: Path | None = None
) -> int | None:
    """The single validate-only namespace that has not yet formed an attempt."""
    prepared: list[int] = []
    for sequence in range(1, next_sequence("formal", ledger_path)):
        paths = namespace_paths(
            "formal", sequence, reports_dir=reports_dir, prefix_base=prefix_base
        )
        if paths["readiness"].is_file() and not paths["attempt"].exists():
            prepared.append(sequence)
    if len(prepared) > 1:
        raise ValueError("more than one formal namespace has an unexecuted readiness")
    return prepared[0] if prepared else None


def namespace_paths(
    kind: str,
    sequence: int,
    *,
    reports_dir: Path | None = None,
    prefix_base: str = REPORT_PREFIX,
) -> dict[str, Path]:
    if kind not in PHASES:
        raise ValueError("unknown evaluation phase")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
        raise ValueError("namespace sequence must be a positive integer")
    directory = Path(reports_dir or (ROOT / "reports"))
    prefix = directory / f"{prefix_base}_{kind}_{sequence:03d}"
    suffixes = {
        "readiness": "readiness",
        "attempt": "attempt",
        "ledger": "ledger",
        "progress": "progress",
        "face": "face",
        "generations": "generations",
        "result": "result",
        "cost_receipt": "cost_receipt",
        "review": "review",
        "failure": "failure",
        "endpoint_manifest": "endpoints",
    }
    return {
        key: (
            prefix.with_name(f"{prefix.name}_{suffix}.jsonl")
            if key in {"ledger", "progress"}
            else prefix.with_name(f"{prefix.name}_{suffix}.json")
        )
        for key, suffix in suffixes.items()
    }


def append_window_event(
    context: Context,
    event: Mapping[str, Any],
    *,
    durability_api: Any | None = None,
) -> dict[str, Any]:
    payload = dict(event)
    payload.setdefault("recovery_id", RECOVERY_ID)
    payload.setdefault("timestamp_utc", datetime.now(UTC).isoformat())
    payload.setdefault("can_promote", False)
    api = durability_api or context.durability_api or frozen.WindowsDurabilityApi()
    frozen._append_jsonl(context.ledger_path, payload, flush_file=api.flush_file)
    return payload


@dataclass(frozen=True)
class Context:
    kind: str
    sequence: int
    reports_dir: Path = ROOT / "reports"
    prefix_base: str = REPORT_PREFIX
    contract_path: Path | None = None
    marker_path: Path = MARKER
    ledger_path: Path = WINDOW_LEDGER
    extension_preflight_fn: Callable[[], dict[str, Any]] = adapter.extension_preflight
    old_evidence_fn: Callable[[], dict[str, Any]] = default_old_evidence
    durability_api: Any | None = None

    def paths(self) -> dict[str, Path]:
        return namespace_paths(
            self.kind,
            self.sequence,
            reports_dir=self.reports_dir,
            prefix_base=self.prefix_base,
        )

    def preflight(self) -> dict[str, Any]:
        return preflight(self)

    def budget_factory(self, **kwargs: Any) -> frozen.EvaluationBudget:
        return frozen.EvaluationBudget(
            cpu_limit_seconds=FORMAL_CPU_LIMIT_SECONDS,
            wall_limit_seconds=FORMAL_WALL_LIMIT_SECONDS,
            **kwargs,
        )

    def prompt_selector(
        self,
        *,
        publish_fn: Callable[[Mapping[str, Any]], str] | None = None,
        selector_fn: Callable[[], Any] | None = None,
        durability_api: Any | None = None,
    ) -> Callable[[], Any]:
        return make_prompt_selector(
            self, publish_fn=publish_fn, selector_fn=selector_fn, durability_api=durability_api
        )

    def require_face_access(self) -> None:
        require_face_access(self)

    def record_outcome(
        self,
        event: str,
        *,
        cpu_seconds: float,
        wall_seconds: float,
        unknown_tail: bool,
        **extra: Any,
    ) -> dict[str, Any]:
        return append_window_event(
            self,
            {
                "event": event,
                "phase": self.kind,
                "sequence": self.sequence,
                "cpu_seconds_measured": _number(cpu_seconds, "phase CPU"),
                "wall_seconds_measured": _number(wall_seconds, "phase wall"),
                "unknown_tail": bool(unknown_tail),
                **extra,
            },
            durability_api=self.durability_api,
        )


def window_claims_face(context: Context) -> bool:
    """A durable ledger claim exists even when marker publication itself failed."""
    return any(
        event.get("event")
        in {"face_access_marker_published", "face_access_marker_publication_failed"}
        for event in read_window_events(context.ledger_path)
    )


def _marker_blocks(context: Context, marker: Mapping[str, Any] | None, contract: Path) -> bool:
    if marker is None:
        return window_claims_face(context)
    if marker.get("state") == "unknown":
        return True
    if marker.get("contract_sha256") != _sha(contract):
        return True
    if context.kind != "formal":
        return True
    attempt_path = context.paths()["attempt"]
    if not attempt_path.is_file():
        return True
    try:
        attempt = frozen.read_json(attempt_path)
    except (OSError, json.JSONDecodeError):
        return True
    return marker.get("attempt_id") != attempt.get("attempt_id")


def require_balance(context: Context, summary: Mapping[str, Any] | None = None) -> None:
    state = dict(summary or window_summary(context.ledger_path))
    need_cpu = (
        ADMISSION_CPU_LIMIT_SECONDS if context.kind == "admission" else FORMAL_CPU_LIMIT_SECONDS
    )
    need_wall = (
        ADMISSION_WALL_LIMIT_SECONDS if context.kind == "admission" else FORMAL_WALL_LIMIT_SECONDS
    )
    if state["remaining_cpu_seconds"] < need_cpu or state["remaining_wall_seconds"] < need_wall:
        raise ValueError(
            "evaluation window cannot cover the next phase "
            f"(remaining cpu={state['remaining_cpu_seconds']:.3f}/{need_cpu:.3f}, "
            f"wall={state['remaining_wall_seconds']:.3f}/{need_wall:.3f})"
        )


def _verify_old_evidence(contract: Mapping[str, Any], observed: Mapping[str, Any]) -> None:
    for key in ("old_cc1", "old_v1"):
        expected = contract[key]
        actual = observed.get(key.split("_", 1)[1])
        if not isinstance(actual, Mapping):
            raise ValueError(f"{key} evidence could not be audited")
        if actual.get("attempt_id") != expected["attempt_id"]:
            raise ValueError(f"{key} attempt identity changed")
        if actual.get("raw_sha256") != expected["raw_sha256"]:
            raise ValueError(f"{key} raw evidence changed")
        if actual.get("ledger_sha256") != expected["ledger_sha256"]:
            raise ValueError(f"{key} ledger changed")
        if actual.get("stopped_before_selection_in_this_attempt") is not True:
            raise ValueError(f"{key} is no longer a pre-selection engineering failure")


def preflight(context: Context) -> dict[str, Any]:
    """Single authorization seam: contract, old evidence, marker, balance, endpoints."""
    contract_path = context.contract_path or active_contract_path(context.reports_dir)
    contract = frozen.read_json(contract_path)
    require_authorization(contract)  # Before endpoints, checkpoints or any face access.
    _verify_old_evidence(contract, context.old_evidence_fn())
    marker = read_marker(context.marker_path)
    if _marker_blocks(context, marker, contract_path):
        raise frozen.Refuse(
            "formal face access is already consumed; this namespace may not read the dev face"
        )
    require_balance(context)
    result = context.extension_preflight_fn()
    observed = {row["endpoint_id"]: row["sha256"] for row in result["endpoints"]}
    if observed != contract["endpoint_checkpoint_sha256"]:
        raise ValueError("endpoint identity differs from the adopted recovery contract")
    for name, expected in contract["approved_source_sha256"].items():
        source = (ROOT / name).resolve()
        if (
            not source.is_relative_to(ROOT)
            or not source.is_file()
            or audit.sha256(source) != expected
        ):
            raise ValueError(f"recovery source changed: {name}")
    result["source_hashes"].update(
        {
            str((ROOT / name).resolve()): expected
            for name, expected in contract["approved_source_sha256"].items()
        }
    )
    result["source_hashes"][str(contract_path.resolve())] = audit.sha256(contract_path)
    return result


def _publish_marker(context: Context, payload: Mapping[str, Any]) -> str:
    api = frozen.WindowsDurabilityApi()
    qualification = frozen.probe_ntfs_durability(context.marker_path.parent, api=api)
    return frozen._durable_write_json(
        context.marker_path, payload, qualification=qualification, api=api
    )


def _marker_payload(
    context: Context,
    attempt: Mapping[str, Any],
    attempt_sha: str,
    contract_path: Path,
) -> dict[str, Any]:
    readiness = attempt.get("readiness")
    if not isinstance(readiness, Mapping):
        raise frozen.Refuse("formal attempt has no readiness binding for the face marker")
    source_hashes = attempt.get("source_hashes")
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        raise frozen.Refuse("formal attempt has no source binding for the face marker")
    checkpoints = attempt.get("endpoint_checkpoint_sha256")
    if not isinstance(checkpoints, Mapping) or set(checkpoints) != set(frozen.ENDPOINT_ORDER):
        raise frozen.Refuse("formal attempt has no four-endpoint binding for the face marker")
    return {
        "schema": MARKER_SCHEMA,
        "recovery_id": RECOVERY_ID,
        "phase": context.kind,
        "sequence": context.sequence,
        "attempt_id": attempt["attempt_id"],
        "attempt_sha256": attempt_sha,
        "readiness_sha256": canonical_sha(readiness),
        "endpoint_checkpoint_sha256": dict(checkpoints),
        "source_hashes_sha256": canonical_sha(source_hashes),
        "contract_sha256": _sha(contract_path),
        "window_ledger_sha256": (
            _sha(context.ledger_path) if context.ledger_path.is_file() else None
        ),
        "selector_pid": os.getpid(),
        "published_at_utc": datetime.now(UTC).isoformat(),
        "can_promote": False,
    }


def make_prompt_selector(
    context: Context,
    *,
    publish_fn: Callable[[Mapping[str, Any]], str] | None = None,
    selector_fn: Callable[[], Any] | None = None,
    durability_api: Any | None = None,
) -> Callable[[], Any]:
    """Publish the one durable marker *before* touching the real dev selector."""

    def select() -> Any:
        if read_marker(context.marker_path) is not None or window_claims_face(context):
            raise frozen.Refuse("formal face access is already consumed; selector is refused")
        attempt_path = context.paths()["attempt"]
        attempt = frozen.read_json(attempt_path)
        if attempt.get("schema") != ATTEMPT_SCHEMA or attempt.get("status") != "started":
            raise frozen.Refuse("face access requires a live formal evaluation attempt")
        payload = _marker_payload(
            context,
            attempt,
            _sha(attempt_path),
            context.contract_path or active_contract_path(context.reports_dir),
        )
        publish = publish_fn or (lambda value: _publish_marker(context, value))
        try:
            marker_sha = publish(payload)
        except BaseException as exc:
            append_window_event(
                context,
                {
                    "event": "face_access_marker_publication_failed",
                    "phase": context.kind,
                    "sequence": context.sequence,
                    "attempt_id": attempt.get("attempt_id"),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                durability_api=durability_api,
            )
            raise
        append_window_event(
            context,
            {
                "event": "face_access_marker_published",
                "phase": context.kind,
                "sequence": context.sequence,
                "attempt_id": attempt["attempt_id"],
                "marker_sha256": marker_sha,
            },
            durability_api=durability_api,
        )
        return (selector_fn or base._select_dev_prompts)()

    return select


def require_face_access(context: Context) -> None:
    """Worker generation gate: no generation before the formal marker exists."""
    marker = read_marker(context.marker_path)
    if marker is None or marker.get("state") == "unknown":
        raise frozen.Refuse("worker generation is refused before the formal face marker")
    paths = context.paths()
    attempt = frozen.read_json(paths["attempt"])
    if marker.get("attempt_id") != attempt.get("attempt_id"):
        raise frozen.Refuse("worker generation marker belongs to another attempt")
    if marker.get("endpoint_checkpoint_sha256") != attempt.get("endpoint_checkpoint_sha256"):
        raise frozen.Refuse("worker generation marker endpoint identity differs from the attempt")
    if not paths["face"].is_file():
        raise frozen.Refuse("worker generation requires the persisted face of this namespace")


def new_task_token() -> tuple[str, str]:
    return secrets.token_urlsafe(32), secrets.token_hex(8)
