"""Development Learning & Verified Outcome Credit Router (`taiji-development-learning-v1`, VISION §3.4/VX4).

Enforces `GOV-LEARN` & `TAIJI_NATIVE_ARCHITECTURE_V1 §6.3`:
  * Strictly separates `state`, `episodic`, `fast_f`, `medium_psi`, and `slow_theta` planes;
  * Blocks `imagined`, `unknown`, `self_generated`, unauthorized, and duplicate events
    from entering parameter credit (`OutcomeRouter.route_event`);
  * Marks unmeasured uncertainty/learnability explicitly as `None` (`is_measured=False`),
    never fabricating numeric placeholders;
  * Verifies `retention_drop <= max_retention_drop` and `gain >= min_gain` with `LearningReceipt`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

SCHEMA = "taiji-development-learning-v1"
ProvenanceKind = Literal["real", "imagined", "unknown", "self_generated"]


@dataclass
class LearningPermissionPolicy:
    allow_state: bool = True
    allow_episodic: bool = True
    allow_fast_f: bool = True
    allow_medium_psi: bool = True
    allow_slow_theta: bool = False
    max_retention_drop: float = 0.02
    min_gain: float = 0.0005

    def allowed_planes(self) -> list[str]:
        planes: list[str] = []
        if self.allow_state:
            planes.append("state")
        if self.allow_episodic:
            planes.append("episodic")
        if self.allow_fast_f:
            planes.append("fast_f")
        if self.allow_medium_psi:
            planes.append("medium_psi")
        if self.allow_slow_theta:
            planes.append("slow_theta")
        return planes


@dataclass(frozen=True)
class VerifiedOutcomeEvent:
    event_id: str
    action_ref: str
    provenance: ProvenanceKind
    authorized: bool
    reward: float
    prediction_error: float
    cost: float


@dataclass(frozen=True)
class LearningGap:
    gap_kind: str
    source_ref: str
    calibrated_uncertainty: float | None
    estimated_learnability: float | None
    is_measured: bool


@dataclass(frozen=True)
class LearningReceipt:
    schema: str
    receipt_id: str
    event_id: str
    accepted: bool
    reject_reason: str
    updated_planes: tuple[str, ...]
    retention_drop: float
    gain: float
    cost: float


class OutcomeRouter:
    """Credit router enforcing provenance, authorization, idempotency, and retention gates."""

    def __init__(self, policy: LearningPermissionPolicy | None = None) -> None:
        self.policy = policy or LearningPermissionPolicy()
        self.settled_event_ids: set[str] = set()
        self.receipts: list[LearningReceipt] = []

    def route_event(
        self,
        event: VerifiedOutcomeEvent,
        *,
        apply_fn: Callable[[list[str]], Any],
        eval_old_fn: Callable[[], float],
        eval_new_fn: Callable[[], float],
    ) -> LearningReceipt:
        rid = f"lr_{len(self.receipts) + 1}"
        if event.event_id in self.settled_event_ids:
            rec = LearningReceipt(
                schema=SCHEMA,
                receipt_id=rid,
                event_id=event.event_id,
                accepted=False,
                reject_reason="duplicate_event",
                updated_planes=(),
                retention_drop=0.0,
                gain=0.0,
                cost=event.cost,
            )
            self.receipts.append(rec)
            return rec

        if event.provenance != "real":
            rec = LearningReceipt(
                schema=SCHEMA,
                receipt_id=rid,
                event_id=event.event_id,
                accepted=False,
                reject_reason=f"forbidden_provenance:{event.provenance}",
                updated_planes=(),
                retention_drop=0.0,
                gain=0.0,
                cost=event.cost,
            )
            self.receipts.append(rec)
            return rec

        if not event.authorized:
            rec = LearningReceipt(
                schema=SCHEMA,
                receipt_id=rid,
                event_id=event.event_id,
                accepted=False,
                reject_reason="unauthorized_feedback",
                updated_planes=(),
                retention_drop=0.0,
                gain=0.0,
                cost=event.cost,
            )
            self.receipts.append(rec)
            return rec

        planes = self.policy.allowed_planes()
        old_before = float(eval_old_fn())
        new_before = float(eval_new_fn())

        apply_fn(planes)

        old_after = float(eval_old_fn())
        new_after = float(eval_new_fn())
        retention_drop = round(max(0.0, old_before - old_after), 6)
        gain = round(new_after - new_before, 6)

        if retention_drop > self.policy.max_retention_drop:
            rec = LearningReceipt(
                schema=SCHEMA,
                receipt_id=rid,
                event_id=event.event_id,
                accepted=False,
                reject_reason=(
                    f"retention_breach:{retention_drop:.4f}>{self.policy.max_retention_drop:.4f}"
                ),
                updated_planes=(),
                retention_drop=retention_drop,
                gain=gain,
                cost=event.cost,
            )
            self.receipts.append(rec)
            return rec

        if gain < self.policy.min_gain:
            rec = LearningReceipt(
                schema=SCHEMA,
                receipt_id=rid,
                event_id=event.event_id,
                accepted=False,
                reject_reason=f"insufficient_gain:{gain:.4f}<{self.policy.min_gain:.4f}",
                updated_planes=(),
                retention_drop=retention_drop,
                gain=gain,
                cost=event.cost,
            )
            self.receipts.append(rec)
            return rec

        self.settled_event_ids.add(event.event_id)
        rec = LearningReceipt(
            schema=SCHEMA,
            receipt_id=rid,
            event_id=event.event_id,
            accepted=True,
            reject_reason="",
            updated_planes=tuple(planes),
            retention_drop=retention_drop,
            gain=gain,
            cost=event.cost,
        )
        self.receipts.append(rec)
        return rec
