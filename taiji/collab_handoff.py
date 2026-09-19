"""HANDOFF-M4 composition rule (``rule_revision = 1``) as a product component.

Source of truth: the P5.2b group-causal gate's ``_member_episode`` selection
semantics (WP-3 landed 2026-09-15; preregistration
``M5_P5_2B_GROUP_CAUSAL_CORPORA_PREREGISTRATION``).  The gate keeps its own
inline copy as the frozen measurement instrument; this module is the
product-side rule so that collaboration selection no longer exists only inside
an instrument (the 01 section 3 gap this package closes).

Semantics (identical to the gate's decision block):
- every active member is genuinely invoked each tick; a member whose
  prediction binds (``bind_failure is None``) is a candidate;
- no bindable candidate  -> terminal step ``all_members_exhausted``;
- revision 1 selection   -> the first candidate in ``active_members`` order
  whose member is not *blocked*, where a member is blocked when its most
  recent selection attempt after the last successful step was not executed;
  every candidate blocked -> terminal step ``all_members_blocked`` (failure
  hands off instead of retrying to STEP_CAP);
- revision 0 (historical, kept reproducible) -> the first bindable candidate.

The cue-side rule (a member is queried at its OWN executed step count + 1,
not the episode's) is exposed as :func:`member_cue_count` for callers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .contracts import _check_version

COMPOSITION_RULE = "m4_failure_handoff"
RULE_REVISION = 1
RULE_REVISION_0 = 0
STOP_ALL_MEMBERS_EXHAUSTED = "all_members_exhausted"
STOP_ALL_MEMBERS_BLOCKED = "all_members_blocked"


@dataclass(frozen=True)
class MemberCall:
    """One active member's bound prediction for the current tick."""

    member: str
    bind_failure: str | None  # None = the prediction bound (candidate)


@dataclass(frozen=True)
class ExecutionStep:
    """The policy-relevant projection of one recorded step (list order = tick order)."""

    chosen: str | None
    executed: bool


@dataclass(frozen=True)
class HandoffDecision:
    chosen: MemberCall | None
    stop: str | None  # None = a member was chosen and execution may proceed


class FailureHandoffPolicy:
    """The HANDOFF-M4 member selection policy (product side)."""

    def __init__(
        self,
        active_members: Sequence[str],
        *,
        rule_revision: int = RULE_REVISION,
        version: int = 1,
    ) -> None:
        members = tuple(str(item) for item in active_members)
        if len(members) < 2:
            raise ValueError("handoff policy needs at least two active members")
        if len(set(members)) != len(members):
            raise ValueError("handoff policy active members must be unique")
        if rule_revision not in (RULE_REVISION, RULE_REVISION_0):
            raise ValueError(f"unsupported rule revision: {rule_revision}")
        self.active_members = members
        self.rule_revision = int(rule_revision)
        self.composition_rule = COMPOSITION_RULE
        self.version = int(version)
        _check_version(self.version)

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": "taiji-collab-handoff-policy-v1",
            "composition_rule": self.composition_rule,
            "rule_revision": self.rule_revision,
            "active_members": list(self.active_members),
            "version": self.version,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> FailureHandoffPolicy:
        if payload.get("format") != "taiji-collab-handoff-policy-v1":
            raise ValueError("not a collab handoff policy payload")
        if payload.get("composition_rule") != COMPOSITION_RULE:
            raise ValueError(f"unknown composition rule: {payload.get('composition_rule')}")
        return cls(
            tuple(payload["active_members"]),
            rule_revision=int(payload["rule_revision"]),
            version=int(payload.get("version", 1)),
        )

    def select(
        self,
        calls: Sequence[MemberCall],
        steps: Sequence[ExecutionStep],
    ) -> HandoffDecision:
        """One selection decision for the current tick.

        ``calls`` covers every active member in ``active_members`` order;
        ``steps`` is the episode's prior step projection (ordered).
        """

        by_member: dict[str, MemberCall] = {}
        for call in calls:
            if call.member in by_member:
                raise ValueError(f"duplicate member call: {call.member}")
            if call.member not in self.active_members:
                raise ValueError(f"call from non-active member: {call.member}")
            by_member[call.member] = call
        bindable = [by_member[member] for member in self.active_members if member in by_member and by_member[member].bind_failure is None]

        if not bindable:
            return HandoffDecision(chosen=None, stop=STOP_ALL_MEMBERS_EXHAUSTED)

        if self.rule_revision == RULE_REVISION_0:
            return HandoffDecision(chosen=bindable[0], stop=None)

        last_success_index = max(
            (index for index, step in enumerate(steps) if step.executed),
            default=-1,
        )
        blocked = {
            step.chosen
            for index, step in enumerate(steps)
            if step.chosen is not None and not step.executed and index > last_success_index
        }
        chosen = next((call for call in bindable if call.member not in blocked), None)
        if chosen is None:
            return HandoffDecision(chosen=None, stop=STOP_ALL_MEMBERS_BLOCKED)
        return HandoffDecision(chosen=chosen, stop=None)


def member_own_steps(steps: Sequence[ExecutionStep], member: str) -> int:
    """How many executed steps this member already owns (cue rule, §(b))."""

    return sum(1 for step in steps if step.executed and step.chosen == member)


def member_cue_count(steps: Sequence[ExecutionStep], member: str) -> int:
    """The member is queried at its OWN executed step count + 1 (never the
    episode's step count, so a late joiner is queried inside its repertoire)."""

    return member_own_steps(steps, member) + 1
