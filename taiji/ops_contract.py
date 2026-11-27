"""Pure-stdlib N36/N37 operational contract & 3-tier change classifier.

Kept free of `torch` and `scripts.*` imports so `api.routes_taiji_ops` can be
imported during Docker image build before `scripts/` is copied.
"""

from __future__ import annotations

from typing import Any

SCOPE = "bounded_closed_vocab_and_multimodule_v1"
BOUNDARIES: dict[str, Any] = {
    "check_reads": 0,
    "final_reads": 0,
    "open_chinese": "deferred_post_N35",
    "physical_sensors": "synthetic_two_family_only",
    "not_signed": ["24-S", "25-S", "29", "final", "C", "D", "J"],
}

STATES: tuple[str, ...] = (
    "Qualified",
    "Trial",
    "Running",
    "Waiting",
    "Shadow",
    "Restricted",
    "Paused",
    "Archived",
)

ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    "Qualified": frozenset({"Trial"}),
    "Trial": frozenset({"Running", "Paused"}),
    "Running": frozenset({"Waiting", "Shadow", "Restricted", "Paused", "Archived"}),
    "Waiting": frozenset({"Running", "Paused"}),
    "Shadow": frozenset({"Running", "Paused"}),
    "Restricted": frozenset({"Paused", "Running"}),
    "Paused": frozenset({"Trial", "Running", "Archived"}),
    "Archived": frozenset(),
}

_TIER1_OPS = frozenset({"ops_refactor", "logging", "cli"})
_TIER2_SHADOW = frozenset(
    {
        "ecl_calibration",
        "unc_recalibration",
        "phy_bounded_step",
        "growth_proposal",
        "harmful_phy_corrupt",
    }
)
_TIER3_CONTRACT: dict[str, list[str]] = {
    "readout_class": ["N31", "N32", "N35", "N36"],
    "objective_change": ["N32", "N33", "N35", "N36"],
    "dynamics_rule": ["N34", "N35", "N36"],
    "modality_change": ["N32", "N34", "N35", "N36"],
    "rights_boundary": ["N35", "N36", "N37"],
}


class OutOfContractChangeError(RuntimeError):
    """Raised when a Tier-3 change attempts to run inside daily Shadow."""


def classify_change(change_kind: str) -> dict[str, Any]:
    """Route a proposed change into Tier-1 ops, Tier-2 Shadow, or Tier-3 research reopen."""
    if change_kind in _TIER1_OPS:
        return {
            "change_kind": change_kind,
            "tier": "tier1_ops_only",
            "route": "engineering_regression",
            "allowed_in_shadow": False,
            "reopen_research_required": False,
            "impacted_stages": [],
        }
    if change_kind in _TIER2_SHADOW:
        return {
            "change_kind": change_kind,
            "tier": "tier2_in_contract_shadow",
            "route": "daily_shadow",
            "allowed_in_shadow": True,
            "reopen_research_required": False,
            "impacted_stages": [],
        }
    if change_kind in _TIER3_CONTRACT:
        return {
            "change_kind": change_kind,
            "tier": "tier3_contract_or_architecture",
            "route": "reopen_research_required",
            "allowed_in_shadow": False,
            "reopen_research_required": True,
            "impacted_stages": list(_TIER3_CONTRACT[change_kind]),
        }
    raise OutOfContractChangeError(
        f"reopen_research_required: unknown/out-of-contract change_kind {change_kind!r}"
    )
