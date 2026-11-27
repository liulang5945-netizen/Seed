"""Temporary legacy-policy bridge for frozen N1 fresh-zero source bundles."""

from __future__ import annotations

LEGACY_ZERO_SCHEMA = "n1-transition-factorial-paired-zero-v1"


def load_source(payload: dict, loader, policy: dict, legacy_policy: dict):
    """Validate a legacy zero under its original policy; keep all new bundles coupled."""
    if payload.get("schema") != LEGACY_ZERO_SCHEMA:
        return loader(payload)
    current_policy = dict(policy)
    policy.clear()
    policy.update(legacy_policy)
    try:
        return loader(payload)
    finally:
        policy.clear()
        policy.update(current_policy)
