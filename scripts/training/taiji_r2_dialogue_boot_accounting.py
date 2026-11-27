"""Resolve extension-only CPU upper bounds when boot timestamps differ by float noise."""

from __future__ import annotations

import math
from typing import Any

BOOT_TIMESTAMP_TOLERANCE_SECONDS = 0.001


def reconciled_host_cpu_upper(
    basis_charge: float,
    basis_snapshot: dict[str, Any],
    *,
    current_boot_identity: float,
    current_host_cpu_seconds: float,
) -> float:
    """Keep the entire prepared basis and add all host CPU since that basis.

    This is an upper bound for each arm's process CPU, including failed workers.
    It does not recalculate or credit any cost before the prepared snapshot.
    A different boot or a regressed host counter leaves the uncertainty unresolved.
    """
    values = (
        basis_charge,
        basis_snapshot["charged_cpu_seconds"],
        basis_snapshot["boot_identity"],
        basis_snapshot["host_cpu_seconds"],
        current_boot_identity,
        current_host_cpu_seconds,
    )
    if any(isinstance(value, bool) or not math.isfinite(float(value)) for value in values):
        raise ValueError("CPU reconciliation requires finite numeric evidence")
    if basis_charge < 0 or basis_charge != float(basis_snapshot["charged_cpu_seconds"]):
        raise ValueError("prepared charge and snapshot differ")
    if (
        abs(float(basis_snapshot["boot_identity"]) - current_boot_identity)
        > BOOT_TIMESTAMP_TOLERANCE_SECONDS
    ):
        raise ValueError("boot identity differs beyond timestamp precision")
    delta = current_host_cpu_seconds - float(basis_snapshot["host_cpu_seconds"])
    if delta < 0:
        raise ValueError("host CPU counter regressed")
    return float(basis_charge) + delta
