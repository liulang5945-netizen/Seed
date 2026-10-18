from __future__ import annotations

import pytest

from scripts.training import _taiji_n1_plan118_lossless_receipt as receipt


def _receipt_measurement(ratios: dict[str, float]) -> dict:
    radius = 3**0.5
    proposal_norm = sum(value**2 for value in ratios.values()) ** 0.5
    scale = min(1.0, radius / proposal_norm) if proposal_norm else 1.0
    shared_actual = proposal_norm * scale
    groups = {}
    for name, ratio in ratios.items():
        cap = 0.01
        groups[name] = {
            "actual_l2": cap * ratio * scale,
            "cap": cap,
            "cap_scope": "shared_normalized_rwl_radius",
            "clipped": scale < 1.0,
            "gradient_l2": 0.1,
            "group_normalized_proposal": ratio,
            "parameter_l2": 10.0,
            "proposed_l2": cap * ratio,
            "scale": scale,
            "shared_normalized_actual_l2": shared_actual,
            "shared_normalized_proposal_l2": proposal_norm,
            "shared_normalized_radius": radius,
            "shared_normalized_scale": scale,
        }
    for name in ("context", "relation"):
        groups[name] = {
            "actual_l2": 0.005,
            "cap": 0.01,
            "cap_scope": "per_group",
            "clipped": False,
            "gradient_l2": 0.1,
            "parameter_l2": 10.0,
            "proposed_l2": 0.005,
            "scale": 1.0,
        }
    return {"groups": groups}


def test_receipt_accepts_shared_radius_when_one_rwl_group_exceeds_its_cap() -> None:
    measurement = _receipt_measurement({"reader": 1.4, "writer": 0.8, "local": 0.5})

    assert measurement["groups"]["reader"]["actual_l2"] > measurement["groups"]["reader"]["cap"]
    receipt.validate_adam_proposal_measurement(measurement)


def test_receipt_rejects_invalid_shared_radius_and_independent_cap() -> None:
    outside_radius = _receipt_measurement({"reader": 1.8, "writer": 0.1, "local": 0.1})
    proposal_norm = (1.8**2 + 0.1**2 * 2) ** 0.5
    for group in ("reader", "writer", "local"):
        metrics = outside_radius["groups"][group]
        metrics["scale"] = 1.0
        metrics["clipped"] = False
        metrics["actual_l2"] = metrics["proposed_l2"]
        metrics["shared_normalized_scale"] = 1.0
        metrics["shared_normalized_actual_l2"] = proposal_norm

    with pytest.raises(ValueError, match="bounded actual Adam proposal"):
        receipt.validate_adam_proposal_measurement(outside_radius)

    independent_over_cap = _receipt_measurement({"reader": 0.8, "writer": 0.7, "local": 0.6})
    independent = independent_over_cap["groups"]["context"]
    independent["actual_l2"] = independent["cap"] * 1.1

    with pytest.raises(ValueError, match="bounded actual Adam proposal"):
        receipt.validate_adam_proposal_measurement(independent_over_cap)
