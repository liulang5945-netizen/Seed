from __future__ import annotations

import pytest

from scripts.training import _taiji_n1_plan120_lossless_receipt as receipt


def _measurement(ratios: dict[str, float]) -> dict:
    radius = 2 * 3**0.5
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


def test_receipt_accepts_shared_radius_with_an_over_cap_group() -> None:
    measurement = _measurement({"reader": 2.5, "writer": 1.0, "local": 0.5})

    assert measurement["groups"]["reader"]["actual_l2"] > measurement["groups"]["reader"]["cap"]
    receipt.validate_adam_proposal_measurement(measurement)


def test_receipt_rejects_outside_shared_radius_and_independent_cap() -> None:
    outside = _measurement({"reader": 3.6, "writer": 0.1, "local": 0.1})
    proposal_norm = (3.6**2 + 0.1**2 * 2) ** 0.5
    for name in ("reader", "writer", "local"):
        group = outside["groups"][name]
        group["scale"] = 1.0
        group["clipped"] = False
        group["actual_l2"] = group["proposed_l2"]
        group["shared_normalized_scale"] = 1.0
        group["shared_normalized_actual_l2"] = proposal_norm

    with pytest.raises(ValueError, match="bounded actual Adam proposal"):
        receipt.validate_adam_proposal_measurement(outside)

    independent = _measurement({"reader": 1.2, "writer": 1.0, "local": 0.8})["groups"]["context"]
    independent["actual_l2"] = independent["cap"] * 1.1
    measurement = _measurement({"reader": 1.2, "writer": 1.0, "local": 0.8})
    measurement["groups"]["context"] = independent

    with pytest.raises(ValueError, match="bounded actual Adam proposal"):
        receipt.validate_adam_proposal_measurement(measurement)
