"""Contract checks for the W7-G0 versioned Gate manifests."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = {
    "W7-R1": ROOT / "plans/manifests/taiji_w7_r1_provider_watchdog_v1.json",
    "W7-R2": ROOT / "plans/manifests/taiji_w7_r2_interaction_group_v1.json",
    "W7-R3": ROOT / "plans/manifests/taiji_w7_r3_visual_desktop_v1.json",
    "W7-R4": ROOT / "plans/manifests/taiji_w7_r4_cuda_v1.json",
    "W7-R5": ROOT / "plans/manifests/taiji_w7_r5_open_domain_growth_v1.json",
    "W7-R5A": ROOT / "plans/manifests/taiji_w7_r5_internalization_v1.json",
    "W7-R5B": ROOT / "plans/manifests/taiji_w7_r5_effector_registry_v1.json",
}


def _load(work_package: str) -> dict[str, object]:
    with MANIFESTS[work_package].open(encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["work_package"] == work_package
    assert payload["version"] == 1
    return payload


def _assert_internalization_contract(payload: dict[str, object]) -> None:
    owner = payload.get("owner")
    checkpoint = payload.get("checkpoint")
    claim = payload.get("claim")
    boundary = payload.get("boundary")
    assert isinstance(owner, dict)
    assert isinstance(checkpoint, dict)
    assert isinstance(claim, dict)
    assert isinstance(boundary, list)
    assert owner.get("converter") == "taiji/internalization.py"
    assert "seed_platform executor" in owner.get("forbidden_owners", [])
    assert checkpoint.get("required") is True
    assert any("not an execution channel" in item for item in claim.get("must_not_claim", []))
    assert any("never" in item and "executors" in item.lower() for item in boundary)


def _assert_effector_contract(payload: dict[str, object]) -> None:
    owner = payload.get("owner")
    checkpoint = payload.get("checkpoint")
    claim = payload.get("claim")
    boundary = payload.get("boundary")
    assert isinstance(owner, dict)
    assert isinstance(checkpoint, dict)
    assert isinstance(claim, dict)
    assert isinstance(boundary, list)
    assert owner.get("registry") == "seed_platform/capability_registry.py"
    assert "taiji cognition" in owner.get("forbidden_owners", [])
    assert checkpoint.get("required") is True
    assert any("bypass policy" in item for item in claim.get("must_not_claim", []))
    assert any("not deletable" in item for item in boundary)


def test_w7_g0_freezes_every_follow_on_gate() -> None:
    for work_package in MANIFESTS:
        payload = _load(work_package)
        assert payload["status"] in {"contract_frozen", "hardware-blocked"}
        for section in (
            "claim",
            "owner",
            "input",
            "output",
            "trace",
            "resources",
            "checkpoint",
            "evidence",
            "failure_isolation",
            "rollback",
            "implementation",
            "boundary",
        ):
            assert section in payload
        evidence = payload["evidence"]
        assert [layer["id"] for layer in evidence["layers"]] == ["S0", "S1", "S2"]
        assert evidence["red_proof"]
        assert evidence["holdout"]
        assert evidence["lesion"]
        assert payload["checkpoint"]["required"] is True
        assert payload["implementation"]["status"] != "passed"


def test_cuda_manifest_stays_explicitly_blocked_without_hardware() -> None:
    payload = _load("W7-R4")
    assert payload["status"] == "hardware-blocked"
    assert payload["input"]["current_host"] == {
        "device": "cpu",
        "cuda_available": False,
        "state": "hardware-blocked",
    }
    assert "CPU measurements are not CUDA support" in payload["claim"]["must_not_claim"]


def test_w7_manifests_forbid_frontend_or_provider_cognitive_ownership() -> None:
    r1 = _load("W7-R1")
    r3 = _load("W7-R3")
    assert "frontend" in r1["owner"]["forbidden_owners"]
    assert "mock capability fixtures in packaged smoke" in r3["owner"]["forbidden_owners"]


def test_r5a_and_r5b_are_separate_non_substituting_contracts() -> None:
    r5a = _load("W7-R5A")
    r5b = _load("W7-R5B")

    assert r5a["work_package"] != r5b["work_package"]
    assert r5a["owner"]["converter"] == "taiji/internalization.py"
    assert r5b["owner"]["registry"] == "seed_platform/capability_registry.py"
    assert "seed_platform executor" in r5a["owner"]["forbidden_owners"]
    assert "taiji cognition" in r5b["owner"]["forbidden_owners"]
    assert any("grounded" in item for item in r5a["claim"]["must_not_claim"])
    assert any("policy" in item for item in r5b["claim"]["must_not_claim"])
    assert "external_description_tombstoned" in r5a["trace"]["required_events"]
    assert "bundle_retired" in r5b["trace"]["required_events"]
    assert "executor" in " ".join(r5b["checkpoint"]["must_roundtrip"])
    assert "rejected" in r5a["checkpoint"]["continuation"]
    assert r5a["implementation"]["status"] == "not_started"
    assert r5b["implementation"]["status"] == "not_started"
    _assert_internalization_contract(r5a)
    _assert_effector_contract(r5b)


def test_r5_split_contract_rejects_missing_or_mixed_boundaries() -> None:
    r5a = _load("W7-R5A")
    r5b = _load("W7-R5B")

    missing_owner = deepcopy(r5a)
    del missing_owner["owner"]["converter"]
    with pytest.raises(AssertionError):
        _assert_internalization_contract(missing_owner)

    mixed_owner = deepcopy(r5b)
    mixed_owner["owner"]["registry"] = "taiji/internalization.py"
    with pytest.raises(AssertionError):
        _assert_effector_contract(mixed_owner)

    missing_checkpoint = deepcopy(r5a)
    del missing_checkpoint["checkpoint"]
    with pytest.raises(AssertionError):
        _assert_internalization_contract(missing_checkpoint)

    cognition_owner = deepcopy(r5b)
    cognition_owner["owner"]["forbidden_owners"].remove("taiji cognition")
    with pytest.raises(AssertionError):
        _assert_effector_contract(cognition_owner)

    unsafe_deletion = deepcopy(r5a)
    unsafe_deletion["boundary"][1] = "Executors may be physically deleted as knowledge artifacts."
    with pytest.raises(AssertionError):
        _assert_internalization_contract(unsafe_deletion)
