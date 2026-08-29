"""Contract checks for the W7-G0 versioned Gate manifests."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = {
    "W7-R1": ROOT / "plans/manifests/taiji_w7_r1_provider_watchdog_v1.json",
    "W7-R2": ROOT / "plans/manifests/taiji_w7_r2_interaction_group_v1.json",
    "W7-R3": ROOT / "plans/manifests/taiji_w7_r3_visual_desktop_v1.json",
    "W7-R4": ROOT / "plans/manifests/taiji_w7_r4_cuda_v1.json",
    "W7-R5": ROOT / "plans/manifests/taiji_w7_r5_open_domain_growth_v1.json",
}


def _load(work_package: str) -> dict[str, object]:
    with MANIFESTS[work_package].open(encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["work_package"] == work_package
    assert payload["version"] == 1
    return payload


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
