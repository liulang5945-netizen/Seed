from __future__ import annotations

from scripts.training.eval_taiji_m3r4_preview_approval import run_gate


def test_m3r4_preview_approval_dry_run_gate(tmp_path) -> None:
    report = run_gate(tmp_path / "m3r4.json")
    assert report["status"] == "passed"
    assert report["can_promote"] is False
    assert all(report["gates"].values())
