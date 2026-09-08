from __future__ import annotations

from scripts.training.eval_taiji_m3r5_approved_execution import run_gate


def test_m3r5_isolated_approved_execution_gate(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SEED_M3R5_TMPDIR", str(tmp_path))
    report = run_gate(tmp_path / "m3r5.json")
    assert report["status"] == "passed"
    assert report["can_promote"] is False
    assert all(report["gates"].values())
