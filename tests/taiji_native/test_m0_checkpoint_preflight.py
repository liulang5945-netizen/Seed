from __future__ import annotations

import json
from pathlib import Path

from scripts.training.eval_taiji_m0_checkpoint_preflight import run_gate


def test_m0_checkpoint_preflight_uses_a_fresh_process() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    tmp_path = repo_root / ".seed_test_tmp" / "m0-checkpoint-preflight-current"
    report = run_gate(tmp_path / "checkpoints", tmp_path / "report.json")

    assert report["status"] == "passed"
    assert report["checks"] == {
        "parent_saved": True,
        "child_saved_after_fresh_restore": True,
        "fresh_process_completed": True,
        "fresh_process_next_step_matches": True,
        "fresh_process_checkpoint_matches": True,
        "report_written": True,
    }
    saved = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert saved["status"] == "passed"
