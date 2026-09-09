from __future__ import annotations

import json
from pathlib import Path

REPORT = (
    Path(__file__).resolve().parents[2]
    / "reports"
    / "taiji_m4v2_b3_k_update_signature_audit_20260910.json"
)


def test_b3_k_update_signature_audit_covers_six_real_episodes() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["status"] == "passed"
    assert report["run_kind"] == "per-episode-update-signature-audit"
    assert report["episode_seeds"] == [0, 1, 2, 3, 4, 5]
    assert report["cell_count"] == 6
    assert report["same_parent"] is True
    assert report["input_signatures_distinct"] is True
    assert report["technical_gate_passed"] is True
    assert report["can_promote"] is False
    assert report["worker_candidate_update_collision_groups"]["k1.semantic"]
    assert report["worker_candidate_update_collision_groups"]["k2.transition"]


def test_b3_k_update_signature_audit_records_batch_collisions_and_boundaries() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert len(report["candidate_update_digests"]) == 6
    assert all(
        cell["report"]["train_episode_count"] == 1
        and cell["report"]["training_update_steps"] == 2
        and cell["report"]["checks"]["no_update_control"]
        and cell["report"]["checks"]["candidate_checkpoint_fresh_restore"]
        and cell["report"]["checks"]["adapter_rollback_restored"]
        and cell["report"]["checks"]["k3_owner_unchanged"]
        for cell in report["cells"]
    )
    assert set(report["batch_comparison"]) == {"three_sliding", "three_non_sliding"}
    assert all(
        comparison["candidate_digest_matches_single"]
        for comparison in report["batch_comparison"].values()
    )
