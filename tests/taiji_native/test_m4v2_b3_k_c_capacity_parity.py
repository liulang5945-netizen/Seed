import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "reports" / "taiji_m4v2_b3_k_c_capacity_parity_audit_20260910.json"


def test_capacity_parity_audit_blocks_confounded_comparison() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    assert report["report_format"] == "taiji-m4v2-b3-k-c-capacity-parity-audit-v1"
    assert report["status"] == "blocked-current-comparison-confounded"
    assert report["cell_count"] == 9
    assert report["aggregate"]["parameter_bytes_within_tolerance_all"] is False
    assert report["aggregate"]["training_update_steps_equal_all"] is False
    assert report["aggregate"]["same_training_episode_indexes_all"] is True
    assert report["current_verdict"]["candidate_learning_evidence_valid"] is True
    assert report["current_verdict"]["comparison_valid_for_learning_rule_claim"] is False
    assert report["current_verdict"]["can_promote"] is False
    assert report["preregistration"]["status"] == "pre-registered-blocked-until-parity"
    assert "parameter_bytes_mismatch" in report["current_verdict"]["reason_codes"]
    assert "training_update_steps_mismatch" in report["current_verdict"]["reason_codes"]
