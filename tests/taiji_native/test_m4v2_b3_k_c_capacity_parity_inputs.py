import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "reports" / "taiji_m4v2_b3_k_c_capacity_parity_input_preflight_20260910.json"


def test_capacity_parity_input_preflight_is_ready_for_build_only() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    assert report["report_format"] == "taiji-m4v2-b3-k-c-capacity-parity-input-preflight-v1"
    assert report["status"] == "ready-for-candidate-design"
    assert report["cell_count"] == 9
    assert report["fixed_large_ready"] is True
    assert report["candidate_artifact_present"] is False
    assert report["candidate_design_ready"] is False
    assert report["can_start_candidate_design"] is True
    assert report["can_start_candidate_parity_build"] is False
    assert report["can_start_formal"] is False
    assert report["can_promote"] is False
    assert report["contract"]["target_parameter_bytes"] == 38664
    assert report["contract"]["target_update_steps_per_cell"] == 14252
    assert report["contract"]["target_checkpoint_count_per_cell"] == 9
    assert report["contract"]["naive_replica_duplication_forbidden"] is True
