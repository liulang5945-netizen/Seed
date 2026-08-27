from __future__ import annotations

from scripts.training.eval_taiji_p3_transition_adjudication import evaluate


def test_p3_transition_adjudication_gate_passes() -> None:
    report = evaluate(seeds=(11,))

    assert report["aggregate"]["passed"] is True
    assert report["aggregate"]["first_calibration_min"] == 1.0
    assert report["aggregate"]["cross_episode_calibration_min"] == 1.0
    assert report["aggregate"]["contradiction_rejected_min"] == 1.0
    assert report["aggregate"]["stochastic_tie_rejected_min"] == 1.0
    assert report["aggregate"]["stochastic_mode_min"] == 1.0
    assert report["aggregate"]["stochastic_clear_leader_min"] == 1.0
    assert report["aggregate"]["no_update_on_reject_min"] == 1.0
    assert report["aggregate"]["checkpoint_registry_min"] == 1.0
    assert report["aggregate"]["checkpoint_network_min"] == 1.0
    assert report["aggregate"]["checkpoint_continuation_min"] == 1.0
