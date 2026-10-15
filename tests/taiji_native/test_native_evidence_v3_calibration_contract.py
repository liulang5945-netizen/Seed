"""New curriculum calibration must preserve strict train and checkpoint boundaries."""

import pytest
import torch

from scripts.training import run_taiji_n1_native_evidence_paired as paired
from scripts.training import run_taiji_n1_v3_train_calibration as calibration
from taiji.native_evidence_candidate import NativeEvidenceCandidate


def test_calibration_checkpoint_restores_optimizer_and_rejects_old_protocol(tmp_path):
    torch.set_num_threads(1)
    model = NativeEvidenceCandidate()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    item = {
        "observations": ["thing uses abcdef."],
        "question": "material?",
        "answer": "abcdef",
        "split": "train",
    }
    paired.observe(model, paired.project(item), "zero-probe")
    paired.train_step(model, optimizer, item, "fixture-step", "position")
    path = tmp_path / "candidate.pt"
    saved = calibration.save(model, optimizer, path, 1)
    assert calibration.restore(path)["signature"] == saved["expected_signature"]
    assert saved["candidate"]["optimizer"]["state"]
    with pytest.raises(FileExistsError):
        calibration.save(model, optimizer, path, 1)
    saved["protocol"] = {"schema": "old"}
    torch.save(saved, tmp_path / "wrong.pt")
    with pytest.raises(ValueError, match="protocol"):
        calibration.restore(tmp_path / "wrong.pt")


def test_train_diagnostic_rejects_heldout_before_model_access():
    with pytest.raises(ValueError, match="dev or holdout"):
        calibration.measure(None, {"split": "holdout"}, "position", "forbidden")


def test_sufficiency_requires_each_kind_as_well_as_total():
    assert calibration.sufficient(
        {"free_exact": 173, "free_by_kind": {"a": 43, "b": 43, "c": 43, "d": 44}}
    )
    assert not calibration.sufficient(
        {"free_exact": 173, "free_by_kind": {"a": 48, "b": 48, "c": 48, "d": 29}}
    )
