"""The bounded continuation owns new checkpoints, never the formal parent."""

import pytest
import torch

from scripts.training import run_taiji_n1_native_evidence_paired as paired
from scripts.training import run_taiji_n1_train_only_learning_curve as curve
from taiji.native_evidence_candidate import NativeEvidenceCandidate


def test_curve_checkpoint_is_independent_self_contained_and_refuses_overwrite(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = NativeEvidenceCandidate()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    item = {
        "observations": ["copper"],
        "question": "material?",
        "split": "train",
        "answer": "copper",
    }
    paired.observe(model, paired.project(item), "zero-probe")
    paired.train_step(model, optimizer, item, "one", "position")
    path = tmp_path / "curve.pt"
    saved = curve.save(model, optimizer, path, 1, "original-parent")
    assert curve.restore(path)["signature"] == saved["expected_signature"]
    assert saved["protocol"]["additional_updates"] == 3072
    assert saved["parent_sha256"] == "original-parent"
    with pytest.raises(FileExistsError):
        curve.save(model, optimizer, path, 1, "original-parent")
