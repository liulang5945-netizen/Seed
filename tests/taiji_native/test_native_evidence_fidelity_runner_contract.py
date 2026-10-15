"""Training objective identity and Adam must survive fidelity checkpoint recovery."""

import pytest
import torch

from scripts.training import run_taiji_n1_observation_fidelity as fidelity


def test_fidelity_checkpoint_owns_protocol_and_resumes_adam(tmp_path):
    torch.set_num_threads(1)
    item = {
        "split": "train",
        "observations": ["entity uses abcdef."],
        "question": "material?",
        "answer": "abcdef",
    }
    model, optimizer = fidelity.fresh([item])
    fidelity.train_step(model, optimizer, item, "first")
    path = tmp_path / "fidelity.pt"
    saved = fidelity.save(model, optimizer, path, 1)
    assert fidelity.restore(path)["signature"] == saved["expected_signature"]
    assert saved["candidate"]["optimizer"]["state"]
    restored = fidelity.NativeEvidenceCandidate.restore_bundle(saved["candidate"])
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.001)
    restored_optimizer.load_state_dict(saved["candidate"]["optimizer"])
    fidelity.train_step(model, optimizer, item, "next")
    fidelity.train_step(restored, restored_optimizer, item, "next")
    for name, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])
    with pytest.raises(FileExistsError):
        fidelity.save(model, optimizer, path, 2)
    saved["protocol"] = {"schema": "QA-only"}
    torch.save(saved, tmp_path / "wrong.pt")
    with pytest.raises(ValueError, match="protocol"):
        fidelity.restore(tmp_path / "wrong.pt")


def test_fidelity_rejects_heldout_updates_before_model_access():
    with pytest.raises(ValueError, match="train episodes"):
        fidelity.train_step(None, None, {"split": "holdout"}, "forbidden")
