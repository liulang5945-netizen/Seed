"""Two-hop training checkpoints own both runtime and optimizer identity."""

import pytest
import torch

from scripts.training import run_taiji_n1_iterative_read as iterative
from scripts.training import run_taiji_n1_observation_fidelity as one_hop


def test_two_hop_checkpoint_restores_adam_and_rejects_one_hop_protocol(tmp_path):
    torch.set_num_threads(1)
    item = {
        "split": "train",
        "observations": ["entity uses abcdef."],
        "question": "material?",
        "answer": "abcdef",
    }
    model, optimizer = iterative.fresh([item])
    iterative.train_step(model, optimizer, item, "first")
    path = tmp_path / "two-hop.pt"
    saved = iterative.save(model, optimizer, path, 1)
    assert saved["candidate"]["read_hops"] == 2
    assert iterative.restore(path)["signature"] == saved["expected_signature"]
    restored = iterative.IterativeEvidenceCandidate.restore_bundle(saved["candidate"])
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.001)
    restored_optimizer.load_state_dict(saved["candidate"]["optimizer"])
    iterative.train_step(model, optimizer, item, "next")
    iterative.train_step(restored, restored_optimizer, item, "next")
    assert all(
        torch.equal(value, restored.state_dict()[name])
        for name, value in model.state_dict().items()
    )
    with pytest.raises(ValueError, match="protocol"):
        one_hop.restore(path)
    saved["protocol"] = one_hop.PROTOCOL
    torch.save(saved, tmp_path / "wrong.pt")
    with pytest.raises(ValueError, match="protocol"):
        iterative.restore(tmp_path / "wrong.pt")
    with pytest.raises(ValueError, match="train episodes"):
        iterative.train_step(None, None, {"split": "holdout"}, "forbidden")
