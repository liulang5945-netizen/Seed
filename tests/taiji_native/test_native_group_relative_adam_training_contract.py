"""Fixed calibration must retain its rule and reject engineering/partial checkpoints."""

from copy import deepcopy

import pytest

from scripts.training import run_taiji_n1_group_relative_adam as train


def test_original_zero_full_recovery_and_reject_wrong_phase_or_rule(tmp_path):
    train.torch.set_num_threads(1)
    _, _, paths = train.readiness()
    source = train.torch.load(paths["zero"], map_location="cpu", weights_only=True)
    model = train.runner.transfer.LocalRelationEvidenceCandidate.restore_bundle(source["candidate"])
    optimizer = train.torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.load_state_dict(deepcopy(source["candidate"]["optimizer"]))
    saved = train.engineering.save(
        model, optimizer, tmp_path / "zero.pt", 0, protocol=train.PROTOCOL
    )
    assert train.restore(tmp_path / "zero.pt")["signature"] == saved["expected_signature"]
    assert train.paired.fingerprint(model.bundle(optimizer)) == train.paired.fingerprint(
        source["candidate"]
    )
    for changed in (
        dict(saved, completed_updates=2),
        dict(saved, protocol=train.engineering.PROTOCOL),
        dict(saved, update_policy=None),
        dict(saved, update_policy=dict(train.bounded.POLICY, relative_limit=0.01)),
    ):
        with pytest.raises(ValueError, match="protocol|marker"):
            train.load(changed)
    assert train.PROTOCOL["seed"] == 13
    assert train.PROTOCOL["updates"] == 6144
    assert train.PROTOCOL["qa_sequences"] == 9216
    assert train.PROTOCOL["signature_clone_updates"] == 4
    assert train.PROTOCOL["confirmation_queries"] == 0
    assert train.PROTOCOL["cpu_limit_seconds"] == 300
