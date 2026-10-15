"""Fresh parameters, empty histories and real from-start cut credit contracts."""

from copy import deepcopy

import pytest

from scripts.training import audit_taiji_n1_fresh_full_history as audit


def test_complete_seed_pairing_and_no_inherited_state():
    audit.torch.set_num_threads(1)
    fingerprints = []
    for seed in audit.SEEDS:
        values = []
        for _ in audit.ARMS:
            model, optimizer = audit.fresh(seed)
            assert not optimizer.state and not model._observations
            assert sum(p.numel() for p in model.parameters()) == 62056
            values.append(audit.paired.fingerprint(model.bundle(optimizer)))
        assert len(set(values)) == 1
        fingerprints.append(values[0])
    assert len(set(fingerprints)) == 3
    with pytest.raises(ValueError, match="seed"):
        audit.fresh(14)


@pytest.mark.parametrize("arm", audit.ARMS)
def test_two_step_signature_and_full_same_arm_restore(tmp_path, arm):
    audit.torch.set_num_threads(1)
    model, optimizer = audit.fresh(13)
    before = audit.paired.fingerprint(model.bundle(optimizer))
    path = tmp_path / f"{arm}.pt"
    saved = audit.save(model, optimizer, path, 13, arm)
    signature = saved["expected_signature"]
    assert audit.paired.fingerprint(model.bundle(optimizer)) == before
    assert signature["trajectory"][1]["relation_gru_gradient_l2"] > 0
    assert (
        [s["adam_states"] for s in signature["trajectory"]] == [37, 37]
        if arm == "cut"
        else [s["adam_states"] for s in signature["trajectory"]] == [41, 41]
    )
    if arm == "cut":
        assert all(s["measurements"]["writer"]["gradient_l2"] == 0 for s in signature["trajectory"])
    assert audit.restore(path)["signature"] == signature
    if arm == "position":
        result = audit.independent(path, signature, audit.paired.CpuBudget(120, clock=lambda: 0))
        assert result["equal"] and result["signature_clone_updates"] == 2
    for changed in (
        dict(saved, initialization=None),
        dict(saved, completed_updates=2),
        dict(saved, update_policy=None),
        dict(saved, seed=11),
    ):
        with pytest.raises(ValueError, match="fresh|marker"):
            audit.load(changed)


def test_stripped_moments_do_not_make_trained_weights_fresh(tmp_path):
    audit.torch.set_num_threads(1)
    model, optimizer = audit.fresh(11)
    saved = audit.save(model, optimizer, tmp_path / "zero.pt", 11, "cut")
    changed = deepcopy(saved)
    changed["candidate"]["state"]["embedding.weight"].add_(0.001)
    assert not changed["candidate"]["optimizer"]["state"]
    with pytest.raises(ValueError, match="exact fresh"):
        audit.load(changed)
    assert audit.PROTOCOL["main_trajectory_updates"] == 0
    assert audit.PROTOCOL["signature_clone_updates"] == 36
    assert audit.PROTOCOL["synthetic_qa_sequences"] == 72
    assert audit.PROTOCOL["confirmation_queries"] == 0
