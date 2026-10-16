"""Artifact-free nine-arm quartet recovery and strict calibration admission."""

from copy import deepcopy

import pytest

from scripts.training import audit_taiji_n1_transition_factorial_paired as audit


@pytest.mark.parametrize("seed", audit.SEEDS)
@pytest.mark.parametrize("arm", audit.ARMS)
def test_all_fresh_same_arm_quartet_roundtrips_without_curriculum(tmp_path, monkeypatch, seed, arm):
    audit.torch.set_num_threads(1)
    monkeypatch.setattr(
        audit.calibration.runner.core,
        "measure",
        lambda *args, **kwargs: pytest.fail("no curriculum measurements in engineering"),
    )
    model, optimizer = audit.fresh(seed)
    before = audit.paired.fingerprint(model.bundle(optimizer))
    path = tmp_path / f"s{seed}_{arm}.pt"
    payload = audit.save(model, optimizer, path, seed, arm)
    assert audit.paired.fingerprint(model.bundle(optimizer)) == before
    recovered = audit.restore(path)
    assert recovered["signature"] == payload["expected_signature"]
    assert recovered["signature_clone_updates"] == 2
    steps = recovered["signature"]["trajectory"]
    assert all(s["loss"]["qa_sequences"] == 4 for s in steps)
    assert all(s["adam_states"] == (37 if arm == "cut" else 41) for s in steps)
    assert steps[1]["relation_gru_gradient_l2"] > 0
    assert all((s["measurements"]["writer"]["gradient_l2"] == 0) == (arm == "cut") for s in steps)


def test_seed_distinction_and_same_seed_arm_identity():
    fingerprints = []
    for seed in audit.SEEDS:
        values = [audit.paired.fingerprint(audit.fresh(seed)[0].bundle()) for _ in audit.ARMS]
        assert len(set(values)) == 1
        fingerprints.append(values[0])
    assert len(set(fingerprints)) == 3


def test_bad_seed_arm_and_trained_zero_marker_rejected(tmp_path):
    with pytest.raises(ValueError):
        audit.fresh(99)
    model, optimizer = audit.fresh(13)
    with pytest.raises(ValueError):
        audit.signature(model, optimizer, 13, "other")
    payload = audit.save(model, optimizer, tmp_path / "zero.pt", 13, "position")
    payload["completed_updates"] = 1
    with pytest.raises(ValueError):
        audit.load(payload)


def receipt():
    summary = dict(
        free_exact=192,
        free_by_kind={k: 48 for k in audit.paired.KINDS},
        pair_exact=96,
        rename_free_exact=192,
        rename_free_by_kind={k: 48 for k in audit.paired.KINDS},
        rename_pair_exact=96,
        factorial={v: dict(joint_four_exact=48) for v in ("original", "rename")},
    )
    return dict(
        complete=True,
        training_sufficient=True,
        code_identity=audit.calibration.identity(),
        protocol=audit.calibration.PROTOCOL,
        completed_updates=24576,
        losses=[dict(qa_sequences=n) for n in (2, 2, 1, 1, 4, 4, 2, 2)] * 3072,
        measurements=[{}] * 24576,
        rows=[{}] * 4800,
        score_queries=4800,
        new_session_count=43008,
        final_steps={str(i): 24576 for i in range(41)},
        restore_checks=[dict(equal=True)] * 2,
        signature_clone_updates=8,
        synthetic_qa_sequences=32,
        cpu_seconds=1302,
        sources_unchanged=True,
        code_unchanged=True,
        cpu_accounting_complete=True,
        milestones=[summary],
        dev_queries=0,
        holdout_queries=0,
        confirmation_queries=0,
    )


@pytest.mark.parametrize(
    "defect", [None, "complete", "gate", "losses", "rows", "history", "adam", "query", "cpu"]
)
def test_strict_full_calibration_admission(defect):
    record = receipt()
    if defect == "complete":
        record["complete"] = False
    elif defect == "gate":
        record["milestones"][0]["factorial"]["original"]["joint_four_exact"] = 43
    elif defect in ("losses", "rows"):
        record[defect] = record[defect][:-1]
    elif defect == "history":
        record["new_session_count"] -= 1
    elif defect == "adam":
        record["final_steps"]["0"] = 24575
    elif defect == "query":
        record["confirmation_queries"] = 1
    elif defect == "cpu":
        record["cpu_accounting_complete"] = False
    if defect:
        with pytest.raises(ValueError):
            audit.validate_calibration(record)
    else:
        audit.validate_calibration(record)


def test_attempt_blocks_before_source_or_checkpoint_reads(tmp_path, monkeypatch):
    path = tmp_path / "attempt.json"
    path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(audit, "ATTEMPT", path)
    monkeypatch.setattr(audit, "readiness", lambda: pytest.fail("must refuse first"))
    with pytest.raises(FileExistsError):
        audit.audit()


def test_signature_rejects_history_and_nonempty_adam_parent():
    model, optimizer = audit.fresh(13)
    audit.update.train_step(model, optimizer, deepcopy(audit.SYNTHETIC), "trained", "position")
    with pytest.raises(ValueError):
        audit.signature(model, optimizer, 13, "position")
