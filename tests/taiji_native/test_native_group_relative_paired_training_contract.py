"""Artifact-free fixed training admission, same-arm recovery and evidence contracts."""

import json
from copy import deepcopy

import pytest

from scripts.training import run_taiji_n1_group_relative_paired as train
from tests.taiji_native._native_relation_fixtures import paired_sources


@pytest.mark.parametrize("arm", train.ARMS)
def test_zero_restore_and_phase_policy_rejection(tmp_path, arm):
    train.torch.set_num_threads(1)
    ready, source = paired_sources()
    model, optimizer = train.runner.fresh(source, 12, ready)
    before = train.paired.fingerprint(model.bundle(optimizer))
    path = tmp_path / f"{arm}.pt"
    saved = train.save(model, optimizer, path, 12, arm, 0)
    assert train.restore(path)["signature"] == saved["expected_signature"]
    assert train.paired.fingerprint(model.bundle(optimizer)) == before
    if arm == "terminal":
        result = train.independent(path, saved["expected_signature"], train.paired.CpuBudget(2400))
        assert result["equal"] and result["signature_clone_updates"] == 1
    for changed in (
        dict(saved, completed_updates=2),
        dict(saved, protocol=train.engineering.PROTOCOL),
        dict(saved, update_policy=None),
        dict(saved, update_policy=dict(train.update.POLICY, relative_limit=0.01)),
        dict(saved, arm="invalid"),
    ):
        with pytest.raises(ValueError, match="marker|phase|identity"):
            train.load(changed)
    item = train.engineering.SYNTHETIC
    second = train.runner.core.renaming.second_question(item)
    for step in range(2):
        loss, groups = train.update.train_step(
            model, optimizer, item, f"synthetic70-{step}", arm, second
        )
        assert loss["qa_sequences"] == 2
        assert all(g["actual_l2"] <= g["cap"] * 1.0001 + 1e-6 for g in groups.values())
    if arm == "cut":
        assert groups["writer"]["gradient_l2"] == 0
        assert groups["writer"]["actual_l2"] == 0
    with pytest.raises(ValueError, match="train"):
        train.update.train_step(
            model, optimizer, dict(item, split="holdout"), "forbidden70", arm, second
        )


def test_synthetic_final_layout_checks_all_moments_steps_and_history(tmp_path):
    """Explicit layout fixture only, no claim of 6144 trained updates."""
    train.torch.set_num_threads(1)
    ready, source = paired_sources()
    model, optimizer = train.runner.fresh(source, 13, ready)
    source_sessions = deepcopy(model.bundle()["sessions"])
    item = train.engineering.SYNTHETIC
    train.update.train_step(
        model,
        optimizer,
        item,
        "layout70-0",
        "cut",
        train.runner.core.renaming.second_question(item),
    )
    observations = model._observations["layout70-0"]
    for i in range(1, 6144):
        model._observations[f"layout70-{i}"] = deepcopy(observations)
    for name, parameter in model.named_parameters():
        optimizer.state[parameter]["step"].fill_(train.runner.expected_step(name, "cut", 6144))
    assert train.check_final(model, optimizer, source_sessions, "cut") == dict(
        source_sessions_unchanged=True, new_session_count=6144
    )
    payload = dict(
        schema="n1-group-relative-paired-checkpoint-v1",
        protocol=train.PROTOCOL,
        update_policy=train.update.POLICY,
        seed=13,
        arm="cut",
        completed_updates=6144,
        candidate=model.bundle(optimizer),
    )
    restored, opt = train.load(payload)
    assert train.paired.fingerprint(restored.bundle(opt)) == train.paired.fingerprint(
        payload["candidate"]
    )
    saved = train.save(model, optimizer, tmp_path / "final.pt", 13, "cut", 6144)
    assert train.restore(tmp_path / "final.pt")["signature"] == saved["expected_signature"]
    parameter = next(p for n, p in model.named_parameters() if n.startswith("writer."))
    optimizer.state[parameter]["step"].add_(1)
    with pytest.raises(ValueError, match="Adam steps"):
        train.check_final(model, optimizer, source_sessions, "cut")
    optimizer.state[parameter]["step"].sub_(1)
    del model._observations["layout70-1"]
    with pytest.raises(ValueError, match="history"):
        train.check_final(model, optimizer, source_sessions, "cut")


def test_fixed_scope_and_lossless_exclusive_report(tmp_path):
    assert train.PROTOCOL["total_updates"] == 55296
    assert train.PROTOCOL["total_qa_sequences"] == 82944
    assert train.PROTOCOL["signature_clone_updates"] == 36
    assert train.PROTOCOL["signature_clone_qa_sequences"] == 72
    assert train.PROTOCOL["cpu_limit_seconds"] == 2400
    assert train.PROTOCOL["confirmation_queries"] == 0
    path = tmp_path / "report.json"
    evidence = dict(
        rows=[dict(answer="石英", exact=True)], measurements=[dict(actual_l2=0.00001234567890123)]
    )
    train.write_new(path, evidence)
    assert json.loads(path.read_text(encoding="utf-8")) == evidence
    assert path.read_text(encoding="utf-8").count("\n") == 1
    with pytest.raises(FileExistsError):
        train.write_new(path, evidence)
