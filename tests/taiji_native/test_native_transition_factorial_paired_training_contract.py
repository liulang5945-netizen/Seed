"""Artifact-free nine-model training state and recovery contracts."""

from copy import deepcopy

import pytest

from scripts.training import run_taiji_n1_transition_factorial_paired as train


@pytest.mark.parametrize("seed,arm", [(11, "position"), (12, "terminal"), (13, "cut")])
def test_zero_same_arm_independent_recovery(seed, arm, tmp_path):
    train.torch.set_num_threads(1)
    model, optimizer = train.engineering.fresh(seed)
    before = train.paired.fingerprint(model.bundle(optimizer))
    path = tmp_path / "zero.pt"
    saved = train.save(model, optimizer, path, 0, seed, arm)
    assert train.restore(path)["signature"] == saved["expected_signature"]
    check = train.independent(
        path, saved["expected_signature"], train.paired.CpuBudget(14400, clock=lambda: 0)
    )
    assert check["equal"]
    assert train.paired.fingerprint(model.bundle(optimizer)) == before
    for changed in (
        dict(saved, phase="partial"),
        dict(saved, completed_updates=1),
        dict(saved, protocol=train.engineering.PROTOCOL),
        dict(saved, seed=99),
        dict(saved, arm="invalid"),
    ):
        with pytest.raises(ValueError):
            train.load(changed)


@pytest.mark.parametrize("arm", ["position", "terminal", "cut"])
def test_synthetic_final_counters_writer_guard_and_signature(arm, tmp_path):
    # Counter-layout fixture only, no claim that full training was performed.
    train.torch.set_num_threads(1)
    model, opt = train.engineering.fresh(11)
    item = train.engineering.SYNTHETIC
    train.update.train_step(
        model, opt, item, "fixture0", arm, train.runner.core.renaming.second_question(item)
    )
    observations = model._observations["fixture0/original"]
    model._observations.update({f"fixture{i}": observations for i in range(43008 - 4)})
    for state in opt.state.values():
        state["step"].fill_(train.UPDATES)
    train.validate_state(model, opt, train.UPDATES, 11, arm)
    path = tmp_path / "final.pt"
    saved = train.save(model, opt, path, train.UPDATES, 11, arm)
    assert train.restore(path)["signature"] == saved["expected_signature"]
    if arm == "cut":
        writer = next(p for n, p in model.named_parameters() if n.startswith("writer."))
        with train.torch.no_grad():
            writer.add_(0.01)
        with pytest.raises(ValueError, match="writer"):
            train.validate_state(model, opt, train.UPDATES, 11, arm)
    else:
        next(iter(opt.state.values()))["step"].sub_(1)
        with pytest.raises(ValueError, match="Adam"):
            train.validate_state(model, opt, train.UPDATES, 11, arm)


def test_partial_lossless_shard_and_missing_counts_rejection(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "REPORT", tmp_path / "train.json")
    record = dict(
        seed=11,
        arm="cut",
        protocol=deepcopy(train.PROTOCOL),
        completed_updates=1,
        complete=False,
        training_sufficient=False,
        losses=[dict(qa_sequences=2)],
        measurements=[dict(groups={})],
        rows=[dict(raw="kept")],
        milestones=[],
        restore_checks=[],
        signature_clone_updates=0,
        error="stopped",
        score_queries=1,
    )
    seal = train.seal_record(record)
    restored = train.shard.read_checked(train.Path(seal["shard"]["path"]), seal["shard"])
    assert restored["rows"] == record["rows"] and restored["phase"] == "partial"
    assert not seal["complete"]
    with pytest.raises(FileExistsError):
        train.seal_record(record)
    bad = dict(record, arm="position", complete=True)
    with pytest.raises(ValueError, match="counts"):
        train.seal_record(bad)


def test_fixed_schedule_no_queries_and_exclusive_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(
        train.runner.core.renaming, "train_item", lambda step, items, curriculum: (step, step)
    )
    for step in (0, 6143, 6144, 12288, 18432, 24575):
        assert train.scheduled(step, None, None) == (step % 6144, step % 6144)
    for step in (-1, 24576, True):
        with pytest.raises(ValueError):
            train.scheduled(step, None, None)
    monkeypatch.setattr(train, "REPORT", tmp_path / "existing.json")
    train.REPORT.write_text("sealed", encoding="utf-8")
    monkeypatch.setattr(train, "readiness", lambda: pytest.fail("existing batch must stop first"))
    with pytest.raises(FileExistsError):
        train.audit()
    assert train.PROTOCOL["total_updates"] == 221184
    assert train.PROTOCOL["cpu_limit_seconds"] == 14400


@pytest.mark.parametrize("seed,arm", [(11, "cut"), (12, "terminal"), (13, "position")])
def test_generic_scoring_six_views_with_exact_native_reproduction(monkeypatch, seed, arm):
    model, optimizer = train.engineering.fresh(seed)
    item = train.engineering.SYNTHETIC
    views = {
        v: [dict(item, id=f"synth-{i}", kind=train.paired.KINDS[i % 4]) for i in range(n)]
        for v, n in (("original", 192), ("pair", 96), ("rename", 192), ("rename_pair", 96))
    }

    def measure(model, item, measured_arm, session):
        assert measured_arm == arm
        labels = [*item["answer"].encode(), 256]
        return dict(
            id=item["id"],
            kind=item["kind"],
            target_symbols=labels,
            teacher_symbols=labels,
            teacher_tokens_correct=len(labels),
            teacher_exact=True,
            teacher_ce=0.0,
            raw_hex=item["answer"].encode().hex(),
            output_hex=item["answer"].encode().hex(),
            stop_reason="boundary",
            generated_steps=len(labels),
            free_exact=True,
        )

    monkeypatch.setattr(train.runner.core, "measure", measure)
    record = dict(rows=[], score_queries=0)
    before = train.paired.fingerprint(model.bundle(optimizer))
    rows, summary = train.evaluate(
        model, optimizer, seed, arm, 0, views, train.paired.CpuBudget(60), record
    )
    assert len(rows) == len(record["rows"]) == record["score_queries"] == 960
    assert all(r["seed"] == seed and r["arm"] == arm for r in rows)
    assert summary["factorial"]["original"]["joint_four_exact"] == 48
    assert train.paired.fingerprint(model.bundle(optimizer)) == before
