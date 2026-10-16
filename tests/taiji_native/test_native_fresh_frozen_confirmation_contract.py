"""Synthetic confirmation contracts; never query the sealed face in unit tests."""

from copy import deepcopy

import pytest

from scripts.training import eval_taiji_n1_fresh_frozen_confirmation as evaluation


def rows(position=16, terminal=4, cut=12):
    return [
        dict(
            seed=seed,
            arm=arm,
            id=f"synthetic-{i}",
            kind=evaluation.paired.KINDS[i % 4],
            target_symbols=[97, 256],
            teacher_tokens_correct=2,
            teacher_exact=True,
            teacher_ce=0.0,
            free_exact=i < score,
        )
        for seed in evaluation.SEEDS
        for arm, score in (("position", position), ("terminal", terminal), ("cut", cut))
        for i in range(32)
    ]


def test_original14_gate_boundaries_and_complete_pairs():
    result = evaluation.adjudicate(rows())
    assert result["behavioral_pass"]
    assert result["aggregate_exact"] == dict(position=48, terminal=12, cut=36)
    assert not evaluation.adjudicate(rows(cut=13))["gates"]["advantage_cut"]
    assert not evaluation.adjudicate(rows(terminal=13))["gates"]["advantage_terminal"]
    assert not evaluation.adjudicate(rows(position=15, cut=0))["gates"]["each_position_seed"]
    bad = rows(position=24, cut=0)
    for row in bad:
        if (
            row["seed"] == 11
            and row["arm"] == "position"
            and row["kind"] == evaluation.paired.KINDS[0]
        ):
            row["free_exact"] = False
    assert not evaluation.adjudicate(bad)["gates"]["each_position_kind"]
    with pytest.raises(ValueError, match="288"):
        evaluation.adjudicate(rows()[:-1])
    bad = rows()
    bad[-1] = deepcopy(bad[-2])
    with pytest.raises(ValueError, match="unique"):
        evaluation.adjudicate(bad)


def test_explicit_holdout_path_is_full_state_neutral_and_partial_rows_survive(monkeypatch):
    evaluation.torch.set_num_threads(1)
    model, optimizer = evaluation.train.engineering.fresh(11)
    item = dict(evaluation.train.engineering.SYNTHETIC, id="synthetic71", split="holdout")
    before = evaluation.paired.fingerprint(model.bundle(optimizer))
    measured = evaluation.evaluate(
        model, optimizer, 11, "cut", [item], evaluation.paired.CpuBudget(180, clock=lambda: 0)
    )
    assert measured[0]["split"] == "holdout"
    assert measured[0]["free_exact"] == (
        measured[0]["raw_hex"] == item["answer"].encode().hex()
        and measured[0]["stop_reason"] == "boundary"
    )
    assert evaluation.paired.fingerprint(model.bundle(optimizer)) == before
    for split in ("train", "dev"):
        with pytest.raises(ValueError, match="holdout"):
            evaluation.measure(model, dict(item, split=split), "cut", "rejected71")
    original = evaluation.measure
    calls = []

    def fail_second(*args):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("synthetic interrupted evaluation")
        return original(*args)

    monkeypatch.setattr(evaluation, "measure", fail_second)
    retained = []
    accounting = dict(confirmation_queries=0)
    with pytest.raises(RuntimeError, match="interrupted"):
        evaluation.evaluate(
            model,
            optimizer,
            11,
            "cut",
            [item, dict(item, id="second71")],
            evaluation.paired.CpuBudget(180, clock=lambda: 0),
            output=retained,
            accounting=accounting,
        )
    assert len(retained) == 1 and accounting["confirmation_queries"] == 2
    assert evaluation.paired.fingerprint(model.bundle(optimizer)) == before


def test_attempt_prevents_any_retry_before_readiness(tmp_path, monkeypatch):
    attempt = tmp_path / "attempt.json"
    evaluation.train.engineering.train.write_new(attempt, dict(face_consumed=True))
    monkeypatch.setattr(evaluation, "ATTEMPT", attempt)
    monkeypatch.setattr(evaluation, "REPORT", tmp_path / "result.json")

    def forbidden():
        raise AssertionError("retry reached readiness")

    monkeypatch.setattr(evaluation, "readiness", forbidden)
    with pytest.raises(FileExistsError, match="consumed"):
        evaluation.audit()
    assert evaluation.PROTOCOL["parameter_updates"] == 0
    assert evaluation.PROTOCOL["signature_clone_updates"] == 0
    assert evaluation.PROTOCOL["confirmation_queries"] == 288
    assert evaluation.PROTOCOL["cpu_limit_seconds"] == 180


def test_unbound_terminal_sha_refuses_before_model_or_face_reads(monkeypatch):
    monkeypatch.setattr(evaluation, "TRAIN_SHA", None)

    def forbidden(*args, **kwargs):
        pytest.fail("unbound receipt reached model load or hash reads")

    monkeypatch.setattr(evaluation.paired, "sha", forbidden)
    monkeypatch.setattr(evaluation.torch, "load", forbidden)
    with pytest.raises(ValueError, match="not yet bound"):
        evaluation.readiness()


def test_receipts_are_exact_lf_and_exclusive(tmp_path):
    target = tmp_path / "receipt.json"
    evaluation.write_new(target, dict(synthetic=True))
    assert target.read_bytes() == b'{"synthetic":true}\n'
    with pytest.raises(FileExistsError):
        evaluation.write_new(target, dict(synthetic=False))
    assert target.read_bytes() == b'{"synthetic":true}\n'
