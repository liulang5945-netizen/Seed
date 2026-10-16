"""Artifact-free fixed schedule, strict six gates and failed-score evidence."""

from copy import deepcopy

import pytest

from scripts.training import run_taiji_n1_transition_factorial_calibration as calibration


def test_full_schedule_counts_and_out_of_range_rejection():
    items = [
        dict(id=str(i), kind=("lookup", "revision", "transition", "unknown")[i % 4])
        for i in range(192)
    ]
    qa = sessions = 0
    for step in range(calibration.UPDATES):
        kind = items[(step % 6144 // 2) % 192]["kind"]
        qa += 4 if kind == "transition" else 2 if kind in ("lookup", "unknown") else 1
        sessions += 4 if kind == "transition" else 1
        assert calibration.expected_sessions(step + 1) == sessions
    assert qa == 55296 and sessions == 43008
    for bad in (-1, 24577, True, 1.5):
        with pytest.raises(ValueError):
            calibration.expected_sessions(bad)
        with pytest.raises(ValueError):
            calibration.scheduled(bad, [], {})


def summary():
    return dict(
        free_exact=192,
        free_by_kind={k: 48 for k in calibration.paired.KINDS},
        pair_exact=96,
        rename_free_exact=192,
        rename_free_by_kind={k: 48 for k in calibration.paired.KINDS},
        rename_pair_exact=96,
        factorial={v: dict(joint_four_exact=44) for v in ("original", "rename")},
    )


@pytest.mark.parametrize(
    "failure",
    [None, "old", "pair", "rename", "rename_pair", "factorial_original", "factorial_rename"],
)
def test_six_gates_conjunctive(failure):
    value = summary()
    if failure == "old":
        value["free_by_kind"]["transition"] = 39
    elif failure == "pair":
        value["pair_exact"] = 86
    elif failure == "rename":
        value["rename_free_by_kind"]["transition"] = 39
    elif failure == "rename_pair":
        value["rename_pair_exact"] = 86
    elif failure:
        value["factorial"][failure.removeprefix("factorial_")]["joint_four_exact"] = 43
    assert calibration.sufficient(value) == (failure is None)


def test_quartet_denominators_and_joint_requirement():
    rows = [
        dict(parent_id=str(i), condition=c, free_exact=c != "both")
        for i in range(48)
        for c in calibration.engineering.factorial.CONDITIONS
    ]
    assert calibration.quartet_summary(rows)["joint_four_exact"] == 0
    rows[-1]["parent_id"] = "other"
    with pytest.raises(ValueError):
        calibration.quartet_summary(rows)
    with pytest.raises(ValueError):
        calibration.quartet_summary(rows[:-1])


def test_zero_checkpoint_two_new_objective_steps_roundtrip(tmp_path):
    calibration.torch.set_num_threads(1)
    model, optimizer = calibration.engineering.engineering.fresh(13)
    before = calibration.paired.fingerprint(model.bundle(optimizer))
    path = tmp_path / "zero.pt"
    payload = calibration.save(model, optimizer, path, 0)
    assert calibration.paired.fingerprint(model.bundle(optimizer)) == before
    assert calibration.restore(path)["signature"] == payload["expected_signature"]
    wrong = deepcopy(payload)
    wrong["seed"] = 11
    with pytest.raises(ValueError):
        calibration.load(wrong)
    with pytest.raises(ValueError):
        calibration.save(model, optimizer, tmp_path / "partial.pt", 1)


def test_failed_scoring_retains_first_row_and_attempt_count(monkeypatch):
    calibration.torch.set_num_threads(1)
    model, optimizer = calibration.engineering.engineering.fresh(13)
    item = calibration.engineering.SYNTHETIC
    calls = []
    original = calibration.runner.core.measure

    def measure(*args):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("synthetic score failure")
        return original(*args)

    monkeypatch.setattr(calibration.runner.core, "measure", measure)
    result = dict(rows=[], score_queries=0)
    before = calibration.paired.fingerprint(model.bundle(optimizer))
    with pytest.raises(RuntimeError):
        calibration.evaluate(
            model,
            optimizer,
            0,
            dict(original=[item, item]),
            calibration.paired.CpuBudget(60),
            result,
        )
    assert result["score_queries"] == 2 and len(result["rows"]) == 1
    assert calibration.paired.fingerprint(model.bundle(optimizer)) == before


def test_attempt_blocks_before_source_reads(tmp_path, monkeypatch):
    attempt = tmp_path / "attempt.json"
    attempt.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(calibration, "ATTEMPT", attempt)
    monkeypatch.setattr(calibration, "readiness", lambda: pytest.fail("retry must fail first"))
    with pytest.raises(FileExistsError):
        calibration.audit()


def test_complete_score_assembly_reproduces_native_rows_without_artifacts(monkeypatch):
    model, optimizer = calibration.engineering.engineering.fresh(13)
    items = [
        dict(
            calibration.engineering.SYNTHETIC,
            id=f"synthetic-{i}",
            kind=calibration.paired.KINDS[i % 4],
        )
        for i in range(192)
    ]
    pairs = [dict(item) for item in items if item["kind"] in ("lookup", "unknown")]

    def measure(model, item, arm, session):
        return dict(
            id=item["id"],
            kind=item["kind"],
            target_symbols=[1],
            teacher_symbols=[1],
            teacher_tokens_correct=1,
            teacher_exact=True,
            teacher_ce=0.0,
            raw_hex="00",
            output_hex="00",
            stop_reason="boundary",
            generated_steps=1,
            free_exact=True,
        )

    monkeypatch.setattr(calibration.runner.core, "measure", measure)
    evidence = dict(rows=[], score_queries=0)
    rows, value = calibration.evaluate(
        model,
        optimizer,
        0,
        dict(original=items, pair=pairs, rename=items, rename_pair=pairs),
        calibration.paired.CpuBudget(60),
        evidence,
    )
    assert len(rows) == len(evidence["rows"]) == evidence["score_queries"] == 960
    assert calibration.sufficient(value)
    assert all(value["factorial"][v]["joint_four_exact"] == 48 for v in ("original", "rename"))
