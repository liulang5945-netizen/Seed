"""Synthetic factorial algebra, exact reproduction and state-neutral failure paths."""

from copy import deepcopy

import pytest

from scripts.training import diagnose_taiji_n1_fresh_transition_factorial as d


def parent(index=0):
    direction = d.DIRECTIONS[index % 4]
    turn = "left" if index % 8 < 4 else "right"
    return dict(
        id=f"synthetic-{index}",
        split="train",
        kind="transition",
        observations=[f"entity faces {direction}.", f"entity turns {turn}."],
        question="direction of entity?",
        answer=d.DIRECTIONS[(index % 4 + (1 if turn == "right" else -1)) % 4],
    )


def test_factorial_algebra_preserves_parent_and_inference_metadata():
    item = parent()
    saved = deepcopy(item)
    entries = d.variants(item)
    assert item == saved and entries[0]["item"] == item
    answers = [e["item"]["answer"] for e in entries]
    assert answers[0] == answers[3] and answers[1] == answers[2]
    assert (d.DIRECTIONS.index(answers[1]) - d.DIRECTIONS.index(answers[0])) % 4 == 2
    for entry in entries:
        assert entry["item"]["id"] == item["id"]
        assert "condition" not in entry["item"]
        payload = d.train.paired.project(entry["item"])
        assert payload.question == item["question"].encode()
    assert len(d.build([parent(i) for i in range(48)])["entries"]) == 192
    assert d.build([parent(i) for i in range(48)]) == d.build([parent(i) for i in range(48)])


@pytest.mark.parametrize(
    "change",
    [
        dict(split="holdout"),
        dict(kind="lookup"),
        dict(answer="wrong"),
        dict(observations=["entity faces north.", "other turns left."]),
    ],
)
def test_invalid_parents_rejected(change):
    with pytest.raises(ValueError):
        d.variants(dict(parent(), **change))


def synthetic_rows():
    return [
        dict(
            seed=seed,
            arm=arm,
            **{k: v for k, v in entry.items() if k != "item"},
            raw_hex="ff",
            stop_reason="boundary",
            free_exact=entry["condition"] == "original",
            teacher_exact=False,
        )
        for seed, arm in d.MODELS
        for i in range(48)
        for entry in d.variants(parent(i))
    ]


def test_invariance_is_not_composition():
    summaries = d.summarize(synthetic_rows())
    for summary in summaries:
        assert summary["joint_four_exact"] == 0
        assert summary["double_flip_unchanged"] == 48
        assert summary["single_flip_changed"] == dict(turn_flip=0, direction_opposite=0)
        assert summary["conditions"]["original"]["free_exact"] == 48
        assert sum(c["items"] for c in summary["conditions"]["both"]["cells"].values()) == 48


@pytest.mark.parametrize("defect", ["missing", "duplicate", "parent", "condition", "case"])
def test_incomplete_or_misaligned_summary_rejected(defect):
    rows = synthetic_rows()
    if defect == "missing":
        rows.pop()
    elif defect == "duplicate":
        rows[-1] = rows[-2]
    elif defect == "parent":
        rows[-1]["parent_id"] = "other"
    elif defect == "condition":
        rows[-1]["condition"] = "original"
    else:
        rows[-1]["case_id"] = "different"
    with pytest.raises(ValueError):
        d.summarize(rows)


@pytest.mark.parametrize("failure", [None, "mismatch", "second"])
def test_real_synthetic_model_queries_state_neutral_even_on_failure(monkeypatch, failure):
    d.torch.set_num_threads(1)
    model, optimizer = d.train.engineering.fresh(11)
    item = parent()
    native = d.train.runner.core.measure(model, item, "position", "native-contract")
    model._observations.pop("native-contract")
    if failure == "mismatch":
        native = dict(native, raw_hex="different")
    original_measure = d.train.runner.core.measure
    calls = []

    def measure(*args):
        calls.append(1)
        if failure == "second" and len(calls) == 2:
            raise RuntimeError("synthetic measurement failure")
        return original_measure(*args)

    monkeypatch.setattr(d.train.runner.core, "measure", measure)
    before = d.train.paired.fingerprint(model.bundle(optimizer))
    result = dict(rows=[], model_queries=0, native_reproductions=0)

    def run():
        d.evaluate(
            model,
            optimizer,
            11,
            "position",
            d.variants(item),
            {item["id"]: native},
            d.train.paired.CpuBudget(60),
            result,
        )

    if failure:
        with pytest.raises((ValueError, RuntimeError)):
            run()
        assert len(result["rows"]) == 1
        assert result["model_queries"] == (2 if failure == "second" else 1)
    else:
        run()
        assert len(result["rows"]) == result["model_queries"] == 4
        assert result["native_reproductions"] == 1
    assert d.train.paired.fingerprint(model.bundle(optimizer)) == before
    assert not model._observations


def test_exclusive_lf_writer_and_attempt_guard(tmp_path, monkeypatch):
    path = tmp_path / "attempt.json"
    d.write_new(path, dict(value="test"))
    assert b"\r" not in path.read_bytes()
    with pytest.raises(FileExistsError):
        d.write_new(path, {})
    monkeypatch.setattr(d, "ATTEMPT", path)
    monkeypatch.setattr(d, "REPORT", tmp_path / "report.json")
    monkeypatch.setattr(d, "readiness", lambda: pytest.fail("must refuse before source reads"))
    with pytest.raises(FileExistsError):
        d.execute()
