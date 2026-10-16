"""Artifact-free receipt failures cannot authorize a new confirmation face."""

from copy import deepcopy

import pytest

from scripts.training import _taiji_n1_fresh_training_receipt as receipt


def synthetic_model(seed=11, arm="position", sufficient=True):
    summary = dict(
        items=192,
        free_exact=192 if sufficient else 0,
        free_by_kind={k: 48 if sufficient else 0 for k in receipt.train.paired.KINDS},
        pair_exact=96,
        rename_free_exact=192,
        rename_free_by_kind={k: 48 for k in receipt.train.paired.KINDS},
        rename_pair_exact=96,
    )
    record = dict(
        seed=seed,
        arm=arm,
        protocol=receipt.train.PROTOCOL,
        phase="complete",
        complete=True,
        status="complete",
        completed_updates=24576,
        new_session_count=24576,
        signature_clone_updates=8,
        training_sufficient=sufficient,
        losses=[dict(qa_sequences=1)] * 12288 + [dict(qa_sequences=2)] * 12288,
        measurements=[{}] * 24576,
        rows=[{}] * 2880,
        milestones=[dict(summary, completed_updates=s) for s in receipt.train.MILESTONES],
        final_steps={
            **{f"writer.{i}": None if arm == "cut" else 24576 for i in range(4)},
            **{f"other.{i}": 24576 for i in range(37)},
        },
        restore_checks=[
            dict(path=f"s{seed}_{arm}_{phase}.pt", equal=True, signature_clone_updates=2)
            for phase in ("zero", "final")
        ],
    )
    model = {
        k: v
        for k, v in record.items()
        if k not in ("losses", "measurements", "rows", "status", "phase", "protocol")
    }
    return model, record


def synthetic_batch(monkeypatch):
    monkeypatch.setattr(receipt.train, "identity", lambda: {"synthetic": "receipt-test"})
    models, records = [], {}
    for seed in (11, 12, 13):
        for arm in ("position", "terminal", "cut"):
            model, record = synthetic_model(seed, arm, sufficient=arm != "terminal")
            models.append(model)
            records[seed, arm] = record
    data = dict(
        complete=True,
        status="complete",
        protocol=receipt.train.PROTOCOL,
        code_identity=receipt.train.identity(),
        models=models,
        completed_updates=221184,
        signature_clone_updates=72,
        synthetic_qa_sequences=144,
        cpu_seconds=8000,
        sources_unchanged=True,
        code_unchanged=True,
        cpu_accounting_complete=True,
        dev_queries=0,
        holdout_queries=0,
        confirmation_queries=0,
        training_sufficient=True,
    )
    return data, records


def test_complete_receipt_keeps_weak_control_and_no_model_loading(monkeypatch):
    data, records = synthetic_batch(monkeypatch)

    def forbid(*args, **kwargs):
        pytest.fail("receipt validation must not load models")

    monkeypatch.setattr(receipt.train.torch, "load", forbid)
    assert receipt.validate(data, read_record=lambda m: records[m["seed"], m["arm"]]) == dict(
        models=9,
        independent_restores=18,
        known_rows=25920,
        updates=221184,
        model_loads=0,
        confirmation_queries=0,
    )


@pytest.mark.parametrize(
    "change",
    [
        "missing_model",
        "duplicate_model",
        "partial",
        "queries",
        "budget",
        "accounting",
        "position_gate",
    ],
)
def test_incomplete_or_insufficient_batch_rejected(monkeypatch, change):
    data, records = synthetic_batch(monkeypatch)
    if change == "missing_model":
        data["models"].pop()
    elif change == "duplicate_model":
        data["models"][-1] = data["models"][0]
    elif change == "partial":
        data["complete"] = False
    elif change == "queries":
        data["confirmation_queries"] = 1
    elif change == "budget":
        data["cpu_seconds"] = 9001
    elif change == "accounting":
        data["cpu_accounting_complete"] = False
    else:
        model, record = synthetic_model(12, "position", False)
        data["models"][3] = model
        records[12, "position"] = record
    with pytest.raises(ValueError):
        receipt.validate(data, read_record=lambda m: records[m["seed"], m["arm"]])


@pytest.mark.parametrize("change", ["rows", "losses", "summary", "adam", "restore", "milestone"])
def test_each_full_shard_must_match_counts_and_main_summary(change):
    model, record = synthetic_model(11, "cut")
    model = deepcopy(model)
    if change == "rows":
        record["rows"].pop()
    elif change == "losses":
        record["losses"].pop()
    elif change == "summary":
        model["new_session_count"] -= 1
    elif change == "adam":
        record["final_steps"]["writer.0"] = 24576
        model["final_steps"]["writer.0"] = 24576
    elif change == "restore":
        record["restore_checks"][0]["equal"] = False
        model["restore_checks"][0]["equal"] = False
    else:
        record["milestones"][1]["completed_updates"] = 6143
        model["milestones"][1]["completed_updates"] = 6143
    with pytest.raises(ValueError):
        receipt.validate_model(model, record)
