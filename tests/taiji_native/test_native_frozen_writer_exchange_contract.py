"""Synthetic writer intervention contracts; no frozen local assets required."""

import pytest

from scripts.training import audit_taiji_n1_frozen_writer_exchange as audit
from tests.taiji_native._native_relation_fixtures import paired_sources


def fresh():
    audit.torch.set_num_threads(1)
    ready, source = paired_sources()
    return audit.runner.fresh(source, 13, ready)


def test_only_four_parameters_change_and_exception_restores_full_parent():
    model, optimizer = fresh()
    before = audit.paired.fingerprint(model.bundle(optimizer))
    state = {n: p.detach().clone() for n, p in model.named_parameters()}
    original = audit.writer_state(state)
    donor = {n: v + 0.01 for n, v in original.items()}
    moments = audit.paired.fingerprint(optimizer.state_dict())
    with pytest.raises(RuntimeError, match="synthetic"), audit.exchange(model, optimizer, donor):
        for n, p in model.named_parameters():
            assert audit.torch.equal(p, donor[n] if n in donor else state[n])
        assert audit.paired.fingerprint(optimizer.state_dict()) == moments
        raise RuntimeError("synthetic interrupted diagnostic")
    assert audit.paired.fingerprint(model.bundle(optimizer)) == before
    assert audit.drift(original, donor)["l2"] > 0
    for bad in (
        {},
        dict(donor, extra=audit.torch.zeros(1)),
        {n: v.double() for n, v in donor.items()},
    ):
        with pytest.raises(ValueError, match="exact writer"), audit.exchange(model, optimizer, bad):
            raise AssertionError("invalid donor entered intervention")
    assert audit.paired.fingerprint(model.bundle(optimizer)) == before


def test_known_measurements_neutral_partial_rows_preserved_and_holdout_rejected():
    model, optimizer = fresh()
    before = audit.paired.fingerprint(model.bundle(optimizer))
    item = dict(audit.train.engineering.SYNTHETIC, id="synthetic72")
    second = audit.runner.core.renaming.second_question(item)
    views = dict(original=[item], pair=[second], rename=[item], rename_pair=[second])
    output = []
    rows, summary = audit.measure_condition(
        model,
        optimizer,
        13,
        "cut",
        "native",
        views,
        audit.paired.CpuBudget(180, clock=lambda: 0),
        output,
    )
    assert len(rows) == len(output) == 4
    assert all(r["condition"] == "native" for r in output)
    assert all(r["completed_updates"] == 6144 for r in rows)
    assert summary["items"] == 1
    assert audit.paired.fingerprint(model.bundle(optimizer)) == before
    output = []
    with pytest.raises(ValueError, match="holdout"):
        audit.measure_condition(
            model,
            optimizer,
            13,
            "cut",
            "mother_writer",
            dict(original=[item, dict(item, split="holdout", id="forbidden72")]),
            audit.paired.CpuBudget(180, clock=lambda: 0),
            output,
        )
    assert len(output) == 1
    assert audit.paired.fingerprint(model.bundle(optimizer)) == before
    assert audit.PROTOCOL["known_rows"] == 6912
    assert all(
        audit.PROTOCOL[k] == 0
        for k in (
            "parameter_updates",
            "optimizer_steps",
            "signature_clone_updates",
            "dev_queries",
            "holdout_queries",
            "confirmation_queries",
        )
    )
