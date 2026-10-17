"""Train-only arm semantics, exact credit accounting and recoverable states."""

from copy import deepcopy

import pytest

from scripts.training import run_taiji_n1_relation_paired as runner
from tests.taiji_native._native_relation_fixtures import paired_sources


def test_all_arms_two_steps_and_full_restore(tmp_path):
    runner.torch.set_num_threads(1)
    ready, payload = paired_sources()
    item = dict(
        id="synthetic63",
        kind="lookup",
        split="train",
        observations=["alpha uses quartz.", "bravo uses copper."],
        question="material of alpha?",
        answer="quartz",
    )
    second = runner.core.renaming.second_question(item)
    for arm in runner.ARMS:
        model, optimizer = runner.fresh(payload, 11, ready)
        before = {
            n: (p.detach().clone(), deepcopy(optimizer.state[p]))
            for n, p in model.named_parameters()
            if n.startswith("writer.")
        }
        zero = tmp_path / f"{arm}-zero.pt"
        saved = runner.save(model, optimizer, zero, 11, arm, 0)
        assert runner.restore(zero)["signature"] == saved["expected_signature"]
        for step in (1, 2):
            loss = runner.train_step(model, optimizer, item, f"test63-{arm}-{step}", arm, second)
            assert loss["qa_sequences"] == 2
            for name, parameter in model.named_parameters():
                assert int(optimizer.state[parameter]["step"]) == runner.expected_step(
                    name, arm, step
                )
        if arm == "cut":
            for name, parameter in model.named_parameters():
                if name.startswith("writer."):
                    assert runner.torch.equal(parameter, before[name][0])
                    assert runner.paired.fingerprint(
                        optimizer.state[parameter]
                    ) == runner.paired.fingerprint(before[name][1])
        final = tmp_path / f"{arm}-final.pt"
        saved = runner.save(model, optimizer, final, 11, arm, 2)
        assert runner.restore(final)["signature"] == saved["expected_signature"]
        with pytest.raises(ValueError, match="train"):
            runner.train_step(
                model, optimizer, dict(item, split="holdout"), "forbidden", arm, second
            )


def test_four_gates_and_no_confirmation_entry():
    kinds = {k: 48 for k in ("lookup", "revision", "transition", "unknown")}
    summary = dict(
        free_exact=192,
        free_by_kind=kinds,
        pair_exact=96,
        rename_free_exact=192,
        rename_free_by_kind=kinds,
        rename_pair_exact=96,
    )
    assert runner.sufficient(summary)
    for field, value in (
        ("pair_exact", 86),
        ("rename_pair_exact", 86),
        ("free_exact", 172),
        ("rename_free_exact", 172),
    ):
        assert not runner.sufficient(dict(summary, **{field: value}))
    assert not runner.sufficient(dict(summary, free_by_kind={"lookup": 48}))
    assert runner.PROTOCOL["total_updates"] == 55296
    assert runner.PROTOCOL["total_qa_sequences"] == 82944
    assert runner.PROTOCOL["confirmation_queries"] == 0
    assert runner.PROTOCOL["cpu_limit_seconds"] == 2400


def test_evaluation_is_history_neutral_and_labels_control_arm():
    runner.torch.set_num_threads(1)
    ready, payload = paired_sources()
    model, optimizer = runner.fresh(payload, 12, ready)
    item = dict(
        id="synthetic-eval63",
        kind="lookup",
        split="train",
        observations=["alpha uses quartz.", "bravo uses copper."],
        question="material of alpha?",
        answer="quartz",
    )
    second = runner.core.renaming.second_question(item)
    before = runner.paired.fingerprint(model.bundle(optimizer))
    views = dict(original=[item], pair=[second], rename=[item], rename_pair=[second])
    rows, summary = runner.evaluate(
        model, 12, "terminal", 0, views, runner.paired.CpuBudget(2400, clock=lambda: 0)
    )
    assert len(rows) == 4 and all(r["seed"] == 12 and r["arm"] == "terminal" for r in rows)
    assert summary["items"] == 1
    assert all(
        summary["rename_free_by_kind"][k] == 0 for k in ("revision", "transition", "unknown")
    )
    assert before == runner.paired.fingerprint(model.bundle(optimizer))
