"""Frozen novel-name schedule, semantic consistency and full Adam continuation."""

import json

import pytest
import torch

from scripts.training import run_taiji_n1_renaming as runner


def test_frozen_schedule_names_and_pair_semantics():
    train = runner.checked_train()
    curriculum = runner.checked_curriculum(train)
    assert len(curriculum["stream"]) == 3072 and len(curriculum["probes"]) == 192
    names = [n for e in curriculum["stream"] + curriculum["probes"] for n in e["names"]]
    assert len(names) == len(set(names)) == 9792
    pool = {
        n
        for e in json.loads(runner.calibration.MANIFEST.read_text())["episodes"]
        for n in e["entities"]
    }
    assert not pool & set(names)
    pairs = {p["parent_id"]: p for p in runner.checked_pairs(train)}
    for item in train:
        assert runner.renaming.second_question(item) == pairs.get(item["id"])
    for step in (0, 1, 382, 383, 384, 385, 6142, 6143):
        item, pair = runner.renaming.train_item(step, train, curriculum)
        old = train[(step // 2) % 192]
        assert item["id"] == old["id"] and item["answer"] == old["answer"]
        assert len(item["question"]) == len(old["question"])
        assert list(map(len, item["observations"])) == list(map(len, old["observations"]))
        if step % 2:
            for name in old["entities"]:
                assert name not in item["question"] and all(
                    name not in s for s in item["observations"]
                )
        else:
            assert item == old
        if pair is not None:
            assert pair["observations"] == item["observations"]
            assert pair["answer"] != item["answer"]
    assert (
        sum(
            1 if train[(s // 2) % 192]["kind"] not in ("lookup", "unknown") else 2
            for s in range(6144)
        )
        == 9216
    )
    with pytest.raises(ValueError, match="6144"):
        runner.renaming.train_item(6144, train, curriculum)
    with pytest.raises(ValueError, match="train"):
        runner.renaming.rename_item(dict(train[0], split="dev"), curriculum["probes"][0]["names"])


def test_complete_trained_adam_restore_and_initial_controls(tmp_path):
    torch.set_num_threads(1)
    train = runner.checked_train()
    model, optimizer = runner.fresh(train)
    record, payload, _ = runner.transfer_source()
    assert optimizer.state and all(int(v["step"]) == 6144 for v in optimizer.state.values())
    assert runner.pretrained.signature(model, optimizer) == payload["expected_signature"]
    saved = runner.save(model, optimizer, tmp_path / "zero.pt", 0)
    assert runner.restore(tmp_path / "zero.pt")["signature"] == saved["expected_signature"]
    assert runner.code_identity()["protocol"]["curriculum_sha256"] == runner.CURRICULUM_SHA
    for subset in ("original", "pair"):
        rows = [
            dict(r)
            for r in record["rows"]
            if r["completed_updates"] == 6144 and r["evaluation_set"] == subset
        ]
        assert runner.transfer_equal(rows, record, subset)
        rows[0]["raw_hex"] = "00"
        assert not runner.transfer_equal(rows, record, subset)


def test_new_batch_sessions_do_not_collide_with_restored_history():
    model, _ = runner.fresh(runner.checked_train())
    names = [
        runner.session_name(view, step, f"train-{i:03d}")
        for view in ("original", "pair", "rename", "rename_pair")
        for step in (0, 3072, 6144)
        for i in range(192)
    ]
    assert len(names) == len(set(names))
    assert all(name not in model._observations for name in names)
