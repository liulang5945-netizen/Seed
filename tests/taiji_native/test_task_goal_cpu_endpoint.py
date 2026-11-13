"""The CPU endpoint restores parameters, optimizer, cursor and real-step accounting."""

import json

import pytest
import torch

from scripts.training import run_task_goal_learning_22_5 as runner
from taiji.u2_history import assert_values_equal


def test_one_overbudget_update_is_saved_and_then_fully_rolled_back(tmp_path, monkeypatch):
    clock = [0.0]
    loaded = []

    def load(path):
        payload = torch.load(path, weights_only=False)
        model = torch.nn.Linear(1, 1, bias=False)
        model.load_state_dict(payload["parameters"])
        model.parameter_revision = payload["revision"]
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        optimizer.load_state_dict(payload["optimizer"])
        loaded.append(payload)
        return model, optimizer, payload["metadata"]

    def save(path, model, optimizer, architecture, lineage, **metadata):
        torch.save(
            {
                "parameters": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "revision": model.parameter_revision,
                "metadata": {**metadata, "lineage": lineage},
            },
            path,
        )

    def loss(model, partition, episode, history):
        clock[0] += 0.6
        return {"loss": (model.weight - 1).square().sum(), "first_ended": True}

    initial = tmp_path / "initial.bin"
    model = torch.nn.Linear(1, 1, bias=False)
    torch.nn.init.zeros_(model.weight)
    model.parameter_revision = 7
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    save(initial, model, optimizer, "toy", "parent", update=0, cursor=0)
    monkeypatch.setattr(runner, "load_learner", load)
    monkeypatch.setattr(runner, "save_learner", save)
    monkeypatch.setattr(runner, "history_loss", loss)
    monkeypatch.setattr(runner, "sources", lambda: {"test": "fixed"})
    monkeypatch.setattr(runner.time, "process_time", lambda: clock[0])
    counters = {"optimizer_steps": 0}
    restored, summary = runner.train_group(
        "A",
        {
            "initial": str(initial),
            "initial_sha256": runner.digest(initial),
            "history": "teacher",
            "architecture": "toy",
        },
        {"order": [0] * 180, "data": {"data_digest": "fixture"}},
        {
            "max_updates": 10,
            "cpu_per_group": 1.0,
            "max_single_update_cpu": 2.0,
            "source_pins": {"test": "fixed"},
        },
        {"train": {"episodes": [{}]}},
        tmp_path,
        lambda: None,
        counters,
    )
    assert counters["optimizer_steps"] == summary["model_optimizer_steps"] == 2
    assert summary["selected_updates"] == 1
    assert summary["selected_training_cpu"] == pytest.approx(0.6)
    assert summary["spent_training_cpu"] == pytest.approx(1.2)
    selected = torch.load(summary["checkpoint"], weights_only=False)
    rejected = torch.load(summary["overshoot"]["checkpoint"], weights_only=False)
    assert restored.parameter_revision == selected["revision"] == 8
    assert rejected["revision"] == 9
    assert_values_equal(restored.state_dict(), selected["parameters"])
    assert_values_equal(loaded[-1]["optimizer"], selected["optimizer"])
    assert loaded[-1]["metadata"]["update"] == loaded[-1]["metadata"]["cursor"] == 1
    rows = [json.loads(line) for line in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert [row["status"] for row in rows] == ["opened", "accepted", "opened", "quarantined"]
    assert not torch.equal(selected["parameters"]["weight"], rejected["parameters"]["weight"])
