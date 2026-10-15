"""Safety and evidence gates for the approved PLAN14 experiment."""

from copy import deepcopy

import pytest
import torch

from scripts.training import run_taiji_n1_native_evidence_paired as runner
from taiji.native_evidence_candidate import NativeEvidenceCandidate


def test_inference_projection_drops_gold_and_evaluator_fields():
    class Poison:
        def __str__(self):
            raise AssertionError("metadata reached inference")

    item = {
        "observations": ["new fact"],
        "question": "which?",
        "answer": Poison(),
        "kind": Poison(),
        "id": Poison(),
        "split": Poison(),
    }
    assert runner.project(item) == runner.EpisodeInput(("new fact",), b"which?")


def test_execution_requires_approval_before_reading_or_writing(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("execution started without approval")

    monkeypatch.setattr(runner, "checked_manifest", forbidden)
    monkeypatch.setattr(runner, "get_external_path", forbidden)
    with pytest.raises(PermissionError):
        runner.run()


def test_budget_counts_children_and_reserves_checkpoint_time():
    budget = runner.CpuBudget(limit=10, clock=lambda: 3)
    budget.children = 4
    assert budget.used() == 7
    budget.guard()
    with pytest.raises(runner.BudgetStop):
        budget.guard(active_child=1)


def test_training_refuses_dev_before_touching_model():
    with pytest.raises(ValueError, match="only train"):
        runner.train_step(None, None, {"split": "dev"}, "x", "position")


def test_checkpoint_uses_live_parent_signature_and_optimizer(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(15)
    model = NativeEvidenceCandidate()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    item = {
        "observations": ["agent uses copper"],
        "question": "material?",
        "answer": "copper",
        "split": "train",
    }
    runner.train_step(model, optimizer, item, "one", "position")
    path = tmp_path / "model.pt"
    payload = runner.checkpoint(model, optimizer, path, 11, "position", 1, "one", item)
    loaded = torch.load(path, weights_only=True)
    assert runner.restore_signature(loaded) == payload["expected_signature"]
    assert payload["candidate"]["optimizer"]["state"]
    damaged = deepcopy(loaded)
    damaged["candidate"]["state"]["readout.bias"][0] += 1
    assert runner.restore_signature(damaged) != payload["expected_signature"]
    with pytest.raises(FileExistsError):
        runner.checkpoint(model, optimizer, path, 11, "position", 1, "one", item)


def test_verdict_requires_full_surface_all_seeds_and_both_controls():
    rows = []
    for seed in runner.SEEDS:
        for arm in runner.ARMS:
            for split in ("dev", "holdout"):
                for i in range(32):
                    rows.append(
                        {
                            "seed": seed,
                            "arm": arm,
                            "split": split,
                            "id": f"{split}-{i:03d}",
                            "kind": runner.KINDS[i % 4],
                            "exact": arm == "position" and i < 16,
                        }
                    )
    assert runner.verdict(rows, 3456)["can_advance"]
    assert not runner.verdict(rows[:-1], 3456)["can_advance"]
    assert not runner.verdict(rows, 3455)["can_advance"]
    with pytest.raises(ValueError, match="duplicate"):
        runner.verdict(rows + rows[:1], 3456)
    tied = [
        dict(row, exact=row["exact"] or (row["arm"] == "cut" and int(row["id"][-3:]) < 16))
        for row in rows
    ]
    assert not runner.verdict(tied, 3456)["can_advance"]
