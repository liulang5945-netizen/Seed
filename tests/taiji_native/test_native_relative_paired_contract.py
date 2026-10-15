"""The first heldout read requires every model, sufficient training and restores."""

from copy import deepcopy

import pytest
import torch

from scripts.training import run_taiji_n1_relative_paired as runner
from scripts.training import run_taiji_n1_relative_read as pilot


def readiness():
    training, rows, restores = [], [], []
    for seed in runner.paired.SEEDS:
        for arm in runner.paired.ARMS:
            training.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "updates": 6144,
                    "train_summary": {
                        "free_exact": 180,
                        "free_by_kind": {k: 45 for k in runner.paired.KINDS},
                    },
                }
            )
            rows.extend({"seed": seed, "arm": arm, "id": f"train-{i:03d}"} for i in range(192))
            restores.extend(
                {"seed": seed, "arm": arm, "phase": p, "equal": True} for p in ("zero", "final")
            )
    return training, rows, restores


def test_all_seeds_arms_rows_and_restores_before_any_holdout():
    assert not {"seed", "arm", "updates", "milestones", "holdout_queries"} & runner.PROTOCOL.keys()
    assert runner.PROTOCOL["holdout_rows_if_sufficient"] == 288
    training, rows, restores = readiness()
    assert runner.training_ready(training, rows, restores)
    assert not runner.training_ready(training[:-1], rows, restores)
    assert not runner.training_ready(training, rows[:-1] + [rows[0]], restores)
    assert not runner.training_ready(training, rows, restores[:-1] + [restores[0]])
    training[0]["train_summary"]["free_exact"] = 172
    assert not runner.training_ready(training, rows, restores)
    with pytest.raises(ValueError, match="heldout query"):
        runner.measure(None, {"split": "holdout"}, "position", "forbidden")
    with pytest.raises(ValueError, match="dev forbidden"):
        runner.measure(None, {"split": "dev"}, "position", "forbidden", holdout_ready=True)
    with pytest.raises(ValueError, match="train"):
        runner.train_step(None, None, {"split": "holdout"}, "forbidden", "position")


def test_behavior_gate_keeps_each_seed_kind_and_both_control_gaps():
    rows = []
    for seed in runner.paired.SEEDS:
        for arm in runner.paired.ARMS:
            for i in range(32):
                rows.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "id": f"holdout-{i:03d}",
                        "kind": runner.paired.KINDS[i % 4],
                        "free_exact": i < (20 if arm == "position" else 16),
                    }
                )
    assert runner.verdict(rows)["can_advance"]
    assert not runner.verdict(rows[:-1] + [rows[0]])["can_advance"]
    too_strong = deepcopy(rows)
    for row in too_strong:
        if row["arm"] == "cut":
            row["free_exact"] = int(row["id"].split("-")[-1]) < 17
    assert not runner.verdict(too_strong)["can_advance"]
    bad_kind = deepcopy(rows)
    for row in bad_kind:
        if row["seed"] == 11 and row["arm"] == "position" and row["kind"] == runner.paired.KINDS[0]:
            row["free_exact"] = False
    assert not runner.verdict(bad_kind)["can_advance"]


def test_paired_checkpoint_optimizer_continuation_and_protocol_separation(tmp_path):
    torch.set_num_threads(1)
    item = {
        "split": "train",
        "observations": ["entity uses abcdef."],
        "question": "material?",
        "answer": "abcdef",
    }
    model, optimizer = runner.fresh(11, [item])
    runner.train_step(model, optimizer, item, "first", "position")
    path = tmp_path / "paired.pt"
    saved = runner.save(model, optimizer, path, 11, "position", 1)
    assert runner.restore(path)["signature"] == saved["expected_signature"]
    with pytest.raises(ValueError, match="protocol"):
        pilot.restore(path)
    restored = runner.RelativeEvidenceCandidate.restore_bundle(saved["candidate"])
    other_optimizer = torch.optim.Adam(restored.parameters(), lr=0.001)
    other_optimizer.load_state_dict(saved["candidate"]["optimizer"])
    runner.train_step(model, optimizer, item, "next", "position")
    runner.train_step(restored, other_optimizer, item, "next", "position")
    assert all(
        torch.equal(value, restored.state_dict()[key]) for key, value in model.state_dict().items()
    )
