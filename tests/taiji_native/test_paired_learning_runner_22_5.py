"""Input controls, full denominators and shadow ownership for CPU-endpoint learning."""

import pytest
import torch

from scripts.training.paired_compute_22_5 import shadow_optimizer
from scripts.training.run_paired_learning_22_5 import control_partition, evaluate
from taiji.u2_history import value_digest


def test_shadow_adam_cannot_mutate_live_parameters_or_moments():
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for parameter in model.parameters():
        optimizer.state[parameter] = {
            "step": torch.tensor(3.0),
            "exp_avg": torch.full_like(parameter, 0.1),
            "exp_avg_sq": torch.full_like(parameter, 0.2),
        }
    before = value_digest(model.state_dict()), value_digest(optimizer.state_dict())
    temporary, shadow = shadow_optimizer(model, optimizer)
    initial_shadow = [p.detach().clone() for p in temporary]
    for parameter in temporary:
        parameter.grad = torch.ones_like(parameter)
    shadow.step()
    assert any(not torch.equal(a, b) for a, b in zip(initial_shadow, temporary, strict=True))
    assert before == (value_digest(model.state_dict()), value_digest(optimizer.state_dict()))


def test_controls_preserve_targets_and_original_data():
    data = {
        "worlds": {
            "one": {
                "public": {
                    "observations": {"rgb": torch.ones(1), "audio": torch.ones(1)},
                    "coordinates": {"rgb": torch.ones(1), "audio": torch.ones(1)},
                }
            }
        },
        "episodes": [{"requests": (b"first", b"second"), "targets": (b"left", b"unknown")}],
    }
    before = value_digest(data)
    masked = control_partition(data, "without_rgb")
    assert "rgb" not in masked["worlds"]["one"]["public"]["observations"]
    assert "rgb" not in masked["worlds"]["one"]["public"]["coordinates"]
    neutral = control_partition(data, "without_questions")
    assert neutral["episodes"][0]["requests"] == ("请回答。".encode(),) * 2
    assert neutral["episodes"][0]["targets"] == data["episodes"][0]["targets"]
    assert value_digest(data) == before
    with pytest.raises(ValueError, match="unknown_paired_input_control"):
        control_partition(data, "other")


def test_evaluation_preserves_blocked_rounds_and_excludes_runtime_state(monkeypatch):
    from scripts.training import run_paired_learning_22_5 as runner

    def fake(*_args, **_kwargs):
        return {
            "rows": [
                {"correct": False, "status": "byte_budget_exhausted"},
                {"correct": False, "status": "blocked_by_prior_round"},
            ],
            "state": {"private_tensor": torch.ones(2)},
        }

    monkeypatch.setattr(runner, "rollout_episode", fake)
    calls = []
    result = evaluate(
        object(), {"episodes": [{"id": str(i)} for i in range(180)]}, lambda: calls.append(True)
    )
    assert result["denominator"] == 360 and result["correct"] == 0
    assert len(calls) == 180 and all("state" not in row for row in result["rows"])
