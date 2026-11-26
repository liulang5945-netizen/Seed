"""Measure first-answer world-by-request interaction through the frozen full graph."""

# ruff: noqa: E402 -- include startup in the diagnostic cost
from __future__ import annotations

import copy
import hashlib
import json
import sys
import time
from pathlib import Path

STARTED = time.monotonic()
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training.check_report_run_conditions import write_report
from scripts.training.paired_episode_22_5 import start_episode
from scripts.training.task_goal_learner_22_5 import load_learner
from scripts.training.task_history_loss_22_5 import score_answer
from scripts.training.verify_taiji_u2_metadata_groups_22_4 import StepGuard
from taiji.u2_history import atomic_load, value_digest

CONTRACT = ROOT / "plans/manifests/taiji_u2_first_binding_contract_20261113.json"
REPORT = ROOT / "reports/taiji_u2_first_binding_20261113.json"


def interaction(values):
    """Rows are worlds and columns are requests, with matching on the diagonal."""
    return values[0, 0] - values[0, 1] - values[1, 0] + values[1, 1]


def select_cells(partition, position, colors):
    episodes = partition["episodes"]
    cells = []
    targets = None
    for visible in colors:
        present_first = next(
            episode
            for episode in episodes
            if episode["audit"]["visible"] == visible
            and episode["audit"]["position"] == position
            and episode["audit"]["absent"] == next(color for color in colors if color != visible)
            and episode["targets"][0] != episode["targets"][1]
            and not episode["audit"]["reverse"]
        )
        absent_first = next(
            episode
            for episode in episodes
            if episode["world_id"] == present_first["world_id"]
            and episode["requests"][0] == present_first["requests"][1]
            and episode["requests"][1] == present_first["requests"][0]
        )
        if targets is None:
            targets = present_first["targets"]
        assert targets == present_first["targets"]
        cells.append(
            (present_first, absent_first) if visible == colors[0] else (absent_first, present_first)
        )
    assert cells[0][0]["requests"][0] == cells[1][0]["requests"][0]
    assert cells[0][1]["requests"][0] == cells[1][1]["requests"][0]
    assert targets is not None and all(len(target) == 6 for target in targets)
    return cells, targets


def probe():
    if REPORT.exists():
        raise FileExistsError(REPORT)
    torch.set_num_threads(1)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == contract["source_sha256"]
    evidence = ROOT / "reports/taiji_u2_task_goal_learning_20261113.json"
    assert hashlib.sha256(evidence.read_bytes()).hexdigest() == contract["learning_report_sha256"]
    previous = json.loads(evidence.read_text(encoding="utf-8"))
    data = atomic_load(contract["data_path"])
    assert value_digest(data) == contract["data_digest"]
    partition = data["train"]
    reports = {}
    with StepGuard() as guard:
        for name, precision in contract["profiles"]:
            checkpoint = previous["results"][name]["checkpoint"]
            assert (
                hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
                == contract["checkpoint_sha256"][name]
            )
            model, optimizer, _ = load_learner(previous["results"][name]["checkpoint"])
            assert value_digest(model.state_dict()) == previous["results"][name]["parameter_digest"]
            dtype = getattr(torch, precision)
            model = model.to(dtype=dtype)
            current_partition = copy.deepcopy(partition)
            for world in current_partition["worlds"].values():
                public = world["public"]
                for key in ("observations", "coordinates"):
                    public[key] = {key: value.to(dtype=dtype) for key, value in public[key].items()}
                public["outcome"] = public["outcome"].to(dtype=dtype)
                public["action"] = public["action"].to(dtype=dtype)
            parameter_digest = value_digest(model.state_dict())
            optimizer_digest = value_digest(optimizer.state_dict())
            rows = []
            for position in contract["positions"]:
                for colors in contract["color_pairs"]:
                    if (
                        time.process_time() > contract["cpu_cap"]
                        or time.monotonic() - STARTED > contract["wall_cap"]
                    ):
                        raise RuntimeError("first_binding_budget_stop")
                    cells, targets = select_cells(current_partition, position, colors)
                    scores = []
                    for cell_row in cells:
                        row = []
                        for episode in cell_row:
                            state = start_episode(model, current_partition, episode)
                            positive, _ = score_answer(model, state, targets[0])
                            negative, _ = score_answer(model, state, targets[1])
                            row.append(7 * (negative - positive))
                        scores.append(torch.stack(row))
                    values = torch.stack(scores)
                    delta = interaction(values)
                    named = list(model.named_parameters())
                    gradients = torch.autograd.grad(
                        delta, [parameter for _, parameter in named], allow_unused=True
                    )
                    rows.append(
                        {
                            "position": position,
                            "colors": colors,
                            "sequence_log_odds": values.detach().tolist(),
                            "interaction": float(delta.detach()),
                            "parameter_gradient_norms": {
                                key: None if value is None else float(value.norm())
                                for (key, _), value in zip(named, gradients, strict=True)
                            },
                        }
                    )
            assert value_digest(model.state_dict()) == parameter_digest
            assert value_digest(optimizer.state_dict()) == optimizer_digest
            reports[f"{name}-{precision}"] = {
                "rows": rows,
                "positive_interactions": sum(row["interaction"] > 0 for row in rows),
                "parameter_digest": parameter_digest,
            }
    assert (
        time.process_time() < contract["cpu_cap"]
        and time.monotonic() - STARTED < contract["wall_cap"]
    )
    output = {
        "schema": "taiji-u2-first-binding-interaction-v1",
        "verdict": "measured_development_interactions_only",
        "groups": reports,
        "optimizer_steps": guard.calls,
        "check_reads": 0,
        "final_reads": 0,
        "capability_qualifications": [],
        "cpu_seconds": time.process_time(),
        "wall_seconds": time.monotonic() - STARTED,
        "run_conditions": {"device": "cpu", "threads": 1, "torch": torch.__version__},
    }
    write_report(REPORT, output)
    print(
        json.dumps(
            {
                "cpu_seconds": output["cpu_seconds"],
                "positive_interactions": {
                    name: group["positive_interactions"] for name, group in reports.items()
                },
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    probe()
