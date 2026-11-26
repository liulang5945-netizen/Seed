"""Reconstruct a selected reference from source, recipe and factual update choices."""

# ruff: noqa: E402 -- include startup in the reconstruction price
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

STARTED = time.monotonic()
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training import calibrate_content_credit_22_5 as content
from scripts.training import calibrate_first_round_22_5 as original
from scripts.training.check_report_run_conditions import write_report
from scripts.training.first_round_baseline_22_5 import FirstRoundBaseline, generate, make_batch
from scripts.training.paired_world_22_5 import build_partition
from seed_platform.paths import get_external_path
from taiji.u2_history import (
    assert_values_equal,
    atomic_load,
    atomic_save,
    detached_tree,
    value_digest,
)

RECIPE = ROOT / "plans/manifests/taiji_u2_reference_rebuild_20261113.json"
REPORT = ROOT / "reports/taiji_u2_reference_rebuild_20261113.json"


def rebuild():
    if REPORT.exists():
        raise FileExistsError(REPORT)
    torch.set_num_threads(1)
    recipe = json.loads(RECIPE.read_text(encoding="utf-8"))
    for name, pin in recipe["source_pins"].items():
        assert (
            hashlib.sha256((ROOT / name).read_text(encoding="utf-8-sig").encode()).hexdigest()
            == pin
        )
    trace_path = ROOT / recipe["learning_trace"]
    assert hashlib.sha256(trace_path.read_bytes()).hexdigest() == recipe["learning_trace_sha256"]
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    data = {name: build_partition(name) for name in ("train", "dev")}
    assert value_digest(data) == recipe["data_digest"]
    batches = {name: make_batch(data[name]) for name in data}
    batch = batches["train"]
    labels, classes, weights = content.labels_for(batch)
    model = FirstRoundBaseline(recipe["body_seed"])
    assert value_digest(model.state_dict()) == recipe["initial_parameters_digest"]
    head = content.head_for(recipe["head_seed"], len(labels))
    optimizer = original.optimizer_for(model)
    optimizer.add_param_group({"params": list(head.parameters())})
    torch.set_rng_state(
        torch.tensor(list(bytes.fromhex(recipe["initial_rng_hex"])), dtype=torch.uint8)
    )
    accepted = 0
    largest_objective_difference = 0.0

    def guard():
        if (
            time.process_time() > recipe["cpu_cap"]
            or time.monotonic() - STARTED > recipe["wall_cap"]
        ):
            raise RuntimeError("reference_rebuild_budget_stop")

    for event in trace["decisions"][: recipe["selected_attempts"]]:
        guard()
        optimizer.zero_grad(set_to_none=True)
        loss = content.terms(model, head, batch, classes, weights)["total"]
        largest_objective_difference = max(
            largest_objective_difference, abs(float(loss.detach()) - event["before"])
        )
        loss.backward()
        parameters = [*model.parameters(), *head.parameters()]
        before = [parameter.detach().clone() for parameter in parameters]
        gradients = [parameter.grad.detach().clone() for parameter in parameters]
        optimizer.step()
        with torch.no_grad():
            direction = [
                parameter.detach().double() - old.double()
                for parameter, old in zip(parameters, before, strict=True)
            ]
            if event["fallback"]:
                step_norm = math.sqrt(sum(float(value.square().sum()) for value in direction))
                gradient_norm = math.sqrt(
                    sum(float(value.double().square().sum()) for value in gradients)
                )
                direction = [-value.double() * (step_norm / gradient_norm) for value in gradients]
            for parameter, old, move in zip(parameters, before, direction, strict=True):
                parameter.copy_(old.double() + event["scale"] * move if event["accepted"] else old)
            accepted += event["accepted"]
    assert accepted == recipe["accepted_parameter_updates"]
    assert value_digest(model.state_dict()) == recipe["final_parameters_digest"]
    full_state = optimizer.state_dict()
    body_optimizer = original.optimizer_for(model)
    body_optimizer.load_state_dict(
        detached_tree(
            {
                "param_groups": full_state["param_groups"][:1],
                "state": {
                    key: value
                    for key, value in full_state["state"].items()
                    if key < len(list(model.parameters()))
                },
            }
        )
    )
    assert value_digest(body_optimizer.state_dict()) == recipe["final_body_optimizer_digest"]
    payload = original.payload_for(
        model,
        body_optimizer,
        recipe["selected_attempts"],
        recipe["data_digest"],
        batch["episode_ids"],
        recipe["selected_training_cpu"],
    )
    payload["training_provenance"] = recipe["inference_training_provenance"]
    root = Path(get_external_path(f"data/experiments/taiji_u2/reference_rebuild/{time.time_ns()}"))
    output = root / "inference-only.bin"
    atomic_save(payload, output)
    restored, other = original.restore(atomic_load(output), recipe["data_digest"])
    assert_values_equal(model.state_dict(), restored.state_dict())
    assert_values_equal(body_optimizer.state_dict(), other.state_dict())
    actual = {name: generate(restored, value, guard) for name, value in batches.items()}
    assert_values_equal(actual, trace["final"], "reference.rebuilt_generation")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    classification = (
        "regenerable_exact"
        if digest == recipe["original_file_sha256"]
        else "regenerable_equivalent"
    )
    guard()
    report = {
        "schema": "taiji-u2-reference-rebuild-v1",
        "verdict": classification,
        "constructed_from": "repository source, deterministic public-world recipe, initial CPU RNG and factual optimizer decisions",
        "optimizer_replay_calls": recipe["selected_attempts"],
        "accepted_parameter_updates": accepted,
        "largest_objective_difference": largest_objective_difference,
        "data_digest": recipe["data_digest"],
        "parameter_digest": value_digest(restored.state_dict()),
        "optimizer_digest": value_digest(other.state_dict()),
        "generation_exact": True,
        "runtime_parameter_count": sum(parameter.numel() for parameter in restored.parameters()),
        "reconstructed_checkpoint": str(output),
        "sha256": digest,
        "original_file_sha256": recipe["original_file_sha256"],
        "cpu_seconds": time.process_time(),
        "wall_seconds": time.monotonic() - STARTED,
        "check_reads": 0,
        "final_reads": 0,
        "capability_qualifications": [],
        "run_conditions": {"device": "cpu", "threads": 1, "torch": torch.__version__},
        "recipe_sha256": hashlib.sha256(RECIPE.read_bytes()).hexdigest(),
    }
    write_report(REPORT, report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("verdict", "optimizer_replay_calls", "generation_exact", "cpu_seconds")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    rebuild()
