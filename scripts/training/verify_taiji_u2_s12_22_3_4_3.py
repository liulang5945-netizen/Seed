"""Independently verify an immutable U2 S12 attempt without optimizer calls."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

STARTED = time.monotonic()
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training.run_taiji_u2_22_3_4_3 import context
from scripts.training.u2_update_executor_22_3_4_3 import load_manifest, write_once
from scripts.training.u2_update_support_22_3_4_3 import (
    check_adam_equations,
    distributions,
    file_sha,
    fingerprint,
    fresh_load,
    gradients_for,
    parameter_rows,
)
from taiji.u2_history import assert_values_equal, value_digest
from taiji.u2_learning_transaction import rebuild_state
from taiji.u2_recovery import run_conditions


def compare_state(left, right, tolerance, path="state"):
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and left.shape == right.shape, path
        assert torch.allclose(left, right, **tolerance), path
    elif isinstance(left, dict):
        assert set(left) == set(right), path
        for key in left:
            compare_state(left[key], right[key], tolerance, path + "." + key)
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            compare_state(a, b, tolerance, path + f"[{index}]")
    else:
        assert_values_equal(left, right, path)


def verify(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest_path = ROOT / payload["execution_manifest"]
    manifest = load_manifest(manifest_path)
    assert file_sha(manifest_path) == payload["manifest_sha256"]
    root = ROOT / payload["run_root"]
    run, ledger = context(root, manifest, manifest_path)
    state = ledger.read()
    identity = payload["attempt_id"]
    opened, closed = state["opened"][identity], state["closed"][identity]
    assert (
        opened["seq"] == payload["ledger_seq_open"] and closed["seq"] == payload["ledger_seq_close"]
    )
    assert closed["s12_sha256"] == file_sha(path)
    assert closed["outcome"] == payload["status"] and closed["readback_ok"]
    parent_path = ROOT / payload["parent_checkpoint"]
    assert file_sha(parent_path) == payload["parent_sha256"] == opened["parent_sha256"]
    parent = fresh_load(parent_path)
    before = fingerprint(parent)
    for key, field in [
        ("state_sha256", "pre_state_sha256"),
        ("parameter_sha256", "pre_parameter_sha256"),
        ("moment_sha256", "pre_moment_sha256"),
    ]:
        assert before[key] == payload[field] == opened[field]
    equations = []
    if payload["status"] == "accepted":
        claim = json.loads(
            (root / "internal/step_claims" / f"{identity}.json").read_text(encoding="utf-8")
        )
        assert claim == {"attempt_id": identity, "run_id": run["run_id"], "open_seq": opened["seq"]}
        post_path = ROOT / payload["post_checkpoint"]
        assert file_sha(post_path) == payload["post_sha256"] == closed["post_sha256"]
        post = fresh_load(post_path)
        gradients, losses = gradients_for(parent, payload["group"], manifest)
        assert value_digest(gradients) == payload["gradient_sha256"] == closed["gradient_sha256"]
        assert_values_equal(losses, payload["losses"], "worker.losses")
        rows = parameter_rows(parent, post, gradients)
        assert_values_equal(rows, payload["parameter_rows"], "worker.parameter_rows")
        equations = check_adam_equations(parent, post, gradients, manifest, payload["group"])
        assert post.model.parameter_revision == parent.model.parameter_revision + 1
        tolerance = manifest["tolerances"][payload["dtype"]]
        rebuilt, _ = rebuild_state(post.model, post.history, post.state)
        compare_state(post.state, rebuilt, tolerance)
        assert payload["optimizer_step_calls"] == 1
    else:
        assert payload["post_checkpoint"] is None and payload["post_sha256"] is None
        assert payload["optimizer_step_calls"] == 0 and payload["gradient_sha256"] == value_digest(
            None
        )
        post = parent
    after = fingerprint(post)
    for key, field in [
        ("state_sha256", "post_state_sha256"),
        ("parameter_sha256", "post_parameter_sha256"),
        ("moment_sha256", "post_moment_sha256"),
    ]:
        assert after[key] == payload[field] == closed[field]
    assert after["history_reference"] == payload["history_reference"]
    assert after["revision"] == payload["revision"]
    assert_values_equal(
        distributions(post),
        {"next_output": payload["next_output"], "policy": payload["policy"]},
        "worker.output",
    )
    assert run_conditions(post.model) == payload["run_conditions"]
    return {
        "verdict": "pass",
        "attempt_id": identity,
        "run_id": run["run_id"],
        "s12_sha256": file_sha(path),
        "ledger_seq_open": opened["seq"],
        "ledger_seq_close": closed["seq"],
        "parent_verified": before,
        "post_verified": after,
        "adam_equations": equations,
        "optimizer_step_calls": 0,
        "run_conditions": run_conditions(post.model),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s12", type=Path, required=True)
    args = parser.parse_args()
    path = args.s12.resolve()
    output = path.with_name(path.stem + "_worker.json")
    if output.exists():
        parser.error("Worker evidence is immutable")
    torch.set_num_threads(1)
    calls = 0
    original_adam, original_sgd = torch.optim.Adam.step, torch.optim.SGD.step

    def forbidden(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("S12 worker cannot call optimizer.step")

    torch.optim.Adam.step = torch.optim.SGD.step = forbidden
    try:
        report = verify(path)
    except Exception as error:
        report = {
            "verdict": "fail",
            "attempt_id": path.stem,
            "error": f"{type(error).__name__}: {error}",
            "optimizer_step_calls": calls,
            "threads": torch.get_num_threads(),
        }
    finally:
        torch.optim.Adam.step, torch.optim.SGD.step = original_adam, original_sgd
    report.update(
        schema="taiji-u2-s12-worker-v1",
        pid=os.getpid(),
        parent_pid=os.getppid(),
        cpu_seconds=time.process_time(),
        wall_seconds=time.monotonic() - STARTED,
    )
    write_once(output, report)
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "attempt_id": report["attempt_id"],
                "cpu_seconds": report["cpu_seconds"],
            }
        )
    )
    return 0 if report["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
