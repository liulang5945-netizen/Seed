"""One charged U2 engineering attempt and its frozen evidence payload."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from scripts.training.check_report_run_conditions import write_report
from scripts.training.u2_attempt_ledger_22_3_4_3 import AuthorizationError
from scripts.training.u2_update_support_22_3_4_3 import (
    distributions,
    fingerprint,
    fork_candidate,
    gradients_for,
)
from taiji.u2_history import assert_values_equal, value_digest
from taiji.u2_recovery import run_conditions, validate_subject_state

ROOT = Path(__file__).resolve().parents[2]


def write_once(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    staging = path.with_name(path.name + ".writing")
    write_report(staging, payload)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(staging.read_text(encoding="utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    staging.unlink()


def load_manifest(path: Path) -> dict[str, Any]:
    from scripts.training.verify_taiji_u2_runtime_22_3 import digest

    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "frozen"
        or manifest.get("run_authorized") is not True
        or not manifest.get("owner_approval_ref")
    ):
        raise AuthorizationError("The execution manifest is not frozen and authorized")
    for name, expected in manifest["source_pins"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    return manifest


def admit(model, state, optimizer):
    validate_subject_state(model, state)
    with torch.no_grad():
        probability = model.output(
            state, model.goal(state["goal_raw"]), state["prediction_action"]
        )[0].exp()
        policy = model.policy(state)["log_policy"].exp()
    return bool(torch.isfinite(probability).all() and torch.isfinite(policy).all())


def failed_attempt(runtime, error, step_calls):
    current = fingerprint(runtime)
    return {
        "status": "exception",
        "pre": current,
        "post": current,
        "gradient_sha256": value_digest(None),
        "gradients": None,
        "parameter_rows": [],
        "losses": None,
        "migration": {"status": "failed", "error": f"{type(error).__name__}: {error}"},
        **distributions(runtime),
        "run_conditions": run_conditions(runtime.model),
        "optimizer_step_calls": step_calls,
        "readbacks": [],
        "error": str(error),
    }


def perform(runtime, group, identity, manifest, counter, post_path):
    started, wall = time.process_time(), time.monotonic()
    before = fingerprint(runtime)
    parent_parameters = {
        name: value.detach().clone() for name, value in runtime.model.state_dict().items()
    }
    counter.current = identity
    if identity.startswith("fault-"):
        gradients, losses = None, None
        candidate, optimizer = fork_candidate(runtime, None, fault=True, manifest=manifest)
        if identity == "fault-rejection":
            migration = runtime.migrate_parameters(
                candidate, optimizer=optimizer, acceptance=lambda model, state, opt: False
            )
            assert migration["status"] == "rejected"
            status = "rejected"
        else:
            try:
                runtime.migrate_parameters(
                    candidate, optimizer=optimizer, acceptance=admit, fault="before_commit"
                )
            except RuntimeError as error:
                if str(error) != "injected before_commit":
                    raise
                migration = {"status": "exception", "error": str(error)}
            else:
                raise AssertionError("The frozen fault was not detected")
            status = "exception"
        assert_values_equal(before, fingerprint(runtime), "fault_preserved_parent")
        rows = []
    else:
        gradients, losses = gradients_for(runtime, group, manifest)
        candidate, optimizer = fork_candidate(runtime, gradients, manifest=manifest)
        migration = runtime.migrate_parameters(candidate, optimizer=optimizer, acceptance=admit)
        assert migration["status"] == "committed"
        after_parameters = runtime.model.state_dict()
        rows = [
            {
                "parameter": name,
                "gradient_norm": None if gradient is None else float(gradient.norm()),
                "max_parameter_change": float(
                    (parent_parameters[name] - after_parameters[name]).abs().max()
                ),
            }
            for name, gradient in gradients.items()
        ]
        changes = {row["parameter"]: row for row in rows}
        floor = manifest["activity_floors"][manifest["groups"][group]["dtype"]][
            "max_parameter_change"
        ]
        for name in manifest["groups"][group]["active_parameters"]:
            if changes[name]["max_parameter_change"] < floor:
                raise ValueError(f"Frozen parameter-change floor failed: {name}")
        runtime.save(post_path)
        status = "accepted"
    return {
        "status": status,
        "pre": before,
        "post": fingerprint(runtime),
        "gradient_sha256": value_digest(gradients),
        "gradients": gradients,
        "parameter_rows": rows,
        "losses": losses,
        "migration": migration,
        **distributions(runtime),
        "run_conditions": run_conditions(runtime.model),
        "optimizer_step_calls": int(identity in counter.called),
        "cpu_seconds": time.process_time() - started,
        "wall_seconds": time.monotonic() - wall,
        "readbacks": runtime.readbacks,
    }
