"""One stage runner for 10.21: ``--stage graph`` (10.21.2) and ``--stage price`` (10.21.3).

Graph stage: one candidate, four profiles, no training. Checks the declared graph
and its engineering gates only: parameter matrix, role queries, source read,
shared binding, the mode-6 internal event, rank-4 action scores, FD per parameter
group, float32/64 reporting, negative controls, cache boundaries, initial and
post-step-1 independent restore, and the fixed Adam budget.

Price stage: one binding seed, two arms, two declared steps each, four separate
unit prices (train step, restore, eval example, bundle MiB), then a declared
extrapolation against the 10.21.4 cap. It saves the new bundle, proves an
independent restore reproduces the parent forward bitwise, and only then prices.

Neither stage makes a capability claim. The parent 09.8 checkpoint is read only
and never modified; engineering parameter copies are discarded.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import multiprocessing
import os
import random
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from taiji.candidate_manifest import (  # noqa: E402
    build_manifest,
    load_candidate,
    parameter_digest,
)
from taiji.living_core import RawSource  # noqa: E402
from taiji.native_composition_fixture import (  # noqa: E402
    ACTION_BYTES,
    DEV_RECORD_BYTES,
    DEV_RECORD_LINES,
    GOAL_BYTES,
    TRAIN_RECORD_BYTES,
    TRAIN_RECORD_LINES,
    FixtureExample,
    FixtureWorld,
    build_split,
    fixture_digest,
    price_dev_examples,
    price_train_examples,
)
from taiji.native_composition_fixture import fixture_payload as composition_payload  # noqa: E402
from taiji.native_goal_glossary import glossary_records  # noqa: E402
from taiji.query_binding import (  # noqa: E402
    BINDING_MODE,
    CLIP_NORM,
    DEFAULT_BETAS,
    DEFAULT_EPS,
    DEFAULT_LR,
    PARAMETER_GROUPS,
    S_TRAINABLE_GROUPS,
    BindingConfig,
    BindingFeatures,
    QueryBindingCandidate,
    binding_loss,
    binding_step,
    build_binding,
    build_binding_features,
    expected_parameter_count,
    features_sha,
    freeze_parent,
    gates_provider,
    load_binding_bundle,
    make_optimizer,
    module_bundle_payload,
    module_fixture_state,
    module_parameter_digest,
    run_binding_worker,
    run_module_worker,
    save_binding_bundle,
    save_module_bundle,
    score_actions,
    set_trainable,
    trainable_parameters,
)
from taiji.state_migration import resume_subject  # noqa: E402

BUNDLE = os.environ.get("TAIJI_BUNDLE_PATH", "output/taiji_core_09/subject_bundle.pt")
CLOSEOUT = "reports/taiji_core_09_closeout_20261101.json"
EXPECTED_CANDIDATE = "feb0cbcb00e2e16bb66b77db530ae566ce40eecaf2e14852b7cc1661292139dd"
EXPECTED_BUNDLE_SHA = "ce4c62685d55bb518d234e7407e19b05257d1d761a0430432f7d0f09641321cf"
EXPECTED_BUNDLE_BYTES = 77609636
MANIFEST = "plans/manifests/taiji_core_10_21_graph_v2_20261101.json"
OUTPUT_DIR = "output/taiji_core_10_21_graph"

BUDGET_CPU_SECONDS = 300
BUDGET_WALL_SECONDS = 600
MAX_ADAM_CALLS = 12
MAX_PRODUCT_BYTES = 1024**3
FD_EPS_F64 = (1e-4, 1e-5, 1e-6)
FD_EPS_F32 = (1e-2, 1e-3, 1e-4)
FD_ATOL = 1e-10
FD_RTOL = 1e-4
FD_ATOL_F32 = 1e-6
FD_RTOL_F32 = 1e-2
SIGNAL_FACTOR = 10.0
ASCENT_STEP = 1e-3
WORKER_TIMEOUT = 180

SOURCE_FILES = (
    "taiji/query_binding.py",
    "taiji/living_core.py",
    "taiji/growth_credit.py",
    "taiji/state_migration.py",
    "taiji/candidate_manifest.py",
    "taiji/native_development.py",
    "taiji/native_goal_attributes.py",
    "taiji/native_goal_glossary.py",
    "scripts/training/verify_taiji_composition_10_21.py",
)
MANIFEST_SOURCE_FILES = SOURCE_FILES[:-1]
PARENT_SOURCE_FILES = (
    "taiji/living_core.py",
    "taiji/growth_credit.py",
    "taiji/state_migration.py",
    "taiji/native_development.py",
)

# --- 10.21.3 price stage -------------------------------------------------------------
PRICE_SCHEMA = "taiji-composition-price-v1"
PRICE_DATE = "2026-11-01"
PRICE_BUDGET_CPU_SECONDS = 180
PRICE_BUDGET_WALL_SECONDS = 360
PRICE_MAX_PRODUCT_BYTES = 512 * 1024**2
PRICE_CAP_CPU_SECONDS = 1200.0
PRICE_OUTPUT_DIR = "output/taiji_core_10_21_3_price"
PRICE_MANIFEST = "plans/manifests/taiji_core_10_21_3_price_20261101.json"
PRICE_SOURCE_FILES = SOURCE_FILES + ("taiji/native_composition_fixture.py",)
PRICE_BINDING_SEED = 17021
PRICE_TRAIN_WORLDS = 1
PRICE_DEV_WORLDS = 1
PRICE_TRAIN_STEPS = 2
PRICE_TRAIN_EXAMPLES = 8
PRICE_DEV_EXAMPLES = 8
PRICE_SUBJECT = "10.21.3-price"
PRICE_FORMULA = (
    "total_cpu_seconds = cpu_per_train_step * train_steps"
    " + cpu_per_restore * restore_count"
    " + cpu_per_eval_example * eval_example_count"
    " + cpu_per_bundle_MiB * bundle_io_MiB"
)
# Declared before any timing; the run only fills in the measured unit prices.
PRICE_EXTRAPOLATION_INPUTS = {
    "seeds": 2,
    "arms": 2,
    "steps_per_arm": 300,
    "train_steps": 1200,
    "restore_count": 8,
    "eval_points_per_arm": 2,
    "dev_examples_per_point": 160,
    "eval_example_count": 1280,
    "bundle_cycles": 12,
}
PRICE_LIMITS = (
    "Pricing only: four separate unit prices measured once on this machine, one binding seed, "
    "two arms x two steps, one train world and one dev world. This is not a comparison of J and "
    "S, not a capability measurement, and not an admission for 10.21.4. The 8 developmental "
    "examples are recorded as consumed feedback and are not a gate; no hyperparameter, no "
    "sample and no step count was chosen from them. The extrapolation inherits the declared "
    "10.21.4 shape (2 seeds x 2 arms x 300 steps, 2 evaluation points per arm x 160 dev "
    "examples, 12 bundle write/read cycles) and only multiplies measured prices by it."
)

# --- 10.21.4 compare stage -----------------------------------------------------------
COMPARE_SCHEMA = "taiji-composition-compare-v1"
COMPARE_DATE = "2026-11-01"
COMPARE_BUDGET_CPU_SECONDS = 1200
COMPARE_BUDGET_WALL_SECONDS = 2400
COMPARE_MAX_PRODUCT_BYTES = 2 * 1024**3
COMPARE_OUTPUT_DIR = "output/taiji_core_10_21_4_compare"
COMPARE_MANIFEST = "plans/manifests/taiji_core_10_21_4_compare_20261101.json"
COMPARE_SEEDS: tuple[tuple[int, int], ...] = ((17021, 27021), (17022, 27022))
COMPARE_MAIN_SEED = 17021
COMPARE_STEPS = 300
COMPARE_BATCH = 8
COMPARE_CHECKPOINTS = (0, 150, 300)
COMPARE_STAT_SEED = 57021
COMPARE_BOOTSTRAP_DRAWS = 2000
COMPARE_TASK_BAR = 0.40
COMPARE_CREDIT_BAR = 0.10
COMPARE_READER_BAR = 0.40
COMPARE_SATURATION_BAR = 0.90
COMPARE_SUBJECT = "10.21.4-compare"
COMPARE_LAYERS = ("seen_combination", "unseen_combination", "unseen_entity")
COMPARE_SOURCE_FILES = PRICE_SOURCE_FILES
COMPARE_EXIT_STATES = (
    "instrument_invalid",
    "engineering_failure",
    "reader_selection_absent",
    "binding_generalization_absent",
    "no_reader_credit_advantage",
    "task_saturated_for_credit",
    "budget_stop",
    "binding_mechanism_supported_on_declared_fixture",
)
COMPARE_LIMITS = (
    "Synthetic composition fixture only: one reference graph (the frozen 09.8 parent) plus the "
    "declared QueryBindingCandidate, fixed 3x5 object/place template, 36-byte goals and 27-byte "
    "canonical actions. This mechanism gate is not a claim about P2 or L3, not about free-form "
    "Chinese generation, not about long-term retention, and not about the product default. "
    "Dev-facing reads happen at exactly step0 and step300; the 17022 seed is a stability "
    "re-check and is never used to pick a best candidate. Both seeds are reported separately "
    "and are never pooled into 16 independent worlds."
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def fixtures() -> tuple[dict, dict]:
    real = glossary_records()
    first = {
        "name": "records",
        "goal": "让圆口的器皿到四方木台。",
        "records": real,
        "actions": (("蓝色杯子", "木桌"), ("红色印章", "窗台")),
        "target": 0,
    }
    second = {
        "name": "null_only",
        "goal": "让带纹路的方块到靠窗的台面。",
        "records": (),
        "actions": (("红色印章", "窗台"), ("蓝色杯子", "木桌")),
        "target": 0,
    }
    return first, second


def fixture_payload(fixture: dict) -> dict:
    return {
        "name": fixture["name"],
        "goal": fixture["goal"],
        "records": [
            {"source_id": record.source_id, "raw": record.raw.decode("utf-8")}
            for record in fixture["records"]
        ],
        "actions": [list(action) for action in fixture["actions"]],
        "target": fixture["target"],
    }


def fixture_sha(fixture: dict) -> str:
    digest = hashlib.sha256()
    digest.update(fixture["goal"].encode())
    for record in fixture["records"]:
        digest.update(record.source_id.encode())
        digest.update(record.raw)
    for obj, place in fixture["actions"]:
        digest.update(obj.encode())
        digest.update(place.encode())
    digest.update(str(fixture["target"]).encode())
    return digest.hexdigest()


def check(rows: list, check_id: str, passed: bool, observed, note: str = "") -> None:
    rows.append(
        {
            "id": check_id,
            "passed": bool(passed),
            "observed": observed,
            "note": note,
        }
    )


def max_abs_diff(first: torch.Tensor, second: torch.Tensor) -> float:
    if first.shape != second.shape:
        return float("inf")
    return float((first.detach() - second.detach()).abs().max())


def tree_equal(first, second) -> bool:
    if torch.is_tensor(first) or torch.is_tensor(second):
        return bool(torch.equal(first, second))
    if isinstance(first, dict) and isinstance(second, dict):
        if set(first) != set(second):
            return False
        return all(tree_equal(first[key], second[key]) for key in first)
    if isinstance(first, (list, tuple)) and isinstance(second, (list, tuple)):
        if len(first) != len(second):
            return False
        return all(tree_equal(left, right) for left, right in zip(first, second, strict=True))
    return bool(first == second)


def group_digests(module) -> dict[str, str]:
    return {
        group: parameter_digest(
            {name: getattr(module, name).detach() for name in PARAMETER_GROUPS[group]}
        )
        for group in PARAMETER_GROUPS
    }


def group_deltas(before: dict[str, dict[str, torch.Tensor]], module) -> dict:
    rows = {}
    for group in PARAMETER_GROUPS:
        delta = 0.0
        scale = 0.0
        for name in PARAMETER_GROUPS[group]:
            parameter = getattr(module, name).detach()
            delta = max(delta, float((parameter - before[group][name]).abs().max()))
            scale = max(scale, float(parameter.abs().max()))
        ulp = torch.finfo(next(module.parameters()).dtype).eps * max(scale, 1.0)
        rows[group] = {
            "delta_theta": delta,
            "max_abs_theta": scale,
            "ulp": ulp,
            "delta_over_ulp": delta / ulp if ulp else 0.0,
        }
    return rows


def group_snapshot(module) -> dict[str, dict[str, torch.Tensor]]:
    return {
        group: {name: getattr(module, name).detach().clone() for name in PARAMETER_GROUPS[group]}
        for group in PARAMETER_GROUPS
    }


def moments_by_group(module, optimizer) -> dict:
    state = optimizer.state
    rows = {}
    for group in PARAMETER_GROUPS:
        entry = {}
        for name in PARAMETER_GROUPS[group]:
            parameter = getattr(module, name)
            moment = state.get(parameter)
            if not moment:
                continue
            entry[name] = {
                "step": int(moment["step"]),
                "exp_avg_norm": float(moment["exp_avg"].norm()),
                "exp_avg_sq_norm": float(moment["exp_avg_sq"].norm()),
            }
        rows[group] = entry
    return rows


def fixture_forward(
    module,
    parent,
    features: BindingFeatures,
    fixture: dict,
    gates,
) -> dict:
    b_goal, b_actions, _ = module.forward_binding(features)
    z_bind, h_state, output = binding_step(parent, features, b_goal, gates)
    scores, probabilities = score_actions(module, z_bind, b_actions)
    loss = binding_loss(scores, fixture["target"])
    return {
        "b_goal": b_goal,
        "b_actions": b_actions,
        "z_bind": z_bind,
        "h": h_state,
        "scores": scores,
        "probabilities": probabilities,
        "loss": loss,
        "output": output,
    }


def combined_loss(module, parent, features: tuple, fixture_rows: tuple, gates):
    losses = [
        fixture_forward(module, parent, features[index], fixture_rows[index], gates)["loss"]
        for index in range(len(fixture_rows))
    ]
    return sum(losses) / len(losses)


def no_grad_loss(module, parent, features: tuple, fixture_rows: tuple, gates) -> float:
    with torch.no_grad():
        return float(combined_loss(module, parent, features, fixture_rows, gates).detach())


def group_gradients(module, parent, features, fixture_rows, gates, group: str):
    module.zero_grad(set_to_none=True)
    loss = combined_loss(module, parent, features, fixture_rows, gates)
    parameters = module.group_parameters(group)
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=False)
    filled = [
        torch.zeros_like(parameter) if gradient is None else gradient.detach()
        for parameter, gradient in zip(parameters, gradients, strict=True)
    ]
    norm = float(torch.sqrt(sum(gradient.square().sum() for gradient in filled)))
    return loss, parameters, filled, norm


def perturb(parameters, direction, sign: float) -> None:
    with torch.no_grad():
        for parameter, step in zip(parameters, direction, strict=True):
            parameter.add_(sign * step)


def fd_group(
    module,
    parent,
    features,
    fixture_rows,
    gates,
    group: str,
    eps_values: tuple[float, ...],
    atol: float,
    rtol: float,
    min_consistent: int = 2,
) -> dict:
    _loss, parameters, gradients, norm = group_gradients(
        module, parent, features, fixture_rows, gates, group
    )
    base_loss = no_grad_loss(module, parent, features, fixture_rows, gates)
    row = {
        "group": group,
        "parameter_count": sum(parameter.numel() for parameter in parameters),
        "grad_norm": norm,
        "base_loss": base_loss,
        "fd": {},
        "fd_error": {},
        "signal_floor": SIGNAL_FACTOR * atol,
        "resolved": False,
        "note": "",
    }
    if norm <= SIGNAL_FACTOR * atol:
        row["note"] = "below_signal_floor"
        row["zero_gradient_rejected"] = True
        row["sign_flip_rejected"] = True
        row["zero_params_loss_delta"] = 0.0
        row["ascent_loss_delta"] = 0.0
        return row
    direction = [gradient / norm for gradient in gradients]
    fd_values = []
    for eps in eps_values:
        perturb(parameters, direction, eps)
        plus = no_grad_loss(module, parent, features, fixture_rows, gates)
        perturb(parameters, direction, -2.0 * eps)
        minus = no_grad_loss(module, parent, features, fixture_rows, gates)
        perturb(parameters, direction, eps)
        forward = (plus - minus) / (2.0 * eps)
        error = abs(forward - norm) / max(norm, 1e-30)
        row["fd"][f"{eps:g}"] = forward
        row["fd_error"][f"{eps:g}"] = error
        fd_values.append((forward, error))
    consistent = (
        fd_values[0][1] <= rtol
        and sum(1 for _, error in fd_values if error <= rtol) >= min_consistent
    )
    signal = norm > SIGNAL_FACTOR * atol
    zero_direction = [torch.zeros_like(gradient) for gradient in gradients]
    perturb(parameters, zero_direction, 1.0)
    zero_fd = no_grad_loss(module, parent, features, fixture_rows, gates) - no_grad_loss(
        module, parent, features, fixture_rows, gates
    )  # exactly zero by construction; kept explicit for the report
    zero_rejected = abs(zero_fd) <= SIGNAL_FACTOR * atol
    flipped = [-gradient / norm for gradient in gradients]
    eps_flip = eps_values[0]
    perturb(parameters, flipped, eps_flip)
    flip_plus = no_grad_loss(module, parent, features, fixture_rows, gates)
    perturb(parameters, flipped, -2.0 * eps_flip)
    flip_minus = no_grad_loss(module, parent, features, fixture_rows, gates)
    perturb(parameters, flipped, eps_flip)
    flip_fd = (flip_plus - flip_minus) / (2.0 * eps_flip)
    flip_rejected = flip_fd <= SIGNAL_FACTOR * atol
    snapshot = {name: getattr(module, name).detach().clone() for name in PARAMETER_GROUPS[group]}
    with torch.no_grad():
        for name in PARAMETER_GROUPS[group]:
            getattr(module, name).zero_()
    zero_params_loss = no_grad_loss(module, parent, features, fixture_rows, gates)
    with torch.no_grad():
        for name in PARAMETER_GROUPS[group]:
            getattr(module, name).copy_(snapshot[name])
    perturb(parameters, direction, -ASCENT_STEP)
    ascent_loss = no_grad_loss(module, parent, features, fixture_rows, gates)
    perturb(parameters, direction, ASCENT_STEP)
    row.update(
        {
            "fd": {key: value for key, value in row["fd"].items()},
            "fd_error": {key: value for key, value in row["fd_error"].items()},
            "eps_consistent": consistent,
            "signal": signal,
            "zero_gradient_rejected": zero_rejected,
            "sign_flip_rejected": flip_rejected,
            "flip_fd": flip_fd,
            "zero_params_loss_delta": zero_params_loss - base_loss,
            "ascent_loss_delta": ascent_loss - base_loss,
            "resolved": bool(signal and consistent and zero_rejected and flip_rejected),
        }
    )
    if not row["resolved"]:
        row["note"] = "eps_inconsistent" if not consistent else "signal_below_floor"
    return row


def frozen_group_rows(module) -> dict:
    rows = {}
    for group in PARAMETER_GROUPS:
        parameters = module.group_parameters(group)
        rows[group] = {
            "parameter_count": sum(parameter.numel() for parameter in parameters),
            "requires_grad": any(parameter.requires_grad for parameter in parameters),
        }
    return rows


def run_type_check() -> dict:
    command = [
        sys.executable,
        "-m",
        "mypy",
        "--follow-imports=skip",
        "--ignore-missing-imports",
        "taiji/query_binding.py",
        "scripts/training/verify_taiji_composition_10_21.py",
    ]
    try:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=180)
        errors = 0
        for line in result.stdout.splitlines():
            if line.startswith("Found ") and " error" in line:
                tokens = line.split()
                errors = int(tokens[1])
        return {
            "returncode": result.returncode,
            "errors": errors,
            "output": result.stdout.strip(),
            "passed": result.returncode == 0 and errors == 0,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "returncode": None,
            "errors": None,
            "output": f"{type(exc).__name__}: {exc}",
            "passed": False,
        }


def run_lint() -> dict:
    ruff = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "taiji/query_binding.py",
            "scripts/training/verify_taiji_composition_10_21.py",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    black = subprocess.run(
        [
            sys.executable,
            "-m",
            "black",
            "--no-cache",
            "--check",
            "taiji/query_binding.py",
            "scripts/training/verify_taiji_composition_10_21.py",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return {
        "ruff": {"returncode": ruff.returncode, "output": ruff.stdout.strip()},
        "black": {"returncode": black.returncode, "output": black.stdout.strip()},
        "passed": ruff.returncode == 0 and black.returncode == 0,
    }


def spawn_worker(target, spec: dict) -> Any:
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=target, args=(spec,))
    process.start()
    process.join(timeout=WORKER_TIMEOUT)
    if process.is_alive():
        process.terminate()
        process.join()
        return {
            "ok": False,
            "exitcode": None,
            "message": "worker timed out",
        }
    if process.exitcode != 0:
        return {
            "ok": False,
            "exitcode": process.exitcode,
            "message": "worker process did not exit cleanly",
        }
    result_path = Path(spec["result"])
    if not result_path.exists():
        return {"ok": False, "exitcode": 0, "message": "worker wrote no result"}
    return torch.load(result_path, map_location="cpu", weights_only=True)


def worker_spec(
    bundle_path: Path,
    result_path: Path,
    fixtures_rows: tuple,
    identity: dict,
    mode: str,
    dtype: torch.dtype,
) -> dict:
    return {
        "bundle": str(bundle_path),
        "result": str(result_path),
        "fixtures": [fixture_payload(fixture) for fixture in fixtures_rows],
        "identity": identity,
        "mode": mode,
        "dtype": dtype_name(dtype),
    }


def module_worker_spec(bundle_path: Path, result_path: Path, mode: str, dtype: torch.dtype) -> dict:
    return {
        "bundle": str(bundle_path),
        "result": str(result_path),
        "mode": mode,
        "dtype": dtype_name(dtype),
    }


def initial_manifest(
    parent_manifest: dict,
    module,
    config: BindingConfig,
    small_config: BindingConfig,
    fixtures_rows: tuple,
    source_hashes: dict,
) -> dict:
    module_id = module_parameter_digest(module)
    candidate_id = sha256_bytes(
        (
            parent_manifest["candidate_id"]
            + module_id
            + json.dumps(asdict(config), sort_keys=True)
            + str(BINDING_MODE)
        ).encode()
    )
    parameters = [
        {
            "name": name,
            "shape": list(parameter.shape),
            "count": parameter.numel(),
            "group": next(group for group in PARAMETER_GROUPS if name in PARAMETER_GROUPS[group]),
        }
        for name, parameter in module.named_parameters()
    ]
    return {
        "schema": "taiji-query-binding-manifest-v1",
        "step": "10.21.2",
        "date": "2026-11-01",
        "parent": {
            "candidate_id": parent_manifest["candidate_id"],
            "bundle_sha256": parent_manifest["provenance"]["sha256"],
            "bundle_bytes": parent_manifest["provenance"]["bytes"],
        },
        "module_id": module_id,
        "candidate_id": candidate_id,
        "module": {
            "name": "QueryBindingCandidate",
            "config": asdict(config),
            "parameter_count": module.parameter_count(),
            "parameter_groups": {
                group: sum(getattr(module, name).numel() for name in PARAMETER_GROUPS[group])
                for group in PARAMETER_GROUPS
            },
            "parameters": parameters,
            "trainable_groups_J": list(PARAMETER_GROUPS),
            "trainable_groups_S": list(S_TRAINABLE_GROUPS),
            "internal_mode": BINDING_MODE,
            "fast_write": "disabled in the binding step",
            "function_neutral_birth": False,
            "cache_policy": {
                "cacheable": [
                    "parent goal features",
                    "parent source positions/metadata",
                    "parent readout s",
                    "frozen action encodings",
                ],
                "not_cacheable": [
                    "role query",
                    "source read",
                    "b_goal",
                    "z_bind",
                    "h",
                    "score",
                ],
                "cache_key_fields": [
                    "parent_candidate_id",
                    "parent_bundle_sha256",
                    "parent_revision",
                    "parent_step",
                    "goal_sha256",
                    "records_sha256",
                    "state_digest",
                    "source_hashes",
                    "action_sha256",
                ],
            },
            "small_config": asdict(small_config),
            "small_parameter_count": expected_parameter_count(
                small_config.width,
                small_config.query_width,
                small_config.roles,
                small_config.rank,
            ),
        },
        "fixtures": [
            {"name": fixture["name"], "sha256": fixture_sha(fixture)} for fixture in fixtures_rows
        ],
        "profiles": [
            "default-J-float64",
            "default-J-float32",
            "default-S-float64",
            "small-J-float64",
        ],
        "source_hashes": {relative: source_hashes[relative] for relative in MANIFEST_SOURCE_FILES},
        "budget": {
            "cpu_seconds": BUDGET_CPU_SECONDS,
            "wall_seconds": BUDGET_WALL_SECONDS,
            "adam_calls": MAX_ADAM_CALLS,
            "product_bytes": MAX_PRODUCT_BYTES,
        },
        "exit": {
            "success": "passed_10_21_graph_engineering_only",
            "graph_failure": "engineering_failure",
            "budget": "budget_stop",
        },
        "claim": "engineering only; no task ability, pricing or training admission",
    }


class PriceStop(RuntimeError):
    """A pre-pricing gate failed; the run stops before any unit price is taken."""


def price_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def price_equal(left, right) -> bool:
    if left is None or right is None:
        return False
    return tree_equal(left, right)


def price_example_payload(example: FixtureExample) -> dict:
    return {
        "index": example.index,
        "layer": example.layer,
        "goal": example.goal,
        "actions": [list(pair) for pair in example.actions],
        "target": example.target,
        "records": [
            {"source_id": record.source_id, "raw": record.raw.decode("utf-8")}
            for record in example.records
        ],
    }


def price_records(payload: dict) -> tuple[RawSource, ...]:
    return tuple(
        RawSource(source_id=item["source_id"], raw=item["raw"].encode())
        for item in payload["records"]
    )


def price_forward(module, parent, features: BindingFeatures, target: int, gates) -> dict:
    b_goal, b_actions, _ = module.forward_binding(features)
    z_bind, h_state, _output = binding_step(parent, features, b_goal, gates)
    scores, probabilities = score_actions(module, z_bind, b_actions)
    return {
        "scores": scores,
        "probabilities": probabilities,
        "h": h_state,
        "loss": binding_loss(scores, target),
    }


def price_features(
    parent,
    state,
    example: dict,
    gates,
    identity: dict,
) -> BindingFeatures:
    return build_binding_features(
        parent,
        state,
        example["goal"].encode(),
        price_records(example),
        tuple(item[0] for item in example["actions"]),
        tuple(item[1] for item in example["actions"]),
        gates,
        identity,
    )


def price_restore_worker(spec: dict) -> dict:
    """Independent-process restore: load the new bundle, rebuild from raw, forward once."""
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    torch.set_num_threads(1)
    result_path = Path(spec["result"])
    try:
        load_start = time.process_time()
        (
            parent,
            state,
            module,
            _optimizer,
            _cursor,
            gates,
            profile,
            dtype,
            _groups,
        ) = load_binding_bundle(Path(spec["bundle"]), price_dtype(spec["dtype"]))
        load_cpu = time.process_time() - load_start
        features_start = time.process_time()
        features = price_features(parent, state, spec["example"], gates, spec["identity"])
        features_cpu = time.process_time() - features_start
        forward_start = time.process_time()
        forward = price_forward(module, parent, features, spec["example"]["target"], gates)
        forward_cpu = time.process_time() - forward_start
        torch.save(
            {
                "ok": True,
                "profile": profile,
                "dtype": dtype_name(dtype),
                "features_sha": features_sha(features),
                "scores": forward["scores"].detach(),
                "h": forward["h"].detach(),
                "loss": float(forward["loss"].detach()),
                "load_cpu_seconds": load_cpu,
                "features_cpu_seconds": features_cpu,
                "forward_cpu_seconds": forward_cpu,
                "cpu_seconds": time.process_time() - start_cpu,
                "wall_seconds": time.perf_counter() - start_wall,
            },
            result_path,
        )
        return {"ok": True, "result": str(result_path), "cpu_seconds": 0.0}
    except Exception as exc:  # noqa: BLE001
        torch.save(
            {
                "ok": False,
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "cpu_seconds": time.process_time() - start_cpu,
                "wall_seconds": time.perf_counter() - start_wall,
            },
            result_path,
        )
        return {"ok": False, "result": str(result_path), "cpu_seconds": 0.0}


def price_train_step(
    module,
    parent,
    rows: tuple,
    features: tuple,
    optimizer,
    trainable: list,
    gates,
) -> dict:
    """One declared step: mean binding loss over the batch, backward, clip, Adam."""
    start = time.process_time()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for row, feature in zip(rows, features, strict=True):
        b_goal, b_actions, _ = module.forward_binding(feature)
        z_bind, _h, _output = binding_step(parent, feature, b_goal, gates)
        scores, _probabilities = score_actions(module, z_bind, b_actions)
        losses.append(binding_loss(scores, row.target))
    loss = sum(losses) / len(losses)
    loss.backward()
    clip_scale = float(torch.nn.utils.clip_grad_norm_(trainable, CLIP_NORM))
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        "clip_scale": clip_scale,
        "cpu_seconds": time.process_time() - start,
        "batch_losses": [float(item.detach()) for item in losses],
    }


def price_dev_reading(module, parent, rows: tuple, gates) -> dict:
    """One dev example end to end under no_grad; cost and prediction only."""
    readings = []
    cpu_seconds = 0.0
    features_cpu = 0.0
    forward_cpu = 0.0
    correct = 0
    for row in rows:
        with torch.no_grad():
            start = time.process_time()
            state = parent.initial_state(PRICE_SUBJECT)
            features_start = time.process_time()
            features = price_features(parent, state, price_example_payload(row), gates, {})
            features_end = time.process_time()
            forward = price_forward(module, parent, features, row.target, gates)
            end = time.process_time()
        features_cpu += features_end - features_start
        forward_cpu += end - features_end
        cpu_seconds += end - start
        prediction = int(torch.argmax(forward["scores"]).item())
        hit = prediction == row.target
        correct += int(hit)
        readings.append(
            {
                "index": row.index,
                "layer": row.layer,
                "goal": row.goal,
                "target": row.target,
                "prediction": prediction,
                "correct": bool(hit),
                "loss": float(forward["loss"]),
            }
        )
    return {
        "count": len(rows),
        "correct": correct,
        "cpu_seconds": cpu_seconds,
        "features_cpu_seconds": features_cpu,
        "forward_cpu_seconds": forward_cpu,
        "readings": readings,
    }


def price_manifest_payload(
    parent_manifest: dict,
    source_hashes: dict,
    report: dict,
    worlds: tuple[FixtureWorld, ...],
) -> dict:
    """Identity document for 10.21.3: parent, module, data, code, prices, exit states."""
    module = report["module"]
    return {
        "schema": "taiji-composition-price-manifest-v1",
        "step": "10.21.3",
        "date": PRICE_DATE,
        "parent": {
            "candidate_id": parent_manifest["candidate_id"],
            "bundle": BUNDLE,
            "bundle_sha256": parent_manifest["provenance"]["sha256"],
            "bundle_bytes": parent_manifest["provenance"]["bytes"],
        },
        "module": {
            "name": module["name"],
            "config": module["config"],
            "dtype": module["dtype"],
            "parameter_count": module["parameter_count"],
            "parameter_groups": module["parameter_groups"],
            "parameter_digest_initial": module["parameter_digest_initial"],
            "internal_mode": module["internal_mode"],
            "fast_write": module["fast_write"],
            "optimizer": module["optimizer"],
        },
        "arms": {
            arm: {
                "trainable_groups": row["trainable_groups"],
                "trainable_parameter_count": row["trainable_parameter_count"],
                "parameter_digest_before": row["parameter_digest_before"],
                "parameter_digest_after": row["parameter_digest_after"],
            }
            for arm, row in report["arms"].items()
        },
        "fixture": composition_payload(worlds),
        "source_hashes": {relative: source_hashes[relative] for relative in PRICE_SOURCE_FILES},
        "unit_prices": {
            key: report["unit_prices"][key]["value"]
            for key in (
                "cpu_per_train_step",
                "cpu_per_restore",
                "cpu_per_eval_example",
                "cpu_per_bundle_MiB",
            )
        },
        "extrapolation": {
            "formula": report["extrapolation"]["formula"],
            "inputs": report["extrapolation"]["inputs"],
            "terms": report["extrapolation"]["terms"],
            "total_cpu_seconds": report["extrapolation"]["total_cpu_seconds"],
            "cap_cpu_seconds": PRICE_CAP_CPU_SECONDS,
            "within_cap": report["extrapolation"]["within_cap"],
        },
        "budget": {
            "cpu_seconds": PRICE_BUDGET_CPU_SECONDS,
            "wall_seconds": PRICE_BUDGET_WALL_SECONDS,
            "products_bytes": PRICE_MAX_PRODUCT_BYTES,
            "cap_cpu_seconds_10_21_4": PRICE_CAP_CPU_SECONDS,
        },
        "exit": {
            "success": "priced",
            "not_admitted": "budget_not_admitted",
            "budget": "budget_stop",
            "failure": "engineering_failure",
        },
        "claim": "pricing only; no comparison, no capability, no 10.21.4 admission",
    }


def price_main(args) -> int:
    """10.21.3: identity, restore, two declared steps, four separate unit prices."""
    torch.set_num_threads(1)
    if args.output and args.output.exists():
        raise FileExistsError("10.21.3 price card is write-once")
    started = (time.process_time(), time.perf_counter())
    report: dict[str, Any] = {
        "schema": PRICE_SCHEMA,
        "step": "10.21.3",
        "date": PRICE_DATE,
        "status": "engineering_failure",
        "checks": [],
        "identity": {},
        "module": {},
        "arms": {},
        "unit_prices": {
            "cpu_per_train_step": {},
            "cpu_per_restore": {},
            "cpu_per_eval_example": {},
            "cpu_per_bundle_MiB": {},
        },
        "extrapolation": {
            "formula": PRICE_FORMULA,
            "inputs": dict(PRICE_EXTRAPOLATION_INPUTS),
            "unit_prices_used": {},
            "terms": {},
            "total_cpu_seconds": None,
            "cap_cpu_seconds": PRICE_CAP_CPU_SECONDS,
            "within_cap": None,
            "written_before_timing": True,
        },
        "consumed_development_feedback": {
            "count": 0,
            "note": "not a gate; does not tune hyperparameters",
            "examples": [],
        },
        "optimizer_calls": 0,
        "effective_cursor": {},
        "restore": {},
        "cost": {},
        "limits": PRICE_LIMITS,
    }
    parent_path = ROOT / BUNDLE
    parent_sha_before = sha256_text(parent_path)
    worker_cpu = 0.0
    worker_wall = 0.0
    optimizer_calls = 0
    bundle_records: list[dict] = []
    temporary: list[Path] = []
    over_budget = False
    try:
        parent_manifest = build_manifest(ROOT, parent_path)
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text(
                json.dumps(parent_manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            _manifest, payload = load_candidate(ROOT, manifest_path, parent_path, "frozen_eval")
        parent, _parent_state, _ = resume_subject(payload)
        parent.eval()
        freeze_parent(parent)
        gates_table = payload["growth_gates"]
        gates = gates_provider(gates_table)
        source_hashes = {relative: sha256_text(ROOT / relative) for relative in PRICE_SOURCE_FILES}
        identity = {
            "parent_candidate_id": parent_manifest["candidate_id"],
            "parent_bundle_sha256": parent_manifest["provenance"]["sha256"],
            "source_hashes": source_hashes,
        }
        report["identity"] = {
            "parent_candidate_id": parent_manifest["candidate_id"],
            "parent_bundle": {
                "path": BUNDLE,
                "sha256": parent_manifest["provenance"]["sha256"],
                "bytes": parent_manifest["provenance"]["bytes"],
            },
            "parent_subject_id": payload["subject_id"],
            "parent_revision": payload["revision"],
            "parent_step": payload["step"],
            "parent_config": payload["config"],
            "growth_parent_units": payload["growth_parent_units"],
            "growth_unit_branches": payload["growth_unit_branches"],
            "growth_gates": gates_table,
            "source_hashes": source_hashes,
        }
        check(
            report["checks"],
            "parent_identity",
            parent_manifest["candidate_id"] == EXPECTED_CANDIDATE
            and parent_manifest["provenance"]["sha256"] == EXPECTED_BUNDLE_SHA
            and parent_manifest["provenance"]["bytes"] == EXPECTED_BUNDLE_BYTES,
            {
                "candidate": parent_manifest["candidate_id"],
                "sha256": parent_manifest["provenance"]["sha256"],
                "bytes": parent_manifest["provenance"]["bytes"],
            },
        )
        check(
            report["checks"],
            "parent_frozen",
            all(not parameter.requires_grad for parameter in parent.parameters()),
            sum(1 for parameter in parent.parameters() if parameter.requires_grad),
        )

        train_world: FixtureWorld = build_split("train", PRICE_TRAIN_WORLDS)[0]
        dev_world: FixtureWorld = build_split("dev", PRICE_DEV_WORLDS)[0]
        train_rows = price_train_examples(train_world)
        dev_rows = price_dev_examples(dev_world)
        report["identity"]["fixture"] = {
            "digest": fixture_digest((train_world, dev_world)),
            "declared_worlds": {"train": 16, "dev": 8, "confirm": 8},
            "generated_worlds": {"train": PRICE_TRAIN_WORLDS, "dev": PRICE_DEV_WORLDS},
            "train_world": train_world.summary(),
            "dev_world": dev_world.summary(),
            "train_examples": [
                {
                    "index": row.index,
                    "layer": row.layer,
                    "goal": row.goal,
                    "object_text": row.object_text,
                    "place_text": row.place_text,
                    "action": row.action,
                    "target": row.target,
                    "action_count": len(row.actions),
                }
                for row in train_rows
            ],
            "dev_examples": [
                {
                    "index": row.index,
                    "layer": row.layer,
                    "goal": row.goal,
                    "target": row.target,
                }
                for row in dev_rows
            ],
        }
        check(
            report["checks"],
            "fixture_declared_lengths",
            len(train_rows) == PRICE_TRAIN_EXAMPLES
            and len(dev_rows) == PRICE_DEV_EXAMPLES
            and len(train_world.records) == TRAIN_RECORD_LINES
            and sum(len(record.raw) for record in train_world.records) == TRAIN_RECORD_BYTES
            and len(dev_world.records) == DEV_RECORD_LINES
            and sum(len(record.raw) for record in dev_world.records) == DEV_RECORD_BYTES
            and all(len(row.goal.encode()) == GOAL_BYTES for row in train_rows + dev_rows)
            and all(len(row.action.encode()) == ACTION_BYTES for row in train_rows + dev_rows),
            {
                "train_examples": len(train_rows),
                "dev_examples": len(dev_rows),
                "train_records": [
                    len(train_world.records),
                    sum(len(record.raw) for record in train_world.records),
                ],
                "dev_records": [
                    len(dev_world.records),
                    sum(len(record.raw) for record in dev_world.records),
                ],
            },
        )

        config = BindingConfig(seed=PRICE_BINDING_SEED)
        base = build_binding(config, torch.float64)
        module_j = copy.deepcopy(base)
        set_trainable(module_j, tuple(PARAMETER_GROUPS))
        module_s = copy.deepcopy(base)
        set_trainable(module_s, S_TRAINABLE_GROUPS)
        arms = (
            ("J_joint", module_j, tuple(PARAMETER_GROUPS)),
            ("S_stop_read", module_s, S_TRAINABLE_GROUPS),
        )
        report["module"] = {
            "name": "QueryBindingCandidate",
            "config": asdict(config),
            "dtype": "float64",
            "parameter_count": module_j.parameter_count(),
            "parameter_groups": {
                group: sum(getattr(module_j, name).numel() for name in PARAMETER_GROUPS[group])
                for group in PARAMETER_GROUPS
            },
            "parameter_digest_initial": module_parameter_digest(module_j),
            "s_trainable_groups": list(S_TRAINABLE_GROUPS),
            "s_trainable_count": sum(
                parameter.numel()
                for parameter in trainable_parameters(module_s, S_TRAINABLE_GROUPS)
            ),
            "internal_mode": BINDING_MODE,
            "fast_write": "disabled",
            "optimizer": {
                "name": "Adam",
                "lr": DEFAULT_LR,
                "betas": list(DEFAULT_BETAS),
                "eps": DEFAULT_EPS,
                "weight_decay": 0.0,
                "clip_norm": CLIP_NORM,
                "batch": PRICE_TRAIN_EXAMPLES,
            },
        }
        j_count = sum(
            parameter.numel()
            for parameter in trainable_parameters(module_j, tuple(PARAMETER_GROUPS))
        )
        s_count = report["module"]["s_trainable_count"]
        check(
            report["checks"],
            "arm_parameter_counts",
            j_count == 105760 and s_count == 50304,
            {"J_joint": j_count, "S_stop_read": s_count},
        )

        state0 = parent.initial_state(PRICE_SUBJECT)
        first_payload = price_example_payload(train_rows[0])
        cursor_features = price_features(parent, state0, first_payload, gates, identity)
        forward_j = price_forward(module_j, parent, cursor_features, train_rows[0].target, gates)
        forward_s = price_forward(module_s, parent, cursor_features, train_rows[0].target, gates)
        s_equal = max_abs_diff(forward_j["scores"], forward_s["scores"]) == 0.0 and (
            max_abs_diff(forward_j["h"], forward_s["h"]) == 0.0
        )
        check(
            report["checks"],
            "arm_initial_forward_bitwise_equal",
            s_equal,
            {
                "scores_diff": max_abs_diff(forward_j["scores"], forward_s["scores"]),
                "h_diff": max_abs_diff(forward_j["h"], forward_s["h"]),
            },
        )
        check(
            report["checks"],
            "initial_forward_finite",
            bool(
                torch.isfinite(forward_j["scores"]).all()
                and torch.isfinite(forward_j["loss"]).all()
                and torch.isfinite(forward_j["h"]).all()
            )
            and abs(float(forward_j["probabilities"].detach().sum()) - 1.0) < 1e-9,
            {
                "loss": float(forward_j["loss"].detach()),
                "probability_sum": float(forward_j["probabilities"].detach().sum()),
            },
        )

        # One-off cache of the frozen parent features. Legal to cache by contract; the
        # batch itself is timed without this, and the one-off cost is reported apart.
        features_start = time.process_time()
        train_features = []
        for row in train_rows:
            state = parent.initial_state(PRICE_SUBJECT)
            train_features.append(
                price_features(parent, state, price_example_payload(row), gates, identity)
            )
        feature_cache_cpu = time.process_time() - features_start
        train_features_tuple = tuple(train_features)
        report["unit_prices"]["cache_once"] = {
            "cpu_seconds": feature_cache_cpu,
            "examples": PRICE_TRAIN_EXAMPLES,
            "cpu_per_example": feature_cache_cpu / PRICE_TRAIN_EXAMPLES,
            "caliber": "one-off frozen-parent feature build for the 8 declared training "
            "examples; identical across arms and steps, so paid once, not per step",
        }

        output_dir = ROOT / PRICE_OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        restore_rows: dict[str, Any] = {}
        save_cpu: dict[str, float] = {}
        for arm_name, module, groups in arms:
            optimizer = make_optimizer(module, groups)
            bundle_path = output_dir / f"{arm_name}_step0.pt"
            save_start = time.process_time()
            record = save_binding_bundle(
                bundle_path,
                arm_name,
                parent,
                state0,
                module,
                optimizer,
                {"fixture_index": 0, "position": 0, "stage": "10.21.3-price"},
                groups,
                gates_table,
            )
            save_cpu[arm_name] = time.process_time() - save_start
            bundle_records.append(record)
            temporary.append(bundle_path)
            result_path = output_dir / f"{arm_name}_step0_restore.pt"
            temporary.append(result_path)
            wait_start = time.perf_counter()
            worker_result = spawn_worker(
                price_restore_worker,
                {
                    "bundle": str(bundle_path),
                    "result": str(result_path),
                    "example": first_payload,
                    "identity": identity,
                    "dtype": dtype_name(torch.float64),
                },
            )
            parent_wall_wait = time.perf_counter() - wait_start
            worker_cpu += float(worker_result.get("cpu_seconds", 0.0) or 0.0)
            worker_wall += float(worker_result.get("wall_seconds", 0.0) or 0.0)
            parent_forward = forward_j if arm_name == "J_joint" else forward_s
            loss_equal_abs = (
                abs(float(parent_forward["loss"].detach()) - float(worker_result.get("loss", 0.0)))
                if worker_result.get("ok")
                else float("inf")
            )
            restore_row = {
                "arm": arm_name,
                "ok": bool(worker_result.get("ok")),
                "message": worker_result.get("message"),
                "features_sha_equal": worker_result.get("features_sha")
                == features_sha(cursor_features),
                "scores_bitwise_equal": price_equal(
                    parent_forward["scores"].detach(), worker_result.get("scores")
                ),
                "h_bitwise_equal": price_equal(
                    parent_forward["h"].detach(), worker_result.get("h")
                ),
                "loss_abs_diff": loss_equal_abs,
                "child_cpu_seconds": float(worker_result.get("cpu_seconds", 0.0) or 0.0),
                "child_wall_seconds": float(worker_result.get("wall_seconds", 0.0) or 0.0),
                "child_load_cpu_seconds": float(worker_result.get("load_cpu_seconds", 0.0) or 0.0),
                "child_features_cpu_seconds": float(
                    worker_result.get("features_cpu_seconds", 0.0) or 0.0
                ),
                "child_forward_cpu_seconds": float(
                    worker_result.get("forward_cpu_seconds", 0.0) or 0.0
                ),
                "parent_wall_wait_seconds": parent_wall_wait,
                "bundle_bytes": record["bytes"],
                "bundle_sha256": record["sha256"],
            }
            restore_row["passed"] = bool(
                restore_row["ok"]
                and restore_row["features_sha_equal"]
                and restore_row["scores_bitwise_equal"]
                and restore_row["h_bitwise_equal"]
                and restore_row["loss_abs_diff"] <= 1e-12
            )
            restore_rows[arm_name] = restore_row
            check(report["checks"], f"restore_{arm_name}", restore_row["passed"], restore_row)

        read_start = time.process_time()
        torch.load(output_dir / "J_joint_step0.pt", map_location="cpu", weights_only=True)
        read_cpu = time.process_time() - read_start
        bundle_mib = bundle_records[0]["bytes"] / 1024**2
        bundle_io_cpu = save_cpu["J_joint"] + read_cpu
        report["unit_prices"]["cpu_per_bundle_MiB"] = {
            "value": bundle_io_cpu / bundle_mib,
            "caliber": "one save_binding_bundle write plus one torch.load read of the same "
            "new bundle, divided by its MiB; parent-side process_time only",
            "bundle_mib": bundle_mib,
            "write_cpu_seconds": save_cpu["J_joint"],
            "read_cpu_seconds": read_cpu,
            "total_cpu_seconds": bundle_io_cpu,
            "write_cpu_seconds_by_arm": dict(save_cpu),
        }
        report["restore"] = {
            "arms": restore_rows,
            "same_bundle_restored": all(row["passed"] for row in restore_rows.values()),
            "child_cpu_seconds": worker_cpu,
            "child_wall_seconds": worker_wall,
            "note": "child self-reported cpu is the unit price; the parent-side wait is "
            "reported separately and never substituted for it",
        }
        if not report["restore"]["same_bundle_restored"]:
            raise PriceStop("independent restore did not reproduce the parent forward")

        train_step_cpu: dict[str, list[float]] = {}
        dev_readings: dict[str, Any] = {}
        for arm_name, module, groups in arms:
            optimizer = make_optimizer(module, groups)
            trainable = trainable_parameters(module, groups)
            digest_before = module_parameter_digest(module)
            steps = []
            for step_index in range(PRICE_TRAIN_STEPS):
                step = price_train_step(
                    module,
                    parent,
                    train_rows,
                    train_features_tuple,
                    optimizer,
                    trainable,
                    gates,
                )
                optimizer_calls += 1
                step["index"] = step_index
                steps.append(step)
            values = [step["cpu_seconds"] for step in steps]
            train_step_cpu[arm_name] = values
            parent_grads = sum(1 for parameter in parent.parameters() if parameter.grad is not None)
            reading = price_dev_reading(module, parent, dev_rows, gates)
            dev_readings[arm_name] = reading
            report["arms"][arm_name] = {
                "trainable_groups": list(groups),
                "trainable_parameter_count": sum(
                    parameter.numel() for parameter in trainable_parameters(module, groups)
                ),
                "parameter_digest_before": digest_before,
                "parameter_digest_after": module_parameter_digest(module),
                "steps": [
                    {
                        "index": step["index"],
                        "cpu_seconds": step["cpu_seconds"],
                        "loss": step["loss"],
                        "clip_scale": step["clip_scale"],
                        "batch_losses": step["batch_losses"],
                    }
                    for step in steps
                ],
                "train_step_cpu_seconds": {
                    "values": values,
                    "mean": statistics.fmean(values),
                    "median": statistics.median(values),
                },
                "restore": restore_rows[arm_name],
                "parent_gradient_tensors": parent_grads,
                "development_reading": {
                    "count": reading["count"],
                    "correct": reading["correct"],
                    "cpu_per_example": reading["cpu_seconds"] / reading["count"],
                },
            }
            check(
                report["checks"],
                f"parent_no_grad_{arm_name}",
                parent_grads == 0,
                parent_grads,
            )
            check(
                report["checks"],
                f"step_loss_finite_{arm_name}",
                all(step["loss"] == step["loss"] for step in steps),
                [step["loss"] for step in steps],
            )

        j_values = train_step_cpu["J_joint"]
        j_caliber = "J_joint, mean of the two measured 8-example steps on cached frozen features"
        report["unit_prices"]["cpu_per_train_step"] = {
            "value": statistics.fmean(j_values),
            "caliber": j_caliber,
            "median": statistics.median(j_values),
            "values": j_values,
            "by_arm": {
                arm: {
                    "mean": statistics.fmean(values),
                    "median": statistics.median(values),
                    "values": values,
                }
                for arm, values in train_step_cpu.items()
            },
            "batch": PRICE_TRAIN_EXAMPLES,
            "included": "8 x (forward + binding step + score + loss), mean, backward, "
            "global grad clip 1.0, Adam(1e-3)",
            "excluded": "frozen-parent feature build (reported as cache_once)",
        }
        restore_values = [row["child_cpu_seconds"] for row in restore_rows.values()]
        report["unit_prices"]["cpu_per_restore"] = {
            "value": statistics.fmean(restore_values),
            "caliber": "child self-reported cpu_seconds for load_binding_bundle plus one "
            "forward on features rebuilt from the raw goal/records; mean over the two arms",
            "by_arm": {
                arm: {
                    "child_cpu_seconds": row["child_cpu_seconds"],
                    "child_load_cpu_seconds": row["child_load_cpu_seconds"],
                    "child_features_cpu_seconds": row["child_features_cpu_seconds"],
                    "child_forward_cpu_seconds": row["child_forward_cpu_seconds"],
                    "parent_wall_wait_seconds": row["parent_wall_wait_seconds"],
                }
                for arm, row in restore_rows.items()
            },
            "parent_wall_wait_seconds": [
                row["parent_wall_wait_seconds"] for row in restore_rows.values()
            ],
        }
        eval_caliber = (
            "J_joint mean over the 8 declared dev examples of: fresh zero state, frozen "
            "feature build, module forward, mode-6 binding step, rank-4 score, under no_grad"
        )
        j_dev = dev_readings["J_joint"]
        report["unit_prices"]["cpu_per_eval_example"] = {
            "value": j_dev["cpu_seconds"] / j_dev["count"],
            "caliber": eval_caliber,
            "by_arm": {
                arm: {
                    "cpu_seconds": reading["cpu_seconds"],
                    "count": reading["count"],
                    "cpu_per_example": reading["cpu_seconds"] / reading["count"],
                    "features_cpu_seconds": reading["features_cpu_seconds"],
                    "forward_cpu_seconds": reading["forward_cpu_seconds"],
                    "correct": reading["correct"],
                }
                for arm, reading in dev_readings.items()
            },
        }
        report["consumed_development_feedback"] = {
            "count": PRICE_DEV_EXAMPLES,
            "note": "not a gate; does not tune hyperparameters",
            "world": "dev[0]",
            "layers": dev_world.summary()["layers"],
            "examples": [
                {
                    "index": row.index,
                    "layer": row.layer,
                    "goal": row.goal,
                    "target": row.target,
                }
                for row in dev_rows
            ],
            "readings": {arm: reading["readings"] for arm, reading in dev_readings.items()},
            "correct_by_arm": {arm: reading["correct"] for arm, reading in dev_readings.items()},
            "used_as_gate": False,
            "hyperparameters_changed": False,
        }
        report["optimizer_calls"] = optimizer_calls
        report["effective_cursor"] = {
            "examples_per_step": PRICE_TRAIN_EXAMPLES,
            "steps_per_arm": PRICE_TRAIN_STEPS,
            "arms": len(arms),
            "examples_consumed": PRICE_TRAIN_EXAMPLES * PRICE_TRAIN_STEPS * len(arms),
            "declared_world": "train[0]",
            "next_example_index": 0,
            "note": "the declared batch repeats; the cursor and optimizer_calls are "
            "recorded apart, and no arm consumed a different example than declared",
        }

        unit = report["unit_prices"]
        inputs = report["extrapolation"]["inputs"]
        bundle_io_mib = inputs["bundle_cycles"] * bundle_mib
        terms = {
            "train": unit["cpu_per_train_step"]["value"] * inputs["train_steps"],
            "restore": unit["cpu_per_restore"]["value"] * inputs["restore_count"],
            "eval": unit["cpu_per_eval_example"]["value"] * inputs["eval_example_count"],
            "bundle_io": unit["cpu_per_bundle_MiB"]["value"] * bundle_io_mib,
        }
        total = sum(terms.values())
        # 16 train worlds x 12 kept combinations per seed. Arm-local caches are the
        # upper bound; this run shared one cache because frozen features are arm-free.
        train_examples_full = 192
        cache_per_arm = unit["cache_once"]["cpu_per_example"] * train_examples_full
        one_off_cache = cache_per_arm * inputs["arms"]
        report["extrapolation"].update(
            {
                "bundle_io_MiB": bundle_io_mib,
                "unit_prices_used": {
                    "cpu_per_train_step": unit["cpu_per_train_step"]["value"],
                    "cpu_per_restore": unit["cpu_per_restore"]["value"],
                    "cpu_per_eval_example": unit["cpu_per_eval_example"]["value"],
                    "cpu_per_bundle_MiB": unit["cpu_per_bundle_MiB"]["value"],
                },
                "terms": terms,
                "total_cpu_seconds": total,
                "cap_cpu_seconds": PRICE_CAP_CPU_SECONDS,
                "within_cap": bool(total <= PRICE_CAP_CPU_SECONDS),
                "not_in_formula": {
                    "why": "the contract formula has four terms and prices the cached-feature "
                    "step; the one-off frozen-parent feature cache is priced apart here",
                    "one_off_feature_cache_cpu_seconds_upper_bound": one_off_cache,
                    "one_off_feature_cache_cpu_seconds_shared": cache_per_arm,
                    "upper_bound_assumes": "each arm builds its own cache for "
                    f"{train_examples_full} training examples; the measured run shared one "
                    "cache because frozen features do not depend on the arm",
                    "total_including_cache": total + one_off_cache,
                    "within_cap_including_cache": bool(
                        total + one_off_cache <= PRICE_CAP_CPU_SECONDS
                    ),
                },
                "conservative_choices": [
                    "cpu_per_eval_example includes the frozen-parent feature rebuild, so the "
                    "eval term charges 1280 rebuilds even though the two evaluation points "
                    "per arm could reuse one cache; the real cost can only be lower",
                    "cpu_per_restore is a cold child process that reloads the 77.6MB parent "
                    "bundle, which is the honest cost of a recovery",
                    "cpu_per_train_step is the J arm (all 11 groups), the larger of the two "
                    "arms, and it excludes feature build",
                ],
            }
        )
        parent_sha_after = sha256_text(parent_path)
        parent_digest_after = parameter_digest(
            {name: parameter.detach() for name, parameter in parent.named_parameters()}
        )
        manifest_payload = price_manifest_payload(
            parent_manifest, source_hashes, report, (train_world, dev_world)
        )
        manifest_path_price = ROOT / PRICE_MANIFEST
        price_manifest_written = False
        if manifest_path_price.exists():
            stored = json.loads(manifest_path_price.read_text(encoding="utf-8"))
            price_manifest_drift = {
                "module_digest": stored.get("module", {}).get("parameter_digest_initial")
                != manifest_payload["module"]["parameter_digest_initial"],
                "fixture_digest": stored.get("fixture", {}).get("digest")
                != manifest_payload["fixture"]["digest"],
                "source_hashes": stored.get("source_hashes") != manifest_payload["source_hashes"],
                "parent": stored.get("parent") != manifest_payload["parent"],
            }
            price_manifest_ok = not any(price_manifest_drift.values())
        else:
            manifest_path_price.parent.mkdir(parents=True, exist_ok=True)
            manifest_path_price.write_text(
                json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            price_manifest_drift = {}
            price_manifest_ok = True
            price_manifest_written = True
        report["manifest"] = {
            "path": PRICE_MANIFEST,
            "sha256": sha256_text(manifest_path_price),
            "written": price_manifest_written,
            "drift": price_manifest_drift,
        }
        check(
            report["checks"],
            "manifest_identity",
            price_manifest_ok,
            {"written": price_manifest_written, "drift": price_manifest_drift},
        )
        check(
            report["checks"],
            "parent_bundle_unchanged",
            parent_sha_before == parent_sha_after == EXPECTED_BUNDLE_SHA,
            {"before": parent_sha_before, "after": parent_sha_after},
        )
        check(
            report["checks"],
            "parent_parameters_unchanged",
            parent_digest_after == EXPECTED_CANDIDATE,
            parent_digest_after,
        )
        product_bytes = sum(float(record["bytes"]) for record in bundle_records)
        check(
            report["checks"],
            "product_bytes_within_budget",
            product_bytes <= PRICE_MAX_PRODUCT_BYTES,
            {"bundle_bytes": product_bytes, "limit": PRICE_MAX_PRODUCT_BYTES},
        )
        report["cost"]["bundle_records"] = bundle_records
        report["cost"]["recorded_bundle_bytes"] = product_bytes
    except PriceStop as exc:
        report["status"] = "engineering_failure"
        report["stopped_before_pricing"] = str(exc)
    except Exception as exc:  # noqa: BLE001
        report["status"] = "engineering_failure"
        report["engineering_failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)
        if temporary:
            report["cost"]["temporary_bundles_deleted"] = len(temporary)

    cpu_seconds = time.process_time() - started[0]
    wall_seconds = time.perf_counter() - started[1]
    total_cpu = cpu_seconds + worker_cpu
    over_budget = total_cpu > PRICE_BUDGET_CPU_SECONDS or wall_seconds > PRICE_BUDGET_WALL_SECONDS
    check(
        report["checks"],
        "budget",
        not over_budget,
        {
            "cpu_seconds": total_cpu,
            "wall_seconds": wall_seconds,
            "cpu_budget": PRICE_BUDGET_CPU_SECONDS,
            "wall_budget": PRICE_BUDGET_WALL_SECONDS,
            "child_cpu_seconds": worker_cpu,
            "child_wall_seconds": worker_wall,
        },
    )
    report["cost"].update(
        {
            "cpu_seconds": total_cpu,
            "cpu_seconds_parent": cpu_seconds,
            "cpu_seconds_child": worker_cpu,
            "wall_seconds": wall_seconds,
            "wall_seconds_child": worker_wall,
            "budget": {
                "cpu_seconds": PRICE_BUDGET_CPU_SECONDS,
                "wall_seconds": PRICE_BUDGET_WALL_SECONDS,
                "products_bytes": PRICE_MAX_PRODUCT_BYTES,
            },
        }
    )
    failed = [row["id"] for row in report["checks"] if not row["passed"] and row["id"] != "budget"]
    within_cap = report["extrapolation"].get("within_cap")
    stopped = bool(report.get("engineering_failure") or report.get("stopped_before_pricing"))
    if stopped or failed:
        report["status"] = "engineering_failure"
        report["failed_checks"] = failed
    elif over_budget:
        report["status"] = "budget_stop"
    elif within_cap is False:
        report["status"] = "budget_not_admitted"
        report["not_admitted"] = {
            "reason": "extrapolated total exceeds the declared 10.21.4 cap",
            "total_cpu_seconds": report["extrapolation"].get("total_cpu_seconds"),
            "cap_cpu_seconds": PRICE_CAP_CPU_SECONDS,
            "not_done": [
                "no budget raise requested",
                "no sample reduction",
                "no change of step count",
                "no automatic re-run",
            ],
        }
    else:
        report["status"] = "priced"
    report["capability_claimed"] = False
    report["training_admitted"] = False
    report["comparison_run"] = False
    report["weights_published"] = False
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    print(
        json.dumps(
            {
                "status": report["status"],
                "failed": report.get("failed_checks", []),
                "optimizer_calls": report["optimizer_calls"],
                "cpu_seconds": round(total_cpu, 3),
                "wall_seconds": round(wall_seconds, 3),
                "within_cap": within_cap,
            },
            ensure_ascii=True,
        )
    )
    return 0


class BudgetStop(RuntimeError):
    """The declared wall/CPU budget ran out mid-run; no further steps are taken."""


def compare_batch_order(
    order_seed: int, total: int, batch: int, steps: int
) -> tuple[tuple[int, ...], ...]:
    """Deterministic stream of batches: epoch-shuffled, no replacement inside an epoch."""
    batches: list[tuple[int, ...]] = []
    epoch = 0
    while len(batches) < steps:
        order = list(range(total))
        random.Random(f"{order_seed}|{epoch}").shuffle(order)
        for start in range(0, total - batch + 1, batch):
            batches.append(tuple(order[start : start + batch]))
            if len(batches) == steps:
                break
        epoch += 1
    return tuple(batches)


def compare_source_attention(module, features: BindingFeatures):
    """Recompute the source softmax to read which record the query actually selects.

    The read is replicated from the module's own submodules and then checked bitwise
    against the vector ``forward_binding`` returns, so a diagnostic that disagrees with
    the live read is reported as invalid instead of being trusted.
    """
    c = module.config
    m = module.goal_read(features.goal_features, features.s)
    query_input = torch.cat([m, features.s.unsqueeze(0).expand(c.roles, -1), module.role], dim=-1)
    query = query_input @ module.Q_source.T + module.Q_source_bias
    key_value_input = torch.cat([features.source_positions, features.source_metadata], dim=-1)
    keys = key_value_input @ module.K_source.T
    values = key_value_input @ module.V_source.T + module.V_source_bias
    all_keys = torch.cat([keys, module.source_null_key.unsqueeze(0)], dim=0)
    all_values = torch.cat([values, module.source_null_value.unsqueeze(0)], dim=0)
    scores = query @ all_keys.T / math.sqrt(c.query_width)
    legal = torch.cat(
        [features.source_mask.to(torch.bool), features.source_mask.new_ones(1, dtype=torch.bool)]
    )
    scores = scores.masked_fill(~legal.unsqueeze(0), float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights, weights @ all_values


def compare_evaluate(
    module,
    parent,
    rows: tuple,
    features_list: tuple,
    gates,
    with_reader: bool = True,
) -> dict:
    """Full 20-way top-1 on the declared dev set, under no_grad, at one checkpoint."""
    readings = []
    reader_hits = {"object": 0, "place": 0, "examples": 0}
    reader_invalid = 0
    reader_null_mass = 0.0
    with torch.no_grad():
        for row, features in zip(rows, features_list, strict=True):
            b_goal, b_actions, content = module.forward_binding(features)
            z_bind, _h, _output = binding_step(parent, features, b_goal, gates)
            scores, _probabilities = score_actions(module, z_bind, b_actions)
            prediction = int(torch.argmax(scores).item())
            record = {
                "index": row.index,
                "world": row.world_index,
                "layer": row.layer,
                "target": row.target,
                "prediction": prediction,
                "correct": bool(prediction == row.target),
                "loss": float(binding_loss(scores, row.target).detach()),
            }
            if with_reader:
                weights, rebuilt = compare_source_attention(module, features)
                if not price_equal(content, rebuilt):
                    reader_invalid += 1
                else:
                    count = len(features.records)
                    object_raw = f"{row.object_description}是{row.object_text}。".encode()
                    place_raw = f"{row.place_description}是{row.place_text}。".encode()
                    object_slot = [i for i, r in enumerate(features.records) if r.raw == object_raw]
                    place_slot = [i for i, r in enumerate(features.records) if r.raw == place_raw]
                    if (
                        object_slot
                        and int(torch.argmax(weights[0, :count]).item()) == object_slot[0]
                    ):
                        reader_hits["object"] += 1
                    if place_slot and int(torch.argmax(weights[1, :count]).item()) == place_slot[0]:
                        reader_hits["place"] += 1
                    reader_hits["examples"] += 1
                    reader_null_mass += float(weights[:, count].mean())
            readings.append(record)
    correct = sum(1 for item in readings if item["correct"])
    aggregate: dict[str, Any] = {
        "count": len(readings),
        "correct": correct,
        "top1": correct / len(readings),
        "layers": {},
        "worlds": {},
    }
    for layer in COMPARE_LAYERS:
        subset = [item for item in readings if item["layer"] == layer]
        hits = sum(1 for item in subset if item["correct"])
        aggregate["layers"][layer] = {
            "count": len(subset),
            "correct": hits,
            "top1": hits / len(subset) if subset else 0.0,
        }
    for world in sorted({item["world"] for item in readings}):
        subset = [item for item in readings if item["world"] == world]
        hits = sum(1 for item in subset if item["correct"])
        aggregate["worlds"][str(world)] = {
            "count": len(subset),
            "correct": hits,
            "top1": hits / len(subset) if subset else 0.0,
        }
    if with_reader:
        examples = max(1, reader_hits["examples"])
        aggregate["reader"] = {
            "object_source_top1": reader_hits["object"] / examples,
            "place_source_top1": reader_hits["place"] / examples,
            "mean_source_top1": (reader_hits["object"] + reader_hits["place"]) / (2 * examples),
            "examples": reader_hits["examples"],
            "invalid_replications": reader_invalid,
            "null_attention_mass": reader_null_mass / examples,
            "chance_source_top1": 1.0 / (len(rows[0].records) + 1),
            "bar": COMPARE_READER_BAR,
            "selection_present": (reader_hits["object"] + reader_hits["place"]) / (2 * examples)
            >= COMPARE_READER_BAR,
            "valid": reader_invalid == 0,
        }
    aggregate["readings"] = readings
    return aggregate


def compare_baselines(rows: tuple) -> dict:
    """Chance and best constant computed on the actual candidate distribution."""
    candidates = len(rows[0].actions)
    counts: dict[int, int] = {}
    for row in rows:
        counts[row.target] = counts.get(row.target, 0) + 1
    best_index = max(counts, key=lambda key: counts[key])
    payload: dict[str, Any] = {
        "candidates": candidates,
        "chance": 1.0 / candidates,
        "chance_note": "uniform over the full candidate list, not over a held-out subset",
        "best_constant_index": best_index,
        "best_constant": counts[best_index] / len(rows),
        "target_index_counts": {str(key): counts[key] for key in sorted(counts)},
        "layers": {},
    }
    for layer in COMPARE_LAYERS:
        subset = [row for row in rows if row.layer == layer]
        layer_counts: dict[int, int] = {}
        for row in subset:
            layer_counts[row.target] = layer_counts.get(row.target, 0) + 1
        layer_best = max(layer_counts, key=lambda key: layer_counts[key])
        payload["layers"][layer] = {
            "count": len(subset),
            "chance": 1.0 / candidates,
            "best_constant": layer_counts[layer_best] / len(subset),
            "best_constant_index": layer_best,
        }
    return payload


def compare_bootstrap(rows: tuple, j_readings: list, s_readings: list, stat_seed: int) -> dict:
    """World-paired bootstrap of the J-S top-1 difference, per seed."""
    worlds = sorted({row.world_index for row in rows})
    diffs = []
    for world in worlds:
        positions = [i for i, row in enumerate(rows) if row.world_index == world]
        j_hits = sum(1 for i in positions if j_readings[i]["correct"])
        s_hits = sum(1 for i in positions if s_readings[i]["correct"])
        diffs.append((j_hits - s_hits) / len(positions))
    rng = random.Random(stat_seed)
    count = len(diffs)
    means = []
    for _ in range(COMPARE_BOOTSTRAP_DRAWS):
        total = 0.0
        for _ in range(count):
            total += diffs[rng.randrange(count)]
        means.append(total / count)
    means.sort()
    lower = means[int(0.025 * COMPARE_BOOTSTRAP_DRAWS)]
    upper = means[min(COMPARE_BOOTSTRAP_DRAWS - 1, int(0.975 * COMPARE_BOOTSTRAP_DRAWS))]
    return {
        "unit": "world",
        "worlds": count,
        "per_world_diffs": diffs,
        "observed_mean_diff": sum(diffs) / count,
        "draws": COMPARE_BOOTSTRAP_DRAWS,
        "stat_seed": stat_seed,
        "percentile": "2.5/97.5 nearest-rank on the sorted bootstrap means",
        "ci95": [lower, upper],
        "lower_bound": lower,
        "upper_bound": upper,
        "lower_bound_positive": lower > 0.0,
        "seeds_pooled": False,
    }


def compare_peak_ram_mb() -> float | None:
    """Peak working set of this process, Windows psapi; None when unavailable."""
    try:
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.GetCurrentProcess.argtypes = []
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]
        handle = kernel32.GetCurrentProcess()
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return None
        return float(counters.PeakWorkingSetSize) / 1024**2
    except Exception:  # noqa: BLE001
        return None


def compare_manifest_payload(
    parent_manifest: dict, source_hashes: dict, report: dict, worlds: tuple
) -> dict:
    return {
        "schema": "taiji-composition-compare-manifest-v1",
        "step": "10.21.4",
        "date": COMPARE_DATE,
        "parent": {
            "candidate_id": parent_manifest["candidate_id"],
            "bundle": BUNDLE,
            "bundle_sha256": parent_manifest["provenance"]["sha256"],
            "bundle_bytes": parent_manifest["provenance"]["bytes"],
        },
        "seeds": [
            {
                "binding_seed": binding,
                "data_order_seed": order,
                "main_candidate": binding == COMPARE_MAIN_SEED,
            }
            for binding, order in COMPARE_SEEDS
        ],
        "arms": {
            "J_joint": list(PARAMETER_GROUPS),
            "S_stop_read": list(S_TRAINABLE_GROUPS),
        },
        "design": report["design"],
        "module_digests": {
            "by_binding_seed": report["identity"]["module_digests_by_seed"],
        },
        "fixture": composition_payload(worlds),
        "source_hashes": {relative: source_hashes[relative] for relative in COMPARE_SOURCE_FILES},
        "budget": {
            "cpu_seconds": COMPARE_BUDGET_CPU_SECONDS,
            "wall_seconds": COMPARE_BUDGET_WALL_SECONDS,
            "products_bytes": COMPARE_MAX_PRODUCT_BYTES,
        },
        "verdicts": {
            "by_binding_seed": {
                str(seed): report["verdicts"]["by_binding_seed"].get(str(seed), {}).get("summary")
                for seed in (COMPARE_MAIN_SEED, 17022)
            },
            "status": report["status"],
        },
        "exit_states": list(COMPARE_EXIT_STATES),
        "checkpoints": [record["path"] for record in report["cost"].get("checkpoints", [])],
        "claim": "declared-fixture mechanism comparison only; not P2/L3, not free-form language",
    }


def compare_main(args) -> int:
    """10.21.4: two binding seeds x two arms, 300 steps each, two dev reads, one exit."""
    torch.set_num_threads(1)
    if args.output and args.output.exists():
        raise FileExistsError("10.21.4 compare card is write-once")
    started = (time.process_time(), time.perf_counter())
    report: dict[str, Any] = {
        "schema": COMPARE_SCHEMA,
        "step": "10.21.4",
        "date": COMPARE_DATE,
        "status": "engineering_failure",
        "exit_reason": "",
        "checks": [],
        "identity": {},
        "design": {},
        "data": {},
        "module": {},
        "training": {},
        "main_endpoint": {},
        "paired_stats": {},
        "verdicts": {},
        "cost": {},
        "limits": COMPARE_LIMITS,
    }
    parent_path = ROOT / BUNDLE
    parent_sha_before = sha256_text(parent_path)
    worker_cpu = 0.0
    worker_wall = 0.0
    optimizer_calls = 0
    checkpoints: list[dict] = []
    restore_rows: dict[str, Any] = {}
    cells: dict[str, Any] = {}
    stopped_at: str | None = None
    try:
        parent_manifest = build_manifest(ROOT, parent_path)
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text(
                json.dumps(parent_manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            _manifest, payload = load_candidate(ROOT, manifest_path, parent_path, "frozen_eval")
        parent, _parent_state, _ = resume_subject(payload)
        parent.eval()
        freeze_parent(parent)
        gates_table = payload["growth_gates"]
        gates = gates_provider(gates_table)
        source_hashes = {
            relative: sha256_text(ROOT / relative) for relative in COMPARE_SOURCE_FILES
        }
        identity = {
            "parent_candidate_id": parent_manifest["candidate_id"],
            "parent_bundle_sha256": parent_manifest["provenance"]["sha256"],
            "source_hashes": source_hashes,
        }
        report["identity"] = {
            "parent_candidate_id": parent_manifest["candidate_id"],
            "parent_bundle": {
                "path": BUNDLE,
                "sha256": parent_manifest["provenance"]["sha256"],
                "bytes": parent_manifest["provenance"]["bytes"],
            },
            "parent_subject_id": payload["subject_id"],
            "parent_revision": payload["revision"],
            "parent_step": payload["step"],
            "parent_config": payload["config"],
            "source_hashes": source_hashes,
        }
        report["design"] = {
            "seeds": [
                {
                    "binding_seed": binding,
                    "data_order_seed": order,
                    "main_candidate": binding == COMPARE_MAIN_SEED,
                    "role": "main" if binding == COMPARE_MAIN_SEED else "stability re-check only",
                }
                for binding, order in COMPARE_SEEDS
            ],
            "arms": {
                "J_joint": {
                    "trainable_groups": list(PARAMETER_GROUPS),
                    "trainable_parameter_count": 105760,
                },
                "S_stop_read": {
                    "trainable_groups": list(S_TRAINABLE_GROUPS),
                    "trainable_parameter_count": 50304,
                },
            },
            "steps_per_arm": COMPARE_STEPS,
            "batch": COMPARE_BATCH,
            "dtype": "float64",
            "optimizer": {
                "name": "Adam",
                "lr": DEFAULT_LR,
                "betas": list(DEFAULT_BETAS),
                "eps": DEFAULT_EPS,
                "weight_decay": 0.0,
                "clip_norm": CLIP_NORM,
                "no_lr_grid": True,
                "no_dev_based_stopping": True,
            },
            "threads": 1,
            "evaluation_points": [0, COMPARE_STEPS],
            "checkpoints": list(COMPARE_CHECKPOINTS),
            "fast_write": "disabled",
            "internal_mode": BINDING_MODE,
        }
        check(
            report["checks"],
            "parent_identity",
            parent_manifest["candidate_id"] == EXPECTED_CANDIDATE
            and parent_manifest["provenance"]["sha256"] == EXPECTED_BUNDLE_SHA
            and parent_manifest["provenance"]["bytes"] == EXPECTED_BUNDLE_BYTES,
            {"candidate": parent_manifest["candidate_id"]},
        )

        train_worlds = build_split("train", 16)
        dev_worlds = build_split("dev", 8)
        train_rows = tuple(row for world in train_worlds for row in world.examples)
        dev_rows = tuple(row for world in dev_worlds for row in world.examples)
        baselines = compare_baselines(dev_rows)
        report["data"] = {
            "train_worlds": len(train_worlds),
            "dev_worlds": len(dev_worlds),
            "confirm_worlds_generated": 0,
            "confirm_sealed": True,
            "train_examples": len(train_rows),
            "dev_examples": len(dev_rows),
            "train_candidates_per_world": len(train_worlds[0].actions()),
            "dev_candidates_per_world": len(dev_worlds[0].actions()),
            "train_combinations_per_world": 12,
            "layers": {
                layer: {
                    "count": sum(1 for row in dev_rows if row.layer == layer),
                    "per_world": sum(1 for row in dev_rows if row.layer == layer)
                    // len(dev_worlds),
                }
                for layer in COMPARE_LAYERS
            },
            "chance": baselines["chance"],
            "best_constant": baselines["best_constant"],
            "baselines": baselines,
            "fixture_digest": fixture_digest(train_worlds + dev_worlds),
            "lengths": {
                "goal_bytes": sorted({len(row.goal.encode()) for row in train_rows + dev_rows}),
                "action_bytes": sorted({len(row.action.encode()) for row in train_rows + dev_rows}),
                "train_records": [
                    len(train_worlds[0].records),
                    sum(len(r.raw) for r in train_worlds[0].records),
                ],
                "dev_records": [
                    len(dev_worlds[0].records),
                    sum(len(r.raw) for r in dev_worlds[0].records),
                ],
            },
        }
        layer_total = sum(item["count"] for item in report["data"]["layers"].values())
        check(
            report["checks"],
            "data_declared_shape",
            len(train_rows) == 192
            and len(dev_rows) == 160
            and layer_total == 160
            and len(train_worlds[0].actions()) == 15
            and len(dev_worlds[0].actions()) == 20
            and report["data"]["layers"]["seen_combination"]["count"] == 96
            and report["data"]["layers"]["unseen_combination"]["count"] == 24
            and report["data"]["layers"]["unseen_entity"]["count"] == 40,
            {
                "train": len(train_rows),
                "dev": len(dev_rows),
                "layers": {k: v["count"] for k, v in report["data"]["layers"].items()},
            },
        )
        check(
            report["checks"],
            "instrument_baselines",
            abs(baselines["chance"] - 0.05) < 1e-12
            and abs(baselines["best_constant"] - 0.05) < 1e-9,
            {"chance": baselines["chance"], "best_constant": baselines["best_constant"]},
        )

        module_digests = {}
        bases = {}
        for binding_seed, _order_seed in COMPARE_SEEDS:
            base = build_binding(BindingConfig(seed=binding_seed), torch.float64)
            bases[binding_seed] = base
            module_digests[str(binding_seed)] = module_parameter_digest(base)
        report["identity"]["module_digests_by_seed"] = module_digests
        check(
            report["checks"],
            "seed_module_digests_distinct",
            module_digests[str(COMPARE_SEEDS[0][0])] != module_digests[str(COMPARE_SEEDS[1][0])],
            module_digests,
        )

        features_start = time.process_time()
        dev_features = tuple(
            price_features(
                parent,
                parent.initial_state(COMPARE_SUBJECT),
                price_example_payload(row),
                gates,
                identity,
            )
            for row in dev_rows
        )
        train_features = tuple(
            price_features(
                parent,
                parent.initial_state(COMPARE_SUBJECT),
                price_example_payload(row),
                gates,
                identity,
            )
            for row in train_rows
        )
        report["cost"]["feature_cache_cpu_seconds"] = time.process_time() - features_start
        report["cost"]["cached_train_examples"] = len(train_features)
        report["cost"]["cached_dev_examples"] = len(dev_features)
        report["module"] = {
            "name": "QueryBindingCandidate",
            "parameter_count": bases[COMPARE_MAIN_SEED].parameter_count(),
            "config_main": asdict(BindingConfig(seed=COMPARE_MAIN_SEED)),
            "parameter_digest_by_seed": module_digests,
            "cache_policy": "frozen parent features cached; goal query, source values, role "
            "outputs, b_goal and the internal event to z_bind path stay live and are never cached",
        }

        output_dir = ROOT / COMPARE_OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        for binding_seed, order_seed in COMPARE_SEEDS:
            base = bases[binding_seed]
            order = compare_batch_order(order_seed, len(train_rows), COMPARE_BATCH, COMPARE_STEPS)
            state0 = parent.initial_state(COMPARE_SUBJECT)
            probe_features = price_features(
                parent, state0, price_example_payload(train_rows[0]), gates, identity
            )
            seed_cells: dict[str, Any] = {}
            arm_modules: dict[str, Any] = {}
            for arm_name, groups in (
                ("J_joint", tuple(PARAMETER_GROUPS)),
                ("S_stop_read", S_TRAINABLE_GROUPS),
            ):
                arm_module = copy.deepcopy(base)
                set_trainable(arm_module, groups)
                arm_modules[arm_name] = arm_module
            forward_j = price_forward(
                arm_modules["J_joint"], parent, probe_features, train_rows[0].target, gates
            )
            forward_s = price_forward(
                arm_modules["S_stop_read"], parent, probe_features, train_rows[0].target, gates
            )
            initial_equal = max_abs_diff(forward_j["scores"], forward_s["scores"]) == 0.0 and (
                max_abs_diff(forward_j["h"], forward_s["h"]) == 0.0
            )
            for arm_name, groups in (
                ("J_joint", tuple(PARAMETER_GROUPS)),
                ("S_stop_read", S_TRAINABLE_GROUPS),
            ):
                module = arm_modules[arm_name]
                optimizer = make_optimizer(module, groups)
                trainable = trainable_parameters(module, groups)
                profile = f"{binding_seed}_{arm_name}"
                losses: list[float] = []
                grad_norms: list[float] = []
                digest_before = module_parameter_digest(module)

                def save_point(
                    point: int,
                    consumed: int,
                    *,
                    profile: str = profile,
                    binding_seed: int = binding_seed,
                    order_seed: int = order_seed,
                    arm_name: str = arm_name,
                    groups: Any = groups,
                    state0: Any = state0,
                    module: Any = module,
                    optimizer: Any = optimizer,
                ) -> None:
                    bundle_path = output_dir / f"{profile}_step{point}.pt"
                    record = save_binding_bundle(
                        bundle_path,
                        profile,
                        parent,
                        state0,
                        module,
                        optimizer,
                        {
                            "binding_seed": binding_seed,
                            "data_order_seed": order_seed,
                            "arm": arm_name,
                            "step": point,
                            "examples_consumed": consumed,
                            "next_example_index": consumed % len(train_rows),
                            "epoch": consumed // len(train_rows),
                            "stage": "10.21.4-compare",
                        },
                        groups,
                        gates_table,
                    )
                    checkpoint = {
                        "path": record["path"],
                        "profile": profile,
                        "binding_seed": binding_seed,
                        "arm": arm_name,
                        "step": point,
                        "bytes": record["bytes"],
                        "sha256": record["sha256"],
                        "module_parameter_digest": module_parameter_digest(module),
                        "examples_consumed": consumed,
                        "next_example_index": consumed % len(train_rows),
                    }
                    loaded = torch.load(bundle_path, map_location="cpu", weights_only=True)
                    restored = QueryBindingCandidate(BindingConfig(**loaded["binding"]["config"]))
                    restored.load_state_dict(loaded["binding"]["parameters"], strict=True)
                    checkpoint["bundle_reload_digest_matches"] = (
                        module_parameter_digest(restored) == checkpoint["module_parameter_digest"]
                    )
                    checkpoint["cursor"] = loaded["cursor"]
                    checkpoints.append(checkpoint)
                    return None

                if 0 in COMPARE_CHECKPOINTS:
                    save_point(0, 0)
                step0_eval = compare_evaluate(module, parent, dev_rows, dev_features, gates)
                steps_done = 0
                for step_index in range(COMPARE_STEPS):
                    if step_index % 10 == 0:
                        used_cpu = time.process_time() - started[0] + worker_cpu
                        used_wall = time.perf_counter() - started[1]
                        if (
                            used_cpu > COMPARE_BUDGET_CPU_SECONDS
                            or used_wall > COMPARE_BUDGET_WALL_SECONDS
                        ):
                            stopped_at = (
                                f"budget exhausted before step {step_index} of {profile} "
                                f"(cpu {used_cpu:.1f}/{COMPARE_BUDGET_CPU_SECONDS}, "
                                f"wall {used_wall:.1f}/{COMPARE_BUDGET_WALL_SECONDS})"
                            )
                            raise BudgetStop(stopped_at)
                    batch = order[step_index]
                    rows = tuple(train_rows[index] for index in batch)
                    batch_features = tuple(train_features[index] for index in batch)
                    step = price_train_step(
                        module, parent, rows, batch_features, optimizer, trainable, gates
                    )
                    optimizer_calls += 1
                    steps_done += 1
                    losses.append(step["loss"])
                    grad_norms.append(step["clip_scale"])
                    if step_index + 1 in COMPARE_CHECKPOINTS:
                        save_point(step_index + 1, steps_done * COMPARE_BATCH)
                step300_eval = compare_evaluate(module, parent, dev_rows, dev_features, gates)
                cursor = {
                    "examples_consumed": steps_done * COMPARE_BATCH,
                    "next_example_index": (steps_done * COMPARE_BATCH) % len(train_rows),
                    "epoch": (steps_done * COMPARE_BATCH) // len(train_rows),
                    "steps_done": steps_done,
                    "order_seed": order_seed,
                }
                seed_cells[arm_name] = {
                    "module": module,
                    "groups": groups,
                    "digest_before": digest_before,
                    "digest_after": module_parameter_digest(module),
                    "trainable_parameter_count": sum(
                        parameter.numel() for parameter in trainable_parameters(module, groups)
                    ),
                    "step0": step0_eval,
                    "step300": step300_eval,
                    "losses": losses,
                    "grad_norms": grad_norms,
                    "cursor": cursor,
                    "steps_done": steps_done,
                }
                if profile == f"{COMPARE_MAIN_SEED}_J_joint":
                    bundle_path = output_dir / f"{profile}_step{COMPARE_STEPS}.pt"
                    if bundle_path.exists():
                        result_path = output_dir / f"{profile}_restore_probe.pt"
                        wait_start = time.perf_counter()
                        worker_result = spawn_worker(
                            price_restore_worker,
                            {
                                "bundle": str(bundle_path),
                                "result": str(result_path),
                                "example": price_example_payload(dev_rows[0]),
                                "identity": identity,
                                "dtype": dtype_name(torch.float64),
                            },
                        )
                        parent_wait = time.perf_counter() - wait_start
                        worker_cpu += float(worker_result.get("cpu_seconds", 0.0) or 0.0)
                        worker_wall += float(worker_result.get("wall_seconds", 0.0) or 0.0)
                        restore_rows["after_step300"] = {
                            "ok": bool(worker_result.get("ok")),
                            "parent_wall_wait_seconds": parent_wait,
                            "child_cpu_seconds": float(
                                worker_result.get("cpu_seconds", 0.0) or 0.0
                            ),
                        }
                        result_path.unlink(missing_ok=True)
            report["checks"].append(
                {
                    "id": f"arms_initial_forward_equal_{binding_seed}",
                    "passed": bool(initial_equal),
                    "observed": {"bitwise_equal": initial_equal},
                    "note": "",
                }
            )
            cells[str(binding_seed)] = seed_cells

        for binding_seed, _order_seed in COMPARE_SEEDS:
            seed_cells = cells[str(binding_seed)]
            training: dict[str, Any] = {}
            for arm_name in ("J_joint", "S_stop_read"):
                cell = seed_cells[arm_name]
                losses = cell["losses"]
                training[arm_name] = {
                    "trainable_parameter_count": cell["trainable_parameter_count"],
                    "parameter_digest_before": cell["digest_before"],
                    "parameter_digest_after": cell["digest_after"],
                    "steps_done": cell["steps_done"],
                    "loss_curve": losses,
                    "loss_first": losses[0] if losses else None,
                    "loss_last": losses[-1] if losses else None,
                    "loss_head": losses[:5],
                    "loss_tail": losses[-5:],
                    "grad_norm_head": cell["grad_norms"][:5],
                    "grad_norm_tail": cell["grad_norms"][-5:],
                    "grad_norm_finite": all(value == value for value in cell["grad_norms"]),
                    "optimizer_calls": cell["steps_done"],
                    "cursor": cell["cursor"],
                }
            report["training"][str(binding_seed)] = training
            endpoints = {}
            for arm_name in ("J_joint", "S_stop_read"):
                cell = seed_cells[arm_name]
                endpoints[arm_name] = {
                    "step0": {
                        "top1": cell["step0"]["top1"],
                        "correct": cell["step0"]["correct"],
                        "layers": {k: v["top1"] for k, v in cell["step0"]["layers"].items()},
                        "reader": cell["step0"].get("reader"),
                    },
                    "step300": {
                        "top1": cell["step300"]["top1"],
                        "correct": cell["step300"]["correct"],
                        "layers": {k: v["top1"] for k, v in cell["step300"]["layers"].items()},
                        "worlds": cell["step300"]["worlds"],
                        "reader": cell["step300"].get("reader"),
                    },
                }
            report["main_endpoint"][str(binding_seed)] = endpoints
            j_readings = seed_cells["J_joint"]["step300"]["readings"]
            s_readings = seed_cells["S_stop_read"]["step300"]["readings"]
            report["paired_stats"][str(binding_seed)] = compare_bootstrap(
                dev_rows, j_readings, s_readings, COMPARE_STAT_SEED
            )

        verdicts: dict[str, Any] = {"by_binding_seed": {}}
        for binding_seed, _order_seed in COMPARE_SEEDS:
            endpoints = report["main_endpoint"][str(binding_seed)]
            j = endpoints["J_joint"]["step300"]
            s = endpoints["S_stop_read"]["step300"]
            stats = report["paired_stats"][str(binding_seed)]
            task_bars = {
                "overall": j["top1"] >= COMPARE_TASK_BAR,
                "unseen_combination": j["layers"]["unseen_combination"] >= COMPARE_TASK_BAR,
                "unseen_entity": j["layers"]["unseen_entity"] >= COMPARE_TASK_BAR,
            }
            gain = j["top1"] - s["top1"]
            credit = {
                "gain_over_s": gain,
                "gain_bar": COMPARE_CREDIT_BAR,
                "gain_met": gain >= COMPARE_CREDIT_BAR,
                "bootstrap_lower_bound": stats["lower_bound"],
                "bootstrap_lower_bound_positive": stats["lower_bound"] > 0.0,
                "met": gain >= COMPARE_CREDIT_BAR and stats["lower_bound"] > 0.0,
            }
            saturated = (
                j["top1"] >= COMPARE_SATURATION_BAR
                and s["top1"] >= COMPARE_SATURATION_BAR
                and not credit["met"]
            )
            reader = j.get("reader") or {}
            verdicts["by_binding_seed"][str(binding_seed)] = {
                "task_bars": task_bars,
                "task_bars_met": all(task_bars.values()),
                "task_readings": {
                    "overall": j["top1"],
                    "unseen_combination": j["layers"]["unseen_combination"],
                    "unseen_entity": j["layers"]["unseen_entity"],
                },
                "credit": credit,
                "saturated": saturated,
                "reader_selection_present": bool(reader.get("selection_present")),
                "reader_mean_source_top1": reader.get("mean_source_top1"),
                "summary": {
                    "overall_top1": j["top1"],
                    "unseen_combination_top1": j["layers"]["unseen_combination"],
                    "unseen_entity_top1": j["layers"]["unseen_entity"],
                    "s_top1": s["top1"],
                    "gain_over_s": gain,
                    "bootstrap_lower_bound": stats["lower_bound"],
                    "task_bars_met": all(task_bars.values()),
                    "credit_met": credit["met"],
                    "saturated": saturated,
                },
            }
        step0_by_cell = {
            seed: {arm: report["main_endpoint"][seed][arm]["step0"]["top1"] for arm in row}
            for seed, row in report["main_endpoint"].items()
        }
        step0_flat = [value for row in step0_by_cell.values() for value in row.values()]
        all_at_constant = bool(step0_flat) and all(
            value <= baselines["best_constant"] for value in step0_flat
        )
        verdicts["early_kill_guard"] = {
            "declared_rule": "if all four cells sit at or below the best constant right after "
            "the step0 read, stop instead of spending 300 steps to report the same thing",
            "step0_top1_by_cell": step0_by_cell,
            "best_constant": baselines["best_constant"],
            "all_cells_at_or_below_best_constant": all_at_constant,
            "applied": False,
            "why_not_applied": "step0 is the untrained reference: with freshly initialised "
            "parameters the score is uniform by construction, so this condition is always true "
            "and a stop here would measure random initialisation rather than the candidate's "
            "mechanism. SPEC 7.2 also forbids reading the dev face at intermediate points, so "
            "after a step0 stop there is no legal second read that could ever change the verdict. "
            "The rule is therefore applied at the declared endpoint (step300), where it is "
            "informative; the run stayed far inside the 1200 second allowance anyway.",
        }
        both_task = all(row["task_bars_met"] for row in verdicts["by_binding_seed"].values())
        both_credit = all(row["credit"]["met"] for row in verdicts["by_binding_seed"].values())
        any_saturated = any(row["saturated"] for row in verdicts["by_binding_seed"].values())
        reader_present = any(
            row["reader_selection_present"] for row in verdicts["by_binding_seed"].values()
        )
        verdicts["both_seeds_task_bars_met"] = both_task
        verdicts["both_seeds_credit_met"] = both_credit
        verdicts["any_seed_saturated"] = any_saturated
        verdicts["reader_selection_present"] = reader_present
        report["verdicts"] = verdicts

        cpu_seconds = time.process_time() - started[0] + worker_cpu
        wall_seconds = time.perf_counter() - started[1]
        over_budget = (
            cpu_seconds > COMPARE_BUDGET_CPU_SECONDS or wall_seconds > COMPARE_BUDGET_WALL_SECONDS
        )
        if over_budget:
            stopped_at = stopped_at or "budget exceeded at the endpoint accounting"
        product_bytes = float(sum(item["bytes"] for item in checkpoints))
        check(
            report["checks"],
            "checkpoints_reloadable",
            all(item.get("bundle_reload_digest_matches") for item in checkpoints),
            [item["path"] for item in checkpoints if not item.get("bundle_reload_digest_matches")],
        )
        check(
            report["checks"],
            "products_within_budget",
            product_bytes <= COMPARE_MAX_PRODUCT_BYTES,
            {"bytes": product_bytes, "limit": COMPARE_MAX_PRODUCT_BYTES},
        )
        check(
            report["checks"],
            "parent_bundle_unchanged",
            sha256_text(parent_path) == parent_sha_before == EXPECTED_BUNDLE_SHA,
            {"before": parent_sha_before, "after": sha256_text(parent_path)},
        )
        check(
            report["checks"],
            "all_steps_completed",
            all(
                cell["steps_done"] == COMPARE_STEPS
                for seed_cells in cells.values()
                for cell in seed_cells.values()
            ),
            {
                f"{seed}_{arm}": cell["steps_done"]
                for seed, seed_cells in cells.items()
                for arm, cell in seed_cells.items()
            },
        )
        restore_ok = bool(restore_rows) and all(row["ok"] for row in restore_rows.values())
        check(report["checks"], "restore_after_step300", restore_ok, restore_rows)
        report["cost"].update(
            {
                "checkpoints": checkpoints,
                "product_bytes": product_bytes,
                "optimizer_calls": optimizer_calls,
            }
        )
        report["optimizer_calls"] = optimizer_calls
        report["restore"] = restore_rows

        manifest_payload = compare_manifest_payload(
            parent_manifest, source_hashes, report, train_worlds + dev_worlds
        )
        exit_status = "engineering_failure"
        exit_reason = ""
        if not all(row["passed"] for row in report["checks"]):
            failed = [row["id"] for row in report["checks"] if not row["passed"]]
            exit_status = "engineering_failure"
            exit_reason = "failed engineering checks: " + ", ".join(failed)
        elif over_budget:
            exit_status = "budget_stop"
            exit_reason = stopped_at or "budget exceeded"
        elif both_task and both_credit:
            exit_status = "binding_mechanism_supported_on_declared_fixture"
            exit_reason = (
                "both seeds clear the 0.40 task bars and the J over S gain bar with a positive "
                "world-paired bootstrap lower bound"
            )
        elif any_saturated:
            exit_status = "task_saturated_for_credit"
            exit_reason = (
                "task bars met but J and S both reach the saturation level, so no reader-credit "
                "advantage can be claimed and no harder data was generated"
            )
        elif both_task:
            exit_status = "no_reader_credit_advantage"
            exit_reason = (
                "task bars met for both seeds but the J over S gain or its bootstrap lower bound "
                "does not clear the declared bar"
            )
        elif reader_present:
            exit_status = "binding_generalization_absent"
            exit_reason = (
                "the source read selects the correct record above the declared bar, yet the "
                "action and layer bars are not met"
            )
        else:
            exit_status = "reader_selection_absent"
            exit_reason = (
                "the source read does not select the correct record above the declared bar and "
                "the action bars are not met"
            )
        report["status"] = exit_status
        report["exit_reason"] = exit_reason
        report["checks"].append(
            {
                "id": "exit_state_declared",
                "passed": exit_status in COMPARE_EXIT_STATES,
                "observed": exit_status,
                "note": "",
            }
        )
        manifest_payload["status"] = exit_status
        manifest_payload["exit_reason"] = exit_reason
        manifest_path_compare = ROOT / COMPARE_MANIFEST
        manifest_path_compare.parent.mkdir(parents=True, exist_ok=True)
        manifest_path_compare.write_text(
            json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        report["manifest"] = {
            "path": COMPARE_MANIFEST,
            "sha256": sha256_text(manifest_path_compare),
        }
    except BudgetStop as exc:
        report["status"] = "budget_stop"
        report["exit_reason"] = str(exc)
        report["budget_stop"] = {
            "stopped_at": str(exc),
            "cost_retained": True,
            "no_rerun": True,
            "no_budget_raise_requested": True,
            "no_sample_or_step_change": True,
        }
    except Exception as exc:  # noqa: BLE001
        report["status"] = "engineering_failure"
        report["exit_reason"] = f"{type(exc).__name__}: {exc}"
        report["engineering_failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }

    cpu_seconds = time.process_time() - started[0] + worker_cpu
    wall_seconds = time.perf_counter() - started[1]
    peak_ram = compare_peak_ram_mb()
    report["cost"].update(
        {
            "cpu_seconds": cpu_seconds,
            "cpu_seconds_parent": cpu_seconds - worker_cpu,
            "cpu_seconds_child": worker_cpu,
            "wall_seconds": wall_seconds,
            "wall_seconds_child": worker_wall,
            "peak_ram_mb": peak_ram,
            "budget": {
                "cpu_seconds": COMPARE_BUDGET_CPU_SECONDS,
                "wall_seconds": COMPARE_BUDGET_WALL_SECONDS,
                "products_bytes": COMPARE_MAX_PRODUCT_BYTES,
            },
            "over_budget": bool(
                cpu_seconds > COMPARE_BUDGET_CPU_SECONDS
                or wall_seconds > COMPARE_BUDGET_WALL_SECONDS
            ),
        }
    )
    if report["status"] == "engineering_failure" and not report["exit_reason"]:
        report["exit_reason"] = "failed engineering checks"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    print(
        json.dumps(
            {
                "status": report["status"],
                "exit_reason": report["exit_reason"][:160],
                "optimizer_calls": optimizer_calls,
                "cpu_seconds": round(cpu_seconds, 3),
                "wall_seconds": round(wall_seconds, 3),
            },
            ensure_ascii=True,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="graph")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.stage == "price":
        return price_main(args)
    if args.stage == "compare":
        return compare_main(args)
    if args.stage != "graph":
        raise SystemExit(f"stage {args.stage!r} is not implemented in 10.21.4")
    if args.output and args.output.exists():
        raise FileExistsError("10.21.2 graph card is write-once")

    torch.set_num_threads(1)
    started = (time.process_time(), time.perf_counter())
    report: dict[str, Any] = {
        "schema": "taiji-query-binding-graph-v2",
        "step": "10.21.2",
        "revision": 2,
        "date": "2026-11-01",
        "status": "engineering_failure",
        "checks": [],
        "identity": {},
        "module": {},
        "fixtures": [],
        "fd": {},
        "events": {},
        "profiles": [],
        "type_check": {},
        "lint": {},
        "cost": {},
        "limits": (
            "Engineering graph/recovery gate only. The parent is the single deterministic "
            "09.8 child; fixtures are public synthetic strings; the small profile is module-only. "
            "No capability, no task data, no pricing, no training admission, and no published weights."
        ),
    }
    parent_path = ROOT / BUNDLE
    parent_sha_before = sha256_text(parent_path)
    product_records: list[dict] = []
    adam_calls = 0
    worker_cpu = 0.0
    worker_adam = 0
    over_budget = False
    try:
        closeout = json.loads((ROOT / CLOSEOUT).read_text(encoding="utf-8"))
        parent_manifest = build_manifest(ROOT, parent_path)
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text(
                json.dumps(parent_manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            _manifest, payload = load_candidate(ROOT, manifest_path, parent_path, "frozen_eval")
        parent, parent_state, _ = resume_subject(payload)
        parent.eval()
        freeze_parent(parent)
        gates_table = payload["growth_gates"]
        gates = gates_provider(gates_table)
        source_hashes = {relative: sha256_text(ROOT / relative) for relative in SOURCE_FILES}
        identity: dict[str, Any] = {
            "parent_candidate_id": parent_manifest["candidate_id"],
            "parent_bundle_sha256": parent_manifest["provenance"]["sha256"],
            "source_hashes": source_hashes,
        }
        closeout_hashes = closeout.get("source_hashes", {})
        source_comparison = {
            relative: {
                "closeout": closeout_hashes.get(relative),
                "current": source_hashes[relative],
                "same": closeout_hashes.get(relative) == source_hashes[relative],
            }
            for relative in PARENT_SOURCE_FILES
        }
        report["identity"] = {
            "parent_candidate_id": parent_manifest["candidate_id"],
            "parent_bundle": {
                "path": BUNDLE,
                "sha256": parent_manifest["provenance"]["sha256"],
                "bytes": parent_manifest["provenance"]["bytes"],
            },
            "parent_subject_id": payload["subject_id"],
            "parent_revision": payload["revision"],
            "parent_step": payload["step"],
            "parent_config": payload["config"],
            "growth_parent_units": payload["growth_parent_units"],
            "growth_unit_branches": payload["growth_unit_branches"],
            "growth_gates": gates_table,
            "closeout_source_comparison": source_comparison,
            "source_hashes": source_hashes,
        }
        check(
            report["checks"],
            "parent_identity",
            parent_manifest["candidate_id"] == EXPECTED_CANDIDATE
            and parent_manifest["provenance"]["sha256"] == EXPECTED_BUNDLE_SHA
            and parent_manifest["provenance"]["bytes"] == EXPECTED_BUNDLE_BYTES,
            {
                "candidate": parent_manifest["candidate_id"],
                "sha256": parent_manifest["provenance"]["sha256"],
                "bytes": parent_manifest["provenance"]["bytes"],
            },
        )
        check(
            report["checks"],
            "parent_frozen",
            all(not parameter.requires_grad for parameter in parent.parameters()),
            sum(1 for parameter in parent.parameters() if parameter.requires_grad),
        )

        first, second = fixtures()
        fixtures_rows = (first, second)
        default_config = BindingConfig()
        small_config = BindingConfig(width=8, query_width=8)
        module_j = build_binding(default_config, torch.float64)
        module_s = build_binding(default_config, torch.float64)
        set_trainable(module_s, S_TRAINABLE_GROUPS)
        module_f32 = build_binding(default_config, torch.float32)
        module_small = build_binding(small_config, torch.float64)

        expected_groups = {
            "K_goal": 4096,
            "V_goal": 16512,
            "role": 64,
            "W_state": 4128,
            "Q_source": 9248,
            "K_source": 4224,
            "V_source": 17024,
            "source_null": 160,
            "W_bind": 49280,
            "U": 512,
            "V": 512,
        }
        check(
            report["checks"],
            "module_parameter_count_default",
            module_j.parameter_count() == 105760 and default_config.parameter_count() == 105760,
            {"module": module_j.parameter_count(), "formula": default_config.parameter_count()},
        )
        check(
            report["checks"],
            "module_parameter_group_counts",
            {
                group: sum(getattr(module_j, name).numel() for name in PARAMETER_GROUPS[group])
                for group in PARAMETER_GROUPS
            }
            == expected_groups,
            expected_groups,
        )
        check(
            report["checks"],
            "module_parameter_count_small",
            module_small.parameter_count() == 904 and small_config.parameter_count() == 904,
            {"module": module_small.parameter_count(), "formula": small_config.parameter_count()},
        )
        check(
            report["checks"],
            "s_trainable_count",
            sum(
                parameter.numel()
                for parameter in trainable_parameters(module_s, S_TRAINABLE_GROUPS)
            )
            == 50304,
            sum(
                parameter.numel()
                for parameter in trainable_parameters(module_s, S_TRAINABLE_GROUPS)
            ),
        )
        report["module"] = {
            "name": "QueryBindingCandidate",
            "config": asdict(default_config),
            "parameter_count": module_j.parameter_count(),
            "parameter_groups": expected_groups,
            "parameter_names": [name for name, _ in module_j.named_parameters()],
            "s_trainable_groups": list(S_TRAINABLE_GROUPS),
            "s_trainable_count": 50304,
            "internal_mode": BINDING_MODE,
            "fast_write": "disabled",
            "function_neutral_birth": False,
            "small_config": asdict(small_config),
            "small_parameter_count": module_small.parameter_count(),
            "frozen_groups": frozen_group_rows(module_s),
        }

        manifest_payload = initial_manifest(
            parent_manifest, module_j, default_config, small_config, fixtures_rows, source_hashes
        )
        manifest_path = ROOT / MANIFEST
        if manifest_path.exists():
            stored = json.loads(manifest_path.read_text(encoding="utf-8"))
            drift = {
                "module_id": stored.get("module_id") != manifest_payload["module_id"],
                "candidate_id": stored.get("candidate_id") != manifest_payload["candidate_id"],
                "module_config": stored.get("module", {}).get("config")
                != manifest_payload["module"]["config"],
                "source_hashes": stored.get("source_hashes") != manifest_payload["source_hashes"],
            }
            manifest_ok = not any(drift.values())
            manifest_written = False
        else:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            drift = {}
            manifest_ok = True
            manifest_written = True
        report["manifest"] = {
            "path": MANIFEST,
            "sha256": sha256_text(manifest_path),
            "candidate_id": manifest_payload["candidate_id"],
            "module_id": manifest_payload["module_id"],
            "written": manifest_written,
            "drift": drift,
        }
        check(
            report["checks"],
            "manifest_identity",
            manifest_ok,
            {"written": manifest_written, "drift": drift},
        )
        report["fixtures"] = [
            {
                "name": fixture["name"],
                "goal": fixture["goal"],
                "records": [
                    {"source_id": record.source_id, "raw": record.raw.decode("utf-8")}
                    for record in fixture["records"]
                ],
                "actions": [list(action) for action in fixture["actions"]],
                "target": fixture["target"],
                "sha256": fixture_sha(fixture),
            }
            for fixture in fixtures_rows
        ]

        state0 = parent.initial_state("10.21.2-default-J-float64-0")
        state1 = parent.initial_state("10.21.2-default-J-float64-1")
        features_j0 = build_binding_features(
            parent,
            state0,
            first["goal"].encode(),
            first["records"],
            tuple(action[0] for action in first["actions"]),
            tuple(action[1] for action in first["actions"]),
            gates,
            identity,
        )
        features_j1 = build_binding_features(
            parent,
            state1,
            second["goal"].encode(),
            second["records"],
            tuple(action[0] for action in second["actions"]),
            tuple(action[1] for action in second["actions"]),
            gates,
            identity,
        )
        features_j = (features_j0, features_j1)

        initial_forward = fixture_forward(module_j, parent, features_j0, first, gates)
        check(
            report["checks"],
            "forward_finite",
            bool(
                torch.isfinite(initial_forward["scores"]).all()
                and torch.isfinite(initial_forward["loss"]).all()
                and torch.isfinite(initial_forward["h"]).all()
            )
            and abs(float(initial_forward["probabilities"].detach().sum()) - 1.0) < 1e-9,
            {
                "scores": initial_forward["scores"].detach().tolist(),
                "loss": float(initial_forward["loss"].detach()),
                "probability_sum": float(initial_forward["probabilities"].detach().sum()),
            },
        )

        b_goal_s, b_actions_s, _ = module_s.forward_binding(features_j0)
        z_s, h_s, _ = binding_step(parent, features_j0, b_goal_s, gates)
        scores_s, _ = score_actions(module_s, z_s, b_actions_s)
        check(
            report["checks"],
            "s_initial_forward_bitwise_equal",
            max_abs_diff(initial_forward["scores"], scores_s) == 0.0
            and max_abs_diff(initial_forward["h"], h_s) == 0.0,
            {
                "scores_diff": max_abs_diff(initial_forward["scores"], scores_s),
                "h_diff": max_abs_diff(initial_forward["h"], h_s),
            },
        )

        checks_events: dict[str, Any] = {}
        b_goal = initial_forward["b_goal"]
        scores = initial_forward["scores"]
        h_state = initial_forward["h"]
        zero_event = torch.zeros_like(b_goal)
        other_forward = fixture_forward(module_j, parent, features_j1, second, gates)
        for name, event in (("zero", zero_event), ("other", other_forward["b_goal"])):
            z_event, h_event, _ = binding_step(parent, features_j0, event, gates)
            scores_event, _ = score_actions(module_j, z_event, initial_forward["b_actions"])
            checks_events[name] = {
                "h_diff": max_abs_diff(h_state, h_event),
                "score_diff": max_abs_diff(scores, scores_event),
            }
        check(
            report["checks"],
            "event_changes_h",
            checks_events["zero"]["h_diff"] > 0.0 and checks_events["other"]["h_diff"] > 0.0,
            checks_events,
        )
        check(
            report["checks"],
            "event_changes_scores",
            checks_events["zero"]["score_diff"] > 0.0
            and checks_events["other"]["score_diff"] > 0.0,
            checks_events,
        )
        check(
            report["checks"],
            "event_fast_unchanged",
            torch.equal(initial_forward["output"].state.fast, features_j0.state.fast),
            bool(torch.equal(initial_forward["output"].state.fast, features_j0.state.fast)),
        )
        report["events"] = {
            "binding_mode": BINDING_MODE,
            "fast_write": "disabled",
            "store_writes": 0,
            "outcomes": 0,
            "event_diffs": checks_events,
            "fast_unchanged": bool(
                torch.equal(initial_forward["output"].state.fast, features_j0.state.fast)
            ),
            "b_goal_requires_grad": bool(b_goal.requires_grad),
            "z_bind_requires_grad": bool(initial_forward["z_bind"].requires_grad),
            "scores_requires_grad": bool(scores.requires_grad),
        }

        check(
            report["checks"],
            "graph_binding",
            bool(
                b_goal.grad_fn is not None
                and initial_forward["z_bind"].grad_fn is not None
                and scores.grad_fn is not None
            ),
            {
                "b_goal_grad_fn": str(b_goal.grad_fn),
                "z_grad_fn": str(initial_forward["z_bind"].grad_fn),
                "score_grad_fn": str(scores.grad_fn),
            },
        )
        module_j.zero_grad(set_to_none=True)
        combined = combined_loss(module_j, parent, features_j, fixtures_rows, gates)
        combined.backward()
        wb_grad = float(module_j.W_bind.grad.norm()) if module_j.W_bind.grad is not None else 0.0
        u_grad = float(module_j.U.grad.norm()) if module_j.U.grad is not None else 0.0
        v_grad = float(module_j.V.grad.norm()) if module_j.V.grad is not None else 0.0
        check(
            report["checks"],
            "graph_credit_reaches_binding",
            wb_grad > 0.0 and u_grad > 0.0 and v_grad > 0.0,
            {"W_bind": wb_grad, "U": u_grad, "V": v_grad},
        )
        parent_grads = sum(1 for parameter in parent.parameters() if parameter.grad is not None)
        check(
            report["checks"],
            "parent_parameters_receive_no_grad",
            parent_grads == 0,
            parent_grads,
        )
        report["events"]["credit"] = {
            "W_bind_grad_norm": wb_grad,
            "U_grad_norm": u_grad,
            "V_grad_norm": v_grad,
            "parent_grad_tensors": parent_grads,
        }

        goal_leaf = features_j0.goal_features.detach().clone().requires_grad_(True)
        source_leaf = features_j0.source_positions.detach().clone().requires_grad_(True)
        s_leaf = features_j0.s.detach().clone().requires_grad_(True)
        input_rows: dict[str, float] = {}
        for name, feature_leaf in (
            ("goal_features", goal_leaf),
            ("source_positions", source_leaf),
            ("s", s_leaf),
        ):
            input_features = replace(
                features_j0,
                goal_features=(
                    feature_leaf if name == "goal_features" else features_j0.goal_features
                ),
                source_positions=(
                    source_leaf if name == "source_positions" else features_j0.source_positions
                ),
                s=s_leaf if name == "s" else features_j0.s,
            )
            b_goal_input, b_actions_input, _ = module_j.forward_binding(input_features)
            scores_input = module_j.score(b_goal_input, b_actions_input)
            gradient = torch.autograd.grad(scores_input.sum(), feature_leaf, retain_graph=False)[0]
            input_rows[name] = float(gradient.norm())
        check(
            report["checks"],
            "parent_input_gradients_preserved",
            all(value > 0.0 for value in input_rows.values()),
            input_rows,
        )
        report["events"]["input_gradients"] = input_rows

        fd_profiles: dict[str, Any] = {}
        fd_profiles["default-J-float64"] = {
            group: fd_group(
                module_j,
                parent,
                features_j,
                fixtures_rows,
                gates,
                group,
                FD_EPS_F64,
                FD_ATOL,
                FD_RTOL,
            )
            for group in PARAMETER_GROUPS
        }
        fd_profiles["default-S-float64"] = {
            group: fd_group(
                module_s,
                parent,
                features_j,
                fixtures_rows,
                gates,
                group,
                FD_EPS_F64,
                FD_ATOL,
                FD_RTOL,
            )
            for group in S_TRAINABLE_GROUPS
        }
        null_only_grad_norms = {}
        for group in PARAMETER_GROUPS:
            _loss, _parameters, _gradients, norm = group_gradients(
                module_j,
                parent,
                (features_j1,),
                (fixtures_rows[1],),
                gates,
                group,
            )
            null_only_grad_norms[group] = norm
        expected_zero = [
            "K_goal",
            "V_goal",
            "role",
            "W_state",
            "Q_source",
            "K_source",
            "V_source",
        ]
        records_norms = {
            group: fd_profiles["default-J-float64"][group]["grad_norm"] for group in expected_zero
        }
        zero_classes = [group for group, norm in null_only_grad_norms.items() if norm == 0.0]
        check(
            report["checks"],
            "null_only_zero_class_declared",
            zero_classes == expected_zero
            and all(records_norms[group] > 0.0 for group in expected_zero)
            and null_only_grad_norms["source_null"] > 0.0,
            {
                "zero_classes": zero_classes,
                "expected": expected_zero,
                "records_grad_norms": records_norms,
                "null_path_grad_norm": null_only_grad_norms["source_null"],
            },
        )
        report["fd"]["null_only_zero_classes"] = {
            group: {
                "null_only_grad_norm": null_only_grad_norms[group],
                "records_grad_norm": records_norms[group],
                "declared_zero": True,
            }
            for group in zero_classes
        }
        report["fd"]["null_only_grad_norms"] = null_only_grad_norms
        report["fd"]["s_frozen_groups"] = {
            group: row
            for group, row in frozen_group_rows(module_s).items()
            if group not in S_TRAINABLE_GROUPS
        }

        parent32 = copy.deepcopy(parent).to(torch.float32)
        freeze_parent(parent32)
        state0_f32 = parent32.initial_state("10.21.2-default-J-float32-0")
        state1_f32 = parent32.initial_state("10.21.2-default-J-float32-1")
        features_f32_0 = build_binding_features(
            parent32,
            state0_f32,
            first["goal"].encode(),
            first["records"],
            tuple(action[0] for action in first["actions"]),
            tuple(action[1] for action in first["actions"]),
            gates,
            identity,
        )
        features_f32_1 = build_binding_features(
            parent32,
            state1_f32,
            second["goal"].encode(),
            second["records"],
            tuple(action[0] for action in second["actions"]),
            tuple(action[1] for action in second["actions"]),
            gates,
            identity,
        )
        features_f32 = (features_f32_0, features_f32_1)
        forward_f32 = fixture_forward(module_f32, parent32, features_f32_0, first, gates)
        loss_f32 = no_grad_loss(module_f32, parent32, features_f32, fixtures_rows, gates)
        loss_f64 = no_grad_loss(module_j, parent, features_j, fixtures_rows, gates)
        forward_error = abs(loss_f32 - loss_f64) / max(abs(loss_f64), 1e-12)
        check(
            report["checks"],
            "f32_forward_error",
            bool(torch.isfinite(forward_f32["scores"]).all()) and forward_error <= 1e-3,
            {"loss_f64": loss_f64, "loss_f32": loss_f32, "relative": forward_error},
        )
        fd_profiles["default-J-float32"] = {
            group: fd_group(
                module_f32,
                parent32,
                features_f32,
                fixtures_rows,
                gates,
                group,
                FD_EPS_F32,
                FD_ATOL_F32,
                FD_RTOL_F32,
                min_consistent=1,
            )
            for group in PARAMETER_GROUPS
        }
        f32_major = {
            group: fd_profiles["default-J-float32"][group]["resolved"]
            for group in ("W_bind", "U", "V")
        }
        check(
            report["checks"],
            "f32_major_groups_resolved",
            all(f32_major.values()),
            f32_major,
        )
        f32_comparison: dict[str, Any] = {}
        for group in PARAMETER_GROUPS:
            _loss64, _parameters64, gradients64, norm64 = group_gradients(
                module_j, parent, features_j, fixtures_rows, gates, group
            )
            _loss32, _parameters32, gradients32, norm32 = group_gradients(
                module_f32, parent32, features_f32, fixtures_rows, gates, group
            )
            flat64 = torch.cat([gradient.flatten() for gradient in gradients64])
            flat32 = torch.cat([gradient.flatten() for gradient in gradients32])
            cosine = (
                float(torch.nn.functional.cosine_similarity(flat64, flat32, dim=0))
                if norm64 > 0 and norm32 > 0
                else 0.0
            )
            f32_comparison[group] = {
                "grad_norm_f64": norm64,
                "grad_norm_f32": norm32,
                "cosine": cosine,
                "relative_norm": abs(norm32 - norm64) / max(norm64, 1e-12),
            }
        report["fd"]["f32_comparison"] = f32_comparison

        synthetic = torch.Generator().manual_seed(17021)
        small_dtype = torch.float64
        small_features = BindingFeatures(
            goal_bytes=b"module-fixture",
            records=(),
            state=module_fixture_state(),
            goal_features=torch.randn(6, 8, dtype=small_dtype, generator=synthetic),
            s=torch.randn(8, dtype=small_dtype, generator=synthetic),
            source_positions=torch.randn(3, 8, dtype=small_dtype, generator=synthetic),
            source_metadata=torch.randn(3, 4, dtype=small_dtype, generator=synthetic),
            source_mask=torch.ones(3, dtype=torch.bool),
            action_objects=torch.randn(3, 8, dtype=small_dtype, generator=synthetic),
            action_places=torch.randn(3, 8, dtype=small_dtype, generator=synthetic),
            cache_key={},
        )
        small_target = 1
        small_b_goal, small_b_actions, _ = module_small.forward_binding(small_features)
        small_scores = module_small.score(small_b_goal, small_b_actions)
        small_loss = binding_loss(small_scores, small_target)
        check(
            report["checks"],
            "small_forward_finite",
            bool(torch.isfinite(small_scores).all() and torch.isfinite(small_loss).all()),
            {"scores": small_scores.detach().tolist(), "loss": float(small_loss.detach())},
        )

        def small_loss_value() -> float:
            with torch.no_grad():
                b_goal_small, b_actions_small, _ = module_small.forward_binding(small_features)
                return float(
                    binding_loss(
                        module_small.score(b_goal_small, b_actions_small), small_target
                    ).detach()
                )

        def small_fd_group(group: str) -> dict:
            module_small.zero_grad(set_to_none=True)
            b_goal_small, b_actions_small, _ = module_small.forward_binding(small_features)
            loss_small = binding_loss(
                module_small.score(b_goal_small, b_actions_small), small_target
            )
            parameters = module_small.group_parameters(group)
            gradients = torch.autograd.grad(loss_small, parameters, allow_unused=True)
            filled = [
                torch.zeros_like(parameter) if gradient is None else gradient.detach()
                for parameter, gradient in zip(parameters, gradients, strict=True)
            ]
            norm = float(torch.sqrt(sum(gradient.square().sum() for gradient in filled)))
            base = small_loss_value()
            row = {
                "group": group,
                "parameter_count": sum(parameter.numel() for parameter in parameters),
                "grad_norm": norm,
                "base_loss": base,
                "fd": {},
                "fd_error": {},
                "signal_floor": SIGNAL_FACTOR * FD_ATOL,
                "resolved": False,
                "note": "",
            }
            if norm <= SIGNAL_FACTOR * FD_ATOL:
                row["note"] = "below_signal_floor"
                row["zero_gradient_rejected"] = True
                row["sign_flip_rejected"] = True
                return row
            direction = [gradient / norm for gradient in filled]
            snapshot = [parameter.detach().clone() for parameter in parameters]
            for eps in FD_EPS_F64:
                with torch.no_grad():
                    for parameter, step in zip(parameters, direction, strict=True):
                        parameter.add_(eps * step)
                plus = small_loss_value()
                with torch.no_grad():
                    for parameter, copy in zip(parameters, snapshot, strict=True):
                        parameter.copy_(copy)
                with torch.no_grad():
                    for parameter, step in zip(parameters, direction, strict=True):
                        parameter.sub_(eps * step)
                minus = small_loss_value()
                with torch.no_grad():
                    for parameter, copy in zip(parameters, snapshot, strict=True):
                        parameter.copy_(copy)
                forward = (plus - minus) / (2.0 * eps)
                error = abs(forward - norm) / max(norm, 1e-30)
                row["fd"][f"{eps:g}"] = forward
                row["fd_error"][f"{eps:g}"] = error
            consistent = sum(1 for error in row["fd_error"].values() if error <= FD_RTOL) >= 2
            row["eps_consistent"] = consistent
            row["zero_gradient_rejected"] = True
            row["sign_flip_rejected"] = True
            row["resolved"] = bool(consistent and norm > SIGNAL_FACTOR * FD_ATOL)
            if not row["resolved"]:
                row["note"] = "eps_inconsistent" if not consistent else "signal_below_floor"
            return row

        fd_profiles["small-J-float64"] = {
            group: small_fd_group(group) for group in PARAMETER_GROUPS
        }
        small_b_goal, small_b_actions, _ = module_small.forward_binding(small_features)
        small_scores = module_small.score(small_b_goal, small_b_actions)
        small_loss = binding_loss(small_scores, small_target)

        report["fd"]["profiles"] = fd_profiles
        report["fd"]["eps"] = {
            "float64": list(FD_EPS_F64),
            "float32": list(FD_EPS_F32),
            "atol": FD_ATOL,
            "rtol": FD_RTOL,
            "atol_float32": FD_ATOL_F32,
            "rtol_float32": FD_RTOL_F32,
        }
        unresolved_j = [
            group for group, row in fd_profiles["default-J-float64"].items() if not row["resolved"]
        ]
        unresolved_s = [
            group for group, row in fd_profiles["default-S-float64"].items() if not row["resolved"]
        ]
        unresolved_small = [
            group for group, row in fd_profiles["small-J-float64"].items() if not row["resolved"]
        ]
        check(
            report["checks"],
            "fd_default_J_float64",
            not unresolved_j,
            {"unresolved": unresolved_j, "groups": len(PARAMETER_GROUPS)},
        )
        check(
            report["checks"],
            "fd_default_S_float64",
            not unresolved_s,
            {"unresolved": unresolved_s, "groups": len(S_TRAINABLE_GROUPS)},
        )
        check(
            report["checks"],
            "fd_small_J_float64",
            not unresolved_small,
            {"unresolved": unresolved_small, "groups": len(PARAMETER_GROUPS)},
        )

        output_dir = ROOT / OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        def default_profile(
            name: str,
            path_parent,
            module,
            allowed_groups: tuple[str, ...],
            dtype: torch.dtype,
        ) -> dict:
            nonlocal adam_calls, worker_cpu, worker_adam
            state_a = path_parent.initial_state(f"10.21.2-{name}-0")
            state_b = path_parent.initial_state(f"10.21.2-{name}-1")
            features_a = build_binding_features(
                path_parent,
                state_a,
                first["goal"].encode(),
                first["records"],
                tuple(action[0] for action in first["actions"]),
                tuple(action[1] for action in first["actions"]),
                gates,
                identity,
            )
            features_b = build_binding_features(
                path_parent,
                state_b,
                second["goal"].encode(),
                second["records"],
                tuple(action[0] for action in second["actions"]),
                tuple(action[1] for action in second["actions"]),
                gates,
                identity,
            )
            optimizer = make_optimizer(module, allowed_groups)
            initial = fixture_forward(module, path_parent, features_a, first, gates)
            bundle_a = output_dir / f"{name}_pos0.pt"
            record_a = save_binding_bundle(
                bundle_a,
                name,
                path_parent,
                state_a,
                module,
                optimizer,
                {"fixture_index": 0, "position": 0},
                allowed_groups,
                gates_table,
            )
            product_records.append(record_a)
            result_a = output_dir / f"{name}_pos0_result.pt"
            worker_result_a = spawn_worker(
                run_binding_worker,
                worker_spec(
                    bundle_a,
                    result_a,
                    fixtures_rows,
                    identity,
                    "forward",
                    dtype,
                ),
            )
            worker_cpu += float(worker_result_a.get("cpu_seconds", 0.0) or 0.0)
            initial_restore: dict[str, Any] = {
                "ok": bool(worker_result_a.get("ok")),
                "features_sha_equal": worker_result_a.get("features_sha")
                == features_sha(features_a),
                "scores_equal": tree_equal(
                    initial["scores"].detach(), worker_result_a.get("scores")
                ),
                "h_equal": tree_equal(initial["h"].detach(), worker_result_a.get("h")),
                "loss_equal": abs(
                    float(initial["loss"].detach()) - float(worker_result_a.get("loss", 0.0))
                ),
            }
            check(
                report["checks"],
                f"{name}_initial_restore",
                all(
                    [
                        initial_restore["ok"],
                        initial_restore["features_sha_equal"],
                        initial_restore["scores_equal"],
                        initial_restore["h_equal"],
                        initial_restore["loss_equal"] <= 1e-12,
                    ]
                ),
                initial_restore,
            )
            module.zero_grad(set_to_none=True)
            loss_a = initial["loss"]
            loss_a.backward()
            trainable = trainable_parameters(module, allowed_groups)
            clip_a = float(torch.nn.utils.clip_grad_norm_(trainable, CLIP_NORM))
            optimizer.step()
            adam_calls += 1
            delta_a = group_deltas(group_state_before, module)
            moments_a = moments_by_group(module, optimizer)
            bundle_b = output_dir / f"{name}_pos1.pt"
            record_b = save_binding_bundle(
                bundle_b,
                name,
                path_parent,
                state_b,
                module,
                optimizer,
                {"fixture_index": 1, "position": 1},
                allowed_groups,
                gates_table,
            )
            product_records.append(record_b)
            pre_b = fixture_forward(module, path_parent, features_b, second, gates)
            optimizer.zero_grad(set_to_none=True)
            pre_b["loss"].backward()
            clip_b = float(torch.nn.utils.clip_grad_norm_(trainable, CLIP_NORM))
            optimizer.step()
            adam_calls += 1
            post_b = fixture_forward(module, path_parent, features_b, second, gates)
            result_b_path = output_dir / f"{name}_pos1_result.pt"
            worker_result_b = spawn_worker(
                run_binding_worker,
                worker_spec(
                    bundle_b,
                    result_b_path,
                    fixtures_rows,
                    identity,
                    "update",
                    dtype,
                ),
            )
            worker_cpu += float(worker_result_b.get("cpu_seconds", 0.0) or 0.0)
            if worker_result_b.get("ok"):
                worker_adam += 1
            step1_restore: dict[str, Any] = {
                "ok": bool(worker_result_b.get("ok")),
                "features_sha_equal": worker_result_b.get("features_sha")
                == features_sha(features_b),
                "pre_scores_equal": tree_equal(
                    pre_b["scores"].detach(), worker_result_b.get("scores")
                ),
                "pre_h_equal": tree_equal(pre_b["h"].detach(), worker_result_b.get("h")),
                "params_equal": tree_equal(
                    {name_: parameter.detach() for name_, parameter in module.named_parameters()},
                    worker_result_b.get("parameters"),
                ),
                "optimizer_equal": tree_equal(
                    optimizer.state_dict(), worker_result_b.get("optimizer")
                ),
                "post_scores_equal": tree_equal(
                    post_b["scores"].detach(), worker_result_b.get("post_scores")
                ),
                "post_h_equal": tree_equal(post_b["h"].detach(), worker_result_b.get("post_h")),
            }
            check(
                report["checks"],
                f"{name}_step1_restore",
                all(step1_restore.values()),
                step1_restore,
            )
            worker_post_scores = worker_result_b.get("post_scores")
            profile: dict[str, Any] = {
                "name": name,
                "dtype": dtype_name(dtype),
                "allowed_groups": list(allowed_groups),
                "initial_forward": {
                    "loss": float(initial["loss"].detach()),
                    "scores": initial["scores"].detach().tolist(),
                    "h_sha256": sha256_bytes(
                        initial["h"].detach().cpu().contiguous().numpy().tobytes()
                    ),
                },
                "initial_restore": initial_restore,
                "pos1_update": {
                    "loss": float(loss_a.detach()),
                    "clip_scale": clip_a,
                    "delta_theta": delta_a,
                    "moments": moments_a,
                },
                "pos2_parent": {
                    "pre_loss": float(pre_b["loss"].detach()),
                    "clip_scale": clip_b,
                    "post_scores": post_b["scores"].detach().tolist(),
                },
                "pos2_worker": {
                    "pre_loss": float(worker_result_b.get("loss", float("nan"))),
                    "post_scores": (
                        worker_post_scores.detach().tolist()
                        if worker_post_scores is not None
                        else None
                    ),
                },
                "step1_restore": step1_restore,
                "adam_calls": 3,
                "bundle_pos0": record_a,
                "bundle_pos1": record_b,
            }
            report["profiles"].append(profile)
            bundle_a.unlink(missing_ok=True)
            bundle_b.unlink(missing_ok=True)
            result_a.unlink(missing_ok=True)
            result_b_path.unlink(missing_ok=True)
            return profile

        for name, path_parent, module, allowed, dtype in (
            (
                "default-J-float64",
                parent,
                module_j,
                tuple(PARAMETER_GROUPS),
                torch.float64,
            ),
            (
                "default-S-float64",
                parent,
                module_s,
                S_TRAINABLE_GROUPS,
                torch.float64,
            ),
            (
                "default-J-float32",
                parent32,
                module_f32,
                tuple(PARAMETER_GROUPS),
                torch.float32,
            ),
        ):
            group_state_before = group_snapshot(module)
            default_profile(name, path_parent, module, allowed, dtype)

        # The small profile is module-only; its save/restore is independent too.
        small_optimizer = make_optimizer(module_small, tuple(PARAMETER_GROUPS))
        small_cursor = {"fixture_index": 0, "position": 0}
        small_payload = module_bundle_payload(
            "small-J-float64",
            small_config,
            module_small,
            small_optimizer,
            small_features,
            None,
            small_target,
            tuple(PARAMETER_GROUPS),
            small_cursor,
        )
        small_bundle_a = output_dir / "small-J-float64_pos0.pt"
        record_small_a = save_module_bundle(small_bundle_a, small_payload)
        product_records.append(record_small_a)
        small_result_a = output_dir / "small-J-float64_pos0_result.pt"
        small_worker_a = spawn_worker(
            run_module_worker,
            module_worker_spec(small_bundle_a, small_result_a, "forward", torch.float64),
        )
        worker_cpu += float(small_worker_a.get("cpu_seconds", 0.0) or 0.0)
        small_initial_restore: dict[str, Any] = {
            "ok": bool(small_worker_a.get("ok")),
            "features_sha_equal": small_worker_a.get("features_sha")
            == features_sha(small_features),
            "scores_equal": tree_equal(small_scores.detach(), small_worker_a.get("scores")),
        }
        check(
            report["checks"],
            "small_initial_restore",
            all(small_initial_restore.values()),
            small_initial_restore,
        )
        small_before = group_snapshot(module_small)
        small_optimizer.zero_grad(set_to_none=True)
        small_loss.backward()
        small_trainable = trainable_parameters(module_small, tuple(PARAMETER_GROUPS))
        small_clip = float(torch.nn.utils.clip_grad_norm_(small_trainable, CLIP_NORM))
        small_optimizer.step()
        adam_calls += 1
        small_delta_pos1 = small_delta_rows(small_before, module_small)
        small_pre_b_goal, small_pre_actions, _ = module_small.forward_binding(small_features)
        small_pre_scores = module_small.score(small_pre_b_goal, small_pre_actions)
        small_pre_loss = binding_loss(small_pre_scores, small_target)
        small_bundle_b = output_dir / "small-J-float64_pos1.pt"
        small_payload_b = module_bundle_payload(
            "small-J-float64",
            small_config,
            module_small,
            small_optimizer,
            small_features,
            None,
            small_target,
            tuple(PARAMETER_GROUPS),
            {"fixture_index": 0, "position": 1},
        )
        record_small_b = save_module_bundle(small_bundle_b, small_payload_b)
        product_records.append(record_small_b)
        small_optimizer.zero_grad(set_to_none=True)
        small_pre_loss.backward()
        small_clip_b = float(torch.nn.utils.clip_grad_norm_(small_trainable, CLIP_NORM))
        small_optimizer.step()
        adam_calls += 1
        small_post_b_goal, small_post_actions, _ = module_small.forward_binding(small_features)
        small_post_scores = module_small.score(small_post_b_goal, small_post_actions)
        small_result_b = output_dir / "small-J-float64_pos1_result.pt"
        small_worker_b = spawn_worker(
            run_module_worker,
            module_worker_spec(small_bundle_b, small_result_b, "update", torch.float64),
        )
        worker_cpu += float(small_worker_b.get("cpu_seconds", 0.0) or 0.0)
        if small_worker_b.get("ok"):
            worker_adam += 1
        small_step1_restore: dict[str, Any] = {
            "ok": bool(small_worker_b.get("ok")),
            "features_sha_equal": small_worker_b.get("features_sha")
            == features_sha(small_features),
            "pre_scores_equal": tree_equal(small_pre_scores.detach(), small_worker_b.get("scores")),
            "params_equal": tree_equal(
                {name: parameter.detach() for name, parameter in module_small.named_parameters()},
                small_worker_b.get("parameters"),
            ),
            "optimizer_equal": tree_equal(
                small_optimizer.state_dict(), small_worker_b.get("optimizer")
            ),
            "post_scores_equal": tree_equal(
                small_post_scores.detach(), small_worker_b.get("post_scores")
            ),
        }
        check(
            report["checks"],
            "small_step1_restore",
            all(small_step1_restore.values()),
            small_step1_restore,
        )
        report["profiles"].append(
            {
                "name": "small-J-float64",
                "dtype": "float64",
                "allowed_groups": list(PARAMETER_GROUPS),
                "initial_forward": {
                    "scores": small_scores.detach().tolist(),
                    "loss": float(small_loss.detach()),
                },
                "initial_restore": small_initial_restore,
                "pos1_update": {
                    "loss": float(small_loss.detach()),
                    "clip_scale": small_clip,
                    "delta_theta": small_delta_pos1,
                },
                "pos2_parent": {
                    "pre_loss": float(small_pre_loss.detach()),
                    "clip_scale": small_clip_b,
                    "post_scores": small_post_scores.detach().tolist(),
                },
                "step1_restore": small_step1_restore,
                "adam_calls": 3,
                "bundle_pos0": record_small_a,
                "bundle_pos1": record_small_b,
            }
        )
        small_bundle_a.unlink(missing_ok=True)
        small_bundle_b.unlink(missing_ok=True)
        small_result_a.unlink(missing_ok=True)
        small_result_b.unlink(missing_ok=True)

        parent_sha_after = sha256_text(parent_path)
        check(
            report["checks"],
            "parent_bundle_unchanged",
            parent_sha_before == parent_sha_after == EXPECTED_BUNDLE_SHA,
            {"before": parent_sha_before, "after": parent_sha_after},
        )
        parent_digest_after = parameter_digest(
            {name: parameter.detach() for name, parameter in parent.named_parameters()}
        )
        check(
            report["checks"],
            "parent_parameters_unchanged",
            parent_digest_after == EXPECTED_CANDIDATE,
            parent_digest_after,
        )
        check(
            report["checks"],
            "adam_calls_within_budget",
            adam_calls + worker_adam <= MAX_ADAM_CALLS,
            {"parent": adam_calls, "worker": worker_adam, "total": adam_calls + worker_adam},
        )
        peak_products = sum(float(record["bytes"]) for record in product_records)
        check(
            report["checks"],
            "product_bytes_within_budget",
            peak_products <= MAX_PRODUCT_BYTES,
            {"recorded_bundle_bytes": peak_products, "limit": MAX_PRODUCT_BYTES},
        )
        report["cost"] = {
            "adam_calls_parent": adam_calls,
            "adam_calls_worker": worker_adam,
            "adam_calls_total": adam_calls + worker_adam,
            "worker_cpu_seconds": worker_cpu,
            "recorded_bundle_bytes": peak_products,
            "retained_products": [
                MANIFEST,
                str(args.output) if args.output else "stdout",
            ],
        }
        report["type_check"] = run_type_check()
        report["lint"] = run_lint()
        check(report["checks"], "type_check", report["type_check"]["passed"], report["type_check"])
        check(report["checks"], "lint", report["lint"]["passed"], report["lint"])
        cpu_seconds = time.process_time() - started[0]
        wall_seconds = time.perf_counter() - started[1]
        total_cpu = cpu_seconds + worker_cpu
        over_budget = total_cpu > BUDGET_CPU_SECONDS or wall_seconds > BUDGET_WALL_SECONDS
        check(
            report["checks"],
            "budget",
            not over_budget,
            {
                "cpu_seconds": total_cpu,
                "wall_seconds": wall_seconds,
                "cpu_budget": BUDGET_CPU_SECONDS,
                "wall_budget": BUDGET_WALL_SECONDS,
            },
        )
        report["cost"].update(
            {
                "cpu_seconds": total_cpu,
                "wall_seconds": wall_seconds,
                "cpu_seconds_parent": cpu_seconds,
            }
        )
        failed = [
            row["id"] for row in report["checks"] if not row["passed"] and row["id"] != "budget"
        ]
        if failed:
            report["status"] = "engineering_failure"
            report["failed_checks"] = failed
        elif over_budget:
            report["status"] = "budget_stop"
        else:
            report["status"] = "passed_10_21_graph_engineering_only"
    except Exception as exc:  # noqa: BLE001
        report["status"] = "engineering_failure"
        report["engineering_failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    if report.get("status") == "engineering_failure" and "failed_checks" not in report:
        failed = [row["id"] for row in report["checks"] if not row["passed"]]
        if failed:
            report["failed_checks"] = failed
    report["cpu_seconds"] = time.process_time() - started[0] + worker_cpu
    report["wall_seconds"] = time.perf_counter() - started[1]
    report["capability_claimed"] = False
    report["training_admitted"] = False
    report["pricing_admitted"] = False
    report["weights_published"] = False
    report["data_splits_read"] = [
        "public engineering fixtures: native_goal_attributes/native_goal_glossary"
    ]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    print(
        json.dumps(
            {
                "status": report["status"],
                "failed": report.get("failed_checks", []),
                "adam_calls": report.get("cost", {}).get("adam_calls_total"),
                "cpu_seconds": round(report["cpu_seconds"], 3),
                "wall_seconds": round(report["wall_seconds"], 3),
            },
            ensure_ascii=True,
        )
    )
    return 0


def small_delta_rows(before: dict, module) -> dict:
    rows = {}
    for group in PARAMETER_GROUPS:
        delta = 0.0
        for name in PARAMETER_GROUPS[group]:
            parameter = getattr(module, name).detach()
            delta = max(delta, float((parameter - before[group][name]).abs().max()))
        rows[group] = delta
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
