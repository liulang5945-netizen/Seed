"""14.4 record-relation graph, intervention, credit and restore gate.

This script runs one declared engineering package only.  It does not train a
task, does not price a model, never reads the sealed confirm split and never
publishes a capability claim.

Stages:
  graph   -- the 14.4 formal graph gate (this file's only implemented stage)
  price   -- reserved for 14.5
  compare -- reserved for 14.6
  confirm -- reserved for 14.7
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import multiprocessing
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import contextlib

import torch  # noqa: E402

from taiji.candidate_manifest import build_manifest  # noqa: E402
from taiji.core_09_bundle import BUNDLE_RELATIVE_PATH as CORE_09_BUNDLE_PATH  # noqa: E402
from taiji.core_09_bundle import EXPECTED_BYTES as CORE_09_BUNDLE_BYTES  # noqa: E402
from taiji.core_09_bundle import EXPECTED_SHA256 as CORE_09_BUNDLE_SHA  # noqa: E402
from taiji.living_core import EOS, RawSource  # noqa: E402
from taiji.native_goal_glossary import glossary_records  # noqa: E402
from taiji.record_relation import (  # noqa: E402
    RecordRelationConfig,
    expected_parameter_count,
)
from taiji.record_relation_core import (  # noqa: E402
    ARM_FLAT,
    ARM_FULL,
    ARM_LOCAL,
    ARMS,
    CLIP_NORM,
    CONTENT_REVISION,
    DISABLED_BY_ARM,
    DISABLED_NAMES_BY_ARM,
    NEW_PARAMETER_GROUPS,
    PARAMETER_GROUPS,
    SUBJECT_TRAINABLE_GROUPS,
    answer_step,
    build_candidate,
    candidate_digest,
    candidate_parameter_map,
    combined_loss,
    decode_endpoint,
    forced_read,
    generate_bytes,
    group_delta,
    make_optimizer,
    observe_goal,
    optimizer_moment_rows,
    prepare,
    run_worker,
    save_bundle,
    score_example,
    score_from_state,
)

BUNDLE = CORE_09_BUNDLE_PATH
EXPECTED_CANDIDATE = "feb0cbcb00e2e16bb66b77db530ae566ce40eecaf2e14852b7cc1661292139dd"
EXPECTED_BUNDLE_SHA = CORE_09_BUNDLE_SHA
EXPECTED_BUNDLE_BYTES = CORE_09_BUNDLE_BYTES
MANIFEST = "plans/manifests/taiji_core_record_relation_14_4_20261102.json"
REPORT = "reports/taiji_core_record_relation_14_4_20261102.json"
OUTPUT_DIR = "output/taiji_core_record_relation_14"

BUDGET_CPU_SECONDS = 600
BUDGET_WALL_SECONDS = 1200
MAX_ADAM_CALLS = 12
MAX_PRODUCT_BYTES = 2 * 1024**3
FD_EPS_F64 = (1e-4, 1e-5, 1e-6)
FD_ATOL_F64 = 1e-10
FD_RTOL_F64 = 1e-4
FD_EPS_F32 = (1e-2, 1e-3, 1e-4)
FD_ATOL_F32 = 1e-6
FD_RTOL_F32 = 1e-2
SIGNAL_FACTOR = 10.0
ASCENT_STEP = 1e-3
WORKER_TIMEOUT = 240

SOURCE_FILES = (
    "taiji/record_relation.py",
    "taiji/record_relation_core.py",
    "scripts/training/verify_taiji_record_relation_14.py",
    "tests/taiji_native/test_record_relation_14_4.py",
)

EXPECTED_NEW_GROUPS = {
    "gru_forward": 37248,
    "gru_backward": 37248,
    "W_c": 16512,
    "pool": 16640,
    "Q_goal": 4128,
    "K_goal": 4096,
    "V_goal": 16512,
    "Q_read": 10272,
    "K_record": 4224,
    "null_record": 160,
    "Q_value": 5152,
    "K_position": 4096,
    "V_position": 16512,
    "eta": 1,
}

FORWARD_FUNCTIONS = {
    "prepare",
    "observe_goal",
    "answer_step",
    "advance_h",
    "generate_bytes",
    "decode_endpoint",
    "encode_units",
    "build_view",
    "pool",
    "read",
    "_records",
    "_flat",
    "_forced",
    "_null_only",
    "goal_attention",
}
BANNED_NAMES = {
    "gold",
    "span",
    "target",
    "correct_source",
    "scorer",
    "render",
    "oracle",
    "action_index",
    "answer_index",
}
RAW_FIELDS = {"source_id", "raw", "observed_at", "version", "owner", "scope", "real"}


class BudgetStop(RuntimeError):
    """The declared CPU, wall, Adam or product budget was exhausted."""


class InstrumentError(RuntimeError):
    """An identity or schema invariant failed before the graph could be trusted."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def check(
    rows: list,
    check_id: str,
    passed: bool,
    observed,
    note: str = "",
    *,
    required: bool = True,
) -> None:
    rows.append(
        {
            "id": check_id,
            "passed": bool(passed),
            "required": bool(required),
            "observed": observed,
            "note": note,
        }
    )


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


def max_abs_diff(first: torch.Tensor, second: torch.Tensor) -> float:
    return float((first.detach() - second.detach()).abs().max())


def build_fixtures() -> tuple[dict, dict]:
    records = glossary_records()
    first = {
        "name": "records",
        "goal": "让圆口的器皿到四方木台。",
        "records": records,
        "target_action": "把蓝色杯子放到木桌",
        "target_bytes": tuple("把蓝色杯子放到木桌".encode()) + (EOS,),
    }
    second = {
        "name": "null_only",
        "goal": "让带纹路的方块到靠窗的台面。",
        "records": (),
        "target_action": "把红色印章放到窗台",
        "target_bytes": tuple("把红色印章放到窗台".encode()) + (EOS,),
    }
    for fixture in (first, second):
        if len(str(fixture["target_action"]).encode()) != 27:
            raise InstrumentError("Fixture action is not the declared 27 bytes")
        if len(fixture["target_bytes"]) != 28:
            raise InstrumentError("Fixture target must be 27 bytes plus EOS")
    return first, second


def fixture_payload(fixture: dict) -> dict:
    return {
        "name": fixture["name"],
        "goal": fixture["goal"],
        "records": [
            {"source_id": record.source_id, "raw": record.raw.decode("utf-8")}
            for record in fixture["records"]
        ],
        "target_action": fixture["target_action"],
        "target_bytes": list(fixture["target_bytes"]),
        "goal_bytes": len(fixture["goal"].encode()),
        "action_bytes": len(fixture["target_action"].encode()),
    }


def fixture_sha(fixture: dict) -> str:
    digest = hashlib.sha256()
    digest.update(fixture["goal"].encode())
    for record in fixture["records"]:
        digest.update(record.source_id.encode())
        digest.update(record.raw)
    digest.update(fixture["target_action"].encode())
    return digest.hexdigest()


def fixture_records(fixture: dict) -> tuple[RawSource, ...]:
    return tuple(fixture["records"])


def fixture_loss_rows(fixtures: tuple[dict, ...]) -> tuple[dict, ...]:
    return tuple(
        {
            "goal": fixture["goal"],
            "records": fixture["records"],
            "target_bytes": fixture["target_bytes"],
        }
        for fixture in fixtures
    )


def no_grad_loss(candidate, fixtures) -> float:
    with torch.no_grad():
        return float(combined_loss(candidate, fixture_loss_rows(fixtures)).detach())


def group_gradients(candidate, fixtures, group: str):
    candidate.zero_grad(set_to_none=True)
    loss = combined_loss(candidate, fixture_loss_rows(fixtures))
    parameters = candidate.trainable_group_parameters(group)
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=False)
    filled = [
        torch.zeros_like(parameter) if gradient is None else gradient.detach()
        for parameter, gradient in zip(parameters, gradients, strict=True)
    ]
    norm = float(torch.sqrt(sum(gradient.square().sum() for gradient in filled)))
    return loss.detach(), parameters, filled, norm


def perturb(parameters, direction, sign: float) -> None:
    with torch.no_grad():
        for parameter, step in zip(parameters, direction, strict=True):
            parameter.add_(sign * step)


def fd_group(
    candidate,
    fixtures,
    group: str,
    *,
    eps_values: tuple[float, ...],
    atol: float,
    rtol: float,
) -> dict:
    parameters = candidate.trainable_group_parameters(group)
    row: dict[str, Any] = {
        "group": group,
        "parameter_count": int(sum(parameter.numel() for parameter in parameters)),
        "disabled": not bool(parameters),
        "grad_norm": 0.0,
        "base_loss": None,
        "fd": {},
        "fd_error": {},
        "signal_floor": SIGNAL_FACTOR * atol,
        "resolved": False,
        "note": "",
    }
    if not parameters:
        row["note"] = "disabled_by_arm"
        return row
    _loss, _parameters, gradients, norm = group_gradients(candidate, fixtures, group)
    base_loss = no_grad_loss(candidate, fixtures)
    row.update({"grad_norm": norm, "base_loss": base_loss})
    if norm <= SIGNAL_FACTOR * atol:
        row["note"] = "below_signal_floor"
        row["zero_gradient_rejected"] = True
        row["sign_flip_rejected"] = True
        row["zero_direction_rejected"] = True
        row["zero_params_loss_delta"] = 0.0
        row["ascent_loss_delta"] = 0.0
        return row
    direction = [gradient / norm for gradient in gradients]
    fd_values = []
    for eps in eps_values:
        perturb(parameters, direction, eps)
        plus = no_grad_loss(candidate, fixtures)
        perturb(parameters, direction, -2.0 * eps)
        minus = no_grad_loss(candidate, fixtures)
        perturb(parameters, direction, eps)
        forward = (plus - minus) / (2.0 * eps)
        error = abs(forward - norm) / max(norm, 1e-30)
        row["fd"][f"{eps:g}"] = forward
        row["fd_error"][f"{eps:g}"] = error
        fd_values.append((forward, error))
    consistent = fd_values[0][1] <= rtol and sum(1 for _, error in fd_values if error <= rtol) >= 2
    zero_fd = no_grad_loss(candidate, fixtures) - no_grad_loss(candidate, fixtures)
    zero_rejected = abs(zero_fd) <= SIGNAL_FACTOR * atol
    flipped = [-gradient / norm for gradient in gradients]
    eps_flip = eps_values[0]
    perturb(parameters, flipped, eps_flip)
    flip_plus = no_grad_loss(candidate, fixtures)
    perturb(parameters, flipped, -2.0 * eps_flip)
    flip_minus = no_grad_loss(candidate, fixtures)
    perturb(parameters, flipped, eps_flip)
    flip_fd = (flip_plus - flip_minus) / (2.0 * eps_flip)
    flip_rejected = flip_fd <= SIGNAL_FACTOR * atol
    snapshot = [parameter.detach().clone() for parameter in parameters]
    with torch.no_grad():
        for parameter in parameters:
            parameter.zero_()
    zero_params_loss = no_grad_loss(candidate, fixtures)
    with torch.no_grad():
        for parameter, saved in zip(parameters, snapshot, strict=True):
            parameter.copy_(saved)
    perturb(parameters, direction, -ASCENT_STEP)
    ascent_loss = no_grad_loss(candidate, fixtures)
    perturb(parameters, direction, ASCENT_STEP)
    row.update(
        {
            "eps_consistent": consistent,
            "signal": norm > SIGNAL_FACTOR * atol,
            "zero_gradient_rejected": zero_rejected,
            "zero_direction_rejected": zero_rejected,
            "sign_flip_rejected": flip_rejected,
            "flip_fd": flip_fd,
            "zero_params_loss_delta": zero_params_loss - base_loss,
            "ascent_loss_delta": ascent_loss - base_loss,
            "resolved": bool(
                consistent and zero_rejected and flip_rejected and norm > SIGNAL_FACTOR * atol
            ),
        }
    )
    if not row["resolved"]:
        row["note"] = "eps_inconsistent" if not consistent else "control_failed"
    return row


def spawn_worker(spec: dict) -> dict:
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=run_worker, args=(spec,))
    process.start()
    process.join(timeout=WORKER_TIMEOUT)
    if process.is_alive():
        process.terminate()
        process.join()
        return {"ok": False, "message": "worker timed out", "cpu_seconds": 0.0, "wall_seconds": 0.0}
    if process.exitcode != 0:
        return {
            "ok": False,
            "exitcode": process.exitcode,
            "message": "worker did not exit cleanly",
            "cpu_seconds": 0.0,
            "wall_seconds": 0.0,
        }
    path = Path(spec["result"])
    if not path.exists():
        return {
            "ok": False,
            "message": "worker wrote no result",
            "cpu_seconds": 0.0,
            "wall_seconds": 0.0,
        }
    loaded: dict = torch.load(path, map_location="cpu", weights_only=True)
    return loaded


def worker_spec(
    bundle: Path,
    result: Path,
    fixtures: tuple[dict, ...],
    *,
    dtype: torch.dtype,
    mode: str,
    previous_byte: int,
    fixture_index: int = 0,
    force: dict | None = None,
    gamma_override: float | None = None,
    gate_index: int = 0,
) -> dict:
    return {
        "bundle": str(bundle),
        "result": str(result),
        "fixtures": tuple(
            {
                "goal": fixture["goal"],
                "records": fixture["records"],
                "target_bytes": fixture["target_bytes"],
            }
            for fixture in fixtures
        ),
        "dtype": dtype_name(dtype),
        "mode": mode,
        "previous_byte": previous_byte,
        "fixture_index": fixture_index,
        "force": force,
        "gamma_override": gamma_override,
        "gate_index": gate_index,
    }


class Budget:
    def __init__(self) -> None:
        self.main_cpu = 0.0
        self.worker_cpu = 0.0
        self.wall = 0.0
        self.adam_calls = 0
        self.live_product_bytes = 0
        self.max_product_bytes = 0
        self.cumulative_product_bytes = 0
        self.peak_ram_mb = 0.0

    def total_cpu(self) -> float:
        return self.main_cpu + self.worker_cpu

    def add_worker(self, result: dict) -> None:
        self.worker_cpu += float(result.get("cpu_seconds", 0.0))
        self.wall += float(result.get("wall_seconds", 0.0))
        self.adam_calls += int(result.get("adam_calls", 0))

    def add_product(self, info: dict) -> None:
        size = int(info["bytes"])
        self.live_product_bytes += size
        self.cumulative_product_bytes += size
        self.max_product_bytes = max(self.max_product_bytes, self.live_product_bytes)

    def drop_product(self, info: dict) -> None:
        self.live_product_bytes -= int(info["bytes"])

    def sample_ram(self) -> None:
        try:
            import psutil

            used = psutil.Process().memory_info().rss / 1024**2
            self.peak_ram_mb = max(self.peak_ram_mb, used)
        except Exception:  # noqa: BLE001
            return

    def ensure(self) -> None:
        cpu_now = time.process_time()
        wall_now = time.perf_counter()
        self.main_cpu = cpu_now
        if self.total_cpu() > BUDGET_CPU_SECONDS:
            raise BudgetStop(f"CPU budget exceeded: {self.total_cpu():.2f}")
        if self.wall > BUDGET_WALL_SECONDS or wall_now - STARTED_WALL > BUDGET_WALL_SECONDS:
            raise BudgetStop(f"wall budget exceeded: {max(self.wall, wall_now - STARTED_WALL):.2f}")
        if self.adam_calls > MAX_ADAM_CALLS:
            raise BudgetStop(f"Adam budget exceeded: {self.adam_calls}")
        if self.max_product_bytes > MAX_PRODUCT_BYTES:
            raise BudgetStop(f"product budget exceeded: {self.max_product_bytes}")


STARTED_WALL = time.perf_counter()
STARTED_CPU = time.process_time()


def parameter_rows(candidate) -> dict:
    named = dict(candidate.named_parameters())
    rows = {}
    for group, names in PARAMETER_GROUPS.items():
        rows[group] = {
            "names": list(names),
            "parameter_count": int(sum(named[name].numel() for name in names)),
            "trainable_count": int(
                sum(named[name].numel() for name in names if named[name].requires_grad)
            ),
        }
    return rows


def disabled_identity_rows(candidate) -> dict:
    named = dict(candidate.named_parameters())
    rows = {}
    for name in DISABLED_NAMES_BY_ARM[candidate.arm]:
        rows[name] = {
            "numel": int(named[name].numel()),
            "requires_grad": bool(named[name].requires_grad),
        }
    return rows


def ast_scan() -> dict:
    hits = []
    scanned = []
    for relative in ("taiji/record_relation.py", "taiji/record_relation_core.py"):
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or node.name not in FORWARD_FUNCTIONS:
                continue
            scanned.append(f"{relative}:{node.name}")
            names = {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}
            attributes = {item.attr for item in ast.walk(node) if isinstance(item, ast.Attribute)}
            found = sorted((names | attributes) & BANNED_NAMES)
            if found:
                hits.append({"function": f"{relative}:{node.name}", "identifiers": found})
    return {"scanned": scanned, "hits": hits, "passed": not hits}


def trace_calls(function) -> tuple[Any, list]:
    events = []
    modules = {
        "taiji.record_relation",
        "taiji.record_relation_core",
        "taiji.living_core",
    }

    def tracer(frame, event, _arg):
        if event == "call":
            module = frame.f_globals.get("__name__", "")
            if module in modules:
                events.append(f"{module}:{frame.f_code.co_name}")
        return tracer

    sys.settrace(tracer)
    try:
        result = function()
    finally:
        sys.settrace(None)
    return result, events


def boundary_audit(candidate, fixture: dict) -> dict:
    stat = ast_scan()
    first, second = fixture["goal"].encode(), fixture["records"]
    result, events = trace_calls(lambda: generate_bytes(candidate, first, second))
    banned_calls = sorted({event for event in events if event.rsplit(":", 1)[-1] in BANNED_NAMES})
    accesses: list[str] = []

    class AuditedSource(RawSource):
        def __getattribute__(self, name):
            if name in RAW_FIELDS:
                accesses.append(name)
            return super().__getattribute__(name)

    audited = tuple(
        AuditedSource(
            source_id=record.source_id,
            raw=record.raw,
            observed_at=record.observed_at,
            version=record.version,
            owner=record.owner,
            scope=record.scope,
            real=record.real,
        )
        for record in fixture["records"]
    )
    generate_bytes(candidate, first, audited)
    fields = set(RawSource.__dataclass_fields__)
    run = {
        "events": len(events),
        "functions": sorted(set(events)),
        "banned_calls": banned_calls,
        "raw_fields_accessed": sorted(set(accesses)),
        "raw_fields_declared": sorted(fields),
        "raw_fields_extra": sorted(fields - RAW_FIELDS),
        "bytes": len(result["raw_bytes"]),
    }
    passed = (
        stat["passed"]
        and not banned_calls
        and set(accesses) <= RAW_FIELDS
        and not (fields - RAW_FIELDS)
    )
    return {"ast": stat, "call_path": run, "passed": passed}


def run_profile(
    parent_payload: dict,
    parent_identity: dict,
    fixtures: tuple[dict, ...],
    *,
    arm: str,
    dtype: torch.dtype,
    profile: str,
    output_dir: Path,
    budget: Budget,
    subject_id: str,
) -> dict:
    candidate = build_candidate(parent_payload, arm=arm, dtype=dtype)
    optimizer = make_optimizer(candidate)
    zero_state = candidate.subject.initial_state(subject_id, revision=CONTENT_REVISION)
    first = fixtures[0]
    prepared0 = prepare(candidate, zero_state, first["goal"].encode(), first["records"])
    observation_state, gate_index = observe_goal(candidate, prepared0)
    probe_initial = answer_step(
        candidate,
        prepared0,
        observation_state,
        observation_state.coverage,
        EOS,
        gate_index=gate_index,
    )
    bundle0 = output_dir / f"{profile}_initial.pt"
    result0 = output_dir / f"{profile}_initial_probe.pt"
    info0 = save_bundle(
        bundle0,
        profile=profile,
        candidate=candidate,
        state=observation_state,
        optimizer=optimizer,
        update_index=0,
        cursor={"fixture_index": 0, "profile": profile},
        parent_identity=parent_identity,
    )
    budget.add_product(info0)
    worker0 = spawn_worker(
        worker_spec(
            bundle0,
            result0,
            fixtures,
            dtype=dtype,
            mode="probe",
            previous_byte=EOS,
            fixture_index=0,
            gate_index=gate_index,
        )
    )
    budget.add_worker(worker0)
    initial_restore = {
        "ok": bool(worker0.get("ok")),
        "equal": bool(worker0.get("ok"))
        and bool(torch.equal(probe_initial.probabilities, worker0["probabilities"])),
        "max_abs_diff": (
            None
            if not worker0.get("ok")
            else max_abs_diff(probe_initial.probabilities, worker0["probabilities"])
        ),
        "state_step": None if not worker0.get("ok") else int(worker0["step"]),
    }
    before = candidate_parameter_map(candidate)

    optimizer.zero_grad()
    loss1 = combined_loss(candidate, fixture_loss_rows(fixtures))
    loss1.backward()
    clip1 = float(torch.nn.utils.clip_grad_norm_(candidate.trainable_parameters(), CLIP_NORM))
    optimizer.step()
    budget.adam_calls += 1
    budget.ensure()

    deltas = {group: group_delta(before, candidate, group) for group in PARAMETER_GROUPS}
    moments = optimizer_moment_rows(candidate, optimizer)

    score1 = score_example(
        candidate,
        first["goal"].encode(),
        first["records"],
        tuple(first["target_bytes"][:-1]),
        subject_id=subject_id,
    )
    probe1 = answer_step(
        candidate,
        score1.prepared,
        score1.prefix_state,
        score1.prefix_coverage,
        first["target_bytes"][-2],
        gate_index=0,
    )
    bundle1 = output_dir / f"{profile}_step1.pt"
    result1 = output_dir / f"{profile}_step1_probe.pt"
    info1 = save_bundle(
        bundle1,
        profile=profile,
        candidate=candidate,
        state=score1.prefix_state,
        optimizer=optimizer,
        update_index=1,
        cursor={"fixture_index": 0, "profile": profile},
        parent_identity=parent_identity,
    )
    budget.add_product(info1)
    worker1 = spawn_worker(
        worker_spec(
            bundle1,
            result1,
            fixtures,
            dtype=dtype,
            mode="update",
            previous_byte=first["target_bytes"][-2],
            fixture_index=0,
        )
    )
    budget.add_worker(worker1)
    restore_probe = {
        "ok": bool(worker1.get("ok")),
        "equal": bool(worker1.get("ok"))
        and bool(torch.equal(probe1.probabilities, worker1["probabilities"])),
        "max_abs_diff": (
            None
            if not worker1.get("ok")
            else max_abs_diff(probe1.probabilities, worker1["probabilities"])
        ),
    }

    optimizer.zero_grad()
    loss2 = combined_loss(candidate, fixture_loss_rows(fixtures))
    loss2.backward()
    clip2 = float(torch.nn.utils.clip_grad_norm_(candidate.trainable_parameters(), CLIP_NORM))
    optimizer.step()
    budget.adam_calls += 1
    budget.ensure()

    post = score_example(
        candidate,
        first["goal"].encode(),
        first["records"],
        tuple(first["target_bytes"][:-1]),
        subject_id=subject_id,
    )
    post_probe = answer_step(
        candidate,
        post.prepared,
        post.prefix_state,
        post.prefix_coverage,
        first["target_bytes"][-2],
        gate_index=0,
    )
    after_parameters = candidate_parameter_map(candidate)
    if worker1.get("ok"):
        params_equal = tree_equal(after_parameters, worker1["parameters"])
        optimizer_equal = tree_equal(optimizer.state_dict(), worker1["optimizer"])
        post_equal = bool(torch.equal(post_probe.probabilities, worker1["post_probabilities"]))
        post_diff = max_abs_diff(post_probe.probabilities, worker1["post_probabilities"])
    else:
        params_equal = optimizer_equal = post_equal = False
        post_diff = None

    for info, _path in (
        (info0, bundle0),
        (info1, bundle1),
    ):
        budget.drop_product(info)
    for path in (bundle0, bundle1, result0, result1):
        if path.exists():
            path.unlink()
    budget.sample_ram()
    return {
        "profile": profile,
        "arm": arm,
        "dtype": dtype_name(dtype),
        "parameter_count": candidate.parameter_count(),
        "trainable_count": int(
            sum(parameter.numel() for parameter in candidate.trainable_parameters())
        ),
        "disabled": candidate.disabled_rows(),
        "candidate_digest": candidate_digest(candidate),
        "initial_restore": initial_restore,
        "step1_restore": restore_probe,
        "step2_restore": {
            "ok": bool(worker1.get("ok")),
            "parameters_equal": params_equal,
            "optimizer_equal": optimizer_equal,
            "post_probabilities_equal": post_equal,
            "post_max_abs_diff": post_diff,
        },
        "losses": {
            "step1": float(loss1.detach()),
            "step2": float(loss2.detach()),
            "worker_loss": None if not worker1.get("ok") else float(worker1["loss"]),
        },
        "clip_scale": {"step1": clip1, "step2": clip2},
        "deltas": deltas,
        "moments": moments,
        "adam_calls": 3,
        "worker_cpu_seconds": float(worker0.get("cpu_seconds", 0.0))
        + float(worker1.get("cpu_seconds", 0.0)),
        "worker_wall_seconds": float(worker0.get("wall_seconds", 0.0))
        + float(worker1.get("wall_seconds", 0.0)),
    }


def run_relationship_checks(parent: dict, fixtures: tuple[dict, ...]) -> dict:
    record = fixtures[0]["records"][0]
    early, late = 5, 20
    replacement = record.raw[:18] + "红色印章".encode() + record.raw[30:]
    if len(replacement) != len(record.raw):
        raise InstrumentError("Discrete record replacement changed the length")
    rows = {}
    for arm in (ARM_FULL, ARM_LOCAL):
        candidate = build_candidate(parent, arm=arm)
        units = candidate.relation.encode_units(candidate.subject, record.raw)
        gradient = torch.autograd.grad(
            units[early].sum(), candidate.subject.encoder.embedding.weight, allow_unused=True
        )[0]
        derivative = float(gradient[record.raw[late]].abs().max())
        replaced = candidate.relation.encode_units(candidate.subject, replacement)
        delta = float((replaced[early].detach() - units[early].detach()).abs().max())
        view = candidate.relation.build_view(candidate.subject, fixtures[0]["records"])
        alone = candidate.relation.encode_units(candidate.subject, record.raw)
        other_delta = float((view.units[0].detach() - alone.detach()).abs().max())
        del candidate
        gc.collect()
        rows[arm] = {
            "derivative_early_wrt_late_byte": derivative,
            "discrete_replacement_early_delta": delta,
            "other_records_early_delta": other_delta,
        }
    full = rows[ARM_FULL]
    local = rows[ARM_LOCAL]
    return {
        "rows": rows,
        "passed": full["derivative_early_wrt_late_byte"] > 1e-6
        and full["discrete_replacement_early_delta"] > 1e-6
        and local["derivative_early_wrt_late_byte"] == 0.0
        and local["discrete_replacement_early_delta"] == 0.0
        and full["other_records_early_delta"] == 0.0
        and local["other_records_early_delta"] == 0.0,
    }


def run_content_checks(
    parent_payload: dict,
    parent_identity: dict,
    fixtures: tuple[dict, ...],
    output_dir: Path,
    budget: Budget,
) -> dict:
    candidate = build_candidate(parent_payload)
    first = fixtures[0]
    result = score_example(
        candidate,
        first["goal"].encode(),
        first["records"],
        tuple(first["target_bytes"][:-1]),
    )
    observation_state = result.observation_state
    coverage = result.observation_coverage
    prepared = result.prepared
    view = prepared.view
    correct_index = next(index for index, raw in enumerate(view.raw) if "蓝色杯子".encode() in raw)
    correct_offset = view.raw[correct_index].index("蓝色杯子".encode())
    correct_byte = view.raw[correct_index][correct_offset]
    wrong_index = next(index for index, raw in enumerate(view.raw) if "木桌".encode() in raw)
    wrong_offset = view.raw[wrong_index].index("木桌".encode())
    wrong_byte = view.raw[wrong_index][wrong_offset]

    unforced = answer_step(candidate, prepared, observation_state, coverage, EOS, gate_index=0)
    normalization = {
        "copy_sum": float(unforced.read.copy_weights.sum().detach()),
        "alpha_null": float(unforced.read.alpha_null.detach()),
        "total": float((unforced.read.copy_weights.sum() + unforced.read.alpha_null).detach()),
        "byte_mass_sum": float(unforced.read.copy_byte_mass.sum().detach()),
        "probability_sum": float(unforced.probabilities.sum().detach()),
        "copy_eos_mass": float(unforced.read.copy_byte_mass[EOS].detach()),
        "copy_weights_matches_mass": bool(
            torch.equal(
                unforced.read.copy_weights.sum().reshape(1),
                unforced.read.copy_byte_mass.sum().reshape(1),
            )
        ),
    }
    correct = forced_read(
        candidate,
        prepared,
        observation_state,
        coverage,
        EOS,
        record_index=correct_index,
        offset=correct_offset,
        gamma_override=0.0,
    )
    correct_row = {
        "record_index": correct_index,
        "offset": correct_offset,
        "byte": correct_byte,
        "probability": float(correct.probabilities[correct_byte].detach()),
        "sum": float(correct.probabilities.sum().detach()),
        "eos": float(correct.probabilities[EOS].detach()),
        "passed": abs(float(correct.probabilities[correct_byte].detach()) - 1.0) <= 1e-12
        and abs(float(correct.probabilities.sum().detach()) - 1.0) <= 1e-12
        and abs(float(correct.probabilities[EOS].detach())) <= 1e-12,
    }
    wrong = forced_read(
        candidate,
        prepared,
        observation_state,
        coverage,
        EOS,
        record_index=wrong_index,
        offset=wrong_offset,
        gamma_override=0.0,
    )
    wrong_row = {
        "record_index": wrong_index,
        "offset": wrong_offset,
        "byte": wrong_byte,
        "probability": float(wrong.probabilities[wrong_byte].detach()),
        "correct_byte_mass": float(wrong.probabilities[correct_byte].detach()),
        "passed": abs(float(wrong.probabilities[wrong_byte].detach()) - 1.0) <= 1e-12
        and abs(float(wrong.probabilities[correct_byte].detach())) <= 1e-12,
    }
    null = forced_read(
        candidate,
        prepared,
        observation_state,
        coverage,
        EOS,
        record_index=None,
        offset=None,
        gamma_override=0.0,
    )
    null_row = {
        "equals_generated": bool(torch.equal(null.probabilities, null.generated)),
        "sum": float(null.probabilities.sum().detach()),
        "eos": float(null.probabilities[EOS].detach()),
        "passed": bool(torch.equal(null.probabilities, null.generated)),
    }
    reversed_records = tuple(reversed(first["records"]))
    reversed_view = candidate.relation.build_view(candidate.subject, reversed_records)
    prepared_rev = prepare(
        candidate,
        observation_state,
        first["goal"].encode(),
        reversed_records,
    )
    reversed_entity = next(
        index for index, raw in enumerate(reversed_view.raw) if "左盒".encode() in raw
    )
    reversed_offset = reversed_view.raw[reversed_entity].index("左盒".encode())
    reversed_byte = reversed_view.raw[reversed_entity][reversed_offset]
    permuted = forced_read(
        candidate,
        prepared_rev,
        observation_state,
        prepared_rev.coverage,
        EOS,
        record_index=0,
        offset=reversed_offset if reversed_entity == 0 else 15,
        gamma_override=0.0,
    )
    permutation_row = {
        "reversed_entity_index": reversed_entity,
        "forced_index": 0,
        "byte": reversed_byte,
        "probability": float(permuted.probabilities[reversed_byte].detach()),
        "original_correct_byte_mass": float(permuted.probabilities[correct_byte].detach()),
        "passed": abs(float(permuted.probabilities[reversed_byte].detach()) - 1.0) <= 1e-12
        and abs(float(permuted.probabilities[correct_byte].detach())) <= 1e-12,
    }

    bundle = output_dir / "content_restore.pt"
    probe_result_path = output_dir / "content_restore_probe.pt"
    optimizer = make_optimizer(candidate)
    info = save_bundle(
        bundle,
        profile="content-restore",
        candidate=candidate,
        state=observation_state,
        optimizer=optimizer,
        update_index=0,
        cursor={"probe": "content"},
        parent_identity=parent_identity,
    )
    budget.add_product(info)
    worker = spawn_worker(
        worker_spec(
            bundle,
            probe_result_path,
            fixtures,
            dtype=torch.float64,
            mode="probe",
            previous_byte=EOS,
            fixture_index=0,
            force={
                "alpha": [
                    1.0 if index == correct_index else 0.0 for index in range(view.record_count)
                ]
                + [0.0],
                "betas": [
                    [
                        1.0 if index == correct_offset else 0.0
                        for index in range(view.lengths[record])
                    ]
                    for record in range(view.record_count)
                ],
            },
            gamma_override=0.0,
        )
    )
    budget.add_worker(worker)
    restore_row = {
        "ok": bool(worker.get("ok")),
        "equal": bool(worker.get("ok"))
        and bool(torch.equal(correct.probabilities, worker["probabilities"])),
        "max_abs_diff": (
            None
            if not worker.get("ok")
            else max_abs_diff(correct.probabilities, worker["probabilities"])
        ),
    }
    bundle.unlink()
    if probe_result_path.exists():
        probe_result_path.unlink()
    budget.drop_product(info)

    return {
        "normalization": normalization,
        "correct": correct_row,
        "wrong": wrong_row,
        "null": null_row,
        "permutation": permutation_row,
        "restore": restore_row,
        "passed": abs(normalization["total"] - 1.0) <= 1e-12
        and abs(normalization["byte_mass_sum"] - normalization["copy_sum"]) <= 1e-12
        and normalization["copy_eos_mass"] == 0.0
        and correct_row["passed"]
        and wrong_row["passed"]
        and null_row["passed"]
        and permutation_row["passed"]
        and restore_row["equal"],
    }


def run_host_checks(parent: dict, fixtures: tuple[dict, ...]) -> dict:
    candidate = build_candidate(parent)
    first = fixtures[0]
    prefix = tuple(first["target_bytes"][:-1])
    wrong = tuple(fixtures[1]["target_bytes"][:-1])
    base = score_example(candidate, first["goal"].encode(), first["records"], prefix)
    swapped = score_example(
        candidate, first["goal"].encode(), first["records"], prefix, labels=wrong
    )
    label_row = {
        "distributions_equal": bool(torch.equal(base.distributions, swapped.distributions)),
        "base_nll": float(base.nll.detach()),
        "swapped_nll": float(swapped.nll.detach()),
        "nll_differ": abs(float(base.nll.detach()) - float(swapped.nll.detach())) > 1e-9,
    }

    mapping = {record.source_id: f"r14-{index}" for index, record in enumerate(first["records"])}
    renamed = tuple(
        RawSource(
            source_id=mapping[record.source_id],
            raw=record.raw,
            observed_at=record.observed_at,
            version=record.version,
            owner=record.owner,
            scope=record.scope,
            real=record.real,
        )
        for record in first["records"]
    )
    renamed_state = replace(
        base.prefix_state,
        keys=tuple(
            (mapping[source], version, offset) for source, version, offset in base.prefix_state.keys
        ),
    )
    prepared_renamed = prepare(candidate, renamed_state, first["goal"].encode(), renamed)
    original_step = answer_step(
        candidate,
        base.prepared,
        base.prefix_state,
        base.prefix_coverage,
        first["target_bytes"][-2],
        gate_index=0,
    )
    renamed_step = answer_step(
        candidate,
        prepared_renamed,
        renamed_state,
        renamed_state.coverage,
        first["target_bytes"][-2],
        gate_index=0,
    )
    identity_row = {
        "equal": bool(torch.equal(original_step.probabilities, renamed_step.probabilities)),
        "max_abs_diff": max_abs_diff(original_step.probabilities, renamed_step.probabilities),
        "coverage_length": int(renamed_state.coverage.numel()),
    }

    wrong_byte = "红色印章".encode()[0]
    wrong_distribution = torch.zeros(257, dtype=torch.float64)
    wrong_distribution[wrong_byte] = 1.0
    wrong_endpoint = decode_endpoint(wrong_distribution)
    eos_distribution = torch.zeros(257, dtype=torch.float64)
    eos_distribution[EOS] = 1.0
    eos_endpoint = decode_endpoint(eos_distribution)
    endpoint_row = {
        "wrong_byte": wrong_byte,
        "wrong_endpoint_byte": wrong_endpoint["byte"],
        "wrong_endpoint_eos": wrong_endpoint["eos"],
        "wrong_preserved": wrong_endpoint["byte"] == wrong_byte
        and not wrong_endpoint["eos"]
        and bool(torch.equal(wrong_endpoint["probabilities"], wrong_distribution)),
        "eos_endpoint_byte": eos_endpoint["byte"],
        "eos_endpoint_eos": eos_endpoint["eos"],
        "eos_preserved": eos_endpoint["eos"]
        and eos_endpoint["byte"] == EOS
        and bool(torch.equal(eos_endpoint["probabilities"], eos_distribution)),
    }

    audit = boundary_audit(candidate, first)
    return {
        "label_isolation": label_row,
        "identity_isolation": identity_row,
        "output_negative_control": endpoint_row,
        "boundary_audit": audit,
        "passed": label_row["distributions_equal"]
        and label_row["nll_differ"]
        and identity_row["equal"]
        and endpoint_row["wrong_preserved"]
        and endpoint_row["eos_preserved"]
        and audit["passed"],
    }


def run_state_checks(parent: dict, fixtures: tuple[dict, ...]) -> dict:
    candidate = build_candidate(parent)
    first = fixtures[0]
    labels = tuple(first["target_bytes"][:-1])
    result = score_example(candidate, first["goal"].encode(), first["records"], labels)
    baseline = float(result.nll)
    zero_state = replace(result.observation_state, h=torch.zeros_like(result.observation_state.h))
    _distributions, zero_nll = score_from_state(
        candidate,
        result.prepared,
        zero_state,
        result.observation_coverage,
        labels,
    )
    groups = result.observation_state.h.shape[0]
    order = torch.tensor(list(reversed(range(groups))))
    shuffled_state = replace(result.observation_state, h=result.observation_state.h[order])
    _d, shuffled_nll = score_from_state(
        candidate,
        result.prepared,
        shuffled_state,
        result.observation_coverage,
        labels,
    )
    return {
        "baseline_nll": baseline,
        "zero_h_nll": float(zero_nll.detach()),
        "zero_h_delta": abs(float(zero_nll.detach()) - baseline),
        "shuffled_h_nll": float(shuffled_nll.detach()),
        "shuffled_h_delta": abs(float(shuffled_nll.detach()) - baseline),
        "state_effect_observed": abs(float(zero_nll.detach()) - baseline) > 1e-9,
    }


def read_parent() -> tuple[dict, dict]:
    path = ROOT / BUNDLE
    payload = torch.load(path, map_location="cpu", weights_only=True)
    manifest = build_manifest(ROOT, path)
    return payload, manifest


def initial_manifest(
    parent: dict,
    parent_manifest: dict,
    candidate,
    fixtures: tuple[dict, ...],
    source_hashes: dict,
) -> dict:
    parameters = parameter_rows(candidate)
    new_parameters = {group: parameters[group]["parameter_count"] for group in NEW_PARAMETER_GROUPS}
    inherited = {group: parameters[group]["parameter_count"] for group in SUBJECT_TRAINABLE_GROUPS}
    return {
        "schema": "taiji-record-relation-manifest-v1",
        "step": "14.4",
        "date": "2026-11-02",
        "content_schema": "taiji-record-relation-v1",
        "parent": {
            "candidate_id": parent_manifest["candidate_id"],
            "bundle": BUNDLE,
            "sha256": parent_manifest["provenance"]["sha256"],
            "bytes": parent_manifest["provenance"]["bytes"],
            "subject_id": parent["subject_id"],
            "revision": parent["revision"],
            "step": parent["step"],
            "config": parent["config"],
            "growth_parent_units": parent["growth_parent_units"],
            "growth_unit_branches": parent["growth_unit_branches"],
            "growth_gates": parent["growth_gates"],
        },
        "candidate": {
            "candidate_id": candidate_digest(candidate),
            "arm": candidate.arm,
            "relation_config": asdict(candidate.config),
            "total_parameters": candidate.parameter_count(),
            "trainable_parameters": int(
                sum(parameter.numel() for parameter in candidate.trainable_parameters())
            ),
            "new_parameters": candidate.new_parameter_count(),
            "new_groups": new_parameters,
            "inherited_groups": inherited,
            "disabled": candidate.disabled_rows(),
            "internal_mode": 0,
            "fast_write": "disabled",
            "function_neutral_birth": False,
        },
        "arms": {
            arm: {
                "disabled_groups": list(DISABLED_BY_ARM[arm]),
                "disabled_names": list(DISABLED_NAMES_BY_ARM[arm]),
            }
            for arm in ARMS
        },
        "state_schema": {
            "content_revision": CONTENT_REVISION,
            "keys": "tuple[(source_id, version, byte_offset)]",
            "coverage": "float tensor aligned to keys, zero at answer start",
            "h": list(candidate.subject.initial_state("schema", revision=1).h.shape),
            "fast": list(candidate.subject.initial_state("schema", revision=1).fast.shape),
            "per_sample": "isolated zero H/F copy",
        },
        "fixtures": [
            {**fixture_payload(fixture), "sha256": fixture_sha(fixture)} for fixture in fixtures
        ],
        "seeds": {
            "relation_module": 27031,
            "relation_reader": 27032,
            "data_order": [37031, 37032],
        },
        "budget": {
            "cpu_seconds": BUDGET_CPU_SECONDS,
            "wall_seconds": BUDGET_WALL_SECONDS,
            "adam_calls": MAX_ADAM_CALLS,
            "product_bytes": MAX_PRODUCT_BYTES,
            "peak_ram_gib": 8,
        },
        "source_hashes": source_hashes,
        "limits": (
            "Engineering graph/intervention/credit/restore gate only. Public synthetic "
            "fixtures; no task data, no pricing, no capability claim, no adoption."
        ),
    }


ARCHIVED_WRITE_PREFIXES = ("reports/", "plans/manifests/")


def refuse_archived_overwrite(path: Path) -> None:
    """默认拒写已归档的读数件。

    这些报告与清单是 2026-11-02 当年跑出来的原始读数；本脚本当前版本既缺当年的输入件
    （`output/taiji_core_09/subject_bundle.pt` 不在库），也没有记录运行条件，重跑即使同路径
    落盘也不构成复核，只会把原始读数覆盖掉。要出新读数，请用 `--report`／`--manifest`
    指向一个尚不存在的路径。
    """
    try:
        rel = path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return
    if path.exists() and rel.startswith(ARCHIVED_WRITE_PREFIXES):
        raise SystemExit(
            f"拒绝覆盖已归档读数 {rel}：本脚本缺当年输入件与运行条件，"
            "重写该文件会让原始读数不可追溯；请改用 --report／--manifest 指向新路径。"
        )


def write_json(path: Path, payload: dict) -> dict:
    refuse_archived_overwrite(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {"path": str(path), "sha256": sha256_text(path), "bytes": path.stat().st_size}


def graph_main(args) -> int:
    report_path = Path(args.report) if args.report else ROOT / REPORT
    manifest_path = Path(args.manifest) if args.manifest else ROOT / MANIFEST
    output_dir = Path(args.payload_dir) if args.payload_dir else ROOT / OUTPUT_DIR
    if report_path.exists():
        raise FileExistsError("14.4 graph card is write-once")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    budget = Budget()
    checks: list[dict] = []
    report: dict[str, Any] = {
        "schema": "taiji-record-relation-graph-v1",
        "step": "14.4",
        "date": "2026-11-02",
        "status": "engineering_failure",
        "checks": checks,
        "identity": {},
        "module": {},
        "fixtures": [],
        "fd": {},
        "relationship": {},
        "content": {},
        "host": {},
        "state": {},
        "profiles": [],
        "cost": {},
        "type_check": {},
        "lint": {},
        "limits": (
            "Engineering only. The parent is the fixed 09.8 child; both fixtures are "
            "public synthetic strings. No task training, no pricing, no capability "
            "claim, no published weights, no sealed confirm read."
        ),
    }
    error: str | None = None
    status_override: str | None = None
    parent_path = ROOT / BUNDLE
    parent_sha_before = sha256_text(parent_path)
    source_hashes = {relative: sha256_text(ROOT / relative) for relative in SOURCE_FILES}
    try:
        parent, parent_manifest = read_parent()
        if (
            parent_manifest["candidate_id"] != EXPECTED_CANDIDATE
            or parent_manifest["provenance"]["sha256"] != EXPECTED_BUNDLE_SHA
            or parent_manifest["provenance"]["bytes"] != EXPECTED_BUNDLE_BYTES
        ):
            raise InstrumentError("Parent identity drifted from the fixed 09.8 asset")
        identity = {
            "parent_candidate_id": parent_manifest["candidate_id"],
            "parent_bundle_sha256": parent_manifest["provenance"]["sha256"],
            "parent_bundle_bytes": parent_manifest["provenance"]["bytes"],
            "parent_subject_id": parent["subject_id"],
            "parent_revision": parent["revision"],
            "parent_step": parent["step"],
            "parent_config": parent["config"],
            "growth_parent_units": parent["growth_parent_units"],
            "growth_unit_branches": parent["growth_unit_branches"],
            "growth_gates": parent["growth_gates"],
            "source_hashes": source_hashes,
        }
        report["identity"] = identity
        check(
            checks,
            "parent_identity",
            True,
            {
                "candidate_id": parent_manifest["candidate_id"],
                "sha256": parent_manifest["provenance"]["sha256"],
                "bytes": parent_manifest["provenance"]["bytes"],
            },
        )
        parent_sha_loaded = sha256_text(parent_path)
        check(
            checks,
            "parent_asset_untouched",
            parent_sha_loaded == parent_sha_before,
            {"sha256": parent_sha_loaded},
        )

        fixtures = build_fixtures()
        report["fixtures"] = [
            {**fixture_payload(fixture), "sha256": fixture_sha(fixture)} for fixture in fixtures
        ]
        candidate = build_candidate(parent)

        expected_groups = {
            group: sum(parameter.numel() for parameter in candidate.group_parameters(group))
            for group in NEW_PARAMETER_GROUPS
        }
        check(
            checks,
            "new_parameter_count",
            candidate.new_parameter_count() == 172801
            and RecordRelationConfig().parameter_count() == 172801
            and expected_parameter_count(128, 32, 64) == 172801,
            {
                "module": candidate.new_parameter_count(),
                "formula": RecordRelationConfig().parameter_count(),
            },
        )
        check(
            checks,
            "new_parameter_groups",
            expected_groups == EXPECTED_NEW_GROUPS,
            expected_groups,
        )
        rows = parameter_rows(candidate)
        inherited_total = sum(rows[group]["parameter_count"] for group in SUBJECT_TRAINABLE_GROUPS)
        new_total = sum(rows[group]["parameter_count"] for group in NEW_PARAMETER_GROUPS)
        trainable_total = int(
            sum(parameter.numel() for parameter in candidate.trainable_parameters())
        )
        check(
            checks,
            "parameter_accounting",
            trainable_total == inherited_total + new_total - candidate.disabled_rows()["elements"],
            {
                "total_parameters": candidate.parameter_count(),
                "trainable": trainable_total,
                "inherited_plastic": inherited_total,
                "new": new_total,
                "disabled": candidate.disabled_rows()["elements"],
            },
        )
        report["module"] = {
            "name": "RecordRelationCandidate",
            "arm": candidate.arm,
            "config": asdict(candidate.config),
            "parameter_count": candidate.parameter_count(),
            "trainable_count": trainable_total,
            "new_parameter_count": new_total,
            "inherited_parameter_count": inherited_total,
            "groups": rows,
            "disabled": candidate.disabled_rows(),
            "growth_parent_units": getattr(candidate.subject, "growth_parent_units", None),
            "content_revision": CONTENT_REVISION,
        }
        arm_rows: dict[str, Any] = {}
        for arm in ARMS:
            arm_candidate = build_candidate(parent, arm=arm)
            arm_rows[arm] = {
                "trainable": int(
                    sum(parameter.numel() for parameter in arm_candidate.trainable_parameters())
                ),
                "disabled": arm_candidate.disabled_rows(),
                "disabled_identity": disabled_identity_rows(arm_candidate),
                "digest": candidate_digest(arm_candidate),
            }
            del arm_candidate
            gc.collect()
        check(
            checks,
            "arm_contracts",
            all(row["disabled"]["elements"] >= 0 for row in arm_rows.values())
            and arm_rows[ARM_FULL]["disabled"]["elements"] == 0
            and arm_rows[ARM_LOCAL]["disabled"]["elements"] == 91008
            and arm_rows[ARM_FLAT]["disabled"]["elements"] == 9504,
            arm_rows,
        )
        report["module"]["arms"] = arm_rows

        manifest_payload = initial_manifest(
            parent, parent_manifest, candidate, fixtures, source_hashes
        )
        if manifest_path.exists():
            stored = json.loads(manifest_path.read_text(encoding="utf-8"))
            drift = {
                "candidate_id": stored.get("candidate", {}).get("candidate_id")
                != manifest_payload["candidate"]["candidate_id"],
                "source_hashes": stored.get("source_hashes") != manifest_payload["source_hashes"],
                "parent": stored.get("parent", {}).get("sha256")
                != manifest_payload["parent"]["sha256"],
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
            "path": (
                str(manifest_path.relative_to(ROOT))
                if manifest_path.is_relative_to(ROOT)
                else str(manifest_path)
            ),
            "sha256": sha256_text(manifest_path),
            "candidate_id": manifest_payload["candidate"]["candidate_id"],
            "written": manifest_written,
            "drift": drift,
        }
        check(
            checks, "manifest_identity", manifest_ok, {"written": manifest_written, "drift": drift}
        )

        report["relationship"] = run_relationship_checks(parent, fixtures)
        check(
            checks,
            "record_relationship_path",
            report["relationship"]["passed"],
            report["relationship"],
        )

        content = run_content_checks(parent, identity, fixtures, output_dir, budget)
        report["content"] = content
        check(checks, "content_path", content["passed"], content)

        host = run_host_checks(parent, fixtures)
        report["host"] = host
        check(checks, "host_semantic_isolation", host["passed"], host)

        state = run_state_checks(parent, fixtures)
        report["state"] = state
        check(
            checks,
            "state_participation",
            state["state_effect_observed"],
            state,
            "If false, the answer depends only on the current source; continuous-"
            "cognition scope is reduced.",
            required=False,
        )

        fd_candidate = build_candidate(parent)
        fd_rows = {}
        for group in PARAMETER_GROUPS:
            fd_rows[group] = fd_group(
                fd_candidate,
                fixtures,
                group,
                eps_values=FD_EPS_F64,
                atol=FD_ATOL_F64,
                rtol=FD_RTOL_F64,
            )
            budget.ensure()
        report["fd"]["f64"] = fd_rows
        disabled = [group for group, row in fd_rows.items() if row.get("disabled")]
        unresolved = [
            group
            for group, row in fd_rows.items()
            if not row.get("disabled") and not row["resolved"]
        ]
        check(
            checks,
            "fd_f64_all_declared_groups",
            not unresolved and not disabled,
            {"unresolved": unresolved, "disabled": disabled},
        )
        recursive_groups = ("W", "P", "R", "xi", "query_read")
        check(
            checks,
            "recursive_task_credit",
            all(fd_rows[group]["resolved"] for group in recursive_groups),
            {group: fd_rows[group]["resolved"] for group in recursive_groups},
        )
        del fd_candidate
        gc.collect()

        fd32_candidate = build_candidate(parent, dtype=torch.float32)
        fd32_rows = {}
        for group in PARAMETER_GROUPS:
            fd32_rows[group] = fd_group(
                fd32_candidate,
                fixtures,
                group,
                eps_values=FD_EPS_F32,
                atol=FD_ATOL_F32,
                rtol=FD_RTOL_F32,
            )
            budget.ensure()
        report["fd"]["f32"] = fd32_rows
        check(
            checks,
            "fd_f32_disclosure",
            len(fd32_rows) == len(PARAMETER_GROUPS)
            and all("resolved" in row for row in fd32_rows.values()),
            {
                "resolved": [group for group, row in fd32_rows.items() if row["resolved"]],
                "unresolved": [group for group, row in fd32_rows.items() if not row["resolved"]],
            },
            "float32 resolution is disclosed, not used as the main credit gate.",
            required=False,
        )
        del fd32_candidate
        gc.collect()

        profiles = (
            (ARM_FULL, torch.float64, "A_full-float64"),
            (ARM_FULL, torch.float32, "A_full-float32"),
            (ARM_LOCAL, torch.float64, "B_local-float64"),
            (ARM_FLAT, torch.float64, "C_flat-float64"),
        )
        for _index, (arm, dtype, profile) in enumerate(profiles):
            row = run_profile(
                parent,
                identity,
                fixtures,
                arm=arm,
                dtype=dtype,
                profile=profile,
                output_dir=output_dir,
                budget=budget,
                subject_id=f"14.4-{profile}",
            )
            report["profiles"].append(row)
            budget.ensure()
        check(
            checks,
            "restore_chain",
            all(
                row["initial_restore"]["equal"]
                and row["step1_restore"]["equal"]
                and row["step2_restore"]["parameters_equal"]
                and row["step2_restore"]["optimizer_equal"]
                and row["step2_restore"]["post_probabilities_equal"]
                for row in report["profiles"]
            ),
            [
                {
                    "profile": row["profile"],
                    "initial": row["initial_restore"]["equal"],
                    "step1": row["step1_restore"]["equal"],
                    "params": row["step2_restore"]["parameters_equal"],
                    "optimizer": row["step2_restore"]["optimizer_equal"],
                    "post": row["step2_restore"]["post_probabilities_equal"],
                }
                for row in report["profiles"]
            ],
        )
        check(
            checks,
            "adam_budget",
            budget.adam_calls == 12 and budget.adam_calls <= MAX_ADAM_CALLS,
            {"calls": budget.adam_calls, "cap": MAX_ADAM_CALLS},
        )
        check(
            checks,
            "profile_updates_sane",
            all(
                torch.isfinite(torch.tensor(row["losses"]["step1"]))
                and torch.isfinite(torch.tensor(row["losses"]["step2"]))
                for row in report["profiles"]
            )
            and all(row["losses"]["worker_loss"] is not None for row in report["profiles"]),
            [
                {
                    "profile": row["profile"],
                    "step1": row["losses"]["step1"],
                    "step2": row["losses"]["step2"],
                    "worker": row["losses"]["worker_loss"],
                }
                for row in report["profiles"]
            ],
        )

        parent_sha_after = sha256_text(parent_path)
        check(
            checks,
            "parent_bundle_unchanged",
            parent_sha_after == parent_sha_before,
            {"before": parent_sha_before, "after": parent_sha_after},
        )
        check(
            checks,
            "products_within_limit",
            budget.max_product_bytes <= MAX_PRODUCT_BYTES,
            {
                "max_live_bytes": budget.max_product_bytes,
                "cumulative_bytes": budget.cumulative_product_bytes,
                "cap": MAX_PRODUCT_BYTES,
            },
        )
        report["cost"] = {
            "main_cpu_seconds": time.process_time() - STARTED_CPU,
            "worker_cpu_seconds": budget.worker_cpu,
            "total_cpu_seconds": budget.total_cpu(),
            "wall_seconds": time.perf_counter() - STARTED_WALL,
            "adam_calls": budget.adam_calls,
            "max_live_product_bytes": budget.max_product_bytes,
            "cumulative_product_bytes": budget.cumulative_product_bytes,
            "peak_ram_mb": budget.peak_ram_mb,
        }
        check(
            checks,
            "cost_within_budget",
            budget.total_cpu() <= BUDGET_CPU_SECONDS
            and budget.wall <= BUDGET_WALL_SECONDS
            and time.perf_counter() - STARTED_WALL <= BUDGET_WALL_SECONDS,
            report["cost"],
        )
        report["type_check"] = run_type_check()
        check(checks, "type_check", report["type_check"]["passed"], report["type_check"])
        report["lint"] = run_lint()
        check(checks, "lint", report["lint"]["passed"], report["lint"])
        status_override = None
    except BudgetStop as exc:
        error = f"BudgetStop: {exc}"
        status_override = "budget_stop"
    except InstrumentError as exc:
        error = f"InstrumentError: {exc}"
        status_override = "instrument_invalid"
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        status_override = "engineering_failure"
    finally:
        for path in output_dir.glob("*"):
            with contextlib.suppress(OSError):
                path.unlink()
        report["cost"] = report.get("cost") or {
            "main_cpu_seconds": time.process_time() - STARTED_CPU,
            "worker_cpu_seconds": budget.worker_cpu,
            "total_cpu_seconds": budget.total_cpu(),
            "wall_seconds": time.perf_counter() - STARTED_WALL,
            "adam_calls": budget.adam_calls,
            "max_live_product_bytes": budget.max_product_bytes,
            "cumulative_product_bytes": budget.cumulative_product_bytes,
            "peak_ram_mb": budget.peak_ram_mb,
        }
        if (
            any(not row["passed"] for row in checks if row.get("required", True))
            or error is not None
        ):
            passed = False
        else:
            passed = True
        report["error"] = error
        report["required_checks"] = sum(1 for row in checks if row.get("required", True))
        report["failed_required"] = [
            row["id"] for row in checks if row.get("required", True) and not row["passed"]
        ]
        report["passed"] = passed
        if status_override:
            report["status"] = status_override
        elif passed:
            report["status"] = "passed_record_relation_graph_engineering_only"
        else:
            report["status"] = "engineering_failure"
        info = write_json(report_path, report)
        report["report_file"] = info
        info["bytes"] = info["bytes"]
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "passed": report["passed"],
                    "failed_required": report["failed_required"],
                    "cost": report["cost"],
                    "report": info["path"],
                },
                ensure_ascii=False,
            )
        )
    return 0 if report["passed"] else 1


def run_type_check() -> dict:
    command = [
        sys.executable,
        "-m",
        "mypy",
        "--follow-imports=skip",
        "--ignore-missing-imports",
        "taiji/record_relation.py",
        "taiji/record_relation_core.py",
        "scripts/training/verify_taiji_record_relation_14.py",
        "tests/taiji_native/test_record_relation_14_4.py",
    ]
    try:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=240)
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
    paths = [
        "taiji/record_relation.py",
        "taiji/record_relation_core.py",
        "scripts/training/verify_taiji_record_relation_14.py",
        "tests/taiji_native/test_record_relation_14_4.py",
    ]
    ruff = subprocess.run(
        [sys.executable, "-m", "ruff", "check", *paths],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    black = subprocess.run(
        [sys.executable, "-m", "black", "--no-cache", "--check", *paths],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return {
        "ruff": {"returncode": ruff.returncode, "output": ruff.stdout.strip()},
        "black": {"returncode": black.returncode, "output": black.stdout.strip()},
        "passed": ruff.returncode == 0 and black.returncode == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="graph")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--payload-dir", type=Path)
    args = parser.parse_args()
    if args.stage == "graph":
        return graph_main(args)
    raise SystemExit(f"stage {args.stage!r} is not implemented in 14.4")


if __name__ == "__main__":
    raise SystemExit(main())
