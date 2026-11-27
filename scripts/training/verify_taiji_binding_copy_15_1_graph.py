"""15.1 rev2 stage 0b: R1 span-alignment graph, intervention, credit and restore.

This stage builds the formal R1 reading graph on top of the fixed 14.6 parent,
declares the 30,133 new parameters and the inherited/disabled/reused accounting,
runs finite-difference derivative checks on every trainable group, exercises the
record / position / gamma intervention interfaces separately, verifies the
identity-rescue property (removing an intervention returns the distribution
bit-for-bit to the unforced one), saves and restores the new bundle schema in an
independent process, and runs three real Adam steps for each of four profiles.

It does not train a task, does not price a model, never reads the sealed confirm
split and never publishes a capability claim.  The parent 14.6 bundle is read
only and must be byte-for-byte unchanged on exit.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import multiprocessing
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from taiji.binding_copy import (  # noqa: E402
    BINDING_COPY_SCHEMA,
    R1_PARAMETER_GROUPS,
    BindingCopyConfig,
    answer_step,
    build_binding_candidate,
    candidate_parameter_map,
    combined_loss,
    expected_new_parameter_count,
    forced_read,
    initial_binding_state,
    observe_goal,
    score_example,
)
from taiji.binding_copy import (
    group_delta as r1_group_delta,
)
from taiji.binding_copy import (
    make_optimizer as r1_make_optimizer,
)
from taiji.binding_copy import (
    optimizer_moment_rows as r1_optimizer_moment_rows,
)
from taiji.binding_copy import (
    prepare as r1_prepare,
)
from taiji.binding_copy import (
    run_worker as r1_run_worker,
)
from taiji.binding_copy import (
    save_bundle as r1_save_bundle,
)
from taiji.living_core import EOS, RawSource  # noqa: E402
from taiji.record_relation import RecordRelationConfig  # noqa: E402
from taiji.record_relation_core import (  # noqa: E402
    ARM_FULL,
    CLIP_NORM,
    CONTENT_REVISION,
    RecordRelationCandidate,
    build_subject_from_candidate_bundle,
)
from taiji.record_relation_core import (
    PARAMETER_GROUPS as BASE_GROUPS,
)
from taiji.record_relation_core import (
    answer_step as base_answer_step,
)
from taiji.record_relation_core import (
    candidate_parameter_map as base_candidate_parameter_map,
)
from taiji.record_relation_core import (
    combined_loss as base_combined_loss,
)
from taiji.record_relation_core import (
    group_delta as base_group_delta,
)
from taiji.record_relation_core import (
    make_optimizer as base_make_optimizer,
)
from taiji.record_relation_core import (
    observe_goal as base_observe_goal,
)
from taiji.record_relation_core import (
    optimizer_moment_rows as base_optimizer_moment_rows,
)
from taiji.record_relation_core import (
    prepare as base_prepare,
)
from taiji.record_relation_core import (
    run_worker as base_run_worker,
)
from taiji.record_relation_core import (
    save_bundle as base_save_bundle,
)
from taiji.record_relation_core import (
    score_example as base_score_example,
)

PARENT_BUNDLE = "output/taiji_core_record_relation_14_6/27031_A_full_step300.pt"
PARENT_MANIFEST = "plans/manifests/taiji_core_record_relation_14_6_20261102.json"
PARENT_CARD = "reports/taiji_core_record_relation_14_6_20261102.json"
REPORT = "reports/taiji_binding_copy_15_1_graph_20261102.json"
MANIFEST = "plans/manifests/taiji_binding_copy_15_1_graph_20261102.json"
OUTPUT_DIR = "output/taiji_core_binding_copy_15_1"

BUDGET_CPU_SECONDS = 600.0
BUDGET_WALL_SECONDS = 1200.0
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

EXPECTED_NEW_PARAMETERS = 30133
EXPECTED_DISABLED_ELEMENTS = 48481
EXPECTED_REUSED_ELEMENTS = 16672

SOURCE_FILES = (
    "taiji/binding_copy.py",
    "taiji/record_relation.py",
    "taiji/record_relation_core.py",
    "scripts/training/verify_taiji_binding_copy_15_1_graph.py",
)

STARTED_WALL = time.perf_counter()
STARTED_CPU = time.process_time()


def sha256_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach() - b.detach()).abs().max())


def tree_equal(a: Any, b: Any) -> Any:
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return bool(torch.equal(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(tree_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(tree_equal(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def check(
    rows: list,
    check_id: str,
    passed: bool,
    observed: Any,
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


def write_json(path: Path, payload: Any) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_text(path)}


def build_base_from_parent(
    payload: dict, *, arm: str = ARM_FULL, dtype: torch.dtype = torch.float64
) -> RecordRelationCandidate:
    config = RecordRelationConfig(**payload["relation_config"])
    subject = build_subject_from_candidate_bundle(payload)
    candidate = RecordRelationCandidate(subject, config, arm, list(payload["growth_gates"]))
    named = dict(candidate.named_parameters())
    for name, value in payload["parameters"].items():
        if name.startswith("relation."):
            named[name].data.copy_(value)
    candidate.to(dtype)
    candidate.configure_trainable()
    return candidate


def build_fixtures() -> tuple[dict, dict]:
    records = (
        RawSource(source_id="alpha", raw=b"typeA=abz."),
        RawSource(source_id="beta", raw=b"typeB=cdy."),
    )
    first = {
        "name": "records",
        "goal": "typeA",
        "records": records,
        "target_action": "abz.",
        "target_bytes": tuple(b"abz.") + (EOS,),
    }
    second = {
        "name": "null_only",
        "goal": "typeZ",
        "records": (),
        "target_action": "xyz.",
        "target_bytes": tuple(b"xyz.") + (EOS,),
    }
    return first, second


def fixture_sha(fixture: dict) -> str:
    digest = hashlib.sha256()
    digest.update(fixture["goal"].encode())
    for record in fixture["records"]:
        digest.update(record.source_id.encode())
        digest.update(record.raw)
    digest.update(fixture["target_action"].encode())
    return digest.hexdigest()


def fixture_loss_rows(fixtures: tuple[dict, ...]) -> tuple[dict, ...]:
    return tuple(
        {
            "goal": fixture["goal"],
            "records": fixture["records"],
            "target_bytes": fixture["target_bytes"],
        }
        for fixture in fixtures
    )


def no_grad_loss(candidate, fixtures, *, r1: bool) -> float:
    loss_fn = combined_loss if r1 else base_combined_loss
    with torch.no_grad():
        return float(loss_fn(candidate, fixture_loss_rows(fixtures)).detach())


def group_gradients(candidate, fixtures, group: str, *, r1: bool):
    candidate.zero_grad(set_to_none=True)
    loss_fn = combined_loss if r1 else base_combined_loss
    loss = loss_fn(candidate, fixture_loss_rows(fixtures))
    parameters = candidate.trainable_group_parameters(group)
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=False)
    filled = [
        torch.zeros_like(p) if g is None else g.detach()
        for p, g in zip(parameters, gradients, strict=True)
    ]
    squared = sum((g.square().sum() for g in filled), start=torch.tensor(0.0))
    norm = float(torch.sqrt(squared))
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
    r1: bool,
    eps_values: tuple[float, ...],
    atol: float,
    rtol: float,
) -> dict:
    parameters = candidate.trainable_group_parameters(group)
    row: dict[str, Any] = {
        "group": group,
        "parameter_count": int(sum(p.numel() for p in parameters)),
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
        row["note"] = "disabled_by_config"
        return row
    _loss, _parameters, gradients, norm = group_gradients(candidate, fixtures, group, r1=r1)
    base_loss = no_grad_loss(candidate, fixtures, r1=r1)
    row.update({"grad_norm": norm, "base_loss": base_loss})
    if norm <= SIGNAL_FACTOR * atol:
        row["note"] = "below_signal_floor"
        return row
    direction = [g / norm for g in gradients]
    fd_values = []
    for eps in eps_values:
        perturb(parameters, direction, eps)
        plus = no_grad_loss(candidate, fixtures, r1=r1)
        perturb(parameters, direction, -2.0 * eps)
        minus = no_grad_loss(candidate, fixtures, r1=r1)
        perturb(parameters, direction, eps)
        forward = (plus - minus) / (2.0 * eps)
        error = abs(forward - norm) / max(norm, 1e-30)
        row["fd"][f"{eps:g}"] = forward
        row["fd_error"][f"{eps:g}"] = error
        fd_values.append((forward, error))
    consistent = fd_values[0][1] <= rtol and sum(1 for _, e in fd_values if e <= rtol) >= 2
    flipped = [-g / norm for g in gradients]
    eps_flip = eps_values[0]
    perturb(parameters, flipped, eps_flip)
    flip_plus = no_grad_loss(candidate, fixtures, r1=r1)
    perturb(parameters, flipped, -2.0 * eps_flip)
    flip_minus = no_grad_loss(candidate, fixtures, r1=r1)
    perturb(parameters, flipped, eps_flip)
    flip_fd = (flip_plus - flip_minus) / (2.0 * eps_flip)
    flip_rejected = flip_fd <= SIGNAL_FACTOR * atol
    snapshot = [p.detach().clone() for p in parameters]
    with torch.no_grad():
        for p in parameters:
            p.zero_()
    zero_params_loss = no_grad_loss(candidate, fixtures, r1=r1)
    with torch.no_grad():
        for p, saved in zip(parameters, snapshot, strict=True):
            p.copy_(saved)
    perturb(parameters, direction, -ASCENT_STEP)
    ascent_loss = no_grad_loss(candidate, fixtures, r1=r1)
    perturb(parameters, direction, ASCENT_STEP)
    row.update(
        {
            "eps_consistent": consistent,
            "signal": norm > SIGNAL_FACTOR * atol,
            "sign_flip_rejected": flip_rejected,
            "flip_fd": flip_fd,
            "zero_params_loss_delta": zero_params_loss - base_loss,
            "ascent_loss_delta": ascent_loss - base_loss,
            "resolved": bool(consistent and flip_rejected and norm > SIGNAL_FACTOR * atol),
        }
    )
    if not row["resolved"]:
        row["note"] = "eps_inconsistent" if not consistent else "flip_not_opposite"
    return row


def spawn_worker(spec: dict, *, r1: bool) -> dict:
    target = r1_run_worker if r1 else base_run_worker
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=target, args=(spec,))
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
    gamma_override: float | None = None,
    gate_index: int = 0,
) -> dict:
    return {
        "bundle": str(bundle),
        "result": str(result),
        "fixtures": tuple(
            {
                "goal": f["goal"],
                "records": f["records"],
                "target_bytes": f["target_bytes"],
            }
            for f in fixtures
        ),
        "dtype": str(dtype).replace("torch.", ""),
        "mode": mode,
        "previous_byte": previous_byte,
        "fixture_index": fixture_index,
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

    def ensure(self) -> None:
        self.main_cpu = time.process_time() - STARTED_CPU
        if self.total_cpu() > BUDGET_CPU_SECONDS:
            raise BudgetStop(f"CPU budget exceeded: {self.total_cpu():.2f}")
        wall_now = time.perf_counter() - STARTED_WALL
        if wall_now > BUDGET_WALL_SECONDS:
            raise BudgetStop(f"wall budget exceeded: {wall_now:.2f}")
        if self.adam_calls > MAX_ADAM_CALLS:
            raise BudgetStop(f"Adam budget exceeded: {self.adam_calls}")
        if self.max_product_bytes > MAX_PRODUCT_BYTES:
            raise BudgetStop(f"product budget exceeded: {self.max_product_bytes}")


class BudgetStop(Exception):
    pass


def parameter_rows(candidate, *, r1: bool) -> dict:
    named = dict(candidate.named_parameters())
    groups = R1_PARAMETER_GROUPS if r1 else BASE_GROUPS
    rows = {}
    for group, names in groups.items():
        rows[group] = {
            "names": list(names),
            "parameter_count": int(sum(named[n].numel() for n in names if n in named)),
            "trainable_count": int(
                sum(named[n].numel() for n in names if n in named and named[n].requires_grad)
            ),
        }
    return rows


def run_profile(
    payload: dict,
    parent_identity: dict,
    fixtures: tuple[dict, ...],
    *,
    kind: str,
    dtype: torch.dtype,
    profile: str,
    output_dir: Path,
    budget: Budget,
    subject_id: str,
    small_config: BindingCopyConfig | None = None,
) -> dict:
    r1 = kind == "R1"
    candidate: Any
    optimizer: Any
    prepared: Any
    obs_state: Any
    gate_index: int
    probe_initial: Any
    save_fn: Any
    delta_fn: Any
    moment_fn: Any
    loss_fn: Any
    score_fn: Any
    if r1:
        base = build_base_from_parent(payload, arm=ARM_FULL, dtype=dtype)
        config = small_config or BindingCopyConfig()
        candidate = build_binding_candidate(base, config=config, profile=profile, dtype=dtype)
        optimizer = r1_make_optimizer(candidate)
        state = initial_binding_state(candidate, subject_id)
        prepared = r1_prepare(
            candidate, state, fixtures[0]["goal"].encode(), fixtures[0]["records"]
        )
        obs_state, gate_index = observe_goal(candidate, prepared)
        probe_initial = answer_step(candidate, prepared, obs_state, EOS, gate_index=gate_index)
        save_fn = r1_save_bundle
        delta_fn = r1_group_delta
        moment_fn = r1_optimizer_moment_rows
        loss_fn = combined_loss
        score_fn = score_example
    else:
        candidate = build_base_from_parent(payload, arm=ARM_FULL, dtype=dtype)
        optimizer = base_make_optimizer(candidate)
        zero_state = candidate.subject.initial_state(subject_id, revision=CONTENT_REVISION)
        prepared = base_prepare(
            candidate, zero_state, fixtures[0]["goal"].encode(), fixtures[0]["records"]
        )
        obs_state, gate_index = base_observe_goal(candidate, prepared)
        probe_initial = base_answer_step(
            candidate, prepared, obs_state, obs_state.coverage, EOS, gate_index=gate_index
        )
        save_fn = base_save_bundle
        delta_fn = base_group_delta
        moment_fn = base_optimizer_moment_rows
        loss_fn = base_combined_loss
        score_fn = base_score_example

    bundle0 = output_dir / f"{profile}_initial.pt"
    result0 = output_dir / f"{profile}_initial_probe.pt"
    info0 = save_fn(
        bundle0,
        profile=profile,
        candidate=candidate,
        state=obs_state,
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
        ),
        r1=r1,
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
    }

    before = candidate_parameter_map(candidate) if r1 else base_candidate_parameter_map(candidate)
    optimizer.zero_grad()
    loss1 = loss_fn(candidate, fixture_loss_rows(fixtures))
    loss1.backward()
    clip1 = float(torch.nn.utils.clip_grad_norm_(candidate.trainable_parameters(), CLIP_NORM))
    optimizer.step()
    budget.adam_calls += 1
    budget.ensure()

    deltas = {
        group: delta_fn(before, candidate, group)
        for group in (R1_PARAMETER_GROUPS if r1 else BASE_GROUPS)
    }
    moments = moment_fn(candidate, optimizer)

    score1 = score_fn(
        candidate,
        fixtures[0]["goal"].encode(),
        fixtures[0]["records"],
        tuple(fixtures[0]["target_bytes"][:-1]),
        subject_id=subject_id,
    )
    if r1:
        probe1: Any = answer_step(
            candidate,
            score1.prepared,
            score1.prefix,
            fixtures[0]["target_bytes"][-2],
            gate_index=0,
        )
        save_state = score1.prefix
    else:
        probe1 = base_answer_step(
            candidate,
            score1.prepared,
            score1.prefix_state,
            score1.prefix_coverage,
            fixtures[0]["target_bytes"][-2],
            gate_index=0,
        )
        save_state = score1.prefix_state

    bundle1 = output_dir / f"{profile}_step1.pt"
    result1 = output_dir / f"{profile}_step1_probe.pt"
    info1 = save_fn(
        bundle1,
        profile=profile,
        candidate=candidate,
        state=save_state,
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
            previous_byte=fixtures[0]["target_bytes"][-2],
            fixture_index=0,
        ),
        r1=r1,
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
    loss2 = loss_fn(candidate, fixture_loss_rows(fixtures))
    loss2.backward()
    clip2 = float(torch.nn.utils.clip_grad_norm_(candidate.trainable_parameters(), CLIP_NORM))
    optimizer.step()
    budget.adam_calls += 1
    budget.ensure()

    post = score_fn(
        candidate,
        fixtures[0]["goal"].encode(),
        fixtures[0]["records"],
        tuple(fixtures[0]["target_bytes"][:-1]),
        subject_id=subject_id,
    )
    if r1:
        answer_step(
            candidate,
            post.prepared,
            post.prefix,
            fixtures[0]["target_bytes"][-2],
            gate_index=0,
        )
    else:
        base_answer_step(
            candidate,
            post.prepared,
            post.prefix_state,
            post.prefix_coverage,
            fixtures[0]["target_bytes"][-2],
            gate_index=0,
        )
    after_parameters = (
        candidate_parameter_map(candidate) if r1 else base_candidate_parameter_map(candidate)
    )
    if worker1.get("ok"):
        params_equal = tree_equal(after_parameters, worker1["parameters"])
        optimizer_equal = tree_equal(optimizer.state_dict(), worker1["optimizer"])
    else:
        params_equal = optimizer_equal = False

    for info in (info0, info1):
        budget.drop_product(info)
    for path in (bundle0, bundle1, result0, result1):
        if path.exists():
            path.unlink()

    return {
        "profile": profile,
        "kind": kind,
        "dtype": str(dtype).replace("torch.", ""),
        "parameter_count": candidate.parameter_count(),
        "trainable_count": int(sum(p.numel() for p in candidate.trainable_parameters())),
        "initial_restore": initial_restore,
        "step1_restore": restore_probe,
        "step2_restore": {
            "ok": bool(worker1.get("ok")),
            "parameters_equal": params_equal,
            "optimizer_equal": optimizer_equal,
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


def run_intervention_checks(payload: dict, fixtures: tuple[dict, ...]) -> dict:
    base = build_base_from_parent(payload, arm=ARM_FULL, dtype=torch.float64)
    candidate = build_binding_candidate(
        base, config=BindingCopyConfig(), profile="R1-f64", dtype=torch.float64
    )
    first = fixtures[0]
    state = initial_binding_state(candidate, "intervention")
    prepared = r1_prepare(candidate, state, first["goal"].encode(), first["records"])
    obs_state, gate_index = observe_goal(candidate, prepared)

    unforced = answer_step(candidate, prepared, obs_state, EOS, gate_index=gate_index)
    unforced_probs = unforced.probabilities.detach().clone()

    record_forced = forced_read(
        candidate,
        prepared,
        obs_state,
        EOS,
        record_index=0,
        offset=0,
        gate_index=gate_index,
    )
    record_diff = max_abs_diff(record_forced.probabilities, unforced_probs)

    position_forced = forced_read(
        candidate,
        prepared,
        obs_state,
        EOS,
        record_index=0,
        offset=2,
        gate_index=gate_index,
    )
    position_diff = max_abs_diff(position_forced.probabilities, unforced_probs)

    gamma_zero = answer_step(
        candidate,
        prepared,
        obs_state,
        EOS,
        gate_index=gate_index,
        gamma_override=0.0,
    )
    gamma_one = answer_step(
        candidate,
        prepared,
        obs_state,
        EOS,
        gate_index=gate_index,
        gamma_override=1.0,
    )
    gamma_diff = max_abs_diff(gamma_zero.probabilities, gamma_one.probabilities)

    rescue = answer_step(candidate, prepared, obs_state, EOS, gate_index=gate_index)
    rescue_equal = bool(torch.equal(rescue.probabilities, unforced_probs))

    del base, candidate
    gc.collect()
    return {
        "unforced_sum": float(unforced_probs.sum()),
        "record_intervention_diff": record_diff,
        "position_intervention_diff": position_diff,
        "gamma_intervention_diff": gamma_diff,
        "record_vs_position_differ": record_diff > 0
        and position_diff > 0
        and not torch.allclose(record_forced.probabilities, position_forced.probabilities),
        "identity_rescue_equal": rescue_equal,
        "identity_rescue_max_abs_diff": max_abs_diff(rescue.probabilities, unforced_probs),
        "passed": bool(record_diff > 0 and position_diff > 0 and gamma_diff > 0 and rescue_equal),
    }


def graph_main() -> int:
    report_path = ROOT / REPORT
    manifest_path = ROOT / MANIFEST
    output_dir = ROOT / OUTPUT_DIR
    if report_path.exists():
        raise FileExistsError("15.1.0b graph card is write-once")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    budget = Budget()
    checks: list[dict] = []
    report: dict[str, Any] = {
        "schema": "taiji-binding-copy-graph-v1",
        "step": "15.1.0b",
        "date": "2026-11-02",
        "status": "engineering_failure",
        "checks": checks,
        "identity": {},
        "module": {},
        "fixtures": [],
        "fd": {},
        "intervention": {},
        "profiles": [],
        "cost": {},
        "limits": (
            "Engineering only. The parent is the fixed 14.6 A_full step300 bundle; "
            "both fixtures are public synthetic strings. No task training, no "
            "pricing, no capability claim, no published weights, no sealed "
            "confirm read."
        ),
    }
    error: str | None = None
    parent_path = ROOT / PARENT_BUNDLE
    parent_sha_before = sha256_text(parent_path)
    source_hashes = {relative: sha256_text(ROOT / relative) for relative in SOURCE_FILES}
    try:
        payload = torch.load(parent_path, map_location="cpu", weights_only=True)
        identity = {
            "parent_schema": payload["schema"],
            "parent_step": payload["step"],
            "parent_arm": payload["arm"],
            "parent_bundle_sha256": parent_sha_before,
            "parent_bundle_bytes": parent_path.stat().st_size,
            "parent_update_index": payload["update_index"],
            "source_hashes": source_hashes,
        }
        report["identity"] = identity
        check(checks, "parent_identity", True, identity)

        fixtures = build_fixtures()
        report["fixtures"] = [
            {
                "name": f["name"],
                "goal": f["goal"],
                "records": [
                    {"source_id": r.source_id, "raw": r.raw.decode("utf-8")} for r in f["records"]
                ],
                "target_action": f["target_action"],
                "target_bytes": list(f["target_bytes"]),
                "sha256": fixture_sha(f),
            }
            for f in fixtures
        ]

        base = build_base_from_parent(payload, arm=ARM_FULL, dtype=torch.float64)
        base_trainable = int(sum(p.numel() for p in base.trainable_parameters()))
        candidate = build_binding_candidate(
            base, config=BindingCopyConfig(), profile="R1-f64", dtype=torch.float64
        )

        expected_groups = {
            group: sum(p.numel() for p in candidate.group_parameters(group))
            for group in R1_PARAMETER_GROUPS
        }
        check(
            checks,
            "new_parameter_count",
            candidate.new_parameter_count() == EXPECTED_NEW_PARAMETERS
            and expected_new_parameter_count() == EXPECTED_NEW_PARAMETERS,
            {"module": candidate.new_parameter_count(), "formula": expected_new_parameter_count()},
        )
        check(
            checks,
            "r1_parameter_groups",
            sum(expected_groups.values()) == EXPECTED_NEW_PARAMETERS,
            expected_groups,
        )

        disabled = candidate.disabled_rows()
        reused = candidate.reused_rows()
        check(
            checks,
            "disabled_accounting",
            disabled["elements"] == EXPECTED_DISABLED_ELEMENTS,
            disabled,
        )
        check(
            checks,
            "reused_accounting",
            reused["elements"] == EXPECTED_REUSED_ELEMENTS and reused["trainable"],
            reused,
        )
        trainable_total = int(sum(p.numel() for p in candidate.trainable_parameters()))
        inherited_total = base.parameter_count()
        check(
            checks,
            "parameter_accounting",
            candidate.parameter_count() == inherited_total + EXPECTED_NEW_PARAMETERS
            and trainable_total == base_trainable - disabled["elements"] + EXPECTED_NEW_PARAMETERS,
            {
                "total_parameters": candidate.parameter_count(),
                "inherited": inherited_total,
                "new": EXPECTED_NEW_PARAMETERS,
                "trainable": trainable_total,
                "disabled": disabled["elements"],
            },
        )
        report["module"] = {
            "name": "BindingCopyCandidate",
            "config": {"key_width": candidate.config.key_width, "width": candidate.config.width},
            "parameter_count": candidate.parameter_count(),
            "trainable_count": trainable_total,
            "new_parameter_count": candidate.new_parameter_count(),
            "inherited_parameter_count": inherited_total,
            "disabled": disabled,
            "reused": reused,
            "r1_groups": parameter_rows(candidate, r1=True),
            "content_schema": BINDING_COPY_SCHEMA,
        }
        del base, candidate
        gc.collect()

        intervention = run_intervention_checks(payload, fixtures)
        report["intervention"] = intervention
        check(checks, "intervention_interfaces", intervention["passed"], intervention)
        check(
            checks,
            "identity_rescue",
            intervention["identity_rescue_equal"],
            {"max_abs_diff": intervention["identity_rescue_max_abs_diff"]},
        )

        fd_candidate_base = build_base_from_parent(payload, arm=ARM_FULL, dtype=torch.float64)
        fd_candidate = build_binding_candidate(
            fd_candidate_base, config=BindingCopyConfig(), profile="R1-f64", dtype=torch.float64
        )
        fd_groups = list(R1_PARAMETER_GROUPS.keys()) + [
            g for g in BASE_GROUPS if g not in R1_PARAMETER_GROUPS
        ]
        fd_rows: dict[str, Any] = {}
        for group in fd_groups:
            fd_rows[group] = fd_group(
                fd_candidate,
                fixtures,
                group,
                r1=True,
                eps_values=FD_EPS_F64,
                atol=FD_ATOL_F64,
                rtol=FD_RTOL_F64,
            )
            budget.ensure()
        report["fd"]["f64"] = fd_rows
        disabled_groups = [g for g, row in fd_rows.items() if row.get("disabled")]
        unresolved = [
            g for g, row in fd_rows.items() if not row.get("disabled") and not row["resolved"]
        ]
        check(
            checks,
            "fd_f64_all_trainable_groups",
            not unresolved,
            {"unresolved": unresolved, "disabled": disabled_groups},
        )
        check(
            checks,
            "r1_fd_resolved",
            all(fd_rows[g]["resolved"] for g in R1_PARAMETER_GROUPS),
            {g: fd_rows[g]["resolved"] for g in R1_PARAMETER_GROUPS},
        )
        del fd_candidate, fd_candidate_base
        gc.collect()

        fd32_base = build_base_from_parent(payload, arm=ARM_FULL, dtype=torch.float32)
        fd32 = build_binding_candidate(
            fd32_base, config=BindingCopyConfig(), profile="R1-f32", dtype=torch.float32
        )
        fd32_rows: dict[str, Any] = {}
        for group in fd_groups:
            fd32_rows[group] = fd_group(
                fd32,
                fixtures,
                group,
                r1=True,
                eps_values=FD_EPS_F32,
                atol=FD_ATOL_F32,
                rtol=FD_RTOL_F32,
            )
            budget.ensure()
        report["fd"]["f32"] = fd32_rows
        check(
            checks,
            "fd_f32_disclosure",
            len(fd32_rows) == len(fd_groups),
            {
                "resolved": [g for g, row in fd32_rows.items() if row["resolved"]],
                "unresolved": [g for g, row in fd32_rows.items() if not row["resolved"]],
            },
            "float32 resolution is disclosed, not used as the main credit gate.",
            required=False,
        )
        del fd32, fd32_base
        gc.collect()

        parent_identity = {
            "schema": payload["schema"],
            "bundle_sha256": parent_sha_before,
            "bundle_bytes": parent_path.stat().st_size,
            "update_index": payload["update_index"],
        }
        profiles = (
            ("R0", torch.float64, "R0-f64", None),
            ("R1", torch.float64, "R1-f64", None),
            ("R1", torch.float32, "R1-f32", None),
            ("R1", torch.float64, "R1-small-f64", BindingCopyConfig(key_width=16, width=128)),
        )
        for kind, dtype, profile_name, small in profiles:
            row = run_profile(
                payload,
                parent_identity,
                fixtures,
                kind=kind,
                dtype=dtype,
                profile=profile_name,
                output_dir=output_dir,
                budget=budget,
                subject_id=f"15.1.0b-{profile_name}",
                small_config=small,
            )
            report["profiles"].append(row)
            budget.ensure()

        check(
            checks,
            "restore_chain",
            all(
                r["initial_restore"]["equal"]
                and r["step1_restore"]["equal"]
                and r["step2_restore"]["parameters_equal"]
                and r["step2_restore"]["optimizer_equal"]
                for r in report["profiles"]
            ),
            [
                {
                    "profile": r["profile"],
                    "initial": r["initial_restore"]["equal"],
                    "step1": r["step1_restore"]["equal"],
                    "params": r["step2_restore"]["parameters_equal"],
                    "optimizer": r["step2_restore"]["optimizer_equal"],
                }
                for r in report["profiles"]
            ],
        )
        check(
            checks,
            "adam_budget",
            budget.adam_calls == MAX_ADAM_CALLS and budget.adam_calls <= MAX_ADAM_CALLS,
            {"calls": budget.adam_calls, "cap": MAX_ADAM_CALLS},
        )
        check(
            checks,
            "profile_updates_sane",
            all(
                torch.isfinite(torch.tensor(r["losses"]["step1"]))
                and torch.isfinite(torch.tensor(r["losses"]["step2"]))
                for r in report["profiles"]
            ),
            [
                {
                    "profile": r["profile"],
                    "step1": r["losses"]["step1"],
                    "step2": r["losses"]["step2"],
                }
                for r in report["profiles"]
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
            {"max_live_bytes": budget.max_product_bytes, "cap": MAX_PRODUCT_BYTES},
        )

        report["cost"] = {
            "main_cpu_seconds": time.process_time() - STARTED_CPU,
            "worker_cpu_seconds": budget.worker_cpu,
            "total_cpu_seconds": budget.total_cpu(),
            "wall_seconds": time.perf_counter() - STARTED_WALL,
            "adam_calls": budget.adam_calls,
            "max_product_bytes": budget.max_product_bytes,
            "cumulative_product_bytes": budget.cumulative_product_bytes,
        }
        check(
            checks,
            "stage_budget",
            budget.total_cpu() <= BUDGET_CPU_SECONDS and budget.wall <= BUDGET_WALL_SECONDS,
            {
                "cpu": budget.total_cpu(),
                "cpu_cap": BUDGET_CPU_SECONDS,
                "wall": budget.wall,
                "wall_cap": BUDGET_WALL_SECONDS,
            },
        )

        required = [c for c in checks if c["required"]]
        all_passed = all(c["passed"] for c in required)
        report["status"] = "passed" if all_passed else "failed"
        report["summary"] = {
            "required_checks": len(required),
            "required_passed": sum(1 for c in required if c["passed"]),
            "optional_checks": len(checks) - len(required),
            "status": report["status"],
        }

        manifest_info = write_json(
            manifest_path,
            {
                "schema": "taiji-binding-copy-graph-manifest-v1",
                "step": "15.1.0b",
                "candidate": {"new_parameter_count": EXPECTED_NEW_PARAMETERS},
                "parent": {"sha256": parent_sha_before, "bytes": parent_path.stat().st_size},
                "source_hashes": source_hashes,
                "report_sha256": None,
                "profiles": [r["profile"] for r in report["profiles"]],
                "status": report["status"],
            },
        )
        report["manifest"] = {
            "path": str(manifest_path.relative_to(ROOT)),
            "sha256": manifest_info["sha256"],
            "written": True,
        }
        report_info = write_json(report_path, report)
        report["report_sha256"] = report_info["sha256"]
        return 0 if all_passed else 1

    except BudgetStop as exc:
        error = f"budget_stop: {exc}"
        report["status"] = "budget_stop"
        report["error"] = error
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        report["status"] = "engineering_failure"
        report["error"] = error
        report["traceback"] = traceback.format_exc()

    report["cost"] = {
        "main_cpu_seconds": time.process_time() - STARTED_CPU,
        "worker_cpu_seconds": budget.worker_cpu,
        "total_cpu_seconds": budget.total_cpu(),
        "wall_seconds": time.perf_counter() - STARTED_WALL,
        "adam_calls": budget.adam_calls,
    }
    write_json(report_path, report)
    print(f"0b graph stage failed: {error}")
    return 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="graph", choices=("graph",))
    args = parser.parse_args()
    if args.stage != "graph":
        raise SystemExit("only the graph stage is implemented")
    return graph_main()


if __name__ == "__main__":
    raise SystemExit(main())
