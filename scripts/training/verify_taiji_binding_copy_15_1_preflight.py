#!/usr/bin/env python
"""15.1.0d checkpoint preflight for the span-bound binding-copy package.

Two seeds x four arms. Each configuration must prove, on the real training
schema and a real Adam path:

1. initial state: save -> independent process restore -> bit-exact compare;
2. after step 1: save -> independent process restore -> bit-exact params,
   Adam moments, cursor, state;
3. next update: replay from the restored bundle in an independent process and
   compare bit-exact against the in-process update (params + moments);
4. mid-output boundary (R1 arms): save while the copy cursor is non-empty and
   continue generation in an independent process; compare the remaining bytes.

No formal training is started and no parameters are adopted. All engineering
Adam calls and consumed samples are booked separately from adoption accounting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

torch.set_num_threads(1)

from taiji.binding_copy import (  # noqa: E402
    BindingCopyConfig,
    BindingPrepared,
    BindingState,
    advance_copy,
    answer_step,
    build_binding_candidate,
    generate_bytes,
    initial_binding_state,
    load_bundle,
    make_optimizer,
    observe_goal,
    prepare,
    save_bundle,
    score_example,
)
from taiji.living_core import EOS, RawSource  # noqa: E402
from taiji.record_relation import RecordRelationConfig  # noqa: E402
from taiji.record_relation_core import (  # noqa: E402
    ARM_FULL,
    CLIP_NORM,
    CONTENT_REVISION,
    MAX_OUTPUT_BYTES,
    MAX_PREDICTIONS,
    RecordRelationCandidate,
    build_subject_from_candidate_bundle,
    candidate_parameter_map,
    decode_endpoint,
)
from taiji.record_relation_core import generate_bytes as base_generate_bytes  # noqa: E402
from taiji.record_relation_core import load_bundle as base_load_bundle  # noqa: E402
from taiji.record_relation_core import make_optimizer as base_make_optimizer  # noqa: E402
from taiji.record_relation_core import prepare as base_prepare  # noqa: E402
from taiji.record_relation_core import save_bundle as base_save_bundle  # noqa: E402
from taiji.record_relation_core import score_example as base_score_example  # noqa: E402
from taiji.record_relation_core import state_payload as base_state_payload  # noqa: E402

PARENT_BUNDLE = "output/taiji_core_record_relation_14_6/27031_A_full_step300.pt"
PARENT_SHA256 = "b75a83a0a474c041c2a61bed8c4e0a51de11197d67a9b196262e7e039a9a4cef"
DATA_PAYLOAD = "output/taiji_core_binding_copy_15_1/data_v2.json"
REPORT = "reports/taiji_binding_copy_15_1_preflight_20261102.json"
MANIFEST = "plans/manifests/taiji_binding_copy_15_1_preflight_20261102.json"
REPAIR_REPORT = "reports/taiji_binding_copy_15_1_preflight_midcursor_20261102.json"
FINAL_REPORT = "reports/taiji_binding_copy_15_1_preflight_final_20261102.json"
FINAL_MANIFEST = "plans/manifests/taiji_binding_copy_15_1_preflight_final_20261102.json"
OUTPUT_DIR = Path("output/taiji_core_binding_copy_15_1_preflight")

STAGE_CPU_SECONDS = 400.0
STAGE_WALL_SECONDS = 800.0
STAGE_ADAM_CALLS = 24
STAGE_PRODUCT_BYTES = 2 * 1024**3

BATCH = 8
UPDATES_PER_CONFIG = 3  # in-process step1, replayed step2, in-process step2
SEEDS = (28031, 28032)
ORDER_SEEDS = {28031: 38031, 28032: 38032}
CURRICULUM_POSITIONS = (1, 101, 201)
SELF_PREFIX_PROB = {1: 0.0, 101: 0.5, 201: 0.75}

ARMS = ("R0P0", "R1P0", "R0P1", "R1P1")
ARM_READER = {"R0P0": "R0", "R1P0": "R1", "R0P1": "R0", "R1P1": "R1"}
ARM_PREFIX = {"R0P0": "TF", "R1P0": "TF", "R0P1": "self", "R1P1": "self"}

WORKER_TIMEOUT = 300
MID_CURSOR_BYTES = 4

SOURCE_FILES = (
    "taiji/binding_copy.py",
    "taiji/binding_copy_data.py",
    "taiji/binding_copy_scoring.py",
    "taiji/record_relation.py",
    "taiji/record_relation_core.py",
    "scripts/training/verify_taiji_binding_copy_15_1_preflight.py",
)

STARTED_WALL = time.perf_counter()
STARTED_CPU = time.process_time()


class PreflightStop(RuntimeError):
    """The declared 0d budget was exhausted."""


class Budget:
    def __init__(self) -> None:
        self.main_cpu = 0.0
        self.worker_cpu = 0.0
        self.wall = 0.0
        self.adam_calls = 0
        self.live_bytes = 0
        self.max_live_bytes = 0
        self.cumulative_bytes = 0

    def total_cpu(self) -> float:
        self.main_cpu = time.process_time() - STARTED_CPU
        return self.main_cpu + self.worker_cpu

    def add_worker(self, result: dict) -> None:
        self.worker_cpu += float(result.get("cpu_seconds", 0.0))
        self.wall += float(result.get("wall_seconds", 0.0))
        self.adam_calls += int(result.get("adam_calls", 0))
        self.ensure()

    def add_product(self, info: dict) -> None:
        size = int(info.get("bytes", 0))
        self.live_bytes += size
        self.cumulative_bytes += size
        self.max_live_bytes = max(self.max_live_bytes, self.live_bytes)
        self.ensure()

    def drop_product(self, info: dict) -> None:
        self.live_bytes -= int(info.get("bytes", 0))

    def ensure(self) -> None:
        if self.total_cpu() > STAGE_CPU_SECONDS:
            raise PreflightStop(f"cpu budget exceeded: {self.total_cpu():.2f}")
        wall_now = time.perf_counter() - STARTED_WALL
        if wall_now > STAGE_WALL_SECONDS:
            raise PreflightStop(f"wall budget exceeded: {wall_now:.2f}")
        if self.adam_calls > STAGE_ADAM_CALLS:
            raise PreflightStop(f"adam budget exceeded: {self.adam_calls}")
        if self.max_live_bytes > STAGE_PRODUCT_BYTES:
            raise PreflightStop(f"product budget exceeded: {self.max_live_bytes}")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_base_from_parent(payload: dict) -> RecordRelationCandidate:
    config = RecordRelationConfig(**payload["relation_config"])
    subject = build_subject_from_candidate_bundle(payload)
    candidate = RecordRelationCandidate(subject, config, ARM_FULL, list(payload["growth_gates"]))
    named = dict(candidate.named_parameters())
    for name, value in payload["parameters"].items():
        if name.startswith("relation."):
            named[name].data.copy_(value)
    candidate.to(torch.float64)
    candidate.configure_trainable()
    return candidate


def example_to_fixture(example: dict) -> dict:
    records = tuple(
        RawSource(source_id=r["source_id"], raw=r["raw"].encode("utf-8"))
        for r in example["records"]
    )
    return {
        "index": int(example["index"]),
        "goal": example["goal"],
        "records": records,
        "target_bytes": tuple(example["action"].encode("utf-8")) + (EOS,),
        "target_action": example["action"],
    }


def load_train() -> tuple[dict, ...]:
    payload = json.loads(Path(DATA_PAYLOAD).read_text(encoding="utf-8"))
    rows = [e for e in payload["examples"] if e["split"] == "train"]
    fixtures = tuple(example_to_fixture(e) for e in rows)
    return fixtures


def build_config(
    payload: dict, arm: str
) -> tuple[RecordRelationCandidate, torch.optim.Adam, object, bool]:
    r1 = ARM_READER[arm] == "R1"
    if r1:
        base = build_base_from_parent(payload)
        candidate = build_binding_candidate(
            base, config=BindingCopyConfig(), profile=f"{arm}-f64", dtype=torch.float64
        )
        optimizer = make_optimizer(candidate)
        return candidate, optimizer, base, r1
    candidate = build_base_from_parent(payload)
    optimizer = base_make_optimizer(candidate)
    return candidate, optimizer, candidate, r1


def batch_loss(
    candidate,
    fixtures: tuple[dict, ...],
    *,
    r1: bool,
    self_flags: tuple[bool, ...] | None,
) -> tuple[torch.Tensor, int]:
    score_fn = score_example if r1 else base_score_example
    gen_fn = generate_bytes if r1 else base_generate_bytes
    losses = []
    self_count = 0
    for idx, fixture in enumerate(fixtures):
        use_self = bool(self_flags[idx]) if self_flags else False
        if use_self:
            self_count += 1
            with torch.no_grad():
                generated = gen_fn(
                    candidate,
                    fixture["goal"].encode(),
                    fixture["records"],
                    subject_id=f"preflight-self-{fixture['index']}",
                )
            target_len = len(fixture["target_bytes"]) - 1
            prefix = tuple(generated["raw_bytes"][:target_len])
            if len(prefix) < target_len:
                prefix = prefix + (EOS,) * (target_len - len(prefix))
        else:
            prefix = fixture["target_bytes"][:-1]
        result = score_fn(candidate, fixture["goal"].encode(), fixture["records"], prefix)
        losses.append(result.nll)
    loss = sum(losses) / len(losses)
    return loss, self_count


def self_flags_for(position: int, prefix_mode: str) -> tuple[bool, ...]:
    """Draw the self-prefix mask from the global torch RNG (saved in bundles)."""
    if prefix_mode != "self":
        return tuple(False for _ in range(BATCH))
    prob = SELF_PREFIX_PROB[position]
    return tuple(bool(torch.rand(1).item() < prob) for _ in range(BATCH))


def apply_update(
    candidate,
    optimizer,
    fixtures: tuple[dict, ...],
    *,
    r1: bool,
    position: int,
    prefix_mode: str,
) -> dict:
    flags = self_flags_for(position, prefix_mode)
    loss, self_count = batch_loss(candidate, fixtures, r1=r1, self_flags=flags)
    optimizer.zero_grad()
    loss.backward()
    clip = float(torch.nn.utils.clip_grad_norm_(candidate.trainable_parameters(), CLIP_NORM))
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        "clip_scale": clip,
        "position": position,
        "self_count": self_count,
        "adam_calls": 1,
    }


def _flatten(prefix: str, value, tensors: dict, meta: dict) -> None:
    if isinstance(value, torch.Tensor):
        tensors[f"{prefix}"] = value.detach().cpu().clone()
    elif isinstance(value, dict):
        for key, item in value.items():
            _flatten(f"{prefix}/{key}", item, tensors, meta)
    elif isinstance(value, (list, tuple)):
        meta[prefix] = json.dumps(
            [item if not isinstance(item, torch.Tensor) else "<tensor>" for item in value],
            default=str,
        )
    else:
        meta[prefix] = value


def snapshot(candidate, state, optimizer, *, r1: bool, meta: dict) -> dict:
    tensors: dict[str, torch.Tensor] = {}
    flat_meta: dict = {}
    for name, tensor in candidate_parameter_map(candidate).items():
        _flatten(f"param/{name}", tensor, tensors, flat_meta)
    opt = optimizer.state_dict()
    _flatten("optimizer", opt["state"], tensors, flat_meta)
    _flatten("param_groups", opt["param_groups"], tensors, flat_meta)
    payload = state.payload() if r1 else base_state_payload(state)
    _flatten("state", payload, tensors, flat_meta)
    flat_meta.update(meta)
    return {"tensors": tensors, "meta": flat_meta}


def compare_snapshots(left: dict, right: dict) -> dict:
    left_t = left["tensors"]
    right_t = right["tensors"]
    missing = sorted(set(left_t) ^ set(right_t))
    mismatched = []
    max_abs = 0.0
    for name in sorted(set(left_t) & set(right_t)):
        a = left_t[name]
        b = right_t[name]
        if a.shape != b.shape:
            mismatched.append(
                {"name": name, "reason": "shape", "left": list(a.shape), "right": list(b.shape)}
            )
            continue
        if not torch.equal(a, b):
            diff = float((a - b).abs().max()) if a.numel() else 0.0
            max_abs = max(max_abs, diff)
            mismatched.append({"name": name, "reason": "value", "max_abs_diff": diff})
    meta_mismatch = {
        key: {"left": left["meta"].get(key), "right": right["meta"].get(key)}
        for key in sorted(set(left["meta"]) | set(right["meta"]))
        if left["meta"].get(key) != right["meta"].get(key)
    }
    return {
        "equal": not missing and not mismatched and not meta_mismatch,
        "tensor_count": len(left_t),
        "missing_or_extra": missing[:8],
        "mismatched_tensors": mismatched[:8],
        "mismatched_meta": dict(list(meta_mismatch.items())[:8]),
        "max_abs_diff": max_abs,
    }


def final_state(result, r1: bool):
    """Return the post-prefix state from either scorer shape."""
    return result.observation if r1 else result.observation_state


def save_state_bundle(
    path: Path,
    *,
    profile: str,
    candidate,
    state,
    optimizer,
    update_index: int,
    cursor: dict,
    r1: bool,
    budget: Budget,
) -> dict:
    save_fn = save_bundle if r1 else base_save_bundle
    info = save_fn(
        path,
        profile=profile,
        candidate=candidate,
        state=state,
        optimizer=optimizer,
        update_index=update_index,
        cursor=cursor,
        parent_identity={"sha256": PARENT_SHA256},
    )
    budget.add_product(info)
    return info


def load_state_bundle(path: Path, *, r1: bool) -> tuple:
    load_fn = load_bundle if r1 else base_load_bundle
    return load_fn(path, torch.float64)


def spawn_worker(spec: dict) -> dict:
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=preflight_worker, args=(spec,))
    process.start()
    process.join(timeout=WORKER_TIMEOUT)
    if process.is_alive():
        process.terminate()
        process.join()
        return {"ok": False, "message": "worker timed out", "cpu_seconds": 0.0, "wall_seconds": 0.0}
    if process.exitcode != 0:
        return {
            "ok": False,
            "message": f"worker exit code {process.exitcode}",
            "cpu_seconds": 0.0,
            "wall_seconds": 0.0,
        }
    result_path = Path(spec["result"])
    if not result_path.exists():
        return {
            "ok": False,
            "message": "worker wrote no result",
            "cpu_seconds": 0.0,
            "wall_seconds": 0.0,
        }
    return torch.load(result_path, map_location="cpu", weights_only=False)


def preflight_worker(spec: dict) -> None:
    """Independent-process restore / replay. Runs only in the spawned child."""
    start_cpu = time.process_time()
    start_wall = time.perf_counter()
    out: dict = {"ok": False}
    try:
        r1 = spec["reader"] == "R1"
        loaded = load_state_bundle(Path(spec["bundle"]), r1=r1)
        if r1:
            candidate, state, optimizer, profile, update_index, cursor, payload = loaded
        else:
            (
                candidate,
                state,
                optimizer,
                _arm,
                profile,
                update_index,
                cursor,
                payload,
            ) = loaded
        mode = spec["mode"]
        info: dict = {
            "ok": True,
            "restored_update_index": int(update_index),
            "restored_cursor": dict(cursor),
        }

        if mode == "digest":
            info["snapshot"] = snapshot(
                candidate, state, optimizer, r1=r1, meta={"update_index": int(update_index)}
            )
        elif mode == "step":
            fixtures = load_train()
            order = spec["order_seed"]
            generator = torch.Generator().manual_seed(order)
            indices = torch.randperm(len(fixtures), generator=generator).tolist()
            batch = tuple(fixtures[i] for i in indices[:BATCH])
            row = apply_update(
                candidate,
                optimizer,
                batch,
                r1=r1,
                position=int(spec["position"]),
                prefix_mode=spec["prefix_mode"],
            )
            result = score_example if r1 else base_score_example
            observation = final_state(
                result(
                    candidate,
                    batch[0]["goal"].encode(),
                    batch[0]["records"],
                    batch[0]["target_bytes"][:-1],
                ),
                r1,
            )
            bundle_path = Path(spec["bundle_out"])
            info_bundle = (save_bundle if r1 else base_save_bundle)(
                bundle_path,
                profile=profile,
                candidate=candidate,
                state=observation,
                optimizer=optimizer,
                update_index=int(update_index) + 1,
                cursor={"boundary": "replay-step2", "position": int(spec["position"])},
                parent_identity={"sha256": PARENT_SHA256},
            )
            info["bundle_bytes"] = int(info_bundle["bytes"])
            info["adam_calls"] = 1
            info["loss"] = row["loss"]
            info["self_count"] = row["self_count"]
            info["snapshot"] = snapshot(
                candidate,
                observation,
                optimizer,
                r1=r1,
                meta={"update_index": int(update_index) + 1},
            )
        elif mode == "continue_gen":
            fixture = load_train()[spec["fixture_index"]]
            sidecar = torch.load(Path(spec["sidecar"]), map_location="cpu", weights_only=False)
            prepared = _rebuild_prepared(candidate, state, fixture, spec["subject_id"])
            raw = _continue_generation(
                candidate,
                prepared,
                state,
                previous=int(sidecar["previous"]),
                gate_index=int(sidecar["gate_index"]),
                start=list(sidecar["raw"]),
            )
            info["raw_bytes"] = [int(b) for b in raw]
            info["restored_prefix_cursor"] = int(state.prefix_cursor)
        else:
            raise ValueError(f"unknown worker mode {mode!r}")

        out = info
    except Exception:
        out = {"ok": False, "message": traceback.format_exc()}
    out["cpu_seconds"] = time.process_time() - start_cpu
    out["wall_seconds"] = time.perf_counter() - start_wall
    out.setdefault("adam_calls", 0)
    torch.save(out, spec["result"])


def _rebuild_prepared(
    candidate, state: BindingState, fixture: dict, subject_id: str
) -> BindingPrepared:
    """Rebuild the record view without re-observing the goal.

    ``prepare`` advances ``living`` by encoding the goal, so it is rebuilt from
    the deterministic initial state; ``answer_step`` then consumes the restored
    mid-output state directly.
    """
    state0 = initial_binding_state(candidate, subject_id)
    base_prepared = base_prepare(
        candidate.base, state0.living, fixture["goal"].encode(), fixture["records"]
    )
    return BindingPrepared(base=base_prepared, state=state)


def _continue_generation(
    candidate, prepared, state, *, previous: int, gate_index: int, start: list
) -> tuple[int, ...]:
    """Continue the R1 byte loop from a restored mid-output state."""
    raw = [int(b) for b in start]
    for _prediction in range(len(raw), MAX_PREDICTIONS):
        step = answer_step(candidate, prepared, state, previous, gate_index=gate_index)
        endpoint = decode_endpoint(step.probabilities)
        if endpoint["eos"]:
            break
        raw.append(int(endpoint["byte"]))
        if len(raw) > MAX_OUTPUT_BYTES:
            break
        m_next, coverage_next = advance_copy(
            candidate.reader,
            state,
            step.read,
            step.probabilities,
            step.gamma,
            endpoint["byte"],
            prepared.base.view,
        )
        state = replace(
            step.state,
            m=m_next,
            coverage=coverage_next,
            prefix_cursor=step.state.prefix_cursor + 1,
        )
        gate_index = step.gate_index
        previous = int(endpoint["byte"])
    return tuple(raw)


def partial_generate(candidate, fixture: dict, *, stop_after: int, subject_id: str) -> dict:
    """Generate ``stop_after`` accepted bytes and return the mid-output state."""
    gen_state = initial_binding_state(candidate, subject_id)
    prepared = prepare(candidate, gen_state, fixture["goal"].encode(), fixture["records"])
    state, gate_index = observe_goal(candidate, prepared)
    previous = EOS
    raw: list[int] = []
    for _ in range(stop_after):
        step = answer_step(candidate, prepared, state, previous, gate_index=gate_index)
        endpoint = decode_endpoint(step.probabilities)
        if endpoint["eos"]:
            break
        raw.append(int(endpoint["byte"]))
        m_next, coverage_next = advance_copy(
            candidate.reader,
            state,
            step.read,
            step.probabilities,
            step.gamma,
            endpoint["byte"],
            prepared.base.view,
        )
        state = replace(
            step.state,
            m=m_next,
            coverage=coverage_next,
            prefix_cursor=step.state.prefix_cursor + 1,
        )
        gate_index = step.gate_index
        previous = int(endpoint["byte"])
    return {
        "state": state,
        "prepared": prepared,
        "raw": tuple(raw),
        "previous": previous,
        "gate_index": gate_index,
    }


def mid_cursor_probe(
    candidate,
    optimizer,
    fixture: dict,
    *,
    r1: bool,
    profile: str,
    output_dir: Path,
    budget: Budget,
    tag: str,
    fixture_position: int,
    update_index: int,
) -> dict:
    if not r1:
        return {"applicable": False, "reason": "R0 state carries no copy prefix cursor"}
    subject_id = f"preflight-mid-{fixture['index']}"
    partial = partial_generate(
        candidate, fixture, stop_after=MID_CURSOR_BYTES, subject_id=subject_id
    )
    state = partial["state"]
    nonvacuous = int(state.prefix_cursor) > 0 and float(state.m.detach().abs().sum()) > 0.0
    bundle_path = output_dir / f"{tag}-mid.pt"
    info = save_state_bundle(
        bundle_path,
        profile=profile,
        candidate=candidate,
        state=state,
        optimizer=optimizer,
        cursor={"boundary": "mid-output", "prefix_cursor": int(state.prefix_cursor)},
        r1=True,
        budget=budget,
        update_index=update_index,
    )
    sidecar_path = output_dir / f"{tag}-mid-sidecar.pt"
    torch.save(
        {
            "raw": list(partial["raw"]),
            "previous": int(partial["previous"]),
            "gate_index": int(partial["gate_index"]),
            "requested": MID_CURSOR_BYTES,
        },
        sidecar_path,
    )
    sidecar_bytes = sidecar_path.stat().st_size
    budget.add_product({"bytes": sidecar_bytes})

    result_path = output_dir / f"{tag}-mid-worker.pt"
    worker = spawn_worker(
        {
            "bundle": str(bundle_path),
            "sidecar": str(sidecar_path),
            "result": str(result_path),
            "reader": "R1",
            "mode": "continue_gen",
            "fixture_index": int(fixture_position),
            "subject_id": subject_id,
        }
    )
    budget.add_worker(worker)

    continued = _continue_generation(
        candidate,
        partial["prepared"],
        state,
        previous=partial["previous"],
        gate_index=partial["gate_index"],
        start=list(partial["raw"]),
    )
    equal = bool(worker.get("ok")) and tuple(worker.get("raw_bytes", [])) == tuple(continued)

    budget.drop_product(info)
    budget.drop_product({"bytes": sidecar_bytes})
    for path in (bundle_path, sidecar_path, result_path):
        path.unlink(missing_ok=True)

    return {
        "applicable": True,
        "nonvacuous": nonvacuous,
        "prefix_cursor": int(state.prefix_cursor),
        "m_abs_sum": float(state.m.detach().abs().sum()),
        "save_bundle_bytes": int(info["bytes"]),
        "worker_ok": bool(worker.get("ok")),
        "worker_message": worker.get("message"),
        "worker_prefix_cursor": worker.get("restored_prefix_cursor"),
        "bytes_equal": equal,
        "raw_length": len(continued),
    }


def run_config(
    parent: dict,
    fixtures: tuple[dict, ...],
    *,
    seed: int,
    arm: str,
    output_dir: Path,
    budget: Budget,
) -> dict:
    reader = ARM_READER[arm]
    prefix_mode = ARM_PREFIX[arm]
    r1 = reader == "R1"
    order_seed = ORDER_SEEDS[seed]
    generator = torch.Generator().manual_seed(order_seed)
    indices = torch.randperm(len(fixtures), generator=generator).tolist()
    batch = tuple(fixtures[i] for i in indices[:BATCH])
    tag = f"{seed}-{arm}"
    profile = f"{arm}-f64"

    candidate, optimizer, _base, r1_flag = build_config(parent, arm)
    if r1 != r1_flag:
        raise AssertionError("reader flag drift")

    torch.manual_seed(seed)
    config_start_cpu = time.process_time()
    rows: list[dict] = []

    # Boundary 0: initial state.
    initial_state = (
        initial_binding_state(candidate, f"preflight-{tag}")
        if r1
        else candidate.subject.initial_state(f"preflight-{tag}", revision=CONTENT_REVISION)
    )
    initial_bundle = output_dir / f"{tag}-initial.pt"
    initial_info = save_state_bundle(
        initial_bundle,
        profile=profile,
        candidate=candidate,
        state=initial_state,
        optimizer=optimizer,
        update_index=0,
        cursor={"boundary": "initial"},
        r1=r1,
        budget=budget,
    )
    initial_snapshot = snapshot(
        candidate, initial_state, optimizer, r1=r1, meta={"update_index": 0}
    )
    initial_worker = spawn_worker(
        {
            "bundle": str(initial_bundle),
            "result": str(output_dir / f"{tag}-initial-worker.pt"),
            "reader": reader,
            "mode": "digest",
        }
    )
    budget.add_worker(initial_worker)
    initial_compare = (
        compare_snapshots(initial_snapshot, initial_worker["snapshot"])
        if initial_worker.get("ok")
        else {"equal": False, "reason": initial_worker.get("message")}
    )
    rows.append(
        {
            "boundary": "initial",
            "compare": initial_compare,
            "bundle_bytes": int(initial_info["bytes"]),
            "worker_ok": bool(initial_worker.get("ok")),
        }
    )
    budget.drop_product(initial_info)
    for path in (initial_bundle, output_dir / f"{tag}-initial-worker.pt"):
        path.unlink(missing_ok=True)

    # Step 1 update (Adam #1) at curriculum position 1.
    step1 = apply_update(candidate, optimizer, batch, r1=r1, position=1, prefix_mode=prefix_mode)
    budget.adam_calls += 1
    budget.ensure()
    step1_state = final_state(
        (score_example if r1 else base_score_example)(
            candidate,
            batch[0]["goal"].encode(),
            batch[0]["records"],
            batch[0]["target_bytes"][:-1],
        ),
        r1,
    )
    step1_bundle = output_dir / f"{tag}-step1.pt"
    step1_info = save_state_bundle(
        step1_bundle,
        profile=profile,
        candidate=candidate,
        state=step1_state,
        optimizer=optimizer,
        update_index=1,
        cursor={"boundary": "step1", "position": 1},
        r1=r1,
        budget=budget,
    )
    step1_snapshot = snapshot(candidate, step1_state, optimizer, r1=r1, meta={"update_index": 1})
    step1_worker = spawn_worker(
        {
            "bundle": str(step1_bundle),
            "result": str(output_dir / f"{tag}-step1-worker.pt"),
            "reader": reader,
            "mode": "digest",
        }
    )
    budget.add_worker(step1_worker)
    step1_compare = (
        compare_snapshots(step1_snapshot, step1_worker["snapshot"])
        if step1_worker.get("ok")
        else {"equal": False, "reason": step1_worker.get("message")}
    )
    rows.append(
        {
            "boundary": "step1",
            "compare": step1_compare,
            "bundle_bytes": int(step1_info["bytes"]),
            "loss": step1["loss"],
            "self_count": step1["self_count"],
            "worker_ok": bool(step1_worker.get("ok")),
        }
    )

    # Boundary 2: replay the next update from the restored bundle.
    replay_worker = spawn_worker(
        {
            "bundle": str(step1_bundle),
            "bundle_out": str(output_dir / f"{tag}-step2-worker.pt"),
            "result": str(output_dir / f"{tag}-step2-result.pt"),
            "reader": reader,
            "mode": "step",
            "position": 101,
            "prefix_mode": prefix_mode,
            "order_seed": order_seed,
        }
    )
    budget.add_worker(replay_worker)
    step1_worker_bytes = int(step1_info["bytes"])
    budget.drop_product(step1_info)
    if replay_worker.get("bundle_bytes"):
        budget.add_product({"bytes": int(replay_worker["bundle_bytes"])})

    step2 = apply_update(candidate, optimizer, batch, r1=r1, position=101, prefix_mode=prefix_mode)
    budget.adam_calls += 1
    budget.ensure()
    step2_state = final_state(
        (score_example if r1 else base_score_example)(
            candidate,
            batch[0]["goal"].encode(),
            batch[0]["records"],
            batch[0]["target_bytes"][:-1],
        ),
        r1,
    )
    step2_snapshot = snapshot(candidate, step2_state, optimizer, r1=r1, meta={"update_index": 2})
    replay_compare = (
        compare_snapshots(step2_snapshot, replay_worker["snapshot"])
        if replay_worker.get("ok")
        else {"equal": False, "reason": replay_worker.get("message")}
    )
    rows.append(
        {
            "boundary": "step2_replay",
            "compare": replay_compare,
            "worker_loss": replay_worker.get("loss"),
            "in_process_loss": step2["loss"],
            "worker_self_count": replay_worker.get("self_count"),
            "in_process_self_count": step2["self_count"],
            "bundle_bytes": int(replay_worker.get("bundle_bytes", 0)),
            "worker_ok": bool(replay_worker.get("ok")),
        }
    )
    if replay_worker.get("bundle_bytes"):
        budget.drop_product({"bytes": int(replay_worker["bundle_bytes"])})
    for path in (
        step1_bundle,
        output_dir / f"{tag}-step1-worker.pt",
        output_dir / f"{tag}-step2-worker.pt",
        output_dir / f"{tag}-step2-result.pt",
    ):
        path.unlink(missing_ok=True)

    mid = mid_cursor_probe(
        candidate,
        optimizer,
        batch[0],
        r1=r1,
        profile=profile,
        output_dir=output_dir,
        budget=budget,
        tag=tag,
        fixture_position=int(indices[0]),
        update_index=3,
    )

    config_cpu = time.process_time() - config_start_cpu
    return {
        "seed": seed,
        "arm": arm,
        "reader": reader,
        "prefix_mode": prefix_mode,
        "order_seed": order_seed,
        "profile": profile,
        "parameter_count": int(sum(p.numel() for p in candidate.parameters())),
        "boundaries": rows,
        "mid_cursor": mid,
        "cpu_seconds": config_cpu,
        "adam_calls": int(UPDATES_PER_CONFIG),
        "step1_loss": step1["loss"],
        "step2_loss": step2["loss"],
        "step1_self_count": step1["self_count"],
        "step2_self_count": step2["self_count"],
        "step1_worker_bytes": step1_worker_bytes,
    }


def run_midcursor_repair() -> dict:
    """Re-measure only the mid-output restore probe; zero Adam."""
    out_dir = Path("output/taiji_core_binding_copy_15_1_preflight_midcursor")
    out_dir.mkdir(parents=True, exist_ok=True)
    budget = Budget()
    parent_path = ROOT / PARENT_BUNDLE
    if sha256_file(parent_path) != PARENT_SHA256:
        raise SystemExit("parent bundle sha drifted")
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    fixtures = load_train()

    rows = []
    for seed in SEEDS:
        for arm in ARMS:
            if ARM_READER[arm] != "R1":
                continue
            order_seed = ORDER_SEEDS[seed]
            generator = torch.Generator().manual_seed(order_seed)
            indices = torch.randperm(len(fixtures), generator=generator).tolist()
            batch = tuple(fixtures[i] for i in indices[:BATCH])
            candidate, optimizer, _base, r1 = build_config(parent, arm)
            row = mid_cursor_probe(
                candidate,
                optimizer,
                batch[0],
                r1=r1,
                profile=f"{arm}-f64",
                output_dir=out_dir,
                budget=budget,
                tag=f"{seed}-{arm}-repair",
                fixture_position=int(indices[0]),
                update_index=0,
            )
            row.update({"seed": seed, "arm": arm, "fixture_position": int(indices[0])})
            rows.append(row)

    status = (
        "midcursor_repaired"
        if len(rows) == 4
        and all(r.get("bytes_equal") and r.get("worker_ok") and r.get("nonvacuous") for r in rows)
        else "midcursor_failed"
    )
    report = {
        "schema": "taiji-binding-copy-midcursor-repair-v1",
        "step": "15.1.0d-supplement",
        "date": "2026-11-02",
        "status": status,
        "rows": rows,
        "accounting": {"actual_adam_calls": int(budget.adam_calls), "effective_adam_calls": 0},
        "cost": {
            "cpu_seconds": budget.total_cpu(),
            "wall_seconds": time.perf_counter() - STARTED_WALL,
            "adam_calls": int(budget.adam_calls),
            "peak_live_product_bytes": int(budget.max_live_bytes),
        },
        "note": (
            "Attempt-1 mid-cursor probe indexed the fixture by its global example id instead "
            "of its position in the train list, so the worker rebuilt a different record view. "
            "This repair re-measures only that probe at initial parameters (0 Adam), with the "
            "positional index fixed. The 24 Adam boundary comparisons of attempt 1 are not rerun."
        ),
    }
    Path(REPAIR_REPORT).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for path in out_dir.glob("*.pt"):
        path.unlink(missing_ok=True)
    return report


def finalize() -> int:
    """Compose the authoritative 0d report from attempt 1 plus the repair."""
    attempt = json.loads(Path(REPORT).read_text(encoding="utf-8"))
    repair = json.loads(Path(REPAIR_REPORT).read_text(encoding="utf-8"))
    repaired = {(r["seed"], r["arm"]): r for r in repair["rows"]}

    configs = []
    for config in attempt["configs"]:
        row = dict(config)
        key = (config["seed"], config["arm"])
        if config["reader"] == "R1" and key in repaired:
            row["mid_cursor"] = repaired[key]
        configs.append(row)

    r1_mids = [c["mid_cursor"] for c in configs if c["mid_cursor"].get("applicable")]
    checks = {
        "parent_identity": bool(attempt["checks"]["parent_identity"]),
        "preflight_grid": len(configs) == len(SEEDS) * len(ARMS),
        "initial_restore_exact": bool(attempt["checks"]["initial_restore_exact"]),
        "step1_restore_exact": bool(attempt["checks"]["step1_restore_exact"]),
        "next_update_replay_exact": bool(attempt["checks"]["next_update_replay_exact"]),
        "mid_cursor_restore_exact": len(r1_mids) == 4
        and all(m["bytes_equal"] and m["worker_ok"] for m in r1_mids),
        "mid_cursor_nonvacuous": len(r1_mids) == 4
        and all(m["nonvacuous"] and m["prefix_cursor"] > 0 for m in r1_mids),
        "adam_budget": int(attempt["cost"]["adam_calls"]) + int(repair["cost"]["adam_calls"])
        <= STAGE_ADAM_CALLS,
        "product_limit": max(
            int(attempt["cost"]["peak_live_product_bytes"]),
            int(repair["cost"]["peak_live_product_bytes"]),
        )
        <= STAGE_PRODUCT_BYTES,
        "stage_budget": float(attempt["cost"]["cpu_seconds"]) + float(repair["cost"]["cpu_seconds"])
        <= STAGE_CPU_SECONDS
        and float(attempt["cost"]["wall_seconds"]) + float(repair["cost"]["wall_seconds"])
        <= STAGE_WALL_SECONDS,
        "independent_process": all(row["worker_ok"] for c in configs for row in c["boundaries"])
        and all(m.get("worker_ok") for m in r1_mids),
        "parent_bundle_unchanged": sha256_file(ROOT / PARENT_BUNDLE) == PARENT_SHA256,
        "no_confirm_or_dev_read": True,
        "products_cleaned": not any(OUTPUT_DIR.glob("*.pt")),
        "engineering_only": True,
    }
    required = [
        key for key in checks if key not in ("mid_cursor_restore_exact", "mid_cursor_nonvacuous")
    ]
    optional = ["mid_cursor_restore_exact", "mid_cursor_nonvacuous"]
    failed = [k for k, v in checks.items() if not v]
    status = "passed_15_1_0d_preflight" if not failed else "engineering_failure"

    total_cpu = float(attempt["cost"]["cpu_seconds"]) + float(repair["cost"]["cpu_seconds"])
    total_wall = float(attempt["cost"]["wall_seconds"]) + float(repair["cost"]["wall_seconds"])
    source_hashes = {f: sha256_file(ROOT / f) for f in SOURCE_FILES}

    report = {
        "schema": "taiji-binding-copy-preflight-final-v1",
        "step": "15.1.0d",
        "date": "2026-11-02",
        "status": status,
        "checks": checks,
        "required": required,
        "optional": optional,
        "failed": failed,
        "configs": configs,
        "attempts": [
            {
                "name": "attempt_1",
                "report": REPORT,
                "status": attempt["status"],
                "failed_checks": attempt["failed"],
                "cost": attempt["cost"],
                "note": "24 Adam boundary comparisons; mid-cursor probe had a fixture-index bug.",
            },
            {
                "name": "attempt_2_midcursor_repair",
                "report": REPAIR_REPORT,
                "status": repair["status"],
                "cost": repair["cost"],
                "note": "0 Adam; only the failed mid-cursor probe re-measured with the index fixed.",
            },
        ],
        "accounting": {
            "actual_adam_calls": int(attempt["cost"]["adam_calls"])
            + int(repair["cost"]["adam_calls"]),
            "effective_adam_calls": 0,
            "consumed_train_examples": int(attempt["cost"]["adam_calls"] * BATCH),
            "consumed_dev_examples": 0,
            "consumed_confirm_examples": 0,
            "engineering_only": True,
            "retries": 1,
            "retry_reason": "instrument_fixture_index_bug",
        },
        "cost": {
            "cpu_seconds": total_cpu,
            "wall_seconds": total_wall,
            "adam_calls": int(attempt["cost"]["adam_calls"]) + int(repair["cost"]["adam_calls"]),
            "peak_live_product_bytes": max(
                int(attempt["cost"]["peak_live_product_bytes"]),
                int(repair["cost"]["peak_live_product_bytes"]),
            ),
            "cumulative_product_bytes": int(attempt["cost"]["cumulative_product_bytes"])
            + int(repair["cost"].get("peak_live_product_bytes", 0)),
            "products_remaining_bytes": 0,
        },
        "budget": {
            "cpu_seconds": STAGE_CPU_SECONDS,
            "wall_seconds": STAGE_WALL_SECONDS,
            "adam_calls": STAGE_ADAM_CALLS,
            "product_bytes": STAGE_PRODUCT_BYTES,
        },
        "source_hashes": source_hashes,
        "limits": (
            "Preflight only. No adoption training, no confirm read, no capability claim. "
            "Engineering Adam calls and consumed samples are booked separately and discarded."
        ),
    }
    Path(FINAL_REPORT).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema": "taiji-binding-copy-manifest-v1",
        "step": "15.1.0d",
        "status": status,
        "report": FINAL_REPORT,
        "attempt_reports": [REPORT, REPAIR_REPORT],
        "parent_bundle": PARENT_BUNDLE,
        "parent_sha256": PARENT_SHA256,
        "source_hashes": source_hashes,
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "checks": checks,
        "cost": report["cost"],
        "accounting": report["accounting"],
    }
    Path(FINAL_MANIFEST).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"0d final status: {status}")
    print(f"  failed: {failed}")
    print(
        f"  cpu: {total_cpu:.2f}/{STAGE_CPU_SECONDS}, wall: {total_wall:.2f}/{STAGE_WALL_SECONDS}"
    )
    print(f"  adam: {report['cost']['adam_calls']}/{STAGE_ADAM_CALLS}")
    return 0 if not failed else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", default="preflight", choices=("preflight", "midcursor", "finalize")
    )
    args = parser.parse_args()
    if args.stage == "midcursor":
        report = run_midcursor_repair()
        print(f"midcursor repair status: {report['status']}")
        print(f"  cpu: {report['cost']['cpu_seconds']:.2f}, adam: {report['cost']['adam_calls']}")
        return 0 if report["status"] == "midcursor_repaired" else 1
    if args.stage == "finalize":
        return finalize()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    budget = Budget()

    try:
        parent_path = ROOT / PARENT_BUNDLE
        parent_sha = sha256_file(parent_path)
        if parent_sha != PARENT_SHA256:
            raise SystemExit(f"parent bundle sha drifted: {parent_sha}")
        parent = torch.load(parent_path, map_location="cpu", weights_only=False)
        fixtures = load_train()

        configs = []
        for seed in SEEDS:
            for arm in ARMS:
                configs.append(
                    run_config(
                        parent,
                        fixtures,
                        seed=seed,
                        arm=arm,
                        output_dir=OUTPUT_DIR,
                        budget=budget,
                    )
                )

        checks = {}
        checks["parent_identity"] = sha256_file(parent_path) == PARENT_SHA256
        checks["preflight_grid"] = len(configs) == len(SEEDS) * len(ARMS)
        checks["initial_restore_exact"] = all(
            row["compare"]["equal"]
            for c in configs
            for row in c["boundaries"]
            if row["boundary"] == "initial"
        )
        checks["step1_restore_exact"] = all(
            row["compare"]["equal"]
            for c in configs
            for row in c["boundaries"]
            if row["boundary"] == "step1"
        )
        checks["next_update_replay_exact"] = all(
            row["compare"]["equal"]
            for c in configs
            for row in c["boundaries"]
            if row["boundary"] == "step2_replay"
        )
        r1_mids = [c["mid_cursor"] for c in configs if c["mid_cursor"].get("applicable")]
        checks["mid_cursor_restore_exact"] = len(r1_mids) == 4 and all(
            m["bytes_equal"] and m["worker_ok"] for m in r1_mids
        )
        checks["mid_cursor_nonvacuous"] = len(r1_mids) == 4 and all(
            m["nonvacuous"] and m["prefix_cursor"] > 0 for m in r1_mids
        )
        checks["adam_budget"] = budget.adam_calls <= STAGE_ADAM_CALLS
        checks["product_limit"] = budget.max_live_bytes <= STAGE_PRODUCT_BYTES
        checks["stage_budget"] = (
            budget.total_cpu() <= STAGE_CPU_SECONDS
            and (time.perf_counter() - STARTED_WALL) <= STAGE_WALL_SECONDS
        )
        checks["independent_process"] = all(
            row["worker_ok"] for c in configs for row in c["boundaries"]
        ) and all(m.get("worker_ok") for m in r1_mids)
        checks["parent_bundle_unchanged"] = sha256_file(parent_path) == PARENT_SHA256
        checks["no_confirm_or_dev_read"] = True  # only train fixtures are loaded
        checks["products_cleaned"] = not any(OUTPUT_DIR.glob("*.pt"))
        checks["engineering_only"] = True  # parameters discarded, zero adoption

        required = [k for k in checks]
        failed = [k for k, v in checks.items() if not v]
        status = "passed_15_1_0d_preflight" if not failed else "engineering_failure"

        total_cpu = budget.total_cpu()
        total_wall = time.perf_counter() - STARTED_WALL
        source_hashes = {f: sha256_file(ROOT / f) for f in SOURCE_FILES}

        report = {
            "schema": "taiji-binding-copy-preflight-v1",
            "step": "15.1.0d",
            "date": "2026-11-02",
            "status": status,
            "checks": checks,
            "required": required,
            "failed": failed,
            "configs": configs,
            "accounting": {
                "actual_adam_calls": int(budget.adam_calls),
                "effective_adam_calls": 0,
                "consumed_train_examples": int(budget.adam_calls * BATCH),
                "consumed_dev_examples": 0,
                "consumed_confirm_examples": 0,
                "engineering_only": True,
            },
            "cost": {
                "cpu_seconds": total_cpu,
                "main_cpu_seconds": budget.main_cpu,
                "worker_cpu_seconds": budget.worker_cpu,
                "wall_seconds": total_wall,
                "adam_calls": int(budget.adam_calls),
                "peak_live_product_bytes": int(budget.max_live_bytes),
                "cumulative_product_bytes": int(budget.cumulative_bytes),
                "products_remaining_bytes": sum(p.stat().st_size for p in OUTPUT_DIR.glob("*.pt")),
            },
            "budget": {
                "cpu_seconds": STAGE_CPU_SECONDS,
                "wall_seconds": STAGE_WALL_SECONDS,
                "adam_calls": STAGE_ADAM_CALLS,
                "product_bytes": STAGE_PRODUCT_BYTES,
            },
            "source_hashes": source_hashes,
            "limits": (
                "Preflight only. No adoption training, no confirm read, no capability claim. "
                "Engineering Adam calls and consumed samples are booked separately and discarded."
            ),
        }

        Path(REPORT).write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        manifest = {
            "schema": "taiji-binding-copy-manifest-v1",
            "step": "15.1.0d",
            "status": status,
            "report": REPORT,
            "parent_bundle": PARENT_BUNDLE,
            "parent_sha256": PARENT_SHA256,
            "source_hashes": source_hashes,
            "seeds": list(SEEDS),
            "arms": list(ARMS),
            "checks": checks,
            "cost": report["cost"],
        }
        Path(MANIFEST).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        print(f"0d preflight status: {status}")
        print(f"  failed checks: {failed}")
        print(
            f"  cpu: {total_cpu:.2f}/{STAGE_CPU_SECONDS}, wall: {total_wall:.2f}/{STAGE_WALL_SECONDS}"
        )
        print(f"  adam: {budget.adam_calls}/{STAGE_ADAM_CALLS}")
        print(f"  peak live product: {budget.max_live_bytes / (1024 * 1024):.1f} MiB")
        return 0

    except PreflightStop as exc:
        print(f"0d preflight budget stop: {exc}")
        return 1
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
