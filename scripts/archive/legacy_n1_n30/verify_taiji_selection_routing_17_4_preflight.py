"""N17.4 checkpoint preflight: 6 cells x 3 Adam, restore and continuation bitwise.

This is the restore gate before the formal 17.5 dose.  Nothing here trains the
300-update run and nothing here reads confirm data.

Per cell (2 parent seeds x 3 arms):

U1  Adam #1, selection stage, batch 8 -> save step-1 routing bundle;
U2  Adam #2, same batch order -> prototype step-2 params/Adam/probability digests;
R2  an independent spawn process loads the step-1 bundle, runs the same update 2
    as Adam #3, and must reproduce the three prototype digests bit for bit;
C   teacher-force the frozen target bytes to two checkpoints -- position 8
    (mid-object copy, non-empty m) and position 15 (object->place switch, empty
    m but non-empty coverage) -- persist BindingState (m/coverage/keys/
    prefix_cursor) plus previous byte and gate index, restore in a process on
    the *same* step-2 candidate, and compare every later probability row.

A failure is reported with the observed differences; thresholds are never
relaxed.  Temporary bundles are deleted after the comparison and their bytes
are registered in the report and manifest.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import multiprocessing
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

torch.set_num_threads(1)

from taiji.binding_copy import (  # noqa: E402
    BindingPrepared,
    initial_binding_state,
    observe_goal,
    prepare,
)
from taiji.candidate_identity import candidate_identity  # noqa: E402
from taiji.living_core import EOS  # noqa: E402
from taiji.record_relation_core import CLIP_NORM  # noqa: E402
from taiji.selection_routing import (  # noqa: E402
    ARMS,
    _advanced_state,
    build_context,
    route_labels,
    routing_step,
    selection_ce,
)
from taiji.selection_routing_checkpoint import (  # noqa: E402
    binding_state_from_payload,
    build_routing_candidate,
    digest_optimizer_moments,
    digest_parameters,
    digest_probabilities,
    load_routing_bundle,
    optimizer_moment_tensors,
    parameter_tensors,
    save_routing_bundle,
)

SEEDS = (28031, 28032)
PARENTS = {
    28031: "output/taiji_core_binding_select_16_1_compare/bundles/28031-A-locked-step300.pt",
    28032: "output/taiji_core_binding_select_16_1_compare/bundles/28032-A-locked-step300.pt",
}
COMPARE_SCRIPT = "scripts/training/verify_taiji_binding_select_16_1_compare.py"
SELECT_PAYLOAD = "output/taiji_core_binding_select_16_1/select_v1.json"
DATA_PAYLOAD = "output/taiji_core_binding_copy_15_1/data_v2.json"
REPORT = Path("reports/taiji_selection_routing_17_4_preflight_20261103_v2.json")
MANIFEST = Path("plans/manifests/taiji_selection_routing_17_4_preflight_20261103_v2.json")
TMP_DIR = Path("output/taiji_core_selection_routing_17_4_preflight")

SOURCE_FILES = (
    "taiji/selection_routing.py",
    "taiji/selection_routing_checkpoint.py",
    "taiji/binding_select.py",
    "taiji/binding_copy.py",
    "taiji/record_relation_core.py",
    "taiji/living_core.py",
    "taiji/candidate_identity.py",
    "scripts/training/verify_taiji_binding_select_16_1_compare.py",
    "scripts/training/verify_taiji_selection_routing_17_4_preflight.py",
)

BATCH = 8
UPDATES_PREFLIGHT = 2
CHECKPOINTS = (2, 4, 6, 8, 10, 12, 14, 15)
CHECKPOINT_LABELS = {
    2: "prefix_before_object",
    4: "object_first_2_bytes",
    6: "object_4_bytes",
    8: "mid_object_copy",
    10: "object_7_bytes",
    12: "object_9_bytes",
    14: "object_last_byte",
    15: "object_to_place_switch",
}
OBJECT_BYTES_ADVANCED = 12
FIXTURE_INDEX = 0
ADAM_PER_CELL = 3
CONFIRM_READS = 0
WORKER_TIMEOUT = 2400

STARTED_CPU = time.process_time()
STARTED_WALL = time.perf_counter()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_compare_module():
    spec = importlib.util.spec_from_file_location("cmp17_4", ROOT / COMPARE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_fixtures() -> tuple[tuple[dict, ...], tuple[dict, ...], dict]:
    module = load_compare_module()
    train, dev, select_payload, data = module.load_data()
    splits = sorted({str(example["split"]) for example in data["examples"]})
    materialised = bool(select_payload.get("confirm_materialised", False))
    if splits != ["dev", "train"]:
        raise RuntimeError(f"unexpected data splits {splits}")
    if materialised:
        raise RuntimeError("confirm data is materialised; the preflight must not read it")
    audit = {
        "splits": splits,
        "examples": len(data["examples"]),
        "confirm_materialised": materialised,
        "confirm_examples_read": 0,
    }
    return train, dev, audit


def digest_sorted(items: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(items):
        array = items[name].detach().to(torch.float64).cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(f"|{tuple(array.shape)}|f64|".encode())
        digest.update(array.numpy().tobytes())
    return digest.hexdigest()


def binding_digest(binding, *, trainable: bool | None = None) -> str:
    items = {}
    for name, parameter in binding.named_parameters():
        if trainable is not None and bool(parameter.requires_grad) != trainable:
            continue
        items[f"binding.{name}"] = parameter.detach().to(torch.float64).cpu().clone()
    return digest_sorted(items)


def trainable_names(candidate) -> list[str]:
    names = [
        f"binding.{name}"
        for name, parameter in candidate.binding.named_parameters()
        if parameter.requires_grad
    ]
    names += [f"head.{name}" for name, _parameter in candidate.head.named_parameters()]
    if candidate.span_head is not None:
        names += [f"span.{name}" for name, _parameter in candidate.span_head.named_parameters()]
    return names


def selection_batch_loss(candidate, batch, *, tag: str, update: int) -> torch.Tensor:
    terms = []
    binding = candidate.binding
    for fixture in batch:
        goal = fixture["goal"].encode("utf-8")
        state = initial_binding_state(binding, f"{tag}-u{update}-{fixture['index']}")
        prepared = prepare(binding, state, goal, fixture["records"])
        context = build_context(candidate, goal, tuple(fixture["record_keys"]))
        routes = route_labels(fixture)
        terms.append(
            selection_ce(
                candidate,
                prepared,
                fixture["target_prefix"],
                fixture["target_bytes"],
                routes[0],
                routes[1],
                context,
                candidate.head,
            )["loss"]
        )
    return sum(terms) / len(terms)


def apply_selection_update(candidate, optimizer, batch, *, tag: str, update: int) -> dict:
    loss = selection_batch_loss(candidate, batch, tag=tag, update=update)
    optimizer.zero_grad()
    loss.backward()
    clip = float(torch.nn.utils.clip_grad_norm_(list(candidate.trainable_parameters()), CLIP_NORM))
    optimizer.step()
    return {"update": update, "loss": float(loss.detach()), "clip_scale": clip}


def forward_inputs(candidate, fixture: dict, tag: str):
    binding = candidate.binding
    goal = fixture["goal"].encode("utf-8")
    initial = initial_binding_state(binding, tag)
    prepared = prepare(binding, initial, goal, fixture["records"])
    state, gate_index = observe_goal(binding, prepared, gate_index=0)
    context = build_context(candidate, goal, tuple(fixture["record_keys"]))
    return initial, prepared, state, gate_index, context


def advance_rows(
    candidate,
    prepared,
    context,
    state,
    gate_index,
    previous,
    fixture: dict,
    arm: str,
    start_t: int,
    end_t: int | None = None,
):
    """Teacher-forced routing rows in [start_t, end_t); advances by target bytes."""
    total = len(fixture["target_bytes"])
    limit = total if end_t is None else end_t
    rows = []
    with torch.no_grad():
        for t in range(start_t, limit):
            step = routing_step(
                candidate,
                prepared,
                state,
                previous,
                context,
                candidate.head,
                arm=arm,
                gate_index=gate_index,
            )
            rows.append(step.probabilities.detach().clone())
            if t == total - 1:
                break
            accepted = fixture["target_prefix"][t]
            state, _, gate_index = _advanced_state(candidate, prepared, step, accepted)
            previous = accepted
    return rows, state, gate_index, previous


def teacher_forced_rows(candidate, fixture: dict, arm: str, tag: str) -> list[torch.Tensor]:
    _initial, prepared, state, gate_index, context = forward_inputs(candidate, fixture, tag)
    rows, _state, _gate, _previous = advance_rows(
        candidate, prepared, context, state, gate_index, EOS, fixture, arm, 0, None
    )
    return rows


def continuation_prototype(candidate, fixture: dict, arm: str, tag: str, checkpoint_t: int) -> dict:
    initial, prepared, state, gate_index, context = forward_inputs(candidate, fixture, tag)
    rows_full, _state, _gate, _previous = advance_rows(
        candidate, prepared, context, state, gate_index, EOS, fixture, arm, 0, None
    )
    initial_ckpt, prepared_ckpt, state_ckpt, gate_ckpt, context_ckpt = forward_inputs(
        candidate, fixture, tag
    )
    _rows_head, mid_state, gate_mid, previous_mid = advance_rows(
        candidate,
        prepared_ckpt,
        context_ckpt,
        state_ckpt,
        gate_ckpt,
        EOS,
        fixture,
        arm,
        0,
        checkpoint_t,
    )
    rows_cont, _state, _gate, _previous = advance_rows(
        candidate,
        prepared_ckpt,
        context_ckpt,
        mid_state,
        gate_mid,
        previous_mid,
        fixture,
        arm,
        checkpoint_t,
        None,
    )
    return {
        "rows_full": rows_full,
        "rows_continuation": rows_cont,
        "initial_payload": initial_ckpt.payload(),
        "mid_state": mid_state,
        "previous": int(previous_mid),
        "gate_index": int(gate_mid),
    }


def state_statistics(state) -> dict:
    m = state.m.detach()
    coverage = state.coverage.detach()
    return {
        "m_numel": int(m.numel()),
        "coverage_numel": int(coverage.numel()),
        "m_nonzero": int((m != 0).sum()),
        "coverage_nonzero": int((coverage != 0).sum()),
        "coverage_sum": float(coverage.sum()),
        "m_shape": list(m.shape),
        "coverage_shape": list(coverage.shape),
        "keys": len(state.keys),
        "record_offsets": len(state.record_offsets),
        "prefix_cursor": int(state.prefix_cursor),
    }


def restored_continuation_rows(candidate, fixture: dict, state_payload: dict, arm: str):
    binding = candidate.binding
    initial = binding_state_from_payload(state_payload["initial_state"], binding)
    goal = fixture["goal"].encode("utf-8")
    prepared_initial = prepare(binding, initial, goal, fixture["records"])
    mid = binding_state_from_payload(state_payload["mid_state"], binding)
    keys_match = mid.keys == prepared_initial.state.keys
    offsets_match = mid.record_offsets == prepared_initial.state.record_offsets
    if not keys_match or not offsets_match:
        raise ValueError("restored binding state keys or record offsets drifted")
    # prepare() only rebuilds the frozen view/nu surfaces; the carried m/coverage
    # stay the saved tensors, exactly as the prototype held them.
    prepared = BindingPrepared(base=prepared_initial.base, state=mid)
    context = build_context(candidate, goal, tuple(fixture["record_keys"]))
    rows, _state, _gate, _previous = advance_rows(
        candidate,
        prepared,
        context,
        mid,
        int(state_payload["gate_index"]),
        int(state_payload["previous"]),
        fixture,
        arm,
        int(state_payload["checkpoint_t"]),
        None,
    )
    return rows, keys_match and offsets_match


def _restore_worker(spec: dict) -> None:
    """Independent-process target: load step-1, Adam #3, digests + continuation."""
    started = time.process_time()
    result: dict = {"ok": False}
    try:
        candidate, optimizer, _state, update_index, _cursor = load_routing_bundle(spec["step1"])
        if update_index != 1:
            raise ValueError(f"step-1 bundle reports update {update_index}")
        train, dev, _audit = load_fixtures()
        fixture = dev[int(spec["fixture_index"])]
        module = load_compare_module()
        batch = module.schedule(int(spec["seed"]), train, UPDATES_PREFLIGHT)[1]
        row = apply_selection_update(
            candidate,
            optimizer,
            batch,
            tag=f"{spec['seed']}-{spec['arm']}-restore",
            update=2,
        )
        reference = torch.load(spec["reference"], map_location="cpu", weights_only=True)
        params = digest_parameters(candidate, reference=reference["parameters"])
        moments = digest_optimizer_moments(optimizer, reference=reference["moments"])
        probabilities = digest_probabilities(
            teacher_forced_rows(candidate, fixture, spec["arm"], f"{spec['seed']}-restore-probs"),
            reference=reference["probabilities"],
        )
        continuations = {}
        for checkpoint_t in spec["checkpoints"]:
            state_payload = torch.load(
                spec["states"][str(checkpoint_t)], map_location="cpu", weights_only=True
            )
            rows, state_ok = restored_continuation_rows(
                candidate, fixture, state_payload, spec["arm"]
            )
            digest = digest_probabilities(
                rows, reference=reference["continuations"][int(checkpoint_t)]
            )
            continuations[str(checkpoint_t)] = {
                "probs_max_abs_diff": digest["max_abs_diff"],
                "rows_compared": digest["rows"],
                "sha256": digest["sha256"],
                "state_keys_match": bool(state_ok),
            }
        result = {
            "ok": True,
            "seed": int(spec["seed"]),
            "arm": spec["arm"],
            "adam_calls": 1,
            "update_index": 2,
            "update": row,
            "parameter_sha256": params["sha256"],
            "adam_sha256": moments["sha256"],
            "probs_sha256": probabilities["sha256"],
            "restore": {
                "params_max_abs_diff": params["max_abs_diff"],
                "adam_max_abs_diff": moments["max_abs_diff"],
                "probs_max_abs_diff": probabilities["max_abs_diff"],
            },
            "continuation": continuations,
            "binding_sha256": binding_digest(candidate.binding),
            "nontrainable_sha256": binding_digest(candidate.binding, trainable=False),
        }
    except Exception:
        result = {
            "ok": False,
            "seed": spec.get("seed"),
            "arm": spec.get("arm"),
            "message": traceback.format_exc(),
        }
    result["cpu_seconds"] = time.process_time() - started
    Path(spec["result"]).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def spawn_restore(spec: dict) -> dict:
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=_restore_worker, args=(spec,))
    process.start()
    process.join(timeout=WORKER_TIMEOUT)
    if process.is_alive():
        process.terminate()
        process.join()
        return {"ok": False, "message": "restore worker timed out"}
    if process.exitcode != 0:
        return {"ok": False, "message": f"restore worker exit {process.exitcode}"}
    path = Path(spec["result"])
    if not path.exists():
        return {"ok": False, "message": "restore worker wrote no result"}
    return json.loads(path.read_text(encoding="utf-8"))


def file_entry(path: Path, kind: str) -> dict:
    return {
        "path": str(path),
        "kind": kind,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def discard(entries: list[dict]) -> list[dict]:
    for entry in entries:
        path = Path(entry["path"])
        if path.exists():
            path.unlink()
        entry["deleted"] = True
    return entries


def route_audit(fixture: dict) -> dict:
    record_count = len(fixture["record_keys"])
    route_index, route_stage = route_labels(fixture)
    if len(fixture["target_bytes"]) != 28 or len(fixture["target_prefix"]) != 27:
        raise RuntimeError("target length drifted from the frozen 28-position form")
    object_positions = [
        t for t, label in enumerate(route_index) if label == int(fixture["selection_label"])
    ]
    null_positions = [t for t, label in enumerate(route_index) if label == record_count]
    if object_positions[:1] != [3] or len(object_positions) < OBJECT_BYTES_ADVANCED:
        raise RuntimeError("object span does not cover the frozen 12 object bytes")
    return {
        "record_count": record_count,
        "object_positions": object_positions[:12],
        "null_positions": null_positions,
        "null_positions_after_checkpoint": {
            str(checkpoint_t): [t for t in null_positions if t >= checkpoint_t]
            for checkpoint_t in CHECKPOINTS
        },
        "stages": sorted(set(route_stage)),
    }


def run_cell(seed: int, arm: str, train, dev, module, artifacts: list[dict]) -> dict:
    tag = f"{seed}-{arm}"
    parent_rel = PARENTS[seed]
    parent_path = ROOT / parent_rel
    parent_sha_before = sha256_file(parent_path)
    parent_bytes = parent_path.stat().st_size

    candidate, optimizer = build_routing_candidate(parent_path, seed, arm)
    identity = candidate_identity(candidate)
    binding_before = binding_digest(candidate.binding)
    nontrainable_before = binding_digest(candidate.binding, trainable=False)
    names = trainable_names(candidate)
    names_digest = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()

    batches = module.schedule(seed, train, UPDATES_PREFLIGHT)
    update_1 = apply_selection_update(
        candidate, optimizer, batches[0], tag=f"{tag}-proto", update=1
    )
    state0 = initial_binding_state(candidate.binding, f"ckpt-{tag}")
    step1_path = TMP_DIR / f"{tag}-step1.pt"
    step1 = save_routing_bundle(
        step1_path,
        candidate,
        optimizer,
        update_index=1,
        cursor={"stage": "N17.4", "seed": seed, "arm": arm, "update": 1},
        parent_identity={
            "bundle": parent_rel,
            "sha256": parent_sha_before,
            "seed": seed,
            "arm": arm,
        },
        state=state0,
    )
    artifacts.append({**step1, "kind": "routing_step1_bundle"})

    update_2 = apply_selection_update(
        candidate, optimizer, batches[1], tag=f"{tag}-proto", update=2
    )
    binding_after = binding_digest(candidate.binding)
    nontrainable_after = binding_digest(candidate.binding, trainable=False)

    fixture = dev[FIXTURE_INDEX]
    audit = route_audit(fixture)
    prototype_params = digest_parameters(candidate)
    prototype_moments = digest_optimizer_moments(optimizer)
    continuations = {}
    for checkpoint_t in CHECKPOINTS:
        continuations[checkpoint_t] = continuation_prototype(
            candidate,
            fixture,
            arm,
            f"{tag}-continuation-{checkpoint_t}",
            checkpoint_t,
        )
    prototype_probs = digest_probabilities(continuations[CHECKPOINTS[0]]["rows_full"])
    prototype_continuations = {
        checkpoint_t: digest_probabilities(row["rows_continuation"])
        for checkpoint_t, row in continuations.items()
    }
    stats_by_checkpoint = {
        checkpoint_t: state_statistics(row["mid_state"])
        for checkpoint_t, row in continuations.items()
    }

    reference_path = TMP_DIR / f"{tag}-reference.pt"
    torch.save(
        {
            "parameters": parameter_tensors(candidate),
            "moments": optimizer_moment_tensors(optimizer),
            "probabilities": continuations[CHECKPOINTS[0]]["rows_full"],
            "continuations": {
                checkpoint_t: row["rows_continuation"]
                for checkpoint_t, row in continuations.items()
            },
        },
        reference_path,
    )
    reference_entry = file_entry(reference_path, "prototype_reference")
    artifacts.append(reference_entry)

    state_entries = []
    state_paths = {}
    for checkpoint_t, row in continuations.items():
        state_path = TMP_DIR / f"{tag}-state-{checkpoint_t}.pt"
        torch.save(
            {
                "initial_state": row["initial_payload"],
                "mid_state": row["mid_state"].payload(),
                "previous": row["previous"],
                "gate_index": row["gate_index"],
                "checkpoint_t": checkpoint_t,
                "fixture_index": FIXTURE_INDEX,
                "arm": arm,
                "seed": seed,
            },
            state_path,
        )
        state_paths[str(checkpoint_t)] = str(state_path.resolve())
        entry = file_entry(state_path, f"continuation_state_{checkpoint_t}")
        state_entries.append(entry)
        artifacts.append(entry)

    result_path = TMP_DIR / f"{tag}-restore.json"
    spec = {
        "step1": str(step1_path.resolve()),
        "reference": str(reference_path.resolve()),
        "states": state_paths,
        "checkpoints": list(CHECKPOINTS),
        "result": str(result_path.resolve()),
        "seed": seed,
        "arm": arm,
        "fixture_index": FIXTURE_INDEX,
    }
    child = spawn_restore(spec)
    result_entry = file_entry(result_path, "restore_result") if result_path.exists() else None
    if result_entry is not None:
        artifacts.append(result_entry)

    parent_sha_after = sha256_file(parent_path)
    parent_unchanged = parent_sha_after == parent_sha_before
    protection = {
        "binding_unchanged": binding_after == binding_before,
        "nontrainable_unchanged": nontrainable_after == nontrainable_before,
        "child_binding_sha256": child.get("binding_sha256"),
        "child_nontrainable_sha256": child.get("nontrainable_sha256"),
        "child_binding_matches": child.get("binding_sha256") == binding_before,
        "child_nontrainable_matches": child.get("nontrainable_sha256") == nontrainable_before,
    }
    child_ok = bool(child.get("ok"))
    restore = {
        "params_max_abs_diff": child.get("restore", {}).get("params_max_abs_diff"),
        "adam_max_abs_diff": child.get("restore", {}).get("adam_max_abs_diff"),
        "probs_max_abs_diff": child.get("restore", {}).get("probs_max_abs_diff"),
        "all_bitwise_equal": False,
        "prototype_parameter_sha256": prototype_params["sha256"],
        "restored_parameter_sha256": child.get("parameter_sha256"),
        "prototype_adam_sha256": prototype_moments["sha256"],
        "restored_adam_sha256": child.get("adam_sha256"),
        "prototype_probs_sha256": prototype_probs["sha256"],
        "restored_probs_sha256": child.get("probs_sha256"),
    }
    if child_ok:
        restore["all_bitwise_equal"] = bool(
            child["restore"]["params_max_abs_diff"] == 0.0
            and child["restore"]["adam_max_abs_diff"] == 0.0
            and child["restore"]["probs_max_abs_diff"] == 0.0
            and child.get("parameter_sha256") == prototype_params["sha256"]
            and child.get("adam_sha256") == prototype_moments["sha256"]
            and child.get("probs_sha256") == prototype_probs["sha256"]
        )

    continuation_blocks = {}
    for checkpoint_t in CHECKPOINTS:
        row = continuations[checkpoint_t]
        child_cont = (child.get("continuation") or {}).get(str(checkpoint_t), {})
        block = {
            "label": CHECKPOINT_LABELS[checkpoint_t],
            "fixture_index": FIXTURE_INDEX,
            "checkpoint_position": checkpoint_t,
            "bytes_consumed": checkpoint_t,
            "prefix_bytes": checkpoint_t - OBJECT_BYTES_ADVANCED,
            "object_bytes_advanced": checkpoint_t - 3,
            "steps_compared": prototype_continuations[checkpoint_t]["rows"],
            "probs_max_abs_diff": child_cont.get("probs_max_abs_diff"),
            "restored_rows": child_cont.get("rows_compared"),
            "all_bitwise_equal": False,
            "prototype_sha256": prototype_continuations[checkpoint_t]["sha256"],
            "restored_sha256": child_cont.get("sha256"),
            "keys_match": child_cont.get("state_keys_match"),
            "previous_byte": row["previous"],
            "gate_index": row["gate_index"],
            "previous_byte_is_target_prefix_at_checkpoint": row["previous"]
            == fixture["target_prefix"][checkpoint_t - 1],
            "state": stats_by_checkpoint[checkpoint_t],
            "route": {
                "object_positions": audit["object_positions"],
                "null_positions_in_sequence": audit["null_positions"],
                "null_positions_after_checkpoint": audit["null_positions_after_checkpoint"][
                    str(checkpoint_t)
                ],
                "record_count": audit["record_count"],
                "null_route_covered": bool(
                    audit["null_positions_after_checkpoint"][str(checkpoint_t)]
                ),
            },
        }
        if child_ok:
            block["all_bitwise_equal"] = bool(
                child_cont.get("probs_max_abs_diff") == 0.0
                and child_cont.get("sha256") == prototype_continuations[checkpoint_t]["sha256"]
                and child_cont.get("rows_compared") == prototype_continuations[checkpoint_t]["rows"]
                and child_cont.get("state_keys_match")
            )
        continuation_blocks[str(checkpoint_t)] = block

    object_states = {
        checkpoint_t: block["state"]
        for checkpoint_t, block in continuation_blocks.items()
        if 3 <= int(checkpoint_t) <= 14
    }
    non_empty_positions = [
        int(checkpoint_t)
        for checkpoint_t, state in object_states.items()
        if state["m_numel"] > 0 and state["m_nonzero"] > 0 and state["coverage_nonzero"] > 0
    ]
    non_empty_state = bool(non_empty_positions)
    mid_copy_contrast = continuation_blocks.get("8", {}).get("state")
    switch_point = continuation_blocks[str(CHECKPOINTS[-1])]
    cell = {
        "seed": seed,
        "arm": arm,
        "candidate": identity,
        "adam_calls": 2 + int(child.get("adam_calls", 0)),
        "step1_sha256": step1["sha256"],
        "step1_bytes": step1["bytes"],
        "updates": [update_1, update_2],
        "restore": restore,
        "continuation": continuation_blocks,
        "non_empty_state": non_empty_state,
        "non_empty_mid_copy": bool(non_empty_positions),
        "non_empty_positions": non_empty_positions,
        "object_state_table": {
            checkpoint_t: {
                "m_nonzero": state["m_nonzero"],
                "m_numel": state["m_numel"],
                "coverage_nonzero": state["coverage_nonzero"],
                "coverage_sum": state["coverage_sum"],
                "prefix_cursor": state.get("prefix_cursor"),
            }
            for checkpoint_t, state in object_states.items()
        },
        "empty_mid_contrast_at_8": mid_copy_contrast,
        "switch_point_m": switch_point["state"]["m_nonzero"],
        "switch_point_coverage_nonzero": switch_point["state"]["coverage_nonzero"],
        "parent_unchanged": parent_unchanged,
        "protection": protection,
        "parameter_names_sha256": names_digest,
        "trainable_parameters": len(names),
        "binding_sha256": binding_before,
        "nontrainable_sha256": nontrainable_before,
        "prototype": {
            "parameter_sha256": prototype_params["sha256"],
            "adam_sha256": prototype_moments["sha256"],
            "probs_sha256": prototype_probs["sha256"],
            "continuation_sha256": {
                str(checkpoint_t): row["sha256"]
                for checkpoint_t, row in prototype_continuations.items()
            },
        },
        "parent": {
            "path": parent_rel,
            "sha256_before": parent_sha_before,
            "sha256_after": parent_sha_after,
            "bytes": parent_bytes,
        },
        "child": {
            "ok": child_ok,
            "cpu_seconds": child.get("cpu_seconds"),
            "message": child.get("message"),
        },
        "artifacts": [],
    }
    discard_entries = [entry for entry in (reference_entry, *state_entries, result_entry) if entry]
    cell["artifacts"] = discard(discard_entries)
    del candidate, optimizer, continuations
    gc.collect()
    torch.set_num_threads(1)
    return cell


def cell_passed(cell: dict) -> bool:
    return bool(
        cell["restore"]["all_bitwise_equal"]
        and all(block["all_bitwise_equal"] for block in cell["continuation"].values())
        and cell["non_empty_state"]
        and cell["non_empty_mid_copy"]
        and cell["parent_unchanged"]
        and cell["protection"]["binding_unchanged"]
        and cell["protection"]["nontrainable_unchanged"]
        and cell["protection"]["child_binding_matches"]
        and cell["protection"]["child_nontrainable_matches"]
        and cell["adam_calls"] == ADAM_PER_CELL
    )


def failure_reason(cell: dict) -> str:
    reasons = []
    if not cell["restore"]["all_bitwise_equal"]:
        reasons.append(
            "restore mismatch params="
            f"{cell['restore']['params_max_abs_diff']} "
            f"adam={cell['restore']['adam_max_abs_diff']} "
            f"probs={cell['restore']['probs_max_abs_diff']}"
        )
    for checkpoint_t, block in cell["continuation"].items():
        if not block["all_bitwise_equal"]:
            reasons.append(
                f"continuation t={checkpoint_t} mismatch diff={block['probs_max_abs_diff']}"
            )
    if not cell["non_empty_mid_copy"]:
        reasons.append("no non-empty m/coverage at any object-span checkpoint (3..14)")
    elif not cell["non_empty_state"]:
        reasons.append("checkpoint m/coverage is empty")
    if not cell["parent_unchanged"]:
        reasons.append("parent bundle bytes changed")
    if not cell["protection"]["nontrainable_unchanged"]:
        reasons.append("frozen parameter digest changed")
    if cell["adam_calls"] != ADAM_PER_CELL:
        reasons.append(f"adam calls {cell['adam_calls']} != {ADAM_PER_CELL}")
    return "; ".join(reasons)


def main() -> int:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    train, dev, audit = load_fixtures()
    module = load_compare_module()
    artifacts: list[dict] = []
    cells: list[dict] = []
    failures: list[str] = []
    for seed in SEEDS:
        for arm in ARMS:
            cell = run_cell(seed, arm, train, dev, module, artifacts)
            cells.append(cell)
            print(
                json.dumps(
                    {
                        "cell": f"{seed}-{arm}",
                        "adam": cell["adam_calls"],
                        "restore": cell["restore"]["all_bitwise_equal"],
                        "continuation": {
                            key: block["all_bitwise_equal"]
                            for key, block in cell["continuation"].items()
                        },
                        "non_empty_positions": cell["non_empty_positions"],
                        "m_at_8": cell["empty_mid_contrast_at_8"]["m_nonzero"],
                        "state": cell["non_empty_state"],
                        "candidate_sha": cell["candidate"]["candidate_sha"][:12],
                    },
                    ensure_ascii=False,
                )
            )
            if not cell_passed(cell):
                failures.append(f"{seed}-{arm}: {failure_reason(cell)}")
    stop_reason = "; ".join(failures) or None

    # Discard the six step-1 bundles after the comparison; bytes stay registered.
    step1_entries = [entry for entry in artifacts if entry["kind"] == "routing_step1_bundle"]
    discard(step1_entries)

    identity_shas = sorted({cell["candidate"]["candidate_sha"] for cell in cells})
    if len(identity_shas) != 1:
        stop_reason = stop_reason or f"candidate_sha mismatch across cells: {identity_shas}"
    candidate_block = cells[0]["candidate"] if cells else None

    passed = (
        stop_reason is None and len(cells) == len(SEEDS) * len(ARMS) and len(identity_shas) == 1
    )
    adam_steps = sum(cell["adam_calls"] for cell in cells)
    report = {
        "schema": "taiji-selection-routing-17-4-preflight-v2",
        "step": "N17.4",
        "date": "2026-11-03",
        "candidate": candidate_block,
        "candidate_sha_consistent": bool(len(identity_shas) == 1),
        "passed": bool(passed),
        "stop_reason": stop_reason,
        "cells": cells,
        "totals": {
            "cells": len(SEEDS) * len(ARMS),
            "completed_cells": len(cells),
            "adam_steps": adam_steps,
            "adam_per_cell": ADAM_PER_CELL,
            "training_updates_per_cell": UPDATES_PREFLIGHT,
        },
        "limits": {
            "confirm_reads": CONFIRM_READS,
            "confirm_materialised": audit["confirm_materialised"],
            "confirm_examples_read": audit["confirm_examples_read"],
            "formal_training": False,
            "formal_updates": False,
            "selection_head_only": True,
        },
        "data": {
            "splits": audit["splits"],
            "examples": audit["examples"],
        },
        "dose": {
            "stage": "selection",
            "formal_updates": "1-100",
            "preflight_updates": UPDATES_PREFLIGHT,
            "batch": BATCH,
            "dtype": "float64",
            "optimizer": "Adam",
            "lr": 1e-3,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.0,
            "clip_norm": CLIP_NORM,
            "head_seed": "parent seed (28031/28032)",
            "arms": list(ARMS),
        },
        "timing": {
            "cpu_seconds": time.process_time() - STARTED_CPU,
            "wall_seconds": time.perf_counter() - STARTED_WALL,
        },
        "protection": {
            "parent_unchanged": all(cell["parent_unchanged"] for cell in cells),
            "nontrainable_unchanged": all(
                cell["protection"]["nontrainable_unchanged"] for cell in cells
            ),
            "binding_unchanged": all(cell["protection"]["binding_unchanged"] for cell in cells),
            "confirm_reads": CONFIRM_READS,
        },
        "artifacts": artifacts,
        "cleanup": {
            "deleted_files": sum(1 for entry in artifacts if entry.get("deleted")),
            "deleted_bytes": sum(entry["bytes"] for entry in artifacts if entry.get("deleted")),
            "step1_bundles_deleted": len(step1_entries),
        },
        "manifest": str(MANIFEST),
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "schema": "taiji-selection-routing-17-4-preflight-manifest-v2",
        "step": "N17.4",
        "date": "2026-11-03",
        "candidate_sha": candidate_block["candidate_sha"] if candidate_block else None,
        "parameter_counts": candidate_block["parameter_counts"] if candidate_block else None,
        "candidate_sha_consistent": bool(len(identity_shas) == 1),
        "supersedes": {
            "path": "reports/taiji_selection_routing_17_4_preflight_20261103.json",
            "status": "superseded",
            "reason": "before hard-point transport and mid-copy checkpoint",
        },
        "report": {
            "path": str(REPORT),
            "bytes": REPORT.stat().st_size,
            "sha256": sha256_file(REPORT),
        },
        "source_sha256": {path: sha256_file(ROOT / path) for path in SOURCE_FILES},
        "parent_bundles": [
            {
                "path": PARENTS[seed],
                "bytes": (ROOT / PARENTS[seed]).stat().st_size,
                "sha256": sha256_file(ROOT / PARENTS[seed]),
            }
            for seed in SEEDS
        ],
        "data": [
            {
                "path": path,
                "bytes": (ROOT / path).stat().st_size,
                "sha256": sha256_file(ROOT / path),
            }
            for path in (SELECT_PAYLOAD, DATA_PAYLOAD)
        ],
        "frozen_dose": {
            "stage": "selection",
            "updates": "1-100 formal; 2 in this preflight",
            "batch": BATCH,
            "dtype": "float64",
            "adam": {
                "lr": 1e-3,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.0,
                "clip_norm": CLIP_NORM,
            },
            "order_seeds": {"28031": 38031, "28032": 38032},
        },
        "adam_per_cell": ADAM_PER_CELL,
        "adam_total": adam_steps,
        "cells": [
            {
                "seed": cell["seed"],
                "arm": cell["arm"],
                "adam_calls": cell["adam_calls"],
                "step1_sha256": cell["step1_sha256"],
                "step1_bytes": cell["step1_bytes"],
                "restore_all_bitwise_equal": cell["restore"]["all_bitwise_equal"],
                "continuation_all_bitwise_equal": {
                    key: block["all_bitwise_equal"] for key, block in cell["continuation"].items()
                },
                "non_empty_mid_copy": cell["non_empty_mid_copy"],
            }
            for cell in cells
        ],
        "artifacts": [
            {
                "path": entry["path"],
                "kind": entry["kind"],
                "bytes": entry["bytes"],
                "sha256": entry["sha256"],
                "deleted": bool(entry.get("deleted")),
            }
            for entry in artifacts
        ],
        "deleted": {
            "files": sum(1 for entry in artifacts if entry.get("deleted")),
            "bytes": sum(entry["bytes"] for entry in artifacts if entry.get("deleted")),
        },
        "limits": {
            "confirm_reads": CONFIRM_READS,
            "formal_training": False,
        },
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": passed,
                "cells": len(cells),
                "adam_steps": adam_steps,
                "cpu_seconds": report["timing"]["cpu_seconds"],
                "wall_seconds": report["timing"]["wall_seconds"],
                "report": str(REPORT),
                "report_bytes": REPORT.stat().st_size,
                "manifest": str(MANIFEST),
                "manifest_bytes": MANIFEST.stat().st_size,
            },
            ensure_ascii=False,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
