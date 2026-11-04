"""18.1 content-transport probe: split the "whole-string is 0" reading.

Read-only, zero-update probe over a finished 18.1 bundle.  It reports the
content-byte (object 12 bytes + place 6 bytes) hit rate under four conditions on
the same dev slice:

* ``a_free``      - free decode, alpha/beta/gamma all from the model;
* ``b_transport`` - record forced to gold, pointer (beta) free, gold prefix fed
  back.  The first content byte of each entity is anchored at the gold span
  start ("gold start"), later steps are free;
* ``b_pure``      - the same mitigation with gamma forced to 0 (pure copy);
* ``c_gate``      - record forced to gold and *every* content position forced,
  gamma kept at the model value.

Attribution rule (pre-registered in the request):

* ``b_pure`` high and ``c_gate`` low -> the gate is the bottleneck;
* ``b_pure`` low -> the beta/transport pointer is the bottleneck;
* ``c_gate`` high while ``a_free`` fails -> the gate was open, the failure is
  elsewhere (template/selection).

The natural (unforced) step is the frozen ``routing_step``; the forced step
re-uses the same tail (content -> advance_h -> readout -> gamma -> probabilities)
so the only difference is the forced read and the gamma override.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

torch.set_num_threads(1)

from taiji import data_premise_18_0 as premise  # noqa: E402
from taiji.binding_copy import MAX_COVERAGE, BindingRead, advance_copy, observe_goal  # noqa: E402
from taiji.living_core import EOS  # noqa: E402
from taiji.record_relation_core import (  # noqa: E402
    _gate,
    _mode_vector,
    advance_h,
)
from taiji.selection_routing import (  # noqa: E402
    _advanced_state,
    _s_t,
    build_context,
    routing_generate,
    routing_step,
)
from taiji.selection_routing_checkpoint import load_routing_bundle  # noqa: E402

SEEDS = (28031, 28032)
ARMS = ("A_route_lock", "B_stop_lock", "C_unlocked")
OBJECT_POSITIONS = tuple(range(3, 15))
PLACE_POSITIONS = tuple(range(21, 27))
CONTENT_POSITIONS = OBJECT_POSITIONS + PLACE_POSITIONS
PER_WORLD = 4

OUTPUT_DIR = Path("output/taiji_selection_routing_18_1")
REPORT_TEMPLATE = "reports/taiji_content_transport_18_1_{checkpoint}_20261104.json"
MANIFEST_TEMPLATE = "plans/manifests/taiji_content_transport_18_1_{checkpoint}_20261104.json"
CHECKPOINTS = {"step150": 150, "step300": 300}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def dev_slice(dev: tuple[dict, ...]) -> tuple[dict, ...]:
    """Fixed 48-example slice: 12 worlds x (object 0/1 x place 0/1)."""
    by_world: dict[int, list[dict]] = {}
    for fixture in dev:
        by_world.setdefault(int(fixture["world"]), []).append(fixture)
    picked: list[dict] = []
    for world in sorted(by_world):
        block = by_world[world]
        picked.extend(block[index] for index in (0, 1, 4, 5) if index < len(block))
    return tuple(picked)


def _span(raw: bytes, name: bytes) -> int:
    index = raw.find(name)
    if index < 0:
        raise RuntimeError(f"entity {name!r} not found in record {raw!r}")
    return index


def _place_record(fixture: dict) -> int:
    for index, record in enumerate(fixture["raw_example"]["records"]):
        if record["kind"] == "place" and record["description"] == fixture["place_description"]:
            return index
    raise RuntimeError("place record not found")


def _one_hot(length: int, index: int) -> torch.Tensor:
    vector = torch.zeros(length)
    vector[index] = 1.0
    return vector


def content_positions(fixture: dict) -> dict[int, tuple[int, int]]:
    """target position -> (record index, byte offset inside that record)."""
    object_record = int(fixture["selection_label"])
    place_record = _place_record(fixture)
    records = fixture["raw_example"]["records"]
    object_start = _span(
        records[object_record]["raw"].encode("utf-8"), fixture["object_name"].encode("utf-8")
    )
    place_start = _span(
        records[place_record]["raw"].encode("utf-8"), fixture["place_name"].encode("utf-8")
    )
    mapping = {}
    for offset, position in enumerate(OBJECT_POSITIONS):
        mapping[position] = (object_record, object_start + offset)
    for offset, position in enumerate(PLACE_POSITIONS):
        mapping[position] = (place_record, place_start + offset)
    return mapping


class ProbeContext:
    def __init__(self, candidate) -> None:
        self.candidate = candidate
        self.subject = candidate.subject
        self.base = None
        self.view = None

    def prepare(self, fixture: dict) -> tuple[Any, Any]:
        goal = fixture["goal"].encode("utf-8")
        prepared, context = None, None
        from taiji.binding_copy import initial_binding_state, prepare

        state = initial_binding_state(
            self.candidate.binding, f"probe-{fixture['world']}-{fixture['index']}"
        )
        prepared = prepare(self.candidate.binding, state, goal, fixture["records"])
        context = build_context(self.candidate.binding, goal, tuple(fixture["record_keys"]))
        return prepared, context

    def _tail(self, prepared, state, previous, read, gamma_override, gate_index):
        base = prepared.base
        view = base.view
        subject = self.subject
        previous_embedding = subject.encoder.embedding(
            torch.tensor(previous, device=subject.W.device)
        )
        q_bytes = read.q_bytes
        q_null = read.q_null
        inherited = self.candidate.reader.inherited
        units = torch.cat(view.units) if view.record_count else torch.zeros(0)
        content = (
            q_bytes @ inherited.V_position(units) + q_null * inherited.null_value
            if len(q_bytes)
            else q_null * inherited.null_value
        )
        event = (
            torch.zeros_like(content)
            if previous == EOS
            else subject.encoder.event(
                self.candidate.base.relation.encode_units(subject, bytes([previous])).mean(0)
            )
        )
        next_h = advance_h(
            subject,
            state.living.h,
            state.living.fast,
            event,
            base.r_goal,
            base.nu,
            content,
            _gate(self.candidate, gate_index),
        )
        xi = subject.xi(
            torch.cat([base.r_goal, previous_embedding, _mode_vector(subject.W, 0), base.nu])
        )
        z = subject.readout(next_h, xi)[0]
        generated = torch.softmax(
            subject.expression.generate(torch.cat([z, previous_embedding])), dim=0
        )
        gamma_raw = torch.sigmoid(
            subject.expression.mix(torch.cat([z, content, previous_embedding]))
        )
        head = self.candidate.head
        floor = head.config.gamma_floor
        gamma_eff = floor + (1.0 - floor) * gamma_raw
        if gamma_override is not None:
            gamma_eff = torch.as_tensor(gamma_override, dtype=gamma_eff.dtype)
        copy_byte_mass = torch.zeros(MAX_COVERAGE, dtype=torch.float64)
        if len(q_bytes):
            copy_byte_mass = copy_byte_mass.index_add(0, view.byte_ids, q_bytes)
        probabilities = (gamma_eff + (1.0 - gamma_eff) * q_null) * generated + (
            1.0 - gamma_eff
        ) * copy_byte_mass
        return probabilities, gamma_eff, content, next_h, q_bytes, q_null, read

    def natural_step(self, prepared, state, previous, context, gate_index, pointer_hard):
        return routing_step(
            self.candidate,
            prepared,
            state,
            previous,
            context,
            self.candidate.head,
            arm=self.arm,
            gate_index=gate_index,
            span_head=self.candidate.span_head,
            pointer_temperature=1.0,
            pointer_hard=pointer_hard,
        )

    def read_natural(self, prepared, state, previous):
        base = prepared.base
        view = base.view
        subject = self.subject
        previous_embedding = subject.encoder.embedding(
            torch.tensor(previous, device=subject.W.device)
        )
        s_t = _s_t(self.candidate, prepared, state, previous)
        bias = None
        if self.candidate.span_head is not None and view.record_count:
            bias = tuple(
                self.candidate.span_head.start(view.units[index]).squeeze(-1)
                for index in range(view.record_count)
            )
        return self.candidate.reader.read(
            s_t,
            previous_embedding,
            base.goal_units,
            view,
            state.m,
            pointer_bias=bias,
            pointer_temperature=1.0,
            pointer_hard=True,
        )

    def forced_step(
        self,
        prepared,
        state,
        previous,
        gate_index,
        force,
        m_override,
        gamma_override,
    ):
        base = prepared.base
        view = base.view
        subject = self.subject
        previous_embedding = subject.encoder.embedding(
            torch.tensor(previous, device=subject.W.device)
        )
        s_t = _s_t(self.candidate, prepared, state, previous)
        read = self.candidate.reader.read(
            s_t,
            previous_embedding,
            base.goal_units,
            view,
            state.m,
            force=force,
            m_override=m_override,
            pointer_temperature=1.0,
            pointer_hard=True,
        )
        return self._tail(prepared, state, previous, read, gamma_override, gate_index)


def advance_custom(
    candidate,
    prepared,
    state,
    read,
    probabilities,
    gamma_eff,
    accepted,
    content,
    next_h,
    q_bytes,
    q_null,
):
    view = prepared.base.view
    copy_byte_mass = torch.zeros(MAX_COVERAGE, dtype=torch.float64)
    if len(q_bytes):
        copy_byte_mass = copy_byte_mass.index_add(0, view.byte_ids, q_bytes)
    locked = BindingRead(
        content=content,
        copy_weights=q_bytes,
        copy_byte_mass=copy_byte_mass,
        alpha=read.alpha,
        betas=read.betas,
        alpha_null=q_null,
        record_mass=read.record_mass,
        s_bytes=read.s_bytes,
        q_bytes=q_bytes,
        q_null=q_null,
        m_mass=read.m_mass,
        continuation=read.continuation,
        forced=False,
    )
    next_m, next_coverage = advance_copy(
        candidate.reader, state, locked, probabilities, gamma_eff, accepted, view
    )
    from dataclasses import replace

    living = replace(state.living, h=next_h)
    return replace(state, living=living, m=next_m, coverage=next_coverage)


def content_stats(raw: list[int], target: tuple[int, ...]) -> dict:
    hits = 0
    first = None
    for position in CONTENT_POSITIONS:
        got = raw[position] if position < len(raw) else None
        if got == target[position]:
            hits += 1
        elif first is None:
            first = position
    if first is None:
        category = "none"
    elif first == 3:
        category = "object_first_byte"
    elif first == 21:
        category = "place_first_byte"
    elif first < 15:
        category = "object_later_byte"
    else:
        category = "place_later_byte"
    return {
        "hits": hits,
        "denominator": len(CONTENT_POSITIONS),
        "first_divergence": first,
        "first_divergence_category": category,
    }


def free_decode(probe: ProbeContext, fixture: dict) -> dict:
    prepared, context = probe.prepare(fixture)
    generated = routing_generate(
        probe.candidate,
        prepared,
        context,
        probe.candidate.head,
        arm=probe.arm,
        span_head=probe.candidate.span_head,
        pointer_temperature=1.0,
        pointer_hard=True,
    )
    raw = list(generated["raw_bytes"])
    target = tuple(fixture["target_bytes"])
    stats = content_stats(raw, target)
    gammas = [
        generated["gamma"][position]
        for position in CONTENT_POSITIONS
        if position < len(generated["gamma"])
    ]
    stats.update(
        {
            "completed": bool(generated["terminated"]),
            "generated_bytes": len(raw),
            "gamma_mean": statistics.fmean(gammas) if gammas else 0.0,
            "raw": raw[:27],
        }
    )
    return stats


def forced_condition(probe: ProbeContext, fixture: dict, mode: str, gamma_override) -> dict:
    """Teacher-forced replay with gold input prefix.

    ``mode`` is ``"b"`` (gold record, free pointer, gold start anchored at the
    first content byte) or ``"c"`` (gold record, every content position forced).
    """
    prepared, context = probe.prepare(fixture)
    state, gate_index = observe_goal(probe.candidate, prepared, gate_index=0)
    target = tuple(fixture["target_bytes"])
    mapping = content_positions(fixture)
    record_count = len(fixture["record_keys"])
    object_record = int(fixture["selection_label"])
    place_record = _place_record(fixture)
    probabilities: list[float] = []
    gammas: list[float] = []
    hits = 0
    first = None
    raw: list[int] = []
    previous = EOS
    with torch.no_grad():
        for position in range(len(target)):
            if position in mapping:
                record, offset = mapping[position]
                alpha = _one_hot(record_count + 1, record)
                if mode == "c":
                    lengths = [
                        int(prepared.base.view.units[index].shape[0])
                        for index in range(record_count)
                    ]
                    betas = tuple(
                        (
                            _one_hot(lengths[index], offset if index == record else 0)
                            if index == record
                            else torch.zeros(lengths[index])
                        )
                        for index in range(record_count)
                    )
                    pointer = tuple(
                        (
                            _one_hot(lengths[index], offset if index == record else 0)
                            if index == record
                            else torch.zeros(lengths[index])
                        )
                        for index in range(record_count)
                    )
                    force = {"alpha": alpha, "betas": betas, "pointer": pointer}
                    m_override = torch.zeros(len(prepared.base.view.byte_ids))
                    m_override[prepared.base.view.record_offsets[record][0] + offset] = 1.0
                    out = probe.forced_step(
                        prepared, state, previous, gate_index, force, m_override, gamma_override
                    )
                else:
                    natural = probe.read_natural(prepared, state, previous)
                    force = {"alpha": alpha, "betas": natural.betas}
                    m_override = None
                    if position in (3, 21):
                        start_record = object_record if position == 3 else place_record
                        start_offset = mapping[position][1]
                        m_override = torch.zeros(len(prepared.base.view.byte_ids))
                        m_override[
                            prepared.base.view.record_offsets[start_record][0] + start_offset
                        ] = 1.0
                    out = probe.forced_step(
                        prepared, state, previous, gate_index, force, m_override, gamma_override
                    )
                step_probabilities, gamma_eff, content, next_h, q_bytes, q_null, read = out
                byte = int(torch.argmax(step_probabilities).item())
                probabilities.append(float(step_probabilities[target[position]]))
                gammas.append(float(gamma_eff.detach()))
                if byte == target[position]:
                    hits += 1
                elif first is None:
                    first = position
                accepted = target[position]
                state = advance_custom(
                    probe.candidate,
                    prepared,
                    state,
                    read,
                    step_probabilities,
                    gamma_eff,
                    accepted,
                    content,
                    next_h,
                    q_bytes,
                    q_null,
                )
            else:
                step = probe.natural_step(prepared, state, previous, context, gate_index, True)
                byte = int(torch.argmax(step.probabilities).item())
                accepted = target[position]
                state, _coverage, gate_index = _advanced_state(
                    probe.candidate, prepared, step, accepted
                )
            raw.append(byte)
            previous = accepted
    if first is None:
        category = "none"
    elif first == 3:
        category = "object_first_byte"
    elif first == 21:
        category = "place_first_byte"
    elif first < 15:
        category = "object_later_byte"
    else:
        category = "place_later_byte"
    return {
        "hits": hits,
        "denominator": len(CONTENT_POSITIONS),
        "rate": hits / len(CONTENT_POSITIONS),
        "first_divergence": first,
        "first_divergence_category": category,
        "p_gold_mean": statistics.fmean(probabilities) if probabilities else 0.0,
        "gamma_mean": statistics.fmean(gammas) if gammas else 0.0,
        "raw": raw[:27],
    }


def _record_length(prepared, index: int) -> int:
    return int(prepared.base.view.units[index].shape[0])


def run_cell(seed: int, arm: str, checkpoint: str, limit: int | None = None) -> dict:
    start_cpu = time.process_time()
    start_wall = time.perf_counter()
    path = OUTPUT_DIR / f"{seed}-{arm}-{checkpoint}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    candidate, _optimizer, _state, update, _payload = load_routing_bundle(path)
    probe = ProbeContext(candidate)
    probe.arm = arm
    payload = premise.build_payload()
    dev = dev_slice(premise.fixtures(payload, "dev"))
    if limit is not None:
        dev = dev[:limit]
    rows = []
    for fixture in dev:
        free = free_decode(probe, fixture)
        b = forced_condition(probe, fixture, "b", None)
        b_pure = forced_condition(probe, fixture, "b", 0.0)
        c = forced_condition(probe, fixture, "c", None)
        rows.append(
            {
                "world": fixture["world"],
                "index": fixture["index"],
                "a_free": free,
                "b_transport": b,
                "b_pure": b_pure,
                "c_gate": c,
            }
        )

    def summarize(key: str) -> dict:
        entries = [row[key] for row in rows]
        hits = sum(entry["hits"] for entry in entries)
        denominators = sum(entry["denominator"] for entry in entries)
        categories: dict[str, int] = {}
        for entry in entries:
            categories[entry["first_divergence_category"]] = (
                categories.get(entry["first_divergence_category"], 0) + 1
            )
        summary = {
            "content_byte_hits": hits,
            "content_byte_denominator": denominators,
            "content_byte_rate": hits / denominators if denominators else 0.0,
            "first_divergence_categories": categories,
            "examples": len(entries),
        }
        if "gamma_mean" in entries[0]:
            summary["gamma_mean"] = statistics.fmean(entry["gamma_mean"] for entry in entries)
        if "p_gold_mean" in entries[0]:
            summary["p_gold_mean"] = statistics.fmean(entry["p_gold_mean"] for entry in entries)
        if key == "a_free":
            summary["completed_examples"] = sum(1 for entry in entries if entry["completed"])
            summary["generated_bytes_mean"] = statistics.fmean(
                entry["generated_bytes"] for entry in entries
            )
        return summary

    a = summarize("a_free")
    b = summarize("b_transport")
    b_pure = summarize("b_pure")
    c = summarize("c_gate")
    transport_ok = b_pure["content_byte_rate"] >= 0.90
    diagnosis = {
        "transport_ok": transport_ok,
        "gate_bottleneck": bool(transport_ok and c["content_byte_rate"] < 0.50),
        "pointer_bottleneck": bool(not transport_ok),
        "c_gate_hit_rate": c["content_byte_rate"],
        "b_pure_hit_rate": b_pure["content_byte_rate"],
    }
    if transport_ok and c["content_byte_rate"] < 0.50:
        diagnosis["layer"] = "gamma_gate"
    elif not transport_ok:
        diagnosis["layer"] = "beta_transport_pointer"
    elif c["content_byte_rate"] >= 0.50 and a["content_byte_rate"] < 0.50:
        diagnosis["layer"] = "template_or_selection"
    else:
        diagnosis["layer"] = "undetermined"
    return {
        "seed": seed,
        "arm": arm,
        "checkpoint": checkpoint,
        "bundle": {"path": str(path), "bytes": path.stat().st_size},
        "update": update,
        "examples": len(rows),
        "a_free": a,
        "b_transport": b,
        "b_pure": b_pure,
        "c_gate": c,
        "diagnosis": diagnosis,
        "rows": rows,
        "cost": {
            "cpu_seconds": time.process_time() - start_cpu,
            "wall_seconds": time.perf_counter() - start_wall,
            "adam_steps": 0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", choices=sorted(CHECKPOINTS), required=True)
    parser.add_argument("--cells", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    checkpoint = arguments.checkpoint
    cells = [(seed, arm) for seed in SEEDS for arm in ARMS]
    if arguments.cells:
        cells = [tuple(value.split(":")) for value in arguments.cells]
        cells = [(int(seed), arm) for seed, arm in cells]

    results = []
    failures = []
    with ProcessPoolExecutor(max_workers=6) as pool:
        futures = {
            pool.submit(run_cell, seed, arm, checkpoint, arguments.limit): (seed, arm)
            for seed, arm in cells
        }
        for future in as_completed(futures):
            seed, arm = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(
                    json.dumps(
                        {
                            "cell": f"{seed}-{arm}",
                            "a": round(result["a_free"]["content_byte_rate"], 3),
                            "b": round(result["b_transport"]["content_byte_rate"], 3),
                            "b0": round(result["b_pure"]["content_byte_rate"], 3),
                            "c": round(result["c_gate"]["content_byte_rate"], 3),
                            "layer": result["diagnosis"]["layer"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            except Exception:  # noqa: BLE001
                failures.append({"seed": seed, "arm": arm, "traceback": traceback.format_exc()})
    if failures:
        write_json(OUTPUT_DIR / f"probe_failures_{checkpoint}.json", {"failures": failures})
        return 1
    results.sort(key=lambda item: (item["seed"], item["arm"]))
    report = {
        "schema": "taiji-content-transport-probe-18-1-v1",
        "step": "N18.1",
        "checkpoint": checkpoint,
        "update": CHECKPOINTS[checkpoint],
        "date": "2026-11-04",
        "dtype": "float64",
        "threads": 1,
        "data": {
            "file": "output/taiji_data_premise_18_0/data.json",
            "dev_examples_used": len(results[0]["rows"]) if results else 0,
            "slice_rule": "12 dev worlds x (object 0/1 x place 0/1) = 48 examples",
            "content_bytes_per_example": len(CONTENT_POSITIONS),
        },
        "conditions": {
            "a_free": "free decode; alpha/beta/gamma from the model",
            "b_transport": "gold record forced, first content byte anchored at the gold span start, pointer free afterwards, gold input prefix",
            "b_pure": "b_transport with gamma forced to 0 (pure copy)",
            "c_gate": "gold record forced and every content position forced; gamma keeps the model value",
        },
        "attribution_rule": "b_pure high + c_gate low => gamma gate; b_pure low => beta/transport pointer; c_gate high + a_free low => template/selection",
        "cells": {f"{item['seed']}-{item['arm']}": item for item in results},
        "summary": {
            "a_free_rate": statistics.fmean(
                item["a_free"]["content_byte_rate"] for item in results
            ),
            "b_transport_rate": statistics.fmean(
                item["b_transport"]["content_byte_rate"] for item in results
            ),
            "b_pure_rate": statistics.fmean(
                item["b_pure"]["content_byte_rate"] for item in results
            ),
            "c_gate_rate": statistics.fmean(
                item["c_gate"]["content_byte_rate"] for item in results
            ),
        },
        "limits": {"zero_updates": True, "read_only": True, "confirm_reads": 0},
    }
    if arguments.smoke:
        write_json(OUTPUT_DIR / f"probe_smoke_{checkpoint}.json", report)
        print(json.dumps({"smoke": True, "summary": report["summary"]}, ensure_ascii=False))
        return 0
    write_json(ROOT / REPORT_TEMPLATE.format(checkpoint=checkpoint), report)
    report_path = ROOT / REPORT_TEMPLATE.format(checkpoint=checkpoint)
    import hashlib

    write_json(
        ROOT / MANIFEST_TEMPLATE.format(checkpoint=checkpoint),
        {
            "schema": "taiji-content-transport-probe-18-1-manifest-v1",
            "step": "N18.1",
            "checkpoint": checkpoint,
            "report": str(report_path),
            "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            "report_bytes": report_path.stat().st_size,
            "bundles": {key: item["bundle"] for key, item in report["cells"].items()},
            "summary": report["summary"],
            "limits": report["limits"],
        },
    )
    print(
        json.dumps({"report": str(report_path), "summary": report["summary"]}, ensure_ascii=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
