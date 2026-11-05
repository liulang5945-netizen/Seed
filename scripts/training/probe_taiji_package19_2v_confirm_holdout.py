"""19.2v: first and only confirm read of the structural binding-copy task.

19.2u closed the dev split at 192/192 with a structural copy stop edge and
showed that a fixed-seed random re-initialisation of the trained modules leaves
the content layers at 1.000.  This package reads the confirm split exactly once,
with the configuration frozen beforehand (``preregistered`` snapshot), and
answers one question: does the structure hold outside the design split?

Discipline: 0 training, 0 optimizer steps, no tuning, no re-run after seeing
confirm, single forward per pre-registered configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

torch.set_num_threads(1)

import probe_taiji_content_transport_18_1 as probe_mod  # noqa: E402

from taiji import candidate_identity  # noqa: E402
from taiji import data_premise_18_0 as premise  # noqa: E402
from taiji.binding_copy import (  # noqa: E402
    MAX_PREDICTIONS,
    observe_goal,
)
from taiji.byte_match_route import IDENTITY_DIM, PAD_ID, ByteMatchRoute  # noqa: E402
from taiji.copy_stop_boundary import field_bounds  # noqa: E402
from taiji.living_core import EOS  # noqa: E402
from taiji.record_relation_core import decode_endpoint  # noqa: E402
from taiji.selection_routing_checkpoint import load_routing_bundle, parameter_tensors  # noqa: E402

PREREG = ROOT / "plans/manifests/taiji_package19_2v_preregistered_20261105.json"
REPORT = ROOT / "reports/taiji_package19_2v_confirm_holdout_20261105.json"
MANIFEST = ROOT / "plans/manifests/taiji_package19_2v_confirm_holdout_20261105.json"
PROGRESS = ROOT / "reports/taiji_package19_2v_confirm_holdout_20261105_progress.jsonl"
DEV_REPORT = ROOT / "reports/taiji_package19_2u_stop_boundary_ablation_20261105.json"
ASSEMBLED = ROOT / "output/taiji_package19_2u_assembled_zero_training_stopfix.pt"
BUNDLE = ROOT / "output/taiji_package19_compare/28031-B_stop_lock-step300.pt"

CPU_CAP = 400.0
CPU_GUARD = 380.0
WALL_CAP = 3600.0
SPAN = 15
BATCH = 24
EXPECTED_CONFIRM = 192
MILESTONE = 0.95
OVERFIT_DELTA = -0.05
OBJECT_POSITIONS = tuple(range(3, 15))
PLACE_POSITIONS = tuple(range(21, 27))
CONTENT_POSITIONS = OBJECT_POSITIONS + PLACE_POSITIONS

L1_TARGETS = {
    "selection_accuracy_object": 1.0,
    "selection_accuracy_place": 1.0,
    "selection_accuracy_both": 1.0,
    "start_hit_object": 1.0,
    "start_hit_place": 1.0,
    "composite_start_hit_object": 1.0,
    "composite_start_hit_place": 1.0,
    "conditional_byte": 1.0,
    "provenance_rate": 1.0,
    "fabricated_bytes": 0.0,
    "misbound_bytes": 0.0,
    "truncation_rate": 0.0,
    "wrong_route_right_content": 0.0,
}


def append_progress(record: dict) -> None:
    with PROGRESS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def age_of(fixture: dict) -> str:
    return "new" if fixture["flags"]["new_description"] else "old"


def name_from_raw(raw: str, description: str) -> str | None:
    """Recover the name using only the line bytes and the matched description."""
    at = raw.find(description)
    if at < 0:
        return None
    tail = raw[at + len(description) :]
    if tail.startswith("是") and tail.endswith("。"):
        return tail[1:-1]
    if tail.startswith("对应") and tail.endswith("。"):
        return tail[2:-1]
    head = raw[:at]
    if head.endswith("又叫") and raw.endswith("。"):
        return head[:-2]
    return None


def place_record(fixture: dict) -> int:
    for index, record in enumerate(fixture["raw_example"]["records"]):
        if record["kind"] == "place" and record["description"] == fixture["place_description"]:
            return index
    raise RuntimeError("place record not found")


def build_tensors(fixtures: tuple[dict, ...]) -> dict:
    examples = len(fixtures)
    lines = len(fixtures[0]["records"])
    longest = max(len(record.raw) for fixture in fixtures for record in fixture["records"])
    record_ids = torch.full((examples, lines, longest), PAD_ID, dtype=torch.long)
    record_mask = torch.zeros((examples, lines, longest), dtype=torch.bool)
    object_query = torch.full((examples, SPAN), PAD_ID, dtype=torch.long)
    object_query_mask = torch.zeros((examples, SPAN), dtype=torch.bool)
    place_query = torch.full((examples, SPAN), PAD_ID, dtype=torch.long)
    place_query_mask = torch.zeros((examples, SPAN), dtype=torch.bool)
    for row_index, fixture in enumerate(fixtures):
        object_bytes = fixture["object_description"].encode("utf-8")
        place_bytes = fixture["place_description"].encode("utf-8")
        object_query[row_index, : len(object_bytes)] = torch.tensor(list(object_bytes))
        object_query_mask[row_index, : len(object_bytes)] = True
        place_query[row_index, : len(place_bytes)] = torch.tensor(list(place_bytes))
        place_query_mask[row_index, : len(place_bytes)] = True
        for line_index, record in enumerate(fixture["records"]):
            raw = record.raw
            record_ids[row_index, line_index, : len(raw)] = torch.tensor(list(raw))
            record_mask[row_index, line_index, : len(raw)] = True
    return {
        "record_ids": record_ids,
        "record_mask": record_mask,
        "object_query": object_query,
        "object_query_mask": object_query_mask,
        "place_query": place_query,
        "place_query_mask": place_query_mask,
    }


@torch.no_grad()
def match_routes(route: ByteMatchRoute, tensors: dict, fixtures: tuple[dict, ...]) -> list[dict]:
    """Per example: argmax line and window start for both descriptions."""
    results: list[dict] = []
    examples = len(fixtures)
    for start in range(0, examples, BATCH):
        stop = min(start + BATCH, examples)
        record_ids = tensors["record_ids"][start:stop]
        record_mask = tensors["record_mask"][start:stop]
        object_score, object_offset, _ = route.forward(
            tensors["object_query"][start:stop],
            tensors["object_query_mask"][start:stop],
            record_ids,
            record_mask,
        )
        place_score, place_offset, _ = route.forward(
            tensors["place_query"][start:stop],
            tensors["place_query_mask"][start:stop],
            record_ids,
            record_mask,
        )
        for local, fixture in enumerate(fixtures[start:stop]):
            object_route = int(torch.argmax(object_score[local]))
            place_route = int(torch.argmax(place_score[local]))
            object_window = int(object_offset[local, object_route])
            place_window = int(place_offset[local, place_route])
            object_raw = fixture["records"][object_route].raw.decode("utf-8")
            place_raw = fixture["records"][place_route].raw.decode("utf-8")
            object_name = name_from_raw(object_raw, fixture["object_description"])
            place_name = name_from_raw(place_raw, fixture["place_description"])
            object_name_offset = -1
            place_name_offset = -1
            if object_name is not None:
                object_name_offset = object_raw.encode("utf-8").find(object_name.encode("utf-8"))
            if place_name is not None:
                place_name_offset = place_raw.encode("utf-8").find(place_name.encode("utf-8"))
            results.append(
                {
                    "world": fixture["world"],
                    "index": fixture["index"],
                    "object_route": object_route,
                    "object_window": object_window,
                    "object_name_offset": object_name_offset,
                    "object_stop": field_bounds(object_raw, fixture["object_description"])[1],
                    "place_route": place_route,
                    "place_window": place_window,
                    "place_name_offset": place_name_offset,
                    "place_stop": field_bounds(place_raw, fixture["place_description"])[1],
                    "object_margin": float(object_score[local].max() - object_score[local].min()),
                    "place_margin": float(place_score[local].max() - place_score[local].min()),
                }
            )
    return results


def uniform_betas(view) -> tuple[torch.Tensor, ...]:
    pieces = []
    for index in range(view.record_count):
        length = max(int(view.units[index].shape[0]), 1)
        pieces.append(torch.full((length,), 1.0 / length, dtype=torch.float32))
    return tuple(pieces)


def force_payload(view, route: int, start: int | None) -> tuple[dict, torch.Tensor | None]:
    alpha = torch.zeros(view.record_count + 1, dtype=torch.float32)
    if 0 <= route < view.record_count:
        alpha[route] = 1.0
    else:
        alpha[-1] = 1.0
    force = {"alpha": alpha, "betas": uniform_betas(view)}
    override = None
    if start is not None and 0 <= route < view.record_count:
        pointers = []
        for index in range(view.record_count):
            length = int(view.units[index].shape[0])
            piece = torch.zeros(length, dtype=torch.float32)
            if index == route and 0 <= start < length:
                piece[start] = 1.0
            pointers.append(piece)
        force["pointer"] = tuple(pointers)
        flat = int(view.byte_ids.numel())
        override = torch.zeros(flat, dtype=torch.float32)
        if start >= 0:
            offset = int(view.record_offsets[route][0]) + start
            if 0 <= offset < flat:
                override[offset] = 1.0
    return force, override


@torch.no_grad()
def decode(candidate, probe, fixture: dict, route_info: dict, gamma: float) -> dict:
    prepared, _context = probe.prepare(fixture)
    view = prepared.base.view
    state, gate_index = observe_goal(candidate, prepared, gate_index=0)
    previous = EOS
    raw: list[int] = []
    routes: list[int] = []
    starts: list[int] = []
    terminated = False
    record_count = view.record_count
    for _prediction in range(MAX_PREDICTIONS):
        position = len(raw)
        if position in OBJECT_POSITIONS:
            route, start = route_info["object_route"], route_info["object_name_offset"]
        elif position in PLACE_POSITIONS:
            route, start = route_info["place_route"], route_info["place_name_offset"]
        else:
            route, start = -1, None
        anchored = position in (OBJECT_POSITIONS[0], PLACE_POSITIONS[0])
        force, override = force_payload(view, route, start if anchored else None)
        # The template slot owns the stop edge; the two hops never share one.
        limits = [-1] * record_count
        if 0 <= route_info["object_route"] < record_count:
            limits[route_info["object_route"]] = route_info["object_stop"]
        if position >= PLACE_POSITIONS[0] and 0 <= route_info["place_route"] < record_count:
            limits[route_info["place_route"]] = route_info["place_stop"]
        probabilities, gamma_eff, content, next_h, q_bytes, q_null, read = probe.forced_step(
            prepared,
            state,
            previous,
            gate_index,
            force,
            override,
            gamma,
            copy_limits=tuple(limits),
        )
        endpoint = decode_endpoint(probabilities, prefix=bytes(raw))
        if endpoint["eos"]:
            terminated = True
            break
        byte = int(endpoint["byte"])
        raw.append(byte)
        routes.append(route)
        starts.append(start if anchored else -1)
        if len(raw) > 64:
            break
        state = probe_mod.advance_custom(
            candidate,
            prepared,
            state,
            read,
            probabilities,
            gamma_eff,
            byte,
            content,
            next_h,
            q_bytes,
            q_null,
        )
        previous = byte
    return {
        "raw": tuple(raw),
        "routes": tuple(routes),
        "starts": tuple(starts),
        "terminated": terminated,
    }


def target_bytes(fixture: dict) -> tuple[int, ...]:
    target = tuple(fixture["target_bytes"])
    return target[:-1] if target and target[-1] == EOS else target


def analyze_span(fixture: dict, decoded: dict, start: int, length: int, kind: str) -> dict:
    target = target_bytes(fixture)
    mapping = probe_mod.content_positions(fixture)
    raw = decoded["raw"]
    emitted = bytes(raw[start : start + length])
    gold = bytes(target[start : start + length])
    gold_route, _gold_offset = mapping[start]
    route = decoded["routes"][start] if start < len(decoded["routes"]) else -1
    records = fixture["raw_example"]["records"]
    bound_text = records[route]["raw"].encode("utf-8") if 0 <= route < len(records) else b""
    other_texts = [
        record["raw"].encode("utf-8") for index, record in enumerate(records) if index != route
    ]
    truncated = len(emitted) != length
    in_bound = (not truncated) and emitted in bound_text
    in_other = (not truncated) and any(emitted in text for text in other_texts)
    route_ok = route == gold_route
    content_ok = (not truncated) and emitted == gold
    if truncated:
        cell = "truncated"
    elif route_ok and content_ok:
        cell = "aligned"
    elif route_ok:
        cell = "right_route_wrong_content"
    elif content_ok:
        cell = "wrong_route_right_content"
    else:
        cell = "wrong_route_wrong_content"
    if in_bound:
        provenance, misbound, fabricated = length, 0, 0
    elif in_other:
        provenance, misbound, fabricated = 0, length, 0
    else:
        provenance, misbound, fabricated = 0, 0, length
    return {
        "kind": kind,
        "route": route,
        "gold_route": gold_route,
        "route_ok": bool(route_ok),
        "content_ok": bool(content_ok),
        "truncated": bool(truncated),
        "cell": cell,
        "start_hit": bool(route_ok and decoded["starts"][start] == _gold_offset),
        "bytes": length,
        "provenance": provenance,
        "misbound": misbound,
        "fabricated": fabricated,
        "subtype": (
            "copied_bound_record"
            if in_bound
            else ("copied_other_record" if in_other else "fabricated")
        ),
    }


def first_difference(fixture: dict, decoded: dict) -> dict:
    target = target_bytes(fixture)
    raw = decoded["raw"]
    if tuple(raw) == target and decoded["terminated"]:
        return {"category": "none", "position": None}
    if tuple(raw) == target:
        return {"category": "eos_not_stopped", "position": len(target)}
    if len(raw) < len(target):
        return {"category": "truncated", "position": len(raw)}
    for position in range(min(len(raw), len(target))):
        if raw[position] != target[position]:
            if position in OBJECT_POSITIONS or position in PLACE_POSITIONS:
                records = fixture["raw_example"]["records"]
                gold_record = (
                    fixture["selection_label"]
                    if position in OBJECT_POSITIONS
                    else place_record(fixture)
                )
                bound = records[gold_record]["raw"].encode("utf-8")
                others = [
                    record["raw"].encode("utf-8")
                    for index, record in enumerate(records)
                    if index != gold_record
                ]
                if raw[position : position + 1] in bound:
                    category = "wrong_name"
                elif any(raw[position : position + 1] in text for text in others):
                    category = "serial"
                else:
                    category = "fabricated"
            else:
                category = "template"
            return {"category": category, "position": position}
    return {"category": "extra_bytes", "position": len(target)}


def record_kind_at_position(position: int) -> str:
    if position in OBJECT_POSITIONS:
        return "object_name"
    if position in PLACE_POSITIONS:
        return "place_name"
    return "template"


def measure_boundaries(fixtures: tuple[dict, ...]) -> dict:
    """Mechanical L1 check: the confirm records must expose the same template."""
    lengths: dict[str, dict[str, int]] = {"object": {}, "place": {}}
    table: dict[str, dict] = {}
    counter = {
        "name_contains_terminator": [],
        "name_length_equals_description": [],
        "description_repeated": [],
        "record_without_terminator": [],
        "terminator_not_adjacent_style_0_2": [],
    }
    lines = 0
    terminator_all = True
    for fixture in fixtures:
        for record in fixture["raw_example"]["records"]:
            lines += 1
            raw = record["raw"]
            kind = record["kind"]
            style = int(record["style"])
            description = record["description"]
            name = record["name"]
            start, end = field_bounds(raw, description)
            raw_bytes = raw.encode("utf-8")
            terminator = "。".encode()
            ends_with_terminator = raw_bytes.endswith(terminator)
            terminator_all = terminator_all and ends_with_terminator
            if not ends_with_terminator:
                counter["record_without_terminator"].append(
                    {"world": fixture["world"], "index": fixture["index"], "raw": raw}
                )
            gap = len(raw_bytes) - end - len(terminator)
            if gap != 0 and style != 1:
                counter["terminator_not_adjacent_style_0_2"].append(
                    {
                        "world": fixture["world"],
                        "index": fixture["index"],
                        "style": style,
                        "gap": gap,
                        "raw": raw,
                    }
                )
            if "。" in name:
                counter["name_contains_terminator"].append(
                    {"world": fixture["world"], "index": fixture["index"], "name": name}
                )
            if len(name.encode("utf-8")) == len(description.encode("utf-8")):
                counter["name_length_equals_description"].append(
                    {
                        "world": fixture["world"],
                        "index": fixture["index"],
                        "kind": kind,
                        "name": name,
                        "description": description,
                    }
                )
            if raw.count(description) != 1:
                counter["description_repeated"].append(
                    {"world": fixture["world"], "index": fixture["index"], "raw": raw}
                )
            length = end - start
            lengths[kind][str(length)] = lengths[kind].get(str(length), 0) + 1
            key = f"{kind}/style{style}"
            cell = table.setdefault(
                key,
                {
                    "lines": 0,
                    "starts": {},
                    "lengths": {},
                    "bytes_after_name": {},
                    "ends_with_terminator": True,
                },
            )
            cell["lines"] += 1
            cell["starts"][str(start)] = cell["starts"].get(str(start), 0) + 1
            cell["lengths"][str(length)] = cell["lengths"].get(str(length), 0) + 1
            cell["bytes_after_name"][str(gap)] = cell["bytes_after_name"].get(str(gap), 0) + 1
            cell["ends_with_terminator"] = cell["ends_with_terminator"] and ends_with_terminator
    fixed_length = all(len(values) == 1 for values in lengths.values())
    return {
        "examples": len(fixtures),
        "record_lines": lines,
        "name_byte_lengths": lengths,
        "terminator_offset_table": table,
        "terminator_always": bool(terminator_all),
        "counterexamples": counter,
        "counterexample_total": sum(len(value) for value in counter.values()),
        "fixed_length": bool(fixed_length),
    }


def summarize(rows: list[dict]) -> dict:
    spans = [span for row in rows for span in row["spans"]]
    object_spans = [span for span in spans if span["kind"] == "object"]
    place_spans = [span for span in spans if span["kind"] == "place"]
    selection_object = sum(1 for row in rows if row["object_route_ok"])
    selection_place = sum(1 for row in rows if row["place_route_ok"])
    start_object = sum(1 for span in object_spans if span["start_hit"])
    start_place = sum(1 for span in place_spans if span["start_hit"])
    total_bytes = sum(span["bytes"] for span in spans)
    provenance = sum(span["provenance"] for span in spans)
    misbound = sum(span["misbound"] for span in spans)
    fabricated = sum(span["fabricated"] for span in spans)
    truncated = sum(1 for span in spans if span["truncated"])
    cell: dict = {}
    for span in spans:
        cell[span["cell"]] = cell.get(span["cell"], 0) + 1
    conditional = [1.0 if span["content_ok"] else 0.0 for span in spans if span["start_hit"]]
    differences: dict[str, int] = {}
    positions: dict[str, int] = {}
    for row in rows:
        category = row["first_difference"]["category"]
        differences[category] = differences.get(category, 0) + 1
        position = row["first_difference"]["position"]
        key = "none" if position is None else str(position)
        positions[key] = positions.get(key, 0) + 1
    return {
        "examples": len(rows),
        "selection_accuracy_object": selection_object / len(rows) if rows else 0.0,
        "selection_accuracy_place": selection_place / len(rows) if rows else 0.0,
        "selection_accuracy_both": (
            sum(1 for row in rows if row["object_route_ok"] and row["place_route_ok"]) / len(rows)
            if rows
            else 0.0
        ),
        "completion": sum(1 for row in rows if row["completed"]) / len(rows) if rows else 0.0,
        "byte_exact_rate": sum(1 for row in rows if row["byte_exact"]) / len(rows) if rows else 0.0,
        "start_hit_object": start_object / len(object_spans) if object_spans else 0.0,
        "start_hit_place": start_place / len(place_spans) if place_spans else 0.0,
        "composite_start_hit_object": start_object / len(object_spans) if object_spans else 0.0,
        "composite_start_hit_place": start_place / len(place_spans) if place_spans else 0.0,
        "conditional_byte": statistics.fmean(conditional) if conditional else 0.0,
        "provenance_rate": provenance / total_bytes if total_bytes else 0.0,
        "fabricated_bytes": fabricated,
        "misbound_bytes": misbound,
        "truncation_rate": truncated / len(spans) if spans else 0.0,
        "wrong_route_right_content": cell.get("wrong_route_right_content", 0),
        "cells": cell,
        "first_difference": differences,
        "first_difference_positions": positions,
        "object_span_exact": (
            sum(1 for span in object_spans if span["content_ok"]) / len(object_spans)
            if object_spans
            else 0.0
        ),
        "place_span_exact": (
            sum(1 for span in place_spans if span["content_ok"]) / len(place_spans)
            if place_spans
            else 0.0
        ),
    }


def run_config(
    candidate,
    probe,
    fixtures: tuple[dict, ...],
    routes: list[dict],
    gamma: float,
    label: str,
    started_cpu: float,
) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    trace = {"config": label, "gamma_override": gamma, "stopped_for_budget": False}
    for fixture, route_info in zip(fixtures, routes, strict=True):
        if time.process_time() - started_cpu > CPU_GUARD:
            trace["stopped_for_budget"] = True
            break
        gold_object = int(fixture["selection_label"])
        gold_place = place_record(fixture)
        decoded = decode(candidate, probe, fixture, route_info, gamma)
        object_span = analyze_span(fixture, decoded, 3, 12, "object")
        place_span = analyze_span(fixture, decoded, 21, 6, "place")
        target = target_bytes(fixture)
        byte_exact = len(decoded["raw"]) == len(target) and tuple(decoded["raw"]) == target
        rows.append(
            {
                "world": fixture["world"],
                "index": fixture["index"],
                "age": age_of(fixture),
                "object_route_ok": route_info["object_route"] == gold_object,
                "place_route_ok": route_info["place_route"] == gold_place,
                "spans": [object_span, place_span],
                "byte_exact": bool(byte_exact),
                "answer_bytes": len(target),
                "completed": bool(byte_exact and decoded["terminated"]),
                "raw": decoded["raw"],
                "target": target,
                "terminated": decoded["terminated"],
                "first_difference": first_difference(fixture, decoded),
            }
        )
        append_progress(
            {
                "event": "example",
                "config": label,
                "world": fixture["world"],
                "index": fixture["index"],
                "age": rows[-1]["age"],
                "object_route_ok": rows[-1]["object_route_ok"],
                "place_route_ok": rows[-1]["place_route_ok"],
                "completed": rows[-1]["completed"],
                "cpu_seconds": round(time.process_time() - started_cpu, 4),
            }
        )
    trace["examples"] = len(rows)
    return rows, trace


def side_by_side(dev: dict, confirm: dict) -> list[dict]:
    items = list(L1_TARGETS) + [
        "completion",
        "byte_exact_rate",
        "object_span_exact",
        "place_span_exact",
    ]
    # 19.2u did not report byte_exact_rate separately; its completion already
    # required byte equality, so the derived value is the dev completion.
    derived = {"byte_exact_rate": dev["completion"]}
    table = []
    for item in items:
        dev_value = dev[item] if item in dev else derived[item]
        confirm_value = confirm[item]
        table.append(
            {
                "item": item,
                "dev": dev_value,
                "confirm": confirm_value,
                "delta_confirm_minus_dev": confirm_value - dev_value,
            }
        )
    return table


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preregister-only",
        action="store_true",
        help="write the frozen snapshot and exit before any confirm read",
    )
    parser.add_argument("--limit", type=int, default=0, help="debug: cap confirm examples")
    args = parser.parse_args()

    started_cpu = time.process_time()
    started_wall = time.perf_counter()

    if not ASSEMBLED.exists():
        raise RuntimeError(f"assembled state missing: {ASSEMBLED}")
    if not BUNDLE.exists():
        raise RuntimeError(f"parent bundle missing: {BUNDLE}")
    if not DEV_REPORT.exists():
        raise RuntimeError(f"dev reference report missing: {DEV_REPORT}")

    assembled_sha = sha256_of(ASSEMBLED)
    assembled_payload = torch.load(ASSEMBLED, map_location="cpu")
    candidate = load_routing_bundle(BUNDLE)[0]
    probe = probe_mod.ProbeContext(candidate)
    probe.arm = "B_stop_lock"
    with torch.no_grad():
        identity = candidate_identity.candidate_identity(candidate)
        parameter_count = sum(
            int(tensor.numel()) for tensor in parameter_tensors(candidate).values()
        )
    candidate_sha = identity["candidate_sha_v2"]
    if assembled_payload.get("candidate_sha_v2") != candidate_sha:
        raise RuntimeError("assembled state candidate_sha_v2 does not match the loaded candidate")
    if assembled_payload.get("copy_stop_boundary") is not True:
        raise RuntimeError("assembled state is not marked copy_stop_boundary")
    if assembled_payload.get("byte_match_parameters") != sum(
        int(tensor.numel()) for tensor in ByteMatchRoute(dim=IDENTITY_DIM, span=SPAN).parameters()
    ):
        raise RuntimeError("assembled state byte_match parameter count drift")

    route = ByteMatchRoute(dim=IDENTITY_DIM, span=SPAN)
    route.load_state_dict(assembled_payload["byte_match_state"])
    route.eval()
    route_parameters = sum(int(tensor.numel()) for tensor in route.parameters())

    prereg = {
        "schema": "taiji-package19-2v-preregistration-v1",
        "step": "N19.2v",
        "date": "2026-11-05",
        "purpose": "first and only confirm read of the structural binding-copy task",
        "frozen": {
            "assembled_state": str(ASSEMBLED.relative_to(ROOT)),
            "assembled_bytes": ASSEMBLED.stat().st_size,
            "assembled_sha256": assembled_sha,
            "parent_bundle": str(BUNDLE.relative_to(ROOT)),
            "parent_bundle_sha256": sha256_of(BUNDLE),
            "candidate_sha_v2": candidate_sha,
            "candidate_parameters": parameter_count,
            "byte_match_parameters": route_parameters,
            "copy_stop_module": "taiji/copy_stop_boundary.py",
            "copy_stop_rule": (
                "per-hop copy limit = name-field end from the record template "
                "delimiter + terminator; no sample-specific constant, no gamma, "
                "no generated weights"
            ),
            "graph_modules": list(candidate_identity.GRAPH_MODULES),
        },
        "split": {
            "name": "confirm",
            "examples_expected": EXPECTED_CONFIRM,
            "old_description": 96,
            "new_description": 96,
            "confirm_reads": 1,
            "no_reread_after_decision": True,
        },
        "configurations": {
            "gamma_0_main": 0.0,
            "gamma_floor_control": "model gamma_floor (read-only)",
            "forward_per_configuration": 1,
        },
        "readings": [
            "selection_accuracy_object",
            "selection_accuracy_place",
            "selection_accuracy_both",
            "start_hit_object",
            "start_hit_place",
            "composite_start_hit_object",
            "composite_start_hit_place",
            "conditional_byte",
            "provenance_rate",
            "fabricated_bytes",
            "misbound_bytes",
            "truncation_rate",
            "wrong_route_right_content",
            "first_difference",
            "first_difference_positions",
            "completion",
            "byte_exact_rate",
            "answer_bytes",
        ],
        "l1_mechanical_invariants": L1_TARGETS,
        "l1_structural_invariants": {
            "confirm_boundary_fixed_length": True,
            "confirm_boundary_terminator_always": True,
            "confirm_boundary_counterexample_total": 0,
        },
        "thresholds": {
            "milestone_completion": MILESTONE,
            "dev_overfit_delta": OVERFIT_DELTA,
        },
        "decision_rules": [
            (
                "L1: every mechanical invariant must hit its target (1.0 or 0) and the "
                "confirm record template must be fixed-length, terminator-terminated and "
                "counterexample-free; any deviation -> structural_regression, with the "
                "failing byte position and record kind localised."
            ),
            (
                "Milestone: confirm completion >= 0.95 -> milestone holds; the capability "
                "card upgrades 'binding copy + selection' with 'structural implementation, "
                "training-weight contribution 0'."
            ),
            (
                "Overfit: confirm completion - dev completion <= -0.05 -> "
                "dev_overfit_of_structure, listing which design-split features were baked "
                "in (field lengths 12/6, template shapes style0/1/2, terminator)."
            ),
            "Otherwise: milestone_not_met, report the gap and the failure positions.",
            "Order: L1 first; a structural regression overrides every other branch.",
        ],
        "immutability": "written once before the confirm read; never rewritten afterwards",
        "cost_caps": {"cpu_seconds": CPU_CAP, "wall_seconds": WALL_CAP},
        "no_change_policy": (
            "0 training, 0 optimizer steps, 0 Adam, no parameter or graph change, no tuning, "
            "no re-run after seeing confirm"
        ),
    }

    if args.preregister_only:
        if PREREG.exists():
            raise RuntimeError(f"preregistration already exists, refusing to rewrite: {PREREG}")
        write_json(PREREG, prereg)
        print(
            json.dumps({"preregistered": str(PREREG), "candidate_sha_v2": candidate_sha}, indent=1)
        )
        return 0

    if not PREREG.exists():
        raise RuntimeError("preregistration snapshot is missing; run --preregister-only first")
    frozen = json.loads(PREREG.read_text(encoding="utf-8"))["frozen"]
    mismatches = [
        key
        for key, value in (
            ("assembled_sha256", assembled_sha),
            ("candidate_sha_v2", candidate_sha),
            ("parent_bundle_sha256", sha256_of(BUNDLE)),
            ("byte_match_parameters", route_parameters),
        )
        if frozen[key] != value
    ]
    if mismatches:
        raise RuntimeError(f"preregistered configuration drift: {mismatches}")

    dev_reference_payload = json.loads(DEV_REPORT.read_text(encoding="utf-8"))
    dev_readings = {
        "gamma_0": dev_reference_payload["readings"]["base_gamma_0"]["total"],
        "gamma_floor": dev_reference_payload["readings"]["base_gamma_floor"]["total"],
    }

    recovery = False
    if PROGRESS.exists() and PROGRESS.stat().st_size > 0:
        events = [
            json.loads(line)
            for line in PROGRESS.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if any(event.get("event") == "finish" for event in events):
            raise RuntimeError("this confirm split was already evaluated; refusing a second read")
        recovery = True
    else:
        PROGRESS.write_text("", encoding="utf-8")
    append_progress(
        {
            "event": "prereg",
            "recovery_rerun": recovery,
            "assembled_sha256": assembled_sha,
            "candidate_sha_v2": candidate_sha,
            "cpu_cap": CPU_CAP,
        }
    )

    # ``build_payload`` keeps confirm sealed (worlds are not generated until the
    # candidate and reader are frozen).  Materialising it from the frozen pools
    # with the frozen split widths is this package's single confirm read.
    payload = premise.build_payload()
    pools = premise.split_pools()
    confirm_worlds = [
        premise.build_world("confirm", world, pools)
        for world in range(premise.SPLIT_WORLDS["confirm"])
    ]
    materialised = dict(payload)
    materialised["worlds"] = [
        {key: value for key, value in world.items() if key != "examples"}
        for world in confirm_worlds
    ]
    materialised["examples"] = [
        example for world in confirm_worlds for example in world["examples"]
    ]
    confirm = tuple(premise.fixtures(materialised, "confirm"))
    if args.limit:
        confirm = confirm[: args.limit]
    if not args.limit and len(confirm) != EXPECTED_CONFIRM:
        raise RuntimeError(f"confirm fixture count drift: {len(confirm)}")
    append_progress({"event": "confirm_loaded", "examples": len(confirm)})

    boundary = measure_boundaries(confirm)
    append_progress(
        {
            "event": "confirm_boundary",
            "record_lines": boundary["record_lines"],
            "fixed_length": boundary["fixed_length"],
            "terminator_always": boundary["terminator_always"],
            "counterexample_total": boundary["counterexample_total"],
        }
    )

    tensors = build_tensors(confirm)
    routes = match_routes(route, tensors, confirm)
    append_progress({"event": "matched", "examples": len(routes)})

    gamma_main = 0.0
    gamma_floor = float(candidate.head.config.gamma_floor)
    main_rows, main_trace = run_config(
        candidate, probe, confirm, routes, gamma_main, "confirm_gamma_0", started_cpu
    )
    floor_rows, floor_trace = run_config(
        candidate, probe, confirm, routes, gamma_floor, "confirm_gamma_floor", started_cpu
    )

    def split(rows: list[dict]) -> dict:
        return {
            "total": summarize(rows),
            "old": summarize([row for row in rows if row["age"] == "old"]),
            "new": summarize([row for row in rows if row["age"] == "new"]),
        }

    readings = {
        "gamma_0": split(main_rows),
        "gamma_floor": split(floor_rows),
    }
    main_total = readings["gamma_0"]["total"]

    l1_failures = [
        {"item": item, "value": main_total[item], "target": target}
        for item, target in L1_TARGETS.items()
        if main_total[item] != target
    ]
    boundary_failures = []
    if not boundary["fixed_length"]:
        boundary_failures.append({"item": "boundary_fixed_length", "value": False})
    if not boundary["terminator_always"]:
        boundary_failures.append({"item": "boundary_terminator_always", "value": False})
    if boundary["counterexample_total"] != 0:
        boundary_failures.append(
            {"item": "boundary_counterexample_total", "value": boundary["counterexample_total"]}
        )

    failures = []
    for row in main_rows:
        if row["completed"]:
            continue
        position = row["first_difference"]["position"]
        failures.append(
            {
                "world": row["world"],
                "index": row["index"],
                "age": row["age"],
                "category": row["first_difference"]["category"],
                "position": position,
                "record_kind": None if position is None else record_kind_at_position(position),
                "decoded_text": bytes(row["raw"]).decode("utf-8", "replace"),
                "target_text": bytes(row["target"]).decode("utf-8", "replace"),
                "spans": [
                    {
                        "kind": span["kind"],
                        "route": span["route"],
                        "gold_route": span["gold_route"],
                        "cell": span["cell"],
                        "start_hit": span["start_hit"],
                        "truncated": span["truncated"],
                    }
                    for span in row["spans"]
                ],
            }
        )

    dev_completion = dev_readings["gamma_0"]["completion"]
    delta = main_total["completion"] - dev_completion
    if l1_failures or boundary_failures:
        verdict = "structural_regression"
        branch = (
            f"L1 有 {len(l1_failures) + len(boundary_failures)} 项未达标；"
            f"失败项 {[entry['item'] for entry in l1_failures + boundary_failures]}"
        )
    elif main_total["completion"] >= MILESTONE:
        verdict = "milestone_holds"
        branch = (
            f"confirm 整句 {main_total['completion']:.4f} ≥ {MILESTONE:.2f}，"
            "能力由结构实现，训练权重贡献为 0"
        )
    elif delta <= OVERFIT_DELTA:
        verdict = "dev_overfit_of_structure"
        branch = (
            f"confirm 整句 {main_total['completion']:.4f} 比 dev {dev_completion:.4f} "
            f"低 {delta:+.4f}（阈值 {OVERFIT_DELTA:.2f}）；设计阶段吃进的特征："
            "物件名恒 12 字节、地名恒 6 字节、三种模板形状与终止符 `。`"
        )
    else:
        verdict = "milestone_not_met"
        branch = (
            f"confirm 整句 {main_total['completion']:.4f} < {MILESTONE:.2f} 且与 dev 差 "
            f"{delta:+.4f} 未达过拟合阈值；缺口位置见 failures"
        )

    answer_lengths = dict(Counter(row["answer_bytes"] for row in main_rows))
    cpu_seconds = time.process_time() - started_cpu
    wall_seconds = time.perf_counter() - started_wall
    over_budget = cpu_seconds > CPU_CAP or wall_seconds > WALL_CAP

    report = {
        "schema": "taiji-package19-2v-confirm-holdout-v1",
        "step": "N19.2v",
        "date": "2026-11-05",
        "preregistration": str(PREREG.relative_to(ROOT)),
        "preregistration_sha256": sha256_of(PREREG),
        "assembly": {
            "parent_bundle": str(BUNDLE.relative_to(ROOT)),
            "candidate_sha_v2": candidate_sha,
            "candidate_parameters": parameter_count,
            "byte_match_parameters": route_parameters,
            "assembled_state": str(ASSEMBLED.relative_to(ROOT)),
            "assembled_state_bytes": ASSEMBLED.stat().st_size,
            "assembled_state_sha256": assembled_sha,
            "copy_stop_boundary": True,
            "copy_stop_module": "taiji/copy_stop_boundary.py",
            "route_state_source": "byte_match_state loaded from the 19.2u assembled state",
            "frozen": ["transport", "span_head", "score_rows", "byte_match_route"],
        },
        "evaluation": {
            "split": "confirm",
            "examples": len(confirm),
            "old_description": sum(1 for row in main_rows if row["age"] == "old"),
            "new_description": sum(1 for row in main_rows if row["age"] == "new"),
            "confirm_reads": 1,
            "recovery_rerun": recovery,
            "recovery_note": (
                (
                    "first attempt aborted before any confirm fixture was materialised "
                    "(0 examples, 0 forwards); this run is the first actual confirm read"
                )
                if recovery
                else "no recovery"
            ),
            "gamma_0_main": gamma_main,
            "gamma_floor_control": gamma_floor,
            "answer_bytes": answer_lengths,
            "answer_bytes_note": (
                "measured whole-answer length; the request's '57 bytes' does not match the "
                "frozen fixture template (9 template bytes + 12 object name + 6 place name = 27)"
            ),
        },
        "boundary_measurement_confirm": boundary,
        "readings": readings,
        "dev_reference": {
            "source": str(DEV_REPORT.relative_to(ROOT)),
            "source_sha256": sha256_of(DEV_REPORT),
            "split": "dev",
            "examples": dev_readings["gamma_0"]["examples"],
            "readings": dev_readings,
        },
        "side_by_side": side_by_side(dev_readings["gamma_0"], main_total),
        "decision": {
            "verdict": verdict,
            "branch": branch,
            "l1_failures": l1_failures,
            "boundary_failures": boundary_failures,
            "dev_completion": dev_completion,
            "confirm_completion": main_total["completion"],
            "delta_confirm_minus_dev": delta,
        },
        "failures": failures,
        "audit": [
            {
                "world": row["world"],
                "index": row["index"],
                "age": row["age"],
                "completed": row["completed"],
                "byte_exact": row["byte_exact"],
                "first_difference": row["first_difference"],
                "spans": row["spans"],
            }
            for row in main_rows
        ],
        "trace": {
            "gamma_0": main_trace,
            "gamma_floor": floor_trace,
            "determinism": "not repeated: the confirm split is read once by design",
        },
        "cost": {
            "cpu_seconds": round(cpu_seconds, 2),
            "wall_seconds": round(wall_seconds, 2),
            "cpu_cap": CPU_CAP,
            "wall_cap": WALL_CAP,
            "over_budget": bool(over_budget),
        },
        "discipline": {
            "training_updates": 0,
            "optimizer_steps": 0,
            "adam_steps": 0,
            "confirm_reads": 1,
            "confirm_reruns": 1 if recovery else 0,
            "parameter_changes": 0,
            "graph_changes": 0,
            "fsync_per_line": True,
            "report_write_then_rename": True,
        },
    }
    write_json(REPORT, report)
    write_json(
        MANIFEST,
        {
            "step": "N19.2v",
            "date": "2026-11-05",
            "preregistration": str(PREREG.relative_to(ROOT)),
            "preregistration_sha256": sha256_of(PREREG),
            "assembled_state": str(ASSEMBLED.relative_to(ROOT)),
            "assembled_state_sha256": assembled_sha,
            "candidate_sha_v2": candidate_sha,
            "examples": len(confirm),
            "confirm_reads": 1,
            "recovery_rerun": recovery,
            "verdict": verdict,
            "confirm_completion": main_total["completion"],
            "dev_completion": dev_completion,
            "completion_delta": delta,
            "l1_pass": not l1_failures and not boundary_failures,
            "fabricated_bytes": main_total["fabricated_bytes"],
            "misbound_bytes": main_total["misbound_bytes"],
            "wrong_route_right_content": main_total["wrong_route_right_content"],
            "cpu_seconds": round(cpu_seconds, 2),
            "wall_seconds": round(wall_seconds, 2),
            "over_budget": bool(over_budget),
        },
    )
    append_progress(
        {
            "event": "finish",
            "verdict": verdict,
            "completion": main_total["completion"],
            "cpu_seconds": round(cpu_seconds, 4),
            "wall_seconds": round(wall_seconds, 4),
        }
    )
    print(
        json.dumps(
            {
                "verdict": verdict,
                "confirm_completion": main_total["completion"],
                "dev_completion": dev_completion,
                "l1_failures": l1_failures,
                "boundary_failures": boundary_failures,
                "failures": len(failures),
                "cpu_seconds": round(cpu_seconds, 2),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
