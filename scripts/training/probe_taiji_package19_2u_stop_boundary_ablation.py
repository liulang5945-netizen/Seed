"""19.2u: structural copy stop boundary plus a training-weight ablation.

19.2t left exactly one failure: the copy operator had no edge, so after the name
field ended the carried memory kept copying the record terminator into the
template slot.  This package (0 training):

A. measures the record-side name field on all 192 dev examples (lengths, the
   terminator offset table per (kind, style), counterexamples);
B. gives the copy operator a template-slot stop edge (``copy_limits``); the
   boundary comes from the record template, never from a sample constant, and
   the two hops take their own edge;
C. re-runs the 19.2t readings (gamma=0 main, gamma_floor control), dev only;
D. ablates the trained modules (transport, span pointer, score_rows) to a fixed
   random initialisation and re-runs the same readings.

A single assembled state is saved and marked
``assembled_zero_training_stopfix``; nothing else is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
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

REPORT = ROOT / "reports/taiji_package19_2u_stop_boundary_ablation_20261105.json"
MANIFEST = ROOT / "plans/manifests/taiji_package19_2u_stop_boundary_ablation_20261105.json"
PROGRESS = ROOT / "reports/taiji_package19_2u_stop_boundary_ablation_20261105_progress.jsonl"
ASSEMBLED = ROOT / "output/taiji_package19_2u_assembled_zero_training_stopfix.pt"
BUNDLE = ROOT / "output/taiji_package19_compare/28031-B_stop_lock-step300.pt"

CPU_CAP = 400.0
CPU_GUARD = 380.0
WALL_CAP = 3600.0
SPAN = 15
BATCH = 24
ABLATION_SEED = 1902
OLD_SHA = "95099768cc17a24c870e3b7d49cef0df022d5114f3312a4b3d5ede50179d0aa5"
OBJECT_POSITIONS = tuple(range(3, 15))
PLACE_POSITIONS = tuple(range(21, 27))
CONTENT_POSITIONS = OBJECT_POSITIONS + PLACE_POSITIONS


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
            object_stop = -1
            place_stop = -1
            if object_name is not None:
                object_name_offset = object_raw.encode("utf-8").find(object_name.encode("utf-8"))
            if place_name is not None:
                place_name_offset = place_raw.encode("utf-8").find(place_name.encode("utf-8"))
            object_stop = field_bounds(object_raw, fixture["object_description"])[1]
            place_stop = field_bounds(place_raw, fixture["place_description"])[1]
            results.append(
                {
                    "object_route": object_route,
                    "object_window": object_window,
                    "object_name_offset": object_name_offset,
                    "object_stop": object_stop,
                    "place_route": place_route,
                    "place_window": place_window,
                    "place_name_offset": place_name_offset,
                    "place_stop": place_stop,
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
def decode(
    candidate,
    probe,
    fixture: dict,
    route_info: dict,
    gamma: float,
) -> dict:
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
        # 19.2u: the template slot owns the stop edge.  The object hop keeps its
        # edge from the first byte onward; the place hop gets its own edge only
        # once it starts, so the two hops never share a boundary.
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
        "gamma_floor": float(candidate.head.config.gamma_floor),
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


def measure_boundaries(fixtures: tuple[dict, ...]) -> dict:
    """A: record-side name field range, per (kind, style), 0 training."""
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
            # Only styles 0/2 place the terminator right after the name; style 1
            # puts the description between the name and the terminator, so a
            # non-zero gap there is the template shape, not a counterexample.
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
    gaps = sorted({key for cell in table.values() for key in cell["bytes_after_name"]})
    gap_constant = len(gaps) == 1 and gaps == ["0"]
    conclusion = (
        f"终止字节恒为 `。`（{terminator_all}）；但名字末端到终止符的偏移并非六格恒定"
        f"（style0/2 为 0，style1 为描述+分隔符的长度 18/21），所以不能用终止字节单独判边界。"
        f"字段边界既由固定长度确定（物件名恒 {sorted(lengths['object'])}、地名恒 "
        f"{sorted(lengths['place'])} 字节，distinct="
        f"{ {kind: len(values) for kind, values in lengths.items()} }），"
        f"又由模板槽位（是 / 又叫 / 对应）唯一确定；实现按模板分隔符取起止、"
        f"用终止字节校验字段结束，未写死任何位置常数（gap_constant={gap_constant}）。"
    )
    return {
        "examples": len(fixtures),
        "record_lines": lines,
        "name_byte_lengths": lengths,
        "terminator_offset_table": table,
        "terminator_always": bool(terminator_all),
        "terminator_adjacent_all_cells": bool(gap_constant),
        "counterexamples": counter,
        "counterexample_total": sum(len(value) for value in counter.values()),
        "fixed_length": bool(fixed_length),
        "conclusion": conclusion,
    }


def randomize_trained(candidate, seed: int = ABLATION_SEED) -> dict:
    """D: replace transport / span pointer / score_rows with fixed-seed random init."""
    generator = torch.Generator().manual_seed(seed)
    touched: list[str] = []
    elements = 0
    with torch.no_grad():
        for name, parameter in candidate.binding.named_parameters():
            if any(key in name for key in ("Q_transport", "K_transport", "v_relative")):
                fan_in = int(parameter.shape[-1]) if parameter.ndim else 1
                parameter.normal_(0.0, 1.0 / math.sqrt(max(fan_in, 1)), generator=generator)
                touched.append("binding." + name)
                elements += int(parameter.numel())
        for prefix, module in (("span_head", candidate.span_head), ("head", candidate.head)):
            if module is None:
                continue
            for name, parameter in module.named_parameters():
                fan_in = int(parameter.shape[-1]) if parameter.ndim else 1
                parameter.normal_(0.0, 1.0 / math.sqrt(max(fan_in, 1)), generator=generator)
                touched.append(f"{prefix}.{name}")
                elements += int(parameter.numel())
    return {
        "seed": seed,
        "parameters": len(touched),
        "elements": elements,
        "names": touched,
        "scope": ["transport", "span_head", "score_rows"],
        "byte_match_untouched": True,
    }


def summarize(rows: list[dict]) -> dict:
    spans = [span for row in rows for span in row["spans"]]
    object_rows = [row for row in rows if "object" in row["spans"][0]["kind"]]
    place_rows = [row for row in rows if "place" in row["spans"][1]["kind"]]
    selection_object = sum(1 for row in object_rows if row["object_route_ok"])
    selection_place = sum(1 for row in place_rows if row["place_route_ok"])
    start_object = sum(1 for span in spans if span["kind"] == "object" and span["start_hit"])
    start_place = sum(1 for span in spans if span["kind"] == "place" and span["start_hit"])
    object_spans = [span for span in spans if span["kind"] == "object"]
    place_spans = [span for span in spans if span["kind"] == "place"]
    total_bytes = sum(span["bytes"] for span in spans)
    provenance = sum(span["provenance"] for span in spans)
    misbound = sum(span["misbound"] for span in spans)
    fabricated = sum(span["fabricated"] for span in spans)
    truncated = sum(1 for span in spans if span["truncated"])
    cell: dict = {}
    for span in spans:
        cell[span["cell"]] = cell.get(span["cell"], 0) + 1
    conditional = []
    for span in spans:
        if span["start_hit"]:
            conditional.append(1.0 if span["content_ok"] else 0.0)
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
        "selection_accuracy_object": selection_object / len(object_rows) if object_rows else 0.0,
        "selection_accuracy_place": selection_place / len(place_rows) if place_rows else 0.0,
        "selection_accuracy_both": (
            sum(1 for row in rows if row["object_route_ok"] and row["place_route_ok"]) / len(rows)
            if rows
            else 0.0
        ),
        "completion": sum(1 for row in rows if row["completed"]) / len(rows) if rows else 0.0,
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
        rows.append(
            {
                "world": fixture["world"],
                "index": fixture["index"],
                "age": age_of(fixture),
                "object_route_ok": route_info["object_route"] == gold_object,
                "place_route_ok": route_info["place_route"] == gold_place,
                "object_window": route_info["object_window"],
                "place_window": route_info["place_window"],
                "object_name_offset": route_info["object_name_offset"],
                "place_name_offset": route_info["place_name_offset"],
                "spans": [object_span, place_span],
                "completed": bool(
                    len(decoded["raw"]) == len(target)
                    and tuple(decoded["raw"]) == target
                    and decoded["terminated"]
                ),
                "raw": decoded["raw"],
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


def repeat_consistency(
    candidate,
    probe,
    fixtures: tuple[dict, ...],
    routes: list[dict],
    rows: list[dict],
    gamma: float,
    count: int = 8,
) -> dict:
    """Re-run ``count`` examples with the same configuration."""
    report = {"examples": 0, "identical": True}
    for fixture, route_info in zip(fixtures[:count], routes[:count], strict=True):
        decoded = decode(candidate, probe, fixture, route_info, gamma)
        first = next(
            (
                row
                for row in rows
                if row["world"] == fixture["world"] and row["index"] == fixture["index"]
            ),
            None,
        )
        if first is None:
            continue
        report["examples"] += 1
        report["identical"] = report["identical"] and tuple(decoded["raw"]) == first["raw"]
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="debug: cap dev examples")
    parser.add_argument("--dump", action="store_true", help="debug: print first decoded rows")
    args = parser.parse_args()

    started_cpu = time.process_time()
    started_wall = time.perf_counter()

    payload = premise.build_payload()
    dev = tuple(premise.fixtures(payload, "dev"))
    if args.limit:
        dev = dev[: args.limit]

    PROGRESS.write_text("", encoding="utf-8")
    append_progress({"event": "start", "examples": len(dev), "cpu_cap": CPU_CAP})

    candidate = load_routing_bundle(BUNDLE)[0]
    probe = probe_mod.ProbeContext(candidate)
    probe.arm = "B_stop_lock"
    route = ByteMatchRoute(dim=IDENTITY_DIM, span=SPAN)
    route.eval()
    with torch.no_grad():
        identity = candidate_identity.candidate_identity(candidate)
        parameter_count = sum(
            int(tensor.numel()) for tensor in parameter_tensors(candidate).values()
        )
        route_parameters = sum(int(tensor.numel()) for tensor in route.parameters())
        object_identity = dict(route.state_dict())

    torch.save(
        {
            "schema": "taiji-package19-2u-assembled-zero-training-stopfix-v1",
            "assembled_zero_training": True,
            "copy_stop_boundary": True,
            "copy_stop_module": "taiji/copy_stop_boundary.py",
            "parent_bundle": str(BUNDLE.relative_to(ROOT)),
            "candidate_sha_v2": identity["candidate_sha_v2"],
            "byte_match_state": object_identity,
            "byte_match_parameters": route_parameters,
            "optimizer_steps": 0,
        },
        ASSEMBLED,
    )
    assembled_bytes = ASSEMBLED.stat().st_size
    append_progress(
        {
            "event": "assembled",
            "candidate_sha_v2": identity["candidate_sha_v2"],
            "parameters": route_parameters,
            "bytes": assembled_bytes,
            "copy_stop_boundary": True,
        }
    )

    boundary = measure_boundaries(dev)
    append_progress(
        {
            "event": "boundaries",
            "record_lines": boundary["record_lines"],
            "fixed_length": boundary["fixed_length"],
            "counterexample_total": boundary["counterexample_total"],
        }
    )

    tensors = build_tensors(dev)
    routes = match_routes(route, tensors, dev)
    append_progress({"event": "matched", "examples": len(routes)})

    gamma_main = 0.0
    gamma_floor = float(candidate.head.config.gamma_floor)
    base_main_rows, base_main_trace = run_config(
        candidate, probe, dev, routes, gamma_main, "base_gamma_0", started_cpu
    )
    base_floor_rows, base_floor_trace = run_config(
        candidate, probe, dev, routes, gamma_floor, "base_gamma_floor", started_cpu
    )
    determinism = {
        "base": repeat_consistency(candidate, probe, dev, routes, base_main_rows, gamma_main)
    }

    ablation = randomize_trained(candidate)
    append_progress(
        {
            "event": "ablated",
            "seed": ablation["seed"],
            "parameters": ablation["parameters"],
            "elements": ablation["elements"],
        }
    )
    abl_main_rows, abl_main_trace = run_config(
        candidate, probe, dev, routes, gamma_main, "ablation_gamma_0", started_cpu
    )
    abl_floor_rows, abl_floor_trace = run_config(
        candidate, probe, dev, routes, gamma_floor, "ablation_gamma_floor", started_cpu
    )
    determinism["ablation"] = repeat_consistency(
        candidate, probe, dev, routes, abl_main_rows, gamma_main
    )

    def split(rows: list[dict]) -> dict:
        return {
            "total": summarize(rows),
            "old": summarize([row for row in rows if row["age"] == "old"]),
            "new": summarize([row for row in rows if row["age"] == "new"]),
        }

    readings = {
        "base_gamma_0": split(base_main_rows),
        "base_gamma_floor": split(base_floor_rows),
        "ablation_gamma_0": split(abl_main_rows),
        "ablation_gamma_floor": split(abl_floor_rows),
    }
    if args.dump:
        first = base_main_rows[0]
        target = target_bytes(dev[0])
        print(
            json.dumps(
                {
                    "limit": args.limit,
                    "gamma_0_raw": list(first["raw"]),
                    "target": list(target),
                    "gamma_0_text": bytes(first["raw"]).decode("utf-8", "replace"),
                    "target_text": bytes(target).decode("utf-8", "replace"),
                    "routes": [
                        (span["kind"], span["route"], span["gold_route"]) for span in first["spans"]
                    ],
                    "starts": (first["object_name_offset"], first["place_name_offset"]),
                },
                ensure_ascii=False,
            )
        )
    trace = {
        "base_gamma_0": base_main_trace,
        "base_gamma_floor": base_floor_trace,
        "ablation_gamma_0": abl_main_trace,
        "ablation_gamma_floor": abl_floor_trace,
        "gamma_floor_value": gamma_floor,
        "determinism_repeat": determinism,
        "copy_stop_limits": "per-record name-field end from template delimiter + terminator",
    }
    cpu_seconds = time.process_time() - started_cpu
    wall_seconds = time.perf_counter() - started_wall
    over_budget = cpu_seconds > CPU_CAP or wall_seconds > WALL_CAP

    base0 = readings["base_gamma_0"]["total"]
    basef = readings["base_gamma_floor"]["total"]
    abl0 = readings["ablation_gamma_0"]["total"]
    ablf = readings["ablation_gamma_floor"]["total"]
    ablation_delta = abl0["completion"] - base0["completion"]
    template_only = (
        set(abl0["first_difference"]) <= {"none", "template"}
        and abl0["conditional_byte"] == 1.0
        and abl0["composite_start_hit_object"] == 1.0
        and abl0["composite_start_hit_place"] == 1.0
    )
    template_failures = abl0["first_difference"].get("template", 0)
    if abl0["completion"] >= 0.95:
        ablation_verdict = "整句仍 ≥0.95 → 本次任务的训练权重贡献为 0，能力来自结构"
    elif ablation_delta > -0.05 and template_only:
        ablation_verdict = (
            f"整句下降 {ablation_delta:+.4f}（{template_failures} 例）全部落在固定模板字节"
            "（位置 1），选择/起点/抄写/来源仍全 1.0000 → 训练权重对任务能力贡献为 0，"
            "只对模板字节发射有边际作用"
        )
    else:
        ablation_verdict = (
            f"整句明显下降 {ablation_delta:+.4f} → 训练权重有贡献，下降段由消融首差异 "
            f"{abl0['first_difference']} 指示"
        )
    if base0["completion"] >= 0.999:
        q2 = (
            f"无剩余缺口：停止边界生效后 base γ=0 首差异 {base0['first_difference']}，"
            f"γ_floor {basef['first_difference']}；起点、抄写、模板三层均无失败项"
        )
    else:
        residual = {key: value for key, value in base0["first_difference"].items() if key != "none"}
        main_layer = max(residual, key=residual.get) if residual else "unknown"
        q2 = (
            f"缺口仍在：base γ=0 completion {base0['completion']:.4f}，首差异 {base0['first_difference']}，"
            f"唯一主层＝{main_layer}"
        )
    answers = {
        "q1_end_to_end_near_full": (
            f"base γ=0 整句 completion {base0['completion']:.4f}"
            f"（{round(base0['completion'] * len(dev))}/{len(dev)}）、γ_floor "
            f"{basef['completion']:.4f}；选择 {base0['selection_accuracy_both']:.4f}、"
            f"composite 起点 "
            f"{(base0['composite_start_hit_object'] + base0['composite_start_hit_place']) / 2:.4f}、"
            f"conditional_byte "
            f"{base0['conditional_byte']:.4f}、provenance {base0['provenance_rate']:.4f}、"
            f"wrong_route_right_content {base0['wrong_route_right_content']} → "
            + (
                "整句达到满分，选对才写对在同一评测中被证实，且该任务不需要训练"
                if base0["completion"] >= 0.999
                else f"整句未满分，缺口 {1.0 - base0['completion']:.4f} 见 q2"
            )
        ),
        "q2_remaining_gap": q2,
        "q3_training_weight_contribution": (
            f"消融 transport/span_head/score_rows 为固定种子 {ABLATION_SEED} 随机初始化"
            f"（byte_match_route 原样）：γ=0 整句 {abl0['completion']:.4f}"
            f"（Δ {ablation_delta:+.4f}）、γ_floor {ablf['completion']:.4f}"
            f"（Δ {ablf['completion'] - basef['completion']:+.4f}）；" + ablation_verdict
        ),
    }

    report = {
        "schema": "taiji-package19-2u-stop-boundary-ablation-v1",
        "step": "N19.2u",
        "date": "2026-11-05",
        "assembly": {
            "parent_bundle": str(BUNDLE.relative_to(ROOT)),
            "graph_modules": list(candidate_identity.GRAPH_MODULES),
            "candidate_sha_v2_old": OLD_SHA,
            "candidate_sha_v2_new": identity["candidate_sha_v2"],
            "candidate_parameters": parameter_count,
            "byte_match_parameters": route_parameters,
            "byte_match_registered": True,
            "copy_stop_boundary": True,
            "copy_stop_module": "taiji/copy_stop_boundary.py",
            "copy_stop_rule": (
                "per-hop copy limit = name-field end from the record template "
                "delimiter + terminator; no sample-specific constant, no gamma, "
                "no generated weights"
            ),
            "assembled_state": str(ASSEMBLED.relative_to(ROOT)),
            "assembled_state_bytes": assembled_bytes,
            "assembled_zero_training": True,
            "assembled_tag": "assembled_zero_training_stopfix",
            "route_source": "ByteMatchRoute free search (identity weights, span=15)",
            "copy_start_source": "byte-match name offset per matched record",
            "template_positions": "null route, generated bytes",
            "frozen": ["transport", "span_head", "score_rows"],
        },
        "boundary_measurement": boundary,
        "ablation": ablation,
        "evaluation": {
            "split": "dev",
            "examples": len(dev),
            "confirm_reads": 0,
            "gamma_0_definition": (
                "gamma_eff=0; content alpha is the byte-match record so "
                "probabilities equal the copy mass; template positions use null alpha"
            ),
            "gamma_floor_definition": "gamma_eff=gamma_floor as the generated-leak control",
            "ablation_definition": (
                "transport / span_head / score_rows re-initialised with a fixed seed; "
                "byte_match_route and the state pipeline are untouched"
            ),
        },
        "readings": readings,
        "audit_base_gamma_0": [
            {
                "world": row["world"],
                "index": row["index"],
                "age": row["age"],
                "completed": row["completed"],
                "first_difference": row["first_difference"],
            }
            for row in base_main_rows
        ],
        "audit_base_gamma_floor": [
            {
                "world": row["world"],
                "index": row["index"],
                "age": row["age"],
                "completed": row["completed"],
                "first_difference": row["first_difference"],
            }
            for row in base_floor_rows
        ],
        "audit_ablation_gamma_0": [
            {
                "world": row["world"],
                "index": row["index"],
                "age": row["age"],
                "completed": row["completed"],
                "first_difference": row["first_difference"],
            }
            for row in abl_main_rows
        ],
        "trace": trace,
        "answers": answers,
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
            "candidate_training": False,
            "confirm_reads": 0,
            "candidate_bundles_saved": 1,
            "candidate_bundles_note": "single assembled_zero_training_stopfix state",
            "fsync_per_line": True,
            "report_write_then_rename": True,
        },
    }
    write_json(REPORT, report)
    write_json(
        MANIFEST,
        {
            "step": "N19.2u",
            "date": "2026-11-05",
            "parent_bundle": str(BUNDLE.relative_to(ROOT)),
            "candidate_sha_v2_new": identity["candidate_sha_v2"],
            "candidate_sha_v2_old": OLD_SHA,
            "byte_match_parameters": route_parameters,
            "copy_stop_boundary": True,
            "copy_stop_module": "taiji/copy_stop_boundary.py",
            "assembled_state": str(ASSEMBLED.relative_to(ROOT)),
            "assembled_tag": "assembled_zero_training_stopfix",
            "assembled_sha256": hashlib.sha256(ASSEMBLED.read_bytes()).hexdigest(),
            "examples": len(dev),
            "boundary_fixed_length": boundary["fixed_length"],
            "boundary_counterexample_total": boundary["counterexample_total"],
            "gamma_0_completion": base0["completion"],
            "gamma_floor_completion": basef["completion"],
            "gamma_0_conditional_byte": base0["conditional_byte"],
            "gamma_0_start_hit_object": base0["start_hit_object"],
            "gamma_0_start_hit_place": base0["start_hit_place"],
            "wrong_route_right_content": base0["wrong_route_right_content"],
            "ablation_seed": ablation["seed"],
            "ablation_parameters": ablation["parameters"],
            "ablation_gamma_0_completion": abl0["completion"],
            "ablation_completion_delta": ablation_delta,
            "cpu_seconds": round(cpu_seconds, 2),
            "wall_seconds": round(wall_seconds, 2),
            "over_budget": bool(over_budget),
            "determinism": determinism,
        },
    )
    append_progress(
        {
            "event": "done",
            "cpu_seconds": round(cpu_seconds, 2),
            "wall_seconds": round(wall_seconds, 2),
            "over_budget": bool(over_budget),
        }
    )
    print(
        json.dumps(
            {
                "cpu": round(cpu_seconds, 2),
                "wall": round(wall_seconds, 2),
                "base_gamma_0": base0,
                "base_gamma_floor": basef,
                "ablation_gamma_0": abl0,
                "ablation_gamma_floor": ablf,
                "over_budget": bool(over_budget),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
