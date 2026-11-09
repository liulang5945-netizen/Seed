"""R58: answer-shape extrapolation gate for the structural stop rule.

R57 closed Task1 confirm at 192/192 with ``sequence_stop`` (the end of the last
hop's declared field is also the end of the sequence).  That read only covers
one shape: two hops, 27 B answer, 12 B and 6 B name fields.  This gate asks
whether the rule is a *mechanism* or a Task1 constant.

Two halves:

* **structural** - hop counts 1/2/3 crossed with name fields of 3/6/9/12 B.
  The chain "record bytes -> ``field_bounds`` -> hop span -> ``sequence_stop``"
  must land exactly on the contract end.  No model, no forward pass.  A small
  boundary probe (names that contain the template delimiters) is reported
  separately and never enters the verdict.
* **model** - the Task1 two-hop template with synthetic name lengths
  (3/6/9/12 B), decoded freely up to the declared end.  Every stop edge reports
  the argmax byte, the EOS probability and its rank, and the copy mass, so the
  question "is the rule the only thing holding the cut" is answered by numbers.
  At least one shape must show a continuation byte with probability clearly
  above EOS; otherwise the gate does not prove the rule is load-bearing.

Verdicts: ``stop_rule_shape_general`` / ``stop_rule_partial_general`` /
``stop_rule_is_task1_constant``.  Discipline: dev/synthetic only, 0 training,
0 parameters, no edit to ``copy_stop_boundary``.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "scripts" / "training"
for entry in (ROOT, TRAINING):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import torch  # noqa: E402

torch.set_num_threads(1)

import probe_taiji_content_transport_18_1 as probe_mod  # noqa: E402

from taiji import data_premise_18_0 as premise  # noqa: E402
from taiji.binding_copy import MAX_PREDICTIONS, observe_goal  # noqa: E402
from taiji.copy_stop_boundary import CopyBoundaryError, field_bounds, sequence_stop  # noqa: E402
from taiji.living_core import EOS  # noqa: E402
from taiji.record_relation_core import decode_endpoint  # noqa: E402
from taiji.selection_routing_checkpoint import load_routing_bundle  # noqa: E402

MANIFEST = ROOT / "plans/manifests/taiji_stop_shape_r58_20261109.json"
REPORT = ROOT / "reports/taiji_stop_shape_r58_20261109.json"
PROGRESS = ROOT / "reports/taiji_stop_shape_r58_20261109_progress.jsonl"
BUNDLE = ROOT / "output/taiji_package19_compare/28031-B_stop_lock-step300.pt"

CPU_CAP = 200.0
WALL_CAP = 1800.0
SINGLE_RUN_ESTIMATE = 10.0
HOPS = (1, 2, 3)
NAME_BYTES = (3, 6, 9, 12)
# (object name bytes, place name bytes); the last entry is the Task1 shape itself.
MODEL_SHAPES = ((3, 3), (6, 6), (9, 9), (12, 12), (12, 6))
FROZEN_FILES = (
    "scripts/training/probe_taiji_stop_shape_r58_20261109.py",
    "scripts/training/probe_taiji_content_transport_18_1.py",
    "taiji/copy_stop_boundary.py",
    "taiji/binding_copy.py",
    "taiji/record_relation_core.py",
    "taiji/living_core.py",
    "taiji/data_premise_18_0.py",
)
DEFINITION = {
    "gate": "R58 answer-shape extrapolation for sequence_stop",
    "axes": {
        "hops": list(HOPS),
        "name_bytes": list(NAME_BYTES),
        "model_shapes": [list(pair) for pair in MODEL_SHAPES],
    },
    "criteria": {
        "structural": "record bytes -> field_bounds -> field length -> answer layout -> sequence_stop lands on the contract end for every hop/length cell",
        "stop": "every model shape terminates exactly at the declared end with zero bytes past it",
        "content": "byte-exactness of the decoded answer is reported but does not gate the stop rule",
        "adversarial": "at least one place-stop edge has a non-EOS argmax with probability >= 2x EOS",
        "per_edge": "every stop edge reports argmax byte, EOS probability and EOS rank, plus copy mass",
    },
    "exits": [
        "stop_rule_shape_general",
        "stop_rule_partial_general",
        "stop_rule_is_task1_constant",
    ],
    "scope": "dev/synthetic only; the sealed confirm split is not read",
    "estimate_rule": "D-64: single run x 3 (one run plus two replay margins) + report assembly",
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


def definition_digest() -> str:
    body = json.dumps(DEFINITION, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def file_hashes() -> dict:
    return {name: sha256_of(ROOT / name) for name in FROZEN_FILES}


CONTRACT = {
    1: ("取",),
    2: ("把", "放到"),
    3: ("把", "从", "搬到"),
}
GLYPHS = "甲乙丙丁戊己庚辛"


def names_for(hops: int, nbytes: int, offset: int = 0) -> list[str]:
    per = nbytes // 3
    return [
        "".join(GLYPHS[(offset + hop * per + k) % len(GLYPHS)] for k in range(per))
        for hop in range(hops)
    ]


def build_contract(parts: tuple[str, ...], names: list[str]) -> bytes:
    body = parts[0]
    for index, name in enumerate(names):
        body += name
        if index + 1 < len(parts):
            body += parts[index + 1]
    return body.encode("utf-8")


def record_for(name: str, desc: str, style: str) -> str:
    return f"{name}又叫{desc}。" if style == "又叫" else f"{desc}{style}{name}。"


def structural_cell(hops: int, nbytes: int) -> dict:
    """Record bytes -> field_bounds -> field length -> answer layout -> stop.

    The chain is deliberately split: ``field_bounds`` is the only thing that
    reads the record, and the answer layout is built from the recovered field
    *lengths*.  A cell passes when every recovered field length matches the name
    it was written from and ``sequence_stop`` lands on the contract end.
    """
    names = names_for(hops, nbytes)
    styles = ("是", "又叫", "对应")
    parts = CONTRACT[hops]
    spans = []
    for index, name in enumerate(names):
        style = styles[(index + nbytes // 3) % len(styles)]
        desc = f"说明{index}"
        raw = record_for(name, desc, style)
        try:
            start, end = field_bounds(raw, desc)
        except CopyBoundaryError as error:
            spans.append({"style": style, "record": raw, "error": str(error)})
            continue
        spans.append(
            {
                "style": style,
                "record": raw,
                "start": start,
                "end": end,
                "field_bytes": end - start,
            }
        )
    contract = build_contract(parts, names)
    hop_spans = []
    cursor = len(parts[0].encode("utf-8"))
    layout_ok = True
    for name, span in zip(names, spans, strict=True):
        length = len(name.encode("utf-8"))
        hop_spans.append({"start": cursor, "end": cursor + length})
        if "error" not in span and span["field_bytes"] != length:
            layout_ok = False
        cursor += length
        if len(hop_spans) < hops:
            cursor += len(parts[len(hop_spans)].encode("utf-8"))
    positions = tuple(range(hop_spans[-1]["start"], hop_spans[-1]["end"])) if hop_spans else ()
    declared_end = sequence_stop(positions) if positions else -1
    return {
        "hops": hops,
        "name_bytes": nbytes,
        "styles": [span["style"] for span in spans],
        "field_bounds": spans,
        "hop_spans": hop_spans,
        "contract_bytes": len(contract),
        "declared_end": declared_end,
        "exact": bool(
            declared_end == len(contract)
            and layout_ok
            and all("error" not in span for span in spans)
        ),
    }


def boundary_probe() -> list[dict]:
    cases = [
        ("是", "又叫人", "delimiter inside a stated name"),
        ("是", "甲。乙", "terminator inside a name"),
        ("又叫", "名", "aka style, name before the split"),
    ]
    out = []
    for style, name, note in cases:
        desc = "说明"
        raw = f"{name}又叫{desc}。" if style == "又叫" else f"{desc}{style}{name}。"
        try:
            start, end = field_bounds(raw, desc)
            field = raw.encode("utf-8")[start:end].decode("utf-8", "replace")
            out.append(
                {
                    "note": note,
                    "record": raw,
                    "recovered_field": field,
                    "expected_field": name,
                    "ok": field == name,
                }
            )
        except CopyBoundaryError as error:
            out.append({"note": note, "record": raw, "error": str(error), "ok": False})
    return out


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
            if offset < flat:
                override[offset] = 1.0
    return force, override


def byte_label(value: int) -> str:
    if value == EOS:
        return "<EOS>"
    return f"0x{value:02x}" if value < 0x20 or value > 0x7E else chr(value)


def top_candidates(probabilities: torch.Tensor, count: int = 3) -> list[dict]:
    values, indices = torch.topk(probabilities, k=count)
    return [
        {"byte": int(index), "label": byte_label(int(index)), "probability": float(value)}
        for value, index in zip(values.tolist(), indices.tolist(), strict=True)
    ]


def edge_report(probabilities: torch.Tensor, q_bytes, view, limits, prefix: bytes) -> dict:
    copy_mass = torch.zeros(probabilities.shape[0], dtype=torch.float64)
    if len(q_bytes):
        copy_mass = copy_mass.index_add(0, view.byte_ids, q_bytes)
    order = torch.argsort(probabilities, descending=True).tolist()
    candidates = top_candidates(probabilities)
    endpoint = decode_endpoint(probabilities, prefix=prefix)
    eos_probability = float(probabilities[EOS])
    top_probability = candidates[0]["probability"]
    return {
        "eos_probability": eos_probability,
        "eos_rank": order.index(EOS) + 1,
        "argmax_byte": candidates[0]["byte"],
        "argmax_label": candidates[0]["label"],
        "top": candidates,
        "pre_mask_label": byte_label(int(endpoint["pre_mask_byte"])),
        "masked": bool(endpoint["masked"]),
        "emitted_label": byte_label(int(endpoint["byte"])),
        "copy_mass_total": float(copy_mass.sum()),
        "copy_limit_active": max(int(value) for value in limits) if limits else -1,
        "continuation_desire": top_probability / eos_probability if eos_probability else None,
    }


@torch.no_grad()
def decode_shape(candidate, probe, fixture: dict, shape: dict, traced: bool) -> dict:
    """Decode the two-hop template with this shape's spans and forced content.

    The position *at* the declared end is probed before terminating, so the
    stop edge has a real distribution (argmax, EOS probability and rank)
    instead of only a "did it terminate" flag.
    """
    prepared, _context = probe.prepare(fixture)
    view = prepared.base.view
    state, gate_index = observe_goal(candidate, prepared, gate_index=0)
    previous = EOS
    raw: list[int] = []
    terminated = False
    traces: dict[str, dict] = {}
    record_count = view.record_count
    for _prediction in range(MAX_PREDICTIONS):
        position = len(raw)
        at_end = position >= shape["stop"]
        if shape["object_start"] <= position < shape["object_stop"]:
            route, start, anchor = (
                shape["object_route"],
                shape["object_name_offset"],
                position == shape["object_start"],
            )
        elif shape["place_start"] <= position < shape["place_stop"]:
            route, start, anchor = (
                shape["place_route"],
                shape["place_name_offset"],
                position == shape["place_start"],
            )
        else:
            route, start, anchor = -1, None, False
        limits = [-1] * record_count
        if position >= shape["object_start"]:
            limits[shape["object_route"]] = shape["object_copy_stop"]
        if position >= shape["place_start"]:
            limits[shape["place_route"]] = shape["place_copy_stop"]
        force, override = force_payload(view, route, start if anchor else None)
        probabilities, gamma_eff, content, next_h, q_bytes, q_null, read = probe.forced_step(
            prepared,
            state,
            previous,
            gate_index,
            force,
            override,
            0.0,
            copy_limits=tuple(limits),
        )
        if traced and position in (shape["object_stop"], shape["place_stop"]):
            payload = edge_report(probabilities, q_bytes, view, tuple(limits), bytes(raw))
            payload["position"] = position
            payload["at_declared_end"] = at_end
            payload["gamma_eff"] = float(gamma_eff)
            payload["q_null"] = float(q_null)
            traces[str(position)] = payload
        if at_end:
            terminated = True
            break
        endpoint = decode_endpoint(probabilities, prefix=bytes(raw))
        if endpoint["eos"]:
            terminated = True
            break
        byte = int(endpoint["byte"])
        raw.append(byte)
        if len(raw) > 96:
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
    return {"raw": tuple(raw), "terminated": terminated, "traces": traces}


def with_raw(record, raw: bytes):
    clone = copy.copy(record)
    try:
        clone.raw = raw
    except (AttributeError, dataclasses.FrozenInstanceError):
        clone = dataclasses.replace(record, raw=raw)
    return clone


def load_candidate():
    if not BUNDLE.exists():
        raise RuntimeError(f"parent bundle missing: {BUNDLE}")
    candidate = load_routing_bundle(BUNDLE)[0]
    probe = probe_mod.ProbeContext(candidate)
    probe.arm = "B_stop_lock"
    return candidate, probe


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()

    started_cpu = time.process_time()
    started_wall = time.perf_counter()

    if args.freeze:
        if MANIFEST.exists():
            raise RuntimeError(f"preregistration already exists: {MANIFEST}")
        write_json(
            MANIFEST,
            {
                "schema": "taiji-stop-shape-r58-manifest-v1",
                "step": "R58",
                "date": "2026-11-09",
                "definition": DEFINITION,
                "definition_digest": definition_digest(),
                "frozen_files": file_hashes(),
                "bundle": str(BUNDLE.relative_to(ROOT)),
                "bundle_sha256": sha256_of(BUNDLE),
                "cost": {
                    "single_run_estimate": SINGLE_RUN_ESTIMATE,
                    "estimate_with_replays": SINGLE_RUN_ESTIMATE * 3,
                    "cpu_cap": CPU_CAP,
                    "wall_cap": WALL_CAP,
                },
                "immutability": "written once before the run; never rewritten",
                "sealed_split": "not read",
            },
        )
        print(json.dumps({"frozen": str(MANIFEST)}, ensure_ascii=False))
        return 0

    if not MANIFEST.exists():
        raise RuntimeError("preregistration missing; run --freeze first")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["definition_digest"] != definition_digest():
        raise RuntimeError("definition digest drift")
    drifted = [
        name for name, value in manifest["frozen_files"].items() if sha256_of(ROOT / name) != value
    ]
    if drifted:
        raise RuntimeError(f"frozen input drift: {drifted}")
    if manifest["bundle_sha256"] != sha256_of(BUNDLE):
        raise RuntimeError("bundle drift")
    if PROGRESS.exists() and PROGRESS.stat().st_size:
        events = [
            json.loads(line)
            for line in PROGRESS.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if any(event.get("event") == "finish" for event in events):
            raise RuntimeError("R58 already finished")
    else:
        PROGRESS.write_text("", encoding="utf-8")
    append_progress({"event": "start", "definition_digest": definition_digest()})

    structural = [structural_cell(hops, nbytes) for hops in HOPS for nbytes in NAME_BYTES]
    probe_cases = boundary_probe()
    append_progress(
        {
            "event": "structural",
            "cells": len(structural),
            "exact": sum(1 for cell in structural if cell["exact"]),
        }
    )

    base = premise.build_payload()
    pools = premise.split_pools()
    worlds = [
        premise.build_world("dev", world, pools) for world in range(premise.SPLIT_WORLDS["dev"])
    ]
    materialised = dict(base)
    materialised["worlds"] = [
        {key: value for key, value in world.items() if key != "examples"} for world in worlds
    ]
    materialised["examples"] = [example for world in worlds for example in world["examples"]]
    dev = tuple(premise.fixtures(materialised, "dev"))

    candidate, probe = load_candidate()
    shapes: list[dict] = []
    for index, (object_bytes, place_bytes) in enumerate(MODEL_SHAPES):
        fixture = dev[index]
        object_name = names_for(1, object_bytes)[0]
        place_name = names_for(1, place_bytes, offset=4)[0]
        object_raw = f"{fixture['object_description']}是{object_name}。".encode()
        place_raw = f"{fixture['place_description']}对应{place_name}。".encode()
        records = list(fixture["records"])
        records[0] = with_raw(records[0], object_raw)
        records[1] = with_raw(records[1], place_raw)
        shaped = dict(fixture)
        shaped["records"] = records
        object_copy_start, object_copy_stop = field_bounds(
            object_raw.decode("utf-8"), fixture["object_description"]
        )
        place_copy_start, place_copy_stop = field_bounds(
            place_raw.decode("utf-8"), fixture["place_description"]
        )
        object_start = 3  # "把"
        object_stop = object_start + len(object_name.encode("utf-8"))
        place_start = object_stop + 6  # "放到"
        place_stop = place_start + len(place_name.encode("utf-8"))
        target = f"把{object_name}放到{place_name}".encode()
        shape = {
            "object_bytes": object_bytes,
            "place_bytes": place_bytes,
            "object_name": object_name,
            "place_name": place_name,
            "object_start": object_start,
            "object_stop": object_stop,
            "place_start": place_start,
            "place_stop": place_stop,
            "stop": place_stop,
            "object_route": 0,
            "place_route": 1,
            "object_name_offset": object_copy_start,
            "place_name_offset": place_copy_start,
            "object_copy_stop": object_copy_stop,
            "place_copy_stop": place_copy_stop,
            "field_lengths_ok": (
                object_copy_stop - object_copy_start == len(object_name.encode("utf-8"))
                and place_copy_stop - place_copy_start == len(place_name.encode("utf-8"))
            ),
            "target": target,
            "target_bytes": len(target),
        }
        decoded = decode_shape(candidate, probe, shaped, shape, traced=True)
        raw = decoded["raw"]
        row = {
            "object_bytes": object_bytes,
            "place_bytes": place_bytes,
            "role": (
                "control_task1_shape" if (object_bytes, place_bytes) == (12, 6) else "extension"
            ),
            "object_name": object_name,
            "place_name": place_name,
            "target_text": target.decode("utf-8"),
            "decoded_text": bytes(raw).decode("utf-8", "replace"),
            "object_stop": shape["object_stop"],
            "declared_end": shape["stop"],
            "field_lengths_ok": shape["field_lengths_ok"],
            "answer_bytes": len(raw),
            "zero_extra": len(raw) == shape["stop"] and decoded["terminated"],
            "byte_exact": tuple(raw) == tuple(target),
            "traces": decoded["traces"],
        }
        shapes.append(row)
        append_progress(
            {
                "event": "shape",
                "object_bytes": object_bytes,
                "place_bytes": place_bytes,
                "zero_extra": row["zero_extra"],
                "byte_exact": row["byte_exact"],
                "cpu_seconds": round(time.process_time() - started_cpu, 4),
            }
        )

    structural_pass = all(cell["exact"] for cell in structural)
    extension_rows = [row for row in shapes if row["role"] == "extension"]
    control_rows = [row for row in shapes if row["role"] == "control_task1_shape"]
    stop_pass = all(row["zero_extra"] for row in extension_rows)
    content_pass = all(row["byte_exact"] for row in extension_rows)
    control_pass = all(row["zero_extra"] and row["byte_exact"] for row in control_rows)
    edge_rows: list[dict] = []
    for row in shapes:
        for edge_name, position in (
            ("object_stop", row["object_stop"]),
            ("place_stop", row["declared_end"]),
        ):
            trace = row["traces"].get(str(position)) if position is not None else None
            if trace is None:
                continue
            edge_rows.append(
                {
                    "object_bytes": row["object_bytes"],
                    "place_bytes": row["place_bytes"],
                    "role": row["role"],
                    "edge": edge_name,
                    "position": position,
                    "argmax_label": trace["argmax_label"],
                    "argmax_byte": trace["argmax_byte"],
                    "masked": trace["masked"],
                    "emitted_label": trace["emitted_label"],
                    "eos_probability": trace["eos_probability"],
                    "eos_rank": trace["eos_rank"],
                    "continuation_desire": trace["continuation_desire"],
                    "copy_mass_total": trace["copy_mass_total"],
                }
            )
    load_bearing = [
        edge
        for edge in edge_rows
        if edge["role"] == "extension"
        and edge["edge"] == "place_stop"
        and edge["argmax_byte"] != EOS
    ]
    adversarial = [
        edge
        for edge in load_bearing
        if not edge["masked"] and (edge["continuation_desire"] or 0.0) >= 2.0
    ]
    adversarial_pass = bool(adversarial)
    failing_axes = [
        name
        for name, ok in (
            ("structural", structural_pass),
            ("stop_at_declared_end", stop_pass),
            ("content_byte_exact", content_pass),
            ("control_task1_shape", control_pass),
            ("adversarial_edge", adversarial_pass),
        )
        if not ok
    ]
    if not structural_pass:
        verdict = "stop_rule_is_task1_constant"
    elif not failing_axes:
        verdict = "stop_rule_shape_general"
    else:
        verdict = "stop_rule_partial_general"

    cpu_seconds = time.process_time() - started_cpu
    wall_seconds = time.perf_counter() - started_wall
    report = {
        "schema": "taiji-stop-shape-r58-report-v1",
        "step": "R58",
        "date": "2026-11-09",
        "relation_to_main_task": (
            "决定 Task1 的 192/192 能否外推到其它答案形状；本步不改变任何已有读数，"
            "只界定 192/192 的适用范围。"
        ),
        "preregistration": str(MANIFEST.relative_to(ROOT)),
        "preregistration_sha256": sha256_of(MANIFEST),
        "scope": {"sealed_split": "not read", "split": "dev", "examples_used": len(shapes)},
        "structural": {
            "cells": structural,
            "exact": sum(1 for cell in structural if cell["exact"]),
            "total": len(structural),
            "boundary_probe": probe_cases,
        },
        "model_shapes": shapes,
        "stop_edges": edge_rows,
        "gates": {
            "structural_pass": structural_pass,
            "stop_pass": stop_pass,
            "content_pass": content_pass,
            "control_pass": control_pass,
            "adversarial_pass": adversarial_pass,
            "failing_axes": failing_axes,
            "load_bearing_edges": [
                {
                    "object_bytes": edge["object_bytes"],
                    "place_bytes": edge["place_bytes"],
                    "argmax_label": edge["argmax_label"],
                    "eos_probability": edge["eos_probability"],
                    "eos_rank": edge["eos_rank"],
                    "continuation_desire": edge["continuation_desire"],
                }
                for edge in load_bearing
            ],
            "adversarial_shapes": [
                [edge["object_bytes"], edge["place_bytes"]] for edge in adversarial
            ],
        },
        "outcome": {
            "verdict": verdict,
            "note": (
                "1-hop and 3-hop are structural only: the model was never trained on those "
                "templates, so no model-side claim is made for them"
            ),
        },
        "cost": {
            "cpu_seconds": round(cpu_seconds, 2),
            "wall_seconds": round(wall_seconds, 2),
            "single_run_estimate": SINGLE_RUN_ESTIMATE,
            "estimate_with_replays": SINGLE_RUN_ESTIMATE * 3,
            "cpu_cap": CPU_CAP,
            "wall_cap": WALL_CAP,
            "over_budget": cpu_seconds > CPU_CAP or wall_seconds > WALL_CAP,
        },
        "discipline": {
            "training_runs": 0,
            "optimizer_steps": 0,
            "parameters_added": 0,
            "graph_changes": 0,
            "confirm_reads": 0,
        },
        "closed_state": {"frozen_files_unchanged": file_hashes() == manifest["frozen_files"]},
    }
    write_json(REPORT, report)
    append_progress({"event": "finish", "verdict": verdict, "cpu_seconds": round(cpu_seconds, 4)})
    print(
        json.dumps(
            {
                "verdict": verdict,
                "structural": f"{report['structural']['exact']}/{report['structural']['total']}",
                "model_zero_extra": [row["zero_extra"] for row in shapes],
                "model_byte_exact": [row["byte_exact"] for row in shapes],
                "shapes": [[row["object_bytes"], row["place_bytes"]] for row in shapes],
                "adversarial": report["gates"]["adversarial_shapes"],
                "cost": report["cost"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
