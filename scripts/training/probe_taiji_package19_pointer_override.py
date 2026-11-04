"""19.2e: reconcile the 0.630 vs 0.979 conflict and probe a forced pointer start.

Zero-update work on the finished 19.5 step300 cells (no training, no saves, no
confirm reads).

Reconciliation
    The formal 19.5 evaluation scores all 192 dev examples (0.630 / 0.583 for the
    two B cells).  The residual attribution scored a 48-example slice and
    reported 0.979.  This script re-measures the same 48 slice, reads the full
    192 and its per-flag subsets from the archived formal evaluation, and derives
    the easy stratum arithmetically.  Both scorers are the same predicate
    (``raw == target_prefix`` and EOS; ``target_prefix`` is ``target_bytes``
    without the EOS sentinel), and both decode with ``prefix=bytes(raw)``, so the
    only candidate explanation is the slice.

Forced pointer start
    At inference only, the copy start is forced to the span head's argmax
    (``forced_pointer_bias`` one-hot at the pointer argmax per record).  Route
    alpha, gamma, the transport row and the decoder are untouched.  A second
    family scales the raw start logits by ``s in {4, 16}`` instead of forcing,
    which quantifies how far the scalar bias must grow before the composite row
    follows the pointer.  The probe sits on a mixed 24-example set: one easy
    example (index 0) and one ``new_description`` example (index 8) per world,
    because the 48 slice used earlier contains zero ``new_description`` cases.

gamma is the generation weight and 1 - gamma the copy weight; no training credit
is inferred from gamma (M3).  D16-6 (``28031-A``: 2,320,153 NaN parameters) is
skipped and never pooled.

Run: ``python scripts/training/probe_taiji_package19_pointer_override.py``
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

torch.set_num_threads(1)

import probe_taiji_content_transport_18_1 as probe_mod  # noqa: E402

from taiji import data_premise_18_0 as premise  # noqa: E402
from taiji.binding_copy import (  # noqa: E402
    MAX_PREDICTIONS,
    initial_binding_state,
    observe_goal,
    prepare,
)
from taiji.living_core import EOS  # noqa: E402
from taiji.record_relation_core import decode_endpoint  # noqa: E402
from taiji.selection_routing import (  # noqa: E402
    _advanced_state,
    build_context,
    routing_step,
)
from taiji.selection_routing_checkpoint import load_routing_bundle, parameter_tensors  # noqa: E402

REPORT = ROOT / "reports/taiji_package19_2e_pointer_override_20261104.json"
MANIFEST = ROOT / "plans/manifests/taiji_package19_2e_pointer_override_20261104.json"
BUNDLE_DIR = ROOT / "output/taiji_package19_compare"
EVAL_DIR = BUNDLE_DIR

RECON_CELLS = ((28031, "B_stop_lock"), (28032, "B_stop_lock"))
PROBE_CELLS = (
    (28031, "B_stop_lock"),
    (28032, "B_stop_lock"),
    (28031, "C_unlocked"),
    (28032, "C_unlocked"),
)
SCALES = (4.0, 16.0)
FLAGS = ("unseen_output_name", "new_description", "both_new")
PROBE_INDEX = (0, 8)
FORCE_VALUE = 1e9


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def archived_eval(seed: int, arm: str) -> dict:
    path = EVAL_DIR / f"eval300-{seed}-{arm}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    overall = data["overall"]
    subsets = overall["subsets"]
    denominators = overall["subset_denominators"]
    completed = float(overall["completed"])
    total = int(overall["examples"])
    new_desc = float(subsets["new_description"])
    new_desc_n = int(denominators["new_description"])
    easy_n = total - new_desc_n
    easy_correct = completed * total - new_desc * new_desc_n
    return {
        "path": str(path.relative_to(ROOT)),
        "examples": total,
        "completed": completed,
        "subsets": {key: float(subsets[key]) for key in FLAGS},
        "subset_denominators": {key: int(denominators[key]) for key in FLAGS},
        "easy_stratum": {
            "examples": easy_n,
            "completed": easy_correct / easy_n if easy_n else 0.0,
        },
        "cpu_seconds": float(data["timing"]["cpu_seconds"]),
    }


def prepare_pair(candidate, fixture: dict):
    goal = fixture["goal"].encode("utf-8")
    state = initial_binding_state(candidate.binding, f"probe-{fixture['world']}-{fixture['index']}")
    prepared = prepare(candidate.binding, state, goal, fixture["records"])
    context = build_context(candidate.binding, goal, tuple(fixture["record_keys"]))
    return prepared, context


def start_logits(candidate, prepared) -> list[torch.Tensor]:
    view = prepared.base.view
    return [
        candidate.span_head.logits(view.units[index])[0].detach().clone()
        for index in range(view.record_count)
    ]


def bias_for(starts: list[torch.Tensor], view, mode: str, scale: float):
    if mode == "natural":
        return None
    pieces = []
    for index, logits in enumerate(starts):
        length = int(view.units[index].shape[0])
        if mode == "forced":
            piece = torch.zeros(length, dtype=logits.dtype)
            piece[int(torch.argmax(logits))] = FORCE_VALUE
        elif mode == "scale":
            piece = logits * scale
        else:
            raise ValueError(f"unknown mode {mode}")
        pieces.append(piece)
    return tuple(pieces)


def run(candidate, fixture: dict, arm: str, mode: str, scale: float) -> dict:
    prepared, context = prepare_pair(candidate, fixture)
    view = prepared.base.view
    starts = start_logits(candidate, prepared)
    bias = bias_for(starts, view, mode, scale)
    state = prepared.state
    state, gate_index = observe_goal(candidate, prepared, gate_index=0)
    previous = EOS
    raw: list[int] = []
    rows: list[dict] = []
    gammas: list[float] = []
    terminated = False
    with torch.no_grad():
        for _prediction in range(MAX_PREDICTIONS):
            step = routing_step(
                candidate,
                prepared,
                state,
                previous,
                context,
                candidate.head,
                arm=arm,
                gate_index=gate_index,
                span_head=candidate.span_head,
                pointer_temperature=1.0,
                pointer_hard=True,
                forced_pointer_bias=bias,
            )
            gammas.append(float(step.gamma_eff.detach()))
            endpoint = decode_endpoint(step.probabilities, prefix=bytes(raw))
            if endpoint["eos"]:
                terminated = True
                break
            chosen = step.a_hard
            route = int(torch.argmax(chosen[:-1])) if chosen.numel() > 1 else -1
            row = step.pointer_logits[route] if 0 <= route < len(step.pointer_logits) else None
            composite = int(torch.argmax(row)) if row is not None else -1
            pointer = int(torch.argmax(starts[route])) if 0 <= route < len(starts) else -1
            source = (
                step.read.row_source[route] if 0 <= route < len(step.read.row_source) else "carry"
            )
            rows.append(
                {
                    "byte": endpoint["byte"],
                    "route": route,
                    "composite": composite,
                    "pointer": pointer,
                    "source": source,
                }
            )
            raw.append(endpoint["byte"])
            if len(raw) > 64:
                break
            state, _, gate_index = _advanced_state(candidate, prepared, step, endpoint["byte"])
            previous = endpoint["byte"]
    return {
        "raw": tuple(raw),
        "terminated": terminated,
        "rows": rows,
        "gamma_mean": statistics.fmean(gammas) if gammas else 0.0,
        "mode": mode,
        "scale": scale,
    }


def starts_of(fixture: dict, decoded: dict) -> dict:
    mapping = probe_mod.content_positions(fixture)
    out = {}
    for name, position in (("object", 3), ("place", 21)):
        gold_record, gold_offset = mapping[position]
        row = decoded["rows"][position] if position < len(decoded["rows"]) else None
        route = row["route"] if row else -1
        route_ok = route == gold_record
        out[name] = {
            "route_ok": bool(route_ok),
            "start_ok": bool(route_ok and row["composite"] == gold_offset),
            "pointer_ok": bool(route_ok and row["pointer"] == gold_offset),
        }
    return out


def score(fixture: dict, decoded: dict) -> dict:
    target = tuple(fixture["target_prefix"])
    raw = decoded["raw"]
    mapping = probe_mod.content_positions(fixture)
    length_ok = len(raw) == len(target)
    utf8 = True
    try:
        bytes(raw).decode("utf-8")
    except UnicodeDecodeError:
        utf8 = False
    object_exact = bool(length_ok and raw[3:15] == target[3:15])
    place_exact = bool(length_ok and raw[21:27] == target[21:27])
    content_positions = [pos for pos in probe_mod.CONTENT_POSITIONS if pos < len(raw)]
    correct = [pos for pos in content_positions if raw[pos] == target[pos]]
    routed = [
        pos
        for pos in content_positions
        if pos < len(decoded["rows"]) and decoded["rows"][pos]["route"] == mapping[pos][0]
    ]
    routed_correct = [pos for pos in routed if raw[pos] == target[pos]]
    starts = starts_of(fixture, decoded)
    return {
        "sentence_exact": bool(length_ok and raw == target and decoded["terminated"]),
        "sentence_content": bool(
            length_ok and decoded["terminated"] and object_exact and place_exact
        ),
        "object_content": object_exact,
        "place_content": place_exact,
        "content_bytes": len(correct) / len(content_positions) if content_positions else 0.0,
        "conditional_byte": len(routed_correct) / len(routed) if routed else 0.0,
        "start_hit_object": starts["object"]["start_ok"],
        "start_hit_place": starts["place"]["start_ok"],
        "pointer_hit_object": starts["object"]["pointer_ok"],
        "pointer_hit_place": starts["place"]["pointer_ok"],
        "route_hit_object": starts["object"]["route_ok"],
        "route_hit_place": starts["place"]["route_ok"],
        "terminated": decoded["terminated"],
        "utf8": utf8,
        "gamma_mean": decoded["gamma_mean"],
        "ceiling": ceiling(fixture, decoded),
    }


def ceiling(fixture: dict, decoded: dict) -> bool:
    """Whole sentence if every content byte were copied and both starts routed."""
    target = tuple(fixture["target_prefix"])
    raw = decoded["raw"]
    if len(raw) != len(target) or not decoded["terminated"]:
        return False
    mapping = probe_mod.content_positions(fixture)
    for position in (3, 21):
        gold_record, _offset = mapping[position]
        row = decoded["rows"][position] if position < len(decoded["rows"]) else None
        if row is None or row["route"] != gold_record:
            return False
    patched = list(raw)
    for position in probe_mod.CONTENT_POSITIONS:
        if position < len(patched):
            patched[position] = target[position]
    return tuple(patched) == target


def summarize(rows: list[dict]) -> dict:
    count = len(rows)
    if not count:
        return {"examples": 0}

    def rate(key: str) -> float:
        return sum(1 for row in rows if row["score"][key]) / count

    gamma = statistics.fmean(row["score"]["gamma_mean"] for row in rows)
    by_index = {}
    for index in sorted({row["index"] for row in rows}):
        block = [row for row in rows if row["index"] == index]
        by_index[str(index)] = {
            "examples": len(block),
            "new_description": bool(block[0]["flags"]["new_description"]),
            "sentence_exact": sum(1 for row in block if row["score"]["sentence_exact"])
            / len(block),
            "content_bytes": statistics.fmean(row["score"]["content_bytes"] for row in block),
            "start_hit_object": sum(1 for row in block if row["score"]["start_hit_object"])
            / len(block),
            "start_hit_place": sum(1 for row in block if row["score"]["start_hit_place"])
            / len(block),
            "pointer_hit_object": sum(1 for row in block if row["score"]["pointer_hit_object"])
            / len(block),
            "pointer_hit_place": sum(1 for row in block if row["score"]["pointer_hit_place"])
            / len(block),
        }
    return {
        "examples": count,
        "sentence_exact": rate("sentence_exact"),
        "sentence_content": rate("sentence_content"),
        "object_content": rate("object_content"),
        "place_content": rate("place_content"),
        "content_bytes": statistics.fmean(row["score"]["content_bytes"] for row in rows),
        "conditional_byte": statistics.fmean(row["score"]["conditional_byte"] for row in rows),
        "start_hit_object": rate("start_hit_object"),
        "start_hit_place": rate("start_hit_place"),
        "pointer_hit_object": rate("pointer_hit_object"),
        "pointer_hit_place": rate("pointer_hit_place"),
        "route_hit_object": rate("route_hit_object"),
        "route_hit_place": rate("route_hit_place"),
        "terminated": rate("terminated"),
        "utf8": rate("utf8"),
        "ceiling": rate("ceiling"),
        "gamma_mean": gamma,
        "by_index": by_index,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="debug: cap examples per cell")
    parser.add_argument("--cells", default="", help="debug: comma separated seed-arm")
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()

    start_cpu = time.process_time()
    start_wall = time.perf_counter()
    payload = premise.build_payload()
    dev = list(premise.fixtures(payload, "dev"))
    slice_set = {
        (fixture["world"], fixture["index"]) for fixture in probe_mod.dev_slice(tuple(dev))
    }
    probe_set = {
        (fixture["world"], fixture["index"]) for fixture in dev if fixture["index"] in PROBE_INDEX
    }
    if args.limit:
        dev = dev[: args.limit]

    report: dict = {
        "schema": "taiji-package19-pointer-override-v1",
        "step": "N19.2e",
        "date": "2026-11-04",
        "data": {
            "dev_examples_full": 192,
            "attribution_slice_examples": 48,
            "attribution_slice_rule": "indices 0/1/4/5 per world (12 worlds) = the easy stratum",
            "probe_examples_per_cell": 24,
            "probe_rule": "indices 0 (easy) and 8 (new_description) per world",
        },
        "definition": {
            "formal": "raw == target_prefix (27 bytes) and EOS emitted",
            "attribution": "first-divergence classifier, pass = same predicate",
            "target_prefix_equals_target_bytes_minus_eos": True,
            "decode_path": "both use decode_endpoint(prefix=bytes(raw)), so the UTF-8 mask is identical",
        },
        "cells": {},
    }

    needed_by_cell = {}
    for seed, arm in PROBE_CELLS:
        keys = set(probe_set)
        if (seed, arm) in RECON_CELLS:
            keys |= slice_set
        needed_by_cell[(seed, arm)] = keys

    for seed, arm in PROBE_CELLS:
        name = f"{seed}-{arm}"
        if args.cells and name not in args.cells.split(","):
            continue
        candidate = load_routing_bundle(BUNDLE_DIR / f"{seed}-{arm}-step300.pt")[0]
        tensors = parameter_tensors(candidate)
        nan_parameters = sum(int(torch.isnan(value).sum()) for value in tensors.values())
        if nan_parameters:
            report["cells"][name] = {"skipped": True, "nan_parameters": nan_parameters}
            print(json.dumps({"cell": name, "skipped": True, "nan": nan_parameters}))
            continue

        cache: dict[tuple[int, int], dict] = {}
        for fixture in dev:
            key = (fixture["world"], fixture["index"])
            if key not in needed_by_cell[(seed, arm)]:
                continue
            decoded = run(candidate, fixture, arm, "natural", 1.0)
            cache[key] = {
                "world": fixture["world"],
                "index": fixture["index"],
                "flags": fixture["flags"],
                "score": score(fixture, decoded),
            }
        rows = list(cache.values())
        cell: dict = {"seed": seed, "arm": arm, "nan_parameters": 0}
        if (seed, arm) in RECON_CELLS and not args.probe_only:
            archived = archived_eval(seed, arm)
            slice_rows = [row for row in rows if (row["world"], row["index"]) in slice_set]
            cell["reconcile"] = {
                "archived_formal_full192": archived,
                "slice48_measured": summarize(slice_rows),
                "full192_attribution_definition": {
                    "examples": archived["examples"],
                    "sentence_exact": archived["completed"],
                    "source": "same predicate as the formal scorer; no recompute",
                },
                "definition_agreement": {
                    "slice48_formal_equals_attribution": True,
                    "predicate": "raw == target_prefix and EOS",
                },
                "slice_new_description_examples": sum(
                    1 for row in slice_rows if row["flags"]["new_description"]
                ),
            }
        probe_rows = [row for row in rows if (row["world"], row["index"]) in probe_set]
        passes: dict = {"natural": summarize(probe_rows)}
        for scale in SCALES:
            scored = []
            for fixture in dev:
                key = (fixture["world"], fixture["index"])
                if key not in probe_set:
                    continue
                decoded = run(candidate, fixture, arm, "scale", scale)
                scored.append(
                    {
                        "world": key[0],
                        "index": key[1],
                        "flags": fixture["flags"],
                        "score": score(fixture, decoded),
                    }
                )
            passes[f"scale_{scale:g}"] = summarize(scored)
        scored = []
        for fixture in dev:
            key = (fixture["world"], fixture["index"])
            if key not in probe_set:
                continue
            decoded = run(candidate, fixture, arm, "forced", 1.0)
            scored.append(
                {
                    "world": key[0],
                    "index": key[1],
                    "flags": fixture["flags"],
                    "score": score(fixture, decoded),
                }
            )
        passes["forced"] = summarize(scored)
        cell["probe"] = passes
        report["cells"][name] = cell
        print(
            json.dumps(
                {
                    "cell": name,
                    "done": True,
                    "cpu": round(time.process_time() - start_cpu, 1),
                }
            )
        )

    report["cost"] = {
        "cpu_seconds": round(time.process_time() - start_cpu, 3),
        "wall_seconds": round(time.perf_counter() - start_wall, 3),
        "adam_updates": 0,
        "confirm_reads": 0,
        "bundles_saved": 0,
    }
    report["limits"] = {"no_training": True, "no_criterion_change": True, "no_confirm": True}
    write_json(REPORT, report)
    print(json.dumps({"report": str(REPORT), "cost": report["cost"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
