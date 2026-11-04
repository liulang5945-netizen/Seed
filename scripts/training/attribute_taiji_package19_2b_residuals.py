"""Residual failure attribution on the 19.5 step300 bundles (0 updates).

Every dev example that misses the whole-sentence gate (exact bytes + EOS) is put
into exactly one bucket by its first divergence, scanning the emitted bytes
against the target:

1. ``route_miss``          - at an entity's first content byte the hard route
                             picks a record other than the gold record;
2. ``start_miss``          - the record is right but the first content byte is
                             the wrong byte (landing spot wrong);
3. ``content_byte_wrong``  - first byte right, a later content byte wrong;
4. ``frame_byte_wrong``    - every content byte right, a template/fixed byte wrong;
5. ``eos_not_stopped``     - no EOS, or EOS before the target length;
6. ``order_or_extra``      - target bytes all match but extra bytes were emitted.

The whole-sentence ceiling under "start = 1.0 and copy = 1.0" is computed by
repairing every content position to its gold byte and re-checking the full
sequence, keeping the route gate: an example only counts if both entity starts
were routed to the gold record.  A second ceiling that also ignores the route is
reported as a sensitivity bound.  gamma is the generation weight and 1 - gamma
the copy weight; no training credit is inferred from gamma (M3).

``28031-A_route_lock`` is registered as engineering defect D16-6 (all binding and
head parameters NaN) and is excluded from every pooled statistic; it is not
counted as positive evidence of structural failure.

Run: ``python scripts/training/attribute_taiji_package19_2b_residuals.py``
"""

from __future__ import annotations

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
from taiji.binding_copy import MAX_PREDICTIONS, observe_goal  # noqa: E402
from taiji.living_core import EOS  # noqa: E402
from taiji.record_relation_core import decode_endpoint  # noqa: E402
from taiji.selection_routing import _advanced_state, routing_step  # noqa: E402
from taiji.selection_routing_checkpoint import load_routing_bundle, parameter_tensors  # noqa: E402

SEEDS = (28031, 28032)
ARMS = ("A_route_lock", "B_stop_lock", "C_unlocked")
BUNDLE_DIR = ROOT / "output/taiji_package19_compare"
REPORT = ROOT / "reports/taiji_package19_2b_residual_attribution_20261104.json"
MANIFEST = ROOT / "plans/manifests/taiji_package19_2b_residual_attribution_20261104.json"

CLASSES = (
    "route_miss",
    "start_miss",
    "content_byte_wrong",
    "frame_byte_wrong",
    "eos_not_stopped",
    "order_or_extra",
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def replay(probe, fixture: dict) -> dict:
    """Free decode, keeping the hard route at every emitted position."""
    candidate = probe.candidate
    prepared, context = probe.prepare(fixture)
    state, gate_index = observe_goal(candidate, prepared, gate_index=0)
    previous = EOS
    raw: list[int] = []
    routes: list[int] = []
    sources: list[str] = []
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
                arm=probe.arm,
                gate_index=gate_index,
                span_head=candidate.span_head,
                pointer_temperature=1.0,
                pointer_hard=True,
            )
            endpoint = decode_endpoint(step.probabilities, prefix=bytes(raw))
            if endpoint["eos"]:
                terminated = True
                break
            raw.append(endpoint["byte"])
            routes.append(int(torch.argmax(step.a_hard)))
            record = int(torch.argmax(step.a_hard[:-1])) if step.a_hard.numel() > 1 else -1
            sources.append(
                step.read.row_source[record] if 0 <= record < len(step.read.row_source) else "carry"
            )
            if len(raw) > 64:
                break
            state, _, gate_index = _advanced_state(candidate, prepared, step, endpoint["byte"])
            previous = endpoint["byte"]
    return {
        "raw": raw,
        "routes": routes,
        "sources": sources,
        "terminated": terminated,
    }


def payload_bytes(fixture: dict) -> tuple[int, ...]:
    """Target bytes without the trailing EOS sentinel."""
    target = tuple(fixture["target_bytes"])
    return target[:-1] if target and target[-1] == EOS else target


def classify(fixture: dict, decoded: dict) -> dict:
    target = payload_bytes(fixture)
    raw = decoded["raw"]
    mapping = probe_mod.content_positions(fixture)
    start_positions = {probe_mod.OBJECT_POSITIONS[0], probe_mod.PLACE_POSITIONS[0]}
    first_mismatch = None
    for position in range(min(len(raw), len(target))):
        if raw[position] != target[position]:
            first_mismatch = position
            break
    if first_mismatch is None:
        if len(raw) == len(target) and decoded["terminated"]:
            return {"bucket": "pass", "position": None, "route": None}
        if len(raw) > len(target):
            return {"bucket": "order_or_extra", "position": len(target), "route": None}
        return {"bucket": "eos_not_stopped", "position": len(raw), "route": None}
    position = first_mismatch
    if position in mapping:
        gold_record, _offset = mapping[position]
        route = decoded["routes"][position] if position < len(decoded["routes"]) else -1
        if position in start_positions:
            if route != gold_record:
                return {"bucket": "route_miss", "position": position, "route": route}
            return {"bucket": "start_miss", "position": position, "route": route}
        return {"bucket": "content_byte_wrong", "position": position, "route": route}
    return {"bucket": "frame_byte_wrong", "position": position, "route": None}


def frame_split(fixture: dict, decoded: dict, position: int) -> str:
    """Model generation vs decode/EOS logic for a frame substitution."""
    target = payload_bytes(fixture)
    raw = decoded["raw"]
    got = raw[position]
    neighbours = {
        target[offset] for offset in (position - 1, position + 1) if 0 <= offset < len(target)
    }
    copied = {target[offset] for offset in probe_mod.CONTENT_POSITIONS if offset < len(target)}
    if got in neighbours or got in copied:
        return "decode_eos_logic"
    return "model_generation"


def route_starts_ok(fixture: dict, decoded: dict) -> bool:
    mapping = probe_mod.content_positions(fixture)
    for position in (probe_mod.OBJECT_POSITIONS[0], probe_mod.PLACE_POSITIONS[0]):
        gold_record = mapping[position][0]
        route = decoded["routes"][position] if position < len(decoded["routes"]) else -1
        if route != gold_record:
            return False
    return True


def ceiling(fixture: dict, decoded: dict, require_route: bool) -> bool:
    """Would the sentence pass if every content byte were copied perfectly?"""
    target = payload_bytes(fixture)
    raw = decoded["raw"]
    if len(raw) != len(target) or not decoded["terminated"]:
        return False
    if require_route and not route_starts_ok(fixture, decoded):
        return False
    patched = list(raw)
    for position in probe_mod.CONTENT_POSITIONS:
        if position < len(patched):
            patched[position] = target[position]
    return tuple(patched) == target


def cell_key(seed: int, arm: str) -> str:
    return f"{seed}-{arm}"


def main() -> int:
    start_cpu = time.process_time()
    start_wall = time.perf_counter()
    payload = premise.build_payload()
    dev = probe_mod.dev_slice(premise.fixtures(payload, "dev"))

    cells: dict[str, dict] = {}
    contaminated: list[str] = []
    for seed in SEEDS:
        for arm in ARMS:
            path = BUNDLE_DIR / f"{seed}-{arm}-step300.pt"
            candidate = load_routing_bundle(path)[0]
            name = cell_key(seed, arm)
            tensors = parameter_tensors(candidate)
            nan_parameters = sum(int(torch.isnan(value).sum()) for value in tensors.values())
            if nan_parameters:
                contaminated.append(name)
                cells[name] = {
                    "seed": seed,
                    "arm": arm,
                    "examples": len(dev),
                    "passed": None,
                    "sentence_rate": None,
                    "failures": None,
                    "buckets": None,
                    "frame_split": None,
                    "ceiling_route": None,
                    "ceiling_free": None,
                    "skipped": True,
                    "nan_parameters": nan_parameters,
                    "examples_detail": [],
                }
                print(json.dumps({"cell": name, "skipped": True, "nan": nan_parameters}))
                continue
            probe = probe_mod.ProbeContext(candidate)
            probe.arm = arm
            per_example = []
            for fixture in dev:
                decoded = replay(probe, fixture)
                verdict = classify(fixture, decoded)
                bucket = verdict["bucket"]
                if bucket == "frame_byte_wrong":
                    verdict["frame_subtype"] = frame_split(fixture, decoded, verdict["position"])
                verdict["ceiling_route"] = ceiling(fixture, decoded, True)
                verdict["ceiling_free"] = ceiling(fixture, decoded, False)
                verdict["world"] = fixture["world"]
                verdict["index"] = fixture["index"]
                per_example.append(verdict)
            name = cell_key(seed, arm)
            counts = {bucket: 0 for bucket in CLASSES}
            frame = {"model_generation": 0, "decode_eos_logic": 0}
            passed = 0
            for verdict in per_example:
                bucket = verdict["bucket"]
                if bucket == "pass":
                    passed += 1
                else:
                    counts[bucket] += 1
                    if bucket == "frame_byte_wrong":
                        frame[verdict["frame_subtype"]] += 1
            total = len(per_example)
            table = {
                bucket: {
                    "count": counts[bucket],
                    "share": counts[bucket] / total if total else 0.0,
                }
                for bucket in CLASSES
            }
            cells[name] = {
                "seed": seed,
                "arm": arm,
                "examples": total,
                "passed": passed,
                "sentence_rate": passed / total if total else 0.0,
                "failures": total - passed,
                "buckets": table,
                "frame_split": frame,
                "ceiling_route": (
                    sum(1 for v in per_example if v["ceiling_route"]) / total if total else 0.0
                ),
                "ceiling_free": (
                    sum(1 for v in per_example if v["ceiling_free"]) / total if total else 0.0
                ),
                "examples_detail": [
                    {
                        "world": v["world"],
                        "index": v["index"],
                        "bucket": v["bucket"],
                        "position": v["position"],
                        "frame_subtype": v.get("frame_subtype"),
                        "route": v["route"],
                        "ceiling_route": v["ceiling_route"],
                        "ceiling_free": v["ceiling_free"],
                    }
                    for v in per_example
                ],
                "skipped": False,
                "nan_parameters": 0,
            }
            print(
                json.dumps(
                    {
                        "cell": name,
                        "sentence": round(cells[name]["sentence_rate"], 3),
                        "buckets": {k: round(v["share"], 3) for k, v in table.items()},
                        "ceiling_route": round(cells[name]["ceiling_route"], 3),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    healthy = [name for name in cells if name not in contaminated]

    def pooled(field: str) -> float:
        return statistics.fmean(cells[name][field] for name in healthy)

    by_arm = {}
    for arm in ARMS:
        members = [name for name in healthy if name.endswith(f"-{arm}")]
        if not members:
            continue
        by_arm[arm] = {
            "cells": members,
            "examples": sum(cells[name]["examples"] for name in members),
            "sentence_rate": statistics.fmean(cells[name]["sentence_rate"] for name in members),
            "ceiling_route": statistics.fmean(cells[name]["ceiling_route"] for name in members),
            "buckets": {
                bucket: {
                    "count": sum(cells[name]["buckets"][bucket]["count"] for name in members),
                    "share": statistics.fmean(
                        cells[name]["buckets"][bucket]["share"] for name in members
                    ),
                }
                for bucket in CLASSES
            },
            "frame_split": {
                key: sum(cells[name]["frame_split"][key] for name in members)
                for key in ("model_generation", "decode_eos_logic")
            },
        }
    pooled_buckets = {
        bucket: {
            "share": statistics.fmean(cells[name]["buckets"][bucket]["share"] for name in healthy),
            "count": sum(cells[name]["buckets"][bucket]["count"] for name in healthy),
        }
        for bucket in CLASSES
    }

    cpu = time.process_time() - start_cpu
    wall = time.perf_counter() - start_wall
    report = {
        "schema": "taiji-package19-2b-residual-attribution-v1",
        "step": "N19.2b / residual attribution (0 updates)",
        "date": "2026-11-04",
        "purpose": "bucket every non-passing dev example into exactly one death cause",
        "design": {
            "bundles": str(BUNDLE_DIR.relative_to(ROOT)),
            "bundles_are": "19.5 six step300 cells (no 19.2b training)",
            "examples_per_cell": len(dev),
            "decode": "free decode, pointer_hard, temperature 1.0",
            "bucket_rule": "first divergence against the target, content positions before frame positions",
            "sentence_gate": "exact bytes and EOS",
            "sensitivity": "ceiling_free ignores the route gate; ceiling_route keeps it",
            "saves": 0,
            "confirm_reads": 0,
            "adam_steps": 0,
        },
        "pooled_healthy_cells": healthy,
        "contaminated_cells": contaminated,
        "d16_6_registration": {
            "id": "D16-6 (candidate)",
            "cell": "28031-A_route_lock-step300.pt",
            "class": "engineering defect, not structural failure evidence",
            "symptom": "all binding/head parameters NaN; decode readings unusable",
            "origin": "NaN weight start in that cell's run; gamma readings missing; excluded from pooled gates",
            "action": "excluded from every pooled statistic; bundle kept untouched",
        },
        "by_cell": cells,
        "by_arm": by_arm,
        "pooled": {
            "sentence_rate": pooled("sentence_rate"),
            "ceiling_route": pooled("ceiling_route"),
            "ceiling_free": pooled("ceiling_free"),
            "buckets": pooled_buckets,
            "frame_split": {
                key: sum(cells[name]["frame_split"][key] for name in healthy)
                for key in ("model_generation", "decode_eos_logic")
            },
        },
        "cost": {"cpu_seconds": cpu, "wall_seconds": wall, "adam_steps": 0},
    }
    manifest = {
        "schema": "taiji-package19-2b-residual-attribution-manifest-v1",
        "step": "N19.2b residual attribution",
        "date": "2026-11-04",
        "report": str(REPORT.relative_to(ROOT)),
        "healthy_cells": healthy,
        "contaminated_cells": contaminated,
        "pooled_buckets": {k: round(v["share"], 4) for k, v in pooled_buckets.items()},
        "ceiling_route": round(report["pooled"]["ceiling_route"], 4),
        "ceiling_free": round(report["pooled"]["ceiling_free"], 4),
        "cost_cpu": cpu,
        "adam_steps": 0,
        "confirm_reads": 0,
    }
    write_json(REPORT, report)
    write_json(MANIFEST, manifest)
    print(
        json.dumps(
            {
                "report": str(REPORT),
                "pooled_sentence": round(report["pooled"]["sentence_rate"], 3),
                "ceiling_route": round(report["pooled"]["ceiling_route"], 3),
                "cpu": round(cpu, 1),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
