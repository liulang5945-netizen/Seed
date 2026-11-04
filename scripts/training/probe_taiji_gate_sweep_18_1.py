"""N18.1 gate-sweep suite: four read-only probes on the six step300 bundles.

Zero updates, zero confirm reads, no training.  Each cell re-uses the frozen
18.1 forward path and only overrides the gamma gate:

* ``sweep``   - model alpha/beta route, gamma forced to {0.6,0.5,0.4,0.3,0.2,0.0};
* ``c_gamma0``- gold record + every content position pinned + gamma = 0, the
  only point where both the gate and the pointer are optimal;
* ``utf8``    - pure generation (gamma = 1), copy-dominant decode (gamma = 0)
  and the correct-routing copy control (``c_gamma0``) on the same examples,
  to locate whether illegal bytes come from the generate path or the copy
  re-assembly;
* per-position error distribution under ``b_pure`` and ``c_gamma0``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

torch.set_num_threads(1)

from dataclasses import replace  # noqa: E402

import probe_taiji_content_transport_18_1 as probe_mod  # noqa: E402

from taiji.binding_copy import MAX_COVERAGE, observe_goal  # noqa: E402
from taiji.binding_copy_scoring import score_generation  # noqa: E402
from taiji.living_core import EOS  # noqa: E402
from taiji.record_relation_core import MAX_PREDICTIONS, decode_endpoint  # noqa: E402
from taiji.selection_routing import _advanced_state, routing_step  # noqa: E402

SEEDS = (28031, 28032)
ARMS = ("A_route_lock", "B_stop_lock", "C_unlocked")
SWEEP = (0.6, 0.5, 0.4, 0.3, 0.2, 0.0)
OUTPUT_DIR = Path("output/taiji_selection_routing_18_1")
REPORT = Path("reports/taiji_gate_sweep_18_1_20261104.json")
MANIFEST = Path("plans/manifests/taiji_gate_sweep_18_1_20261104.json")
PREREG = Path("plans/manifests/taiji_byte_gate_prereg_20261104.json")

OBJECT_POSITIONS = probe_mod.OBJECT_POSITIONS
PLACE_POSITIONS = probe_mod.PLACE_POSITIONS
CONTENT_POSITIONS = probe_mod.CONTENT_POSITIONS


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def free_decode_gamma(probe, fixture: dict, gamma_override: float, keep_raw: bool = False) -> dict:
    """Free decode on the frozen routing step with a post-hoc gamma override.

    The step, the read and the arm-locked quantities are exactly the frozen
    ``routing_step`` path (so alpha/beta stay the model's own route for every
    arm); only the probability mix is rebuilt with the forced gamma.
    """
    prepared, context = probe.prepare(fixture)
    state, gate_index = observe_goal(probe.candidate, prepared, gate_index=0)
    target = tuple(fixture["target_bytes"])
    previous = EOS
    raw: list[int] = []
    gammas: list[float] = []
    terminated = False
    with torch.no_grad():
        for _ in range(MAX_PREDICTIONS):
            step = routing_step(
                probe.candidate,
                prepared,
                state,
                previous,
                context,
                probe.candidate.head,
                arm=probe.arm,
                gate_index=gate_index,
                span_head=probe.candidate.span_head,
                pointer_temperature=1.0,
                pointer_hard=True,
            )
            gamma_eff = torch.as_tensor(gamma_override, dtype=step.gamma_eff.dtype)
            copy_byte_mass = torch.zeros(MAX_COVERAGE, dtype=torch.float64)
            if len(step.q_bytes):
                copy_byte_mass = copy_byte_mass.index_add(
                    0, prepared.base.view.byte_ids, step.q_bytes
                )
            probabilities = (gamma_eff + (1.0 - gamma_eff) * step.q_null) * step.generated + (
                1.0 - gamma_eff
            ) * copy_byte_mass
            gammas.append(float(gamma_eff.detach()))
            endpoint = decode_endpoint(probabilities)
            if endpoint["eos"]:
                terminated = True
                break
            byte = endpoint["byte"]
            raw.append(byte)
            custom = replace(step, probabilities=probabilities, gamma_eff=gamma_eff)
            state, _coverage, gate_index = _advanced_state(probe.candidate, prepared, custom, byte)
            previous = byte
    scored = score_generation(bytes(raw), fixture["target_prefix"], terminated)
    object_bytes = fixture["object_name"].encode("utf-8")
    place_bytes = fixture["place_name"].encode("utf-8")
    object_hits = sum(
        1
        for offset, position in enumerate(OBJECT_POSITIONS)
        if position < len(raw) and raw[position] == target[position]
    )
    place_hits = sum(
        1
        for offset, position in enumerate(PLACE_POSITIONS)
        if position < len(raw) and raw[position] == target[position]
    )
    return {
        "world": fixture["world"],
        "index": fixture["index"],
        "completed": bool(scored["completed"]),
        "both_names_in_output": object_bytes in bytes(raw) and place_bytes in bytes(raw),
        "unseen_flag": bool(fixture["flags"]["unseen_output_name"]),
        "object_byte_hits": object_hits,
        "place_byte_hits": place_hits,
        "gamma_effective": statistics.fmean(gammas) if gammas else gamma_override,
        "terminated": terminated,
        "generated_bytes": len(raw),
        "raw": bytes(raw) if keep_raw else None,
    }


def position_hits(raw: list[int], target: tuple[int, ...]) -> dict[int, int]:
    return {
        position: int(position < len(raw) and raw[position] == target[position])
        for position in CONTENT_POSITIONS
    }


def forced_with_positions(probe, fixture: dict, mode: str, gamma_override: float) -> dict:
    out = probe_mod.forced_condition(probe, fixture, mode, gamma_override)
    out["position_hits"] = position_hits(out["raw"], tuple(fixture["target_bytes"]))
    if mode == "c":
        # The c condition is the correct-routing copy control: record and every
        # content position are pinned, gamma = 0.  Its raw is returned as a
        # 27-byte prefix (the target is 28 bytes), so an incomplete trailing
        # sequence is not counted as illegal.
        out["utf8"] = utf8_scan(bytes(out["raw"]), allow_truncated=True)
    out.pop("raw", None)
    return out


def utf8_scan(data: bytes, allow_truncated: bool = False) -> dict:
    """Locate illegal UTF-8 sequence starts; returns legality and positions."""
    bad: list[int] = []
    index = 0
    length = len(data)
    while index < length:
        byte = data[index]
        if byte < 0x80:
            index += 1
            continue
        if 0xC2 <= byte <= 0xDF:
            need = 1
        elif 0xE0 <= byte <= 0xEF:
            need = 2
        elif 0xF0 <= byte <= 0xF4:
            need = 3
        else:
            bad.append(index)
            index += 1
            continue
        if index + need >= length:
            if not allow_truncated:
                bad.append(index)
            break
        if not all(0x80 <= data[index + step] <= 0xBF for step in range(1, need + 1)):
            bad.append(index)
            index += 1
            continue
        index += need + 1
    return {"legal": not bad, "illegal_positions": bad}


def run_cell(seed: int, arm: str, checkpoint: str, limit: int | None = None) -> dict:
    start_cpu = time.process_time()
    start_wall = time.perf_counter()
    path = OUTPUT_DIR / f"{seed}-{arm}-{checkpoint}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    candidate, _optimizer, _state, update, _payload = probe_mod.load_routing_bundle(path)
    probe = probe_mod.ProbeContext(candidate)
    probe.arm = arm
    payload = probe_mod.premise.build_payload()
    dev = probe_mod.dev_slice(probe_mod.premise.fixtures(payload, "dev"))
    if limit is not None:
        dev = dev[:limit]
    rows = []
    for fixture in dev:
        sweep = {}
        for gamma in SWEEP:
            entry = free_decode_gamma(probe, fixture, gamma)
            entry.pop("raw", None)
            sweep[f"{gamma:.1f}"] = entry
        b_pure = forced_with_positions(probe, fixture, "b", 0.0)
        c_gamma0 = forced_with_positions(probe, fixture, "c", 0.0)
        generate_only = free_decode_gamma(probe, fixture, 1.0, keep_raw=True)
        copy_only = free_decode_gamma(probe, fixture, 0.0, keep_raw=True)
        rows.append(
            {
                "world": fixture["world"],
                "index": fixture["index"],
                "flags": dict(fixture["flags"]),
                "sweep": sweep,
                "b_pure": b_pure,
                "c_gamma0": c_gamma0,
                "utf8_generate": {
                    "raw_bytes": len(generate_only["raw"]),
                    **utf8_scan(generate_only["raw"]),
                },
                "utf8_copy": {
                    "raw_bytes": len(copy_only["raw"]),
                    **utf8_scan(copy_only["raw"]),
                },
            }
        )

    def sweep_summary(gamma_key: str) -> dict:
        entries = [row["sweep"][gamma_key] for row in rows]
        completed = sum(1 for entry in entries if entry["completed"])
        copied = sum(1 for entry in entries if entry["both_names_in_output"])
        unseen = [entry for entry in entries if entry["unseen_flag"]]
        unseen_completed = sum(1 for entry in unseen if entry["completed"])
        object_hits = sum(entry["object_byte_hits"] for entry in entries)
        place_hits = sum(entry["place_byte_hits"] for entry in entries)
        return {
            "gamma_effective": statistics.fmean(entry["gamma_effective"] for entry in entries),
            "overall": completed / len(entries) if entries else 0.0,
            "new_name_copied": copied / len(entries) if entries else 0.0,
            "unseen_output_name": unseen_completed / len(unseen) if unseen else 0.0,
            "object_byte_rate": (
                object_hits / (len(entries) * len(OBJECT_POSITIONS)) if entries else 0.0
            ),
            "place_byte_rate": (
                place_hits / (len(entries) * len(PLACE_POSITIONS)) if entries else 0.0
            ),
            "content_byte_rate": (
                (object_hits + place_hits) / (len(entries) * len(CONTENT_POSITIONS))
                if entries
                else 0.0
            ),
            "terminated": sum(1 for entry in entries if entry["terminated"]),
            "examples": len(entries),
            "unseen_denominator": len(unseen),
        }

    def position_summary(key: str) -> dict:
        entries = [row[key] for row in rows]
        object_rates = {
            offset + 1: sum(entry["position_hits"][position] for entry in entries) / len(entries)
            for offset, position in enumerate(OBJECT_POSITIONS)
        }
        place_rates = {
            offset + 1: sum(entry["position_hits"][position] for entry in entries) / len(entries)
            for offset, position in enumerate(PLACE_POSITIONS)
        }
        return {
            "object_positions": object_rates,
            "place_positions": place_rates,
            "object_first": object_rates[1],
            "object_last": object_rates[len(OBJECT_POSITIONS)],
            "place_first": place_rates[1],
            "place_last": place_rates[len(PLACE_POSITIONS)],
            "content_byte_rate": sum(entry["hits"] for entry in entries)
            / (len(entries) * len(CONTENT_POSITIONS)),
        }

    def utf8_summary(key: str) -> dict:
        entries = [row[key] for row in rows]
        return _utf8_stats(entries)

    def _utf8_stats(entries: list[dict]) -> dict:
        legal = sum(1 for entry in entries if entry["legal"])
        histogram: dict[str, int] = {}
        for entry in entries:
            for position in entry["illegal_positions"]:
                histogram[str(position)] = histogram.get(str(position), 0) + 1
        return {
            "legal_rate": legal / len(entries) if entries else 0.0,
            "legal_examples": legal,
            "examples": len(entries),
            "illegal_position_histogram": histogram,
        }

    return {
        "seed": seed,
        "arm": arm,
        "checkpoint": checkpoint,
        "update": update,
        "bundle": {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)},
        "examples": len(rows),
        "sweep": {key: sweep_summary(key) for key in (f"{gamma:.1f}" for gamma in SWEEP)},
        "c_gamma0": {
            "content_byte_rate": sum(row["c_gamma0"]["hits"] for row in rows)
            / (len(rows) * len(CONTENT_POSITIONS)),
            "object_byte_rate": sum(
                sum(row["c_gamma0"]["position_hits"][position] for position in OBJECT_POSITIONS)
                for row in rows
            )
            / (len(rows) * len(OBJECT_POSITIONS)),
            "place_byte_rate": sum(
                sum(row["c_gamma0"]["position_hits"][position] for position in PLACE_POSITIONS)
                for row in rows
            )
            / (len(rows) * len(PLACE_POSITIONS)),
            "gamma_effective": statistics.fmean(row["c_gamma0"]["gamma_mean"] for row in rows),
        },
        "positions": {
            "b_pure": position_summary("b_pure"),
            "c_gamma0": position_summary("c_gamma0"),
        },
        "utf8": {
            "generate_only": utf8_summary("utf8_generate"),
            "copy_only": utf8_summary("utf8_copy"),
            "c_gamma0": _utf8_stats([row["c_gamma0"]["utf8"] for row in rows]),
        },
        "rows": rows,
        "cost": {
            "cpu_seconds": time.process_time() - start_cpu,
            "wall_seconds": time.perf_counter() - start_wall,
            "adam_steps": 0,
        },
    }


def mean_of(results: list[dict], path: list[str]) -> float:
    values = []
    for result in results:
        node = result
        for key in path:
            node = node[key]
        values.append(float(node))
    return statistics.fmean(values) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="step300", choices=sorted(probe_mod.CHECKPOINTS))
    parser.add_argument("--cells", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--prereg-only",
        action="store_true",
        help="re-derive the prereg and manifest from the existing report, no forward pass",
    )
    arguments = parser.parse_args()
    if arguments.prereg_only:
        report = json.loads((ROOT / REPORT).read_text(encoding="utf-8"))
        write_json(ROOT / PREREG, build_prereg(report["pooled"], REPORT))
        write_json(ROOT / MANIFEST, build_manifest(report, report["checkpoint"]))
        print(json.dumps({"prereg": str(PREREG), "manifest": str(MANIFEST)}))
        return 0
    cells = [(seed, arm) for seed in SEEDS for arm in ARMS]
    if arguments.cells:
        cells = [(int(seed), arm) for seed, arm in (value.split(":") for value in arguments.cells)]

    results = []
    failures = []
    with ProcessPoolExecutor(max_workers=6) as pool:
        futures = {
            pool.submit(run_cell, seed, arm, arguments.checkpoint, arguments.limit): (seed, arm)
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
                            "c_g0": round(result["c_gamma0"]["content_byte_rate"], 3),
                            "sweep0": round(result["sweep"]["0.0"]["overall"], 3),
                            "utf8_gen": round(result["utf8"]["generate_only"]["legal_rate"], 3),
                            "utf8_copy": round(result["utf8"]["copy_only"]["legal_rate"], 3),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            except Exception:  # noqa: BLE001
                failures.append({"seed": seed, "arm": arm, "traceback": traceback.format_exc()})
    if failures:
        write_json(OUTPUT_DIR / "probe_gate_sweep_failures.json", {"failures": failures})
        return 1
    results.sort(key=lambda item: (item["seed"], item["arm"]))
    total_cpu = sum(result["cost"]["cpu_seconds"] for result in results)

    pooled = {
        "sweep": {
            key: {
                "overall": mean_of(results, ["sweep", key, "overall"]),
                "new_name_copied": mean_of(results, ["sweep", key, "new_name_copied"]),
                "unseen_output_name": mean_of(results, ["sweep", key, "unseen_output_name"]),
                "object_byte_rate": mean_of(results, ["sweep", key, "object_byte_rate"]),
                "place_byte_rate": mean_of(results, ["sweep", key, "place_byte_rate"]),
                "content_byte_rate": mean_of(results, ["sweep", key, "content_byte_rate"]),
            }
            for key in (f"{gamma:.1f}" for gamma in SWEEP)
        },
        "c_gamma0": {
            "overall": 0.0,
            "content_byte_rate": mean_of(results, ["c_gamma0", "content_byte_rate"]),
            "object_byte_rate": mean_of(results, ["c_gamma0", "object_byte_rate"]),
            "place_byte_rate": mean_of(results, ["c_gamma0", "place_byte_rate"]),
        },
        "positions": {
            "b_pure": {
                "object_first": mean_of(results, ["positions", "b_pure", "object_first"]),
                "object_last": mean_of(results, ["positions", "b_pure", "object_last"]),
                "place_first": mean_of(results, ["positions", "b_pure", "place_first"]),
                "place_last": mean_of(results, ["positions", "b_pure", "place_last"]),
            },
            "c_gamma0": {
                "object_first": mean_of(results, ["positions", "c_gamma0", "object_first"]),
                "object_last": mean_of(results, ["positions", "c_gamma0", "object_last"]),
                "place_first": mean_of(results, ["positions", "c_gamma0", "place_first"]),
                "place_last": mean_of(results, ["positions", "c_gamma0", "place_last"]),
            },
        },
        "utf8": {
            "generate_only_legal_rate": mean_of(results, ["utf8", "generate_only", "legal_rate"]),
            "copy_only_legal_rate": mean_of(results, ["utf8", "copy_only", "legal_rate"]),
            "c_gamma0_legal_rate": mean_of(results, ["utf8", "c_gamma0", "legal_rate"]),
        },
    }
    report = {
        "schema": "taiji-gate-sweep-18-1-v1",
        "step": "N18.1",
        "date": "2026-11-04",
        "checkpoint": arguments.checkpoint,
        "dtype": "float64",
        "threads": 1,
        "data": {
            "dev_examples_used": results[0]["examples"] if results else 0,
            "slice_rule": "12 dev worlds x (object 0/1 x place 0/1) = 48 examples",
            "content_bytes_per_example": len(CONTENT_POSITIONS),
        },
        "conditions": {
            "sweep": "frozen routing_step (arm-locked alpha/beta); probability mix rebuilt with gamma forced to 0.6/0.5/0.4/0.3/0.2/0.0",
            "c_gamma0": "gold record, every content position pinned, gamma = 0",
            "positions": "per content byte hit rate under b_pure and c_gamma0",
            "utf8": "free decode under gamma = 1 (pure generate) vs gamma = 0 (copy-dominant; the null-route generated mass of q_null remains) vs c_gamma0 (correct-routing copy control)",
        },
        "cells": {f"{item['seed']}-{item['arm']}": item for item in results},
        "pooled": pooled,
        "cost": {
            "cpu_seconds": total_cpu,
            "wall_seconds": max(result["cost"]["wall_seconds"] for result in results),
            "adam_steps": 0,
        },
        "limits": {"zero_updates": True, "read_only": True, "confirm_reads": 0, "training": False},
    }
    write_json(ROOT / REPORT, report)

    write_json(ROOT / PREREG, build_prereg(pooled, REPORT))
    write_json(ROOT / MANIFEST, build_manifest(report, arguments.checkpoint))
    print(
        json.dumps(
            {
                "report": str(REPORT),
                "prereg": str(PREREG),
                "pooled": pooled,
                "cpu": round(total_cpu, 1),
            },
            ensure_ascii=False,
        )
    )
    return 0


def build_prereg(pooled: dict, report_path: Path) -> dict:
    """Byte-gate pre-registration for the next package (not retroactive).

    The gap is measured twice, deliberately:

    * the *current* point is the gate-only sweep - the model's own alpha/beta
      route with the best gamma per entity - because that is what can actually
      be reached today without touching the pointer;
    * the *ceiling* is ``c_gamma0`` (record and every content position pinned,
      gamma = 0), which shows the per-byte gate is reachable at all.
    """
    object_now = max(entry["object_byte_rate"] for entry in pooled["sweep"].values())
    place_now = max(entry["place_byte_rate"] for entry in pooled["sweep"].values())
    return {
        "schema": "taiji-byte-gate-prereg-v1",
        "step": "N19.0-proposal",
        "date": "2026-11-04",
        "status": "preregistered_for_next_package_not_retroactive",
        "does_not_change_n18_1": True,
        "gates": {
            "object_byte_rate": 0.95,
            "place_byte_rate": 0.95,
            "new_name_copied": 0.60,
            "overall": 0.80,
        },
        "reason": "18 content bytes multiply: 0.80^(1/18) = 0.9877 per byte, unreachable at the current precision. Byte gates give a climbable gradient without relaxing the sentence gate.",
        "per_byte_equivalent_of_overall": 0.8 ** (1.0 / 18.0),
        "current_gap": {
            "measured_at": "gate-only sweep: model alpha/beta route, best forced gamma per entity",
            "object_byte_rate_best_gate_only": object_now,
            "place_byte_rate_best_gate_only": place_now,
            "object_gap_to_gate": max(0.0, 0.95 - object_now),
            "place_gap_to_gate": max(0.0, 0.95 - place_now),
            "object_byte_rate_at_c_gamma0": pooled["c_gamma0"]["object_byte_rate"],
            "place_byte_rate_at_c_gamma0": pooled["c_gamma0"]["place_byte_rate"],
            "note": "c_gamma0 is the routing-pinned ceiling (both entities 1.00), not the current operating point",
            "new_name_copied_at_step300": 0.0,
            "overall_at_step300": 0.0,
            "evidence": str(report_path),
        },
    }


def build_manifest(report: dict, checkpoint: str) -> dict:
    return {
        "schema": "taiji-gate-sweep-18-1-manifest-v1",
        "step": "N18.1",
        "checkpoint": checkpoint,
        "report": str(REPORT),
        "report_sha256": sha256(ROOT / REPORT),
        "report_bytes": (ROOT / REPORT).stat().st_size,
        "prereg": str(PREREG),
        "prereg_sha256": sha256(ROOT / PREREG),
        "bundles": {key: item["bundle"] for key, item in report["cells"].items()},
        "pooled": report["pooled"],
        "cost": report["cost"],
        "limits": report["limits"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
