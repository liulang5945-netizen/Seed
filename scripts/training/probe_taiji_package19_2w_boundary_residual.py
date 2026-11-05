"""19.2w: reproduce and fix the one-byte stop-boundary residual without touching confirm.

The only confirm read allowed in this project was 19.2v; this probe never calls
``premise.fixtures(..., "confirm")`` and never materialises confirm worlds.  All
reproduction data is built from the same frozen 18.0 generator with custom
name pools, so the trigger class (place names ending in a chosen final
character) can be sampled without a sealed read.

Stages:

* ``repro``  - old decode rule, tail/control/robustness samples, diagnosis trace.
* ``fix``    - structural sequence stop, three validation gates, new assembly.
* ``all``    - both (default).

Discipline: 0 training, 0 Adam, 0 confirm reads, row-wise fsync, write-then-rename.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "scripts" / "training"
for entry in (ROOT, TRAINING):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import torch  # noqa: E402

torch.set_num_threads(1)

import probe_taiji_content_transport_18_1 as probe_mod  # noqa: E402
import probe_taiji_package19_2v_confirm_holdout as v  # noqa: E402

from taiji import (
    candidate_identity,  # noqa: E402
    copy_stop_boundary,  # noqa: E402
)
from taiji import data_premise_18_0 as premise  # noqa: E402
from taiji.binding_copy import MAX_PREDICTIONS, observe_goal  # noqa: E402
from taiji.byte_match_route import IDENTITY_DIM, ByteMatchRoute  # noqa: E402
from taiji.copy_stop_boundary import sequence_stop  # noqa: E402
from taiji.living_core import EOS  # noqa: E402
from taiji.record_relation_core import decode_endpoint  # noqa: E402
from taiji.selection_routing_checkpoint import load_routing_bundle, parameter_tensors  # noqa: E402

REPORT = ROOT / "reports/taiji_package19_2w_boundary_residual_20261105.json"
MANIFEST = ROOT / "plans/manifests/taiji_package19_2w_boundary_residual_20261105.json"
PROGRESS = ROOT / "reports/taiji_package19_2w_boundary_residual_20261105_progress.jsonl"
ASSEMBLED = ROOT / "output/taiji_package19_2u_assembled_zero_training_stopfix.pt"
ASSEMBLED_R2 = ROOT / "output/taiji_package19_2w_assembled_stopfix_r2.pt"
BUNDLE = ROOT / "output/taiji_package19_compare/28031-B_stop_lock-step300.pt"

CPU_CAP = 400.0
CPU_GUARD = 380.0
WALL_CAP = 3600.0
SPAN = 15
TAIL_EXAMPLES = 64
CONTROL_EXAMPLES = 64
ROBUST_EXAMPLES = 8
OBJECT_POSITIONS = v.OBJECT_POSITIONS
PLACE_POSITIONS = v.PLACE_POSITIONS
STOP_POSITION = sequence_stop(PLACE_POSITIONS)


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


def synthetic_fixtures(place_pool: tuple[str, ...], worlds: int) -> tuple[dict, ...]:
    """Build dev-split fixtures from the frozen generator with a custom place pool."""
    base = premise.build_payload()
    real = premise.split_pools()
    object_pool = real["dev"][0]
    pools = {"train": real["train"], "dev": (object_pool, place_pool)}
    built = [premise.build_world("dev", world, pools) for world in range(worlds)]
    payload = dict(base)
    payload["worlds"] = [
        {key: value for key, value in world.items() if key != "examples"} for world in built
    ]
    payload["examples"] = [example for world in built for example in world["examples"]]
    return tuple(premise.fixtures(payload, "dev"))


def names_ending(character: str) -> tuple[str, ...]:
    return tuple(name for name in premise.place_names() if name.endswith(character))


def suffix_classes() -> tuple[str, ...]:
    counts = Counter(name[-1] for name in premise.place_names())
    return tuple(sorted(counts))


def scalar(value) -> int:
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    return int(tensor.reshape(-1)[0])


@torch.no_grad()
def decode_traced(
    candidate,
    probe,
    fixture: dict,
    route_info: dict,
    gamma: float,
    stop_after: int | None,
) -> dict:
    """Same loop as 19.2v, plus an optional structural sequence stop and a tail trace."""
    prepared, _context = probe.prepare(fixture)
    view = prepared.base.view
    state, gate_index = observe_goal(candidate, prepared, gate_index=0)
    previous = EOS
    raw: list[int] = []
    routes: list[int] = []
    starts: list[int] = []
    terminated = False
    trace: list[dict] = []
    record_count = view.record_count
    record_bytes = b"".join(record.raw for record in fixture["records"])
    for _prediction in range(MAX_PREDICTIONS):
        position = len(raw)
        if stop_after is not None and position >= stop_after:
            terminated = True
            break
        if position in OBJECT_POSITIONS:
            route, start = route_info["object_route"], route_info["object_name_offset"]
        elif position in PLACE_POSITIONS:
            route, start = route_info["place_route"], route_info["place_name_offset"]
        else:
            route, start = -1, None
        anchored = position in (OBJECT_POSITIONS[0], PLACE_POSITIONS[0])
        force, override = v.force_payload(view, route, start if anchored else None)
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
        emitted = int(endpoint["byte"])
        if position >= PLACE_POSITIONS[-1]:
            top = int(torch.argmax(probabilities))
            trace.append(
                {
                    "position": position,
                    "emitted": emitted,
                    "eos": bool(endpoint["eos"]),
                    "eos_probability": float(probabilities[EOS]),
                    "top_byte": top,
                    "top_probability": float(probabilities[top]),
                    "copy_content": scalar(content),
                    "copy_mass_max": float(q_bytes.max()) if hasattr(q_bytes, "max") else None,
                    "emitted_in_records": (emitted < 256 and bytes([emitted]) in record_bytes),
                }
            )
        if endpoint["eos"]:
            terminated = True
            break
        byte = emitted
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
        "trace": tuple(trace),
    }


def extra_bytes(row: dict) -> int:
    return max(0, len(row["raw"]) - len(row["target"]))


@torch.no_grad()
def run_rows(
    candidate,
    probe,
    fixtures: tuple[dict, ...],
    routes: list[dict],
    gamma: float,
    label: str,
    started_cpu: float,
    stop_after: int | None,
    limit: int,
    keep_trace: bool,
) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    trace: dict = {"config": label, "gamma_override": gamma, "stopped_for_budget": False}
    for fixture, route_info in zip(fixtures[:limit], routes[:limit], strict=True):
        if time.process_time() - started_cpu > CPU_GUARD:
            trace["stopped_for_budget"] = True
            break
        gold_object = int(fixture["selection_label"])
        gold_place = v.place_record(fixture)
        decoded = decode_traced(candidate, probe, fixture, route_info, gamma, stop_after)
        object_span = v.analyze_span(fixture, decoded, 3, 12, "object")
        place_span = v.analyze_span(fixture, decoded, 21, 6, "place")
        target = v.target_bytes(fixture)
        byte_exact = len(decoded["raw"]) == len(target) and tuple(decoded["raw"]) == target
        row = {
            "world": fixture["world"],
            "index": fixture["index"],
            "age": v.age_of(fixture),
            "object_route_ok": route_info["object_route"] == gold_object,
            "place_route_ok": route_info["place_route"] == gold_place,
            "spans": [object_span, place_span],
            "byte_exact": bool(byte_exact),
            "answer_bytes": len(target),
            "completed": bool(byte_exact and decoded["terminated"]),
            "raw": decoded["raw"],
            "target": target,
            "terminated": decoded["terminated"],
            "first_difference": v.first_difference(fixture, decoded),
            "extra_bytes": extra_bytes({"raw": decoded["raw"], "target": target}),
        }
        if keep_trace and not byte_exact:
            row["trace"] = list(decoded["trace"])
            place_name = fixture["place_name"]
            row["place_name_last_byte"] = place_name.encode("utf-8")[-1]
        rows.append(row)
        append_progress(
            {
                "event": "example",
                "config": label,
                "world": fixture["world"],
                "index": fixture["index"],
                "completed": row["completed"],
                "extra_bytes": row["extra_bytes"],
                "cpu_seconds": round(time.process_time() - started_cpu, 4),
            }
        )
    trace["examples"] = len(rows)
    return rows, trace


def dataset_report(rows: list[dict]) -> dict:
    summary = v.summarize(rows) if rows else {}
    extras = [row for row in rows if row["extra_bytes"] > 0]
    return {
        "examples": len(rows),
        "byte_exact": sum(1 for row in rows if row["byte_exact"]),
        "completed": sum(1 for row in rows if row["completed"]),
        "extra_examples": len(extras),
        "extra_rate": len(extras) / len(rows) if rows else 0.0,
        "extra_bytes_total": sum(row["extra_bytes"] for row in rows),
        "extra_positions": dict(
            Counter(str(row["first_difference"]["position"]) for row in extras)
        ),
        "first_difference": dict(Counter(row["first_difference"]["category"] for row in rows)),
        "metrics": summary,
    }


def load_candidate() -> tuple:
    if not ASSEMBLED.exists():
        raise RuntimeError(f"assembled state missing: {ASSEMBLED}")
    if not BUNDLE.exists():
        raise RuntimeError(f"parent bundle missing: {BUNDLE}")
    payload = torch.load(ASSEMBLED, map_location="cpu")
    candidate = load_routing_bundle(BUNDLE)[0]
    probe = probe_mod.ProbeContext(candidate)
    probe.arm = "B_stop_lock"
    with torch.no_grad():
        identity = candidate_identity.candidate_identity(candidate)
        parameter_count = sum(
            int(tensor.numel()) for tensor in parameter_tensors(candidate).values()
        )
    route = ByteMatchRoute(dim=IDENTITY_DIM, span=SPAN)
    route.load_state_dict(payload["byte_match_state"])
    route.eval()
    return payload, candidate, probe, identity, parameter_count, route


@torch.no_grad()
def measure(
    candidate,
    probe,
    route,
    fixtures: tuple[dict, ...],
    label: str,
    started_cpu: float,
    stop_after: int | None,
    limit: int,
    keep_trace: bool,
) -> tuple[list[dict], dict]:
    tensors = v.build_tensors(fixtures[:limit])
    routes = v.match_routes(route, tensors, fixtures[:limit])
    return run_rows(
        candidate,
        probe,
        fixtures,
        routes,
        0.0,
        label,
        started_cpu,
        stop_after,
        limit,
        keep_trace,
    )


def structural_note(rows: list[dict]) -> dict:
    """Describe the 19.2v residual structurally without quoting sealed bytes."""
    extras = [row for row in rows if row["extra_bytes"] > 0]
    last_bytes = Counter(str(row["place_name_last_byte"]) for row in extras)
    return {
        "extra_examples": len(extras),
        "place_name_last_byte": dict(last_bytes),
        "extra_bytes_per_example": dict(Counter(row["extra_bytes"] for row in extras)),
        "first_difference_position": dict(
            Counter(str(row["first_difference"]["position"]) for row in extras)
        ),
        "record_kind": dict(Counter(row["first_difference"].get("record_kind") for row in extras)),
    }


def confirm_residual_probe() -> dict:
    """Read the published 19.2v report only; never touch the sealed fixtures."""
    path = ROOT / "reports/taiji_package19_2v_confirm_holdout_20261105.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    failures = payload.get("failures", [])
    return {
        "source": str(path.relative_to(ROOT)),
        "confirm_examples": payload["evaluation"]["examples"],
        "failure_examples": len(failures),
        "failure_rate": len(failures) / payload["evaluation"]["examples"],
        "first_difference_categories": dict(
            Counter(item.get("first_difference", {}).get("category") for item in failures)
        ),
        "first_difference_positions": dict(
            Counter(str(item.get("first_difference", {}).get("position")) for item in failures)
        ),
        "record_kind": dict(Counter(item.get("record_kind") for item in failures)),
        "note": (
            "structural fields only; the sealed byte bodies stay in the report "
            "and are not re-read here"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("repro", "fix", "all"), default="all")
    parser.add_argument("--tail-limit", type=int, default=TAIL_EXAMPLES)
    parser.add_argument("--control-limit", type=int, default=CONTROL_EXAMPLES)
    parser.add_argument("--robust-limit", type=int, default=ROBUST_EXAMPLES)
    parser.add_argument("--dev-limit", type=int, default=192)
    args = parser.parse_args()

    started_cpu = time.process_time()
    started_wall = time.perf_counter()
    if not REPORT.parent.exists() or not MANIFEST.parent.exists():
        raise RuntimeError("report or manifest directory missing")
    if not PROGRESS.exists():
        PROGRESS.write_text("", encoding="utf-8")

    payload, candidate, probe, identity, parameter_count, route = load_candidate()
    old_sha = payload.get("candidate_sha_v2")
    classes = suffix_classes()
    if "尾" not in classes:
        raise RuntimeError("frozen place pool has no name ending in the trigger character")
    tail_pool = names_ending("尾")
    control_pool = names_ending("畔")

    append_progress(
        {
            "event": "start",
            "stage": args.stage,
            "candidate_sha_v2": identity["candidate_sha_v2"],
            "assembled_sha256": sha256_of(ASSEMBLED),
            "stop_position": STOP_POSITION,
        }
    )

    result: dict = {
        "schema": "taiji-package19-2w-boundary-residual-v1",
        "step": "N19.2w",
        "date": "2026-11-05",
        "purpose": "reproduce and fix the one-byte stop residual on synthetic data only",
        "frozen_inputs": {
            "source_assembly": str(ASSEMBLED.relative_to(ROOT)),
            "source_assembly_sha256": sha256_of(ASSEMBLED),
            "source_candidate_sha_v2": old_sha,
            "source_candidate_parameters": parameter_count,
            "parent_bundle_sha256": sha256_of(BUNDLE),
            "place_suffix_classes": list(classes),
            "stop_position": STOP_POSITION,
            "decode_rule_old": "stop only when the model emits EOS",
            "decode_rule_new": "stop one byte past the final hop's declared field end",
        },
        "confirm_residual": confirm_residual_probe(),
    }

    if args.stage in ("repro", "all"):
        tail_fixtures = synthetic_fixtures(tail_pool, worlds=5)
        control_fixtures = synthetic_fixtures(control_pool, worlds=5)
        tail_rows, tail_trace = measure(
            candidate,
            probe,
            route,
            tail_fixtures,
            "repro_tail",
            started_cpu,
            None,
            args.tail_limit,
            True,
        )
        control_rows, control_trace = measure(
            candidate,
            probe,
            route,
            control_fixtures,
            "repro_control",
            started_cpu,
            None,
            args.control_limit,
            False,
        )
        result["reproduction"] = {
            "tail": dataset_report(tail_rows),
            "control": dataset_report(control_rows),
            "tail_success_rate": 0.0,
            "tail_structural_note": structural_note(tail_rows),
            "diagnostics": [
                {
                    "world": row["world"],
                    "index": row["index"],
                    "extra_bytes": row["extra_bytes"],
                    "decoded_bytes": len(row["raw"]),
                    "target_bytes": len(row["target"]),
                    "first_difference": row["first_difference"],
                    "place_name_last_byte": row.get("place_name_last_byte"),
                    "trace": row.get("trace", []),
                }
                for row in tail_rows
                if row["extra_bytes"] > 0
            ],
            "traces": tail_trace,
            "control_trace": control_trace,
        }
        result["reproduction"]["tail_success_rate"] = result["reproduction"]["tail"]["extra_rate"]
        append_progress(
            {
                "event": "repro",
                "tail_extra_rate": result["reproduction"]["tail"]["extra_rate"],
                "control_extra_rate": result["reproduction"]["control"]["extra_rate"],
                "cpu_seconds": round(time.process_time() - started_cpu, 4),
            }
        )

    if args.stage in ("fix", "all"):
        r2 = dict(payload)
        r2["candidate_sha_v2"] = identity["candidate_sha_v2"]
        r2["sequence_stop_boundary"] = True
        r2["label"] = "assembled_stopfix_r2"
        r2["source_assembly_label"] = payload.get("label", "assembled_zero_training_stopfix")
        torch.save(r2, ASSEMBLED_R2)
        dev = premise.fixtures(premise.build_payload(), "dev")
        tail_fixtures = synthetic_fixtures(tail_pool, worlds=5)
        control_fixtures = synthetic_fixtures(control_pool, worlds=5)
        tail_rows, tail_trace = measure(
            candidate,
            probe,
            route,
            tail_fixtures,
            "fix_tail",
            started_cpu,
            STOP_POSITION,
            args.tail_limit,
            False,
        )
        control_rows, control_trace = measure(
            candidate,
            probe,
            route,
            control_fixtures,
            "fix_control",
            started_cpu,
            STOP_POSITION,
            args.control_limit,
            False,
        )
        dev_rows, dev_trace = measure(
            candidate,
            probe,
            route,
            dev,
            "fix_dev",
            started_cpu,
            STOP_POSITION,
            args.dev_limit,
            False,
        )
        robust: dict[str, dict] = {}
        for suffix in classes:
            fixtures = synthetic_fixtures(names_ending(suffix), worlds=1)
            rows, trace = measure(
                candidate,
                probe,
                route,
                fixtures,
                f"fix_robust_{suffix}",
                started_cpu,
                STOP_POSITION,
                args.robust_limit,
                False,
            )
            robust[suffix] = dataset_report(rows)
        dev_summary = v.summarize(dev_rows) if dev_rows else {}
        l1_pass = all(
            abs(float(dev_summary.get(key, -1.0)) - target) <= 1e-12
            for key, target in v.L1_TARGETS.items()
        )
        validation = {
            "synthetic_tail": dataset_report(tail_rows),
            "synthetic_control": dataset_report(control_rows),
            "dev": {
                "examples": len(dev_rows),
                "byte_exact": sum(1 for row in dev_rows if row["byte_exact"]),
                "completion": dev_summary.get("completion"),
                "metrics": dev_summary,
                "l1_pass": l1_pass,
                "first_difference": dict(
                    Counter(row["first_difference"]["category"] for row in dev_rows)
                ),
                "extra_examples": sum(1 for row in dev_rows if row["extra_bytes"] > 0),
            },
            "robustness": robust,
            "trace_tail": tail_trace,
            "trace_control": control_trace,
            "trace_dev": dev_trace,
        }
        gates = {
            "synthetic_tail_exact": (
                validation["synthetic_tail"]["byte_exact"] == args.tail_limit
                and validation["synthetic_tail"]["extra_bytes_total"] == 0
            ),
            "synthetic_control_exact": (
                validation["synthetic_control"]["byte_exact"] == args.control_limit
                and validation["synthetic_control"]["extra_bytes_total"] == 0
            ),
            "dev_l1_exact": l1_pass,
            "dev_sentence_exact": (
                validation["dev"]["byte_exact"] == len(dev_rows) and len(dev_rows) == args.dev_limit
            ),
            "robustness_clean": all(
                item["extra_bytes_total"] == 0 and item["byte_exact"] == args.robust_limit
                for item in robust.values()
            ),
        }
        result["fix"] = {
            "assembled_r2": str(ASSEMBLED_R2.relative_to(ROOT)),
            "assembled_r2_sha256": sha256_of(ASSEMBLED_R2),
            "candidate_sha_v2_before": old_sha,
            "candidate_sha_v2_after": identity["candidate_sha_v2"],
            "candidate_parameters": parameter_count,
            "rule": "decode terminates one byte past the final hop's declared field end",
            "parameters_added": 0,
            "sample_constants": 0,
            "gap_constant": False,
            "validation": validation,
            "gates": gates,
            "verdict": "fixed" if all(gates.values()) else "not_fixed",
        }
        append_progress(
            {
                "event": "fix",
                "verdict": result["fix"]["verdict"],
                "gates": gates,
                "cpu_seconds": round(time.process_time() - started_cpu, 4),
            }
        )

    result["candidate_identity"] = {
        "candidate_sha_v2": identity["candidate_sha_v2"],
        "parameters": parameter_count,
        "decode_rules": candidate_identity.DECODE_RULES,
        "modules": list(candidate_identity.GRAPH_MODULES),
        "stop_helper": copy_stop_boundary.sequence_stop.__name__,
    }
    result["cost"] = {
        "cpu_seconds": round(time.process_time() - started_cpu, 2),
        "wall_seconds": round(time.perf_counter() - started_wall, 2),
        "cpu_cap": CPU_CAP,
        "wall_cap": WALL_CAP,
        "over_budget": time.process_time() - started_cpu > CPU_CAP,
    }
    result["discipline"] = {
        "training_runs": 0,
        "optimizer_steps": 0,
        "adam_steps": 0,
        "confirm_reads": 0,
        "parameters_added": 0,
        "graph_changes": 0,
        "progress_fsync_per_row": True,
        "report_write_then_rename": True,
    }
    write_json(REPORT, result)
    write_json(
        MANIFEST,
        {
            "schema": "taiji-package19-2w-manifest-v1",
            "report": str(REPORT.relative_to(ROOT)),
            "candidate_sha_v2": identity["candidate_sha_v2"],
            "verdict": result.get("fix", {}).get("verdict", "repro_only"),
            "cost": result["cost"],
        },
    )
    append_progress(
        {
            "event": "finish",
            "cpu_seconds": round(time.process_time() - started_cpu, 4),
            "over_budget": result["cost"]["over_budget"],
        }
    )
    print(json.dumps(result["cost"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
