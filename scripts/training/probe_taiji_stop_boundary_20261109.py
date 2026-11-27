"""R57: stop-boundary gate on the sealed confirm split (second and final read).

Context: 19.2v read confirm once and landed 188/192.  All four failures are one
extra byte at position 27 (``把…放到{地点}到``), with both hops ``aligned``; the
copy stop edge works, the *sequence* edge does not exist under the old decode
rule (only an emitted EOS ends decoding).  19.2w implemented the structural fix
(``copy_stop_boundary.sequence_stop``) but only validated it on synthetic/dev
data.  This probe performs the preregistered second confirm read defined in
D-20261109-60.

Stages
------

* ``--freeze``  write the definition digest and the hashes of every input.  The
  file is never rewritten; a later run refuses if any hash drifted.
* default       read confirm once, decode all 192 with the structural stop
  (``stop_after = sequence_stop(PLACE_POSITIONS)``), decode the four published
  failures once more under the old rule to obtain the position-27 distribution.

Hard constraints (D-60): no retuning on the sealed split, no change to
``sequence_stop``, no new parameter, no tolerance, 0 training.  If the gate
fails, the only allowed next step is a dev/synthetic reproduction.
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

from taiji import candidate_identity  # noqa: E402
from taiji import data_premise_18_0 as premise  # noqa: E402
from taiji.binding_copy import MAX_PREDICTIONS, observe_goal  # noqa: E402
from taiji.byte_match_route import IDENTITY_DIM, ByteMatchRoute  # noqa: E402
from taiji.copy_stop_boundary import sequence_stop  # noqa: E402
from taiji.living_core import EOS  # noqa: E402
from taiji.record_relation_core import decode_endpoint  # noqa: E402
from taiji.selection_routing_checkpoint import load_routing_bundle, parameter_tensors  # noqa: E402

MANIFEST = ROOT / "plans/manifests/taiji_stop_boundary_20261109.json"
REPORT = ROOT / "reports/taiji_stop_boundary_20261109.json"
PROGRESS = ROOT / "reports/taiji_stop_boundary_20261109_progress.jsonl"
PREV_REPORT = ROOT / "reports/taiji_package19_2v_confirm_holdout_20261105.json"
ASSEMBLED = ROOT / "output/taiji_package19_2u_assembled_zero_training_stopfix.pt"
BUNDLE = ROOT / "output/taiji_package19_compare/28031-B_stop_lock-step300.pt"

CPU_CAP = 400.0
CPU_GUARD = 380.0
WALL_CAP = 3600.0
SPAN = 15
EXPECTED_CONFIRM = 192
EXPECTED_ANSWER_BYTES = 27
STOP_POSITION = sequence_stop(v.PLACE_POSITIONS)
TRACE_POSITIONS = (v.OBJECT_POSITIONS[-1] + 1, STOP_POSITION)
FROZEN_FILES = (
    "scripts/training/probe_taiji_stop_boundary_20261109.py",
    "scripts/training/probe_taiji_package19_2v_confirm_holdout.py",
    "scripts/training/probe_taiji_content_transport_18_1.py",
    "taiji/copy_stop_boundary.py",
    "taiji/binding_copy.py",
    "taiji/byte_match_route.py",
    "taiji/record_relation_core.py",
    "taiji/living_core.py",
    "taiji/data_premise_18_0.py",
    "taiji/candidate_identity.py",
)
DEFINITION = {
    "gate": "R57 stop-boundary on confirm",
    "stop_rule": "decode terminates one byte past the final hop's declared field end",
    "stop_position": STOP_POSITION,
    "answer_bytes_expected": EXPECTED_ANSWER_BYTES,
    "criteria": {
        "a": "the four published failures become exactly 27 B and byte-equal to target",
        "b": "the other 188 stay byte-equal to target (per-example trace now stored)",
        "c": "report position-27 stop probability and runner-up bytes with probabilities",
    },
    "exits": [
        "stop_edge_fixed",
        "stop_edge_probability_side",
        "stop_edge_not_the_cause",
    ],
    "retune_policy": "no retune on the sealed split; any failed gate goes back to dev/synthetic",
    "recovery_policy": (
        "a crash after the read but before the report may be recovered by one "
        "rerun of the identical frozen configuration; the rerun must reproduce "
        "every logged per-example (answer_bytes, byte_exact, terminated) triple"
    ),
    "confirm_read_index": 2,
    "previous_confirm_read": "N19.2v",
    "identity_policy": (
        "the 19.2u assembly predates the declarative decode rule 'sequence_stop'; "
        "identity drift is accepted only if dropping that one key reproduces the "
        "assembled candidate_sha_v2 exactly, which leaves weights, modules and "
        "graph structure untouched"
    ),
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


def previous_failures() -> list[dict]:
    payload = json.loads(PREV_REPORT.read_text(encoding="utf-8"))
    return [
        {
            "world": item["world"],
            "index": item["index"],
            "decoded_text": item["decoded_text"],
            "target_text": item["target_text"],
            "position": item["position"],
        }
        for item in payload["failures"]
    ]


def identity_reconciliation(candidate, identity: dict, assembled_declared: str) -> dict:
    """Explain a declarative-only identity drift, or prove there is none.

    ``candidate_sha_v2`` covers graph shape, module list and the declared decode
    rules, but not weight values.  The 19.2u assembly was written before the
    ``sequence_stop`` rule was declared, so its stored digest is one key short.
    Dropping exactly that key must reproduce the stored digest; anything else is
    real drift and refuses to run.
    """
    from taiji import candidate_identity as ci

    parameters = ci.declared_parameters(candidate)
    trainable = ci.trainable_declarations(candidate)
    alignment = bool(getattr(candidate.head.config, "alignment", False))
    current = dict(ci.DECODE_RULES)
    without_stop = {key: value for key, value in current.items() if key != "sequence_stop"}
    if without_stop == current:
        rebuilt = identity["candidate_sha_v2"]
        removed = None
    else:
        payload = ci._semantic_payload(parameters, trainable, alignment)
        payload["decode_rules"] = without_stop
        rebuilt = ci._digest(payload)
        removed = "decode_rules.sequence_stop"
    return {
        "assembled_declared": assembled_declared,
        "current_declared": identity["candidate_sha_v2"],
        "rebuilt_without_sequence_stop": rebuilt,
        "removed_key": removed,
        "identical": assembled_declared == identity["candidate_sha_v2"],
        "explained": assembled_declared == rebuilt,
    }


def load_candidate():
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


def byte_label(value: int) -> str:
    if value == EOS:
        return "<EOS>"
    if value < 0x20 or value > 0x7E:
        return f"0x{value:02x}"
    return chr(value)


def top_candidates(probabilities: torch.Tensor, count: int = 3) -> list[dict]:
    values, indices = torch.topk(probabilities, k=count)
    return [
        {
            "byte": int(index),
            "label": byte_label(int(index)),
            "probability": float(value),
        }
        for value, index in zip(values.tolist(), indices.tolist(), strict=True)
    ]


def slot_distribution(probabilities: torch.Tensor, prefix: bytes, q_bytes, view, limits) -> dict:
    """What the model wanted to emit at one position, plus the two stop channels."""
    copy_mass = torch.zeros(probabilities.shape[0], dtype=torch.float64)
    if len(q_bytes):
        copy_mass = copy_mass.index_add(0, view.byte_ids, q_bytes)
    candidates = top_candidates(probabilities)
    order = torch.argsort(probabilities, descending=True).tolist()
    eos_rank = order.index(EOS) + 1
    endpoint = decode_endpoint(probabilities, prefix=prefix)
    return {
        "eos_probability": float(probabilities[EOS]),
        "eos_rank": int(eos_rank),
        "top": candidates,
        "runner_up": candidates[1] if len(candidates) > 1 else None,
        "argmax_byte": int(torch.argmax(probabilities)),
        "argmax_label": byte_label(int(torch.argmax(probabilities))),
        "pre_mask_byte": int(endpoint["pre_mask_byte"]),
        "masked": bool(endpoint["masked"]),
        "copy_mass_total": float(copy_mass.sum()),
        "copy_mass_max": float(copy_mass.max()),
        "copy_mass_at_argmax": float(copy_mass[int(torch.argmax(probabilities))]),
        "copy_limits": [int(value) for value in limits],
        "copy_limit_active": max(int(value) for value in limits) if limits else -1,
    }


@torch.no_grad()
def decode_case(
    candidate,
    probe,
    fixture: dict,
    route_info: dict,
    stop_after: int | None,
    traced: bool,
) -> dict:
    prepared, _context = probe.prepare(fixture)
    view = prepared.base.view
    state, gate_index = observe_goal(candidate, prepared, gate_index=0)
    previous = EOS
    raw: list[int] = []
    routes: list[int] = []
    starts: list[int] = []
    terminated = False
    traces: dict[int, dict] = {}
    record_count = view.record_count
    for _prediction in range(MAX_PREDICTIONS):
        position = len(raw)
        if stop_after is not None and position >= stop_after:
            terminated = True
            break
        if position in v.OBJECT_POSITIONS:
            route, start = route_info["object_route"], route_info["object_name_offset"]
        elif position in v.PLACE_POSITIONS:
            route, start = route_info["place_route"], route_info["place_name_offset"]
        else:
            route, start = -1, None
        anchored = position in (v.OBJECT_POSITIONS[0], v.PLACE_POSITIONS[0])
        force, override = v.force_payload(view, route, start if anchored else None)
        limits = [-1] * record_count
        if 0 <= route_info["object_route"] < record_count:
            limits[route_info["object_route"]] = route_info["object_stop"]
        if position >= v.PLACE_POSITIONS[0] and 0 <= route_info["place_route"] < record_count:
            limits[route_info["place_route"]] = route_info["place_stop"]
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
        if traced and position in TRACE_POSITIONS:
            payload = slot_distribution(probabilities, bytes(raw), q_bytes, view, tuple(limits))
            payload["position"] = position
            payload["gamma_eff"] = float(gamma_eff)
            payload["q_null"] = float(q_null)
            traces[position] = payload
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
        "traces": traces,
    }


def row_for(fixture: dict, route_info: dict, decoded: dict) -> dict:
    target = v.target_bytes(fixture)
    raw = decoded["raw"]
    object_span = v.analyze_span(fixture, decoded, 3, 12, "object")
    place_span = v.analyze_span(fixture, decoded, 21, 6, "place")
    byte_exact = len(raw) == len(target) and tuple(raw) == target
    return {
        "world": fixture["world"],
        "index": fixture["index"],
        "age": v.age_of(fixture),
        "object_route_ok": route_info["object_route"] == int(fixture["selection_label"]),
        "place_route_ok": route_info["place_route"] == v.place_record(fixture),
        "byte_exact": bool(byte_exact),
        "completed": bool(byte_exact and decoded["terminated"]),
        "terminated": bool(decoded["terminated"]),
        "answer_bytes": len(raw),
        "target_bytes": len(target),
        "decoded_text": bytes(raw).decode("utf-8", "replace"),
        "target_text": bytes(target).decode("utf-8", "replace"),
        "first_difference": v.first_difference(fixture, decoded),
        "spans": [object_span, place_span],
        "raw": raw,
        "target": target,
        "routes": decoded["routes"],
        "starts": decoded["starts"],
    }


def summarize_192(fixtures: tuple[dict, ...], failures: list[dict]) -> dict:
    """Length/character statistics over all 192 confirm examples (0 forward)."""
    failure_keys = {(item["world"], item["index"]) for item in failures}
    place_chars = Counter(len(fixture["place_name"]) for fixture in fixtures)
    object_chars = Counter(len(fixture["object_name"]) for fixture in fixtures)
    place_bytes = Counter(len(fixture["place_name"].encode("utf-8")) for fixture in fixtures)
    last_chars = Counter(fixture["place_name"][-1] for fixture in fixtures)
    last_chars_fail = Counter(
        fixture["place_name"][-1]
        for fixture in fixtures
        if (fixture["world"], fixture["index"]) in failure_keys
    )
    last_chars_ok = Counter(
        fixture["place_name"][-1]
        for fixture in fixtures
        if (fixture["world"], fixture["index"]) not in failure_keys
    )
    failure_place_chars = Counter(
        len(fixture["place_name"])
        for fixture in fixtures
        if (fixture["world"], fixture["index"]) in failure_keys
    )
    success_place_chars = Counter(
        len(fixture["place_name"])
        for fixture in fixtures
        if (fixture["world"], fixture["index"]) not in failure_keys
    )
    trigger = "尾"
    trigger_fail = last_chars_fail.get(trigger, 0)
    trigger_all = last_chars.get(trigger, 0)
    trigger_rate = trigger_fail / trigger_all if trigger_all else 0.0
    other_rate = (
        (len(failures) - trigger_fail) / (len(fixtures) - trigger_all)
        if trigger_all < len(fixtures)
        else 0.0
    )
    return {
        "examples": len(fixtures),
        "place_char_length": {str(key): value for key, value in sorted(place_chars.items())},
        "object_char_length": {str(key): value for key, value in sorted(object_chars.items())},
        "place_name_byte_length": {str(key): value for key, value in sorted(place_bytes.items())},
        "place_last_char": dict(sorted(last_chars.items(), key=lambda item: -item[1])),
        "place_last_char_failures": dict(
            sorted(last_chars_fail.items(), key=lambda item: -item[1])
        ),
        "place_last_char_successes": dict(sorted(last_chars_ok.items(), key=lambda item: -item[1])),
        "failure_place_char_length": {
            str(key): value for key, value in sorted(failure_place_chars.items())
        },
        "success_place_char_length": {
            str(key): value for key, value in sorted(success_place_chars.items())
        },
        "universal_place_char_length": len(place_chars) == 1,
        "universal_object_char_length": len(object_chars) == 1,
        "two_char_place_is_necessary": False if len(place_chars) == 1 else None,
        "trigger_char": trigger,
        "trigger_failures": trigger_fail,
        "trigger_examples": trigger_all,
        "trigger_failure_rate": trigger_rate,
        "non_trigger_failure_rate": other_rate,
        "note": (
            "place names are uniform in length across all 192, so 'two-character place "
            "name' cannot be a necessary condition for the four failures; the suffix "
            "table separates enrichment from necessity"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true", help="write the preregistration only")
    args = parser.parse_args()

    started_cpu = time.process_time()
    started_wall = time.perf_counter()

    if args.freeze:
        if MANIFEST.exists():
            raise RuntimeError(f"preregistration already exists, refusing to rewrite: {MANIFEST}")
        failures = previous_failures()
        payload, candidate, probe, identity, parameter_count, route = load_candidate()
        manifest = {
            "schema": "taiji-stop-boundary-r57-manifest-v1",
            "step": "R57",
            "date": "2026-11-09",
            "preregistration": str(MANIFEST.relative_to(ROOT)),
            "definition": DEFINITION,
            "definition_digest": definition_digest(),
            "frozen_files": file_hashes(),
            "previous_confirm_read": {
                "report": str(PREV_REPORT.relative_to(ROOT)),
                "sha256": sha256_of(PREV_REPORT),
                "failure_examples": failures,
                "failure_count": len(failures),
            },
            "assembly": {
                "assembled_state": str(ASSEMBLED.relative_to(ROOT)),
                "assembled_sha256": sha256_of(ASSEMBLED),
                "parent_bundle_sha256": sha256_of(BUNDLE),
                "candidate_sha_v2": identity["candidate_sha_v2"],
                "candidate_parameters": parameter_count,
                "byte_match_parameters": sum(int(t.numel()) for t in route.parameters()),
            },
            "identity_reconciliation": identity_reconciliation(
                candidate, identity, payload.get("candidate_sha_v2")
            ),
            "cost_caps": {"cpu_seconds": CPU_CAP, "wall_seconds": WALL_CAP},
            "immutability": "written once before the second confirm read; never rewritten",
            "retune_policy": DEFINITION["retune_policy"],
            "discipline": {
                "training_runs": 0,
                "optimizer_steps": 0,
                "confirm_reads": 1,
                "parameters_added": 0,
                "graph_changes": 0,
            },
        }
        write_json(MANIFEST, manifest)
        print(
            json.dumps({"frozen": str(MANIFEST), "digest": manifest["definition_digest"]}, indent=1)
        )
        return 0

    if not MANIFEST.exists():
        raise RuntimeError("preregistration missing; run --freeze first")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    digest_now = definition_digest()
    if manifest["definition_digest"] != digest_now:
        raise RuntimeError("definition digest drift: the gate definition changed after freezing")
    drifted = [
        name for name, value in manifest["frozen_files"].items() if sha256_of(ROOT / name) != value
    ]
    if drifted:
        raise RuntimeError(f"frozen input drift: {drifted}")
    if manifest["previous_confirm_read"]["sha256"] != sha256_of(PREV_REPORT):
        raise RuntimeError("previous confirm report changed after freezing")

    payload, candidate, probe, identity, parameter_count, route = load_candidate()
    reconciliation = identity_reconciliation(candidate, identity, payload.get("candidate_sha_v2"))
    if identity["candidate_sha_v2"] != manifest["assembly"]["candidate_sha_v2"]:
        raise RuntimeError("live candidate_sha_v2 drifted after freezing")
    if not (reconciliation["identical"] or reconciliation["explained"]):
        raise RuntimeError(
            "assembled state identity is neither current nor explained by the "
            f"pre-registered policy: {reconciliation}"
        )

    previous_examples: list[dict] = []
    if not PROGRESS.exists() or PROGRESS.stat().st_size == 0:
        PROGRESS.write_text("", encoding="utf-8")
    else:
        events = [
            json.loads(line)
            for line in PROGRESS.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if any(event.get("event") == "finish" for event in events):
            raise RuntimeError("R57 already finished; the sealed split is not read twice")
        # A crash before the report is written leaves a complete per-example log;
        # the recovery pass must reproduce it row by row (same frozen config).
        previous_examples = [
            event
            for event in events
            if event.get("event") == "example" and event.get("read_index", 1) == 1
        ]
    recovery = bool(previous_examples)

    append_progress(
        {
            "event": "start",
            "definition_digest": digest_now,
            "candidate_sha_v2": identity["candidate_sha_v2"],
            "stop_position": STOP_POSITION,
            "recovery_rerun": recovery,
        }
    )

    # ---- the single confirm read ------------------------------------------------
    base = premise.build_payload()
    pools = premise.split_pools()
    worlds = [
        premise.build_world("confirm", world, pools)
        for world in range(premise.SPLIT_WORLDS["confirm"])
    ]
    materialised = dict(base)
    materialised["worlds"] = [
        {key: value for key, value in world.items() if key != "examples"} for world in worlds
    ]
    materialised["examples"] = [example for world in worlds for example in world["examples"]]
    confirm = tuple(premise.fixtures(materialised, "confirm"))
    if len(confirm) != EXPECTED_CONFIRM:
        raise RuntimeError(f"confirm fixture count drift: {len(confirm)}")
    failures_prev = manifest["previous_confirm_read"]["failure_examples"]
    failure_keys = {(item["world"], item["index"]) for item in failures_prev}
    append_progress({"event": "confirm_loaded", "examples": len(confirm), "read_index": 2})

    tensors = v.build_tensors(confirm)
    routes = v.match_routes(route, tensors, confirm)
    append_progress({"event": "matched", "examples": len(routes)})

    rows: list[dict] = []
    for fixture, route_info in zip(confirm, routes, strict=True):
        if time.process_time() - started_cpu > CPU_GUARD:
            raise RuntimeError("CPU guard hit before finishing the sealed read")
        decoded = decode_case(candidate, probe, fixture, route_info, STOP_POSITION, traced=False)
        rows.append(row_for(fixture, route_info, decoded))
        append_progress(
            {
                "event": "example",
                "read_index": 2 if recovery else 1,
                "world": fixture["world"],
                "index": fixture["index"],
                "answer_bytes": rows[-1]["answer_bytes"],
                "byte_exact": rows[-1]["byte_exact"],
                "terminated": rows[-1]["terminated"],
                "cpu_seconds": round(time.process_time() - started_cpu, 4),
            }
        )

    recovery_check: dict = {"recovery_rerun": recovery}
    if recovery:
        by_key = {(row["world"], row["index"]): row for row in rows}
        mismatches = []
        for event in previous_examples:
            key = (event["world"], event["index"])
            row = by_key.get(key)
            if row is None:
                mismatches.append({"world": key[0], "index": key[1], "reason": "missing"})
                continue
            same = (
                row["byte_exact"] == bool(event["byte_exact"])
                and row["answer_bytes"] == int(event["answer_bytes"])
                and row["terminated"] == bool(event["terminated"])
            )
            if not same:
                mismatches.append(
                    {
                        "world": key[0],
                        "index": key[1],
                        "before": {
                            "answer_bytes": event["answer_bytes"],
                            "byte_exact": event["byte_exact"],
                            "terminated": event["terminated"],
                        },
                        "after": {
                            "answer_bytes": row["answer_bytes"],
                            "byte_exact": row["byte_exact"],
                            "terminated": row["terminated"],
                        },
                    }
                )
        recovery_check.update(
            {
                "compared": len(previous_examples),
                "mismatches": mismatches[:10],
                "mismatch_total": len(mismatches),
                "identical": not mismatches,
            }
        )
        if mismatches:
            raise RuntimeError(f"recovery pass diverged from the logged read: {mismatches[:3]}")

    # ---- position-27 diagnostics on the four published failures (old rule) -----
    per_slot: list[dict] = []
    reproduction: list[dict] = []
    for fixture, route_info in zip(confirm, routes, strict=True):
        key = (fixture["world"], fixture["index"])
        if key not in failure_keys:
            continue
        decoded = decode_case(candidate, probe, fixture, route_info, None, traced=True)
        new_row = next(row for row in rows if (row["world"], row["index"]) == key)
        old_text = bytes(decoded["raw"]).decode("utf-8", "replace")
        expected = next(item for item in failures_prev if (item["world"], item["index"]) == key)
        reproduction.append(
            {
                "world": fixture["world"],
                "index": fixture["index"],
                "old_rule_text": old_text,
                "previous_read_text": expected["decoded_text"],
                "reproduced": old_text == expected["decoded_text"],
                "new_rule_text": new_row["decoded_text"],
                "new_rule_bytes": new_row["answer_bytes"],
                "target_text": new_row["target_text"],
            }
        )
        per_slot.append(
            {
                "world": fixture["world"],
                "index": fixture["index"],
                "place_name": fixture["place_name"],
                "object_name": fixture["object_name"],
                "place_name_last_char": fixture["place_name"][-1],
                "old_rule_text": old_text,
                "target_text": new_row["target_text"],
                "traces": {str(key): value for key, value in sorted(decoded["traces"].items())},
            }
        )
        append_progress(
            {
                "event": "failure_traced",
                "world": fixture["world"],
                "index": fixture["index"],
                "position_27_eos": decoded["traces"].get(STOP_POSITION, {}).get("eos_probability"),
                "position_27_top": decoded["traces"].get(STOP_POSITION, {}).get("argmax_label"),
                "cpu_seconds": round(time.process_time() - started_cpu, 4),
            }
        )

    # ---- gates ------------------------------------------------------------------
    four_rows = [row for row in rows if (row["world"], row["index"]) in failure_keys]
    rest_rows = [row for row in rows if (row["world"], row["index"]) not in failure_keys]
    gate_a = {
        "examples": len(four_rows),
        "expected": len(failure_keys),
        "all_exactly_27": all(row["answer_bytes"] == EXPECTED_ANSWER_BYTES for row in four_rows),
        "all_byte_exact": all(row["byte_exact"] for row in four_rows),
        "all_terminated": all(row["terminated"] for row in four_rows),
        "rows": [
            {
                "world": row["world"],
                "index": row["index"],
                "answer_bytes": row["answer_bytes"],
                "byte_exact": row["byte_exact"],
                "decoded_text": row["decoded_text"],
                "target_text": row["target_text"],
            }
            for row in four_rows
        ],
    }
    gate_a["pass"] = (
        gate_a["examples"] == gate_a["expected"]
        and gate_a["all_exactly_27"]
        and gate_a["all_byte_exact"]
        and gate_a["all_terminated"]
    )
    gate_b = {
        "examples": len(rest_rows),
        "expected": EXPECTED_CONFIRM - len(failure_keys),
        "all_byte_exact": all(row["byte_exact"] for row in rest_rows),
        "all_terminated": all(row["terminated"] for row in rest_rows),
        "all_exactly_27": all(row["answer_bytes"] == EXPECTED_ANSWER_BYTES for row in rest_rows),
        "answer_byte_lengths": dict(Counter(str(row["answer_bytes"]) for row in rest_rows)),
    }
    gate_b["pass"] = (
        gate_b["examples"] == gate_b["expected"]
        and gate_b["all_byte_exact"]
        and gate_b["all_terminated"]
        and gate_b["all_exactly_27"]
    )
    read_integrity = all(item["reproduced"] for item in reproduction)
    model_wants_stop = all(
        item["traces"].get(str(STOP_POSITION), {}).get("argmax_byte") == EOS for item in per_slot
    )
    gate_c = {
        "diagnostics_present": all(str(STOP_POSITION) in item["traces"] for item in per_slot),
        "eos_is_argmax_at_27": model_wants_stop,
        "per_example": [
            {
                "world": item["world"],
                "index": item["index"],
                "position_27": item["traces"].get(str(STOP_POSITION)),
                "position_15": item["traces"].get(str(TRACE_POSITIONS[0])),
            }
            for item in per_slot
        ],
    }

    if not (gate_a["pass"] and gate_b["pass"]):
        verdict = "stop_edge_not_the_cause"
        branch = "结构性停止没有把 4 条变成恰好 27 B，或另有 188 之外的偏移"
    elif model_wants_stop:
        verdict = "stop_edge_fixed"
        branch = "三条全过：27 B 处 EOS 已是 argmax，结构停止与模型侧一致"
    else:
        verdict = "stop_edge_probability_side"
        branch = (
            "4 条已被结构停止修正为 27 B，但位置 27 的 argmax 仍是字节而非 EOS："
            "停止边界只存在于解码器，模型自身没有长出停止概率"
        )

    readings = v.summarize(rows)
    outcome = {
        "verdict": verdict,
        "branch": branch,
        "gates": {"a": gate_a, "b": gate_b, "c": gate_c},
        "read_integrity": {
            "previous_read_reproduced": read_integrity,
            "rows": reproduction,
        },
        "completion": readings["completion"],
        "byte_exact_rate": readings["byte_exact_rate"],
    }

    cpu_seconds = time.process_time() - started_cpu
    wall_seconds = time.perf_counter() - started_wall
    report = {
        "schema": "taiji-stop-boundary-r57-report-v1",
        "step": "R57",
        "date": "2026-11-09",
        "relation_to_main_task": (
            "本步直接面向主任务 Task1：19.2v confirm 188/192 的最后 4 例是唯一未收口的"
            "读数，本闸决定它能否变成 192/192。"
        ),
        "preregistration": str(MANIFEST.relative_to(ROOT)),
        "preregistration_sha256": sha256_of(MANIFEST),
        "read_register": {
            "confirm_read_index": 2,
            "recovery_rerun": recovery,
            "recovery_check": recovery_check,
            "previous_confirm_read": "N19.2v",
            "authorization": "D-20261109-60",
            "retune_forbidden": True,
            "single_forward_per_configuration": True,
        },
        "assembly": {
            "candidate_sha_v2": identity["candidate_sha_v2"],
            "candidate_parameters": parameter_count,
            "assembled_state_sha256": sha256_of(ASSEMBLED),
            "copy_stop_module": "taiji/copy_stop_boundary.py",
            "stop_position": STOP_POSITION,
            "identity_reconciliation": reconciliation,
        },
        "evaluation": {
            "examples": len(rows),
            "rule": "sequence_stop on the final hop's declared field end",
            "readings": readings,
            "first_difference": dict(Counter(row["first_difference"]["category"] for row in rows)),
        },
        "statistics_192": summarize_192(confirm, failures_prev),
        "per_slot_stop": per_slot,
        "outcome": outcome,
        "trace": {
            "examples_per_row": len(rows),
            "rows": [
                {
                    "world": row["world"],
                    "index": row["index"],
                    "age": row["age"],
                    "answer_bytes": row["answer_bytes"],
                    "target_bytes": row["target_bytes"],
                    "byte_exact": row["byte_exact"],
                    "terminated": row["terminated"],
                    "decoded_text": row["decoded_text"],
                    "target_text": row["target_text"],
                }
                for row in rows
            ],
        },
        "cost": {
            "cpu_seconds": round(cpu_seconds, 2),
            "wall_seconds": round(wall_seconds, 2),
            "cpu_cap": CPU_CAP,
            "wall_cap": WALL_CAP,
            "over_budget": cpu_seconds > CPU_CAP or wall_seconds > WALL_CAP,
            "estimated_before_read": 75.0,
        },
        "discipline": {
            "training_runs": 0,
            "optimizer_steps": 0,
            "confirm_reads": 1,
            "parameters_added": 0,
            "graph_changes": 0,
            "progress_fsync_per_row": True,
            "report_write_then_rename": True,
        },
        "closed_state": {
            "frozen_files_unchanged": file_hashes() == manifest["frozen_files"],
            "module_hashes": file_hashes(),
        },
    }
    write_json(REPORT, report)
    write_json(
        MANIFEST.with_name(MANIFEST.stem + "_result.json"),
        {
            "schema": "taiji-stop-boundary-r57-result-v1",
            "report": str(REPORT.relative_to(ROOT)),
            "report_sha256": sha256_of(REPORT),
            "definition_digest": digest_now,
            "verdict": verdict,
            "completion": readings["completion"],
            "cost": report["cost"],
        },
    )
    append_progress(
        {
            "event": "finish",
            "verdict": verdict,
            "completion": readings["completion"],
            "cpu_seconds": round(cpu_seconds, 4),
        }
    )
    print(
        json.dumps(
            {
                "verdict": verdict,
                "completion": readings["completion"],
                "gate_a": gate_a["pass"],
                "gate_b": gate_b["pass"],
                "eos_argmax": model_wants_stop,
                "read_integrity": read_integrity,
                "cost": report["cost"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
