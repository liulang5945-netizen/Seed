#!/usr/bin/env python
"""N16.1.3 read-only architecture probes (0 updates, 0 training forward).

P1  Order-blindness of the selection head: byte permutations of the goal must
    leave the head scores unchanged (mean byte embedding is a multiset
    function), including the swap witness pair.
P2  Memorization ceiling: selection top-1 on the 288 train rows versus the dev
    subsets, from the stored step-300 bundle.
P3  Entity-start oracle: force the first decoding step onto the name-span
    start inside the correct record, then continue locked generation; compare
    completion with the stored arm-A reading on the same 40 examples.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

torch.set_num_threads(1)

from taiji.binding_copy import (  # noqa: E402
    advance_copy,
    answer_step,
    forced_read,
    initial_binding_state,
    observe_goal,
    prepare,
)
from taiji.binding_copy_scoring import score_generation  # noqa: E402
from taiji.binding_select import GAMMA_FLOOR, load_bundle, locked_step, select  # noqa: E402
from taiji.living_core import EOS  # noqa: E402
from taiji.record_relation_core import MAX_PREDICTIONS, decode_endpoint  # noqa: E402

BUNDLE = "output/taiji_core_binding_select_16_1_compare/bundles/28031-A-locked-step300.pt"
COMPARE_SCRIPT = "scripts/training/verify_taiji_binding_select_16_1_compare.py"
OUT = "reports/taiji_binding_select_16_1_arch_probe_20261103.json"

STARTED_CPU = time.process_time()
STARTED_WALL = time.perf_counter()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_fixtures():
    spec = importlib.util.spec_from_file_location("cmp", ROOT / COMPARE_SCRIPT)
    cmp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cmp)
    train, dev, _select, _data = cmp.load_data()
    return train, dev


def probe_order_blindness(candidate, dev) -> dict:
    head = candidate.head
    rows = []
    with torch.no_grad():
        for fixture in dev[:8]:
            goal = fixture["goal"].encode("utf-8")
            keys = fixture["record_keys"]
            base_scores = select(candidate, goal, keys)["scores"]
            for name, permuted in (
                ("reversed", bytes(reversed(goal))),
                ("sorted", bytes(sorted(goal))),
            ):
                perm_scores = select(candidate, permuted, keys)["scores"]
                rows.append(
                    {
                        "index": fixture["index"],
                        "world": fixture["world"],
                        "permutation": name,
                        "multiset_equal": sorted(goal) == sorted(permuted),
                        "max_abs_score_diff": max(
                            abs(a - b) for a, b in zip(base_scores, perm_scores, strict=True)
                        ),
                    }
                )
        witness_a = "把甲从乙搬到丙".encode()
        witness_b = "把乙从甲搬到丙".encode()
        enc_a = head.encode(witness_a)
        enc_b = head.encode(witness_b)
        witness = {
            "pair": ["把甲从乙搬到丙", "把乙从甲搬到丙"],
            "byte_multiset_equal": sorted(witness_a) == sorted(witness_b),
            "encoding_max_abs_diff": float((enc_a - enc_b).abs().max()),
            "encoding_bit_equal": bool(torch.equal(enc_a, enc_b)),
        }
    return {
        "permutation_rows": rows,
        "max_abs_score_diff_over_rows": max(row["max_abs_score_diff"] for row in rows),
        "witness": witness,
        "invariant_holds": max(row["max_abs_score_diff"] for row in rows) < 1e-10,
        "note": (
            "SelectionHead.encode is a mean over byte embeddings, so the head sees only the "
            "byte multiset of the goal; any permutation of the goal bytes must give the same "
            "scores up to float64 summation order noise."
        ),
    }


def probe_memorization(candidate, train, dev) -> dict:
    with torch.no_grad():
        train_correct = sum(
            1
            for fixture in train
            if select(
                candidate,
                fixture["goal"].encode("utf-8"),
                fixture["record_keys"],
                label=fixture["selection_label"],
            )["correct"]
        )
        dev_rows = []
        for fixture in dev:
            outcome = select(
                candidate,
                fixture["goal"].encode("utf-8"),
                fixture["record_keys"],
                label=fixture["selection_label"],
            )
            dev_rows.append((fixture["flags"], bool(outcome["correct"])))

    def subset_rate(name: str) -> dict:
        rows = [correct for flags, correct in dev_rows if flags[name]]
        return {
            "correct": sum(rows),
            "n": len(rows),
            "rate": sum(rows) / len(rows) if rows else None,
        }

    seen_rows = [correct for flags, correct in dev_rows if not any(flags.values())]
    return {
        "train_top1": {
            "correct": train_correct,
            "n": len(train),
            "rate": train_correct / len(train),
        },
        "dev_seen": {
            "correct": sum(seen_rows),
            "n": len(seen_rows),
            "rate": sum(seen_rows) / len(seen_rows),
        },
        "dev_unseen_output_name": subset_rate("unseen_output_name"),
        "dev_new_description": subset_rate("new_description"),
        "dev_both_new": subset_rate("both_new"),
        "dev_unseen_combination": subset_rate("unseen_combination"),
    }


def oracle_entity_start(candidate, fixture: dict, *, subject_id: str) -> dict:
    base = candidate.binding
    label = fixture["selection_label"]
    record = fixture["raw_example"]["records"][label]
    name_bytes = fixture["object_name"].encode("utf-8")
    raw_bytes = record["raw"].encode("utf-8")
    offset = raw_bytes.find(name_bytes)
    goal = fixture["goal"].encode("utf-8")
    state = initial_binding_state(base, subject_id)
    prepared = prepare(base, state, goal, fixture["records"])
    state, gate_index = observe_goal(base, prepared)
    previous = EOS
    raw: list[int] = []
    terminated = False
    with torch.no_grad():
        for step_index in range(MAX_PREDICTIONS):
            if step_index == 0:
                probe = answer_step(base, prepared, state, previous, gate_index=gate_index)
                gamma = torch.clamp(probe.gamma, min=GAMMA_FLOOR)
                step = forced_read(
                    base,
                    prepared,
                    state,
                    previous,
                    record_index=label,
                    offset=offset,
                    gate_index=gate_index,
                    gamma_override=gamma,
                )
            else:
                step = locked_step(
                    candidate,
                    prepared,
                    state,
                    previous,
                    gate_index=gate_index,
                    chosen_record=label,
                )
            endpoint = decode_endpoint(step.probabilities)
            if endpoint["eos"]:
                terminated = True
                break
            raw.append(int(endpoint["byte"]))
            m_next, coverage_next = advance_copy(
                base.reader,
                state,
                step.read,
                step.probabilities,
                step.gamma,
                endpoint["byte"],
                prepared.base.view,
            )
            state = replace(
                step.state,
                m=m_next,
                coverage=coverage_next,
                prefix_cursor=step.state.prefix_cursor + 1,
            )
            gate_index = step.gate_index
            previous = int(endpoint["byte"])
    scored = score_generation(bytes(raw), fixture["target_prefix"], terminated)
    return {
        "index": fixture["index"],
        "world": fixture["world"],
        "record": label,
        "name_offset": offset,
        "completed": scored["completed"],
        "object_exact": scored["object_exact"],
        "place_exact": scored["place_exact"],
        "first_divergence": scored["first_divergence"],
        "bytes": len(raw),
    }


def oracle_full_object_span(candidate, fixture: dict, *, subject_id: str) -> dict:
    """While generation stays in sync, force every object-name byte (positions
    3..14 of the 27-byte action) to be copied from the correct record with
    gamma=0; the template and place segments follow the locked model."""
    base = candidate.binding
    label = fixture["selection_label"]
    record = fixture["raw_example"]["records"][label]
    name_bytes = fixture["object_name"].encode("utf-8")
    raw_record = record["raw"].encode("utf-8")
    name_start = raw_record.find(name_bytes)
    target = fixture["target_prefix"]
    goal = fixture["goal"].encode("utf-8")
    state = initial_binding_state(base, subject_id)
    prepared = prepare(base, state, goal, fixture["records"])
    state, gate_index = observe_goal(base, prepared)
    previous = EOS
    raw: list[int] = []
    terminated = False
    forced_steps = 0
    with torch.no_grad():
        for _prediction in range(MAX_PREDICTIONS):
            emitted = len(raw)
            in_sync = all(raw[i] == target[i] for i in range(emitted))
            force = in_sync and 3 <= emitted < 15 and name_start >= 0
            if force:
                step = forced_read(
                    base,
                    prepared,
                    state,
                    previous,
                    record_index=label,
                    offset=name_start + (emitted - 3),
                    gate_index=gate_index,
                    gamma_override=0.0,
                )
                forced_steps += 1
            else:
                step = locked_step(
                    candidate,
                    prepared,
                    state,
                    previous,
                    gate_index=gate_index,
                    chosen_record=label,
                )
            endpoint = decode_endpoint(step.probabilities)
            if endpoint["eos"]:
                terminated = True
                break
            raw.append(int(endpoint["byte"]))
            m_next, coverage_next = advance_copy(
                base.reader,
                state,
                step.read,
                step.probabilities,
                step.gamma,
                endpoint["byte"],
                prepared.base.view,
            )
            state = replace(
                step.state,
                m=m_next,
                coverage=coverage_next,
                prefix_cursor=step.state.prefix_cursor + 1,
            )
            gate_index = step.gate_index
            previous = int(endpoint["byte"])
    scored = score_generation(bytes(raw), target, terminated)
    return {
        "index": fixture["index"],
        "world": fixture["world"],
        "forced_steps": forced_steps,
        "completed": scored["completed"],
        "object_exact": scored["object_exact"],
        "place_exact": scored["place_exact"],
        "first_divergence": scored["first_divergence"],
        "bytes": len(raw),
    }


def main() -> int:
    train, dev = load_fixtures()
    candidate, _state, _optimizer, profile, update_index, _cursor, _payload = load_bundle(
        ROOT / BUNDLE, torch.float64
    )
    candidate.configure_trainable()

    p1 = probe_order_blindness(candidate, dev)
    p2 = probe_memorization(candidate, train, dev)
    oracle_rows = [
        oracle_entity_start(candidate, fixture, subject_id=f"arch-oracle-{fixture['index']}")
        for fixture in dev[:40]
    ]
    p3 = {
        "examples": len(oracle_rows),
        "completed": sum(1 for row in oracle_rows if row["completed"]),
        "completed_rate": sum(1 for row in oracle_rows if row["completed"]) / len(oracle_rows),
        "object_exact_rate": sum(1 for row in oracle_rows if row["object_exact"])
        / len(oracle_rows),
        "place_exact_rate": sum(1 for row in oracle_rows if row["place_exact"]) / len(oracle_rows),
        "rows": oracle_rows,
        "note": (
            "First decoding step forced onto the name-span start of the correct record, then "
            "locked generation continues; this is the entity-start oracle (upper bound and "
            "reachability only, not a gate)."
        ),
    }

    p4_rows = [
        oracle_full_object_span(candidate, fixture, subject_id=f"arch-span-{fixture['index']}")
        for fixture in dev[:40]
    ]
    p4 = {
        "examples": len(p4_rows),
        "completed": sum(1 for row in p4_rows if row["completed"]),
        "completed_rate": sum(1 for row in p4_rows if row["completed"]) / len(p4_rows),
        "object_exact_rate": sum(1 for row in p4_rows if row["object_exact"]) / len(p4_rows),
        "place_exact_rate": sum(1 for row in p4_rows if row["place_exact"]) / len(p4_rows),
        "rows": p4_rows,
        "note": (
            "Full object-span oracle: while in sync, every object-name byte is force-copied "
            "from the correct record with gamma=0; template and place follow the locked model. "
            "Separates object-copy reachability from downstream template/place generation."
        ),
    }

    report = {
        "schema": "taiji-binding-select-arch-probe-v1",
        "step": "N16.1.3-arch-probe",
        "date": "2026-11-03",
        "status": "read_only_probe",
        "bundle": {
            "path": BUNDLE,
            "sha256": sha256_file(ROOT / BUNDLE),
            "profile": profile,
            "update_index": update_index,
        },
        "probes": {
            "P1_order_blindness": p1,
            "P2_memorization": p2,
            "P3_entity_start_oracle": p3,
            "P4_full_object_span_oracle": p4,
        },
        "accounting": {
            "updates": 0,
            "adam_calls": 0,
            "forward_training": 0,
            "inference_generations": p3["examples"] + p4["examples"],
            "confirm_examples": 0,
        },
        "cost": {
            "cpu_seconds": time.process_time() - STARTED_CPU,
            "wall_seconds": time.perf_counter() - STARTED_WALL,
        },
        "source_hashes": {
            "taiji/binding_select.py": sha256_file(ROOT / "taiji/binding_select.py"),
            COMPARE_SCRIPT: sha256_file(ROOT / COMPARE_SCRIPT),
            "scripts/training/probe_taiji_binding_select_16_1_arch.py": sha256_file(
                ROOT / "scripts/training/probe_taiji_binding_select_16_1_arch.py"
            ),
        },
        "limits": "Read-only probes on one stored bundle; no training, no confirm read.",
    }
    Path(ROOT / OUT).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"P1 invariant holds: {p1['invariant_holds']} (max diff {p1['max_abs_score_diff_over_rows']:.2e})"
    )
    print(f"P1 witness encoding diff: {p1['witness']['encoding_max_abs_diff']:.2e}")
    print(f"P2 train top1: {p2['train_top1']['rate']:.4f}, dev seen: {p2['dev_seen']['rate']:.4f}")
    print(
        f"P3 oracle: {p3['completed']}/{p3['examples']} completed "
        f"({p3['completed_rate']:.4f}), object_exact {p3['object_exact_rate']:.4f}"
    )
    print(
        f"P4 span oracle: {p4['completed']}/{p4['examples']} completed "
        f"({p4['completed_rate']:.4f}), object {p4['object_exact_rate']:.4f}, "
        f"place {p4['place_exact_rate']:.4f}"
    )
    print(
        f"cost: {report['cost']['cpu_seconds']:.2f} cpu / {report['cost']['wall_seconds']:.2f} wall"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
