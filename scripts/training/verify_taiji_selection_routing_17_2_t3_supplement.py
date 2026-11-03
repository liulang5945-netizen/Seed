"""N17.2 supplement for ledger T3: forced final-Q point mass, read-only.

T3 (frozen): "R2a final-Q correct-source-byte point mass. With ``q_null=gamma=0``
the correct byte probability must be f64 >= 1-1e-12 and f32 >= 1-1e-6.
A failure is first checked against the intervention and the label mapping; it
may not be replaced by extra position training."

Method (no training, 0 optimizer steps, 0 confirm reads):

* the same 40 dev examples as the 17.3 probe, teacher-forced over all 28 steps
  with arm ``B_stop_lock`` and an untrained ``OrderedSelectionHead`` (seed 1731);
* on each of the 720 record steps (12 object bytes + 6 place bytes per example)
  force ``alpha`` one-hot onto the gold record and ``beta`` one-hot onto the gold
  byte position of that record, set ``gamma_override=0`` so that
  ``probabilities = copy_byte_mass`` and ``q_null = 0``;
* take ``probabilities[gold_next_byte]`` in the f64 graph and replay the same
  forced read on a float32 copy of the reader;
* also record the transport row, the cursor component, ``q_bytes.sum()``,
  ``gamma_eff`` and ``q_null`` so a sub-threshold step can be attributed to
  transport diffusion, cursor mixing, or an intervention/mapping error.

The earlier route-argmax gate was not T3 and is demoted to a diagnostic
(``route_agreement``).  G1 (``a_hard`` one-hot) and G3 (mass split) stay as
structural side evidence only.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

torch.set_num_threads(1)

from taiji.binding_copy import (  # noqa: E402
    MAX_COVERAGE,
    forced_read,
    initial_binding_state,
    observe_goal,
    prepare,
)
from taiji.binding_select import load_bundle  # noqa: E402
from taiji.living_core import EOS  # noqa: E402
from taiji.selection_routing import (  # noqa: E402
    OrderedSelectionHead,
    RoutingConfig,
    _advanced_state,
    _s_t,
    build_context,
    route_labels,
    routing_step,
)

BUNDLE = Path("output/taiji_core_binding_select_16_1_compare/bundles/28031-A-locked-step300.pt")
COMPARE_SCRIPT = Path("scripts/training/verify_taiji_binding_select_16_1_compare.py")
SELECT_PAYLOAD = Path("output/taiji_core_binding_select_16_1/select_v1.json")
DATA_PAYLOAD = Path("output/taiji_core_binding_copy_15_1/data_v2.json")
SCRIPT = Path("scripts/training/verify_taiji_selection_routing_17_2_t3_supplement.py")
OUT = Path("reports/taiji_selection_routing_17_2_t3_supplement_20261103.json")
MANIFEST = Path("plans/manifests/taiji_selection_routing_17_2_t3_supplement_20261103.json")

ARM = "B_stop_lock"
HEAD_SEED = 1731
LIMIT = 40
TOL = 1e-12
TOL_F32 = 1e-6
TARGET_LENGTH = 28

STARTED_CPU = time.process_time()
STARTED_WALL = time.perf_counter()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_dev() -> tuple[tuple[dict, ...], dict]:
    spec = importlib.util.spec_from_file_location("cmp17_2", ROOT / COMPARE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _train, dev, select_payload, data = module.load_data()
    used = min(LIMIT, len(dev))
    audit = {
        "split": "dev",
        "dev_examples": len(dev),
        "used": used,
        "index_first": int(dev[0]["index"]),
        "index_last": int(dev[used - 1]["index"]),
        "data_examples": len(data["examples"]),
        "confirm_materialised": bool(select_payload.get("confirm_materialised", False)),
    }
    if audit["confirm_materialised"]:
        raise RuntimeError("confirm data is materialised; this supplement must not read it")
    return dev[:used], audit


def byte_mass(vector: torch.Tensor, byte_ids: torch.Tensor) -> torch.Tensor:
    out = vector.new_zeros(MAX_COVERAGE)
    if vector.numel():
        out.index_add_(0, byte_ids, vector)
    return out


def gold_offset(view, record_index: int, gold: int, previous: int) -> tuple[int, int, bool]:
    raw = view.raw[record_index]
    occurrences = [position for position, byte in enumerate(raw) if byte == gold]
    if not occurrences:
        raise RuntimeError(f"gold byte {gold} is absent from record {record_index}")
    matched = [
        position
        for position in occurrences
        if previous != EOS and position > 0 and raw[position - 1] == previous
    ]
    offset = matched[0] if matched else occurrences[0]
    return offset, len(occurrences), bool(matched)


def make_reader32(binding):
    reader32 = copy.deepcopy(binding.reader).float()
    object.__setattr__(reader32, "inherited", copy.deepcopy(binding.reader.inherited).float())
    return reader32


def forced_probability_f32(
    reader32,
    s_t: torch.Tensor,
    previous_embedding: torch.Tensor,
    goal_units: torch.Tensor,
    view32,
    m: torch.Tensor,
    alpha: torch.Tensor,
    betas: tuple[torch.Tensor, ...],
    gold: int,
) -> float:
    with torch.no_grad():
        read32 = reader32.read(
            s_t.float(),
            previous_embedding.float(),
            goal_units.float(),
            view32,
            m.float() if m.numel() else m,
            force={"alpha": alpha.float(), "betas": tuple(row.float() for row in betas)},
        )
    return float(read32.copy_byte_mass[gold])


def evaluate_fixture(binding, reader32, head, fixture: dict) -> tuple[dict, list[dict]]:
    goal = fixture["goal"].encode("utf-8")
    prepared = prepare(
        binding,
        initial_binding_state(binding, f"17.2-t3-{fixture['world']}-{fixture['index']}"),
        goal,
        fixture["records"],
    )
    state, gate_index = observe_goal(binding, prepared, gate_index=0)
    context = build_context(binding, goal, tuple(fixture["record_keys"]))
    route_index, _stage = route_labels(fixture)
    targets = tuple(fixture["target_bytes"])
    prefix = tuple(fixture["target_prefix"])
    record_count = len(fixture["record_keys"])
    if len(targets) != TARGET_LENGTH or len(prefix) != TARGET_LENGTH - 1:
        raise RuntimeError(f"target length drift at index {fixture['index']}")
    if len(route_index) != TARGET_LENGTH:
        raise RuntimeError(f"route label length drift at index {fixture['index']}")
    view = prepared.base.view
    view32 = replace(
        view,
        units=tuple(unit.float() for unit in view.units),
        metadata=view.metadata.float(),
    )
    rows: list[dict] = []
    previous = EOS
    with torch.no_grad():
        for t, gold in enumerate(targets):
            gold_byte = int(gold)
            step = routing_step(
                binding,
                prepared,
                state,
                previous,
                context,
                head,
                arm=ARM,
                gate_index=gate_index,
            )
            label = route_index[t]
            a_hard = step.a_hard
            a_sum = float(a_hard.sum())
            a_nonzero = int((a_hard != 0).sum())
            chosen = int(torch.argmax(a_hard))
            q_null = float(step.q_null)
            q_bytes_sum = float(step.q_bytes.sum()) if step.q_bytes.numel() else 0.0
            is_record = label < record_count
            check1 = bool(abs(a_sum - 1.0) <= TOL and a_nonzero == 1)
            if is_record:
                check3 = bool(abs(q_null) <= TOL and abs(q_bytes_sum - 1.0) <= TOL)
            else:
                check3 = bool(
                    abs(float(a_hard[record_count]) - 1.0) <= TOL
                    and abs(q_null - 1.0) <= TOL
                    and abs(q_bytes_sum) <= TOL
                )
            row = {
                "world": int(fixture["world"]),
                "index": int(fixture["index"]),
                "t": t,
                "kind": "record" if is_record else "null",
                "route": int(label),
                "chosen": chosen,
                "a_sum": a_sum,
                "a_nonzero": a_nonzero,
                "q_null_natural": q_null,
                "q_bytes_sum_natural": q_bytes_sum,
                "check1": check1,
                "check3": check3,
                "natural_gold": float(step.probabilities[gold_byte]),
                "natural_copy_gold": float(step.read.copy_byte_mass[gold_byte]),
                "natural_top1": False,
                "natural_entropy": None,
            }
            if is_record:
                offset, occurrences, context_matched = gold_offset(view, label, gold_byte, previous)
                forced = forced_read(
                    binding,
                    prepared,
                    state,
                    previous,
                    record_index=label,
                    offset=offset,
                    gate_index=gate_index,
                    gamma_override=0.0,
                )
                read = forced.read
                probabilities = forced.probabilities
                probability_sum = float(probabilities.sum())
                p64 = float(probabilities[gold_byte])
                alpha_null = float(read.alpha[-1])
                gamma_eff = float(forced.gamma)
                forced_q_null = float(read.q_null)
                forced_q_bytes_sum = float(read.q_bytes.sum())
                rebuilt = byte_mass(read.q_bytes, view.byte_ids)
                rebuild_error = float((rebuilt - read.copy_byte_mass).abs().max())
                s_mass = byte_mass(read.s_bytes, view.byte_ids)
                s_gold = float(s_mass[gold_byte])
                record_slice = view.slice(label)
                row_values = read.s_bytes[record_slice]
                row_gold = float(row_values[offset])
                row_argmax = int(torch.argmax(row_values))
                row_delta = bool(abs(float(row_values.max()) - 1.0) <= TOL)
                lam = float(read.continuation)
                m_mass = float(read.m_mass)
                m_gold = float(byte_mass(state.m, view.byte_ids)[gold_byte])
                transport_component = (1.0 - lam * m_mass) * s_gold
                cursor_component = lam * m_gold
                p32 = forced_probability_f32(
                    reader32,
                    _s_t(binding, prepared, state, previous),
                    binding.subject.encoder.embedding(torch.tensor(previous)),
                    prepared.base.goal_units,
                    view32,
                    state.m,
                    read.alpha,
                    read.betas,
                    gold_byte,
                )
                natural_q = step.q_bytes
                if natural_q.numel() and float(natural_q.sum()) > 0.0:
                    masses = byte_mass(natural_q.detach(), view.byte_ids)
                    row["natural_top1"] = bool(int(torch.argmax(masses)) == gold_byte)
                    weights = natural_q.detach() / natural_q.detach().sum()
                    entropy = float(-(weights * weights.clamp_min(1e-300).log()).sum())
                    row["natural_entropy"] = entropy
                row.update(
                    {
                        "gold": gold_byte,
                        "offset": offset,
                        "occurrences": occurrences,
                        "context_matched": context_matched,
                        "p64": p64,
                        "p32": p32,
                        "pass64": bool(p64 >= 1.0 - TOL),
                        "pass32": bool(p32 >= 1.0 - TOL_F32),
                        "probability_sum": probability_sum,
                        "alpha_null": alpha_null,
                        "gamma_eff": gamma_eff,
                        "q_null_forced": forced_q_null,
                        "q_bytes_sum_forced": forced_q_bytes_sum,
                        "rebuild_error": rebuild_error,
                        "transport_gold": s_gold,
                        "transport_row_gold": row_gold,
                        "transport_row_argmax": row_argmax,
                        "transport_row_delta": row_delta,
                        "lambda": lam,
                        "m_mass": m_mass,
                        "cursor_gold": cursor_component,
                        "transport_component": transport_component,
                    }
                )
            rows.append(row)
            if t == len(targets) - 1:
                break
            accepted = prefix[t]
            state, _, gate_index = _advanced_state(binding, prepared, step, accepted)
            previous = accepted
    record_rows = [row for row in rows if row["kind"] == "record"]
    summary = {
        "world": int(fixture["world"]),
        "index": int(fixture["index"]),
        "steps": len(rows),
        "record_steps": len(record_rows),
        "pass64": sum(1 for row in record_rows if row["pass64"]),
        "pass32": sum(1 for row in record_rows if row["pass32"]),
        "min_p64": min(row["p64"] for row in record_rows),
        "min_p32": min(row["p32"] for row in record_rows),
        "transport_delta_steps": sum(1 for row in record_rows if row["transport_row_delta"]),
        "route_matches": sum(1 for row in rows if row["chosen"] == row["route"]),
        "fail_steps": [
            {
                key: row[key]
                for key in (
                    "t",
                    "route",
                    "gold",
                    "offset",
                    "p64",
                    "p32",
                    "transport_gold",
                    "transport_row_gold",
                    "transport_row_argmax",
                    "lambda",
                    "m_mass",
                    "cursor_gold",
                    "q_bytes_sum_forced",
                    "q_null_forced",
                    "gamma_eff",
                )
            }
            for row in record_rows
            if not row["pass64"]
        ],
    }
    return summary, rows


def span(values: list[float]) -> dict:
    if not values:
        return {"min": None, "median": None, "max": None}
    return {"min": min(values), "median": statistics.median(values), "max": max(values)}


def main() -> int:
    dev, audit = load_dev()
    bundle_sha = sha256_file(ROOT / BUNDLE)
    binding = load_bundle(ROOT / BUNDLE, torch.float64)[0].binding
    reader32 = make_reader32(binding)
    head = OrderedSelectionHead(RoutingConfig(), seed=HEAD_SEED)

    summaries = []
    readings: list[dict] = []
    for fixture in dev:
        summary, rows = evaluate_fixture(binding, reader32, head, fixture)
        summaries.append(summary)
        readings.extend(rows)
        print(
            json.dumps(
                {
                    "world": summary["world"],
                    "index": summary["index"],
                    "pass64": summary["pass64"],
                    "min_p64": summary["min_p64"],
                },
                ensure_ascii=False,
            )
        )

    record_rows = [row for row in readings if row["kind"] == "record"]
    p64 = [row["p64"] for row in record_rows]
    p32 = [row["p32"] for row in record_rows]
    fail64 = [row for row in record_rows if not row["pass64"]]
    fail32 = [row for row in record_rows if not row["pass32"]]
    transport_fail = [row for row in fail64 if row["transport_component"] < 1.0 - TOL]
    cursor_only = [
        row
        for row in fail64
        if row["transport_component"] >= 1.0 - TOL and row["cursor_gold"] > 0.0
    ]
    instrument_errors = [
        row
        for row in record_rows
        if max(
            abs(row["probability_sum"] - 1.0),
            row["rebuild_error"],
            abs(row["alpha_null"]),
            abs(row["q_null_forced"]),
            abs(row["gamma_eff"]),
            abs(row["q_bytes_sum_forced"] - 1.0),
            abs(row["p64"] - row["transport_component"] - row["cursor_gold"]),
        )
        > TOL
    ]
    if instrument_errors:
        layer = "instrument_or_intervention_mapping"
        explanation = (
            "forced probabilities, q_bytes normalization, gamma/q_null or the byte "
            "index_add mapping deviate from the forced one-hot contract"
        )
    elif not fail64 and not fail32:
        layer = "none"
        explanation = "every forced record step keeps the gold byte at point mass"
    elif transport_fail:
        layer = "architecture_transport_softmax_diffusion"
        explanation = (
            "one-hot alpha/beta is spread by reader._transport, whose rows are softmax, "
            "not the identity; the arm has no post-transport R2a point-mass insertion"
        )
    else:
        layer = "architecture_cursor_mixing"
        explanation = "the transport row at the gold position is not the limiter"
    representative = (
        min(fail64, key=lambda row: row["p64"])
        if fail64
        else min(record_rows, key=lambda row: row["p64"])
    )

    route_matched = sum(1 for row in readings if row["chosen"] == row["route"])
    natural_gold = [row["natural_gold"] for row in record_rows]
    natural_entropy = [
        row["natural_entropy"] for row in record_rows if row["natural_entropy"] is not None
    ]
    natural_top1 = sum(1 for row in record_rows if row["natural_top1"])
    g1_fail = sum(1 for row in readings if not row["check1"])
    g3_fail = sum(1 for row in readings if not row["check3"])

    t3 = {
        "f64_threshold": 1.0 - TOL,
        "f32_threshold": 1.0 - TOL_F32,
        "steps": len(record_rows),
        "f64": {
            **span(p64),
            "passed_steps": len(record_rows) - len(fail64),
            "failed_steps": len(fail64),
            "passed": not fail64,
        },
        "f32": {
            **span(p32),
            "passed_steps": len(record_rows) - len(fail32),
            "failed_steps": len(fail32),
            "passed": not fail32,
        },
        "passed": bool(not fail64 and not fail32),
    }

    report = {
        "schema": "taiji-selection-routing-17-2-t3-supplement-v2",
        "step": "N17.2-supplement",
        "date": "2026-11-03",
        "t3_definition": {
            "ledger": "INVARIANT-LEDGER_20261103 T3",
            "quote": (
                "R2a 最终Q正确源字节一点质量、q_null=gamma=0 时，正确字节概率 "
                "f64≥1−1e-12、f32≥1−1e-6"
            ),
            "failure_rule": "失败先查干预和映射，不能用追加位置训练代替",
            "intervention": (
                "alpha one-hot gold record, beta one-hot gold byte position of that record, "
                "gamma_override=0, q_null=0"
            ),
            "population": "40 dev examples x 18 record steps = 720 steps",
            "gate": "f64 >= 1-1e-12 and f32 >= 1-1e-6 at every record step",
            "excluded": "argmax(a_hard) vs gold route is not T3; kept only as route_agreement",
        },
        "passed": t3["passed"],
        "n": len(summaries),
        "steps": len(record_rows),
        "arm": ARM,
        "head": {"class": "OrderedSelectionHead", "seed": HEAD_SEED, "trained": False},
        "bundle": {
            "path": str(BUNDLE),
            "bytes": (ROOT / BUNDLE).stat().st_size,
            "sha256": bundle_sha,
        },
        "data": audit,
        "forced_gold_probability": t3,
        "diagnostics": {
            "unforced_gold_probability": {
                **span(natural_gold),
                "note": "includes the generated branch, not copy alone",
            },
            "unforced_copy_byte_mass": span([row["natural_copy_gold"] for row in record_rows]),
            "unforced_q_bytes": {
                "top1_equals_gold_steps": natural_top1,
                "steps": len(record_rows),
                "entropy_median": statistics.median(natural_entropy) if natural_entropy else None,
            },
            "transport_rows": {
                "steps": len(record_rows),
                "row_delta_steps": sum(1 for row in record_rows if row["transport_row_delta"]),
                "row_gold": span([row["transport_row_gold"] for row in record_rows]),
                "argmax_at_gold_steps": sum(
                    1 for row in record_rows if row["transport_row_argmax"] == row["offset"]
                ),
            },
            "gold_position_alignment": {
                "steps": len(record_rows),
                "context_matched_steps": sum(1 for row in record_rows if row["context_matched"]),
                "unique_occurrence_steps": sum(1 for row in record_rows if row["occurrences"] == 1),
            },
            "route_agreement": {
                "gate": False,
                "steps": len(readings),
                "matched": route_matched,
                "rate": float(route_matched) / len(readings),
            },
            "structural_g1_a_hard_one_hot": {
                "gate": False,
                "steps": len(readings),
                "failed_steps": g1_fail,
                "a_hard_sum": span([row["a_sum"] for row in readings]),
                "a_hard_nonzero": span([float(row["a_nonzero"]) for row in readings]),
            },
            "structural_g3_mass": {
                "gate": False,
                "steps": len(readings),
                "failed_steps": g3_fail,
                "record_q_null_abs_max": max(
                    (abs(row["q_null_natural"]) for row in record_rows), default=0.0
                ),
                "record_q_bytes_sum_min": min(
                    (row["q_bytes_sum_natural"] for row in record_rows), default=0.0
                ),
            },
            "instrument": {
                "probability_sum_abs_error_max": max(
                    abs(row["probability_sum"] - 1.0) for row in record_rows
                ),
                "copy_byte_mass_rebuild_error_max": max(
                    row["rebuild_error"] for row in record_rows
                ),
                "alpha_null_abs_max": max(abs(row["alpha_null"]) for row in record_rows),
                "q_null_forced_abs_max": max(abs(row["q_null_forced"]) for row in record_rows),
                "gamma_eff_abs_max": max(abs(row["gamma_eff"]) for row in record_rows),
                "q_bytes_sum_minus_one_abs_max": max(
                    abs(row["q_bytes_sum_forced"] - 1.0) for row in record_rows
                ),
                "component_identity_error_max": max(
                    abs(row["p64"] - row["transport_component"] - row["cursor_gold"])
                    for row in record_rows
                ),
                "error_steps": len(instrument_errors),
            },
        },
        "failure_layer": {
            "layer": layer,
            "explanation": explanation,
            "steps_below_f64": len(fail64),
            "steps_below_f32": len(fail32),
            "transport_diffusion_steps": len(transport_fail),
            "cursor_mixing_steps": len(cursor_only),
            "instrument_error_steps": len(instrument_errors),
            "representative": {
                key: representative[key]
                for key in (
                    "world",
                    "index",
                    "t",
                    "route",
                    "gold",
                    "offset",
                    "occurrences",
                    "context_matched",
                    "p64",
                    "p32",
                    "transport_gold",
                    "transport_row_gold",
                    "transport_row_argmax",
                    "transport_row_delta",
                    "lambda",
                    "m_mass",
                    "cursor_gold",
                    "transport_component",
                    "q_bytes_sum_forced",
                    "gamma_eff",
                    "q_null_forced",
                    "probability_sum",
                    "rebuild_error",
                )
            },
        },
        "records": summaries,
        "limits": {"training_updates": 0, "optimizer_steps": 0, "confirm_reads": 0},
        "timing": {
            "cpu_seconds": time.process_time() - STARTED_CPU,
            "wall_seconds": time.perf_counter() - STARTED_WALL,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "schema": "taiji-selection-routing-17-2-t3-supplement-manifest-v2",
        "step": "N17.2-supplement",
        "date": "2026-11-03",
        "passed": t3["passed"],
        "report": {
            "path": str(OUT),
            "bytes": OUT.stat().st_size,
            "sha256": sha256_file(OUT),
        },
        "source_sha256": {
            path: sha256_file(ROOT / path)
            for path in (
                "taiji/selection_routing.py",
                "taiji/binding_copy.py",
                "taiji/binding_select.py",
                "taiji/record_relation.py",
                "taiji/record_relation_core.py",
                "taiji/living_core.py",
                "scripts/training/verify_taiji_binding_select_16_1_compare.py",
                str(SCRIPT),
            )
        },
        "parent_bundle": {
            "path": str(BUNDLE),
            "bytes": (ROOT / BUNDLE).stat().st_size,
            "sha256": bundle_sha,
        },
        "data": [
            {
                "path": str(path),
                "bytes": (ROOT / path).stat().st_size,
                "sha256": sha256_file(ROOT / path),
            }
            for path in (SELECT_PAYLOAD, DATA_PAYLOAD)
        ],
        "criteria": {
            "t3_quote": (
                "R2a 最终Q正确源字节一点质量、q_null=gamma=0 时，正确字节概率 "
                "f64≥1−1e-12、f32≥1−1e-6"
            ),
            "thresholds": {"f64": 1.0 - TOL, "f32": 1.0 - TOL_F32, "tolerance": TOL},
            "intervention": (
                "alpha one-hot gold record, beta one-hot gold byte position, "
                "gamma_override=0, q_null=0"
            ),
            "gate": "all 720 record steps pass in both dtypes",
            "f32_method": (
                "forced read replayed on a float32 copy of the reader with the same state; "
                "routing_step elsewhere is float64 by contract"
            ),
            "demoted": "route argmax is a diagnostic, not a gate",
            "frozen": "never relaxed",
        },
        "readings": {
            "n": len(summaries),
            "steps": len(record_rows),
            "p64": {**span(p64), "failed_steps": len(fail64)},
            "p32": {**span(p32), "failed_steps": len(fail32)},
            "transport_diffusion_steps": len(transport_fail),
            "cursor_mixing_steps": len(cursor_only),
            "instrument_error_steps": len(instrument_errors),
            "transport_row_delta_steps": sum(
                1 for row in record_rows if row["transport_row_delta"]
            ),
            "gold_position_context_matched_steps": sum(
                1 for row in record_rows if row["context_matched"]
            ),
            "failure_layer": layer,
            "route_agreement_rate": float(route_matched) / len(readings),
        },
        "products": {
            "report": {
                "path": str(OUT),
                "bytes": OUT.stat().st_size,
                "sha256": sha256_file(OUT),
            },
            "script": {
                "path": str(SCRIPT),
                "bytes": (ROOT / SCRIPT).stat().st_size,
                "sha256": sha256_file(ROOT / SCRIPT),
            },
        },
        "limits": {"training_updates": 0, "optimizer_steps": 0, "confirm_reads": 0},
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": t3["passed"],
                "steps": len(record_rows),
                "p64_min": min(p64),
                "p64_median": statistics.median(p64),
                "p64_max": max(p64),
                "p32_min": min(p32),
                "steps_below_f64": len(fail64),
                "failure_layer": layer,
                "cpu_seconds": report["timing"]["cpu_seconds"],
                "wall_seconds": report["timing"]["wall_seconds"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if t3["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
