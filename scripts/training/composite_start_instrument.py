"""D16-5 instrument: the composite start position actually used by the decode.

The 19.5 stage gate read ``start_hit`` as ``route_hit and argmax(read.betas) ==
gold_offset``.  ``read.betas`` is the frozen reader's own position posterior and
does not contain the teacher-span pointer, so span supervision could never move
that number.  This module defines one measurement on the quantity the decode
actually lands on:

1. ``alpha[i]`` scales the record mass and ``betas[i]`` picks the source byte:
   the transport source is ``argmax(alpha[i] * betas[i])`` (an alpha scalar
   cannot reorder the row, so the source is beta's mode whenever alpha > 0);
2. the transport row is ``transport_logits = transport(q_w, units)``;
3. the span head adds its start bias: ``row_logits = transport_logits + bias``
   (``taiji/binding_copy.py`` ``_point_transport``);
4. a committed carry (m peak or explicit cursor) bypasses the sum and sets
   ``pointer = row_logits = one-hot(carried position)``.

``composite_start_hit`` is therefore read on ``row_logits`` (the final row
argmax), conditioned on the alpha route owning the gold record.  ``route_hit``
stays a separate 0.95 clause.  ``beta_start_hit`` and ``pointer_start_hit``
(pure span bias) are reported in the same table, never starred or cherry-picked.
The 19.5 numbers stay in ``legacy_instrument`` untouched.

gamma is the generation weight and 1 - gamma the copy weight; nothing here
infers training credit (M3).
"""

from __future__ import annotations

import math
import statistics
from typing import Any

import probe_taiji_content_transport_18_1 as probe_mod
import torch

from taiji.binding_copy import observe_goal
from taiji.living_core import EOS
from taiji.selection_routing import _advanced_state, routing_step

OBJECT_POSITIONS = probe_mod.OBJECT_POSITIONS
PLACE_POSITIONS = probe_mod.PLACE_POSITIONS


def _mean(values) -> float:
    finite = [float(value) for value in values if value == value]
    return statistics.fmean(finite) if finite else 0.0


def measure_fixture(probe, fixture: dict) -> dict:
    """Teacher-forced read of every content position with the composite row.

    The gold byte is fed after each step, so the route keeps its own trajectory
    and the sequence never drifts.  Nothing is trained and no prefix override is
    used.
    """
    prepared, context = probe.prepare(fixture)
    state, gate_index = observe_goal(probe.candidate, prepared, gate_index=0)
    target = tuple(fixture["target_bytes"])
    mapping = probe_mod.content_positions(fixture)
    offsets = prepared.base.view.record_offsets
    arm = getattr(probe, "arm", "A_route_lock")
    previous = EOS
    steps = []
    with torch.no_grad():
        for position in range(len(target)):
            step = routing_step(
                probe.candidate,
                prepared,
                state,
                previous,
                context,
                probe.candidate.head,
                arm=arm,
                gate_index=gate_index,
                span_head=probe.candidate.span_head,
                pointer_temperature=1.0,
                pointer_hard=True,
            )
            predicted = int(torch.argmax(step.probabilities))
            record, offset = mapping.get(position, (-1, -1))
            entry: dict[str, Any] = {
                "position": position,
                "hit": predicted == target[position],
                "gamma": float(step.gamma_eff.detach()),
                "record": record,
                "offset": offset,
                "route_peak": int(torch.argmax(step.a_hard)),
                "route_hit": None,
                "beta_peak": -1,
                "source_peak": -1,
                "source_equals_beta_peak": None,
                "pointer_bias_peak": -1,
                "prebias_peak": -1,
                "composite_peak": -1,
                "pointer_peak": -1,
                "row_source": "none",
                "composite_identity_maxdiff": None,
                "pointer_onehot_at_composite": None,
                "beta_row_hit": None,
                "pointer_row_hit": None,
                "composite_row_hit": None,
                "carry_active": None,
                "carry_hit": None,
                "decode_copy_peak_flat": -1,
                "decode_peak_matches_composite": None,
                "decode_hit": None,
            }
            if record >= 0:
                read = step.read
                route_hit = int(torch.argmax(step.a_hard)) == record
                beta_peak = int(torch.argmax(read.betas[record]))
                source_peak = int(torch.argmax(read.alpha[record] * read.betas[record]))
                bias = read.row_bias[record] if len(read.row_bias) else None
                prebias = read.row_prebias[record] if len(read.row_prebias) else None
                row = read.pointer_logits[record]
                composite_peak = int(torch.argmax(row))
                pointer_peak = int(torch.argmax(read.pointer[record]))
                source = read.row_source[record] if len(read.row_source) else "unknown"
                identity = None
                if prebias is not None and bias is not None:
                    identity = float((row - (prebias + bias)).abs().max())
                onehot = None
                if len(read.pointer[record]):
                    onehot = (
                        float(read.pointer[record][composite_peak]) == 1.0
                        and float(read.pointer[record].sum()) == 1.0
                    )
                flat = offsets[record][0] + composite_peak
                copy_peak = int(torch.argmax(step.q_bytes)) if len(step.q_bytes) else -1
                entry.update(
                    {
                        "route_hit": route_hit,
                        "beta_peak": beta_peak,
                        "source_peak": source_peak,
                        "source_equals_beta_peak": source_peak == beta_peak,
                        "pointer_bias_peak": (int(torch.argmax(bias)) if bias is not None else -1),
                        "prebias_peak": (int(torch.argmax(prebias)) if prebias is not None else -1),
                        "composite_peak": composite_peak,
                        "pointer_peak": pointer_peak,
                        "row_source": source,
                        "composite_identity_maxdiff": identity,
                        "pointer_onehot_at_composite": onehot,
                        "beta_row_hit": beta_peak == offset,
                        "pointer_row_hit": (
                            int(torch.argmax(bias)) == offset if bias is not None else False
                        ),
                        "composite_row_hit": composite_peak == offset,
                        "carry_active": source == "carry",
                        "carry_hit": int(read.m_peak) == offsets[record][0] + offset,
                        "decode_copy_peak_flat": copy_peak,
                        "decode_peak_matches_composite": copy_peak == flat,
                        "decode_hit": copy_peak == offsets[record][0] + offset,
                    }
                )
            steps.append(entry)
            state, _coverage, gate_index = _advanced_state(
                probe.candidate, prepared, step, target[position]
            )
            previous = target[position]
    return {"steps": steps}


def metrics(ledger: dict) -> dict:
    """Per-example start claims and tail hits for both entities."""
    steps = {step["position"]: step for step in ledger["steps"]}
    out = {"gamma_mean": _mean(step["gamma"] for step in ledger["steps"])}
    for name, positions in (("object", OBJECT_POSITIONS), ("place", PLACE_POSITIONS)):
        start = steps[positions[0]]
        tail = [steps[position] for position in positions[1:]]
        entity = [steps[position] for position in positions]
        out[name] = {
            "start": start,
            "tail_hits": sum(1 for step in tail if step["hit"]),
            "tail_total": len(tail),
            "entity_byte_hits_all": sum(1 for step in entity if step["hit"]),
            "beta_hit_rate": sum(1 for step in entity if step["beta_row_hit"]) / len(entity),
            "route_hit_rate": sum(1 for step in entity if step["route_hit"]) / len(entity),
        }
    return out


_CONDITIONS = {
    "composite": lambda start: bool(start["route_hit"]) and bool(start["composite_row_hit"]),
    "beta": lambda start: bool(start["route_hit"]) and bool(start["beta_row_hit"]),
    "pointer": lambda start: bool(start["route_hit"]) and bool(start["pointer_row_hit"]),
    "composite_row_only": lambda start: bool(start["composite_row_hit"]),
    "beta_row_only": lambda start: bool(start["beta_row_hit"]),
    "pointer_row_only": lambda start: bool(start["pointer_row_hit"]),
}


def _conditional(entries: list[dict], condition) -> dict:
    selected = [entry for entry in entries if condition(entry["start"])]
    hits = sum(entry["tail_hits"] for entry in selected)
    total = sum(entry["tail_total"] for entry in selected)
    return {
        "byte": hits / total if total else 0.0,
        "denominator_bytes": total,
        "examples": len(selected),
    }


def summarize(entries_all: list[dict]) -> dict:
    """Aggregate the composite instrument over examples.

    ``start_hit_rate`` is kept as the route-and-beta alias so the archived 19.5
    shape still parses; every new reading is named explicitly.
    """
    out: dict[str, dict] = {}
    for name in ("object", "place"):
        entries = [entry[name] for entry in entries_all]
        starts = [entry["start"] for entry in entries]
        rates = {
            key: sum(1 for start in starts if condition(start)) / len(starts)
            for key, condition in _CONDITIONS.items()
        }
        conditionals = {
            key: _conditional(entries, condition) for key, condition in _CONDITIONS.items()
        }
        all_hits = sum(entry["tail_hits"] for entry in entries)
        all_total = sum(entry["tail_total"] for entry in entries)
        identity = [
            entry["composite_identity_maxdiff"]
            for entry in starts
            if entry["composite_identity_maxdiff"] is not None
            and math.isfinite(entry["composite_identity_maxdiff"])
        ]
        out[name] = {
            "examples": len(entries),
            "route_hit_rate": _mean(entry["route_hit_rate"] for entry in entries),
            "beta_hit_rate": _mean(entry["beta_hit_rate"] for entry in entries),
            "start_hit_rate": rates["beta"],
            "legacy_start_hit_rate": rates["beta"],
            "beta_start_hit_rate": rates["beta"],
            "pointer_start_hit_rate": rates["pointer"],
            "composite_start_hit_rate": rates["composite"],
            "beta_row_hit_rate": rates["beta_row_only"],
            "pointer_row_hit_rate": rates["pointer_row_only"],
            "composite_row_hit_rate": rates["composite_row_only"],
            "carry_active_rate": sum(1 for start in starts if start["carry_active"]) / len(starts),
            "carry_start_hit_rate": sum(1 for start in starts if start["carry_hit"]) / len(starts),
            "start_byte_hit_rate": sum(1 for start in starts if start["hit"]) / len(starts),
            "decode_start_hit_rate": sum(1 for start in starts if start["decode_hit"])
            / len(starts),
            "source_equals_beta_peak_rate": sum(
                1 for start in starts if start["source_equals_beta_peak"]
            )
            / len(starts),
            "pointer_onehot_at_composite_rate": sum(
                1 for start in starts if start["pointer_onehot_at_composite"]
            )
            / len(starts),
            "decode_peak_matches_composite_rate": sum(
                1 for start in starts if start["decode_peak_matches_composite"]
            )
            / len(starts),
            "composite_identity_maxdiff": max(identity) if identity else 0.0,
            "conditional_byte": conditionals["composite"]["byte"],
            "conditional_denominator_bytes": conditionals["composite"]["denominator_bytes"],
            "conditional_examples": conditionals["composite"]["examples"],
            "start_hit_examples": conditionals["composite"]["examples"],
            "conditional_byte_beta_diagnostic": conditionals["beta"]["byte"],
            "conditional_byte_pointer_diagnostic": conditionals["pointer"]["byte"],
            "conditional_byte_composite_row_only_diagnostic": conditionals["composite_row_only"][
                "byte"
            ],
            "conditional_byte_beta_row_only_diagnostic": conditionals["beta_row_only"]["byte"],
            "conditional_byte_pointer_row_only_diagnostic": conditionals["pointer_row_only"][
                "byte"
            ],
            "unconditional_byte": all_hits / all_total if all_total else 0.0,
        }
    out["gamma_mean"] = _mean(entry["gamma_mean"] for entry in entries_all)
    out["row_source_at_start"] = {
        name: {
            source: sum(1 for entry in entries_all if entry[name]["start"]["row_source"] == source)
            for source in ("transport", "carry", "forced", "none", "unknown")
        }
        for name in ("object", "place")
    }
    return out
