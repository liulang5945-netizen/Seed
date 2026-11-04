"""18.2 probe: explicit byte cursor + UTF-8 endpoint mask, 0/2 engineering updates.

Read-only relative to the frozen 18.1 evidence: a *new* evidence file that is
side-carried next to 18.1 (ledger §8 naming rule).  No confirm examples, no
pricing, no task-training comparison.  Budget: <= 60 engineering Adam / 400 CPU.

Conditions
----------
b_pure  gold record + gold start intervention + gamma=0, natural betas: the
        18.1 baseline condition, now with the carried cursor in the read.
c_gold  every content position pinned (18.1 mode c), gamma=0: upper bound.
free    routing_step free decode with the UTF-8 mask; gamma overrides 1.0
        (generation-only) and 0.0 (copy-only) for the source split.

Readings
--------
* object 12 bytes / place 6 bytes per-position hit rate, first and last byte;
* free-decode UTF-8 legal rate and pre-mask illegal-byte count, split by source;
* bundle round-trip: parameters / Adam / m+coverage / prefix_cursor+copy_cursor.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "training"))

import torch  # noqa: E402

torch.set_num_threads(1)

import probe_taiji_content_transport_18_1 as p18  # noqa: E402

from taiji import data_premise_18_0 as premise  # noqa: E402
from taiji.binding_copy import (  # noqa: E402
    MAX_COVERAGE,
    BindingRead,
    advance_copy,
    advance_copy_cursor,
    observe_goal,
)
from taiji.living_core import EOS  # noqa: E402
from taiji.record_relation_core import _utf8_state as utf8_state  # noqa: E402
from taiji.record_relation_core import decode_endpoint, utf8_allowed_mask  # noqa: E402
from taiji.selection_routing import (  # noqa: E402
    _advanced_state,
)
from taiji.selection_routing_checkpoint import (  # noqa: E402
    build_routing_candidate,
    load_routing_bundle,
    save_routing_bundle,
)

PARENT = Path("output/taiji_core_binding_select_16_1_compare/bundles/28031-A-locked-step300.pt")
OUTPUT_DIR = Path("output/taiji_cursor_18_2")
REPORT = Path("reports/taiji_cursor_utf8_18_2_20261104.json")
EVIDENCE = Path("reports/taiji_cursor_utf8_18_2_probe_20261104.json")

CONTENT = p18.CONTENT_POSITIONS
OBJECT = p18.OBJECT_POSITIONS
PLACE = p18.PLACE_POSITIONS

GATES = {
    "b_pure_content": 0.95,
    "first_last_delta": 0.02,
    "utf8_free": 0.99,
}
LIMIT_EXAMPLES = 16
UPDATES = 2


class CursorProbe(p18.ProbeContext):
    def read_natural(self, prepared, state, previous):
        base = prepared.base
        view = base.view
        subject = self.subject
        previous_embedding = subject.encoder.embedding(
            torch.tensor(previous, device=subject.W.device)
        )
        s_t = p18._s_t(self.candidate, prepared, state, previous)
        bias = None
        if self.candidate.span_head is not None and view.record_count:
            bias = tuple(
                self.candidate.span_head.start(view.units[index]).squeeze(-1)
                for index in range(view.record_count)
            )
        return self.candidate.reader.read(
            s_t,
            previous_embedding,
            base.goal_units,
            view,
            state.m,
            pointer_bias=bias,
            pointer_temperature=1.0,
            pointer_hard=True,
            cursor=(state.copy_record, state.copy_cursor),
        )

    def forced_step(self, prepared, state, previous, gate_index, force, m_override, gamma_override):
        base = prepared.base
        view = base.view
        subject = self.subject
        previous_embedding = subject.encoder.embedding(
            torch.tensor(previous, device=subject.W.device)
        )
        s_t = p18._s_t(self.candidate, prepared, state, previous)
        read = self.candidate.reader.read(
            s_t,
            previous_embedding,
            base.goal_units,
            view,
            state.m,
            force=force,
            m_override=m_override,
            pointer_temperature=1.0,
            pointer_hard=True,
            cursor=(state.copy_record, state.copy_cursor),
        )
        return self._tail(prepared, state, previous, read, gamma_override, gate_index)


def advance_custom(candidate, prepared, state, read, probabilities, gamma_eff, accepted, **rest):
    """18.1 advance with the carried cursor and prefix counter."""
    view = prepared.base.view
    copy_byte_mass = torch.zeros(MAX_COVERAGE, dtype=torch.float64)
    q_bytes = rest["q_bytes"]
    if len(q_bytes):
        copy_byte_mass = copy_byte_mass.index_add(0, view.byte_ids, q_bytes)
    locked = BindingRead(
        content=rest["content"],
        copy_weights=q_bytes,
        copy_byte_mass=copy_byte_mass,
        alpha=read.alpha,
        betas=read.betas,
        alpha_null=rest["q_null"],
        record_mass=read.record_mass,
        s_bytes=read.s_bytes,
        q_bytes=q_bytes,
        q_null=rest["q_null"],
        m_mass=read.m_mass,
        continuation=read.continuation,
        forced=False,
    )
    next_m, next_coverage = advance_copy(
        candidate.reader, state, locked, probabilities, gamma_eff, accepted, view
    )
    route_record = -1
    if view.record_count:
        route_index = int(torch.argmax(read.alpha))
        if route_index < view.record_count:
            route_record = route_index
    copy_record, copy_cursor = advance_copy_cursor(state, read, view, accepted, route_record)
    living = replace(state.living, h=rest["next_h"])
    return replace(
        state,
        living=living,
        m=next_m,
        coverage=next_coverage,
        prefix_cursor=state.prefix_cursor + 1,
        copy_record=copy_record,
        copy_cursor=copy_cursor,
    )


def content_run(ctx, fixture, mode, gamma_override=0.0):
    prepared, _context = ctx.prepare(fixture)
    state, gate_index = observe_goal(ctx.candidate, prepared, gate_index=0)
    target = tuple(fixture["target_bytes"])
    mapping = p18.content_positions(fixture)
    record_count = len(fixture["record_keys"])
    object_record = int(fixture["selection_label"])
    place_record = p18._place_record(fixture)
    raw: list[int] = []
    previous = EOS
    hits: list[bool] = []
    cursor_trace: list[int] = []
    trace_rows: list[dict] = []
    state_mid = None
    with torch.no_grad():
        for position in range(len(target)):
            if position in mapping:
                record, offset = mapping[position]
                alpha = p18._one_hot(record_count + 1, record)
                if mode == "c":
                    lengths = [
                        int(prepared.base.view.units[index].shape[0])
                        for index in range(record_count)
                    ]
                    betas = tuple(
                        (
                            p18._one_hot(lengths[index], offset if index == record else 0)
                            if index == record
                            else torch.zeros(lengths[index])
                        )
                        for index in range(record_count)
                    )
                    pointer = tuple(
                        (
                            p18._one_hot(lengths[index], offset if index == record else 0)
                            if index == record
                            else torch.zeros(lengths[index])
                        )
                        for index in range(record_count)
                    )
                    force = {"alpha": alpha, "betas": betas, "pointer": pointer}
                    m_override = torch.zeros(len(prepared.base.view.byte_ids))
                    m_override[prepared.base.view.record_offsets[record][0] + offset] = 1.0
                else:
                    natural = ctx.read_natural(prepared, state, previous)
                    force = {"alpha": alpha, "betas": natural.betas}
                    m_override = None
                    if position in (3, 21):
                        start_record = object_record if position == 3 else place_record
                        start_offset = mapping[position][1]
                        m_override = torch.zeros(len(prepared.base.view.byte_ids))
                        m_override[
                            prepared.base.view.record_offsets[start_record][0] + start_offset
                        ] = 1.0
                out = ctx.forced_step(
                    prepared, state, previous, gate_index, force, m_override, gamma_override
                )
                probabilities, gamma_eff, content, next_h, q_bytes, q_null, read = out
                byte = int(torch.argmax(probabilities))
                hits.append(byte == target[position])
                if getattr(ctx, "debug", False):
                    trace_rows.append(
                        {
                            "position": position,
                            "record": record,
                            "gold_offset": offset,
                            "cursor_in": list(cursor_trace[-1]) if cursor_trace else [-1, -1],
                            "m_peak": read.m_peak,
                            "m_peak_value": read.m_peak_value,
                            "pointer_argmax": (
                                int(torch.argmax(read.pointer[record]))
                                if len(read.pointer) > record
                                else -1
                            ),
                            "q_argmax_in_record": int(
                                torch.argmax(read.q_bytes[prepared.base.view.slice(record)])
                            ),
                            "alpha": float(read.alpha[record]),
                            "logit_gold": float(read.pointer_logits[record][offset]),
                            "logit_top": float(
                                read.pointer_logits[record][int(torch.argmax(read.pointer[record]))]
                            ),
                            "pointer_gold": float(read.pointer[record][offset]),
                            "target": int(target[position]),
                            "top": byte,
                            "hit": byte == target[position],
                        }
                    )
                accepted = target[position]
                cursor_trace.append((state.copy_record, state.copy_cursor))
                state = advance_custom(
                    ctx.candidate,
                    prepared,
                    state,
                    read,
                    probabilities,
                    gamma_eff,
                    accepted,
                    content=content,
                    next_h=next_h,
                    q_bytes=q_bytes,
                    q_null=q_null,
                )
                if position == 15:
                    state_mid = state
            else:
                step = ctx.natural_step(prepared, state, previous, _context, gate_index, True)
                byte = int(torch.argmax(step.probabilities))
                accepted = target[position]
                state, _coverage, gate_index = _advanced_state(
                    ctx.candidate, prepared, step, accepted
                )
                if position == 15:
                    state_mid = state
            raw.append(byte)
            previous = accepted
    return {
        "hits": hits,
        "raw": raw[:27],
        "cursor_trace": cursor_trace,
        "state_mid": state_mid,
        "trace_rows": trace_rows,
        "gamma_mean": float(gamma_eff.detach()),
    }


def rate_stats(hits: list[bool], positions) -> dict:
    subset = [hits[CONTENT.index(position)] for position in positions if position in CONTENT]
    return {
        "rate": sum(subset) / len(subset) if subset else 0.0,
        "first": subset[0] if subset else None,
        "last": subset[-1] if subset else None,
        "first_last_delta": (subset[0] - subset[-1]) if subset and len(subset) > 1 else 0.0,
    }


def aggregate(runs: list[dict]) -> dict:
    object_hits = [run["hits"][CONTENT.index(position)] for run in runs for position in OBJECT]
    place_hits = [run["hits"][CONTENT.index(position)] for run in runs for position in PLACE]
    per_example = []
    for run in runs:
        hits = run["hits"]
        per_example.append(
            {
                "content": sum(hits) / len(CONTENT),
                "object": sum(hits[CONTENT.index(p)] for p in OBJECT) / len(OBJECT),
                "place": sum(hits[CONTENT.index(p)] for p in PLACE) / len(PLACE),
                "object_first": hits[CONTENT.index(OBJECT[0])],
                "object_last": hits[CONTENT.index(OBJECT[-1])],
                "place_first": hits[CONTENT.index(PLACE[0])],
                "place_last": hits[CONTENT.index(PLACE[-1])],
            }
        )
    return {
        "examples": len(runs),
        "content": sum(object_hits + place_hits) / (len(object_hits) + len(place_hits)),
        "object": sum(object_hits) / len(object_hits),
        "place": sum(place_hits) / len(place_hits),
        "object_first": sum(row["object_first"] for row in per_example) / len(per_example),
        "object_last": sum(row["object_last"] for row in per_example) / len(per_example),
        "place_first": sum(row["place_first"] for row in per_example) / len(per_example),
        "place_last": sum(row["place_last"] for row in per_example) / len(per_example),
        "object_first_last_delta": (
            sum(row["object_first"] - row["object_last"] for row in per_example) / len(per_example)
        ),
        "place_first_last_delta": (
            sum(row["place_first"] - row["place_last"] for row in per_example) / len(per_example)
        ),
        "per_example": per_example,
    }


def free_decode(ctx, fixture, gamma_override, masked):
    prepared, context = ctx.prepare(fixture)
    state, gate_index = observe_goal(ctx.candidate, prepared, gate_index=0)
    raw: list[int] = []
    illegal = 0
    repaired = 0
    previous = EOS
    with torch.no_grad():
        for _ in range(28):
            step = ctx.natural_step(prepared, state, previous, context, gate_index, True)
            probabilities = step.probabilities
            if gamma_override is not None:
                gamma = torch.as_tensor(gamma_override, dtype=probabilities.dtype)
                probabilities = (gamma + (1.0 - gamma) * step.read.q_null) * step.generated + (
                    1.0 - gamma
                ) * step.read.copy_byte_mass
            if masked:
                endpoint = decode_endpoint(probabilities, prefix=bytes(raw))
            else:
                byte = int(torch.argmax(probabilities))
                allowed, _valid = utf8_allowed_mask(bytes(raw))
                endpoint = {
                    "byte": byte,
                    "eos": byte == EOS,
                    "masked": not bool(allowed[byte]),
                    "pre_mask_byte": byte,
                }
            if endpoint["masked"]:
                if masked:
                    repaired += 1
                else:
                    illegal += 1
            if endpoint["eos"]:
                break
            raw.append(endpoint["byte"])
            state, _coverage, gate_index = _advanced_state(
                ctx.candidate, prepared, step, endpoint["byte"]
            )
    text = bytes(raw)
    try:
        text.decode("utf-8")
        state_code = "complete"
    except UnicodeDecodeError:
        state_code = "illegal"
    valid, need, _lo, _hi = utf8_state(text)
    if not valid:
        state_code = "illegal"
    elif need:
        state_code = "truncated"
    return {
        "raw": raw,
        "utf8": state_code == "complete",
        "legal_prefix": valid,
        "state_code": state_code,
        "illegal_bytes": illegal,
        "repaired": repaired,
    }


def load_dev(limit: int) -> tuple[dict, list[dict]]:
    payload = json.loads(
        (ROOT / "output/taiji_data_premise_18_0/data.json").read_text(encoding="utf-8")
    )
    dev = premise.fixtures(payload, "dev")
    return payload, [dict(fixture) for fixture in dev[:limit]]


def update_rows(candidate, optimizer, fixtures, arm, steps, cmp17):
    rows = []
    for update in range(1, steps + 1):
        batch = fixtures[:8]
        stage = "copy"
        parameters = cmp17.set_trainable(candidate, stage)
        loss, stats = cmp17.batch_loss(candidate, batch, arm, stage, False, update, "18.2")
        optimizer.zero_grad()
        loss.backward()
        clip = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0))
        optimizer.step()
        rows.append(
            {
                "update": update,
                "loss": float(loss.detach()),
                "clip": clip,
                "ce": stats["ce"],
                "span": stats["span"],
                "nll": stats["nll"],
            }
        )
    return rows


def _named(candidate) -> dict[str, Any]:
    modules = (
        ("binding", candidate.binding),
        ("head", candidate.head),
        ("span", candidate.span_head),
    )
    out: dict[str, Any] = {}
    for prefix, module in modules:
        if module is None:
            continue
        for name, parameter in module.named_parameters():
            out[f"{prefix}.{name}"] = parameter
    return out


def _adam(candidate, optimizer) -> dict:
    names = {id(parameter): name for name, parameter in _named(candidate).items()}
    return {
        names.get(id(parameter), repr(parameter)): {
            key: value.detach().clone() if torch.is_tensor(value) else value
            for key, value in row.items()
        }
        for parameter, row in optimizer.state.items()
    }


def snapshot(candidate, optimizer, state) -> dict:
    return {
        "parameters": {
            name: parameter.detach().clone() for name, parameter in _named(candidate).items()
        },
        "adam": _adam(candidate, optimizer),
        "state": state,
    }


def restore_diff(path: Path, before: dict) -> dict:
    candidate, optimizer, state, _update, _payload = load_routing_bundle(path)
    param_diff = 0.0
    for name, parameter in _named(candidate).items():
        if name in before["parameters"]:
            param_diff = max(
                param_diff,
                float(
                    (parameter.detach() - before["parameters"][name].to(parameter.dtype))
                    .abs()
                    .max()
                ),
            )
    adam = _adam(candidate, optimizer)
    adam_diff = 0.0
    for name, row in adam.items():
        old = before["adam"].get(name, {})
        for key in ("exp_avg", "exp_avg_sq"):
            if key in row and key in old:
                adam_diff = max(
                    adam_diff, float((row[key] - old[key].to(row[key].dtype)).abs().max())
                )
            elif key in row:
                adam_diff = max(adam_diff, float(row[key].abs().max()))
    copy_diff = 0.0
    if state.m.numel() and before["state"].m.numel():
        copy_diff = max(copy_diff, float((state.m - before["state"].m).abs().max()))
        copy_diff = max(copy_diff, float((state.coverage - before["state"].coverage).abs().max()))
    cursor_equal = (
        state.prefix_cursor == before["state"].prefix_cursor
        and state.copy_record == before["state"].copy_record
        and state.copy_cursor == before["state"].copy_cursor
    )
    return {
        "parameters_max_abs_diff": param_diff,
        "adam_max_abs_diff": adam_diff,
        "copy_state_max_abs_diff": copy_diff,
        "cursor_fields": {
            "prefix_cursor": [before["state"].prefix_cursor, state.prefix_cursor],
            "copy_record": [before["state"].copy_record, state.copy_record],
            "copy_cursor": [before["state"].copy_cursor, state.copy_cursor],
        },
        "cursor_bitwise_equal": cursor_equal,
        "restored_update": _update,
    }


def run_cell(arm: str, updates: int, fixtures: list[dict], data: dict, debug: bool) -> dict:
    candidate, optimizer = build_routing_candidate(ROOT / PARENT, 28031, arm)
    ctx = CursorProbe(candidate)
    ctx.arm = arm
    ctx.debug = debug

    update_log: list[dict] = []
    if updates:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "cmp17r", ROOT / "scripts/training/compare_taiji_selection_routing_17_5.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        train = [dict(fixture) for fixture in premise.fixtures(data, "train")[:8]]
        update_log = update_rows(candidate, optimizer, train, arm, updates, module)

    b_pure = [content_run(ctx, fixture, "b", 0.0) for fixture in fixtures]
    c_gold = [content_run(ctx, fixture, "c", 0.0) for fixture in fixtures]
    b_stats = aggregate(b_pure)
    c_stats = aggregate(c_gold)

    free_g1 = [free_decode(ctx, fixture, 1.0, True) for fixture in fixtures]
    free_g0 = [free_decode(ctx, fixture, 0.0, True) for fixture in fixtures]
    free_unmasked_g1 = [free_decode(ctx, fixture, 1.0, False) for fixture in fixtures]
    free_unmasked_g0 = [free_decode(ctx, fixture, 0.0, False) for fixture in fixtures]

    def legal(runs):
        return sum(1 for run in runs if run["legal_prefix"]) / len(runs)

    def complete(runs):
        return sum(1 for run in runs if run["utf8"]) / len(runs)

    def codes(runs):
        out: dict[str, int] = {}
        for run in runs:
            out[run["state_code"]] = out.get(run["state_code"], 0) + 1
        return out

    mid_states = [run["state_mid"] for run in b_pure if run["state_mid"] is not None]
    restore = None
    if mid_states:
        snapshot_before = snapshot(candidate, optimizer, mid_states[0])
        path = OUTPUT_DIR / f"28031-{arm}-cursor-step{updates}.pt"
        save_routing_bundle(
            path,
            candidate,
            optimizer,
            updates,
            {"stage": "N18.2-probe", "seed": 28031, "arm": arm, "updates": updates},
            {"sha256": "probe", "path": str(PARENT)},
            mid_states[0],
        )
        restore = restore_diff(path, snapshot_before)

    utf8 = {
        "generate_only_masked": legal(free_g1),
        "copy_only_masked": legal(free_g0),
        "generate_only_masked_complete": complete(free_g1),
        "copy_only_masked_complete": complete(free_g0),
        "generate_only_unmasked": legal(free_unmasked_g1),
        "copy_only_unmasked": legal(free_unmasked_g0),
        "generate_only_illegal_bytes": sum(run["illegal_bytes"] for run in free_unmasked_g1),
        "copy_only_illegal_bytes": sum(run["illegal_bytes"] for run in free_unmasked_g0),
        "generate_only_repaired": sum(run["repaired"] for run in free_g1),
        "copy_only_repaired": sum(run["repaired"] for run in free_g0),
        "generate_only_codes": codes(free_g1),
        "copy_only_codes": codes(free_g0),
        "examples": len(fixtures),
    }
    gates = {
        "b_pure_content": b_stats["content"] >= GATES["b_pure_content"],
        "first_last_delta": max(
            abs(b_stats["object_first_last_delta"]), abs(b_stats["place_first_last_delta"])
        )
        <= GATES["first_last_delta"],
        "utf8_free": utf8["generate_only_masked"] >= GATES["utf8_free"]
        and utf8["copy_only_masked"] >= GATES["utf8_free"],
        "restore_bitwise": bool(
            restore
            and restore["cursor_bitwise_equal"]
            and restore["parameters_max_abs_diff"] == 0.0
            and restore["adam_max_abs_diff"] == 0.0
            and restore["copy_state_max_abs_diff"] == 0.0
        ),
    }
    return {
        "arm": arm,
        "updates": updates,
        "update_log": update_log,
        "b_pure": {key: value for key, value in b_stats.items() if key != "per_example"},
        "c_gold": {key: value for key, value in c_stats.items() if key != "per_example"},
        "cursor_trace_first": b_pure[0]["cursor_trace"][:4] if b_pure else [],
        "utf8": utf8,
        "restore": restore,
        "gates": gates,
        "all_gates": all(gates.values()),
        "_evidence": {
            "b_pure": [
                {key: value for key, value in run.items() if key != "state_mid"} for run in b_pure
            ],
            "c_gold": [
                {key: value for key, value in run.items() if key != "state_mid"} for run in c_gold
            ],
            "free_g1": free_g1,
            "free_g0": free_g0,
            "free_unmasked_g1": free_unmasked_g1,
            "free_unmasked_g0": free_unmasked_g0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--limit", type=int, default=LIMIT_EXAMPLES)
    parser.add_argument("--arms", default="A_route_lock,B_stop_lock")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    start_cpu, start_wall = time.process_time(), time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data, fixtures = load_dev(args.limit)
    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]

    cells: dict[str, dict] = {}
    for arm in arms:
        cell = run_cell(arm, args.updates, fixtures, data, args.debug)
        evidence = cell.pop("_evidence")
        cells[arm] = cell
        EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
        tag = arm.split("_")[0]
        (EVIDENCE.parent / f"taiji_cursor_utf8_18_2_probe_{tag}_20261104.json").write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    payload = {
        "schema": "taiji-cursor-utf8-18-2-probe-v1",
        "date": "2026-11-04",
        "supersedes": "reports/taiji_gate_sweep_18_1_20261104.json (readings retained; 18.1 files untouched)",
        "parent": str(PARENT),
        "updates": args.updates,
        "examples": len(fixtures),
        "structure": {
            "state_fields": ["prefix_cursor", "copy_record", "copy_cursor"],
            "carry_rule": "committed carry overrides the transport point; transport decides a run start",
            "frozen_parameter_count_unchanged": True,
            "decode_mask": "utf8_allowed_mask before argmax; prefix=None keeps 18.1 reading",
        },
        "cells": cells,
        "all_gates": all(cell["all_gates"] for cell in cells.values()),
        "cost": {
            "cpu_seconds": time.process_time() - start_cpu,
            "wall_seconds": time.perf_counter() - start_wall,
            "adam_updates": args.updates * len(arms),
        },
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
