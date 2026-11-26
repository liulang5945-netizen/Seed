"""10.21a-2 rev2: the channel moved with the table, but does it select the descriptor?

10.21a answered its own pre-registered question (criterion 4: the slots must differ
between the real and the permuted glossary -> 5/5 moved).  It also produced a reading
that was *not* pre-registered and points the other way: two different descriptors
under the same glossary sit 14x closer than one descriptor under two glossaries.
This probe explains that reading.  It cannot overturn 10.21a; it only says whether a
slot is query-conditioned or content-dominated, which is what the two-slot head needs.

rev2 disclosure: the first run scored descriptor separability with leave-one-out 1-NN.
That statistic is tautological here -- one sample per class with labels equal to the
class index means the nearest *other* point always carries a different label, so the
score can never exceed 0.  That card was voided (reports/..._10_21a2_20261101_void.json)
before any document referenced it.  rev2 replaces it with a two-way decomposition and
also records why a label-shuffle null cannot be used at all: the between-column sum of
squares is invariant under relabelling (checked numerically below, not asserted).

Design: five cyclic rotations of the five entity assignments give a Latin square, so
the descriptor factor and the glossary factor are orthogonal by construction.  For
each of the 5x5 cells one read is taken with that descriptor as the query against that
glossary.  With one observation per cell the only available null for the descriptor
factor is the interaction residual, so the reported p is the F(df1=4, df2=16) tail.

The slot is squeezed to one dimension here rather than in ``read_slot``: that helper is
imported by ``verify_taiji_10_21_1.py``, whose published card binds its bytes.

Readings (no training, no parameter changes, no capability claim):

1. determinism: the same read twice must be exactly equal;
2. variance shares: descriptor vs glossary, plus F and its tail;
3. degeneracy check: relabelling columns leaves the between-column SS unchanged;
4. nearest-neighbour identity: is slot(d, real) closer to slot(d, rotated) than to
   any slot(d', real)?  0/4 is a clean content-dominance statement;
5. attention displacement: does the descriptor move *where* the read looks?
6. a one-line glossary: the same decomposition when the table holds a single line.
   If the descriptor share rises there, the failure is averaging over the table and
   not a dead query path.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from scipy.stats import f as f_distribution  # noqa: E402

from taiji.candidate_manifest import build_manifest, load_candidate  # noqa: E402
from taiji.living_core import RawSource  # noqa: E402
from taiji.native_goal_attributes import OBJECT_DESCRIPTIONS, PLACE_DESCRIPTIONS  # noqa: E402
from taiji.slot_composition_head import read_slot, relative_distance  # noqa: E402
from taiji.state_migration import resume_subject  # noqa: E402

BUNDLE = "output/taiji_core_09/subject_bundle.pt"
BUDGET_CPU_SECONDS = 180
BUDGET_WALL_SECONDS = 400
DOMINANCE_RATIO = 2.0
ALPHA = 0.10

DECLARED = (
    "Headline status uses the full five-line table. shares_descriptor = SS_descriptor / SS_total, "
    "shares_glossary = SS_glossary / SS_total. slots_query_conditioned when "
    "shares_descriptor >= 2*shares_glossary and the F tail p <= 0.10; slots_content_dominated "
    "when shares_glossary >= 2*shares_descriptor; slots_mixed otherwise. The single-line "
    "condition is a diagnostic contrast and does not set the status."
)

VOID_CARD = "reports/taiji_core_slot_descriptor_10_21a2_20261101_void.json"


def records_for(items) -> tuple[RawSource, ...]:
    return tuple(
        RawSource(source_id=f"g{index}", raw=f"{text}是{entity}。".encode())
        for index, (text, entity) in enumerate(items)
    )


def descriptor_entity_pairs():
    return [
        *[(text, entity) for entity, text in OBJECT_DESCRIPTIONS.items()],
        *[(text, entity) for entity, text in PLACE_DESCRIPTIONS.items()],
    ]


def rotations(pairs) -> list[list[tuple[str, str]]]:
    entities = [entity for _, entity in pairs]
    size = len(pairs)
    return [
        [(text, entities[(index + shift) % size]) for index, (text, _) in enumerate(pairs)]
        for shift in range(size)
    ]


def decompose(cells: torch.Tensor) -> dict:
    """cells: (K, D, W) in float64. Two-way additive decomposition without replication."""
    rows, columns, _ = cells.shape
    grand = cells.mean(dim=(0, 1))
    row_means = cells.mean(dim=1)
    column_means = cells.mean(dim=0)
    total = float(((cells - grand) ** 2).sum())
    row_ss = float(columns * ((row_means - grand) ** 2).sum())
    column_ss = float(rows * ((column_means - grand) ** 2).sum())
    residual_ss = max(total - row_ss - column_ss, 0.0)
    df_columns, df_residual = columns - 1, (rows - 1) * (columns - 1)
    if residual_ss == 0.0:
        statistic = math.inf
        tail = 0.0
    else:
        statistic = (column_ss / df_columns) / (residual_ss / df_residual)
        tail = float(f_distribution.sf(statistic, df_columns, df_residual))
    return {
        "rows": rows,
        "columns": columns,
        "ss_total": total,
        "ss_descriptor": column_ss,
        "ss_glossary": row_ss,
        "ss_residual": residual_ss,
        "share_descriptor": column_ss / total if total else 0.0,
        "share_glossary": row_ss / total if total else 0.0,
        "share_residual": residual_ss / total if total else 0.0,
        "f_descriptor": statistic,
        "f_tail_p": tail,
        "df": [df_columns, df_residual],
    }


def label_shuffle_invariance(cells: torch.Tensor) -> float:
    """Between-column SS must not move when columns are relabelled: no usable null that way."""
    columns = cells.shape[1]
    grand = cells.mean(dim=(0, 1))
    baseline = float((cells.mean(dim=0) - grand).pow(2).sum())
    worst = 0.0
    generator = torch.Generator().manual_seed(0)
    for _ in range(5):
        order = torch.randperm(columns, generator=generator)
        shuffled = float((cells.mean(dim=0)[order] - grand).pow(2).sum())
        worst = max(worst, abs(shuffled - baseline) / baseline if baseline else 0.0)
    return worst


def attention_stats(attention: torch.Tensor) -> dict:
    weights = attention.detach().double()
    positive = weights[weights > 0]
    entropy = float(-(positive * positive.log()).sum())
    return {
        "positions": int(weights.numel()),
        "max_weight": float(weights.max()),
        "normalized_entropy": entropy / math.log(weights.numel()) if weights.numel() > 1 else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("10.21a-2 rev2 card is write-once")

    torch.set_num_threads(1)
    total_cpu, total_wall = time.process_time(), time.perf_counter()
    bundle = ROOT / BUNDLE
    manifest = build_manifest(ROOT, bundle)
    with tempfile.TemporaryDirectory() as directory:
        manifest_path = Path(directory) / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        _manifest, payload = load_candidate(ROOT, manifest_path, bundle, "frozen_eval")
    model, state, _optimizer = resume_subject(payload)

    pairs = descriptor_entity_pairs()
    descriptors = [text for text, _ in pairs]
    tables = rotations(pairs)

    full_cells, single_cells, attention_full = [], [], []
    determinism_gap = 0.0
    for table in tables:
        full_row, attention_row = [], []
        for descriptor in descriptors:
            slot, attention = read_slot(model, state, descriptor, records_for(table))
            again, _ = read_slot(model, state, descriptor, records_for(table))
            slot = slot.reshape(-1)
            determinism_gap = max(determinism_gap, float((slot - again.reshape(-1)).norm()))
            full_row.append(slot.double())
            attention_row.append(attention.reshape(-1))
        full_cells.append(torch.stack(full_row))
        attention_full.append(attention_row)
        single_cells.append(
            torch.stack(
                [
                    read_slot(
                        model,
                        state,
                        descriptor,
                        records_for([(descriptor, table[index][1])]),
                    )[0]
                    .reshape(-1)
                    .double()
                    for index, descriptor in enumerate(descriptors)
                ]
            )
        )

    full_cells = torch.stack(full_cells)
    single_cells = torch.stack(single_cells)
    full_stats = decompose(full_cells)
    single_stats = decompose(single_cells)
    shuffle_gap = label_shuffle_invariance(full_cells)

    real_slots = full_cells[0]
    real_permuted = full_cells[1]
    neighbour_identity = 0
    neighbour_rows = []
    for index in range(len(descriptors)):
        own = float((real_slots[index] - real_permuted[index]).norm())
        nearest_other, nearest_distance = None, math.inf
        for other in range(len(descriptors)):
            if other == index:
                continue
            distance = float((real_slots[index] - real_slots[other]).norm())
            if distance < nearest_distance:
                nearest_other, nearest_distance = other, distance
        neighbour_identity += int(nearest_distance > own)
        neighbour_rows.append(
            {
                "descriptor": descriptors[index],
                "own_rotated_distance": own,
                "nearest_other_descriptor": descriptors[nearest_other],
                "nearest_other_distance": nearest_distance,
                "closer_to_own_rotated": nearest_distance > own,
            }
        )

    attention_across_descriptor = [
        float((attention_full[0][i] - attention_full[0][j]).abs().sum())
        for i, j in itertools.combinations(range(len(descriptors)), 2)
    ]
    attention_across_glossary = [
        float((attention_full[0][i] - attention_full[k][i]).abs().sum())
        for i in range(len(descriptors))
        for k in range(1, len(tables))
    ]

    median_across_glossary_descriptor = _median(
        [
            relative_distance(full_cells[i][d], full_cells[k][d])
            for i in range(len(tables))
            for k in range(i + 1, len(tables))
            for d in range(len(descriptors))
        ]
    )

    if full_stats["share_glossary"] >= DOMINANCE_RATIO * full_stats["share_descriptor"]:
        status = "slots_content_dominated"
    elif (
        full_stats["share_descriptor"] >= DOMINANCE_RATIO * full_stats["share_glossary"]
        and full_stats["f_tail_p"] <= ALPHA
    ):
        status = "slots_query_conditioned"
    else:
        status = "slots_mixed"

    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    if cpu_seconds > BUDGET_CPU_SECONDS or wall_seconds > BUDGET_WALL_SECONDS:
        status = "budget_stop"

    report = {
        "schema": "taiji-slot-descriptor-10-21a2-v2",
        "step": "10.21a-2",
        "revision": 2,
        "date": "2026-11-01",
        "status": status,
        "candidate_id": manifest["candidate_id"],
        "declared_before_running": True,
        "criterion": DECLARED,
        "supersedes": VOID_CARD,
        "revision_reason": (
            "rev1 used leave-one-out 1-NN with one sample per class and labels equal to the "
            "class index; that statistic cannot exceed zero by construction. Voided before any "
            "document cited it; rev2 uses a two-way decomposition."
        ),
        "note": (
            "Diagnostic only. 10.21a's pre-registered channel criterion stays as the verdict for "
            "the channel gate; these readings explain its unregistered descriptor contrast and "
            "cannot overturn it."
        ),
        "determinism_max_gap": determinism_gap,
        "label_shuffle_max_relative_move": shuffle_gap,
        "label_shuffle_null_is_degenerate": shuffle_gap <= 1e-9,
        "descriptors": len(descriptors),
        "tables": len(tables),
        "full_table": full_stats,
        "single_line": single_stats,
        "neighbour_identity": neighbour_identity,
        "neighbour_rows": neighbour_rows,
        "median_relative_same_descriptor_across_tables": median_across_glossary_descriptor,
        "attention_l1_across_descriptor": _median(attention_across_descriptor),
        "attention_l1_across_glossary": _median(attention_across_glossary),
        "attention_rows": [attention_stats(attention) for attention in attention_full[0]],
        "model_involved": True,
        "parameters_changed": False,
        "capability_claimed": False,
        "training_admitted": False,
        "limits": (
            "Five descriptors, five rotations, one fixture. With one observation per cell the "
            "only null for the descriptor factor is the interaction residual, so the reported p "
            "assumes additivity; it is a mechanism reading, not a capability gate. The "
            "single-line condition is a diagnostic, not part of the 10.15 fixture."
        ),
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "source_hashes": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in (
                "taiji/slot_composition_head.py",
                "taiji/candidate_manifest.py",
                "taiji/native_goal_attributes.py",
                "scripts/training/probe_slot_descriptor_10_21a2.py",
                VOID_CARD,
            )
        },
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    print(
        json.dumps(
            {
                "status": status,
                "share_descriptor": round(full_stats["share_descriptor"], 6),
                "share_glossary": round(full_stats["share_glossary"], 6),
                "f_descriptor": round(full_stats["f_descriptor"], 3),
                "f_tail_p": round(full_stats["f_tail_p"], 6),
                "single_line_share_descriptor": round(single_stats["share_descriptor"], 6),
                "single_line_share_glossary": round(single_stats["share_glossary"], 6),
                "single_line_f_tail_p": round(single_stats["f_tail_p"], 6),
                "neighbour_identity": neighbour_identity,
                "determinism_max_gap": determinism_gap,
                "label_shuffle_gap": shuffle_gap,
                "cpu_seconds": round(cpu_seconds, 3),
            },
            ensure_ascii=True,
        )
    )
    return 0


def _median(values) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


if __name__ == "__main__":
    raise SystemExit(main())
