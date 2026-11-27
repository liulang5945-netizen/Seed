"""19.2l phase 2: order-only decoy activity gate (0 parameters, 0 Adam).

Owner ruling 2026-11-05: 19.2j closed the scoring-form road (object route
+0.1146 with 2000/2000 bootstrap rounds, but the swap hard argmax stayed 0-2/20
and the sentence was equivalent).  19.2f measured the gold-decoy margin at a
median of about 7.3 while the whole order swap moves the score by about 1.24,
so the template signal outruns the order signal roughly 6x.  The bottleneck is
therefore the negative-sample composition, not the scoring form, and the
scoring-side changes (19.2i/19.2g/19.2h) must not be repeated.

This script builds only the decoy and runs the mandatory pre-training gate:

* decoy = the gold record key split at its midpoint with the two halves swapped.
  UTF-8 byte multiset, byte length and byte inventory are identical, so only the
  order differs and a bag-of-bytes feature cannot separate the two;
* 64 train examples x 2 content starts = 128 readings of
  ``score_gold - score_decoy`` on the fresh 19.2j O head (seed 28031, from
  zero, shared frozen parent binding);
* gate (owner): activation rate >= 20% (``max(0, m - margin) > 0`` with
  m = 1.0) **and** margin median < 1.0.  Otherwise the verdict is
  ``decoy_not_hard``: no training runs and only the mechanism is reported.

The two scores come from one 2-key context.  ``score_rows`` scores each key row
independently, so this reading is identical to scoring the pair inside the real
8-record context (the other six rows do not enter gold/decoy).

No training, no Adam step, no bundle, no confirm read.

Run: ``python scripts/training/probe_taiji_package19_2l_order_decoy_activity.py``
"""

from __future__ import annotations

import hashlib
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

import compare_taiji_selection_routing_17_5 as compare_module  # noqa: E402
import probe_taiji_content_transport_18_1 as probe_mod  # noqa: E402
import train_taiji_package19_2j_from_scratch_selection_ab as j  # noqa: E402

from taiji import data_premise_18_0 as premise  # noqa: E402
from taiji.living_core import EOS  # noqa: E402
from taiji.selection_routing import (  # noqa: E402
    _advanced_state,
    _s_t,
    build_context,
    observe_goal,
    routing_step,
)
from taiji.selection_routing_checkpoint import load_routing_bundle  # noqa: E402

SEED = 28031
ARM = "O"
SAMPLE = 64
MARGIN_M = 1.0
ACTIVATION_GATE = 0.20
MEDIAN_GATE = 1.0
STARTS = (probe_mod.OBJECT_POSITIONS[0], probe_mod.PLACE_POSITIONS[0])
START_NAMES = ("object", "place")

REPORT = ROOT / "reports/taiji_package19_2l_order_decoy_activity_20261105.json"
MANIFEST = ROOT / "plans/manifests/taiji_package19_2l_order_decoy_activity_20261105.json"
PROGRESS = ROOT / "reports/taiji_package19_2l_progress_20261105.jsonl"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def order_decoy(text: str) -> dict:
    """Split the record key at its character midpoint and swap the two halves."""
    if len(text) < 2:
        return {"decoy": None, "degenerate": True, "reason": "description shorter than 2 chars"}
    half = len(text) // 2
    decoy = text[half:] + text[:half]
    source_bytes = sorted(text.encode("utf-8"))
    decoy_bytes = sorted(decoy.encode("utf-8"))
    if decoy == text:
        return {"decoy": decoy, "degenerate": True, "reason": "half swap is the identity"}
    if source_bytes != decoy_bytes:
        return {"decoy": decoy, "degenerate": True, "reason": "byte multiset changed"}
    return {"decoy": decoy, "degenerate": False, "reason": None}


def start_state(candidate, prepared, context, prefix, start: int):
    """State and previous byte at target position ``start`` (teacher-forced)."""
    state = prepared.state
    state, gate_index = observe_goal(candidate, prepared, gate_index=0)
    previous = EOS
    for position in range(start):
        with torch.no_grad():
            step = routing_step(
                candidate,
                prepared,
                state,
                previous,
                context,
                candidate.head,
                arm="B_stop_lock",
                gate_index=gate_index,
                span_head=candidate.span_head,
                pointer_temperature=1.0,
                pointer_hard=True,
            )
        accepted = prefix[position]
        state, _coverage, gate_index = _advanced_state(candidate, prepared, step, accepted)
        previous = accepted
    with torch.no_grad():
        s_t = _s_t(candidate, prepared, state, previous)
        previous_embedding = candidate.subject.encoder.embedding(
            torch.tensor(previous, device=candidate.subject.W.device)
        )
    return s_t, previous_embedding


def margin_at(candidate, prepared, context, prefix, start, goal, gold_text, decoy_text):
    s_t, previous_embedding = start_state(candidate, prepared, context, prefix, start)
    two = build_context(candidate.binding, goal, (gold_text, decoy_text))
    with torch.no_grad():
        scores = candidate.head.score_rows(two.goal_vec, two.key_vecs, s_t, previous_embedding)
    return float(scores[0] - scores[1]), float(scores[0]), float(scores[1])


def distribution(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    count = len(ordered)

    def rank(fraction: float) -> float:
        return ordered[min(count - 1, max(0, int(round(fraction * (count - 1)))))]

    return {
        "count": count,
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p10": rank(0.10),
        "p25": rank(0.25),
        "p75": rank(0.75),
        "p90": rank(0.90),
        "min": ordered[0],
        "max": ordered[-1],
    }


def even_sample(train: tuple[dict, ...], count: int) -> tuple[dict, ...]:
    """Even stride over the frozen train order, so the sample spans all worlds."""
    stride = max(1, len(train) // count)
    picked = train[::stride][:count]
    if len(picked) != count:
        raise RuntimeError(f"even sample produced {len(picked)} of {count} examples")
    return tuple(picked)


def main() -> int:
    start_cpu = time.process_time()
    start_wall = time.perf_counter()
    PROGRESS.unlink(missing_ok=True)

    payload = premise.build_payload()
    train = premise.fixtures(payload, "train")
    if len(train) < SAMPLE:
        raise RuntimeError("training split too small for the activity sample")
    sample = even_sample(train, SAMPLE)

    parent, _opt, _state, _index, _cursor = load_routing_bundle(j.PARENT)
    if any(torch.isnan(t).any().item() for t in parent.head.parameters()):
        raise RuntimeError("parent head carries NaN (D16-6)")
    candidate, _optimizer, _params = j.build_run(parent, SEED, ARM)
    identity = j.candidate_identity(candidate)

    j.progress_append(
        PROGRESS,
        {
            "event": "activity_start",
            "seed": SEED,
            "arm": ARM,
            "sample": SAMPLE,
            "starts": len(STARTS),
            "adam_updates": 0,
        },
    )

    margins: dict[str, list[float]] = {name: [] for name in START_NAMES}
    rows = []
    degenerate = 0
    started = time.process_time()
    for fixture in sample:
        prepared, context = compare_module.prepared_pair(candidate, fixture, "19.2l-gate", "gate")
        prefix = tuple(fixture["target_prefix"])
        goal = fixture["goal"].encode("utf-8")
        mapping = probe_mod.content_positions(fixture)
        for start, name in zip(STARTS, START_NAMES, strict=True):
            gold_index, _offset = mapping[start]
            gold_text = tuple(fixture["record_keys"])[gold_index]
            built = order_decoy(gold_text)
            if built["degenerate"]:
                degenerate += 1
                j.progress_append(
                    PROGRESS,
                    {
                        "event": "decoy_degenerate",
                        "world": fixture["world"],
                        "index": fixture["index"],
                        "start": name,
                        "gold": gold_text,
                        "reason": built["reason"],
                    },
                )
                continue
            margin, score_gold, score_decoy = margin_at(
                candidate,
                prepared,
                context,
                prefix,
                start,
                goal,
                gold_text,
                built["decoy"],
            )
            margins[name].append(margin)
            row = {
                "world": fixture["world"],
                "index": fixture["index"],
                "start": name,
                "gold": gold_text,
                "decoy": built["decoy"],
                "margin": margin,
                "score_gold": score_gold,
                "score_decoy": score_decoy,
                "activated": margin < MARGIN_M,
            }
            rows.append(row)
            j.progress_append(PROGRESS, {"event": "margin_reading", **row})

    all_margins = margins["object"] + margins["place"]
    stats = distribution(all_margins)
    activation = (
        sum(1 for value in all_margins if value < MARGIN_M) / len(all_margins)
        if all_margins
        else 0.0
    )
    gate_pass = (
        bool(all_margins) and activation >= ACTIVATION_GATE and stats["median"] < MEDIAN_GATE
    )
    verdict = {
        "code": "pass" if gate_pass else "decoy_not_hard",
        "activation_rate": activation,
        "activation_required": ACTIVATION_GATE,
        "median_margin": stats["median"] if all_margins else None,
        "median_required_below": MEDIAN_GATE,
        "training_allowed": gate_pass,
        "mechanism": (
            "activation is the share of starts with max(0, m - margin) > 0.  A low share "
            "means the decoy still scores far below gold, so the template signal wins before "
            "the order signal is ever compared; the margin term would stay identically zero "
            "and 30 updates would be wasted (the 19.2g lesson)."
        ),
    }

    cpu = time.process_time() - start_cpu
    wall = time.perf_counter() - start_wall
    report = {
        "schema": "taiji-package19-2l-order-decoy-activity-v1",
        "step": "N19.2l order-only decoy",
        "phase": "activity gate (owner item 2)",
        "date": "2026-11-05",
        "mode": "forward-only, 0 Adam steps, 0 parameters changed, 0 bundles, 0 confirm",
        "basis": {
            "owner_ruling": (
                "the bottleneck is negative-sample composition, not the scoring form; "
                "19.2i/19.2g/19.2h scoring-side changes are all rejected and must not be "
                "repeated"
            ),
            "19_2j": (
                "alignment block is alive from zero: object route +0.1146 with 2000/2000 "
                "bootstrap rounds, but swap hard argmax 0-2/20 and the sentence is equivalent"
            ),
            "19_2f": (
                "gold-decoy margin median about 7.3 while the whole order swap moves the "
                "score by a median about 1.24, so the template signal suppresses the order "
                "signal by roughly 6x and argmax cannot flip"
            ),
            "do_not_rewrite": True,
        },
        "decoy_rule": {
            "construction": (
                "gold record key split at its character midpoint with the two halves swapped "
                "(for example '细颈的瓶子' -> '的瓶子细颈')"
            ),
            "family": (
                "the owner's segment-swap family: the literal object-segment/place-segment "
                "swap is unavailable because one record carries exactly one description "
                "segment, so the implemented member of the family permutes the two half "
                "segments of that description"
            ),
            "invariants": {
                "byte_multiset": "identical",
                "byte_length": "identical",
                "byte_inventory": "identical",
                "word_order": "different",
            },
            "per_start": 1,
            "multiple_negatives": False,
            "degenerate_policy": "skipped and counted, never scored as a tie",
            "bag_feature_note": (
                "a feature that only knows which bytes are present cannot separate gold from "
                "this decoy; only order-sensitive features can"
            ),
        },
        "sample": {
            "split": "train",
            "sampling_rule": "even stride over the frozen train order, first 64 rows",
            "examples": len(sample),
            "starts_per_example": len(STARTS),
            "readings": len(all_margins),
            "degenerate_skipped": degenerate,
            "seed": SEED,
            "arm": ARM,
            "head": "19.2j O, drawn from zero; parent binding and span pointer inherited frozen",
            "parent_bundle": str(j.PARENT.relative_to(ROOT)),
            "parent_sha256": sha256_file(j.PARENT),
            "candidate_sha_v2": identity["candidate_sha_v2"],
        },
        "activity_gate": {
            "margin_m": MARGIN_M,
            "activation_rate": activation,
            "activation_count": sum(1 for value in all_margins if value < MARGIN_M),
            "activation_required": ACTIVATION_GATE,
            "median_required_below": MEDIAN_GATE,
            "distribution_all": stats,
            "distribution_object": distribution(margins["object"]),
            "distribution_place": distribution(margins["place"]),
            "pass": gate_pass,
        },
        "verdict": verdict,
        "interpretation": {
            "decoy_is_hard": (
                "all readings fall inside m = 1.0, so the margin term is active by the "
                "owner's definition"
            ),
            "near_zero_margin": (
                f"median {stats['median']:+.4f} and mean {stats['mean']:+.4f}: the pooled "
                "scoring surface barely separates gold from a pure byte permutation"
            ),
            "gradient_caveat": (
                "activation alone is not a usable-gradient proof.  If the gold and decoy key "
                "vectors are near-identical, the margin term's gradient cancels; phase 3 must "
                "read the first-batch gradient norm of that term before trusting 30 updates."
            ),
            "asymmetry": (
                f"object starts average {distribution(margins['object'])['mean']:+.4f} and "
                f"place starts average {distribution(margins['place'])['mean']:+.4f}, so the "
                "decoy sometimes wins outright"
            ),
        },
        "readings": rows,
        "cost": {
            "adam_updates": 0,
            "parameters_changed": 0,
            "bundles": 0,
            "confirm_reads": 0,
            "cpu_seconds": cpu,
            "wall_seconds": wall,
            "reading_cpu_seconds": time.process_time() - started,
        },
        "next": {
            "phase_3": "pending owner approval after this gate",
            "plan": (
                "1 seed 28031, O baseline, from zero, 30 of 300 selection updates; CE-only "
                "control arm vs order-decoy margin arm with one variable; diagnostic swap "
                "gate >= 8/20, new-description route_miss down >= 0.10 versus the CE arm, "
                "conditional_byte no regression > 0.05, old description no regression; "
                "<= 30 independent Adam per arm (60 total), <= 400 CPU"
            ),
        },
    }
    j.write_json_safe(REPORT, report)
    j.write_json_safe(
        MANIFEST,
        {
            "schema": "taiji-package19-2l-manifest-v1",
            "step": "N19.2l",
            "phase": "activity_gate",
            "date": "2026-11-05",
            "report": str(REPORT.relative_to(ROOT)),
            "verdict": verdict["code"],
            "activation_rate": activation,
            "median_margin": stats["median"] if all_margins else None,
            "readings": len(all_margins),
            "adam_updates": 0,
            "parameters_changed": 0,
            "confirm_reads": 0,
        },
    )
    print(
        json.dumps(
            {
                "verdict": verdict["code"],
                "activation": round(activation, 4),
                "median_margin": round(stats["median"], 4) if all_margins else None,
                "readings": len(all_margins),
                "degenerate": degenerate,
                "cpu": round(cpu, 1),
                "wall": round(wall, 1),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
