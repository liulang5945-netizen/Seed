"""19.2g: record-matching contrastive margin on the 19.5 B parent (loss only).

19.2f showed the ordered signal is already inside the encoder (swap score
max-abs diff median 1.19 > theta_score = 1e-3, 40/40), yet the hard argmax never
moves (0/40) and the top record probability sits at 0.98-0.99.  This package
tests the pre-registered hypothesis that the *margin*, not the encoding, is
swallowed.  At each content start the route cross-entropy is joined by a
same-world same-kind hard-negative margin

    max(0, m - (score_gold - score_decoy)),   m = 1.0

with decoy = the highest-scoring non-gold record of the same kind (objects
against objects, places against places).  Nothing else changes: same parent
(19.5 ``28031-B_stop_lock-step300``), same selection stage, same 37,120
parameter head and 258 parameter span head, +0 parameters, no graph change, no
new transport or pointer path.

When one example carries two content starts the *worse* margin enters the loss
(max over positions).  The mean-over-positions reading is logged on the same
batch as an aggregation ablation; it is read-only (no gradient), because a
trained mean variant would need a second 30-update trajectory and the package
cap is 30 independent Adam updates total.

Caps: one independent trajectory, 30 Adam updates, 0 confirm reads, one saved
bundle and only after the checkpoint preflight round-trips.

gamma is the generation weight and 1 - gamma the copy weight; nothing here
infers training credit from gamma (M3).  D16-6 (``28031-A_route_lock`` all-NaN)
is not touched.

Run: ``python scripts/training/train_taiji_package19_2g_record_margin.py``
"""

from __future__ import annotations

import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

torch.set_num_threads(1)

import attribute_taiji_package19_2f_new_description_wall as wall  # noqa: E402
import composite_start_instrument as instrument  # noqa: E402
import probe_taiji_content_transport_18_1 as probe_mod  # noqa: E402

from taiji import data_premise_18_0 as premise  # noqa: E402
from taiji.candidate_identity import candidate_identity  # noqa: E402
from taiji.living_core import EOS  # noqa: E402
from taiji.selection_routing import (  # noqa: E402
    _advanced_state,
    _s_t,
    observe_goal,
    resolve_prefix,
    route_labels,
    routing_step,
)
from taiji.selection_routing_checkpoint import (  # noqa: E402
    load_routing_bundle,
    make_routing_optimizer,
    parameter_tensors,
    save_routing_bundle,
)

COMPARE_MODULE = ROOT / "scripts/training/compare_taiji_selection_routing_17_5.py"
PARENT = ROOT / "output/taiji_package19_compare/28031-B_stop_lock-step300.pt"
SEED = 28031
ARM = "B_stop_lock"
MARGIN_VALUE = 1.0
UPDATES = 30
ADAM_CAP = 30
PROBE_AT = (0, 5, 10, 15, 20, 25, 30)
SLICE_PROBE_AT = (5, 10, 15, 20, 25)
CONTENT_STARTS = (3, 21)
PAIRS_PER_STRATUM = 20
ARGMAX_GATE = 18
SATURATION_TOP1 = 0.90

OUTPUT_DIR = ROOT / "output/taiji_package19_2g_record_margin"
FINAL_BUNDLE = OUTPUT_DIR / "28031-B_stop_lock-step30-margin.pt"
PREFLIGHT_BUNDLE = OUTPUT_DIR / "preflight-28031-B_stop_lock-step0-margin.pt"
REPORT = ROOT / "reports/taiji_package19_2g_record_margin_20261104.json"
MANIFEST = ROOT / "plans/manifests/taiji_package19_2g_record_margin_20261104.json"

# Budget discipline (owner, 19.2g): functional item list x 3.3 safety factor.
BUDGET_ITEMS = (
    ("30 Adam selection updates, batch 8, CE + span + margin", 135.0),
    ("two full 192-example passes (free decode + 40 swap pairs each)", 200.0),
    ("two start-score extractions on the 96 new-description examples", 50.0),
    ("five 48-example slice probes", 30.0),
    ("payload, health, checkpoint dry-run, report", 15.0),
)
SAFETY_FACTOR = 3.3

SIX_CELL_MEASURED = {
    "training_cpu": 29574.2,
    "step150_probe_cpu": 1513.7,
    "step300_probe_cpu": 2137.4,
    "formal_adam": 1800,
    "package_cells": 6,
}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def cmp():
    spec = importlib.util.spec_from_file_location("cmp175_19_2g", COMPARE_MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def schedule(train: tuple[dict, ...], updates: int) -> list[tuple[dict, ...]]:
    batches = []
    take = len(train) - len(train) % 8
    generator = torch.Generator().manual_seed(premise.ORDER_SEED)
    permutation = torch.randperm(take, generator=generator).tolist()
    cursor = 0
    epoch = 0
    while len(batches) < updates:
        if cursor + 8 > len(permutation):
            epoch += 1
            generator = torch.Generator().manual_seed(premise.ORDER_SEED + epoch)
            permutation = torch.randperm(take, generator=generator).tolist()
            cursor = 0
        batches.append(tuple(train[index] for index in permutation[cursor : cursor + 8]))
        cursor += 8
    return batches


def summary(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    count = len(ordered)

    def rank(fraction: float) -> float:
        index = min(count - 1, max(0, int(round(fraction * (count - 1)))))
        return ordered[index]

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


def start_targets(fixture: dict) -> dict[int, tuple[int, tuple[int, ...]]]:
    """Target position -> (gold record, same-kind non-gold records)."""
    records = fixture["raw_example"]["records"]
    object_gold = int(fixture["selection_label"])
    place_gold = wall.place_record(fixture)
    object_decoys = tuple(
        index
        for index, record in enumerate(records)
        if record["kind"] == "object" and index != object_gold
    )
    place_decoys = tuple(
        index
        for index, record in enumerate(records)
        if record["kind"] == "place" and index != place_gold
    )
    if not object_decoys or not place_decoys:
        raise RuntimeError("same-kind hard-negative decoy missing")
    return {3: (object_gold, object_decoys), 21: (place_gold, place_decoys)}


def selection_margin_loss(
    candidate,
    prepared,
    prefix,
    target_bytes,
    route_index,
    route_stage,
    context,
    head,
    targets,
    *,
    margin_value: float = MARGIN_VALUE,
) -> dict:
    """Selection CE (unchanged weighting) plus the two content-start margins."""
    state = prepared.state
    state, gate_index = observe_goal(candidate, prepared, gate_index=0)
    input_prefix = resolve_prefix(prefix, len(target_bytes) - 1)
    terms: dict[int, list[torch.Tensor]] = {0: [], 1: [], 2: []}
    goal_vec = context.goal_vec.detach()
    key_vecs = [key.detach() for key in context.key_vecs]
    margins: list[torch.Tensor] = []
    active_margins = 0
    previous = EOS
    for t, _label in enumerate(target_bytes):
        with torch.no_grad():
            step = routing_step(
                candidate,
                prepared,
                state,
                previous,
                context,
                head,
                arm=ARM,
                gate_index=gate_index,
            )
            s_t = _s_t(candidate, prepared, state, previous)
            previous_embedding = candidate.subject.encoder.embedding(
                torch.tensor(previous, device=candidate.subject.W.device)
            )
        scores = head.score_rows(goal_vec, key_vecs, s_t, previous_embedding)
        route_label = route_index[t]
        if route_label >= 0:
            terms[route_stage[t]].append(
                F.cross_entropy(scores.unsqueeze(0), torch.tensor([route_label]))
            )
        if t in targets:
            gold, decoys = targets[t]
            hard_negative = torch.stack([scores[index] for index in decoys]).max()
            term = torch.clamp(margin_value - (scores[gold] - hard_negative), min=0.0)
            margins.append(term)
            active_margins += int(float(term.detach()) > 0.0)
        if t == len(target_bytes) - 1:
            break
        accepted = input_prefix[t]
        with torch.no_grad():
            state, _, gate_index = _advanced_state(candidate, prepared, step, accepted)
            previous = accepted
    present = [torch.stack(values).mean() for values in terms.values() if values]
    ce = torch.stack(present).mean() if present else torch.zeros((), dtype=torch.float64)
    worst = torch.stack(margins).max()
    mean = torch.stack(margins).mean()
    return {
        "ce": ce,
        "worst": worst,
        "mean": mean,
        "active_margins": active_margins,
        "positions": len(margins),
    }


def batch_margin_loss(candidate, batch, tag: str) -> tuple[torch.Tensor, dict]:
    module = cmp()
    ce_values = []
    worst_values = []
    mean_values = []
    span_values = []
    active = 0
    positions = 0
    for fixture in batch:
        prepared, context = module.prepared_pair(candidate, fixture, tag, "train")
        span, _spans = module.span_term(prepared, fixture, candidate.span_head)
        span_values.append(span)
        routes = route_labels(fixture)
        output = selection_margin_loss(
            candidate,
            prepared,
            module.teacher_prefix(fixture),
            tuple(fixture["target_bytes"]),
            routes[0],
            routes[1],
            context,
            candidate.head,
            start_targets(fixture),
        )
        ce_values.append(output["ce"])
        worst_values.append(output["worst"])
        mean_values.append(output["mean"])
        active += output["active_margins"]
        positions += output["positions"]
    ce_total = torch.stack(ce_values).mean()
    worst_total = torch.stack(worst_values).mean()
    mean_total = torch.stack(mean_values).mean()
    span_total = torch.stack(span_values).mean()
    loss = ce_total + worst_total + span_total
    return loss, {
        "ce": float(ce_total.detach()),
        "margin_worst": float(worst_total.detach()),
        "margin_mean_ablation": float(mean_total.detach()),
        "span": float(span_total.detach()),
        "active_margins": active,
        "margin_positions": positions,
    }


def position_scores(candidate, prepared, context, prefix, positions) -> dict[int, torch.Tensor]:
    state = prepared.state
    state, gate_index = observe_goal(candidate, prepared, gate_index=0)
    previous = EOS
    captured: dict[int, torch.Tensor] = {}
    for t in range(max(positions) + 1):
        with torch.no_grad():
            step = routing_step(
                candidate,
                prepared,
                state,
                previous,
                context,
                candidate.head,
                arm=ARM,
                gate_index=gate_index,
                span_head=candidate.span_head,
                pointer_temperature=1.0,
                pointer_hard=True,
            )
        if t in positions:
            captured[t] = step.scores.detach().clone()
        accepted = prefix[t]
        state, _, gate_index = _advanced_state(candidate, prepared, step, accepted)
        previous = accepted
    return captured


def score_snapshot(candidate, fixtures: tuple[dict, ...]) -> dict:
    """Top-1 route probability and gold-versus-hard-negative margin at both starts."""
    module = cmp()
    top1_values: list[float] = []
    margin_values: list[float] = []
    by_start: dict[str, dict[str, list[float]]] = {
        "object": {"top1": [], "margin": []},
        "place": {"top1": [], "margin": []},
    }
    for fixture in fixtures:
        prepared, context = module.prepared_pair(candidate, fixture, "19.2g-score", "score")
        prefix = module.teacher_prefix(fixture)
        captured = position_scores(candidate, prepared, context, prefix, CONTENT_STARTS)
        targets = start_targets(fixture)
        for position, scores in captured.items():
            top1 = float(torch.softmax(scores, dim=0).max())
            gold, decoys = targets[position]
            margin = float(scores[gold] - torch.stack([scores[index] for index in decoys]).max())
            top1_values.append(top1)
            margin_values.append(margin)
            name = "object" if position == 3 else "place"
            by_start[name]["top1"].append(top1)
            by_start[name]["margin"].append(margin)
    return {
        "examples": len(fixtures),
        "points": len(top1_values),
        "top1_probability": summary(top1_values),
        "gold_decoy_margin": summary(margin_values),
        "by_start": {
            name: {
                "points": len(values["top1"]),
                "top1_probability": summary(values["top1"]),
                "gold_decoy_margin": summary(values["margin"]),
            }
            for name, values in by_start.items()
        },
        "share_top1_ge_0_99": (
            sum(1 for value in top1_values if value >= 0.99) / len(top1_values)
            if top1_values
            else 0.0
        ),
        "share_margin_ge_1": (
            sum(1 for value in margin_values if value >= MARGIN_VALUE) / len(margin_values)
            if margin_values
            else 0.0
        ),
    }


def margin_activity(candidate, fixtures: tuple[dict, ...], tag: str, stride: int = 1) -> dict:
    """Does the added term carry any gradient at all?  0 active means the clamp is shut."""
    module = cmp()
    margins: list[float] = []
    top1_values: list[float] = []
    for fixture in fixtures[::stride]:
        prepared, context = module.prepared_pair(candidate, fixture, tag, "activity")
        captured = position_scores(
            candidate, prepared, context, module.teacher_prefix(fixture), CONTENT_STARTS
        )
        targets = start_targets(fixture)
        for position, scores in captured.items():
            gold, decoys = targets[position]
            margins.append(
                float(scores[gold] - torch.stack([scores[index] for index in decoys]).max())
            )
            top1_values.append(float(torch.softmax(scores, dim=0).max()))
    active = sum(1 for value in margins if value < MARGIN_VALUE)
    return {
        "examples": len(fixtures[::stride]),
        "points": len(margins),
        "active_at_margin_1_0": active,
        "active_fraction": active / len(margins) if margins else 0.0,
        "gold_decoy_margin": summary(margins),
        "top1_probability": summary(top1_values),
        "gradient_carried": active > 0,
    }


def slice_probe(candidate, dev_slice: tuple[dict, ...]) -> dict:
    probe = probe_mod.ProbeContext(candidate)
    probe.arm = ARM
    metrics = [
        instrument.metrics(instrument.measure_fixture(probe, fixture)) for fixture in dev_slice
    ]
    result = instrument.summarize(metrics)
    keys = (
        "route_hit_rate",
        "composite_start_hit_rate",
        "decode_start_hit_rate",
        "conditional_byte",
        "unconditional_byte",
    )
    return {
        "examples": len(dev_slice),
        **{f"{world}_{key}": result[world][key] for world in ("object", "place") for key in keys},
        "gamma_mean": result["gamma_mean"],
    }


def full_snapshot(candidate, dev, easy_pairs, new_pairs, score_fixtures) -> dict:
    probe = probe_mod.ProbeContext(candidate)
    probe.arm = ARM
    rows = []
    for fixture in dev:
        decoded = wall.decode_fixture(probe, fixture)
        verdict = wall.classify(fixture, decoded)
        if verdict["bucket"] == "frame_byte_wrong":
            verdict["frame_subtype"] = wall.frame_split(fixture, decoded, verdict["position"])
        metrics = wall.example_metrics(fixture, decoded)
        metrics["bucket"] = verdict["bucket"]
        metrics["frame_subtype"] = verdict.get("frame_subtype")
        metrics["position"] = verdict["position"]
        rows.append(metrics)
    swap_easy = wall.swap_summary(
        [wall.swap_pair(candidate, fixture, ARM) for fixture in easy_pairs]
    )
    swap_new = wall.swap_summary([wall.swap_pair(candidate, fixture, ARM) for fixture in new_pairs])
    return {
        "full_192": wall.aggregate(rows, None),
        "old_description_96": wall.aggregate(rows, False),
        "new_description_96": wall.aggregate(rows, True),
        "swap_easy": swap_easy,
        "swap_new": swap_new,
        "scores_new96": score_snapshot(candidate, score_fixtures),
        "examples_detail": [
            {
                "world": row["world"],
                "index": row["index"],
                "new_description": row["new_description"],
                "completed": row["completed"],
                "bucket": row["bucket"],
                "position": row["position"],
                "frame_subtype": row["frame_subtype"],
                "object_route": row["object"]["route_ok"],
                "place_route": row["place"]["route_ok"],
                "object_start": row["object"]["start_ok"],
                "place_start": row["place"]["start_ok"],
            }
            for row in rows
        ],
    }


def parent_health(candidate) -> dict:
    tensors = parameter_tensors(candidate)
    nan = sum(int(torch.isnan(value).sum()) for value in tensors.values())
    return {
        "path": str(PARENT.relative_to(ROOT)),
        "parameter_tensors": len(tensors),
        "nan_parameters": nan,
        "all_finite": nan == 0,
    }


def checkpoint_preflight(candidate, optimizer, state_fixture) -> dict:
    """Save a bundle, load it back, compare every parameter, then remove it."""
    module = cmp()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prepared, _context = module.prepared_pair(candidate, state_fixture, "19.2g-dryrun", "save")
    identity = candidate_identity(candidate)
    before = parameter_tensors(candidate)
    info = save_routing_bundle(
        PREFLIGHT_BUNDLE,
        candidate,
        optimizer,
        0,
        {"stage": "N19.2g-preflight", "seed": SEED, "arm": ARM},
        identity,
        prepared.state,
    )
    loaded, _loaded_optimizer, _state, update_index, cursor = load_routing_bundle(PREFLIGHT_BUNDLE)
    after = parameter_tensors(loaded)
    worst = 0.0
    if set(before) != set(after):
        worst = float("inf")
    else:
        for name, value in before.items():
            worst = max(worst, float((value - after[name]).abs().max()))
    PREFLIGHT_BUNDLE.unlink(missing_ok=True)
    return {
        "dry_run_path": str(PREFLIGHT_BUNDLE.relative_to(ROOT)),
        "dry_run_bytes": info["bytes"],
        "dry_run_sha256": info["sha256"],
        "restored_update_index": update_index,
        "restored_cursor": dict(cursor),
        "max_abs_diff": worst,
        "round_trip_ok": worst <= 0.0,
    }


def six_cell_price() -> dict:
    """Formal six-cell extrapolation repriced with the 3.3 safety factor."""
    training = SIX_CELL_MEASURED["training_cpu"]
    probes = SIX_CELL_MEASURED["step150_probe_cpu"] + SIX_CELL_MEASURED["step300_probe_cpu"]
    overhead = 500.0
    ideal = training + probes + overhead
    staged_ideal = training * 100 / 300 + 6 * 2 * 110.0 + 200.0
    return {
        "margin_new_forward_cost": 0.0,
        "margin_extra_reads_per_example": 2,
        "measured_19_5_training_cpu": training,
        "measured_probe_cpu": probes,
        "package_overhead_cpu": overhead,
        "functional_ideal_cpu": ideal,
        "safety_factor": SAFETY_FACTOR,
        "repriced_cpu": ideal * SAFETY_FACTOR,
        "staged": {
            "stage": "selection only, 6 cells x updates 1-100",
            "functional_ideal_cpu": staged_ideal,
            "repriced_cpu": staged_ideal * SAFETY_FACTOR,
            "adam": 600,
        },
        "note": (
            "measured 19.5 training package is already a realized number, but the owner rule requires "
            "functional list x 3.3 before any release; the six-cell rerun stays blocked until owner rules"
        ),
    }


def budget_block() -> dict:
    ideal = sum(value for _name, value in BUDGET_ITEMS)
    return {
        "functional_items": [{"item": name, "cpu": value} for name, value in BUDGET_ITEMS],
        "functional_ideal_cpu": ideal,
        "safety_factor": SAFETY_FACTOR,
        "budget_cpu": ideal * SAFETY_FACTOR,
        "rule": "budget = functional item list x 3.3; 19.2f reported 100-150 and spent 500.5 is the source",
    }


def gates_view(before: dict, after: dict) -> dict:
    new_before = before["new_description_96"]
    new_after = after["new_description_96"]
    old_after = after["old_description_96"]
    swap_new = after["swap_new"]
    frames_after = (
        new_after["bucket_counts"]["frame_byte_wrong"]
        + new_after["bucket_counts"]["eos_not_stopped"]
    )
    frames_before = (
        new_before["bucket_counts"]["frame_byte_wrong"]
        + new_before["bucket_counts"]["eos_not_stopped"]
    )
    return {
        "new_description_route_miss": {
            "before": new_before["buckets_of_all"]["route_miss"],
            "after": new_after["buckets_of_all"]["route_miss"],
            "threshold": 0.05,
            "pass": new_after["buckets_of_all"]["route_miss"] <= 0.05,
        },
        "composite_start_hit_object": {
            "before": new_before["composite_start_hit_object"],
            "after": new_after["composite_start_hit_object"],
            "threshold": 0.70,
            "pass": new_after["composite_start_hit_object"] >= 0.70,
        },
        "composite_start_hit_place": {
            "before": new_before["composite_start_hit_place"],
            "after": new_after["composite_start_hit_place"],
            "threshold": 0.70,
            "pass": new_after["composite_start_hit_place"] >= 0.70,
        },
        "conditional_byte_object": {
            "before": new_before["conditional_byte_object"],
            "after": new_after["conditional_byte_object"],
            "threshold": 0.90,
            "pass": new_after["conditional_byte_object"] >= 0.90,
        },
        "conditional_byte_place": {
            "before": new_before["conditional_byte_place"],
            "after": new_after["conditional_byte_place"],
            "threshold": 0.90,
            "pass": new_after["conditional_byte_place"] >= 0.90,
        },
        "sentence_new_description_96": {
            "before": new_before["completed"],
            "after": new_after["completed"],
            "threshold": 0.80,
            "pass": new_after["completed"] >= 0.80,
        },
        "sentence_old_description_96": {
            "before": before["old_description_96"]["completed"],
            "after": old_after["completed"],
            "threshold": 0.9375,
            "pass": old_after["completed"] >= 0.9375,
        },
        "t2_hard_argmax_changed_new": {
            "before": before["swap_new"]["pairs_argmax_changed"],
            "after": swap_new["pairs_argmax_changed"],
            "of": swap_new["pairs"],
            "threshold": f">={ARGMAX_GATE}/20",
            "pass": swap_new["pairs_argmax_changed"] >= ARGMAX_GATE,
        },
        "frozen_path_eos_frame": {
            "before": frames_before,
            "after": frames_after,
            "threshold": "no regression",
            "pass": frames_after <= frames_before,
        },
    }


def branch_view(gates: dict, after: dict, saturated: bool) -> dict:
    all_pass = all(gate["pass"] for gate in gates.values())
    route_miss_dropped = (
        gates["new_description_route_miss"]["after"] < gates["new_description_route_miss"]["before"]
    )
    sentence_new_flat = not gates["sentence_new_description_96"]["pass"]
    if all_pass:
        branch = "margin_is_the_swallowed_cause"
    elif route_miss_dropped and sentence_new_flat:
        branch = "selection_is_not_the_only_bottleneck"
    elif saturated:
        branch = "scoring_head_saturation_is_structural"
    else:
        branch = "no_dose_extension"
    return {
        "branch": branch,
        "all_gates_pass": all_pass,
        "route_miss_dropped": route_miss_dropped,
        "new_sentence_flat": sentence_new_flat,
        "top1_still_saturated": saturated,
        "rule": (
            "saturation = median top-1 probability >= 0.90 and new-pair hard argmax changes < 18/20"
        ),
    }


def main() -> int:
    start_cpu = time.process_time()
    start_wall = time.perf_counter()
    module = cmp()
    payload = premise.build_payload()
    train = premise.fixtures(payload, "train")
    dev = premise.fixtures(payload, "dev")
    new_fixtures = tuple(fixture for fixture in dev if fixture["flags"]["new_description"])
    old_fixtures = tuple(fixture for fixture in dev if not fixture["flags"]["new_description"])
    if len(dev) != 192 or len(new_fixtures) != 96 or len(old_fixtures) != 96:
        raise RuntimeError("dev stratification drift")
    easy_pairs, new_pairs = wall.pair_fixtures(dev)
    dev_slice = probe_mod.dev_slice(dev)
    batches = schedule(train, UPDATES)

    candidate = load_routing_bundle(PARENT)[0]
    health = parent_health(candidate)
    if not health["all_finite"]:
        raise RuntimeError("parent carries NaN parameters (D16-6 discipline)")
    identity = candidate_identity(candidate)
    optimizer = make_routing_optimizer(candidate)
    preflight = checkpoint_preflight(candidate, optimizer, dev[0])
    if not preflight["round_trip_ok"]:
        raise RuntimeError("checkpoint dry-run did not round-trip")

    train_margin_before = margin_activity(candidate, train, "19.2g-activity", stride=10)
    before = full_snapshot(candidate, dev, easy_pairs, new_pairs, new_fixtures)
    probes = [{"update": 0, "slice": slice_probe(candidate, dev_slice)}]
    training_log = []
    total_adam = 0
    for update in range(1, UPDATES + 1):
        parameters = module.set_trainable(candidate, "selection")
        loss, stats = batch_margin_loss(candidate, batches[update - 1], f"19.2g-{update}")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        total_adam += 1
        if total_adam > ADAM_CAP:
            raise RuntimeError("adam cap exceeded")
        training_log.append({"update": update, **stats})
        if update in SLICE_PROBE_AT:
            probes.append({"update": update, "slice": slice_probe(candidate, dev_slice)})
        print(
            json.dumps(
                {
                    "update": update,
                    "ce": round(stats["ce"], 4),
                    "margin_worst": round(stats["margin_worst"], 4),
                    "margin_mean": round(stats["margin_mean_ablation"], 4),
                    "active": f"{stats['active_margins']}/{stats['margin_positions']}",
                }
            ),
            flush=True,
        )

    after = full_snapshot(candidate, dev, easy_pairs, new_pairs, new_fixtures)
    train_margin_after = margin_activity(candidate, train, "19.2g-activity-after", stride=10)
    fixed_state = module.prepared_pair(candidate, dev[0], "19.2g-save", "save")[0].state
    saved = save_routing_bundle(
        FINAL_BUNDLE,
        candidate,
        optimizer,
        UPDATES,
        {
            "stage": "N19.2g",
            "seed": SEED,
            "arm": ARM,
            "update": UPDATES,
            "loss": "record_margin_worst_of_two_starts",
            "margin": MARGIN_VALUE,
        },
        identity,
        fixed_state,
    )
    reloaded, _optimizer, _state, reload_index, reload_cursor = load_routing_bundle(FINAL_BUNDLE)
    reload_diff = 0.0
    restored = parameter_tensors(reloaded)
    trained = parameter_tensors(candidate)
    for name, value in trained.items():
        reload_diff = max(reload_diff, float((value - restored[name]).abs().max()))

    gates = gates_view(before, after)
    saturated = (
        after["scores_new96"]["top1_probability"]["median"] >= SATURATION_TOP1
        and after["swap_new"]["pairs_argmax_changed"] < ARGMAX_GATE
    )
    decision = branch_view(gates, after, saturated)
    price = six_cell_price()
    budget = budget_block()
    cpu = time.process_time() - start_cpu
    wall_time = time.perf_counter() - start_wall

    aggregation_gap = {
        "updates_where_mean_below_worst": sum(
            1 for row in training_log if row["margin_mean_ablation"] < row["margin_worst"]
        ),
        "final_worst": training_log[-1]["margin_worst"],
        "final_mean_ablation": training_log[-1]["margin_mean_ablation"],
        "reading": (
            "mean-over-positions is logged on the same batch as an ablation but never backpropagated: "
            "a trained mean variant needs a second 30-update trajectory, over the 30 Adam cap"
        ),
        "why_worst": (
            "one example carries two content starts; the max keeps both routes under pressure and "
            "prevents a saturated easy start from cancelling the hard one, which the 19.2f per-world "
            "asymmetry (object route 0.469 versus place 0.844) shows is exactly the failure shape"
        ),
    }

    report = {
        "schema": "taiji-package19-2g-record-margin-v1",
        "step": "N19.2g record-matching contrastive margin (loss only, +0 parameters)",
        "date": "2026-11-04",
        "purpose": (
            "test whether the saturated record margin, not the ordered encoding, swallows the "
            "swap signal found in 19.2f"
        ),
        "design": {
            "parent": str(PARENT.relative_to(ROOT)),
            "parent_arm": ARM,
            "parent_update": 300,
            "parent_old96_19_5": before["old_description_96"]["completed"],
            "parent_new96_19_5": before["new_description_96"]["completed"],
            "margin": "max(0, m - (score_gold - score_decoy)), m = 1.0",
            "decoy": "hardest same-world same-kind non-gold record (3 candidates per start)",
            "aggregation": "max over the two content starts (worst case)",
            "graph_change": "none",
            "new_parameters": 0,
            "span_head": "inherited 258-parameter head, still trained by the selection stage as in 19.2c",
            "stage": "selection only",
            "updates": UPDATES,
            "adam_cap": ADAM_CAP,
            "batch": 8,
            "order_seed": premise.ORDER_SEED,
            "probes_at": list(PROBE_AT),
            "slice_probes_at": list(SLICE_PROBE_AT),
            "formal_eval": "192 examples, exact 27-byte action + EOS, free decode",
            "saves": 1,
            "confirm_reads": 0,
        },
        "parent_health": health,
        "checkpoint_preflight": preflight,
        "candidate_sha_v2": identity["candidate_sha_v2"],
        "parameter_counts": {
            "head": sum(parameter.numel() for parameter in candidate.head.parameters()),
            "span": sum(parameter.numel() for parameter in candidate.span_head.parameters()),
        },
        "before": before,
        "after": after,
        "slice_probes": probes,
        "training_log": training_log,
        "train_margin_activity": {
            "before": train_margin_before,
            "after": train_margin_after,
            "meaning": (
                "0 active points means max(0, m - (score_gold - score_decoy)) is identically zero "
                "on the training split, so the added term carries no gradient there"
            ),
        },
        "aggregation_ablation": aggregation_gap,
        "gates": gates,
        "decision": decision,
        "six_cell_repricing": price,
        "budget": {**budget, "spent_cpu": cpu, "over_budget": cpu > budget["budget_cpu"]},
        "saved_bundle": {
            **saved,
            "reload_max_abs_diff": reload_diff,
            "reload_update_index": reload_index,
        },
        "attribution_rules": [
            "gamma is the generation weight and 1 - gamma the copy weight",
            "gamma never implies training credit (M3)",
            "the margin term is a training objective only; the graph is untouched",
        ],
        "cost": {
            "cpu_seconds": cpu,
            "wall_seconds": wall_time,
            "adam_updates": total_adam,
            "independent_trajectories": 1,
            "bundles_saved": 1,
            "confirm_reads": 0,
        },
    }
    manifest = {
        "schema": "taiji-package19-2g-record-margin-manifest-v1",
        "step": "N19.2g",
        "date": "2026-11-04",
        "report": str(REPORT.relative_to(ROOT)),
        "parent": str(PARENT.relative_to(ROOT)),
        "parent_health": health,
        "checkpoint_preflight_ok": preflight["round_trip_ok"],
        "candidate_sha_v2": identity["candidate_sha_v2"],
        "adam_updates": total_adam,
        "bundles_saved": 1,
        "saved_bundle": saved["path"],
        "confirm_reads": 0,
        "gates": gates,
        "branch": decision["branch"],
        "top1_before_median": before["scores_new96"]["top1_probability"]["median"],
        "top1_after_median": after["scores_new96"]["top1_probability"]["median"],
        "route_miss_before_count": before["new_description_96"]["bucket_counts"]["route_miss"],
        "route_miss_after_count": after["new_description_96"]["bucket_counts"]["route_miss"],
        "swap_argmax_before": before["swap_new"]["pairs_argmax_changed"],
        "swap_argmax_after": after["swap_new"]["pairs_argmax_changed"],
        "train_margin_active_before": train_margin_before["active_at_margin_1_0"],
        "train_margin_active_points_before": train_margin_before["points"],
        "train_margin_active_after": train_margin_after["active_at_margin_1_0"],
        "six_cell_repriced_cpu": price["repriced_cpu"],
        "cost_cpu": cpu,
        "budget_cpu": budget["budget_cpu"],
        "over_budget": cpu > budget["budget_cpu"],
    }
    write_json(REPORT, report)
    write_json(MANIFEST, manifest)
    print(
        json.dumps(
            {
                "report": str(REPORT),
                "branch": decision["branch"],
                "cpu_seconds": round(cpu, 1),
                "route_miss_before": manifest["route_miss_before_count"],
                "route_miss_after": manifest["route_miss_after_count"],
                "top1_before": round(manifest["top1_before_median"], 4),
                "top1_after": round(manifest["top1_after_median"], 4),
                "train_margin_active": (
                    f"{manifest['train_margin_active_before']}/"
                    f"{manifest['train_margin_active_points_before']}"
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
