"""19.2l-B: can 30 order-decoy margin updates move the readout at all?

Owner ruling 2026-05 (19.2l phase B): the 19.2l gate showed the order-only decoy
is hard (activation 128/128).  The A structural reading then showed the margin
term is *not* structurally zero on the N head: gradient norm 1.6015 with the
added block contributing 0.0089, against 1.5965 for the O head.  So the decoy
is not the problem and the gradient does not cancel at the source.  What is left
is the question of whether the term, at this dose, can move the readout:

* experiment arm = N structure x (selection CE + order-decoy margin, m = 1.0)
* control arm    = N structure x selection CE only

Both arms are built with ``j.build_run(parent, 28031, "N")`` so they share the
exact same initialisation (verified bit for bit before any update) and the same
frozen binding.  30 updates, batch 8, LR 1e-3, clip 1.0, ``freeze_to_selection``,
data order from ``h.schedule(train, 30)``, so every batch is identical across
arms.

Readout: 192 dev examples (12 worlds x 16 indices; 0-7 old description, 8-15 new
description) through the 2h ``full_snapshot``: new-description route_miss,
route_hit (object/place), conditional_byte (object/place), old-description pass,
and the 20 new-description swap pairs' hard argmax change count.  The step0
snapshot is measured once and shared by both arms (``shared_init_snapshot``).

The owner's three new readings are implemented:

1. ``key_diff_after`` - max|key_gold - key_decoy| over the same 128 starts the A
   reading used (64 train examples x 2 starts, same even stride), after training.
   The binding is frozen by ``freeze_to_selection``, so this is a measurement of
   the pipeline, not an assumption.
2. ``score_margin_after`` / ``decoy_margin_after`` - full distributions of
   ``score_gold - score_decoy`` after training, on the 128 starts and on the
   first training batch's decoy rows.
3. ``gradient_share_by_param`` - the margin term's *own* gradient norm per
   parameter name, accumulated over all 30 treatment updates.  One extra
   ``torch.autograd.grad`` of the margin term per update (the extra backward the
   owner allows); sampling rate 100%.

The attribution verdict is one of three mutually exclusive codes, decided by a
rule fixed in this file before the run (priority: upstream washout, then
not-trainable-in-30, then at-readout washout).  Each outcome carries its own
"what may be done next and nothing else" sentence.

The gates below are *diagnostic* gates, not task gates: they only say whether
the margin term reached the readout, never that a capability shipped.

Pricing discipline: one forward+backward and a 24-example evaluation slice are
measured with 0 Adam before training; the extrapolation is written into the
report.  If the estimate exceeds 800 CPU the run stops and writes
``pricing_gate: over_budget`` instead of training.  The same two units are
measured again after the last update (same process, batch 8, same method) so the
report can say whether the opening price was contention or a real price regime.

``machine_hygiene`` is the owner's pre-run measurement, written in verbatim; it
is not reconstructed by this script.

Run: ``python scripts/training/train_taiji_package19_2l_order_decoy_diagnostic.py``
"""

from __future__ import annotations

import hashlib
import json
import math
import random
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
import probe_taiji_content_transport_18_1 as probe_mod  # noqa: E402
import probe_taiji_package19_2l_order_decoy_activity as gate  # noqa: E402
import train_taiji_package19_2h_sequence_surface as h  # noqa: E402
import train_taiji_package19_2j_from_scratch_selection_ab as j  # noqa: E402

from taiji import data_premise_18_0 as premise  # noqa: E402
from taiji.candidate_identity import candidate_identity  # noqa: E402
from taiji.living_core import EOS  # noqa: E402
from taiji.selection_routing import (  # noqa: E402
    _advanced_state,
    _s_t,
    observe_goal,
    routing_step,
)
from taiji.selection_routing_checkpoint import (  # noqa: E402
    load_routing_bundle,
    parameter_tensors,
    save_routing_bundle,
)

SEED = 28031
ARM = "N"
ROUTING_ARM = "B_stop_lock"
UPDATES = 30
BATCH = 8
CLIP = 1.0
MARGIN_M = 1.0
SAMPLE = 64
STARTS = (probe_mod.OBJECT_POSITIONS[0], probe_mod.PLACE_POSITIONS[0])
START_NAMES = ("object", "place")
MARGIN_TAG = "19.2l-B-margin"

# Owner's fixed attribution thresholds (written here before the run).
BASELINE_KEY_DIFF_MEDIAN = 1.491174147595295
KEY_DROP_THRESHOLD = 0.20
MARGIN_MEDIAN_FLOOR = -0.003
MARGIN_IMPROVE_FLOOR = 0.01
MARGIN_SIGNIFICANT = 0.05

# Diagnostic gates (not task gates).
ARGMAX_GATE = 8
ROUTE_MISS_DELTA_GATE = 0.10
CONDITIONAL_SLACK = 0.05

BOOTSTRAP_ROUNDS = 2000
BOOTSTRAP_SEED = 20261105
CPU_BUDGET = 800.0
ADAM_CAP = 60
PRICING_SAFETY = 1.0
SAVE_OVERHEAD_CPU = 8.0
FIRST_BATCH_AFTER_CPU = 2.5
REPORT_OVERHEAD_CPU = 3.0
CE_BEFORE_REFERENCE = 6.02
MARGIN_BEFORE_REFERENCE = 2.17
CONTENTION_RATIO = 0.7

MACHINE_HYGIENE = {
    "python_process_count_before": 0,
    "black_python_processes": 0,
    "logical_cores": 24,
    "torch_num_threads": 12,
    "experiment_pins_torch_threads": 1,
    "top3_by_cumulative_cpu": [
        {"name": "Trae CN", "pid": 44532, "total_cpu_s": 43734.4, "live": "0.6%"},
        {"name": "Trae CN", "pid": 35156, "total_cpu_s": 19830.0, "live": "0.9%"},
        {"name": "ProcessLasso", "pid": 25848, "total_cpu_s": 17273.1, "live": "0.1%"},
    ],
    "cleanup": {
        "process": "in-process black wrapper (black.main)",
        "pid": 40008,
        "cumulative_cpu_s": 641.8,
        "action": "killed by owner before the run",
        "recheck_after_cleanup": {"python_process_count": 0, "black_python_processes": 0},
    },
    "operational_waste_cpu_s": 641.8,
    "source": "owner-measured before the 800-budget run; not reconstructed",
}

OUTPUT_DIR = ROOT / "output/taiji_package19_2l_order_decoy_diagnostic"
FINAL_BUNDLE = OUTPUT_DIR / "28031-N-alignment-step30-order-decoy.pt"
REPORT = ROOT / "reports/taiji_package19_2l_order_decoy_diagnostic_20261105.json"
MANIFEST = ROOT / "plans/manifests/taiji_package19_2l_order_decoy_diagnostic_20261105.json"
PROGRESS = ROOT / "reports/taiji_package19_2l_order_decoy_diagnostic_progress_20261105.jsonl"

NEXT_STEPS = {
    "washout_upstream": (
        "只许做：修上游 key 传输，让一条纯字节置换在 key 空间里产生可区分的分离"
        "（binding 位置敏感传输方案）；禁止继续在 scorer 上改损失、加训练步数或读换序门。"
    ),
    "margin_not_trainable_in_30": (
        "只许做：按本报告外推的每步改善速率重设边距项剂量（更硬的负样本或更高权重/更长步数），"
        "并预先登记所需步数；禁止再以换序 argmax 作为证据、禁止改 readout 结构。"
    ),
    "washout_at_readout_proven": (
        "只许做：在 readout 层做顺序敏感的重排归一化（row 间成对比较或只保留 top-1 之外的抑制），"
        "让已经拉开的 gold-decoy 边距穿过 argmax；禁止再加工训练步数、禁止再动上游 key。"
    ),
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_at(candidate, prepared, context, prefix, start: int):
    """Teacher-forced state at ``start``; routing is detached, as in training."""
    with torch.no_grad():
        state, gate_index = observe_goal(candidate, prepared, gate_index=0)
        previous = EOS
        for position in range(start):
            step = routing_step(
                candidate,
                prepared,
                state,
                previous,
                context,
                candidate.head,
                arm=ROUTING_ARM,
                gate_index=gate_index,
                span_head=candidate.span_head,
                pointer_temperature=1.0,
                pointer_hard=True,
            )
            accepted = prefix[position]
            state, _coverage, gate_index = _advanced_state(candidate, prepared, step, accepted)
            previous = accepted
        s_t = _s_t(candidate, prepared, state, previous)
        previous_embedding = candidate.subject.encoder.embedding(
            torch.tensor(previous, device=candidate.subject.W.device)
        )
    return s_t.detach(), previous_embedding.detach()


def decoy_key(candidate, text: str) -> torch.Tensor:
    relation = candidate.binding.base.relation
    with torch.no_grad():
        units = relation.encode_units(candidate.binding.subject, text.encode("utf-8"))
        return relation.pool(units).detach()


def content_rows(candidate, examples, tag: str = MARGIN_TAG):
    """One row per (example, content start) with the order-decoy key row."""
    module = j.cmp()
    rows = []
    degenerate = 0
    for fixture in examples:
        prepared, context = module.prepared_pair(candidate, fixture, tag, "margin")
        prefix = tuple(fixture["target_prefix"])
        mapping = probe_mod.content_positions(fixture)
        for start, name in zip(STARTS, START_NAMES, strict=True):
            gold_index, _offset = mapping[start]
            gold_text = tuple(fixture["record_keys"])[gold_index]
            built = gate.order_decoy(gold_text)
            if built["degenerate"]:
                degenerate += 1
                continue
            s_t, previous_embedding = state_at(candidate, prepared, context, prefix, start)
            gold_key = context.key_vecs[gold_index].detach()
            decoy = decoy_key(candidate, built["decoy"])
            rows.append(
                {
                    "world": fixture["world"],
                    "index": fixture["index"],
                    "start": name,
                    "gold": gold_text,
                    "decoy": built["decoy"],
                    "goal_vec": context.goal_vec.detach(),
                    "gold_key": gold_key,
                    "decoy_key": decoy,
                    "s_t": s_t,
                    "previous_embedding": previous_embedding,
                    "key_max_abs_delta": float((gold_key - decoy).abs().max()),
                }
            )
    return rows, degenerate


def margin_average(candidate, rows) -> tuple[torch.Tensor, list[float]]:
    losses = []
    margins = []
    for row in rows:
        scores = candidate.head.score_rows(
            row["goal_vec"],
            [row["gold_key"], row["decoy_key"]],
            row["s_t"],
            row["previous_embedding"],
        )
        margin = scores[0] - scores[1]
        margins.append(float(margin.detach()))
        losses.append(F.relu(MARGIN_M - margin))
    return torch.stack(losses).mean(), margins


def margin_values(candidate, rows) -> list[float]:
    _loss, margins = margin_average(candidate, rows)
    return margins


def margin_summary(values: list[float]) -> dict:
    return {
        "distribution": gate.distribution(values),
        "activated_rate": (
            sum(1 for value in values if value < MARGIN_M) / len(values) if values else 0.0
        ),
        "activated_count": sum(1 for value in values if value < MARGIN_M),
    }


def margin_reading(candidate, rows) -> dict:
    return margin_summary(margin_values(candidate, rows))


def arm_view(snapshot: dict, shared: bool) -> dict:
    new = snapshot["new_description_96"]
    old = snapshot["old_description_96"]
    swap = snapshot["swap_new"]
    view = {
        "new_route_miss": new["buckets_of_all"]["route_miss"],
        "new_route_hit_object": new["route_hit_object"],
        "new_route_hit_place": new["route_hit_place"],
        "new_conditional_byte_object": new["conditional_byte_object"],
        "new_conditional_byte_place": new["conditional_byte_place"],
        "new_route_hit_both": new["route_hit_both"],
        "new_completed": new["completed"],
        "old_pass": old["completed"],
        "old_route_miss": old["buckets_of_all"]["route_miss"],
        "swap_argmax_changed": swap["pairs_argmax_changed"],
        "swap_pairs": swap["pairs"],
        "swap_decode_route_argmax_changed": swap["pairs_decode_route_argmax_changed"],
        "swap_median_score": swap["median_score"],
        "new_failures": new["failures"],
    }
    if shared:
        view["shared_init_snapshot"] = True
    return view


def metric_value(row: dict, key: str) -> float:
    if key == "route_miss":
        return 1.0 if row["bucket"] == "route_miss" else 0.0
    if key == "route_hit_object":
        return 1.0 if row["object"]["route_ok"] else 0.0
    if key == "route_hit_place":
        return 1.0 if row["place"]["route_ok"] else 0.0
    if key == "completed":
        return 1.0 if row["completed"] else 0.0
    raise ValueError(key)


def cluster_bootstrap(treatment_rows, control_rows, key: str, rounds: int, seed: int) -> dict:
    """World-clustered paired bootstrap: resample the 12 worlds, not the examples."""
    treat = {(row["world"], row["index"]): metric_value(row, key) for row in treatment_rows}
    control = {(row["world"], row["index"]): metric_value(row, key) for row in control_rows}
    if set(treat) != set(control):
        raise RuntimeError("paired bootstrap needs identical example sets")
    groups: dict[int, list[tuple[float, float]]] = {}
    for (world, _index), value in treat.items():
        groups.setdefault(world, []).append((value, control[(world, _index)]))
    worlds = sorted(groups)
    if not worlds:
        return {"metric": key, "count": 0}
    point = statistics.fmean(
        value for pairs in groups.values() for value, _other in pairs
    ) - statistics.fmean(other for pairs in groups.values() for _value, other in pairs)
    per_world = {
        world: (
            statistics.fmean(value for value, _other in groups[world])
            - statistics.fmean(other for _value, other in groups[world])
        )
        for world in worlds
    }
    generator = random.Random(seed)
    deltas = []
    for _round in range(rounds):
        picks = [generator.choice(worlds) for _index in range(len(worlds))]
        pooled = [pair for world in picks for pair in groups[world]]
        left = statistics.fmean(value for value, _other in pooled)
        right = statistics.fmean(other for _value, other in pooled)
        deltas.append(left - right)
    deltas.sort()
    alpha = 0.025
    low = deltas[max(0, int(round(alpha * (rounds - 1))))]
    high = deltas[min(rounds - 1, int(round((1 - alpha) * (rounds - 1))))]
    return {
        "metric": key,
        "count": sum(len(pairs) for pairs in groups.values()),
        "treatment_rate": statistics.fmean(
            value for pairs in groups.values() for value, _other in pairs
        ),
        "control_rate": statistics.fmean(
            other for pairs in groups.values() for _value, other in pairs
        ),
        "delta": point,
        "ci_low": low,
        "ci_high": high,
        "excludes_zero": low > 0.0 or high < 0.0,
        "n_wins": sum(1 for value in deltas if value > 0.0),
        "rounds": rounds,
        "n_worlds_positive": sum(1 for value in per_world.values() if value > 0.0),
        "worlds": len(worlds),
        "per_world": per_world,
        "cluster_unit": "world (12 clusters, resampled with replacement)",
    }


def paired_bootstrap(before: list[float], after: list[float], rounds: int, seed: int) -> dict:
    if len(before) != len(after) or not before:
        return {"count": 0}
    deltas = [after[index] - before[index] for index in range(len(before))]
    generator = random.Random(seed)
    draws = []
    for _round in range(rounds):
        picks = [generator.randrange(len(deltas)) for _index in range(len(deltas))]
        draws.append(statistics.fmean(deltas[index] for index in picks))
    draws.sort()
    alpha = 0.025
    low_index = max(0, int(round(alpha * (rounds - 1))))
    high_index = min(rounds - 1, int(round((1 - alpha) * (rounds - 1))))
    return {
        "count": len(deltas),
        "mean_delta": statistics.fmean(deltas),
        "median_delta": statistics.median(deltas),
        "ci_low": draws[low_index],
        "ci_high": draws[high_index],
        "rounds": rounds,
        "excludes_zero": draws[low_index] > 0.0 or draws[high_index] < 0.0,
        "pairing": (
            "paired by (example, start); each side uses its own arm state path, so the delta "
            "includes the routing-state change"
        ),
    }


def gradient_share(accumulated: dict[str, float]) -> dict:
    total = sum(accumulated.values())
    share = {name: (value / total if total > 0.0 else 0.0) for name, value in accumulated.items()}
    added = sum(value for name, value in accumulated.items() if j._is_added(name))
    return {
        "updates": UPDATES,
        "sampling_rate": "every update (100%)",
        "method": (
            "one extra torch.autograd.grad of the margin term per update, taken before the CE "
            "and margin terms are summed for the step; the routing state path is detached, so "
            "the margin gradient flows through the scorer only"
        ),
        "accumulated_norm": accumulated,
        "share": share,
        "added_norm_accumulated": added,
        "trunk_norm_accumulated": total - added,
        "added_share": added / total if total > 0.0 else 0.0,
        "required_names_present": all(
            name in share
            for name in (
                "W_hidden.weight",
                "W_hidden.bias",
                "w_out.weight",
                "k_null",
                "W_align.weight",
                "offset_embed",
            )
        ),
    }


def measure_ce(candidate, params, batch, tag) -> float:
    start = time.process_time()
    ce = j.batch_ce(candidate, batch, tag)
    ce.backward()
    for parameter in params:
        parameter.grad = None
    return time.process_time() - start


def measure_margin(candidate, params, batch) -> tuple[float, int]:
    start = time.process_time()
    rows, _degenerate = content_rows(candidate, batch, MARGIN_TAG)
    margin_loss, _margins = margin_average(candidate, rows)
    torch.autograd.grad(margin_loss, params, allow_unused=True)
    return time.process_time() - start, len(rows)


def pricing_units(treatment, params, batches, dev, new_fixtures, new_pairs) -> dict:
    """0-Adam cost reading: one CE backward, one margin backward, a 24-example slice."""
    ce_unit = measure_ce(treatment, params, batches[0], "19.2l-B-pricing-ce")
    margin_unit, margin_rows = measure_margin(treatment, params, batches[0])

    start = time.process_time()
    h.decode_rows(treatment, tuple(dev[:2]))
    warm_cpu = time.process_time() - start

    slice_fixtures = tuple(dev[index] for index in range(0, len(dev), 8))[:24]
    start = time.process_time()
    h.decode_rows(treatment, slice_fixtures)
    decode_slice_cpu = time.process_time() - start

    start = time.process_time()
    for fixture in new_pairs[:2]:
        wall.swap_pair(treatment, fixture, ROUTING_ARM)
    swap_slice_cpu = time.process_time() - start

    start = time.process_time()
    h.score_snapshot(treatment, tuple(new_fixtures[:6]), "19.2l-B-pricing")
    score_slice_cpu = time.process_time() - start

    decode_scale = len(dev) / len(slice_fixtures)
    swap_scale = 20 / 2
    score_scale = 96 / 6
    snapshot_cpu = (
        decode_slice_cpu * decode_scale
        + swap_slice_cpu * swap_scale
        + score_slice_cpu * score_scale
    )
    return {
        "warmup_slice_cpu": warm_cpu,
        "ce_update_cpu": ce_unit,
        "margin_update_cpu": margin_unit,
        "margin_rows": margin_rows,
        "unit_batch_index": 0,
        "decode_slice_examples": len(slice_fixtures),
        "decode_slice_cpu": decode_slice_cpu,
        "swap_slice_pairs": 2,
        "swap_slice_cpu": swap_slice_cpu,
        "score_slice_examples": 6,
        "score_slice_cpu": score_slice_cpu,
        "decode_scale": decode_scale,
        "swap_scale": swap_scale,
        "score_scale": score_scale,
        "snapshot_estimate_cpu": snapshot_cpu,
        "unit_method": (
            "one forward+backward per unit on batch 8; the prior run measured the same method "
            "twice and the second measurement was within 5% of the first, so a single reading "
            "is used and no blanket safety factor is applied"
        ),
    }


def estimate_total(units: dict, spent_cpu: float, reading_cpu: float) -> dict:
    training = UPDATES * units["ce_update_cpu"] + UPDATES * (
        units["ce_update_cpu"] + units["margin_update_cpu"]
    )
    snapshots = 3 * units["snapshot_estimate_cpu"]
    readings_after = reading_cpu + FIRST_BATCH_AFTER_CPU
    after_price = units["ce_update_cpu"] + units["margin_update_cpu"]
    raw = (
        training
        + snapshots
        + readings_after
        + SAVE_OVERHEAD_CPU
        + after_price
        + REPORT_OVERHEAD_CPU
        + spent_cpu
    )
    estimate = raw * PRICING_SAFETY
    return {
        "formula": (
            "estimate = 30*ce + 30*(ce+margin) + 3*snapshot + (after_reading + first_batch) "
            "+ save_overhead + after_price + report_overhead + cpu_already_spent; no blanket "
            "safety factor because every component is a directly measured unit (the only "
            "extrapolation is the 24->192 decode slice)"
        ),
        "ce_unit_cpu": units["ce_update_cpu"],
        "margin_unit_cpu": units["margin_update_cpu"],
        "snapshot_estimate_cpu": units["snapshot_estimate_cpu"],
        "unit_method": units["unit_method"],
        "training_estimate_cpu": training,
        "three_snapshots_estimate_cpu": snapshots,
        "minimum_floor_cpu": training + snapshots,
        "floor_over_budget": (training + snapshots) > CPU_BUDGET,
        "floor_share": {
            "training": training / (training + snapshots),
            "three_full_192_snapshots": snapshots / (training + snapshots),
        },
        "readings_after_estimate_cpu": readings_after,
        "save_overhead_cpu": SAVE_OVERHEAD_CPU,
        "after_price_estimate_cpu": after_price,
        "report_overhead_cpu": REPORT_OVERHEAD_CPU,
        "spent_before_training_cpu": spent_cpu,
        "raw_estimate_cpu": raw,
        "safety_factor": PRICING_SAFETY,
        "estimate_cpu": estimate,
        "budget_cpu": CPU_BUDGET,
        "over_budget": estimate > CPU_BUDGET,
        "adam_updates": 0,
        "extrapolated_not_measured": True,
    }


def treatment_update(candidate, optimizer, params, names, batch, update, accumulated):
    ce = j.batch_ce(candidate, batch, f"19.2l-B-treat-{update}")
    rows, _degenerate = content_rows(candidate, batch, MARGIN_TAG)
    margin_loss, margins = margin_average(candidate, rows)
    total = ce + margin_loss
    total_grads = torch.autograd.grad(total, params, retain_graph=True, allow_unused=True)
    margin_grads = torch.autograd.grad(margin_loss, params, allow_unused=True)
    for name, parameter, grad, margin_grad in zip(
        names, params, total_grads, margin_grads, strict=True
    ):
        parameter.grad = grad
        accumulated[name] += float(margin_grad.norm()) if margin_grad is not None else 0.0
    torch.nn.utils.clip_grad_norm_(params, CLIP)
    optimizer.step()
    return {
        "update": update,
        "ce": float(ce.detach()),
        "margin_loss": float(margin_loss.detach()),
        "margin_mean": statistics.fmean(margins),
        "margin_median": statistics.median(margins),
        "activated": sum(1 for value in margins if value < MARGIN_M),
        "readings": len(margins),
        "accumulated_margin_gradient_norms": dict(accumulated),
    }


def diagnostic_gates(treatment_view: dict, control_view: dict, init_view: dict) -> dict:
    route_miss_drop = control_view["new_route_miss"] - treatment_view["new_route_miss"]
    gates = {
        "swap_hard_argmax_ge_8_of_20": {
            "measured": treatment_view["swap_argmax_changed"],
            "of": treatment_view["swap_pairs"],
            "threshold": ">= 8/20",
            "pass": treatment_view["swap_argmax_changed"] >= ARGMAX_GATE,
            "reference": {
                "init": init_view["swap_argmax_changed"],
                "control": control_view["swap_argmax_changed"],
            },
        },
        "new_route_miss_down_vs_ce_ge_0_10": {
            "treatment": treatment_view["new_route_miss"],
            "control": control_view["new_route_miss"],
            "drop": route_miss_drop,
            "threshold": ">= 0.10 (control - treatment)",
            "pass": route_miss_drop >= ROUTE_MISS_DELTA_GATE,
        },
        "new_conditional_byte_object_no_regression_gt_0_05": {
            "treatment": treatment_view["new_conditional_byte_object"],
            "control": control_view["new_conditional_byte_object"],
            "drop": control_view["new_conditional_byte_object"]
            - treatment_view["new_conditional_byte_object"],
            "threshold": "<= 0.05 regression versus control",
            "pass": treatment_view["new_conditional_byte_object"]
            >= control_view["new_conditional_byte_object"] - CONDITIONAL_SLACK,
        },
        "new_conditional_byte_place_no_regression_gt_0_05": {
            "treatment": treatment_view["new_conditional_byte_place"],
            "control": control_view["new_conditional_byte_place"],
            "drop": control_view["new_conditional_byte_place"]
            - treatment_view["new_conditional_byte_place"],
            "threshold": "<= 0.05 regression versus control",
            "pass": treatment_view["new_conditional_byte_place"]
            >= control_view["new_conditional_byte_place"] - CONDITIONAL_SLACK,
        },
        "old_description_pass_no_regression": {
            "treatment": treatment_view["old_pass"],
            "control": control_view["old_pass"],
            "init": init_view["old_pass"],
            "threshold": ">= control (and >= init for reference)",
            "pass": treatment_view["old_pass"] >= control_view["old_pass"],
        },
    }
    return {
        "kind": "diagnostic_gate_not_task_gate",
        "gates": gates,
        "all_pass": all(entry["pass"] for entry in gates.values()),
    }


def attribution(treatment_view: dict, readings: dict, improvement_ci: dict) -> dict:
    key_after = readings["key_diff_after"]["median"]
    key_drop = (BASELINE_KEY_DIFF_MEDIAN - key_after) / BASELINE_KEY_DIFF_MEDIAN
    margin_before = readings["score_margin_before"]["median"]
    margin_after = readings["score_margin_after"]["median"]
    improvement = margin_after - margin_before
    key_fell = key_drop > KEY_DROP_THRESHOLD
    margin_flat = margin_after <= MARGIN_MEDIAN_FLOOR or improvement < MARGIN_IMPROVE_FLOOR
    if key_fell:
        code = "washout_upstream"
    elif margin_flat:
        code = "margin_not_trainable_in_30"
    else:
        code = "washout_at_readout_proven"
    rate = improvement / UPDATES
    if rate > 0.0:
        steps_to_significant = math.ceil((MARGIN_SIGNIFICANT - margin_before) / rate)
        extrapolated = {
            "linear_extrapolation": True,
            "rate_per_update": rate,
            "steps_to_median_0_05": steps_to_significant,
            "note": (
                "linear extrapolation of the observed 30-update median change; it is not a "
                "measurement and the dose may saturate"
            ),
        }
    else:
        extrapolated = {
            "linear_extrapolation": True,
            "rate_per_update": rate,
            "steps_to_median_0_05": None,
            "note": "the observed median change is non-positive, so no step count is finite",
        }
    return {
        "code": code,
        "priority_rule": (
            "1) key_diff drop > 20% versus 1.491 -> washout_upstream; 2) margin median still "
            "<= -0.003 or improvement < 0.01 -> margin_not_trainable_in_30; 3) otherwise "
            "washout_at_readout_proven"
        ),
        "inputs": {
            "key_diff_baseline_median": BASELINE_KEY_DIFF_MEDIAN,
            "key_diff_after_median": key_after,
            "key_diff_relative_drop": key_drop,
            "key_diff_fell_over_20pct": key_fell,
            "margin_before_median": margin_before,
            "margin_after_median": margin_after,
            "margin_improvement": improvement,
            "margin_flat": margin_flat,
        },
        "margin_significance": {
            "threshold": MARGIN_SIGNIFICANT,
            "median_after": margin_after,
            "mean_after": readings["score_margin_after"]["mean"],
            "significant_by_median": margin_after >= MARGIN_SIGNIFICANT,
            "paired_bootstrap_improvement": improvement_ci,
        },
        "extrapolation": extrapolated,
        "readout_evidence": {
            "swap_argmax_changed": treatment_view["swap_argmax_changed"],
            "new_route_miss": treatment_view["new_route_miss"],
        },
        "next_step_only": NEXT_STEPS[code],
        "all_next_steps": NEXT_STEPS,
    }


def over_budget_report(design, estimate, units, cost, identities, parent_sha) -> dict:
    return {
        "schema": "taiji-package19-2l-order-decoy-diagnostic-v1",
        "step": "N19.2l-B order-decoy margin diagnostic",
        "phase": "pricing gate (stopped before any Adam step)",
        "date": "2026-11-05",
        "design": design,
        "pricing_gate": {
            "code": "over_budget",
            "stopped_before_training": True,
            "note": (
                f"the measured extrapolation exceeds the {CPU_BUDGET:g} CPU cap: training "
                f"{estimate['training_estimate_cpu']:.2f} + three full 192-example snapshots "
                f"{estimate['three_snapshots_estimate_cpu']:.2f} = "
                f"{estimate['minimum_floor_cpu']:.2f} floor, plus readings, save, closing "
                f"price and report overhead, plus the {estimate['spent_before_training_cpu']:.2f}"
                f" already spent, totals {estimate['estimate_cpu']:.2f}; nothing was trained "
                "and no bundle was written"
            ),
            **estimate,
            "units": units,
        },
        "machine_hygiene": MACHINE_HYGIENE,
        "price_measurement": {
            "unit_price_before": {
                "ce": units["ce_update_cpu"],
                "margin": units["margin_update_cpu"],
            },
            "owner_reference_before": {
                "ce": CE_BEFORE_REFERENCE,
                "margin": MARGIN_BEFORE_REFERENCE,
            },
            "unit_price_after": {"status": "not_measured, 0 updates"},
            "price_verdict": "not_evaluated",
            "reason": "the pricing gate stopped the run before any update, so no closing price exists",
        },
        "arms": {
            "status": "not_run_over_budget",
            "treatment": "N x (selection CE + order-decoy margin), 0/30 updates",
            "control": "N x selection CE only, 0/30 updates",
        },
        "delta_bootstrap": {"status": "not_run_over_budget"},
        "argmax": {"status": "not_run_over_budget"},
        "readings": {
            "key_diff_after": {"status": "not_measured, 0 updates"},
            "score_margin_after": {"status": "not_measured, 0 updates"},
            "decoy_margin_after": {"status": "not_measured, 0 updates"},
            "gradient_share_by_param": {"status": "not_measured, 0 updates"},
        },
        "attribution_verdict": {
            "code": None,
            "status": "not_evaluated",
            "reason": "pricing gate stopped the run before any Adam step",
            "priority_rule": (
                "1) key_diff drop > 20% versus 1.491 -> washout_upstream; 2) margin median still "
                "<= -0.003 or improvement < 0.01 -> margin_not_trainable_in_30; 3) otherwise "
                "washout_at_readout_proven"
            ),
        },
        "diagnostic_gates": {"status": "not_run_over_budget"},
        "density": {
            "planned_init_readings": 128,
            "planned_first_batch_readings": 16,
            "planned_dev_examples": 192,
            "planned_swap_pairs": 20,
            "planned_updates_per_arm": 30,
            "planned_snapshot_points": 3,
        },
        "basis": {
            "kind": "diagnostic",
            "not_a_task_gate": True,
            "status": "not_run_over_budget",
        },
        "cost": cost,
        "identities": identities,
        "reproducibility": {
            "parent_bundle": str(j.PARENT.relative_to(ROOT)),
            "parent_sha256": parent_sha,
            "seed": SEED,
            "torch_threads": 1,
        },
    }


def main() -> int:
    start_cpu = time.process_time()
    start_wall = time.perf_counter()
    PROGRESS.unlink(missing_ok=True)

    payload = premise.build_payload()
    train = premise.fixtures(payload, "train")
    dev = premise.fixtures(payload, "dev")
    new_fixtures = tuple(fixture for fixture in dev if fixture["flags"]["new_description"])
    if len(dev) != 192 or len(new_fixtures) != 96:
        raise RuntimeError("dev stratification drift")
    _easy_pairs, new_pairs = wall.pair_fixtures(dev)
    if len(new_pairs) != 20:
        raise RuntimeError("new-description swap pair drift")
    sample = gate.even_sample(train, SAMPLE)
    batches = h.schedule(train, UPDATES)

    parent = load_routing_bundle(j.PARENT)[0]
    if any(torch.isnan(tensor).any().item() for tensor in parent.head.parameters()):
        raise RuntimeError("parent head carries NaN (D16-6)")
    parent_sha = sha256_file(j.PARENT)

    treatment, treatment_optimizer, treatment_params = j.build_run(parent, SEED, ARM)
    control, control_optimizer, control_params = j.build_run(parent, SEED, ARM)
    named = [
        (name, parameter)
        for name, parameter in treatment.head.named_parameters()
        if parameter.requires_grad
    ]
    names = [name for name, _parameter in named]
    params = [parameter for _name, parameter in named]
    if len(params) != len(treatment_params):
        raise RuntimeError("head parameter order drift")
    init_diff = max(
        float((left.detach() - right.detach()).abs().max())
        for left, right in zip(treatment.head.parameters(), control.head.parameters(), strict=True)
    )
    if init_diff != 0.0:
        raise RuntimeError("the two arms do not share the same initialisation")
    treatment_identity = candidate_identity(treatment)
    control_identity = candidate_identity(control)
    if treatment_identity["candidate_sha_v2"] != control_identity["candidate_sha_v2"]:
        raise RuntimeError("N/N arms must have identical identity at step 0")

    order_sha = hashlib.sha256(
        json.dumps(
            [[[fixture["world"], fixture["index"]] for fixture in batch] for batch in batches]
        ).encode("utf-8")
    ).hexdigest()

    j.progress_append(
        PROGRESS,
        {
            "event": "start",
            "seed": SEED,
            "structure": ARM,
            "updates": UPDATES,
            "adam_cap": ADAM_CAP,
            "cpu_budget": CPU_BUDGET,
            "init_max_abs_diff": init_diff,
            "data_order_sha256": order_sha,
        },
    )

    # ---- pricing gate: measured units, 0 Adam ----------------------------
    units = pricing_units(treatment, params, batches, dev, new_fixtures, new_pairs)
    reading_start = time.process_time()
    before_rows, degenerate = content_rows(treatment, sample, MARGIN_TAG)
    reading_cpu = time.process_time() - reading_start
    spent = time.process_time() - start_cpu
    estimate = estimate_total(units, spent, reading_cpu)
    j.progress_append(
        PROGRESS,
        {"event": "pricing", **units, **estimate, "reading_cpu": reading_cpu},
    )

    design = {
        "parent": str(j.PARENT.relative_to(ROOT)),
        "parent_arm": "B_stop_lock",
        "parent_update": 300,
        "seed": SEED,
        "arms": {
            "treatment": "N alignment structure x (selection CE + order-decoy margin, m = 1.0)",
            "control": "N alignment structure x selection CE only",
        },
        "only_variable": "the objective (margin term present or absent); structure, init, data order, LR, clip and update count are shared",
        "decoy_rule": (
            "gold record key split at its character midpoint with the two halves swapped; "
            "byte multiset, byte length and byte inventory identical, only the order differs"
        ),
        "margin_term": (
            "per example 2 content starts (object start 3 / place start 21), one order decoy "
            "each; loss = mean(relu(1.0 - (score_gold - score_decoy))); the decoy key is "
            "binding.base.relation.pool(binding.base.relation.encode_units(subject, decoy))"
        ),
        "training": {
            "updates_per_arm": UPDATES,
            "batch": BATCH,
            "lr": 1e-3,
            "clip": CLIP,
            "adam_cap_total": ADAM_CAP,
            "stage": "selection only, freeze_to_selection",
            "data_order": "h.schedule(train, 30), identical batches for both arms",
            "order_sha256": order_sha,
        },
        "evaluation": {
            "dev_examples": len(dev),
            "new_description_examples": len(new_fixtures),
            "swap_pairs": len(new_pairs),
            "points": ["step0 (shared by both arms)", "step30"],
            "shared_init_snapshot": True,
        },
        "budget": {
            "cpu_cap": CPU_BUDGET,
            "adam_cap": ADAM_CAP,
            "bundles": 1,
            "confirm_reads": 0,
        },
    }

    if estimate["over_budget"]:
        cost = {
            "cpu_seconds": time.process_time() - start_cpu,
            "wall_seconds": time.perf_counter() - start_wall,
            "cpu_budget": CPU_BUDGET,
            "adam_updates": 0,
            "bundles_saved": 0,
            "confirm_reads": 0,
            "over_budget_triggered": True,
            "operational_waste_cpu_s": MACHINE_HYGIENE["operational_waste_cpu_s"],
        }
        report = over_budget_report(
            design,
            estimate,
            units,
            cost,
            {
                "treatment_sha_v2": treatment_identity["candidate_sha_v2"],
                "control_sha_v2": control_identity["candidate_sha_v2"],
            },
            parent_sha,
        )
        j.write_json_safe(REPORT, report)
        j.write_json_safe(
            MANIFEST,
            {
                "schema": "taiji-package19-2l-order-decoy-diagnostic-manifest-v1",
                "step": "N19.2l-B",
                "date": "2026-11-05",
                "report": str(REPORT.relative_to(ROOT)),
                "pricing_gate": "over_budget",
                "estimate_cpu": estimate["estimate_cpu"],
                "minimum_floor_cpu": estimate["minimum_floor_cpu"],
                "adam_updates": 0,
                "bundles_saved": 0,
                "confirm_reads": 0,
            },
        )
        print(
            json.dumps(
                {
                    "pricing_gate": "over_budget",
                    "estimate_cpu": round(estimate["estimate_cpu"], 1),
                    "cpu": round(cost["cpu_seconds"], 1),
                },
                ensure_ascii=False,
            )
        )
        return 0

    # ---- step0: one shared snapshot for both arms ----------------------------
    init_snapshot = h.full_snapshot(treatment, dev, new_fixtures, new_pairs, "19.2l-B-step0")
    init_view = arm_view(init_snapshot, shared=True)
    j.progress_append(
        PROGRESS,
        {
            "event": "step0_snapshot",
            "shared_init_snapshot": True,
            "new_route_miss": init_view["new_route_miss"],
            "old_pass": init_view["old_pass"],
            "swap_argmax_changed": init_view["swap_argmax_changed"],
            "new_conditional_byte_object": init_view["new_conditional_byte_object"],
            "new_conditional_byte_place": init_view["new_conditional_byte_place"],
        },
    )

    before_values = margin_values(treatment, before_rows)
    margin_before = margin_summary(before_values)
    key_before = gate.distribution([row["key_max_abs_delta"] for row in before_rows])
    first_before_rows, _first_degenerate = content_rows(treatment, batches[0], MARGIN_TAG)
    decoy_margin_before = margin_reading(treatment, first_before_rows)
    j.progress_append(
        PROGRESS,
        {
            "event": "step0_readings",
            "margin_before": margin_before["distribution"],
            "key_diff_before": key_before,
            "decoy_margin_before": decoy_margin_before["distribution"],
            "degenerate_decoys": degenerate,
        },
    )

    # ---- two trajectories, identical batches --------------------------------
    accumulated = {name: 0.0 for name in names}
    treatment_log = []
    for update in range(1, UPDATES + 1):
        entry = treatment_update(
            treatment,
            treatment_optimizer,
            params,
            names,
            batches[update - 1],
            update,
            accumulated,
        )
        treatment_log.append(entry)
        j.progress_append(PROGRESS, {"event": "treatment_update", **entry})

    control_log, control_updates, _stopped = j.train_run(
        control, control_optimizer, control_params, batches, 0, "19.2l-B-control"
    )
    for entry in control_log:
        j.progress_append(PROGRESS, {"event": "control_update", **entry})

    total_adam = UPDATES + control_updates
    if total_adam > ADAM_CAP:
        raise RuntimeError("adam cap exceeded")

    # ---- step30 snapshots ---------------------------------------------------
    treatment_snapshot = h.full_snapshot(
        treatment, dev, new_fixtures, new_pairs, "19.2l-B-step30-treat"
    )
    control_snapshot = h.full_snapshot(
        control, dev, new_fixtures, new_pairs, "19.2l-B-step30-control"
    )
    treatment_view = arm_view(treatment_snapshot, shared=False)
    control_view = arm_view(control_snapshot, shared=False)
    for label, view in (("treatment", treatment_view), ("control", control_view)):
        j.progress_append(PROGRESS, {"event": "step30_snapshot", "arm": label, **view})

    # ---- closing unit price: same two units, same process, batch 8 ----------
    after_price_start = time.process_time()
    ce_after_unit = measure_ce(treatment, params, batches[0], "19.2l-B-post-ce")
    margin_after_unit, _post_rows = measure_margin(treatment, params, batches[0])
    after_price_cpu = time.process_time() - after_price_start
    j.progress_append(
        PROGRESS,
        {
            "event": "unit_price_after",
            "ce": ce_after_unit,
            "margin": margin_after_unit,
            "measured_after_price_cpu": after_price_cpu,
            "unit_batch_index": 0,
        },
    )

    # ---- after readings -----------------------------------------------------
    after_rows, after_degenerate = content_rows(treatment, sample, MARGIN_TAG)
    after_values = margin_values(treatment, after_rows)
    margin_after = margin_summary(after_values)
    key_after = gate.distribution([row["key_max_abs_delta"] for row in after_rows])
    first_after_rows, _first_after = content_rows(treatment, batches[0], MARGIN_TAG)
    decoy_margin_after = margin_reading(treatment, first_after_rows)

    improvement_ci = paired_bootstrap(
        before_values, after_values, BOOTSTRAP_ROUNDS, BOOTSTRAP_SEED + 7
    )
    j.progress_append(
        PROGRESS,
        {
            "event": "readings_after",
            "key_diff_after": key_after,
            "score_margin_after": margin_after,
            "decoy_margin_after": decoy_margin_after,
            "margin_improvement_bootstrap": improvement_ci,
            "degenerate_decoys": after_degenerate,
        },
    )

    # ---- delta: treatment - control, world-clustered bootstrap --------------
    delta = {
        "route_miss": cluster_bootstrap(
            treatment_snapshot["rows_new96"],
            control_snapshot["rows_new96"],
            "route_miss",
            BOOTSTRAP_ROUNDS,
            BOOTSTRAP_SEED,
        ),
        "route_hit_object": cluster_bootstrap(
            treatment_snapshot["rows_new96"],
            control_snapshot["rows_new96"],
            "route_hit_object",
            BOOTSTRAP_ROUNDS,
            BOOTSTRAP_SEED + 1,
        ),
        "route_hit_place": cluster_bootstrap(
            treatment_snapshot["rows_new96"],
            control_snapshot["rows_new96"],
            "route_hit_place",
            BOOTSTRAP_ROUNDS,
            BOOTSTRAP_SEED + 2,
        ),
        "completed": cluster_bootstrap(
            treatment_snapshot["rows_new96"],
            control_snapshot["rows_new96"],
            "completed",
            BOOTSTRAP_ROUNDS,
            BOOTSTRAP_SEED + 3,
        ),
        "rule": (
            "Delta = N(treatment) - N(control) on the 96 new-description dev examples; the "
            "bootstrap resamples the 12 worlds with replacement (2000 rounds, fixed seed), so "
            "the example dependence inside a world is preserved"
        ),
    }
    j.progress_append(
        PROGRESS,
        {
            "event": "delta",
            **{
                key: {name: value for name, value in cell.items() if name != "per_world"}
                for key, cell in delta.items()
                if isinstance(cell, dict)
            },
        },
    )

    # ---- save the one candidate bundle, then save -> load bitwise -----------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prepared, _context = j.cmp().prepared_pair(treatment, dev[0], "19.2l-B-save", "save")
    saved = save_routing_bundle(
        FINAL_BUNDLE,
        treatment,
        treatment_optimizer,
        UPDATES,
        {
            "stage": "N19.2l-B",
            "seed": SEED,
            "structure": ARM,
            "update": UPDATES,
            "loss": "selection_ce_plus_order_decoy_margin",
            "margin_m": MARGIN_M,
        },
        candidate_identity(treatment),
        prepared.state,
    )
    reloaded, _reload_optimizer, _state, reload_index, reload_cursor = load_routing_bundle(
        FINAL_BUNDLE
    )
    trained_tensors = parameter_tensors(treatment)
    reloaded_tensors = parameter_tensors(reloaded)
    reload_diff = float("inf")
    if set(trained_tensors) == set(reloaded_tensors):
        reload_diff = max(
            float((value - reloaded_tensors[name]).abs().max())
            for name, value in trained_tensors.items()
        )
    round_trip_ok = reload_diff == 0.0
    j.progress_append(
        PROGRESS,
        {
            "event": "saved_bundle",
            "path": saved["path"],
            "bytes": saved["bytes"],
            "sha256": saved["sha256"],
            "reload_max_abs_diff": reload_diff,
            "round_trip_ok": round_trip_ok,
            "reload_update_index": reload_index,
        },
    )

    # ---- attribution -----------------------------------------------------
    readings = {
        "key_diff_after": {
            **key_after,
            "before": key_before,
            "baseline_A_median": BASELINE_KEY_DIFF_MEDIAN,
            "relative_drop_vs_A_median": (BASELINE_KEY_DIFF_MEDIAN - key_after["median"])
            / BASELINE_KEY_DIFF_MEDIAN,
            "note": (
                "the binding is frozen by freeze_to_selection, so the key readings cannot move "
                "under training; the after value is measured, not assumed"
            ),
        },
        "score_margin_before": {
            **margin_before["distribution"],
            "activated_rate": margin_before["activated_rate"],
            "sample": "64 train examples x 2 content starts, the A even-stride sample, step0",
        },
        "score_margin_after": {
            **margin_after["distribution"],
            "activated_rate": margin_after["activated_rate"],
            "activated_count": margin_after["activated_count"],
            "before": margin_before["distribution"],
            "before_activated_rate": margin_before["activated_rate"],
            "improvement": {
                "median": margin_after["distribution"]["median"]
                - margin_before["distribution"]["median"],
                "mean": margin_after["distribution"]["mean"]
                - margin_before["distribution"]["mean"],
            },
            "paired_bootstrap_improvement": improvement_ci,
            "margin_m": MARGIN_M,
            "sample": "64 train examples x 2 content starts, the A even-stride sample",
        },
        "decoy_margin_after": {
            **decoy_margin_after["distribution"],
            "activated_rate": decoy_margin_after["activated_rate"],
            "before": decoy_margin_before["distribution"],
            "sample": "first training batch, 8 examples x 2 content starts (16 readings)",
        },
        "gradient_share_by_param": gradient_share(accumulated),
    }
    verdict = attribution(treatment_view, readings, improvement_ci)
    j.progress_append(PROGRESS, {"event": "attribution", **verdict})

    gates = diagnostic_gates(treatment_view, control_view, init_view)
    j.progress_append(
        PROGRESS,
        {
            "event": "diagnostic_gates",
            **{name: entry["pass"] for name, entry in gates["gates"].items()},
            "all_pass": gates["all_pass"],
        },
    )

    cpu = time.process_time() - start_cpu
    wall_time = time.perf_counter() - start_wall
    price_verdict = (
        "contention_at_start"
        if ce_after_unit <= CE_BEFORE_REFERENCE * CONTENTION_RATIO
        else "price_regime_changed"
    )
    price_measurement = {
        "unit_price_before": {
            "ce": CE_BEFORE_REFERENCE,
            "margin": MARGIN_BEFORE_REFERENCE,
        },
        "unit_price_before_this_run": {
            "ce": units["ce_update_cpu"],
            "margin": units["margin_update_cpu"],
        },
        "unit_price_after": {"ce": ce_after_unit, "margin": margin_after_unit},
        "ce_ratio_vs_before": ce_after_unit / CE_BEFORE_REFERENCE,
        "rule": (
            "contention_at_start if ce_after <= 6.02 * 0.7; otherwise price_regime_changed "
            "(recorded only, the 800 budget accounting is unchanged)"
        ),
        "price_verdict": price_verdict,
        "budget_occupancy": cpu / CPU_BUDGET,
        "budget_cpu": CPU_BUDGET,
        "cpu_seconds": cpu,
        "unit_method": units["unit_method"],
    }
    cost = {
        "cpu_seconds": cpu,
        "wall_seconds": wall_time,
        "cpu_budget": CPU_BUDGET,
        "budget_occupancy": cpu / CPU_BUDGET,
        "adam_updates": total_adam,
        "adam_per_arm": UPDATES,
        "adam_cap": ADAM_CAP,
        "bundles_saved": 1,
        "confirm_reads": 0,
        "over_budget_triggered": False,
        "pricing_estimate_cpu": estimate["estimate_cpu"],
        "price_verdict": price_verdict,
        "operational_waste_cpu_s": MACHINE_HYGIENE["operational_waste_cpu_s"],
        "pricing_units_cpu": {
            "ce_update": units["ce_update_cpu"],
            "margin_update": units["margin_update_cpu"],
            "snapshot_estimate": units["snapshot_estimate_cpu"],
        },
        "over_budget_cpu": cpu > CPU_BUDGET,
    }

    report = {
        "schema": "taiji-package19-2l-order-decoy-diagnostic-v1",
        "step": "N19.2l-B order-decoy margin diagnostic",
        "phase": "30-update treatment/control diagnostic (not a task gate)",
        "date": "2026-11-05",
        "purpose": (
            "decide whether the order-decoy margin term, which is not structurally zero on the "
            "N head, reaches the readout at a 30-update dose: upstream key washout, "
            "not-trainable-in-30, or at-readout washout"
        ),
        "design": design,
        "pricing_gate": {**estimate, "units": units, "code": "under_budget"},
        "machine_hygiene": MACHINE_HYGIENE,
        "price_measurement": price_measurement,
        "identities": {
            "parent_sha_v2": candidate_identity(parent)["candidate_sha_v2"],
            "treatment_step0_sha_v2": treatment_identity["candidate_sha_v2"],
            "control_step0_sha_v2": control_identity["candidate_sha_v2"],
            "treatment_step30_sha_v2": candidate_identity(treatment)["candidate_sha_v2"],
            "control_step30_sha_v2": candidate_identity(control)["candidate_sha_v2"],
            "init_max_abs_diff_between_arms": init_diff,
        },
        "arms": {
            "treatment": {
                "structure": "N_alignment",
                "loss": "selection CE + order-decoy margin (m = 1.0)",
                "step0": init_view,
                "step30": treatment_view,
                "training_log": treatment_log,
            },
            "control": {
                "structure": "N_alignment",
                "loss": "selection CE only",
                "step0": init_view,
                "step30": control_view,
                "training_log": control_log,
            },
        },
        "delta_bootstrap": delta,
        "argmax": {
            "step0_shared": {
                "pairs": init_view["swap_pairs"],
                "hard_argmax_changed": init_view["swap_argmax_changed"],
                "decode_route_argmax_changed": init_view["swap_decode_route_argmax_changed"],
            },
            "treatment_step30": {
                "pairs": treatment_view["swap_pairs"],
                "hard_argmax_changed": treatment_view["swap_argmax_changed"],
                "decode_route_argmax_changed": treatment_view["swap_decode_route_argmax_changed"],
            },
            "control_step30": {
                "pairs": control_view["swap_pairs"],
                "hard_argmax_changed": control_view["swap_argmax_changed"],
                "decode_route_argmax_changed": control_view["swap_decode_route_argmax_changed"],
            },
        },
        "readings": readings,
        "attribution_verdict": verdict,
        "diagnostic_gates": gates,
        "density": {
            "init_readings": len(before_rows),
            "first_batch_readings": len(first_before_rows),
            "dev_examples": len(dev),
            "new_description_examples": len(new_fixtures),
            "swap_pairs": len(new_pairs),
            "worlds": 12,
            "updates_per_arm": UPDATES,
            "snapshot_points": 3,
            "shared_init_snapshot": True,
            "bootstrap_rounds": BOOTSTRAP_ROUNDS,
        },
        "basis": {
            "kind": "diagnostic",
            "not_a_task_gate": True,
            "rule": (
                "the gates and the verdict only say where the order-decoy signal is lost "
                "(upstream key, dose, or readout); they never license a task-level capability "
                "claim and no confirm example is read"
            ),
        },
        "saved_bundle": {
            **saved,
            "reload_max_abs_diff": reload_diff,
            "round_trip_ok": round_trip_ok,
            "reload_update_index": reload_index,
            "reload_cursor": dict(reload_cursor),
        },
        "cost": cost,
        "reproducibility": {
            "parent_bundle": str(j.PARENT.relative_to(ROOT)),
            "parent_sha256": parent_sha,
            "seed": SEED,
            "torch_threads": 1,
            "data_order_sha256": order_sha,
            "two_arms_identical_at_step0": init_diff == 0.0,
        },
    }
    j.write_json_safe(REPORT, report)
    manifest = {
        "schema": "taiji-package19-2l-order-decoy-diagnostic-manifest-v1",
        "step": "N19.2l-B",
        "date": "2026-11-05",
        "report": str(REPORT.relative_to(ROOT)),
        "parent_sha256": parent_sha,
        "pricing_gate": "under_budget",
        "pricing_estimate_cpu": estimate["estimate_cpu"],
        "attribution_verdict": verdict["code"],
        "key_diff_prior_median": BASELINE_KEY_DIFF_MEDIAN,
        "key_diff_after_median": readings["key_diff_after"]["median"],
        "score_margin_before_median": readings["score_margin_before"]["median"],
        "score_margin_after_median": readings["score_margin_after"]["median"],
        "decoy_margin_after_median": readings["decoy_margin_after"]["median"],
        "swap_argmax_step0": init_view["swap_argmax_changed"],
        "swap_argmax_treatment": treatment_view["swap_argmax_changed"],
        "swap_argmax_control": control_view["swap_argmax_changed"],
        "new_route_miss_treatment": treatment_view["new_route_miss"],
        "new_route_miss_control": control_view["new_route_miss"],
        "old_pass_treatment": treatment_view["old_pass"],
        "old_pass_control": control_view["old_pass"],
        "delta_route_miss": delta["route_miss"]["delta"],
        "delta_route_miss_ci": [
            delta["route_miss"]["ci_low"],
            delta["route_miss"]["ci_high"],
        ],
        "diagnostic_gates_all_pass": gates["all_pass"],
        "price_verdict": price_verdict,
        "unit_price_before": price_measurement["unit_price_before"],
        "unit_price_after": price_measurement["unit_price_after"],
        "budget_occupancy": cpu / CPU_BUDGET,
        "operational_waste_cpu_s": MACHINE_HYGIENE["operational_waste_cpu_s"],
        "saved_bundle": saved["path"],
        "saved_bundle_sha256": saved["sha256"],
        "round_trip_ok": round_trip_ok,
        "adam_updates": total_adam,
        "bundles_saved": 1,
        "confirm_reads": 0,
        "cost_cpu": cpu,
        "over_budget": cpu > CPU_BUDGET,
    }
    j.write_json_safe(MANIFEST, manifest)
    print(
        json.dumps(
            {
                "attribution_verdict": verdict["code"],
                "key_diff_after_median": round(readings["key_diff_after"]["median"], 4),
                "margin_before_median": round(readings["score_margin_before"]["median"], 4),
                "margin_after_median": round(readings["score_margin_after"]["median"], 4),
                "margin_after_mean": round(readings["score_margin_after"]["mean"], 4),
                "decoy_margin_after_median": round(readings["decoy_margin_after"]["median"], 4),
                "swap_argmax": [
                    init_view["swap_argmax_changed"],
                    control_view["swap_argmax_changed"],
                    treatment_view["swap_argmax_changed"],
                ],
                "route_miss": [
                    init_view["new_route_miss"],
                    control_view["new_route_miss"],
                    treatment_view["new_route_miss"],
                ],
                "old_pass": [
                    init_view["old_pass"],
                    control_view["old_pass"],
                    treatment_view["old_pass"],
                ],
                "gates_all_pass": gates["all_pass"],
                "price_verdict": price_verdict,
                "unit_price_after": {
                    "ce": round(ce_after_unit, 3),
                    "margin": round(margin_after_unit, 3),
                },
                "budget_occupancy": round(cpu / CPU_BUDGET, 4),
                "adam": total_adam,
                "cpu": round(cpu, 1),
                "wall": round(wall_time, 1),
                "saved_round_trip": round_trip_ok,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
