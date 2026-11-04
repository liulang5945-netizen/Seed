"""19.2j: from-scratch selection-stage architecture A/B (O baseline vs N alignment).

Owner ruling (2026-11-04): 19.2g/19.2h zero results are instrument failures, not
structural ceilings (D16-12).  Architecture claims therefore have to be tested on
fresh, identically initialised selection heads trained from zero, with the only
difference being the scoring form:

  O arm  -- current form: ``W_hidden(4*width + prev_width) -> w_out`` over pooled
            description/record features.
  N arm  -- 19.2h sequence-surface form: the same trunk plus ``W_align`` 32x32,
            a 5x8 offset embedding and four extra row inputs, **re-initialised at
            trunk scale** (``1/sqrt(hidden)``) instead of the +-0.01 used in 19.2h.

Everything else is held equal: same parent binding graph (frozen), same parent
span pointer (frozen, no span supervision), same data order, same batch/LR/clip,
same 300 updates, same loss.  1,200 Adam total (2 seeds x 2 arms x 300).

Run::

    python scripts/training/train_taiji_package19_2j_from_scratch_selection_ab.py
"""

from __future__ import annotations

import copy
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

torch.set_num_threads(1)

import attribute_taiji_package19_2f_new_description_wall as wall  # noqa: E402
import train_taiji_package19_2h_sequence_surface as h  # noqa: E402

from taiji import data_premise_18_0 as premise  # noqa: E402
from taiji.candidate_identity import candidate_identity  # noqa: E402
from taiji.selection_routing import (  # noqa: E402
    OrderedSelectionHead,
    RoutingConfig,
    route_labels,
)
from taiji.selection_routing_checkpoint import (  # noqa: E402
    RoutingCandidate,
    load_routing_bundle,
    make_routing_optimizer,
    parameter_tensors,
    save_routing_bundle,
)

PARENT = ROOT / "output/taiji_package19_compare/28031-B_stop_lock-step300.pt"
SEEDS = (28031, 28032)
UPDATES = 300
BATCH = 8
ADAM_CAP = 1200
PAIRS_PER_STRATUM = 20
ARGMAX_GATE = 18
ROUTE_HIT_GATE = 0.90
ROUTE_MISS_GATE = 0.05
OLD_ROUTE_SLACK = 0.02
OBJECT_START_GATE = 0.70
CONDITIONAL_GATE = 0.90
BOOTSTRAP_ROUNDS = 2000
BOOTSTRAP_SEED = 20261104
RESCUE_EXAMPLES = 8
WARMUP_UPDATES = 5

OUTPUT_DIR = ROOT / "output/taiji_package19_2j_from_scratch_selection_ab"
PREFLIGHT_BUNDLE = OUTPUT_DIR / "preflight-step0.pt"
CORRUPT_BUNDLE = OUTPUT_DIR / "drift-probe.pt"
REPORT = ROOT / "reports/taiji_package19_2j_from_scratch_selection_ab_20261104.json"
MANIFEST = ROOT / "plans/manifests/taiji_package19_2j_from_scratch_selection_ab_20261104.json"

# Owner budget rule: functional item list x 3.3, hard cap 12,000 CPU.  The first
# item is re-priced with the measured marginal CPU after the five-update warm-up;
# the listed 4,800 is the owner's own 4 CPU/update estimate, kept as the upper bound.
BUDGET_ITEMS = (
    (
        "1,200 from-scratch selection updates (2 seeds x 2 arms x 300), batch 8, CE only"
        " (owner estimate 4 CPU/update; re-priced by the warm-up measurement)",
        4800.0,
    ),
    ("four full 192-example decode passes at step 300 (one per run, ~90 CPU each)", 380.0),
    ("four swap summaries on 20 new-description pairs", 40.0),
    ("identity rescue 8 examples + checkpoint dry-run + N-arm drift rejection", 40.0),
    ("gradient probe at init and step 300, payload, parent health, report", 40.0),
)
SAFETY_FACTOR = 3.3
BUDGET_CAP = 12000.0
ANCILLARY_CPU = 500.0
MARGINAL_CPU_PER_UPDATE = 3.7

ARMS = {"O": "O_baseline", "N": "N_alignment"}


def cmp():
    return h.cmp()


def build_run(parent, seed: int, arm: str):
    """Fresh selection head from zero; O and N share every shared tensor exactly.

    The O head is drawn with ``seed``.  The N head is drawn with the same seed and
    then overwritten on the shared tensors, so the two arms start from identical
    ``W_hidden[:, :576]`` / ``bias`` / ``w_out`` / ``k_null``.  Only the alignment
    block differs, and it is re-initialised at trunk scale:

    * ``W_align``     uniform +-1/sqrt(32)  (same rule the head already uses for 32-d blocks)
    * ``offset_embed`` uniform +-1/sqrt(hidden) = +-0.125

    19.2h used +-0.01 for ``offset_embed``, which made the four extra inputs two
    orders of magnitude smaller than the pooled features; with the 5 correlations
    in [-1, 1] that is exactly the magnitude that gets drowned at this dose.
    Trunk scale keeps the added inputs the same order as the existing 576 inputs.
    """
    base = OrderedSelectionHead(RoutingConfig(alignment=False), seed=seed)
    if arm == "O":
        head = base
    else:
        head = OrderedSelectionHead(RoutingConfig(alignment=True), seed=seed)
        with torch.no_grad():
            width_in = base.W_hidden.weight.shape[1]
            head.W_hidden.weight[:, :width_in].copy_(base.W_hidden.weight)
            head.W_hidden.bias.copy_(base.W_hidden.bias)
            head.w_out.weight.copy_(base.w_out.weight)
            head.k_null.copy_(base.k_null)
            bound = 1.0 / float(head.config.hidden) ** 0.5
            generator = torch.Generator().manual_seed(seed)
            block = head.config.width // 4
            align_bound = 1.0 / float(block) ** 0.5
            head.W_align.weight.uniform_(-align_bound, align_bound, generator=generator)
            head.offset_embed.uniform_(-bound, bound, generator=generator)
    span = copy.deepcopy(parent.span_head)
    candidate = RoutingCandidate(
        parent.binding,
        head,
        arm="B_stop_lock",
        seed=int(seed),
        span_head=span,
    )
    candidate.configure_trainable()
    optimizer = make_routing_optimizer(candidate)
    params = freeze_to_selection(candidate)
    return candidate, optimizer, params


def freeze_to_selection(candidate) -> list[torch.nn.Parameter]:
    """Binding and span pointer stay frozen: selection stage only, no span loss."""
    for parameter in candidate.binding.parameters():
        parameter.requires_grad_(False)
    for parameter in candidate.span_head.parameters():
        parameter.requires_grad_(False)
    for parameter in candidate.head.parameters():
        parameter.requires_grad_(True)
    return [parameter for parameter in candidate.head.parameters() if parameter.requires_grad]


def batch_ce(candidate, batch, tag: str):
    module = cmp()
    values = []
    for fixture in batch:
        prepared, context = module.prepared_pair(candidate, fixture, tag, "train")
        route_index, route_stage = route_labels(fixture)
        values.append(
            h.selection_loss(
                candidate,
                prepared,
                module.teacher_prefix(fixture),
                tuple(fixture["target_bytes"]),
                route_index,
                route_stage,
                context,
            )
        )
    return torch.stack(values).mean()


def train_run(
    candidate, optimizer, params, batches, first: int, tag: str, limit: int | None = None
):
    """Selection CE only; returns the per-update log and the next update index."""
    log = []
    update = first
    stop = len(batches) if limit is None else min(len(batches), first + limit)
    for batch in batches[first:stop]:
        loss = batch_ce(candidate, batch, f"{tag}-{update + 1}")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        update += 1
        log.append({"update": update, "ce": float(loss.detach())})
    return log, update


def _is_added(name: str) -> bool:
    return name.startswith("W_align") or name.startswith("offset_embed")


def gradient_probe(candidate, batch, tag: str) -> dict:
    """D16-8 (non-zero) plus the D16-12 identifiability reading.

    Two ratios are reported because they answer different questions:

    * ``log10_ratio``  - aggregate gradient norms.  The trunk owns 37,120
      parameters against 1,064 in the added block, so a size-blind comparison
      already contributes ``log10(sqrt(37120/1064)) = +0.77`` of the budget.
    * ``log10_rms_ratio`` - per-parameter RMS gradient.  This is the fair
      scale comparison and the one the gate uses.

    Both are measured at initialisation (before any Adam step) and again at
    step 300.  Adam rescales each parameter by its own second moment, so a
    small absolute gradient is not by itself fatal; the trained-state reading
    is what says whether the added block actually participates.
    """
    params = [parameter for parameter in candidate.head.parameters() if parameter.requires_grad]
    loss = batch_ce(candidate, batch, tag)
    loss.backward()
    tensors = dict(candidate.head.named_parameters())
    norms = {
        name: float(parameter.grad.detach().norm())
        for name, parameter in tensors.items()
        if parameter.grad is not None and parameter.grad.numel() > 0
    }
    added = [value for name, value in norms.items() if _is_added(name)]
    trunk = [value for name, value in norms.items() if not _is_added(name)]
    added_count = sum(tensors[name].numel() for name in norms if _is_added(name))
    trunk_count = sum(tensors[name].numel() for name in norms if not _is_added(name))
    added_norm = float(sum(value * value for value in added) ** 0.5) if added else 0.0
    trunk_norm = float(sum(value * value for value in trunk) ** 0.5) if trunk else 0.0
    ratio = (
        math.log10(added_norm / trunk_norm)
        if added_norm > 0.0 and trunk_norm > 0.0
        else (float("inf") if added_norm > 0.0 else float("-inf"))
    )
    added_rms = added_norm / math.sqrt(added_count) if added_count else 0.0
    trunk_rms = trunk_norm / math.sqrt(trunk_count) if trunk_count else 0.0
    rms_ratio = (
        math.log10(added_rms / trunk_rms)
        if added_rms > 0.0 and trunk_rms > 0.0
        else (float("inf") if added_rms > 0.0 else float("-inf"))
    )
    for parameter in params:
        parameter.grad = None
    return {
        "gradient": {name: value for name, value in norms.items()},
        "added_norm": added_norm,
        "trunk_norm": trunk_norm,
        "added_parameters": added_count,
        "trunk_parameters": trunk_count,
        "log10_ratio": ratio,
        "added_rms": added_rms,
        "trunk_rms": trunk_rms,
        "log10_rms_ratio": rms_ratio,
        "identifiable": abs(rms_ratio) <= 1.0 if math.isfinite(rms_ratio) else False,
        "meaning": (
            "D16-12 identifiability gate: the added block's per-parameter RMS gradient must "
            "be within one order of magnitude of the trunk, otherwise no structural claim"
        ),
    }


def need_fix_share(candidate, fixtures, tag: str) -> dict:
    module = cmp()
    wrong = 0
    buckets: dict[str, int] = {}
    for fixture in fixtures:
        prepared, context = module.prepared_pair(candidate, fixture, tag, "needfix")
        prefix = module.teacher_prefix(fixture)
        scores = h.object_start_scores(candidate, prepared, context, prefix)
        gold = int(fixture["selection_label"])
        chosen = h.scored_route(scores)
        record_count = len(fixture["record_keys"])
        key = "gold" if chosen == gold else ("null" if chosen >= record_count else "other_record")
        buckets[key] = buckets.get(key, 0) + 1
        wrong += int(chosen != gold)
    return {
        "examples": len(fixtures),
        "wrong": wrong,
        "share": wrong / len(fixtures) if fixtures else 0.0,
        "bucket": buckets,
        "gate": 0.20,
        "pass": (wrong / len(fixtures) if fixtures else 0.0) >= 0.20,
    }


def metric_value(row: dict, key: str) -> float:
    if key == "completed":
        return 1.0 if row["completed"] else 0.0
    if key == "route_miss":
        return 1.0 if row["bucket"] == "route_miss" else 0.0
    if key == "object_start":
        return 1.0 if row["object"]["start_ok"] else 0.0
    if key == "object_route":
        return 1.0 if row["object"]["route_ok"] else 0.0
    raise ValueError(key)


def rates(rows) -> dict:
    if not rows:
        return {"examples": 0}
    return {
        "examples": len(rows),
        "completed": statistics.fmean(metric_value(row, "completed") for row in rows),
        "route_miss": statistics.fmean(metric_value(row, "route_miss") for row in rows),
        "object_route": statistics.fmean(metric_value(row, "object_route") for row in rows),
        "object_start": statistics.fmean(metric_value(row, "object_start") for row in rows),
    }


def world_bootstrap(n_rows, o_rows, key: str, rounds: int, seed: int) -> dict:
    """Cluster bootstrap over the 12 worlds; delta is N minus O."""
    if len(n_rows) != len(o_rows):
        raise RuntimeError("paired comparison needs equal-length arms")
    n_pairs = sorted(
        (
            (
                n_rows[index]["world"],
                metric_value(n_rows[index], key),
                metric_value(o_rows[index], key),
            )
            for index in range(len(n_rows))
        ),
        key=lambda item: (item[0], item[1], item[2]),
    )
    by_world: dict[str, list[tuple[float, float]]] = {}
    for world, left, right in n_pairs:
        by_world.setdefault(world, []).append((left, right))
    worlds = sorted(by_world)
    generator = random.Random(seed)
    deltas = []
    for _round in range(rounds):
        picks = [worlds[generator.randrange(len(worlds))] for _index in worlds]
        left = [value for world in picks for value, _right in by_world[world]]
        right = [value for world in picks for _left, value in by_world[world]]
        deltas.append(statistics.fmean(left) - statistics.fmean(right))
    deltas.sort()
    alpha = 0.025
    low = deltas[max(0, int(round(alpha * (rounds - 1))))]
    high = deltas[min(rounds - 1, int(round((1 - alpha) * (rounds - 1))))]
    all_n = [value for world in worlds for value, _right in by_world[world]]
    all_o = [value for world in worlds for _left, value in by_world[world]]
    return {
        "metric": f"{key} (N - O)",
        "worlds": len(worlds),
        "count": len(all_n),
        "n_rate": statistics.fmean(all_n),
        "o_rate": statistics.fmean(all_o),
        "delta": statistics.fmean(all_n) - statistics.fmean(all_o),
        "ci_low": low,
        "ci_high": high,
        "excludes_zero": low > 0.0 or high < 0.0,
        "n_better": low > 0.0,
        "rounds": rounds,
    }


def checkpoint_preflight(candidate, optimizer, fixture) -> dict:
    module = cmp()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prepared, _context = module.prepared_pair(candidate, fixture, "19.2j-dryrun", "save")
    before = parameter_tensors(candidate)
    info = save_routing_bundle(
        PREFLIGHT_BUNDLE,
        candidate,
        optimizer,
        0,
        {"stage": "N19.2j-preflight", "seed": int(candidate.seed)},
        candidate_identity(candidate),
        prepared.state,
    )
    reloaded, reload_optimizer, _state, reload_index, _cursor = load_routing_bundle(
        PREFLIGHT_BUNDLE
    )
    after = parameter_tensors(reloaded)
    diff = max(float((before[name] - after[name]).abs().max()) for name in before)
    moments = 0
    saved = reload_optimizer.state_dict()["state"]
    for group in optimizer.state_dict()["state"].values():
        index = int(group.get("step", torch.tensor(0)))
        moments = max(moments, index)
    PREFLIGHT_BUNDLE.unlink(missing_ok=True)
    return {
        "round_trip_ok": diff == 0.0,
        "max_abs_diff": diff,
        "optimizer_states": len(saved),
        "update_index": reload_index,
        "moments_step": moments,
        "bytes": info["bytes"],
        "sha256": info["sha256"],
    }


def drift_rejection(candidate, fixture) -> dict:
    module = cmp()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prepared, _context = module.prepared_pair(candidate, fixture, "19.2j-drift", "save")
    save_routing_bundle(
        CORRUPT_BUNDLE,
        candidate,
        make_routing_optimizer(candidate),
        0,
        {"stage": "N19.2j-drift"},
        candidate_identity(candidate),
        prepared.state,
    )
    payload = torch.load(CORRUPT_BUNDLE, map_location="cpu", weights_only=True)
    payload["head_parameters"].pop("W_align.weight")
    torch.save(payload, CORRUPT_BUNDLE)
    rejected = False
    message = ""
    try:
        load_routing_bundle(CORRUPT_BUNDLE)
    except Exception as error:  # noqa: BLE001 - any refusal is the pass condition
        rejected = True
        message = f"{type(error).__name__}: {str(error)[:160]}"
    CORRUPT_BUNDLE.unlink(missing_ok=True)
    return {
        "corruption": "head_parameters.W_align.weight removed",
        "rejected": rejected,
        "message": message,
    }


def identity_rescue(candidate, fixtures, tag: str) -> dict:
    module = cmp()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    candidate.configure_trainable()
    prepared, _context = module.prepared_pair(candidate, fixtures[0], "19.2j-rescue-save", "save")
    save_routing_bundle(
        PREFLIGHT_BUNDLE,
        candidate,
        make_routing_optimizer(candidate),
        0,
        {"stage": "N19.2j-rescue", "seed": int(candidate.seed)},
        candidate_identity(candidate),
        prepared.state,
    )
    reloaded = load_routing_bundle(PREFLIGHT_BUNDLE)[0]
    worst = 0.0
    agree = 0
    for index, fixture in enumerate(fixtures):
        left = h.teacher_forced(candidate, fixture, f"19.2j-rescue-{tag}-{index}")
        right = h.teacher_forced(reloaded, fixture, f"19.2j-rescue-{tag}-r{index}")
        diff = max(float((a - b).abs().max()) for a, b in zip(left, right, strict=True))
        worst = max(worst, diff)
        agree += int(diff <= 0.0)
    PREFLIGHT_BUNDLE.unlink(missing_ok=True)
    return {"examples": len(fixtures), "agree": agree, "max_abs_diff": worst}


def gates_view(o_cell: dict, n_cell: dict) -> dict:
    frame_o = o_cell["sentences"]["frame_model_generation"]
    frame_n = n_cell["sentences"]["frame_model_generation"]
    return {
        "new_route_hit_object": {
            "o": o_cell["aggregate"]["route_hit_object"],
            "n": n_cell["aggregate"]["route_hit_object"],
            "threshold": ROUTE_HIT_GATE,
            "pass": n_cell["aggregate"]["route_hit_object"] >= ROUTE_HIT_GATE,
        },
        "new_route_miss": {
            "o": o_cell["rates"]["route_miss"],
            "n": n_cell["rates"]["route_miss"],
            "threshold": ROUTE_MISS_GATE,
            "pass": n_cell["rates"]["route_miss"] <= ROUTE_MISS_GATE,
        },
        "swap_hard_argmax_new": {
            "o": o_cell["swap"]["pairs_argmax_changed"],
            "n": n_cell["swap"]["pairs_argmax_changed"],
            "of": n_cell["swap"]["pairs"],
            "threshold": f">={ARGMAX_GATE}/20",
            "pass": n_cell["swap"]["pairs_argmax_changed"] >= ARGMAX_GATE,
        },
        "old_route_hit_no_regression": {
            "o": o_cell["aggregate_old"]["route_hit_object"],
            "n": n_cell["aggregate_old"]["route_hit_object"],
            "threshold": f">= O - {OLD_ROUTE_SLACK}",
            "pass": n_cell["aggregate_old"]["route_hit_object"]
            >= o_cell["aggregate_old"]["route_hit_object"] - OLD_ROUTE_SLACK,
        },
        "new_composite_start_hit_object": {
            "o": o_cell["aggregate"]["composite_start_hit_object"],
            "n": n_cell["aggregate"]["composite_start_hit_object"],
            "threshold": OBJECT_START_GATE,
            "pass": n_cell["aggregate"]["composite_start_hit_object"] >= OBJECT_START_GATE,
        },
        "new_conditional_byte_object": {
            "o": o_cell["aggregate"]["conditional_byte_object"],
            "n": n_cell["aggregate"]["conditional_byte_object"],
            "threshold": CONDITIONAL_GATE,
            "pass": n_cell["aggregate"]["conditional_byte_object"] >= CONDITIONAL_GATE,
        },
        "frozen_path_eos_frame": {
            "o": frame_o,
            "n": frame_n,
            "threshold": "no regression",
            "pass": frame_n <= frame_o,
        },
    }


def branch_view(gates: dict, delta: dict, identifiable: bool) -> dict:
    all_pass = all(gate["pass"] for gate in gates.values())
    old_regressed = not gates["old_route_hit_no_regression"]["pass"]
    n_better = bool(delta.get("completed", {}).get("n_better", False))
    equivalent = not delta.get("completed", {}).get("excludes_zero", False)
    if not identifiable:
        branch = "not_identifiable_no_structural_claim"
    elif old_regressed:
        branch = "n_arm_sacrifices_old_capability"
    elif n_better:
        branch = "scoring_form_is_the_object_side_cause"
    elif equivalent:
        branch = "scoring_form_is_not_the_bottleneck"
    else:
        branch = "n_arm_worse"
    return {
        "branch": branch,
        "all_gates_pass": all_pass,
        "old_regressed": old_regressed,
        "n_better": n_better,
        "equivalent": equivalent,
        "identifiable": identifiable,
        "rule": (
            "D16-12: only a from-scratch, single-variable A/B may speak about architecture; "
            "a world-bootstrap lower bound above 0 for N-minus-O means the scoring form is the cause"
        ),
    }


def zero_cost_check() -> dict:
    """Owner item 1: is 19.2i a new objective or the one already in the loop?

    ``h.selection_loss`` (running since 19.2e) is a listwise cross-entropy over
    R + 1 rows (12 records + one null row) taken at every content step whose
    ``route_label >= 0`` (18 steps per example), mean within stage then mean
    across stages, temperature 1.  19.2i proposed ``CE(softmax(scores[:R]/0.1),
    gold_record)`` summed over the two content starts.  Same family, same gold
    target, and both of 19.2i's differences (lower temperature, fewer steps)
    are strictly narrower.  Therefore ``duplicate_objective``: retired.
    """
    return {
        "item": "19.2i 12-way listwise CE vs the selection CE already trained against",
        "existing": {
            "where": "taiji/selection_routing.py :: head.selection_loss, used by every 19.x train script",
            "formula": "F.cross_entropy(scores.unsqueeze(0), [route_label])",
            "input_tensor": "score_rows(goal, keys, s_t, prev_emb) -> R+1 rows (12 records + null)",
            "reduction": "mean over every content step in a stage, then mean across stages (18 steps)",
            "temperature": 1.0,
            "gold_objective": "record-level listwise NLL, null row competes",
        },
        "proposal_19_2i": {
            "formula": "CE(softmax(scores[:R]/tau), gold_record)",
            "input_tensor": "same tensor, first R rows (null dropped)",
            "reduction": "sum over the 2 content starts only",
            "temperature": 0.1,
            "gold_objective": "same record-level NLL, null excluded",
        },
        "differences": [
            "temperature 1 -> 0.1 (sharper, not new supervision)",
            "coverage 18 content steps -> 2 starts (weaker, not new supervision)",
        ],
        "verdict": "duplicate_objective (D16-11): same objective family, both deltas narrower",
        "action": "19.2i retired; 1,089 CPU not spent; architecture question routed to 19.2j",
    }


def preregistration_block() -> dict:
    return {
        "written_before_training": True,
        "variable_table": {
            "shared_bit_for_bit": [
                "head.W_hidden.weight[:, :576]",
                "head.W_hidden.bias",
                "head.w_out.weight",
                "head.k_null",
                "parent.binding (frozen)",
                "parent.span_head (frozen, no span loss)",
            ],
            "only_difference": "N adds W_align(32x32) + offset_embed(5x8) + 4 extra W_hidden columns",
            "same": [
                "init seed 28031 / 28032",
                "data order (premise.ORDER_SEED schedule over 640 train fixtures)",
                "batch 8, LR 1e-3, clip 1.0, 300 updates, selection CE only",
                "dev 192 (96 old + 96 new description), swap pairs 20",
            ],
            "design_error_rule": "any other difference between O and N voids the package",
        },
        "gates": {
            "new_route_hit_object": f">= {ROUTE_HIT_GATE}",
            "new_route_miss": f"<= {ROUTE_MISS_GATE}",
            "swap_hard_argmax_new": f">= {ARGMAX_GATE}/20",
            "old_route_hit_object": f">= O - {OLD_ROUTE_SLACK}",
            "new_composite_start_hit_object": f">= {OBJECT_START_GATE}",
            "new_conditional_byte_object": f">= {CONDITIONAL_GATE}",
            "frozen_path_eos_frame": "no regression",
            "identifiability": "|log10(added per-parameter RMS grad / trunk RMS grad)| <= 1 at step 300",
        },
        "attribution": (
            "per-seed and pooled N-O deltas with a 12-world cluster bootstrap (2000 rounds); "
            "the sign is N minus O so that a lower bound above 0 means N wins"
        ),
        "branches": {
            "n_wins": "bootstrap lower bound > 0 -> scoring form is the object-side cause; wait for owner",
            "equivalent": "interval contains 0 -> scoring form is not the bottleneck; go to byte-fingerprint oracle probe",
            "n_loses": "record the negative, retire the form, no dose escalation",
            "not_identifiable": "no structural claim at all",
        },
    }


def budget_block(unit_cpu: float | None = None) -> dict:
    items = [{"item": name, "cpu": value} for name, value in BUDGET_ITEMS]
    if unit_cpu is not None:
        items[0]["cpu"] = ADAM_CAP * unit_cpu
        items[0]["item"] = (
            "1,200 from-scratch selection updates, re-priced at the measured "
            f"{unit_cpu:.3f} CPU/update"
        )
    ideal = sum(item["cpu"] for item in items)
    return {
        "functional_items": items,
        "functional_total_cpu": ideal,
        "safety_factor": SAFETY_FACTOR,
        "budget_cpu": ideal * SAFETY_FACTOR,
        "cap_cpu": BUDGET_CAP,
        "within_cap": ideal * SAFETY_FACTOR <= BUDGET_CAP,
        "priced_on_measurement": unit_cpu is not None,
    }


def cell_view(snapshot: dict) -> dict:
    rows = snapshot["rows_new96"]
    return {
        "aggregate": snapshot["new_description_96"],
        "aggregate_old": snapshot["old_description_96"],
        "swap": snapshot["swap_new"],
        "scores": snapshot["scores_new96"],
        "rates": rates(rows),
        "sentences": {
            "frame_model_generation": snapshot["new_description_96"]["frame_split"][
                "model_generation"
            ]
        },
        "completed_examples": sum(1 for row in rows if row["completed"]),
        "rows": rows,
    }


def main() -> int:
    start_cpu = time.process_time()
    start_wall = time.perf_counter()
    payload = premise.build_payload()
    train = premise.fixtures(payload, "train")
    dev = premise.fixtures(payload, "dev")
    new_fixtures = tuple(fixture for fixture in dev if fixture["flags"]["new_description"])
    if len(dev) != 192 or len(new_fixtures) != 96:
        raise RuntimeError("dev stratification drift")
    _easy_pairs, new_pairs = wall.pair_fixtures(dev)
    if len(new_pairs) != PAIRS_PER_STRATUM:
        raise RuntimeError("swap pair drift")
    batches = h.schedule(train, UPDATES)

    parent = load_routing_bundle(PARENT)[0]
    parent_nan = sum(int(torch.isnan(value).sum()) for value in parameter_tensors(parent).values())
    if parent_nan:
        raise RuntimeError("parent carries NaN parameters (D16-6)")
    parent_identity = candidate_identity(parent)

    runs = {}
    for seed in SEEDS:
        for arm in ("O", "N"):
            candidate, optimizer, params = build_run(parent, seed, arm)
            runs[(seed, arm)] = {
                "candidate": candidate,
                "optimizer": optimizer,
                "params": params,
                "identity": candidate_identity(candidate),
            }

    # ---- D16-8 / D16-12 gates, before any Adam step ----------------------
    activity = {
        arm: {
            "gradient": gradient_probe(
                runs[(SEEDS[0], arm)]["candidate"], batches[0], f"19.2j-activity-{arm}"
            ),
            "need_fix": need_fix_share(
                runs[(SEEDS[0], arm)]["candidate"], new_fixtures, f"19.2j-needfix-{arm}"
            ),
        }
        for arm in ("N", "O")
    }
    n_probe = activity["N"]["gradient"]
    activity["N"]["identifiable"] = n_probe["identifiable"]
    activity["N"]["gradient"]["non_zero"] = all(
        value > 0.0 for value in n_probe["gradient"].values()
    )
    gate_pass = activity["N"]["gradient"]["non_zero"] and activity["N"]["need_fix"]["pass"]

    if not gate_pass:
        cpu = time.process_time() - start_cpu
        report = {
            "schema": "taiji-package19-2j-from-scratch-selection-ab-v1",
            "step": "N19.2j from-scratch selection A/B",
            "date": "2026-11-04",
            "zero_cost_check": zero_cost_check(),
            "preregistration": preregistration_block(),
            "activity_precheck": activity,
            "decision": {
                "branch": "activity_precheck_failed",
                "rule": "D16-8: no update is allowed when the added block carries no gradient",
            },
            "budget": {**budget_block(), "spent_cpu": cpu},
            "cost": {"cpu_seconds": cpu, "adam_updates": 0, "confirm_reads": 0},
        }
        h.write_json(REPORT, report)
        h.write_json(
            MANIFEST,
            {
                "schema": "taiji-package19-2j-manifest-v1",
                "step": "N19.2j",
                "date": "2026-11-04",
                "report": str(REPORT.relative_to(ROOT)),
                "branch": "activity_precheck_failed",
                "adam_updates": 0,
                "cost_cpu": cpu,
            },
        )
        print(json.dumps({"branch": "activity_precheck_failed", "cpu": round(cpu, 1)}))
        return 0

    # ---- checkpoint round-trip before training ---------------------------
    preflight = checkpoint_preflight(
        runs[(SEEDS[0], "O")]["candidate"],
        runs[(SEEDS[0], "O")]["optimizer"],
        dev[0],
    )
    preflight_n = checkpoint_preflight(
        runs[(SEEDS[0], "N")]["candidate"],
        runs[(SEEDS[0], "N")]["optimizer"],
        dev[0],
    )
    if not (preflight["round_trip_ok"] and preflight_n["round_trip_ok"]):
        raise RuntimeError("checkpoint dry-run did not round-trip; refusing to train")
    drift = drift_rejection(runs[(SEEDS[0], "N")]["candidate"], dev[0])

    # ---- measured-price guard, then the four trajectories ----------------
    logs: dict[tuple[int, str], list] = {}
    warm_cpu = time.process_time()
    warm_log, next_update = train_run(
        runs[(SEEDS[0], "O")]["candidate"],
        runs[(SEEDS[0], "O")]["optimizer"],
        runs[(SEEDS[0], "O")]["params"],
        batches,
        0,
        "19.2j-O-28031",
        limit=WARMUP_UPDATES,
    )
    warm_elapsed = time.process_time() - warm_cpu
    unit_cpu = warm_elapsed / max(1, next_update)
    projected = (unit_cpu * ADAM_CAP + ANCILLARY_CPU) * SAFETY_FACTOR
    price_guard = {
        "unit_cpu_per_update": unit_cpu,
        "updates_measured": next_update,
        "projected_adam_cpu": unit_cpu * ADAM_CAP,
        "ancillary_cpu": ANCILLARY_CPU,
        "projected_total_cpu": projected,
        "cap_cpu": BUDGET_CAP,
        "within_cap": projected <= BUDGET_CAP,
        "owner_estimate_cpu_per_update": 4.0,
        "owner_estimate_total_cpu": (4.0 * ADAM_CAP + ANCILLARY_CPU) * SAFETY_FACTOR,
        "basis": (
            "measured marginal CPU on the first five updates of seed 28031 / O arm, plus "
            "500 CPU of evaluation/probe items, then x3.3"
        ),
    }
    if projected > BUDGET_CAP:
        cpu = time.process_time() - start_cpu
        report = {
            "schema": "taiji-package19-2j-from-scratch-selection-ab-v1",
            "step": "N19.2j from-scratch selection A/B",
            "date": "2026-11-04",
            "zero_cost_check": zero_cost_check(),
            "preregistration": preregistration_block(),
            "activity_precheck": activity,
            "price_guard": price_guard,
            "decision": {
                "branch": "budget_extrapolation_exceeds_cap",
                "rule": "owner rule: if the measured extrapolation exceeds 12,000 CPU, report first and do not run",
            },
            "notes": [
                "only the five price-measurement updates ran; the 1,200-update A/B was not started",
                "the D16-12 identifiability gate is pre-registered at step 300 and was never reached; "
                "the initialisation reading (log10 per-parameter RMS ratio -1.425) is reported but "
                "licenses no structural claim either way",
            ],
            "budget": {**budget_block(unit_cpu), "spent_cpu": cpu},
            "cost": {"cpu_seconds": cpu, "adam_updates": next_update, "confirm_reads": 0},
        }
        h.write_json(REPORT, report)
        h.write_json(
            MANIFEST,
            {
                "schema": "taiji-package19-2j-manifest-v1",
                "step": "N19.2j",
                "date": "2026-11-04",
                "report": str(REPORT.relative_to(ROOT)),
                "branch": "budget_extrapolation_exceeds_cap",
                "adam_updates": next_update,
                "cost_cpu": cpu,
            },
        )
        print(json.dumps({"branch": "budget_extrapolation_exceeds_cap", "cpu": round(cpu, 1)}))
        return 0

    logs[(SEEDS[0], "O")] = warm_log
    for seed in SEEDS:
        for arm in ("O", "N"):
            if (seed, arm) == (SEEDS[0], "O"):
                continue
            run = runs[(seed, arm)]
            logs[(seed, arm)], _end = train_run(
                run["candidate"],
                run["optimizer"],
                run["params"],
                batches,
                0,
                f"19.2j-{arm}-{seed}",
            )

    # ---- identifiability at step 300 (the reading the D16-12 gate uses) --
    final_probe = gradient_probe(runs[(SEEDS[0], "N")]["candidate"], batches[0], "19.2j-final-N")
    activity["N"]["gradient_at_step300"] = final_probe

    # ---- evaluation at step 300 --------------------------------------------
    cells = {}
    bundles = {}
    for seed in SEEDS:
        for arm in ("O", "N"):
            key = (seed, arm)
            run = runs[key]
            snapshot = h.full_snapshot(
                run["candidate"], dev, new_fixtures, new_pairs, f"19.2j-{arm}-{seed}"
            )
            cells[key] = cell_view(snapshot)
            state = (
                cmp()
                .prepared_pair(run["candidate"], dev[0], f"19.2j-save-{arm}-{seed}", "save")[0]
                .state
            )
            final = OUTPUT_DIR / f"{seed}-{ARMS[arm]}-step{UPDATES}.pt"
            bundles[f"{seed}-{arm}"] = save_routing_bundle(
                final,
                run["candidate"],
                run["optimizer"],
                UPDATES,
                {"stage": "N19.2j", "seed": seed, "arm": ARMS[arm]},
                run["identity"],
                state,
            )

    # ---- O vs N, per seed and pooled --------------------------------------
    delta = {}
    for seed in SEEDS:
        n_rows = cells[(seed, "N")]["rows"]
        o_rows = cells[(seed, "O")]["rows"]
        delta[str(seed)] = {
            key: world_bootstrap(n_rows, o_rows, key, BOOTSTRAP_ROUNDS, BOOTSTRAP_SEED)
            for key in ("completed", "route_miss", "object_route", "object_start")
        }
    pooled_n = cells[(SEEDS[0], "N")]["rows"] + cells[(SEEDS[1], "N")]["rows"]
    pooled_o = cells[(SEEDS[0], "O")]["rows"] + cells[(SEEDS[1], "O")]["rows"]
    pooled = {
        key: world_bootstrap(pooled_n, pooled_o, key, BOOTSTRAP_ROUNDS, BOOTSTRAP_SEED)
        for key in ("completed", "route_miss", "object_route", "object_start")
    }
    delta["pooled"] = pooled

    gates = {}
    for seed in SEEDS:
        gates[str(seed)] = gates_view(cells[(seed, "O")], cells[(seed, "N")])
    identifiable = bool(final_probe["identifiable"])
    decision = branch_view(
        {name: gate for name, gate in gates[str(SEEDS[0])].items()},
        {"completed": pooled["completed"]},
        identifiable,
    )
    decision["per_seed_branches"] = {
        str(seed): branch_view(
            gates[str(seed)], {"completed": delta[str(seed)]["completed"]}, identifiable
        )["branch"]
        for seed in SEEDS
    }

    identity = {
        "parent": parent_identity["candidate_sha_v2"],
        "O": runs[(SEEDS[0], "O")]["identity"]["candidate_sha_v2"],
        "N": runs[(SEEDS[0], "N")]["identity"]["candidate_sha_v2"],
        "changed": runs[(SEEDS[0], "N")]["identity"]["candidate_sha_v2"]
        != runs[(SEEDS[0], "O")]["identity"]["candidate_sha_v2"],
        "rescue": identity_rescue(
            runs[(SEEDS[0], "N")]["candidate"], tuple(dev[:RESCUE_EXAMPLES]), "N"
        ),
    }

    total_updates = sum(len(log) for log in logs.values())
    cpu = time.process_time() - start_cpu
    wall_time = time.perf_counter() - start_wall
    budget = {
        **budget_block(unit_cpu),
        "spent_cpu": cpu,
        "spent_share": cpu / BUDGET_CAP,
        "over_cap": cpu > BUDGET_CAP,
    }
    report = {
        "schema": "taiji-package19-2j-from-scratch-selection-ab-v1",
        "step": "N19.2j from-scratch selection A/B",
        "date": "2026-11-04",
        "zero_cost_check": zero_cost_check(),
        "preregistration": preregistration_block(),
        "design": {
            "question": "is the scoring form the object-side cause, tested from zero instead of by micro-adaptation",
            "oe_arm": "O = pooled description/record dot-product form (RoutingConfig(alignment=False))",
            "n_arm": "N = 19.2h sequence-surface form, re-initialised at trunk scale",
            "held_equal": [
                "parent binding graph (frozen, no copy/joint training)",
                "parent span pointer (frozen, no span supervision)",
                "initialisation seed (shared tensors copied bit for bit from O to N)",
                "data order (one schedule over 640 training fixtures)",
                "batch 8 / LR 1e-3 / clip 1.0 / 300 updates / selection CE only",
            ],
            "only_difference": "scoring form (+1,320 parameters in N)",
            "initialisation": {
                "W_align": "uniform +-1/sqrt(32) = +-0.177 (same rule the head uses for 32-d blocks)",
                "offset_embed": "uniform +-1/sqrt(64) = +-0.125 (trunk scale; 19.2h used +-0.01)",
                "trunk": "inherited from the O head, bit for bit",
            },
            "seeds": list(SEEDS),
            "updates_per_run": UPDATES,
            "adam_total": ADAM_CAP,
        },
        "activity_precheck": activity,
        "price_guard": price_guard,
        "checkpoint_preflight": {"O": preflight, "N": preflight_n, "drift": drift},
        "identities": identity,
        "training": {
            "updates": total_updates,
            "first": {key: log[0] for key, log in logs.items()},
            "last": {key: log[-1] for key, log in logs.items()},
            "curve": {
                f"{seed}-{arm}": [entry["ce"] for entry in log[::50]]
                for (seed, arm), log in logs.items()
            },
        },
        "cells": {
            f"{seed}-{arm}": {
                "aggregate": cells[(seed, arm)]["aggregate"],
                "aggregate_old": cells[(seed, arm)]["aggregate_old"],
                "swap": cells[(seed, arm)]["swap"],
                "rates": cells[(seed, arm)]["rates"],
                "completed_examples": cells[(seed, arm)]["completed_examples"],
                "frame_model_generation": cells[(seed, arm)]["sentences"]["frame_model_generation"],
            }
            for seed in SEEDS
            for arm in ("O", "N")
        },
        "delta": delta,
        "gates": gates,
        "decision": decision,
        "saved_bundles": bundles,
        "budget": budget,
        "cost": {
            "cpu_seconds": cpu,
            "wall_seconds": wall_time,
            "adam_updates": total_updates,
            "runs": len(logs),
            "confirm_reads": 0,
        },
    }
    h.write_json(REPORT, report)
    h.write_json(
        MANIFEST,
        {
            "schema": "taiji-package19-2j-manifest-v1",
            "step": "N19.2j",
            "date": "2026-11-04",
            "report": str(REPORT.relative_to(ROOT)),
            "parent": str(PARENT.relative_to(ROOT)),
            "parent_sha": parent_identity["candidate_sha_v2"],
            "identities": {"O": identity["O"], "N": identity["N"], "changed": identity["changed"]},
            "branch": decision["branch"],
            "gates": {name: gate["pass"] for name, gate in gates[str(SEEDS[0])].items()},
            "adam_updates": total_updates,
            "cost_cpu": cpu,
            "confirm_reads": 0,
        },
    )
    print(
        json.dumps(
            {
                "branch": decision["branch"],
                "cpu": round(cpu, 1),
                "pooled_completed": pooled["completed"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
