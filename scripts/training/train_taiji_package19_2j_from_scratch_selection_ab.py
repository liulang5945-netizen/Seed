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
import os
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
import probe_taiji_content_transport_18_1 as probe_mod  # noqa: E402
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

# Owner 2026-11-05 item 4: the crash was at the end of the pricing segment, so a
# dry run has to reach that point and then continue with real updates plus a
# probe call.  Six engineering Adam total (3 price + 3 continue), <= 200 CPU.
DRY_WARMUP = 3
DRY_CONTINUE = 3
DRY_REPORT = ROOT / "reports/taiji_package19_2j_dry_run_pricing_20261105.json"

OUTPUT_DIR = ROOT / "output/taiji_package19_2j_from_scratch_selection_ab"
PREFLIGHT_BUNDLE = OUTPUT_DIR / "preflight-step0.pt"
CORRUPT_BUNDLE = OUTPUT_DIR / "drift-probe.pt"
REPORT = ROOT / "reports/taiji_package19_2j_from_scratch_selection_ab_20261104.json"
MANIFEST = ROOT / "plans/manifests/taiji_package19_2j_from_scratch_selection_ab_20261104.json"
PROGRESS_LOG = OUTPUT_DIR / "progress-19_2j.jsonl"
REBUILD_PROGRESS = ROOT / "reports/taiji_package19_2j_rebuild_progress_20261105.jsonl"
TRAINING_WALL_SECONDS = 4961.0
REBUILD_CPU_CAP = 500.0

# Owner mid-run gate (2026-11-04, zero design change): read the added-block
# gradient every 50 selection updates, and at step 100 stop the package if the
# per-parameter RMS ratio is still <= -1.  Old-description route_hit is monitored
# on the same points; any drop below the run's best-so-far by more than 0.02
# stops the package under "must not sacrifice old capability".
MID_RUN_STEPS = (50, 100, 150, 200, 250)
IDENT_STEPS = (100, 200)
IDENT_STOP_STEP = 100
IDENT_STOP_THRESHOLD = -1.0

# Owner budget rule: functional item list x 3.3, cap 13,000 CPU (raised from
# 12,000 on 2026-11-04).  The update item is re-priced with the measured
# marginal CPU after the five-update warm-up; the listed 4,800 is the owner's own
# 4 CPU/update estimate, kept as the upper bound.
BUDGET_ITEMS = (
    (
        "1,200 from-scratch selection updates (2 seeds x 2 arms x 300), batch 8, CE only"
        " (owner estimate 4 CPU/update; re-priced by the warm-up measurement)",
        4800.0,
    ),
    (
        "20 mid-run route readouts (5 points x 4 runs, 192 examples each, re-priced by"
        " measurement)",
        200.0,
    ),
    ("four full 192-example decode passes at step 300 (one per run)", 308.0),
    ("four swap summaries on 20 new-description pairs", 40.0),
    ("identity rescue 8 examples + checkpoint dry-run + N-arm drift rejection", 40.0),
    ("gradient probes (init, 100, 200, 300), payload, parent health, report", 40.0),
)
SAFETY_FACTOR = 3.3
BUDGET_CAP = 13000.0
FIXED_CPU = 428.0
MARGINAL_CPU_PER_UPDATE = 3.7
MID_RUN_READOUTS = len(MID_RUN_STEPS) * 4

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


def build_readout_cache(candidate, fixtures) -> dict:
    """Prepare the 192 dev fixtures once; the binding graph is frozen for all runs.

    ``prepared_pair`` only touches ``candidate.binding`` (identical across all four
    runs) and a per-fixture subject tag, so one cache serves every checkpoint of
    every run.  This is what keeps the mid-run gate near-zero cost: a readout is
    then 3 routing steps + one score per example, with no re-encoding.
    """
    module = cmp()
    cache = {}
    for fixture in fixtures:
        prepared, context = module.prepared_pair(candidate, fixture, "19.2j-readout", "mid")
        cache[id(fixture)] = (prepared, context)
    return cache


def route_readout(candidate, cache: dict, new_fixtures, old_fixtures) -> dict:
    """Owner mid-run item 2: route-only selection readings, no extra forward pass.

    Same object-start scoring path as ``need_fix_share`` (one score per example,
    no generation) on the unchanged 192-example dev set: 96 old + 96 new.
    """
    module = cmp()

    def hit_count(fixtures) -> int:
        good = 0
        for fixture in fixtures:
            prepared, context = cache[id(fixture)]
            scores = h.object_start_scores(
                candidate, prepared, context, module.teacher_prefix(fixture)
            )
            good += int(h.scored_route(scores) == int(fixture["selection_label"]))
        return good

    new_hits = hit_count(new_fixtures)
    old_hits = hit_count(old_fixtures)
    return {
        "new_route_hit": new_hits / len(new_fixtures),
        "new_route_miss": 1.0 - new_hits / len(new_fixtures),
        "old_route_hit": old_hits / len(old_fixtures),
        "new_examples": len(new_fixtures),
        "old_examples": len(old_fixtures),
    }


def train_run(
    candidate,
    optimizer,
    params,
    batches,
    first: int,
    tag: str,
    limit: int | None = None,
    probe=None,
):
    """Selection CE only; returns the log, the next update index and a stop flag.

    ``probe(update)`` is called after every update and may return True to stop
    the trajectory early (owner mid-run gate).
    """
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
        if probe is not None and probe(update):
            return log, update, True
    return log, update, False


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
    n_wins = sum(1 for value in deltas if value > 0.0)
    n_losses = sum(1 for value in deltas if value < 0.0)
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
        "n_wins": n_wins,
        "n_losses": n_losses,
        "n_ties": rounds - n_wins - n_losses,
        "win_share": n_wins / rounds,
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
        "mid_run_gate": {
            "added": "owner 2026-11-04, zero design change",
            "readout_points": list(MID_RUN_STEPS),
            "route_metrics": "new-description route_hit / route_miss, old-description route_hit (192 examples)",
            "identifiability_points": list(IDENT_STEPS),
            "stop_rules": [
                f"step {IDENT_STOP_STEP} per-parameter RMS ratio <= {IDENT_STOP_THRESHOLD} -> stop, report still_not_identifiable",
                f"old-description route_hit drops more than {OLD_ROUTE_SLACK} below the run's best so far -> stop",
            ],
        },
        "branches": {
            "n_wins": "bootstrap lower bound > 0 -> scoring form is the object-side cause; wait for owner",
            "equivalent": "interval contains 0 -> scoring form is not the bottleneck; go to byte-fingerprint oracle probe",
            "n_loses": "record the negative, retire the form, no dose escalation",
            "not_identifiable": "no structural claim at all",
        },
    }


def budget_block(unit_cpu: float | None = None, readout_cpu: float | None = None) -> dict:
    items = [{"item": name, "cpu": value} for name, value in BUDGET_ITEMS]
    if unit_cpu is not None:
        items[0]["cpu"] = ADAM_CAP * unit_cpu
        items[0]["item"] = (
            "1,200 from-scratch selection updates, re-priced at the measured "
            f"{unit_cpu:.3f} CPU/update"
        )
    if readout_cpu is not None:
        items[1]["cpu"] = MID_RUN_READOUTS * readout_cpu
        items[1]["item"] = (
            f"{MID_RUN_READOUTS} mid-run route readouts (192 examples each), re-priced at "
            f"the measured {readout_cpu:.3f} CPU/readout"
        )
    ideal = sum(item["cpu"] for item in items)
    return {
        "functional_items": items,
        "functional_total_cpu": ideal,
        "safety_factor": SAFETY_FACTOR,
        "budget_cpu": ideal * SAFETY_FACTOR,
        "cap_cpu": BUDGET_CAP,
        "within_cap": ideal * SAFETY_FACTOR <= BUDGET_CAP,
        "priced_on_measurement": unit_cpu is not None and readout_cpu is not None,
    }


def operational_waste_block() -> dict:
    """Owner 2026-11-05 item 1: idle ``python -m black`` processes, measured then killed.

    ``python -m black`` spawns a process pool that never exits in this managed
    environment; the CPU column kept climbing while it sat idle.  Every main and
    child process was enumerated, its accumulated CPU read, then terminated.
    A verification pass afterwards returned zero remaining black processes.
    """
    per_process = [
        (1876, 0.6, "child"),
        (2540, 4454.8, "main"),
        (3364, 0.6, "child"),
        (3952, 8726.9, "main"),
        (4876, 3359.6, "main"),
        (6924, 1196.1, "main"),
        (7484, 2497.2, "main"),
        (8232, 0.5, "child"),
        (9816, 11378.6, "main"),
        (11356, 0.5, "child"),
        (11768, 0.7, "child"),
        (13580, 17333.9, "main"),
        (14436, 1.0, "child"),
        (14852, 0.6, "child"),
        (15428, 3231.3, "main"),
        (15516, 11634.9, "main"),
        (15716, 0.5, "child"),
        (16068, 4161.5, "main"),
        (16388, 0.6, "child"),
        (16764, 2519.5, "main"),
        (17220, 3459.7, "main"),
        (18460, 0.6, "child"),
        (18948, 0.6, "child"),
        (22380, 3024.8, "main"),
        (23920, 0.6, "child"),
        (26056, 0.8, "child"),
        (28468, 0.5, "child"),
        (31432, 0.7, "child"),
        (32476, 0.7, "child"),
        (33208, 4267.7, "main"),
        (33540, 0.5, "child"),
        (35060, 0.8, "child"),
        (35164, 0.6, "child"),
        (35832, 0.6, "child"),
        (37260, 970.5, "main"),
        (38588, 0.7, "child"),
        (38888, 0.7, "child"),
        (39092, 1643.7, "main"),
        (39124, 0.6, "child"),
        (41764, 0.6, "child"),
        (42000, 0.6, "child"),
        (42088, 0.6, "child"),
        (42288, 6701.7, "main"),
        (43048, 0.5, "child"),
        (43720, 2802.6, "main"),
        (44164, 0.6, "child"),
        (45636, 0.5, "child"),
        (47068, 0.7, "child"),
        (47388, 11155.3, "main"),
        (48124, 1544.0, "main"),
        (48356, 6510.1, "main"),
        (48932, 0.5, "child"),
        (48996, 0.7, "child"),
        (50700, 0.6, "child"),
        (51044, 0.7, "child"),
        (51352, 0.6, "child"),
        (51404, 4236.8, "main"),
        (51752, 8488.9, "main"),
        (52172, 0.5, "child"),
        (52556, 0.6, "child"),
        (53284, 0.6, "child"),
    ]
    main_count = sum(1 for _pid, _cpu, kind in per_process if kind == "main")
    total = sum(cpu for _pid, cpu, _kind in per_process)
    return {
        "rule": (
            "D16-13: no ``python -m black`` subprocess call is allowed in any form "
            "(write mode or --check); use the in-process black_check_gate.py, and if "
            "that gate is unavailable skip formatting and register '格式门未执行'. "
            "Single-machine format concurrency is capped at 1."
        ),
        "measured_on": "2026-11-05, Asia/Shanghai",
        "black_processes_cleaned": len(per_process),
        "black_processes_main": main_count,
        "black_processes_child": len(per_process) - main_count,
        "accumulated_cpu_seconds": round(total, 1),
        "per_process": [
            {"pid": pid, "cpu_seconds": cpu, "kind": kind} for pid, cpu, kind in per_process
        ],
        "verified_remaining_black_processes": 0,
        "crashed_run_engineering_cost": {
            "note": (
                "the 2026-11-04 launch died at the end of the pricing segment "
                "(ValueError: too many values to unpack) after five price-measurement "
                "Adam; no timing file was written, so its CPU is not recoverable. "
                "It is engineering cost, billed outside the 13,000 experiment cap."
            ),
            "adam_updates": 5,
            "cpu_seconds": "not_recorded",
            "counted_in_experiment_cap": False,
        },
        "possible_link_not_proven": (
            "the 19.2c slowdown (7.48 CPU/update, 2.2x the first run) was attributed to "
            "external contention; idle black pools are a candidate cause but this is "
            "'to be proven' and is not merged into any conclusion"
        ),
    }


def black_process_count() -> int:
    """Owner 2026-11-05 item 2: zero-cost guard, read-only, never stops training.

    Counts live python processes whose command line mentions black.  A non-zero
    count is recorded and warned about in ``operational_waste``; it is never
    treated as an experiment failure, because the cause is external.
    """
    try:
        import psutil
    except Exception:
        return -1
    count = 0
    for process in psutil.process_iter(["name", "cmdline"]):
        try:
            name = (process.info["name"] or "").lower()
            if "python" not in name:
                continue
            parts = process.info["cmdline"] or []
            if any("black" in part for part in parts):
                count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return count


def sanitize_key(key):
    """D16-15: turn a dict key into a JSON-legal scalar."""
    if isinstance(key, tuple):
        return "|".join(str(part) for part in key)
    if isinstance(key, bool) or key is None or isinstance(key, (int, float, str)):
        return key
    item = getattr(key, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            return str(key)
    return str(key)


def sanitize_keys(value):
    """D16-15: recursively make a payload JSON-safe (tuple keys, numpy scalars)."""
    if isinstance(value, dict):
        return {sanitize_key(key): sanitize_keys(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_keys(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (int, float, str)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            return str(value)
    return str(value)


def progress_append(path: Path, entry: dict) -> None:
    """D16-15: a reading exists only once it is on disk, one line at a time.

    Append-only, flushed and fsynced per line, so a crash after this call loses
    nothing and a crash before it never pretends the reading existed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    row = sanitize_keys({**entry, "logged_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json_safe(path: Path, payload: dict) -> dict:
    """D16-15: write-then-rename, never overwrite the old report on failure."""
    clean = sanitize_keys(payload)
    text = json.dumps(clean, ensure_ascii=False, indent=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text + "\n", encoding="utf-8")
    tmp.replace(path)
    return {"bytes": len(text.encode("utf-8")), "path": str(path)}


def serialization_precheck() -> dict:
    """D16-15: serialize a full report skeleton before any training may start.

    The 2026-11-05 run died after 1,200 updates because ``training.first`` used
    ``(seed, arm)`` tuple keys.  This gate therefore round-trips every report
    section (with fake readings) through a temp file and back; if anything is
    not JSON-serialisable the package must not start.
    """
    fake_steps = list(MID_RUN_STEPS)
    fake_readings = {
        f"{seed}|{arm}": {
            step: {
                "new_route_hit": 0.0,
                "new_route_miss": 1.0,
                "old_route_hit": 0.0,
                "new_examples": 96,
                "old_examples": 96,
            }
            for step in fake_steps
        }
        for seed in SEEDS
        for arm in ("O", "N")
    }
    skeleton = {
        "schema": "taiji-package19-2j-from-scratch-selection-ab-v1",
        "zero_cost_check": zero_cost_check(),
        "preregistration": preregistration_block(),
        "design": {"held_equal": ["a", "b"], "seeds": list(SEEDS), "adam_total": ADAM_CAP},
        "activity_precheck": {"O": {"need_fix": 96}, "N": {"need_fix": 96}},
        "price_guard": {"unit_cpu_per_update": 1.0, "within_cap": True},
        "checkpoint_preflight": {"O": True, "N": True, "drift": True},
        "identities": {"parent": "x", "O": "x", "N": "y", "changed": True},
        "operational_waste": operational_waste_block(),
        "crash_log": {"error": "x"},
        "mid_run_gate": {
            "steps": fake_steps,
            "readings": fake_readings,
            "updates_by_run": {f"{seed}|{arm}": 0 for seed in SEEDS for arm in ("O", "N")},
            "identifiability_at_step100": {"measured": False},
        },
        "training": {
            "first": {f"{seed}|{arm}": {"ce": 0.0} for seed in SEEDS for arm in ("O", "N")},
            "last": {f"{seed}|{arm}": {"ce": 0.0} for seed in SEEDS for arm in ("O", "N")},
            "curve": {f"{seed}|{arm}": [0.0] for seed in SEEDS for arm in ("O", "N")},
        },
        "cells": {f"{seed}|{arm}": {"completed": 0} for seed in SEEDS for arm in ("O", "N")},
        "delta": {"pooled": {"completed": {"lower": 0.0}}},
        "gates": {"28031": {"x": {"pass": True}}},
        "decision": {"branch": "x"},
        "saved_bundles": {f"{seed}|{arm}": {"path": "x"} for seed in SEEDS for arm in ("O", "N")},
        "budget": {"functional_total_cpu": 0.0},
        "cost": {"cpu_seconds": 0.0, "wall_seconds": 0.0},
    }
    skeleton["deliberate_tuple_probe"] = {("a", "b"): 1}
    tmp = ROOT / "reports/taiji_package19_2j_serialization_probe_20261105.json"
    info = write_json_safe(tmp, skeleton)
    round_trip = json.loads(tmp.read_text(encoding="utf-8"))
    tmp.unlink(missing_ok=True)
    return {
        "passed": isinstance(round_trip, dict),
        "bytes": info["bytes"],
        "sections": sorted(skeleton),
        "tuple_key_normalised": "a|b" in round_trip.get("deliberate_tuple_probe", {}),
    }


def crash_log_block() -> dict:
    """Owner 2026-11-05 item 3: the crash, its cause, and every call site audited."""
    return {
        "date": "2026-11-04",
        "error": "ValueError: too many values to unpack (expected 2)",
        "site": "main() warm-up pricing call, line 851 of the training script",
        "cause": (
            "the mid-run-gate change made train_run return (log, update, stopped); "
            "only the trajectory loop was updated, the pricing call still unpacked two"
        ),
        "fix": "warm_log, next_update, _warm_stopped = train_run(...)",
        "call_sites_audited": [
            {
                "line": 249,
                "role": "definition",
                "returns": "(log, update, stopped)",
                "verdict": "reference",
            },
            {
                "line": 851,
                "role": "pricing warm-up",
                "before": "2-value unpack",
                "after": "3-value unpack",
                "verdict": "fixed",
            },
            {
                "line": 983,
                "role": "trajectory loop",
                "before": "3-value unpack",
                "after": "3-value unpack",
                "verdict": "already correct",
            },
        ],
        "signature_sites_found": 3,
        "extra_clause": (
            "D16-13: any change to a return tuple requires auditing every call site in "
            "the same edit, and a pre-run must cover the changed path before a run starts"
        ),
        "second_crash": {
            "date": "2026-11-05",
            "phase": "after all 1,200 updates and all four step-300 bundles were on disk",
            "error": "TypeError: keys must be str, int, float, bool or None, not tuple",
            "site": "training.first / training.last built with (seed, arm) tuple keys",
            "cost": "the training result survived; every in-memory reading and the CPU timer died",
            "fix": (
                "keys rewritten as f'{seed}|{arm}', sanitize_keys() added for every payload, "
                "write_json_safe() does write-then-rename, and serialization_precheck() now "
                "round-trips the full report skeleton before any Adam step"
            ),
            "classification": "engineering failure, not an experiment failure (D16-15)",
        },
    }


def dry_run_pricing(runs, batches, dev, new_fixtures, old_fixtures, start_cpu, start_wall) -> int:
    """Owner 2026-11-05 item 4: a six-Adam pre-run that reaches past the crash point.

    The 2026-11-04 launch died at the *end* of the pricing segment, immediately
    after ``train_run`` returned.  Covering that path therefore means: three Adam
    price the update, the readout cache and one cached readout price the gate,
    the projected total is computed (the exact statements around the crash),
    then three more Adam drive the probe callback and one gradient probe exercises
    the identifiability branch.  No bundle, no main report, six Adam, <= 200 CPU.
    """
    o_run = runs[(SEEDS[0], "O")]
    precheck = serialization_precheck()
    if not precheck["passed"]:
        print(json.dumps({"dry_run": "blocked", "reason": "serialization_precheck"}))
        return 1
    warm_cpu = time.process_time()
    warm_log, first, warm_stopped = train_run(
        o_run["candidate"],
        o_run["optimizer"],
        o_run["params"],
        batches,
        0,
        "19.2j-dry-pricing",
        limit=DRY_WARMUP,
    )
    unit_cpu = (time.process_time() - warm_cpu) / max(1, first)

    cache_cpu = time.process_time()
    readout_cache = build_readout_cache(o_run["candidate"], dev)
    cache_built_cpu = time.process_time() - cache_cpu
    read_cpu = time.process_time()
    price_readout = route_readout(o_run["candidate"], readout_cache, new_fixtures, old_fixtures)
    readout_cpu = time.process_time() - read_cpu

    projected = (unit_cpu * ADAM_CAP + readout_cpu * MID_RUN_READOUTS + FIXED_CPU) * SAFETY_FACTOR
    price_guard = {
        "unit_cpu_per_update": unit_cpu,
        "updates_measured": first,
        "projected_adam_cpu": unit_cpu * ADAM_CAP,
        "readout_cpu_each": readout_cpu,
        "readouts": MID_RUN_READOUTS,
        "projected_readout_cpu": readout_cpu * MID_RUN_READOUTS,
        "readout_cache_build_cpu": cache_built_cpu,
        "fixed_cpu": FIXED_CPU,
        "projected_total_cpu": projected,
        "cap_cpu": BUDGET_CAP,
        "within_cap": projected <= BUDGET_CAP,
    }

    probe_calls = []

    def probe(update: int) -> bool:
        row = route_readout(o_run["candidate"], readout_cache, new_fixtures, old_fixtures)
        probe_calls.append({"update": update, **row})
        return False

    cont_cpu = time.process_time()
    cont_log, end_update, cont_stopped = train_run(
        o_run["candidate"],
        o_run["optimizer"],
        o_run["params"],
        batches,
        first,
        "19.2j-dry-continue",
        limit=DRY_CONTINUE,
        probe=probe,
    )
    cont_elapsed = time.process_time() - cont_cpu
    ident = gradient_probe(runs[(SEEDS[1], "N")]["candidate"], batches[0], "19.2j-dry-ident")

    cpu = time.process_time() - start_cpu
    wall = time.perf_counter() - start_wall
    adam = first + (end_update - first)
    report = {
        "schema": "taiji-package19-2j-dry-run-pricing-v1",
        "step": "N19.2j dry-run pricing coverage",
        "date": "2026-11-05",
        "purpose": (
            "cover the 2026-11-04 crash path before the 1,200-update run: the pricing "
            "segment end, three real updates after it, and the mid-run probe callback"
        ),
        "pricing": price_guard,
        "price_readout": price_readout,
        "pricing_updates": first,
        "continuation_updates": end_update - first,
        "continuation_log": cont_log,
        "continuation_stopped": cont_stopped,
        "probe_calls": probe_calls,
        "identifiability_probe": ident,
        "crash_point_covered": True,
        "adam_updates_total": adam,
        "within_pre_run_budget": adam <= 6 and cpu <= 200.0,
        "cost": {
            "cpu_seconds": cpu,
            "wall_seconds": wall,
            "adam_updates": adam,
            "pricing_cpu_seconds": unit_cpu * first,
            "continuation_cpu_seconds": cont_elapsed,
            "readout_cpu_seconds": readout_cpu,
            "cache_build_cpu_seconds": cache_built_cpu,
        },
        "not_counted_in": "the 13,000 CPU experiment cap (owner 2026-11-05 item 4)",
        "warm_log": warm_log,
        "warm_stopped": warm_stopped,
    }
    write_json_safe(DRY_REPORT, report)
    print(
        json.dumps(
            {
                "dry_run": "ok",
                "adam": adam,
                "projected_total_cpu": round(projected, 1),
                "within_cap": price_guard["within_cap"],
                "cpu": round(cpu, 1),
            }
        )
    )
    return 0


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


def utf8_legal(raw) -> bool:
    try:
        bytes(raw).decode("utf-8")
    except (UnicodeDecodeError, ValueError):
        return False
    return True


def decode_rows_ext(candidate, fixtures) -> list[dict]:
    """``h.decode_rows`` plus the two owner readings: EOS reached and UTF-8 legality."""
    probe = probe_mod.ProbeContext(candidate)
    probe.arm = "B_stop_lock"
    rows = []
    for fixture in fixtures:
        decoded = wall.decode_fixture(probe, fixture)
        verdict = wall.classify(fixture, decoded)
        metrics = wall.example_metrics(fixture, decoded)
        metrics["bucket"] = verdict["bucket"]
        metrics["frame_subtype"] = (
            wall.frame_split(fixture, decoded, verdict["position"])
            if verdict["bucket"] == "frame_byte_wrong"
            else None
        )
        metrics["position"] = verdict["position"]
        metrics["eos_stopped"] = bool(decoded["terminated"])
        metrics["utf8_valid"] = utf8_legal(decoded["raw"])
        rows.append(metrics)
    return rows


def encoding_block(rows) -> dict:
    def one(selected: list[dict]) -> dict:
        total = len(selected)
        return {
            "examples": total,
            "eos_stopped_rate": (
                sum(1 for row in selected if row["eos_stopped"]) / total if total else 0.0
            ),
            "utf8_valid_rate": (
                sum(1 for row in selected if row["utf8_valid"]) / total if total else 0.0
            ),
            "eos_not_stopped": sum(1 for row in selected if not row["eos_stopped"]),
            "utf8_invalid": sum(1 for row in selected if not row["utf8_valid"]),
        }

    return {
        "full_192": one(rows),
        "new_description_96": one([row for row in rows if row["new_description"]]),
        "old_description_96": one([row for row in rows if not row["new_description"]]),
    }


def rebuild_snapshot(candidate, dev, new_fixtures, new_pairs, tag: str) -> dict:
    rows = decode_rows_ext(candidate, dev)
    return {
        "rows_new96": [row for row in rows if row["new_description"]],
        "new_description_96": wall.aggregate(rows, True),
        "old_description_96": wall.aggregate(rows, False),
        "full_192": wall.aggregate(rows, None),
        "swap_new": wall.swap_summary(
            [wall.swap_pair(candidate, fixture, "B_stop_lock") for fixture in new_pairs]
        ),
        "scores_new96": h.score_snapshot(candidate, new_fixtures, tag),
        "encoding": encoding_block(rows),
    }


def assemble_main() -> int:
    """Owner 2026-11-05 item 1: rebuild the step-300 readings, zero Adam.

    The 2026-11-04 run finished all 1,200 updates and saved four bundles, then
    died serialising ``training.first`` (tuple keys).  Every reading that lived
    only in memory is gone; everything the bundles can reproduce is rebuilt here
    by forward-only evaluation.  No optimizer step, no bundle write, no confirm
    read.  Step-50/100/200/250 readings stay ``unrecoverable``: there is no
    intermediate checkpoint, and step 300 must not be used to infer them.
    """
    start_cpu = time.process_time()
    start_wall = time.perf_counter()
    payload = premise.build_payload()
    train = premise.fixtures(payload, "train")
    dev = premise.fixtures(payload, "dev")
    new_fixtures = tuple(fixture for fixture in dev if fixture["flags"]["new_description"])
    old_fixtures = tuple(fixture for fixture in dev if not fixture["flags"]["new_description"])
    _easy_pairs, new_pairs = wall.pair_fixtures(dev)
    if (
        len(dev) != 192
        or len(new_fixtures) != 96
        or len(old_fixtures) != 96
        or len(new_pairs) != PAIRS_PER_STRATUM
    ):
        raise RuntimeError("dev stratification drift")
    batches = h.schedule(train, UPDATES)

    REBUILD_PROGRESS.unlink(missing_ok=True)
    candidates = {}
    bundles = {}
    for seed in SEEDS:
        for arm in ("O", "N"):
            path = OUTPUT_DIR / f"{seed}-{ARMS[arm]}-step{UPDATES}.pt"
            candidate, _optimizer, _state, index, _cursor = load_routing_bundle(path)
            candidates[(seed, arm)] = candidate
            bundles[f"{seed}-{arm}"] = {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "update_index": index,
                "alignment": bool(candidate.head.config.alignment),
                "head_parameters": sum(p.numel() for p in candidate.head.parameters()),
                "parameter_sha": candidate_identity(candidate)["candidate_sha_v2"],
            }
    progress_append(
        REBUILD_PROGRESS,
        {"event": "assemble_start", "adam_updates": 0, "bundles": sorted(bundles)},
    )

    cells = {}
    readings = {}
    for seed in SEEDS:
        for arm in ("O", "N"):
            key = (seed, arm)
            snapshot = rebuild_snapshot(
                candidates[key], dev, new_fixtures, new_pairs, f"19.2j-rebuild-{arm}-{seed}"
            )
            cells[key] = cell_view(snapshot)
            reading = {
                "update": UPDATES,
                "new_route_hit": cells[key]["aggregate"]["route_hit_object"],
                "new_route_miss": cells[key]["rates"]["route_miss"],
                "old_route_hit": cells[key]["aggregate_old"]["route_hit_object"],
                "new_sentence": cells[key]["aggregate"]["completed"],
                "old_sentence": cells[key]["aggregate_old"]["completed"],
                "composite_start_object": cells[key]["aggregate"]["composite_start_hit_object"],
                "composite_start_place": cells[key]["aggregate"]["composite_start_hit_place"],
                "conditional_byte_object": cells[key]["aggregate"]["conditional_byte_object"],
                "conditional_byte_place": cells[key]["aggregate"]["conditional_byte_place"],
                "swap_argmax_changed": cells[key]["swap"]["pairs_argmax_changed"],
                "swap_pairs": cells[key]["swap"]["pairs"],
                "encoding": snapshot["encoding"],
                "measured": True,
            }
            readings[f"{seed}-{arm}"] = reading
            progress_append(
                REBUILD_PROGRESS,
                {
                    "event": "route_readout",
                    "run": f"{seed}-{arm}",
                    "update": UPDATES,
                    **{name: value for name, value in reading.items() if name != "encoding"},
                },
            )

    freeze_to_selection(candidates[(SEEDS[0], "N")])
    ident_300 = gradient_probe(candidates[(SEEDS[0], "N")], batches[0], "19.2j-rebuild-step300-N")
    progress_append(
        REBUILD_PROGRESS,
        {
            "event": "gradient_probe",
            "run": f"{SEEDS[0]}-N",
            "update": UPDATES,
            "log10_rms_ratio": ident_300["log10_rms_ratio"],
            "log10_ratio": ident_300["log10_ratio"],
            "measured": True,
        },
    )

    delta = {}
    for seed in SEEDS:
        n_rows = cells[(seed, "N")]["rows"]
        o_rows = cells[(seed, "O")]["rows"]
        delta[str(seed)] = {
            name: world_bootstrap(n_rows, o_rows, name, BOOTSTRAP_ROUNDS, BOOTSTRAP_SEED)
            for name in ("completed", "route_miss", "object_route", "object_start")
        }
    pooled_n = cells[(SEEDS[0], "N")]["rows"] + cells[(SEEDS[1], "N")]["rows"]
    pooled_o = cells[(SEEDS[0], "O")]["rows"] + cells[(SEEDS[1], "O")]["rows"]
    pooled = {
        name: world_bootstrap(pooled_n, pooled_o, name, BOOTSTRAP_ROUNDS, BOOTSTRAP_SEED)
        for name in ("completed", "route_miss", "object_route", "object_start")
    }
    delta["pooled"] = pooled

    gates = {str(seed): gates_view(cells[(seed, "O")], cells[(seed, "N")]) for seed in SEEDS}
    identifiable = bool(ident_300["identifiable"])
    decision = branch_view(gates[str(SEEDS[0])], {"completed": pooled["completed"]}, identifiable)
    decision["per_seed_branches"] = {
        str(seed): branch_view(
            gates[str(seed)], {"completed": delta[str(seed)]["completed"]}, identifiable
        )["branch"]
        for seed in SEEDS
    }

    cpu = time.process_time() - start_cpu
    wall_time = time.perf_counter() - start_wall
    unrecoverable = {
        "step_route_readings": {
            str(step): {
                "status": "unrecoverable",
                "measured": False,
                "reason": (
                    "no intermediate checkpoint was saved; the reading existed only in "
                    "memory and the report died at serialisation"
                ),
                "not_reconstructed": "step 300 readings are not used to infer earlier points",
            }
            for step in MID_RUN_STEPS
        },
        "mid_run_identifiability_at_step100_step200": {
            "status": "unrecoverable as numbers",
            "measured": False,
            "reason": "both probes ran in memory; neither was written before the crash",
        },
        "activity_precheck": {"status": "unrecoverable", "measured": False},
        "formal_price_guard": {
            "status": "unrecoverable",
            "measured": False,
            "proxy": str(DRY_REPORT.relative_to(ROOT)),
        },
        "checkpoint_preflight": {"status": "unrecoverable", "measured": False},
        "training_ce_curve": {"status": "unrecoverable", "measured": False},
        "cpu_timing": {
            "status": "unrecoverable",
            "measured": False,
            "reason": "the timer lived in memory only; this is the D16-15 trigger",
        },
    }
    inferred = {
        "step100_identifiability": {
            "status": "inferred",
            "measured": False,
            "reading": "log10 per-parameter RMS ratio > -1 (the step-100 gate did not fire)",
            "basis": (
                "28031-N reached step 300 and the gate is enforced at step 100; a value <= -1 "
                "would have stopped the package before any step-300 bundle existed"
            ),
            "not_a_numeric_value": True,
        }
    }
    report = {
        "schema": "taiji-package19-2j-from-scratch-selection-ab-v1",
        "step": "N19.2j from-scratch selection A/B",
        "date": "2026-11-05",
        "mode": "--assemble (read-only rebuild from the four step-300 bundles)",
        "rebuild": {
            "owner_item": "2026-11-05 item 1: rebuild the readings, zero Adam, <= 500 CPU",
            "bundles": bundles,
            "adam_updates": 0,
            "cpu_seconds": cpu,
            "wall_seconds": wall_time,
            "cpu_cap": REBUILD_CPU_CAP,
            "within_cap": cpu <= REBUILD_CPU_CAP,
            "cpu_timer_logged": str(REBUILD_PROGRESS.relative_to(ROOT)),
        },
        "supersedes": {
            "report": str(REPORT.relative_to(ROOT)),
            "branch": "budget_extrapolation_exceeds_cap",
            "why": "the owner raised the cap to 13,000 and the 1,200-update run then executed",
        },
        "design": {
            "question": "is the scoring form the object-side cause, tested from zero",
            "oe_arm": "O = pooled description/record dot-product form",
            "n_arm": "N = 19.2h sequence-surface form, re-initialised at trunk scale",
            "held_equal": [
                "parent binding graph (frozen, no copy/joint training)",
                "parent span pointer (frozen, no span supervision)",
                "initialisation seed (shared tensors copied bit for bit)",
                "data order (one schedule over the same train fixtures)",
                "batch 8 / LR 1e-3 / clip 1.0 / 300 updates / selection CE only",
            ],
            "only_difference": "scoring form (+1,320 parameters in N)",
            "seeds": list(SEEDS),
            "updates_per_run": UPDATES,
            "adam_total": ADAM_CAP,
        },
        "training": {
            "updates_completed": ADAM_CAP,
            "updates_by_run": {f"{seed}-{arm}": UPDATES for seed in SEEDS for arm in ("O", "N")},
            "wall_seconds": TRAINING_WALL_SECONDS,
            "wall_source": "bundle file timestamps 00:51:08 -> 02:13:49 (82.7 min, owner snapshot)",
            "cpu_seconds": None,
            "cpu_status": "instrumentation_gap",
            "readings_lost": "mid-run gate, activity pre-check, price guard, CE curve",
        },
        "readings": readings,
        "unrecoverable": unrecoverable,
        "inferred": inferred,
        "identifiability_at_step300": ident_300,
        "cells": {
            f"{seed}-{arm}": {
                "aggregate": cells[(seed, arm)]["aggregate"],
                "aggregate_old": cells[(seed, arm)]["aggregate_old"],
                "swap": cells[(seed, arm)]["swap"],
                "rates": cells[(seed, arm)]["rates"],
                "completed_examples": cells[(seed, arm)]["completed_examples"],
                "frame_model_generation": cells[(seed, arm)]["sentences"]["frame_model_generation"],
                "encoding": readings[f"{seed}-{arm}"]["encoding"],
            }
            for seed in SEEDS
            for arm in ("O", "N")
        },
        "delta": delta,
        "gates": gates,
        "decision": decision,
        "serialization_precheck": serialization_precheck(),
        "instrumentation_gap": {
            "what_is_missing": [
                "CPU seconds for the 1,200-update run",
                "step-50/100/200/250 route readings",
                "step-100/200 identifiability numbers",
                "activity pre-check, formal price guard, checkpoint pre-flight, CE curve",
            ],
            "what_is_available": [
                "four step-300 bundles (parameters + identity)",
                "wall time from file timestamps (4,961 s)",
                "the 00:47:54 six-Adam dry-run price report",
            ],
            "ruled_into": "D16-15: silent in-memory readings are a design defect, not bad luck",
        },
        "budget": {
            "experiment_cap_cpu": BUDGET_CAP,
            "experiment_cpu": None,
            "experiment_cpu_status": "instrumentation_gap",
            "rebuild_cap_cpu": REBUILD_CPU_CAP,
            "rebuild_cpu": cpu,
            "rebuild_within_cap": cpu <= REBUILD_CPU_CAP,
            "training_wall_seconds": TRAINING_WALL_SECONDS,
        },
        "cost": {
            "training_adam_updates": ADAM_CAP,
            "rebuild_adam_updates": 0,
            "confirm_reads": 0,
            "rebuild_cpu_seconds": cpu,
            "rebuild_wall_seconds": wall_time,
            "training_wall_seconds": TRAINING_WALL_SECONDS,
            "training_cpu_seconds": None,
        },
    }
    write_json_safe(REPORT, report)
    write_json_safe(
        MANIFEST,
        {
            "schema": "taiji-package19-2j-manifest-v1",
            "step": "N19.2j",
            "date": "2026-11-05",
            "mode": "assemble",
            "report": str(REPORT.relative_to(ROOT)),
            "parent": str(PARENT.relative_to(ROOT)),
            "bundles": bundles,
            "branch": decision["branch"],
            "gates": {name: gate["pass"] for name, gate in gates[str(SEEDS[0])].items()},
            "pooled_completed_delta": pooled["completed"]["delta"],
            "pooled_completed_n_wins": pooled["completed"]["n_wins"],
            "training_adam_updates": ADAM_CAP,
            "rebuild_adam_updates": 0,
            "training_wall_seconds": TRAINING_WALL_SECONDS,
            "training_cpu_seconds": None,
            "cpu_status": "instrumentation_gap",
            "confirm_reads": 0,
        },
    )
    print(
        json.dumps(
            {
                "branch": decision["branch"],
                "rebuild_cpu": round(cpu, 1),
                "within_cap": cpu <= REBUILD_CPU_CAP,
                "pooled_completed": pooled["completed"]["delta"],
                "pooled_completed_n_wins": pooled["completed"]["n_wins"],
                "identifiable": identifiable,
            },
            ensure_ascii=False,
        )
    )
    return 0


def main() -> int:
    if "--assemble" in sys.argv or "--eval-only" in sys.argv:
        return assemble_main()
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
            "operational_waste": operational_waste_block(),
            "crash_log": crash_log_block(),
            "decision": {
                "branch": "activity_precheck_failed",
                "rule": "D16-8: no update is allowed when the added block carries no gradient",
            },
            "budget": {**budget_block(), "spent_cpu": cpu},
            "cost": {"cpu_seconds": cpu, "adam_updates": 0, "confirm_reads": 0},
        }
        write_json_safe(REPORT, report)
        write_json_safe(
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
    old_fixtures = tuple(fixture for fixture in dev if not fixture["flags"]["new_description"])

    # ---- D16-15: the full report must round-trip before any Adam ----------
    serial_gate = serialization_precheck()
    progress_append(
        PROGRESS_LOG,
        {
            "event": "serialization_precheck",
            "passed": serial_gate["passed"],
            "tuple_key_normalised": serial_gate["tuple_key_normalised"],
            "adam_updates": 0,
        },
    )
    if not serial_gate["passed"]:
        raise RuntimeError("report serialization pre-check failed; refusing to train")

    # ---- owner 2026-11-05 item 4: six-Adam pre-run that reaches past the crash --
    if "--dry-run-pricing" in sys.argv:
        return dry_run_pricing(
            runs, batches, dev, new_fixtures, old_fixtures, start_cpu, start_wall
        )

    # ---- measured-price guard, then the four trajectories ----------------
    logs: dict[tuple[int, str], list] = {}
    warm_cpu = time.process_time()
    warm_log, next_update, _warm_stopped = train_run(
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

    cache_cpu = time.process_time()
    readout_cache = build_readout_cache(runs[(SEEDS[0], "O")]["candidate"], dev)
    cache_built_cpu = time.process_time() - cache_cpu
    read_cpu = time.process_time()
    price_readout = route_readout(
        runs[(SEEDS[0], "O")]["candidate"], readout_cache, new_fixtures, old_fixtures
    )
    readout_cpu = time.process_time() - read_cpu

    projected = (unit_cpu * ADAM_CAP + readout_cpu * MID_RUN_READOUTS + FIXED_CPU) * SAFETY_FACTOR
    progress_append(
        PROGRESS_LOG,
        {
            "event": "price_guard",
            "updates_measured": next_update,
            "warm_cpu_seconds": warm_elapsed,
            "cache_build_cpu_seconds": cache_built_cpu,
            "readout_cpu_seconds": readout_cpu,
            "projected_total_cpu": projected,
            "within_cap": projected <= BUDGET_CAP,
            "adam_updates": next_update,
        },
    )
    price_guard = {
        "unit_cpu_per_update": unit_cpu,
        "updates_measured": next_update,
        "projected_adam_cpu": unit_cpu * ADAM_CAP,
        "readout_cpu_each": readout_cpu,
        "readouts": MID_RUN_READOUTS,
        "projected_readout_cpu": readout_cpu * MID_RUN_READOUTS,
        "readout_cache_build_cpu": cache_built_cpu,
        "fixed_cpu": FIXED_CPU,
        "projected_total_cpu": projected,
        "cap_cpu": BUDGET_CAP,
        "within_cap": projected <= BUDGET_CAP,
        "owner_estimate_cpu_per_update": 4.0,
        "owner_estimate_total_cpu": (4.0 * ADAM_CAP + FIXED_CPU) * SAFETY_FACTOR,
        "basis": (
            "measured marginal CPU on the first five updates of seed 28031 / O arm, the "
            "measured cost of one cached 192-example route readout x20, 428 CPU of fixed "
            "items, then x3.3"
        ),
        "first_readout_values": price_readout,
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
            "operational_waste": operational_waste_block(),
            "crash_log": crash_log_block(),
            "decision": {
                "branch": "budget_extrapolation_exceeds_cap",
                "rule": "owner rule: if the measured extrapolation exceeds the cap, report first and do not run",
            },
            "notes": [
                "only the five price-measurement updates ran; the 1,200-update A/B was not started",
                "the D16-12 identifiability gate is pre-registered at step 300 and was never reached; "
                "the initialisation reading (log10 per-parameter RMS ratio -1.425) is reported but "
                "licenses no structural claim either way",
            ],
            "budget": {**budget_block(unit_cpu, readout_cpu), "spent_cpu": cpu},
            "cost": {"cpu_seconds": cpu, "adam_updates": next_update, "confirm_reads": 0},
        }
        write_json_safe(REPORT, report)
        write_json_safe(
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

    logs: dict[tuple[int, str], list] = {}
    readings: dict[str, dict[int, dict]] = {
        f"{seed}-{arm}": {} for seed in SEEDS for arm in ("O", "N")
    }
    stop_state: dict = {"flag": False, "reason": None, "where": None}
    guard: list[dict] = []

    def record_guard(point: str) -> None:
        """Owner 2026-11-05 item 2: warn-only black-process guard."""
        guard.append({"point": point, "black_processes": black_process_count()})

    record_guard("start")
    progress_append(
        PROGRESS_LOG,
        {
            "event": "run_start",
            "adam_planned": ADAM_CAP,
            "seeds": list(SEEDS),
            "mid_run_steps": list(MID_RUN_STEPS),
            "ident_steps": list(IDENT_STEPS),
        },
    )

    def make_probe(seed: int, arm: str):
        key = f"{seed}-{arm}"

        def probe(update: int) -> bool:
            candidate = runs[(seed, arm)]["candidate"]
            if update in MID_RUN_STEPS:
                record_guard(f"{key}@{update}")
                row = route_readout(candidate, readout_cache, new_fixtures, old_fixtures)
                readings[key][update] = row
                progress_append(
                    PROGRESS_LOG,
                    {
                        "event": "route_readout",
                        "run": key,
                        "update": update,
                        **row,
                        "cpu_seconds": time.process_time() - start_cpu,
                    },
                )
                best = max(
                    (
                        entry["old_route_hit"]
                        for entry in readings[key].values()
                        if "old_route_hit" in entry
                    ),
                    default=0.0,
                )
                fired = row["old_route_hit"] < best - OLD_ROUTE_SLACK
                progress_append(
                    PROGRESS_LOG,
                    {
                        "event": "stop_decision",
                        "run": key,
                        "update": update,
                        "rule": "old_capability_regression",
                        "fire": fired,
                        "old_route_hit": row["old_route_hit"],
                        "best_old_route_hit": best,
                    },
                )
                if fired:
                    stop_state.update(
                        flag=True,
                        reason="old_capability_regression",
                        where=f"{key}@{update}",
                    )
                    return True
            if arm == "N" and update in IDENT_STEPS:
                probe_row = gradient_probe(candidate, batches[0], f"19.2j-ident-{key}-{update}")
                readings[key].setdefault(update, {})["identifiability"] = probe_row
                progress_append(
                    PROGRESS_LOG,
                    {
                        "event": "gradient_probe",
                        "run": key,
                        "update": update,
                        "log10_rms_ratio": probe_row["log10_rms_ratio"],
                        "log10_ratio": probe_row["log10_ratio"],
                        "identifiable": probe_row["identifiable"],
                        "measured": True,
                    },
                )
                fired = (
                    update == IDENT_STOP_STEP
                    and probe_row["log10_rms_ratio"] <= IDENT_STOP_THRESHOLD
                )
                progress_append(
                    PROGRESS_LOG,
                    {
                        "event": "stop_decision",
                        "run": key,
                        "update": update,
                        "rule": "still_not_identifiable",
                        "fire": fired,
                        "log10_rms_ratio": probe_row["log10_rms_ratio"],
                    },
                )
                if fired:
                    stop_state.update(
                        flag=True, reason="still_not_identifiable", where=f"{key}@{update}"
                    )
                    return True
            return False

        return probe

    for seed in SEEDS:
        for arm in ("O", "N"):
            key = (seed, arm)
            run = runs[key]
            first = next_update if key == (SEEDS[0], "O") else 0
            run_cpu = time.process_time()
            log, _end, stopped = train_run(
                run["candidate"],
                run["optimizer"],
                run["params"],
                batches,
                first,
                f"19.2j-{arm}-{seed}",
                probe=make_probe(seed, arm),
            )
            progress_append(
                PROGRESS_LOG,
                {
                    "event": "run_segment",
                    "run": f"{seed}-{arm}",
                    "updates": len(log),
                    "stopped": stopped,
                    "cpu_seconds": time.process_time() - run_cpu,
                    "cumulative_cpu_seconds": time.process_time() - start_cpu,
                },
            )
            logs[key] = [*warm_log, *log] if key == (SEEDS[0], "O") else log
            if stopped:
                break
        if stop_state["flag"]:
            break

    # ---- owner mid-run gate fired: stop the package, no structural claim ---
    if stop_state["flag"]:
        cpu = time.process_time() - start_cpu
        updates_by_run = {f"{seed}-{arm}": len(log) for (seed, arm), log in logs.items()}
        ident_100 = readings[f"{SEEDS[0]}-N"].get(IDENT_STOP_STEP, {}).get("identifiability")
        mechanism = (
            "the alignment block enters the row score as four extra inputs multiplied by the "
            "freshly initialised W_hidden columns (bound 1/sqrt(64) = 0.125); at score level its "
            "contribution stays around 1e-2 of the pooled path, so its own gradient stays two "
            "orders below the trunk even under Adam (Adam rescales per parameter, it does not "
            "restore a drowned forward path). A from-scratch run therefore reproduces the 19.2h "
            "ratio instead of escaping it"
        )
        minimal_fix = (
            "two single-variable options, neither started: (a) output-side scale alignment - "
            "normalise the four added row inputs to unit RMS before W_hidden, so the block's "
            "contribution is O(1) from step 0; (b) feed the alignment features as trunk input "
            "instead of an additive row input (replace, not append). Both change only the "
            "scoring form's wiring, which is exactly the variable under test"
        )
        report = {
            "schema": "taiji-package19-2j-from-scratch-selection-ab-v1",
            "step": "N19.2j from-scratch selection A/B",
            "date": "2026-11-04",
            "zero_cost_check": zero_cost_check(),
            "preregistration": preregistration_block(),
            "activity_precheck": activity,
            "price_guard": price_guard,
            "operational_waste": {
                **operational_waste_block(),
                "runtime_guard": guard,
                "runtime_guard_violations": sum(
                    1 for entry in guard if entry["black_processes"] > 0
                ),
            },
            "crash_log": crash_log_block(),
            "mid_run_gate": {
                "rule": (
                    "owner 2026-11-04: record the added-block vs trunk per-parameter RMS gradient "
                    "ratio at step 100 and 200; stop the package if step 100 is still <= -1; "
                    "monitor old-description route_hit every 50 steps and stop on a drop > 0.02 "
                    "below the run's best so far"
                ),
                "steps": list(MID_RUN_STEPS),
                "ident_steps": list(IDENT_STEPS),
                "readings": readings,
                "identifiability_at_step100": ident_100,
                "updates_by_run": updates_by_run,
            },
            "decision": {
                "branch": "mid_run_gate_stopped",
                "reason": stop_state["reason"],
                "where": stop_state["where"],
                "no_structural_conclusion": (
                    "the package stopped before step 300, so the O/N comparison, the swap argmax "
                    "gate and the step-300 identifiability gate are all unmeasured; this package "
                    "supports no structural claim in either direction"
                ),
            },
            "mechanism": mechanism,
            "minimal_improvement_not_started": minimal_fix,
            "budget": {
                **budget_block(unit_cpu, readout_cpu),
                "spent_cpu": cpu,
                "spent_share": cpu / BUDGET_CAP,
                "remaining_cpu_not_spent": BUDGET_CAP - cpu,
            },
            "cost": {
                "cpu_seconds": cpu,
                "adam_updates": sum(updates_by_run.values()),
                "confirm_reads": 0,
            },
        }
        write_json_safe(REPORT, report)
        write_json_safe(
            MANIFEST,
            {
                "schema": "taiji-package19-2j-manifest-v1",
                "step": "N19.2j",
                "date": "2026-11-04",
                "report": str(REPORT.relative_to(ROOT)),
                "branch": "mid_run_gate_stopped",
                "reason": stop_state["reason"],
                "adam_updates": sum(updates_by_run.values()),
                "cost_cpu": cpu,
                "bundles": [],
            },
        )
        print(
            json.dumps(
                {
                    "branch": "mid_run_gate_stopped",
                    "reason": stop_state["reason"],
                    "where": stop_state["where"],
                    "cpu": round(cpu, 1),
                }
            )
        )
        return 0

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
    ident_100 = readings[f"{SEEDS[0]}-N"].get(IDENT_STOP_STEP, {}).get("identifiability")
    ident_200 = readings[f"{SEEDS[0]}-N"].get(200, {}).get("identifiability")
    cpu = time.process_time() - start_cpu
    wall_time = time.perf_counter() - start_wall
    budget = {
        **budget_block(unit_cpu, readout_cpu),
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
        "operational_waste": {
            **operational_waste_block(),
            "runtime_guard": guard,
            "runtime_guard_violations": sum(1 for entry in guard if entry["black_processes"] > 0),
        },
        "crash_log": crash_log_block(),
        "mid_run_gate": {
            "rule": (
                "owner 2026-11-04: added-block vs trunk per-parameter RMS gradient ratio "
                "logged at step 100 and 200; stop if step 100 is still <= -1. Route readings "
                "every 50 steps on the unchanged 192-example dev set; stop on an old-description "
                "route_hit drop > 0.02 below the run's best so far"
            ),
            "steps": list(MID_RUN_STEPS),
            "ident_steps": list(IDENT_STEPS),
            "readings": readings,
            "updates_by_run": {f"{seed}-{arm}": len(log) for (seed, arm), log in logs.items()},
            "identifiability_at_step100": ident_100,
            "identifiability_at_step200": ident_200,
            "fired": False,
        },
        "training": {
            "updates": total_updates,
            "first": {f"{seed}|{arm}": log[0] for (seed, arm), log in logs.items()},
            "last": {f"{seed}|{arm}": log[-1] for (seed, arm), log in logs.items()},
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
    write_json_safe(REPORT, report)
    write_json_safe(
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
