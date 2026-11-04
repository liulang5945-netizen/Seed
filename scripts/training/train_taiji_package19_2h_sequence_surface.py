"""19.2h: sequence-surface alignment on the 19.5 B parent (parameterised, +1,320).

19.2g closed the loss-only road: the record-matching margin was identically zero
on the training split (min margin 5.313, median 8.08, m = 1.0), so 30 Adam only
re-trained the CE and the swap argmax stayed at 0/20.  The owner therefore moved
the dose from the objective to the *architecture* and added two hard
pre-conditions (activity pre-check and an equal-dose CE-only control arm).

The alignment block reads the pooled 128-d encoding as four ordered 32-d
subspaces, projects them with one shared 32x32 ``W_align``, and compares a
record's subspace grid with the goal's at five relative displacements
(-2..+2) by cosine agreement.  The five scalars are mixed by a 5x8 offset
embedding and folded into four extra row inputs of ``W_hidden``.  The pooled
features (goal, key, s_t, previous embedding, goal*key) stay untouched as a
residual, so the added cost is exactly

    32*32 (W_align)  +  5*8 (offset embedding)  +  4*64 (extra W_hidden columns)
    = 1,024 + 40 + 256 = 1,320 parameters  (+3.6% of the 37,120 head)

Two trains of exactly 30 Adam updates share one parent, one data order and one
loss (selection CE + inherited span term).  The only difference is the graph:

* ``19.2h``  - parent binding + alignment head (38,440 parameters).
* ``CE-only`` - parent binding + parent head verbatim (37,120 parameters).

``Delta = 19.2h - CE-only`` is reported with a paired bootstrap interval over
the 96 new-description examples.  No margin term is re-introduced: 19.2g
already showed it cannot fire on this data.

Activity pre-check (owner discipline D16-8) runs *before* any update:
1. gradient norms of every alignment tensor on the first training batch must be
   non-zero, and
2. the share of new-description examples whose object-start route is wrong must
   be >= 20% (the 19.2f target was 37/96 = 38.5%).
A failure stops the package with ``activity_precheck_failed`` and no training.

Because the head changed, ``candidate_sha_v2`` changes.  The package therefore
saves a new bundle and runs the 19.4-style preflight: identity 8/8 rescue,
save -> restore bitwise, module-drift rejection.

gamma is the generation weight and 1 - gamma the copy weight; nothing here
infers training credit from gamma (M3).  D16-6 (all-NaN ``28031-A``) is not
touched and no confirm example is read.

Run: ``python scripts/training/train_taiji_package19_2h_sequence_surface.py``
"""

from __future__ import annotations

import copy
import importlib.util
import json
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

from taiji import data_premise_18_0 as premise  # noqa: E402
from taiji.candidate_identity import candidate_identity  # noqa: E402
from taiji.living_core import EOS  # noqa: E402
from taiji.selection_routing import (  # noqa: E402
    ALIGN_FEATURES,
    OrderedSelectionHead,
    RoutingConfig,
    _advanced_state,
    _s_t,
    observe_goal,
    resolve_prefix,
    route_labels,
    routing_step,
)
from taiji.selection_routing_checkpoint import (  # noqa: E402
    RoutingCandidate,
    load_routing_bundle,
    make_routing_optimizer,
    parameter_tensors,
    save_routing_bundle,
)

COMPARE_MODULE = ROOT / "scripts/training/compare_taiji_selection_routing_17_5.py"
PARENT = ROOT / "output/taiji_package19_compare/28031-B_stop_lock-step300.pt"
SEED = 28031
ARM = "B_stop_lock"
UPDATES = 30
ADAM_CAP = 60
OBJECT_START = 3
PLACE_START = 21
CONTENT_STARTS = (OBJECT_START, PLACE_START)
PAIRS_PER_STRATUM = 20
ARGMAX_GATE = 18
SATURATION_TOP1 = 0.90
NEED_FIX_GATE = 0.20
BOOTSTRAP_ROUNDS = 2000
BOOTSTRAP_SEED = 20261104
RESCUE_EXAMPLES = 8

OUTPUT_DIR = ROOT / "output/taiji_package19_2h_sequence_surface"
FINAL_BUNDLE = OUTPUT_DIR / "28031-B_stop_lock-step30-align.pt"
PREFLIGHT_BUNDLE = OUTPUT_DIR / "preflight-28031-B_stop_lock-step0-align.pt"
CORRUPT_BUNDLE = OUTPUT_DIR / "drift-probe-align.pt"
REPORT = ROOT / "reports/taiji_package19_2h_sequence_surface_20261104.json"
MANIFEST = ROOT / "plans/manifests/taiji_package19_2h_sequence_surface_20261104.json"

# Budget discipline (owner, 19.2h): functional item list x 3.3, cap 1,500 CPU.
BUDGET_ITEMS = (
    ("2 x 30 Adam selection updates (alignment head + CE-only control), batch 8, CE + span", 140.0),
    ("two full 192-example decode passes (parent before, 19.2h after)", 180.0),
    ("one 96-example new-description decode pass (CE-only control after)", 30.0),
    ("two start-score extractions on the 96 new-description examples", 30.0),
    ("activity pre-check: first-batch gradient norms + object-start need-fix share", 25.0),
    ("payload, parent health, checkpoint dry-run, identity preflight, report", 40.0),
)
SAFETY_FACTOR = 3.3
BUDGET_CAP = 1500.0

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
    spec = importlib.util.spec_from_file_location("cmp175_19_2h", COMPARE_MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def schedule(train: tuple[dict, ...], updates: int) -> list[tuple[dict, ...]]:
    """One data order shared by both arms (same seed, same batch size)."""
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


def build_alignment_candidate(parent):
    """Parent binding + span head, new 38,440-parameter head with alignment."""
    head = OrderedSelectionHead(RoutingConfig(alignment=True), seed=SEED)
    old = parent.head
    extra = head.W_hidden.weight.shape[1] - old.W_hidden.weight.shape[1]
    if extra != ALIGN_FEATURES:
        raise RuntimeError(f"unexpected alignment row growth: {extra}")
    with torch.no_grad():
        head.W_hidden.weight[:, : old.W_hidden.weight.shape[1]].copy_(old.W_hidden.weight)
        head.W_hidden.bias.copy_(old.W_hidden.bias)
        head.w_out.weight.copy_(old.w_out.weight)
        head.k_null.copy_(old.k_null)
    candidate = RoutingCandidate(
        parent.binding,
        head,
        arm=ARM,
        seed=SEED,
        span_head=copy.deepcopy(parent.span_head),
    )
    candidate.configure_trainable()
    return candidate


def selection_loss(candidate, prepared, prefix, target_bytes, route_index, route_stage, context):
    """Selection-only CE, graph frozen, mirrors ``selection_ce``."""
    state = prepared.state
    state, gate_index = observe_goal(candidate, prepared, gate_index=0)
    input_prefix = resolve_prefix(prefix, len(target_bytes) - 1)
    terms: dict[int, list[torch.Tensor]] = {0: [], 1: [], 2: []}
    goal_vec = context.goal_vec.detach()
    key_vecs = [key.detach() for key in context.key_vecs]
    previous = EOS
    for t, _label in enumerate(target_bytes):
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
            )
            s_t = _s_t(candidate, prepared, state, previous)
            previous_embedding = candidate.subject.encoder.embedding(
                torch.tensor(previous, device=candidate.subject.W.device)
            )
        scores = candidate.head.score_rows(goal_vec, key_vecs, s_t, previous_embedding)
        route_label = route_index[t]
        if route_label >= 0:
            terms[route_stage[t]].append(
                F.cross_entropy(scores.unsqueeze(0), torch.tensor([route_label]))
            )
        if t == len(target_bytes) - 1:
            break
        accepted = input_prefix[t]
        with torch.no_grad():
            state, _, gate_index = _advanced_state(candidate, prepared, step, accepted)
            previous = accepted
    present = [torch.stack(values).mean() for values in terms.values() if values]
    return torch.stack(present).mean() if present else torch.zeros((), dtype=torch.float64)


def batch_loss(candidate, batch, tag: str) -> tuple[torch.Tensor, dict]:
    """Selection CE + inherited span term; identical for both arms."""
    module = cmp()
    ce_values = []
    span_values = []
    for fixture in batch:
        prepared, context = module.prepared_pair(candidate, fixture, tag, "train")
        span, _spans = module.span_term(prepared, fixture, candidate.span_head)
        span_values.append(span)
        routes = route_labels(fixture)
        ce_values.append(
            selection_loss(
                candidate,
                prepared,
                module.teacher_prefix(fixture),
                tuple(fixture["target_bytes"]),
                routes[0],
                routes[1],
                context,
            )
        )
    ce_total = torch.stack(ce_values).mean()
    span_total = torch.stack(span_values).mean()
    return ce_total + span_total, {
        "ce": float(ce_total.detach()),
        "span": float(span_total.detach()),
    }


def parameter_grad_norm(candidate, name: str) -> float:
    for parameter_name, parameter in candidate.head.named_parameters():
        if parameter_name == name:
            if parameter.grad is None:
                return 0.0
            return float(parameter.grad.detach().pow(2).sum()) ** 0.5
    return 0.0


def gradient_probe(candidate, batch, tag: str) -> dict:
    """First-batch gradient norms per alignment tensor; the D16-8 gate."""
    module = cmp()
    parameters = module.set_trainable(candidate, "selection")
    loss, stats = batch_loss(candidate, batch, tag)
    loss.backward()
    hidden_all = parameter_grad_norm(candidate, "W_hidden.weight")
    hidden_new = 0.0
    for parameter_name, parameter in candidate.head.named_parameters():
        if parameter_name == "W_hidden.weight" and parameter.grad is not None:
            hidden_new = float(parameter.grad.detach()[:, -ALIGN_FEATURES:].pow(2).sum()) ** 0.5
    norms = {
        "W_align.weight": parameter_grad_norm(candidate, "W_align.weight"),
        "offset_embed": parameter_grad_norm(candidate, "offset_embed"),
        "W_hidden.weight_new_columns": hidden_new,
        "W_hidden.weight_all": hidden_all,
        "w_out.weight": parameter_grad_norm(candidate, "w_out.weight"),
        "k_null": parameter_grad_norm(candidate, "k_null"),
        "span_head": sum(
            float(parameter.grad.detach().pow(2).sum())
            for parameter in candidate.span_head.parameters()
            if parameter.grad is not None
        )
        ** 0.5,
        "loss": float(loss.detach()),
        "ce": stats["ce"],
        "span": stats["span"],
    }
    module.set_trainable(candidate, "selection")
    for parameter in parameters:
        parameter.grad = None
    candidate.head.zero_grad(set_to_none=True)
    return {
        "per_tensor_norm": norms,
        "non_zero_alignment": (
            norms["W_align.weight"] > 0.0
            and norms["offset_embed"] > 0.0
            and norms["W_hidden.weight_new_columns"] > 0.0
        ),
        "meaning": (
            "the alignment tensors must carry gradient on the very first batch; a zero norm "
            "means the added block is unreachable from the loss (D16-8)"
        ),
    }


def object_start_scores(candidate, prepared, context, prefix) -> torch.Tensor:
    state = prepared.state
    state, gate_index = observe_goal(candidate, prepared, gate_index=0)
    previous = EOS
    captured = None
    for t in range(OBJECT_START + 1):
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
        if t == OBJECT_START:
            captured = step.scores.detach().clone()
            break
        accepted = prefix[t]
        state, _, gate_index = _advanced_state(candidate, prepared, step, accepted)
        previous = accepted
    return captured


def scored_route(scores: torch.Tensor) -> int:
    index = int(torch.argmax(scores))
    null_index = len(scores) - 1
    if scores[null_index] - scores[index] >= -1e-12:
        index = null_index
    return index


def need_fix_share(candidate, fixtures: tuple[dict, ...], tag: str) -> dict:
    """Share of examples whose object-start route is wrong; the D16-8 dose check."""
    module = cmp()
    wrong = 0
    buckets: dict[str, int] = {}
    for fixture in fixtures:
        prepared, context = module.prepared_pair(candidate, fixture, tag, "needfix")
        prefix = module.teacher_prefix(fixture)
        scores = object_start_scores(candidate, prepared, context, prefix)
        gold = int(fixture["selection_label"])
        chosen = scored_route(scores)
        record_count = len(fixture["record_keys"])
        key = "gold" if chosen == gold else ("null" if chosen >= record_count else "other_record")
        buckets[key] = buckets.get(key, 0) + 1
        if chosen != gold:
            wrong += 1
    return {
        "examples": len(fixtures),
        "wrong": wrong,
        "share": wrong / len(fixtures) if fixtures else 0.0,
        "bucket": buckets,
        "gate": NEED_FIX_GATE,
        "pass": (wrong / len(fixtures) if fixtures else 0.0) >= NEED_FIX_GATE,
    }


def decode_rows(candidate, fixtures: tuple[dict, ...]) -> list[dict]:
    probe = probe_mod.ProbeContext(candidate)
    probe.arm = ARM
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
        rows.append(metrics)
    return rows


def score_snapshot(candidate, fixtures: tuple[dict, ...], tag: str) -> dict:
    """Top-1 route probability and route correctness at both content starts."""
    module = cmp()
    top1_values: list[float] = []
    by_start: dict[str, dict[str, list[float]]] = {
        "object": {"top1": [], "route_ok": []},
        "place": {"top1": [], "route_ok": []},
    }
    for fixture in fixtures:
        prepared, context = module.prepared_pair(candidate, fixture, tag, "score")
        prefix = module.teacher_prefix(fixture)
        targets = {
            OBJECT_START: int(fixture["selection_label"]),
            PLACE_START: module.place_index(fixture),
        }
        captured: dict[int, torch.Tensor] = {}
        state = prepared.state
        state, gate_index = observe_goal(candidate, prepared, gate_index=0)
        previous = EOS
        for t in range(max(CONTENT_STARTS) + 1):
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
            if t in CONTENT_STARTS:
                captured[t] = step.scores.detach().clone()
            if t == max(CONTENT_STARTS):
                break
            accepted = prefix[t]
            state, _, gate_index = _advanced_state(candidate, prepared, step, accepted)
            previous = accepted
        for position, scores in captured.items():
            name = "object" if position == OBJECT_START else "place"
            top1 = float(torch.softmax(scores, dim=0).max())
            top1_values.append(top1)
            by_start[name]["top1"].append(top1)
            by_start[name]["route_ok"].append(float(scored_route(scores) == targets[position]))
    return {
        "examples": len(fixtures),
        "points": len(top1_values),
        "top1_probability": summary(top1_values),
        "route_share": {
            name: {
                "points": len(values["route_ok"]),
                "route_hit": statistics.fmean(values["route_ok"]) if values["route_ok"] else 0.0,
                "top1_probability": summary(values["top1"]),
            }
            for name, values in by_start.items()
        },
        "share_top1_ge_0_99": (
            sum(1 for value in top1_values if value >= 0.99) / len(top1_values)
            if top1_values
            else 0.0
        ),
    }


def full_snapshot(candidate, dev, new_fixtures, new_pairs, tag: str) -> dict:
    rows = decode_rows(candidate, dev)
    return {
        "full_192": wall.aggregate(rows, None),
        "old_description_96": wall.aggregate(rows, False),
        "new_description_96": wall.aggregate(rows, True),
        "swap_new": wall.swap_summary(
            [wall.swap_pair(candidate, fixture, ARM) for fixture in new_pairs]
        ),
        "scores_new96": score_snapshot(candidate, new_fixtures, tag),
        "rows_new96": [row for row in rows if row["new_description"]],
    }


def subset_snapshot(candidate, fixtures: tuple[dict, ...]) -> dict:
    rows = decode_rows(candidate, fixtures)
    return {
        "new_description_96": wall.aggregate(rows, True),
        "rows_new96": rows,
    }


def train_arm(candidate, optimizer, batches, tag: str) -> tuple[list[dict], int]:
    module = cmp()
    log = []
    updates = 0
    for update in range(1, UPDATES + 1):
        parameters = module.set_trainable(candidate, "selection")
        loss, stats = batch_loss(candidate, batches[update - 1], f"{tag}-{update}")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        updates += 1
        log.append({"update": update, **stats})
    return log, updates


def teacher_forced(candidate, fixture, tag: str) -> list[torch.Tensor]:
    module = cmp()
    prepared, context = module.prepared_pair(candidate, fixture, tag, "tf")
    prefix = module.teacher_prefix(fixture)
    target_bytes = tuple(fixture["target_bytes"])
    state = prepared.state
    state, gate_index = observe_goal(candidate, prepared, gate_index=0)
    previous = EOS
    distributions = []
    with torch.no_grad():
        for t in range(len(target_bytes)):
            step = routing_step(
                candidate,
                prepared,
                state,
                previous,
                context,
                candidate.head,
                arm=ARM,
                gate_index=gate_index,
            )
            distributions.append(step.probabilities.detach().clone())
            if t == len(target_bytes) - 1:
                break
            accepted = prefix[t]
            state, _, gate_index = _advanced_state(candidate, prepared, step, accepted)
            previous = accepted
    return distributions


def identity_rescue(candidate, fixtures: tuple[dict, ...], tag: str) -> dict:
    """8/8 identity rescue: reload must reproduce the same distributions bit for bit."""
    module = cmp()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    candidate.configure_trainable()
    fixed_state = module.prepared_pair(candidate, fixtures[0], "19.2h-rescue-save", "save")[0].state
    identity = candidate_identity(candidate)
    info = save_routing_bundle(
        PREFLIGHT_BUNDLE,
        candidate,
        make_routing_optimizer(candidate),
        0,
        {"stage": "N19.2h-rescue", "seed": SEED, "arm": ARM},
        identity,
        fixed_state,
    )
    reloaded = load_routing_bundle(PREFLIGHT_BUNDLE)[0]
    worst = 0.0
    agree = 0
    for index, fixture in enumerate(fixtures):
        before = teacher_forced(candidate, fixture, f"19.2h-rescue-{tag}-{index}")
        after = teacher_forced(reloaded, fixture, f"19.2h-rescue-{tag}-r{index}")
        diff = 0.0
        for left, right in zip(before, after, strict=True):
            diff = max(diff, float((left - right).abs().max()))
        worst = max(worst, diff)
        agree += int(diff <= 0.0)
    PREFLIGHT_BUNDLE.unlink(missing_ok=True)
    return {
        "examples": len(fixtures),
        "agree": agree,
        "all_agree": agree == len(fixtures),
        "max_abs_diff": worst,
        "dry_run_bytes": info["bytes"],
        "dry_run_sha256": info["sha256"],
        "meaning": "reloaded bundle must reproduce 8/8 teacher-forced trajectories exactly",
    }


def checkpoint_preflight(candidate, optimizer, state_fixture) -> dict:
    module = cmp()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prepared, _context = module.prepared_pair(candidate, state_fixture, "19.2h-dryrun", "save")
    identity = candidate_identity(candidate)
    before = parameter_tensors(candidate)
    info = save_routing_bundle(
        PREFLIGHT_BUNDLE,
        candidate,
        optimizer,
        0,
        {"stage": "N19.2h-preflight", "seed": SEED, "arm": ARM},
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


def module_drift_rejection(candidate, state_fixture) -> dict:
    """A bundle missing one head tensor must be refused, not silently loaded."""
    module = cmp()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    candidate.configure_trainable()
    prepared, _context = module.prepared_pair(candidate, state_fixture, "19.2h-drift", "save")
    identity = candidate_identity(candidate)
    save_routing_bundle(
        CORRUPT_BUNDLE,
        candidate,
        make_routing_optimizer(candidate),
        0,
        {"stage": "N19.2h-drift", "seed": SEED, "arm": ARM},
        identity,
        prepared.state,
    )
    payload = torch.load(CORRUPT_BUNDLE, map_location="cpu", weights_only=True)
    payload["head_parameters"].pop("W_align.weight")
    torch.save(payload, CORRUPT_BUNDLE)
    rejected = False
    message = ""
    try:
        load_routing_bundle(CORRUPT_BUNDLE)
    except Exception as error:  # the rejection itself is the assertion
        rejected = True
        message = f"{type(error).__name__}: {str(error)[:160]}"
    CORRUPT_BUNDLE.unlink(missing_ok=True)
    return {
        "corruption": "head_parameters.W_align.weight removed",
        "rejected": rejected,
        "message": message,
        "meaning": "module drift must raise, never load quietly",
    }


def bootstrap_delta(align_rows, control_rows, key: str, rounds: int, seed: int) -> dict:
    def value(row: dict) -> float:
        if key == "completed":
            return 1.0 if row["completed"] else 0.0
        if key == "route_miss":
            return 1.0 if row["bucket"] == "route_miss" else 0.0
        if key == "object_start":
            return 1.0 if row["object"]["start_ok"] else 0.0
        raise ValueError(key)

    if len(align_rows) != len(control_rows):
        raise RuntimeError("paired bootstrap needs equal-length arms")
    align = [value(row) for row in align_rows]
    control = [value(row) for row in control_rows]
    if not align:
        return {"count": 0}
    generator = random.Random(seed)
    count = len(align)
    deltas = []
    for _round in range(rounds):
        picks = [generator.randrange(count) for _index in range(count)]
        left = statistics.fmean(align[index] for index in picks)
        right = statistics.fmean(control[index] for index in picks)
        deltas.append(left - right)
    deltas.sort()
    alpha = 0.025
    low = deltas[max(0, int(round(alpha * (rounds - 1))))]
    high = deltas[min(rounds - 1, int(round((1 - alpha) * (rounds - 1))))]
    point = statistics.fmean(align) - statistics.fmean(control)
    return {
        "metric": key,
        "count": count,
        "align_rate": statistics.fmean(align),
        "control_rate": statistics.fmean(control),
        "delta": point,
        "ci_low": low,
        "ci_high": high,
        "excludes_zero": low > 0.0 or high < 0.0,
        "gain_is_real": low > 0.0,
        "rounds": rounds,
    }


def six_cell_price() -> dict:
    training = SIX_CELL_MEASURED["training_cpu"]
    probes = SIX_CELL_MEASURED["step150_probe_cpu"] + SIX_CELL_MEASURED["step300_probe_cpu"]
    overhead = 500.0
    ideal = training + probes + overhead
    staged_ideal = training * 100 / 300 + 6 * 2 * 110.0 + 200.0
    return {
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
            "the six-cell rerun stays blocked until the owner rules; no branch of this package "
            "may imply a release"
        ),
    }


def budget_block() -> dict:
    ideal = sum(value for _name, value in BUDGET_ITEMS)
    return {
        "functional_items": [{"item": name, "cpu": value} for name, value in BUDGET_ITEMS],
        "functional_ideal_cpu": ideal,
        "safety_factor": SAFETY_FACTOR,
        "budget_cpu": ideal * SAFETY_FACTOR,
        "cap_cpu": BUDGET_CAP,
        "under_cap": ideal * SAFETY_FACTOR <= BUDGET_CAP,
        "rule": "budget = functional item list x 3.3; 19.2f 100-150 vs 500.5 actual is the source",
    }


def gates_view(before: dict, after: dict) -> dict:
    new_before = before["new_description_96"]
    new_after = after["new_description_96"]
    old_after = after["old_description_96"]
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
            "after": after["swap_new"]["pairs_argmax_changed"],
            "of": after["swap_new"]["pairs"],
            "threshold": f">={ARGMAX_GATE}/20",
            "pass": after["swap_new"]["pairs_argmax_changed"] >= ARGMAX_GATE,
        },
        "frozen_path_eos_frame": {
            "before": frames_before,
            "after": frames_after,
            "threshold": "no regression",
            "pass": frames_after <= frames_before,
        },
    }


def branch_view(gates: dict, delta: dict, saturated: bool) -> dict:
    all_pass = all(gate["pass"] for gate in gates.values())
    old_regressed = not gates["sentence_old_description_96"]["pass"]
    gain_real = bool(delta.get("gain_is_real", False))
    control_equivalent = not delta.get("excludes_zero", False) or not gain_real
    if old_regressed:
        branch = "negative_old_capability_sacrificed"
    elif all_pass and gain_real:
        branch = "sequence_surface_alignment_opens_object_side"
    elif control_equivalent:
        branch = "gain_is_extra_training"
    else:
        branch = "no_dose_extension"
    return {
        "branch": branch,
        "all_gates_pass": all_pass,
        "old_description_regressed": old_regressed,
        "delta_gain_is_real": gain_real,
        "control_equivalent": control_equivalent,
        "top1_still_saturated": saturated,
        "rule": (
            "full pass needs every gate and a positive Delta whose bootstrap interval excludes 0; "
            "otherwise an equivalent control arm means the gain is just the extra dose"
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
    if len(dev) != 192 or len(new_fixtures) != 96:
        raise RuntimeError("dev stratification drift")
    _easy_pairs, new_pairs = wall.pair_fixtures(dev)
    if len(new_pairs) != PAIRS_PER_STRATUM:
        raise RuntimeError("new-description swap pair drift")
    batches = schedule(train, UPDATES)

    parent = load_routing_bundle(PARENT)[0]
    parent_identity = candidate_identity(parent)
    parent_head_nan = sum(
        int(torch.isnan(value).sum()) for value in parameter_tensors(parent).values()
    )
    if parent_head_nan:
        raise RuntimeError("parent carries NaN parameters (D16-6 discipline)")

    aligned = build_alignment_candidate(parent)
    aligned_identity = candidate_identity(aligned)
    if aligned_identity["candidate_sha_v2"] == parent_identity["candidate_sha_v2"]:
        raise RuntimeError("alignment head must change candidate_sha_v2")

    # Built while the binding is still declared trainable: the reload path always
    # calls ``configure_trainable`` before ``make_routing_optimizer``, so the saved
    # optimizer group must have the same 79 tensors.
    align_optimizer = make_routing_optimizer(aligned)

    head_count = sum(parameter.numel() for parameter in aligned.head.parameters())
    span_count = sum(parameter.numel() for parameter in aligned.span_head.parameters())
    parent_head_count = sum(parameter.numel() for parameter in parent.head.parameters())

    # ---- D16-8 activity pre-check, before any Adam step -------------------
    activity_gradient = gradient_probe(aligned, batches[0], "19.2h-activity")
    activity_need_fix = need_fix_share(aligned, new_fixtures, "19.2h-needfix")
    activity_pass = activity_gradient["non_zero_alignment"] and activity_need_fix["pass"]

    before_gradient = None
    before_need_fix = None
    if not activity_pass:
        before_gradient = gradient_probe(parent, batches[0], "19.2h-activity-parent")
        before_need_fix = need_fix_share(parent, new_fixtures, "19.2h-needfix-parent")

    if not activity_pass:
        price = six_cell_price()
        budget = budget_block()
        cpu = time.process_time() - start_cpu
        wall_time = time.perf_counter() - start_wall
        report = {
            "schema": "taiji-package19-2h-sequence-surface-v1",
            "step": "N19.2h sequence-surface alignment",
            "date": "2026-11-04",
            "activity_precheck": {
                "gradient": activity_gradient,
                "need_fix": activity_need_fix,
                "pass": False,
                "parent_gradient": before_gradient,
                "parent_need_fix": before_need_fix,
            },
            "decision": {
                "branch": "activity_precheck_failed",
                "rule": "D16-8: no update is allowed when the block carries no gradient or <20% needs fixing",
            },
            "six_cell_repricing": price,
            "budget": {**budget, "spent_cpu": cpu, "over_budget": cpu > budget["budget_cpu"]},
            "cost": {
                "cpu_seconds": cpu,
                "wall_seconds": wall_time,
                "adam_updates": 0,
                "independent_trajectories": 0,
                "bundles_saved": 0,
                "confirm_reads": 0,
            },
        }
        manifest = {
            "schema": "taiji-package19-2h-sequence-surface-manifest-v1",
            "step": "N19.2h",
            "date": "2026-11-04",
            "report": str(REPORT.relative_to(ROOT)),
            "branch": "activity_precheck_failed",
            "adam_updates": 0,
            "gradient_non_zero": activity_gradient["non_zero_alignment"],
            "need_fix_share": activity_need_fix["share"],
            "cost_cpu": cpu,
        }
        write_json(REPORT, report)
        write_json(MANIFEST, manifest)
        print(json.dumps({"branch": "activity_precheck_failed", "cpu": round(cpu, 1)}))
        return 0

    # ---- shared parent snapshots ----------------------------------------
    before = full_snapshot(parent, dev, new_fixtures, new_pairs, "19.2h-before")

    # ---- two equal-dose trajectories -------------------------------------
    align_log, align_updates = train_arm(aligned, align_optimizer, batches, "19.2h-align")

    control = load_routing_bundle(PARENT)[0]
    control.configure_trainable()
    control_identity = candidate_identity(control)
    control_optimizer = make_routing_optimizer(control)
    control_log, control_updates = train_arm(control, control_optimizer, batches, "19.2h-control")

    total_adam = align_updates + control_updates
    if total_adam > ADAM_CAP:
        raise RuntimeError("adam cap exceeded")

    after_align = full_snapshot(aligned, dev, new_fixtures, new_pairs, "19.2h-after")
    after_control_new = subset_snapshot(control, new_fixtures)

    # ---- attribution: Delta = 19.2h - CE-only, paired bootstrap ----------
    align_rows = after_align["rows_new96"]
    control_rows = after_control_new["rows_new96"]
    delta = {
        "completed": bootstrap_delta(
            align_rows, control_rows, "completed", BOOTSTRAP_ROUNDS, BOOTSTRAP_SEED
        ),
        "route_miss": bootstrap_delta(
            align_rows, control_rows, "route_miss", BOOTSTRAP_ROUNDS, BOOTSTRAP_SEED + 1
        ),
        "object_start": bootstrap_delta(
            align_rows, control_rows, "object_start", BOOTSTRAP_ROUNDS, BOOTSTRAP_SEED + 2
        ),
    }

    # ---- checkpoint, identity and drift preflight -----------------------
    rescue = identity_rescue(aligned, dev[:RESCUE_EXAMPLES], "align")
    preflight = checkpoint_preflight(aligned, align_optimizer, dev[0])
    if not preflight["round_trip_ok"]:
        raise RuntimeError("checkpoint dry-run did not round-trip")
    drift = module_drift_rejection(aligned, dev[0])

    fixed_state = module.prepared_pair(aligned, dev[0], "19.2h-save", "save")[0].state
    saved = save_routing_bundle(
        FINAL_BUNDLE,
        aligned,
        align_optimizer,
        UPDATES,
        {
            "stage": "N19.2h",
            "seed": SEED,
            "arm": ARM,
            "update": UPDATES,
            "loss": "selection_ce_plus_span",
            "graph": "alignment_block",
        },
        aligned_identity,
        fixed_state,
    )
    reloaded, _optimizer, _state, reload_index, reload_cursor = load_routing_bundle(FINAL_BUNDLE)
    reload_diff = 0.0
    restored = parameter_tensors(reloaded)
    trained = parameter_tensors(aligned)
    for name, value in trained.items():
        reload_diff = max(reload_diff, float((value - restored[name]).abs().max()))

    gates = gates_view(before, after_align)
    saturated = (
        after_align["scores_new96"]["top1_probability"]["median"] >= SATURATION_TOP1
        and after_align["swap_new"]["pairs_argmax_changed"] < ARGMAX_GATE
    )
    decision = branch_view(gates, delta["completed"], saturated)
    price = six_cell_price()
    budget = budget_block()
    cpu = time.process_time() - start_cpu
    wall_time = time.perf_counter() - start_wall

    report = {
        "schema": "taiji-package19-2h-sequence-surface-v1",
        "step": "N19.2h sequence-surface alignment (+1,320 parameters)",
        "date": "2026-11-04",
        "purpose": (
            "test whether a parameterised local cross-correlation over the pooled encoding's "
            "four subspaces opens the object-side selection that 19.2f/19.2g could not move"
        ),
        "design": {
            "parent": str(PARENT.relative_to(ROOT)),
            "parent_arm": ARM,
            "parent_update": 300,
            "parent_old96": before["old_description_96"]["completed"],
            "parent_new96": before["new_description_96"]["completed"],
            "block": (
                "W_align 32x32 shared projection over four 32-d subspaces, cosine agreement at "
                "five offsets (-2..+2), 5x8 offset embedding, four extra W_hidden row inputs"
            ),
            "new_parameters": head_count - parent_head_count,
            "head_parameters": head_count,
            "span_parameters": span_count,
            "graph_change": "selection head only; binding, transport, pointer and decode untouched",
            "loss": "selection CE + inherited span term (no margin term)",
            "arms": {
                "19.2h": "alignment head, 30 Adam updates",
                "CE-only": "parent head verbatim, 30 Adam updates, same batches",
            },
            "stage": "selection only",
            "updates_per_arm": UPDATES,
            "adam_cap": ADAM_CAP,
            "batch": 8,
            "order_seed": premise.ORDER_SEED,
            "formal_eval": "192 examples, exact 27-byte action + EOS, free decode",
            "swap_pairs_new": PAIRS_PER_STRATUM,
            "saves": 1,
            "confirm_reads": 0,
        },
        "activity_precheck": {
            "gradient": activity_gradient,
            "need_fix": activity_need_fix,
            "pass": activity_pass,
            "gate_rule": "non-zero gradient on every alignment tensor and need-fix share >= 0.20",
        },
        "identities": {
            "parent_candidate_sha_v2": parent_identity["candidate_sha_v2"],
            "aligned_candidate_sha_v2": aligned_identity["candidate_sha_v2"],
            "control_candidate_sha_v2": control_identity["candidate_sha_v2"],
            "changed": aligned_identity["candidate_sha_v2"] != parent_identity["candidate_sha_v2"],
        },
        "identity_preflight": {
            "rescue": rescue,
            "save_restore": preflight,
            "module_drift": drift,
            "three_of_three": rescue["all_agree"]
            and preflight["round_trip_ok"]
            and drift["rejected"],
        },
        "before": before,
        "after": after_align,
        "control_after": after_control_new,
        "training_log": {"align": align_log, "control": control_log},
        "delta": delta,
        "gates": gates,
        "decision": decision,
        "six_cell_repricing": price,
        "budget": {**budget, "spent_cpu": cpu, "over_budget": cpu > budget["budget_cpu"]},
        "saved_bundle": {
            **saved,
            "reload_max_abs_diff": reload_diff,
            "reload_update_index": reload_index,
            "reload_cursor": dict(reload_cursor),
        },
        "attribution_rules": [
            "gamma is the generation weight and 1 - gamma the copy weight",
            "gamma never implies training credit (M3)",
            "the alignment block is a graph change; identity must change and the new bundle stands",
            "gain is reported as Delta = 19.2h - CE-only with a paired bootstrap interval",
        ],
        "cost": {
            "cpu_seconds": cpu,
            "wall_seconds": wall_time,
            "adam_updates": total_adam,
            "independent_trajectories": 2,
            "bundles_saved": 1,
            "confirm_reads": 0,
        },
    }
    manifest = {
        "schema": "taiji-package19-2h-sequence-surface-manifest-v1",
        "step": "N19.2h",
        "date": "2026-11-04",
        "report": str(REPORT.relative_to(ROOT)),
        "parent": str(PARENT.relative_to(ROOT)),
        "candidate_sha_v2_parent": parent_identity["candidate_sha_v2"],
        "candidate_sha_v2_aligned": aligned_identity["candidate_sha_v2"],
        "candidate_sha_v2_control": control_identity["candidate_sha_v2"],
        "identity_preflight_ok": report["identity_preflight"]["three_of_three"],
        "activity_gradient_non_zero": activity_gradient["non_zero_alignment"],
        "activity_need_fix_share": activity_need_fix["share"],
        "activity_need_fix_wrong": activity_need_fix["wrong"],
        "adam_updates": total_adam,
        "bundles_saved": 1,
        "saved_bundle": saved["path"],
        "confirm_reads": 0,
        "delta_completed": delta["completed"]["delta"],
        "delta_completed_ci": [delta["completed"]["ci_low"], delta["completed"]["ci_high"]],
        "delta_route_miss": delta["route_miss"]["delta"],
        "delta_object_start": delta["object_start"]["delta"],
        "gates": gates,
        "branch": decision["branch"],
        "route_miss_before_count": before["new_description_96"]["bucket_counts"]["route_miss"],
        "route_miss_after_count": after_align["new_description_96"]["bucket_counts"]["route_miss"],
        "swap_argmax_before": before["swap_new"]["pairs_argmax_changed"],
        "swap_argmax_after": after_align["swap_new"]["pairs_argmax_changed"],
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
                "need_fix": f"{activity_need_fix['wrong']}/{activity_need_fix['examples']}",
                "grad": {
                    key: round(value, 6)
                    for key, value in activity_gradient["per_tensor_norm"].items()
                },
                "delta_completed": round(delta["completed"]["delta"], 4),
                "delta_ci": [
                    round(delta["completed"]["ci_low"], 4),
                    round(delta["completed"]["ci_high"], 4),
                ],
                "route_miss": [
                    manifest["route_miss_before_count"],
                    manifest["route_miss_after_count"],
                ],
                "swap": [manifest["swap_argmax_before"], manifest["swap_argmax_after"]],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
