"""17.5 three-arm paired comparison on the frozen 17.0 routing graph.

Frozen dose: 2 seeds (28031, 28032) x 3 arms (A_route_lock, B_stop_lock,
C_unlocked) x 300 updates, batch 8, stages selection (1-100), copy (101-200),
joint (201-300).  One Adam over the declared trainable groups; frozen groups
keep grad=None and their Adam moments are not reset at stage changes.

Four pre-registered design choices (DESIGN §15) are active for every arm:

* pointer temperature T(u) = 1 - 0.95*(u-1)/99 for u<=100, then 0.05;
* pointer entropy bonus 0.01*T(u) on the selected record's pointer row;
* pathwise/ST credit only (no REINFORCE baseline), grad-norm variance logged;
* teacher-span supervision (weight 0.5) on the object and place records.

The self-prefix curriculum keeps the frozen per-batch rule: teacher-forced
completion rate >= 0.5 triggers it, never the update index.  The input prefix
and the target labels stay separated (target = original 27 bytes + EOS).

Dev evaluation is at step 0 and step 300 only; step 150 saves a bundle and runs
a 40-example readiness probe.  No confirm read.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

torch.set_num_threads(1)

from taiji.binding_copy import initial_binding_state, prepare  # noqa: E402
from taiji.binding_copy_scoring import score_generation  # noqa: E402
from taiji.candidate_identity import candidate_identity  # noqa: E402
from taiji.living_core import EOS  # noqa: E402
from taiji.record_relation_core import MAX_PREDICTIONS, decode_endpoint  # noqa: E402
from taiji.selection_routing import (  # noqa: E402
    ARMS,
    _advanced_state,
    build_context,
    observe_goal,
    resolve_prefix,
    route_labels,
    routing_generate,
    routing_loss,
    routing_step,
    selection_ce,
    span_loss,
    teacher_span,
)
from taiji.selection_routing_checkpoint import (  # noqa: E402
    build_routing_candidate,
    load_routing_bundle,
    save_routing_bundle,
)

COMPARE_SCRIPT = Path("scripts/training/verify_taiji_binding_select_16_1_compare.py")
PARENTS = {
    28031: Path("output/taiji_core_binding_select_16_1_compare/bundles/28031-A-locked-step300.pt"),
    28032: Path("output/taiji_core_binding_select_16_1_compare/bundles/28032-A-locked-step300.pt"),
}
ORDER_SEEDS = {28031: 38031, 28032: 38032}
SEEDS = (28031, 28032)

BATCH = 8
UPDATES = 300
LR = 1e-3
CLIP = 1.0
CHECKPOINT_UPDATE = 150
CURRICULUM_THRESHOLD = 0.5
STAGES = (("selection", 1, 100), ("copy", 101, 200), ("joint", 201, 300))

T_START = 1.0
T_MIN = 0.05
T_DECAY_UPDATES = 100
ENTROPY_COEF = 0.01
SPAN_LAMBDA = 0.5

OUTPUT_DIR = Path("output/taiji_selection_routing_17_5")
REPORT = Path("reports/taiji_selection_routing_17_5_compare_20261104.json")
MANIFEST = Path("plans/manifests/taiji_selection_routing_17_5_compare_20261104.json")
THRESHOLDS = Path("plans/manifests/taiji_selection_routing_17_3_r1_thresholds_20261103.json")

DEV = 240
POSITION_PROBE = 40
CONTENT_SAMPLE_MODULO = 10
PRIOR_CPU = 1584.7
PRIOR_WALL = 1811.0
PRIOR_BREAKDOWN = {
    "before_window": {"cpu": 552.8, "wall": 740.2},
    "verify_17_3_recent": {"cpu": 48.8, "wall": 51.2},
    "verify_17_3_two_invalid_runs": {"cpu": 105.0, "wall": 109.5},
    "engineering_pilot_20": {"cpu": 52.5, "wall": 60.0},
    "engineering_pilot_220": {"cpu": 825.6, "wall": 850.0},
}
ENGINEERING_ADAM = 240 + 18
WATCH_CPU = 20000.0
WATCH_WALL = 30000.0

PLAN_GATES = {
    "overall": 0.80,
    "unseen_output_name": 0.80,
    "new_description": 0.80,
    "reversed": 0.95,
    "selection_object": 0.90,
    "selection_place": 0.90,
    "content_drop": 0.10,
    "identity_rescue": 8,
    "a_minus_c": 0.10,
}


def sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_compare_module():
    spec = importlib.util.spec_from_file_location("cmp17c", ROOT / COMPARE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_MODULE: dict[str, object] = {}


def compare_module():
    if "module" not in _MODULE:
        _MODULE["module"] = load_compare_module()
    return _MODULE["module"]


def load_fixtures() -> tuple[tuple[dict, ...], tuple[dict, ...]]:
    if "fixtures" not in _MODULE:
        _MODULE["fixtures"] = compare_module().load_data()[:2]
    return _MODULE["fixtures"]


def schedule(seed: int, train: tuple[dict, ...], updates: int) -> list[tuple[dict, ...]]:
    batches: list[tuple[dict, ...]] = []
    epoch = 0
    generator = torch.Generator().manual_seed(ORDER_SEEDS[seed])
    permutation = torch.randperm(len(train) - len(train) % BATCH, generator=generator).tolist()
    cursor = 0
    while len(batches) < updates:
        if cursor + BATCH > len(permutation):
            epoch += 1
            generator = torch.Generator().manual_seed(ORDER_SEEDS[seed] + epoch)
            permutation = torch.randperm(
                len(train) - len(train) % BATCH, generator=generator
            ).tolist()
            cursor = 0
        batches.append(tuple(train[index] for index in permutation[cursor : cursor + BATCH]))
        cursor += BATCH
    return batches


def stage_of(update: int) -> str:
    for name, low, high in STAGES:
        if low <= update <= high:
            return name
    raise ValueError(f"update {update} outside the frozen stages")


def temperature(update: int) -> float:
    if update > T_DECAY_UPDATES:
        return T_MIN
    return T_START - (T_START - T_MIN) * (update - 1) / (T_DECAY_UPDATES - 1)


def entropy_coefficient(update: int) -> float:
    return ENTROPY_COEF * temperature(update)


def subject_tag(tag: str, fixture: dict, role: str) -> str:
    return f"{tag}-{role}-{fixture['world']}-{fixture['index']}"


def prepared_pair(candidate, fixture: dict, tag: str, role: str):
    binding = candidate.binding
    goal = fixture["goal"].encode("utf-8")
    state = initial_binding_state(binding, subject_tag(tag, fixture, role))
    prepared = prepare(binding, state, goal, fixture["records"])
    context = build_context(binding, goal, tuple(fixture["record_keys"]))
    return prepared, context


def place_index(fixture: dict) -> int:
    for index, record in enumerate(fixture["raw_example"]["records"]):
        if record["kind"] == "place" and record["description"] == fixture["place_description"]:
            return index
    raise RuntimeError("place record not found")


def span_term(prepared, fixture: dict, span_head) -> tuple[torch.Tensor, dict]:
    """Start/end CE on the object and place records, weight 0.5 each."""
    view = prepared.base.view
    object_row = int(fixture["selection_label"])
    place_row = place_index(fixture)
    object_raw = fixture["raw_example"]["records"][object_row]["raw"].encode("utf-8")
    place_raw = fixture["raw_example"]["records"][place_row]["raw"].encode("utf-8")
    object_span = teacher_span(object_raw, fixture["object_name"].encode("utf-8"))
    place_span = teacher_span(place_raw, fixture["place_name"].encode("utf-8"))
    object_loss = span_loss(span_head, view.units[object_row], object_span[0], object_span[1])[
        "loss"
    ]
    place_loss = span_loss(span_head, view.units[place_row], place_span[0], place_span[1])["loss"]
    return SPAN_LAMBDA * (object_loss + place_loss), {
        "object_span": list(object_span),
        "place_span": list(place_span),
    }


def teacher_prefix(fixture: dict) -> tuple[int, ...]:
    return tuple(fixture["target_prefix"])


def self_prefix(candidate, fixture: dict, arm: str, update: int, tag: str) -> tuple[int, ...]:
    prepared, context = prepared_pair(candidate, fixture, tag, "self")
    with torch.no_grad():
        generated = routing_generate(
            candidate.binding,
            prepared,
            context,
            candidate.head,
            arm=arm,
            span_head=candidate.span_head,
            pointer_temperature=1.0,
            pointer_hard=True,
        )
    own = tuple(generated["raw_bytes"])[: len(fixture["target_prefix"])]
    if len(own) < len(fixture["target_prefix"]):
        own = own + (EOS,) * (len(fixture["target_prefix"]) - len(own))
    return own


def teacher_forced_completion(candidate, fixture: dict, arm: str, update: int, tag: str) -> bool:
    prepared, context = prepared_pair(candidate, fixture, tag, "tf")
    state = prepared.state
    state, gate_index = observe_goal(candidate.binding, prepared, gate_index=0)
    previous = EOS
    raw: list[int] = []
    terminated = False
    with torch.no_grad():
        for _prediction in range(MAX_PREDICTIONS):
            step = routing_step(
                candidate.binding,
                prepared,
                state,
                previous,
                context,
                candidate.head,
                arm=arm,
                gate_index=gate_index,
                span_head=candidate.span_head,
                pointer_temperature=temperature(update),
                pointer_hard=True,
            )
            endpoint = decode_endpoint(step.probabilities)
            if endpoint["eos"]:
                terminated = True
                break
            raw.append(endpoint["byte"])
            if len(raw) > 64:
                break
            state, _, gate_index = _advanced_state(
                candidate.binding, prepared, step, endpoint["byte"]
            )
            previous = endpoint["byte"]
    return score_generation(bytes(raw), fixture["target_prefix"], terminated)["completed"]


def batch_loss(
    candidate, batch, arm: str, stage: str, use_self_prefix: bool, update: int, tag: str
):
    span_values = []
    entropy_values = []
    nll_values = []
    ce_values = []
    for fixture in batch:
        prepared, context = prepared_pair(candidate, fixture, tag, "train")
        routes = route_labels(fixture)
        span, _spans = span_term(prepared, fixture, candidate.span_head)
        span_values.append(span)
        if stage == "selection":
            output = selection_ce(
                candidate.binding,
                prepared,
                teacher_prefix(fixture),
                tuple(fixture["target_bytes"]),
                routes[0],
                routes[1],
                context,
                candidate.head,
            )
            ce_values.append(output["loss"])
            continue
        prefix = (
            self_prefix(candidate, fixture, arm, update, tag)
            if use_self_prefix
            else teacher_prefix(fixture)
        )
        output = routing_loss(
            candidate.binding,
            prepared,
            prefix,
            tuple(fixture["target_bytes"]),
            routes[0],
            routes[1],
            context,
            candidate.head,
            arm=arm,
            span_head=candidate.span_head,
            pointer_temperature=temperature(update),
            pointer_hard=False,
            pointer_entropy_coef=entropy_coefficient(update),
        )
        nll_values.append(output["nll"])
        entropy_values.append(output["entropy"].detach())
        if stage == "joint":
            ce_values.append(output["ce"])
    span_total = torch.stack(span_values).mean()
    if stage == "selection":
        loss = torch.stack(ce_values).mean() + span_total
        return loss, {
            "stage": stage,
            "ce": float(torch.stack(ce_values).detach().mean()),
            "nll": 0.0,
            "span": float(span_total.detach()),
            "entropy": 0.0,
            "steps": 0,
        }
    nll_total = torch.stack(nll_values).mean()
    loss = nll_total + span_total
    entropy = 0.0
    if entropy_values:
        entropy_tensor = torch.stack(entropy_values).mean()
        entropy = float(entropy_tensor)
        loss = loss - entropy_coefficient(update) * entropy_tensor
    ce = 0.0
    if ce_values:
        ce_tensor = torch.stack(ce_values).mean()
        ce = float(ce_tensor.detach())
        loss = loss + ce_tensor
    return loss, {
        "stage": stage,
        "ce": ce,
        "nll": float(nll_total.detach()),
        "span": float(span_total.detach()),
        "entropy": entropy,
        "steps": len(batch) * len(next(iter(batch))["target_bytes"]),
    }


def set_trainable(candidate, stage: str) -> list[torch.nn.Parameter]:
    binding_trainable = list(candidate.binding.trainable_parameters())
    for parameter in candidate.binding.parameters():
        parameter.requires_grad_(False)
    for parameter in binding_trainable:
        parameter.requires_grad_(stage in ("copy", "joint"))
    for parameter in candidate.head.parameters():
        parameter.requires_grad_(stage in ("selection", "joint"))
    for parameter in candidate.span_head.parameters():
        parameter.requires_grad_(True)
    return [
        parameter
        for module in (candidate.binding, candidate.head, candidate.span_head)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]


def group_norm(parameters: list[torch.nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().pow(2).sum())
    return math.sqrt(total)


def grad_norms(candidate) -> dict[str, float]:
    head = group_norm([p for p in candidate.head.parameters() if p.requires_grad])
    span = group_norm([p for p in candidate.span_head.parameters() if p.requires_grad])
    binding = group_norm([p for p in candidate.binding.trainable_parameters() if p.requires_grad])
    return {"head": head, "span": span, "binding": binding}


class ForcedRouteHead:
    """Test-only head: same scoring interface, one-hot at a chosen route index.

    The production graph never accepts route labels; the intervention lives in
    this diagnostic wrapper so the T1 input whitelist stays intact.
    """

    def __init__(self, index: int, config) -> None:
        self.index = int(index)
        self.config = config

    def score_rows(self, goal_vec, key_vecs, s_t, previous_embedding):
        scores = torch.full(
            (len(key_vecs) + 1,), -1e9, dtype=goal_vec.dtype, device=goal_vec.device
        )
        scores[self.index] = 1e9
        return scores


def generate(
    candidate,
    fixture: dict,
    arm: str,
    *,
    records=None,
    keys=None,
    forced_route: int | None = None,
    offset_zero: bool = False,
    tag: str,
) -> dict:
    binding = candidate.binding
    goal = fixture["goal"].encode("utf-8")
    recs = fixture["records"] if records is None else records
    fragments = fixture["record_keys"] if keys is None else keys
    state = initial_binding_state(binding, f"{tag}-gen-{fixture['world']}-{fixture['index']}")
    prepared = prepare(binding, state, goal, recs)
    context = build_context(binding, goal, tuple(fragments))
    bias = None
    if offset_zero and prepared.base.view.record_count:
        bias = tuple(
            torch.cat(
                [
                    torch.full((1,), 1e9, dtype=torch.float64),
                    torch.full(
                        (prepared.base.view.units[index].shape[0] - 1,),
                        -1e9,
                        dtype=torch.float64,
                    ),
                ]
            )
            for index in range(prepared.base.view.record_count)
        )
    with torch.no_grad():
        route_head = (
            candidate.head
            if forced_route is None
            else ForcedRouteHead(forced_route, candidate.head.config)
        )
        return routing_generate(
            binding,
            prepared,
            context,
            route_head,
            arm=arm,
            span_head=candidate.span_head,
            pointer_temperature=1.0,
            pointer_hard=True,
            forced_pointer_bias=bias,
        )


def summarize(rows: list[dict]) -> dict:
    flag_keys = ("unseen_combination", "unseen_output_name", "new_description", "both_new")
    denominators = {key: sum(1 for row in rows if row["flags"][key]) for key in flag_keys}
    subsets = {
        key: (
            sum(1 for row in rows if row["flags"][key] and row["score"]["completed"])
            / denominators[key]
            if denominators[key]
            else 0.0
        )
        for key in flag_keys
    }
    by_world: dict[str, dict] = {}
    for row in rows:
        entry = by_world.setdefault(str(row["world"]), {"n": 0, "completed": 0})
        entry["n"] += 1
        entry["completed"] += int(row["score"]["completed"])
    return {
        "completed": sum(1 for row in rows if row["score"]["completed"]) / len(rows),
        "subsets": subsets,
        "subset_denominators": denominators,
        "utf8": sum(1 for row in rows if row["score"]["utf8"]) / len(rows),
        "eos": sum(1 for row in rows if row["score"]["terminated"]) / len(rows),
        "full_sentence": sum(1 for row in rows if row["score"]["full_sentence"]) / len(rows),
        "by_world": by_world,
        "examples": len(rows),
    }


def gamma_stats(per_example: list[list[float]]) -> dict:
    steps = sum(len(rows) for rows in per_example)
    flat = [value for rows in per_example for value in rows]
    return {
        "examples": len(per_example),
        "steps": steps,
        "gamma_effective_mean": sum(flat) / steps if steps else 0.0,
        "gamma_effective_min": min(flat) if flat else 0.0,
        "gamma_effective_max": max(flat) if flat else 0.0,
        "steps_below_floor": sum(1 for value in flat if value < 0.2),
        "floored_fraction": (sum(1 for value in flat if value < 0.2) / steps if steps else 0.0),
    }


def selection_probe(candidate, dev, arm: str, tag: str) -> dict:
    """Teacher-prefix route accuracy at the object start (3) and place start (21)."""
    binding = candidate.binding
    object_rows = []
    place_rows = []
    for fixture in dev:
        prepared, context = prepared_pair(candidate, fixture, tag, "sel")
        state = prepared.state
        state, gate_index = observe_goal(binding, prepared, gate_index=0)
        previous = EOS
        prefix = resolve_prefix(teacher_prefix(fixture), len(fixture["target_bytes"]) - 1)
        object_index = int(fixture["selection_label"])
        place_row = place_index(fixture)
        chosen_at = {}
        with torch.no_grad():
            for t, _label in enumerate(fixture["target_bytes"]):
                step = routing_step(
                    binding,
                    prepared,
                    state,
                    previous,
                    context,
                    candidate.head,
                    arm=arm,
                    gate_index=gate_index,
                    span_head=candidate.span_head,
                    pointer_temperature=1.0,
                    pointer_hard=True,
                )
                chosen_at[t] = int(torch.argmax(step.scores))
                if t == len(fixture["target_bytes"]) - 1:
                    break
                accepted = prefix[t]
                state, _, gate_index = _advanced_state(binding, prepared, step, accepted)
                previous = accepted
        object_rows.append(
            {
                "world": fixture["world"],
                "index": fixture["index"],
                "chosen": chosen_at[3],
                "label": object_index,
                "correct": chosen_at[3] == object_index,
            }
        )
        place_rows.append(
            {
                "world": fixture["world"],
                "index": fixture["index"],
                "chosen": chosen_at[21],
                "label": place_row,
                "correct": chosen_at[21] == place_row,
            }
        )
    return {
        "examples": len(object_rows),
        "object_top1": sum(1 for row in object_rows if row["correct"]) / len(object_rows),
        "place_top1": sum(1 for row in place_rows if row["correct"]) / len(place_rows),
        "rows": {"object": object_rows, "place": place_rows},
    }


def evaluate_dev(candidate, dev, arm: str, tag: str, full: bool) -> dict:
    rows = []
    gamma_per_example = []
    for fixture in dev:
        generated = generate(candidate, fixture, arm, tag=tag)
        scored = score_generation(
            generated["raw_bytes"], fixture["target_prefix"], generated["terminated"]
        )
        rows.append(
            {
                "index": fixture["index"],
                "world": fixture["world"],
                "flags": fixture["flags"],
                "score": scored,
                "raw": generated["raw_bytes"].decode("utf-8", "replace"),
            }
        )
        gamma_per_example.append(generated["gamma"])
    out = {
        "overall": summarize(rows),
        "gamma": gamma_stats(gamma_per_example),
        "selection": selection_probe(candidate, dev, arm, tag),
        "examples": len(dev),
    }
    if not full:
        return out
    true_gen = {(row["world"], row["index"]): row["raw"] for row in rows}
    reversed_rows = []
    for fixture in dev:
        generated = generate(
            candidate,
            fixture,
            arm,
            records=tuple(reversed(fixture["records"])),
            keys=list(reversed(fixture["record_keys"])),
            tag=tag,
        )
        scored = score_generation(
            generated["raw_bytes"], fixture["target_prefix"], generated["terminated"]
        )
        reversed_rows.append(
            {
                "index": fixture["index"],
                "world": fixture["world"],
                "completed": scored["completed"],
                "paired": generated["raw_bytes"].decode("utf-8", "replace")
                == true_gen.get((fixture["world"], fixture["index"]), ""),
            }
        )
    out["reversed"] = {
        "accuracy": sum(1 for row in reversed_rows if row["completed"]) / len(reversed_rows),
        "paired_rate": sum(1 for row in reversed_rows if row["paired"]) / len(reversed_rows),
        "examples": len(reversed_rows),
    }
    module = compare_module()
    counter_rows = []
    for fixture in dev:
        swapped = module.swap_object_name(fixture)
        if swapped is None:
            continue
        generated = generate(candidate, fixture, arm, records=swapped["records"], tag=tag)
        scored = score_generation(
            generated["raw_bytes"], swapped["target_prefix"], generated["terminated"]
        )
        counter_rows.append(
            {
                "index": fixture["index"],
                "world": fixture["world"],
                "counterfactual_completed": scored["completed"],
                "true_completed": next(
                    row["score"]["completed"]
                    for row in rows
                    if row["world"] == fixture["world"] and row["index"] == fixture["index"]
                ),
            }
        )
    out["counterfactual"] = {
        "applicable": len(counter_rows),
        "both_correct": (
            sum(
                1
                for row in counter_rows
                if row["counterfactual_completed"] and row["true_completed"]
            )
            / len(counter_rows)
            if counter_rows
            else 0.0
        ),
    }
    rescue_rows = []
    for fixture in dev[:8]:
        prepared, context = prepared_pair(candidate, fixture, tag, "rescue")
        state = prepared.state
        state, gate_index = observe_goal(candidate.binding, prepared, gate_index=0)
        with torch.no_grad():
            baseline = routing_step(
                candidate.binding,
                prepared,
                state,
                EOS,
                context,
                candidate.head,
                arm=arm,
                gate_index=gate_index,
                span_head=candidate.span_head,
                pointer_hard=True,
            )
            _intervened = routing_step(
                candidate.binding,
                prepared,
                state,
                EOS,
                context,
                ForcedRouteHead(prepared.base.view.record_count, candidate.head.config),
                arm=arm,
                gate_index=gate_index,
                span_head=candidate.span_head,
                pointer_hard=True,
            )
            restored = routing_step(
                candidate.binding,
                prepared,
                state,
                EOS,
                context,
                candidate.head,
                arm=arm,
                gate_index=gate_index,
                span_head=candidate.span_head,
                pointer_hard=True,
            )
        rescue_rows.append(
            {
                "index": fixture["index"],
                "equal": bool(torch.equal(restored.probabilities, baseline.probabilities)),
                "max_abs_diff": float(
                    (restored.probabilities - baseline.probabilities).abs().max()
                ),
            }
        )
    out["identity_rescue"] = {
        "equal": sum(1 for row in rescue_rows if row["equal"]),
        "denominator": len(rescue_rows),
        "rows": rescue_rows,
    }
    intervention_rows = []
    targets = [fixture for fixture in dev if fixture["index"] % CONTENT_SAMPLE_MODULO == 0]
    for fixture in targets:
        true_row = next(
            row["score"]["completed"]
            for row in rows
            if row["world"] == fixture["world"] and row["index"] == fixture["index"]
        )
        no_source = generate(candidate, fixture, arm, records=(), keys=(), tag=tag)
        no_source_completed = score_generation(
            no_source["raw_bytes"], fixture["target_prefix"], no_source["terminated"]
        )["completed"]
        alternate = module.alternate_object_index(fixture)
        wrong_record_completed = None
        if alternate is not None:
            wrong_record = generate(candidate, fixture, arm, forced_route=alternate, tag=tag)
            wrong_record_completed = score_generation(
                wrong_record["raw_bytes"], fixture["target_prefix"], wrong_record["terminated"]
            )["completed"]
        wrong_position = generate(candidate, fixture, arm, offset_zero=True, tag=tag)
        wrong_position_completed = score_generation(
            wrong_position["raw_bytes"], fixture["target_prefix"], wrong_position["terminated"]
        )["completed"]
        intervention_rows.append(
            {
                "index": fixture["index"],
                "world": fixture["world"],
                "true_completed": true_row,
                "no_source_completed": no_source_completed,
                "wrong_record_completed": wrong_record_completed,
                "wrong_position_completed": wrong_position_completed,
            }
        )
    true_rate = sum(1 for row in intervention_rows if row["true_completed"]) / len(
        intervention_rows
    )
    drops = {}
    for name in ("no_source", "wrong_record", "wrong_position"):
        rates = [
            row[f"{name}_completed"]
            for row in intervention_rows
            if row[f"{name}_completed"] is not None
        ]
        rate = sum(1 for value in rates if value) / len(rates) if rates else 0.0
        drops[name] = {
            "completed_rate": rate,
            "drop": true_rate - rate,
            "denominator": len(rates),
        }
    out["content_interventions"] = {
        "applicable": len(intervention_rows),
        "true_completed_rate": true_rate,
        "drops": drops,
        "rows": intervention_rows,
    }
    nll_rows = []
    for fixture in dev:
        prepared, context = prepared_pair(candidate, fixture, tag, "nll")
        routes = route_labels(fixture)
        with torch.no_grad():
            output = routing_loss(
                candidate.binding,
                prepared,
                teacher_prefix(fixture),
                tuple(fixture["target_bytes"]),
                routes[0],
                routes[1],
                context,
                candidate.head,
                arm=arm,
                span_head=candidate.span_head,
                pointer_temperature=1.0,
                pointer_hard=True,
            )
        nll_rows.append(float(output["nll"]))
    out["nll"] = {"mean_nll": statistics.fmean(nll_rows), "examples": len(nll_rows)}
    return out


def readiness_probe(candidate, dev, arm: str, tag: str) -> dict:
    rows = []
    for fixture in dev[:POSITION_PROBE]:
        generated = generate(candidate, fixture, arm, tag=tag)
        scored = score_generation(
            generated["raw_bytes"], fixture["target_prefix"], generated["terminated"]
        )
        rows.append(
            {
                "index": fixture["index"],
                "world": fixture["world"],
                "object_exact": scored["object_exact"],
                "place_exact": scored["place_exact"],
                "completed": scored["completed"],
            }
        )
    selection = selection_probe(candidate, dev[:POSITION_PROBE], arm, tag)
    return {
        "n": len(rows),
        "object_exact_rate": sum(1 for row in rows if row["object_exact"]) / len(rows),
        "place_exact_rate": sum(1 for row in rows if row["place_exact"]) / len(rows),
        "completed_rate": sum(1 for row in rows if row["completed"]) / len(rows),
        "selection_object_top1": selection["object_top1"],
        "selection_place_top1": selection["place_top1"],
        "status": "readiness_only_not_task_verdict",
    }


def run_one(seed: int, arm: str, resume: bool = False) -> dict:
    tag = f"{seed}-{arm}"
    start_cpu = time.process_time()
    start_wall = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    parent = ROOT / PARENTS[seed]
    parent_sha = sha256_file(parent)
    candidate, optimizer = build_routing_candidate(parent, seed, arm)
    train, dev = load_fixtures()
    batches = schedule(seed, train, UPDATES)

    step0 = evaluate_dev(candidate, dev, arm, tag, full=False)
    resumed_from = 0
    checkpoint_path = OUTPUT_DIR / f"{tag}-step{CHECKPOINT_UPDATE}.pt"
    if resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"resume requested but {checkpoint_path} is missing")
        (
            candidate,
            optimizer,
            _state,
            resumed_from,
            _payload,
        ) = load_routing_bundle(checkpoint_path)
        candidate.configure_trainable()

    rows = []
    grad_rows = []
    for update in range(resumed_from + 1, UPDATES + 1):
        stage = stage_of(update)
        batch = batches[update - 1]
        if stage == "selection":
            use_self_prefix = False
            rate = 0.0
        else:
            completed = [
                teacher_forced_completion(candidate, fixture, arm, update, tag) for fixture in batch
            ]
            rate = sum(1 for flag in completed if flag) / len(completed)
            use_self_prefix = rate >= CURRICULUM_THRESHOLD
        parameters = set_trainable(candidate, stage)
        loss, stats = batch_loss(candidate, batch, arm, stage, use_self_prefix, update, tag)
        optimizer.zero_grad()
        loss.backward()
        norms = grad_norms(candidate)
        clip = float(torch.nn.utils.clip_grad_norm_(parameters, CLIP))
        optimizer.step()
        rows.append(
            {
                "update": update,
                "stage": stage,
                "loss": float(loss.detach()),
                "nll": stats["nll"],
                "ce": stats["ce"],
                "span": stats["span"],
                "entropy": stats["entropy"],
                "temperature": temperature(update),
                "entropy_coef": entropy_coefficient(update),
                "curriculum_rate": rate,
                "self_prefix": bool(use_self_prefix),
                "clip_scale": clip,
            }
        )
        grad_rows.append({"update": update, "stage": stage, **norms})
        if update % 10 == 0 or update == UPDATES:
            write_json(
                OUTPUT_DIR / f"progress-{tag}.json",
                {
                    "seed": seed,
                    "arm": arm,
                    "update": update,
                    "updates_total": UPDATES,
                    "stage": stage,
                    "last_loss": rows[-1]["loss"],
                    "self_prefix_updates": sum(1 for row in rows if row["self_prefix"]),
                    "cost": {
                        "cpu_seconds": time.process_time() - start_cpu,
                        "wall_seconds": time.perf_counter() - start_wall,
                    },
                },
            )
        if update == CHECKPOINT_UPDATE:
            info = save_routing_bundle(
                checkpoint_path,
                candidate,
                optimizer,
                update,
                {"stage": "N17.5", "seed": seed, "arm": arm, "update": update},
                {"sha256": parent_sha, "path": str(PARENTS[seed])},
                initial_binding_state(candidate.binding, f"ckpt-{tag}"),
            )
            readiness = readiness_probe(candidate, dev, arm, tag)
            write_json(
                OUTPUT_DIR / f"readiness-{tag}.json",
                {"seed": seed, "arm": arm, "bundle": info, "probe": readiness},
            )

    final_path = OUTPUT_DIR / f"{tag}-step{UPDATES}.pt"
    final_info = save_routing_bundle(
        final_path,
        candidate,
        optimizer,
        UPDATES,
        {"stage": "N17.5", "seed": seed, "arm": arm, "update": UPDATES},
        {"sha256": parent_sha, "path": str(PARENTS[seed])},
        initial_binding_state(candidate.binding, f"final-{tag}"),
    )
    progress_path = OUTPUT_DIR / f"progress-{tag}.json"
    progress_update = int(json.loads(progress_path.read_text(encoding="utf-8"))["update"])
    bundle_update = int(load_routing_bundle(final_path)[3])
    consistency = {
        "progress_update": progress_update,
        "bundle_update": bundle_update,
        "bundle_exists": final_path.exists(),
        "expected_update": UPDATES,
    }
    if not (final_path.exists() and progress_update == UPDATES and bundle_update == UPDATES):
        raise RuntimeError(f"artifact inconsistency for {tag}: {consistency}")
    try:
        final = evaluate_dev(candidate, dev, arm, tag, full=True)
    except Exception:
        write_json(
            OUTPUT_DIR / f"evaluation-{tag}.json",
            {
                "seed": seed,
                "arm": arm,
                "status": "evaluation_failed_weights_preserved",
                "bundle": final_info,
                "consistency": consistency,
                "traceback": traceback.format_exc(),
            },
        )
        raise

    def stage_rows(name: str) -> list[dict]:
        return [row for row in rows if row["stage"] == name]

    variance = {}
    for name in ("head", "span", "binding"):
        for stage in ("selection", "copy", "joint"):
            values = [row[name] for row in grad_rows if row["stage"] == stage]
            if values and any(values):
                variance[f"{name}|{stage}"] = {
                    "n": len(values),
                    "mean": statistics.fmean(values),
                    "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                    "cv": (
                        statistics.pstdev(values) / statistics.fmean(values)
                        if len(values) > 1 and statistics.fmean(values)
                        else 0.0
                    ),
                }
    self_prefix_updates = sum(1 for row in rows if row["self_prefix"])
    cpu = time.process_time() - start_cpu
    wall = time.perf_counter() - start_wall
    return {
        "schema": "taiji-selection-routing-17-5-run-v1",
        "step": "N17.5",
        "seed": seed,
        "arm": arm,
        "parent": {"path": str(PARENTS[seed]), "sha256": parent_sha},
        "candidate": {"candidate_sha": candidate_identity(candidate)["candidate_sha"]},
        "resumed_from": resumed_from,
        "step0": step0,
        "step300": final,
        "train": {
            "updates": len(rows),
            "first_loss": rows[0]["loss"],
            "last_loss": rows[-1]["loss"],
            "loss_by_stage": {
                name: {
                    "updates": len(stage_rows(name)),
                    "first": stage_rows(name)[0]["loss"],
                    "last": stage_rows(name)[-1]["loss"],
                    "mean_nll": statistics.fmean(row["nll"] for row in stage_rows(name)),
                    "mean_ce": statistics.fmean(row["ce"] for row in stage_rows(name)),
                    "mean_span": statistics.fmean(row["span"] for row in stage_rows(name)),
                    "mean_entropy": statistics.fmean(row["entropy"] for row in stage_rows(name)),
                }
                for name in ("selection", "copy", "joint")
            },
            "self_prefix_updates": self_prefix_updates,
            "curriculum_rate_max": max(row["curriculum_rate"] for row in rows),
            "curriculum_by_stage": {
                name: {
                    "updates": len(stage_rows(name)),
                    "mean_rate": statistics.fmean(
                        row["curriculum_rate"] for row in stage_rows(name)
                    ),
                    "self_prefix_updates": sum(1 for row in stage_rows(name) if row["self_prefix"]),
                }
                for name in ("selection", "copy", "joint")
            },
            "temperature_min": min(row["temperature"] for row in rows),
            "temperature_max": max(row["temperature"] for row in rows),
            "grad_variance": variance,
        },
        "products": {
            "step150_bundle": {
                "path": str(checkpoint_path),
                "bytes": checkpoint_path.stat().st_size if checkpoint_path.exists() else 0,
            },
            "step300_bundle": {"path": str(final_info["path"]), "bytes": final_info["bytes"]},
        },
        "consistency": consistency,
        "evaluation": {"ok": True},
        "timing": {"cpu_seconds": cpu, "wall_seconds": wall},
    }


def bootstrap(deltas: list[float], resamples: int = 2000, seed: int = 58031) -> dict:
    generator = torch.Generator().manual_seed(seed)
    samples = []
    for _ in range(resamples):
        index = torch.randint(0, len(deltas), (len(deltas),), generator=generator).tolist()
        samples.append(statistics.fmean(deltas[value] for value in index))
    samples.sort()
    return {
        "mean": statistics.fmean(deltas),
        "lower": samples[int(0.025 * len(samples))],
        "upper": samples[int(0.975 * len(samples))],
        "resamples": resamples,
        "seed": seed,
    }


def task_gate(entry: dict) -> dict:
    overall = entry["step300"]["overall"]
    checks = {
        "overall": overall["completed"] >= PLAN_GATES["overall"],
        "unseen_output_name": overall["subsets"]["unseen_output_name"]
        >= PLAN_GATES["unseen_output_name"],
        "new_description": overall["subsets"]["new_description"] >= PLAN_GATES["new_description"],
        "reversed": entry["step300"]["reversed"]["paired_rate"] >= PLAN_GATES["reversed"],
        "selection_object": entry["step300"]["selection"]["object_top1"]
        >= PLAN_GATES["selection_object"],
        "selection_place": entry["step300"]["selection"]["place_top1"]
        >= PLAN_GATES["selection_place"],
        "content_drop": all(
            value["drop"] >= PLAN_GATES["content_drop"]
            for name, value in entry["step300"]["content_interventions"]["drops"].items()
            if value["denominator"]
        ),
        "identity_rescue": entry["step300"]["identity_rescue"]["equal"]
        >= PLAN_GATES["identity_rescue"],
    }
    return {
        "checks": checks,
        "readings": {
            "overall": overall["completed"],
            "unseen_output_name": overall["subsets"]["unseen_output_name"],
            "new_description": overall["subsets"]["new_description"],
            "reversed": entry["step300"]["reversed"]["paired_rate"],
            "selection_object": entry["step300"]["selection"]["object_top1"],
            "selection_place": entry["step300"]["selection"]["place_top1"],
            "content_drops": {
                name: value["drop"]
                for name, value in entry["step300"]["content_interventions"]["drops"].items()
            },
            "identity_rescue": entry["step300"]["identity_rescue"]["equal"],
        },
        "passed": all(checks.values()),
    }


def aggregate(results: list[dict], prior: dict) -> dict:
    runs = {f"{result['seed']}-{result['arm']}": result for result in results}
    total_cpu = sum(result["timing"]["cpu_seconds"] for result in results)
    total_wall = max(result["timing"]["wall_seconds"] for result in results)
    seed_gate = {}
    for seed in SEEDS:
        a = runs[f"{seed}-A_route_lock"]
        c = runs[f"{seed}-C_unlocked"]
        gate = task_gate(a)
        a_by = a["step300"]["overall"]["by_world"]
        c_by = c["step300"]["overall"]["by_world"]
        deltas = [
            a_by[world]["completed"] / a_by[world]["n"]
            - c_by[world]["completed"] / c_by[world]["n"]
            for world in sorted(a_by)
            if world in c_by
        ]
        interval = bootstrap(deltas)
        gate["a_minus_c"] = {
            "delta": statistics.fmean(deltas),
            "bootstrap": interval,
            "gate_delta": statistics.fmean(deltas) >= PLAN_GATES["a_minus_c"],
            "gate_lower": interval["lower"] > 0,
            "passed": statistics.fmean(deltas) >= PLAN_GATES["a_minus_c"] and interval["lower"] > 0,
        }
        gate["passed"] = gate["passed"] and gate["a_minus_c"]["passed"]
        seed_gate[str(seed)] = gate
    readiness = {}
    for tag in runs:
        path = OUTPUT_DIR / f"readiness-{tag}.json"
        if path.exists():
            readiness[tag] = json.loads(path.read_text(encoding="utf-8"))["probe"]
    dev_gate = all(seed_gate[str(seed)]["passed"] for seed in SEEDS)
    total_watch = {
        "cpu_seconds": prior["cpu"] + total_cpu,
        "wall_seconds": prior["wall"] + total_wall,
        "watch_cpu": WATCH_CPU,
        "watch_wall": WATCH_WALL,
        "below_watch": prior["cpu"] + total_cpu < WATCH_CPU
        and prior["wall"] + total_wall < WATCH_WALL,
    }
    return {
        "schema": "taiji-selection-routing-17-5-compare-v1",
        "step": "N17.5",
        "date": "2026-11-04",
        "dtype": "float64",
        "threads": 1,
        "config": {
            "seeds": list(SEEDS),
            "arms": list(ARMS),
            "updates": UPDATES,
            "batch": BATCH,
            "stages": {"selection": [1, 100], "copy": [101, 200], "joint": [201, 300]},
            "lr": LR,
            "clip": CLIP,
            "span_lambda": SPAN_LAMBDA,
            "temperature": {
                "start": T_START,
                "min": T_MIN,
                "decay_updates": T_DECAY_UPDATES,
                "formula": "T(u) = 1.0 - 0.95*(u-1)/99 for u<=100, then 0.05",
                "inference": "pointer argmax",
            },
            "pointer_entropy": {"coefficient": ENTROPY_COEF, "annealed_with_temperature": True},
            "credit": "route ST (arm A) + pointer pathwise; no REINFORCE baseline",
            "curriculum": {
                "rule": "per-batch teacher-forced completion rate",
                "threshold": CURRICULUM_THRESHOLD,
            },
            "checkpoint_update": CHECKPOINT_UPDATE,
        },
        "runs": runs,
        "seed_gates": seed_gate,
        "development_gate": {
            "passed": dev_gate,
            "release_confirm_n17": dev_gate,
            "note": "17.6 only if this is true; otherwise not_released and 17.7 closes out",
        },
        "step150_readiness": readiness,
        "totals": {
            "formal_adam": len(results) * UPDATES,
            "engineering_adam": ENGINEERING_ADAM,
            "cpu_seconds": total_cpu,
            "wall_seconds_parallel": total_wall,
            "prior_cpu_seconds": prior["cpu"],
            "prior_wall_seconds": prior["wall"],
            "prior_breakdown": dict(PRIOR_BREAKDOWN),
            "cumulative": total_watch,
        },
        "limits": {
            "confirm_reads": 0,
            "no_seed_extension": True,
            "no_dose_change": True,
            "no_criterion_change": True,
        },
    }


def preflight_roundtrip(seed: int) -> dict:
    """Save/reload without an Adam step: architecture and values must match."""
    parent = ROOT / PARENTS[seed]
    candidate, optimizer = build_routing_candidate(parent, seed, "A_route_lock")
    path = OUTPUT_DIR / f"roundtrip-{seed}.pt"
    info = save_routing_bundle(
        path,
        candidate,
        optimizer,
        0,
        {"stage": "N17.5.preflight"},
        {"sha256": sha256_file(parent)},
        initial_binding_state(candidate.binding, "preflight"),
    )
    loaded, _optimizer, _state, _update, _payload = load_routing_bundle(path)
    diffs = []
    original = list(candidate.parameters())
    restored = list(loaded.parameters())
    for left, right in zip(original, restored, strict=True):
        diffs.append(float((left.detach() - right.detach()).abs().max()))
    path.unlink()
    return {
        "seed": seed,
        "parameters": len(original),
        "max_abs_diff": max(diffs),
        "ok": max(diffs) == 0.0,
        "bundle_bytes": info["bytes"],
    }


def smoke(seed: int = 28031) -> dict:
    """Zero-Adam plumbing check: forward, backward, probes on two fixtures."""
    parent = ROOT / PARENTS[seed]
    candidate, optimizer = build_routing_candidate(parent, seed, "A_route_lock")
    train, dev = load_fixtures()
    batch = train[:2]
    out = {}
    try:
        with torch.no_grad():
            generated = generate(candidate, dev[0], "A_route_lock", tag="smoke")
        out["generate"] = {
            "bytes": len(generated["raw_bytes"]),
            "eos": generated["terminated"],
            "gamma_steps": len(generated["gamma"]),
        }
        with torch.no_grad():
            no_source = generate(
                candidate, dev[0], "A_route_lock", records=(), keys=(), tag="smoke"
            )
            wrong = generate(
                candidate,
                dev[0],
                "A_route_lock",
                forced_route=compare_module().alternate_object_index(dev[0]),
                tag="smoke",
            )
            offset = generate(candidate, dev[0], "A_route_lock", offset_zero=True, tag="smoke")
        out["interventions"] = {
            "no_source_bytes": len(no_source["raw_bytes"]),
            "wrong_record_bytes": len(wrong["raw_bytes"]),
            "offset_zero_bytes": len(offset["raw_bytes"]),
        }
        for stage in ("selection", "copy", "joint"):
            set_trainable(candidate, stage)
            loss, stats = batch_loss(
                candidate, tuple(batch), "A_route_lock", stage, True, 201, "smoke"
            )
            optimizer.zero_grad()
            loss.backward()
            out[stage] = {
                "loss": float(loss.detach()),
                **{key: value for key, value in stats.items()},
                "grads": grad_norms(candidate),
            }
            optimizer.zero_grad()
        out["parameters"] = candidate.parameter_count()
        out["status"] = "ok"
    finally:
        del candidate, optimizer
    return out


def evaluate_check() -> dict:
    """Zero-update regression: evaluate_dev(full=True) must run on every arm/seed."""
    _train, dev = load_fixtures()
    sample = dev[:24]
    cells = []
    for seed in SEEDS:
        for arm in ARMS:
            candidate, optimizer = build_routing_candidate(ROOT / PARENTS[seed], seed, arm)
            out = evaluate_dev(candidate, sample, arm, f"evalcheck-{seed}-{arm}", full=True)
            cells.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "examples": out["examples"],
                    "overall_completed": out["overall"]["completed"],
                    "selection_object_top1": out["selection"]["object_top1"],
                    "identity_rescue_equal": out["identity_rescue"]["equal"],
                    "mean_nll": out["nll"]["mean_nll"],
                }
            )
            del candidate, optimizer
    return {"status": "evaluate_dev_full_check", "cells": cells, "ok": True}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--arm")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--evalcheck", action="store_true")
    arguments = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if arguments.smoke:
        print(json.dumps(smoke(), ensure_ascii=False, indent=2))
        return 0
    if arguments.evalcheck:
        print(json.dumps(evaluate_check(), ensure_ascii=False, indent=2))
        return 0
    if arguments.preflight:
        print(json.dumps([preflight_roundtrip(seed) for seed in SEEDS]))
        return 0
    if arguments.worker:
        result = run_one(arguments.seed, arguments.arm, resume=arguments.resume)
        write_json(OUTPUT_DIR / f"run-{arguments.seed}-{arguments.arm}.json", result)
        print(
            json.dumps(
                {
                    "seed": arguments.seed,
                    "arm": arguments.arm,
                    "cpu": result["timing"]["cpu_seconds"],
                    "wall": result["timing"]["wall_seconds"],
                }
            )
        )
        return 0
    start_cpu = time.process_time()
    start_wall = time.perf_counter()
    check = evaluate_check()
    print(json.dumps({"evalcheck": "ok", "cells": len(check["cells"])}, ensure_ascii=False))
    results = []
    failures = []
    with ProcessPoolExecutor(max_workers=len(SEEDS) * len(ARMS)) as pool:
        futures = {
            pool.submit(run_one, seed, arm, arguments.resume): (seed, arm)
            for seed in SEEDS
            for arm in ARMS
        }
        for future in as_completed(futures):
            seed, arm = futures[future]
            try:
                result = future.result()
                results.append(result)
                write_json(OUTPUT_DIR / f"run-{seed}-{arm}.json", result)
                print(
                    json.dumps(
                        {
                            "done": f"{seed}-{arm}",
                            "cpu": round(result["timing"]["cpu_seconds"], 1),
                            "wall": round(result["timing"]["wall_seconds"], 1),
                            "step300_overall": result["step300"]["overall"]["completed"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            except Exception:  # noqa: BLE001 - record failures, do not hide them
                failures.append({"seed": seed, "arm": arm, "traceback": traceback.format_exc()})
                print(json.dumps({"failed": f"{seed}-{arm}"}), flush=True)
    if failures:
        write_json(OUTPUT_DIR / "failures.json", {"failures": failures})
        print(json.dumps({"failures": len(failures)}))
        return 1
    report = aggregate(results, {"cpu": PRIOR_CPU, "wall": PRIOR_WALL})
    report["driver_timing"] = {
        "cpu_seconds": time.process_time() - start_cpu,
        "wall_seconds": time.perf_counter() - start_wall,
    }
    report["source"] = {
        "compare_script": str(Path(__file__).resolve().relative_to(ROOT)),
        "routing": "taiji/selection_routing.py",
        "checkpoint": "taiji/selection_routing_checkpoint.py",
        "binding_copy": "taiji/binding_copy.py",
    }
    report["thresholds"] = json.loads((ROOT / THRESHOLDS).read_text(encoding="utf-8"))
    reference, _ = build_routing_candidate(ROOT / PARENTS[SEEDS[0]], SEEDS[0], ARMS[0])
    identity = candidate_identity(reference)
    cell_shas = {
        f"{result['seed']}-{result['arm']}": result["candidate"]["candidate_sha"]
        for result in results
    }
    report["candidate"] = {
        "candidate_sha": identity["candidate_sha"],
        "candidate_sha_v2": identity["candidate_sha_v2"],
        "identity_basis": identity["identity_basis"],
        "parameter_counts": identity["parameter_counts"],
        "source_sha": identity["source_sha"],
        "consistent_across_cells": all(
            value == identity["candidate_sha"] for value in cell_shas.values()
        ),
        "per_cell": cell_shas,
    }
    write_json(REPORT, report)
    write_json(
        MANIFEST,
        {
            "schema": "taiji-selection-routing-17-5-manifest-v1",
            "step": "N17.5",
            "date": "2026-11-04",
            "report": str(REPORT),
            "report_sha256": sha256_file(REPORT),
            "report_bytes": REPORT.stat().st_size,
            "inputs": {
                "parents": {
                    str(seed): {"path": str(path), "sha256": sha256_file(ROOT / path)}
                    for seed, path in PARENTS.items()
                },
                "thresholds": {
                    "path": str(THRESHOLDS),
                    "sha256": sha256_file(ROOT / THRESHOLDS),
                },
            },
            "outputs": {
                "bundles": [
                    str(OUTPUT_DIR / f"{seed}-{arm}-step300.pt") for seed in SEEDS for arm in ARMS
                ],
                "step150": [
                    str(OUTPUT_DIR / f"{seed}-{arm}-step150.pt") for seed in SEEDS for arm in ARMS
                ],
            },
            "formal_adam": len(results) * UPDATES,
            "confirm_reads": 0,
        },
    )
    print(
        json.dumps(
            {
                "report": str(REPORT),
                "dev_gate": report["development_gate"]["passed"],
                "cpu": round(report["totals"]["cpu_seconds"], 1),
                "cumulative_cpu": round(report["totals"]["cumulative"]["cpu_seconds"], 1),
                "wall": round(report["driver_timing"]["wall_seconds"], 1),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
