"""19.2l-A: is the order-decoy margin gradient structurally zero on a bag scorer?

Owner ruling 2026-11-05 (amendment to 19.2l phase 3): before any training, prove
structurally that an order-only decoy cannot move a scorer that sees only the
byte multiset.  The claim to test is

    if the scorer output depends only on the record's byte multiset, then the
    margin term of any byte-permutation negative has gradient exactly 0.

Two arms share one parent binding, one data order, one state path and one
loss; the only difference is the scorer:

* ``O`` - 19.2j O baseline head (``W_hidden(4*width+prev_width) -> w_out``),
  no ``W_align`` / ``offset_embed`` position features;
* ``N`` - 19.2j N alignment head (the same trunk plus the 32x32 ``W_align`` and
  the 5x8 ``offset_embed``, +1,320 parameters).

The margin term is ``max(0, m - (score_gold - score_decoy))`` with m = 1.0, the
same term the gate used.  Its gradient is read at the teacher-forced content
start, with the routing state treated as a constant (the state is produced by
the 19.2l gate's O head and reused for both arms, so the scorer is the only
variable).  Readings are the mean loss over 128 starts (64 train examples x 2
starts) and over the first training batch (8 examples / 16 starts).

Also reported: the gold/decoy key-vector distance (the mechanism behind the
number), the N-arm margin distribution, and the N-arm init identifiability
(per-parameter RMS ratio of the added block versus the trunk on the CE loss).

0 Adam steps, 0 parameters changed, 0 bundles, 0 confirm; budget <= 80 CPU.

Run: ``python scripts/training/probe_taiji_package19_2l_margin_gradient_structure.py``
"""

from __future__ import annotations

import hashlib
import json
import math
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

import compare_taiji_selection_routing_17_5 as compare_module  # noqa: E402
import probe_taiji_content_transport_18_1 as probe_mod  # noqa: E402
import probe_taiji_package19_2l_order_decoy_activity as gate  # noqa: E402
import train_taiji_package19_2h_sequence_surface as h  # noqa: E402
import train_taiji_package19_2j_from_scratch_selection_ab as j  # noqa: E402

from taiji import data_premise_18_0 as premise  # noqa: E402
from taiji.living_core import EOS  # noqa: E402
from taiji.selection_routing import (  # noqa: E402
    _advanced_state,
    _s_t,
    observe_goal,
    routing_step,
)
from taiji.selection_routing_checkpoint import load_routing_bundle  # noqa: E402

SEED = 28031
SAMPLE = 64
MARGIN_M = 1.0
EXACT_ZERO = 1e-12
GRADIENT_FLOOR = 1e-6
STARTS = (probe_mod.OBJECT_POSITIONS[0], probe_mod.PLACE_POSITIONS[0])
START_NAMES = ("object", "place")

REPORT = ROOT / "reports/taiji_package19_2l_margin_gradient_20261105.json"
MANIFEST = ROOT / "plans/manifests/taiji_package19_2l_margin_gradient_20261105.json"
PROGRESS = ROOT / "reports/taiji_package19_2l_margin_gradient_progress_20261105.jsonl"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        "p90": rank(0.90),
        "min": ordered[0],
        "max": ordered[-1],
    }


def state_at(candidate, prepared, context, prefix, start: int):
    """Teacher-forced state at ``start``; the same candidate produces both arms' states.

    Everything here is ``no_grad``: the margin term is read at a fixed state so the
    only differentiable path is the scorer.
    """
    state = prepared.state
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
                arm="B_stop_lock",
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


def readings(state_candidate, examples, tag: str) -> list[dict]:
    """Key tensors and fixed states for every (example, start) pair."""
    out = []
    for fixture in examples:
        prepared, context = compare_module.prepared_pair(state_candidate, fixture, tag, "grad")
        prefix = tuple(fixture["target_prefix"])
        mapping = probe_mod.content_positions(fixture)
        for start, name in zip(STARTS, START_NAMES, strict=True):
            gold_index, _offset = mapping[start]
            gold_key = context.key_vecs[gold_index].detach()
            gold_text = tuple(fixture["record_keys"])[gold_index]
            built = gate.order_decoy(gold_text)
            if built["degenerate"]:
                continue
            decoy_key = state_candidate.binding.base.relation.pool(
                state_candidate.binding.base.relation.encode_units(
                    state_candidate.binding.subject, built["decoy"].encode("utf-8")
                )
            ).detach()
            s_t, previous_embedding = state_at(state_candidate, prepared, context, prefix, start)
            out.append(
                {
                    "world": fixture["world"],
                    "index": fixture["index"],
                    "start": name,
                    "gold": gold_text,
                    "decoy": built["decoy"],
                    "goal_vec": context.goal_vec.detach(),
                    "gold_key": gold_key,
                    "decoy_key": decoy_key,
                    "s_t": s_t,
                    "previous_embedding": previous_embedding,
                    "key_max_abs_delta": float((gold_key - decoy_key).abs().max()),
                }
            )
    return out


def margin_loss(candidate, rows: list[dict]) -> tuple[torch.Tensor, list[float]]:
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


def gradient_norms(candidate, rows: list[dict]) -> dict:
    named = [(name, parameter) for name, parameter in candidate.head.named_parameters()]
    parameters = [parameter for _name, parameter in named if parameter.requires_grad]
    names = [name for name, parameter in named if parameter.requires_grad]
    loss, margins = margin_loss(candidate, rows)
    grads = torch.autograd.grad(loss, parameters, allow_unused=True)
    per_parameter = {}
    squared = 0.0
    max_abs = 0.0
    for name, grad in zip(names, grads, strict=True):
        if grad is None:
            per_parameter[name] = 0.0
            continue
        norm = float(grad.detach().norm())
        per_parameter[name] = norm
        squared += norm * norm
        max_abs = max(max_abs, float(grad.detach().abs().max()))
    total = math.sqrt(squared)
    added = [value for name, value in per_parameter.items() if j._is_added(name)]
    trunk = [value for name, value in per_parameter.items() if not j._is_added(name)]
    added_norm = math.sqrt(sum(value * value for value in added)) if added else 0.0
    trunk_norm = math.sqrt(sum(value * value for value in trunk)) if trunk else 0.0
    return {
        "loss": float(loss.detach()),
        "margin": distribution(margins),
        "activated": sum(1 for value in margins if value < MARGIN_M),
        "readings": len(margins),
        "gradient_norm": total,
        "max_abs_gradient": max_abs,
        "added_norm": added_norm,
        "trunk_norm": trunk_norm,
        "per_parameter": per_parameter,
        "exactly_zero": total == 0.0,
        "below_1e_12": total <= EXACT_ZERO,
        "above_1e_6": total > GRADIENT_FLOOR,
    }


def main() -> int:
    start_cpu = time.process_time()
    start_wall = time.perf_counter()
    PROGRESS.unlink(missing_ok=True)

    payload = premise.build_payload()
    train = premise.fixtures(payload, "train")
    sample = gate.even_sample(train, SAMPLE)
    first_batch = tuple(h.schedule(train, 1)[0])

    parent, _opt, _state, _index, _cursor = load_routing_bundle(j.PARENT)
    if any(torch.isnan(t).any().item() for t in parent.head.parameters()):
        raise RuntimeError("parent head carries NaN (D16-6)")

    state_candidate, _o_opt, _o_params = j.build_run(parent, SEED, "O")
    rows = readings(state_candidate, sample, "19.2l-A-sample")
    first_rows = readings(state_candidate, first_batch, "19.2l-A-first")

    j.progress_append(
        PROGRESS,
        {
            "event": "structure_start",
            "readings_sample": len(rows),
            "readings_first_batch": len(first_rows),
            "adam_updates": 0,
        },
    )

    key_delta = distribution([row["key_max_abs_delta"] for row in rows])
    all_keys_identical = all(row["key_max_abs_delta"] == 0.0 for row in rows)

    arms = {}
    for arm in ("O", "N"):
        candidate, _optimizer, _params = j.build_run(parent, SEED, arm)
        arm_readings = gradient_norms(candidate, rows)
        first = gradient_norms(candidate, first_rows)
        added_parameters = sum(
            parameter.numel()
            for name, parameter in candidate.head.named_parameters()
            if j._is_added(name)
        )
        arm_readings.update(
            {
                "head_parameters": sum(
                    parameter.numel() for parameter in candidate.head.parameters()
                ),
                "added_parameters": added_parameters,
                "first_batch": first,
            }
        )
        if arm == "N":
            arm_readings["init_identifiability"] = j.gradient_probe(
                candidate, first_batch, "19.2l-A-init"
            )
        arms[arm] = arm_readings
        j.progress_append(
            PROGRESS,
            {
                "event": "arm_done",
                "arm": arm,
                "gradient_norm_sample": arm_readings["gradient_norm"],
                "gradient_norm_first_batch": first["gradient_norm"],
                "max_abs": arm_readings["max_abs_gradient"],
            },
        )

    o_norm = arms["O"]["gradient_norm"]
    n_norm = arms["N"]["gradient_norm"]
    n_first = arms["N"]["first_batch"]["gradient_norm"]
    if o_norm <= EXACT_ZERO and n_norm > GRADIENT_FLOOR:
        code = "bag_zero_position_positive"
    elif o_norm <= EXACT_ZERO:
        code = "both_zero"
    else:
        code = "frozen_transport_is_order_sensitive"

    report = {
        "schema": "taiji-package19-2l-margin-gradient-v1",
        "step": "N19.2l-A order-decoy margin gradient structure",
        "phase": "structural proof before phase 3 (owner amendment)",
        "date": "2026-11-05",
        "mode": "forward + backward of the margin term only; 0 Adam, 0 bundles, 0 confirm",
        "proposition": (
            "if the scorer output depends only on the record's byte multiset, then the "
            "margin term of any byte-permutation negative has gradient exactly 0"
        ),
        "design": {
            "arms": {
                "O": "19.2j O baseline head; no W_align / offset_embed position features",
                "N": ("19.2j N alignment head; same trunk plus W_align 32x32 and 5x8 offset_embed"),
            },
            "only_variable": (
                "the scorer (head architecture); parent binding, states, data and loss are shared"
            ),
            "margin_term": (
                "mean over readings of max(0, m - (score_gold - score_decoy)), m = 1.0"
            ),
            "state_definition": (
                "teacher-forced content-start state from the 19.2l gate O head, held fixed and "
                "identical for both arms; the routing state path is constant (no gradient "
                "through routing), so the only differentiable path is the scorer"
            ),
            "samples": {
                "sample": "64 train examples / 128 starts (same even-stride set as the gate)",
                "first_batch": "first training batch of the frozen 19.2j order / 16 starts",
            },
            "exact_zero_threshold": EXACT_ZERO,
            "gradient_floor": GRADIENT_FLOOR,
        },
        "key_vector": {
            "max_abs_delta_distribution": key_delta,
            "all_identical": all_keys_identical,
            "note": (
                "if the frozen transport mapped a permutation to the same pooled key, every "
                "bag scorer would see identical rows; a non-zero delta is the reason an O-head "
                "gradient can be non-zero even though the head has no explicit position features"
            ),
        },
        "arms": arms,
        "verdict": {
            "code": code,
            "o_gradient_norm": o_norm,
            "n_gradient_norm": n_norm,
            "n_first_batch_gradient_norm": n_first,
            "o_meets_exact_zero": o_norm <= EXACT_ZERO,
            "n_beats_gradient_floor": n_norm > GRADIENT_FLOOR,
            "n_first_batch_beats_floor": n_first > GRADIENT_FLOOR,
            "gradient_cancels_at_source": not (n_first > GRADIENT_FLOOR),
            "proposition_confirmed": o_norm <= EXACT_ZERO and n_norm > GRADIENT_FLOOR,
            "mechanism": (
                "the proposition is conditional: it holds whenever the scorer's row input is a "
                "function of the byte multiset only.  The O and N readings say whether the "
                "current pipeline instantiates that premise, and whether the explicit position "
                "features give the margin term a usable gradient"
            ),
        },
        "d16_16": {
            "rule": (
                "D16-16: if the scorer output depends only on the record's byte multiset, the "
                "margin term of any byte-permutation negative has gradient exactly 0; every "
                "future hard-negative proposal must pass this structural reading before any "
                "training budget is spent"
            ),
            "measured": {"O": o_norm, "N": n_norm},
        },
        "next": {
            "phase_b": "pending owner approval after this report",
            "plan": (
                "N structure x order-decoy margin term (experiment) versus N structure x CE-only "
                "(control); same seed 28031, order, init, 30 updates; <=30 Adam per arm (60 "
                "total), <=400 CPU, 1 bundle, 0 confirm; diagnostic gates: swap hard argmax "
                ">=8/20, new-description route_miss down >=0.10 versus the CE arm, "
                "conditional_byte no regression >0.05, old description no regression, "
                "decoy_margin_after median clearly above -0.003"
            ),
        },
        "cost": {
            "adam_updates": 0,
            "parameters_changed": 0,
            "bundles": 0,
            "confirm_reads": 0,
            "cpu_seconds": time.process_time() - start_cpu,
            "wall_seconds": time.perf_counter() - start_wall,
        },
        "reproducibility": {
            "parent_bundle": str(j.PARENT.relative_to(ROOT)),
            "parent_sha256": sha256_file(j.PARENT),
            "seed": SEED,
            "torch_threads": 1,
        },
    }
    j.write_json_safe(REPORT, report)
    j.write_json_safe(
        MANIFEST,
        {
            "schema": "taiji-package19-2l-margin-gradient-manifest-v1",
            "step": "N19.2l-A",
            "date": "2026-11-05",
            "report": str(REPORT.relative_to(ROOT)),
            "verdict": code,
            "o_gradient_norm": o_norm,
            "n_gradient_norm": n_norm,
            "n_first_batch_gradient_norm": n_first,
            "key_all_identical": all_keys_identical,
            "adam_updates": 0,
            "confirm_reads": 0,
        },
    )
    print(
        json.dumps(
            {
                "verdict": code,
                "o_norm": o_norm,
                "n_norm": n_norm,
                "n_first_batch": n_first,
                "keys_identical": all_keys_identical,
                "cpu": round(time.process_time() - start_cpu, 1),
                "wall": round(time.perf_counter() - start_wall, 1),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
