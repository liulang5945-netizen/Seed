"""19.2b low-update validation with the D16-5 composite instrument.

One independent trajectory, at most 30 more Adam updates (the earlier run spent
60 Adam on two arms whose selection-stage trajectory is bit-identical, i.e. one
30-update trajectory run twice; that run is charged as 30 independent updates).
Seed 28031, selection stage only, no saves, no formal dose, no confirm read.

The 19.5 ``start_hit`` read ``argmax(read.betas)`` -- a frozen-reader posterior
that never sees the teacher-span pointer -- so it could not move under span
supervision.  This script reads the D16-5 object instead:

* ``composite_row_hit``   - the final transport row argmax for the gold record
  equals the entity's first byte (the quantity span supervision acts on);
* ``composite_start_hit`` - the same, additionally conditioned on the alpha
  route owning the gold record (route is a separate >= 0.95 gate and does not
  mature within 30 updates from the blind parent);
* ``pointer_row_hit``     - the pure span-bias argmax lands on the first byte;
* ``beta_start_hit``      - the legacy frozen-reader reading, kept for continuity;
* ``conditional_byte``    - tail bytes among route-and-composite start hits;
* span CE                 - the loss the head is actually trained on.

The release reading is pre-registered as ``composite_row_hit`` rising on both
worlds (object and place) between update 0 and update 30.  gamma is the
generation weight and 1 - gamma the copy weight; no credit is inferred (M3).

Run: ``python scripts/training/validate_taiji_package19_19_2b.py``
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

torch.set_num_threads(1)

import composite_start_instrument as instrument  # noqa: E402
import probe_taiji_content_transport_18_1 as probe_mod  # noqa: E402

from taiji.candidate_identity import candidate_identity  # noqa: E402
from taiji.selection_routing import RoutingConfig, expected_span_parameter_count  # noqa: E402
from taiji.selection_routing_checkpoint import build_routing_candidate  # noqa: E402

COMPARE_MODULE = ROOT / "scripts/training/compare_taiji_selection_routing_17_5.py"
PARENT = ROOT / "output/taiji_core_binding_select_16_1_compare/bundles/28031-A-locked-step300.pt"
SEED = 28031
ARM = "B_stop_lock"
UPDATES = 30
PROBE_EVERY = 5
SPAN_BOTTLENECK = 32
PRIOR_ADAM_RUN_TWICE = 60
PRIOR_TRUE_TRAJECTORIES = 30

REPORT = ROOT / "reports/taiji_package19_2b_lowupdate_composite_20261104.json"
MANIFEST = ROOT / "plans/manifests/taiji_package19_2b_lowupdate_composite_20261104.json"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def cmp():
    spec = importlib.util.spec_from_file_location("cmp175_lowupdate", COMPARE_MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def schedule(seed: int, train: tuple[dict, ...], updates: int) -> list[tuple[dict, ...]]:
    from taiji import data_premise_18_0 as premise

    batches = []
    epoch = 0
    take = len(train) - len(train) % 8
    generator = torch.Generator().manual_seed(premise.ORDER_SEED + (seed - 28031))
    permutation = torch.randperm(take, generator=generator).tolist()
    cursor = 0
    while len(batches) < updates:
        if cursor + 8 > len(permutation):
            epoch += 1
            generator = torch.Generator().manual_seed(premise.ORDER_SEED + (seed - 28031) + epoch)
            permutation = torch.randperm(take, generator=generator).tolist()
            cursor = 0
        batches.append(tuple(train[index] for index in permutation[cursor : cursor + 8]))
        cursor += 8
    return batches


def ledger_snapshot(candidate, arm: str, dev: tuple[dict, ...]) -> dict:
    probe = probe_mod.ProbeContext(candidate)
    probe.arm = arm
    metrics = [instrument.metrics(instrument.measure_fixture(probe, fixture)) for fixture in dev]
    summary = instrument.summarize(metrics)
    keys = (
        "beta_start_hit_rate",
        "pointer_start_hit_rate",
        "composite_start_hit_rate",
        "route_hit_rate",
        "beta_row_hit_rate",
        "pointer_row_hit_rate",
        "composite_row_hit_rate",
        "carry_active_rate",
        "conditional_byte",
        "unconditional_byte",
    )
    snapshot = {world: {key: summary[world][key] for key in keys} for world in ("object", "place")}
    snapshot["gamma_mean"] = summary["gamma_mean"]
    snapshot["pooled"] = {
        "composite_row_hit_rate": statistics.fmean(
            [
                summary["object"]["composite_row_hit_rate"],
                summary["place"]["composite_row_hit_rate"],
            ]
        ),
        "composite_start_hit_rate": statistics.fmean(
            [
                summary["object"]["composite_start_hit_rate"],
                summary["place"]["composite_start_hit_rate"],
            ]
        ),
    }
    return snapshot


def main() -> int:
    start_cpu = time.process_time()
    start_wall = time.perf_counter()
    module = cmp()
    from taiji import data_premise_18_0 as premise

    payload = premise.build_payload()
    train = premise.fixtures(payload, "train")
    dev = probe_mod.dev_slice(premise.fixtures(payload, "dev"))
    batches = schedule(SEED, train, UPDATES)
    config = RoutingConfig(span_bottleneck=SPAN_BOTTLENECK)

    candidate, optimizer = build_routing_candidate(
        PARENT, SEED, ARM, span_bottleneck=SPAN_BOTTLENECK
    )
    identity = candidate_identity(candidate)
    probes = [{"update": 0, **ledger_snapshot(candidate, ARM, dev)}]
    span_losses = []
    total_adam = 0
    for update in range(1, UPDATES + 1):
        stage = "selection"
        parameters = module.set_trainable(candidate, stage)
        loss, stats = module.batch_loss(
            candidate, batches[update - 1], ARM, stage, False, update, f"low-{ARM}"
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        total_adam += 1
        span_losses.append(float(stats["span"]))
        if update % PROBE_EVERY == 0:
            probes.append({"update": update, **ledger_snapshot(candidate, ARM, dev)})
            print(
                json.dumps(
                    {
                        "arm": ARM,
                        "update": update,
                        "composite_obj": round(probes[-1]["object"]["composite_row_hit_rate"], 3),
                        "composite_place": round(probes[-1]["place"]["composite_row_hit_rate"], 3),
                        "composite_start_obj": round(
                            probes[-1]["object"]["composite_start_hit_rate"], 3
                        ),
                        "route_obj": round(probes[-1]["object"]["route_hit_rate"], 3),
                        "span": round(span_losses[-1], 3),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    first, last = probes[0], probes[-1]
    deltas = {
        world: {
            key: last[world][key] - first[world][key]
            for key in (
                "beta_start_hit_rate",
                "pointer_start_hit_rate",
                "composite_start_hit_rate",
                "composite_row_hit_rate",
                "route_hit_rate",
                "conditional_byte",
            )
        }
        for world in ("object", "place")
    }
    rising = all(deltas[world]["composite_row_hit_rate"] > 0.0 for world in ("object", "place"))

    cpu = time.process_time() - start_cpu
    wall = time.perf_counter() - start_wall
    report = {
        "schema": "taiji-package19-2b-lowupdate-composite-v2",
        "step": "N19.2b / D16-5",
        "date": "2026-11-04",
        "purpose": "does the composite row argmax rise with gold-start supervision",
        "design": {
            "seed": SEED,
            "arm": ARM,
            "updates": UPDATES,
            "stage": "selection (1-100 window); only selection CE + span CE are trained",
            "span_head": "SpanPointerHead 32-d bottleneck (4,194 parameters)",
            "supervision": "teacher_span(record_raw, entity_bytes) gold start/end on object and place records",
            "pointer_injection": "span start logits enter routing_step as pointer_bias; no alpha/beta/encoder/transport change",
            "dev_examples": len(dev),
            "saves": 0,
            "confirm_reads": 0,
            "release_reading": "composite_row_hit_rate must rise on both object and place; "
            "route is a separate gate and does not mature in this window",
        },
        "candidate_sha_v2": identity["candidate_sha_v2"],
        "span_parameters": sum(parameter.numel() for parameter in candidate.span_head.parameters()),
        "span_expected": expected_span_parameter_count(config),
        "probes": probes,
        "deltas": deltas,
        "composite_row_rising_on_both_worlds": rising,
        "span_loss": {
            "first": span_losses[0],
            "last": span_losses[-1],
            "mean": statistics.fmean(span_losses),
        },
        "accounting": {
            "adam_steps": total_adam,
            "independent_trajectories": 1,
            "adam_cap": 60,
            "prior_run": {
                "charged_adam": PRIOR_ADAM_RUN_TWICE,
                "true_independent_updates": PRIOR_TRUE_TRAJECTORIES,
                "reason": "A_route_lock and B_stop_lock are bit-identical in the selection "
                "stage, so the earlier 2 x 30 Adam was one 30-update trajectory run twice",
            },
            "cumulative_independent_updates": PRIOR_TRUE_TRAJECTORIES + total_adam,
        },
        "instrument": {
            "object": "argmax(row_logits = transport_logits(alpha_i * betas_i) + span_bias)",
            "legacy": "beta_start_hit_rate is the frozen-reader posterior, kept for continuity",
            "gamma": "gamma = generation weight / 1 - gamma = copy weight; no credit inferred (M3)",
        },
        "cost": {
            "cpu_seconds": cpu,
            "wall_seconds": wall,
            "adam_steps": total_adam,
            "adam_cap": 60,
        },
        "limits": {"formal_dose": False, "training_bundles_kept": False, "confirm_reads": 0},
    }
    write_json(REPORT, report)
    write_json(
        MANIFEST,
        {
            "schema": "taiji-package19-2b-lowupdate-composite-manifest-v1",
            "step": "N19.2b / D16-5",
            "report": str(REPORT.relative_to(ROOT)),
            "adam_steps": total_adam,
            "composite_row_rising": rising,
            "deltas": deltas,
            "cost": {"cpu_seconds": cpu, "wall_seconds": wall},
        },
    )
    print(
        json.dumps(
            {
                "report": str(REPORT.relative_to(ROOT)),
                "adam": total_adam,
                "rising": rising,
                "deltas": deltas,
                "cpu": round(cpu, 1),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
