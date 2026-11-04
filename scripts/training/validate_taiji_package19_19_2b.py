"""19.2b low-update validation: does start_hit rise with gold-start supervision?

At most 60 Adam steps (2 arms x 30 updates, seed 28031, selection stage only).
No saves, no formal dose, no confirm read.  Measures the teacher-forced ledger
every 10 updates on the frozen 48-example dev slice and reports:

* ``start_hit``  - alpha route owns the record AND the span pointer puts the
  copy start on the entity's first byte (the supervised quantity);
* ``route_hit``  - alpha's hard route only;
* ``conditional_byte`` - tail bytes among start-hit examples;
* span CE - the loss the head is actually trained on.

gamma is the generation weight and 1 - gamma the copy weight; the ledger reads
the model's own route, no credit inference (M3).

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

import probe_taiji_content_transport_18_1 as probe_mod  # noqa: E402
import probe_taiji_package19 as ledger_mod  # noqa: E402

from taiji.candidate_identity import candidate_identity  # noqa: E402
from taiji.selection_routing import RoutingConfig, expected_span_parameter_count  # noqa: E402
from taiji.selection_routing_checkpoint import build_routing_candidate  # noqa: E402

COMPARE_MODULE = ROOT / "scripts/training/compare_taiji_selection_routing_17_5.py"
PARENT = ROOT / "output/taiji_core_binding_select_16_1_compare/bundles/28031-A-locked-step300.pt"
SEED = 28031
ARMS = ("A_route_lock", "B_stop_lock")
UPDATES = 30
PROBE_EVERY = 10
SPAN_BOTTLENECK = 32

REPORT = ROOT / "reports/taiji_package19_2b_lowupdate_20261104.json"
MANIFEST = ROOT / "plans/manifests/taiji_package19_2b_lowupdate_20261104.json"


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
    metrics = [
        ledger_mod.ledger_metrics(ledger_mod.ledger_teacher_forcing(probe, fixture))
        for fixture in dev
    ]
    summary = ledger_mod.ledger_summary(metrics)
    return {
        "object": {
            "start_hit_rate": summary["object"]["start_hit_rate"],
            "route_hit_rate": summary["object"]["route_hit_rate"],
            "beta_start_hit_rate": summary["object"]["beta_start_hit_rate"],
            "conditional_byte": summary["object"]["conditional_byte"],
            "unconditional_byte": summary["object"]["unconditional_byte"],
        },
        "place": {
            "start_hit_rate": summary["place"]["start_hit_rate"],
            "route_hit_rate": summary["place"]["route_hit_rate"],
            "beta_start_hit_rate": summary["place"]["beta_start_hit_rate"],
            "conditional_byte": summary["place"]["conditional_byte"],
            "unconditional_byte": summary["place"]["unconditional_byte"],
        },
        "pooled": {
            "start_hit_rate": statistics.fmean(
                [summary["object"]["start_hit_rate"], summary["place"]["start_hit_rate"]]
            ),
            "route_hit_rate": statistics.fmean(
                [summary["object"]["route_hit_rate"], summary["place"]["route_hit_rate"]]
            ),
        },
    }


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

    cells = []
    total_adam = 0
    for arm in ARMS:
        candidate, optimizer = build_routing_candidate(
            PARENT, SEED, arm, span_bottleneck=SPAN_BOTTLENECK
        )
        identity = candidate_identity(candidate)
        probes = [{"update": 0, **ledger_snapshot(candidate, arm, dev)}]
        span_losses = []
        for update in range(1, UPDATES + 1):
            stage = "selection"
            parameters = module.set_trainable(candidate, stage)
            loss, stats = module.batch_loss(
                candidate, batches[update - 1], arm, stage, False, update, f"low-{arm}"
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            total_adam += 1
            span_losses.append(float(stats["span"]))
            if update % PROBE_EVERY == 0:
                probes.append({"update": update, **ledger_snapshot(candidate, arm, dev)})
                print(
                    json.dumps(
                        {
                            "arm": arm,
                            "update": update,
                            "start_obj": round(probes[-1]["object"]["start_hit_rate"], 3),
                            "start_place": round(probes[-1]["place"]["start_hit_rate"], 3),
                            "route_obj": round(probes[-1]["object"]["route_hit_rate"], 3),
                            "span": round(span_losses[-1], 3),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        cells.append(
            {
                "seed": SEED,
                "arm": arm,
                "span_bottleneck": SPAN_BOTTLENECK,
                "span_parameters": sum(p.numel() for p in candidate.span_head.parameters()),
                "span_expected": expected_span_parameter_count(config),
                "candidate_sha_v2": identity["candidate_sha_v2"],
                "updates": UPDATES,
                "probes": probes,
                "span_loss": {
                    "first": span_losses[0],
                    "last": span_losses[-1],
                    "mean": statistics.fmean(span_losses),
                },
            }
        )

    cpu = time.process_time() - start_cpu
    wall = time.perf_counter() - start_wall
    deltas = {}
    for arm_cell in cells:
        first, last = arm_cell["probes"][0], arm_cell["probes"][-1]
        deltas[arm_cell["arm"]] = {
            "start_hit_object_delta": last["object"]["start_hit_rate"]
            - first["object"]["start_hit_rate"],
            "start_hit_place_delta": last["place"]["start_hit_rate"]
            - first["place"]["start_hit_rate"],
            "route_hit_object_delta": last["object"]["route_hit_rate"]
            - first["object"]["route_hit_rate"],
            "conditional_object_delta": last["object"]["conditional_byte"]
            - first["object"]["conditional_byte"],
        }
    rising = all(
        value["start_hit_object_delta"] > 0.0 and value["start_hit_place_delta"] > 0.0
        for value in deltas.values()
    )

    report = {
        "schema": "taiji-package19-2b-lowupdate-v1",
        "step": "N19.2b",
        "date": "2026-11-04",
        "purpose": "does start_hit rise with gold-start supervision inside 60 Adam",
        "design": {
            "seed": SEED,
            "arms": list(ARMS),
            "updates_per_arm": UPDATES,
            "stage": "selection (1-100 window); only selection CE + span CE are trained",
            "span_head": "SpanPointerHead 32-d bottleneck (4,194 parameters)",
            "supervision": "teacher_span(record_raw, entity_bytes) gold start/end on object and place records",
            "pointer_injection": "span start logits enter routing_step as pointer_bias; no alpha/beta/encoder/transport change",
            "dev_examples": len(dev),
            "saves": 0,
            "confirm_reads": 0,
        },
        "cells": cells,
        "deltas": deltas,
        "start_hit_rising_on_both_worlds": rising,
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
            "schema": "taiji-package19-2b-lowupdate-manifest-v1",
            "step": "N19.2b",
            "report": str(REPORT.relative_to(ROOT)),
            "adam_steps": total_adam,
            "start_hit_rising": rising,
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
