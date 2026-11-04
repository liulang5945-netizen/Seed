"""19.2c: does span supervision move the landing spot when the route is mature?

19.2b's low-update probe started from the blind 16.1 parent, where route was still
~0.03 at update 30, so ``composite_start_hit = route AND row-argmax`` was capped
by the route gate and "route ceiling" could not be separated from "supervision
does not act on the landing spot".  This diagnostic swaps the parent only:

* parent: 19.5 ``28031-B_stop_lock-step300`` (route ~1.0, composite 0.917/0.979);
* fresh ``SpanPointerHead`` 32-d bottleneck (4,194 parameters) on top;
* at most 60 independent Adam updates, selection stage only, no saves, no confirm;
* readings at updates 0/5/10/15/20/25/30/45/60.

It is a separate diagnostic package (19.2c).  It is not a dose of 19.2b, does not
extend the trajectory, and does not spend the six-cell 32,256.6 CPU.

Decision reading (pre-registered): ``composite_start_hit`` must rise on both
worlds.  With route ~1.0 the gate object and the route-free ``composite_row_hit``
should coincide; if the row reading rises but the routed reading does not, the
route is the ceiling; if neither rises, supervision does not act on the landing
spot.  gamma is the generation weight and 1 - gamma the copy weight; no training
credit is inferred from gamma (M3).

Run: ``python scripts/training/validate_taiji_package19_2c_mature_parent.py``
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
from taiji.selection_routing import (  # noqa: E402
    RoutingConfig,
    SpanPointerHead,
    expected_span_parameter_count,
)
from taiji.selection_routing_checkpoint import (  # noqa: E402
    load_routing_bundle,
    make_routing_optimizer,
    parameter_tensors,
)

COMPARE_MODULE = ROOT / "scripts/training/compare_taiji_selection_routing_17_5.py"
PARENT = ROOT / "output/taiji_package19_compare/28031-B_stop_lock-step300.pt"
SEED = 28031
ARM = "B_stop_lock"
UPDATES = 60
PROBE_AT = (0, 5, 10, 15, 20, 25, 30, 45, 60)
SPAN_BOTTLENECK = 32

REPORT = ROOT / "reports/taiji_package19_2c_mature_parent_20261104.json"
MANIFEST = ROOT / "plans/manifests/taiji_package19_2c_mature_parent_20261104.json"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def cmp():
    spec = importlib.util.spec_from_file_location("cmp175_19_2c", COMPARE_MODULE)
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
        "decode_start_hit_rate",
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
    return snapshot


def parent_health(candidate) -> dict:
    tensors = parameter_tensors(candidate)
    nan = sum(int(torch.isnan(value).sum()) for value in tensors.values())
    return {
        "path": str(PARENT.relative_to(ROOT)),
        "parameter_tensors": len(tensors),
        "nan_parameters": nan,
        "all_finite": nan == 0,
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

    candidate = load_routing_bundle(PARENT)[0]
    candidate.span_head = SpanPointerHead(config, seed=SEED)
    candidate.configure_trainable()
    optimizer = make_routing_optimizer(candidate)
    health = parent_health(candidate)
    identity = candidate_identity(candidate)

    probes = [{"update": 0, **ledger_snapshot(candidate, ARM, dev)}]
    span_losses = []
    total_adam = 0
    for update in range(1, UPDATES + 1):
        parameters = module.set_trainable(candidate, "selection")
        loss, stats = module.batch_loss(
            candidate, batches[update - 1], ARM, "selection", False, update, f"19.2c-{ARM}"
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        total_adam += 1
        span_losses.append(float(stats["span"]))
        if update in PROBE_AT:
            probes.append({"update": update, **ledger_snapshot(candidate, ARM, dev)})
            print(
                json.dumps(
                    {
                        "arm": ARM,
                        "update": update,
                        "composite_start_obj": round(
                            probes[-1]["object"]["composite_start_hit_rate"], 3
                        ),
                        "composite_start_place": round(
                            probes[-1]["place"]["composite_start_hit_rate"], 3
                        ),
                        "composite_row_obj": round(
                            probes[-1]["object"]["composite_row_hit_rate"], 3
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
                "composite_start_hit_rate",
                "composite_row_hit_rate",
                "pointer_start_hit_rate",
                "beta_start_hit_rate",
                "route_hit_rate",
                "conditional_byte",
            )
        }
        for world in ("object", "place")
    }
    start_rising = all(
        deltas[world]["composite_start_hit_rate"] > 0.0 for world in ("object", "place")
    )
    row_rising = all(deltas[world]["composite_row_hit_rate"] > 0.0 for world in ("object", "place"))
    if start_rising:
        separation = "supervision acts on the landing spot under a mature route"
    elif row_rising:
        separation = (
            "route ceiling confounded 19.2b: the row reading rises, the routed reading does not"
        )
    else:
        separation = "supervision does not act on the landing spot even with a mature route"

    cpu = time.process_time() - start_cpu
    wall = time.perf_counter() - start_wall
    report = {
        "schema": "taiji-package19-2c-mature-parent-v1",
        "step": "N19.2c (diagnostic, not a 19.2b dose)",
        "date": "2026-11-04",
        "purpose": "separate the route ceiling from span-supervision ineffectiveness",
        "design": {
            "parent": str(PARENT.relative_to(ROOT)),
            "parent_route_hit_19_5": {"object": 0.998, "place": 1.0},
            "parent_composite_start_19_5": {"object": 0.917, "place": 0.979},
            "seed": SEED,
            "arm": ARM,
            "updates": UPDATES,
            "stage": "selection (1-100 window); only selection CE + span CE are trained",
            "span_head": "fresh SpanPointerHead 32-d bottleneck (4,194 parameters)",
            "probes_at": list(PROBE_AT),
            "saves": 0,
            "confirm_reads": 0,
            "no_dose_extension": "independent diagnostic package; does not extend the 19.2b 60-update cap",
        },
        "parent_health": health,
        "candidate_sha_v2": identity["candidate_sha_v2"],
        "span_parameters": sum(parameter.numel() for parameter in candidate.span_head.parameters()),
        "span_expected": expected_span_parameter_count(config),
        "probes": probes,
        "deltas": deltas,
        "composite_start_rising_on_both_worlds": start_rising,
        "composite_row_rising_on_both_worlds": row_rising,
        "separation_conclusion": separation,
        "span_loss": {
            "first": span_losses[0],
            "last": span_losses[-1],
            "mean": statistics.fmean(span_losses),
        },
        "accounting": {
            "adam_steps": total_adam,
            "independent_trajectories": 1,
            "package": "19.2c",
            "excluded_from_19_2b_cap": True,
            "prior_19_2b_independent_updates": 60,
        },
        "attribution_rules": [
            "gamma is the generation weight and 1 - gamma the copy weight",
            "gamma never implies training credit (M3)",
        ],
        "cost": {"cpu_seconds": cpu, "wall_seconds": wall, "adam_steps": total_adam},
    }
    manifest = {
        "schema": "taiji-package19-2c-manifest-v1",
        "step": "N19.2c",
        "date": "2026-11-04",
        "report": str(REPORT.relative_to(ROOT)),
        "parent": str(PARENT.relative_to(ROOT)),
        "parent_health": health,
        "candidate_sha_v2": identity["candidate_sha_v2"],
        "adam_steps": total_adam,
        "probes_at": list(PROBE_AT),
        "cost_cpu": cpu,
        "spent_six_cell_budget": False,
        "confirm_reads": 0,
        "verdict": separation,
    }
    write_json(REPORT, report)
    write_json(MANIFEST, manifest)
    print(json.dumps({"report": str(REPORT), "separation": separation, "cpu": round(cpu, 1)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
