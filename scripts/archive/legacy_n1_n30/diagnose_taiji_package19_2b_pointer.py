"""19.2b diagnostic: which quantity does the span pointer actually move?

Zero updates, zero confirm.  The 19.5 stage gate read ``start_hit`` as
``route_hit and argmax(read.betas[record]) == gold_offset``.  ``read.betas`` is
the frozen reader's own position posterior: it does not contain the span
pointer.  The span pointer enters ``routing_step`` as ``pointer_bias`` and lands
in ``step.pointer`` / ``step.pointer_logits`` (and the transport mass), so this
script measures all three at the same gold start positions:

* ``beta_argmax_hit``      - the 19.5 metric (frozen reader, not supervised);
* ``pointer_logits_hit``   - argmax of the span-biased transport row logits;
* ``pointer_hit``          - argmax of the emitted pointer (after any carry);
* ``carry_active``         - whether the m carry bypassed the transport at this
  start position (a bypass is not a span failure).

Run on the six 19.5 step300 bundles (the 19.2b baseline) and optionally on any
step300 bundle directory.

Run:
  python scripts/training/diagnose_taiji_package19_2b_pointer.py
"""

from __future__ import annotations

import argparse
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

from taiji import data_premise_18_0 as premise  # noqa: E402
from taiji.binding_copy import observe_goal  # noqa: E402
from taiji.living_core import EOS  # noqa: E402
from taiji.selection_routing import _advanced_state, routing_step  # noqa: E402
from taiji.selection_routing_checkpoint import load_routing_bundle  # noqa: E402

SEEDS = (28031, 28032)
ARMS = ("A_route_lock", "B_stop_lock", "C_unlocked")
DEFAULT_DIR = ROOT / "output/taiji_package19_compare"
REPORT = ROOT / "reports/taiji_package19_2b_pointer_diagnostic_20261104.json"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def diagnose_cell(path: Path, fixtures: tuple[dict, ...], untrained: int = 0) -> dict:
    if untrained:
        from taiji.selection_routing_checkpoint import build_routing_candidate

        parent = (
            ROOT / "output/taiji_core_binding_select_16_1_compare/bundles/28031-A-locked-step300.pt"
        )
        candidate, _optimizer = build_routing_candidate(
            parent, 28031, "A_route_lock", span_bottleneck=untrained
        )
    else:
        candidate = load_routing_bundle(path)[0]
    probe = probe_mod.ProbeContext(candidate)
    arm = path.name.split("-")[1]
    rows = {"object": [], "place": []}
    for fixture in fixtures:
        prepared, context = probe.prepare(fixture)
        state, gate_index = observe_goal(candidate, prepared, gate_index=0)
        target = tuple(fixture["target_bytes"])
        mapping = probe_mod.content_positions(fixture)
        previous = EOS
        with torch.no_grad():
            for position in range(len(target)):
                step = routing_step(
                    candidate,
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
                record, offset = mapping.get(position, (-1, -1))
                if record >= 0 and offset == 0:
                    world = "object" if position in probe_mod.OBJECT_POSITIONS else "place"
                    beta = int(torch.argmax(step.read.betas[record]))
                    logits = int(torch.argmax(step.pointer_logits[record]))
                    pointer = int(torch.argmax(step.pointer[record]))
                    rows[world].append(
                        {
                            "index": fixture["index"],
                            "beta_argmax_hit": beta == offset,
                            "pointer_logits_hit": logits == offset,
                            "pointer_hit": pointer == offset,
                            "beta_peak": beta,
                            "pointer_logits_peak": logits,
                            "pointer_peak": pointer,
                        }
                    )
                state, _coverage, gate_index = _advanced_state(
                    candidate, prepared, step, target[position]
                )
                previous = target[position]
                if position > 40:
                    break
    out = {}
    for world, items in rows.items():
        out[world] = {
            "examples": len(items),
            "beta_argmax_hit_rate": (
                statistics.fmean(item["beta_argmax_hit"] for item in items) if items else 0.0
            ),
            "pointer_logits_hit_rate": (
                statistics.fmean(item["pointer_logits_hit"] for item in items) if items else 0.0
            ),
            "pointer_hit_rate": (
                statistics.fmean(item["pointer_hit"] for item in items) if items else 0.0
            ),
            "pointer_peak_histogram": {
                str(peak): sum(1 for item in items if item["pointer_peak"] == peak)
                for peak in sorted({item["pointer_peak"] for item in items})
            },
        }
    return out


def main() -> int:
    start_cpu = time.process_time()
    start_wall = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_DIR))
    parser.add_argument("--step", default="step300")
    parser.add_argument("--report", default=str(REPORT))
    parser.add_argument("--untrained-span", type=int, default=0)
    arguments = parser.parse_args()
    directory = Path(arguments.output_dir)
    payload = premise.build_payload()
    dev = probe_mod.dev_slice(premise.fixtures(payload, "dev"))
    cells = {}
    if arguments.untrained_span:
        label = f"untrained-bottleneck-{arguments.untrained_span}"
        cells[label] = diagnose_cell(
            directory / "28031-A_route_lock-step300.pt",
            dev,
            untrained=arguments.untrained_span,
        )
    else:
        for seed in SEEDS:
            for arm in ARMS:
                path = directory / f"{seed}-{arm}-{arguments.step}.pt"
                if not path.exists():
                    raise FileNotFoundError(path)
                cells[f"{seed}-{arm}"] = diagnose_cell(path, dev)
    pooled = {}
    for world in ("object", "place"):
        pooled[world] = {
            key: statistics.fmean(cells[cell][world][key] for cell in cells)
            for key in ("beta_argmax_hit_rate", "pointer_logits_hit_rate", "pointer_hit_rate")
        }
    report = {
        "schema": "taiji-package19-2b-pointer-diagnostic-v1",
        "step": "N19.2b",
        "date": "2026-11-04",
        "bundle_dir": str(
            directory.relative_to(ROOT) if directory.is_relative_to(ROOT) else directory
        ),
        "checkpoint": arguments.step,
        "reading": "gold start position of the object/place record, teacher-forced prefix, "
        "model's own alpha route; beta_argmax_hit is the 19.5 start_hit definition, "
        "pointer_logits_hit / pointer_hit are the span-head quantities",
        "cells": cells,
        "pooled": pooled,
        "cost": {
            "cpu_seconds": time.process_time() - start_cpu,
            "wall_seconds": time.perf_counter() - start_wall,
            "adam_steps": 0,
        },
        "limits": {"zero_updates": True, "read_only": True, "confirm_reads": 0},
    }
    write_json(Path(arguments.report), report)
    print(json.dumps({"pooled": pooled, "report": str(arguments.report)})[:900])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
