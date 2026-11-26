"""19.2b pricing: re-price the six-cell dose from 19.5 measured receipts.

Zero optimizer steps, zero forward passes, zero confirm reads.  This script only
reads the finished 19.5 receipts and rewrites the price.  The 19.3 price was
built from a 4-step cold probe (~5.4 CPU/update) and then multiplied by 300; the
finished 19.5 run showed that rate was low by roughly 3x.  The correction is
recorded here as methodology, not as an excuse:

* measured first half (updates 0 -> 150) = each cell's step150 receipt cost;
* measured second half (150 -> 300) = run total minus step150 receipt;
* the second-half all-in rate is 20.8 - 26.7 CPU/update, matching the owner's
  measured band (20 - 29 CPU/update);
* the stage split used for pricing is selection 4 / copy 10 / joint 29 CPU per
  update: selection is the cheapest (19.3 cold probe: 2.625), copy carries the
  teacher-forced full graph (19.3 cold probe: 6.656; 19.5 first-half aggregate
  implies ~14-17 all-in), joint carries copy + route CE + curriculum and is the
  highest (19.5 second-half all-in, up to 26.7).

Segment budgets are produced per cell and for all six cells (2 seeds x 3 arms),
then protected by the frozen 2.11 contention factor.  The measured 19.5 totals
are printed next to the price as a cross-check; the price is intentionally above
the measured total rather than below it.

Run: ``python scripts/training/price_taiji_package19_19_2b.py``
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

COMPARE = ROOT / "reports/taiji_package19_compare_20261104.json"
PRICING_19_3 = ROOT / "reports/taiji_pricing_19_3_20261104.json"
CLOSEOUT = ROOT / "reports/taiji_package19_closeout_19_7_20261104.json"
REPORT = ROOT / "reports/taiji_pricing_19_2b_20261104.json"
MANIFEST = ROOT / "plans/manifests/taiji_pricing_19_2b_20261104.json"

SEEDS = (28031, 28032)
ARMS = ("A_route_lock", "B_stop_lock", "C_unlocked")
CELLS = tuple((seed, arm) for seed in SEEDS for arm in ARMS)

UPDATES = 300
SEGMENTS = ((0, 150), (150, 300))  # half-open, update counts
STAGE_UPDATES = {"selection": 100, "copy": 100, "joint": 100}

# Owner-fixed measured slopes (CPU per update).  selection ~4, copy ~10, joint
# highest in the 19.5 second-half band (20-29).
SLOPES = {"selection": 4.0, "copy": 10.0, "joint": 29.0}
PROTECTION = 2.11
WATCH_CPU = (20000.0, 30000.0)

# 19.5 measured non-training line items (CPU per cell), from the 19.3 component
# probe on the same graph and the 19.5 receipts.
STEP0_LIGHT_EVAL = 172.0  # 192 dev examples x 0.598 + copy probe generation
MID_LIGHT_EVAL = 172.0
FULL_EVAL = 732.0  # median of the six (timing - progress) full-eval receipts
SAVE = 0.05
RESTORE = 0.08


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def stage_span(low: int, high: int) -> dict[str, int]:
    """Update counts per frozen stage inside a half-open segment."""
    counts = {"selection": 0, "copy": 0, "joint": 0}
    for update in range(low + 1, high + 1):
        if update <= 100:
            counts["selection"] += 1
        elif update <= 200:
            counts["copy"] += 1
        else:
            counts["joint"] += 1
    return counts


def segment_train(segment: tuple[int, int]) -> float:
    counts = stage_span(*segment)
    return sum(counts[stage] * SLOPES[stage] for stage in counts)


def segment_nontrain(segment: tuple[int, int]) -> float:
    low, high = segment
    fixed = SAVE
    if low == 0:
        fixed += STEP0_LIGHT_EVAL + MID_LIGHT_EVAL
    else:
        fixed += FULL_EVAL
    return fixed


def measured() -> dict:
    compare = read_json(COMPARE)
    runs = compare["runs"]
    receipt_dir = ROOT / "output/taiji_package19_compare"
    rows = []
    for key, entry in sorted(runs.items()):
        timing = float(entry["timing"]["cpu_seconds"])
        mid_path = receipt_dir / f"midreport-{key}.json"
        if mid_path.exists():
            step150 = float(read_json(mid_path)["cost"]["cpu_seconds"])
        else:
            step150 = float(entry["step150"]["cost"]["cpu_seconds"])
        rows.append(
            {
                "cell": key,
                "first_half_cpu": step150,
                "second_half_cpu": timing - step150,
                "total_cpu": timing,
                "second_half_per_update": (timing - step150) / 150.0,
            }
        )
    first = sum(row["first_half_cpu"] for row in rows)
    second = sum(row["second_half_cpu"] for row in rows)
    total = sum(row["total_cpu"] for row in rows)
    rates = [row["second_half_per_update"] for row in rows]
    return {
        "cells": rows,
        "first_half_cpu_six": first,
        "second_half_cpu_six": second,
        "total_cpu_six": total,
        "second_half_per_update_band": [min(rates), max(rates)],
        "per_update_all_in": total / (len(rows) * UPDATES),
    }


def main() -> int:
    data = measured()
    pricing_19_3 = read_json(PRICING_19_3)
    closeout = read_json(CLOSEOUT)

    priced_cells = []
    for seed, arm in CELLS:
        segments = {}
        for segment in SEGMENTS:
            train = segment_train(segment)
            fixed = segment_nontrain(segment)
            segments[f"{segment[0]}_to_{segment[1]}"] = {
                "updates": segment[1] - segment[0],
                "stage_updates": stage_span(*segment),
                "train_cpu": train,
                "nontrain_cpu": fixed,
                "bare_cpu": train + fixed,
                "protected_cpu": (train + fixed) * PROTECTION,
            }
        bare = sum(item["bare_cpu"] for item in segments.values())
        protected = sum(item["protected_cpu"] for item in segments.values())
        priced_cells.append(
            {
                "seed": seed,
                "arm": arm,
                "segments": segments,
                "bare_cpu": bare,
                "protected_cpu": protected,
            }
        )

    segment_totals = {}
    for segment in SEGMENTS:
        key = f"{segment[0]}_to_{segment[1]}"
        bare = sum(cell["segments"][key]["bare_cpu"] for cell in priced_cells)
        protected = sum(cell["segments"][key]["protected_cpu"] for cell in priced_cells)
        segment_totals[key] = {
            "updates_per_cell": segment[1] - segment[0],
            "stage_updates_per_cell": stage_span(*segment),
            "bare_cpu_six": bare,
            "protected_cpu_six": protected,
            "train_cpu_six": sum(cell["segments"][key]["train_cpu"] for cell in priced_cells),
            "nontrain_cpu_six": sum(cell["segments"][key]["nontrain_cpu"] for cell in priced_cells),
        }

    bare_six = sum(cell["bare_cpu"] for cell in priced_cells)
    protected_six = sum(cell["protected_cpu"] for cell in priced_cells)
    ratio_to_measured = bare_six / data["total_cpu_six"]

    correction = {
        "19_3_bare_six_cpu": 1621.875 * 6,
        "19_3_bare_six_note": "single flat 5.4 CPU/update x 300 x 6, evals treated as rounding",
        "19_5_measured_six_cpu": data["total_cpu_six"],
        "19_5_over_19_3_ratio": data["total_cpu_six"] / (1621.875 * 6),
        "19_5_over_protected_19_3": data["total_cpu_six"]
        / (
            pricing_19_3["budget"]["six_cells_protected_cpu"]
            if "budget" in pricing_19_3 and "six_cells_protected_cpu" in pricing_19_3["budget"]
            else pricing_19_3["budget"]["six_cells_protected"]
        ),
        "root_causes": [
            "19.3 metered four cold optimizer steps (selection 2.625 / copy 6.656 / "
            "joint 6.938) and applied one flat rate to all 300 updates",
            "the 19.5 second half is the real cost: 20.8-26.7 CPU per update all-in "
            "(copy + joint + final full evaluation), not 5.4",
            "light and full dev evaluations were priced as negligible; measured "
            "step0+step150 light evaluations are ~344 CPU per cell and the full "
            "step300 evaluation ~732 CPU per cell",
        ],
        "correction": "price every stage at its measured 19.5 slope and list the "
        "measured evaluation line items explicitly; do not reuse the 5.4 flat rate",
    }

    report = {
        "schema": "taiji-package19-pricing-v2",
        "step": "N19.2b",
        "date": "2026-11-04",
        "dataset": {
            "file": "output/taiji_data_premise_18_0/data.json",
            "train_examples": 640,
            "dev_examples": 192,
            "confirm_reads": 0,
        },
        "design": {
            "cells": 6,
            "seeds": list(SEEDS),
            "arms": list(ARMS),
            "updates_per_cell": UPDATES,
            "stages": STAGE_UPDATES,
            "span_head": "SpanPointerHead with a 32-d bottleneck, 4,194 parameters",
            "unchanged": [
                "alpha/beta routing",
                "record encoder",
                "transport operator",
                "gate numbers",
                "no new data",
            ],
        },
        "slopes_cpu_per_update": SLOPES,
        "protection_factor": PROTECTION,
        "nontrain_line_items_cpu_per_cell": {
            "step0_light_eval": STEP0_LIGHT_EVAL,
            "step150_light_eval": MID_LIGHT_EVAL,
            "step300_full_eval": FULL_EVAL,
            "save": SAVE,
            "restore_preflight_only": RESTORE,
            "note": "measured on the same graph; the 19.3 price omitted these",
        },
        "methodology_correction_19_3": correction,
        "measured_19_5": data,
        "priced_cells": priced_cells,
        "segment_budget": segment_totals,
        "budget": {
            "six_cells_bare_cpu": bare_six,
            "six_cells_protected_cpu": protected_six,
            "bare_over_measured_19_5": ratio_to_measured,
            "six_cells_19_5_measured_cpu": data["total_cpu_six"],
            "watchlines_cpu": list(WATCH_CPU),
            "watchline_rule": "crossing 20,000 / 30,000 writes watchline and continues; "
            "the reason must be recorded, extrapolation must use the measured slope, "
            "never a fixed start-up cost",
            "hard_cap_protected_cpu": protected_six,
        },
        "comparison_baseline": {
            "19_5_six_step300_bundles": "output/taiji_package19_compare/ (kept, not overwritten)",
            "19_5_verdict": closeout.get("verdict", "negative"),
        },
        "confirm_policy": {
            "reads_allowed": 1,
            "only_if": "stage gate and final gate both pass",
            "sealed_after_read": True,
        },
        "zero_cost_confirmation": {
            "optimizer_steps": 0,
            "forward_passes": 0,
            "training": False,
            "confirm_reads": 0,
        },
    }
    write_json(REPORT, report)
    write_json(
        MANIFEST,
        {
            "schema": "taiji-package19-pricing-manifest-v2",
            "step": "N19.2b",
            "report": str(REPORT.relative_to(ROOT)),
            "budget_bare_cpu": bare_six,
            "budget_protected_cpu": protected_six,
            "segment_bare_cpu": {
                key: value["bare_cpu_six"] for key, value in segment_totals.items()
            },
            "segment_protected_cpu": {
                key: value["protected_cpu_six"] for key, value in segment_totals.items()
            },
            "slopes": SLOPES,
            "protection": PROTECTION,
            "watchlines": list(WATCH_CPU),
            "measured_19_5_cpu": data["total_cpu_six"],
        },
    )
    print(
        json.dumps(
            {
                "report": str(REPORT.relative_to(ROOT)),
                "manifest": str(MANIFEST.relative_to(ROOT)),
                "bare_six": round(bare_six, 1),
                "protected_six": round(protected_six, 1),
                "measured_19_5_six": round(data["total_cpu_six"], 1),
                "segments": {
                    key: {
                        "bare": round(value["bare_cpu_six"], 1),
                        "protected": round(value["protected_cpu_six"], 1),
                    }
                    for key, value in segment_totals.items()
                },
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
