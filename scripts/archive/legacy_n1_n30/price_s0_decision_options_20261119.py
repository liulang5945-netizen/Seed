"""Zero-training price pass for the three S0 decision options (D-20261119-09 §7).

Option 1 is the proposed bounded D-side localization package (organ operating
point only, no architecture/parameter change).  Option 2 is re-earning the A
parent under the ``pixel-salience-v2`` identity so all three seeds have a
strong parent.  Option 3 is the zero-training alternative: a single strong-seed
v1 accepted parent with the joint-gate D criterion restated relative to the
parent.

Everything here is measured on throwaway models with one-off backward calls
(threads=1, discarded optimizer state) plus sourced numbers from already
executed cards; zero formal training, zero check/final reads.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from scripts.training.first_a_support_22_5_v2 import (  # noqa: E402
    generate_examples as generate_v2,
)
from scripts.training.run_a_gate_pixel_20261117 import (  # noqa: E402
    FIRST_WEIGHT,
    masked_loss,
)
from taiji.u2_record_organ import RecordOrgan  # noqa: E402
from taiji.u2_subject import U2Config, U2Subject  # noqa: E402

OUT = ROOT / "reports/taiji_u2_s0_decision_options_price_20261119.json"
A_GATE_PRICE = ROOT / "reports/taiji_u2_a_gate_package_price_20261119.json"
S0_DIAGNOSTIC = ROOT / "reports/taiji_u2_s0_diagnostic_3seed_20261119.json"
LOAD_PROBE = ROOT / "reports/taiji_u2_s0_accepted_parent_load_20261119.json"
DOSE = 2000
BATCH = 8
SEEDS = 3


def timed(count: int, fn) -> float:
    started = time.process_time()
    for _ in range(count):
        fn()
    return (time.process_time() - started) / count


def build(identity: str) -> U2Subject:
    return (
        U2Subject(replace(U2Config.small(), seed=20261110, readout=identity, d=32))
        .metadata_candidate()
        .request_candidate()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    a_gate = json.loads(A_GATE_PRICE.read_text(encoding="utf-8"))
    s0 = json.loads(S0_DIAGNOSTIC.read_text(encoding="utf-8"))
    load_probe = json.loads(LOAD_PROBE.read_text(encoding="utf-8"))
    frozen_unit = a_gate["measured"]["per_update_cpu"]
    fixture_leg = a_gate["price"]["fixture_leg_cpu_per_seed"]
    formal_leg = a_gate["sourced"]["formal_leg_cpu_allowance"]
    check_worker = a_gate["sourced"]["check_worker_cpu"]
    organ_steps = s0["config"]["organ_steps"]
    organ_cpu_per_seed = max(row["phase_cpu"]["organ_train"] for row in s0["seeds"])
    suite_cpu_per_seed = max(row["phase_cpu"]["evaluate"] for row in s0["seeds"])

    v2_examples = generate_v2("train", 8)
    # The fixture is identity independent; both arms read the same public
    # examples so the ratio isolates the readout identity.
    v1_model, v2_model = build("pixel-salience"), build("pixel-salience-v2")
    for model in (v1_model, v2_model):
        masked_loss(model, v2_examples[:1], FIRST_WEIGHT, False).backward()
        model.zero_grad(set_to_none=True)
    v1_per_example = timed(
        20,
        lambda: (
            masked_loss(v1_model, v2_examples[:1], FIRST_WEIGHT, False).backward(),
            v1_model.zero_grad(set_to_none=True),
        ),
    )
    v2_per_example = timed(
        20,
        lambda: (
            masked_loss(v2_model, v2_examples[:1], FIRST_WEIGHT, False).backward(),
            v2_model.zero_grad(set_to_none=True),
        ),
    )
    ratio = v2_per_example / v1_per_example
    v2_per_update = frozen_unit * ratio

    organ = RecordOrgan()
    request = "红色圆片现在在哪里？".encode()
    records = [
        "通知：红色圆片改为左边。".encode(),
        "最新记录：蓝色圆片现在在右边。".encode(),
    ]
    organ.plan(request, records)  # warmup
    plan_per_call = timed(20, lambda: organ.plan(request, records))

    localization_seeds = [20261110, 20261112]
    localization_per_seed = organ_cpu_per_seed + suite_cpu_per_seed + 6 * plan_per_call
    localization_total = len(localization_seeds) * localization_per_seed + 20.0
    localization_cap = localization_total * 1.25

    v2_train_leg = DOSE * v2_per_update
    v2_arm_per_seed = v2_train_leg + fixture_leg + check_worker + formal_leg
    v2_total = SEEDS * v2_arm_per_seed
    v2_cap = v2_total * 1.25

    payload = {
        "schema": "taiji-u2-s0-decision-options-price-v1",
        "card": "22.5.1/S0 decision options (D-20261119-09 §7)",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "measured": {
            "v1_per_example_cpu": round(v1_per_example, 6),
            "v2_per_example_cpu": round(v2_per_example, 6),
            "v2_over_v1_ratio": round(ratio, 4),
            "frozen_v1_per_update_cpu": frozen_unit,
            "v2_per_update_cpu": round(v2_per_update, 6),
            "organ_plan_per_call_cpu": round(plan_per_call, 6),
            "a_gate_fixture64_cpu": a_gate["measured"]["fixture64_cpu"],
            "s0_organ_cpu_per_seed_measured": round(organ_cpu_per_seed, 2),
            "s0_suite_cpu_per_seed_measured": round(suite_cpu_per_seed, 2),
        },
        "factual_base": {
            "accepted_parent_load_verdict": load_probe["verdict"],
            "accepted_parent_fit_recorded_vs_loaded": {
                str(row["seed"]): [row.get("recorded_fit_correct"), row.get("loaded_fit_correct")]
                for row in load_probe["rows"]
            },
            "accepted_parent_pin_mismatch": {
                str(row["seed"]): row.get("source_pin_mismatches") for row in load_probe["rows"]
            },
            "note": (
                "v1 accepted parents 3/3 load strictly and reproduce the recorded fit after the "
                "delta-head v1 layout restoration; only taiji/u2_subject.py pin differs."
            ),
        },
        "options": {
            "1_d_localization": {
                "what": (
                    "bounded D-side operating-point localization on the two strong accepted v1 "
                    "parents; organ-only 300-step training, frozen inference-time bias sweep, no "
                    "architecture/parameter change, no criterion change"
                ),
                "seeds": localization_seeds,
                "per_seed_cpu": round(localization_per_seed, 2),
                "price_cpu": round(localization_total, 2),
                "cap_cpu_proposed": round(localization_cap, 2),
                "cap_wall_minutes": 20,
                "optimizer_steps": len(localization_seeds) * organ_steps,
                "check_final_reads": 0,
                "training_authority": None,
                "model": "priced, not authorized",
            },
            "2_v2_a_parent_reearn": {
                "what": (
                    "new A-parent package under pixel-salience-v2 so all three seeds have a strong "
                    "fit parent before the joint gate; needs a fresh A-gate contract and a new "
                    "check-exposure ruling (22.5 check 2/2 already used)"
                ),
                "dose_per_seed": DOSE,
                "batch": BATCH,
                "seeds": SEEDS,
                "train_leg_cpu_per_seed": round(v2_train_leg, 2),
                "fixture_leg_cpu_per_seed": fixture_leg,
                "check_worker_cpu": check_worker,
                "formal_leg_cpu": formal_leg,
                "per_seed_cpu": round(v2_arm_per_seed, 2),
                "price_cpu": round(v2_total, 2),
                "cap_cpu_proposed": round(v2_cap, 2),
                "check_exposure": "requires owner ruling; 22.5 ledger 2/2 used",
                "training_authority": None,
                "model": "priced, not authorized",
            },
            "3_v1_single_candidate": {
                "what": (
                    "single strong-seed accepted v1 parent (20261110 fit 30/32 or 20261112 fit "
                    "31/32) as the joint-gate main candidate; D criterion restated as "
                    "child >= parent - 0.02 one-sided upper bound; seed 20261111 excluded by "
                    "structural parent bound (fit 14/32) and must not be pooled"
                ),
                "training_cpu": 0.0,
                "prerequisites": [
                    "owner amendment: single primary candidate + relative-parent D criterion",
                    "S2 zero-training design/pricing then still follows",
                ],
                "training_authority": None,
                "model": "zero training",
            },
        },
        "recommendation": (
            "option 1 first (cheapest localization of the failure), then decide between option 3 "
            "(zero training) and option 2 (new identity A parent) on its result"
        ),
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {
                "measured": payload["measured"],
                "option1": {
                    key: payload["options"]["1_d_localization"][key]
                    for key in ("price_cpu", "cap_cpu_proposed")
                },
                "option2": {
                    key: payload["options"]["2_v2_a_parent_reearn"][key]
                    for key in ("price_cpu", "cap_cpu_proposed")
                },
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
