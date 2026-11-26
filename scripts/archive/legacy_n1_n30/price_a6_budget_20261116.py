"""Price A.6 (the per-error update-budget curve) on the saved A.5 bundles.

The A.5 run showed the correction rate at zero because every single update was
rolled back.  The next question is what a fixed budget per error buys when the
rollback clause is removed: k = 1, 2, 4, 8 updates per error, same first pass,
same subjects, same read.

This script measures the units the card will be charged in - it starts no
training run and writes no checkpoint.  Every unit is measured on the saved
pre-feedback bundle from A.5, because that is exactly where the card starts.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training import run_a2_byte_20261115 as a2
from scripts.training import run_a5_22_5 as a5
from taiji.u2_history import atomic_load

CONTRACT = json.loads(
    (ROOT / "plans/manifests/taiji_u2_a5_protocol_20261116.json").read_text(encoding="utf-8")
)
REPORT = ROOT / "reports/taiji_u2_a6_budget_price_20261116.json"
BUNDLE = ROOT / "data/experiments/taiji_u2/a5/20261116T0520Z/seed-20261110/bundle-pre-feedback.pt"
PLAN_SEEDS = [20261110, 20261111, 20261112]
BUDGETS = [1, 2, 4, 8]
REPEATS = 8


def timed(function, repeats: int = REPEATS) -> tuple[float, list[float]]:
    samples = []
    for _ in range(repeats):
        start = time.process_time()
        function()
        samples.append(time.process_time() - start)
    return statistics.median(samples), samples


def main() -> None:
    torch.set_num_threads(1)
    cpu_start = time.process_time()
    data = atomic_load(ROOT / CONTRACT["data_root"])
    items, _ = a5.task_b_examples(data, CONTRACT)
    model = a5.build_subject(CONTRACT)
    model.load_state_dict(torch.load(BUNDLE, weights_only=False)["model"])
    model.eval()
    a5.set_trainable(model, CONTRACT)
    head = a2.load_head(CONTRACT)
    prior = a2.prior_of(data)
    codes = a2.code_by_id(model, head, prior, data["train"] + data["fit"], CONTRACT)
    optimizer = a5.optimizer_for(
        [p for p in model.parameters() if p.requires_grad], CONTRACT["feedback_optimizer"]
    )
    item = items[0]
    public = a5.p4_2.column_public(item["example"])
    code = codes[item["example"]["id"]]

    def one_update() -> None:
        optimizer.zero_grad(set_to_none=True)
        loss = a5.record_loss(
            model, public, item["example"]["id"], item["records"][0], item["target"], code
        )
        loss.backward()
        optimizer.step()

    update_unit, update_samples = timed(one_update)
    round_unit, round_samples = timed(lambda: a5.b_round(model, item, code, 0), repeats=4)
    a_unit, a_samples = timed(
        lambda: a5.a_round(model, data["fit"][0], codes[data["fit"][0]["id"]]), repeats=4
    )

    errors = len(items)
    if True:
        updates = sum(BUDGETS) * errors
        rounds = len(BUDGETS) * errors
        a_evals = len(BUDGETS) * len(data["fit"])
        per_seed = {
            "updates": updates,
            "second_pass_rounds": rounds,
            "a_fit_rounds": a_evals,
            "update_cpu": round(updates * update_unit, 6),
            "round_cpu": round(rounds * round_unit, 6),
            "a_cpu": round(a_evals * a_unit, 6),
            "seed_cpu": round(updates * update_unit + rounds * round_unit + a_evals * a_unit, 6),
        }
    seeds = len(PLAN_SEEDS)
    estimate = per_seed["seed_cpu"] * seeds
    cap = round(estimate * 1.15, 3)
    payload = {
        "schema": "taiji-u2-a6-budget-price-v1",
        "card": "22.5.A.6",
        "date": "2026-11-16",
        "probe_start": str(BUNDLE.relative_to(ROOT)).replace("\\", "/"),
        "units_cpu": {
            "record_update": round(update_unit, 6),
            "b_round": round(round_unit, 6),
            "a_round": round(a_unit, 6),
        },
        "unit_samples": {
            "record_update": [round(value, 6) for value in update_samples],
            "b_round": [round(value, 6) for value in round_samples],
            "a_round": [round(value, 6) for value in a_samples],
        },
        "plan": {
            "seeds": PLAN_SEEDS,
            "budgets": BUDGETS,
            "errors_per_arm": errors,
            "per_seed": per_seed,
            "estimate_cpu": round(estimate, 3),
            "contingency": 0.15,
            "cap_cpu": cap,
            "cap_wall": round(cap * 2, 3),
        },
        "discipline": {
            "subject_training_updates": 0,
            "checkpoints_written": 0,
            "capability_claims": 0,
            "accepted_parent": False,
        },
        "note": (
            "the units are measured on the A.5 pre-feedback bundle because the card starts there; "
            "a record-conditioned update rebuilds the episode (legal release included), which is "
            "the term A.5's price table underestimated by three times"
        ),
        "cost": {"cpu": round(time.process_time() - cpu_start, 6)},
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps({k: payload["units_cpu"][k] for k in payload["units_cpu"]}, ensure_ascii=False)
    )
    print(json.dumps(payload["plan"], ensure_ascii=False))


main()
