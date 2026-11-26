"""Zero-training prior correction for the A.1b colour head (A.1b-cal).

The head's raw argmax over-predicts red/background. Subtracting the log class
prior of the train64 columns rebalances it. No gradient step, no new parameter,
no fit label is used for the prior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training import run_p4_1_22_5 as p4_1
from scripts.training import run_p4_2_22_5 as p4_2
from scripts.training.check_report_run_conditions import write_report
from scripts.training.run_p5_4_a1b_colour import (
    ColourHead,
    build_subject,
    colour_of,
    column_tokens,
    position_of,
)
from taiji.u2_history import atomic_load

CONTRACT = ROOT / "plans/manifests/taiji_u2_a1b_colour_20261115.json"
A1B_REPORT = ROOT / "reports/taiji_u2_a1b_colour_20261115.json"
REPORT = ROOT / "reports/taiji_u2_a1b_cal_20261115.json"
GUARD_REPORT = ROOT / "reports/taiji_u2_a1b_cal_guard_20261115.json"
LABELS = ("左边", "中间", "右边", "未知")


def train_prior(examples: list[dict]) -> torch.Tensor:
    counts = torch.zeros(7)
    for example in examples:
        counts += torch.bincount(p4_1.column_labels(example), minlength=7)
    return counts / counts.sum()


def read(model, head, example: dict, prior: torch.Tensor | None) -> int:
    with torch.no_grad():
        logits = head(column_tokens(model, example))
        if prior is not None:
            logits = logits - torch.log(prior)
        predicted = logits.argmax(-1).tolist()
    columns = [index for index, label in enumerate(predicted) if label == colour_of(example)]
    if not columns:
        return 3
    return position_of(2 * (sum(columns) / len(columns) + 0.5) / 16 - 1)


def accuracy(model, head, examples: list[dict], prior: torch.Tensor | None) -> dict:
    predictions = [read(model, head, example, prior) for example in examples]
    correct = sum(
        int(LABELS[prediction] == example["target"].decode())
        for prediction, example in zip(predictions, examples, strict=True)
    )
    return {
        "correct": correct,
        "denominator": len(examples),
        "accuracy": correct / len(examples),
        "predictions": predictions,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8-sig"))
    a1b = json.loads(A1B_REPORT.read_text(encoding="utf-8"))
    data = atomic_load(Path(contract["data"]["root"]))
    model = build_subject(contract)
    checkpoint = torch.load(ROOT / a1b["checkpoint"], weights_only=False)
    head = ColourHead()
    head.load_state_dict(checkpoint["head"])
    prior = train_prior(data["train"])
    fit_prior = train_prior(data["fit"])
    painted = prior.clone()
    painted[6] *= 0.05
    painted = painted / painted.sum()
    rows = {}
    for name, value in (("none", None), ("train_prior", prior), ("painted_boost", painted)):
        rows[name] = {
            "fit": accuracy(model, head, data["fit"], value),
            "train": accuracy(model, head, data["train"], value),
        }
    even, odd = data["train"][0::2], data["train"][1::2]
    cross = {
        "prior_even_on_odd": accuracy(model, head, odd, train_prior(even)),
        "prior_odd_on_even": accuracy(model, head, even, train_prior(odd)),
    }
    payload = {
        "schema": "taiji-u2-a1b-cal-v1",
        "card": "22.5.A.1b-cal",
        "date": "2026-11-15",
        "status": "post_hoc_zero_training",
        "verdict": "a_parent_read_calibrated_post_hoc",
        "a1b_checkpoint": a1b["checkpoint"],
        "a1b_checkpoint_sha256": hashlib.sha256(
            (ROOT / a1b["checkpoint"]).read_bytes()
        ).hexdigest(),
        "prior_source": "train64_column_labels",
        "fit_labels_used_in_prior": False,
        "prior": [round(float(value), 8) for value in prior],
        "fit_prior_for_red_direction": [round(float(value), 8) for value in fit_prior],
        "rows": {
            name: {
                "fit": {key: row["fit"][key] for key in ("correct", "denominator", "accuracy")},
                "train": {key: row["train"][key] for key in ("correct", "denominator", "accuracy")},
            }
            for name, row in rows.items()
        },
        "cross_split": {
            name: {key: row[key] for key in ("correct", "denominator", "accuracy")}
            for name, row in cross.items()
        },
        "caveats": {
            "post_hoc": True,
            "fit_was_seen_before_this_correction": True,
            "cross_split_min": min(row["accuracy"] for row in cross.values()),
            "check2": "not_read",
            "accepted_parent": False,
            "clean_confirmation": "requires a pre-registered frozen card or a held-out split",
        },
        "cost": {"cpu": 0.0, "wall": 0.0},
        "discipline": {
            "training": 0,
            "optimizer_steps": 0,
            "check_reads": 0,
            "final_reads": 0,
            "capability_claims": 0,
            "accepted_parent": False,
        },
        "run_conditions": {"threads": 1, "torch": torch.__version__},
        "source_pins": {
            "runner": p4_2.sha_of(Path(__file__)),
            "a1b_runner": p4_2.sha_of(ROOT / "scripts/training/run_p5_4_a1b_colour.py"),
            "a1b_contract": p4_2.sha_of(CONTRACT),
        },
    }
    write_report(REPORT, payload)
    print(
        json.dumps(
            {
                "verdict": payload["verdict"],
                "fit": rows["train_prior"]["fit"]["accuracy"],
                "train": rows["train_prior"]["train"]["accuracy"],
                "cross_min": payload["caveats"]["cross_split_min"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
