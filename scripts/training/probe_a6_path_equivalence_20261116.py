"""Post-hoc check: A.5's A4 arm and A.6's arm are the same update path.

The A.6 result card left one unresolved discrepancy: A.5's attribution arm
reported 0.333 correction at eight updates per error (192 updates), while A.6
read 0.0 at four (96). Two explanations were open - a threshold between the two
budgets, or a difference between the two update implementations.

This probe gives both paths the same A.5 pre-feedback bundle, the same single
error and one update, and compares their parameter summaries and their A-task
round digests. It takes two optimizer steps on throwaway copies: it is an
equivalence check, not a card, and it writes no checkpoint and no ability claim.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training import run_a2_byte_20261115 as a2
from scripts.training import run_a5_22_5 as a5
from scripts.training import run_a6_budget_20261116 as a6
from taiji.u2_history import atomic_load

REPORT = ROOT / "reports/taiji_u2_a6_path_equivalence_20261116.json"
C6 = json.loads(
    (ROOT / "plans/manifests/taiji_u2_a6_budget_20261116.json").read_text(encoding="utf-8")
)
C5 = json.loads(
    (ROOT / "plans/manifests/taiji_u2_a5_protocol_20261116.json").read_text(encoding="utf-8")
)


def main() -> None:
    torch.set_num_threads(1)
    started = time.process_time()
    data = atomic_load(ROOT / C6["data_root"])
    items, _ = a5.task_b_examples(data, C6)
    head = a2.load_head(C6)
    prior = a2.prior_of(data)
    seed = int(C6["plan"]["seeds"][0])

    a = a6.load_start(C6, seed)
    a5.set_trainable(a, C5)
    codes = a2.code_by_id(a, head, prior, data["train"] + data["fit"], C6)
    item = items[0]

    one = dict(C5)
    one["feedback_updates_a4"] = 1
    stats5 = a5.protocol_accumulated_a4(a, [item], codes, one, lambda: None, lambda index: 0)
    abs5 = sum(
        float(parameter.detach().abs().sum())
        for parameter in a.parameters()
        if parameter.requires_grad
    )
    rows5 = a5.b_pass(a, [item], codes, lambda index: 0)
    path5 = {
        "updates": stats5["updates"],
        "parameter_abs_sum": round(abs5, 8),
        "correct": sum(int(row["correct"]) for row in rows5),
        "output_hex": rows5[0]["output_hex"],
        "a_fit_digest": a5.rows_digest(
            [a5.a_round(a, example, codes[example["id"]]) for example in data["fit"]]
        ),
    }

    stats6 = a6.arm(C6, seed, 1, codes, [item], [0], data, lambda: None)
    path6 = {
        "updates": stats6["updates"],
        "parameter_abs_sum": round(stats6["parameter_abs_sum"], 8),
        "correct": stats6["correct"],
        "a_fit_digest": stats6["a_fit_digest"],
    }

    equivalent = (
        path5["updates"] == path6["updates"]
        and abs(path5["parameter_abs_sum"] - path6["parameter_abs_sum"]) < 1e-6
        and path5["a_fit_digest"] == path6["a_fit_digest"]
        and path5["correct"] == path6["correct"]
    )
    payload = {
        "schema": "taiji-u2-a6-path-equivalence-v1",
        "card": "22.5.A.6",
        "date": "2026-11-16",
        "verdict": "paths_equivalent_at_one_update" if equivalent else "paths_differ",
        "same_start_bundle": C6["start_bundles"][str(seed)],
        "a5_path": path5,
        "a6_path": path6,
        "parameter_abs_delta": round(path5["parameter_abs_sum"] - path6["parameter_abs_sum"], 10),
        "conclusion": (
            "the two update paths agree at one update on the same start bundle and error, so the "
            "0.333 against 0.0 discrepancy is not an implementation difference at this level; "
            "what remains open is the k=8 point itself (a threshold between 96 and 192 updates)"
            if equivalent
            else "the paths differ at the first update; the 0.333 against 0.0 gap is an implementation issue"
        ),
        "limits": (
            "one error, one update: equivalence here does not prove equivalence over 192 updates, "
            "and the k=8 arm still has not been measured"
        ),
        "discipline": {
            "optimizer_updates": 2,
            "checkpoints_written": 0,
            "capability_claims": 0,
            "accepted_parent": False,
        },
        "cost": {"cpu": round(time.process_time() - started, 6)},
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "verdict": payload["verdict"],
                "abs_delta": payload["parameter_abs_delta"],
                "same_digest": path5["a_fit_digest"] == path6["a_fit_digest"],
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )


main()
