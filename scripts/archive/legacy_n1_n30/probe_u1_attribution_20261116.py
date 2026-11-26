"""22.5.U.1 attribution probe (diagnostic, not the frozen gate).

U.1 failed the read prerequisite at the frozen 384 updates/seed: fit 0.22-0.41,
train 0.36-0.41 (the graph barely fits even the training set), RGB-zeroed
agreement ~1.0 (the readout largely ignores RGB). This probe separates dose from
architecture: (1) dose-response on one seed up to 3000 updates, (2) RGB vs
question input-encoder gradient credit at start, (3) prediction class distribution.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training import run_u1_unit_read_20261116 as u1

REPORT = ROOT / "reports/taiji_u2_u1_attribution_20261116.json"
SEED = 2026111602
DOSES = [384, 768, 1536, 3000]


def main() -> None:
    torch.set_num_threads(1)
    started = time.process_time()
    contract = json.loads(u1.CONTRACT.read_text(encoding="utf-8"))
    data = u1.load_data(contract)
    episodes = u1.prepare_episodes(data, u1.UnitGraphSpec().n_units)
    graph = u1.build_graph(SEED)
    optimizer = torch.optim.Adam(graph.parameters(), lr=contract["criteria"]["learning_rate"])

    # gradient credit to rgb vs question input encoders at start.
    item = u1.cast_item(episodes["train"][0], graph.spec.torch_dtype)
    graph.zero_grad(set_to_none=True)
    logits = graph(item["channels"], item["gates"], graph.initial_state())["readout"].mean(dim=0)
    target = torch.tensor(item["label"], dtype=torch.long)
    torch.nn.functional.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0)).backward()
    credit = {
        name: float(param.grad.abs().sum())
        for name, param in graph.named_parameters()
        if param.grad is not None
    }
    rgb_credit = credit.get("input_encoders.rgb.weight", 0.0)
    question_credit = credit.get("input_encoders.question.weight", 0.0)

    dose_rows = []
    done = 0
    for dose in DOSES:
        while done < dose:
            u1.one_update(graph, optimizer, episodes["train"][done % len(episodes["train"])])
            done += 1
        fit = u1.evaluate(graph, episodes["fit"])
        train = u1.evaluate(graph, episodes["train"])
        from collections import Counter

        dose_rows.append(
            {
                "updates": dose,
                "fit_accuracy": fit["accuracy"],
                "train_accuracy": train["accuracy"],
                "fit_distinct": fit["distinct"],
                "fit_pred_distribution": dict(Counter(fit["preds"])),
            }
        )
    report = {
        "schema": "taiji-u2-u1-attribution-v1",
        "card": "22.5.U.1",
        "phase": "attribution",
        "seed": SEED,
        "start_gradient_credit": {
            "rgb_encoder": rgb_credit,
            "question_encoder": question_credit,
            "readout_weight": credit.get("readout.weight", 0.0),
            "units_r": credit.get("units.r", 0.0),
            "units_p": credit.get("units.p", 0.0),
            "query": credit.get("query", 0.0),
        },
        "dose_response": dose_rows,
        "discipline": {
            "frozen_gate_changed": False,
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0,
        },
        "cost": {"cpu": round(time.process_time() - started, 4)},
    }
    if REPORT.exists():
        raise FileExistsError(REPORT)
    REPORT.write_bytes(json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8") + b"\n")
    print(
        json.dumps(
            {
                "rgb_credit": rgb_credit,
                "question_credit": question_credit,
                "dose": [(d["updates"], d["train_accuracy"], d["fit_accuracy"]) for d in dose_rows],
                "cpu": report["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
