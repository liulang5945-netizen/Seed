"""P3.12 input-side ceiling: can the position be decoded from raw RGB alone?

Zero training, no model. Ridge probe from the flattened 16x48 RGB frame
(train64) to the four-way position label, tested on fit32, with a shuffled-label
control. Decides whether the encoder loses information that the input carries.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training.check_report_run_conditions import write_report
from taiji.u2_history import atomic_load, value_digest

CONTRACT = ROOT / "plans/manifests/taiji_u2_p3_12_input_ceiling_20261114.json"
OUT = ROOT / "reports/taiji_u2_p3_12_input_ceiling_20261114.json"
LABELS = ["左边", "中间", "右边", "未知"]


def probe(train_x: torch.Tensor, train_y: torch.Tensor, test_x: torch.Tensor, lam: float) -> dict:
    mean = train_x.mean(0, keepdim=True)
    scale = train_x.std(0, keepdim=True).clamp_min(1e-6)
    x = (train_x - mean) / scale
    x_test = (test_x - mean) / scale
    onehot = torch.zeros(train_y.numel(), len(LABELS))
    onehot[torch.arange(train_y.numel()), train_y] = 1.0
    weights = torch.linalg.solve(x.T @ x + lam * torch.eye(x.shape[1]), x.T @ onehot)
    return {"predictions": (x_test @ weights).argmax(-1)}


def main() -> int:
    torch.set_num_threads(1)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["status"] == "frozen"
    current = hashlib.sha256(Path(__file__).read_text(encoding="utf-8-sig").encode()).hexdigest()
    if current != contract["source_pins"]["script"]:
        raise ValueError("script_source_drift")
    data = atomic_load(Path(contract["data_root"]))
    assert value_digest(data) == contract["data_digest"]
    train, fit = data["train"], data["fit"]

    def features(example: dict) -> torch.Tensor:
        return example["public"]["observations"]["rgb"].flatten()

    train_x = torch.stack([features(example) for example in train])
    test_x = torch.stack([features(example) for example in fit])
    train_y = torch.tensor([LABELS.index(example["target"].decode()) for example in train])
    test_y = torch.tensor([LABELS.index(example["target"].decode()) for example in fit])
    lam = contract["criteria"]["ridge_lambda"]
    predicted = probe(train_x, train_y, test_x, lam)["predictions"]
    correct = int((predicted == test_y).sum())
    generator = torch.Generator().manual_seed(contract["criteria"]["shuffle_seed"])
    shuffled = train_y[torch.randperm(train_y.numel(), generator=generator)]
    shuffled_predicted = probe(train_x, shuffled, test_x, lam)["predictions"]
    shuffled_correct = int((shuffled_predicted == test_y).sum())
    majority = int(torch.bincount(test_y, minlength=4).max()) / test_y.numel()

    accuracy = correct / test_y.numel()
    low = contract["criteria"]["probe_low"]
    high = contract["criteria"]["probe_high"]
    if accuracy >= high and shuffled_correct / test_y.numel() < accuracy - 0.2:
        verdict = "input_contains_position_encoder_loses_it"
    elif accuracy <= low:
        verdict = "input_split_not_linearly_separable"
    else:
        verdict = "inconclusive_input_ceiling"

    report = {
        "schema": "taiji-u2-p3-12-input-ceiling-v1",
        "date": "2026-11-14",
        "card": "22.5.P3.12",
        "phase": "input_ceiling",
        "verdict": verdict,
        "features": "flattened 16x48 rgb frame, 768 dimensions, standardized",
        "probe": {
            "correct": correct,
            "denominator": int(test_y.numel()),
            "accuracy": accuracy,
            "shuffled_control_accuracy": shuffled_correct / test_y.numel(),
            "majority_baseline": round(majority, 6),
            "ridge_lambda": lam,
            "protocol": "ridge fit on train64, test on fit32",
        },
        "discipline": {
            "optimizer_calls": 0,
            "model_forward_passes": 0,
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0,
            "capability_claims": 0,
        },
        "run_conditions": {"threads": torch.get_num_threads(), "torch": torch.__version__},
        "measured": {
            "cpu_seconds": round(time.process_time() - cpu_start, 6),
            "wall_seconds": round(time.perf_counter() - wall_start, 6),
        },
        "scope": "input-side ceiling only; no model, no training, no capability claim",
    }
    write_report(OUT, report)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "accuracy": accuracy,
                "shuffle": report["probe"]["shuffled_control_accuracy"],
                "measured": report["measured"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
