"""P3.15 aggregation attribution on the frozen 1844 bundle.

Compares two closed-form readouts of the same per-patch encoding:

* mean pooling over the 16 patches (the P3.11 protocol, 8 features),
* patch-preserving flattening (16 x 8 = 128 features).

Both fit on train64 and test on fit32, each with a shuffled-label control. The
question is whether the position is lost by aggregation or absent from the
representation. Zero training, no optimizer step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training.check_report_run_conditions import write_report
from scripts.training.first_a_support_22_5 import load_learner
from taiji.u2_history import atomic_load, value_digest

CONTRACT = ROOT / "plans/manifests/taiji_u2_p3_15_aggregation_20261114.json"
OUT = ROOT / "reports/taiji_u2_p3_15_aggregation_20261114.json"
LABELS = ["左边", "中间", "右边", "未知"]


def require_pin(contract: dict) -> None:
    current = hashlib.sha256(Path(__file__).read_text(encoding="utf-8-sig").encode()).hexdigest()
    if current != contract["source_pins"]["script"]:
        raise ValueError("script_source_drift")


def tokens(model, example: dict) -> torch.Tensor:
    public = example["public"]
    encoded = model.encode(public["observations"], public["coordinates"], public["outcome"])
    return encoded["rgb"]


def probe(
    train_x: torch.Tensor, train_y: torch.Tensor, test_x: torch.Tensor, lam: float
) -> torch.Tensor:
    mean = train_x.mean(0, keepdim=True)
    scale = train_x.std(0, keepdim=True).clamp_min(1e-6)
    x = (train_x - mean) / scale
    x_test = (test_x - mean) / scale
    onehot = torch.zeros(train_y.numel(), len(LABELS))
    onehot[torch.arange(train_y.numel()), train_y] = 1.0
    weights = torch.linalg.solve(x.T @ x + lam * torch.eye(x.shape[1]), x.T @ onehot)
    return (x_test @ weights).argmax(-1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["status"] == "frozen"
    require_pin(contract)
    data = atomic_load(Path(contract["data_root"]))
    assert value_digest(data) == contract["data_digest"]
    model, _, _ = load_learner(Path(contract["parent_bundle"]))
    train, fit = data["train"], data["fit"]
    with torch.no_grad():
        train_tokens = torch.stack([tokens(model, example) for example in train])
        fit_tokens = torch.stack([tokens(model, example) for example in fit])
    train_y = torch.tensor([LABELS.index(example["target"].decode()) for example in train])
    fit_y = torch.tensor([LABELS.index(example["target"].decode()) for example in fit])
    lam = contract["criteria"]["ridge_lambda"]
    pooled_train = train_tokens.mean(1)
    pooled_fit = fit_tokens.mean(1)
    flat_train = train_tokens.flatten(1)
    flat_fit = fit_tokens.flatten(1)
    generator = torch.Generator().manual_seed(contract["criteria"]["shuffle_seed"])
    shuffled = train_y[torch.randperm(train_y.numel(), generator=generator)]

    def score(prediction: torch.Tensor) -> dict:
        correct = int((prediction == fit_y).sum())
        return {
            "correct": correct,
            "denominator": int(fit_y.numel()),
            "accuracy": correct / fit_y.numel(),
        }

    pooled = score(probe(pooled_train, train_y, pooled_fit, lam))
    flat = score(probe(flat_train, train_y, flat_fit, lam))
    pooled_shuffle = score(probe(pooled_train, shuffled, pooled_fit, lam))
    flat_shuffle = score(probe(flat_train, shuffled, flat_fit, lam))
    high = contract["criteria"]["probe_high"]
    low = contract["criteria"]["probe_low"]
    if flat["accuracy"] >= high and flat_shuffle["accuracy"] <= contract["criteria"]["shuffle_max"]:
        verdict = "aggregation_is_the_bottleneck"
    elif flat["accuracy"] <= low:
        verdict = "representation_is_the_bottleneck"
    else:
        verdict = "inconclusive_aggregation"
    report = {
        "schema": "taiji-u2-p3-15-aggregation-v1",
        "date": "2026-11-14",
        "card": "22.5.P3.15",
        "phase": "aggregation_attribution",
        "verdict": verdict,
        "parent_bundle": contract["parent_bundle"],
        "protocol": "ridge fit on train64, test on fit32; features standardized",
        "pooled": pooled,
        "patch_preserving": flat,
        "pooled_shuffled": pooled_shuffle,
        "patch_preserving_shuffled": flat_shuffle,
        "features": {"pooled": 8, "patch_preserving": int(flat_train.shape[1])},
        "discipline": {
            "optimizer_calls": 0,
            "checkpoints_written": 0,
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
        "scope": "attribution only; one bundle, train64 and fit32, no training and no capability claim",
    }
    if not args.no_write:
        write_report(OUT, report)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "pooled": pooled["accuracy"],
                "flat": flat["accuracy"],
                "pooled_shuffle": pooled_shuffle["accuracy"],
                "flat_shuffle": flat_shuffle["accuracy"],
                "measured": report["measured"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
