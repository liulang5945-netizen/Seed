"""P3.11 conditioning attribution on the P3.10 final bundle.

Zero training. Measures three things on the frozen 1844-update bundle:

1. whether the encoded observation changes when RGB is zeroed,
2. how much the encoding varies across fit scenes,
3. whether a closed-form ridge probe can decode the position label from the
   encoding (fit on train64, test on fit32) and from the readout context.

Also reports grouped gradient norms on one batch as a descriptive reading.
No optimizer step, no checkpoint, no sealed read.
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
from scripts.training.first_a_support_22_5 import load_learner, prepare, task_loss
from taiji.u2_history import atomic_load, value_digest

CONTRACT = ROOT / "plans/manifests/taiji_u2_p3_11_attribution_20261114.json"
OUT = ROOT / "reports/taiji_u2_p3_11_attribution_20261114.json"
LABELS = ["左边", "中间", "右边", "未知"]
ORDER = ("rgb", "audio", "proprio", "outcome")


def pooled(tokens: dict) -> torch.Tensor:
    parts = [tokens[name].mean(0) for name in ORDER if name in tokens]
    return torch.cat(parts)


def embed(model, example: dict, zero_rgb: bool = False) -> torch.Tensor:
    public = example["public"]
    observations = public["observations"]
    if zero_rgb:
        observations = dict(observations)
        observations["rgb"] = torch.zeros_like(observations["rgb"])
    tokens = model.encode(observations, public["coordinates"], public["outcome"])
    return pooled(tokens)


def context_vector(model, example: dict) -> torch.Tensor:
    public = example["public"]
    state = prepare(model, public, example["id"])
    goal = model.goal(public["goal"])
    return model.branch_context(state, goal, public["action"]).mean(0)


def ridge_probe(
    train_x: torch.Tensor, train_y: list[int], test_x: torch.Tensor, test_y: list[int], lam: float
) -> dict:
    mean = train_x.mean(0, keepdim=True)
    scale = train_x.std(0, keepdim=True).clamp_min(1e-6)
    x = (train_x - mean) / scale
    x_test = (test_x - mean) / scale
    onehot = torch.zeros(len(train_y), len(LABELS))
    for row, label in enumerate(train_y):
        onehot[row, label] = 1.0
    gram = x.T @ x + lam * torch.eye(x.shape[1])
    weights = torch.linalg.solve(gram, x.T @ onehot)
    predictions = (x_test @ weights).argmax(-1)
    correct = int((predictions == torch.tensor(test_y)).sum())
    return {
        "correct": correct,
        "denominator": len(test_y),
        "accuracy": correct / len(test_y),
        "predictions": predictions.tolist(),
    }


def label_index(target: bytes) -> int:
    return LABELS.index(target.decode())


def grouped_gradients(model, batch: list[dict]) -> dict[str, float]:
    model.zero_grad(set_to_none=True)
    loss = task_loss(model, batch)
    loss.backward()
    groups: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if name.startswith("input_encoders"):
            key = "input_encoders"
        elif name.startswith("text_input") or name.startswith("goal"):
            key = "text_or_goal"
        elif name.startswith("units"):
            key = "units"
        elif name.startswith("w_g") or name.startswith("h_"):
            key = "h_path"
        else:
            key = "other"
        groups[key] = groups.get(key, 0.0) + float(parameter.grad.norm()) ** 2
    model.zero_grad(set_to_none=True)
    return {key: round(value**0.5, 6) for key, value in groups.items()}


def require_pin(contract: dict) -> None:
    """Fail closed when the script bytes drifted from the frozen pin."""

    current = hashlib.sha256(Path(__file__).read_text(encoding="utf-8-sig").encode()).hexdigest()
    if current != contract["source_pins"]["script"]:
        raise ValueError("script_source_drift")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if contract["status"] != "frozen":
        raise ValueError("contract_not_frozen")
    require_pin(contract)
    data = atomic_load(Path(contract["data_root"]))
    if value_digest(data) != contract["data_digest"]:
        raise ValueError("data_digest_mismatch")
    model, _, payload = load_learner(Path(contract["parent_bundle"]))
    if payload["update"] != contract["parent_update"]:
        raise ValueError("parent_update_mismatch")

    train, fit = data["train"], data["fit"]
    with torch.no_grad():
        enc_train = torch.stack([embed(model, example) for example in train])
        enc_fit = torch.stack([embed(model, example) for example in fit])
        enc_fit_zero = torch.stack([embed(model, example, zero_rgb=True) for example in fit])
        ctx_train = torch.stack([context_vector(model, example) for example in train])
        ctx_fit = torch.stack([context_vector(model, example) for example in fit])
    y_train = [label_index(example["target"]) for example in train]
    y_fit = [label_index(example["target"]) for example in fit]

    delta = (enc_fit - enc_fit_zero).norm(dim=1)
    scale = enc_fit.norm(dim=1).clamp_min(1e-9)
    relative = delta / scale
    spread = enc_fit.std(0)
    probe_enc = ridge_probe(
        enc_train, y_train, enc_fit, y_fit, contract["criteria"]["ridge_lambda"]
    )
    probe_ctx = ridge_probe(
        ctx_train, y_train, ctx_fit, y_fit, contract["criteria"]["ridge_lambda"]
    )
    generator = torch.Generator().manual_seed(contract["criteria"]["shuffle_seed"])
    shuffled = torch.tensor(y_train)[torch.randperm(len(y_train), generator=generator)].tolist()
    probe_shuffle = ridge_probe(
        enc_train, shuffled, enc_fit, y_fit, contract["criteria"]["ridge_lambda"]
    )
    majority = max(y_fit.count(index) for index in range(len(LABELS))) / len(y_fit)
    gradients = grouped_gradients(model, train[:2])

    threshold = contract["criteria"]["encoder_invariance_threshold"]
    median_relative = float(relative.median())
    if median_relative < threshold:
        verdict = "representation_invariant"
    elif (
        probe_enc["accuracy"] >= contract["criteria"]["probe_high"]
        and probe_shuffle["accuracy"] <= contract["criteria"]["shuffle_max"]
    ):
        verdict = "information_present_credit_missing"
    elif probe_enc["accuracy"] <= contract["criteria"]["probe_low"]:
        verdict = "information_absent_from_representation"
    else:
        verdict = "inconclusive_attribution"

    report = {
        "schema": "taiji-u2-p3-11-attribution-v1",
        "date": "2026-11-14",
        "card": "22.5.P3.11",
        "phase": "attribution",
        "verdict": verdict,
        "parent_bundle": contract["parent_bundle"],
        "parent_update": payload["update"],
        "encoder_sensitivity": {
            "rgb_zero_relative_delta_median": round(median_relative, 9),
            "rgb_zero_relative_delta_max": round(float(relative.max()), 9),
            "rgb_zero_absolute_delta_max": round(float(delta.max()), 9),
            "threshold": threshold,
        },
        "embedding_spread": {
            "mean_dim_std": round(float(spread.mean()), 9),
            "max_dim_std": round(float(spread.max()), 9),
            "dims_below_1e_6": int((spread < 1e-6).sum()),
            "dimensions": int(spread.numel()),
        },
        "probes": {
            "encoding": probe_enc,
            "context": probe_ctx,
            "shuffled_control": probe_shuffle,
            "majority_baseline": round(majority, 6),
            "protocol": "ridge fit on train64, test on fit32; features standardized",
        },
        "gradient_groups": gradients,
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
        "cap_cpu": contract["cpu_cap"],
        "scope": "attribution only; one bundle, train64 and fit32, no training and no capability claim",
    }
    if not args.no_write:
        write_report(OUT, report)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "encoder": report["encoder_sensitivity"],
                "probe": {key: round(probe_enc[key], 4) for key in ("correct", "accuracy")},
                "context": round(probe_ctx["accuracy"], 4),
                "shuffle": round(probe_shuffle["accuracy"], 4),
                "measured": report["measured"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
