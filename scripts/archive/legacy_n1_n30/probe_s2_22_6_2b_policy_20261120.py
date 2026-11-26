"""22.6.2b contextual cost-sensitive policy over the C-task pool (zero check).

The 22.6.2 negative came from a greedy argmax over an offline utility predictor.
Here the 7 fixed timing/verify policies are treated as actions in a contextual
cost-sensitive problem on the 32 admitted pool worlds: context = t0 observations
of all bodies, targets = each policy's utility (fully observed in simulation),
model = MLP + bootstrap ensemble, decision = argmax(mean - beta * std) with beta
chosen by 5-fold CV on the training worlds.  Held-out worlds compare the learned
policy with the best fixed policy and the greedy (beta=0) baseline.

Gates: policy mean utility >= best fixed + 0.05; 3 seeds same direction; paired
one-sided 95% lower bound > 0; zero check, cap 150 CPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from scripts.training.curate_s2_22_6_pool_20261120 import (  # noqa: E402
    policy_utilities,
    tick_stream,
)

OUT = ROOT / "reports/taiji_u2_22_6_2b_policy_20261120.json"
POOL = ROOT / "reports/taiji_u2_22_6_pool_admission_20261120.json"
CAP_CPU = 150.0
SEEDS = (0, 1, 2)
BETA_GRID = (0.0, 0.05, 0.10, 0.20)
BOOTSTRAP = 5
HIDDEN = 32
EPOCHS = 300
LR = 1e-2
HELDOUT_WORLDS = 8
ACTIONS = (
    "answer_no_observe",
    "rest_then_answer",
    "observe_t0_answer",
    "observe_t7_answer",
    "observe_t14_answer",
    "observe_deadline_answer",
    "verify_t0_answer",
)
POSITIONS = ("左边", "中间", "右边", "未知")


def build_cases(seeds: list[int]) -> list[dict]:
    cases = []
    for seed in seeds:
        ticks = tick_stream(seed)
        present = list(ticks[0]["observed"])
        for name in present:
            truth = ticks[-1]["truth"][name]
            observed = [tick["observed"][name] for tick in ticks]
            utility = policy_utilities({"truth": truth, "observed": observed})
            context = []
            for other in present:
                value = ticks[0]["observed"][other]
                onehot = [0.0] * 4
                onehot[POSITIONS.index(value)] = 1.0
                context.extend(onehot)
            which = [0.0] * len(present)
            which[present.index(name)] = 1.0
            context.extend(which)
            cases.append(
                {
                    "seed": seed,
                    "entity": name,
                    "features": torch.tensor(context, dtype=torch.float32),
                    "utilities": torch.tensor([utility[action] for action in ACTIONS]),
                }
            )
    return cases


def train_ensemble(cases: list[dict], seed: int) -> list[torch.nn.Module]:
    x = torch.stack([case["features"] for case in cases])
    y = torch.stack([case["utilities"] for case in cases])
    models = []
    for member in range(BOOTSTRAP):
        generator = torch.Generator().manual_seed(seed * 100 + member)
        order = torch.randint(0, x.shape[0], (x.shape[0],), generator=generator)
        network = torch.nn.Sequential(
            torch.nn.Linear(x.shape[1], HIDDEN),
            torch.nn.Tanh(),
            torch.nn.Linear(HIDDEN, len(ACTIONS)),
        )
        optimizer = torch.optim.Adam(network.parameters(), lr=LR)
        for _ in range(EPOCHS):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(network(x[order]), y[order])
            loss.backward()
            optimizer.step()
        models.append(network)
    return models


def predict(
    models: list[torch.nn.Module], features: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        stacked = torch.stack([model(features) for model in models])
    return stacked.mean(dim=0), stacked.std(dim=0)


def choose(mean: torch.Tensor, std: torch.Tensor, beta: float) -> int:
    return int((mean - beta * std).argmax().item())


def evaluate(cases: list[dict], models: list[torch.nn.Module], beta: float) -> list[float]:
    values = []
    for case in cases:
        mean, std = predict(models, case["features"][None, :])
        index = choose(mean[0], std[0], beta)
        values.append(float(case["utilities"][index]))
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    pool = json.loads(POOL.read_text(encoding="utf-8"))
    seeds = [entry["seed"] for entry in pool["admitted_worlds"]]
    cases = build_cases(seeds)
    train_seeds = seeds[:-HELDOUT_WORLDS]
    heldout_seeds = seeds[-HELDOUT_WORLDS:]
    train_cases = [case for case in cases if case["seed"] in train_seeds]
    heldout_cases = [case for case in cases if case["seed"] in heldout_seeds]
    fixed_means = {}
    for index, action in enumerate(ACTIONS):
        fixed_means[action] = float(
            np.mean([float(case["utilities"][index]) for case in heldout_cases])
        )
    best_fixed = max(fixed_means.items(), key=lambda item: item[1])
    results = []
    for seed in SEEDS:
        models = train_ensemble(train_cases, seed)
        # 5-fold CV over training worlds for beta selection
        folds = np.array_split(np.arange(len(train_cases)), 5)
        best_beta, best_cv = BETA_GRID[0], -1e9
        for beta in BETA_GRID:
            values = []
            for fold in folds:
                fold_cases = [train_cases[index] for index in fold]
                rest = [
                    case
                    for index, case in enumerate(train_cases)
                    if index not in set(fold.tolist())
                ]
                fold_models = train_ensemble(rest, seed + 7)
                values.extend(evaluate(fold_cases, fold_models, beta))
            mean_value = float(np.mean(values))
            if mean_value > best_cv:
                best_cv, best_beta = mean_value, beta
        learned = evaluate(heldout_cases, models, best_beta)
        greedy = evaluate(heldout_cases, models, 0.0)
        differences = np.asarray(learned) - float(best_fixed[1])
        rng = np.random.default_rng(20261120)
        draws = rng.choice(differences, size=(5000, differences.size), replace=True).mean(axis=1)
        results.append(
            {
                "seed": seed,
                "beta": best_beta,
                "learned_mean": round(float(np.mean(learned)), 4),
                "greedy_mean": round(float(np.mean(greedy)), 4),
                "paired_lower_bound": round(float(np.quantile(draws, 0.05)), 4),
            }
        )
    means = [result["learned_mean"] for result in results]
    gain = float(np.mean(means) - best_fixed[1])
    gates = {
        "gain_0_05": gain >= 0.05,
        "all_seeds_positive": all(result["learned_mean"] > best_fixed[1] for result in results),
        "paired_lower_bound": all(result["paired_lower_bound"] > 0 for result in results),
        "greedy_comparison_reported": True,
    }
    verdict = "policy_gain_pass" if all(gates.values()) else "policy_gain_absent"
    payload = {
        "schema": "taiji-u2-22-6-2b-policy-v1",
        "card": "22.6.2b contextual cost-sensitive policy over the C-task pool",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "pool_worlds": len(seeds),
            "train_worlds": len(train_seeds),
            "heldout_worlds": len(heldout_seeds),
            "actions": list(ACTIONS),
            "beta_grid": list(BETA_GRID),
            "bootstrap_members": BOOTSTRAP,
            "epochs": EPOCHS,
            "seeds": list(SEEDS),
        },
        "fixed_means_heldout": {name: round(value, 4) for name, value in fixed_means.items()},
        "best_fixed": {"name": best_fixed[0], "mean": round(best_fixed[1], 4)},
        "results": results,
        "mean_learned": round(float(np.mean(means)), 4),
        "gain_over_best_fixed": round(gain, 4),
        "gates": gates,
        "verdict": verdict,
        "discipline": {
            "optimizer_steps": EPOCHS * BOOTSTRAP * 6 * len(SEEDS),
            "check_reads": 0,
            "final_reads": 0,
        },
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "gain": payload["gain_over_best_fixed"],
                "best_fixed": payload["best_fixed"],
                "learned": results,
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "policy_gain_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
