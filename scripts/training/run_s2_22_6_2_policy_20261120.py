"""22.6.2 strategy comparison on the admitted C-session pool (learnable vs fixed).

Pre-registered arms (same 32 admitted worlds, same deterministic tick streams,
3 tie-breaking seeds per arm):

* ``learnable``: retrain the frozen 22.6.1 predictor recipe (MLP 10->32->4,
  fixed seed) and act greedily by its predicted utility at every decision point;
* ``fixed_observe_t0``: observe at tick 0, answer immediately;
* ``fixed_wait_deadline``: wait to the deadline, observe, answer (strongest fixed
  baseline on this pool: 0.83);
* ``reactive_stability``: observe each tick and answer when the last two
  observations agree, else at the deadline;
* ``wait_control``: wait to the deadline and answer with the latest/unknown word.

Primary gate: mean utility of ``learnable`` minus the strongest fixed arm
``>= 0.05``, world-paired one-sided 95% lower bound ``> 0`` and all 3 seeds
positive.  Zero check/final reads; no capability signing beyond this card.
Modes: --freeze -> --authorize -> --smoke -> --run -> --closeout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.build_s2_22_6_1_labels_20261120 import (  # noqa: E402
    ACTIONS,
    FEATURE_NAMES,
    build_rows,
    one_hot,
    vectorize,
)
from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from scripts.training.curate_s2_22_6_pool_20261120 import tick_stream  # noqa: E402
from taiji.u2_history import value_digest  # noqa: E402

CONTRACT = ROOT / "plans/manifests/taiji_u2_22_6_2_policy_20261120.json"
SMOKE = ROOT / "reports/taiji_u2_22_6_2_policy_smoke_20261120.json"
RUN = ROOT / "reports/taiji_u2_22_6_2_policy_run_20261120.json"
CAP_CPU = 130.0
SEEDS = (0, 1, 2)
DEADLINE = 15
OBSERVE_COST = 0.02
WAIT_COST = 0.01
VERIFY_COST = 0.10
HIDDEN = 32
EPOCHS = 200
LR = 1e-3
TORCH_SEED = 20261120
GAIN_MIN = 0.05
ARMS = (
    "learnable",
    "fixed_observe_t0",
    "fixed_wait_deadline",
    "reactive_stability",
    "wait_control",
)
PINNED_FILES = (
    "scripts/training/run_s2_22_6_2_policy_20261120.py",
    "scripts/training/build_s2_22_6_1_labels_20261120.py",
    "scripts/training/run_s2_22_6_1_predictor_20261120.py",
    "scripts/training/curate_s2_22_6_pool_20261120.py",
)


def sha_of(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8-sig").encode()).hexdigest()


def admission_sha() -> str:
    return hashlib.sha256(
        (ROOT / "reports/taiji_u2_22_6_pool_admission_20261120.json").read_bytes()
    ).hexdigest()


def criteria() -> dict[str, Any]:
    return {
        "card": "22.6.2 strategy comparison (C sessions)",
        "pool": {
            "admission_receipt": "reports/taiji_u2_22_6_pool_admission_20261120.json",
            "admission_sha256": admission_sha(),
            "worlds": 32,
            "sessions_per_world": 3,
            "decisions": 16,
        },
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "learnable_policy": {
            "source": "22.6.1 predictor recipe retrained inside this card (identical seed)",
            "model": {
                "input": 10,
                "hidden": HIDDEN,
                "output": len(ACTIONS),
                "loss": "huber",
                "epochs": EPOCHS,
            },
            "rule": "greedy argmax over predicted action utilities",
        },
        "strongest_fixed_expected": 0.83,
        "thresholds": {
            "net_gain_min": GAIN_MIN,
            "paired_lower_bound_min": 0.0,
            "all_seeds_positive": True,
        },
        "exits_order": [
            "engineering_failure",
            "not_admitted",
            "budget_stop",
            "policy_gain_absent",
            "policy_gain_pass",
        ],
        "check_reads": 0,
        "final_reads": 0,
        "costs": {"observe": OBSERVE_COST, "wait": WAIT_COST, "verify": VERIFY_COST},
    }


def guards(contract: dict[str, Any], mode: str) -> None:
    if contract.get("schema") != "taiji-u2-22-6-2-policy-v1":
        raise ValueError("unsupported_policy_contract")
    if contract.get("criteria_sha256") != value_digest(contract["criteria"]):
        raise ValueError("criteria_digest_mismatch")
    pins = contract.get("source_pins") or {}
    for name in PINNED_FILES:
        path = ROOT / name
        if not path.is_file():
            raise ValueError(f"source_pin_missing:{name}")
        if pins.get(name) != sha_of(path):
            raise ValueError(f"source_pin_mismatch:{name}")
    if mode in {"smoke", "run"} and contract.get("status") != "frozen":
        raise ValueError("contract_not_frozen")
    if mode == "run":
        if not contract.get("training_authority"):
            raise ValueError("missing_training_authority")
        if not contract.get("cap_cpu"):
            raise ValueError("missing_cap")


def freeze(out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "taiji-u2-22-6-2-policy-v1",
        "card": "DESIGN-TAIJI_U2_22_6_2_POLICY_20261120.md",
        "status": "designed_priced_authorization_pending",
        "training_authority": None,
        "criteria": criteria(),
        "criteria_sha256": None,
        "plan": {"price_cpu": 97.5, "optimizer_steps": EPOCHS, "price_source": "runner docstring"},
        "cap_cpu": CAP_CPU,
        "check_exposure": {"reads": 0},
        "source_pins": {name: sha_of(ROOT / name) for name in PINNED_FILES},
        "result": {"run": False, "closed": False, "verdict": None},
    }
    payload["criteria_sha256"] = value_digest(payload["criteria"])
    out.write_bytes(json.dumps(payload, ensure_ascii=False, indent=2).encode() + b"\n")
    print(json.dumps({"frozen": str(out), "criteria_sha256": payload["criteria_sha256"]}))
    return 0


def authorize(contract_path: Path, authority: str) -> int:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("status") != "designed_priced_authorization_pending":
        raise ValueError("unexpected_contract_status")
    contract["status"] = "frozen"
    contract["training_authority"] = authority
    contract_path.write_bytes(json.dumps(contract, ensure_ascii=False, indent=2).encode() + b"\n")
    guards(contract, "run")
    print(json.dumps({"authorized": authority, "cap_cpu": contract["cap_cpu"]}))
    return 0


def train_predictor() -> Any:
    rows = build_rows()
    features, labels = vectorize(rows)
    train_mask = np.asarray([row["split"] == "train" for row in rows])
    mean, std = features[train_mask].mean(axis=0), features[train_mask].std(axis=0)
    std[std < 1e-6] = 1.0
    torch.manual_seed(TORCH_SEED)
    model = torch.nn.Sequential(
        torch.nn.Linear(features.shape[1], HIDDEN),
        torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, len(ACTIONS)),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.HuberLoss(delta=1.0)
    x = torch.from_numpy(((features[train_mask] - mean) / std).astype(np.float32))
    y = torch.from_numpy(labels[train_mask])
    for _ in range(EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
    model.eval()
    return model, mean, std


def predictor_scores(
    model: Any, mean: np.ndarray, std: np.ndarray, features: dict
) -> dict[str, float]:
    vector = np.asarray(
        [features[name] for name in FEATURE_NAMES] + list(features["last_word"]),
        dtype=np.float32,
    )
    with torch.no_grad():
        prediction = model(torch.from_numpy(((vector - mean) / std)[None])).numpy()[0]
    return {name: float(prediction[index]) for index, name in enumerate(ACTIONS)}


def simulate(arm: str, ticks: list[dict], target: str, seed: int, predictor: Any) -> float:
    truth = [tick["truth"].get(target, "未知") for tick in ticks]
    observed = [tick["observed"].get(target, "未知") for tick in ticks]
    truth_deadline = truth[DEADLINE]
    last_word = "未知"
    last_tick = -1
    cost = 0.0
    verify_used = 0
    history: list[str] = []
    rng = np.random.default_rng(seed)
    tick = 0
    while tick < 16:
        if arm == "fixed_observe_t0":
            if tick == 0:
                cost += OBSERVE_COST
                last_word, last_tick = observed[0], 0
            hit = 1.0 if last_word == truth_deadline else 0.0
            return hit - cost
        if arm in {"fixed_wait_deadline", "wait_control"}:
            if tick < DEADLINE:
                cost += WAIT_COST
                tick += 1
                continue
            if arm == "fixed_wait_deadline":
                cost += OBSERVE_COST
                last_word = observed[DEADLINE]
            hit = 1.0 if last_word == truth_deadline else 0.0
            return hit - cost
        if arm == "reactive_stability":
            cost += OBSERVE_COST
            history.append(observed[tick])
            if tick > 0 and history[-1] == history[-2] and history[-1] != "未知":
                last_word = history[-1]
                hit = 1.0 if last_word == truth_deadline else 0.0
                return hit - cost
            tick += 1
            continue
        # learnable: greedy argmax of predicted utilities.
        features = {
            "tick": tick,
            "decisions_left": DEADLINE - tick,
            "staleness": tick - last_tick if last_tick >= 0 else 99,
            "last_word": one_hot(last_word),
            "observations": tick,
            "verify_left": max(0.0, 2.0 - verify_used),
            "cost_so_far": round(cost, 4),
        }
        scores = predictor_scores(predictor[0], predictor[1], predictor[2], features)
        action = max(scores, key=scores.get)
        ties = [name for name, value in scores.items() if abs(value - scores[action]) < 1e-9]
        if len(ties) > 1:
            action = ties[int(rng.integers(0, len(ties)))]
        if action == "ANSWER_NOW":
            hit = 1.0 if last_word == truth_deadline else 0.0
            return hit - cost
        if action == "OBSERVE_THEN_DEADLINE":
            cost += OBSERVE_COST
            last_word, last_tick = observed[tick], tick
            tick += 1
            continue
        if action == "VERIFY_THEN_NOW":
            if verify_used >= 2:
                tick += 1
                continue
            verify_used += 1
            cost += VERIFY_COST
            last_word = truth[tick]
            hit = 1.0 if last_word == truth_deadline else 0.0
            return hit - cost
        # REST_THEN_DEADLINE: keep current information, advance one tick.
        cost += WAIT_COST
        tick += 1
    hit = 1.0 if last_word == truth_deadline else 0.0
    return hit - cost


def evaluate(predictor: Any) -> dict[str, Any]:
    admission = json.loads(
        (ROOT / "reports/taiji_u2_22_6_pool_admission_20261120.json").read_text(encoding="utf-8")
    )
    seeds = [entry["seed"] for entry in admission["admitted_worlds"]]
    results: dict[str, dict[str, list[float]]] = {arm: {} for arm in ARMS}
    for seed_value in SEEDS:
        utilities: dict[str, list[float]] = {arm: [] for arm in ARMS}
        for world in seeds:
            ticks = tick_stream(world)
            for target in list(ticks[0]["observed"]):
                for arm in ARMS:
                    utilities[arm].append(simulate(arm, ticks, target, seed_value, predictor))
        for arm in ARMS:
            results[arm][str(seed_value)] = utilities[arm]
    return {"seeds": seeds, "results": results}


def summarize(evaluation: dict[str, Any]) -> dict[str, Any]:
    results = evaluation["results"]
    means = {
        arm: {seed: round(float(np.mean(values)), 4) for seed, values in per_seed.items()}
        for arm, per_seed in results.items()
    }
    strongest_fixed_arm = max(
        (arm for arm in ARMS if arm != "learnable"),
        key=lambda arm: float(np.mean(list(means[arm].values()))),
    )
    gains = []
    for seed in means["learnable"]:
        gains.append(means["learnable"][seed] - means[strongest_fixed_arm][seed])
    paired = np.asarray(results["learnable"]["0"]) - np.asarray(results[strongest_fixed_arm]["0"])
    rng = np.random.default_rng(20261120)
    draws = rng.choice(paired, size=(10000, paired.size), replace=True).mean(axis=1)
    lower = float(np.quantile(draws, 0.05))
    return {
        "means": means,
        "strongest_fixed_arm": strongest_fixed_arm,
        "gains": [round(gain, 4) for gain in gains],
        "mean_gain": round(float(np.mean(gains)), 4),
        "paired_lower_bound": round(lower, 4),
    }


def smoke(contract: dict[str, Any]) -> int:
    started = time.process_time()
    guards(contract, "smoke")
    admission = json.loads(
        (ROOT / "reports/taiji_u2_22_6_pool_admission_20261120.json").read_text(encoding="utf-8")
    )
    checks = {
        "admission_matches": hashlib.sha256(
            (ROOT / "reports/taiji_u2_22_6_pool_admission_20261120.json").read_bytes()
        ).hexdigest()
        == contract["criteria"]["pool"]["admission_sha256"],
        "pool_size": len(admission["admitted_worlds"]) == 32,
        "pinned_files_present": all((ROOT / name).is_file() for name in PINNED_FILES),
        "zero_optimizer_steps": True,
    }
    passed = all(checks.values())
    payload = {
        "schema": "taiji-u2-22-6-2-policy-smoke-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "checks": checks,
        "verdict": "smoke_passed" if passed else "smoke_failed",
        "discipline": {"optimizer_steps": 0, "check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(SMOKE, payload)
    print(json.dumps({"verdict": payload["verdict"], "cpu": payload["cost"]["cpu"]}))
    return 0 if passed else 1


def run(contract: dict[str, Any]) -> int:
    started = time.process_time()
    guards(contract, "run")
    predictor = train_predictor()
    evaluation = evaluate(predictor)
    summary = summarize(evaluation)
    budget_ok = time.process_time() - started <= CAP_CPU
    passed = (
        summary["mean_gain"] >= GAIN_MIN
        and summary["paired_lower_bound"] > 0
        and all(gain > 0 for gain in summary["gains"])
    )
    verdict = (
        "not_admitted"
        if len(evaluation["seeds"]) != 32
        else (
            "budget_stop"
            if not budget_ok
            else "policy_gain_pass" if passed else "policy_gain_absent"
        )
    )
    payload = {
        "schema": "taiji-u2-22-6-2-policy-run-v1",
        "criteria_sha256": contract["criteria_sha256"],
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "summary": summary,
        "verdict": verdict,
        "budget": {"cap_cpu": CAP_CPU, "cpu": round(time.process_time() - started, 2)},
        "discipline": {"check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(RUN, payload)
    print(
        json.dumps(
            {"verdict": verdict, "summary": summary, "cpu": payload["cost"]["cpu"]},
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "policy_gain_pass" else 1


def closeout(contract_path: Path) -> int:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    report = json.loads(RUN.read_text(encoding="utf-8"))
    contract["result"] = {
        "run": True,
        "closed": True,
        "verdict": report["verdict"],
        "cost_cpu": report["cost"]["cpu"],
        "report": str(RUN.relative_to(ROOT)),
    }
    contract_path.write_bytes(json.dumps(contract, ensure_ascii=False, indent=2).encode() + b"\n")
    print(json.dumps({"closed": report["verdict"]}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true")
    mode.add_argument("--authorize", metavar="AUTHORITY")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--closeout", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        return freeze(CONTRACT)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if args.authorize:
        return authorize(CONTRACT, args.authorize)
    if args.smoke:
        return smoke(contract)
    if args.closeout:
        return closeout(CONTRACT)
    return run(contract)


if __name__ == "__main__":
    raise SystemExit(main())
