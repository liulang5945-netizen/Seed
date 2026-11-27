"""Pre-registered label dataset for the 22.6.1 gain/risk/cost predictor (zero training).

For every decision point of the 32 admitted development worlds (3 sessions each,
16 decisions), record the observable context and the realized utility of four
candidate actions under a FROZEN continuation rule:

* ANSWER_NOW: answer with the latest observed word, locking the current info;
* OBSERVE_THEN_DEADLINE: observe at the current tick, wait to the deadline,
  observe again at the deadline, then answer;
* VERIFY_THEN_NOW: spend one VERIFY credit for the true current word and answer;
* REST_THEN_DEADLINE: do not observe again, answer at the deadline with the
  latest available word.

Continuation is fixed and pre-registered, so labels are realized utilities, not
model preferences.  ``build_rows()`` is deterministic and is imported by the
22.6.1 contract runner.  Zero optimizer steps, zero check/final reads.
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

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from scripts.training.curate_s2_22_6_pool_20261120 import tick_stream  # noqa: E402

OUT = ROOT / "reports/taiji_u2_22_6_1_labels_20261120.json"
ADMISSION = ROOT / "reports/taiji_u2_22_6_pool_admission_20261120.json"
DEADLINE = 15
OBSERVE_COST = 0.02
WAIT_COST = 0.01
VERIFY_COST = 0.10
ACTIONS = ("ANSWER_NOW", "OBSERVE_THEN_DEADLINE", "VERIFY_THEN_NOW", "REST_THEN_DEADLINE")
WORDS = ("左边", "中间", "右边", "未知")
FEATURE_NAMES = (
    "tick",
    "decisions_left",
    "staleness",
    "observations",
    "verify_left",
    "cost_so_far",
)


def one_hot(word: str) -> list[float]:
    return [1.0 if word == item else 0.0 for item in WORDS]


def build_rows() -> list[dict]:
    admission = json.loads(ADMISSION.read_text(encoding="utf-8"))
    seeds = [entry["seed"] for entry in admission["admitted_worlds"]]
    train_worlds = set(seeds[:24])
    rows: list[dict] = []
    for seed in seeds:
        ticks = tick_stream(seed)
        targets = list(ticks[0]["observed"])
        for target_index, target in enumerate(targets):
            truth_by_tick = [tick["truth"].get(target, "未知") for tick in ticks]
            observed_by_tick = [tick["observed"].get(target, "未知") for tick in ticks]
            last_obs_word = "未知"
            last_obs_tick = -1
            cost = 0.0
            for tick in range(16):
                staleness = tick - last_obs_tick if last_obs_tick >= 0 else 99
                features = {
                    "tick": tick,
                    "decisions_left": DEADLINE - tick,
                    "staleness": staleness,
                    "last_word": one_hot(last_obs_word),
                    "observations": tick,
                    "verify_left": max(0.0, 2.0 - float(tick // 8)),
                    "cost_so_far": round(cost, 4),
                }
                truth_now = truth_by_tick[tick]
                truth_deadline = truth_by_tick[DEADLINE]
                answer_now = 1.0 if last_obs_word == truth_deadline else 0.0
                future_obs_cost = OBSERVE_COST
                future_wait_cost = (DEADLINE - tick) * WAIT_COST
                observe_word = (
                    observed_by_tick[DEADLINE] if tick < DEADLINE else observed_by_tick[tick]
                )
                observe_hit = 1.0 if observe_word == truth_deadline else 0.0
                verify_hit = 1.0 if truth_now == truth_deadline else 0.0
                rest_hit = 1.0 if last_obs_word == truth_deadline else 0.0
                labels = {
                    "ANSWER_NOW": round(answer_now - cost, 4),
                    "OBSERVE_THEN_DEADLINE": round(
                        observe_hit - cost - future_obs_cost - future_wait_cost, 4
                    ),
                    "VERIFY_THEN_NOW": round(verify_hit - cost - VERIFY_COST, 4),
                    "REST_THEN_DEADLINE": round(rest_hit - cost - future_wait_cost, 4),
                }
                rows.append(
                    {
                        "world": seed,
                        "target": target,
                        "session": target_index,
                        "tick": tick,
                        "features": features,
                        "labels": labels,
                        "split": "train" if seed in train_worlds else "heldout",
                    }
                )
                cost += OBSERVE_COST
                last_obs_word = observed_by_tick[tick]
                last_obs_tick = tick
    return rows


def vectorize(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(
        [
            [row["features"][name] for name in FEATURE_NAMES] + list(row["features"]["last_word"])
            for row in rows
        ],
        dtype=np.float32,
    )
    labels = np.asarray(
        [[row["labels"][name] for name in ACTIONS] for row in rows], dtype=np.float32
    )
    return features, labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    rows = build_rows()
    features, labels = vectorize(rows)
    payload = {
        "schema": "taiji-u2-22-6-1-labels-v1",
        "card": "22.6.1 predictor label dataset (zero training)",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "pool_source": str(ADMISSION.relative_to(ROOT)),
            "worlds": len({row["world"] for row in rows}),
            "train_worlds": len({row["world"] for row in rows if row["split"] == "train"}),
            "heldout_worlds": len({row["world"] for row in rows if row["split"] == "heldout"}),
            "sessions_per_world": 3,
            "decisions": 16,
            "actions": list(ACTIONS),
            "continuation": "frozen per-action continuation (see module docstring)",
            "costs": {"observe": OBSERVE_COST, "wait": WAIT_COST, "verify": VERIFY_COST},
            "default_behavior_policy": "OBSERVE at every tick (labels are counterfactual)",
            "features": list(FEATURE_NAMES),
            "label_means": {
                name: round(float(labels[:, index].mean()), 4) for index, name in enumerate(ACTIONS)
            },
            "label_std": {
                name: round(float(labels[:, index].std()), 4) for index, name in enumerate(ACTIONS)
            },
        },
        "n_rows": len(rows),
        "feature_dim": int(features.shape[1]),
        "label_dim": int(labels.shape[1]),
        "rows_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
        "discipline": {"optimizer_steps": 0, "check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    dataset_path = ROOT / "output/s2_22_6_1/labels.jsonl"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    payload["dataset"] = {
        "path": str(dataset_path.relative_to(ROOT)),
        "sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {
                "n_rows": len(rows),
                "label_means": payload["protocol"]["label_means"],
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
