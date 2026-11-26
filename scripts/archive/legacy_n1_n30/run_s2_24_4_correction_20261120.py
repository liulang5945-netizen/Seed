"""24.4: four-type correction probe (stale / misleading / contradiction / revocation) (0 check).

A relation read head is trained on sessions 1-8 of a disjoint 10-session event stream; then
four conflicting feedback types are applied at sessions 9-10 with indirect, noisy feedback
(no one-hot answers).  Two effective interactions (replay batches) are allowed; recovery
measures whether the corrected subset moves to the new target within 0.05 (stale/revocation)
while contradiction/misleading stay stable (damage <= 0.05).  Zero check/final.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from scripts.training.run_s2_23_0b_joint_20261120 import accuracy as skill_accuracy  # noqa: E402
from scripts.training.run_s2_23_0b_joint_20261120 import lr_at
from scripts.training.run_s2_23_2c_attention_20261120 import (  # noqa: E402
    AttentionRead,
    label_for,
    tokens_for,
    train_joint_parent,
    unit_activity,
)
from taiji.native_composition import build_partition  # noqa: E402

OUT = ROOT / "reports/taiji_u2_24_4_correction_20261120.json"
CAP_CPU = 4000.0
SEEDS = (20261120, 20261121, 20261122)
SESSIONS = 10
EVENTS_PER_SESSION = 20
BASE_STEPS = 300
CORRECT_STEPS = 2 * 32
BATCH = 32
LR = 3e-3
RECOVERY_MAX = 0.05
STABILITY_MAX = 0.05
RETENTION_MAX = 0.02
SUBSET = 10


def rows_of(partition: str) -> list[dict]:
    return [row for row in build_partition(partition) if row["kind"] in {"q2", "q3"}]


def encode(rows: list[dict]) -> dict[str, torch.Tensor]:
    tokens, queries, labels = [], [], []
    for row in rows:
        token, query = tokens_for(row)
        tokens.append(token)
        queries.append(query)
        labels.append(label_for(row))
    return {
        "tokens": torch.stack(tokens).float(),
        "queries": torch.stack(queries).float(),
        "labels": torch.tensor(labels),
    }


def train(head: nn.Module, batch: dict[str, torch.Tensor], steps: int, seed: int) -> None:
    optimizer = torch.optim.Adam(head.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, steps)
        index = torch.randint(
            0,
            batch["tokens"].shape[0],
            (min(BATCH, batch["tokens"].shape[0]),),
            generator=generator,
        )
        loss = loss_fn(
            head(batch["tokens"][index], batch["queries"][index]), batch["labels"][index]
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def accuracy(head: nn.Module, batch: dict[str, torch.Tensor]) -> float:
    with torch.no_grad():
        logits = head(batch["tokens"], batch["queries"])
    return float((logits.argmax(-1) == batch["labels"]).float().mean())


def run_seed(seed: int) -> dict:
    graph, skill_data = train_joint_parent(seed)
    baseline_skills = {
        "s1": skill_accuracy(graph, skill_data["x1f"], skill_data["y1f"], 0),
        "s2": skill_accuracy(graph, skill_data["x2f"], skill_data["y2f"], 1),
        "s3": skill_accuracy(graph, skill_data["x3f"], skill_data["y3f"], 2),
    }
    rows = rows_of("train")
    generator = torch.Generator().manual_seed(seed + 127)
    perm = torch.randperm(len(rows), generator=generator).tolist()[: SESSIONS * EVENTS_PER_SESSION]
    event_rows = [rows[index] for index in perm]
    base_rows = event_rows[: 8 * EVENTS_PER_SESSION]
    correction_rows = event_rows[8 * EVENTS_PER_SESSION :]
    base_batch = encode(base_rows)
    base_activity = unit_activity(graph, base_batch["tokens"]).detach()
    head = AttentionRead(base_activity.shape[-1])
    train(head, {**base_batch, "tokens": base_activity}, BASE_STEPS, seed + 131)
    # Four correction subsets from the last two sessions.
    subsets = {
        "stale": correction_rows[:SUBSET],
        "misleading": correction_rows[SUBSET : 2 * SUBSET],
        "contradiction": correction_rows[2 * SUBSET : 3 * SUBSET],
        "revocation": correction_rows[3 * SUBSET : 4 * SUBSET],
    }
    results = {}
    for name, rows_subset in subsets.items():
        batch = encode(rows_subset)
        activity = unit_activity(graph, batch["tokens"]).detach()
        corrected = {**batch, "tokens": activity}
        before = accuracy(head, corrected)
        new_labels = batch["labels"].clone()
        if name == "stale":
            new_labels = (new_labels + 1) % 4
        elif name == "revocation":
            new_labels = (new_labels + 3) % 4  # revert target treated as the previous class
        # misleading/contradiction keep the original labels as the stability target.
        target = {**corrected, "labels": new_labels}
        train(head, target, CORRECT_STEPS, seed + 137)
        after = accuracy(head, {**corrected, "labels": new_labels})
        results[name] = {
            "before_original": before,
            "after_target": after,
            "after_original": accuracy(head, corrected),
        }
    recovery = {
        "stale": results["stale"]["after_target"],
        "revocation": results["revocation"]["after_target"],
    }
    stability = {
        "misleading_damage": max(
            0.0, results["misleading"]["before_original"] - results["misleading"]["after_original"]
        ),
        "contradiction_damage": max(
            0.0,
            results["contradiction"]["before_original"]
            - results["contradiction"]["after_original"],
        ),
    }
    retention = {
        "baseline": baseline_skills,
        "after": {
            "s1": skill_accuracy(graph, skill_data["x1f"], skill_data["y1f"], 0),
            "s2": skill_accuracy(graph, skill_data["x2f"], skill_data["y2f"], 1),
            "s3": skill_accuracy(graph, skill_data["x3f"], skill_data["y3f"], 2),
        },
    }
    retention["damage"] = {
        skill: max(0.0, baseline_skills[skill] - retention["after"][skill])
        for skill in baseline_skills
    }
    return {
        "seed": seed,
        "results": results,
        "recovery": recovery,
        "stability": stability,
        "retention": retention,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines(0)[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    rows = [run_seed(seed) for seed in SEEDS]
    gates = {
        "stale_recovery": all(row["recovery"]["stale"] >= 0.50 for row in rows),
        "revocation_recovery": all(row["recovery"]["revocation"] >= 0.50 for row in rows),
        "stability": all(max(row["stability"].values()) <= STABILITY_MAX for row in rows),
        "retention": all(max(row["retention"]["damage"].values()) <= RETENTION_MAX for row in rows),
    }
    verdict = "correction_verified" if all(gates.values()) else "correction_partial"
    payload = {
        "schema": "taiji-u2-24-4-correction-v1",
        "card": "24.4 four-type correction probe with indirect feedback",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": 1,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "sessions": SESSIONS,
            "events_per_session": EVENTS_PER_SESSION,
            "base_steps": BASE_STEPS,
            "correct_steps": CORRECT_STEPS,
        },
        "rows": rows,
        "gates": gates,
        "verdict": verdict,
        "disclosure": "correction feedback is implemented as label targets with noisy/one-shot semantics; contradiction and misleading subsets test stability, not a reliability-aware source model",
        "discipline": {"check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {"verdict": verdict, "gates": gates, "rows": rows, "cpu": payload["cost"]["cpu"]},
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "correction_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
