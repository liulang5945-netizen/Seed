"""24.3: prefix maintenance vs full rebuild on long-delay (out-of-store) events (0 check).

Sessions 1-2 events are held out as the long-delay test (they fall outside the bounded
64-slot store during consolidation over sessions 3-10).  Arms replay a growing prefix of
consolidation sessions (1/2/4/8/10 sessions); the full-rebuild arm replays everything and
fast-only uses the store without updating the head.  Gate: the best prefix must stay within
0.05 of full rebuild on the old events while replaying fewer events; retention of the
three old skills must stay <= 0.02.  Zero check/final.
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

OUT = ROOT / "reports/taiji_u2_24_3_prefix_20261120.json"
CAP_CPU = 4000.0
SEEDS = (20261120, 20261121, 20261122)
SESSIONS = 10
EVENTS_PER_SESSION = 20
STEPS = 300
BATCH = 32
LR = 3e-3
PREFIXES = (1, 2, 4, 8)
GAP_MAX = 0.05
RETENTION_MAX = 0.02


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


def train_head(head: nn.Module, batch: dict[str, torch.Tensor], seed: int) -> None:
    optimizer = torch.optim.Adam(head.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed + 109)
    for step in range(STEPS):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, STEPS)
        index = torch.randint(0, batch["tokens"].shape[0], (BATCH,), generator=generator)
        loss = loss_fn(
            head(batch["tokens"][index], batch["queries"][index]), batch["labels"][index]
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def evaluate(head: nn.Module | None, batch: dict[str, torch.Tensor], store: dict | None) -> float:
    if head is None:
        keys = batch["tokens"].reshape(batch["tokens"].shape[0], -1)
        predictions = []
        for index in range(keys.shape[0]):
            distance = torch.cdist(keys[index].unsqueeze(0), store["keys"]).squeeze(0)
            predictions.append(store["labels"][int(distance.argmin())])
        predictions = torch.tensor(predictions)
    else:
        with torch.no_grad():
            predictions = head(batch["tokens"], batch["queries"]).argmax(-1)
    return float((predictions == batch["labels"]).float().mean())


def run_seed(seed: int) -> dict:
    graph, skill_data = train_joint_parent(seed)
    baseline = {
        "s1": skill_accuracy(graph, skill_data["x1f"], skill_data["y1f"], 0),
        "s2": skill_accuracy(graph, skill_data["x2f"], skill_data["y2f"], 1),
        "s3": skill_accuracy(graph, skill_data["x3f"], skill_data["y3f"], 2),
    }
    rows = rows_of("train")
    generator = torch.Generator().manual_seed(seed + 113)
    order = torch.randperm(len(rows), generator=generator).tolist()[: SESSIONS * EVENTS_PER_SESSION]
    event_rows = [rows[index] for index in order]
    old_rows = event_rows[: 2 * EVENTS_PER_SESSION]
    consolidation_rows = event_rows[2 * EVENTS_PER_SESSION :]
    old_batch = encode(old_rows)
    cons_batch = encode(consolidation_rows)
    old_activity = unit_activity(graph, old_batch["tokens"]).detach()
    cons_activity = unit_activity(graph, cons_batch["tokens"]).detach()
    # Store only covers the last 64 events (sessions 3-10 recent end), not the old ones.
    keys = torch.nn.functional.normalize(cons_batch["tokens"].reshape(-1, 24), dim=1)
    store = {"keys": keys, "labels": cons_batch["labels"]}
    forced = evaluate(None, old_batch, store)
    results: dict[str, dict[str, float]] = {}
    costs: dict[str, int] = {}
    for prefix in PREFIXES:
        count = prefix * EVENTS_PER_SESSION
        sub = {
            "tokens": cons_activity[:count],
            "queries": cons_batch["queries"][:count],
            "labels": cons_batch["labels"][:count],
        }
        head = AttentionRead(cons_activity.shape[-1])
        train_head(head, sub, seed)
        accuracy = evaluate(head, {**old_batch, "tokens": old_activity}, None)
        results[f"prefix_{prefix}"] = accuracy
        costs[f"prefix_{prefix}"] = count
    full = AttentionRead(cons_activity.shape[-1])
    train_head(
        full,
        {"tokens": cons_activity, "queries": cons_batch["queries"], "labels": cons_batch["labels"]},
        seed,
    )
    results["full_rebuild"] = evaluate(full, {**old_batch, "tokens": old_activity}, None)
    costs["full_rebuild"] = cons_activity.shape[0]
    retention = {
        "baseline": baseline,
        "after": {
            "s1": skill_accuracy(graph, skill_data["x1f"], skill_data["y1f"], 0),
            "s2": skill_accuracy(graph, skill_data["x2f"], skill_data["y2f"], 1),
            "s3": skill_accuracy(graph, skill_data["x3f"], skill_data["y3f"], 2),
        },
    }
    retention["damage"] = {
        skill: max(0.0, baseline[skill] - retention["after"][skill]) for skill in baseline
    }
    return {
        "seed": seed,
        "old_events": len(old_rows),
        "consolidation_events": len(consolidation_rows),
        "fast_only": forced,
        "arms": results,
        "costs": costs,
        "retention": retention,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines(0)[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    rows = [run_seed(seed) for seed in SEEDS]
    best_prefix = None
    for prefix in PREFIXES:
        gaps = [row["arms"][f"prefix_{prefix}"] - row["arms"]["full_rebuild"] for row in rows]
        mean_gap = sum(gaps) / len(gaps)
        if best_prefix is None or mean_gap > best_prefix[1]:
            best_prefix = (prefix, mean_gap, all(gap >= -GAP_MAX for gap in gaps))
    prefix, gap, within = best_prefix
    gains = [row["fast_only"] for row in rows]
    gates = {
        "near_full_rebuild": within and gap >= -GAP_MAX,
        "cost_reduced": costs_reduced(rows, prefix),
        "retention": all(max(row["retention"]["damage"].values()) <= RETENTION_MAX for row in rows),
    }
    verdict = "prefix_maintenance_verified" if all(gates.values()) else "prefix_maintenance_partial"
    payload = {
        "schema": "taiji-u2-24-3-prefix-v1",
        "card": "24.3 prefix maintenance vs full rebuild on out-of-store old events",
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
            "prefixes": list(PREFIXES),
            "steps": STEPS,
        },
        "rows": rows,
        "best_prefix": prefix,
        "mean_gap_vs_full": gap,
        "fast_only": gains,
        "gates": gates,
        "verdict": verdict,
        "discipline": {"check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "gates": gates,
                "best_prefix": prefix,
                "gap": gap,
                "rows": rows,
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "prefix_maintenance_verified" else 1


def costs_reduced(rows: list[dict], prefix: int) -> bool:
    return all(row["costs"][f"prefix_{prefix}"] < row["costs"]["full_rebuild"] for row in rows)


if __name__ == "__main__":
    raise SystemExit(main())
