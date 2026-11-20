"""24.2: consolidation-replay comparison on the frozen 23.0b parent (0 check).

Simulated 10-session event stream (20 events/session) from the frozen relation fixture.
Arms differ only in what the slow read head replays: priority (recency-weighted), random,
recent-only, and frozen-slow (no update); fast-only uses the bounded store for recall
without updating the head.  Two delayed windows (sessions 9-10 proxy) measure whether the
gain persists; the W_new+0 probe compares with/without the last two sessions replayed.
Zero check/final.
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

OUT = ROOT / "reports/taiji_u2_24_2_consolidation_20261120.json"
CAP_CPU = 6000.0
SEEDS = (20261120, 20261121, 20261122)
SESSIONS = 10
EVENTS_PER_SESSION = 20
STEPS = 300
BATCH = 32
LR = 3e-3
RECENT_SESSIONS = 2
GAIN_MIN = 0.05
RETENTION_MAX = 0.02
WINDOWS = 2


def relation_rows(partition: str) -> list[dict]:
    return [row for row in build_partition(partition) if row["kind"] in {"q2", "q3"}]


def encode_rows(rows: list[dict]) -> dict[str, torch.Tensor]:
    tokens = []
    queries = []
    labels = []
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


def train_head(
    head: nn.Module,
    events: dict[str, torch.Tensor],
    arm: str,
    seed: int,
    steps: int = STEPS,
) -> None:
    optimizer = torch.optim.Adam(head.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed + 101)
    total = events["tokens"].shape[0]
    sessions = total // EVENTS_PER_SESSION
    weights = torch.ones(total)
    if arm == "priority":
        for index in range(total):
            session = index // EVENTS_PER_SESSION
            weights[index] = 1.0 + float(session) / max(1, sessions - 1) * 2.0
    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, steps)
        if arm == "random":
            index = torch.randint(0, total, (BATCH,), generator=generator)
        elif arm == "recent":
            start = max(0, total - RECENT_SESSIONS * EVENTS_PER_SESSION)
            index = torch.randint(start, total, (BATCH,), generator=generator)
        elif arm == "priority":
            index = torch.multinomial(weights, BATCH, replacement=True, generator=generator)
        else:
            raise ValueError(arm)
        loss = loss_fn(
            head(events["tokens"][index], events["queries"][index]), events["labels"][index]
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def evaluate(head: nn.Module | None, store: dict, batch: dict, fast_only: bool) -> float:
    if fast_only:
        keys = batch["tokens"].reshape(batch["tokens"].shape[0], -1)
        predictions = []
        for index in range(keys.shape[0]):
            distance = torch.cdist(keys[index].unsqueeze(0), store["keys"]).squeeze(0)
            slot = int(distance.argmin())
            predictions.append(store["labels"][slot])
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
    rows = relation_rows("train")
    generator = torch.Generator().manual_seed(seed + 103)
    order = torch.randint(
        0, len(rows), (SESSIONS * EVENTS_PER_SESSION,), generator=generator
    ).tolist()
    event_rows = [rows[index] for index in order]
    events = encode_rows(event_rows)
    # Bounded store for the fast-only arm and the W_new+0 key storage.
    keys = torch.nn.functional.normalize(events["tokens"].reshape(-1, 24), dim=1)
    store = {"keys": keys, "labels": events["labels"]}
    activity = unit_activity(graph, events["tokens"]).detach()
    held = relation_rows("fit")
    split = len(held) // 2
    windows = [encode_rows(held[:split]), encode_rows(held[split:])]

    arms: dict[str, dict] = {"fast_only": {}}
    for name in ("priority", "random", "recent", "frozen_slow"):
        arms[name] = {}
    for window_index, window in enumerate(windows):
        arms["fast_only"][window_index] = evaluate(None, store, window, fast_only=True)
    for name in ("priority", "random", "recent"):
        head = AttentionRead(activity.shape[-1])
        train_head(
            head,
            {"tokens": activity, "queries": events["queries"], "labels": events["labels"]},
            name,
            seed,
        )
        for window_index, window in enumerate(windows):
            window_activity = unit_activity(graph, window["tokens"]).detach()
            arms[name][window_index] = evaluate(
                head,
                store,
                {
                    "tokens": window_activity,
                    "queries": window["queries"],
                    "labels": window["labels"],
                },
                fast_only=False,
            )
    frozen = AttentionRead(activity.shape[-1])
    for window_index, window in enumerate(windows):
        window_activity = unit_activity(graph, window["tokens"]).detach()
        arms["frozen_slow"][window_index] = evaluate(
            frozen,
            store,
            {"tokens": window_activity, "queries": window["queries"], "labels": window["labels"]},
            fast_only=False,
        )
    # W_new+0 probe: train without the last RECENT_SESSIONS events and compare.
    cutoff = (SESSIONS - RECENT_SESSIONS) * EVENTS_PER_SESSION
    head_no_recent = AttentionRead(activity.shape[-1])
    train_head(
        head_no_recent,
        {
            "tokens": activity[:cutoff],
            "queries": events["queries"][:cutoff],
            "labels": events["labels"][:cutoff],
        },
        "priority",
        seed + 7,
    )
    no_recent = []
    for window in windows:
        window_activity = unit_activity(graph, window["tokens"]).detach()
        no_recent.append(
            evaluate(
                head_no_recent,
                store,
                {
                    "tokens": window_activity,
                    "queries": window["queries"],
                    "labels": window["labels"],
                },
                fast_only=False,
            )
        )
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
        "arms": arms,
        "no_recent": no_recent,
        "w_new_plus_0_gap": [arms["priority"][i] - no_recent[i] for i in range(len(windows))],
        "retention": retention,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines(0)[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    rows = [run_seed(seed) for seed in SEEDS]
    phases = {}
    for name in ("priority", "random", "recent", "frozen_slow", "fast_only"):
        phases[name] = [[row["arms"][name][window] for window in range(WINDOWS)] for row in rows]
    priority = [sum(row["arms"]["priority"]) / WINDOWS for row in rows]
    fast = [sum(row["arms"]["fast_only"]) / WINDOWS for row in rows]
    gains = [priority[i] - fast[i] for i in range(len(rows))]
    mean_gain = sum(gains) / len(gains)
    persistent = all(
        row["arms"]["priority"][w] > row["arms"]["fast_only"][w]
        for row in rows
        for w in range(WINDOWS)
    )
    gates = {
        "consolidation_gain": mean_gain >= GAIN_MIN and all(gain > 0 for gain in gains),
        "persistent": persistent,
        "retention": all(max(row["retention"]["damage"].values()) <= RETENTION_MAX for row in rows),
    }
    verdict = "consolidation_verified" if all(gates.values()) else "consolidation_partial"
    payload = {
        "schema": "taiji-u2-24-2-consolidation-v1",
        "card": "24.2 consolidation replay comparison (priority/random/recent/frozen/fast-only)",
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
            "steps": STEPS,
            "windows": WINDOWS,
        },
        "rows": rows,
        "mean_gain": mean_gain,
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
                "mean_gain": mean_gain,
                "arms": phases,
                "components": [(row["arms"]["priority"], row["arms"]["fast_only"]) for row in rows],
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "consolidation_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
