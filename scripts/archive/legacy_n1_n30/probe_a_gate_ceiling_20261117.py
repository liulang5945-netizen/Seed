"""A-gate readout ceiling: is 0.80 reachable by a converged head at all?

The coverage audit (``probe_a_gate_coverage_20261117``) reproduced D-43 exactly
(``train12_fit3`` median 0.526) and then failed its own admissibility rule: on
three-body training worlds two of three seeds never fitted the training set
(``train 0.8008/0.3021/0.2552``) inside 2000 full-batch steps.  A head that has
not fitted train is not a ceiling, so no coverage verdict may be read from it.

This probe therefore measures the ceiling on train-to-convergence instead of a
fixed step count, holding everything else from D-43 fixed (``CapacityHead``
width 64, two blocks, Adam lr 0.01, weight decay 1e-4, ``raw`` 16x48 patches
with 2-d coordinates, three head seeds).  Training stops early once train
accuracy reaches 0.995 or at the declared step cap.

Arms:

* ``train12_fit12``  -- 1-2 body training, held-out 1-2 body fit;
* ``train12_fit3``   -- the D-43 arm, held-out three-body fit;
* ``train3072_fit3`` -- 3072 1-2 body training worlds, three-body fit.

Declared exits:

1. ``ceiling_not_convergent``: any arm stays below 0.99 train accuracy at the
   step cap.  Then no fit reading from that arm is a bound, and the ceiling
   procedure itself is the open problem.
2. ``data_scale_binding``: ``train3072_fit3`` median fit >= 0.80.
3. ``within_distribution_only``: ``train12_fit12`` median fit >= 0.80 but
   ``train3072_fit3`` median fit < 0.80.
4. ``fixture_not_learnable_at_scale``: both medians < 0.80.

No U2 optimizer step, no check read and no final read occur here.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as functional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.first_a_support_22_5 import generate_examples  # noqa: E402
from scripts.training.probe_a_gate_coverage_20261117 import (  # noqa: E402
    FIT12_PARTITION,
    _bodies,
    _flat,
    _split,
)
from scripts.training.probe_a_query_bind_head_20261117 import (  # noqa: E402
    HEAD_SEEDS,
    CapacityHead,
    _confusion,
    _extract,
)

REPORT = ROOT / "reports/taiji_u2_22_5_a_gate_ceiling_20261117.json"
CACHE = ROOT / "output/a_gate_ceiling_20261117/features.pt"
STEP_CAP = 6000
TARGET_TRAIN = 0.995
EVAL_EVERY = 250
COUNT = 768
FIT_COUNT = 192
SCALED = 3072


def _scaled_train(count: int) -> list[dict]:
    """Widen the 1-2 body training split under the same body-count rule."""
    return _split("train", count, _bodies)


def _features() -> dict:
    if CACHE.exists():
        return torch.load(CACHE, weights_only=False)
    rows = {
        "train12": _extract(
            None, [_flat(row) for row in generate_examples("train", COUNT)], raw_only=True
        ),
        "fit3": _extract(
            None, [_flat(row) for row in generate_examples("fit", FIT_COUNT)], raw_only=True
        ),
        "fit12": _extract(None, _split(FIT12_PARTITION, FIT_COUNT, _bodies), raw_only=True),
        "train3072": _extract(None, _scaled_train(SCALED), raw_only=True),
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rows, CACHE)
    return rows


def _fit_to_convergence(train: dict, fit: dict, seed: int) -> dict:
    started = time.process_time()
    torch.manual_seed(seed)
    head = CapacityHead(train["raw"].shape[2])
    optimizer = torch.optim.Adam(head.parameters(), lr=0.01, weight_decay=1e-4)
    steps = 0
    train_accuracy = 0.0
    while steps < STEP_CAP:
        optimizer.zero_grad(set_to_none=True)
        logits = head(train["raw"], train["request"], train["mask"])
        loss = functional.cross_entropy(logits, train["label"])
        loss.backward()
        optimizer.step()
        steps += 1
        if steps % EVAL_EVERY == 0:
            with torch.no_grad():
                train_accuracy = (
                    (
                        head(train["raw"], train["request"], train["mask"]).argmax(-1)
                        == train["label"]
                    )
                    .float()
                    .mean()
                    .item()
                )
            if train_accuracy >= TARGET_TRAIN:
                break
    head.eval()
    with torch.no_grad():
        train_logits = head(train["raw"], train["request"], train["mask"])
        logits = head(fit["raw"], fit["request"], fit["mask"])
    train_accuracy = (train_logits.argmax(-1) == train["label"]).float().mean().item()
    return {
        "seed": seed,
        "steps": steps,
        "converged": train_accuracy >= 0.99,
        "train_accuracy": train_accuracy,
        "fit_accuracy": (logits.argmax(-1) == fit["label"]).float().mean().item(),
        "train_confusion": _confusion(train_logits.argmax(-1), train["label"]),
        "fit_confusion": _confusion(logits.argmax(-1), fit["label"]),
        "cpu": round(time.process_time() - started, 4),
    }


def _median(rows: list[dict], key: str) -> float:
    return sorted(row[key] for row in rows)[len(rows) // 2]


def _verdict(arms: dict) -> str:
    if not all(row["converged"] for arm in arms.values() for row in arm["results"]):
        return "ceiling_not_convergent"
    if arms["train3072_fit3"]["median_fit_accuracy"] >= 0.80:
        return "data_scale_binding"
    if arms["train12_fit12"]["median_fit_accuracy"] >= 0.80:
        return "within_distribution_only"
    return "fixture_not_learnable_at_scale"


def main() -> int:
    if REPORT.exists():
        raise FileExistsError(REPORT)
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    rows = _features()
    arms = {}
    for name, train_key, fit_key in (
        ("train12_fit12", "train12", "fit12"),
        ("train12_fit3", "train12", "fit3"),
        ("train3072_fit3", "train3072", "fit3"),
    ):
        results = [_fit_to_convergence(rows[train_key], rows[fit_key], seed) for seed in HEAD_SEEDS]
        arms[name] = {
            "train_split": train_key,
            "fit_split": fit_key,
            "train_examples": int(rows[train_key]["label"].numel()),
            "fit_examples": int(rows[fit_key]["label"].numel()),
            "results": results,
            "median_train_accuracy": _median(results, "train_accuracy"),
            "median_fit_accuracy": _median(results, "fit_accuracy"),
        }
    verdict = _verdict(arms)
    payload = {
        "schema": "taiji-u2-22-5-a-gate-ceiling-v1",
        "card": "22.5.0.A",
        "verdict": verdict,
        "why": (
            "the D-43 capacity comparison fixed 2000 full-batch steps; on three-body "
            "worlds two of three seeds never fitted train, so its fit numbers were "
            "not a ceiling.  This probe stops on train accuracy instead of a step count."
        ),
        "conditions": {
            "head": "CapacityHead width 64, two blocks",
            "adam": {"lr": 0.01, "weight_decay": 0.0001},
            "step_cap": STEP_CAP,
            "target_train_accuracy": TARGET_TRAIN,
            "evaluate_every": EVAL_EVERY,
            "seeds": list(HEAD_SEEDS),
            "source": "raw 16x48 rgb patches plus 2-d patch coordinates",
        },
        "arms": arms,
        "medians": {name: arm["median_fit_accuracy"] for name, arm in arms.items()},
        "declared_verdict": {
            "ceiling_not_convergent": "any arm below 0.99 train accuracy at the step cap",
            "data_scale_binding": "train3072_fit3 median fit >= 0.80",
            "within_distribution_only": "train12_fit12 median fit >= 0.80 only",
            "fixture_not_learnable_at_scale": "both medians < 0.80",
        },
        "discipline": {
            "u2_optimizer_steps": 0,
            "head_optimizer_steps": sum(
                row["steps"] for arm in arms.values() for row in arm["results"]
            ),
            "check_reads": 0,
            "final_reads": 0,
            "project_data_reads": 0,
        },
        "cost": {
            "cpu": round(time.process_time() - started_cpu, 4),
            "wall": round(time.perf_counter() - started_wall, 4),
        },
        "source_pins": {
            "scripts/training/probe_a_gate_ceiling_20261117.py": hashlib.sha256(
                Path(__file__).read_text(encoding="utf-8-sig").encode()
            ).hexdigest(),
            "scripts/training/probe_a_gate_coverage_20261117.py": hashlib.sha256(
                (ROOT / "scripts/training/probe_a_gate_coverage_20261117.py")
                .read_text(encoding="utf-8-sig")
                .encode()
            ).hexdigest(),
            "scripts/training/probe_a_query_bind_head_20261117.py": hashlib.sha256(
                (ROOT / "scripts/training/probe_a_query_bind_head_20261117.py")
                .read_text(encoding="utf-8-sig")
                .encode()
            ).hexdigest(),
            "scripts/training/first_a_support_22_5.py": hashlib.sha256(
                (ROOT / "scripts/training/first_a_support_22_5.py")
                .read_text(encoding="utf-8-sig")
                .encode()
            ).hexdigest(),
        },
    }
    REPORT.write_bytes(json.dumps(payload, ensure_ascii=False, indent=2).encode() + b"\n")
    print(
        "ceiling verdict="
        + verdict
        + " steps="
        + json.dumps({name: [row["steps"] for row in arm["results"]] for name, arm in arms.items()})
        + " medians="
        + json.dumps({name: round(value, 4) for name, value in payload["medians"].items()})
        + " cost="
        + str(payload["cost"]["cpu"])
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
