"""A-gate fixture coverage audit: is the body-count shift the wall?

D-20261117-43 measured a wide frozen readout on the A-gate fixture and got
``train 1.000/1.000/0.913`` against ``fit 0.526/0.385/0.646``.  The declared
verdict was ``capacity_feasible_generalization_gap``, with one confound left
open: ``first_a_support_22_5.generate_examples`` draws train worlds with 1-2
bodies and every fit world with 3 bodies.  "Memorised the training worlds" and
"never saw a three-body scene" cannot be separated by that reading alone.

This audit holds the head (``CapacityHead``: width 64, two blocks, 2000
full-batch steps, three seeds), the token source (``raw`` = 16x48 rgb patches
with their 2-d coordinates) and the request encoder fixed, and varies only the
training distribution:

* ``train12_fit3``   -- the D-43 arm, reproduced exactly as a self-check;
* ``train12_fit12``  -- within-distribution held-out (1-2 bodies);
* ``train3_fit3``    -- coverage arm (three-body training);
* ``train3_fit12``   -- reverse transfer;
* ``mixed_fit3``     -- 1-2 and 3 body training pooled, fit on three bodies.

Declared exits, evaluated in this order on medians over the three head seeds:

1. ``within_distribution_gap``: ``train12_fit12`` median < 0.80.  The head
   cannot generalise inside its own body-count band, so coverage is not the
   first problem.
2. ``coverage_binding_constraint``: ``train3_fit3`` median >= 0.80.  Three-body
   scenes are learnable at the readout ceiling, so the A-gate gap is data
   coverage, not representational headroom.
3. ``partial_coverage_gain``: 0.60 <= ``train3_fit3`` median < 0.80.
4. ``coverage_insufficient_to_close_gap``: ``train3_fit3`` median < 0.60.

No U2 optimizer step, no check read and no final read occur here.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.first_a_support_22_5 import (  # noqa: E402
    LABELS,
    RGB_COLORS,
    generate_examples,
    seed_for,
)
from scripts.training.probe_a_query_bind_head_20261117 import (  # noqa: E402
    CAPACITY_STEPS,
    HEAD_SEEDS,
    _confusion,
    _extract,
    _fit_capacity,
)
from taiji.u2_task_episode import NAMES, locate_public, position_word  # noqa: E402
from taiji.u2_task_world import PhysicalWorld  # noqa: E402

REPORT = ROOT / "reports/taiji_u2_22_5_a_gate_coverage_20261117.json"
CAPACITY_REPORT = ROOT / "reports/taiji_u2_22_5_a_query_bind_capacity_20261117.json"
COUNTS = {"train": 768, "fit": 192}
FIT12_PARTITION = "coverage-fit12"


def _bodies(index: int) -> int:
    """The frozen fixture's own body-count rule for non-fit splits."""
    return 1 + ((index // 4) % 2)


def _split(partition: str, count: int, objects_of) -> list[dict]:
    """Mirror ``generate_examples`` with an explicit body-count schedule.

    The label, eligibility and oracle logic is copied verbatim so that the only
    difference from the frozen fixture is ``body_count``.  ``seed_for`` is
    shared, so a world index keeps the same seed derivation.
    """
    examples = []
    for index in range(count):
        label = LABELS[index % 4]
        for attempt in range(256):
            seed = seed_for(partition, index, attempt)
            objects = objects_of(index)
            world = PhysicalWorld.reset(seed, {"body_count": objects})
            result = world.elapse("public-scene")
            bodies = world.snapshot()["bodies"]
            names = [NAMES[RGB_COLORS.index(tuple(body["color"]))] for body in bodies]
            if label == "未知":
                absent = [name for name in NAMES if name not in names]
                entity = absent[seed % len(absent)]
            else:
                eligible = [
                    i
                    for i, body in enumerate(bodies)
                    if position_word(body["position"][0]) == label
                    and min(abs(body["position"][0] + 1 / 3), abs(body["position"][0] - 1 / 3))
                    >= 0.1
                ]
                if not eligible:
                    continue
                entity = names[eligible[seed % len(eligible)]]
            estimated = locate_public(result["raw_media"]["rgb"], entity)
            oracle = "未知" if estimated is None else position_word(estimated)
            if oracle != label:
                continue
            examples.append(
                {
                    "id": f"{partition}-{index}",
                    "request": f"{entity}现在在哪里？".encode(),
                    "observations": result["observations"],
                    "coordinates": result["coordinates"],
                    "outcome": result["execution_outcome"],
                    "label": LABELS.index(label),
                    "audit": {"seed": seed, "attempt": attempt, "objects": objects},
                }
            )
            break
        else:
            raise ValueError("coverage_fixture_generation_exhausted")
    return examples


def _flat(example: dict) -> dict:
    """Project a frozen-fixture example onto the probe's flat feature shape."""
    public = example["public"]
    return {
        "id": example["id"],
        "request": public["request"],
        "observations": public["observations"],
        "coordinates": public["coordinates"],
        "outcome": public["outcome"],
        "label": LABELS.index(example["target"].decode()),
        "audit": example["audit"],
    }


def _concat(first: dict, second: dict) -> dict:
    return {key: torch.cat([first[key], second[key]], 0) for key in first}


def _fit_arm(train: dict, fit: dict, seed: int) -> dict:
    head, train_accuracy, train_logits = _fit_capacity(train, seed)
    with torch.no_grad():
        logits = head(fit["raw"], fit["request"], fit["mask"])
    return {
        "seed": seed,
        "train_accuracy": train_accuracy,
        "fit_accuracy": (logits.argmax(-1) == fit["label"]).float().mean().item(),
        "train_confusion": _confusion(train_logits.argmax(-1), train["label"]),
        "fit_confusion": _confusion(logits.argmax(-1), fit["label"]),
    }


def _median(rows: list[dict], key: str) -> float:
    return sorted(row[key] for row in rows)[len(rows) // 2]


def _verdict(medians: dict[str, float]) -> str:
    if medians["train12_fit12"] < 0.80:
        return "within_distribution_gap"
    if medians["train3_fit3"] >= 0.80:
        return "coverage_binding_constraint"
    if medians["train3_fit3"] >= 0.60:
        return "partial_coverage_gain"
    return "coverage_insufficient_to_close_gap"


def main() -> int:
    if REPORT.exists():
        raise FileExistsError(REPORT)
    frozen = json.loads(CAPACITY_REPORT.read_text(encoding="utf-8"))
    frozen_rows = {row["seed"]: row for row in frozen["capacity"]["results"]}

    train12_examples = [_flat(row) for row in generate_examples("train", COUNTS["train"])]
    fit3_examples = [_flat(row) for row in generate_examples("fit", COUNTS["fit"])]
    train3_examples = _split("train", COUNTS["train"], lambda index: 3)
    fit12_examples = _split(FIT12_PARTITION, COUNTS["fit"], _bodies)

    train12 = _extract(None, train12_examples, raw_only=True)
    train3 = _extract(None, train3_examples, raw_only=True)
    fit3 = _extract(None, fit3_examples, raw_only=True)
    fit12 = _extract(None, fit12_examples, raw_only=True)
    mixed = _concat(train12, train3)

    splits = {
        "train12": train12,
        "train3": train3,
        "mixed": mixed,
        "fit3": fit3,
        "fit12": fit12,
    }
    arms = (
        ("train12_fit3", "train12", "fit3"),
        ("train12_fit12", "train12", "fit12"),
        ("train3_fit3", "train3", "fit3"),
        ("train3_fit12", "train3", "fit12"),
        ("mixed_fit3", "mixed", "fit3"),
    )
    results = {}
    for name, train_key, fit_key in arms:
        rows = [_fit_arm(splits[train_key], splits[fit_key], seed) for seed in HEAD_SEEDS]
        results[name] = {
            "train_split": train_key,
            "fit_split": fit_key,
            "train_examples": int(splits[train_key]["label"].numel()),
            "fit_examples": int(splits[fit_key]["label"].numel()),
            "results": rows,
            "median_train_accuracy": _median(rows, "train_accuracy"),
            "median_fit_accuracy": _median(rows, "fit_accuracy"),
        }

    reproduced = all(
        abs(row["train_accuracy"] - frozen_rows[row["seed"]]["train_accuracy"]) < 1e-12
        and abs(row["fit_accuracy"] - frozen_rows[row["seed"]]["fit_accuracy"]) < 1e-12
        for row in results["train12_fit3"]["results"]
    )
    medians = {name: results[name]["median_fit_accuracy"] for name in results}
    verdict = _verdict(medians)
    payload = {
        "schema": "taiji-u2-22-5-a-gate-coverage-v1",
        "card": "22.5.0.A",
        "verdict": verdict,
        "reproduction_of_d43": reproduced,
        "frozen_d43": {
            "median_train_accuracy": frozen["capacity"]["median_train_accuracy"],
            "median_fit_accuracy": frozen["capacity"]["median_fit_accuracy"],
            "verdict": frozen["capacity"]["verdict"],
        },
        "fixture": {
            "train12_bodies": "1-2 (frozen rule)",
            "train3_bodies": 3,
            "fit3_bodies": 3,
            "fit12_bodies": "1-2",
            "fit12_partition": FIT12_PARTITION,
            "note": (
                "the frozen A-gate fixture draws train with 1-2 bodies and fit with 3; "
                "fit12 is a new held-out namespace so both body-count bands have a "
                "held-out split"
            ),
        },
        "arms": results,
        "medians": medians,
        "declared_verdict": {
            "within_distribution_gap": "train12_fit12 median < 0.80",
            "coverage_binding_constraint": "train3_fit3 median >= 0.80",
            "partial_coverage_gain": "0.60 <= train3_fit3 median < 0.80",
            "coverage_insufficient_to_close_gap": "train3_fit3 median < 0.60",
        },
        "conditions": {
            "head": "CapacityHead width 64, two blocks",
            "steps": CAPACITY_STEPS,
            "seeds": list(HEAD_SEEDS),
            "source": "raw 16x48 rgb patches plus 2-d patch coordinates",
        },
        "discipline": {
            "u2_optimizer_steps": 0,
            "head_optimizer_steps": CAPACITY_STEPS * len(HEAD_SEEDS) * len(arms),
            "check_reads": 0,
            "final_reads": 0,
        },
        "source_pins": {
            "scripts/training/probe_a_gate_coverage_20261117.py": hashlib.sha256(
                Path(__file__).read_text(encoding="utf-8-sig").encode()
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
        "coverage verdict="
        + verdict
        + " reproduced="
        + str(reproduced)
        + " medians="
        + json.dumps({name: round(value, 4) for name, value in medians.items()})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
