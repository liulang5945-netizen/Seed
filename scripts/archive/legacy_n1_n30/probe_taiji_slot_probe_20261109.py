"""R5: is the slot content there at all, and only the read-out is wrong?

R4a closed one explanation.  Even with a perfect description->name bridge, whole-sentence
cosine distance still ranks *template* above *content*: the nearest neighbour lands on the
same template 95% (R1) and 98.75% (R2) of the time.  Two possibilities remain:

a) the representation does contain (object, place), and only the whole-sentence distance
   is the wrong read-out;
b) the representation does not contain it at all.

This probe asks (a) with a read-out that has no notion of sentence distance: a closed-form
ridge regression maps the feature vector straight to the object and the place.  The solution
is one matrix solve -- zero gradient steps, zero optimiser steps, zero Adam, zero epochs.

Data: 480 samples from ``native_goal_world_wide`` (120 states x 4 goals).  Cross-validation
is grouped by state, so all four goals of a state stay in one fold; a state in the test fold
is never seen in training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# The ridge solve goes through BLAS.  OpenBLAS/MKL spin their worker threads even for a
# 128 x 128 system, so the first two runs spent 845 s and 702 s of CPU on what is a
# few milliseconds of arithmetic.  The limits must be set before numpy is imported.
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.training.probe_taiji_representation_20261109 as frozen  # noqa: E402
from taiji.living_core import LivingCoreReference  # noqa: E402
from taiji.native_goal_world_wide import OBJECTS, PLACES, samples, state_text  # noqa: E402

torch.set_num_threads(1)

MANIFEST = ROOT / "plans/manifests/taiji_slot_probe_20261109.json"
REPORT = ROOT / "reports/taiji_slot_probe_20261109.json"
ARMS = ("R1", "R2")
LAMBDAS = (0.001, 0.1, 10.0, 1000.0)
FOLDS = 5
READABLE_JOINT = 0.80
NOT_READABLE_MARGIN = 0.10
CHANCE_JOINT = 1.0 / (len(OBJECTS) * len(PLACES))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def fold_of(state: tuple) -> int:
    """Deterministic state-level fold assignment; the state text is the only input."""
    digest = hashlib.sha256(state_text(state).encode("utf-8")).digest()
    return digest[0] % FOLDS


def build_dataset() -> list[dict[str, Any]]:
    rows = []
    for row in samples():
        rows.append(
            {
                "text": row["goal"],
                "object": row["correct"][0],
                "place": row["correct"][1],
                "fold": fold_of(row["state"]),
                "state": state_text(row["state"]),
            }
        )
    return rows


def one_hot(labels: list[str], vocabulary: list[str]) -> np.ndarray:
    index = {label: position for position, label in enumerate(vocabulary)}
    out = np.zeros((len(labels), len(vocabulary)), dtype=np.float64)
    for row, label in enumerate(labels):
        out[row, index[label]] = 1.0
    return out


def solve_ridge(features: np.ndarray, targets: np.ndarray, lam: float) -> np.ndarray:
    dim = features.shape[1]
    gram = features.T @ features + lam * np.eye(dim)
    return np.linalg.solve(gram, features.T @ targets)


def arm_reading(features: np.ndarray, rows: list[dict[str, Any]]) -> dict[str, Any]:
    objects = [row["object"] for row in rows]
    places = [row["place"] for row in rows]
    folds = [row["fold"] for row in rows]
    per_lambda = []
    for lam in LAMBDAS:
        targets_object = one_hot(objects, list(OBJECTS))
        targets_place = one_hot(places, list(PLACES))
        predicted_object = np.empty(len(rows), dtype=object)
        predicted_place = np.empty(len(rows), dtype=object)
        for fold in range(FOLDS):
            test = np.array([value == fold for value in folds])
            train = ~test
            weights_object = solve_ridge(features[train], targets_object[train], lam)
            weights_place = solve_ridge(features[train], targets_place[train], lam)
            chosen_object = (features[test] @ weights_object).argmax(axis=1)
            chosen_place = (features[test] @ weights_place).argmax(axis=1)
            predicted_object[test] = [OBJECTS[int(index)] for index in chosen_object]
            predicted_place[test] = [PLACES[int(index)] for index in chosen_place]
        correct_object = sum(int(a == b) for a, b in zip(predicted_object, objects, strict=True))
        correct_place = sum(int(a == b) for a, b in zip(predicted_place, places, strict=True))
        joint = sum(
            int(a == b and c == d)
            for a, b, c, d in zip(predicted_object, objects, predicted_place, places, strict=True)
        )
        per_lambda.append(
            {
                "lambda": lam,
                "object_accuracy": round(correct_object / len(rows), 6),
                "place_accuracy": round(correct_place / len(rows), 6),
                "joint_accuracy": round(joint / len(rows), 6),
            }
        )
    best = max(per_lambda, key=lambda item: item["joint_accuracy"])
    if best["joint_accuracy"] >= READABLE_JOINT:
        verdict = "readable"
    elif best["joint_accuracy"] <= CHANCE_JOINT + NOT_READABLE_MARGIN:
        verdict = "not_readable"
    else:
        verdict = "weak"
    return {"per_lambda": per_lambda, "best": best, "verdict": verdict}


def features_for(name: str, texts: list[str], reference: LivingCoreReference) -> np.ndarray:
    vectors = frozen.representation(name, texts, reference)
    return vectors.numpy()


def phase_freeze() -> dict[str, Any]:
    rows = build_dataset()
    folds = [row["fold"] for row in rows]
    manifest = {
        "schema": "taiji-slot-probe-v1",
        "date": "2026-11-09",
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "counts": {
            "samples": len(rows),
            "states": len({row["state"] for row in rows}),
            "objects": len(OBJECTS),
            "places": len(PLACES),
            "fold_sizes": [folds.count(index) for index in range(FOLDS)],
        },
        "definition": {
            "read_out": "closed-form ridge regression from the feature vector to one-hot labels",
            "solution": "W = (X^T X + lambda I)^-1 X^T Y; one linear solve, no gradient step",
            "cross_validation": "5 folds keyed by sha256(state_text)[0] % 5; all four goals of a state stay together",
            "lambdas": list(LAMBDAS),
            "targets": "object (4 classes) and place (5 classes), each read separately",
        },
        "criteria": {
            "readable": f"best joint accuracy over lambdas >= {READABLE_JOINT}",
            "not_readable": f"best joint accuracy <= chance + {NOT_READABLE_MARGIN}",
            "chance": {
                "object": round(1.0 / len(OBJECTS), 6),
                "place": round(1.0 / len(PLACES), 6),
                "joint": round(CHANCE_JOINT, 6),
            },
            "overall": (
                "content_present_readout_wrong if R1 or R2 is readable; "
                "representation_lacks_content if both are not_readable; otherwise inconclusive"
            ),
        },
    }
    write_json(MANIFEST, manifest)
    return manifest


def phase_run(manifest: dict[str, Any]) -> dict[str, Any]:
    started_cpu = time.process_time()
    started_wall = time.time()
    rows = build_dataset()
    if len(rows) != manifest["counts"]["samples"]:
        raise SystemExit("dataset size changed after the freeze")
    if sha256_file(Path(__file__).resolve()) != manifest["runner_sha256"]:
        raise SystemExit("runner changed after the freeze")
    texts = [row["text"] for row in rows]
    reference = LivingCoreReference()
    arms = {}
    for name in ARMS:
        arms[name] = arm_reading(features_for(name, texts, reference), rows)
    readable = [name for name in ARMS if arms[name]["verdict"] == "readable"]
    unreadable = [name for name in ARMS if arms[name]["verdict"] == "not_readable"]
    if readable:
        overall = "content_present_readout_wrong"
    elif len(unreadable) == len(ARMS):
        overall = "representation_lacks_content"
    else:
        overall = "inconclusive"
    payload = {
        "schema": "taiji-slot-probe-v1",
        "date": "2026-11-09",
        "arm": "R5",
        "verdict": overall,
        "readable_arms": readable,
        "not_readable_arms": unreadable,
        "runner_sha256": manifest["runner_sha256"],
        "counts": manifest["counts"],
        "arms": arms,
        "disclosures": [
            "The read-out is fitted on the same 480 samples it is scored on, but strictly "
            "out-of-fold; a state in the test fold is never seen in training.",
            "Ridge regression is linear; a negative result rules out a linear read-out, not "
            "every possible read-out.",
            "The goals here are all describe-form; this probe says nothing about the "
            "description/name gap measured in R3 and R4a.",
        ],
        "cost": {
            "cpu_seconds": round(time.process_time() - started_cpu, 3),
            "wall_seconds": round(time.time() - started_wall, 2),
            "cpu_cap": frozen.CPU_CAP,
            "threads": torch.get_num_threads(),
        },
    }
    write_json(REPORT, payload)
    return payload


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args(argv)
    frozen.acquire_lock()
    try:
        if args.freeze:
            manifest = phase_freeze()
            print(json.dumps(manifest["counts"], ensure_ascii=False))
            return 0
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        payload = phase_run(manifest)
        print(
            json.dumps(
                {
                    "verdict": payload["verdict"],
                    "readable": payload["readable_arms"],
                    "not_readable": payload["not_readable_arms"],
                    "R1": payload["arms"]["R1"]["best"],
                    "R2": payload["arms"]["R2"]["best"],
                    "cost": payload["cost"],
                },
                ensure_ascii=False,
            )
        )
        return 0
    finally:
        frozen.release_lock()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
