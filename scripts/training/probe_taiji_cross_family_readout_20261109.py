"""R6 -- does the *read-out* fix the cross-wording problem?

R3 and R4a showed that on the 80-sentence fixture every distance-based read-out
fails across lexical families: the same goal written as a description and as a
paraphrase sits farther apart than unrelated goals.  R5 then showed that the
slot (object, place) is *linearly readable* from R1/R2 -- but only inside the
``describe`` family, where the sentence literally names the slot.

This round asks the decisive follow-up with zero gradient steps:

    take the same 80 sentences -- 20 goals, each written four ways
    (``describe`` + ``needs`` / ``hope`` / ``state``).  Fit a closed-form ridge
    read-out for object (4 classes), place (5 classes) and joint (20 classes)
    and test it on goals never seen in training (5 folds grouped by goal, so the
    four wordings of a goal always fall in the same fold).  If the joint
    accuracy stays high, the earlier failures were a read-out defect, not a
    missing-content defect.  If it collapses to chance, the content is only
    readable inside one lexical family.

Arms (identical protocol):

* R1 -- 128-bin byte histogram, L2-normalised (existing content key).
* R2 -- mean of the untrained project encoder's context vectors, L2-normalised.
* R3 -- frozen ``sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2``.
  If the weights are not in the local cache the arm is reported ``unavailable``
  and the verdict falls back to R1/R2; no download is attempted here because the
  R3 weights were already fetched under D-20261109-05.

Read-out: ``W = (X^T X + lambda I)^-1 X^T Y`` on one-hot targets, lambda in
{0.001, 0.1, 10, 1000}; prediction is the arg-max.  5 folds grouped by goal
(16 goals train / 4 goals test).  Chance: object 0.25, place 0.20, joint 0.05.

Frozen verdict, per arm: ``readable`` iff joint >= 0.75 and object >= 0.85 and
place >= 0.80; ``not_readable`` iff joint <= chance + 0.10; ``inconclusive``
otherwise.  Overall: ``cross_family_readout_transfers`` if any arm is readable;
``cross_family_readout_fails`` if every available arm is not_readable.

Discipline: 0 training, 0 optimizer steps, 0 Adam, 0 confirm, 0 new parameters;
``process_guard`` first; single-instance lock; CPU cap 200 s; BLAS/torch pinned
to one thread; readings written then renamed; dual runner sha.  No
``python -m black``.
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

# Must precede numpy: OpenBLAS/MKL spin their worker pool even for a 128 x 128 solve.
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.training.probe_taiji_representation_20261109 as frozen  # noqa: E402
from taiji.living_core import LivingCoreReference  # noqa: E402

torch.set_num_threads(1)

MANIFEST = ROOT / "plans/manifests/taiji_cross_family_readout_20261109.json"
REPORT = ROOT / "reports/taiji_cross_family_readout_20261109.json"

ARMS = ("R1", "R2", "R3")
MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LAMBDAS = (0.001, 0.1, 10.0, 1000.0)
FOLDS = 5
READABLE_JOINT = 0.75
READABLE_OBJECT = 0.85
READABLE_PLACE = 0.80
NOT_READABLE_MARGIN = 0.10
CHANCE_OBJECT = 1.0 / 4.0
CHANCE_PLACE = 1.0 / 5.0
CHANCE_JOINT = CHANCE_OBJECT * CHANCE_PLACE
CPU_CAP = 200.0

RUNNER_SHA_START = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()

DEFINITION = {
    "fixture": (
        "frozen.build_fixture(): 20 goals = 4 objects x 5 places, each written four ways "
        "(describe + needs / hope / state) = 80 sentences.  No sentence is reused."
    ),
    "labels": "object (4 classes), place (5 classes), joint (20 classes), one-hot targets.",
    "readout": (
        "closed form ridge W = (X^T X + lambda I)^-1 X^T Y; prediction is the arg-max over the "
        "held-out rows.  One matrix solve per (arm, lambda, fold, target); no gradient step."
    ),
    "folds": (
        "5 folds grouped by goal: the goals are ordered by sha256(goal), then dealt modulo 5, so "
        "each fold holds exactly 4 goals = 16 sentences and all four wordings of a goal stay "
        "together.  A held-out goal is never seen in training."
    ),
    "arms": {
        "R1": "LivingCoreReference.content_feature (128-bin byte histogram, L2-normalised).",
        "R2": (
            "mean of reference.encoder.context(raw) over positions (128 dims), L2-normalised; "
            "weights are the random initialisation at seed 20261031, i.e. untrained."
        ),
        "R3": (
            "instruments.document_embedding.DocumentEmbedder on the frozen sentence-transformer; "
            "upper bound only, reported unavailable if the local cache is missing."
        ),
    },
    "chance": {"object": 0.25, "place": 0.20, "joint": 0.05},
}

CRITERIA = {
    "per_arm": (
        "readable iff joint >= 0.75 and object >= 0.85 and place >= 0.80; not_readable iff "
        "joint <= chance + 0.10; otherwise inconclusive.  Thresholds are fixed here, before the "
        "first reading, and are not revisited afterwards."
    ),
    "overall": (
        "cross_family_readout_transfers if any arm is readable; cross_family_readout_fails if "
        "every available arm is not_readable; otherwise inconclusive."
    ),
    "lambda": "the reported reading is the best joint accuracy over the four lambdas.",
    "no_fit": "0 training, 0 optimizer steps, 0 Adam, 0 confirm, 0 new parameters.",
}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def fixture_hashes() -> dict[str, str]:
    return {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in frozen.FIXTURE_FILES
    }


def fold_map(goals: list[str]) -> dict[str, int]:
    order = sorted(set(goals), key=lambda goal: hashlib.sha256(goal.encode()).hexdigest())
    return {goal: index % FOLDS for index, goal in enumerate(order)}


def one_hot(labels: list[str], classes: list[str]) -> np.ndarray:
    index = {name: position for position, name in enumerate(classes)}
    target = np.zeros((len(labels), len(classes)), dtype=np.float64)
    for row, label in enumerate(labels):
        target[row, index[label]] = 1.0
    return target


def solve_ridge(features: np.ndarray, targets: np.ndarray, lam: float) -> np.ndarray:
    gram = features.T @ features
    return np.linalg.solve(gram + lam * np.eye(gram.shape[0]), features.T @ targets)


def l2_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def arm_reading(matrix: np.ndarray, rows: list[dict[str, Any]]) -> dict[str, Any]:
    goals = [row["goal"] for row in rows]
    assignment = fold_map(goals)
    folds = np.array([assignment[goal] for goal in goals])
    labels_by_target = {
        "object": [row["object"] for row in rows],
        "place": [row["place"] for row in rows],
        "joint": [f"{row['object']}|{row['place']}" for row in rows],
    }
    classes = {name: sorted(set(values)) for name, values in labels_by_target.items()}

    readings = []
    for lam in LAMBDAS:
        correct = dict.fromkeys(labels_by_target, 0)
        for fold in range(FOLDS):
            train, test = folds != fold, folds == fold
            for name, labels in labels_by_target.items():
                targets = one_hot(
                    [label for label, flag in zip(labels, train, strict=True) if flag],
                    classes[name],
                )
                weights = solve_ridge(matrix[train], targets, lam)
                predicted = (matrix[test] @ weights).argmax(axis=1)
                truth = [label for label, flag in zip(labels, test, strict=True) if flag]
                correct[name] += sum(
                    classes[name][column] == label
                    for column, label in zip(predicted, truth, strict=True)
                )
        readings.append(
            {
                "lambda": lam,
                "object_accuracy": round(correct["object"] / len(rows), 6),
                "place_accuracy": round(correct["place"] / len(rows), 6),
                "joint_accuracy": round(correct["joint"] / len(rows), 6),
            }
        )

    best = max(readings, key=lambda row: (row["joint_accuracy"], -row["lambda"]))
    if (
        best["joint_accuracy"] >= READABLE_JOINT
        and best["object_accuracy"] >= READABLE_OBJECT
        and best["place_accuracy"] >= READABLE_PLACE
    ):
        verdict = "readable"
    elif best["joint_accuracy"] <= CHANCE_JOINT + NOT_READABLE_MARGIN:
        verdict = "not_readable"
    else:
        verdict = "inconclusive"
    return {
        "per_lambda": readings,
        "best": best,
        "verdict": verdict,
        "fold_sizes": [int((folds == fold).sum()) for fold in range(FOLDS)],
    }


def phase_freeze() -> dict[str, Any]:
    rows = frozen.build_fixture()
    goals = [row["goal"] for row in rows]
    manifest = {
        "schema": "taiji-cross-family-readout-v1",
        "date": "2026-11-09",
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runner_sha256": RUNNER_SHA_START,
        "fixture_sha256": fixture_hashes(),
        "fixture_files": list(frozen.FIXTURE_FILES),
        "question": (
            "with the same 80 sentences and a closed-form linear read-out instead of a distance, "
            "can the slot be decoded across the four wordings of a goal?"
        ),
        "counts": {
            "sentences": len(rows),
            "goals": len(set(goals)),
            "variants_per_goal": sorted(
                {sum(1 for row in rows if row["goal"] == goal) for goal in set(goals)}
            ),
            "objects": len({row["object"] for row in rows}),
            "places": len({row["place"] for row in rows}),
            "folds": FOLDS,
            "fold_assignment": fold_map(goals),
        },
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "lambdas": list(LAMBDAS),
    }
    write_json(MANIFEST, manifest)
    print(json.dumps({"manifest": str(MANIFEST), "counts": manifest["counts"]}, ensure_ascii=False))
    return manifest


def phase_run(manifest: dict[str, Any]) -> dict[str, Any]:
    started_cpu = time.process_time()
    started_wall = time.time()
    if fixture_hashes() != manifest["fixture_sha256"]:
        raise SystemExit("fixture changed after the freeze; re-freeze before running")
    if manifest["runner_sha256"] != RUNNER_SHA_START:
        raise SystemExit("runner changed after the freeze; re-freeze before running")

    rows = frozen.build_fixture()
    texts = [row["text"] for row in rows]
    reference = LivingCoreReference()
    arms: dict[str, Any] = {}
    model: dict[str, Any] = {"model_id": MODEL_ID, "status": "not_requested"}
    for arm in ARMS:
        if arm == "R3":
            try:
                from instruments.document_embedding import DocumentEmbedder

                embedder = DocumentEmbedder(model_id=MODEL_ID)
                model = {
                    "model_id": MODEL_ID,
                    "status": "available",
                    "payload": embedder.to_payload(),
                }
                matrix = embedder.embed(texts).to(torch.float64).numpy()
                del embedder
            except Exception as exc:  # noqa: BLE001 - an unavailable arm must not kill the run
                arms["R3"] = {"verdict": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
                continue
            matrix = l2_normalise(matrix)
        else:
            matrix = l2_normalise(frozen.representation(arm, texts, reference).numpy())

        reading = arm_reading(matrix, rows)
        reading["dimensions"] = int(matrix.shape[1])
        arms[arm] = reading

    verdicts = [arm["verdict"] for arm in arms.values() if arm["verdict"] != "unavailable"]
    if "readable" in verdicts:
        overall = "cross_family_readout_transfers"
    elif verdicts and all(item == "not_readable" for item in verdicts):
        overall = "cross_family_readout_fails"
    else:
        overall = "inconclusive"

    report = {
        "schema": "taiji-cross-family-readout-v1",
        "date": "2026-11-09",
        "runner_sha256": RUNNER_SHA_START,
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "fixture_sha256": manifest["fixture_sha256"],
        "counts": manifest["counts"],
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "model": model,
        "arms": arms,
        "verdict": overall,
        "cost": {
            "cpu_seconds": round(time.process_time() - started_cpu, 3),
            "wall_seconds": round(time.time() - started_wall, 2),
            "cpu_cap": CPU_CAP,
            "threads": torch.get_num_threads(),
        },
    }
    write_json(REPORT, report)
    print(
        json.dumps(
            {"report": str(REPORT), "verdict": overall, "cost": report["cost"]},
            ensure_ascii=False,
        )
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    arguments = parser.parse_args()
    from taiji.process_guard import assert_no_black_processes

    assert_no_black_processes()
    frozen.acquire_lock()
    try:
        if arguments.freeze:
            phase_freeze()
            return 0
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        phase_run(manifest)
        return 0
    finally:
        frozen.release_lock()


if __name__ == "__main__":
    raise SystemExit(main())
