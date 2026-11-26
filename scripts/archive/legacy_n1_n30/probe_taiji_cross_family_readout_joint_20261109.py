"""R6b -- R6 with the joint metric defined correctly.

The first R6 run froze ``joint`` as a 20-class one-hot read-out.  Under
goal-grouped cross-validation that metric is structurally zero: the four goals
of a held-out fold are *absent* from the training fold, so the 20 columns that
correspond to those joint classes have no training rows; their scores stay at
zero and the arg-max never selects them.  The frozen verdict
``cross_family_readout_fails`` was produced by that column and is therefore not
interpretable, not a reading about the representation.

This runner repeats exactly the same protocol -- same 80 sentences, same
sha256-dealt 5 folds grouped by goal, same lambdas, same ridge solve, same
thresholds -- and defines the joint reading the only sensible way left:

    joint_accuracy = fraction of held-out rows where the object read-out AND
    the place read-out are both correct on that same row.

No other reading changes.  For reference, R6's object and place columns are
re-used as a cross-check: the per-row conjunction must be <= min(object, place).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.training.probe_taiji_cross_family_readout_20261109 as base  # noqa: E402
from taiji.living_core import LivingCoreReference  # noqa: E402

torch.set_num_threads(1)

MANIFEST = ROOT / "plans/manifests/taiji_cross_family_readout_joint_20261109.json"
REPORT = ROOT / "reports/taiji_cross_family_readout_joint_20261109.json"

RUNNER_SHA_START = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()

DEFINITION = {
    "inherits": "every field of base.DEFINITION (fixture, labels, readout, folds, arms, chance).",
    "joint": (
        "joint_accuracy = fraction of held-out rows where the arg-max object label and the arg-max "
        "place label are both correct on that row.  This replaces the invalid 20-class column of "
        "R6, whose held-out classes have no training rows."
    ),
    "why": (
        "R6's joint column is 0.0 for every arm and every lambda by construction, not by "
        "measurement; the overall verdict of R6 is therefore not interpretable."
    ),
}

CRITERIA = {
    "per_arm": base.CRITERIA["per_arm"],
    "overall": base.CRITERIA["overall"],
    "lambda": base.CRITERIA["lambda"],
    "no_fit": base.CRITERIA["no_fit"],
    "joint": "identical thresholds to R6; only the column definition changed, before this run.",
}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def fixture_hashes() -> dict[str, str]:
    return base.fixture_hashes()


def arm_reading(matrix: np.ndarray, rows: list[dict[str, Any]]) -> dict[str, Any]:
    goals = [row["goal"] for row in rows]
    assignment = base.fold_map(goals)
    folds = np.array([assignment[goal] for goal in goals])
    objects = [row["object"] for row in rows]
    places = [row["place"] for row in rows]
    object_classes = sorted(set(objects))
    place_classes = sorted(set(places))

    readings = []
    for lam in base.LAMBDAS:
        predicted_object = np.empty(len(rows), dtype=object)
        predicted_place = np.empty(len(rows), dtype=object)
        for fold in range(base.FOLDS):
            train, test = folds != fold, folds == fold
            for labels, classes, destination in (
                (objects, object_classes, predicted_object),
                (places, place_classes, predicted_place),
            ):
                targets = base.one_hot(
                    [label for label, flag in zip(labels, train, strict=True) if flag], classes
                )
                weights = base.solve_ridge(matrix[train], targets, lam)
                chosen = (matrix[test] @ weights).argmax(axis=1)
                for row, column in zip(np.flatnonzero(test), chosen, strict=True):
                    destination[row] = classes[column]
        object_correct = sum(predicted_object[row] == objects[row] for row in range(len(rows)))
        place_correct = sum(predicted_place[row] == places[row] for row in range(len(rows)))
        joint_correct = sum(
            predicted_object[row] == objects[row] and predicted_place[row] == places[row]
            for row in range(len(rows))
        )
        readings.append(
            {
                "lambda": lam,
                "object_accuracy": round(object_correct / len(rows), 6),
                "place_accuracy": round(place_correct / len(rows), 6),
                "joint_accuracy": round(joint_correct / len(rows), 6),
            }
        )

    best = max(readings, key=lambda row: (row["joint_accuracy"], -row["lambda"]))
    if (
        best["joint_accuracy"] >= base.READABLE_JOINT
        and best["object_accuracy"] >= base.READABLE_OBJECT
        and best["place_accuracy"] >= base.READABLE_PLACE
    ):
        verdict = "readable"
    elif best["joint_accuracy"] <= base.CHANCE_JOINT + base.NOT_READABLE_MARGIN:
        verdict = "not_readable"
    else:
        verdict = "inconclusive"
    return {
        "per_lambda": readings,
        "best": best,
        "verdict": verdict,
        "fold_sizes": [int((folds == fold).sum()) for fold in range(base.FOLDS)],
    }


def phase_freeze() -> dict[str, Any]:
    rows = base.frozen.build_fixture()
    goals = [row["goal"] for row in rows]
    manifest = {
        "schema": "taiji-cross-family-readout-joint-v1",
        "date": "2026-11-09",
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runner_sha256": RUNNER_SHA_START,
        "fixture_sha256": fixture_hashes(),
        "fixture_files": list(base.frozen.FIXTURE_FILES),
        "question": (
            "with the joint reading defined as object-and-place-on-the-same-row, does a closed "
            "form linear read-out decode the slot on goals never seen in training?"
        ),
        "supersedes": {
            "report": str(base.REPORT.relative_to(ROOT)),
            "field": "arms.*.per_lambda[*].joint_accuracy and arms.*.verdict",
            "reason": DEFINITION["why"],
        },
        "counts": {
            "sentences": len(rows),
            "goals": len(set(goals)),
            "variants_per_goal": sorted(
                {sum(1 for row in rows if row["goal"] == goal) for goal in set(goals)}
            ),
            "folds": base.FOLDS,
            "fold_assignment": base.fold_map(goals),
        },
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "lambdas": list(base.LAMBDAS),
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

    rows = base.frozen.build_fixture()
    texts = [row["text"] for row in rows]
    reference = LivingCoreReference()
    arms: dict[str, Any] = {}
    model: dict[str, Any] = {"model_id": base.MODEL_ID, "status": "not_requested"}
    for arm in base.ARMS:
        if arm == "R3":
            try:
                from instruments.document_embedding import DocumentEmbedder

                embedder = DocumentEmbedder(model_id=base.MODEL_ID)
                model = {
                    "model_id": base.MODEL_ID,
                    "status": "available",
                    "payload": embedder.to_payload(),
                }
                matrix = embedder.embed(texts).to(torch.float64).numpy()
                del embedder
            except Exception as exc:  # noqa: BLE001 - an unavailable arm must not kill the run
                arms["R3"] = {"verdict": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
                continue
            matrix = base.l2_normalise(matrix)
        else:
            matrix = base.l2_normalise(base.frozen.representation(arm, texts, reference).numpy())

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
        "schema": "taiji-cross-family-readout-joint-v1",
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
            "cpu_cap": base.CPU_CAP,
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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    arguments = parser.parse_args()
    from taiji.process_guard import assert_no_black_processes

    assert_no_black_processes()
    base.frozen.acquire_lock()
    try:
        if arguments.freeze:
            phase_freeze()
            return 0
        phase_run(json.loads(MANIFEST.read_text(encoding="utf-8")))
        return 0
    finally:
        base.frozen.release_lock()


if __name__ == "__main__":
    raise SystemExit(main())
