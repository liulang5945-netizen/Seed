"""R7 -- zero-shot cross-family read-out: describe -> paraphrase.

R6b showed that a closed-form linear read-out decodes object and place on the
same 80 sentences, but with goal-grouped folds every entity *name* string still
appears in the training fold (each object has four goals, so its name shows up
in the other goals' paraphrases).  The 1.0 accuracy is therefore consistent
with plain vocabulary lookup; it does not show a bridge between the descriptions
(``describe_goal``) and the names used by the paraphrase templates.

This round separates the two.  The split is by *variant*, not by fold:

* train -- the 20 ``describe`` sentences, one per goal;
* test  -- the 60 ``needs`` / ``hope`` / ``state`` sentences.

No test sentence has a counterpart in training, and no object or place **name**
occurs in any training sentence (checked at run time and reported as
``leak_check``).  Everything else is inherited unchanged from R6b: same ridge
solve, same lambdas, same thresholds, same three arms, same 40 / 60 sentence
matrix built from the frozen fixture.

Reading: object accuracy, place accuracy, and joint = the same row's object AND
place both correct, all over the 60 held-out sentences.  Chance is 0.25 / 0.20 /
0.05.  Per arm, ``readable`` iff joint >= 0.75 and object >= 0.85 and place >=
0.80; ``not_readable`` iff joint <= chance + 0.10; otherwise ``inconclusive``.
A read-out that predicts a single class everywhere is additionally flagged
``degenerate`` and cannot be ``not_readable`` -- "no signal" and "constant
output" are different readings.  Overall: ``cross_family_semantic_bridge`` if
any arm is readable; ``cross_family_zero_shot_fails`` if every available arm is
not_readable; otherwise ``inconclusive``.

Zero training, zero optimizer steps, zero Adam, zero new parameters.
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
from taiji.native_goal_world_wide import OBJECTS, PLACES  # noqa: E402

torch.set_num_threads(1)

MANIFEST = ROOT / "plans/manifests/taiji_cross_family_zero_shot_20261109.json"
REPORT = ROOT / "reports/taiji_cross_family_zero_shot_20261109.json"

RUNNER_SHA_START = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()

DEFINITION = {
    "train": "the 20 describe sentences, one per (object, place) goal; no paraphrase.",
    "test": (
        "the 60 needs / hope / state sentences (3 per goal).  No test sentence has a training "
        "counterpart and the split is by variant, not by fold."
    ),
    "leak_check": (
        "for every object and place name, assert the name string does not occur in any of the 20 "
        "training sentences; the count of test sentences that do contain their own name is "
        "reported so the mapping is verifiable rather than assumed."
    ),
    "readout": base.DEFINITION["readout"],
    "lambdas": list(base.LAMBDAS),
    "arms": list(base.ARMS),
    "joint": "fraction of the 60 held-out rows where object AND place are both correct.",
}

CRITERIA = {
    "per_arm": (
        "readable iff joint >= 0.75 and object >= 0.85 and place >= 0.80; not_readable iff "
        "joint <= chance + 0.10; otherwise inconclusive.  A degenerate read-out (one predicted "
        "class for object or for place) is inconclusive, never not_readable."
    ),
    "overall": (
        "cross_family_semantic_bridge if any arm is readable; cross_family_zero_shot_fails if "
        "every available arm is not_readable; otherwise inconclusive."
    ),
    "lambda": "the reported reading is the best joint accuracy over the four lambdas.",
    "no_fit": base.CRITERIA["no_fit"],
    "frozen": "thresholds and both splits are fixed before the first reading.",
}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def fixture_hashes() -> dict[str, str]:
    return base.fixture_hashes()


def split_rows(rows: list[dict[str, Any]]) -> tuple[list[int], list[int]]:
    train = [index for index, row in enumerate(rows) if row["variant"] == "describe"]
    test = [index for index, row in enumerate(rows) if row["variant"] != "describe"]
    return train, test


def leak_check(rows: list[dict[str, Any]], train: list[int], test: list[int]) -> dict[str, Any]:
    training_texts = [rows[index]["text"] for index in train]
    names = list(OBJECTS) + list(PLACES)
    leaked = [name for name in names if any(name in text for text in training_texts)]
    test_with_object = sum(rows[index]["object"] in rows[index]["text"] for index in test)
    test_with_place = sum(rows[index]["place"] in rows[index]["text"] for index in test)
    if leaked:
        raise SystemExit(f"name leaked into the training split: {leaked}")
    return {
        "names_checked": names,
        "names_in_training": leaked,
        "test_rows": len(test),
        "test_rows_containing_object_name": test_with_object,
        "test_rows_containing_place_name": test_with_place,
    }


def arm_reading(
    matrix: np.ndarray, rows: list[dict[str, Any]], train: list[int], test: list[int]
) -> dict[str, Any]:
    objects = [row["object"] for row in rows]
    places = [row["place"] for row in rows]
    object_classes = sorted(set(objects))
    place_classes = sorted(set(places))
    train_index = np.array(train)
    test_index = np.array(test)

    readings = []
    for lam in base.LAMBDAS:
        predicted = {}
        for label, classes in (("object", object_classes), ("place", place_classes)):
            labels = objects if label == "object" else places
            targets = base.one_hot([labels[index] for index in train], classes)
            weights = base.solve_ridge(matrix[train_index], targets, lam)
            chosen = (matrix[test_index] @ weights).argmax(axis=1)
            predicted[label] = [classes[column] for column in chosen]
        object_correct = sum(
            predicted["object"][row] == objects[index] for row, index in enumerate(test)
        )
        place_correct = sum(
            predicted["place"][row] == places[index] for row, index in enumerate(test)
        )
        joint_correct = sum(
            predicted["object"][row] == objects[index] and predicted["place"][row] == places[index]
            for row, index in enumerate(test)
        )
        readings.append(
            {
                "lambda": lam,
                "object_accuracy": round(object_correct / len(test), 6),
                "place_accuracy": round(place_correct / len(test), 6),
                "joint_accuracy": round(joint_correct / len(test), 6),
                "distinct_object_predictions": len(set(predicted["object"])),
                "distinct_place_predictions": len(set(predicted["place"])),
            }
        )

    best = max(readings, key=lambda row: (row["joint_accuracy"], -row["lambda"]))
    degenerate = best["distinct_object_predictions"] < 2 or best["distinct_place_predictions"] < 2
    if (
        best["joint_accuracy"] >= base.READABLE_JOINT
        and best["object_accuracy"] >= base.READABLE_OBJECT
        and best["place_accuracy"] >= base.READABLE_PLACE
    ):
        verdict = "readable"
    elif degenerate:
        verdict = "inconclusive"
    elif best["joint_accuracy"] <= base.CHANCE_JOINT + base.NOT_READABLE_MARGIN:
        verdict = "not_readable"
    else:
        verdict = "inconclusive"
    return {
        "per_lambda": readings,
        "best": best,
        "degenerate": degenerate,
        "verdict": verdict,
        "train_rows": len(train),
        "test_rows": len(test),
    }


def phase_freeze() -> dict[str, Any]:
    rows = base.frozen.build_fixture()
    train, test = split_rows(rows)
    manifest = {
        "schema": "taiji-cross-family-zero-shot-v1",
        "date": "2026-11-09",
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runner_sha256": RUNNER_SHA_START,
        "fixture_sha256": fixture_hashes(),
        "fixture_files": list(base.frozen.FIXTURE_FILES),
        "question": (
            "with only the 20 describe sentences in training, can the read-out recover object and "
            "place on the 60 paraphrase sentences whose entity names never occur in training?"
        ),
        "counts": {
            "sentences": len(rows),
            "train": len(train),
            "test": len(test),
            "variants_test": sorted({rows[index]["variant"] for index in test}),
        },
        "leak_check": leak_check(rows, train, test),
        "definition": DEFINITION,
        "criteria": CRITERIA,
    }
    write_json(MANIFEST, manifest)
    print(
        json.dumps(
            {"manifest": str(MANIFEST), "counts": manifest["counts"]},
            ensure_ascii=False,
        )
    )
    return manifest


def phase_run(manifest: dict[str, Any]) -> dict[str, Any]:
    started_cpu = time.process_time()
    started_wall = time.time()
    if fixture_hashes() != manifest["fixture_sha256"]:
        raise SystemExit("fixture changed after the freeze; re-freeze before running")
    if manifest["runner_sha256"] != RUNNER_SHA_START:
        raise SystemExit("runner changed after the freeze; re-freeze before running")

    rows = base.frozen.build_fixture()
    train, test = split_rows(rows)
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
                arms["R3"] = {
                    "verdict": "unavailable",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
                continue
            matrix = base.l2_normalise(matrix)
        else:
            matrix = base.l2_normalise(base.frozen.representation(arm, texts, reference).numpy())

        reading = arm_reading(matrix, rows, train, test)
        reading["dimensions"] = int(matrix.shape[1])
        arms[arm] = reading

    verdicts = [arm["verdict"] for arm in arms.values() if arm["verdict"] != "unavailable"]
    if "readable" in verdicts:
        overall = "cross_family_semantic_bridge"
    elif verdicts and all(item == "not_readable" for item in verdicts):
        overall = "cross_family_zero_shot_fails"
    else:
        overall = "inconclusive"

    report = {
        "schema": "taiji-cross-family-zero-shot-v1",
        "date": "2026-11-09",
        "runner_sha256": RUNNER_SHA_START,
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "fixture_sha256": manifest["fixture_sha256"],
        "counts": manifest["counts"],
        "leak_check": manifest["leak_check"],
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "chance": {
            "object": 0.25,
            "place": 0.20,
            "joint": base.CHANCE_JOINT,
        },
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
