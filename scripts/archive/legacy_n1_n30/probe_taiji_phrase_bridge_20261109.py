"""R10 -- leave-one-out phrase bridge: does a closed-form phrase->name map generalise?

R9 established that the description phrase survives inside the sentence (R3 place
localisation 20/20) while the place *name* is still only 0.60 reachable (R8).
That leaves one question the two rounds cannot answer between them: is the
phrase->name link a *generalising* map -- something a bridge layer can be built
from -- or only a memorised table where each phrase returns its own entry?

Design (zero training, zero optimizer steps, zero new parameters):

* bank = the 9 (phrase, name) pairs, 4 object and 5 place.  Names never occur
  as input; they are the regression target and the candidate set.
* bridge = closed-form ridge from an input vector to the name embedding, then
  nearest name by cosine among the slot's full name set.  The removed name
  stays a *candidate* -- only its training row disappears.
* column A ``before``: fit on all n rows, read every row back (in-sample).  This
  is the "can the table be memorised" side.
* column B ``after``: leave-one-out.  For a held-out row the bank drops from
  4 to 3 (object) or 5 to 4 (place): exactly **one entry removed per item**.
  This is the "does it generalise" side.
* two input modes: ``phrase`` (same domain, the literal phrase->name arrow) and
  ``sentence`` (cross domain: fit on phrases, read the describe sentence -- the
  pipeline R9 makes possible).  Sentence mode runs at the primary lambda only.
* arms as in R7/R8/R9.  R1 is the machinery control: if its in-sample fit is not
  >= 0.95 on both slots the closed-form path is broken and the round is
  ``control_failed``.  **R3 is an upper bound only; it never enters the
  capability account.**

Lambda is fixed a priori at 1.0 for the verdict (unit ridge on L2-normalised
features).  The grid {1e-3, 1e-1, 1, 1e1, 1e3} is reported as robustness; a
drop to chance must hold at *every* lambda before ``memorised`` is claimed.

Frozen attribution, read on R3 place:

* ``phrase_name_bridge_generalises`` iff LOO >= 0.80 at lambda 1.0;
* ``phrase_name_bridge_memorised`` iff LOO <= 0.60 at every lambda in the grid;
* ``bridge_inconclusive`` otherwise.

Zero training, zero optimizer steps, zero Adam, zero confirm, zero new parameters.
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
from instruments.document_embedding import DocumentEmbedder  # noqa: E402
from taiji.living_core import LivingCoreReference  # noqa: E402
from taiji.native_goal_world_wide import (  # noqa: E402
    OBJECT_DESCRIPTIONS,
    OBJECTS,
    PLACE_DESCRIPTIONS,
    PLACES,
)

torch.set_num_threads(1)

MANIFEST = ROOT / "plans/manifests/taiji_phrase_bridge_20261109.json"
REPORT = ROOT / "reports/taiji_phrase_bridge_20261109.json"
R9_REPORT = ROOT / "reports/taiji_phrase_localisation_20261109.json"

RUNNER_SHA_START = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()

LAMBDA_GRID = (0.001, 0.1, 1.0, 10.0, 1000.0)
PRIMARY_LAMBDA = 1.0
CONTROL_FLOOR = 0.95
GENERALISE_FLOOR = 0.80
MEMORISED_CEILING = 0.60
COST_ESTIMATE_CPU_UPPER = 50.0

HEADER_NOTE = "R3 使用外部冻结句向量，只作上界；它的读数不进能力账，也不升级/降级任何既有判定。"

DEFINITION = {
    "bank": (
        "the 9 (phrase, name) pairs from OBJECT_DESCRIPTIONS (4) and PLACE_DESCRIPTIONS (5).  "
        "Names are never bridge input: they are the regression target and the candidate set."
    ),
    "bridge": (
        "closed-form ridge W = (X^T X + lambda I)^-1 X^T N, where N holds the slot's name "
        "embeddings; the read-out is x W, classified by nearest name by cosine."
    ),
    "column_before": (
        "fit on all n rows and read every row back (in-sample).  Establishes that the table "
        "itself is representable; it is not a generalisation reading."
    ),
    "column_after": (
        "leave-one-out: for each held-out row fit on the other n-1 rows and read that row.  "
        "The bank therefore drops 4->3 (object) or 5->4 (place): one entry removed per item.  "
        "The removed name stays a candidate, so the answer is not excluded by construction."
    ),
    "modes": {
        "phrase": "fit and test on the phrase vectors (same-domain phrase->name arrow).",
        "sentence": (
            "fit on the phrase bank and read the describe sentence (cross-domain: this is the "
            "pipeline R9 makes possible).  Runs at the primary lambda only."
        ),
    },
    "lambda": (
        "verdict at lambda = 1.0, fixed before the first reading.  The grid "
        "{1e-3, 1e-1, 1, 1e1, 1e3} is reported as robustness and is used only to reject "
        "the memorised claim."
    ),
    "arms": {
        "R1": "LivingCoreReference.content_feature (128-bin byte histogram, L2-normalised).",
        "R2": "mean of reference.encoder.context(raw) over positions (128 dims), untrained weights.",
        "R3": "DocumentEmbedder on the frozen sentence-transformer; upper bound only.",
    },
    "chance": {"object": 0.25, "place": 0.20},
}

CRITERIA = {
    "control": (
        "machinery control: R1 in-sample accuracy must be >= 0.95 on both slots at lambda 1.0.  "
        "If it is not, the closed-form path is broken and the round is control_failed."
    ),
    "generalises": (
        "phrase_name_bridge_generalises iff R3 place leave-one-out >= 0.80 at lambda 1.0."
    ),
    "memorised": (
        "phrase_name_bridge_memorised iff R3 place leave-one-out <= 0.60 at every lambda in "
        "the grid; otherwise bridge_inconclusive."
    ),
    "project_side": (
        "computed on R2 only (D-20261109-10): project_bridge_generalises iff R2 place "
        "leave-one-out >= 0.80, project_bridge_memorised iff <= 0.60, else inconclusive."
    ),
    "r3_unavailable": "if the R3 weights are not cached the verdict is r3_unavailable.",
    "no_fit": base.CRITERIA["no_fit"],
}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def payload_fixture() -> dict[str, Any]:
    rows = base.frozen.build_fixture()
    return {
        "phrase_object": [
            {"text": phrase, "target": name} for name, phrase in OBJECT_DESCRIPTIONS.items()
        ],
        "phrase_place": [
            {"text": phrase, "target": name} for name, phrase in PLACE_DESCRIPTIONS.items()
        ],
        "describe": [row for row in rows if row["variant"] == "describe"],
        "object_names": list(OBJECTS),
        "place_names": list(PLACES),
    }


def embed_arm(arm: str, texts: list[str], reference: LivingCoreReference) -> np.ndarray | None:
    if arm == "R3":
        embedder = DocumentEmbedder(model_id=base.MODEL_ID)
        return base.l2_normalise(embedder.embed(texts).to(torch.float64).numpy())
    return base.l2_normalise(base.frozen.representation(arm, texts, reference).numpy())


def fit_ridge(features: np.ndarray, targets: np.ndarray, lam: float) -> np.ndarray:
    width = features.shape[1]
    gram = features.T @ features + lam * np.eye(width)
    return np.linalg.solve(gram, features.T @ targets)


def nearest_name(predictions: np.ndarray, names: np.ndarray) -> list[int]:
    norms = np.linalg.norm(predictions, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    scores = (predictions / norms) @ names.T
    return [int(np.argmax(row)) for row in scores]


def accuracy(predicted: list[int], targets: list[int]) -> tuple[int, float]:
    correct = sum(int(left == right) for left, right in zip(predicted, targets, strict=True))
    return correct, round(correct / len(targets), 6)


def slot_bridge(
    fit_features: np.ndarray,
    fit_targets: np.ndarray,
    read_features: np.ndarray,
    read_targets: list[str],
    name_matrix: np.ndarray,
    names: list[str],
    lam: float,
    delete_index: list[int] | None = None,
) -> dict[str, Any]:
    index_of = {name: position for position, name in enumerate(names)}
    predicted: list[int] = []
    for row, deletion in zip(read_features, delete_index or [-1] * len(read_features), strict=True):
        if deletion < 0:
            keep = np.ones(len(fit_features), dtype=bool)
        else:
            keep = np.arange(len(fit_features)) != deletion
        weights = fit_ridge(fit_features[keep], fit_targets[keep], lam)
        predicted.append(nearest_name(row[None, :] @ weights, name_matrix)[0])
    correct, score = accuracy(predicted, [index_of[target] for target in read_targets])
    return {
        "accuracy": score,
        "correct": correct,
        "of": len(read_targets),
        "bank_size": len(fit_features) - (1 if delete_index else 0),
        "predicted": [names[index] for index in predicted],
    }


def slot_reading(
    phrase_matrix: np.ndarray,
    sentence_matrix: np.ndarray,
    name_matrix: np.ndarray,
    sentence_targets: list[str],
    phrase_targets: list[str],
    names: list[str],
) -> dict[str, Any]:
    phrase_index = [names.index(target) for target in phrase_targets]
    grid: dict[str, Any] = {}
    for lam in LAMBDA_GRID:
        before = slot_bridge(
            phrase_matrix, name_matrix, phrase_matrix, phrase_targets, name_matrix, names, lam
        )
        after = slot_bridge(
            phrase_matrix,
            name_matrix,
            phrase_matrix,
            phrase_targets,
            name_matrix,
            names,
            lam,
            delete_index=list(range(len(phrase_targets))),
        )
        grid[f"{lam:g}"] = {"before": before, "after": after}
    sentence = slot_bridge(
        phrase_matrix,
        name_matrix,
        sentence_matrix,
        sentence_targets,
        name_matrix,
        names,
        PRIMARY_LAMBDA,
        delete_index=[
            phrase_index[phrase_targets.index(target)] if target in phrase_targets else -1
            for target in sentence_targets
        ],
    )
    return {
        "bank_size_before": len(phrase_targets),
        "bank_size_after": len(phrase_targets) - 1,
        "deleted_per_item": 1,
        "grid": grid,
        "primary": grid[f"{PRIMARY_LAMBDA:g}"],
        "sentence_mode": sentence,
    }


def arm_reading(
    arm: str, payload: dict[str, Any], reference: LivingCoreReference
) -> dict[str, Any]:
    object_phrases = [item["text"] for item in payload["phrase_object"]]
    place_phrases = [item["text"] for item in payload["phrase_place"]]
    object_names = payload["object_names"]
    place_names = payload["place_names"]
    sentences = [row["text"] for row in payload["describe"]]
    texts = object_phrases + place_phrases + object_names + place_names + sentences
    matrix = embed_arm(arm, texts, reference)
    if matrix is None:
        return {"verdict": "unavailable"}
    cursor = 0
    object_phrase_matrix = matrix[cursor : cursor + len(object_phrases)]
    cursor += len(object_phrases)
    place_phrase_matrix = matrix[cursor : cursor + len(place_phrases)]
    cursor += len(place_phrases)
    object_name_matrix = matrix[cursor : cursor + len(object_names)]
    cursor += len(object_names)
    place_name_matrix = matrix[cursor : cursor + len(place_names)]
    cursor += len(place_names)
    sentence_matrix = matrix[cursor:]
    return {
        "dimensions": int(matrix.shape[1]),
        "object": slot_reading(
            object_phrase_matrix,
            sentence_matrix,
            object_name_matrix,
            [row["object"] for row in payload["describe"]],
            [item["target"] for item in payload["phrase_object"]],
            object_names,
        ),
        "place": slot_reading(
            place_phrase_matrix,
            sentence_matrix,
            place_name_matrix,
            [row["place"] for row in payload["describe"]],
            [item["target"] for item in payload["phrase_place"]],
            place_names,
        ),
    }


def classify(score: float) -> str:
    if score >= GENERALISE_FLOOR:
        return "generalises"
    if score <= MEMORISED_CEILING:
        return "memorised"
    return "inconclusive"


def r9_reference() -> dict[str, Any]:
    if not R9_REPORT.exists():
        return {"status": "R9 report missing"}
    report = json.loads(R9_REPORT.read_text(encoding="utf-8"))
    reading = report["arms"].get("R3", {})
    if reading.get("verdict") == "unavailable":
        return {"status": "R9 R3 unavailable"}
    return {
        "status": "cited from reports/taiji_phrase_localisation_20261109.json",
        "note": "R9 已发布读数，本闸不重算，只作上界对照。",
        "R3_localisation": {
            "object": reading["object"]["accuracy"],
            "place": reading["place"]["accuracy"],
        },
        "R9_verdict": report["verdict"],
    }


def phase_freeze() -> dict[str, Any]:
    payload = payload_fixture()
    manifest = {
        "schema": "taiji-phrase-bridge-v1",
        "date": "2026-11-09",
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runner_sha256": RUNNER_SHA_START,
        "fixture_sha256": base.fixture_hashes(),
        "fixture_files": list(base.frozen.FIXTURE_FILES),
        "question": (
            "R9 说短语在句内、名字接不上：这条“短语→名字”是可外推的映射，还是只有背过的词条？"
        ),
        "counts": {
            "describe_sentences": len(payload["describe"]),
            "object_phrases": len(payload["phrase_object"]),
            "place_phrases": len(payload["phrase_place"]),
            "bank_before": {"object": 4, "place": 5},
            "bank_after": {"object": 3, "place": 4},
            "deleted_per_held_out": 1,
            "texts_per_arm": 2 * 9 + 20,
        },
        "lambda_grid": list(LAMBDA_GRID),
        "primary_lambda": PRIMARY_LAMBDA,
        "cost_estimate_cpu_upper": COST_ESTIMATE_CPU_UPPER,
        "header_note": HEADER_NOTE,
        "definition": DEFINITION,
        "criteria": CRITERIA,
    }
    write_json(MANIFEST, manifest)
    print(json.dumps({"manifest": str(MANIFEST), "counts": manifest["counts"]}, ensure_ascii=False))
    return manifest


def phase_run(manifest: dict[str, Any]) -> dict[str, Any]:
    started_cpu = time.process_time()
    started_wall = time.time()
    if base.fixture_hashes() != manifest["fixture_sha256"]:
        raise SystemExit("fixture changed after the freeze; re-freeze before running")
    if manifest["runner_sha256"] != RUNNER_SHA_START:
        raise SystemExit("runner changed after the freeze; re-freeze before running")

    payload = payload_fixture()
    reference = LivingCoreReference()
    arms: dict[str, Any] = {}
    model: dict[str, Any] = {"model_id": base.MODEL_ID, "status": "not_requested"}
    for arm in base.ARMS:
        if arm == "R3":
            try:
                embedder = DocumentEmbedder(model_id=base.MODEL_ID)
                model = {
                    "model_id": base.MODEL_ID,
                    "status": "available",
                    "payload": embedder.to_payload(),
                }
                del embedder
            except Exception as exc:  # noqa: BLE001 - an unavailable arm must not kill the run
                arms["R3"] = {"verdict": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
                continue
        arms[arm] = arm_reading(arm, payload, reference)

    control_ok = arms["R1"]["object"]["primary"]["before"]["accuracy"] >= CONTROL_FLOOR and (
        arms["R1"]["place"]["primary"]["before"]["accuracy"] >= CONTROL_FLOOR
    )
    upper = arms.get("R3", {})
    if not control_ok:
        diagnosis = "control_failed"
    elif upper.get("verdict") == "unavailable":
        diagnosis = "r3_unavailable"
    else:
        place = upper["place"]
        primary = place["primary"]["after"]["accuracy"]
        worst = max(place["grid"][key]["after"]["accuracy"] for key in place["grid"])
        if primary >= GENERALISE_FLOOR:
            diagnosis = "phrase_name_bridge_generalises"
        elif worst <= MEMORISED_CEILING:
            diagnosis = "phrase_name_bridge_memorised"
        else:
            diagnosis = "bridge_inconclusive"

    r2 = arms.get("R2", {})
    if r2.get("verdict") == "unavailable":
        project_line = "project_bridge_unavailable"
    else:
        project_line = "project_bridge_" + classify(r2["place"]["primary"]["after"]["accuracy"])

    report = {
        "schema": "taiji-phrase-bridge-v1",
        "date": "2026-11-09",
        "header_note": HEADER_NOTE,
        "runner_sha256": RUNNER_SHA_START,
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "fixture_sha256": manifest["fixture_sha256"],
        "counts": manifest["counts"],
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "lambda_grid": list(LAMBDA_GRID),
        "primary_lambda": PRIMARY_LAMBDA,
        "model": model,
        "arms": arms,
        "control_ok": control_ok,
        "verdict": diagnosis,
        "project_line": project_line,
        "slot_verdict": (
            {
                slot: {
                    "before": classify(upper[slot]["primary"]["before"]["accuracy"]),
                    "after": classify(upper[slot]["primary"]["after"]["accuracy"]),
                }
                for slot in ("object", "place")
            }
            if upper.get("verdict") != "unavailable"
            else {"status": "r3_unavailable"}
        ),
        "contrast": {
            "r9_reference": r9_reference(),
            "r10_R3_place": (
                {
                    "before": upper["place"]["primary"]["before"]["accuracy"],
                    "after": upper["place"]["primary"]["after"]["accuracy"],
                    "after_best_lambda": max(
                        upper["place"]["grid"][key]["after"]["accuracy"]
                        for key in upper["place"]["grid"]
                    ),
                }
                if upper.get("verdict") != "unavailable"
                else {"status": "unavailable"}
            ),
        },
        "cost": {
            "cpu_seconds": round(time.process_time() - started_cpu, 3),
            "wall_seconds": round(time.time() - started_wall, 2),
            "cpu_cap": base.CPU_CAP,
            "estimate_cpu_upper": manifest["cost_estimate_cpu_upper"],
            "threads": torch.get_num_threads(),
        },
    }
    write_json(REPORT, report)
    summary = {
        "verdict": diagnosis,
        "control_ok": control_ok,
        "project_line": project_line,
        "slot_verdict": report["slot_verdict"],
        "contrast": report["contrast"]["r10_R3_place"],
        "cost": report["cost"],
    }
    for arm in ("R1", "R2", "R3"):
        if arm in arms and arms[arm].get("verdict") != "unavailable":
            summary[arm] = {
                slot: {
                    "before": arms[arm][slot]["primary"]["before"]["accuracy"],
                    "after": arms[arm][slot]["primary"]["after"]["accuracy"],
                    "after_best": max(
                        arms[arm][slot]["grid"][key]["after"]["accuracy"]
                        for key in arms[arm][slot]["grid"]
                    ),
                    "sentence_after": arms[arm][slot]["sentence_mode"]["accuracy"],
                }
                for slot in ("object", "place")
            }
    print(json.dumps(summary, ensure_ascii=False))
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
