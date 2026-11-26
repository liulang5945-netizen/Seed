"""R8 -- slot-wise bridge diagnosis: isolated description phrase vs full sentence.

R7 left one asymmetry unexplained: with the same frozen encoder, the read-out
recovers object names from descriptions at 0.983 but place names only at 0.583.
Two causes fit: the description->name link is weaker for places (semantic gap),
or the link exists but is diluted when the phrase sits inside a sentence.

This round separates them with one metric held fixed -- nearest-name retrieval,
zero training -- and varies only the *text* the encoder sees:

* ``phrase``            -- the 9 description phrases on their own, mapped to the
                           9 names they stand for (4 object phrases, 5 place);
* ``describe_sentence`` -- the 20 ``describe`` sentences, each asked for its
                           object name and its place name (20 samples per slot);
* ``paraphrase_sentence`` -- the 60 paraphrase sentences as a *control*: they
                           contain the names verbatim, so a correct metric must
                           score them high.  This is how a low score on the
                           describe sentences is kept from being blamed on the
                           metric itself.

Arms are the R7 arms unchanged: R1 byte histogram, R2 untrained project encoder,
R3 the frozen sentence-transformer.  **R3 is an upper bound only; its reading
never enters the capability account.**  Chance is 0.25 for object and 0.20 for
place; no threshold is applied to any distance.

Frozen attribution (per arm, then overall on R3):

* ``place_dilution``   iff R3 phrase place >= 0.80, R3 sentence place <= 0.65 and
  the drop is >= 0.25;
* ``place_semantic_gap`` iff R3 phrase place <= 0.60;
* ``place_inconclusive`` otherwise.

The project-side line is reported separately and cannot be upgraded by R3.

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
from instruments.document_embedding import DocumentEmbedder  # noqa: E402
from taiji.living_core import LivingCoreReference  # noqa: E402
from taiji.native_goal_world_wide import (  # noqa: E402
    OBJECT_DESCRIPTIONS,
    OBJECTS,
    PLACE_DESCRIPTIONS,
    PLACES,
)

torch.set_num_threads(1)

MANIFEST = ROOT / "plans/manifests/taiji_slot_bridge_20261109.json"
REPORT = ROOT / "reports/taiji_slot_bridge_20261109.json"

RUNNER_SHA_START = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()

PHRASE_READABLE_OBJECT = 0.75
PHRASE_READABLE_PLACE = 0.80
DILUTION_DROP = 0.25
DILUTION_PHRASE_FLOOR = 0.80
DILUTION_SENTENCE_CEILING = 0.65
GAP_PHRASE_CEILING = 0.60
COST_ESTIMATE_CPU_UPPER = 60.0

HEADER_NOTE = "R3 使用外部冻结句向量，只作上界；它的读数不进能力账，也不升级/降级任何既有判定。"

DEFINITION = {
    "phrase": (
        "the 9 description phrases from OBJECT_DESCRIPTIONS / PLACE_DESCRIPTIONS, each embedded "
        "alone and matched against the 4 object names or the 5 place names it may stand for."
    ),
    "describe_sentence": (
        "the 20 describe_goal sentences.  Each sentence yields one object prediction and one "
        "place prediction, so this level has 20 samples per slot."
    ),
    "paraphrase_sentence": (
        "the 60 needs / hope / state sentences, as a control: they contain the names verbatim, "
        "so a sound metric must score them high."
    ),
    "metric": (
        "top-1 nearest name by cosine similarity within the slot's candidate set; zero training, "
        "ties broken by the smaller candidate index.  The same metric is used at every level, so "
        "phrase and sentence numbers are directly comparable."
    ),
    "dilution_index": "phrase accuracy minus describe-sentence accuracy, per slot.",
    "chance": {"object": 0.25, "place": 0.20},
}

CRITERIA = {
    "phrase_readable": (
        "an arm is phrase-readable iff object phrase accuracy >= 0.75 and place phrase accuracy "
        ">= 0.80 (the same thresholds as R7, applied to 4 and 5 phrases)."
    ),
    "dilution": (
        "place_dilution iff R3 phrase place >= 0.80, R3 describe-sentence place <= 0.65 and the "
        "drop is >= 0.25; place_semantic_gap iff R3 phrase place <= 0.60; else inconclusive."
    ),
    "project_side": (
        "the R1/R2 line is reported on its own: project_bridge_absent iff neither R1 nor R2 is "
        "phrase-readable.  R3 cannot change this line."
    ),
    "control": (
        "the paraphrase control must score >= 0.90 on both slots for every available arm; if it "
        "does not, the metric is broken and the round is inconclusive regardless of other numbers."
    ),
    "no_fit": base.CRITERIA["no_fit"],
}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def texts_and_targets() -> dict[str, Any]:
    phrase_object = [
        {"text": phrase, "target": name, "slot": "object"}
        for name, phrase in OBJECT_DESCRIPTIONS.items()
    ]
    phrase_place = [
        {"text": phrase, "target": name, "slot": "place"}
        for name, phrase in PLACE_DESCRIPTIONS.items()
    ]
    rows = base.frozen.build_fixture()
    sentences = [row["text"] for row in rows]
    return {
        "phrase_object": phrase_object,
        "phrase_place": phrase_place,
        "describe_sentences": [
            {"text": row["text"], "object": row["object"], "place": row["place"]}
            for row in rows
            if row["variant"] == "describe"
        ],
        "paraphrase_sentences": [
            {"text": row["text"], "object": row["object"], "place": row["place"]}
            for row in rows
            if row["variant"] != "describe"
        ],
        "sentences": sentences,
        "object_names": list(OBJECTS),
        "place_names": list(PLACES),
    }


def embed_arm(arm: str, texts: list[str], reference: LivingCoreReference) -> np.ndarray | None:
    if arm == "R3":
        embedder = DocumentEmbedder(model_id=base.MODEL_ID)
        return base.l2_normalise(embedder.embed(texts).to(torch.float64).numpy())
    return base.l2_normalise(base.frozen.representation(arm, texts, reference).numpy())


def nearest_names(matrix: np.ndarray, names: np.ndarray) -> list[int]:
    scores = matrix @ names.T
    return [int(np.argmax(row)) for row in scores]


def arm_reading(
    arm: str, payload: dict[str, Any], reference: LivingCoreReference
) -> dict[str, Any]:
    object_names = payload["object_names"]
    place_names = payload["place_names"]
    texts = (
        [item["text"] for item in payload["phrase_object"]]
        + [item["text"] for item in payload["phrase_place"]]
        + [item["text"] for item in payload["describe_sentences"]]
        + [item["text"] for item in payload["paraphrase_sentences"]]
        + object_names
        + place_names
    )
    matrix = embed_arm(arm, texts, reference)
    if matrix is None:
        return {"verdict": "unavailable"}
    n_object = len(payload["phrase_object"])
    n_place = len(payload["phrase_place"])
    n_describe = len(payload["describe_sentences"])
    n_paraphrase = len(payload["paraphrase_sentences"])
    cursor = 0
    phrase_object_matrix = matrix[cursor : cursor + n_object]
    cursor += n_object
    phrase_place_matrix = matrix[cursor : cursor + n_place]
    cursor += n_place
    describe_matrix = matrix[cursor : cursor + n_describe]
    cursor += n_describe
    paraphrase_matrix = matrix[cursor : cursor + n_paraphrase]
    cursor += n_paraphrase
    object_name_matrix = matrix[cursor : cursor + len(object_names)]
    cursor += len(object_names)
    place_name_matrix = matrix[cursor : cursor + len(place_names)]

    def detail(
        rows: list[dict[str, Any]], predicted: list[int], slot: str, names: list[str]
    ) -> list[dict[str, Any]]:
        return [
            {
                "text": row["text"],
                "target": row[slot],
                "predicted": names[index],
                "correct": names[index] == row[slot],
            }
            for row, index in zip(rows, predicted, strict=True)
        ]

    phrase_object_pred = nearest_names(phrase_object_matrix, object_name_matrix)
    phrase_place_pred = nearest_names(phrase_place_matrix, place_name_matrix)
    describe_object_pred = nearest_names(describe_matrix, object_name_matrix)
    describe_place_pred = nearest_names(describe_matrix, place_name_matrix)
    paraphrase_object_pred = nearest_names(paraphrase_matrix, object_name_matrix)
    paraphrase_place_pred = nearest_names(paraphrase_matrix, place_name_matrix)

    phrase_object_rows = detail(
        payload["phrase_object"], phrase_object_pred, "target", object_names
    )
    phrase_place_rows = detail(payload["phrase_place"], phrase_place_pred, "target", place_names)
    describe_object_rows = detail(
        payload["describe_sentences"], describe_object_pred, "object", object_names
    )
    describe_place_rows = detail(
        payload["describe_sentences"], describe_place_pred, "place", place_names
    )
    paraphrase_object_rows = detail(
        payload["paraphrase_sentences"], paraphrase_object_pred, "object", object_names
    )
    paraphrase_place_rows = detail(
        payload["paraphrase_sentences"], paraphrase_place_pred, "place", place_names
    )

    def accuracy(rows: list[dict[str, Any]]) -> float:
        return round(sum(row["correct"] for row in rows) / len(rows), 6)

    object_phrase = accuracy(phrase_object_rows)
    place_phrase = accuracy(phrase_place_rows)
    return {
        "dimensions": int(matrix.shape[1]),
        "phrase": {
            "object_accuracy": object_phrase,
            "place_accuracy": place_phrase,
            "object_detail": phrase_object_rows,
            "place_detail": phrase_place_rows,
        },
        "describe_sentence": {
            "object_accuracy": accuracy(describe_object_rows),
            "place_accuracy": accuracy(describe_place_rows),
            "object_detail": describe_object_rows,
            "place_detail": describe_place_rows,
        },
        "paraphrase_control": {
            "object_accuracy": accuracy(paraphrase_object_rows),
            "place_accuracy": accuracy(paraphrase_place_rows),
        },
        "dilution_index": {
            "object": round(object_phrase - accuracy(describe_object_rows), 6),
            "place": round(place_phrase - accuracy(describe_place_rows), 6),
        },
        "phrase_readable": object_phrase >= PHRASE_READABLE_OBJECT
        and place_phrase >= PHRASE_READABLE_PLACE,
    }


def classify_place(reading: dict[str, Any]) -> str:
    phrase_place = reading["phrase"]["place_accuracy"]
    sentence_place = reading["describe_sentence"]["place_accuracy"]
    if (
        phrase_place >= DILUTION_PHRASE_FLOOR
        and sentence_place <= DILUTION_SENTENCE_CEILING
        and phrase_place - sentence_place >= DILUTION_DROP
    ):
        return "place_dilution"
    if phrase_place <= GAP_PHRASE_CEILING:
        return "place_semantic_gap"
    return "place_inconclusive"


def phase_freeze() -> dict[str, Any]:
    payload = texts_and_targets()
    manifest = {
        "schema": "taiji-slot-bridge-v1",
        "date": "2026-11-09",
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runner_sha256": RUNNER_SHA_START,
        "fixture_sha256": base.fixture_hashes(),
        "fixture_files": list(base.frozen.FIXTURE_FILES),
        "question": ("R7 的地点名 0.583 是句内稀释，还是描述到名字的桥本来就不存在？"),
        "counts": {
            "phrases": len(payload["phrase_object"]) + len(payload["phrase_place"]),
            "describe_sentences": len(payload["describe_sentences"]),
            "paraphrase_control": len(payload["paraphrase_sentences"]),
            "texts_per_arm": (
                len(payload["phrase_object"])
                + len(payload["phrase_place"])
                + len(payload["describe_sentences"])
                + len(payload["paraphrase_sentences"])
                + len(payload["object_names"])
                + len(payload["place_names"])
            ),
        },
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

    payload = texts_and_targets()
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

    control_ok = all(
        arm["paraphrase_control"]["object_accuracy"] >= 0.90
        and arm["paraphrase_control"]["place_accuracy"] >= 0.90
        for arm in arms.values()
        if arm.get("verdict") != "unavailable"
    )
    upper = arms.get("R3", {})
    if upper.get("verdict") == "unavailable":
        diagnosis = "r3_unavailable"
    elif not control_ok:
        diagnosis = "control_failed"
    else:
        diagnosis = classify_place(upper)

    project_readable = any(arms[arm].get("phrase_readable") for arm in ("R1", "R2") if arm in arms)
    report = {
        "schema": "taiji-slot-bridge-v1",
        "date": "2026-11-09",
        "header_note": HEADER_NOTE,
        "runner_sha256": RUNNER_SHA_START,
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "fixture_sha256": manifest["fixture_sha256"],
        "counts": manifest["counts"],
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "model": model,
        "arms": arms,
        "control_ok": control_ok,
        "verdict": diagnosis,
        "project_bridge": "project_bridge_present" if project_readable else "project_bridge_absent",
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
        "project_bridge": report["project_bridge"],
        "control_ok": control_ok,
        "cost": report["cost"],
    }
    for arm in ("R1", "R2", "R3"):
        if arm in arms and arms[arm].get("verdict") != "unavailable":
            summary[arm] = {
                "phrase": [
                    arms[arm]["phrase"]["object_accuracy"],
                    arms[arm]["phrase"]["place_accuracy"],
                ],
                "sentence": [
                    arms[arm]["describe_sentence"]["object_accuracy"],
                    arms[arm]["describe_sentence"]["place_accuracy"],
                ],
                "control": [
                    arms[arm]["paraphrase_control"]["object_accuracy"],
                    arms[arm]["paraphrase_control"]["place_accuracy"],
                ],
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
