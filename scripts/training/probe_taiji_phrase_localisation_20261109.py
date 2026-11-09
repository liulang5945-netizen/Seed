"""R9 -- phrase-localisation gate: does the description phrase survive inside the sentence?

R8 closed one half of the bridge question.  With the same frozen encoder:

* phrase alone -> name: object 1.00, place 0.60 (R3);
* describe sentence -> name: object 0.85, place 0.60 (R3);
* the phrase->sentence drop on places is 0.00, so the place gap is *not* in-sentence dilution.

What R8 could not separate: the place phrase may be absent from the sentence
representation (a representation-level loss), or it may be present but not linked
to the place *name* (a mapping-level gap).  R8 measured the second arrow only.

This round measures the first arrow without touching names at all:

    take the 20 describe sentences and the 9 description phrases (4 object,
    5 place).  For every sentence, ask whether its own description phrase is the
    top-1 nearest phrase within its slot (4-way for object, 5-way for place).
    Chance is 0.25 / 0.20.  Zero training.

Metric positive control: R1 (``content_feature``) is an additive 128-bin byte
histogram, and the describe sentence contains the phrase as a literal substring
(``让{phrase_obj}到{phrase_place}。``).  Its own phrase must therefore win; if R1
does not localise, the metric is broken and the round is ``control_failed`` --
the text is not blamed.

Arms are the R7/R8 arms unchanged: R1 byte histogram, R2 untrained project
encoder, R3 the frozen sentence-transformer.  **R3 is an upper bound only; its
reading never enters the capability account and cannot upgrade or downgrade any
earlier verdict.**

Frozen attribution (read from R3 place localisation):

* ``phrase_survives_name_link_absent`` iff R3 place localisation >= 0.80 -- the
  phrase is still in the sentence, so the 0.60 place name link of R8 is a
  name-mapping gap, not a representation gap;
* ``phrase_diluted_or_absent`` iff R3 place localisation <= 0.60 -- the phrase
  itself is lost inside the sentence;
* ``inconclusive`` otherwise.

The project side (R1/R2) is reported on its own line and is not affected by R3.

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

MANIFEST = ROOT / "plans/manifests/taiji_phrase_localisation_20261109.json"
REPORT = ROOT / "reports/taiji_phrase_localisation_20261109.json"
R8_REPORT = ROOT / "reports/taiji_slot_bridge_20261109.json"

RUNNER_SHA_START = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()

CONTROL_FLOOR = 0.95
LOCALISED_FLOOR = 0.80
NOT_LOCALISED_CEILING = 0.60
COST_ESTIMATE_CPU_UPPER = 45.0

HEADER_NOTE = "R3 使用外部冻结句向量，只作上界；它的读数不进能力账，也不升级/降级任何既有判定。"

DEFINITION = {
    "fixture": (
        "frozen.build_fixture() describe rows only: 20 goals = 4 objects x 5 places, each written "
        "once as describe_goal = '让{object phrase}到{place phrase}。'.  No paraphrase row is used."
    ),
    "phrases": (
        "the 9 description phrases from OBJECT_DESCRIPTIONS (4) and PLACE_DESCRIPTIONS (5).  They "
        "are the only candidates; names (蓝色杯子 / 木桌 / ...) do not appear in this round at all."
    ),
    "reading": (
        "for each describe sentence, cosine similarity against the 4 object phrases and against "
        "the 5 place phrases; a slot is correct iff the sentence's own phrase is the arg-max.  "
        "20 samples per slot, so chance is 0.25 (object) and 0.20 (place)."
    ),
    "detail": (
        "besides accuracy the report keeps, per sample, the own-phrase cosine, the best other "
        "phrase, the margin own - best-other, and the 1-based rank of the own phrase."
    ),
    "control": (
        "R1 is an additive byte histogram and the phrase occurs verbatim inside the sentence, so "
        "R1 localisation is the metric's positive control: it must be >= 0.95 on both slots."
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
    "chance": {"object": 0.25, "place": 0.20},
    "chance_rank": {"object": 2.5, "place": 3.0},
}

CRITERIA = {
    "control": (
        "control_ok iff R1 object localisation >= 0.95 and R1 place localisation >= 0.95.  If it "
        "fails the round is control_failed and no other reading is interpreted."
    ),
    "localised": (
        "a slot is localised iff accuracy >= 0.80, not_localised iff accuracy <= 0.60, otherwise "
        "inconclusive.  Thresholds are fixed here, before the first reading."
    ),
    "attribution": (
        "read on R3 place only: phrase_survives_name_link_absent iff >= 0.80; "
        "phrase_diluted_or_absent iff <= 0.60; otherwise inconclusive."
    ),
    "project_side": (
        "R1/R2 localisation is reported separately: project_phrase_localised iff either arm is "
        "localised on both slots, project_phrase_not_localised iff neither arm is.  R3 cannot "
        "change this line."
    ),
    "r3_unavailable": "if the R3 weights are not cached the verdict is r3_unavailable and no attribution is made.",
    "no_fit": base.CRITERIA["no_fit"],
}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def payload_fixture() -> dict[str, Any]:
    rows = base.frozen.build_fixture()
    describe = [row for row in rows if row["variant"] == "describe"]
    return {
        "phrase_object": [
            {"text": phrase, "target": name} for name, phrase in OBJECT_DESCRIPTIONS.items()
        ],
        "phrase_place": [
            {"text": phrase, "target": name} for name, phrase in PLACE_DESCRIPTIONS.items()
        ],
        "describe": describe,
        "object_names": list(OBJECTS),
        "place_names": list(PLACES),
    }


def embed_arm(arm: str, texts: list[str], reference: LivingCoreReference) -> np.ndarray | None:
    if arm == "R3":
        embedder = DocumentEmbedder(model_id=base.MODEL_ID)
        return base.l2_normalise(embedder.embed(texts).to(torch.float64).numpy())
    return base.l2_normalise(base.frozen.representation(arm, texts, reference).numpy())


def slot_reading(
    sentence_matrix: np.ndarray,
    phrase_matrix: np.ndarray,
    targets: list[str],
    names: list[str],
    texts: list[str],
    phrases: list[str],
) -> dict[str, Any]:
    index_of = {name: position for position, name in enumerate(names)}
    phrases_of = {name: phrase for name, phrase in zip(names, phrases, strict=True)}
    scores = sentence_matrix @ phrase_matrix.T
    details: list[dict[str, Any]] = []
    correct = 0
    rank_sum = 0
    own_sum = 0.0
    margin_sum = 0.0
    for row, scores_row in zip(targets, scores, strict=True):
        own = index_of[row]
        order = np.argsort(-scores_row, kind="stable")
        rank = int(np.where(order == own)[0][0]) + 1
        best_other = max(float(scores_row[other]) for other in range(len(names)) if other != own)
        hit = rank == 1
        correct += int(hit)
        rank_sum += rank
        own_sum += float(scores_row[own])
        margin_sum += float(scores_row[own]) - best_other
        details.append(
            {
                "target": row,
                "predicted": phrases[int(order[0])],
                "correct": hit,
                "rank": rank,
                "own_cosine": round(float(scores_row[own]), 6),
                "best_other": round(best_other, 6),
                "margin": round(float(scores_row[own]) - best_other, 6),
            }
        )
    for detail, text in zip(details, texts, strict=True):
        detail["text"] = text
        detail["own_phrase"] = phrases_of[detail["target"]]
    return {
        "accuracy": round(correct / len(targets), 6),
        "correct": correct,
        "of": len(targets),
        "mean_rank": round(rank_sum / len(targets), 6),
        "mean_own_cosine": round(own_sum / len(targets), 6),
        "mean_margin": round(margin_sum / len(targets), 6),
        "detail": details,
    }


def arm_reading(
    arm: str, payload: dict[str, Any], reference: LivingCoreReference
) -> dict[str, Any]:
    object_phrases = [item["text"] for item in payload["phrase_object"]]
    place_phrases = [item["text"] for item in payload["phrase_place"]]
    sentences = [row["text"] for row in payload["describe"]]
    texts = sentences + object_phrases + place_phrases
    matrix = embed_arm(arm, texts, reference)
    if matrix is None:
        return {"verdict": "unavailable"}
    n_sentence = len(sentences)
    n_object = len(object_phrases)
    sentence_matrix = matrix[:n_sentence]
    object_matrix = matrix[n_sentence : n_sentence + n_object]
    place_matrix = matrix[n_sentence + n_object :]
    object_reading = slot_reading(
        sentence_matrix,
        object_matrix,
        [row["object"] for row in payload["describe"]],
        payload["object_names"],
        sentences,
        object_phrases,
    )
    place_reading = slot_reading(
        sentence_matrix,
        place_matrix,
        [row["place"] for row in payload["describe"]],
        payload["place_names"],
        sentences,
        place_phrases,
    )
    return {
        "dimensions": int(matrix.shape[1]),
        "object": object_reading,
        "place": place_reading,
        "localised_both": object_reading["accuracy"] >= LOCALISED_FLOOR
        and place_reading["accuracy"] >= LOCALISED_FLOOR,
    }


def classify_slot(accuracy: float) -> str:
    if accuracy >= LOCALISED_FLOOR:
        return "localised"
    if accuracy <= NOT_LOCALISED_CEILING:
        return "not_localised"
    return "inconclusive"


def r8_reference() -> dict[str, Any]:
    if not R8_REPORT.exists():
        return {"status": "R8 report missing"}
    report = json.loads(R8_REPORT.read_text(encoding="utf-8"))
    reading = report["arms"].get("R3", {})
    if reading.get("verdict") == "unavailable":
        return {"status": "R8 R3 unavailable"}
    return {
        "status": "cited from reports/taiji_slot_bridge_20261109.json",
        "note": "R8 已发布读数，本闸不重算，只作两段箭头的并排对照。",
        "R3_phrase_to_name": {
            "object": reading["phrase"]["object_accuracy"],
            "place": reading["phrase"]["place_accuracy"],
        },
        "R3_sentence_to_name": {
            "object": reading["describe_sentence"]["object_accuracy"],
            "place": reading["describe_sentence"]["place_accuracy"],
        },
        "R8_verdict": report["verdict"],
    }


def phase_freeze() -> dict[str, Any]:
    payload = payload_fixture()
    manifest = {
        "schema": "taiji-phrase-localisation-v1",
        "date": "2026-11-09",
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runner_sha256": RUNNER_SHA_START,
        "fixture_sha256": base.fixture_hashes(),
        "fixture_files": list(base.frozen.FIXTURE_FILES),
        "question": (
            "R7/R8 的地点名 0.60：是描述短语在整句表示里丢了（表示层），还是短语还在但接不上名字"
            "（映射层）？"
        ),
        "counts": {
            "describe_sentences": len(payload["describe"]),
            "object_phrases": len(payload["phrase_object"]),
            "place_phrases": len(payload["phrase_place"]),
            "texts_per_arm": len(payload["describe"])
            + len(payload["phrase_object"])
            + len(payload["phrase_place"]),
            "samples_per_slot": len(payload["describe"]),
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

    r1 = arms.get("R1", {})
    control_ok = (
        r1.get("verdict") != "unavailable"
        and r1["object"]["accuracy"] >= CONTROL_FLOOR
        and r1["place"]["accuracy"] >= CONTROL_FLOOR
    )
    upper = arms.get("R3", {})
    if not control_ok:
        diagnosis = "control_failed"
    elif upper.get("verdict") == "unavailable":
        diagnosis = "r3_unavailable"
    else:
        place_accuracy = upper["place"]["accuracy"]
        if place_accuracy >= LOCALISED_FLOOR:
            diagnosis = "phrase_survives_name_link_absent"
        elif place_accuracy <= NOT_LOCALISED_CEILING:
            diagnosis = "phrase_diluted_or_absent"
        else:
            diagnosis = "inconclusive"

    usable = [
        arm for arm in ("R1", "R2") if arm in arms and arms[arm].get("verdict") != "unavailable"
    ]
    project_localised = any(arms[arm].get("localised_both") for arm in usable)
    project_any = any(
        arms[arm]["object"]["accuracy"] > NOT_LOCALISED_CEILING
        or arms[arm]["place"]["accuracy"] > NOT_LOCALISED_CEILING
        for arm in usable
    )
    if project_localised:
        project_line = "project_phrase_localised"
    elif project_any:
        project_line = "project_phrase_inconclusive"
    else:
        project_line = "project_phrase_not_localised"

    report = {
        "schema": "taiji-phrase-localisation-v1",
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
        "slot_verdict": {
            arm: {
                "object": classify_slot(arms[arm]["object"]["accuracy"]),
                "place": classify_slot(arms[arm]["place"]["accuracy"]),
            }
            for arm in arms
            if arms[arm].get("verdict") != "unavailable"
        },
        "project_line": project_line,
        "contrast": {
            "r8_reference": r8_reference(),
            "r9_localisation_R3": (
                {
                    "object": upper["object"]["accuracy"],
                    "place": upper["place"]["accuracy"],
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
        "cost": report["cost"],
    }
    for arm in ("R1", "R2", "R3"):
        if arm in arms and arms[arm].get("verdict") != "unavailable":
            summary[arm] = {
                "object": [
                    arms[arm]["object"]["accuracy"],
                    arms[arm]["object"]["mean_rank"],
                    arms[arm]["object"]["mean_margin"],
                ],
                "place": [
                    arms[arm]["place"]["accuracy"],
                    arms[arm]["place"]["mean_rank"],
                    arms[arm]["place"]["mean_margin"],
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
