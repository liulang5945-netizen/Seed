"""R10b -- multi-wording leave-one-out gate: is the phrase->name bridge a map or a table?

R10 showed the 1:1 bank (one phrase per name) drops to chance once that single
row is removed.  That zero is *constructive*: with one training target per
concept there is nothing left to generalise from, so R10 only proves the 1:1
table must be written, not that a bridge cannot generalise (D-20261109-11).

R10b removes that confound.  Every name gets a **second description wording**,
defined here in the runner (never in ``taiji/native_goal_world_wide.py``, whose
bytes are frozen by six earlier gates).  The bank becomes 2 rows per name --
8 object, 10 place -- while the sentences still use the first (primary) wording.

Reading, per slot:

* ``before``  -- fit the whole two-wording bank and read the primary rows back
  (in-sample).  This is the "can the table be represented at all" side.
* ``after``   -- for each primary row, drop **that one entry** from the bank
  (2n -> 2n-1), refit, and read the held-out row.  The other wording of the same
  name stays in, so a high reading means the bridge generalises *across
  wordings* rather than returning the memorised entry.
* ``sentence_before`` / ``sentence_after`` -- the pipeline R9 makes possible:
  fit on the phrase bank and read the describe sentence.  ``sentence_after``
  deletes exactly the wording that sentence itself uses, keeping the other; a
  drop to chance here means the sentence->name link needs the literal phrase
  written into the address layer.

Arms, lambda grid, control and the discipline are inherited from R10:

* R1 -- 128-bin byte histogram, L2-normalised (machinery control).
* R2 -- mean of the untrained encoder's context vectors (project side).
* R3 -- frozen sentence-transformer; upper bound only, never enters the
  capability account.

Verdict, frozen before the first reading, on R3 place ``sentence_after``:

* ``phrase_bridge_generalises_across_wordings`` iff >= 0.80 at lambda 1.0;
* ``phrase_bridge_memorised_per_wording`` iff <= 0.60 at every lambda in the
  grid (chance 0.20 for place, 0.25 for object);
* ``phrase_bridge_inconclusive`` otherwise.

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

MANIFEST = ROOT / "plans/manifests/taiji_phrase_bridge_alt_20261109.json"
REPORT = ROOT / "reports/taiji_phrase_bridge_alt_20261109.json"
R10_REPORT = ROOT / "reports/taiji_phrase_bridge_20261109.json"

RUNNER_SHA_START = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()

# Second wording per name.  Deliberately shares no characters with the entity
# name (names are the target, never input) and as little as possible with the
# primary wording, so a positive ``after`` cannot be explained by surface reuse.
ALTERNATE_DESCRIPTIONS = {
    "蓝色杯子": "盛水用的圆柱容器",
    "红色印章": "盖戳用的朱砂小砖",
    "绿色瓶子": "装油的细颈器皿",
    "黄色球": "能弹跳的圆形玩具",
    "木桌": "四条腿的家具面",
    "窗台": "玻璃边的横板",
    "左盒": "近旁装物的方匣",
    "书架": "堆卷的格板",
    "地毯": "踩踏的软织垫",
}

LAMBDA_GRID = (0.001, 0.1, 1.0, 10.0, 1000.0)
PRIMARY_LAMBDA = 1.0
CONTROL_FLOOR = 0.95
GENERALISE_FLOOR = 0.80
MEMORISED_CEILING = 0.60
COST_ESTIMATE_CPU_UPPER = 40.0

HEADER_NOTE = (
    "R3 使用外部冻结句向量，只作上界；它的读数不进能力账，也不升级/降级任何既有判定。"
    "第二种说法只在 runner 内定义，taiji/ 夹具字节未动。"
)

DEFINITION = {
    "bank": (
        "two wordings per name: the primary wording from OBJECT_DESCRIPTIONS / "
        "PLACE_DESCRIPTIONS and the alternate wording defined in this runner "
        "(ALTERNATE_DESCRIPTIONS).  Object 4->8 rows, place 5->10 rows.  Names are "
        "never input: they are the regression target and the candidate set."
    ),
    "bridge": (
        "closed-form ridge W = (X^T X + lambda I)^-1 X^T N over the slot's name "
        "embeddings; read-out x W classified by nearest name by cosine."
    ),
    "column_before": (
        "fit the full two-wording bank and read the primary rows back (in-sample).  "
        "Establishes that both wordings are representable; not a generalisation reading."
    ),
    "column_after": (
        "for each primary row delete exactly that row (2n -> 2n-1, one entry per item), "
        "refit and read it.  The other wording of the same name stays in the bank, so a "
        "high reading is cross-wording generalisation, not table lookup."
    ),
    "sentence_mode": (
        "fit the phrase bank and read the 20 describe sentences (each sentence uses the "
        "primary wording).  sentence_after deletes the primary row of the sentence's own "
        "target name and keeps the alternate: the decisive 'do we still get the name "
        "without the literal wording' reading."
    ),
    "lambda": (
        "verdict at lambda = 1.0, fixed before the first reading.  The grid "
        "{1e-3, 1e-1, 1, 1e1, 1e3} is robustness and is used only to reject the "
        "memorised claim."
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
        "machinery control: R1 primary-only in-sample accuracy (bank = one wording per "
        "name, exactly R10's bank) must be >= 0.95 on both slots at lambda 1.0; kept "
        "independent of the two-wording bank so a control failure means the closed-form "
        "path broke, not that the new bank is harder."
    ),
    "generalises": (
        "phrase_bridge_generalises_across_wordings iff R3 place sentence_after >= 0.80 "
        "at lambda 1.0."
    ),
    "memorised": (
        "phrase_bridge_memorised_per_wording iff R3 place sentence_after <= 0.60 at every "
        "lambda in the grid; otherwise phrase_bridge_inconclusive."
    ),
    "project_side": (
        "computed on R2 only (D-20261109-10): project_bridge_generalises iff R2 place "
        "sentence_after >= 0.80, project_bridge_memorised iff <= 0.60, else inconclusive."
    ),
    "r3_unavailable": "if the R3 weights are not cached the verdict is r3_unavailable.",
    "no_fit": base.CRITERIA["no_fit"],
}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def alt_sha() -> str:
    blob = json.dumps(ALTERNATE_DESCRIPTIONS, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def payload_fixture() -> dict[str, Any]:
    for name, text in ALTERNATE_DESCRIPTIONS.items():
        primary = {**OBJECT_DESCRIPTIONS, **PLACE_DESCRIPTIONS}[name]
        if text == primary or name in text:
            raise SystemExit(f"alternate wording for {name} reuses the primary or the name")
    rows = base.frozen.build_fixture()
    return {
        "names": {
            "object": list(OBJECTS),
            "place": list(PLACES),
        },
        "primary": {
            "object": [OBJECT_DESCRIPTIONS[name] for name in OBJECTS],
            "place": [PLACE_DESCRIPTIONS[name] for name in PLACES],
        },
        "alternate": {
            "object": [ALTERNATE_DESCRIPTIONS[name] for name in OBJECTS],
            "place": [ALTERNATE_DESCRIPTIONS[name] for name in PLACES],
        },
        "describe": [row for row in rows if row["variant"] == "describe"],
    }


def bank_of(slot: str, payload: dict[str, Any]) -> tuple[list[str], list[str]]:
    names = payload["names"][slot]
    texts: list[str] = []
    targets: list[str] = []
    for index, name in enumerate(names):
        texts.append(payload["primary"][slot][index])
        targets.append(name)
        texts.append(payload["alternate"][slot][index])
        targets.append(name)
    return texts, targets


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


def reading(
    fit_features: np.ndarray,
    fit_targets: list[str],
    read_features: np.ndarray,
    read_targets: list[str],
    name_matrix: np.ndarray,
    names: list[str],
    lam: float,
    delete_index: list[int] | None = None,
) -> dict[str, Any]:
    index_of = {name: position for position, name in enumerate(names)}
    target_matrix = np.stack([name_matrix[index_of[target]] for target in fit_targets])
    predicted: list[int] = []
    for row, deletion in zip(read_features, delete_index or [-1] * len(read_features), strict=True):
        if deletion < 0:
            keep = np.ones(len(fit_features), dtype=bool)
        else:
            keep = np.arange(len(fit_features)) != deletion
        weights = fit_ridge(fit_features[keep], target_matrix[keep], lam)
        predicted.append(nearest_name(row[None, :] @ weights, name_matrix)[0])
    correct = sum(
        int(left == right)
        for left, right in zip(
            predicted, [index_of[target] for target in read_targets], strict=True
        )
    )
    return {
        "accuracy": round(correct / len(read_targets), 6),
        "correct": correct,
        "of": len(read_targets),
        "bank_size": len(fit_features) - (1 if delete_index else 0),
        "predicted": [names[index] for index in predicted],
    }


def slot_reading(
    bank_matrix: np.ndarray,
    sentence_matrix: np.ndarray,
    name_matrix: np.ndarray,
    names: list[str],
    bank_targets: list[str],
    sentence_targets: list[str],
) -> dict[str, Any]:
    count = len(names)
    primary_rows = [2 * index for index in range(count)]
    primary_matrix = bank_matrix[primary_rows]

    control = reading(
        primary_matrix, names, primary_matrix, names, name_matrix, names, PRIMARY_LAMBDA
    )

    grid: dict[str, Any] = {}
    for lam in LAMBDA_GRID:
        before = reading(bank_matrix, bank_targets, primary_matrix, names, name_matrix, names, lam)
        after = reading(
            bank_matrix,
            bank_targets,
            primary_matrix,
            names,
            name_matrix,
            names,
            lam,
            delete_index=primary_rows,
        )
        sentence_before = reading(
            bank_matrix, bank_targets, sentence_matrix, sentence_targets, name_matrix, names, lam
        )
        sentence_after = reading(
            bank_matrix,
            bank_targets,
            sentence_matrix,
            sentence_targets,
            name_matrix,
            names,
            lam,
            delete_index=[primary_rows[names.index(target)] for target in sentence_targets],
        )
        grid[f"{lam:g}"] = {
            "before": before,
            "after": after,
            "sentence_before": sentence_before,
            "sentence_after": sentence_after,
        }
    return {
        "control": control,
        "control_ok": control["accuracy"] >= CONTROL_FLOOR,
        "bank_size_before": len(bank_targets),
        "bank_size_after": len(bank_targets) - 1,
        "deleted_per_item": 1,
        "grid": grid,
        "primary": grid[f"{PRIMARY_LAMBDA:g}"],
    }


def arm_reading(
    arm: str, payload: dict[str, Any], reference: LivingCoreReference
) -> dict[str, Any]:
    object_texts, object_targets = bank_of("object", payload)
    place_texts, place_targets = bank_of("place", payload)
    sentences = [row["text"] for row in payload["describe"]]
    texts = object_texts + place_texts + list(OBJECTS) + list(PLACES) + sentences
    matrix = embed_arm(arm, texts, reference)
    if matrix is None:
        return {"verdict": "unavailable"}
    cursor = 0
    object_bank = matrix[cursor : cursor + len(object_texts)]
    cursor += len(object_texts)
    place_bank = matrix[cursor : cursor + len(place_texts)]
    cursor += len(place_texts)
    object_names = matrix[cursor : cursor + len(OBJECTS)]
    cursor += len(OBJECTS)
    place_names = matrix[cursor : cursor + len(PLACES)]
    cursor += len(PLACES)
    sentence_matrix = matrix[cursor:]
    return {
        "dimensions": int(matrix.shape[1]),
        "object": slot_reading(
            object_bank,
            sentence_matrix,
            object_names,
            list(OBJECTS),
            object_targets,
            [row["object"] for row in payload["describe"]],
        ),
        "place": slot_reading(
            place_bank,
            sentence_matrix,
            place_names,
            list(PLACES),
            place_targets,
            [row["place"] for row in payload["describe"]],
        ),
    }


def classify(score: float) -> str:
    if score >= GENERALISE_FLOOR:
        return "generalises"
    if score <= MEMORISED_CEILING:
        return "memorised"
    return "inconclusive"


def r10_reference() -> dict[str, Any]:
    if not R10_REPORT.exists():
        return {"status": "R10 report missing"}
    report = json.loads(R10_REPORT.read_text(encoding="utf-8"))
    reading_arm = report["arms"].get("R3", {})
    if reading_arm.get("verdict") == "unavailable":
        return {"status": "R10 R3 unavailable"}
    return {
        "status": "cited from reports/taiji_phrase_bridge_20261109.json",
        "note": "1:1 词表，删掉该概念唯一训练目标；0.00 是构造性零（D-20261109-11）。",
        "R10_verdict": report["verdict"],
        "R10_bank": {"object": 4, "place": 5},
        "R10_R3_place": {
            "before": reading_arm["place"]["primary"]["before"]["accuracy"],
            "after": reading_arm["place"]["primary"]["after"]["accuracy"],
            "sentence_after": reading_arm["place"]["sentence_mode"]["accuracy"],
        },
    }


WRITTEN_BRIDGE_SKETCH = {
    "status": "design only; not implemented in this round",
    "applies_if": "the after column falls back to chance (memorised verdict)",
    "key": (
        "the phrase's set of 2-byte tokens, exactly as sc_sfk features / "
        "content_feature already build them (order-free set, no positions)."
    ),
    "value": "(slot, name) as a string pair, e.g. ('place', '书架').",
    "write": (
        "one record per (phrase, slot+name) observation; same key + same value merges "
        "(count+1), same key + different value replaces (old value into history), a "
        "batch that carries two different values for one key is rejected whole."
    ),
    "read": (
        "sentence -> locate the phrase (R9 localisation) -> phrase token set -> table "
        "read; answer only when exactly one address anchors the set, otherwise "
        "no_match / ambiguous and the answer is withheld."
    ),
    "size": "9 rows (4 object + 5 place), 0 training, 0 gradient.",
    "why_minimal": (
        "it writes only the mapping the closed-form bridge cannot generalise; it adds no "
        "parameter, no threshold, and reuses the existing write/read discipline."
    ),
}


def phase_freeze() -> dict[str, Any]:
    payload = payload_fixture()
    manifest = {
        "schema": "taiji-phrase-bridge-alt-v1",
        "date": "2026-11-09",
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runner_sha256": RUNNER_SHA_START,
        "fixture_sha256": base.fixture_hashes(),
        "fixture_files": list(base.frozen.FIXTURE_FILES),
        "alt_phrasing_sha256": alt_sha(),
        "alt_phrasing_source": "ALTERNATE_DESCRIPTIONS constant in this runner",
        "question": (
            "R10 的 0.00 是构造性零：给每个名字第二种说法后，删掉句子自用的那种说法，"
            "桥还能不能读出名字（真泛化），还是只有背过的词条？"
        ),
        "counts": {
            "describe_sentences": len(payload["describe"]),
            "phrasings_per_name": 2,
            "bank_before": {"object": 8, "place": 10},
            "bank_after": {"object": 7, "place": 9},
            "deleted_per_held_out": 1,
            "control_bank": {"object": 4, "place": 5},
            "texts_per_arm": 2 * 9 + 9 + 20,
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
    if manifest["alt_phrasing_sha256"] != alt_sha():
        raise SystemExit("alternate wordings changed after the freeze; re-freeze")

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

    control_ok = arms["R1"]["object"]["control_ok"] and arms["R1"]["place"]["control_ok"]
    upper = arms.get("R3", {})
    if not control_ok:
        diagnosis = "control_failed"
    elif upper.get("verdict") == "unavailable":
        diagnosis = "r3_unavailable"
    else:
        place = upper["place"]
        primary = place["primary"]["sentence_after"]["accuracy"]
        worst = max(place["grid"][key]["sentence_after"]["accuracy"] for key in place["grid"])
        if primary >= GENERALISE_FLOOR:
            diagnosis = "phrase_bridge_generalises_across_wordings"
        elif worst <= MEMORISED_CEILING:
            diagnosis = "phrase_bridge_memorised_per_wording"
        else:
            diagnosis = "phrase_bridge_inconclusive"

    r2 = arms.get("R2", {})
    if r2.get("verdict") == "unavailable":
        project_line = "project_bridge_unavailable"
    else:
        project_line = "project_bridge_" + classify(
            r2["place"]["primary"]["sentence_after"]["accuracy"]
        )

    report = {
        "schema": "taiji-phrase-bridge-alt-v1",
        "date": "2026-11-09",
        "header_note": HEADER_NOTE,
        "runner_sha256": RUNNER_SHA_START,
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "fixture_sha256": manifest["fixture_sha256"],
        "alt_phrasing_sha256": manifest["alt_phrasing_sha256"],
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
                    "sentence_before": classify(
                        upper[slot]["primary"]["sentence_before"]["accuracy"]
                    ),
                    "sentence_after": classify(
                        upper[slot]["primary"]["sentence_after"]["accuracy"]
                    ),
                }
                for slot in ("object", "place")
            }
            if upper.get("verdict") != "unavailable"
            else {"status": "r3_unavailable"}
        ),
        "contrast": {
            "r10_reference": r10_reference(),
            "r10b_R3": (
                {
                    slot: {
                        "before": upper[slot]["primary"]["before"]["accuracy"],
                        "after": upper[slot]["primary"]["after"]["accuracy"],
                        "sentence_before": upper[slot]["primary"]["sentence_before"]["accuracy"],
                        "sentence_after": upper[slot]["primary"]["sentence_after"]["accuracy"],
                        "sentence_after_worst_lambda": min(
                            upper[slot]["grid"][key]["sentence_after"]["accuracy"]
                            for key in upper[slot]["grid"]
                        ),
                        "chance": DEFINITION["chance"][slot],
                    }
                    for slot in ("object", "place")
                }
                if upper.get("verdict") != "unavailable"
                else {"status": "unavailable"}
            ),
        },
        "written_bridge_sketch": WRITTEN_BRIDGE_SKETCH,
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
        "contrast": report["contrast"]["r10b_R3"],
        "cost": report["cost"],
    }
    for arm in ("R1", "R2", "R3"):
        if arm in arms and arms[arm].get("verdict") != "unavailable":
            summary[arm] = {
                slot: {
                    "before": arms[arm][slot]["primary"]["before"]["accuracy"],
                    "after": arms[arm][slot]["primary"]["after"]["accuracy"],
                    "sentence_before": arms[arm][slot]["primary"]["sentence_before"]["accuracy"],
                    "sentence_after": arms[arm][slot]["primary"]["sentence_after"]["accuracy"],
                    "sentence_after_worst": min(
                        arms[arm][slot]["grid"][key]["sentence_after"]["accuracy"]
                        for key in arms[arm][slot]["grid"]
                    ),
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
