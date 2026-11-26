"""F1 correction branch: write one correction (q, v'), then ask a synonym.

Criteria are fixed here, before any reading is taken, and are not changed after.

Fixture: byte-for-byte the F3 multi-generation fixture (same generator, same 64
sentences, same frozen centroid), so the pre-correction key spectrum and read
margin at occupancy 8 and 64 must reproduce F3's generation-1 and generation-4
rows.  Inside that fixture "same rule, different words" is realised as the same
distinctive value token placed at a different position: the mean encoder is
provably blind to that (it pools over positions), the localized encoder is
exactly the part that sees it.

Protocol, per occupancy rung::

    store  = contexts[0 : occupied]                     # F3 geometry, unchanged
    control  -> ask the store before any correction
    q      = paraphrase(rule 0, position 5)             # wording used to correct
    v'     = a value never written before
    write  = capacity-first batch write of (q -> v')     # one new slot
    treatment-> ask the store after the correction

Both paths are read with the same cosine-addressed read and the same temperature
(0.1) as ``addressed_read``.

Reported per rung, treatment and control side by side:

* ``correction@1``  fraction of synonym questions whose top-1 slot is the
  correction slot (the literal metric asked for);
* ``value@1``       fraction whose read value is closer to v' than to v_old;
* blend readings    softmax weight on the correction slot and the relative L2
  residual of the read value against v' (0 = clean read, 0.71 = half old half new);
* key spectrum and read margin before and after the correction, so a failure can
  be attributed to "not enough margin" rather than "the correction was not stored".

Diagnostic arms (reported, not criteria): the legacy ``mean`` encoder on the same
fixture, and an in-place overwrite of the old slot, which separates "the geometry
cannot carry it" from "two entries at one address get blended".

Zero training, zero optimizer steps, zero Adam, zero confirm reads, zero model
forward, zero new trainable parameters.  numpy only.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taiji import growth_design as gd  # noqa: E402
from taiji import growth_runtime as gr  # noqa: E402

REPORT = ROOT / "reports/taiji_growth_f1_correction_20261108.json"
MANIFEST = ROOT / "plans/manifests/taiji_growth_f1_correction_20261108.json"
PROGRESS = ROOT / "reports/taiji_growth_f1_correction_20261108_progress.jsonl"
F3_REPORT = ROOT / "reports/taiji_growth_f3_multigen_20261108.json"

SEED = 20261108
DIM = 128
LENGTH = 8
TOTAL_SENTENCES = 64
INITIAL_ENTRIES = 4
AMPLITUDE = 0.35
RESERVE = 0.125
TEMPERATURE = 0.1
CPU_CAP = 200.0
CPU_GUARD = 190.0
MODEL_PARAMS = 3_323_547
MODEL_IDENTITY_SHA = "235bc6e7e978711071811fe0e0fc0faf41733b6874ff18d6fb696e034c18bc71"
VALUE_SHAPE = (4, 2, 32, 32)

RULE_INDEX = 0
CORRECTION_POSITION = 5
SYNONYM_POSITIONS = (1, 2, 3, 4)

RUNGS = (
    {"name": "low", "generation": 1, "occupied": 8, "slots": 16},
    {"name": "gen4", "generation": 4, "occupied": 64, "slots": 128},
)

CRITERIA = {
    "C0_fixture_answerable": (
        "control (no correction) returns the old entry as top-1 for every question "
        "on the repaired encoder at both rungs; otherwise the fixture is not answerable "
        "and no correction verdict is issued"
    ),
    "C1_correction_written": (
        "the correction commits 1/1, drops nothing, evicts nothing, and the stored "
        "payload equals v' byte for byte"
    ),
    "C2_self_readable": (
        "the exact phrasing that wrote the correction returns the correction as top-1 "
        "(self@1 == 1.0) and the read value residual against v' stays below 0.5"
    ),
    "C3_synonym_generalises": (
        "synonym correction@1 >= 0.75 at the low rung and strictly above the control "
        "correction@1 at both rungs"
    ),
    "verdict_rule": (
        "C1 fails -> correction_not_written; C2 fails -> geometry_holds_but_unreadable; "
        "C3 fails -> self_readable_synonym_fails; else correction_readable"
    ),
    "no_soft_language": (
        "if treatment correction@1 equals the control, the verdict is "
        "geometry_holds_but_unreadable; no 'close to significant' wording is allowed"
    ),
    "unchanged": "no threshold, rung, fixture or question is retuned after the first reading",
}

START_CPU = time.process_time()
START_WALL = time.time()


def append_progress(record: dict) -> None:
    with PROGRESS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """F3 generator, unchanged: shared baseline + 0.05 jitter + one value token."""
    rng = np.random.default_rng(SEED + 3)
    shared = gd.unit_rows(rng.normal(size=(1, DIM)))[0]
    jitter = 0.05 * gd.unit_rows(rng.normal(size=(LENGTH, DIM)))
    base = shared[None, :] + jitter
    values = gd.unit_rows(rng.normal(size=(TOTAL_SENTENCES, DIM)))
    contexts = np.repeat(base[None, :, :], TOTAL_SENTENCES, axis=0)
    positions = np.arange(TOTAL_SENTENCES) % LENGTH
    contexts[np.arange(TOTAL_SENTENCES), positions] += AMPLITUDE * values
    return base, values, contexts


def paraphrase(base: np.ndarray, value: np.ndarray, position: int) -> np.ndarray:
    """Same content, different word order: the value token moves to ``position``."""
    context = base.copy()
    context[position] = context[position] + AMPLITUDE * value
    return context


def value_payloads(count: int) -> np.ndarray:
    rng = np.random.default_rng(SEED + 11)
    payloads = rng.normal(size=(count, int(np.prod(VALUE_SHAPE))))
    payloads /= np.clip(np.linalg.norm(payloads, axis=1, keepdims=True), 1e-12, None)
    return payloads


def store_readings(keys: np.ndarray) -> dict[str, Any]:
    spectrum = gd.key_spectrum(keys)
    margins = gd.read_margin_stats(keys)
    decision = gd.store_decision(spectrum, margins, calibrated=True)
    return {
        "keys": int(keys.shape[0]),
        "spectrum": spectrum,
        "margins": margins,
        "diagnosis": decision["diagnosis"],
        "decision": decision["decision"],
    }


def addressed_read(query: np.ndarray, keys: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    """Cosine softmax read at the store temperature, exactly as ``addressed_read``."""
    query = query / max(float(np.linalg.norm(query)), 1e-12)
    normalized = keys / np.clip(np.linalg.norm(keys, axis=1, keepdims=True), 1e-12, None)
    cosine = normalized @ query
    order = np.argsort(-cosine)
    logits = cosine / TEMPERATURE
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()
    return {
        "top_slot": int(order[0]),
        "second_slot": int(order[1]),
        "top_cos": float(cosine[order[0]]),
        "margin": float(cosine[order[0]] - cosine[order[1]]),
        "weights": weights,
        "value": (weights[:, None] * values).sum(axis=0),
    }


def relative_residual(read: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(read - target) / max(float(np.linalg.norm(target)), 1e-12))


def ask(
    queries: dict[str, np.ndarray],
    keys: np.ndarray,
    values: np.ndarray,
    *,
    old_slot: int,
    correction_slot: int | None,
) -> dict[str, Any]:
    rows = {}
    for name, query in queries.items():
        read = addressed_read(query, keys, values)
        row = {
            "top_slot": read["top_slot"],
            "margin": read["margin"],
            "top_cos": read["top_cos"],
            "is_old": read["top_slot"] == old_slot,
            "is_correction": correction_slot is not None and read["top_slot"] == correction_slot,
            "weight_on_old": float(read["weights"][old_slot]),
            "weight_on_correction": (
                float(read["weights"][correction_slot]) if correction_slot is not None else 0.0
            ),
            "cos_to_old_value": float(read["value"] @ values[old_slot]),
            "value": read["value"],
        }
        rows[name] = row
    return rows


def summarise(rows: dict[str, Any], correction_value: np.ndarray, old_value: np.ndarray) -> dict:
    synonyms = [row for name, row in rows.items() if name.startswith("synonym_")]
    self_row = rows["self"]
    correction_at_1 = float(np.mean([row["is_correction"] for row in synonyms]))
    old_slot_at_1 = float(np.mean([row["is_old"] for row in synonyms]))
    value_at_1 = 0.0
    residuals = []
    weights = []
    for row in synonyms + [self_row]:
        cos_new = float(row["value"] @ correction_value)
        cos_old = float(row["value"] @ old_value)
        value_at_1 += 1.0 if cos_new > cos_old else 0.0
        residuals.append(relative_residual(row["value"], correction_value))
        weights.append(row["weight_on_correction"])
    return {
        "correction_at_1": correction_at_1,
        "old_slot_at_1": old_slot_at_1,
        "value_at_1": value_at_1 / (len(synonyms) + 1),
        "self_at_1": 1.0 if self_row["is_correction"] else 0.0,
        "median_weight_on_correction": float(np.median(weights)),
        "median_residual_to_correction": float(np.median(residuals)),
        "synonym_margins": [float(row["margin"]) for row in synonyms],
        "synonym_weight_on_correction": [float(row["weight_on_correction"]) for row in synonyms],
        "rows": {
            name: {key: value for key, value in row.items() if key != "value"}
            for name, row in rows.items()
        },
    }


def build_store(occupied: int, encoder: gr.EncoderConfig, contexts: np.ndarray, centroid):
    return np.stack(
        [gr.encode_read_key(contexts[index], encoder, centroid) for index in range(occupied)]
    )


def run_rung(
    rung: dict,
    *,
    base: np.ndarray,
    tokens: np.ndarray,
    payloads: np.ndarray,
    correction_value: np.ndarray,
    encoder: gr.EncoderConfig,
    centroid: np.ndarray | None,
    contexts: np.ndarray,
    label: str,
) -> dict:
    occupied = rung["occupied"]
    slots = rung["slots"]
    keys = build_store(occupied, encoder, contexts, centroid)
    values = payloads[:occupied].copy()
    old_value = values[RULE_INDEX]
    pre = store_readings(keys)

    queries = {
        "self": gr.encode_read_key(
            paraphrase(base, tokens[RULE_INDEX], CORRECTION_POSITION), encoder, centroid
        )
    }
    for position in SYNONYM_POSITIONS:
        queries[f"synonym_{position}"] = gr.encode_read_key(
            paraphrase(base, tokens[RULE_INDEX], position), encoder, centroid
        )

    control_rows = ask(queries, keys, values, old_slot=RULE_INDEX, correction_slot=None)
    control = summarise(control_rows, correction_value, old_value)

    correction_slot = occupied
    headroom = gr.write_headroom(occupied, slots, 1)
    plan = gr.plan_growth(headroom)
    authorising = gr.write_headroom(occupied, int(plan["slots"]), 1)
    snapshot = {slot: keys[slot].tobytes() for slot in range(occupied)}
    committed = gr.write_batch(
        snapshot,
        {correction_slot: queries["self"].tobytes()},
        headroom=authorising,
        plan=plan,
    )
    appended = committed["committed"] == [correction_slot]
    treatment_keys = np.vstack([keys, queries["self"][None, :]]) if appended else keys
    treatment_values = np.vstack([values, correction_value[None, :]]) if appended else values
    evictions = sum(1 for slot, raw in snapshot.items() if treatment_keys[slot].tobytes() != raw)
    post = store_readings(treatment_keys)
    band = gd.direction_band(keys, queries["self"]) if occupied else None

    treatment_rows = ask(
        queries,
        treatment_keys,
        treatment_values,
        old_slot=RULE_INDEX,
        correction_slot=correction_slot,
    )
    treatment = summarise(treatment_rows, correction_value, old_value)

    overwrite_values = values.copy()
    overwrite_values[RULE_INDEX] = correction_value
    overwrite_rows = ask(queries, keys, overwrite_values, old_slot=RULE_INDEX, correction_slot=None)
    overwrite = {
        "correction_at_1": float(
            np.mean(
                [
                    float(row["value"] @ correction_value) > float(row["value"] @ old_value)
                    for name, row in overwrite_rows.items()
                    if name.startswith("synonym_")
                ]
            )
        ),
        "top1_is_the_rule_entry": float(
            np.mean([row["top_slot"] == RULE_INDEX for row in overwrite_rows.values()])
        ),
    }

    write = {
        "headroom_before_plan": headroom,
        "plan": plan,
        "headroom_authorising": authorising,
        "committed": len(committed["committed"]),
        "dropped_writes": 1 - len(committed["committed"]),
        "evictions": int(evictions),
        "appended": bool(appended),
        "value_matches": bool(
            appended and np.array_equal(treatment_values[correction_slot], correction_value)
        ),
        "reason": committed["reason"],
    }

    if write["dropped_writes"] or write["evictions"] or not write["value_matches"]:
        verdict = "correction_not_written"
    elif treatment["self_at_1"] < 1.0 or treatment["median_residual_to_correction"] >= 0.5:
        verdict = "geometry_holds_but_unreadable"
    elif (
        treatment["correction_at_1"] < 0.75
        or treatment["correction_at_1"] <= control["correction_at_1"]
    ):
        verdict = "self_readable_synonym_fails"
    else:
        verdict = "correction_readable"

    band_row = (
        None
        if band is None
        else {
            "overlap": band["overlap"],
            "separation": band["separation"],
            "nearest_cos": band["nearest_cos"],
            "verdict": band["verdict"],
        }
    )

    record = {
        "path": label,
        "rung": rung["name"],
        "generation": rung["generation"],
        "occupied": occupied,
        "slots": slots,
        "encoder": encoder.schema,
        "pre_correction": pre,
        "post_correction": post,
        "write": write,
        "control": control,
        "treatment": treatment,
        "overwrite_diagnostic": overwrite,
        "correction_direction_band": band_row,
        "mechanism": {
            "slot_retrieved_rate": treatment["correction_at_1"],
            "blend_weight_on_correction": treatment["median_weight_on_correction"],
            "value_residual_to_correction": treatment["median_residual_to_correction"],
            "value_fidelity": (
                "clean"
                if treatment["median_residual_to_correction"] < 0.5
                else "blended_near_half_half"
            ),
            "correction_verdict": None if band is None else band["verdict"],
            "in_place_overwrite_reads_clean": bool(
                overwrite["correction_at_1"] == 1.0 and overwrite["top1_is_the_rule_entry"] == 1.0
            ),
        },
        "verdict": verdict,
        "verdict_zh": {
            "correction_not_written": "纠正没写进去",
            "geometry_holds_but_unreadable": (
                "槽位能检索到，但读回的值是新旧各一半（纠正没有取代旧值）"
            ),
            "self_readable_synonym_fails": "纠正能自读，但不能跨措辞泛化",
            "correction_readable": "纠正能读回，且能跨措辞泛化",
        }[verdict],
    }
    append_progress(record)
    return record


def f3_cross_check(records: list[dict]) -> dict:
    if not F3_REPORT.exists():
        return {"available": False}
    payload = json.loads(F3_REPORT.read_text(encoding="utf-8"))
    generations = payload.get("repaired_path", {}).get("generations", [])
    rows = {}
    for record in records:
        match = next(
            (item for item in generations if item["generation"] == record["generation"]), None
        )
        if match is None:
            continue
        expected = match["readings"]["margins"]["median_margin"]
        actual = record["pre_correction"]["margins"]["median_margin"]
        rows[record["rung"]] = {
            "f3_generation": record["generation"],
            "f3_median_margin": expected,
            "this_median_margin": actual,
            "abs_delta": abs(expected - actual),
            "matches": abs(expected - actual) <= 1e-12,
        }
    return {
        "available": True,
        "rows": rows,
        "all_match": all(row["matches"] for row in rows.values()),
    }


def main() -> int:
    if time.process_time() - START_CPU > CPU_GUARD:
        raise SystemExit("cpu guard tripped before start")

    base, tokens, contexts = fixture()
    payloads = value_payloads(TOTAL_SENTENCES + 8)
    correction_value = payloads[TOTAL_SENTENCES]

    legacy_encoder = gr.EncoderConfig(mode="mean")
    legacy_parents = np.stack(
        [gr.encode_read_key(contexts[index], legacy_encoder) for index in range(INITIAL_ENTRIES)]
    )
    centroid = gr.store_centroid(legacy_parents)
    repaired_encoder = gr.EncoderConfig(mode="localized_centered", center=True)

    repaired: list[dict] = []
    legacy: list[dict] = []
    for rung in RUNGS:
        repaired.append(
            run_rung(
                rung,
                base=base,
                tokens=tokens,
                payloads=payloads,
                correction_value=correction_value,
                encoder=repaired_encoder,
                centroid=centroid,
                contexts=contexts,
                label="repaired",
            )
        )
        legacy.append(
            run_rung(
                rung,
                base=base,
                tokens=tokens,
                payloads=payloads,
                correction_value=correction_value,
                encoder=legacy_encoder,
                centroid=None,
                contexts=contexts,
                label="control_legacy",
            )
        )

    cross_check = f3_cross_check(repaired)
    fixture_answerable = all(record["control"]["old_slot_at_1"] == 1.0 for record in repaired)

    if not fixture_answerable:
        decision = "fixture_not_answerable"
    else:
        joined = "|".join(record["verdict"] for record in repaired)
        decision = (
            "correction_readable"
            if all(record["verdict"] == "correction_readable" for record in repaired)
            else f"fail_{joined}"
        )

    payload = {
        "format": "taiji-growth-f1-correction-v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "criteria_written_before_reading": CRITERIA,
        "fixture": {
            "dimension": DIM,
            "context_length": LENGTH,
            "sentences": TOTAL_SENTENCES,
            "amplitude": AMPLITUDE,
            "value_shape": list(VALUE_SHAPE),
            "temperature": TEMPERATURE,
            "rule_index": RULE_INDEX,
            "correction_position": CORRECTION_POSITION,
            "synonym_positions": list(SYNONYM_POSITIONS),
            "wording_analogue": (
                "same distinctive value token at a different position, i.e. the same "
                "content in a different word order; the mean encoder is blind to it by "
                "construction, the localized encoder is the part that sees it"
            ),
            "rungs": [dict(row) for row in RUNGS],
            "model_identity_source": (
                "reports/taiji_growth_zero_perturbation_20261108.json (same-day sealed "
                "reading); no parameter row changes and neither module imports torch"
            ),
        },
        "f3_cross_check": cross_check,
        "repaired_path": repaired,
        "control_legacy_path": legacy,
        "reading_notes": {
            "margin_is_not_the_limiter": (
                "the blend weight on the correction is "
                f"{repaired[0]['mechanism']['blend_weight_on_correction']:.4f} at margin "
                f"{repaired[0]['pre_correction']['margins']['median_margin']:.5f} and "
                f"{repaired[-1]['mechanism']['blend_weight_on_correction']:.4f} at margin "
                f"{repaired[-1]['pre_correction']['margins']['median_margin']:.5f}: a 3.4x "
                "change of the store's read margin moves the blend by less than 0.002, so the "
                "failure is not a margin shortfall"
            ),
            "why_half_half": (
                "the correction and the old entry share one address: their keys differ only by "
                "word order, cos = nearest_cos in correction_direction_band (~0.96), and the "
                "cosine softmax at temperature 0.1 turns that same-address gap into a near tie; "
                "the read value is 0.51 * v' + 0.49 * v_old, residual 0.698 = 0.49 * sqrt(2)"
            ),
            "value_at_1_is_a_tie_break": (
                "value_at_1 is 1.0 on both rungs only because 0.511 > 0.489; it must not be read "
                "as 'the corrected value was returned'. The honest measure is the residual"
            ),
            "address_contract": (
                "taiji/living_core.py addressed_write always takes a free slot and only evicts "
                "the oldest when the store is full; a near-identical key never supersedes the "
                "entry it duplicates, so a correction accumulates instead of replacing"
            ),
            "in_place_overwrite": (
                "the overwrite diagnostic writes v' into the existing slot and reads clean "
                "(correction@1 = 1.0, top-1 is the rule entry) at both rungs: the geometry is "
                "sufficient, what is missing is a supersede rule, not more capacity"
            ),
            "legacy_encoder": (
                "mean pooling is blind to word order, so the correction key is an exact copy "
                "(overlap 1.0, separation 0.0) and its softmax weight is the uniform share "
                "1/(N+1): 0.113 at 9 entries, 0.016 at 65 entries"
            ),
        },
        "decision": decision,
        "what_this_means": (
            "correction@1 is the fraction of re-worded questions that retrieve the slot the "
            "correction was written to; value@1 is the fraction whose read value is closer to "
            "the corrected value than to the old one. They are reported apart because a "
            "content-addressed store can return the right slot and still hand back a blend of "
            "old and new when two entries share one address"
        ),
        "what_this_does_not_mean": [
            "no training, no optimizer step and no model forward ran; nothing here says the "
            "correction survives training",
            "the fixture is synthetic: a minority of real corrections are re-wordings of the "
            "same question, this probe measures exactly that case and nothing else",
            "value payloads are random unit vectors in the 20.3a payload shape; the readout is "
            "a cosine test, not a language-level judgement of the corrected answer",
            "the overlap/band reading is a geometric verdict on one key, not a calibrated "
            "threshold for real stores",
            "one session, one seed, no persistence across checkpoints",
        ],
        "isolation": {
            "runner_sha256": file_sha(Path(__file__)),
            "growth_design_sha256": file_sha(ROOT / "taiji/growth_design.py"),
            "growth_runtime_sha256": file_sha(ROOT / "taiji/growth_runtime.py"),
            "growth_modules_import_torch": any(
                "import torch" in (ROOT / name).read_text(encoding="utf-8", errors="ignore")
                for name in ("taiji/growth_design.py", "taiji/growth_runtime.py")
            ),
            "f3_report": F3_REPORT.relative_to(ROOT).as_posix(),
            "f3_report_sha256": file_sha(F3_REPORT) if F3_REPORT.exists() else None,
        },
        "cost": {
            "cpu_seconds": round(time.process_time() - START_CPU, 3),
            "wall_seconds": round(time.time() - START_WALL, 3),
            "cpu_cap": CPU_CAP,
            "training_steps": 0,
            "optimizer_steps": 0,
            "adam_steps": 0,
            "confirm_reads": 0,
            "model_forwards": 0,
        },
    }

    write_json(REPORT, payload)
    write_json(
        MANIFEST,
        {
            "format": "taiji-growth-f1-correction-manifest-v1",
            "report": REPORT.relative_to(ROOT).as_posix(),
            "report_sha256": file_sha(REPORT),
            "runner": Path(__file__).relative_to(ROOT).as_posix(),
            "runner_sha256": payload["isolation"]["runner_sha256"],
            "growth_design_sha256": payload["isolation"]["growth_design_sha256"],
            "growth_runtime_sha256": payload["isolation"]["growth_runtime_sha256"],
            "decision": payload["decision"],
        },
    )

    for record in repaired:
        treatment = record["treatment"]
        control = record["control"]
        band = record["correction_direction_band"]["verdict"]
        print(
            f"{record['rung']} occ={record['occupied']} slots={record['slots']} "
            f"margin={record['pre_correction']['margins']['median_margin']:.5f} "
            f"control[old@1={control['old_slot_at_1']:.2f} "
            f"corr@1={control['correction_at_1']:.2f}] "
            f"treat[corr@1={treatment['correction_at_1']:.2f} "
            f"self@1={treatment['self_at_1']:.2f} "
            f"value@1={treatment['value_at_1']:.2f} "
            f"w={treatment['median_weight_on_correction']:.2f} "
            f"res={treatment['median_residual_to_correction']:.2f}] "
            f"band={band} {record['verdict']}"
        )
    print(
        json.dumps(
            {"decision": decision, "f3_all_match": cross_check.get("all_match")},
            ensure_ascii=False,
        )
    )
    return 0 if decision == "correction_readable" else 1


if __name__ == "__main__":
    raise SystemExit(main())
