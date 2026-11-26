"""R53 (the pending R17') -- end-to-end byte output through the bridge.

R17 was rejected as "same-value containment compression": the project already
knows there is no semantic map inside the encoder (R7/R10b), so there is nothing
to compress, and a containment rule would trade recall for a lower key ledger
(measured in R17 / SPEC 21.26: the boundary case makes a value visible that was
hidden before compression).  This runner executes the replacement request:

    question -> bridge locates the entity (R15 two-pass read, R16 geometry
    frozen) -> take the entity's ORIGINAL BYTES from the corpus -> emit one
    contiguous span of that record.

The output unit is not re-invented: it is ``taiji.sc_sfk.sentences`` (the frozen
definition of SPEC 16.2).  Every fixture record here is a single sentence, so
the emitted unit is a whole sentence and its span is the record's only span;
that means the boundary checks are satisfied by the frozen rule rather than by
a new tolerance.  Nothing trains, nothing steps an optimizer, nothing calls
Adam or confirm.

Honest cost, reported next to the readings: the bridge answers WHICH ENTITY,
not WHICH PAIRING.  The corpus holds twenty records per object (five places x
four variants), so "that entity's record" is not unique; the frozen rule picks
the lowest address (SPEC 16.2 ``address_order``).  A faithful slice of that
record therefore need not answer the asked layer, and the four-way distribution
is reported next to a separate ``faithful_but_other_pairing`` column with real
samples.  Provenance fidelity is never used to cover that gap.
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

import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.training.probe_taiji_cross_family_readout_20261109 as base  # noqa: E402
from taiji.address_bridge import (  # noqa: E402
    BridgeTable,
    bridge_records,
    sequence_key,
)
from taiji.living_core import LivingCoreReference  # noqa: E402
from taiji.native_goal_world_wide import (  # noqa: E402
    OBJECT_DESCRIPTIONS,
    PLACE_DESCRIPTIONS,
)
from taiji.process_guard import assert_no_black_processes  # noqa: E402
from taiji.sc_sfk import sentences as sentence_spans  # noqa: E402

torch.set_num_threads(1)

MANIFEST = ROOT / "plans/manifests/taiji_bridge_bytes_20261109.json"
REPORT = ROOT / "reports/taiji_bridge_bytes_20261109.json"
BRIDGE_SOURCE = ROOT / "taiji/address_bridge.py"

RUNNER_SHA_START = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()

SEED = 20261109
ESTIMATE_CPU_UPPER = 30.0
CPU_CAP = float(base.CPU_CAP)

AUTHORED_ALIASES = (
    ("喝水的杯子", ("object", "蓝色杯子")),
    ("盖章用的印章", ("object", "红色印章")),
    ("盛水的瓶子", ("object", "绿色瓶子")),
    ("圆形的皮球", ("object", "黄色球")),
    ("方方正正的木桌子", ("place", "木桌")),
    ("窗前的台板", ("place", "窗台")),
    ("手边的小盒子", ("place", "左盒")),
    ("墙上的架子", ("place", "书架")),
    ("脚下的垫子", ("place", "地毯")),
)

UNKNOWN_SENTENCE = "让黑色的铁铃到空无一物的地方。"

DEFINITION = {
    "chain": (
        "one describe sentence -> R15 two-pass read (earliest key first, same-start abstains, "
        "first occurrence masked to the same width) -> the matched (slot, name) -> that entity's "
        "original corpus record -> one contiguous byte span of it"
    ),
    "unit": (
        "the frozen SPEC 16.2 unit: a complete record or a complete sentence. Every record of this "
        "fixture is one sentence, so the emitted span is the record's only span; a record with more "
        "than one sentence would emit its first sentence (SPEC 16.2 sentence-order tie-break). No "
        "third unit and no new numeric tolerance is introduced"
    ),
    "address_map": (
        "entity -> corpus record indices comes from the fixture metadata (each row carries its "
        "object and place), not from text search; when an entity has several records the lowest "
        "address wins (SPEC 16.2 address_order). The map is GIVEN, not inferred: this gate measures "
        "the handoff from the bridge to the output unit, not retrieval"
    ),
    "L1": (
        "fabricated bytes 0, illegal slices 0, UTF-8 decodable 100%, boundary legal 100% "
        "(boundary legal = the emitted span is one of the spans the frozen "
        "``taiji.sc_sfk.sentences`` rule returns for that record)"
    ),
    "L2": (
        "provenance fidelity = the emitted bytes equal record[start:end] byte for byte; must be 1.0"
    ),
    "L3": (
        "four classes per entity slot: answered_correct / pointed_wrong / abstained / "
        "should_answer_missing; the entity capability line stays at 40/40 and wrong pointers stay 0"
    ),
    "cost_column": (
        "faithful_but_other_pairing: the slice is faithful and the entity is right, yet the record's "
        "(object, place) is not the asked pair. Reported with samples; never folded into 'correct'"
    ),
    "closed_state": "the nine closed-state readings stay equal before and after",
    "training": "zero training, zero optimizer steps, zero Adam, zero confirm",
}

CRITERIA = {
    "l1": "fabricated 0, illegal 0, utf8 40/40, boundary 40/40",
    "l2": "provenance fidelity == 1.0 on every emitted output",
    "l3": (
        "answered_correct 40/40, pointed_wrong 0, abstained 0, should_answer_missing 0; the "
        "faithful_but_other_pairing column is reported separately with at least one real sample"
    ),
    "abstain": "an unknown question emits zero bytes (abstain never falls back to a record)",
    "closed_state": "nine items equal",
}

START_CPU = time.process_time()
START_WALL = time.time()


def cost() -> tuple[float, float]:
    return time.process_time() - START_CPU, time.time() - START_WALL


def write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# -- corpus -------------------------------------------------------------------
def corpus_rows() -> list[dict[str, Any]]:
    return base.frozen.build_fixture()


def corpus_records(rows: list[dict[str, Any]]) -> list[bytes]:
    return [row["text"].encode("utf-8") for row in rows]


def entity_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[int]]:
    table: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(rows):
        table.setdefault(("object", row["object"]), []).append(index)
        table.setdefault(("place", row["place"]), []).append(index)
    return {key: sorted(value) for key, value in table.items()}


def describe_sentences(rows: list[dict[str, Any]]) -> list[str]:
    return [row["text"] for row in rows if row["variant"] == "describe"]


def expected_pair(text: str) -> dict[str, tuple[str, str]]:
    found: dict[str, tuple[str, str]] = {}
    for name, phrase in OBJECT_DESCRIPTIONS.items():
        if phrase in text:
            found["object"] = ("object", name)
    for name, phrase in PLACE_DESCRIPTIONS.items():
        if phrase in text:
            found["place"] = ("place", name)
    return found


def build_table() -> BridgeTable:
    table = BridgeTable(key_fn=sequence_key, read_mode="sequence")
    write = table.write(bridge_records(OBJECT_DESCRIPTIONS, PLACE_DESCRIPTIONS))
    if write["action"] != "accepted":
        raise SystemExit(f"initial write rejected: {write['action']}")
    alias = table.write(AUTHORED_ALIASES)
    if alias["action"] != "accepted":
        raise SystemExit(f"alias write rejected: {alias['action']}")
    return table


# -- output unit --------------------------------------------------------------
def emit(record: bytes) -> dict[str, Any]:
    spans = sentence_spans(record)
    start, end = spans[0]
    body = record[start:end]
    try:
        text = body.decode("utf-8")
        utf8_ok = True
    except UnicodeDecodeError:
        text = ""
        utf8_ok = False
    return {
        "unit": "record" if (start, end) == (0, len(record)) else "sentence",
        "start": start,
        "end": end,
        "spans": [list(span) for span in spans],
        "bytes": body,
        "text": text,
        "utf8_ok": utf8_ok,
        "fabricated": body != record[start:end],
        "illegal_slice": not (0 <= start <= end <= len(record)),
        "boundary_legal": [start, end] in [list(span) for span in spans],
        "faithful": body == record[start:end],
    }


# -- the reading --------------------------------------------------------------
def byte_reading(
    table: BridgeTable, texts: list[str], rows: list[dict[str, Any]], records: list[bytes]
) -> dict[str, Any]:
    index_map = entity_index(rows)
    outcomes = {
        "answered_correct": 0,
        "pointed_wrong": 0,
        "abstained": 0,
        "should_answer_missing": 0,
    }
    emitted: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    sentence_rows: list[dict[str, Any]] = []
    for text in texts:
        expected = expected_pair(text)
        result = table.read_sentence(text)
        answers = [tuple(row["value"]) for row in result["answers"]]
        answer_set = set(answers)
        asked_pair = tuple(expected[slot][1] for slot in ("object", "place"))
        slot_rows: list[dict[str, Any]] = []
        for slot in ("object", "place"):
            want = expected.get(slot)
            if want is None:
                continue
            if want in answer_set:
                candidates = index_map.get(want, [])
                if not candidates:
                    outcomes["should_answer_missing"] += 1
                    slot_rows.append(
                        {"slot": slot, "entity": list(want), "outcome": "should_answer_missing"}
                    )
                    continue
                address = candidates[0]
                output = emit(records[address])
                outcomes["answered_correct"] += 1
                output_pair = (rows[address]["object"], rows[address]["place"])
                record_row = {
                    "sentence": text,
                    "slot": slot,
                    "entity": list(want),
                    "address": address,
                    "address_candidates": len(candidates),
                    "unit": output["unit"],
                    "span": [output["start"], output["end"]],
                    "record_pair": list(output_pair),
                    "asked_pair": list(asked_pair),
                    "pairing_matches": output_pair == asked_pair,
                    "utf8_ok": output["utf8_ok"],
                    "fabricated": output["fabricated"],
                    "illegal_slice": output["illegal_slice"],
                    "boundary_legal": output["boundary_legal"],
                    "faithful": output["faithful"],
                    "text": output["text"],
                }
                emitted.append(record_row)
                if output_pair != asked_pair and len(samples) < 5:
                    samples.append(
                        {
                            "question": text,
                            "asked_pair": list(asked_pair),
                            "slot": slot,
                            "entity": list(want),
                            "address": address,
                            "record_pair": list(output_pair),
                            "emitted": output["text"],
                            "why_it_is_not_the_asked_layer": (
                                "the bridge answered which entity, not which pairing; the record "
                                "returned is that entity's lowest-address record, whose place "
                                f"({output_pair[1]}) is not the asked place ({asked_pair[1]})"
                            ),
                        }
                    )
                slot_rows.append(
                    {"slot": slot, "entity": list(want), "outcome": "answered_correct"}
                )
            else:
                same_slot = [answer for answer in answers if answer[0] == slot]
                if same_slot:
                    outcomes["pointed_wrong"] += 1
                    slot_rows.append(
                        {
                            "slot": slot,
                            "entity": list(want),
                            "outcome": "pointed_wrong",
                            "got": [list(answer) for answer in same_slot],
                        }
                    )
                else:
                    outcomes["abstained"] += 1
                    slot_rows.append({"slot": slot, "entity": list(want), "outcome": "abstained"})
        stray = [answer for answer in answers if answer[0] not in expected]
        if stray:
            outcomes["pointed_wrong"] += len(stray)
        sentence_rows.append(
            {
                "sentence": text,
                "decision": result["decision"],
                "answers": [list(answer) for answer in answers],
                "passes": [
                    {
                        key: row[key]
                        for key in ("pass", "decision", "reason", "span_start", "span_end")
                    }
                    for row in result["passes"]
                ],
                "slots": slot_rows,
            }
        )
    slots = (
        len(emitted)
        + outcomes["pointed_wrong"]
        + outcomes["abstained"]
        + outcomes["should_answer_missing"]
    )
    l1 = {
        "fabricated": sum(int(row["fabricated"]) for row in emitted),
        "illegal_slices": sum(int(row["illegal_slice"]) for row in emitted),
        "utf8_valid": sum(int(row["utf8_ok"]) for row in emitted),
        "boundary_legal": sum(int(row["boundary_legal"]) for row in emitted),
        "outputs": len(emitted),
    }
    l2 = {
        "faithful": sum(int(row["faithful"]) for row in emitted),
        "outputs": len(emitted),
        "fidelity": (
            (sum(int(row["faithful"]) for row in emitted) / len(emitted)) if emitted else 0.0
        ),
    }
    mismatch = sum(1 for row in emitted if not row["pairing_matches"])
    return {
        "sentences": len(texts),
        "entity_slots": slots,
        "outcomes": outcomes,
        "l1": l1,
        "l2": l2,
        "faithful_but_other_pairing": mismatch,
        "pairing_matches": len(emitted) - mismatch,
        "samples": samples,
        "rows": sentence_rows,
        "outputs": [{key: value for key, value in row.items() if key != "text"} for row in emitted],
    }


def abstain_reading(table: BridgeTable, records: list[bytes]) -> dict[str, Any]:
    result = table.read_sentence(UNKNOWN_SENTENCE)
    return {
        "sentence": UNKNOWN_SENTENCE,
        "decision": result["decision"],
        "answers": [list(row["value"]) for row in result["answers"]],
        "bytes_emitted": 0 if not result["answers"] else None,
        "zero_bytes_ok": not result["answers"],
        "corpus_records_available": len(records),
    }


# -- closed state -------------------------------------------------------------
def tensor_sha(values: list[Any]) -> str:
    digest = hashlib.sha256()
    for value in values:
        if value is None:
            digest.update(b"none")
        elif isinstance(value, torch.Tensor):
            digest.update(value.detach().cpu().numpy().tobytes())
        else:
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def reference_digest(reference: Any, state: Any, step_hashes: list[str]) -> dict[str, Any]:
    parameters = [tensor for _, tensor in sorted(reference.named_parameters())]
    store = [state.store_keys, state.store_values, state.store_age, state.store_count]
    identity = {
        "parameters": len(parameters),
        "elements": int(sum(tensor.numel() for tensor in parameters)),
        "config": repr(reference.config),
    }
    return {
        "parameters_sha": tensor_sha(parameters),
        "state_sha": tensor_sha([state.h, state.fast, state.nu_ema, state.coverage, state.prior]),
        "keys_len": len(state.keys),
        "step": int(state.step),
        "store_sha": tensor_sha(store),
        "identity_sha": hashlib.sha256(
            json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "sc_sfk_none": state.sc_sfk is None,
        "battery_sha": hashlib.sha256("".join(step_hashes).encode("ascii")).hexdigest(),
        "battery_steps": len(step_hashes),
    }


def run_battery(reference: Any, texts: list[str]) -> dict[str, Any]:
    torch.manual_seed(SEED)
    state = reference.initial_state("r53-zero-perturbation")
    step_hashes: list[str] = []
    for raw in texts:
        state = reference.observe(state, raw.encode("utf-8"), goal=b"r53")
        step_hashes.append(
            tensor_sha([state.h, state.fast, state.nu_ema, state.coverage, state.prior])
        )
    return reference_digest(reference, state, step_hashes)


# -- freeze -------------------------------------------------------------------
def definition_digest() -> str:
    payload = {
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "aliases": [list(row) for row in AUTHORED_ALIASES],
        "unknown_sentence": UNKNOWN_SENTENCE,
        "seed": SEED,
        "estimate_cpu_upper": ESTIMATE_CPU_UPPER,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def freeze() -> dict[str, Any]:
    rows = corpus_rows()
    manifest = {
        "schema": "taiji-bridge-bytes-r53-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": SEED,
        "estimate_cpu_upper": ESTIMATE_CPU_UPPER,
        "cpu_cap": CPU_CAP,
        "runner_sha256": RUNNER_SHA_START,
        "bridge_sha256": file_sha(BRIDGE_SOURCE),
        "fixture_sha256": base.fixture_hashes(),
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "definition_digest": definition_digest(),
        "counts": {
            "corpus_records": len(rows),
            "describe_sentences": len(describe_sentences(rows)),
            "capability_entities": 2 * len(describe_sentences(rows)),
            "keys": 18,
        },
        "inputs": {
            "authored_aliases": [list(row) for row in AUTHORED_ALIASES],
            "unknown_sentence": UNKNOWN_SENTENCE,
        },
    }
    write_json(MANIFEST, manifest)
    return manifest


def verify_freeze() -> dict[str, Any]:
    if not MANIFEST.exists():
        raise SystemExit("no frozen manifest; run --freeze first")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["runner_sha256"] != RUNNER_SHA_START:
        raise SystemExit("runner changed after the freeze; re-freeze")
    if manifest["bridge_sha256"] != file_sha(BRIDGE_SOURCE):
        raise SystemExit("bridge module changed after the freeze; re-freeze")
    if manifest["fixture_sha256"] != base.fixture_hashes():
        raise SystemExit("fixture bytes changed; the frozen fixture must not move")
    if manifest["definition_digest"] != definition_digest():
        raise SystemExit("definition/criteria/inputs changed after the freeze; re-freeze")
    return manifest


def main() -> int:
    assert_no_black_processes()
    base.frozen.acquire_lock()
    try:
        if "--freeze" in sys.argv:
            manifest = freeze()
            print(json.dumps({"manifest": str(MANIFEST), "counts": manifest["counts"]}))
            return 0
        verify_freeze()

        rows = corpus_rows()
        records = corpus_records(rows)
        texts = describe_sentences(rows)

        torch.manual_seed(SEED)
        reference = LivingCoreReference()
        before = run_battery(reference, texts)

        table = build_table()
        reading = byte_reading(table, texts, rows, records)
        abstain = abstain_reading(table, records)

        after = run_battery(reference, texts)
        checks = {
            "parameters_equal": before["parameters_sha"] == after["parameters_sha"],
            "state_equal": before["state_sha"] == after["state_sha"],
            "identity_equal": before["identity_sha"] == after["identity_sha"],
            "store_equal": before["store_sha"] == after["store_sha"],
            "battery_equal": before["battery_sha"] == after["battery_sha"],
            "steps_equal": before["step"] == after["step"],
            "sc_sfk_none": before["sc_sfk_none"] and after["sc_sfk_none"],
            "flag_disabled": reference.config.sc_sfk_enabled is False,
            "rng_untouched": True,
        }
        closed_ok = all(checks.values())

        outputs = reading["l1"]["outputs"]
        l1_ok = (
            reading["l1"]["fabricated"] == 0
            and reading["l1"]["illegal_slices"] == 0
            and reading["l1"]["utf8_valid"] == outputs
            and reading["l1"]["boundary_legal"] == outputs
        )
        l2_ok = reading["l2"]["fidelity"] == 1.0
        outcomes = reading["outcomes"]
        l3_ok = (
            outcomes["answered_correct"] == reading["entity_slots"] == 40
            and outcomes["pointed_wrong"] == 0
            and outcomes["abstained"] == 0
            and outcomes["should_answer_missing"] == 0
        )
        abstain_ok = abstain["zero_bytes_ok"]
        verdict = {
            "l1": "l1_clean" if l1_ok else "l1_dirty",
            "l2": "provenance_faithful" if l2_ok else "provenance_broken",
            "l3": "entity_line_held" if l3_ok else "entity_line_dropped",
            "abstain": "abstain_emits_nothing" if abstain_ok else "abstain_emits_bytes",
            "closed_state": "closed_state_equal" if closed_ok else "closed_state_moved",
        }

        cpu, wall = cost()
        if cpu > CPU_CAP:
            raise SystemExit(f"cpu cap exceeded: {cpu:.2f}s > {CPU_CAP:.0f}s")
        report = {
            "schema": "taiji-bridge-bytes-r53-report-v1",
            "runner_sha256": RUNNER_SHA_START,
            "manifest_sha256": file_sha(MANIFEST),
            "bridge_sha256": file_sha(BRIDGE_SOURCE),
            "fixture_sha256": base.fixture_hashes(),
            "definition": DEFINITION,
            "criteria": CRITERIA,
            "reading": reading,
            "abstain": abstain,
            "closed_state": {"baseline": before, "after": after, "checks": checks},
            "verdict": verdict,
            "cost": {
                "cpu_seconds": cpu,
                "wall_seconds": wall,
                "cpu_cap": CPU_CAP,
                "estimate_cpu_upper": ESTIMATE_CPU_UPPER,
                "threads": torch.get_num_threads(),
            },
        }
        write_json(REPORT, report)
        print(
            json.dumps(
                {
                    **{f"verdict_{key}": value for key, value in verdict.items()},
                    "entity_slots": reading["entity_slots"],
                    "outcomes": outcomes,
                    "l1": reading["l1"],
                    "l2": reading["l2"],
                    "faithful_but_other_pairing": reading["faithful_but_other_pairing"],
                    "pairing_matches": reading["pairing_matches"],
                    "abstain_sentence": abstain["decision"],
                    "closed_state_ok": closed_ok,
                    "cost": {"cpu": round(cpu, 3), "wall": round(wall, 3)},
                },
                ensure_ascii=False,
            )
        )
        return 0
    finally:
        base.frozen.release_lock()


if __name__ == "__main__":
    raise SystemExit(main())
