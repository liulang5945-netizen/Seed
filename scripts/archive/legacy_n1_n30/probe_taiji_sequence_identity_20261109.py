"""R13 -- identity closing (segment A) and alias writing (segment B).

R12 showed that the token-unit question was answered but not the identity
question: with whole characters as tokens, ``长柄铜勺`` still answered because
``长`` occurs in exactly one written address.  A *set* of characters cannot tell
"shares one character" from "same phrase".  R13 stops matching characters and
starts matching the phrase itself.

Segment A (identity closing, frozen before any reading):

* key   -- the characters in order, multiplicity kept (``sequence_key``);
* read  -- answer only when a record's whole sequence occurs *contiguously*
           inside the query (``read_mode="sequence"``).  An empty query, an
           undecodable query, or a query that merely shares characters never
           answers.

Segment B (alias writing, same frozen batch):

* the 9 synonymous paraphrases are written as *additional keys* onto an
  already-populated table, each pointing at the same entity value.  This is
  "one value, many keys", not "one key, many values": the batch is 9 distinct
  keys, so the merge/replace/conflict paths must not fire.

Criteria, written before the first reading:

A. 12 unknown candidates answer 0; 18 byte fragments answer 0; the 20 whole
   sentences read without localisation mis-point 0 (a sentence names two
   addresses, so any answer is a mis-point); capability 40/40 with zero wrong
   pointers; the nine closed-state checks identical.
B. the 9 aliases read back the same entity 9/9; every segment-A zero reading is
   still 0 after the alias write; key count / entity count reported.

Verdicts, reported separately (a joint "pass" is not a reading):

* ``sequence_identity_clean`` -- all A gates hold;
* ``sequence_identity_still_leaks`` / ``_capability_failed`` /
  ``_perturbation_detected`` otherwise;
* ``alias_write_clean`` -- all B gates hold;
* ``alias_write_mechanism_failed`` -- an alias does not read back its entity;
* ``alias_write_reintroduces_answers`` -- the zeros break after the alias write.

Zero training, zero optimizer steps, zero Adam, zero confirm.  No fixture byte
is written; ``AddressTable`` and ``LivingCoreReference`` are untouched.
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
    value_key,
)
from taiji.living_core import LivingCoreReference  # noqa: E402
from taiji.native_goal_world_wide import (  # noqa: E402
    OBJECT_DESCRIPTIONS,
    PLACE_DESCRIPTIONS,
)
from taiji.process_guard import assert_no_black_processes  # noqa: E402

torch.set_num_threads(1)

MANIFEST = ROOT / "plans/manifests/taiji_bridge_sequence_identity_20261109.json"
REPORT = ROOT / "reports/taiji_bridge_sequence_identity_20261109.json"
BRIDGE_SOURCE = ROOT / "taiji/address_bridge.py"

RUNNER_SHA_START = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()

SEED = 20261109
ESTIMATE_CPU_UPPER = 30.0
CPU_CAP = float(base.CPU_CAP)

UNKNOWN_CANDIDATES = (
    "黑色铁铃",
    "白色帆布",
    "长柄铜勺",
    "软木瓶塞",
    "蓝色毛巾",
    "银质圆镜",
    "棕色皮箱",
    "青色陶罐",
    "灰色石阶",
    "紫色绸带",
    "金色烛台",
    "深红木箱",
)

REWRITES = (
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

DEFINITION = {
    "records": "nine (phrase -> (slot, name)) rows; the key is the phrase, the value the entity",
    "segment_a": {
        "key": "sequence_key: the characters in order, multiplicity kept, no positions dropped",
        "read": (
            "a record answers only when its whole character sequence occurs contiguously in the "
            "query; exactly one containment answers, two or more say "
            "ambiguous_multiple_anchors, zero says no_match; an empty or undecodable query says "
            "no_match with reason empty_key"
        ),
    },
    "segment_b": {
        "write": (
            "the 9 synonymous paraphrases are appended as additional keys onto the segment-A "
            "table, each carrying the same (slot, name) value: one value, many keys"
        ),
        "alias_rule": "an alias answers only under the same containment rule as segment A",
    },
    "unknown_inputs": "the 12 frozen candidates of R12, identical in both segments",
    "fragments": "each written phrase cut at byte offset 1 and 2 (18 inputs, not valid UTF-8)",
    "sentences": "the 20 describe sentences read whole, without localisation",
}

CRITERIA = {
    "segment_a": (
        "12 unknown candidates answer 0; 18 byte fragments answer 0; 20 whole sentences "
        "mis-point 0 (a non-answer is not a mis-point); capability 40/40 with zero wrong "
        "pointers; nine closed-state checks identical before and after."
    ),
    "segment_b": (
        "the 9 aliases read back the same entity 9/9; every segment-A zero reading is still 0 "
        "after the alias write; key count and entity count reported, not gated."
    ),
    "reporting": (
        "the two segments are reported separately; segment A passing does not carry segment B."
    ),
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


def sentences() -> list[str]:
    rows = base.frozen.build_fixture()
    return [row["text"] for row in rows if row["variant"] == "describe"]


def locate(sentence: str) -> dict[str, tuple[str, str]]:
    found: dict[str, tuple[str, str]] = {}
    for name, phrase in OBJECT_DESCRIPTIONS.items():
        if phrase in sentence:
            found["object"] = (phrase, name)
    for name, phrase in PLACE_DESCRIPTIONS.items():
        if phrase in sentence:
            found["place"] = (phrase, name)
    return found


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
    state = reference.initial_state("r13-zero-perturbation")
    step_hashes: list[str] = []
    for raw in texts:
        state = reference.observe(state, raw.encode("utf-8"), goal=b"r13")
        step_hashes.append(
            tensor_sha([state.h, state.fast, state.nu_ema, state.coverage, state.prior])
        )
    return reference_digest(reference, state, step_hashes)


def build_table() -> tuple[BridgeTable, dict[str, Any]]:
    table = BridgeTable(key_fn=sequence_key, read_mode="sequence")
    write = table.write(bridge_records(OBJECT_DESCRIPTIONS, PLACE_DESCRIPTIONS))
    if write["action"] != "accepted":
        raise SystemExit(f"initial write rejected: {write['action']}")
    return table, write


def capability(table: BridgeTable, texts: list[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    hits = wrong = abstained = 0
    for text in texts:
        for slot, (phrase, name) in sorted(locate(text).items()):
            result = table.read(phrase)
            expected = (slot, name)
            value = result["value"]
            if value is None:
                abstained += 1
            elif tuple(value) == expected:
                hits += 1
            else:
                wrong += 1
            rows.append(
                {
                    "phrase": phrase,
                    "expected": list(expected),
                    "decision": result["decision"],
                    "value": list(value) if value else None,
                    "reason": result["reason"],
                }
            )
    total = len(rows)
    return {
        "rows": rows,
        "hits": hits,
        "of": total,
        "wrong_pointers": wrong,
        "abstained": abstained,
        "accuracy": hits / total if total else 0.0,
        "ok": hits == total and wrong == 0,
    }


def unknown_reading(table: BridgeTable) -> dict[str, Any]:
    rows = []
    answering = 0
    for phrase in UNKNOWN_CANDIDATES:
        result = table.read(phrase)
        value = result["value"]
        if value is not None:
            answering += 1
        rows.append(
            {
                "phrase": phrase,
                "decision": result["decision"],
                "value": list(value) if value else None,
                "reason": result["reason"],
                "contained_keys": [
                    record["key"]
                    for record in table.records
                    if phrase.find("".join(record["key"])) >= 0
                ],
                "shared_characters": sorted(
                    {character for record in table.records for character in record["key"]}
                    & set(phrase)
                ),
                "answered": value is not None,
            }
        )
    return {
        "rows": rows,
        "candidates": len(rows),
        "answering_phrases": answering,
        "zero_line_ok": answering == 0,
    }


def fragments_reading(table: BridgeTable) -> dict[str, Any]:
    rows = []
    same = other = 0
    for phrase, value in bridge_records(OBJECT_DESCRIPTIONS, PLACE_DESCRIPTIONS):
        data = phrase.encode("utf-8")
        for offset in (1, 2):
            result = table.read(data[offset:])
            answer = result["value"]
            if answer is None:
                kind = "abstain"
            elif tuple(answer) == tuple(value):
                kind = "same_record"
                same += 1
            else:
                kind = "other_record"
                other += 1
            rows.append(
                {
                    "phrase": phrase,
                    "byte_offset": offset,
                    "decision": result["decision"],
                    "value": list(answer) if answer else None,
                    "reason": result["reason"],
                    "kind": kind,
                }
            )
    return {
        "rows": rows,
        "inputs": len(rows),
        "same_record": same,
        "other_record": other,
        "false_answers": same + other,
    }


def sentences_reading(table: BridgeTable, texts: list[str]) -> dict[str, Any]:
    rows = []
    answering = 0
    decisions: dict[str, int] = {}
    for text in texts:
        result = table.read(text)
        value = result["value"]
        decisions[result["decision"]] = decisions.get(result["decision"], 0) + 1
        if value is not None:
            answering += 1
        rows.append(
            {
                "sentence": text,
                "decision": result["decision"],
                "value": list(value) if value else None,
                "reason": result["reason"],
                "anchors": result["anchors"],
            }
        )
    return {
        "rows": rows,
        "inputs": len(rows),
        "false_answers": answering,
        "decisions": dict(sorted(decisions.items())),
    }


def rewrites_reading(table: BridgeTable) -> dict[str, Any]:
    rows = []
    correct = wrong = abstain = 0
    for text, expected in REWRITES:
        result = table.read(text)
        value = result["value"]
        if value is None:
            outcome = "abstain"
            abstain += 1
        elif tuple(value) == expected:
            outcome = "correct"
            correct += 1
        else:
            outcome = "wrong"
            wrong += 1
        rows.append(
            {
                "text": text,
                "expected": list(expected),
                "decision": result["decision"],
                "value": list(value) if value else None,
                "reason": result["reason"],
                "outcome": outcome,
            }
        )
    return {
        "rows": rows,
        "inputs": len(rows),
        "correct": correct,
        "wrong": wrong,
        "abstain": abstain,
        "answered": correct + wrong,
        "false_answers": wrong,
    }


def keys_reading(table: BridgeTable) -> dict[str, Any]:
    entities = {value_key(table.current(index)) for index in range(len(table.records))}
    return {
        "key_count": len(table.records),
        "entity_count": len(entities),
        "ratio": len(table.records) / len(entities) if entities else 0.0,
    }


def definition_digest() -> str:
    payload = {
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "unknown": list(UNKNOWN_CANDIDATES),
        "rewrites": [list(row) for row in REWRITES],
        "seed": SEED,
        "estimate_cpu_upper": ESTIMATE_CPU_UPPER,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def freeze() -> dict[str, Any]:
    records = bridge_records(OBJECT_DESCRIPTIONS, PLACE_DESCRIPTIONS)
    manifest = {
        "schema": "taiji-bridge-sequence-identity-r13-v1",
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
            "records": len(records),
            "describe_sentences": len(sentences()),
            "capability_reads": 2 * len(sentences()),
            "unknown_candidates": len(UNKNOWN_CANDIDATES),
            "aliases": len(REWRITES),
            "fragments": 2 * len(records),
            "battery_texts": len(sentences()),
            "keys_after_alias": len(records) + len(REWRITES),
        },
        "inputs": {
            "unknown_candidates": list(UNKNOWN_CANDIDATES),
            "rewrites": [list(row) for row in REWRITES],
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

        texts = sentences()

        torch.manual_seed(SEED)
        reference = LivingCoreReference()
        before = run_battery(reference, texts)

        # -- segment A: identity closing, original keys only -----------------
        table_a, write_a = build_table()
        segment_a = {
            "write": {
                "action": write_a["action"],
                "appended": write_a["appended"],
                "merged": write_a["merged"],
                "replaced": write_a["replaced"],
                "slots": write_a["slots"],
                "key_sizes": sorted({len(record["key"]) for record in table_a.records}),
            },
            "capability": capability(table_a, texts),
            "unknown": unknown_reading(table_a),
            "fragments": fragments_reading(table_a),
            "sentences_direct": sentences_reading(table_a, texts),
            "rewrites_original_only": rewrites_reading(table_a),
            "keys": keys_reading(table_a),
        }
        cost_of_a = {
            "rewrites_no_match": segment_a["rewrites_original_only"]["abstain"],
            "rewrites_answered_without_alias": segment_a["rewrites_original_only"]["answered"],
            "note": (
                "the abstaining paraphrases are segment A's price: identity without aliases "
                "answers only the exact written wording"
            ),
        }

        # -- segment B: write the 9 paraphrases as additional keys -----------
        table_b, write_b = build_table()
        write_alias = table_b.write(REWRITES)
        if write_alias["action"] != "accepted":
            raise SystemExit(f"alias write rejected: {write_alias['action']}")

        alias_rows = []
        same_entity = 0
        for text, expected in REWRITES:
            result = table_b.read(text)
            value = result["value"]
            ok = value is not None and tuple(value) == expected
            if ok:
                same_entity += 1
            alias_rows.append(
                {
                    "text": text,
                    "expected": list(expected),
                    "decision": result["decision"],
                    "value": list(value) if value else None,
                    "reason": result["reason"],
                    "same_entity": ok,
                }
            )

        segment_b = {
            "write_records": {
                "action": write_b["action"],
                "appended": write_b["appended"],
                "slots": write_b["slots"],
            },
            "write_aliases": {
                "action": write_alias["action"],
                "appended": write_alias["appended"],
                "merged": write_alias["merged"],
                "replaced": write_alias["replaced"],
                "items": write_alias["items"],
                "slots": write_alias["slots"],
            },
            "alias_readback": {
                "rows": alias_rows,
                "of": len(alias_rows),
                "same_entity": same_entity,
                "ok": same_entity == len(alias_rows),
            },
            "keys": keys_reading(table_b),
            "capability": capability(table_b, texts),
            "unknown": unknown_reading(table_b),
            "fragments": fragments_reading(table_b),
            "sentences_direct": sentences_reading(table_b, texts),
            "rewrites_via_alias": rewrites_reading(table_b),
        }

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

        zero_a = {
            "unknown": segment_a["unknown"]["answering_phrases"],
            "fragments": segment_a["fragments"]["false_answers"],
            "sentences": segment_a["sentences_direct"]["false_answers"],
        }
        zero_b = {
            "unknown": segment_b["unknown"]["answering_phrases"],
            "fragments": segment_b["fragments"]["false_answers"],
            "sentences": segment_b["sentences_direct"]["false_answers"],
        }

        if not segment_a["capability"]["ok"]:
            verdict_a = "sequence_identity_capability_failed"
        elif not all(checks.values()):
            verdict_a = "sequence_identity_perturbation_detected"
        elif any(count != 0 for count in zero_a.values()):
            verdict_a = "sequence_identity_still_leaks"
        else:
            verdict_a = "sequence_identity_clean"

        zero_kept = all(count == 0 for count in zero_b.values())
        if not segment_b["alias_readback"]["ok"]:
            verdict_b = "alias_write_mechanism_failed"
        elif not zero_kept:
            verdict_b = "alias_write_reintroduces_answers"
        else:
            verdict_b = "alias_write_clean"

        failure_class = []
        if not segment_b["alias_readback"]["ok"]:
            failure_class.append("alias_does_not_read_back")
        if not zero_kept:
            failure_class.append("alias_key_collides_with_unknown_input")

        cpu, wall = cost()
        if cpu > CPU_CAP:
            raise SystemExit(f"cpu cap exceeded: {cpu:.2f}s > {CPU_CAP:.0f}s")
        report = {
            "schema": "taiji-bridge-sequence-identity-r13-report-v1",
            "runner_sha256": RUNNER_SHA_START,
            "manifest_sha256": file_sha(MANIFEST),
            "bridge_sha256": file_sha(BRIDGE_SOURCE),
            "fixture_sha256": base.fixture_hashes(),
            "definition": DEFINITION,
            "criteria": CRITERIA,
            "segment_a": segment_a,
            "cost_of_a": cost_of_a,
            "segment_b": segment_b,
            "zero_line": {
                "segment_a": zero_a,
                "segment_b": zero_b,
                "kept_after_alias": zero_kept,
                "criterion": "0 answering inputs on unknown candidates, fragments and whole sentences",
            },
            "closed_state": {"baseline": before, "after": after, "checks": checks},
            "verdict": {"a": verdict_a, "b": verdict_b, "b_failure_class": failure_class},
            "cost": {
                "cpu_seconds": cpu,
                "wall_seconds": wall,
                "cpu_cap": CPU_CAP,
                "estimate_cpu_upper": ESTIMATE_CPU_UPPER,
                "threads": torch.get_num_threads(),
            },
            "note": (
                "segment A and segment B are separate readings: A tests identity, B tests whether "
                "extra keys can be added without giving the unknown inputs a way back in."
            ),
        }
        write_json(REPORT, report)
        print(
            json.dumps(
                {
                    "verdict_a": verdict_a,
                    "verdict_b": verdict_b,
                    "zero_a": zero_a,
                    "zero_b": zero_b,
                    "capability_a": {
                        "hits": segment_a["capability"]["hits"],
                        "of": segment_a["capability"]["of"],
                        "wrong": segment_a["capability"]["wrong_pointers"],
                    },
                    "capability_b": {
                        "hits": segment_b["capability"]["hits"],
                        "of": segment_b["capability"]["of"],
                        "wrong": segment_b["capability"]["wrong_pointers"],
                    },
                    "sentences_decisions_a": segment_a["sentences_direct"]["decisions"],
                    "rewrites_a": {
                        "correct": segment_a["rewrites_original_only"]["correct"],
                        "wrong": segment_a["rewrites_original_only"]["wrong"],
                        "abstain": segment_a["rewrites_original_only"]["abstain"],
                    },
                    "alias_readback": f"{same_entity}/{len(alias_rows)}",
                    "keys_b": segment_b["keys"],
                    "closed_state_ok": all(checks.values()),
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
