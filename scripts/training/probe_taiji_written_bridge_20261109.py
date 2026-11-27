"""R11 -- written bridge: the phrase->name map as nine address records.

R9 showed the description phrase survives inside the sentence.  R10 showed a
one-wording table cannot generalise (constructive zero).  R10b showed a
two-wording table carries only a lambda-fragile, place-only signal.  Together
they say the mapping must be *written*, not fitted.

This round is the minimal implementation of that write, and nothing else:

* nine records, key = the phrase's 2-byte token set, value = (slot, name);
* three write outcomes with their own readings -- merge (same key, same value),
  replace (same key, different value, old value kept in history), and a
  batch-internal collision that rolls the whole batch back (proved by digest
  equality, not by promise);
* abstention is reported as an output, not as a residual: an unwritten phrase
  and a query that anchors two records must both return no value, with a reason;
* the default graph is replayed twice around the bridge work and compared on
  parameters, state, identity and store signature (closed-state zero
  perturbation).

Frozen criteria, written before the first reading:

* capability: 20 describe sentences -> locate the phrase -> read the record ->
  (slot, name); accuracy >= 0.95 and **zero wrong pointers** (a wrong pointer is
  a non-empty value that is not the expected pair);
* mechanism: merge == 9, replace == 9 with all old values still in history,
  exactly one rejected batch whose table digest is unchanged;
* abstention: every unwritten phrase returns no_match with value None, every
  two-record query returns ambiguous_multiple_anchors with value None, and no
  battery produces a non-empty wrong answer;
* closed state: parameters, state, identity, store signature and the emitted
  battery bytes are identical across the two replays, and ``sc_sfk`` stays None.

Verdict: ``written_bridge_operational`` iff all four hold; otherwise the first
failing group names it (``written_bridge_capability_failed`` /
``_mechanism_failed`` / ``_abstention_failed`` / ``_perturbation_detected``).

Zero training, zero optimizer steps, zero Adam, zero confirm.  The 9-name
fixture is read, never modified; the bridge is a new module and no default path
imports it.
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
from taiji.address_bridge import BridgeTable, bridge_records, phrase_key  # noqa: E402
from taiji.living_core import LivingCoreConfig, LivingCoreReference  # noqa: E402
from taiji.native_goal_world_wide import (  # noqa: E402
    OBJECT_DESCRIPTIONS,
    PLACE_DESCRIPTIONS,
)

torch.set_num_threads(1)

MANIFEST = ROOT / "plans/manifests/taiji_written_bridge_20261109.json"
REPORT = ROOT / "reports/taiji_written_bridge_20261109.json"
BRIDGE_SOURCE = ROOT / "taiji/address_bridge.py"

RUNNER_SHA_START = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()

ACCURACY_FLOOR = 0.95
COST_ESTIMATE_CPU_UPPER = 60.0
SEED = 20261109

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

DEFINITION = {
    "key": (
        "the phrase's 2-byte token set (hex), order-free, deduped; a sliding 2-byte "
        "window over the UTF-8 bytes, so it is alignment-sensitive by construction."
    ),
    "value": "(slot, name), e.g. ('place', '书架').  Names never enter the key.",
    "record_count": 9,
    "write_outcomes": {
        "appended": "the key is new.",
        "merged": "same key, same value: count+1, no new record.",
        "replaced": "same key, different value: the old value moves to history.",
        "rejected_batches": "one batch carries two different values for one key; the batch "
        "is validated before it is applied and the table is restored from a snapshot.",
    },
    "read_rule": (
        "an address anchors the query only through a token that occurs in exactly one "
        "address; exactly one anchor answers, two or more say ambiguous_multiple_anchors, "
        "zero says no_match.  A non-answer returns value None."
    ),
    "route": (
        "sentence -> locate the written phrase inside it (exact substring) -> the phrase's "
        "token set -> read.  Reading the raw sentence instead is reported separately: the "
        "2-byte window drifts by one byte after the 3-byte lead character, so the sentence "
        "cannot be used as the key directly."
    ),
}

CRITERIA = {
    "capability": (
        "20 describe sentences: accuracy >= 0.95 and zero wrong pointers (a wrong pointer is "
        "a non-empty value different from the expected (slot, name))."
    ),
    "mechanism": (
        "one merge battery with merged == 9 and slots == 9; one replace battery with "
        "replaced == 9, slots == 9 and every original value still in history; one conflict "
        "battery with rejected_batches == 1, rollbacks == 1 and an unchanged table digest."
    ),
    "abstention": (
        "every unwritten phrase returns no_match with value None; every two-record query "
        "returns ambiguous_multiple_anchors with value None; zero non-empty wrong answers "
        "across all abstention batteries."
    ),
    "closed_state": (
        "same battery (20 sentences) replayed before and after the bridge work: parameters, "
        "state tensors, identity digest, store signature and emitted bytes are identical, and "
        "state.sc_sfk stays None."
    ),
}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sentences() -> list[str]:
    rows = base.frozen.build_fixture()
    return [row["text"] for row in rows if row["variant"] == "describe"]


def locate(sentence: str) -> dict[str, Any]:
    """The written phrase that appears verbatim in the sentence, per slot."""
    found: dict[str, Any] = {}
    for name, phrase in OBJECT_DESCRIPTIONS.items():
        if phrase in sentence:
            found["object"] = {"phrase": phrase, "name": name}
    for name, phrase in PLACE_DESCRIPTIONS.items():
        if phrase in sentence:
            found["place"] = {"phrase": phrase, "name": name}
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


def reference_digest(
    reference: LivingCoreReference, state: Any, step_hashes: list[str]
) -> dict[str, Any]:
    parameters = [tensor for _, tensor in sorted(reference.named_parameters())]
    living = state
    store = [living.store_keys, living.store_values, living.store_age, living.store_count]
    identity = {
        "parameters": len(parameters),
        "elements": int(sum(tensor.numel() for tensor in parameters)),
        "config": repr(reference.config),
    }
    return {
        "parameters_sha": tensor_sha(parameters),
        "state_sha": tensor_sha(
            [living.h, living.fast, living.nu_ema, living.coverage, living.prior]
        ),
        "keys_len": len(living.keys),
        "step": int(living.step),
        "store_sha": tensor_sha(store),
        "identity_sha": hashlib.sha256(
            json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "sc_sfk": living.sc_sfk,
        "sc_sfk_none": living.sc_sfk is None,
        "battery_sha": hashlib.sha256("".join(step_hashes).encode("ascii")).hexdigest(),
        "battery_steps": len(step_hashes),
    }


def run_battery(reference: LivingCoreReference, texts: list[str]) -> dict[str, Any]:
    torch.manual_seed(SEED)
    state = reference.initial_state("r11-zero-perturbation")
    step_hashes: list[str] = []
    for raw in texts:
        state = reference.observe(state, raw.encode("utf-8"), goal=b"r11")
        step_hashes.append(
            tensor_sha([state.h, state.fast, state.nu_ema, state.coverage, state.prior])
        )
    record = reference_digest(reference, state, step_hashes)
    record["sc_sfk"] = None if record["sc_sfk"] is None else "present"
    return record


def capability(table: BridgeTable, texts: list[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    hits = wrong = abstain = 0
    for text in texts:
        found = locate(text)
        located = sorted(found.items())
        for slot, target in located:
            result = table.read(target["phrase"])
            expected = (slot, target["name"])
            ok = result["value"] == expected
            if result["value"] is None:
                abstain += 1
            elif not ok:
                wrong += 1
            else:
                hits += 1
            rows.append(
                {
                    "sentence": text,
                    "slot": slot,
                    "phrase": target["phrase"],
                    "expected": list(expected),
                    "decision": result["decision"],
                    "value": list(result["value"]) if result["value"] else None,
                    "reason": result["reason"],
                    "ok": bool(ok),
                }
            )
    total = len(rows)
    return {
        "rows": rows,
        "of": total,
        "hits": hits,
        "wrong_pointers": wrong,
        "abstained": abstain,
        "accuracy": round(hits / total, 6),
        "sentences": len(texts),
        "located_both_slots": sum(1 for row in rows) == 2 * len(texts),
    }


def mechanism_batteries(records: list[tuple[str, tuple[str, str]]]) -> dict[str, Any]:
    merge_table = BridgeTable()
    merge_first = merge_table.write(records)
    merge_second = merge_table.write(records)
    merge = {
        "batches": merge_table.batches,
        "first_batch": merge_first,
        "second_batch": merge_second,
        "slots": len(merge_table.records),
        "appended": merge_table.appended,
        "merged": merge_table.merged,
        "replaced": merge_table.replaced,
        "rejected_batches": merge_table.rejected_batches,
        "rollbacks": merge_table.rollbacks,
        "counts": [record["count"] for record in merge_table.records],
        "read_after_merge": [merge_table.read(phrase)["value"] for phrase, _ in records],
    }

    replace_table = BridgeTable()
    replace_table.write(records)
    originals = [replace_table.current(index) for index in range(len(replace_table.records))]
    replacement = ("replaced", "sentinel")
    replace_result = replace_table.write([(phrase, replacement) for phrase, _ in records])
    history_ok = all(
        replace_table.history_at(index) == [originals[index]]
        for index in range(len(replace_table.records))
    )
    replace = {
        "batch": replace_result,
        "slots": len(replace_table.records),
        "appended": replace_table.appended,
        "merged": replace_table.merged,
        "replaced": replace_table.replaced,
        "rejected_batches": replace_table.rejected_batches,
        "rollbacks": replace_table.rollbacks,
        "originals": [list(value) for value in originals],
        "history": [
            [list(value) for value in replace_table.history_at(index)]
            for index in range(len(replace_table.records))
        ],
        "old_value_preserved": history_ok,
        "read_after_replace": [replace_table.read(phrase)["value"] for phrase, _ in records],
        "reads_return_replacement": all(
            replace_table.read(phrase)["value"] == replacement for phrase, _ in records
        ),
    }

    conflict_table = BridgeTable()
    conflict_table.write(records)
    digest_before = conflict_table.content_digest()
    slots_before = len(conflict_table.records)
    values_before = [conflict_table.current(index) for index in range(slots_before)]
    first_phrase, first_value = records[0]
    conflict_result = conflict_table.write(
        [
            (first_phrase, first_value),
            (first_phrase, ("conflict", "different")),
            ("全新的短语甲", ("object", "甲")),
            ("全新的短语乙", ("place", "乙")),
        ]
    )
    conflict = {
        "batch": conflict_result,
        "content_digest_before": digest_before,
        "content_digest_after": conflict_table.content_digest(),
        "digest_unchanged": conflict_table.content_digest() == digest_before,
        "slots_before": slots_before,
        "slots_after": len(conflict_table.records),
        "values_unchanged": values_before
        == [conflict_table.current(index) for index in range(slots_before)],
        "rejected_batches": conflict_table.rejected_batches,
        "rollbacks": conflict_table.rollbacks,
        "rejected_reasons": dict(conflict_table.rejected_reasons),
        "new_key_absent": conflict_table.read("全新的短语甲")["decision"] == "no_match",
    }
    return {"merge": merge, "replace": replace, "conflict": conflict}


def shared_token_probe(table: BridgeTable) -> dict[str, Any]:
    """A query made only of a token that two records share: no unique token, no answer."""
    document_frequency: dict[str, int] = {}
    owners: dict[str, list[int]] = {}
    for index, record in enumerate(table.records):
        for token in record["key"]:
            document_frequency[token] = document_frequency.get(token, 0) + 1
            owners.setdefault(token, []).append(index)
    shared = [token for token, count in document_frequency.items() if count >= 2]
    if not shared:
        return {"status": "no_shared_token", "decision": None, "value": None}
    token = sorted(shared)[0]
    result = table.read_tokens(frozenset({token}))
    return {
        "status": "shared_token_only",
        "token": token,
        "document_frequency": document_frequency[token],
        "owners": owners[token],
        "decision": result["decision"],
        "reason": result["reason"],
        "value": None if result["value"] is None else list(result["value"]),
    }


def abstention(table: BridgeTable, texts: list[str], unknown_phrases: list[str]) -> dict[str, Any]:
    key_tokens: set[str] = set()
    for record in table.records:
        key_tokens |= record["key"]
    unknown_rows = []
    for phrase in unknown_phrases:
        tokens = phrase_key(phrase)
        result = table.read_tokens(tokens)
        unknown_rows.append(
            {
                "phrase": phrase,
                "tokens": len(tokens),
                "disjoint_from_keys": not (tokens & key_tokens),
                "decision": result["decision"],
                "value": None if result["value"] is None else list(result["value"]),
                "reason": result["reason"],
            }
        )
    shared_row = shared_token_probe(table)

    pairs = []
    keys = [
        (phrase, record["key"])
        for phrase, record in zip(
            [phrase for phrase, _ in bridge_records(OBJECT_DESCRIPTIONS, PLACE_DESCRIPTIONS)],
            table.records,
            strict=True,
        )
    ]
    for left in range(len(keys)):
        for right in range(left + 1, len(keys)):
            union = keys[left][1] | keys[right][1]
            result = table.read_tokens(union)
            pairs.append(
                {
                    "pair": [keys[left][0], keys[right][0]],
                    "decision": result["decision"],
                    "anchors": len(result["anchors"]),
                    "value": None if result["value"] is None else list(result["value"]),
                }
            )

    sentence_rows = []
    for text in texts:
        found = locate(text)
        if len(found) < 2:
            continue
        union = frozenset()
        for target in found.values():
            union |= phrase_key(target["phrase"])
        result = table.read_tokens(union)
        sentence_rows.append(
            {
                "sentence": text,
                "decision": result["decision"],
                "anchors": len(result["anchors"]),
                "value": None if result["value"] is None else list(result["value"]),
            }
        )

    direct_rows = []
    direct_wrong = 0
    for text in texts:
        result = table.read(text)
        value = None if result["value"] is None else list(result["value"])
        expected = [(slot, target["name"]) for slot, target in locate(text).items()]
        if value is not None and tuple(value) not in expected:
            direct_wrong += 1
        direct_rows.append(
            {
                "sentence": text,
                "decision": result["decision"],
                "reason": result["reason"],
                "unknown_tokens": len(result["unknown_tokens"]),
                "value": value,
                "wrong_pointer": bool(value is not None and tuple(value) not in expected),
            }
        )

    def tally(rows: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["decision"]] = counts.get(row["decision"], 0) + 1
        return dict(sorted(counts.items()))

    return {
        "unknown": unknown_rows,
        "unknown_tally": tally(unknown_rows),
        "unknown_all_disjoint": all(row["disjoint_from_keys"] for row in unknown_rows),
        "shared_token_only": shared_row,
        "ambiguous_pairs": pairs,
        "ambiguous_pair_tally": tally(pairs),
        "ambiguous_sentences": sentence_rows,
        "ambiguous_sentence_tally": tally(sentence_rows),
        "sentence_direct": direct_rows,
        "sentence_direct_tally": tally(direct_rows),
        "sentence_direct_wrong_pointers": direct_wrong,
        "sentence_direct_status": (
            "diagnostic only: the raw sentence carries both phrases, so it anchors two "
            "records and is withheld; it also drifts a byte after the 3-byte lead "
            "character.  This is why the route localises the phrase first."
        ),
        "false_answers": sum(
            1 for row in unknown_rows + pairs + sentence_rows if row["value"] is not None
        )
        + (1 if shared_row["value"] is not None else 0),
    }


def select_unknown(table: BridgeTable, count: int = 5) -> dict[str, Any]:
    """Unwritten phrases whose token set is disjoint from every key.

    A two-byte window crosses character boundaries, so "semantically unrelated"
    does not imply "no shared token": a phrase can inherit a neighbouring
    character's trailing byte and anchor a record.  Those candidates are
    recorded as rejected, not silently used.
    """
    key_tokens: set[str] = set()
    for record in table.records:
        key_tokens |= record["key"]
    chosen: list[str] = []
    rejected: list[dict[str, Any]] = []
    for phrase in UNKNOWN_CANDIDATES:
        tokens = phrase_key(phrase)
        shared = sorted(tokens & key_tokens)
        if shared:
            rejected.append({"phrase": phrase, "shared_tokens": shared})
        else:
            chosen.append(phrase)
        if len(chosen) == count:
            break
    return {"chosen": chosen, "rejected": rejected, "requested": count}


def phase_freeze() -> dict[str, Any]:
    texts = sentences()
    probe_table = BridgeTable()
    probe_table.write(bridge_records(OBJECT_DESCRIPTIONS, PLACE_DESCRIPTIONS))
    selection = select_unknown(probe_table)
    manifest = {
        "schema": "taiji-written-bridge-v1",
        "date": "2026-11-09",
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runner_sha256": RUNNER_SHA_START,
        "bridge_sha256": file_sha(BRIDGE_SOURCE),
        "fixture_sha256": base.fixture_hashes(),
        "fixture_files": list(base.frozen.FIXTURE_FILES),
        "question": (
            "把 9 条 (短语 -> (槽位,名字)) 真正写进地址层后，20 条 describe 句能否零训练读出"
            "名字、且不说错；写入的三种结局与两种“不知道”是否都可读？"
        ),
        "counts": {
            "records": 9,
            "describe_sentences": len(texts),
            "capability_reads": 2 * len(texts),
            "unknown_phrases": len(selection["chosen"]),
            "unknown_candidates": len(UNKNOWN_CANDIDATES),
            "ambiguous_pairs": 36,
            "battery_texts": len(texts),
        },
        "unknown_selection": selection,
        "accuracy_floor": ACCURACY_FLOOR,
        "cost_estimate_cpu_upper": COST_ESTIMATE_CPU_UPPER,
        "seed": SEED,
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
    if manifest["bridge_sha256"] != file_sha(BRIDGE_SOURCE):
        raise SystemExit("bridge module changed after the freeze; re-freeze")

    texts = sentences()
    records = bridge_records(OBJECT_DESCRIPTIONS, PLACE_DESCRIPTIONS)
    if len(records) != 9:
        raise SystemExit(f"expected 9 records, built {len(records)}")

    reference = LivingCoreReference(LivingCoreConfig())
    baseline = run_battery(reference, texts)
    rng_before = torch.get_rng_state().clone()

    table = BridgeTable()
    first_write = table.write(records)
    unknown_selection = select_unknown(table)
    if unknown_selection != manifest["unknown_selection"]:
        raise SystemExit("unknown-phrase selection changed after the freeze; re-freeze")
    capability_reading = capability(table, texts)
    mechanism = mechanism_batteries(records)
    abstain = abstention(table, texts, unknown_selection["chosen"])
    rng_untouched = bool(torch.equal(rng_before, torch.get_rng_state()))

    after = run_battery(reference, texts)

    checks = {
        "parameters_equal": baseline["parameters_sha"] == after["parameters_sha"],
        "state_equal": baseline["state_sha"] == after["state_sha"],
        "identity_equal": baseline["identity_sha"] == after["identity_sha"],
        "store_equal": baseline["store_sha"] == after["store_sha"],
        "battery_equal": baseline["battery_sha"] == after["battery_sha"],
        "steps_equal": baseline["battery_steps"] == after["battery_steps"],
        "sc_sfk_none": baseline["sc_sfk_none"] and after["sc_sfk_none"],
        "flag_disabled": reference.config.sc_sfk_enabled is False,
        "rng_untouched": rng_untouched,
    }

    capability_ok = (
        capability_reading["accuracy"] >= ACCURACY_FLOOR
        and capability_reading["wrong_pointers"] == 0
    )
    mechanism_ok = bool(
        mechanism["merge"]["merged"] == 9
        and mechanism["merge"]["slots"] == 9
        and mechanism["merge"]["appended"] == 9
        and mechanism["replace"]["replaced"] == 9
        and mechanism["replace"]["slots"] == 9
        and mechanism["replace"]["old_value_preserved"]
        and mechanism["replace"]["reads_return_replacement"]
        and mechanism["conflict"]["rejected_batches"] == 1
        and mechanism["conflict"]["rollbacks"] == 1
        and mechanism["conflict"]["digest_unchanged"]
        and mechanism["conflict"]["slots_after"] == 9
        and mechanism["conflict"]["values_unchanged"]
        and mechanism["conflict"]["new_key_absent"]
    )
    abstention_ok = bool(
        abstain["false_answers"] == 0
        and abstain["unknown_tally"] == {"no_match": len(unknown_selection["chosen"])}
        and abstain["ambiguous_pair_tally"] == {"ambiguous_multiple_anchors": 36}
        and abstain["ambiguous_sentence_tally"] == {"ambiguous_multiple_anchors": len(texts)}
        and abstain["shared_token_only"]["decision"] == "no_match"
        and abstain["shared_token_only"]["reason"] == "no_unique_token"
    )
    closed_state_ok = all(checks.values())

    if not capability_ok:
        verdict = "written_bridge_capability_failed"
    elif not mechanism_ok:
        verdict = "written_bridge_mechanism_failed"
    elif not abstention_ok:
        verdict = "written_bridge_abstention_failed"
    elif not closed_state_ok:
        verdict = "written_bridge_perturbation_detected"
    else:
        verdict = "written_bridge_operational"

    report = {
        "schema": "taiji-written-bridge-v1",
        "date": "2026-11-09",
        "runner_sha256": RUNNER_SHA_START,
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "bridge_sha256": manifest["bridge_sha256"],
        "fixture_sha256": manifest["fixture_sha256"],
        "counts": manifest["counts"],
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "verdict": verdict,
        "groups": {
            "capability_ok": capability_ok,
            "mechanism_ok": mechanism_ok,
            "abstention_ok": abstention_ok,
            "closed_state_ok": closed_state_ok,
        },
        "capability": capability_reading,
        "mechanism": mechanism,
        "abstention": abstain,
        "closed_state": {
            "checks": checks,
            "baseline": baseline,
            "after": after,
            "note": (
                "the same reference object is reused for both replays; only the bridge "
                "work happens in between, so any difference would be a leak from the "
                "new module into the default path"
            ),
        },
        "write_baseline": first_write,
        "table_readings": {
            "slots": len(table.records),
            "appended": table.appended,
            "merged": table.merged,
            "replaced": table.replaced,
            "rejected_batches": table.rejected_batches,
            "rollbacks": table.rollbacks,
            "digest": table.digest(),
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
    print(
        json.dumps(
            {
                "verdict": verdict,
                "groups": report["groups"],
                "capability": {
                    "accuracy": capability_reading["accuracy"],
                    "hits": capability_reading["hits"],
                    "of": capability_reading["of"],
                    "wrong_pointers": capability_reading["wrong_pointers"],
                    "abstained": capability_reading["abstained"],
                },
                "merge": {
                    "appended": mechanism["merge"]["appended"],
                    "merged": mechanism["merge"]["merged"],
                    "slots": mechanism["merge"]["slots"],
                },
                "replace": {
                    "replaced": mechanism["replace"]["replaced"],
                    "old_value_preserved": mechanism["replace"]["old_value_preserved"],
                },
                "conflict": {
                    "rejected_batches": mechanism["conflict"]["rejected_batches"],
                    "rollbacks": mechanism["conflict"]["rollbacks"],
                    "digest_unchanged": mechanism["conflict"]["digest_unchanged"],
                },
                "abstain": {
                    "unknown": abstain["unknown_tally"],
                    "ambiguous_pairs": abstain["ambiguous_pair_tally"],
                    "ambiguous_sentences": abstain["ambiguous_sentence_tally"],
                    "sentence_direct": abstain["sentence_direct_tally"],
                    "false_answers": abstain["false_answers"],
                },
                "cost": report["cost"],
            },
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
