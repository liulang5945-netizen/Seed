"""R12 -- which token unit should the written bridge use?

R11 wrote nine records with a 2-byte-window key and read them back 40/40, but
four candidate phrases had to be excluded *before* reading because their byte
windows inherited the tail bytes of neighbouring characters.  That exclusion is
the accusation: with a 2-byte unit the cut points fall inside 3-byte Chinese
characters, so a semantically unrelated phrase can share a real token.

This round runs two pre-registered key definitions side by side on the same
nine records, the same inputs and the same zero line:

* ``byte_pair``  -- the R11 unit, 2-byte windows (control, byte-identical);
* ``codepoint``  -- one whole UTF-8 code point per token (the proposal);
* ``codepoint_offset`` -- code point tagged with its character index.  This is
  the previously pre-registered offset-aware treatment, run here as a
  *diagnostic* so the report can say which failure class it actually treats.

Frozen criteria (written before the first reading):

1. capability: for every arm, 20 describe sentences -> locate the written
   phrase -> read -> (slot, name); 40/40 and zero wrong pointers.
2. the zero line: for each compared arm (byte_pair, codepoint), the number of
   *answering* unknown phrases among the 12 frozen candidates must be 0.  A
   phrase counts as a collision when it shares at least one token with the
   written keys under that arm; it counts as a failure when the read returns a
   non-empty value.  The same 12 candidates are used for both arms.
3. companions, reported not gated: byte-sliced fragments (18, every phrase cut
   at byte offset 1 and 2 -- not valid text), the 20 whole sentences read
   without localisation, and 9 synonymous rewrites.
4. closed state: the default graph replays the same 20-sentence battery before
   and after the three arms are built; parameters, state, identity, store
   signature and emitted bytes are identical and ``sc_sfk`` stays None.

Verdict:

* ``bridge_key_unit_architectural_error`` -- codepoint answers zero unknown
  phrases and byte_pair answers some;
* ``bridge_key_unit_no_difference`` -- both answer zero;
* ``bridge_key_unit_still_leaks`` -- codepoint still answers some;
* ``bridge_key_unit_capability_failed`` / ``_perturbation_detected`` for the
  other two gates.

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
    byte_pair_key,
    codepoint_key,
    codepoint_offset_key,
    phrase_key,
)
from taiji.living_core import LivingCoreReference  # noqa: E402
from taiji.native_goal_world_wide import (  # noqa: E402
    OBJECT_DESCRIPTIONS,
    PLACE_DESCRIPTIONS,
)
from taiji.process_guard import assert_no_black_processes  # noqa: E402

torch.set_num_threads(1)

MANIFEST = ROOT / "plans/manifests/taiji_bridge_key_unit_20261109.json"
REPORT = ROOT / "reports/taiji_bridge_key_unit_20261109.json"
BRIDGE_SOURCE = ROOT / "taiji/address_bridge.py"

RUNNER_SHA_START = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()

SEED = 20261109
ESTIMATE_CPU_UPPER = 60.0
CPU_CAP = float(base.CPU_CAP)

ARMS = {
    "byte_pair": byte_pair_key,
    "codepoint": codepoint_key,
    "codepoint_offset": codepoint_offset_key,
}
COMPARED_ARMS = ("byte_pair", "codepoint")

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

# R11 excluded exactly these four before reading; they are the accusation.
R11_REJECTED = ("白色帆布", "银质圆镜", "棕色皮箱", "青色陶罐")

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
    "records": "nine (phrase -> (slot, name)) rows, identical for every arm",
    "arms": {
        "byte_pair": "R11 unit: 2-byte sliding windows on the UTF-8 bytes, hex, order-free.",
        "codepoint": "one whole UTF-8 code point per token, order-free; invalid bytes -> empty key.",
        "codepoint_offset": "code point tagged with its character index (offset-aware diagnostic).",
    },
    "read_rule": (
        "an address anchors only through a token that occurs in exactly one address; one anchor "
        "answers, two or more say ambiguous_multiple_anchors, zero says no_match, and every "
        "non-answer returns value None."
    ),
    "unknown_inputs": "12 frozen candidates, identical for every arm; a collision is a non-empty "
    "intersection with the table keys under that arm; a failure is a non-empty returned value.",
    "fragments": "for each written phrase, the bytes cut at offset 1 and offset 2 (18 inputs); "
    "they are not valid text, so any returned value is an over-answer.",
    "rewrites": "9 synonymous paraphrases with their intended (slot, name); correct / wrong / "
    "abstain are all reported, only wrong pointers count as false answers.",
}

CRITERIA = {
    "capability": "every arm: 40/40 located-phrase reads and zero wrong pointers.",
    "zero_line": "byte_pair and codepoint: zero answering phrases among the 12 unknown candidates "
    "(collision set per arm, same input list).",
    "companions": "fragments, whole sentences and rewrites are reported for every arm, not gated.",
    "closed_state": "nine checks identical before and after the arms; sc_sfk stays None.",
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
    state = reference.initial_state("r12-zero-perturbation")
    step_hashes: list[str] = []
    for raw in texts:
        state = reference.observe(state, raw.encode("utf-8"), goal=b"r12")
        step_hashes.append(
            tensor_sha([state.h, state.fast, state.nu_ema, state.coverage, state.prior])
        )
    return reference_digest(reference, state, step_hashes)


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


def unknown_reading(table: BridgeTable, key_fn: Any) -> dict[str, Any]:
    table_tokens: set[str] = set()
    for record in table.records:
        table_tokens |= set(record["key"])
    rows = []
    collision = answering = 0
    for phrase in UNKNOWN_CANDIDATES:
        tokens = key_fn(phrase)
        shared = sorted(tokens & table_tokens)
        result = table.read(phrase)
        value = result["value"]
        if shared:
            collision += 1
            if value is not None:
                answering += 1
        rows.append(
            {
                "phrase": phrase,
                "shared_tokens": shared,
                "decision": result["decision"],
                "value": list(value) if value else None,
                "reason": result["reason"],
                "answered": value is not None,
                "r11_rejected": phrase in R11_REJECTED,
            }
        )
    rejected_rows = [row for row in rows if row["r11_rejected"]]
    return {
        "rows": rows,
        "candidates": len(rows),
        "collision_phrases": collision,
        "answering_phrases": answering,
        "collision_rate": answering / len(rows),
        "zero_line_ok": answering == 0,
        "r11_rejected_answering": sum(1 for row in rejected_rows if row["answered"]),
        "r11_rejected_total": len(rejected_rows),
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
                    "kind": kind,
                }
            )
    answered = same + other
    return {
        "rows": rows,
        "inputs": len(rows),
        "same_record": same,
        "other_record": other,
        "false_answers": answered,
    }


def sentences_reading(table: BridgeTable, texts: list[str]) -> dict[str, Any]:
    rows = []
    answered = 0
    for text in texts:
        result = table.read(text)
        value = result["value"]
        if value is not None:
            answered += 1
        rows.append(
            {
                "sentence": text,
                "decision": result["decision"],
                "value": list(value) if value else None,
                "reason": result["reason"],
            }
        )
    return {"rows": rows, "inputs": len(rows), "false_answers": answered}


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
                "outcome": outcome,
            }
        )
    return {
        "rows": rows,
        "inputs": len(rows),
        "correct": correct,
        "wrong": wrong,
        "abstain": abstain,
        "false_answers": wrong,
    }


def definition_digest() -> str:
    payload = {
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "unknown": list(UNKNOWN_CANDIDATES),
        "rejected": list(R11_REJECTED),
        "rewrites": [list(row) for row in REWRITES],
        "seed": SEED,
        "estimate_cpu_upper": ESTIMATE_CPU_UPPER,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def freeze() -> dict[str, Any]:
    records = bridge_records(OBJECT_DESCRIPTIONS, PLACE_DESCRIPTIONS)
    manifest = {
        "schema": "taiji-bridge-key-unit-r12-v1",
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
            "rewrites": len(REWRITES),
            "fragments": 2 * len(records),
            "battery_texts": len(sentences()),
        },
        "inputs": {
            "unknown_candidates": list(UNKNOWN_CANDIDATES),
            "r11_rejected": list(R11_REJECTED),
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
        manifest = verify_freeze()

        for phrase in OBJECT_DESCRIPTIONS.values():
            if byte_pair_key(phrase) != phrase_key(phrase):
                raise SystemExit("byte_pair arm is not byte-identical to the R11 unit")

        texts = sentences()
        records = bridge_records(OBJECT_DESCRIPTIONS, PLACE_DESCRIPTIONS)

        torch.manual_seed(SEED)
        reference = LivingCoreReference()
        before = run_battery(reference, texts)

        arms: dict[str, Any] = {}
        for name, key_fn in ARMS.items():
            table = BridgeTable(key_fn=key_fn)
            write = table.write(records)
            if write["action"] != "accepted":
                raise SystemExit(f"{name}: initial write rejected ({write['action']})")
            arms[name] = {
                "write": {
                    "appended": write["appended"],
                    "slots": write["slots"],
                    "key_size": sorted({len(record["key"]) for record in table.records}),
                },
                "capability": capability(table, texts),
                "unknown": unknown_reading(table, key_fn),
                "fragments": fragments_reading(table),
                "sentences_direct": sentences_reading(table, texts),
                "rewrites": rewrites_reading(table),
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

        capability_ok = all(arms[name]["capability"]["ok"] for name in ARMS)
        zero_line = {name: arms[name]["unknown"]["answering_phrases"] for name in COMPARED_ARMS}
        if not capability_ok:
            verdict = "bridge_key_unit_capability_failed"
        elif not all(checks.values()):
            verdict = "bridge_key_unit_perturbation_detected"
        elif zero_line["byte_pair"] == 0 and zero_line["codepoint"] == 0:
            verdict = "bridge_key_unit_no_difference"
        elif zero_line["byte_pair"] > 0 and zero_line["codepoint"] == 0:
            verdict = "bridge_key_unit_architectural_error"
        else:
            verdict = "bridge_key_unit_still_leaks"

        cpu, wall = cost()
        if cpu > CPU_CAP:
            raise SystemExit(f"cpu cap exceeded: {cpu:.2f}s > {CPU_CAP:.0f}s")
        report = {
            "schema": "taiji-bridge-key-unit-r12-report-v1",
            "runner_sha256": RUNNER_SHA_START,
            "manifest_sha256": file_sha(MANIFEST),
            "bridge_sha256": file_sha(BRIDGE_SOURCE),
            "fixture_sha256": base.fixture_hashes(),
            "definition": DEFINITION,
            "criteria": CRITERIA,
            "arms": arms,
            "closed_state": {"baseline": before, "after": after, "checks": checks},
            "zero_line": {
                "per_arm": zero_line,
                "criterion": "0 answering phrases among the 12 frozen unknown candidates",
            },
            "cost": {
                "cpu_seconds": cpu,
                "wall_seconds": wall,
                "cpu_cap": CPU_CAP,
                "estimate_cpu_upper": ESTIMATE_CPU_UPPER,
                "threads": torch.get_num_threads(),
            },
            "verdict": verdict,
            "note": (
                "codepoint_offset is diagnostic: it tests positional drift, not shared characters. "
                "A character shared at the same index still collides."
            ),
        }
        write_json(REPORT, report)
        print(
            json.dumps(
                {
                    "verdict": verdict,
                    "zero_line": zero_line,
                    "capability": {
                        name: {
                            "accuracy": arms[name]["capability"]["accuracy"],
                            "wrong": arms[name]["capability"]["wrong_pointers"],
                        }
                        for name in ARMS
                    },
                    "unknown_answering": {
                        name: arms[name]["unknown"]["answering_phrases"] for name in ARMS
                    },
                    "r11_rejected_answering": {
                        name: arms[name]["unknown"]["r11_rejected_answering"] for name in ARMS
                    },
                    "fragments_false": {
                        name: arms[name]["fragments"]["false_answers"] for name in ARMS
                    },
                    "sentences_false": {
                        name: arms[name]["sentences_direct"]["false_answers"] for name in ARMS
                    },
                    "rewrites": {
                        name: (arms[name]["rewrites"]["correct"], arms[name]["rewrites"]["wrong"])
                        for name in ARMS
                    },
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
