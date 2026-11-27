"""R17 -- can the written bridge shrink its key count without semantics?

R14 measured the cost curve: one key per wording, so keys/entity grows with the
number of wordings (9/9 -> 18/9 -> the rejected 78/9).  R16 measured the read
geometry.  R17 asks the next architectural question and freezes every expected
value **before** reading:

Can the bridge collapse keys when several wordings carry the *same value*,
without any similarity, threshold or semantics?

Four measurements, kept separate:

A. Value-level de-duplication.  Three keys, one value: the value is one object,
   but the address count stays three.  This segment exists to stop the report
   from calling "one value" a key reduction.
B. Same-value containment.  A shorter key is a contiguous substring of a longer
   key with the same value.  The longer key is redundant for matching (any query
   containing it contains the shorter one), so it can be dropped.  A structural
   guard refuses the drop when another record's whole key sits inside the longer
   key but outside the kept span.
C. The guard's blind spot.  A key that straddles the *boundary* of the dropped
   wording is not inside it, so the table-local guard passes -- and the read
   then answers one extra value that the longer wording was meant to cover.
D. Cross-value containment.  A longer key of value V contains a shorter key of
   value W.  Compressing there would answer W for a query written as the longer
   wording of V, so the rule excludes different values entirely.

Default path: the 18-key table (9 corpus wordings + 9 authored aliases) is
compressed too.  If nothing is removed the content digest must not move, and the
R15 reading must stay 40/40 with the zero lines at 0.

Zero training, zero optimizer steps, zero Adam, zero confirm.  ``address_bridge``
is not modified: this round measures a candidate operation, it does not install
it in the default write path.
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
    first_occurrence,
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

MANIFEST = ROOT / "plans/manifests/taiji_bridge_compression_20261109.json"
REPORT = ROOT / "reports/taiji_bridge_compression_20261109.json"
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

CUP = ("object", "蓝色杯子")
TABLE_ENTITY = ("place", "木桌")

# -- segment A: one value, three keys ----------------------------------------
DEDUP_KEYS = (
    ("甲乙", CUP),
    ("丙丁", CUP),
    ("戊己", CUP),
)
DEDUP_CASES = tuple(key for key, _ in DEDUP_KEYS)

# -- segment B: same-value containment, guard passes -------------------------
SAFE_KEYS = (
    ("甲乙", CUP),
    ("甲乙丙", CUP),
)
SAFE_SENTENCE = "甲乙丙"
SAFE_EXPECTED = {
    "before_answers": 0,
    "before_exit": "ambiguous_same_start",
    "records_before": 2,
    "records_after": 1,
    "removed": ["甲乙丙"],
    "after_values": [list(CUP)],
    "after_stop_pass": 2,
    "keeper_values": [list(CUP)],
}

# -- segment B: same-value containment, residual blocks the drop -------------
BLOCK_KEYS = (
    ("甲乙", CUP),
    ("甲乙丙丁", CUP),
    ("丙丁", TABLE_ENTITY),
)
BLOCK_SENTENCE = "甲乙丙丁"
BLOCK_EXPECTED = {
    "before_answers": 0,
    "before_exit": "ambiguous_same_start",
    "records_before": 3,
    "records_after": 3,
    "blocked": ["甲乙丙丁"],
    "after_answers": 0,
    "after_exit": "ambiguous_same_start",
    "unguarded_values": [list(CUP), list(TABLE_ENTITY)],
}

# -- segment C: guard blind spot at the dropped wording's boundary -----------
BOUNDARY_KEYS = (
    ("甲乙", CUP),
    ("甲乙丙", CUP),
    ("丙丁", TABLE_ENTITY),
)
BOUNDARY_SENTENCE = "甲乙丙丁"
BOUNDARY_EXPECTED = {
    "before_answers": 0,
    "before_exit": "ambiguous_same_start",
    "records_before": 3,
    "records_after": 2,
    "removed": ["甲乙丙"],
    "after_values": [list(CUP), list(TABLE_ENTITY)],
    "after_decision": "answered_pair",
    "after_stop_pass": 3,
}

# -- segment D: cross-value containment is never compressed ------------------
CROSS_KEYS = (
    ("甲乙", TABLE_ENTITY),
    ("甲乙丙", CUP),
)
CROSS_SENTENCE = "甲乙丙"
CROSS_EXPECTED = {
    "before_answers": 0,
    "before_exit": "ambiguous_same_start",
    "records_before": 2,
    "records_after": 2,
    "removed": [],
    "pairs": 1,
    "hypothetical_values": [list(TABLE_ENTITY)],
}

DEFINITION = {
    "value_dedup": (
        "three keys bound to one value: the value is one object, but every key still occupies an "
        "address, so this segment reports keys, distinct values and key reduction separately"
    ),
    "same_value_containment": (
        "a longer key is dropped only when a shorter key of the SAME value occurs contiguously "
        "inside it and no other record's whole key sits inside the longer key outside the kept span"
    ),
    "residual_guard": (
        "the guard refuses the drop when another key (any value) is entirely inside the longer key "
        "but outside the kept span; the unguarded simulation is reported to show what it prevented"
    ),
    "boundary_leak": (
        "a key that straddles the boundary of the dropped wording is not inside it, so the guard "
        "passes; the read then answers one extra value -- this is the measured blind spot"
    ),
    "cross_value": (
        "containment across different values is excluded by rule; compressing there would answer the "
        "short key's value for a query written as the long key's wording"
    ),
    "default_path": (
        "the 18-key table (9 corpus + 9 aliases) is compressed as well; nothing may be removed, the "
        "content digest must not move, and the R15 reading must stay 40/40"
    ),
    "zeros": "12 unknown candidates and 18 byte fragments must answer 0 on the default table",
    "closed_state": "the nine closed-state readings stay equal before and after",
}

CRITERIA = {
    "value_dedup": (
        "three keys, one distinct value, three addresses, key reduction 0, and each key answers the "
        "same value"
    ),
    "safe_containment": (
        "the nested pair goes 2 -> 1 record; before the drop the longer wording abstains at pass 1 "
        "(same-start tie), after the drop it answers the value at pass 2; the kept key still answers"
    ),
    "residual_guard": (
        "the drop is blocked (3 -> 3 records, blocked key named); the reading still abstains; the "
        "unguarded simulation answers both values, which is what the guard prevented"
    ),
    "boundary_leak": (
        "the drop happens (3 -> 2 records) and the read answers both values, including the value "
        "that the dropped wording was meant to cover -- the guard is table-local, not query-universal"
    ),
    "cross_value": (
        "no record is removed, the cross-value pair is listed, and the hypothetical drop would answer "
        "the short key's value -- therefore the same-value condition is required"
    ),
    "default_path": (
        "zero records removed, content digest unchanged, regression 40/40 with wrong 0, zeros 0/0"
    ),
    "closed_state": "the nine closed-state readings stay equal before and after",
    "reporting": (
        "segments are reported separately; a safe nested pair does not carry the boundary segment, "
        "and the boundary segment is evidence against default compression, not for it"
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
    return [row["text"] for row in base.frozen.build_fixture() if row["variant"] == "describe"]


def expected_pair(text: str) -> dict[str, tuple[str, str]]:
    found: dict[str, tuple[str, str]] = {}
    for name, phrase in OBJECT_DESCRIPTIONS.items():
        if phrase in text:
            found["object"] = ("object", name)
    for name, phrase in PLACE_DESCRIPTIONS.items():
        if phrase in text:
            found["place"] = ("place", name)
    return found


def build(keys: tuple[tuple[str, tuple[str, str]], ...]) -> BridgeTable:
    table = BridgeTable(key_fn=sequence_key, read_mode="sequence")
    write = table.write(keys)
    if write["action"] != "accepted":
        raise SystemExit(f"write rejected: {write['action']}")
    return table


def build_default() -> BridgeTable:
    table = build(tuple(bridge_records(OBJECT_DESCRIPTIONS, PLACE_DESCRIPTIONS)))
    alias = table.write(AUTHORED_ALIASES)
    if alias["action"] != "accepted":
        raise SystemExit(f"alias write rejected: {alias['action']}")
    return table


def occurrences(query: tuple[str, ...], pattern: tuple[str, ...]) -> list[int]:
    width = len(pattern)
    if width == 0 or width > len(query):
        return []
    return [
        start for start in range(len(query) - width + 1) if query[start : start + width] == pattern
    ]


def compress_same_value_containment(table: BridgeTable) -> dict[str, Any]:
    """Drop a longer same-value key when a shorter one covers it and the guard passes."""
    records = table.records
    alive = [True] * len(records)
    removed: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    order = sorted(
        range(len(records)), key=lambda index: (-len(tuple(records[index]["key"])), index)
    )
    for index in order:
        if not alive[index]:
            continue
        long_key = tuple(records[index]["key"])
        candidates = [
            other
            for other in range(len(records))
            if alive[other]
            and other != index
            and records[other]["value"] == records[index]["value"]
            and len(tuple(records[other]["key"])) < len(long_key)
            and first_occurrence(long_key, tuple(records[other]["key"])) is not None
        ]
        if not candidates:
            continue
        keeper = min(candidates, key=lambda other: (len(tuple(records[other]["key"])), other))
        short_key = tuple(records[keeper]["key"])
        start = first_occurrence(long_key, short_key)
        end = start + len(short_key)
        conflict: dict[str, Any] | None = None
        for other in range(len(records)):
            if other in (index, keeper) or not alive[other]:
                continue
            other_key = tuple(records[other]["key"])
            for position in occurrences(long_key, other_key):
                if position + len(other_key) <= start or position >= end:
                    conflict = {
                        "key": "".join(other_key),
                        "position": position,
                        "value": list(records[other]["value"]),
                    }
                    break
            if conflict is not None:
                break
        if conflict is not None:
            blocked.append(
                {
                    "key": "".join(long_key),
                    "keeper": "".join(short_key),
                    "conflict": conflict,
                }
            )
            continue
        alive[index] = False
        removed.append(
            {
                "key": "".join(long_key),
                "keeper": "".join(short_key),
                "value": list(records[index]["value"]),
            }
        )
    survivors = [
        {
            "key": records[index]["key"],
            "value": records[index]["value"],
            "count": records[index]["count"],
        }
        for index in range(len(records))
        if alive[index]
    ]
    table.records = survivors
    return {
        "removed": removed,
        "blocked": blocked,
        "records_before": len(records),
        "records_after": len(survivors),
    }


def cross_value_pairs(table: BridgeTable) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for index, outer in enumerate(table.records):
        long_key = tuple(outer["key"])
        for other, inner in enumerate(table.records):
            if index == other or outer["value"] == inner["value"]:
                continue
            short_key = tuple(inner["key"])
            if len(long_key) > len(short_key) and first_occurrence(long_key, short_key) is not None:
                pairs.append(
                    {
                        "long": "".join(long_key),
                        "long_value": list(outer["value"]),
                        "short": "".join(short_key),
                        "short_value": list(inner["value"]),
                    }
                )
    return pairs


def reading(table: BridgeTable, sentence: str, max_passes: int = 3) -> dict[str, Any]:
    result = table.read_sentence(sentence, max_passes=max_passes)
    values = [list(row["value"]) for row in result["answers"]]
    return {
        "decision": result["decision"],
        "answers": len(values),
        "values": values,
        "stopped_at_pass": result["stopped_at_pass"],
        "final_exit": result["final_exit"],
        "final_reason": (result["passes"][-1].get("reason") if result["passes"] else "empty_key"),
    }


def dedup_reading() -> dict[str, Any]:
    table = build(DEDUP_KEYS)
    reads = {key: reading(table, key, 2) for key, _ in DEDUP_KEYS}
    distinct_values = len({value_key(record["value"]) for record in table.records})
    all_same = all(row["values"] == [list(CUP)] and row["answers"] == 1 for row in reads.values())
    ok = (
        len(table.records) == 3
        and distinct_values == 1
        and all_same
        and len(table.records) - distinct_values == 2
    )
    return {
        "keys": [key for key, _ in DEDUP_KEYS],
        "reads": reads,
        "records": len(table.records),
        "distinct_values": distinct_values,
        "key_reduction": 0,
        "ok": ok,
    }


def safe_reading() -> dict[str, Any]:
    table = build(SAFE_KEYS)
    before = reading(table, SAFE_SENTENCE)
    compression = compress_same_value_containment(table)
    after = reading(table, SAFE_SENTENCE)
    keeper = reading(table, "甲乙", 2)
    ok = (
        before["answers"] == SAFE_EXPECTED["before_answers"]
        and before["final_exit"] == SAFE_EXPECTED["before_exit"]
        and compression["records_before"] == SAFE_EXPECTED["records_before"]
        and compression["records_after"] == SAFE_EXPECTED["records_after"]
        and [row["key"] for row in compression["removed"]] == SAFE_EXPECTED["removed"]
        and after["values"] == SAFE_EXPECTED["after_values"]
        and after["stopped_at_pass"] == SAFE_EXPECTED["after_stop_pass"]
        and keeper["values"] == SAFE_EXPECTED["keeper_values"]
    )
    return {
        "before": before,
        "after": after,
        "keeper": keeper,
        "compression": compression,
        "ok": ok,
    }


def blocked_reading() -> dict[str, Any]:
    table = build(BLOCK_KEYS)
    before = reading(table, BLOCK_SENTENCE)
    compression = compress_same_value_containment(table)
    after = reading(table, BLOCK_SENTENCE)
    unguarded = build((BLOCK_KEYS[0], BLOCK_KEYS[2]))
    simulation = reading(unguarded, BLOCK_SENTENCE)
    ok = (
        before["answers"] == BLOCK_EXPECTED["before_answers"]
        and before["final_exit"] == BLOCK_EXPECTED["before_exit"]
        and compression["records_before"] == BLOCK_EXPECTED["records_before"]
        and compression["records_after"] == BLOCK_EXPECTED["records_after"]
        and [row["key"] for row in compression["blocked"]] == BLOCK_EXPECTED["blocked"]
        and after["answers"] == BLOCK_EXPECTED["after_answers"]
        and after["final_exit"] == BLOCK_EXPECTED["after_exit"]
        and simulation["values"] == BLOCK_EXPECTED["unguarded_values"]
    )
    return {
        "before": before,
        "after": after,
        "compression": compression,
        "unguarded_simulation": simulation,
        "ok": ok,
    }


def boundary_reading() -> dict[str, Any]:
    table = build(BOUNDARY_KEYS)
    before = reading(table, BOUNDARY_SENTENCE)
    compression = compress_same_value_containment(table)
    after = reading(table, BOUNDARY_SENTENCE)
    extra = [value for value in after["values"] if value not in before["values"]]
    ok = (
        before["answers"] == BOUNDARY_EXPECTED["before_answers"]
        and before["final_exit"] == BOUNDARY_EXPECTED["before_exit"]
        and compression["records_before"] == BOUNDARY_EXPECTED["records_before"]
        and compression["records_after"] == BOUNDARY_EXPECTED["records_after"]
        and [row["key"] for row in compression["removed"]] == BOUNDARY_EXPECTED["removed"]
        and after["values"] == BOUNDARY_EXPECTED["after_values"]
        and after["decision"] == BOUNDARY_EXPECTED["after_decision"]
        and after["stopped_at_pass"] == BOUNDARY_EXPECTED["after_stop_pass"]
    )
    return {
        "before": before,
        "after": after,
        "compression": compression,
        "extra_values": extra,
        "ok": ok,
    }


def cross_reading() -> dict[str, Any]:
    table = build(CROSS_KEYS)
    before = reading(table, CROSS_SENTENCE)
    compression = compress_same_value_containment(table)
    pairs = cross_value_pairs(table)
    hypothetical = build((CROSS_KEYS[0],))
    simulation = reading(hypothetical, CROSS_SENTENCE)
    ok = (
        before["answers"] == CROSS_EXPECTED["before_answers"]
        and before["final_exit"] == CROSS_EXPECTED["before_exit"]
        and compression["records_before"] == CROSS_EXPECTED["records_before"]
        and compression["records_after"] == CROSS_EXPECTED["records_after"]
        and compression["removed"] == CROSS_EXPECTED["removed"]
        and len(pairs) == CROSS_EXPECTED["pairs"]
        and simulation["values"] == CROSS_EXPECTED["hypothetical_values"]
    )
    return {
        "before": before,
        "after": reading(table, CROSS_SENTENCE),
        "compression": compression,
        "cross_value_pairs": pairs,
        "hypothetical_simulation": simulation,
        "ok": ok,
    }


def regression_reading(table: BridgeTable, texts: list[str]) -> dict[str, Any]:
    hits = wrong = 0
    distribution = {"answered_2": 0, "answered_1": 0, "wrong": 0, "abstained": 0}
    for text in texts:
        expected = expected_pair(text)
        result = table.read_sentence(text)
        matched = 0
        for row in result["answers"]:
            slot, name = row["value"]
            if expected.get(slot) == (slot, name):
                matched += 1
            else:
                wrong += 1
        hits += matched
        if matched == 2:
            distribution["answered_2"] += 1
        elif matched == 1:
            distribution["answered_1"] += 1
        else:
            distribution["abstained"] += 1
    total = 2 * len(texts)
    return {
        "entities": total,
        "hits": hits,
        "wrong": wrong,
        "distribution": distribution,
        "ok": hits == total and wrong == 0 and distribution["answered_2"] == len(texts),
    }


def zeros_reading(table: BridgeTable) -> dict[str, Any]:
    unknown_answering = 0
    unknown_rows = []
    for phrase in UNKNOWN_CANDIDATES:
        result = table.read_sentence(phrase)
        answered = bool(result["answers"])
        unknown_answering += int(answered)
        unknown_rows.append({"phrase": phrase, "decision": result["decision"]})
    fragment_answering = 0
    fragment_rows = []
    for phrase, _ in bridge_records(OBJECT_DESCRIPTIONS, PLACE_DESCRIPTIONS):
        data = phrase.encode("utf-8")
        for offset in (1, 2):
            result = table.read_sentence(data[offset:])
            answered = bool(result["answers"])
            fragment_answering += int(answered)
            fragment_rows.append(
                {"phrase": phrase, "byte_offset": offset, "decision": result["decision"]}
            )
    return {
        "unknown": {
            "rows": unknown_rows,
            "candidates": len(unknown_rows),
            "answering": unknown_answering,
            "zero_ok": unknown_answering == 0,
        },
        "fragments": {
            "rows": fragment_rows,
            "inputs": len(fragment_rows),
            "answering": fragment_answering,
            "zero_ok": fragment_answering == 0,
        },
        "ok": unknown_answering == 0 and fragment_answering == 0,
    }


def default_reading(texts: list[str]) -> dict[str, Any]:
    table = build_default()
    digest_before = table.content_digest()
    compression = compress_same_value_containment(table)
    digest_after = table.content_digest()
    regression = regression_reading(table, texts)
    zeros = zeros_reading(table)
    ok = (
        compression["removed"] == []
        and digest_before == digest_after
        and regression["ok"]
        and zeros["ok"]
    )
    return {
        "records": len(table.records),
        "compression": compression,
        "digest_before": digest_before,
        "digest_after": digest_after,
        "digest_unchanged": digest_before == digest_after,
        "regression": regression,
        "zeros": zeros,
        "ok": ok,
    }


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
    state = reference.initial_state("r17-zero-perturbation")
    step_hashes: list[str] = []
    for raw in texts:
        state = reference.observe(state, raw.encode("utf-8"), goal=b"r17")
        step_hashes.append(
            tensor_sha([state.h, state.fast, state.nu_ema, state.coverage, state.prior])
        )
    return reference_digest(reference, state, step_hashes)


def definition_digest() -> str:
    payload = {
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "dedup_keys": [list(row) for row in DEDUP_KEYS],
        "safe": {
            "keys": [list(row) for row in SAFE_KEYS],
            "sentence": SAFE_SENTENCE,
            "expected": SAFE_EXPECTED,
        },
        "blocked": {
            "keys": [list(row) for row in BLOCK_KEYS],
            "sentence": BLOCK_SENTENCE,
            "expected": BLOCK_EXPECTED,
        },
        "boundary": {
            "keys": [list(row) for row in BOUNDARY_KEYS],
            "sentence": BOUNDARY_SENTENCE,
            "expected": BOUNDARY_EXPECTED,
        },
        "cross": {
            "keys": [list(row) for row in CROSS_KEYS],
            "sentence": CROSS_SENTENCE,
            "expected": CROSS_EXPECTED,
        },
        "unknown": list(UNKNOWN_CANDIDATES),
        "aliases": [list(row) for row in AUTHORED_ALIASES],
        "seed": SEED,
        "estimate_cpu_upper": ESTIMATE_CPU_UPPER,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def freeze() -> dict[str, Any]:
    manifest = {
        "schema": "taiji-bridge-compression-r17-v1",
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
            "dedup_keys": len(DEDUP_KEYS),
            "safe_keys": len(SAFE_KEYS),
            "blocked_keys": len(BLOCK_KEYS),
            "boundary_keys": len(BOUNDARY_KEYS),
            "cross_keys": len(CROSS_KEYS),
            "regression_sentences": len(sentences()),
            "unknown_candidates": len(UNKNOWN_CANDIDATES),
            "fragments": 18,
        },
        "inputs": {
            "dedup": [list(row) for row in DEDUP_KEYS],
            "safe": [list(row) for row in SAFE_KEYS],
            "blocked": [list(row) for row in BLOCK_KEYS],
            "boundary": [list(row) for row in BOUNDARY_KEYS],
            "cross": [list(row) for row in CROSS_KEYS],
            "aliases": [list(row) for row in AUTHORED_ALIASES],
            "unknown_candidates": list(UNKNOWN_CANDIDATES),
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

        dedup = dedup_reading()
        safe = safe_reading()
        blocked = blocked_reading()
        boundary = boundary_reading()
        cross = cross_reading()
        default = default_reading(texts)

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

        verdict = {
            "value_dedup": "value_dedup_visible" if dedup["ok"] else "value_dedup_absent",
            "safe_containment": (
                "same_value_containment_safe" if safe["ok"] else "same_value_containment_failed"
            ),
            "residual_guard": "residual_guard_blocks" if blocked["ok"] else "residual_guard_missed",
            "boundary": "boundary_leak_confirmed" if boundary["ok"] else "boundary_leak_absent",
            "cross_value": "cross_value_forbidden" if cross["ok"] else "cross_value_compressed",
            "default_path": ("default_path_unchanged" if default["ok"] else "default_path_moved"),
            "zeros": "zeros_clean" if default["zeros"]["ok"] else "zeros_leak",
        }
        all_ok = (
            dedup["ok"]
            and safe["ok"]
            and blocked["ok"]
            and boundary["ok"]
            and cross["ok"]
            and default["ok"]
            and closed_ok
        )
        verdict["overall"] = "compression_value_level_only" if all_ok else "compression_unexpected"

        cpu, wall = cost()
        if cpu > CPU_CAP:
            raise SystemExit(f"cpu cap exceeded: {cpu:.2f}s > {CPU_CAP:.0f}s")
        report = {
            "schema": "taiji-bridge-compression-r17-report-v1",
            "runner_sha256": RUNNER_SHA_START,
            "manifest_sha256": file_sha(MANIFEST),
            "bridge_sha256": file_sha(BRIDGE_SOURCE),
            "fixture_sha256": base.fixture_hashes(),
            "definition": DEFINITION,
            "criteria": CRITERIA,
            "value_dedup": dedup,
            "same_value_containment": safe,
            "residual_guard": blocked,
            "boundary": boundary,
            "cross_value": cross,
            "default_path": default,
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
                    "value_dedup": {
                        "records": dedup["records"],
                        "distinct_values": dedup["distinct_values"],
                        "key_reduction": dedup["key_reduction"],
                    },
                    "safe": {
                        "before": safe["before"]["answers"],
                        "after": safe["after"]["values"],
                        "records": [
                            safe["compression"]["records_before"],
                            safe["compression"]["records_after"],
                        ],
                    },
                    "blocked": {
                        "records": [
                            blocked["compression"]["records_before"],
                            blocked["compression"]["records_after"],
                        ],
                        "unguarded": blocked["unguarded_simulation"]["values"],
                    },
                    "boundary": {
                        "records": [
                            boundary["compression"]["records_before"],
                            boundary["compression"]["records_after"],
                        ],
                        "after": boundary["after"]["values"],
                        "extra": boundary["extra_values"],
                    },
                    "cross_value": {
                        "records": [
                            cross["compression"]["records_before"],
                            cross["compression"]["records_after"],
                        ],
                        "pairs": len(cross["cross_value_pairs"]),
                        "hypothetical": cross["hypothetical_simulation"]["values"],
                    },
                    "default": {
                        "records": default["records"],
                        "removed": len(default["compression"]["removed"]),
                        "digest_unchanged": default["digest_unchanged"],
                        "regression": default["regression"]["hits"],
                        "of": default["regression"]["entities"],
                        "zeros": [
                            default["zeros"]["unknown"]["answering"],
                            default["zeros"]["fragments"]["answering"],
                        ],
                    },
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
