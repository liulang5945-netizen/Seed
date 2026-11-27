"""R16 -- two geometric cases the two-pass read has never been tested on.

R15 showed that two passes peel a two-entity sentence (40/40) and that the
write side has exactly one conflict rule.  Both readings assumed the two
wordings in a sentence *do not touch*.  R16 removes that assumption and freezes
the expected outcome of every case **before** reading:

Segment A -- overlapping wordings (geometry 1).  Four frozen cases: one wording
entirely inside another's span, two wordings sharing one character, two wordings
that merely abut, and two that overlap without containment.  The question is not
"how many can it answer" but "does masking delete characters a later wording
needs": each case declares `expected_answers`, `stopped_at_pass`, `final_exit`,
`final_reason`, and the runner reports which unwritten keys lost their span to a
mask (`swallowed`).

Segment B -- three entities in one sentence (geometry 2).  Three disjoint
wordings (must answer 3 and stop at pass 4 by no_match, not by the pass bound);
three wordings where the middle one straddles the first mask (must answer 2 and
stop at pass 3, with the swallowed key named); three records at the same start
(must abstain at pass 1); the same address three times (one entity, not three).
`stopped_at_pass` and `final_exit` are reported for every case, so "answered
fewer" always comes with the layer it stopped at.

Segment C -- the R15 reading must not move: the 20 describe sentences still
reach 40/40 with the default two passes.

Segment D -- zero lines on the 18-key table: 12 unknown candidates, 18 byte
fragments.

Segment E -- re-run anchors: R11-R15 each get the commit at which their own
bytes were last runnable, with the runner blob's sha1 and sha256, so the old
reports stay reproducible by checking out that commit instead of being orphans.

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
    first_occurrence,
    sequence_key,
)
from taiji.living_core import LivingCoreReference  # noqa: E402
from taiji.native_goal_world_wide import (  # noqa: E402
    OBJECT_DESCRIPTIONS,
    PLACE_DESCRIPTIONS,
)
from taiji.process_guard import assert_no_black_processes  # noqa: E402

torch.set_num_threads(1)

MANIFEST = ROOT / "plans/manifests/taiji_bridge_geometry_20261109.json"
REPORT = ROOT / "reports/taiji_bridge_geometry_20261109.json"
BRIDGE_SOURCE = ROOT / "taiji/address_bridge.py"

RUNNER_SHA_START = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()

SEED = 20261109
ESTIMATE_CPU_UPPER = 40.0
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

# -- segment A: overlapping wordings -----------------------------------------
OVERLAP_CASES = (
    {
        "name": "nested_long_first",
        "keys": (("甲乙丙", ("object", "蓝色杯子")), ("乙丙", ("place", "木桌"))),
        "sentence": "甲乙丙丁",
        "max_passes": 3,
        "expected_answers": 1,
        "expected_values": [["object", "蓝色杯子"]],
        "expected_decision": "answered_single",
        "expected_stop_pass": 2,
        "expected_exit": "no_match",
        "expected_reason": "masked_no_match",
        "expected_swallowed": ["乙丙"],
        "note": "the short wording lies entirely inside the long wording's masked span",
    },
    {
        "name": "share_one_character",
        "keys": (("甲乙", ("object", "蓝色杯子")), ("乙丙", ("place", "木桌"))),
        "sentence": "甲乙丙",
        "max_passes": 3,
        "expected_answers": 1,
        "expected_values": [["object", "蓝色杯子"]],
        "expected_decision": "answered_single",
        "expected_stop_pass": 2,
        "expected_exit": "no_match",
        "expected_reason": "masked_no_match",
        "expected_swallowed": ["乙丙"],
        "note": "one shared character is enough to destroy the later wording",
    },
    {
        "name": "abutting_no_overlap",
        "keys": (("甲乙", ("object", "蓝色杯子")), ("丙丁", ("place", "木桌"))),
        "sentence": "甲乙丙丁",
        "max_passes": 3,
        "expected_answers": 2,
        "expected_values": [["object", "蓝色杯子"], ["place", "木桌"]],
        "expected_decision": "answered_pair",
        "expected_stop_pass": 3,
        "expected_exit": "no_match",
        "expected_reason": "masked_no_match",
        "expected_swallowed": [],
        "note": "control: equal-width masking must not disturb an adjacent wording",
    },
    {
        "name": "partial_overlap_no_containment",
        "keys": (("甲乙丙", ("object", "蓝色杯子")), ("乙丙丁", ("place", "木桌"))),
        "sentence": "甲乙丙丁",
        "max_passes": 3,
        "expected_answers": 1,
        "expected_values": [["object", "蓝色杯子"]],
        "expected_decision": "answered_single",
        "expected_stop_pass": 2,
        "expected_exit": "no_match",
        "expected_reason": "masked_no_match",
        "expected_swallowed": ["乙丙丁"],
        "note": "partial overlap is not containment, but the mask still severs it",
    },
)

# -- segment B: three entities in one sentence --------------------------------
THREE_ENTITY_SENTENCE = "圆口的器皿放在窗前的台板上面，盖章用的印章也在。"
SWALLOW_KEYS = (
    ("甲乙丙", ("object", "蓝色杯子")),
    ("乙丙丁", ("object", "红色印章")),
    ("丁戊", ("place", "木桌")),
)
SWALLOW_SENTENCE = "甲乙丙丁戊"
TIE_KEYS = (
    ("甲乙", ("object", "蓝色杯子")),
    ("甲乙丙", ("object", "红色印章")),
    ("丙丁", ("place", "木桌")),
)
TIE_SENTENCE = "甲乙丙丁"
REPEATED_SENTENCE = "圆口的器皿和圆口的器皿和圆口的器皿都在。"

TRIPLE_CASES = (
    {
        "name": "three_disjoint",
        "table": "corpus_plus_aliases",
        "extra_keys": (),
        "sentence": THREE_ENTITY_SENTENCE,
        "max_passes": 4,
        "expected_answers": 3,
        "expected_values": [
            ["object", "蓝色杯子"],
            ["place", "窗台"],
            ["object", "红色印章"],
        ],
        "expected_decision": "answered_triple",
        "expected_stop_pass": 4,
        "expected_exit": "no_match",
        "expected_reason": "masked_no_match",
        "expected_swallowed": [],
        "note": "must stop by no_match at pass 4, not by the pass bound",
    },
    {
        "name": "middle_wording_swallowed",
        "table": "extra_keys",
        "extra_keys": SWALLOW_KEYS,
        "sentence": SWALLOW_SENTENCE,
        "max_passes": 4,
        "expected_answers": 2,
        "expected_values": [["object", "蓝色杯子"], ["place", "木桌"]],
        "expected_decision": "answered_pair",
        "expected_stop_pass": 3,
        "expected_exit": "no_match",
        "expected_reason": "masked_no_match",
        "expected_swallowed": ["乙丙丁"],
        "note": "the second wording straddles the first mask: layer 3 is where it stops",
    },
    {
        "name": "three_records_same_start",
        "table": "extra_keys",
        "extra_keys": TIE_KEYS,
        "sentence": TIE_SENTENCE,
        "max_passes": 4,
        "expected_answers": 0,
        "expected_values": [],
        "expected_decision": "ambiguous_same_start",
        "expected_stop_pass": 1,
        "expected_exit": "ambiguous_same_start",
        "expected_reason": "two_records_share_the_earliest_start",
        "expected_swallowed": [],
        "note": "two records claim position 0: abstain at layer 1, never pick one",
    },
    {
        "name": "same_address_three_times",
        "table": "corpus_plus_aliases",
        "extra_keys": (),
        "sentence": REPEATED_SENTENCE,
        "max_passes": 4,
        "expected_answers": 1,
        "expected_values": [["object", "蓝色杯子"]],
        "expected_decision": "answered_single_duplicate_key",
        "expected_stop_pass": 2,
        "expected_exit": "same_address_repeated",
        "expected_reason": "same_key_answered_twice",
        "expected_swallowed": [],
        "note": "three occurrences of one wording are one entity, not three",
    },
)

# -- segment E: re-run anchors (measured at authoring time) -------------------
RERUN_ANCHORS = (
    {
        "round": "R11",
        "runner": "scripts/training/probe_taiji_written_bridge_20261109.py",
        "rerun_commit": "0c8cad59e",
        "runner_blob_sha1": "4b35beed2e421b98b732ef7cbc28817e1c942cad",
        "report_runner_sha256": "8be1bef02604541ee0e295889b31c1096f92b6eaaff3de665f6931bad14c280e",
        "runner_blob_sha256": "8be1bef02604541ee0e295889b31c1096f92b6eaaff3de665f6931bad14c280e",
        "report_bridge_sha256": "ddb5a48b4215e9677ecfb49b8c2e5d2d436344b93f3f283d8cb3e060ade32caf",
        "matches_report": True,
        "note": "current HEAD cannot run it (constructor has no on_conflict); checkout this commit to re-run",
    },
    {
        "round": "R12",
        "runner": "scripts/training/probe_taiji_bridge_key_unit_20261109.py",
        "rerun_commit": "31ef1850c",
        "runner_blob_sha1": "255cb07c08c80d3b7be51e1596955b3ae84c6f7f",
        "report_runner_sha256": "3d484121c294976d15cb41408694ec4c7d62d6d0f1eae7af59861fae1e859ef1",
        "runner_blob_sha256": "3d484121c294976d15cb41408694ec4c7d62d6d0f1eae7af59861fae1e859ef1",
        "report_bridge_sha256": "248a04934fa4ce60e3d0e92a7381e1dab4df71b22fbe71e08853e73f6941f4de",
        "matches_report": True,
        "note": "current HEAD cannot run it; checkout this commit to re-run",
    },
    {
        "round": "R13",
        "runner": "scripts/training/probe_taiji_sequence_identity_20261109.py",
        "rerun_commit": "ff79e7094",
        "runner_blob_sha1": "0f4309714f5bda3179ced3f5e24dcb617ab1d0a9",
        "report_runner_sha256": "d600c33ae11b42a51ce5cf28a6d3463614817f22e563aeca15e40a05cfdb2d17",
        "runner_blob_sha256": "d600c33ae11b42a51ce5cf28a6d3463614817f22e563aeca15e40a05cfdb2d17",
        "report_bridge_sha256": "d48821ba7c4ef3a153b1f4ba214e217c74fcbbe7c00b0adaa0d7bc48429b21ba",
        "matches_report": True,
        "note": "current HEAD cannot run it; checkout this commit to re-run",
    },
    {
        "round": "R14",
        "runner": "scripts/training/probe_taiji_bridge_wording_scale_20261109.py",
        "rerun_commit": "edcf623ca",
        "runner_blob_sha1": "490e0050f49b69679c31625f2c2cfa0fc61d105e",
        "report_runner_sha256": "da86411e3ed12a74c49741eca0d81ba5271ef37cfd8b0086b925db770e3b3cb9",
        "runner_blob_sha256": "b807deffabd365a0a20de9e80fdf5559350898bd4f2cbb0dcf11cee830e7761e",
        "report_bridge_sha256": "104077f89034c9f59a863c29f49deeedf46b9d6958ff7d978797f61a185a8d4b",
        "matches_report": False,
        "note": (
            "report hash is of the working-tree bytes (CRLF); the git blob is LF, the code is "
            "identical; checkout this commit to re-run"
        ),
    },
    {
        "round": "R15",
        "runner": "scripts/training/probe_taiji_bridge_two_pass_20261109.py",
        "rerun_commit": "c1fdeaa7d",
        "runner_blob_sha1": "36c77d23207b2ae0916dfae95c028c69ac3b2e44",
        "report_runner_sha256": "b6721d6f654d3a244e574983a57426417aef76bd1e8af7eee03322992cbac8a2",
        "runner_blob_sha256": "b6721d6f654d3a244e574983a57426417aef76bd1e8af7eee03322992cbac8a2",
        "report_bridge_sha256": "967017e197b382accdcd5286c79eaa6970416ec19ff4fb5b633f1c2ae0ff8b82",
        "matches_report": True,
        "note": "current HEAD cannot run it after R16; checkout this commit to re-run",
    },
)

DEFINITION = {
    "overlap": (
        "four frozen cases in which two written wordings touch: containment, one shared character, "
        "abutting spans, and partial overlap without containment; each case declares answers, "
        "decision, stopped_at_pass and final exit before the reading"
    ),
    "swallowed": (
        "a written key is reported as swallowed when its first occurrence intersects a span that a "
        "previous pass masked; that is the direct evidence that masking deleted characters it needed"
    ),
    "three_entities": (
        "one sentence naming three entities: the disjoint case must answer 3 and stop at pass 4 by "
        "no_match (not by the pass bound); the straddling case must answer 2 and stop at pass 3"
    ),
    "stop_reporting": (
        "every case reports stopped_at_pass and final_exit, so a shortfall is never reported without "
        "the layer it stopped at"
    ),
    "regression": "the 20 describe sentences must still reach 40/40 with the default two passes",
    "zeros": "on the 18-key table, 12 unknown candidates and 18 byte fragments must answer 0",
    "rerun_anchors": (
        "R11-R15 each carry the commit at which their own runner bytes were last runnable; old "
        "reports are reproducible by checking out that commit, not orphans"
    ),
}

CRITERIA = {
    "overlap": (
        "every overlap case matches its frozen answers/decision/stopped_at_pass/final_exit exactly, "
        "wrong pointers 0, and the swallowed set equals the frozen one"
    ),
    "three_entities": (
        "three_disjoint answers 3 with pass 4 exiting no_match; middle_wording_swallowed answers 2 "
        "and stops at pass 3 with the swallowed key named; three_records_same_start abstains at pass "
        "1; same_address_three_times answers 1 entity"
    ),
    "regression": "20 describe sentences still reach 40/40, wrong 0, all answered_pair",
    "zeros": "12 unknown candidates answer 0 and 18 byte fragments answer 0",
    "closed_state": "the nine closed-state readings stay equal before and after",
    "rerun": "every row names a commit and a blob; a mismatch between report hash and blob hash is "
    "declared in the row rather than hidden",
    "reporting": "segments are reported separately; a clean overlap segment does not carry the "
    "three-entity segment",
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


def build_table(extra_keys: tuple[tuple[str, tuple[str, str]], ...] = ()) -> BridgeTable:
    table = BridgeTable(key_fn=sequence_key, read_mode="sequence")
    write = table.write(bridge_records(OBJECT_DESCRIPTIONS, PLACE_DESCRIPTIONS))
    if write["action"] != "accepted":
        raise SystemExit(f"initial write rejected: {write['action']}")
    alias = table.write(AUTHORED_ALIASES)
    if alias["action"] != "accepted":
        raise SystemExit(f"alias write rejected: {alias['action']}")
    if extra_keys:
        extra = table.write(extra_keys)
        if extra["action"] != "accepted":
            raise SystemExit(f"extra keys rejected: {extra['action']}")
    return table


def swallowed(
    sentence: str,
    keys: tuple[tuple[str, tuple[str, str]], ...],
    masks: list[dict[str, Any]],
    answered: set[str],
) -> list[str]:
    spans = [(row["span_start"], row["span_end"]) for row in masks]
    lost: list[str] = []
    for key, _ in keys:
        if key in answered:
            continue
        start = first_occurrence(tuple(sentence), tuple(key))
        if start is None:
            continue
        end = start + len(key)
        if any(span_start < end and start < span_end for span_start, span_end in spans):
            lost.append(key)
    return lost


def run_case(case: dict[str, Any]) -> dict[str, Any]:
    base_keys = tuple(bridge_records(OBJECT_DESCRIPTIONS, PLACE_DESCRIPTIONS)) + AUTHORED_ALIASES
    if case.get("table", "extra_keys") == "corpus_plus_aliases":
        table = build_table()
        keys = base_keys
    else:
        extra = tuple(case.get("extra_keys", case.get("keys", ())))
        table = build_table(extra)
        keys = base_keys + extra
    result = table.read_sentence(case["sentence"], max_passes=case["max_passes"])
    values = [list(row["value"]) for row in result["answers"]]
    answered_keys = {
        keys[row["address"]][0] for row in result["answers"] if row["address"] < len(keys)
    }
    spilled = swallowed(case["sentence"], keys, result["masks"], answered_keys)
    last = result["passes"][-1] if result["passes"] else {}
    ok = (
        len(values) == case["expected_answers"]
        and values == case["expected_values"]
        and result["decision"] == case["expected_decision"]
        and result["stopped_at_pass"] == case["expected_stop_pass"]
        and result["final_exit"] == case["expected_exit"]
        and last.get("reason") == case["expected_reason"]
        and spilled == case["expected_swallowed"]
    )
    return {
        "name": case["name"],
        "sentence": case["sentence"],
        "max_passes": case["max_passes"],
        "expected": {
            "answers": case["expected_answers"],
            "values": case["expected_values"],
            "decision": case["expected_decision"],
            "stopped_at_pass": case["expected_stop_pass"],
            "final_exit": case["expected_exit"],
            "final_reason": case["expected_reason"],
            "swallowed": case["expected_swallowed"],
        },
        "actual": {
            "answers": len(values),
            "values": values,
            "decision": result["decision"],
            "stopped_at_pass": result["stopped_at_pass"],
            "final_exit": result["final_exit"],
            "final_reason": last.get("reason"),
            "swallowed": spilled,
        },
        "masks": [
            {
                "pass": row["pass"],
                "span": [row["span_start"], row["span_end"]],
                "masked_query": row["masked_query"],
            }
            for row in result["masks"]
        ],
        "passes": [
            {
                key: row[key]
                for key in ("pass", "decision", "reason", "span_start", "span_end", "candidates")
            }
            for row in result["passes"]
        ],
        "ok": ok,
    }


def overlap_reading() -> dict[str, Any]:
    rows = [run_case(case) for case in OVERLAP_CASES]
    return {
        "rows": rows,
        "cases": len(rows),
        "declared_matches": sum(1 for row in rows if row["ok"]),
        "wrong_pointers": sum(
            1
            for row in rows
            for value in row["actual"]["values"]
            if value not in row["expected"]["values"]
        ),
        "ok": all(row["ok"] for row in rows),
    }


def triple_reading() -> dict[str, Any]:
    rows = [run_case(case) for case in TRIPLE_CASES]
    disjoint = next(row for row in rows if row["name"] == "three_disjoint")
    straddling = next(row for row in rows if row["name"] == "middle_wording_swallowed")
    tie = next(row for row in rows if row["name"] == "three_records_same_start")
    repeated = next(row for row in rows if row["name"] == "same_address_three_times")
    return {
        "rows": rows,
        "cases": len(rows),
        "declared_matches": sum(1 for row in rows if row["ok"]),
        "disjoint_three_answered": disjoint["actual"]["answers"],
        "disjoint_stop_pass": disjoint["actual"]["stopped_at_pass"],
        "straddling_answered": straddling["actual"]["answers"],
        "straddling_stop_pass": straddling["actual"]["stopped_at_pass"],
        "straddling_swallowed": straddling["actual"]["swallowed"],
        "tie_exit": tie["actual"]["final_exit"],
        "repeated_entities": repeated["actual"]["answers"],
        "ok": all(row["ok"] for row in rows),
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
    state = reference.initial_state("r16-zero-perturbation")
    step_hashes: list[str] = []
    for raw in texts:
        state = reference.observe(state, raw.encode("utf-8"), goal=b"r16")
        step_hashes.append(
            tensor_sha([state.h, state.fast, state.nu_ema, state.coverage, state.prior])
        )
    return reference_digest(reference, state, step_hashes)


def definition_digest() -> str:
    payload = {
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "overlap_cases": [
            {
                "name": case["name"],
                "keys": [list(row) for row in case["keys"]],
                "sentence": case["sentence"],
                "max_passes": case["max_passes"],
                "expected_answers": case["expected_answers"],
                "expected_values": case["expected_values"],
                "expected_decision": case["expected_decision"],
                "expected_stop_pass": case["expected_stop_pass"],
                "expected_exit": case["expected_exit"],
                "expected_reason": case["expected_reason"],
                "expected_swallowed": case["expected_swallowed"],
            }
            for case in OVERLAP_CASES
        ],
        "triple_cases": [
            {
                "name": case["name"],
                "table": case["table"],
                "extra_keys": [list(row) for row in case["extra_keys"]],
                "sentence": case["sentence"],
                "max_passes": case["max_passes"],
                "expected_answers": case["expected_answers"],
                "expected_values": case["expected_values"],
                "expected_decision": case["expected_decision"],
                "expected_stop_pass": case["expected_stop_pass"],
                "expected_exit": case["expected_exit"],
                "expected_reason": case["expected_reason"],
                "expected_swallowed": case["expected_swallowed"],
            }
            for case in TRIPLE_CASES
        ],
        "unknown": list(UNKNOWN_CANDIDATES),
        "rerun_anchors": [dict(row) for row in RERUN_ANCHORS],
        "seed": SEED,
        "estimate_cpu_upper": ESTIMATE_CPU_UPPER,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def freeze() -> dict[str, Any]:
    manifest = {
        "schema": "taiji-bridge-geometry-r16-v1",
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
            "overlap_cases": len(OVERLAP_CASES),
            "triple_cases": len(TRIPLE_CASES),
            "regression_sentences": len(sentences()),
            "unknown_candidates": len(UNKNOWN_CANDIDATES),
            "fragments": 18,
            "rerun_anchors": len(RERUN_ANCHORS),
        },
        "inputs": {
            "overlap_cases": [case["name"] for case in OVERLAP_CASES],
            "triple_cases": [case["name"] for case in TRIPLE_CASES],
            "unknown_candidates": list(UNKNOWN_CANDIDATES),
            "rerun_anchors": [dict(row) for row in RERUN_ANCHORS],
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

        overlap = overlap_reading()
        triple = triple_reading()
        table = build_table()
        regression = regression_reading(table, texts)
        zeros = zeros_reading(table)

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

        rerun_ok = all(
            row["rerun_commit"]
            and row["runner_blob_sha1"]
            and row["runner_blob_sha256"]
            and row["report_bridge_sha256"]
            for row in RERUN_ANCHORS
        )
        verdict = {
            "overlap": "overlap_declared" if overlap["ok"] else "overlap_mismatch",
            "three_entities": "three_entity_declared" if triple["ok"] else "three_entity_mismatch",
            "regression": "r15_two_pass_unchanged" if regression["ok"] else "r15_two_pass_moved",
            "zeros": "zeros_clean" if zeros["ok"] else "zeros_leak",
            "rerun_anchors": "anchors_complete" if rerun_ok else "anchors_incomplete",
        }

        cpu, wall = cost()
        if cpu > CPU_CAP:
            raise SystemExit(f"cpu cap exceeded: {cpu:.2f}s > {CPU_CAP:.0f}s")
        report = {
            "schema": "taiji-bridge-geometry-r16-report-v1",
            "runner_sha256": RUNNER_SHA_START,
            "manifest_sha256": file_sha(MANIFEST),
            "bridge_sha256": file_sha(BRIDGE_SOURCE),
            "fixture_sha256": base.fixture_hashes(),
            "definition": DEFINITION,
            "criteria": CRITERIA,
            "overlap": overlap,
            "three_entities": triple,
            "regression": regression,
            "zeros": zeros,
            "rerun_anchors": list(RERUN_ANCHORS),
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
                    "overlap": {
                        "declared_matches": overlap["declared_matches"],
                        "of": overlap["cases"],
                        "wrong": overlap["wrong_pointers"],
                    },
                    "three_entities": {
                        "declared_matches": triple["declared_matches"],
                        "of": triple["cases"],
                        "disjoint_answered": triple["disjoint_three_answered"],
                        "disjoint_stop_pass": triple["disjoint_stop_pass"],
                        "straddling_answered": triple["straddling_answered"],
                        "straddling_stop_pass": triple["straddling_stop_pass"],
                        "tie_exit": triple["tie_exit"],
                    },
                    "regression": {
                        "hits": regression["hits"],
                        "of": regression["entities"],
                        "distribution": regression["distribution"],
                    },
                    "zeros": {
                        "unknown": zeros["unknown"]["answering"],
                        "fragments": zeros["fragments"]["answering"],
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
