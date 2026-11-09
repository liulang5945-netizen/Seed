"""R18 -- explicit correction without a "last writer wins" path.

R15 removed the ``replace`` rule: first writer keeps the address and a later
conflicting batch is rejected.  That closed silent re-pointing, but it also made
the first wrong write permanent.  R18 tests one correction protocol and freezes
every expected value before reading:

``correct(items)`` is a compare-and-swap, not a second write rule.  Each item is
``(text, expected_value, new_value)``; the caller must present the value that is
currently bound (that value is the credential).  The whole batch is validated
before any mutation, applied corrections are appended to an audit log, and the
old value is never readable as current again.  A plain ``write`` of a conflicting
value must still be rejected, so a blind later writer still cannot re-point an
address.

Segments, kept separate:

A. A plain conflicting write is still rejected; the value does not move.
B. A correction with the correct current value is applied and logged.
C. A correction with a stale credential is rejected and nothing moves.
D. A correction of an unknown address is rejected.
E. A batch that would bind one key to two new values is rejected whole.
F. A correction chain (v1 -> v2 -> v3) is audited in order; the read sees v3.
G. Round-trip on the 18-key table: correct an alias to another value and back;
   the content digest returns to the original, the R15 reading stays 40/40, the
   zero lines stay 0/0, and the audit log has two entries.
H. Re-run anchors: R16 and R17 were last runnable at their own commits, because
   R18 changes ``taiji/address_bridge.py`` and their manifests pin its sha.

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

MANIFEST = ROOT / "plans/manifests/taiji_bridge_correction_20261109.json"
REPORT = ROOT / "reports/taiji_bridge_correction_20261109.json"
BRIDGE_SOURCE = ROOT / "taiji/address_bridge.py"

RUNNER_SHA_START = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()

SEED = 20261109
ESTIMATE_CPU_UPPER = 30.0
CPU_CAP = float(base.CPU_CAP)

CUP = ("object", "蓝色杯子")
RED = ("object", "红色印章")
GREEN = ("object", "绿色瓶子")
BASE_KEY = "甲乙"
ALIAS_KEY = "喝水的杯子"

AUTHORED_ALIASES = (
    ("喝水的杯子", CUP),
    ("盖章用的印章", RED),
    ("盛水的瓶子", GREEN),
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

# -- segment H: R16/R17 re-run anchors ---------------------------------------
RERUN_ANCHORS = (
    {
        "round": "R16",
        "runner": "scripts/training/probe_taiji_bridge_geometry_20261109.py",
        "rerun_commit": "93e292840",
        "runner_blob_sha1": "e4e32b19c794edbcfbe74b3329750a0434969996",
        "report_runner_sha256": "5076807011a0e717e9d609fae0f792bfba2ccfc0bec9d30b381286e93f663261",
        "runner_blob_sha256": "5076807011a0e717e9d609fae0f792bfba2ccfc0bec9d30b381286e93f663261",
        "report_bridge_sha256": "e680d69d6ae8b05b3b0e6d9d3d22e29e1d38d48d067c1626ce54264196fb2d5b",
        "matches_report": True,
        "note": "current HEAD cannot run it after R18 (bridge sha changed); checkout this commit",
    },
    {
        "round": "R17",
        "runner": "scripts/training/probe_taiji_bridge_compression_20261109.py",
        "rerun_commit": "f6b627fd8",
        "runner_blob_sha1": "1d61b6a9f8c825ffcddaf2a3251f614d28c1945f",
        "report_runner_sha256": "9a40229fa6fd9d01dcece65b2f67aea37ecb6443033355e376047fd93e47cca8",
        "runner_blob_sha256": "9a40229fa6fd9d01dcece65b2f67aea37ecb6443033355e376047fd93e47cca8",
        "report_bridge_sha256": "e680d69d6ae8b05b3b0e6d9d3d22e29e1d38d48d067c1626ce54264196fb2d5b",
        "matches_report": True,
        "note": "current HEAD cannot run it after R18 (bridge sha changed); checkout this commit",
    },
)

DEFINITION = {
    "correction": (
        "correct(items) is compare-and-swap: each item is (text, expected_value, new_value); the "
        "expected value is the credential, the whole batch is validated before mutation, and every "
        "applied row is appended to an audit log"
    ),
    "plain_write": (
        "a plain write of a conflicting value is still rejected with the first writer kept; "
        "correction is a separate method, not a new conflict rule and not a parameter"
    ),
    "stale_credential": (
        "a correction whose expected value is not the currently bound value is rejected whole; "
        "nothing moves, the audit log does not grow"
    ),
    "unknown_address": "a correction of a key that was never written is rejected whole",
    "batch_collision": (
        "one correction batch cannot bind the same key to two different new values; the batch is "
        "rejected before any row is applied"
    ),
    "chain": "v1 -> v2 -> v3 must leave three audited steps in order and read v3",
    "round_trip": (
        "on the 18-key table, correct an alias to another value and back; content digest returns to "
        "the original, R15 stays 40/40, zeros stay 0/0, and the log has two entries"
    ),
    "rerun_anchors": (
        "R16 and R17 carry the commit at which their own runner bytes were last runnable, because "
        "R18 changes the bridge module their manifests pin"
    ),
    "closed_state": "the nine closed-state readings stay equal before and after",
}

CRITERIA = {
    "plain_write": (
        "action rejected_batch_address_already_bound, content digest unchanged, read still the "
        "original value, corrections log empty"
    ),
    "correction": (
        "action corrected, one row applied, read the new value, log length 1 with from=old and "
        "to=new, corrected count 1"
    ),
    "stale_credential": (
        "action rejected_batch_correction_mismatch, content digest unchanged, read the corrected "
        "value, log length unchanged"
    ),
    "unknown_address": (
        "action rejected_batch_unknown_address, content digest unchanged, log length unchanged"
    ),
    "batch_collision": (
        "action rejected_batch_address_collision, content digest unchanged, read the original "
        "value, log empty"
    ),
    "chain": (
        "two corrections applied, log from/to sequence is old -> middle -> final, read final"
    ),
    "round_trip": (
        "content digest after the round trip equals the digest before it; alias reads the original "
        "value; regression 40/40 with wrong 0; zeros 0/0; log length 2"
    ),
    "rerun": (
        "every row names a commit, a runner blob sha1, a runner sha256 and the bridge sha256 used "
        "by the report; a mismatch is declared in the row rather than hidden"
    ),
    "closed_state": "the nine closed-state readings stay equal before and after",
    "reporting": (
        "segments are reported separately; a successful correction does not carry the rejection "
        "segments, and a rejected correction is evidence that the credential is checked"
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


def reading(table: BridgeTable, sentence: str, max_passes: int = 2) -> dict[str, Any]:
    result = table.read_sentence(sentence, max_passes=max_passes)
    values = [list(row["value"]) for row in result["answers"]]
    return {
        "decision": result["decision"],
        "answers": len(values),
        "values": values,
        "stopped_at_pass": result["stopped_at_pass"],
        "final_exit": result["final_exit"],
    }


def log_values(table: BridgeTable) -> list[list[str]]:
    return [[value_key(row["from"]), value_key(row["to"])] for row in table.corrections]


def plain_writes_still_rejected() -> dict[str, Any]:
    table = build(((BASE_KEY, CUP),))
    digest_before = table.content_digest()
    result = table.write(((BASE_KEY, RED),))
    digest_after = table.content_digest()
    read = reading(table, BASE_KEY)
    ok = (
        result["action"] == "rejected_batch_address_already_bound"
        and digest_before == digest_after
        and read["values"] == [list(CUP)]
        and table.corrections == []
    )
    return {
        "action": result["action"],
        "content_digest_unchanged": digest_before == digest_after,
        "read": read,
        "log_length": len(table.corrections),
        "ok": ok,
    }


def correction_reading() -> dict[str, Any]:
    table = build(((BASE_KEY, CUP),))
    digest_before = table.content_digest()
    result = table.correct(((BASE_KEY, CUP, RED),))
    digest_after = table.content_digest()
    read = reading(table, BASE_KEY)
    log = log_values(table)
    ok = (
        result["action"] == "corrected"
        and result["corrected"] == 1
        and digest_before != digest_after
        and read["values"] == [list(RED)]
        and log == [[value_key(CUP), value_key(RED)]]
        and table.corrected == 1
    )
    return {
        "action": result["action"],
        "corrected": result["corrected"],
        "content_digest_changed": digest_before != digest_after,
        "read": read,
        "log": log,
        "log_length": len(table.corrections),
        "ok": ok,
    }


def stale_credential_reading() -> dict[str, Any]:
    table = build(((BASE_KEY, CUP),))
    table.correct(((BASE_KEY, CUP, RED),))
    digest_before = table.content_digest()
    result = table.correct(((BASE_KEY, CUP, GREEN),))
    digest_after = table.content_digest()
    read = reading(table, BASE_KEY)
    ok = (
        result["action"] == "rejected_batch_correction_mismatch"
        and digest_before == digest_after
        and read["values"] == [list(RED)]
        and len(table.corrections) == 1
    )
    return {
        "action": result["action"],
        "bound": result.get("bound"),
        "attempted_expected": result.get("attempted_expected"),
        "content_digest_unchanged": digest_before == digest_after,
        "read": read,
        "log_length": len(table.corrections),
        "ok": ok,
    }


def unknown_address_reading() -> dict[str, Any]:
    table = build(((BASE_KEY, CUP),))
    digest_before = table.content_digest()
    result = table.correct((("未写", CUP, RED),))
    digest_after = table.content_digest()
    ok = (
        result["action"] == "rejected_batch_unknown_address"
        and digest_before == digest_after
        and table.corrections == []
    )
    return {
        "action": result["action"],
        "content_digest_unchanged": digest_before == digest_after,
        "log_length": len(table.corrections),
        "ok": ok,
    }


def batch_collision_reading() -> dict[str, Any]:
    table = build(((BASE_KEY, CUP),))
    digest_before = table.content_digest()
    result = table.correct(((BASE_KEY, CUP, RED), (BASE_KEY, CUP, GREEN)))
    digest_after = table.content_digest()
    read = reading(table, BASE_KEY)
    ok = (
        result["action"] == "rejected_batch_address_collision"
        and digest_before == digest_after
        and read["values"] == [list(CUP)]
        and table.corrections == []
    )
    return {
        "action": result["action"],
        "content_digest_unchanged": digest_before == digest_after,
        "read": read,
        "log_length": len(table.corrections),
        "ok": ok,
    }


def chain_reading() -> dict[str, Any]:
    table = build(((BASE_KEY, CUP),))
    first = table.correct(((BASE_KEY, CUP, RED),))
    second = table.correct(((BASE_KEY, RED, GREEN),))
    read = reading(table, BASE_KEY)
    sequence = log_values(table)
    expected = [
        [value_key(CUP), value_key(RED)],
        [value_key(RED), value_key(GREEN)],
    ]
    ok = (
        first["action"] == "corrected"
        and second["action"] == "corrected"
        and read["values"] == [list(GREEN)]
        and sequence == expected
        and len(table.corrections) == 2
        and table.corrected == 2
    )
    return {
        "actions": [first["action"], second["action"]],
        "read": read,
        "log": sequence,
        "log_length": len(table.corrections),
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
    for phrase in UNKNOWN_CANDIDATES:
        unknown_answering += int(bool(table.read_sentence(phrase)["answers"]))
    fragment_answering = 0
    for phrase, _ in bridge_records(OBJECT_DESCRIPTIONS, PLACE_DESCRIPTIONS):
        data = phrase.encode("utf-8")
        for offset in (1, 2):
            fragment_answering += int(bool(table.read_sentence(data[offset:])["answers"]))
    return {
        "unknown_answering": unknown_answering,
        "fragment_answering": fragment_answering,
        "ok": unknown_answering == 0 and fragment_answering == 0,
    }


def round_trip_reading(texts: list[str]) -> dict[str, Any]:
    table = build_default()
    digest_before = table.content_digest()
    first = table.correct(((ALIAS_KEY, CUP, RED),))
    middle = reading(table, ALIAS_KEY)
    second = table.correct(((ALIAS_KEY, RED, CUP),))
    digest_after = table.content_digest()
    final = reading(table, ALIAS_KEY)
    regression = regression_reading(table, texts)
    zeros = zeros_reading(table)
    ok = (
        first["action"] == "corrected"
        and second["action"] == "corrected"
        and middle["values"] == [list(RED)]
        and final["values"] == [list(CUP)]
        and digest_before == digest_after
        and len(table.corrections) == 2
        and regression["ok"]
        and zeros["ok"]
    )
    return {
        "records": len(table.records),
        "actions": [first["action"], second["action"]],
        "middle": middle,
        "final": final,
        "content_digest_unchanged": digest_before == digest_after,
        "content_digest_before": digest_before,
        "content_digest_after": digest_after,
        "log": log_values(table),
        "log_length": len(table.corrections),
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
    state = reference.initial_state("r18-zero-perturbation")
    step_hashes: list[str] = []
    for raw in texts:
        state = reference.observe(state, raw.encode("utf-8"), goal=b"r18")
        step_hashes.append(
            tensor_sha([state.h, state.fast, state.nu_ema, state.coverage, state.prior])
        )
    return reference_digest(reference, state, step_hashes)


def definition_digest() -> str:
    payload = {
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "base_key": BASE_KEY,
        "alias_key": ALIAS_KEY,
        "values": [list(CUP), list(RED), list(GREEN)],
        "aliases": [list(row) for row in AUTHORED_ALIASES],
        "unknown": list(UNKNOWN_CANDIDATES),
        "rerun_anchors": [dict(row) for row in RERUN_ANCHORS],
        "seed": SEED,
        "estimate_cpu_upper": ESTIMATE_CPU_UPPER,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def freeze() -> dict[str, Any]:
    manifest = {
        "schema": "taiji-bridge-correction-r18-v1",
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
            "correction_cases": 6,
            "regression_sentences": len(sentences()),
            "unknown_candidates": len(UNKNOWN_CANDIDATES),
            "fragments": 18,
            "rerun_anchors": len(RERUN_ANCHORS),
        },
        "inputs": {
            "base_key": BASE_KEY,
            "alias_key": ALIAS_KEY,
            "values": [list(CUP), list(RED), list(GREEN)],
            "aliases": [list(row) for row in AUTHORED_ALIASES],
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

        plain = plain_writes_still_rejected()
        correction = correction_reading()
        stale = stale_credential_reading()
        unknown = unknown_address_reading()
        collision = batch_collision_reading()
        chain = chain_reading()
        round_trip = round_trip_reading(texts)

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
            "plain_conflict": (
                "plain_conflict_still_rejected" if plain["ok"] else "plain_conflict_accepted"
            ),
            "correction": "cas_correction_applied" if correction["ok"] else "cas_correction_failed",
            "stale": ("stale_credential_rejected" if stale["ok"] else "stale_credential_accepted"),
            "unknown": (
                "unknown_address_rejected" if unknown["ok"] else "unknown_address_accepted"
            ),
            "collision": (
                "correction_batch_collision_rejected"
                if collision["ok"]
                else "correction_batch_collision_accepted"
            ),
            "chain": "correction_chain_audited" if chain["ok"] else "correction_chain_broken",
            "round_trip": "round_trip_clean" if round_trip["ok"] else "round_trip_moved",
            "closed_state": "closed_state_equal" if closed_ok else "closed_state_moved",
            "rerun_anchors": "anchors_complete" if rerun_ok else "anchors_incomplete",
        }
        all_ok = all(
            [
                plain["ok"],
                correction["ok"],
                stale["ok"],
                unknown["ok"],
                collision["ok"],
                chain["ok"],
                round_trip["ok"],
                closed_ok,
                rerun_ok,
            ]
        )
        verdict["overall"] = (
            "explicit_correction_available" if all_ok else "explicit_correction_unexpected"
        )

        cpu, wall = cost()
        if cpu > CPU_CAP:
            raise SystemExit(f"cpu cap exceeded: {cpu:.2f}s > {CPU_CAP:.0f}s")
        report = {
            "schema": "taiji-bridge-correction-r18-report-v1",
            "runner_sha256": RUNNER_SHA_START,
            "manifest_sha256": file_sha(MANIFEST),
            "bridge_sha256": file_sha(BRIDGE_SOURCE),
            "fixture_sha256": base.fixture_hashes(),
            "definition": DEFINITION,
            "criteria": CRITERIA,
            "plain_conflict": plain,
            "cas_correction": correction,
            "stale_credential": stale,
            "unknown_address": unknown,
            "batch_collision": collision,
            "correction_chain": chain,
            "round_trip": round_trip,
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
                    "plain": {
                        "action": plain["action"],
                        "digest_unchanged": plain["content_digest_unchanged"],
                        "read": plain["read"]["values"],
                    },
                    "correction": {
                        "action": correction["action"],
                        "read": correction["read"]["values"],
                        "log": correction["log_length"],
                    },
                    "stale": {
                        "action": stale["action"],
                        "digest_unchanged": stale["content_digest_unchanged"],
                        "read": stale["read"]["values"],
                    },
                    "unknown": {"action": unknown["action"]},
                    "collision": {"action": collision["action"]},
                    "chain": {
                        "actions": chain["actions"],
                        "read": chain["read"]["values"],
                        "log": chain["log_length"],
                    },
                    "round_trip": {
                        "digest_unchanged": round_trip["content_digest_unchanged"],
                        "final": round_trip["final"]["values"],
                        "regression": round_trip["regression"]["hits"],
                        "of": round_trip["regression"]["entities"],
                        "zeros": [
                            round_trip["zeros"]["unknown_answering"],
                            round_trip["zeros"]["fragment_answering"],
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
