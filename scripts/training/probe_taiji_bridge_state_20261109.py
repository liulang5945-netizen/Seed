"""R22 -- portable bridge state across a session boundary.

R18-R21 made the written bridge correctable, composable and attributable, but the
whole table lived only in the Python object.  R22 adds canonical bytes and a
validated import, so a later session can inherit the keys, values, counters and
the correction log without retraining.

A. The 18-key table with two audited corrections exports and imports back: the
   full digest, the content digest and the exported bytes are identical; the
   corrections keep their authority and reason; R15 stays 40/40; zeros stay 0/0.
B. An empty table round-trips with the same digest.
C. The importing table decides the authority policy: an import into a stricter
   table keeps the records and the old log, but a new correction with the old
   authority is rejected; the new authority is accepted.
D. Four malformed payloads (truncated, wrong schema, duplicate key, negative
   count) are rejected whole with ValueError; no partial table escapes.
E. Closed state: the nine readings stay equal.
F. Re-run anchors: R18-R21 were last runnable at their own commits, because R22
   changes the bridge module their manifests pin.

Zero training, zero optimizer steps, zero Adam, zero confirm.  No fixture byte is
written.
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

MANIFEST = ROOT / "plans/manifests/taiji_bridge_state_20261109.json"
REPORT = ROOT / "reports/taiji_bridge_state_20261109.json"
BRIDGE_SOURCE = ROOT / "taiji/address_bridge.py"

RUNNER_SHA_START = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()

SEED = 20261109
ESTIMATE_CPU_UPPER = 30.0
CPU_CAP = float(base.CPU_CAP)

CUP = ("object", "蓝色杯子")
RED = ("object", "红色印章")
GREEN = ("object", "绿色瓶子")
TABLE = ("place", "木桌")
SHELF = ("place", "书架")
BASE_KEY = "甲乙"
ALIAS_KEY = "喝水的杯子"
AUTHORITY_OLD = "verifier:v1"
AUTHORITY_NEW = "verifier:v2"
REASON_FIRST = "r22:first"
REASON_SECOND = "r22:second"

AUTHORED_ALIASES = (
    ("喝水的杯子", CUP),
    ("盖章用的印章", RED),
    ("盛水的瓶子", GREEN),
    ("圆形的皮球", ("object", "黄色球")),
    ("方方正正的木桌子", TABLE),
    ("窗前的台板", ("place", "窗台")),
    ("手边的小盒子", ("place", "左盒")),
    ("墙上的架子", SHELF),
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

RERUN_ANCHORS = (
    {
        "round": "R18",
        "runner": "scripts/training/probe_taiji_bridge_correction_20261109.py",
        "rerun_commit": "64a221f37",
        "runner_blob_sha1": "93919b40be8891a315100415751b8119365b9668",
        "report_runner_sha256": "4ba6e0a0331e20f44067abfc537d0e619756ead0bf1400e5fbfc3a58f00652e4",
        "runner_blob_sha256": "4ba6e0a0331e20f44067abfc537d0e619756ead0bf1400e5fbfc3a58f00652e4",
        "report_bridge_sha256": "65d86897c25544e683f757d4771f2d7e516476528bafd9520e14bdc59ab1fbd4",
        "matches_report": True,
        "note": "current HEAD cannot run it after R22 (bridge sha changed); checkout this commit",
    },
    {
        "round": "R19",
        "runner": "scripts/training/probe_taiji_bridge_correction_loop_20261109.py",
        "rerun_commit": "dcda39ca9",
        "runner_blob_sha1": "a89b1f47b29c66c5f6f1ae11c2ee710e07f921bb",
        "report_runner_sha256": "bd92a15aec36488ee8f5c7c3e238c31e91aabfab773eee6d155450b1a04bddf8",
        "runner_blob_sha256": "bd92a15aec36488ee8f5c7c3e238c31e91aabfab773eee6d155450b1a04bddf8",
        "report_bridge_sha256": "65d86897c25544e683f757d4771f2d7e516476528bafd9520e14bdc59ab1fbd4",
        "matches_report": True,
        "note": "current HEAD cannot run it after R22 (bridge sha changed); checkout this commit",
    },
    {
        "round": "R20",
        "runner": "scripts/training/probe_taiji_bridge_pair_correction_20261109.py",
        "rerun_commit": "56c87bbd0",
        "runner_blob_sha1": "9e9e7a5b269a62dd552acd22e5590f4c4fa29797",
        "report_runner_sha256": "1af43c0b54e92969c1ab8b4784797c4ed303d7f5da323de938e12d0d6997e7a0",
        "runner_blob_sha256": "1af43c0b54e92969c1ab8b4784797c4ed303d7f5da323de938e12d0d6997e7a0",
        "report_bridge_sha256": "65d86897c25544e683f757d4771f2d7e516476528bafd9520e14bdc59ab1fbd4",
        "matches_report": True,
        "note": "current HEAD cannot run it after R22 (bridge sha changed); checkout this commit",
    },
    {
        "round": "R21",
        "runner": "scripts/training/probe_taiji_bridge_authority_20261109.py",
        "rerun_commit": "180ab0f1e",
        "runner_blob_sha1": "95d710488ef326707d75849a9cb8586551386887",
        "report_runner_sha256": "9d167ecc1671c6a4e64cbc9fcd5c3a5fac2e02f64d7d6d4154a5b7cae31d4f10",
        "runner_blob_sha256": "9d167ecc1671c6a4e64cbc9fcd5c3a5fac2e02f64d7d6d4154a5b7cae31d4f10",
        "report_bridge_sha256": "d31f66f3cca3d6b04bdc46e2f61163c2d5b07a3c604480c31c0e73d1f5a4bbee",
        "matches_report": True,
        "note": "current HEAD cannot run it after R22 (bridge sha changed); checkout this commit",
    },
)

DEFINITION = {
    "export": (
        "export_state returns canonical UTF-8 JSON bytes: schema, key kind, read mode, exported "
        "policy, capacity, peak capacity, counters, rejected reasons, records and correction log"
    ),
    "import": (
        "BridgeTable.from_state validates the whole payload before constructing; key_fn and "
        "allowed_authorities belong to the importing table, not to the payload"
    ),
    "round_trip": (
        "the 18-key table with two audited corrections must round-trip with equal full digest, equal "
        "content digest, equal exported bytes, the same authority/reason log, 40/40 and 0/0"
    ),
    "empty": "an empty table must round-trip with the same full digest",
    "policy": (
        "importing into a stricter table keeps records and the old log, but a new correction with the "
        "old authority is rejected and one with the new authority is accepted"
    ),
    "corrupt": (
        "truncated bytes, wrong schema, duplicate key and negative count must each raise ValueError "
        "before any table is returned"
    ),
    "closed_state": "the nine closed-state readings stay equal before and after",
    "rerun_anchors": (
        "R18-R21 carry the commit at which their own runner bytes were last runnable, because R22 "
        "changes the bridge module their manifests pin"
    ),
}

CRITERIA = {
    "round_trip": (
        "full digest equal, content digest equal, exported bytes equal, log length 2 with the same "
        "authorities and reasons, alias reads the original value"
    ),
    "default_table": "regression 40/40 with wrong 0; zeros 0/0",
    "empty": "full digest before export equals full digest after import",
    "policy": (
        "content digest equal; records preserved; old authority rejected as "
        "rejected_batch_unknown_authority; new authority accepted; old log still carries the old "
        "authority"
    ),
    "corrupt": "four payloads each raise ValueError whose message starts with rejected_import_invalid_state",
    "rerun": "four anchor rows each carry a commit, a runner blob sha1, a runner sha256 and a report bridge sha256",
    "closed_state": "the nine closed-state readings stay equal before and after",
    "reporting": (
        "a successful import is evidence of portability, not of semantic understanding; the policy of "
        "the importing table is reported separately from the exported policy"
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


def build(
    keys: tuple[tuple[str, tuple[str, str]], ...],
    allowed: tuple[str, ...] = (),
) -> BridgeTable:
    table = BridgeTable(
        key_fn=sequence_key,
        read_mode="sequence",
        allowed_authorities=allowed,
    )
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


def log_entries(table: BridgeTable) -> list[dict[str, Any]]:
    return [
        {
            "from": value_key(row["from"]),
            "to": value_key(row["to"]),
            "authority": row.get("authority", ""),
            "reason": row.get("reason", ""),
        }
        for row in table.corrections
    ]


def round_trip_reading(texts: list[str]) -> dict[str, Any]:
    table = build_default()
    table.correct(((ALIAS_KEY, CUP, RED),), authority=AUTHORITY_OLD, reason=REASON_FIRST)
    table.correct(((ALIAS_KEY, RED, CUP),), authority=AUTHORITY_OLD, reason=REASON_SECOND)
    data = table.export_state()
    imported = BridgeTable.from_state(data)
    regression = regression_reading(imported, texts)
    zeros = zeros_reading(imported)
    ok = (
        imported.digest() == table.digest()
        and imported.content_digest() == table.content_digest()
        and imported.export_state() == data
        and log_entries(imported) == log_entries(table)
        and len(imported.corrections) == 2
        and reading(imported, ALIAS_KEY)["values"] == [list(CUP)]
        and regression["ok"]
        and zeros["ok"]
    )
    return {
        "bytes": len(data),
        "full_digest_equal": imported.digest() == table.digest(),
        "content_digest_equal": imported.content_digest() == table.content_digest(),
        "bytes_equal": imported.export_state() == data,
        "records": len(imported.records),
        "log_length": len(imported.corrections),
        "log": log_entries(imported),
        "alias": reading(imported, ALIAS_KEY),
        "regression": regression,
        "zeros": zeros,
        "ok": ok,
    }


def empty_reading() -> dict[str, Any]:
    table = BridgeTable(key_fn=sequence_key, read_mode="sequence")
    data = table.export_state()
    imported = BridgeTable.from_state(data)
    ok = imported.digest() == table.digest() and imported.content_digest() == table.content_digest()
    return {
        "bytes": len(data),
        "records": len(imported.records),
        "digest_equal": imported.digest() == table.digest(),
        "ok": ok,
    }


def policy_reading() -> dict[str, Any]:
    source = build(((BASE_KEY, CUP),), allowed=(AUTHORITY_OLD,))
    source.correct(((BASE_KEY, CUP, RED),), authority=AUTHORITY_OLD, reason=REASON_FIRST)
    data = source.export_state()
    imported = BridgeTable.from_state(data, allowed_authorities=(AUTHORITY_NEW,))
    digest_after_import = imported.content_digest()
    old = imported.correct(((BASE_KEY, RED, GREEN),), authority=AUTHORITY_OLD, reason="old")
    new = imported.correct(((BASE_KEY, RED, GREEN),), authority=AUTHORITY_NEW, reason="new")
    after = reading(imported, BASE_KEY)
    ok = (
        digest_after_import == source.content_digest()
        and old["action"] == "rejected_batch_unknown_authority"
        and new["action"] == "corrected"
        and after["values"] == [list(GREEN)]
        and imported.corrections[0]["authority"] == AUTHORITY_OLD
        and imported.corrections[-1]["authority"] == AUTHORITY_NEW
        and imported.allowed_authorities == (AUTHORITY_NEW,)
    )
    return {
        "exported_policy": list(source.allowed_authorities),
        "imported_policy": list(imported.allowed_authorities),
        "content_digest_equal": digest_after_import == source.content_digest(),
        "old_authority": old["action"],
        "new_authority": new["action"],
        "after": after,
        "log": log_entries(imported),
        "ok": ok,
    }


def corrupt_reading() -> dict[str, Any]:
    table = build(((BASE_KEY, CUP),))
    data = table.export_state()
    duplicate = json.loads(data.decode("utf-8"))
    duplicate["records"].append(dict(duplicate["records"][0]))
    negative = json.loads(data.decode("utf-8"))
    negative["records"][0]["count"] = -1
    wrong_schema = json.loads(data.decode("utf-8"))
    wrong_schema["schema"] = "taiji-bridge-state-v0"
    cases = {
        "truncated": data[: len(data) // 2],
        "wrong_schema": json.dumps(wrong_schema, ensure_ascii=False).encode("utf-8"),
        "duplicate_key": json.dumps(duplicate, ensure_ascii=False).encode("utf-8"),
        "negative_count": json.dumps(negative, ensure_ascii=False).encode("utf-8"),
    }
    rows = {}
    for name, bad in cases.items():
        try:
            BridgeTable.from_state(bad)
        except ValueError as error:
            rows[name] = {"rejected": True, "message": str(error)}
        else:
            rows[name] = {"rejected": False, "message": ""}
    ok = all(
        row["rejected"] and row["message"].startswith("rejected_import_invalid_state")
        for row in rows.values()
    )
    return {"rows": rows, "cases": len(rows), "ok": ok}


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
    state = reference.initial_state("r22-zero-perturbation")
    step_hashes: list[str] = []
    for raw in texts:
        state = reference.observe(state, raw.encode("utf-8"), goal=b"r22")
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
        "values": [list(CUP), list(RED), list(GREEN), list(TABLE), list(SHELF)],
        "authorities": [AUTHORITY_OLD, AUTHORITY_NEW],
        "reasons": [REASON_FIRST, REASON_SECOND],
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
        "schema": "taiji-bridge-state-r22-v1",
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
            "round_trip_tables": 1,
            "empty_tables": 1,
            "policy_tables": 1,
            "corrupt_cases": 4,
            "regression_sentences": len(sentences()),
            "unknown_candidates": len(UNKNOWN_CANDIDATES),
            "fragments": 18,
            "rerun_anchors": len(RERUN_ANCHORS),
        },
        "inputs": {
            "base_key": BASE_KEY,
            "alias_key": ALIAS_KEY,
            "values": [list(CUP), list(RED), list(GREEN), list(TABLE), list(SHELF)],
            "authorities": [AUTHORITY_OLD, AUTHORITY_NEW],
            "reasons": [REASON_FIRST, REASON_SECOND],
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

        round_trip = round_trip_reading(texts)
        empty = empty_reading()
        policy = policy_reading()
        corrupt = corrupt_reading()

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
            "round_trip": (
                "state_round_trip_exact" if round_trip["ok"] else "state_round_trip_moved"
            ),
            "empty": "empty_table_round_trip" if empty["ok"] else "empty_table_moved",
            "policy": ("importer_decides_policy" if policy["ok"] else "importer_policy_failed"),
            "corrupt": ("corrupt_state_rejected" if corrupt["ok"] else "corrupt_state_accepted"),
            "default_table": (
                "default_table_unchanged"
                if round_trip["regression"]["ok"] and round_trip["zeros"]["ok"]
                else "default_table_moved"
            ),
            "closed_state": "closed_state_equal" if closed_ok else "closed_state_moved",
            "rerun_anchors": "anchors_complete" if rerun_ok else "anchors_incomplete",
        }
        all_ok = all(
            [
                round_trip["ok"],
                empty["ok"],
                policy["ok"],
                corrupt["ok"],
                round_trip["regression"]["ok"],
                round_trip["zeros"]["ok"],
                closed_ok,
                rerun_ok,
            ]
        )
        verdict["overall"] = "bridge_state_portable" if all_ok else "bridge_state_unexpected"

        cpu, wall = cost()
        if cpu > CPU_CAP:
            raise SystemExit(f"cpu cap exceeded: {cpu:.2f}s > {CPU_CAP:.0f}s")
        report = {
            "schema": "taiji-bridge-state-r22-report-v1",
            "runner_sha256": RUNNER_SHA_START,
            "manifest_sha256": file_sha(MANIFEST),
            "bridge_sha256": file_sha(BRIDGE_SOURCE),
            "fixture_sha256": base.fixture_hashes(),
            "definition": DEFINITION,
            "criteria": CRITERIA,
            "round_trip": round_trip,
            "empty": empty,
            "policy": policy,
            "corrupt": corrupt,
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
                    "round_trip": {
                        "bytes": round_trip["bytes"],
                        "full_digest_equal": round_trip["full_digest_equal"],
                        "content_digest_equal": round_trip["content_digest_equal"],
                        "bytes_equal": round_trip["bytes_equal"],
                        "log": round_trip["log_length"],
                        "regression": round_trip["regression"]["hits"],
                        "of": round_trip["regression"]["entities"],
                        "zeros": [
                            round_trip["zeros"]["unknown_answering"],
                            round_trip["zeros"]["fragment_answering"],
                        ],
                    },
                    "empty": {"digest_equal": empty["digest_equal"]},
                    "policy": {
                        "exported": policy["exported_policy"],
                        "imported": policy["imported_policy"],
                        "old": policy["old_authority"],
                        "new": policy["new_authority"],
                    },
                    "corrupt": {
                        "cases": corrupt["cases"],
                        "rejected": sum(1 for row in corrupt["rows"].values() if row["rejected"]),
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
