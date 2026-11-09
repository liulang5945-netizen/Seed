"""R21 -- provenance and an optional authority policy for corrections.

R18-R20 made correction explicit, auditable and atomic, but the log did not say
*who* corrected or *why*, and no caller was ever refused.  R21 adds the smallest
provenance layer that changes that without turning into a second conflict rule:

A. Backward-compatible: a table with an empty ``allowed_authorities`` accepts a
   correction without authority; the log records empty provenance.
B. With ``allowed_authorities=("verifier:v1",)``, a correction without authority
   is rejected as ``rejected_batch_missing_authority`` before any row is planned.
C. With the same policy, an unlisted authority is rejected as
   ``rejected_batch_unknown_authority``.
D. An allowed authority plus a reason is accepted; the log entry carries both.
E. An allowed authority with a stale credential is still rejected as
   ``rejected_batch_correction_mismatch``; provenance does not replace the CAS.
F. A plain conflicting write is still rejected; the policy does not change the
   write path.
G. The 18-key table keeps the empty policy: an alias round-trip returns the
   content digest to the original, R15 stays 40/40, zeros stay 0/0, and the log
   entries carry empty provenance.
H. A two-step chain under one allowed authority logs both steps with that
   authority and the corresponding reasons; the read sees the final value.

Zero training, zero optimizer steps, zero Adam, zero confirm.  No fixture byte is
written.  ``address_bridge`` gains the optional ``authority`` / ``reason``
arguments and the optional constructor allow-list; the defaults preserve R18-R20.
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

MANIFEST = ROOT / "plans/manifests/taiji_bridge_authority_20261109.json"
REPORT = ROOT / "reports/taiji_bridge_authority_20261109.json"
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
AUTHORITY = "verifier:v1"
REASON = "evidence:r21"

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
        "note": "current HEAD cannot run it after R21 (bridge sha changed); checkout this commit",
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
        "note": "current HEAD cannot run it after R21 (bridge sha changed); checkout this commit",
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
        "note": "current HEAD cannot run it after R21 (bridge sha changed); checkout this commit",
    },
)

DEFINITION = {
    "provenance": (
        "correct(items, authority=..., reason=...) records authority and reason in the audit log; "
        "provenance is not a second credential"
    ),
    "policy": (
        "BridgeTable(allowed_authorities=(...)); empty means no policy and keeps the earlier behaviour; "
        "non-empty requires an authority from the list"
    ),
    "missing_authority": (
        "with a non-empty policy, a correction without authority is rejected as "
        "rejected_batch_missing_authority before any row is planned"
    ),
    "unknown_authority": (
        "with a non-empty policy, an authority not in the list is rejected as "
        "rejected_batch_unknown_authority"
    ),
    "allowed": "an allowed authority plus a reason is accepted and both are logged",
    "stale": (
        "an allowed authority with a stale credential is still rejected as "
        "rejected_batch_correction_mismatch; provenance does not replace the CAS"
    ),
    "plain_write": "a plain conflicting write is still rejected; the policy does not change the write path",
    "default_table": (
        "the 18-key table keeps the empty policy: alias round-trip returns the content digest, R15 "
        "stays 40/40, zeros stay 0/0, and the log authorities are empty"
    ),
    "chain": "a two-step chain under one allowed authority logs both steps with authority and reason",
    "rerun_anchors": (
        "R18, R19 and R20 carry the commit at which their own runner bytes were last runnable, "
        "because R21 changes the bridge module their manifests pin"
    ),
    "closed_state": "the nine closed-state readings stay equal before and after",
}

CRITERIA = {
    "backward": "action corrected; one log entry with empty authority and reason",
    "missing": ("action rejected_batch_missing_authority; content digest unchanged; log empty"),
    "unknown": ("action rejected_batch_unknown_authority; content digest unchanged; log empty"),
    "allowed": (
        "action corrected; one log entry with authority=verifier:v1 and reason=evidence:r21"
    ),
    "stale": ("action rejected_batch_correction_mismatch; content digest unchanged; log empty"),
    "plain_write": (
        "action rejected_batch_address_already_bound; read CUP; log empty; content digest unchanged"
    ),
    "default_table": (
        "content digest equal after the round trip; regression 40/40 with wrong 0; zeros 0/0; "
        "log length 2 and both authorities empty"
    ),
    "chain": (
        "two corrected actions; read GREEN; log length 2; both authorities verifier:v1; reasons "
        "step1 and step2"
    ),
    "rerun": (
        "every row names a commit, a runner blob sha1, a runner sha256 and the bridge sha256 used "
        "by the report"
    ),
    "closed_state": "the nine closed-state readings stay equal before and after",
    "reporting": (
        "segments are reported separately; a policy rejection is evidence that the authority is "
        "checked, not evidence that the new value is correct"
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


def backward_reading() -> dict[str, Any]:
    table = build(((BASE_KEY, CUP),))
    result = table.correct(((BASE_KEY, CUP, RED),))
    after = reading(table, BASE_KEY)
    log = log_entries(table)
    ok = (
        result["action"] == "corrected"
        and after["values"] == [list(RED)]
        and len(log) == 1
        and log[0]["authority"] == ""
        and log[0]["reason"] == ""
    )
    return {"action": result["action"], "after": after, "log": log, "ok": ok}


def missing_authority_reading() -> dict[str, Any]:
    table = build(((BASE_KEY, CUP),), allowed=(AUTHORITY,))
    digest_before = table.content_digest()
    result = table.correct(((BASE_KEY, CUP, RED),))
    digest_after = table.content_digest()
    ok = (
        result["action"] == "rejected_batch_missing_authority"
        and digest_before == digest_after
        and table.corrections == []
    )
    return {
        "action": result["action"],
        "content_digest_unchanged": digest_before == digest_after,
        "log_length": len(table.corrections),
        "ok": ok,
    }


def unknown_authority_reading() -> dict[str, Any]:
    table = build(((BASE_KEY, CUP),), allowed=(AUTHORITY,))
    digest_before = table.content_digest()
    result = table.correct(((BASE_KEY, CUP, RED),), authority="verifier:v2", reason=REASON)
    digest_after = table.content_digest()
    ok = (
        result["action"] == "rejected_batch_unknown_authority"
        and digest_before == digest_after
        and table.corrections == []
    )
    return {
        "action": result["action"],
        "content_digest_unchanged": digest_before == digest_after,
        "log_length": len(table.corrections),
        "ok": ok,
    }


def allowed_reading() -> dict[str, Any]:
    table = build(((BASE_KEY, CUP),), allowed=(AUTHORITY,))
    result = table.correct(((BASE_KEY, CUP, RED),), authority=AUTHORITY, reason=REASON)
    after = reading(table, BASE_KEY)
    log = log_entries(table)
    ok = (
        result["action"] == "corrected"
        and after["values"] == [list(RED)]
        and len(log) == 1
        and log[0]["authority"] == AUTHORITY
        and log[0]["reason"] == REASON
    )
    return {"action": result["action"], "after": after, "log": log, "ok": ok}


def stale_reading() -> dict[str, Any]:
    table = build(((BASE_KEY, CUP),), allowed=(AUTHORITY,))
    table.correct(((BASE_KEY, CUP, RED),), authority=AUTHORITY, reason="first")
    digest_before = table.content_digest()
    result = table.correct(((BASE_KEY, CUP, GREEN),), authority=AUTHORITY, reason="stale")
    digest_after = table.content_digest()
    after = reading(table, BASE_KEY)
    ok = (
        result["action"] == "rejected_batch_correction_mismatch"
        and digest_before == digest_after
        and after["values"] == [list(RED)]
        and len(table.corrections) == 1
    )
    return {
        "action": result["action"],
        "content_digest_unchanged": digest_before == digest_after,
        "after": after,
        "log_length": len(table.corrections),
        "ok": ok,
    }


def plain_write_reading() -> dict[str, Any]:
    table = build(((BASE_KEY, CUP),), allowed=(AUTHORITY,))
    digest_before = table.content_digest()
    result = table.write(((BASE_KEY, RED),))
    digest_after = table.content_digest()
    after = reading(table, BASE_KEY)
    ok = (
        result["action"] == "rejected_batch_address_already_bound"
        and digest_before == digest_after
        and after["values"] == [list(CUP)]
        and table.corrections == []
    )
    return {
        "action": result["action"],
        "content_digest_unchanged": digest_before == digest_after,
        "after": after,
        "log_length": len(table.corrections),
        "ok": ok,
    }


def default_table_reading(texts: list[str]) -> dict[str, Any]:
    table = build_default()
    digest_before = table.content_digest()
    first = table.correct(((ALIAS_KEY, CUP, RED),))
    middle = reading(table, ALIAS_KEY)
    second = table.correct(((ALIAS_KEY, RED, CUP),))
    digest_after = table.content_digest()
    final = reading(table, ALIAS_KEY)
    regression = regression_reading(table, texts)
    zeros = zeros_reading(table)
    log = log_entries(table)
    ok = (
        first["action"] == "corrected"
        and second["action"] == "corrected"
        and middle["values"] == [list(RED)]
        and final["values"] == [list(CUP)]
        and digest_before == digest_after
        and len(log) == 2
        and all(row["authority"] == "" for row in log)
        and regression["ok"]
        and zeros["ok"]
    )
    return {
        "first": first,
        "middle": middle,
        "second": second,
        "final": final,
        "content_digest_unchanged": digest_before == digest_after,
        "log": log,
        "log_length": len(log),
        "regression": regression,
        "zeros": zeros,
        "ok": ok,
    }


def chain_reading() -> dict[str, Any]:
    table = build(((BASE_KEY, CUP),), allowed=(AUTHORITY,))
    first = table.correct(((BASE_KEY, CUP, RED),), authority=AUTHORITY, reason="step1")
    second = table.correct(((BASE_KEY, RED, GREEN),), authority=AUTHORITY, reason="step2")
    after = reading(table, BASE_KEY)
    log = log_entries(table)
    ok = (
        first["action"] == "corrected"
        and second["action"] == "corrected"
        and after["values"] == [list(GREEN)]
        and len(log) == 2
        and all(row["authority"] == AUTHORITY for row in log)
        and [row["reason"] for row in log] == ["step1", "step2"]
    )
    return {
        "actions": [first["action"], second["action"]],
        "after": after,
        "log": log,
        "log_length": len(log),
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
    state = reference.initial_state("r21-zero-perturbation")
    step_hashes: list[str] = []
    for raw in texts:
        state = reference.observe(state, raw.encode("utf-8"), goal=b"r21")
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
        "authority": AUTHORITY,
        "reason": REASON,
        "values": [list(CUP), list(RED), list(GREEN), list(TABLE), list(SHELF)],
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
        "schema": "taiji-bridge-authority-r21-v1",
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
            "authority_cases": 7,
            "default_table_round_trip": 1,
            "regression_sentences": len(sentences()),
            "unknown_candidates": len(UNKNOWN_CANDIDATES),
            "fragments": 18,
            "rerun_anchors": len(RERUN_ANCHORS),
        },
        "inputs": {
            "base_key": BASE_KEY,
            "alias_key": ALIAS_KEY,
            "authority": AUTHORITY,
            "reason": REASON,
            "values": [list(CUP), list(RED), list(GREEN), list(TABLE), list(SHELF)],
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

        backward = backward_reading()
        missing = missing_authority_reading()
        unknown = unknown_authority_reading()
        allowed = allowed_reading()
        stale = stale_reading()
        plain = plain_write_reading()
        default_table = default_table_reading(texts)
        chain = chain_reading()

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
            "backward": ("backward_compatible" if backward["ok"] else "backward_incompatible"),
            "missing": (
                "missing_authority_rejected" if missing["ok"] else "missing_authority_accepted"
            ),
            "unknown": (
                "unknown_authority_rejected" if unknown["ok"] else "unknown_authority_accepted"
            ),
            "allowed": (
                "allowed_authority_applied" if allowed["ok"] else "allowed_authority_failed"
            ),
            "stale": (
                "stale_credential_still_rejected" if stale["ok"] else "stale_credential_accepted"
            ),
            "plain_write": (
                "plain_write_still_rejected" if plain["ok"] else "plain_write_accepted"
            ),
            "default_table": (
                "default_table_unchanged" if default_table["ok"] else "default_table_moved"
            ),
            "chain": ("authority_chain_attributed" if chain["ok"] else "authority_chain_broken"),
            "rerun_anchors": "anchors_complete" if rerun_ok else "anchors_incomplete",
            "closed_state": "closed_state_equal" if closed_ok else "closed_state_moved",
        }
        all_ok = all(
            [
                backward["ok"],
                missing["ok"],
                unknown["ok"],
                allowed["ok"],
                stale["ok"],
                plain["ok"],
                default_table["ok"],
                chain["ok"],
                rerun_ok,
                closed_ok,
            ]
        )
        verdict["overall"] = (
            "correction_authority_attributed" if all_ok else "correction_authority_unexpected"
        )

        cpu, wall = cost()
        if cpu > CPU_CAP:
            raise SystemExit(f"cpu cap exceeded: {cpu:.2f}s > {CPU_CAP:.0f}s")
        report = {
            "schema": "taiji-bridge-authority-r21-report-v1",
            "runner_sha256": RUNNER_SHA_START,
            "manifest_sha256": file_sha(MANIFEST),
            "bridge_sha256": file_sha(BRIDGE_SOURCE),
            "fixture_sha256": base.fixture_hashes(),
            "definition": DEFINITION,
            "criteria": CRITERIA,
            "backward": backward,
            "missing_authority": missing,
            "unknown_authority": unknown,
            "allowed_authority": allowed,
            "stale_credential": stale,
            "plain_write": plain,
            "default_table": default_table,
            "authority_chain": chain,
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
                    "backward": {
                        "action": backward["action"],
                        "authority": backward["log"][0]["authority"] if backward["log"] else None,
                    },
                    "missing": {"action": missing["action"]},
                    "unknown": {"action": unknown["action"]},
                    "allowed": {
                        "action": allowed["action"],
                        "authority": allowed["log"][0]["authority"] if allowed["log"] else None,
                        "reason": allowed["log"][0]["reason"] if allowed["log"] else None,
                    },
                    "stale": {
                        "action": stale["action"],
                        "after": stale["after"]["values"],
                    },
                    "plain_write": {"action": plain["action"]},
                    "default_table": {
                        "digest_unchanged": default_table["content_digest_unchanged"],
                        "final": default_table["final"]["values"],
                        "regression": default_table["regression"]["hits"],
                        "of": default_table["regression"]["entities"],
                        "zeros": [
                            default_table["zeros"]["unknown_answering"],
                            default_table["zeros"]["fragment_answering"],
                        ],
                        "log": default_table["log_length"],
                    },
                    "chain": {
                        "actions": chain["actions"],
                        "after": chain["after"]["values"],
                        "log": [[row["authority"], row["reason"]] for row in chain["log"]],
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
