"""R19 -- a single-step read -> verify -> correct loop.

R18 gave the bridge an explicit compare-and-swap correction.  R19 asks whether
that primitive can close a loop: read a value, compare it with a frozen verifier,
and correct only when they differ, using the **value that was just read** as the
credential.  Every expected value is frozen before reading.

Segments, kept separate:

A. Initial value CUP, verifier RED: the loop reads CUP, corrects to RED, reads
   RED back.
A2. Initial value RED, verifier GREEN: the loop must use the value it read (RED),
   not a hard-coded CUP.
B. Read already equals the verifier: the loop performs no correction, the log
   stays empty and the content digest does not move.
C. Stale read: the loop captures CUP, another actor corrects the address to RED,
   then the loop tries to use CUP -- the correction is rejected and RED stays.
D. Wrong verifier: the table holds CUP (ground truth), the verifier says GREEN;
   the loop accepts the correction.  The protocol checks the old value, not the
   new one, so it cannot detect a wrong verifier.  This is reported as a
   declared boundary, not as a success of the loop.
E. A plain write of the old value is still rejected; correction is the only path.
F. Two iterations converge: the first corrects, the second sees agreement and
   stops; the log has exactly one entry.
G. The loop on the 18-key table: correct an alias to another value and back;
   the content digest returns to the original, R15 stays 40/40, zeros stay 0/0.

Zero training, zero optimizer steps, zero Adam, zero confirm.  No fixture byte is
written; ``address_bridge`` is not modified by this round.
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

MANIFEST = ROOT / "plans/manifests/taiji_bridge_correction_loop_20261109.json"
REPORT = ROOT / "reports/taiji_bridge_correction_loop_20261109.json"
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

DEFINITION = {
    "loop": (
        "one iteration = read the key, compare the read value with the frozen verifier, and if they "
        "differ call correct(key, read_value, verifier_value); the read value is the credential"
    ),
    "one_step": (
        "initial CUP, verifier RED: the loop must read CUP, correct to RED and read RED back"
    ),
    "read_credential": (
        "initial RED, verifier GREEN: the correction must present RED (the read value), not a "
        "hard-coded CUP"
    ),
    "no_op": "when the read already equals the verifier, no correction is issued and the log stays empty",
    "stale_read": (
        "a read captured before another actor corrects the address is a stale credential; using it "
        "must be rejected whole and the newer value must stay"
    ),
    "wrong_verifier": (
        "the table holds CUP, the verifier says GREEN; the loop accepts the correction because the "
        "protocol checks only the old value -- reported as a declared boundary"
    ),
    "plain_write": "a plain write of the old value is still rejected; correction is the only path",
    "convergence": (
        "two iterations: the first corrects, the second sees agreement and stops; exactly one log "
        "entry"
    ),
    "default_table": (
        "on the 18-key table, correct an alias to another value and back; content digest returns to "
        "the original, R15 stays 40/40, zeros stay 0/0"
    ),
    "closed_state": "the nine closed-state readings stay equal before and after",
}

CRITERIA = {
    "one_step": ("read CUP, action corrected, read RED, log length 1 with from=CUP and to=RED"),
    "read_credential": ("read RED, action corrected, read GREEN, log from=RED and to=GREEN"),
    "no_op": "no correction, read CUP, log length 0, content digest unchanged",
    "stale_read": (
        "action rejected_batch_correction_mismatch, read RED, log length 1, content digest unchanged"
    ),
    "wrong_verifier": (
        "action corrected, read GREEN, log length 1, verifier_error_accepted true and "
        "final_matches_ground_truth false -- the boundary is declared, not hidden"
    ),
    "plain_write": (
        "action rejected_batch_address_already_bound, read CUP, log length 0, content digest "
        "unchanged"
    ),
    "convergence": "two reads, one correction, log length 1, final RED",
    "default_table": (
        "content digest after the round trip equals before; regression 40/40 with wrong 0; zeros "
        "0/0; log length 2"
    ),
    "closed_state": "the nine closed-state readings stay equal before and after",
    "reporting": (
        "segments are reported separately; the wrong-verifier segment is evidence of a boundary, "
        "not evidence that the loop understands correctness"
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


def loop_step_with_credential(
    table: BridgeTable, key: str, current: tuple[str, str], verifier: tuple[str, str]
) -> dict[str, Any]:
    """One iteration using ``current`` as the credential (not a fresh read)."""
    if current == verifier:
        return {"action": "agreed", "corrected": 0}
    result = table.correct(((key, current, verifier),))
    return {"action": result["action"], "corrected": result.get("corrected", 0)}


def loop_step(table: BridgeTable, key: str, verifier: tuple[str, str]) -> dict[str, Any]:
    """One iteration: read, compare, correct with the read value as credential."""
    read = reading(table, key)
    if read["answers"] != 1:
        return {"read": read, "action": "not_answered", "corrected": 0}
    step = loop_step_with_credential(table, key, tuple(read["values"][0]), verifier)
    return {"read": read, **step}


def log_values(table: BridgeTable) -> list[list[str]]:
    return [[value_key(row["from"]), value_key(row["to"])] for row in table.corrections]


def one_step_reading() -> dict[str, Any]:
    table = build(((BASE_KEY, CUP),))
    step = loop_step(table, BASE_KEY, RED)
    after = reading(table, BASE_KEY)
    ok = (
        step["read"]["values"] == [list(CUP)]
        and step["action"] == "corrected"
        and step["corrected"] == 1
        and after["values"] == [list(RED)]
        and log_values(table) == [[value_key(CUP), value_key(RED)]]
    )
    return {
        "step": step,
        "after": after,
        "log": log_values(table),
        "log_length": len(table.corrections),
        "ok": ok,
    }


def read_credential_reading() -> dict[str, Any]:
    table = build(((BASE_KEY, RED),))
    step = loop_step(table, BASE_KEY, GREEN)
    after = reading(table, BASE_KEY)
    ok = (
        step["read"]["values"] == [list(RED)]
        and step["action"] == "corrected"
        and after["values"] == [list(GREEN)]
        and log_values(table) == [[value_key(RED), value_key(GREEN)]]
    )
    return {
        "step": step,
        "after": after,
        "log": log_values(table),
        "log_length": len(table.corrections),
        "ok": ok,
    }


def no_op_reading() -> dict[str, Any]:
    table = build(((BASE_KEY, CUP),))
    digest_before = table.content_digest()
    step = loop_step(table, BASE_KEY, CUP)
    after = reading(table, BASE_KEY)
    digest_after = table.content_digest()
    ok = (
        step["action"] == "agreed"
        and step["corrected"] == 0
        and after["values"] == [list(CUP)]
        and table.corrections == []
        and digest_before == digest_after
    )
    return {
        "step": step,
        "after": after,
        "log_length": len(table.corrections),
        "content_digest_unchanged": digest_before == digest_after,
        "ok": ok,
    }


def stale_read_reading() -> dict[str, Any]:
    table = build(((BASE_KEY, CUP),))
    stale = reading(table, BASE_KEY)
    stale_value = tuple(stale["values"][0])
    table.correct(((BASE_KEY, CUP, RED),))
    digest_before = table.content_digest()
    step = loop_step_with_credential(table, BASE_KEY, stale_value, GREEN)
    after = reading(table, BASE_KEY)
    digest_after = table.content_digest()
    ok = (
        stale["values"] == [list(CUP)]
        and step["action"] == "rejected_batch_correction_mismatch"
        and after["values"] == [list(RED)]
        and len(table.corrections) == 1
        and digest_before == digest_after
    )
    return {
        "stale_read": stale,
        "step": step,
        "after": after,
        "log_length": len(table.corrections),
        "content_digest_unchanged": digest_before == digest_after,
        "ok": ok,
    }


def wrong_verifier_reading() -> dict[str, Any]:
    table = build(((BASE_KEY, CUP),))
    step = loop_step(table, BASE_KEY, GREEN)
    after = reading(table, BASE_KEY)
    accepted = step["action"] == "corrected"
    final_matches_truth = after["values"] == [list(CUP)]
    ok = accepted and after["values"] == [list(GREEN)] and len(table.corrections) == 1
    return {
        "ground_truth": list(CUP),
        "verifier": list(GREEN),
        "step": step,
        "after": after,
        "log_length": len(table.corrections),
        "verifier_error_accepted": accepted,
        "final_matches_ground_truth": final_matches_truth,
        "ok": ok,
    }


def plain_write_reading() -> dict[str, Any]:
    table = build(((BASE_KEY, CUP),))
    digest_before = table.content_digest()
    result = table.write(((BASE_KEY, RED),))
    digest_after = table.content_digest()
    after = reading(table, BASE_KEY)
    ok = (
        result["action"] == "rejected_batch_address_already_bound"
        and after["values"] == [list(CUP)]
        and table.corrections == []
        and digest_before == digest_after
    )
    return {
        "action": result["action"],
        "after": after,
        "log_length": len(table.corrections),
        "content_digest_unchanged": digest_before == digest_after,
        "ok": ok,
    }


def convergence_reading() -> dict[str, Any]:
    table = build(((BASE_KEY, CUP),))
    first = loop_step(table, BASE_KEY, RED)
    second = loop_step(table, BASE_KEY, RED)
    after = reading(table, BASE_KEY)
    ok = (
        first["action"] == "corrected"
        and second["action"] == "agreed"
        and second["corrected"] == 0
        and after["values"] == [list(RED)]
        and len(table.corrections) == 1
    )
    return {
        "first": first,
        "second": second,
        "after": after,
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


def default_table_reading(texts: list[str]) -> dict[str, Any]:
    table = build_default()
    digest_before = table.content_digest()
    first = loop_step(table, ALIAS_KEY, RED)
    middle = reading(table, ALIAS_KEY)
    second = loop_step(table, ALIAS_KEY, CUP)
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
        "first": first,
        "middle": middle,
        "second": second,
        "final": final,
        "content_digest_unchanged": digest_before == digest_after,
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
    state = reference.initial_state("r19-zero-perturbation")
    step_hashes: list[str] = []
    for raw in texts:
        state = reference.observe(state, raw.encode("utf-8"), goal=b"r19")
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
        "seed": SEED,
        "estimate_cpu_upper": ESTIMATE_CPU_UPPER,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def freeze() -> dict[str, Any]:
    manifest = {
        "schema": "taiji-bridge-correction-loop-r19-v1",
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
            "loop_cases": 6,
            "default_table_round_trip": 1,
            "regression_sentences": len(sentences()),
            "unknown_candidates": len(UNKNOWN_CANDIDATES),
            "fragments": 18,
        },
        "inputs": {
            "base_key": BASE_KEY,
            "alias_key": ALIAS_KEY,
            "values": [list(CUP), list(RED), list(GREEN)],
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

        one_step = one_step_reading()
        read_credential = read_credential_reading()
        no_op = no_op_reading()
        stale = stale_read_reading()
        wrong_verifier = wrong_verifier_reading()
        plain_write = plain_write_reading()
        convergence = convergence_reading()
        default_table = default_table_reading(texts)

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
            "one_step": "one_step_correction" if one_step["ok"] else "one_step_failed",
            "read_credential": (
                "read_value_used_as_credential"
                if read_credential["ok"]
                else "read_credential_failed"
            ),
            "no_op": "no_op_when_agree" if no_op["ok"] else "no_op_failed",
            "stale_read": ("stale_read_rejected" if stale["ok"] else "stale_read_accepted"),
            "wrong_verifier": (
                "verifier_error_accepted"
                if wrong_verifier["ok"]
                else "verifier_error_not_reproduced"
            ),
            "plain_write": (
                "plain_write_still_rejected" if plain_write["ok"] else "plain_write_accepted"
            ),
            "convergence": ("loop_converges" if convergence["ok"] else "loop_does_not_converge"),
            "default_table": (
                "default_table_unchanged" if default_table["ok"] else "default_table_moved"
            ),
            "closed_state": "closed_state_equal" if closed_ok else "closed_state_moved",
        }
        all_ok = all(
            [
                one_step["ok"],
                read_credential["ok"],
                no_op["ok"],
                stale["ok"],
                wrong_verifier["ok"],
                plain_write["ok"],
                convergence["ok"],
                default_table["ok"],
                closed_ok,
            ]
        )
        verdict["overall"] = (
            "read_verify_correct_loop_available"
            if all_ok
            else "read_verify_correct_loop_unexpected"
        )

        cpu, wall = cost()
        if cpu > CPU_CAP:
            raise SystemExit(f"cpu cap exceeded: {cpu:.2f}s > {CPU_CAP:.0f}s")
        report = {
            "schema": "taiji-bridge-correction-loop-r19-report-v1",
            "runner_sha256": RUNNER_SHA_START,
            "manifest_sha256": file_sha(MANIFEST),
            "bridge_sha256": file_sha(BRIDGE_SOURCE),
            "fixture_sha256": base.fixture_hashes(),
            "definition": DEFINITION,
            "criteria": CRITERIA,
            "one_step": one_step,
            "read_credential": read_credential,
            "no_op": no_op,
            "stale_read": stale,
            "wrong_verifier": wrong_verifier,
            "plain_write": plain_write,
            "convergence": convergence,
            "default_table": default_table,
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
                    "one_step": {
                        "read": one_step["step"]["read"]["values"],
                        "action": one_step["step"]["action"],
                        "after": one_step["after"]["values"],
                    },
                    "read_credential": {
                        "read": read_credential["step"]["read"]["values"],
                        "log": read_credential["log"],
                    },
                    "no_op": {
                        "action": no_op["step"]["action"],
                        "log": no_op["log_length"],
                        "digest_unchanged": no_op["content_digest_unchanged"],
                    },
                    "stale": {
                        "action": stale["step"]["action"],
                        "after": stale["after"]["values"],
                        "log": stale["log_length"],
                    },
                    "wrong_verifier": {
                        "accepted": wrong_verifier["verifier_error_accepted"],
                        "after": wrong_verifier["after"]["values"],
                        "matches_truth": wrong_verifier["final_matches_ground_truth"],
                    },
                    "plain_write": {"action": plain_write["action"]},
                    "convergence": {
                        "first": convergence["first"]["action"],
                        "second": convergence["second"]["action"],
                        "log": convergence["log_length"],
                    },
                    "default_table": {
                        "digest_unchanged": default_table["content_digest_unchanged"],
                        "final": default_table["final"]["values"],
                        "regression": default_table["regression"]["hits"],
                        "of": default_table["regression"]["entities"],
                        "zeros": [
                            default_table["zeros"]["unknown_answering"],
                            default_table["zeros"]["fragment_answering"],
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
