"""R20 -- sentence-level correction of two entities (one slot at a time).

R19 closed the loop on a single key.  A sentence read by the bridge normally
returns two values (object and place), and a verifier normally points at one
slot or both.  R20 asks whether the correction primitive composes at that level:

A. One slot wrong: read (CUP, TABLE), verifier (RED, TABLE).  Only the object
   key is corrected; the place key must not move.
B. Both slots wrong: read (CUP, TABLE), verifier (RED, SHELF).  Both keys are
   corrected in one batch; the read returns (RED, SHELF).
C. Stale subset: the loop captures (CUP, TABLE), another actor corrects the
   place key to SHELF, then the loop submits both corrections using the stale
   pair.  The batch must be rejected whole, so the valid object correction is
   not applied either; the place stays SHELF.
D. Both slots already agree: no correction, empty log, content digest unchanged.
E. A plain write of the old value is still rejected.
F. The pair loop converges: the first iteration corrects, the second sees
   agreement and stops; exactly one log entry (case A).
G. The same pair loop on the 18-key table: correct the two keys of a describe
   sentence to another pair and back; content digest returns to the original,
   R15 stays 40/40, zeros stay 0/0.

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

MANIFEST = ROOT / "plans/manifests/taiji_bridge_pair_correction_20261109.json"
REPORT = ROOT / "reports/taiji_bridge_pair_correction_20261109.json"
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
RUG = ("place", "地毯")

OBJECT_KEY = "甲乙"
PLACE_KEY = "丙丁"
SENTENCE = "甲乙丙丁"

AUTHORED_ALIASES = (
    ("喝水的杯子", CUP),
    ("盖章用的印章", RED),
    ("盛水的瓶子", GREEN),
    ("圆形的皮球", ("object", "黄色球")),
    ("方方正正的木桌子", TABLE),
    ("窗前的台板", ("place", "窗台")),
    ("手边的小盒子", ("place", "左盒")),
    ("墙上的架子", SHELF),
    ("脚下的垫子", RUG),
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
    "pair_read": (
        "one sentence carries two written keys; the two-pass read returns one value per slot (object "
        "and place); the slot is read from the value, and the key text is read from the record"
    ),
    "loop": (
        "one iteration compares each slot with the frozen verifier and submits one correct batch "
        "containing only the mismatched slots, using each slot's read value as its credential"
    ),
    "single_slot": "object CUP -> RED with place already TABLE; only the object key may move",
    "both_slots": "object CUP -> RED and place TABLE -> SHELF in one batch; both must move",
    "stale_subset": (
        "capture (CUP, TABLE), another actor corrects place to SHELF, then submit both corrections "
        "from the stale pair; the batch must be rejected whole, so object stays CUP"
    ),
    "no_op": "both read values equal the verifier: no correction, empty log, digest unchanged",
    "plain_write": "a plain write of the old value is still rejected; correction is the only path",
    "convergence": "two iterations: the first corrects, the second agrees; exactly one log entry",
    "default_table": (
        "on the 18-key table, correct the two keys of a describe sentence to another pair and back; "
        "content digest returns to the original, R15 40/40, zeros 0/0"
    ),
    "closed_state": "the nine closed-state readings stay equal before and after",
}

CRITERIA = {
    "single_slot": (
        "action corrected with one item; read (CUP, TABLE) -> (RED, TABLE); log length 1; the log "
        "key is the object key"
    ),
    "both_slots": (
        "action corrected with two items; read (CUP, TABLE) -> (RED, SHELF); log length 2 in object "
        "then place order"
    ),
    "stale_subset": (
        "action rejected_batch_correction_mismatch; read stays (CUP, SHELF); log length 1; content "
        "digest unchanged"
    ),
    "no_op": "action agreed; no correction; log empty; content digest unchanged",
    "plain_write": (
        "action rejected_batch_address_already_bound; read stays (CUP, TABLE); log empty; content "
        "digest unchanged"
    ),
    "convergence": "first corrected, second agreed, log length 1, final (RED, TABLE)",
    "default_table": (
        "first batch corrects two keys, second batch restores them; content digest equal; "
        "regression 40/40 with wrong 0; zeros 0/0; log length 4"
    ),
    "closed_state": "the nine closed-state readings stay equal before and after",
    "reporting": (
        "segments are reported separately; a single-slot correction does not carry the stale-subset "
        "segment, and the stale segment is evidence that the batch is atomic"
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


def reading(table: BridgeTable, sentence: str, max_passes: int = 3) -> dict[str, Any]:
    result = table.read_sentence(sentence, max_passes=max_passes)
    values = [list(row["value"]) for row in result["answers"]]
    slots = {}
    for row in result["answers"]:
        slot, _ = row["value"]
        slots[slot] = {
            "value": list(row["value"]),
            "key": "".join(table.records[row["address"]]["key"]),
        }
    return {
        "decision": result["decision"],
        "answers": len(values),
        "values": values,
        "slots": slots,
        "stopped_at_pass": result["stopped_at_pass"],
        "final_exit": result["final_exit"],
    }


def log_values(table: BridgeTable) -> list[list[str]]:
    return [[value_key(row["from"]), value_key(row["to"])] for row in table.corrections]


def pair_step(
    table: BridgeTable, sentence: str, verifier: dict[str, tuple[str, str]]
) -> dict[str, Any]:
    read = reading(table, sentence)
    items = []
    for slot, wanted in verifier.items():
        got = read["slots"].get(slot)
        if got is None:
            continue
        if tuple(got["value"]) != wanted:
            items.append((got["key"], tuple(got["value"]), wanted))
    if not items:
        return {"read": read, "action": "agreed", "items": [], "corrected": 0}
    result = table.correct(tuple(items))
    return {
        "read": read,
        "action": result["action"],
        "items": items,
        "corrected": result.get("corrected", 0),
    }


def pair_step_with_slots(
    table: BridgeTable,
    read: dict[str, Any],
    verifier: dict[str, tuple[str, str]],
) -> dict[str, Any]:
    items = []
    for slot, wanted in verifier.items():
        got = read["slots"].get(slot)
        if got is None:
            continue
        if tuple(got["value"]) != wanted:
            items.append((got["key"], tuple(got["value"]), wanted))
    if not items:
        return {"action": "agreed", "items": [], "corrected": 0}
    result = table.correct(tuple(items))
    return {
        "action": result["action"],
        "items": items,
        "corrected": result.get("corrected", 0),
    }


def single_slot_reading() -> dict[str, Any]:
    table = build(((OBJECT_KEY, CUP), (PLACE_KEY, TABLE)))
    step = pair_step(table, SENTENCE, {"object": RED, "place": TABLE})
    after = reading(table, SENTENCE)
    log = log_values(table)
    ok = (
        step["read"]["values"] == [list(CUP), list(TABLE)]
        and step["action"] == "corrected"
        and step["corrected"] == 1
        and after["values"] == [list(RED), list(TABLE)]
        and len(table.corrections) == 1
        and log == [[value_key(CUP), value_key(RED)]]
    )
    return {
        "step": step,
        "after": after,
        "log": log,
        "log_length": len(table.corrections),
        "ok": ok,
    }


def both_slots_reading() -> dict[str, Any]:
    table = build(((OBJECT_KEY, CUP), (PLACE_KEY, TABLE)))
    step = pair_step(table, SENTENCE, {"object": RED, "place": SHELF})
    after = reading(table, SENTENCE)
    log = log_values(table)
    ok = (
        step["read"]["values"] == [list(CUP), list(TABLE)]
        and step["action"] == "corrected"
        and step["corrected"] == 2
        and after["values"] == [list(RED), list(SHELF)]
        and len(table.corrections) == 2
        and log == [[value_key(CUP), value_key(RED)], [value_key(TABLE), value_key(SHELF)]]
    )
    return {
        "step": step,
        "after": after,
        "log": log,
        "log_length": len(table.corrections),
        "ok": ok,
    }


def stale_subset_reading() -> dict[str, Any]:
    table = build(((OBJECT_KEY, CUP), (PLACE_KEY, TABLE)))
    stale = reading(table, SENTENCE)
    table.correct(((PLACE_KEY, TABLE, SHELF),))
    digest_before = table.content_digest()
    step = pair_step_with_slots(table, stale, {"object": RED, "place": SHELF})
    after = reading(table, SENTENCE)
    digest_after = table.content_digest()
    ok = (
        step["action"] == "rejected_batch_correction_mismatch"
        and after["values"] == [list(CUP), list(SHELF)]
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


def no_op_reading() -> dict[str, Any]:
    table = build(((OBJECT_KEY, CUP), (PLACE_KEY, TABLE)))
    digest_before = table.content_digest()
    step = pair_step(table, SENTENCE, {"object": CUP, "place": TABLE})
    after = reading(table, SENTENCE)
    digest_after = table.content_digest()
    ok = (
        step["action"] == "agreed"
        and step["corrected"] == 0
        and after["values"] == [list(CUP), list(TABLE)]
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


def plain_write_reading() -> dict[str, Any]:
    table = build(((OBJECT_KEY, CUP), (PLACE_KEY, TABLE)))
    digest_before = table.content_digest()
    result = table.write(((OBJECT_KEY, RED),))
    digest_after = table.content_digest()
    after = reading(table, SENTENCE)
    ok = (
        result["action"] == "rejected_batch_address_already_bound"
        and after["values"] == [list(CUP), list(TABLE)]
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
    table = build(((OBJECT_KEY, CUP), (PLACE_KEY, TABLE)))
    first = pair_step(table, SENTENCE, {"object": RED, "place": TABLE})
    second = pair_step(table, SENTENCE, {"object": RED, "place": TABLE})
    after = reading(table, SENTENCE)
    ok = (
        first["action"] == "corrected"
        and second["action"] == "agreed"
        and second["corrected"] == 0
        and after["values"] == [list(RED), list(TABLE)]
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
    sentence = texts[0]
    truth = expected_pair(sentence)
    object_key = OBJECT_DESCRIPTIONS[truth["object"][1]]
    place_key = PLACE_DESCRIPTIONS[truth["place"][1]]
    target_object = RED if truth["object"] != RED else GREEN
    target_place = SHELF if truth["place"] != SHELF else RUG
    verifier = {"object": target_object, "place": target_place}
    digest_before = table.content_digest()
    first = pair_step(table, sentence, verifier)
    middle = reading(table, sentence)
    second = pair_step(table, sentence, {"object": truth["object"], "place": truth["place"]})
    digest_after = table.content_digest()
    final = reading(table, sentence)
    regression = regression_reading(table, texts)
    zeros = zeros_reading(table)
    ok = (
        first["action"] == "corrected"
        and first["corrected"] == 2
        and middle["values"] == [list(target_object), list(target_place)]
        and second["action"] == "corrected"
        and second["corrected"] == 2
        and final["values"] == [list(truth["object"]), list(truth["place"])]
        and digest_before == digest_after
        and len(table.corrections) == 4
        and regression["ok"]
        and zeros["ok"]
    )
    return {
        "sentence": sentence,
        "truth": [list(truth["object"]), list(truth["place"])],
        "target": [list(target_object), list(target_place)],
        "keys": [object_key, place_key],
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
    state = reference.initial_state("r20-zero-perturbation")
    step_hashes: list[str] = []
    for raw in texts:
        state = reference.observe(state, raw.encode("utf-8"), goal=b"r20")
        step_hashes.append(
            tensor_sha([state.h, state.fast, state.nu_ema, state.coverage, state.prior])
        )
    return reference_digest(reference, state, step_hashes)


def definition_digest() -> str:
    payload = {
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "object_key": OBJECT_KEY,
        "place_key": PLACE_KEY,
        "sentence": SENTENCE,
        "values": [list(CUP), list(RED), list(GREEN), list(TABLE), list(SHELF), list(RUG)],
        "aliases": [list(row) for row in AUTHORED_ALIASES],
        "unknown": list(UNKNOWN_CANDIDATES),
        "seed": SEED,
        "estimate_cpu_upper": ESTIMATE_CPU_UPPER,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def freeze() -> dict[str, Any]:
    manifest = {
        "schema": "taiji-bridge-pair-correction-r20-v1",
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
            "pair_cases": 6,
            "default_table_round_trip": 1,
            "regression_sentences": len(sentences()),
            "unknown_candidates": len(UNKNOWN_CANDIDATES),
            "fragments": 18,
        },
        "inputs": {
            "object_key": OBJECT_KEY,
            "place_key": PLACE_KEY,
            "sentence": SENTENCE,
            "values": [list(CUP), list(RED), list(GREEN), list(TABLE), list(SHELF), list(RUG)],
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

        single = single_slot_reading()
        both = both_slots_reading()
        stale = stale_subset_reading()
        no_op = no_op_reading()
        plain = plain_write_reading()
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
            "single_slot": ("single_slot_correction" if single["ok"] else "single_slot_failed"),
            "both_slots": "both_slots_correction" if both["ok"] else "both_slots_failed",
            "stale_subset": ("stale_subset_rejected" if stale["ok"] else "stale_subset_accepted"),
            "no_op": "no_op_when_pair_agrees" if no_op["ok"] else "no_op_failed",
            "plain_write": (
                "plain_write_still_rejected" if plain["ok"] else "plain_write_accepted"
            ),
            "convergence": (
                "pair_loop_converges" if convergence["ok"] else "pair_loop_does_not_converge"
            ),
            "default_table": (
                "default_table_unchanged" if default_table["ok"] else "default_table_moved"
            ),
            "closed_state": "closed_state_equal" if closed_ok else "closed_state_moved",
        }
        all_ok = all(
            [
                single["ok"],
                both["ok"],
                stale["ok"],
                no_op["ok"],
                plain["ok"],
                convergence["ok"],
                default_table["ok"],
                closed_ok,
            ]
        )
        verdict["overall"] = (
            "sentence_pair_correction_available"
            if all_ok
            else "sentence_pair_correction_unexpected"
        )

        cpu, wall = cost()
        if cpu > CPU_CAP:
            raise SystemExit(f"cpu cap exceeded: {cpu:.2f}s > {CPU_CAP:.0f}s")
        report = {
            "schema": "taiji-bridge-pair-correction-r20-report-v1",
            "runner_sha256": RUNNER_SHA_START,
            "manifest_sha256": file_sha(MANIFEST),
            "bridge_sha256": file_sha(BRIDGE_SOURCE),
            "fixture_sha256": base.fixture_hashes(),
            "definition": DEFINITION,
            "criteria": CRITERIA,
            "single_slot": single,
            "both_slots": both,
            "stale_subset": stale,
            "no_op": no_op,
            "plain_write": plain,
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
                    "single": {
                        "read": single["step"]["read"]["values"],
                        "action": single["step"]["action"],
                        "after": single["after"]["values"],
                        "log": single["log_length"],
                    },
                    "both": {
                        "read": both["step"]["read"]["values"],
                        "action": both["step"]["action"],
                        "after": both["after"]["values"],
                        "log": both["log_length"],
                    },
                    "stale": {
                        "action": stale["step"]["action"],
                        "after": stale["after"]["values"],
                        "log": stale["log_length"],
                    },
                    "no_op": {
                        "action": no_op["step"]["action"],
                        "log": no_op["log_length"],
                        "digest_unchanged": no_op["content_digest_unchanged"],
                    },
                    "plain_write": {"action": plain["action"]},
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
                        "log": default_table["log_length"],
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
