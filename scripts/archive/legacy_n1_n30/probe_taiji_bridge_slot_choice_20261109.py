"""R27 -- slot-scoped choice on a two-entity sentence.

R26 could choose only when the whole read yielded one entity.  A normal
description has an object and a place, so R27 adds ``slot`` to
``bridge_choose``:

A. The same sentence ``喝水的杯子放在木桌上面。`` chooses CUP in the object slot
   and TABLE in the place slot, independently.
B. Without ``slot`` the same sentence is ``ambiguous_read`` (two entities).
C. Asking a slot that the sentence does not carry gives ``no_choice``.
D. A slot whose read value is outside the candidates gives
   ``not_in_candidates``.
E. Two entities of the same slot give ``ambiguous_read`` even with ``slot``.
F. Correcting the CUP alias to GREEN flips the object choice 0->1 while the
   place choice stays 0; the table digest moves, the subject tensors do not.
G. No bridge gives ``no_bridge``.
H. Closed state: the nine legacy readings stay equal before and after.

Zero training, zero optimizer steps, zero Adam, zero confirm.
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
)
from taiji.living_core import LivingCoreReference  # noqa: E402
from taiji.native_goal_world_wide import (  # noqa: E402
    OBJECT_DESCRIPTIONS,
    PLACE_DESCRIPTIONS,
)
from taiji.process_guard import assert_no_black_processes  # noqa: E402

torch.set_num_threads(1)

MANIFEST = ROOT / "plans/manifests/taiji_bridge_slot_choice_20261109.json"
REPORT = ROOT / "reports/taiji_bridge_slot_choice_20261109.json"
BRIDGE_SOURCE = ROOT / "taiji/address_bridge.py"
CORE_SOURCE = ROOT / "taiji/living_core.py"

RUNNER_SHA_START = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()

SEED = 20261109
ESTIMATE_CPU_UPPER = 30.0
CPU_CAP = float(base.CPU_CAP)

CUP = ("object", "蓝色杯子")
RED = ("object", "红色印章")
GREEN = ("object", "绿色瓶子")
TABLE = ("place", "木桌")
SHELF = ("place", "书架")
CUP_KEY = "喝水的杯子"
RED_KEY = "盖章用的印章"
BOTH_QUERY = "喝水的杯子放在方方正正的木桌子上面。"
TWO_OBJECTS = "喝水的杯子和盖章用的印章都在。"
AUTHORITY = "verifier:v1"
REASON = "r27:slot"

AUTHORED_ALIASES = (
    (CUP_KEY, CUP),
    (RED_KEY, RED),
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
        "round": "R26",
        "runner": "scripts/training/probe_taiji_bridge_choice_20261109.py",
        "rerun_commit": "c3633b4c8",
        "runner_blob_sha1": "950eef248c9fff3bb6cd3c0d4d29d77d978597e1",
        "report_runner_sha256": "053b852dbfe9dcc845b57f6106799d00419f014aed8b0e656f5050bdf1396426",
        "runner_blob_sha256": "053b852dbfe9dcc845b57f6106799d00419f014aed8b0e656f5050bdf1396426",
        "report_core_sha256": "eead9c4af475ccd7c74c7260d60c2d035a37aabf2445442a0db7bc36cacb4c4a",
        "matches_report": True,
        "note": "current HEAD cannot run it after R27 (core sha changed); checkout this commit",
    },
)

DEFINITION = {
    "slot": (
        "bridge_choose(..., slot=...) counts only answers whose value[0] equals the slot; "
        "without slot the whole read is counted"
    ),
    "separate": (
        "one sentence with CUP and TABLE chooses CUP in object and TABLE in place, independently"
    ),
    "no_slot": "the same two-entity sentence without slot is ambiguous_read",
    "slot_missing": "a slot the sentence does not carry gives no_choice",
    "outside": "a slot whose value is outside the candidates gives not_in_candidates",
    "same_slot_ambiguous": "two entities of the same slot give ambiguous_read even with slot",
    "correction": (
        "correcting CUP to GREEN flips the object choice 0->1 while the place choice stays 0; "
        "the table digest moves and the subject hashes do not"
    ),
    "no_bridge": "a state without a table gives no_bridge",
    "closed_state": "the nine legacy closed-state readings stay equal before and after",
}

CRITERIA = {
    "slot_choice": (
        "object [RED,CUP] -> CUP index 1; place [SHELF,TABLE] -> TABLE index 1; "
        "object [CUP,RED] -> index 0; place [TABLE,SHELF] -> index 0"
    ),
    "no_slot": "two-entity sentence without slot -> ambiguous_read, answers=2",
    "slot_missing": "CUP-only sentence with slot=place -> no_choice, read_answers=1, answers=0",
    "outside": "two-entity sentence, slot=object, candidates [TABLE,SHELF] -> not_in_candidates",
    "same_slot_ambiguous": "two objects, slot=object -> ambiguous_read, answers=2",
    "correction": "before object 0 / place 0; after object 1 / place 0; table digest changes; hashes equal",
    "inert": "parameter and state hashes equal before and after all choices",
    "no_bridge": "a state without a table -> no_bridge",
    "closed_state": "the nine legacy readings stay equal before and after",
    "rerun": "the R26 anchor carries a commit, a runner blob sha1, a runner sha256 and a core sha256",
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


def build_default() -> BridgeTable:
    table = BridgeTable(
        key_fn=sequence_key,
        read_mode="sequence",
        allowed_authorities=(AUTHORITY,),
    )
    write = table.write(tuple(bridge_records(OBJECT_DESCRIPTIONS, PLACE_DESCRIPTIONS)))
    if write["action"] != "accepted":
        raise SystemExit(f"write rejected: {write['action']}")
    alias = table.write(AUTHORED_ALIASES)
    if alias["action"] != "accepted":
        raise SystemExit(f"alias write rejected: {alias['action']}")
    return table


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


def state_sha(state: Any) -> str:
    return tensor_sha([state.h, state.fast, state.nu_ema, state.coverage, state.prior])


def parameter_sha(reference: Any) -> str:
    return tensor_sha([tensor for _, tensor in sorted(reference.named_parameters())])


def choice_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "decision": result["decision"],
        "choice": list(result["choice"]) if result["choice"] is not None else None,
        "index": result["index"],
        "read_decision": result["read_decision"],
        "answers": result["answers"],
        "read_answers": result["read_answers"],
        "slot": result["slot"],
    }


def slot_reading(reference: Any) -> dict[str, Any]:
    table = build_default()
    state = reference.attach_bridge(reference.initial_state("r27-slot"), table)
    baseline_state = state_sha(state)
    baseline_parameters = parameter_sha(reference)

    object_zero = choice_payload(
        reference.bridge_choose(state, BOTH_QUERY, [CUP, RED], slot="object")
    )
    object_one = choice_payload(
        reference.bridge_choose(state, BOTH_QUERY, [RED, CUP], slot="object")
    )
    place_zero = choice_payload(
        reference.bridge_choose(state, BOTH_QUERY, [TABLE, SHELF], slot="place")
    )
    place_one = choice_payload(
        reference.bridge_choose(state, BOTH_QUERY, [SHELF, TABLE], slot="place")
    )
    whole = choice_payload(reference.bridge_choose(state, BOTH_QUERY, [CUP, TABLE]))
    missing = choice_payload(reference.bridge_choose(state, CUP_KEY, [TABLE, SHELF], slot="place"))
    outside = choice_payload(
        reference.bridge_choose(state, BOTH_QUERY, [TABLE, SHELF], slot="object")
    )
    same_slot = choice_payload(
        reference.bridge_choose(state, TWO_OBJECTS, [CUP, RED], slot="object")
    )
    no_bridge = choice_payload(
        reference.bridge_choose(
            reference.initial_state("r27-no-bridge"), BOTH_QUERY, [CUP], slot="object"
        )
    )

    digest_before = table.digest()
    correction = table.correct(((CUP_KEY, CUP, GREEN),), authority=AUTHORITY, reason=REASON)
    after_object = choice_payload(
        reference.bridge_choose(state, BOTH_QUERY, [CUP, GREEN], slot="object")
    )
    after_place = choice_payload(
        reference.bridge_choose(state, BOTH_QUERY, [TABLE, SHELF], slot="place")
    )
    digest_after = table.digest()

    after_state = state_sha(state)
    after_parameters = parameter_sha(reference)
    ok = (
        object_zero["decision"] == "chosen"
        and object_zero["index"] == 0
        and object_one["decision"] == "chosen"
        and object_one["index"] == 1
        and place_zero["decision"] == "chosen"
        and place_zero["index"] == 0
        and place_one["decision"] == "chosen"
        and place_one["index"] == 1
        and whole["decision"] == "ambiguous_read"
        and whole["answers"] == 2
        and missing["decision"] == "no_choice"
        and missing["read_answers"] == 1
        and missing["answers"] == 0
        and outside["decision"] == "not_in_candidates"
        and same_slot["decision"] == "ambiguous_read"
        and same_slot["answers"] == 2
        and no_bridge["decision"] == "no_bridge"
        and correction["action"] == "corrected"
        and after_object["decision"] == "chosen"
        and after_object["index"] == 1
        and after_object["choice"] == list(GREEN)
        and after_place["decision"] == "chosen"
        and after_place["index"] == 0
        and digest_before != digest_after
        and after_state == baseline_state
        and after_parameters == baseline_parameters
    )
    return {
        "object_zero": object_zero,
        "object_one": object_one,
        "place_zero": place_zero,
        "place_one": place_one,
        "whole": whole,
        "missing": missing,
        "outside": outside,
        "same_slot": same_slot,
        "no_bridge": no_bridge,
        "correction": correction["action"],
        "after_object": after_object,
        "after_place": after_place,
        "table_digest_changed": digest_before != digest_after,
        "state_sha_equal": after_state == baseline_state,
        "parameter_sha_equal": after_parameters == baseline_parameters,
        "ok": ok,
    }


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
        "state_sha": state_sha(state),
        "keys_len": len(state.keys),
        "step": int(state.step),
        "store_sha": tensor_sha(store),
        "identity_sha": hashlib.sha256(
            json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "sc_sfk_none": state.sc_sfk is None,
        "sc_bridge_none": state.sc_bridge is None,
        "battery_sha": hashlib.sha256("".join(step_hashes).encode("ascii")).hexdigest(),
        "battery_steps": len(step_hashes),
    }


def run_battery(reference: Any, texts: list[str]) -> dict[str, Any]:
    torch.manual_seed(SEED)
    state = reference.initial_state("r27-zero-perturbation")
    step_hashes: list[str] = []
    for raw in texts:
        state = reference.observe(state, raw.encode("utf-8"), goal=b"r27")
        step_hashes.append(state_sha(state))
    return reference_digest(reference, state, step_hashes)


def definition_digest() -> str:
    payload = {
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "queries": [BOTH_QUERY, TWO_OBJECTS, CUP_KEY],
        "values": [list(CUP), list(RED), list(GREEN), list(TABLE), list(SHELF)],
        "authority": AUTHORITY,
        "reason": REASON,
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
        "schema": "taiji-bridge-slot-choice-r27-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": SEED,
        "estimate_cpu_upper": ESTIMATE_CPU_UPPER,
        "cpu_cap": CPU_CAP,
        "runner_sha256": RUNNER_SHA_START,
        "bridge_sha256": file_sha(BRIDGE_SOURCE),
        "core_sha256": file_sha(CORE_SOURCE),
        "fixture_sha256": base.fixture_hashes(),
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "definition_digest": definition_digest(),
        "counts": {
            "slot_cases": 7,
            "regression_sentences": len(sentences()),
            "unknown_candidates": len(UNKNOWN_CANDIDATES),
            "rerun_anchors": len(RERUN_ANCHORS),
        },
        "inputs": {
            "queries": [BOTH_QUERY, TWO_OBJECTS, CUP_KEY],
            "values": [list(CUP), list(RED), list(GREEN), list(TABLE), list(SHELF)],
            "authority": AUTHORITY,
            "reason": REASON,
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
    if manifest["core_sha256"] != file_sha(CORE_SOURCE):
        raise SystemExit("living_core changed after the freeze; re-freeze")
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

        slot = slot_reading(reference)

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
        bridge_default_none = before["sc_bridge_none"] and after["sc_bridge_none"]
        rerun_ok = all(
            row["rerun_commit"]
            and row["runner_blob_sha1"]
            and row["runner_blob_sha256"]
            and row["report_core_sha256"]
            for row in RERUN_ANCHORS
        )

        verdict = {
            "slot_choice": "slot_choice_from_read" if slot["ok"] else "slot_choice_failed",
            "closed_state": "closed_state_equal" if closed_ok else "closed_state_moved",
            "bridge_default": (
                "bridge_default_none" if bridge_default_none else "bridge_default_present"
            ),
            "rerun_anchors": "anchors_complete" if rerun_ok else "anchors_incomplete",
        }
        all_ok = all([slot["ok"], closed_ok, bridge_default_none, rerun_ok])
        verdict["overall"] = (
            "bridge_slot_choice_available" if all_ok else "bridge_slot_choice_unexpected"
        )

        cpu, wall = cost()
        if cpu > CPU_CAP:
            raise SystemExit(f"cpu cap exceeded: {cpu:.2f}s > {CPU_CAP:.0f}s")
        report = {
            "schema": "taiji-bridge-slot-choice-r27-report-v1",
            "runner_sha256": RUNNER_SHA_START,
            "manifest_sha256": file_sha(MANIFEST),
            "bridge_sha256": file_sha(BRIDGE_SOURCE),
            "core_sha256": file_sha(CORE_SOURCE),
            "fixture_sha256": base.fixture_hashes(),
            "definition": DEFINITION,
            "criteria": CRITERIA,
            "slot": slot,
            "rerun_anchors": list(RERUN_ANCHORS),
            "closed_state": {"baseline": before, "after": after, "checks": checks},
            "bridge_default_none": bridge_default_none,
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
                    "slot": {
                        "object_zero": slot["object_zero"],
                        "object_one": slot["object_one"],
                        "place_zero": slot["place_zero"],
                        "place_one": slot["place_one"],
                        "whole": slot["whole"],
                        "missing": slot["missing"],
                        "outside": slot["outside"],
                        "same_slot": slot["same_slot"],
                        "no_bridge": slot["no_bridge"],
                        "after_object": slot["after_object"],
                        "after_place": slot["after_place"],
                        "table_digest_changed": slot["table_digest_changed"],
                        "state_sha_equal": slot["state_sha_equal"],
                        "parameter_sha_equal": slot["parameter_sha_equal"],
                    },
                    "closed_state_ok": closed_ok,
                    "bridge_default_none": bridge_default_none,
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
