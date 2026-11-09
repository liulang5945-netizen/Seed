"""R24 -- attach the written bridge to the subject state, read side only.

R23 proved the table can cross a process boundary.  R24 asks the next
architectural question: can the *subject state* carry that table without
changing any tensor?

A. Closed state: ``sc_bridge`` is ``None``; the nine legacy readings stay equal.
B. Attach: ``attach_bridge`` puts the table on the state; the subject reads the
   20 describe sentences through ``bridge_read`` at 40/40, zeros stay 0/0, and
   the parameter/state hashes do not move.
C. Correct: a correction through the table changes the table, the alias reads
   the new value, and the parameter/state hashes still do not move.
D. Clear: ``clear_bridge`` returns ``sc_bridge`` to ``None``; ``bridge_read``
   answers ``no_bridge`` and the state hash returns to the baseline.
E. Handback: the table exported after the correction is imported and attached to
   a fresh state; it reads the corrected alias and 40/40.
F. Closed state: the nine readings stay equal before and after.

Zero training, zero optimizer steps, zero Adam, zero confirm.  The bridge stays a
pure Python side table: it is not in ``state_dict``, not in ``candidate_sha_v2``,
and not fed to any tensor.
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

MANIFEST = ROOT / "plans/manifests/taiji_bridge_attach_20261109.json"
REPORT = ROOT / "reports/taiji_bridge_attach_20261109.json"
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
ALIAS_KEY = "喝水的杯子"
AUTHORITY = "verifier:v1"
REASON = "r24:attach"

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

DEFINITION = {
    "attach": (
        "LivingState gains an optional sc_bridge field (default None); "
        "attach_bridge replaces the state with the table, clear_bridge removes it"
    ),
    "read": (
        "bridge_read delegates to table.read_sentence; with no table it returns "
        "decision=no_bridge and no answers"
    ),
    "read_side": (
        "attach, read and correct must not change any parameter, h, fast, nu_ema, coverage, "
        "prior, sc_sfk or the closed-state flags"
    ),
    "correction": (
        "one correction through the attached table changes the alias to RED; the table digest "
        "moves, the subject hashes do not"
    ),
    "clear": "clear_bridge returns sc_bridge to None and bridge_read answers no_bridge",
    "handback": (
        "the table exported after the correction is imported into a fresh state and attached; it "
        "reads RED and 40/40"
    ),
    "closed_state": "the nine legacy closed-state readings stay equal before and after",
}

CRITERIA = {
    "attach": (
        "sc_bridge is the same object; 20 describe sentences at 40/40; zeros 0/0; parameter and "
        "state hashes equal to the baseline"
    ),
    "correction": (
        "table digest changes; alias reads RED; parameter and state hashes still equal; 40/40"
    ),
    "clear": (
        "sc_bridge is None; bridge_read decision is no_bridge; state hash equals the baseline"
    ),
    "handback": (
        "imported table digest equals the corrected table; attached read is RED; 40/40; 0/0"
    ),
    "closed_state": "the nine legacy readings stay equal before and after",
    "reporting": (
        "the bridge is a Python side table; the report keeps table digest, parameter hash and "
        "state hash in separate columns so 'no tensor moved' is visible"
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


def subject_state_sha(state: Any) -> str:
    return tensor_sha([state.h, state.fast, state.nu_ema, state.coverage, state.prior])


def parameter_sha(reference: Any) -> str:
    return tensor_sha([tensor for _, tensor in sorted(reference.named_parameters())])


def bridge_reading(reference: Any, state: Any, sentence: str) -> dict[str, Any]:
    result = reference.bridge_read(state, sentence)
    values = [list(row["value"]) for row in result["answers"]]
    return {
        "decision": result["decision"],
        "answers": len(values),
        "values": values,
        "final_exit": result["final_exit"],
    }


def regression_reading(reference: Any, state: Any, texts: list[str]) -> dict[str, Any]:
    hits = wrong = 0
    distribution = {"answered_2": 0, "answered_1": 0, "wrong": 0, "abstained": 0}
    for text in texts:
        expected = expected_pair(text)
        result = reference.bridge_read(state, text)
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


def zeros_reading(reference: Any, state: Any) -> dict[str, Any]:
    unknown_answering = 0
    for phrase in UNKNOWN_CANDIDATES:
        unknown_answering += int(bool(reference.bridge_read(state, phrase)["answers"]))
    fragment_answering = 0
    for phrase, _ in bridge_records(OBJECT_DESCRIPTIONS, PLACE_DESCRIPTIONS):
        data = phrase.encode("utf-8")
        for offset in (1, 2):
            fragment_answering += int(bool(reference.bridge_read(state, data[offset:])["answers"]))
    return {
        "unknown_answering": unknown_answering,
        "fragment_answering": fragment_answering,
        "ok": unknown_answering == 0 and fragment_answering == 0,
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
        "state_sha": subject_state_sha(state),
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
    state = reference.initial_state("r24-zero-perturbation")
    step_hashes: list[str] = []
    for raw in texts:
        state = reference.observe(state, raw.encode("utf-8"), goal=b"r24")
        step_hashes.append(subject_state_sha(state))
    return reference_digest(reference, state, step_hashes)


def attach_reading(reference: Any, texts: list[str]) -> dict[str, Any]:
    table = build_default()
    state = reference.initial_state("r24-attach")
    baseline_state = subject_state_sha(state)
    baseline_parameters = parameter_sha(reference)
    baseline_digest = table.digest()
    attached = reference.attach_bridge(state, table)
    regression = regression_reading(reference, attached, texts)
    zeros = zeros_reading(reference, attached)
    after_read_state = subject_state_sha(attached)
    after_read_parameters = parameter_sha(reference)
    ok = (
        attached.sc_bridge is table
        and regression["ok"]
        and zeros["ok"]
        and after_read_state == baseline_state
        and after_read_parameters == baseline_parameters
        and attached.sc_sfk is None
        and table.digest() == baseline_digest
    )
    return {
        "attached": attached.sc_bridge is table,
        "regression": regression,
        "zeros": zeros,
        "state_sha_equal": after_read_state == baseline_state,
        "parameter_sha_equal": after_read_parameters == baseline_parameters,
        "sc_sfk_none": attached.sc_sfk is None,
        "table_digest_equal": table.digest() == baseline_digest,
        "ok": ok,
    }


def correction_reading(reference: Any) -> dict[str, Any]:
    table = build_default()
    state = reference.attach_bridge(reference.initial_state("r24-correct"), table)
    baseline_state = subject_state_sha(state)
    baseline_parameters = parameter_sha(reference)
    digest_before = table.digest()
    result = table.correct(((ALIAS_KEY, CUP, RED),), authority=AUTHORITY, reason=REASON)
    alias = bridge_reading(reference, state, ALIAS_KEY)
    after_state = subject_state_sha(state)
    after_parameters = parameter_sha(reference)
    ok = (
        result["action"] == "corrected"
        and alias["values"] == [list(RED)]
        and table.digest() != digest_before
        and after_state == baseline_state
        and after_parameters == baseline_parameters
        and len(table.corrections) == 1
        and table.corrections[0]["authority"] == AUTHORITY
        and table.corrections[0]["reason"] == REASON
    )
    return {
        "action": result["action"],
        "alias": alias,
        "table_digest_changed": table.digest() != digest_before,
        "state_sha_equal": after_state == baseline_state,
        "parameter_sha_equal": after_parameters == baseline_parameters,
        "log": [
            {
                "from": value_key(row["from"]),
                "to": value_key(row["to"]),
                "authority": row["authority"],
                "reason": row["reason"],
            }
            for row in table.corrections
        ],
        "ok": ok,
    }


def clear_reading(reference: Any) -> dict[str, Any]:
    table = build_default()
    state = reference.attach_bridge(reference.initial_state("r24-clear"), table)
    baseline_state = subject_state_sha(state)
    baseline_parameters = parameter_sha(reference)
    cleared = reference.clear_bridge(state)
    after = bridge_reading(reference, cleared, ALIAS_KEY)
    after_state = subject_state_sha(cleared)
    after_parameters = parameter_sha(reference)
    ok = (
        cleared.sc_bridge is None
        and after["decision"] == "no_bridge"
        and after["answers"] == 0
        and after_state == baseline_state
        and after_parameters == baseline_parameters
    )
    return {
        "sc_bridge_none": cleared.sc_bridge is None,
        "read": after,
        "state_sha_equal": after_state == baseline_state,
        "parameter_sha_equal": after_parameters == baseline_parameters,
        "ok": ok,
    }


def handback_reading(reference: Any, texts: list[str]) -> dict[str, Any]:
    table = build_default()
    table.correct(((ALIAS_KEY, CUP, RED),), authority=AUTHORITY, reason=REASON)
    imported = BridgeTable.from_state(table.export_state(), allowed_authorities=(AUTHORITY,))
    state = reference.attach_bridge(reference.initial_state("r24-handback"), imported)
    regression = regression_reading(reference, state, texts)
    zeros = zeros_reading(reference, state)
    alias = bridge_reading(reference, state, ALIAS_KEY)
    ok = (
        imported.digest() == table.digest()
        and alias["values"] == [list(RED)]
        and regression["ok"]
        and zeros["ok"]
        and state.sc_bridge is imported
    )
    return {
        "digest_equal": imported.digest() == table.digest(),
        "alias": alias,
        "regression": regression,
        "zeros": zeros,
        "ok": ok,
    }


def definition_digest() -> str:
    payload = {
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "alias_key": ALIAS_KEY,
        "authority": AUTHORITY,
        "reason": REASON,
        "values": [list(CUP), list(RED), list(GREEN), list(TABLE), list(SHELF)],
        "aliases": [list(row) for row in AUTHORED_ALIASES],
        "unknown": list(UNKNOWN_CANDIDATES),
        "seed": SEED,
        "estimate_cpu_upper": ESTIMATE_CPU_UPPER,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def freeze() -> dict[str, Any]:
    manifest = {
        "schema": "taiji-bridge-attach-r24-v1",
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
            "attach_cases": 4,
            "regression_sentences": len(sentences()),
            "unknown_candidates": len(UNKNOWN_CANDIDATES),
            "fragments": 18,
        },
        "inputs": {
            "alias_key": ALIAS_KEY,
            "authority": AUTHORITY,
            "reason": REASON,
            "values": [list(CUP), list(RED), list(GREEN), list(TABLE), list(SHELF)],
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

        attach = attach_reading(reference, texts)
        correction = correction_reading(reference)
        clear = clear_reading(reference)
        handback = handback_reading(reference, texts)

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

        verdict = {
            "attach": ("bridge_attached_read_side" if attach["ok"] else "bridge_attach_failed"),
            "correction": (
                "bridge_correction_read_side"
                if correction["ok"]
                else "bridge_correction_moved_tensors"
            ),
            "clear": "bridge_cleared" if clear["ok"] else "bridge_clear_failed",
            "handback": (
                "bridge_handback_attached" if handback["ok"] else "bridge_handback_failed"
            ),
            "closed_state": "closed_state_equal" if closed_ok else "closed_state_moved",
            "bridge_default": (
                "bridge_default_none" if bridge_default_none else "bridge_default_present"
            ),
        }
        all_ok = all(
            [
                attach["ok"],
                correction["ok"],
                clear["ok"],
                handback["ok"],
                closed_ok,
                bridge_default_none,
            ]
        )
        verdict["overall"] = "bridge_read_side_attached" if all_ok else "bridge_attach_unexpected"

        cpu, wall = cost()
        if cpu > CPU_CAP:
            raise SystemExit(f"cpu cap exceeded: {cpu:.2f}s > {CPU_CAP:.0f}s")
        report = {
            "schema": "taiji-bridge-attach-r24-report-v1",
            "runner_sha256": RUNNER_SHA_START,
            "manifest_sha256": file_sha(MANIFEST),
            "bridge_sha256": file_sha(BRIDGE_SOURCE),
            "core_sha256": file_sha(CORE_SOURCE),
            "fixture_sha256": base.fixture_hashes(),
            "definition": DEFINITION,
            "criteria": CRITERIA,
            "attach": attach,
            "correction": correction,
            "clear": clear,
            "handback": handback,
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
                    "attach": {
                        "attached": attach["attached"],
                        "regression": attach["regression"]["hits"],
                        "of": attach["regression"]["entities"],
                        "zeros": [
                            attach["zeros"]["unknown_answering"],
                            attach["zeros"]["fragment_answering"],
                        ],
                        "state_sha_equal": attach["state_sha_equal"],
                        "parameter_sha_equal": attach["parameter_sha_equal"],
                    },
                    "correction": {
                        "action": correction["action"],
                        "alias": correction["alias"]["values"],
                        "table_digest_changed": correction["table_digest_changed"],
                        "state_sha_equal": correction["state_sha_equal"],
                        "parameter_sha_equal": correction["parameter_sha_equal"],
                    },
                    "clear": {
                        "sc_bridge_none": clear["sc_bridge_none"],
                        "read": clear["read"]["decision"],
                        "state_sha_equal": clear["state_sha_equal"],
                    },
                    "handback": {
                        "digest_equal": handback["digest_equal"],
                        "alias": handback["alias"]["values"],
                        "regression": handback["regression"]["hits"],
                        "zeros": [
                            handback["zeros"]["unknown_answering"],
                            handback["zeros"]["fragment_answering"],
                        ],
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
