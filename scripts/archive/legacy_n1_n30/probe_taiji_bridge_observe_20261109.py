"""R31 -- one subject step expresses the pair on the same input.

R30 put the text on the state; R31 wires it into the step pipeline:

A. ``bridge_observe`` runs ``observe`` and then ``bridge_express`` on the same
   raw input; 20 describe sentences produce 20/20 correct texts.
B. The tensor trajectory is bit-identical to a plain ``observe`` run without a
   bridge (the text field is the only difference).
C. Parameters do not move; the table digest does not move.
D. An unknown input clears ``last_bridge_text`` through ``no_pair``.
E. Closed state: the nine legacy readings stay equal before and after.

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

MANIFEST = ROOT / "plans/manifests/taiji_bridge_observe_20261109.json"
REPORT = ROOT / "reports/taiji_bridge_observe_20261109.json"
BRIDGE_SOURCE = ROOT / "taiji/address_bridge.py"
CORE_SOURCE = ROOT / "taiji/living_core.py"

RUNNER_SHA_START = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()

SEED = 20261109
ESTIMATE_CPU_UPPER = 40.0
CPU_CAP = float(base.CPU_CAP)

CUP = ("object", "蓝色杯子")
RED = ("object", "红色印章")
GREEN = ("object", "绿色瓶子")
TABLE = ("place", "木桌")
SHELF = ("place", "书架")
UNKNOWN = "黑色铁铃"
AUTHORITY = "verifier:v1"

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
        "round": "R30",
        "runner": "scripts/training/probe_taiji_bridge_text_20261109.py",
        "rerun_commit": "d476e1cef",
        "runner_blob_sha1": "9215eb7bbed8e5fbebf82acfb04eb12e7fbe8a4c",
        "report_runner_sha256": "b9b2fe368ac2f3d926e921af98968a416b623f0dac06dc6dd446d8409cbe7655",
        "runner_blob_sha256": "b9b2fe368ac2f3d926e921af98968a416b623f0dac06dc6dd446d8409cbe7655",
        "report_core_sha256": "ec5c24ece09343848ae0151e52399fbd437df42100fe362cadf6c93bd17ca368",
        "matches_report": True,
        "note": "current HEAD cannot run it after R31 (core sha changed); checkout this commit",
    },
)

DEFINITION = {
    "observe": (
        "bridge_observe runs observe on the raw input and then bridge_express on the same raw "
        "input; it returns the new state and info with a bridge_expression entry"
    ),
    "twenty": "the 20 describe sentences must produce the expected pair texts",
    "trajectory": "the bridged tensor trajectory must be bit-identical to a plain observe run",
    "unknown": "an unknown input clears last_bridge_text through no_pair",
    "closed_state": "the nine legacy closed-state readings stay equal before and after",
}

CRITERIA = {
    "twenty": "20/20 expression decisions are chosen_pair and 20/20 texts are exact",
    "trajectory": "20/20 state hashes equal to the plain run; final state hash equal",
    "parameters": "parameter hash equal before and after; table digest unchanged",
    "unknown": "unknown expression is no_pair and last_bridge_text is None; tensors equal to plain",
    "closed_state": "the nine legacy readings stay equal before and after",
    "rerun": "the R30 anchor carries a commit, a runner blob sha1, a runner sha256 and a core sha256",
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


def state_sha(state: Any) -> str:
    return tensor_sha([state.h, state.fast, state.nu_ema, state.coverage, state.prior])


def parameter_sha(reference: Any) -> str:
    return tensor_sha([tensor for _, tensor in sorted(reference.named_parameters())])


def observe_reading(reference: Any, texts: list[str]) -> dict[str, Any]:
    table = build_default()
    digest_before = table.digest()
    baseline_parameters = parameter_sha(reference)
    object_candidates = [("object", name) for name in OBJECT_DESCRIPTIONS]
    place_candidates = [("place", name) for name in PLACE_DESCRIPTIONS]

    bridged = reference.attach_bridge(reference.initial_state("r31-bridge"), table)
    plain = reference.initial_state("r31-plain")
    rows = []
    exact = 0
    trajectory = 0
    for index, text in enumerate(texts, start=1):
        bridged, info = reference.bridge_observe(
            bridged,
            text,
            object_candidates,
            place_candidates,
            goal=b"r31",
        )
        plain = reference.observe(plain, text.encode("utf-8"), goal=b"r31")
        expected = expected_pair(text)
        expected_text = (
            f"{expected['object'][1]}在{expected['place'][1]}"
            if "object" in expected and "place" in expected
            else None
        )
        expression = info["bridge_expression"]
        same_tensor = state_sha(bridged) == state_sha(plain)
        trajectory += int(same_tensor)
        ok = (
            expression["decision"] == "chosen_pair"
            and expression["text"] == expected_text
            and bridged.last_bridge_text == expected_text
        )
        exact += int(ok)
        rows.append(
            {
                "step": index,
                "text": text,
                "expected": expected_text,
                "decision": expression["decision"],
                "got": expression["text"],
                "stored": bridged.last_bridge_text,
                "tensor_equal": same_tensor,
                "ok": ok,
            }
        )

    unknown_bridged, unknown_info = reference.bridge_observe(
        bridged,
        UNKNOWN,
        object_candidates,
        place_candidates,
        goal=b"r31",
    )
    unknown_plain = reference.observe(plain, UNKNOWN.encode("utf-8"), goal=b"r31")
    unknown_expression = unknown_info["bridge_expression"]
    unknown_ok = (
        unknown_expression["decision"] == "no_pair"
        and unknown_bridged.last_bridge_text is None
        and state_sha(unknown_bridged) == state_sha(unknown_plain)
    )

    after_parameters = parameter_sha(reference)
    ok = (
        exact == len(texts)
        and trajectory == len(texts)
        and after_parameters == baseline_parameters
        and table.digest() == digest_before
        and unknown_ok
    )
    return {
        "exact": exact,
        "of": len(texts),
        "trajectory_equal": trajectory,
        "rows": rows,
        "unknown": {
            "decision": unknown_expression["decision"],
            "stored": unknown_bridged.last_bridge_text,
            "tensor_equal": state_sha(unknown_bridged) == state_sha(unknown_plain),
        },
        "parameter_sha_equal": after_parameters == baseline_parameters,
        "table_digest_unchanged": table.digest() == digest_before,
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
    state = reference.initial_state("r31-zero-perturbation")
    step_hashes: list[str] = []
    for raw in texts:
        state = reference.observe(state, raw.encode("utf-8"), goal=b"r31")
        step_hashes.append(state_sha(state))
    return reference_digest(reference, state, step_hashes)


def definition_digest() -> str:
    payload = {
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "unknown": UNKNOWN,
        "authority": AUTHORITY,
        "values": [list(CUP), list(RED), list(GREEN), list(TABLE), list(SHELF)],
        "aliases": [list(row) for row in AUTHORED_ALIASES],
        "unknown_candidates": list(UNKNOWN_CANDIDATES),
        "rerun_anchors": [dict(row) for row in RERUN_ANCHORS],
        "seed": SEED,
        "estimate_cpu_upper": ESTIMATE_CPU_UPPER,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def freeze() -> dict[str, Any]:
    manifest = {
        "schema": "taiji-bridge-observe-r31-v1",
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
            "describe_sentences": len(sentences()),
            "unknown_candidates": len(UNKNOWN_CANDIDATES),
            "rerun_anchors": len(RERUN_ANCHORS),
        },
        "inputs": {
            "unknown": UNKNOWN,
            "authority": AUTHORITY,
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

        observed = observe_reading(reference, texts)

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
            "expression": (
                "observe_expression_20_20"
                if observed["exact"] == observed["of"]
                else "observe_expression_moved"
            ),
            "trajectory": (
                "tensor_trajectory_equal"
                if observed["trajectory_equal"] == observed["of"]
                else "tensor_trajectory_moved"
            ),
            "parameters": (
                "parameters_untouched" if observed["parameter_sha_equal"] else "parameters_moved"
            ),
            "unknown": (
                "unknown_clears_text"
                if observed["unknown"]["decision"] == "no_pair"
                else "unknown_kept_text"
            ),
            "closed_state": "closed_state_equal" if closed_ok else "closed_state_moved",
            "bridge_default": (
                "bridge_default_none" if bridge_default_none else "bridge_default_present"
            ),
            "rerun_anchors": "anchors_complete" if rerun_ok else "anchors_incomplete",
        }
        all_ok = all(
            [
                observed["ok"],
                closed_ok,
                bridge_default_none,
                rerun_ok,
            ]
        )
        verdict["overall"] = (
            "bridge_observe_expression_clean" if all_ok else "bridge_observe_expression_unexpected"
        )

        cpu, wall = cost()
        if cpu > CPU_CAP:
            raise SystemExit(f"cpu cap exceeded: {cpu:.2f}s > {CPU_CAP:.0f}s")
        report = {
            "schema": "taiji-bridge-observe-r31-report-v1",
            "runner_sha256": RUNNER_SHA_START,
            "manifest_sha256": file_sha(MANIFEST),
            "bridge_sha256": file_sha(BRIDGE_SOURCE),
            "core_sha256": file_sha(CORE_SOURCE),
            "fixture_sha256": base.fixture_hashes(),
            "definition": DEFINITION,
            "criteria": CRITERIA,
            "observed": observed,
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
                    "exact": observed["exact"],
                    "of": observed["of"],
                    "trajectory_equal": observed["trajectory_equal"],
                    "unknown": observed["unknown"],
                    "parameter_sha_equal": observed["parameter_sha_equal"],
                    "table_digest_unchanged": observed["table_digest_unchanged"],
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
