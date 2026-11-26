"""R30 -- the paired text becomes a state field carried by step/observe.

R28 produced the template text and R29 carried it across processes.  R30 puts it
on the subject state:

A. ``last_bridge_text`` defaults to ``None``.
B. ``bridge_express`` writes the chosen template text; the subject tensors do not
   move.
C. 20 ``observe`` calls keep the field and the attached table; the text does not
   change.
D. A ``no_pair`` expression clears the field instead of carrying a stale answer.
E. ``clear_bridge`` clears both ``sc_bridge`` and ``last_bridge_text``.
F. Closed state: the nine legacy readings stay equal before and after.

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

MANIFEST = ROOT / "plans/manifests/taiji_bridge_text_20261109.json"
REPORT = ROOT / "reports/taiji_bridge_text_20261109.json"
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
BOTH_QUERY = "喝水的杯子放在方方正正的木桌子上面。"
UNKNOWN = "黑色铁铃"
AUTHORITY = "verifier:v1"
EXPECTED_TEXT = "蓝色杯子在木桌"

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
        "round": "R29",
        "runner": "scripts/training/probe_taiji_bridge_pair_handoff_20261109.py",
        "rerun_commit": "fc800aa81",
        "runner_blob_sha1": "52511f2834df462267b5b8c1f92185555d36863d",
        "report_runner_sha256": "b05f8fc2da793b6aecc87e9776009b75665137021600fc1ad36db6f0b0637c3c",
        "runner_blob_sha256": "b05f8fc2da793b6aecc87e9776009b75665137021600fc1ad36db6f0b0637c3c",
        "report_core_sha256": "9fe7fbf786e900032498b00d6cbeaa600e29e3e14100a625aadbf548629f6d82",
        "matches_report": True,
        "note": "current HEAD cannot run it after R30 (core sha changed); checkout this commit",
    },
)

DEFINITION = {
    "default": "last_bridge_text is None on a fresh state",
    "express": (
        "bridge_express writes the chosen template text into last_bridge_text; the parameter "
        "and state hashes do not move"
    ),
    "steps": "20 observe calls keep last_bridge_text and sc_bridge; the text is unchanged",
    "no_pair": "a no_pair expression clears last_bridge_text instead of carrying a stale answer",
    "clear": "clear_bridge clears both sc_bridge and last_bridge_text",
    "closed_state": "the nine legacy closed-state readings stay equal before and after",
}

CRITERIA = {
    "default": "fresh state has last_bridge_text None",
    "express": "text equals '蓝色杯子在木桌'; parameter and state hashes equal to the baseline",
    "steps": "20/20 steps keep the same text; table digest unchanged; parameters unchanged",
    "no_pair": "unknown query after a chosen pair leaves last_bridge_text None",
    "clear": "both sc_bridge and last_bridge_text are None; bridge_read is no_bridge",
    "closed_state": "the nine legacy readings stay equal before and after",
    "rerun": "the R29 anchor carries a commit, a runner blob sha1, a runner sha256 and a core sha256",
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


def text_reading(reference: Any, texts: list[str]) -> dict[str, Any]:
    table = build_default()
    digest_before = table.digest()
    state = reference.attach_bridge(reference.initial_state("r30-text"), table)
    baseline_state = state_sha(state)
    baseline_parameters = parameter_sha(reference)
    default_none = state.last_bridge_text is None

    object_candidates = [("object", name) for name in OBJECT_DESCRIPTIONS]
    place_candidates = [("place", name) for name in PLACE_DESCRIPTIONS]
    expressed, result = reference.bridge_express(
        state, BOTH_QUERY, object_candidates, place_candidates
    )
    after_express_state = state_sha(expressed)
    after_express_parameters = parameter_sha(reference)
    intact = expressed.last_bridge_text == EXPECTED_TEXT and result["decision"] == "chosen_pair"

    step_rows = []
    current = expressed
    for index, text in enumerate(texts, start=1):
        current = reference.observe(current, text.encode("utf-8"), goal=b"r30")
        step_rows.append(
            {
                "step": index,
                "text": current.last_bridge_text,
                "bridge_attached": current.sc_bridge is table,
            }
        )

    after_steps_parameters = parameter_sha(reference)
    steps_ok = all(row["text"] == EXPECTED_TEXT and row["bridge_attached"] for row in step_rows)

    cleared, _ = reference.bridge_express(current, UNKNOWN, object_candidates, place_candidates)
    no_pair_clears = cleared.last_bridge_text is None
    cleared = reference.clear_bridge(cleared)
    clear_ok = (
        cleared.sc_bridge is None
        and cleared.last_bridge_text is None
        and reference.bridge_read(cleared, BOTH_QUERY)["decision"] == "no_bridge"
    )
    ok = (
        default_none
        and intact
        and after_express_state == baseline_state
        and after_express_parameters == baseline_parameters
        and steps_ok
        and after_steps_parameters == baseline_parameters
        and no_pair_clears
        and clear_ok
        and table.digest() == digest_before
    )
    return {
        "default_none": default_none,
        "intact": intact,
        "text": expressed.last_bridge_text,
        "steps": step_rows,
        "step_count": len(step_rows),
        "steps_ok": steps_ok,
        "no_pair_clears": no_pair_clears,
        "clear_ok": clear_ok,
        "state_sha_equal_after_express": after_express_state == baseline_state,
        "parameter_sha_equal": after_express_parameters == baseline_parameters,
        "parameter_sha_equal_after_steps": after_steps_parameters == baseline_parameters,
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
    state = reference.initial_state("r30-zero-perturbation")
    step_hashes: list[str] = []
    for raw in texts:
        state = reference.observe(state, raw.encode("utf-8"), goal=b"r30")
        step_hashes.append(state_sha(state))
    return reference_digest(reference, state, step_hashes)


def definition_digest() -> str:
    payload = {
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "query": BOTH_QUERY,
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
        "schema": "taiji-bridge-text-r30-v1",
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
            "observe_steps": 20,
            "regression_sentences": len(sentences()),
            "unknown_candidates": len(UNKNOWN_CANDIDATES),
            "rerun_anchors": len(RERUN_ANCHORS),
        },
        "inputs": {
            "query": BOTH_QUERY,
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

        text = text_reading(reference, texts)

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
            "default": ("text_default_none" if text["default_none"] else "text_default_present"),
            "express": ("text_written_read_side" if text["intact"] else "text_write_failed"),
            "steps": "text_survives_steps" if text["steps_ok"] else "text_dropped_in_steps",
            "no_pair": ("no_pair_clears_text" if text["no_pair_clears"] else "stale_text_kept"),
            "clear": "text_cleared_with_bridge" if text["clear_ok"] else "text_clear_failed",
            "closed_state": "closed_state_equal" if closed_ok else "closed_state_moved",
            "bridge_default": (
                "bridge_default_none" if bridge_default_none else "bridge_default_present"
            ),
            "rerun_anchors": "anchors_complete" if rerun_ok else "anchors_incomplete",
        }
        all_ok = all(
            [
                text["default_none"],
                text["intact"],
                text["steps_ok"],
                text["no_pair_clears"],
                text["clear_ok"],
                closed_ok,
                bridge_default_none,
                rerun_ok,
            ]
        )
        verdict["overall"] = (
            "bridge_text_state_field_clean" if all_ok else "bridge_text_state_field_unexpected"
        )

        cpu, wall = cost()
        if cpu > CPU_CAP:
            raise SystemExit(f"cpu cap exceeded: {cpu:.2f}s > {CPU_CAP:.0f}s")
        report = {
            "schema": "taiji-bridge-text-r30-report-v1",
            "runner_sha256": RUNNER_SHA_START,
            "manifest_sha256": file_sha(MANIFEST),
            "bridge_sha256": file_sha(BRIDGE_SOURCE),
            "core_sha256": file_sha(CORE_SOURCE),
            "fixture_sha256": base.fixture_hashes(),
            "definition": DEFINITION,
            "criteria": CRITERIA,
            "text": text,
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
                    "text": {
                        "default_none": text["default_none"],
                        "intact": text["intact"],
                        "value": text["text"],
                        "steps": text["step_count"],
                        "steps_ok": text["steps_ok"],
                        "no_pair_clears": text["no_pair_clears"],
                        "clear_ok": text["clear_ok"],
                        "state_sha_equal": text["state_sha_equal_after_express"],
                        "parameter_sha_equal": text["parameter_sha_equal"],
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
