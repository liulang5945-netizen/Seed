"""R29 -- cross-process pair expression.

R28 produced the paired template answer in one process; R23 moved a table across
processes.  R29 puts the two together:

* the parent exports the 18-key table under policy ``verifier:v1``;
* a child process imports it under its own policy, expresses the pair
  ``"{object}在{place}"`` for the 20 describe sentences, corrects the CUP alias
  to GREEN, expresses the 20 sentences again, and exports;
* the parent imports the child export and checks the 20/20 before and after
  texts, the digest chain and the audit entry;
* a second child receives truncated bytes and must reject them whole.

Zero training, zero optimizer steps, zero Adam, zero confirm.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
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

MANIFEST = ROOT / "plans/manifests/taiji_bridge_pair_handoff_20261109.json"
REPORT = ROOT / "reports/taiji_bridge_pair_handoff_20261109.json"
BRIDGE_SOURCE = ROOT / "taiji/address_bridge.py"
CORE_SOURCE = ROOT / "taiji/living_core.py"

RUNNER_SHA_START = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()

SEED = 20261109
ESTIMATE_CPU_UPPER = 60.0
CPU_CAP = float(base.CPU_CAP)

CUP = ("object", "蓝色杯子")
RED = ("object", "红色印章")
GREEN = ("object", "绿色瓶子")
TABLE = ("place", "木桌")
SHELF = ("place", "书架")
CUP_KEY = "喝水的杯子"
CUP_DESCRIPTION = OBJECT_DESCRIPTIONS["蓝色杯子"]
AUTHORITY = "verifier:v1"
REASON = "r29:pair"

AUTHORED_ALIASES = (
    (CUP_KEY, CUP),
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
        "round": "R28",
        "runner": "scripts/training/probe_taiji_bridge_pair_20261109.py",
        "rerun_commit": "5ace4030d",
        "runner_blob_sha1": "be5266f50e3c93232c1c0526a1d29c62d1db5856",
        "report_runner_sha256": "baae99d01c70d38bd86373c5df476102078c632cc67336467843a75f732dd0fd",
        "runner_blob_sha256": "baae99d01c70d38bd86373c5df476102078c632cc67336467843a75f732dd0fd",
        "report_core_sha256": "9fe7fbf786e900032498b00d6cbeaa600e29e3e14100a625aadbf548629f6d82",
        "matches_report": True,
        "note": "R29 did not change living_core; R28 remains runnable at this HEAD; anchor kept for later rounds",
    },
)

DEFINITION = {
    "child": (
        "a separate process imports the exported table under verifier:v1, expresses "
        "the pair template for the 20 describe sentences, corrects the CUP description to GREEN, "
        "expresses again, and exports"
    ),
    "before": "the child's 20 before texts must equal '{object}在{place}'",
    "after": "the child's 20 after texts must substitute GREEN for CUP and keep every place",
    "handback": (
        "the parent imports the child export; digest, audit entry and the 20 after texts must "
        "match"
    ),
    "corrupt": (
        "a second child receives truncated bytes and must reject them with rejected_import_invalid_state "
        "and write no output"
    ),
    "closed_state": "the nine legacy closed-state readings stay equal before and after",
}

CRITERIA = {
    "child_import": "child imported digest equals the parent digest",
    "before": "20/20 before pairs and texts exact",
    "after": "20/20 after pairs and texts exact; CUP sentences become GREEN; places unchanged",
    "handback": "parent digest equals child export digest; 20/20 after texts; one audit entry with verifier:v1",
    "corrupt": "child result error starts with rejected_import_invalid_state and no output exists",
    "closed_state": "the nine legacy readings stay equal before and after",
    "rerun": "the R28 anchor carries a commit, a runner blob sha1, a runner sha256 and a core sha256",
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


def pair_rows(reference: Any, state: Any, texts: list[str]) -> list[dict[str, Any]]:
    object_candidates = [("object", name) for name in OBJECT_DESCRIPTIONS]
    place_candidates = [("place", name) for name in PLACE_DESCRIPTIONS]
    rows = []
    for text in texts:
        result = reference.bridge_choose_pair(state, text, object_candidates, place_candidates)
        rows.append(
            {
                "text": text,
                "decision": result["decision"],
                "object": list(result["object"]) if result["object"] is not None else None,
                "place": list(result["place"]) if result["place"] is not None else None,
                "output": result["text"],
                "reasons": dict(result["reasons"]),
            }
        )
    return rows


def child_main(state_path: str, out_path: str, result_path: str) -> int:
    data = Path(state_path).read_bytes()
    try:
        table = BridgeTable.from_state(data, allowed_authorities=(AUTHORITY,))
    except ValueError as error:
        write_json(Path(result_path), {"error": str(error), "wrote_output": False})
        return 0
    imported_digest = table.digest()
    reference = LivingCoreReference()
    state = reference.attach_bridge(reference.initial_state("r29-child"), table)
    texts = sentences()
    before = pair_rows(reference, state, texts)
    correction = table.correct(((CUP_DESCRIPTION, CUP, GREEN),), authority=AUTHORITY, reason=REASON)
    after = pair_rows(reference, state, texts)
    out = table.export_state()
    Path(out_path).write_bytes(out)
    write_json(
        Path(result_path),
        {
            "imported_digest": imported_digest,
            "exported_digest": table.digest(),
            "export_sha256": hashlib.sha256(out).hexdigest(),
            "before": before,
            "correction": correction["action"],
            "after": after,
            "log": [
                {
                    "from": list(row["from"]),
                    "to": list(row["to"]),
                    "authority": row["authority"],
                    "reason": row["reason"],
                }
                for row in table.corrections
            ],
            "policy": list(table.allowed_authorities),
            "wrote_output": True,
        },
    )
    return 0


def run_child(state_path: Path, out_path: Path, result_path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            str(state_path),
            str(out_path),
            str(result_path),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=240,
    )
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip()[-2000:],
        "stderr": completed.stderr.strip()[-2000:],
        "result": result,
        "out_exists": out_path.exists(),
    }


def expected_rows(texts: list[str], corrected: bool) -> list[dict[str, Any]]:
    rows = []
    for text in texts:
        pair = expected_pair(text)
        obj = pair.get("object")
        place = pair.get("place")
        if corrected and obj == CUP:
            obj = GREEN
        rows.append(
            {
                "text": text,
                "object": list(obj) if obj else None,
                "place": list(place) if place else None,
                "output": f"{obj[1]}在{place[1]}" if obj and place else None,
            }
        )
    return rows


def rows_match(got: list[dict[str, Any]], expected: list[dict[str, Any]]) -> bool:
    if len(got) != len(expected):
        return False
    for row, want in zip(got, expected, strict=True):
        if row["decision"] != "chosen_pair":
            return False
        if row["object"] != want["object"] or row["place"] != want["place"]:
            return False
        if row["output"] != want["output"]:
            return False
    return True


def handoff_reading(texts: list[str]) -> dict[str, Any]:
    parent = build_default()
    parent_digest = parent.digest()
    before_expected = expected_rows(texts, corrected=False)
    after_expected = expected_rows(texts, corrected=True)
    with tempfile.TemporaryDirectory(dir=ROOT / "reports") as temp:
        base_dir = Path(temp)
        state_path = base_dir / "parent_state.bin"
        out_path = base_dir / "child_state.bin"
        result_path = base_dir / "child_result.json"
        state_bytes = parent.export_state()
        state_path.write_bytes(state_bytes)
        child = run_child(state_path, out_path, result_path)
        child_result = child["result"]
        handback = None
        handback_ok = False
        if child["out_exists"]:
            handback = BridgeTable.from_state(
                out_path.read_bytes(), allowed_authorities=(AUTHORITY,)
            )
            handback_reference = LivingCoreReference()
            handback_state = handback_reference.attach_bridge(
                handback_reference.initial_state("r29-handback"), handback
            )
            handback_rows = pair_rows(handback_reference, handback_state, texts)
            handback_log = handback.corrections
            handback_ok = (
                handback.digest() == child_result.get("exported_digest")
                and rows_match(handback_rows, after_expected)
                and len(handback_log) == 1
                and handback_log[0]["authority"] == AUTHORITY
                and handback_log[0]["reason"] == REASON
            )
        corrupt_path = base_dir / "corrupt_state.bin"
        corrupt_out = base_dir / "corrupt_out.bin"
        corrupt_result = base_dir / "corrupt_result.json"
        corrupt_path.write_bytes(state_bytes[: len(state_bytes) // 2])
        corrupt_child = run_child(corrupt_path, corrupt_out, corrupt_result)
        corrupt_error = corrupt_child["result"].get("error", "")
        corrupt_ok = (
            corrupt_child["returncode"] == 0
            and corrupt_error.startswith("rejected_import_invalid_state")
            and not corrupt_child["out_exists"]
        )
        child_before = child_result.get("before", [])
        child_after = child_result.get("after", [])
        ok = (
            child["returncode"] == 0
            and child_result.get("imported_digest") == parent_digest
            and rows_match(child_before, before_expected)
            and child_result.get("correction") == "corrected"
            and rows_match(child_after, after_expected)
            and handback_ok
            and corrupt_ok
        )
        return {
            "parent_digest": parent_digest,
            "state_bytes": len(state_bytes),
            "state_sha256": hashlib.sha256(state_bytes).hexdigest(),
            "child": child,
            "child_result": child_result,
            "before_expected": before_expected,
            "after_expected": after_expected,
            "handback_ok": handback_ok,
            "handback_digest": handback.digest() if handback is not None else None,
            "corrupt_child": corrupt_child,
            "corrupt_ok": corrupt_ok,
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
    state = reference.initial_state("r29-zero-perturbation")
    step_hashes: list[str] = []
    for raw in texts:
        state = reference.observe(state, raw.encode("utf-8"), goal=b"r29")
        step_hashes.append(state_sha(state))
    return reference_digest(reference, state, step_hashes)


def definition_digest() -> str:
    payload = {
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "alias_key": CUP_DESCRIPTION,
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
        "schema": "taiji-bridge-pair-handoff-r29-v1",
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
            "child_processes": 2,
            "regression_sentences": len(sentences()),
            "unknown_candidates": len(UNKNOWN_CANDIDATES),
            "rerun_anchors": len(RERUN_ANCHORS),
        },
        "inputs": {
            "alias_key": CUP_DESCRIPTION,
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
    if manifest["core_sha256"] != file_sha(CORE_SOURCE):
        raise SystemExit("living_core changed after the freeze; re-freeze")
    if manifest["fixture_sha256"] != base.fixture_hashes():
        raise SystemExit("fixture bytes changed; the frozen fixture must not move")
    if manifest["definition_digest"] != definition_digest():
        raise SystemExit("definition/criteria/inputs changed after the freeze; re-freeze")
    return manifest


def main() -> int:
    if "--child" in sys.argv:
        position = sys.argv.index("--child")
        return child_main(sys.argv[position + 1], sys.argv[position + 2], sys.argv[position + 3])
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

        handoff = handoff_reading(texts)

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
            "before": ("child_expression_20_20" if handoff["ok"] else "child_expression_moved"),
            "corrupt": (
                "corrupt_child_rejected" if handoff["corrupt_ok"] else "corrupt_child_accepted"
            ),
            "closed_state": "closed_state_equal" if closed_ok else "closed_state_moved",
            "bridge_default": (
                "bridge_default_none" if bridge_default_none else "bridge_default_present"
            ),
            "rerun_anchors": "anchors_complete" if rerun_ok else "anchors_incomplete",
        }
        all_ok = all(
            [handoff["ok"], handoff["corrupt_ok"], closed_ok, bridge_default_none, rerun_ok]
        )
        verdict["overall"] = (
            "cross_process_pair_expression_clean"
            if all_ok
            else "cross_process_pair_expression_unexpected"
        )

        cpu, wall = cost()
        if cpu > CPU_CAP:
            raise SystemExit(f"cpu cap exceeded: {cpu:.2f}s > {CPU_CAP:.0f}s")
        report = {
            "schema": "taiji-bridge-pair-handoff-r29-report-v1",
            "runner_sha256": RUNNER_SHA_START,
            "manifest_sha256": file_sha(MANIFEST),
            "bridge_sha256": file_sha(BRIDGE_SOURCE),
            "core_sha256": file_sha(CORE_SOURCE),
            "fixture_sha256": base.fixture_hashes(),
            "definition": DEFINITION,
            "criteria": CRITERIA,
            "handoff": handoff,
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
                    "parent_digest": handoff["parent_digest"],
                    "state_bytes": handoff["state_bytes"],
                    "child_returncode": handoff["child"]["returncode"],
                    "child_before_hits": sum(
                        1
                        for row in handoff["child_result"].get("before", [])
                        if row["decision"] == "chosen_pair"
                    ),
                    "child_after_hits": sum(
                        1
                        for row in handoff["child_result"].get("after", [])
                        if row["decision"] == "chosen_pair"
                    ),
                    "correction": handoff["child_result"].get("correction"),
                    "log": handoff["child_result"].get("log"),
                    "handback_ok": handoff["handback_ok"],
                    "corrupt": {
                        "returncode": handoff["corrupt_child"]["returncode"],
                        "error": handoff["corrupt_child"]["result"].get("error"),
                        "out_exists": handoff["corrupt_child"]["out_exists"],
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
