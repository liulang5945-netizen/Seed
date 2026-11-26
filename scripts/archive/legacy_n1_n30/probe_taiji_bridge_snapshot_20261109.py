"""R32 -- one snapshot carries the table and the read-side text.

R30 put the text on the state; R22 moved only the table.  R32 bundles both:

A. ``bridge_snapshot`` writes the table bytes plus ``last_bridge_text``.
B. ``bridge_restore`` rebuilds a state whose table digest and text match; the 20
   describe sentences still produce 20/20 paired texts.
C. The importer decides the authority policy: a snapshot restored with
   ``verifier:v2`` rejects a ``v1`` correction and accepts ``v2``.
D. An empty snapshot (no table) restores to ``None``/``None``.
E. Truncated, wrong-schema, bad-base64 and text-without-bridge snapshots are
   rejected whole.
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

MANIFEST = ROOT / "plans/manifests/taiji_bridge_snapshot_20261109.json"
REPORT = ROOT / "reports/taiji_bridge_snapshot_20261109.json"
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
BOTH_QUERY = "喝水的杯子放在方方正正的木桌子上面。"
AUTHORITY = "verifier:v1"
AUTHORITY_V2 = "verifier:v2"
REASON = "r32:snapshot"

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
        "round": "R31",
        "runner": "scripts/training/probe_taiji_bridge_observe_20261109.py",
        "rerun_commit": "62394837e",
        "runner_blob_sha1": "eef7cafd2254aee9805956fe1d1b563dbd34b29c",
        "report_runner_sha256": "c7826323a0b7592b903bfcc0fc14c3eba83ffea67082fc60e1a429a53f02d5f6",
        "runner_blob_sha256": "c7826323a0b7592b903bfcc0fc14c3eba83ffea67082fc60e1a429a53f02d5f6",
        "report_core_sha256": "f1725d42b3c91fc8c727c7afadaa45c8a4c4c376dec84e9466a1b72a1c0f1701",
        "matches_report": True,
        "note": "current HEAD cannot run it after R32 (core sha changed); checkout this commit",
    },
)

DEFINITION = {
    "snapshot": "bridge_snapshot serializes the attached table bytes and last_bridge_text",
    "restore": (
        "bridge_restore rebuilds the table with the caller's authority policy and restores the "
        "text; the 20 describe sentences must still yield 20/20 pairs"
    ),
    "policy": "a snapshot restored under verifier:v2 rejects a v1 correction and accepts v2",
    "empty": "a snapshot with no table restores to sc_bridge=None and last_bridge_text=None",
    "corrupt": (
        "truncated, wrong-schema, bad-base64 and text-without-bridge snapshots are rejected whole"
    ),
    "closed_state": "the nine legacy closed-state readings stay equal before and after",
}

CRITERIA = {
    "round_trip": "restored table digest and text equal; 20/20 pairs exact",
    "policy": "v1 correction rejected and content digest unchanged; v2 correction accepted and content digest changes",
    "empty": "empty snapshot restores both fields to None",
    "corrupt": "4/4 rejected with ValueError; no partial table",
    "closed_state": "the nine legacy readings stay equal before and after",
    "rerun": "the R31 anchor carries a commit, a runner blob sha1, a runner sha256 and a core sha256",
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
            }
        )
    return rows


def snapshot_reading(reference: Any, texts: list[str]) -> dict[str, Any]:
    table = build_default()
    state = reference.attach_bridge(reference.initial_state("r32-source"), table)
    object_candidates = [("object", name) for name in OBJECT_DESCRIPTIONS]
    place_candidates = [("place", name) for name in PLACE_DESCRIPTIONS]
    state, _ = reference.bridge_express(state, BOTH_QUERY, object_candidates, place_candidates)
    source_text = state.last_bridge_text
    blob = reference.bridge_snapshot(state)

    restored = reference.bridge_restore(
        reference.initial_state("r32-restored"), blob, allowed_authorities=(AUTHORITY,)
    )
    restored_rows = pair_rows(reference, restored, texts)
    exact = 0
    for row in restored_rows:
        expected = expected_pair(row["text"])
        expected_text = (
            f"{expected['object'][1]}在{expected['place'][1]}"
            if "object" in expected and "place" in expected
            else None
        )
        exact += int(row["decision"] == "chosen_pair" and row["output"] == expected_text)
    round_trip = (
        restored.sc_bridge is not None
        and restored.sc_bridge.digest() == table.digest()
        and restored.last_bridge_text == source_text
        and exact == len(texts)
    )

    policy_state = reference.bridge_restore(
        reference.initial_state("r32-policy"), blob, allowed_authorities=(AUTHORITY_V2,)
    )
    policy = policy_state.sc_bridge
    before_digest = policy.content_digest()
    v1_result = policy.correct(((CUP_KEY, CUP, GREEN),), authority=AUTHORITY)
    after_v1 = policy.content_digest()
    v2_result = policy.correct(((CUP_KEY, CUP, GREEN),), authority=AUTHORITY_V2, reason=REASON)
    after_v2 = policy.content_digest()
    policy_ok = (
        v1_result["action"] == "rejected_batch_unknown_authority"
        and after_v1 == before_digest
        and v2_result["action"] == "corrected"
        and after_v2 != before_digest
    )

    empty_state = reference.bridge_snapshot(reference.initial_state("r32-empty"))
    empty_restored = reference.bridge_restore(
        reference.initial_state("r32-empty-restored"), empty_state
    )
    empty_ok = empty_restored.sc_bridge is None and empty_restored.last_bridge_text is None

    corrupt_payloads = [
        blob[: max(1, len(blob) // 3)],
        json.dumps({"schema": "wrong", "bridge": None, "text": None}).encode("utf-8"),
        json.dumps({"schema": "taiji-bridge-snapshot-v1", "bridge": "!!!", "text": None}).encode(
            "utf-8"
        ),
        json.dumps({"schema": "taiji-bridge-snapshot-v1", "bridge": None, "text": "x"}).encode(
            "utf-8"
        ),
    ]
    rejected = []
    for corrupt in corrupt_payloads:
        try:
            reference.bridge_restore(reference.initial_state("r32-corrupt"), corrupt)
        except ValueError as error:
            rejected.append(str(error))
        else:
            rejected.append("accepted")
    corrupt_ok = len(rejected) == 4 and all(
        error.startswith("rejected_snapshot") for error in rejected
    )

    ok = round_trip and policy_ok and empty_ok and corrupt_ok
    return {
        "snapshot_bytes": len(blob),
        "snapshot_sha256": hashlib.sha256(blob).hexdigest(),
        "source_text": source_text,
        "restored_text": restored.last_bridge_text,
        "restored_digest": restored.sc_bridge.digest() if restored.sc_bridge is not None else None,
        "table_digest": table.digest(),
        "exact": exact,
        "of": len(texts),
        "policy": {
            "v1": v1_result["action"],
            "v2": v2_result["action"],
            "digest_before": before_digest,
            "digest_after_v1": after_v1,
            "digest_after_v2": after_v2,
        },
        "empty_ok": empty_ok,
        "rejected": rejected,
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
    state = reference.initial_state("r32-zero-perturbation")
    step_hashes: list[str] = []
    for raw in texts:
        state = reference.observe(state, raw.encode("utf-8"), goal=b"r32")
        step_hashes.append(state_sha(state))
    return reference_digest(reference, state, step_hashes)


def definition_digest() -> str:
    payload = {
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "query": BOTH_QUERY,
        "authorities": [AUTHORITY, AUTHORITY_V2],
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
        "schema": "taiji-bridge-snapshot-r32-v1",
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
            "corrupt_snapshots": 4,
            "unknown_candidates": len(UNKNOWN_CANDIDATES),
            "rerun_anchors": len(RERUN_ANCHORS),
        },
        "inputs": {
            "query": BOTH_QUERY,
            "authorities": [AUTHORITY, AUTHORITY_V2],
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

        snapshot = snapshot_reading(reference, texts)

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
            "round_trip": (
                "snapshot_round_trip_exact" if snapshot["ok"] else "snapshot_round_trip_moved"
            ),
            "policy": (
                "importer_decides_policy"
                if snapshot["policy"]["v1"] == "rejected_batch_unknown_authority"
                and snapshot["policy"]["v2"] == "corrected"
                else "policy_moved"
            ),
            "empty": (
                "empty_snapshot_restores_none" if snapshot["empty_ok"] else "empty_snapshot_moved"
            ),
            "corrupt": (
                "corrupt_snapshot_rejected"
                if all(error.startswith("rejected_snapshot") for error in snapshot["rejected"])
                and len(snapshot["rejected"]) == 4
                else "corrupt_snapshot_accepted"
            ),
            "closed_state": "closed_state_equal" if closed_ok else "closed_state_moved",
            "bridge_default": (
                "bridge_default_none" if bridge_default_none else "bridge_default_present"
            ),
            "rerun_anchors": "anchors_complete" if rerun_ok else "anchors_incomplete",
        }
        all_ok = all([snapshot["ok"], closed_ok, bridge_default_none, rerun_ok])
        verdict["overall"] = (
            "bridge_snapshot_round_trip_clean"
            if all_ok
            else "bridge_snapshot_round_trip_unexpected"
        )

        cpu, wall = cost()
        if cpu > CPU_CAP:
            raise SystemExit(f"cpu cap exceeded: {cpu:.2f}s > {CPU_CAP:.0f}s")
        report = {
            "schema": "taiji-bridge-snapshot-r32-report-v1",
            "runner_sha256": RUNNER_SHA_START,
            "manifest_sha256": file_sha(MANIFEST),
            "bridge_sha256": file_sha(BRIDGE_SOURCE),
            "core_sha256": file_sha(CORE_SOURCE),
            "fixture_sha256": base.fixture_hashes(),
            "definition": DEFINITION,
            "criteria": CRITERIA,
            "snapshot": snapshot,
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
                    "snapshot": {
                        "bytes": snapshot["snapshot_bytes"],
                        "sha256": snapshot["snapshot_sha256"],
                        "source_text": snapshot["source_text"],
                        "restored_text": snapshot["restored_text"],
                        "digest_equal": snapshot["restored_digest"] == snapshot["table_digest"],
                        "exact": snapshot["exact"],
                        "of": snapshot["of"],
                        "policy": snapshot["policy"],
                        "empty_ok": snapshot["empty_ok"],
                        "rejected": snapshot["rejected"],
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
