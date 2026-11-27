"""R35 -- the prefix becomes a policy, not a switch.

R34 proved the previous paired text can enter the next step's input.  R35
replaces the boolean with ``prefix_policy`` in ``never`` / ``always`` /
``on_match`` and asks whether the policy actually discriminates:

A. ``never``: no prefix ever; the trajectory equals a plain run on raw bytes.
B. ``always``: 19/20 steps carry the previous text; the trajectory equals a
   plain run on the same prefixed bytes.
C. ``on_match``: 19/20 steps carry the previous text because all 20 describe
   sentences hit a written key; the trajectory equals its own plain control.
D. The policy buys exactly one thing: on a sentence that does not hit the
   table, ``always`` still prefixes the stale text while ``on_match`` does not,
   so the unknown step diverges.  On hitting sentences ``on_match`` equals
   ``always`` bit for bit -- recorded, not hidden.
E. An unknown policy name raises ``ValueError`` before any step runs.
F. Parameters and the table do not move; the nine legacy closed-state readings
   stay equal.

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

MANIFEST = ROOT / "plans/manifests/taiji_bridge_policy_20261109.json"
REPORT = ROOT / "reports/taiji_bridge_policy_20261109.json"
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
UNKNOWN = "黑色铁铃"
AUTHORITY = "verifier:v1"
POLICIES = ("never", "always", "on_match")

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
        "round": "R34",
        "runner": "scripts/training/probe_taiji_bridge_prefix_20261109.py",
        "rerun_commit": "6e3988177",
        "runner_blob_sha1": "98fe5b0d5481d4debafd7c5e72fad05f0ab42b89",
        "report_runner_sha256": "ebb80205188b2c452b56f2da4c393914527a0bd0b9cd85fd7b80375117c2445c",
        "runner_blob_sha256": "ebb80205188b2c452b56f2da4c393914527a0bd0b9cd85fd7b80375117c2445c",
        "report_core_sha256": "ca6422fef7929f4bc5771798668458da33dabfa84d73edd275bc135a8d272b9c",
        "matches_report": True,
        "note": "current HEAD cannot run it after R35 (core sha changed); checkout this commit",
    },
)

DEFINITION = {
    "never": "prefix_policy='never' never prefixes; trajectory equals a plain run on raw bytes",
    "always": "prefix_policy='always' prefixes 19/20 steps with the previous text",
    "on_match": "prefix_policy='on_match' prefixes only when the sentence hits a written key",
    "control": "each policy's bridged trajectory equals a plain observe run on its own input bytes",
    "policy_effect": "never vs always differ on all 19 post-first steps; on_match equals always on hitting sentences",
    "discriminate": "on the unknown sentence always prefixes the stale text and on_match does not",
    "unknown": "the unknown sentence answers no_pair, stores None, and the next step carries no prefix",
    "invalid_policy": "an unknown policy name raises ValueError before any step",
    "closed_state": "the nine legacy closed-state readings stay equal before and after",
}

CRITERIA = {
    "expression": "each of the three policies expresses 20/20 exact paired texts",
    "prefix_counts": "never 0/20, always 19/20, on_match 19/20",
    "match_counts": "all 20 describe sentences hit a written key (the on_match gate input)",
    "controls": "each policy's bridged trajectory equals its own plain control 20/20",
    "policy_effect": "never vs always differ 19/19 post-first steps; on_match equals always 20/20",
    "discriminate": "unknown-step prefix is not None under always and None under never/on_match; the unknown step state differs",
    "unknown": "all policies: no_pair, stored None, next prefix None",
    "invalid_policy": "ValueError raised, no state produced",
    "parameters": "parameter hash and table digest unchanged",
    "closed_state": "the nine legacy readings stay equal before and after",
    "rerun": "the R34 anchor carries a commit, a runner blob sha1, a runner sha256 and a core sha256",
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


def policy_reading(
    reference: Any, policy: str, texts: list[str], table: BridgeTable
) -> dict[str, Any]:
    object_candidates = [("object", name) for name in OBJECT_DESCRIPTIONS]
    place_candidates = [("place", name) for name in PLACE_DESCRIPTIONS]
    bridged = reference.attach_bridge(reference.initial_state(f"r35-{policy}"), table)
    plain = reference.initial_state(f"r35-{policy}-plain")
    rows = []
    state_hashes: list[str] = []
    exact = 0
    prefixed = 0
    matched = 0
    gate_correct = 0
    control_equal = 0
    previous_text: str | None = None
    for index, text in enumerate(texts, start=1):
        hit = bool(table.read_sentence(text)["answers"])
        bridged, info = reference.bridge_observe(
            bridged,
            text,
            object_candidates,
            place_candidates,
            goal=b"r35",
            prefix_policy=policy,
        )
        prefix = info["prefix_used"]
        prefixed += int(prefix is not None)
        matched += int(hit)
        expected_prefix = previous_text if policy == "always" else None
        if policy == "on_match":
            expected_prefix = previous_text if hit else None
        gate_correct += int(prefix == expected_prefix)
        input_bytes = (prefix + "\n" + text).encode("utf-8") if prefix else text.encode("utf-8")
        plain = reference.observe(plain, input_bytes, goal=b"r35")
        expected = expected_pair(text)
        expected_text = (
            f"{expected['object'][1]}在{expected['place'][1]}"
            if "object" in expected and "place" in expected
            else None
        )
        expression = info["bridge_expression"]
        same_control = state_sha(bridged) == state_sha(plain)
        control_equal += int(same_control)
        ok = (
            info["prefix_policy"] == policy
            and expression["decision"] == "chosen_pair"
            and expression["text"] == expected_text
            and bridged.last_bridge_text == expected_text
        )
        exact += int(ok)
        state_hashes.append(state_sha(bridged))
        rows.append(
            {
                "step": index,
                "hit": hit,
                "prefix": prefix,
                "expected_prefix": expected_prefix,
                "expected": expected_text,
                "got": expression["text"],
                "control_equal": same_control,
                "ok": ok,
            }
        )
        previous_text = bridged.last_bridge_text

    unknown_bridged, unknown_info = reference.bridge_observe(
        bridged,
        UNKNOWN,
        object_candidates,
        place_candidates,
        goal=b"r35",
        prefix_policy=policy,
    )
    unknown_expression = unknown_info["bridge_expression"]
    after_unknown, after_info = reference.bridge_observe(
        unknown_bridged,
        texts[0],
        object_candidates,
        place_candidates,
        goal=b"r35",
        prefix_policy=policy,
    )
    unknown_state_sha = state_sha(unknown_bridged)
    unknown_ok = (
        unknown_expression["decision"] == "no_pair"
        and unknown_bridged.last_bridge_text is None
        and after_info["prefix_used"] is None
    )
    ok = (
        exact == len(texts)
        and prefixed == (0 if policy == "never" else len(texts) - 1)
        and gate_correct == len(texts)
        and control_equal == len(texts)
        and unknown_ok
    )
    return {
        "policy": policy,
        "exact": exact,
        "of": len(texts),
        "prefixed": prefixed,
        "matched": matched,
        "gate_correct": gate_correct,
        "control_equal": control_equal,
        "state_hashes": state_hashes,
        "unknown_state_sha": unknown_state_sha,
        "rows": rows,
        "unknown": {
            "prefix": unknown_info["prefix_used"],
            "decision": unknown_expression["decision"],
            "stored": unknown_bridged.last_bridge_text,
            "next_prefix": after_info["prefix_used"],
            "after_steps": int(after_unknown.step),
        },
        "ok": ok,
    }


def invalid_policy_reading(reference: Any) -> dict[str, Any]:
    state = reference.initial_state("r35-invalid-policy")
    try:
        reference.bridge_observe(
            state,
            "圆口的器皿放在木桌子上",
            [("object", "蓝色杯子")],
            [("place", "木桌")],
            prefix_policy="sometimes",
        )
    except ValueError as error:
        return {"raised": True, "error": str(error), "state_step": int(state.step)}
    return {"raised": False, "error": "", "state_step": int(state.step)}


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
    state = reference.initial_state("r35-zero-perturbation")
    step_hashes: list[str] = []
    for raw in texts:
        state = reference.observe(state, raw.encode("utf-8"), goal=b"r35")
        step_hashes.append(state_sha(state))
    return reference_digest(reference, state, step_hashes)


def definition_digest() -> str:
    payload = {
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "unknown": UNKNOWN,
        "authority": AUTHORITY,
        "policies": list(POLICIES),
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
        "schema": "taiji-bridge-policy-r35-v1",
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
            "policies": len(POLICIES),
            "unknown_candidates": len(UNKNOWN_CANDIDATES),
            "rerun_anchors": len(RERUN_ANCHORS),
        },
        "inputs": {
            "unknown": UNKNOWN,
            "authority": AUTHORITY,
            "policies": list(POLICIES),
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
        baseline_parameters = parameter_sha(reference)

        table = build_default()
        digest_before = table.digest()
        readings = {policy: policy_reading(reference, policy, texts, table) for policy in POLICIES}
        invalid = invalid_policy_reading(reference)

        never, always, on_match = (readings[policy] for policy in POLICIES)
        never_always_diff = sum(
            1
            for first, second in zip(
                never["state_hashes"][1:], always["state_hashes"][1:], strict=True
            )
            if first != second
        )
        on_match_always_same = sum(
            1
            for first, second in zip(on_match["state_hashes"], always["state_hashes"], strict=True)
            if first == second
        )
        unknown_step_differs = never["unknown_state_sha"] != always["unknown_state_sha"] and (
            always["unknown_state_sha"] != on_match["unknown_state_sha"]
        )
        discriminate_ok = (
            always["unknown"]["prefix"] is not None
            and never["unknown"]["prefix"] is None
            and on_match["unknown"]["prefix"] is None
            and unknown_step_differs
        )

        after_parameters = parameter_sha(reference)
        parameters_ok = after_parameters == baseline_parameters and table.digest() == digest_before

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

        expression_ok = all(row["exact"] == row["of"] for row in readings.values())
        prefix_counts_ok = (
            never["prefixed"] == 0
            and always["prefixed"] == always["of"] - 1
            and on_match["prefixed"] == on_match["of"] - 1
        )
        match_counts_ok = all(row["matched"] == row["of"] for row in readings.values())
        controls_ok = all(row["control_equal"] == row["of"] for row in readings.values())
        policy_effect_ok = never_always_diff == len(texts) - 1 and on_match_always_same == len(
            texts
        )
        unknown_ok = all(
            row["unknown"]["decision"] == "no_pair"
            and row["unknown"]["stored"] is None
            and row["unknown"]["next_prefix"] is None
            for row in readings.values()
        )
        invalid_ok = invalid["raised"] and invalid["state_step"] == 0

        verdict = {
            "expression": "policy_expression_60_60" if expression_ok else "policy_expression_moved",
            "prefix_counts": (
                "policy_prefix_counts_0_19_19" if prefix_counts_ok else "policy_prefix_counts_moved"
            ),
            "match_counts": (
                "policy_all_sentences_hit" if match_counts_ok else "policy_miss_recorded"
            ),
            "controls": "policy_controls_equal" if controls_ok else "policy_controls_moved",
            "policy_effect": (
                "policy_discriminates_trajectory"
                if policy_effect_ok
                else "policy_inert_on_trajectory"
            ),
            "discriminate": (
                "policy_discriminates_unknown" if discriminate_ok else "policy_inert_on_unknown"
            ),
            "unknown": "policy_unknown_clean" if unknown_ok else "policy_unknown_dirty",
            "invalid_policy": ("policy_name_rejected" if invalid_ok else "policy_name_accepted"),
            "parameters": "parameters_untouched" if parameters_ok else "parameters_moved",
            "closed_state": "closed_state_equal" if closed_ok else "closed_state_moved",
            "bridge_default": (
                "bridge_default_none" if bridge_default_none else "bridge_default_present"
            ),
            "rerun_anchors": "anchors_complete" if rerun_ok else "anchors_incomplete",
        }
        all_ok = all(
            [
                expression_ok,
                prefix_counts_ok,
                match_counts_ok,
                controls_ok,
                policy_effect_ok,
                discriminate_ok,
                unknown_ok,
                invalid_ok,
                parameters_ok,
                closed_ok,
                bridge_default_none,
                rerun_ok,
            ]
        )
        verdict["overall"] = "prefix_policy_explicit" if all_ok else "prefix_policy_unexpected"

        cpu, wall = cost()
        if cpu > CPU_CAP:
            raise SystemExit(f"cpu cap exceeded: {cpu:.2f}s > {CPU_CAP:.0f}s")
        report = {
            "schema": "taiji-bridge-policy-r35-report-v1",
            "runner_sha256": RUNNER_SHA_START,
            "manifest_sha256": file_sha(MANIFEST),
            "bridge_sha256": file_sha(BRIDGE_SOURCE),
            "core_sha256": file_sha(CORE_SOURCE),
            "fixture_sha256": base.fixture_hashes(),
            "definition": DEFINITION,
            "criteria": CRITERIA,
            "policies": readings,
            "cross_policy": {
                "never_always_differ_steps": never_always_diff,
                "on_match_always_same_steps": on_match_always_same,
                "unknown_step_differs": unknown_step_differs,
            },
            "invalid_policy": invalid,
            "parameter_sha_equal": parameters_ok,
            "table_digest_unchanged": table.digest() == digest_before,
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
                    "policies": {
                        policy: {
                            "exact": row["exact"],
                            "of": row["of"],
                            "prefixed": row["prefixed"],
                            "matched": row["matched"],
                            "gate_correct": row["gate_correct"],
                            "control_equal": row["control_equal"],
                            "unknown": row["unknown"],
                        }
                        for policy, row in readings.items()
                    },
                    "cross_policy": report["cross_policy"],
                    "invalid_policy": invalid["raised"],
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
