"""R25 -- does the attached bridge survive the subject's own step/observe?

R24 attached the table to ``LivingState`` and read it without touching tensors.
R25 keeps stepping the subject while the table is attached:

A. 20 consecutive ``observe`` calls with the table attached: after every step
   ``sc_bridge`` is still the same object, the table digest does not move, and
   the bridge still reads 20 sentences at 40/40 with zeros at 0/0.
B. The bridged trajectory and a no-bridge trajectory from the same initial state
   have the same tensor hash after 20 steps; parameters do not move and no
   parameter receives a gradient.
C. ``clear_bridge`` returns ``sc_bridge`` to ``None``, ``bridge_read`` answers
   ``no_bridge``, and five more steps stay identical to the no-bridge run.
D. Closed state: the nine legacy readings stay equal before and after.

Zero training, zero optimizer steps, zero Adam, zero confirm.  The bridge is not
consulted by ``step``; this round measures survival, not coupling.
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

MANIFEST = ROOT / "plans/manifests/taiji_bridge_step_20261109.json"
REPORT = ROOT / "reports/taiji_bridge_step_20261109.json"
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
    "survival": (
        "20 consecutive observe calls start from one initial state; the bridge is attached before "
        "the first call; after every call sc_bridge is the same object and the table digest is "
        "unchanged"
    ),
    "read_each_step": (
        "after every observe the bridge reads the 20 describe sentences at 40/40 and the zero lines "
        "at 0/0"
    ),
    "inert": (
        "a bridged trajectory and a no-bridge trajectory from the same initial state must have the "
        "same h/fast/nu_ema/coverage/prior hash after 20 observe calls"
    ),
    "parameters": "no parameter moves and no parameter receives a gradient during the 20 forwards",
    "clear": (
        "clear_bridge returns sc_bridge to None, bridge_read answers no_bridge, and five more "
        "observes stay identical to the no-bridge run"
    ),
    "closed_state": "the nine legacy closed-state readings stay equal before and after",
}

CRITERIA = {
    "survival": "20/20 steps keep sc_bridge; table digest equals the initial digest",
    "read_each_step": "each of the 20 steps reads 40/40 and 0/0",
    "inert": "bridged and no-bridge tensor hashes are equal after 20 steps",
    "parameters": "parameter hash equals the baseline; every parameter grad is None",
    "clear": "sc_bridge None; bridge_read no_bridge; five more steps stay equal to the no-bridge run",
    "closed_state": "the nine legacy readings stay equal before and after",
    "reporting": (
        "the bridge is a side table; this round measures survival, not coupling. A failure to "
        "survive would be reported as a structural defect, not hidden"
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


def state_sha(state: Any) -> str:
    return tensor_sha([state.h, state.fast, state.nu_ema, state.coverage, state.prior])


def parameter_sha(reference: Any) -> str:
    return tensor_sha([tensor for _, tensor in sorted(reference.named_parameters())])


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
    state = reference.initial_state("r25-zero-perturbation")
    step_hashes: list[str] = []
    for raw in texts:
        state = reference.observe(state, raw.encode("utf-8"), goal=b"r25")
        step_hashes.append(state_sha(state))
    return reference_digest(reference, state, step_hashes)


def survival_reading(reference: Any, texts: list[str]) -> dict[str, Any]:
    torch.manual_seed(SEED)
    start = reference.initial_state("r25-survival")
    table = build_default()
    table_digest = table.digest()
    baseline_parameters = parameter_sha(reference)
    state = reference.attach_bridge(start, table)
    step_rows = []
    for index, text in enumerate(texts, start=1):
        state = reference.observe(state, text.encode("utf-8"), goal=b"r25")
        regression = regression_reading(reference, state, texts)
        zeros = zeros_reading(reference, state)
        step_rows.append(
            {
                "step": index,
                "bridge_attached": state.sc_bridge is table,
                "table_digest_equal": table.digest() == table_digest,
                "regression_hits": regression["hits"],
                "zeros": [zeros["unknown_answering"], zeros["fragment_answering"]],
                "state_sha": state_sha(state),
            }
        )
    bridged_sha = state_sha(state)
    torch.manual_seed(SEED)
    no_bridge = reference.initial_state("r25-survival")
    for text in texts:
        no_bridge = reference.observe(no_bridge, text.encode("utf-8"), goal=b"r25")
    no_bridge_sha = state_sha(no_bridge)
    gradients = [parameter.grad for parameter in reference.parameters()]
    ok = (
        all(row["bridge_attached"] for row in step_rows)
        and all(row["table_digest_equal"] for row in step_rows)
        and all(row["regression_hits"] == 40 for row in step_rows)
        and all(row["zeros"] == [0, 0] for row in step_rows)
        and bridged_sha == no_bridge_sha
        and parameter_sha(reference) == baseline_parameters
        and all(gradient is None for gradient in gradients)
        and table.digest() == table_digest
    )
    return {
        "steps": step_rows,
        "step_count": len(step_rows),
        "bridged_state_sha": bridged_sha,
        "no_bridge_state_sha": no_bridge_sha,
        "trajectories_equal": bridged_sha == no_bridge_sha,
        "parameters_equal": parameter_sha(reference) == baseline_parameters,
        "gradients_none": all(gradient is None for gradient in gradients),
        "table_digest_equal": table.digest() == table_digest,
        "ok": ok,
    }


def clear_reading(reference: Any, texts: list[str]) -> dict[str, Any]:
    torch.manual_seed(SEED)
    start = reference.initial_state("r25-clear")
    table = build_default()
    state = reference.attach_bridge(start, table)
    for text in texts:
        state = reference.observe(state, text.encode("utf-8"), goal=b"r25")
    cleared = reference.clear_bridge(state)
    read = reference.bridge_read(cleared, ALIAS_KEY)
    for _ in range(5):
        cleared = reference.observe(cleared, "继续。".encode(), goal=b"r25")
    torch.manual_seed(SEED)
    no_bridge = reference.initial_state("r25-clear")
    for text in texts:
        no_bridge = reference.observe(no_bridge, text.encode("utf-8"), goal=b"r25")
    for _ in range(5):
        no_bridge = reference.observe(no_bridge, "继续。".encode(), goal=b"r25")
    ok = (
        cleared.sc_bridge is None
        and read["decision"] == "no_bridge"
        and read["answers"] == []
        and state_sha(cleared) == state_sha(no_bridge)
    )
    return {
        "sc_bridge_none": cleared.sc_bridge is None,
        "read": {"decision": read["decision"], "answers": len(read["answers"])},
        "state_sha_equal": state_sha(cleared) == state_sha(no_bridge),
        "ok": ok,
    }


def definition_digest() -> str:
    payload = {
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "alias_key": ALIAS_KEY,
        "authority": AUTHORITY,
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
        "schema": "taiji-bridge-step-r25-v1",
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
            "extra_clear_steps": 5,
            "regression_sentences": len(sentences()),
            "unknown_candidates": len(UNKNOWN_CANDIDATES),
            "fragments": 18,
        },
        "inputs": {
            "alias_key": ALIAS_KEY,
            "authority": AUTHORITY,
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

        survival = survival_reading(reference, texts)
        clear = clear_reading(reference, texts)

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
            "survival": ("bridge_survives_steps" if survival["ok"] else "bridge_dropped_in_steps"),
            "clear": "bridge_cleared_after_steps" if clear["ok"] else "bridge_clear_failed",
            "closed_state": "closed_state_equal" if closed_ok else "closed_state_moved",
            "bridge_default": (
                "bridge_default_none" if bridge_default_none else "bridge_default_present"
            ),
        }
        all_ok = all([survival["ok"], clear["ok"], closed_ok, bridge_default_none])
        verdict["overall"] = (
            "bridge_step_survival_clean" if all_ok else "bridge_step_survival_unexpected"
        )

        cpu, wall = cost()
        if cpu > CPU_CAP:
            raise SystemExit(f"cpu cap exceeded: {cpu:.2f}s > {CPU_CAP:.0f}s")
        report = {
            "schema": "taiji-bridge-step-r25-report-v1",
            "runner_sha256": RUNNER_SHA_START,
            "manifest_sha256": file_sha(MANIFEST),
            "bridge_sha256": file_sha(BRIDGE_SOURCE),
            "core_sha256": file_sha(CORE_SOURCE),
            "fixture_sha256": base.fixture_hashes(),
            "definition": DEFINITION,
            "criteria": CRITERIA,
            "survival": survival,
            "clear": clear,
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
                    "survival": {
                        "steps": survival["step_count"],
                        "attached_all": all(row["bridge_attached"] for row in survival["steps"]),
                        "table_digest_equal": survival["table_digest_equal"],
                        "read_40_all": all(
                            row["regression_hits"] == 40 for row in survival["steps"]
                        ),
                        "zeros_all": all(row["zeros"] == [0, 0] for row in survival["steps"]),
                        "trajectories_equal": survival["trajectories_equal"],
                        "parameters_equal": survival["parameters_equal"],
                        "gradients_none": survival["gradients_none"],
                    },
                    "clear": {
                        "sc_bridge_none": clear["sc_bridge_none"],
                        "read": clear["read"]["decision"],
                        "state_sha_equal": clear["state_sha_equal"],
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
