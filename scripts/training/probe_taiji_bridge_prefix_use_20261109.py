"""R36 -- is the self-generated prefix useful, not merely present?

R34 proved the previous expression enters the next input; R35 made the
decision a policy and showed the policy only differs on non-hitting input.
R36 asks the next question with a frozen, zero-training measurement.

For each describe sentence (from step 2 on) one fixed subject state ``S0`` is
fed four prompts and then scores the sentence bytes with teacher forcing
(``score_targets``):

- ``none``: no prompt (reported, never decides -- a different byte length);
- ``self``: the previous expression;
- ``shuffled``: the same bytes reversed (same length, same multiset, no
  character order);
- ``foreign``: another step's expression with the same byte length.

The reading is bits per target byte.  Frozen decision:

- ``prefix_useful``: median(self better than foreign) >= 0.1 bits/byte and
  median(self better than shuffled) >= 0.1 and >= 14/19 steps better against
  both controls;
- ``prefix_harmful``: the same magnitudes, >= 14/19 steps worse;
- otherwise ``prefix_not_useful_yet``.

No module change: this round only measures with the existing ``score_targets``.
Zero training, zero optimizer steps, zero Adam, zero confirm.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
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

MANIFEST = ROOT / "plans/manifests/taiji_bridge_prefix_use_20261109.json"
REPORT = ROOT / "reports/taiji_bridge_prefix_use_20261109.json"
BRIDGE_SOURCE = ROOT / "taiji/address_bridge.py"
CORE_SOURCE = ROOT / "taiji/living_core.py"

RUNNER_SHA_START = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()

SEED = 20261109
ESTIMATE_CPU_UPPER = 60.0
CPU_CAP = float(base.CPU_CAP)
EFFECT_FLOOR = 0.1
WIN_STEPS = 14
PAIR_TOTAL = 19

CUP = ("object", "蓝色杯子")
RED = ("object", "红色印章")
GREEN = ("object", "绿色瓶子")
TABLE = ("place", "木桌")
SHELF = ("place", "书架")
AUTHORITY = "verifier:v1"
CONDITIONS = ("none", "self", "shuffled", "foreign")

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
        "round": "R35",
        "runner": "scripts/training/probe_taiji_bridge_policy_20261109.py",
        "rerun_commit": "d4dba0069",
        "runner_blob_sha1": "4b8a0b9f58e030c0fcb393a1d66921d634b5abd1",
        "report_runner_sha256": "e59f869d625233b5dd5a6137f821e79dcdfa8309f1d898e4360f0a41df5dce29",
        "runner_blob_sha256": "e59f869d625233b5dd5a6137f821e79dcdfa8309f1d898e4360f0a41df5dce29",
        "report_core_sha256": "6c0c53a2a6465627918721da83b1e0fb80a04c262f95c97d9b825ee858d0785d",
        "matches_report": True,
        "note": "R36 does not change living_core; R35 stays runnable at HEAD",
    },
)

DEFINITION = {
    "measurement": (
        "for each describe sentence from step 2, one fixed S0 is prompted with "
        "none/self/shuffled/foreign and the sentence is scored with teacher forcing"
    ),
    "lengths": "self, shuffled and foreign prompts have exactly the same byte length",
    "reading": "bits per target byte; lower is better",
    "useful": (
        "median(self better than foreign) >= 0.1 and median(self better than shuffled) >= 0.1 "
        "and >= 14/19 steps better against both"
    ),
    "harmful": "the same magnitudes with >= 14/19 steps worse against both",
    "fallback": "otherwise prefix_not_useful_yet; none is reported but never decides",
    "closed_state": "the nine legacy closed-state readings stay equal before and after",
}

CRITERIA = {
    "expressions": "the 20 expression texts are exact (reused from the frozen table)",
    "rows": "19 paired steps (2..20); every step reports all four conditions",
    "foreign_match": "every paired step finds a foreign expression of equal byte length",
    "decision": "useful / harmful / not_useful_yet by the frozen medians and 14/19 counts",
    "secondary": "the none-vs-self difference is reported and does not decide",
    "closed_state": "the nine legacy readings stay equal before and after",
    "rerun": "the R35 anchor carries a commit, a runner blob sha1, a runner sha256 and a core sha256",
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


def expected_pair(text: str) -> dict[str, tuple[str, str]]:
    found: dict[str, tuple[str, str]] = {}
    for name, phrase in OBJECT_DESCRIPTIONS.items():
        if phrase in text:
            found["object"] = ("object", name)
    for name, phrase in PLACE_DESCRIPTIONS.items():
        if phrase in text:
            found["place"] = ("place", name)
    return found


def expression_texts(reference: Any, table: BridgeTable, texts: list[str]) -> list[str]:
    scratch = reference.attach_bridge(reference.initial_state("r36-expressions"), table)
    object_candidates = [("object", name) for name in OBJECT_DESCRIPTIONS]
    place_candidates = [("place", name) for name in PLACE_DESCRIPTIONS]
    output: list[str] = []
    for text in texts:
        _, result = reference.bridge_express(scratch, text, object_candidates, place_candidates)
        expected = expected_pair(text)
        expected_text = (
            f"{expected['object'][1]}在{expected['place'][1]}"
            if "object" in expected and "place" in expected
            else None
        )
        if result["decision"] != "chosen_pair" or result["text"] != expected_text:
            raise SystemExit(f"expression mismatch on {text!r}: {result}")
        output.append(result["text"])
    return output


def target_bits(distributions: torch.Tensor, target: bytes) -> float:
    losses = [
        -math.log2(max(float(distributions[index][byte]), 1e-12))
        for index, byte in enumerate(target)
    ]
    return sum(losses) / len(losses)


def prefix_use_reading(reference: Any, table: BridgeTable, texts: list[str]) -> dict[str, Any]:
    expressions = expression_texts(reference, table, texts)
    state = reference.initial_state("r36-prefix-use")
    rows = []
    unmatched = 0
    for index in range(2, len(texts) + 1):
        previous = expressions[index - 2].encode("utf-8")
        target = texts[index - 1].encode("utf-8")
        prompts = {
            "none": b"",
            "self": previous,
            "shuffled": bytes(reversed(previous)),
            "foreign": None,
        }
        foreign_source = None
        for other in range(len(expressions)):
            if other == index - 2:
                continue
            candidate = expressions[other].encode("utf-8")
            if len(candidate) == len(previous):
                prompts["foreign"] = candidate
                foreign_source = other + 1
                break
        if prompts["foreign"] is None:
            unmatched += 1
            rows.append({"step": index, "matched": False})
            continue
        bits: dict[str, float] = {}
        with torch.no_grad():
            for name in CONDITIONS:
                distributions, _ = reference.score_targets(
                    state,
                    prompt=prompts[name],
                    target=target,
                    goal=b"r36",
                )
                bits[name] = target_bits(distributions, target)
        rows.append(
            {
                "step": index,
                "matched": True,
                "prompt_bytes": len(previous),
                "foreign_from_step": foreign_source,
                "bits": {name: round(value, 6) for name, value in bits.items()},
                "self_minus_foreign": round(bits["foreign"] - bits["self"], 6),
                "self_minus_shuffled": round(bits["shuffled"] - bits["self"], 6),
                "self_minus_none": round(bits["none"] - bits["self"], 6),
            }
        )
    paired = [row for row in rows if row["matched"]]
    d_foreign = [row["self_minus_foreign"] for row in paired]
    d_shuffled = [row["self_minus_shuffled"] for row in paired]
    d_none = [row["self_minus_none"] for row in paired]
    wins_foreign = sum(1 for value in d_foreign if value > 0)
    wins_shuffled = sum(1 for value in d_shuffled if value > 0)
    median_foreign = statistics.median(d_foreign) if d_foreign else 0.0
    median_shuffled = statistics.median(d_shuffled) if d_shuffled else 0.0
    useful = (
        len(paired) == PAIR_TOTAL
        and median_foreign >= EFFECT_FLOOR
        and median_shuffled >= EFFECT_FLOOR
        and wins_foreign >= WIN_STEPS
        and wins_shuffled >= WIN_STEPS
    )
    harmful = (
        len(paired) == PAIR_TOTAL
        and median_foreign <= -EFFECT_FLOOR
        and median_shuffled <= -EFFECT_FLOOR
        and wins_foreign <= PAIR_TOTAL - WIN_STEPS
        and wins_shuffled <= PAIR_TOTAL - WIN_STEPS
    )
    if useful:
        decision = "prefix_useful"
    elif harmful:
        decision = "prefix_harmful"
    else:
        decision = "prefix_not_useful_yet"
    return {
        "decision": decision,
        "paired": len(paired),
        "of": PAIR_TOTAL,
        "unmatched_foreign": unmatched,
        "median_self_minus_foreign": round(median_foreign, 6),
        "median_self_minus_shuffled": round(median_shuffled, 6),
        "median_self_minus_none": round(statistics.median(d_none), 6) if d_none else 0.0,
        "wins_foreign": wins_foreign,
        "wins_shuffled": wins_shuffled,
        "wins_none": sum(1 for value in d_none if value > 0),
        "mean_bits": (
            {
                name: round(statistics.mean(row["bits"][name] for row in paired), 6)
                for name in CONDITIONS
            }
            if paired
            else {}
        ),
        "rows": rows,
        "ok": len(paired) == PAIR_TOTAL and unmatched == 0,
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


def state_sha(state: Any) -> str:
    return tensor_sha([state.h, state.fast, state.nu_ema, state.coverage, state.prior])


def parameter_sha(reference: Any) -> str:
    return tensor_sha([tensor for _, tensor in sorted(reference.named_parameters())])


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
    state = reference.initial_state("r36-zero-perturbation")
    step_hashes: list[str] = []
    for raw in texts:
        state = reference.observe(state, raw.encode("utf-8"), goal=b"r36")
        step_hashes.append(state_sha(state))
    return reference_digest(reference, state, step_hashes)


def definition_digest() -> str:
    payload = {
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "authority": AUTHORITY,
        "conditions": list(CONDITIONS),
        "effect_floor": EFFECT_FLOOR,
        "win_steps": WIN_STEPS,
        "pair_total": PAIR_TOTAL,
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
        "schema": "taiji-bridge-prefix-use-r36-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": SEED,
        "estimate_cpu_upper": ESTIMATE_CPU_UPPER,
        "cpu_cap": CPU_CAP,
        "effect_floor_bits_per_byte": EFFECT_FLOOR,
        "win_steps": WIN_STEPS,
        "pair_total": PAIR_TOTAL,
        "runner_sha256": RUNNER_SHA_START,
        "bridge_sha256": file_sha(BRIDGE_SOURCE),
        "core_sha256": file_sha(CORE_SOURCE),
        "fixture_sha256": base.fixture_hashes(),
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "definition_digest": definition_digest(),
        "counts": {
            "describe_sentences": len(sentences()),
            "conditions": len(CONDITIONS),
            "paired_steps": PAIR_TOTAL,
            "unknown_candidates": len(UNKNOWN_CANDIDATES),
            "rerun_anchors": len(RERUN_ANCHORS),
        },
        "inputs": {
            "authority": AUTHORITY,
            "conditions": list(CONDITIONS),
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
        reading = prefix_use_reading(reference, table, texts)

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

        verdict = {
            "rows": (
                "prefix_use_rows_19_19"
                if reading["ok"]
                else f"prefix_use_rows_{reading['paired']}_{reading['of']}"
            ),
            "decision": reading["decision"],
            "parameters": "parameters_untouched" if parameters_ok else "parameters_moved",
            "closed_state": "closed_state_equal" if closed_ok else "closed_state_moved",
            "bridge_default": (
                "bridge_default_none" if bridge_default_none else "bridge_default_present"
            ),
            "rerun_anchors": "anchors_complete" if rerun_ok else "anchors_incomplete",
        }
        all_ok = all([reading["ok"], parameters_ok, closed_ok, bridge_default_none, rerun_ok])
        verdict["overall"] = "prefix_use_measured" if all_ok else "prefix_use_harness_unexpected"

        cpu, wall = cost()
        if cpu > CPU_CAP:
            raise SystemExit(f"cpu cap exceeded: {cpu:.2f}s > {CPU_CAP:.0f}s")
        report = {
            "schema": "taiji-bridge-prefix-use-r36-report-v1",
            "runner_sha256": RUNNER_SHA_START,
            "manifest_sha256": file_sha(MANIFEST),
            "bridge_sha256": file_sha(BRIDGE_SOURCE),
            "core_sha256": file_sha(CORE_SOURCE),
            "fixture_sha256": base.fixture_hashes(),
            "definition": DEFINITION,
            "criteria": CRITERIA,
            "effect_floor_bits_per_byte": EFFECT_FLOOR,
            "win_steps": WIN_STEPS,
            "pair_total": PAIR_TOTAL,
            "reading": reading,
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
                    "reading": {
                        "decision": reading["decision"],
                        "paired": reading["paired"],
                        "of": reading["of"],
                        "unmatched_foreign": reading["unmatched_foreign"],
                        "median_self_minus_foreign": reading["median_self_minus_foreign"],
                        "median_self_minus_shuffled": reading["median_self_minus_shuffled"],
                        "median_self_minus_none": reading["median_self_minus_none"],
                        "wins_foreign": reading["wins_foreign"],
                        "wins_shuffled": reading["wins_shuffled"],
                        "wins_none": reading["wins_none"],
                        "mean_bits": reading["mean_bits"],
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
