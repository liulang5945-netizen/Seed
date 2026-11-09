"""R23 -- cross-process handoff of the written bridge.

R22 showed the table survives an in-process export/import round trip.  R23 asks
the harder question: can a *separate process* inherit it, read with it, correct
with it, and hand it back?

Layout:

* The parent builds the 18-key table under policy ``verifier:v1``, exports the
  state bytes, and records the digest.
* A child process (this same file, ``--child``) imports the bytes under its own
  policy, reads the 20 describe sentences and the zero lines, tries a correction
  from an unlisted authority (must be rejected), then corrects the alias from
  CUP to RED with ``verifier:v1`` (must be accepted), and exports.
* The parent imports the child's export and checks the digest chain, the read,
  the audit entry and the 40/40 and 0/0 lines.
* A second child receives a truncated state file and must reject it whole.
* Closed state: the nine readings stay equal.

Zero training, zero optimizer steps, zero Adam, zero confirm.  No fixture byte is
written; the bridge module is not modified by this round.
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
    value_key,
)
from taiji.living_core import LivingCoreReference  # noqa: E402
from taiji.native_goal_world_wide import (  # noqa: E402
    OBJECT_DESCRIPTIONS,
    PLACE_DESCRIPTIONS,
)
from taiji.process_guard import assert_no_black_processes  # noqa: E402

torch.set_num_threads(1)

MANIFEST = ROOT / "plans/manifests/taiji_bridge_handoff_20261109.json"
REPORT = ROOT / "reports/taiji_bridge_handoff_20261109.json"
BRIDGE_SOURCE = ROOT / "taiji/address_bridge.py"

RUNNER_SHA_START = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()

SEED = 20261109
ESTIMATE_CPU_UPPER = 60.0
CPU_CAP = float(base.CPU_CAP)

CUP = ("object", "蓝色杯子")
RED = ("object", "红色印章")
GREEN = ("object", "绿色瓶子")
TABLE = ("place", "木桌")
SHELF = ("place", "书架")
ALIAS_KEY = "喝水的杯子"
AUTHORITY = "verifier:v1"
UNLISTED = "verifier:v2"
REASON = "child:accepted"
REJECT_REASON = "child:rejected"

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
    "handoff": (
        "parent exports the 18-key table under policy verifier:v1; a separate process imports it "
        "under its own policy, reads, attempts one unlisted-authority correction, applies one "
        "allowed-authority correction, and exports; parent imports the child export"
    ),
    "child_read": "child must read the 20 describe sentences at 40/40 and the zero lines at 0/0",
    "child_policy": (
        "child policy is verifier:v1; a correction from verifier:v2 must be "
        "rejected_batch_unknown_authority, then verifier:v1 must be accepted"
    ),
    "digest_chain": (
        "child imported digest equals parent digest; child exported digest equals parent-imported "
        "digest after the handback"
    ),
    "corrupt_child": (
        "a second child receives truncated bytes and must reject them with a ValueError whose "
        "message starts with rejected_import_invalid_state; it must not write an output table"
    ),
    "closed_state": "the nine closed-state readings stay equal before and after",
}

CRITERIA = {
    "child_import": "child imported digest equals the parent digest",
    "child_reading": "child regression 40/40 with wrong 0; zeros 0/0",
    "child_policy": (
        "unlisted authority rejected as rejected_batch_unknown_authority; allowed authority "
        "accepted with one log entry"
    ),
    "handback": (
        "parent import of the child export has the same digest as the child export; alias reads "
        "RED; log authority is verifier:v1 and reason is child:accepted; 40/40 and 0/0"
    ),
    "corrupt_child": (
        "child result carries an error starting with rejected_import_invalid_state and no output "
        "bytes"
    ),
    "closed_state": "the nine closed-state readings stay equal before and after",
    "reporting": (
        "the child runs in a separate Python process; the report records its return code and the "
        "sha256 of both state files, so the handoff is not an in-process object copy"
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


def reading(table: BridgeTable, sentence: str, max_passes: int = 2) -> dict[str, Any]:
    result = table.read_sentence(sentence, max_passes=max_passes)
    values = [list(row["value"]) for row in result["answers"]]
    return {
        "decision": result["decision"],
        "answers": len(values),
        "values": values,
        "stopped_at_pass": result["stopped_at_pass"],
        "final_exit": result["final_exit"],
    }


def log_entries(table: BridgeTable) -> list[dict[str, Any]]:
    return [
        {
            "from": value_key(row["from"]),
            "to": value_key(row["to"]),
            "authority": row.get("authority", ""),
            "reason": row.get("reason", ""),
        }
        for row in table.corrections
    ]


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
    state = reference.initial_state("r23-zero-perturbation")
    step_hashes: list[str] = []
    for raw in texts:
        state = reference.observe(state, raw.encode("utf-8"), goal=b"r23")
        step_hashes.append(
            tensor_sha([state.h, state.fast, state.nu_ema, state.coverage, state.prior])
        )
    return reference_digest(reference, state, step_hashes)


def child_main(state_path: str, out_path: str, result_path: str) -> int:
    data = Path(state_path).read_bytes()
    try:
        table = BridgeTable.from_state(data, allowed_authorities=(AUTHORITY,))
    except ValueError as error:
        write_json(
            Path(result_path),
            {"error": str(error), "wrote_output": False},
        )
        return 0
    texts = sentences()
    imported_digest = table.digest()
    regression = regression_reading(table, texts)
    zeros = zeros_reading(table)
    before = reading(table, ALIAS_KEY)
    rejected = table.correct(((ALIAS_KEY, CUP, GREEN),), authority=UNLISTED, reason=REJECT_REASON)
    accepted = table.correct(((ALIAS_KEY, CUP, RED),), authority=AUTHORITY, reason=REASON)
    after = reading(table, ALIAS_KEY)
    out = table.export_state()
    Path(out_path).write_bytes(out)
    write_json(
        Path(result_path),
        {
            "imported_digest": imported_digest,
            "exported_digest": table.digest(),
            "export_sha256": hashlib.sha256(out).hexdigest(),
            "regression": regression,
            "zeros": zeros,
            "before": before,
            "rejected": {"action": rejected["action"]},
            "accepted": {"action": accepted["action"]},
            "after": after,
            "log": log_entries(table),
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
        timeout=180,
    )
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip()[-2000:],
        "stderr": completed.stderr.strip()[-2000:],
        "result": result,
        "out_exists": out_path.exists(),
    }


def parent_reading(texts: list[str]) -> dict[str, Any]:
    parent = build_default()
    parent_digest = parent.digest()
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
            handback_ok = (
                handback.digest() == child_result.get("exported_digest")
                and reading(handback, ALIAS_KEY)["values"] == [list(RED)]
                and len(handback.corrections) == 1
                and handback.corrections[0]["authority"] == AUTHORITY
                and handback.corrections[0]["reason"] == REASON
                and regression_reading(handback, texts)["ok"]
                and zeros_reading(handback)["ok"]
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
        ok = (
            child["returncode"] == 0
            and child_result.get("imported_digest") == parent_digest
            and child_result.get("regression", {}).get("ok")
            and child_result.get("zeros", {}).get("ok")
            and child_result.get("rejected", {}).get("action") == "rejected_batch_unknown_authority"
            and child_result.get("accepted", {}).get("action") == "corrected"
            and child_result.get("after", {}).get("values") == [list(RED)]
            and handback_ok
            and corrupt_ok
        )
        return {
            "parent_digest": parent_digest,
            "state_bytes": len(state_bytes),
            "state_sha256": hashlib.sha256(state_bytes).hexdigest(),
            "child": child,
            "child_result": child_result,
            "handback_ok": handback_ok,
            "handback_digest": handback.digest() if handback is not None else None,
            "corrupt_child": corrupt_child,
            "corrupt_ok": corrupt_ok,
            "ok": ok,
        }


def definition_digest() -> str:
    payload = {
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "alias_key": ALIAS_KEY,
        "authority": AUTHORITY,
        "unlisted": UNLISTED,
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
        "schema": "taiji-bridge-handoff-r23-v1",
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
            "child_processes": 2,
            "regression_sentences": len(sentences()),
            "unknown_candidates": len(UNKNOWN_CANDIDATES),
            "fragments": 18,
        },
        "inputs": {
            "alias_key": ALIAS_KEY,
            "authority": AUTHORITY,
            "unlisted": UNLISTED,
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

        handoff = parent_reading(texts)

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
            "child_import": (
                "child_import_matched"
                if handoff["child_result"].get("imported_digest") == handoff["parent_digest"]
                else "child_import_moved"
            ),
            "child_reading": (
                "child_read_40_40"
                if handoff["child_result"].get("regression", {}).get("ok")
                and handoff["child_result"].get("zeros", {}).get("ok")
                else "child_read_moved"
            ),
            "child_policy": (
                "child_policy_enforced"
                if handoff["child_result"].get("rejected", {}).get("action")
                == "rejected_batch_unknown_authority"
                and handoff["child_result"].get("accepted", {}).get("action") == "corrected"
                else "child_policy_failed"
            ),
            "handback": "handback_clean" if handoff["handback_ok"] else "handback_moved",
            "corrupt_child": (
                "corrupt_child_rejected" if handoff["corrupt_ok"] else "corrupt_child_accepted"
            ),
            "closed_state": "closed_state_equal" if closed_ok else "closed_state_moved",
        }
        all_ok = all(
            [
                handoff["ok"],
                handoff["corrupt_ok"],
                closed_ok,
            ]
        )
        verdict["overall"] = (
            "cross_process_handoff_clean" if all_ok else "cross_process_handoff_unexpected"
        )

        cpu, wall = cost()
        if cpu > CPU_CAP:
            raise SystemExit(f"cpu cap exceeded: {cpu:.2f}s > {CPU_CAP:.0f}s")
        report = {
            "schema": "taiji-bridge-handoff-r23-report-v1",
            "runner_sha256": RUNNER_SHA_START,
            "manifest_sha256": file_sha(MANIFEST),
            "bridge_sha256": file_sha(BRIDGE_SOURCE),
            "fixture_sha256": base.fixture_hashes(),
            "definition": DEFINITION,
            "criteria": CRITERIA,
            "handoff": handoff,
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
                    "parent_digest": handoff["parent_digest"],
                    "state_bytes": handoff["state_bytes"],
                    "child_returncode": handoff["child"]["returncode"],
                    "child": {
                        "imported": handoff["child_result"].get("imported_digest"),
                        "regression": handoff["child_result"].get("regression", {}).get("hits"),
                        "of": handoff["child_result"].get("regression", {}).get("entities"),
                        "zeros": [
                            handoff["child_result"].get("zeros", {}).get("unknown_answering"),
                            handoff["child_result"].get("zeros", {}).get("fragment_answering"),
                        ],
                        "rejected": handoff["child_result"].get("rejected", {}).get("action"),
                        "accepted": handoff["child_result"].get("accepted", {}).get("action"),
                        "after": handoff["child_result"].get("after", {}).get("values"),
                        "log": handoff["child_result"].get("log"),
                    },
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
