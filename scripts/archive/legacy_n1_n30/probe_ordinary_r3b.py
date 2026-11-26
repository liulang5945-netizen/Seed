"""R3b-2 structural probe for the ordinary end-to-end baseline.

Same public fixture and same structural readouts as R3/R3b-1, but the model sees
the whole history as raw text and has no hard parse gate, no discrete store and
no time-conditioned reader. No training, no capability claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taiji.conditional_language_v3 import TaskCoreConfigV3  # noqa: E402
from taiji.corrected_old_baseline import (  # noqa: E402
    BaselinePredictionV1,
    score_fixed_denominator,
)
from taiji.internalization import content_digest  # noqa: E402
from taiji.ordinary_baseline import OrdinaryBundleV1  # noqa: E402

MANIFEST = ROOT / "plans/manifests/native_core_r1_public_fixture_v1.json"
REPORT = ROOT / "reports/taiji_native_core_r3b2_ordinary_20261030.json"
MODULE = ROOT / "taiji/ordinary_baseline.py"
TARGET = b"ab"
_CALLS = {"teacher_forced": 0, "answers": 0, "restore_probe": 0, "fresh_process": 0}


def _config() -> TaskCoreConfigV3:
    return TaskCoreConfigV3(
        width=64,
        embedding_width=32,
        key_width=32,
        slots=8,
        chunk_bytes=8,
        max_observation_bytes=512,
        max_input_evidence_bytes=512,
        max_output_position=128,
        max_generation_bytes=64,
        seed=301,
        position_extension_seed=302,
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _materialise(case: dict) -> OrdinaryBundleV1:
    bundle = OrdinaryBundleV1.create(_config())
    for entry in case["history"]:
        bundle.observe(entry["source_text"])
    if case.get("distractor"):
        bundle.observe(case["distractor"]["source_text"])
    return bundle


def _logits(bundle: OrdinaryBundleV1, case: dict, time_intent: str | None = None):
    intent = case["time_intent"] if time_intent is None else time_intent
    assert case["query_text"] is not None
    _CALLS["teacher_forced"] += 1
    return bundle.score_target(
        query=case["query_text"].encode("utf-8"), target=TARGET, time_intent=intent
    )


def _answer_bundle(bundle: OrdinaryBundleV1, query: bytes, time_intent: str, max_new_bytes: int):
    _CALLS["answers"] += 1
    return bundle.answer(query=query, time_intent=time_intent, max_new_bytes=max_new_bytes)


def _delta(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).abs().max().detach())


def _digest(logits: torch.Tensor) -> str:
    return content_digest([[round(float(value), 6) for value in row] for row in logits.tolist()])


def _restore_probe() -> dict[str, Any]:
    bundle = OrdinaryBundleV1.create(_config())
    bundle.observe("雨伞的颜色是红色。")
    bundle.observe("雨伞的颜色是蓝色。")
    query = "雨伞的颜色是什么？".encode()
    _CALLS["restore_probe"] += 2
    expected = bundle.probe_digest(query=query)
    path = ROOT / ".seed_test_tmp" / "r3b2_ordinary_bundle.pt"
    if path.exists():
        path.unlink()
    bundle.save(path)
    restored = OrdinaryBundleV1.load(path)
    in_process = restored.probe_digest(query=query)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    code = (
        "import sys; "
        "from taiji.ordinary_baseline import OrdinaryBundleV1; "
        "b=OrdinaryBundleV1.load(sys.argv[1]); "
        "print(b.probe_digest(query=sys.argv[2].encode()))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(path), query.decode("utf-8")],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    fresh = result.stdout.strip()
    _CALLS["fresh_process"] += 1
    return {
        "saved_parameter_revision": bundle.model.parameter_revision,
        "saved_history": len(bundle.history),
        "expected_digest": expected,
        "in_process_digest": in_process,
        "fresh_process_digest": fresh,
        "in_process_equal": in_process == expected,
        "fresh_process_equal": fresh == expected,
    }


def _probe() -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    cases = manifest["cases"]
    groups: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        groups[case["group_id"]].append(case)

    checks: dict[str, list[str]] = {"pass": [], "fail": []}

    def check(condition: bool, label: str, detail: str = "") -> None:
        if condition:
            checks["pass"].append(label)
        else:
            checks["fail"].append(label if not detail else f"{label}: {detail}")

    rows: list[dict[str, Any]] = []
    bundles: dict[str, OrdinaryBundleV1] = {}
    logits_by_case: dict[str, torch.Tensor] = {}
    predictions: dict[str, BaselinePredictionV1] = {}

    for case in cases:
        bundle = _materialise(case)
        bundles[case["fixture_id"]] = bundle
        if case["query_text"] is None:
            prediction = _answer_bundle(bundle, b"", "current", 32)
            predictions[case["fixture_id"]] = BaselinePredictionV1(
                case_id=case["fixture_id"], generated=prediction, parse_ok=True
            )
            rows.append(
                {
                    "fixture_id": case["fixture_id"],
                    "group_id": case["group_id"],
                    "query_text": None,
                    "time_intent": None,
                    "prompt_hex_digest": content_digest(bundle.prompt(query=b"")),
                    "history_texts": len(bundle.history),
                }
            )
            continue
        logits = _logits(bundle, case)
        logits_by_case[case["fixture_id"]] = logits
        prediction = _answer_bundle(
            bundle, case["query_text"].encode("utf-8"), case["time_intent"], 32
        )
        predictions[case["fixture_id"]] = BaselinePredictionV1(
            case_id=case["fixture_id"], generated=prediction, parse_ok=True
        )
        rows.append(
            {
                "fixture_id": case["fixture_id"],
                "group_id": case["group_id"],
                "query_text": case["query_text"],
                "time_intent": case["time_intent"],
                "prompt_hex_digest": content_digest(
                    bundle.prompt(
                        query=case["query_text"].encode("utf-8"),
                        time_intent=case["time_intent"],
                    )
                ),
                "history_texts": len(bundle.history),
                "logits_digest": _digest(logits),
            }
        )

    same_query = groups.get("same-query-current", [])
    if len(same_query) == 2:
        left, right = same_query
        check(
            _delta(logits_by_case[left["fixture_id"]], logits_by_case[right["fixture_id"]]) > 0,
            "same query with different current history changes the readout",
        )

    same_history = groups.get("same-current-history", [])
    if len(same_history) == 2:
        left, right = same_history
        delta = _delta(logits_by_case[left["fixture_id"]], logits_by_case[right["fixture_id"]])
        check(
            delta > 0,
            "ordinary sees the whole history: same current value with different history differs",
            f"{delta}",
        )

    same_time = groups.get("same-query-time", [])
    if len(same_time) == 2:
        current_case, previous_case = same_time
        current_logits = _logits(bundles[current_case["fixture_id"]], current_case, "current")
        previous_logits = _logits(bundles[previous_case["fixture_id"]], previous_case, "previous")
        check(
            _delta(current_logits, previous_logits) > 0,
            "same query text with different time marker changes the readout",
        )

    correction = groups.get("correction", [])[0]
    bundle = bundles[correction["fixture_id"]]
    baseline = logits_by_case[correction["fixture_id"]]
    saved = bundle.history
    bundle.history = saved[:-1]
    cut = _logits(bundle, correction)
    cut_delta = _delta(baseline, cut)
    bundle.history = saved
    rescue_delta = _delta(baseline, _logits(bundle, correction))
    check(cut_delta > 0, "removing the current fact from raw history changes the readout")
    check(rescue_delta == 0, "restoring the history restores the exact readout", f"{rescue_delta}")

    left_rows = bundle.score_target(
        query=correction["query_text"].encode("utf-8"), target=TARGET, time_intent="current"
    )
    right_rows = bundle.score_target(
        query=correction["query_text"].encode("utf-8"), target=b"cb", time_intent="current"
    )
    _CALLS["teacher_forced"] += 2
    row0_delta = _delta(left_rows[0], right_rows[0])
    row1_delta = _delta(left_rows[1], right_rows[1])
    check(row0_delta == 0, "step-0 logits do not depend on target[0]", f"{row0_delta}")
    check(row1_delta > 0, "step-1 logits depend on the observed first target byte", f"{row1_delta}")

    score = score_fixed_denominator(cases, predictions)
    check(score["scored_turns"] == len(cases), "fixed denominator counts every declared turn")
    check(
        score["parse_failures"] == 0,
        "ordinary has no parse gate; parse failures cannot silently leave the denominator",
    )

    restore = _restore_probe()
    check(restore["in_process_equal"], "zero-step ordinary bundle restores in-process")
    check(restore["fresh_process_equal"], "zero-step ordinary bundle restores in a fresh process")

    return {
        "schema": "taiji-ordinary-r3b2-probe-v1",
        "status": "r3b2_structure_passed" if not checks["fail"] else "r3b2_structure_failed",
        "fixture": {
            "path": str(MANIFEST.relative_to(ROOT)).replace("\\", "/"),
            "sha256": _sha256_file(MANIFEST),
            "cases": len(cases),
        },
        "config": _config().__dict__,
        "parameters": OrdinaryBundleV1.create(_config()).model.parameter_count(),
        "trainable_parameters": len(
            OrdinaryBundleV1.create(_config()).model.trainable_parameters()
        ),
        "cases": rows,
        "checks": checks,
        "readouts": {
            "cut_delta": cut_delta,
            "rescue_delta": rescue_delta,
            "step0_target_byte_delta": row0_delta,
            "step1_prefix_delta": row1_delta,
            "same_current_history_delta": (
                _delta(
                    logits_by_case[same_history[0]["fixture_id"]],
                    logits_by_case[same_history[1]["fixture_id"]],
                )
                if len(same_history) == 2
                else None
            ),
        },
        "fixed_denominator": score,
        "restore": restore,
        "model_calls": {
            "teacher_forced": _CALLS["teacher_forced"],
            "generated_answers": _CALLS["answers"],
            "restore_probe": _CALLS["restore_probe"],
            "fresh_process": _CALLS["fresh_process"],
        },
        "optimizer_updates": 0,
        "training_started": False,
        "sealed_text_read": 0,
        "can_promote": False,
        "source_sha256": {
            "taiji/ordinary_baseline.py": _sha256_file(MODULE),
            "tests/taiji_native/test_ordinary_baseline.py": _sha256_file(
                ROOT / "tests/taiji_native/test_ordinary_baseline.py"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    started_cpu = time.process_time()
    started_wall = time.perf_counter()
    payload = _probe()
    payload["cpu_seconds"] = time.process_time() - started_cpu
    payload["wall_seconds"] = time.perf_counter() - started_wall
    if args.report.exists():
        raise SystemExit(f"report already exists (write-once): {args.report}")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps({"status": payload["status"], "checks": payload["checks"]}, ensure_ascii=False)
    )
    return 0 if payload["status"] == "r3b2_structure_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
