"""R3b structural probe for the corrected old baseline.

Runs the public R1 fixture through the corrected old graph (hard parse, discrete
store, current/previous selection, frozen V3 head, one fixed value bias) and
reports the same structural readouts as the R3 core: candidate counts,
same-query/different-current separation, same-current history separation,
time-intent separation, cut/rescue, first-byte supervision, fixed-denominator
scoring and zero-step restore. No training, no evaluation of capability.
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
    CorrectedOldBundleV1,
    score_fixed_denominator,
)
from taiji.internalization import content_digest  # noqa: E402

MANIFEST = ROOT / "plans/manifests/native_core_r1_public_fixture_v1.json"
REPORT = ROOT / "reports/taiji_native_core_r3b_baseline_20261030.json"
MODULE = ROOT / "taiji/corrected_old_baseline.py"
TARGET = b"ab"
OWNER, SCOPE, SESSION = "public", "r1-fixture", "fixture"


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
        seed=201,
        position_extension_seed=202,
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _materialise(case: dict) -> CorrectedOldBundleV1:
    bundle = CorrectedOldBundleV1.create(_config())
    for turn, entry in enumerate(case["history"]):
        bundle.ingest(
            source_text=entry["source_text"],
            entity=case["entity"],
            attribute=case["attribute"],
            value=entry["value"],
            source_input_id=entry["source_input_id"],
            owner=OWNER,
            scope=SCOPE,
            session=SESSION,
            turn_index=turn,
            revision=entry["revision"],
            valid_from=entry["valid_from"],
        )
    if case.get("distractor"):
        distractor = case["distractor"]
        bundle.ingest(
            source_text=distractor["source_text"],
            entity=distractor["entity"],
            attribute=distractor["attribute"],
            value=distractor["value"],
            source_input_id=f"r1-public:distractor:{case['fixture_id']}",
            owner=OWNER,
            scope=SCOPE,
            session=SESSION,
            turn_index=len(case["history"]) + 1,
            revision=0,
            valid_from=len(case["history"]) + 1,
        )
    return bundle


def _logits(bundle: CorrectedOldBundleV1, case: dict, time_intent: str | None = None):
    intent = case["time_intent"] if time_intent is None else time_intent
    assert case["query_text"] is not None
    logits, selection = bundle.score_target(
        query=case["query_text"].encode("utf-8"),
        target=TARGET,
        time_intent=intent,
        owner=OWNER,
        scope=SCOPE,
        session=SESSION,
    )
    return logits, selection


def _delta(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).abs().max().detach())


def _digest(logits: torch.Tensor) -> str:
    return content_digest([[round(float(value), 6) for value in row] for row in logits.tolist()])


def _restore_probe() -> dict[str, Any]:
    bundle = CorrectedOldBundleV1.create(_config())
    for turn, value in enumerate(("红色", "蓝色")):
        bundle.ingest(
            source_text=f"雨伞的颜色是{value}。",
            entity="雨伞",
            attribute="颜色",
            value=value,
            source_input_id=f"r3b-probe:restore:{turn}",
            owner="probe",
            scope="probe",
            session="probe",
            turn_index=turn,
            revision=turn,
        )
    query = "雨伞的颜色是什么？".encode()
    expected = bundle.probe_digest(query=query)
    path = ROOT / ".seed_test_tmp" / "r3b_corrected_old_bundle.pt"
    if path.exists():
        path.unlink()
    bundle.save(path)
    restored = CorrectedOldBundleV1.load(path)
    in_process = restored.probe_digest(query=query)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    code = (
        "import sys; "
        "from taiji.corrected_old_baseline import CorrectedOldBundleV1; "
        "b=CorrectedOldBundleV1.load(sys.argv[1]); "
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
    return {
        "saved_parameter_revision": bundle.model.parameter_revision,
        "saved_facts": len(bundle.store.facts),
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
    bundles: dict[str, CorrectedOldBundleV1] = {}
    logits_by_case: dict[str, torch.Tensor] = {}
    predictions: dict[str, BaselinePredictionV1] = {}

    for case in cases:
        bundle = _materialise(case)
        bundles[case["fixture_id"]] = bundle
        current_facts = [
            item
            for item in bundle.store.facts
            if item.valid_to is None and item.owner == OWNER and item.scope == SCOPE
        ]
        if case["query_text"] is None:
            rows.append(
                {
                    "fixture_id": case["fixture_id"],
                    "group_id": case["group_id"],
                    "query_text": None,
                    "time_intent": None,
                    "parse_ok": True,
                    "reason": "non_query",
                    "selected_value": None,
                    "current_facts": len(current_facts),
                }
            )
            predictions[case["fixture_id"]] = BaselinePredictionV1(
                case_id=case["fixture_id"], generated=b"", parse_ok=True
            )
            continue
        logits, selection = _logits(bundle, case)
        logits_by_case[case["fixture_id"]] = logits
        prediction = bundle.answer(
            query=case["query_text"].encode("utf-8"),
            time_intent=case["time_intent"],
            owner=OWNER,
            scope=SCOPE,
            session=SESSION,
            max_new_bytes=32,
        )
        predictions[case["fixture_id"]] = BaselinePredictionV1(
            case_id=case["fixture_id"],
            generated=prediction.generated,
            parse_ok=selection.parse.parsed,
        )
        rows.append(
            {
                "fixture_id": case["fixture_id"],
                "group_id": case["group_id"],
                "query_text": case["query_text"],
                "time_intent": case["time_intent"],
                "parse_ok": selection.parse.parsed,
                "reason": selection.reason,
                "selected_value": selection.value,
                "current_facts": len(current_facts),
                "logits_digest": _digest(logits),
            }
        )

    parsed_queries = [row for row in rows if row["query_text"] is not None]
    check(
        all(row["parse_ok"] for row in parsed_queries),
        "hard parser covers every public query text",
        str([row["fixture_id"] for row in parsed_queries if not row["parse_ok"]]),
    )

    same_query = groups.get("same-query-current", [])
    if len(same_query) == 2:
        left, right = same_query
        check(
            _delta(logits_by_case[left["fixture_id"]], logits_by_case[right["fixture_id"]]) > 0,
            "same query with different current values changes the readout",
        )

    same_history = groups.get("same-current-history", [])
    if len(same_history) == 2:
        left, right = same_history
        check(
            _delta(logits_by_case[left["fixture_id"]], logits_by_case[right["fixture_id"]]) == 0,
            "same current value keeps current readout identical",
        )
        previous_left = next(
            item
            for item in bundles[left["fixture_id"]].store.facts
            if item.entity == left["entity"]
            and item.attribute == left["attribute"]
            and item.revision == 0
        )
        previous_right = next(
            item
            for item in bundles[right["fixture_id"]].store.facts
            if item.entity == right["entity"]
            and item.attribute == right["attribute"]
            and item.revision == 0
        )
        check(
            previous_left.value != previous_right.value,
            "same-current pair has different previous values",
        )
        left_previous, _ = _logits(bundles[left["fixture_id"]], left, "previous")
        right_previous, _ = _logits(bundles[right["fixture_id"]], right, "previous")
        check(
            _delta(left_previous, right_previous) > 0,
            "same current value with different history changes the previous readout",
        )

    same_time = groups.get("same-query-time", [])
    if len(same_time) == 2:
        current_case, previous_case = same_time
        current_logits, _ = _logits(bundles[current_case["fixture_id"]], current_case, "current")
        previous_logits, _ = _logits(
            bundles[previous_case["fixture_id"]], previous_case, "previous"
        )
        check(
            _delta(current_logits, previous_logits) > 0,
            "same query text with different time intent changes the readout",
        )

    three = groups.get("three-versions", [])
    if three:
        case = three[0]
        _, selection = _logits(bundles[case["fixture_id"]], case)
        check(selection.value == "蓝色", "three versions: previous is revision 1")

    same_value = groups.get("same-value-version", [])
    if same_value:
        case = same_value[0]
        _, selection = _logits(bundles[case["fixture_id"]], case)
        current = bundles[case["fixture_id"]].store.current(case["entity"], case["attribute"])
        check(
            selection.value is not None
            and current is not None
            and selection.value == current.value
            and selection.fact is not None
            and selection.fact.fact_id != current.fact_id,
            "same-value correction keeps a distinct previous revision",
        )

    unknown_previous = groups.get("unknown-previous", [])
    if unknown_previous:
        case = unknown_previous[0]
        _, selection = _logits(bundles[case["fixture_id"]], case)
        check(
            selection.fact is None and selection.reason == "no_fact",
            "previous without a predecessor has no fact and does not fall back",
        )

    for name in ("unknown-current", "ambiguous", "negation"):
        group = groups.get(name, [])
        if group:
            case = group[0]
            _, selection = _logits(bundles[case["fixture_id"]], case)
            check(
                selection.fact is None and selection.reason == "no_fact",
                f"{name}: empty candidate and no fabricated value",
            )

    owner_group = groups.get("owner", [])
    if owner_group:
        case = owner_group[0]
        selection = bundles[case["fixture_id"]].condition(
            query=case["query_text"].encode("utf-8"),
            time_intent=case["time_intent"],
            owner="other",
            scope=SCOPE,
            session=SESSION,
        )
        check(
            selection.fact is None and selection.reason == "not_visible",
            "owner boundary: another owner sees no fact",
        )

    correction = groups.get("correction", [])[0]
    bundle = bundles[correction["fixture_id"]]
    baseline, selection = _logits(bundle, correction)
    assert selection.fact is not None
    saved = bundle.store
    bundle.store = saved.cut(selection.fact.fact_id)
    cut, cut_selection = _logits(bundle, correction)
    check(
        cut_selection.fact is None or cut_selection.fact.fact_id != selection.fact.fact_id,
        "cut removed the selected fact",
    )
    cut_delta = _delta(baseline, cut)
    bundle.store = saved
    rescue_delta = _delta(baseline, _logits(bundle, correction)[0])
    check(cut_delta > 0, "cut changes the readout")
    check(rescue_delta == 0, "legal rescue restores the exact baseline readout", f"{rescue_delta}")

    left_rows = bundle.model.teacher_forced(
        query=correction["query_text"].encode("utf-8"),
        target=TARGET,
        value=selection.value,
        owner=OWNER,
        scope=SCOPE,
        session=SESSION,
    )
    right_rows = bundle.model.teacher_forced(
        query=correction["query_text"].encode("utf-8"),
        target=b"cb",
        value=selection.value,
        owner=OWNER,
        scope=SCOPE,
        session=SESSION,
    )
    row0_delta = _delta(left_rows[0], right_rows[0])
    row1_delta = _delta(left_rows[1], right_rows[1])
    check(row0_delta == 0, "step-0 logits do not depend on target[0]", f"{row0_delta}")
    check(row1_delta > 0, "step-1 logits depend on the observed first target byte", f"{row1_delta}")

    score = score_fixed_denominator(cases, predictions)
    check(score["scored_turns"] == len(cases), "fixed denominator counts every declared turn")
    check(
        score["parse_failures"] == 0,
        "every public query parsed; parse failures would still be counted",
        str(score["parse_failures"]),
    )

    restore = _restore_probe()
    check(restore["in_process_equal"], "zero-step baseline restores in-process")
    check(restore["fresh_process_equal"], "zero-step baseline restores in a fresh process")

    return {
        "schema": "taiji-corrected-old-r3b-probe-v1",
        "status": "r3b_structure_passed" if not checks["fail"] else "r3b_structure_failed",
        "fixture": {
            "path": str(MANIFEST.relative_to(ROOT)).replace("\\", "/"),
            "sha256": _sha256_file(MANIFEST),
            "cases": len(cases),
        },
        "config": _config().__dict__,
        "parameters": CorrectedOldBundleV1.create(_config()).model.parameter_count(),
        "trainable_parameters": list(
            CorrectedOldBundleV1.create(_config()).model.trainable_parameters()
        ),
        "cases": rows,
        "checks": checks,
        "readouts": {
            "cut_delta": cut_delta,
            "rescue_delta": rescue_delta,
            "step0_target_byte_delta": row0_delta,
            "step1_prefix_delta": row1_delta,
        },
        "fixed_denominator": score,
        "restore": restore,
        "model_calls": {
            "teacher_forced": len(parsed_queries) + 2,
            "generated_answers": len(parsed_queries),
            "restore_probe": 2,
            "fresh_process": 1,
        },
        "optimizer_updates": 0,
        "training_started": False,
        "sealed_text_read": 0,
        "can_promote": False,
        "source_sha256": {
            "taiji/corrected_old_baseline.py": _sha256_file(MODULE),
            "tests/taiji_native/test_corrected_old_baseline.py": _sha256_file(
                ROOT / "tests/taiji_native/test_corrected_old_baseline.py"
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
    return 0 if payload["status"] == "r3b_structure_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
