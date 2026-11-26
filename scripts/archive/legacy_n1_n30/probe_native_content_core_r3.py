"""R3 structural probe for the native content core.

Reads the public R1 fixture, materialises a versioned store per case, and
reports mechanism-level readouts without training: candidate counts per time
intent, same-query/different-current separation, same-current history
separation, time-intent separation, related vs unrelated swap, cut and legal
rescue, first-target-byte supervision, and zero-step bundle restoration in a
fresh process. It does not train, does not evaluate answer correctness and does
not read any sealed text.
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

from taiji.internalization import content_digest  # noqa: E402
from taiji.native_content_core import (  # noqa: E402
    NativeContentBundleV1,
    NativeContentCoreConfigV1,
    NativeContentCoreV1,
    NativeContentStoreV1,
)

MANIFEST = ROOT / "plans/manifests/native_core_r1_public_fixture_v1.json"
REPORT = ROOT / "reports/taiji_native_core_r3_probe_20261030.json"
MODULE = ROOT / "taiji/native_content_core.py"
TARGET = b"ab"
NEGATIVE_LATENT_SPAN = "黄色".encode()
OWNER, SCOPE, SESSION = "public", "r1-fixture", "fixture"
_CALLS = {"teacher_forced": 0, "read": 0, "restore_probe": 0, "fresh_process": 0}


def _teacher_forced(bundle: NativeContentBundleV1, condition, target: bytes = TARGET):
    _CALLS["teacher_forced"] += 1
    return bundle.model.teacher_forced_conditioned(
        condition, target, scope_id=SCOPE, session_id=SESSION
    )


def _config() -> NativeContentCoreConfigV1:
    return NativeContentCoreConfigV1(
        width=64,
        embedding_width=32,
        key_width=32,
        slots=8,
        chunk_bytes=8,
        max_observation_bytes=512,
        max_input_evidence_bytes=512,
        max_output_position=128,
        max_generation_bytes=64,
        content_width=32,
        seed=101,
        position_extension_seed=102,
        read_seed=103,
    )


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _materialise(case: dict) -> NativeContentBundleV1:
    bundle = NativeContentBundleV1.create(_config(), cursor={"fixture_id": case["fixture_id"]})
    for turn, entry in enumerate(case["history"]):
        bundle.ingest(
            raw_span=entry["source_text"].encode("utf-8"),
            source_input_id=entry["source_input_id"],
            source_sha256=entry["source_sha256"],
            owner=OWNER,
            scope=SCOPE,
            session=SESSION,
            turn_index=turn,
            revision=entry["revision"],
            valid_from=entry["valid_from"],
            entity=case["entity"],
            attribute=case["attribute"],
            value=entry["value"],
            status="resolved",
        )
    if case.get("distractor"):
        distractor = case["distractor"]
        bundle.ingest(
            raw_span=distractor["source_text"].encode("utf-8"),
            source_input_id=f"r1-public:distractor:{case['fixture_id']}",
            source_sha256=_sha256_text(distractor["source_text"]),
            owner=OWNER,
            scope=SCOPE,
            session=SESSION,
            turn_index=len(case["history"]) + 1,
            revision=0,
            valid_from=len(case["history"]) + 1,
            entity=distractor["entity"],
            attribute=distractor["attribute"],
            value=distractor["value"],
            status="resolved",
        )
    return bundle


def _condition(bundle: NativeContentBundleV1, case: dict, time_intent: str | None = None):
    intent = case["time_intent"] if time_intent is None else time_intent
    assert case["query_text"] is not None
    return bundle.condition(
        query=case["query_text"].encode("utf-8"),
        time_intent=intent,
        owner=OWNER,
        scope=SCOPE,
        session=SESSION,
    )


def _logits(bundle: NativeContentBundleV1, case: dict, time_intent: str | None = None):
    return _teacher_forced(bundle, _condition(bundle, case, time_intent))


def _selected_id(bundle: NativeContentBundleV1, case: dict) -> str | None:
    _CALLS["read"] += 1
    condition = _condition(bundle, case)
    state = bundle.model.conditioned_state(
        condition, scope_id=SCOPE, session_id=SESSION, native_state_ref="r3-probe"
    )
    cache = bundle.model.content_read_cache(state, condition)
    with torch.no_grad():
        _, scores = bundle.model._read(cache, state.hidden)
    if not cache.content_ids:
        return None
    return cache.content_ids[int(scores.argmax())]


def _delta(left, right) -> float:
    return float((left - right).abs().max().detach())


def _digest(logits) -> str:
    return content_digest([[round(float(value), 6) for value in row] for row in logits.tolist()])


def _restore_probe() -> dict[str, Any]:
    bundle = NativeContentBundleV1.create(_config(), cursor={"fixture_id": "restore-probe"})
    for turn, (value, revision) in enumerate((("红色", 0), ("蓝色", 1))):
        text = f"雨伞的颜色是{value}。"
        bundle.ingest(
            raw_span=text.encode("utf-8"),
            source_input_id=f"r3-probe:restore:{revision}",
            source_sha256=_sha256_text(text),
            owner="probe",
            scope="probe",
            session="probe",
            turn_index=turn,
            revision=revision,
            valid_from=turn,
            entity="雨伞",
            attribute="颜色",
            value=value,
            status="resolved",
        )
    query = "雨伞的颜色是什么？".encode()
    _CALLS["restore_probe"] += 2
    expected = bundle.probe_digest(query=query)
    path = ROOT / ".seed_test_tmp" / "r3_native_content_bundle.pt"
    if path.exists():
        path.unlink()
    bundle.save(path)
    restored = NativeContentBundleV1.load(path)
    in_process = restored.probe_digest(query=query)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    code = (
        "import sys; "
        "from taiji.native_content_core import NativeContentBundleV1; "
        "b=NativeContentBundleV1.load(sys.argv[1]); "
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
    _CALLS["fresh_process"] += 1
    fresh_process = result.stdout.strip()
    return {
        "saved_parameter_revision": bundle.model.parameter_revision,
        "saved_records": len(bundle.store.records),
        "expected_digest": expected,
        "in_process_digest": in_process,
        "fresh_process_digest": fresh_process,
        "in_process_equal": in_process == expected,
        "fresh_process_equal": fresh_process == expected,
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
    bundles: dict[str, NativeContentBundleV1] = {}
    logits_by_case: dict[str, Any] = {}
    previous_ids: dict[str, str] = {}

    for case in cases:
        bundle = _materialise(case)
        bundles[case["fixture_id"]] = bundle
        if case["query_text"] is None:
            current = bundle.store.current(case["entity"], case["attribute"])
            rows.append(
                {
                    "fixture_id": case["fixture_id"],
                    "group_id": case["group_id"],
                    "query_text": None,
                    "time_intent": None,
                    "candidates": None,
                    "current_value": None if current is None else current.value,
                }
            )
            check(
                current is not None and current.value == case["expected_state"]["value"],
                f"{case['fixture_id']}: non-query receipt keeps the written current value",
            )
            continue
        current_rows = bundle.store.readable(
            owner=OWNER, scope=SCOPE, session=SESSION, time_intent="current"
        )
        previous_rows = bundle.store.readable(
            owner=OWNER, scope=SCOPE, session=SESSION, time_intent="previous"
        )
        expected_current = (1 if case["history"] else 0) + (1 if case.get("distractor") else 0)
        expected_previous = 1 if len(case["history"]) >= 2 else 0
        check(
            len(current_rows) == expected_current,
            f"{case['fixture_id']}: current candidate count",
            f"{len(current_rows)} != {expected_current}",
        )
        check(
            len(previous_rows) == expected_previous,
            f"{case['fixture_id']}: previous candidate count",
            f"{len(previous_rows)} != {expected_previous}",
        )
        if previous_rows:
            previous_ids[case["fixture_id"]] = previous_rows[0].content_id
        logits = _logits(bundle, case)
        logits_by_case[case["fixture_id"]] = logits
        rows.append(
            {
                "fixture_id": case["fixture_id"],
                "group_id": case["group_id"],
                "query_text": case["query_text"],
                "time_intent": case["time_intent"],
                "candidates": [item.content_id for item in current_rows],
                "previous_candidates": [item.content_id for item in previous_rows],
                "current_values": [item.value for item in current_rows],
                "previous_values": [item.value for item in previous_rows],
                "logits_digest": _digest(logits),
            }
        )

    def paired(name: str) -> list[dict]:
        return groups.get(name, [])

    same_query = paired("same-query-current")
    if len(same_query) == 2:
        left, right = same_query
        check(
            _delta(logits_by_case[left["fixture_id"]], logits_by_case[right["fixture_id"]]) > 0,
            "same query with different current values changes the readout",
        )
        check(
            _digest(logits_by_case[left["fixture_id"]])
            != _digest(logits_by_case[right["fixture_id"]]),
            "same query different current: distinct logits digest",
        )

    same_history = paired("same-current-history")
    if len(same_history) == 2:
        left, right = same_history
        current_delta = _delta(
            logits_by_case[left["fixture_id"]], logits_by_case[right["fixture_id"]]
        )
        check(current_delta == 0, "same current value keeps current readout identical")
        previous_left = next(
            item
            for item in bundles[left["fixture_id"]].store.records
            if item.content_id == previous_ids.get(left["fixture_id"])
        )
        previous_right = next(
            item
            for item in bundles[right["fixture_id"]].store.records
            if item.content_id == previous_ids.get(right["fixture_id"])
        )
        check(
            previous_left.value != previous_right.value,
            "same-current pair has different previous values",
        )
        check(
            _delta(
                _logits(bundles[left["fixture_id"]], left, "previous"),
                _logits(bundles[right["fixture_id"]], right, "previous"),
            )
            > 0,
            "same current value with different history changes the previous readout",
        )

    same_time = paired("same-query-time")
    if len(same_time) == 2:
        current_case, previous_case = same_time
        check(
            _delta(
                _logits(bundles[current_case["fixture_id"]], current_case, "current"),
                _logits(bundles[previous_case["fixture_id"]], previous_case, "previous"),
            )
            > 0,
            "same query text with different time intent changes the readout",
        )

    three = paired("three-versions")
    if three:
        case = three[0]
        previous = bundles[case["fixture_id"]].store.readable(
            owner=OWNER, scope=SCOPE, session=SESSION, time_intent="previous"
        )
        check(
            len(previous) == 1 and previous[0].value == "蓝色",
            "three versions: previous is revision 1, not revision 0",
        )

    same_value = paired("same-value-version")
    if same_value:
        case = same_value[0]
        bundle = bundles[case["fixture_id"]]
        current = bundle.store.current(case["entity"], case["attribute"])
        previous = bundle.store.previous(case["entity"], case["attribute"])
        check(
            current is not None
            and previous is not None
            and previous.value == current.value
            and previous.content_id != current.content_id,
            "same-value correction keeps a distinct previous revision",
        )

    unknown_previous = paired("unknown-previous")
    if unknown_previous:
        case = unknown_previous[0]
        check(
            not bundles[case["fixture_id"]].store.readable(
                owner=OWNER, scope=SCOPE, session=SESSION, time_intent="previous"
            ),
            "previous without a predecessor has no candidate and does not fall back to current",
        )

    for name in ("unknown-current", "ambiguous", "negation"):
        group = paired(name)
        if group:
            case = group[0]
            check(
                not bundles[case["fixture_id"]].store.readable(
                    owner=OWNER, scope=SCOPE, session=SESSION, time_intent="current"
                ),
                f"{name}: empty legal candidate set",
            )

    alias = paired("owner")
    if alias:
        case = alias[0]
        check(
            not bundles[case["fixture_id"]].store.readable(
                owner="other", scope=SCOPE, session=SESSION, time_intent="current"
            ),
            "owner boundary: another owner sees no candidate",
        )

    # Relevant/unrelated swap, cut and legal rescue on the correction case.
    correction = paired("correction")[0]
    bundle = bundles[correction["fixture_id"]]
    selected = _selected_id(bundle, correction)
    assert selected is not None
    baseline = _logits(bundle, correction)
    selected_record = next(item for item in bundle.store.records if item.content_id == selected)
    with torch.no_grad():
        negative_latent = bundle.model.encode_content(NEGATIVE_LATENT_SPAN).tolist()
    swapped = selected_record.reencode(
        latent=negative_latent, encoder_revision=bundle.model.parameter_revision
    )
    saved_store: NativeContentStoreV1 = bundle.store
    bundle.store = saved_store.swap(swapped)
    relevant_delta = _delta(baseline, _logits(bundle, correction))
    bundle.store = saved_store
    rescue_delta = _delta(baseline, _logits(bundle, correction))
    bundle.store = saved_store.cut(selected)
    cut_delta = _delta(baseline, _logits(bundle, correction))
    bundle.store = saved_store
    check(relevant_delta > 0, "relevant content swap changes the readout")
    check(cut_delta > 0, "cut removes the selected candidate and changes the readout")
    check(rescue_delta == 0, "legal rescue restores the exact baseline readout", f"{rescue_delta}")

    distractor = paired("distractor")
    unrelated_delta = None
    if distractor:
        case = distractor[0]
        bundle = bundles[case["fixture_id"]]
        condition = _condition(bundle, case)
        main = next(item for item in condition.records if item.value != case["distractor"]["value"])
        unrelated = next(item for item in condition.records if item.content_id != main.content_id)
        baseline = _logits(bundle, case)
        with torch.no_grad():
            other_latent = bundle.model.encode_content("他的".encode()).tolist()
        replacement = unrelated.reencode(
            latent=other_latent, encoder_revision=bundle.model.parameter_revision
        )
        saved_store = bundle.store
        bundle.store = saved_store.swap(replacement)
        unrelated_delta = _delta(baseline, _logits(bundle, case))
        bundle.store = saved_store

    # First target byte supervision: row 0 does not see target[0], row 1 does.
    correction_bundle = bundles[correction["fixture_id"]]
    left = _logits(correction_bundle, correction)
    right = _teacher_forced(correction_bundle, _condition(correction_bundle, correction), b"cb")
    row0_delta = _delta(left[0], right[0])
    row1_delta = _delta(left[1], right[1])
    check(row0_delta == 0, "step-0 logits do not depend on target[0]", f"{row0_delta}")
    check(row1_delta > 0, "step-1 logits depend on the observed first target byte", f"{row1_delta}")

    restore = _restore_probe()
    check(restore["in_process_equal"], "zero-step bundle restores in-process")
    check(restore["fresh_process_equal"], "zero-step bundle restores in a fresh process")

    readouts = {
        "relevant_swap_delta": relevant_delta,
        "unrelated_swap_delta": unrelated_delta,
        "relevant_to_unrelated_ratio": (
            None if not unrelated_delta else relevant_delta / unrelated_delta
        ),
        "cut_delta": cut_delta,
        "rescue_delta": rescue_delta,
        "step0_target_byte_delta": row0_delta,
        "step1_prefix_delta": row1_delta,
        "same_query_current_delta": (
            _delta(
                logits_by_case[same_query[0]["fixture_id"]],
                logits_by_case[same_query[1]["fixture_id"]],
            )
            if len(same_query) == 2
            else None
        ),
        "same_current_history_current_delta": (
            _delta(
                logits_by_case[same_history[0]["fixture_id"]],
                logits_by_case[same_history[1]["fixture_id"]],
            )
            if len(same_history) == 2
            else None
        ),
    }

    return {
        "schema": "taiji-native-content-core-r3-probe-v1",
        "status": "r3_structure_passed" if not checks["fail"] else "r3_structure_failed",
        "fixture": {
            "path": str(MANIFEST.relative_to(ROOT)).replace("\\", "/"),
            "sha256": _sha256_file(MANIFEST),
            "cases": len(cases),
            "sealed": manifest["world"]["sealed"],
        },
        "config": _config().__dict__,
        "parameters": NativeContentCoreV1(_config()).parameter_count(),
        "cases": rows,
        "checks": checks,
        "readouts": readouts,
        "restore": restore,
        "byte_protocol": {
            "target": "target_t_at_step_t",
            "first_byte_supervised": True,
            "forbidden": "target[1:]",
        },
        "model_calls": sum(_CALLS.values()),
        "model_call_breakdown": dict(_CALLS),
        "fresh_process_restores": 1,
        "optimizer_updates": 0,
        "training_started": False,
        "sealed_text_read": 0,
        "can_promote": False,
        "source_sha256": {
            "taiji/native_content_core.py": _sha256_file(MODULE),
            "tests/taiji_native/test_native_content_core.py": _sha256_file(
                ROOT / "tests/taiji_native/test_native_content_core.py"
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
    return 0 if payload["status"] == "r3_structure_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
