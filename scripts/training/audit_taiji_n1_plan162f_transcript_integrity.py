"""离线复核N1-162F封存progress/result的题面覆盖和输出字节。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import analyze_taiji_n1_plan162d_transition_output_type_attribution as n162d
from scripts.training import build_taiji_n1_v27_length_robust_training_after_n149 as n150_builder
from scripts.training import (
    run_taiji_n1_plan162f_transition_template_crossover_diagnostic as engine,
)

PLAN = ROOT / "plans/reference/PLAN-N1-162F1_transcript_integrity_revalidation_20261020.md"
PLAN_SHA256 = "2346d669beeecc98390784a5b3940d88b1a804bb5de009e5fb8068a4e670e8eb"
F_PLAN = engine.PLAN
F_PLAN_SHA256 = engine.PLAN_SHA256
F_ATTEMPT = engine.ATTEMPT
F_ATTEMPT_SHA256 = "edc3707545da389837d7125d6b61518c29374699eb3cc9c45705f5b433c054d5"
F_PROGRESS = engine.PROGRESS
F_PROGRESS_SHA256 = "baed16dc5cad86e6ceabcf457f5cddec561f89e65ba370666e9b1417ed5d47f6"
F_RESULT = engine.RESULT
F_RESULT_SHA256 = "bb42a2017a7935c600a5b91c43025aa2381dedf7a9335ed86f071283c52498d8"
F_ENGINE = Path(engine.__file__).resolve()
F_RUNNER_SHA_AT_COMPLETION = "10e1772f2c6bafc3ef3feb481a923dffa327f14b6fdb64d81c7074dfc3681999"
F_CURRENT_RUNNER_SHA = "a6faaa1405f3239a671534b9666cb52201d635edc613206c44ba326e78d4594c"
F_TEST = engine.TEST
REPORT = ROOT / "reports/taiji_n1_plan162f_transcript_integrity_20261020.json"
TEST = ROOT / "tests/taiji_native/test_n1_plan162f1_transcript_integrity_contract.py"
CPU_LIMIT_SECONDS = 120.0


class Refuse(RuntimeError):
    """封存F题面、progress/result或来源合同不成立。"""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Refuse(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path: Path, value: dict) -> None:
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()


def _query_key(row: dict) -> tuple:
    return (
        str(row.get("lineage")),
        int(row.get("seed", -1)),
        str(row.get("view")),
        str(row.get("parent_id")),
        str(row.get("condition")),
        str(row.get("case_id")),
        str(row.get("question_template_id")),
    )


def _expected_queries(cases: dict[tuple, dict], endpoints: list[dict]) -> dict[tuple, dict]:
    expected = {}
    for endpoint in endpoints:
        for case in cases.values():
            entity = engine.n132._query_subject(case["item"]["question"])
            for template_id, template in engine.TEMPLATES.items():
                key = (
                    endpoint["lineage"],
                    int(endpoint["seed"]),
                    case["view"],
                    case["parent_id"],
                    case["condition"],
                    case["case_id"],
                    template_id,
                )
                if key in expected:
                    raise Refuse(f"duplicate expected F query key: {key}")
                expected[key] = {
                    "question": template.format(entity=entity),
                    "expected_answer": case["expected_answer"],
                    "initial_direction": case["initial_direction"],
                    "turn": case["turn"],
                }
    return expected


def _read_progress() -> tuple[list[dict], dict[str, int], int]:
    rows = []
    counts = Counter()
    malformed = []
    total = 0
    with F_PROGRESS.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            total += 1
            try:
                row = json.loads(line)
            except Exception as exc:
                malformed.append({"line": number, "error": type(exc).__name__})
                continue
            counts[str(row.get("event"))] += 1
            rows.append(row)
    require(not malformed, f"F progress has malformed rows: {malformed[:3]}")
    require(
        dict(counts) == {"transition_template_query": engine.TOTAL_QUERIES},
        f"F progress event count differs: {dict(counts)}",
    )
    return rows, dict(sorted(counts.items())), total


def _compare_old_source_hashes(result: dict) -> dict:
    hashes = result.get("source_hashes", {})
    runner_path = str(F_ENGINE.resolve())
    require(
        hashes.get(runner_path) == F_RUNNER_SHA_AT_COMPLETION,
        "F result expected runner SHA differs from recorded drift",
    )
    require(
        sha256(F_ENGINE) == F_CURRENT_RUNNER_SHA,
        "current F runner SHA changed after revalidation plan",
    )
    unchanged = {}
    bad = []
    missing = []
    for path, expected in hashes.items():
        if path == runner_path:
            continue
        file_path = Path(path)
        if not file_path.is_file():
            missing.append(path)
            continue
        actual = sha256(file_path)
        if actual != expected:
            bad.append({"path": path, "expected_sha256": expected, "actual_sha256": actual})
        else:
            unchanged[path] = expected
    return {
        "recorded_source_count": len(hashes),
        "current_source_count_excluding_runner": len(unchanged),
        "runner_path": runner_path,
        "runner_sha256_at_execution": F_RUNNER_SHA_AT_COMPLETION,
        "runner_sha256_current": F_CURRENT_RUNNER_SHA,
        "runner_source_drift": True,
        "other_source_drift": bad,
        "missing_sources": missing,
        "all_other_sources_unchanged": not bad and not missing,
    }


def _validate_transcript(result: dict, progress: list[dict], expected: dict) -> dict:
    result_rows = result.get("output_rows", [])
    require(
        len(result_rows) == len(expected) == len(progress),
        "F output/progress row denominator differs",
    )
    result_by_key = {}
    for row in result_rows:
        key = _query_key(row)
        if key in result_by_key:
            raise Refuse(f"duplicate result query key: {key}")
        result_by_key[key] = row
    progress_by_key = {}
    for row in progress:
        if row.get("event") != "transition_template_query":
            raise Refuse("unexpected F progress event")
        key = _query_key(row)
        if key in progress_by_key:
            raise Refuse(f"duplicate progress query key: {key}")
        progress_by_key[key] = {name: value for name, value in row.items() if name != "event"}
    require(
        set(result_by_key) == set(expected) == set(progress_by_key),
        "F result/progress key coverage differs from reconstructed plan",
    )
    field_mismatches = Counter()
    output_type_mismatches = []
    for key, want in expected.items():
        row = result_by_key[key]
        for field, value in want.items():
            if row.get(field) != value:
                field_mismatches[field] += 1
        if progress_by_key[key] != row:
            differing = [
                field
                for field in set(progress_by_key[key]) | set(row)
                if progress_by_key[key].get(field) != row.get(field)
            ]
            for field in differing:
                field_mismatches[f"progress_result/{field}"] += 1
        output_type = n162d._classify_output(
            str(row["raw_hex"]),
            str(row["stop_reason"]),
            str(want["expected_answer"]),
            str(want["initial_direction"]),
            str(want["turn"]),
        )
        if output_type != row.get("output_type") or (output_type == "correct_exact") != (
            row.get("free_exact") is True
        ):
            output_type_mismatches.append(
                {
                    "key": key,
                    "stored_type": row.get("output_type"),
                    "recomputed_type": output_type,
                    "stored_free_exact": row.get("free_exact"),
                }
            )
    require(
        not field_mismatches and not output_type_mismatches,
        f"F transcript result/answer/output recomputation differs: {dict(field_mismatches)} / {len(output_type_mismatches)}",
    )
    recomputed_summary, recomputed_pairs = engine._summarize(result_rows)
    require(
        recomputed_summary == result.get("summary_by_endpoint_view_condition_template"),
        "F per-endpoint summary differs from output rows",
    )
    require(
        recomputed_pairs == result.get("paired_template_type_transitions"),
        "F paired-template summary differs from output rows",
    )
    return {
        "expected_query_keys": len(expected),
        "result_rows": len(result_rows),
        "progress_rows": len(progress),
        "result_progress_rows_match": True,
        "question_answer_fields_recomputed": True,
        "output_types_recomputed": len(result_rows),
        "endpoint_summaries_recomputed": len(recomputed_summary),
        "paired_template_rows_recomputed": len(recomputed_pairs),
        "field_mismatches": dict(field_mismatches),
        "output_type_mismatches": output_type_mismatches,
    }


def audit() -> dict:
    started = time.process_time()
    if REPORT.exists():
        raise FileExistsError(f"F1 transcript report is write-once: {REPORT}")
    require(sha256(PLAN) == PLAN_SHA256, "PLAN-N1-162F1 plan SHA changed")
    require(sha256(F_PLAN) == F_PLAN_SHA256, "PLAN-N1-162F plan SHA changed")
    require(sha256(F_ATTEMPT) == F_ATTEMPT_SHA256, "F attempt SHA changed")
    require(sha256(F_PROGRESS) == F_PROGRESS_SHA256, "F progress SHA changed")
    require(sha256(F_RESULT) == F_RESULT_SHA256, "F result SHA changed")
    result = read_json(F_RESULT)
    require(
        result.get("complete") is True and result.get("status") == "complete",
        "F prior attempt is not complete",
    )
    require(
        result.get("queries") == engine.TOTAL_QUERIES
        and result.get("parameter_updates") == 0
        and result.get("can_promote") is False,
        "F prior run crossed its frozen boundary",
    )
    prior_runner = _compare_old_source_hashes(result)
    require(
        prior_runner["all_other_sources_unchanged"] is True,
        "F transcript has source drift beyond the known runner change",
    )
    source_hashes, endpoints, source_context = engine._source_and_endpoints()
    require(len(endpoints) == 9, "F1 revalidation requires nine frozen endpoints")
    bound = engine.n132._build_context(load_inputs=True)
    cases = engine._build_cases(bound["inputs"], bound["g"]["calibration"].engineering.factorial)
    expected = _expected_queries(cases, endpoints)
    require(len(expected) == engine.TOTAL_QUERIES, "F plan-derived expected query count differs")
    progress, event_counts, line_count = _read_progress()
    transcript = _validate_transcript(result, progress, expected)
    require(
        sha256(F_PROGRESS) == F_PROGRESS_SHA256 and sha256(F_RESULT) == F_RESULT_SHA256,
        "F transcript changed during revalidation",
    )
    require(n150_builder.verify_hashes(source_hashes), "F1 parent/checkpoint/source hashes changed")
    cpu = time.process_time() - started
    complete = (
        cpu <= CPU_LIMIT_SECONDS
        and transcript["expected_query_keys"] == engine.TOTAL_QUERIES
        and prior_runner["all_other_sources_unchanged"]
    )
    audit_sources = {
        **source_hashes,
        str(F_PLAN.resolve()): F_PLAN_SHA256,
        str(PLAN.resolve()): PLAN_SHA256,
        str(F_ATTEMPT.resolve()): F_ATTEMPT_SHA256,
        str(F_PROGRESS.resolve()): F_PROGRESS_SHA256,
        str(F_RESULT.resolve()): F_RESULT_SHA256,
        str(F_ENGINE.resolve()): F_CURRENT_RUNNER_SHA,
        str(Path(__file__).resolve()): sha256(Path(__file__)),
        str(TEST.resolve()): sha256(TEST),
    }
    report = {
        "schema": "taiji-n1-plan162f1-transcript-integrity-revalidation-v1",
        "status": "complete_with_runner_source_drift" if complete else "stopped",
        "complete": complete,
        "plan_sha256": PLAN_SHA256,
        "f_plan_sha256": F_PLAN_SHA256,
        "f_attempt_sha256": F_ATTEMPT_SHA256,
        "f_progress_sha256": F_PROGRESS_SHA256,
        "f_result_sha256": F_RESULT_SHA256,
        "source_hashes": audit_sources,
        "source_hashes_unchanged": n150_builder.verify_hashes(audit_sources),
        "prior_runner_drift": prior_runner,
        "transcript_revalidation": transcript,
        "endpoint_count": len(endpoints),
        "case_count_per_endpoint": len(cases),
        "queries_revalidated": transcript["result_rows"],
        "progress_line_count": line_count,
        "progress_event_counts": event_counts,
        "training_input_view_count": sum(
            len(bound["inputs"]["views"][name]) for name in ("original", "rename")
        ),
        "factorial_cases_rederived": len(cases),
        "execution_cpu_seconds": round(cpu, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "new_model_loads": 0,
        "new_checkpoint_loads": 0,
        "new_inference_calls": 0,
        "new_model_queries": 0,
        "parameter_updates": 0,
        "dev_queries": 0,
        "holdout_queries": 0,
        "confirmation_queries": 0,
        "can_promote": False,
        "summary_zh": f"F1逐题转录复核{'完成' if complete else '停止'}：旧F结果/进度原样保留，6,912个问句/答案键、输出类别及free_exact逐行复算；唯一来源漂移为F runner SHA，其他来源不变，0新查询/加载/更新，CPU{cpu:.3f}/{CPU_LIMIT_SECONDS:.0f}秒。",
    }
    write_new(REPORT, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="PLAN-N1-162F1 transcript integrity revalidation")
    parser.add_argument("--audit", action="store_true", required=True)
    args = parser.parse_args()
    del args
    result = audit()
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "status",
                    "complete",
                    "execution_cpu_seconds",
                    "queries_revalidated",
                    "new_model_queries",
                    "parameter_updates",
                    "can_promote",
                    "summary_zh",
                )
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0 if result.get("complete") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
