"""Read-only attribution for the already-consumed R2 dialogue generation sample."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "plans/reference/PLAN-R2-DAILY-DIALOGUE-ATTRIBUTION_20261021.md"
PLAN_SHA256 = "e26068b0e69458fa22c200ffd00ab39caa7973bec88cf3bd302eff4984798516"
RESULT = ROOT / "reports/taiji_r2_daily_dialogue_20261021_retry2_result.json"
GENERATIONS = ROOT / "reports/taiji_r2_daily_dialogue_20261021_retry2_generations.json"
PROGRESS = ROOT / "reports/taiji_r2_daily_dialogue_20261021_retry2_progress.jsonl"
REPORT = ROOT / "reports/taiji_r2_daily_dialogue_20261021_attribution.json"
RESULT_SHA256 = "e939e4c91219b709e0a3799db13610fc9cb1407916d3a9240b4af451f00411f0"
GENERATIONS_SHA256 = "32b97f4a66d562a607967eb17ac0fbb452307cd429806f551ff3c42acbc0d02f"
PROGRESS_SHA256 = "de48bf18b38ae80216debf3f413e0534b626f46e1d58a66fa09ead7881bfe2fd"
ENDPOINTS = (
    "parent_with_history",
    "parent_single_turn",
    "dialogue_context",
    "single_turn",
)


class Refuse(RuntimeError):
    """Frozen R2 dialogue outputs cannot be attributed safely."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"R2 dialogue attribution output is write-once: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()


def _summary(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(outputs)
    texts = [str(row["generated_text"]) for row in outputs]
    return {
        "generation_count": count,
        "utf8_valid_rate": sum(bool(row["utf8_valid"]) for row in outputs) / count,
        "no_replacement_rate": sum(bool(row["no_replacement"]) for row in outputs) / count,
        "boundary_rate": sum(bool(row["response_boundary_present"]) for row in outputs) / count,
        "mean_output_bytes": sum(int(row["generated_bytes_length"]) for row in outputs) / count,
        "unique_output_count": len(set(texts)),
        "duplicate_output_rate": 1.0 - len(set(texts)) / count,
        "mean_repeated_4gram_rate": sum(float(row["repeated_4gram_rate"]) for row in outputs)
        / count,
    }


def _fourgram_repeat_rate(text: str) -> float:
    compact = "".join(text.split())
    grams = [compact[index : index + 4] for index in range(max(0, len(compact) - 3))]
    return 0.0 if not grams else 1.0 - len(set(grams)) / len(grams)


def run() -> dict[str, Any]:
    started = time.process_time()
    if REPORT.exists():
        raise FileExistsError(f"R2 dialogue attribution report already exists: {REPORT}")
    if sha256(PLAN) != PLAN_SHA256:
        raise Refuse("R2 dialogue attribution plan SHA changed")
    for path, expected in (
        (RESULT, RESULT_SHA256),
        (GENERATIONS, GENERATIONS_SHA256),
        (PROGRESS, PROGRESS_SHA256),
    ):
        if sha256(path) != expected:
            raise Refuse(f"consumed R2 dialogue source SHA changed: {path}")

    result = json.loads(RESULT.read_text(encoding="utf-8"))
    generation_report = json.loads(GENERATIONS.read_text(encoding="utf-8"))
    if (
        result.get("status") != "complete"
        or result.get("can_promote") is not False
        or result.get("generation_query_count") != 48
        or result.get("reference_answer_scoring_queries") != 0
        or result.get("dev_queries") != 0
        or generation_report.get("status") != "complete"
        or generation_report.get("generation_configuration", {}).get("reference_answer_included")
        is not False
        or generation_report.get("generation_configuration", {}).get("exact_response_scored")
        is not False
    ):
        raise Refuse("R2 dialogue source reports do not match the frozen attribution scope")

    records = generation_report.get("records", [])
    if len(records) != 12 or any(
        set(row).intersection({"reference_response", "expected_answer"}) for row in records
    ):
        raise Refuse("generation records are incomplete or include a reference target")
    for row in records:
        if (
            row.get("reference_answer_included") is not False
            or row.get("exact_response_scored") is not False
        ):
            raise Refuse("a consumed generation row exposed or scored its target")
        if set(ENDPOINTS) - set(row):
            raise Refuse(f"generation endpoint rows are incomplete: {row.get('episode_id')}")

    recomputed = {endpoint: _summary([row[endpoint] for row in records]) for endpoint in ENDPOINTS}
    bound_metrics = generation_report.get("metrics", {})
    for endpoint in ENDPOINTS:
        for metric, value in recomputed[endpoint].items():
            if abs(float(value) - float(bound_metrics[endpoint][metric])) > 1e-12:
                raise Refuse(f"generation summary mismatch: {endpoint}/{metric}")

    with_history = [row for row in records if int(row["history_turn_count"]) > 0]
    without_history = [row for row in records if int(row["history_turn_count"]) == 0]
    parent_context_pairs = [
        row["parent_with_history"]["generated_text"] == row["parent_single_turn"]["generated_text"]
        for row in with_history
    ]
    same_input_candidate_control = [
        row["dialogue_context"]["generated_text"] == row["single_turn"]["generated_text"]
        for row in without_history
    ]

    report = {
        "schema": "taiji-r2-daily-dialogue-generation-attribution-v1",
        "status": "complete",
        "plan_sha256": PLAN_SHA256,
        "source_hashes": {
            str(path.resolve()): sha256(path) for path in (PLAN, RESULT, GENERATIONS, PROGRESS)
        },
        "input_row_count": len(records),
        "endpoint_summary_recomputed": recomputed,
        "parent_history_ablation": {
            "paired_rows": len(parent_context_pairs),
            "exactly_same_outputs": sum(parent_context_pairs),
            "outputs_differed": sum(not value for value in parent_context_pairs),
            "interpretation": "描述输入历史是否改变父档输出；不代表回答是否正确。",
        },
        "same_current_turn_candidate_control": {
            "paired_rows": len(same_input_candidate_control),
            "exactly_same_outputs": sum(same_input_candidate_control),
            "outputs_differed": sum(not value for value in same_input_candidate_control),
            "interpretation": "仅在没有历史的六条样本上比较candidate和control；不代表语义质量。",
        },
        "manual_sample_review": {
            "status": "reviewed",
            "observation_zh": "样例中的candidate仍以重复字词和局部片段拼接为主，暂未形成稳定、连贯、贴题的日常中文回复。边界率改善不能替代语义对话质量。",
            "score": None,
            "is_automated_semantic_metric": False,
        },
        "limitations_zh": [
            "final面只抽取12棵自然对话树，属于定性样例，不是统计性泛化门。",
            "本报告没有参考答案、exact-match或teacher-forced评分。",
            "候选与单轮对照在最终输入历史上不同；候选相对control差异只支持整体续训包描述。",
        ],
        "model_loads": 0,
        "checkpoint_loads": 0,
        "generation_queries": 0,
        "parameter_updates": 0,
        "cpu_seconds": round(time.process_time() - started, 6),
        "can_promote": False,
    }
    write_new(REPORT, report)
    return report


def main() -> int:
    try:
        report = run()
    except Exception as exc:
        print(f"R2 dialogue attribution refused: {type(exc).__name__}: {exc}")
        return 1
    print(
        json.dumps(
            {
                "status": report["status"],
                "generation_queries": report["generation_queries"],
                "parent_history_ablation": report["parent_history_ablation"],
                "same_input_candidate_control": report["same_current_turn_candidate_control"],
                "can_promote": report["can_promote"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
