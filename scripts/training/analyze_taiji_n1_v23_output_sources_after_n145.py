"""PLAN-N1-146：离线分类v23确认输出与题内事实值的对应关系。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PLAN = ROOT / "plans/reference/PLAN-N1-146_v23_output_source_item_attribution_draft_20261019.md"
PLAN_SHA256 = "9b97dc49206077076d513ba172d15bd267716a6c12a6a4d47d5e601ee21b1baa"
N144_PLAN = (
    ROOT / "plans/reference/PLAN-N1-144_v23_long_context_confirmation_after_n143_draft_20261019.md"
)
N144_PLAN_SHA256 = "b5527b5f8a56aa12604a7dbc40dad1fa74c3f4ea8e953a6b2539f1fa3c5fcf42"
N145_PLAN = ROOT / "plans/reference/PLAN-N1-145_v23_itemwise_error_attribution_draft_20261019.md"
N145_PLAN_SHA256 = "647c70092b6efae729f29849bf77e36032f899b63c05399f8c497e679fa651d9"
FACE = ROOT / "plans/manifests/n1_v23_confirmation_seed123_20261019.json"
FACE_SHA256 = "e0ddf062e3c5f081dd010db3a8951d9aa426d9717de39872527a4eeabb76f6ae"
N144_RESULT = ROOT / "reports/taiji_n1_v23_confirmation_result_20261019.json"
N144_RESULT_SHA256 = "67038a9848b3c7ac088fea1983d9ba40ea153e91c8a4c436db830e8894e80388"
N145_RESULT = ROOT / "reports/taiji_n1_v23_itemwise_error_attribution_20261019.json"
N145_RESULT_SHA256 = "07e4a18ffae7b2deb94f1bf6e49985839658f77188ecf940eae80e98da3883f1"
ATTEMPT = ROOT / "reports/taiji_n1_v23_confirmation_attempt_20261019.json"
PROGRESS = ROOT / "reports/taiji_n1_v23_confirmation_progress_20261019.jsonl"
OUTPUT = ROOT / "reports/taiji_n1_v23_output_source_attribution_20261019.json"
TEST = ROOT / "tests/taiji_native/test_n1_plan146_v23_output_source_attribution_contract.py"

CPU_LIMIT_SECONDS = 30.0
KINDS = ("lookup", "revision", "transition", "unknown")
DIRECTIONS = ("north", "east", "south", "west")
SEEDS = (11, 12, 13)


class Refuse(RuntimeError):
    """冻结输入或题内事实映射不符合合同。"""


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


def parse_material_facts(item: dict) -> list[dict]:
    facts = []
    for index, observation in enumerate(item.get("observations", [])):
        parts = observation.removesuffix(".").split()
        if len(parts) == 3 and parts[1] == "uses":
            facts.append(
                {"entity": parts[0], "value": parts[2], "role": "uses", "observation_index": index}
            )
        elif len(parts) == 4 and parts[1:3] == ["now", "uses"]:
            facts.append(
                {
                    "entity": parts[0],
                    "value": parts[3],
                    "role": "now_uses",
                    "observation_index": index,
                }
            )
    return facts


def _hamming(first: str, second: str) -> int | None:
    if len(first) != len(second):
        return None
    return sum(left != right for left, right in zip(first, second, strict=True))


def _decode_output(output_hex: str) -> tuple[str | None, str | None]:
    try:
        raw = bytes.fromhex(output_hex)
    except (TypeError, ValueError):
        return None, "invalid_hex"
    try:
        return raw.decode("utf-8", errors="strict"), None
    except UnicodeDecodeError:
        return None, "invalid_utf8"


def classify_row(row: dict, item: dict) -> dict:
    output_text, decode_error = _decode_output(row.get("output_hex"))
    kind = item["kind"]
    target = item["entities"][0]
    exact = row["free_exact"] is True
    classification: dict = {
        "kind": kind,
        "free_exact": exact,
        "decoded_output": output_text,
        "decode_error": decode_error,
        "matched_entity": None,
        "matched_value": None,
        "hamming_distance_to_target": None,
        "output_class": None,
        "free_exact_text_consistent": None,
    }
    if decode_error:
        classification["output_class"] = decode_error
        return classification

    facts = parse_material_facts(item)
    if kind in ("lookup", "revision"):
        if output_text == item["answer"]:
            classification.update(
                {
                    "output_class": "gold_answer_text",
                    "matched_entity": target,
                    "matched_value": item["answer"],
                    "hamming_distance_to_target": 0,
                    "free_exact_text_consistent": exact,
                }
            )
            return classification
        matches = [fact for fact in facts if fact["value"] == output_text]
        if len(matches) > 1:
            classification["output_class"] = "ambiguous_multiple_fact_values"
            return classification
        if len(matches) == 1:
            match = matches[0]
            distance = _hamming(target, match["entity"])
            classification.update(
                {
                    "matched_entity": match["entity"],
                    "matched_value": match["value"],
                    "hamming_distance_to_target": distance,
                }
            )
            if kind == "revision" and match["entity"] == target and match["role"] == "uses":
                classification["output_class"] = "target_entity_stale_value"
            elif match["entity"] == target:
                classification["output_class"] = "target_entity_non_gold_value"
            else:
                classification["output_class"] = "other_entity_material_value"
            return classification
        classification["output_class"] = "unmatched_material_value"
        return classification

    if kind == "transition":
        answer = item.get("answer_direction")
        turn = item.get("turn_template")
        initial = None
        if answer in DIRECTIONS and turn in ("left", "right"):
            delta = 1 if turn == "right" else -1
            initial = DIRECTIONS[(DIRECTIONS.index(answer) - delta) % len(DIRECTIONS)]
        if output_text == answer:
            classification["output_class"] = "gold_direction_text"
        elif output_text == initial:
            classification["output_class"] = "initial_direction"
        elif output_text in DIRECTIONS:
            classification["output_class"] = "other_valid_direction"
        else:
            classification["output_class"] = "non_direction_output"
        classification["free_exact_text_consistent"] = (output_text == answer) == exact
        return classification

    if kind == "unknown":
        classification["output_class"] = (
            "unknown_literal" if output_text == "unknown" else "other_abstention_output"
        )
        classification["free_exact_text_consistent"] = (output_text == "unknown") == exact
        return classification

    raise Refuse(f"unexpected task kind: {kind}")


def expected_endpoints() -> dict[str, tuple[str, str, int]]:
    endpoints = {}
    for seed in SEEDS:
        endpoints[f"N1-132/seed{seed}/parent"] = ("N1-132", "parent", seed)
        endpoints[f"N1-140/seed{seed}/block_candidate"] = ("N1-140", "candidate", seed)
        endpoints[f"N1-141/seed{seed}/candidate"] = ("N1-141", "candidate", seed)
        endpoints[f"N1-141/seed{seed}/control"] = ("N1-141", "control", seed)
    return endpoints


def validate_and_classify(rows: list[dict], face: dict) -> list[dict]:
    episodes = [item for item in face.get("episodes", []) if item.get("subset") == "confirm"]
    require(
        face.get("schema") == "n1-v23-confirmation-face-v1"
        and face.get("seed") == 123
        and face.get("plan_sha256") == N144_PLAN_SHA256
        and len(episodes) == 80,
        "v23 confirm face does not match frozen N1-144 contract",
    )
    face_by_id = {str(item.get("id")): item for item in episodes}
    require(len(face_by_id) == 80, "v23 confirm face IDs are not unique")
    endpoint_meta = expected_endpoints()
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        endpoint_id = str(row.get("endpoint_id"))
        require(
            endpoint_id in endpoint_meta and row.get("subset") == "confirm",
            f"unexpected query row: {endpoint_id}",
        )
        require(
            type(row.get("free_exact")) is bool,
            f"missing frozen score: {endpoint_id}/{row.get('id')}",
        )
        require(row.get("id") in face_by_id, f"query row has no face item: {row.get('id')}")
        item = face_by_id[row["id"]]
        require(row.get("kind") == item.get("kind"), f"row/face kind mismatch: {row.get('id')}")
        source, branch, seed = endpoint_meta[endpoint_id]
        require(
            (row.get("source"), row.get("branch"), row.get("seed")) == (source, branch, seed),
            f"endpoint metadata mismatch: {endpoint_id}",
        )
        grouped[endpoint_id].append(row)
    require(
        len(rows) == 960 and set(grouped) == set(endpoint_meta),
        "input must contain 960 rows for all 12 endpoints",
    )

    shared_ids: set[str] | None = None
    classified = []
    for endpoint_id, selected in sorted(grouped.items()):
        ids = [str(row["id"]) for row in selected]
        require(
            len(selected) == 80 and len(set(ids)) == 80,
            f"endpoint rows incomplete or duplicated: {endpoint_id}",
        )
        require(
            all(Counter(row["kind"] for row in selected)[kind] == 20 for kind in KINDS),
            f"class balance differs: {endpoint_id}",
        )
        current_ids = set(ids)
        require(
            shared_ids is None or current_ids == shared_ids,
            "all endpoints must use the same confirmation items",
        )
        shared_ids = current_ids
        for row in selected:
            item = face_by_id[row["id"]]
            output_hex = row.get("output_hex")
            require(
                isinstance(output_hex, str), f"output_hex is missing: {endpoint_id}/{row['id']}"
            )
            outcome = classify_row(row, item)
            classified.append(
                {
                    "endpoint_id": endpoint_id,
                    "source": row["source"],
                    "branch": row["branch"],
                    "seed": row["seed"],
                    "item_id": row["id"],
                    **outcome,
                }
            )
    require(set(face_by_id) == (shared_ids or set()), "face item set differs from queried rows")
    return classified


def summarize(classified: list[dict]) -> dict:
    by_endpoint_and_kind: dict[str, dict] = {}
    counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    by_kind: dict[str, Counter] = {kind: Counter() for kind in KINDS}
    consistency_mismatches = 0
    for row in classified:
        endpoint_id = row["endpoint_id"]
        kind = row["kind"]
        category = row["output_class"]
        counts[(endpoint_id, kind)][category] += 1
        by_kind[kind][category] += 1
        if row["free_exact_text_consistent"] is False:
            consistency_mismatches += 1
    for endpoint_id in sorted({row["endpoint_id"] for row in classified}):
        by_endpoint_and_kind[endpoint_id] = {
            kind: {
                "items": 20,
                "output_classes": dict(sorted(counts[(endpoint_id, kind)].items())),
            }
            for kind in KINDS
        }
    return {
        "by_endpoint_and_kind": by_endpoint_and_kind,
        "by_kind": {kind: dict(sorted(counter.items())) for kind, counter in by_kind.items()},
        "free_exact_text_consistency_mismatches": consistency_mismatches,
    }


def validate_frozen_inputs() -> tuple[dict, dict, dict, dict[str, str]]:
    for path, expected in (
        (PLAN, PLAN_SHA256),
        (N144_PLAN, N144_PLAN_SHA256),
        (N145_PLAN, N145_PLAN_SHA256),
        (FACE, FACE_SHA256),
        (N144_RESULT, N144_RESULT_SHA256),
        (N145_RESULT, N145_RESULT_SHA256),
    ):
        require(path.is_file() and sha256(path) == expected, f"frozen input hash mismatch: {path}")
    require(not OUTPUT.exists(), f"write-once output already exists: {OUTPUT}")
    n144 = read_json(N144_RESULT)
    n145 = read_json(N145_RESULT)
    face = read_json(FACE)
    require(
        n144.get("status") == "confirmation_failed"
        and n144.get("complete") is True
        and n144.get("confirmation_queries") == 960
        and n144.get("source_hashes_unchanged") is True
        and n144.get("can_promote") is False,
        "N1-144 result is not the complete frozen confirmation result",
    )
    require(
        n145.get("status") == "complete"
        and n145.get("complete") is True
        and n145.get("input_confirmation_rows") == 960
        and n145.get("recomputed_confirmation_summary_matches") is True
        and n145.get("can_promote") is False,
        "N1-145 offline attribution is incomplete or bound to another result",
    )
    require(
        n144.get("face_sha256") == FACE_SHA256 and n145.get("face_sha256") == FACE_SHA256,
        "face hash binding differs",
    )
    require(
        n145.get("n144_result_sha256") == N144_RESULT_SHA256,
        "N1-145 attribution binds another N1-144 result",
    )
    attempt_sha = sha256(ATTEMPT)
    progress_sha = sha256(PROGRESS)
    require(
        n144.get("attempt_sha256") == attempt_sha and n144.get("progress_sha256") == progress_sha,
        "N1-144 attempt/progress hash mismatch",
    )
    input_hashes = {
        str(path.resolve()): sha256(path)
        for path in (
            PLAN,
            N144_PLAN,
            N145_PLAN,
            FACE,
            N144_RESULT,
            N145_RESULT,
            ATTEMPT,
            PROGRESS,
            Path(__file__).resolve(),
            TEST.resolve(),
        )
    }
    return n144, n145, face, input_hashes


def analyze() -> dict:
    started = time.process_time()
    n144, n145, face, input_hashes = validate_frozen_inputs()
    rows = n144["confirmation_rows"]
    classified = validate_and_classify(rows, face)
    summary = summarize(classified)
    require(len(classified) == 960, "classified output row count differs")
    require(
        all(sha256(Path(path)) == digest for path, digest in input_hashes.items()),
        "analysis input changed during run",
    )
    cpu_seconds = time.process_time() - started
    require(
        cpu_seconds <= CPU_LIMIT_SECONDS, f"PLAN-N1-146 CPU ceiling exceeded: {cpu_seconds:.6f}s"
    )
    return {
        "schema": "n1-plan146-v23-output-source-attribution-v1",
        "status": "complete",
        "complete": True,
        "plan_sha256": PLAN_SHA256,
        "n144_plan_sha256": N144_PLAN_SHA256,
        "n145_plan_sha256": N145_PLAN_SHA256,
        "face_sha256": FACE_SHA256,
        "n144_result_sha256": N144_RESULT_SHA256,
        "n145_result_sha256": N145_RESULT_SHA256,
        "attempt_sha256": n144["attempt_sha256"],
        "progress_sha256": n144["progress_sha256"],
        "analysis_source_hashes": input_hashes,
        "analysis_source_hashes_unchanged": True,
        "classified_rows": len(classified),
        "endpoint_count": 12,
        "items_per_endpoint": 80,
        "summary": summary,
        "row_classifications": classified,
        "cpu_seconds": round(cpu_seconds, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "model_loads": 0,
        "checkpoint_loads": 0,
        "model_queries": 0,
        "parameter_updates": 0,
        "dev_queries": 0,
        "v3_holdout_queries": 0,
        "causal_claim": False,
        "capability_claim": False,
        "can_promote": False,
        "summary_zh": "N1-144已消费输出完成严格UTF-8解码及题内事实值匹配分类；仅为描述性post-hoc分析，未重评分、查询或训练，不作因果或能力主张。",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PLAN-N1-146 read-only v23 output source attribution"
    )
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    if not args.analyze:
        parser.error("one-shot invocation requires --analyze")
    result = analyze()
    write_new(OUTPUT, result)
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "status",
                    "complete",
                    "classified_rows",
                    "cpu_seconds",
                    "model_queries",
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
