"""对PLAN-N1-161已消费输出做零查询逐题错误类型归因。"""

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

from scripts.training import build_taiji_n1_plan153_v28_fresh_context_length_face as n153

PLAN = ROOT / "plans/reference/PLAN-N1-161A_question_surface_itemwise_attribution_20261020.md"
PLAN_SHA256 = "09ab12b712a3d441ddac753a9823dee2ef02286ebad6d6987f758ad5ab7115e4"
RESULT = ROOT / "reports/taiji_n1_plan161_question_surface_result_20261020.json"
RESULT_SHA256 = "2808a969299143c692643bf4bd5ccabdd67a63021f5ad9ba0cab65bcb51a385a"
FACE = ROOT / "plans/manifests/n1_plan161_v33_question_surface_seed131_20261020.json"
FACE_SHA256 = "39ce277831a0461a45bf7fb0f793d57f89ade84e988d1ba830703b07346393ef"
FACE_AUDIT = ROOT / "reports/taiji_n1_plan161_question_surface_face_audit_20261020.json"
FACE_AUDIT_SHA256 = "cc08a8ec7af299207c8f474db967b8d0eda76aca2fc8be2480b06fe41e2b3b46"
PROGRESS = ROOT / "reports/taiji_n1_plan161_question_surface_progress_20261020.jsonl"
PROGRESS_SHA256 = "dd80604cd66519f04bfe48d5929756e0f2f5436a68435d3a5df7a31068f66cce"
REPORT = ROOT / "reports/taiji_n1_plan161_question_surface_itemwise_attribution_20261020.json"
TEST = ROOT / "tests/taiji_native/test_n1_plan161_question_surface_itemwise_attribution_contract.py"
SEEDS = (11, 12, 13)
BRANCHES = ("unknown8_rebalanced", "length_mixed_parent")
LENGTHS = (8, 16, 32)
KINDS = ("lookup", "revision", "transition", "unknown")
TEMPLATES = (
    "material_active",
    "material_passive",
    "direction_after_turning",
    "direction_after_turn",
)
TOTAL_ROWS = 2_880
CPU_LIMIT_SECONDS = 30.0
UNKNOWN = n153.UNKNOWN
DIRECTIONS = frozenset(n153.DIRECTIONS)
TURNS = frozenset(n153.TURNS)


class Refuse(RuntimeError):
    """输入绑定或逐题归因契约错误。"""


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


def _material_values(observations: list[str]) -> list[str]:
    values = []
    for sentence in observations:
        parts = sentence.removesuffix(".").split()
        if len(parts) == 3 and parts[1] == "uses":
            values.append(parts[2])
        elif len(parts) == 4 and parts[1:3] == ["now", "uses"]:
            values.append(parts[3])
    return values


def output_class(row: dict, base: dict) -> str:
    raw = bytes.fromhex(row["raw_hex"])
    target = base["answer"].encode("utf-8")
    if raw == target and row.get("stop_reason") == "boundary":
        return "exact"
    text = raw.decode("utf-8", errors="replace").strip()
    if text == UNKNOWN:
        return "literal_unknown"
    observations = base["observations_by_length"][str(row["context_length"])]
    if base["kind"] == "revision":
        old_value = observations[0].removesuffix(".").split()[-1]
        if text == old_value:
            return "stale_revision_value"
    material_values = _material_values(observations)
    if text in material_values:
        return "other_observed_material"
    if text in DIRECTIONS or text in TURNS:
        return "direction_or_turn_token"
    if not text:
        return "empty_output"
    return "other_output"


def calculate() -> dict:
    started = time.process_time()
    require(sha256(PLAN) == PLAN_SHA256, "PLAN-N1-161A SHA changed")
    require(sha256(RESULT) == RESULT_SHA256, "N1-161 result SHA changed")
    require(sha256(FACE) == FACE_SHA256, "N1-161 face SHA changed")
    require(sha256(FACE_AUDIT) == FACE_AUDIT_SHA256, "N1-161 face audit SHA changed")
    require(sha256(PROGRESS) == PROGRESS_SHA256, "N1-161 progress SHA changed")
    result = read_json(RESULT)
    face = read_json(FACE)
    face_audit = read_json(FACE_AUDIT)
    progress_rows = [json.loads(line) for line in PROGRESS.read_text(encoding="utf-8").splitlines()]
    rows = result.get("fresh_diagnostic_rows", [])
    require(
        result.get("complete") is True
        and result.get("source_hashes_unchanged") is True
        and result.get("can_promote") is False
        and result.get("fresh_queries") == TOTAL_ROWS
        and len(rows) == TOTAL_ROWS
        and len(progress_rows) == TOTAL_ROWS
        and progress_rows == rows
        and face.get("plan_sha256") == result.get("plan_sha256")
        and face.get("seed") == 131
        and face.get("total_queries") == TOTAL_ROWS
        and face_audit.get("source_hashes_unchanged") is True
        and face_audit.get("status") == "complete"
        and face_audit.get("face_sha256") == FACE_SHA256,
        "N1-161 frozen result/face/progress is incomplete",
    )
    base_by_id = {item["parent_id"]: item for item in face["items"]}
    require(len(base_by_id) == 160, "N1-161 face parent IDs are not unique")
    seen = set()
    attributed = []
    summaries = defaultdict(Counter)
    paired = {}
    teacher_free_mismatches = 0
    for row in rows:
        parent_id = row.get("parent_id")
        require(parent_id in base_by_id, f"unmatched N1-161 parent_id: {parent_id}")
        base = base_by_id[parent_id]
        key = (
            int(row["seed"]),
            str(row["branch"]),
            parent_id,
            int(row["context_length"]),
            str(row["kind"]),
            str(row["question_template_id"]),
        )
        require(key not in seen, f"duplicate N1-161 row key: {key}")
        seen.add(key)
        require(
            row.get("question") == base.get("question")
            and row.get("kind") == base.get("kind")
            and row.get("question_template_id") == base.get("question_template_id"),
            f"N1-161 question/manifest association differs: {key}",
        )
        category = output_class(row, base)
        exact_recomputed = category == "exact"
        require(
            exact_recomputed is row.get("free_exact"), f"free_exact recomputation differs: {key}"
        )
        teacher_free_mismatches += row.get("teacher_exact") is not row.get("free_exact")
        summary_key = (
            str(row["branch"]),
            int(row["seed"]),
            str(row["kind"]),
            str(row["question_template_id"]),
            int(row["context_length"]),
        )
        summaries[summary_key][category] += 1
        pair_key = (
            int(row["seed"]),
            parent_id,
            int(row["context_length"]),
            str(row["kind"]),
            str(row["question_template_id"]),
        )
        paired.setdefault(pair_key, {})[str(row["branch"])] = exact_recomputed
        attributed.append(
            {
                "seed": int(row["seed"]),
                "branch": str(row["branch"]),
                "parent_id": parent_id,
                "context_length": int(row["context_length"]),
                "kind": str(row["kind"]),
                "question_template_id": str(row["question_template_id"]),
                "free_exact": exact_recomputed,
                "output_class": category,
                "raw_hex": row["raw_hex"],
            }
        )
    require(len(seen) == TOTAL_ROWS, "N1-161 attributed denominator differs")
    paired_summary = defaultdict(Counter)
    for key, branches in paired.items():
        require(set(branches) == set(BRANCHES), f"N1-161 paired branch missing: {key}")
        seed, _parent_id, length, kind, template_id = key
        counts = paired_summary[(kind, template_id, length)]
        candidate = branches["unknown8_rebalanced"]
        parent = branches["length_mixed_parent"]
        if candidate and not parent:
            counts["candidate_only"] += 1
        elif parent and not candidate:
            counts["parent_only"] += 1
        elif candidate:
            counts["both_correct"] += 1
        else:
            counts["both_wrong"] += 1
    cpu = time.process_time() - started
    require(cpu <= CPU_LIMIT_SECONDS, f"N1-161A CPU limit exceeded: {cpu:.6f}")
    return {
        "schema": "taiji-n1-plan161-question-surface-itemwise-attribution-v1",
        "status": "complete",
        "complete": True,
        "plan_sha256": PLAN_SHA256,
        "source_result_sha256": RESULT_SHA256,
        "source_face_sha256": FACE_SHA256,
        "source_face_audit_sha256": FACE_AUDIT_SHA256,
        "source_progress_sha256": PROGRESS_SHA256,
        "source_hashes_unchanged": True,
        "row_count": len(attributed),
        "pair_count": len(paired),
        "itemwise_output_classes": attributed,
        "summary_by_branch_seed_kind_template_length": [
            {
                "branch": key[0],
                "seed": key[1],
                "kind": key[2],
                "question_template_id": key[3],
                "context_length": key[4],
                "items": sum(counts.values()),
                "output_classes": dict(sorted(counts.items())),
            }
            for key, counts in sorted(summaries.items())
        ],
        "paired_correctness_by_kind_template_length": [
            {
                "kind": key[0],
                "question_template_id": key[1],
                "context_length": key[2],
                **dict(sorted(counts.items())),
            }
            for key, counts in sorted(paired_summary.items())
        ],
        "teacher_free_exact_mismatches": teacher_free_mismatches,
        "model_loads": 0,
        "checkpoint_loads": 0,
        "model_queries": 0,
        "parameter_updates": 0,
        "dev_queries": 0,
        "holdout_queries": 0,
        "confirmation_queries": 0,
        "cpu_seconds": round(cpu, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "can_promote": False,
        "summary_zh": (
            f"N1-161A完成{len(attributed)}行逐题输出分类和{len(paired)}组candidate/parent配对复算；"
            f"CPU{cpu:.3f}/{CPU_LIMIT_SECONDS:.0f}秒，0模型/checkpoint加载、0查询、0更新。"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="PLAN-N1-161A offline itemwise output attribution")
    parser.add_argument("--analyze", action="store_true", required=True)
    args = parser.parse_args()
    del args
    if REPORT.exists():
        raise FileExistsError("N1-161A report is write-once")
    started = time.process_time()
    try:
        report = calculate()
    except BaseException as exc:
        report = {
            "schema": "taiji-n1-plan161-question-surface-itemwise-attribution-v1",
            "status": "stopped",
            "complete": False,
            "plan_sha256": PLAN_SHA256,
            "source_result_sha256": RESULT_SHA256,
            "source_face_sha256": FACE_SHA256,
            "source_face_audit_sha256": FACE_AUDIT_SHA256,
            "source_progress_sha256": PROGRESS_SHA256,
            "errors": [f"{type(exc).__name__}: {exc}"],
            "model_loads": 0,
            "checkpoint_loads": 0,
            "model_queries": 0,
            "parameter_updates": 0,
            "dev_queries": 0,
            "holdout_queries": 0,
            "confirmation_queries": 0,
            "cpu_seconds": round(time.process_time() - started, 6),
            "cpu_limit_seconds": CPU_LIMIT_SECONDS,
            "can_promote": False,
            "summary_zh": "N1-161A离线归因停止；已写入失败回执，不执行重试。",
        }
    with REPORT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
    print(
        json.dumps(
            {
                k: report.get(k)
                for k in (
                    "status",
                    "complete",
                    "row_count",
                    "pair_count",
                    "teacher_free_exact_mismatches",
                    "cpu_seconds",
                    "model_queries",
                    "can_promote",
                    "summary_zh",
                )
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0 if report.get("complete") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
