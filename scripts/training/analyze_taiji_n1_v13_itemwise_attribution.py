"""Offline itemwise attribution for the already-consumed PLAN-N1-123 rows."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
import sys as _sys

_sys.path.insert(0, str(ROOT))

from scripts.training import seal_ast  # noqa: E402

PLAN = ROOT / "plans/reference/PLAN-N1-123_v13_long_context_range_probe_draft_20261019.md"
SOURCE_EXCLUSIONS = ROOT / "plans/manifests/n1_v13_after_n122_source_exclusions_20261019.json"
READINESS = ROOT / "reports/taiji_n1_v13_after_n122_confirmation_readiness_20261019.json"
FACE = ROOT / "plans/manifests/n1_hard_neighbor_v13_confirmation_seed113_20261019.json"
FACE_CHECKS = ROOT / "reports/taiji_n1_v13_after_n122_face_checks_20261019.json"
RANGE_REPORT = ROOT / "reports/taiji_n1_v13_after_n122_range_20261019.json"
CONFIRM_REPORT = ROOT / "reports/taiji_n1_v13_after_n122_confirmation_20261019.json"
OUTPUT = ROOT / "reports/taiji_n1_v13_after_n122_itemwise_attribution_20261019.json"
SUPERSEDED_DRAFT_SHA256 = "fb4698203df0ad12e706f630986cdd9ac1f5c239164933818e416cbe7ba76364"
ARMS = ("position", "cut", "terminal")
COHORTS = ("baseline", "candidate")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _target_positions(item: dict) -> list[int]:
    subject = item["entities"][0]
    if item["kind"] == "lookup":
        prefixes = (subject + " uses ",)
    elif item["kind"] == "revision":
        prefixes = (subject + " uses ", subject + " now uses ")
    elif item["kind"] == "transition":
        prefixes = (subject + " faces ", subject + " turns ")
    else:
        return []
    return [index for index, row in enumerate(item["observations"]) if row.startswith(prefixes)]


def _paired(left: dict[str, bool], right: dict[str, bool]) -> dict:
    if set(left) != set(right):
        raise ValueError("paired arm item identities differ")
    counts = Counter((left[item_id], right[item_id]) for item_id in left)
    return {
        "both_correct": counts[(True, True)],
        "left_only_correct": counts[(True, False)],
        "right_only_correct": counts[(False, True)],
        "both_wrong": counts[(False, False)],
        "left_minus_right": sum(left.values()) - sum(right.values()),
    }


def build_report() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"one-shot attribution report already exists: {OUTPUT}")
    started = time.process_time()
    face = read_json(FACE)
    checks = read_json(FACE_CHECKS)
    range_report = read_json(RANGE_REPORT)
    confirmation = read_json(CONFIRM_REPORT)
    if (
        face.get("schema") != "n1-hard-neighbor-v13-confirmation-v1"
        or face.get("seed") != 113
        or face.get("observations_per_item") != 16
        or len(face.get("episodes", [])) != 40
        or checks.get("accepted") is not True
        or not seal_ast.receipt_binding_matches(checks.get("face_sha256"), FACE)
        or range_report.get("complete") is not True
        or range_report.get("range_gate", {}).get("range_check_passed") is not True
        or range_report.get("model_queries") != 180
        or not seal_ast.receipt_binding_matches(range_report.get("face_sha256"), FACE)
        or range_report.get("readiness_sha256") != sha256(READINESS)
        or confirmation.get("complete") is not True
        or confirmation.get("confirmation_queries") != 360
        or confirmation.get("total_model_queries") != 540
        or confirmation.get("source_hashes_unchanged") is not True
        or confirmation.get("can_promote") is not False
        or not seal_ast.receipt_binding_matches(confirmation.get("face_sha256"), FACE)
        or confirmation.get("face_checks_sha256") != sha256(FACE_CHECKS)
        or confirmation.get("range_report_sha256") != sha256(RANGE_REPORT)
        or confirmation.get("readiness_sha256") != sha256(READINESS)
    ):
        raise ValueError("PLAN-N1-123 consumed evidence is incomplete or inconsistent")

    confirm_ids = set(face["subsets"]["confirm"])
    confirm_items = {item["id"]: item for item in face["episodes"] if item["id"] in confirm_ids}
    if len(confirm_items) != 20:
        raise ValueError("exactly twenty frozen confirmation items are required")
    rows_by_arm: dict[tuple[str, int, str], dict[str, dict]] = {}
    for cohort in COHORTS:
        rows = confirmation[f"{cohort}_rows"]
        if len(rows) != 180:
            raise ValueError(f"{cohort} must contain 180 frozen confirm rows")
        for seed in (11, 12, 13):
            for arm in ARMS:
                selected = [
                    row for row in rows if row.get("seed") == seed and row.get("arm") == arm
                ]
                if len(selected) != 20:
                    raise ValueError(f"{cohort}/{seed}/{arm} must have twenty confirm rows")
                key = (cohort, seed, arm)
                row_map = {row["id"]: row for row in selected}
                if len(row_map) != 20 or set(row_map) != confirm_ids:
                    raise ValueError(
                        f"{cohort}/{seed}/{arm} does not map to the frozen confirm face"
                    )
                rows_by_arm[key] = row_map

    gate_counts = {}
    for cohort in COHORTS:
        gate_counts[cohort] = {}
        for seed in (11, 12, 13):
            gate_counts[cohort][str(seed)] = {}
            for arm in ARMS:
                row_map = rows_by_arm[(cohort, seed, arm)]
                gate_counts[cohort][str(seed)][arm] = {
                    kind: {
                        "items": sum(confirm_items[item_id]["kind"] == kind for item_id in row_map),
                        "free_exact": sum(
                            confirm_items[item_id]["kind"] == kind and row["free_exact"] is True
                            for item_id, row in row_map.items()
                        ),
                    }
                    for kind in ("lookup", "revision", "transition", "unknown")
                }

    paired = {}
    for cohort in COHORTS:
        paired[cohort] = {}
        for seed in (11, 12, 13):
            by_seed = {}
            for left_arm, right_arm in (("position", "cut"), ("position", "terminal")):
                left = {
                    item_id: row["free_exact"] is True
                    for item_id, row in rows_by_arm[(cohort, seed, left_arm)].items()
                }
                right = {
                    item_id: row["free_exact"] is True
                    for item_id, row in rows_by_arm[(cohort, seed, right_arm)].items()
                }
                by_seed[f"{left_arm}_vs_{right_arm}"] = _paired(left, right)
            by_seed["position_vs_cut_by_kind"] = {}
            for kind in ("lookup", "revision", "transition", "unknown"):
                ids = [item_id for item_id, item in confirm_items.items() if item["kind"] == kind]
                by_seed["position_vs_cut_by_kind"][kind] = _paired(
                    {
                        item_id: rows_by_arm[(cohort, seed, "position")][item_id]["free_exact"]
                        is True
                        for item_id in ids
                    },
                    {
                        item_id: rows_by_arm[(cohort, seed, "cut")][item_id]["free_exact"] is True
                        for item_id in ids
                    },
                )
            paired[cohort][str(seed)] = by_seed

    position_features = {}
    for item_id, item in confirm_items.items():
        positions = _target_positions(item)
        expected_count = (
            0
            if item["kind"] == "unknown"
            else 2 if item["kind"] in ("revision", "transition") else 1
        )
        if len(positions) != expected_count:
            raise ValueError(f"target observation positions are malformed for {item_id}")
        position_features[item_id] = {
            "kind": item["kind"],
            "target_observation_indices": positions,
            "first_target_index": positions[0] if positions else None,
            "answer_length": len(item["answer"]),
        }

    position_by_context = {}
    candidate_position = {
        item_id: row["free_exact"] is True
        for item_id, row in rows_by_arm[("candidate", 11, "position")].items()
    }
    # Report coverage by the first evidence-row index; do not treat this as a held-out claim.
    for kind in ("lookup", "revision", "transition", "unknown"):
        buckets = defaultdict(list)
        for item_id, feature in position_features.items():
            if feature["kind"] == kind:
                buckets[feature["first_target_index"]].append(candidate_position[item_id])
        position_by_context[kind] = {
            "target_start_indices": sorted(
                {
                    feature["first_target_index"]
                    for feature in position_features.values()
                    if feature["kind"] == kind and feature["first_target_index"] is not None
                }
            ),
            "target_evidence_row_indices": sorted(
                {
                    position
                    for feature in position_features.values()
                    if feature["kind"] == kind
                    for position in feature["target_observation_indices"]
                }
            ),
            "by_first_target_index": {
                str(index) if index is not None else "no_target": {
                    "items": len(values),
                    "seed11_candidate_position_exact": sum(values),
                }
                for index, values in sorted(
                    buckets.items(), key=lambda pair: (-1 if pair[0] is None else pair[0])
                )
            },
        }

    paths = (
        PLAN,
        SOURCE_EXCLUSIONS,
        READINESS,
        FACE,
        FACE_CHECKS,
        RANGE_REPORT,
        CONFIRM_REPORT,
        Path(__file__).resolve(),
    )
    source_hashes = {str(path.resolve()): sha256(path) for path in paths}
    total_cpu = time.process_time() - started
    report = {
        "schema": "n1-plan123-v13-confirmation-itemwise-attribution-v1",
        "status": "complete",
        "supersedes_prepublication_draft_sha256": SUPERSEDED_DRAFT_SHA256,
        "source_hashes": source_hashes,
        "face_sha256": sha256(FACE),
        "range_report_sha256": sha256(RANGE_REPORT),
        "confirmation_report_sha256": sha256(CONFIRM_REPORT),
        "confirm_rows_per_cohort": 180,
        "parameter_updates": 0,
        "model_loads": 0,
        "model_queries": 0,
        "dev_queries": 0,
        "holdout_queries": 0,
        "can_promote": False,
        "free_exact_by_seed_kind": gate_counts,
        "paired_correctness": paired,
        "target_position_coverage": {
            "candidate_seed11_position_arm": position_by_context,
            "all_non_unknown_items_have_target_evidence": True,
            "unknown_has_no_target_evidence": True,
        },
        "interpretation": [
            "The candidate position arm leads terminal by 23/60 but cut by only 5/60; the preregistered +12/60 cut margin fails.",
            "Position/cut deficits are concentrated in lookup and transition: their candidate exact counts are 7/15 and 5/15 for position versus 4/15 and 3/15 for cut; revision is 15/15 in both, and unknown is 14/15 in both.",
            "The confirm face starts target evidence only at indices 0-4; two-row revision/transition evidence therefore occupies indices 0-5 (unknown has no target row), despite sixteen observation rows. This tests distractor interference after early evidence, not retrieval from evidence placed throughout the full context.",
            "These are paired descriptive results on the consumed face; they do not estimate an observation-count causal effect or generalization to new items.",
        ],
        "cpu_seconds": round(total_cpu, 6),
        "can_promote_final": False,
    }
    with OUTPUT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
    return report


if __name__ == "__main__":
    result = build_report()
    print(
        json.dumps(
            {key: result[key] for key in ("schema", "status", "cpu_seconds", "can_promote")},
            sort_keys=True,
        )
    )
