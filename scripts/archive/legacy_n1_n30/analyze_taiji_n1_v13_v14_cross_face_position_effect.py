"""Compare already-consumed v13/v14 position-vs-cut rows without model access."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FACES = {
    "v13": ROOT / "plans/manifests/n1_hard_neighbor_v13_confirmation_seed113_20261019.json",
    "v14": ROOT / "plans/manifests/n1_hard_neighbor_v14_confirmation_seed114_20261019.json",
}
CONFIRMATIONS = {
    "v13": ROOT / "reports/taiji_n1_v13_after_n122_confirmation_20261019.json",
    "v14": ROOT / "reports/taiji_n1_v14_after_n123_confirmation_20261019.json",
}
RANGES = {
    "v13": ROOT / "reports/taiji_n1_v13_after_n122_range_20261019.json",
    "v14": ROOT / "reports/taiji_n1_v14_after_n123_range_20261019.json",
}
ATTRIBUTIONS = {
    "v13": ROOT / "reports/taiji_n1_v13_after_n122_itemwise_attribution_20261019.json",
    "v14": ROOT / "reports/taiji_n1_v14_after_n123_itemwise_attribution_v2_20261019.json",
}
PLANS = {
    "v13": ROOT / "plans/reference/PLAN-N1-123_v13_long_context_range_probe_draft_20261019.md",
    "v14": ROOT
    / "plans/reference/PLAN-N1-124_v14_full_context_evidence_position_probe_draft_20261019.md",
}
OUTPUT = ROOT / "reports/taiji_n1_v13_v14_cross_face_position_effect_v2_20261019.json"
ARMS = ("position", "cut", "terminal")
KINDS = ("lookup", "revision", "transition", "unknown")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def paired_delta(left: dict[str, bool], right: dict[str, bool]) -> dict:
    if set(left) != set(right):
        raise ValueError("paired item identities differ")
    cells = Counter((left[item_id], right[item_id]) for item_id in left)
    return {
        "items": len(left),
        "position_exact": sum(left.values()),
        "cut_exact": sum(right.values()),
        "position_minus_cut": sum(left.values()) - sum(right.values()),
        "position_only_correct": cells[(True, False)],
        "cut_only_correct": cells[(False, True)],
        "both_correct": cells[(True, True)],
        "both_wrong": cells[(False, False)],
    }


def target_start(item: dict) -> int | None:
    subject = item["entities"][0]
    if item["kind"] == "lookup":
        prefixes = (subject + " uses ",)
    elif item["kind"] == "revision":
        prefixes = (subject + " uses ", subject + " now uses ")
    elif item["kind"] == "transition":
        prefixes = (subject + " faces ", subject + " turns ")
    else:
        return None
    positions = [
        index for index, row in enumerate(item["observations"]) if row.startswith(prefixes)
    ]
    if not positions:
        raise ValueError(f"target evidence is missing for {item['id']}")
    return positions[0]


def build_report() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"cross-face report already exists: {OUTPUT}")
    started = time.process_time()
    source_paths = []
    faces = {}
    reports = {}
    ranges = {}
    for label in ("v13", "v14"):
        face = read_json(FACES[label])
        confirmation = read_json(CONFIRMATIONS[label])
        range_report = read_json(RANGES[label])
        attribution = read_json(ATTRIBUTIONS[label])
        if (
            face.get("seed") != (113 if label == "v13" else 114)
            or face.get("observations_per_item") != 16
            or len(face.get("episodes", [])) != 40
            or confirmation.get("complete") is not True
            or confirmation.get("confirmation_queries") != 360
            or confirmation.get("total_model_queries") != 540
            or confirmation.get("source_hashes_unchanged") is not True
            or range_report.get("complete") is not True
            or range_report.get("range_gate", {}).get("range_check_passed") is not True
            or attribution.get("status") != "complete"
        ):
            raise ValueError(f"{label} consumed evidence is incomplete")
        faces[label] = face
        reports[label] = confirmation
        ranges[label] = range_report
        source_paths.extend(
            (FACES[label], CONFIRMATIONS[label], RANGES[label], ATTRIBUTIONS[label], PLANS[label])
        )

    summaries = {}
    position_by_start = {}
    for label in ("v13", "v14"):
        face = faces[label]
        confirmation = reports[label]
        confirm_ids = set(face["subsets"]["confirm"])
        items = {item["id"]: item for item in face["episodes"] if item["id"] in confirm_ids}
        rows_by_seed_arm = {}
        for seed in (11, 12, 13):
            for arm in ARMS:
                rows = [
                    row
                    for row in confirmation["candidate_rows"]
                    if row.get("seed") == seed and row.get("arm") == arm
                ]
                if len(rows) != 20 or {row["id"] for row in rows} != confirm_ids:
                    raise ValueError(f"{label}/candidate/{seed}/{arm} row mapping is incomplete")
                rows_by_seed_arm[seed, arm] = {row["id"]: row for row in rows}
        aggregate = {}
        for arm in ARMS:
            aggregate[arm] = {
                kind: sum(
                    items[item_id]["kind"] == kind and row["free_exact"] is True
                    for seed in (11, 12, 13)
                    for item_id, row in rows_by_seed_arm[seed, arm].items()
                )
                for kind in KINDS
            }
            aggregate[arm]["total"] = sum(aggregate[arm].values())
        # Pair each item within each seed, then pool the descriptive cells by class.
        pooled_by_kind = {}
        for kind in KINDS:
            left = {}
            right = {}
            for seed in (11, 12, 13):
                for item_id, item in items.items():
                    if item["kind"] == kind:
                        left[f"{seed}:{item_id}"] = (
                            rows_by_seed_arm[seed, "position"][item_id]["free_exact"] is True
                        )
                        right[f"{seed}:{item_id}"] = (
                            rows_by_seed_arm[seed, "cut"][item_id]["free_exact"] is True
                        )
            pooled_by_kind[kind] = paired_delta(left, right)

        starts = defaultdict(lambda: {"position": {}, "cut": {}})
        for item_id, item in items.items():
            start = target_start(item)
            key = str(start) if start is not None else "no_target"
            for arm in ("position", "cut"):
                starts[key][arm][item_id] = {
                    seed: rows_by_seed_arm[seed, arm][item_id]["free_exact"] is True
                    for seed in (11, 12, 13)
                }
        position_by_start[label] = {
            key: {
                "items": len(values["position"]),
                "position_exact": sum(
                    value
                    for seed_values in values["position"].values()
                    for value in seed_values.values()
                ),
                "cut_exact": sum(
                    value
                    for seed_values in values["cut"].values()
                    for value in seed_values.values()
                ),
                "position_minus_cut": sum(
                    value
                    for seed_values in values["position"].values()
                    for value in seed_values.values()
                )
                - sum(
                    value
                    for seed_values in values["cut"].values()
                    for value in seed_values.values()
                ),
            }
            for key, values in sorted(starts.items())
        }
        summaries[label] = {
            "range_usable_endpoints": ranges[label]["range_gate"]["usable_endpoints"],
            "candidate_exact_by_arm": aggregate,
            "candidate_position_vs_cut_by_kind": pooled_by_kind,
        }

    source_hashes = {str(path.resolve()): sha256(path) for path in source_paths}
    source_hashes[str(Path(__file__).resolve())] = sha256(Path(__file__).resolve())
    result = {
        "schema": "n1-v13-v14-cross-face-position-effect-v1",
        "status": "complete",
        "source_hashes": source_hashes,
        "model_loads": 0,
        "model_queries": 0,
        "parameter_updates": 0,
        "dev_queries": 0,
        "holdout_queries": 0,
        "can_promote": False,
        "face_summaries": summaries,
        "candidate_position_minus_cut": {
            label: summaries[label]["candidate_exact_by_arm"]["position"]["total"]
            - summaries[label]["candidate_exact_by_arm"]["cut"]["total"]
            for label in ("v13", "v14")
        },
        "position_by_target_start": position_by_start,
        "interpretation": [
            "The v13 early-evidence face and v14 distributed-evidence face are different sampled items; their differences are descriptive and cannot estimate a causal observation-position effect.",
            "Both faces fail the preregistered position-over-cut gate: the candidate position-minus-cut totals are +5/60 in v13 and +1/60 in v14.",
            "Neither sampled face meets the position-over-cut margin. These observational results do not identify a failed mechanism or exclude a placement effect; audit the existing reader and saved outputs before selecting a revision.",
        ],
        "cpu_seconds": round(time.process_time() - started, 6),
    }
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    result = build_report()
    print(
        json.dumps({key: result[key] for key in ("schema", "status", "cpu_seconds", "can_promote")})
    )
