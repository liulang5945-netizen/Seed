"""Read-only itemwise comparison of sealed N1-117 and N1-118 position shards."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

N117_REPORT = ROOT / "reports/taiji_n1_coupled_rwl_length_balanced_r1_20261018.json"
N117_REPORT_SHA256 = "1d4f43519790a1e36cb7dfbc580506d697815dccbdecb95fd576ec3a7bc2e391"
N118_REPORT = ROOT / "reports/taiji_n1_normalized_coupled_rwl_length_balanced_20261018.json"
N118_REPORT_SHA256 = "5ceabffadc7d62e2aa7be3996847c92713fe9cce604c0f10e463ad97fbe8217e"
N118_REVALIDATION = (
    ROOT
    / "reports/taiji_n1_normalized_coupled_rwl_length_balanced_20261018_receipt_revalidation.json"
)
N118_REVALIDATION_SHA256 = "7fa1007bda4d9099b0778789057fb59eaf5d72621787a5f9d182db30650cbd24"
OUTPUT = ROOT / "reports/taiji_n1_n117_n118_position_itemwise_attribution_20261019.json"
CPU_LIMIT_SECONDS = 60
MILESTONES = (0, 6144, 12288, 18432, 24576)
CONDITIONS = ("original", "direction_opposite", "turn_flip", "both")
VIEWS = ("original", "rename")
RWL_GROUPS = ("reader", "writer", "local")
EXPECTED_SHARDS = {
    ("n117", 11): "9aae12b5d707b664c6776d3fabc182b15599e99ca6506a9fc9ef1d615041a2ff",
    ("n117", 12): "5ce2fd970ba0da2f26d77235f197e6c68d6b9f3fc537d3789c0fdfa3e5a4fecb",
    ("n117", 13): "58481413c35da22b6815269e05eca51a68c36a8159a3dccac406550cc9ab8cfa",
    ("n118", 11): "231370ccfaeac1b4753692c19f293344173d3a780dd3b6f908f0d1dfb440932f",
    ("n118", 12): "b7e7504ecc71c580018276ce0b122b275e713bcfcefa9f0e399a6f30fbda3cf3",
    ("n118", 13): "c45c3ccbec502d88c026ddea3fabcec8032a273d12ec8f1f6fee6cd5210481be",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_source(path: Path, expected_sha256: str) -> dict:
    if sha256(path) != expected_sha256:
        raise ValueError(f"frozen source changed: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_position_shards(label: str, report: dict) -> dict[int, dict]:
    by_seed = {}
    models = [model for model in report["models"] if model.get("arm") == "position"]
    if len(models) != 3 or {model.get("seed") for model in models} != {11, 12, 13}:
        raise ValueError(f"three frozen {label} position endpoints required")
    for model in models:
        seed = model["seed"]
        seal = model["shard"]
        expected_sha = EXPECTED_SHARDS[label, seed]
        path = Path(seal["path"])
        if seal.get("compressed_sha256") != expected_sha or sha256(path) != expected_sha:
            raise ValueError(f"frozen {label} seed{seed} shard changed: {path}")
        envelope = json.loads(gzip.open(path, "rt", encoding="utf-8").read())
        record = envelope["record"]
        if (
            envelope.get("schema") != "n1-lossless-training-evidence-shard-v1"
            or envelope.get("identity", {}).get("seed") != seed
            or envelope.get("identity", {}).get("arm") != "position"
            or record.get("completed_updates") != 24576
            or [item.get("completed_updates") for item in record.get("milestones", [])]
            != list(MILESTONES)
        ):
            raise ValueError(f"incomplete {label} seed{seed} position shard")
        by_seed[seed] = record
    return by_seed


def summarize_factorial_rows(record: dict) -> dict:
    result = {}
    for milestone in MILESTONES:
        views = {}
        for view in VIEWS:
            evaluation_set = f"factorial_{view}"
            rows = [
                row
                for row in record["rows"]
                if row.get("completed_updates") == milestone
                and row.get("evaluation_set") == evaluation_set
            ]
            parent_conditions = defaultdict(dict)
            for row in rows:
                parent = row["parent_id"]
                condition = row["condition"]
                if (
                    row.get("kind") != "transition"
                    or condition not in CONDITIONS
                    or condition in parent_conditions[parent]
                ):
                    raise ValueError(f"invalid or duplicate factorial row at {milestone}/{view}")
                parent_conditions[parent][condition] = row["free_exact"] is True
            if (
                len(parent_conditions) != 48
                or len(rows) != 192
                or any(set(values) != set(CONDITIONS) for values in parent_conditions.values())
            ):
                raise ValueError(
                    f"complete 48-parent factorial rows required at {milestone}/{view}"
                )

            by_condition = {
                condition: sum(values[condition] for values in parent_conditions.values())
                for condition in CONDITIONS
            }
            joint = sum(all(values.values()) for values in parent_conditions.values())
            stored = record["milestones"][list(MILESTONES).index(milestone)]["factorial"][view]
            if (
                stored.get("items") != len(rows)
                or stored.get("parents") != len(parent_conditions)
                or stored.get("by_condition") != by_condition
                or stored.get("joint_four_exact") != joint
            ):
                raise ValueError(
                    f"stored factorial milestone differs from rows at {milestone}/{view}"
                )
            views[view] = {
                "by_condition": by_condition,
                "joint_four_exact": joint,
                "parent_conditions": dict(parent_conditions),
            }
        result[str(milestone)] = views
    return result


def classify_parent_changes(old: dict, new: dict) -> dict:
    result = {}
    for view in VIEWS:
        old_rows = old["24576"][view]["parent_conditions"]
        new_rows = new["24576"][view]["parent_conditions"]
        parent_ids = sorted(old_rows)
        if parent_ids != sorted(new_rows):
            raise ValueError(f"parent identity changed in {view}")
        joint_old = {parent: all(old_rows[parent].values()) for parent in parent_ids}
        joint_new = {parent: all(new_rows[parent].values()) for parent in parent_ids}
        groups = {
            "both_correct": sorted(
                parent for parent in parent_ids if joint_old[parent] and joint_new[parent]
            ),
            "n117_only": sorted(
                parent for parent in parent_ids if joint_old[parent] and not joint_new[parent]
            ),
            "n118_only": sorted(
                parent for parent in parent_ids if not joint_old[parent] and joint_new[parent]
            ),
            "neither": sorted(
                parent for parent in parent_ids if not joint_old[parent] and not joint_new[parent]
            ),
        }
        condition_changes = {}
        for condition in CONDITIONS:
            old_correct = {parent for parent in parent_ids if old_rows[parent][condition]}
            new_correct = {parent for parent in parent_ids if new_rows[parent][condition]}
            condition_changes[condition] = {
                "n117_only": sorted(old_correct - new_correct),
                "n118_only": sorted(new_correct - old_correct),
                "both_correct": sorted(old_correct & new_correct),
                "neither": sorted(set(parent_ids) - (old_correct | new_correct)),
            }
        result[view] = {
            "joint_four": {key: len(value) for key, value in groups.items()},
            "joint_four_parent_ids": groups,
            "per_condition": condition_changes,
        }
    return result


def summarize_cycles(record: dict, *, normalized: bool) -> list[dict]:
    summaries = []
    for cycle in range(4):
        indexes = [
            i
            for i, measurement in enumerate(record["measurements"])
            if measurement["cycle"] == cycle
        ]
        if len(indexes) != 6144:
            raise ValueError(f"exact cycle measurements required: {cycle}")
        losses = [record["losses"][index] for index in indexes]
        shared_scales = []
        group_scales = {group: [] for group in RWL_GROUPS}
        clipped = {group: 0 for group in RWL_GROUPS}
        for index in indexes:
            groups = record["measurements"][index]["groups"]
            for group in RWL_GROUPS:
                value = groups[group]["scale"]
                group_scales[group].append(value)
                clipped[group] += value < 1.0
            scale_key = "shared_normalized_scale" if normalized else "scale"
            shared_scales.append(groups["reader"][scale_key])
        summaries.append(
            {
                "cycle": cycle,
                "updates": len(indexes),
                "mean_qa_loss": sum(item["qa"] for item in losses) / len(losses),
                "mean_observation_loss": sum(item["observation"] for item in losses) / len(losses),
                "mean_shared_scale": sum(shared_scales) / len(shared_scales),
                "shared_clipped_updates": sum(value < 1.0 for value in shared_scales),
                "mean_group_scales": {
                    group: sum(values) / len(values) for group, values in group_scales.items()
                },
                "group_clipped_updates": clipped,
            }
        )
    return summaries


def audit() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"write-once attribution report exists: {OUTPUT}")
    started = time.process_time()
    n117 = read_source(N117_REPORT, N117_REPORT_SHA256)
    n118 = read_source(N118_REPORT, N118_REPORT_SHA256)
    revalidation = read_source(N118_REVALIDATION, N118_REVALIDATION_SHA256)
    if (
        n117.get("complete") is not True
        or n117.get("schedule_sha256") != n118.get("schedule_sha256")
        or n118.get("status") != "stopped"
        or revalidation.get("complete") is not True
        or revalidation.get("model_receipts_validated") != 9
        or revalidation.get("parameter_updates") != 0
    ):
        raise ValueError("matched and receipt-validated N1-117/N1-118 evidence required")

    old_records = read_position_shards("n117", n117)
    new_records = read_position_shards("n118", n118)
    model_results = []
    for seed in (11, 12, 13):
        old = summarize_factorial_rows(old_records[seed])
        new = summarize_factorial_rows(new_records[seed])
        model_results.append(
            {
                "seed": seed,
                "n117_milestones": {
                    step: {
                        view: {k: v for k, v in data.items() if k != "parent_conditions"}
                        for view, data in views.items()
                    }
                    for step, views in old.items()
                },
                "n118_milestones": {
                    step: {
                        view: {k: v for k, v in data.items() if k != "parent_conditions"}
                        for view, data in views.items()
                    }
                    for step, views in new.items()
                },
                "final_parent_changes": classify_parent_changes(old, new),
                "n117_cycle_trajectory": summarize_cycles(old_records[seed], normalized=False),
                "n118_cycle_trajectory": summarize_cycles(new_records[seed], normalized=True),
            }
        )

    cpu_seconds = round(time.process_time() - started, 6)
    report = {
        "schema": "n1-n117-n118-position-itemwise-attribution-v1",
        "status": "complete" if cpu_seconds <= CPU_LIMIT_SECONDS else "stopped",
        "complete": cpu_seconds <= CPU_LIMIT_SECONDS,
        "n117_report_sha256": N117_REPORT_SHA256,
        "n118_report_sha256": N118_REPORT_SHA256,
        "n118_receipt_revalidation_sha256": N118_REVALIDATION_SHA256,
        "source_shards": {
            f"{label}_seed{seed}": digest for (label, seed), digest in EXPECTED_SHARDS.items()
        },
        "schedule_sha256": n117["schedule_sha256"],
        "milestones": list(MILESTONES),
        "conditions": list(CONDITIONS),
        "models": model_results,
        "model_shards_read": 6,
        "factorial_rows_recomputed": 6 * len(MILESTONES) * len(VIEWS) * 192,
        "model_loads": 0,
        "checkpoint_loads": 0,
        "model_queries": 0,
        "parameter_updates": 0,
        "dev_queries": 0,
        "holdout_queries": 0,
        "confirmation_queries": 0,
        "cpu_seconds": cpu_seconds,
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "causal_claim": False,
        "can_promote": False,
    }
    with OUTPUT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
    return report


if __name__ == "__main__":
    result = audit()
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "complete",
                    "model_shards_read",
                    "factorial_rows_recomputed",
                    "cpu_seconds",
                    "can_promote",
                )
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
