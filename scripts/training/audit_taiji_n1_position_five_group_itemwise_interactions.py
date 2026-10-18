"""Offline item-level audit of the completed PLAN-N1-115 factorial report."""

from __future__ import annotations

import hashlib
import itertools
import json
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "reports/taiji_n1_position_five_group_parameter_factorial_s13_20261018.json"
OUTPUT = ROOT / "reports/taiji_n1_position_five_group_itemwise_interactions_s13_20261018.json"
EXPECTED_SOURCE_SHA256 = "d91ba6c6d36f4ba42c2a0e151965570cfe27c4f646209709b6569e5a47def435"
GROUPS = ("reader", "writer", "local", "context", "relation")
CONDITIONS = ("original", "turn_flip", "direction_opposite", "both")
VIEWS = ("original", "rename")
MASKS = tuple(range(32))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interaction(f00: float, f10: float, f01: float, f11: float) -> float:
    """Second-order finite difference for a matched pair of binary factors."""
    return f11 - f10 - f01 + f00


def _edge_counts(values: list[float]) -> dict:
    return {
        "comparisons": len(values),
        "negative": sum(value < 0 for value in values),
        "zero": sum(value == 0 for value in values),
        "positive": sum(value > 0 for value in values),
        "sum": sum(values),
        "mean": statistics.fmean(values) if values else 0.0,
    }


def analyze(report: dict) -> dict:
    if (
        report.get("complete") is not True
        or report.get("status") != "complete"
        or report.get("model_queries") != 11_520
        or report.get("reused_anchor_rows") != 768
        or report.get("unique_state_case_rows") != 12_288
        or report.get("parameter_updates") != 0
        or report.get("optimizer_steps") != 0
        or report.get("checkpoint_saves") != 0
        or any(
            report.get(key) != 0
            for key in ("dev_queries", "holdout_queries", "confirmation_queries")
        )
    ):
        raise ValueError("complete PLAN-N1-115 train-only report required")

    keyed: dict[tuple[int, str, str, str], bool] = {}
    parents_by_view = {view: set() for view in VIEWS}
    for row in report.get("rows", []):
        key = (
            row.get("parameter_state_mask"),
            row.get("view"),
            row.get("parent_id"),
            row.get("condition"),
        )
        if (
            key[0] not in MASKS
            or key[1] not in VIEWS
            or key[3] not in CONDITIONS
            or not isinstance(row.get("free_exact"), bool)
            or key in keyed
        ):
            raise ValueError("duplicate or invalid state/view/parent/condition row")
        keyed[key] = row["free_exact"]
        parents_by_view[key[1]].add(key[2])
    if len(keyed) != 12_288 or any(len(parents) != 48 for parents in parents_by_view.values()):
        raise ValueError("exact 32 x 2 x 48 x 4 row coverage required")

    parents = {view: tuple(sorted(parents)) for view, parents in parents_by_view.items()}

    def condition_score(mask: int, view: str, parent: str) -> int:
        return sum(keyed[(mask, view, parent, condition)] for condition in CONDITIONS)

    def joint_score(mask: int, view: str, parent: str) -> int:
        return int(all(keyed[(mask, view, parent, condition)] for condition in CONDITIONS))

    # Independently reproduce the sealed aggregate counts before attribution.
    for mask in MASKS:
        for view in VIEWS:
            source_summary = report["summaries"][str(mask)][view]
            for condition in CONDITIONS:
                count = sum(keyed[(mask, view, parent, condition)] for parent in parents[view])
                if count != source_summary["by_condition"][condition]:
                    raise ValueError(f"condition count mismatch at mask={mask}, view={view}")
            joint = sum(joint_score(mask, view, parent) for parent in parents[view])
            if joint != source_summary["joint_four_exact"]:
                raise ValueError(f"joint-four mismatch at mask={mask}, view={view}")

    by_view = {}
    for view in VIEWS:
        group_effects = {}
        harmed_parent_sets = {}
        parent_effect_matrix = {parent: {} for parent in parents[view]}
        for index, group in enumerate(GROUPS):
            bit = 1 << index
            condition_deltas = []
            joint_deltas = []
            parent_net = {}
            for parent in parents[view]:
                parent_condition_delta = 0
                parent_joint_delta = 0
                for mask in MASKS:
                    if mask & bit:
                        continue
                    condition_delta = condition_score(mask | bit, view, parent) - condition_score(
                        mask, view, parent
                    )
                    joint_delta = joint_score(mask | bit, view, parent) - joint_score(
                        mask, view, parent
                    )
                    condition_deltas.append(condition_delta)
                    joint_deltas.append(joint_delta)
                    parent_condition_delta += condition_delta
                    parent_joint_delta += joint_delta
                parent_net[parent] = {
                    "condition_exact_delta_across_16_backgrounds": parent_condition_delta,
                    "joint_four_delta_across_16_backgrounds": parent_joint_delta,
                }
                parent_effect_matrix[parent][group] = parent_joint_delta

            harmed = sorted(
                parent
                for parent, item in parent_net.items()
                if item["joint_four_delta_across_16_backgrounds"] < 0
            )
            helped = sorted(
                parent
                for parent, item in parent_net.items()
                if item["joint_four_delta_across_16_backgrounds"] > 0
            )
            harmed_parent_sets[group] = set(harmed)
            group_effects[group] = {
                "candidate_edge_condition_exact_delta": _edge_counts(condition_deltas),
                "candidate_edge_joint_four_delta": {
                    **_edge_counts(joint_deltas),
                    "degraded": sum(value < 0 for value in joint_deltas),
                    "rescued": sum(value > 0 for value in joint_deltas),
                    "unchanged": sum(value == 0 for value in joint_deltas),
                },
                "parents_net_harmed": len(harmed),
                "parents_net_helped": len(helped),
                "parents_net_unchanged": 48 - len(harmed) - len(helped),
                "parent_net_joint_four_deltas": parent_net,
            }

        pairwise = {}
        for left_index, right_index in itertools.combinations(range(len(GROUPS)), 2):
            left_bit, right_bit = 1 << left_index, 1 << right_index
            exact_interactions = []
            joint_interactions = []
            for mask in MASKS:
                if mask & (left_bit | right_bit):
                    continue
                for parent in parents[view]:
                    f00 = condition_score(mask, view, parent)
                    f10 = condition_score(mask | left_bit, view, parent)
                    f01 = condition_score(mask | right_bit, view, parent)
                    f11 = condition_score(mask | left_bit | right_bit, view, parent)
                    exact_interactions.append(interaction(f00, f10, f01, f11))
                    j00 = joint_score(mask, view, parent)
                    j10 = joint_score(mask | left_bit, view, parent)
                    j01 = joint_score(mask | right_bit, view, parent)
                    j11 = joint_score(mask | left_bit | right_bit, view, parent)
                    joint_interactions.append(interaction(j00, j10, j01, j11))
            pairwise[f"{GROUPS[left_index]}×{GROUPS[right_index]}"] = {
                "condition_exact_interaction": _edge_counts(exact_interactions),
                "joint_four_interaction": _edge_counts(joint_interactions),
            }

        overlap = {}
        for left, right in itertools.combinations(GROUPS, 2):
            union = harmed_parent_sets[left] | harmed_parent_sets[right]
            overlap[f"{left}×{right}"] = {
                "shared_net_harmed_parents": len(
                    harmed_parent_sets[left] & harmed_parent_sets[right]
                ),
                "jaccard": (
                    len(harmed_parent_sets[left] & harmed_parent_sets[right]) / len(union)
                    if union
                    else 1.0
                ),
            }
        by_view[view] = {
            "group_effects": group_effects,
            "pairwise_interactions": pairwise,
            "net_harmed_parent_overlap": overlap,
            "per_parent_group_joint_four_delta": parent_effect_matrix,
        }

    return {
        "schema": "n1-position-five-group-itemwise-interaction-attribution-v1",
        "status": "complete",
        "source_rows": len(keyed),
        "method": {
            "row_metric": "free_exact, matching PLAN-N1-115 factorial summaries",
            "group_edge": "candidate group minus baseline group for each of 16 matched backgrounds and 48 parents",
            "pair_interaction": "f11 - f10 - f01 + f00 over each of 8 matched backgrounds and 48 parents",
            "scope": "offline attribution of consumed train-only outputs; no model or checkpoint access",
        },
        "model_loads": 0,
        "checkpoint_payload_reads": 0,
        "model_queries": 0,
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "checkpoint_saves": 0,
        "dev_queries": 0,
        "holdout_queries": 0,
        "confirmation_queries": 0,
        "summaries_reproduced": True,
        "by_view": by_view,
    }


def main() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"write-once itemwise audit already exists: {OUTPUT}")
    if sha256(SOURCE) != EXPECTED_SOURCE_SHA256:
        raise ValueError("PLAN-N1-115 source report hash mismatch")
    start = time.process_time()
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    result = analyze(source)
    result["source_report"] = {
        "path": str(SOURCE.relative_to(ROOT)),
        "sha256": EXPECTED_SOURCE_SHA256,
    }
    result["analysis_script_sha256"] = sha256(Path(__file__).resolve())
    result["cpu_seconds"] = time.process_time() - start
    encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    with OUTPUT.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded + "\n")
    return result


if __name__ == "__main__":
    result = main()
    print(
        json.dumps(
            {
                "status": result["status"],
                "source_rows": result["source_rows"],
                "report": str(OUTPUT.relative_to(ROOT)),
                "cpu_seconds": result["cpu_seconds"],
            },
            ensure_ascii=False,
        )
    )
