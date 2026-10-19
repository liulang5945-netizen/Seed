from __future__ import annotations

from scripts.training import eval_taiji_n1_v15_after_n132 as evaluator


def _rows(seed: int, arm: str, source: str, exact_by_kind: dict[str, int]) -> list[dict]:
    rows = []
    for kind_index, kind in enumerate(evaluator.KINDS):
        for ordinal in range(5):
            rows.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "endpoint_source": source,
                    "endpoint_role": "synthetic",
                    "id": f"v15-confirm-{kind_index * 5 + ordinal:02d}",
                    "split": "holdout",
                    "kind": kind,
                    "free_exact": ordinal < exact_by_kind[kind],
                }
            )
    return rows


def test_range_summary_counts_only_fifteen_decodable_rows() -> None:
    endpoints = []
    for index, (seed, arm) in enumerate(
        [(seed, arm) for arm in ("position", "terminal", "cut") for seed in (11, 12, 13)]
    ):
        rows = _rows(
            seed, arm, "N1-120", {"lookup": 0, "revision": 0, "transition": 0, "unknown": 0}
        )
        decodable_errors = 6 if index < 6 else 5
        for row_index, row in enumerate(rows):
            row["free_exact"] = row_index < 15 - decodable_errors if row_index < 15 else False
        endpoints.append({"seed": seed, "arm": arm, "rows": rows})

    summary = evaluator.n120_eval._range_summary(endpoints)

    assert summary["decodable_items_per_endpoint"] == 15
    assert summary["usable_endpoints"] == 6
    assert summary["range_check_passed"] is True


def test_confirmation_summary_uses_candidate_and_control_gates_only() -> None:
    rows = []
    for seed in evaluator.SEEDS:
        rows.extend(_rows(seed, "position", "N1-132", {kind: 3 for kind in evaluator.KINDS}))
        rows.extend(_rows(seed, "position", "N1-120", {kind: 2 for kind in evaluator.KINDS}))
        rows.extend(_rows(seed, "terminal", "N1-120", {kind: 0 for kind in evaluator.KINDS}))
        rows.extend(_rows(seed, "cut", "N1-120", {kind: 0 for kind in evaluator.KINDS}))

    summary = evaluator._confirmation_summary(rows)

    assert summary["gate_passed"] is True
    assert summary["package_vs_n120_controls"]["all_four_gates"] is True
    assert (
        summary["same_seed_n120_position_parent_descriptive"][11]["lookup"][
            "candidate_only_correct"
        ]
        == 1
    )
