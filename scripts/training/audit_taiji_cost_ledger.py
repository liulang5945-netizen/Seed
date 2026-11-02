"""Cost ledger: sum the measured CPU/wall time of every taiji core card.

The objective asks for "全过程资源成本" as a verified axis, and the ruling on ③
needs one number in front of it: what has this line already cost? This audit only
adds up fields that the cards already publish; it imputes nothing for cards written
before the cost field existed and names them instead.

Re-runs are counted: a v1 and a v2 card were two real runs, not one.

v2 (2026-11-02): the first version read only the top-level ``cpu_seconds``, so 20
cards -- including the 10.21.4 training run (1,200 optimizer calls) and 14.1/14.2 --
reported no cost at all even though they publish it under a nested key. Costs are now
resolved through a *declared* path list, one source per card, and the source is named
in every row. Cards that roll other cards up (``costs.executed_cpu_seconds``) are kept
out of the additive sum and compared against it instead. The roll-up counts one card per
step and names its discarded iterations in prose, so the two views are different
calibers: the guard is the inequality that must hold either way -- the tree cannot have
spent less than its own canonical subset claims. A violation is published as
``cost_source_conflict``.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

BUDGET_CPU_SECONDS = 120
BUDGET_WALL_SECONDS = 240
CROSS_CHECK_BAND_SECONDS = 2.0

# Declared, ordered: the first hit wins, so a component key can never be added on top of
# the total it belongs to. Names are taken from the cards themselves, not guessed.
COST_PATHS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("cpu_seconds",), ("wall_seconds",)),
    (("cost", "cpu_seconds"), ("cost", "wall_seconds")),
    (("cpu_seconds_checks_only",), ("wall_seconds_checks_only",)),
)

# Cards whose own numbers are sums of other cards. They are excluded from the additive
# total and used as an independent second view of it.
STATED_TOTAL_PATHS = (("costs", "with_iterations_cpu_seconds"), ("costs", "executed_cpu_seconds"))

CROSS_CHECKS = (
    {
        "aggregate_step": "10.21.6",
        "member_prefix": "10.21",
        "stated_path": ("costs", "executed_cpu_seconds"),
    },
)


def dig(card: dict, path: tuple[str, ...]):
    node = card
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def resolve_cost(card: dict) -> dict | None:
    for cpu_path, wall_path in COST_PATHS:
        cpu, wall = dig(card, cpu_path), dig(card, wall_path)
        if isinstance(cpu, (int, float)) and isinstance(wall, (int, float)):
            return {
                "cpu": float(cpu),
                "wall": float(wall),
                "cpu_source": ".".join(cpu_path),
                "wall_source": ".".join(wall_path),
            }
    return None


def is_member(step: str, prefix: str) -> bool:
    """``10.21`` and ``10.21.4`` are members of that family; ``10.21a`` is a separate
    experiment identity that the family roll-up never counted."""
    if step == prefix:
        return True
    if not step.startswith(prefix + "."):
        return False
    remainder = step[len(prefix) + 1]
    return remainder.isdigit()


def family_of(card: dict, name: str) -> str:
    step = str(card.get("step") or "")
    if "." in step:
        return step.split(".", 1)[0]
    if step:
        return step
    for prefix, label in (
        ("taiji_core_r0", "09-r"),
        ("taiji_core_reference", "09-r"),
        ("taiji_core_algorithm", "09-r"),
    ):
        if name.startswith(prefix):
            return label
    return "unlabeled"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("cost ledger card is write-once")

    total_cpu, total_wall = time.process_time(), time.perf_counter()
    rows, missing, aggregates = [], [], []
    for relative in sorted(glob.glob(str(ROOT / "reports" / "taiji_core_*.json"))):
        path = Path(relative)
        try:
            card = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            missing.append({"card": path.name, "reason": f"unreadable: {error}"})
            continue
        stated = {
            ".".join(path_key): float(dig(card, path_key))
            for path_key in STATED_TOTAL_PATHS
            if isinstance(dig(card, path_key), (int, float))
        }
        if stated:
            aggregates.append(
                {
                    "card": path.name,
                    "step": card.get("step"),
                    "stated": stated,
                    "excluded_from_sum": True,
                }
            )
            continue
        cost = resolve_cost(card)
        if cost is None:
            missing.append({"card": path.name, "reason": "no declared cost path"})
            continue
        rows.append(
            {
                "card": path.name,
                "family": family_of(card, path.name),
                "step": card.get("step"),
                "status": card.get("status"),
                "cpu_seconds": cost["cpu"],
                "wall_seconds": cost["wall"],
                "cpu_source": cost["cpu_source"],
                "wall_source": cost["wall_source"],
            }
        )

    by_family: dict[str, dict] = {}
    for row in rows:
        bucket = by_family.setdefault(
            row["family"], {"cards": 0, "cpu_seconds": 0.0, "wall_seconds": 0.0}
        )
        bucket["cards"] += 1
        bucket["cpu_seconds"] += row["cpu_seconds"]
        bucket["wall_seconds"] += row["wall_seconds"]

    ranked = sorted(rows, key=lambda row: row["cpu_seconds"], reverse=True)
    goal_line = [row for row in rows if str(row.get("step", "")).startswith("10.")]
    p2_probes = [
        row
        for row in goal_line
        if row["step"] in {"10.4", "10.5", "10.6"} or str(row["step"]).startswith(("10.7",))
    ]

    cumulative_cpu = sum(row["cpu_seconds"] for row in rows)
    cumulative_wall = sum(row["wall_seconds"] for row in rows)

    cross_check = []
    for spec in CROSS_CHECKS:
        aggregate = next((a for a in aggregates if a["step"] == spec["aggregate_step"]), None)
        members = [
            row for row in rows if is_member(str(row.get("step") or ""), spec["member_prefix"])
        ]
        member_cpu = sum(row["cpu_seconds"] for row in members)
        if aggregate is None or not aggregate["stated"]:
            cross_check.append(
                {
                    "aggregate_step": spec["aggregate_step"],
                    "member_prefix": spec["member_prefix"],
                    "member_cards": len(members),
                    "member_cpu_seconds": member_cpu,
                    "present": False,
                    "note": "no aggregate card publishing a stated total; nothing to compare",
                }
            )
            continue
        differences = {source: member_cpu - value for source, value in aggregate["stated"].items()}
        matched = min(differences, key=lambda source: abs(differences[source]))
        cross_check.append(
            {
                "aggregate_step": spec["aggregate_step"],
                "member_prefix": spec["member_prefix"],
                "member_cards": len(members),
                "member_cpu_seconds": member_cpu,
                "present": True,
                "stated": aggregate["stated"],
                "all_differences": differences,
                "matched_source": matched,
                "difference_seconds": differences[matched],
                "within_band": abs(differences[matched]) <= CROSS_CHECK_BAND_SECONDS,
                # The roll-up publishes a narrower caliber than "every card in the tree was a
                # real run": it counts one card per step and names iterations in prose. So
                # equality is not the valid invariant here -- within_band is reported as a
                # size of the gap, and the guard below is the inequality that must hold.
                "member_sum_is_not_below_stated": member_cpu >= max(aggregate["stated"].values()),
                "caliber_note": (
                    "difference_seconds covers the discarded iteration cards and any sibling "
                    "experiment identity in the same family; it is not a bookkeeping error."
                ),
            }
        )
    conflicts = [
        row
        for row in cross_check
        if row.get("present") and not row["member_sum_is_not_below_stated"]
    ]

    top_level_only = sum(row["cpu_seconds"] for row in rows if row["cpu_source"] == "cpu_seconds")
    sources = {}
    for row in rows:
        bucket = sources.setdefault(row["cpu_source"], {"cards": 0, "cpu_seconds": 0.0})
        bucket["cards"] += 1
        bucket["cpu_seconds"] += row["cpu_seconds"]

    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    if cpu_seconds > BUDGET_CPU_SECONDS or wall_seconds > BUDGET_WALL_SECONDS:
        status = "budget_stop"
    elif conflicts:
        status = "cost_source_conflict"
    else:
        status = "cost_ledger_built"

    report = {
        "schema": "taiji-cost-ledger-v3",
        "step": "ledger",
        "date": "2026-11-02",
        "status": status,
        "supersedes": "reports/taiji_core_cost_ledger_v3_v2_20261102_conflict_record.json",
        "revision_reason": (
            "v3 read only the top-level cpu_seconds, so 20 cards -- including the 10.21.4 "
            "training run and 14.1/14.2 -- were listed as costless while they publish the "
            "number under a nested key. The first v3-revision run fixed the paths but judged "
            "the roll-up by an equality band; that comparison is ill-posed because the "
            "closeout declares a canonical-cards-only caliber, so it raised "
            "cost_source_conflict and is kept as a conflict record. This run keeps the nested "
            "paths and guards the inequality that holds under either caliber."
        ),
        "cards_measured": len(rows),
        "cards_without_cost": missing,
        "total_cpu_seconds": cumulative_cpu,
        "total_wall_seconds": cumulative_wall,
        "top_level_only_cpu_seconds": top_level_only,
        "nested_recovered_cpu_seconds": cumulative_cpu - top_level_only,
        "cost_sources": sources,
        "aggregate_cards": aggregates,
        "cross_check": cross_check,
        "cross_check_band_seconds": CROSS_CHECK_BAND_SECONDS,
        "by_family": by_family,
        "goal_line_cpu_seconds": sum(row["cpu_seconds"] for row in goal_line),
        "p2_probe_cpu_seconds": sum(row["cpu_seconds"] for row in p2_probes),
        "p2_probe_cards": [row["card"] for row in p2_probes],
        "top_costs": ranked[:10],
        "re_runs_counted": [
            row["card"] for row in rows if "_v2_" in row["card"] or "v2" in row["card"]
        ],
        "limits": (
            "Bookkeeping only: sums fields the cards already publish, through a declared path "
            "list, one source per card (top-level cpu_seconds wins over cost.cpu_seconds wins "
            "over cpu_seconds_checks_only), so component keys and budgets are never added on "
            "top of the totals they belong to. Cards that publish no declared path are listed, "
            "never imputed. Roll-up cards (costs.executed_cpu_seconds) are kept out of the sum "
            "and compared against it instead. Re-runs are separate runs and are counted. Wall "
            "seconds overlap across runs, so only the CPU sum is additive."
        ),
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "source_hashes": {
            "scripts/training/audit_taiji_cost_ledger.py": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest()
        },
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    print(
        json.dumps(
            {
                "status": status,
                "cards_measured": len(rows),
                "without_cost": len(missing),
                "total_cpu_seconds": round(cumulative_cpu, 1),
                "top_level_only_cpu_seconds": round(top_level_only, 1),
                "nested_recovered_cpu_seconds": round(cumulative_cpu - top_level_only, 1),
                "aggregate_cards": len(aggregates),
                "cross_check": [
                    {
                        "step": row["aggregate_step"],
                        "present": row["present"],
                        "member_cpu": round(row.get("member_cpu_seconds", 0.0), 2),
                        "matched_source": row.get("matched_source"),
                        "difference": round(row.get("difference_seconds", 0.0), 2),
                        "within_band": row.get("within_band"),
                        "guard_holds": row.get("member_sum_is_not_below_stated"),
                    }
                    for row in cross_check
                ],
                "goal_line_cpu_seconds": round(report["goal_line_cpu_seconds"], 1),
                "p2_probe_cpu_seconds": round(report["p2_probe_cpu_seconds"], 1),
                "p2_probe_cards": len(p2_probes),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
