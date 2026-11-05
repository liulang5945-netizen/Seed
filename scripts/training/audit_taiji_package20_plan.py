"""Audit the package-20 plan using public finite rules and frozen report metadata.

No Taiji imports, model construction, weights, dataset materialization, sealed
examples, optimizer, or training are involved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DRAFT = ROOT / "plans/manifests/taiji_package20_learning_correction_draft_20261105.json"
ACTION = (
    "20.1：实现规则学习夹具与CF1/B_EH单一参考图，完成信息边界与参数/状态清单；"
    "正式训练未准入，旧confirm保持封存。"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def public_rule_audit() -> dict[str, Any]:
    rules = [(mask, bit) for mask in range(1, 16) if mask.bit_count() in (1, 2) for bit in (0, 1)]
    anchors = [0, 1, 2, 4, 8]
    queries = [value for value in range(16) if value not in anchors]

    def label(rule: tuple[int, int], value: int) -> int:
        return ((rule[0] & value).bit_count() % 2) ^ rule[1]

    signatures = {tuple(label(rule, value) for value in anchors) for rule in rules}
    assert len(rules) == len(signatures) == 20
    changes = Counter(
        sum(label(first, value) != label(second, value) for value in queries)
        for first in rules
        for second in rules
        if first != second
    )
    assert dict(changes) == {4: 60, 5: 120, 6: 120, 7: 60, 11: 20}
    groups: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
    for rule in rules:
        labels = tuple(label(rule, value) for value in anchors)
        assert labels != labels[1:] + labels[:1]
        for query in queries:
            groups[query, sum(labels)][label(rule, query)] += 1
    shortcut_hits = sum(max(counter.values()) for counter in groups.values())
    shortcut = Fraction(shortcut_hits, len(rules) * len(queries))
    assert shortcut == Fraction(7, 11)
    # A blind flip of the old rule is perfect on the changed-only subset,
    # despite not observing any new feedback. It is not a correction gate.
    changed_count = sum(size * count for size, count in changes.items())
    flip_whole = Fraction(changed_count, sum(changes.values()) * len(queries))
    assert flip_whole == Fraction(10, 19)
    misleading_pool = Fraction(11 + 7, 22)
    assert (misleading_pool - shortcut) / (1 - shortcut) == Fraction(1, 2)
    expected_ceiling = (41 + Fraction(5, 2) + Fraction(50, 19)) / 51
    assert expected_ceiling == Fraction(1753, 1938)
    return {
        "rule_count": len(rules),
        "unique_anchor_signatures": len(signatures),
        "non_anchor_queries": len(queries),
        "ordered_distinct_rule_pairs": sum(changes.values()),
        "changed_query_distribution": dict(changes),
        "label_multiset_shortcut": {"correct": shortcut_hits, "total": 220, "rate": str(shortcut)},
        "blind_old_rule_flip": {"changed_only": "1", "all_current_queries": str(flip_whole)},
        "pooled_masking_counterexample": str(misleading_pool),
        "all_rounds": 10 + 24 + 17,
        "new_first_queries": 5 + 11 + 6,
        "all_round_expected_information_reference": str(expected_ceiling),
        "reference_is_not_single_sample_hard_ceiling": True,
        "five_cycle_always_changes_anchor_label_binding": True,
        "scope": "Public rule family only; no world, source name, split or confirm example generated.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    output = None
    if args.report is not None:
        output = args.report.resolve()
        if output.parent != (ROOT / "reports").resolve() or output.exists():
            parser.error("Choose a new file directly inside reports/.")
    started = time.process_time()
    draft = read_json(DRAFT)
    confirmed_path = ROOT / "reports/taiji_package19_2v_confirm_holdout_20261105.json"
    fixed_path = ROOT / "reports/taiji_package19_2w_boundary_residual_20261105.json"
    ablation_path = ROOT / "reports/taiji_package19_2u_stop_boundary_ablation_20261105.json"
    confirmed, fixed, ablation = map(read_json, (confirmed_path, fixed_path, ablation_path))
    c = draft["parent_assets"]["confirmed_reference"]
    r = draft["parent_assets"]["development_parent"]
    assert confirmed["assembly"]["candidate_sha_v2"] == c["candidate_sha_v2"]
    assert confirmed["assembly"]["assembled_state_sha256"] == c["assembly_sha256"]
    assert fixed["fix"]["candidate_sha_v2_after"] == r["candidate_sha_v2"]
    assert fixed["fix"]["assembled_r2_sha256"] == r["assembly_sha256"]
    assert (
        fixed["frozen_inputs"]["parent_bundle_sha256"]
        == draft["parent_assets"]["root_bundle_sha256"]
    )
    assert fixed["disclosure"]["sealed_set_recheck"] == "not_performed"
    assert confirmed["readings"]["gamma_0"]["total"]["completion"] == 188 / 192
    assert ablation["ablation"]["parameters"] == 12
    assert ablation["ablation"]["elements"] == 46642
    assert not draft["training_authorized"] and not draft["training_admitted"]
    assert not draft["data_design"]["fresh_confirm_materialized"]
    assert draft["data_design"]["actual_worlds_materialized_this_turn"] == 0
    plan = draft["public_task"]
    assert sum(plan["session_answer_rounds"]) == plan["all_answer_rounds"] == 51
    assert plan["first_query_rounds"] == 22
    assert draft["comparison_design"]["independently_trained_arms"] == ["A_F", "B_EH"]
    for name in (
        "plans/MAINLINE.md",
        "plans/MASTER-PLAN.md",
        "plans/reference/ROADMAP-NEXT_20261104.md",
    ):
        assert (ROOT / name).read_text(encoding="utf-8").count(ACTION) == 1, name
    evidence = {
        str(path.relative_to(ROOT))
        .replace("\\", "/"): hashlib.sha256(path.read_bytes())
        .hexdigest()
        for path in (DRAFT, confirmed_path, fixed_path, ablation_path)
    }
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    report = {
        "schema": "taiji-package20-planning-review-v1",
        "date": "2026-11-05",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "basis_commit": revision,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "evidence_sha256": evidence,
        "status": "planning_static_checks_passed",
        "passed": True,
        "public_rule_checks": public_rule_audit(),
        "parent_metadata_checked": True,
        "same_queue_in_three_entries": True,
        "candidate_confirmed": False,
        "training_admitted": False,
        "counts": {
            "models_constructed": 0,
            "weights_loaded": False,
            "optimizer_steps": 0,
            "candidate_evaluations": 0,
            "actual_worlds_generated": 0,
            "sealed_dataset_reads": 0,
        },
        "static_audit_cpu_seconds": time.process_time() - started,
        "scope": "Planning and public finite-family proof only; no Task2 capability or promotion claim.",
    }
    if output is not None:
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "rules": 20,
                "queue_agrees": True,
                "report": str(output) if output else None,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
