"""Run the bounded Taiji learning and self-evolution simulation gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_auto_growth import evaluate as evaluate_auto_growth  # noqa: E402
from scripts.training.eval_taiji_cross_region_learning import (  # noqa: E402
    evaluate as evaluate_cross_region_learning,
)
from scripts.training.eval_taiji_p6_online_content_credit import (  # noqa: E402
    evaluate as evaluate_content_credit,
)
from scripts.training.eval_taiji_structural_growth import (  # noqa: E402
    evaluate as evaluate_structural_growth,
)

REPORT_FORMAT = "taiji-w7-p4-2-small-simulation-gate-v1"


def evaluate() -> dict[str, object]:
    """Evaluate state transitions, credit, rollback, and checkpoint continuation."""

    auto_growth = evaluate_auto_growth()
    cross_region = evaluate_cross_region_learning()
    content_credit = evaluate_content_credit()
    structural_growth = evaluate_structural_growth()

    metrics = {
        "state_transition_error_to_growth_proposal": bool(
            auto_growth["gate"]["passed"]
            and auto_growth["metrics"]["proposal_emitted_after_steps"] == 3
            and auto_growth["metrics"]["committed"]
        ),
        "cross_region_credit_changes_route_selection": bool(
            cross_region["gate"]["passed"]
            and cross_region["metrics"]["selected_holdout_transfer"]
            > cross_region["metrics"]["fixed_full_holdout_transfer"]
        ),
        "outcome_credit_changes_content_selection": bool(
            content_credit["gate"]["passed"]
            and content_credit["metrics"]["first_content"] != content_credit["metrics"]["second_content"]
            and content_credit["metrics"]["checkpoint_feedback_applied"]
        ),
        "neuron_growth_rollback_restores_parent_state": bool(
            auto_growth["gate"]["passed"]
            and auto_growth["metrics"]["rollback"]
            and auto_growth["metrics"]["unit_count_after_rollback"] == 2
        ),
        "structural_growth_rollback_restores_budget_and_lineage": bool(
            structural_growth["gate"]["passed"]
            and structural_growth["metrics"]["rollback"]
            and structural_growth["metrics"]["budget_after_rollback"] == 1
            and structural_growth["metrics"]["growth_count_after_rollback"] == 0
        ),
        "checkpoint_continuation_preserves_learning_state": bool(
            auto_growth["metrics"]["checkpoint_continuation"]
            and cross_region["metrics"]["checkpoint_continuation"]
            and content_credit["metrics"]["checkpoint_feedback_applied"]
            and structural_growth["metrics"]["checkpoint_request_status"] == "accepted"
        ),
        "resource_and_budget_guards_fail_closed": bool(
            auto_growth["metrics"]["rejected_without_budget"]
            and cross_region["metrics"]["resource_constrained_selection"]
            and structural_growth["metrics"]["rejected_request_status"] == "rejected"
        ),
    }
    component_gates = {
        "auto_growth": auto_growth["gate"],
        "cross_region_learning": cross_region["gate"],
        "content_credit": content_credit["gate"],
        "structural_growth": structural_growth["gate"],
    }
    return {
        "format": REPORT_FORMAT,
        "metrics": metrics,
        "component_gates": component_gates,
        "gate": {
            "passed": all(metrics.values()) and all(
                bool(gate["passed"]) for gate in component_gates.values()
            ),
            "criterion": (
                "bounded native simulations must expose an evidence-driven state transition, "
                "credit updates that change selection, explicit resource/budget rejection, "
                "rollback to the parent state, and checkpoint continuation without mutation "
                "outside the admitted transition"
            ),
        },
        "boundary": (
            "This is a deterministic closed-world CPU mechanism gate. It does not claim a "
            "real-provider quality gain, open-domain transfer, unrestricted self-evolution, "
            "CUDA performance, or general intelligence."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_p4_2_small_simulation_20260831.json",
    )
    args = parser.parse_args()
    report = evaluate()
    report_path = args.report if args.report.is_absolute() else PROJECT_ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
