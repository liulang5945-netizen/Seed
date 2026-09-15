"""§4.2 只读诊断：multifamily 选择门的**边界敏感度**测量。

目的：CI 的 `test (3.10)` 腿报 `selector did not choose a group for held-out …`，
即 `InteractionGroupUtilityLearner.select(resource_budget=2.0)` 返回 `None`。
`select` 的判据是 `utility >= minimum_utility(0.0)` 且 `resource_cost <= budget(2.0)`，
而 `interaction = pair - first - second`、`resource_cost` 为均值 —— 都是浮点量。

本探针**不改源码、不训练**，只复现 leave-one-out 的候选与 learner 状态，
打印每个 group 的 utility / resource_cost、**距两个阈值的距离**，
以及"给输入加 ±ε 后 select 结论是否翻转"，用来判定这是否是**边界敏感**（跨解释器可达）。
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_interaction_group_multifamily import (  # noqa: E402
    COMPLEMENTARY_FAMILIES,
    LEARNER_SEEDS,
    _build_corpus,
    _evaluator,
)
from taiji.interaction_group_learning import InteractionGroupUtilityLearner  # noqa: E402

MINIMUM_UTILITY = 0.0
RESOURCE_BUDGET = 2.0
EPSILONS = (1e-12, 1e-9, 1e-6)


def _predict(rows: list[tuple[str, float, float]], *, utility_shift: float, cost_shift: float):
    """在给定微扰下重放 select 的判据，返回被选中的 group_id（或 None）。"""

    usable = [
        (group_id, utility + utility_shift, cost + cost_shift)
        for group_id, utility, cost in rows
        if utility + utility_shift >= MINIMUM_UTILITY and cost + cost_shift <= RESOURCE_BUDGET
    ]
    if not usable:
        return None
    return min(usable, key=lambda item: (-item[1], item[2], item[0]))[0]


def diagnose() -> dict[str, Any]:
    corpus, _ = _build_corpus()
    evaluator = _evaluator()
    report: dict[str, Any] = {
        "format": "taiji-p3b-s42-boundary-diagnosis-v1",
        "thresholds": {"minimum_utility": MINIMUM_UTILITY, "resource_budget": RESOURCE_BUDGET},
        "cases": [],
    }

    for family in COMPLEMENTARY_FAMILIES:
        for seed in LEARNER_SEEDS:
            train = tuple(
                episode
                for episode in corpus.train
                if episode.context_id != f"train-workbench-{family}"
            )
            holdout = tuple(
                episode
                for episode in corpus.holdout
                if episode.context_id == f"holdout-workbench-{family}"
            )
            leave_out = replace(corpus, train=train, holdout=holdout)
            candidates = evaluator.train_only_candidates(leave_out)
            learner = InteractionGroupUtilityLearner(learning_rate=1.0, minimum_utility=0.0)
            ordered = candidates if seed % 2 else tuple(reversed(candidates))
            learner.observe(ordered)

            rows = [
                (group.group_id, float(group.utility), float(group.resource_cost))
                for group in learner.groups
            ]
            margins = [
                {
                    "group_id": group_id,
                    "utility": utility,
                    "utility_margin": utility - MINIMUM_UTILITY,
                    "resource_cost": cost,
                    "cost_margin": RESOURCE_BUDGET - cost,
                }
                for group_id, utility, cost in rows
            ]
            closest = min(
                (min(abs(item["utility_margin"]), abs(item["cost_margin"])) for item in margins),
                default=None,
            )
            flips: dict[str, Any] = {}
            for epsilon in EPSILONS:
                base = _predict(rows, utility_shift=0.0, cost_shift=0.0)
                up = _predict(rows, utility_shift=epsilon, cost_shift=epsilon)
                down = _predict(rows, utility_shift=-epsilon, cost_shift=-epsilon)
                flips[str(epsilon)] = {
                    "base": base,
                    "plus_epsilon": up,
                    "minus_epsilon": down,
                    "flips": (base != up) or (base != down),
                }
            report["cases"].append(
                {
                    "family": family,
                    "seed": seed,
                    "candidate_count": len(candidates),
                    "group_count": len(rows),
                    "closest_boundary_margin": closest,
                    "margins": margins,
                    "epsilon_probe": flips,
                }
            )
    return report


def main() -> int:
    report = diagnose()
    out = PROJECT_ROOT / "reports" / "taiji_p3b_s42_boundary_diagnosis_20260915.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for case in report["cases"]:
        verdict = [k for k, v in case["epsilon_probe"].items() if v["flips"]]
        print(
            f"{case['family']:24} seed={case['seed']:<4} groups={case['group_count']} "
            f"closest_margin={case['closest_boundary_margin']} flips_at={verdict}"
        )
    print(f"report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
