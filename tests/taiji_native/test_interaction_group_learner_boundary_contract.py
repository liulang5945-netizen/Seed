"""契约测试：`InteractionGroupUtilityLearner.select` 的**边界语义**。

存在理由（技术债册 §4.2 / [诊断](../../plans/reference/M5_S42_BOUNDARY_DIAGNOSIS_20260915.md)）：
CI 的 `test (3.10)` 腿报 `selector did not choose a group`，实测根因是候选
**恰好压在阈值上**（`closest_boundary_margin = 0.0`，±1e-12 即翻转），
而 `select` 用的是**无容差**的浮点比较：

    item.utility >= self.minimum_utility                    # 0.0
    and (resource_budget is None or item.resource_cost <= float(resource_budget))   # 2.0

本文件把"恰好压线时**必须**被选中"这一事实钉住 —— 任何对阈值、比较符或排序的改动
都会在这里显式失败，而不是静默改变被测机制的含义。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from taiji.interaction_group_learning import InteractionGroupUtilityLearner
from taiji.interaction_groups import InteractionGroupRecord

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIGEST = "0" * 64
MINIMUM_UTILITY = 0.0
RESOURCE_BUDGET = 2.0


def _learner(
    *, utilities: dict[str, float], costs: dict[str, float]
) -> InteractionGroupUtilityLearner:
    """`learning_rate=1.0` 且每条只 observe 一次 ⇒ `utility == record.interaction`，精确可控。"""

    learner = InteractionGroupUtilityLearner(learning_rate=1.0, minimum_utility=MINIMUM_UTILITY)
    learner.observe(
        [
            InteractionGroupRecord(
                group_id=group_id,
                member_ids=("m-a", "m-b"),
                source_trace_digest=DIGEST,
                checkpoint_revision=0,
                contribution=0.0,
                interaction=utilities[group_id],
                uncertainty=0.0,
                resource_cost=costs[group_id],
                owner_policy="test-policy",
                status="candidate",
            )
            for group_id in utilities
        ]
    )
    return learner


def test_exactly_on_the_utility_boundary_is_selected() -> None:
    """§4.2 核心事实：`utility == minimum_utility`（margin = 0）**必须被选中**。"""

    learner = _learner(utilities={"g": MINIMUM_UTILITY}, costs={"g": 1.0})
    selected = learner.select(resource_budget=RESOURCE_BUDGET)
    assert selected is not None
    assert selected.group_id == "g"


def test_exactly_on_the_cost_boundary_is_selected() -> None:
    """`resource_cost == budget`（margin = 0）**必须被选中**。"""

    learner = _learner(utilities={"g": 1.0}, costs={"g": RESOURCE_BUDGET})
    selected = learner.select(resource_budget=RESOURCE_BUDGET)
    assert selected is not None
    assert selected.group_id == "g"


def test_negative_zero_utility_is_still_selected() -> None:
    """`-0.0 >= 0.0` 为真 ⇒ 记录该边界事实（不是想当然）。"""

    learner = _learner(utilities={"g": -0.0}, costs={"g": 1.0})
    assert learner.select(resource_budget=RESOURCE_BUDGET) is not None


def test_just_below_the_utility_boundary_is_rejected() -> None:
    """阈值另一侧：略负即不选 —— 说明失败确实只由**末位差异**决定。"""

    learner = _learner(utilities={"g": -1e-18}, costs={"g": 1.0})
    assert learner.select(resource_budget=RESOURCE_BUDGET) is None


def test_just_over_the_cost_boundary_is_rejected() -> None:
    """阈值另一侧：刚过界即不选（用**可表示**的增量 1e-12）。"""

    learner = _learner(utilities={"g": 1.0}, costs={"g": RESOURCE_BUDGET + 1e-12})
    assert learner.select(resource_budget=RESOURCE_BUDGET) is None


def test_epsilon_below_float_resolution_is_absorbed() -> None:
    """记录事实：`2.0 + 1e-18 == 2.0`（被浮点舍入吞掉）。

    这正是"构造刚过界的情形"容易写错的地方 —— 也提醒：边界敏感的**真实尺度**
    由双精度相对精度（约 2.2e-16 × 量级）决定，而不是随便选个"极小值"。
    """

    learner = _learner(utilities={"g": 1.0}, costs={"g": RESOURCE_BUDGET + 1e-18})
    assert learner.groups[0].resource_cost == RESOURCE_BUDGET  # 增量被吞掉
    assert learner.select(resource_budget=RESOURCE_BUDGET) is not None


def test_all_over_budget_returns_none() -> None:
    learner = _learner(utilities={"g1": 1.0, "g2": 2.0}, costs={"g1": 3.0, "g2": 4.0})
    assert learner.select(resource_budget=RESOURCE_BUDGET) is None


def test_no_groups_returns_none() -> None:
    learner = InteractionGroupUtilityLearner(learning_rate=1.0, minimum_utility=MINIMUM_UTILITY)
    assert learner.select(resource_budget=RESOURCE_BUDGET) is None


def test_none_budget_ignores_cost() -> None:
    learner = _learner(utilities={"g": 1.0}, costs={"g": 1e6})
    selected = learner.select(resource_budget=None)
    assert selected is not None and selected.group_id == "g"


def test_negative_budget_is_rejected() -> None:
    learner = _learner(utilities={"g": 1.0}, costs={"g": 1.0})
    with pytest.raises(ValueError):
        learner.select(resource_budget=-0.1)


def test_selection_order_is_deterministic() -> None:
    """tie-break 已经确定性：先 utility 大、再 cost 小、最后 group_id 字典序。"""

    # utility 相同 ⇒ 取 cost 更小者
    learner = _learner(utilities={"g1": 1.0, "g2": 1.0}, costs={"g1": 2.0, "g2": 1.0})
    assert learner.select(resource_budget=RESOURCE_BUDGET).group_id == "g2"
    # utility 与 cost 都相同 ⇒ 取 group_id 更小者
    learner = _learner(utilities={"g1": 1.0, "g2": 1.0}, costs={"g1": 1.0, "g2": 1.0})
    assert learner.select(resource_budget=RESOURCE_BUDGET).group_id == "g1"


def test_diagnosis_fixture_recorded_zero_margin() -> None:
    """若诊断报告存在，其结论必须仍是「压线」（防止报告被静默改写）。"""

    report = PROJECT_ROOT / "reports" / "taiji_p3b_s42_boundary_diagnosis_20260915.json"
    if not report.is_file():  # 诊断产物可选：缺失时跳过本项，不制造假失败
        pytest.skip("boundary diagnosis report not present")
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["thresholds"] == {
        "minimum_utility": MINIMUM_UTILITY,
        "resource_budget": RESOURCE_BUDGET,
    }
    cases = payload["cases"]
    assert cases, "diagnosis must contain cases"
    assert all(
        case["closest_boundary_margin"] == 0.0 for case in cases
    ), "所有 case 的最近边界距离都应为 0.0（恰好压线）"
    # 且最小 ε 就足以翻转
    assert all(
        case["epsilon_probe"]["1e-12"]["flips"] is True for case in cases
    ), "±1e-12 应足以翻转 select 的结论"
    assert math.isfinite(sum(case["closest_boundary_margin"] for case in cases))
