"""Each fairness rule has to be able to fail, and to pass, on a known fixture."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taiji.goal_fixture_fairness import (  # noqa: E402
    check_control,
    check_goal_carries_no_action_template,
    check_goal_names_no_answer_entity,
    check_goal_not_identical_to_action,
    check_goal_shares_no_bigram_with_action,
    check_mapping_only_reachable_through_records,
    check_split,
)

FAIR = {
    "goal": "让圆口的器皿到四方木台。",
    "action": "把蓝杯放到木桌",
    "answer_entities": ("蓝杯", "木桌"),
    "holdout_entities": ("木桌",),
    "records": ["圆口的器皿是蓝杯。", "四方木台是木桌。"],
    "split": "train",
}
LEAKY = {
    "goal": "把蓝杯放到木桌",
    "action": "把蓝杯放到木桌",
    "answer_entities": ("蓝杯", "木桌"),
    "holdout_entities": ("木桌",),
    "records": [],
    "split": "dev",
}
ENTITIES = ("蓝杯", "红印", "木桌", "窗台")


def rule(row: dict) -> bool | None:
    return row["passed"]


def test_identity_rule_distinguishes_the_two_faces():
    assert rule(check_goal_not_identical_to_action([LEAKY])) is False
    assert rule(check_goal_not_identical_to_action([FAIR])) is True


def test_entity_naming_rule_distinguishes_the_two_faces():
    assert rule(check_goal_names_no_answer_entity([LEAKY], ENTITIES)) is False
    assert rule(check_goal_names_no_answer_entity([FAIR], ENTITIES)) is True


def test_bigram_rule_catches_the_boundary_leak_shape():
    # 10.15's real defect was a shared bigram across the sentence boundary, not a full
    # copy: the descriptor "地上的织物" leaks "到地" against the action "…放到地毯".
    boundary = dict(FAIR, goal="让圆口的器皿到地上的织物。", action="把蓝杯放到地毯")
    shared = check_goal_shares_no_bigram_with_action([boundary])
    assert rule(shared) is False
    assert "到地" in shared["observed"]["worst_shared"]
    assert rule(check_goal_shares_no_bigram_with_action([FAIR])) is True


def test_template_rule_distinguishes_the_two_faces():
    assert rule(check_goal_carries_no_action_template([LEAKY])) is False
    assert rule(check_goal_carries_no_action_template([FAIR])) is True


def test_records_rule_fails_when_the_name_is_missing_or_already_in_the_goal():
    assert rule(check_mapping_only_reachable_through_records([LEAKY])) is False
    assert rule(check_mapping_only_reachable_through_records([FAIR])) is True
    hidden = dict(FAIR, records=["今天天气不错。"])
    assert rule(check_mapping_only_reachable_through_records([hidden])) is False


def test_split_rule_reports_not_applicable_instead_of_passing_silently():
    row = check_split([{k: v for k, v in FAIR.items() if k != "split"}])
    assert row["passed"] is None
    assert row["observed"]["note"]


def test_split_rule_fails_when_the_holdout_reuses_a_train_task():
    assert rule(check_split([FAIR, dict(FAIR, split="dev")])) is False


def test_split_rule_fails_when_the_holdout_entity_leaks_into_train():
    holdout = dict(FAIR, split="dev", goal="让细长的容器到四方木台。", records=["x", "y"])
    assert rule(check_split([FAIR, holdout])) is False


def test_split_rule_passes_on_a_genuinely_new_holdout_task():
    holdout = dict(
        FAIR,
        split="dev",
        goal="让圆口的器皿到靠窗的台面。",
        records=["新表。"],
        holdout_entities=("窗台",),
    )
    assert rule(check_split([FAIR, holdout])) is True


def test_control_rule_needs_the_same_universe_and_no_unchanged_assignment():
    real = [("圆口的器皿", "蓝杯"), ("带纹路的方块", "红印")]
    rotated = [("圆口的器皿", "红印"), ("带纹路的方块", "蓝杯")]
    assert rule(check_control(rotated, real)) is True
    # A rotation that leaves one descriptor its own entity is not a contrast.
    assert rule(check_control([("圆口的器皿", "蓝杯"), ("带纹路的方块", "红印")], real)) is False
    # Duplicated targets are not a permutation.
    assert rule(check_control([("圆口的器皿", "红印"), ("带纹路的方块", "红印")], real)) is False
    # Nor is a mapping into a different universe of values.
    assert rule(check_control([("圆口的器皿", "沙发"), ("带纹路的方块", "红印")], real)) is False
    assert check_control(None, None)["passed"] is None
