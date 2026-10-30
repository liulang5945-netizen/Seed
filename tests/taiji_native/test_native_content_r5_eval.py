"""R5 evaluation helper contracts: fixed denominator, receipts, mutations, bootstrap."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from taiji.native_content_eval import (
    MutationV1,
    _swapped_text,
    bootstrap_diff,
    exact_vector,
    expected_turns,
    mechanism_mutations,
    receipt_ratio,
)
from taiji.native_content_evaluator import ExpectedTurnV1, TurnPredictionV1

ROOT = Path(__file__).resolve().parents[2]
_script = importlib.util.spec_from_file_location(
    "evaluate_native_content_core_r5",
    ROOT / "scripts/training/evaluate_native_content_core_r5.py",
)
assert _script is not None and _script.loader is not None
R5 = importlib.util.module_from_spec(_script)
sys.modules[_script.name] = R5
_script.loader.exec_module(R5)

ANSWER = "雨伞的颜色是蓝色。<|end|>".encode()


def _episode() -> dict:
    return {
        "episode_id": "r1h:test:v0",
        "turns": [
            {
                "index": 0,
                "role": "fact_v0",
                "scored": False,
                "text": "雨伞的颜色是蓝色。",
                "entity": "雨伞",
                "attribute": "颜色",
                "value": "蓝色",
            },
            {
                "index": 2,
                "role": "query_current",
                "scored": True,
                "category": "current_query",
                "text": "雨伞是什么颜色？",
                "target": {"target_bytes_utf8": list(ANSWER)},
            },
            {
                "index": 5,
                "role": "distractor",
                "scored": False,
                "text": "背包的颜色是红色。",
                "entity": "背包",
                "attribute": "颜色",
                "value": "红色",
            },
        ],
    }


def test_expected_turns_keep_every_scored_turn():
    turns = expected_turns([_episode()])
    assert len(turns) == 1
    assert turns[0].category == "current_query"
    assert turns[0].expected == ANSWER


def test_exact_vector_counts_missing_and_parse_failures_as_wrong():
    turn = ExpectedTurnV1("r1h:test:v0", 2, "current_query", ANSWER)
    good = TurnPredictionV1("r1h:test:v0", 2, ANSWER)
    bad_parse = TurnPredictionV1("r1h:test:v0", 2, ANSWER, parse_ok=False)
    wrong = TurnPredictionV1("r1h:test:v0", 2, "我不知道。<|end|>".encode())
    assert exact_vector([turn], {turn.key: good}) == [1.0]
    assert exact_vector([turn], {turn.key: bad_parse}) == [0.0]
    assert exact_vector([turn], {turn.key: wrong}) == [0.0]
    assert exact_vector([turn], {}) == [0.0]


def test_receipt_ratio_requires_a_correct_turn_and_a_verbatim_value():
    turn = ExpectedTurnV1("r1h:test:v0", 2, "current_query", ANSWER)
    correct = TurnPredictionV1("r1h:test:v0", 2, ANSWER, selected_value="蓝色")
    paraphrased = TurnPredictionV1("r1h:test:v0", 2, ANSWER, selected_value="深蓝色")
    wrong = TurnPredictionV1("r1h:test:v0", 2, "我不知道。<|end|>".encode(), selected_value="蓝色")
    assert receipt_ratio([turn], {turn.key: correct}) == (1, 1)
    assert receipt_ratio([turn], {turn.key: paraphrased}) == (0, 1)
    assert receipt_ratio([turn], {turn.key: wrong}) == (0, 0)
    assert receipt_ratio([turn], {}) == (0, 0)


def test_mechanism_mutations_stay_in_domain_and_keep_unrelated_answer():
    episode = _episode()
    mutations = mechanism_mutations(episode, 2, ("nc1:test", "蓝色", 0), value_seed=7)
    assert mutations["cut"] == MutationV1(0, "skip")
    assert mutations["relevant_swap"].turn_index == 0
    assert mutations["relevant_swap"].value in ("红色", "绿色", "黄色")
    assert mutations["unrelated_swap"].turn_index == 5
    assert mutations["unrelated_swap"].value in ("蓝色", "绿色", "黄色")
    replacement = mutations["relevant_swap"].value
    assert mutations["expected"] == ANSWER.replace("蓝色".encode(), replacement.encode())


def test_swapped_text_handles_owner_pronoun_forms():
    assert _swapped_text({"text": "雨伞现在归我。", "value": "我的"}, "你的") == "雨伞现在归你。"
    assert (
        _swapped_text({"text": "雨伞的颜色是蓝色。", "value": "蓝色"}, "红色")
        == "雨伞的颜色是红色。"
    )
    assert _swapped_text({"text": "没有对应值。", "value": "我的"}, "你的") is None


def test_bootstrap_diff_separates_clear_distributions():
    same = bootstrap_diff([1, 0, 1, 1], [1, 0, 1, 1], draws=200, seed=0)
    assert abs(same["mean_diff"]) < 0.2
    clear = bootstrap_diff([1, 1, 1, 1], [0, 0, 0, 0], draws=200, seed=0)
    assert clear["mean_diff"] == 1.0
    assert clear["lower_95"] == 1.0


def _passing_families() -> dict:
    return {
        "behavior": {"answer_exact": {"value": 1.0, "pass": True}},
        "mechanism": {
            "cut_drop": {"value": 1.0, "pass": True},
            "rescue_gain": {"value": 1.0, "pass": True},
            "change_difference_ci_lower": {"value": 0.5, "pass": True},
        },
        "retention": {
            "train_like_exact": {"value": 1.0, "pass": True},
            "holdout_minus_train_like": {"value": 0.0, "pass": True},
            "receipt_exact": {"value": 1.0, "pass": True},
        },
        "comparison": {
            "holdout_margin": {"value": 0.5, "pass": True},
            "holdout_ci_lower": {"value": 0.4, "pass": True},
            "query_not_below": {"value": 0.0, "pass": True},
            "q2_margin": {"value": 0.2, "pass": True},
            "q2_ci_lower": {"value": 0.1, "pass": True},
        },
        "cost": {"new_core_over_best_baseline": {"value": 0.7, "pass": True}},
        "recovery": {"checkpoints": {"value": True, "pass": True}},
    }


def test_verdict_passes_only_when_every_family_passes():
    assert R5._verdict(_passing_families()) == ("PASS_Q1", [])


def test_verdict_reports_the_first_failing_family():
    families = _passing_families()
    families["mechanism"]["cut_drop"] = {"value": 0.2, "pass": False}
    families["retention"]["receipt_exact"] = {"value": 0.1, "pass": False}
    status, failed = R5._verdict(families)
    assert status == "FAIL_MECHANISM"
    assert failed == ["mechanism.cut_drop", "retention.receipt_exact"]


def test_verdict_treats_a_missing_value_as_inconclusive():
    families = _passing_families()
    families["mechanism"]["cut_drop"] = {"value": None, "pass": False}
    assert R5._verdict(families) == ("INCONCLUSIVE", [])
