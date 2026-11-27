"""Human ratings remain bound to blinded text and fixed denominators."""

import copy

import pytest

from scripts.training.adjudicate_taiji_n1_plan175_review import (
    finalize,
    parse_markdown,
    summarize,
    validate_review,
)

IDENTITY = "a" * 32


def example():
    text = (
        "评阅者：待填写。已填评分：0/312。\n"
        "问题：你好\n"
        f"| `{IDENTITY}` | 待填 | 待填 |\n"
        "```text\n"
        f"| `{'b' * 32}` | 2 | 这是待评价文本，不是评分 |\n"
        "```\n"
    )
    blind = {"cases": [{"case_id": "case", "responses": [{"response_id": IDENTITY}]}]}
    return text, blind, parse_markdown(text)["canonical_sha256"]


def test_zero_is_a_completed_rating_and_fenced_answer_is_never_a_rating():
    text, blind, expected = example()
    text = text.replace("评阅者：待填写", "评阅者：测试评阅者").replace(
        f"| `{IDENTITY}` | 待填 | 待填 |", f"| `{IDENTITY}` | 0 | 没有回答问题 |"
    )
    result = validate_review(text, blind, expected)
    assert result["complete"] and result["pending"] == 0
    assert result["ratings"][IDENTITY]["score"] == 0
    assert len(result["ratings"]) == 1


def test_changed_question_or_answer_is_rejected():
    text, blind, expected = example()
    with pytest.raises(ValueError, match="immutable"):
        validate_review(text.replace("问题：你好", "问题：再见"), blind, expected)


def test_popup_progress_status_preserves_the_same_immutable_content():
    text, blind, expected = example()
    baseline = "状态：待人工判读；评分与评阅责任人尚未填写。\n" + text
    expected = parse_markdown(baseline)["canonical_sha256"]
    changed = baseline.replace(
        "状态：待人工判读；评分与评阅责任人尚未填写。",
        "状态：人工判读进行中；已记录弹窗评分。",
    ).replace("评阅者：待填写", "评阅者：用户（弹窗）")
    result = validate_review(changed, blind, expected)
    assert result["pending"] == 1 and not result["complete"]


@pytest.mark.parametrize("score,reason", [("3", "理由"), ("0", "待填"), ("2", "")])
def test_invalid_or_unreasoned_ratings_are_rejected(score, reason):
    text, _, _ = example()
    with pytest.raises(ValueError):
        parse_markdown(text.replace("待填 | 待填", f"{score} | {reason}"))


def test_duplicate_response_is_rejected():
    text, _, _ = example()
    with pytest.raises(ValueError, match="duplicate"):
        parse_markdown(text + f"| `{IDENTITY}` | 0 | 理由 |\n")


def test_incomplete_review_does_not_create_a_freeze_or_open_key(tmp_path):
    text, blind, expected = example()
    review = validate_review(text, blind, expected)
    paths = {name: tmp_path / name for name in ("freeze", "result", "key", "rollout")}
    with pytest.raises(ValueError, match="incomplete"):
        finalize(review, paths, "unused")
    assert not list(tmp_path.iterdir())


def test_freeze_is_durable_before_a_key_integrity_failure(tmp_path):
    text, blind, expected = example()
    text = text.replace("评阅者：待填写", "评阅者：测试评阅者").replace(
        "待填 | 待填", "不可判 | 无法核实"
    )
    review = validate_review(text, blind, expected)
    paths = {name: tmp_path / name for name in ("freeze", "result", "key", "rollout")}
    paths["key"].write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Git object"):
        finalize(review, paths, "wrong")
    assert paths["freeze"].exists()
    assert not paths["result"].exists()


def synthetic_review():
    ratings, mapping = {}, []
    for index in range(52):
        for seed in (169, 170, 171):
            for condition in ("matched", "mismatched_prompt"):
                identity = f"{index}-{seed}-{condition}"
                score = 2 if index < (16 if condition == "matched" else 8) else 0
                ratings[identity] = {
                    "score": score,
                    "case_id": str(index),
                    "not_evaluable": False,
                }
                mapping.append(
                    {
                        "response_id": identity,
                        "case_id": str(index),
                        "seed": seed,
                        "condition": condition,
                    }
                )
    rollout = {"runs": [{"seed": s, "rollout_gate_passed": False} for s in (169, 170, 171)]}
    return {"complete": True, "ratings": ratings}, mapping, rollout


def test_fixed_64_denominator_and_failed_R_survive_good_semantic_scores():
    review, mapping, rollout = synthetic_review()
    result = summarize(review, mapping, rollout, 64)
    assert result["status"] == "INCONCLUSIVE_ROLLOUT"
    assert not result["joint_passed"] and not result["can_promote"]
    assert all(row["S_passed"] for row in result["runs"])
    assert all(row["counts"]["matched"]["denominator"] == 64 for row in result["runs"])
    assert all(row["counts"]["matched"]["excluded_no_replacement"] == 12 for row in result["runs"])


def test_wrong_case_binding_is_rejected():
    review, mapping, rollout = synthetic_review()
    mapping = copy.deepcopy(mapping)
    mapping[0]["case_id"] = "wrong"
    with pytest.raises(ValueError, match="binding"):
        summarize(review, mapping, rollout, 64)
