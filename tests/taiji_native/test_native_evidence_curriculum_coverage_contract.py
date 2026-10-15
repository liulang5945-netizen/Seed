"""Target support is counted from labels, not from merely visible input bytes."""

from scripts.training.audit_taiji_n1_curriculum_coverage import coverage


def test_observed_character_is_not_positive_target_training_support():
    manifest = {
        "episodes": [
            {"id": "train-0", "split": "train", "answer": "aa", "observations": ["qz"]},
            {"id": "dev-0", "split": "dev", "answer": "qa"},
            {"id": "holdout-0", "split": "holdout", "answer": "z"},
        ]
    }
    result = coverage(manifest)
    assert result["train_positive_target_byte_counts"] == {"97": 2}
    assert result["by_split"]["dev"]["unseen_target_characters"] == "q"
    assert result["by_split"]["holdout"]["affected_ids"] == ["holdout-0"]
