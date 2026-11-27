"""The post-pilot diagnostic cannot silently become heldout fitting."""

import pytest

from scripts.training.audit_taiji_n1_native_evidence_train_fit import measure, training_items
from scripts.training.build_taiji_n1_evidence_curriculum import build


def test_diagnostic_selects_all_train_and_no_dev_or_holdout():
    selected = training_items(build())
    assert len(selected) == 96 and {item["split"] for item in selected} == {"train"}
    with pytest.raises(ValueError):
        training_items({"episodes": selected[:-1]})


def test_diagnostic_rejects_heldout_before_touching_model():
    for split in ("dev", "holdout"):
        with pytest.raises(ValueError, match="must not query"):
            measure(None, {"split": split}, "position", "never")
